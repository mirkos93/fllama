// fllama.cpp — Phase 3: thin adapter over llama.cpp's server_context
//
// fllama_inference() spawns a reader thread that posts a server_task and
// streams results back via the Dart callback.  Multiple concurrent calls
// are batched automatically by server_context::update_slots().

#include "fllama.h"
#include "fllama_inference_queue.h"
#include "fllama_log.h"
#include "fllama_mtmd.h"

// server-context headers (no HTTP / httplib dependency)
#include "server-context.h"
#include "server-task.h"
#include "server-common.h"

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include "ggml-backend.h"

// ── Logging ──────────────────────────────────────────────────────────────────

static void log_message(const char *msg,
                        fllama_log_callback logger = nullptr) {
  if (!logger) {
    fprintf(stderr, "%s\n", msg);
    fflush(stderr);
    return;
  }
  static std::mutex mtx;
  static std::deque<std::string> q;
  std::string s(msg);
  for (size_t p = 0; (p = s.find('\n', p)) != std::string::npos; p += 4)
    s.replace(p, 1, "[NL]");
  std::lock_guard<std::mutex> lk(mtx);
  q.push_back(std::move(s));
  while (q.size() > 1000)
    q.pop_front();
  logger(q.back().c_str());
}
static void log_message(const std::string &m,
                        fllama_log_callback l = nullptr) {
  log_message(m.c_str(), l);
}

// ── Globals ──────────────────────────────────────────────────────────────────

// Intentionally leaked — avoids static destruction order crash on exit.
// (ggml Metal statics may be destroyed before g_mgr's destructor runs,
//  causing ggml_abort when server_context tries to free Metal resources.)
static ServerManager &g_mgr = *new ServerManager();
static std::once_flag  g_backend_init;

static void fllama_backend_init_once() {
  std::call_once(g_backend_init, [] {
    // Install the global llama.cpp log handler before backend init so we
    // capture the very first GGML / Metal / CPU-feature messages.
    fllama_init_logging();
    ggml_backend_load_all();
    llama_backend_init();
  });
}

static std::vector<ggml_backend_dev_t> fllama_get_gpu_devices() {
  fllama_backend_init_once();

  std::vector<ggml_backend_dev_t> devices;
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    auto * dev = ggml_backend_dev_get(i);
    if (dev == nullptr) {
      continue;
    }
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
      continue;
    }
    devices.push_back(dev);
  }
  return devices;
}

static void fllama_copy_cstr(char * dst, size_t cap, const char * src) {
  if (dst == nullptr || cap == 0) {
    return;
  }
  std::snprintf(dst, cap, "%s", src ? src : "");
}

// FLLAMA-PATCH: map a string like "f16" / "q8_0" / "q4_0" to a ggml_type
// for KV-cache quantization. Mirrors `kv_cache_type_from_str` from
// llama.cpp/common/arg.cpp (which is `static` so we re-implement here).
// Returns GGML_TYPE_F16 on null / empty / unknown input — the safe
// upstream default. Allowed values match `--cache-type-k` upstream.
static ggml_type fllama_kv_cache_type_from_str(const char * s) {
  if (s == nullptr || s[0] == '\0') return GGML_TYPE_F16;
  std::string name(s);
  static const ggml_type kv_cache_types[] = {
      GGML_TYPE_F32,
      GGML_TYPE_F16,
      GGML_TYPE_BF16,
      GGML_TYPE_Q8_0,
      GGML_TYPE_Q4_0,
      GGML_TYPE_Q4_1,
      GGML_TYPE_IQ4_NL,
      GGML_TYPE_Q5_0,
      GGML_TYPE_Q5_1,
  };
  for (const auto & t : kv_cache_types) {
    if (name == ggml_type_name(t)) return t;
  }
  return GGML_TYPE_F16;
}

// ── The actual inference logic (runs on per-request thread) ──────────────────

static void run_inference(fllama_inference_request request,
                          fllama_inference_callback callback) {
  // Route llama.cpp's internal logs (GGML / model load / KV cache, etc.) to
  // this request's dart_logger for the duration of the call. Without this,
  // a load_model failure surfaces only as the opaque "Failed to create
  // inference context" with the real reason swallowed.
  fllama_set_thread_logger(request.dart_logger);
  struct LoggerGuard {
    ~LoggerGuard() { fllama_set_thread_logger(nullptr); }
  } _logger_guard;

  try {
    int64_t t0 = ggml_time_ms();
    log_message("[fllama] Inference start", request.dart_logger);

    // One-time backend init.
    fllama_backend_init_once();

    // ── 1. Build common_params ────────────────────────────────────────

    common_params params;
    params.model.path       = request.model_path;
    params.n_ctx            = request.context_size;
#if TARGET_OS_IPHONE
    // iPhone/iPad: smaller prompt-processing batches reduce Metal
    // compute-buffer spikes and leave the Flutter UI responsive under
    // memory pressure. This is a smoothness trade-off, not a throughput
    // win: long prompt ingestion can be slower, but the app remains usable.
    params.n_batch          = std::min<int32_t>(request.context_size, 1024);
    params.n_ubatch         = std::min<int32_t>(params.n_batch, 128);
#else
    // Match llama.cpp server defaults more closely instead of tying batch
    // sizes to the full context window.
    params.n_batch          = std::min<int32_t>(request.context_size, 2048);
    params.n_ubatch         = std::min<int32_t>(params.n_batch, 512);
#endif
    params.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_AUTO;
    params.n_parallel       = ServerManager::DEFAULT_N_PARALLEL;
    params.n_predict        = request.max_tokens;
    params.sampling.temp    = request.temperature;
    params.sampling.top_p   = request.top_p;
    params.sampling.penalty_freq   = request.penalty_freq;
    params.sampling.penalty_repeat = request.penalty_repeat;
    params.cpuparams.n_threads     = request.num_threads;
    params.use_jinja = true;
    params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    // Default is 8192 MiB — way too much for mobile/embedded.
    // 0 = disable host-memory prompt caching entirely.
    // The KV cache in the llama_context still handles prompt reuse;
    // this only controls the EXTRA host-RAM cache from PR #16391.
    params.cache_ram_mib = 0;

    // FLLAMA-PATCH: KV-cache quantization. F16 by default; Tacita's
    // RuntimePlanner overrides on RAM-tight devices to roughly double
    // effective context for ~3% perplexity. Both K and V can be set
    // independently so callers can mix (e.g. f16 K, q8_0 V).
    params.cache_type_k = fllama_kv_cache_type_from_str(request.kv_cache_type_k);
    params.cache_type_v = fllama_kv_cache_type_from_str(request.kv_cache_type_v);

#if TARGET_IPHONE_SIMULATOR
    params.n_gpu_layers = 0;
#else
    params.n_gpu_layers = request.num_gpu_layers;
#endif

    if (request.model_mmproj_path && strlen(request.model_mmproj_path) > 0)
      params.mmproj.path = request.model_mmproj_path;

    // ── 2. Get or create server_context ───────────────────────────────

    auto *srv = g_mgr.get_or_create(
        request.model_path, params, request.dart_logger);
    if (!srv || !srv->srv_ctx) {
      callback("Error: Failed to create inference context", "", true);
      return;
    }
    // RAII — release when we leave scope.
    struct Guard {
      ServerManager &m; std::string p;
      ~Guard() { m.release(p); }
    } guard{g_mgr, request.model_path};

    log_message("[fllama] Model ready (" +
                    std::to_string(ggml_time_ms() - t0) + " ms)",
                request.dart_logger);

    // ── 3. Build the prompt ───────────────────────────────────────────

    std::string prompt = request.input ? request.input : "";
    common_chat_parser_params parser_params;
    bool is_oai = false;

    if (request.openai_request_json_string) {
      is_oai = true;
      try {
        auto body = nlohmann::ordered_json::parse(
            request.openai_request_json_string);

        std::string jinja_tmpl;
        if (body.contains("jinja_template") &&
            body["jinja_template"].is_string()) {
          jinja_tmpl = body["jinja_template"].get<std::string>();
          body.erase("jinja_template");
        }

        auto *lctx  = srv->srv_ctx->get_llama_context();
        auto *model  = llama_get_model(lctx);
        auto  tmpls  = common_chat_templates_init(model, jinja_tmpl);

        try {
          std::map<std::string, std::string> empty;
          common_chat_format_example(tmpls.get(), true, empty);
        } catch (...) {
          tmpls = common_chat_templates_init(model, "chatml");
        }

        if (body.contains("messages") && body["messages"].is_array()) {
          // Flatten OpenAI-style multimodal content arrays back into the
          // single-string content the upstream parser accepts. Each
          // `image_url` part is appended to the running text as the
          // <img src="data:...;base64,..."> sentinel that
          // fllama_extract_images() picks up after templating.
          //
          // Refusing images without a configured mmproj here produces a
          // clear error rather than letting the model silently ignore the
          // pixels (the prompt would still contain an opaque marker the
          // model can't interpret).
          bool has_image_parts = false;
          auto & msgs_json = body["messages"];
          for (auto & message : msgs_json) {
            if (!message.is_object()) continue;
            if (!message.contains("content")) continue;
            auto & content = message["content"];
            if (!content.is_array()) continue;

            std::string flattened;
            for (const auto & part : content) {
              if (!part.is_object() || !part.contains("type")) continue;
              const std::string ptype = part.at("type").get<std::string>();
              if (ptype == "text" && part.contains("text") && part.at("text").is_string()) {
                if (!flattened.empty()) flattened += "\n";
                flattened += part.at("text").get<std::string>();
              } else if (ptype == "image_url" && part.contains("image_url")) {
                const auto & iu = part.at("image_url");
                if (!iu.is_object() || !iu.contains("url") || !iu.at("url").is_string()) {
                  continue;
                }
                const std::string url = iu.at("url").get<std::string>();
                // Only data: URLs are supported on-device. Reject http(s).
                if (url.rfind("data:", 0) != 0) {
                  log_message("[fllama] Skipping non-data: image_url (only "
                              "inline base64 supported)",
                              request.dart_logger);
                  continue;
                }
                if (!flattened.empty()) flattened += "\n";
                flattened += "<img src=\"";
                flattened += url;
                flattened += "\">";
                has_image_parts = true;
              }
              // media_marker / unknown types: ignored. The upstream parser
              // would have rejected them anyway.
            }
            content = flattened;
          }

          if (has_image_parts &&
              (!request.model_mmproj_path ||
               strlen(request.model_mmproj_path) == 0)) {
            callback("Error: image content provided but no mmproj path set "
                     "on the request — vision requires a multimodal projector "
                     "(.mmproj.gguf).",
                     "", true);
            g_mgr.clear_cancel(request.request_id);
            g_mgr.unregister_request_thread(request.request_id);
            return;
          }

          common_chat_templates_inputs inputs;
          inputs.use_jinja = true;
          inputs.add_generation_prompt = true;
          inputs.messages =
              common_chat_msgs_parse_oaicompat(body["messages"]);

          // Default to automatic reasoning extraction for modern reasoning/
          // channel-based templates (Qwen, GPT-OSS/Harmony, etc). Allow the
          // request body to override explicitly.
          inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
          inputs.enable_thinking = true;
          if (body.contains("reasoning_format") && body["reasoning_format"].is_string()) {
            inputs.reasoning_format = common_reasoning_format_from_name(
                body["reasoning_format"].get<std::string>());
          }
          if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean()) {
            inputs.enable_thinking = body["enable_thinking"].get<bool>();
          }
          inputs.chat_template_kwargs["enable_thinking"] =
              inputs.enable_thinking ? "true" : "false";
          if (body.contains("chat_template_kwargs") &&
              body["chat_template_kwargs"].is_object()) {
            for (const auto & item : body["chat_template_kwargs"].items()) {
              inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
            auto enable_thinking_kwarg =
                inputs.chat_template_kwargs.find("enable_thinking");
            if (enable_thinking_kwarg != inputs.chat_template_kwargs.end()) {
              if (enable_thinking_kwarg->second == "true") {
                inputs.enable_thinking = true;
              } else if (enable_thinking_kwarg->second == "false") {
                inputs.enable_thinking = false;
              }
            }
          }

          if (body.contains("tools")) {
            inputs.tools =
                common_chat_tools_parse_oaicompat(body["tools"]);
            inputs.tool_choice =
                body.contains("tool_choice")
                    ? common_chat_tool_choice_parse_oaicompat(
                          body["tool_choice"]
                              .template get<std::string>())
                    : COMMON_CHAT_TOOL_CHOICE_AUTO;
          }

          auto result =
              common_chat_templates_apply(tmpls.get(), inputs);
          prompt = result.prompt;
          if (!inputs.enable_thinking) {
            const std::string empty_gemma4_thought =
                "<|channel>thought\n<channel|>";
            size_t pos = std::string::npos;
            while ((pos = prompt.find(empty_gemma4_thought)) !=
                   std::string::npos) {
              prompt.erase(pos, empty_gemma4_thought.size());
            }
          }
          parser_params = common_chat_parser_params(result);
          parser_params.reasoning_format = inputs.reasoning_format;
          parser_params.reasoning_in_content =
              (inputs.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
          if (!result.parser.empty()) {
            parser_params.parser.load(result.parser);
          }

          log_message("[JPZ] fllama inputs.reasoning_format=" +
                          std::string(common_reasoning_format_name(inputs.reasoning_format)),
                      request.dart_logger);
          log_message("[fllama] Chat format: " +
                          std::string(common_chat_format_name(
                              result.format)),
                      request.dart_logger);
          log_message("[JPZ] PROMPT (" +
                          std::to_string(prompt.size()) + " chars):\n" +
                          prompt,
                      request.dart_logger);
        }
      } catch (const std::exception &e) {
        log_message(std::string("[fllama] OAI parse error: ") + e.what(),
                    request.dart_logger);
        is_oai = false;
      }
    }

    // ── 4. Multimodal — extract base64 → raw bytes ───────────────────

    std::vector<raw_buffer> files;
    if (fllama_prompt_contains_image(prompt)) {
      auto img = fllama_extract_images(prompt);
      prompt = std::move(img.text_with_markers);
      for (auto &fb : img.file_bytes)
        files.push_back(std::move(fb));
      log_message("[fllama] Extracted " +
                      std::to_string(files.size()) + " image(s)",
                  request.dart_logger);
    }

    // ── 5. Create & post the server task ──────────────────────────────

    auto reader = srv->srv_ctx->get_response_reader();

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id         = reader.get_new_id();
    task.index      = 0;
    task.cli        = true;
    task.cli_prompt = prompt;
    task.cli_files  = std::move(files);

    task.params.stream       = true;
    task.params.cache_prompt = true;
    task.params.n_predict    = request.max_tokens;
    task.params.sampling.temp           = request.temperature;
    task.params.sampling.top_p          = request.top_p;
    task.params.sampling.penalty_freq   = request.penalty_freq;
    task.params.sampling.penalty_repeat = request.penalty_repeat;

    // FLLAMA-PATCH (Phase 1 — roleplay samplers). Pull the extended
    // sampler surface out of the OAI JSON body. Every key mirrors
    // llama.cpp's `common_params_sampling` field name so the same JSON
    // shape works against an upstream llama-server. Missing keys leave
    // the sampler on the upstream default. The `min_p` forwarding here
    // closes the silently-dropped bug — the chat path was previously
    // pinned to llama.cpp's hardcoded 0.05 regardless of host intent.
    if (is_oai && request.openai_request_json_string) {
      try {
        auto body = nlohmann::ordered_json::parse(
            request.openai_request_json_string);
        if (body.contains("min_p") && body["min_p"].is_number()) {
          task.params.sampling.min_p = body["min_p"].get<float>();
        }
        if (body.contains("top_k") && body["top_k"].is_number()) {
          task.params.sampling.top_k = body["top_k"].get<int32_t>();
        }
        if (body.contains("penalty_last_n") &&
            body["penalty_last_n"].is_number()) {
          task.params.sampling.penalty_last_n =
              body["penalty_last_n"].get<int32_t>();
        }
        if (body.contains("mirostat") && body["mirostat"].is_number()) {
          task.params.sampling.mirostat = body["mirostat"].get<int32_t>();
        }
        if (body.contains("mirostat_tau") &&
            body["mirostat_tau"].is_number()) {
          task.params.sampling.mirostat_tau =
              body["mirostat_tau"].get<float>();
        }
        if (body.contains("mirostat_eta") &&
            body["mirostat_eta"].is_number()) {
          task.params.sampling.mirostat_eta =
              body["mirostat_eta"].get<float>();
        }
        if (body.contains("dynatemp_range") &&
            body["dynatemp_range"].is_number()) {
          task.params.sampling.dynatemp_range =
              body["dynatemp_range"].get<float>();
        }
        if (body.contains("dynatemp_exponent") &&
            body["dynatemp_exponent"].is_number()) {
          task.params.sampling.dynatemp_exponent =
              body["dynatemp_exponent"].get<float>();
        }
        if (body.contains("top_n_sigma") &&
            body["top_n_sigma"].is_number()) {
          task.params.sampling.top_n_sigma =
              body["top_n_sigma"].get<float>();
        }
        if (body.contains("min_keep") && body["min_keep"].is_number()) {
          task.params.sampling.min_keep = body["min_keep"].get<int32_t>();
        }
        if (body.contains("dry_multiplier") &&
            body["dry_multiplier"].is_number()) {
          task.params.sampling.dry_multiplier =
              body["dry_multiplier"].get<float>();
        }
        if (body.contains("dry_base") && body["dry_base"].is_number()) {
          task.params.sampling.dry_base = body["dry_base"].get<float>();
        }
        if (body.contains("dry_allowed_length") &&
            body["dry_allowed_length"].is_number()) {
          task.params.sampling.dry_allowed_length =
              body["dry_allowed_length"].get<int32_t>();
        }
        if (body.contains("dry_penalty_last_n") &&
            body["dry_penalty_last_n"].is_number()) {
          task.params.sampling.dry_penalty_last_n =
              body["dry_penalty_last_n"].get<int32_t>();
        }
        if (body.contains("xtc_probability") &&
            body["xtc_probability"].is_number()) {
          task.params.sampling.xtc_probability =
              body["xtc_probability"].get<float>();
        }
        if (body.contains("xtc_threshold") &&
            body["xtc_threshold"].is_number()) {
          task.params.sampling.xtc_threshold =
              body["xtc_threshold"].get<float>();
        }
        // logit_bias: array of [token_id, bias] pairs. Each pair
        // becomes a `llama_logit_bias` entry. Upstream llama-server
        // also accepts strings (resolved to token ids), but the
        // host-side mapping is more reliable than parsing here.
        if (body.contains("logit_bias") && body["logit_bias"].is_array()) {
          for (const auto & entry : body["logit_bias"]) {
            if (!entry.is_array() || entry.size() != 2) continue;
            if (!entry[0].is_number()) continue;
            if (!entry[1].is_number()) continue;
            llama_logit_bias lb;
            lb.token = entry[0].get<llama_token>();
            lb.bias  = entry[1].get<float>();
            task.params.sampling.logit_bias.push_back(lb);
          }
        }
      } catch (const std::exception & e) {
        log_message(std::string("[fllama] sampler parse error: ") + e.what(),
                    request.dart_logger);
      }
    }

    // Seed: prefer host-supplied value (for reproducibility tests),
    // otherwise pin a fresh one per task so successive generations don't
    // collide on the same RNG state.
    bool host_seed_set = false;
    if (is_oai && request.openai_request_json_string) {
      try {
        auto body = nlohmann::ordered_json::parse(
            request.openai_request_json_string);
        if (body.contains("seed") && body["seed"].is_number_integer()) {
          task.params.sampling.seed =
              body["seed"].get<uint32_t>();
          host_seed_set = true;
        }
      } catch (...) {
        // Already logged above; fall through to random.
      }
    }
    if (!host_seed_set) {
      std::random_device rd;
      task.params.sampling.seed = rd();
    }

    if (is_oai) {
      task.params.res_type           = TASK_RESPONSE_TYPE_OAI_CHAT;
      task.params.oaicompat_model    = request.model_path;
      task.params.oaicompat_cmpl_id  = gen_chatcmplid();
      task.params.chat_parser_params = parser_params;
    }

    reader.post_task(std::move(task));

    // ── 6. Read results, invoke callbacks ─────────────────────────────

    int rid = request.request_id;
    auto should_stop = [&] { return g_mgr.is_cancelled(rid); };

    std::string full_content;
    std::string last_json;

    while (reader.has_next()) {
      server_task_result_ptr res;
      try {
        res = reader.next(should_stop);
      } catch (const std::exception &e) {
        // Final parse can fail (e.g. doubled generated_text in update_chat_msg).
        // Log and break — we still have the accumulated text + last good JSON.
        log_message(std::string("[JPZ] reader.next() threw: ") + e.what(),
                    request.dart_logger);
        break;
      }
      if (!res) break;

      if (res->is_error()) {
        auto ej = res->to_json();
        std::string msg = ej.contains("message")
                              ? ej["message"].get<std::string>()
                              : ej.dump();
        callback(msg.c_str(), "", true);
        g_mgr.clear_cancel(rid);
        g_mgr.unregister_request_thread(rid);
        return;
      }

      auto *partial =
          dynamic_cast<server_task_result_cmpl_partial *>(res.get());
      if (partial) {
        full_content += partial->content;
        log_message("[fllama] token: \"" + partial->content +
                    "\"  cumulative(" + std::to_string(full_content.size()) +
                    " chars)",
                    request.dart_logger);
        auto j = res->to_json();
        if (!j.is_null()) {
          last_json = j.dump();
          callback(full_content.c_str(), last_json.c_str(), false);
        }
        continue;
      }

      auto *final_r =
          dynamic_cast<server_task_result_cmpl_final *>(res.get());
      if (final_r) {
        // Keep accumulated full_content — final_r->content can be
        // empty or corrupted for tool-call / reasoning completions.
        try {
          auto j = res->to_json();
          last_json = j.is_null() ? "" : j.dump();
          log_message("[JPZ] final to_json() is_null=" +
                          std::to_string(j.is_null()) +
                          " type=" + std::to_string((int)j.type()) +
                          " size=" + std::to_string(j.size()) +
                          " dump=" + last_json.substr(0, 200),
                      request.dart_logger);
        } catch (const std::exception &e) {
          log_message(std::string("[JPZ] final to_json() THREW: ") + e.what(),
                      request.dart_logger);
          last_json = "";
        }
        log_message("[JPZ] final_r->content(" +
                        std::to_string(final_r->content.size()) +
                        ")=\"" + final_r->content.substr(0, 100) + "\"",
                    request.dart_logger);
        callback(full_content.c_str(), last_json.c_str(), true);

        log_message("[fllama] Done. " +
                        std::to_string(final_r->n_decoded) + " tok, " +
                        std::to_string(ggml_time_ms() - t0) + " ms",
                    request.dart_logger);
        g_mgr.clear_cancel(rid);
        g_mgr.unregister_request_thread(rid);
        return;
      }
    }

    // Cancelled or exhausted without final result.
    callback(full_content.c_str(), last_json.c_str(), true);
    g_mgr.clear_cancel(rid);
    g_mgr.unregister_request_thread(rid);

  } catch (const std::exception &e) {
    std::string msg = "Error: " + std::string(e.what());
    callback(msg.c_str(), "", true);
    g_mgr.clear_cancel(request.request_id);
    g_mgr.unregister_request_thread(request.request_id);
  } catch (...) {
    callback("Error: Unknown exception", "", true);
    g_mgr.clear_cancel(request.request_id);
    g_mgr.unregister_request_thread(request.request_id);
  }
}

// ── FFI entry points ─────────────────────────────────────────────────────────

extern "C" {

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_device_count(void) {
  return static_cast<int>(fllama_get_gpu_devices().size());
}

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_memory_info(
    int gpu_index,
    struct fllama_gpu_memory_info * out_info) {
  if (out_info == nullptr) {
    return 1;
  }

  std::memset(out_info, 0, sizeof(*out_info));

  if (gpu_index < 0) {
    return 2;
  }

  auto devices = fllama_get_gpu_devices();
  if (static_cast<size_t>(gpu_index) >= devices.size()) {
    return 3;
  }

  auto * dev = devices[static_cast<size_t>(gpu_index)];
  ggml_backend_dev_props props{};
  ggml_backend_dev_get_props(dev, &props);

  size_t total = props.memory_total;
  size_t free = props.memory_free;

  // Metal reports free as recommendedMaxWorkingSetSize - currentAllocatedSize.
  // If currentAllocatedSize exceeds the recommendation, the backend can
  // underflow the unsigned subtraction. Clamp that to zero here.
  if (total == 0) {
    free = 0;
  } else if (free > total) {
    free = 0;
  }

  out_info->device_index = gpu_index;
  out_info->total_bytes = static_cast<uint64_t>(total);
  out_info->free_bytes = static_cast<uint64_t>(free);
  fllama_copy_cstr(out_info->name, sizeof(out_info->name), props.name);
  fllama_copy_cstr(
      out_info->description,
      sizeof(out_info->description),
      props.description);
  fllama_copy_cstr(
      out_info->device_id,
      sizeof(out_info->device_id),
      props.device_id);
  return 0;
}

EMSCRIPTEN_KEEPALIVE void fllama_inference(fllama_inference_request request,
                                           fllama_inference_callback callback) {
  int rid = request.request_id;
  std::thread t([request, callback] { run_inference(request, callback); });
  g_mgr.register_request_thread(rid, std::move(t));
}

EMSCRIPTEN_KEEPALIVE void
fllama_inference_sync(fllama_inference_request request,
                      fllama_inference_callback callback) {
  // Synchronous variant — blocks the calling thread.
  run_inference(request, callback);
}

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void
fllama_inference_cancel(int request_id) {
  g_mgr.cancel(request_id);
}

} // extern "C"
