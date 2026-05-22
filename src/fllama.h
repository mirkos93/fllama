#ifndef FLLAMA_H
#define FLLAMA_H

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#if _WIN32
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FFI_PLUGIN_EXPORT
#endif

#include <stdint.h> // For uint8_t

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*fllama_inference_callback)(const char *response, const char * openai_response_json_string, uint8_t done);
typedef void (*fllama_log_callback)(const char *);

// FLLAMA-PATCH (Tacita T-RECALL-EMBED-ENGINE / D-2): embedding callback.
// `embedding_floats` is a borrowed pointer valid only for the duration
// of the call; the Dart side must copy the bytes before returning.
// `n_embd` is the per-token embedding dimensionality (e.g. 768 for
// EmbeddingGemma 300M) and equals the length of [embedding_floats].
// `error_message` is NULL on success and a NUL-terminated C string on
// failure (in which case [embedding_floats] is NULL and [n_embd] is 0).
typedef void (*fllama_embed_callback)(const float * embedding_floats,
                                      int32_t n_embd,
                                      const char * error_message);

struct fllama_gpu_memory_info {
  int32_t device_index;
  uint64_t total_bytes;
  uint64_t free_bytes;
  char name[128];
  char description[256];
  char device_id[128];
};

struct fllama_inference_request {
  int request_id; // Required: unique ID for the request. Used for cancellation.
  int context_size;        // Required: context size
  char *input;             // Required: input text
  int max_tokens;          // Required: max tokens to generate
  char *model_path;        // Required: .ggml model file path
  char *model_mmproj_path; // Optional: .mmproj file for multimodal models.
  int num_gpu_layers; // Required: number of GPU layers. 0 for CPU only. 99 for
                      // all layers. Automatically 0 on iOS simulator.
  int num_threads; // Required: 2 recommended. Platforms can be highly sensitive
                   // to this, ex. Android stopped working with 4 suddenly.
  float
      temperature; // Optional: temperature. Defaults to 0. (llama.cpp behavior)
  float top_p; // Optional: 0 < top_p <= 1. Defaults to 1. (llama.cpp behavior)
  float penalty_freq;   // Optional: 0 <= penalty_freq <= 1. Defaults to 0.0,
                        // which means disabled. (llama.cpp behavior)
  float penalty_repeat; // Optional: 0 <= penalty_repeat <= 1. Defaults to 1.0,
                        // which means disabled. (llama.cpp behavior)
  char *
      grammar; // Optional: BNF-like grammar to constrain sampling. Defaults to
               // "" (llama.cpp behavior). See
               // https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
  char *eos_token; // Optional: end of sequence token. Defaults to one in model file. (llama.cpp behavior)
                   // For example, in ChatML / OpenAI, <|im_end|> means the message is complete.
                   // Often times GGUF files were created incorrectly, and this should be overridden.
                   // Using fllamaChat from Dart handles this automatically.
  fllama_log_callback
      dart_logger; // Optional: Dart caller logger. Defaults to NULL.
  char * openai_request_json_string; // Optional: OpenAI JSON string. Defaults to NULL.
  // FLLAMA-PATCH: KV-cache quantization knobs.
  // Optional ggml type name strings (e.g. "f16", "q8_0", "q4_0").
  // NULL or empty means "use llama.cpp default" (f16). Tacita's
  // RuntimePlanner picks `q8_0` on RAM-tight devices to roughly double
  // effective context for ~3% perplexity. Allowed values match the
  // upstream `--cache-type-k` flag: f32, f16, bf16, q8_0, q4_0, q4_1,
  // iq4_nl, q5_0, q5_1.
  char * kv_cache_type_k;
  char * kv_cache_type_v;
};

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_inference(struct fllama_inference_request request,
                                        fllama_inference_callback callback);
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_inference_sync(struct fllama_inference_request request,
                           fllama_inference_callback callback);
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_inference_cancel(int request_id);

// ── Embedding API (FLLAMA-PATCH, Tacita D-2) ──────────────────────────────
//
// Encodes [input] with the model at [model_path] and invokes [callback]
// once with the L2-normalised, sequence-pooled embedding vector. The
// model must have a pooling type set (LAST / MEAN / CLS); models built
// for generation only (`pooling_type == LLAMA_POOLING_TYPE_NONE`) will
// fail with an error.
//
// `pooling_type`:
//   -1 = LLAMA_POOLING_TYPE_UNSPECIFIED (let the GGUF declare it; this
//        is the right choice for EmbeddingGemma, whose metadata pins
//        MEAN pooling, and for any embedding-trained GGUF).
//    0 = NONE (per-token output, NOT supported by this entry point).
//    1 = MEAN, 2 = CLS, 3 = LAST. See `enum llama_pooling_type`.
//
// `embd_normalize` matches llama-server's `--embd-normalize`:
//   -1 = none, 0 = max-absolute int16 scale, 1 = taxicab/L1,
//    2 = euclidean/L2 (recommended for cosine similarity),
//   >2 = p-norm.
//
// Context lifetime: a dedicated `server_context` is created per model
// path and cached for [MODEL_INACTIVITY_TIMEOUT_SEC] after the last
// embedding call (same cache machinery as the chat path; eviction is
// independent because the `n_parallel` / pooling-type params differ
// from a generative load and `params_match()` rejects reuse).
struct fllama_embed_request {
  int request_id;        // Required: unique ID for the request.
  char * input;          // Required: NUL-terminated text to embed.
  char * model_path;     // Required: path to a .gguf embedding model.
  int num_gpu_layers;    // 0 for CPU only, 99 for all layers.
  int num_threads;       // 0 = let llama.cpp pick from device cores.
  int context_size;      // Required: max tokens per embedding batch.
                         // Tacita uses 2048 for EmbeddingGemma (the
                         // model's native max is 2048).
  int32_t pooling_type;  // See enum above (-1 = unspecified/default).
  int32_t embd_normalize;// See enum above (2 = L2 normalise).
  fllama_log_callback dart_logger; // Optional.
};

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_embed(struct fllama_embed_request request,
                                                        fllama_embed_callback callback);
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_embed_sync(struct fllama_embed_request request,
                                                              fllama_embed_callback callback);
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_embed_cancel(int request_id);

// GPU device information.
// Returns the number of GPU devices visible to ggml/llama.cpp.
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_device_count(void);

// Fills [out_info] for the GPU at [gpu_index].
// Returns 0 on success, non-zero on failure.
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_memory_info(
    int gpu_index,
    struct fllama_gpu_memory_info * out_info);
#ifdef __cplusplus
}
#endif

#endif // FLLAMA_H