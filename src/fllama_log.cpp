#include "fllama_log.h"

#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace {

thread_local fllama_log_callback tls_logger = nullptr;
thread_local bool tls_silence = false;

std::once_flag g_log_init;

void fllama_global_log_handler(ggml_log_level /*level*/,
                               const char *text,
                               void * /*user_data*/) {
  if (tls_silence) return;
  if (!text || !*text) return;

  size_t len = std::strlen(text);
  while (len > 0 && (text[len - 1] == '\n' || text[len - 1] == '\r')) {
    --len;
  }
  if (len == 0) return;

  std::string msg(text, len);
  if (tls_logger) {
    tls_logger(msg.c_str());
  } else {
    std::fputs(msg.c_str(), stderr);
    std::fputc('\n', stderr);
  }
}

} // namespace

extern "C" {

void fllama_init_logging(void) {
  std::call_once(g_log_init, [] {
    llama_log_set(fllama_global_log_handler, nullptr);
  });
}

void fllama_set_thread_logger(fllama_log_callback logger) {
  tls_logger = logger;
}

void fllama_set_thread_log_silence(int silenced) {
  tls_silence = silenced != 0;
}

} // extern "C"
