#ifndef FLLAMA_LOG_H
#define FLLAMA_LOG_H

#include "fllama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Install the global llama.cpp log handler. Idempotent.
void fllama_init_logging(void);

// Route llama.cpp logs from this thread to [logger]. Pass NULL to fall back
// to stderr.
void fllama_set_thread_logger(fllama_log_callback logger);

// Drop llama.cpp logs on this thread when [silenced] is non-zero.
// Used by fllama_tokenize to suppress the model-load log burst.
void fllama_set_thread_log_silence(int silenced);

#ifdef __cplusplus
}
#endif

#endif
