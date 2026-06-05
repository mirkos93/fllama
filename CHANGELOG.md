## Unreleased

* Bump vendored llama.cpp from `b8635075f` (2026-04-04) to upstream master
  `2016bf2b3` (2026-06-05), 863 commits. Motivation: the new `GGML_TYPE_Q1_0`
  tensor type (enum value 41; `GGML_TYPE_COUNT` grows 41 → 42). GGUFs that use
  it — e.g. HauhauCS's Gemma-4-E2B quants — were rejected at load by the old
  pin because their tensor type equalled the old `GGML_TYPE_COUNT`. They now
  load. `LLM_ARCH_GEMMA4` (text + vision) remains present upstream, so the
  fork's reason-to-exist is preserved across the bump.
  * The vendored tree was pristine upstream `b8635075f` plus exactly ONE local
    patch (`vendor/cpp-httplib/httplib.h`, the iOS keychain-cert gate from
    `4d2174f`). That patch is now obsolete — upstream cpp-httplib gates the
    macOS-keychain define behind `#if TARGET_OS_OSX` and `#error`s on iOS, so
    it was NOT re-applied (would be a no-op / conflict). No other fork patches
    existed to carry forward.
  * API adaptation: `server_context::load_model()` now takes a *non-const*
    `common_params &`; `ServerManager::get_or_create()` receives params by
    const-ref, so it loads from a local mutable copy.
  * CMake (`src/CMakeLists.txt`): the common target is now `llama-common`
    (was `common`); the hand-rolled `server-context` library mirrors master's
    file list — adds `server-chat.cpp` + `server-tools.cpp`. The server
    refactor's Android-NDK hazard (`server-models.cpp` → `subprocess.h` /
    `posix_spawn_file_actions_*`) is avoided because master isolates those in
    the separate `llama-server-impl` target, which we do not build; the
    `server-context` target itself links only `llama-common` + `mtmd`, with no
    HTTP / subprocess dependency.
  * CMake build-info: the vendored tree has no `.git`, so upstream's
    `build-info.cmake` left `LLAMA_BUILD_NUMBER` empty (invalid
    `int LLAMA_BUILD_NUMBER = ;`). `src/CMakeLists.txt` now defines
    `LLAMA_BUILD_NUMBER` / `LLAMA_BUILD_COMMIT` explicitly from the fllama
    repo's own git before `add_subdirectory("llama.cpp")`.
* Previously: Add support for the Gemma 4 architecture (`gemma4`) — text and
  vision variants. GGUFs that previously failed to load with the opaque error
  "Failed to create inference context" now load.
* Route llama.cpp's internal logs to the per-request `dart_logger` via
  `llama_log_set`. Failures inside `load_model` (unknown arch, tensor
  mismatch, etc.) now surface in Dart instead of being silenced.
* `fllama_tokenize` no longer installs a global silencing log handler;
  it now silences logs only on the calling thread, leaving concurrent
  inference logging unaffected.
* CMake: add `cpp-httplib` as a subdirectory before `common` — starting
  with this llama.cpp drop, `common/CMakeLists.txt` unconditionally links
  the `cpp-httplib` target. We do not ship the HTTP server, but the link
  step still requires the target to exist.

Pin selection rationale: `b8635075f` is the first llama.cpp commit that
contains the Gemma 4 parser (PR #21418, merged 2026-04-04) and predates
the 2026-04-22 server refactor (PR #20690). That refactor renames the
CMake `common` target to `llama-common-base`, splits `server-chat.cpp`
out of `server-context.cpp`, and forces compilation of
`tools/server/server-models.cpp` — which transitively pulls in
`vendor/sheredom/subprocess.h` and `posix_spawn_file_actions_*`, which
the Android NDK does not provide for `armeabi-v7a` on API 26+.

## 0.0.1

* TODO: Describe initial release.
