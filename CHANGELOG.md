## Unreleased

* Pin llama.cpp at `b8635075f` (build `b8665`, 2026-04-04, PR #21418).
* Add support for the Gemma 4 architecture (`gemma4`) — text and vision
  variants. GGUFs that previously failed to load with the opaque error
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
