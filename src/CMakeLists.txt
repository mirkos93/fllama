# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(fllama_library VERSION 0.0.1 LANGUAGES CXX)

# Build-info values.
#
# The vendored llama.cpp tree is a plain source copy with NO .git of its own,
# so llama.cpp's cmake/build-info.cmake (run from inside add_subdirectory below)
# cannot derive a build number / commit via `git rev-list` — it leaves
# LLAMA_BUILD_NUMBER empty, which makes common/build-info.cpp.in emit the
# invalid line `int LLAMA_BUILD_NUMBER = ;`. So we make sure both names are
# DEFINED before add_subdirectory("llama.cpp"), which turns the subdir's
# `if(NOT DEFINED LLAMA_BUILD_NUMBER)` guards into no-ops.
#
# Priority:
#   1. Values passed in via -D (hook/build.dart sets these to the vendored
#      upstream pin) win — we do not override them.
#   2. Otherwise derive from the fllama repo's own git (standalone CMake build).
#   3. Otherwise fall back to placeholders.
if(NOT DEFINED LLAMA_BUILD_NUMBER OR "${LLAMA_BUILD_NUMBER}" STREQUAL "")
  find_package(Git QUIET)
  set(_FLLAMA_BUILD_NUMBER "0")
  set(_FLLAMA_BUILD_COMMIT "unknown")
  if(Git_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE _GIT_COUNT OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _GIT_RES_COUNT ERROR_QUIET)
    if(_GIT_RES_COUNT EQUAL 0 AND NOT "${_GIT_COUNT}" STREQUAL "")
      set(_FLLAMA_BUILD_NUMBER "${_GIT_COUNT}")
    endif()
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE _GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _GIT_RES_HASH ERROR_QUIET)
    if(_GIT_RES_HASH EQUAL 0 AND NOT "${_GIT_HASH}" STREQUAL "")
      set(_FLLAMA_BUILD_COMMIT "${_GIT_HASH}")
    endif()
  endif()
  set(LLAMA_BUILD_NUMBER "${_FLLAMA_BUILD_NUMBER}")
  set(LLAMA_BUILD_COMMIT "${_FLLAMA_BUILD_COMMIT}")
endif()
if(NOT DEFINED LLAMA_BUILD_COMMIT OR "${LLAMA_BUILD_COMMIT}" STREQUAL "")
  set(LLAMA_BUILD_COMMIT "unknown")
endif()
message(STATUS "fllama build-info: number=${LLAMA_BUILD_NUMBER} commit=${LLAMA_BUILD_COMMIT}")


set(BUILD_SHARED_LIBS OFF)

# Otherwise ex. Android build on macOS fails with `error: unknown target CPU 'cyclone'`
set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)

# Disable HTTP/CURL support on all platforms (we handle model downloads in Dart, not C++)
set(LLAMA_HTTPLIB OFF CACHE BOOL "Disable httplib for downloading" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "Disable CURL (deprecated)" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server (requires httplib)" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "Disable OpenSSL/HTTPS in cpp-httplib" FORCE)
set(LLAMA_BUILD_BORINGSSL OFF CACHE BOOL "Disable BoringSSL in cpp-httplib" FORCE)
set(LLAMA_BUILD_LIBRESSL OFF CACHE BOOL "Disable LibreSSL in cpp-httplib" FORCE)

if(ANDROID)
    # Debug: Print build information for Android
    message(STATUS "Android build detected")
    message(STATUS "CMAKE_ANDROID_ARCH_ABI: ${CMAKE_ANDROID_ARCH_ABI}")
    message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
    
    # Android-specific settings for llama.cpp (apply to all Android builds)
    set(GGML_LLAMAFILE OFF CACHE BOOL "Disable llamafile for Android" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP for Android" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Use static libs for Android" FORCE)
    
    # Only apply ARM-specific optimizations for ARM64 builds
    if(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        message(STATUS "ARM64 Android build - enabling dot product optimizations with KleiDiAI support")
        # ARMv8.2-A with dot product only (i8mm not supported on this device)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod -O3")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod -O3")
        
        # Enable KleiDiAI ONLY for ARM64 builds (it's ARM64-specific)
        # Disabled: it requires CMake 4.0+ and src/CMakeLists.txt (via flutter template)
        # strongly suggests not going over 3.22, i.e. what is required for Flutter.
        # Because KleidiAI showed a small lift on performance (well within variation, 3-5%) 
        # for Android CPU on Pixel Fold, it is better to disable it for now. It is also
        # worth leaving here as ersatz documentation of how to get it working.
        # set(GGML_CPU_KLEIDIAI ON CACHE BOOL "Enable KleiDiAI optimizations" FORCE)
    else()
        message(STATUS "Non-ARM64 Android build (${CMAKE_ANDROID_ARCH_ABI}) - skipping ARM optimizations")
        # Explicitly disable KleiDiAI for non-ARM64 architectures
        # set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "Disable KleiDiAI for non-ARM64" FORCE)
    endif()
endif()

if(WIN32)
  set(LLAMA_VULKAN ON CACHE BOOL "llama: enable Vulkan" FORCE)
  message(STATUS "Windows detected, enabling LLAMA_VULKAN")
endif()

if(APPLE)
  # Metal GPU acceleration — previously set by podspec/shell scripts.
  # Now handled here so the CMake build is self-contained.
  set(GGML_METAL ON CACHE BOOL "Enable Metal" FORCE)
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed Metal shader library" FORCE)
endif()

# Provide a fallback version so mtmd's set_target_properties(VERSION ...)
# doesn't fail when there is no git history (e.g. copied source tree).
if(NOT DEFINED LLAMA_INSTALL_VERSION OR LLAMA_INSTALL_VERSION STREQUAL "")
  set(LLAMA_INSTALL_VERSION "0.0.0")
endif()

add_subdirectory("llama.cpp")
# cpp-httplib must be added before common: common/CMakeLists.txt PRIVATE-links
# the cpp-httplib target (for the model-download helper). We don't ship the
# HTTP server, but common still references httplib symbols at link time, so
# the target must be defined.
add_subdirectory("llama.cpp/vendor/cpp-httplib")
add_subdirectory("llama.cpp/common")
add_subdirectory("llama.cpp/tools/mtmd")

# Build the server-context library directly from source files.
# We cannot use add_subdirectory(tools/server) because it also defines the
# llama-server executable + llama-server-impl, which pull in cpp-httplib's
# HTTP server (server-http.cpp) and server-models.cpp's subprocess.h
# (unavailable on the Android NDK for armeabi-v7a). We hand-pick the core
# files master's `server-context` target is built from, MINUS server-tools.cpp.
#
# server-tools.cpp is part of master's server-context target but `#include`s
# <sheredom/subprocess.h> (subprocess_create → posix_spawn_file_actions_*),
# which the Android NDK does NOT provide for armeabi-v7a on API 26+ — the exact
# hazard that drove the original pin choice. It implements the OPTIONAL builtin
# "server tools" (web-search / code-exec subprocess runners) used only by the
# HTTP server layer; nothing in server_context or fllama.cpp references it (no
# server_tools member, no calls). So we exclude it. The remaining 5 files form
# a clean STATIC lib linking only `llama-common` + `mtmd`, with no HTTP /
# subprocess dependency.
find_package(Threads REQUIRED)
set(SERVER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/server")
add_library(server-context STATIC
  ${SERVER_SRC_DIR}/server-chat.cpp
  ${SERVER_SRC_DIR}/server-task.cpp
  ${SERVER_SRC_DIR}/server-queue.cpp
  ${SERVER_SRC_DIR}/server-common.cpp
  ${SERVER_SRC_DIR}/server-context.cpp
)
target_include_directories(server-context PUBLIC  ${SERVER_SRC_DIR})
target_include_directories(server-context PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd
  ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp
)
if (BUILD_SHARED_LIBS)
  set_target_properties(server-context PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_link_libraries(server-context PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

add_library(fllama SHARED
  "fllama_chat_template.cpp"
  "fllama_download_stub.cpp"
  "fllama_eos.cpp"
  "fllama_inference_queue.cpp"
  "fllama_log.cpp"
  "fllama_mtmd.cpp"
  "fllama_tokenize.cpp"
  "fllama.cpp"
)

set_target_properties(fllama PROPERTIES
  PUBLIC_HEADER "fllama.h;fllama_eos.h;fllama_tokenize.h"
  OUTPUT_NAME "fllama"
)

target_compile_definitions(fllama PUBLIC DART_SHARED_LIB)

if (ANDROID)
  # Support Android 15 16k page size.
  target_link_options(fllama PRIVATE "-Wl,-z,max-page-size=16384")
endif()
target_include_directories(fllama PUBLIC .)
target_link_libraries(fllama PUBLIC llama llama-common mtmd server-context)

# Apple: link system frameworks for Metal, Accelerate, etc.
# These are needed because we statically link ggml-metal into this shared lib.
if(APPLE)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(METALKIT_LIBRARY MetalKit REQUIRED)
  find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
  target_link_libraries(fllama PUBLIC
    ${FOUNDATION_LIBRARY}
    ${METAL_LIBRARY}
    ${METALKIT_LIBRARY}
    ${ACCELERATE_LIBRARY}
  )
endif()

if(ANDROID)
    find_library(LOG_LIB log) # Find the log library
    target_link_libraries(fllama PUBLIC
      ${LOG_LIB} # Add this to link against the log library for Android
    )
endif()
