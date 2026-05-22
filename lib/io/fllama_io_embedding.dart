// FLLAMA-PATCH (Tacita D-2): Dart-side wrapper around the native
// `fllama_embed` entry point.
//
// Mirrors the isolate-marshalling pattern of
// `fllama_io_inference.dart`: a helper isolate owns the native call,
// the main isolate hands it an `_EmbedRequest`, and the result comes
// back as either a `Float32List` (copy of the native floats) or an
// error string.
//
// The native side fires the callback exactly once, with a pointer
// that is valid only for the duration of the call. We copy the
// floats inside the callback before signalling completion — never
// hold the borrowed pointer past callback return.

import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:fllama/fllama_io.dart';
import 'package:fllama/io/fllama_bindings_generated.dart';
import 'package:fllama/io/fllama_io_helpers.dart';

/// Pooling type for [FllamaEmbedRequest.poolingType]. Mirrors
/// llama.cpp's `enum llama_pooling_type`.
class FllamaPoolingType {
  const FllamaPoolingType._();

  /// Let the model's GGUF metadata declare the pooling type.
  /// This is the right choice for EmbeddingGemma (MEAN baked into
  /// metadata) and for any GGUF that ships with a pooling kv pair.
  static const int unspecified = -1;
  static const int none = 0;
  static const int mean = 1;
  static const int cls = 2;
  static const int last = 3;
}

/// Output of [fllamaEmbed]. Either [vector] is set (success) or
/// [error] is set (failure). Mutually exclusive.
class FllamaEmbedResult {
  const FllamaEmbedResult({this.vector, this.error});
  final Float32List? vector;
  final String? error;

  bool get isError => error != null;
}

/// Parameters for [fllamaEmbed].
class FllamaEmbedRequest {
  FllamaEmbedRequest({
    required this.input,
    required this.modelPath,
    this.numGpuLayers = 0,
    this.numThreads = 0,
    this.contextSize = 2048,
    this.poolingType = FllamaPoolingType.unspecified,
    this.embdNormalize = 2,
    this.logger,
  });

  /// Text to encode. Must be non-empty (the native side returns an
  /// error if tokenisation yields zero tokens).
  final String input;

  /// Absolute path to a .gguf embedding model.
  final String modelPath;

  /// 0 = CPU only, 99 = offload every layer to GPU/Metal.
  final int numGpuLayers;

  /// 0 = let llama.cpp pick from device cores (Tacita's
  /// RuntimePlanner.tune normally picks an explicit value).
  final int numThreads;

  /// Max tokens per embedding batch. Embedding models require
  /// `n_ubatch >= n_tokens`, so this also caps the longest text
  /// the native side can ingest in one call. 2048 is the native
  /// window of EmbeddingGemma 300M.
  final int contextSize;

  /// See [FllamaPoolingType]. Default `unspecified` keeps whatever
  /// the GGUF declares — preferred for embedding models authored
  /// with their pooling type baked in.
  final int poolingType;

  /// Normalisation applied by the native side BEFORE the result
  /// is handed to Dart. `2` is L2 normalisation, which makes the
  /// returned vector's dot product equal to cosine similarity.
  final int embdNormalize;

  /// Optional logger — receives metadata-only diagnostic strings
  /// from the native side. Never logs the input text or floats.
  final void Function(String)? logger;
}

class _IsolateEmbedRequest {
  _IsolateEmbedRequest(this.id, this.request);
  final int id;
  final FllamaEmbedRequest request;
}

class _IsolateEmbedResponse {
  _IsolateEmbedResponse(this.id, this.result);
  final int id;
  final FllamaEmbedResult result;
}

class _IsolateEmbedLogMessage {
  _IsolateEmbedLogMessage(this.id, this.message);
  final int id;
  final String message;
}

int _nextEmbedRequestId = 0;
final Map<int, Completer<FllamaEmbedResult>> _embedCompleters =
    <int, Completer<FllamaEmbedResult>>{};
final Map<int, void Function(String)?> _embedLoggers =
    <int, void Function(String)?>{};

Future<SendPort> _helperEmbedSendPort = () async {
  final completer = Completer<SendPort>();
  final receivePort = ReceivePort()
    ..listen((dynamic data) {
      if (data is SendPort) {
        completer.complete(data);
        return;
      }
      if (data is _IsolateEmbedLogMessage) {
        final logger = _embedLoggers[data.id];
        if (logger != null && data.message.trim().isNotEmpty) {
          logger(data.message);
        }
        return;
      }
      if (data is _IsolateEmbedResponse) {
        final c = _embedCompleters.remove(data.id);
        _embedLoggers.remove(data.id);
        if (c != null && !c.isCompleted) {
          c.complete(data.result);
        }
        return;
      }
    });

  await Isolate.spawn(_fllamaEmbedIsolate, receivePort.sendPort);
  return completer.future;
}();

/// Run a single embedding call. Returns the L2-normalised vector
/// the model produced, or a [FllamaEmbedResult] with [error] set on
/// failure.
///
/// Threading: the call runs on a dedicated helper isolate so it
/// doesn't block the host's main isolate even when CPU layers
/// dominate. Cancellation: not implemented yet (embedding calls
/// are short, typically < 100 ms for a 500-char input on a 6 GB
/// phone with `numGpuLayers = 99`).
Future<FllamaEmbedResult> fllamaEmbed(FllamaEmbedRequest request) async {
  final port = await _helperEmbedSendPort;
  final id = _nextEmbedRequestId++;
  final c = Completer<FllamaEmbedResult>();
  _embedCompleters[id] = c;
  if (request.logger != null) {
    _embedLoggers[id] = request.logger;
  }
  port.send(_IsolateEmbedRequest(id, request));
  return c.future;
}

Pointer<fllama_embed_request> _toNativeEmbedRequest(
  FllamaEmbedRequest dart,
  int requestId,
  Map<int, NativeCallable> loggerCallbacks,
  SendPort sendPort,
) {
  final p = calloc<fllama_embed_request>();
  final r = p.ref;
  r.request_id = requestId;
  r.input = dart.input.toNativeUtf8().cast<Char>();
  r.model_path = dart.modelPath.toNativeUtf8().cast<Char>();
  r.num_gpu_layers = dart.numGpuLayers;
  r.num_threads = dart.numThreads;
  r.context_size = dart.contextSize;
  r.pooling_type = dart.poolingType;
  r.embd_normalize = dart.embdNormalize;

  if (dart.logger != null) {
    void onLog(Pointer<Char> msg) {
      final s = pointerCharToString(msg);
      sendPort.send(_IsolateEmbedLogMessage(requestId, s));
    }

    final cb = NativeCallable<
        Void Function(Pointer<Char>)>.listener(onLog);
    r.dart_logger = cb.nativeFunction;
    loggerCallbacks[requestId] = cb;
  }
  return p;
}

void _fllamaEmbedIsolate(SendPort sendPort) {
  final helperReceivePort = ReceivePort();
  // Keep callbacks alive for the lifetime of the FFI call. The
  // map is per-isolate.
  final loggerCallbacks = <int, NativeCallable>{};

  helperReceivePort.listen((dynamic data) {
    if (data is! _IsolateEmbedRequest) return;
    try {
      final nativePtr =
          _toNativeEmbedRequest(data.request, data.id, loggerCallbacks, sendPort);
      final native = nativePtr.ref;

      late final NativeCallable<
          Void Function(Pointer<Float>, Int32, Pointer<Char>)> cb;

      void onEmbed(
        Pointer<Float> embedding,
        int nEmbd,
        Pointer<Char> errorMessage,
      ) {
        FllamaEmbedResult result;
        if (errorMessage != nullptr) {
          final err = pointerCharToString(errorMessage);
          result = FllamaEmbedResult(error: err);
        } else if (embedding == nullptr || nEmbd <= 0) {
          result = const FllamaEmbedResult(
            error: 'fllama_embed: empty result',
          );
        } else {
          // Copy floats while the borrowed pointer is still valid.
          final list = Float32List(nEmbd);
          for (var i = 0; i < nEmbd; i++) {
            list[i] = embedding[i];
          }
          result = FllamaEmbedResult(vector: list);
        }

        sendPort.send(_IsolateEmbedResponse(data.id, result));

        // Free native allocations.
        calloc.free(native.input);
        calloc.free(native.model_path);
        calloc.free(nativePtr);

        // Release the logger callback if any.
        final lc = loggerCallbacks.remove(data.id);
        lc?.close();

        // Release ourselves — `listener` callbacks must be closed.
        cb.close();
      }

      cb = NativeCallable<
          Void Function(
              Pointer<Float>, Int32, Pointer<Char>)>.listener(onEmbed);

      fllamaBindings.fllama_embed(native, cb.nativeFunction);
    } catch (e, s) {
      // ignore: avoid_print
      print('[fllama embed isolate] ERROR: $e. STACK: $s');
      sendPort.send(_IsolateEmbedResponse(
        data.id,
        FllamaEmbedResult(error: 'fllama_embed isolate threw: $e'),
      ),);
    }
  });

  sendPort.send(helperReceivePort.sendPort);
}
