import 'dart:convert';
import 'dart:typed_data';

import 'package:fllama/misc/openai_tool.dart';

enum Role {
  assistant,
  system,
  tool,
  user;

  String get openAiName {
    switch (this) {
      case Role.assistant:
        return 'assistant';
      case Role.system:
        return 'system';
      case Role.user:
        return 'user';
      case Role.tool:
        return 'tool';
    }
  }
}

/// One image attached to a [Message].
///
/// The bytes are the raw, undecoded contents of an image file (PNG, JPEG,
/// WebP, …). The native side decodes them via stb_image / mtmd, so any format
/// supported by llama.cpp's mtmd helper is accepted.
///
/// [mimeType] is forwarded as the `data:<mime>;base64,...` prefix and is
/// purely informational: the native decoder sniffs the magic bytes regardless.
/// When omitted, defaults to `image/png`.
class MessageImage {
  final Uint8List bytes;
  final String mimeType;

  const MessageImage(this.bytes, {this.mimeType = 'image/png'});
}

class Message {
  final Role role;
  final String text;

  /// Optional name of tool in the message.
  final String? toolResponseName;
  final List<Map<String, dynamic>>? toolCalls;

  /// Optional images attached to this message. When non-empty, the native
  /// runtime feeds them to the multimodal projector (`mmprojPath` on the
  /// owning [OpenAiRequest]). The model must be a vision-capable LLM and an
  /// `mmprojPath` must be set, otherwise inference will fail with a clear
  /// error.
  ///
  /// Order is preserved across the wire and follows OpenAI's `image_url`
  /// content-part shape on the JSON the native side receives.
  final List<MessageImage>? images;

  Message(
    this.role,
    this.text, {
    this.toolCalls,
    this.toolResponseName,
    this.images,
  });
}

/// Corresponds to COMMON_CHAT_TOOL_CHOICE_* in llama.cpp.
///
/// In OpenAI's API, required must specify a tool name and guarantees a
/// response with that tool.
///
/// To replicate that behavior, pass required with a single tool in [tools].
enum ToolChoice {
  auto,
  none,
  required;

  String get jsonName {
    switch (this) {
      case ToolChoice.auto:
        return 'auto';
      case ToolChoice.none:
        return 'none';
      case ToolChoice.required:
        return 'required';
    }
  }
}

class OpenAiRequest {
  final List<Message> messages;
  final List<Tool> tools;
  final double temperature;
  final int maxTokens;
  final double topP;
  final double frequencyPenalty;
  final double presencePenalty;
  // Not in OpenAI, but used by llama.
  final String modelPath;
  final String? mmprojPath;
  final int numGpuLayers;
  final int contextSize;
  final String? jinjaTemplate;
  final Function(String)? logger;
  final ToolChoice? toolChoice;
  // Forwarded to the Jinja chat template as `enable_thinking`. Models
  // with reasoning channels (Gemma 4, Qwen-3, etc.) emit a thought
  // block when this is true. Set false to skip reasoning and stream
  // the answer directly. Default true preserves prior behaviour.
  final bool enableThinking;

  /// FLLAMA-PATCH: number of CPU threads. `0` (default) lets llama.cpp
  /// pick from device cores. Hosts that know their device tier should
  /// override (Tacita's `RuntimePlanner.tune()` sets this per chip
  /// class to avoid leaving the 4–6 big cores on Tensor G3 idle).
  final int numThreads;

  /// FLLAMA-PATCH: optional KV-cache quantization for the K matrix.
  /// One of `f32`, `f16` (default), `bf16`, `q8_0`, `q4_0`, `q4_1`,
  /// `iq4_nl`, `q5_0`, `q5_1`. Null leaves llama.cpp on `f16`.
  final String? kvCacheTypeK;

  /// FLLAMA-PATCH: optional KV-cache quantization for the V matrix.
  /// Same allowed values as [kvCacheTypeK].
  final String? kvCacheTypeV;

  String toJsonString() {
    final Map<String, dynamic> json = {
      'messages': messages.map((m) {
        final Map<String, dynamic> entry = {'role': m.role.openAiName};

        final hasImages = m.images != null && m.images!.isNotEmpty;
        final hasText = m.text.trim().isNotEmpty;

        if (hasImages) {
          // OpenAI multimodal content shape: an array of typed parts.
          // The native side flattens this back into a text prompt with the
          // existing fllama <img src="data:...;base64,..."> sentinel that the
          // mtmd extractor recognises.
          final parts = <Map<String, dynamic>>[];
          if (hasText) {
            parts.add({'type': 'text', 'text': m.text});
          }
          for (final img in m.images!) {
            parts.add({
              'type': 'image_url',
              'image_url': {
                'url':
                    'data:${img.mimeType};base64,${base64Encode(img.bytes)}',
              },
            });
          }
          entry['content'] = parts;
        } else if (hasText) {
          // Backwards-compatible string content for text-only messages so we
          // don't perturb the existing parser path or callers' diff baselines.
          entry['content'] = m.text;
        }
        // else: tool-call-only assistant message — leave `content` unset.

        if (m.toolResponseName != null) entry['name'] = m.toolResponseName;
        if (m.toolCalls?.isNotEmpty == true) entry['tool_calls'] = m.toolCalls;
        return entry;
      }).toList(),
      'tools': tools.map((t) {
        return {
          'type': 'function',
          'function': {
            'name': t.name,
            'description': t.description,
            'parameters': jsonDecode(t.jsonSchema),
          },
        };
      }).toList(),
      'temperature': temperature,
      'max_tokens': maxTokens,
      'top_p': topP,
      'frequency_penalty': frequencyPenalty,
      'presence_penalty': presencePenalty,
      if (toolChoice != null) 'tool_choice': toolChoice?.jsonName,
      if (jinjaTemplate != null) 'jinja_template': jinjaTemplate,
      'enable_thinking': enableThinking,
    };
    return jsonEncode(json);
  }

  OpenAiRequest({
    this.messages = const [],
    this.tools = const [],
    this.toolChoice,
    // Randomness of the output.
    // Higher numbers mean more likelihood of non-top tokens being chosen.
    // 0 <= temperature <= any positive number
    this.temperature = 0.7,
    // 333 * 3/4 word per token ~= 250 words ~= 1 page ~= 1 minute reading time
    this.maxTokens = 333,
    // Percent of tokens to consider. 1.0 means all tokens are considered.
    // 0.05 means only the top 5% of tokens are considered.
    this.topP = 1.0,
    this.frequencyPenalty = 0.0,
    // Match default penalty_repeat of 1.1 in llama.cpp.
    this.presencePenalty = 1.1,
    //
    // Following arguments aren't actually in OpenAI, but are used by Fllama.
    //
    //
    // Path to model's gguf.
    required this.modelPath,
    // Path to mmproj's gguf. (optional, only used for multimodal models)
    this.mmprojPath,
    // Number of layers to run on GPU. 0 means all layers on CPU. 99 means all
    // layers on GPU.
    this.numGpuLayers = 0,
    // ultra-safe for mobile inference, but rather small: ChatGPT launched with
    // 4096, today it has 16384. 1000 tokens ~= 3 pages ~= 750 words ~= 3
    // minutes reading time.
    this.contextSize = 2048,
    // Optional logger.
    this.logger,
    this.jinjaTemplate,
    // Whether the Jinja template should enable the model's reasoning
    // channel. Defaults to true (matches prior hardcoded behaviour).
    this.enableThinking = true,
    // FLLAMA-PATCH: 0 = "let llama.cpp pick from device cores".
    this.numThreads = 0,
    this.kvCacheTypeK,
    this.kvCacheTypeV,
  });
}
