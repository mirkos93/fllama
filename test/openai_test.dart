import 'dart:convert';
import 'dart:typed_data';

import 'package:fllama/misc/openai.dart';
import 'package:test/test.dart';

void main() {
  group('OpenAiRequest.toJsonString', () {
    test('text-only message keeps the legacy string `content` field', () {
      final req = OpenAiRequest(
        modelPath: '/tmp/fake.gguf',
        messages: [
          Message(Role.user, 'hello'),
        ],
      );

      final body = jsonDecode(req.toJsonString()) as Map<String, dynamic>;
      final messages = body['messages'] as List<dynamic>;
      expect(messages.length, 1);
      final m0 = messages.first as Map<String, dynamic>;
      expect(m0['role'], 'user');
      // Backwards-compat: a plain string, not a parts array.
      expect(m0['content'], 'hello');
    });

    test('message with images emits OpenAI multimodal parts array', () {
      final png = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47]);
      final req = OpenAiRequest(
        modelPath: '/tmp/fake.gguf',
        mmprojPath: '/tmp/fake.mmproj.gguf',
        messages: [
          Message(
            Role.user,
            'describe this',
            images: [MessageImage(png, mimeType: 'image/png')],
          ),
        ],
      );

      final body = jsonDecode(req.toJsonString()) as Map<String, dynamic>;
      final m0 = (body['messages'] as List).first as Map<String, dynamic>;
      expect(m0['role'], 'user');
      expect(m0['content'], isA<List<dynamic>>());

      final parts = (m0['content'] as List).cast<Map<String, dynamic>>();
      expect(parts.length, 2);
      expect(parts[0], {'type': 'text', 'text': 'describe this'});
      expect(parts[1]['type'], 'image_url');
      final url = (parts[1]['image_url'] as Map)['url'] as String;
      expect(url, startsWith('data:image/png;base64,'));
      // Round-trip: the suffix must decode to the original bytes.
      final b64 = url.substring('data:image/png;base64,'.length);
      expect(base64Decode(b64), equals(png));
    });

    test('image-only message (empty text) still emits the image part', () {
      final jpg = Uint8List.fromList([0xFF, 0xD8, 0xFF]);
      final req = OpenAiRequest(
        modelPath: '/tmp/fake.gguf',
        mmprojPath: '/tmp/fake.mmproj.gguf',
        messages: [
          Message(Role.user, '', images: [MessageImage(jpg, mimeType: 'image/jpeg')]),
        ],
      );

      final body = jsonDecode(req.toJsonString()) as Map<String, dynamic>;
      final m0 = (body['messages'] as List).first as Map<String, dynamic>;
      final parts = (m0['content'] as List).cast<Map<String, dynamic>>();
      // No empty text part; just the image.
      expect(parts.length, 1);
      expect(parts[0]['type'], 'image_url');
    });

    test('mimeType defaults to image/png', () {
      final bytes = Uint8List.fromList([1, 2, 3]);
      final img = MessageImage(bytes);
      expect(img.mimeType, 'image/png');
    });
  });
}
