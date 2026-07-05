# Future Multimodal Extension

Multimodal support is explicitly out of scope for the first 0.3 implementation.
It should only start after text belief-revision results show a measurable gain.

The future extension should reuse the same evidence-grounding principle:

- visual fact ledger for image, OCR, document, and frame evidence;
- claim-to-region grounding for object, attribute, count, and spatial claims;
- OCR span verification for document and screenshot tasks;
- visual prompt-injection defense that separates task instructions from text
  found inside untrusted media;
- multimodal RevisionTrace rows that bind each claim to a region, span, or
  frame before allowing correction or acceptance.

The success metric should remain unsupported persistence, but the evidence unit
changes from text documents to visual regions, OCR spans, tables, and video
frames.

