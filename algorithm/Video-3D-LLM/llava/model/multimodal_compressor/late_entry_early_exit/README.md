# LLaVA-OV Late Entry / Early Exit

`late_entry_early_exit` is an LLM-stage compressor for inference and teacher-forced post-training. It migrates the late-entry and early-exit stages of HiDrop without importing HiDrop and without its intermediate-layer token pruning.

Layer numbers are one-based. With the library defaults `late_entry_layer=9` and `early_exit_layer=25` on the 28-layer LLaVA-OV Qwen2 decoder:

- layers 1-8 process only text tokens;
- layers 9-24 process the complete text and visual sequence;
- layers 25-28 process only text tokens.

These are library defaults, not the GeoGR launcher settings. The maintained GeoGR scripts override them to `late_entry_layer=8, early_exit_layer=24` for LLaVA-OneVision and `late_entry_layer=8, early_exit_layer=25` for Video-3D LLM; see the [paper protocol](../../../../../../docs/paper_protocol.md).

Visual tokens are physically removed rather than attention-masked. Original projector embeddings are inserted immediately before the late-entry layer, and the resulting visual hidden states are removed immediately before the early-exit layer. No visual token is selected, merged, or pruned within the active interval.

Prompt KV cache lengths therefore differ by layer. Decode uses each layer's own cache length and contiguous RoPE position so the shortened text-only cache remains valid. The current LLaVA-OV integration requires `batch_size=1`. Generation uses `use_cache=true`; training uses a single prefill pass with `use_cache=false` so gradient checkpointing remains valid.
