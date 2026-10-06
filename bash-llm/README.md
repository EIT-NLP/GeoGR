# LLM Compression

This directory contains late-entry/early-exit adaptation and final GeoGR inference. **Training enables the visual window; group-wise skipping is applied only at inference and requires no additional training.**

| Directory | Purpose |
| --- | --- |
| [`ov-train/`](ov-train/) | LLaVA-OneVision Stage-II late-entry/early-exit adaptation |
| [`video3d-train/`](video3d-train/) | Video-3D LLM Stage-II late-entry/early-exit adaptation |
| [`ILVAS/group-wise-skip-recovery/`](ILVAS/group-wise-skip-recovery/) | LLaVA-OneVision GroupRoute inference; `MODE=baseline` selects the window-only comparison |
| [`ILVAS/video3d/`](ILVAS/video3d/) | Video-3D LLM GroupRoute inference |

The training scripts retain the name `train_grouproute.sh`, but pass `late_entry_early_exit` as the LLM compressor. They do not train with `group_wise_skip_recovery`.

The visual window is `l8/e24` for LLaVA-OneVision and `l8/e25` for Video-3D LLM, with exclusive exit boundaries. Final inference uses anchors `[9,13]` and keep ratios `[0.5,0.5]`.

See the [compression pipeline](../docs/compression_pipeline.md) for execution details and the [reproduction guide](../docs/reproduction.md) for runnable commands.
