# LLaVA-OneVision GroupRoute

This directory contains evaluation launchers. Query-conditioned decoder-block bypass is implemented by the [shared compressor](../../../algorithm/Video-3D-LLM/llava/model/multimodal_compressor/group_wise_skip_recovery/compressor.py), with a LLaVA-OneVision backend adapter. The maintained configuration is:

```text
late_entry_layer=8
early_exit_layer=24
anchor_layers=[9,13]
keep_ratios=[0.5,0.5]
recovery_layers=0
score_mode=query_attention
```

Each anchor scores patch tokens from its block-input representations and executes all retained visual tokens. The groups govern subsequent routed layers, where low-priority patches bypass the decoder block while text, high-priority patches, and grid/newline format tokens execute. Skipped patch states and position IDs remain available for regrouping at the next anchor. Visual patches and format tokens follow the late-entry/early-exit window; text executes throughout the decoder.

Run from the GeoGR repository root in the shared `geogr` environment after Stage-II window adaptation:

```bash
# Replace xxx with your actual Stage-II late-entry/early-exit checkpoint path.
ENV_NAME=geogr MODEL_PATH=xxx GPU_IDS=0,1,2,3 \
  bash bash-llm/ILVAS/group-wise-skip-recovery/eval_geogr.sh
```

Set `MODE=baseline` on the same launcher for window-only evaluation without group-wise skipping. For an eight-rank evaluation, use `eval_llava_ov_group_wise_skip_recovery_8gpu.sh`. Outputs default to `results/ov/geogr` unless `OUTPUT_ROOT` is set. Shared environment and data-path setup is in the [reproduction guide](../../../docs/reproduction.md).
