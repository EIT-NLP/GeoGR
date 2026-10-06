# Video3D GroupRoute

This implementation mirrors LLaVA-OV GroupRoute while preserving Video3D's three-axis rotary position IDs and grounding semantics. Generation requests score from the final prompt token; ScanRefer and Multi3DRefer score from the labeled `<ground>` token consumed by the grounding head.

The final route uses `l8/e25`, anchors `[9,13]`, keep ratios `[0.5,0.5]`, and `recovery_layers=0`. Low-priority patches bypass selected decoder blocks but remain in their original sequence positions.

```bash
# Replace xxx with your actual Stage-II late-entry/early-exit checkpoint path.
MODEL_PATH=xxx GPU_IDS=0,1,2,3 \
bash bash-llm/ILVAS/video3d/eval_geogr.sh
```

The default task list contains all five retained Video3D benchmarks. Override `TASKS`, `GPU_IDS`, or `OUTPUT_ROOT` as needed.
