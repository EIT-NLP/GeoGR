# GroupRoute Evaluation

This directory contains launchers for GeoGR's decoder routing. The shared implementations live under `algorithm/Video-3D-LLM/llava/model/multimodal_compressor/`; LLaVA-OneVision uses backend adapters.

- [group-wise-skip-recovery](group-wise-skip-recovery/README.md): LLaVA-OneVision GroupRoute and window-only evaluation.
- [video3d](video3d/README.md): Video-3D LLM GroupRoute with three-axis position IDs and grounding-query support.

The maintained protocol uses query-attention grouping at anchors `[9,13]`, keeps 50% of patch tokens in each routed interval, and sets `recovery_layers=0`. LLaVA-OneVision uses `late_entry_layer=8, early_exit_layer=24`; Video-3D LLM uses `late_entry_layer=8, early_exit_layer=25`. Layer numbers are one-based and exit boundaries are exclusive.

Late-entry/early-exit adaptation is performed during Stage II. Group-wise skipping is enabled at inference without additional training. Anchor and keep-ratio settings are exposed as launcher variables.
