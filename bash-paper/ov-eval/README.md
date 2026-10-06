# LLaVA-OneVision Evaluation

These launchers evaluate projector-level compression on ScanQA and SQA3D by default. They share `_common_ov_eval.sh`, which configures greedy generation, data-parallel evaluation, ScanNet3D paths, coordinate caches, and result manifests. For the distinction between projector-only evaluation and decoder routing, see the [compression pipeline](../../docs/compression_pipeline.md).

Run from the GeoGR repository root in the shared `geogr` environment:

```bash
export ENV_NAME=geogr
# Replace xxx with your actual checkpoint path.
export MODEL_PATH=xxx
export THREE_D_CONFIG=/absolute/path/to/data_paths.local.yaml
# Replace xxx with your actual SigLIP checkpoint path.
export SIGLIP_MODEL_PATH=xxx
export GPU_IDS=0,1,2,3
```

The maintained entry points are:

```bash
bash bash-paper/ov-eval/eval_no_compression.sh
bash bash-paper/ov-eval/eval_geosemzip.sh
bash bash-paper/ov-eval/eval_visionzip.sh
bash bash-paper/ov-eval/eval_vispruner.sh
bash bash-paper/ov-eval/eval_voxel_vtc.sh
bash bash-paper/ov-eval/eval_voxel_dtc.sh
bash bash-paper/ov-eval/eval_segpruner.sh
```

All maintained projector compressor launchers use `grid_drop`. GeoSemZip uses `voxel_size=0.1`, `target_keep_ratio=0.30`, `dominant_ratio=0.85`, and max attention reduction. Other method settings are declared in each launcher.

`SCANNET3D_COORDS_CACHE_ROOT` and `OV_POOLED_COORDS_ROOT` select the precomputed coordinate cache. See `python bash-paper/ov-eval/precompute_pooled_coords.py --help` for cache preparation arguments. Use `LIMIT=1` for a small evaluation run; it still loads the model and performs inference.

Full GeoGR evaluation uses a Stage-II adapted model with [the GroupRoute launcher](../../bash-llm/ILVAS/group-wise-skip-recovery/README.md). `OUTPUT_ROOT` selects the output location; projector-only results default to `results/ov/`.
