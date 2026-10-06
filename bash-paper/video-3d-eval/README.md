# Video-3D LLM Evaluation

These launchers evaluate Video-3D LLM on ScanQA, SQA3D, Scan2Cap, ScanRefer, and Multi3DRefer. `_common_video3d_eval.sh` configures model loading, dataset paths, distributed execution, metrics, and outputs. See the [compression pipeline](../../docs/compression_pipeline.md) for projector-only and full GeoGR evaluation semantics.

Run from the GeoGR repository root in the shared `geogr` environment:

```bash
export ENV_NAME=geogr
# Replace xxx with your actual checkpoint path.
export MODEL_PATH=xxx
export THREE_D_CONFIG=/absolute/path/to/data_paths.local.yaml
export GPU_IDS=0,1,2,3
```

If scene assets are outside the repository's `data/` folder, set `VIDEO_FOLDER` and `EMBODIEDSCAN_FOLDER` as described in [data preparation](../../docs/data_preparation.md#3-prepare-scene-assets-and-held-out-annotations).

Evaluate Stage-I GeoSemZip with:

```bash
bash bash-paper/video-3d-eval/eval_geosemzip.sh
```

The directory also contains no-compression, VisionZip, VisPruner, Voxel-VTC, Voxel-DTC, and SegPruner launchers. Maintained projector compressors use `grid_drop`. Use `TASKS` to select benchmarks and `LIMIT=1` for a small run that still loads the model and performs inference.

Full GeoGR evaluation uses a Stage-II adapted model with [the GroupRoute launcher](../../bash-llm/ILVAS/video3d/README.md). `OUTPUT_ROOT` selects the output location; projector-only results default to `results/video3d/`.
