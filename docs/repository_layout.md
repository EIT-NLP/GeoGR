# Repository Layout

This guide maps the released model code, launchers, and documentation. See [compression_pipeline.md](compression_pipeline.md) for the method and [reproduction.md](reproduction.md) for running experiments.

## Model code and evaluator

| Directory | Purpose |
| --- | --- |
| [`algorithm/LLaVA-NeXT/`](../algorithm/LLaVA-NeXT/) | LLaVA-OneVision backend and compression adapters |
| [`algorithm/Video-3D-LLM/`](../algorithm/Video-3D-LLM/) | Video-3D LLM backend and shared compressor implementations |
| [`lmms-eval/`](../lmms-eval/) | Project-specific evaluator for the two backends and five ScanNet-based benchmarks |

The backends expose the same `llava` package name and share the `geogr` environment. The maintained launchers select the matching backend source directory; see [installation](../README.md#installation). Projector compression is implemented under `multimodal_compressor/voxel_vtc_post/`; the decoder components are `late_entry_early_exit/` and `group_wise_skip_recovery/`. Backend adapters preserve the appropriate position-ID conventions.

## Training and evaluation launchers

| Directory | Purpose |
| --- | --- |
| [`bash-paper/ov-train/`](../bash-paper/ov-train/) | LLaVA-OneVision Stage-I GeoSemZip adaptation |
| [`bash-paper/video-3d-train/`](../bash-paper/video-3d-train/) | Video-3D LLM Stage-I GeoSemZip adaptation |
| [`bash-paper/ov-eval/`](../bash-paper/ov-eval/) | LLaVA-OneVision projector-only evaluation and comparisons |
| [`bash-paper/video-3d-eval/`](../bash-paper/video-3d-eval/) | Video-3D LLM projector-only evaluation and comparisons |
| [`bash-llm/ov-train/`](../bash-llm/ov-train/) | LLaVA-OneVision Stage-II late-entry/early-exit adaptation |
| [`bash-llm/video3d-train/`](../bash-llm/video3d-train/) | Video-3D LLM Stage-II late-entry/early-exit adaptation |
| [`bash-llm/ILVAS/group-wise-skip-recovery/`](../bash-llm/ILVAS/group-wise-skip-recovery/) | LLaVA-OneVision GroupRoute inference and window-only evaluation |
| [`bash-llm/ILVAS/video3d/`](../bash-llm/ILVAS/video3d/) | Video-3D LLM GroupRoute inference |

`train_grouproute.sh` adapts the late-entry/early-exit window. Group-wise skipping is enabled by `eval_geogr.sh` at inference and needs no additional training.

## Evaluation tasks

| Entry | Purpose |
| --- | --- |
| `scanqa_val` | ScanQA validation |
| `sqa3d_test` | SQA3D test |
| `scan2cap_val` | Scan2Cap validation |
| `scanrefer_val` | ScanRefer validation |
| `multi3drefer_val` | Multi3DRefer validation |
| `video_3d_5bench` | Group containing all five tasks |

LLaVA-OneVision defaults to ScanQA and SQA3D; Video-3D LLM supports all five tasks. The shared `lmms_eval/tasks/_task_utils/scannet3d/` package provides data loading, coordinate lookup, and metrics.

## Data and local configuration

The default selected training annotations live under `data/processed/subsets/`. Configure other locations with `DATA_YAML` and the backend-specific scene-root variables described in [data_preparation.md](data_preparation.md).

Create `lmms-eval/lmms_eval/tasks/_task_utils/scannet3d/data_paths.local.yaml` from [`scannet3d_data_paths.example.yaml`](../lmms-eval/scannet3d_data_paths.example.yaml). Both evaluation backends read it through `THREE_D_CONFIG`; this local file is covered by the evaluator's `.gitignore`.

Use `SCANNET3D_COORDS_CACHE_ROOT` for coordinate caches and `OUTPUT_ROOT` for training/evaluation outputs. These are runtime paths, not additional source components.

## Documentation

- [Data preparation](data_preparation.md): annotation placement and scene paths.
- [Reproduction](reproduction.md): environment variables and executable entry points.
- [Compression pipeline](compression_pipeline.md): projector compression, visual windows, and routing behavior.
- [Paper protocol](paper_protocol.md): benchmark splits and training/evaluation settings.
