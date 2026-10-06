# Reproduction Guide

This guide provides the paths and commands for GeoGR post-training and evaluation. Install the selected backend following the [README](../README.md#installation), then prepare the [training annotations and scene assets](data_preparation.md). The method's implementation details are in the [compression pipeline](compression_pipeline.md).

## 1. Environment and paths

Use the shared `geogr` environment for LLaVA-OneVision and Video-3D LLM. Install both backend training extras and the evaluator together following the [README](../README.md#installation). The maintained launchers select the matching backend source directory when importing `llava`. They default to the environment name `compress3d`, so set `ENV_NAME=geogr`.

```bash
conda activate geogr
export ENV_NAME=geogr
# Set this if the launchers cannot find conda:
# export CONDA_BASE=/absolute/path/to/miniconda3

# Replace xxx with your actual SigLIP checkpoint path.
export SIGLIP_MODEL_PATH=xxx
export VISION_TOWER="$SIGLIP_MODEL_PATH"
export THREE_D_CONFIG="$PWD/lmms-eval/lmms_eval/tasks/_task_utils/scannet3d/data_paths.local.yaml"
export SCANNET3D_COORDS_CACHE_ROOT=/absolute/path/to/scannet3d-coords-cache
export GPU_IDS=0,1,2,3
export OFFLINE_MODE=1
```

`THREE_D_CONFIG` is the local data-path YAML for both backends. Video-3D LLM training reads the vision-weight path from `VISION_TOWER`. Prepare local model and vision weights before running with `OFFLINE_MODE=1`.

Place the five selected JSON files under `data/processed/subsets/`. LLaVA-OneVision can use its bundled manifest directly. For Video-3D LLM, set `DATA_YAML` to a copy with absolute `json_path` entries before either training stage; see [data preparation](data_preparation.md#2-connect-the-selected-annotations). Configure the manifest and scene roots as needed:

```bash
export DATA_YAML=/absolute/path/to/geogr_train.yaml
export DATA_ROOT=/absolute/path/to/scene-data
export IMAGE_FOLDER=/absolute/path/to/scene-data
export VIDEO_FOLDER=/absolute/path/to/scene-data
export EMBODIEDSCAN_FOLDER=/absolute/path/to/scene-data/embodiedscan
```

`DATA_ROOT` is used by LLaVA-OneVision data preparation; the remaining scene-folder variables configure Video-3D LLM. See [data preparation](data_preparation.md) for the expected layout.

### Coordinate-cache preparation

LLaVA-OneVision training prepares frame folders and precomputes required coordinates by default. To prepare a small training-cache sample without starting optimization:

```bash
# Replace xxx with your actual checkpoint path.
MODEL_PATH=xxx \
COORD_PRECOMPUTE_LIMIT=1 PRECOMPUTE_ONLY=1 \
  bash bash-paper/ov-train/train_geosemzip.sh
```

For the full training cache, use `PRECOMPUTE_ONLY=1` without `COORD_PRECOMPUTE_LIMIT`. Prepare evaluation coordinates with the matching backend tool; its help lists the available arguments:

```bash
python bash-paper/ov-eval/precompute_pooled_coords.py --help
python bash-paper/precompute_scannet3d_coord_cache.py --help
```

The cache must match frame sampling, crop/pooling settings, token ordering, and backend-specific fields.

## 2. Stage I: GeoSemZip adaptation

Stage I freezes the vision encoder and trains the projector and full language model with GeoSemZip enabled. Run the command for your selected backend:

```bash
# LLaVA-OneVision, in geogr.
# Replace xxx with your actual checkpoint path.
MODEL_PATH=xxx \
OUTPUT_ROOT=/absolute/path/to/outputs/ov-stage1 \
  bash bash-paper/ov-train/train_geosemzip.sh

# Video-3D LLM, in the same geogr environment.
# Replace xxx with your actual checkpoint path.
MODEL_PATH=xxx \
OUTPUT_ROOT=/absolute/path/to/outputs/video3d-stage1 \
  bash bash-paper/video-3d-train/train_geosemzip.sh
```

The default projector settings are `voxel_size=0.1`, `target_keep_ratio=0.30`, `dominant_ratio=0.85`, `attention_reduce=max`, `newline_strategy=grid_drop`, `residual_merge=true`, and `coverage_rule=morton`.

Projector-only evaluation uses the corresponding Stage-I model:

```bash
# Replace xxx with your actual Stage-I GeoSemZip checkpoint path.
MODEL_PATH=xxx \
  bash bash-paper/ov-eval/eval_geosemzip.sh

# Replace xxx with your actual Stage-I GeoSemZip checkpoint path.
MODEL_PATH=xxx \
  bash bash-paper/video-3d-eval/eval_geosemzip.sh
```

## 3. Stage II: Late-entry/early-exit adaptation

Stage II starts from the matching Stage-I model. It freezes the vision encoder and projector and trains the language model with the visual-computation window enabled. The scripts are named `train_grouproute.sh`, but they configure `late_entry_early_exit`, not `group_wise_skip_recovery`. **Group-wise skipping is not enabled during this training stage.**

```bash
# LLaVA-OneVision: visual window [8,23].
# Replace xxx with your actual Stage-I GeoSemZip checkpoint path.
MODEL_PATH=xxx \
OUTPUT_ROOT=/absolute/path/to/outputs/ov-stage2 \
  bash bash-llm/ov-train/train_grouproute.sh

# Video-3D LLM: visual window [8,24].
# Replace xxx with your actual Stage-I GeoSemZip checkpoint path.
MODEL_PATH=xxx \
OUTPUT_ROOT=/absolute/path/to/outputs/video3d-stage2 \
  bash bash-llm/video3d-train/train_grouproute.sh
```

Layer numbers are one-based. The exit boundary is exclusive: `EARLY_EXIT_LAYER=24` for LLaVA-OneVision and `25` for Video-3D LLM. Stage II requires `PER_DEVICE_TRAIN_BATCH_SIZE=1`; gradient accumulation preserves the default global batch size of 16.

Video-3D LLM's native 3D position embedding and grounding head remain trainable by default. Set `FREEZE_WORLD_POSITION_EMBEDDING=1` and `FREEZE_GROUND_HEAD=1` for the LLM-only variant. The [paper protocol](paper_protocol.md) lists the remaining training settings.

## 4. Training-free group-wise skipping at inference

Evaluate the Stage-II model with GeoSemZip, late entry/early exit, and recoverable group-wise skipping. This inference step requires no additional training:

```bash
# LLaVA-OneVision: ScanQA and SQA3D.
# Replace xxx with your actual Stage-II late-entry/early-exit checkpoint path.
MODEL_PATH=xxx \
OUTPUT_ROOT=/absolute/path/to/results/ov \
  bash bash-llm/ILVAS/group-wise-skip-recovery/eval_geogr.sh

# Video-3D LLM: all five benchmarks.
# Replace xxx with your actual Stage-II late-entry/early-exit checkpoint path.
MODEL_PATH=xxx \
OUTPUT_ROOT=/absolute/path/to/results/video3d \
  bash bash-llm/ILVAS/video3d/eval_geogr.sh
```

The inference route uses anchors `[9,13]`, keep ratios `[0.5,0.5]`, query-attention scoring, and `recovery_layers=0`. Low-priority patches preserve their states and can rejoin at the next anchor. Override `TASKS`, `GPU_IDS`, `LIMIT`, and `OUTPUT_ROOT` as needed.

For the LLaVA-OneVision window-only comparison, use the same Stage-II model with `MODE=baseline`:

```bash
# Replace xxx with your actual Stage-II late-entry/early-exit checkpoint path.
MODEL_PATH=xxx MODE=baseline \
  bash bash-llm/ILVAS/group-wise-skip-recovery/eval_geogr.sh
```
