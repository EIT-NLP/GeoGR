# LLaVA-OneVision Post-Training

These launchers freeze the vision encoder and tune the multimodal projector and language model. The default dataset manifest is:

`bash-paper/ov-train/data/multi_full.yaml`

It contains the five maintained ScanNet3D training sources: SQA3D, ScanQA, Scan2Cap, ScanRefer, and Multi3DRefer. OV reports primarily use ScanQA and SQA3D, but the post-training mixture also includes the grounding sources so that the shared model protocol is consistent with Video3D.

The manifest points to externally prepared subsets under `data/processed/subsets/`. Paper reproduction requires the PRISM-selected 30% mixture (66,940 examples); preparation does not perform that selection. Both adaptation stages use one epoch, learning rate `1e-5`, zero weight decay, cosine decay, warm-up ratio `0.03`, BF16, and DeepSpeed ZeRO-3.

Dataset paths in YAML manifests are resolved relative to the manifest itself. Environment variables and `~` are also supported.

## Launchers

Run the no-compression baseline:

```bash
bash bash-paper/ov-train/train_no_compression.sh
```

Run Stage-I GeoSemZip adaptation (the legacy name is `train_voxel_vtc_visionzip.sh`):

```bash
bash bash-paper/ov-train/train_geosemzip.sh
```

The compressor defaults are defined by the launcher and can be overridden with environment variables such as `VOXEL_SIZE`, `TARGET_KEEP_RATIO`, `DOMINANT_RATIO`, and `ATTENTION_REDUCE`.

ScanNet coordinates are auxiliary projector inputs used for token selection and merging. They are not added to the model as visual or text tokens. Compressors that require coordinates use the cache under `cache/scannet3d_coords`; missing entries are treated as errors instead of silently skipping samples.

## Batch Configuration

The default is four GPUs with an effective global batch size of 16:

```bash
GPU_IDS=0,1,2,3
PER_DEVICE_TRAIN_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=16
```

The common launcher computes gradient accumulation as:

```text
gradient_accumulation_steps = GLOBAL_BATCH_SIZE /
    (number_of_gpus * PER_DEVICE_TRAIN_BATCH_SIZE)
```

Per-method launchers define compressor settings only. Shared optimization and batch settings live in `_common_ov_train.sh`.

## Paths and Outputs

`MODEL_PATH` and `SIGLIP_MODEL_PATH` must be supplied for a local run. Output checkpoints are written to `checkpoints/ov/<run_name>` by default. Override `MODEL_PATH`, `SIGLIP_MODEL_PATH`, `OUTPUT_ROOT`, or `RUN_NAME` for another layout.

Runs resume from the latest `checkpoint-*` directory when the output directory already contains checkpoints. To select one explicitly:

```bash
# Replace xxx with the actual training checkpoint path to resume from.
RESUME_FROM_CHECKPOINT=xxx \
bash bash-paper/ov-train/train_geosemzip.sh
```

The data preparation step creates the frame layout expected by LLaVA-OV under `bash-paper/ov-train/data/prepared_full_ov`. This generated directory is a local training artifact.

`DRY_RUN=1` still validates paths and prepares or reuses the OV frame layout, but skips coordinate precomputation and optimization and prints the final command:

```bash
DRY_RUN=1 bash bash-paper/ov-train/train_no_compression.sh
DRY_RUN=1 bash bash-paper/ov-train/train_geosemzip.sh
```

Set `FORCE_PREPARE_OV_DATA=1` when changing the source manifest or annotations to rebuild an existing prepared layout.

Precompute a small coordinate-cache sample:

```bash
COORD_PRECOMPUTE_LIMIT=1 PRECOMPUTE_ONLY=1 \
bash bash-paper/ov-train/train_geosemzip.sh
```
