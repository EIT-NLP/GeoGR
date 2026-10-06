# Data Preparation

Download the selected training annotations from [EIT-NLP/GeoRG-67K-CAT](https://huggingface.co/datasets/EIT-NLP/GeoRG-67K-CAT/tree/main) and follow the dataset repository's instructions for downloading and extracting its files. The paper uses 66,940 PRISM-selected examples. Use the Hub identifier exactly as written: `GeoRG-67K-CAT`.

## 1. Default annotation layout

Place the training JSON files under the GeoGR repository root:

```text
data/processed/subsets/
├── sqa3d_train_llava_style.json
├── scanqa_train_llava_style.json
├── scan2cap_train_llava_style.json
├── scanrefer_vg_train_llava_style.json
└── multi3drefer_train_llava_style.json
```

The default manifests for [LLaVA-OneVision](../bash-paper/ov-train/data/multi_full.yaml) and [Video-3D LLM](../bash-paper/video-3d-train/data/multi_full.yaml) reference these files. `sampling_strategy: all` uses every example in each selected subset. The preparation scripts convert data formats and scene layouts; they do not perform PRISM selection.

## 2. Connect the selected annotations

LLaVA-OneVision needs no manifest override for the default layout; its preparation script resolves relative `json_path` entries against the manifest directory and expands environment variables and `~`. For a custom location, copy the bundled manifest and update its paths.

For Video-3D LLM, copy the bundled manifest and replace all five `json_path` entries with fully expanded absolute paths, even when using the default annotation layout. The launcher validates relative paths against the manifest directory, but the training loader opens them relative to `algorithm/Video-3D-LLM/`; the bundled relative entries therefore do not work unchanged. Set an absolute manifest path for both training stages:

```bash
export DATA_YAML=/absolute/path/to/geogr_train.yaml
```

A manifest with absolute annotation paths can be shared across both backends. Do not leave `$VARIABLE` or `~` in Video-3D LLM manifest entries; its training loader does not expand them.

## 3. Prepare scene assets and held-out annotations

Follow the [bundled Video-3D LLM preprocessing guide](../algorithm/Video-3D-LLM/scripts/3d/preprocessing/README.md) for ScanNet RGB/depth views, camera parameters, EmbodiedScan annotations, and scene metadata. Keep the required `scannet/`, `embodiedscan/`, and `metadata/` folders under a common scene-data root. Prepare the validation/test annotations separately from the selected training data.

If scene assets are outside the default `data/` directory, configure their roots:

```bash
# LLaVA-OneVision training preparation.
export DATA_ROOT=/absolute/path/to/scene-data

# Video-3D LLM training and evaluation.
export IMAGE_FOLDER=/absolute/path/to/scene-data
export VIDEO_FOLDER=/absolute/path/to/scene-data
export EMBODIEDSCAN_FOLDER=/absolute/path/to/scene-data/embodiedscan
```

## 4. Evaluation paths and coordinates

Copy the path template and edit its fields to match your local datasets and backend source directory:

```bash
cp lmms-eval/scannet3d_data_paths.example.yaml \
  lmms-eval/lmms_eval/tasks/_task_utils/scannet3d/data_paths.local.yaml
export THREE_D_CONFIG="$PWD/lmms-eval/lmms_eval/tasks/_task_utils/scannet3d/data_paths.local.yaml"
export SCANNET3D_COORDS_CACHE_ROOT=/absolute/path/to/scannet3d-coords-cache
```

Both evaluation backends use `THREE_D_CONFIG`. GeoSemZip needs world coordinates aligned with the sampled visual patches. Cache preparation commands are in the [reproduction guide](reproduction.md#coordinate-cache-preparation).
