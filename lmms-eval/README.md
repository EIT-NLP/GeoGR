# Project-Specific LMMs-Eval

This directory contains a focused fork of [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) used by GeoGR. It intentionally includes only the runtime required by the LLaVA-OneVision and Video3D-LLM experiments in this repository.

## Included models

- `llava_onevision`: base adapter for the local `llava_onevision_3d` plugin.
- `video_3d`: Video3D-LLM adapter.

## Included benchmarks

- `scanqa_val`
- `sqa3d_test`
- `scan2cap_val`
- `scanrefer_val`
- `multi3drefer_val`
- `video_3d_5bench`: group containing the five tasks above.

The evaluator intentionally has no upstream catch-all benchmark registry: the task directory is an allowlist for this project. LLaVA-OneVision launchers default to `scanqa_val,sqa3d_test`; Video3D launchers may select the full five- task group when grounding and captioning results are required.

The five task entries above are the complete benchmark surface of this release. Upstream benchmark directories are deliberately excluded; adding an external task requires an explicit project change rather than being discovered from the original upstream tree.

Dataset paths are configured through the ScanNet3D environment variables documented by the repository-level README and evaluation launchers. Start from `scannet3d_data_paths.example.yaml` when creating a local path configuration. The local file is intentionally ignored by Git:

```bash
mkdir -p lmms-eval/lmms_eval/tasks/_task_utils/scannet3d
cp lmms-eval/scannet3d_data_paths.example.yaml \
  lmms-eval/lmms_eval/tasks/_task_utils/scannet3d/data_paths.local.yaml
```

No unrelated benchmark task, baseline registry, or remote LLM-judge service is shipped in this focused fork. Run comparisons through the maintained launchers under `bash-paper` and `bash-llm`.

## Usage

Install this package in editable mode from the repository root:

```bash
python -m pip install -e "./lmms-eval[video]"
```

List the retained tasks and models:

```bash
python -m lmms_eval tasks list
python -m lmms_eval models
```

Optional experiment-reporting dependencies are isolated from the default runtime:

```bash
python -m pip install -e "./lmms-eval[reporting]"
```

The focused CLI exposes `eval`, `tasks`, `models`, and `version`.

Launch normal evaluations through `bash-paper/ov-eval` or `bash-paper/video-3d-eval`; those scripts set model plugins and data paths consistently.

The original LMMs-Eval license is retained in `LICENSE`. General-purpose benchmarks, unrelated model adapters, examples, hosted services, and web UI code from upstream are not included in this focused fork.
