# Video3D-LLM Backend

This directory is a focused fork of
[Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM) used by GeoGR. It
keeps the model, training, 3D preprocessing, and multimodal-compression code
required by the project. The duplicated legacy benchmark adapter, Web demo,
hosted-service code, and sample media are intentionally excluded.

Use the repository-level entry points:

- Stage-I training: `bash-paper/video-3d-train/train_geosemzip.sh`
- Stage-I evaluation: `bash-paper/video-3d-eval/eval_geosemzip.sh`
- Stage-II training: `bash-llm/video3d-train/train_grouproute.sh`
- Full GeoGR evaluation: `bash-llm/ILVAS/video3d/eval_geogr.sh`

ScanNet3D data preparation utilities remain under
`scripts/3d/preprocessing`. Benchmark definitions and metric implementations
are centralized in the repository-level `lmms-eval` package. The original
Apache-2.0 license is retained in `LICENSE`.
