# LLaVA-OneVision Backend

This directory is a focused fork of
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) used by GeoGR. It keeps
the model, training, and multimodal-compression code required for
LLaVA-OneVision experiments. Upstream demos, hosted services, unrelated
evaluation programs, archived launchers, and sample media are intentionally
excluded.

Use the repository-level entry points instead of invoking this backend
directly:

- Stage-I training: `bash-paper/ov-train/train_geosemzip.sh`
- Stage-I evaluation: `bash-paper/ov-eval/eval_geosemzip.sh`
- Stage-II training: `bash-llm/ov-train/train_grouproute.sh`
- Full GeoGR evaluation:
  `bash-llm/ILVAS/group-wise-skip-recovery/eval_geogr.sh`

The launchers add this directory to `PYTHONPATH`, so it does not need to be
installed alongside the separate Video3D `llava` package. The original
Apache-2.0 license is retained in `LICENSE`.
