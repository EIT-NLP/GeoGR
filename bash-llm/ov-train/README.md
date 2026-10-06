# LLaVA-OV Stage-II Training

`train_grouproute.sh` starts from a trained GeoSemZip checkpoint, preserves its projector compressor, freezes the vision encoder and projector, and trains only the language model under the `l8/e24` late-entry/early-exit window. The script verifies the Stage-I compressor configuration before launching training.

```bash
# Replace xxx with your actual Stage-I GeoSemZip checkpoint path.
MODEL_PATH=xxx \
OUTPUT_ROOT=/path/to/checkpoints \
GPU_IDS=0,1,2,3 \
bash bash-llm/ov-train/train_grouproute.sh
```

The default effective global batch size is 16. Group-wise routing is not enabled during this stage; it is applied inference-only by `eval_geogr.sh`.
