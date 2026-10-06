# Video3D Stage-II Training

`train_grouproute.sh` starts from a Video3D GeoSemZip checkpoint and trains the language model under the `l8/e25` late-entry/early-exit window. The vision encoder and projector remain frozen. Video3D's position embedding and grounding head remain trainable by default and can be frozen explicitly with `FREEZE_WORLD_POSITION_EMBEDDING=1` or `FREEZE_GROUND_HEAD=1`.

```bash
# Replace xxx with your actual Stage-I GeoSemZip checkpoint path.
MODEL_PATH=xxx \
OUTPUT_ROOT=/path/to/checkpoints \
GPU_IDS=0,1,2,3 \
bash bash-llm/video3d-train/train_grouproute.sh
```

Use the manifest with absolute annotation paths described in [data preparation](../../docs/data_preparation.md#2-connect-the-selected-annotations) for this stage as well.

Group-wise routing is inference-only and is enabled by `bash-llm/ILVAS/video3d/eval_geogr.sh` after this adaptation stage.
