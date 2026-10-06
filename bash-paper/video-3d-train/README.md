# Video3D Compression Post-Training

All scripts use the native Video3D five-task training mixture:

- SQA3D
- ScanQA
- Scan2Cap
- ScanRefer
- Multi3DRefer

The manifest consumes externally prepared annotations under `data/processed/subsets/`. Paper reproduction uses the PRISM-selected 30% mixture (66,940 examples). Each adaptation stage runs for one epoch with learning rate `1e-5`, zero weight decay, cosine decay, warm-up ratio `0.03`, BF16, and DeepSpeed ZeRO-3.

The default effective global batch is 16. On four GPUs, the Video3D micro-batch is 1 per GPU and gradient accumulation is 4. The vision encoder is frozen. The projector, language model, 3D position embedding, and grounding head remain trainable, following the native Video3D five-task training path.

This is compression post-training rather than a literal rerun of the original Video3D stage-2 recipe. The original recipe also trains the vision tower and enables `torch_compile`; these scripts keep the encoder frozen and leave `torch_compile` disabled because compressed visual sequence lengths are dynamic.

Each compression script mirrors the default compressor configuration in `bash-paper/video-3d-eval`. Override a setting with an environment variable, for example:

```bash
GPU_IDS=0,1,2,3 VOXEL_SIZE=0.15 \
  bash bash-paper/video-3d-train/train_voxel_vtc.sh
```

Before either adaptation stage, set `DATA_YAML` to an absolute path to a manifest whose `json_path` entries are fully expanded absolute annotation paths; see [data preparation](../../docs/data_preparation.md#2-connect-the-selected-annotations). The bundled relative entries pass launcher validation but are opened from a different directory by the training loader.

Use `DRY_RUN=1` to run launcher validation and print the command without loading the model or starting training. This does not verify that the training loader can open the annotations. Checkpoints are written under `video3d-checkpoint` by default.
