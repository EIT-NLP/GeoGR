# Video-3D LLM Environment Notes

Follow the [repository installation instructions](../../../README.md#installation) to create the shared `geogr` environment and install both backend training extras plus the evaluator. Both backends provide `llava`; the maintained launchers select the matching source directory.

```bash
conda activate geogr
export ENV_NAME=geogr
```

The maintained launchers default to `ENV_NAME=compress3d` if no override is provided. Run installation and launcher commands from the GeoGR repository root. Install FlashAttention with a build compatible with your PyTorch, CUDA, Python, and C++ ABI versions; the repository does not provide a machine-specific environment lock.

Video-3D LLM models must retain the tokenizer-compatible `config.json`, including the original `vocab_size`. Configure local model paths, data paths, and coordinate caches following the [reproduction guide](../../../docs/reproduction.md). For training, use a manifest with absolute annotation paths as described in [data preparation](../../../docs/data_preparation.md#2-connect-the-selected-annotations).
