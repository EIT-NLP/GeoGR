# GeoGR Paper Protocol

This document maps the paper's compression and post-training protocol to the released launchers. Configure local data paths as described in [data preparation](data_preparation.md) and run the commands in the [reproduction guide](reproduction.md).

## Benchmark splits

| Task | Evaluation split | Backends |
| --- | --- | --- |
| `scanqa_val` | ScanQA validation | LLaVA-OneVision and Video-3D LLM |
| `sqa3d_test` | SQA3D test | LLaVA-OneVision and Video-3D LLM |
| `scan2cap_val` | Scan2Cap validation | Video-3D LLM |
| `scanrefer_val` | ScanRefer validation | Video-3D LLM |
| `multi3drefer_val` | Multi3DRefer validation | Video-3D LLM |

`video_3d_5bench` groups the five tasks. LLaVA-OneVision reports ScanQA and SQA3D; Video-3D LLM also supports captioning and grounding.

## Stage I: GeoSemZip

The projector compressor consolidates cross-view patches into voxels, selects dominant and contextual tokens, and merges remaining voxel features into contextual anchors. The maintained settings are:

```text
voxel_size=0.1
target_keep_ratio=0.30
dominant_ratio=0.85
attention_reduce=max
newline_strategy=grid_drop
residual_merge=true
coverage_rule=morton
```

The target token budget is `min(number_of_voxels, ceil(0.30 * input_patch_tokens))`, where `input_patch_tokens` is measured before voxel consolidation. Format tokens are accounted for separately.

Use [`ov-train/train_geosemzip.sh`](../bash-paper/ov-train/train_geosemzip.sh) or [`video-3d-train/train_geosemzip.sh`](../bash-paper/video-3d-train/train_geosemzip.sh) for Stage-I adaptation. The vision encoder is frozen; the projector and full LLM are trainable. Projector-only evaluation uses the corresponding scripts under `bash-paper/ov-eval/` and `bash-paper/video-3d-eval/`.

## Stage II: Late-entry/early-exit adaptation

Stage II starts from Stage I, keeps GeoSemZip enabled, freezes the vision encoder and projector, and trains the LLM under the visual-computation window:

| Setting | LLaVA-OneVision | Video-3D LLM |
| --- | --- | --- |
| `late_entry_layer` | 8 | 8 |
| `early_exit_layer` | 24 | 25 |
| Active visual layers, inclusive | [8,23] | [8,24] |
| Per-device training batch size | 1 | 1 |

Layer numbers are one-based and `early_exit_layer` is exclusive. Use [`ov-train/train_grouproute.sh`](../bash-llm/ov-train/train_grouproute.sh) or [`video3d-train/train_grouproute.sh`](../bash-llm/video3d-train/train_grouproute.sh). Despite the filenames, both scripts train with `late_entry_early_exit`; they do not enable group-wise skipping.

Video-3D LLM's native 3D position embedding and grounding head remain trainable by default. Set both `FREEZE_WORLD_POSITION_EMBEDDING=1` and `FREEZE_GROUND_HEAD=1` for the LLM-only variant.

## GroupRoute inference

After window adaptation, enable recoverable group-wise skipping without additional training:

```text
anchor_layers=[9,13]
keep_ratios=[0.50,0.50]
recovery_layers=0
score_mode=query_attention
```

Anchor layers process all retained visual tokens and recompute the groups from their block-input representations. Between anchors, low-priority patches bypass complete decoder blocks while preserving their states and sequence slots. Text tokens execute throughout the decoder; grid/newline format tokens remain active throughout the visual window and are excluded from group-wise skipping.

Use [`LLaVA-OneVision eval_geogr.sh`](../bash-llm/ILVAS/group-wise-skip-recovery/eval_geogr.sh) or [`Video-3D LLM eval_geogr.sh`](../bash-llm/ILVAS/video3d/eval_geogr.sh). For the LLaVA-OneVision window-only comparison, set `MODE=baseline` on the same Stage-II model.

## Post-training data

The paper uses the selected training mixture hosted at [GeoRG-67K-CAT](https://huggingface.co/datasets/EIT-NLP/GeoRG-67K-CAT/tree/main):

| Source | Selected examples |
| --- | ---: |
| ScanRefer | 11,000 |
| Multi3DRefer | 13,151 |
| Scan2Cap | 11,000 |
| ScanQA | 7,955 |
| SQA3D | 23,834 |
| **Total** | **66,940** |

The paper selects 30% of each source from an original pool of 223,128 examples using PRISM. The bundled training manifests reference the prepared JSON files under `data/processed/subsets/`. For Video-3D LLM, use a manifest copy with absolute annotation paths as described in [data preparation](data_preparation.md#2-connect-the-selected-annotations). `multi_full.yaml` means all examples in the selected subsets; the launchers do not perform PRISM selection.

## Optimization and decoding

| Training setting | Both adaptation stages |
| --- | --- |
| Epochs | 1 |
| Global batch size | 16 |
| Learning rate | `1e-5` |
| Weight decay | 0 |
| Schedule | Cosine decay |
| Warm-up ratio | `0.03` |
| Precision | BF16 |
| Distributed optimizer | DeepSpeed ZeRO-3 |

Evaluation uses batch size 1, greedy decoding, temperature 0, one beam, and at most 512 new tokens. These are the maintained defaults; overrides must be recorded when comparing results.
