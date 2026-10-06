# GeoGR Compression Pipeline

GeoGR compresses visual representations at the projector output and visual computation inside the language decoder. This guide explains the implementation; executable commands are in [reproduction.md](reproduction.md).

## 1. End-to-End View

The two adaptation stages and the inference policy are distinct:

| Step | Compression enabled | Model adaptation |
| --- | --- | --- |
| Stage I | GeoSemZip at the projector output | Train the projector and full LLM with the vision encoder frozen |
| Stage II | GeoSemZip plus late entry/early exit | Adapt the LLM to the visual window with the vision encoder and projector frozen |
| Final inference | GeoSemZip plus GroupRoute | Apply recoverable group-wise skipping without additional training |

The inference path is:

```text
Posed multi-view input
        |
Vision encoder + projector
        |
GeoSemZip: voxel consolidation, selection, and residual merging
        |
Serialized visual tokens + text prompt
        |
Late-entry/early-exit visual window
        |
Query-conditioned group-wise skipping within the window
        |
Decoder output and benchmark prediction
```

The projector implementation is named `voxel_vtc_visionzip`. The window-only decoder implementation is `late_entry_early_exit`; the full GroupRoute inference implementation is `group_wise_skip_recovery`.

## 2. Stage I: Projector-Side Compression

### 2.1 Inputs and coordinate alignment

Each projected patch is paired with a world coordinate reconstructed from aligned depth, camera intrinsics, and camera-to-world poses. Features, coordinates, and saliency scores must refer to the same sampled and pooled visual grid:

```text
projector features: [input patches, hidden dimension]
world coordinates: [input patches, 3]
saliency scores:   [input patches]
```

Coordinate caches avoid repeated geometry reconstruction. They do not store a replacement tokenization or bypass the compressor's current feature-dependent selection. Cache fields and ordering must match the selected backend.

### 2.2 Voxel correspondence and aggregation

GeoSemZip groups patches using rounded world coordinates with `voxel_size=0.1`. Within each occupied voxel, it averages patch features and uses the maximum attention score under the maintained `attention_reduce=max` setting. This consolidates spatially co-located observations across views.

### 2.3 Semantic and spatial budget allocation

Let `N` be the number of input patches before voxel consolidation and `K` the number of occupied voxels. The budget is `M = min(K, ceil(0.30 * N))`. It is computed from the original patch count, not 30% of the voxel count.

`dominant_ratio=0.85` allocates approximately 85% of this budget to the highest-saliency voxel tokens. The remaining budget selects contextual anchors from unselected voxels using Morton-order spatial coverage. The implementation adjusts small budgets so both branches are represented when possible.

With `residual_merge=true`, each remaining voxel is assigned to a contextual anchor by feature cosine similarity. The mean of assigned features is added to that anchor as a residual, retaining additional scene information without increasing the token budget.

### 2.4 Serialization and positions

`newline_strategy=grid_drop` serializes the selected features while preserving emitted patch order, frame association, coordinates, and selection metadata. Grid/newline format tokens are tracked separately from visual patch tokens.

Stage-II routing acts on the resulting multimodal sequence. LLaVA-OneVision uses one-dimensional Qwen2 position IDs; Video-3D LLM preserves its three-axis positional representation.

### 2.5 Stage-I training

Stage-I adaptation starts from the matching base model, enables GeoSemZip, freezes the vision encoder, and trains the projector and full LLM. Video-3D LLM's native 3D position embedding and grounding head remain trainable by default and can be controlled with explicit freeze flags.

```text
voxel_size=0.1
target_keep_ratio=0.30
dominant_ratio=0.85
attention_reduce=max
newline_strategy=grid_drop
residual_merge=true
coverage_rule=morton
```

Use [`LLaVA-OneVision train_geosemzip.sh`](../bash-paper/ov-train/train_geosemzip.sh) or [`Video-3D LLM train_geosemzip.sh`](../bash-paper/video-3d-train/train_geosemzip.sh).

### 2.6 Stage-I evaluation

Projector-only evaluation enables GeoSemZip without decoder compression. LLaVA-OneVision reports ScanQA and SQA3D; Video-3D LLM supports all five released tasks. Use [`LLaVA-OneVision eval_geosemzip.sh`](../bash-paper/ov-eval/eval_geosemzip.sh) or [`Video-3D LLM eval_geosemzip.sh`](../bash-paper/video-3d-eval/eval_geosemzip.sh). The older `eval_voxel_vtc_visionzip.sh` filenames are compatibility aliases.

## 3. Stage II: Decoder-Side Compression

### 3.1 Late visual-token entry and early visual-token exit

Visual tokens execute layers 8–23 on LLaVA-OneVision and 8–24 on Video-3D LLM. The launchers use one-based layer numbers and exclusive exit boundaries: `late_entry_layer=8`, with `early_exit_layer=24` or `25`, respectively.

Before entry, decoder layers process text tokens while visual embeddings are retained for later insertion. At entry, the visual tokens rejoin the sequence with their original full-sequence positions. At exit, visual tokens are removed from subsequent decoder computation and text processing continues.

### 3.2 GroupRoute inference

Recoverable group-wise skipping operates inside the visual window:

```text
anchor_layers=[9,13]
keep_ratios=[0.50,0.50]
recovery_layers=0
score_mode=query_attention
```

Each anchor processes all retained visual tokens. Query-attention scoring uses the anchor's block-input representations to form the groups that govern subsequent layers. Generation tasks use the final prompt token as the query; Video-3D LLM grounding tasks use the labeled `<ground>` query token.

Between anchors, high-priority patches execute normally while low-priority patches bypass entire decoder blocks. Skipped patches retain their hidden states, position IDs, and sequence slots. The next anchor updates all visual tokens and recomputes the groups, allowing previously skipped patches to rejoin. With `recovery_layers=0`, regrouping at the next anchor is the maintained recovery mechanism.

Text tokens remain active through all decoder layers. Grid/newline format tokens execute throughout the visual window and are excluded from group-wise skipping; they follow the visual sequence's late-entry/early-exit boundaries.

### 3.3 Stage-II training

Stage-II adaptation starts from the Stage-I model and preserves its GeoSemZip projector configuration. It freezes the vision encoder and projector and trains the LLM under `late_entry_early_exit`. **The maintained training launchers do not enable group-wise skipping.** It is applied at inference without an additional training stage.

Both Stage-II scripts are named `train_grouproute.sh`; this filename refers to preparing the model for final GeoGR evaluation. The runtime compressor passed to training is the window-only `late_entry_early_exit`, not `group_wise_skip_recovery`.

Video-3D LLM's native 3D position embedding and grounding head remain trainable by default. Set `FREEZE_WORLD_POSITION_EMBEDDING=1` and `FREEZE_GROUND_HEAD=1` for the LLM-only variant. Both backends require a per-device Stage-II batch size of 1.

### 3.4 Stage-II evaluation

Final evaluation supplies GeoSemZip and `group_wise_skip_recovery` to the Stage-II adapted model. For the LLaVA-OneVision window-only comparison, `MODE=baseline` enables `late_entry_early_exit` on the same model without group-wise skipping.

The inference default is `score_mode=query_attention`. Alternative modes such as `projector_vtc_visionzip` reuse projector metadata for comparisons; they do not replace the main query-conditioned protocol.

The runtime profile distinguishes retained patches (`visual_patch_tokens`) from format tokens (`visual_format_tokens`). `prefill_layer_active_patch_tokens` records active patches per decoder layer. `llm_stage_output_tokens` reports the rounded average active visual sequence length per layer, including format tokens; it is not the number of projector output patches.

## 4. Protocol Consistency

Keep the following settings fixed when comparing compression policies on a given backend:

- The starting model and Stage-I/Stage-II adaptation lineage.
- Projector settings, including token budget, voxel size, saliency reduction, residual merging, and serialization.
- Benchmark split, prompts, frame sampling, and decoding settings.
- Coordinate-cache fields, version, and sample/token alignment.
- One-based layer numbers and exclusive early-exit boundaries.
- Backend-specific positional conventions.

The launchers write the resolved model arguments and compressor settings to result manifests. Use those records to confirm the configuration of each run.

## 5. Implementation Map

| Component | LLaVA-OneVision | Video-3D LLM |
| --- | --- | --- |
| Stage-I training | [`train_geosemzip.sh`](../bash-paper/ov-train/train_geosemzip.sh) | [`train_geosemzip.sh`](../bash-paper/video-3d-train/train_geosemzip.sh) |
| Stage-I evaluation | [`eval_geosemzip.sh`](../bash-paper/ov-eval/eval_geosemzip.sh) | [`eval_geosemzip.sh`](../bash-paper/video-3d-eval/eval_geosemzip.sh) |
| Stage-II window adaptation | [`train_grouproute.sh`](../bash-llm/ov-train/train_grouproute.sh) | [`train_grouproute.sh`](../bash-llm/video3d-train/train_grouproute.sh) |
| GroupRoute inference | [`eval_geogr.sh`](../bash-llm/ILVAS/group-wise-skip-recovery/eval_geogr.sh) | [`eval_geogr.sh`](../bash-llm/ILVAS/video3d/eval_geogr.sh) |

Shared compression implementations live under `algorithm/Video-3D-LLM/llava/model/multimodal_compressor/`; LLaVA-OneVision uses the shared implementations through its backend adapters. The corresponding multimodal wiring is in each backend's `llava_arch.py` and `language_model/llava_qwen.py`.
