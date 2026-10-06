from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

# The package is shared by both model backends.  ``llava`` resolves to the
# active backend, so this selects the matching one-dimensional OV adapter or
# three-axis Video3D adapter without duplicating GroupRoute itself.
from llava.model.multimodal_compressor.base import CompressorConfig
from llava.model.multimodal_compressor.late_entry_early_exit_adapter import LateEntryEarlyExitCompressor
from llava.model.multimodal_compressor.registry import register_compressor


@dataclass
class GroupWiseSkipRecoveryConfig(CompressorConfig):
    position: str = "llm"
    late_entry_layer: int = 8
    early_exit_layer: int = 25
    anchor_layers: Tuple[int, ...] = (9, 13)
    keep_ratios: Tuple[float, ...] = (0.5, 0.5)
    recovery_layers: int = 0
    score_mode: str = "query_attention"
    projector_dominant_ratio: float = 0.85
    projector_voxel_size: float = 0.1
    random_seed: int = 0
    debug_trace: bool = False
    # dynamic: recompute Top-K from the current model; write: compute with the
    # frozen teacher and persist it; read: require and reuse the teacher mask.
    mask_cache_mode: str = "dynamic"
    mask_cache_dir: Optional[str] = None


def _as_tuple(values: Sequence[Any], cast, field_name: str) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a JSON array, not a string.")
    try:
        return tuple(cast(value) for value in values)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable.") from None


@register_compressor("group_wise_skip_recovery")
class GroupWiseSkipRecoveryCompressor(LateEntryEarlyExitCompressor):
    """Video3D group-wise decoder scheduling with three-axis RoPE support."""

    def __init__(self, config: Dict[str, Any]):
        cfg = GroupWiseSkipRecoveryConfig(**config) if isinstance(config, dict) else config
        cfg.anchor_layers = _as_tuple(cfg.anchor_layers, int, "anchor_layers")
        cfg.keep_ratios = _as_tuple(cfg.keep_ratios, float, "keep_ratios")
        self._validate_config(cfg)
        super().__init__(
            {
                "position": "llm",
                "late_entry_layer": int(cfg.late_entry_layer),
                "early_exit_layer": int(cfg.early_exit_layer),
            }
        )
        self.group_config = cfg
        self.compression_config = cfg
        self._sample_seed = int(cfg.random_seed)
        self._debug_trace = []
        self._pending_projector_patch_metadata: Optional[Dict[str, torch.Tensor]] = None
        self._mask_cache_dir = Path(cfg.mask_cache_dir).expanduser() if cfg.mask_cache_dir else None
        if cfg.mask_cache_mode == "write":
            self._mask_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_config(cfg: GroupWiseSkipRecoveryConfig) -> None:
        entry = int(cfg.late_entry_layer)
        exit_layer = int(cfg.early_exit_layer)
        anchors = tuple(cfg.anchor_layers)
        ratios = tuple(cfg.keep_ratios)
        recovery = int(cfg.recovery_layers)
        if cfg.position != "llm":
            raise ValueError("group_wise_skip_recovery requires position='llm'.")
        if entry < 1 or exit_layer <= entry:
            raise ValueError("Expected 1 <= late_entry_layer < early_exit_layer.")
        if not anchors or len(anchors) != len(ratios):
            raise ValueError("anchor_layers and keep_ratios must be non-empty and have equal lengths.")
        if tuple(sorted(set(anchors))) != anchors:
            raise ValueError("anchor_layers must be strictly increasing and unique.")
        if anchors[0] < entry or anchors[-1] >= exit_layer:
            raise ValueError("Every anchor must be within [late_entry_layer, early_exit_layer).")
        if any(not 0.0 < ratio <= 1.0 for ratio in ratios):
            raise ValueError("Every keep ratio must be within (0, 1].")
        if recovery < 0:
            raise ValueError("recovery_layers must be non-negative.")
        boundaries = (*anchors[1:], exit_layer)
        for anchor, boundary in zip(anchors, boundaries):
            if recovery > boundary - anchor - 1:
                raise ValueError(
                    f"recovery_layers={recovery} leaves no valid interval after anchor {anchor} "
                    f"and before boundary {boundary}."
                )
        if cfg.score_mode not in {"query_attention", "projector_vtc_visionzip", "random"}:
            raise ValueError(
                "score_mode must be 'query_attention', 'projector_vtc_visionzip', or 'random'."
            )
        if cfg.mask_cache_mode not in {"dynamic", "write", "read"}:
            raise ValueError("mask_cache_mode must be 'dynamic', 'write', or 'read'.")
        if cfg.mask_cache_mode != "dynamic" and not cfg.mask_cache_dir:
            raise ValueError("mask_cache_dir is required when mask_cache_mode is write or read.")
        if not 0.0 < float(cfg.projector_dominant_ratio) < 1.0:
            raise ValueError("projector_dominant_ratio must be within (0, 1).")
        if float(cfg.projector_voxel_size) <= 0.0:
            raise ValueError("projector_voxel_size must be positive.")

    def set_projector_patch_metadata(
        self,
        *,
        scores: torch.Tensor,
        coordinates: torch.Tensor,
        priority: Optional[torch.Tensor] = None,
    ) -> None:
        """Receive projector metadata aligned to final serialized patch tokens."""
        if self.group_config.score_mode != "projector_vtc_visionzip":
            return
        scores = torch.as_tensor(scores).detach().flatten()
        coordinates = torch.as_tensor(coordinates).detach()
        if priority is None:
            raise ValueError(
                "Projector-derived grouping requires the Stage-I projector_patch_priority metadata."
            )
        priority = torch.as_tensor(priority).detach().flatten()
        if scores.numel() == 0 or not bool(torch.isfinite(scores).all()):
            raise ValueError("Projector patch scores must be non-empty and finite.")
        if coordinates.dim() != 2 or coordinates.shape[1] != 3:
            raise ValueError("Projector patch coordinates must have shape [N, 3].")
        if int(coordinates.shape[0]) != int(scores.numel()):
            raise ValueError(
                "Projector score/coordinate counts must match: "
                f"scores={int(scores.numel())}, coordinates={int(coordinates.shape[0])}."
            )
        if priority.numel() != scores.numel():
            raise ValueError(
                "Projector patch priorities must align with scores: "
                f"priority={int(priority.numel())}, scores={int(scores.numel())}."
            )
        if priority.is_floating_point():
            if not bool(torch.isfinite(priority).all()) or not bool(torch.equal(priority, priority.round())):
                raise ValueError("Projector patch priorities must be finite integer ranks.")
        priority = priority.to(dtype=torch.long)
        if int(torch.unique(priority).numel()) != int(priority.numel()):
            raise ValueError("Projector patch priorities must be unique.")
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("Projector patch coordinates must be finite.")
        if self._pending_projector_patch_metadata is not None:
            raise ValueError("Only one compressed visual sequence is supported per request.")
        self._pending_projector_patch_metadata = {
            "scores": scores,
            "coordinates": coordinates,
            "priority": priority,
        }

    def prepare_generation_compression(self, **kwargs) -> None:
        # ``sample_cache_ids`` is dataset metadata owned by this compressor.
        # The LEE parent only accepts prompt-layout arguments, so do not pass
        # this private field through the parent signature.
        parent_kwargs = dict(kwargs)
        parent_kwargs.pop("sample_cache_ids", None)
        super().prepare_generation_compression(**parent_kwargs)
        self._generation_state["query_mode"] = "text_generation"
        ids = kwargs["original_input_ids"].detach().to(device="cpu", dtype=torch.int64).contiguous()
        digest = hashlib.blake2b(ids.numpy().tobytes(), digest_size=8).digest()
        self._sample_seed = int(self.group_config.random_seed) ^ int.from_bytes(digest, "little")
        self._generation_state["input_ids_digest"] = digest.hex()
        sample_cache_ids = kwargs.get("sample_cache_ids")
        if sample_cache_ids is not None:
            if len(sample_cache_ids) != 1 or not str(sample_cache_ids[0]):
                raise ValueError(
                    "Fixed-mask sample identity requires exactly one non-empty sample_cache_id."
                )
            self._generation_state["sample_cache_id"] = str(sample_cache_ids[0])
        if self.group_config.mask_cache_mode != "dynamic" and "sample_cache_id" not in self._generation_state:
            raise ValueError(
                "mask_cache_mode=write/read requires a stable dataset sample ID; "
                "set LLAVA_FIXED_MASK_SAMPLE_IDS=1 in the training launcher."
            )
        self._debug_trace = []
        if self.group_config.score_mode == "projector_vtc_visionzip":
            if self._pending_projector_patch_metadata is None:
                raise ValueError(
                    "projector_vtc_visionzip grouping requires aligned scores and coordinates "
                    "from voxel_vtc_visionzip."
                )
            self._generation_state["projector_patch_metadata"] = self._pending_projector_patch_metadata
            self._pending_projector_patch_metadata = None

    def prepare_grounding_compression(self, **kwargs) -> None:
        """Register group-wise compression using the ``<ground>`` query token."""
        ground_token_ids = kwargs.pop("ground_token_ids", None)
        if ground_token_ids is None:
            raise ValueError("Grounding group-wise compression requires ground_token_ids.")
        ground_token_ids = tuple(int(token_id) for token_id in ground_token_ids)
        if not ground_token_ids:
            raise ValueError("ground_token_ids must contain at least one token id.")
        self.prepare_generation_compression(**kwargs)
        self._generation_state["query_mode"] = "grounding"
        self._generation_state["ground_token_ids"] = ground_token_ids

    def prepare_training_compression(self, **kwargs) -> None:
        # Reuse the exact prompt-layout registration and projector-score
        # handoff used by generation, then mark this request as training.
        self.prepare_generation_compression(**kwargs)
        if self._generation_state is not None:
            self._generation_state["mode"] = "training"

    def clear_generation_compression(self) -> None:
        super().clear_generation_compression()
        self._debug_trace = []
        self._pending_projector_patch_metadata = None

    @staticmethod
    def _spatial_sort_indices(coordinates: torch.Tensor, voxel_size: float) -> torch.Tensor:
        """Match voxel_vtc_visionzip's deterministic 3D Morton ordering."""
        if coordinates.shape[0] == 0:
            return torch.empty(0, device=coordinates.device, dtype=torch.long)
        voxel_indices = torch.round(coordinates.float() / voxel_size).to(dtype=torch.long)
        keys = voxel_indices.detach().cpu().tolist()
        offset = -int(voxel_indices.min().item()) if int(voxel_indices.min().item()) < 0 else 0
        max_value = int((voxel_indices + offset).max().item())
        max_bits = max(1, max_value.bit_length())

        def morton_code(key) -> int:
            x, y, z = [int(value) + offset for value in key]
            code = 0
            for bit in range(max_bits):
                code |= ((x >> bit) & 1) << (3 * bit + 2)
                code |= ((y >> bit) & 1) << (3 * bit + 1)
                code |= ((z >> bit) & 1) << (3 * bit)
            return code

        order = sorted(range(len(keys)), key=lambda idx: (morton_code(keys[idx]), keys[idx]))
        return torch.tensor(order, device=coordinates.device, dtype=torch.long)

    def _select_projector_vtc_visionzip(
        self,
        *,
        patch_indices: torch.Tensor,
        keep_count: int,
    ) -> torch.Tensor:
        metadata = self._generation_state.get("projector_patch_metadata")
        if metadata is None:
            raise RuntimeError("Projector VTC-VisionZip metadata was not registered for this request.")
        scores = metadata["scores"].to(device=patch_indices.device, dtype=torch.float32).flatten()
        coordinates = metadata["coordinates"].to(device=patch_indices.device, dtype=torch.float32)
        patch_count = int(patch_indices.numel())
        if int(scores.numel()) != patch_count or tuple(coordinates.shape) != (patch_count, 3):
            raise ValueError(
                "Projector metadata does not match final LLM patch tokens: "
                f"scores={int(scores.numel())}, coordinates={tuple(coordinates.shape)}, "
                f"patches={patch_count}."
            )
        if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(coordinates).all()):
            raise ValueError("Projector scores and coordinates must remain finite.")

        dominant_count = int(round(keep_count * float(self.group_config.projector_dominant_ratio)))
        if keep_count > 1:
            dominant_count = min(max(dominant_count, 1), keep_count - 1)
        else:
            dominant_count = keep_count
        contextual_count = keep_count - dominant_count

        # Final serialized token order is the deterministic tie breaker.
        dominant_local = torch.argsort(scores, descending=True, stable=True)[:dominant_count]
        residual_mask = torch.ones(patch_count, device=patch_indices.device, dtype=torch.bool)
        residual_mask[dominant_local] = False
        residual_local = torch.nonzero(residual_mask, as_tuple=False).flatten()
        if contextual_count <= 0 or residual_local.numel() == 0:
            return dominant_local
        residual_order = self._spatial_sort_indices(
            coordinates.index_select(0, residual_local),
            float(self.group_config.projector_voxel_size),
        )
        residual_sorted = residual_local.index_select(0, residual_order)
        contextual_count = min(contextual_count, int(residual_sorted.numel()))
        step = max(1, int(residual_sorted.numel() // contextual_count))
        positions = torch.arange(
            0, residual_sorted.numel(), step, device=residual_sorted.device
        )[:contextual_count]
        contextual_local = residual_sorted.index_select(0, positions)
        return torch.cat((dominant_local, contextual_local), dim=0)

    @staticmethod
    def _skip_anchor_for_layer(
        layer_number: int,
        anchors: Tuple[int, ...],
        early_exit_layer: int,
        recovery_layers: int,
    ) -> Optional[int]:
        if layer_number in anchors:
            return None
        boundaries = (*anchors[1:], early_exit_layer)
        for interval_idx, (anchor, boundary) in enumerate(zip(anchors, boundaries)):
            # A following anchor executes and is the final recovery layer;
            # early_exit itself does not execute visual tokens.
            is_anchor_boundary = interval_idx < len(anchors) - 1
            recovery_start = boundary - recovery_layers + (1 if is_anchor_boundary else 0)
            if anchor < layer_number < recovery_start:
                return anchor
        return None

    @staticmethod
    def _query_attention_scores(
        *,
        layer,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        patch_mask: torch.Tensor,
        text_mask: torch.Tensor,
        full_position_span: int,
    ) -> torch.Tensor:
        text_indices = torch.nonzero(text_mask, as_tuple=False).flatten()
        if text_indices.numel() == 0:
            raise ValueError("Query-attention grouping requires at least one text token.")
        patch_indices = torch.nonzero(patch_mask, as_tuple=False).flatten()
        if patch_indices.numel() == 0:
            raise ValueError("Query-attention grouping requires at least one visual patch token.")

        self_attn = layer.self_attn
        normalized = layer.input_layernorm(hidden_states)
        batch_size, seq_len, _ = normalized.shape
        query_states = self_attn.q_proj(normalized).view(
            batch_size, seq_len, self_attn.num_heads, self_attn.head_dim
        ).transpose(1, 2)
        key_states = self_attn.k_proj(normalized).view(
            batch_size, seq_len, self_attn.num_key_value_heads, self_attn.head_dim
        ).transpose(1, 2)
        if position_ids.dim() == 3:
            # Video3D uses its local three-axis mRoPE implementation. Import it
            # only on this path so the shared compressor remains importable by
            # LLaVA-OV, whose package intentionally has no local qwen2 module.
            from llava.model.language_model.qwen2.modeling_qwen2 import (
                apply_rotary_pos_emb,
            )

            cos, sin = self_attn.rotary_emb(key_states, position_ids)
        elif position_ids.dim() == 2:
            from transformers.models.qwen2.modeling_qwen2 import (
                apply_rotary_pos_emb,
            )

            cos, sin = self_attn.rotary_emb(
                key_states,
                seq_len=max(int(full_position_span), int(seq_len)),
            )
        else:
            raise ValueError(
                "Query-attention position_ids must be two-dimensional for "
                f"LLaVA-OV or three-dimensional for Video3D, got {tuple(position_ids.shape)}."
            )
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # This is the standard grouped-query attention KV expansion. Keeping
        # it local avoids binding the shared compressor to either Qwen2 module.
        repeats = int(self_attn.num_key_value_groups)
        if repeats != 1:
            batch, kv_heads, key_length, head_dim = key_states.shape
            key_states = (
                key_states[:, :, None, :, :]
                .expand(batch, kv_heads, repeats, key_length, head_dim)
                .reshape(batch, kv_heads * repeats, key_length, head_dim)
            )

        query_idx = int(text_indices[-1].item())
        query = query_states[:, :, query_idx : query_idx + 1, :]
        logits = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(self_attn.head_dim)
        # Keep the definition causal even if a future prompt layout places tokens after the query.
        # Causality follows serialized prompt order, not the numeric values of
        # spatial position IDs.  Video3D's x/y/z coordinates are not a
        # monotonic sequence index.
        causal_keys = torch.arange(seq_len, device=position_ids.device) <= query_idx
        logits = logits.masked_fill(~causal_keys.view(1, 1, 1, -1), torch.finfo(logits.dtype).min)
        probabilities = F.softmax(logits.float(), dim=-1)
        scores = probabilities[0, :, 0, patch_indices].mean(dim=0)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Group-wise query-attention scores contain NaN or Inf.")
        return scores

    def _select_high_mask(
        self,
        *,
        layer,
        full_hidden: torch.Tensor,
        full_positions: torch.Tensor,
        patch_mask: torch.Tensor,
        text_mask: torch.Tensor,
        keep_ratio: float,
        anchor_number: int,
        full_position_span: int,
    ) -> torch.Tensor:
        patch_indices = torch.nonzero(patch_mask, as_tuple=False).flatten()
        keep_count = min(
            int(patch_indices.numel()),
            max(1, int(math.ceil(int(patch_indices.numel()) * float(keep_ratio)))),
        )
        if self.group_config.score_mode == "query_attention":
            scores = self._query_attention_scores(
                layer=layer,
                hidden_states=full_hidden,
                position_ids=full_positions,
                patch_mask=patch_mask,
                text_mask=text_mask,
                full_position_span=full_position_span,
            )
            # Stable sorting makes equal-score behavior reproducible by original token order.
            selected_local = torch.argsort(scores, descending=True, stable=True)[:keep_count]
        elif self.group_config.score_mode == "projector_vtc_visionzip":
            selected_local = self._select_projector_vtc_visionzip(
                patch_indices=patch_indices,
                keep_count=keep_count,
            )
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed((self._sample_seed + anchor_number * 1_000_003) % (2**63 - 1))
            selected_local = torch.randperm(int(patch_indices.numel()), generator=generator)[:keep_count]
            selected_local = selected_local.to(device=patch_indices.device)
        selected = patch_indices[selected_local]
        high_mask = torch.zeros_like(patch_mask)
        high_mask[selected] = True
        return high_mask

    @staticmethod
    def _resolve_keep_count(patch_count: int, keep_ratio: float) -> int:
        """Return the exact cardinality produced by the Top-K route."""
        return min(
            int(patch_count),
            max(1, int(math.ceil(int(patch_count) * float(keep_ratio)))),
        )

    def _fixed_mask_cache_key(
        self,
        *,
        patch_mask: torch.Tensor,
    ) -> str:
        """Build a reproducible key from sample identity and mask protocol."""
        hasher = hashlib.blake2b(digest_size=20)
        hasher.update(b"fixed-mask-v2\0")
        hasher.update(str(self._generation_state["sample_cache_id"]).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(self._generation_state["input_ids_digest"]).encode("ascii"))
        protocol = (
            int(self.group_config.late_entry_layer),
            int(self.group_config.early_exit_layer),
            tuple(self.group_config.anchor_layers),
            tuple(float(value) for value in self.group_config.keep_ratios),
            int(self.group_config.recovery_layers),
            str(self.group_config.score_mode),
            int(patch_mask.numel()),
            int(patch_mask.sum().item()),
        )
        hasher.update(repr(protocol).encode("ascii"))
        return hasher.hexdigest()

    def _mask_cache_path(self, cache_key: str) -> Path:
        if self._mask_cache_dir is None:
            raise RuntimeError("Fixed-mask cache directory is not configured.")
        return self._mask_cache_dir / cache_key[:2] / f"{cache_key}.pt"

    def _load_fixed_masks(
        self,
        *,
        cache_key: str,
        patch_mask: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        cache_path = self._mask_cache_path(cache_key)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing fixed teacher mask for cache key {cache_key}: {cache_path}"
            )
        payload = torch.load(cache_path, map_location="cpu")
        expected_protocol = {
            "late_entry_layer": int(self.group_config.late_entry_layer),
            "early_exit_layer": int(self.group_config.early_exit_layer),
            "anchor_layers": list(self.group_config.anchor_layers),
            "keep_ratios": [float(value) for value in self.group_config.keep_ratios],
            "recovery_layers": int(self.group_config.recovery_layers),
            "score_mode": str(self.group_config.score_mode),
            "sequence_length": int(patch_mask.numel()),
            "patch_tokens": int(patch_mask.sum().item()),
        }
        if not isinstance(payload, dict) or payload.get("version") != 2:
            raise ValueError(f"Invalid fixed-mask cache payload: {cache_path}")
        if payload.get("sample_cache_id") != self._generation_state.get("sample_cache_id"):
            raise ValueError(f"Fixed-mask sample identity mismatch: {cache_path}")
        if payload.get("input_ids_digest") != self._generation_state.get("input_ids_digest"):
            raise ValueError(f"Fixed-mask prompt digest mismatch: {cache_path}")
        for field, expected in expected_protocol.items():
            if payload.get(field) != expected:
                raise ValueError(
                    f"Fixed-mask cache protocol mismatch for {field}: "
                    f"cached={payload.get(field)!r}, expected={expected!r}, path={cache_path}"
                )
        raw_masks = payload.get("high_masks")
        if not isinstance(raw_masks, dict):
            raise ValueError(f"Fixed-mask cache has no high_masks mapping: {cache_path}")
        loaded = {}
        for anchor, keep_ratio in zip(
            self.group_config.anchor_layers, self.group_config.keep_ratios
        ):
            mask = raw_masks.get(str(anchor))
            if not torch.is_tensor(mask) or mask.dtype != torch.bool:
                raise ValueError(f"Missing bool mask for anchor {anchor}: {cache_path}")
            mask = mask.flatten()
            if mask.numel() != patch_mask.numel() or bool((mask & ~patch_mask.cpu()).any()):
                raise ValueError(f"Fixed mask is not aligned to patch tokens at anchor {anchor}: {cache_path}")
            expected_keep = min(
                int(patch_mask.sum().item()),
                max(1, int(math.ceil(int(patch_mask.sum().item()) * float(keep_ratio)))),
            )
            if int(mask.sum().item()) != expected_keep:
                raise ValueError(
                    f"Fixed mask keep count mismatch at anchor {anchor}: "
                    f"cached={int(mask.sum().item())}, expected={expected_keep}, path={cache_path}"
                )
            loaded[int(anchor)] = mask.to(device=patch_mask.device)
        return loaded

    def _write_fixed_masks(
        self,
        *,
        cache_key: str,
        patch_mask: torch.Tensor,
        high_by_anchor: Dict[int, torch.Tensor],
    ) -> None:
        cache_path = self._mask_cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "cache_key": cache_key,
            "sample_cache_id": str(self._generation_state["sample_cache_id"]),
            "input_ids_digest": str(self._generation_state["input_ids_digest"]),
            "mask_source": "frozen_teacher_dynamic_gws",
            "late_entry_layer": int(self.group_config.late_entry_layer),
            "early_exit_layer": int(self.group_config.early_exit_layer),
            "anchor_layers": list(self.group_config.anchor_layers),
            "keep_ratios": [float(value) for value in self.group_config.keep_ratios],
            "recovery_layers": int(self.group_config.recovery_layers),
            "score_mode": str(self.group_config.score_mode),
            "sequence_length": int(patch_mask.numel()),
            "patch_tokens": int(patch_mask.sum().item()),
            "high_masks": {
                str(anchor): high_by_anchor[int(anchor)].detach().to(device="cpu", dtype=torch.bool)
                for anchor in self.group_config.anchor_layers
            },
        }
        temporary = cache_path.with_suffix(f".tmp.{os.getpid()}.{torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}")
        torch.save(payload, temporary)
        os.replace(temporary, cache_path)

    def _prefill(
        self,
        *,
        causal_lm,
        inputs_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask_2d: Optional[torch.Tensor],
        past_key_values,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        labels: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        is_training = bool(causal_lm.training or labels is not None)
        if not is_training and not use_cache:
            raise ValueError("group_wise_skip_recovery generation requires use_cache=True.")
        state = self._generation_state
        backbone = causal_lm.model
        num_layers = len(backbone.layers)
        entry_idx = int(self.group_config.late_entry_layer) - 1
        exit_idx = int(self.group_config.early_exit_layer) - 1
        anchors = tuple(self.group_config.anchor_layers)
        ratios = dict(zip(anchors, self.group_config.keep_ratios))
        if not 0 <= entry_idx < exit_idx <= num_layers:
            raise ValueError(
                "Expected early_exit_layer <= num_layers + 1, got "
                f"{self.group_config.early_exit_layer} for {num_layers} layers."
            )
        if inputs_embeds.shape[0] != 1 or inputs_embeds.shape[1] != state["active_mask"].numel():
            raise ValueError("Expanded embeddings do not match the registered prompt layout.")

        active_mask = state["active_mask"].to(inputs_embeds.device)
        full_hidden = inputs_embeds[:, active_mask, :]
        # Keep the expanded multimodal prompt length separate from the compact
        # text-only tensor used after early exit.  The former is the input
        # sequence length reported to efficiency accounting; the latter is
        # only the final decoder state shape.
        full_prompt_length = int(full_hidden.shape[1])
        visual_mask = state["visual_mask"].to(inputs_embeds.device)[active_mask]
        text_mask = ~visual_mask
        active_labels = None
        score_text_mask = text_mask
        if labels is not None:
            if labels.shape != (1, active_mask.numel()):
                raise ValueError(
                    "Training labels must match the expanded prompt layout, "
                    f"got {tuple(labels.shape)} for {(1, active_mask.numel())}."
                )
            active_labels = labels[:, active_mask.to(labels.device)]
            labels_on_device = active_labels[0].to(inputs_embeds.device)
            if state.get("query_mode", "text_generation") == "grounding":
                ground_token_ids = state.get("ground_token_ids")
                if not ground_token_ids:
                    raise ValueError("Grounding group-wise compression has no ground token ids.")
                ground_query_mask = torch.zeros_like(text_mask)
                for token_id in ground_token_ids:
                    ground_query_mask |= text_mask & (labels_on_device == int(token_id))
                if not bool(ground_query_mask.any()):
                    raise ValueError("Cannot locate the grounding <ground> query token in labels.")
                # predict_box consumes the hidden state at <ground>. Use that
                # token as the semantic query for visual-token selection.
                score_text_mask = ground_query_mask
            else:
                ignored_indices = torch.nonzero(labels_on_device == -100, as_tuple=False).flatten()
                if ignored_indices.numel() == 0:
                    raise ValueError("Cannot locate the masked user prompt in training labels.")
                # preprocess_qwen leaves five labeled template tokens after the
                # masked user content: user <|im_end|>, newline, <|im_start|>,
                # assistant, newline. Generation prefill ends at the fifth token;
                # answer tokens follow it only under teacher forcing.
                query_idx = int(ignored_indices[-1].item()) + 5
                if query_idx >= text_mask.numel() or not bool(text_mask[query_idx]):
                    raise ValueError("Cannot locate the Qwen assistant-generation prompt in training labels.")
                response_indices = torch.nonzero(
                    (labels_on_device != -100)
                    & text_mask
                    & (torch.arange(text_mask.numel(), device=text_mask.device) > query_idx),
                    as_tuple=False,
                ).flatten()
                if response_indices.numel() == 0:
                    raise ValueError("Group-wise training requires a response after the generation prompt.")
                # Match generation prefill and prevent teacher-forced answer leakage.
                score_text_mask = torch.zeros_like(text_mask)
                score_text_mask[query_idx] = True
        newline = getattr(backbone, "image_newline", None)
        if newline is None:
            raise ValueError("group_wise_skip_recovery requires LLaVA-OV image_newline.")
        visual_embeddings = full_hidden[0, visual_mask]
        newline = newline.detach().to(device=full_hidden.device, dtype=full_hidden.dtype).view(1, -1)
        if newline.shape[-1] != full_hidden.shape[-1]:
            raise ValueError("image_newline hidden size does not match prompt embeddings.")
        visual_format_local = torch.all(visual_embeddings == newline, dim=-1)
        format_mask = torch.zeros_like(visual_mask)
        format_mask[visual_mask] = visual_format_local
        patch_mask = visual_mask & ~format_mask
        if not bool(patch_mask.any()):
            raise ValueError("No visual patch token remains after excluding format tokens.")
        # This is the only patch-cardinality synchronization needed for the
        # whole prefill.  Every later skipped-layer count is determined by the
        # same keep-ratio rule and does not need another GPU -> CPU sync.
        patch_token_count = int(patch_mask.sum().item())
        num_format_tokens = int(format_mask.sum().item())
        mask_cache_key = None
        cached_high_by_anchor: Dict[int, torch.Tensor] = {}
        if self.group_config.mask_cache_mode != "dynamic":
            mask_cache_key = self._fixed_mask_cache_key(
                patch_mask=patch_mask,
            )
        if self.group_config.mask_cache_mode == "read":
            cached_high_by_anchor = self._load_fixed_masks(
                cache_key=mask_cache_key,
                patch_mask=patch_mask,
            )

        projector_profile = causal_lm.get_last_compression_profile() if hasattr(causal_lm, "get_last_compression_profile") else {}
        samples = projector_profile.get("samples", []) if isinstance(projector_profile, dict) else []
        if len(samples) == 1 and isinstance(samples[0], dict):
            expected = samples[0].get("visual_patch_tokens")
            if isinstance(expected, (int, float)) and int(expected) != patch_token_count:
                raise ValueError(
                    "Patch/format split disagrees with projector profile: "
                    f"observed={patch_token_count}, expected={int(expected)}."
                )

        expects_3d = self._expects_3d_position_ids(backbone)
        full_positions = self._normalize_position_ids(
            position_ids,
            sequence_length=active_mask.numel(),
            device=full_hidden.device,
            expects_3d=expects_3d,
        )[:, active_mask]
        # Position IDs encode spatial coordinates for visual tokens in Video3D;
        # they are intentionally not required to be monotonic or unique.
        if not bool(torch.isfinite(full_positions.float()).all()):
            raise ValueError("Video3D position_ids contain NaN or Inf.")
        full_position_span = int(full_positions.max().item()) + 1

        original_visual = full_hidden[:, visual_mask, :].clone()
        hidden_states = full_hidden[:, text_mask, :]
        current_positions = full_positions[:, text_mask]
        if use_cache:
            cache = (
                DynamicCache.from_legacy_cache(past_key_values)
                if not isinstance(past_key_values, Cache)
                else past_key_values
            )
        else:
            cache = None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        high_by_anchor: Dict[int, torch.Tensor] = {}
        high_counts = {
            int(anchor): self._resolve_keep_count(patch_token_count, ratios[anchor])
            for anchor in anchors
        }
        per_layer_lengths = []
        active_patch_counts = []
        patch_token_layer_total = 0
        visual_token_layer_total = 0
        self._debug_trace = []

        for layer_idx, layer in enumerate(backbone.layers):
            layer_number = layer_idx + 1
            if layer_idx == entry_idx:
                restored = torch.empty_like(full_hidden)
                restored[:, text_mask, :] = hidden_states
                restored[:, visual_mask, :] = original_visual
                full_hidden = restored
            if layer_idx == exit_idx:
                hidden_states = full_hidden[:, text_mask, :]
                current_positions = full_positions[:, text_mask]

            in_visual_window = entry_idx <= layer_idx < exit_idx
            skip_anchor = self._skip_anchor_for_layer(
                layer_number,
                anchors,
                int(self.group_config.early_exit_layer),
                int(self.group_config.recovery_layers),
            ) if in_visual_window else None
            if in_visual_window:
                execute_mask = torch.ones_like(text_mask) if skip_anchor is None else (
                    text_mask | format_mask | high_by_anchor[skip_anchor]
                )
                hidden_states = full_hidden[:, execute_mask, :]
                current_positions = full_positions[:, execute_mask]
                current_visual_mask = visual_mask[execute_mask]
            else:
                execute_mask = text_mask
                current_visual_mask = None

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            per_layer_lengths.append(int(hidden_states.shape[1]))
            if not in_visual_window:
                patch_count = 0
            elif skip_anchor is None:
                patch_count = patch_token_count
            else:
                patch_count = high_counts[skip_anchor]
            active_patch_counts.append(patch_count)
            patch_token_layer_total += patch_count
            if in_visual_window:
                visual_token_layer_total += patch_count + num_format_tokens

            pending_high = None
            if layer_number in ratios:
                # Top-k routing is discrete. Avoid retaining a second graph for
                # score computation; selected hidden states remain attached to
                # the normal decoder graph below.
                if self.group_config.mask_cache_mode == "read":
                    pending_high = cached_high_by_anchor[layer_number]
                else:
                    with torch.no_grad():
                        pending_high = self._select_high_mask(
                            layer=layer,
                            full_hidden=full_hidden,
                            full_positions=full_positions,
                            patch_mask=patch_mask,
                            text_mask=score_text_mask,
                            keep_ratio=ratios[layer_number],
                            anchor_number=layer_number,
                            full_position_span=full_position_span,
                        )

            layer_attention_2d = torch.ones(
                1, hidden_states.shape[1], device=hidden_states.device, dtype=torch.bool
            )
            layer_mask = self._prepare_mask(
                backbone, layer_attention_2d, hidden_states, 0, output_attentions, use_cache
            )
            layer_args = (
                layer,
                hidden_states,
                layer_mask,
                current_positions,
                cache,
                output_attentions,
                use_cache,
                full_position_span,
            )
            if backbone.gradient_checkpointing and backbone.training:
                layer_outputs = backbone._gradient_checkpointing_func(self._layer_forward, *layer_args)
            else:
                layer_outputs = self._layer_forward(*layer_args)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if in_visual_window:
                if not is_training and skip_anchor is None:
                    # Anchor/recovery layers execute the complete visual
                    # sequence. Their output already has the full sequence
                    # shape, so avoid cloning and scattering it back.
                    full_hidden = hidden_states
                else:
                    updated_full = full_hidden.clone()
                    updated_full[:, execute_mask, :] = hidden_states
                    full_hidden = updated_full
            if pending_high is not None:
                high_by_anchor[layer_number] = pending_high

            if self.group_config.debug_trace:
                self._debug_trace.append(
                    {
                        "layer": layer_number,
                        "skip_anchor": skip_anchor,
                        "execute_mask": execute_mask.detach().cpu(),
                        "patch_count": patch_count,
                        "high_mask": None if pending_high is None else pending_high.detach().cpu(),
                        "full_hidden": full_hidden.detach().cpu(),
                    }
                )

        if exit_idx == num_layers:
            hidden_states = full_hidden
        hidden_states = backbone.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        patch_tokens = patch_token_count
        format_tokens = num_format_tokens
        visual_tokens = int(visual_mask.sum().item())
        text_tokens = int(text_mask.sum().item())
        final_visual_tokens = visual_tokens if exit_idx == num_layers else 0
        average_visual_tokens = float(visual_token_layer_total) / float(num_layers)
        group_counts = {
            str(anchor): {
                "high": high_counts[anchor],
                "low": patch_tokens - high_counts[anchor],
            }
            for anchor in anchors
        }
        if self.group_config.mask_cache_mode == "write":
            self._write_fixed_masks(
                cache_key=mask_cache_key,
                patch_mask=patch_mask,
                high_by_anchor=high_by_anchor,
            )
        self._last_profile = {
            "compressor_name": "group_wise_skip_recovery",
            "position_id_strategy": "preserve_full_sequence",
            "late_entry_layer": int(self.group_config.late_entry_layer),
            "early_exit_layer": int(self.group_config.early_exit_layer),
            "anchor_layers": list(anchors),
            "keep_ratios": list(self.group_config.keep_ratios),
            "recovery_layers": int(self.group_config.recovery_layers),
            "score_mode": self.group_config.score_mode,
            "query_mode": state.get("query_mode", "text_generation"),
            "mask_cache_mode": self.group_config.mask_cache_mode,
            "mask_source": (
                "teacher_cache" if self.group_config.mask_cache_mode == "read"
                else "frozen_teacher_dynamic_gws" if self.group_config.mask_cache_mode == "write"
                else "current_model_dynamic_gws"
            ),
            "mask_cache_key": mask_cache_key,
            "group_counts": group_counts,
            "visual_patch_tokens": patch_tokens,
            "visual_format_tokens": format_tokens,
            "visual_sequence_tokens": visual_tokens,
            "text_prompt_tokens": text_tokens,
            "prompt_sequence_length_before_prune": full_prompt_length,
            "prompt_sequence_length": full_prompt_length,
            "final_sequence_length": int(hidden_states.shape[1]),
            "prefill_layer_token_lengths": per_layer_lengths,
            "prefill_layer_active_patch_tokens": active_patch_counts,
            "prefill_final_visual_tokens": float(final_visual_tokens),
            "patch_token_layer_total": patch_token_layer_total,
            "visual_token_layer_total": visual_token_layer_total,
            "compressor_input_tokens": visual_tokens,
            "compressor_output_tokens": int(round(average_visual_tokens)),
            "llm_stage_input_tokens": visual_tokens,
            "llm_stage_output_tokens": int(round(average_visual_tokens)),
            "llm_stage_keep_ratio": float(visual_token_layer_total) / float(max(1, visual_tokens * num_layers)),
            "token_keep_ratio": float(visual_token_layer_total) / float(max(1, visual_tokens * num_layers)),
        }
        self._update_stats(
            input_tokens=visual_tokens * num_layers,
            output_tokens=visual_token_layer_total,
            late_entry_layer=int(self.group_config.late_entry_layer),
            early_exit_layer=int(self.group_config.early_exit_layer),
        )
        # Video3D's LMMS adapter merges projector and LLM statistics by reading
        # this backbone-owned slot.  The direct compressor path otherwise only
        # retains the profile inside the compressor, which is cleared when
        # generation finishes and makes a successful group-wise run appear as
        # projector-only in the result JSON.
        backbone._last_llm_compression_profile = {"samples": [self._last_profile.copy()]}
        if active_labels is not None:
            self._last_training_labels = (
                active_labels[:, text_mask.to(labels.device)]
                if exit_idx < num_layers
                else active_labels
            )
        state["phase"] = "decode"
        state["next_decode_position"] = self._next_decode_position(
            backbone=backbone,
            full_positions=full_positions,
            state=state,
        )
        next_cache = next_decoder_cache.to_legacy_cache() if use_cache else None
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
