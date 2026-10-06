import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import BaseCompressor, CompressorConfig, CompressorOutput
from ..registry import register_compressor


_NEWLINE_STRATEGIES = {"grid_drop", "full_grid_drop", "frame_newline", "one_token", "no_token"}


@dataclass
class SegPrunerProjectorConfig(CompressorConfig):
    visual_token_num: Optional[int] = None
    token_keep_ratio: Optional[float] = None
    important_ratio: float = 0.0
    lam: float = 0.5
    newline_strategy: str = "grid_drop"


@dataclass
class SpatialKCenterMergeConfig(CompressorConfig):
    target_tokens: Optional[int] = None
    target_keep_ratio: Optional[float] = None
    important_ratio: float = 0.35
    merge_weight: float = 0.35
    score_weight: float = 0.35
    newline_strategy: str = "grid_drop"


class _SpatialSelectionMixin:
    @staticmethod
    def _validate_target_spec(target_tokens: Optional[int], target_keep_ratio: Optional[float]) -> None:
        if (target_tokens is None) == (target_keep_ratio is None):
            raise ValueError("Exactly one of target_tokens and target_keep_ratio must be provided.")
        if target_tokens is not None and int(target_tokens) <= 0:
            raise ValueError(f"target_tokens must be positive, got {target_tokens}.")
        if target_keep_ratio is not None and not 0 < float(target_keep_ratio) <= 1:
            raise ValueError(f"target_keep_ratio must be within (0, 1], got {target_keep_ratio}.")

    @staticmethod
    def _resolve_target_tokens(
        input_tokens: int,
        target_tokens: Optional[int],
        target_keep_ratio: Optional[float],
    ) -> int:
        if target_tokens is not None:
            return min(max(1, int(target_tokens)), input_tokens)
        return min(max(1, int(round(input_tokens * float(target_keep_ratio)))), input_tokens)

    @staticmethod
    def _normalize_features(features: torch.Tensor, num_frames: int, tokens_per_frame: int) -> torch.Tensor:
        if features.dim() == 3 and features.shape[:2] == (num_frames, tokens_per_frame):
            return features
        if features.dim() == 3 and features.shape[:2] == (1, num_frames * tokens_per_frame):
            return features.view(num_frames, tokens_per_frame, features.shape[-1])
        raise ValueError(
            f"Unexpected feature layout {tuple(features.shape)} for frames={num_frames}, "
            f"tokens={tokens_per_frame}."
        )

    @staticmethod
    def _normalize_attention(
        attn_weights: Optional[torch.Tensor],
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attn_weights is None:
            raise ValueError("Spatial selection requires visual attention scores.")
        if attn_weights.dim() == 3 and attn_weights.shape[0] == 1:
            attn_weights = attn_weights[0]
        if attn_weights.dim() == 2 and attn_weights.shape == (num_frames, tokens_per_frame):
            return attn_weights.to(device=device, dtype=torch.float32)
        if attn_weights.dim() == 2 and attn_weights.shape == (1, num_frames * tokens_per_frame):
            return attn_weights.view(num_frames, tokens_per_frame).to(device=device, dtype=torch.float32)
        raise ValueError(
            f"Unexpected attention layout {tuple(attn_weights.shape)} for frames={num_frames}, "
            f"tokens={tokens_per_frame}."
        )

    @staticmethod
    def _normalize_coordinates(
        coordinates: Optional[torch.Tensor],
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if coordinates is None:
            raise ValueError("Spatial selection requires patch-level 3D coordinates.")
        if coordinates.dim() >= 4 and coordinates.shape[:2] == (1, num_frames):
            coordinates = coordinates[0]
        elif coordinates.dim() >= 3 and coordinates.shape[0] == num_frames:
            pass
        elif coordinates.dim() >= 3 and coordinates.shape[:2] == (1, num_frames * tokens_per_frame):
            coordinates = coordinates.view(num_frames, tokens_per_frame, *coordinates.shape[2:])
        else:
            raise ValueError(
                f"Unexpected coordinate layout {tuple(coordinates.shape)} for frames={num_frames}, "
                f"tokens={tokens_per_frame}."
            )
        if coordinates.shape[:2] != (num_frames, tokens_per_frame):
            raise ValueError(
                f"Coordinate layout {tuple(coordinates.shape)} does not match frames={num_frames}, "
                f"tokens={tokens_per_frame}."
            )
        return coordinates.to(device=device)

    @staticmethod
    def _flatten_video_state(
        features: torch.Tensor,
        coordinates: torch.Tensor,
        grouping_coordinates: torch.Tensor,
        attentions: torch.Tensor,
        frame_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        num_frames, tokens_per_frame, hidden_size = features.shape
        height, width = frame_shape
        rows, cols = torch.meshgrid(
            torch.arange(height, device=features.device, dtype=torch.float32),
            torch.arange(width, device=features.device, dtype=torch.float32),
            indexing="ij",
        )
        patch_positions = torch.stack((rows, cols), dim=-1).view(1, tokens_per_frame, 2)
        patch_positions = patch_positions.repeat(num_frames, 1, 1)
        return {
            "features": features.reshape(num_frames * tokens_per_frame, hidden_size),
            "coordinates": coordinates.reshape(num_frames * tokens_per_frame, coordinates.shape[-1]),
            "grouping_coordinates": grouping_coordinates.reshape(
                num_frames * tokens_per_frame,
                grouping_coordinates.shape[-1],
            ),
            "attentions": attentions.reshape(num_frames * tokens_per_frame),
            "patch_positions": patch_positions.reshape(num_frames * tokens_per_frame, 2),
            "frame_ids": torch.arange(num_frames, device=features.device).repeat_interleave(tokens_per_frame),
            "order_ids": torch.arange(num_frames * tokens_per_frame, device=features.device),
        }

    def _pack_selected(
        self,
        *,
        state: Dict[str, torch.Tensor],
        selected_indices: torch.Tensor,
        output_features: Optional[torch.Tensor],
        num_frames: int,
        input_tokens: int,
        newline_strategy: str,
        extra_metadata: Dict[str, Any],
    ) -> CompressorOutput:
        selected_indices = selected_indices.to(device=state["features"].device, dtype=torch.long)
        if output_features is None:
            output_features = state["features"].index_select(0, selected_indices)
        frame_ids = state["frame_ids"].index_select(0, selected_indices)
        order_ids = state["order_ids"].index_select(0, selected_indices)
        order = order_ids.argsort(stable=True)
        output_features = output_features.index_select(0, order)
        # Selection and merge scores are computed in FP32, but embeddings must
        # retain the projector dtype expected by the language model.
        output_features = output_features.to(
            device=state["features"].device,
            dtype=state["features"].dtype,
        )
        selected_indices = selected_indices.index_select(0, order)
        frame_ids = frame_ids.index_select(0, order)
        order_ids = order_ids.index_select(0, order)
        frame_token_counts = [int(value) for value in torch.bincount(frame_ids, minlength=num_frames).tolist()]
        output_tokens = int(output_features.shape[0])

        self._update_stats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            frame_token_counts=frame_token_counts,
            **{key: value for key, value in extra_metadata.items() if isinstance(value, (int, float, bool, str))},
        )
        return CompressorOutput(
            features=output_features.unsqueeze(0),
            compression_ratio=float(output_tokens) / float(input_tokens) if input_tokens else 1.0,
            metadata={
                "newline_strategy": newline_strategy,
                "frame_token_counts": frame_token_counts,
                "compressed_coordinates": state["coordinates"].index_select(0, selected_indices).unsqueeze(0),
                "compressed_patch_positions": state["patch_positions"].index_select(0, selected_indices).unsqueeze(0),
                "retained_order_ids": [int(value) for value in order_ids.detach().cpu().tolist()],
                **extra_metadata,
            },
        )

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required.update(attn_weights=True, raw_features_before_proj=True, coordinates=True)
        return required

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False


@register_compressor("segpruner")
class SegPrunerProjectorCompressor(_SpatialSelectionMixin, BaseCompressor):
    """Attention-important plus geometry/semantic-diverse token selection.

    ``visual_token_num`` and ``token_keep_ratio`` are interpreted per frame,
    matching the reference image-level SegPruner budget.
    """

    def __init__(self, config: Dict[str, Any]):
        seg_config = SegPrunerProjectorConfig(**config) if isinstance(config, dict) else config
        self._validate_target_spec(seg_config.visual_token_num, seg_config.token_keep_ratio)
        if not 0 <= float(seg_config.important_ratio) <= 1:
            raise ValueError(f"important_ratio must be within [0, 1], got {seg_config.important_ratio}.")
        if not 0 <= float(seg_config.lam) <= 1:
            raise ValueError(f"lam must be within [0, 1], got {seg_config.lam}.")
        if seg_config.newline_strategy not in _NEWLINE_STRATEGIES:
            raise ValueError(f"Unexpected newline_strategy: {seg_config.newline_strategy}.")
        super().__init__(seg_config)
        self.seg_config = seg_config

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        grouping_coordinates: Optional[torch.Tensor] = None,
        raw_features_before_proj: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        if num_frames is None or frame_shape is None:
            raise ValueError("segpruner requires num_frames and frame_shape.")
        tokens_per_frame = int(frame_shape[0] * frame_shape[1])
        candidate_features = self._normalize_features(features, num_frames, tokens_per_frame)
        semantic_features = self._normalize_features(
            raw_features_before_proj if raw_features_before_proj is not None else features,
            num_frames,
            tokens_per_frame,
        ).to(device=candidate_features.device)
        attentions = self._normalize_attention(attn_weights, num_frames, tokens_per_frame, candidate_features.device)
        coords = self._normalize_coordinates(coordinates, num_frames, tokens_per_frame, candidate_features.device)
        grouping_coords = self._normalize_coordinates(
            grouping_coordinates if grouping_coordinates is not None else coordinates,
            num_frames,
            tokens_per_frame,
            candidate_features.device,
        )
        state = self._flatten_video_state(candidate_features, coords, grouping_coords, attentions, frame_shape)
        semantic = semantic_features.reshape(num_frames * tokens_per_frame, semantic_features.shape[-1])

        input_tokens = int(state["features"].shape[0])

        # SegPruner's visual_token_num is a per-image budget in the reference
        # implementation.  Apply a keep ratio to each frame independently as
        # well, rather than letting high-attention frames consume another
        # frame's budget.
        frame_target = self._resolve_target_tokens(
            tokens_per_frame,
            self.seg_config.visual_token_num,
            self.seg_config.token_keep_ratio,
        )
        frame_important = min(
            frame_target,
            int(frame_target * float(self.seg_config.important_ratio)),
        )
        frame_diverse = frame_target - frame_important
        normalized_semantic = F.normalize(semantic.float(), dim=-1) if frame_diverse > 0 else None
        lam = float(self.seg_config.lam)

        # The reference implementation vectorizes the independent per-frame
        # FPS loops over the frame dimension.  This preserves frame isolation
        # while avoiding one Python/kernel loop for every video frame.
        frame_target_tokens = [int(frame_target)] * num_frames
        frame_important_tokens = [int(frame_important)] * num_frames
        attention_order = attentions.argsort(dim=-1, descending=True, stable=True)
        important_indices = attention_order[:, :frame_important]
        residual_indices = attention_order[:, frame_important:]
        residual_count = int(residual_indices.shape[1])

        if frame_diverse > 0 and residual_count > 0:
            semantic_frames = normalized_semantic.reshape(
                num_frames, tokens_per_frame, normalized_semantic.shape[-1]
            )
            grouping_frames = state["grouping_coordinates"].reshape(
                num_frames,
                tokens_per_frame,
                state["grouping_coordinates"].shape[-1],
            )
            coordinate_index = residual_indices.unsqueeze(-1).expand(
                -1, -1, grouping_frames.shape[-1]
            )
            residual_coords = torch.gather(grouping_frames, 1, coordinate_index)
            semantic_index = residual_indices.unsqueeze(-1).expand(
                -1, -1, semantic_frames.shape[-1]
            )
            residual_semantic = torch.gather(semantic_frames, 1, semantic_index)

            selection_count = min(frame_diverse, residual_count)
            centroids = torch.empty(
                (num_frames, selection_count),
                device=residual_indices.device,
                dtype=torch.long,
            )
            available = torch.ones(
                (num_frames, residual_count),
                device=residual_indices.device,
                dtype=torch.bool,
            )
            farthest = torch.zeros(
                num_frames,
                device=residual_indices.device,
                dtype=torch.long,
            )
            min_fused = torch.full(
                (num_frames, residual_count),
                float("inf"),
                device=residual_indices.device,
            )
            batch_indices = torch.arange(
                num_frames,
                device=residual_indices.device,
                dtype=torch.long,
            )
            # dx is a separate scale for each frame, computed at the first
            # centroid and reused for all later FPS iterations.
            dx = None
            for iteration in range(selection_count):
                centroids[:, iteration] = farthest
                available[batch_indices, farthest] = False
                centroid_coords = residual_coords[batch_indices, farthest].unsqueeze(1)
                distance = torch.cdist(
                    residual_coords.float(), centroid_coords.float()
                ).squeeze(-1)
                if iteration == 0:
                    dx = distance.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
                semantic_scores = torch.bmm(
                    residual_semantic[batch_indices, farthest].unsqueeze(1),
                    residual_semantic.transpose(1, 2),
                ).squeeze(1)
                fused = lam * (distance / dx) + (1.0 - lam) * (1.0 - semantic_scores)
                min_fused = torch.minimum(min_fused, fused)
                if iteration + 1 < selection_count:
                    farthest = min_fused.masked_fill(~available, -float("inf")).argmax(dim=-1)
            diverse_indices = residual_indices.gather(1, centroids)
            frame_diverse_tokens = [int(selection_count)] * num_frames
            frame_dx_values = dx.squeeze(-1)
        else:
            diverse_indices = residual_indices[:, :0]
            frame_diverse_tokens = [0] * num_frames
            frame_dx_values = torch.full(
                (num_frames,),
                float("nan"),
                device=state["features"].device,
                dtype=torch.float32,
            )

        frame_dx = [
            None if math.isnan(value) else float(value)
            for value in frame_dx_values.detach().cpu().tolist()
        ]
        frame_offsets = (
            torch.arange(num_frames, device=state["features"].device, dtype=torch.long)
            * tokens_per_frame
        ).unsqueeze(1)
        selected_indices = torch.cat((important_indices, diverse_indices), dim=1)
        selected_indices = (selected_indices + frame_offsets).reshape(-1)
        total_target_tokens = int(sum(frame_target_tokens))
        important_count = int(sum(frame_important_tokens))
        diverse_count = int(sum(frame_diverse_tokens))

        return self._pack_selected(
            state=state,
            selected_indices=selected_indices[:total_target_tokens],
            output_features=None,
            num_frames=num_frames,
            input_tokens=input_tokens,
            newline_strategy=self.seg_config.newline_strategy,
            extra_metadata={
                "selection_method": "segpruner_projector_attention_geometry",
                # Keep visual_token_num as the per-frame value for parity
                # with the original implementation; total_target_tokens is
                # provided for callers that need the whole-video count.
                "visual_token_num": int(frame_target),
                "per_frame_visual_token_num": int(frame_target),
                "target_tokens": total_target_tokens,
                "total_target_tokens": total_target_tokens,
                "important_ratio": float(self.seg_config.important_ratio),
                "lam": float(self.seg_config.lam),
                "important_tokens": important_count,
                "diverse_tokens": diverse_count,
                "frame_target_tokens": frame_target_tokens,
                "frame_important_tokens": frame_important_tokens,
                "frame_diverse_tokens": frame_diverse_tokens,
                "frame_dx": frame_dx,
                "budget_scope": "per_frame",
                "fps_scope": "per_frame",
                "dx_scope": "per_frame_first_iteration",
            },
        )


@register_compressor("spatial_kcenter_merge")
class SpatialKCenterMergeCompressor(_SpatialSelectionMixin, BaseCompressor):
    """Attention seeds plus 3D K-center coverage and semantic feature merging."""

    def __init__(self, config: Dict[str, Any]):
        kcenter_config = SpatialKCenterMergeConfig(**config) if isinstance(config, dict) else config
        self._validate_target_spec(kcenter_config.target_tokens, kcenter_config.target_keep_ratio)
        for name in ("important_ratio", "merge_weight", "score_weight"):
            value = float(getattr(kcenter_config, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1], got {value}.")
        if kcenter_config.newline_strategy not in _NEWLINE_STRATEGIES:
            raise ValueError(f"Unexpected newline_strategy: {kcenter_config.newline_strategy}.")
        super().__init__(kcenter_config)
        self.kcenter_config = kcenter_config

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        grouping_coordinates: Optional[torch.Tensor] = None,
        raw_features_before_proj: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        if num_frames is None or frame_shape is None:
            raise ValueError("spatial_kcenter_merge requires num_frames and frame_shape.")
        tokens_per_frame = int(frame_shape[0] * frame_shape[1])
        candidate_features = self._normalize_features(features, num_frames, tokens_per_frame)
        semantic_features = self._normalize_features(
            raw_features_before_proj if raw_features_before_proj is not None else features,
            num_frames,
            tokens_per_frame,
        ).to(device=candidate_features.device)
        attentions = self._normalize_attention(attn_weights, num_frames, tokens_per_frame, candidate_features.device)
        coords = self._normalize_coordinates(coordinates, num_frames, tokens_per_frame, candidate_features.device)
        grouping_coords = self._normalize_coordinates(
            grouping_coordinates if grouping_coordinates is not None else coordinates,
            num_frames,
            tokens_per_frame,
            candidate_features.device,
        )
        state = self._flatten_video_state(candidate_features, coords, grouping_coords, attentions, frame_shape)
        semantic = semantic_features.reshape(num_frames * tokens_per_frame, semantic_features.shape[-1])

        input_tokens = int(state["features"].shape[0])
        target_tokens = self._resolve_target_tokens(
            input_tokens,
            self.kcenter_config.target_tokens,
            self.kcenter_config.target_keep_ratio,
        )
        important_count = min(target_tokens, int(round(target_tokens * float(self.kcenter_config.important_ratio))))
        attention_order = state["attentions"].argsort(descending=True, stable=True)
        important_indices = attention_order[:important_count]
        residual_indices = attention_order[important_count:]
        selected = [idx for idx in important_indices]

        if not selected and residual_indices.numel() > 0:
            selected.append(residual_indices[0])
            residual_indices = residual_indices[1:]

        remaining_count = target_tokens - len(selected)
        if remaining_count > 0 and residual_indices.numel() > 0:
            scores = state["attentions"]
            scores = (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1e-6)
            candidate_indices = residual_indices
            candidate_coords = state["grouping_coordinates"].index_select(0, candidate_indices)
            selected_tensor = torch.stack(selected).to(device=candidate_indices.device, dtype=torch.long)
            selected_coords = state["grouping_coordinates"].index_select(0, selected_tensor)
            min_dist = torch.cdist(candidate_coords.float(), selected_coords.float()).min(dim=-1).values
            available = torch.ones(candidate_indices.numel(), device=candidate_indices.device, dtype=torch.bool)
            score_weight = float(self.kcenter_config.score_weight)
            for _ in range(min(remaining_count, int(candidate_indices.numel()))):
                dist_norm = min_dist / min_dist.max().clamp_min(1e-6)
                combined = (1.0 - score_weight) * dist_norm + score_weight * scores.index_select(0, candidate_indices)
                local_idx = combined.masked_fill(~available, -float("inf")).argmax()
                selected_idx = candidate_indices[local_idx]
                selected.append(selected_idx)
                available[local_idx] = False
                new_dist = torch.cdist(
                    candidate_coords.float(), state["grouping_coordinates"][selected_idx].view(1, -1).float()
                ).squeeze(-1)
                min_dist = torch.minimum(min_dist, new_dist)
                if not available.any():
                    break

        selected_indices = torch.stack(selected).to(device=state["features"].device, dtype=torch.long)
        selected_features = state["features"].index_select(0, selected_indices).clone()
        residual_mask = torch.ones(input_tokens, device=state["features"].device, dtype=torch.bool)
        residual_mask[selected_indices] = False
        merge_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()
        merge_weight = float(self.kcenter_config.merge_weight)
        if merge_indices.numel() > 0 and merge_weight > 0:
            normalized_semantic = F.normalize(semantic.float(), dim=-1)
            assignment = (
                normalized_semantic.index_select(0, merge_indices)
                @ normalized_semantic.index_select(0, selected_indices).transpose(0, 1)
            ).argmax(dim=-1)
            aggregated = torch.zeros_like(selected_features)
            counts = torch.zeros(selected_indices.numel(), device=selected_features.device, dtype=torch.float32)
            aggregated.index_add_(0, assignment, state["features"].index_select(0, merge_indices))
            counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float32))
            merged_mean = aggregated / counts.clamp_min(1.0).unsqueeze(-1).to(dtype=selected_features.dtype)
            has_merge = counts > 0
            selected_features[has_merge] = (
                (1.0 - merge_weight) * selected_features[has_merge] + merge_weight * merged_mean[has_merge]
            ).to(dtype=selected_features.dtype)

        return self._pack_selected(
            state=state,
            selected_indices=selected_indices,
            output_features=selected_features,
            num_frames=num_frames,
            input_tokens=input_tokens,
            newline_strategy=self.kcenter_config.newline_strategy,
            extra_metadata={
                "selection_method": "spatial_kcenter_semantic_merge",
                "target_tokens": target_tokens,
                "important_ratio": float(self.kcenter_config.important_ratio),
                "merge_weight": merge_weight,
                "score_weight": float(self.kcenter_config.score_weight),
                "important_tokens": important_count,
            },
        )
