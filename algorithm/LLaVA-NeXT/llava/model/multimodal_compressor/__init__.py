"""Adapter to reuse the shared Video3D projector-level compressors.

The compression algorithms live in Video3D-LLM and are loaded from the
shared framework so LLaVA-NeXT does not keep a divergent copy.
"""

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


_SHARED_PACKAGE_NAME = "_video3d_shared_multimodal_compressor"
_SHARED_ALLOWED_COMPRESSORS = {
    "group_wise_skip_recovery",
    "late_entry_early_exit",
    "segpruner",
    "vispruner",
    "visionzip",
    "voxel_dtc",
    "voxel_vtc",
    "voxel_vtc_visionzip",
}
_OV_LOCAL_COMPRESSORS = set()
_ALLOWED_COMPRESSORS = _SHARED_ALLOWED_COMPRESSORS | _OV_LOCAL_COMPRESSORS
_ALLOWED_SUBMODULE_SUFFIXES = {
    ".base",
    ".builder",
    ".group_wise_skip_recovery",
    ".group_wise_skip_recovery.compressor",
    ".late_entry_early_exit",
    ".late_entry_early_exit.compressor",
    ".registry",
    ".spatial_selection",
    ".spatial_selection.compressor",
    ".visionzip",
    ".visionzip.compressor",
    ".vispruner",
    ".vispruner.compressor",
    ".voxel_dtc",
    ".voxel_dtc.compressor",
    ".voxel_vtc_post",
    ".voxel_vtc_post.compressor",
}


def _candidate_package_dirs():
    explicit_package = os.environ.get("VIDEO3D_COMPRESSOR_PACKAGE")
    if explicit_package:
        yield Path(explicit_package)

    explicit_root = os.environ.get("VIDEO3D_COMP_ROOT")
    if explicit_root:
        yield Path(explicit_root) / "algorithm" / "Video-3D-LLM" / "llava" / "model" / "multimodal_compressor"

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        yield parent / "algorithm" / "Video-3D-LLM" / "llava" / "model" / "multimodal_compressor"
        yield parent / "Video-3D-LLM" / "llava" / "model" / "multimodal_compressor"


def _load_shared_package():
    if _SHARED_PACKAGE_NAME in sys.modules:
        return sys.modules[_SHARED_PACKAGE_NAME]

    package_dir = None
    for candidate in _candidate_package_dirs():
        init_file = candidate / "__init__.py"
        if init_file.exists():
            package_dir = candidate
            break

    if package_dir is None:
        searched = ", ".join(str(path) for path in _candidate_package_dirs())
        raise ImportError(
            "Cannot find Video3D shared multimodal_compressor package. "
            f"Set VIDEO3D_COMPRESSOR_PACKAGE or VIDEO3D_COMP_ROOT. Searched: {searched}"
        )

    spec = importlib.util.spec_from_file_location(
        _SHARED_PACKAGE_NAME,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load multimodal_compressor package from {package_dir}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_SHARED_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


_shared = _load_shared_package()

_ALLOWED_EXPORTS = {
    "BaseCompressor",
    "CompressorConfig",
    "CompressorOutput",
    "IdentityCompressor",
    "LLMForwardContext",
    "LLMForwardOutput",
    "LateEntryEarlyExitCompressor",
    "LateEntryEarlyExitConfig",
    "build_compressor",
    "build_compressor_from_llava_config",
    "get_compressor_spec_from_llava_config",
    "get_compressor",
    "list_compressors",
    "is_registered",
    "COMPRESSOR_REGISTRY",
    "VisPrunerCompressor",
    "VisPrunerConfig",
    "VisionZipCompressor",
    "VisionZipConfig",
    "VoxelDTCCompressor",
    "VoxelDTCConfig",
    "VoxelVTCCompressor",
    "VoxelVTCConfig",
    "VoxelVTCVisionZipCompressor",
    "VoxelVTCVisionZipConfig",
    "SegPrunerProjectorCompressor",
    "SegPrunerProjectorConfig",
    "SpatialKCenterMergeCompressor",
    "SpatialKCenterMergeConfig",
}

__all__ = [name for name in getattr(_shared, "__all__", []) if name in _ALLOWED_EXPORTS]
for _name in __all__:
    globals()[_name] = getattr(_shared, _name)

_shared_build_compressor = getattr(_shared, "build_compressor")
_shared_build_compressor_from_llava_config = getattr(_shared, "build_compressor_from_llava_config")
_shared_get_compressor_spec_from_llava_config = getattr(_shared, "get_compressor_spec_from_llava_config")
_shared_get_compressor = getattr(_shared, "get_compressor")
_shared_registry = getattr(_shared, "COMPRESSOR_REGISTRY")
BaseCompressor = getattr(_shared, "BaseCompressor")
CompressorConfig = getattr(_shared, "CompressorConfig")
CompressorOutput = getattr(_shared, "CompressorOutput")


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


class _OVProjectorVideoMixin:
    def _validate_target_spec(self, target_tokens: Optional[int], target_keep_ratio: Optional[float]) -> None:
        if target_tokens is None and target_keep_ratio is None:
            raise ValueError(f"{self.__class__.__name__} requires either target_tokens or target_keep_ratio.")
        if target_tokens is not None and target_keep_ratio is not None:
            raise ValueError(f"{self.__class__.__name__} accepts either target_tokens or target_keep_ratio, not both.")
        if target_tokens is not None and int(target_tokens) <= 0:
            raise ValueError(f"target_tokens must be positive, got {target_tokens}.")
        if target_keep_ratio is not None and not 0 < float(target_keep_ratio) <= 1:
            raise ValueError(f"target_keep_ratio must be within (0, 1], got {target_keep_ratio}.")

    def _resolve_target_tokens(
        self,
        input_tokens: int,
        target_tokens: Optional[int],
        target_keep_ratio: Optional[float],
    ) -> int:
        if target_tokens is not None:
            return min(max(1, int(target_tokens)), input_tokens)
        return min(max(1, int(round(input_tokens * float(target_keep_ratio)))), input_tokens)

    def _normalize_features(self, features: torch.Tensor, num_frames: int, tokens_per_frame: int) -> torch.Tensor:
        if features.dim() == 3 and features.shape[0] == num_frames and features.shape[1] == tokens_per_frame:
            return features
        if features.dim() == 3 and features.shape[0] == 1 and features.shape[1] == num_frames * tokens_per_frame:
            return features.view(num_frames, tokens_per_frame, features.shape[-1])
        raise ValueError(
            f"Unexpected feature layout {tuple(features.shape)} for frames={num_frames}, tokens={tokens_per_frame}."
        )

    def _normalize_attention(
        self,
        attn_weights: torch.Tensor,
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attn_weights is None:
            raise ValueError(f"{self.__class__.__name__} requires visual attentions.")
        if attn_weights.dim() == 3 and attn_weights.shape[0] == 1:
            attn_weights = attn_weights[0]
        if attn_weights.dim() == 2 and attn_weights.shape == (num_frames, tokens_per_frame):
            return attn_weights.to(device=device, dtype=torch.float32)
        if attn_weights.dim() == 2 and attn_weights.shape == (1, num_frames * tokens_per_frame):
            return attn_weights.view(num_frames, tokens_per_frame).to(device=device, dtype=torch.float32)
        raise ValueError(
            f"Unexpected attention layout {tuple(attn_weights.shape)} for frames={num_frames}, tokens={tokens_per_frame}."
        )

    def _normalize_coordinates(
        self,
        coordinates: torch.Tensor,
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if coordinates is None:
            raise ValueError(f"{self.__class__.__name__} requires patch-level 3D coordinates.")
        if coordinates.dim() >= 4 and coordinates.shape[0] == 1 and coordinates.shape[1] == num_frames:
            coordinates = coordinates[0]
        elif coordinates.dim() >= 3 and coordinates.shape[0] == num_frames:
            pass
        elif coordinates.dim() >= 3 and coordinates.shape[0] == 1 and coordinates.shape[1] == num_frames * tokens_per_frame:
            coordinates = coordinates.view(num_frames, tokens_per_frame, *coordinates.shape[2:])
        else:
            raise ValueError(
                f"Unexpected coordinate layout {tuple(coordinates.shape)} for frames={num_frames}, tokens={tokens_per_frame}."
            )
        if coordinates.shape[0] != num_frames or coordinates.shape[1] != tokens_per_frame:
            raise ValueError(
                f"Coordinate layout {tuple(coordinates.shape)} does not match frames={num_frames}, tokens={tokens_per_frame}."
            )
        return coordinates.to(device=device, dtype=torch.float32)

    def _patch_positions(self, num_frames: int, frame_shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        height, width = frame_shape
        row_ids = torch.arange(height, device=device, dtype=torch.float32)
        col_ids = torch.arange(width, device=device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(row_ids, col_ids, indexing="ij")
        return torch.stack((row_grid, col_grid), dim=-1).view(1, height * width, 2).repeat(num_frames, 1, 1)

    def _flatten_video_state(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        attentions: torch.Tensor,
        frame_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        num_frames, tokens_per_frame, _ = features.shape
        device = features.device
        patch_positions = self._patch_positions(num_frames, frame_shape, device)
        return {
            "features": features.reshape(num_frames * tokens_per_frame, features.shape[-1]),
            "coordinates": coordinates.reshape(num_frames * tokens_per_frame, coordinates.shape[-1]),
            "attentions": attentions.reshape(num_frames * tokens_per_frame),
            "patch_positions": patch_positions.reshape(num_frames * tokens_per_frame, 2),
            "frame_ids": torch.arange(num_frames, device=device, dtype=torch.long).repeat_interleave(tokens_per_frame),
            "order_ids": torch.arange(num_frames * tokens_per_frame, device=device, dtype=torch.long),
        }

    def _pack_selected(
        self,
        *,
        flat_state: Dict[str, torch.Tensor],
        selected_indices: torch.Tensor,
        output_features: Optional[torch.Tensor],
        num_frames: int,
        input_tokens: int,
        newline_strategy: str,
        extra_metadata: Dict[str, Any],
    ) -> CompressorOutput:
        selected_indices = selected_indices.to(device=flat_state["features"].device, dtype=torch.long)
        if output_features is None:
            output_features = flat_state["features"].index_select(0, selected_indices)
        selected_frame_ids = flat_state["frame_ids"].index_select(0, selected_indices)
        selected_order_ids = flat_state["order_ids"].index_select(0, selected_indices)
        order = selected_order_ids.argsort(stable=True)

        output_features = output_features.index_select(0, order)
        selected_indices = selected_indices.index_select(0, order)
        selected_frame_ids = selected_frame_ids.index_select(0, order)
        selected_order_ids = selected_order_ids.index_select(0, order)
        frame_token_counts = torch.bincount(selected_frame_ids, minlength=num_frames).tolist()

        output_tokens = int(output_features.shape[0])
        self._update_stats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            frame_token_counts=[int(count) for count in frame_token_counts],
            **{key: value for key, value in extra_metadata.items() if isinstance(value, (int, float, bool, str))},
        )
        metadata = {
            "newline_strategy": newline_strategy,
            "frame_token_counts": [int(count) for count in frame_token_counts],
            "compressed_coordinates": flat_state["coordinates"].index_select(0, selected_indices).unsqueeze(0),
            "compressed_patch_positions": flat_state["patch_positions"].index_select(0, selected_indices).unsqueeze(0),
            "retained_order_ids": [int(value) for value in selected_order_ids.detach().cpu().tolist()],
            **extra_metadata,
        }
        return CompressorOutput(
            features=output_features.unsqueeze(0),
            compression_ratio=(float(output_tokens) / float(input_tokens)) if input_tokens > 0 else 1.0,
            metadata=metadata,
        )

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["attn_weights"] = True
        required["raw_features_before_proj"] = True
        required["coordinates"] = True
        return required

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False


class SegPrunerProjectorCompressor(_OVProjectorVideoMixin, BaseCompressor):
    """Projector-level SegPruner-style selector for OV video tokens.

    This keeps the semantic-attention plus geometric-diversity idea, but it
    operates on OV pooled projector tokens. It is not the raw 27x27 pre-projector
    patch from the SegPruner paper.
    """

    def __init__(self, config: Dict[str, Any]):
        seg_config = SegPrunerProjectorConfig(**config) if isinstance(config, dict) else config
        if seg_config.visual_token_num is None and seg_config.token_keep_ratio is None:
            raise ValueError("segpruner requires either visual_token_num or token_keep_ratio.")
        if seg_config.visual_token_num is not None and seg_config.token_keep_ratio is not None:
            raise ValueError("segpruner accepts either visual_token_num or token_keep_ratio, not both.")
        if seg_config.visual_token_num is not None and int(seg_config.visual_token_num) <= 0:
            raise ValueError(f"visual_token_num must be positive, got {seg_config.visual_token_num}.")
        if seg_config.token_keep_ratio is not None and not 0 < float(seg_config.token_keep_ratio) <= 1:
            raise ValueError(f"token_keep_ratio must be within (0, 1], got {seg_config.token_keep_ratio}.")
        if not 0 <= float(seg_config.important_ratio) <= 1:
            raise ValueError(f"important_ratio must be within [0, 1], got {seg_config.important_ratio}.")
        if not 0 <= float(seg_config.lam) <= 1:
            raise ValueError(f"lam must be within [0, 1], got {seg_config.lam}.")
        if seg_config.newline_strategy not in {"grid_drop", "full_grid_drop", "frame_newline", "one_token", "no_token"}:
            raise ValueError(f"Unexpected newline_strategy: {seg_config.newline_strategy}.")
        super().__init__(seg_config)
        self.seg_config = seg_config

    def _resolve_budget(self, input_tokens: int) -> int:
        if self.seg_config.visual_token_num is not None:
            return min(max(1, int(self.seg_config.visual_token_num)), input_tokens)
        return min(max(1, int(round(input_tokens * float(self.seg_config.token_keep_ratio)))), input_tokens)

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
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
        flat_state = self._flatten_video_state(candidate_features, coords, attentions, frame_shape)
        flat_semantic = semantic_features.reshape(num_frames * tokens_per_frame, semantic_features.shape[-1])

        input_tokens = int(flat_state["features"].shape[0])
        target_tokens = self._resolve_budget(input_tokens)
        important_count = min(target_tokens, int(round(target_tokens * float(self.seg_config.important_ratio))))
        diverse_count = target_tokens - important_count

        order = flat_state["attentions"].argsort(descending=True, stable=True)
        important_indices = order[:important_count]
        residual_indices = order[important_count:]
        if diverse_count <= 0 or residual_indices.numel() == 0:
            selected_indices = important_indices
        else:
            normalized_semantic = F.normalize(flat_semantic.float(), dim=-1)
            residual_coords = flat_state["coordinates"].index_select(0, residual_indices)
            residual_semantic = normalized_semantic.index_select(0, residual_indices)
            selected_local = []
            available = torch.ones(residual_indices.shape[0], device=residual_indices.device, dtype=torch.bool)
            farthest = torch.zeros((), device=residual_indices.device, dtype=torch.long)
            min_fused = torch.full((residual_indices.shape[0],), float("inf"), device=residual_indices.device)
            lam = float(self.seg_config.lam)
            for _ in range(min(diverse_count, int(residual_indices.numel()))):
                selected_local.append(farthest)
                available[farthest] = False
                centroid_coord = residual_coords[farthest].view(1, -1)
                dist = torch.cdist(residual_coords.float(), centroid_coord.float()).squeeze(-1)
                dist_scale = dist.max().clamp_min(1e-6)
                semantic_scores = residual_semantic[farthest].view(1, -1) @ residual_semantic.transpose(0, 1)
                fused = lam * (dist / dist_scale) + (1.0 - lam) * (1.0 - semantic_scores.squeeze(0))
                min_fused = torch.minimum(min_fused, fused)
                masked = min_fused.masked_fill(~available, -float("inf"))
                if not available.any():
                    break
                farthest = masked.argmax()
            diverse_indices = residual_indices.index_select(0, torch.stack(selected_local)) if selected_local else residual_indices[:0]
            selected_indices = torch.cat((important_indices, diverse_indices), dim=0)

        return self._pack_selected(
            flat_state=flat_state,
            selected_indices=selected_indices[:target_tokens],
            output_features=None,
            num_frames=num_frames,
            input_tokens=input_tokens,
            newline_strategy=self.seg_config.newline_strategy,
            extra_metadata={
                "selection_method": "segpruner_projector_attention_geometry",
                "visual_token_num": int(target_tokens),
                "important_ratio": float(self.seg_config.important_ratio),
                "lam": float(self.seg_config.lam),
                "important_tokens": int(important_count),
                "diverse_tokens": int(max(0, min(diverse_count, target_tokens - important_count))),
            },
        )


class SpatialKCenterMergeCompressor(_OVProjectorVideoMixin, BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        kcenter_config = SpatialKCenterMergeConfig(**config) if isinstance(config, dict) else config
        self._validate_target_spec(kcenter_config.target_tokens, kcenter_config.target_keep_ratio)
        if not 0 <= float(kcenter_config.important_ratio) <= 1:
            raise ValueError(f"important_ratio must be within [0, 1], got {kcenter_config.important_ratio}.")
        if not 0 <= float(kcenter_config.merge_weight) <= 1:
            raise ValueError(f"merge_weight must be within [0, 1], got {kcenter_config.merge_weight}.")
        if not 0 <= float(kcenter_config.score_weight) <= 1:
            raise ValueError(f"score_weight must be within [0, 1], got {kcenter_config.score_weight}.")
        if kcenter_config.newline_strategy not in {"grid_drop", "full_grid_drop", "frame_newline", "one_token", "no_token"}:
            raise ValueError(f"Unexpected newline_strategy: {kcenter_config.newline_strategy}.")
        super().__init__(kcenter_config)
        self.kcenter_config = kcenter_config

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
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
        flat_state = self._flatten_video_state(candidate_features, coords, attentions, frame_shape)
        flat_semantic = semantic_features.reshape(num_frames * tokens_per_frame, semantic_features.shape[-1])

        input_tokens = int(flat_state["features"].shape[0])
        target_tokens = self._resolve_target_tokens(
            input_tokens,
            self.kcenter_config.target_tokens,
            self.kcenter_config.target_keep_ratio,
        )
        important_count = min(target_tokens, int(round(target_tokens * float(self.kcenter_config.important_ratio))))
        anchor_order = flat_state["attentions"].argsort(descending=True, stable=True)
        important_indices = anchor_order[:important_count]
        residual_indices = anchor_order[important_count:]
        remaining_count = target_tokens - important_count

        selected = [idx for idx in important_indices]
        if remaining_count > 0 and residual_indices.numel() > 0:
            normalized_scores = flat_state["attentions"]
            if normalized_scores.numel() > 0:
                score_min = normalized_scores.min()
                score_range = (normalized_scores.max() - score_min).clamp_min(1e-6)
                normalized_scores = (normalized_scores - score_min) / score_range
            candidate_indices = residual_indices
            candidate_coords = flat_state["coordinates"].index_select(0, candidate_indices)
            min_dist = torch.full((candidate_indices.shape[0],), float("inf"), device=candidate_indices.device)
            if selected:
                selected_tensor = torch.stack(selected).to(device=candidate_indices.device, dtype=torch.long)
                selected_coords = flat_state["coordinates"].index_select(0, selected_tensor)
                min_dist = torch.cdist(candidate_coords.float(), selected_coords.float()).min(dim=-1).values
            available = torch.ones(candidate_indices.shape[0], device=candidate_indices.device, dtype=torch.bool)
            score_weight = float(self.kcenter_config.score_weight)
            for _ in range(min(remaining_count, int(candidate_indices.numel()))):
                dist_norm = min_dist / min_dist.max().clamp_min(1e-6)
                candidate_scores = normalized_scores.index_select(0, candidate_indices)
                combined = (1.0 - score_weight) * dist_norm + score_weight * candidate_scores
                combined = combined.masked_fill(~available, -float("inf"))
                local_idx = combined.argmax()
                selected_idx = candidate_indices[local_idx]
                selected.append(selected_idx)
                available[local_idx] = False
                new_dist = torch.cdist(candidate_coords.float(), flat_state["coordinates"][selected_idx].view(1, -1).float()).squeeze(-1)
                min_dist = torch.minimum(min_dist, new_dist)
                if not available.any():
                    break

        selected_indices = torch.stack(selected).to(device=flat_state["features"].device, dtype=torch.long)
        selected_features = flat_state["features"].index_select(0, selected_indices).clone()
        residual_mask = torch.ones(input_tokens, device=flat_state["features"].device, dtype=torch.bool)
        residual_mask[selected_indices] = False
        merge_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()
        if merge_indices.numel() > 0 and selected_indices.numel() > 0 and float(self.kcenter_config.merge_weight) > 0:
            normalized_semantic = F.normalize(flat_semantic.float(), dim=-1)
            merge_semantic = normalized_semantic.index_select(0, merge_indices)
            selected_semantic = normalized_semantic.index_select(0, selected_indices)
            assignment = (merge_semantic @ selected_semantic.transpose(0, 1)).argmax(dim=-1)
            merge_features = flat_state["features"].index_select(0, merge_indices)
            aggregated = torch.zeros_like(selected_features)
            counts = torch.zeros(selected_indices.shape[0], device=selected_features.device, dtype=torch.float32)
            aggregated.index_add_(0, assignment, merge_features)
            counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float32))
            merged_mean = aggregated / counts.clamp_min(1.0).unsqueeze(-1).to(dtype=selected_features.dtype)
            mask = counts > 0
            merge_weight = float(self.kcenter_config.merge_weight)
            selected_features[mask] = (
                (1.0 - merge_weight) * selected_features[mask]
                + merge_weight * merged_mean[mask]
            ).to(dtype=selected_features.dtype)

        return self._pack_selected(
            flat_state=flat_state,
            selected_indices=selected_indices,
            output_features=selected_features,
            num_frames=num_frames,
            input_tokens=input_tokens,
            newline_strategy=self.kcenter_config.newline_strategy,
            extra_metadata={
                "selection_method": "spatial_kcenter_semantic_merge",
                "target_tokens": int(target_tokens),
                "important_ratio": float(self.kcenter_config.important_ratio),
                "merge_weight": float(self.kcenter_config.merge_weight),
                "score_weight": float(self.kcenter_config.score_weight),
                "important_tokens": int(important_count),
            },
        )


_LOCAL_REGISTRY = {}

# Keep the scheduler shared but bind the model-specific OV adapter locally.
from .late_entry_early_exit_adapter import (  # noqa: E402
    LateEntryEarlyExitCompressor as _OVLateEntryEarlyExitCompressor,
    LateEntryEarlyExitConfig as _OVLateEntryEarlyExitConfig,
)
_LOCAL_REGISTRY["late_entry_early_exit"] = _OVLateEntryEarlyExitCompressor
globals()["LateEntryEarlyExitCompressor"] = _OVLateEntryEarlyExitCompressor
globals()["LateEntryEarlyExitConfig"] = _OVLateEntryEarlyExitConfig

__all__ = sorted(
    set(__all__)
    | {
        "SegPrunerProjectorCompressor",
        "SegPrunerProjectorConfig",
        "SpatialKCenterMergeCompressor",
        "SpatialKCenterMergeConfig",
    }
)


def _ensure_allowed_compressor(compressor_type):
    if compressor_type in (None, "", "none", "identity"):
        return
    if compressor_type not in _ALLOWED_COMPRESSORS:
        raise ValueError(
            "LLaVA-OV compressor adapter only keeps "
            f"{sorted(_ALLOWED_COMPRESSORS)}; got {compressor_type!r}."
        )


def build_compressor(compressor_type=None, compressor_config=None, **kwargs):
    _ensure_allowed_compressor(compressor_type)
    if compressor_type in _LOCAL_REGISTRY:
        config = compressor_config or {}
        config = config.copy()
        config.update(kwargs)
        return _LOCAL_REGISTRY[compressor_type](config)
    return _shared_build_compressor(compressor_type, compressor_config, **kwargs)


def build_compressor_from_llava_config(llava_config, location="projector"):
    compressor_type, _ = _shared_get_compressor_spec_from_llava_config(llava_config, location=location)
    _ensure_allowed_compressor(compressor_type)
    if compressor_type in _LOCAL_REGISTRY:
        _, compressor_config = _shared_get_compressor_spec_from_llava_config(llava_config, location=location)
        default_position = "after_projector" if location == "projector" else "llm"
        return build_compressor(compressor_type, compressor_config, position=default_position)
    return _shared_build_compressor_from_llava_config(llava_config, location=location)


def get_compressor(name):
    _ensure_allowed_compressor(name)
    if name in _LOCAL_REGISTRY:
        return _LOCAL_REGISTRY[name]
    return _shared_get_compressor(name)


def list_compressors():
    return sorted(_ALLOWED_COMPRESSORS)


def is_registered(name):
    return name in _ALLOWED_COMPRESSORS


COMPRESSOR_REGISTRY = {
    name: _shared_registry[name]
    for name in sorted(_SHARED_ALLOWED_COMPRESSORS)
    if name in _shared_registry
}
COMPRESSOR_REGISTRY.update(_LOCAL_REGISTRY)

globals()["build_compressor"] = build_compressor
globals()["build_compressor_from_llava_config"] = build_compressor_from_llava_config
globals()["get_compressor"] = get_compressor
globals()["list_compressors"] = list_compressors
globals()["is_registered"] = is_registered
globals()["COMPRESSOR_REGISTRY"] = COMPRESSOR_REGISTRY
globals()["SegPrunerProjectorCompressor"] = getattr(_shared, "SegPrunerProjectorCompressor")
globals()["SegPrunerProjectorConfig"] = getattr(_shared, "SegPrunerProjectorConfig")
globals()["SpatialKCenterMergeCompressor"] = getattr(_shared, "SpatialKCenterMergeCompressor")
globals()["SpatialKCenterMergeConfig"] = getattr(_shared, "SpatialKCenterMergeConfig")

for _module_name, _module in list(sys.modules.items()):
    if _module_name == _SHARED_PACKAGE_NAME or not _module_name.startswith(f"{_SHARED_PACKAGE_NAME}."):
        continue
    _suffix = _module_name[len(_SHARED_PACKAGE_NAME):]
    if _suffix not in _ALLOWED_SUBMODULE_SUFFIXES:
        continue
    _alias = f"{__name__}{_module_name[len(_SHARED_PACKAGE_NAME):]}"
    sys.modules[_alias] = _module
