import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import BaseCompressor, CompressorConfig, CompressorOutput
from ..registry import register_compressor
from ..voxel_dtc.compressor import _VoxelCompressionMixin


@dataclass
class VoxelGeoSemAnchorPruneConfig(CompressorConfig):
    voxel_size: float = 0.1
    target_tokens: Optional[int] = None
    target_keep_ratio: Optional[float] = None
    attention_reduce: str = "max"
    alpha: float = 0.45
    beta: float = 0.20
    gamma: float = 0.15
    delta: float = 0.20
    newline_strategy: str = "grid_drop"


@dataclass
class VoxelGeoSemAnchorMergeConfig(CompressorConfig):
    voxel_size: float = 0.1
    target_tokens: Optional[int] = None
    target_keep_ratio: Optional[float] = None
    attention_reduce: str = "max"
    knn_k: int = 6
    anchor_knn: int = 8
    alpha: float = 0.45
    beta: float = 0.20
    gamma: float = 0.15
    delta: float = 0.20
    lambda_match: float = 0.60
    mu_match: float = 0.30
    nu_match: float = 0.10
    rho_weight: float = 0.50
    eta_merge: float = 1.00
    newline_strategy: str = "grid_drop"


class _VoxelGeoSemMixin(_VoxelCompressionMixin):
    _NEWLINE_STRATEGIES = {"grid_drop", "full_grid_drop", "frame_newline", "one_token", "no_token"}

    def _validate_newline_strategy(self, value: str) -> None:
        if value not in self._NEWLINE_STRATEGIES:
            raise ValueError(f"Unexpected GeoSem newline_strategy: {value!r}.")

    def _validate_target_spec(self, target_tokens: Optional[int], target_keep_ratio: Optional[float]) -> None:
        if target_tokens is None and target_keep_ratio is None:
            raise ValueError("GeoSem compressors require either target_tokens or target_keep_ratio.")
        if target_tokens is not None and target_keep_ratio is not None:
            raise ValueError("GeoSem compressors accept either target_tokens or target_keep_ratio, not both.")
        if target_tokens is not None and int(target_tokens) <= 0:
            raise ValueError(f"target_tokens must be positive, got {target_tokens}.")
        if target_keep_ratio is not None and not 0 < float(target_keep_ratio) <= 1:
            raise ValueError(f"target_keep_ratio must be within (0, 1], got {target_keep_ratio}.")

    def _validate_attention_reduce(self, attention_reduce: str) -> None:
        if attention_reduce not in {"max", "mean"}:
            raise ValueError(f"attention_reduce must be 'max' or 'mean', got {attention_reduce}.")

    def _normalize_attention_scores(
        self,
        attn_weights: Optional[torch.Tensor],
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attn_weights is None:
            raise ValueError("GeoSem compressors require attn_weights.")
        if attn_weights.dim() == 3 and attn_weights.shape[0] == 1:
            attn_weights = attn_weights[0]
        if attn_weights.dim() == 2 and attn_weights.shape == (num_frames, tokens_per_frame):
            return attn_weights.to(device=device, dtype=torch.float32)
        if attn_weights.dim() == 2 and attn_weights.shape == (1, num_frames * tokens_per_frame):
            return attn_weights.view(num_frames, tokens_per_frame).to(device=device, dtype=torch.float32)
        raise ValueError(
            "Unexpected GeoSem attention layout: "
            f"{tuple(attn_weights.shape)} for num_frames={num_frames}, tokens_per_frame={tokens_per_frame}."
        )

    def _resolve_target_tokens(self, input_tokens: int, target_tokens: Optional[int], target_keep_ratio: Optional[float]) -> int:
        if target_tokens is not None:
            return max(1, int(target_tokens))
        return max(1, int(math.ceil(input_tokens * float(target_keep_ratio))))

    def _minmax_normalize(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values
        values = values.to(dtype=torch.float32)
        min_value = values.min()
        max_value = values.max()
        scale = max_value - min_value
        if torch.abs(scale).item() < 1e-8:
            return torch.zeros_like(values)
        return (values - min_value) / scale

    def _reduce_attention(self, group_attention: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "max":
            return group_attention.max()
        return group_attention.mean()

    def _compute_joint_anchor_score(
        self,
        *,
        voxel_state: Dict[str, Any],
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
    ) -> torch.Tensor:
        attention_norm = self._minmax_normalize(voxel_state["voxel_attention"])
        view_norm = self._minmax_normalize(voxel_state["voxel_view_support"])
        heterogeneity_norm = self._minmax_normalize(voxel_state["voxel_heterogeneity"])
        boundaryness_norm = self._minmax_normalize(voxel_state["voxel_boundaryness"])
        return (
            float(alpha) * attention_norm
            + float(beta) * view_norm
            + float(gamma) * heterogeneity_norm
            + float(delta) * boundaryness_norm
        )

    def _build_voxel_state(
        self,
        *,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        grouping_coordinates: Optional[torch.Tensor],
        attentions: torch.Tensor,
        num_frames: int,
        frame_shape: Tuple[int, int],
        voxel_size: float,
        attention_reduce: str,
        compute_boundaryness: bool = True,
        vectorized: bool = False,
        materialize_group_metadata: bool = True,
        compute_heterogeneity: bool = True,
    ) -> Dict[str, Any]:
        if num_frames is None:
            raise ValueError("GeoSem compressors require num_frames.")
        if frame_shape is None:
            raise ValueError("GeoSem compressors require frame_shape.")

        features = self._normalize_features(features, num_frames)
        if features.shape[0] != 1:
            raise ValueError("GeoSem compressors currently support one video sample at a time.")

        _, total_tokens, _ = features.shape
        height, width = frame_shape
        tokens_per_frame = height * width
        if total_tokens != num_frames * tokens_per_frame:
            raise ValueError(f"Expected {num_frames * tokens_per_frame} tokens, got {total_tokens}.")

        normalized_coordinates, normalized_grouping_coordinates = self._resolve_coordinate_pair(
            coordinates=coordinates,
            grouping_coordinates=grouping_coordinates,
            batch_size=1,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            device=features.device,
        )
        normalized_coordinates = normalized_coordinates[0]
        normalized_grouping_coordinates = normalized_grouping_coordinates[0]
        normalized_attentions = self._normalize_attention_scores(attentions, num_frames, tokens_per_frame, features.device)
        patch_positions = self._build_patch_positions(num_frames, frame_shape, features.device)

        flat_features, flat_coordinates, flat_patch_positions, frame_ids, order_ids = self._flatten_token_state(
            features=features[0],
            coordinates=normalized_coordinates,
            patch_positions=patch_positions,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
        )
        flat_grouping_coordinates = normalized_grouping_coordinates.view(
            num_frames * tokens_per_frame,
            normalized_grouping_coordinates.shape[-1],
        )
        flat_attentions = normalized_attentions.reshape(-1)

        if vectorized:
            return self._build_voxel_state_vectorized(
                flat_features=flat_features,
                flat_coordinates=flat_coordinates,
                flat_grouping_coordinates=flat_grouping_coordinates,
                flat_patch_positions=flat_patch_positions,
                frame_ids=frame_ids,
                order_ids=order_ids,
                flat_attentions=flat_attentions,
                num_frames=num_frames,
                input_tokens=total_tokens,
                voxel_size=voxel_size,
                attention_reduce=attention_reduce,
                compute_boundaryness=compute_boundaryness,
                materialize_group_metadata=materialize_group_metadata,
                compute_heterogeneity=compute_heterogeneity,
            )

        voxel_groups = self._group_tokens_by_voxel(flat_grouping_coordinates, voxel_size)

        voxel_features: List[torch.Tensor] = []
        voxel_coordinates: List[torch.Tensor] = []
        voxel_grouping_coordinates: List[torch.Tensor] = []
        voxel_patch_positions: List[torch.Tensor] = []
        voxel_frame_ids: List[torch.Tensor] = []
        voxel_order_ids: List[torch.Tensor] = []
        voxel_attention_scores: List[torch.Tensor] = []
        voxel_view_support: List[float] = []
        voxel_heterogeneity: List[torch.Tensor] = []
        voxel_occupancies: List[int] = []
        voxel_group_frame_ids: List[torch.Tensor] = []

        for group in voxel_groups:
            group_index = torch.tensor(group, device=features.device, dtype=torch.long)
            representative_idx = group_index[order_ids[group_index].argmin()]
            group_features = flat_features[group_index]
            group_coords = flat_coordinates[group_index]
            group_grouping_coords = flat_grouping_coordinates[group_index]
            group_attentions = flat_attentions[group_index]
            group_frame_ids = frame_ids[group_index]
            voxel_feature = group_features.mean(dim=0)
            voxel_features.append(voxel_feature)
            voxel_coordinates.append(group_coords.mean(dim=0))
            voxel_grouping_coordinates.append(group_grouping_coords.mean(dim=0))
            voxel_patch_positions.append(flat_patch_positions[representative_idx])
            voxel_frame_ids.append(frame_ids[representative_idx])
            voxel_order_ids.append(order_ids[representative_idx])
            voxel_attention_scores.append(self._reduce_attention(group_attentions, attention_reduce))
            voxel_view_support.append(float(group_frame_ids.unique().numel()) / float(num_frames))
            voxel_heterogeneity.append(((group_features - voxel_feature) ** 2).sum(dim=-1).mean())
            voxel_occupancies.append(int(group_index.numel()))
            voxel_group_frame_ids.append(group_frame_ids.unique(sorted=True))

        voxel_features_tensor = torch.stack(voxel_features, dim=0)
        voxel_coordinates_tensor = torch.stack(voxel_coordinates, dim=0).to(dtype=normalized_coordinates.dtype)
        voxel_grouping_coordinates_tensor = torch.stack(voxel_grouping_coordinates, dim=0).to(
            dtype=normalized_grouping_coordinates.dtype
        )
        voxel_patch_positions_tensor = torch.stack(voxel_patch_positions, dim=0)
        voxel_frame_ids_tensor = torch.stack(voxel_frame_ids, dim=0)
        voxel_order_ids_tensor = torch.stack(voxel_order_ids, dim=0)
        voxel_attention_tensor = torch.stack(voxel_attention_scores, dim=0).to(dtype=torch.float32)
        voxel_view_support_tensor = torch.tensor(voxel_view_support, device=features.device, dtype=torch.float32)
        voxel_heterogeneity_tensor = torch.stack(voxel_heterogeneity, dim=0).to(dtype=torch.float32)
        if compute_boundaryness:
            voxel_boundaryness_tensor = self._compute_boundaryness(
                voxel_features=voxel_features_tensor,
                voxel_coordinates=voxel_grouping_coordinates_tensor,
                voxel_size=voxel_size,
                knn_k=getattr(self, "_boundary_knn_k", 6),
            )
        else:
            voxel_boundaryness_tensor = torch.zeros(
                voxel_features_tensor.shape[0],
                device=voxel_features_tensor.device,
                dtype=torch.float32,
            )

        return {
            "input_tokens": total_tokens,
            "num_frames": num_frames,
            "voxel_groups": voxel_groups,
            "voxel_features": voxel_features_tensor,
            "voxel_coordinates": voxel_coordinates_tensor,
            "voxel_grouping_coordinates": voxel_grouping_coordinates_tensor,
            "voxel_patch_positions": voxel_patch_positions_tensor,
            "voxel_frame_ids": voxel_frame_ids_tensor,
            "voxel_order_ids": voxel_order_ids_tensor,
            "voxel_attention": voxel_attention_tensor,
            "voxel_view_support": voxel_view_support_tensor,
            "voxel_heterogeneity": voxel_heterogeneity_tensor,
            "voxel_boundaryness": voxel_boundaryness_tensor,
            "voxel_occupancies": torch.tensor(voxel_occupancies, device=features.device, dtype=torch.float32),
            "voxel_group_frame_ids": voxel_group_frame_ids,
        }

    def _segment_reduce(
        self,
        values: torch.Tensor,
        reduction: str,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce contiguous group segments without launching one op per voxel.

        ``torch.segment_reduce`` is available in the supported PyTorch runtime.
        The fallback keeps the vectorized path usable on older installations
        and is still free of the old Python loop over individual voxels.
        """
        segment_reduce = getattr(torch, "segment_reduce", None)
        if segment_reduce is not None:
            return segment_reduce(values, reduction, lengths=lengths)

        num_segments = int(lengths.numel())
        output_shape = (num_segments,) + tuple(values.shape[1:])
        if reduction == "max":
            output = torch.full(
                output_shape,
                float("-inf"),
                device=values.device,
                dtype=values.dtype,
            )
            group_ids = torch.repeat_interleave(
                torch.arange(num_segments, device=values.device, dtype=torch.long),
                lengths,
            )
            expand_shape = (group_ids.shape[0],) + (1,) * (values.dim() - 1)
            output.scatter_reduce_(
                0,
                group_ids.view(expand_shape).expand_as(values),
                values,
                reduce="amax",
                include_self=True,
            )
            return output

        output = torch.zeros(output_shape, device=values.device, dtype=values.dtype)
        group_ids = torch.repeat_interleave(
            torch.arange(num_segments, device=values.device, dtype=torch.long),
            lengths,
        )
        output.index_add_(0, group_ids, values)
        divisor = lengths.to(dtype=values.dtype).clamp_min_(1)
        divisor = divisor.view((num_segments,) + (1,) * (values.dim() - 1))
        return output / divisor

    def _build_voxel_state_vectorized(
        self,
        *,
        flat_features: torch.Tensor,
        flat_coordinates: torch.Tensor,
        flat_grouping_coordinates: torch.Tensor,
        flat_patch_positions: torch.Tensor,
        frame_ids: torch.Tensor,
        order_ids: torch.Tensor,
        flat_attentions: torch.Tensor,
        num_frames: int,
        input_tokens: int,
        voxel_size: float,
        attention_reduce: str,
        compute_boundaryness: bool,
        materialize_group_metadata: bool,
        compute_heterogeneity: bool,
    ) -> Dict[str, Any]:
        """Build voxel statistics with batched GPU reductions.

        The reference implementation above intentionally mirrors the original
        Python grouping semantics.  This implementation keeps those semantics
        while replacing the per-voxel CUDA launches with segment reductions:

        * ``torch.unique`` computes the same rounded voxel membership;
        * a stable sort preserves the original token order inside each group;
        * groups are reordered by their first token to recover dict insertion
          order from the reference implementation;
        * representative patches, attention reduction, means, occupancies,
          and frame support are all computed from the same group assignment.
        """
        voxel_indices = self._discretize_voxel_indices(flat_grouping_coordinates, voxel_size)
        _, inverse, counts = torch.unique(
            voxel_indices,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )
        num_voxels = int(counts.numel())
        if num_voxels == 0:
            raise ValueError("Voxel grouping produced no groups for a non-empty token sequence.")

        # Stable sorting makes each segment retain the original token order.
        # Therefore the first element of every segment is the same representative
        # patch selected by the reference dict/list implementation.
        sort_order = torch.argsort(inverse, stable=True)
        sorted_inverse = inverse.index_select(0, sort_order)
        segment_starts = counts.cumsum(dim=0) - counts
        representative_indices = sort_order.index_select(0, segment_starts)
        group_order = torch.argsort(representative_indices, stable=True)

        sorted_features = flat_features.index_select(0, sort_order)
        sorted_coordinates = flat_coordinates.reshape(input_tokens, -1).index_select(0, sort_order)
        sorted_grouping_coordinates = flat_grouping_coordinates.index_select(0, sort_order)

        voxel_features = self._segment_reduce(sorted_features, "mean", counts)
        voxel_coordinates = self._segment_reduce(sorted_coordinates, "mean", counts).reshape(
            (num_voxels,) + tuple(flat_coordinates.shape[1:])
        )
        voxel_grouping_coordinates = self._segment_reduce(
            sorted_grouping_coordinates,
            "mean",
            counts,
        )

        if attention_reduce == "max":
            attention_reduction = "max"
        else:
            attention_reduction = "mean"
        voxel_attention = self._segment_reduce(
            flat_attentions.index_select(0, sort_order),
            attention_reduction,
            counts,
        ).to(dtype=torch.float32)

        if compute_heterogeneity:
            # Match ((group_features - group_mean) ** 2).sum(-1).mean() while
            # retaining the same feature dtype as the reference group mean.
            centered_features = sorted_features - voxel_features.index_select(0, sorted_inverse)
            voxel_heterogeneity = self._segment_reduce(
                (centered_features ** 2).sum(dim=-1),
                "mean",
                counts,
            ).to(dtype=torch.float32)
        else:
            # Voxel-VTC-VisionZip never consumes heterogeneity.  Keep a
            # shape-compatible field for the shared state contract without
            # allocating the per-token centered feature tensor.
            voxel_heterogeneity = torch.zeros(
                num_voxels,
                device=flat_features.device,
                dtype=torch.float32,
            )

        sorted_frame_ids = frame_ids.index_select(0, sort_order)
        frame_starts = torch.ones(
            input_tokens,
            device=sorted_frame_ids.device,
            dtype=torch.float32,
        )
        if input_tokens > 1:
            frame_starts[1:] = (
                (sorted_inverse[1:] != sorted_inverse[:-1])
                | (sorted_frame_ids[1:] != sorted_frame_ids[:-1])
            ).to(dtype=torch.float32)
        frame_counts = self._segment_reduce(frame_starts, "sum", counts).to(dtype=torch.long)
        voxel_view_support = frame_counts.to(dtype=torch.float32) / float(num_frames)

        def reorder(values: torch.Tensor) -> torch.Tensor:
            return values.index_select(0, group_order)

        representative_patch_positions = flat_patch_positions.index_select(0, representative_indices)
        representative_frame_ids = frame_ids.index_select(0, representative_indices)
        representative_order_ids = order_ids.index_select(0, representative_indices)

        unique_frame_ids = sorted_frame_ids[frame_starts.to(dtype=torch.bool)]
        if materialize_group_metadata:
            # Keep the public Python-list contract for callers that request
            # legacy metadata.  VTC-VisionZip disables this branch because it
            # only consumes the tensor fields below.
            sort_order_cpu = sort_order.detach().cpu().tolist()
            counts_cpu = counts.detach().cpu().tolist()
            group_order_cpu = group_order.detach().cpu().tolist()
            groups_by_unique_key: List[List[int]] = []
            offset = 0
            for count in counts_cpu:
                groups_by_unique_key.append(sort_order_cpu[offset : offset + count])
                offset += count
            voxel_groups = [groups_by_unique_key[idx] for idx in group_order_cpu]

            frame_id_segments = torch.split(unique_frame_ids, frame_counts.detach().cpu().tolist())
            voxel_group_frame_ids = [frame_id_segments[idx] for idx in group_order_cpu]
        else:
            voxel_groups = []
            voxel_group_frame_ids = []

        reordered_grouping_coordinates = reorder(voxel_grouping_coordinates)
        if compute_boundaryness:
            voxel_boundaryness = self._compute_boundaryness(
                voxel_features=reorder(voxel_features),
                voxel_coordinates=reordered_grouping_coordinates,
                voxel_size=voxel_size,
                knn_k=getattr(self, "_boundary_knn_k", 6),
            )
        else:
            voxel_boundaryness = torch.zeros(
                num_voxels,
                device=flat_features.device,
                dtype=torch.float32,
            )

        return {
            "input_tokens": input_tokens,
            "num_frames": num_frames,
            "voxel_groups": voxel_groups,
            "voxel_features": reorder(voxel_features),
            "voxel_coordinates": reorder(voxel_coordinates).to(dtype=flat_coordinates.dtype),
            "voxel_grouping_coordinates": reordered_grouping_coordinates.to(
                dtype=flat_grouping_coordinates.dtype
            ),
            "voxel_patch_positions": reorder(representative_patch_positions),
            "voxel_frame_ids": reorder(representative_frame_ids),
            "voxel_order_ids": reorder(representative_order_ids),
            "voxel_attention": reorder(voxel_attention),
            "voxel_view_support": reorder(voxel_view_support),
            "voxel_heterogeneity": reorder(voxel_heterogeneity),
            "voxel_boundaryness": voxel_boundaryness,
            "voxel_occupancies": reorder(counts.to(dtype=torch.float32)),
            "voxel_group_frame_ids": voxel_group_frame_ids,
        }

    def _compute_boundaryness(
        self,
        *,
        voxel_features: torch.Tensor,
        voxel_coordinates: torch.Tensor,
        voxel_size: float,
        knn_k: int,
    ) -> torch.Tensor:
        num_voxels = voxel_features.shape[0]
        if num_voxels <= 1:
            return torch.zeros(num_voxels, device=voxel_features.device, dtype=torch.float32)

        k = min(knn_k, num_voxels - 1)
        if k <= 0:
            return torch.zeros(num_voxels, device=voxel_features.device, dtype=torch.float32)

        normalized_features = F.normalize(voxel_features.float(), dim=-1)
        distance_matrix = torch.cdist(voxel_coordinates.float(), voxel_coordinates.float(), p=2)
        distance_matrix.fill_diagonal_(float("inf"))
        neighbor_distances, neighbor_indices = torch.topk(distance_matrix, k=k, largest=False, dim=-1)

        center_features = normalized_features.unsqueeze(1).expand(-1, k, -1)
        neighbor_features = normalized_features.index_select(0, neighbor_indices.reshape(-1)).view(num_voxels, k, -1)
        contrast = 1.0 - (center_features * neighbor_features).sum(dim=-1)

        tau = max(float(voxel_size) ** 2, 1e-6)
        geo_affinity = torch.exp(-(neighbor_distances ** 2) / tau)
        return (contrast * geo_affinity).mean(dim=-1)

    def _select_topk_indices(self, scores: torch.Tensor, order_ids: torch.Tensor, k: int) -> torch.Tensor:
        sorted_indices = sorted(
            range(scores.shape[0]),
            key=lambda idx: (-float(scores[idx].item()), int(order_ids[idx].item())),
        )
        return torch.tensor(sorted_indices[:k], device=scores.device, dtype=torch.long)

    def _select_topk_indices_vectorized(
        self,
        scores: torch.Tensor,
        order_ids: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Select scores with the reference sort's exact tie-break rule.

        The reference implementation sorts Python scalars by
        ``(-score, order_id)``.  Voxel groups are normally finite, so two
        stable tensor sorts are equivalent: first establish ascending
        ``order_id`` for ties, then stably sort by descending score.  Preserve
        the reference path for non-finite values because Python's ordering of
        NaNs is not defined by tensor sorting in the same way.
        """
        if k <= 0 or scores.numel() == 0:
            return torch.empty(0, device=scores.device, dtype=torch.long)
        k = min(int(k), int(scores.numel()))
        if not bool(torch.isfinite(scores).all().item()):
            return self._select_topk_indices(scores, order_ids, k)

        by_order = torch.argsort(order_ids, stable=True)
        by_score = torch.argsort(
            scores.index_select(0, by_order),
            descending=True,
            stable=True,
        )
        return by_order.index_select(0, by_score[:k])

    def _gather_outputs(
        self,
        *,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        patch_positions: torch.Tensor,
        frame_ids: torch.Tensor,
        order_ids: torch.Tensor,
        num_frames: int,
        input_tokens: int,
        extra_metadata: Dict[str, Any],
    ) -> CompressorOutput:
        return self._finalize_outputs(
            features=features,
            coordinates=coordinates,
            patch_positions=patch_positions,
            frame_ids=frame_ids,
            order_ids=order_ids,
            num_frames=num_frames,
            input_tokens=input_tokens,
            extra_metadata=extra_metadata,
        )


@register_compressor("voxel_geosem_anchorprune")
class VoxelGeoSemAnchorPruneCompressor(_VoxelGeoSemMixin, BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        prune_config = VoxelGeoSemAnchorPruneConfig(**config) if isinstance(config, dict) else config
        if prune_config.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {prune_config.voxel_size}.")
        self._validate_target_spec(prune_config.target_tokens, prune_config.target_keep_ratio)
        self._validate_attention_reduce(prune_config.attention_reduce)
        self._validate_newline_strategy(prune_config.newline_strategy)
        super().__init__(prune_config)
        self.prune_config = prune_config

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        voxel_state = self._build_voxel_state(
            features=features,
            coordinates=coordinates,
            grouping_coordinates=kwargs.get("grouping_coordinates"),
            attentions=attn_weights,
            num_frames=num_frames,
            frame_shape=frame_shape,
            voxel_size=float(self.prune_config.voxel_size),
            attention_reduce=self.prune_config.attention_reduce,
        )

        input_tokens = int(voxel_state["input_tokens"])
        num_voxels = int(voxel_state["voxel_features"].shape[0])
        target_tokens = self._resolve_target_tokens(
            input_tokens,
            self.prune_config.target_tokens,
            self.prune_config.target_keep_ratio,
        )
        anchor_count = min(num_voxels, target_tokens)
        score = self._compute_joint_anchor_score(
            voxel_state=voxel_state,
            alpha=float(self.prune_config.alpha),
            beta=float(self.prune_config.beta),
            gamma=float(self.prune_config.gamma),
            delta=float(self.prune_config.delta),
        )
        anchor_indices = self._select_topk_indices(
            score,
            voxel_state["voxel_order_ids"],
            anchor_count,
        )

        output = self._gather_outputs(
            features=voxel_state["voxel_features"][anchor_indices],
            coordinates=voxel_state["voxel_coordinates"][anchor_indices],
            patch_positions=voxel_state["voxel_patch_positions"][anchor_indices],
            frame_ids=voxel_state["voxel_frame_ids"][anchor_indices],
            order_ids=voxel_state["voxel_order_ids"][anchor_indices],
            num_frames=int(voxel_state["num_frames"]),
            input_tokens=input_tokens,
            extra_metadata={
                "newline_strategy": self.prune_config.newline_strategy,
                "voxel_method": "geosem_anchorprune",
                "voxel_size": float(self.prune_config.voxel_size),
                "attention_reduce": self.prune_config.attention_reduce,
                "selection_method": "joint_semantic_geometry_topk",
                "alpha": float(self.prune_config.alpha),
                "beta": float(self.prune_config.beta),
                "gamma": float(self.prune_config.gamma),
                "delta": float(self.prune_config.delta),
                "num_voxels_before_anchor": num_voxels,
                "num_anchors": int(anchor_count),
                "anchor_keep_ratio": float(anchor_count / num_voxels) if num_voxels > 0 else 1.0,
                "mean_view_support": float(voxel_state["voxel_view_support"].mean().item()) if num_voxels > 0 else 0.0,
                "mean_attention_score": float(voxel_state["voxel_attention"].mean().item()) if num_voxels > 0 else 0.0,
                "mean_boundaryness": float(voxel_state["voxel_boundaryness"].mean().item()) if num_voxels > 0 else 0.0,
                "mean_heterogeneity": float(voxel_state["voxel_heterogeneity"].mean().item()) if num_voxels > 0 else 0.0,
            },
        )
        return output

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["coordinates"] = True
        required["attn_weights"] = True
        return required

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False


@register_compressor("voxel_geosem_anchormerge")
class VoxelGeoSemAnchorMergeCompressor(_VoxelGeoSemMixin, BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        merge_config = VoxelGeoSemAnchorMergeConfig(**config) if isinstance(config, dict) else config
        if merge_config.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {merge_config.voxel_size}.")
        self._validate_target_spec(merge_config.target_tokens, merge_config.target_keep_ratio)
        self._validate_attention_reduce(merge_config.attention_reduce)
        self._validate_newline_strategy(merge_config.newline_strategy)
        if merge_config.knn_k <= 0:
            raise ValueError(f"knn_k must be positive, got {merge_config.knn_k}.")
        if merge_config.anchor_knn <= 0:
            raise ValueError(f"anchor_knn must be positive, got {merge_config.anchor_knn}.")
        super().__init__(merge_config)
        self.merge_config = merge_config
        self._boundary_knn_k = int(merge_config.knn_k)

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        voxel_state = self._build_voxel_state(
            features=features,
            coordinates=coordinates,
            grouping_coordinates=kwargs.get("grouping_coordinates"),
            attentions=attn_weights,
            num_frames=num_frames,
            frame_shape=frame_shape,
            voxel_size=float(self.merge_config.voxel_size),
            attention_reduce=self.merge_config.attention_reduce,
        )

        input_tokens = int(voxel_state["input_tokens"])
        num_frames = int(voxel_state["num_frames"])
        num_voxels = int(voxel_state["voxel_features"].shape[0])
        target_tokens = self._resolve_target_tokens(
            input_tokens,
            self.merge_config.target_tokens,
            self.merge_config.target_keep_ratio,
        )
        anchor_count = min(num_voxels, target_tokens)
        score = self._compute_joint_anchor_score(
            voxel_state=voxel_state,
            alpha=float(self.merge_config.alpha),
            beta=float(self.merge_config.beta),
            gamma=float(self.merge_config.gamma),
            delta=float(self.merge_config.delta),
        )

        anchor_indices = self._select_topk_indices(score, voxel_state["voxel_order_ids"], anchor_count)
        residual_mask = torch.ones(num_voxels, device=score.device, dtype=torch.bool)
        residual_mask[anchor_indices] = False
        residual_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()
        if residual_indices.numel() > 0:
            residual_scores = score.index_select(0, residual_indices)
            sorted_residual = residual_scores.argsort(descending=True)
            residual_indices = residual_indices.index_select(0, sorted_residual)

        anchor_features = voxel_state["voxel_features"].index_select(0, anchor_indices).clone()
        anchor_coordinates = voxel_state["voxel_coordinates"].index_select(0, anchor_indices).clone()
        anchor_grouping_coordinates = voxel_state["voxel_grouping_coordinates"].index_select(0, anchor_indices).clone()
        anchor_patch_positions = voxel_state["voxel_patch_positions"].index_select(0, anchor_indices).clone()
        anchor_frame_ids = voxel_state["voxel_frame_ids"].index_select(0, anchor_indices).clone()
        anchor_order_ids = voxel_state["voxel_order_ids"].index_select(0, anchor_indices).clone()
        anchor_scores = score.index_select(0, anchor_indices).clone()
        anchor_view_support = voxel_state["voxel_view_support"].index_select(0, anchor_indices).clone()
        anchor_occupancies = voxel_state["voxel_occupancies"].index_select(0, anchor_indices).clone()

        residual_merged = 0
        tau = max(float(self.merge_config.voxel_size) ** 2, 1e-6)
        for residual_idx in residual_indices.tolist():
            if anchor_features.shape[0] == 0:
                break

            residual_feature = voxel_state["voxel_features"][residual_idx]
            residual_coordinate = voxel_state["voxel_coordinates"][residual_idx]
            residual_grouping_coordinate = voxel_state["voxel_grouping_coordinates"][residual_idx]
            residual_score = score[residual_idx]
            residual_view_support = voxel_state["voxel_view_support"][residual_idx]
            residual_occupancy = voxel_state["voxel_occupancies"][residual_idx]

            anchor_distances = torch.norm(
                anchor_grouping_coordinates.float() - residual_grouping_coordinate.float().unsqueeze(0),
                dim=-1,
            )
            candidate_count = min(int(self.merge_config.anchor_knn), int(anchor_features.shape[0]))
            nearest_anchor_positions = torch.topk(anchor_distances, k=candidate_count, largest=False).indices

            candidate_features = anchor_features.index_select(0, nearest_anchor_positions)
            candidate_grouping_coordinates = anchor_grouping_coordinates.index_select(0, nearest_anchor_positions)
            candidate_view_support = anchor_view_support.index_select(0, nearest_anchor_positions)
            cosine_scores = F.cosine_similarity(
                F.normalize(candidate_features.float(), dim=-1),
                F.normalize(residual_feature.float().unsqueeze(0).expand_as(candidate_features), dim=-1),
                dim=-1,
            )
            geo_affinity = torch.exp(
                -(((candidate_grouping_coordinates.float() - residual_grouping_coordinate.float().unsqueeze(0)) ** 2).sum(dim=-1))
                / tau
            )
            match_scores = (
                float(self.merge_config.lambda_match) * cosine_scores
                + float(self.merge_config.mu_match) * geo_affinity
                + float(self.merge_config.nu_match) * candidate_view_support
            )
            best_local_idx = nearest_anchor_positions[match_scores.argmax()]

            anchor_weight = anchor_scores[best_local_idx] * (
                1.0 + float(self.merge_config.rho_weight) * anchor_view_support[best_local_idx]
            ) * torch.log1p(anchor_occupancies[best_local_idx])
            residual_weight = residual_score * (
                1.0 + float(self.merge_config.rho_weight) * residual_view_support
            ) * torch.log1p(residual_occupancy)

            total_weight = anchor_weight + float(self.merge_config.eta_merge) * residual_weight
            if total_weight.item() > 0:
                merged_feature = (
                    anchor_weight * anchor_features[best_local_idx]
                    + float(self.merge_config.eta_merge) * residual_weight * residual_feature
                ) / total_weight
                merged_coordinate = (
                    anchor_weight * anchor_coordinates[best_local_idx].float()
                    + float(self.merge_config.eta_merge) * residual_weight * residual_coordinate.float()
                ) / total_weight
                merged_grouping_coordinate = (
                    anchor_weight * anchor_grouping_coordinates[best_local_idx].float()
                    + float(self.merge_config.eta_merge) * residual_weight * residual_grouping_coordinate.float()
                ) / total_weight
                anchor_features[best_local_idx] = merged_feature.to(dtype=anchor_features.dtype)
                anchor_coordinates[best_local_idx] = merged_coordinate.to(dtype=anchor_coordinates.dtype)
                anchor_grouping_coordinates[best_local_idx] = merged_grouping_coordinate.to(
                    dtype=anchor_grouping_coordinates.dtype
                )
            anchor_occupancies[best_local_idx] = anchor_occupancies[best_local_idx] + residual_occupancy
            residual_merged += 1

        return self._gather_outputs(
            features=anchor_features,
            coordinates=anchor_coordinates,
            patch_positions=anchor_patch_positions,
            frame_ids=anchor_frame_ids,
            order_ids=anchor_order_ids,
            num_frames=num_frames,
            input_tokens=input_tokens,
            extra_metadata={
                "newline_strategy": self.merge_config.newline_strategy,
                "voxel_method": "geosem_anchormerge",
                "voxel_size": float(self.merge_config.voxel_size),
                "attention_reduce": self.merge_config.attention_reduce,
                "num_voxels_before_anchor": num_voxels,
                "num_anchors": int(anchor_count),
                "num_residual_merged": int(residual_merged),
                "anchor_keep_ratio": float(anchor_count / num_voxels) if num_voxels > 0 else 1.0,
                "mean_view_support": float(voxel_state["voxel_view_support"].mean().item()) if num_voxels > 0 else 0.0,
                "mean_boundaryness": float(voxel_state["voxel_boundaryness"].mean().item()) if num_voxels > 0 else 0.0,
                "mean_heterogeneity": float(voxel_state["voxel_heterogeneity"].mean().item()) if num_voxels > 0 else 0.0,
            },
        )

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["coordinates"] = True
        required["attn_weights"] = True
        return required

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False
