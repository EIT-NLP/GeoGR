import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import BaseCompressor, CompressorConfig, CompressorOutput
from ..registry import register_compressor
from ..voxel_geosem.compressor import _VoxelGeoSemMixin


@dataclass
class _VoxelVTCPostConfig(CompressorConfig):
    voxel_size: float = 0.1
    target_tokens: Optional[int] = None
    target_keep_ratio: Optional[float] = None
    attention_reduce: str = "max"
    newline_strategy: str = "grid_drop"


@dataclass
class VoxelVTCRandomConfig(_VoxelVTCPostConfig):
    random_seed: int = 0


@dataclass
class VoxelVTCAttnPruneConfig(_VoxelVTCPostConfig):
    pass


@dataclass
class VoxelVTCVisPrunerConfig(_VoxelVTCPostConfig):
    important_ratio: float = 0.5


@dataclass
class VoxelVTCVisionZipConfig(_VoxelVTCPostConfig):
    dominant_ratio: float = 0.75
    residual_merge: bool = True
    coverage_rule: str = "morton"
    random_seed: int = 0


@dataclass
class VoxelVTCToMeConfig(_VoxelVTCPostConfig):
    local_k: int = 8
    max_rounds: int = 8


class _VoxelVTCPostMixin(_VoxelGeoSemMixin):
    _COVERAGE_RULES = frozenset({"morton", "fps", "random", "feature_cluster"})

    def _validate_base_config(self, config: _VoxelVTCPostConfig) -> None:
        if not math.isfinite(float(config.voxel_size)) or float(config.voxel_size) <= 0:
            raise ValueError(f"voxel_size must be positive, got {config.voxel_size}.")
        self._validate_target_spec(config.target_tokens, config.target_keep_ratio)
        self._validate_attention_reduce(config.attention_reduce)
        if config.newline_strategy not in {"frame_newline", "grid_drop", "one_token", "no_token"}:
            raise ValueError(
                "VoxelVTC post newline_strategy must be one of "
                "'frame_newline', 'grid_drop', 'one_token', or 'no_token', "
                f"got {config.newline_strategy!r}."
            )

    def _zero_attention(
        self,
        *,
        features: torch.Tensor,
        num_frames: int,
        frame_shape: Tuple[int, int],
    ) -> torch.Tensor:
        _, total_tokens, _ = self._normalize_features(features, num_frames).shape
        tokens_per_frame = frame_shape[0] * frame_shape[1]
        if total_tokens != num_frames * tokens_per_frame:
            raise ValueError(f"Expected {num_frames * tokens_per_frame} tokens, got {total_tokens}.")
        return torch.zeros(num_frames, tokens_per_frame, device=features.device, dtype=torch.float32)

    def _build_vtc_post_state(
        self,
        *,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        grouping_coordinates: Optional[torch.Tensor],
        attn_weights: Optional[torch.Tensor],
        num_frames: int,
        frame_shape: Tuple[int, int],
        voxel_size: float,
        attention_reduce: str,
        vectorized_state: bool = False,
    ) -> Dict[str, Any]:
        if attn_weights is None:
            attn_weights = self._zero_attention(features=features, num_frames=num_frames, frame_shape=frame_shape)
        return self._build_voxel_state(
            features=features,
            coordinates=coordinates,
            grouping_coordinates=grouping_coordinates,
            attentions=attn_weights,
            num_frames=num_frames,
            frame_shape=frame_shape,
            voxel_size=voxel_size,
            attention_reduce=attention_reduce,
            compute_boundaryness=False,
            vectorized=vectorized_state,
            # The deployed VisionZip path only consumes tensor voxel state.
            # Keep the legacy Python group metadata and heterogeneity statistic
            # for the reference/non-vectorized path and skip them in the
            # optimized path.
            materialize_group_metadata=not vectorized_state,
            compute_heterogeneity=not vectorized_state,
        )

    def _spatial_sort_indices_reference(self, coordinates: torch.Tensor, voxel_size: float) -> torch.Tensor:
        if coordinates.shape[0] == 0:
            return torch.empty(0, device=coordinates.device, dtype=torch.long)
        voxel_indices = self._discretize_voxel_indices(coordinates, voxel_size)
        keys = voxel_indices.detach().cpu().tolist()
        offset = -int(voxel_indices.min().item()) if voxel_indices.numel() > 0 and int(voxel_indices.min().item()) < 0 else 0
        max_value = int((voxel_indices + offset).max().item()) if voxel_indices.numel() > 0 else 0
        max_bits = max(1, max_value.bit_length())

        def morton_code(key: List[int]) -> int:
            x, y, z = [int(v) + offset for v in key]
            code = 0
            for bit in range(max_bits):
                code |= ((x >> bit) & 1) << (3 * bit + 2)
                code |= ((y >> bit) & 1) << (3 * bit + 1)
                code |= ((z >> bit) & 1) << (3 * bit)
            return code

        order = sorted(range(len(keys)), key=lambda idx: (morton_code(keys[idx]), keys[idx]))
        return torch.tensor(order, device=coordinates.device, dtype=torch.long)

    def _spatial_sort_indices(self, coordinates: torch.Tensor, voxel_size: float) -> torch.Tensor:
        """Return the historical Morton order without a host-side sort.

        The reference implementation sorts ``(morton_code, x, y, z)`` using
        Python integers.  Stable tensor sorts reproduce the same lexicographic
        tie-break while keeping the voxel coordinates on their original
        device.  Very large coordinate ranges fall back to the reference path
        because Python integers do not have the int64 Morton-code limit.
        """
        if coordinates.shape[0] == 0:
            return torch.empty(0, device=coordinates.device, dtype=torch.long)

        voxel_indices = self._discretize_voxel_indices(coordinates, voxel_size)
        if voxel_indices.shape[-1] != 3:
            return self._spatial_sort_indices_reference(coordinates, voxel_size)

        min_value = int(voxel_indices.min().item())
        offset = -min_value if min_value < 0 else 0
        shifted = voxel_indices + offset
        max_value = int(shifted.max().item()) if shifted.numel() > 0 else 0
        max_bits = max(1, max_value.bit_length())
        if max_bits > 21:
            return self._spatial_sort_indices_reference(coordinates, voxel_size)

        morton = torch.zeros(
            shifted.shape[0],
            device=shifted.device,
            dtype=torch.long,
        )
        for bit in range(max_bits):
            morton |= ((shifted[:, 0] >> bit) & 1) << (3 * bit + 2)
            morton |= ((shifted[:, 1] >> bit) & 1) << (3 * bit + 1)
            morton |= ((shifted[:, 2] >> bit) & 1) << (3 * bit)

        # Stable passes implement the same secondary key as Python's
        # ``(morton, [x, y, z])`` tuple: sort z, then y, then x, then Morton.
        order = torch.arange(shifted.shape[0], device=shifted.device, dtype=torch.long)
        for component in (2, 1, 0):
            component_order = torch.argsort(
                shifted.index_select(0, order)[:, component],
                stable=True,
            )
            order = order.index_select(0, component_order)
        morton_order = torch.argsort(morton.index_select(0, order), stable=True)
        return order.index_select(0, morton_order)

    def _select_diverse_indices(
        self,
        candidate_features: torch.Tensor,
        candidate_scores: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        if target_count <= 0 or candidate_features.shape[0] == 0:
            return torch.empty(0, device=candidate_features.device, dtype=torch.long)
        if target_count >= candidate_features.shape[0]:
            return torch.arange(candidate_features.shape[0], device=candidate_features.device, dtype=torch.long)

        raw_features = candidate_features.float()
        normalized = F.normalize(raw_features, dim=-1, eps=1e-12)
        first_idx = int(candidate_scores.argmax().item())
        selected = [first_idx]
        available = torch.ones(candidate_features.shape[0], device=candidate_features.device, dtype=torch.bool)
        available[first_idx] = False
        remaining = int(candidate_features.shape[0] - 1)
        min_distance = torch.full(
            (candidate_features.shape[0],),
            float("inf"),
            device=candidate_features.device,
            dtype=torch.float32,
        )
        raw_min_distance = torch.full_like(min_distance, float("inf"))

        while len(selected) < target_count and remaining > 0:
            latest = normalized[selected[-1]]
            distance = 1.0 - normalized @ latest
            min_distance = torch.minimum(min_distance, distance)
            raw_latest = raw_features[selected[-1]]
            raw_distance = torch.linalg.vector_norm(raw_features - raw_latest, dim=-1)
            raw_min_distance = torch.minimum(raw_min_distance, raw_distance)
            best_idx = int(
                torch.argmax(
                    (min_distance + 1e-6 * raw_min_distance).masked_fill(
                        ~available, float("-inf")
                    )
                ).item()
            )
            selected.append(best_idx)
            available[best_idx] = False
            remaining -= 1

        return torch.tensor(selected, device=candidate_features.device, dtype=torch.long)

    def _select_fps_indices(
        self,
        candidate_coordinates: torch.Tensor,
        candidate_scores: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        """Deterministic farthest-point sampling over voxel coordinates.

        The highest-attention candidate is used as the first point so that the
        coverage branch remains deterministic and starts from the same salient
        region as the dominant branch. Ties are resolved by candidate order.
        """
        if target_count <= 0 or candidate_coordinates.shape[0] == 0:
            return torch.empty(0, device=candidate_coordinates.device, dtype=torch.long)
        if target_count >= candidate_coordinates.shape[0]:
            return torch.arange(candidate_coordinates.shape[0], device=candidate_coordinates.device, dtype=torch.long)

        coordinates = torch.nan_to_num(candidate_coordinates.float(), nan=0.0, posinf=0.0, neginf=0.0)
        first_idx = int(candidate_scores.argmax().item())
        selected = [first_idx]
        available = torch.ones(candidate_coordinates.shape[0], device=candidate_coordinates.device, dtype=torch.bool)
        available[first_idx] = False
        remaining = int(candidate_coordinates.shape[0] - 1)
        min_distance = torch.full(
            (candidate_coordinates.shape[0],),
            float("inf"),
            device=candidate_coordinates.device,
            dtype=torch.float32,
        )

        while len(selected) < target_count and remaining > 0:
            last = coordinates[selected[-1]]
            distance = ((coordinates - last) ** 2).sum(dim=-1)
            min_distance = torch.minimum(min_distance, distance)
            scores = min_distance.masked_fill(~available, float("-inf"))
            next_idx = int(torch.argmax(scores).item())
            selected.append(next_idx)
            available[next_idx] = False
            remaining -= 1

        return torch.tensor(selected, device=candidate_coordinates.device, dtype=torch.long)

    def _select_contextual_indices(
        self,
        *,
        coverage_rule: str,
        candidate_coordinates: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_scores: torch.Tensor,
        target_count: int,
        voxel_size: float,
        random_seed: int,
    ) -> torch.Tensor:
        """Select contextual anchors using one of the documented rules.

        Returned indices are local to the candidate tensors. Keeping this
        operation separate from residual merging makes the budget accounting
        identical across all design-choice variants.
        """
        if target_count <= 0 or candidate_coordinates.shape[0] == 0:
            return torch.empty(0, device=candidate_coordinates.device, dtype=torch.long)
        target_count = min(int(target_count), int(candidate_coordinates.shape[0]))
        rule = str(coverage_rule).strip().lower()
        if rule == "morton":
            spatial_order = self._spatial_sort_indices(candidate_coordinates, voxel_size)
            step = max(1, int(spatial_order.numel() // target_count))
            local_positions = torch.arange(
                0,
                spatial_order.numel(),
                step,
                device=spatial_order.device,
            )[:target_count]
            return spatial_order.index_select(0, local_positions)
        if rule == "fps":
            return self._select_fps_indices(candidate_coordinates, candidate_scores, target_count)
        if rule == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(random_seed))
            return torch.randperm(
                candidate_coordinates.shape[0],
                generator=generator,
            )[:target_count].to(device=candidate_coordinates.device)
        if rule == "feature_cluster":
            return self._select_diverse_indices(candidate_features, candidate_scores, target_count)
        raise ValueError(
            "coverage_rule must be one of 'morton', 'fps', 'random', or 'feature_cluster', "
            f"got {coverage_rule!r}."
        )

    def _gather_selected_state(
        self,
        *,
        state: Dict[str, Any],
        selected_indices: torch.Tensor,
        extra_metadata: Dict[str, Any],
        newline_strategy: Optional[str] = None,
        selected_order_ids: Optional[torch.Tensor] = None,
    ) -> CompressorOutput:
        extra_metadata = {
            "newline_strategy": newline_strategy or self.config.newline_strategy,
            **extra_metadata,
        }
        if selected_order_ids is None:
            selected_order_ids = state["voxel_order_ids"].index_select(0, selected_indices)
        return self._gather_outputs(
            features=state["voxel_features"].index_select(0, selected_indices),
            coordinates=state["voxel_coordinates"].index_select(0, selected_indices),
            patch_positions=state["voxel_patch_positions"].index_select(0, selected_indices),
            frame_ids=state["voxel_frame_ids"].index_select(0, selected_indices),
            order_ids=selected_order_ids,
            num_frames=int(state["num_frames"]),
            input_tokens=int(state["input_tokens"]),
            extra_metadata=extra_metadata,
        )

    def _gather_scored_selected_state(
        self,
        *,
        state: Dict[str, Any],
        selected_indices: torch.Tensor,
        selected_scores: torch.Tensor,
        extra_metadata: Dict[str, Any],
        selected_priority: Optional[torch.Tensor] = None,
    ) -> CompressorOutput:
        """Attach projector metadata in the exact emitted patch order.

        ``selected_priority`` records the order in which Stage-I selected the
        retained voxel tokens (dominant tokens first, followed by contextual
        anchors).  It is kept separate from ``retained_order_ids`` because the
        latter describes serialization order, not selection priority.
        """
        if selected_scores.dim() != 1 or selected_scores.numel() != selected_indices.numel():
            raise ValueError("Projector scores must align one-to-one with selected patch tokens.")
        if selected_priority is None:
            selected_priority = torch.arange(
                selected_indices.numel(),
                device=selected_indices.device,
                dtype=torch.long,
            )
        if selected_priority.dim() != 1 or selected_priority.numel() != selected_indices.numel():
            raise ValueError("Projector selection priority must align one-to-one with selected patches.")
        selected_priority = selected_priority.to(device=selected_scores.device, dtype=torch.long)
        if int(torch.unique(selected_priority).numel()) != int(selected_priority.numel()):
            raise ValueError("Projector selection priority must be unique for retained patches.")
        selected_order_ids = state["voxel_order_ids"].index_select(0, selected_indices)
        if int(torch.unique(selected_order_ids).numel()) != int(selected_order_ids.numel()):
            raise ValueError("Selected projector patch order IDs must be unique.")

        # ``_finalize_outputs`` stably sorts these same order IDs before
        # serialization. Reuse that tensor permutation directly instead of
        # synchronizing to the host and rebuilding a Python order-to-index map.
        # This only changes metadata alignment overhead; the emitted order is
        # exactly the historical order_ids.argsort(stable=True) order.
        emitted_local = torch.argsort(selected_order_ids, stable=True)

        output = self._gather_selected_state(
            state=state,
            selected_indices=selected_indices,
            extra_metadata=extra_metadata,
            selected_order_ids=selected_order_ids,
        )
        emitted_order_ids = output.metadata.get("retained_order_ids")
        if not isinstance(emitted_order_ids, list) or len(emitted_order_ids) != int(selected_scores.numel()):
            raise ValueError("Cannot align projector scores with emitted patch order.")
        expected_order_ids = selected_order_ids.index_select(0, emitted_local).tolist()
        if emitted_order_ids != expected_order_ids:
            raise ValueError("Emitted projector patch order differs from the stable order-id permutation.")
        output.metadata["projector_patch_scores"] = (
            selected_scores.float().index_select(0, emitted_local).detach()
        )
        output.metadata["projector_patch_priority"] = (
            selected_priority.index_select(0, emitted_local).detach()
        )
        # Preserve metric grouping coordinates separately from Video3D's mRoPE
        # coordinates. Discretized mRoPE positions are not valid for the
        # projector-style Morton grouping reused inside the LLM.
        output.metadata["projector_patch_coordinates"] = (
            state["voxel_grouping_coordinates"]
            .index_select(0, selected_indices)
            .index_select(0, emitted_local)
            .detach()
        )
        output.metadata["projector_score_definition"] = (
            f"vtc_visionzip_voxel_attention_{self.config.attention_reduce}"
        )
        return output

    def _collect_common_metadata(
        self,
        state: Dict[str, Any],
        target_tokens: int,
        nominal_target_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        num_voxels = int(state["voxel_features"].shape[0])
        input_tokens = int(state["input_tokens"])
        nominal_target_tokens = int(target_tokens if nominal_target_tokens is None else nominal_target_tokens)
        return {
            "num_voxels_before_post": num_voxels,
            "target_tokens": int(target_tokens),
            "input_tokens_before_post": input_tokens,
            "nominal_target_tokens": nominal_target_tokens,
            "budget_limited_by_voxel_count": bool(num_voxels < nominal_target_tokens),
            "mean_view_support": float(state["voxel_view_support"].mean().item()) if num_voxels > 0 else 0.0,
            "mean_attention_score": float(state["voxel_attention"].mean().item()) if num_voxels > 0 else 0.0,
            "mean_occupancy": float(state["voxel_occupancies"].mean().item()) if num_voxels > 0 else 0.0,
        }


@register_compressor("voxel_vtc_random")
class VoxelVTCRandomCompressor(_VoxelVTCPostMixin, BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        random_config = VoxelVTCRandomConfig(**config) if isinstance(config, dict) else config
        self._validate_base_config(random_config)
        super().__init__(random_config)
        self.random_config = random_config

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        state = self._build_vtc_post_state(
            features=features,
            coordinates=coordinates,
            grouping_coordinates=kwargs.get("grouping_coordinates"),
            attn_weights=attn_weights,
            num_frames=num_frames,
            frame_shape=frame_shape,
            voxel_size=float(self.random_config.voxel_size),
            attention_reduce=self.random_config.attention_reduce,
        )
        num_voxels = int(state["voxel_features"].shape[0])
        target_tokens = min(
            num_voxels,
            self._resolve_target_tokens(
                int(state["input_tokens"]),
                self.random_config.target_tokens,
                self.random_config.target_keep_ratio,
            ),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_config.random_seed))
        selected_indices = torch.randperm(num_voxels, generator=generator)[:target_tokens].to(device=state["voxel_features"].device)
        return self._gather_selected_state(
            state=state,
            selected_indices=selected_indices,
            extra_metadata={
                "voxel_method": "vtc_random",
                "voxel_size": float(self.random_config.voxel_size),
                "random_seed": int(self.random_config.random_seed),
                **self._collect_common_metadata(state, target_tokens),
            },
        )

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["coordinates"] = True
        return required


@register_compressor("voxel_vtc_attn_prune")
class VoxelVTCAttnPruneCompressor(_VoxelVTCPostMixin, BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        prune_config = VoxelVTCAttnPruneConfig(**config) if isinstance(config, dict) else config
        self._validate_base_config(prune_config)
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
        state = self._build_vtc_post_state(
            features=features,
            coordinates=coordinates,
            grouping_coordinates=kwargs.get("grouping_coordinates"),
            attn_weights=attn_weights,
            num_frames=num_frames,
            frame_shape=frame_shape,
            voxel_size=float(self.prune_config.voxel_size),
            attention_reduce=self.prune_config.attention_reduce,
        )
        num_voxels = int(state["voxel_features"].shape[0])
        target_tokens = min(
            num_voxels,
            self._resolve_target_tokens(
                int(state["input_tokens"]),
                self.prune_config.target_tokens,
                self.prune_config.target_keep_ratio,
            ),
        )
        selected_indices = self._select_topk_indices(
            state["voxel_attention"],
            state["voxel_order_ids"],
            target_tokens,
        )
        return self._gather_selected_state(
            state=state,
            selected_indices=selected_indices,
            extra_metadata={
                "voxel_method": "vtc_attn_prune",
                "voxel_size": float(self.prune_config.voxel_size),
                "attention_reduce": self.prune_config.attention_reduce,
                **self._collect_common_metadata(state, target_tokens),
            },
        )

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["coordinates"] = True
        required["attn_weights"] = True
        return required


@register_compressor("voxel_vtc_vispruner")
class VoxelVTCVisPrunerCompressor(_VoxelVTCPostMixin, BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        vispruner_config = VoxelVTCVisPrunerConfig(**config) if isinstance(config, dict) else config
        self._validate_base_config(vispruner_config)
        if not 0.0 <= vispruner_config.important_ratio <= 1.0:
            raise ValueError(f"important_ratio must be within [0, 1], got {vispruner_config.important_ratio}.")
        super().__init__(vispruner_config)
        self.vispruner_config = vispruner_config

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        state = self._build_vtc_post_state(
            features=features,
            coordinates=coordinates,
            grouping_coordinates=kwargs.get("grouping_coordinates"),
            attn_weights=attn_weights,
            num_frames=num_frames,
            frame_shape=frame_shape,
            voxel_size=float(self.vispruner_config.voxel_size),
            attention_reduce=self.vispruner_config.attention_reduce,
        )
        num_voxels = int(state["voxel_features"].shape[0])
        target_tokens = min(
            num_voxels,
            self._resolve_target_tokens(
                int(state["input_tokens"]),
                self.vispruner_config.target_tokens,
                self.vispruner_config.target_keep_ratio,
            ),
        )
        if target_tokens == num_voxels:
            selected_indices = torch.arange(num_voxels, device=state["voxel_features"].device, dtype=torch.long)
            important_count = target_tokens
            diverse_count = 0
        else:
            important_count = min(target_tokens, int(round(target_tokens * float(self.vispruner_config.important_ratio))))
            if target_tokens > 0 and important_count == 0:
                important_count = 1
            important_indices = self._select_topk_indices(
                state["voxel_attention"],
                state["voxel_order_ids"],
                important_count,
            )
            residual_mask = torch.ones(num_voxels, device=state["voxel_features"].device, dtype=torch.bool)
            residual_mask[important_indices] = False
            residual_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()
            diverse_count = min(target_tokens - important_count, int(residual_indices.numel()))
            if diverse_count > 0:
                residual_scores = state["voxel_attention"].index_select(0, residual_indices)
                diverse_local = self._select_diverse_indices(
                    candidate_features=state["voxel_features"].index_select(0, residual_indices),
                    candidate_scores=residual_scores,
                    target_count=diverse_count,
                )
                diverse_indices = residual_indices.index_select(0, diverse_local)
                selected_indices = torch.cat([important_indices, diverse_indices], dim=0)
            else:
                selected_indices = important_indices
        return self._gather_selected_state(
            state=state,
            selected_indices=selected_indices,
            extra_metadata={
                "voxel_method": "vtc_vispruner",
                "voxel_size": float(self.vispruner_config.voxel_size),
                "attention_reduce": self.vispruner_config.attention_reduce,
                "important_ratio": float(self.vispruner_config.important_ratio),
                "important_token_num": int(important_count),
                "diverse_token_num": int(diverse_count),
                **self._collect_common_metadata(state, target_tokens),
            },
        )

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["coordinates"] = True
        required["attn_weights"] = True
        return required


@register_compressor("voxel_vtc_visionzip")
class VoxelVTCVisionZipCompressor(_VoxelVTCPostMixin, BaseCompressor):
    # This path is mathematically equivalent to the reference group loop but
    # avoids thousands of tiny CUDA launches in the deployed method.  Keeping
    # the switch local to VisionZip leaves the other voxel compressors on their
    # original implementation unless they explicitly opt in.
    _use_vectorized_voxel_state = True

    def __init__(self, config: Dict[str, Any]):
        visionzip_config = VoxelVTCVisionZipConfig(**config) if isinstance(config, dict) else config
        self._validate_base_config(visionzip_config)
        if not math.isfinite(float(visionzip_config.dominant_ratio)) or not 0.0 <= float(visionzip_config.dominant_ratio) <= 1.0:
            raise ValueError(f"dominant_ratio must be within [0, 1], got {visionzip_config.dominant_ratio}.")
        if not isinstance(visionzip_config.residual_merge, bool):
            raise ValueError(
                "residual_merge must be a boolean, "
                f"got {visionzip_config.residual_merge!r}."
            )
        coverage_rule = str(visionzip_config.coverage_rule).strip().lower()
        if coverage_rule not in self._COVERAGE_RULES:
            raise ValueError(
                "coverage_rule must be one of 'morton', 'fps', 'random', or 'feature_cluster', "
                f"got {visionzip_config.coverage_rule!r}."
            )
        visionzip_config.coverage_rule = coverage_rule
        try:
            random_seed = int(visionzip_config.random_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"random_seed must be a non-negative integer, got {visionzip_config.random_seed!r}.") from exc
        if random_seed < 0:
            raise ValueError(f"random_seed must be a non-negative integer, got {visionzip_config.random_seed!r}.")
        visionzip_config.random_seed = random_seed
        super().__init__(visionzip_config)
        self.visionzip_config = visionzip_config

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        state = self._build_vtc_post_state(
            features=features,
            coordinates=coordinates,
            grouping_coordinates=kwargs.get("grouping_coordinates"),
            attn_weights=attn_weights,
            num_frames=num_frames,
            frame_shape=frame_shape,
            voxel_size=float(self.visionzip_config.voxel_size),
            attention_reduce=self.visionzip_config.attention_reduce,
            vectorized_state=self._use_vectorized_voxel_state,
        )
        num_voxels = int(state["voxel_features"].shape[0])
        nominal_target_tokens = self._resolve_target_tokens(
            int(state["input_tokens"]),
            self.visionzip_config.target_tokens,
            self.visionzip_config.target_keep_ratio,
        )
        target_tokens = min(
            num_voxels,
            nominal_target_tokens,
        )
        requested_dominant_tokens = int(round(target_tokens * float(self.visionzip_config.dominant_ratio)))
        requested_contextual_tokens = target_tokens - requested_dominant_tokens
        if target_tokens == num_voxels:
            selected_indices = torch.arange(num_voxels, device=state["voxel_features"].device, dtype=torch.long)
            return self._gather_scored_selected_state(
                state=state,
                selected_indices=selected_indices,
                selected_scores=state["voxel_attention"].index_select(0, selected_indices),
                selected_priority=torch.arange(
                    selected_indices.numel(),
                    device=selected_indices.device,
                    dtype=torch.long,
                ),
                extra_metadata={
                    "voxel_method": "vtc_visionzip",
                    "voxel_size": float(self.visionzip_config.voxel_size),
                    "attention_reduce": self.visionzip_config.attention_reduce,
                    "dominant_ratio": float(self.visionzip_config.dominant_ratio),
                    "coverage_rule": self.visionzip_config.coverage_rule,
                    "random_seed": int(self.visionzip_config.random_seed),
                    "residual_merge": bool(self.visionzip_config.residual_merge),
                    "residual_merge_applied": False,
                    "num_residual_merged": 0,
                    "requested_dominant_tokens": int(requested_dominant_tokens),
                    "requested_contextual_tokens": int(requested_contextual_tokens),
                    "dominant_tokens": int(target_tokens),
                    "contextual_tokens": 0,
                    **self._collect_common_metadata(
                        state, target_tokens, nominal_target_tokens=nominal_target_tokens
                    ),
                },
            )

        dominant_tokens = requested_dominant_tokens
        if target_tokens > 1:
            dominant_tokens = min(max(dominant_tokens, 1), target_tokens - 1)
        else:
            dominant_tokens = target_tokens
        contextual_tokens = target_tokens - dominant_tokens

        dominant_indices = self._select_topk_indices_vectorized(
            state["voxel_attention"],
            state["voxel_order_ids"],
            dominant_tokens,
        )
        residual_mask = torch.ones(num_voxels, device=state["voxel_features"].device, dtype=torch.bool)
        residual_mask[dominant_indices] = False
        residual_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()
        if residual_indices.numel() == 0 or contextual_tokens <= 0:
            selected_indices = dominant_indices
            return self._gather_scored_selected_state(
                state=state,
                selected_indices=selected_indices,
                selected_scores=state["voxel_attention"].index_select(0, selected_indices),
                selected_priority=torch.arange(
                    selected_indices.numel(),
                    device=selected_indices.device,
                    dtype=torch.long,
                ),
                extra_metadata={
                    "voxel_method": "vtc_visionzip",
                    "voxel_size": float(self.visionzip_config.voxel_size),
                    "attention_reduce": self.visionzip_config.attention_reduce,
                    "dominant_ratio": float(self.visionzip_config.dominant_ratio),
                    "coverage_rule": self.visionzip_config.coverage_rule,
                    "random_seed": int(self.visionzip_config.random_seed),
                    "residual_merge": bool(self.visionzip_config.residual_merge),
                    "residual_merge_applied": False,
                    "num_residual_merged": 0,
                    "requested_dominant_tokens": int(requested_dominant_tokens),
                    "requested_contextual_tokens": int(requested_contextual_tokens),
                    "dominant_tokens": int(dominant_indices.numel()),
                    "contextual_tokens": 0,
                    **self._collect_common_metadata(
                        state, target_tokens, nominal_target_tokens=nominal_target_tokens
                    ),
                },
            )

        residual_coordinates = state["voxel_grouping_coordinates"].index_select(0, residual_indices)
        residual_features = state["voxel_features"].index_select(0, residual_indices)
        residual_scores = state["voxel_attention"].index_select(0, residual_indices)
        contextual_local_indices = self._select_contextual_indices(
            coverage_rule=self.visionzip_config.coverage_rule,
            candidate_coordinates=residual_coordinates,
            candidate_features=residual_features,
            candidate_scores=residual_scores,
            target_count=contextual_tokens,
            voxel_size=float(self.visionzip_config.voxel_size),
            random_seed=int(self.visionzip_config.random_seed),
        )
        contextual_tokens = int(contextual_local_indices.numel())
        contextual_anchor_indices = residual_indices.index_select(0, contextual_local_indices)

        contextual_mask = torch.ones(residual_indices.shape[0], device=residual_indices.device, dtype=torch.bool)
        contextual_mask[contextual_local_indices] = False
        merge_indices = residual_indices.index_select(0, torch.nonzero(contextual_mask, as_tuple=False).flatten())

        contextual_features = state["voxel_features"].index_select(0, contextual_anchor_indices).clone()
        contextual_coords = state["voxel_coordinates"].index_select(0, contextual_anchor_indices).clone()
        contextual_grouping_coords = state["voxel_grouping_coordinates"].index_select(0, contextual_anchor_indices).clone()
        contextual_patch_positions = state["voxel_patch_positions"].index_select(0, contextual_anchor_indices).clone()
        contextual_frame_ids = state["voxel_frame_ids"].index_select(0, contextual_anchor_indices).clone()
        contextual_order_ids = state["voxel_order_ids"].index_select(0, contextual_anchor_indices).clone()
        contextual_scores = state["voxel_attention"].index_select(0, contextual_anchor_indices).float().clone()

        # ``requested_contextual_tokens`` is zero at the rho=1 endpoint. The
        # effective one-token anchor retained by the clipping rule is only a
        # deterministic endpoint guard; it must not trigger residual merging.
        residual_merge_enabled = (
            bool(self.visionzip_config.residual_merge)
            and requested_contextual_tokens > 0
            and contextual_tokens > 0
        )
        residual_merged_count = 0
        if merge_indices.numel() > 0 and residual_merge_enabled:
            target_hidden = contextual_features.clone()
            tokens_to_merge = F.normalize(state["voxel_features"].index_select(0, merge_indices).float(), dim=-1)
            target_tokens_norm = F.normalize(target_hidden.float(), dim=-1)
            similarity = tokens_to_merge @ target_tokens_norm.transpose(0, 1)
            assignment = similarity.argmax(dim=-1)

            merge_scores = state["voxel_attention"].index_select(0, merge_indices).float()
            if self.visionzip_config.attention_reduce == "max":
                contextual_scores.scatter_reduce_(
                    0,
                    assignment,
                    merge_scores,
                    reduce="amax",
                    include_self=True,
                )
            else:
                score_sum = contextual_scores.clone()
                score_count = torch.ones_like(contextual_scores)
                score_sum.index_add_(0, assignment, merge_scores)
                score_count.index_add_(0, assignment, torch.ones_like(merge_scores))
                contextual_scores = score_sum / score_count.clamp_min_(1.0)

            aggregated_hidden = torch.zeros_like(target_hidden)
            counts = torch.zeros(target_hidden.shape[0], device=target_hidden.device, dtype=torch.float32)
            merge_features = state["voxel_features"].index_select(0, merge_indices)
            aggregated_hidden.index_add_(0, assignment, merge_features)
            counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float32))
            counts = counts.clamp_min_(1.0).unsqueeze(-1)
            aggregated_hidden = (aggregated_hidden / counts).to(dtype=target_hidden.dtype)
            contextual_features = (target_hidden + aggregated_hidden).to(dtype=target_hidden.dtype)
            residual_merged_count = int(merge_indices.numel())

        merged_state = {
            "voxel_features": torch.cat(
                [
                    state["voxel_features"].index_select(0, dominant_indices),
                    contextual_features,
                ],
                dim=0,
            ),
            "voxel_coordinates": torch.cat(
                [
                    state["voxel_coordinates"].index_select(0, dominant_indices),
                    contextual_coords,
                ],
                dim=0,
            ),
            "voxel_grouping_coordinates": torch.cat(
                [
                    state["voxel_grouping_coordinates"].index_select(0, dominant_indices),
                    contextual_grouping_coords,
                ],
                dim=0,
            ),
            "voxel_patch_positions": torch.cat(
                [
                    state["voxel_patch_positions"].index_select(0, dominant_indices),
                    contextual_patch_positions,
                ],
                dim=0,
            ),
            "voxel_frame_ids": torch.cat(
                [
                    state["voxel_frame_ids"].index_select(0, dominant_indices),
                    contextual_frame_ids,
                ],
                dim=0,
            ),
            "voxel_order_ids": torch.cat(
                [
                    state["voxel_order_ids"].index_select(0, dominant_indices),
                    contextual_order_ids,
                ],
                dim=0,
            ),
            "num_frames": state["num_frames"],
            "input_tokens": state["input_tokens"],
            "voxel_attention": state["voxel_attention"],
            "voxel_view_support": state["voxel_view_support"],
            "voxel_occupancies": state["voxel_occupancies"],
        }
        selected_indices = torch.arange(merged_state["voxel_features"].shape[0], device=merged_state["voxel_features"].device, dtype=torch.long)
        merged_scores = torch.cat(
            [state["voxel_attention"].index_select(0, dominant_indices).float(), contextual_scores],
            dim=0,
        )
        return self._gather_scored_selected_state(
            state=merged_state,
            selected_indices=selected_indices,
            selected_scores=merged_scores,
            selected_priority=torch.arange(
                selected_indices.numel(),
                device=selected_indices.device,
                dtype=torch.long,
            ),
            extra_metadata={
                "voxel_method": "vtc_visionzip",
                "voxel_size": float(self.visionzip_config.voxel_size),
                "attention_reduce": self.visionzip_config.attention_reduce,
                "dominant_ratio": float(self.visionzip_config.dominant_ratio),
                "coverage_rule": self.visionzip_config.coverage_rule,
                "random_seed": int(self.visionzip_config.random_seed),
                "residual_merge": bool(self.visionzip_config.residual_merge),
                "residual_merge_applied": bool(residual_merge_enabled and residual_merged_count > 0),
                "num_residual_merged": int(residual_merged_count),
                "requested_dominant_tokens": int(requested_dominant_tokens),
                "requested_contextual_tokens": int(requested_contextual_tokens),
                "dominant_tokens": int(dominant_indices.numel()),
                "contextual_tokens": int(contextual_anchor_indices.numel()),
                **self._collect_common_metadata(
                    state, target_tokens, nominal_target_tokens=nominal_target_tokens
                ),
            },
        )

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["coordinates"] = True
        required["attn_weights"] = True
        return required


@register_compressor("voxel_vtc_tome")
class VoxelVTCToMeCompressor(_VoxelVTCPostMixin, BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        tome_config = VoxelVTCToMeConfig(**config) if isinstance(config, dict) else config
        self._validate_base_config(tome_config)
        if tome_config.local_k <= 0:
            raise ValueError(f"local_k must be positive, got {tome_config.local_k}.")
        if tome_config.max_rounds <= 0:
            raise ValueError(f"max_rounds must be positive, got {tome_config.max_rounds}.")
        super().__init__(tome_config)
        self.tome_config = tome_config

    def _tome_round(
        self,
        *,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        grouping_coordinates: torch.Tensor,
        patch_positions: torch.Tensor,
        frame_ids: torch.Tensor,
        order_ids: torch.Tensor,
        token_sizes: torch.Tensor,
        reduce_by: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if reduce_by <= 0 or features.shape[0] <= 1:
            return features, coordinates, grouping_coordinates, patch_positions, frame_ids, order_ids, token_sizes, 0

        spatial_order = self._spatial_sort_indices(grouping_coordinates, float(self.tome_config.voxel_size))
        features = features.index_select(0, spatial_order)
        coordinates = coordinates.index_select(0, spatial_order)
        grouping_coordinates = grouping_coordinates.index_select(0, spatial_order)
        patch_positions = patch_positions.index_select(0, spatial_order)
        frame_ids = frame_ids.index_select(0, spatial_order)
        order_ids = order_ids.index_select(0, spatial_order)
        token_sizes = token_sizes.index_select(0, spatial_order)

        a_idx = torch.arange(0, features.shape[0], 2, device=features.device)
        b_idx = torch.arange(1, features.shape[0], 2, device=features.device)
        if a_idx.numel() == 0 or b_idx.numel() == 0:
            return features, coordinates, grouping_coordinates, patch_positions, frame_ids, order_ids, token_sizes, 0

        feat_a = F.normalize(features.index_select(0, a_idx).float(), dim=-1)
        feat_b = F.normalize(features.index_select(0, b_idx).float(), dim=-1)
        scores = feat_a @ feat_b.transpose(0, 1)

        local_k = min(int(self.tome_config.local_k), int(b_idx.numel()))
        distances = torch.cdist(
            grouping_coordinates.index_select(0, a_idx).float(),
            grouping_coordinates.index_select(0, b_idx).float(),
            p=2,
        )
        nearest = torch.topk(distances, k=local_k, largest=False, dim=-1).indices
        local_mask = torch.zeros_like(scores, dtype=torch.bool)
        local_mask.scatter_(1, nearest, True)
        scores = scores.masked_fill(~local_mask, float("-inf"))

        node_max, node_idx = scores.max(dim=-1)
        valid = torch.isfinite(node_max)
        if not valid.any():
            return features, coordinates, grouping_coordinates, patch_positions, frame_ids, order_ids, token_sizes, 0

        candidate_a = torch.nonzero(valid, as_tuple=False).flatten()
        candidate_scores = node_max.index_select(0, candidate_a)
        sorted_candidates = candidate_scores.argsort(descending=True)
        num_merges = min(reduce_by, int(candidate_a.numel()))
        selected_a_local = candidate_a.index_select(0, sorted_candidates[:num_merges])
        selected_b_local = node_idx.index_select(0, selected_a_local)

        keep_a_mask = torch.ones(a_idx.numel(), device=features.device, dtype=torch.bool)
        keep_a_mask[selected_a_local] = False
        keep_a_idx = a_idx.index_select(0, torch.nonzero(keep_a_mask, as_tuple=False).flatten())

        dst_idx = b_idx
        dst_token_sizes = token_sizes.index_select(0, dst_idx)
        dst_feature_sum = features.index_select(0, dst_idx) * dst_token_sizes.unsqueeze(-1)
        dst_coord_weights = dst_token_sizes.reshape(
            -1,
            *([1] * (coordinates.dim() - 1)),
        )
        dst_grouping_coord_weights = dst_token_sizes.reshape(
            -1,
            *([1] * (grouping_coordinates.dim() - 1)),
        )
        dst_coord_sum = coordinates.index_select(0, dst_idx).float() * dst_coord_weights
        dst_grouping_coord_sum = (
            grouping_coordinates.index_select(0, dst_idx).float() * dst_grouping_coord_weights
        )
        dst_sizes = dst_token_sizes.clone()

        src_idx = a_idx.index_select(0, selected_a_local)
        src_sizes = token_sizes.index_select(0, src_idx)
        dst_feature_sum.index_add_(0, selected_b_local, features.index_select(0, src_idx) * src_sizes.unsqueeze(-1))
        src_coord_weights = src_sizes.reshape(
            -1,
            *([1] * (coordinates.dim() - 1)),
        )
        src_grouping_coord_weights = src_sizes.reshape(
            -1,
            *([1] * (grouping_coordinates.dim() - 1)),
        )
        dst_coord_sum.index_add_(0, selected_b_local, coordinates.index_select(0, src_idx).float() * src_coord_weights)
        dst_grouping_coord_sum.index_add_(
            0,
            selected_b_local,
            grouping_coordinates.index_select(0, src_idx).float() * src_grouping_coord_weights,
        )
        dst_sizes.index_add_(0, selected_b_local, src_sizes)

        merged_dst_features = (dst_feature_sum / dst_sizes.unsqueeze(-1)).to(dtype=features.dtype)
        coord_denominator = dst_sizes.reshape(
            -1,
            *([1] * (coordinates.dim() - 1)),
        )
        grouping_coord_denominator = dst_sizes.reshape(
            -1,
            *([1] * (grouping_coordinates.dim() - 1)),
        )
        merged_dst_coordinates = dst_coord_sum / coord_denominator
        merged_dst_grouping_coordinates = dst_grouping_coord_sum / grouping_coord_denominator

        new_features = torch.cat([features.index_select(0, keep_a_idx), merged_dst_features], dim=0)
        new_coordinates = torch.cat(
            [coordinates.index_select(0, keep_a_idx), merged_dst_coordinates.to(dtype=coordinates.dtype)],
            dim=0,
        )
        new_grouping_coordinates = torch.cat(
            [
                grouping_coordinates.index_select(0, keep_a_idx),
                merged_dst_grouping_coordinates.to(dtype=grouping_coordinates.dtype),
            ],
            dim=0,
        )
        new_patch_positions = torch.cat([patch_positions.index_select(0, keep_a_idx), patch_positions.index_select(0, dst_idx)], dim=0)
        new_frame_ids = torch.cat([frame_ids.index_select(0, keep_a_idx), frame_ids.index_select(0, dst_idx)], dim=0)
        new_order_ids = torch.cat([order_ids.index_select(0, keep_a_idx), order_ids.index_select(0, dst_idx)], dim=0)
        new_sizes = torch.cat([token_sizes.index_select(0, keep_a_idx), dst_sizes], dim=0)
        return new_features, new_coordinates, new_grouping_coordinates, new_patch_positions, new_frame_ids, new_order_ids, new_sizes, num_merges

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        state = self._build_vtc_post_state(
            features=features,
            coordinates=coordinates,
            grouping_coordinates=kwargs.get("grouping_coordinates"),
            attn_weights=attn_weights,
            num_frames=num_frames,
            frame_shape=frame_shape,
            voxel_size=float(self.tome_config.voxel_size),
            attention_reduce=self.tome_config.attention_reduce,
        )
        target_tokens = min(
            int(state["voxel_features"].shape[0]),
            self._resolve_target_tokens(
                int(state["input_tokens"]),
                self.tome_config.target_tokens,
                self.tome_config.target_keep_ratio,
            ),
        )

        current_features = state["voxel_features"]
        current_coordinates = state["voxel_coordinates"]
        current_grouping_coordinates = state["voxel_grouping_coordinates"]
        current_patch_positions = state["voxel_patch_positions"]
        current_frame_ids = state["voxel_frame_ids"]
        current_order_ids = state["voxel_order_ids"]
        current_sizes = state["voxel_occupancies"].clone()

        round_history: List[Dict[str, Any]] = []
        for round_idx in range(int(self.tome_config.max_rounds)):
            if current_features.shape[0] <= target_tokens:
                break
            reduce_by = int(current_features.shape[0] - target_tokens)
            (
                current_features,
                current_coordinates,
                current_grouping_coordinates,
                current_patch_positions,
                current_frame_ids,
                current_order_ids,
                current_sizes,
                merged_count,
            ) = self._tome_round(
                features=current_features,
                coordinates=current_coordinates,
                grouping_coordinates=current_grouping_coordinates,
                patch_positions=current_patch_positions,
                frame_ids=current_frame_ids,
                order_ids=current_order_ids,
                token_sizes=current_sizes,
                reduce_by=reduce_by,
            )
            round_history.append(
                {
                    "round": int(round_idx),
                    "merged": int(merged_count),
                    "tokens_after": int(current_features.shape[0]),
                }
            )
            if merged_count == 0:
                break

        reduced_state = {
            "voxel_features": current_features,
            "voxel_coordinates": current_coordinates,
            "voxel_grouping_coordinates": current_grouping_coordinates,
            "voxel_patch_positions": current_patch_positions,
            "voxel_frame_ids": current_frame_ids,
            "voxel_order_ids": current_order_ids,
            "num_frames": state["num_frames"],
            "input_tokens": state["input_tokens"],
            "voxel_attention": state["voxel_attention"],
            "voxel_view_support": state["voxel_view_support"],
            "voxel_occupancies": state["voxel_occupancies"],
        }
        selected_indices = torch.arange(current_features.shape[0], device=current_features.device, dtype=torch.long)
        return self._gather_selected_state(
            state=reduced_state,
            selected_indices=selected_indices,
            extra_metadata={
                "voxel_method": "vtc_tome",
                "voxel_size": float(self.tome_config.voxel_size),
                "local_k": int(self.tome_config.local_k),
                "num_rounds": int(len(round_history)),
                "round_history": round_history,
                **self._collect_common_metadata(state, target_tokens),
            },
        )

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["coordinates"] = True
        return required
