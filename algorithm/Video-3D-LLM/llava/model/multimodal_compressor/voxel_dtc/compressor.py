import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import BaseCompressor, CompressorConfig, CompressorOutput
from ..registry import register_compressor


@dataclass
class VoxelVTCConfig(CompressorConfig):
    voxel_size: float = 0.1
    order_strategy: str = "representative"
    newline_strategy: str = "grid_drop"


@dataclass
class VoxelDTCConfig(CompressorConfig):
    initial_voxel_size: float = 0.1
    voxel_size_step: float = 0.02
    edge_keep_ratio: float = 0.4
    newline_strategy: str = "grid_drop"
    num_iterations: Optional[int] = 1
    max_iterations: Optional[int] = 8
    target_tokens: Optional[int] = None
    target_keep_ratio: Optional[float] = None
    random_seed: int = 0


class _VoxelCompressionMixin:
    def _normalize_features(self, features: torch.Tensor, num_frames: int) -> torch.Tensor:
        if features.dim() == 3 and features.shape[0] == num_frames:
            return features.contiguous().view(1, -1, features.shape[-1])
        if features.dim() == 3:
            return features
        raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")

    def _normalize_coordinates(
        self,
        coordinates: Optional[torch.Tensor],
        batch_size: int,
        num_frames: int,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        if coordinates is None:
            raise ValueError("Voxel-based compression requires patch-level coordinates.")
        if coordinates.dim() >= 3 and coordinates.shape[0] == num_frames:
            coordinates = coordinates.unsqueeze(0)
        elif coordinates.dim() >= 4 and coordinates.shape[1] == num_frames:
            pass
        elif coordinates.dim() >= 3 and coordinates.shape[0] == batch_size and coordinates.shape[1] == num_frames * tokens_per_frame:
            tail_shape = coordinates.shape[2:]
            coordinates = coordinates.view(batch_size, num_frames, tokens_per_frame, *tail_shape)
        else:
            raise ValueError(f"Unsupported coordinate shape: {tuple(coordinates.shape)}")

        if coordinates.shape[0] != batch_size or coordinates.shape[1] != num_frames or coordinates.shape[2] != tokens_per_frame:
            raise ValueError(
                "Coordinate tensor shape does not match feature layout: "
                f"{tuple(coordinates.shape)} vs batch={batch_size}, frames={num_frames}, tokens={tokens_per_frame}"
            )
        return coordinates

    def _resolve_coordinate_pair(
        self,
        *,
        coordinates: Optional[torch.Tensor],
        grouping_coordinates: Optional[torch.Tensor],
        batch_size: int,
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_coordinates = self._normalize_coordinates(coordinates, batch_size, num_frames, tokens_per_frame)
        if normalized_coordinates.device != device:
            normalized_coordinates = normalized_coordinates.to(device=device)

        if grouping_coordinates is None:
            normalized_grouping_coordinates = normalized_coordinates
            if normalized_coordinates.dim() > 4:
                reduce_dims = tuple(range(3, normalized_coordinates.dim() - 1))
                normalized_grouping_coordinates = normalized_coordinates.float().mean(dim=reduce_dims).to(
                    dtype=normalized_coordinates.dtype
                )
            return normalized_coordinates, normalized_grouping_coordinates

        normalized_grouping_coordinates = self._normalize_coordinates(
            grouping_coordinates,
            batch_size,
            num_frames,
            tokens_per_frame,
        )
        if normalized_grouping_coordinates.device != device:
            normalized_grouping_coordinates = normalized_grouping_coordinates.to(device=device)
        return normalized_coordinates, normalized_grouping_coordinates

    def _build_patch_positions(
        self,
        num_frames: int,
        frame_shape: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        height, width = frame_shape
        row_ids = torch.arange(height, device=device, dtype=torch.float32)
        col_ids = torch.arange(width, device=device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(row_ids, col_ids, indexing="ij")
        patch_positions = torch.stack((row_grid, col_grid), dim=-1).view(1, height * width, 2)
        return patch_positions.repeat(num_frames, 1, 1)

    def _flatten_token_state(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        patch_positions: torch.Tensor,
        num_frames: int,
        tokens_per_frame: int,
    ):
        flat_features = features.view(num_frames * tokens_per_frame, features.shape[-1])
        flat_coordinates = coordinates.reshape(num_frames * tokens_per_frame, *coordinates.shape[2:])
        flat_patch_positions = patch_positions.view(num_frames * tokens_per_frame, patch_positions.shape[-1])
        frame_ids = torch.arange(num_frames, device=features.device, dtype=torch.long).repeat_interleave(tokens_per_frame)
        order_ids = torch.arange(num_frames * tokens_per_frame, device=features.device, dtype=torch.long)
        return flat_features, flat_coordinates, flat_patch_positions, frame_ids, order_ids

    def _discretize_voxel_indices(
        self,
        coordinates: torch.Tensor,
        voxel_size: float,
    ) -> torch.Tensor:
        return torch.round(coordinates.float() / voxel_size).to(dtype=torch.long)

    def _group_tokens_by_voxel(
        self,
        coordinates: torch.Tensor,
        voxel_size: float,
    ) -> List[List[int]]:
        return [group for _, group in self._group_tokens_by_voxel_with_indices(coordinates, voxel_size)]

    def _group_tokens_by_voxel_with_indices(
        self,
        coordinates: torch.Tensor,
        voxel_size: float,
    ) -> List[Tuple[Tuple[int, int, int], List[int]]]:
        voxel_indices = self._discretize_voxel_indices(coordinates, voxel_size)
        voxel_index_list = voxel_indices.detach().cpu().tolist()
        groups: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, key in enumerate(voxel_index_list):
            groups.setdefault(tuple(key), []).append(idx)
        return list(groups.items())

    def _finalize_outputs(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        patch_positions: torch.Tensor,
        frame_ids: torch.Tensor,
        order_ids: torch.Tensor,
        num_frames: int,
        input_tokens: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
        sort_ids: Optional[torch.Tensor] = None,
        frame_token_counts_override: Optional[List[int]] = None,
    ) -> CompressorOutput:
        if sort_ids is None:
            sort_ids = order_ids
        order = sort_ids.argsort(stable=True)
        features = features[order]
        coordinates = coordinates[order]
        patch_positions = patch_positions[order]
        frame_ids = frame_ids[order]
        order_ids = order_ids[order]
        sort_ids = sort_ids[order]

        if frame_token_counts_override is None:
            frame_token_counts = torch.bincount(frame_ids, minlength=num_frames).tolist()
        else:
            frame_token_counts = frame_token_counts_override
        metadata = {
            "layout_type": "per_frame",
            "frame_token_counts": [int(count) for count in frame_token_counts],
            "compressed_coordinates": coordinates.unsqueeze(0),
            "compressed_patch_positions": patch_positions.unsqueeze(0),
            "retained_order_ids": [int(order_id) for order_id in order_ids.tolist()],
            "sort_ids": [int(sort_id) for sort_id in sort_ids.tolist()],
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        output_tokens = features.shape[0]
        self._update_stats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            frame_token_counts=metadata["frame_token_counts"],
            **{
                key: value
                for key, value in (extra_metadata or {}).items()
                if isinstance(value, (int, float, bool, str))
            },
        )
        return CompressorOutput(
            features=features.unsqueeze(0),
            compression_ratio=output_tokens / input_tokens if input_tokens > 0 else 1.0,
            metadata=metadata,
        )


@register_compressor("voxel_vtc")
class VoxelVTCCompressor(_VoxelCompressionMixin, BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        voxel_config = VoxelVTCConfig(**config) if isinstance(config, dict) else config
        if voxel_config.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_config.voxel_size}.")
        if voxel_config.order_strategy not in {"representative", "voxel_id"}:
            raise ValueError(
                "VoxelVTC order_strategy must be either 'representative' or 'voxel_id', "
                f"got {voxel_config.order_strategy!r}."
            )
        if voxel_config.newline_strategy not in {"one_token", "grid_drop", "frame_newline", "no_token"}:
            raise ValueError(
                "VoxelVTC newline_strategy must be one of 'one_token', 'grid_drop', 'frame_newline', or 'no_token', "
                f"got {voxel_config.newline_strategy!r}."
            )
        super().__init__(voxel_config)
        self.voxel_config = voxel_config

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        grouping_coordinates: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        if num_frames is None:
            raise ValueError("VoxelVTC requires num_frames.")
        if frame_shape is None:
            raise ValueError("VoxelVTC requires frame_shape.")

        features = self._normalize_features(features, num_frames)
        if features.shape[0] != 1:
            raise ValueError("VoxelVTC currently supports one video sample at a time.")
        _, total_tokens, _ = features.shape
        height, width = frame_shape
        tokens_per_frame = height * width
        if total_tokens != num_frames * tokens_per_frame:
            raise ValueError(
                f"Expected {num_frames * tokens_per_frame} tokens, got {total_tokens}."
            )

        coordinates, grouping_coordinates = self._resolve_coordinate_pair(
            coordinates=coordinates,
            grouping_coordinates=grouping_coordinates,
            batch_size=1,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            device=features.device,
        )
        coordinates = coordinates[0]
        grouping_coordinates = grouping_coordinates[0]
        patch_positions = self._build_patch_positions(num_frames, frame_shape, features.device)
        flat_features, flat_coordinates, flat_patch_positions, frame_ids, order_ids = self._flatten_token_state(
            features=features[0],
            coordinates=coordinates,
            patch_positions=patch_positions,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
        )
        flat_grouping_coordinates = grouping_coordinates.view(num_frames * tokens_per_frame, grouping_coordinates.shape[-1])

        voxel_groups = self._group_tokens_by_voxel_with_indices(
            flat_grouping_coordinates,
            self.voxel_config.voxel_size,
        )
        voxel_rank_by_key = {
            key: rank
            for rank, key in enumerate(sorted(key for key, _ in voxel_groups))
        }

        reduced_features = []
        reduced_coordinates = []
        reduced_patch_positions = []
        reduced_frame_ids = []
        reduced_order_ids = []
        reduced_sort_ids = []
        voxel_occupancies = []
        for voxel_key, group in voxel_groups:
            group_index = torch.tensor(group, device=features.device, dtype=torch.long)
            voxel_occupancies.append(int(group_index.numel()))
            voxel_center = torch.tensor(
                voxel_key,
                device=features.device,
                dtype=flat_grouping_coordinates.dtype,
            ) * float(self.voxel_config.voxel_size)
            group_distances = torch.norm(
                flat_grouping_coordinates[group_index].float() - voxel_center.float(),
                dim=-1,
            )
            nearest_mask = group_distances == group_distances.min()
            nearest_group_index = group_index[nearest_mask]
            representative_idx = nearest_group_index[order_ids[nearest_group_index].argmin()]
            reduced_feature = flat_features[group_index].mean(dim=0)
            reduced_coordinate = flat_coordinates[group_index].mean(dim=0)

            reduced_features.append(reduced_feature)
            reduced_coordinates.append(reduced_coordinate)
            reduced_patch_positions.append(flat_patch_positions[representative_idx])
            reduced_frame_ids.append(frame_ids[representative_idx])
            reduced_order_ids.append(order_ids[representative_idx])
            if self.voxel_config.order_strategy == "voxel_id":
                reduced_sort_ids.append(torch.tensor(voxel_rank_by_key[voxel_key], device=features.device, dtype=torch.long))
            else:
                reduced_sort_ids.append(order_ids[representative_idx])

        reduced_features = torch.stack(reduced_features, dim=0)
        reduced_coordinates = torch.stack(reduced_coordinates, dim=0).to(dtype=coordinates.dtype)
        reduced_patch_positions = torch.stack(reduced_patch_positions, dim=0)
        reduced_frame_ids = torch.stack(reduced_frame_ids, dim=0)
        reduced_order_ids = torch.stack(reduced_order_ids, dim=0)
        reduced_sort_ids = torch.stack(reduced_sort_ids, dim=0)
        frame_token_counts_override = None
        if self.voxel_config.order_strategy == "voxel_id":
            frame_token_counts_override = [int(reduced_features.shape[0])]

        return self._finalize_outputs(
            features=reduced_features,
            coordinates=reduced_coordinates,
            patch_positions=reduced_patch_positions,
            frame_ids=reduced_frame_ids,
            order_ids=reduced_order_ids,
            num_frames=num_frames,
            input_tokens=total_tokens,
            sort_ids=reduced_sort_ids,
            frame_token_counts_override=frame_token_counts_override,
            extra_metadata={
                "newline_strategy": self.voxel_config.newline_strategy,
                "voxel_method": "vtc",
                "voxel_size": float(self.voxel_config.voxel_size),
                "order_strategy": self.voxel_config.order_strategy,
                "preserve_segment_order": self.voxel_config.order_strategy == "voxel_id",
                "layout_type": "scene" if self.voxel_config.order_strategy == "voxel_id" else "per_frame",
                "num_voxels": int(len(voxel_groups)),
                "max_voxel_occupancy": int(max(voxel_occupancies) if voxel_occupancies else 0),
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


@register_compressor("voxel_dtc")
class VoxelDTCCompressor(_VoxelCompressionMixin, BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        voxel_config = VoxelDTCConfig(**config) if isinstance(config, dict) else config
        if voxel_config.initial_voxel_size <= 0:
            raise ValueError(
                f"initial_voxel_size must be positive, got {voxel_config.initial_voxel_size}."
            )
        if voxel_config.voxel_size_step <= 0:
            raise ValueError(f"voxel_size_step must be positive, got {voxel_config.voxel_size_step}.")
        if not 0 < voxel_config.edge_keep_ratio <= 1:
            raise ValueError(f"edge_keep_ratio must be in (0, 1], got {voxel_config.edge_keep_ratio}.")
        if voxel_config.newline_strategy not in {"one_token", "grid_drop"}:
            raise ValueError(
                "VoxelDTC newline_strategy must be either 'one_token' or 'grid_drop', "
                f"got {voxel_config.newline_strategy!r}."
            )
        if voxel_config.target_tokens is not None and voxel_config.target_keep_ratio is not None:
            raise ValueError("VoxelDTC accepts either target_tokens or target_keep_ratio, not both.")
        if voxel_config.target_tokens is not None and int(voxel_config.target_tokens) <= 0:
            raise ValueError(f"target_tokens must be positive, got {voxel_config.target_tokens}.")
        if voxel_config.target_keep_ratio is not None and not 0 < voxel_config.target_keep_ratio <= 1:
            raise ValueError(
                f"target_keep_ratio must be in (0, 1], got {voxel_config.target_keep_ratio}."
            )
        if voxel_config.num_iterations is not None and voxel_config.num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {voxel_config.num_iterations}.")
        if voxel_config.max_iterations is not None and voxel_config.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {voxel_config.max_iterations}.")
        super().__init__(voxel_config)
        self.voxel_config = voxel_config

    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        coordinates: Optional[torch.Tensor] = None,
        grouping_coordinates: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressorOutput:
        if num_frames is None:
            raise ValueError("VoxelDTC requires num_frames.")
        if frame_shape is None:
            raise ValueError("VoxelDTC requires frame_shape.")

        features = self._normalize_features(features, num_frames)
        if features.shape[0] != 1:
            raise ValueError("VoxelDTC currently supports one video sample at a time.")
        _, total_tokens, _ = features.shape
        height, width = frame_shape
        tokens_per_frame = height * width
        if total_tokens != num_frames * tokens_per_frame:
            raise ValueError(
                f"Expected {num_frames * tokens_per_frame} tokens, got {total_tokens}."
            )

        coordinates, grouping_coordinates = self._resolve_coordinate_pair(
            coordinates=coordinates,
            grouping_coordinates=grouping_coordinates,
            batch_size=1,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            device=features.device,
        )
        coordinates = coordinates[0]
        grouping_coordinates = grouping_coordinates[0]
        patch_positions = self._build_patch_positions(num_frames, frame_shape, features.device)
        current_features, current_coordinates, current_patch_positions, frame_ids, order_ids = self._flatten_token_state(
            features=features[0],
            coordinates=coordinates,
            patch_positions=patch_positions,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
        )
        current_grouping_coordinates = grouping_coordinates.view(
            num_frames * tokens_per_frame,
            grouping_coordinates.shape[-1],
        )

        target_tokens = self._resolve_target_tokens(total_tokens)
        current_voxel_size = float(self.voxel_config.initial_voxel_size)
        iteration_history: List[Dict[str, Any]] = []
        stop_reason = "not_started"
        iteration_idx = 0
        while True:
            if self._should_stop(
                current_tokens=current_features.shape[0],
                target_tokens=target_tokens,
                iteration_idx=iteration_idx,
            ):
                if target_tokens is not None and current_features.shape[0] <= target_tokens:
                    stop_reason = "target_tokens_reached"
                else:
                    stop_reason = "iteration_limit_reached"
                break

            previous_tokens = int(current_features.shape[0])
            step_output = self._run_dynamic_iteration(
                features=current_features,
                coordinates=current_coordinates,
                grouping_coordinates=current_grouping_coordinates,
                patch_positions=current_patch_positions,
                frame_ids=frame_ids,
                order_ids=order_ids,
                voxel_size=current_voxel_size,
                iteration_idx=iteration_idx,
            )
            current_features = step_output["features"]
            current_coordinates = step_output["coordinates"]
            current_grouping_coordinates = step_output["grouping_coordinates"]
            current_patch_positions = step_output["patch_positions"]
            frame_ids = step_output["frame_ids"]
            order_ids = step_output["order_ids"]
            current_tokens = int(current_features.shape[0])
            iteration_history.append(
                {
                    "iteration": int(iteration_idx),
                    "voxel_size": float(current_voxel_size),
                    "kept_pairs": int(step_output["kept_pairs"]),
                    "tokens_after": current_tokens,
                }
            )
            iteration_idx += 1
            if target_tokens is not None and current_tokens <= target_tokens:
                stop_reason = "target_tokens_reached"
                break
            if current_tokens <= 1:
                stop_reason = "single_token_remaining"
                break
            if target_tokens is None and current_tokens >= previous_tokens:
                stop_reason = "no_progress"
                break
            current_voxel_size += self.voxel_config.voxel_size_step

        return self._finalize_outputs(
            features=current_features,
            coordinates=current_coordinates.to(dtype=coordinates.dtype),
            patch_positions=current_patch_positions,
            frame_ids=frame_ids,
            order_ids=order_ids,
            num_frames=num_frames,
            input_tokens=total_tokens,
            extra_metadata={
                "newline_strategy": self.voxel_config.newline_strategy,
                "voxel_method": "dtc",
                "initial_voxel_size": float(self.voxel_config.initial_voxel_size),
                "voxel_size_step": float(self.voxel_config.voxel_size_step),
                "edge_keep_ratio": float(self.voxel_config.edge_keep_ratio),
                "target_tokens": int(target_tokens) if target_tokens is not None else None,
                "num_iterations_run": int(len(iteration_history)),
                "stop_reason": stop_reason,
                "iteration_history": iteration_history,
            },
        )

    def _resolve_target_tokens(self, input_tokens: int) -> Optional[int]:
        if self.voxel_config.target_tokens is not None:
            return max(1, int(self.voxel_config.target_tokens))
        if self.voxel_config.target_keep_ratio is not None:
            return max(1, int(math.ceil(input_tokens * self.voxel_config.target_keep_ratio)))
        return None

    def _should_stop(
        self,
        current_tokens: int,
        target_tokens: Optional[int],
        iteration_idx: int,
    ) -> bool:
        if target_tokens is not None:
            return current_tokens <= target_tokens
        if self.voxel_config.num_iterations is not None and iteration_idx >= self.voxel_config.num_iterations:
            return True
        if self.voxel_config.max_iterations is not None and iteration_idx >= self.voxel_config.max_iterations:
            return True
        return False

    def _run_dynamic_iteration(
        self,
        *,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        grouping_coordinates: torch.Tensor,
        patch_positions: torch.Tensor,
        frame_ids: torch.Tensor,
        order_ids: torch.Tensor,
        voxel_size: float,
        iteration_idx: int,
    ) -> Dict[str, torch.Tensor]:
        voxel_groups = self._group_tokens_by_voxel(grouping_coordinates, voxel_size)
        edges = self._build_similarity_edges(features, voxel_groups, iteration_idx)
        if not edges:
            return {
                "features": features,
                "coordinates": coordinates,
                "grouping_coordinates": grouping_coordinates,
                "patch_positions": patch_positions,
                "frame_ids": frame_ids,
                "order_ids": order_ids,
                "kept_pairs": 0,
            }

        keep_edges = max(1, int(math.ceil(len(edges) * self.voxel_config.edge_keep_ratio)))
        edges.sort(key=lambda edge: edge[2], reverse=True)
        selected_edges = edges[:keep_edges]
        kept_indices, accepted_pairs = self._greedy_pairwise_keep_first(
            features.shape[0],
            selected_edges,
            order_ids,
        )
        kept_indices = torch.tensor(kept_indices, device=features.device, dtype=torch.long)
        kept_indices = kept_indices[order_ids[kept_indices].argsort(stable=True)]

        return {
            "features": features[kept_indices],
            "coordinates": coordinates[kept_indices],
            "grouping_coordinates": grouping_coordinates[kept_indices],
            "patch_positions": patch_positions[kept_indices],
            "frame_ids": frame_ids[kept_indices],
            "order_ids": order_ids[kept_indices],
            "kept_pairs": int(accepted_pairs),
        }

    def _build_similarity_edges(
        self,
        features: torch.Tensor,
        voxel_groups: List[List[int]],
        iteration_idx: int,
    ) -> List[Tuple[int, int, float]]:
        edges: List[Tuple[int, int, float]] = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.voxel_config.random_seed + iteration_idx))

        for group in voxel_groups:
            if len(group) < 2:
                continue
            permutation = torch.randperm(len(group), generator=generator).tolist()
            shuffled_group = [group[idx] for idx in permutation]
            split = len(shuffled_group) // 2
            group_a = shuffled_group[:split]
            group_b = shuffled_group[split:]
            if not group_a or not group_b:
                continue

            group_a_tensor = torch.tensor(group_a, device=features.device, dtype=torch.long)
            group_b_tensor = torch.tensor(group_b, device=features.device, dtype=torch.long)
            feat_a = F.normalize(features[group_a_tensor], p=2, dim=-1)
            feat_b = F.normalize(features[group_b_tensor], p=2, dim=-1)
            similarity = feat_a @ feat_b.transpose(0, 1)
            best_similarity, best_target = similarity.max(dim=1)
            for src_idx, dst_offset, sim_score in zip(group_a, best_target.tolist(), best_similarity.tolist()):
                edges.append((int(src_idx), int(group_b[dst_offset]), float(sim_score)))
        return edges

    def _greedy_pairwise_keep_first(
        self,
        num_tokens: int,
        edges: List[Tuple[int, int, float]],
        order_ids: torch.Tensor,
    ) -> Tuple[List[int], int]:
        used_tokens = set()
        dropped_tokens = set()
        accepted_pairs = 0

        for src_idx, dst_idx, _ in edges:
            src_idx = int(src_idx)
            dst_idx = int(dst_idx)
            if src_idx == dst_idx:
                continue
            if src_idx in used_tokens or dst_idx in used_tokens:
                continue

            src_order = int(order_ids[src_idx].item())
            dst_order = int(order_ids[dst_idx].item())
            if src_order <= dst_order:
                keep_idx, drop_idx = src_idx, dst_idx
            else:
                keep_idx, drop_idx = dst_idx, src_idx

            used_tokens.add(keep_idx)
            used_tokens.add(drop_idx)
            dropped_tokens.add(drop_idx)
            accepted_pairs += 1

        keep_indices = [token_idx for token_idx in range(num_tokens) if token_idx not in dropped_tokens]
        return keep_indices, accepted_pairs

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["coordinates"] = True
        return required
