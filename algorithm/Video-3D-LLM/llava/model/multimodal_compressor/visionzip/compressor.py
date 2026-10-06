from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import BaseCompressor, CompressorConfig, CompressorOutput
from ..registry import register_compressor


@dataclass
class VisionZipConfig(CompressorConfig):
    # Scaled from the original LLaVA default (54 dominant + 10 contextual on 24x24=576 tokens)
    # to the native Video3D pooled grid (14x14=196 tokens): about 18 dominant + 4 contextual.
    dominant_tokens: int = 18
    contextual_tokens: int = 4
    newline_strategy: str = "grid_drop"


@register_compressor("visionzip")
class VisionZipCompressor(BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        visionzip_config = VisionZipConfig(**config) if isinstance(config, dict) else config
        if visionzip_config.dominant_tokens < 0:
            raise ValueError(f"dominant_tokens must be non-negative, got {visionzip_config.dominant_tokens}.")
        if visionzip_config.contextual_tokens < 0:
            raise ValueError(f"contextual_tokens must be non-negative, got {visionzip_config.contextual_tokens}.")
        if visionzip_config.dominant_tokens + visionzip_config.contextual_tokens <= 0:
            raise ValueError("VisionZip requires at least one retained token per frame.")
        valid_newline_strategies = {"grid_drop", "full_grid_drop", "frame_newline", "one_token", "no_token"}
        if visionzip_config.newline_strategy not in valid_newline_strategies:
            raise ValueError(
                f"newline_strategy must be one of {sorted(valid_newline_strategies)}, "
                f"got {visionzip_config.newline_strategy!r}."
            )
        super().__init__(visionzip_config)
        self.visionzip_config = visionzip_config

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
        if num_frames is None:
            raise ValueError("VisionZip requires num_frames.")
        if frame_shape is None:
            raise ValueError("VisionZip requires frame_shape.")
        if attn_weights is None:
            raise ValueError("VisionZip requires attn_weights.")
        if raw_features_before_proj is None:
            raise ValueError("VisionZip requires raw_features_before_proj for contextual token merging.")

        height, width = frame_shape
        tokens_per_frame = height * width

        candidate_features = self._normalize_feature_tensor(features, num_frames, tokens_per_frame)
        merge_features = self._normalize_feature_tensor(raw_features_before_proj, num_frames, tokens_per_frame)
        if merge_features.device != candidate_features.device:
            merge_features = merge_features.to(candidate_features.device)
        attention_scores = self._normalize_attention_scores(attn_weights, num_frames, tokens_per_frame, candidate_features.device)
        coordinates = self._normalize_coordinates(coordinates, num_frames, tokens_per_frame)
        if coordinates is not None and coordinates.device != candidate_features.device:
            coordinates = coordinates.to(candidate_features.device)
        patch_positions = self._build_patch_positions(num_frames, frame_shape, candidate_features.device)

        per_frame_budget = min(tokens_per_frame, self.visionzip_config.dominant_tokens + self.visionzip_config.contextual_tokens)
        dominant_budget = min(self.visionzip_config.dominant_tokens, per_frame_budget)
        contextual_budget = max(0, per_frame_budget - dominant_budget)

        selected_features = []
        selected_coordinates = [] if coordinates is not None else None
        selected_patch_positions = []
        frame_token_counts = []
        dominant_counts = []
        contextual_counts = []

        metric_features = F.normalize(merge_features.float(), dim=-1)

        for frame_idx in range(num_frames):
            frame_scores = attention_scores[frame_idx]
            dominant_indices = frame_scores.topk(dominant_budget, dim=-1).indices if dominant_budget > 0 else frame_scores[:0].long()

            frame_features = candidate_features[frame_idx]
            frame_patch_positions = patch_positions[frame_idx]
            frame_coords = None if coordinates is None else coordinates[frame_idx]

            dominant_features = frame_features.index_select(0, dominant_indices) if dominant_indices.numel() > 0 else frame_features[:0]
            dominant_patch_positions = frame_patch_positions.index_select(0, dominant_indices) if dominant_indices.numel() > 0 else frame_patch_positions[:0]
            dominant_coordinates = (
                frame_coords.index_select(0, dominant_indices)
                if frame_coords is not None and dominant_indices.numel() > 0
                else (frame_coords[:0] if frame_coords is not None else None)
            )

            residual_mask = torch.ones(tokens_per_frame, device=frame_features.device, dtype=torch.bool)
            if dominant_indices.numel() > 0:
                residual_mask[dominant_indices] = False
            residual_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()

            effective_contextual_budget = min(contextual_budget, int(residual_indices.numel()))
            contextual_features = frame_features[:0]
            contextual_patch_positions = frame_patch_positions[:0]
            contextual_coordinates = frame_coords[:0] if frame_coords is not None else None

            if effective_contextual_budget > 0:
                if effective_contextual_budget == residual_indices.numel():
                    target_indices = residual_indices
                    merge_indices = residual_indices[:0]
                else:
                    step = max(1, int(residual_indices.numel() // effective_contextual_budget))
                    local_positions = torch.arange(0, residual_indices.numel(), step, device=residual_indices.device)[:effective_contextual_budget]
                    target_indices = residual_indices.index_select(0, local_positions)
                    target_mask = torch.ones(residual_indices.shape[0], device=residual_indices.device, dtype=torch.bool)
                    target_mask[local_positions] = False
                    merge_indices = residual_indices.index_select(0, torch.nonzero(target_mask, as_tuple=False).flatten())

                contextual_features = frame_features.index_select(0, target_indices).clone()
                contextual_patch_positions = frame_patch_positions.index_select(0, target_indices).clone()
                if frame_coords is not None:
                    contextual_coordinates = frame_coords.index_select(0, target_indices).clone()

                if merge_indices.numel() > 0:
                    target_metric = metric_features[frame_idx].index_select(0, target_indices)
                    merge_metric = metric_features[frame_idx].index_select(0, merge_indices)
                    similarity = merge_metric @ target_metric.transpose(0, 1)
                    assignment = similarity.argmax(dim=-1)

                    aggregated_hidden = torch.zeros_like(contextual_features)
                    counts = torch.zeros(
                        target_indices.shape[0],
                        device=frame_features.device,
                        dtype=contextual_features.dtype,
                    )
                    merge_hidden = frame_features.index_select(0, merge_indices)
                    aggregated_hidden.index_add_(0, assignment, merge_hidden)
                    counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=contextual_features.dtype))
                    aggregated_hidden = aggregated_hidden / counts.clamp_min_(1.0).unsqueeze(-1)
                    contextual_features = (contextual_features + aggregated_hidden).to(dtype=frame_features.dtype)

            frame_output_features = torch.cat([dominant_features, contextual_features], dim=0)
            frame_output_patch_positions = torch.cat([dominant_patch_positions, contextual_patch_positions], dim=0)
            if selected_coordinates is not None:
                frame_output_coordinates = torch.cat([dominant_coordinates, contextual_coordinates], dim=0)
                selected_coordinates.append(frame_output_coordinates)

            selected_features.append(frame_output_features)
            selected_patch_positions.append(frame_output_patch_positions)
            frame_token_counts.append(int(frame_output_features.shape[0]))
            dominant_counts.append(int(dominant_features.shape[0]))
            contextual_counts.append(int(contextual_features.shape[0]))

        compressed_features = torch.cat(selected_features, dim=0).unsqueeze(0)
        compressed_patch_positions = torch.cat(selected_patch_positions, dim=0).unsqueeze(0)
        compressed_coordinates = None
        if selected_coordinates is not None:
            compressed_coordinates = torch.cat(selected_coordinates, dim=0).unsqueeze(0)

        input_tokens = num_frames * tokens_per_frame
        output_tokens = compressed_features.shape[1]
        self._update_stats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_frames=num_frames,
            frame_shape=frame_shape,
            dominant_tokens_per_frame=int(dominant_budget),
            contextual_tokens_per_frame=int(contextual_budget),
        )

        metadata = {
            "newline_strategy": self.visionzip_config.newline_strategy,
            "frame_token_counts": frame_token_counts,
            "compressed_patch_positions": compressed_patch_positions,
            "dominant_tokens_per_frame": dominant_counts,
            "contextual_tokens_per_frame": contextual_counts,
            "selection_method": "dominant_attention_contextual_merge",
            "similarity_source": "raw_features_before_proj",
        }
        if compressed_coordinates is not None:
            metadata["compressed_coordinates"] = compressed_coordinates

        return CompressorOutput(
            features=compressed_features,
            compression_ratio=(float(output_tokens) / float(input_tokens)) if input_tokens > 0 else 1.0,
            metadata=metadata,
        )

    def get_required_inputs(self) -> Dict[str, bool]:
        required = super().get_required_inputs()
        required["attn_weights"] = True
        required["raw_features_before_proj"] = True
        return required

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return True

    def _normalize_feature_tensor(
        self,
        tensor: torch.Tensor,
        num_frames: int,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        if tensor.dim() == 3 and tensor.shape[0] == num_frames and tensor.shape[1] == tokens_per_frame:
            return tensor
        if tensor.dim() == 3 and tensor.shape[0] == 1 and tensor.shape[1] == num_frames * tokens_per_frame:
            return tensor.view(num_frames, tokens_per_frame, tensor.shape[-1])
        raise ValueError(
            "Unexpected VisionZip feature layout: "
            f"{tuple(tensor.shape)} for num_frames={num_frames}, tokens_per_frame={tokens_per_frame}."
        )

    def _normalize_attention_scores(
        self,
        attn_weights: torch.Tensor,
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attn_weights.dim() == 3 and attn_weights.shape[0] == 1:
            attn_weights = attn_weights[0]
        if attn_weights.dim() == 2 and attn_weights.shape == (num_frames, tokens_per_frame):
            return attn_weights.to(device=device)
        if attn_weights.dim() == 2 and attn_weights.shape == (1, num_frames * tokens_per_frame):
            return attn_weights.view(num_frames, tokens_per_frame).to(device=device)
        raise ValueError(
            "Unexpected VisionZip attention layout: "
            f"{tuple(attn_weights.shape)} for num_frames={num_frames}, tokens_per_frame={tokens_per_frame}."
        )

    def _normalize_coordinates(
        self,
        coordinates: Optional[torch.Tensor],
        num_frames: int,
        tokens_per_frame: int,
    ) -> Optional[torch.Tensor]:
        if coordinates is None:
            return None
        if coordinates.dim() >= 3 and coordinates.shape[0] == num_frames and coordinates.shape[1] == tokens_per_frame:
            return coordinates
        if coordinates.dim() >= 3 and coordinates.shape[0] == 1 and coordinates.shape[1] == num_frames * tokens_per_frame:
            tail_shape = coordinates.shape[2:]
            return coordinates.view(num_frames, tokens_per_frame, *tail_shape)
        raise ValueError(
            "Unexpected VisionZip coordinate layout: "
            f"{tuple(coordinates.shape)} for num_frames={num_frames}, tokens_per_frame={tokens_per_frame}."
        )

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
