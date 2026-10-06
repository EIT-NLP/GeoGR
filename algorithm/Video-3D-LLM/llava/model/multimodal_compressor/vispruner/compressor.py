from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import BaseCompressor, CompressorConfig, CompressorOutput
from ..registry import register_compressor


@dataclass
class VisPrunerConfig(CompressorConfig):
    visual_token_num: Optional[int] = None
    token_keep_ratio: Optional[float] = None
    important_ratio: float = 0.5
    prune_step: int = 8
    newline_strategy: str = "grid_drop"


@register_compressor("vispruner")
class VisPrunerCompressor(BaseCompressor):
    def __init__(self, config: Dict[str, Any]):
        vispruner_config = VisPrunerConfig(**config) if isinstance(config, dict) else config
        if vispruner_config.visual_token_num is None and vispruner_config.token_keep_ratio is None:
            raise ValueError("VisPruner requires either visual_token_num or token_keep_ratio.")
        if vispruner_config.visual_token_num is not None and vispruner_config.token_keep_ratio is not None:
            raise ValueError("VisPruner accepts either visual_token_num or token_keep_ratio, not both.")
        if not 0.0 <= vispruner_config.important_ratio <= 1.0:
            raise ValueError("important_ratio must be within [0, 1].")
        if vispruner_config.token_keep_ratio is not None and not 0.0 < vispruner_config.token_keep_ratio <= 1.0:
            raise ValueError("token_keep_ratio must be within (0, 1].")
        if vispruner_config.visual_token_num is not None and vispruner_config.visual_token_num <= 0:
            raise ValueError("visual_token_num must be positive.")
        if vispruner_config.prune_step <= 0:
            raise ValueError("prune_step must be positive.")
        if vispruner_config.newline_strategy not in {"grid_drop", "full_grid_drop", "one_token"}:
            raise ValueError("VisPruner newline_strategy must be one of 'grid_drop', 'full_grid_drop', or 'one_token'.")
        super().__init__(vispruner_config)
        self.vispruner_config = vispruner_config

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
            raise ValueError("VisPruner requires num_frames.")
        if frame_shape is None:
            raise ValueError("VisPruner requires frame_shape.")
        if attn_weights is None:
            raise ValueError("VisPruner requires attn_weights.")

        height, width = frame_shape
        tokens_per_frame = height * width
        candidate_features = self._normalize_feature_tensor(features, num_frames, tokens_per_frame)
        selection_source = raw_features_before_proj if raw_features_before_proj is not None else features
        selection_features = self._normalize_feature_tensor(selection_source, num_frames, tokens_per_frame)
        if selection_features.device != candidate_features.device:
            selection_features = selection_features.to(candidate_features.device)
        attention_scores = self._normalize_attention_scores(attn_weights, num_frames, tokens_per_frame, candidate_features.device)
        coordinates = self._normalize_coordinates(coordinates, num_frames, tokens_per_frame)
        if coordinates is not None and coordinates.device != candidate_features.device:
            coordinates = coordinates.to(candidate_features.device)
        patch_positions = self._build_patch_positions(num_frames, frame_shape, candidate_features.device)

        token_budget = self._resolve_token_budget(tokens_per_frame)
        important_token_num = int(token_budget * self.vispruner_config.important_ratio)
        diverse_token_num = token_budget - important_token_num

        selected_features = []
        selected_coordinates = [] if coordinates is not None else None
        selected_patch_positions = []
        frame_token_counts = []

        selection_features = F.normalize(selection_features.float(), dim=-1)

        for frame_idx in range(num_frames):
            frame_scores = attention_scores[frame_idx]
            frame_order = frame_scores.argsort(dim=-1, descending=True)
            important_indices = frame_order[:important_token_num]
            residual_indices = frame_order[important_token_num:]
            diverse_indices = self._select_diverse_tokens(
                selection_features[frame_idx],
                residual_indices,
                diverse_token_num,
            )

            selected_indices = torch.cat((important_indices, diverse_indices), dim=0)
            if selected_indices.numel() > token_budget:
                selected_indices = selected_indices[:token_budget]
            selected_indices = selected_indices.sort().values

            selected_features.append(candidate_features[frame_idx].index_select(0, selected_indices))
            selected_patch_positions.append(patch_positions[frame_idx].index_select(0, selected_indices))
            if selected_coordinates is not None:
                selected_coordinates.append(coordinates[frame_idx].index_select(0, selected_indices))
            frame_token_counts.append(int(selected_indices.numel()))

        compressed_features = torch.cat(selected_features, dim=0).unsqueeze(0)
        compressed_patch_positions = torch.cat(selected_patch_positions, dim=0).unsqueeze(0)
        compressed_coordinates = None
        if selected_coordinates is not None:
            compressed_coordinates = torch.cat(selected_coordinates, dim=0).unsqueeze(0)

        output_tokens = compressed_features.shape[1]
        input_tokens = num_frames * tokens_per_frame
        self._update_stats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_frames=num_frames,
            frame_shape=frame_shape,
            visual_token_num_per_frame=token_budget,
            important_token_num_per_frame=important_token_num,
            diverse_token_num_per_frame=diverse_token_num,
        )

        metadata = {
            "newline_strategy": self.vispruner_config.newline_strategy,
            "frame_token_counts": frame_token_counts,
            "compressed_patch_positions": compressed_patch_positions,
            "visual_token_num_per_frame": token_budget,
            "important_token_num_per_frame": important_token_num,
            "diverse_token_num_per_frame": diverse_token_num,
        }
        if compressed_coordinates is not None:
            metadata["compressed_coordinates"] = compressed_coordinates

        return CompressorOutput(
            features=compressed_features,
            compression_ratio=output_tokens / input_tokens,
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

    def _resolve_token_budget(self, tokens_per_frame: int) -> int:
        if self.vispruner_config.visual_token_num is not None:
            token_budget = int(self.vispruner_config.visual_token_num)
        else:
            token_budget = int(round(tokens_per_frame * float(self.vispruner_config.token_keep_ratio)))

        if token_budget <= 0 or token_budget > tokens_per_frame:
            raise ValueError(
                "VisPruner token budget must be within [1, tokens_per_frame], "
                f"got {token_budget} for {tokens_per_frame} tokens."
            )
        return token_budget

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
            "Unexpected VisPruner feature layout: "
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
            "Unexpected VisPruner attention layout: "
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
            "Unexpected VisPruner coordinate layout: "
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

    def _select_diverse_tokens(
        self,
        normalized_features: torch.Tensor,
        residual_indices: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        if target_count <= 0:
            return residual_indices[:0]

        residual_indices = residual_indices.clone()
        while residual_indices.numel() > target_count:
            residual_count = residual_indices.numel()
            remove_count = min(self.vispruner_config.prune_step, residual_count - target_count)
            if remove_count <= 0:
                break

            residual_tokens = normalized_features.index_select(0, residual_indices)
            even_tokens = residual_tokens[::2]
            odd_tokens = residual_tokens[1::2]
            if odd_tokens.shape[0] == 0:
                residual_indices = residual_indices[:target_count]
                break

            similarity_scores = even_tokens @ odd_tokens.transpose(0, 1)
            similarity_scores = similarity_scores.max(dim=-1).values
            kept_even_indices = similarity_scores.argsort(dim=-1, descending=True)[remove_count:]
            residual_indices = torch.cat(
                (
                    residual_indices[::2].index_select(0, kept_even_indices),
                    residual_indices[1::2],
                ),
                dim=0,
            )

        if residual_indices.numel() > target_count:
            residual_indices = residual_indices[:target_count]
        return residual_indices
