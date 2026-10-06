#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from .multimodal_compressor import build_compressor_from_llava_config, get_compressor_spec_from_llava_config

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            self.mm_projector_compressor = build_compressor_from_llava_config(config, location="projector")
            self.mm_llm_compressor = build_compressor_from_llava_config(config, location="llm")
            self.mm_compressor = self.mm_projector_compressor

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                embed_std = 1 / torch.sqrt(torch.tensor(config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(config.hidden_size, dtype=self.dtype) * embed_std)
        
        if hasattr(self.config, 'world_position_embedding_type'):
            from llava.model.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingMLP

            if "sample9" in self.config.world_position_embedding_type:
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                n_points = 5
            elif "minmax" in self.config.world_position_embedding_type:
                n_points = 2
            else:
                n_points = 1
        
            if "mlp" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingMLP(config.hidden_size, n_points=n_points)
            elif "sin3d" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingSine3D(config.hidden_size, n_points=n_points)
            # elif "slp" in self.config.world_position_embedding_type:
            #     self.world_position_embedding = PositionEmbeddingSine3DMLP(config.hidden_size, n_points=n_points)
            

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        if getattr(model_args, "mm_compressor_type", None) is not None:
            self.config.mm_compressor_type = model_args.mm_compressor_type
        if getattr(model_args, "mm_compressor_config", None) is not None:
            self.config.mm_compressor_config = model_args.mm_compressor_config
        if getattr(model_args, "mm_projector_compressor_type", None) is not None:
            self.config.mm_projector_compressor_type = model_args.mm_projector_compressor_type
        if getattr(model_args, "mm_projector_compressor_config", None) is not None:
            self.config.mm_projector_compressor_config = model_args.mm_projector_compressor_config
        if getattr(model_args, "mm_llm_compressor_type", None) is not None:
            self.config.mm_llm_compressor_type = model_args.mm_llm_compressor_type
        if getattr(model_args, "mm_llm_compressor_config", None) is not None:
            self.config.mm_llm_compressor_config = model_args.mm_llm_compressor_config

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if "unpad" in mm_patch_merge_type and not hasattr(self, "image_newline"):
            embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
            self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)

        # Training may override compressor settings after loading a checkpoint whose
        # config had no compressor. Rebuild from the current config here instead of
        # keeping a stale IdentityCompressor created at model construction time.
        self.mm_projector_compressor = build_compressor_from_llava_config(self.config, location="projector")
        self.mm_llm_compressor = build_compressor_from_llava_config(self.config, location="llm")
        self.mm_compressor = self.mm_projector_compressor

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def _init_compression_token_stats(self):
        self._compression_token_stats = {
            "total_original_tokens": 0,
            "total_output_tokens": 0,
            "video_count": 0,
        }
        self._last_compression_profile = {"samples": []}

    def get_compression_token_stats(self):
        if not hasattr(self, "_compression_token_stats"):
            self._init_compression_token_stats()
        stats = self._compression_token_stats.copy()
        video_count = stats["video_count"]
        stats["avg_output_tokens"] = stats["total_output_tokens"] / video_count if video_count > 0 else 0
        original = stats["total_original_tokens"]
        stats["compression_ratio"] = stats["total_output_tokens"] / original if original > 0 else 0
        return stats

    def reset_compression_token_stats(self):
        self._init_compression_token_stats()

    def get_last_compression_profile(self):
        if not hasattr(self, "_last_compression_profile"):
            self._last_compression_profile = {"samples": []}
        projector_profile = self._last_compression_profile
        llm_profile = None
        llm_model = self.get_model()
        if hasattr(llm_model, "get_last_llm_compression_profile"):
            llm_profile = llm_model.get_last_llm_compression_profile()

        projector_samples = self._normalize_profile_samples(projector_profile)
        llm_samples = self._normalize_profile_samples(llm_profile)
        merged_samples = []
        max_samples = max(len(projector_samples), len(llm_samples))
        for sample_idx in range(max_samples):
            merged_samples.append(
                self._merge_stage_profiles(
                    projector_samples[sample_idx] if sample_idx < len(projector_samples) else None,
                    llm_samples[sample_idx] if sample_idx < len(llm_samples) else None,
                )
            )

        merged_samples = [sample for sample in merged_samples if sample]
        if not merged_samples:
            return {}
        if len(merged_samples) == 1:
            return merged_samples[0].copy()
        return {"samples": [sample.copy() for sample in merged_samples if sample]}

    def reset_last_compression_profile(self):
        self._last_compression_profile = {"samples": []}
        llm_model = self.get_model()
        if hasattr(llm_model, "reset_last_llm_compression_profile"):
            llm_model.reset_last_llm_compression_profile()

    def pop_last_compression_profile(self):
        profile = self.get_last_compression_profile()
        self.reset_last_compression_profile()
        return profile

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def _normalize_profile_samples(self, profile):
        if not profile:
            return []
        if isinstance(profile, dict):
            samples = profile.get("samples")
            if isinstance(samples, list):
                return [sample.copy() for sample in samples if sample]
            if profile:
                return [profile.copy()]
        return []

    def _resolve_prefill_final_visual_tokens(self, profile):
        if not isinstance(profile, dict):
            return None
        text_prompt_tokens = profile.get("text_prompt_tokens")
        visual_format_tokens = profile.get("visual_format_tokens")
        layer_token_lengths = profile.get("prefill_layer_token_lengths")
        if (
            isinstance(layer_token_lengths, list)
            and layer_token_lengths
            and isinstance(text_prompt_tokens, (int, float))
            and isinstance(visual_format_tokens, (int, float))
            and all(isinstance(length, (int, float)) for length in layer_token_lengths)
        ):
            per_layer_visual_tokens = [
                max(int(length) - int(text_prompt_tokens) - int(visual_format_tokens), 0)
                for length in layer_token_lengths
            ]
            return float(sum(per_layer_visual_tokens)) / float(len(per_layer_visual_tokens))

        llm_input_tokens = profile.get("llm_stage_input_tokens", profile.get("compressor_input_tokens"))
        llm_output_tokens = profile.get("llm_stage_output_tokens", profile.get("compressor_output_tokens"))
        llm_prune_layer = profile.get("llm_prune_layer")
        if (
            isinstance(layer_token_lengths, list)
            and layer_token_lengths
            and isinstance(llm_input_tokens, (int, float))
            and isinstance(llm_output_tokens, (int, float))
            and isinstance(llm_prune_layer, (int, float))
        ):
            layer_count = len(layer_token_lengths)
            prune_layer = min(max(int(llm_prune_layer), 0), layer_count)
            total_visual_tokens = int(llm_input_tokens) * prune_layer + int(llm_output_tokens) * (layer_count - prune_layer)
            return float(total_visual_tokens) / float(layer_count)

        visual_patch_tokens = profile.get("visual_patch_tokens")
        if isinstance(visual_patch_tokens, (int, float)):
            return int(visual_patch_tokens)

        projector_output_tokens = profile.get("projector_stage_output_tokens")
        if isinstance(projector_output_tokens, (int, float)):
            return int(projector_output_tokens)

        compressor_output_tokens = profile.get("compressor_output_tokens")
        if isinstance(compressor_output_tokens, (int, float)):
            return int(compressor_output_tokens)

        fallback_value = profile.get("prefill_final_visual_tokens")
        if isinstance(fallback_value, (int, float)):
            return float(fallback_value)

        final_sequence_length = profile.get("final_sequence_length")
        if (
            isinstance(final_sequence_length, (int, float))
            and isinstance(text_prompt_tokens, (int, float))
            and isinstance(visual_format_tokens, (int, float))
        ):
            return max(int(final_sequence_length) - int(text_prompt_tokens) - int(visual_format_tokens), 0)

        prompt_sequence_length = profile.get("prompt_sequence_length")
        if (
            isinstance(prompt_sequence_length, (int, float))
            and isinstance(text_prompt_tokens, (int, float))
            and isinstance(visual_format_tokens, (int, float))
        ):
            return max(int(prompt_sequence_length) - int(text_prompt_tokens) - int(visual_format_tokens), 0)
        return None

    def _merge_stage_profiles(self, projector_profile, llm_profile):
        if projector_profile is None:
            if llm_profile is None:
                return {}
            merged = llm_profile.copy()
            prefill_final_visual_tokens = self._resolve_prefill_final_visual_tokens(merged)
            if prefill_final_visual_tokens is not None:
                merged["prefill_final_visual_tokens"] = float(prefill_final_visual_tokens)
            return merged
        if llm_profile is None:
            merged = projector_profile.copy()
            merged["projector_stage_input_tokens"] = projector_profile.get("compressor_input_tokens")
            merged["projector_stage_output_tokens"] = projector_profile.get("compressor_output_tokens")
            merged["projector_stage_keep_ratio"] = projector_profile.get("token_keep_ratio")
            prefill_final_visual_tokens = self._resolve_prefill_final_visual_tokens(merged)
            if prefill_final_visual_tokens is not None:
                merged["prefill_final_visual_tokens"] = float(prefill_final_visual_tokens)
            return merged

        merged = projector_profile.copy()
        merged["compressor_name"] = llm_profile.get("compressor_name", projector_profile.get("compressor_name"))
        projector_input_tokens = projector_profile.get("compressor_input_tokens")
        projector_output_tokens = projector_profile.get("compressor_output_tokens")
        llm_input_tokens = llm_profile.get("compressor_input_tokens")
        llm_output_tokens = llm_profile.get("compressor_output_tokens")
        merged["projector_stage_input_tokens"] = projector_input_tokens
        merged["projector_stage_output_tokens"] = projector_output_tokens
        merged["projector_stage_keep_ratio"] = projector_profile.get("token_keep_ratio")
        merged["llm_stage_input_tokens"] = llm_input_tokens
        merged["llm_stage_output_tokens"] = llm_output_tokens
        if isinstance(llm_input_tokens, (int, float)) and isinstance(llm_output_tokens, (int, float)):
            merged["llm_stage_keep_ratio"] = (
                float(llm_output_tokens) / float(llm_input_tokens)
                if int(llm_input_tokens) > 0
                else 1.0
            )
        merged["compressor_input_tokens"] = projector_input_tokens if projector_input_tokens is not None else llm_input_tokens
        merged["compressor_output_tokens"] = llm_output_tokens if llm_output_tokens is not None else projector_output_tokens
        if isinstance(merged["compressor_input_tokens"], (int, float)) and isinstance(merged["compressor_output_tokens"], (int, float)):
            compressor_input_tokens = int(merged["compressor_input_tokens"])
            compressor_output_tokens = int(merged["compressor_output_tokens"])
            merged["compressor_input_tokens"] = compressor_input_tokens
            merged["compressor_output_tokens"] = compressor_output_tokens
            merged["token_keep_ratio"] = (
                float(compressor_output_tokens) / float(compressor_input_tokens)
                if compressor_input_tokens > 0
                else 1.0
            )
        if "llm_prune_layer" in llm_profile:
            merged["llm_prune_layer"] = llm_profile["llm_prune_layer"]
        if "prune_method" in llm_profile:
            merged["prune_method"] = llm_profile["prune_method"]
        merged["text_prompt_tokens"] = llm_profile.get("text_prompt_tokens", projector_profile.get("text_prompt_tokens"))
        merged["final_sequence_length"] = llm_profile.get("final_sequence_length", projector_profile.get("final_sequence_length"))
        for key in ("visual_patch_tokens", "visual_sequence_tokens", "visual_format_tokens"):
            if key in llm_profile:
                merged[key] = llm_profile[key]
            elif key in projector_profile:
                merged[key] = projector_profile[key]
        merged["prompt_sequence_length_before_prune"] = llm_profile.get(
            "prompt_sequence_length_before_prune",
            projector_profile.get("prompt_sequence_length_before_prune"),
        )
        merged["prompt_sequence_length"] = llm_profile.get("prompt_sequence_length", projector_profile.get("prompt_sequence_length"))
        if "prefill_layer_token_lengths" in llm_profile:
            merged["prefill_layer_token_lengths"] = list(llm_profile["prefill_layer_token_lengths"])
        prefill_final_visual_tokens = self._resolve_prefill_final_visual_tokens(merged)
        if prefill_final_visual_tokens is not None:
            merged["prefill_final_visual_tokens"] = float(prefill_final_visual_tokens)
        return merged

    def _record_video_token_stats(self, original_tokens: int, output_tokens: int):
        if not hasattr(self, "_compression_token_stats"):
            self._init_compression_token_stats()
        self._compression_token_stats["total_original_tokens"] += original_tokens
        self._compression_token_stats["total_output_tokens"] += output_tokens
        self._compression_token_stats["video_count"] += 1

    def _build_video_compression_profile(
        self,
        *,
        compressor_name: str,
        input_tokens: int,
        output_tokens: int,
    ):
        return {
            "compressor_name": compressor_name,
            "compressor_input_tokens": int(input_tokens),
            "compressor_output_tokens": int(output_tokens),
            "token_keep_ratio": (float(output_tokens) / float(input_tokens)) if input_tokens > 0 else 1.0,
        }

    def _flatten_video_position_coords(self, coords: torch.Tensor, device: torch.device) -> torch.Tensor:
        if coords.dim() == 2 and coords.shape[-1] == 3:
            return coords.to(device=device)
        if coords.dim() == 3 and coords.shape[-1] == 3:
            return coords.to(device=device).reshape(-1, 3)
        raise NotImplementedError("mRoPE with compressed video tokens currently requires coordinates shaped as (tokens, 3) or (frames, tokens, 3).")

    def _build_video_patch_positions(
        self,
        num_frames: int,
        frame_shape,
        device: torch.device,
    ) -> torch.Tensor:
        height, width = frame_shape
        row_ids = torch.arange(height, device=device, dtype=torch.float32)
        col_ids = torch.arange(width, device=device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(row_ids, col_ids, indexing="ij")
        patch_positions = torch.stack((row_grid, col_grid), dim=-1).view(1, height * width, 2)
        return patch_positions.repeat(num_frames, 1, 1)

    def _compress_video_tokens(
        self,
        image_feat: torch.Tensor,
        coords: torch.Tensor = None,
        grouping_coords: torch.Tensor = None,
        compressor=None,
        raw_features_before_proj: torch.Tensor = None,
        attn_weights: torch.Tensor = None,
    ):
        pooled_feat = self.get_2dPool(image_feat)
        pooled_raw_features = None if raw_features_before_proj is None else self.get_2dPool(raw_features_before_proj)
        pooled_attn_weights = None if attn_weights is None else self.get_2dPool_scores(attn_weights)
        pooled_tokens = pooled_feat.shape[1]
        side = math.isqrt(pooled_tokens)
        if side * side != pooled_tokens:
            raise ValueError(f"Expected square pooled features, got {pooled_tokens} tokens.")

        baseline_tokens = image_feat.shape[0] * pooled_tokens
        compressor_name = getattr(compressor, "_compressor_name", "none") if compressor is not None else "none"
        if compressor is None:
            if coords is not None and coords.shape[:2] != pooled_feat.shape[:2]:
                raise ValueError(
                    "Video coordinates must stay aligned with pooled features before compression: "
                    f"{tuple(coords.shape)} vs {tuple(pooled_feat.shape)}"
                )
            self._record_video_token_stats(baseline_tokens, pooled_feat.shape[0] * pooled_feat.shape[1])
            profile = self._build_video_compression_profile(
                compressor_name=compressor_name,
                input_tokens=baseline_tokens,
                output_tokens=pooled_feat.shape[0] * pooled_feat.shape[1],
            )
            return pooled_feat, coords, None, (side, side), profile

        required_inputs = compressor.get_required_inputs()
        if required_inputs.get("raw_features_before_proj", False) and pooled_raw_features is None:
            raise ValueError(
                f"Configured compressor '{compressor_name}' requires raw_features_before_proj."
            )
        if required_inputs.get("attn_weights", False) and pooled_attn_weights is None:
            raise ValueError(
                f"Configured compressor '{compressor_name}' requires pooled visual attentions."
            )

        compress_output = compressor.compress(
            features=pooled_feat,
            num_frames=pooled_feat.shape[0],
            frame_shape=(side, side),
            coordinates=coords,
            grouping_coordinates=grouping_coords,
            raw_features_before_proj=pooled_raw_features,
            attn_weights=pooled_attn_weights,
        )
        compressed_coords = compress_output.metadata.get("compressed_coordinates", coords)
        if coords is not None and compressed_coords is None:
            raise ValueError("A video compressor must return compressed_coordinates when coordinates are provided.")
        if compressed_coords is not None and compressed_coords.shape[:2] != compress_output.features.shape[:2]:
            raise ValueError(
                "Compressed coordinates must stay aligned with compressed features: "
                f"{tuple(compressed_coords.shape)} vs {tuple(compress_output.features.shape)}"
            )
        compressed_tokens = compress_output.features.shape[0] * compress_output.features.shape[1]
        self._record_video_token_stats(baseline_tokens, compressed_tokens)
        profile = self._build_video_compression_profile(
            compressor_name=compressor_name,
            input_tokens=baseline_tokens,
            output_tokens=compressed_tokens,
        )
        return compress_output.features, compressed_coords, compress_output, (side, side), profile

    def _split_compressed_video_segments(
        self,
        image_feature: torch.Tensor,
        coords: torch.Tensor,
        compress_output,
        frame_shape,
    ):
        if compress_output is None:
            feature_segments = [image_feature[frame_idx] for frame_idx in range(image_feature.shape[0])]
            coord_segments = None if coords is None else [coords[frame_idx] for frame_idx in range(coords.shape[0])]
            patch_position_segments = [
                patch_positions for patch_positions in self._build_video_patch_positions(
                    num_frames=image_feature.shape[0],
                    frame_shape=frame_shape,
                    device=image_feature.device,
                )
            ]
            return feature_segments, coord_segments, patch_position_segments, None

        metadata = compress_output.metadata
        raw_compressed_features = getattr(compress_output, "features", None)
        if image_feature.dim() != 3 or image_feature.shape[0] != 1:
            raise ValueError(
                "Compressed video features for spatial_unpad must have shape (1, tokens, hidden), "
                f"got {tuple(image_feature.shape)}."
            )
        if raw_compressed_features is not None and (
            raw_compressed_features.dim() != 3
            or raw_compressed_features.shape[:2] != image_feature.shape[:2]
        ):
            raise ValueError(
                "Positioned compressed video features must stay aligned with compressor output: "
                f"{tuple(image_feature.shape)} vs {tuple(raw_compressed_features.shape)}."
            )

        flat_features = image_feature[0]
        flat_patch_positions = metadata["compressed_patch_positions"]
        if flat_patch_positions.dim() != 3 or flat_patch_positions.shape[0] != 1:
            raise ValueError(
                "Compressed video patch positions for spatial_unpad must have shape (1, tokens, 2), "
                f"got {tuple(flat_patch_positions.shape)}."
            )
        flat_patch_positions = flat_patch_positions[0]
        flat_coords = None
        if coords is None:
            coords = metadata.get("compressed_coordinates")
        if coords is not None:
            if coords.dim() < 3 or coords.shape[0] != 1:
                raise ValueError(
                    "Compressed video coordinates for spatial_unpad must have shape (1, tokens, ...), "
                    f"got {tuple(coords.shape)}."
                )
            flat_coords = coords[0]

        flat_priority = metadata.get("projector_patch_priority")
        if flat_priority is not None:
            flat_priority = torch.as_tensor(flat_priority).flatten()
            if int(flat_priority.numel()) != int(flat_features.shape[0]):
                raise ValueError(
                    "Projector patch priorities must align with compressed video features: "
                    f"priority={int(flat_priority.numel())}, features={int(flat_features.shape[0])}."
                )

        feature_segments = []
        coord_segments = [] if flat_coords is not None else None
        patch_position_segments = []
        priority_segments = [] if flat_priority is not None else None
        offset = 0
        hidden_dim = flat_features.shape[-1]

        if "frame_token_counts" in metadata:
            frame_token_counts = metadata["frame_token_counts"]
            if not isinstance(frame_token_counts, (list, tuple)):
                raise ValueError(
                    "Compressed frame_token_counts must be a list or tuple, "
                    f"got {type(frame_token_counts).__name__}."
                )
            for frame_token_count in frame_token_counts:
                frame_token_count = int(frame_token_count)
                next_offset = offset + frame_token_count
                feature_segments.append(flat_features[offset:next_offset])
                patch_position_segments.append(flat_patch_positions[offset:next_offset])
                if coord_segments is not None:
                    coord_segments.append(flat_coords[offset:next_offset])
                if priority_segments is not None:
                    priority_segments.append(flat_priority[offset:next_offset])
                offset = next_offset

            if offset != flat_features.shape[0]:
                raise ValueError(
                    "Compressed frame_token_counts do not cover the full token sequence: "
                    f"consumed {offset}, total {flat_features.shape[0]}."
                )
            return feature_segments, coord_segments, patch_position_segments, priority_segments

        required_keys = (
            "static_sizes",
            "dynamic_sizes",
            "dynamic_tokens_per_frame",
            "window_sizes",
            "compressed_patch_positions",
        )
        missing_keys = [key for key in required_keys if key not in metadata]
        if missing_keys:
            raise ValueError(f"Compressed video metadata is missing required keys: {missing_keys}")

        for static_size, dynamic_size, dynamic_tokens_per_frame, window_size in zip(
            metadata["static_sizes"],
            metadata["dynamic_sizes"],
            metadata["dynamic_tokens_per_frame"],
            metadata["window_sizes"],
        ):
            if static_size:
                next_offset = offset + static_size
                feature_segments.append(flat_features[offset:next_offset])
                if coord_segments is not None:
                    coord_segments.append(flat_coords[offset:next_offset])
                if priority_segments is not None:
                    priority_segments.append(flat_priority[offset:next_offset])
                patch_position_segments.append(flat_patch_positions[offset:next_offset])
                offset = next_offset

            if dynamic_size:
                if dynamic_tokens_per_frame * window_size != dynamic_size:
                    raise ValueError(
                        "Compressed dynamic token metadata is inconsistent: "
                        f"dynamic_size={dynamic_size}, per_frame={dynamic_tokens_per_frame}, window_size={window_size}."
                    )
                next_offset = offset + dynamic_size
                dynamic_features = flat_features[offset:next_offset].reshape(window_size, dynamic_tokens_per_frame, hidden_dim)
                dynamic_patch_positions = flat_patch_positions[offset:next_offset].reshape(window_size, dynamic_tokens_per_frame, flat_patch_positions.shape[-1])
                for frame_idx in range(window_size):
                    feature_segments.append(dynamic_features[frame_idx])
                    patch_position_segments.append(dynamic_patch_positions[frame_idx])
                if coord_segments is not None:
                    dynamic_coords = flat_coords[offset:next_offset].reshape(
                        window_size,
                        dynamic_tokens_per_frame,
                        *flat_coords.shape[1:],
                    )
                    for frame_idx in range(window_size):
                        coord_segments.append(dynamic_coords[frame_idx])
                if priority_segments is not None:
                    dynamic_priority = flat_priority[offset:next_offset].reshape(
                        window_size, dynamic_tokens_per_frame
                    )
                    for frame_idx in range(window_size):
                        priority_segments.append(dynamic_priority[frame_idx])
                offset = next_offset

        if offset != flat_features.shape[0]:
            raise ValueError(
                "Compressed video metadata does not cover the full token sequence: "
                f"consumed {offset}, total {flat_features.shape[0]}."
            )

        return feature_segments, coord_segments, patch_position_segments, priority_segments

    def _serialize_video_grid_segments(
        self,
        feature_segments,
        coord_segments=None,
        patch_position_segments=None,
        projector_score_segments=None,
        projector_coordinate_segments=None,
        projector_priority_segments=None,
        frame_shape=None,
        newline_strategy: str = "grid_drop",
        preserve_segment_order: bool = False,
        return_serialization_metadata: bool = False,
    ):
        if frame_shape is None:
            raise ValueError("frame_shape is required to serialize video grid segments.")
        height, width = frame_shape
        if width <= 0:
            raise ValueError(f"frame_shape width must be positive, got {width}.")
        if newline_strategy not in ("grid_drop", "full_grid_drop", "frame_newline", "one_token"):
            raise ValueError(f"Unexpected video newline_strategy: {newline_strategy}")

        serialized_features = []
        serialized_coords = [] if coord_segments is not None else None
        serialized_projector_scores = []
        serialized_projector_coordinates = []
        serialized_projector_priority = []
        newline_token = self.model.image_newline[None]
        serialization_metadata = {
            "patch_positions_by_view": [],
        }
        serialized_offset = 0

        for segment_idx, feature_segment in enumerate(feature_segments):
            if feature_segment.dim() != 2:
                raise ValueError(f"Expected video segment features shaped as (tokens, hidden), got {tuple(feature_segment.shape)}.")
            if feature_segment.shape[0] == 0 and newline_strategy == "grid_drop":
                serialization_metadata["patch_positions_by_view"].append([])
                continue

            coord_segment = None if coord_segments is None else coord_segments[segment_idx]
            if coord_segment is not None and coord_segment.shape[0] != feature_segment.shape[0]:
                raise ValueError(
                    "Video segment coordinates must stay aligned with segment features: "
                    f"{tuple(coord_segment.shape)} vs {tuple(feature_segment.shape)}."
                )
            patch_position_segment = None if patch_position_segments is None else patch_position_segments[segment_idx]
            if patch_position_segment is not None and patch_position_segment.shape[0] != feature_segment.shape[0]:
                raise ValueError(
                    "Video segment patch positions must stay aligned with segment features: "
                    f"{tuple(patch_position_segment.shape)} vs {tuple(feature_segment.shape)}."
                )
            projector_score_segment = (
                None if projector_score_segments is None else projector_score_segments[segment_idx]
            )
            if projector_score_segment is not None and projector_score_segment.numel() != feature_segment.shape[0]:
                raise ValueError(
                    "Projector scores must stay aligned with video features during serialization: "
                    f"{int(projector_score_segment.numel())} vs {int(feature_segment.shape[0])}."
                )
            projector_coordinate_segment = (
                None
                if projector_coordinate_segments is None
                else projector_coordinate_segments[segment_idx]
            )
            if projector_coordinate_segment is not None and (
                projector_coordinate_segment.dim() != 2
                or projector_coordinate_segment.shape != (feature_segment.shape[0], 3)
            ):
                raise ValueError(
                    "Projector grouping coordinates must have shape [tokens, 3] during serialization, "
                    f"got {tuple(projector_coordinate_segment.shape)}."
                )
            projector_priority_segment = (
                None if projector_priority_segments is None else projector_priority_segments[segment_idx]
            )
            if projector_priority_segment is not None and (
                projector_priority_segment.dim() != 1
                or projector_priority_segment.shape[0] != feature_segment.shape[0]
            ):
                raise ValueError(
                    "Projector priorities must stay aligned with video features during serialization: "
                    f"{tuple(projector_priority_segment.shape)} vs {tuple(feature_segment.shape)}."
                )

            if patch_position_segment is None:
                patch_position_segment = self._build_video_patch_positions(
                    num_frames=1,
                    frame_shape=frame_shape,
                    device=feature_segment.device,
                )[0][: feature_segment.shape[0]]

            patch_position_segment = patch_position_segment.to(device=feature_segment.device)
            row_ids = patch_position_segment[:, 0].round().to(dtype=torch.long).clamp_(0, height - 1)
            col_ids = patch_position_segment[:, 1].round().to(dtype=torch.long).clamp_(0, width - 1)
            if preserve_segment_order:
                order = torch.arange(feature_segment.shape[0], device=feature_segment.device)
            else:
                order = (row_ids * width + col_ids).argsort(stable=True)

            ordered_features = feature_segment[order]
            ordered_rows = row_ids[order]
            ordered_coords = None if coord_segment is None else coord_segment[order]
            if projector_score_segment is not None:
                serialized_projector_scores.append(
                    projector_score_segment.to(device=order.device).flatten().index_select(0, order)
                )
            if projector_coordinate_segment is not None:
                serialized_projector_coordinates.append(
                    projector_coordinate_segment.to(device=order.device).index_select(0, order)
                )
            if projector_priority_segment is not None:
                serialized_projector_priority.append(
                    projector_priority_segment.to(device=order.device).index_select(0, order)
                )
            segment_patch_positions = []

            if newline_strategy in ("frame_newline", "one_token"):
                if ordered_features.shape[0] > 0:
                    serialized_features.append(ordered_features)
                    segment_patch_positions.extend(range(serialized_offset, serialized_offset + ordered_features.shape[0]))
                    serialized_offset += ordered_features.shape[0]
                    newline_device = ordered_features.device
                    newline_dtype = ordered_features.dtype
                else:
                    newline_device = feature_segment.device
                    newline_dtype = feature_segment.dtype

                if newline_strategy == "frame_newline":
                    serialized_features.append(newline_token.to(device=newline_device, dtype=newline_dtype))
                    serialized_offset += 1

                if serialized_coords is not None:
                    if ordered_features.shape[0] > 0:
                        serialized_coords.append(ordered_coords)
                    if newline_strategy == "frame_newline":
                        coord_device = coord_segment.device
                        coord_dtype = coord_segment.dtype
                        coord_shape = coord_segment.shape[1:]
                        serialized_coords.append(
                            torch.zeros(
                                (1, *coord_shape),
                                device=coord_device,
                                dtype=coord_dtype,
                            )
                        )
            elif newline_strategy == "full_grid_drop":
                for row_idx in range(height):
                    row_mask = ordered_rows == row_idx
                    row_features = ordered_features[row_mask]
                    if row_features.shape[0] > 0:
                        serialized_features.append(row_features)
                        segment_patch_positions.extend(range(serialized_offset, serialized_offset + row_features.shape[0]))
                        serialized_offset += row_features.shape[0]
                        newline_device = row_features.device
                        newline_dtype = row_features.dtype
                    else:
                        newline_device = feature_segment.device
                        newline_dtype = feature_segment.dtype

                    serialized_features.append(newline_token.to(device=newline_device, dtype=newline_dtype))
                    serialized_offset += 1

                    if serialized_coords is not None:
                        coord_device = coord_segment.device
                        coord_dtype = coord_segment.dtype
                        coord_shape = coord_segment.shape[1:]
                        if row_features.shape[0] > 0:
                            serialized_coords.append(ordered_coords[row_mask])
                        serialized_coords.append(
                            torch.zeros(
                                (1, *coord_shape),
                                device=coord_device,
                                dtype=coord_dtype,
                            )
                        )
            else:
                row_boundaries = torch.where(ordered_rows[1:] != ordered_rows[:-1])[0] + 1
                boundaries = torch.cat(
                    (
                        ordered_rows.new_tensor([0]),
                        row_boundaries,
                        ordered_rows.new_tensor([ordered_rows.shape[0]]),
                    )
                )

                for start_idx, end_idx in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
                    row_features = ordered_features[start_idx:end_idx]
                    serialized_features.append(row_features)
                    segment_patch_positions.extend(range(serialized_offset, serialized_offset + row_features.shape[0]))
                    serialized_offset += row_features.shape[0]
                    serialized_features.append(newline_token.to(device=row_features.device, dtype=row_features.dtype))
                    serialized_offset += 1

                    if serialized_coords is not None:
                        row_coords = ordered_coords[start_idx:end_idx]
                        serialized_coords.append(row_coords)
                        serialized_coords.append(
                            torch.zeros(
                                (1, *row_coords.shape[1:]),
                                device=row_coords.device,
                                dtype=row_coords.dtype,
                            )
                        )

            serialization_metadata["patch_positions_by_view"].append(segment_patch_positions)

        if newline_strategy == "one_token":
            if serialized_features:
                newline_device = serialized_features[-1].device
                newline_dtype = serialized_features[-1].dtype
            else:
                first_segment = feature_segments[0]
                newline_device = first_segment.device
                newline_dtype = first_segment.dtype
            serialized_features.append(newline_token.to(device=newline_device, dtype=newline_dtype))

            if serialized_coords is not None:
                coord_template = next((segment for segment in coord_segments if segment is not None), None)
                if coord_template is None:
                    raise ValueError("one_token coordinate serialization requires a coordinate template.")
                serialized_coords.append(
                    torch.zeros(
                        (1, *coord_template.shape[1:]),
                        device=coord_template.device,
                        dtype=coord_template.dtype,
                    )
                )

        if not serialized_features:
            raise ValueError("Cannot serialize an empty video token sequence.")

        flat_features = torch.cat(serialized_features, dim=0)
        flat_coords = None if serialized_coords is None else torch.cat(serialized_coords, dim=0)
        if return_serialization_metadata:
            serialization_metadata["visual_patch_token_count"] = sum(
                len(view_positions) for view_positions in serialization_metadata["patch_positions_by_view"]
            )
            serialization_metadata["visual_token_count"] = int(flat_features.shape[0])
            if projector_score_segments is not None:
                serialization_metadata["serialized_projector_patch_scores"] = torch.cat(
                    serialized_projector_scores, dim=0
                ).detach()
            if projector_coordinate_segments is not None:
                serialization_metadata["serialized_projector_patch_coordinates"] = torch.cat(
                    serialized_projector_coordinates, dim=0
                ).detach()
            if projector_priority_segments is not None:
                serialization_metadata["serialized_projector_patch_priority"] = torch.cat(
                    serialized_projector_priority, dim=0
                ).detach()
            return flat_features, flat_coords, serialization_metadata
        return flat_features, flat_coords

    def _format_video_tokens_for_spatial_unpad(
        self,
        image_feature: torch.Tensor,
        coords: torch.Tensor = None,
        compress_output=None,
        frame_shape=None,
        return_serialization_metadata: bool = False,
    ):
        if frame_shape is None:
            raise ValueError("frame_shape is required to serialize video tokens for spatial_unpad.")

        feature_segments, coord_segments, patch_position_segments, projector_priority_segments = self._split_compressed_video_segments(
            image_feature=image_feature,
            coords=coords,
            compress_output=compress_output,
            frame_shape=frame_shape,
        )
        projector_score_segments = None
        projector_coordinate_segments = None
        if compress_output is not None:
            projector_scores = compress_output.metadata.get("projector_patch_scores")
            projector_coordinates = compress_output.metadata.get("projector_patch_coordinates")
            segment_lengths = [int(segment.shape[0]) for segment in feature_segments]
            expected_tokens = sum(segment_lengths)
            if projector_scores is not None:
                projector_scores = torch.as_tensor(projector_scores).flatten()
                if int(projector_scores.numel()) != expected_tokens:
                    raise ValueError(
                        "Projector scores do not align with compressed Video3D features: "
                        f"scores={int(projector_scores.numel())}, features={expected_tokens}."
                    )
                projector_score_segments = list(torch.split(projector_scores, segment_lengths))
            if projector_coordinates is not None:
                projector_coordinates = torch.as_tensor(projector_coordinates)
                if projector_coordinates.dim() == 3 and projector_coordinates.shape[0] == 1:
                    projector_coordinates = projector_coordinates[0]
                if projector_coordinates.shape != (expected_tokens, 3):
                    raise ValueError(
                        "Projector grouping coordinates must align with compressed Video3D features: "
                        f"coordinates={tuple(projector_coordinates.shape)}, features={expected_tokens}."
                    )
                projector_coordinate_segments = list(
                    torch.split(projector_coordinates, segment_lengths, dim=0)
                )
        newline_strategy = "grid_drop"
        preserve_segment_order = False
        if compress_output is not None:
            newline_strategy = compress_output.metadata.get("newline_strategy", newline_strategy)
            preserve_segment_order = bool(compress_output.metadata.get("preserve_segment_order", False))
        return self._serialize_video_grid_segments(
            feature_segments=feature_segments,
            coord_segments=coord_segments,
            patch_position_segments=patch_position_segments,
            projector_score_segments=projector_score_segments,
            projector_coordinate_segments=projector_coordinate_segments,
            projector_priority_segments=projector_priority_segments,
            frame_shape=frame_shape,
            newline_strategy=newline_strategy,
            preserve_segment_order=preserve_segment_order,
            return_serialization_metadata=return_serialization_metadata,
        )

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def get_2dPool_scores(self, token_scores, stride=2):
        if token_scores.dim() == 3:
            token_scores = token_scores.mean(dim=1)
        if token_scores.dim() != 2:
            raise ValueError(f"Expected token_scores shaped as (frames, tokens), got {tuple(token_scores.shape)}.")

        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens = token_scores.shape
        if num_tokens != height * width:
            raise ValueError(
                "Token scores must match the vision tower grid before native pooling: "
                f"{num_tokens} vs {height * width}."
            )

        token_scores = token_scores.view(num_frames, 1, height, width)
        if self.config.mm_spatial_pool_mode == "average":
            token_scores = nn.functional.avg_pool2d(token_scores, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            token_scores = nn.functional.max_pool2d(token_scores, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            token_scores = nn.functional.interpolate(token_scores, size=scaled_shape, mode="bilinear")
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")

        return token_scores.view(num_frames, -1)


    def average_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]
        world_coords_avg = torch.nn.functional.avg_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        patch_num = world_coords_avg.shape[-1]
        world_coords_avg = world_coords_avg.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        return world_coords_avg

    def minmax_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]

        world_coords_max = torch.nn.functional.max_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_max = world_coords_max.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        world_coords_min = - torch.nn.functional.max_pool2d(-world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_min = world_coords_min.permute(0, 2, 3, 1)     # [32, 14, 14, 3]
        world_coords = torch.stack([world_coords_min, world_coords_max], dim=3) # [32, 14, 14, 2, 3]

        return world_coords
    
    def sample_n_points(self, world_coords, n_points=9):

        V, H, W, D = world_coords.size() # D = 3
        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :] 
        world_coords = world_coords.view(-1, 14, 27, 14, 27, 3).permute(0, 1, 3, 2, 4, 5)
        if n_points == 9:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
        elif n_points == 5:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            world_coords_sample = world_coords_sample[:, :, :, 0::2, :].reshape(V, 14, 14, 5, 3)
        elif n_points == 1:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            world_coords_sample = world_coords_sample[:, :, :, 4, :].reshape(V, 14, 14, 3)
        else:
            raise NotImplementedError
        
        return world_coords_sample

    def discrete_coords(self, world_coords, xyz_min):

        # V, H, W, D = world_coords.size() # D = 3
        # world_coords_discrete = (world_coords.view(-1, 3) - xyz_min.view(1, 3)) / self.config.voxel_size

        min_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)
        max_xyz_range = torch.tensor(self.config.max_xyz_range).to(world_coords.device)

        world_coords = torch.maximum(world_coords, min_xyz_range)
        world_coords = torch.minimum(world_coords, max_xyz_range)
        world_coords_discrete = (world_coords - min_xyz_range) / self.config.voxel_size
        world_coords_discrete = world_coords_discrete.round()

        return world_coords_discrete.detach()


    def encode_images(
        self,
        images,
        world_coords=None,
        output_attentions: bool = False,
        return_raw_features: bool = False,
    ):
        vision_tower = self.get_model().get_vision_tower()
        if output_attentions:
            try:
                vision_outputs = vision_tower(images, output_attentions=True)
            except TypeError as exc:
                raise NotImplementedError(
                    f"Vision tower '{type(vision_tower).__name__}' does not support output_attentions=True, "
                    "which is required by the configured compressor."
                ) from exc
        else:
            vision_outputs = vision_tower(images)

        if isinstance(vision_outputs, tuple):
            raw_image_features, image_attentions = vision_outputs
        else:
            raw_image_features = vision_outputs
            image_attentions = None
            if output_attentions:
                raise NotImplementedError(
                    f"Vision tower '{type(vision_tower).__name__}' did not return attentions when requested."
                )

        image_features = self.get_model().mm_projector(raw_image_features)
        if not output_attentions and not return_raw_features:
            return image_features

        aux_outputs = {}
        if return_raw_features:
            aux_outputs["raw_features_before_proj"] = raw_image_features
        if output_attentions:
            aux_outputs["attn_weights"] = image_attentions
        return image_features, aux_outputs


    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if isinstance(slower_img_feat, torch.Tensor):
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            return image_feature
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images, 
        modalities=["image"], 
        image_sizes=None, 
        video_dict=None,
        use_object_proposals: bool = False,
    ):
        

        object_boxes = None
        if use_object_proposals:
            object_boxes = video_dict["objects"][0]
            object_boxes_center = object_boxes[:, :3]
            object_features = []
            obj_num = len(object_boxes)

            object_patch = []
            # ignore the batch dimension here
            world_coords = video_dict["world_coords"][0]

            for l in range(obj_num):
                box = object_boxes[l]
                min_xyz = box[:3] - box[3:] / 2
                max_xyz = box[:3] + box[3:] / 2
                
                if "patch27" in self.config.object_feature_type:
                    world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 14, 27, 14, 27, 3).transpose(2, 3).flatten(3, 4)  # [32, 14, 14, 27*27, 3]
                    cur_object_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)     # [32, 14, 14, 27*27]
                    cur_object_patch = cur_object_patch.sum(dim=3) >= int(27 * 27 * 0.25)
                    object_patch.append(cur_object_patch)
                elif "patch14" in self.config.object_feature_type:
                    world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 27, 14, 27, 14, 3).transpose(2, 3).flatten(3, 4)  # [32, 14, 14, 27*27, 3]
                    cur_object_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)     # [32, 14, 14, 27*27]
                    cur_object_patch = cur_object_patch.sum(dim=3) >= int(14 * 14 * 0.5)
                    object_patch.append(cur_object_patch)
                else:
                    raise NotImplementedError
        
        
        use_mrope_position_embedding = False
        use_sin3d_pe = False
        use_mlp_pe = False
        if hasattr(self.config, 'world_position_embedding_type') and past_key_values is None:
            use_mrope_position_embedding = 'mrope' in self.config.world_position_embedding_type
            use_sin3d_pe = "sin3d" in self.config.world_position_embedding_type
            use_mlp_pe = "mlp" in self.config.world_position_embedding_type
            B = input_ids.shape[0]
            world_coords = video_dict['world_coords']
            xyz_min = world_coords.view(B, -1, 3).min(dim=1)[0]

            if len(video_dict['box_input']):
                box_input = video_dict['box_input']     # [1, 3]
            else:
                box_input = None

            n_points = 1
            if 'avg' in self.config.world_position_embedding_type:
                world_coords = [self.average_coordinate_in_patch(coords) for coords in world_coords]
            elif "sample9" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=9) for coords in world_coords]
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=5) for coords in world_coords]
                n_points = 5
            elif "sample1" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=1) for coords in world_coords]
            elif "minmax" in self.config.world_position_embedding_type:
                world_coords = [self.minmax_coordinate_in_patch(coords) for coords in world_coords]
                n_points = 2

            if n_points > 1:
                if box_input is not None:
                    box_input = box_input[:, None, :].repeat(1, n_points, 1)
                if object_boxes is not None:
                    object_boxes_center = object_boxes_center[:, None, :].repeat(1, n_points, 1)

            if 'discrete' in self.config.world_position_embedding_type or use_mrope_position_embedding:
                world_coords_discrete = [self.discrete_coords(coords, xyz_min[i]) for i, coords in enumerate(world_coords)]
                if box_input is not None:
                    box_input = self.discrete_coords(box_input, None)
                if object_boxes is not None:
                    object_boxes_center = self.discrete_coords(object_boxes_center, None)


        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            llm_model = self.get_model()
            if hasattr(llm_model, "clear_pending_llm_compression_state"):
                llm_model.clear_pending_llm_compression_state()
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        self.reset_last_compression_profile()

        if isinstance(modalities, str):
            modalities = [modalities]

        object_feature_type = getattr(self.config, "object_feature_type", "") or ""
        projector_compressor = getattr(self.get_model(), "mm_projector_compressor", None)
        projector_compressor_type, _ = get_compressor_spec_from_llava_config(self.config, location="projector")
        configured_projector_compressor = (
            projector_compressor is not None
            and projector_compressor_type not in ("none", "identity", None, "")
        )
        llm_compressor = getattr(self.get_model(), "mm_llm_compressor", None)
        llm_compressor_type, _ = get_compressor_spec_from_llava_config(self.config, location="llm")
        configured_llm_compressor = (
            llm_compressor is not None
            and llm_compressor_type not in ("none", "identity", None, "")
        )

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            projector_required_inputs = projector_compressor.get_required_inputs() if configured_projector_compressor else {}
            projector_requires_coordinates = configured_projector_compressor and projector_required_inputs.get("coordinates", False)
            projector_requires_raw_features = configured_projector_compressor and projector_required_inputs.get("raw_features_before_proj", False)
            projector_requires_attn = configured_projector_compressor and projector_required_inputs.get("attn_weights", False)
            if configured_projector_compressor and use_object_proposals and "patch14" not in object_feature_type:
                raise NotImplementedError(
                    "Video-3D-LLM grounding with projector-level compression currently requires "
                    "object_feature_type containing 'patch14' so proposal features stay on the "
                    "pre-compression visual branch."
                )
            if configured_projector_compressor and not projector_compressor.supports_video():
                raise NotImplementedError(
                    f"Configured projector compressor '{projector_compressor_type}' does not support video inputs."
                )
            if configured_llm_compressor and not llm_compressor.supports_video():
                raise NotImplementedError(
                    f"Configured llm compressor '{llm_compressor_type}' does not support video inputs."
                )
            if configured_projector_compressor and getattr(self.config, "mm_patch_merge_type", "flat") != "spatial_unpad":
                raise NotImplementedError(
                    "Video compression in Video-3D-LLM requires mm_patch_merge_type='spatial_unpad' "
                    "to stay consistent with the native video token layout."
                )
            if configured_projector_compressor and getattr(self.config, "mm_newline_position", "one_token") != "grid":
                raise NotImplementedError(
                    "Video compression in Video-3D-LLM requires mm_newline_position='grid' "
                    "to stay consistent with the native spatial_unpad recipe."
                )
            if (configured_projector_compressor or configured_llm_compressor) and getattr(self.config, "add_faster_video", False):
                raise NotImplementedError("Video compression is not supported together with add_faster_video in Video-3D-LLM.")

            encoded_outputs = self.encode_images(
                concat_images,
                output_attentions=projector_requires_attn,
                return_raw_features=projector_requires_raw_features,
            )
            if isinstance(encoded_outputs, tuple):
                encoded_image_features, encoder_aux = encoded_outputs
            else:
                encoded_image_features = encoded_outputs
                encoder_aux = {}

            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            raw_features_before_proj = None
            if projector_requires_raw_features:
                raw_features_before_proj = torch.split(encoder_aux["raw_features_before_proj"], split_sizes)
            attention_scores = None
            if projector_requires_attn:
                attention_scores = torch.split(encoder_aux["attn_weights"], split_sizes)

            image_features = []
            video_position_coords = []
            video_compression_outputs = []
            video_frame_shapes = []
            video_metric_profiles = []
            video_llm_serialization_metadata = []

            projector_compressor_world_coords = None
            if configured_projector_compressor and projector_requires_coordinates:
                if video_dict is None or "world_coords" not in video_dict:
                    raise ValueError(
                        f"Configured projector compressor '{projector_compressor_type}' requires world_coords in video_dict."
                    )
                # Projector-side geometric compressors should always operate on continuous
                # metric patch coordinates. These coordinates are intentionally decoupled
                # from any PE-specific discretization or mRoPE routing.
                projector_compressor_world_coords = [
                    self.average_coordinate_in_patch(coords)
                    for coords in video_dict["world_coords"]
                ]

            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    position_coord_tensor = None
                    grouping_coord_tensor = None
                    if use_sin3d_pe or use_mlp_pe or use_mrope_position_embedding:
                        if "discrete" in self.config.world_position_embedding_type or use_mrope_position_embedding:
                            position_coord_tensor = world_coords_discrete[idx].flatten(1, 2)
                        else:
                            position_coord_tensor = world_coords[idx].flatten(1, 2)
                    if projector_compressor_world_coords is not None:
                        grouping_coord_tensor = projector_compressor_world_coords[idx].flatten(1, 2)
                        if position_coord_tensor is None:
                            position_coord_tensor = grouping_coord_tensor

                    image_feature, position_coord_tensor, compress_output, frame_shape, compression_profile = self._compress_video_tokens(
                        image_feat,
                        coords=position_coord_tensor,
                        grouping_coords=grouping_coord_tensor,
                        compressor=projector_compressor if configured_projector_compressor else None,
                        raw_features_before_proj=(
                            raw_features_before_proj[idx]
                            if raw_features_before_proj is not None
                            else None
                        ),
                        attn_weights=(
                            attention_scores[idx]
                            if attention_scores is not None
                            else None
                        ),
                    )
                    if position_coord_tensor is not None and (
                        (hasattr(self.config, "world_position_embedding_type") and "discrete" in self.config.world_position_embedding_type)
                        or use_mrope_position_embedding
                    ):
                        position_coord_tensor = position_coord_tensor.round()

                    image_features.append(image_feature)
                    video_position_coords.append(position_coord_tensor)
                    video_compression_outputs.append(compress_output)
                    video_frame_shapes.append(frame_shape)
                    video_metric_profiles.append(compression_profile)
                    video_llm_serialization_metadata.append(None)
                else:
                    image_features.append(image_feat)
                    video_position_coords.append(None)
                    video_compression_outputs.append(None)
                    video_frame_shapes.append(None)
                    video_metric_profiles.append(None)
                    video_llm_serialization_metadata.append(None)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if use_object_proposals:
                object_features = []
                valid_obj_num = 0
                for l in range(obj_num):
                    if "patch27" in self.config.object_feature_type:
                        proposal_source = image_features[0]
                        proposal_mask = object_patch[l].to(device=proposal_source.device).view(-1, 196)
                        cur_object_features = proposal_source[proposal_mask]
                    elif "patch14" in self.config.object_feature_type:
                        proposal_source = encoded_image_features[0]
                        proposal_mask = object_patch[l].to(device=proposal_source.device).view(-1, 729)
                        cur_object_features = proposal_source[proposal_mask]
                    else:
                        raise NotImplementedError

                    if len(cur_object_features) == 0:
                        cur_object_features = torch.zeros(
                            proposal_source.shape[-1],
                            device=proposal_source.device,
                            dtype=proposal_source.dtype,
                        )
                    else:
                        cur_object_features = cur_object_features.mean(dim=0)
                        valid_obj_num += 1
                    object_features.append(cur_object_features)
                object_features = torch.stack(object_features)
                if use_mlp_pe or use_sin3d_pe:
                    box_center_features = self.get_model().world_position_embedding(object_boxes_center.unsqueeze(0)).squeeze(0)      
                    object_features += box_center_features
            else:
                object_features =  None

            
            if use_sin3d_pe or use_mlp_pe:
                new_image_features = []
                for idx, image_feat in enumerate(image_features):
                    coords = video_position_coords[idx]
                    if coords is not None:
                        if coords.shape[:2] != image_feat.shape[:2]:
                            raise ValueError(
                                "World coordinates and image features are misaligned before position embedding: "
                                f"{tuple(coords.shape)} vs {tuple(image_feat.shape)}"
                            )
                        image_feat = image_feat + self.get_model().world_position_embedding(coords.detach())
                    new_image_features.append(image_feat)
                image_features = new_image_features


            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    if image_idx in video_idx_in_batch:  # video operations
                        if mm_newline_position == "grid":
                            # Grid-wise
                            compress_output = video_compression_outputs[image_idx]
                            formatter_outputs = self._format_video_tokens_for_spatial_unpad(
                                image_feature=image_feature,
                                coords=video_position_coords[image_idx] if use_mrope_position_embedding else None,
                                compress_output=compress_output,
                                frame_shape=video_frame_shapes[image_idx],
                                return_serialization_metadata=True,
                            )
                            image_feature, serialized_coords, serialization_metadata = formatter_outputs
                            video_llm_serialization_metadata[image_idx] = serialization_metadata
                            serialized_projector_scores = serialization_metadata.get(
                                "serialized_projector_patch_scores"
                            )
                            serialized_projector_coordinates = serialization_metadata.get(
                                "serialized_projector_patch_coordinates"
                            )
                            serialized_projector_priority = serialization_metadata.get(
                                "serialized_projector_patch_priority"
                            )
                            if serialized_projector_scores is not None:
                                if serialized_projector_coordinates is None:
                                    raise ValueError(
                                        "Serialized projector coordinates are required when projector scores are provided."
                                    )
                                if hasattr(llm_compressor, "set_projector_patch_metadata"):
                                    llm_compressor.set_projector_patch_metadata(
                                        scores=serialized_projector_scores,
                                        coordinates=serialized_projector_coordinates,
                                        priority=serialized_projector_priority,
                                    )
                            if use_mrope_position_embedding:
                                video_position_coords[image_idx] = serialized_coords
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                image_feature = torch.cat(concat_slow_fater_token)
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_world_coords = []
        sample_compression_profiles = []
        sample_llm_compression_metadata = []
        sample_visual_patch_positions_by_view = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                sample_compression_profiles.append({})
                sample_llm_compression_metadata.append({})
                sample_visual_patch_positions_by_view.append([])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cat_cur_input_ids_noim = torch.cat(cur_input_ids_noim)
            cur_input_embeds = self.get_model().embed_tokens(cat_cur_input_ids_noim)

            # Add input coord PE
            if hasattr(self.config, "coord_token_ids") and (use_sin3d_pe or use_mlp_pe):
                query_coord_tokens = (cat_cur_input_ids_noim == self.config.coord_token_ids[0])
                if query_coord_tokens.sum() != 0:
                    coord_token_embedding = self.get_model().world_position_embedding(box_input.unsqueeze(0).detach())[:, 0]
                    coord_token_embedding = coord_token_embedding.to(
                        device=cur_input_embeds.device,
                        dtype=cur_input_embeds.dtype,
                    )
                    cur_input_embeds = cur_input_embeds.clone()
                    cur_input_embeds[query_coord_tokens] = cur_input_embeds[query_coord_tokens] + coord_token_embedding

            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_world_coords = []
            cur_video_profiles = []
            cur_visual_sequence_token_total = 0
            cur_visual_patch_token_total = 0
            cur_pos_index = 0
            cur_prompt_offset = 0
            cur_visual_patch_positions_by_view = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_prompt_offset += len(cur_input_embeds_no_im[i])
                if use_mrope_position_embedding:
                    cur_new_world_coords.append(
                        torch.arange(cur_pos_index, cur_pos_index + len(cur_input_embeds_no_im[i])).to(cur_input_embeds_no_im[i].device).unsqueeze(1).repeat(1, 3)
                    )
                    cur_pos_index += len(cur_input_embeds_no_im[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    
                    profile = video_metric_profiles[cur_image_idx]
                    if profile is not None:
                        cur_video_profiles.append(profile)
                        cur_visual_sequence_token_total += int(cur_image_features.shape[0])

                    serialization_metadata = video_llm_serialization_metadata[cur_image_idx]
                    if serialization_metadata is not None:
                        metadata_patch_count = int(
                            serialization_metadata.get(
                                "visual_patch_token_count",
                                sum(
                                    len(view_patch_positions)
                                    for view_patch_positions in serialization_metadata.get("patch_positions_by_view", [])
                                ),
                            )
                        )
                        cur_visual_patch_token_total += metadata_patch_count
                        for view_patch_positions in serialization_metadata.get("patch_positions_by_view", []):
                            cur_visual_patch_positions_by_view.append(
                                torch.tensor(
                                    [cur_prompt_offset + int(position) for position in view_patch_positions],
                                    device=cur_image_features.device,
                                    dtype=torch.long,
                                )
                            )
                    elif profile is not None:
                        cur_visual_patch_token_total += int(profile.get("compressor_output_tokens", cur_image_features.shape[0]))

                    if use_mrope_position_embedding:
                        coords = video_position_coords[cur_image_idx]
                        flattened_coords = self._flatten_video_position_coords(coords, cur_input_embeds_no_im[i].device)
                        cur_pos_index += flattened_coords.shape[0]
                        cur_new_world_coords.append(flattened_coords)


                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_prompt_offset += cur_image_features.shape[0]

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            if cur_video_profiles:
                compressor_input_tokens = sum(profile["compressor_input_tokens"] for profile in cur_video_profiles)
                compressor_output_tokens = sum(profile["compressor_output_tokens"] for profile in cur_video_profiles)
                visual_patch_tokens = int(cur_visual_patch_token_total)
                visual_sequence_tokens = int(cur_visual_sequence_token_total)
                visual_format_tokens = max(visual_sequence_tokens - visual_patch_tokens, 0)
                sample_compression_profiles.append(
                    {
                        "compressor_name": cur_video_profiles[0]["compressor_name"],
                        "compressor_input_tokens": int(compressor_input_tokens),
                        "compressor_output_tokens": int(compressor_output_tokens),
                        "token_keep_ratio": (
                            float(compressor_output_tokens) / float(compressor_input_tokens)
                            if compressor_input_tokens > 0
                            else 1.0
                        ),
                        "text_prompt_tokens": int(sum(split_sizes)),
                        "prompt_sequence_length": int(cur_new_input_embeds.shape[0]),
                        "visual_patch_tokens": visual_patch_tokens,
                        "visual_sequence_tokens": visual_sequence_tokens,
                        "visual_format_tokens": visual_format_tokens,
                    }
                )
            else:
                sample_compression_profiles.append({})

            sample_visual_patch_positions_by_view.append(
                [positions.detach().cpu() for positions in cur_visual_patch_positions_by_view]
            )

            if configured_llm_compressor and cur_visual_patch_positions_by_view:
                sample_llm_compression_metadata.append(
                    {
                        "visual_patch_positions_by_view": [positions.detach().cpu() for positions in cur_visual_patch_positions_by_view],
                        "text_prompt_tokens": int(sum(split_sizes)),
                        "prompt_sequence_length_before_prune": int(cur_new_input_embeds.shape[0]),
                        "visual_patch_tokens": int(cur_visual_patch_token_total),
                        "visual_sequence_tokens": int(cur_visual_sequence_token_total),
                        "visual_format_tokens": max(int(cur_visual_sequence_token_total) - int(cur_visual_patch_token_total), 0),
                    }
                )
            else:
                sample_llm_compression_metadata.append({})

            if use_mrope_position_embedding:
                cur_new_world_coords = torch.cat(cur_new_world_coords, dim=0)
                new_world_coords.append(cur_new_world_coords)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        for sample_idx, sample_profile in enumerate(sample_compression_profiles):
            if sample_profile:
                prompt_length = int(new_input_embeds[sample_idx].shape[0])
                text_prompt_tokens = int(sample_profile.get("text_prompt_tokens", 0))
                visual_sequence_tokens = max(prompt_length - text_prompt_tokens, 0)
                patch_positions_by_view = (
                    sample_visual_patch_positions_by_view[sample_idx]
                    if sample_idx < len(sample_visual_patch_positions_by_view)
                    else []
                )
                if patch_positions_by_view:
                    visual_patch_tokens = sum(
                        int((view_positions < prompt_length).sum().item())
                        for view_positions in patch_positions_by_view
                    )
                else:
                    visual_patch_tokens = min(
                        int(sample_profile.get("visual_patch_tokens", visual_sequence_tokens)),
                        visual_sequence_tokens,
                    )
                sample_profile["prompt_sequence_length"] = prompt_length
                sample_profile["visual_patch_tokens"] = int(visual_patch_tokens)
                sample_profile["visual_sequence_tokens"] = int(visual_sequence_tokens)
                sample_profile["visual_format_tokens"] = max(int(visual_sequence_tokens) - int(visual_patch_tokens), 0)
        for sample_idx, sample_meta in enumerate(sample_llm_compression_metadata):
            if not sample_meta:
                continue
            prompt_length = int(new_input_embeds[sample_idx].shape[0])
            clipped_by_view = []
            for view_positions in sample_meta["visual_patch_positions_by_view"]:
                clipped = view_positions[view_positions < prompt_length]
                clipped_by_view.append(clipped)
            sample_meta["visual_patch_positions_by_view"] = clipped_by_view
            visual_patch_tokens = sum(int(view_positions.numel()) for view_positions in clipped_by_view)
            text_prompt_tokens = int(sample_meta.get("text_prompt_tokens", 0))
            visual_sequence_tokens = max(prompt_length - text_prompt_tokens, 0)
            sample_meta["visual_patch_tokens"] = int(visual_patch_tokens)
            sample_meta["visual_sequence_tokens"] = int(visual_sequence_tokens)
            sample_meta["visual_format_tokens"] = max(int(visual_sequence_tokens) - int(visual_patch_tokens), 0)
            sample_meta["prompt_sequence_length_before_prune"] = prompt_length
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        mrope_position_ids = torch.zeros((batch_size, max_len, 3), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            llm_sample_meta = sample_llm_compression_metadata[i]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                pad_offset = max_len - cur_len
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if use_mrope_position_embedding:
                        mrope_position_ids[i, -cur_len:, :] = new_world_coords[i][-cur_len:, :]

            else:
                pad_offset = 0
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if use_mrope_position_embedding:
                        mrope_position_ids[i, :cur_len, :] = new_world_coords[i][:cur_len, :]

            if llm_sample_meta:
                shifted_by_view = []
                for view_positions in llm_sample_meta["visual_patch_positions_by_view"]:
                    shifted_by_view.append((view_positions + pad_offset).tolist())
                llm_sample_meta["visual_patch_positions_by_view"] = shifted_by_view
                llm_sample_meta["prompt_sequence_length"] = int(cur_len)

        # mrope_position_ids = mrope_position_ids.permute(2, 0, 1)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        self._last_compression_profile = {"samples": [profile.copy() for profile in sample_compression_profiles if profile]}
        llm_model = self.get_model()
        if hasattr(llm_model, "set_pending_llm_compression_state"):
            if configured_llm_compressor and past_key_values is None:
                llm_model.set_pending_llm_compression_state(
                    {
                        "samples": [sample.copy() for sample in sample_llm_compression_metadata],
                        "padding_side": getattr(self.config, "tokenizer_padding_side", "right"),
                    }
                )
            else:
                llm_model.clear_pending_llm_compression_state()

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        
        if use_mrope_position_embedding:
            position_ids = mrope_position_ids

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, object_features, object_boxes

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
