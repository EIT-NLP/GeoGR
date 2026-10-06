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
import os
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
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

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
        samples = self._last_compression_profile.get("samples", [])
        if len(samples) == 1:
            return samples[0].copy()
        return {"samples": [sample.copy() for sample in samples]}

    def reset_last_compression_profile(self):
        self._last_compression_profile = {"samples": []}

    def merge_last_compression_profile(self, llm_profile):
        """Merge an LLM-stage profile with the preceding projector profile."""
        if not llm_profile:
            return

        current = self.get_last_compression_profile()
        if isinstance(current, dict) and isinstance(current.get("samples"), list):
            projector_profiles = current["samples"]
        elif current:
            projector_profiles = [current]
        else:
            projector_profiles = []

        projector_profile = projector_profiles[0].copy() if projector_profiles else {}
        merged = projector_profile.copy()
        merged.update(llm_profile)

        projector_name = projector_profile.get("projector_compressor_name")
        if projector_name is None and projector_profile:
            projector_name = projector_profile.get("compressor_name")
        llm_name = llm_profile.get("llm_compressor_name", llm_profile.get("compressor_name"))
        if projector_name is not None:
            merged["projector_compressor_name"] = projector_name
        if llm_name is not None:
            merged["llm_compressor_name"] = llm_name

        # Keep stage-specific counts even though generic compressor fields belong
        # to the final LLM stage after the update above.
        for key in (
            "projector_stage_input_tokens",
            "projector_stage_output_tokens",
        ):
            if key in projector_profile:
                merged[key] = projector_profile[key]

        self._last_compression_profile = {"samples": [merged]}

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def _projector_compressor_state(self):
        compressor = getattr(self.get_model(), "mm_projector_compressor", None)
        compressor_type, _ = get_compressor_spec_from_llava_config(self.config, location="projector")
        configured = compressor is not None and compressor_type not in ("none", "identity", None, "")
        return configured, compressor, compressor_type

    def _record_video_token_stats(self, original_tokens: int, output_tokens: int):
        if not hasattr(self, "_compression_token_stats"):
            self._init_compression_token_stats()
        self._compression_token_stats["total_original_tokens"] += int(original_tokens)
        self._compression_token_stats["total_output_tokens"] += int(output_tokens)
        self._compression_token_stats["video_count"] += 1

    def _build_video_compression_profile(
        self,
        compressor_name: str,
        input_tokens: int,
        output_tokens: int,
        sequence_tokens: int = None,
    ):
        sequence_tokens = int(output_tokens if sequence_tokens is None else sequence_tokens)
        return {
            "compressor_name": compressor_name,
            "compressor_input_tokens": int(input_tokens),
            "compressor_output_tokens": int(output_tokens),
            "projector_stage_input_tokens": int(input_tokens),
            "projector_stage_output_tokens": int(output_tokens),
            "token_keep_ratio": (float(output_tokens) / float(input_tokens)) if input_tokens > 0 else 1.0,
            "visual_patch_tokens": int(output_tokens),
            "visual_sequence_tokens": sequence_tokens,
            "visual_format_tokens": max(sequence_tokens - int(output_tokens), 0),
            "prefill_final_visual_tokens": float(output_tokens),
        }

    def _stack_aux_frames(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            if all(torch.is_tensor(item) for item in value):
                return torch.stack(list(value), dim=0)
        return value

    def _select_aux_for_sample(self, value, sample_idx: int, num_frames: int):
        value = self._stack_aux_frames(value)
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) == num_frames and all(torch.is_tensor(item) for item in value):
                return torch.stack(list(value), dim=0)
            if sample_idx < len(value):
                return self._stack_aux_frames(value[sample_idx])
            if len(value) == 1:
                return self._stack_aux_frames(value[0])
            return None
        if not torch.is_tensor(value):
            return None
        if value.dim() == 4 and tuple(value.shape[-2:]) in {(3, 3), (3, 4), (4, 4)}:
            if value.shape[0] > sample_idx and value.shape[1] in {1, num_frames}:
                return value[sample_idx]
            if value.shape[0] == 1 and value.shape[1] in {1, num_frames}:
                return value[0]
        if value.dim() >= 5 and value.shape[0] > sample_idx:
            return value[sample_idx]
        if value.dim() == 4:
            if value.shape[-1] == 3 and value.shape[0] == num_frames:
                return value
            if value.shape[0] > sample_idx:
                return value[sample_idx]
        if value.dim() >= 3 and value.shape[0] == num_frames:
            return value
        if value.dim() >= 4 and value.shape[0] == 1:
            return value[0]
        return value

    def _infer_aux_batch_length(self, value, num_frames: int):
        value = self._stack_aux_frames(value)
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return len(value)
        if not torch.is_tensor(value):
            return None
        if value.dim() >= 5:
            return int(value.shape[0])
        if value.dim() == 4 and tuple(value.shape[-2:]) in {(3, 3), (3, 4), (4, 4)}:
            return int(value.shape[0])
        if value.dim() == 4 and not (value.shape[-1] == 3 and value.shape[0] == num_frames):
            return int(value.shape[0])
        return None

    def _select_world_coords_for_sample(self, video_dict, sample_idx: int, num_frames: int):
        if video_dict is None or "world_coords" not in video_dict:
            return None
        return self._select_aux_for_sample(video_dict["world_coords"], sample_idx, num_frames)

    def _select_aux_for_multimodal_sample(
        self,
        value,
        sample_idx: int,
        video_sample_idx: int,
        num_frames: int,
        batch_size: int,
        num_videos: int,
    ):
        top_level_len = self._infer_aux_batch_length(value, num_frames)
        if top_level_len == batch_size and sample_idx < batch_size:
            return self._select_aux_for_sample(value, sample_idx, num_frames)
        if top_level_len == num_videos and video_sample_idx < num_videos:
            return self._select_aux_for_sample(value, video_sample_idx, num_frames)

        selected = self._select_aux_for_sample(value, sample_idx, num_frames)
        if selected is None and video_sample_idx != sample_idx:
            selected = self._select_aux_for_sample(value, video_sample_idx, num_frames)
        return selected

    def _select_world_coords_for_multimodal_sample(
        self,
        video_dict,
        sample_idx: int,
        video_sample_idx: int,
        num_frames: int,
        batch_size: int,
        num_videos: int,
    ):
        if video_dict is None or "world_coords" not in video_dict:
            return None
        return self._select_aux_for_multimodal_sample(
            video_dict["world_coords"],
            sample_idx=sample_idx,
            video_sample_idx=video_sample_idx,
            num_frames=num_frames,
            batch_size=batch_size,
            num_videos=num_videos,
        )

    def _expand_per_frame_matrix(self, matrix: torch.Tensor, num_frames: int, name: str):
        if matrix is None:
            raise ValueError(f"{name} is required to compute voxel coordinates from depth.")
        if not torch.is_tensor(matrix):
            matrix = torch.as_tensor(matrix)
        if matrix.dim() >= 2 and matrix.shape[-2:] == (3, 4):
            bottom = torch.zeros(*matrix.shape[:-2], 1, 4, dtype=matrix.dtype, device=matrix.device)
            bottom[..., 0, 3] = 1
            matrix = torch.cat((matrix, bottom), dim=-2)
        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0).repeat(num_frames, 1, 1)
        elif matrix.dim() == 3 and matrix.shape[0] == 1:
            matrix = matrix.repeat(num_frames, 1, 1)
        elif matrix.dim() != 3 or matrix.shape[0] != num_frames:
            raise ValueError(f"{name} must be shaped as (frames, M, M) or (M, M), got {tuple(matrix.shape)}.")
        return matrix

    def _compute_world_coords_from_depth(self, depths, cam2world, intrinsic, device):
        if depths is None:
            return None
        depths = self._stack_aux_frames(depths)
        if not torch.is_tensor(depths):
            return None
        if depths.dim() == 4 and depths.shape[0] == 1:
            depths = depths[0]
        if depths.dim() != 3:
            raise ValueError(f"depths must be shaped as (frames, H, W), got {tuple(depths.shape)}.")

        depths = depths.to(device=device, dtype=torch.float32)
        num_frames, height, width = depths.shape
        if depths.numel() > 0 and float(depths.max().item()) > 100.0:
            depths = depths / float(getattr(self.config, "mm_depth_scale", 1000.0))

        cam2world = self._expand_per_frame_matrix(cam2world, num_frames, "cam2world").to(device=device, dtype=torch.float32)
        intrinsic = self._expand_per_frame_matrix(intrinsic, num_frames, "intrinsic").to(device=device, dtype=torch.float32)

        rows = torch.arange(height, device=device, dtype=torch.float32)
        cols = torch.arange(width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        xx = xx.reshape(1, -1).repeat(num_frames, 1)
        yy = yy.reshape(1, -1).repeat(num_frames, 1)
        z = depths.reshape(num_frames, -1)

        fx = intrinsic[:, 0, 0].unsqueeze(-1)
        fy = intrinsic[:, 1, 1].unsqueeze(-1)
        cx = intrinsic[:, 0, 2].unsqueeze(-1)
        cy = intrinsic[:, 1, 2].unsqueeze(-1)

        x = (xx - cx) * z / fx.clamp_min(1e-6)
        y = (yy - cy) * z / fy.clamp_min(1e-6)
        ones = torch.ones_like(z)
        cam_xyz = torch.stack((x, y, z, ones), dim=-1)
        world = torch.bmm(cam2world, cam_xyz.transpose(1, 2)).transpose(1, 2)
        world = world[..., :3] / world[..., 3:].clamp_min(1e-6)
        return world.reshape(num_frames, height, width, 3)

    def _pool_world_coords_to_frame_shape(self, world_coords, num_frames: int, frame_shape, device):
        if world_coords is None:
            return None
        if not torch.is_tensor(world_coords):
            world_coords = torch.as_tensor(world_coords)
        if world_coords.dim() == 5 and world_coords.shape[0] == 1:
            world_coords = world_coords[0]
        if world_coords.dim() == 3 and world_coords.shape[0] == num_frames and world_coords.shape[-1] == 3:
            tokens_per_frame = frame_shape[0] * frame_shape[1]
            if world_coords.shape[1] != tokens_per_frame:
                raise ValueError(
                    "Pooled world_coords token count must match frame_shape, "
                    f"got {world_coords.shape[1]} vs {tokens_per_frame}."
                )
            return torch.nan_to_num(world_coords.to(device=device, dtype=torch.float32))
        if world_coords.dim() != 4 or world_coords.shape[0] != num_frames or world_coords.shape[-1] != 3:
            raise ValueError(
                "world_coords must be shaped as (frames, H, W, 3) or (frames, pooled_tokens, 3), "
                f"got {tuple(world_coords.shape)} for num_frames={num_frames}."
            )
        height, width = frame_shape
        if world_coords.shape[1] == height and world_coords.shape[2] == width:
            return torch.nan_to_num(world_coords.to(device=device, dtype=torch.float32)).view(num_frames, height * width, 3)
        coords = torch.nan_to_num(world_coords.to(device=device, dtype=torch.float32))
        coords = coords.permute(0, 3, 1, 2).contiguous()
        coords = nn.functional.adaptive_avg_pool2d(coords, output_size=(height, width))
        coords = coords.permute(0, 2, 3, 1).contiguous().view(num_frames, height * width, 3)
        return coords

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

    def _split_compressed_video_segments(self, compress_output, frame_shape):
        features = compress_output.features
        if features.dim() == 3 and features.shape[0] == 1:
            flat_features = features[0]
        elif features.dim() == 3:
            flat_features = features.flatten(0, 1)
        elif features.dim() == 2:
            flat_features = features
        else:
            raise ValueError(f"Unexpected compressed video feature shape: {tuple(features.shape)}.")

        metadata = compress_output.metadata
        patch_positions = metadata.get("compressed_patch_positions")
        if patch_positions is None:
            height, width = frame_shape
            token_count = flat_features.shape[0]
            row_ids = torch.arange(token_count, device=flat_features.device, dtype=torch.float32) // width
            col_ids = torch.arange(token_count, device=flat_features.device, dtype=torch.float32) % width
            patch_positions = torch.stack((row_ids, col_ids), dim=-1).unsqueeze(0)
        if patch_positions.dim() == 3 and patch_positions.shape[0] == 1:
            flat_patch_positions = patch_positions[0].to(device=flat_features.device)
        elif patch_positions.dim() == 3:
            flat_patch_positions = patch_positions.flatten(0, 1).to(device=flat_features.device)
        elif patch_positions.dim() == 2:
            flat_patch_positions = patch_positions.to(device=flat_features.device)
        else:
            raise ValueError(f"Unexpected compressed patch position shape: {tuple(patch_positions.shape)}.")

        if flat_patch_positions.shape[0] != flat_features.shape[0]:
            raise ValueError(
                "Compressed patch positions must align with compressed features: "
                f"{tuple(flat_patch_positions.shape)} vs {tuple(flat_features.shape)}."
            )

        frame_token_counts = metadata.get("frame_token_counts")
        if frame_token_counts is None:
            return [flat_features], [flat_patch_positions]
        if not isinstance(frame_token_counts, (list, tuple)):
            raise ValueError(f"frame_token_counts must be a list or tuple, got {type(frame_token_counts).__name__}.")

        feature_segments = []
        patch_position_segments = []
        offset = 0
        for count in frame_token_counts:
            count = int(count)
            next_offset = offset + count
            feature_segments.append(flat_features[offset:next_offset])
            patch_position_segments.append(flat_patch_positions[offset:next_offset])
            offset = next_offset
        if offset != flat_features.shape[0]:
            raise ValueError(
                "frame_token_counts do not cover compressed features: "
                f"consumed {offset}, total {flat_features.shape[0]}."
            )
        return feature_segments, patch_position_segments

    def _serialize_vtc_visionzip_grid_drop(
        self,
        compress_output,
        frame_shape,
        feature_segments,
        patch_position_segments,
        projector_score_segments=None,
        projector_coordinate_segments=None,
        projector_priority_segments=None,
    ):
        nonempty_segment_indices = [
            segment_idx
            for segment_idx, feature_segment in enumerate(feature_segments)
            if feature_segment.shape[0] > 0
        ]
        if not nonempty_segment_indices:
            raise ValueError("Cannot serialize an empty compressed video sequence.")

        features = compress_output.features
        if features.dim() == 3 and features.shape[0] == 1:
            flat_features = features[0]
        elif features.dim() == 3:
            flat_features = features.flatten(0, 1)
        else:
            flat_features = features
        flat_patch_positions = torch.cat(
            [patch_position_segments[segment_idx] for segment_idx in nonempty_segment_indices],
            dim=0,
        )
        segment_counts = torch.tensor(
            [feature_segment.shape[0] for feature_segment in feature_segments],
            device=flat_features.device,
            dtype=torch.long,
        )
        frame_ids = torch.repeat_interleave(
            torch.arange(len(feature_segments), device=flat_features.device),
            segment_counts,
            output_size=flat_features.shape[0],
        )

        height, width = frame_shape
        row_ids = flat_patch_positions[:, 0].round().to(dtype=torch.long).clamp_(0, height - 1)
        col_ids = flat_patch_positions[:, 1].round().to(dtype=torch.long).clamp_(0, width - 1)
        spatial_keys = (frame_ids * height + row_ids) * width + col_ids
        order = spatial_keys.argsort(stable=True)
        ordered_features = flat_features.index_select(0, order)
        ordered_frames = frame_ids.index_select(0, order)
        ordered_rows = row_ids.index_select(0, order)

        row_ends = torch.ones(ordered_features.shape[0], device=ordered_features.device, dtype=torch.bool)
        if ordered_features.shape[0] > 1:
            row_ends[:-1] = (ordered_frames[1:] != ordered_frames[:-1]) | (
                ordered_rows[1:] != ordered_rows[:-1]
            )

        # One device synchronization replaces the old per-row .tolist() synchronization.
        newline_count = int(row_ends.sum().item())
        row_end_prefix = row_ends.to(dtype=torch.long).cumsum(dim=0)
        patch_output_positions = (
            torch.arange(ordered_features.shape[0], device=ordered_features.device)
            + row_end_prefix
            - row_ends.to(dtype=torch.long)
        )

        newline_token = self.model.image_newline.to(
            device=ordered_features.device,
            dtype=ordered_features.dtype,
        ).unsqueeze(0)
        source_tokens = torch.cat((ordered_features, newline_token), dim=0)
        source_indices = torch.full(
            (ordered_features.shape[0] + newline_count,),
            ordered_features.shape[0],
            device=ordered_features.device,
            dtype=torch.long,
        )
        source_indices.scatter_(
            0,
            patch_output_positions,
            torch.arange(ordered_features.shape[0], device=ordered_features.device),
        )

        def serialize_projector_metadata(segments, metadata_key):
            if segments is None:
                return
            flat_values = torch.cat(
                [
                    segments[segment_idx].to(device=order.device)
                    for segment_idx in nonempty_segment_indices
                ],
                dim=0,
            )
            compress_output.metadata[metadata_key] = flat_values.index_select(0, order).detach()

        serialize_projector_metadata(projector_score_segments, "serialized_projector_patch_scores")
        serialize_projector_metadata(
            projector_coordinate_segments,
            "serialized_projector_patch_coordinates",
        )
        serialize_projector_metadata(projector_priority_segments, "serialized_projector_patch_priority")
        return source_tokens.index_select(0, source_indices)

    def _serialize_compressed_video_features(self, compress_output, frame_shape):
        feature_segments, patch_position_segments = self._split_compressed_video_segments(compress_output, frame_shape)
        projector_scores = compress_output.metadata.get("projector_patch_scores")
        projector_coordinates = compress_output.metadata.get(
            "projector_patch_coordinates",
            compress_output.metadata.get("compressed_coordinates"),
        )
        projector_priority = compress_output.metadata.get("projector_patch_priority")
        projector_score_segments = None
        projector_coordinate_segments = None
        projector_priority_segments = None
        if projector_scores is not None:
            projector_scores = torch.as_tensor(projector_scores).flatten()
            expected_scores = sum(int(segment.shape[0]) for segment in feature_segments)
            if int(projector_scores.numel()) != expected_scores:
                raise ValueError(
                    "Projector patch scores do not align with compressed features: "
                    f"scores={int(projector_scores.numel())}, features={expected_scores}."
                )
            projector_score_segments = []
            score_offset = 0
            for feature_segment in feature_segments:
                next_offset = score_offset + int(feature_segment.shape[0])
                projector_score_segments.append(projector_scores[score_offset:next_offset])
                score_offset = next_offset
        if projector_priority is not None:
            projector_priority = torch.as_tensor(projector_priority).flatten()
            expected_priority = sum(int(segment.shape[0]) for segment in feature_segments)
            if int(projector_priority.numel()) != expected_priority:
                raise ValueError(
                    "Projector patch priorities do not align with compressed features: "
                    f"priority={int(projector_priority.numel())}, features={expected_priority}."
                )
            projector_priority_segments = []
            priority_offset = 0
            for feature_segment in feature_segments:
                next_offset = priority_offset + int(feature_segment.shape[0])
                projector_priority_segments.append(projector_priority[priority_offset:next_offset])
                priority_offset = next_offset
        if projector_coordinates is not None:
            projector_coordinates = torch.as_tensor(projector_coordinates)
            if projector_coordinates.dim() == 3 and projector_coordinates.shape[0] == 1:
                projector_coordinates = projector_coordinates[0]
            elif projector_coordinates.dim() == 3:
                projector_coordinates = projector_coordinates.flatten(0, 1)
            if projector_coordinates.dim() != 2 or projector_coordinates.shape[1] != 3:
                raise ValueError(
                    "Compressed coordinates must have shape [N, 3], got "
                    f"{tuple(projector_coordinates.shape)}."
                )
            expected_coordinates = sum(int(segment.shape[0]) for segment in feature_segments)
            if int(projector_coordinates.shape[0]) != expected_coordinates:
                raise ValueError(
                    "Compressed coordinates do not align with compressed features: "
                    f"coordinates={int(projector_coordinates.shape[0])}, features={expected_coordinates}."
                )
            projector_coordinate_segments = []
            coordinate_offset = 0
            for feature_segment in feature_segments:
                next_offset = coordinate_offset + int(feature_segment.shape[0])
                projector_coordinate_segments.append(
                    projector_coordinates[coordinate_offset:next_offset]
                )
                coordinate_offset = next_offset
        metadata_newline_strategy = compress_output.metadata.get("newline_strategy")
        config_newline_position = getattr(self.config, "mm_newline_position", "one_token")
        if metadata_newline_strategy in {"grid_drop", "full_grid_drop", "frame_newline", "one_token", "no_token"}:
            newline_strategy = metadata_newline_strategy
        elif config_newline_position == "no_token":
            newline_strategy = "no_token"
        elif config_newline_position == "grid":
            newline_strategy = "grid_drop"
        elif config_newline_position == "frame":
            newline_strategy = "frame_newline"
        elif config_newline_position == "one_token":
            newline_strategy = "one_token"
        else:
            raise ValueError(f"Unexpected mm_newline_position: {config_newline_position}")

        preserve_segment_order = bool(compress_output.metadata.get("preserve_segment_order", False))
        if (
            compress_output.metadata.get("voxel_method") == "vtc_visionzip"
            and newline_strategy == "grid_drop"
            and not preserve_segment_order
        ):
            return self._serialize_vtc_visionzip_grid_drop(
                compress_output=compress_output,
                frame_shape=frame_shape,
                feature_segments=feature_segments,
                patch_position_segments=patch_position_segments,
                projector_score_segments=projector_score_segments,
                projector_coordinate_segments=projector_coordinate_segments,
                projector_priority_segments=projector_priority_segments,
            )

        height, width = frame_shape
        serialized_features = []
        serialized_projector_scores = []
        serialized_projector_coordinates = []
        serialized_projector_priority = []
        newline_token = self.model.image_newline[None]
        for segment_idx, (feature_segment, patch_position_segment) in enumerate(
            zip(feature_segments, patch_position_segments)
        ):
            if feature_segment.shape[0] == 0:
                continue
            row_ids = patch_position_segment[:, 0].round().to(dtype=torch.long).clamp_(0, height - 1)
            col_ids = patch_position_segment[:, 1].round().to(dtype=torch.long).clamp_(0, width - 1)
            if preserve_segment_order:
                order = torch.arange(feature_segment.shape[0], device=feature_segment.device)
            else:
                order = (row_ids * width + col_ids).argsort(stable=True)
            ordered_features = feature_segment[order]
            ordered_rows = row_ids[order]
            if projector_score_segments is not None:
                serialized_projector_scores.append(
                    projector_score_segments[segment_idx].to(device=order.device).index_select(0, order)
                )
            if projector_coordinate_segments is not None:
                serialized_projector_coordinates.append(
                    projector_coordinate_segments[segment_idx].to(device=order.device).index_select(0, order)
                )
            if projector_priority_segments is not None:
                serialized_projector_priority.append(
                    projector_priority_segments[segment_idx].to(device=order.device).index_select(0, order)
                )

            if newline_strategy in {"no_token", "one_token", "frame_newline"}:
                serialized_features.append(ordered_features)
                if newline_strategy == "frame_newline":
                    serialized_features.append(newline_token.to(device=ordered_features.device, dtype=ordered_features.dtype))
                continue

            if newline_strategy == "full_grid_drop":
                for row_idx in range(height):
                    row_features = ordered_features[ordered_rows == row_idx]
                    if row_features.shape[0] > 0:
                        serialized_features.append(row_features)
                        newline_device = row_features.device
                        newline_dtype = row_features.dtype
                    else:
                        newline_device = ordered_features.device
                        newline_dtype = ordered_features.dtype
                    serialized_features.append(newline_token.to(device=newline_device, dtype=newline_dtype))
                continue

            if newline_strategy == "grid_drop":
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
                    serialized_features.append(newline_token.to(device=row_features.device, dtype=row_features.dtype))
                continue

            raise ValueError(f"Unexpected compressed video newline_strategy: {newline_strategy}")

        if newline_strategy == "one_token":
            if serialized_features:
                newline_device = serialized_features[-1].device
                newline_dtype = serialized_features[-1].dtype
            else:
                template = compress_output.features
                newline_device = template.device
                newline_dtype = template.dtype
            serialized_features.append(newline_token.to(device=newline_device, dtype=newline_dtype))

        if not serialized_features:
            raise ValueError("Cannot serialize an empty compressed video sequence.")
        if projector_score_segments is not None:
            compress_output.metadata["serialized_projector_patch_scores"] = torch.cat(
                serialized_projector_scores, dim=0
            ).detach()
        if projector_coordinate_segments is not None:
            compress_output.metadata["serialized_projector_patch_coordinates"] = torch.cat(
                serialized_projector_coordinates, dim=0
            ).detach()
        if projector_priority_segments is not None:
            compress_output.metadata["serialized_projector_patch_priority"] = torch.cat(
                serialized_projector_priority, dim=0
            ).detach()
        return torch.cat(serialized_features, dim=0)

    def _compress_projector_video_tokens(
        self,
        image_feat: torch.Tensor,
        compressor,
        compressor_name: str,
        world_coords=None,
        raw_features_before_proj=None,
        attn_weights=None,
    ):
        pooled_feat = self.get_2dPool(image_feat)
        pooled_tokens = pooled_feat.shape[1]
        side = math.isqrt(pooled_tokens)
        if side * side != pooled_tokens:
            raise ValueError(f"Expected square pooled video features, got {pooled_tokens} tokens.")
        frame_shape = (side, side)
        input_tokens = int(pooled_feat.shape[0] * pooled_feat.shape[1])

        coords = self._pool_world_coords_to_frame_shape(
            world_coords,
            num_frames=pooled_feat.shape[0],
            frame_shape=frame_shape,
            device=pooled_feat.device,
        )
        pooled_raw_features = None if raw_features_before_proj is None else self.get_2dPool(raw_features_before_proj)
        pooled_attn_weights = None if attn_weights is None else self.get_2dPool_scores(attn_weights)

        required_inputs = compressor.get_required_inputs()
        if required_inputs.get("coordinates", False) and coords is None:
            raise ValueError(f"Configured projector compressor '{compressor_name}' requires patch-level 3D coordinates.")
        if required_inputs.get("raw_features_before_proj", False) and pooled_raw_features is None:
            raise ValueError(f"Configured projector compressor '{compressor_name}' requires raw_features_before_proj.")
        if required_inputs.get("attn_weights", False) and pooled_attn_weights is None:
            raise ValueError(f"Configured projector compressor '{compressor_name}' requires visual attentions.")

        measure_projector_compression = str(
            os.environ.get("OV_MEASURE_PROJECTOR_COMPRESSION_TIME", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        measure_device = pooled_feat.device if pooled_feat.is_cuda else None
        if measure_projector_compression and measure_device is not None:
            torch.cuda.synchronize(measure_device)
        compression_start = time.perf_counter()
        compress_output = compressor.compress(
            features=pooled_feat,
            num_frames=pooled_feat.shape[0],
            frame_shape=frame_shape,
            coordinates=coords,
            grouping_coordinates=coords,
            raw_features_before_proj=pooled_raw_features,
            attn_weights=pooled_attn_weights,
        )
        if measure_projector_compression and measure_device is not None:
            torch.cuda.synchronize(measure_device)
        projector_compression_time_ms = (
            (time.perf_counter() - compression_start) * 1000.0
            if measure_projector_compression
            else None
        )

        compressed_patch_tokens = int(compress_output.features.shape[0] * compress_output.features.shape[1])
        compressed = self._serialize_compressed_video_features(compress_output, frame_shape)
        serialized_projector_scores = compress_output.metadata.get("serialized_projector_patch_scores")
        serialized_projector_coordinates = compress_output.metadata.get(
            "serialized_projector_patch_coordinates"
        )
        serialized_projector_priority = compress_output.metadata.get(
            "serialized_projector_patch_priority"
        )
        llm_compressor = getattr(self.get_model(), "mm_llm_compressor", None)
        if hasattr(llm_compressor, "set_projector_patch_metadata"):
            if serialized_projector_scores is not None and serialized_projector_coordinates is None:
                raise ValueError(
                    "Serialized projector coordinates are required when projector scores are provided."
                )
            if serialized_projector_scores is not None:
                llm_compressor.set_projector_patch_metadata(
                    scores=serialized_projector_scores,
                    coordinates=serialized_projector_coordinates,
                    priority=serialized_projector_priority,
                )
        self._record_video_token_stats(input_tokens, compressed_patch_tokens)
        compression_profile = self._build_video_compression_profile(
            compressor_name=compressor_name,
            input_tokens=input_tokens,
            output_tokens=compressed_patch_tokens,
            sequence_tokens=int(compressed.shape[0]),
        )
        # Preserve scalar projector diagnostics in the per-sample profile.
        # They are needed by component ablations to verify the selected budget
        # and method-specific routing. Tensor-valued coordinates/scores remain
        # out of the JSON profile.
        diagnostic_keys = (
            "selection_method",
            "visual_token_num",
            "per_frame_visual_token_num",
            "target_tokens",
            "total_target_tokens",
            "important_ratio",
            "lam",
            "important_tokens",
            "diverse_tokens",
            "budget_scope",
            "fps_scope",
            "dx_scope",
            "voxel_method",
            "voxel_size",
            "attention_reduce",
            "coverage_rule",
            "random_seed",
            "num_voxels_before_post",
            "input_tokens_before_post",
            "nominal_target_tokens",
            "budget_limited_by_voxel_count",
            "dominant_ratio",
            "requested_dominant_tokens",
            "requested_contextual_tokens",
            "dominant_tokens",
            "contextual_tokens",
            "residual_merge",
            "residual_merge_applied",
            "num_residual_merged",
        )
        compression_stats = compressor.get_compression_info()
        for key in diagnostic_keys:
            value = compression_stats.get(key)
            if isinstance(value, (int, float, bool, str)):
                compression_profile[key] = value
        if projector_compression_time_ms is not None:
            compression_profile["projector_compression_time_ms"] = float(projector_compression_time_ms)
        return compressed, compress_output, compression_profile

    def encode_images(self, images, output_attentions: bool = False, return_raw_features: bool = False):
        vision_tower = self.get_model().get_vision_tower()
        if output_attentions:
            try:
                attention_kwargs = {"output_attentions": True}
                if getattr(vision_tower, "supports_last_layer_attention_only", False):
                    attention_kwargs["output_attentions_last_only"] = True
                if getattr(vision_tower, "supports_reduced_attention", False):
                    # Voxel-VTC consumes mean-over-query attention scores.
                    # Returning the reduced score directly avoids carrying the
                    # full final query-by-key matrix through the multimodal
                    # wrapper without changing the score definition.
                    attention_kwargs["output_attentions_reduced"] = True
                vision_outputs = vision_tower(images, **attention_kwargs)
            except TypeError as exc:
                raise NotImplementedError(
                    f"Vision tower '{type(vision_tower).__name__}' does not support output_attentions=True."
                ) from exc
        else:
            vision_outputs = vision_tower(images)

        if isinstance(vision_outputs, tuple):
            raw_image_features, image_attentions = vision_outputs[:2]
        else:
            raw_image_features = vision_outputs
            image_attentions = None
            if output_attentions:
                raise NotImplementedError(
                    f"Vision tower '{type(vision_tower).__name__}' did not return attentions when requested."
                )

        # image_features = self.get_model().vision_resampler(image_features, images=images)
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
            if not isinstance(slower_img_feat, int):
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
        depths=None,
        poses=None,
        cam2world=None,
        intrinsic=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]
        self.reset_last_compression_profile()

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
            configured_projector_compressor, projector_compressor, projector_compressor_type = self._projector_compressor_state()
            has_video = len(video_idx_in_batch) > 0
            projector_required_inputs = (
                projector_compressor.get_required_inputs()
                if configured_projector_compressor and has_video
                else {}
            )
            if configured_projector_compressor and has_video:
                if not projector_compressor.supports_video():
                    raise ValueError(f"Configured projector compressor '{projector_compressor_type}' does not support video inputs.")
                if getattr(self.config, "add_faster_video", False):
                    raise NotImplementedError("Projector compression is not supported together with add_faster_video.")

            encoder_outputs = self.encode_images(
                concat_images,
                output_attentions=bool(projector_required_inputs.get("attn_weights", False)),
                return_raw_features=bool(projector_required_inputs.get("raw_features_before_proj", False)),
            )
            if isinstance(encoder_outputs, tuple):
                encoded_image_features, encoder_aux = encoder_outputs
            else:
                encoded_image_features = encoder_outputs
                encoder_aux = {}
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            raw_features_before_proj = None
            if "raw_features_before_proj" in encoder_aux:
                raw_features_before_proj = torch.split(encoder_aux["raw_features_before_proj"], split_sizes)
            attention_scores = None
            if "attn_weights" in encoder_aux:
                attention_scores = torch.split(encoder_aux["attn_weights"], split_sizes)

            image_features = []
            compressed_video_indices = set()
            compression_profiles = []
            video_sample_idx = 0
            num_videos = len(video_idx_in_batch)
            batch_size = len(encoded_image_features)
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    if configured_projector_compressor:
                        current_world_coords = self._select_world_coords_for_multimodal_sample(
                            video_dict,
                            sample_idx=idx,
                            video_sample_idx=video_sample_idx,
                            num_frames=image_feat.shape[0],
                            batch_size=batch_size,
                            num_videos=num_videos,
                        )
                        if current_world_coords is None and projector_required_inputs.get("coordinates", False):
                            current_depths = self._select_aux_for_multimodal_sample(
                                depths,
                                sample_idx=idx,
                                video_sample_idx=video_sample_idx,
                                num_frames=image_feat.shape[0],
                                batch_size=batch_size,
                                num_videos=num_videos,
                            )
                            current_cam2world = self._select_aux_for_multimodal_sample(
                                cam2world if cam2world is not None else poses,
                                sample_idx=idx,
                                video_sample_idx=video_sample_idx,
                                num_frames=image_feat.shape[0],
                                batch_size=batch_size,
                                num_videos=num_videos,
                            )
                            current_intrinsic = self._select_aux_for_multimodal_sample(
                                intrinsic,
                                sample_idx=idx,
                                video_sample_idx=video_sample_idx,
                                num_frames=image_feat.shape[0],
                                batch_size=batch_size,
                                num_videos=num_videos,
                            )
                            current_world_coords = self._compute_world_coords_from_depth(
                                current_depths,
                                current_cam2world,
                                current_intrinsic,
                                device=image_feat.device,
                            )
                        compressed_feature, _, compression_profile = self._compress_projector_video_tokens(
                            image_feat=image_feat,
                            compressor=projector_compressor,
                            compressor_name=projector_compressor_type,
                            world_coords=current_world_coords,
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
                        image_features.append(compressed_feature)
                        compressed_video_indices.add(idx)
                        compression_profiles.append(compression_profile)
                    else:
                        image_features.append(self.get_2dPool(image_feat))
                    video_sample_idx += 1
                else:
                    image_features.append(image_feat)
            if compression_profiles:
                self._last_compression_profile = {"samples": [profile.copy() for profile in compression_profiles]}
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) if x.dim() > 2 else x for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    if image_idx in compressed_video_indices:
                        new_image_features.append(image_feature)
                    elif image_idx in video_idx_in_batch:  # video operations
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
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
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
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
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
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

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

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
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

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
