from __future__ import annotations

import copy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.models.model_utils.compression_efficiency import estimate_generation_flops
from lmms_eval.models.model_utils.gen_metrics import LatencyStreamer
from lmms_eval.models.simple.llava_onevision import Llava_OneVision
from lmms_eval.tasks._task_utils.scannet3d.coord_cache import coords_cache_path
from lmms_eval.tasks._task_utils.scannet3d.config import get_data_paths
from lmms_eval.tasks._task_utils.scannet3d.data import get_asset_manager

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.model.multimodal_compressor import build_compressor_from_llava_config


eval_logger = logging.getLogger("lmms-eval")


DEFAULT_OV_3D_EXTRA_PROMPT = "The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions."
ALLOWED_OV_PROJECTOR_COMPRESSORS = {
    "segpruner",
    "vispruner",
    "visionzip",
    "voxel_dtc",
    "voxel_vtc",
    "voxel_vtc_visionzip",
}
ALLOWED_OV_LLM_COMPRESSORS = {
    "group_wise_skip_recovery",
    "late_entry_early_exit",
}


def _as_bool(value: Union[bool, str]) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _log_compression_profile_enabled() -> bool:
    return str(os.environ.get("OV_LOG_COMPRESSION_PROFILE", "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _runtime_profile_enabled() -> bool:
    return str(os.environ.get("OV_MEASURE_RUNTIME_PROFILE", "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_json_object(raw: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if raw in (None, ""):
        return None
    if isinstance(raw, dict):
        return raw
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("mm_projector_compressor_config must decode to a JSON object.")
    return parsed


def _resolve_prefill_final_visual_tokens(profile: Dict[str, Any]) -> Optional[float]:
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

    visual_patch_tokens = profile.get("visual_patch_tokens")
    if isinstance(visual_patch_tokens, (int, float)):
        return float(visual_patch_tokens)

    projector_output_tokens = profile.get("projector_stage_output_tokens")
    if isinstance(projector_output_tokens, (int, float)):
        return float(projector_output_tokens)

    compressor_output_tokens = profile.get("compressor_output_tokens")
    if isinstance(compressor_output_tokens, (int, float)):
        return float(compressor_output_tokens)

    fallback_value = profile.get("prefill_final_visual_tokens")
    if isinstance(fallback_value, (int, float)):
        return float(fallback_value)

    final_sequence_length = profile.get("final_sequence_length")
    if (
        isinstance(final_sequence_length, (int, float))
        and isinstance(text_prompt_tokens, (int, float))
        and isinstance(visual_format_tokens, (int, float))
    ):
        return float(max(int(final_sequence_length) - int(text_prompt_tokens) - int(visual_format_tokens), 0))

    prompt_sequence_length = profile.get("prompt_sequence_length")
    if (
        isinstance(prompt_sequence_length, (int, float))
        and isinstance(text_prompt_tokens, (int, float))
        and isinstance(visual_format_tokens, (int, float))
    ):
        return float(max(int(prompt_sequence_length) - int(text_prompt_tokens) - int(visual_format_tokens), 0))
    return None


def _build_public_compression_efficiency(
    compression_profile: Optional[Dict[str, Any]] = None,
    *,
    latency_metrics: Optional[Dict[str, Any]] = None,
    flops_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = compression_profile or {}
    metrics: Dict[str, Any] = {}

    projector_input_tokens = profile.get("projector_stage_input_tokens")
    projector_output_tokens = profile.get("projector_stage_output_tokens")

    # LLM-only compressors run after an identity projector and therefore do
    # not naturally emit a Stage-I profile.  Fill the identity fields from
    # the actual serialized patch count when they are absent.  Do not
    # overwrite a real projector-compression profile supplied by Stage I.
    identity_patch_tokens = profile.get("visual_patch_tokens")
    if not isinstance(identity_patch_tokens, (int, float)):
        sequence_tokens = profile.get("visual_sequence_tokens")
        format_tokens = profile.get("visual_format_tokens", 0)
        if isinstance(sequence_tokens, (int, float)) and isinstance(format_tokens, (int, float)):
            identity_patch_tokens = max(float(sequence_tokens) - float(format_tokens), 0.0)
    if isinstance(identity_patch_tokens, (int, float)) and math.isfinite(float(identity_patch_tokens)):
        identity_patch_tokens = int(round(float(identity_patch_tokens)))
        if not isinstance(projector_input_tokens, (int, float)):
            projector_input_tokens = identity_patch_tokens
        if not isinstance(projector_output_tokens, (int, float)):
            projector_output_tokens = identity_patch_tokens

    if isinstance(projector_input_tokens, (int, float)):
        metrics["projector_stage_input_tokens"] = int(projector_input_tokens)

    if isinstance(projector_output_tokens, (int, float)):
        metrics["projector_stage_output_tokens"] = int(projector_output_tokens)

    if isinstance(projector_input_tokens, (int, float)) and isinstance(projector_output_tokens, (int, float)):
        projector_input_tokens = int(projector_input_tokens)
        projector_output_tokens = int(projector_output_tokens)
        metrics["projector_stage_keep_ratio"] = (
            float(projector_output_tokens) / float(projector_input_tokens)
            if projector_input_tokens > 0
            else 1.0
        )

    # Keep scalar projector diagnostics available in result samples. These
    # fields are intentionally separate from the standard token/latency
    # metrics so ablation summaries can verify K, M_d/M_c, and residual merge
    # without serializing tensor-valued coordinates or attention scores.
    diagnostic_fields = (
        "voxel_method",
        "voxel_size",
        "attention_reduce",
        "coverage_rule",
        "random_seed",
        "num_voxels_before_post",
        "input_tokens_before_post",
        "nominal_target_tokens",
        "budget_limited_by_voxel_count",
        "target_tokens",
        "dominant_ratio",
        "requested_dominant_tokens",
        "requested_contextual_tokens",
        "dominant_tokens",
        "contextual_tokens",
        "residual_merge",
        "residual_merge_applied",
        "num_residual_merged",
        "projector_compression_time_ms",
        "multimodal_prepare_time_ms",
        "post_prepare_ttft_wall_ms",
        "llm_prefill_cuda_ms",
        "llm_prefill_wall_ms",
        "llm_prefill_sequence_length",
        "llm_decode_cuda_ms",
        "llm_decode_forward_count",
        "llm_forward_count",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
        "peak_memory_allocated_delta_mb",
        "peak_memory_reserved_delta_mb",
    )
    for field in diagnostic_fields:
        value = profile.get(field)
        if isinstance(value, (int, float, bool, str)):
            metrics[field] = value

    prefill_final_visual_tokens = _resolve_prefill_final_visual_tokens(profile)
    if prefill_final_visual_tokens is not None:
        metrics["prefill_final_visual_tokens"] = float(prefill_final_visual_tokens)

    if latency_metrics:
        ttft_wall_ms = latency_metrics.get("ttft_wall_ms")
        ttft_cuda_ms = latency_metrics.get("ttft_cuda_ms")
        if isinstance(ttft_wall_ms, (int, float)):
            metrics["ttft_wall_ms"] = float(ttft_wall_ms)
        if isinstance(ttft_cuda_ms, (int, float)):
            metrics["ttft_cuda_ms"] = float(ttft_cuda_ms)

        for field in (
            "e2e_wall_ms",
            "e2e_cuda_ms",
            "tpot_wall_ms",
            "tpot_cuda_ms",
            "decode_cuda_ms",
        ):
            value = latency_metrics.get(field)
            if isinstance(value, (int, float)):
                metrics[field] = float(value)

        # Keep both clocks: wall time is the end-to-end critical-path measure,
        # while CUDA events provide a current-stream reference. Event timing
        # can include delays before the second event is enqueued and can miss
        # work running on unrelated streams, so it must not be treated as a
        # complete replacement for wall time.
        requested_clock = str(os.environ.get("OV_TTFT_METRIC", "cuda")).strip().lower()
        if requested_clock == "wall":
            ttft_ms = ttft_wall_ms
        elif requested_clock == "cuda":
            ttft_ms = ttft_cuda_ms
        else:
            raise ValueError("OV_TTFT_METRIC must be either 'cuda' or 'wall'.")
        if isinstance(ttft_ms, (int, float)):
            metrics["ttft_ms"] = float(ttft_ms)

    if flops_metrics:
        kv_cache_mb = flops_metrics.get("kv_cache_mb")
        if isinstance(kv_cache_mb, (int, float)):
            metrics["kv_cache_mb"] = float(kv_cache_mb)
        tflops = flops_metrics.get("tflops")
        if isinstance(tflops, (int, float)):
            metrics["tflops"] = float(tflops)

    return metrics


def _normalize_single_compression_profile(raw_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw_profile, dict):
        return {}
    samples = raw_profile.get("samples")
    if isinstance(samples, list):
        if len(samples) == 1 and isinstance(samples[0], dict):
            return samples[0].copy()
        return {}
    return raw_profile.copy()


def _count_text_prompt_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> int:
    prompt_tokens = int(attention_mask.sum().item()) if torch.is_tensor(attention_mask) else int(input_ids.shape[-1])
    image_placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item()) if torch.is_tensor(input_ids) else 0
    return max(prompt_tokens - image_placeholders, 0)


def _resolve_prompt_sequence_tokens(
    profile: Dict[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[int, int]:
    explicit_prompt_tokens = profile.get("prompt_sequence_length")
    text_prompt_tokens = int(profile.get("text_prompt_tokens", _count_text_prompt_tokens(input_ids, attention_mask)))
    visual_sequence_tokens = profile.get("visual_sequence_tokens")
    minimum_prompt_tokens = None
    if isinstance(visual_sequence_tokens, (int, float)):
        minimum_prompt_tokens = text_prompt_tokens + int(visual_sequence_tokens)
    if isinstance(explicit_prompt_tokens, (int, float)):
        explicit_prompt_tokens = int(explicit_prompt_tokens)
        if minimum_prompt_tokens is None or explicit_prompt_tokens >= minimum_prompt_tokens:
            return explicit_prompt_tokens, text_prompt_tokens

        # A scheduler may expose the final text-only state as
        # ``prompt_sequence_length`` even though the multimodal prompt was
        # never physically pruned. Prefer the recorded expanded length when
        # the two fields disagree; silently publishing the text-only length
        # corrupts input-token and FLOPs accounting.
        expanded_prompt_tokens = profile.get("prompt_sequence_length_before_prune")
        if isinstance(expanded_prompt_tokens, (int, float)):
            expanded_prompt_tokens = int(expanded_prompt_tokens)
            if expanded_prompt_tokens >= minimum_prompt_tokens:
                return expanded_prompt_tokens, text_prompt_tokens

        return minimum_prompt_tokens, text_prompt_tokens

    if isinstance(visual_sequence_tokens, (int, float)):
        return text_prompt_tokens + int(visual_sequence_tokens), text_prompt_tokens

    visual_patch_tokens = _resolve_prefill_final_visual_tokens(profile)
    if isinstance(visual_patch_tokens, (int, float)):
        visual_format_tokens = profile.get("visual_format_tokens")
        visual_format_tokens = int(visual_format_tokens) if isinstance(visual_format_tokens, (int, float)) else 0
        return text_prompt_tokens + int(visual_patch_tokens) + visual_format_tokens, text_prompt_tokens

    return int(attention_mask.sum().item()) if torch.is_tensor(attention_mask) else int(input_ids.shape[-1]), text_prompt_tokens


def _resolve_generated_tokens(output_ids: torch.Tensor, input_ids: torch.Tensor) -> int:
    if not torch.is_tensor(output_ids):
        return 0
    output_len = int(output_ids.shape[-1]) if output_ids.ndim > 0 else int(output_ids.numel())
    prompt_len = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) and input_ids.ndim > 0 else 0
    if output_len > prompt_len:
        return max(output_len - prompt_len, 0)
    return max(output_len, 0)


def _infer_ov_video_patch_tokens(image_tensor: Any, spatial_pool_stride: int) -> Optional[Tuple[int, int, int]]:
    if isinstance(image_tensor, list):
        if not image_tensor or not torch.is_tensor(image_tensor[0]):
            return None
        tensor = image_tensor[0]
    elif torch.is_tensor(image_tensor):
        tensor = image_tensor
    else:
        return None

    if tensor.ndim != 4:
        return None
    frames = int(tensor.shape[0])
    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    patch_size = 14
    grid_h = max(height // patch_size, 1)
    grid_w = max(width // patch_size, 1)
    stride = max(int(spatial_pool_stride or 1), 1)
    pooled_h = max(int(math.ceil(grid_h / stride)), 1)
    pooled_w = max(int(math.ceil(grid_w / stride)), 1)
    patch_tokens = int(frames * pooled_h * pooled_w)
    return frames, patch_tokens, pooled_w


def _infer_ov_video_visual_profile(
    image_tensor: Any,
    *,
    spatial_pool_stride: int,
    newline_position: str,
    patch_merge_type: str,
) -> Dict[str, Any]:
    inferred = _infer_ov_video_patch_tokens(image_tensor, spatial_pool_stride)
    if inferred is None:
        return {}
    frames, patch_tokens, pooled_w = inferred
    newline_position = str(newline_position or "one_token")
    patch_merge_type = str(patch_merge_type or "")
    if newline_position == "grid":
        visual_format_tokens = int(frames * pooled_w)
    elif newline_position == "frame":
        visual_format_tokens = int(frames)
    elif newline_position == "one_token" and "unpad" in patch_merge_type:
        visual_format_tokens = 1
    else:
        visual_format_tokens = 0
    visual_sequence_tokens = int(patch_tokens + visual_format_tokens)
    return {
        "compressor_name": "none",
        "projector_stage_input_tokens": int(patch_tokens),
        "projector_stage_output_tokens": int(patch_tokens),
        "token_keep_ratio": 1.0,
        "visual_patch_tokens": int(patch_tokens),
        "visual_sequence_tokens": visual_sequence_tokens,
        "visual_format_tokens": visual_format_tokens,
        "prefill_final_visual_tokens": float(patch_tokens),
    }


class LlavaOneVision3D(Llava_OneVision):
    """LLaVA-OneVision wrapper with projector-level Video3D compressors.

    This wrapper keeps OV as a video model. Depth and camera metadata are used
    only to compute compressor coordinates for voxel methods.
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = "llava_qwen",
        model_base: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,
        customized_config: Optional[str] = None,
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",
        video_decode_backend: str = "decord",
        three_d_config: Optional[str] = None,
        enable_3d_aux: Union[bool, str] = True,
        extra_prompt: str = DEFAULT_OV_3D_EXTRA_PROMPT,
        mm_patch_merge_type: str = "spatial_unpad",
        mm_newline_position: str = "grid",
        mm_projector_compressor_type: Optional[str] = None,
        mm_projector_compressor_config: Optional[Union[str, Dict[str, Any]]] = None,
        mm_llm_compressor_type: Optional[str] = None,
        mm_llm_compressor_config: Optional[Union[str, Dict[str, Any]]] = None,
        mm_depth_scale: float = 1000.0,
        pooled_coords_root: Optional[str] = None,
        scannet_visual_mode: str = "video",
        **kwargs,
    ) -> None:
        if kwargs:
            raise ValueError(f"Unexpected kwargs for llava_onevision_3d: {sorted(kwargs)}")

        super().__init__(
            pretrained=pretrained,
            truncation=truncation,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            model_base=model_base,
            attn_implementation=attn_implementation,
            device_map=device_map,
            conv_template=conv_template,
            use_cache=use_cache,
            truncate_context=truncate_context,
            customized_config=customized_config,
            max_frames_num=max_frames_num,
            mm_spatial_pool_stride=mm_spatial_pool_stride,
            mm_spatial_pool_mode=mm_spatial_pool_mode,
            token_strategy=token_strategy,
            video_decode_backend=video_decode_backend,
        )
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            self._rank = int(os.environ.get("RANK", self._rank))
            self._world_size = int(os.environ.get("WORLD_SIZE", self._world_size))

        self.three_d_config = three_d_config
        self.enable_3d_aux = _as_bool(enable_3d_aux)
        self.extra_prompt = extra_prompt
        self._frame_info_cache: Dict[str, Dict[str, dict]] = {}
        # Coordinate cache files contain RGB and full-resolution coordinate
        # tensors in addition to the small OV pooled tensor we consume here.
        # Keep the validated pooled tensor by resolved path so repeated
        # requests (including distributed padding requests) do not deserialize
        # the same file again.
        self._pooled_coords_tensor_cache: Dict[str, torch.Tensor] = {}
        self.pooled_coords_root = str(pooled_coords_root or os.environ.get("OV_POOLED_COORDS_ROOT", "")).strip()
        self._coords_cache_aliases_cache: Optional[List[Tuple[str, str]]] = None
        self._coords_cache_log_count = 0
        self._projector_compressor_requires_coordinates = False
        self.scannet_visual_mode = str(scannet_visual_mode or "video").strip().lower()
        if self.scannet_visual_mode not in {"video", "multi_image"}:
            raise ValueError("scannet_visual_mode must be either 'video' or 'multi_image'.")

        cfg = self.model.config
        cfg.mm_patch_merge_type = mm_patch_merge_type
        cfg.mm_newline_position = mm_newline_position
        cfg.mm_depth_scale = float(mm_depth_scale)
        if mm_projector_compressor_type not in (None, "", "none", "identity"):
            if mm_projector_compressor_type not in ALLOWED_OV_PROJECTOR_COMPRESSORS:
                raise ValueError(
                    "LLaVA-OV projector compression only keeps "
                    f"{sorted(ALLOWED_OV_PROJECTOR_COMPRESSORS)}; got {mm_projector_compressor_type!r}."
                )
            cfg.mm_projector_compressor_type = mm_projector_compressor_type
            cfg.mm_projector_compressor_config = _parse_json_object(mm_projector_compressor_config) or {}
            self._install_projector_compressor()
        else:
            cfg.mm_projector_compressor_type = None
            cfg.mm_projector_compressor_config = None

        if mm_llm_compressor_type not in (None, "", "none", "identity"):
            if mm_llm_compressor_type not in ALLOWED_OV_LLM_COMPRESSORS:
                raise ValueError(
                    "LLaVA-OV LLM compression/analysis only keeps "
                    f"{sorted(ALLOWED_OV_LLM_COMPRESSORS)}; got {mm_llm_compressor_type!r}."
                )
            cfg.mm_llm_compressor_type = mm_llm_compressor_type
            cfg.mm_llm_compressor_config = _parse_json_object(mm_llm_compressor_config) or {}
            self._install_llm_compressor()
        else:
            cfg.mm_llm_compressor_type = None
            cfg.mm_llm_compressor_config = None

        if self._projector_compressor_requires_coordinates and not self.enable_3d_aux:
            raise ValueError("Voxel projector compression requires enable_3d_aux=true.")
        if cfg.mm_projector_compressor_type is not None and self.scannet_visual_mode != "video":
            raise ValueError("Projector compression is only implemented for scannet_visual_mode='video'.")
        if cfg.mm_llm_compressor_type is not None and self.scannet_visual_mode != "video":
            raise ValueError("LLM analysis is only implemented for scannet_visual_mode='video'.")

    def _install_projector_compressor(self) -> None:
        compressor = build_compressor_from_llava_config(self.model.config, location="projector")
        core_model = self.model.get_model() if hasattr(self.model, "get_model") else self.model.model
        core_model.mm_projector_compressor = compressor
        core_model.mm_compressor = compressor
        self._projector_compressor_requires_coordinates = bool(
            compressor.get_required_inputs().get("coordinates", False)
        )

    def _install_llm_compressor(self) -> None:
        compressor = build_compressor_from_llava_config(self.model.config, location="llm")
        if hasattr(compressor, "configure_tokenizer"):
            compressor.configure_tokenizer(self.tokenizer)
        core_model = self.model.get_model() if hasattr(self.model, "get_model") else self.model.model
        core_model.mm_llm_compressor = compressor

    def _get_llm_compressor(self):
        core_model = self.model.get_model() if hasattr(self.model, "get_model") else self.model.model
        return getattr(core_model, "mm_llm_compressor", None)

    def _asset_manager(self):
        return get_asset_manager(self.three_d_config)

    def _pooled_coords_cache_key(self, doc: dict, frame_shape: Tuple[int, int]) -> str:
        cache_path = coords_cache_path(
            ".",
            video_id=doc.get("video_id"),
            frame_files=doc.get("frame_files", []),
            crop_size=int(self._image_processor.crop_size.get("width", 384)),
            frame_shape=frame_shape,
            depth_scale=float(getattr(self.model.config, "mm_depth_scale", 1000.0)),
        )
        return cache_path.stem

    def _pooled_coords_cache_path(self, doc: dict, frame_shape: Tuple[int, int]) -> Optional[Path]:
        return coords_cache_path(
            self.pooled_coords_root,
            video_id=doc.get("video_id"),
            frame_files=doc.get("frame_files", []),
            crop_size=int(self._image_processor.crop_size.get("width", 384)),
            frame_shape=frame_shape,
            depth_scale=float(getattr(self.model.config, "mm_depth_scale", 1000.0)),
        )

    def _coords_cache_path_aliases(self) -> List[Tuple[str, str]]:
        if self._coords_cache_aliases_cache is not None:
            return self._coords_cache_aliases_cache

        aliases: List[Tuple[str, str]] = []
        raw_aliases = os.environ.get("OV_COORDS_CACHE_PATH_ALIASES", "").strip()
        if raw_aliases:
            for item in raw_aliases.split(","):
                if "=" not in item:
                    continue
                source, target = item.split("=", 1)
                aliases.append((source.strip(), target.strip()))

        try:
            paths = get_data_paths(self.three_d_config)
        except Exception:
            paths = None

        project_root = os.environ.get("VIDEO3D_COMP_ROOT", "").strip()
        if not project_root:
            for parent in Path(__file__).resolve().parents:
                if (parent / "algorithm" / "Video-3D-LLM").is_dir():
                    project_root = str(parent)
                    break

        if paths is not None and project_root:
            project_data = os.path.join(project_root, "data")
            # Older precomputed caches hashed the symlinked 3d-com/data path.
            aliases.append((paths.video_3d_llm_root, project_data))
            aliases.append((paths.scannet_root, os.path.join(project_data, "scannet")))

        normalized = []
        seen = set()
        for source, target in aliases:
            if not source or not target:
                continue
            source = os.path.normpath(os.path.abspath(os.path.expanduser(source)))
            target = os.path.normpath(os.path.abspath(os.path.expanduser(target)))
            if source == target or (source, target) in seen:
                continue
            normalized.append((source, target))
            seen.add((source, target))
        self._coords_cache_aliases_cache = normalized
        return normalized

    @staticmethod
    def _rewrite_frame_files_for_cache_alias(frame_files: List[str], source: str, target: str) -> Optional[List[str]]:
        source_prefix = source + os.sep
        rewritten = []
        changed = False
        for frame_file in frame_files:
            normalized = os.path.normpath(os.path.abspath(os.path.expanduser(str(frame_file))))
            if normalized == source:
                rewritten.append(target)
                changed = True
            elif normalized.startswith(source_prefix):
                rewritten.append(os.path.join(target, normalized[len(source_prefix) :]))
                changed = True
            else:
                rewritten.append(str(frame_file))
        return rewritten if changed else None

    def _pooled_coords_cache_path_candidates(self, doc: dict, frame_shape: Tuple[int, int]) -> List[Path]:
        candidates: List[Path] = []
        primary = self._pooled_coords_cache_path(doc, frame_shape)
        if primary is not None:
            candidates.append(primary)

        frame_files = [str(path) for path in doc.get("frame_files", [])]
        for source, target in self._coords_cache_path_aliases():
            alias_frame_files = self._rewrite_frame_files_for_cache_alias(frame_files, source, target)
            if alias_frame_files is None:
                continue
            alias_path = coords_cache_path(
                self.pooled_coords_root,
                video_id=doc.get("video_id"),
                frame_files=alias_frame_files,
                crop_size=int(self._image_processor.crop_size.get("width", 384)),
                frame_shape=frame_shape,
                depth_scale=float(getattr(self.model.config, "mm_depth_scale", 1000.0)),
            )
            if alias_path is not None and alias_path not in candidates:
                candidates.append(alias_path)
        return candidates

    @staticmethod
    def _load_coords_cache_payload(cache_path: Path):
        """Load a coordinate cache without eagerly reading unrelated tensors.

        The cache was written with ``torch.save`` and is therefore compatible
        with PyTorch's file-backed mmap loader.  ``weights_only`` and ``mmap``
        are optional across the environments used by this project, so retain a
        conservative fallback chain for older PyTorch versions or legacy cache
        payloads.  A failed optimized load is retried normally, while a
        genuinely unreadable file still raises from the final load.
        """
        path = os.fspath(cache_path)
        try:
            return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        except Exception:
            try:
                return torch.load(path, map_location="cpu", mmap=True)
            except Exception:
                return torch.load(path, map_location="cpu")

    def _load_pooled_world_coords(self, doc: dict, frame_shape: Tuple[int, int]) -> Optional[torch.Tensor]:
        cache_path = None
        candidates = self._pooled_coords_cache_path_candidates(doc, frame_shape)
        for candidate in candidates:
            if candidate.exists():
                cache_path = candidate
                break
        if cache_path is None:
            if _as_bool(os.environ.get("OV_LOG_COORDS_CACHE", "0")):
                count = int(getattr(self, "_coords_cache_log_count", 0))
                limit = int(os.environ.get("OV_LOG_COORDS_CACHE_LIMIT", "20"))
                if count < limit:
                    eval_logger.warning(
                        "OV pooled coords cache miss: video_id=%s candidates=%s",
                        doc.get("video_id"),
                        [str(candidate) for candidate in candidates],
                    )
                    setattr(self, "_coords_cache_log_count", count + 1)
            return None
        if _as_bool(os.environ.get("OV_LOG_COORDS_CACHE", "0")):
            count = int(getattr(self, "_coords_cache_log_count", 0))
            limit = int(os.environ.get("OV_LOG_COORDS_CACHE_LIMIT", "20"))
            if count < limit:
                eval_logger.warning(
                    "OV pooled coords cache hit: video_id=%s path=%s",
                    doc.get("video_id"),
                    cache_path,
                )
                setattr(self, "_coords_cache_log_count", count + 1)
        cache_key = os.path.realpath(os.path.abspath(os.fspath(cache_path)))
        expected = (len(doc.get("frame_files", [])), int(frame_shape[0]), int(frame_shape[1]), 3)
        cached_coords = self._pooled_coords_tensor_cache.get(cache_key)
        if cached_coords is not None:
            if tuple(cached_coords.shape) != expected:
                # Do not silently reuse an entry if a cache path was replaced
                # in place with an incompatible artifact during a long run.
                self._pooled_coords_tensor_cache.pop(cache_key, None)
            else:
                return cached_coords

        payload = self._load_coords_cache_payload(cache_path)
        coords = payload.get("ov_pad_avg14") if isinstance(payload, dict) else payload
        if coords is None and isinstance(payload, dict):
            coords = payload.get("world_coords")
        if not torch.is_tensor(coords):
            raise TypeError(f"Cached pooled coords must be a tensor: {cache_path}")
        if tuple(coords.shape) != expected:
            raise ValueError(f"Cached pooled coords shape mismatch at {cache_path}: {tuple(coords.shape)} vs {expected}")
        coords = coords.contiguous()
        self._pooled_coords_tensor_cache[cache_key] = coords
        return coords

    def _resolve_frame_info(self, doc: dict, frame_file: str) -> Optional[dict]:
        video_id = doc.get("video_id")
        if not video_id:
            return None
        cache = self._frame_info_cache.get(video_id)
        assets = self._asset_manager()
        if cache is None:
            scene = assets.get_scene(video_id)
            cache = {}
            for image_info in scene.get("images", []):
                resolved = os.path.abspath(assets._resolve_scannet_path(image_info["img_path"]))
                cache[resolved] = image_info
                cache[os.path.normpath(image_info["img_path"])] = image_info
            self._frame_info_cache[video_id] = cache

        abs_frame = os.path.abspath(frame_file)
        if abs_frame in cache:
            return cache[abs_frame]
        normalized = os.path.normpath(frame_file)
        if normalized in cache:
            return cache[normalized]

        # Fall back to matching the ScanNet-relative suffix used in task docs.
        for key, value in cache.items():
            if isinstance(key, str) and normalized.endswith(os.path.normpath(value.get("img_path", ""))):
                return value
        return None

    def _load_world_coords(self, doc: dict) -> Optional[torch.Tensor]:
        if not self.enable_3d_aux or "frame_files" not in doc or "video_id" not in doc:
            return None
        cached = self._load_pooled_world_coords(doc, frame_shape=(14, 14))
        if cached is not None:
            return cached

        assets = self._asset_manager()
        paths = get_data_paths(self.three_d_config)
        scene = assets.get_scene(doc["video_id"])
        axis_align = torch.as_tensor(np.array(scene.get("axis_align_matrix", np.eye(4))), dtype=torch.float32)
        intrinsic = torch.as_tensor(np.array(scene.get("depth_cam2img", scene.get("cam2img"))), dtype=torch.float32)

        depths = []
        poses = []
        for frame_file in doc["frame_files"]:
            frame_info = self._resolve_frame_info(doc, frame_file)
            if frame_info is None:
                eval_logger.warning("Missing frame metadata for %s; voxel compressor will not receive 3D coordinates.", frame_file)
                return None

            depth_path = frame_info.get("depth_path") or frame_file.replace(".jpg", ".png")
            if not os.path.isabs(depth_path):
                depth_path = assets._resolve_scannet_path(depth_path)
            if not os.path.exists(depth_path):
                # Some metadata stores paths relative to the ScanNet root.
                depth_path = os.path.join(paths.scannet_root, os.path.normpath(depth_path).lstrip("/"))
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Cannot find depth image for OV voxel compression: {depth_path}")

            with PIL.Image.open(depth_path) as depth_img:
                depths.append(torch.from_numpy(np.asarray(depth_img).astype(np.float32)))

            cam2global = torch.as_tensor(np.array(frame_info["cam2global"]), dtype=torch.float32)
            poses.append(axis_align @ cam2global)

        depths_t = torch.stack(depths, dim=0)
        poses_t = torch.stack(poses, dim=0)
        intrinsic_t = intrinsic.unsqueeze(0).repeat(depths_t.shape[0], 1, 1)
        world_coords = self._unproject(intrinsic_t, poses_t, depths_t)
        return self._resize_world_coords_like_ov_pad(world_coords)

    def _unproject(self, intrinsics: torch.Tensor, poses: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        num_frames, height, width = depths.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        xx = xx.reshape(1, -1).repeat(num_frames, 1)
        yy = yy.reshape(1, -1).repeat(num_frames, 1)
        z = depths.reshape(num_frames, -1)
        if z.numel() > 0 and float(z.max().item()) > 100.0:
            z = z / float(getattr(self.model.config, "mm_depth_scale", 1000.0))

        fx = intrinsics[:, 0, 0].unsqueeze(-1)
        fy = intrinsics[:, 1, 1].unsqueeze(-1)
        cx = intrinsics[:, 0, 2].unsqueeze(-1)
        cy = intrinsics[:, 1, 2].unsqueeze(-1)
        x = (xx - cx) * z / fx.clamp_min(1e-6)
        y = (yy - cy) * z / fy.clamp_min(1e-6)
        cam_xyz = torch.stack((x, y, z, torch.ones_like(z)), dim=-1)
        world = torch.bmm(poses, cam_xyz.transpose(1, 2)).transpose(1, 2)
        world = world[..., :3] / world[..., 3:].clamp_min(1e-6)
        return world.reshape(num_frames, height, width, 3)

    def _resize_world_coords_like_ov_pad(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Match LLaVA-OV's multi-frame `image_aspect_ratio=pad` preprocessing."""
        crop_size = int(self._image_processor.crop_size.get("width", 384))
        _, height, width, _ = world_coords.shape
        coords = world_coords.permute(0, 3, 1, 2).contiguous()
        if width > height:
            pad_top = (width - height) // 2
            pad_bottom = width - height - pad_top
            coords = F.pad(coords, (0, 0, pad_top, pad_bottom), mode="replicate")
        elif height > width:
            pad_left = (height - width) // 2
            pad_right = height - width - pad_left
            coords = F.pad(coords, (pad_left, pad_right, 0, 0), mode="replicate")
        coords = F.interpolate(coords, size=(crop_size, crop_size), mode="nearest")
        return coords.permute(0, 2, 3, 1).contiguous()

    def _pool_world_coords_for_offline_cache(self, world_coords: torch.Tensor, frame_shape: Tuple[int, int] = (14, 14)) -> torch.Tensor:
        if world_coords.dim() != 4 or world_coords.shape[-1] != 3:
            raise ValueError(f"world_coords must be shaped as (frames, H, W, 3), got {tuple(world_coords.shape)}.")
        coords = torch.nan_to_num(world_coords.to(dtype=torch.float32))
        coords = coords.permute(0, 3, 1, 2).contiguous()
        coords = F.adaptive_avg_pool2d(coords, output_size=frame_shape)
        return coords.permute(0, 2, 3, 1).contiguous()

    def compute_pooled_world_coords_for_doc(self, doc: dict, frame_shape: Tuple[int, int] = (14, 14)) -> torch.Tensor:
        previous_root = self.pooled_coords_root
        self.pooled_coords_root = ""
        try:
            world_coords = self._load_world_coords(doc)
        finally:
            self.pooled_coords_root = previous_root
        if world_coords is None:
            raise ValueError(f"Could not compute world coords for {doc.get('video_id')}.")
        if tuple(world_coords.shape[1:3]) == tuple(frame_shape):
            return world_coords.to(dtype=torch.float32).contiguous()
        return self._pool_world_coords_for_offline_cache(world_coords, frame_shape=frame_shape)

    def _with_extra_prompt(self, context: str) -> str:
        prompt = (context or "").strip()
        extra = (self.extra_prompt or "").strip()
        if extra and not prompt.startswith(extra):
            prompt = f"{extra}\n{prompt}" if prompt else extra
        return prompt

    def generate_until(self, requests: List[Instance]) -> List[Union[str, GenerationResult]]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        request_payloads = [
            (*reg.args, bool(reg.metadata.get("__padding_only__", False)))
            for reg in requests
        ]
        re_ords = utils.Collator(request_payloads, _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_id,
                batched_task,
                batched_split,
                batched_padding_only,
            ) = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            padding_only = bool(batched_padding_only[0])
            docs = [self.task_dict[task][split][ids] for ids in batched_doc_id]
            batched_visuals = [batched_doc_to_visual[0](doc) for doc in docs]
            assert len(batched_visuals) == 1, "LLaVA-OneVision generation is currently batch_size=1."

            gen_kwargs = dict(all_gen_kwargs[0])
            gen_kwargs.pop("until", None)
            doc = docs[0]
            visual = batched_visuals[0]
            image_tensor = None
            video_dict = None
            task_type = "text"
            placeholder_count = 0

            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio

            if visual:
                is_scannet_video = "frame_files" in doc and "video_id" in doc
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = gen_kwargs.get("image_aspect_ratio", "pad")

                if is_scannet_video and self.scannet_visual_mode == "video":
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if isinstance(image_tensor, list):
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    if torch.is_tensor(image_tensor) and image_tensor.ndim == 4:
                        image_tensor = [image_tensor]

                    world_coords = self._load_world_coords(doc) if self._projector_compressor_requires_coordinates else None
                    if world_coords is not None:
                        video_dict = {"world_coords": world_coords.unsqueeze(0).to(device=self.device, dtype=torch.float32)}
                    task_type = "video"
                    placeholder_count = 1
                elif is_scannet_video and self.scannet_visual_mode == "multi_image":
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if isinstance(image_tensor, list):
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    task_type = "image"
                    placeholder_count = len(visual)
                elif isinstance(visual[0], PIL.Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if isinstance(image_tensor, list):
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    task_type = "image"
                    placeholder_count = len(visual) if isinstance(visual, list) else 1
                else:
                    raise TypeError(f"Unsupported visual payload for llava_onevision_3d: {type(visual[0]).__name__}")

            context = self._with_extra_prompt(batched_contexts[0])
            if image_tensor is not None and DEFAULT_IMAGE_TOKEN not in context:
                question = " ".join([DEFAULT_IMAGE_TOKEN] * placeholder_count) + "\n" + context
            else:
                question = context

            conv = copy.deepcopy(conv_templates[self.conv_template]) if "llama_3" in self.conv_template else conv_templates[self.conv_template].copy()
            if utils.is_json(question):
                question = json.loads(question)
                for idx, item in enumerate(question):
                    conv.append_message(conv.roles[idx % 2], item["value"])
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
            else:
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            if task_type in {"image", "video"}:
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                gen_kwargs["stopping_criteria"] = [KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)]

            if task_type == "image":
                gen_kwargs["image_sizes"] = [item.size for item in visual]
            elif task_type == "video":
                gen_kwargs["modalities"] = ["video"]
                if video_dict is not None:
                    gen_kwargs["video_dict"] = video_dict
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            gen_kwargs.setdefault("max_new_tokens", 1024)
            gen_kwargs.setdefault("do_sample", False)
            gen_kwargs.setdefault("num_beams", 1)
            gen_kwargs.pop("image_aspect_ratio", None)
            if not gen_kwargs.get("do_sample", False):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)

            if hasattr(self.model, "reset_last_compression_profile"):
                self.model.reset_last_compression_profile()
            llm_compressor = self._get_llm_compressor()
            if (
                llm_compressor is not None
                and hasattr(llm_compressor, "set_next_sample_key")
                and llm_compressor.should_analyze()
                and not padding_only
            ):
                llm_compressor.set_next_sample_key(f"{task}:{split}:{batched_doc_id[0]}")
            measure_runtime_profile = _runtime_profile_enabled()
            runtime_profile = {}
            memory_start_allocated_mb = None
            memory_start_reserved_mb = None
            if measure_runtime_profile and self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                memory_start_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024.0 ** 2)
                memory_start_reserved_mb = torch.cuda.memory_reserved(self.device) / (1024.0 ** 2)
                torch.cuda.reset_peak_memory_stats(self.device)
            streamer = LatencyStreamer(device=self.device)
            streamer.start()
            with torch.inference_mode():
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    skip_llm_analysis=padding_only,
                    use_cache=self.use_cache,
                    streamer=streamer,
                    **gen_kwargs,
                )
            streamer.end()

            if hasattr(self.model, "get_last_runtime_profile"):
                runtime_profile = self.model.get_last_runtime_profile()
            if measure_runtime_profile and self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                peak_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
                peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 2)
                runtime_profile.update(
                    {
                        "peak_memory_allocated_mb": float(peak_allocated_mb),
                        "peak_memory_reserved_mb": float(peak_reserved_mb),
                    }
                )
                if memory_start_allocated_mb is not None:
                    runtime_profile["peak_memory_allocated_delta_mb"] = float(
                        peak_allocated_mb - memory_start_allocated_mb
                    )
                if memory_start_reserved_mb is not None:
                    runtime_profile["peak_memory_reserved_delta_mb"] = float(
                        peak_reserved_mb - memory_start_reserved_mb
                    )

            raw_profile = self.model.get_last_compression_profile() if hasattr(self.model, "get_last_compression_profile") else {}
            compression_profile = _normalize_single_compression_profile(raw_profile)
            if hasattr(self.model, "reset_last_compression_profile"):
                self.model.reset_last_compression_profile()
            if not compression_profile and task_type == "video":
                compression_profile = _infer_ov_video_visual_profile(
                    image_tensor,
                    spatial_pool_stride=self.mm_spatial_pool_stride,
                    newline_position=getattr(self._config, "mm_newline_position", "one_token"),
                    patch_merge_type=getattr(self._config, "mm_patch_merge_type", ""),
                )

            latency_metrics = streamer.get_metrics() or {}
            prepare_time_ms = runtime_profile.get("multimodal_prepare_time_ms")
            ttft_wall_ms = latency_metrics.get("ttft_wall_ms")
            if isinstance(prepare_time_ms, (int, float)) and isinstance(ttft_wall_ms, (int, float)):
                # This is the post-preparation path to the first token.  It
                # includes the LLM prefill and first-token generation work,
                # but is deliberately not labeled as a pure decoder kernel
                # time.
                runtime_profile["post_prepare_ttft_wall_ms"] = max(
                    float(ttft_wall_ms) - float(prepare_time_ms), 0.0
                )
            if runtime_profile:
                compression_profile.update(runtime_profile)

            prompt_tokens, text_prompt_tokens = _resolve_prompt_sequence_tokens(
                compression_profile,
                input_ids,
                attention_masks,
            )
            if compression_profile:
                compression_profile.setdefault("text_prompt_tokens", text_prompt_tokens)
                compression_profile.setdefault("prompt_sequence_length", prompt_tokens)
            generated_tokens = latency_metrics.get("generated_tokens")
            if not isinstance(generated_tokens, (int, float)):
                generated_tokens = _resolve_generated_tokens(cont, input_ids)
            flops_metrics = estimate_generation_flops(
                self.model.config,
                prompt_tokens=prompt_tokens,
                generated_tokens=int(generated_tokens),
                compression_efficiency=compression_profile,
                model=self.model,
            )
            compression_efficiency = _build_public_compression_efficiency(
                compression_profile,
                latency_metrics=latency_metrics,
                flops_metrics=flops_metrics,
            )
            compressor_type = getattr(self.model.config, "mm_projector_compressor_type", None)
            if compressor_type not in (None, "", "none", "identity") and _log_compression_profile_enabled():
                stats = self.model.get_compression_token_stats() if hasattr(self.model, "get_compression_token_stats") else {}
                eval_logger.warning("OV projector compression profile: %s | stats: %s", compression_profile, stats)
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            text_outputs = [response.strip() for response in text_outputs]
            for response in text_outputs:
                res.append(
                    GenerationResult(
                        text=response,
                        token_counts=TokenCounts(
                            input_tokens=prompt_tokens,
                            output_tokens=int(generated_tokens),
                        ),
                        metadata={"compression_efficiency": compression_efficiency},
                    )
                )
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        pbar.close()
        llm_compressor = self._get_llm_compressor()
        if llm_compressor is not None and hasattr(llm_compressor, "finalize"):
            llm_compressor.finalize(accelerator=getattr(self, "accelerator", None))
        return re_ords.get_original(res)
