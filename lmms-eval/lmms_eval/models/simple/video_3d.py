from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.tasks._task_utils.scannet3d.config import get_data_paths
from lmms_eval.models.model_utils.compression_efficiency import estimate_generation_flops
from lmms_eval.models.model_utils.gen_metrics import LatencyStreamer


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_structured_value(value):
    if value is None or isinstance(value, (dict, list, tuple, bool, int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(stripped)
    return value


def _compressor_enabled(compressor_type: Optional[str]) -> bool:
    return compressor_type not in (None, "", "none", "identity")


@contextlib.contextmanager
def _temporary_cwd(path: str):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_task_family(task_name: str) -> str:
    task_name = task_name.lower()
    for family in ("scanqa", "sqa3d", "scan2cap", "scanrefer", "multi3drefer"):
        if task_name.startswith(family):
            return family
    raise ValueError(f"Unsupported Video-3D-LLM task: {task_name}")


def _human_prompt(doc: dict, task_family: str, extra_prompt: str = "") -> str:
    del extra_prompt
    prefix = "<image> "
    if task_family == "scanqa":
        body = f"{doc['question']} Answer the question simply."
    elif task_family == "sqa3d":
        body = f"{doc['situation']} {doc['question']} Answer the question using a single word or phrase."
    elif task_family == "scan2cap":
        body = "Given an object located at <coord> , describe the object in detail."
    elif task_family == "scanrefer":
        prefix = "<image>"
        body = f"Identify the object according to the following description.\n{doc['description']}"
    elif task_family == "multi3drefer":
        prefix = "<image>"
        body = (
            "Identify the object according to the following description.\n"
            f"{doc['description']}\n"
            "There may be no corresponding object, or there may be one or more objects."
        )
    else:
        raise ValueError(f"Unsupported Video-3D-LLM task family: {task_family}")
    return f"{prefix}{body}"


def _prompt_prefix(task_family: str) -> str:
    return "<image>" if task_family in {"scanrefer", "multi3drefer"} else "<image> "


DEFAULT_VIDEO3D_EXTRA_PROMPT = (
    "The video captures 3D spatial information of a scene. "
    "Please focus on the spatial relationships in the video and answer the following questions.\n"
)


def _assistant_target(doc: dict, task_family: str) -> Optional[str]:
    if task_family in {"scanqa", "sqa3d", "scan2cap"}:
        return None
    if task_family in {"scanrefer", "multi3drefer"}:
        return "<ground>"
    raise ValueError(f"Unsupported Video-3D-LLM task family: {task_family}")


def _resolve_prefill_final_visual_tokens(profile: Dict[str, Any]) -> Optional[float]:
    text_prompt_tokens = profile.get("text_prompt_tokens")
    # Older projector profiles did not record formatting tokens. For those
    # profiles the expanded visual sequence is patch-only, so zero is the
    # correct compatibility default.
    visual_format_tokens = profile.get("visual_format_tokens", 0)
    layer_token_lengths = profile.get("prefill_layer_token_lengths")
    has_llm_stage_profile = any(
        key in profile
        for key in (
            "llm_stage_input_tokens",
            "llm_stage_output_tokens",
            "llm_stage_keep_ratio",
            "llm_prune_layer",
            "late_entry_layer",
            "early_exit_layer",
        )
    )
    if (
        has_llm_stage_profile
        and
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
        has_llm_stage_profile
        and
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

    # A projector-only profile may contain a final expanded sequence length
    # but no LLM layer schedule. Preserve that profile's sequence-level
    # meaning instead of interpreting an unrelated layer-length trace as an
    # LLM compression schedule.
    if (
        not has_llm_stage_profile
        and isinstance(final_sequence_length := profile.get("final_sequence_length"), (int, float))
        and isinstance(text_prompt_tokens, (int, float))
    ):
        return float(
            max(int(final_sequence_length) - int(text_prompt_tokens) - int(visual_format_tokens), 0)
        )

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
    if isinstance(projector_input_tokens, (int, float)):
        metrics["projector_stage_input_tokens"] = int(projector_input_tokens)

    projector_output_tokens = profile.get("projector_stage_output_tokens")
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

    # Keep the two-stage fields visible in the result JSON.  They are also
    # consumed by the FLOPs/KV-cache estimator, so dropping them here would
    # silently turn an LLM-compressed run into a projector-only estimate.
    for key in (
        "llm_stage_input_tokens",
        "llm_stage_output_tokens",
        "llm_stage_keep_ratio",
        "llm_prune_layer",
        "late_entry_layer",
        "early_exit_layer",
        "active_visual_layers",
        "prompt_sequence_length_before_prune",
        "prompt_sequence_length",
        "final_sequence_length",
        "text_prompt_tokens",
        "visual_patch_tokens",
        "visual_sequence_tokens",
        "visual_format_tokens",
        "prefill_layer_token_lengths",
        "position_id_strategy",
        "query_mode",
    ):
        value = profile.get(key)
        if isinstance(value, (int, float, str, list, tuple)) and not isinstance(value, bool):
            metrics[key] = list(value) if isinstance(value, tuple) else value

    prefill_final_visual_tokens = _resolve_prefill_final_visual_tokens(profile)
    if prefill_final_visual_tokens is not None:
        metrics["prefill_final_visual_tokens"] = float(prefill_final_visual_tokens)

    if latency_metrics:
        # Keep both clocks and the complete generation-path timing.  The
        # shared LatencyStreamer already excludes the initial HF prompt
        # callback, so these fields are directly comparable with OV runs.
        for field in (
            "ttft_wall_ms",
            "ttft_cuda_ms",
            "e2e_wall_ms",
            "e2e_cuda_ms",
            "tpot_wall_ms",
            "tpot_cuda_ms",
            "decode_cuda_ms",
        ):
            value = latency_metrics.get(field)
            if isinstance(value, (int, float)):
                metrics[field] = float(value)

        ttft_ms = latency_metrics.get("ttft_cuda_ms")
        if not isinstance(ttft_ms, (int, float)):
            ttft_ms = latency_metrics.get("ttft_wall_ms")
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


def _normalize_compression_profile(profile: Any) -> Dict[str, Any]:
    """Convert the model profile contract to the single-sample wrapper form."""
    if not isinstance(profile, dict):
        return {}
    samples = profile.get("samples")
    if samples is None:
        return profile
    if not isinstance(samples, list):
        raise ValueError("compression profile 'samples' must be a list")
    if len(samples) == 0:
        return {}
    if len(samples) != 1 or not isinstance(samples[0], dict):
        raise ValueError(
            "Video3D evaluator supports one request per generation, but received "
            f"{len(samples)} compression profiles."
        )
    return samples[0]


@register_model("video_3d")
class Video3D(lmms):
    def __init__(
        self,
        pretrained: str,
        model_base: Optional[str] = None,
        repo_root: Optional[str] = None,
        three_d_config: Optional[str] = None,
        video_folder: Optional[str] = None,
        embodiedscan_folder: Optional[str] = None,
        device: str = "cuda",
        batch_size: Union[int, str] = 1,
        frame_sampling_strategy: str = "uniform",
        max_frame_num: int = 32,
        force_sample: Union[bool, str] = True,
        overwrite_cfg: Union[bool, str] = False,
        lora_path: Optional[str] = None,
        extra_prompt: str = DEFAULT_VIDEO3D_EXTRA_PROMPT,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        grounding_threshold: float = 0.4,
        attn_implementation: Optional[str] = None,
        mm_patch_merge_type: Optional[str] = None,
        mm_newline_position: Optional[str] = None,
        mm_compressor_type: Optional[str] = None,
        mm_compressor_config: Optional[Union[dict, str]] = None,
        mm_projector_compressor_type: Optional[str] = None,
        mm_projector_compressor_config: Optional[Union[dict, str]] = None,
        mm_llm_compressor_type: Optional[str] = None,
        mm_llm_compressor_config: Optional[Union[dict, str]] = None,
        mm_vision_tower: Optional[str] = None,
        coords_cache_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected kwargs for video_3d: {sorted(kwargs)}")

        self.pretrained = pretrained
        self.model_name = pretrained
        self.model_base = model_base
        self.requested_device = device
        self.batch_size_per_gpu = int(batch_size)
        self.frame_sampling_strategy = frame_sampling_strategy
        self.max_frame_num = int(max_frame_num)
        self.force_sample = _as_bool(force_sample)
        self.overwrite_cfg = _as_bool(overwrite_cfg)
        self.lora_path = lora_path
        self.extra_prompt = extra_prompt
        self.default_temperature = float(temperature)
        self.default_top_p = top_p
        self.default_num_beams = int(num_beams)
        self.grounding_threshold = float(grounding_threshold)
        self.attn_implementation = attn_implementation
        self.mm_patch_merge_type = mm_patch_merge_type
        self.mm_newline_position = mm_newline_position
        self.mm_compressor_type = mm_compressor_type
        self.mm_compressor_config = _parse_structured_value(mm_compressor_config)
        self.mm_projector_compressor_type = mm_projector_compressor_type
        self.mm_projector_compressor_config = _parse_structured_value(mm_projector_compressor_config)
        self.mm_llm_compressor_type = mm_llm_compressor_type
        self.mm_llm_compressor_config = _parse_structured_value(mm_llm_compressor_config)
        self.mm_vision_tower = mm_vision_tower
        self.coords_cache_root = str(coords_cache_root or os.environ.get("SCANNET3D_COORDS_CACHE_ROOT", "")).strip()
        if self.mm_projector_compressor_type not in (None, "") and self.mm_compressor_type not in (None, ""):
            raise ValueError(
                "Both mm_projector_compressor_type and legacy mm_compressor_type are set. "
                "Use only one projector-level compressor entrypoint."
            )
        # torchrun data parallel support: expose rank/world_size to evaluator padding logic
        self._rank = int(os.environ.get("RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))

        paths = get_data_paths(three_d_config)
        self.repo_root = os.path.abspath(repo_root or paths.video_3d_llm_root)
        self.video_folder = os.path.abspath(video_folder or str(Path(paths.scannet_root).parent))
        self.embodiedscan_folder = os.path.abspath(embodiedscan_folder or paths.embodiedscan_root)

        if not os.path.isdir(self.repo_root):
            raise FileNotFoundError(f"Video-3D-LLM repo root does not exist: {self.repo_root}")

        if device not in {"cuda", "cuda:0", "auto"}:
            eval_logger.warning(
                "video_3d currently loads through Video-3D-LLM's native builder with device_map=auto; requested device '{}' may not be honored exactly.",
                device,
            )

        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)

        try:
            from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
            from llava.model.multimodal_compressor import get_compressor_spec_from_llava_config
            from llava.utils import disable_torch_init
            from llava.video_utils import VideoProcessor, merge_video_dict
        except Exception as exc:
            raise ImportError(f"Failed to import Video-3D-LLM runtime from {self.repo_root}: {exc}") from exc

        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._IGNORE_INDEX = IGNORE_INDEX
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._SeparatorStyle = SeparatorStyle
        self._conv_templates = conv_templates
        self._get_model_name_from_path = get_model_name_from_path
        self._load_pretrained_model = load_pretrained_model
        self._get_compressor_spec_from_llava_config = get_compressor_spec_from_llava_config
        self._disable_torch_init = disable_torch_init
        self._VideoProcessor = VideoProcessor
        self._merge_video_dict = merge_video_dict

        self._disable_torch_init()
        model_name = self._get_model_name_from_path(os.path.expanduser(self.pretrained))
        # Video3D checkpoints are commonly named after the experiment rather
        # than "llava" or "qwen". The upstream loader uses those substrings to
        # choose its multimodal branch, so derive the family from config instead.
        checkpoint_config_path = Path(os.path.expanduser(self.pretrained)) / "config.json"
        checkpoint_config = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
        checkpoint_architectures = set(checkpoint_config.get("architectures") or [])
        if (
            checkpoint_config.get("model_type") == "llava_qwen"
            or "LlavaQwenForCausalLM" in checkpoint_architectures
        ) and not ({"llava", "qwen"} & set(model_name.lower().replace("-", "_").split("_"))):
            model_name = f"llava_qwen_{model_name}"

        overwrite_config = {}
        if self.lora_path is not None:
            overwrite_config = AutoConfig.from_pretrained(self.lora_path).to_dict()
        elif self.overwrite_cfg:
            overwrite_config.update({"tie_word_embeddings": False, "use_cache": True, "vocab_size": 151649})
        if self.mm_patch_merge_type is not None:
            overwrite_config["mm_patch_merge_type"] = self.mm_patch_merge_type
        if self.mm_newline_position is not None:
            overwrite_config["mm_newline_position"] = self.mm_newline_position
        if self.mm_compressor_type is not None:
            overwrite_config["mm_compressor_type"] = self.mm_compressor_type
        if self.mm_compressor_config is not None:
            overwrite_config["mm_compressor_config"] = self.mm_compressor_config
        if self.mm_projector_compressor_type is not None:
            overwrite_config["mm_projector_compressor_type"] = self.mm_projector_compressor_type
        if self.mm_projector_compressor_config is not None:
            overwrite_config["mm_projector_compressor_config"] = self.mm_projector_compressor_config
        if self.mm_llm_compressor_type is not None:
            overwrite_config["mm_llm_compressor_type"] = self.mm_llm_compressor_type
        if self.mm_llm_compressor_config is not None:
            overwrite_config["mm_llm_compressor_config"] = self.mm_llm_compressor_config
        if self.mm_vision_tower is not None:
            overwrite_config["mm_vision_tower"] = os.path.expanduser(self.mm_vision_tower)

        load_kwargs = {
            "overwrite_config": overwrite_config,
        }
        if self.attn_implementation is not None:
            load_kwargs["attn_implementation"] = self.attn_implementation

        tokenizer, model, image_processor, context_len = self._load_pretrained_model(
            os.path.expanduser(self.pretrained),
            self.model_base,
            model_name,
            **load_kwargs,
        )

        if self.lora_path is not None:
            from peft import PeftModel
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
            model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(model, self.lora_path, adapter_name="lora")
            model = model.merge_and_unload()
            state_dict = torch.load(os.path.join(self.lora_path, "non_lora_trainables.bin"), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        self._tokenizer = tokenizer
        self._model = model.eval()
        self._image_processor = image_processor
        self._context_len = context_len
        self._conv_mode = "qwen_1_5"
        self._device = self._infer_model_device()
        projector_compressor_type, _ = self._get_compressor_spec_from_llava_config(self._model.config, location="projector")
        llm_compressor_type, _ = self._get_compressor_spec_from_llava_config(self._model.config, location="llm")
        self._active_projector_compressor_type = projector_compressor_type
        self._active_llm_compressor_type = llm_compressor_type

        with _temporary_cwd(self.repo_root):
            self._video_processor = self._VideoProcessor(
                video_folder=self.video_folder,
                annotation_dir=self.embodiedscan_folder,
                frame_sampling_strategy=self.frame_sampling_strategy,
                coords_cache_root=self.coords_cache_root,
            )

        eval_logger.info(
            "Loaded video_3d model from {} with repo_root={}, video_folder={}, embodiedscan_folder={}",
            self.pretrained,
            self.repo_root,
            self.video_folder,
            self.embodiedscan_folder,
        )

    @property
    def config(self):
        return self.model.config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def max_length(self):
        return self._context_len

    def _infer_model_device(self) -> torch.device:
        if hasattr(self._model, "device"):
            return torch.device(self._model.device)
        return next(self._model.parameters()).device

    def _get_llm_compressor(self):
        core_model = self.model.get_model() if hasattr(self.model, "get_model") else self.model.model
        return getattr(core_model, "mm_llm_compressor", None)

    def _grounding_uses_unsupported_compression(self) -> bool:
        object_feature_type = str(getattr(self.model.config, "object_feature_type", "") or "")
        if _compressor_enabled(self._active_projector_compressor_type) and "patch14" not in object_feature_type:
            return True
        return False

    def _ensure_grounding_supported(self, task_family: str) -> None:
        if task_family not in {"scanrefer", "multi3drefer"}:
            return
        if not self._grounding_uses_unsupported_compression():
            return
        projector_type = self._active_projector_compressor_type or "none"
        llm_type = self._active_llm_compressor_type or "none"
        object_feature_type = str(getattr(self.model.config, "object_feature_type", "") or "")
        raise NotImplementedError(
            "Video-3D-LLM grounding with projector-level compression currently requires "
            "object_feature_type containing 'patch14'. "
            f"Found object_feature_type={object_feature_type!r}, "
            f"projector compressor={projector_type!r}, llm compressor={llm_type!r}."
        )

    def _build_grounding_result(
        self,
        *,
        text: str,
        input_ids: torch.Tensor,
        forward_wall_ms: float,
    ) -> GenerationResult:
        compression_profile = {}
        if hasattr(self.model, "pop_last_compression_profile"):
            compression_profile = _normalize_compression_profile(
                self.model.pop_last_compression_profile()
            )

        prompt_tokens = int(compression_profile.get("prompt_sequence_length", input_ids.shape[1]))
        flops_metrics = estimate_generation_flops(
            self.model.config,
            prompt_tokens=prompt_tokens,
            generated_tokens=0,
            compression_efficiency=compression_profile,
            model=self.model,
        )
        compression_efficiency = _build_public_compression_efficiency(
            compression_profile,
            flops_metrics=flops_metrics,
        )

        return GenerationResult(
            text=text,
            token_counts=TokenCounts(
                input_tokens=prompt_tokens,
                output_tokens=0,
            ),
            metadata={"compression_efficiency": compression_efficiency},
        )

    def _preprocess_qwen(
        self,
        sources: List[Dict[str, Optional[str]]],
        has_image: bool = False,
        include_labels: bool = False,
        system_message: str = "You are a helpful assistant.",
    ):
        roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        im_start, im_end = self.tokenizer.additional_special_tokens_ids[:2]
        nl_tokens = self.tokenizer("\n").input_ids
        system_tokens = self.tokenizer("system").input_ids + nl_tokens

        source = sources
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id: List[int] = []
        target: List[int] = []
        system = [im_start] + system_tokens + self.tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [self._IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens

        for sentence in source:
            role = roles[sentence["from"]]
            value = sentence["value"]
            if has_image and value is not None and self._DEFAULT_IMAGE_TOKEN in value:
                num_image = len(re.findall(re.escape(self._DEFAULT_IMAGE_TOKEN), value))
                texts = value.split(self._DEFAULT_IMAGE_TOKEN)
                cur = self.tokenizer(role).input_ids + nl_tokens
                for index, text in enumerate(texts):
                    cur += self.tokenizer(text).input_ids
                    if index < len(texts) - 1:
                        cur += [self._IMAGE_TOKEN_INDEX] + nl_tokens
                cur += [im_end] + nl_tokens
                if sum(token == self._IMAGE_TOKEN_INDEX for token in cur) != num_image:
                    raise ValueError("Image token count mismatch while preparing Video-3D-LLM prompt.")
            else:
                if value is None:
                    cur = self.tokenizer(role).input_ids + nl_tokens
                else:
                    cur = self.tokenizer(role).input_ids + nl_tokens + self.tokenizer(value).input_ids + [im_end] + nl_tokens

            input_id += cur
            if role == "<|im_start|>user":
                cur_target = [im_start] + [self._IGNORE_INDEX] * (len(cur) - 3) + [im_end] + nl_tokens
            elif role == "<|im_start|>assistant":
                prefix_len = len(self.tokenizer(role).input_ids)
                cur_target = [im_start] + [self._IGNORE_INDEX] * prefix_len + cur[prefix_len + 1 : -2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError(role)
            target += cur_target

        input_ids = torch.tensor([input_id], dtype=torch.long, device=self.device)
        if include_labels:
            labels = torch.tensor([target], dtype=torch.long, device=self.device)
            return input_ids, labels
        return input_ids

    def _prepare_video_inputs(self, doc: dict, task_family: str):
        if task_family == "scan2cap" and doc.get("box_input") is None:
            return None

        with _temporary_cwd(self.repo_root):
            video_dict = self._video_processor.process_3d_video(
                doc["video_id"],
                self._image_processor,
                force_sample=self.force_sample,
                frames_upbound=self.max_frame_num,
            )
        if task_family == "scan2cap":
            video_dict["box_input"] = list(doc["box_input"][:3])
        video_dict = self._merge_video_dict([video_dict])
        image_tensors = video_dict.pop("images").half().to(self.device)
        for key, value in video_dict.items():
            video_dict[key] = value.half().to(self.device)
        return image_tensors, video_dict

    def _resolve_human_prompt(self, task_context: Optional[str], doc: dict, task_family: str) -> str:
        """
        Build the actual model input prompt.
        We intentionally prepend extra_prompt for all Video-3D tasks before
        applying the native task prompt template.
        """
        fallback = _human_prompt(doc, task_family)
        context = (task_context or "").strip()
        prompt_text = context if context else fallback

        if task_family == "scan2cap":
            prompt_text = (doc.get("native_prompt") or prompt_text).strip()

        prompt_text = prompt_text.replace(self._DEFAULT_IMAGE_TOKEN, " ").strip()
        if self.extra_prompt:
            extra = self.extra_prompt.strip()
            if extra and not prompt_text.startswith(extra):
                prompt_text = f"{extra}\n{prompt_text}"
        return f"{_prompt_prefix(task_family)}{prompt_text}".strip()

    def _generate_text(
        self,
        doc: dict,
        task_family: str,
        gen_kwargs: dict,
        task_context: Optional[str] = None,
        *,
        skip_llm_analysis: bool = False,
    ) -> GenerationResult:
        prepared = self._prepare_video_inputs(doc, task_family)
        if prepared is None:
            return GenerationResult(text="", token_counts=None, metadata=None)
        image_tensors, video_dict = prepared

        human = {"from": "human", "value": self._resolve_human_prompt(task_context, doc, task_family)}
        assistant = {"from": "gpt", "value": None}
        input_ids = self._preprocess_qwen([human, assistant], has_image=True)

        temperature = float(gen_kwargs.get("temperature", self.default_temperature))
        top_p = gen_kwargs.get("top_p", self.default_top_p)
        num_beams = int(gen_kwargs.get("num_beams", self.default_num_beams))
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 512))

        conv = self._conv_templates[self._conv_mode]
        stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
        if hasattr(self.model, "reset_last_compression_profile"):
            self.model.reset_last_compression_profile()
        streamer = LatencyStreamer(device=self.device)
        streamer.start()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                modalities="video",
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                video_dict=video_dict,
                streamer=streamer,
                skip_llm_analysis=skip_llm_analysis,
            )
        streamer.end()

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        compression_profile = {}
        if hasattr(self.model, "pop_last_compression_profile"):
            compression_profile = _normalize_compression_profile(
                self.model.pop_last_compression_profile()
            )

        latency_metrics = streamer.get_metrics() or {}
        generated_tokens = latency_metrics.get("generated_tokens")
        if generated_tokens is None:
            generated_tokens = max(int(output_ids.shape[1] - input_ids.shape[1]), 0)

        prompt_tokens = int(compression_profile.get("prompt_sequence_length", input_ids.shape[1]))
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

        return GenerationResult(
            text=outputs,
            token_counts=TokenCounts(
                input_tokens=prompt_tokens,
                output_tokens=int(generated_tokens),
            ),
            metadata={"compression_efficiency": compression_efficiency},
        )

    def _ground_single(self, doc: dict, task_context: Optional[str] = None) -> GenerationResult:
        self._ensure_grounding_supported("scanrefer")
        image_tensors, video_dict = self._prepare_video_inputs(doc, "scanrefer")
        human = {"from": "human", "value": self._resolve_human_prompt(task_context, doc, "scanrefer")}
        assistant = {"from": "gpt", "value": _assistant_target(doc, "scanrefer")}
        input_ids, labels = self._preprocess_qwen([human, assistant], has_image=True, include_labels=True)

        if hasattr(self.model, "reset_last_compression_profile"):
            self.model.reset_last_compression_profile()
        start_time = time.perf_counter()
        with torch.inference_mode():
            _, scores = self.model(
                input_ids,
                images=image_tensors,
                modalities="video",
                video_dict=video_dict,
                labels=labels,
                use_object_proposals=True,
                box_labels=None,
            )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        scores = scores.reshape(-1)
        objects = video_dict["objects"][0]
        if len(objects) == 0:
            return self._build_grounding_result(text="[]", input_ids=input_ids, forward_wall_ms=elapsed_ms)
        pred_index = int(torch.argmax(scores).item())
        if pred_index >= len(objects):
            pred_index = int(torch.argmax(scores[:-1]).item())
        return self._build_grounding_result(
            text=json.dumps([float(value) for value in objects[pred_index].tolist()]),
            input_ids=input_ids,
            forward_wall_ms=elapsed_ms,
        )

    def _ground_multi(self, doc: dict, gen_kwargs: dict, task_context: Optional[str] = None) -> GenerationResult:
        self._ensure_grounding_supported("multi3drefer")
        image_tensors, video_dict = self._prepare_video_inputs(doc, "multi3drefer")
        human = {"from": "human", "value": self._resolve_human_prompt(task_context, doc, "multi3drefer")}
        assistant = {"from": "gpt", "value": _assistant_target(doc, "multi3drefer")}
        input_ids, labels = self._preprocess_qwen([human, assistant], has_image=True, include_labels=True)

        if hasattr(self.model, "reset_last_compression_profile"):
            self.model.reset_last_compression_profile()
        start_time = time.perf_counter()
        with torch.inference_mode():
            _, scores = self.model(
                input_ids,
                images=image_tensors,
                modalities="video",
                video_dict=video_dict,
                labels=labels,
                use_object_proposals=True,
                box_labels=None,
            )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        scores = scores.reshape(-1)
        objects = video_dict["objects"][0]
        if len(objects) == 0 or len(scores) == 0:
            return self._build_grounding_result(text="[]", input_ids=input_ids, forward_wall_ms=elapsed_ms)

        pred_boxes = []
        null_index = len(scores) - 1
        if int(torch.argmax(scores).item()) != null_index:
            threshold = float(gen_kwargs.get("selection_threshold", self.grounding_threshold))
            probs = torch.nn.functional.softmax(scores / 0.07, dim=0)[:-1]
            sorted_scores, indices = torch.sort(probs, descending=True)
            running = 0.0
            for index, score in zip(indices.tolist(), sorted_scores.tolist()):
                pred_boxes.append([float(value) for value in objects[index].tolist()])
                running += float(score)
                if running >= threshold:
                    break
        return self._build_grounding_result(
            text=json.dumps(pred_boxes),
            input_ids=input_ids,
            forward_wall_ms=elapsed_ms,
        )

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("video_3d does not implement loglikelihood.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs: List[str] = []
        show_all_rank_progress = _as_bool(os.environ.get("LMMS_SHOW_ALL_RANK_PROGRESS", False))
        iterator = tqdm(
            requests,
            disable=(not show_all_rank_progress and self.rank != 0),
            desc=f"Model Responding[r{self.rank}]",
        )
        for request in iterator:
            context, gen_kwargs, doc_to_visual, doc_id, task_name, split = request.args
            padding_only = bool(request.metadata.get("__padding_only__", False))
            del doc_to_visual
            doc = self.task_dict[task_name][split][doc_id]
            task_family = _resolve_task_family(task_name)

            llm_compressor = self._get_llm_compressor()
            if (
                llm_compressor is not None
                and hasattr(llm_compressor, "set_next_sample_key")
                and llm_compressor.should_analyze()
                and not padding_only
            ):
                llm_compressor.set_next_sample_key(f"{task_name}:{split}:{doc_id}")

            if task_family in {"scanqa", "sqa3d", "scan2cap"}:
                response = self._generate_text(
                    doc,
                    task_family,
                    gen_kwargs,
                    task_context=context,
                    skip_llm_analysis=padding_only,
                )
            elif task_family == "scanrefer":
                response = self._ground_single(doc, task_context=context)
            elif task_family == "multi3drefer":
                response = self._ground_multi(doc, gen_kwargs, task_context=context)
            else:
                raise ValueError(f"Unsupported Video-3D-LLM task family: {task_family}")

            outputs.append(response)
            if hasattr(self, "add_request_response_to_cache"):
                self.add_request_response_to_cache(request, response.text if isinstance(response, GenerationResult) else response)
        llm_compressor = self._get_llm_compressor()
        if llm_compressor is not None and hasattr(llm_compressor, "finalize"):
            llm_compressor.finalize()
        return outputs

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("video_3d does not implement multi-round generation.")
