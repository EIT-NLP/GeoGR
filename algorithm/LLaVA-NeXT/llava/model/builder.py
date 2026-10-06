"""Qwen2/LLaVA model loader used by the public GeoGR release.

The repository evaluates one multimodal backbone: LLaVA-OneVision with a
Qwen2 language model. The loader intentionally keeps only the loading paths
needed by the project while preserving the upstream checkpoint formats:

* a complete multimodal checkpoint;
* a projector-only checkpoint loaded on top of ``model_base``;
* an unmerged PEFT/LoRA checkpoint loaded on top of ``model_base``; and
* optional 4-bit or 8-bit loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from llava.utils import rank0_print


def _dtype_kwargs(load_8bit: bool, load_4bit: bool, torch_dtype: str) -> dict[str, Any]:
    if load_8bit:
        return {"load_in_8bit": True}
    if load_4bit:
        return {
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        }
    if torch_dtype == "float16":
        return {"torch_dtype": torch.float16}
    if torch_dtype == "bfloat16":
        return {"torch_dtype": torch.bfloat16}
    if torch_dtype in {"auto", None}:
        return {}
    raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}; use float16, bfloat16, or auto.")


def _apply_overwrite_config(config: Any, overwrite_config: Optional[Mapping[str, Any]]) -> Any:
    if overwrite_config:
        rank0_print(f"Overwriting config with {dict(overwrite_config)}")
        for key, value in overwrite_config.items():
            setattr(config, key, value)
    return config


def _resolve_config(
    source: str,
    customized_config: Any = None,
    overwrite_config: Optional[Mapping[str, Any]] = None,
) -> LlavaQwenConfig:
    if customized_config is None:
        config = LlavaQwenConfig.from_pretrained(source)
    elif isinstance(customized_config, str):
        config = LlavaQwenConfig.from_pretrained(customized_config)
    elif isinstance(customized_config, Mapping):
        config = LlavaQwenConfig.from_dict(dict(customized_config))
    else:
        config = customized_config
    return _apply_overwrite_config(config, overwrite_config)


def _model_kwargs(
    device_map: str,
    load_8bit: bool,
    load_4bit: bool,
    torch_dtype: str,
    attn_implementation: Optional[str],
    extra_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    kwargs = dict(extra_kwargs)
    kwargs["device_map"] = device_map
    kwargs.update(_dtype_kwargs(load_8bit, load_4bit, torch_dtype))
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def _normalize_non_lora_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {
        (key[11:] if key.startswith("base_model.") else key): value
        for key, value in state_dict.items()
    }
    if any(key.startswith("model.model.") for key in normalized):
        normalized = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in normalized.items()
        }
    return normalized


def _load_non_lora_trainables(model: torch.nn.Module, model_path: str) -> None:
    state_path = Path(model_path) / "non_lora_trainables.bin"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"LoRA checkpoint {model_path!r} does not contain non_lora_trainables.bin. "
            "Use a merged checkpoint or provide a complete model_base."
        )
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(_normalize_non_lora_state_dict(state_dict), strict=False)


def _load_projector_weights(model: torch.nn.Module, model_path: str) -> None:
    projector_path = Path(model_path) / "mm_projector.bin"
    if not projector_path.is_file():
        raise FileNotFoundError(f"Projector checkpoint {model_path!r} does not contain mm_projector.bin.")
    state_dict = torch.load(projector_path, map_location="cpu")
    state_dict = {
        key: value.to(torch.float16) if torch.is_floating_point(value) else value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)


def _load_qwen_model(
    source: str,
    config: LlavaQwenConfig,
    kwargs: Mapping[str, Any],
) -> LlavaQwenForCausalLM:
    return LlavaQwenForCausalLM.from_pretrained(
        source,
        low_cpu_mem_usage=True,
        config=config,
        **dict(kwargs),
    )


def _load_multimodal_model(
    model_path: str,
    model_base: Optional[str],
    model_name: str,
    kwargs: Mapping[str, Any],
    customized_config: Any,
    overwrite_config: Optional[Mapping[str, Any]],
) -> tuple[AutoTokenizer, torch.nn.Module]:
    name = str(model_name).lower()
    is_lora = "lora" in name or (
        model_base is not None and (Path(model_path) / "adapter_config.json").is_file()
    )

    if is_lora:
        if model_base is None:
            raise ValueError(
                "An unmerged LoRA checkpoint requires model_base pointing to the complete Qwen2/LLaVA model."
            )
        config = _resolve_config(model_path, customized_config, overwrite_config)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        model = _load_qwen_model(model_base, config, kwargs)
        model.resize_token_embeddings(len(tokenizer))
        _load_non_lora_trainables(model, model_path)

        from peft import PeftModel

        rank0_print("Loading and merging LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
        return tokenizer, model

    if model_base is not None:
        config = _resolve_config(model_path, customized_config, overwrite_config)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        model = _load_qwen_model(model_base, config, kwargs)
        _load_projector_weights(model, model_path)
        return tokenizer, model

    config = _resolve_config(model_path, customized_config, overwrite_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = _load_qwen_model(model_path, config, kwargs)
    return tokenizer, model


def _finish_multimodal_setup(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device_map: str,
) -> tuple[AutoTokenizer, torch.nn.Module, Any, int]:
    mm_use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))
    mm_use_im_patch_token = bool(getattr(model.config, "mm_use_im_patch_token", True))
    special_tokens = []
    if mm_use_im_patch_token:
        special_tokens.append(DEFAULT_IMAGE_PATCH_TOKEN)
    if mm_use_im_start_end:
        special_tokens.extend([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    if special_tokens:
        tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    image_processor = None
    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            if not vision_tower.is_loaded:
                vision_tower.load_model(device_map=device_map)
            if device_map != "auto":
                vision_device = torch.device(
                    device_map if str(device_map).startswith("cuda") else "cuda"
                )
                vision_tower.to(device=vision_device, dtype=torch.float16)
            image_processor = vision_tower.image_processor

    config = model.config
    if hasattr(config, "max_sequence_length"):
        context_len = int(config.max_sequence_length)
    elif hasattr(config, "max_position_embeddings"):
        context_len = int(config.max_position_embeddings)
    elif hasattr(config, "tokenizer_model_max_length"):
        context_len = int(config.tokenizer_model_max_length)
    else:
        context_len = 2048
    return tokenizer, model, image_processor, context_len


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    torch_dtype="float16",
    attn_implementation="flash_attention_2",
    customized_config=None,
    overwrite_config=None,
    **kwargs,
):
    """Load the Qwen2/LLaVA model used by LLaVA-OneVision experiments."""
    # ``multimodal`` is a legacy dispatch hint, not a Transformers argument.
    kwargs.pop("multimodal", None)
    kwargs.pop("use_flash_attention_2", None)
    model_kwargs = _model_kwargs(
        device_map=device_map,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        extra_kwargs=kwargs,
    )
    tokenizer, model = _load_multimodal_model(
        os.path.expanduser(model_path),
        os.path.expanduser(model_base) if model_base else None,
        model_name,
        model_kwargs,
        customized_config,
        overwrite_config,
    )
    rank0_print(f"Model Class: {model.__class__.__name__}")
    return _finish_multimodal_setup(tokenizer, model, device_map)
