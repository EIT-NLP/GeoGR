import copy
import inspect
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

try:
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_causal_attention_mask,
        _prepare_4d_causal_attention_mask_for_sdpa,
    )
except Exception:  # pragma: no cover - depends on transformers internals
    _prepare_4d_causal_attention_mask = None
    _prepare_4d_causal_attention_mask_for_sdpa = None


_SHADOW_DECODER_CACHE: Dict[Tuple[Any, ...], nn.Module] = {}
_FLOPS_CACHE: Dict[Tuple[Any, ...], float] = {}
_CALFLOPS_FUNC = None
_CALFLOPS_IMPORT_ERROR: Optional[str] = None


def _get_config_value(config: Any, key: str, default: int) -> int:
    value = getattr(config, key, default)
    if value is None:
        return int(default)
    return int(value)


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _coerce_torch_dtype(value: Any) -> Optional[torch.dtype]:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.replace("torch.", "").strip()
        dtype = getattr(torch, normalized, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    return None


def _get_dtype_nbytes(dtype: torch.dtype) -> int:
    try:
        return int(torch.tensor([], dtype=dtype).element_size())
    except Exception:
        return 2


def _coerce_int_sequence(value: Any) -> Optional[list[int]]:
    if not isinstance(value, (list, tuple)):
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        result.append(max(int(item), 0))
    return result


def _as_tflops(value: float) -> float:
    return float(value) / 1e12


def _bytes_to_mb(value: int) -> float:
    return float(value) / float(1024 * 1024)


def _load_calflops():
    global _CALFLOPS_FUNC, _CALFLOPS_IMPORT_ERROR
    if _CALFLOPS_FUNC is not None:
        return _CALFLOPS_FUNC
    if _CALFLOPS_IMPORT_ERROR is not None:
        return None

    try:
        from calflops import calculate_flops
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        _CALFLOPS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None

    _CALFLOPS_FUNC = calculate_flops
    return _CALFLOPS_FUNC


def _resolve_llm_decoder_model(model: Optional[nn.Module]) -> Optional[nn.Module]:
    if model is None or not isinstance(model, nn.Module):
        return None

    decoder = model
    if hasattr(decoder, "get_model"):
        maybe_decoder = decoder.get_model()
        if isinstance(maybe_decoder, nn.Module):
            decoder = maybe_decoder
    elif hasattr(decoder, "get_decoder"):
        maybe_decoder = decoder.get_decoder()
        if isinstance(maybe_decoder, nn.Module):
            decoder = maybe_decoder
    elif hasattr(decoder, "model") and isinstance(decoder.model, nn.Module):
        decoder = decoder.model

    required_attrs = ("layers", "norm", "config")
    if not all(hasattr(decoder, attr) for attr in required_attrs):
        return None
    if not hasattr(decoder, "_prepare_decoder_attention_mask") and _prepare_4d_causal_attention_mask is None:
        return None
    return decoder


def _resolve_module_floating_dtype(module: Optional[nn.Module]) -> Optional[torch.dtype]:
    if module is None:
        return None
    for tensor in module.parameters():
        if torch.is_floating_point(tensor):
            return tensor.dtype
    for tensor in module.buffers():
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return None


def _resolve_llm_cache_dtype(config: Any, model: Optional[nn.Module]) -> torch.dtype:
    decoder = _resolve_llm_decoder_model(model)
    decoder_dtype = _resolve_module_floating_dtype(decoder)
    if decoder_dtype is not None:
        return decoder_dtype

    if decoder is not None:
        decoder_config_dtype = _coerce_torch_dtype(getattr(decoder.config, "torch_dtype", None))
        if decoder_config_dtype is not None:
            return decoder_config_dtype

    model_dtype = _coerce_torch_dtype(getattr(model, "dtype", None))
    if model_dtype is not None:
        return model_dtype

    config_dtype = _coerce_torch_dtype(getattr(config, "torch_dtype", None))
    if config_dtype is not None:
        return config_dtype

    return torch.float16


def _resolve_two_stage_llm_plan(
    num_hidden_layers: int,
    compression_efficiency: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[int, int, int]]:
    llm_prune_layer = _coerce_int((compression_efficiency or {}).get("llm_prune_layer"))
    prompt_tokens_before_prune = _coerce_int((compression_efficiency or {}).get("prompt_sequence_length_before_prune"))
    prompt_tokens_after_prune = _coerce_int((compression_efficiency or {}).get("prompt_sequence_length"))

    use_two_stage_llm_plan = (
        llm_prune_layer is not None
        and prompt_tokens_before_prune is not None
        and prompt_tokens_after_prune is not None
        and 0 <= llm_prune_layer <= num_hidden_layers
        and prompt_tokens_before_prune >= prompt_tokens_after_prune > 0
    )
    if not use_two_stage_llm_plan:
        return None
    return (
        int(llm_prune_layer),
        int(prompt_tokens_before_prune),
        int(prompt_tokens_after_prune),
    )


def _resolve_prefill_layer_token_lengths(
    num_hidden_layers: int,
    prompt_tokens: int,
    compression_efficiency: Optional[Dict[str, Any]] = None,
) -> list[int]:
    explicit_layer_lengths = _coerce_int_sequence((compression_efficiency or {}).get("prefill_layer_token_lengths"))
    if explicit_layer_lengths is not None:
        if len(explicit_layer_lengths) == num_hidden_layers:
            return explicit_layer_lengths
        if len(explicit_layer_lengths) < num_hidden_layers:
            if explicit_layer_lengths:
                return explicit_layer_lengths + [explicit_layer_lengths[-1]] * (num_hidden_layers - len(explicit_layer_lengths))
            return [max(int(prompt_tokens), 0)] * num_hidden_layers
        return explicit_layer_lengths[:num_hidden_layers]

    two_stage_plan = _resolve_two_stage_llm_plan(
        num_hidden_layers=num_hidden_layers,
        compression_efficiency=compression_efficiency,
    )
    if two_stage_plan is not None:
        llm_prune_layer, prompt_tokens_before_prune, prompt_tokens_after_prune = two_stage_plan
        return [prompt_tokens_before_prune] * llm_prune_layer + [prompt_tokens_after_prune] * (num_hidden_layers - llm_prune_layer)

    return [max(int(prompt_tokens), 0)] * num_hidden_layers


def _segment_layer_token_lengths(layer_token_lengths: list[int]) -> list[Tuple[int, int, int]]:
    if not layer_token_lengths:
        return []
    segments: list[Tuple[int, int, int]] = []
    start_idx = 0
    current_length = int(layer_token_lengths[0])
    for layer_idx in range(1, len(layer_token_lengths)):
        layer_length = int(layer_token_lengths[layer_idx])
        if layer_length == current_length:
            continue
        segments.append((start_idx, layer_idx, current_length))
        start_idx = layer_idx
        current_length = layer_length
    segments.append((start_idx, len(layer_token_lengths), current_length))
    return segments


def _resolve_decode_position_start(
    prompt_tokens: int,
    compression_efficiency: Optional[Dict[str, Any]] = None,
    layer_token_lengths: Optional[list[int]] = None,
) -> int:
    prompt_tokens_before_prune = _coerce_int((compression_efficiency or {}).get("prompt_sequence_length_before_prune"))
    if prompt_tokens_before_prune is not None:
        return int(prompt_tokens_before_prune)
    if layer_token_lengths:
        return int(max(layer_token_lengths))
    return max(int(prompt_tokens), 0)


def _estimate_prefill_kv_cache_size(
    config: Any,
    prompt_tokens: int,
    compression_efficiency: Optional[Dict[str, Any]] = None,
    model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    prompt_tokens = max(int(prompt_tokens), 0)
    decoder = _resolve_llm_decoder_model(model)
    llm_config = decoder.config if decoder is not None else config

    num_hidden_layers = _get_config_value(llm_config, "num_hidden_layers", 0)
    num_attention_heads = _get_config_value(llm_config, "num_attention_heads", 1)
    num_key_value_heads = _get_config_value(llm_config, "num_key_value_heads", num_attention_heads)
    hidden_size = _get_config_value(llm_config, "hidden_size", 0)
    head_dim = hidden_size // max(num_attention_heads, 1)
    if (
        prompt_tokens <= 0
        or num_hidden_layers <= 0
        or num_attention_heads <= 0
        or num_key_value_heads <= 0
        or hidden_size <= 0
        or head_dim <= 0
    ):
        return {
            "kv_cache_mb": 0.0,
        }

    dtype = _resolve_llm_cache_dtype(llm_config, model)
    dtype_nbytes = _get_dtype_nbytes(dtype)
    layer_token_lengths = _resolve_prefill_layer_token_lengths(
        num_hidden_layers=num_hidden_layers,
        prompt_tokens=prompt_tokens,
        compression_efficiency=compression_efficiency,
    )
    total_layer_tokens = int(sum(layer_token_lengths))
    total_cache_elements = 2 * int(num_key_value_heads) * int(head_dim) * total_layer_tokens
    total_cache_bytes = int(total_cache_elements * dtype_nbytes)
    return {
        "kv_cache_mb": _bytes_to_mb(total_cache_bytes),
    }


def _delete_attr_if_present(obj: Any, attr_name: str) -> None:
    if not hasattr(obj, attr_name):
        return
    try:
        delattr(obj, attr_name)
    except Exception:
        pass


def _make_decoder_cache_key(decoder: nn.Module) -> Tuple[Any, ...]:
    config = decoder.config
    return (
        type(decoder).__name__,
        getattr(config, "model_type", ""),
        _get_config_value(config, "hidden_size", 0),
        _get_config_value(config, "intermediate_size", 0),
        _get_config_value(config, "num_hidden_layers", 0),
        _get_config_value(config, "num_attention_heads", 0),
        _get_config_value(config, "num_key_value_heads", _get_config_value(config, "num_attention_heads", 0)),
        _get_config_value(config, "vocab_size", 0),
        getattr(config, "_attn_implementation", ""),
    )


def _build_shadow_decoder_model(decoder: nn.Module) -> Optional[nn.Module]:
    cache_key = _make_decoder_cache_key(decoder)
    if cache_key in _SHADOW_DECODER_CACHE:
        return _SHADOW_DECODER_CACHE[cache_key]

    shadow_config = copy.deepcopy(decoder.config)
    if hasattr(shadow_config, "_attn_implementation"):
        shadow_config._attn_implementation = "eager"

    # Avoid constructing the multimodal stack for the shadow decoder.
    for attr_name in (
        "mm_vision_tower",
        "mm_hidden_size",
        "vision_tower_pretrained",
        "world_position_embedding_type",
    ):
        _delete_attr_if_present(shadow_config, attr_name)

    try:
        with torch.device("meta"):
            shadow_decoder = type(decoder)(shadow_config)
        shadow_decoder.eval()
    except Exception:
        return None

    _SHADOW_DECODER_CACHE[cache_key] = shadow_decoder
    return shadow_decoder


def _resolve_position_ids(decoder: nn.Module, seq_length: int, *, start_index: int, device: torch.device) -> torch.Tensor:
    model_name = type(decoder).__name__.lower()
    model_type = str(getattr(decoder.config, "model_type", "") or "").lower()
    base_positions = torch.arange(start_index, start_index + seq_length, dtype=torch.long, device=device)
    if ("qwen" in model_name or "qwen" in model_type) and _decoder_uses_3d_position_ids(decoder):
        return base_positions.view(1, seq_length, 1).repeat(1, 1, 3)
    return base_positions.view(1, seq_length)


def _build_dynamic_past_key_values(decoder: nn.Module, *, batch_size: int, cache_seq_len: int, dtype: torch.dtype, device: torch.device):
    if cache_seq_len <= 0:
        return None

    num_attention_heads = _get_config_value(decoder.config, "num_attention_heads", 1)
    num_key_value_heads = _get_config_value(decoder.config, "num_key_value_heads", num_attention_heads)
    hidden_size = _get_config_value(decoder.config, "hidden_size", 0)
    head_dim = hidden_size // max(num_attention_heads, 1)
    layer_count = max(len(getattr(decoder, "layers", [])), 1)

    legacy_cache = []
    for _ in range(layer_count):
        key_states = torch.zeros((batch_size, num_key_value_heads, cache_seq_len, head_dim), dtype=dtype, device=device)
        value_states = torch.zeros((batch_size, num_key_value_heads, cache_seq_len, head_dim), dtype=dtype, device=device)
        legacy_cache.append((key_states, value_states))
    return DynamicCache.from_legacy_cache(tuple(legacy_cache))


def _decoder_uses_3d_position_ids(decoder: nn.Module) -> bool:
    try:
        first_layer = decoder.layers[0]
        rotary_emb = first_layer.self_attn.rotary_emb
        signature = inspect.signature(rotary_emb.forward)
    except Exception:
        return False
    return "position_ids" in signature.parameters


def _prepare_decoder_attention_mask(
    decoder: nn.Module,
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    seq_length: int,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    output_attentions: bool,
    use_cache: bool,
) -> Optional[torch.Tensor]:
    if hasattr(decoder, "_prepare_decoder_attention_mask"):
        return decoder._prepare_decoder_attention_mask(
            attention_mask,
            batch_size=batch_size,
            seq_length=seq_length,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    attn_impl = str(getattr(decoder, "_attn_implementation", getattr(decoder.config, "_attn_implementation", "")) or "")
    if attn_impl == "flash_attention_2":
        return attention_mask if (attention_mask is not None and 0 in attention_mask) else None

    if attn_impl == "sdpa" and not output_attentions and _prepare_4d_causal_attention_mask_for_sdpa is not None:
        return _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=getattr(decoder.config, "sliding_window", None),
        )

    if _prepare_4d_causal_attention_mask is None:
        raise RuntimeError("Unable to build a causal attention mask for this decoder.")
    return _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
        sliding_window=getattr(decoder.config, "sliding_window", None),
    )


class _PrefillDecoderSliceWrapper(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        *,
        start_layer: int,
        end_layer: int,
        seq_len: int,
        include_norm: bool,
    ):
        super().__init__()
        self.decoder = decoder
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.seq_len = int(seq_len)
        self.include_norm = bool(include_norm)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch_size = int(inputs_embeds.shape[0])
        device = inputs_embeds.device
        hidden_states = inputs_embeds
        attention_mask_2d = torch.ones((batch_size, self.seq_len), dtype=torch.bool, device=device)
        position_ids = _resolve_position_ids(self.decoder, self.seq_len, start_index=0, device=device)
        causal_mask = _prepare_decoder_attention_mask(
            self.decoder,
            attention_mask_2d,
            batch_size=batch_size,
            seq_length=self.seq_len,
            inputs_embeds=hidden_states,
            past_key_values_length=0,
            output_attentions=False,
            use_cache=False,
        )

        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states = self.decoder.layers[layer_idx](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )[0]

        if self.include_norm:
            hidden_states = self.decoder.norm(hidden_states)
        return hidden_states


class _DecodeDecoderSliceWrapper(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        *,
        start_layer: int,
        end_layer: int,
        cache_seq_len: int,
        position_start: int,
        include_norm: bool,
    ):
        super().__init__()
        self.decoder = decoder
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.cache_seq_len = int(cache_seq_len)
        self.position_start = int(position_start)
        self.include_norm = bool(include_norm)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, current_seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        hidden_states = inputs_embeds
        past_key_values = _build_dynamic_past_key_values(
            self.decoder,
            batch_size=batch_size,
            cache_seq_len=self.cache_seq_len,
            dtype=hidden_states.dtype,
            device=device,
        )
        attention_mask_2d = torch.ones((batch_size, self.cache_seq_len + current_seq_len), dtype=torch.bool, device=device)
        position_ids = _resolve_position_ids(
            self.decoder,
            current_seq_len,
            start_index=self.position_start,
            device=device,
        )
        causal_mask = _prepare_decoder_attention_mask(
            self.decoder,
            attention_mask_2d,
            batch_size=batch_size,
            seq_length=current_seq_len,
            inputs_embeds=hidden_states,
            past_key_values_length=self.cache_seq_len,
            output_attentions=False,
            use_cache=True,
        )

        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states = self.decoder.layers[layer_idx](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
            )[0]

        if self.include_norm:
            hidden_states = self.decoder.norm(hidden_states)
        return hidden_states


def _profile_wrapper_flops(wrapper: nn.Module, *, input_shape: Tuple[int, ...], cache_key: Tuple[Any, ...]) -> float:
    if cache_key in _FLOPS_CACHE:
        return _FLOPS_CACHE[cache_key]

    calculate_flops = _load_calflops()
    if calculate_flops is None:
        raise RuntimeError(_CALFLOPS_IMPORT_ERROR or "calflops is unavailable")

    wrapper.eval()
    try:
        first_param = next(wrapper.parameters())
        device = first_param.device
        dtype = first_param.dtype if first_param.is_floating_point() else torch.float32
    except StopIteration:
        device = torch.device("cpu")
        dtype = torch.float32

    inputs_embeds = torch.zeros(input_shape, device=device, dtype=dtype)
    with torch.inference_mode():
        flops, _, _ = calculate_flops(
            model=wrapper,
            args=[inputs_embeds],
            print_results=False,
            print_detailed=False,
            output_as_string=False,
        )
    flops = float(flops)
    _FLOPS_CACHE[cache_key] = flops
    return flops


def _profile_prefill_slice(
    decoder: nn.Module,
    model_key: Tuple[Any, ...],
    *,
    start_layer: int,
    end_layer: int,
    seq_len: int,
    include_norm: bool,
) -> float:
    wrapper = _PrefillDecoderSliceWrapper(
        decoder,
        start_layer=start_layer,
        end_layer=end_layer,
        seq_len=seq_len,
        include_norm=include_norm,
    )
    return _profile_wrapper_flops(
        wrapper,
        input_shape=(1, int(seq_len), _get_config_value(decoder.config, "hidden_size", 0)),
        cache_key=(model_key, "prefill", int(start_layer), int(end_layer), int(seq_len), bool(include_norm)),
    )


def _profile_decode_slice(
    decoder: nn.Module,
    model_key: Tuple[Any, ...],
    *,
    start_layer: int,
    end_layer: int,
    cache_seq_len: int,
    position_start: int,
    include_norm: bool,
) -> float:
    wrapper = _DecodeDecoderSliceWrapper(
        decoder,
        start_layer=start_layer,
        end_layer=end_layer,
        cache_seq_len=cache_seq_len,
        position_start=position_start,
        include_norm=include_norm,
    )
    hidden_size = _get_config_value(decoder.config, "hidden_size", 0)
    return _profile_wrapper_flops(
        wrapper,
        input_shape=(1, 1, hidden_size),
        cache_key=(
            model_key,
            "decode",
            int(start_layer),
            int(end_layer),
            int(cache_seq_len),
            int(position_start),
            bool(include_norm),
        ),
    )


def _build_tflops_result(total_flops: float) -> Dict[str, float]:
    return {
        "tflops": _as_tflops(total_flops),
    }


def _measure_generation_flops_with_calflops(
    model: nn.Module,
    prompt_tokens: int,
    generated_tokens: int,
    compression_efficiency: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    decoder = _resolve_llm_decoder_model(model)
    if decoder is None:
        raise RuntimeError("Unable to resolve an LLM decoder from the provided model.")

    shadow_decoder = _build_shadow_decoder_model(decoder)
    if shadow_decoder is None:
        raise RuntimeError("Unable to build a meta-device shadow decoder for calflops profiling.")

    num_hidden_layers = _get_config_value(shadow_decoder.config, "num_hidden_layers", 0)
    if num_hidden_layers <= 0 or prompt_tokens <= 0:
        return _build_tflops_result(0.0)

    model_key = _make_decoder_cache_key(shadow_decoder)
    decode_steps = max(int(generated_tokens) - 1, 0)
    layer_token_lengths = _resolve_prefill_layer_token_lengths(
        num_hidden_layers=num_hidden_layers,
        prompt_tokens=prompt_tokens,
        compression_efficiency=compression_efficiency,
    )
    segments = _segment_layer_token_lengths(layer_token_lengths)
    decode_position_start = _resolve_decode_position_start(
        prompt_tokens=prompt_tokens,
        compression_efficiency=compression_efficiency,
        layer_token_lengths=layer_token_lengths,
    )

    if len(segments) <= 1:
        prefill_flops = _profile_prefill_slice(
            shadow_decoder,
            model_key,
            start_layer=0,
            end_layer=num_hidden_layers,
            seq_len=layer_token_lengths[0],
            include_norm=True,
        )
        decode_flops = 0.0
        for step_idx in range(decode_steps):
            decode_flops += _profile_decode_slice(
                shadow_decoder,
                model_key,
                start_layer=0,
                end_layer=num_hidden_layers,
                cache_seq_len=layer_token_lengths[0] + step_idx,
                position_start=decode_position_start + step_idx,
                include_norm=True,
            )
        return _build_tflops_result(prefill_flops + decode_flops)

    prefill_flops = 0.0
    for start_layer, end_layer, seq_len in segments:
        prefill_flops += _profile_prefill_slice(
            shadow_decoder,
            model_key,
            start_layer=start_layer,
            end_layer=end_layer,
            seq_len=seq_len,
            include_norm=(end_layer == num_hidden_layers),
        )

    decode_flops = 0.0
    for step_idx in range(decode_steps):
        for start_layer, end_layer, seq_len in segments:
            decode_flops += _profile_decode_slice(
                shadow_decoder,
                model_key,
                start_layer=start_layer,
                end_layer=end_layer,
                cache_seq_len=seq_len + step_idx,
                position_start=decode_position_start + step_idx,
                include_norm=(end_layer == num_hidden_layers),
            )

    return _build_tflops_result(prefill_flops + decode_flops)


def estimate_generation_flops(
    config: Any,
    prompt_tokens: int,
    generated_tokens: int,
    compression_efficiency: Optional[Dict[str, Any]] = None,
    model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    prompt_tokens = max(int(prompt_tokens), 0)
    generated_tokens = max(int(generated_tokens), 0)
    kv_cache_metrics = _estimate_prefill_kv_cache_size(
        config,
        prompt_tokens=prompt_tokens,
        compression_efficiency=compression_efficiency,
        model=model,
    )

    if prompt_tokens <= 0:
        result = _build_tflops_result(0.0)
        result.update(kv_cache_metrics)
        return result

    if model is None:
        raise RuntimeError("calflops-based LLM FLOPs estimation requires the runtime model instance.")

    result = _measure_generation_flops_with_calflops(
        model=model,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        compression_efficiency=compression_efficiency,
    )
    result.update(kv_cache_metrics)
    return result
