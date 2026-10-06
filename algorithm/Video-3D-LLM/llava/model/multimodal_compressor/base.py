from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class CompressorOutput:
    features: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_ratio: float = 1.0


@dataclass
class CompressorConfig:
    enabled: bool = True
    position: str = "after_projector"
    return_stats: bool = False
    save_debug_info: bool = False


@dataclass
class LLMForwardContext:
    hidden_states: torch.Tensor
    causal_mask: Optional[torch.Tensor] = None
    attention_mask_2d: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    past_key_values: Any = None
    cache_position: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    llm_metadata: Optional[Dict[str, Any]] = None
    use_cache: bool = False
    layers: Any = None
    output_hidden_states: bool = False
    output_attentions: bool = False
    gradient_checkpointing: bool = False
    gradient_checkpointing_func: Optional[Callable] = None
    norm_fn: Optional[Callable] = None
    gather_and_pad_fn: Optional[Callable] = None
    prepare_decoder_attention_mask_fn: Optional[Callable] = None


@dataclass
class LLMForwardOutput:
    hidden_states: torch.Tensor
    attention_mask_2d: Optional[torch.Tensor] = None
    past_key_values: Any = None
    all_hidden_states: Optional[Tuple] = None
    all_self_attns: Optional[Tuple] = None
    labels: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCompressor(nn.Module, ABC):
    def __init__(self, config: CompressorConfig):
        super().__init__()
        self.config = config
        self._last_compression_stats: Dict[str, Any] = {}

    @abstractmethod
    def compress(
        self,
        features: torch.Tensor,
        num_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> CompressorOutput:
        raise NotImplementedError

    @abstractmethod
    def supports_video(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def supports_image(self) -> bool:
        raise NotImplementedError

    def get_compression_info(self) -> Dict[str, Any]:
        return self._last_compression_stats.copy()

    def get_required_inputs(self) -> Dict[str, bool]:
        return {
            "attn_weights": False,
            "raw_features_before_proj": False,
            "coordinates": False,
        }

    def has_llm_compress(self) -> bool:
        return bool(self.config.enabled) and self.config.position == "llm"

    def get_llm_prune_layer(self) -> Optional[int]:
        return None

    def select_llm_keep_positions(
        self,
        hidden_states: torch.Tensor,
        llm_metadata: Dict[str, Any],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement llm-level compression.")

    def llm_compress(self, ctx: LLMForwardContext) -> LLMForwardOutput:
        return self.default_decoder_loop(ctx)

    def forward(self, features: torch.Tensor, **kwargs) -> CompressorOutput:
        if not self.config.enabled:
            return CompressorOutput(features=features, compression_ratio=1.0)
        return self.compress(features, **kwargs)

    def _update_stats(self, input_tokens: int, output_tokens: int, **extra):
        self._last_compression_stats = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "compression_ratio": output_tokens / input_tokens if input_tokens > 0 else 1.0,
            "tokens_removed": input_tokens - output_tokens,
            **extra,
        }

    def default_decoder_loop(self, ctx: LLMForwardContext) -> LLMForwardOutput:
        hidden_states = ctx.hidden_states
        causal_mask = ctx.causal_mask
        position_ids = ctx.position_ids
        past_key_values = ctx.past_key_values
        cache_position = ctx.cache_position
        use_cache = ctx.use_cache
        layers = ctx.layers

        all_hidden_states = () if ctx.output_hidden_states else None
        all_self_attns = () if ctx.output_attentions else None
        next_decoder_cache = None

        for decoder_layer in layers:
            if ctx.output_hidden_states:
                all_hidden_states += (hidden_states,)

            if ctx.gradient_checkpointing and ctx.gradient_checkpointing_func is not None:
                layer_outputs = ctx.gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    ctx.output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=ctx.output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if ctx.output_attentions else 1]

            if ctx.output_attentions:
                all_self_attns += (layer_outputs[1],)

        if ctx.norm_fn is not None:
            hidden_states = ctx.norm_fn(hidden_states)

        if ctx.output_hidden_states:
            all_hidden_states += (hidden_states,)

        return LLMForwardOutput(
            hidden_states=hidden_states,
            attention_mask_2d=ctx.attention_mask_2d,
            past_key_values=next_decoder_cache,
            all_hidden_states=all_hidden_states,
            all_self_attns=all_self_attns,
            labels=ctx.labels,
        )


class IdentityCompressor(BaseCompressor):
    def __init__(self, config: Optional[CompressorConfig] = None):
        super().__init__(config or CompressorConfig())

    def compress(self, features: torch.Tensor, **kwargs) -> CompressorOutput:
        if features.dim() == 2:
            input_tokens = output_tokens = features.shape[0]
        else:
            input_tokens = output_tokens = features.shape[-2]
        self._update_stats(input_tokens, output_tokens)
        return CompressorOutput(
            features=features,
            compression_ratio=1.0,
            metadata={"compressor": "identity"},
        )

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return True

    def select_llm_keep_positions(
        self,
        hidden_states: torch.Tensor,
        llm_metadata: Dict[str, Any],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size, seq_length = hidden_states.shape[:2]
        device = hidden_states.device
        keep_positions: List[torch.Tensor] = []
        for batch_idx in range(batch_size):
            if attention_mask is None:
                keep_positions.append(torch.arange(seq_length, device=device, dtype=torch.long))
            else:
                keep_positions.append(torch.nonzero(attention_mask[batch_idx], as_tuple=False).flatten().to(device=device))
        return {
            "keep_positions": keep_positions,
            "profiles": [],
        }
