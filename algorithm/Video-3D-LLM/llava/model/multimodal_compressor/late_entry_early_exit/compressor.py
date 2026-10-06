from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..base import BaseCompressor, CompressorConfig, CompressorOutput
from ..registry import register_compressor


@dataclass
class LateEntryEarlyExitConfig(CompressorConfig):
    position: str = "llm"
    # HiDrop-compatible one-based layer numbers. Visual tokens enter before
    # late_entry_layer and leave before early_exit_layer.
    late_entry_layer: int = 9
    early_exit_layer: int = 25


@register_compressor("late_entry_early_exit")
class LateEntryEarlyExitCompressor(BaseCompressor):
    """Shared physical visual-token late-entry and early-exit scheduler."""

    def __init__(self, config: Dict[str, Any]):
        cfg = LateEntryEarlyExitConfig(**config) if isinstance(config, dict) else config
        if cfg.position != "llm":
            raise ValueError(f"late_entry_early_exit requires position='llm', got {cfg.position!r}.")
        if int(cfg.late_entry_layer) < 1:
            raise ValueError("late_entry_layer must be at least 1.")
        if int(cfg.early_exit_layer) <= int(cfg.late_entry_layer):
            raise ValueError("early_exit_layer must be greater than late_entry_layer.")
        super().__init__(cfg)
        self.compression_config = cfg
        self._generation_state: Optional[Dict[str, Any]] = None
        self._last_profile: Optional[Dict[str, Any]] = None
        self._last_training_labels: Optional[torch.Tensor] = None

    def compress(self, features: torch.Tensor, **kwargs) -> CompressorOutput:
        raise NotImplementedError("late_entry_early_exit operates inside the LLM decoder.")

    def supports_video(self) -> bool:
        return True

    def supports_image(self) -> bool:
        return False

    def has_llm_compress(self) -> bool:
        return True

    def _expects_3d_position_ids(self, backbone) -> bool:
        """Model adapters override this decoder-interface capability."""
        return bool(getattr(backbone, "expects_3d_position_ids", False))

    def _make_decode_position_ids(
        self,
        *,
        backbone,
        position: int,
        device: torch.device,
    ) -> torch.Tensor:
        shape = (1, 1, 3) if self._expects_3d_position_ids(backbone) else (1, 1)
        return torch.full(shape, position, device=device, dtype=torch.long)

    def _next_decode_position(self, *, backbone, full_positions: torch.Tensor, state: Dict[str, Any]) -> int:
        """Return the first generated text position for this model interface."""
        del backbone, state
        return int(full_positions.max().item()) + 1

    def prepare_generation_compression(
        self,
        *,
        original_input_ids: torch.Tensor,
        original_attention_mask: Optional[torch.Tensor],
        expanded_attention_mask: Optional[torch.Tensor],
        expanded_seq_len: int,
        image_token_index: int,
    ) -> None:
        if original_input_ids.dim() != 2 or original_input_ids.shape[0] != 1:
            raise ValueError("LLaVA-OV late-entry/early-exit currently requires batch_size=1.")
        original_active = (
            original_attention_mask[0].to(device=original_input_ids.device, dtype=torch.bool)
            if original_attention_mask is not None
            else torch.ones(original_input_ids.shape[1], device=original_input_ids.device, dtype=torch.bool)
        )
        valid_ids = original_input_ids[0][original_active]
        placeholders = torch.nonzero(valid_ids == int(image_token_index), as_tuple=False).flatten()
        if placeholders.numel() != 1:
            raise ValueError(
                "late_entry_early_exit requires exactly one visual placeholder, "
                f"found {int(placeholders.numel())}."
            )
        if expanded_attention_mask is None:
            expanded_active = torch.arange(expanded_seq_len, device=original_input_ids.device)
        else:
            if expanded_attention_mask.shape != (1, expanded_seq_len):
                raise ValueError(
                    "expanded_attention_mask must have shape (1, expanded_seq_len), "
                    f"got {tuple(expanded_attention_mask.shape)}."
                )
            expanded_active = torch.nonzero(
                expanded_attention_mask[0].to(device=original_input_ids.device, dtype=torch.bool),
                as_tuple=False,
            ).flatten()
        visual_count = int(expanded_active.numel()) - (int(valid_ids.numel()) - 1)
        if visual_count <= 0:
            raise ValueError("Expanded prompt contains no visual tokens.")
        placeholder = int(placeholders[0].item())
        visual_start = placeholder
        visual_positions = expanded_active[visual_start : visual_start + visual_count]
        if int(visual_positions.numel()) != visual_count:
            raise ValueError("Expanded prompt does not contain the expected contiguous visual span.")
        visual_mask = torch.zeros(expanded_seq_len, device=original_input_ids.device, dtype=torch.bool)
        visual_mask[visual_positions] = True
        active_mask = torch.zeros(expanded_seq_len, device=original_input_ids.device, dtype=torch.bool)
        active_mask[expanded_active] = True
        self._generation_state = {
            "phase": "prefill",
            "visual_mask": visual_mask,
            "active_mask": active_mask,
            "visual_count": visual_count,
            "full_prompt_length": int(expanded_active.numel()),
            "decode_steps": 0,
        }
        self._last_profile = None
        self._last_training_labels = None

    def prepare_training_compression(self, **kwargs) -> None:
        self.prepare_generation_compression(**kwargs)
        if self._generation_state is not None:
            self._generation_state["mode"] = "training"

    def should_compress_generation_forward(self) -> bool:
        return self._generation_state is not None

    def consume_last_profile(self) -> Optional[Dict[str, Any]]:
        profile = self._last_profile
        self._last_profile = None
        return None if profile is None else profile.copy()

    def consume_last_training_labels(self) -> Optional[torch.Tensor]:
        labels = self._last_training_labels
        self._last_training_labels = None
        return labels

    def clear_generation_compression(self) -> None:
        """Drop per-request state so a later request cannot reuse this layout."""
        self._generation_state = None
        self._last_profile = None
        self._last_training_labels = None

    def _observe_prefill_layer(
        self,
        *,
        layer_idx: int,
        layer,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        visual_mask: Optional[torch.Tensor],
        full_position_span: int,
    ) -> None:
        """Optional read-only observer used by analysis-only subclasses."""
        return None

    @staticmethod
    def _prepare_mask(
        backbone,
        attention_mask_2d: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        past_length: int,
        output_attentions: bool,
        use_cache: bool,
    ) -> Optional[torch.Tensor]:
        implementation = getattr(backbone, "_attn_implementation", None) or getattr(
            backbone.config, "_attn_implementation", "eager"
        )
        batch_size, seq_len = hidden_states.shape[:2]
        if implementation == "flash_attention_2":
            return attention_mask_2d if attention_mask_2d is not None and 0 in attention_mask_2d else None
        if implementation == "sdpa" and not output_attentions:
            return _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask_2d,
                (batch_size, seq_len),
                hidden_states,
                past_length,
            )
        return _prepare_4d_causal_attention_mask(
            attention_mask_2d,
            (batch_size, seq_len),
            hidden_states,
            past_length,
            sliding_window=getattr(backbone.config, "sliding_window", None),
        )

    @staticmethod
    def _layer_forward(
        layer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_values,
        output_attentions: bool,
        use_cache: bool,
        required_position_span: int,
    ):
        # The local Video3D Qwen2 rotary module consumes position_ids directly.
        # Older stock Transformers rotary modules instead accept seq_len. Only
        # the latter need the full-span compatibility shim.
        rotary = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
        original_rotary_forward = getattr(rotary, "forward", None)
        required_span = int(required_position_span)
        rotary_parameters = set()
        if original_rotary_forward is not None:
            try:
                rotary_parameters = set(inspect.signature(original_rotary_forward).parameters)
            except (TypeError, ValueError):
                rotary_parameters = set()

        if original_rotary_forward is not None and "seq_len" in rotary_parameters and "position_ids" not in rotary_parameters:
            def rotary_forward_with_full_span(x, seq_len=None):
                resolved_seq_len = required_span if seq_len is None else max(int(seq_len), required_span)
                return original_rotary_forward(x, seq_len=resolved_seq_len)

            rotary.forward = rotary_forward_with_full_span
        try:
            return layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        finally:
            if original_rotary_forward is not None:
                rotary.forward = original_rotary_forward

    @staticmethod
    def _normalize_position_ids(
        position_ids: Optional[torch.Tensor],
        *,
        sequence_length: int,
        device: torch.device,
        expects_3d: bool,
    ) -> torch.Tensor:
        """Normalize positions for stock Qwen2 or Video3D's three-axis Qwen2."""
        if position_ids is None:
            normalized = torch.arange(sequence_length, device=device, dtype=torch.long).view(1, -1)
            return normalized.unsqueeze(-1).expand(-1, -1, 3) if expects_3d else normalized

        normalized = position_ids.to(device=device, dtype=torch.long)
        if normalized.dim() == 2:
            if normalized.shape != (1, sequence_length):
                raise ValueError(
                    "position_ids must match the expanded prompt layout, "
                    f"got {tuple(normalized.shape)} for {(1, sequence_length)}."
                )
            return normalized.unsqueeze(-1).expand(-1, -1, 3) if expects_3d else normalized
        if normalized.dim() == 3 and normalized.shape == (1, sequence_length, 3):
            if expects_3d:
                return normalized
            if not bool(torch.all(normalized[..., 1:] == normalized[..., :1])):
                raise ValueError("Stock Qwen2 cannot consume distinct three-axis position IDs.")
            return normalized[..., 0]
        raise ValueError(
            "position_ids must have shape (1, sequence_length) or "
            f"(1, sequence_length, 3), got {tuple(normalized.shape)}."
        )

    def _prefill(
        self,
        *,
        causal_lm,
        inputs_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask_2d: Optional[torch.Tensor],
        past_key_values,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        labels: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        state = self._generation_state
        backbone = causal_lm.model
        num_layers = len(backbone.layers)
        entry_idx = int(self.compression_config.late_entry_layer) - 1
        exit_idx = int(self.compression_config.early_exit_layer) - 1
        # exit_idx == num_layers means that visual tokens remain active through
        # the final decoder layer (late entry without early exit).
        if not 0 <= entry_idx < exit_idx <= num_layers:
            raise ValueError(
                "Expected 1 <= late_entry_layer < early_exit_layer <= num_layers + 1, got "
                f"{self.compression_config.late_entry_layer}, {self.compression_config.early_exit_layer}, {num_layers}."
            )
        if inputs_embeds.shape[0] != 1 or inputs_embeds.shape[1] != state["active_mask"].numel():
            raise ValueError("Expanded LLaVA-OV embeddings do not match the registered prompt layout.")
        is_training = bool(causal_lm.training or labels is not None)
        if not is_training and not use_cache:
            raise ValueError("late_entry_early_exit generation requires use_cache=True.")
        if is_training and inputs_embeds.shape[0] != 1:
            raise ValueError("late_entry_early_exit training currently requires per-device batch_size=1.")

        active_mask = state["active_mask"].to(inputs_embeds.device)
        visual_mask = state["visual_mask"].to(inputs_embeds.device) & active_mask
        text_mask = active_mask & ~visual_mask
        full_hidden = inputs_embeds[:, active_mask, :]
        compact_visual_mask = visual_mask[active_mask]
        compact_text_mask = ~compact_visual_mask
        original_visual = full_hidden[:, compact_visual_mask, :].clone()
        hidden_states = full_hidden[:, compact_text_mask, :]

        # Physical token removal changes sequence lengths but must not renumber
        # text after the visual span. Keeping the original full-sequence RoPE
        # positions matches HiDrop's late-injection semantics.
        full_positions = self._normalize_position_ids(
            position_ids,
            sequence_length=active_mask.numel(),
            device=inputs_embeds.device,
            expects_3d=self._expects_3d_position_ids(backbone),
        )[:, active_mask]
        # Keep the rotary-table span large enough for preserved visual
        # coordinates, but generated text follows the original prompt length.
        # In Video3D mRoPE, visual xyz coordinates can be much larger than
        # the sequence length and must not become the next text position.
        full_position_span = int(full_positions.max().item()) + 1
        text_positions = full_positions[:, compact_text_mask]
        current_positions = text_positions
        if use_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values) if not isinstance(past_key_values, Cache) else past_key_values
        else:
            past_key_values = None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        per_layer_lengths = []

        for layer_idx, layer in enumerate(backbone.layers):
            if layer_idx == entry_idx:
                restored = torch.empty_like(full_hidden)
                restored[:, compact_text_mask, :] = hidden_states
                restored[:, compact_visual_mask, :] = original_visual
                hidden_states = restored
                current_positions = full_positions
            if layer_idx == exit_idx:
                hidden_states = hidden_states[:, compact_text_mask, :]
                current_positions = text_positions
            visual_mask_current = (
                compact_visual_mask
                if entry_idx <= layer_idx < exit_idx
                else None
            )
            self._observe_prefill_layer(
                layer_idx=layer_idx,
                layer=layer,
                hidden_states=hidden_states,
                position_ids=current_positions,
                visual_mask=visual_mask_current,
                full_position_span=full_position_span,
            )
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            per_layer_lengths.append(int(hidden_states.shape[1]))
            layer_attention_2d = torch.ones(
                1, hidden_states.shape[1], device=hidden_states.device, dtype=torch.bool
            )
            layer_mask = self._prepare_mask(
                backbone, layer_attention_2d, hidden_states, 0, output_attentions, use_cache
            )
            layer_args = (
                layer,
                hidden_states,
                layer_mask,
                current_positions,
                past_key_values,
                output_attentions,
                use_cache,
                full_position_span,
            )
            if backbone.gradient_checkpointing and backbone.training:
                layer_outputs = backbone._gradient_checkpointing_func(self._layer_forward, *layer_args)
            else:
                layer_outputs = self._layer_forward(*layer_args)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = backbone.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache.to_legacy_cache() if use_cache else None
        full_len = int(full_hidden.shape[1])
        text_len = int(compact_text_mask.sum().item())
        visual_count = int(compact_visual_mask.sum().item())
        # OV's ``grid`` serialization appends the learned image_newline
        # vector to each serialized row.  These format tokens participate in
        # the decoder computation, but are not visual patches.  Keep both
        # counts in the profile so public efficiency accounting can report
        # patch tokens without changing the execution path.
        visual_format_tokens = 0
        image_newline = getattr(backbone, "image_newline", None)
        if image_newline is not None:
            image_newline = image_newline.detach().to(
                device=full_hidden.device, dtype=full_hidden.dtype
            ).view(1, -1)
            if image_newline.shape[-1] == full_hidden.shape[-1]:
                visual_embeddings = full_hidden[:, compact_visual_mask, :]
                visual_format_tokens = int(
                    torch.all(visual_embeddings == image_newline, dim=-1).sum().item()
                )
        visual_patch_tokens = max(visual_count - visual_format_tokens, 0)
        active_layers = exit_idx - entry_idx
        average_visual = float(visual_count * active_layers) / float(num_layers)
        has_early_exit = exit_idx < num_layers
        final_len = text_len if has_early_exit else full_len
        final_visual_tokens = 0.0 if has_early_exit else float(visual_patch_tokens)
        if labels is not None:
            if labels.shape != (1, active_mask.numel()):
                raise ValueError(
                    "Training labels must match the expanded prompt layout, "
                    f"got {tuple(labels.shape)} for {(1, active_mask.numel())}."
                )
            active_labels = labels[:, active_mask.to(labels.device)]
            self._last_training_labels = (
                active_labels[:, compact_text_mask.to(labels.device)]
                if has_early_exit
                else active_labels
            )
        profile = {
            "compressor_name": "late_entry_early_exit",
            "position_id_strategy": "preserve_full_sequence",
            "late_entry_layer": int(self.compression_config.late_entry_layer),
            "early_exit_layer": int(self.compression_config.early_exit_layer),
            "active_visual_layers": active_layers,
            "visual_patch_tokens": visual_patch_tokens,
            "visual_sequence_tokens": visual_count,
            "visual_format_tokens": visual_format_tokens,
            "text_prompt_tokens": text_len,
            "prompt_sequence_length_before_prune": full_len,
            # Early exit changes the final decoder state/cache path, but does
            # not remove tokens from the multimodal input prompt.  Keep the
            # public input-token count tied to the expanded prompt length.
            "prompt_sequence_length": full_len,
            "final_sequence_length": final_len,
            "prefill_layer_token_lengths": per_layer_lengths,
            "prefill_final_visual_tokens": final_visual_tokens,
            "compressor_input_tokens": visual_count,
            "compressor_output_tokens": int(round(average_visual)),
            "llm_stage_input_tokens": visual_count,
            "llm_stage_output_tokens": int(round(average_visual)),
            "llm_stage_keep_ratio": float(active_layers) / float(num_layers),
            "token_keep_ratio": float(active_layers) / float(num_layers),
        }
        self._update_stats(
            input_tokens=visual_count * num_layers,
            output_tokens=visual_count * active_layers,
            late_entry_layer=int(self.compression_config.late_entry_layer),
            early_exit_layer=int(self.compression_config.early_exit_layer),
        )
        self._last_profile = profile
        state["phase"] = "decode"
        state["next_decode_position"] = self._next_decode_position(
            backbone=backbone,
            full_positions=full_positions,
            state=state,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _decode(
        self,
        *,
        causal_lm,
        inputs_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        past_key_values,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            raise ValueError("late_entry_early_exit decode requires the current token embedding.")
        if inputs_embeds.shape[0] != 1 or inputs_embeds.shape[1] != 1:
            raise ValueError("late_entry_early_exit decode expects one token per step.")
        if not use_cache or past_key_values is None:
            raise ValueError("late_entry_early_exit decode requires an existing KV cache.")
        backbone = causal_lm.model
        cache = DynamicCache.from_legacy_cache(past_key_values) if not isinstance(past_key_values, Cache) else past_key_values
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for layer_idx, layer in enumerate(backbone.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_length = cache.get_usable_length(1, layer_idx=layer_idx)
            layer_position_ids = self._make_decode_position_ids(
                backbone=backbone,
                position=int(self._generation_state["next_decode_position"]),
                device=inputs_embeds.device,
            )
            layer_attention_2d = torch.ones(
                1, past_length + 1, device=hidden_states.device, dtype=torch.bool
            )
            layer_mask = self._prepare_mask(
                backbone, layer_attention_2d, hidden_states, past_length, output_attentions, use_cache
            )
            layer_outputs = self._layer_forward(
                layer,
                hidden_states,
                layer_mask,
                layer_position_ids,
                cache,
                output_attentions,
                use_cache,
                int(self._generation_state["next_decode_position"]) + 1,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = backbone.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        self._generation_state["decode_steps"] += 1
        self._generation_state["next_decode_position"] += 1
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache.to_legacy_cache(),
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def compress_generation_forward(
        self,
        *,
        causal_lm,
        inputs_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask_2d: Optional[torch.Tensor],
        past_key_values=None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        labels=None,
    ) -> BaseModelOutputWithPast:
        if self._generation_state is None:
            raise RuntimeError("No late-entry/early-exit generation state is registered.")
        if self._generation_state["phase"] == "prefill":
            if labels is not None:
                # KV cache is unnecessary during teacher-forced training and
                # conflicts with gradient checkpointing in the decoder.
                use_cache = False
            return self._prefill(
                causal_lm=causal_lm,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask_2d=attention_mask_2d,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                labels=labels,
            )
        if causal_lm.training or labels is not None:
            raise RuntimeError("late_entry_early_exit training only supports a single prefill forward.")
        return self._decode(
            causal_lm=causal_lm,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
