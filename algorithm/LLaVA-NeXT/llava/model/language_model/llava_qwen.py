#    Copyright 2024 Hao Zhang
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


import logging
import os
import time
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM


logger = logging.getLogger(__name__)

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_last_runtime_profile(self):
        """Return optional per-request runtime measurements.

        Profiling is opt-in and the profile is intentionally kept separate
        from compression semantics.  The evaluator can consume it without
        changing the generation inputs or decoder path.
        """
        profile = getattr(self, "_last_runtime_profile", {})
        return profile.copy() if isinstance(profile, dict) else {}

    def reset_last_runtime_profile(self):
        self._last_runtime_profile = {}

    def _start_generation_forward_profiler(self):
        """Measure decoder forward calls without changing the generation path.

        The multimodal preparation timer includes the vision tower and any
        projector compressor.  These hooks are installed only after that
        preparation has finished, so the first record is the actual decoder
        prefill and later records are incremental decode forwards.
        """
        enabled = str(os.environ.get("OV_MEASURE_RUNTIME_PROFILE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled or not torch.cuda.is_available():
            return None

        records = []
        active_record = {"value": None}

        def _sequence_length(args, kwargs):
            value = kwargs.get("inputs_embeds")
            if value is None:
                value = kwargs.get("input_ids")
            if value is None and args:
                value = args[0]
            if torch.is_tensor(value) and value.ndim >= 2:
                return int(value.shape[1])
            return None

        def _pre_hook(_module, args, kwargs):
            past_key_values = kwargs.get("past_key_values")
            if past_key_values is None and len(args) > 3:
                past_key_values = args[3]
            record = {
                "is_prefill": past_key_values is None,
                "sequence_length": _sequence_length(args, kwargs),
                "wall_start": time.perf_counter(),
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": None,
            }
            record["start_event"].record(torch.cuda.current_stream())
            records.append(record)
            active_record["value"] = record

        def _post_hook(_module, args, kwargs, _output):
            record = active_record["value"]
            if record is None or record["end_event"] is not None:
                return
            record["wall_ms"] = (time.perf_counter() - record["wall_start"]) * 1000.0
            record["end_event"] = torch.cuda.Event(enable_timing=True)
            record["end_event"].record(torch.cuda.current_stream())
            active_record["value"] = None

        pre_handle = self.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        post_handle = self.register_forward_hook(_post_hook, with_kwargs=True)
        return {
            "records": records,
            "handles": (pre_handle, post_handle),
        }

    def _finish_generation_forward_profiler(self, profiler):
        if not profiler:
            return
        for handle in profiler.get("handles", ()):
            handle.remove()

        records = profiler.get("records", [])
        if not records:
            return
        torch.cuda.synchronize()
        for record in records:
            end_event = record.get("end_event")
            if end_event is not None:
                record["cuda_ms"] = float(record["start_event"].elapsed_time(end_event))

        prefill_records = [record for record in records if record.get("is_prefill")]
        decode_records = [record for record in records if not record.get("is_prefill")]
        if prefill_records:
            prefill = prefill_records[0]
            self._last_runtime_profile["llm_prefill_cuda_ms"] = float(prefill.get("cuda_ms", 0.0))
            self._last_runtime_profile["llm_prefill_wall_ms"] = float(prefill.get("wall_ms", 0.0))
            if prefill.get("sequence_length") is not None:
                self._last_runtime_profile["llm_prefill_sequence_length"] = int(prefill["sequence_length"])
        if decode_records:
            self._last_runtime_profile["llm_decode_cuda_ms"] = float(
                sum(record.get("cuda_ms", 0.0) for record in decode_records)
            )
            self._last_runtime_profile["llm_decode_forward_count"] = int(len(decode_records))
        self._last_runtime_profile["llm_forward_count"] = int(len(records))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        video_dict: Optional[Dict] = None,
        depths: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        cam2world: Optional[torch.Tensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        sample_cache_ids: Optional[List[str]] = None,
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        original_input_ids = input_ids
        original_attention_mask = attention_mask
        llm_analyzer = getattr(self.get_model(), "mm_llm_compressor", None)
        prepared_training_compression = False
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes,
                video_dict=video_dict,
                depths=depths,
                poses=poses,
                cam2world=cam2world,
                intrinsic=intrinsic,
            )

        if (
            self.training
            and images is not None
            and labels is not None
            and llm_analyzer is not None
            and hasattr(llm_analyzer, "prepare_training_compression")
        ):
            if original_input_ids is None:
                raise ValueError("LLM compression training requires the original multimodal input_ids.")
            llm_analyzer.prepare_training_compression(
                original_input_ids=original_input_ids,
                original_attention_mask=original_attention_mask,
                expanded_attention_mask=attention_mask,
                expanded_seq_len=int(inputs_embeds.shape[1]),
                image_token_index=IMAGE_TOKEN_INDEX,
                sample_cache_ids=sample_cache_ids,
            )
            prepared_training_compression = True

        if (
            llm_analyzer is not None
            and hasattr(llm_analyzer, "should_compress_generation_forward")
            and llm_analyzer.should_compress_generation_forward()
        ):
            try:
                if inputs_embeds is None:
                    if input_ids is None:
                        raise ValueError("LLM compression forward requires input_ids or inputs_embeds.")
                    inputs_embeds = self.get_model().embed_tokens(input_ids)
                resolved_use_cache = use_cache if use_cache is not None else self.config.use_cache
                resolved_output_attentions = (
                    output_attentions if output_attentions is not None else self.config.output_attentions
                )
                resolved_output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                resolved_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                outputs = llm_analyzer.compress_generation_forward(
                    causal_lm=self,
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    attention_mask_2d=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=resolved_use_cache,
                    output_attentions=resolved_output_attentions,
                    output_hidden_states=resolved_output_hidden_states,
                    labels=labels,
                )
                training_labels = (
                    llm_analyzer.consume_last_training_labels()
                    if labels is not None and hasattr(llm_analyzer, "consume_last_training_labels")
                    else labels
                )
                profile = (
                    llm_analyzer.consume_last_profile()
                    if hasattr(llm_analyzer, "consume_last_profile")
                    else None
                )
                if profile:
                    self.merge_last_compression_profile(profile)
                hidden_states = outputs[0]
                logits = self.lm_head(hidden_states).float()
                loss = None
                if training_labels is not None:
                    if logits.shape[:2] != training_labels.shape:
                        raise ValueError(
                            "LLM compression logits and labels must align, "
                            f"got {tuple(logits.shape[:2])} and {tuple(training_labels.shape)}."
                        )
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = training_labels[..., 1:].contiguous()
                    loss = CrossEntropyLoss()(
                        shift_logits.view(-1, self.config.vocab_size),
                        shift_labels.view(-1).to(shift_logits.device),
                    )
                if not resolved_return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            finally:
                if prepared_training_compression and hasattr(llm_analyzer, "clear_generation_compression"):
                    llm_analyzer.clear_generation_compression()

        if (
            llm_analyzer is not None
            and hasattr(llm_analyzer, "has_pending_generation_prefill")
            and llm_analyzer.has_pending_generation_prefill()
        ):
            sample_key = (
                llm_analyzer.pending_sample_key()
                if hasattr(llm_analyzer, "pending_sample_key")
                else None
            )
            resolved_use_cache = use_cache if use_cache is not None else self.config.use_cache
            resolved_output_attentions = (
                output_attentions if output_attentions is not None else self.config.output_attentions
            )
            resolved_output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            resolved_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            try:
                outputs = llm_analyzer.analyze_generation_prefill(
                    causal_lm=self,
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    expanded_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=resolved_use_cache,
                    output_attentions=resolved_output_attentions,
                    output_hidden_states=resolved_output_hidden_states,
                )
            except Exception as exc:
                if hasattr(llm_analyzer, "record_failure"):
                    llm_analyzer.record_failure(
                        sample_key=sample_key,
                        error=exc,
                        num_layers=len(self.model.layers),
                    )
                logger.error(
                    "LLaVA-OV generation-prefill analysis failed for sample %s (%s: %s); "
                    "falling back to the standard model forward.",
                    sample_key,
                    type(exc).__name__,
                    exc,
                )
            else:
                hidden_states = outputs[0]
                logits = self.lm_head(hidden_states).float()
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, self.config.vocab_size),
                        shift_labels.view(-1).to(shift_logits.device),
                    )
                if not resolved_return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        video_dict: Optional[Dict] = None,
        depths: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        cam2world: Optional[torch.Tensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        skip_llm_analysis = bool(kwargs.pop("skip_llm_analysis", False))
        runtime_profile_enabled = str(os.environ.get("OV_MEASURE_RUNTIME_PROFILE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.reset_last_runtime_profile()
        original_inputs = inputs.detach().clone() if inputs is not None else None
        original_attention_mask = attention_mask.detach().clone() if attention_mask is not None else None
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            if runtime_profile_enabled and torch.cuda.is_available():
                torch.cuda.synchronize()
            prepare_start = time.perf_counter() if runtime_profile_enabled else None
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
                video_dict=video_dict,
                depths=depths,
                poses=poses,
                cam2world=cam2world,
                intrinsic=intrinsic,
            )
            if runtime_profile_enabled and torch.cuda.is_available():
                torch.cuda.synchronize()
            if prepare_start is not None:
                self._last_runtime_profile["multimodal_prepare_time_ms"] = (
                    time.perf_counter() - prepare_start
                ) * 1000.0
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        llm_analyzer = getattr(self.get_model(), "mm_llm_compressor", None)
        if (
            images is not None
            and llm_analyzer is not None
            and hasattr(llm_analyzer, "prepare_generation_compression")
        ):
            if int(kwargs.get("num_beams", 1) or 1) != 1 or int(kwargs.get("num_return_sequences", 1) or 1) != 1:
                raise ValueError(
                    "LLaVA-OV late-entry/early-exit requires num_beams=1 and num_return_sequences=1."
                )
            try:
                llm_analyzer.prepare_generation_compression(
                    original_input_ids=original_inputs,
                    original_attention_mask=original_attention_mask,
                    expanded_attention_mask=attention_mask,
                    expanded_seq_len=int(inputs_embeds.shape[1]),
                    image_token_index=IMAGE_TOKEN_INDEX,
                )
            except Exception:
                # Preparation happens before the generation try/finally below.
                # Clear a partially registered layout before propagating the error.
                if hasattr(llm_analyzer, "clear_generation_compression"):
                    llm_analyzer.clear_generation_compression()
                raise
        if (
            images is not None
            and not skip_llm_analysis
            and llm_analyzer is not None
            and hasattr(llm_analyzer, "should_analyze")
            and llm_analyzer.should_analyze()
            and not (
                hasattr(llm_analyzer, "has_llm_compress")
                and llm_analyzer.has_llm_compress()
            )
        ):
            sample_key = None
            try:
                if original_inputs is None:
                    raise ValueError("LLM analysis requires the original multimodal input_ids.")
                sample_key = (
                    llm_analyzer.consume_next_sample_key()
                    if hasattr(llm_analyzer, "consume_next_sample_key")
                    else None
                )
                if hasattr(llm_analyzer, "prepare_generation_prefill"):
                    llm_analyzer.prepare_generation_prefill(
                        original_input_ids=original_inputs,
                        original_attention_mask=original_attention_mask,
                        image_token_index=IMAGE_TOKEN_INDEX,
                        sample_key=sample_key,
                    )
                elif hasattr(llm_analyzer, "analyze_prefill"):
                    llm_analyzer.analyze_prefill(
                        causal_lm=self,
                        inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        expanded_attention_mask=attention_mask,
                        original_input_ids=original_inputs,
                        original_attention_mask=original_attention_mask,
                        image_token_index=IMAGE_TOKEN_INDEX,
                        sample_key=sample_key,
                    )
                else:
                    raise TypeError(
                        f"Unsupported LLM analyzer interface: {type(llm_analyzer).__name__}."
                    )
            except Exception as exc:
                if hasattr(llm_analyzer, "record_failure"):
                    llm_analyzer.record_failure(
                        sample_key=sample_key,
                        error=exc,
                        num_layers=len(self.model.layers),
                    )
                logger.error(
                    "LLaVA-OV LLM analysis failed for sample %s (%s: %s); continuing normal generation.",
                    sample_key,
                    type(exc).__name__,
                    exc,
                )

        forward_profiler = self._start_generation_forward_profiler()
        try:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        finally:
            self._finish_generation_forward_profiler(forward_profiler)
            # The layout belongs to one multimodal request. Keeping it alive
            # would make a subsequent request enter the custom forward path.
            if (
                llm_analyzer is not None
                and hasattr(llm_analyzer, "clear_generation_compression")
            ):
                llm_analyzer.clear_generation_compression()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        video_dict = kwargs.pop("video_dict", None)
        depths = kwargs.pop("depths", None)
        poses = kwargs.pop("poses", None)
        cam2world = kwargs.pop("cam2world", None)
        intrinsic = kwargs.pop("intrinsic", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if video_dict is not None:
            inputs["video_dict"] = video_dict
        if depths is not None:
            inputs["depths"] = depths
        if poses is not None:
            inputs["poses"] = poses
        if cam2world is not None:
            inputs["cam2world"] = cam2world
        if intrinsic is not None:
            inputs["intrinsic"] = intrinsic
        llm_analyzer = getattr(self.get_model(), "mm_llm_compressor", None)
        if (
            past_key_values is not None
            and llm_analyzer is not None
            and hasattr(llm_analyzer, "should_compress_generation_forward")
            and llm_analyzer.should_compress_generation_forward()
        ):
            # HF's default crop uses layer 0's cache length. This compressor
            # intentionally has shorter caches in text-only layers, so that
            # crop can return several generated tokens instead of one.
            inputs["input_ids"] = input_ids[:, -1:]
            inputs["position_ids"] = inputs.get("position_ids", None)
            if inputs["position_ids"] is not None:
                inputs["position_ids"] = inputs["position_ids"][:, -1:]
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
