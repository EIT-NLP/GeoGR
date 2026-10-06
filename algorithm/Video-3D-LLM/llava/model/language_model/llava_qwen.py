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
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ...constants import IMAGE_TOKEN_INDEX
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config
from .qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM


logger = logging.getLogger(__name__)


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

        if hasattr(config, "ground_head_type") and config.ground_head_type is not None:
            self.ground_head_type = config.ground_head_type
            if config.ground_head_type == "mlp":
                # self.ground_head = nn.Sequential(
                #     nn.Linear(config.hidden_size, config.ground_head_hidden_size),
                #     nn.ReLU(),
                #     nn.LayerNorm(config.ground_head_hidden_size),
                #     nn.Linear(config.ground_head_hidden_size, 6)
                # )
                self.ground_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size)
                )
            elif config.ground_head_type == "score":
                self.ground_head_temperature = config.ground_head_temperature
                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_score = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1),
                )
            elif config.ground_head_type == "infonce":
                # self.ground_head_temperature = nn.Parameter(torch.tensor(config.ground_head_temperature))
                try:
                    self.ground_head_temperature = config.ground_head_temperature
                except:
                    self.ground_head_temperature = 0.07
                self.ground_head_zero_target = torch.nn.Parameter(torch.randn(config.hidden_size))

                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
            else:
                raise NotImplementedError
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        video_dict=None,
        use_object_proposals: bool = False,
        box_labels = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        original_input_ids = input_ids
        original_attention_mask = attention_mask
        llm_compressor = getattr(self.get_model(), "mm_llm_compressor", None)
        prepared_compression = False

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, object_features, object_boxes) = \
                self.prepare_inputs_labels_for_multimodal(
                    input_ids, 
                    position_ids, 
                    attention_mask, 
                    past_key_values, 
                    labels, 
                    images, 
                    modalities, 
                    image_sizes, 
                    video_dict,
                    use_object_proposals=use_object_proposals,
                )

        # Training must use the same LLM compression path as inference.  The
        # compressor operates after multimodal expansion, so its layout is
        # registered only after prepare_inputs_labels_for_multimodal().
        if (
            self.training
            and not use_object_proposals
            and images is not None
            and labels is not None
            and llm_compressor is not None
            and hasattr(llm_compressor, "prepare_training_compression")
        ):
            if original_input_ids is None:
                raise ValueError("LLM compression training requires original multimodal input_ids.")
            llm_compressor.prepare_training_compression(
                original_input_ids=original_input_ids,
                original_attention_mask=original_attention_mask,
                expanded_attention_mask=attention_mask,
                expanded_seq_len=int(inputs_embeds.shape[1]),
                image_token_index=IMAGE_TOKEN_INDEX,
            )
            prepared_compression = True

        if (
            use_object_proposals
            and images is not None
            and labels is not None
            and llm_compressor is not None
            and hasattr(llm_compressor, "prepare_grounding_compression")
        ):
            if original_input_ids is None:
                raise ValueError("Grounding LLM compression requires original multimodal input_ids.")
            llm_compressor.prepare_grounding_compression(
                original_input_ids=original_input_ids,
                original_attention_mask=original_attention_mask,
                expanded_attention_mask=attention_mask,
                expanded_seq_len=int(inputs_embeds.shape[1]),
                image_token_index=IMAGE_TOKEN_INDEX,
                ground_token_ids=getattr(self.config, "ground_token_ids", None),
            )
            prepared_compression = True

        if use_object_proposals:
            try:
                return self.predict_box(
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
                    object_features=object_features,
                    object_boxes=object_boxes,
                    box_labels=box_labels,
                )
            finally:
                if prepared_compression and hasattr(llm_compressor, "clear_generation_compression"):
                    llm_compressor.clear_generation_compression()

        if (
            llm_compressor is not None
            and hasattr(llm_compressor, "should_compress_generation_forward")
            and llm_compressor.should_compress_generation_forward()
        ):
            try:
                resolved_use_cache = use_cache if use_cache is not None else self.config.use_cache
                resolved_output_attentions = (
                    output_attentions if output_attentions is not None else self.config.output_attentions
                )
                resolved_output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                resolved_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                compressor_inputs_embeds = inputs_embeds
                if compressor_inputs_embeds is None:
                    if input_ids is None:
                        raise ValueError(
                            "Video3D LLM compression requires input_ids when decode inputs_embeds are absent."
                        )
                    compressor_inputs_embeds = self.get_model().embed_tokens(input_ids)
                outputs = llm_compressor.compress_generation_forward(
                    causal_lm=self,
                    inputs_embeds=compressor_inputs_embeds,
                    position_ids=position_ids,
                    attention_mask_2d=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=resolved_use_cache,
                    output_attentions=resolved_output_attentions,
                    output_hidden_states=resolved_output_hidden_states,
                    labels=labels,
                )
                training_labels = (
                    llm_compressor.consume_last_training_labels()
                    if labels is not None and hasattr(llm_compressor, "consume_last_training_labels")
                    else labels
                )
                profile = (
                    llm_compressor.consume_last_profile()
                    if hasattr(llm_compressor, "consume_last_profile")
                    else None
                )
                if profile and hasattr(self, "merge_last_compression_profile"):
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
                if prepared_compression and hasattr(llm_compressor, "clear_generation_compression"):
                    llm_compressor.clear_generation_compression()
            

        if dpo_forward:
            outputs = self.model(
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

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            labels = self.model.pop_last_llm_pruned_labels(default=labels)
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
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        skip_llm_analysis = bool(kwargs.pop("skip_llm_analysis", False))
        original_inputs = inputs.detach().clone() if inputs is not None else None
        original_attention_mask = attention_mask.detach().clone() if attention_mask is not None else None
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, _, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, video_dict=kwargs.get("video_dict", None))
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # Analysis-only compressors replay the expanded multimodal prefill and
        # never alter the generation inputs or output tokens.
        llm_analyzer = getattr(self.get_model(), "mm_llm_compressor", None)
        if (
            images is not None
            and llm_analyzer is not None
            and hasattr(llm_analyzer, "prepare_generation_compression")
        ):
            if int(kwargs.get("num_beams", 1) or 1) != 1 or int(kwargs.get("num_return_sequences", 1) or 1) != 1:
                raise ValueError(
                    "Video3D late-entry/early-exit requires num_beams=1 and num_return_sequences=1."
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
                    raise ValueError("Video3D LLM analysis requires the original multimodal input_ids.")
                sample_key = (
                    llm_analyzer.consume_next_sample_key()
                    if hasattr(llm_analyzer, "consume_next_sample_key")
                    else None
                )
                if not hasattr(llm_analyzer, "analyze_prefill"):
                    raise TypeError(
                        f"Unsupported Video3D LLM analyzer interface: {type(llm_analyzer).__name__}."
                    )
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
            except Exception as exc:
                if hasattr(llm_analyzer, "record_failure"):
                    llm_analyzer.record_failure(
                        sample_key=sample_key,
                        error=exc,
                        num_layers=len(self.model.layers),
                    )
                logger.error(
                    "Video3D LLM analysis failed for sample %s (%s: %s); continuing normal generation.",
                    sample_key,
                    type(exc).__name__,
                    exc,
                )

        try:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        finally:
            if llm_analyzer is not None and hasattr(llm_analyzer, "clear_generation_compression"):
                llm_analyzer.clear_generation_compression()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        llm_compressor = getattr(self.get_model(), "mm_llm_compressor", None)
        if (
            past_key_values is not None
            and llm_compressor is not None
            and hasattr(llm_compressor, "should_compress_generation_forward")
            and llm_compressor.should_compress_generation_forward()
        ):
            # Layer-0 has a deliberately shorter KV cache than visual-active
            # layers. HF must therefore feed exactly the newest decode token.
            inputs["input_ids"] = input_ids[:, -1:]
            if inputs.get("position_ids") is not None:
                inputs["position_ids"] = inputs["position_ids"][:, -1:]
        return inputs

    
    def predict_box(
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
        cache_position=None,
        video_dict=None,
        object_features=None,
        object_boxes=None,
        box_labels=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        llm_compressor = getattr(self.get_model(), "mm_llm_compressor", None)
        used_direct_llm_compressor = (
            llm_compressor is not None
            and hasattr(llm_compressor, "should_compress_generation_forward")
            and llm_compressor.should_compress_generation_forward()
        )
        if used_direct_llm_compressor:
            outputs = llm_compressor.compress_generation_forward(
                causal_lm=self,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask_2d=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache if use_cache is not None else self.config.use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                labels=labels,
            )
            if hasattr(llm_compressor, "consume_last_training_labels"):
                compressed_labels = llm_compressor.consume_last_training_labels()
                if compressed_labels is not None:
                    labels = compressed_labels
            if hasattr(llm_compressor, "consume_last_profile"):
                profile = llm_compressor.consume_last_profile()
                if profile and hasattr(self, "merge_last_compression_profile"):
                    self.merge_last_compression_profile(profile)
        else:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
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

        hidden_states = outputs[0]
        # Direct LLM compressors return their aligned labels through their own
        # state. The Qwen decoder is bypassed in that path, so its label slot is
        # unset and must not overwrite the compressor labels with None.
        if not used_direct_llm_compressor:
            labels = self.model.pop_last_llm_pruned_labels(default=labels)
        if labels is None:
            raise ValueError("Grounding requires labels so the model can locate <ground> tokens.")
        if labels.device != hidden_states.device:
            labels = labels.to(hidden_states.device)
        if object_features is not None and object_features.device != hidden_states.device:
            object_features = object_features.to(hidden_states.device)
        if hidden_states.shape[:2] != labels.shape[:2]:
            raise ValueError(
                "Grounding labels and hidden states must stay aligned after multimodal preparation/pruning: "
                f"{tuple(labels.shape)} vs {tuple(hidden_states.shape[:2])}."
            )

        ground_locations = (labels >= self.config.ground_token_ids[0]) & (labels <= self.config.ground_token_ids[-1])
        ground_hidden = hidden_states[ground_locations].squeeze(1)
        
        if self.ground_head_type == 'mlp':
            ground_hidden = self.ground_head(ground_hidden).squeeze(0) 
            scores = (ground_hidden * object_features).sum(dim=-1)
        elif self.ground_head_type == 'score':
            obj_feat = self.ground_head_obj(object_features.to(ground_hidden.dtype)) # B, C
            query_feat = self.ground_head_query(ground_hidden) # 1, C
            # sim = (F.normalize(obj_feat) * F.normalize(query_feat)).sum(dim=-1)
            mul_feat = obj_feat * query_feat
            scores = self.ground_head_score(mul_feat) # B, 1
            scores = scores.squeeze(1)

        elif self.ground_head_type == "infonce":
            zero_target = self.ground_head_zero_target.unsqueeze(0).to(
                device=object_features.device,
                dtype=object_features.dtype,
            )
            object_features = torch.cat([object_features, zero_target], dim=0)
            obj_feat = self.ground_head_obj(object_features.to(ground_hidden.dtype))
            query_feat = self.ground_head_query(ground_hidden)
            obj_feat = F.normalize(obj_feat)
            query_feat = F.normalize(query_feat)
            scores = (obj_feat * query_feat).sum(dim=-1)

        loss = None
        if box_labels is not None:
            if self.ground_head_type == "infonce":
                if len(box_labels[0]) == 0: # zero-target
                    box_labels[0].append(-1)
                logits = torch.exp(scores / self.ground_head_temperature)
                loss = - torch.log( logits[box_labels[0]].sum() / logits.sum())
                # negative_logits_sum = logits.sum() - logits[box_labels[0]].sum()
                # for idx in box_labels[0]:
                #     loss += - torch.log(logits[idx] / (negative_logits_sum + logits[idx]))
                # loss /= len(box_labels[0])
            else:
                bce_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                target = torch.zeros_like(scores)
                target[box_labels[0]] = 1
                weight = torch.ones_like(scores)
                if len(box_labels[0]) != 0:
                    weight[box_labels[0]] *= (scores.shape[0] - len(box_labels[0])) / len(box_labels[0])
                
                bce_loss = (bce_loss_fct(scores, target.detach()) * weight).mean()
                loss = bce_loss  
                # nce_loss = 0
                # logits = torch.exp(sim / self.ground_head_temperature)
                # negative_logits_sum = logits.sum() - logits[box_labels[0]].sum()
                # if len(box_labels[0]) != 0:
                #     for idx in box_labels[0]:
                #         nce_loss += - torch.log(logits[idx] / (negative_logits_sum + logits[idx]))
                #     nce_loss /= len(box_labels[0])
                # loss = bce_loss + nce_loss
        return loss, scores

        # loss = None
        # if box_labels is not None:
        #     ## BCE
        #     loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        #     target = torch.zeros_like(scores)
        #     target[box_labels[0]] = 1
        #     weight = torch.ones_like(scores)
        #     weight[box_labels[0]] *= scores.shape[0] - 1
        #     loss = (loss_fct(scores, target.detach()) * weight).mean()
        #     ## CE 
        #     # loss_fct = nn.CrossEntropyLoss()
        #     # loss = loss_fct(scores, box_labels[0]) / self.config.ground_loss_scale


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
