# =================================================================================================
# File: vis_zephyr/model/language_model/vis_zephyr.py
# Description: The core implementation of the Vision-Zephyr model.
#              Inherits: Mistral's architecture and integrates multimodal capabilities.
# =================================================================================================
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from transformers import (
    AutoConfig, AutoModelForCausalLM,
    MistralConfig, MistralModel, MistralForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..vis_zephyr_arch import VisZephyrMetaModel, VisZephyrMetaForCausalLM

class VisZephyrConfig(MistralConfig):
    model_type = "vis_zephyr"

class VisZephyrModel(VisZephyrMetaModel, MistralModel):
    config_class = VisZephyrConfig

    def __init__(self, config: MistralConfig):
        super(VisZephyrModel, self).__init__(config)

class VisZephyrForCausalLM(MistralForCausalLM, VisZephyrMetaForCausalLM):
    """
    Full Vision-Zephyr model for Causal Language Modeling, extending Mistral's architecture.
    Including:
      - LLM head;
      - Orchestration of multimodal inputs, and generation capabilities.
    """
    config_class = VisZephyrConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        
        #Replace the standard MistralModel with Vision-enabled VisZephyrModel
        self.model   = VisZephyrModel(config)
        #Re-initialize the language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
        #Final weights initialization
        self.post_init()

    def get_model(self):
        """Returns the core Vision-Zephyr model instance"""
        return self.model
    
    def forward(
            self,
            input_ids           : torch.LongTensor                  = None,
            attention_mask      : Optional[torch.Tensor]            = None,
            position_ids        : Optional[torch.LongTensor]        = None,
            past_key_values     : Optional[List[torch.FloatTensor]] = None,
            inputs_embeds       : Optional[torch.FloatTensor]       = None,
            labels              : Optional[torch.LongTensor]        = None,
            use_cache           : Optional[bool]                    = None,
            output_attentions   : Optional[bool]                    = None,
            output_hidden_states: Optional[bool]                    = None,
            images              : Optional[torch.FloatTensor]       = None,
            images_size         : Optional[List[List[int]]]         = None,
            return_dict         : Optional[bool]                    = None,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_size,
            )
        
        #Call original forward method
        return super().forward(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            position_ids         = position_ids,
            past_key_values      = past_key_values,
            inputs_embeds        = inputs_embeds,
            labels               = labels,
            use_cache            = use_cache,
            output_attentions    = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict          = return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        images_size: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generation method
        """
        position_ids   = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported in this generate function.")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _, #past_key_values
                inputs_embeds,
                _ #labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None, #past_key_values
                None, #labels
                images,
                images_size,
            )
        else:
            #Text-only generation
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        
        return super().generate(
            position_ids   = position_ids,
            attention_mask = attention_mask,
            inputs_embeds  = inputs_embeds,
            **kwargs
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values = None,
        inputs_embeds   = None,
        **kwargs,
    ):
        """
        Prepare inputs for generation, handling multimodal inputs.
        """
        images      = kwargs.pop("images", None)
        images_size = kwargs.pop("images_size", None)
        
        #prepare_inputs_for_generation from MistralForCausalLM
        inputs = super().prepare_inputs_for_generation(
            input_ids       = input_ids,
            past_key_values = past_key_values,
            inputs_embeds   = inputs_embeds,
            **kwargs
        )

        if images is not None:
            inputs['images'] = images
        if images_size is not None:
            inputs['images_size'] = images_size
        
        return inputs
    
#Register the custom model/config with HF's Auto-classes: Allows `AutoModelForCausalLM.from_pretrained(...)` to work with "vis_zephyr".
AutoConfig.register("vis_zephyr", VisZephyrConfig)
AutoModelForCausalLM.register(VisZephyrConfig, VisZephyrForCausalLM)