# =================================================================================================
# File: vis_zephyr/train/zephyr_flash_attn_monkey_patch.py
# Description: Monkey-patches the standard attention mechanism in Zephyr/Mistral models
#              with the highly optimized Flash Attention 2 implementation
#                - Improves training speed;
#                - Reduces memory usage.
#
# Adapted from: https://github.com/lm-sys/FastChat/blob/main/llava/train/llama_flash_attn_monkey_patch.py
# =================================================================================================

from typing import Optional, Tuple
import warnings
import torch

import transformers
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

def forward(
    self,
    hidden_states    : torch.Tensor,
    attention_mask   : Optional[torch.Tensor] = None,
    position_ids     : Optional[torch.Tensor] = None,
    past_key_value   : Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache        : bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Forward pass for the attention mechanism (of Mistral/Zephyr) using Flash Attention 2.
    """
    if output_attentions:
        warnings.warn("Flash Attention does not support outputting attention weights. Returning None for attention weights.")
    
    #Batch size and query length of the input hidden states
    batch_size, query_length, _ = hidden_states.size()

    #Project inputs -> Q (query), K (key), V (value)
    query_states = (
        self.q_proj(hidden_states)
        .view(batch_size, query_length, self.num_heads, self.head_dim)
        .transpose(1, 2)  # (batch_size, num_heads, query_length, head_dim)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)  # (batch_size, num_key_value_heads, query_length, head_dim)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)  # (batch_size, num_key_value_heads, query_length, head_dim)
    )

    #Apply rotary positional embeddings (RoPE) for relative positional encoding
    keyvalue_seq_length = key_states.shape[-2] #Current sequence length of keys/values
    if past_key_value is not None:
        keyvalue_seq_length += past_key_value[0].shape[-2] #Include cache sequence length
    
    #Calculate rotary embeddings (cos and sin) for keys and values
    cos, sin = self.rotary_emb(value_states, seq_len = keyvalue_seq_length)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    #Handle cached key/values for autoregressive generation
    if past_key_value is not None:
        #Reuse cached key/values if available
        key_states   = torch.cat([past_key_value[0], key_states], dim = 2)
        value_states = torch.cat([past_key_value[1], value_states], dim = 2)
    
    past_key_value = (key_states, value_states) if use_cache else None

    #Grouped-Query Attention (GQA): repeat keys/values heads for efficiency
    key_states   = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    #Prepare concatenated QKV for Flash Attention: shape required by flash-attn [batch_size, seq_len, 3, num_heads, head_dim]
    query_keyvalue   = torch.stack([query_states, key_states, value_states], dim = 2)
    query_keyvalue   = query_keyvalue.transpose(1, 3) #(batch_size, seq_len, 3, num_heads, head_dim)
    key_padding_mask = attention_mask

    #If no padding mask provided, use simple cu_seqlens approach
    if key_padding_mask is None:
        #Flatten to [batch_size * seq_len, 3, num_heads * head_dim]
        query_keyvalue = query_keyvalue.reshape(-1, 3, self.num_heads * self.head_dim)
        #Starting offsets per sequence
        cu_seqlens = torch.arange(
            0, (batch_size + 1)*query_length, step = query_length,
            dtype  = torch.int32,
            device = query_keyvalue.device
        )
        #Max sequence length is the query length
        max_s = query_length

        output = flash_attn_unpadded_qkvpacked_func(
            query_keyvalue,
            cu_seqlens,
            max_s,
            0.0,
            softmax_scale = None,
            causal        = True, 
        ).view(batch_size, query_length, -1)
    else:
        query_keyvalue = query_keyvalue.reshape(batch_size, query_length, -1)
        #Remove padding entries for FlashAttention
        query_keyvalue_unpad, indices, cu_seqlens, max_s = unpad_input(
            query_keyvalue,
            key_padding_mask,
        )
        query_keyvalue_unpad = query_keyvalue_unpad.reshape(-1, 3, self.num_heads, self.head_dim)
        #Apply Flash Attention to unpadded input
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            query_keyvalue_unpad,
            cu_seqlens,
            max_s,
            0.0,
            softmax_scale = None,
            causal        = True, 
        ).reshape(-1, self.num_heads * self.head_dim)
        #Pad the output back to original shape
        output = pad_input(
            output_unpad.reshape(-1, self.num_heads * self.head_dim),
            indices,
            batch_size,
            query_length,
        )
    
    attn_output = self.o_proj(output) #Final linear projection (layer)
    
    #return attention output, None for attention weights, and past key/values if requested
    return attn_output, None, past_key_value


def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """
    Override default mask. FlashAttention manages causal mask internally.
    """
    return attention_mask #[batch_size, seq_length]

def replace_mistral_attn_with_flash_attn(self):
    """
    Monkey-patch Mistral model to use Flash Attention.
    """
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (_prepare_decoder_attention_mask)
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward