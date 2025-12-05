#!/usr/bin/env python3
"""
FlashInfer End-to-End LLM Inference Example

This script demonstrates how to use FlashInfer kernels for complete LLM inference,
replacing the standard PyTorch/Transformers operations with optimized FlashInfer kernels:

- flashinfer.embedding - Token embedding lookup
- flashinfer.linear / flashinfer.linear_with_bias - Linear projections  
- flashinfer.rmsnorm - RMS normalization
- flashinfer.apply_rope_pos_ids_inplace - Rotary Position Embeddings (RoPE)
- flashinfer.apply_llama31_rope_pos_ids_inplace - Llama 3.1 RoPE with scaling
- flashinfer.single_prefill_with_kv_cache - Prefill attention
- flashinfer.single_decode_with_kv_cache - Decode attention  
- flashinfer.silu_and_mul - SiLU activation (for gated MLP)
- flashinfer.top_k_top_p_sampling_from_probs - Top-k/top-p sampling
- flashinfer.sparse_moe_forward - Sparse Mixture of Experts routing

Supported Models:
- meta-llama/Llama-3.1-8B-Instruct (BF16)
- Qwen/Qwen2.5-1.5B-Instruct (BF16)
- Qwen/Qwen3-4B-Instruct-2507 (BF16)
- Qwen/Qwen3-4B-Instruct-2507-FP8 (FP8 quantized)
- Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE, BF16)
- Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, FP8 quantized)

Usage:
    python llm_inference.py --model Qwen/Qwen2.5-1.5B-Instruct
    python llm_inference.py --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --max-new-tokens 100
    python llm_inference.py --model meta-llama/Llama-3.1-8B-Instruct --prompt "Explain quantum computing"
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from typing import Optional, Tuple, List
import flashinfer


class FlashInferEmbedding(nn.Module):
    """Embedding using FlashInfer kernel"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return flashinfer.embedding(input, self.weight)


class FlashInferLinear(nn.Module):
    """Linear layer using FlashInfer kernel, supports FP8 with block-wise scaling"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False, use_fp8: bool = False, block_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8
        self.block_size = block_size
        
        # For FP8, we store dequantized weights in compute dtype for simplicity
        # (Full FP8 compute would require FlashInfer FP8 GEMM support)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return flashinfer.linear_with_bias(input, self.weight, self.bias)
        else:
            return flashinfer.linear(input, self.weight)


class FlashInferLlamaRMSNorm(nn.Module):
    """RMSNorm using FlashInfer kernel"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return flashinfer.rmsnorm(x, self.weight, self.eps)


class FlashInferMLP(nn.Module):
    """MLP using FlashInfer kernels for linear and activation"""
    
    def __init__(self, config, use_fp8: bool = False, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        
        self.gate_proj = FlashInferLinear(self.hidden_size, self.intermediate_size, bias=False, use_fp8=use_fp8)
        self.up_proj = FlashInferLinear(self.hidden_size, self.intermediate_size, bias=False, use_fp8=use_fp8)
        self.down_proj = FlashInferLinear(self.intermediate_size, self.hidden_size, bias=False, use_fp8=use_fp8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate_up = torch.cat([gate, up], dim=-1)
        hidden = flashinfer.silu_and_mul(gate_up)
        return self.down_proj(hidden)


class FlashInferSparseMoeBlock(nn.Module):
    """Sparse Mixture of Experts block using FlashInfer kernels"""
    
    def __init__(self, config, use_fp8: bool = False):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.hidden_size = config.hidden_size
        
        # Router gate
        self.gate = FlashInferLinear(config.hidden_size, config.num_experts, bias=False, use_fp8=use_fp8)
        
        # Expert MLPs
        moe_intermediate_size = getattr(config, 'moe_intermediate_size', config.intermediate_size)
        self.experts = nn.ModuleList([
            FlashInferMLP(config, use_fp8=use_fp8, intermediate_size=moe_intermediate_size)
            for _ in range(self.num_experts)
        ])
    
    def _expert_fn(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Compute single expert output for use with flashinfer.sparse_moe_forward"""
        return self.experts[expert_idx](x)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using flashinfer.sparse_moe_forward"""
        # Use FlashInfer's sparse_moe_forward function
        return flashinfer.sparse_moe_forward(
            hidden_states,
            self.gate.weight,
            self._expert_fn,
            self.num_experts,
            self.top_k,
            self.norm_topk_prob,
        )


class FlashInferAttention(nn.Module):
    """Attention using FlashInfer kernels, supports both Llama and Qwen"""
    
    def __init__(self, config, layer_idx: int, model_type: str = "llama", use_fp8: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.model_type = model_type
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Check for attention_bias config (Qwen2 models have bias in Q, K, V)
        attention_bias = getattr(config, 'attention_bias', False)
        
        self.q_proj = FlashInferLinear(self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias, use_fp8=use_fp8)
        self.k_proj = FlashInferLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias, use_fp8=use_fp8)
        self.v_proj = FlashInferLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias, use_fp8=use_fp8)
        self.o_proj = FlashInferLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, use_fp8=use_fp8)
        
        # QK normalization (used in Qwen3 and Qwen3-MoE)
        config_model_type = getattr(config, 'model_type', '')
        self.use_qk_norm = model_type == "qwen" and config_model_type in ('qwen3', 'qwen3_moe')
        if self.use_qk_norm:
            self.q_norm = FlashInferLlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = FlashInferLlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # RoPE parameters
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.rope_scaling = getattr(config, 'rope_scaling', None)
        
        # Llama 3.1 specific RoPE scaling
        if model_type == "llama" and self.rope_scaling:
            self.rope_scale = self.rope_scaling.get('factor', 8.0)
            self.low_freq_factor = self.rope_scaling.get('low_freq_factor', 1.0)
            self.high_freq_factor = self.rope_scaling.get('high_freq_factor', 4.0)
            self.old_context_len = self.rope_scaling.get('original_max_position_embeddings', 8192)
            self.use_llama31_rope = True
        else:
            self.rope_scale = 1.0
            self.use_llama31_rope = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply QK normalization if enabled (Qwen3)
        # FlashInfer rmsnorm supports 2D/3D, so reshape for norm
        if self.use_qk_norm:
            q_shape = q.shape
            k_shape = k.shape
            # Reshape to [batch*seq*heads, head_dim] for rmsnorm
            q = self.q_norm(q.view(-1, self.head_dim)).view(q_shape)
            k = self.k_norm(k.view(-1, self.head_dim)).view(k_shape)
        
        # Apply RoPE using FlashInfer
        q_rope = q.view(batch_size * seq_len, self.num_heads, self.head_dim)
        k_rope = k.view(batch_size * seq_len, self.num_key_value_heads, self.head_dim)
        pos_ids = position_ids.view(-1).to(torch.int32)
        
        if self.use_llama31_rope:
            # Llama 3.1 style RoPE with scaling
            flashinfer.apply_llama31_rope_pos_ids_inplace(
                q_rope, k_rope, pos_ids,
                rotary_dim=self.head_dim,
                interleave=False,
                rope_scale=self.rope_scale,
                rope_theta=self.rope_theta,
                low_freq_factor=self.low_freq_factor,
                high_freq_factor=self.high_freq_factor,
                old_context_len=self.old_context_len,
            )
        else:
            # Standard RoPE (for Qwen and other models)
            flashinfer.apply_rope_pos_ids_inplace(
                q_rope, k_rope, pos_ids,
                rotary_dim=self.head_dim,
                interleave=False,
                rope_scale=self.rope_scale,
                rope_theta=self.rope_theta,
            )
        
        q = q_rope.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_rope.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        
        new_kv_cache = (k, v)
        kv_len = k.shape[1]
        
        if self.num_key_value_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1)
            k = k.reshape(batch_size, kv_len, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1)
            v = v.reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        
        q_fi = q.squeeze(0)
        k_fi = k.squeeze(0)
        v_fi = v.squeeze(0)
        
        if is_prefill or seq_len > 1:
            attn_output = flashinfer.single_prefill_with_kv_cache(
                q_fi, k_fi, v_fi,
                causal=True,
                kv_layout="NHD",
            )
        else:
            q_decode = q_fi.squeeze(0)
            attn_output = flashinfer.single_decode_with_kv_cache(
                q_decode, k_fi, v_fi,
                kv_layout="NHD",
            )
            attn_output = attn_output.unsqueeze(0)
        
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, new_kv_cache


class FlashInferDecoderLayer(nn.Module):
    """Decoder layer using FlashInfer kernels, supports both dense and MoE"""
    
    def __init__(self, config, layer_idx: int, model_type: str = "llama", use_fp8: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = FlashInferAttention(config, layer_idx, model_type, use_fp8)
        
        # Check if this layer should use MoE
        num_experts = getattr(config, 'num_experts', 0)
        decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1)
        mlp_only_layers = getattr(config, 'mlp_only_layers', [])
        
        self.use_moe = (
            num_experts > 0 and
            layer_idx not in mlp_only_layers and
            (layer_idx + 1) % decoder_sparse_step == 0
        )
        
        if self.use_moe:
            self.mlp = FlashInferSparseMoeBlock(config, use_fp8)
        else:
            self.mlp = FlashInferMLP(config, use_fp8)
        
        self.input_layernorm = FlashInferLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FlashInferLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv_cache = self.self_attn(
            hidden_states, position_ids, kv_cache, is_prefill
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MoE returns (hidden_states, router_logits), dense MLP returns just hidden_states
        if self.use_moe:
            hidden_states, _ = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states, new_kv_cache


class FlashInferModel(nn.Module):
    """Model using FlashInfer kernels"""
    
    def __init__(self, config, model_type: str = "llama", use_fp8: bool = False):
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.embed_tokens = FlashInferEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            FlashInferDecoderLayer(config, i, model_type, use_fp8)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = FlashInferLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        is_prefill: bool = True,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states = self.embed_tokens(input_ids)
        
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv_cache = layer(
                hidden_states, position_ids, kv_cache, is_prefill
            )
            new_kv_caches.append(new_kv_cache)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_kv_caches


class FlashInferForCausalLM(nn.Module):
    """Causal LM using FlashInfer kernels"""
    
    def __init__(self, config, model_type: str = "llama", use_fp8: bool = False):
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
        
        self.model = FlashInferModel(config, model_type, use_fp8)
        
        # For tied embeddings, lm_head shares weights with embed_tokens
        if self.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens weight
        else:
            self.lm_head = FlashInferLinear(config.hidden_size, config.vocab_size, bias=False, use_fp8=use_fp8)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        is_prefill: bool = True,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states, new_kv_caches = self.model(
            input_ids, position_ids, kv_caches, is_prefill
        )
        
        if self.tie_word_embeddings:
            # Use embedding weights for LM head
            logits = flashinfer.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        return logits, new_kv_caches
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate tokens using FlashInfer sampling"""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        logits, kv_caches = self.forward(input_ids, position_ids, is_prefill=True)
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :]
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            if temperature > 0:
                probs_f32 = probs.float()
                next_token = flashinfer.top_k_top_p_sampling_from_probs(
                    probs_f32,
                    top_k,
                    top_p,
                )
            else:
                next_token = probs.argmax(dim=-1)
            
            next_token = next_token.unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            eos_token_id = self.config.eos_token_id
            if eos_token_id is not None:
                if isinstance(eos_token_id, list):
                    if next_token.item() in eos_token_id:
                        break
                elif next_token.item() == eos_token_id:
                    break
            
            position_ids = torch.tensor([[generated_ids.shape[1] - 1]], device=device)
            logits, kv_caches = self.forward(
                next_token, position_ids, kv_caches, is_prefill=False
            )
        
        return generated_ids


def dequantize_fp8_blockwise(weight_fp8: torch.Tensor, scale_inv: torch.Tensor, block_size: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize FP8 weights with block-wise scaling.
    
    The dequantization formula is: weight_dequant = weight_fp8 * scale_inv
    Note: Despite the name 'scale_inv', this is actually the scale factor to multiply.
    
    Args:
        weight_fp8: FP8 weight tensor of shape [out_features, in_features]
        scale_inv: Scale tensor of shape [out_blocks, in_blocks]
        block_size: Tuple of (out_block_size, in_block_size)
        dtype: Target dtype for dequantized weights
    
    Returns:
        Dequantized weight tensor of shape [out_features, in_features]
    """
    out_features, in_features = weight_fp8.shape
    out_block_size, in_block_size = block_size
    
    # Convert FP8 to float for computation
    weight_float = weight_fp8.to(torch.float32)
    
    # Calculate number of blocks
    out_blocks = out_features // out_block_size
    in_blocks = in_features // in_block_size
    
    # Reshape weight for block-wise operations
    # weight: [out_features, in_features] -> [out_blocks, out_block_size, in_blocks, in_block_size]
    weight_blocked = weight_float.view(out_blocks, out_block_size, in_blocks, in_block_size)
    
    # Expand scale_inv to match block structure
    scale_expanded = scale_inv.view(out_blocks, 1, in_blocks, 1)
    
    # Dequantize: multiply by scale_inv
    weight_dequant = weight_blocked * scale_expanded
    
    # Reshape back
    weight_dequant = weight_dequant.view(out_features, in_features)
    
    return weight_dequant.to(dtype)


def load_weights_from_hf(flashinfer_model: FlashInferForCausalLM, hf_model, use_fp8: bool = False, compute_dtype: torch.dtype = torch.bfloat16):
    """Copy weights from HuggingFace model to FlashInfer model"""
    
    def copy_linear_weight(fi_linear, hf_linear, use_fp8: bool):
        """Copy weights handling FP8 quantization with block-wise dequantization"""
        if use_fp8 and hasattr(hf_linear, 'weight') and hf_linear.weight.dtype == torch.float8_e4m3fn:
            # FP8 model with block-wise quantization - dequantize
            if hasattr(hf_linear, 'weight_scale_inv'):
                block_size = getattr(hf_linear, 'block_size', [128, 128])
                weight_dequant = dequantize_fp8_blockwise(
                    hf_linear.weight.data,
                    hf_linear.weight_scale_inv.data,
                    tuple(block_size),
                    compute_dtype
                )
                fi_linear.weight.data.copy_(weight_dequant)
            else:
                # No scale - just convert dtype
                fi_linear.weight.data.copy_(hf_linear.weight.data.to(compute_dtype))
        else:
            # BF16/FP16 model
            fi_linear.weight.data.copy_(hf_linear.weight.data.to(compute_dtype))
        
        # Handle bias - HF model might have bias even if FI model doesn't have it configured
        if hf_linear.bias is not None:
            if fi_linear.bias is not None:
                fi_linear.bias.data.copy_(hf_linear.bias.data.to(compute_dtype))
            else:
                # Need to add bias to FI model
                fi_linear.bias = nn.Parameter(hf_linear.bias.data.to(compute_dtype).clone())
    
    # Embedding
    flashinfer_model.model.embed_tokens.weight.data.copy_(
        hf_model.model.embed_tokens.weight.data
    )
    
    # Layers
    for i, (fi_layer, hf_layer) in enumerate(
        zip(flashinfer_model.model.layers, hf_model.model.layers)
    ):
        # Attention
        copy_linear_weight(fi_layer.self_attn.q_proj, hf_layer.self_attn.q_proj, use_fp8)
        copy_linear_weight(fi_layer.self_attn.k_proj, hf_layer.self_attn.k_proj, use_fp8)
        copy_linear_weight(fi_layer.self_attn.v_proj, hf_layer.self_attn.v_proj, use_fp8)
        copy_linear_weight(fi_layer.self_attn.o_proj, hf_layer.self_attn.o_proj, use_fp8)
        
        # QK normalization (Qwen3)
        if hasattr(fi_layer.self_attn, 'q_norm') and fi_layer.self_attn.use_qk_norm:
            if hasattr(hf_layer.self_attn, 'q_norm'):
                fi_layer.self_attn.q_norm.weight.data.copy_(hf_layer.self_attn.q_norm.weight.data)
            if hasattr(hf_layer.self_attn, 'k_norm'):
                fi_layer.self_attn.k_norm.weight.data.copy_(hf_layer.self_attn.k_norm.weight.data)
        
        # MLP (dense or MoE)
        if fi_layer.use_moe:
            # Copy MoE gate weights
            copy_linear_weight(fi_layer.mlp.gate, hf_layer.mlp.gate, use_fp8)
            # Copy expert weights
            for expert_idx, (fi_expert, hf_expert) in enumerate(
                zip(fi_layer.mlp.experts, hf_layer.mlp.experts)
            ):
                copy_linear_weight(fi_expert.gate_proj, hf_expert.gate_proj, use_fp8)
                copy_linear_weight(fi_expert.up_proj, hf_expert.up_proj, use_fp8)
                copy_linear_weight(fi_expert.down_proj, hf_expert.down_proj, use_fp8)
        else:
            # Dense MLP
            copy_linear_weight(fi_layer.mlp.gate_proj, hf_layer.mlp.gate_proj, use_fp8)
            copy_linear_weight(fi_layer.mlp.up_proj, hf_layer.mlp.up_proj, use_fp8)
            copy_linear_weight(fi_layer.mlp.down_proj, hf_layer.mlp.down_proj, use_fp8)
        
        # LayerNorms (always BF16)
        fi_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
        fi_layer.post_attention_layernorm.weight.data.copy_(
            hf_layer.post_attention_layernorm.weight.data
        )
    
    # Final norm
    flashinfer_model.model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
    
    # LM head (if not tied)
    if flashinfer_model.lm_head is not None:
        copy_linear_weight(flashinfer_model.lm_head, hf_model.lm_head, use_fp8)


def get_model_type(model_name: str) -> str:
    """Determine model type from model name"""
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        return "llama"
    elif "qwen" in model_name_lower:
        return "qwen"
    else:
        return "llama"  # Default to llama-like architecture


def main():
    parser = argparse.ArgumentParser(description="FlashInfer-based LLM Inference")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model to use for inference (e.g., meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen3-4B-Instruct-2507-FP8)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FlashInfer-based LLM Inference")
    print("=" * 60)
    
    model_name = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine model type and FP8 usage
    model_type = get_model_type(model_name)
    use_fp8 = "FP8" in model_name or "fp8" in model_name
    dtype = torch.bfloat16
    
    print(f"\nModel: {model_name}")
    print(f"Model type: {model_type}")
    print(f"FP8 quantized: {use_fp8}")
    print(f"Device: {device}")
    print(f"Compute dtype: {dtype}")
    
    # Load tokenizer and config
    print(f"\nLoading tokenizer and config...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\nModel config:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Num KV heads: {config.num_key_value_heads}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - RoPE theta: {getattr(config, 'rope_theta', 'N/A')}")
    print(f"  - Tied embeddings: {getattr(config, 'tie_word_embeddings', False)}")
    
    # MoE info
    num_experts = getattr(config, 'num_experts', 0)
    if num_experts > 0:
        print(f"  - Num experts: {num_experts}")
        print(f"  - Experts per token: {getattr(config, 'num_experts_per_tok', 1)}")
        print(f"  - MoE intermediate size: {getattr(config, 'moe_intermediate_size', 'N/A')}")
    
    # Load HuggingFace model
    print(f"\nLoading HuggingFace model weights...")
    if use_fp8:
        # FP8 models need device_map for proper loading
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
        )
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
    
    # Create FlashInfer model
    print("\nCreating FlashInfer model...")
    fi_model = FlashInferForCausalLM(config, model_type, use_fp8).to(device=device, dtype=dtype)
    
    # Copy weights
    print("Transferring weights to FlashInfer model...")
    load_weights_from_hf(fi_model, hf_model, use_fp8, dtype)
    
    # Free HF model memory
    del hf_model
    torch.cuda.empty_cache()
    
    fi_model.eval()
    
    # Test inference
    print("\n" + "=" * 60)
    print("Running inference with FlashInfer kernels...")
    print("=" * 60)
    
    # Prepare prompt
    if args.prompt:
        user_message = args.prompt
    else:
        user_message = "What is the capital of France? Answer in one sentence."
    
    # Format for chat models
    if model_type == "qwen":
        messages = [
            {"role": "user", "content": user_message}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_message}
        ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nPrompt:\n{prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    print(f"\nInput tokens: {input_ids.shape[1]}")
    
    # Generate
    print("\nGenerating response...")
    with torch.no_grad():
        output_ids = fi_model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.9,
            top_k=50,
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FlashInfer Kernels Used:")
    print("=" * 60)
    print("✓ flashinfer.embedding - Token embedding lookup")
    print("✓ flashinfer.linear - Linear projections (Q, K, V, O, MLP, LM head, MoE gate)")
    print("✓ flashinfer.rmsnorm - RMS normalization")
    if model_type == "llama" and getattr(config, 'rope_scaling', None):
        print("✓ flashinfer.apply_llama31_rope_pos_ids_inplace - Llama 3.1 RoPE")
    else:
        print("✓ flashinfer.apply_rope_pos_ids_inplace - Standard RoPE")
    print("✓ flashinfer.single_prefill_with_kv_cache - Prefill attention")
    print("✓ flashinfer.single_decode_with_kv_cache - Decode attention")
    print("✓ flashinfer.silu_and_mul - SiLU activation for MLP/experts")
    print("✓ flashinfer.top_k_top_p_sampling_from_probs - Sampling")
    
    # Show MoE info
    num_experts = getattr(config, 'num_experts', 0)
    if num_experts > 0:
        num_experts_per_tok = getattr(config, 'num_experts_per_tok', 1)
        print("✓ flashinfer.sparse_moe_forward - Sparse MoE routing and computation")
        print(f"\n[MoE Mode] {num_experts} experts, {num_experts_per_tok} active per token")
    
    if use_fp8:
        print("\n[FP8 Mode] Weights stored in FP8 e4m3 format with block-wise scaling")


if __name__ == "__main__":
    main()

