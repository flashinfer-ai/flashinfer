#!/usr/bin/env python3
"""
FlashInfer Tensor-Parallel LLM Inference Example

This script demonstrates how to perform tensor-parallel LLM inference using
FlashInfer kernels with Megatron-style tensor parallelism:

- ColumnParallelLinear: Q, K, V, gate_proj, up_proj split across GPUs
- RowParallelLinear: o_proj, down_proj with all-reduce
- VocabParallelEmbedding: Vocabulary split across GPUs
- Expert Parallelism: MoE experts distributed across GPUs

Launch with torchrun:
    torchrun --nproc_per_node=2 llm_inference_tp.py --model Qwen/Qwen2.5-1.5B-Instruct
    torchrun --nproc_per_node=4 llm_inference_tp.py --model Qwen/Qwen3-30B-A3B-Instruct-2507
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from typing import Optional, Tuple, List
import flashinfer


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    return dist.get_world_size(), dist.get_rank(), local_rank


class TensorParallelLlamaRMSNorm(nn.Module):
    """RMSNorm using FlashInfer kernel (not parallelized, replicated on all GPUs)"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return flashinfer.rmsnorm(x, self.weight, self.eps)


class TensorParallelMLP(nn.Module):
    """MLP with tensor parallelism using FlashInfer operators.
    
    gate_proj and up_proj use ColumnParallelLinear (split output)
    down_proj uses RowParallelLinear (split input, all-reduce via flashinfer.all_reduce)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Column parallel: split output dimension
        self.gate_proj = flashinfer.ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.up_proj = flashinfer.ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        # Row parallel: split input dimension, all-reduce output via flashinfer.all_reduce
        self.down_proj = flashinfer.RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True, reduce_results=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate_up = torch.cat([gate, up], dim=-1)
        hidden = flashinfer.silu_and_mul(gate_up)
        return self.down_proj(hidden)


class TensorParallelAttention(nn.Module):
    """Attention with tensor parallelism.
    
    Q, K, V projections use ColumnParallelLinear
    O projection uses RowParallelLinear
    """
    
    def __init__(self, config, layer_idx: int, model_type: str = "llama"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.model_type = model_type
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        
        tp_size = flashinfer.get_tensor_parallel_world_size()
        
        # Ensure heads are divisible by tp_size
        assert self.num_heads % tp_size == 0, f"num_heads ({self.num_heads}) must be divisible by tp_size ({tp_size})"
        assert self.num_key_value_heads % tp_size == 0, f"num_kv_heads ({self.num_key_value_heads}) must be divisible by tp_size ({tp_size})"
        
        self.num_heads_per_partition = self.num_heads // tp_size
        self.num_kv_heads_per_partition = self.num_key_value_heads // tp_size
        self.num_key_value_groups = self.num_heads_per_partition // self.num_kv_heads_per_partition
        
        attention_bias = getattr(config, 'attention_bias', False)
        
        # Column parallel projections
        self.q_proj = flashinfer.ColumnParallelLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias, gather_output=False
        )
        self.k_proj = flashinfer.ColumnParallelLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias, gather_output=False
        )
        self.v_proj = flashinfer.ColumnParallelLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias, gather_output=False
        )
        # Row parallel output projection
        self.o_proj = flashinfer.RowParallelLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False, input_is_parallel=True, reduce_results=True
        )
        
        # QK normalization (Qwen3)
        config_model_type = getattr(config, 'model_type', '')
        self.use_qk_norm = model_type == "qwen" and config_model_type in ('qwen3', 'qwen3_moe')
        if self.use_qk_norm:
            self.q_norm = TensorParallelLlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = TensorParallelLlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # RoPE parameters
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.rope_scaling = getattr(config, 'rope_scaling', None)
        
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
        
        # Project (output is already partitioned)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention (using per-partition head counts)
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)
        
        # Apply QK normalization if enabled
        if self.use_qk_norm:
            q_shape, k_shape = q.shape, k.shape
            q = self.q_norm(q.view(-1, self.head_dim)).view(q_shape)
            k = self.k_norm(k.view(-1, self.head_dim)).view(k_shape)
        
        # Apply RoPE
        q_rope = q.view(batch_size * seq_len, self.num_heads_per_partition, self.head_dim)
        k_rope = k.view(batch_size * seq_len, self.num_kv_heads_per_partition, self.head_dim)
        pos_ids = position_ids.view(-1).to(torch.int32)
        
        if self.use_llama31_rope:
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
            flashinfer.apply_rope_pos_ids_inplace(
                q_rope, k_rope, pos_ids,
                rotary_dim=self.head_dim,
                interleave=False,
                rope_scale=self.rope_scale,
                rope_theta=self.rope_theta,
            )
        
        q = q_rope.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k_rope.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)
        
        # KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        
        new_kv_cache = (k, v)
        kv_len = k.shape[1]
        
        # GQA expansion
        if self.num_key_value_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1)
            k = k.reshape(batch_size, kv_len, self.num_heads_per_partition, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1)
            v = v.reshape(batch_size, kv_len, self.num_heads_per_partition, self.head_dim)
        
        # FlashInfer attention
        q_fi = q.squeeze(0)
        k_fi = k.squeeze(0)
        v_fi = v.squeeze(0)
        
        if is_prefill or seq_len > 1:
            attn_output = flashinfer.single_prefill_with_kv_cache(
                q_fi, k_fi, v_fi, causal=True, kv_layout="NHD"
            )
        else:
            q_decode = q_fi.squeeze(0)
            attn_output = flashinfer.single_decode_with_kv_cache(
                q_decode, k_fi, v_fi, kv_layout="NHD"
            )
            attn_output = attn_output.unsqueeze(0)
        
        # Reshape and project output (row parallel with all-reduce)
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads_per_partition * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, new_kv_cache


class TensorParallelDecoderLayer(nn.Module):
    """Decoder layer with tensor parallelism."""
    
    def __init__(self, config, layer_idx: int, model_type: str = "llama"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TensorParallelAttention(config, layer_idx, model_type)
        
        # Check for MoE
        num_experts = getattr(config, 'num_experts', 0)
        decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1)
        mlp_only_layers = getattr(config, 'mlp_only_layers', [])
        
        self.use_moe = (
            num_experts > 0 and
            layer_idx not in mlp_only_layers and
            (layer_idx + 1) % decoder_sparse_step == 0
        )
        
        if self.use_moe:
            # Use tensor parallelism WITHIN each expert (not expert parallelism)
            # Each expert's gate_proj/up_proj are ColumnParallel, down_proj is RowParallel
            moe_intermediate_size = getattr(config, 'moe_intermediate_size', config.intermediate_size)
            self.mlp = flashinfer.TensorParallelSparseMoeBlock(
                hidden_size=config.hidden_size,
                intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                top_k=config.num_experts_per_tok,
                norm_topk_prob=getattr(config, 'norm_topk_prob', True),
            )
        else:
            # Dense MLP with tensor parallelism
            self.mlp = TensorParallelMLP(config.hidden_size, config.intermediate_size)
        
        self.input_layernorm = TensorParallelLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TensorParallelLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv_cache = self.self_attn(hidden_states, position_ids, kv_cache, is_prefill)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.use_moe:
            hidden_states, _ = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states, new_kv_cache


class TensorParallelModel(nn.Module):
    """Model with tensor parallelism."""
    
    def __init__(self, config, model_type: str = "llama"):
        super().__init__()
        self.config = config
        
        # Vocab parallel embedding
        self.embed_tokens = flashinfer.VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            TensorParallelDecoderLayer(config, i, model_type)
            for i in range(config.num_hidden_layers)
        ])
        
        self.norm = TensorParallelLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
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
            hidden_states, new_kv_cache = layer(hidden_states, position_ids, kv_cache, is_prefill)
            new_kv_caches.append(new_kv_cache)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_kv_caches


class TensorParallelForCausalLM(nn.Module):
    """Causal LM with tensor parallelism."""
    
    def __init__(self, config, model_type: str = "llama"):
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
        
        self.model = TensorParallelModel(config, model_type)
        
        if not self.tie_word_embeddings:
            # Column parallel for LM head (gather output for final logits)
            self.lm_head = flashinfer.ColumnParallelLinear(
                config.hidden_size, config.vocab_size, bias=False, gather_output=True
            )
        else:
            self.lm_head = None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        is_prefill: bool = True,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states, new_kv_caches = self.model(input_ids, position_ids, kv_caches, is_prefill)
        
        if self.tie_word_embeddings:
            # Use embedding weight, need to gather for full vocabulary
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
            logits = flashinfer.all_gather(logits, dim=-1)
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
                next_token = flashinfer.top_k_top_p_sampling_from_probs(probs_f32, top_k, top_p)
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
            logits, kv_caches = self.forward(next_token, position_ids, kv_caches, is_prefill=False)
        
        return generated_ids


def load_weights_tp(model: TensorParallelForCausalLM, hf_model, compute_dtype: torch.dtype):
    """Load weights with tensor parallelism sharding."""
    tp_rank = flashinfer.get_tensor_parallel_rank()
    tp_size = flashinfer.get_tensor_parallel_world_size()
    
    # Embedding
    model.model.embed_tokens.load_weight_shard(hf_model.model.embed_tokens.weight.data)
    
    for i, (tp_layer, hf_layer) in enumerate(zip(model.model.layers, hf_model.model.layers)):
        # Attention projections
        tp_layer.self_attn.q_proj.load_weight_shard(
            hf_layer.self_attn.q_proj.weight.data.to(compute_dtype),
            hf_layer.self_attn.q_proj.bias.data.to(compute_dtype) if hf_layer.self_attn.q_proj.bias is not None else None
        )
        tp_layer.self_attn.k_proj.load_weight_shard(
            hf_layer.self_attn.k_proj.weight.data.to(compute_dtype),
            hf_layer.self_attn.k_proj.bias.data.to(compute_dtype) if hf_layer.self_attn.k_proj.bias is not None else None
        )
        tp_layer.self_attn.v_proj.load_weight_shard(
            hf_layer.self_attn.v_proj.weight.data.to(compute_dtype),
            hf_layer.self_attn.v_proj.bias.data.to(compute_dtype) if hf_layer.self_attn.v_proj.bias is not None else None
        )
        tp_layer.self_attn.o_proj.load_weight_shard(
            hf_layer.self_attn.o_proj.weight.data.to(compute_dtype)
        )
        
        # QK norm (Qwen3)
        if hasattr(tp_layer.self_attn, 'q_norm') and tp_layer.self_attn.use_qk_norm:
            if hasattr(hf_layer.self_attn, 'q_norm'):
                tp_layer.self_attn.q_norm.weight.data.copy_(hf_layer.self_attn.q_norm.weight.data)
            if hasattr(hf_layer.self_attn, 'k_norm'):
                tp_layer.self_attn.k_norm.weight.data.copy_(hf_layer.self_attn.k_norm.weight.data)
        
        # MLP
        if tp_layer.use_moe:
            # MoE gate (replicated on all GPUs)
            tp_layer.mlp.gate.weight.data.copy_(hf_layer.mlp.gate.weight.data.to(compute_dtype))
            
            # All experts with TP sharding within each expert
            for expert_idx in range(len(tp_layer.mlp.experts)):
                tp_expert = tp_layer.mlp.experts[expert_idx]
                hf_expert = hf_layer.mlp.experts[expert_idx]
                
                # gate_proj and up_proj: ColumnParallel (shard output dim)
                tp_expert.gate_proj.load_weight_shard(hf_expert.gate_proj.weight.data.to(compute_dtype))
                tp_expert.up_proj.load_weight_shard(hf_expert.up_proj.weight.data.to(compute_dtype))
                # down_proj: RowParallel (shard input dim)
                tp_expert.down_proj.load_weight_shard(hf_expert.down_proj.weight.data.to(compute_dtype))
        else:
            # Dense MLP
            tp_layer.mlp.gate_proj.load_weight_shard(hf_layer.mlp.gate_proj.weight.data.to(compute_dtype))
            tp_layer.mlp.up_proj.load_weight_shard(hf_layer.mlp.up_proj.weight.data.to(compute_dtype))
            tp_layer.mlp.down_proj.load_weight_shard(hf_layer.mlp.down_proj.weight.data.to(compute_dtype))
        
        # Layer norms (replicated)
        tp_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
        tp_layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)
    
    # Final norm
    model.model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
    
    # LM head
    if model.lm_head is not None:
        model.lm_head.load_weight_shard(hf_model.lm_head.weight.data.to(compute_dtype))


def get_model_type(model_name: str) -> str:
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        return "llama"
    elif "qwen" in model_name_lower:
        return "qwen"
    return "llama"


def main():
    parser = argparse.ArgumentParser(description="FlashInfer Tensor-Parallel LLM Inference")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()
    
    # Setup distributed
    world_size, rank, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Load config first to get hidden_dim for workspace initialization
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden_dim = config.hidden_size
    
    # Initialize FlashInfer tensor parallelism with custom all-reduce workspace
    flashinfer.init_tensor_parallel(
        tp_size=world_size,
        tp_rank=rank,
        max_token_num=8192,  # Workspace size for custom all-reduce
        hidden_dim=hidden_dim,
        use_flashinfer_custom_ar=True,  # Use FlashInfer's optimized all-reduce
    )
    
    if rank == 0:
        print("=" * 60)
        print("FlashInfer Tensor-Parallel LLM Inference")
        print("=" * 60)
        print(f"\nModel: {args.model}")
        print(f"Tensor Parallel Size: {world_size}")
        print(f"Device: {device}")
        print(f"FlashInfer Custom All-Reduce: {flashinfer.is_using_flashinfer_custom_ar()}")
    
    model_name = args.model
    model_type = get_model_type(model_name)
    dtype = torch.bfloat16
    
    # Load tokenizer (config already loaded above for TP initialization)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if rank == 0:
        print(f"\nModel config:")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")
        print(f"  - Num attention heads: {config.num_attention_heads}")
        print(f"  - Num KV heads: {config.num_key_value_heads}")
        print(f"  - Vocab size: {config.vocab_size}")
        
        num_experts = getattr(config, 'num_experts', 0)
        if num_experts > 0:
            print(f"  - Num experts: {num_experts}")
            print(f"  - Experts per token: {getattr(config, 'num_experts_per_tok', 1)}")
    
    # Load HF model on rank 0, broadcast to all
    if rank == 0:
        print(f"\nLoading HuggingFace model...")
    
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    
    # Create tensor-parallel model
    if rank == 0:
        print("Creating tensor-parallel FlashInfer model...")
    
    tp_model = TensorParallelForCausalLM(config, model_type).to(device=device, dtype=dtype)
    
    # Load weights with sharding
    if rank == 0:
        print("Loading weights with tensor parallelism...")
    
    load_weights_tp(tp_model, hf_model, dtype)
    
    # Free HF model
    del hf_model
    torch.cuda.empty_cache()
    
    tp_model.eval()
    
    # Synchronize before inference
    dist.barrier()
    
    # Test inference
    if rank == 0:
        print("\n" + "=" * 60)
        print("Running tensor-parallel inference...")
        print("=" * 60)
    
    # Prepare prompt
    user_message = args.prompt or "What is the capital of France? Answer in one sentence."
    
    if model_type == "qwen":
        messages = [{"role": "user", "content": user_message}]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_message}
        ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if rank == 0:
        print(f"\nPrompt:\n{prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    if rank == 0:
        print(f"\nInput tokens: {input_ids.shape[1]}")
        print("\nGenerating response...")
    
    # Generate
    with torch.no_grad():
        output_ids = tp_model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.9,
            top_k=50,
        )
    
    # Only rank 0 prints result
    if rank == 0:
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nGenerated text:\n{generated_text}")
        
        print("\n" + "=" * 60)
        print("Tensor Parallel Summary:")
        print("=" * 60)
        print(f"✓ Tensor Parallel Size: {world_size}")
        print("✓ ColumnParallelLinear: Q, K, V, gate_proj, up_proj")
        print("✓ RowParallelLinear: o_proj, down_proj")
        print("✓ VocabParallelEmbedding: Token embeddings")
        
        if flashinfer.is_using_flashinfer_custom_ar():
            print("✓ flashinfer.all_reduce: Using FlashInfer trtllm_allreduce_fusion")
        else:
            print("✓ flashinfer.all_reduce: Using NCCL (fallback)")
        print("✓ flashinfer.all_gather: Communication primitive")
        
        num_experts = getattr(config, 'num_experts', 0)
        if num_experts > 0:
            print(f"✓ MoE with TP within experts: {num_experts} experts, each expert sharded across {world_size} GPUs")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

