"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Megatron-style Tensor Parallelism for LLM inference.

This module provides tensor-parallel linear layers following the Megatron-LM style:
- ColumnParallelLinear: Splits output dimension across GPUs
- RowParallelLinear: Splits input dimension across GPUs  
- VocabParallelEmbedding: Splits vocabulary across GPUs
- TensorParallelSparseMoeBlock: MoE with TP within each expert

Communication uses FlashInfer's optimized all-reduce when available.
"""

from typing import Optional, Tuple, List, Callable, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# Global tensor parallel state
_TENSOR_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_TENSOR_PARALLEL_WORLD_SIZE: int = 1
_TENSOR_PARALLEL_RANK: int = 0

# FlashInfer custom all-reduce state
_USE_FLASHINFER_CUSTOM_AR: bool = False
_FLASHINFER_AR_WORKSPACE: Optional[torch.Tensor] = None
_FLASHINFER_AR_WORKSPACE_PTRS: Optional[torch.Tensor] = None
_FLASHINFER_AR_METADATA: Optional[Dict[str, Any]] = None


def init_tensor_parallel(
    tp_size: int = 1,
    tp_rank: int = 0,
    tp_group: Optional[dist.ProcessGroup] = None,
    max_token_num: int = 8192,
    hidden_dim: int = 4096,
    use_flashinfer_custom_ar: bool = True,
) -> None:
    """Initialize tensor parallelism state.
    
    Args:
        tp_size: Number of tensor parallel ranks (GPUs)
        tp_rank: This process's rank in the tensor parallel group
        tp_group: Optional pre-initialized process group
        max_token_num: Maximum tokens for FlashInfer custom all-reduce workspace
        hidden_dim: Hidden dimension for FlashInfer custom all-reduce workspace
        use_flashinfer_custom_ar: Whether to use FlashInfer's optimized custom all-reduce
    """
    global _TENSOR_PARALLEL_GROUP, _TENSOR_PARALLEL_WORLD_SIZE, _TENSOR_PARALLEL_RANK
    global _USE_FLASHINFER_CUSTOM_AR, _FLASHINFER_AR_WORKSPACE, _FLASHINFER_AR_WORKSPACE_PTRS
    global _FLASHINFER_AR_METADATA
    
    _TENSOR_PARALLEL_GROUP = tp_group
    _TENSOR_PARALLEL_WORLD_SIZE = tp_size
    _TENSOR_PARALLEL_RANK = tp_rank
    
    # Initialize FlashInfer custom all-reduce if requested and tp_size > 1
    if use_flashinfer_custom_ar and tp_size > 1:
        try:
            from .comm import (
                trtllm_create_ipc_workspace_for_all_reduce_fusion,
                AllReduceFusionPattern,
            )
            
            # Create IPC workspace for all-reduce fusion
            _, workspace_tensor, metadata = trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=tp_rank,
                tp_size=tp_size,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                use_fp32_lamport=False,
                group=tp_group,
                create_metadata=True,
            )
            
            _FLASHINFER_AR_WORKSPACE = workspace_tensor
            _FLASHINFER_AR_WORKSPACE_PTRS = metadata.get("workspace_ptrs")
            _FLASHINFER_AR_METADATA = metadata
            _USE_FLASHINFER_CUSTOM_AR = True
            
        except Exception as e:
            # Fall back to NCCL if FlashInfer custom AR setup fails
            import warnings
            warnings.warn(
                f"FlashInfer custom all-reduce initialization failed: {e}. "
                f"Falling back to NCCL all-reduce."
            )
            _USE_FLASHINFER_CUSTOM_AR = False


def get_tensor_parallel_world_size() -> int:
    """Get tensor parallel world size."""
    return _TENSOR_PARALLEL_WORLD_SIZE


def get_tensor_parallel_rank() -> int:
    """Get tensor parallel rank."""
    return _TENSOR_PARALLEL_RANK


def get_tensor_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get tensor parallel process group."""
    return _TENSOR_PARALLEL_GROUP


def is_using_flashinfer_custom_ar() -> bool:
    """Check if FlashInfer custom all-reduce is being used."""
    return _USE_FLASHINFER_CUSTOM_AR


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor across tensor parallel group.
    
    Uses FlashInfer's optimized trtllm_allreduce_fusion when available,
    otherwise falls back to NCCL via torch.distributed.
    
    Args:
        tensor: Input tensor to all-reduce (modified in-place)
        
    Returns:
        All-reduced tensor
    """
    if _TENSOR_PARALLEL_WORLD_SIZE == 1:
        return tensor
    
    if _USE_FLASHINFER_CUSTOM_AR and _FLASHINFER_AR_WORKSPACE_PTRS is not None:
        try:
            from .comm import trtllm_allreduce_fusion, AllReduceFusionPattern
            
            # Ensure tensor is contiguous
            tensor = tensor.contiguous()
            
            # Get dimensions
            if tensor.dim() == 2:
                token_num, hidden_dim = tensor.shape
            elif tensor.dim() == 3:
                batch, seq, hidden_dim = tensor.shape
                token_num = batch * seq
                tensor = tensor.view(token_num, hidden_dim)
            else:
                # Fall back to NCCL for unsupported shapes
                raise ValueError(f"Unsupported tensor dim: {tensor.dim()}")
            
            # Create output tensor
            out = torch.empty_like(tensor)
            
            # Use FlashInfer custom all-reduce
            trtllm_allreduce_fusion(
                allreduce_in=tensor,
                world_size=_TENSOR_PARALLEL_WORLD_SIZE,
                world_rank=_TENSOR_PARALLEL_RANK,
                token_num=token_num,
                hidden_dim=hidden_dim,
                workspace_ptrs=_FLASHINFER_AR_WORKSPACE_PTRS,
                launch_with_pdl=True,
                trigger_completion_at_end=True,
                fp32_acc=False,
                pattern_code=AllReduceFusionPattern.kAllReduce,
                use_oneshot=None,
                allreduce_out=out,
                residual_in=None,
                residual_out=None,
                norm_out=None,
                quant_out=None,
                scale_out=None,
                rms_gamma=None,
                rms_eps=None,
                scale_factor=None,
                layout_code=None,
                metadata=_FLASHINFER_AR_METADATA,
            )
            
            return out
            
        except Exception:
            # Fall back to NCCL on any error
            pass
    
    # Fall back to NCCL
    tensor = tensor.contiguous()
    if _TENSOR_PARALLEL_GROUP is not None:
        dist.all_reduce(tensor, group=_TENSOR_PARALLEL_GROUP)
    else:
        dist.all_reduce(tensor)
    return tensor


def all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather tensor across tensor parallel group.
    
    Uses NCCL backend via torch.distributed.
    
    Args:
        tensor: Input tensor to all-gather
        dim: Dimension along which to concatenate gathered tensors
        
    Returns:
        Gathered and concatenated tensor
    """
    if _TENSOR_PARALLEL_WORLD_SIZE == 1:
        return tensor
    
    world_size = _TENSOR_PARALLEL_WORLD_SIZE
    
    # Gather tensors from all ranks
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    if _TENSOR_PARALLEL_GROUP is not None:
        dist.all_gather(tensor_list, tensor.contiguous(), group=_TENSOR_PARALLEL_GROUP)
    else:
        dist.all_gather(tensor_list, tensor.contiguous())
    
    # Concatenate along the specified dimension
    return torch.cat(tensor_list, dim=dim)


def reduce_scatter(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reduce-scatter tensor across tensor parallel group.
    
    Uses NCCL backend via torch.distributed.
    
    Args:
        tensor: Input tensor to reduce-scatter
        dim: Dimension along which to scatter
        
    Returns:
        Reduced and scattered tensor (1/tp_size of input along dim)
    """
    if _TENSOR_PARALLEL_WORLD_SIZE == 1:
        return tensor
    
    world_size = _TENSOR_PARALLEL_WORLD_SIZE
    
    # Split tensor and create output
    input_list = list(torch.chunk(tensor.contiguous(), world_size, dim=dim))
    output = torch.empty_like(input_list[0])
    
    if _TENSOR_PARALLEL_GROUP is not None:
        dist.reduce_scatter(output, input_list, group=_TENSOR_PARALLEL_GROUP)
    else:
        dist.reduce_scatter(output, input_list)
    
    return output


# Backward-compatible aliases
tensor_parallel_all_reduce = all_reduce
tensor_parallel_all_gather = all_gather
tensor_parallel_reduce_scatter = reduce_scatter


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Copy input to tensor parallel region (identity in forward, all-reduce in backward)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        return input_
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return all_reduce(grad_output.contiguous())


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """All-reduce from tensor parallel region (all-reduce in forward, identity in backward)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        return all_reduce(input_.contiguous())
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather from tensor parallel region (all-gather in forward, split in backward)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        return all_gather(input_, dim)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        dim = ctx.dim
        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()
        
        # Split gradient along the gather dimension
        chunks = torch.chunk(grad_output, world_size, dim=dim)
        return chunks[rank].contiguous(), None


def copy_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Copy to tensor parallel region (identity forward, all-reduce backward)."""
    return _CopyToTensorParallelRegion.apply(input_)


def reduce_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Reduce from tensor parallel region (all-reduce forward, identity backward)."""
    return _ReduceFromTensorParallelRegion.apply(input_)


def gather_from_tensor_parallel_region(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather from tensor parallel region (all-gather forward, split backward)."""
    return _GatherFromTensorParallelRegion.apply(input_, dim)


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.
    
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension (output dimension) as A = [A_1, ..., A_p].
    
    Each GPU computes Y_i = XA_i, which is a portion of the full output.
    
    This is used for:
    - Q, K, V projections in attention
    - gate_proj, up_proj in MLP (first layer)
    - MoE gate
    
    Args:
        in_features: Input dimension
        out_features: Output dimension (will be divided by tp_size)
        bias: Whether to use bias
        gather_output: If True, all-gather output to make it available on all GPUs
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        tp_size = get_tensor_parallel_world_size()
        tp_rank = get_tensor_parallel_rank()
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Divide output dimension
        self.out_features_per_partition = divide(out_features, tp_size)
        
        # Create weight for this partition
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Copy input to all GPUs (identity in forward)
        input_parallel = copy_to_tensor_parallel_region(input_)
        
        # Compute local output
        output = F.linear(input_parallel, self.weight, self.bias)
        
        # Optionally gather output across all GPUs
        if self.gather_output:
            output = gather_from_tensor_parallel_region(output, dim=-1)
        
        return output
    
    def load_weight_shard(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None):
        """Load weight shard for this TP rank from the full weight tensor.
        
        Args:
            full_weight: Full weight tensor of shape [out_features, in_features]
            full_bias: Optional full bias tensor of shape [out_features]
        """
        # Shard along output dimension
        start_idx = self.tp_rank * self.out_features_per_partition
        end_idx = start_idx + self.out_features_per_partition
        
        self.weight.data.copy_(full_weight[start_idx:end_idx, :])
        
        if full_bias is not None and self.bias is not None:
            self.bias.data.copy_(full_bias[start_idx:end_idx])


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Merged Column Parallel Linear for fused QKV or gate+up projections.
    
    This handles the case where multiple column-parallel outputs are merged,
    like QKV projection where Q, K, V are concatenated but have different sizes
    (Q has num_heads, K/V have num_kv_heads).
    
    Args:
        in_features: Input dimension
        output_sizes: List of output sizes for each merged tensor
        bias: Whether to use bias
        gather_output: If True, all-gather output
    """
    
    def __init__(
        self,
        in_features: int,
        output_sizes: List[int],
        bias: bool = False,
        gather_output: bool = False,
    ):
        self.output_sizes = output_sizes
        total_output = sum(output_sizes)
        super().__init__(in_features, total_output, bias, gather_output)
        
        # Calculate partition sizes for each output
        tp_size = get_tensor_parallel_world_size()
        self.output_partition_sizes = [divide(size, tp_size) for size in output_sizes]
    
    def load_weight_shard(
        self,
        weights: List[torch.Tensor],
        biases: Optional[List[torch.Tensor]] = None,
    ):
        """Load weight shards from list of weight tensors.
        
        Args:
            weights: List of full weight tensors for each output
            biases: Optional list of full bias tensors
        """
        weight_shards = []
        bias_shards = []
        
        for i, (weight, part_size) in enumerate(zip(weights, self.output_partition_sizes)):
            start_idx = self.tp_rank * part_size
            end_idx = start_idx + part_size
            weight_shards.append(weight[start_idx:end_idx, :])
            
            if biases is not None and biases[i] is not None and self.bias is not None:
                bias_shards.append(biases[i][start_idx:end_idx])
        
        self.weight.data.copy_(torch.cat(weight_shards, dim=0))
        
        if bias_shards:
            self.bias.data.copy_(torch.cat(bias_shards, dim=0))


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    
    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension (input dimension) as A = [A_1; A_2; ...; A_p].
    
    The input X is split across GPUs as X = [X_1, X_2, ..., X_p], and
    each GPU computes Y_i = X_i * A_i. Results are all-reduced: Y = sum(Y_i).
    
    This is used for:
    - o_proj in attention (output projection)
    - down_proj in MLP (second layer)
    
    Args:
        in_features: Input dimension (will be divided by tp_size)
        out_features: Output dimension
        bias: Whether to use bias (applied after all-reduce)
        input_is_parallel: If True, input is already split across GPUs
        reduce_results: If True, all-reduce the results
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        
        tp_size = get_tensor_parallel_world_size()
        tp_rank = get_tensor_parallel_rank()
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Divide input dimension
        self.in_features_per_partition = divide(in_features, tp_size)
        
        # Create weight for this partition
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        
        if bias:
            # Bias is not parallelized, applied after all-reduce
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # If input is not parallel, split it
        if not self.input_is_parallel:
            # Split input along last dimension
            input_list = torch.chunk(input_, self.tp_size, dim=-1)
            input_ = input_list[self.tp_rank]
        
        # Compute local output (without bias)
        output = F.linear(input_, self.weight)
        
        # All-reduce across tensor parallel group using FlashInfer's all_reduce
        if self.reduce_results:
            output = all_reduce(output)
        
        # Add bias after all-reduce
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def load_weight_shard(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None):
        """Load weight shard for this TP rank from the full weight tensor.
        
        Args:
            full_weight: Full weight tensor of shape [out_features, in_features]
            full_bias: Optional full bias tensor of shape [out_features]
        """
        # Shard along input dimension (dim=1 of weight)
        start_idx = self.tp_rank * self.in_features_per_partition
        end_idx = start_idx + self.in_features_per_partition
        
        self.weight.data.copy_(full_weight[:, start_idx:end_idx])
        
        # Bias is not sharded
        if full_bias is not None and self.bias is not None:
            self.bias.data.copy_(full_bias)


class VocabParallelEmbedding(nn.Module):
    """Embedding layer with vocabulary parallelism.
    
    The embedding table is split across GPUs along the vocabulary dimension.
    Each GPU holds a shard of the vocabulary: [vocab_start:vocab_end].
    
    For tokens outside the local shard, the embedding is zero, and
    all-reduce combines the results.
    
    Args:
        num_embeddings: Vocabulary size (will be padded to be divisible by tp_size)
        embedding_dim: Embedding dimension
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        tp_size = get_tensor_parallel_world_size()
        tp_rank = get_tensor_parallel_rank()
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Pad vocabulary to be divisible by tp_size
        self.num_embeddings_padded = (
            (num_embeddings + tp_size - 1) // tp_size * tp_size
        )
        self.num_embeddings_per_partition = self.num_embeddings_padded // tp_size
        
        # Calculate vocabulary range for this partition
        self.vocab_start_idx = tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = min(
            (tp_rank + 1) * self.num_embeddings_per_partition,
            num_embeddings
        )
        
        # Actual embeddings stored on this partition
        num_local_embeddings = self.vocab_end_idx - self.vocab_start_idx
        
        self.weight = nn.Parameter(
            torch.empty(num_local_embeddings, embedding_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.weight)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(input_, self.weight)
        
        # Mask for tokens in this partition's range
        input_mask = (input_ >= self.vocab_start_idx) & (input_ < self.vocab_end_idx)
        
        # Shift indices to local range
        masked_input = input_.clone()
        masked_input[~input_mask] = 0  # Dummy index for out-of-range tokens
        masked_input = masked_input - self.vocab_start_idx
        
        # Look up embeddings
        output = F.embedding(masked_input, self.weight)
        
        # Zero out embeddings for out-of-range tokens
        output = output * input_mask.unsqueeze(-1).to(output.dtype)
        
        # All-reduce to combine embeddings from all partitions using FlashInfer's all_reduce
        output = all_reduce(output)
        
        return output
    
    def load_weight_shard(self, full_weight: torch.Tensor):
        """Load weight shard for this TP rank from the full embedding table.
        
        Args:
            full_weight: Full embedding table of shape [num_embeddings, embedding_dim]
        """
        self.weight.data.copy_(full_weight[self.vocab_start_idx:self.vocab_end_idx, :])


class TensorParallelMLP(nn.Module):
    """MLP with tensor parallelism using FlashInfer operators.
    
    gate_proj and up_proj use ColumnParallelLinear (split output)
    down_proj uses RowParallelLinear (split input, all-reduce output)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        from . import silu_and_mul
        self.silu_and_mul = silu_and_mul
        
        # Column parallel: split output dimension
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        # Row parallel: split input dimension, all-reduce output
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True, reduce_results=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate_up = torch.cat([gate, up], dim=-1)
        hidden = self.silu_and_mul(gate_up)
        return self.down_proj(hidden)


class TensorParallelSparseMoeBlock(nn.Module):
    """Tensor-parallel Sparse Mixture of Experts block.
    
    Uses tensor parallelism WITHIN each expert (not expert parallelism).
    Each expert's MLP layers are sharded across GPUs:
    - gate_proj, up_proj: ColumnParallel (output split)
    - down_proj: RowParallel (input split, all-reduce)
    
    All GPUs have all experts, but each expert's computation is distributed.
    
    Args:
        hidden_size: Hidden dimension
        intermediate_size: MLP intermediate dimension per expert
        num_experts: Total number of experts
        top_k: Number of experts per token
        norm_topk_prob: Whether to normalize top-k probabilities
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        
        tp_size = get_tensor_parallel_world_size()
        tp_rank = get_tensor_parallel_rank()
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Router gate (replicated on all GPUs for consistent routing)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # All experts with tensor parallelism within each expert
        self.experts = nn.ModuleList([
            TensorParallelMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states_flat.shape[0]
        
        # Compute router logits (same on all GPUs due to replicated gate)
        router_logits = self.gate(hidden_states_flat)
        
        # Compute routing weights
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # One-hot encode selected experts
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        # Process each expert that has tokens assigned
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            
            if len(top_x) == 0:
                continue
            
            # Get input states for this expert
            expert_input = hidden_states_flat[top_x]
            
            # Compute expert output (TP happens inside TensorParallelMLP)
            # Each expert's down_proj does all-reduce internally
            expert_output = expert(expert_input)
            
            # Weight by routing weights
            weights = routing_weights[top_x, idx].unsqueeze(-1)
            weighted_output = expert_output * weights
            
            # Add to final output (no additional all-reduce needed,
            # since down_proj already did all-reduce)
            final_hidden_states.index_add_(0, top_x, weighted_output)
        
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits

