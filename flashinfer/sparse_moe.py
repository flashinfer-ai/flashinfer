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

Sparse Mixture of Experts (MoE) operations.
"""

from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_moe_forward(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_fn: Callable[[torch.Tensor, int], torch.Tensor],
    num_experts: int,
    top_k: int,
    norm_topk_prob: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute sparse MoE forward pass.
    
    This function implements the core MoE routing and expert computation logic.
    It selects top-k experts for each token and computes a weighted combination
    of their outputs.
    
    Parameters
    ----------
    hidden_states : torch.Tensor
        Input tensor of shape [batch_size, seq_len, hidden_size].
    gate_weight : torch.Tensor
        Router gate weight tensor of shape [num_experts, hidden_size].
    expert_fn : Callable[[torch.Tensor, int], torch.Tensor]
        A function that computes the expert output given input states and expert index.
        Signature: expert_fn(input: Tensor, expert_idx: int) -> Tensor
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to select per token.
    norm_topk_prob : bool
        Whether to normalize the top-k probabilities to sum to 1.
        
    Returns
    -------
    output : torch.Tensor
        Output tensor of shape [batch_size, seq_len, hidden_size].
    router_logits : torch.Tensor
        Router logits of shape [batch_size * seq_len, num_experts].
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    
    # Compute router logits
    router_logits = F.linear(hidden_states_flat, gate_weight)
    
    # Compute routing weights with softmax
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # Normalize top-k probabilities
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    
    routing_weights = routing_weights.to(hidden_states.dtype)
    
    # Initialize output
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # One-hot encode selected experts
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    # Process each expert that has tokens assigned
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        
        # Get hidden states for this expert
        current_state = hidden_states_flat[top_x]
        
        # Compute expert output weighted by routing weight
        current_hidden_states = expert_fn(current_state, int(expert_idx)) * routing_weights[top_x, idx, None]
        
        # Add to final output
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


class SparseMoeBlock(nn.Module):
    """Sparse Mixture of Experts block.
    
    This module implements a sparse MoE layer where each token is routed
    to the top-k experts based on learned routing weights.
    
    Parameters
    ----------
    hidden_size : int
        Dimension of input/output hidden states.
    intermediate_size : int
        Dimension of the expert intermediate layer.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to select per token.
    norm_topk_prob : bool
        Whether to normalize the top-k probabilities.
    bias : bool
        Whether to use bias in linear layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        
        # Router gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Expert weights - stored as stacked tensors for efficiency
        # gate_proj and up_proj: [num_experts, intermediate_size, hidden_size]
        # down_proj: [num_experts, hidden_size, intermediate_size]
        self.gate_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        
        if bias:
            self.gate_proj_bias = nn.Parameter(torch.empty(num_experts, intermediate_size))
            self.up_proj_bias = nn.Parameter(torch.empty(num_experts, intermediate_size))
            self.down_proj_bias = nn.Parameter(torch.empty(num_experts, hidden_size))
        else:
            self.register_parameter('gate_proj_bias', None)
            self.register_parameter('up_proj_bias', None)
            self.register_parameter('down_proj_bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.gate_proj)
        nn.init.kaiming_uniform_(self.up_proj)
        nn.init.kaiming_uniform_(self.down_proj)
        if self.gate_proj_bias is not None:
            nn.init.zeros_(self.gate_proj_bias)
            nn.init.zeros_(self.up_proj_bias)
            nn.init.zeros_(self.down_proj_bias)
    
    def _expert_fn(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Compute single expert MLP output."""
        from . import silu_and_mul
        
        gate = F.linear(x, self.gate_proj[expert_idx], 
                       self.gate_proj_bias[expert_idx] if self.gate_proj_bias is not None else None)
        up = F.linear(x, self.up_proj[expert_idx],
                     self.up_proj_bias[expert_idx] if self.up_proj_bias is not None else None)
        gate_up = torch.cat([gate, up], dim=-1)
        hidden = silu_and_mul(gate_up)
        return F.linear(hidden, self.down_proj[expert_idx],
                       self.down_proj_bias[expert_idx] if self.down_proj_bias is not None else None)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape [batch_size, seq_len, hidden_size].
            
        Returns
        -------
        output : torch.Tensor
            Output tensor of shape [batch_size, seq_len, hidden_size].
        router_logits : torch.Tensor
            Router logits of shape [batch_size * seq_len, num_experts].
        """
        return sparse_moe_forward(
            hidden_states,
            self.gate.weight,
            self._expert_fn,
            self.num_experts,
            self.top_k,
            self.norm_topk_prob,
        )

