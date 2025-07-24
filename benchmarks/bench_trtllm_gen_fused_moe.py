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
"""
import json
import math
import os
import time

import pytest
import torch
from torch.nn import functional as F

from flashinfer import (
    RoutingMethodType,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.testing.utils import bench_kineto


class moe_args:

    def __init__(
            self,
            num_tokens,
            num_experts,
            hidden_size,
            intermediate_size,
            top_k,
            padding,
            hidden_states,
            hidden_states_scale,
            hidden_states_scale_global,
            expert_logits,
            gemm1_weights,
            gemm1_scales,
            gemm1_scales_global,
            gemm2_weights,
            gemm2_scales,
            gemm2_scales_global,
            permute_info,
            use_routing_scales_on_input,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.hidden_states_scale = hidden_states_scale
        self.hidden_states_scale_global = hidden_states_scale_global
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm1_scales = gemm1_scales
        self.gemm1_scales_global = gemm1_scales_global
        self.gemm2_weights = gemm2_weights
        self.gemm2_scales = gemm2_scales
        self.gemm2_scales_global = gemm2_scales_global
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input


class moe_args_dequant:

    def __init__(
            self,
            num_tokens,
            num_experts,
            hidden_size,
            intermediate_size,
            top_k,
            padding,
            hidden_states,
            expert_logits,
            gemm1_weights,
            gemm2_weights,
            permute_info,
            use_routing_scales_on_input,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm2_weights = gemm2_weights
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input


def routing_reference(expertLogits, topK, padding):
    originalDevice = expertLogits.device
    expertLogits = expertLogits.cpu()
    numTokens, numExperts = expertLogits.shape
    assert topK <= numExperts

    numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK, dtype=torch.int64)

    topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[expertIndex]
            numTokensPerExpert[expertIndex] += 1

    paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1, dtype=torch.int64)
    for ii in range(numExperts):

        def divUpMul(a, b):
            return (a + b - 1) // b * b

        paddedTokensPerExpertPrefixSum[ii + 1] = paddedTokensPerExpertPrefixSum[
                                                     ii
                                                 ] + divUpMul(numTokensPerExpert[ii], padding)
    permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

    expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK, dtype=torch.int64)
    permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
            permutedIdx = offsetForExpert + offsetWithinExpert

            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
            permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
            permutedIdxToTokenIdx[permutedIdx] = tokenIdx
    return {
        "paddedTokensPerExpertPrefixSum": paddedTokensPerExpertPrefixSum.to(
            originalDevice
        ),
        "permutedBufferSize": permutedBufferSize.item(),
        "expandedTokenIdxToPermutedIdx": expandedTokenIdxToPermutedIdx.to(
            originalDevice
        ),
        "permutedIdxToExpandedIdx": permutedIdxToExpandedIdx.to(originalDevice),
        "numTokensPerExpert": numTokensPerExpert.to(originalDevice),
        "expandedTokenIdxToExpert": expandedTokenIdxToExpert.to(originalDevice),
        "topKLogits": topKLogits.to(originalDevice),
        "permutedIdxToTokenIdx": permutedIdxToTokenIdx.to(originalDevice),
        "topKIndices": topKIndices.to(originalDevice),
    }


def noaux_tc_ref(logits, bias, n_group, topk_group, top_k, routed_scaling_factor):
    scores = F.sigmoid(logits)
    scores_with_bias = scores + bias
    if n_group > 1:
        scores_shape = list(scores_with_bias.shape)
        group_scores = torch.sum(
            torch.topk(
                scores_with_bias.view(
                    scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]
                ),
                k=2,
                dim=-1,
                largest=True,
                sorted=True,
            )[0],
            dim=-1,
        )
        _, group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, largest=True, sorted=True
        )
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group])
            .reshape(scores_shape)
        )
        scores_with_bias = scores_with_bias * score_mask

    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    new_mask = torch.zeros_like(scores)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = scores * new_mask
    score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
    scores = scores / score_sum * routed_scaling_factor
    return scores


# Tiered TopK routing used by DeepSeek
def routing_reference_no_aux(
        expert_logits,
        routing_bias,
        top_k,
        n_groups,
        top_k_groups,
        routed_scaling,
        padding,
        use_routing_scales_on_input=False,
):
    routing_logits = expert_logits.to(dtype=torch.float, device="cuda")
    if use_routing_scales_on_input:
        # if using routing scales on input, topK == 1 and the score is a plain sigmoid
        scores = F.sigmoid(routing_logits)
    else:
        scores = noaux_tc_ref(
            routing_logits, routing_bias, n_groups, top_k_groups, top_k, routed_scaling
        )
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


# TopK -> Softmax
def routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)
    topk_values = torch.nn.functional.softmax(topk_values.float(), dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


# Softmax->TopK -> Normalize
def routing_reference_renormalize_naive(expert_logits, top_k, num_experts, padding):
    norm_topk_prob = True
    scores = torch.nn.functional.softmax(expert_logits.float(), dim=-1)
    topk_values, topk_idx = torch.topk(scores, k=top_k, dim=-1)

    if norm_topk_prob:  # only diff with mixtral sparse moe block!
        topk_values /= topk_values.sum(dim=-1, keepdim=True)
    topk_values = topk_values.to(expert_logits.dtype)
    scores = scores.to(expert_logits.dtype)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def run_moe_dequant(args, quant_mode=["fp4"]):
    # Permute
    total_num_padded_tokens = args.permute_info["permutedBufferSize"]
    expanded_idx_to_permuted_idx = args.permute_info[
        "expandedTokenIdxToPermutedIdx"
    ].cpu()
    num_tokens_per_expert = args.permute_info["numTokensPerExpert"].cpu()
    permute_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        for j in range(args.top_k):
            permuted_idx = expanded_idx_to_permuted_idx[i * args.top_k + j]
            permute_output[permuted_idx] = args.hidden_states[i]
    # Gemm1
    gemm1_output = torch.full(
        (total_num_padded_tokens, 2 * args.intermediate_size),
        float("nan"),
        device="cuda",
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = permute_output[i : i + my_num_tokens]
        my_b = args.gemm1_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm1_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if args.use_routing_scales_on_input:
        assert args.top_k == 1
        # For each token and its top_k experts
        for token_idx in range(args.num_tokens):
            for k in range(args.top_k):
                # Get the permuted index for this token's k-th expert
                expanded_idx = token_idx * args.top_k + k
                permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
                expert_weight = args.permute_info["topKLogits"].to(torch.float)
                # Get the expert weight for this token and expert
                weight = expert_weight[token_idx, k]
                # Scale the corresponding row in gemm1_output
                gemm1_output[permuted_idx] *= weight

    # Activation
    activation_output = torch.full(
        (total_num_padded_tokens, args.intermediate_size), float("nan"), device="cuda"
    ).to(torch.float)

    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = gemm1_output[i : i + my_num_tokens]
        my_x1 = my_a[:, : args.intermediate_size]
        my_x2 = my_a[:, args.intermediate_size :]
        activation_output[i : i + my_num_tokens] = F.silu(my_x2) * my_x1
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if quant_mode == "fp4":
        activation_output, c_global_sf = quant_dequant_fp4(
            activation_output.to(torch.bfloat16), False, True
        )
        activation_output = activation_output.to(torch.float)
        args.c_global_sf = c_global_sf

    # Gemm2
    gemm2_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = activation_output[i : i + my_num_tokens]
        my_b = args.gemm2_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm2_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding
    # Finalize
    expert_weight = args.permute_info["topKLogits"].to(torch.float)
    finalize_output = torch.full(
        (args.num_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        acc = torch.zeros(args.hidden_size, dtype=torch.float, device="cuda")
        for top_k_idx in range(args.top_k):
            expanded_idx = i * args.top_k + top_k_idx
            permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
            original_vector = gemm2_output[permuted_idx]
            weight = (
                expert_weight[i, top_k_idx]
                if not args.use_routing_scales_on_input
                else 1.0
            )
            acc += original_vector * weight
        finalize_output[i] = acc
    return finalize_output


def e2m1_and_ufp8_scale_to_float_tensor_v2(
        e2m1_tensor: torch.Tensor,
        ufp8_scale_tensor: torch.Tensor,
        global_scale_tensor: torch.Tensor,
        sf_vec_size,
        ufp8_type: int = 1,
        is_sf_swizzled_layout: bool = True,
):
    float_tensor = e2m1_and_ufp8sf_scale_to_float(
        e2m1_tensor.cpu(),
        ufp8_scale_tensor.cpu().reshape(-1),
        global_scale_tensor.cpu(),
        sf_vec_size,
        ufp8_type,
        is_sf_swizzled_layout,
    )
    return float_tensor


def e2m1_and_ufp8_scale_batches(
        mat_fp4: torch.Tensor,
        scale_tensor: torch.Tensor,
        global_scale_tensor: torch.Tensor,
        sf_vec_size: int,
        ufp8_type: int = 1,
):
    num_batches = mat_fp4.size(0)

    scale_tensor = scale_tensor.view(num_batches, -1)

    tensors = [
        e2m1_and_ufp8_scale_to_float_tensor_v2(
            mat_fp4[b, :, :], scale_tensor[b, :], global_scale_tensor[b], sf_vec_size
        )
        for b in range(num_batches)
    ]

    result = torch.stack(tensors)

    return result


def run_moe_reference_fp4(args):
    sf_vec_size = 16

    hidden_states_dequant = e2m1_and_ufp8_scale_to_float_tensor_v2(
        args.hidden_states,
        args.hidden_states_scale,
        1 / args.hidden_states_scale_global,
        sf_vec_size,
        ).cuda()

    gemm1_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm1_weights, args.gemm1_scales, 1 / args.gemm1_scales_global, sf_vec_size
    ).cuda()

    gemm2_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm2_weights, args.gemm2_scales, 1 / args.gemm2_scales_global, sf_vec_size
    ).cuda()

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
    )

    return run_moe_dequant(args_dequant, "fp4"), args_dequant


def quant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_with_global_sf(
        a, a_global_sf, use_ue8m0=False, is_sf_swizzled_layout=True
):
    """
    Quantize FP4 with pre-calculated global scale factor.
    Used specifically for hidden states in CUDA graph capture to avoid runtime computation.
    """
    sf_vec_size = 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches(a, num_experts, use_ue8m0=False, is_sf_swizzled_layout=True):
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        a_fp4, a_sf, a_global_sf = quant_fp4(a[i], use_ue8m0, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def quant_dequant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(
        a_fp4.cpu(), a_sf.cpu(), 1 / a_global_sf, sf_vec_size
    )

    return a_pt.cuda(), a_global_sf


def check_accuracy(a, b, atol, rtol, percent):
    if torch.any(torch.isnan(a)):
        raise Exception("NaN in a")
    if torch.any(torch.isnan(b)):
        raise Exception("NaN in b")
    if torch.any(torch.isinf(a)):
        raise Exception("Inf in a")
    if torch.any(torch.isinf(b)):
        raise Exception("Inf in b")
    assert a.shape == b.shape
    left = torch.abs(a - b)
    right = atol + rtol * torch.abs(b)
    count = torch.sum(left > right)
    mismatch_percent = count / a.numel()
    if mismatch_percent > 1 - percent:
        raise Exception(
            "Mismatch percentage is %f for rtol %f" % (mismatch_percent, rtol)
        )


def create_expert_logits(num_token, num_experts, k):
    """
    Create deterministic expert logits for testing where specific experts
    are guaranteed to be selected for each token.

    Args:
        num_token: Number of tokens
        num_experts: Number of experts
        k: Top-k value (number of experts to select per token)

    Returns:
        logits: Expert logits tensor [num_token, num_experts] (CUDA bfloat16)
        index: Expected top-k indices [num_token, k] (CUDA)
        large_random: The large random values used [num_token, k] (CUDA)
    """
    # 1. Create logits tensor
    logits = torch.zeros(num_token, num_experts)

    # 2. Set index sequence
    final_size = num_token * k
    repeat_count = math.ceil(final_size / num_experts)
    indices = torch.arange(num_experts, dtype=torch.int32)
    indices = indices.repeat(repeat_count)
    indices = indices[:final_size]
    index = indices.view(num_token, k).contiguous()

    # 3. Generate large random numbers
    large_random = torch.randint(5, 11, (num_token, k), dtype=torch.float32)

    # 4. Put the random number to the place we want
    for token_id in range(num_token):
        for j in range(k):
            expert_idx = index[token_id, j]
            logits[token_id, expert_idx] = large_random[token_id, j]

    # 5. Set smaller random numbers in other places
    mask = logits == 0
    logits[mask] = torch.rand(mask.sum())

    logits = torch.nn.functional.softmax(logits, dim=-1)

    # Convert to CUDA tensors with appropriate dtypes
    logits = logits.to(device="cuda", dtype=torch.bfloat16)
    index = index.to(device="cuda")
    large_random = large_random.to(device="cuda")

    return logits, index, large_random


def compute_moe_reference_with_routing(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        expert_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        top_k,
        padding,
        n_groups,
        top_k_groups,
        routed_scaling,
        routing_method_type,
):
    """
    Compute the reference MoE output using dequantized operations with full routing support.

    Returns:
        output_dequant_reference: Reference output tensor
        args_dequant: Dequantized arguments for debugging
    """
    use_ue8m0 = False

    # Quantize hidden states
    (
        hidden_states_fp4_bytes,
        hidden_states_scale_fp4_bytes,
        hidden_states_scale_global,
    ) = quant_fp4(hidden_states, use_ue8m0, True)

    # Quantize the weights for FC1
    gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = (
        quant_fp4_batches(gemm1_weights, num_experts, use_ue8m0, True)
    )

    # Quantize the weights for FC2
    gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = (
        quant_fp4_batches(gemm2_weights, num_experts, use_ue8m0, True)
    )

    # Generate routing info based on method
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            top_k,
            n_groups,
            top_k_groups,
            routed_scaling,
            padding,
        )
    elif routing_method_type == RoutingMethodType.Renormalize:
        permute_info, scores = routing_reference_renormalize(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, scores = routing_reference_renormalize_naive(
            expert_logits, top_k, num_experts, padding
        )
    else:
        raise NotImplementedError(
            f"Routing method {routing_method_type} not implemented"
        )

    # Create arguments for reference computation
    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states_fp4_bytes,
        hidden_states_scale_fp4_bytes,
        hidden_states_scale_global,
        scores,
        gemm1_weights_fp4_bytes,
        gemm1_scales_fp4_bytes,
        gemm1_scales_global,
        gemm2_weights_fp4_bytes,
        gemm2_scales_fp4_bytes,
        gemm2_scales_global,
        permute_info,
        False,
    )

    # Run the reference implementation
    output_dequant_reference, args_dequant = run_moe_reference_fp4(args)

    return output_dequant_reference, args_dequant, args


def compute_moe_actual_with_routing(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        expert_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        top_k,
        padding,
        n_groups,
        top_k_groups,
        routed_scaling,
        routing_method_type,
        tile_tokens_dim,
        args_dequant,
        args,
):
    """
    Compute the actual MoE output using the optimized kernel with full routing support.

    Returns:
        output_dequant_actual: Actual output tensor from the kernel
    """

    def prepare_static_weights():
        """
        Handle all static weight-related preprocessing.
        This should be done once at model load time in production.

        Returns:
            Dict containing all preprocessed weight tensors and scale factors
        """
        use_ue8m0 = False
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        # Quantize weights with linear layout for kernels
        _, gemm1_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm1_weights, num_experts, use_ue8m0, False
        )
        _, gemm2_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm2_weights, num_experts, use_ue8m0, False
        )

        # Convert quantized weights to proper formats
        gemm1_weights_fp4 = args.gemm1_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2
        )  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 16
        )  # fp8 scaling factors

        gemm2_weights_fp4 = args.gemm2_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )  # packed fp4
        gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, hidden_size, intermediate_size // 16
        )  # fp8 scaling factors

        # Reorder rows of W1 and scales for fused gated activation
        gemm1_weights_fp4_interleaved = []
        gemm1_scales_fp4_interleaved = []
        for i in range(num_experts):
            gemm1_weights_fp4_interleaved.append(
                reorder_rows_for_gated_act_gemm(gemm1_weights_fp4[i].clone())
            )
            gemm1_scales_fp4_interleaved.append(
                reorder_rows_for_gated_act_gemm(gemm1_scales_linear_fp4[i].clone())
            )

        # Stack weights and scales for all experts
        gemm1_weights_fp4_interleaved = torch.stack(
            gemm1_weights_fp4_interleaved
        ).reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
        gemm1_scales_fp4_interleaved = torch.stack(
            gemm1_scales_fp4_interleaved
        ).reshape(num_experts, 2 * intermediate_size, hidden_size // 16)

        # Shuffle weights and scaling factors for transposed mma output
        gemm1_weights_fp4_shuffled = []
        gemm1_scales_fp4_shuffled = []
        gemm2_weights_fp4_shuffled = []
        gemm2_scales_fp4_shuffled = []
        for i in range(num_experts):
            gemm1_weights_fp4_shuffled.append(
                shuffle_matrix_a(
                    gemm1_weights_fp4_interleaved[i].view(torch.uint8), epilogue_tile_m
                )
            )
            gemm1_scales_fp4_shuffled.append(
                shuffle_matrix_sf_a(
                    gemm1_scales_fp4_interleaved[i].view(torch.uint8), epilogue_tile_m
                )
            )

            gemm2_weights_fp4_shuffled.append(
                shuffle_matrix_a(
                    gemm2_weights_fp4[i].view(torch.uint8), epilogue_tile_m
                )
            )
            gemm2_scales_fp4_shuffled.append(
                shuffle_matrix_sf_a(
                    gemm2_scales_linear_fp4[i].view(torch.uint8), epilogue_tile_m
                )
            )

        # Stack weights for all experts
        gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
        gemm1_scales_fp4_shuffled = (
            torch.stack(gemm1_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // 16)
        )

        # Calculate scaling factors that depend on weights
        scale_c_fc1 = (
                args_dequant.c_global_sf
                * (1.0 / args.gemm1_scales_global)
                * (1.0 / args.hidden_states_scale_global)
        )
        scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
                1.0 / args.hidden_states_scale_global
        )
        scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (
                1.0 / args.gemm2_scales_global
        )

        return {
            "gemm1_weights_fp4_shuffled": gemm1_weights_fp4_shuffled,
            "gemm1_scales_fp4_shuffled": gemm1_scales_fp4_shuffled,
            "gemm2_weights_fp4_shuffled": gemm2_weights_fp4_shuffled,
            "gemm2_scales_fp4_shuffled": gemm2_scales_fp4_shuffled,
            "scale_c_fc1": scale_c_fc1,
            "scale_gate_fc1": scale_gate_fc1,
            "scale_c_fc2": scale_c_fc2,
        }

    # Process static weights (would be cached in production)
    static_data = prepare_static_weights()

    # Calculate global scale factor for hidden states offline (precalculated parameter)
    hidden_states_global_sf = (448 * 6) / hidden_states.float().abs().nan_to_num().max()

    use_ue8m0 = False
    do_finalize = True

    # Quantize hidden states with precalculated global scale
    hidden_states_fp4_bytes, hidden_states_scale_linear_fp4_bytes, _ = (
        quant_fp4_with_global_sf(
            hidden_states, hidden_states_global_sf, use_ue8m0, False
        )
    )
    hidden_states_fp4 = hidden_states_fp4_bytes.reshape(num_tokens, hidden_size // 2)
    hidden_states_scale_linear_fp4 = hidden_states_scale_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(-1)

    if 1:
        trace_dir = os.environ.get("BENCH_KINETO_TRACE_DIR")
        [time_gemm1, time_gemm2] = bench_kineto(
            lambda: trtllm_fp4_block_scale_moe(
                expert_logits,
                routing_bias,
                hidden_states_fp4,
                hidden_states_scale_linear_fp4,
                static_data["gemm1_weights_fp4_shuffled"],
                static_data["gemm1_scales_fp4_shuffled"],
                static_data["gemm2_weights_fp4_shuffled"],
                static_data["gemm2_scales_fp4_shuffled"],
                static_data["scale_c_fc1"],
                static_data["scale_gate_fc1"],
                static_data["scale_c_fc2"],
                num_experts,
                top_k,
                n_groups,
                top_k_groups,
                intermediate_size,
                0,
                num_experts,
                routed_scaling,
                tile_tokens_dim,
                routing_method_type,
                do_finalize=True,
            ),
            kernel_names="TODO_what_name",
            num_kernels_per_period=2,
            trace_path=f"{trace_dir}/{time.time()}.trace.json.gz" if trace_dir else None,
        )

        # NOTE MODIFIED
        print(f"MAIN_OUTPUT=" + json.dumps(dict(
            batch_size=batch_size,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            time_gemm1_us=time_gemm1 * 1e6,
            time_gemm2_us=time_gemm2 * 1e6,
        )))

    output = trtllm_fp4_block_scale_moe(
        expert_logits,
        routing_bias,
        hidden_states_fp4,
        hidden_states_scale_linear_fp4,
        static_data["gemm1_weights_fp4_shuffled"],
        static_data["gemm1_scales_fp4_shuffled"],
        static_data["gemm2_weights_fp4_shuffled"],
        static_data["gemm2_scales_fp4_shuffled"],
        static_data["scale_c_fc1"],
        static_data["scale_gate_fc1"],
        static_data["scale_c_fc2"],
        num_experts,
        top_k,
        n_groups,
        top_k_groups,
        intermediate_size,
        0,
        num_experts,
        routed_scaling,
        tile_tokens_dim,
        routing_method_type,
        do_finalize=True,
    )

    output_dequant_actual = output[0].to(torch.float)

    return output_dequant_actual


def compare_moe_outputs(
        output_dequant_reference,
        output_dequant_actual,
        seed,
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        routing_method_type,
):
    """
    Compare reference and actual MoE outputs and perform accuracy analysis.

    Raises:
        Exception: If accuracy test fails
    """
    # Use check_accuracy to validate - it will raise exception if test fails
    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=0.1,
        rtol=0.85,
        percent=0.925,
    )


def test_moe_nvfp4(
        num_tokens,
        hidden_size,
        intermediate_size,
        routing_info,
):
    seed = 42
    torch.random.manual_seed(seed)

    # Extract routing configuration
    top_k = routing_info["top_k"]
    padding = routing_info["padding"]
    n_groups = routing_info["n_groups"]
    top_k_groups = routing_info["top_k_groups"]
    routed_scaling = routing_info["routed_scaling"]
    num_experts = routing_info["num_experts"]
    routing_method_type = routing_info["routing_method_type"]
    tile_tokens_dim = 8

    # Validation checks
    assert top_k <= num_experts
    assert top_k <= 8
    if (top_k_groups is not None) and (n_groups is not None):
        assert top_k_groups <= 4
        assert num_experts > n_groups
        assert num_experts % n_groups == 0
        assert num_experts % 4 == 0
        assert top_k < (top_k_groups * num_experts / n_groups)

    # Create expert logits based on routing method
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.float
        )
    elif (
            routing_method_type == RoutingMethodType.RenormalizeNaive
            or routing_method_type == RoutingMethodType.Renormalize
    ):
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.bfloat16
        )

    # Handle routing bias
    if routing_info["has_routing_bias"]:
        routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)
    else:
        routing_bias = None

    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Compute reference output with updated routing method handling
    output_dequant_reference, args_dequant, args = compute_moe_reference_with_routing(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        expert_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        top_k,
        padding,
        n_groups,
        top_k_groups,
        routed_scaling,
        routing_method_type,
    )

    # Compute actual output using optimized kernel
    output_dequant_actual = compute_moe_actual_with_routing(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        expert_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        top_k,
        padding,
        n_groups,
        top_k_groups,
        routed_scaling,
        routing_method_type,
        tile_tokens_dim,
        args_dequant,
        args,
    )

    # Compare outputs - will raise exception if test fails
    compare_moe_outputs(
        output_dequant_reference,
        output_dequant_actual,
        seed,
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        routing_method_type,
    )

# ---------------------------------------------------------------------------

BATCH_SIZES = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
]

test_configs = [
    # NOTE MODIFIED ADD
    *[
        {
            "hidden_size": 7168,
            "intermediate_size": 2048,
            # RoutingDSv3
            "routing_info": {
                # TODO correct?
                "num_experts": num_experts,
                "top_k": 8,
                "padding": 8,
                "n_groups": 8,
                "top_k_groups": 4,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
        }
        for num_experts in [
            288 // 1,
            288 // 2,
            288 // 4,
            288 // 8,
            288 // 16,
            288 // 32,
            288 // 48,
            288 // 72,
        ]
    ],
]

if __name__ == '__main__':
    for config in test_configs:
        for batch_size in BATCH_SIZES:
            test_moe_nvfp4(
                num_tokens=batch_size,
                hidden_size=config["hidden_size"],
                intermediate_size=config["intermediate_size"],
                routing_info=config["routing_info"],
            )
