import os
import sys
from enum import IntEnum

import pytest
import torch
from torch.nn import functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import flashinfer.fused_moe as fused_moe

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn

workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.int8).cuda()


def is_16byte_aligned(tensor):
    return (tensor.data_ptr() % 16) == 0


# The type of method in top-K routing, for use in torch custom op
# Please keep this in sync with the counterpart defined in cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h
class RoutingMethodType(IntEnum):
    # Default: Softmax -> TopK
    Default = (0,)
    # Renormalize: TopK -> Softmax
    Renormalize = (1,)
    # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups
    DeepSeekV3 = (2,)
    # Llama4: Top1 -> Sigmoid
    Llama4 = (3,)
    # Qwen3: Softmax -> TopK -> Renormalize
    RenormalizeNaive = (4,)
    # Unspecified
    Unspecified = 5.0


def get_reorder_rows_for_gated_act_gemm_row_indices(x) -> torch.Tensor:
    """
    Reorders rows in the gemm/MOE_gemm weight matrix for min-latency
    [r0, r1, r2, r3, ..., rN/2, r(N/2+1), .. r(N-1)]
    to
    [r0, rN/2, r1, rN/2+1, ..., r(N/2-1), r(N-1)]
    """
    assert x.dim() == 2, f"x should be a 2D tensor, not {x.dim()}"
    M, K = x.shape
    assert M % 2 == 0, f"x.shape[0] must be even, not {M}"

    row_indices = torch.arange(M, dtype=torch.long)

    # We split into top half and bottom half, but if M is odd,
    # the bottom half is one row larger.
    top = row_indices[: (M + 1) // 2]  # round up
    bot = row_indices[(M + 1) // 2 :]  # remainder

    # Create the output
    permuted_row_indices = torch.empty_like(row_indices)

    # We'll place rows of `top` and `bot` in alternation
    permuted_row_indices[0::2] = top
    permuted_row_indices[1::2] = bot

    return permuted_row_indices


def reorder_rows_for_gated_act_gemm(x):
    """
    PyTorch implementation of trt-llm gen `reorderRowsForGatedActGemm`
    """
    row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(x)

    permute = lambda x: x[row_indices]

    return permute(x)


# yapf: disable
srcToDstBlk16RowMap = [
    0,  8,
    1,  9,
    2, 10,
    3, 11,
    4, 12,
    5, 13,
    6, 14,
    7, 15
]

srcToDstBlk32RowMap = [
    0,  8, 16, 24,
    1,  9, 17, 25,
    2, 10, 18, 26,
    3, 11, 19, 27,
    4, 12, 20, 28,
    5, 13, 21, 29,
    6, 14, 22, 30,
    7, 15, 23, 31
]
# yapf: enable


def get_shuffle_block_size(epilogue_tile_m: int) -> int:
    shuffle_block_size = 16
    if epilogue_tile_m % 128 == 0:
        shuffle_block_size = 32
    return shuffle_block_size


def get_shuffle_matrix_a_row_indices(
    input_tensor: torch.Tensor, epilogue_tile_m: int
) -> torch.Tensor:
    """
    Higher-level PyTorch approach to reorder the rows in blocks of size 16 or 32.
    - We do NOT try to handle custom e2m1 memory usage (i.e. no 'K/2' bytes).
    - Instead, we purely reorder rows in a standard PyTorch shape [M, K].
    """
    assert (
        input_tensor.dim() == 2
    ), f"input_tensor should be a 2D tensor, not {input_tensor.dim()}"

    # M, K from the input
    M, K = input_tensor.shape

    # Choose block size 16 or 32
    shuffle_block_size = get_shuffle_block_size(epilogue_tile_m)
    row_map = srcToDstBlk16RowMap if shuffle_block_size == 16 else srcToDstBlk32RowMap

    assert (
        M % shuffle_block_size == 0
    ), f"input_tensor.shape[0] must be multiples of {shuffle_block_size}"

    # row_indices[new_row] = old_row
    # so row_indices is an array of size M telling us from which old_row
    # the new_row should be taken.
    row_indices = torch.empty(M, dtype=torch.long)

    for old_row in range(M):
        block_idx = old_row // shuffle_block_size
        row_in_block = old_row % shuffle_block_size
        mapped_row_in_block = row_map[row_in_block]

        new_row = block_idx * shuffle_block_size + mapped_row_in_block

        row_indices[new_row] = old_row

    return row_indices


def shuffle_matrix_a(input_tensor: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    """
    PyTorch equivalent of trtllm-gen `shuffleMatrixA`
    """
    row_indices = get_shuffle_matrix_a_row_indices(input_tensor, epilogue_tile_m)
    return torch.index_select(input_tensor, 0, row_indices.to(input_tensor.device))


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


def dequant_reference_dsfp8(input, scale, transpose_scale, block_m, block_n):
    input = input.to(torch.float)
    scale = scale.to(torch.float)
    if transpose_scale:
        scale = scale.t()
    output = torch.zeros_like(input)
    m, n = input.shape
    m_tile = 128 if block_m else 1
    n_tile = 128 if block_n else 1
    assert m % m_tile == 0
    assert n % n_tile == 0
    assert scale.shape == (m // m_tile, n // n_tile)
    if m_tile == 1:
        for j in range(0, n, n_tile):
            output[:, j : j + n_tile] = (
                input[:, j : j + n_tile] * scale[:, j // n_tile][:, None]
            )
    elif n_tile == 1:
        for i in range(0, m, m_tile):
            output[i : i + m_tile] = input[i : i + m_tile] * scale[i // m_tile]
    else:
        for i in range(0, m, m_tile):
            for j in range(0, n, n_tile):
                output[i : i + m_tile, j : j + n_tile] = (
                    input[i : i + m_tile, j : j + n_tile]
                    * scale[i // m_tile, j // n_tile]
                )
    return output


def run_moe_dequant(args, quant_mode=["dsFp8", "perTensorFp8"]):
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

    if quant_mode == "perTensorFp8":
        activation_output, c_global_sf = quant_dequant_per_tensor_fp8(
            activation_output.to(torch.bfloat16)
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


def run_moe_reference_dsfp8(args):
    hidden_states_dequant = dequant_reference_dsfp8(
        args.hidden_states, args.hidden_states_scale, True, False, True
    )

    gemm1_weights_dequant = {}
    for i in range(args.num_experts):
        gemm1_weights_dequant[i] = dequant_reference_dsfp8(
            args.gemm1_weights[i], args.gemm1_scales[i], False, True, True
        )

    gemm2_weights_dequant = {}
    for i in range(args.num_experts):
        gemm2_weights_dequant[i] = dequant_reference_dsfp8(
            args.gemm2_weights[i], args.gemm2_scales[i], False, True, True
        )

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

    return run_moe_dequant(args_dequant, "dsFp8"), args_dequant


def run_moe_reference_per_tensor_scale_fp8(args):

    hidden_states_dequant = (
        args.hidden_states.to(torch.float) / args.hidden_states_scale_global
    )

    gemm1_weights_dequant = {}
    for i in range(args.num_experts):
        gemm1_weights_dequant[i] = (
            args.gemm1_weights[i].to(torch.float) / args.gemm1_scales_global[i]
        )

    gemm2_weights_dequant = {}
    for i in range(args.num_experts):
        gemm2_weights_dequant[i] = (
            args.gemm2_weights[i].to(torch.float) / args.gemm2_scales_global[i]
        )

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

    return run_moe_dequant(args_dequant, "perTensorFp8"), args_dequant


def quant_fp8_per_tensor(a):
    a_global_sf = 448 / a.float().abs().nan_to_num().max()

    a_fp8 = (a * a_global_sf).to(torch.float8_e4m3fn)

    return a_fp8, a_global_sf


def quant_fp8_per_tensor_batches(a):
    num_batches = a.size(0)
    a_quant = []
    a_scales = []

    for i in range(num_batches):
        a_fp8, a_global_sf = quant_fp8_per_tensor(a[i])
        a_quant.append(a_fp8)
        a_scales.append(a_global_sf)

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


def quant_dequant_per_tensor_fp8(a):
    a_global_sf = 448 / a.float().abs().nan_to_num().max()
    a_fp8 = (a * a_global_sf).to(torch.float8_e4m3fn)
    a_pt = a_fp8.to(torch.float) / a_global_sf

    return a_pt.cuda(), a_global_sf


# @pytest.mark.skipif(
#     getSMVersion() != 100,
#     reason="The kernel only supports Blackwell. Current SM is %d." %
#     getSMVersion(),
# )
@pytest.mark.parametrize("num_tokens", [16, 64, 1024, 4096])
@pytest.mark.parametrize(
    "expert_info", [(32, 8, 4, 8), (32, 1, 1, 5), (72, 1, 1, 6), (256, 8, 4, 8)]
)
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
# @pytest.mark.parametrize("use_autotune", [True, False],
#                          ids=["autotune", "no_autotune"])
def test_moe_fp8(num_tokens, expert_info, hidden_size, intermediate_size):

    torch.random.manual_seed(0)

    #
    # Data Generation
    #
    num_experts, n_groups, top_k_groups, top_k = expert_info
    padding = 8
    routed_scaling = 2.5
    routing_method_type = RoutingMethodType.DeepSeekV3
    tile_tokens_dim = 8 if num_tokens < 1024 else 32

    assert top_k <= num_experts
    assert top_k <= 8
    assert top_k_groups <= 4
    assert num_experts > n_groups
    assert num_experts % n_groups == 0
    assert num_experts % 4 == 0
    assert top_k < (top_k_groups * num_experts / n_groups)

    expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
        torch.float
    )
    routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)

    hidden_states = torch.randn((num_tokens, hidden_size), device="cuda").to(
        torch.float8_e4m3fn
    )
    hidden_states_scale = 2 * torch.rand(
        (hidden_size // 128, num_tokens), device="cuda"
    ).to(torch.float)

    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size), device="cuda"
    ).to(torch.float8_e4m3fn)
    gemm1_scales = 2 * torch.rand(
        (num_experts, 2 * intermediate_size // 128, hidden_size // 128), device="cuda"
    ).to(torch.float)
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size), device="cuda"
    ).to(torch.float8_e4m3fn)
    gemm2_scales = 2 * torch.rand(
        (num_experts, hidden_size // 128, intermediate_size // 128), device="cuda"
    ).to(torch.float)

    permute_info, scores = routing_reference_no_aux(
        expert_logits,
        routing_bias,
        top_k,
        n_groups,
        top_k_groups,
        routed_scaling,
        padding,
    )

    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        hidden_states_scale,
        None,
        scores,
        gemm1_weights,
        gemm1_scales,
        None,
        gemm2_weights,
        gemm2_scales,
        None,
        permute_info,
        False,
    )

    output = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    # with autotune(use_autotune):
    fused_moe.trtllm_fp8_block_scale_moe(
        expert_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_scales,
        gemm2_weights,
        gemm2_scales,
        output,
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
    )

    output_dequant_actual = output.to(torch.float)
    #
    # Run the reference implementations
    #
    output_dequant_reference, _ = run_moe_reference_dsfp8(args)

    #
    # Check the results
    #
    def check_accuracy(a, b, atol, rtol, percent):
        if torch.any(torch.isnan(a)):
            raise Exception("NaN in a")
        if torch.any(torch.isnan(b)):
            raise Exception("NaN in b")
        assert a.shape == b.shape
        left = torch.abs(a - b)
        right = atol + rtol * torch.abs(b)
        count = torch.sum(left > right)
        mismatch_percent = count / a.numel()
        if mismatch_percent > 1 - percent:
            raise Exception(
                "Mismatch percentage is %f for rtol %f" % (mismatch_percent, rtol)
            )

    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=0.1,
        rtol=0.85,
        percent=0.925,
    )


# @pytest.mark.skipif(
#     getSMVersion() != 100,
#     reason="The kernel only supports Blackwell. Current SM is %d." %
#     getSMVersion(),
# )
@pytest.mark.parametrize("num_tokens", [1, 2, 16, 64, 1024, 4096])
@pytest.mark.parametrize("expert_info", [(128, 0, 0, 1, True)])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("intermediate_size", [2048])
def test_moe_fp8_per_tensor_scale(
    num_tokens, expert_info, hidden_size, intermediate_size
):
    torch.random.manual_seed(0)

    #
    # Data Generation
    #
    num_experts, n_groups, top_k_groups, top_k, use_routing_scales_on_input = (
        expert_info
    )
    # FIXME: set to TileN size
    padding = 8
    routed_scaling = 2.5
    routing_method_type = RoutingMethodType.Llama4
    tile_tokens_dim = 8

    assert top_k <= num_experts
    assert top_k <= 8
    assert top_k_groups <= 4
    assert num_experts > n_groups
    assert n_groups == 0 or num_experts % n_groups == 0
    assert num_experts % 4 == 0
    assert n_groups == 0 or top_k < (top_k_groups * num_experts / n_groups)

    expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
        torch.bfloat16
    )
    routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)

    hidden_states = torch.randn((num_tokens, hidden_size), device="cuda").to(
        torch.bfloat16
    )

    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size), device="cuda"
    ).to(torch.bfloat16)
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size), device="cuda"
    ).to(torch.bfloat16)

    hidden_states_quant, hidden_states_global_scale = quant_fp8_per_tensor(
        hidden_states
    )
    gemm1_weights_quant, gemm1_global_scales = quant_fp8_per_tensor_batches(
        gemm1_weights
    )
    gemm2_weights_quant, gemm2_global_scales = quant_fp8_per_tensor_batches(
        gemm2_weights
    )

    permute_info, scores = routing_reference_no_aux(
        expert_logits,
        routing_bias,
        top_k,
        n_groups,
        top_k_groups,
        routed_scaling,
        padding,
        use_routing_scales_on_input,
    )

    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states_quant,
        None,
        hidden_states_global_scale,
        scores,
        gemm1_weights_quant,
        None,
        gemm1_global_scales,
        gemm2_weights_quant,
        None,
        gemm2_global_scales,
        permute_info,
        use_routing_scales_on_input,
    )
    #
    # Run the reference implementations
    #
    # It is important to run the reference implementation before the TRT-LLM kernel
    # because the MoE shuffles the weights in-place.
    output_dequant_reference, args_dequant = run_moe_reference_per_tensor_scale_fp8(
        args
    )

    # FIXME: this depends on the kernel internals
    epilogue_tile_m = 128

    # Reorder rows of W1 for fused gated activation
    gemm1_weights_fp8_interleaved = []
    for i in range(num_experts):
        gemm1_weights_fp8_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights_quant[i].clone())
        )

    # Stack weights and scales for all experts
    gemm1_weights_fp8_interleaved = torch.stack(gemm1_weights_fp8_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp8_interleaved[i].view(torch.uint8), epilogue_tile_m
            )
        )

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights_quant[i].view(torch.uint8), epilogue_tile_m)
        )

    # Stack weights for all experts
    gemm1_weights_fp8_shuffled = torch.stack(gemm1_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )
    gemm2_weights_fp8_shuffled = torch.stack(gemm2_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )

    # c_global_sf: fc2_input_scale
    scale_c_fc1 = (
        args_dequant.c_global_sf
        * (1.0 / args.gemm1_scales_global)
        * (1.0 / args.hidden_states_scale_global)
    )

    # self.fc31_alpha
    scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
        1.0 / args.hidden_states_scale_global
    )

    # self.fc2_alpha
    scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (1.0 / args.gemm2_scales_global)

    output = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    fused_moe.trtllm_fp8_per_tensor_scale_moe(
        routing_logits=expert_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states_quant,
        gemm1_weights=gemm1_weights_fp8_shuffled,
        output1_scales_scalar=scale_c_fc1,
        output1_scales_gate_scalar=scale_gate_fc1,
        gemm2_weights=gemm2_weights_fp8_shuffled,
        output2_scales_scalar=scale_c_fc2,
        output=output,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_groups,
        topk_group=top_k_groups,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=routed_scaling,
        use_routing_scales_on_input=use_routing_scales_on_input,
        tile_tokens_dim=tile_tokens_dim,
        routing_method_type=routing_method_type,
        store_workspace_info=False,  # Enable workspace info storage for debugging
    )

    output_dequant_actual = output.to(torch.float)

    def check_accuracy(a, b, atol, rtol, percent):
        if torch.any(torch.isnan(a)):
            raise Exception("NaN in a")
        if torch.any(torch.isnan(b)):
            raise Exception("NaN in b")
        assert a.shape == b.shape
        left = torch.abs(a - b)
        right = atol + rtol * torch.abs(b)
        count = torch.sum(left > right)
        mismatch_percent = count / a.numel()
        if mismatch_percent > 1 - percent:
            raise Exception(
                "Mismatch percentage is %f for rtol %f" % (mismatch_percent, rtol)
            )

    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=0.1,
        rtol=0.85,
        percent=0.925,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
