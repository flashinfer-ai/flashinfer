# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""The persistent DeepSeek FP8 activation kernel of the trtllm-gen fused MoE.

``activationDeepSeekKernelPersistent128`` replaces the per-pair ``activationDeepSeekKernel`` at
every token count when the runner provides the permuted row -> token map. For finite inputs it
writes the same bytes as the old kernel, so the output tests cannot tell the two apart; this test
reads the launched kernel names from the profiler instead, and checks the one property the output
tests do not cover: the kernel reads routing outputs before its PDL wait, so a run with PDL on
must produce exactly the bytes of a run with PDL off, with and without the per-expert SwiGLU
controls (which the kernel looks up once per piece).
"""

import pytest
import torch

from flashinfer.utils import get_compute_capability

from tests.moe.trtllm_gen_fused_moe_utils import (
    ActivationType,
    FP8BlockScaleMoe,
    Fp8QuantizationType,
    QuantMode,
    RoutingMethodType,
    WeightLayout,
    check_accuracy,
    moe_args,
    routing_reference_renormalize,
    run_moe_reference_dsfp8,
    trtllm_fp8_block_scale_moe,
)
from tests.cake.test_helpers.cake_profiler import cuda_kernel_names

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1024  # 8 scale blocks of 128 per activation row
NUM_EXPERTS = 32
TOP_K = 8

PERSISTENT_KERNEL = "activationDeepSeekKernelPersistent128"
OLD_KERNEL = "activationDeepSeekKernel"


def _launched(names, kernels):
    """Return the kernels of ``kernels`` found among the profiler names (a template name is ``<kernel><...>``)."""
    found = []
    for name in names:
        for kernel in kernels:
            if kernel + "<" in name or kernel + "(" in name:
                found.append(kernel)
                break
    return found


def _skip_unless_sm100_class():
    """Skip unless a CUDA device of the SM100 class is present (where the trtllm-gen fused MoE tests run)."""
    if not torch.cuda.is_available():
        pytest.skip("the trtllm-gen fused MoE needs a CUDA GPU")
    major, _ = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("the trtllm-gen fused MoE runs on SM100-class GPUs only")


def _make_inputs(num_tokens, local_num_experts, local_expert_offset, remote_only=False):
    """Quantized inputs and weights of one DeepSeek FP8 block-scale MoE call (deterministic in ``num_tokens``).

    By default every token gets a large logit for the first local expert, so each token has at
    least one local (token, expert) pair and the activation output is never all zeros. With
    ``remote_only`` every token's top-k experts are all on other ranks instead, so this rank has
    no pair at all (an empty work list for the activation).
    """
    torch.manual_seed(1234 + num_tokens)
    device = torch.device("cuda")
    routing_logits = torch.randn(
        (num_tokens, NUM_EXPERTS), device=device, dtype=torch.bfloat16
    )
    if remote_only:
        remote = [
            e
            for e in range(NUM_EXPERTS)
            if not local_expert_offset <= e < local_expert_offset + local_num_experts
        ]
        assert len(remote) >= TOP_K, "needs expert parallelism to leave the rank empty"
        routing_logits[:, remote[:TOP_K]] += 20.0
    else:
        routing_logits[:, local_expert_offset] += 8.0
    hidden_states = torch.randn(
        (num_tokens, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (local_num_experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (local_num_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=device,
        dtype=torch.bfloat16,
    )
    quant_impl = FP8BlockScaleMoe(
        fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK
    )
    quant_weights = quant_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )
    quant_inputs = quant_impl.quantize_inputs(hidden_states)
    return dict(
        routing_logits=routing_logits,
        hidden_states=quant_inputs["hidden_states"],
        hidden_states_scale=quant_inputs["hidden_states_scale"],
        gemm1_weights=quant_weights["gemm1_weights"],
        gemm1_weights_scale=quant_weights["gemm1_scales"],
        gemm2_weights=quant_weights["gemm2_weights"],
        gemm2_weights_scale=quant_weights["gemm2_scales"],
    )


def _run(inputs, local_num_experts, local_expert_offset, enable_pdl, controls=None):
    """One fused MoE call.

    ``controls`` = (alpha, beta, clamp) for local expert 0, or None; each later expert gets a
    different value, so the per-expert lookup fetches distinct numbers.
    """
    kwargs = {}
    if controls is not None:
        alpha, beta, clamp = controls
        expert = torch.arange(
            local_num_experts, device=torch.device("cuda"), dtype=torch.float32
        )
        kwargs = dict(
            gemm1_alpha=alpha + 0.05 * expert,
            gemm1_beta=beta + 0.1 * expert,
            gemm1_clamp_limit=clamp + 0.5 * expert,
        )
    return trtllm_fp8_block_scale_moe(
        routing_logits=inputs["routing_logits"],
        routing_bias=None,
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        use_shuffled_weight=False,
        weight_layout=WeightLayout.MajorK.value,
        enable_pdl=enable_pdl,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        **kwargs,
    )


# (local experts, offset): every slot local, or one quarter of the experts (expert parallelism
# over four ranks), where three of four (token, slot) pairs belong to other ranks. The kernel
# walks the permuted rows when they are no more than the pairs (P <= pairs), else the pairs, and
# uses 32 lanes per 128-column block up to 8192 blocks, else 8. With 8 blocks per row and an
# 8-row GEMM tile:
#   all local: pairs list at every token count (tile padding always adds rows); 32 lanes up to
#              64 tokens, 8 lanes at 512 and 1024
#   quarter:   rows list from 8 tokens up (the local pairs fill the local experts' tiles, so
#              P <= pairs); at one token the 8 pairs may spread over several local experts, each
#              padded to a tile, so P can exceed the pairs and the pairs list runs; 32 lanes up
#              to 64 tokens, 8 lanes at 512 and 1024
# so both lists and both lane counts are exercised.
EP_CASES = [(NUM_EXPERTS, 0), (NUM_EXPERTS // 4, NUM_EXPERTS // 4)]


@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("local_num_experts, local_expert_offset", EP_CASES)
@pytest.mark.parametrize("num_tokens", [1, 8, 64, 1024])
def test_persistent_activation_is_selected(
    num_tokens, local_num_experts, local_expert_offset, enable_pdl
):
    """The fused MoE launches the persistent activation kernel, and no other, at every token count."""
    _skip_unless_sm100_class()
    inputs = _make_inputs(num_tokens, local_num_experts, local_expert_offset)

    def call():
        return _run(inputs, local_num_experts, local_expert_offset, enable_pdl)

    call()  # JIT build and warm-up outside the profiled window
    torch.cuda.synchronize()
    names = cuda_kernel_names(call)
    assert _launched(names, (PERSISTENT_KERNEL, OLD_KERNEL)) == [PERSISTENT_KERNEL], (
        names
    )


@pytest.mark.parametrize("controls", [None, (1.702, 1.0, 2.0)])
@pytest.mark.parametrize("local_num_experts, local_expert_offset", EP_CASES)
@pytest.mark.parametrize("num_tokens", [1, 8, 64, 512])
def test_persistent_activation_pdl_bitwise(
    num_tokens, local_num_experts, local_expert_offset, controls
):
    """PDL on and PDL off give the same bytes: the routing outputs the kernel reads before its wait are final."""
    _skip_unless_sm100_class()
    inputs = _make_inputs(num_tokens, local_num_experts, local_expert_offset)
    out_pdl = _run(inputs, local_num_experts, local_expert_offset, True, controls)
    out_plain = _run(inputs, local_num_experts, local_expert_offset, False, controls)
    torch.cuda.synchronize()
    assert torch.isfinite(out_pdl.float()).all()
    assert (out_pdl.float().abs().sum(dim=1) > 0).all(), (
        "every token has a local expert"
    )
    assert torch.equal(out_pdl, out_plain)
    if controls is not None:
        out_default = _run(inputs, local_num_experts, local_expert_offset, True)
        torch.cuda.synchronize()
        assert not torch.equal(out_pdl, out_default)


# 8 tokens: with the 8-row GEMM1 tile every local expert that receives a pair pads to one tile, so
# the padded rows are at most the (token, slot) pairs (8 local experts x 8 = 64) and the kernel
# takes the rows list, at its boundary when every local expert is hit. 32 and 64 tokens take it
# with margin: at the launcher's default tiles (16 and 32) the padded rows are about half the
# pairs, 8 local experts x one tile each, as long as no local expert holds more than one tile.
@pytest.mark.parametrize("controls", [False, True])
@pytest.mark.parametrize("num_tokens", [8, 32, 64])
def test_persistent_activation_ep_reference(num_tokens, controls):
    """Expert parallelism against the dequantized reference, with and without per-expert SwiGLU controls.

    The shared reference harness (``run_moe_test``) runs every expert locally, where the kernel walks
    the (token, slot) pairs. Here one quarter of the experts is local, so the kernel walks the permuted
    rows and resolves each row's expert through routing's tile map, the path the per-expert controls
    depend on. The reference computes all experts and gives the remote ones a routing weight of zero,
    which is what this rank's output is: the finalize sums only the rows that exist here.
    """
    _skip_unless_sm100_class()
    local_num_experts, local_expert_offset = EP_CASES[1]
    padding = 8
    torch.manual_seed(4321 + num_tokens)
    device = torch.device("cuda")
    logits = torch.randn((num_tokens, NUM_EXPERTS), device=device, dtype=torch.bfloat16)
    hidden_states = 2 * torch.randn(
        (num_tokens, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    ) / (HIDDEN_SIZE**0.5)
    gemm2_weights = torch.randn(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=device,
        dtype=torch.bfloat16,
    ) / (INTERMEDIATE_SIZE**0.5)
    expert = torch.arange(NUM_EXPERTS, device=device, dtype=torch.float32)
    if controls:
        # neighbouring experts far apart, so a lookup off by one GEMM tile would not match
        alpha = 0.5 + 0.5 * (expert % 4)
        beta = 0.25 * (expert % 3)
        clamp = 1.0 + 1.0 * (expert % 5)
    else:
        alpha = beta = clamp = None

    impl = FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK)
    weights = impl.quantize_weights(gemm1_weights, gemm2_weights, hidden_states)
    inputs = impl.quantize_inputs(hidden_states, weights["hidden_states_scale_global"])

    permute_info, scores = routing_reference_renormalize(
        logits, TOP_K, NUM_EXPERTS, padding
    )
    top_k_indices = permute_info["topKIndices"]
    is_local = (top_k_indices >= local_expert_offset) & (
        top_k_indices < local_expert_offset + local_num_experts
    )
    permute_info["topKLogits"] = permute_info["topKLogits"] * is_local.to(
        permute_info["topKLogits"].dtype
    )
    args = moe_args(
        num_tokens,
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        TOP_K,
        padding,
        inputs["hidden_states"],
        inputs["hidden_states_scale"],
        weights["hidden_states_scale_global"],
        scores,
        weights["gemm1_weights"],
        weights["gemm1_scales"],
        weights["gemm1_scales_global"],
        weights["gemm2_weights"],
        weights["gemm2_scales"],
        weights["gemm2_scales_global"],
        permute_info,
        False,
        ActivationType.Swiglu,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=clamp,
    )
    reference, _ = run_moe_reference_dsfp8(args)

    local = slice(local_expert_offset, local_expert_offset + local_num_experts)
    kwargs = {}
    if controls:
        kwargs = dict(
            gemm1_alpha=alpha[local].contiguous(),
            gemm1_beta=beta[local].contiguous(),
            gemm1_clamp_limit=clamp[local].contiguous(),
        )
    out = trtllm_fp8_block_scale_moe(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=weights["gemm1_weights"][local].contiguous(),
        gemm1_weights_scale=weights["gemm1_scales"][local].contiguous(),
        gemm2_weights=weights["gemm2_weights"][local].contiguous(),
        gemm2_weights_scale=weights["gemm2_scales"][local].contiguous(),
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        use_shuffled_weight=False,
        weight_layout=WeightLayout.MajorK.value,
        enable_pdl=True,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        **kwargs,
    )
    torch.cuda.synchronize()
    assert bool(is_local.any()), "some (token, slot) pair must land on this rank"
    tolerances = impl.get_tolerances()
    check_accuracy(
        reference,
        out.to(torch.float),
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        percent=tolerances["percent"],
    )


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_persistent_activation_empty_rank(num_tokens):
    """No (token, slot) pair on this rank, the common decode case under expert parallelism.

    The kernel is still launched (the grid comes from the host-side bound) and finds an empty
    work list; it must write nothing and the call must give the same bytes with PDL on and off.
    """
    _skip_unless_sm100_class()
    local_num_experts, local_expert_offset = EP_CASES[1]
    inputs = _make_inputs(
        num_tokens, local_num_experts, local_expert_offset, remote_only=True
    )

    def call():
        return _run(inputs, local_num_experts, local_expert_offset, True)

    out_pdl = call()
    out_plain = _run(inputs, local_num_experts, local_expert_offset, False)
    torch.cuda.synchronize()
    assert torch.isfinite(out_pdl.float()).all()
    assert (out_pdl.float() == 0).all(), "no local expert contributes to any token"
    assert torch.equal(out_pdl, out_plain)
    names = cuda_kernel_names(call)
    assert _launched(names, (PERSISTENT_KERNEL, OLD_KERNEL)) == [PERSISTENT_KERNEL], (
        names
    )
