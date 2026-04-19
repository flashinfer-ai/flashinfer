"""Tests for W4A16 (wMXFP4 x BF16) MoE kernel.

Covers the mixed-dtype MoE GEMM path where weights are 4-bit MXFP4
and activations are 16-bit BF16 (Hopper SM90 path via cutlass_fused_moe).

Correctness is verified by dequantizing MXFP4 weights to BF16 on host,
computing a reference MoE output in PyTorch, and comparing with kernel output
via ``torch.testing.assert_close(rtol=1e-1, atol=1e-1)`` — matching
``test_trtllm_cutlass_fused_moe.py`` and TensorRT-LLM's
``test_fused_moe_jiangs.py``.

The SM90 mixed-input GEMM consumes weights in a specific interleaved byte
layout (see ``interleave_moe_weights_for_hopper_mixed_gemm``); every test
below preprocesses weights through that helper before invoking the kernel,
mirroring TensorRT-LLM's weight-load path (PR #12451,
``trtllm::interleave_4bit_weights_for_Hopper_mixed_gemm``).

Target configuration: experts=256, topk=6, hidden_size=4096, intermediate_size=2048
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm90a_supported


# ============================================================================
# Reference implementations (reused from test_trtllm_cutlass_fused_moe.py)
# ============================================================================


def _dequant_mxfp4_batches_host(w_fp4, w_scale):
    """Dequantize MXFP4 weights on CPU for reference computation."""
    from flashinfer import mxfp4_dequantize_host

    return torch.stack(
        [
            mxfp4_dequantize_host(w_fp4[b, :, :], w_scale[b, :, :])
            for b in range(w_fp4.size(0))
        ]
    )


def _compute_with_experts(
    num_experts,
    x,
    w31_weight,
    w2_weight,
    selected_experts,
    routing_weights,
    alpha=None,
    beta=None,
    limit=None,
):
    """Reference MoE output with dequantized weights."""
    results = torch.zeros_like(x)
    for expert_id in range(num_experts):
        mask = selected_experts == expert_id
        if not mask.sum():
            continue
        batch_idx, nth_expert = torch.where(mask)
        w31_expert = w31_weight[expert_id]
        w2_expert = w2_weight[expert_id]
        w3_expert, w1_expert = torch.chunk(w31_expert, 2, dim=0)

        expert_inputs = x[batch_idx]
        if alpha is not None and limit is not None and beta is not None:
            x1 = expert_inputs @ w1_expert.t()
            x1 = x1.clamp_(min=None, max=limit)
            x1_scaled = x1 * torch.sigmoid(alpha * x1)
            x2 = expert_inputs @ w3_expert.t()
            x2 = x2.clamp_(min=-limit, max=limit) + beta
            inter = x1_scaled * x2
        else:
            inter = F.silu(expert_inputs @ w1_expert.t()) * (
                expert_inputs @ w3_expert.t()
            )
        output = inter @ w2_expert.t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output
    return results.view_as(x)


# ============================================================================
# Test configurations
# ============================================================================

# Coverage configs: (batch_size, hidden_size, num_experts, top_k, intermediate_size)
COVERAGE_CONFIGS = [
    # Batch sweep
    (1, 4096, 256, 6, 2048),
    (4, 4096, 256, 6, 2048),
    (16, 4096, 256, 6, 2048),
    (64, 4096, 256, 6, 2048),
    (128, 4096, 256, 6, 2048),
    (256, 4096, 256, 6, 2048),
    # Shape sweep
    (16, 2048, 256, 6, 1024),
    (16, 4096, 256, 6, 2048),
    (16, 7168, 256, 6, 4096),
    # Expert sweep
    (16, 4096, 8, 2, 2048),
    (16, 4096, 64, 4, 2048),
    (16, 4096, 256, 6, 2048),
    # TopK sweep
    (16, 4096, 256, 2, 2048),
    (16, 4096, 256, 4, 2048),
    (16, 4096, 256, 6, 2048),
    (16, 4096, 256, 8, 2048),
    # Edge cases
    (1, 4096, 256, 1, 2048),
    (8, 4096, 256, 6, 2048),
    (32, 4096, 256, 6, 2048),
    (512, 4096, 256, 6, 2048),
    (16, 4096, 128, 6, 2048),
    (16, 4096, 256, 6, 1024),
    (16, 4096, 256, 6, 4096),
    (4, 2048, 64, 4, 1024),
]

ACTIVATION_CONFIGS = [
    (4, 4096, 8, 4, 2048, None, None, None),
    (4, 4096, 8, 4, 2048, 0.5, 0.0, 7.0),
    (4, 4096, 8, 4, 2048, 1.702, 1.0, 7.0),
]

# Correctness configs mirroring TRT-LLM's W4A16 MoE coverage
# (test_fused_moe_jiangs.py uses h=768 / 1024 with group_size=32).
CORRECTNESS_CONFIGS = [
    (1, 128, 2, 2, 128),
    (4, 128, 4, 2, 128),
    (4, 768, 8, 2, 512),
    (4, 2048, 8, 4, 1024),
    (4, 4096, 8, 4, 2048),
]


# ============================================================================
# Core test runner
# ============================================================================


def _run_w4a16_moe(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    alpha=None,
    beta=None,
    limit=None,
    check_correctness=True,
    strict_correctness=True,
):
    """Run W4A16 MoE. If check_correctness=True, verify against dequantized reference.

    strict_correctness=True uses torch.testing.assert_close(rtol=1e-1, atol=1e-1)
    matching the upstream W4A16 test. strict_correctness=False instead allows up
    to 0.1% of elements to exceed the same tolerance, intended for large problem
    sizes where cumulative BF16 accumulation nudges a handful of elements past
    the strict bound (TRT-LLM's own test uses 1% outlier tolerance via
    check_accuracy(percent=0.99); we keep a tighter 99.9%).
    """
    import flashinfer.fused_moe as fused_moe

    torch.manual_seed(42)
    device = torch.device("cuda")

    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    # BF16 activation
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)

    # MXFP4 weights (random uint8 packed)
    w1 = torch.randint(0, 256, (e, 2 * n, k // 2), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, n // 2), device=device, dtype=torch.uint8)

    # MXFP4 scales
    w1_scale = torch.randint(
        118, 123, (e, 2 * n, k // 32), device=device, dtype=torch.uint8
    )
    w2_scale = torch.randint(
        118, 123, (e, k, n // 32), device=device, dtype=torch.uint8
    )

    # Routing
    router_logits = torch.randn(m, e, dtype=torch.bfloat16, device=device)
    routing_weights, selected_experts = torch.topk(
        F.softmax(router_logits.float(), dim=-1), top_k, dim=-1
    )
    routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(
        torch.float32
    )

    # Activation params
    if alpha is not None:
        alpha_t = torch.ones(e, device=device) * alpha
        limit_t = torch.ones(e, device=device) * limit
        beta_t = torch.ones(e, device=device) * beta
    else:
        alpha_t = limit_t = beta_t = None

    # The SM90 mixed-input GEMM expects MXFP4 weights AND scales in a specific
    # interleaved layout. Matching TensorRT-LLM PR #12451
    # (trtllm::interleave_4bit_weights_for_Hopper_mixed_gemm and
    # WFP4A16FusedMoEMethod.load_quant_scales).
    w1_interleaved = fused_moe.interleave_moe_weights_for_hopper_mixed_gemm(w1, "fp4")
    w2_interleaved = fused_moe.interleave_moe_weights_for_hopper_mixed_gemm(w2, "fp4")
    w1_scale_interleaved = fused_moe.interleave_moe_scales_for_hopper_mixed_gemm(w1_scale)
    w2_scale_interleaved = fused_moe.interleave_moe_scales_for_hopper_mixed_gemm(w2_scale)

    quant_scales = [
        w1_scale_interleaved.view(torch.int32),
        w2_scale_interleaved.view(torch.int32),
    ]
    output = torch.zeros_like(x)

    # Run kernel
    result = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int),
        routing_weights,
        w1_interleaved,
        w2_interleaved,
        torch.bfloat16,
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=quant_scales,
        use_w4_group_scaling=True,
        output=output,
    )
    flash_output = result[0] if isinstance(result, list) else result

    # Sanity checks
    assert torch.isfinite(flash_output).all(), (
        f"Non-finite output: NaN={torch.isnan(flash_output).sum()}, "
        f"Inf={torch.isinf(flash_output).sum()}"
    )
    assert flash_output.shape == (m, k), (
        f"Shape mismatch: {flash_output.shape} vs ({m}, {k})"
    )

    if check_correctness:
        dq_w1 = (
            _dequant_mxfp4_batches_host(w1.cpu(), w1_scale.cpu())
            .cuda()
            .to(torch.bfloat16)
        )
        dq_w2 = (
            _dequant_mxfp4_batches_host(w2.cpu(), w2_scale.cpu())
            .cuda()
            .to(torch.bfloat16)
        )
        ref_output = _compute_with_experts(
            e,
            x,
            dq_w1,
            dq_w2,
            selected_experts,
            routing_weights,
            alpha,
            beta,
            limit,
        )
        if strict_correctness:
            torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)
        else:
            diff = torch.abs(ref_output.float() - flash_output.float())
            tol = 0.1 + 1e-1 * torch.abs(ref_output.float())
            close_pct = (diff <= tol).float().mean().item()
            assert close_pct >= 0.999, (
                f"Only {close_pct:.4%} of elements within tolerance (need >= 99.9%). "
                f"max_abs_err={diff.max().item():.4f}"
            )

    return flash_output


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")), reason="W4A16 MoE requires SM90"
)
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size",
    CORRECTNESS_CONFIGS,
    ids=[f"m{c[0]}_h{c[1]}_e{c[2]}_k{c[3]}" for c in CORRECTNESS_CONFIGS],
)
def test_w4a16_moe_correctness(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """Strict correctness verification against dequantized reference (small configs)."""
    _run_w4a16_moe(
        batch_size,
        hidden_size,
        num_experts,
        top_k,
        intermediate_size,
        check_correctness=True,
    )


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")), reason="W4A16 MoE requires SM90"
)
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size",
    COVERAGE_CONFIGS,
    ids=[f"m{c[0]}_h{c[1]}_e{c[2]}_k{c[3]}_n{c[4]}" for c in COVERAGE_CONFIGS],
)
def test_w4a16_moe_coverage(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """Coverage sweep with correctness verification (99.9% elements within tolerance)."""
    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) > num_experts ({num_experts})")
    _run_w4a16_moe(
        batch_size,
        hidden_size,
        num_experts,
        top_k,
        intermediate_size,
        check_correctness=True,
        strict_correctness=False,
    )


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")), reason="W4A16 MoE requires SM90"
)
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size,alpha,beta,limit",
    ACTIVATION_CONFIGS,
    ids=["swiglu_default", "alpha_0.5", "alpha_1.702"],
)
def test_w4a16_moe_activations(
    batch_size, hidden_size, num_experts, top_k, intermediate_size, alpha, beta, limit
):
    """Correctness test with different activation configurations (small configs)."""
    _run_w4a16_moe(
        batch_size,
        hidden_size,
        num_experts,
        top_k,
        intermediate_size,
        alpha,
        beta,
        limit,
        check_correctness=True,
    )


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")), reason="W4A16 MoE requires SM90"
)
def test_w4a16_moe_core_config():
    """Primary target configuration: experts=256, topk=6, hidden=4096, inter=2048."""
    _run_w4a16_moe(
        batch_size=4,
        hidden_size=4096,
        num_experts=256,
        top_k=6,
        intermediate_size=2048,
        check_correctness=True,
        strict_correctness=False,
    )
