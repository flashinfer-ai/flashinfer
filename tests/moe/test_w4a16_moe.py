"""Tests for W4A16 (wMXFP4 x BF16) MoE kernel.

Covers the mixed-dtype MoE GEMM path where weights are 4-bit MXFP4
and activations are 16-bit BF16 (Hopper SM90 path via cutlass_fused_moe).

Target configuration: experts=256, topk=6, hidden_size=4096, intermediate_size=2048
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm90a_supported


# ============================================================================
# Test configurations
# ============================================================================

# Core config from task spec
CORE_CONFIGS = [
    # (batch_size, hidden_size, num_experts, top_k, intermediate_size)
    (4, 4096, 256, 6, 2048),
]

# Sweep batch_size
BATCH_CONFIGS = [
    (1, 4096, 256, 6, 2048),
    (4, 4096, 256, 6, 2048),
    (16, 4096, 256, 6, 2048),
    (64, 4096, 256, 6, 2048),
    (128, 4096, 256, 6, 2048),
    (256, 4096, 256, 6, 2048),
]

# Sweep hidden_size / intermediate_size
SHAPE_CONFIGS = [
    (16, 2048, 256, 6, 1024),
    (16, 4096, 256, 6, 2048),
    (16, 7168, 256, 6, 4096),
]

# Sweep num_experts
EXPERT_CONFIGS = [
    (16, 4096, 8, 2, 2048),
    (16, 4096, 64, 4, 2048),
    (16, 4096, 256, 6, 2048),
]

# Sweep top_k
TOPK_CONFIGS = [
    (16, 4096, 256, 2, 2048),
    (16, 4096, 256, 4, 2048),
    (16, 4096, 256, 6, 2048),
    (16, 4096, 256, 8, 2048),
]

# Activation types
ACTIVATION_CONFIGS = [
    (16, 4096, 256, 6, 2048, None, None, None),  # Swiglu default
    (16, 4096, 256, 6, 2048, 0.5, 0.0, 7.0),  # with alpha/beta/limit
    (16, 4096, 256, 6, 2048, 1.702, 1.0, 7.0),
]

# Edge cases and additional coverage
EDGE_CONFIGS = [
    (1, 4096, 256, 1, 2048),  # single token, topk=1
    (8, 4096, 256, 6, 2048),  # moderate batch
    (32, 4096, 256, 6, 2048),  # larger batch
    (512, 4096, 256, 6, 2048),  # large batch
    (16, 4096, 128, 6, 2048),  # different expert count
    (16, 4096, 256, 6, 1024),  # smaller intermediate
    (16, 4096, 256, 6, 4096),  # larger intermediate
    (4, 2048, 64, 4, 1024),  # small config
]

# Small configs for quick validation
QUICK_CONFIGS = [
    (4, 256, 8, 2, 128),
    (4, 4096, 256, 6, 2048),
]

ALL_CONFIGS = (
    BATCH_CONFIGS + SHAPE_CONFIGS + EXPERT_CONFIGS + TOPK_CONFIGS + EDGE_CONFIGS
)


def _run_w4a16_moe(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    alpha=None,
    beta=None,
    limit=None,
):
    """Run W4A16 MoE and verify output sanity (finite values, correct shape)."""
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

    # MXFP4 scales (uint8, represents FP8 e8m0 scale per 32 elements)
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

    quant_scales = [w1_scale.view(torch.int32), w2_scale.view(torch.int32)]

    output = torch.zeros_like(x)

    result = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int),
        routing_weights,
        w1.contiguous().view(torch.uint8),
        w2.contiguous().view(torch.uint8),
        torch.bfloat16,
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=quant_scales,
        use_w4_group_scaling=True,
        output=output,
    )
    out = result[0] if isinstance(result, list) else result

    # Basic sanity
    assert torch.isfinite(out).all(), (
        f"Non-finite output: NaN={torch.isnan(out).sum()}, Inf={torch.isinf(out).sum()}"
    )
    assert out.shape == (m, k), f"Shape mismatch: {out.shape} vs ({m}, {k})"

    return out


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.skipif(not is_sm90a_supported(), reason="W4A16 MoE requires SM90 (Hopper)")
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size",
    QUICK_CONFIGS,
    ids=[f"m{c[0]}_h{c[1]}_e{c[2]}_k{c[3]}" for c in QUICK_CONFIGS],
)
def test_w4a16_moe_quick(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """Quick functional validation of W4A16 MoE."""
    _run_w4a16_moe(batch_size, hidden_size, num_experts, top_k, intermediate_size)


@pytest.mark.skipif(not is_sm90a_supported(), reason="W4A16 MoE requires SM90 (Hopper)")
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size",
    ALL_CONFIGS,
    ids=[f"m{c[0]}_h{c[1]}_e{c[2]}_k{c[3]}_n{c[4]}" for c in ALL_CONFIGS],
)
def test_w4a16_moe_coverage(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """Coverage test across batch/shape/expert/topk configurations."""
    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) > num_experts ({num_experts})")
    _run_w4a16_moe(batch_size, hidden_size, num_experts, top_k, intermediate_size)


@pytest.mark.skipif(not is_sm90a_supported(), reason="W4A16 MoE requires SM90 (Hopper)")
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size,alpha,beta,limit",
    ACTIVATION_CONFIGS,
    ids=["swiglu_default", "alpha_0.5", "alpha_1.702"],
)
def test_w4a16_moe_activations(
    batch_size, hidden_size, num_experts, top_k, intermediate_size, alpha, beta, limit
):
    """Test W4A16 MoE with different activation configurations."""
    _run_w4a16_moe(
        batch_size,
        hidden_size,
        num_experts,
        top_k,
        intermediate_size,
        alpha,
        beta,
        limit,
    )


@pytest.mark.skipif(not is_sm90a_supported(), reason="W4A16 MoE requires SM90 (Hopper)")
def test_w4a16_moe_core_config():
    """Test the primary target configuration: experts=256, topk=6, hidden=4096, inter=2048."""
    _run_w4a16_moe(
        batch_size=4,
        hidden_size=4096,
        num_experts=256,
        top_k=6,
        intermediate_size=2048,
    )
