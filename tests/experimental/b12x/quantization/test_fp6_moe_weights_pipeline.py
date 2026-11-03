"""Offline MX-FP6 MoE weight-quantization pipeline: round-trip + kernel equivalence."""

from __future__ import annotations

# TODO(port): the kernel-equivalence half of this suite drove the historical
# b12x.integration.tp_moe host API, which is not part of the current API.
# The test is kept as reference behind pytest.skip("pending: fused_moe-based
# rewrite"); rewrite it against the b12x.moe.fused_moe plan/bind/run flow
# once the w6a8_mx run path lands.

import pytest
import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x._lib.fp6 import (
    FLOAT6_E2M3_MAX,
    SF_VEC_SIZE_FP6,
    _ue8m0_scale_from_block_max,
    quantize_grouped_mxfp6_torch,
)
from b12x.quantization.mxfp6 import (
    FP6MoEWeights,
    load_fp6_moe_weights,
    quantize_moe_weights_to_fp6,
    save_fp6_moe_weights,
)
from b12x.testing.reference.helpers import require_b12x


def _reference_recipe(w1_bf, w2_bf, *, fmt="e2m3"):
    """The validated test recipe: quantize_grouped + swizzle_block_scale."""
    experts, w1_rows, k = w1_bf.shape
    n = w2_bf.shape[2]
    device = w1_bf.device
    gs1 = torch.ones(experts, device=device, dtype=torch.float32)
    gs2 = torch.ones(experts, device=device, dtype=torch.float32)
    row1 = torch.full((experts,), w1_rows, dtype=torch.int32, device=device)
    row2 = torch.full((experts,), k, dtype=torch.int32, device=device)
    w1_packed, _ = quantize_grouped_mxfp6_torch(w1_bf, row1, gs1, fmt=fmt)
    w2_packed, _ = quantize_grouped_mxfp6_torch(w2_bf, row2, gs2, fmt=fmt)
    w1_fp6 = w1_packed.permute(2, 0, 1).contiguous()
    w2_fp6 = w2_packed.permute(2, 0, 1).contiguous()

    def _sf(values, gs):
        g, rows, cols = values.shape
        blocks = cols // SF_VEC_SIZE_FP6
        scales = torch.zeros((g, rows, blocks), dtype=torch.uint8, device=device)
        for gi in range(g):
            sl = values[gi].float().view(rows, blocks, SF_VEC_SIZE_FP6)
            scales[gi] = _ue8m0_scale_from_block_max(
                sl.abs().amax(-1) * float(gs[gi]), FLOAT6_E2M3_MAX
            )
        return swizzle_block_scale(scales.view(torch.float8_e8m0fnu)).view(torch.uint8)

    return w1_fp6, _sf(w1_bf, gs1), w2_fp6, _sf(w2_bf, gs2)


def _run_kernel(x, weights: FP6MoEWeights, topk_ids, topk_weights):
    # TODO(port): drove the retired pre-facade b12x.integration.tp_moe host API, which
    # is not part of the current API; rewrite against the
    # b12x.moe.fused_moe plan/bind/run flow once the w6a8_mx run path lands.
    from b12x.integration.tp_moe import (  # retired pre-facade API (see TODO)
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    workspace = allocate_tp_moe_workspace(
        x,
        weights.a1_gscale,
        weights.w1_fp6,
        weights.a2_gscale,
        weights.w2_fp6,
        topk_ids,
        quant_mode="w6a6",
        input_scales_static=True,
    )
    out = b12x_moe_fp6(
        x,
        weights.a1_gscale,
        weights.w1_fp6,
        weights.w1_blockscale,
        weights.w1_alphas,
        weights.a2_gscale,
        weights.w2_fp6,
        weights.w2_blockscale,
        weights.w2_alphas,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        source_format=weights.source_format,
    )
    torch.cuda.synchronize()
    return out


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


def test_fp6_moe_weight_pipeline_roundtrip_and_equivalence(tmp_path) -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    require_b12x()
    device = torch.device("cuda")
    m, k, n, experts, topk = 4, 128, 128, 2, 1
    torch.manual_seed(11)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.randint(0, experts, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).to(
        torch.float32
    )
    w1_bf = torch.randn(experts, 2 * n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(experts, k, n, device=device, dtype=torch.bfloat16) * 0.15

    weights = quantize_moe_weights_to_fp6(w1_bf, w2_bf, source_format="mxfp6_default")
    assert weights.w1_fp6.shape == (experts, 2 * n, k * 3 // 4)
    assert weights.w2_fp6.shape == (experts, k, n * 3 // 4)

    # Block scales must match the validated swizzle recipe bit-for-bit.
    _, w1_sf_ref, _, w2_sf_ref = _reference_recipe(w1_bf, w2_bf)
    assert torch.equal(weights.w1_blockscale, w1_sf_ref)
    assert torch.equal(weights.w2_blockscale, w2_sf_ref)

    out_pipeline = _run_kernel(x, weights, topk_ids, topk_weights)

    # Save -> load must reproduce identical tensors and identical kernel output.
    path = str(tmp_path / "fp6_moe.pt")
    save_fp6_moe_weights(weights, path)
    loaded = load_fp6_moe_weights(path, device=device)
    assert torch.equal(loaded.w1_fp6, weights.w1_fp6)
    assert torch.equal(loaded.w2_blockscale, weights.w2_blockscale)
    assert loaded.k == k and loaded.n == n and loaded.num_experts == experts
    out_loaded = _run_kernel(x, loaded, topk_ids, topk_weights)
    assert _cos(out_pipeline, out_loaded) > 0.999

    # Equivalence to the reference recipe fed through the same kernel.
    w1_fp6_r, w1_sf_r, w2_fp6_r, w2_sf_r = _reference_recipe(w1_bf, w2_bf)
    ref_weights = FP6MoEWeights(
        w1_fp6=w1_fp6_r,
        w1_blockscale=w1_sf_r,
        w1_alphas=weights.w1_alphas,
        w2_fp6=w2_fp6_r,
        w2_blockscale=w2_sf_r,
        w2_alphas=weights.w2_alphas,
        a1_gscale=weights.a1_gscale,
        a2_gscale=weights.a2_gscale,
        num_experts=experts,
        k=k,
        n=n,
        weight_fmt="e2m3",
        source_format="mxfp6_default",
        activation="silu",
    )
    out_ref = _run_kernel(x, ref_weights, topk_ids, topk_weights)
    cos = _cos(out_pipeline, out_ref)
    print(f"\nFP6 weight pipeline vs reference recipe: cos={cos:.5f}")
    assert cos > 0.99, f"pipeline weights diverge from reference recipe: cos={cos:.5f}"
