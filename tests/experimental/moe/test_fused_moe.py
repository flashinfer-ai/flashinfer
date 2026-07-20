# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""moe.fused_moe: end-to-end W4A16 fused MoE under the serving contract.

Numerical correctness is cross-checked in test_ep_moe.py (two-rank
expert-parallel partials must sum to this op's full output). Here the planned
lifecycle replays under CUDA-graph capture with kernel resolution frozen —
zero compile-cache misses may occur inside the capture, per the warm ->
freeze -> capture serving discipline.
"""

from __future__ import annotations

import torch

import flashinfer.experimental.sm12x as sm12x
from flashinfer.experimental.sm12x._lib.compiler import compile_cache_info
from flashinfer.experimental.sm12x._lib.intrinsics import swizzle_block_scale

from .._reference.helpers import make_tp_moe_fp4_binding, prepare_tp_moe_fp4_experts
from ..conftest import require_sm12x


def make_modelopt_weights(
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, ...]:
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = swizzle_block_scale(
        (
            torch.rand(experts, 2 * intermediate_size, hidden_size // 16, device="cuda")
            * 0.25
            + 0.03125
        ).to(torch.float8_e4m3fn)
    )
    w2_scale = swizzle_block_scale(
        (
            torch.rand(experts, hidden_size, intermediate_size // 16, device="cuda")
            * 0.25
            + 0.03125
        ).to(torch.float8_e4m3fn)
    )
    w13_alpha = (torch.rand(experts, device="cuda") * 0.1 + 0.05).float()
    w2_alpha = (torch.rand(experts, device="cuda") * 0.1 + 0.05).float()
    return w13, w13_scale, w13_alpha, w2, w2_scale, w2_alpha


def prepare_experts(
    a: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    expert_ids: torch.Tensor,
):
    w13, w13_scale, w13_alpha, w2, w2_scale, w2_alpha = weights
    selected = expert_ids.to(device=w13.device, dtype=torch.long)
    local_e = int(selected.numel())
    unit = torch.ones(local_e, dtype=torch.float32, device=a.device)
    return prepare_tp_moe_fp4_experts(
        a=a,
        a1_gscale=unit,
        w1_fp4=w13.index_select(0, selected).clone(),
        w1_blockscale=w13_scale.index_select(0, selected).clone(),
        w1_alphas=w13_alpha.index_select(0, selected).clone(),
        a2_gscale=unit,
        w2_fp4=w2.index_select(0, selected).clone(),
        w2_blockscale=w2_scale.index_select(0, selected).clone(),
        w2_alphas=w2_alpha.index_select(0, selected).clone(),
        activation="silu",
        quant_mode="w4a16",
    )


def test_run_w4a16_replays_under_cuda_graph_with_frozen_resolution() -> None:
    require_sm12x()
    torch.manual_seed(20260715)

    from flashinfer.experimental.sm12x.moe import fused_moe

    global_e, hidden_size, intermediate_size = 4, 128, 128
    m, topk = 8, 2
    a = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    weights = make_modelopt_weights(
        experts=global_e, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    experts = prepare_experts(a, weights, torch.arange(global_e))
    topk_ids = torch.randint(0, global_e, (m, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    binding = make_tp_moe_fp4_binding(
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=torch.empty_like(a),
        quant_mode="w4a16",
    )

    eager = fused_moe.run(binding=binding).clone()
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(eager).item()) > 0

    fused_moe.run(binding=binding)  # resolve every kernel variant pre-capture
    torch.cuda.synchronize()

    misses_before = compile_cache_info()["compile_misses"]
    sm12x.freeze_kernel_resolution("fused-moe graph capture test")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused_moe.run(binding=binding)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    finally:
        sm12x.unfreeze_kernel_resolution()

    assert compile_cache_info()["compile_misses"] == misses_before, (
        "no kernel may compile during or after warm capture"
    )
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
