# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""gemm.wo_projection: planned lifecycle parity (plan -> bind -> run) against
a quantized torch reference, plus the fused inverse-RoPE variant replaying
under CUDA-graph capture with poisoned scale padding.

Curated from b12x tests/test_gemm_wo_projection.py (1.2k lines upstream);
split-GEMM leaves, packing round-trips, and byte-identity policy tests stay
in the b12x repo.
"""

from __future__ import annotations

import torch

from flashinfer.experimental.sm12x.gemm import wo_projection as wo
from flashinfer.experimental.sm12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    quantize_wo_projection_weights_mxfp8_torch,
)

from ..conftest import require_sm12x


def test_plan_bind_run_singleton_group_matches_quantized_reference() -> None:
    """TP8 collapses DSV4's eight output groups to one local WO group."""
    require_sm12x()
    torch.manual_seed(31005)

    tokens, groups, group_width, rank, hidden = 3, 1, 512, 128, 128
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    )
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )

    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a_grd, wo_b_hgr)
    plan = wo.plan(
        wo.Caps(
            device=x_tgd.device,
            max_tokens=tokens,
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
            dtype=x_tgd.dtype,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=x_tgd.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    binding = wo.bind(
        plan, scratch=scratch, source_tgd=x_tgd, weights=weights, expected_m=tokens
    )
    actual = wo.run(binding=binding)
    torch.cuda.synchronize()

    x_q = wo.quantize_input(x_tgd)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    wo_a_deq = dequantize_mxfp8_rows_torch(weights.wo_a.values, weights.wo_a.scale_rows)
    tmp = (x_deq @ wo_a_deq.T).to(torch.bfloat16).unsqueeze(-1)
    tmp_q = wo.quantize_input_b(tmp)
    tmp_deq = dequantize_mxfp8_rows_torch(tmp_q.values, tmp_q.scale_rows)
    wo_b_deq = dequantize_mxfp8_rows_torch(weights.wo_b.values, weights.wo_b.scale_rows)
    expected = tmp_deq @ wo_b_deq.T

    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


def test_run_inv_rope_replays_under_graph_with_uninitialized_scale_padding() -> None:
    require_sm12x()
    torch.manual_seed(31007)

    tokens = 1
    groups = 2
    heads_per_group = 4
    nope_dim = 96
    rope_dim = 32
    head_dim = nope_dim + rope_dim
    group_width = heads_per_group * head_dim
    rank, hidden = 64, 128
    o = (
        torch.randn(
            (tokens, groups * heads_per_group, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 4
    ).contiguous()
    positions = torch.zeros((tokens,), device="cuda", dtype=torch.long)
    cos_sin_cache = torch.zeros((4, rope_dim), device="cuda", dtype=torch.float32)
    cos_sin_cache[:, : rope_dim // 2] = 1
    wo_a = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )
    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a, wo_b)

    def run_once() -> torch.Tensor:
        return wo.run_inv_rope(
            o,
            positions,
            cos_sin_cache,
            weights,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            expected_m=1,
        )

    # Warm all compilation/allocation before capture, then verify replay
    # observes changed inputs without allocating or depending on stale scale
    # padding.
    run_once()
    run_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    o.copy_(torch.randn_like(o) / 4)
    graph.replay()
    torch.cuda.synchronize()
    replayed = captured.clone()
    expected = run_once().clone()
    torch.cuda.synchronize()

    assert bool(torch.isfinite(replayed).all().item())
    assert bool((replayed != 0).any().item())
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
