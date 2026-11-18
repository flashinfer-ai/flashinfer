"""moe.fused_moe: end-to-end W4A16 fused MoE under the serving contract.

Numerical correctness is cross-checked in test_ep_moe.py (two-rank
expert-parallel partials must sum to this op's full output). Here the planned
lifecycle replays under CUDA-graph capture with kernel resolution frozen —
zero compile-cache misses may occur inside the capture, per the warm ->
freeze -> capture serving discipline.
"""

from __future__ import annotations

import torch

import b12x
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.compiler import compile_cache_info
from b12x._lib.intrinsics import swizzle_block_scale

from b12x.testing.reference.helpers import (
    make_tp_moe_fp4_binding,
    prepare_tp_moe_fp4_experts,
)
from ..conftest import require_b12x


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
    require_b12x()
    torch.manual_seed(20260715)

    from b12x.moe import fused_moe

    global_e, hidden_size, intermediate_size = 4, 128, 128
    m, topk = 8, 2
    a = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    weights = make_modelopt_weights(
        experts=global_e, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    experts = prepare_experts(a, weights, torch.arange(global_e))
    topk_ids = torch.randint(0, global_e, (m, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    with make_tp_moe_fp4_binding(
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=torch.empty_like(a),
        quant_mode="w4a16",
    ) as binding:
        eager = fused_moe.run(binding=binding).clone()
        torch.cuda.synchronize()
        assert int(torch.count_nonzero(eager).item()) > 0

        fused_moe.run(binding=binding)  # resolve every kernel variant pre-capture
        torch.cuda.synchronize()

        misses_before = compile_cache_info()["compile_misses"]
        with kernel_resolution_guard("fused-moe graph capture test"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fused_moe.run(binding=binding)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()

        assert compile_cache_info()["compile_misses"] == misses_before, (
            "no kernel may compile during or after warm capture"
        )
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        del graph


def test_run_w4a16_m9_graph_replay_with_prequeued_aux_work() -> None:
    """The route-packed W4A16 grid must remain valid beside aux work.

    M=9 is the first size above the fused-micro decode range. It enters the
    route-packed W4A16 body, whose FC1/activation/FC2 phases synchronize every
    CTA through software grid barriers. Prequeued shared-expert work must not
    prevent whole-grid admission when the serving graph replays.
    """
    require_b12x()
    torch.manual_seed(20260723)

    from b12x.moe import fused_moe

    device = torch.device("cuda")
    global_e, hidden_size, intermediate_size = 16, 6144, 512
    m, topk = 9, 8
    a = (torch.randn(m, hidden_size, device=device) * 0.25).to(torch.bfloat16)
    weights = make_modelopt_weights(
        experts=global_e,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    experts = prepare_experts(a, weights, torch.arange(global_e))
    topk_ids = torch.randint(0, global_e, (m, topk), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1)
    output = torch.empty_like(a)
    with make_tp_moe_fp4_binding(
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w4a16",
    ) as binding:
        fused_moe.run(binding=binding)
        torch.cuda.synchronize()
        expected = output.clone()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
            fused_moe.run(binding=binding)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        aux_stream = torch.cuda.Stream()
        aux_a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        aux_b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        aux_out = torch.empty_like(aux_a)
        output.zero_()
        with torch.cuda.stream(aux_stream):
            for _ in range(16):
                torch.mm(aux_a, aux_b, out=aux_out)
        graph.replay()
        torch.cuda.current_stream().wait_stream(aux_stream)
        torch.cuda.synchronize()

        assert output.isfinite().all()
        assert output.abs().sum().item() > 0
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=0.0)
        del graph
