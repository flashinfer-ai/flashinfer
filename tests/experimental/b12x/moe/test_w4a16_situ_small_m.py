"""Small-M MXFP4 W4A16 experts apply the SiTU nonlinearity of Kimi-K3."""

from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.kernels.reference import moe_reference_w4a16_fp4_e8m0_k32
from b12x.testing.reference.w4a16_reference import compare_to_reference
from ..conftest import require_b12x


def _e8m0(shape, *, offset=0, base=119):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    storage = (torch.arange(numel, device="cuda", dtype=torch.int64) + offset) % 4 + base
    return storage.to(torch.uint8).reshape(shape)


@pytest.mark.parametrize("route_mode", ["direct", "packed"])
@pytest.mark.parametrize("rows", [1, 2, 8])
@pytest.mark.parametrize("nonlinearity", ["situ", "silu"])
def test_mxfp4_a16_small_m_matches_reference(rows, route_mode, nonlinearity):
    """Kimi-K3's TP16 expert geometry selects the small-M direct kernel for decode rows."""
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    from b12x.preparation import PreparationSession, PreparedCall

    device = require_b12x()
    torch.manual_seed(3584 + rows)
    experts, hidden, intermediate, topk = 64, 3584, 192, 16
    raw = (
        torch.randint(256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8, device=device),
        # Large FC1 scales drive SiTU's tanh terms into saturation, where it departs from SiLU.
        _e8m0((experts, 2 * intermediate, hidden // 32), base=124),
        torch.ones(experts, device=device),
        torch.randint(256, (experts, hidden, intermediate // 2), dtype=torch.uint8, device=device),
        _e8m0((experts, hidden, intermediate // 32), offset=1),
        torch.ones(experts, device=device),
    )
    x = (torch.randn(rows, hidden, device=device) * 0.125).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, device=device)[:topk] for _ in range(rows)]).to(torch.int32)
    weights = torch.softmax(torch.randn(rows, topk, device=device), dim=-1)
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(mode="a16", nonlinearity=nonlinearity, io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=experts, hidden_size=hidden, intermediate_size=intermediate),
    )
    prepared = fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
        w13=raw[0].clone(), w13_block_scales=raw[1].clone(), w13_global_scales=raw[2].clone(),
        w2=raw[3].clone(), w2_block_scales=raw[4].clone(), w2_global_scales=raw[5].clone()))
    execution = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(max_tokens=rows, top_k=topk),
        routing=fused_moe.RoutingSpec(),
        override=MoeDecodeConfig(backend="w4a16", route_planner="internal", max_active_clusters=None,
                                 w4a16_route_mode=route_mode),
    )
    output = torch.empty_like(x)
    bound = {}

    def prepare(state):
        scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in state.scratch.scratch_specs())
        bound["binding"] = state.bind(scratch=scratch, a=x, experts=prepared, topk_weights=weights,
                                      topk_ids=ids, output=output)
        return PreparedCall(run=lambda: state.run(bound["binding"]), owners=scratch)

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare((execution.request(name=f"w4a16-{route_mode}-{nonlinearity}-{rows}", prepare_call=prepare),))
        expected = moe_reference_w4a16_fp4_e8m0_k32(x, *raw[:1], raw[1].view(torch.float8_e8m0fnu), raw[2],
            raw[3], raw[4].view(torch.float8_e8m0fnu), raw[5], ids, weights, experts, hidden, intermediate,
            activation=nonlinearity, w13_layout="w13")
        metrics = compare_to_reference(output, expected)
        assert metrics.cos >= 0.9975, (route_mode, metrics)
