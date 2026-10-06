"""IQ2_XS expert reduction precision across prepared graph replays."""

from pathlib import Path

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from benchmarks.experimental.b12x.benchmark_iq2_xs_moe import (
    capture_lifetime,
    check,
    make_inputs,
    prepare_experts,
    reference,
)
from benchmarks.experimental.b12x.iq2_xs_checkpoint import IQ2XSLayer
from benchmarks.experimental.b12x.moe_preparation import request_for_capacity
from b12x.testing.reference.helpers import require_b12x
from tests.experimental.b12x.moe.test_iq2_xs import blocks


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "q8_0"])
def test_direct_topk_sum_is_stable_across_graph_replays(mapped, codec):
    device = require_b12x()
    ids = tuple(reversed(range(16))) if mapped else tuple(range(16))
    layer = IQ2XSLayer(
        moe.BlockQuantWeights(blocks(e=16, n=1280, k=1024, codec=codec), blocks(e=16, n=1024, k=1280, codec=codec), codec=codec),
        1024, 1280, 16, 14, ids, Path("synthetic-iq2-xs"), 0, 1, 0,
    )
    experts, _ = prepare_experts(layer, device, activation="relu2")
    declaration = moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(max_tokens=8, top_k=14, route_num_experts=16),
        routing=moe.RoutingSpec(deterministic_output=False),
        override=moe.MoeDecodeConfig(
            backend="w4a16", route_planner="internal", max_active_clusters=None,
            w4a16_route_mode="direct",
        ),
    )
    warmup = make_inputs(layer, 8, device, mapped=mapped)

    def factory(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        bound = state.bind(
            scratch=scratch, a=warmup.x, topk_ids=warmup.ids,
            topk_weights=warmup.probabilities, output=warmup.output,
            route_expert_map=warmup.expert_map,
        )
        return PreparedCall(run=bound.run, output=warmup.output, owners=(scratch, bound, warmup))

    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        session.prepare((request_for_capacity(
            declaration, name="iq2-fp32-route-sum", calls={8: factory},
        ),))
        state = require_prepared(declaration, "moe.decode")
        state = state.variants[8] if hasattr(state, "variants") else state
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        inputs = make_inputs(layer, 1, device, mapped=mapped)
        expected = reference(layer, inputs, activation="relu2")
        binding = moe.bind(
            declaration, scratch=scratch, a=inputs.x, topk_ids=inputs.ids,
            topk_weights=inputs.probabilities, output=inputs.output,
            route_expert_map=inputs.expert_map,
        )
        moe.run(binding=binding)
        graph = torch.cuda.CUDAGraph()
        try:
            with capture_lifetime(), session.capture(), torch.cuda.graph(graph):
                moe.run(binding=binding)
            first = None
            for _ in range(64):
                inputs.output.fill_(float("nan"))
                graph.replay()
                check(inputs.output, expected)
                if first is None:
                    first = inputs.output.clone()
                else:
                    assert torch.equal(inputs.output, first), "route sum changed across graph replays"
        finally:
            graph.reset()
