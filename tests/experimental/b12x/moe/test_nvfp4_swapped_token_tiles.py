"""Validate narrow swapped-FC1 tiles with live routes and frozen graph replay."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe._shared.kernels.reference import moe_reference_nvfp4
from b12x.preparation import PreparationSession, PreparedCall

from .test_cute_migration_moe_standard_corpus import (
    _assert_oracle,
    _make_inputs,
    _make_nvfp4_weights,
)
from ..conftest import require_b12x


@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize("intermediate", [160, 192, 320])
@pytest.mark.parametrize("tile_m,planner,share_input,topk", [
    (16, "internal", False, 2), (16, "triton", False, 2),
    (32, "internal", False, 2),
    (16, "internal", True, 2), (16, "triton", True, 2),
    (16, "triton", True, 8), (16, "triton", True, 10),
])
def test_swapped_token_tile_live_replay(intermediate, tile_m, planner, share_input, topk, fast_math):
    device = require_b12x()
    capacity, hidden, num_experts = min(33, 256 // topk), 512, max(4, 2 * topk)
    weights = _make_nvfp4_weights(
        device, seed=819, num_experts=num_experts, hidden_size=hidden,
        intermediate_size=intermediate,
        logical_intermediate_size=160 if intermediate == 192 else None,
    )
    # Use normal E4M3 activation scales to isolate tile layout from the
    # independently tested subnormal-scale decoder contract. Keep the physical
    # weights unchanged when varying each expert's activation quantization scale.
    a1_scale = torch.linspace(32., 64., num_experts, device=device)
    if share_input:
        a1_scale.fill_(64.)
    a2_scale = torch.linspace(256., 512., num_experts, device=device)
    weights = replace(weights,
        a1_scale=a1_scale, a2_scale=a2_scale,
        w1_alpha=weights.w1_alpha / a1_scale,
        w2_alpha=weights.w2_alpha / a2_scale)
    inputs = _make_inputs(device, m=capacity, seed=820, route_shift=0,
                          num_experts=num_experts, hidden_size=hidden, topk=topk)
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=num_experts, hidden_size=hidden,
                                      intermediate_size=intermediate),
    )
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=fused_moe.PackedWeights(
        w13=weights.w1_fp4, w2=weights.w2_fp4,
        w13_block_scales=weights.w1_scale, w2_block_scales=weights.w2_scale,
        w13_global_scales=weights.w1_alpha * weights.a1_scale,
        w2_global_scales=weights.w2_alpha * weights.a2_scale,
        input_scale=weights.a1_scale, intermediate_scale=weights.a2_scale,
        immutable_input_scales=share_input,
    ))
    config = fused_moe.MoeDecodeConfig(backend="dynamic", route_planner=planner,
        max_active_clusters=None, dynamic_tile_m=tile_m, dynamic_route_mode="grouped",
        nvfp4_share_input=share_input)
    plan = fused_moe.plan_execution(experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=topk),
        invocation={"fast_math": fast_math}, override=config)

    def call(state):
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                        for spec in state.scratch.scratch_specs())
        output = torch.empty_like(inputs.a)
        binding = state.bind(a=inputs.a, topk_ids=inputs.topk_ids, topk_weights=inputs.topk_weights,
                             scratch=scratch, output=output, input_scales_static=True)
        return PreparedCall(run=binding.run, output=output, owners=(binding, scratch))

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare((plan.request(name="swapped-fc1", prepare_call=call),))
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                        for spec in plan.scratch_specs())
        output = torch.empty_like(inputs.a)
        session.freeze()
        for rows in (1, 8, 16, 17, capacity):
            binding = fused_moe.bind(plan, a=inputs.a[:rows], topk_ids=inputs.topk_ids[:rows],
                topk_weights=inputs.topk_weights[:rows], scratch=scratch, output=output[:rows],
                input_scales_static=True)
            binding.run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                binding.run()
            inputs.a.neg_()
            inputs.topk_ids.add_(1).remainder_(num_experts)
            if rows > 1:
                inputs.topk_ids[rows-1].fill_(-1)
            active = inputs.topk_ids[:rows] >= 0
            oracle_ids = inputs.topk_ids[:rows].clamp(min=0)
            oracle_weights = torch.where(active, inputs.topk_weights[:rows], 0.)
            reference = moe_reference_nvfp4(inputs.a[:rows], weights.w1_fp4, weights.w1_scale,
                weights.w1_alpha, weights.w2_fp4, weights.w2_scale, weights.w2_alpha,
                weights.a1_scale, weights.a2_scale, oracle_ids, oracle_weights,
                num_experts, hidden, intermediate, quant_scale_math="dynamic_fast" if fast_math else "dynamic_precise")
            addresses = tuple(t.data_ptr() for t in (*scratch, output, inputs.a, inputs.topk_ids))
            for _ in range(3):
                output.fill_(float("nan"))
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                assert addresses == tuple(t.data_ptr() for t in (*scratch, output, inputs.a, inputs.topk_ids))
                _assert_oracle(output[:rows], reference, context=f"N{intermediate}-M{tile_m}-{planner}-rows{rows}",
                               min_cos=0.9999, max_normalized_rmse=0.015)
                assert torch.isnan(output[rows:]).all()
                if rows > 1:
                    assert torch.count_nonzero(output[rows-1]) == 0
            graph.reset()
