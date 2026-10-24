"""SwiGLU clamp placement is fixed before compiling each MoE schedule."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe._shared.kernels.reference import (
    moe_reference_nvfp4,
    moe_reference_w4a16_f32,
)
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x
from .test_cute_migration_moe_standard_corpus import (
    _assert_oracle,
    _make_inputs,
    _make_nvfp4_weights,
)


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize(
    "backend,rows,materialized,route_mode",
    [
        ("micro", 1, False, None),
        ("micro", 4, False, None),
        ("dynamic", 33, False, None),
        ("dynamic", 128, True, None),
        ("w4a16", 1, False, "direct"),
        ("w4a16", 4, False, "direct"),
        ("w4a16", 4, False, "packed"),
        ("w4a16", 128, False, "packed"),
        ("w4a16-native", 1, False, "direct"),
        ("w4a16-native", 4, False, "direct"),
        ("w4a16-native", 4, False, "packed"),
    ],
)
def test_prepared_clamp_order(backend, rows, materialized, route_mode, after):
    device = require_b12x()
    experts_count, hidden, intermediate, topk = 4, 512, 256, 2
    weights = _make_nvfp4_weights(
        device,
        seed=819,
        num_experts=experts_count,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    weights = replace(
        weights,
        a1_scale=torch.full((experts_count,), 64.0, device=device),
        a2_scale=torch.full((experts_count,), 512.0, device=device),
        w1_alpha=weights.w1_alpha / 64,
        w2_alpha=weights.w2_alpha / 512,
    )
    inputs = _make_inputs(
        device,
        m=rows,
        seed=820,
        route_shift=0,
        num_experts=experts_count,
        hidden_size=hidden,
        topk=topk,
    )
    limit = 0.5
    a16 = backend.startswith("w4a16")
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a16" if backend == "w4a16" else "a4",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=limit,
            swiglu_clamp_after_silu=after,
            a16_max_tokens=rows if backend == "w4a16-native" else 0,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=experts_count,
            hidden_size=hidden,
            intermediate_size=intermediate,
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=weights.w1_fp4.clone(),
            w2=weights.w2_fp4.clone(),
            w13_block_scales=weights.w1_scale,
            w2_block_scales=weights.w2_scale,
            w13_global_scales=weights.w1_alpha * weights.a1_scale,
            w2_global_scales=weights.w2_alpha * weights.a2_scale,
            input_scale=None if backend == "w4a16" else weights.a1_scale,
            intermediate_scale=None if backend == "w4a16" else weights.a2_scale,
            immutable_input_scales=True,
        ),
    )
    config = fused_moe.MoeDecodeConfig(
        backend="w4a16" if a16 else backend,
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=128 if materialized else 16 if backend == "dynamic" else None,
        dynamic_route_mode="grouped" if backend == "dynamic" else None,
        w4a16_route_mode=route_mode,
        nvfp4_materialize_intermediate=materialized,
        nvfp4_share_input=materialized,
    )
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=rows, top_k=topk),
        routing=fused_moe.RoutingSpec(deterministic_output=False),
        invocation={"fast_math": False},
        override=config,
    )

    def call(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        output = torch.empty_like(inputs.a)
        binding = state.bind(
            a=inputs.a,
            topk_ids=inputs.topk_ids,
            topk_weights=inputs.topk_weights,
            scratch=scratch,
            output=output,
            input_scales_static=True,
        )
        return PreparedCall(run=binding.run, output=output, owners=(binding, scratch))

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare((plan.request(name="clamped-swiglu", prepare_call=call),))
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        output = torch.empty_like(inputs.a)
        binding = fused_moe.bind(
            plan,
            a=inputs.a,
            topk_ids=inputs.topk_ids,
            topk_weights=inputs.topk_weights,
            scratch=scratch,
            output=output,
            input_scales_static=True,
        )
        binding.run()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            binding.run()
        inputs.a.neg_()
        inputs.topk_ids.add_(1).remainder_(experts_count)
        reference_args = (
            inputs.a,
            weights.w1_fp4,
            weights.w1_scale,
            weights.w1_alpha,
            weights.w2_fp4,
            weights.w2_scale,
            weights.w2_alpha,
            weights.a1_scale,
            weights.a2_scale,
            inputs.topk_ids,
            inputs.topk_weights,
            experts_count,
            hidden,
            intermediate,
        )
        oracle = moe_reference_nvfp4
        oracle_options = dict(
            quant_scale_math="micro" if backend == "micro" else "dynamic_precise"
        )
        if a16:
            oracle = moe_reference_w4a16_f32
            oracle_options = {}
            reference_args = (
                inputs.a,
                weights.w1_fp4,
                weights.w1_scale,
                weights.w1_alpha * weights.a1_scale,
                weights.w2_fp4,
                weights.w2_scale,
                weights.w2_alpha * weights.a2_scale,
                inputs.topk_ids,
                inputs.topk_weights,
                experts_count,
                hidden,
                intermediate,
            )
        reference = oracle(
            *reference_args,
            swiglu_limit=limit,
            swiglu_clamp_after_silu=after,
            **oracle_options,
        )
        opposite = oracle(
            *reference_args,
            swiglu_limit=limit,
            swiglu_clamp_after_silu=not after,
            **oracle_options,
        )
        assert (
            reference.float() - opposite.float()
        ).norm() / reference.float().norm() > 0.1
        for _ in range(3):
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            _assert_oracle(
                output,
                reference,
                context=f"{backend}-{rows}-after{after}",
                min_cos=0.9999,
                max_normalized_rmse=0.015,
            )
