"""Fused tensor-parallel MoE for SM12x.

The canonical API keeps checkpoint encoding, prepared representation, and
activation precision independent:

    import torch
    from b12x.moe import fused_moe

    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
        w13_layout=fused_moe.W13Layout.W31,
    )
    activation = fused_moe.ActivationSpec(
        mode=fused_moe.ActivationMode.A8,
        nonlinearity="silu",
        io_dtype=torch.bfloat16,
    )
    geometry = fused_moe.MoEGeometry(
        num_experts=128,
        hidden_size=4096,
        intermediate_size=14336,
    )
    weight_plan = fused_moe.plan_weights(
        source=source,
        activation=activation,
        geometry=geometry,
    )
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=weights)
    declaration = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=4096, top_k=8),
    )
    request = declaration.request(
        name="layer.moe",
        prepare_calls={count: make_call for count in declaration.token_counts},
    )
    result = session.prepare((request,))
    binding = fused_moe.bind(
        declaration,
        scratch=scratch,
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    output = fused_moe.run(binding=binding)

Primary capacity preparation is declarative. Runtime binding requires the
session-prepared ``Plan``; route and FC2 work are materialized as part of
that prepared plan rather than public alternate launch paths.

Tensor-based callers may also use ``plan_weights(quant_modes=...)``,
``prepare_weights`` with tensor arguments, and ``plan(Caps(...))``. These
scratch plans use the validated heuristic configuration without autotuning;
warm their bindings before CUDA graph capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="fused_moe",
    group="moe",
    api_style="planned",
    entry_points=(
        "Caps",
        "plan",
        "TrellisExtent",
        "TrellisSource",
        "TrellisStaging",
        "ActivationMode",
        "ActivationSpec",
        "Binding",
        "RouteBinding",
        "SparseBinding",
        "RouteTopKInvocation",
        "FC2Invocation",
        "ExecutionCapacity",
        "MoEGeometry",
        "MoeDecodeConfig",
        "MoeDecodeQuery",
        "PackedSource",
        "PackedSourceFormat",
        "PackedWeights",
        "X4TWeights",
        "IQ2XSWeights",
        "BlockQuantWeights",
        "PreparedExperts",
        "PreparedWeightFormat",
        "RoutingSpec",
        "ScaleEncoding",
        "ScaleFactors",
        "TrellisConfig",
        "TrellisWeights",
        "W13Layout",
        "WeightEncoding",
        "WeightPacking",
        "WeightPlan",
        "WeightPlanConstraints",
        "WeightSource",
        "bind",
        "bind_route",
        "bind_sparse",
        "clear_caches",
        "is_supported",
        "plan_execution",
        "plan_route_topk",
        "plan_fc2",
        "plan_weights",
        "prepare_weights",
        "route_topk",
        "route",
        "run_fc2",
        "run_sparse",
        "run",
    ),
    dtypes=("bf16", "fp16"),
    recipes=(
        "nvfp4",
        "w4a8_mx",
        "w4a8_nvfp4",
        "w6a8_mx",
        "w4a16",
        "b12x_trellis",
    ),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/integration/tp_moe.py", "b12x/moe/"),
    ),
    test_path="tests/experimental/b12x/moe/test_fused_moe.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Caps,
        plan,
        TrellisExtent,
        TrellisSource,
        TrellisStaging,
        ActivationMode,
        ActivationSpec,
        Binding,
        RouteBinding,
        SparseBinding,
        RouteTopKInvocation,
        FC2Invocation,
        ExecutionCapacity,
        MoEGeometry,
        MoeDecodeConfig,
        MoeDecodeQuery,
        PackedSource,
        PackedSourceFormat,
        PackedWeights,
        X4TWeights,
        IQ2XSWeights,
        BlockQuantWeights,
        PreparedExperts,
        PreparedWeightFormat,
        RoutingSpec,
        ScaleEncoding,
        ScaleFactors,
        TrellisConfig,
        TrellisWeights,
        W13Layout,
        WeightEncoding,
        WeightPacking,
        WeightPlan,
        WeightPlanConstraints,
        WeightSource,
        bind,
        bind_route,
        bind_sparse,
        clear_caches,
        is_supported,
        plan_execution,
        plan_route_topk,
        plan_fc2,
        plan_weights,
        prepare_weights,
        route_topk,
        route,
        run_fc2,
        run_sparse,
        run,
    )

install_lazy_api(globals(), META)
