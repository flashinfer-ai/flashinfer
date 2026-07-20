# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Fused tensor-parallel MoE for SM12x: route -> FC1 -> activation -> FC2 ->
scatter, in one launch family.

Recipes (``META.recipes``) are arguments, not separate ops: nvfp4, mxfp4,
w4a8_mx, w4a8_nvfp4, w4a16 (weight layouts packed/modelopt/nf3_2p1).  Activations:
silu, relu2, swigluoai_uninterleave.  Kernel
regimes (micro / dynamic / tiny-decode / w4a16) are selected declaratively by
``plan_execution``.

Weight lifecycle (host-side, one-time):
    ``plan_weights`` -> ``prepare_weights`` -> ``ExpertWeights``
Runtime lifecycle:
    ``plan(Caps)`` -> ``bind`` / ``bind_sparse`` / ``bind_route``
    (allocation-free views) -> ``run`` / ``run_sparse`` / ``route``
    (CUDA-graph capture safe)
``route_topk`` is a standalone one-shot top-k router; ``run_sparse`` fuses
gate -> top-k -> experts from router logits.

Example:
    from flashinfer.experimental.sm12x.moe import fused_moe

    wplan   = fused_moe.plan_weights(quant_modes="nvfp4",
                                     source_format="modelopt_nvfp4", ...)
    experts = fused_moe.prepare_weights(plan=wplan, ...)
    plan    = fused_moe.plan(fused_moe.Caps(...))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=experts,
                             topk_weights=tw, topk_ids=ti)
    out     = fused_moe.run(binding=binding)
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
        "Plan",
        "ExecutionPlan",
        "Binding",
        "SparseBinding",
        "RouteBinding",
        "ExpertWeights",
        "Routing",
        "WeightsPlan",
        "plan",
        "plan_execution",
        "plan_weights",
        "prepare_weights",
        "bind",
        "bind_sparse",
        "bind_route",
        "run",
        "run_sparse",
        "route",
        "route_topk",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16",),
    recipes=("nvfp4", "mxfp4", "w4a8_mx", "w4a8_nvfp4", "w4a16"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/integration/tp_moe.py", "b12x/moe/"),
    ),
    test_path="tests/experimental/moe/test_fused_moe.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        ExecutionPlan,
        ExpertWeights,
        Plan,
        RouteBinding,
        Routing,
        SparseBinding,
        WeightsPlan,
        bind,
        bind_route,
        bind_sparse,
        clear_caches,
        is_supported,
        plan,
        plan_execution,
        plan_weights,
        prepare_weights,
        route,
        route_topk,
        run,
        run_sparse,
    )

install_lazy_api(globals(), META)
