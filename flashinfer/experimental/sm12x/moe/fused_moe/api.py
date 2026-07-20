# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for moe.fused_moe (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from .._shared.execution import (
    MoEWeightPreparationPlan as WeightsPlan,
)
from .._shared.routing import (
    route_topk,
)
from ._impl import (
    SM12XFP4ExpertWeights as ExpertWeights,
)
from ._impl import (
    SM12XTopKRouting as Routing,
)
from ._impl import (
    TPMoEFP4Binding as Binding,
)
from ._impl import (
    TPMoEPlan as ExecutionPlan,
)
from ._impl import (
    TPMoERouteBinding as RouteBinding,
)
from ._impl import (
    TPMoEScratchCaps as Caps,
)
from ._impl import (
    TPMoEScratchPlan as Plan,
)
from ._impl import (
    TPMoESparseFP4Binding as SparseBinding,
)
from ._impl import (
    build_tp_moe_route_binding as bind_route,
)
from ._impl import (
    build_tp_moe_sparse_fp4_binding as bind_sparse,
)
from ._impl import (
    clear_tp_moe_caches as clear_caches,
)
from ._impl import (
    plan_sm12x_fp4_moe_weights as plan_weights,
)
from ._impl import (
    plan_tp_moe_execution as plan_execution,
)
from ._impl import (
    plan_tp_moe_scratch as plan,
)
from ._impl import (
    prepare_sm12x_fp4_moe_weights as prepare_weights,
)
from ._impl import (
    sm12x_moe_fp4 as run,
)
from ._impl import (
    sm12x_route_experts_fast as route,
)
from ._impl import (
    sm12x_sparse_moe_fp4 as run_sparse,
)
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
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
]
