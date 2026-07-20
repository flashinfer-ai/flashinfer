# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for moe.ep_moe (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._impl import (
    EPExpertMap as ExpertMap,
)
from ._impl import (
    EPMoEFP4Binding as Binding,
)
from ._impl import (
    EPMoEScratchCaps as Caps,
)
from ._impl import (
    EPMoEScratchPlan as Plan,
)
from ._impl import (
    plan_ep_moe_scratch as plan,
)
from ._impl import (
    prepare_ep_expert_map as prepare_expert_map,
)
from ._impl import (
    sm12x_ep_moe_fp4 as run,
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
    "Binding",
    "ExpertMap",
    "plan",
    "bind",
    "run",
    "prepare_expert_map",
    "is_supported",
]
