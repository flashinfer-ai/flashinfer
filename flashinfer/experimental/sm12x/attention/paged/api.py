# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for attention.paged (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._forward import (
    clear_paged_caches as clear_caches,
)
from ._forward import (
    paged_attention_forward as run,
)
from ._scratch import (
    SM12XPagedAttentionBinding as Binding,
)
from ._scratch import (
    SM12XPagedAttentionScratchCaps as Caps,
)
from ._scratch import (
    SM12XPagedAttentionScratchPlan as Plan,
)
from ._scratch import (
    plan_paged_attention_scratch as plan,
)
from .planner import (
    PagedPlanBudget as Budget,
)
from .planner import (
    infer_paged_mode as infer_mode,
)
from .workspace import (
    PagedAttentionWorkspace as Workspace,
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
    "Workspace",
    "Budget",
    "plan",
    "bind",
    "run",
    "infer_mode",
    "is_supported",
    "clear_caches",
]
