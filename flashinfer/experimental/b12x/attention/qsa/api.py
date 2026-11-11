"""Public planned API for :mod:`b12x.attention.qsa`."""

from __future__ import annotations

from ._contract import (
    Binding,
    CacheRequirements,
    Caps,
    DraftSelectionPlan,
    DraftSelectionReuse,
    DraftSelectionState,
    LocalSelection,
    attend,
    attend_reuse,
    bind,
    cache_requirements,
    draft_selection_plan,
    invocation_from_descriptors,
    invocation_from_tensors,
    is_supported,
    plan,
    run,
    select,
)
from ._tuning import QsaConfig, QsaQuery
from b12x.preparation import Plan

__all__ = [
    "CacheRequirements",
    "Caps",
    "DraftSelectionPlan",
    "DraftSelectionReuse",
    "DraftSelectionState",
    "LocalSelection",
    "Plan",
    "Binding",
    "QsaConfig",
    "QsaQuery",
    "cache_requirements",
    "draft_selection_plan",
    "invocation_from_descriptors",
    "invocation_from_tensors",
    "plan",
    "bind",
    "run",
    "select",
    "attend",
    "attend_reuse",
    "is_supported",
]
