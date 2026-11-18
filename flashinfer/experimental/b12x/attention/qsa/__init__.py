"""Prepared grouped-selector sparse GQA for Qwen3.8-Flash-Next."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="qsa",
    group="attention",
    api_style="prepared",
    entry_points=(
        "CacheRequirements",
        "Caps",
        "DraftSelectionPlan",
        "DraftSelectionReuse",
        "DraftSelectionState",
        "Plan",
        "Binding",
        "LocalSelection",
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
    ),
    dtypes=("bf16",),
    recipes=("grouped_selector_sparse_gqa",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="3a437ab5168060e4d625f05e1625c04089f1ba37",
        paths=(
            "b12x/attention/dsa_indexer/",
            "b12x/attention/paged/",
            "b12x/attention/sparse_mla/",
        ),
    ),
    test_path="tests/experimental/b12x/attention/test_qsa_contract.py",
    since="1.3.0",
    notes=(
        "QSA declarations require PreparationSession materialization before "
        "binding; all cache and selector mutation remains explicit.",
    ),
)

if TYPE_CHECKING:
    from .api import (
        Binding,
        CacheRequirements,
        Caps,
        DraftSelectionPlan,
        DraftSelectionReuse,
        DraftSelectionState,
        Plan,
        QsaConfig,
        QsaQuery,
        bind,
        cache_requirements,
        draft_selection_plan,
        invocation_from_descriptors,
        invocation_from_tensors,
        LocalSelection,
        attend,
        attend_reuse,
        is_supported,
        plan,
        run,
        select,
    )

install_lazy_api(globals(), META)
