"""Expert-parallel MoE for SM12x (W4A16 experts).

Each rank sees the replicated input and computes partial outputs for its
local experts; cross-rank reduction is the caller's job (typically
``comm.pcie.OneshotAllReduce``) — composed in user code, never imported here.

Lifecycle: ``prepare_expert_map`` -> declarative ``plan`` ->
``PreparationSession.prepare`` -> ``bind(plan, ...)`` -> ``run``.

The rank-local result remains a partial: callers compose it with their actual
EP reduction group; this package never creates or approximates a collective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="ep_moe",
    group="moe",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "ExpertMap",
        "EpMoeConfig",
        "EpMoeQuery",
        "plan",
        "bind",
        "run",
        "prepare_expert_map",
        "invocation_from_tensors",
        "is_supported",
    ),
    dtypes=("bf16",),
    recipes=("w4a16",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/integration/ep_moe.py",),
    ),
    test_path="tests/experimental/b12x/moe/test_ep_moe.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        EpMoeConfig,
        EpMoeQuery,
        ExpertMap,
        Plan,
        bind,
        is_supported,
        plan,
        prepare_expert_map,
        invocation_from_tensors,
        run,
    )

install_lazy_api(globals(), META)
