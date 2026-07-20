# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel MoE for SM12x (W4A16 experts).

Each rank sees the replicated input and computes partial outputs for its
local experts; cross-rank reduction is the caller's job (typically
``comm.pcie.OneshotAllReduce``) — composed in user code, never imported here.

Lifecycle: ``prepare_expert_map`` (host-side, one-time) -> ``plan(Caps)`` ->
``bind`` (views only) -> ``run`` (capture safe).

Example:
    from flashinfer.experimental.sm12x.moe import ep_moe

    emap    = ep_moe.prepare_expert_map(expert_map,        # int32 [global_E]
                                        local_num_experts=local_E)
    plan    = ep_moe.plan(ep_moe.Caps(...))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = ep_moe.bind(plan, scratch=scratch, ...)
    partial = ep_moe.run(binding=binding)   # then all-reduce across ranks
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
        "plan",
        "bind",
        "run",
        "prepare_expert_map",
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
    test_path="tests/experimental/moe/test_ep_moe.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        ExpertMap,
        Plan,
        bind,
        is_supported,
        plan,
        prepare_expert_map,
        run,
    )

install_lazy_api(globals(), META)
