# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Contiguous (non-paged) attention for SM12x: batched and varlen forward.

BF16/FP16 Q/K/V, causal + sliding-window + attention-sink; tile shapes
auto-selected by head_dim/causality. ``run`` is the varlen (cu_seqlens)
entry, ``run_batched`` the fixed-shape batched entry; each has its own
plan/scratch/binding family (``create_plan*`` for the per-shape kernel plan,
``plan*`` for scratch sizing).

Example:
    from flashinfer.experimental.sm12x.attention import varlen

    kplan   = varlen.create_plan_batched(q, k, v, causal=True)
    splan   = varlen.plan_batched(kplan)
    spec    = splan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = splan.bind(scratch=scratch, q=q, k=k, v=v, softmax_scale=s)
    out, lse = varlen.run_batched(binding=binding)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="varlen",
    group="attention",
    api_style="planned",
    entry_points=(
        "BatchedPlan",
        "BatchedScratchPlan",
        "BatchedBinding",
        "VarlenPlan",
        "VarlenScratchPlan",
        "VarlenBinding",
        "create_plan_batched",
        "create_plan",
        "plan_batched",
        "plan",
        "run_batched",
        "run",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp16"),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/attention/contiguous/",),
    ),
    test_path="tests/experimental/attention/test_varlen.py",
    since="0.7.0",
    notes=(
        "Reduced-assurance tier: correctness-tested against a torch "
        "reference; performance not tracked; first candidate for removal."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import *  # noqa: F401,F403

install_lazy_api(globals(), META)
