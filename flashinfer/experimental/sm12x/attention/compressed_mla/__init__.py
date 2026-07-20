# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Compressed MLA decode for SM12x (DeepSeek-V3.2).

Decode directly from compressed KV pages: a sliding-window cache plus an
indexed (top-k-selected) cache, fused-merged into the caller's output.
Head dim is fixed to DSV4's 512 + 64. Single-pass decode on SM121.

Planned lifecycle: ``plan(Caps(...))`` -> ``bind`` (views only) -> ``run``
(capture safe). ``split_chunks_for_contract`` exposes the fixed
split-planning contract integrations preplan against.

Example:
    from flashinfer.experimental.sm12x.attention import compressed_mla

    plan    = compressed_mla.plan(compressed_mla.Caps(...))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = compressed_mla.bind(plan, scratch=scratch, q=q,
                                  swa_indices=idx, swa_lengths=lens, ...)
    out = compressed_mla.run(swa_k_cache=swa, binding=binding,
                             sm_scale=scale, ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="compressed_mla",
    group="attention",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Scratch",
        "plan",
        "bind",
        "run",
        "split_chunks_for_contract",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("dsv4",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=(
            "b12x/attention/mla/compressed_api.py",
            "b12x/integration/compressed_scratch.py",
        ),
    ),
    test_path="tests/experimental/attention/test_compressed_mla.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        Scratch,
        bind,
        clear_caches,
        is_supported,
        plan,
        run,
        split_chunks_for_contract,
    )

install_lazy_api(globals(), META)
