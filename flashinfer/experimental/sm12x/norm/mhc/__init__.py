# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""mHC residual for SM12x: fused RMSNorm + hyper-connection mixing +
projection (DeepSeek-style), BF16 with TF32 projection paths.

Three phases share one plan/binding; the phase is the verb because the
signatures differ: ``run_pre`` (residual -> mixed input + carry),
``run_post`` (output mix-back), ``run_post_pre`` (fused post+pre between
layers). Sinkhorn-normalized mix matrices; hidden sizes 4096/7168; mix
constants exposed as ``MIXES`` / ``MULT`` / ``PARTIALS``.

Planned lifecycle: ``plan(Caps(...))`` -> ``bind`` (views only) ->
``run_*`` (capture safe; torch.compile-safe via opaque custom ops).

Example:
    from flashinfer.experimental.sm12x.norm import mhc

    plan    = mhc.plan(mhc.Caps(...))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = mhc.bind(plan, scratch=scratch, ...)
    residual, post, comb, y = mhc.run_post_pre(..., binding=binding)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mhc",
    group="norm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "plan",
        "bind",
        "run_pre",
        "run_post",
        "run_post_pre",
        "MIXES",
        "MULT",
        "PARTIALS",
        "DEFAULT_SPLIT_K",
        "DEFAULT_BLOCK_K",
        "DEFAULT_BLOCK_H",
        "is_supported",
    ),
    dtypes=("bf16",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=(
            "b12x/integration/residual.py",
            "b12x/integration/residual_kernels.py",
        ),
    ),
    test_path="tests/experimental/norm/test_mhc.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        DEFAULT_BLOCK_H,
        DEFAULT_BLOCK_K,
        DEFAULT_SPLIT_K,
        MIXES,
        MULT,
        PARTIALS,
        Binding,
        Caps,
        Plan,
        bind,
        is_supported,
        plan,
        run_post,
        run_post_pre,
        run_pre,
    )

install_lazy_api(globals(), META)
