# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Fused MLA WO-A / WO-B projections as native MXFP8 GEMMs.

Grouped ``[tokens, groups, group_width]`` input, quantized to MXFP8 inline;
the ``run_inv_rope`` variant fuses inverse-RoPE into the input quantization.
Planned lifecycle: ``pack_weights`` (host-side, one-time) -> ``plan(Caps)``
-> ``bind``/``bind_inv_rope`` (views only) -> ``run``/``run_inv_rope``
(capture-safe).  Standalone quantizers (``quantize_input*``) expose the
input-quant stage for split pipelines.

Example:
    from flashinfer.experimental.sm12x.gemm import wo_projection as wo

    weights = wo.pack_weights(wa_fp8, wa_scale, wb_fp8, wb_scale,   # one-time
                              groups=G, group_width=D, rank=R, hidden=H)
    plan    = wo.plan(wo.Caps(device="cuda", max_tokens=M, groups=G,
                              group_width=D, rank=R, hidden=H))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = wo.bind(plan, scratch=scratch, source_tgd=x_tgd,
                      weights=weights, expected_m=M)
    out     = wo.run(binding=binding)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="wo_projection",
    group="gemm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "InvRopeBinding",
        "Weights",
        "MXFP8Rows",
        "plan",
        "bind",
        "bind_inv_rope",
        "run",
        "run_inv_rope",
        "pack_weights",
        "quantize_input",
        "quantize_input_inv_rope",
        "quantize_input_b",
        "is_supported",
    ),
    dtypes=("bf16",),
    recipes=("mxfp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/wo_projection.py", "b12x/gemm/wo_quant_cute.py"),
    ),
    test_path="tests/experimental/gemm/test_wo_projection.py",
    since="0.7.0",
    notes=(
        "The planned run path is BF16-only; the standalone quantize_input* "
        "facet also compiles for fp16."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        InvRopeBinding,
        MXFP8Rows,
        Plan,
        Weights,
        bind,
        bind_inv_rope,
        is_supported,
        pack_weights,
        plan,
        quantize_input,
        quantize_input_b,
        quantize_input_inv_rope,
        run,
        run_inv_rope,
    )

install_lazy_api(globals(), META)
