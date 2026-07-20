# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Serialized block-FP8 (DeepSeek-style) linear via native MXFP8 GEMM.

Planned lifecycle: ``pack_weight`` requantizes a 128x128-block-scaled FP8
weight into the dense-GEMM MXFP8 layout (host-side, one-time); ``plan(Caps)``
sizes the caller-owned scratch; ``bind`` maps allocation-free views (CUDA
graph capture safe); ``run`` quantizes the BF16/FP16 input to MXFP8 inline
and launches the GEMM.  ``prewarm`` compiles the kernels for a capacity ahead
of serving.

Example:
    from flashinfer.experimental.sm12x.gemm import block_fp8_linear as bfl

    weight  = bfl.pack_weight(w_fp8, w_scale)                      # one-time
    plan    = bfl.plan(bfl.Caps(device="cuda", max_tokens=M,
                                in_features=K, out_features=N))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = bfl.bind(plan, scratch=scratch, source=x,
                       packed_weight=weight, output=out, expected_m=M)
    y       = bfl.run(binding=binding)     # one-shot form: bfl.run(x, weight)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="block_fp8_linear",
    group="gemm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Weight",
        "plan",
        "bind",
        "run",
        "pack_weight",
        "quantize_input",
        "prewarm",
        "is_supported",
    ),
    dtypes=("bf16", "fp16"),
    recipes=("mxfp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/block_fp8_linear.py",),
    ),
    test_path="tests/experimental/gemm/test_block_fp8_linear.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        Weight,
        bind,
        is_supported,
        pack_weight,
        plan,
        prewarm,
        quantize_input,
        run,
    )

install_lazy_api(globals(), META)
