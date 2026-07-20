# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Dense block-scaled GEMM for SM12x: ``C = (A·SFA) @ (B·SFB)``.

One-shot functional op over the shared SM120 warp-MMA engine (no TMEM, no
tcgen05, no 2-CTA).  Recipes: NVFP4 (Float4E2M1 values, e4m3 scales, vec 16),
MXFP4 (e8m0 scales, vec 32), MXFP8 (e4m3 values, e8m0 scales, vec 32).
``expected_m`` is a DeepGEMM-style regime hint (decode vs prefill tiles).
Fused variants quantize the A operand inline (optionally grouped).

Example:
    from flashinfer.experimental.sm12x.gemm import blockscaled

    out = blockscaled.mm(
        (a_fp4, a_sf), (b_fp4, b_sf),
        ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn", sf_vec_size=16,
        c_dtype="bfloat16", alpha=alpha, expected_m=m,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="blockscaled",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "mm",
        "mm_fused_quant_a",
        "mm_fused_quant_a_grouped",
        "is_supported",
    ),
    dtypes=("bf16", "fp16", "fp32", "fp8_e4m3", "fp4_e2m1"),
    recipes=("nvfp4", "mxfp4", "mxfp8"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/dense.py",),
    ),
    test_path="tests/experimental/gemm/test_blockscaled.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        is_supported,
        mm,
        mm_fused_quant_a,
        mm_fused_quant_a_grouped,
    )

install_lazy_api(globals(), META)
