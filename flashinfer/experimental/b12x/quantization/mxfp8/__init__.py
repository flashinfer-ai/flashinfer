"""Prepared BF16/FP16 row quantization into dense-GEMM MXFP8 storage.

``query_from_call`` records the source and output-scale layouts, then
``plan`` declares the fixed CuTe specialization.  A ``PreparationSession``
compiles and primes it; ``quantize_rows(..., plan=plan)`` writes the
caller-owned values and scale buffers without selection or JIT at run time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mxfp8",
    group="quantization",
    api_style="planned",
    entry_points=(
        "Mxfp8Config",
        "Mxfp8Query",
        "plan",
        "query_from_call",
        "quantize_rows",
        "is_supported",
    ),
    dtypes=("bf16", "fp16"),
    recipes=("mxfp8",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/mxfp8_quant_cute.py",),
    ),
    test_path="tests/experimental/b12x/quantization/test_mxfp8.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Mxfp8Config,
        Mxfp8Query,
        is_supported,
        plan,
        query_from_call,
        quantize_rows,
    )

install_lazy_api(globals(), META)
