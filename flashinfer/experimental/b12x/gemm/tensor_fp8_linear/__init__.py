"""Compatibility alias for tensor-scaled FP8 ``blockscaled`` calls.

This mathematical alias uses the same ``query_from_call`` / ``plan`` /
``PreparationSession`` lifecycle as ``blockscaled``. Omitting ``plan`` selects
a cached heuristic plan without autotuning. ``prewarm`` primes serving
capacities before CUDA graph capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="tensor_fp8_linear",
    group="gemm",
    api_style="planned",
    entry_points=(
        "Weight",
        "FixedBlockscaledQuery",
        "plan",
        "query_from_call",
        "mm",
        "pack_weight",
        "prewarm",
        "is_supported",
    ),
    dtypes=("fp8_e4m3", "bf16", "fp16"),
    recipes=("tensor_fp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="1bc4f82",
        paths=("b12x/gemm/tensor_fp8_linear",),
    ),
    test_path="tests/experimental/b12x/gemm/test_tensor_fp8_linear.py",
    since="1.0.1",
)

if TYPE_CHECKING:
    from .api import (
        FixedBlockscaledQuery,
        Weight,
        is_supported,
        mm,
        pack_weight,
        plan,
        prewarm,
        query_from_call,
    )  # noqa: F401

install_lazy_api(globals(), META)
