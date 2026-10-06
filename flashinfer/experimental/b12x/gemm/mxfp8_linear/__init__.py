"""Compatibility alias for ModelOpt MXFP8 ``blockscaled`` calls.

This mathematical alias uses the same ``query_from_call`` / ``plan`` /
``PreparationSession`` lifecycle as ``blockscaled``. BF16 precision queries
and fixed FP16/prequantized queries retain distinct contracts. Omitting
``plan`` selects a cached heuristic plan without autotuning; warm the required
row capacities before CUDA graph capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mxfp8_linear",
    group="gemm",
    api_style="planned",
    entry_points=("Weight", "BlockscaledQuery", "FixedBlockscaledQuery", "plan", "query_from_call", "mm", "pack_weight", "is_supported"),
    dtypes=("bf16", "fp16"),
    recipes=("mxfp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/mxfp8_linear.py",),
    ),
    test_path="tests/experimental/b12x/gemm/test_mxfp8_linear.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import BlockscaledQuery, FixedBlockscaledQuery, Weight, is_supported, mm, pack_weight, plan, query_from_call  # noqa: F401

install_lazy_api(globals(), META)
