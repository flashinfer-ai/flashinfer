"""Fused SM12x MLA absorbed-query projection and assembly.

The operation multiplies BF16 ``q_nope`` by either BF16 ``W_UK_T`` or its
native rowwise-MXFP8 pack and writes the result directly into a token-major MLA
query.  It appends the existing 64-wide RoPE query in the same launch.  The
caller may request BF16 output, or static per-tensor E4M3 output for an FP8
attention backend.  FP8 mode deliberately rounds the GEMM result through BF16
before scaling, preserving the established two-kernel numerical contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mla_query_projection",
    group="gemm",
    api_style="prepared",
    entry_points=(
        "plan",
        "run",
        "can_implement",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("bf16", "mxfp8"),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/b12x",
        commit="f8b62f6",
        paths=(
            "b12x/gemm/mla_query_projection/",
            "b12x/gemm/_shared/mxfp8_bmm.py",
        ),
    ),
    test_path="tests/experimental/b12x/gemm/test_mla_query_projection.py",
    since="1.0.1",
    notes="Fused BF16/MXFP8 query BMM, RoPE append, and optional E4M3 quant.",
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        can_implement,
        clear_caches,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
