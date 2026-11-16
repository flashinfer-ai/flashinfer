"""Trellis-coded dense linear for SM12x.

The operation consumes the checkpoint-native ``trellis_t256`` payload
(``mcg`` or ``lut_e4m3`` codebooks) and
its two Hadamard sign vectors.  Preparation validates and records zero-copy
views; execution performs input rotation, a W4A16 or direct E4M3-W4A8 GEMM,
and output rotation.
Callers that need CUDA-graph capture must provide stable ``output``,
``gemm_output``, and ``c_tmp`` tensors to :func:`run`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="trellis_linear",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "PreparedWeight",
        "TrellisQuery",
        "prepare_weight",
        "prepare_pair_weight",
        "query_from_weight",
        "plan",
        "run",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp16"),
    recipes=(
        "w4a16/trellis_mcg",
        "w4a16/trellis_lut_e4m3",
        "w4a8/trellis_lut_e4m3",
    ),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/b12x",
        commit="c58a381",
        paths=(
            "b12x/gemm/trellis_linear/",
            "b12x/moe/_shared/kernels/w4a16/",
        ),
    ),
    test_path="tests/experimental/b12x/gemm/test_trellis_linear.py",
    since="1.0.1",
    notes="Trellis-coded dense W4A16 and direct E4M3-W4A8 linear.",
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        PreparedWeight,
        TrellisQuery,
        clear_caches,
        is_supported,
        plan,
        prepare_weight,
        prepare_pair_weight,
        query_from_weight,
        run,
    )

install_lazy_api(globals(), META)
