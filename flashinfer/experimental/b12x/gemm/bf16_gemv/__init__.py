"""Prepared CuTe small-N BF16 GEMV for exact decode-sized projections.

Declare an exact source/weight specialization with ``query_from_call`` and
``plan``, prepare it through ``PreparationSession``, then pass the returned
``Plan`` to ``mm``.  Unsupported dtype, layout, alignment, K, or
M inputs fail during declaration admission; this component has no runtime
fallback or standalone precompile path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="bf16_gemv",
    group="gemm",
    api_style="prepared",
    entry_points=(
        "Plan",
        "GemvQuery",
        "GemvConfig",
        "plan",
        "query_from_call",
        "mm",
        "is_supported",
        "is_disabled",
        "SMALL_M_MAX",
        "SMALL_N_GEMV_MAX_OUT",
        "SMALL_N_GEMV_MIN_IN",
    ),
    dtypes=("bf16",),
    provenance=Provenance(
        repo="https://github.com/phaelon74/b12x",
        commit="9c78d553",
        paths=(
            "b12x/gemm/bf16_gemv.py",
            "b12x/gemm/bf16_gemv_op.py",
            "b12x/integration/vllm_plugin.py",
        ),
    ),
    test_path="tests/experimental/b12x/gemm/test_bf16_gemv.py",
    since="1.0.1",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Plan,
        SMALL_M_MAX,
        GemvConfig,
        GemvQuery,
        SMALL_N_GEMV_MAX_OUT,
        SMALL_N_GEMV_MIN_IN,
        is_disabled,
        is_supported,
        mm,
        plan,
        query_from_call,
    )

install_lazy_api(globals(), META)
