"""Prepared BF16 gate projection and mean over normalized residual streams.

For source ``[T, R]``, weights ``[S*H, R]``, and normalized residuals
``[T, S*H]``, return ``[T, H]``. Projection, sigmoid, and multiplication
each round to BF16 before FP32 stream accumulation. Preparation compares a
fused Triton implementation with existing BF16 projections followed by gate
reduction. Outputs and the separate path's logit workspace are caller-owned.

The fused implementation supports SM120/SM121, 1/2/4/8 streams, and projection
ranks divisible by 16. Models must share these BF16 rounding boundaries.
"""

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="hyperconnection_gate",
    group="gemm",
    api_style="planned",
    entry_points=("Query", "Config", "plan", "run", "reference", "is_supported"),
    dtypes=("bf16",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/flashinfer",
        commit="ed8005aa",
        paths=(
            "flashinfer/experimental/b12x/norm/hyperconnection/_kernels.py",
            "flashinfer/experimental/b12x/gemm/bf16_gemv/",
        ),
    ),
    test_path="tests/experimental/b12x/gemm/test_hyperconnection_gate.py",
    since="1.3.0",
)

if TYPE_CHECKING:
    from .api import Config, Query, is_supported, plan, reference, run

install_lazy_api(globals(), META)
