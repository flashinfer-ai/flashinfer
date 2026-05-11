"""
Specialized GEMM kernels used by explicit runtime routing hooks.
"""

from .mm_fp4_sm121 import (
    is_mm_fp4_sm121_specialized_problem,
    run_mm_fp4_sm121_specialized,
)
from .bmm_fp8_sm121 import (
    gen_bmm_fp8_sm121_specialized_cuda_module,
    is_bmm_fp8_sm121_specialized_problem,
    run_bmm_fp8_sm121_specialized,
)

__all__ = [
    "is_mm_fp4_sm121_specialized_problem",
    "run_mm_fp4_sm121_specialized",
    "gen_bmm_fp8_sm121_specialized_cuda_module",
    "is_bmm_fp8_sm121_specialized_problem",
    "run_bmm_fp8_sm121_specialized",
]
