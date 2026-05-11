"""SM121 specialized mm_fp4 kernel routing."""

from .mm_fp4_sm121 import (
    is_mm_fp4_sm121_specialized_problem,
    run_mm_fp4_sm121_specialized,
)

__all__ = [
    "is_mm_fp4_sm121_specialized_problem",
    "run_mm_fp4_sm121_specialized",
]
