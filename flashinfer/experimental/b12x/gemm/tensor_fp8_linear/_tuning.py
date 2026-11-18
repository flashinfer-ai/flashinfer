"""Tensor-FP8 is the canonical packed blockscaled fixed dispatch, not a race.

Use FixedBlockscaledQuery(recipe="tensor_fp8", call_kind="packed", ...).
The public alias and blockscaled.mm share one prepared fixed execution contract.
"""

from b12x.gemm.blockscaled._tuning import (
    BackendConfig,
    FixedBlockscaledQuery,
    FIXED_TUNING as TUNING,
)
