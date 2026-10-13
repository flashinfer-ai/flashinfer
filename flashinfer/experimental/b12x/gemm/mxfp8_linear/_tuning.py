"""Compatibility alias: reuse blockscaled choices and compilation ownership.

BF16 packed calls use TUNING. FP16 and prequantized packed calls use
FIXED_TUNING; they expose no raw dense-engine override.
"""

from b12x.gemm.blockscaled._tuning import (
    BlockscaledConfig,
    BlockscaledQuery,
    FixedBlockscaledQuery,
    TUNING,
    FIXED_TUNING,
)
