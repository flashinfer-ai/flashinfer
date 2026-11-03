"""Exported MegaMoE KernelClass entry points."""

from .kernel_src.rubin.inference.mega.block_scaled_swap_ab_mega_moe_kernel_gen_specialized import (
    BlockScaledSwapAbGenphaseMoeKernel as RubinInferenceGenphaseMegaMoE,
)
from .kernel_src.rubin.inference.mega.block_scaled_swap_ab_mega_moe_kernel import (
    BlockScaledSwapAbMegaMoeKernel as RubinInferenceMegaMoE,
)

__all__ = [
    "RubinInferenceGenphaseMegaMoE",
    "RubinInferenceMegaMoE",
]
