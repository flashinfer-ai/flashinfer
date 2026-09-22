"""Exported MegaMoE KernelClass entry points."""

from .kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel import (
    BlockScaledSwapAbMegaMoeKernel as BlackwellInferenceMegaMoE,
)

__all__ = [
    "BlackwellInferenceMegaMoE",
]
