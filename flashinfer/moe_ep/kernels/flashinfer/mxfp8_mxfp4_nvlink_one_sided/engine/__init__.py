"""Blackwell GenPhase entry point; independently owned SM100/103 source fork."""

from .kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel_gen_specialized import (
    BlockScaledSwapAbGenphaseMoeKernel as BlackwellInferenceGenphaseMegaMoE,
)

__all__ = ["BlackwellInferenceGenphaseMegaMoE"]
