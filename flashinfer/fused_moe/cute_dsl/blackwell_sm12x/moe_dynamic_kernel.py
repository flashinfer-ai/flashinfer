"""Unified constructor for SM120/SM121 dynamic NVFP4 MoE kernels."""

from __future__ import annotations

from typing import Tuple

from ._moe_dynamic.generic import (
    _TASK_SLICE_CHUNK,
    MoEDynamicKernel as _GenericMoEDynamicKernel,
)
from ._moe_dynamic.gated import MoEGatedDynamicKernel
from .moe_activation import is_gated_activation


class MoEDynamicKernel:
    """Construct the compatible dynamic kernel for the requested activation.

    The branch-paired gated implementation currently targets the logical
    M128xN128 dynamic tile. Smaller dynamic M tiles and non-gated activations
    use the current generic implementation from upstream.
    """

    def __new__(
        cls,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        activation: str = "silu",
        swiglu_alpha: float = 1.702,
        swiglu_beta: float = 1.0,
        swiglu_limit: float | None = None,
        share_input_across_experts: bool = False,
    ):
        use_gated_optimized = is_gated_activation(activation) and mma_tiler_mn == (
            128,
            128,
        )
        implementation = (
            MoEGatedDynamicKernel if use_gated_optimized else _GenericMoEDynamicKernel
        )
        return implementation(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            share_input_across_experts=share_input_across_experts,
        )


__all__ = ["MoEDynamicKernel", "_TASK_SLICE_CHUNK"]
