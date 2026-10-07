"""Unified constructor for SM120/SM121 dynamic NVFP4 MoE kernels."""

from __future__ import annotations

from typing import Tuple

from ._moe_dynamic.generic import (
    _TASK_SLICE_CHUNK,
    MoEDynamicKernel as _GenericMoEDynamicKernel,
)
from ._moe_dynamic.gated import MoEGatedDynamicKernel
from .moe_activation import is_gated_activation


_GATED_OPTIMIZED_TILE_MN = (128, 128)
_GATED_OPTIMIZED_SF_VEC_SIZE = 16
_GATED_OPTIMIZED_MAX_HIDDEN_SIZE = 128 * 128
_GATED_OPTIMIZED_RETAINED_SLICES = 4
_GATED_OPTIMIZED_MAX_INTERMEDIATE_SIZE = _GATED_OPTIMIZED_RETAINED_SLICES * 128
_GATED_OPTIMIZED_MAX_TOPK = 16
_MAX_SHARED_INPUT_TOPK = 32


def _can_use_gated_optimized_kernel(
    *,
    activation: str,
    sf_vec_size: int,
    mma_tiler_mn: Tuple[int, int],
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    num_topk: int | None = None,
    share_input_across_experts: bool = False,
) -> bool:
    """Return whether the branch-paired gated kernel supports this shape."""
    if not is_gated_activation(activation):
        return False
    if sf_vec_size != _GATED_OPTIMIZED_SF_VEC_SIZE:
        return False
    if mma_tiler_mn != _GATED_OPTIMIZED_TILE_MN:
        return False
    if hidden_size is not None and hidden_size > _GATED_OPTIMIZED_MAX_HIDDEN_SIZE:
        return False
    if (
        intermediate_size is not None
        and intermediate_size > _GATED_OPTIMIZED_MAX_INTERMEDIATE_SIZE
    ):
        return False
    max_topk = (
        _MAX_SHARED_INPUT_TOPK
        if share_input_across_experts
        else _GATED_OPTIMIZED_MAX_TOPK
    )
    if num_topk is not None and num_topk > max_topk:
        return False
    return True


class MoEDynamicKernel:
    """Construct the compatible dynamic kernel for the requested activation.

    The branch-paired gated implementation targets its validated M128xN128
    Q0/Q1 staging geometry. Unsupported shapes and non-gated activations use
    the generic implementation.
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
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        num_topk: int | None = None,
    ):
        # The shared-input producer reserves 32 route slots per token in both
        # implementations. Fall back to the per-route producer above that.
        share_input_across_experts = bool(
            share_input_across_experts
            and (num_topk is None or num_topk <= _MAX_SHARED_INPUT_TOPK)
        )
        use_gated_optimized = _can_use_gated_optimized_kernel(
            activation=activation,
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_topk=num_topk,
            share_input_across_experts=share_input_across_experts,
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
