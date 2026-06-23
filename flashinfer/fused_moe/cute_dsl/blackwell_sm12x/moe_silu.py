"""SwiGLU wrappers for the activation-specialized fused MoE backends."""

from __future__ import annotations

from typing import Tuple

from .moe_dynamic_kernel import MoEDynamicKernelBackend
from .moe_micro_kernel import MoEMicroKernelBackend


class MoEMicroKernelSilu(MoEMicroKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        fast_math: bool = False,
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        single_token: bool = False,
        dynamic_down_scale: bool = False,
        a8_mx_mode: bool = False,
        scale_format: str = "e4m3_k16",
        e8m0_scale_layout: str = "packed",
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            fast_math=fast_math,
            activation="silu",
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=single_token,
            dynamic_down_scale=dynamic_down_scale,
            a8_mx_mode=a8_mx_mode,
            scale_format=scale_format,
            e8m0_scale_layout=e8m0_scale_layout,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )

    @classmethod
    def is_supported(
        cls,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
    ) -> bool:
        return super().is_supported(m, k, n, num_topk, weight_E)


class MoEDynamicKernelSilu(MoEDynamicKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        fast_math: bool = False,
        dynamic_down_scale: bool = False,
        share_input_across_experts: bool = False,
        deterministic_output: bool = False,
        swap_ab: bool = False,
        quant_recipe: str = "nvfp4",
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            fast_math=fast_math,
            activation="silu",
            dynamic_down_scale=dynamic_down_scale,
            share_input_across_experts=share_input_across_experts,
            deterministic_output=deterministic_output,
            swap_ab=swap_ab,
            quant_recipe=quant_recipe,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )


__all__ = [
    "MoEDynamicKernelSilu",
    "MoEMicroKernelSilu",
]
