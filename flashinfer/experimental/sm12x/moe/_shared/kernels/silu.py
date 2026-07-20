# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/silu.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""SwiGLU wrappers for the activation-specialized fused MoE backends."""

from __future__ import annotations

from typing import Tuple

from flashinfer.experimental.sm12x.moe._shared.kernels.dynamic import (
    MoEDynamicKernelBackend,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.micro import (
    MoEMicroKernelBackend,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.activations import (
    SWIGLUOAI_UNINTERLEAVE,
    normalize_swiglu_alpha_for_activation,
    normalize_swiglu_beta_for_activation,
    normalize_swiglu_limit_for_activation,
)


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
        compile_time_phase: int = 0,
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
            compile_time_phase=compile_time_phase,
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
        num_topk: int = 1,
        swap_ab: bool = False,
        quant_recipe: str = "nvfp4",
        w4a8_repacked: bool = False,
        direct_routing: bool = False,
        work_source: str = "materialized_queue",
        materialize_intermediate: bool = False,
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
            num_topk=num_topk,
            swap_ab=swap_ab,
            quant_recipe=quant_recipe,
            w4a8_repacked=w4a8_repacked,
            direct_routing=direct_routing,
            work_source=work_source,
            materialize_intermediate=materialize_intermediate,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )


class MoEMicroKernelSwiGLUOAI(MoEMicroKernelBackend):
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
        activation = SWIGLUOAI_UNINTERLEAVE
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            fast_math=fast_math,
            activation=activation,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=single_token,
            dynamic_down_scale=dynamic_down_scale,
            a8_mx_mode=a8_mx_mode,
            scale_format=scale_format,
            e8m0_scale_layout=e8m0_scale_layout,
            swiglu_limit=normalize_swiglu_limit_for_activation(
                activation, swiglu_limit
            ),
            swiglu_alpha=normalize_swiglu_alpha_for_activation(
                activation, swiglu_alpha
            ),
            swiglu_beta=normalize_swiglu_beta_for_activation(activation, swiglu_beta),
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


class MoEDynamicKernelSwiGLUOAI(MoEDynamicKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        fast_math: bool = False,
        dynamic_down_scale: bool = False,
        share_input_across_experts: bool = False,
        deterministic_output: bool = False,
        num_topk: int = 1,
        swap_ab: bool = False,
        quant_recipe: str = "nvfp4",
        w4a8_repacked: bool = False,
        direct_routing: bool = False,
        work_source: str = "materialized_queue",
        materialize_intermediate: bool = False,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
    ):
        activation = SWIGLUOAI_UNINTERLEAVE
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            fast_math=fast_math,
            activation=activation,
            dynamic_down_scale=dynamic_down_scale,
            share_input_across_experts=share_input_across_experts,
            deterministic_output=deterministic_output,
            num_topk=num_topk,
            swap_ab=swap_ab,
            quant_recipe=quant_recipe,
            w4a8_repacked=w4a8_repacked,
            direct_routing=direct_routing,
            work_source=work_source,
            materialize_intermediate=materialize_intermediate,
            swiglu_limit=normalize_swiglu_limit_for_activation(
                activation, swiglu_limit
            ),
            swiglu_alpha=normalize_swiglu_alpha_for_activation(
                activation, swiglu_alpha
            ),
            swiglu_beta=normalize_swiglu_beta_for_activation(activation, swiglu_beta),
        )


__all__ = [
    "MoEDynamicKernelSilu",
    "MoEDynamicKernelSwiGLUOAI",
    "MoEMicroKernelSilu",
    "MoEMicroKernelSwiGLUOAI",
]
