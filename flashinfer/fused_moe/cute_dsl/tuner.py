"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Auto-tuner for CuteDSL NVFP4 MoE kernels.

This module provides a TunableRunner implementation for the CuteDSL NVFP4 MoE
kernels, enabling automatic performance tuning across different GEMM tactics.

Tactic format follows TRT-LLM's style:
- GEMM1 (Gather + SwiGLU): (mma_tiler_mn, cluster_shape_mn, raster_along_m)
- GEMM2 (Finalize): (mma_tiler_mn, cluster_shape_mn, raster_along_m)

Reference: TensorRT-LLM/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
- Sm100BlockScaledContiguousGatherGroupedGemmSwigluFusionRunner.get_valid_tactics (line 1867)
- Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner.get_valid_tactics (line 1163)
"""

import itertools
from typing import Any, Callable, Dict, List, Tuple

import torch

from ...autotuner import (
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..utils import (
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)


# =============================================================================
# GEMM1 Tactics (Gather + SwiGLU Fusion)
# =============================================================================
# Reference: TRT-LLM cute_dsl_custom_ops.py line 1867-1897
# Sm100BlockScaledContiguousGatherGroupedGemmSwigluFusionRunner.get_valid_tactics
#
# Format: (mma_tiler_mn, cluster_shape_mn, raster_along_m)
# - mma_tiler_mn: (tile_size, N_tile) where tile_size is 128 or 256, N_tile is 128 or 256
# - cluster_shape_mn: (tile_size // 128, cluster_n) where cluster_n is fixed to 1 for Gather kernel
# - raster_along_m: False (fixed)


def get_gemm1_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid tactics for GEMM1 (Gather + SwiGLU Fusion).

    Reference: TRT-LLM cute_dsl_custom_ops.py line 1879-1897

    Args:
        tile_size: MMA tile M dimension (128 or 256)

    Returns:
        List of (mma_tiler_mn, cluster_shape_mn, raster_along_m) tuples
    """
    # From TRT-LLM line 1879-1883:
    # mma_tiler_mn_candidates = [(self.tile_size, 128), (self.tile_size, 256)]
    # cluster_shape_mn_candidates = [(self.tile_size // 128, 1)]  # Note: Only 1, not 2!
    # raster_along_m_candidates = [False]

    mma_tiler_mn_candidates = [(tile_size, 128), (tile_size, 256)]
    cluster_shape_mn_candidates = [
        (tile_size // 128, 1)
    ]  # Gather kernel only supports cluster_n=1
    raster_along_m_candidates = [False]

    tactics = []
    for mma_tiler_mn, cluster_shape_mn, raster_along_m in itertools.product(
        mma_tiler_mn_candidates, cluster_shape_mn_candidates, raster_along_m_candidates
    ):
        tactics.append((mma_tiler_mn, cluster_shape_mn, raster_along_m))

    return tactics


# =============================================================================
# GEMM2 Tactics (Finalize Fusion)
# =============================================================================
# Reference: TRT-LLM cute_dsl_custom_ops.py line 1163-1193
# Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner.get_valid_tactics
#
# Format: (mma_tiler_mn, cluster_shape_mn, raster_along_m)
# - mma_tiler_mn: (tile_size, N_tile) where tile_size is 128 or 256, N_tile is 128 or 256
# - cluster_shape_mn: (tile_size // 128, cluster_n) where cluster_n is 1 or 2
# - raster_along_m: False (fixed, theoretically more performant)


def get_gemm2_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid tactics for GEMM2 (Finalize Fusion).

    Reference: TRT-LLM cute_dsl_custom_ops.py line 1173-1193

    Args:
        tile_size: MMA tile M dimension (128 or 256)

    Returns:
        List of (mma_tiler_mn, cluster_shape_mn, raster_along_m) tuples
    """
    # From TRT-LLM line 1173-1179:
    # mma_tiler_mn_candidates = [(self.tile_size, 128), (self.tile_size, 256)]
    # cluster_shape_mn_candidates = [(self.tile_size // 128, 1), (self.tile_size // 128, 2)]
    # raster_along_m_candidates = [False]

    mma_tiler_mn_candidates = [(tile_size, 128), (tile_size, 256)]
    cluster_shape_mn_candidates = [(tile_size // 128, 1), (tile_size // 128, 2)]
    raster_along_m_candidates = [False]

    tactics = []
    for mma_tiler_mn, cluster_shape_mn, raster_along_m in itertools.product(
        mma_tiler_mn_candidates, cluster_shape_mn_candidates, raster_along_m_candidates
    ):
        tactics.append((mma_tiler_mn, cluster_shape_mn, raster_along_m))

    return tactics


# =============================================================================
# Combined MoE Tactics
# =============================================================================
# The MoE pipeline uses both GEMM1 and GEMM2, they must share the same tile_size
# (M dimension of mma_tiler_mn) because moe_sort uses tile_size for padding.
#
# Tactic format: (tile_size, gemm1_tactic, gemm2_tactic)
# - tile_size: 128 or 256 (shared by both GEMMs and moe_sort)
# - gemm1_tactic: (mma_tiler_mn, cluster_shape_mn, raster_along_m)
# - gemm2_tactic: (mma_tiler_mn, cluster_shape_mn, raster_along_m)


def get_moe_valid_tactics() -> List[Tuple]:
    """Get all valid MoE tactic combinations.

    Each tactic is a tuple: (tile_size, gemm1_tactic, gemm2_tactic)

    The tile_size must be shared between GEMM1 and GEMM2 because:
    1. moe_sort uses tile_size to pad tokens to tile boundaries
    2. Both GEMMs process the same padded token sequence

    Returns:
        List of (tile_size, gemm1_tactic, gemm2_tactic) tuples
    """
    tactics = []

    for tile_size in [128, 256]:
        gemm1_tactics = get_gemm1_valid_tactics(tile_size)
        gemm2_tactics = get_gemm2_valid_tactics(tile_size)

        for gemm1_tactic, gemm2_tactic in itertools.product(
            gemm1_tactics, gemm2_tactics
        ):
            tactics.append((tile_size, gemm1_tactic, gemm2_tactic))

    return tactics


# Pre-generate all valid tactics
# tile_size=128: 2 GEMM1 tactics × 4 GEMM2 tactics = 8
# tile_size=256: 2 GEMM1 tactics × 4 GEMM2 tactics = 8
# Total: 16 tactics
ALL_MOE_TACTICS = get_moe_valid_tactics()

# Default tactic (tile_size=128, smallest MMA tiles, cluster_n=1)
DEFAULT_MOE_TACTIC = (
    128,  # tile_size
    ((128, 128), (1, 1), False),  # gemm1_tactic
    ((128, 128), (1, 1), False),  # gemm2_tactic
)


def _extract_tactic_params(tactic: Tuple) -> Dict[str, Any]:
    """Extract parameters from a MoE tactic tuple.

    Args:
        tactic: (tile_size, gemm1_tactic, gemm2_tactic)

    Returns:
        Dictionary with all tactic parameters
    """
    tile_size, gemm1_tactic, gemm2_tactic = tactic
    gemm1_mma_tiler_mn, gemm1_cluster_shape_mn, gemm1_raster_along_m = gemm1_tactic
    gemm2_mma_tiler_mn, gemm2_cluster_shape_mn, gemm2_raster_along_m = gemm2_tactic

    return {
        "tile_size": tile_size,
        "gemm1_mma_tiler_mn": gemm1_mma_tiler_mn,
        "gemm1_cluster_shape_mn": gemm1_cluster_shape_mn,
        "gemm1_raster_along_m": gemm1_raster_along_m,
        "gemm2_mma_tiler_mn": gemm2_mma_tiler_mn,
        "gemm2_cluster_shape_mn": gemm2_cluster_shape_mn,
        "gemm2_raster_along_m": gemm2_raster_along_m,
    }


class CuteDslFusedMoENvfp4Runner(TunableRunner):
    """TunableRunner for CuteDSL NVFP4 MoE kernels.

    This runner enables auto-tuning of the CuteDSL NVFP4 MoE pipeline by
    trying different combinations of GEMM tactics.

    Tactic format follows TRT-LLM style:
        (tile_size, gemm1_tactic, gemm2_tactic)
    where:
        - tile_size: 128 or 256
        - gemm1_tactic: (mma_tiler_mn, cluster_shape_mn, raster_along_m)
        - gemm2_tactic: (mma_tiler_mn, cluster_shape_mn, raster_along_m)

    Input tensor indices (for dynamic_tensor_specs):
        0: x (num_tokens, hidden_size//2) - FP4 packed input
        1: x_sf (num_tokens, hidden_size//sf_vec_size) - input scale factors
        2: token_selected_experts (num_tokens, top_k) - expert assignments
        3: token_final_scales (num_tokens, top_k) - routing weights
        4-10: weight tensors (fixed size, don't depend on num_tokens)
        11: moe_output (num_tokens, hidden_size) - output buffer

    Args:
        forward_impl: The actual MoE implementation function.
        num_experts: Total number of experts.
        top_k: Number of experts selected per token.
        num_local_experts: Number of local experts (for expert parallelism).
        local_expert_offset: Starting expert index for this partition.
        use_fused_finalize: Whether to use fused finalize (default: True).
        output_dtype: Output data type (default: torch.bfloat16).
    """

    # Tensor initializers for dynamic tensors (indices 0, 1, 2, 3, 11)
    # These create valid dummy tensors for profiling with different num_tokens
    dynamic_tensor_initializers = [
        # 0: x - FP4 quantized input (uint8 packed)
        lambda shapes, dtype, device: torch.randint(
            0, 256, shapes, dtype=torch.uint8, device=device
        ),
        # 1: x_sf - FP8 scale factors (uint8)
        lambda shapes, dtype, device: torch.randint(
            1, 128, shapes, dtype=torch.uint8, device=device
        ),
        # 2: token_selected_experts - expert indices (int32, 0 to num_experts-1)
        lambda shapes, dtype, device: torch.randint(
            0,
            8,
            shapes,
            dtype=torch.int32,
            device=device,  # num_experts=8 typical
        ),
        # 3: token_final_scales - routing weights (float32, softmax normalized)
        lambda shapes, dtype, device: torch.softmax(
            torch.randn(shapes, device=device), dim=-1
        ).to(torch.float32),
        # 11: moe_output - output buffer (bfloat16)
        lambda shapes, dtype, device: torch.empty(shapes, dtype=dtype, device=device),
    ]

    # Tuning config with dynamic tensor specs for num_tokens dimension
    # Indices 0, 1, 2, 3, 11 all have num_tokens as their first dimension
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1, 2, 3, 11),  # x, x_sf, experts, scales, moe_output
                dim_idx=(0, 0, 0, 0, 0),  # First dimension is num_tokens for all
                gen_tuning_buckets=get_last_power_of_2_num_tokens_buckets(8192),
                map_to_tuning_buckets=lambda x: min(last_positive_power_of_2(x), 8192),
                tensor_initializers=dynamic_tensor_initializers,
            ),
        ),
    )

    def __init__(
        self,
        forward_impl: Callable,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
        use_fused_finalize: bool = True,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.use_fused_finalize = use_fused_finalize
        self.output_dtype = output_dtype

    def __hash__(self):
        return hash(
            (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.use_fused_finalize,
                self.output_dtype,
            )
        )

    def get_valid_tactics(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[Tuple[Any, ...]]:
        """Return list of valid tactics.

        Returns tactics in TRT-LLM format:
            (tile_size, gemm1_tactic, gemm2_tactic)

        Args:
            inputs: List of input tensors (not used for tactic validation).
            profile: Optimization profile (not used for tactic validation).

        Returns:
            List of valid tactic tuples.
        """
        # Return all pre-generated tactics
        # In practice, some might be invalid for certain problem sizes,
        # but the kernel will handle that with can_implement checks
        return ALL_MOE_TACTICS

    def forward(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        tactic: Tuple[Any, ...] = None,  # type: ignore[assignment]
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute the MoE forward pass with the specified tactic.

        Args:
            inputs: List of input tensors:
                [x, x_sf, token_selected_experts, token_final_scales,
                 w1_weight, w1_weight_sf, w1_alpha, fc2_input_scale,
                 w2_weight, w2_weight_sf, w2_alpha, moe_output (optional)]
            tactic: Tactic tuple (tile_size, gemm1_tactic, gemm2_tactic) or None for default.
            do_preparation: If True, perform one-time setup (not used).
            **kwargs: Additional keyword arguments passed to forward_impl.

        Returns:
            Output tensor from the MoE computation.
        """
        if tactic is None or tactic == -1:
            tactic = DEFAULT_MOE_TACTIC

        # Extract parameters from tactic
        params = _extract_tactic_params(tactic)

        # Unpack inputs
        (
            x,
            x_sf,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            fc2_input_scale,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            *optional_inputs,
        ) = inputs

        moe_output = optional_inputs[0] if optional_inputs else None

        # Call the implementation with tactic parameters
        return self.forward_impl(
            x=x,
            x_sf=x_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            tile_size=params["tile_size"],
            gemm1_mma_tiler_mn=params["gemm1_mma_tiler_mn"],
            gemm1_cluster_shape_mn=params["gemm1_cluster_shape_mn"],
            gemm2_mma_tiler_mn=params["gemm2_mma_tiler_mn"],
            gemm2_cluster_shape_mn=params["gemm2_cluster_shape_mn"],
            output_dtype=self.output_dtype,
            use_fused_finalize=self.use_fused_finalize,
            moe_output=moe_output,
            **kwargs,
        )


# =============================================================================
# Utility Functions
# =============================================================================


def print_all_tactics():
    """Print all valid MoE tactics for debugging."""
    print(f"Total MoE tactics: {len(ALL_MOE_TACTICS)}")
    print()
    for i, tactic in enumerate(ALL_MOE_TACTICS):
        tile_size, gemm1_tactic, gemm2_tactic = tactic
        print(f"Tactic {i}:")
        print(f"  tile_size: {tile_size}")
        print(
            f"  gemm1: mma_tiler_mn={gemm1_tactic[0]}, cluster_shape_mn={gemm1_tactic[1]}, raster_along_m={gemm1_tactic[2]}"
        )
        print(
            f"  gemm2: mma_tiler_mn={gemm2_tactic[0]}, cluster_shape_mn={gemm2_tactic[1]}, raster_along_m={gemm2_tactic[2]}"
        )
        print()
