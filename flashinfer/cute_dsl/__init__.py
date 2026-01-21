# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FlashInfer CuTe-DSL Kernels
===========================

This module provides high-performance GPU kernels implemented using NVIDIA CuTe-DSL.
"""

from .utils import (
    is_cute_dsl_available,
    make_ptr,
    get_cutlass_dtype,
    get_num_sm,
    convert_sf_to_mma_layout,
    convert_sf_from_mma_layout,
    get_mma_sf_shape,
)

# Conditionally import CuTe-DSL kernels
if is_cute_dsl_available():
    from .blockscaled_gemm import (
        grouped_gemm_nt_masked,
        Sm100BlockScaledPersistentDenseGemmKernel,
    )
    from .blockscaled_contiguous_grouped_gemm import (
        blockscaled_contiguous_grouped_gemm_nvfp4,
        create_tile_mapping,
        create_scale_factor_tensor,
        Sm100BlockScaledContiguousGroupedGemmKernel,
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
    )
    from .blockscaled_contiguous_grouped_gemm_swiglu_fusion import (
        Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel,
        blockscaled_contiguous_grouped_gemm_swiglu_fusion_nvfp4,
    )
    from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
        blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
        create_finalize_fusion_tensors,
    )
    from .blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import (
        BlockScaledContiguousGatherGroupedGemmKernel,
        blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4,
        create_gather_gemm_tensors,
    )
    from .rmsnorm_fp4quant import (
        rmsnorm_fp4quant,
        RMSNormFP4QuantKernel,
        get_sm_version,
    )
    from .add_rmsnorm_fp4quant import (
        add_rmsnorm_fp4quant,
        AddRMSNormFP4QuantKernel,
    )
    from .fused_moe import (
        cute_dsl_fused_moe_nvfp4,
    )
    from .tuner import (
        CuteDslFusedMoENvfp4Runner,
        get_gemm1_valid_tactics,
        get_gemm2_valid_tactics,
        get_moe_valid_tactics,
        ALL_MOE_TACTICS,
        DEFAULT_MOE_TACTIC,
    )

__all__ = [
    # Utils (always available)
    "is_cute_dsl_available",
    "make_ptr",
    "get_cutlass_dtype",
    "get_num_sm",
    # Scale factor layout conversion utilities
    "convert_sf_to_mma_layout",
    "convert_sf_from_mma_layout",
    "get_mma_sf_shape",
]

if is_cute_dsl_available():
    __all__ += [
        # Blockscaled GEMM
        "grouped_gemm_nt_masked",
        "Sm100BlockScaledPersistentDenseGemmKernel",
        # Blockscaled Contiguous Grouped GEMM (for MoE)
        "blockscaled_contiguous_grouped_gemm_nvfp4",
        "create_tile_mapping",
        "create_scale_factor_tensor",
        "Sm100BlockScaledContiguousGroupedGemmKernel",
        "cvt_sf_MKL_to_M32x4xrm_K4xrk_L",
        # Blockscaled Contiguous Grouped GEMM with SwiGLU Fusion (for MoE)
        "Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel",
        "blockscaled_contiguous_grouped_gemm_swiglu_fusion_nvfp4",
        # Blockscaled Contiguous Grouped GEMM with Finalize Fusion (for MoE)
        "Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel",
        "blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4",
        "create_finalize_fusion_tensors",
        # Blockscaled Contiguous Gather Grouped GEMM with SwiGLU Fusion (for MoE)
        "BlockScaledContiguousGatherGroupedGemmKernel",
        "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4",
        "create_gather_gemm_tensors",
        # RMSNorm + FP4 Quantization
        "rmsnorm_fp4quant",
        "RMSNormFP4QuantKernel",
        "get_sm_version",
        # Add + RMSNorm + FP4 Quantization
        "add_rmsnorm_fp4quant",
        "AddRMSNormFP4QuantKernel",
        # Fused MoE (high-level API)
        "cute_dsl_fused_moe_nvfp4",
        # Auto-tuner for CuteDSL MoE
        "CuteDslFusedMoENvfp4Runner",
        "get_gemm1_valid_tactics",
        "get_gemm2_valid_tactics",
        "get_moe_valid_tactics",
        "ALL_MOE_TACTICS",
        "DEFAULT_MOE_TACTIC",
    ]
