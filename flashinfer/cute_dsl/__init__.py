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

.. deprecated::
    Importing GEMM kernels (``grouped_gemm_nt_masked``,
    ``Sm100BlockScaledPersistentDenseGemmKernel``, ``create_scale_factor_tensor``)
    from ``flashinfer.cute_dsl`` is deprecated.
    Use ``flashinfer.gemm`` instead. The old import paths will be
    removed in a future release.
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
    # Deprecated GEMM symbols: re-exported for backwards compatibility.
    # Use flashinfer.gemm instead.
    from .blockscaled_gemm import (
        grouped_gemm_nt_masked,
        Sm100BlockScaledPersistentDenseGemmKernel,
        create_scale_factor_tensor,
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
    from .gated_delta_rule import (
        gated_delta_rule,
        GatedDeltaRuleKernel,
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
        # Blockscaled GEMM (deprecated, use flashinfer.gemm instead)
        "grouped_gemm_nt_masked",
        "Sm100BlockScaledPersistentDenseGemmKernel",
        "create_scale_factor_tensor",
        # RMSNorm + FP4 Quantization
        "rmsnorm_fp4quant",
        "RMSNormFP4QuantKernel",
        "get_sm_version",
        # Add + RMSNorm + FP4 Quantization
        "add_rmsnorm_fp4quant",
        "AddRMSNormFP4QuantKernel",
        # Gated Delta Rule
        "gated_delta_rule",
        "GatedDeltaRuleKernel",
    ]
