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

from .utils import is_cute_dsl_available, make_ptr, get_cutlass_dtype, get_num_sm

# Conditionally import CuTe-DSL kernels
if is_cute_dsl_available():
    from .blockscaled_gemm import (
        grouped_gemm_nt_masked,
        Sm100BlockScaledPersistentDenseGemmKernel,
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

__all__ = [
    # Utils (always available)
    "is_cute_dsl_available",
    "make_ptr",
    "get_cutlass_dtype",
    "get_num_sm",
]

if is_cute_dsl_available():
    __all__ += [
        # Blockscaled GEMM
        "grouped_gemm_nt_masked",
        "Sm100BlockScaledPersistentDenseGemmKernel",
        # RMSNorm + FP4 Quantization
        "rmsnorm_fp4quant",
        "RMSNormFP4QuantKernel",
        "get_sm_version",
        # Add + RMSNorm + FP4 Quantization
        "add_rmsnorm_fp4quant",
        "AddRMSNormFP4QuantKernel",
    ]
