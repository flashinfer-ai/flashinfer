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

import importlib.util


def is_cute_dsl_available() -> bool:
    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


# Conditionally import CuTe-DSL kernels (including utils which requires cutlass)
if is_cute_dsl_available():
    from .utils import make_ptr, get_cutlass_dtype, get_num_sm
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

    # Backwards-compatible re-exports from flashinfer.norm.kernels submodule
    from ..norm.kernels import (
        # Kernel classes
        RMSNormKernel,
        QKRMSNormKernel,
        RMSNormQuantKernel,
        FusedAddRMSNormKernel,
        FusedAddRMSNormQuantKernel,
        LayerNormKernel,
        # Python API functions
        rmsnorm_cute,
        qk_rmsnorm_cute,
        rmsnorm_quant_cute,
        fused_add_rmsnorm_cute,
        fused_add_rmsnorm_quant_cute,
        layernorm_cute,
    )

__all__ = [
    # Always available
    "is_cute_dsl_available",
]

if is_cute_dsl_available():
    __all__ += [
        # Utils (require cutlass)
        "make_ptr",
        "get_cutlass_dtype",
        "get_num_sm",
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
        # Norm kernels (CuTe DSL) - backwards-compatible re-exports
        "RMSNormKernel",
        "QKRMSNormKernel",
        "RMSNormQuantKernel",
        "FusedAddRMSNormKernel",
        "FusedAddRMSNormQuantKernel",
        "LayerNormKernel",
        "rmsnorm_cute",
        "qk_rmsnorm_cute",
        "rmsnorm_quant_cute",
        "fused_add_rmsnorm_cute",
        "fused_add_rmsnorm_quant_cute",
        "layernorm_cute",
    ]
