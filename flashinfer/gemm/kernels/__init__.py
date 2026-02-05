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
FlashInfer GEMM Kernels
=======================

This module provides high-performance GPU GEMM kernels implemented using NVIDIA CuTe-DSL.
"""

from flashinfer.cute_dsl.utils import is_cute_dsl_available

# Conditionally import CuTe-DSL kernels
if is_cute_dsl_available():
    from .grouped_gemm_masked_blackwell import (
        grouped_gemm_nt_masked,
        Sm100BlockScaledPersistentDenseGemmKernel,
        create_scale_factor_tensor,
    )

__all__ = []

if is_cute_dsl_available():
    __all__ += [
        "grouped_gemm_nt_masked",
        "Sm100BlockScaledPersistentDenseGemmKernel",
        "create_scale_factor_tensor",
    ]
