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
FlashInfer GEMM Kernels (internal)
===================================

Internal module containing GPU GEMM kernel implementations.
Import from ``flashinfer.gemm`` for the public API.

CuTe-DSL GEMM Kernels
=====================

This module contains CuTe-DSL implementations of GEMM kernels for FP8
batched matrix multiplication.

Supported architectures:
- SM100 (Blackwell): bmm_fp8_blackwell.py
- SM107 (Rubin): bmm_fp8_rubin.py
"""

from flashinfer.cute_dsl.availability import (
    is_cute_dsl_available,
    is_rubin_cute_dsl_available,
)

__all__ = []

if is_cute_dsl_available():
    from .bmm_fp8_wrapper import (
        bmm_fp8_cute_dsl,
        cute_bmm_fp8_can_implement,
        SM107_AUTOTUNE_CONFIGS,
        get_valid_sm107_configs,
    )
    from .bmm_fp8_blackwell import PersistentDenseGemmKernel

    # SM107 kernels need CuTe DSL >= 4.8; skip the re-export on older DSL so
    # that importing FlashInfer still works there.
    if is_rubin_cute_dsl_available():
        from .bmm_fp8_rubin import SM107PersistentDenseGemmKernel

    __all__ += [
        # FP8 BMM wrapper functions
        "bmm_fp8_cute_dsl",
        "cute_bmm_fp8_can_implement",
        # Autotune configuration spaces
        "SM107_AUTOTUNE_CONFIGS",
        # Configuration validation helpers
        "get_valid_sm107_configs",
        # Kernel classes
        "PersistentDenseGemmKernel",
    ]

    if is_rubin_cute_dsl_available():
        __all__ += ["SM107PersistentDenseGemmKernel"]
