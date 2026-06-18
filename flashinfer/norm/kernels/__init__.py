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
CuTe DSL Norm Kernels
=====================

Internal kernel implementations using NVIDIA CuTe-DSL.
"""

from .rmsnorm import (
    RMSNormKernel,
    QKRMSNormKernel,
    RMSNormQuantKernel,
    rmsnorm_cute,
    qk_rmsnorm_cute,
    rmsnorm_quant_cute,
)
from .fused_add_rmsnorm import (
    FusedAddRMSNormKernel,
    FusedAddRMSNormQuantKernel,
    fused_add_rmsnorm_cute,
    fused_add_rmsnorm_quant_cute,
)
from .layernorm import (
    LayerNormKernel,
    layernorm_cute,
)

__all__ = [
    # RMSNorm
    "RMSNormKernel",
    "QKRMSNormKernel",
    "RMSNormQuantKernel",
    "rmsnorm_cute",
    "qk_rmsnorm_cute",
    "rmsnorm_quant_cute",
    # Fused Add + RMSNorm
    "FusedAddRMSNormKernel",
    "FusedAddRMSNormQuantKernel",
    "fused_add_rmsnorm_cute",
    "fused_add_rmsnorm_quant_cute",
    # LayerNorm
    "LayerNormKernel",
    "layernorm_cute",
]
