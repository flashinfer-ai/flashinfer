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

from importlib import import_module


# Keep the CUDA specialist importable when optional CuTe DSL dependencies are
# absent. Import each DSL implementation only when its exported name is used.
_KERNEL_MODULES = {
    "RMSNormKernel": "rmsnorm",
    "QKRMSNormKernel": "rmsnorm",
    "RMSNormQuantKernel": "rmsnorm",
    "rmsnorm_cute": "rmsnorm",
    "qk_rmsnorm_cute": "rmsnorm",
    "rmsnorm_quant_cute": "rmsnorm",
    "FusedAddRMSNormKernel": "fused_add_rmsnorm",
    "FusedAddRMSNormQuantKernel": "fused_add_rmsnorm",
    "fused_add_rmsnorm_cute": "fused_add_rmsnorm",
    "fused_add_rmsnorm_quant_cute": "fused_add_rmsnorm",
    "LayerNormKernel": "layernorm",
    "LayerNormQuantKernel": "layernorm",
    "layernorm_cute": "layernorm",
    "layernorm_quant_cute": "layernorm",
    "gated_residual_rmsnorm": "gated_residual_rmsnorm",
    "gated_residual_next_rmsnorm": "gated_residual_next_rmsnorm",
    "rmsnorm_sigmoid_gate": "rmsnorm_sigmoid_gate",
    "supports_rmsnorm_sigmoid_gate": "rmsnorm_sigmoid_gate",
}


def __getattr__(name):
    module = _KERNEL_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


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
    "LayerNormQuantKernel",
    "layernorm_cute",
    "layernorm_quant_cute",
    # Specialized fused normalization kernels
    "gated_residual_rmsnorm",
    "gated_residual_next_rmsnorm",
    "rmsnorm_sigmoid_gate",
    "supports_rmsnorm_sigmoid_gate",
]
