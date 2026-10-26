"""
Copyright (c) 2026 by FlashInfer team.

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

# Experimental Cake W8A8 MXFP8 MegaMoE EP16 backend (SM103) for MoEEpMegaLayer.
#
# Importing ``cake_backend`` registers ``kernel_name="sm103_mxfp8_mxfp8_bf16_cake"``;
# the stable ``flashinfer.moe_ep`` package performs that import through its
# ``backends/mega/kernel/sm103`` package, so everything imported here stays
# lightweight (no tvm-ffi, JIT or NVSHMEM imports at module level). See
# README.md in this directory.

from __future__ import annotations

from .cake_backend import Sm103Mxfp8Mxfp8CakeMegaKernelBackend, routing_overflow
from .cake_config import KERNEL_NAME, Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig
from .cake_weights import TransformedCakeW8a8MegaWeights, preprocess_mega_weights

__all__ = [
    "KERNEL_NAME",
    "Sm103Mxfp8Mxfp8CakeMegaKernelBackend",
    "Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig",
    "TransformedCakeW8a8MegaWeights",
    "preprocess_mega_weights",
    "routing_overflow",
]
