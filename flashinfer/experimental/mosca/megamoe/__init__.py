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

# MOSCA MegaMoE kernels for MoEEpMegaLayer (experimental). See README.md.
#
# Importing this package imports ``backend``, which registers
# ``kernel_name="sm100_fp8_fp4_bf16_mosca"`` with the moe_ep mega-kernel
# registry; core never imports it, so importing the config is the opt-in.
# Everything imported here stays lightweight (no tvm-ffi, JIT or
# symmetric-memory imports at module level).

from __future__ import annotations

from .backend import MoscaWorkspace, Sm100MoscaMegaKernelBackend, split_rows
from .config import (
    DSV4_PRO_DEP8_TILE_RULE,
    KERNEL_NAME,
    TOKEN_LAYOUTS,
    Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig,
)
from .weights import TransformedMoscaMegaWeights, preprocess_mega_weights

__all__ = [
    "DSV4_PRO_DEP8_TILE_RULE",
    "KERNEL_NAME",
    "MoscaWorkspace",
    "Sm100MoscaMegaKernelBackend",
    "Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig",
    "TOKEN_LAYOUTS",
    "TransformedMoscaMegaWeights",
    "preprocess_mega_weights",
    "split_rows",
]
