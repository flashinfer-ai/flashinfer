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

# BF16 x NVFP4 -> BF16 MegaMoE over the Cake fused program (SM103 / SM100).

from .cake_backend import (
    CakeWorkspace,
    Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend,
)
from .cake_config import KERNEL_NAME, Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig
from .cake_plan import CELLS, Cell, select_cell
from .cake_programs import ProgramHandle, programs, programs_available, set_programs
from .cake_weights import (
    TransformedCakeW4a16MegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

__all__ = [
    "CELLS",
    "KERNEL_NAME",
    "CakeWorkspace",
    "Cell",
    "ProgramHandle",
    "Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend",
    "Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig",
    "TransformedCakeW4a16MegaWeights",
    "preprocess_mega_weights",
    "programs",
    "programs_available",
    "select_cell",
    "set_programs",
    "validate_transformed_mega_weights",
]
