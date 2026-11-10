# Copyright (c) 2026 by FlashInfer team.
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

"""Cake SM120 NVFP4 x NVFP4 (W4A4) persistent MegaMoE backend (hybrid LSA + GIN transport)."""

from .cake_nvfp4_backend import Sm120Nvfp4Nvfp4CakeMegaKernelBackend
from .cake_nvfp4_config import Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig
from .cake_nvfp4_weights import (
    TransformedCakeNvfp4MegaWeights,
    mxfp4_scales_to_nvfp4,
    mxfp4_scales_to_nvfp4_shifted,
    nvfp4_quantize_torch,
    preprocess_nvfp4_mega_weights,
)

__all__ = [
    "Sm120Nvfp4Nvfp4CakeMegaKernelBackend",
    "Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig",
    "TransformedCakeNvfp4MegaWeights",
    "mxfp4_scales_to_nvfp4",
    "mxfp4_scales_to_nvfp4_shifted",
    "nvfp4_quantize_torch",
    "preprocess_nvfp4_mega_weights",
]
