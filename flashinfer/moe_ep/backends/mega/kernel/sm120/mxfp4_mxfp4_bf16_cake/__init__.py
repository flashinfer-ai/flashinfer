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

"""Cake SM120 MXFP4 x MXFP4 (W4A4) persistent MegaMoE backend (hybrid LSA + GIN transport)."""

from .cake_mxfp4_backend import Sm120Mxfp4Mxfp4CakeMegaKernelBackend
from .cake_mxfp4_config import Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig
from .cake_mxfp4_quant import (
    mxfp4_dequantize_torch,
    mxfp4_quantize_fused,
    mxfp4_quantize_torch,
)

__all__ = [
    "Sm120Mxfp4Mxfp4CakeMegaKernelBackend",
    "Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig",
    "mxfp4_dequantize_torch",
    "mxfp4_quantize_fused",
    "mxfp4_quantize_torch",
]
