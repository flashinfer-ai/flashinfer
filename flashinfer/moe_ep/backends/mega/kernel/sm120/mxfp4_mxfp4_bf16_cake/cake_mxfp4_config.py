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

"""Configuration for the Cake SM120 MXFP4 x MXFP4 (W4A4) persistent MegaMoE backend."""

from __future__ import annotations

from dataclasses import dataclass

from ..mxfp4_mxfp8_bf16_cake.cake_config import Sm120CakeMegaMoeConfigBase


@dataclass
class Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig(Sm120CakeMegaMoeConfigBase):
    """Static model parameters of the Cake persistent MegaMoE kernel with MXFP4 activations.

    Everything except the activation format is that of
    :class:`~flashinfer.moe_ep.Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig` (same fields, the same
    MXFP4 weight pack, transport tables, shared expert, activations, return codecs and tuning):

    * activations are MXFP4: E2M1 values with one UE8M0 scale per 32 values (``quantize_input``
      uses ``scale = 2^ceil(log2(amax * fl32(1 / 6)))``, see
      ``mxfp4_mxfp4_bf16_cake.cake_mxfp4_quant``), so a dispatch record carries ``H / 2 + H / 32``
      bytes per token instead of ``H + H / 32``;
    * both GEMMs run at the FP4 rate (``mma.sync kind::mxf4nvf4`` with UE8M0 block-32 scales)
      against the unchanged MXFP4 routed weights;
    * the gated-activation output (times the routing weight) is requantized to MXFP4 per 32
      values (``scale = 2^ceil(log2(amax * (1 / 6)))``) before FC2;
    * no global scales (MX formats carry the full dynamic range in the block scales);
    * the in-kernel FP8 shared expert (``shared_expert="fp8_in_kernel"``) reads the E4M3 image of
      the MXFP4 input (exact) with its UE8M0 scales.

    MXFP4 activations need ``hidden_size % 1024 == 0`` (and ``intermediate_size % 256 == 0`` with
    the shared expert).  Accuracy: on DeepSeek-V4-Flash routed experts the MXFP4 intermediate and
    input rounding gives a larger error than NVFP4 activations (``sm120_nvfp4_nvfp4_bf16_cake``,
    E4M3 block-16 scales) at the same speed; this backend serves MXFP4-native recipes that keep
    the checkpoint's MX weights unchanged.
    """

    kernel_name: str = "sm120_mxfp4_mxfp4_bf16_cake"


__all__ = ["Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig"]
