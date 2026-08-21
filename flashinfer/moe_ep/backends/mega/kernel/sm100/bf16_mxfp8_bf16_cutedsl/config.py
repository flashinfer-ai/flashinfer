"""CuTeDSL mixed MXFP8-weight/BF16-activation MegaMoE config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..bf16_bf16_bf16_cutedsl.config import Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig


@dataclass
class Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
    Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig
):
    """Parameters for the mixed SwapAB SM100 CuTeDSL MegaMoE kernel.

    ``intermediate_size`` is the post-SwiGLU width, matching NVFP4 and SGLang.
    Activations and output stay BF16. Expert weights are MXFP8 with one E8M0
    scale per K32 block; canonical BF16 weights are quantized by preprocessing.
    """

    kernel_name: str = "sm100_bf16_mxfp8_bf16_cutedsl"
    kind: Literal["bf16_mxfp8_e4m3", "bf16_mxfp8_e5m2"] = "bf16_mxfp8_e4m3"
    token_back_mode: Literal["epi_warps", "reuse_dispatch_warps"] = "epi_warps"
