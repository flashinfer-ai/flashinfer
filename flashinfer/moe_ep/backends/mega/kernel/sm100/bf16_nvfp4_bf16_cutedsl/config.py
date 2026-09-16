"""CuTeDSL mixed NVFP4-weight/BF16-activation MegaMoE config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

from ..common.bf16_config import Sm100_Bf16_Cutedsl_MegaMoeConfigBase


@dataclass
class Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(Sm100_Bf16_Cutedsl_MegaMoeConfigBase):
    """Parameters for the mixed SwapAB SM100 CuTeDSL MegaMoE kernel.

    ``intermediate_size`` is the post-SwiGLU width, matching NVFP4 and SGLang.
    Activations and output stay BF16. Expert weights are NVFP4 with one E4M3
    scale per K16 block; canonical BF16 weights are quantized by preprocessing.
    """

    kernel_name: str = "sm100_bf16_nvfp4_bf16_cutedsl"
    kind: Literal["bf16_nvfp4"] = "bf16_nvfp4"
    fc1_alpha: "torch.Tensor | None" = None
    fc2_alpha: "torch.Tensor | None" = None
