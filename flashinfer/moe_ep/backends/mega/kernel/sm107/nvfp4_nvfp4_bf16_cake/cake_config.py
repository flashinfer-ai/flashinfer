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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from .cake_plan import GATE_UP_CLAMP, TOP_K

if TYPE_CHECKING:
    import torch


@dataclass
class Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig:
    """Rubin NVFP4 x NVFP4 MegaMoE from generated Cake programs.

    Takes the inputs of ``sm107_nvfp4_nvfp4_bf16_cutedsl``: BF16 activations
    (staged to NVFP4 with ``input_norm_const``) or NVFP4 codes with their
    scales, prequantized NVFP4 expert weights, ``fc1_alpha`` / ``fc2_alpha`` /
    ``fc1_norm_const`` per local expert, routing weights applied before FC2.
    The programs are generated for DeepSeek-V4-Pro's expert-parallel slice
    (hidden 7168, intermediate 3072, 48 experts over 4 ranks, top-6, SwiGLU
    clamped at 10) and three token tiers per rank (1024, 4096, 8192); other
    geometries are refused at initialization. From 4096 tokens per rank the
    combine sums each token's terms per destination rank in FP32 and rounds
    that sum to BF16 before the final reduce (the dedup combine).
    """

    intermediate_size: int
    top_k: int = TOP_K
    kernel_name: str = "sm107_nvfp4_nvfp4_bf16_cake"
    gate_up_clamp: Optional[float] = GATE_UP_CLAMP
    activation: Literal["swiglu"] = "swiglu"
    apply_routing_weights_before_fc2: bool = True
    combine_dtype: Literal["bf16"] = "bf16"
    input_norm_const: float = 1.0
    fc1_alpha: Optional[torch.Tensor] = None
    fc2_alpha: Optional[torch.Tensor] = None
    fc1_norm_const: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.kernel_name != "sm107_nvfp4_nvfp4_bf16_cake":
            raise ValueError(
                "Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig.kernel_name must be "
                f"'sm107_nvfp4_nvfp4_bf16_cake', got {self.kernel_name!r}"
            )
        if self.top_k != TOP_K:
            raise ValueError(
                f"the Cake NVFP4 programs route top-{TOP_K}, got {self.top_k}"
            )
        if self.gate_up_clamp != GATE_UP_CLAMP:
            raise ValueError(
                f"the Cake NVFP4 programs clamp gate and up at {GATE_UP_CLAMP}, "
                f"got gate_up_clamp={self.gate_up_clamp!r}"
            )
        if self.activation != "swiglu":
            raise ValueError(
                f"the Cake NVFP4 programs run SwiGLU, got {self.activation!r}"
            )
        if self.apply_routing_weights_before_fc2 is not True:
            raise ValueError(
                "the Cake NVFP4 programs apply the routing weights before FC2"
            )
        if self.combine_dtype != "bf16":
            raise ValueError(
                f"the Cake NVFP4 programs combine in BF16, got {self.combine_dtype!r}"
            )


__all__ = ["Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig"]
