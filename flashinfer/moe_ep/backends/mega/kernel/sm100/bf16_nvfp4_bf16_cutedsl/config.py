"""CuTe DSL W4A16 MegaMoE configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """BF16 activations and NVFP4 expert weights with online dequantization.

    Input, FC1 activation handoff, dispatch/combine and output remain BF16.
    Routing scores are applied after FC2. Packed weight data, E4M3 block
    scales and optional FP32 global scales are supplied in ``MoEWeightPack``.
    SwiGLU uses the kernel's fixed approximate exp2/reciprocal implementation.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_bf16_nvfp4_bf16_cutedsl"
    gate_up_clamp: float | None = None
    # None looks up a recorded winner or the built-in profile; a dict overrides
    # both. "auto" tunes collectively on the first forward, before capture.
    knobs: dict | Literal["auto"] | None = None

    def __post_init__(self) -> None:
        if (
            self.knobs is not None
            and not isinstance(self.knobs, dict)
            and self.knobs != "auto"
        ):
            raise ValueError("W4A16 knobs must be a dict, 'auto', or None")
