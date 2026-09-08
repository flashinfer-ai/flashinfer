"""CuTe DSL W4A16 MegaMoE configuration."""

from dataclasses import dataclass


@dataclass
class Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """BF16 activations and NVFP4 expert weights with online dequantization.

    Input, FC1 activation handoff, dispatch/combine and output remain BF16.
    Routing scores are applied after FC2. Packed weight data, E4M3 block
    scales and optional FP32 global scales are supplied in ``MoEWeightPack``.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_bf16_nvfp4_bf16_cutedsl"
    gate_up_clamp: float | None = None
    fast_math: bool = True
    knobs: dict | None = None
