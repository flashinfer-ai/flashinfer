"""Blackwell standard MegaMoE with MXFP4 weights and MXFP8 activations."""

from dataclasses import dataclass, field
from typing import Literal

from ..mxfp8_mxfp8_bf16_cutedsl.config import (
    Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
)


@dataclass
class Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
    Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
):
    """Use packed E2M1 weights and E4M3 activations with per-32 E8M0 scales.

    Hidden and intermediate sizes must be multiples of 128. BF16 inputs use
    fused quantization and staging when their layout permits it; prequantized
    inputs supply E4M3 activations and linear E8M0 scales. Routes follow the
    Blackwell caller-validated contract: unique, in-range IDs with finite
    scores, or masked IDs of -1. Masked scores are staged as zero.

    This selects standard inference. ``knobs="auto"`` tunes during collective
    ``autotune_v2()`` warmup, before CUDA graph capture.
    """

    kernel_name: str = field(default="sm100_mxfp8_mxfp4_bf16_cutedsl", init=False)
    kind: Literal["mxfp4_mxfp8"] = field(default="mxfp4_mxfp8", init=False)
    kernel_variant: Literal["inference"] = field(default="inference", init=False)
