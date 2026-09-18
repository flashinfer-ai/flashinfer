"""CuTe DSL W4A16 MegaMoE configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


@dataclass
class Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """BF16 activations and NVFP4 expert weights with online dequantization.

    Input, FC1 activation handoff, dispatch/combine and output remain BF16.
    Routing scores are applied after FC2. ``MoEWeightPack`` supplies packed
    weight data and E4M3 block scales; ``fc1_alpha`` and ``fc2_alpha`` supply
    optional per-expert FP32 epilogue scales (default one).
    SwiGLU uses the kernel's fixed approximate exp2/reciprocal implementation.
    In-kernel FC2 reduction is opt-in: its BF16 atomic accumulation is
    nondeterministic; the default combines BF16 partials in FP32.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_bf16_nvfp4_bf16_cutedsl"
    gate_up_clamp: float | None = None
    # Permit BF16 in-kernel reduction; autotune may still choose external reduction.
    enable_in_kernel_fc2_reduce: bool = False
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
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
