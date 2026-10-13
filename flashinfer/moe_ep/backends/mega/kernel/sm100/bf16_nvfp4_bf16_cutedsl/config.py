"""CuTe DSL W4A16 MegaMoE configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


@dataclass
class Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """BF16 activations and NVFP4 expert weights with online dequantization.

    Input, FC1 activation handoff, dispatch/combine and output remain BF16.
    Routing scores are applied after FC2 by default. ``apply_topk_in_fc1``
    instead multiplies the completed FP32 SwiGLU activation before its BF16
    handoff, changing the rounding contract. ``MoEWeightPack`` supplies packed
    weight data and E4M3 block scales; ``fc1_alpha`` and ``fc2_alpha`` supply
    optional per-expert FP32 epilogue scales (default one). ``fc1_norm_const``
    scales the completed activation, after optional routing weights and before
    the BF16 FC1 handoff; it does not implicitly compensate ``fc2_alpha``.
    MiniMax uses ``swiglu_alpha=1.702, swiglu_beta=1.0``. ``activation="situ"``
    selects ``beta * tanh(gate / beta) * sigmoid(gate)`` on the gate branch,
    with optional ``situ_linear_beta * tanh(up / situ_linear_beta)`` on up.
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

    apply_topk_in_fc1: bool = False
    # Append new fields to preserve existing positional construction.
    fc1_norm_const: torch.Tensor | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    activation: Literal["swiglu", "situ"] = "swiglu"
    situ_beta: float | None = None
    situ_linear_beta: float | None = None
    activation_clamp: float | None = None

    def __post_init__(self) -> None:
        if (self.swiglu_alpha is None) != (self.swiglu_beta is None):
            raise ValueError("swiglu_alpha and swiglu_beta must be set together.")
        if self.activation not in ("swiglu", "situ"):
            raise ValueError(
                f"activation must be 'swiglu' or 'situ', got {self.activation!r}."
            )
        if self.activation == "situ":
            if self.swiglu_alpha is not None:
                raise ValueError("SwiGLU parameters are not supported with SiTU.")
            if self.situ_beta is None:
                raise ValueError("activation='situ' requires situ_beta.")
            if not math.isfinite(self.situ_beta) or self.situ_beta <= 0:
                raise ValueError("situ_beta must be positive and finite.")
            if self.situ_linear_beta is not None and (
                not math.isfinite(self.situ_linear_beta) or self.situ_linear_beta <= 0
            ):
                raise ValueError(
                    "situ_linear_beta must be positive and finite when set."
                )
            if self.gate_up_clamp is not None or self.activation_clamp is not None:
                raise ValueError("activation clamps are not supported with SiTU.")
        elif self.situ_beta is not None or self.situ_linear_beta is not None:
            raise ValueError("SiTU parameters require activation='situ'.")
        if (
            self.gate_up_clamp is not None
            and self.activation_clamp is not None
            and self.gate_up_clamp != self.activation_clamp
        ):
            raise ValueError(
                "gate_up_clamp and activation_clamp disagree "
                f"({self.gate_up_clamp} vs {self.activation_clamp}); pass only one."
            )
        if (
            self.knobs is not None
            and not isinstance(self.knobs, dict)
            and self.knobs != "auto"
        ):
            raise ValueError("W4A16 knobs must be a dict, 'auto', or None")
