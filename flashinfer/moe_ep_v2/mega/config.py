"""Mega-path config: fused DeepGEMM mega-MoE kernel."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeepGemmMegaMoeConfig:
    """Kernel params for ``deep_gemm.fp8_fp4_mega_moe``."""

    intermediate_size: int
    top_k: int
    activation_clamp: float | None = None
    fast_math: bool = True


@dataclass
class MegaConfig:
    """Fused expert-parallel mega kernel (symmetric memory)."""

    kernel: DeepGemmMegaMoeConfig
    stage_inputs: bool = True
    preprocess_weights: bool = True


__all__ = ["DeepGemmMegaMoeConfig", "MegaConfig"]
