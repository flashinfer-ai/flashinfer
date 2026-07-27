"""DeepGEMM mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeepGemmMegaMoeConfig:
    """Kernel params for ``deep_gemm.fp8_fp4_mega_moe``."""

    intermediate_size: int
    top_k: int
    kernel_name: str = "deep_gemm_mega"
    activation_clamp: float | None = None
    fast_math: bool = True
