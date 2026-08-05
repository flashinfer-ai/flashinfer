"""DeepGEMM mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sm100_Fp8_Nvfp4_Bf16_Deepgemm_MegaMoeConfig:
    """Kernel params for ``deep_gemm.fp8_fp4_mega_moe``."""

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_fp8_nvfp4_bf16_deepgemm"
    activation_clamp: float | None = None
    fast_math: bool = True
