"""CuTeDSL NVFP4 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Nvfp4CutedslMegaMoeConfig:
    """Kernel params for ``cutedsl_nvfp4_mega_moe_front_end.nvfp4_mega_moe``."""

    intermediate_size: int
    top_k: int
    kernel_name: str = "nvfp4_cutedsl"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    apply_topk_in_fc1: bool = True
