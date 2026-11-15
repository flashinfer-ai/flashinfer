"""Configuration for the exact Blackwell BF16 rank-major MegaMoE kernel."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig:
    """Fixed production coordinate for the rank-major CUDA backend (SM100 and SM107)."""

    intermediate_size: int = 2048
    top_k: int = 8
    kernel_name: str = field(
        default="sm100_bf16_bf16_bf16_rank_major_cuda",
        init=False,
    )
