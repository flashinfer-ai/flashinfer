"""Configuration for the SM90 push NVFP4 mega-MoE backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm90PushNvFp4MegaMoeConfig:
    """Static dimensions, NVFP4 layout, and push-wire choices."""

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_push_nvfp4"
    nvfp4_mode: Literal["w4a8", "w4a16_rs"] = "w4a8"
    group_size: Literal[32, 64, 128] = 128
    residual_scheme: Literal["generic", "pow2"] = "generic"
    capacity_factor: float = 1.0
    dedup_dispatch: bool = True
    grouped_combine: bool = True
    fuse_act: bool = True
    payload_dtype: Literal["fp8", "bf16"] = "fp8"
    combine_dtype: Literal["fp8", "bf16"] = "fp8"
    rs_n_tactic: Literal[64] = 64
    rs_stages: Literal[3] = 3
    rs_stage_k: Literal[64] = 64
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0


__all__ = ["Sm90PushNvFp4MegaMoeConfig"]
