"""Configuration for the SM90 push FP8 mega-MoE backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm90PushFp8MegaMoeConfig:
    """Static dimensions and protocol choices for the Hopper FP8 backend."""

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_push_fp8"
    capacity_factor: float = 1.0
    dedup_dispatch: bool = True
    grouped_combine: bool = True
    fuse_fc1_epilogue: bool = False
    payload_dtype: Literal["fp8", "bf16"] = "fp8"
    combine_dtype: Literal["fp8", "bf16"] = "fp8"
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0
