"""Configuration for the SM90 push BF16 mega-MoE backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm90PushBf16MegaMoeConfig:
    """Static dimensions and protocol choices for the Hopper BF16 backend."""

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_push_bf16"
    capacity_factor: float = 1.0
    dedup_dispatch: bool = True
    grouped_combine: bool = False
    fuse_fc1_epilogue: bool = False
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0
    wave_schedule: Literal["mono", "serial2", "pipe2"] = "mono"


__all__ = ["Sm90PushBf16MegaMoeConfig"]
