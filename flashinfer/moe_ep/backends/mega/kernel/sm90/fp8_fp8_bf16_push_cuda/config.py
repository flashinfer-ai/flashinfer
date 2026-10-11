"""Configuration for the SM90 push FP8 mega-MoE backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig:
    """Static dimensions and protocol choices for the Hopper FP8 backend.

    Weights: canonical bf16 ``MoEWeightPack`` or an MXFP8 checkpoint (E4M3 +
    E8M0 per-32 scales); both are quantized/converted once at preprocess to
    128x128 FP8 block scales (MXFP8 exactly, except for E4M3 underflow).
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_fp8_fp8_bf16_push_cuda"
    capacity_factor: float = 1.0
    dedup_dispatch: bool = True
    grouped_combine: bool = True
    fuse_fc1_epilogue: bool = False
    payload_dtype: Literal["fp8", "bf16"] = "fp8"
    combine_dtype: Literal["fp8", "bf16"] = "fp8"
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0
