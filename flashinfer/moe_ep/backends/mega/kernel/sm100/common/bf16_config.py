"""Shared SM100 BF16 CuTeDSL MegaMoE configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm100_Bf16_Cutedsl_MegaMoeConfigBase:
    """Common parameters for SM100 BF16-activation CuTeDSL MegaMoE kernels."""

    intermediate_size: int
    top_k: int
    kernel_name: str = ""
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    # "auto" runs the collective tuner over the current single supported
    # geometry, preserving the autotune contract as more geometries arrive.
    knobs: dict | str | None = None
