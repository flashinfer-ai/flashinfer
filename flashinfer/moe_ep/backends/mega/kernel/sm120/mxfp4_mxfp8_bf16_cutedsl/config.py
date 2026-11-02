"""Configuration for the SM120 CuTeDSL MXFP4 x MXFP8 MegaMoE backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig:
    """Static model and tuning parameters.

    ``intermediate_size`` is the post-SwiGLU width. The split kernel receives
    a full gate+up width of ``2 * intermediate_size`` internally.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm120_mxfp4_mxfp8_bf16_cutedsl"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    knobs: dict | None = None
