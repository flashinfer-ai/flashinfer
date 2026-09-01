"""Configuration for the SM120 CuTeDSL NVFP4 x NVFP4 MegaMoE backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """Static model and tuning parameters.

    ``intermediate_size`` is the post-SwiGLU width. The split kernel receives
    a full gate+up width of ``2 * intermediate_size`` internally.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm120_nvfp4_nvfp4_bf16_cutedsl"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    input_norm_const: float = 1.0
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    knobs: dict | None = None
