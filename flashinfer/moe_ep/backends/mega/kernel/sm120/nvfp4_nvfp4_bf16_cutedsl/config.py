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
    # owner_bf16 permits rounding each owner partial before the final reduction.
    numerical_contract: str = "ordinary_topk"

    def __post_init__(self) -> None:
        if self.numerical_contract not in ("ordinary_topk", "owner_bf16"):
            raise ValueError("unknown NVFP4 numerical contract")
        if self.numerical_contract == "ordinary_topk" and (self.knobs or {}).get(
            "rank_local_combine"
        ):
            raise ValueError(
                "rank_local_combine requires owner_bf16 numerical permission"
            )
