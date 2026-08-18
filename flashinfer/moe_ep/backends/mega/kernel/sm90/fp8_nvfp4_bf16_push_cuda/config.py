"""Configuration for the SM90 FP8/NVFP4/BF16 push-CUDA mega-MoE backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig:
    """Static dimensions, NVFP4 layout, and push-wire choices.

    ``packed`` remains the lowest-memory default (241.5 MiB versus 336.1 MiB
    for folded weights in the measured E8 geometry). ``folded`` is the
    performance recommendation (1.63-3.35x measured speedup), but making it
    the default requires a real-checkpoint logits/perplexity gate. ``dual``
    retains both forms (577.6 MiB measured) and requires explicit residency
    acknowledgement.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_fp8_nvfp4_bf16_push_cuda"
    nvfp4_mode: Literal["w4a8", "w4a16_rs"] = "w4a8"
    weight_policy: Literal["packed", "folded", "hot_folded", "dual"] = "packed"
    hot_expert_count: int = 0
    acknowledge_dual_residency: bool = False
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
    tma_cache_capacity: int = 128
    n64_expected_m_per_sm: float = 4.0
    payload_layout: Literal[3, 4] = 4
    allow_legacy_layout: bool = False
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0


__all__ = ["Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig"]
