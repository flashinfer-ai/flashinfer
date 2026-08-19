"""User-facing config for the SM107 (Rubin) mxfp8 block-scaled mega kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

Sm107Mxfp8Kind = Literal["mxfp8_e4m3", "mxfp8_e5m2"]


@dataclass
class Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.sm107.next_cutedsl_megamoe.sm107_block_scaled_mega_moe``.

    The Rubin inference block-scaled swap-AB fused dispatch + FC1 + SwiGLU +
    FC2 + combine mega kernel (``BlockScaledSwapAbMegaMoeKernel``) at quant
    kind mxfp8: mxfp8 activations x mxfp8 weights -> bf16 output, sf_vec_size
    32, gate/up interleave 16.
    """

    intermediate_size: int  # post-SwiGLU width; FC1 GEMM N is 2*intermediate_size
    top_k: int
    kernel_name: str = "sm107_mxfp8_mxfp8_bf16_cutedsl"
    kind: Sm107Mxfp8Kind = "mxfp8_e4m3"
    gate_up_clamp: Optional[float] = None
    activation_clamp: Optional[float] = None  # deprecated alias of gate_up_clamp
    fast_math: bool = True  # accepted for mega API parity; no kernel toggle
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    apply_topk_in_fc1: bool = True
    schedule_policy: Tuple[str, Optional[int]] = ("grouped", None)
    work_id_mode: Literal["grid_stride", "atomic_counter"] = "grid_stride"
    fc2_use_bulk: bool = False
    fc2_tma_stages: Optional[int] = None
    epi_flag_batches: Tuple[int, int] = (4, 2)
    token_in_flag_batch: int = 1
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None  # None -> (256, 128, 128)
    cluster_shape_mn: Optional[Tuple[int, int]] = None  # None -> (2, 1)
    # Mixed-CGA launch: fill leftover SMs with smaller fallback clusters
    # (e.g. preferred (4, 1) + fallback (2, 1)). None -> uniform launch.
    fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None
    # Tuning-knob resolution: None -> the explicit fields above stand;
    # a dict (shim knob keys, see kernel_src.sm107.next_cutedsl_megamoe.KNOB_KEYS)
    # -> explicit overrides; "cache" -> knob-cache lookup (populated by
    # `python -m flashinfer.moe_ep.tune`) with the built-in heuristic
    # fallback. The SM100-style online "auto" sweep is NOT supported on the
    # engine path (the SM107 session bakes knobs at construction).
    knobs: dict | str | None = None
    max_sm_count: Optional[int] = None
