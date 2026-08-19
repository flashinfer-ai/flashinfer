"""User-facing config for the SM107 (Rubin) nvfp4 block-scaled mega kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass
class Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.sm107.next_cutedsl_megamoe.sm107_block_scaled_mega_moe``.

    The Rubin inference block-scaled swap-AB fused dispatch + FC1 + SwiGLU +
    FC2 + combine mega kernel (``BlockScaledSwapAbMegaMoeKernel``) at quant
    kind nvfp4: nvfp4 activations x nvfp4 weights -> bf16 output, sf_vec_size
    16 (FP8-E4M3 block scales), gate/up interleave 16.  The per-expert
    fc1_alpha / fc2_alpha / fc1_norm_const dequant scalars are identically 1
    (weights and activations quantize with norm_const=1.0), so they are
    omitted from the kernel ABI.
    """

    intermediate_size: int  # post-SwiGLU width; FC1 GEMM N is 2*intermediate_size
    top_k: int
    kernel_name: str = "sm107_nvfp4_nvfp4_bf16_cutedsl"
    gate_up_clamp: Optional[float] = None
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
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None  # None -> (256, 128, 256)
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
