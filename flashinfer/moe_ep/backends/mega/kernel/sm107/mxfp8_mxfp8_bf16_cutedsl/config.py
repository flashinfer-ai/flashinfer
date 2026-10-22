"""User-facing config for the SM107 (Rubin) mxfp8 block-scaled mega kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

Sm107Mxfp8Kind = Literal["mxfp8_e4m3", "mxfp8_e5m2"]


@dataclass
class Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig:
    """Configure Rubin inference with MXFP8 E4M3/E5M2 activations and weights.

    The kernel fuses dispatch, FC1, activation, FC2, and combine with BF16 output.
    It uses E8M0 scales per 32 values and 16-row gate/up stripes. SwiGLU is the default;
    SiTU requires both positive, finite beta parameters.
    """

    intermediate_size: int  # width after activation; FC1 N is 2*intermediate_size
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
    # None preserves the fields above; a dict overrides KNOB_KEYS.
    # "cache" uses an offline winner or the heuristic fallback. Online "auto"
    # is unsupported because the kernel fixes its knobs at construction.
    knobs: dict | str | None = None
    max_sm_count: Optional[int] = None
    activation: Literal["swiglu", "situ"] = "swiglu"
    situ_beta: Optional[float] = None
    situ_linear_beta: Optional[float] = None
    # GenPhase requires cluster (4, 1), fc2_use_bulk=True, and <=1024 tokens/rank.
    kernel_variant: Literal["inference", "genphase"] = "inference"
    # Quantizes the FC2 return payload; the operator still returns BF16.
    combine_dtype: Literal["bf16", "nvfp4", "mxfp8"] = "bf16"
