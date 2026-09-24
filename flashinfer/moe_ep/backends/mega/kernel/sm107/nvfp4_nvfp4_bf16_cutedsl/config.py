"""User-facing config for the SM107 (Rubin) nvfp4 block-scaled mega kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple

if TYPE_CHECKING:
    import torch


@dataclass
class Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.sm107.next_cutedsl_megamoe.sm107_block_scaled_mega_moe``.

    The Rubin inference block-scaled swap-AB fused dispatch + FC1 + SwiGLU +
    FC2 + combine mega kernel (``BlockScaledSwapAbMegaMoeKernel``) at quant
    kind nvfp4: nvfp4 activations x nvfp4 weights -> bf16 output, sf_vec_size
    16 (FP8-E4M3 block scales), gate/up interleave 16. SiTU requires both
    positive beta parameters. NVFP4 scaling follows the SM100 interface:
    input_norm_const controls BF16 staging; per-expert alpha tensors correct
    GEMM accumulators, and fc1_norm_const scales intermediate quantization.
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
    # None preserves the fields above; a dict overrides KNOB_KEYS.
    # "cache" uses an offline winner or the heuristic fallback. Online "auto"
    # is unsupported because the kernel fixes its knobs at construction.
    knobs: dict | str | None = None
    max_sm_count: Optional[int] = None
    activation: Literal["swiglu", "situ"] = "swiglu"
    situ_beta: Optional[float] = None
    situ_linear_beta: Optional[float] = None
    input_norm_const: float = 1.0
    fc1_alpha: Optional["torch.Tensor"] = None
    fc2_alpha: Optional["torch.Tensor"] = None
    fc1_norm_const: Optional["torch.Tensor"] = None
