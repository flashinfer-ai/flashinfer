"""User-facing config for the SM107 (Rubin) mxfp8 GLU fprop mega kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

Sm107Mxfp8Kind = Literal["mxfp8_e4m3", "mxfp8_e5m2"]


@dataclass
class Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.next_cutedsl_megamoe.sm107_mxfp8_glu_mega_moe``.

    The Rubin training-forward (fprop) fused dispatch + FC1 + SwiGLU + FC2 +
    combine mega kernel: mxfp8 activations x mxfp8 weights -> bf16 output,
    sf_vec_size 32, gate/up interleave 32.
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
    group_hint: Optional[int] = 768  # fc1->fc2 scheduler lead; None/<=0 -> HW clusters
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None  # None -> (256, 256, 128)
    cluster_shape_mnk: Optional[Tuple[int, int, int]] = None  # None -> (2, 1, 1)
    max_sm_count: Optional[int] = None
