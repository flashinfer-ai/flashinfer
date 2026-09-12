"""SM120 swap-AB CuTeDSL MXFP8 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm120_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.sm120.swapab_cutedsl_megakernel.sm120_mxfp8_mega_moe``.

    ``intermediate_size`` is the post-SwiGLU width, matching the other trees
    and SGLang. The kernel's full FC1 gate+up width is derived internally as
    ``2 * intermediate_size``.

    Expert weights must be MXFP8 at kernel launch; supply bf16 ``MoEWeightPack``
    and enable ``MegaConfig.preprocess_weights`` (default), or pass pre-quantized
    fp8 weights with ``w13_scale`` / ``w2_scale``.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm120_mxfp8_mxfp8_bf16_cutedsl"
    # The SM120 kernel hard-codes E4M3 (no e5m2 selection at the mega level).
    kind: Literal["mxfp8_e4m3"] = "mxfp8_e4m3"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    in_kernel_fc2_reduce: bool = False
    # Where the cross-rank fc2 push-back runs (this drop's native enum, not
    # the sm100 token_back_by_dispatch bool): "epi_warps" (epilogue STG
    # redirect), "standalone_warps" (dedicated warps 12-15), or
    # "reuse_dispatch_warps" (dispatch warps 8-11 push after dispatch_pull).
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    # Kernel tuning knobs: a dict of shim-config field overrides (e.g.
    # mma_tiler_mnk, cluster_shape_mnk, flag_batch, dispatch_pull_mode).
    # None -> kernel defaults. This tree has no knob cache / autotune yet, so
    # "auto" is not supported here.
    knobs: dict | None = None
