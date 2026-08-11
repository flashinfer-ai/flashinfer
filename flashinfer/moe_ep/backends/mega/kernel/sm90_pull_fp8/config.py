"""SM90 (Hopper) pull-style FP8 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm90PullFp8MegaMoeConfig:
    """Kernel params for ``kernel_src.sm90.pull_style_cutedsl_megakernel.hopper_fp8_mega_moe``.

    ``intermediate_size`` is the post-SwiGLU width, matching the SM100 configs
    and SGLang.  The Hopper FP8 kernel's full FC1 gate+up width is derived
    internally as ``2 * intermediate_size``.

    ``fp8_scale_mode`` selects the scale ABI:

    * ``per_tensor``: one static fp32 dequant scale per activation stream and
      one per expert weight (legacy E8M0 SF wire is dispatched but unused by
      the GEMM dequantization).
    * ``blockwise``: DeepGEMM-style fp32 block scales — per token/128-block
      for activations, per 128x128 block for weights.

    ``fc1_activation_dequant_scale`` / ``fc2_activation_dequant_scale`` are
    the per-tensor-mode static calibration scalars (dequant convention:
    ``fp32 ~= fp8_payload * scale``).  They MUST be identical on every EP rank
    — the kernel dequantizes tokens received from peers with the LOCAL rank's
    copy — so derive them from offline calibration, not per-batch amax.
    Ignored in blockwise mode (scales are derived per block at staging /
    preprocess time).

    Expert weights must be kernel-ready FP8 at launch; supply bf16
    ``MoEWeightPack`` and enable ``MegaConfig.preprocess_weights`` (default),
    or pass kernel-ready transformed weights with ``preprocess_weights=False``.

    PORT NOTE: no ``knobs`` field yet — the sm90 tree has no tuner/knob-cache
    module (see the shim's PORT NOTE); geometry knobs (``swap_ab``,
    ``mma_tiler_mnk``) are explicit fields instead.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_pull_fp8"
    kind: Literal["fp8_e4m3", "fp8_e5m2"] = "fp8_e4m3"
    fp8_scale_mode: Literal["per_tensor", "blockwise"] = "per_tensor"
    fp8_accum_mode: Literal["1xacc", "2xacc"] = "1xacc"
    # Kernel geometry: swap_ab selects the swap-A/B specialization;
    # mma_tiler_mnk=None uses the drop driver's default for the geometry
    # ((64, 128, 128) native, (256, 32, 128) swap-AB).
    swap_ab: bool = False
    mma_tiler_mnk: tuple[int, int, int] | None = None
    # Scheduler token-tile assignment; "atomic_counter" is the drop's
    # perf-run setting (run_perf_test.sh), "static" the kernel default.
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    in_kernel_fc2_reduce: bool = False
    token_back_by_dispatch: bool = False
    # Per-tensor static calibration scales (see class docstring).
    fc1_activation_dequant_scale: float = 1.0
    fc2_activation_dequant_scale: float = 1.0
