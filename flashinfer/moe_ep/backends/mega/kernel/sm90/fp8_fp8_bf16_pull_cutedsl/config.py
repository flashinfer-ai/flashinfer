"""SM90 (Hopper) pull-style FP8 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig:
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

    Launch tuning is resolved through the ``knobs`` field (knob cache /
    heuristic table / explicit dict / ``"auto"`` autotune — see the field
    comment below) or, mutually exclusively, through the explicit geometry
    fields (``swap_ab`` / ``pingpong`` / ``mma_tiler_mnk`` /
    ``cluster_shape_mnk``).
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_fp8_fp8_bf16_pull_cutedsl"
    kind: Literal["fp8_e4m3", "fp8_e5m2"] = "fp8_e4m3"
    fp8_scale_mode: Literal["per_tensor", "blockwise"] = "per_tensor"
    fp8_accum_mode: Literal["1xacc", "2xacc"] = "1xacc"
    # Kernel tuning knobs (see kernel_src...pull_style_cutedsl_megakernel
    # shim/tuner.py).  None -> knob-cache lookup, else the drop's
    # token-bucket heuristic table; a dict applies those knobs; "auto" runs
    # the collective autotune sweep at the first compute (never inside a
    # serving engine — tune offline with python -m flashinfer.moe_ep.tune).
    # Mutually exclusive with the explicit geometry fields below.
    knobs: dict | str | None = None
    # Launch geometry / scheduling: leave ALL of swap_ab / pingpong /
    # mma_tiler_mnk / cluster_shape_mnk as None to use the drop driver's
    # token-bucket heuristics (keyed on fp8_scale_mode and max tokens per
    # rank); setting any one switches to manual mode with the drop driver's
    # defaults for the rest (swap_ab=False, pingpong=False, (64, 128, 128)
    # native / (256, 32, 128) swap-AB / (128, 32, 128) swap-AB ping-pong,
    # cluster (1, 1, 1)).
    swap_ab: bool | None = None
    pingpong: bool | None = None
    mma_tiler_mnk: tuple[int, int, int] | None = None
    cluster_shape_mnk: tuple[int, int, int] | None = None
    # Scheduler token-tile assignment; "atomic_counter" is the drop's
    # perf-run setting (run_perf_test.sh), "static" the kernel default.
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    in_kernel_fc2_reduce: bool = False
    # Legacy alias: True maps to token_back_mode="reuse_dispatch_warps".
    token_back_by_dispatch: bool = False
    # Explicit token-back placement; overrides token_back_by_dispatch when set.
    token_back_mode: (
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"] | None
    ) = None
    # Per-tensor static calibration scales (see class docstring).
    fc1_activation_dequant_scale: float = 1.0
    fc2_activation_dequant_scale: float = 1.0
