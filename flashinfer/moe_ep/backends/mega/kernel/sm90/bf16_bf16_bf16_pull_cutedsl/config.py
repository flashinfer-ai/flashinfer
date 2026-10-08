"""SM90 (Hopper) pull-style BF16 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.sm90.pull_style_cutedsl_megakernel.hopper_bf16_mega_moe``.

    Native BF16 twin of ``Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig``: bf16
    dispatch payload, BF16 WGMMA FC1/FC2 with fp32 accumulation and a bf16
    FC1-output staging buffer, bf16 combine wire.  There is no quantization
    anywhere, so the FP8 config's ``kind`` / ``fp8_scale_mode`` /
    ``fp8_accum_mode`` / dequant-scale fields do not exist here.

    ``intermediate_size`` is the post-SwiGLU width, matching the SM100
    configs and SGLang.  The kernel's full FC1 gate+up width is derived
    internally as ``2 * intermediate_size``.

    Expert weights are canonical bf16 ``MoEWeightPack``; ``preprocess_weights``
    (default on) only re-lays them out (gate/up 8-row interleave + K-major
    permute, no copy of the down-proj weight).  Kernel-ready transformed
    weights can be supplied with ``preprocess_weights=False``.

    Launch tuning is resolved through the ``knobs`` field (knob cache /
    heuristic table / explicit dict / ``"auto"`` autotune) or, mutually
    exclusively, through the explicit geometry fields (``swap_ab`` /
    ``pingpong`` / ``mma_tiler_mnk`` / ``cluster_shape_mnk``).
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_bf16_bf16_bf16_pull_cutedsl"
    # Kernel tuning knobs (see kernel_src...pull_style_cutedsl_megakernel
    # shim/tuner_bf16.py).  None -> knob-cache lookup, else the drop's
    # token-bucket heuristic table; a dict applies those knobs; "auto" runs
    # the collective autotune sweep at the first compute (never inside a
    # serving engine — tune offline with python -m flashinfer.moe_ep.tune).
    # Mutually exclusive with the explicit geometry fields below.
    knobs: dict | str | None = None
    # Launch geometry / scheduling: leave ALL of swap_ab / pingpong /
    # mma_tiler_mnk / cluster_shape_mnk as None to use the drop driver's
    # token-bucket heuristics (keyed on max tokens per rank); setting any
    # one switches to manual mode with the drop driver's defaults for the
    # rest (swap_ab=False, pingpong=False, (64, 128, 64) native /
    # (256, 32, 64) swap-AB / (128, 32, 64) swap-AB ping-pong, cluster
    # (1, 1, 1)).  Tile K is 64 for two-byte operands.
    swap_ab: bool | None = None
    pingpong: bool | None = None
    mma_tiler_mnk: tuple[int, int, int] | None = None
    cluster_shape_mnk: tuple[int, int, int] | None = None
    # Scheduler token-tile assignment; "atomic_counter" is the drop's
    # perf-run setting (run_perf_test.sh), "static" the kernel default.
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    # Accepted for API parity with the FP8 config; no effect on the BF16 path.
    fast_math: bool = True
    enable_in_kernel_fc2_reduce: bool = False
    # Legacy alias: True maps to token_back_mode="reuse_dispatch_warps".
    token_back_by_dispatch: bool = False
    # Explicit token-back placement; overrides token_back_by_dispatch when set.
    token_back_mode: (
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"] | None
    ) = None
    # How many of the 4 dispatch warps do token-comm work at all (1/2/4);
    # the rest stay fully idle.  Output-invariant work partitioning.
    active_dispatch_warps: int = 1
    # Empty-warp FC1 store offload (self-gating to non-swap non-ping-pong
    # with register headroom; falls back to early fc1_done publication).
    fc1_store_offload: bool = True
    fc1_early_done_publish: bool = False
    # Fold TMA-A/TMA-B/scheduler into the three idle dispatch-warpgroup slots
    # and drop the producer warpgroup (needs active_dispatch_warps == 1).
    fold_producer_warps: bool = True
    # Training forward: keep the raw pre-SwiGLU fc1 gate+up (BF16, expert-major
    # pool with 128-row expert segments) -- read it back from the workspace's
    # ``fc1_c`` after compute().  Default off (compiled out).
    generate_c: bool = False
    # Tail-split pair tasks for the odd tail cluster block of an expert.  None
    # follows the heuristic table's per-bucket choice; True/False forces it.
    # Legal only with swap-AB cga (1, 2, 1) or non-swap cga (2, 1, 1).
    tail_split_pairs: bool | None = None
