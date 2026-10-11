"""SM90 (Hopper) pull-style BF16 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal


@dataclass
class Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig:
    """Native BF16 expert compute on the SM90 pull-style CuTeDSL megakernel.

    The same fused NVSHMEM dispatch + FC1 + SwiGLU + FC2 + combine kernel as
    :class:`Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig`, compiled with BF16
    operands: BF16 activations and expert weights, BF16 WGMMA with FP32
    accumulation, a BF16 FC1 output (no FP8 anywhere, so no calibration
    scales).  Tokens cross NVLink as BF16, twice the FP8 payload.

    ``intermediate_size`` is the post-SwiGLU width.  The geometry,
    scheduling and combine fields mean the same as on the FP8 config; launch
    tuning resolves through ``knobs`` (knob cache / BF16 token-bucket
    heuristic table / explicit dict / ``"auto"``) or, mutually exclusively,
    the explicit geometry fields.  BF16 doubles the A/B SMEM per pipeline
    stage, so the largest FP8 tiles (e.g. swap-AB M256xN128) do not fit and
    are rejected at compile time; BF16 also accepts K=64 tiles
    (``mma_tiler_mnk=(M, N, 64)``), which halve the stage and keep the
    pipeline deep (the large-token heuristic rows use them).
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_bf16_bf16_bf16_pull_cutedsl"
    knobs: dict | str | None = None
    swap_ab: bool | None = None
    pingpong: bool | None = None
    mma_tiler_mnk: tuple[int, int, int] | None = None
    cluster_shape_mnk: tuple[int, int, int] | None = None
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    enable_in_kernel_fc2_reduce: bool = False
    token_back_by_dispatch: bool = False
    token_back_mode: (
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"] | None
    ) = None
    # COLLECTIVE (must match on every EP rank), as on the FP8 config.
    dedup_dispatch: bool = False
    grouped_token_back: bool = False
    combine_format: Literal["bf16", "32e4m3xe8m0", "32e5m2xe8m0"] = "bf16"
    active_dispatch_warps: int = 1
    compact_pull_buffer: bool = True
    fc1_store_offload: bool = True
    fc1_early_done_publish: bool = False
    fold_producer_warps: bool = True
    generate_c: bool = False
    tail_split_pairs: bool | None = None

    # Fixed by the format.  The kernel runs its per-tensor path with unit
    # dequant scales; the shared pull-style backend plumbing reads these.
    kind: ClassVar[str] = "bf16"
    weight_format: ClassVar[str] = "dense"
    fp8_scale_mode: ClassVar[str] = "per_tensor"
    fp8_accum_mode: ClassVar[str] = "1xacc"
    fc1_activation_dequant_scale: ClassVar[float] = 1.0
    fc2_activation_dequant_scale: ClassVar[float] = 1.0
