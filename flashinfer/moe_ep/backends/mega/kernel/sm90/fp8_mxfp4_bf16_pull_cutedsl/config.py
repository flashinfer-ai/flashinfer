"""SM90 FP8-activation / MXFP4-weight pull-style MegaMoE config.

This config describes the fixed numerical contract of the production Hopper
Humming path. Keeping the format invariants here prevents an ordinary FP8
tactic from being selected as a silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    normalize_sm90_routing_profile,
)


@dataclass
class Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig:
    """Configuration for packed E2M1 MXFP4 weights and E4M3 activations.

    ``intermediate_size`` is the post-SwiGLU width; canonical FC1 weights have
    ``2 * intermediate_size`` rows (gate followed by up).  Humming's exponent
    range, fold, and epilogue compensation are part of the weight ABI rather
    than tunable kernel tactics.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_fp8_mxfp4_bf16_pull_cutedsl"

    # Fixed numerical/format contract.  Literal annotations document it and
    # __post_init__ enforces it at runtime (dataclasses do not enforce Literal).
    kind: Literal["fp8_e4m3"] = "fp8_e4m3"
    fp8_scale_mode: Literal["mxfp4_hybrid"] = "mxfp4_hybrid"
    fp8_accum_mode: Literal["1xacc"] = "1xacc"
    humming_max_range: Literal[11] = 11
    preprocess_expert_chunk_size: int = 4

    # None means cache lookup then heuristic; "auto" runs the bounded
    # collective union; a dict is a complete fused frontend tactic.
    knobs: dict | str | None = None
    swap_ab: bool | None = None
    pingpong: bool | None = None
    mma_tiler_mnk: tuple[int, int, int] | None = None
    cluster_shape_mnk: tuple[int, int, int] | None = None
    # ``None`` means the cache/heuristic owns this axis. Supplying any one of
    # these six execution axes selects the shim's cache-free manual defaults;
    # this preserves the distinction between an omitted selector and an
    # explicit value equal to the manual default.
    load_balance_mode: Literal["static", "atomic_counter"] | None = None
    # Canonical MXFP4 tuning/benchmark contract. Callers may pass None
    # explicitly to disable it or fall back to the legacy activation alias.
    gate_up_clamp: float | None = 10.0
    activation_clamp: float | None = None
    fast_math: bool = True
    enable_in_kernel_fc2_reduce: Literal[False] = False
    token_back_mode: (
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"] | None
    ) = None
    # Latest PR4688 communication/warp-layout knobs. The production MXFP4
    # profile keeps BF16 combine; grouped/quantized combine remains outside
    # this correctness contract.
    dedup_dispatch: bool | None = None
    grouped_token_back: Literal[False] = False
    combine_format: Literal["bf16"] = "bf16"
    active_dispatch_warps: int | None = None
    fc1_store_offload: bool | None = None
    fc1_early_done_publish: bool | None = None
    fold_producer_warps: bool | None = None
    routing_profile: str = field(
        default=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        self.routing_profile = normalize_sm90_routing_profile(self.routing_profile)
        if self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {self.intermediate_size}"
            )
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.kind != "fp8_e4m3":
            raise ValueError(
                f"SM90 MXFP4 x FP8 Humming requires kind='fp8_e4m3'; got {self.kind!r}"
            )
        if self.fp8_scale_mode != "mxfp4_hybrid":
            raise ValueError(
                "SM90 MXFP4 x FP8 requires fp8_scale_mode='mxfp4_hybrid'; "
                f"got {self.fp8_scale_mode!r}"
            )
        if self.fp8_accum_mode != "1xacc":
            raise ValueError(
                "SM90 MXFP4 x FP8 currently requires fp8_accum_mode='1xacc'; "
                f"got {self.fp8_accum_mode!r}"
            )
        if self.humming_max_range != 11:
            raise ValueError(
                "the production Humming ABI fixes humming_max_range=11; "
                f"got {self.humming_max_range}"
            )
        if self.preprocess_expert_chunk_size <= 0:
            raise ValueError(
                "preprocess_expert_chunk_size must be positive, got "
                f"{self.preprocess_expert_chunk_size}"
            )
        for field_name in ("swap_ab", "pingpong"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a bool when set")
        if self.swap_ab is False:
            raise ValueError("SM90 MXFP4 x FP8 supports only the swap-AB kernel")
        if not isinstance(self.enable_in_kernel_fc2_reduce, bool):
            raise ValueError("enable_in_kernel_fc2_reduce must be a bool")
        if self.enable_in_kernel_fc2_reduce:
            raise ValueError(
                "SM90 MXFP4 x FP8 currently requires standalone top-k reduce"
            )
        if self.load_balance_mode is not None and self.load_balance_mode not in (
            "static",
            "atomic_counter",
        ):
            raise ValueError(
                "load_balance_mode must be 'static' or 'atomic_counter', got "
                f"{self.load_balance_mode!r}"
            )
        for field_name in (
            "dedup_dispatch",
            "fc1_store_offload",
            "fc1_early_done_publish",
            "fold_producer_warps",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a bool when set")
        if not isinstance(self.grouped_token_back, bool):
            raise ValueError("grouped_token_back must be a bool")
        if self.grouped_token_back:
            raise ValueError(
                "the production MXFP4 BF16-combine profile does not support "
                "grouped_token_back"
            )
        if self.combine_format != "bf16":
            raise ValueError(
                "the production MXFP4 profile requires combine_format='bf16'"
            )
        if self.active_dispatch_warps is not None and (
            isinstance(self.active_dispatch_warps, bool)
            or self.active_dispatch_warps not in (1, 2, 4)
        ):
            raise ValueError("active_dispatch_warps must be 1, 2, or 4")
        if (
            self.fold_producer_warps is True
            and self.active_dispatch_warps is not None
            and self.active_dispatch_warps != 1
        ):
            raise ValueError(
                "fold_producer_warps=True requires active_dispatch_warps=1"
            )
        if self.knobs is not None and not (
            isinstance(self.knobs, dict) or self.knobs == "auto"
        ):
            raise ValueError(
                f"knobs must be None, a dict, or 'auto'; got {self.knobs!r}"
            )
        manual_fused_selector = any(
            value is not None
            for value in (
                self.swap_ab,
                self.pingpong,
                self.mma_tiler_mnk,
                self.cluster_shape_mnk,
                self.load_balance_mode,
                self.dedup_dispatch,
                self.active_dispatch_warps,
                self.fc1_store_offload,
                self.fc1_early_done_publish,
                self.fold_producer_warps,
            )
        )
        if self.knobs is not None and manual_fused_selector:
            raise ValueError(
                "knobs= is mutually exclusive with explicit fused geometry or "
                "execution axes"
            )


__all__ = ["Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig"]
