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


# Keep this allow-list in lockstep with the MXFP4 shim's strict knob validator.
# It intentionally excludes rank/shape/format fields even though the generic
# FP8 ``with_knobs`` helper would otherwise accept every dataclass field.
_SUPPORTED_FUSED_MXFP4_KNOBS = frozenset(
    {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
        "group_hint",
        "flag_batch",
        "epi_flag_batch",
        "load_balance_mode",
        "token_back_mode",
        "in_kernel_fc2_reduce",
        "clc_bundle_size",
        "num_sched_stages",
    }
)

# Split tactics are complete session identities, not partial fused frontend
# updates. Keep this exact allow-list in lockstep with shim/mxfp4_tuner.py.
_SUPPORTED_SPLIT_MXFP4_KNOBS = frozenset(
    {
        "k1_mma_tiler_mnk",
        "k2_mma_tiler_mnk",
        "k1_cluster_shape_mnk",
        "k2_cluster_shape_mnk",
        "k1_group_hint",
        "k2_group_hint",
        "k1_num_sched_stages",
        "k2_num_sched_stages",
        "k1_sm_count",
        "k2_sm_count",
        "counter_epoch_banks",
        "graph_variant",
        "enable_iket",
    }
)


# Keep this allow-list in lockstep with SplitMegaPlan. Reject unsupported
# Hopper cluster geometry before allocating a split session or entering JIT.
_SUPPORTED_SPLIT_CLUSTER_SHAPES = frozenset(
    {
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
    }
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

    # Split is a purpose-built Green Context graph session, not two sequential
    # calls to the fused frontend. Its K1/K2 geometry is a separate identity.
    execution_mode: Literal["fused", "split"] = "fused"
    split_k1_mma_tiler_mnk: tuple[int, int, int] = (128, 32, 128)
    split_k2_mma_tiler_mnk: tuple[int, int, int] = (128, 32, 128)
    split_k1_cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1)
    split_k2_cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1)
    split_k1_group_hint: int | None = None
    split_k2_group_hint: int | None = None
    split_k1_num_sched_stages: int | None = None
    split_k2_num_sched_stages: int | None = None
    split_k1_sm_count: int | None = None
    split_k2_sm_count: int | None = None
    split_counter_epoch_banks: Literal[1, 2] = 1
    split_graph_variant: Literal["cold_k0", "steady_k3_reset"] = "steady_k3_reset"
    split_enable_iket: bool = False

    # Mode-specific tactic selector. None means dedicated cache lookup then
    # per-token heuristic; "auto" runs the bounded collective union; a dict is
    # a fused frontend tactic or a complete split session tactic according to
    # execution_mode. Legacy explicit split_* fields remain supported when
    # both split SM counts are provided and knobs is None.
    knobs: dict | str | None = None
    swap_ab: bool | None = None
    pingpong: bool | None = None
    mma_tiler_mnk: tuple[int, int, int] | None = None
    cluster_shape_mnk: tuple[int, int, int] | None = None
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    in_kernel_fc2_reduce: Literal[False] = False
    token_back_mode: (
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"] | None
    ) = None
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
        if self.execution_mode not in ("fused", "split"):
            raise ValueError(
                "execution_mode must be 'fused' or 'split', got "
                f"{self.execution_mode!r}"
            )
        if isinstance(
            self.split_counter_epoch_banks, bool
        ) or self.split_counter_epoch_banks not in (1, 2):
            raise ValueError(
                "split_counter_epoch_banks must be 1 or 2, got "
                f"{self.split_counter_epoch_banks!r}"
            )
        if self.split_graph_variant not in ("cold_k0", "steady_k3_reset"):
            raise ValueError(
                "split_graph_variant must be 'cold_k0' or "
                f"'steady_k3_reset', got {self.split_graph_variant!r}"
            )
        for name, tiler in (
            ("split_k1_mma_tiler_mnk", self.split_k1_mma_tiler_mnk),
            ("split_k2_mma_tiler_mnk", self.split_k2_mma_tiler_mnk),
        ):
            if not isinstance(tiler, tuple) or len(tiler) != 3:
                raise ValueError(f"{name} must be an M/N/K triple, got {tiler!r}")
            tile_m, token_n, tile_k = tiler
            if tile_m not in (128, 256) or token_n not in (16, 32, 64, 128):
                raise ValueError(
                    f"{name} requires M in (128,256) and token N in "
                    f"(16,32,64,128), got {tiler!r}"
                )
            if tile_k not in (128, 256):
                raise ValueError(f"{name} requires K in (128,256), got {tiler!r}")

        if (
            self.execution_mode == "split"
            and self.intermediate_size % self.split_k2_mma_tiler_mnk[2]
        ):
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) must be divisible "
                "by split_k2_mma_tiler_mnk K="
                f"{self.split_k2_mma_tiler_mnk[2]}"
            )
        for name, cluster in (
            ("split_k1_cluster_shape_mnk", self.split_k1_cluster_shape_mnk),
            ("split_k2_cluster_shape_mnk", self.split_k2_cluster_shape_mnk),
        ):
            if not isinstance(cluster, tuple) or len(cluster) != 3 or cluster[2] != 1:
                raise ValueError(f"{name} must be an M/N/1 triple, got {cluster!r}")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in cluster
            ):
                raise ValueError(
                    f"{name} entries must be positive integers, got {cluster!r}"
                )
            if (
                self.execution_mode == "split"
                and cluster not in _SUPPORTED_SPLIT_CLUSTER_SHAPES
            ):
                raise ValueError(
                    f"{name} has unsupported Hopper split cluster shape {cluster!r}"
                )
        if self.split_k1_cluster_shape_mnk != self.split_k2_cluster_shape_mnk:
            raise ValueError(
                "K1 and K2 must use the same cluster shape for the byte-exact "
                "split workspace contract"
            )
        if self.split_k1_mma_tiler_mnk[1] != self.split_k2_mma_tiler_mnk[
            1
        ] and self.split_k1_cluster_shape_mnk != (1, 1, 1):
            raise ValueError(
                "independent K1/K2 token-N tiles require split cluster shape (1,1,1)"
            )
        for name, value in (
            ("split_k1_group_hint", self.split_k1_group_hint),
            ("split_k2_group_hint", self.split_k2_group_hint),
            ("split_k1_num_sched_stages", self.split_k1_num_sched_stages),
            ("split_k2_num_sched_stages", self.split_k2_num_sched_stages),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.swap_ab is False:
            raise ValueError("SM90 MXFP4 x FP8 supports only the swap-AB kernel")
        if self.in_kernel_fc2_reduce:
            raise ValueError(
                "SM90 MXFP4 x FP8 currently requires standalone top-k reduce"
            )
        if self.load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                "load_balance_mode must be 'static' or 'atomic_counter', got "
                f"{self.load_balance_mode!r}"
            )
        if self.knobs is not None and not (
            isinstance(self.knobs, dict) or self.knobs == "auto"
        ):
            raise ValueError(
                f"knobs must be None, a dict, or 'auto'; got {self.knobs!r}"
            )

        if self.execution_mode == "split":
            if self.pingpong:
                raise ValueError("split MXFP4 does not support fused ping-pong mode")
            if self.mma_tiler_mnk is not None or self.cluster_shape_mnk is not None:
                raise ValueError(
                    "split execution cannot consume fused MMA/cluster geometry"
                )
            if self.load_balance_mode != "static":
                raise ValueError(
                    "concurrent split K1/K2 requires load_balance_mode='static'"
                )
            if self.token_back_mode not in (None, "epi_warps"):
                raise ValueError(
                    "split K2 performs direct epilogue combine and requires "
                    "token_back_mode='epi_warps'"
                )
            if (
                self.split_counter_epoch_banks == 2
                and self.split_graph_variant != "steady_k3_reset"
            ):
                raise ValueError(
                    "two split counter banks require the steady graph variant"
                )

            # A complete split knobs dict (or auto) owns every K1/K2/session
            # axis. Legacy explicit split_* fields remain a separate selector;
            # fail on mixed representations instead of silently ignoring one.
            if self.knobs is not None:
                manual_split = (
                    self.split_k1_mma_tiler_mnk != (128, 32, 128)
                    or self.split_k2_mma_tiler_mnk != (128, 32, 128)
                    or self.split_k1_cluster_shape_mnk != (1, 1, 1)
                    or self.split_k2_cluster_shape_mnk != (1, 1, 1)
                    or self.split_k1_group_hint is not None
                    or self.split_k2_group_hint is not None
                    or self.split_k1_num_sched_stages is not None
                    or self.split_k2_num_sched_stages is not None
                    or self.split_k1_sm_count is not None
                    or self.split_k2_sm_count is not None
                    or self.split_counter_epoch_banks != 1
                    or self.split_graph_variant != "steady_k3_reset"
                    or self.split_enable_iket
                )
                if manual_split:
                    raise ValueError(
                        "split knobs= is mutually exclusive with explicit split_* "
                        "tactic fields"
                    )
                if isinstance(self.knobs, dict):
                    fields = set(self.knobs)
                    unknown = fields.difference(_SUPPORTED_SPLIT_MXFP4_KNOBS)
                    missing = _SUPPORTED_SPLIT_MXFP4_KNOBS.difference(fields)
                    if unknown or missing:
                        detail = []
                        if unknown:
                            detail.append(
                                "unknown=" + ",".join(sorted(map(str, unknown)))
                            )
                        if missing:
                            detail.append(
                                "missing=" + ",".join(sorted(map(str, missing)))
                            )
                        raise ValueError(
                            "explicit split MXFP4 knobs must be one complete "
                            "split session tactic (" + "; ".join(detail) + ")"
                        )
            else:
                # None + no partition means cache lookup then per-token split
                # heuristic. Supplying either SM count selects the legacy
                # explicit field representation and therefore requires both.
                counts = (self.split_k1_sm_count, self.split_k2_sm_count)
                if (counts[0] is None) != (counts[1] is None):
                    missing_sm_count = (
                        "split_k1_sm_count"
                        if counts[0] is None
                        else "split_k2_sm_count"
                    )
                    raise ValueError(f"{missing_sm_count} must be a positive integer")
                if counts[0] is None:
                    partial_fields = [
                        name
                        for name, specified in (
                            (
                                "split_k1_mma_tiler_mnk",
                                self.split_k1_mma_tiler_mnk != (128, 32, 128),
                            ),
                            (
                                "split_k2_mma_tiler_mnk",
                                self.split_k2_mma_tiler_mnk != (128, 32, 128),
                            ),
                            (
                                "split_k1_cluster_shape_mnk",
                                self.split_k1_cluster_shape_mnk != (1, 1, 1),
                            ),
                            (
                                "split_k2_cluster_shape_mnk",
                                self.split_k2_cluster_shape_mnk != (1, 1, 1),
                            ),
                            (
                                "split_k1_group_hint",
                                self.split_k1_group_hint is not None,
                            ),
                            (
                                "split_k2_group_hint",
                                self.split_k2_group_hint is not None,
                            ),
                            (
                                "split_k1_num_sched_stages",
                                self.split_k1_num_sched_stages is not None,
                            ),
                            (
                                "split_k2_num_sched_stages",
                                self.split_k2_num_sched_stages is not None,
                            ),
                            (
                                "split_counter_epoch_banks",
                                self.split_counter_epoch_banks != 1,
                            ),
                            (
                                "split_graph_variant",
                                self.split_graph_variant != "steady_k3_reset",
                            ),
                            ("split_enable_iket", self.split_enable_iket),
                        )
                        if specified
                    ]
                    if partial_fields:
                        raise ValueError(
                            "partial explicit split_* tactic fields require both "
                            "split_k1_sm_count and split_k2_sm_count; got "
                            + ", ".join(partial_fields)
                        )
                else:
                    cluster_size = (
                        self.split_k1_cluster_shape_mnk[0]
                        * self.split_k1_cluster_shape_mnk[1]
                    )
                    for name, sm_count in (
                        ("split_k1_sm_count", counts[0]),
                        ("split_k2_sm_count", counts[1]),
                    ):
                        if (
                            isinstance(sm_count, bool)
                            or not isinstance(sm_count, int)
                            or sm_count <= 0
                        ):
                            raise ValueError(f"{name} must be a positive integer")
                        if sm_count % cluster_size:
                            raise ValueError(
                                f"{name}={sm_count} must be divisible by cluster "
                                f"size {cluster_size}"
                            )
        elif self.knobs is not None:
            if isinstance(self.knobs, dict):
                unknown = set(self.knobs).difference(_SUPPORTED_FUSED_MXFP4_KNOBS)
                if unknown:
                    rendered = ", ".join(sorted(map(repr, unknown)))
                    raise ValueError(
                        f"unsupported MXFP4 knob field(s): {rendered}; refusing "
                        "to silently ignore non-tactic fields"
                    )
                if self.knobs.get("swap_ab") is not True:
                    raise ValueError(
                        "explicit MXFP4 knobs must include swap_ab=True; native "
                        "A/B fallback is unsupported"
                    )
                if self.knobs.get("fp8_accum_mode", "1xacc") != "1xacc":
                    raise ValueError(
                        "explicit MXFP4 knobs require fp8_accum_mode='1xacc'"
                    )
                if self.knobs.get("in_kernel_fc2_reduce", False) is not False:
                    raise ValueError(
                        "explicit MXFP4 knobs require standalone top-k reduce"
                    )
            if any(
                value is not None
                for value in (
                    self.swap_ab,
                    self.pingpong,
                    self.mma_tiler_mnk,
                    self.cluster_shape_mnk,
                )
            ):
                raise ValueError(
                    "knobs= is mutually exclusive with explicit launch geometry"
                )

    @property
    def split_handoff_token_n(self) -> int:
        return max(
            self.split_k1_mma_tiler_mnk[1],
            self.split_k2_mma_tiler_mnk[1],
        )

    @property
    def split_workspace_counter_tile_tokens(self) -> int:
        return (
            min(
                self.split_k1_mma_tiler_mnk[1],
                self.split_k2_mma_tiler_mnk[1],
            )
            * self.split_k1_cluster_shape_mnk[1]
        )

    @property
    def split_k1_max_active_clusters(self) -> int | None:
        if self.split_k1_sm_count is None:
            return None
        cluster_size = (
            self.split_k1_cluster_shape_mnk[0] * self.split_k1_cluster_shape_mnk[1]
        )
        return self.split_k1_sm_count // cluster_size

    @property
    def split_k2_max_active_clusters(self) -> int | None:
        if self.split_k2_sm_count is None:
            return None
        cluster_size = (
            self.split_k2_cluster_shape_mnk[0] * self.split_k2_cluster_shape_mnk[1]
        )
        return self.split_k2_sm_count // cluster_size


__all__ = ["Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig"]
