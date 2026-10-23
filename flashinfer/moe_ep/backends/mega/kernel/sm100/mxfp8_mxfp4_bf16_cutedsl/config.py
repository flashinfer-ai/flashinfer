# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and tactic schema for native MXFP8 x MXFP4 MegaMoE."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypedDict, cast

from flashinfer.fused_moe.api import QuantConfig, QuantFormat


KERNEL_NAME = "sm100_mxfp8_mxfp4_bf16_cutedsl"
# The pair tile uses eight epilogue warps from this many tokens per rank.
EIGHT_WARP_EPILOGUE_MIN_TOKENS = 256
# The top-k reduction runs in the kernel tail below this many tokens per rank.
IN_KERNEL_REDUCE_MAX_TOKENS = 256
LoadBalanceMode = Literal["static", "atomic_counter", "alternating"]
# The pair tile alternates FC1/FC2 claims from this many tokens per rank,
# the same band that selects the N192 token tile.
ALTERNATING_CLAIM_MIN_TOKENS = 512
# The pair tile claims tiles atomically below this many tokens per rank.
ATOMIC_CLAIM_MAX_TOKENS = 256


class Mxfp8Mxfp4Tactic(TypedDict, total=False):
    """Compile-time and scheduling knobs understood by the native frontend."""

    mma_tiler_mnk: tuple[int, int, int]
    cluster_shape_mnk: tuple[int, int, int]
    num_stages: int | Literal["auto"]
    group_hint: int | None
    persistent_clusters: int | None
    flag_batch: int
    load_balance_mode: LoadBalanceMode
    claim_order: Literal["expert", "longest_first"]
    token_back_mode: Literal["epi_warps"]
    epilogue_warps: Literal[4, 8]
    in_kernel_reduce: bool
    cta_local_counts: bool
    weight_l2_evict_first: bool


_TACTIC_KEYS = frozenset(Mxfp8Mxfp4Tactic.__annotations__)


def validate_tactic(tactic: Mapping[str, Any], *, partial: bool = True) -> None:
    """Validate a pinned tactic without importing CuTe DSL or CUDA modules."""

    unknown = set(tactic) - _TACTIC_KEYS
    if unknown:
        raise ValueError(f"unknown MXFP8 x MXFP4 tactic keys: {sorted(unknown)}")
    if not partial:
        missing = _TACTIC_KEYS - set(tactic)
        if missing:
            raise ValueError(
                f"incomplete MXFP8 x MXFP4 tactic: missing {sorted(missing)}"
            )

    tile = tactic.get("mma_tiler_mnk")
    if tile is not None:
        if not isinstance(tile, tuple) or len(tile) != 3:
            raise ValueError("mma_tiler_mnk must be a three-element tuple")
        m, n, k = tile
        if m not in (128, 256):
            raise ValueError(
                "MXFP8 x MXFP4 MMA M must be 128 (1-CTA) or 256 (2-CTA pair)"
            )
        if n not in (64, 128, 192):
            raise ValueError("mma_tiler_mnk N must be 64, 128, or 192")
        if k <= 0 or k % 32:
            raise ValueError("mma_tiler_mnk K must be a positive multiple of 32")

    cluster = tactic.get("cluster_shape_mnk")
    if cluster is not None:
        if not isinstance(cluster, tuple) or len(cluster) != 3:
            raise ValueError("cluster_shape_mnk must be a three-element tuple")
        if cluster[2] != 1 or any(v not in (1, 2, 4) for v in cluster[:2]):
            raise ValueError("cluster_shape_mnk must be (1|2|4, 1|2|4, 1)")
        if cluster[0] * cluster[1] > 16:
            raise ValueError("cluster_shape_mnk may contain at most 16 CTAs")

    # A partial override may pin M256 alone; resolve_tactic supplies the
    # matching cluster before validating the complete tactic.
    if (
        tile is not None
        and tile[0] == 256
        and (cluster is not None or not partial)
        and cluster != (2, 1, 1)
    ):
        raise ValueError("MMA M=256 (2-CTA) requires cluster_shape_mnk=(2, 1, 1)")

    stages = tactic.get("num_stages")
    if (
        stages is not None
        and stages != "auto"
        and (not isinstance(stages, int) or not 2 <= stages <= 8)
    ):
        raise ValueError("num_stages must be 'auto' or an integer in [2, 8]")
    group_hint = tactic.get("group_hint")
    if group_hint is not None and (not isinstance(group_hint, int) or group_hint <= 0):
        raise ValueError("group_hint must be None or a positive integer")
    clusters = tactic.get("persistent_clusters")
    if clusters is not None and (type(clusters) is not int or clusters <= 0):
        raise ValueError("persistent_clusters must be None or a positive integer")
    flag_batch = tactic.get("flag_batch")
    if flag_batch is not None and (
        not isinstance(flag_batch, int) or not 1 <= flag_batch <= 32
    ):
        raise ValueError("flag_batch must be an integer in [1, 32]")
    if tactic.get("load_balance_mode", "static") not in (
        "alternating",
        "static",
        "atomic_counter",
    ):
        raise ValueError(
            "load_balance_mode must be static, atomic_counter, or alternating"
        )
    if tactic.get("claim_order", "expert") not in ("expert", "longest_first"):
        raise ValueError("claim_order must be expert or longest_first")
    if tactic.get("token_back_mode", "epi_warps") != "epi_warps":
        raise ValueError(
            "token_back_mode must be epi_warps (dispatch-warp and standalone "
            "token-back are unsupported)"
        )
    if tactic.get("epilogue_warps", 4) not in (4, 8):
        raise ValueError("epilogue_warps must be 4 or 8")
    if not isinstance(tactic.get("cta_local_counts", False), bool):
        raise ValueError("cta_local_counts must be a bool")
    if not isinstance(tactic.get("weight_l2_evict_first", False), bool):
        raise ValueError("weight_l2_evict_first must be a bool")
    if not isinstance(tactic.get("in_kernel_reduce", False), bool):
        raise ValueError("in_kernel_reduce must be a bool")


def _default_epilogue_warps(max_tokens_per_rank: int, pair: bool) -> Literal[4, 8]:
    return 8 if pair and max_tokens_per_rank >= EIGHT_WARP_EPILOGUE_MIN_TOKENS else 4


def supports_two_cta(hidden: int | None, intermediate: int | None) -> bool:
    """Whether the 2-CTA (M256) pair tile divides both GEMM feature extents."""

    return (
        hidden is not None
        and intermediate is not None
        and hidden % 256 == 0
        and (2 * intermediate) % 256 == 0
    )


def _default_in_kernel_reduce(max_tokens_per_rank: int) -> bool:
    return max_tokens_per_rank < IN_KERNEL_REDUCE_MAX_TOKENS


def _default_load_balance_mode(max_tokens_per_rank: int, pair: bool) -> LoadBalanceMode:
    if not pair:
        return "static"
    if max_tokens_per_rank < ATOMIC_CLAIM_MAX_TOKENS:
        return "atomic_counter"
    return (
        "alternating"
        if max_tokens_per_rank >= ALTERNATING_CLAIM_MIN_TOKENS
        else "static"
    )


def default_tactic(
    max_tokens_per_rank: int,
    *,
    hidden: int | None = None,
    intermediate: int | None = None,
) -> Mxfp8Mxfp4Tactic:
    """Return a deterministic launch tactic for the supplied workload.

    Token tiles and scheduler groups trade padding and repeated weight loads
    against shared-memory use. Automatic stage depth selects the deepest
    operand pipeline that fits. Offline tuning may override these defaults.
    """

    if max_tokens_per_rank <= 0:
        raise ValueError("max_tokens_per_rank must be positive")
    small_groups = max_tokens_per_rank >= 512
    pair = supports_two_cta(hidden, intermediate)
    token_tile = 64
    if max_tokens_per_rank >= 256:
        token_tile = 128
    if pair and max_tokens_per_rank >= ALTERNATING_CLAIM_MIN_TOKENS:
        token_tile = 192
    tactic: Mxfp8Mxfp4Tactic = {
        "mma_tiler_mnk": (256 if pair else 128, token_tile, 128),
        "cluster_shape_mnk": (2, 1, 1) if pair else (1, 1, 1),
        "num_stages": "auto",
        "persistent_clusters": None,
        "group_hint": 512 if small_groups else 4096,
        "flag_batch": 1,
        "load_balance_mode": _default_load_balance_mode(max_tokens_per_rank, pair),
        "claim_order": "expert",
        "token_back_mode": "epi_warps",
        "epilogue_warps": _default_epilogue_warps(max_tokens_per_rank, pair),
        "in_kernel_reduce": _default_in_kernel_reduce(max_tokens_per_rank),
        "cta_local_counts": True,
        "weight_l2_evict_first": pair and max_tokens_per_rank <= 512,
    }
    validate_tactic(tactic, partial=False)
    return tactic


def candidate_tactics(
    *, hidden: int | None = None, intermediate: int | None = None
) -> list[Mxfp8Mxfp4Tactic]:
    """Bounded search over supported geometry and measured scheduling settings.

    With a shape the pair tile supports (see :func:`default_tactic`), the
    grid spans the axes the defaults choose between: N64/N128/N192 token
    tiles, ``group_hint`` 512/4096, four or eight epilogue warps, and both
    top-k reduction placements with static claims (24), plus alternating
    claims at N192 and atomic claims at N64 (``group_hint`` 4096) for both
    widths and placements (8; ``group_hint`` has no effect with alternating
    claims), all ``num_stages="auto"`` and ``epi_warps`` token back.
    Each paired-tile tactic also tries a 64-cluster cap and both weight L2
    policies, for 128 candidates total. Every default tactic is in this
    grid, so tuning cannot do worse
    than the shipped band on the timed run.

    Without such a shape, the 1-CTA grid: N64/N128 at auto and pinned 3-5
    operand stages (auto fits more than four stages for the MAI geometry),
    ``group_hint`` 512/4096 and both reduction placements. N256 is rejected
    by the assembly.
    """

    candidates: list[Mxfp8Mxfp4Tactic] = []
    if supports_two_cta(hidden, intermediate):
        for n in (64, 128, 192):
            for group_hint in (512, 4096):
                for epilogue_warps in (4, 8):
                    for in_kernel_reduce in (False, True):
                        tactic = default_tactic(
                            1, hidden=hidden, intermediate=intermediate
                        )
                        tactic["mma_tiler_mnk"] = (256, n, 128)
                        tactic["group_hint"] = group_hint
                        tactic["epilogue_warps"] = cast(Literal[4, 8], epilogue_warps)
                        tactic["in_kernel_reduce"] = in_kernel_reduce
                        tactic["load_balance_mode"] = "static"
                        validate_tactic(tactic, partial=False)
                        candidates.append(tactic)
        for epilogue_warps in (4, 8):
            for in_kernel_reduce in (False, True):
                tactic = default_tactic(1, hidden=hidden, intermediate=intermediate)
                tactic["mma_tiler_mnk"] = (256, 192, 128)
                tactic["group_hint"] = 512
                tactic["epilogue_warps"] = cast(Literal[4, 8], epilogue_warps)
                tactic["in_kernel_reduce"] = in_kernel_reduce
                tactic["load_balance_mode"] = "alternating"
                validate_tactic(tactic, partial=False)
                candidates.append(tactic)
                # The atomic-claim default below 256 tokens per rank.
                tactic = tactic.copy()
                tactic["mma_tiler_mnk"] = (256, 64, 128)
                tactic["group_hint"] = 4096
                tactic["load_balance_mode"] = "atomic_counter"
                validate_tactic(tactic, partial=False)
                candidates.append(tactic)
        for tactic in candidates.copy():
            capped = tactic.copy()
            capped["persistent_clusters"] = 64
            validate_tactic(capped, partial=False)
            candidates.append(capped)
        # The weight L2 policy flips between bands (default_tactic), so the
        # grid carries both settings of every paired tactic.
        for tactic in candidates.copy():
            flipped = tactic.copy()
            flipped["weight_l2_evict_first"] = not tactic["weight_l2_evict_first"]
            validate_tactic(flipped, partial=False)
            candidates.append(flipped)
        return candidates
    for n in (64, 128):
        for stages in ("auto", 3, 4, 5):
            for group_hint in (512, 4096):
                for in_kernel_reduce in (False, True):
                    tactic = default_tactic(1)
                    tactic["mma_tiler_mnk"] = (128, n, 128)
                    tactic["num_stages"] = stages
                    tactic["group_hint"] = group_hint
                    tactic["in_kernel_reduce"] = in_kernel_reduce
                    validate_tactic(tactic, partial=False)
                    candidates.append(tactic)
    return candidates


def resolve_tactic(
    max_tokens_per_rank: int,
    overrides: Mxfp8Mxfp4Tactic | None,
    *,
    hidden: int | None = None,
    intermediate: int | None = None,
) -> Mxfp8Mxfp4Tactic:
    """Merge a partial pinned tactic over the deterministic host default.

    An override that pins the MMA M without a cluster shape gets the cluster
    that M implies ((1, 1, 1) for M128, (2, 1, 1) for M256), so pinning a
    1-CTA tile never inherits the default's 2-CTA cluster.
    """

    tactic = default_tactic(
        max_tokens_per_rank, hidden=hidden, intermediate=intermediate
    )
    if overrides is not None:
        validate_tactic(overrides)
        tactic.update(overrides)
        if "mma_tiler_mnk" in overrides and "cluster_shape_mnk" not in overrides:
            tactic["cluster_shape_mnk"] = (
                (2, 1, 1) if overrides["mma_tiler_mnk"][0] == 256 else (1, 1, 1)
            )
    validate_tactic(tactic, partial=False)
    return tactic


def _mxfp8_mxfp4_quant_config() -> QuantConfig:
    return QuantConfig(
        weight=QuantFormat.MXFP4,
        activation=QuantFormat.MXFP8,
        output=QuantFormat.BF16,
        swizzled_scale_factors=True,
        per_token_scale=False,
    )


@dataclass
class Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig:
    """Public config for E4M3/K32 activations and E2M1/K32 weights.

    Returns the weighted top-k sum as BF16 ``[tokens, hidden]``.

    ``knobs="auto"`` tunes each workspace at its first eager compute using
    FlashInfer's general autotuner. ``knobs=None`` uses a general cached winner,
    then the legacy knob cache/default, and participates in ``autotune()``.
    An explicit dictionary (including ``{}``) pins the tactic. Unpinned
    workspaces reserve capacity for the candidate set. Finish tuning and
    warmup on every EP rank before capturing graphs.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = KERNEL_NAME
    quant: QuantConfig = field(default_factory=_mxfp8_mxfp4_quant_config)
    gate_up_clamp: float | None = None
    fast_math: bool = True
    knobs: Mxfp8Mxfp4Tactic | Literal["auto"] | None = None

    def __post_init__(self) -> None:
        if self.kernel_name != KERNEL_NAME:
            raise ValueError(f"kernel_name is fixed to {KERNEL_NAME!r}")
        if self.quant.pair != (QuantFormat.MXFP4, QuantFormat.MXFP8) or (
            self.quant.output is not QuantFormat.BF16
        ):
            raise ValueError(
                "MXFP8 x MXFP4 MegaMoE requires "
                "QuantConfig(weight=MXFP4, activation=MXFP8, output=BF16)"
            )
        if self.quant.swizzled_scale_factors is False:
            raise ValueError("MXFP4 kernel weights require swizzled scale factors")
        if self.quant.per_token_scale is True:
            raise ValueError("MXFP8 activations require E8M0 scales per K32 block")
        if self.intermediate_size <= 0 or self.intermediate_size % 128:
            raise ValueError("intermediate_size must be a positive multiple of 128")
        if self.top_k <= 0 or self.top_k > 32:
            raise ValueError("top_k must be in [1, 32]")
        if self.gate_up_clamp is not None and (
            not math.isfinite(self.gate_up_clamp) or self.gate_up_clamp <= 0
        ):
            raise ValueError("gate_up_clamp must be finite and positive")
        if isinstance(self.knobs, dict):
            validate_tactic(self.knobs)
        elif self.knobs not in (None, "auto"):
            raise TypeError("knobs must be a tactic dict, 'auto', or None")


__all__ = [
    "EIGHT_WARP_EPILOGUE_MIN_TOKENS",
    "ALTERNATING_CLAIM_MIN_TOKENS",
    "ATOMIC_CLAIM_MAX_TOKENS",
    "IN_KERNEL_REDUCE_MAX_TOKENS",
    "KERNEL_NAME",
    "Mxfp8Mxfp4Tactic",
    "Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig",
    "candidate_tactics",
    "default_tactic",
    "supports_two_cta",
    "resolve_tactic",
    "validate_tactic",
]
