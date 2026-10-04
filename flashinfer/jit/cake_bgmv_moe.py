"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Sequence, Tuple, Dict

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)
from .utils import write_if_different

CakeBGMVMoEDType = Literal["bfloat16", "float16"]
CakeBGMVMoEArch = Literal["sm90a", "sm100a", "sm103a"]
CakeBGMVMoESchedule = Literal[
    "token_owned_t64",
    "token_owned",
    "token_owned_dual_col",
]

CAKE_BGMV_MOE_HIDDEN_SIZES = (2688, 3072)
CAKE_BGMV_MOE_DTYPES: tuple[CakeBGMVMoEDType, ...] = (
    "bfloat16",
    "float16",
)
CAKE_BGMV_MOE_SCHEDULE_IDS: dict[CakeBGMVMoESchedule, int] = {
    "token_owned_t64": 0,
    "token_owned": 1,
    "token_owned_dual_col": 2,
}

# Generic-shape bundles: one compile-time LoRA rank per module, any hidden size
# that is a positive multiple of 8 at run time (the shrink kernels stream
# 1024-wide tiles with a masked tail). The specialized hidden 2688/3072 x rank
# 32 bodies above stay preferred when they apply.
CakeBGMVMoEVariant = Literal["specialized", "generic"]
CakeBGMVMoEGenericSchedule = Literal["token_owned_t64", "token_owned_t128"]
CAKE_BGMV_MOE_GENERIC_RANKS = (8, 16, 32, 64)
CAKE_BGMV_MOE_GENERIC_HIDDEN_MULTIPLE = 8
# Token->pair route index shared by both variants: the shrink kernels publish
# every routed pair under its token, the expand kernels read a token's routes
# in O(1) for arbitrary pair order (tokens with more routes take the exact
# serial scan). Counts are monotonic with launch-parity bases, so the plan
# allocates the workspace zeroed once and the kernels never reset it: a
# 4-word header plus, per token, one count, two bases and 16 pair slots.
CAKE_BGMV_MOE_ROUTE_INDEX_MAX_ROUTES = 16
CAKE_BGMV_MOE_ROUTE_INDEX_HEADER_WORDS = 4
CAKE_BGMV_MOE_ROUTE_INDEX_WORDS_PER_TOKEN = 3 + CAKE_BGMV_MOE_ROUTE_INDEX_MAX_ROUTES


# Hidden-split shrink workspace appended to the route index: FP32 partials
# ``[split][pair][64]`` (sized for rank 64, stored as raw 32-bit words) and one
# arrival counter per (pair block, rank block). The generic shrink kernels
# split the hidden tiles over grid z when the pair x rank-block grid is small;
# the last arrival reduces the partials in split order (deterministic) and
# resets its counter, so the zeroed workspace is never memset again.
CAKE_BGMV_MOE_SHRINK_RANK_TILE = 8
CAKE_BGMV_MOE_SHRINK_TILE = 1024
CAKE_BGMV_MOE_SHRINK_SPLIT_MAX = 8
CAKE_BGMV_MOE_SHRINK_SPLIT_MAX_PAIRS = 128
CAKE_BGMV_MOE_SHRINK_SPLIT_TARGET_CTAS = 128
CAKE_BGMV_MOE_SHRINK_SPLIT_PARTIAL_WORDS = (
    CAKE_BGMV_MOE_SHRINK_SPLIT_MAX * CAKE_BGMV_MOE_SHRINK_SPLIT_MAX_PAIRS * 64
)
CAKE_BGMV_MOE_SHRINK_SPLIT_COUNTER_WORDS = CAKE_BGMV_MOE_SHRINK_SPLIT_MAX_PAIRS * (
    64 // CAKE_BGMV_MOE_SHRINK_RANK_TILE
)


# Pair-grouped pipeline (generic bundles, round 5): routes are grouped by their
# unique (LoRA, expert) pair so each pair's A/B weights are streamed once per tile
# of ``CAKE_BGMV_MOE_GROUP_TILE_TOKENS`` routes instead of once per route. The
# plan owns an int32 grouping workspace (header, bin counts/offsets/fill, tile
# table, grouped route ids, per-token route lists, per-CTA histogram rows) rebuilt by the grouping kernels
# on every launch, and FP32 per-route expand partials ``[num_pairs, hidden]``
# that the deterministic per-token combine sums in ascending pair order.
CAKE_BGMV_MOE_GROUP_TILE_TOKENS = 16
CAKE_BGMV_MOE_GROUP_BINS_MAX = 4096
CAKE_BGMV_MOE_GROUP_HEADER_WORDS = 4
CAKE_BGMV_MOE_GROUP_HIST_THREADS = 256
CAKE_BGMV_MOE_GROUP_HIST_PAIRS_PER_LANE = 4
CAKE_BGMV_MOE_GROUP_HIST_CTAS_MAX = 64
CAKE_BGMV_MOE_GROUPED_MIN_PAIRS = 2048
CAKE_BGMV_MOE_GROUPED_MIN_PAIRS_SUB64 = 4096
CAKE_BGMV_MOE_GROUPED_MIN_ROUTES_PER_BIN = 4
CAKE_BGMV_MOE_GROUPED_MIN_WEIGHT_ELEMS = 12288
CAKE_BGMV_MOE_GROUPED_MIN_WEIGHT_ELEMS_R8 = 16384


def cake_bgmv_moe_group_max_tiles(num_pairs: int, bins: int) -> int:
    """Upper bound on the (group, token chunk) tiles of the grouped kernels."""

    return (
        int(num_pairs) + CAKE_BGMV_MOE_GROUP_TILE_TOKENS - 1
    ) // CAKE_BGMV_MOE_GROUP_TILE_TOKENS + int(bins)


def cake_bgmv_moe_group_hist_ctas(num_pairs: int) -> int:
    """CTAs of the grouping histogram / scatter kernels (mirrors the binding's GroupHistCtas)."""

    per_cta = CAKE_BGMV_MOE_GROUP_HIST_THREADS * CAKE_BGMV_MOE_GROUP_HIST_PAIRS_PER_LANE
    return max(
        1,
        min(
            CAKE_BGMV_MOE_GROUP_HIST_CTAS_MAX, (int(num_pairs) + per_cta - 1) // per_cta
        ),
    )


def cake_bgmv_moe_grouped_workspace_words(
    num_pairs: int, num_tokens: int, bins: int
) -> int:
    """int32 words of the grouping workspace (mirrors the binding's ComputeGroupedOffsets)."""

    num_pairs, num_tokens, bins = int(num_pairs), int(num_tokens), int(bins)
    return (
        CAKE_BGMV_MOE_GROUP_HEADER_WORDS
        + bins
        + 1  # bin offsets
        + cake_bgmv_moe_group_max_tiles(num_pairs, bins)
        + num_pairs  # grouped route ids
        + num_tokens  # per-token route counts
        + num_tokens * CAKE_BGMV_MOE_ROUTE_INDEX_MAX_ROUTES
        + 2
        * cake_bgmv_moe_group_hist_ctas(num_pairs)
        * bins  # per-CTA histogram + base rows
    )


def select_cake_bgmv_moe_generic_grouped(
    num_pairs: int,
    num_tokens: int,
    num_loras: int,
    num_experts: int,
    hidden_size: int,
    rank: int,
) -> bool:
    """True when the generic plan should run the pair-grouped pipeline.

    Mirrors the Cake generator's ``select_generic_grouped``: weight reuse pays
    off once the routes clearly outnumber the ``num_loras * num_experts`` bins
    (``CAKE_BGMV_MOE_GROUPED_MIN_PAIRS`` routes and
    ``CAKE_BGMV_MOE_GROUPED_MIN_ROUTES_PER_BIN`` routes per bin) and the
    per-pair weights (``hidden_size * rank``) are large enough for the saved
    traffic to exceed the fixed grouping prologue and the FP32 partials round
    trip; between ``CAKE_BGMV_MOE_GROUPED_MIN_PAIRS`` and
    ``CAKE_BGMV_MOE_GROUPED_MIN_PAIRS_SUB64`` routes only rank 64 wins; the
    grouping prologue bounds the bin count by ``CAKE_BGMV_MOE_GROUP_BINS_MAX``.
    """

    bins = int(num_loras) * int(num_experts)
    if bins <= 0 or bins > CAKE_BGMV_MOE_GROUP_BINS_MAX:
        return False
    if int(num_pairs) < CAKE_BGMV_MOE_GROUPED_MIN_PAIRS:
        return False
    if int(num_pairs) < CAKE_BGMV_MOE_GROUPED_MIN_ROUTES_PER_BIN * bins:
        return False
    if int(rank) < 64 and int(num_pairs) < CAKE_BGMV_MOE_GROUPED_MIN_PAIRS_SUB64:
        return False
    min_elems = (
        CAKE_BGMV_MOE_GROUPED_MIN_WEIGHT_ELEMS
        if int(rank) >= 16
        else CAKE_BGMV_MOE_GROUPED_MIN_WEIGHT_ELEMS_R8
    )
    if int(hidden_size) * int(rank) < min_elems:
        return False
    if (
        int(num_pairs) + CAKE_BGMV_MOE_GROUP_TILE_TOKENS - 1
    ) // CAKE_BGMV_MOE_GROUP_TILE_TOKENS >= 65536:
        return False
    return cake_bgmv_moe_group_max_tiles(num_pairs, bins) < 2**31


def cake_bgmv_moe_route_index_words(num_tokens: int) -> int:
    """int32 words of the route index proper (header + per-token entries)."""

    return CAKE_BGMV_MOE_ROUTE_INDEX_HEADER_WORDS + (
        int(num_tokens) * CAKE_BGMV_MOE_ROUTE_INDEX_WORDS_PER_TOKEN
    )


def cake_bgmv_moe_route_index_numel(num_tokens: int) -> int:
    """int32 elements of the plan workspace (route index + hidden-split region)."""

    return (
        cake_bgmv_moe_route_index_words(num_tokens)
        + CAKE_BGMV_MOE_SHRINK_SPLIT_PARTIAL_WORDS
        + CAKE_BGMV_MOE_SHRINK_SPLIT_COUNTER_WORDS
    )


def select_cake_bgmv_moe_generic_shrink(
    num_pairs: int, rank: int, hidden_size: int
) -> Tuple[int, int]:
    """(decode kernel flag, hidden splits) for the generic shrink launch.

    Mirrors the Cake generator's ``select_generic_shrink_launch``: the 1-pair
    kernel beat the 4-pair decode kernel on every measured small-pair row once
    the partials moved to registers, so the decode flag is always 0.  Hidden
    splits are added only while the pair x rank-block grid is below
    ``CAKE_BGMV_MOE_SHRINK_SPLIT_TARGET_CTAS`` CTAs, bounded by the tile count
    and ``CAKE_BGMV_MOE_SHRINK_SPLIT_MAX``.
    """

    if num_pairs <= 0:
        raise ValueError(f"num_pairs must be positive, got {num_pairs}")
    if rank not in CAKE_BGMV_MOE_GENERIC_RANKS:
        raise ValueError(
            f"rank must be one of {CAKE_BGMV_MOE_GENERIC_RANKS}, got {rank}"
        )
    tiles = (hidden_size + CAKE_BGMV_MOE_SHRINK_TILE - 1) // CAKE_BGMV_MOE_SHRINK_TILE
    ctas = num_pairs * (rank // CAKE_BGMV_MOE_SHRINK_RANK_TILE)
    max_splits = (
        min(tiles, CAKE_BGMV_MOE_SHRINK_SPLIT_MAX)
        if num_pairs <= CAKE_BGMV_MOE_SHRINK_SPLIT_MAX_PAIRS
        else 1
    )
    target = CAKE_BGMV_MOE_SHRINK_SPLIT_TARGET_CTAS
    splits = min(max_splits, max(1, (target + ctas - 1) // ctas))
    return 0, splits


CAKE_BGMV_MOE_GENERIC_SCHEDULE_IDS: dict[CakeBGMVMoEGenericSchedule, int] = {
    "token_owned_t64": 0,
    "token_owned_t128": 1,
}


class CakeBGMVMoEArchTarget(NamedTuple):
    arch: CakeBGMVMoEArch
    capability: Tuple[int, int]
    nvcc_flags: Sequence[str]


# The generated programs use cp.async, warp shuffles and FMA only, so one
# source body serves Hopper and both Blackwell data-center targets; each
# target gets its own cubin and module so the binding can fail closed on a
# mismatched device.
CAKE_BGMV_MOE_ARCH_TARGETS: dict[CakeBGMVMoEArch, CakeBGMVMoEArchTarget] = {
    "sm90a": CakeBGMVMoEArchTarget("sm90a", (9, 0), tuple(sm90a_nvcc_flags)),
    "sm100a": CakeBGMVMoEArchTarget("sm100a", (10, 0), tuple(sm100a_nvcc_flags)),
    "sm103a": CakeBGMVMoEArchTarget("sm103a", (10, 3), tuple(sm103a_nvcc_flags)),
}
CAKE_BGMV_MOE_ARCHES: tuple[CakeBGMVMoEArch, ...] = tuple(CAKE_BGMV_MOE_ARCH_TARGETS)


class CakeBGMVMoEMetadata(NamedTuple):
    body: str
    shrink_decode_symbol: str
    shrink_prefill_symbol: str
    token_t64_symbol: str
    token_symbol: str
    token_dual_col_symbol: str


class CakeBGMVMoEGenericMetadata(NamedTuple):
    body: str
    shrink_decode_symbol: str
    shrink_prefill_symbol: str
    shrink_decode_pdl_symbol: str
    shrink_prefill_pdl_symbol: str
    expand_t64_symbol: str
    expand_t128_symbol: str
    expand_t64_pf_symbol: str
    expand_t128_pf_symbol: str
    group_hist_symbol: str
    group_scan_symbol: str
    group_scatter_symbol: str
    shrink_grouped_symbol: str
    shrink_grouped_single_symbol: str
    expand_grouped_symbol: str
    combine_grouped_symbol: str


def cake_bgmv_moe_arch_for_capability(
    capability: Tuple[int, int],
) -> Optional[CakeBGMVMoEArch]:
    """Return the generated target for a CUDA compute capability, or ``None``."""

    capability = (int(capability[0]), int(capability[1]))
    for target in CAKE_BGMV_MOE_ARCH_TARGETS.values():
        if target.capability == capability:
            return target.arch
    return None


def _check_arch(arch: str) -> CakeBGMVMoEArchTarget:
    try:
        return CAKE_BGMV_MOE_ARCH_TARGETS[arch]  # type: ignore[index]
    except KeyError:
        raise ValueError(
            f"Cake BGMV MoE arch must be one of {CAKE_BGMV_MOE_ARCHES}, got {arch!r}"
        ) from None


def _dtype_tag(dtype: CakeBGMVMoEDType) -> str:
    if dtype == "bfloat16":
        return "bf16"
    if dtype == "float16":
        return "f16"
    raise ValueError(f"unsupported Cake BGMV MoE dtype: {dtype}")


def _check_hidden_size(hidden_size: int) -> None:
    if hidden_size not in CAKE_BGMV_MOE_HIDDEN_SIZES:
        raise ValueError(
            f"Cake BGMV MoE hidden_size must be 2688 or 3072, got {hidden_size}"
        )


def _metadata(hidden_size: int, dtype: CakeBGMVMoEDType) -> CakeBGMVMoEMetadata:
    _check_hidden_size(hidden_size)
    tag = _dtype_tag(dtype)
    return CakeBGMVMoEMetadata(
        body=f"cake_bgmv_moe_{tag}_h{hidden_size}.cu",
        shrink_decode_symbol=(
            f"kernel_flashinfer_bgmv_moe_shrink_{tag}_h{hidden_size}_r32_p4_s3"
        ),
        shrink_prefill_symbol=(
            f"kernel_flashinfer_bgmv_moe_shrink_{tag}_h{hidden_size}_r32_p1_s2"
        ),
        token_t64_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_token_t64_{tag}_h{hidden_size}_r32"
        ),
        token_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_token_{tag}_h{hidden_size}_r32"
        ),
        token_dual_col_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_token_dual_col_{tag}_h{hidden_size}_r32"
        ),
    )


def _check_generic_rank(rank: int) -> None:
    if rank not in CAKE_BGMV_MOE_GENERIC_RANKS:
        raise ValueError(
            "Cake BGMV MoE generic rank must be one of "
            f"{CAKE_BGMV_MOE_GENERIC_RANKS}, got {rank}"
        )


# Token-count window (inclusive) in which the specialized hidden 2688/3072 x
# rank-32 bodies serve, per architecture.  The window is independent of the
# route layout the host cannot see: below it the generic bundle's hidden-split
# shrink and programmatic dependent launch win by 1.5-2.6x at 1-16 tokens on
# both layouts; inside it the specialized bodies win on expert-sorted routes
# (generic 1.08-1.10x slower at 32 tokens, 1.2-1.3x at 64-256, 1.3-1.5x at
# 512-1024 on B200/GB300) while losing 7-21 % on contiguous top-k=2 routes;
# above it the generic pair-grouped pipeline wins (~0.65 at 2048 tokens).  On
# Hopper the generic bundle wins or ties at every token count.  Measured in
# round 5 (lever 8, fair screens with PDL on both arms, 3 interleaved reps).
CAKE_BGMV_MOE_SPECIALIZED_TOKEN_WINDOW: Dict[
    CakeBGMVMoEArch, Optional[Tuple[int, int]]
] = {
    "sm90a": None,
    "sm100a": (32, 1024),
    "sm103a": (32, 1024),
}

# Programmatic dependent launch (PDL) of the per-route expand behind the shrink.
# Mode 0 = plain stream launch, 1 = shrink triggers its dependents at CTA entry,
# 2 = shrink triggers after its tile loop.  Blackwell wins with PDL on every
# decode row (early best for small expand grids, late best for large grids);
# Hopper wins only on small expand grids and loses 1-5 % at 512 tokens with
# either trigger.  "Small" = at most this many expand CTAs per SM, counted at
# 128 output columns per CTA: crossover sweeps (736/1472/2944 x 32 x 32..256
# tokens) put the Blackwell early/late crossover between 768 and 1472 CTAs
# (8/SM) and the Hopper early/off crossover between 1536 and 2944 CTAs (12/SM).
CAKE_BGMV_MOE_PDL_SMALL_EXPAND_CTAS_PER_SM: Dict[CakeBGMVMoEArch, int] = {
    "sm90a": 12,
    "sm100a": 8,
    "sm103a": 8,
}
CAKE_BGMV_MOE_PDL_EXPAND_COLS_NOMINAL = 128


def cake_bgmv_moe_pdl_mode(
    arch: CakeBGMVMoEArch,
    num_tokens: int,
    hidden_size: int,
    sm_count: int,
    variant: CakeBGMVMoEVariant = "generic",
) -> int:
    """PDL launch mode for the per-route shrink -> expand pair (0 off, 1 early, 2 late).

    The specialized 2688/3072 bodies take plain launches on every architecture:
    round-5 A/B vs the plain launch measured the late trigger 1.02-1.04 at
    256-512 tokens and the early trigger 1.09-1.11 at 1024 expert-sorted
    tokens (the earlier "specialized PDL ~1.00" screens ran stale bundles
    without griddepcontrol)."""

    if variant == "specialized":
        return 0
    cols = CAKE_BGMV_MOE_PDL_EXPAND_COLS_NOMINAL
    expand_ctas = int(num_tokens) * ((int(hidden_size) + cols - 1) // cols)
    per_sm = CAKE_BGMV_MOE_PDL_SMALL_EXPAND_CTAS_PER_SM.get(arch, 0)
    small = expand_ctas <= per_sm * int(sm_count)
    if arch in ("sm100a", "sm103a"):
        return 1 if small else 2
    if arch == "sm90a":
        return 1 if small else 0
    return 0


def cake_bgmv_moe_variant(
    hidden_size: int,
    rank: int,
    num_tokens: Optional[int] = None,
    arch: Optional[CakeBGMVMoEArch] = None,
) -> Optional[CakeBGMVMoEVariant]:
    """Return which generated Cake bundle serves ``(hidden_size, rank)``.

    ``"specialized"`` for the measured hidden 2688/3072 x rank 32 bodies when
    ``num_tokens`` lies in ``CAKE_BGMV_MOE_SPECIALIZED_TOKEN_WINDOW[arch]``
    (``num_tokens=None`` means "any token count" and keeps the specialized
    answer for support queries), ``"generic"`` for any other hidden size that
    is a positive multiple of 8 at rank 8, 16, 32 or 64, and ``None`` when no
    generated program applies.
    """

    hidden_size = int(hidden_size)
    rank = int(rank)
    if hidden_size in CAKE_BGMV_MOE_HIDDEN_SIZES and rank == 32:
        if num_tokens is None:
            return "specialized"
        window = CAKE_BGMV_MOE_SPECIALIZED_TOKEN_WINDOW.get(arch or "sm100a")
        if window is not None and window[0] <= int(num_tokens) <= window[1]:
            return "specialized"
        return "generic"
    if (
        rank in CAKE_BGMV_MOE_GENERIC_RANKS
        and hidden_size > 0
        and hidden_size % CAKE_BGMV_MOE_GENERIC_HIDDEN_MULTIPLE == 0
    ):
        return "generic"
    return None


def _generic_metadata(rank: int, dtype: CakeBGMVMoEDType) -> CakeBGMVMoEGenericMetadata:
    _check_generic_rank(rank)
    tag = _dtype_tag(dtype)
    return CakeBGMVMoEGenericMetadata(
        body=f"cake_bgmv_moe_generic_{tag}_r{rank}.cu",
        shrink_decode_symbol=(
            f"kernel_flashinfer_bgmv_moe_shrink_generic_{tag}_r{rank}_p4_s3"
        ),
        shrink_prefill_symbol=(
            f"kernel_flashinfer_bgmv_moe_shrink_generic_{tag}_r{rank}_p1_s2"
        ),
        # PDL forms: griddepcontrol.launch_dependents (early/late by pdl_early);
        # selected by the binding together with the prefetch expand forms.
        shrink_decode_pdl_symbol=(
            f"kernel_flashinfer_bgmv_moe_shrink_generic_{tag}_r{rank}_p4_s3_pdl"
        ),
        shrink_prefill_pdl_symbol=(
            f"kernel_flashinfer_bgmv_moe_shrink_generic_{tag}_r{rank}_p1_s2_pdl"
        ),
        expand_t64_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_generic_token_t64_{tag}_r{rank}"
        ),
        expand_t128_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_generic_token_t128_{tag}_r{rank}"
        ),
        # Register-prefetch forms: both routes' B rows are loaded into registers
        # before griddepcontrol.wait. Selected by the binding for PDL launches
        # (pdl_mode != 0); plain launches take the lower-register forms above.
        expand_t64_pf_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_generic_token_t64_pf_{tag}_r{rank}"
        ),
        expand_t128_pf_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_generic_token_t128_pf_{tag}_r{rank}"
        ),
        group_hist_symbol=f"kernel_flashinfer_bgmv_moe_group_hist_{tag}_r{rank}",
        group_scan_symbol=f"kernel_flashinfer_bgmv_moe_group_scan_{tag}_r{rank}",
        group_scatter_symbol=f"kernel_flashinfer_bgmv_moe_group_scatter_{tag}_r{rank}",
        shrink_grouped_symbol=f"kernel_flashinfer_bgmv_moe_shrink_grouped_{tag}_r{rank}",
        # Lever 20k: one-rank-tile-per-CTA form (rank 8 has one rank tile: same kernel).
        shrink_grouped_single_symbol=(
            f"kernel_flashinfer_bgmv_moe_shrink_grouped_single_{tag}_r{rank}"
            if rank > 8
            else f"kernel_flashinfer_bgmv_moe_shrink_grouped_{tag}_r{rank}"
        ),
        expand_grouped_symbol=f"kernel_flashinfer_bgmv_moe_expand_grouped_{tag}_r{rank}",
        combine_grouped_symbol=f"kernel_flashinfer_bgmv_moe_combine_grouped_{tag}_r{rank}",
    )


def select_cake_bgmv_moe_generic_schedule(
    hidden_size: int,
    num_tokens: int,
    arch: CakeBGMVMoEArch = "sm100a",
) -> CakeBGMVMoEGenericSchedule:
    """Return the expand schedule for the generic-shape bundle.

    The 64-lane token-owned expand wins at decode batch sizes (few tokens,
    more CTAs per token keep the SMs busy); the 128-lane variant wins once
    the grid is wide enough on its own. Both are deterministic.
    """

    if cake_bgmv_moe_variant(hidden_size, CAKE_BGMV_MOE_GENERIC_RANKS[0]) is None:
        raise ValueError(
            "Cake BGMV MoE generic hidden_size must be a positive multiple of "
            f"{CAKE_BGMV_MOE_GENERIC_HIDDEN_MULTIPLE}, got {hidden_size}"
        )
    _check_arch(arch)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if num_tokens <= 8:
        return "token_owned_t64"
    return "token_owned_t128"


def select_cake_bgmv_moe_schedule(
    hidden_size: int,
    num_tokens: int,
    arch: CakeBGMVMoEArch = "sm100a",
) -> CakeBGMVMoESchedule:
    """Return the measured selector for the supported rank-32 portfolio.

    The table was measured on B200 (SM100, 148 SMs); all targets share it.
    Sweeping the three expand schedules over the serving shapes (hidden
    2688/3072, 1..1024 tokens, BF16, CUPTI cold-L2) puts them within 1.2 % of
    each other on B200 and within 3.5 % on H100 (SM90), where
    ``token_owned_dual_col`` trails ``token_owned`` by about 3 % at 1024
    tokens. A per-target table is deliberately not introduced for that gap.
    """

    _check_hidden_size(hidden_size)
    _check_arch(arch)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    if num_tokens in (1, 4, 8):
        return "token_owned_t64"
    if hidden_size == 3072 and num_tokens in (512, 1024):
        return "token_owned_dual_col"
    if hidden_size == 2688 and num_tokens == 1024:
        return "token_owned_dual_col"
    return "token_owned"


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_bgmv_moe"
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_bgmv_moe"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "generated Cake BGMV MoE sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _get_include_dir() -> Path:
    """Locate FlashInfer headers in installed and source checkouts."""

    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR

    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def get_cake_bgmv_moe_uri(
    hidden_size: int,
    dtype: CakeBGMVMoEDType,
    arch: CakeBGMVMoEArch = "sm100a",
) -> str:
    _check_hidden_size(hidden_size)
    _check_arch(arch)
    tag = _dtype_tag(dtype)
    return f"cake_bgmv_moe_{tag}_h{hidden_size}_{arch}"


def _binding_source(
    metadata: CakeBGMVMoEMetadata, hidden_size: int, target: CakeBGMVMoEArchTarget
) -> str:
    input_dtype = "dl_bfloat16" if "_bf16_" in metadata.body else "dl_float16"
    major, minor = target.capability
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#define CAKE_BGMV_MOE_BODY_FILE \"{metadata.body}\"
#define CAKE_BGMV_MOE_HIDDEN {hidden_size}
#define CAKE_BGMV_MOE_INPUT_DTYPE {input_dtype}
#define CAKE_BGMV_MOE_CC_MAJOR {major}
#define CAKE_BGMV_MOE_CC_MINOR {minor}
#define CAKE_BGMV_MOE_SHRINK_DECODE {metadata.shrink_decode_symbol}
#define CAKE_BGMV_MOE_SHRINK_PREFILL {metadata.shrink_prefill_symbol}
#define CAKE_BGMV_MOE_EXPAND_TOKEN_T64 {metadata.token_t64_symbol}
#define CAKE_BGMV_MOE_EXPAND_TOKEN {metadata.token_symbol}
#define CAKE_BGMV_MOE_EXPAND_TOKEN_DUAL {metadata.token_dual_col_symbol}

#include \"cake_bgmv_moe_binding.cuh\"
"""


def get_cake_bgmv_moe_generic_uri(
    rank: int,
    dtype: CakeBGMVMoEDType,
    arch: CakeBGMVMoEArch = "sm100a",
) -> str:
    _check_generic_rank(rank)
    _check_arch(arch)
    tag = _dtype_tag(dtype)
    return f"cake_bgmv_moe_generic_{tag}_r{rank}_{arch}"


def _generic_binding_source(
    metadata: CakeBGMVMoEGenericMetadata, rank: int, target: CakeBGMVMoEArchTarget
) -> str:
    input_dtype = "dl_bfloat16" if "_bf16_" in metadata.body else "dl_float16"
    major, minor = target.capability
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#define CAKE_BGMV_MOE_BODY_FILE \"{metadata.body}\"
#define CAKE_BGMV_MOE_RANK {rank}
#define CAKE_BGMV_MOE_INPUT_DTYPE {input_dtype}
#define CAKE_BGMV_MOE_CC_MAJOR {major}
#define CAKE_BGMV_MOE_CC_MINOR {minor}
#define CAKE_BGMV_MOE_SHRINK_DECODE {metadata.shrink_decode_symbol}
#define CAKE_BGMV_MOE_SHRINK_PREFILL {metadata.shrink_prefill_symbol}
#define CAKE_BGMV_MOE_SHRINK_DECODE_PDL {metadata.shrink_decode_pdl_symbol}
#define CAKE_BGMV_MOE_SHRINK_PREFILL_PDL {metadata.shrink_prefill_pdl_symbol}
#define CAKE_BGMV_MOE_EXPAND_T64 {metadata.expand_t64_symbol}
#define CAKE_BGMV_MOE_EXPAND_T128 {metadata.expand_t128_symbol}
#define CAKE_BGMV_MOE_EXPAND_T64_PF {metadata.expand_t64_pf_symbol}
#define CAKE_BGMV_MOE_EXPAND_T128_PF {metadata.expand_t128_pf_symbol}
#define CAKE_BGMV_MOE_GROUP_HIST {metadata.group_hist_symbol}
#define CAKE_BGMV_MOE_GROUP_SCAN {metadata.group_scan_symbol}
#define CAKE_BGMV_MOE_GROUP_SCATTER {metadata.group_scatter_symbol}
#define CAKE_BGMV_MOE_SHRINK_GROUPED {metadata.shrink_grouped_symbol}
#define CAKE_BGMV_MOE_SHRINK_GROUPED_SINGLE {metadata.shrink_grouped_single_symbol}
#define CAKE_BGMV_MOE_EXPAND_GROUPED {metadata.expand_grouped_symbol}
#define CAKE_BGMV_MOE_COMBINE_GROUPED {metadata.combine_grouped_symbol}

#include \"cake_bgmv_moe_generic_binding.cuh\"
"""


@functools.cache
def gen_cake_bgmv_moe_generic_module(
    rank: int,
    dtype: CakeBGMVMoEDType,
    arch: CakeBGMVMoEArch = "sm100a",
) -> JitSpec:
    metadata = _generic_metadata(rank, dtype)
    target = _check_arch(arch)
    csrc_dir = _get_csrc_dir()
    include_dir = _get_include_dir()
    body = csrc_dir / metadata.body
    binding_header = csrc_dir / "cake_bgmv_moe_generic_binding.cuh"
    if not body.is_file():
        raise FileNotFoundError(
            f"generated Cake BGMV MoE generic body not found: {body}"
        )
    if not binding_header.is_file():
        raise FileNotFoundError(
            f"Cake BGMV MoE generic binding header not found: {binding_header}"
        )

    uri = get_cake_bgmv_moe_generic_uri(rank, dtype, arch)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_bgmv_moe_generic_binding.cu"
    write_if_different(binding, _generic_binding_source(metadata, rank, target))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[*target.nvcc_flags, "-use_fast_math"],
        extra_include_paths=[csrc_dir, csrc_dir.parent, include_dir],
    )
    logger.info("Generated Cake BGMV MoE generic JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_bgmv_moe_generic_module(
    rank: int,
    dtype: CakeBGMVMoEDType,
    arch: CakeBGMVMoEArch = "sm100a",
):
    module = gen_cake_bgmv_moe_generic_module(rank, dtype, arch).build_and_load()
    module.configure()
    logger.info(
        "Loaded Cake BGMV MoE generic module for rank=%d, dtype=%s, arch=%s",
        rank,
        dtype,
        arch,
    )
    return module


def get_cake_bgmv_moe_generic_module(
    rank: int,
    dtype: CakeBGMVMoEDType,
    arch: CakeBGMVMoEArch = "sm100a",
):
    return load_cake_bgmv_moe_generic_module(rank, dtype, arch)


@functools.cache
def gen_cake_bgmv_moe_module(
    hidden_size: int,
    dtype: CakeBGMVMoEDType,
    arch: CakeBGMVMoEArch = "sm100a",
) -> JitSpec:
    metadata = _metadata(hidden_size, dtype)
    target = _check_arch(arch)
    csrc_dir = _get_csrc_dir()
    include_dir = _get_include_dir()
    body = csrc_dir / metadata.body
    binding_header = csrc_dir / "cake_bgmv_moe_binding.cuh"
    if not body.is_file():
        raise FileNotFoundError(f"generated Cake BGMV MoE body not found: {body}")
    if not binding_header.is_file():
        raise FileNotFoundError(
            f"Cake BGMV MoE binding header not found: {binding_header}"
        )

    uri = get_cake_bgmv_moe_uri(hidden_size, dtype, arch)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_bgmv_moe_binding.cu"
    write_if_different(binding, _binding_source(metadata, hidden_size, target))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[*target.nvcc_flags, "-use_fast_math"],
        extra_include_paths=[csrc_dir, csrc_dir.parent, include_dir],
    )
    logger.info("Generated Cake BGMV MoE JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_bgmv_moe_module(
    hidden_size: int,
    dtype: CakeBGMVMoEDType,
    arch: CakeBGMVMoEArch = "sm100a",
):
    module = gen_cake_bgmv_moe_module(hidden_size, dtype, arch).build_and_load()
    module.configure()
    logger.info(
        "Loaded Cake BGMV MoE module for hidden_size=%d, dtype=%s, arch=%s",
        hidden_size,
        dtype,
        arch,
    )
    return module


def get_cake_bgmv_moe_module(
    hidden_size: int,
    dtype: CakeBGMVMoEDType,
    arch: CakeBGMVMoEArch = "sm100a",
):
    return load_cake_bgmv_moe_module(hidden_size, dtype, arch)


__all__ = [
    "CAKE_BGMV_MOE_ARCHES",
    "CAKE_BGMV_MOE_ARCH_TARGETS",
    "CAKE_BGMV_MOE_DTYPES",
    "CAKE_BGMV_MOE_GENERIC_HIDDEN_MULTIPLE",
    "CAKE_BGMV_MOE_GENERIC_RANKS",
    "CAKE_BGMV_MOE_GENERIC_SCHEDULE_IDS",
    "CAKE_BGMV_MOE_HIDDEN_SIZES",
    "CAKE_BGMV_MOE_ROUTE_INDEX_HEADER_WORDS",
    "CAKE_BGMV_MOE_ROUTE_INDEX_MAX_ROUTES",
    "CAKE_BGMV_MOE_ROUTE_INDEX_WORDS_PER_TOKEN",
    "CAKE_BGMV_MOE_SCHEDULE_IDS",
    "CAKE_BGMV_MOE_SHRINK_SPLIT_MAX",
    "CAKE_BGMV_MOE_PDL_EXPAND_COLS_NOMINAL",
    "CAKE_BGMV_MOE_PDL_SMALL_EXPAND_CTAS_PER_SM",
    "CAKE_BGMV_MOE_SPECIALIZED_TOKEN_WINDOW",
    "cake_bgmv_moe_pdl_mode",
    "CAKE_BGMV_MOE_SHRINK_SPLIT_MAX_PAIRS",
    "CakeBGMVMoEArch",
    "CakeBGMVMoEArchTarget",
    "CakeBGMVMoEDType",
    "CakeBGMVMoEGenericMetadata",
    "CakeBGMVMoEGenericSchedule",
    "CakeBGMVMoEMetadata",
    "CakeBGMVMoESchedule",
    "CakeBGMVMoEVariant",
    "cake_bgmv_moe_arch_for_capability",
    "cake_bgmv_moe_route_index_numel",
    "cake_bgmv_moe_route_index_words",
    "cake_bgmv_moe_variant",
    "gen_cake_bgmv_moe_generic_module",
    "gen_cake_bgmv_moe_module",
    "get_cake_bgmv_moe_generic_module",
    "get_cake_bgmv_moe_generic_uri",
    "get_cake_bgmv_moe_module",
    "get_cake_bgmv_moe_uri",
    "select_cake_bgmv_moe_generic_shrink",
    "select_cake_bgmv_moe_generic_grouped",
    "cake_bgmv_moe_grouped_workspace_words",
    "cake_bgmv_moe_group_max_tiles",
    "cake_bgmv_moe_group_hist_ctas",
    "load_cake_bgmv_moe_generic_module",
    "load_cake_bgmv_moe_module",
    "select_cake_bgmv_moe_generic_schedule",
    "select_cake_bgmv_moe_schedule",
]
