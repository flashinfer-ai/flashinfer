# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared contiguous grouped FP8 GEMM launches on Blackwell (SM100a, SM103a).

This is the generated Cake program family for the same mathematical
contract as :func:`flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous`:
``out[r, :] = sum_q (a[r, q-block] . b[g_r, :, q-block]) * a_scale[r, q] *
b_scale[g_r, :, q]`` with FP8 E4M3 operands, per-row 128-wide K-block A scales,
128x128 B block scales, ordered FP32 K-block accumulation and a BF16 output.
Rows are routed to experts by the sorted ``m_indices`` vector.  Unlike the
CuTe-DSL kernel, internal expert boundaries need not be aligned to 128 rows.

Two kernel families serve the contract.  FP32 scales (any values) run the
promotion family: plain E4M3 MMA with FP32 register scaling of every K block.
Packed UE8M0 int32 scales (DeepGEMM's SM100 layout: one int32 holds four
consecutive K-block exponents, activations ``(M, ceil(K/512))`` with any
strides, weights ``(G, N//128, cols)`` or the row-repeated ``(G, N, cols)``)
run the block-scaled family: ``tcgen05.mma.kind::mxf8f6f4`` applies both
scales in the tensor core, ``-1`` entries of ``m_indices`` mark padding rows
of the compact MoE layout (skipped, output rows unspecified) and expert runs
start on multiples of ``alignment`` rows (128-multiples select the single-run
schedule, other multiples of 32 the slower multi-run schedule).  When the
packed words are DeepGEMM's MN-major activation words and the row-repeated
weight words and ``N`` is a multiple of 256 (also an odd multiple of 128,
see :func:`bs_swap_small_route`), the host may route a problem to
the swap-AB programs (:func:`bs_swap_route_for_shape`): the weight rows fill
the 256-row side of a two-CTA MMA and the routed tokens its N side.  With the
same words, a single-run layout of at most 128 routed rows per group runs the
store16 programs (:func:`bs_store16`): they write each expert run's routed
rows rounded up to 16 rows instead of 32.  Every block-scaled route
accumulates each output in the same K order, so the route never changes the
bits of a routed row.

The host side selects one kernel route per problem shape and output
alignment, resolves the launch grid from the device SM count (queried once per
device), and binds the positional argument plan of the registered program.
With ``validate_indices`` the expert-run layout read in the same transfer (and,
for some layouts, the expert starts per tile) may select another route, such
as the expert-aligned program (a speed choice; results are identical).
Tensor maps are encoded by the binding and passed to the kernel by value, so a
prepared launch owns no descriptor storage: every ``launch()`` submits exactly
one kernel on the current stream and may be captured into a CUDA graph,
including the first one.  Preparation binds the problem shape and the expert
weights; the per-token operands (``a``, ``a_scale``, ``m_indices``, ``out``)
may be rebound at every ``launch`` so callers need no staging copies.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import tvm_ffi

from ..jit.gemm.cake_grouped_fp8_gemm import (
    ARG_PLANS,
    MODULES,
    PROGRAMS,
    ROUTE_GEOMETRY,
    SUPPORTED_COMPUTE_CAPABILITIES,
    device_arch,
    generated_program_available,
    load_cake_grouped_fp8_gemm_module,
    route_program,
    select_module,
)

B_SCALE_BLOCK_N = 128
K_BLOCK = 128

# Route names double as the generated-program template names.
K128_ROUTE = "k128_exact"
K384_ROUTE = "k384_exact_three_partial"
DEEPK_C2_ROUTE = "deepk_c2_k_multiple_512"
DEEPK_C2_AB6_SCALE1_ROUTE = "deepk_c2_ab6_scale1_n256_or_m4096"
DEEPK_C2_KG1_ROUTE = "deepk_c2_kg1_non_kg4_k_blocks"
DEEPK_C2_SCALAR_OUTPUT_ROUTE = "deepk_c2_scalar_output"
DEEPK_C2_KG1_DIRECT_STORE_ROUTE = "deepk_c2_kg1_direct_store_m32"
# The KG1 program with 128 x 64 tiles: a KG1 launch of more than
# DIRECT_STORE_MAX_ROWS rows whose 128 x 64 tiles fit in one wave (see _launch_plan).
DEEPK_C2_KG1_N64_ROUTE = "deepk_c2_kg1_n64_k_blocks"
DEEPK_C2_AB6_SCALE1_DIRECT_STORE_ROUTE = "deepk_c2_ab6_scale1_direct_store_m32"
DEEPK_N256_ROUTE = "deepk_n256_ab4_three_panel_kg4"
DEEPK_CG2_RECURRENCE_ROUTE = "deepk_cg2_ab5_x16_pair_wait_kg4"
DEEPK_CG2_FOUR_LOAD_ROUTE = "deepk_cg2_ab5_x16_four_load_wait_kg4"
DEEPK_CG2_FOUR_LOAD_GRID128_ROUTE = "deepk_cg2_ab5_x16_four_load_wait_grid128_kg4"
DEEPK_CG2_RECURRENCE_EARLY4_ROUTE = "deepk_cg2_ab6_early4_output_alias_kg4"
DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE = "deepk_cg2_ab7_bscale_prefetch_kg4"
# Expert-aligned K-group 1 schedule: 128-row tiles that never span two experts,
# cut on the device from m_indices.  Selected only from the layout that
# ``validate_indices`` reads (see ``prepare_group_gemm_fp8_nt_groupwise_contiguous``).
DEEPK_C2_KG1_EXPERT_ALIGNED_ROUTE = "deepk_c2_kg1_expert_aligned"
# The expert-aligned program keeps one tile-table entry per expert in shared memory.
EXPERT_ALIGNED_MAX_GROUPS = 255
# Block-scaled (packed UE8M0 scale) family.
BS_N128_ROUTE = "bs_ue8m0_n128"
BS_N256_ROUTE = "bs_ue8m0_n256"
BS_N128_RUN32_ROUTE = "bs_ue8m0_n128_run32"
BS_N256_RUN32_ROUTE = "bs_ue8m0_n256_run32"
# The store16 programs of the single-run routes (:func:`bs_store16`).
BS_N128_S16_ROUTE = "bs_ue8m0_n128_s16"
BS_N256_S16_ROUTE = "bs_ue8m0_n256_s16"
BS_ROUTES = frozenset(
    {
        BS_N128_ROUTE,
        BS_N256_ROUTE,
        BS_N128_RUN32_ROUTE,
        BS_N256_RUN32_ROUTE,
        BS_N128_S16_ROUTE,
        BS_N256_S16_ROUTE,
    }
)
BS_BLOCK_M = 128
BS_BLOCK_K = 128
BS_SUB_ROWS = 32

_CG2_SELECTED_SHAPES = frozenset(
    {
        (16384, 2048, 4096),
        (16384, 4096, 1024),
        (4096, 2048, 4096),
        (4096, 4096, 1024),
    }
)
# Measured CTA cap: these two shapes run their two-CTA routes on 128 CTAs.
_CG2_GRID128_SHAPES = frozenset({(4096, 2048, 4096), (4096, 4096, 1024)})
# The two-CTA routes run 256-row tiles.  A tile that holds the start of a
# non-empty expert runs one main loop per expert segment, so these routes are
# fast only when every non-empty expert starts on a multiple of 256 rows.
_CG2_TILE_M = 256
# (N, K) classes of the two-CTA routes: the route for a routing aligned to
# 256 rows near the selected M (the 256-row tile count of a selected shape or
# one more), and the single-CTA route for a known unaligned routing.
_CG2_ALIGNED_ROUTE = {
    (2048, 4096): DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE,
    (4096, 1024): DEEPK_CG2_RECURRENCE_ROUTE,
}
_CG2_UNALIGNED_ROUTE = {
    (2048, 4096): DEEPK_C2_AB6_SCALE1_ROUTE,
    (4096, 1024): DEEPK_N256_ROUTE,
}
_CG2_SELECTED_TILES_M = frozenset(m // _CG2_TILE_M for m, _, _ in _CG2_SELECTED_SHAPES)
# The two-CTA route each selected shape of those classes ships for an unknown
# layout.  For a known unaligned layout at these shapes the host keeps it unless
# the single-CTA route shortens the launch's critical path (:func:`_max_load`).
_CG2_SHAPE_ROUTE = {
    (4096, 4096, 1024): DEEPK_CG2_FOUR_LOAD_ROUTE,
    (16384, 4096, 1024): DEEPK_CG2_RECURRENCE_ROUTE,
    (4096, 2048, 4096): DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE,
}
# M-tile heights whose inner expert starts a validated launch reads for
# :func:`_max_load` (the single-CTA and the two-CTA routes of those shapes).
_TILE_START_ROWS = (128, _CG2_TILE_M)
# A K=384 problem of at most this many 128x128 tiles runs the K-loop route.
K384_KG1_MAX_TILES = 4
# At (16384, 4096, 1024) a known unaligned layout leaves the two-CTA pair route
# only when the single-CTA route's busiest worker runs at least this many fewer
# expert segments.
PAIR_N256_MIN_LOAD_GAIN = 2
# A K=512 problem of at least this many 128x128 tiles keeps the K-multiple-of-512
# route, except at N=256 or M=4096.
K512_C2_MIN_TILES = 2
# A launch whose every tile has at most DIRECT_STORE_MAX_ROWS rows (M <= 32) on at
# most DIRECT_STORE_MAX_CTAS CTAs runs the direct-store program of its KG1 / AB6
# route: it stores such a tile with direct row stores instead of the TMA epilogue
# (same output bytes), which shortens the tail of a launch made of such tiles
# (measured faster on B200 and GB300).  Every other launch keeps the route's program.
DIRECT_STORE_MAX_ROWS = 32
DIRECT_STORE_MAX_CTAS = 16
_DIRECT_STORE_ROUTE = {
    DEEPK_C2_KG1_ROUTE: DEEPK_C2_KG1_DIRECT_STORE_ROUTE,
    DEEPK_C2_AB6_SCALE1_ROUTE: DEEPK_C2_AB6_SCALE1_DIRECT_STORE_ROUTE,
}


def route_for_shape(m: int, n: int, k: int) -> str:
    """Select the kernel route for a 16-byte-aligned BF16 output from the shape.

    :func:`launch_plan` refines it with the device's SM count, and a launch
    prepared with ``validate_indices`` also with the expert-run layout it reads.
    """
    return _route(m, n, k)


def _route(
    m: int,
    n: int,
    k: int,
    *,
    sm_count: Optional[int] = None,
    aligned: Optional[bool] = None,
    tile_starts: Optional[Mapping[int, Sequence[int]]] = None,
) -> str:
    """The FP32-scale route of a 16-byte-aligned BF16 output.

    ``aligned`` (whether every expert run starts on a multiple of 256 rows,
    when the layout is known at preparation) selects the two-CTA routes only
    for layouts aligned to their 256-row tiles.  At the shapes of
    ``_CG2_SHAPE_ROUTE`` a known unaligned layout keeps the two-CTA route
    unless the single-CTA route shortens the launch's critical path, counted
    from ``tile_starts`` and ``sm_count`` (:func:`_max_load`).  ``sm_count``
    also lets K=384 problems of at most ``K384_KG1_MAX_TILES`` 128x128 tiles
    run the K-loop route instead of the static-K route, and K=128 problems of
    at most ``sm_count`` 128x128 tiles (one wave) run the K-loop K-group 1
    route, whose accumulator starts from -0.0 at K=128 so that its output
    has the bits of the static-K route's.  K=512 problems of at
    least ``K512_C2_MIN_TILES`` 128x128 tiles keep the K-multiple-of-512 route
    except at N=256 or M=4096.
    """
    if k <= 0 or k % K_BLOCK:
        raise ValueError(
            f"contiguous grouped FP8 GEMM requires positive K divisible by {K_BLOCK}; got K={k}"
        )
    if (n, k) in _CG2_ALIGNED_ROUTE:
        if aligned is False:
            return _unaligned_cg2_route(m, n, k, sm_count, tile_starts)
        if (m, n, k) == (4096, 4096, 1024):
            return DEEPK_CG2_FOUR_LOAD_ROUTE
        if (m, n, k) in _CG2_SELECTED_SHAPES:
            return _CG2_ALIGNED_ROUTE[(n, k)]
        if aligned and math.ceil(m / _CG2_TILE_M) in {
            t + d for t in _CG2_SELECTED_TILES_M for d in (0, 1)
        }:
            return _CG2_ALIGNED_ROUTE[(n, k)]
    if k == 128:
        if sm_count is not None and math.ceil(m / 128) * math.ceil(n / 128) <= sm_count:
            return DEEPK_C2_KG1_ROUTE
        return K128_ROUTE
    if k == 384:
        if (
            sm_count is not None
            and math.ceil(m / 128) * math.ceil(n / 128) <= K384_KG1_MAX_TILES
        ):
            return DEEPK_C2_KG1_ROUTE
        return K384_ROUTE
    if (k // K_BLOCK) % 4:
        return DEEPK_C2_KG1_ROUTE
    if m == 16384 and n in {2048, 4096}:
        return DEEPK_N256_ROUTE
    if (
        k == 512
        and n != 256
        and m != 4096
        and math.ceil(m / 128) * math.ceil(n / 128) >= K512_C2_MIN_TILES
    ):
        return DEEPK_C2_ROUTE
    return DEEPK_C2_AB6_SCALE1_ROUTE


def _max_load(
    route: str, m: int, n: int, tile_starts: Mapping[int, Sequence[int]], workers: int
) -> int:
    """Most expert segments one worker of a persistent ``route`` launch runs.

    Tiles are numbered M-major (``t = m_tile * n_tiles + n_tile``) and worker
    ``w`` (a CTA, or a cluster of a two-CTA route) runs tiles ``w, w + workers,
    ...``.  A tile runs one main loop per expert overlapping its rows: one, plus
    the non-empty experts that start inside it (``tile_starts[tile_m]`` per M
    tile).
    """
    tile_m, tile_n, _ = route_geometry(route)
    n_tiles = math.ceil(n / tile_n)
    total = math.ceil(m / tile_m) * n_tiles
    load = [total // workers + (w < total % workers) for w in range(workers)]
    for m_tile, starts in enumerate(tile_starts[tile_m]):
        if starts:
            for tile in range(m_tile * n_tiles, (m_tile + 1) * n_tiles):
                load[tile % workers] += starts
    return max(load)


def _unaligned_cg2_route(
    m: int,
    n: int,
    k: int,
    sm_count: Optional[int],
    tile_starts: Optional[Mapping[int, Sequence[int]]],
) -> str:
    """Route of a known layout that is not aligned to 256 rows, for ``(n, k)`` in
    ``_CG2_UNALIGNED_ROUTE``.

    A tile holding an expert start runs one more main loop, which costs a
    256-row two-CTA tile more than a 128-row tile.  At the shapes of
    ``_CG2_SHAPE_ROUTE`` the single-CTA route is taken only where its busiest
    worker runs fewer expert segments than the two-CTA route's (at least
    ``PAIR_N256_MIN_LOAD_GAIN`` fewer for the pair route; at N=2048, K=4096: at
    most 1.5 times as many, its segments being shorter); elsewhere those classes
    ship no two-CTA route for such a layout.
    """
    single = _CG2_UNALIGNED_ROUTE[(n, k)]
    shape_route = _CG2_SHAPE_ROUTE.get((m, n, k))
    if shape_route is None or sm_count is None or tile_starts is None:
        return single
    load_cg2 = _max_load(
        shape_route, m, n, tile_starts, _route_workers(shape_route, m, n, k, sm_count)
    )
    load_single = _max_load(
        single, m, n, tile_starts, _route_workers(single, m, n, k, sm_count)
    )
    if shape_route == DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE:
        return single if 2 * load_single <= 3 * load_cg2 else shape_route
    if shape_route == DEEPK_CG2_RECURRENCE_ROUTE:
        return (
            single if load_single <= load_cg2 - PAIR_N256_MIN_LOAD_GAIN else shape_route
        )
    return single if load_single < load_cg2 else shape_route


def bs_block_n(
    m: int,
    n: int,
    k: int,
    sm_count: int,
    *,
    alignment: int = BS_BLOCK_M,
    num_groups: int | None = None,
) -> int:
    """``BLOCK_N`` of the block-scaled family.

    ``m`` counts every row of the compact layout, including the ``-1`` padding
    tail, and each CTA steps over the skipped tiles of that tail: 256-column
    tiles halve that walk and the A traffic, so they are the default.  128
    columns win where they buy parallelism: when a 256-column grid has fewer
    tiles than the device has SMs, and for long-K problems (at least 16 K
    blocks) with fewer than three waves of M blocks.  Two cases of that long-K
    range still run 256 columns:

    - the single-run schedule below 32 K blocks and two waves of M blocks;
    - the 32-row multi-run schedule once the routed rows, sized as
      ``m - num_groups * (alignment - 1) - 127`` (the caller's buffer of routed
      rows plus each group's alignment padding and the 128-row round-up), would
      at one row per run fill more 128-column tiles than the device has SMs.

    The choice never changes the output bits: every route accumulates each
    output in the same K order.
    """
    if n % B_SCALE_BLOCK_N:
        raise ValueError(f"N must be a multiple of {B_SCALE_BLOCK_N}, got {n}")
    if n % 256:
        return 128
    m_blocks = (m + BS_BLOCK_M - 1) // BS_BLOCK_M
    if m_blocks * (n // 256) < sm_count:
        return 128
    k_blocks = k // K_BLOCK
    if k_blocks < 16 or m_blocks >= 3 * sm_count:
        return 256
    if alignment % BS_BLOCK_M == 0:
        return 256 if k_blocks < 32 and m_blocks < 2 * sm_count else 128
    if num_groups is not None:
        routed = m - num_groups * (alignment - 1) - (BS_BLOCK_M - 1)
        run_blocks = (routed * alignment + BS_BLOCK_M - 1) // BS_BLOCK_M
        if routed > 0 and run_blocks * (n // 128) > sm_count:
            return 256
    return 128


def bs_route_for_shape(
    m: int,
    n: int,
    k: int,
    *,
    alignment: int,
    sm_count: int,
    num_groups: int | None = None,
) -> str:
    """Select the block-scaled route for packed UE8M0 scales."""
    if alignment <= 0 or alignment % BS_SUB_ROWS:
        raise ValueError(
            f"alignment must be a positive multiple of {BS_SUB_ROWS}, got {alignment}"
        )
    if k <= 0 or k % BS_BLOCK_K:
        raise ValueError(
            f"block-scaled grouped FP8 GEMM requires positive K divisible by {BS_BLOCK_K}; got K={k}"
        )
    block_n = bs_block_n(m, n, k, sm_count, alignment=alignment, num_groups=num_groups)
    single_run = alignment % BS_BLOCK_M == 0
    return {
        (128, True): BS_N128_ROUTE,
        (256, True): BS_N256_ROUTE,
        (128, False): BS_N128_RUN32_ROUTE,
        (256, False): BS_N256_RUN32_ROUTE,
    }[(block_n, single_run)]


# Swap-AB block-scaled programs: the weight rows fill the 256-row side of the
# two-CTA MMA and the routed tokens its N side.
BS_SWAP_ROUTE = "bs_ue8m0_swap"
BS_SWAP_CARRY_ROUTE = "bs_ue8m0_swap_carry"
BS_SWAP_RUN32_ROUTE = "bs_ue8m0_swap_run32"
BS_SWAP_S16_ROUTE = "bs_ue8m0_swap_s16"
BS_SWAP_CARRY_S16_ROUTE = "bs_ue8m0_swap_carry_s16"
BS_SWAP_RP32_ROUTE = "bs_ue8m0_swap_rp32"
BS_SWAP_RP32_S11_ROUTE = "bs_ue8m0_swap_rp32_s11"
BS_SWAP_C1_ROUTE = "bs_ue8m0_swap_c1"
BS_SWAP_ROUTES = frozenset(
    {
        BS_SWAP_ROUTE,
        BS_SWAP_RUN32_ROUTE,
        BS_SWAP_CARRY_ROUTE,
        BS_SWAP_S16_ROUTE,
        BS_SWAP_CARRY_S16_ROUTE,
        BS_SWAP_RP32_ROUTE,
        BS_SWAP_RP32_S11_ROUTE,
        BS_SWAP_C1_ROUTE,
    }
)
BS_SWAP_ARCHS = frozenset({"sm_100a", "sm_103a"})
# sm_100a swaps a long-K problem of at least 100 rows per group when the
# non-swapped BLOCK_N 128 grid has at most this many tiles (half of 148 SMs).
BS_SWAP_SM100_MAX_TILES = 74
# The 32-row-block swap-AB program (each 32-row M block holds one group's
# rows) serves the 32- and 64-row-aligned layouts of at most 1536 rows on
# these architectures, in place of BS_SWAP_RUN32_ROUTE, while its launch
# balances better than the run32 program's: it spreads 32-row blocks over
# every SM but streams a group's weight tile once per block, so it pays only
# while a run32 launch would leave SMs idle.  Per architecture
# ``(rows, factor)`` bounds the tiles, ``factor * ceil(m / rows) * (n / 256)
# <= sm_count``: sm_100a at most one 32 x 256 tile per SM, sm_103a at most a
# third of ``sm_count`` 128 x 256 run32 tiles (a run32 launch on at most two
# thirds of the SMs).  Without ``sm_count`` both keep BS_SWAP_RUN32_ROUTE.
BS_SWAP_RP32_TILE_BOUND = {"sm_100a": (BS_SUB_ROWS, 1), "sm_103a": (BS_BLOCK_M, 3)}
BS_SWAP_RP32_ARCHS = frozenset(BS_SWAP_RP32_TILE_BOUND)
BS_SWAP_RP32_MAX_ALIGNMENT = 64
BS_SWAP_RP32_MAX_ROWS = 1536
# On these architectures the 32-row-block program also serves the single-run
# layouts (an alignment that is a multiple of 128) of more than
# BS_SWAP_RP32_DENSE_MIN_ROWS rows per group, ahead of the other routes, while
# its launch is one pass of two-CTA clusters, ``2 * ceil(m / 32) * n / 256``
# CTAs of at most ``sm_count``: each cluster then streams one 32 x 256 tile's
# weights, where the 128-row programs would leave most SMs idle.
BS_SWAP_RP32_DENSE_ARCHS = frozenset({"sm_100a"})
BS_SWAP_RP32_DENSE_MIN_ROWS = 128
# The small-block swap-AB programs (32-row M blocks on a compact A slot, so
# more stages fit in shared memory) serve these architectures, ahead of the
# other swap-AB routes, while their launch is one wave on the device:
# BS_SWAP_RP32_S11_ROUTE (two-CTA clusters, 11 stages) for any alignment at
# K of at least BS_SWAP_RP32_S11_MIN_K, ``2 * ceil(m / 32) * n / 256``
# CTAs of at most ``sm_count``; BS_SWAP_C1_ROUTE (one CTA per 32-row block
# and 128-row weight tile, no cluster) for the 32-row alignment at
# BS_SWAP_C1_MIN_K <= K <= BS_SWAP_C1_MAX_K and N of at most
# BS_SWAP_C1_MAX_N, ``ceil(m / 32) * n / 128`` CTAs of at most ``sm_count``.
# The other swap-AB programs need N a multiple of 256; BS_SWAP_C1_ROUTE's
# 128-row weight tile also serves N an odd multiple of 128 for the multi-run
# layouts (an alignment of 32, 64 or 96 rows modulo 128) at
# BS_SWAP_C1_N128_MIN_K <= K <= BS_SWAP_C1_N128_MAX_K, with the same one-wave
# bound.  Without ``sm_count`` none of these is taken.
BS_SWAP_SMALL_ARCHS = frozenset({"sm_103a"})
BS_SWAP_RP32_S11_MIN_K = 2048
BS_SWAP_C1_ALIGNMENT = 32
BS_SWAP_C1_MIN_K = 256
BS_SWAP_C1_MAX_K = 512
BS_SWAP_C1_MAX_N = 512
BS_SWAP_C1_N128_MIN_K = 128
BS_SWAP_C1_N128_MAX_K = 512
# On these architectures the small-block programs also take launches of more
# than one wave, ahead of the other swap-AB routes: at most
# BS_SWAP_SM100_SMALL_TILES_PER_SM one-CTA tiles (32 x 128, ``ceil(m / 32) *
# n / 128`` of them) per SM and K of at most BS_SWAP_SM100_SMALL_MAX_K.  They
# serve the multi-run layouts within the 32-row-block program's bounds (an
# alignment of at most BS_SWAP_RP32_MAX_ALIGNMENT, ``m`` of at most
# BS_SWAP_RP32_MAX_ROWS) at N a multiple of 256, and any layout at N an odd
# multiple of 128 with BS_SWAP_C1_N128_MIN_K <= K <= BS_SWAP_C1_N128_MAX_K.
# BS_SWAP_RP32_S11_ROUTE takes N a multiple of 256 at K of at least
# BS_SWAP_RP32_S11_MIN_K while its launch is one wave or the layout has at
# most BS_SWAP_SM100_S11_MAX_BLOCKS 32-row blocks; BS_SWAP_C1_ROUTE takes the
# others.  Without ``sm_count`` none of these is taken.
BS_SWAP_SM100_SMALL_ARCHS = frozenset({"sm_100a"})
BS_SWAP_SM100_SMALL_MAX_K = 4096
BS_SWAP_SM100_SMALL_TILES_PER_SM = 3
BS_SWAP_SM100_S11_MAX_BLOCKS = 20
# These compact layouts, ``(m, n, k, alignment)``, keep the other swap-AB
# routes on BS_SWAP_SM100_SMALL_ARCHS: on a 148-SM B200 the small-block
# programs were less than 3 % faster than those routes on them, or slower.
BS_SWAP_SM100_SMALL_KEEP = frozenset(
    {
        (256, 4096, 1024, 32),
        (384, 1024, 128, 32),
        (384, 2048, 1024, 32),
        (384, 2560, 2048, 32),
        (384, 3072, 1024, 32),
        (512, 2048, 1024, 32),
        (512, 2048, 2048, 32),
        (512, 2048, 2048, 64),
        (512, 2560, 2048, 32),
        (512, 3072, 1024, 32),
        (640, 1024, 512, 32),
        (640, 1024, 768, 32),
        (640, 1024, 1024, 32),
        (768, 1536, 1024, 32),
        (896, 1024, 512, 64),
        (896, 1280, 1024, 64),
        (1152, 512, 256, 32),
        (1152, 1024, 512, 32),
        (1152, 1024, 1024, 32),
        (1280, 512, 4096, 32),
    }
)


def _bs_swap_small_route_sm100(
    m: int, n: int, k: int, *, alignment: int, sm_count: int
) -> str | None:
    """The small-block route of ``BS_SWAP_SM100_SMALL_ARCHS`` (see :func:`bs_swap_small_route`)."""
    if (m, n, k, alignment) in BS_SWAP_SM100_SMALL_KEEP:
        return None
    blocks = math.ceil(m / BS_SUB_ROWS)
    if (
        k > BS_SWAP_SM100_SMALL_MAX_K
        or blocks * (n // 128) > BS_SWAP_SM100_SMALL_TILES_PER_SM * sm_count
    ):
        return None
    if n % 256:
        if BS_SWAP_C1_N128_MIN_K <= k <= BS_SWAP_C1_N128_MAX_K:
            return BS_SWAP_C1_ROUTE
        return None
    if (
        alignment % BS_BLOCK_M == 0
        or alignment > BS_SWAP_RP32_MAX_ALIGNMENT
        or m > BS_SWAP_RP32_MAX_ROWS
    ):
        return None
    if k >= BS_SWAP_RP32_S11_MIN_K and (
        2 * blocks * (n // 256) <= sm_count or blocks <= BS_SWAP_SM100_S11_MAX_BLOCKS
    ):
        return BS_SWAP_RP32_S11_ROUTE
    return BS_SWAP_C1_ROUTE


def bs_swap_small_route(
    m: int, n: int, k: int, *, alignment: int, arch: str, sm_count: int | None
) -> str | None:
    """Select a small-block swap-AB route, or ``None`` for the other swap-AB routes.

    Only with a known ``sm_count``, N a multiple of 128, K a multiple of 128
    and an alignment that is a multiple of 32.  On ``BS_SWAP_SMALL_ARCHS``
    only while the launch is one wave:

    - ``BS_SWAP_RP32_S11_ROUTE`` for N a multiple of 256 and
      K >= ``BS_SWAP_RP32_S11_MIN_K`` when
      ``2 * ceil(m / 32) * n / 256 <= sm_count`` (two CTAs per 32 x 256 tile);
    - ``BS_SWAP_C1_ROUTE`` when ``ceil(m / 32) * n / 128 <= sm_count`` (one
      CTA per 32 x 128 tile): for N a multiple of 256, an alignment of
      ``BS_SWAP_C1_ALIGNMENT`` rows, ``BS_SWAP_C1_MIN_K <= k <=
      BS_SWAP_C1_MAX_K`` and ``n <= BS_SWAP_C1_MAX_N``; for N an odd
      multiple of 128, an alignment that is not a multiple of 128 (the
      multi-run layouts) and ``BS_SWAP_C1_N128_MIN_K <= k <=
      BS_SWAP_C1_N128_MAX_K``.

    On ``BS_SWAP_SM100_SMALL_ARCHS`` (same preconditions) a launch may take
    up to ``BS_SWAP_SM100_SMALL_TILES_PER_SM`` one-CTA tiles per SM, at K of
    at most ``BS_SWAP_SM100_SMALL_MAX_K``:

    - for N a multiple of 256, the multi-run layouts with an alignment of at
      most ``BS_SWAP_RP32_MAX_ALIGNMENT`` and ``m`` of at most
      ``BS_SWAP_RP32_MAX_ROWS``: ``BS_SWAP_RP32_S11_ROUTE`` at
      K >= ``BS_SWAP_RP32_S11_MIN_K`` when ``2 * ceil(m / 32) * n / 256 <=
      sm_count`` or ``ceil(m / 32) <= BS_SWAP_SM100_S11_MAX_BLOCKS``,
      ``BS_SWAP_C1_ROUTE`` otherwise;
    - for N an odd multiple of 128, any alignment at
      ``BS_SWAP_C1_N128_MIN_K <= k <= BS_SWAP_C1_N128_MAX_K``:
      ``BS_SWAP_C1_ROUTE``.

    On ``BS_SWAP_SM100_SMALL_ARCHS`` the layouts of ``BS_SWAP_SM100_SMALL_KEEP``
    keep the other swap-AB routes.

    Every 32-row block of a layout whose alignment is a multiple of 32 holds
    one group's rows, so both programs serve the single-run and the multi-run
    layouts.
    """
    if (
        (arch not in BS_SWAP_SMALL_ARCHS and arch not in BS_SWAP_SM100_SMALL_ARCHS)
        or sm_count is None
        or n % 128
        or k % BS_BLOCK_K
        or alignment % BS_SUB_ROWS
    ):
        return None
    if arch in BS_SWAP_SM100_SMALL_ARCHS:
        return _bs_swap_small_route_sm100(
            m, n, k, alignment=alignment, sm_count=sm_count
        )
    blocks = math.ceil(m / BS_SUB_ROWS)
    if n % 256:
        if (
            alignment % BS_BLOCK_M
            and BS_SWAP_C1_N128_MIN_K <= k <= BS_SWAP_C1_N128_MAX_K
            and blocks * (n // 128) <= sm_count
        ):
            return BS_SWAP_C1_ROUTE
        return None
    if k >= BS_SWAP_RP32_S11_MIN_K and 2 * blocks * (n // 256) <= sm_count:
        return BS_SWAP_RP32_S11_ROUTE
    if (
        alignment == BS_SWAP_C1_ALIGNMENT
        and BS_SWAP_C1_MIN_K <= k <= BS_SWAP_C1_MAX_K
        and n <= BS_SWAP_C1_MAX_N
        and blocks * (n // 128) <= sm_count
    ):
        return BS_SWAP_C1_ROUTE
    return None


def bs_swap_route_for_shape(
    m: int,
    n: int,
    k: int,
    *,
    alignment: int,
    num_groups: int | None,
    arch: str,
    sm_count: int | None = None,
) -> str | None:
    """Select the swap-AB route for packed UE8M0 scales, or ``None`` for :func:`bs_route_for_shape`.

    The rule reads host-known values only.  ``rows`` estimates the routed rows
    per group, ``(m - num_groups * (alignment - 1)) / num_groups``: the compact
    layout less each group's largest alignment padding.

    - Take the route of :func:`bs_swap_small_route` when it selects one
      (``BS_SWAP_SMALL_ARCHS`` one wave of CTAs, ``BS_SWAP_SM100_SMALL_ARCHS``
      up to ``BS_SWAP_SM100_SMALL_TILES_PER_SM`` tiles per SM; any group
      count; its one-CTA program also takes N an odd multiple of 128).
    - Otherwise keep the non-swapped routes when N is not a multiple of 256
      (the swapped tile height) or the group count is unknown.
    - Swap every layout whose alignment is not a multiple of 128 (the 32-row
      multi-run layouts) through ``BS_SWAP_RUN32_ROUTE``; on
      ``BS_SWAP_RP32_ARCHS``, an alignment of at most
      ``BS_SWAP_RP32_MAX_ALIGNMENT`` with ``m`` at most
      ``BS_SWAP_RP32_MAX_ROWS`` takes ``BS_SWAP_RP32_ROUTE`` instead when its
      tiles meet the architecture's ``BS_SWAP_RP32_TILE_BOUND``
      (``factor * ceil(m / rows) * n / 256 <= sm_count``); with ``sm_count``
      ``None`` it keeps ``BS_SWAP_RUN32_ROUTE``.
    - On ``BS_SWAP_RP32_DENSE_ARCHS`` a single-run layout of more than
      ``BS_SWAP_RP32_DENSE_MIN_ROWS`` rows takes ``BS_SWAP_RP32_ROUTE`` when
      ``2 * ceil(m / 32) * n / 256 <= sm_count`` (one pass of two-CTA
      clusters); with ``sm_count`` ``None`` the rules below apply.
    - For K <= 512, swap at 2 <= rows <= 20, and at 1 <= rows < 2 when K <= 256.
    - For longer K, keep below 0.75 rows and swap below 100 rows.  From 100
      rows, sm_100a swaps for N >= 1024, or for K >= 2048 when the BLOCK_N
      128 grid, ``ceil(m / 128) * n / 128`` tiles, is at most
      ``BS_SWAP_SM100_MAX_TILES``; sm_103a swaps for N < 1024 or at least 400
      rows.

    Swapped single-run layouts with K <= 256, or K <= 2048 and at least 128
    rows, take ``BS_SWAP_CARRY_ROUTE``: it loads the next tile's group entries
    at the head of each tile, so the load overlaps a tile of work.  The other
    swapped layouts read each tile's entries at its head.

    The choice never changes the output bits: every route accumulates each
    output in the same K order.
    """
    if arch not in BS_SWAP_ARCHS or n % 128 or k % BS_BLOCK_K:
        return None
    small = bs_swap_small_route(
        m, n, k, alignment=alignment, arch=arch, sm_count=sm_count
    )
    if small is not None:
        return small
    if n % 256:
        return None
    single_run = alignment % BS_BLOCK_M == 0
    if not single_run:
        if (
            arch in BS_SWAP_RP32_ARCHS
            and alignment <= BS_SWAP_RP32_MAX_ALIGNMENT
            and m <= BS_SWAP_RP32_MAX_ROWS
            and sm_count is not None
        ):
            block_rows, factor = BS_SWAP_RP32_TILE_BOUND[arch]
            if factor * math.ceil(m / block_rows) * (n // 256) <= sm_count:
                return BS_SWAP_RP32_ROUTE
        return BS_SWAP_RUN32_ROUTE
    if not num_groups or num_groups <= 0:
        return None
    # rows = routed / num_groups; compared without division.
    routed = m - num_groups * (alignment - 1)
    if (
        arch in BS_SWAP_RP32_DENSE_ARCHS
        and sm_count is not None
        and routed > BS_SWAP_RP32_DENSE_MIN_ROWS * num_groups
        and 2 * math.ceil(m / BS_SUB_ROWS) * (n // 256) <= sm_count
    ):
        return BS_SWAP_RP32_ROUTE
    if k <= 512:
        swap = 2 * num_groups <= routed <= 20 * num_groups or (
            k <= 256 and num_groups <= routed < 2 * num_groups
        )
    elif 4 * routed < 3 * num_groups:
        swap = False
    elif routed < 100 * num_groups:
        swap = True
    elif arch == "sm_100a":
        tiles = -(-m // BS_BLOCK_M) * (n // 128)
        swap = n >= 1024 or (k >= 2048 and tiles <= BS_SWAP_SM100_MAX_TILES)
    else:
        swap = n < 1024 or routed >= 400 * num_groups
    if not swap:
        return None
    if k <= 256 or (k <= 2048 and routed >= 128 * num_groups):
        return BS_SWAP_CARRY_ROUTE
    return BS_SWAP_ROUTE


# The store16 programs: the single-run schedules with 16-row output stores.
BS_STORE16_ROUTES = {
    BS_N128_ROUTE: BS_N128_S16_ROUTE,
    BS_N256_ROUTE: BS_N256_S16_ROUTE,
    BS_SWAP_ROUTE: BS_SWAP_S16_ROUTE,
    BS_SWAP_CARRY_ROUTE: BS_SWAP_CARRY_S16_ROUTE,
}
BS_STORE16_ARCHS = frozenset({"sm_100a", "sm_103a"})


def bs_store16(m: int, *, alignment: int, num_groups: int | None) -> bool:
    """Whether a single-run block-scaled problem takes the store16 programs.

    The rule reads host-known values only, never the ``m_indices`` contents (a
    launch may rebind them): ``alignment`` a multiple of 128 and at most 128
    routed rows per group, estimated as in :func:`bs_swap_route_for_shape` by
    ``(m - num_groups * (alignment - 1)) / num_groups``.  A store16 program
    writes each expert run's routed rows rounded up to 16 rows instead of 32;
    the routed rows keep their bits.  Above 128 rows per group the 32-row
    stores are kept.
    """
    if not num_groups or num_groups <= 0 or alignment <= 0 or alignment % BS_BLOCK_M:
        return False
    # rows = routed / num_groups <= 128; compared without division.
    return m - num_groups * (alignment - 1) <= BS_BLOCK_M * num_groups


def resolve_route_for_grid(route: str, n: int, grid: tuple[int, int, int]) -> str:
    """Pick the exported program of a route for the grid it will launch with.

    The four-load schedule exists only as its 128-CTA instantiation; a device
    with fewer than 128 SMs (grid below 128 CTAs) runs the recurrence schedule
    instead, which has the same tile geometry and an exported program at every
    grid.
    """
    if route == DEEPK_CG2_FOUR_LOAD_ROUTE and n == 4096:
        if grid == (128, 1, 1):
            return DEEPK_CG2_FOUR_LOAD_GRID128_ROUTE
        if grid[0] < 128:
            return DEEPK_CG2_RECURRENCE_ROUTE
    return route


def route_geometry(route: str) -> tuple[int, int, int]:
    """``(tile_m, tile_n, cluster_ctas)`` of one route from the export registry."""
    geometry = ROUTE_GEOMETRY.get(route)
    if geometry is None:
        raise NotImplementedError(
            f"route {route!r} has no registered tile geometry in this checkout"
        )
    return (
        int(geometry["tile_m"]),
        int(geometry["tile_n"]),
        int(geometry["cluster_ctas"]),
    )


def launch_plan(
    m: int,
    n: int,
    k: int,
    *,
    sm_count: int,
    scalar_output: bool,
    block_scaled: bool = False,
    alignment: int = 128,
    num_groups: int | None = None,
    arch: str | None = None,
) -> tuple[str, tuple[int, int, int]]:
    """Resolve ``(route, grid)`` for a problem shape on a device with ``sm_count`` SMs.

    ``arch`` (the device's, e.g. ``"sm_100a"``) admits the swap-AB routes of
    :func:`bs_swap_route_for_shape` and the store16 programs of
    :func:`bs_store16` for a block-scaled problem; ``None`` keeps the routes of
    :func:`bs_route_for_shape`.  An FP32-scale problem takes :func:`_route`
    with the SM count.  A KG1 / AB6 launch of at most ``DIRECT_STORE_MAX_ROWS``
    rows on at most ``DIRECT_STORE_MAX_CTAS`` CTAs runs that route's
    direct-store program; a KG1 launch of more rows whose 128 x 64 tiles fit
    in one wave runs the KG1 program with 128 x 64 tiles.
    """
    return _launch_plan(
        m,
        n,
        k,
        sm_count=sm_count,
        scalar_output=scalar_output,
        block_scaled=block_scaled,
        alignment=alignment,
        num_groups=num_groups,
        arch=arch,
        aligned=None,
    )


def _launch_plan(
    m: int,
    n: int,
    k: int,
    *,
    sm_count: int,
    scalar_output: bool,
    block_scaled: bool,
    alignment: int,
    num_groups: int | None,
    arch: str | None,
    aligned: Optional[bool],
    tile_starts: Optional[Mapping[int, Sequence[int]]] = None,
) -> tuple[str, tuple[int, int, int]]:
    """:func:`launch_plan` with the layout a validated launch read (``aligned``
    and ``tile_starts``, see :func:`_route`)."""
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if block_scaled:
        if scalar_output:
            raise ValueError(
                "block-scaled grouped FP8 GEMM requires a 16-byte-aligned output"
            )
        route = bs_route_for_shape(
            m, n, k, alignment=alignment, sm_count=sm_count, num_groups=num_groups
        )
        if arch is not None:
            route = (
                bs_swap_route_for_shape(
                    m,
                    n,
                    k,
                    alignment=alignment,
                    num_groups=num_groups,
                    arch=arch,
                    sm_count=sm_count,
                )
                or route
            )
            if (
                arch in BS_STORE16_ARCHS
                and route in BS_STORE16_ROUTES
                and bs_store16(m, alignment=alignment, num_groups=num_groups)
            ):
                route = BS_STORE16_ROUTES[route]
    elif scalar_output:
        route = DEEPK_C2_SCALAR_OUTPUT_ROUTE
    else:
        route = _route(
            m, n, k, sm_count=sm_count, aligned=aligned, tile_starts=tile_starts
        )
    cluster_ctas = route_geometry(route)[2]
    grid = (_route_workers(route, m, n, k, sm_count) * cluster_ctas, 1, 1)
    route = resolve_route_for_grid(route, n, grid)
    if m <= DIRECT_STORE_MAX_ROWS and grid[0] <= DIRECT_STORE_MAX_CTAS:
        route = _DIRECT_STORE_ROUTE.get(route, route)
    elif route == DEEPK_C2_KG1_ROUTE and m > DIRECT_STORE_MAX_ROWS:
        n64_tiles = math.ceil(m / 128) * math.ceil(n / 64)
        if n64_tiles <= sm_count:
            route, grid = DEEPK_C2_KG1_N64_ROUTE, (n64_tiles, 1, 1)
    return route, grid


def _route_workers(route: str, m: int, n: int, k: int, sm_count: int) -> int:
    """Workers (CTAs, or clusters of a two-CTA route) of a ``route`` launch."""
    tile_m, tile_n, cluster_ctas = route_geometry(route)
    total_tiles = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    cta_limit = sm_count
    # The cap keeps every shipped plan; the swap-AB routes and the single-CTA
    # routes of a known unaligned layout at these shapes run on all SMs.
    if (
        (m, n, k) in _CG2_GRID128_SHAPES
        and route not in BS_SWAP_ROUTES
        and route not in _CG2_UNALIGNED_ROUTE.values()
    ):
        cta_limit = min(cta_limit, 128)
    clusters = min(total_tiles, cta_limit // cluster_ctas)
    if clusters <= 0:
        raise ValueError("device cannot schedule one complete kernel cluster")
    return clusters


@functools.cache
def device_sm_count(device_index: int) -> int:
    """Streaming-multiprocessor count of CUDA device ``device_index`` (queried once)."""
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _require_tensor(
    tensor: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    alignment: int = 16,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if int(tensor.data_ptr()) % alignment:
        raise ValueError(f"{name} storage must be at least {alignment}-byte aligned")
    return tensor


def _validate_indices(
    m_indices: torch.Tensor,
    groups: int,
    *,
    allow_padding: bool = False,
    layout_tile_m: Optional[int] = None,
    layout_tile_rows: int = 128,
    read_alignment: bool = False,
) -> tuple[Optional[tuple[int, int, int, int]], Optional[bool]]:
    """Check the routing contract (in range, sorted) with one device-to-host transfer.

    With ``allow_padding`` the value ``-1`` marks padding rows; the remaining
    entries must still be nondecreasing.  Returns ``(layout, aligned)``, both
    ``None`` unless asked for; the caller uses them only to choose a route.
    With ``layout_tile_m`` (no padding) the same transfer also reads
    ``layout = (inner_starts, run_starts, expert_tiles, max_offset)``: the
    number of expert runs starting at a row that is not a multiple of
    ``layout_tile_m`` (so a ``layout_tile_m``-row tile spans two experts), the
    number of expert runs starting after row 0, the number of
    ``layout_tile_rows``-row tiles when no tile spans two experts, and the
    largest offset of such an inner start within its tile (0 without one).
    With ``read_alignment`` (no padding) it also reads ``aligned``: whether
    every expert run starts on a multiple of 256 rows (see :func:`_route`).
    """
    indices = m_indices.to(torch.int64)
    routed = indices[indices >= 0] if allow_padding else indices
    unsorted = (
        (routed[1:] < routed[:-1]).sum()
        if routed.numel() > 1
        else torch.zeros((), dtype=torch.int64, device=indices.device)
    )
    stats = [indices.min(), indices.max(), unsorted]
    if layout_tile_m is not None or read_alignment:
        if allow_padding:
            raise ValueError("the index layout is read only without padding rows")
        rows = torch.arange(
            1, indices.numel(), dtype=torch.int64, device=indices.device
        )
        run_starts = indices[1:] != indices[:-1]
    if read_alignment:
        stats.append((run_starts & (rows % _CG2_TILE_M != 0)).sum())
    if layout_tile_m is not None:
        zero = torch.zeros((), dtype=torch.int64, device=indices.device)
        if indices.numel() > 1:
            offset = rows % layout_tile_m
            inner = run_starts & (offset != 0)
            # First row of the run each row belongs to (0 for the leading run).
            run_first = torch.cummax(
                torch.where(run_starts, rows, torch.zeros_like(rows)), dim=0
            ).values
            stats.append(inner.sum())
            stats.append(run_starts.sum())
            stats.append(((rows - run_first) % layout_tile_rows == 0).sum() + 1)
            stats.append(torch.where(inner, offset, torch.zeros_like(offset)).max())
        else:
            stats += [zero, zero, zero + 1, zero]
    values = torch.stack(stats).tolist()
    lowest, highest, unsorted = values[:3]
    if lowest < (-1 if allow_padding else 0) or highest >= groups:
        if allow_padding:
            raise ValueError("m_indices must satisfy -1 <= index < num_groups")
        raise ValueError(
            "m_indices must satisfy 0 <= index < num_groups; -1 padding needs fill_padding=True"
        )
    if unsorted:
        raise ValueError("m_indices must be sorted in nondecreasing order")
    extra = values[3:]
    aligned: Optional[bool] = None
    if read_alignment:
        aligned, extra = extra[0] == 0, extra[1:]
    layout: Optional[tuple[int, int, int, int]] = None
    if layout_tile_m is not None:
        inner, starts, expert_tiles, max_offset = (int(v) for v in extra[:4])
        layout = inner, starts, expert_tiles, max_offset
    return layout, aligned


def _tile_starts(m_indices: torch.Tensor) -> dict[int, tuple[int, ...]]:
    """Per M-tile height in ``_TILE_START_ROWS``: how many non-empty experts start
    inside each M tile, off its first row (one device-to-host transfer).

    Read only for a validated layout that is not aligned to 256 rows at a shape
    of ``_CG2_SHAPE_ROUTE`` (see :func:`_unaligned_cg2_route`).
    """
    indices = m_indices.to(torch.int64)
    m = indices.numel()
    rows = torch.arange(1, m, dtype=torch.int64, device=indices.device)
    run_starts = indices[1:] != indices[:-1]
    counts: list[torch.Tensor] = []
    for tile_m in _TILE_START_ROWS:
        inner = (run_starts & (rows % tile_m != 0)).to(torch.int64)
        per_tile = torch.zeros(
            math.ceil(m / tile_m), dtype=torch.int64, device=indices.device
        )
        counts.append(per_tile.index_add_(0, rows // tile_m, inner))
    values = torch.cat(counts).tolist()
    out: dict[int, tuple[int, ...]] = {}
    first = 0
    for tile_m in _TILE_START_ROWS:
        tiles = math.ceil(m / tile_m)
        out[tile_m] = tuple(int(v) for v in values[first : first + tiles])
        first += tiles
    return out


def _route_registered(arch: str, route: str) -> bool:
    try:
        route_program(arch, route)
    except NotImplementedError:
        return False
    return True


def _expert_aligned_eligible(
    arch: str,
    groups: int,
    *,
    block_scaled: bool,
    fill_padding: bool,
    scalar_output: bool,
) -> bool:
    """True when the expert-aligned program can serve this launch (FP32 scales, no
    padding rows, a 16-byte-aligned output, at most ``EXPERT_ALIGNED_MAX_GROUPS``
    experts, a program registered for ``arch``)."""
    return (
        not block_scaled
        and not fill_padding
        and not scalar_output
        and groups <= EXPERT_ALIGNED_MAX_GROUPS
        and _route_registered(arch, DEEPK_C2_KG1_EXPERT_ALIGNED_ROUTE)
    )


def _expert_aligned_preferred(
    inner_starts: int, run_starts: int, max_offset: int, tile_m: int
) -> bool:
    """True when at least 90% of the expert runs after row 0 start inside a tile
    of the shape's route and one of them starts at least half a tile in.

    The shape's route computes a tile spanning several experts once per expert
    and finds where each run ends by reading ``m_indices`` row by row; the
    expert-aligned program does neither, but its own schedule is slower than
    several shape routes on homogeneous tiles and it needs more tiles when
    runs start just past a tile boundary.  Measured on B200 and both GB300
    classes, it wins on such layouts (routed tokens put nearly every run start
    inside a tile, at any offset); with half the runs inside a tile, or with
    every inner run starting one row in, it can lose.
    """
    return (
        inner_starts > 0
        and 10 * inner_starts >= 9 * run_starts
        and 2 * max_offset >= tile_m
    )


def _expert_aligned_plan(
    n: int, expert_tiles: int, *, sm_count: int
) -> tuple[str, tuple[int, int, int]]:
    """``(route, grid)`` of the expert-aligned program for ``expert_tiles`` M tiles.

    The program loops over its tiles persistently and counts them on the
    device, so the grid only sizes the launch: a later ``m_indices`` with more
    or fewer tiles stays correct.
    """
    _, tile_n, _ = route_geometry(DEEPK_C2_KG1_EXPERT_ALIGNED_ROUTE)
    ctas = max(1, min(expert_tiles * math.ceil(n / tile_n), sm_count))
    return DEEPK_C2_KG1_EXPERT_ALIGNED_ROUTE, (ctas, 1, 1)


def _packed_scale_strides(
    scale: torch.Tensor, name: str, *, rows: int, k_blocks: int, device: torch.device
) -> tuple[torch.Tensor, int, int]:
    """``(ffi tensor, row stride, col stride)`` of a packed UE8M0 activation-scale tensor.

    Any strides are accepted (the kernel reads ``base[row * rs + col * cs]``); the FFI
    binding needs a contiguous tensor sharing the storage pointer, so the MN-major
    layout is passed as its contiguous transpose.
    """
    if not isinstance(scale, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if scale.device != device or scale.dtype != torch.int32:
        raise ValueError(f"{name} must be an int32 tensor on {device}")
    if (
        scale.ndim != 2
        or int(scale.shape[0]) != rows
        or 4 * int(scale.shape[1]) < k_blocks
    ):
        raise ValueError(
            f"{name} must be packed UE8M0 of shape ({rows}, >= {-(-k_blocks // 4)}), got {tuple(scale.shape)}"
        )
    if int(scale.data_ptr()) % 16:
        raise ValueError(f"{name} storage must be at least 16-byte aligned")
    rs, cs = (int(v) for v in scale.stride())
    if scale.is_contiguous():
        return scale, rs, cs
    if scale.mT.is_contiguous():
        return scale.mT, rs, cs
    raise ValueError(
        f"{name} must be contiguous or the transpose of a contiguous tensor"
    )


def _packed_weight_scale_strides(
    scale: torch.Tensor, *, groups: int, n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, int, int, int]:
    """``(ffi tensor, group stride, 128-row-block stride, col stride)`` of packed UE8M0 weight scales."""
    if not isinstance(scale, torch.Tensor):
        raise ValueError("b_scale must be a torch.Tensor")
    if scale.device != device or scale.dtype != torch.int32:
        raise ValueError(f"b_scale must be an int32 tensor on {device}")
    n_blocks, k_blocks = n // B_SCALE_BLOCK_N, k // K_BLOCK
    if (
        scale.ndim != 3
        or int(scale.shape[0]) != groups
        or 4 * int(scale.shape[2]) < k_blocks
    ):
        raise ValueError(
            f"b_scale must be packed UE8M0 of shape ({groups}, {n_blocks} or {n}, >= {-(-k_blocks // 4)}), "
            f"got {tuple(scale.shape)}"
        )
    if int(scale.data_ptr()) % 16:
        raise ValueError("b_scale storage must be at least 16-byte aligned")
    rows = int(scale.shape[1])
    gs, rs, cs = (int(v) for v in scale.stride())
    if rows == n:
        rs *= B_SCALE_BLOCK_N
    elif rows != n_blocks:
        raise ValueError(
            f"b_scale must have {n_blocks} or {n} rows per group, got {rows}"
        )
    if scale.is_contiguous():
        return scale, gs, rs, cs
    if scale.mT.is_contiguous():
        return scale.mT, gs, rs, cs
    raise ValueError(
        "b_scale must be contiguous or the transpose of a contiguous tensor"
    )


def _swap_scale_surfaces(
    a_scale: torch.Tensor, b_scale: torch.Tensor, *, m: int, n: int, k: int
) -> Optional[tuple[torch.Tensor, torch.Tensor, int]]:
    """The packed scale words as the swap-AB programs' tensor-map surfaces, or ``None``.

    The swap-AB programs load the activation words as the uint32 surface
    ``[cols, M]`` (DeepGEMM's MN-major layout: ``a_scale`` the transpose of a
    contiguous tensor) and the weight words as ``[G * cols_b, N]`` (the
    row-repeated ``(G, N, cols_b)`` layout, also the transpose of a contiguous
    tensor), both read in place; the third value is ``cols_b``, the words per
    group.  Any other layout, and an ``M`` that is not a multiple of 128 rows,
    returns ``None``: the non-swapped routes read it as it is.
    """
    cols = -(-(k // K_BLOCK) // 4)
    groups, rows, cols_b = (int(v) for v in b_scale.shape)
    if m % BS_BLOCK_M or n % 4 or rows != n:
        return None
    if not (a_scale.mT.is_contiguous() and b_scale.mT.is_contiguous()):
        return None
    sfb = b_scale.as_strided(((groups - 1) * cols_b + cols, n), (n, 1))
    return _swap_sfa(a_scale, m=m, k=k), sfb.view(torch.uint32), cols_b


def _swap_sfa(a_scale: torch.Tensor, *, m: int, k: int) -> torch.Tensor:
    """The MN-major activation words as the uint32 surface ``[cols, M]`` with word stride ``M``."""
    cols = -(-(k // K_BLOCK) // 4)
    return a_scale.as_strided((cols, m), (m, 1)).view(torch.uint32)


def argument_slots(arg_plan: Sequence[Sequence[str]]) -> dict[str, tuple[int, ...]]:
    """Positions of every named binding in a generated program's argument plan."""
    slots: dict[str, list[int]] = {}
    for index, (kind, name) in enumerate(arg_plan):
        if kind != "grid":
            slots.setdefault(name, []).append(index)
    return {name: tuple(positions) for name, positions in slots.items()}


def _forward_fill_padding(
    m_indices: torch.Tensor, filled: torch.Tensor, positions: torch.Tensor
) -> None:
    """``filled[r] = max(0, max(m_indices[:r + 1]))``: padding rows follow the preceding expert.

    Two small allocation-free kernels (``cummax`` into prepared scratch, then
    ``clamp``); both are graph-capturable.  Leading padding maps onto expert 0.
    """
    torch.cummax(m_indices, 0, out=(filled, positions))
    filled.clamp_(min=0)


@dataclass
class PreparedGroupGemmFp8NtGroupwiseContiguous:
    """One prepared contiguous grouped FP8 GEMM launch.

    Shapes, dtypes and the expert weights (``b``, ``b_scale``) are bound at
    preparation; tensor *contents* may change between launches, and the
    per-token operands ``a``, ``a_scale``, ``m_indices`` and ``out`` may be
    replaced by same-shape tensors at every ``launch`` (``launch(a=..., ...)``),
    so a caller can run one prepared object per layer on whatever buffers the
    dispatcher produced.  ``launch()`` submits exactly one kernel (plus two
    tiny index kernels when ``fill_padding`` is on) on PyTorch's current stream
    for the bound device and returns the output tensor.  Every launch,
    including the first, may be captured into a CUDA graph.  The prepared
    object keeps the operands given at preparation alive (so a bare
    ``launch()`` works); a long-lived plan that always launches on the caller's
    buffers calls ``release_prepared_operands()`` once, after which it holds no
    per-token device memory and every ``launch`` must supply ``a``,
    ``a_scale``, ``m_indices`` and ``out``.
    """

    route: str
    module_name: str
    grid: tuple[int, int, int]
    device: torch.device
    _out: Optional[torch.Tensor]
    m: int
    n: int
    k: int
    groups: int
    fill_padding: bool
    _entry: Callable[..., Any]
    _arguments: tuple[Any, ...]
    _slots: dict[str, tuple[int, ...]]
    _padding_source: Optional[torch.Tensor]
    _padding_scratch: Optional[tuple[torch.Tensor, torch.Tensor]]
    # Block-scaled family: the prepared activation-scale layout (shape, strides); a
    # replacement a_scale must match it because the strides are bound as parameters.
    _a_scale_layout: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None
    _released: bool = False

    @property
    def block_scaled(self) -> bool:
        return self.route in BS_ROUTES or self.route in BS_SWAP_ROUTES

    @property
    def out(self) -> torch.Tensor:
        """The output tensor bound at preparation (unavailable after ``release_prepared_operands``)."""
        if self._out is None:
            raise ValueError(
                "the prepared operands were released; launch(a=..., a_scale=..., "
                "m_indices=..., out=...) returns the caller's output"
            )
        return self._out

    def release_prepared_operands(self) -> None:
        """Drop the references to the per-token operands bound at preparation.

        ``a``, ``a_scale``, ``out`` and (unless ``fill_padding`` keeps the
        prepared ``m_indices`` as its forward-fill source) ``m_indices`` are
        released so that a plan kept per layer does not pin a dispatcher's
        buffers.  Afterwards every ``launch`` must supply all of them.
        """
        arguments = list(self._arguments)
        names: tuple[str, ...] = (
            "A",
            "A64",
            "A32",
            "C",
            "C_tma",
            "a_scale",
            "SFA",
            "SFA_packed",
        )
        if not self.fill_padding:
            names += ("m_indices",)
        for name in names:
            self._rebind(arguments, name, None)
        self._arguments = tuple(arguments)
        self._out = None
        self._released = True

    def _rebind(self, arguments: list[Any], name: str, value: Any) -> None:
        for position in self._slots.get(name, ()):
            arguments[position] = value

    def launch(
        self,
        a: Optional[torch.Tensor] = None,
        a_scale: Optional[torch.Tensor] = None,
        m_indices: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the GEMM; any operand given here replaces the prepared one for this call.

        Replacement tensors must match the prepared shape, dtype, device and
        alignment class (an output that is 16-byte aligned cannot replace a
        2-byte-aligned one and vice versa, because the store route is fixed at
        preparation).  Rebinding allocates nothing on the device.
        """
        device = self.device
        arguments = self._arguments
        scalar = self.route == DEEPK_C2_SCALAR_OUTPUT_ROUTE
        if self._released and (
            a is None
            or a_scale is None
            or out is None
            or (m_indices is None and not self.fill_padding)
        ):
            raise ValueError(
                "the prepared operands were released; launch must receive a, "
                "a_scale, m_indices and out"
            )
        if (
            a is not None
            or a_scale is not None
            or m_indices is not None
            or out is not None
        ):
            rebound = list(arguments)
            if a is not None:
                a = _require_tensor(
                    a,
                    "a",
                    shape=(self.m, self.k),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                a_u8 = a.view(torch.uint8)
                for name in ("A", "A64", "A32"):
                    self._rebind(rebound, name, a_u8)
                if scalar:
                    self._rebind(rebound, "C_tma", a.view(torch.bfloat16))
            if a_scale is not None:
                if self.block_scaled:
                    ffi_scale, rs, cs = _packed_scale_strides(
                        a_scale,
                        "a_scale",
                        rows=self.m,
                        k_blocks=self.k // K_BLOCK,
                        device=device,
                    )
                    layout = (tuple(int(v) for v in a_scale.shape), (rs, cs))
                    if layout != self._a_scale_layout:
                        raise ValueError(
                            f"a_scale must keep the prepared packed layout {self._a_scale_layout}, got {layout}"
                        )
                    self._rebind(rebound, "SFA", ffi_scale)
                    if self.route in BS_SWAP_ROUTES:
                        self._rebind(
                            rebound,
                            "SFA_packed",
                            _swap_sfa(a_scale, m=self.m, k=self.k),
                        )
                else:
                    a_scale = _require_tensor(
                        a_scale,
                        "a_scale",
                        shape=(self.m, self.k // K_BLOCK),
                        dtype=torch.float32,
                        device=device,
                    )
                    self._rebind(rebound, "a_scale", a_scale)
            if m_indices is not None:
                m_indices = _require_tensor(
                    m_indices,
                    "m_indices",
                    shape=(self.m,),
                    dtype=torch.int32,
                    device=device,
                )
                if not self.fill_padding:
                    self._rebind(rebound, "m_indices", m_indices)
            if out is not None:
                out = _require_tensor(
                    out,
                    "out",
                    shape=(self.m, self.n),
                    dtype=torch.bfloat16,
                    device=device,
                    alignment=2,
                )
                if bool(int(out.data_ptr()) % 16) != scalar:
                    raise ValueError(
                        "out must keep the alignment class of the prepared output "
                        "(16-byte aligned or not); the store route is fixed at preparation"
                    )
                self._rebind(rebound, "C", out)
                if not scalar:
                    self._rebind(rebound, "C_tma", out)
                if self.block_scaled and int(out.data_ptr()) % 16:
                    raise ValueError(
                        "block-scaled grouped FP8 GEMM requires a 16-byte-aligned output"
                    )
            arguments = tuple(rebound)
        with torch.cuda.device(device):
            if self.fill_padding:
                assert self._padding_scratch is not None
                source = m_indices if m_indices is not None else self._padding_source
                assert source is not None
                _forward_fill_padding(source, *self._padding_scratch)
            with tvm_ffi.use_torch_stream():
                self._entry(*arguments)
        return self._out if out is None else out

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.grid[0])


def is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available(
    device: torch.device,
) -> bool:
    """True when a generated program family is registered for ``device``."""
    return device.type == "cuda" and generated_program_available(device)


def bind_arguments(
    arg_plan: list[list[str]],
    bindings: dict[str, Any],
    grid: tuple[int, int, int],
    *,
    module_name: str,
) -> tuple[Any, ...]:
    """Order ``bindings`` by the generated program's positional argument plan."""
    grid_by_axis = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in arg_plan:
        if kind == "grid":
            arguments.append(grid_by_axis[name])
        elif kind in ("tma_buffer", "buffer", "parameter") and name in bindings:
            arguments.append(bindings[name])
        else:
            raise RuntimeError(
                f"generated program {module_name} binds {kind} {name!r}, which this host plan does not declare"
            )
    return tuple(arguments)


def prepare_group_gemm_fp8_nt_groupwise_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    validate_indices: bool = False,
    fill_padding: bool = False,
    alignment: int = 128,
) -> PreparedGroupGemmFp8NtGroupwiseContiguous:
    r"""Prepare a contiguous grouped FP8 GEMM launch on SM100a / SM103a.

    Parameters
    ----------
    a : torch.Tensor
        Contiguous FP8 E4M3 input of shape ``(M, K)``; ``M > 0`` and ``K`` a
        positive multiple of 128.
    b : torch.Tensor
        Contiguous FP8 E4M3 expert weights of shape ``(G, N, K)``; ``N`` a
        positive multiple of 128.
    a_scale : torch.Tensor
        Contiguous float32 scales of shape ``(M, K // 128)`` (promotion family),
        or packed UE8M0 int32 words of shape ``(M, >= ceil(K / 512))`` with any
        strides (block-scaled family; DeepGEMM's MN-major layout is read in
        place).
    b_scale : torch.Tensor
        Contiguous float32 scales of shape ``(G, N // 128, K // 128)``, or
        packed UE8M0 int32 words of shape ``(G, N // 128, cols)`` or the
        row-repeated ``(G, N, cols)`` (``4 * cols >= K // 128``) when
        ``a_scale`` is packed.
    m_indices : torch.Tensor
        Contiguous int32 expert indices of shape ``(M,)``, sorted in
        nondecreasing order with ``0 <= index < G``.  Expert boundaries may
        fall at any row; empty experts are allowed.  With ``fill_padding`` the
        value ``-1`` marks padding rows (the compact MoE layout): such a row is
        computed with the preceding expert's weights (expert 0 for leading
        padding) and its output row is unspecified.  The block-scaled family
        reads ``-1`` natively: it writes each expert run's routed rows rounded
        up to 32 rows (16 on the store16 programs of :func:`bs_store16`), and
        the output rows of ``-1`` entries are unspecified (left untouched or
        holding values of no meaning, as DeepGEMM's 128-row blocks do).  Every
        expert run must start on a multiple of ``alignment`` rows (an expert's
        rows followed by its padding).
    out : Optional[torch.Tensor]
        Contiguous bfloat16 output of shape ``(M, N)``.  Allocated if omitted.
        A 2-byte-aligned output whose address is not a multiple of 16 selects
        the scalar-store route.
    validate_indices : bool
        Check index range and sortedness (one device-to-host transfer).  With
        FP32 scales and no ``fill_padding`` the same transfer also reads where
        the expert runs start.  The two-CTA routes then run when every run
        starts on a multiple of 256 rows; at the shapes whose shape route is
        two-CTA, an unaligned layout keeps it unless the single-CTA route
        gives its busiest worker fewer expert segments (one more transfer reads
        the expert starts per tile).  With a 16-byte-aligned output
        and at most 255 experts, when at least 90% of the runs start inside a
        tile of the shape's route and one starts at least half a tile in, the
        launch uses the expert-aligned program, whose tiles never span two
        experts.  These choices affect speed only.  Every program computes the
        same bits for any valid ``m_indices``, so indices that change after
        preparation (rebound at ``launch`` or rewritten before a graph replay)
        stay correct; only the chosen program may be slower for them.
    fill_padding : bool
        Accept ``-1`` padding rows in ``m_indices`` (promotion family).  The
        prepared object owns an ``(M,)`` int32 plus ``(M,)`` int64 scratch and
        every launch runs two small forward-fill kernels before the GEMM
        (graph-capturable).  Must be ``False`` with packed scales.
    alignment : int
        Block-scaled family only: the row alignment of expert runs in the
        compact layout, a positive multiple of 32.  Multiples of 128 run the
        single-run schedule; other values the slower 32-row multi-run schedule.
        The swap-AB routes need ``N`` a multiple of 256 (128 for the one-CTA
        program of :func:`bs_swap_small_route`), ``M`` a multiple of
        128, the MN-major ``a_scale`` words and the row-repeated ``b_scale``
        words; other problems run the non-swapped routes.  The store16
        programs need the same ``M`` and words, an ``alignment`` multiple of
        128 and at most 128 routed rows per group (:func:`bs_store16`).

    Returns
    -------
    PreparedGroupGemmFp8NtGroupwiseContiguous
        Call ``.launch()`` to run the GEMM on the current stream.

    Notes
    -----
    Requires an SM100a (compute capability 10.0) or SM103a (10.3) device and a
    registered generated program for the selected route.  All tensors must
    live on the same CUDA device and, except ``out``, be 16-byte aligned.
    Preparation performs no device work: the SM count of the device is read
    once per process, and the first launch is as graph-capturable as every
    later one.  Index values are unchecked unless ``validate_indices=True``;
    violating the routing contract is undefined behavior.  The per-token
    operands may be swapped per call: ``prepared.launch(a=a2, a_scale=s2,
    m_indices=i2, out=o2)`` (same shapes, dtypes, device and alignment class).
    """
    if not isinstance(a, torch.Tensor) or a.device.type != "cuda":
        raise ValueError("a must be a CUDA torch.Tensor")
    device = a.device
    device_index = _device_index(device)
    arch = device_arch(device_index)
    if arch is None:
        raise NotImplementedError(
            "prepared contiguous grouped FP8 GEMM requires an SM100a or SM103a device "
            f"(compute capability {sorted(SUPPORTED_COMPUTE_CAPABILITIES)}); "
            f"got {torch.cuda.get_device_capability(device_index)}"
        )
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("a must have shape (M, K) and b must have shape (G, N, K)")
    m, k = (int(v) for v in a.shape)
    groups, n = (int(v) for v in b.shape[:2])
    if m <= 0 or groups <= 0:
        raise ValueError("contiguous grouped FP8 GEMM requires positive M and G")
    if n <= 0 or n % B_SCALE_BLOCK_N:
        raise ValueError(f"N must be a positive multiple of {B_SCALE_BLOCK_N}, got {n}")
    if k <= 0 or k % K_BLOCK:
        raise ValueError(f"K must be a positive multiple of {K_BLOCK}, got {k}")
    a = _require_tensor(a, "a", shape=(m, k), dtype=torch.float8_e4m3fn, device=device)
    b = _require_tensor(
        b, "b", shape=(groups, n, k), dtype=torch.float8_e4m3fn, device=device
    )
    block_scaled = isinstance(a_scale, torch.Tensor) and a_scale.dtype == torch.int32
    swap_surfaces = None
    if block_scaled:
        if fill_padding:
            raise ValueError(
                "fill_padding is not needed with packed UE8M0 scales: -1 padding rows are skipped natively"
            )
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
            or alignment % BS_SUB_ROWS
        ):
            raise ValueError(
                f"alignment must be a positive multiple of {BS_SUB_ROWS}, got {alignment!r}"
            )
        sfa, sfa_rs, sfa_cs = _packed_scale_strides(
            a_scale, "a_scale", rows=m, k_blocks=k // K_BLOCK, device=device
        )
        sfb, sfb_gs, sfb_rs, sfb_cs = _packed_weight_scale_strides(
            b_scale, groups=groups, n=n, k=k, device=device
        )
        swap_surfaces = _swap_scale_surfaces(a_scale, b_scale, m=m, n=n, k=k)
    else:
        a_scale = _require_tensor(
            a_scale,
            "a_scale",
            shape=(m, k // K_BLOCK),
            dtype=torch.float32,
            device=device,
        )
        b_scale = _require_tensor(
            b_scale,
            "b_scale",
            shape=(groups, n // B_SCALE_BLOCK_N, k // K_BLOCK),
            dtype=torch.float32,
            device=device,
        )
    m_indices = _require_tensor(
        m_indices, "m_indices", shape=(m,), dtype=torch.int32, device=device
    )
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    out = _require_tensor(
        out, "out", shape=(m, n), dtype=torch.bfloat16, device=device, alignment=2
    )
    scalar_output = bool(int(out.data_ptr()) % 16)
    layout = None
    aligned: Optional[bool] = None
    layout_tile_m: Optional[int] = None
    if validate_indices:
        padded = fill_padding or block_scaled
        layout_tile_rows = 128
        if _expert_aligned_eligible(
            arch,
            groups,
            block_scaled=block_scaled,
            fill_padding=fill_padding,
            scalar_output=scalar_output,
        ):
            layout_tile_m = route_geometry(route_for_shape(m, n, k))[0]
            layout_tile_rows = route_geometry(DEEPK_C2_KG1_EXPERT_ALIGNED_ROUTE)[0]
        layout, aligned = _validate_indices(
            m_indices,
            groups,
            allow_padding=padded,
            layout_tile_m=layout_tile_m,
            layout_tile_rows=layout_tile_rows,
            read_alignment=not padded,
        )
    padding_scratch = None
    kernel_indices = m_indices
    if fill_padding:
        kernel_indices = torch.empty((m,), dtype=torch.int32, device=device)
        padding_scratch = (
            kernel_indices,
            torch.empty((m,), dtype=torch.int64, device=device),
        )

    if block_scaled and scalar_output:
        raise ValueError(
            "block-scaled grouped FP8 GEMM requires a 16-byte-aligned output"
        )
    sm_count = device_sm_count(device_index)
    tile_starts = None
    if aligned is False and (m, n, k) in _CG2_SHAPE_ROUTE and not scalar_output:
        tile_starts = _tile_starts(m_indices)
    route, grid = _launch_plan(
        m,
        n,
        k,
        sm_count=sm_count,
        scalar_output=scalar_output,
        block_scaled=block_scaled,
        alignment=alignment,
        num_groups=groups,
        arch=arch if swap_surfaces is not None else None,
        aligned=aligned,
        tile_starts=tile_starts,
    )
    if (
        layout is not None
        and layout_tile_m is not None
        and _expert_aligned_preferred(layout[0], layout[1], layout[3], layout_tile_m)
    ):
        route, grid = _expert_aligned_plan(n, layout[2], sm_count=sm_count)
    module_name = select_module(arch, route)
    program = PROGRAMS[MODULES[module_name]["program"]]
    module = load_cake_grouped_fp8_gemm_module(module_name)
    entry = getattr(module, program["ffi_entry"])

    # The scalar-store route never reads its output tensor map; the validated
    # A tensor provides an aligned BF16 view with a compatible descriptor.
    c_tma = a.view(torch.bfloat16) if route == DEEPK_C2_SCALAR_OUTPUT_ROUTE else out
    a_scale_layout = None
    if block_scaled:
        tile_n = route_geometry(route)[1]
        a_u8 = a.view(torch.uint8)
        bindings: dict[str, Any] = {
            "A": a_u8,
            "A64": a_u8,
            "A32": a_u8,
            "B": b.view(torch.uint8),
            "SFA": sfa,
            "SFB": sfb,
            "m_indices": m_indices,
            "C_tma": out,
            "shape_m": m,
            "shape_n": n,
            "grid_n": math.ceil(n / tile_n),
            "k_tiles": k // BS_BLOCK_K,
            "sfa_row_stride": sfa_rs,
            "sfa_col_stride": sfa_cs,
            "sfb_group_stride": sfb_gs,
            "sfb_row_stride": sfb_rs,
            "sfb_col_stride": sfb_cs,
        }
        if route in BS_SWAP_ROUTES:
            assert swap_surfaces is not None
            bindings["SFA_packed"], bindings["SFB_packed"], bindings["sf_cols"] = (
                swap_surfaces
            )
        a_scale_layout = (tuple(int(v) for v in a_scale.shape), (sfa_rs, sfa_cs))
    else:
        bindings = {
            "A": a.view(torch.uint8),
            "B": b.view(torch.uint8),
            "C_tma": c_tma,
            "C": out,
            "a_scale": a_scale,
            "b_scale": b_scale,
            "m_indices": kernel_indices,
            "M": m,
            "N": n,
            "K": k,
            "G": groups,
        }
    arg_plan = ARG_PLANS[program["arg_plan"]]
    return PreparedGroupGemmFp8NtGroupwiseContiguous(
        route=route,
        module_name=module_name,
        grid=grid,
        device=out.device,
        _out=out,
        m=m,
        n=n,
        k=k,
        groups=groups,
        fill_padding=fill_padding,
        _entry=entry,
        _arguments=bind_arguments(arg_plan, bindings, grid, module_name=module_name),
        _slots=argument_slots(arg_plan),
        _padding_source=m_indices if fill_padding else None,
        _padding_scratch=padding_scratch,
        _a_scale_layout=a_scale_layout,
    )


__all__ = [
    "BS_ROUTES",
    "PreparedGroupGemmFp8NtGroupwiseContiguous",
    "argument_slots",
    "bind_arguments",
    "bs_block_n",
    "bs_route_for_shape",
    "device_sm_count",
    "is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available",
    "launch_plan",
    "prepare_group_gemm_fp8_nt_groupwise_contiguous",
    "route_for_shape",
]
