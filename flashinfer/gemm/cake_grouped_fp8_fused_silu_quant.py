# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared fused grouped FP8 gate_up GEMM + SwiGLU + per-token-group FP8 quantization on Blackwell (SM100a, SM103a).

These are the generated Cake programs for the MoE gate_up chain that FlashInfer
otherwise runs as three kernels: :func:`flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous`
(BF16 ``[M, 2H]``), :func:`flashinfer.activation.silu_and_mul` and
:func:`flashinfer.quantization.per_token_group_quant_8bit` (group size 128,
FP8 E4M3).  Every route reproduces the chain's rounding points exactly:
``g = BF16(gate)``, ``u = BF16(up)``, ``h = BF16(silu(g) * u)``,
``scale = max(absmax_128(h), eps) / 448``, ``q = E4M3(clamp(h / scale, -448, 448))``.

* ``M >= SMALL_M_MAX`` (2048 rows): two persistent fused kernels keep the
  GEMM's ordered FP32 K-block accumulation in tensor memory and apply
  SwiGLU + quantization in their epilogue.  The BF16 intermediate, the
  activation launch and the quantization launch disappear.
* ``M < SMALL_M_MAX``: the fused kernel's one-SM-per-tile epilogue does not
  amortize on a few tiles, so the prepared launch runs the FlashInfer grouped
  GEMM into a private BF16 workspace followed by one generated elementwise
  kernel that fuses SwiGLU with the group quantization (two launches instead
  of three).  The GEMM kernel is chosen by :func:`small_m_gemm_backend`.
* Routing-aware rule (:func:`select_route`): when the routing is known —
  the caller passes the rows per expert as ``group_counts`` (no device work),
  or ``validate_indices=True`` derives the 128-row block count of every expert
  from ``m_indices`` with one device-to-host transfer — a large problem's three
  candidate routes (pair + tail, mixed schedule, GEMM + act) are ranked by a
  makespan model fitted on B200 route measurements (``ROUTE_MODEL_B200``: pair
  tiles grid-striding over the clusters plus list-scheduled odd-tail units,
  the mixed kernel's LPT assignment, an affine GEMM + act time; ties resolve
  in that order), and the model's pick replaces the route of the legacy
  odd-tail rule (:func:`_legacy_route`: pair + tail while the odd-tail blocks
  x N256 tiles fit on the SMs the 128-CTA pair grid leaves free — an odd tail
  runs as a single-CTA tile that ingests a whole B slab and costs ~1.5x a pair
  tile — else the mixed schedule when the pair tiles outnumber the clusters
  without dividing evenly, else GEMM + act) only when the model predicts the
  legacy route more than ``ROUTE_MODEL_MARGIN`` (5 %) slower.  The model is
  calibrated at ``2H = 2048, K = 4096`` (``ROUTE_MODEL_GEOMETRY``; its pair-tile
  time does not scale with K), so the prepared launch passes ``k`` and the
  model applies only at that geometry; every other geometry, and a call
  without ``k``, keeps the legacy rule until measured.  Without the routing
  the plan is shape-only.
* Mixed-schedule route (``FUSED_MIXED_ROUTE``): one persistent ``cta_group::2``
  kernel whose clusters (``min(pair_tiles + odd_units, min(sm_count, 128) // 2)``)
  process the complete block pairs as 256-row tiles and every odd tail block
  as a two-CTA M=128 solo unit (no tail kernel, no BF16 intermediate).  The
  ``pair_tiles % clusters`` heavy clusters hold one pair more, so the light
  clusters take the first two solo units each and the remaining solo units
  round-robin over every cluster (longest-processing-time first).  Measured on
  B200 at ``2H=2048, K=4096``: 1.066x the GEMM + wide act route on the random
  128-aligned and mixed-tail routings (M = 4096) and 1.03-1.15x on the
  all-odd, all-one-block and odd-heavy M = 2048/3072 routings the model sends
  here.
* A diverted problem the model keeps on the GEMM + act route (many odd-tail
  units at large M) takes, with at least ``ACT_WIDE_MIN_ITEMS`` (row,
  128-column group) items, the wide act kernel (eight lanes per group, four
  groups per warp, 64 B per lane in flight: 1.09-1.43x the one-warp-per-group
  kernel from M = 2048, H = 1024 on, bitwise-identical outputs); smaller
  problems keep the one-warp-per-group kernel, which is faster on
  latency-bound problems.

Routing contract (the same as the CuTe-DSL contiguous grouped GEMM): rows are
sorted by expert through ``m_indices``; every *internal* expert boundary is a
multiple of 128 rows (``moe_align_block_size`` with block 128), only the final
expert may end in a partial block, empty experts are allowed, and
``M <= 8192``.  The pair kernel processes consecutive same-expert 128-row
blocks as 256-row cluster tiles (two CTAs, ``cta_group::2``); the tail kernel
processes the odd tail block of every odd-count expert as single-CTA 128-row
tiles (``cta_group::1``), so no dummy block is computed on non-uniform
routings.  The tail kernel is launched with the programmatic-dependent-launch
attribute right after the pair kernel (which signals its dependents at start),
so its CTAs fill the SMs the pair grid leaves free or retires from; it reads
only the operator's inputs.

Tensor maps are encoded by the bindings and passed to the kernels by value, so
a prepared launch owns no descriptor storage and allocates nothing after
preparation: every ``launch()``, including the first, only submits work on the
current stream and may be captured into a CUDA graph (the small-M route's
CuTe-DSL GEMM is specialized by one eager call at preparation).  The GEMM +
act routes'
BF16 ``(M, 2H)`` intermediate may be supplied by the caller (``workspace``) and
is otherwise allocated once at preparation.  Preparation binds the problem
shape, the route and the expert weights; the per-token operands (``a``,
``a_scale``, ``m_indices``, ``out_q``, ``out_s``) may be rebound at every
``launch`` so callers need no staging copies.
"""

from __future__ import annotations

import functools
import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import tvm_ffi

from ..jit.gemm.cake_grouped_fp8_fused_silu_quant import (
    ARG_PLANS,
    MODULES,
    PROGRAMS,
    ROUTE_GEOMETRY,
    SUPPORTED_COMPUTE_CAPABILITIES,
    device_arch,
    generated_program_available,
    load_cake_grouped_fp8_fused_silu_quant_module,
    select_stage_module,
)
from .cake_grouped_fp8_gemm import prepare_group_gemm_fp8_nt_groupwise_contiguous

GROUP_SIZE = 128
K_BLOCK = 128
K_MULTIPLE = 512
N2_MULTIPLE = 256
ROW_BLOCK = 128
MAX_M = 8192
CTA_CAP = 128
QUANT_EPS = 1e-10
FP8_E4M3_MAX = 448.0

# Route names double as the generated-program template names of their first kernel.
FUSED_ROUTE = "fused_cg2_ab7_pairsched_solotail_kg4"
FUSED_MIXED_ROUTE = "fused_cg2_ab7_mixedsched_lpt_kg4"
FUSED_ROUTES = frozenset({FUSED_ROUTE, FUSED_MIXED_ROUTE})
ACT_ROUTE = "gemm_then_silu_mul_group_quant_fp8"
ACT_WIDE_ROUTE = "gemm_then_silu_mul_group_quant_fp8_wide"
ACT_ROUTES = frozenset({ACT_ROUTE, ACT_WIDE_ROUTE})
# Generated kernel stages per route in launch order; the pair + tail route's tail kernel has its own template geometry.
FUSED_PAIR_STAGE = "pair"
FUSED_TAIL_STAGE = "tail"
FUSED_MAIN_STAGE = "main"
ACT_STAGE = "main"
ROUTE_STAGES = {
    FUSED_ROUTE: (FUSED_PAIR_STAGE, FUSED_TAIL_STAGE),
    FUSED_MIXED_ROUTE: (FUSED_MAIN_STAGE,),
    ACT_ROUTE: (ACT_STAGE,),
    ACT_WIDE_ROUTE: (ACT_STAGE,),
}
TAIL_GEOMETRY_KEY = f"{FUSED_ROUTE}_{FUSED_TAIL_STAGE}"
# Rows below this take ACT_ROUTE (measured B200 crossover of the fused kernel against grouped GEMM + the fused
# activation kernel; every problem with at most 9 row blocks lost to the chain in the fused kernel).
SMALL_M_MAX = 2048
# Elementwise kernel launch: one warp per (row, 128-column group), four warps per CTA, at most sixteen resident
# CTAs per SM (32 registers x 128 threads, full occupancy); grid-stride beyond that.
ACT_WARPS_PER_CTA = 4
ACT_CTAS_PER_SM = 16
# Wide act kernel launch: four (row, 128-column group) items per warp (eight lanes each), four warps per CTA, at
# most ten resident CTAs per SM (48 registers x 128 threads); grid-stride beyond that.
ACT_WIDE_GROUPS_PER_WARP = 4
ACT_WIDE_CTAS_PER_SM = 10
# A diverted large problem takes the wide act kernel from this many items on (measured B200 crossover envelope:
# 1.085x at 16384 items = M 2048 x H 1024, 1.32-1.43x from 32768; 0.92-0.97x below ~2400 items).
ACT_WIDE_MIN_ITEMS = 16384
# GEMM kernels of the small-M route.
GEMM_BACKEND_CUTE = "cute_dsl"  # group_gemm_fp8_nt_groupwise_contiguous (CuTe-DSL)
GEMM_BACKEND_CAKE = (
    "cake_prepared"  # prepare_group_gemm_fp8_nt_groupwise_contiguous (Cake)
)


def route_geometry(route: str) -> tuple[int, int, int]:
    """``(tile_m, tile_n, cluster_ctas)`` of one route from the export registry.

    ``tile_n`` counts B rows (gate + up), i.e. twice the SwiGLU columns.
    """
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


@functools.cache
def device_sm_count(device_index: int) -> int:
    """Streaming-multiprocessor count of CUDA device ``device_index`` (queried once)."""
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


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


def _argument_slots(arg_plan: Sequence[Sequence[str]]) -> dict[str, tuple[int, ...]]:
    """Positions of every named binding in a generated program's argument plan."""
    slots: dict[str, list[int]] = {}
    for index, (kind, name) in enumerate(arg_plan):
        if kind != "grid":
            slots.setdefault(name, []).append(index)
    return {name: tuple(positions) for name, positions in slots.items()}


def _rebind_arguments(
    arguments: tuple[Any, ...],
    slots: dict[str, tuple[int, ...]],
    bindings: dict[str, Any],
) -> tuple[Any, ...]:
    """``arguments`` with every slot of each named binding replaced; names a stage does not bind are ignored."""
    rebound = list(arguments)
    for name, value in bindings.items():
        for position in slots.get(name, ()):
            rebound[position] = value
    return tuple(rebound)


def _forward_fill_padding(
    m_indices: torch.Tensor, filled: torch.Tensor, positions: torch.Tensor
) -> None:
    """``filled[r] = max(0, max(m_indices[:r + 1]))``: padding rows follow the preceding expert.

    Two small allocation-free kernels (``cummax`` into prepared scratch, then
    ``clamp``); both are graph-capturable.  Leading padding maps onto expert 0.
    """
    torch.cummax(m_indices, 0, out=(filled, positions))
    filled.clamp_(min=0)


# A partial final block hands the small-M GEMM to the Cake kernel from this K on (see small_m_gemm_backend).
PARTIAL_TAIL_CAKE_MIN_K = 1024


def small_m_gemm_backend(m: int, n2: int, k: int) -> str:
    """GEMM kernel of the small-M route (measured B200 policy).

    The CuTe-DSL grouped GEMM specializes on ``M % 128`` (tail predication);
    on 128-aligned M it is the faster kernel for every small-M problem
    measured.  Its predicated-tail build pays a penalty that grows with K
    while the Cake prepared GEMM is insensitive to the tail, so a partial
    final block hands the GEMM to the Cake kernel from ``K = 1024`` on.
    """
    if m <= 0 or n2 <= 0 or k <= 0:
        raise ValueError("small_m_gemm_backend requires positive M, 2H and K")
    if m % ROW_BLOCK and k >= PARTIAL_TAIL_CAKE_MIN_K:
        return GEMM_BACKEND_CAKE
    return GEMM_BACKEND_CUTE


def routing_blocks(group_counts: Sequence[int]) -> tuple[int, ...]:
    """128-row block count of every expert (``ceil(rows / 128)``)."""
    counts = [int(c) for c in group_counts]
    if any(c < 0 for c in counts):
        raise ValueError("group_counts must be non-negative")
    return tuple(-(-c // ROW_BLOCK) for c in counts)


def fused_tile_counts(n2: int, group_blocks: Sequence[int]) -> tuple[int, int]:
    """``(pair_tiles, odd_tail_units)`` the fused route schedules for a routing.

    Complete 128-row block pairs x N256 tiles run on the pair kernel, each
    expert's odd tail block x N256 tiles on the tail kernel.
    """
    _, tile_n, _ = route_geometry(FUSED_ROUTE)
    n_tiles = n2 // tile_n
    blocks = [int(b) for b in group_blocks]
    return sum(b // 2 for b in blocks) * n_tiles, sum(b % 2 for b in blocks) * n_tiles


# Routing-aware route rule (measured on B200 at the wide_ep32 shape).  An odd
# tail block runs on the tail kernel as a single-CTA unit that ingests the whole
# B slab (1.5 MiB per 128 rows against a pair CTA's 1.0 MiB) and takes ~1.5x a
# pair tile; the units that start on the SMs the 128-CTA pair grid leaves free
# finish inside the pair phase, every further unit waits for a retiring cluster
# and adds its full latency.  With at most that many odd-tail units the legacy
# rule (:func:`_legacy_route`) keeps the fused route; beyond it a diverted
# problem takes the mixed-schedule route on its measured envelope (more pair
# tiles than clusters, not dividing evenly) and the CuTe GEMM + act route
# otherwise.  At the model's calibration geometry :func:`select_route` ranks the
# three candidates with the fitted makespan model below and keeps the legacy
# rule's route within ``ROUTE_MODEL_MARGIN``.  ``False`` keeps the shape-only rule.
ROUTING_AWARE_RULE = True


def act_items(m: int, n2: int) -> int:
    """(row, 128-column group) items of the act kernels for ``[M, 2H]``."""
    return m * (n2 // 2 // GROUP_SIZE)


def _diverted_act_route(m: int, n2: int) -> str:
    return ACT_WIDE_ROUTE if act_items(m, n2) >= ACT_WIDE_MIN_ITEMS else ACT_ROUTE


def mixed_clusters(pair_tiles: int, odd_units: int, *, sm_count: int) -> int:
    """Clusters of the mixed-schedule route for a routing: one per unit up to the 128-CTA cap."""
    _, _, cluster_ctas = route_geometry(FUSED_ROUTE)
    return min(pair_tiles + odd_units, min(sm_count, CTA_CAP) // cluster_ctas)


# Makespan model of the three large-M routes.  Verbatim copy of the Cake
# dispatcher's route-model module (the export tool compares both hosts' route on
# every export row; keep the constants and functions identical).
# Fitted on the round-12 route probes of this op: 20 routings x three arms (pair + tail,
# mixed LPT, GEMM + wide act) at [2H = 2048, K = 4096] on B200 (148 SMs, cold-L2
# CUPTI active-union medians), coordinate descent on the squared log error,
# rms-log 5 %.  Microseconds; T = one pair tile at full concurrency.
ROUTE_MODEL_B200 = {
    "T": 16.6953125,  # one pair tile on one cluster, us
    "s_tail": 1.552734375,  # odd-tail unit of the tail kernel, in pair tiles
    "o_pair": 6.5859375,  # pair + tail route overhead, us
    "o_empty": 11.28125,  # pair + tail route overhead without a pair tile, us
    "s_v5": 0.6519531249999999,  # solo unit of the mixed kernel, in pair tiles
    "o_v5": 8.60546875,  # mixed route overhead, us
    "a0": 15.6953125,  # GEMM + wide act: intercept, us
    "a1": 7.9265625,  # GEMM + wide act: us per 1024 rows
    "a2": 2.357421875,  # GEMM + wide act: us per 1024 padded rows
}
# No-regression guard: the model's pick replaces the legacy rule's route only when
# the legacy route is predicted more than this fraction slower (the fit's rms-log
# error).  On the 20 fitted routings the guarded pick never measures slower than
# the legacy pick and matches the measured best on 17 (legacy: 10).
ROUTE_MODEL_MARGIN = 0.05
# Calibration geometry (2H, K) of ROUTE_MODEL_B200: T is one K=4096 pair tile and
# does not scale with K.  select_route applies the guarded decision only when the
# caller passes this geometry and keeps the legacy rule elsewhere until measured.
ROUTE_MODEL_GEOMETRY = (2048, 4096)


def act_pad_rows(group_blocks: Sequence[int]) -> int:
    """Padding feature of the act-route model: rows added when every expert's
    128-row blocks are rounded up to a 256-row multiple (128 per odd expert)."""
    return sum(-(-int(b) // 2) * 256 - int(b) * 128 for b in group_blocks)


def fused_makespan_us(
    pair_tiles: int,
    odd_units: int,
    *,
    sm_count: int,
    cta_cap: int,
    params: Mapping[str, float] = ROUTE_MODEL_B200,
) -> float:
    """Predicted time of the pair + tail route (see the module docstring)."""
    t = params["T"]
    clusters = min(sm_count, cta_cap) // 2
    slots = [0.0] * (sm_count - 2 * clusters)  # SMs free from t = 0
    pair_end = 0.0
    for c in range(clusters):
        tiles = (pair_tiles - c + clusters - 1) // clusters if pair_tiles > c else 0
        end = tiles * t
        pair_end = max(pair_end, end)
        slots += [end, end]
    heapq.heapify(slots)
    tail_end = 0.0
    for _ in range(odd_units):
        start = heapq.heappop(slots)
        end = start + params["s_tail"] * t
        tail_end = max(tail_end, end)
        heapq.heappush(slots, end)
    overhead = params["o_empty"] if pair_tiles == 0 else params["o_pair"]
    return max(pair_end, tail_end) + overhead


def mixed_makespan_us(
    pair_tiles: int,
    odd_units: int,
    *,
    sm_count: int,
    cta_cap: int,
    params: Mapping[str, float] = ROUTE_MODEL_B200,
) -> float:
    """Predicted time of the mixed-schedule route: the worst cluster of the v5
    kernel's LPT assignment (see the module docstring)."""
    t = params["T"]
    clusters = min(pair_tiles + odd_units, min(sm_count, cta_cap) // 2)
    if clusters < 1:
        raise ValueError("the mixed-schedule route needs at least one cluster")
    heavy = pair_tiles % clusters
    light = clusters - heavy
    phase1 = min(2 * light, odd_units)
    rest = odd_units - phase1
    worst = 0.0
    for c in range(clusters):
        pairs = (pair_tiles - c + clusters - 1) // clusters if pair_tiles > c else 0
        off1 = c - heavy
        if off1 >= 0 and phase1 > off1:
            solos1 = (phase1 - off1 + light - 1) // light
        else:
            solos1 = 0
        solos2 = (rest - c + clusters - 1) // clusters if rest > c else 0
        worst = max(worst, pairs * t + (solos1 + solos2) * params["s_v5"] * t)
    return worst + params["o_v5"]


def act_makespan_us(
    m: int, pad_rows: int, *, params: Mapping[str, float] = ROUTE_MODEL_B200
) -> float:
    """Predicted time of the GEMM + act route (affine in M and the padded rows)."""
    return params["a0"] + params["a1"] * m / 1024 + params["a2"] * pad_rows / 1024


def guarded_route(
    legacy: str, predicted: Mapping[str, float], *, margin: float = ROUTE_MODEL_MARGIN
) -> str:
    """The route with the smallest prediction (ties: the first in ``predicted``'s
    iteration order), unless ``legacy`` is predicted within ``margin`` of it."""
    if legacy not in predicted:
        raise ValueError(f"legacy route {legacy!r} has no prediction")
    best = min(predicted, key=predicted.__getitem__)
    if predicted[legacy] > (1.0 + margin) * predicted[best]:
        return best
    return legacy


def _legacy_route(
    m: int, n2: int, pair_tiles: int, odd_units: int, *, sm_count: int
) -> str:
    """Legacy routing-aware rule, the reference the makespan model is guarded
    against: the pair + tail route while the odd-tail units fit on the SMs the
    pair grid leaves free; else the mixed-schedule route when the pair tiles
    outnumber the grid's clusters without dividing evenly
    (:func:`mixed_clusters`); else the GEMM + wide act route (the small-M act
    route below ``ACT_WIDE_MIN_ITEMS`` items).
    """
    _, _, cluster_ctas = route_geometry(FUSED_ROUTE)
    pair_ctas = (min(sm_count, CTA_CAP) // cluster_ctas) * cluster_ctas
    if odd_units <= sm_count - pair_ctas:
        return FUSED_ROUTE
    clusters = mixed_clusters(pair_tiles, odd_units, sm_count=sm_count)
    if pair_tiles > clusters and pair_tiles % clusters:
        return FUSED_MIXED_ROUTE
    return _diverted_act_route(m, n2)


def select_route(
    m: int,
    n2: int,
    *,
    sm_count: int,
    group_blocks: Optional[Sequence[int]] = None,
    k: Optional[int] = None,
) -> str:
    """Route of a problem: ``M < SMALL_M_MAX`` takes the GEMM + act route; larger
    problems take the pair + tail route unless the routing is known
    (``group_blocks``, 128-row blocks per expert).  With the routing and ``k`` at
    the model's calibration geometry (``(n2, k) == ROUTE_MODEL_GEOMETRY``:
    2H = 2048, K = 4096) the three candidates -- pair + tail, mixed schedule
    (when :func:`mixed_clusters` gives at least one cluster) and the act route
    of :func:`_diverted_act_route` -- are ranked by the fitted makespan model
    (``ROUTE_MODEL_B200``; ties resolve in that order), and the model's pick
    replaces the route of :func:`_legacy_route` only when the model predicts
    the legacy route more than ``ROUTE_MODEL_MARGIN`` slower.  Any other
    geometry, or ``k=None``, returns the legacy route unchanged (unmeasured;
    the model's pair-tile time does not scale with K).
    """
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if m <= 0 or n2 <= 0:
        raise ValueError("select_route requires positive M and 2H")
    if k is not None and (k <= 0 or k % K_MULTIPLE):
        raise ValueError(f"K must be a positive multiple of {K_MULTIPLE}, got {k}")
    if m < SMALL_M_MAX:
        return ACT_ROUTE
    if group_blocks is None or not ROUTING_AWARE_RULE:
        return FUSED_ROUTE
    if sum(int(b) for b in group_blocks) != -(-m // ROW_BLOCK):
        raise ValueError(
            f"group_blocks {list(group_blocks)} do not cover M={m} ({-(-m // ROW_BLOCK)} blocks)"
        )
    pair_tiles, odd_units = fused_tile_counts(n2, group_blocks)
    legacy = _legacy_route(m, n2, pair_tiles, odd_units, sm_count=sm_count)
    if k is None or (n2, k) != ROUTE_MODEL_GEOMETRY:
        return legacy
    predicted = {
        FUSED_ROUTE: fused_makespan_us(
            pair_tiles, odd_units, sm_count=sm_count, cta_cap=CTA_CAP
        )
    }
    if mixed_clusters(pair_tiles, odd_units, sm_count=sm_count) >= 1:
        predicted[FUSED_MIXED_ROUTE] = mixed_makespan_us(
            pair_tiles, odd_units, sm_count=sm_count, cta_cap=CTA_CAP
        )
    predicted[_diverted_act_route(m, n2)] = act_makespan_us(
        m, act_pad_rows(group_blocks)
    )
    return guarded_route(legacy, predicted, margin=ROUTE_MODEL_MARGIN)


def launch_plan(
    m: int,
    n2: int,
    *,
    sm_count: int,
    group_blocks: Optional[Sequence[int]] = None,
    k: Optional[int] = None,
) -> tuple[str, tuple[int, int, int]]:
    """Resolve ``(route, grid)`` for ``A[M, K]`` and ``B[G, 2H, K]`` on ``sm_count`` SMs.

    The route follows :func:`select_route` (``group_blocks`` = 128-row blocks
    per expert enables the routing-aware rule, ``k`` its makespan model at the
    calibration geometry; ``prepare_...`` passes both when
    ``validate_indices=True``).  The GEMM + act route launches one warp
    per (row, 128-column group) item, ``ACT_WARPS_PER_CTA`` warps per CTA, at
    most ``ACT_CTAS_PER_SM`` CTAs per SM; the wide act route one warp per
    ``ACT_WIDE_GROUPS_PER_WARP`` items, at most ``ACT_WIDE_CTAS_PER_SM`` CTAs
    per SM.  The fused route launches complete
    CTA pairs; its tile upper bound counts every 128-row block as a lone block,
    and the 128-CTA cap is the measured B200 policy of the parent gate_up route
    (148 CTAs measured slower on every routing).
    """
    route = select_route(m, n2, sm_count=sm_count, group_blocks=group_blocks, k=k)
    if route in ACT_ROUTES:
        route_geometry(route)  # the registry must carry the route
        items = act_items(m, n2)
        if route == ACT_WIDE_ROUTE:
            warps = -(-items // ACT_WIDE_GROUPS_PER_WARP)
            ctas_cap = ACT_WIDE_CTAS_PER_SM * sm_count
        else:
            warps = items
            ctas_cap = ACT_CTAS_PER_SM * sm_count
        ctas = max(1, min(-(-warps // ACT_WARPS_PER_CTA), ctas_cap))
        return route, (ctas, 1, 1)
    _, tile_n, cluster_ctas = route_geometry(route)
    if route == FUSED_MIXED_ROUTE:
        # selected only with the routing: one cluster per pair tile or solo unit up to the cap
        clusters = mixed_clusters(
            *fused_tile_counts(n2, group_blocks), sm_count=sm_count
        )
    else:
        max_tiles = math.ceil(m / ROW_BLOCK) * (n2 // tile_n)
        clusters = min(max_tiles, min(sm_count, CTA_CAP) // cluster_ctas)
    if clusters <= 0:
        raise ValueError("device cannot schedule one complete CTA pair")
    return route, (clusters * cluster_ctas, 1, 1)


def tail_launch_grid(m: int, n2: int, *, sm_count: int) -> tuple[int, int, int]:
    """Grid of the fused route's tail kernel for ``A[M, K]`` and ``B[G, 2H, K]`` on ``sm_count`` SMs.

    One single-CTA 128-row tile per odd-tail block and N256 tile; the grid is
    one CTA per SM up to the unit upper bound (every block an odd tail) and the
    CTAs grid-stride over the actual units, so the grid does not depend on the
    routing.  A routing without odd tails runs an empty grid, overlapped
    with the pair kernel by programmatic dependent launch.
    """
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    tile_m, tile_n, _ = route_geometry(TAIL_GEOMETRY_KEY)
    return (max(1, min(math.ceil(m / tile_m) * (n2 // tile_n), sm_count)), 1, 1)


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
    m_indices: torch.Tensor, groups: int, *, allow_padding: bool = False
) -> tuple[int, ...]:
    """Check the routing contract (in range, sorted, 128-aligned internal boundaries) with one transfer.

    Every statistic is reduced on the device and copied to the host in a single
    transfer.  Returns the 128-row block count of every expert (the routing
    statistics of :func:`select_route`); every block is homogeneous under the
    contract, so the expert of a block is the index of its first row.  With
    ``allow_padding`` the value ``-1`` marks padding rows: the contract applies
    to the forward-filled indices the kernels read, and the remaining entries
    must be nondecreasing.
    """
    indices = m_indices.to(torch.int64)
    m = indices.numel()
    device = indices.device
    zero = torch.zeros((), dtype=torch.int64, device=device)
    lowest = indices.min()
    if allow_padding:
        filled = torch.cummax(indices, 0).values.clamp_(min=0)
        # a routed row below the running maximum breaks the order
        padding_unsorted = ((indices >= 0) & (indices != filled)).sum()
        indices = filled
    else:
        padding_unsorted = zero
    if m > 1:
        step = indices[1:] - indices[:-1]
        unsorted = (step < 0).sum() + padding_unsorted
        misaligned = (
            (step > 0) & (torch.arange(1, m, device=device) % ROW_BLOCK != 0)
        ).sum()
    else:
        unsorted, misaligned = padding_unsorted, zero
    block_experts = indices[::ROW_BLOCK]
    counts = torch.zeros((groups,), dtype=torch.int64, device=device).scatter_add_(
        0, block_experts.clamp(0, groups - 1), torch.ones_like(block_experts)
    )
    stats = torch.cat(
        (torch.stack((lowest, indices.max(), unsorted, misaligned)), counts)
    ).tolist()
    lowest, highest, unsorted, misaligned = stats[:4]
    if lowest < (-1 if allow_padding else 0) or highest >= groups:
        if allow_padding:
            raise ValueError("m_indices must satisfy -1 <= index < num_groups")
        raise ValueError(
            "m_indices must satisfy 0 <= index < num_groups; -1 padding needs fill_padding=True"
        )
    if unsorted:
        raise ValueError("m_indices must be sorted in nondecreasing order")
    if misaligned:
        raise ValueError(
            "m_indices must place every internal expert boundary at a multiple "
            "of 128 rows (only the final expert may end in a partial block)"
        )
    return tuple(int(v) for v in stats[4:])


def _group_blocks_from_counts(
    group_counts: Sequence[int], m: int, groups: int
) -> tuple[int, ...]:
    """Block counts from caller-supplied rows per expert (host integers; no device work)."""
    counts = [int(c) for c in group_counts]
    if len(counts) != groups:
        raise ValueError(
            f"group_counts must have one entry per expert ({groups}), got {len(counts)}"
        )
    if any(c < 0 for c in counts) or sum(counts) != m:
        raise ValueError(
            f"group_counts must be non-negative and sum to M={m}, got sum {sum(counts)}"
        )
    prefix = 0
    for count in counts:
        prefix += count
        if (
            0 < prefix < m and prefix % ROW_BLOCK
        ):  # only the final expert may end in a partial block
            raise ValueError(
                "group_counts must place every internal expert boundary at a multiple of 128 rows"
            )
    return routing_blocks(counts)


@dataclass
class PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant:
    """One prepared gate_up GEMM + SwiGLU + FP8 quantization launch.

    Shapes, dtypes, the route and the expert weights (``b``, ``b_scale``) are
    bound at preparation; tensor *contents* may change between launches, and
    the per-token operands ``a``, ``a_scale``, ``m_indices``, ``out_q`` and
    ``out_s`` may be replaced by same-shape tensors at every ``launch``
    (``launch(a=..., ...)``), so a caller can run one prepared object per layer
    on whatever buffers the dispatcher produced.  The route is planned from the
    prepared routing, so replacement indices must keep the routing contract
    (and, when the routing-aware rule chose the route, the routing statistics
    it was planned for).  ``launch()`` submits exactly ``num_kernels`` kernels
    (two for the pair + tail route: the pair kernel, then the tail kernel with
    the programmatic-dependent-launch attribute; one for the mixed-schedule
    route; two for the GEMM + act routes: the grouped GEMM into the private
    BF16 workspace, then the generated activation kernel; plus two tiny index
    kernels when ``fill_padding`` is on) on PyTorch's current stream for the
    bound device and returns ``(out_q, out_s)``.
    Every launch, including the first, may be captured into a CUDA graph; the
    prepared object retains no descriptor storage, and the small-M route's
    CuTe-DSL GEMM is run once at preparation so that it is specialized before
    any capture.  ``grid`` is the grid of the
    route's first generated kernel; ``stage_grids`` holds every generated
    kernel's grid by stage name.
    """

    route: str
    module_name: str
    grid: tuple[int, int, int]
    out_q: torch.Tensor
    out_s: torch.Tensor
    gemm_backend: Optional[str]
    stage_module_names: dict[str, str]
    stage_grids: dict[str, tuple[int, int, int]]
    m: int
    n2: int
    k: int
    fill_padding: bool
    _entries: tuple[
        tuple[Callable[..., Any], tuple[Any, ...], dict[str, tuple[int, ...]]], ...
    ]
    _gemm: Optional[Callable[..., Any]]
    _gemm_out: Optional[torch.Tensor]
    _padding_source: Optional[torch.Tensor]
    _padding_scratch: Optional[tuple[torch.Tensor, torch.Tensor]]

    def launch(
        self,
        a: Optional[torch.Tensor] = None,
        a_scale: Optional[torch.Tensor] = None,
        m_indices: Optional[torch.Tensor] = None,
        out_q: Optional[torch.Tensor] = None,
        out_s: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the route; any operand given here replaces the prepared one for this call.

        Replacement tensors must match the prepared shape, dtype, device and
        alignment.  Rebinding allocates nothing on the device.
        """
        device = self.out_q.device
        rebound: dict[str, Any] = {}
        gemm_indices = None
        if (
            a is not None
            or a_scale is not None
            or m_indices is not None
            or out_q is not None
            or out_s is not None
        ):
            h = self.n2 // 2
            if a is not None:
                a = _require_tensor(
                    a,
                    "a",
                    shape=(self.m, self.k),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                rebound["A"] = rebound["A64"] = a.view(torch.uint8)
            if a_scale is not None:
                a_scale = rebound["a_scale"] = _require_tensor(
                    a_scale,
                    "a_scale",
                    shape=(self.m, self.k // K_BLOCK),
                    dtype=torch.float32,
                    device=device,
                )
            if m_indices is not None:
                m_indices = _require_tensor(
                    m_indices,
                    "m_indices",
                    shape=(self.m,),
                    dtype=torch.int32,
                    device=device,
                )
                if not self.fill_padding:
                    gemm_indices = rebound["m_indices"] = m_indices
            if out_q is not None:
                rebound["out_q"] = _require_tensor(
                    out_q,
                    "out_q",
                    shape=(self.m, h),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
            if out_s is not None:
                rebound["out_s"] = _require_tensor(
                    out_s,
                    "out_s",
                    shape=(self.m, h // GROUP_SIZE),
                    dtype=torch.float32,
                    device=device,
                    alignment=4,
                )
        with torch.cuda.device(device):
            if self.fill_padding:
                assert self._padding_scratch is not None
                source = m_indices if m_indices is not None else self._padding_source
                assert source is not None
                _forward_fill_padding(source, *self._padding_scratch)
            if self._gemm is not None:
                self._gemm(a=a, a_scale=a_scale, m_indices=gemm_indices)
            with tvm_ffi.use_torch_stream():
                for entry, arguments, slots in self._entries:
                    if rebound:
                        arguments = _rebind_arguments(arguments, slots, rebound)
                    entry(*arguments)
        return rebound.get("out_q", self.out_q), rebound.get("out_s", self.out_s)

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        """CTAs of the route's first generated kernel (the pair, mixed-schedule or activation kernel)."""
        return int(self.grid[0])

    @property
    def tail_grid(self) -> Optional[tuple[int, int, int]]:
        """Grid of the pair + tail route's tail kernel (``None`` on the other routes)."""
        return self.stage_grids.get(FUSED_TAIL_STAGE)

    @property
    def num_kernels(self) -> int:
        return len(self._entries) + (0 if self._gemm is None else 1)


def is_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_prepared_available(
    device: torch.device,
) -> bool:
    """True when a generated fused program is registered for ``device``."""
    return device.type == "cuda" and generated_program_available(device)


def prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out_q: Optional[torch.Tensor] = None,
    out_s: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    *,
    validate_indices: bool = False,
    group_counts: Optional[Sequence[int]] = None,
    fill_padding: bool = False,
) -> PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant:
    r"""Prepare a contiguous grouped FP8 gate_up GEMM + SwiGLU + FP8 quantization launch on SM100a / SM103a.

    Parameters
    ----------
    a : torch.Tensor
        Contiguous FP8 E4M3 input of shape ``(M, K)``; ``0 < M <= 8192`` and
        ``K`` a positive multiple of 512.
    b : torch.Tensor
        Contiguous FP8 E4M3 expert weights of shape ``(G, 2H, K)``: gate rows
        ``[0, H)`` followed by up rows ``[H, 2H)``; ``2H`` a positive multiple
        of 256.
    a_scale : torch.Tensor
        Contiguous float32 scales of shape ``(M, K // 128)``.
    b_scale : torch.Tensor
        Contiguous float32 scales of shape ``(G, 2H // 128, K // 128)``.
    m_indices : torch.Tensor
        Contiguous int32 expert indices of shape ``(M,)``, sorted in
        nondecreasing order with ``0 <= index < G``; every internal expert
        boundary is a multiple of 128 rows (the final expert may be partial);
        empty experts are allowed.  With ``fill_padding`` the value ``-1``
        marks padding rows (the compact MoE layout): such a row is computed
        with the preceding expert's weights (expert 0 for leading padding),
        its output row is unspecified, and the contract applies to the
        forward-filled indices.
    out_q : Optional[torch.Tensor]
        Contiguous FP8 E4M3 output ``(M, H)`` = ``silu(gate) * up`` quantized
        per token in groups of 128 columns.  Allocated if omitted.
    out_s : Optional[torch.Tensor]
        Contiguous float32 scales ``(M, H // 128)`` (row-major, one per
        128-column group, ``max(absmax, 1e-10) / 448``).  Allocated if omitted.
    workspace : Optional[torch.Tensor]
        Contiguous bfloat16 ``(M, 2H)`` intermediate of the GEMM + act routes.
        Allocated at preparation if omitted and the route needs it; ignored by
        the fused routes.
    validate_indices : bool
        Check the routing contract (range, sortedness, 128-aligned internal
        boundaries) with one device-to-host transfer; the per-expert block
        counts it yields drive the routing-aware route rule.
    group_counts : Optional[Sequence[int]]
        Rows per expert as host integers (``len == G``, summing to ``M``).
        Enables the routing-aware rule without any device work; checked for
        consistency against ``m_indices`` when ``validate_indices=True``.
        With ``fill_padding`` an expert's count includes the padding rows that
        follow it.
    fill_padding : bool
        Accept ``-1`` padding rows in ``m_indices``.  The prepared object owns
        an ``(M,)`` int32 plus ``(M,)`` int64 scratch and every launch runs two
        small forward-fill kernels before the route (graph-capturable).

    Returns
    -------
    PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant
        Call ``.launch()`` to run the route's kernel(s) on the current stream;
        ``.launch(a=..., a_scale=..., m_indices=..., out_q=..., out_s=...)``
        swaps the per-token operands for that call.

    Notes
    -----
    Requires an SM100a (compute capability 10.0) or SM103a (10.3) device and a
    registered generated program for the resolved route (:func:`launch_plan`);
    the mixed-schedule route is reachable only with the routing known
    (``group_counts`` or ``validate_indices=True``).  All
    tensors must live on the same CUDA device and be 16-byte aligned
    (``out_s`` 4-byte).  The outputs match the three-kernel FlashInfer chain
    (CuTe-DSL grouped GEMM, ``silu_and_mul``, ``per_token_group_quant_8bit``)
    element for element on the same inputs.  ``M < SMALL_M_MAX`` rows run the
    FlashInfer grouped GEMM (:func:`small_m_gemm_backend`) into the BF16
    ``(M, 2H)`` ``workspace``, then one generated SwiGLU + group-quantization
    kernel.  Preparation performs no device work beyond the optional index
    validation; the device SM count is read once per process.  Index values
    are unchecked unless ``validate_indices=True``; violating the routing
    contract is undefined behavior.
    """
    if not isinstance(a, torch.Tensor) or a.device.type != "cuda":
        raise ValueError("a must be a CUDA torch.Tensor")
    device = a.device
    device_index = (
        torch.cuda.current_device() if device.index is None else int(device.index)
    )
    arch = device_arch(device_index)
    if arch is None:
        raise NotImplementedError(
            "prepared fused grouped FP8 gate_up+SwiGLU+quant requires an SM100a or SM103a device "
            f"(compute capability {sorted(SUPPORTED_COMPUTE_CAPABILITIES)}); "
            f"got {torch.cuda.get_device_capability(device_index)}"
        )
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("a must have shape (M, K) and b must have shape (G, 2H, K)")
    m, k = (int(v) for v in a.shape)
    groups, n2 = (int(v) for v in b.shape[:2])
    if m <= 0 or groups <= 0:
        raise ValueError("fused grouped FP8 gate_up requires positive M and G")
    if m > MAX_M:
        raise ValueError(f"M must be at most {MAX_M} rows (64 blocks of 128), got {m}")
    if n2 <= 0 or n2 % N2_MULTIPLE:
        raise ValueError(f"2H must be a positive multiple of {N2_MULTIPLE}, got {n2}")
    if k <= 0 or k % K_MULTIPLE:
        raise ValueError(f"K must be a positive multiple of {K_MULTIPLE}, got {k}")
    h = n2 // 2
    a = _require_tensor(a, "a", shape=(m, k), dtype=torch.float8_e4m3fn, device=device)
    b = _require_tensor(
        b, "b", shape=(groups, n2, k), dtype=torch.float8_e4m3fn, device=device
    )
    a_scale = _require_tensor(
        a_scale, "a_scale", shape=(m, k // K_BLOCK), dtype=torch.float32, device=device
    )
    b_scale = _require_tensor(
        b_scale,
        "b_scale",
        shape=(groups, n2 // GROUP_SIZE, k // K_BLOCK),
        dtype=torch.float32,
        device=device,
    )
    m_indices = _require_tensor(
        m_indices, "m_indices", shape=(m,), dtype=torch.int32, device=device
    )
    if out_q is None:
        out_q = torch.empty((m, h), dtype=torch.float8_e4m3fn, device=device)
    out_q = _require_tensor(
        out_q, "out_q", shape=(m, h), dtype=torch.float8_e4m3fn, device=device
    )
    if out_s is None:
        out_s = torch.empty((m, h // GROUP_SIZE), dtype=torch.float32, device=device)
    out_s = _require_tensor(
        out_s,
        "out_s",
        shape=(m, h // GROUP_SIZE),
        dtype=torch.float32,
        device=device,
        alignment=4,
    )
    group_blocks: Optional[tuple[int, ...]] = None
    if group_counts is not None:
        group_blocks = _group_blocks_from_counts(group_counts, m, groups)
    if validate_indices:
        validated = _validate_indices(m_indices, groups, allow_padding=fill_padding)
        if group_blocks is not None and validated != group_blocks:
            raise ValueError(
                f"group_counts {list(group_counts)} disagree with m_indices (blocks {list(validated)})"
            )
        group_blocks = validated

    padding_scratch = None
    kernel_indices = m_indices
    if fill_padding:
        kernel_indices = torch.empty((m,), dtype=torch.int32, device=device)
        padding_scratch = (
            kernel_indices,
            torch.empty((m,), dtype=torch.int64, device=device),
        )

    sm_count = device_sm_count(device_index)
    route, grid = launch_plan(m, n2, sm_count=sm_count, group_blocks=group_blocks, k=k)
    stage_grids: dict[str, tuple[int, int, int]] = {ROUTE_STAGES[route][0]: grid}
    if route == FUSED_ROUTE:
        stage_grids[FUSED_TAIL_STAGE] = tail_launch_grid(m, n2, sm_count=sm_count)
    stage_module_names = {
        stage: select_stage_module(arch, route, stage) for stage in ROUTE_STAGES[route]
    }
    module_name = stage_module_names[ROUTE_STAGES[route][0]]

    gemm_backend: Optional[str] = None
    gemm: Optional[Callable[[], Any]] = None
    gemm_out: Optional[torch.Tensor] = None
    if route in FUSED_ROUTES:
        bindings: dict[str, Any] = {
            "A": a.view(torch.uint8),
            "B": b.view(torch.uint8),
            "out_q": out_q,
            "out_s": out_s,
            "a_scale": a_scale,
            "b_scale": b_scale,
            "m_indices": kernel_indices,
            "M": m,
            "N": n2,
            "K": k,
            "G": groups,
        }
        if route == FUSED_MIXED_ROUTE:
            # the solo units' 64-row A box: a second descriptor over the same A tensor
            bindings["A64"] = bindings["A"]
    else:
        gemm_backend = small_m_gemm_backend(m, n2, k)
        if workspace is None:
            workspace = torch.empty((m, n2), dtype=torch.bfloat16, device=device)
        gemm_out = _require_tensor(
            workspace, "workspace", shape=(m, n2), dtype=torch.bfloat16, device=device
        )
        if gemm_backend == GEMM_BACKEND_CAKE:
            prepared_gemm = prepare_group_gemm_fp8_nt_groupwise_contiguous(
                a, b, a_scale, b_scale, kernel_indices, out=gemm_out
            )

            def gemm(
                a: Optional[torch.Tensor] = None,
                a_scale: Optional[torch.Tensor] = None,
                m_indices: Optional[torch.Tensor] = None,
                bound: tuple[torch.Tensor, ...] = (a, a_scale, kernel_indices),
                y: torch.Tensor = gemm_out,
            ) -> None:
                if a is None and a_scale is None and m_indices is None:
                    prepared_gemm.launch()
                    return
                # Rebound operands: prepare the GEMM on them (host work only, so the
                # call stays graph-capturable).
                prepare_group_gemm_fp8_nt_groupwise_contiguous(
                    bound[0] if a is None else a,
                    b,
                    bound[1] if a_scale is None else a_scale,
                    b_scale,
                    bound[2] if m_indices is None else m_indices,
                    out=y,
                ).launch()

        else:
            from .gemm_base import group_gemm_fp8_nt_groupwise_contiguous

            def gemm(
                a: Optional[torch.Tensor] = None,
                a_scale: Optional[torch.Tensor] = None,
                m_indices: Optional[torch.Tensor] = None,
                bound: tuple[torch.Tensor, ...] = (a, a_scale, kernel_indices),
                y: torch.Tensor = gemm_out,
            ) -> None:
                group_gemm_fp8_nt_groupwise_contiguous(
                    bound[0] if a is None else a,
                    b,
                    bound[1] if a_scale is None else a_scale,
                    b_scale,
                    bound[2] if m_indices is None else m_indices,
                    out=y,
                )

            # The CuTe-DSL kernel is specialized (compiled on a cold cache) by its
            # first call: run it once here, into the intermediate it owns anyway,
            # so that every launch() only submits work and may be captured.
            if padding_scratch is not None:
                _forward_fill_padding(m_indices, *padding_scratch)
            gemm()

        bindings = {"y": gemm_out, "out_q": out_q, "out_s": out_s, "M": m, "H": h}

    entries = []
    for stage in ROUTE_STAGES[route]:
        stage_module_name = stage_module_names[stage]
        program = PROGRAMS[MODULES[stage_module_name]["program"]]
        module = load_cake_grouped_fp8_fused_silu_quant_module(stage_module_name)
        entry = getattr(module, program["ffi_entry"])
        arg_plan = ARG_PLANS[program["arg_plan"]]
        arguments = bind_arguments(
            arg_plan,
            bindings,
            stage_grids[stage],
            module_name=stage_module_name,
        )
        entries.append((entry, arguments, _argument_slots(arg_plan)))
    return PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant(
        route=route,
        module_name=module_name,
        grid=grid,
        out_q=out_q,
        out_s=out_s,
        gemm_backend=gemm_backend,
        stage_module_names=stage_module_names,
        stage_grids=stage_grids,
        m=m,
        n2=n2,
        k=k,
        fill_padding=fill_padding,
        _entries=tuple(entries),
        _gemm=gemm,
        _gemm_out=gemm_out,
        _padding_source=m_indices if fill_padding else None,
        _padding_scratch=padding_scratch,
    )


__all__ = [
    "ACT_ROUTE",
    "ACT_ROUTES",
    "ACT_WIDE_MIN_ITEMS",
    "ACT_WIDE_ROUTE",
    "act_items",
    "FUSED_MIXED_ROUTE",
    "FUSED_ROUTE",
    "FUSED_ROUTES",
    "GEMM_BACKEND_CAKE",
    "GEMM_BACKEND_CUTE",
    "SMALL_M_MAX",
    "MAX_M",
    "PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant",
    "is_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_prepared_available",
    "fused_tile_counts",
    "launch_plan",
    "mixed_clusters",
    "prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant",
    "routing_blocks",
    "select_route",
    "small_m_gemm_backend",
    "tail_launch_grid",
]
