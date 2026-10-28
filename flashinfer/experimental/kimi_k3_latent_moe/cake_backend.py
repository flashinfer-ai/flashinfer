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

Generated-program backend: Kimi-K3 Stable LatentMoE front / tail projections
(SM100 / SM103; flashinfer-ai/flashinfer#4568, tracker #4254).

Per rank and layer, around the routed experts of ``nvidia/Kimi-K3-NVFP4``::

    front:  logits     = FP32(x @ gate_weight.T)                     [T, 896]  router logits
            latent     = BF16(x @ down_weight.T)                     [T, 3584] routed-expert input
            shared_act = SiTU(x @ shared_gate.T, x @ shared_up.T)    [T, 6144 / TP]
    tail:   y   = KimiRMSNorm(sum of the P routed partials)          [T, 3584] (caller-owned workspace)
            out = BF16(y[:, cols] @ up_weight[:, cols].T + shared_act @ shared_down.T)   [T, 7168]
                  (cols = this rank's 3584 / TP latent slice; TP > 1 all-reduces ``out`` outside)

The host route mirrors the Cake production launcher exactly: for ``T <= 128``
tokens one weight-streaming swapped-AB tcgen05 kernel per stage (the tail's
KimiRMSNorm is fused into that launch), for ``T > 128`` a persistent 2-CTA
tcgen05 GEMM per stage (the tail additionally runs a one-pass RMSNorm kernel
first and launches its GEMM programmatic-dependent).  The decode planner
(N padding, ring depth, cluster pairs, resident B operand, staged rows, issue
gate) and the prefill planner (persistent pair tiles, trailing-wave stream-K,
the front's fold form, norm PDL trigger) are re-implemented here byte-for-byte from the Cake modules
``kimi_k3_latent_moe_decode`` / ``kimi_k3_latent_moe_front`` /
``kimi_k3_latent_moe_tail``; every plan names the physical generated kernel
through a logical key resolved in ``cake_jit.KERNELS``.  Weights are read in
the model layout (``nn.Linear`` ``[out, in]`` BF16); nothing is copied or
packed, and a prepared runner is CUDA Graph safe.  Distinct runners share no
mutable device state across streams: each prepared runner owns its
zero-initialised control words (``_CtrlSlab``), and an eager launch of a split
prefill plan uses the FP32 stream-K partial workspace of its (device, stream),
grown to the largest split plan launched there (a launch under capture takes a
block of the graph's memory pool).  Every
plan is derived from the device's own architecture and SM count (``QUALIFIED_SM_COUNTS``
names the physical counts the programs are qualified on; other counts are refused).
See ``README.md`` in this package (section "Concurrency").
"""

from __future__ import annotations

import ctypes
import functools
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from . import cake_jit
from .cake_jit import (
    MODULES,
    kernel_module_name,
    load_cake_kimi_k3_latent_moe_module,
)

HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896
SHARED_INTERMEDIATE = 6144
RMS_EPS = 1.0e-5
SUPPORTED_TP = (1, 8)
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
#: Reference SM count of the frozen plan rules (B200 and B300 expose 148 SMs): the default
#: argument of the host planners when no device is given (CPU callers, ``required_kernel_keys``).
SM_COUNT = 148
#: Physical SM counts the programs are qualified on: 148 (B200, B300) and 152 (GB200, GB300).
#: ``prepare_*`` plans every route from the device's ``multi_processor_count`` -- the decode
#: grids are tile-bound (94 / 112 / 131 co-resident CTAs on either count) and the prefill plans
#: pass their resident-pair windows (``num_items`` / ``sk_*``) as runtime arguments -- and admits
#: only these counts: the cooperative programs (the fused decode tail instances, the fused tail
#: GEMMs and the front's fold form) need every CTA of their grid resident and the stream-K
#: windows were validated on these parts only (README, "Tested physical configurations").
QUALIFIED_SM_COUNTS = (148, 152)
#: Largest token count served by the decode (weight-streaming) kernels.
DECODE_MAX_T = 128
#: Token counts of the validated route set (both stages, TP 1 and 8).
ROW_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

# ---------------------------------------------------------------------------
# Decode planner (port of the Cake decode kernel module ``kimi_k3_latent_moe_decode``)
# ---------------------------------------------------------------------------

BLOCK_ROWS = 128
HALF_ROWS = 64
CHUNK_K = 64
EPI_WARPS = 4
MAX_DYN_SMEM = 232448
SMEM_BARRIER_RESERVE = 1024
FLAG_BYTES = 1024
SUPPORTED_N_PAD = (8, 16, 32, 64, 128)
MAX_STAGES = 12
GATE_MAX_DOWN_UNITS = 16
GATE_PRO = 3
SMEM_B1_MAX_TOKENS = 2 * EPI_WARPS
SMEM_B1_MIN_STAGES = 6
SMEM_B1_MAX_TOKENS_RS = 4 * EPI_WARPS
SMEM_B1_MIN_STAGES_RS = 5
# Round-7 lever 6a (Cake ``kimi_k3_latent_moe_decode._land_alias_auto``): in cluster mode the DSMEM landing zone of the
# K-split combine aliases the drained A ring instead of owning smem, which buys the N_PAD 128 ring two more stages
# (5 -> 7). The pair handshake it needs costs the short TP8 tail rows 1-2 %, so only the long weight streams (>= 32
# chunk units per CTA of the pair) take it: the tail TP1 T=128 and front TP8 T=128 instances.
LAND_ALIAS_N_PAD = 128
LAND_ALIAS_MIN_UNITS = 32


def land_alias_auto(n_pad: int, k_units: int) -> bool:
    return int(n_pad) == LAND_ALIAS_N_PAD and int(k_units) // 2 >= LAND_ALIAS_MIN_UNITS


TAIL_STAGE_TRIM_MAX_N_PAD = 8
TAIL_STAGE_TRIM = 1
ROWS_NO_TMAP_PREFETCH_MIN_N_PAD = 16
# Round-7 lever 3b: the tail instance that skips the tensor-map prefetch (staged rows at N_PAD 16) defers the cluster
# rendezvous to the epilogue warps only, so its TMA / MMA warps start the ring fill right after the CTA-local setup
# (+2.7 / +2.1 % on B200 / B300 in two processes); every other instance waits for the whole pair before streaming.
#: Front stage: K depth 2 while the depth-2 ring keeps this many stages per cluster mode
#: (1 = one CTA per tile, 2 = cluster pairs); the tail keeps depth 1 (kernel module constant).
FRONT_KD2_MIN_STAGES = {1: 4, 2: 6}
TL_SLOTS = 16
TL_MAX_GRID = 512
DECODE_THREADS = 192
#: Staggered cold fill of the staged-rows instance without the descriptor prefetch (the TP8 N_PAD 16 tail decode
#: instance): the TMA warp waits for ring stage 0 to land before issuing stage 2 (symbol suffix ``_st2``).
TAIL_ROWS_STAGGER = 2


def _cached_plan(fn):
    """Memoise a pure host plan (integer / boolean arguments); callers receive their own copy."""
    cached = functools.lru_cache(maxsize=None)(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return dict(cached(*args, **kwargs))

    wrapper.cache_clear = cached.cache_clear
    return wrapper


# Exact keyword set of the decode program's ``run`` entry (argument plan of the
# generated binding); ``grid`` is expanded to ``grid_x/y/z``.
DECODE_KWARGS = (
    "A_R",
    "A_L",
    "A_S",
    "A_2",
    "B_1",
    "B_2",
    "out_r",
    "out_l",
    "out_s",
    "counters",
    "routed",
    "norm_w",
    "y_out",
    "tl",
    "num_tokens",
    "k1_off",
    "num_partials",
    "eps",
    "grid",
)


def n_pad_for(num_tokens: int) -> int:
    for n in SUPPORTED_N_PAD:
        if num_tokens <= n:
            return n
    raise ValueError(
        f"the decode kernels support at most {SUPPORTED_N_PAD[-1]} tokens, got {num_tokens}"
    )


def plan_stages(
    *,
    n_pad: int,
    kdepth: int = 1,
    max_stages: int = MAX_STAGES,
    cluster: int = 1,
    extra_bytes: int = 0,
    land_alias: bool = False,
    split_fin: bool = False,
) -> int:
    stage_bytes = (BLOCK_ROWS + n_pad) * CHUNK_K * 2 * kdepth
    # Cluster mode: the DSMEM landing zone of the K-split combine owns smem unless it aliases the drained A ring
    # (``land_alias``); with the split finalize (``split_fin``) each CTA receives half the partial, so the zone is
    # half as large.  Single-CTA instances keep the 1 KiB flag/landing prefix.
    land_bytes = n_pad * BLOCK_ROWS * (2 if split_fin else 4)
    red_alloc = (0 if land_alias else land_bytes) if cluster == 2 else 1024
    return max(
        1,
        min(
            max_stages,
            (MAX_DYN_SMEM - SMEM_BARRIER_RESERVE - FLAG_BYTES - red_alloc - extra_bytes)
            // stage_bytes,
        ),
    )


def smem_b1_max_tokens(k2_units_per_cta: int) -> int:
    return SMEM_B1_MAX_TOKENS if k2_units_per_cta > GATE_MAX_DOWN_UNITS else EPI_WARPS


def plan_partition(
    *,
    tiles: int,
    k1_chunks: int,
    k2_chunks: int,
    kdepth: int,
    sm_count: int,
    n_pad: int = 8,
) -> dict[str, int]:
    """Aligned cluster pairs when ``2 * tiles`` fits one wave, else one CTA per tile."""
    mpt = (k1_chunks + k2_chunks) // kdepth
    cluster = 2 if (2 * tiles <= sm_count and mpt >= 2 and n_pad % 4 == 0) else 1
    g = 2 * tiles if cluster == 2 else tiles
    if g > sm_count:
        raise ValueError(f"grid {g} exceeds the SM count {sm_count} (co-residency)")
    return {"grid": g, "cluster": cluster, "mpt": mpt, "units": tiles * mpt}


def kdepth_for(*k_chunks: int, requested: Optional[int] = None) -> int:
    for d in (requested,) if requested else (4, 2, 1):
        if d and all(k % d == 0 for k in k_chunks):
            return d
    raise ValueError(f"kdepth {requested} does not divide K segments {k_chunks}")


def choose_config(
    stage: str,
    *,
    tiles: int,
    n_pad: int,
    k_chunks: tuple[int, ...],
    sm_count: int,
    land_alias: bool = False,
    split_fin: bool = False,
) -> tuple[int, int, int, int]:
    """``(kdepth, grid, stages, cluster)`` of the production rule: depth 1 (ring takes all smem);
    the front stage takes depth 2 while its ring keeps ``FRONT_KD2_MIN_STAGES[cluster]`` stages."""
    k2 = k_chunks[1] if len(k_chunks) > 1 else 0
    kd = kdepth_for(*k_chunks, requested=1)
    part = plan_partition(
        tiles=tiles,
        k1_chunks=k_chunks[0],
        k2_chunks=k2,
        kdepth=kd,
        sm_count=sm_count,
        n_pad=n_pad,
    )
    if stage == "front" and all(k % 2 == 0 for k in k_chunks):
        stages2 = plan_stages(
            n_pad=n_pad,
            kdepth=2,
            cluster=part["cluster"],
            land_alias=land_alias,
            split_fin=split_fin,
        )
        if stages2 >= FRONT_KD2_MIN_STAGES[part["cluster"]]:
            part2 = plan_partition(
                tiles=tiles,
                k1_chunks=k_chunks[0],
                k2_chunks=k2,
                kdepth=2,
                sm_count=sm_count,
                n_pad=n_pad,
            )
            return 2, part2["grid"], stages2, part2["cluster"]
    return (
        kd,
        part["grid"],
        plan_stages(
            n_pad=n_pad,
            kdepth=kd,
            cluster=part["cluster"],
            land_alias=land_alias,
            split_fin=split_fin,
        ),
        part["cluster"],
    )


@_cached_plan
def decode_front_plan(
    num_tokens: int, i_local: int, sm_count: int = SM_COUNT
) -> dict[str, Any]:
    """Instance configuration of ``front_decode`` (router + latent + shared SiTU tiles, one launch)."""
    if i_local % HALF_ROWS:
        raise ValueError("I_LOCAL must be a multiple of 64")
    n_pad = n_pad_for(num_tokens)
    r_tiles, l_tiles, s_tiles = (
        NUM_EXPERTS // BLOCK_ROWS,
        LATENT // BLOCK_ROWS,
        i_local // HALF_ROWS,
    )
    tiles = r_tiles + l_tiles + s_tiles
    k_chunks = HIDDEN // CHUNK_K
    land_alias = land_alias_auto(n_pad, k_chunks)
    split_fin = split_fin_route("front", n_pad)
    kdepth, grid, stages, cluster = choose_config(
        "front",
        tiles=tiles,
        n_pad=n_pad,
        k_chunks=(k_chunks,),
        sm_count=sm_count,
        land_alias=land_alias,
        split_fin=split_fin,
    )
    land_alias = land_alias and cluster == 2
    split_fin = split_fin and cluster == 2
    l2_promotion = (
        i_local == i_local_for_tp(FRONT_L2_PROMOTION_TP)
        and n_pad <= FRONT_L2_PROMOTION_MAX_N_PAD
    )
    return dict(
        grid=grid,
        n_pad=n_pad,
        stages=stages,
        r_tiles=r_tiles,
        l_tiles=l_tiles,
        s_tiles=s_tiles,
        i_local=i_local,
        k1_chunks=k_chunks,
        k2_chunks=0,
        out_r_ld=NUM_EXPERTS,
        out_l_ld=LATENT,
        out_s_ld=i_local,
        kdepth=kdepth,
        pdl=False,
        packed=False,
        pack_k1=k_chunks,
        stream_only=False,
        cluster=int(cluster),
        fused=False,
        fused_probe=0,
        issue_gate=False,
        smem_b1=False,
        rows_smem=False,
        timeline=False,
        gate_pro=GATE_PRO,
        tmap_prefetch=True,
        stagger=0,
        wait_warps=False,
        land_alias=bool(land_alias),
        a_evict_first=DECODE_A_EVICT_FIRST,
        a_l2_promotion=DECODE_A_L2_PROMOTION if l2_promotion else "none",
        split_fin=bool(split_fin),
        early_arrive=False,
        up_prefetch=False,
        sb_norm=False,
        partial_prefetch=False,
        tiles=tiles,
    )


@_cached_plan
def decode_tail_plan(
    num_tokens: int,
    i_local: int,
    tp: int,
    num_partials: int = 1,
    sm_count: int = SM_COUNT,
) -> dict[str, Any]:
    """Instance configuration of ``tail_decode_fused`` (port of ``_tail_plan`` with the production defaults); the
    few-token multi-partial rows of ``TAIL_SB_NORM_ROUTES`` take the global-y instance of their N_PAD with the
    resident-B arithmetic (``sb_norm``), and the cells of ``TAIL_PARTIAL_PREFETCH_ROUTES`` the instance the table
    names with the routed partials prefetched into L2 (``partial_prefetch``)."""
    sb_norm_n_pad = tail_sb_norm_n_pad(num_tokens, tp, num_partials)
    prefetch = tail_partial_prefetch(num_tokens, tp, num_partials)
    if prefetch is not None:
        sb_norm_n_pad = TAIL_SB_NORM_ROUTES[int(tp)][0] if prefetch else 0
    if sb_norm_n_pad:
        plan = decode_tail_plan(sb_norm_n_pad, i_local, tp, num_partials, sm_count)
        if plan["smem_b1"] or plan["split_fin"] or plan["partial_prefetch"]:
            raise AssertionError(
                f"tp{tp} N_PAD {sb_norm_n_pad} P={num_partials}: not a global-y instance"
            )
        plan["sb_norm"] = True
        plan["partial_prefetch"] = prefetch is not None
        return plan
    n_pad = n_pad_for(num_tokens)
    tiles = HIDDEN // BLOCK_ROWS
    k_up = LATENT // tp
    if k_up % CHUNK_K or i_local % CHUNK_K:
        raise ValueError("K slices must be multiples of 64")
    k1, k2 = k_up // CHUNK_K, i_local // CHUNK_K
    land_alias = land_alias_auto(n_pad, k1 + k2)
    split_fin = split_fin_route("tail", n_pad, num_partials)
    kdepth, grid, stages, cluster = choose_config(
        "tail",
        tiles=tiles,
        n_pad=n_pad,
        k_chunks=(k1, k2),
        sm_count=sm_count,
        land_alias=land_alias,
        split_fin=split_fin,
    )
    land_alias = land_alias and cluster == 2
    split_fin = split_fin and cluster == 2
    k1_macros = k1 // kdepth
    cl_u0 = (k1_macros + 1) // 2 if cluster == 2 else k1_macros
    bn_units = max(cl_u0, k1_macros - cl_u0) if cluster == 2 else k1_macros
    bn_bytes = bn_units * n_pad * CHUNK_K * 2 * kdepth
    k2_units = -(-k2 // (2 if cluster == 2 else 1))
    rows_ok = kdepth == 1 and int(num_partials) == 1 and k2_units <= GATE_MAX_DOWN_UNITS
    rows_bytes = (n_pad * LATENT + LATENT) * 2
    use_bn = kdepth == 1 and num_tokens <= (
        SMEM_B1_MAX_TOKENS_RS if rows_ok else smem_b1_max_tokens(k2_units)
    )
    use_rows = use_bn and rows_ok
    gate_default = k2_units <= GATE_MAX_DOWN_UNITS and not use_rows
    if use_bn:
        bn_stages = plan_stages(
            n_pad=n_pad,
            kdepth=kdepth,
            cluster=cluster,
            extra_bytes=bn_bytes + (rows_bytes if use_rows else 0),
            land_alias=land_alias,
            split_fin=split_fin,
        )
        if bn_stages < SMEM_B1_MIN_STAGES_RS and use_rows:
            use_rows = False
            use_bn = num_tokens <= smem_b1_max_tokens(k2_units)
            gate_default = k2_units <= GATE_MAX_DOWN_UNITS
            bn_stages = plan_stages(
                n_pad=n_pad,
                kdepth=kdepth,
                cluster=cluster,
                extra_bytes=bn_bytes,
                land_alias=land_alias,
                split_fin=split_fin,
            )
        if use_bn and bn_stages < (
            SMEM_B1_MIN_STAGES_RS if use_rows else SMEM_B1_MIN_STAGES
        ):
            use_bn = False
            use_rows = False
        elif use_bn:
            stages = bn_stages
    if n_pad <= TAIL_STAGE_TRIM_MAX_N_PAD:
        stages = max(
            stages - TAIL_STAGE_TRIM,
            SMEM_B1_MIN_STAGES_RS
            if use_rows
            else (SMEM_B1_MIN_STAGES if use_bn else 2),
        )
    if prefetch is not None and not (use_bn and not use_rows):
        raise AssertionError(
            f"tp{tp} T={num_tokens} P={num_partials}: not a resident-B instance without staged rows"
        )
    # The global-y protocol's levers (the resident-B instances normalise in smem and take neither).
    early_arrive = split_fin and not use_bn and tp == TAIL_EARLY_ARRIVE_TP
    up_prefetch = (
        split_fin
        and not use_bn
        and tp == TAIL_UP_PREFETCH_TP
        and n_pad in TAIL_UP_PREFETCH_N_PAD
    )
    return dict(
        grid=grid,
        n_pad=n_pad,
        stages=stages,
        r_tiles=0,
        l_tiles=tiles,
        s_tiles=0,
        i_local=i_local,
        k1_chunks=k1,
        k2_chunks=k2,
        out_r_ld=NUM_EXPERTS,
        out_l_ld=HIDDEN,
        out_s_ld=i_local,
        kdepth=kdepth,
        pdl=False,
        packed=False,
        pack_k1=LATENT // CHUNK_K,
        stream_only=False,
        cluster=int(cluster),
        fused=True,
        fused_probe=0,
        issue_gate=bool(gate_default),
        smem_b1=bool(use_bn),
        rows_smem=bool(use_rows),
        timeline=False,
        gate_pro=GATE_PRO,
        tmap_prefetch=not (use_rows and n_pad >= ROWS_NO_TMAP_PREFETCH_MIN_N_PAD),
        stagger=TAIL_ROWS_STAGGER
        if (use_rows and n_pad >= ROWS_NO_TMAP_PREFETCH_MIN_N_PAD)
        else 0,
        wait_warps=bool(use_rows and n_pad >= ROWS_NO_TMAP_PREFETCH_MIN_N_PAD),
        land_alias=bool(land_alias),
        a_evict_first=bool(DECODE_A_EVICT_FIRST and tp == 1),
        a_l2_promotion=DECODE_A_L2_PROMOTION if tp == TAIL_L2_PROMOTION_TP else "none",
        split_fin=bool(split_fin),
        early_arrive=bool(early_arrive),
        up_prefetch=bool(up_prefetch),
        sb_norm=False,
        partial_prefetch=prefetch is not None,
        tiles=tiles,
        k1=k1,
        k2=k2,
    )


# The decode kernels stream every weight box once per launch: they load it with the ``evict_first`` L2 policy, so the
# stream evicts its own lines first and the activations and outputs other kernels left in L2 survive it (front TP1 / TP8
# and tail TP1 rows 8-16 % faster under a cold, dirty L2); the TP8 tail's short stream keeps the default policy (neutral).
DECODE_A_EVICT_FIRST = True
# The TP1 tail and the TP8 front up to N_PAD 64 also promote the weight maps' L2 fills to 256 bytes
# (``a_l2_promotion``, symbol suffix ``_p256b``): a weight box row's 128-byte miss brings the row's next 128 bytes (the
# K chunk the same CTA streams next) into L2 with the same DRAM access.  Host-side tensor-map encoding only: the
# device code and the outputs are the unpromoted instance's.  The TP1 front, the TP8 front at N_PAD 128 and the TP8
# tail keep no promotion (slower or mixed with it).
DECODE_A_L2_PROMOTION = "l2_256b"
TAIL_L2_PROMOTION_TP = 1
FRONT_L2_PROMOTION_TP = 8
FRONT_L2_PROMOTION_MAX_N_PAD = 64
# Split finalize (``split_fin``, symbol suffix ``_sf``): both CTAs of a cluster pair finalize half the values -- each
# pushes the partial quads its peer finalizes into the peer's landing zone and adds the quads it receives -- so the pair
# combine, the epilogue math and the output stores are split over the pair and each landing zone holds half the partial
# (one more ring stage for the TP8 tail at N_PAD 128).  Every value is still rank 0's partial plus rank 1's, so the
# outputs are bitwise those of the single-finisher instance.  The cluster-pair instances take it: the TP8 front from
# N_PAD 16 and the fused-norm tails of one routed partial from N_PAD 32 (P >= 2 keeps the single finisher; at N_PAD 16
# the tails measured within run-to-run noise of the single finisher).
SPLIT_FIN_MIN_N_PAD = 16
TAIL_SPLIT_FIN_MIN_N_PAD = 32
# With it, the TP1 tail's epilogue warp 0 arrives on the fused norm's grid counter as soon as the CTA's normalised rows
# are written (``early_arrive``, ``_ea``) instead of its TMA warp at the first up-segment unit, and the TP8 tail's TMA
# warp at N_PAD 32 and 64 prefetches the weight tiles of its first up-segment units into L2 while the grid's rows are
# pending (``up_prefetch``, ``_upf``).
TAIL_EARLY_ARRIVE_TP = 1
TAIL_UP_PREFETCH_TP = 8
TAIL_UP_PREFETCH_N_PAD = (32, 64)
# Resident-B arithmetic on the global-y tails (``sb_norm``, symbol suffix ``_sbn``).  The resident-B tail instances
# (``_sb``: TP1 up to 8 tokens, TP8 up to 4) normalise every row in every CTA: each CTA sums the P routed partials of
# all T rows in a dependent loop before its first up unit, one HBM round trip per partial under the weight stream, and
# from three partials (TP1, five to eight tokens) the loop outlasts the down units it overlaps.  The global-y instance
# at N_PAD ``n`` with ``sb_norm`` normalises each row once, in the warp that owns it, with the resident-B arithmetic
# statement for statement (bf16 of the FP32 partial sum, sum of squares in element order, warp reduction, rsqrt,
# ``bf16(bf16(sum) x rstd)``, ``bf16(w x normed)`` in FP32), so its ``y`` and ``out`` are bitwise the resident-B
# instance's for every input, signed zeros and subnormals included.  ``TAIL_SB_NORM_ROUTES`` maps a TP to the N_PAD of
# that instance and, per token count, the fewest partials it serves: the cells where it measured faster than the
# resident-B instance on B300, GB300 and GB200 (paired CUDA-graph timing, both passes beyond the A/A band).  The TP8
# instance serves only with the routed partials prefetched (``TAIL_PARTIAL_PREFETCH_ROUTES``), faster on every cell it
# served, so its entry keeps the N_PAD alone.
TAIL_SB_NORM_ROUTES: dict[int, tuple[int, dict[int, int]]] = {
    1: (32, {1: 5, 2: 5, 3: 5, 4: 5, 5: 3, 6: 3, 7: 3, 8: 3}),
    8: (8, {}),
}


def split_fin_route(stage: str, n_pad: int, num_partials: int = 1) -> bool:
    """Whether the decode instance of ``stage`` at ``n_pad`` (and, for the tail, ``num_partials`` routed partials)
    takes the split finalize when it runs as cluster pairs."""
    if stage != "tail":
        return int(n_pad) >= SPLIT_FIN_MIN_N_PAD
    return int(n_pad) >= TAIL_SPLIT_FIN_MIN_N_PAD and int(num_partials) == 1


def tail_sb_norm_n_pad(num_tokens: int, tp: int, num_partials: int) -> int:
    """N_PAD of the global-y instance with the resident-B arithmetic that serves the decode tail of ``num_tokens``
    tokens and ``num_partials`` routed partials at ``tp`` (``TAIL_SB_NORM_ROUTES``), or 0 for the default instance."""
    n_pad, min_partials = TAIL_SB_NORM_ROUTES.get(int(tp), (0, {}))
    least = min_partials.get(int(num_tokens))
    return n_pad if least is not None and int(num_partials) >= least else 0


# Routed-partial L2 prefetch (``partial_prefetch``, symbol suffix ``_ppf``).  The resident-B tail instance and the
# global-y instances with the resident-B arithmetic sum the P routed partials in a dependent loop, one HBM round trip
# per partial under the weight stream.  Their ``_ppf`` forms warm the partial rows in L2 with 7 KiB bulk prefetches
# issued as soon as the partial stride is known (the global-y instance: lanes 1 .. P-1 of the warp that owns a row; the
# resident-B instance: the (row, partial) pairs spread over the grid, one prefetch per pair); the arithmetic, and so
# every output, is the instance's own.  ``TAIL_PARTIAL_PREFETCH_ROUTES`` maps a TP and a token count to the partial
# ranges ``(first, last, sb_norm)`` each form serves -- ``sb_norm`` true: the global-y instance of
# ``TAIL_SB_NORM_ROUTES`` (``_sbn_ppf``), false: the resident-B instance (``_sb..._ppf``, TP1); ``last`` ``None``: from
# ``first`` up, as ``TAIL_SB_NORM_ROUTES`` serves from its fewest partials up -- the cells where it measured faster
# than the route without it on every part measured (paired CUDA-graph timing, both passes beyond the A/A band; 2 to
# 16 partials).
_PrefetchCells = dict[int, tuple[tuple[int, Optional[int], bool], ...]]
TAIL_PARTIAL_PREFETCH_ROUTES: dict[int, _PrefetchCells] = {
    1: {
        1: ((4, 6, False), (7, None, True)),
        2: ((4, 6, False), (7, None, True)),
        3: ((4, 6, False), (7, None, True)),
        4: ((4, 6, False), (7, None, True)),
        5: ((3, 3, False), (5, None, True)),
        6: ((3, 3, False), (5, None, True)),
        7: ((3, 3, False), (5, None, True)),
        8: ((3, 3, False), (5, None, True)),
    },
    8: {
        1: ((4, None, True),),
        2: ((4, None, True),),
        3: ((4, None, True),),
        4: ((3, None, True),),
    },
}


def tail_partial_prefetch(
    num_tokens: int, tp: int, num_partials: int
) -> Optional[bool]:
    """``sb_norm`` of the routed-partial prefetch instance that serves the decode tail of ``num_tokens`` tokens and
    ``num_partials`` routed partials at ``tp`` (``TAIL_PARTIAL_PREFETCH_ROUTES``), or ``None`` for the route without it."""
    cells = TAIL_PARTIAL_PREFETCH_ROUTES.get(int(tp), {})
    for first, last, sb_norm in cells.get(int(num_tokens), ()):
        if first <= int(num_partials) and (last is None or int(num_partials) <= last):
            return sb_norm
    return None


def decode_symbol(plan: dict[str, Any]) -> str:
    """Kernel symbol of one decode instance (``build_decode_gemm_ir`` naming rule)."""
    fused = bool(plan["fused"])
    probe = f"_fp{plan['fused_probe']}" if (fused and plan["fused_probe"]) else ""
    gate = f"_g{plan['gate_pro']}" if (fused and plan["issue_gate"]) else ""
    smem_b1 = "_sb" if plan["smem_b1"] else ""
    rows = "_rs" if (plan["smem_b1"] and plan["rows_smem"]) else ""
    stagger = f"_st{int(plan['stagger'])}" if plan["stagger"] else ""
    return (
        f"kimi_k3_latent_moe_decode_g{plan['grid']}_n{plan['n_pad']}_r{plan['stages']}_d{plan['kdepth']}"
        f"_p{1 if plan['pdl'] else 0}_w{1 if plan['packed'] else 0}"
        f"_t{plan['r_tiles']}_{plan['l_tiles']}_{plan['s_tiles']}_k{plan['k1_chunks']}_{plan['k2_chunks']}"
        f"_o{plan['out_l_ld']}{'_so' if plan['stream_only'] else ''}_c{plan['cluster']}{'_f' if fused else ''}"
        f"{probe}{gate}{smem_b1}{rows}{'_tl' if plan['timeline'] else ''}{'' if plan['tmap_prefetch'] else '_np'}"
        f"{stagger}"
        f"{'_ww' if plan['wait_warps'] else ''}{'_la' if plan['land_alias'] else ''}"
        f"{'_ef' if plan.get('a_evict_first') else ''}"
        f"{_l2_promotion_suffix(plan.get('a_l2_promotion', 'none'))}"
        f"{'_sf' if plan.get('split_fin') else ''}{'_ea' if plan.get('early_arrive') else ''}"
        f"{'_upf' if plan.get('up_prefetch') else ''}{'_sbn' if plan.get('sb_norm') else ''}"
        f"{'_ppf' if plan.get('partial_prefetch') else ''}"
    )


def _l2_promotion_suffix(promotion: str) -> str:
    """``_p256b`` for ``l2_256b`` (the kernel module's naming rule), nothing without promotion."""
    return "" if promotion == "none" else "_" + promotion.replace("l2_", "p")


def decode_kernel_key(plan: dict[str, Any]) -> str:
    return f"decode:{decode_symbol(plan)}"


# ---------------------------------------------------------------------------
# Prefill planner (ports of kimi_k3_latent_moe_front / kimi_k3_latent_moe_tail)
# ---------------------------------------------------------------------------

BLOCK_M = 128
BLOCK_N = 256
B_HALF_N = 128
CTA_GROUP = 2
BLOCK_K = 64
FRONT_R_TILES = (NUM_EXPERTS + BLOCK_N - 1) // BLOCK_N  # 4
FRONT_L_TILES = LATENT // BLOCK_N  # 14
TAIL_EPI_WARPS = 8  # prefill tail GEMM epilogue warps (fused norm: one latent row each)
GROUP_M = 16
NORM_THREADS = 128
NORM_ROWS_PER_CTA = NORM_THREADS // 32
MAX_SEG = 4
SK_MIN_NUM_K = 64
SK_MIN_ITERS = 32
SK_FIXUP_ITERS = 12
SK_MIN_SAVING_ITERS = (
    32  # a stream-K cut must undercut the whole-tile plan by this many K iterations
)
SK_MIN_REUSE_ROWS = 8
SK_PARTIAL_REUSE_MAX_REM_PCT = 55

FRONT_KWARGS = (
    "A",
    "WG",
    "WD",
    "WS",
    "logits",
    "latent",
    "shared_act",
    "ws",
    "counters",
    "M",
    "m_tiles",
    "num_items",
    "full_items",
    "sk_ipc",
    "sk_max_seg",
    "sk_total",
    "grid",
)
NORM_KWARGS = ("routed", "norm_weight", "y_out", "M", "num_partials", "eps", "grid")
TAIL_GEMM_KWARGS = (
    "A1",
    "B1",
    "A2",
    "B2",
    "out",
    "out_map",
    "ws",
    "counters",
    "M",
    "m_tiles",
    "k0_blocks",
    "num_items",
    "full_items",
    "sk_ipc",
    "sk_max_seg",
    "sk_total",
    "routed",
    "norm_weight",
    "y_out",
    "norm_counter",
    "num_partials",
    "eps",
    "grid",
)


def i_local_for_tp(tp: int) -> int:
    if tp not in SUPPORTED_TP:
        raise ValueError(f"tp must be one of {SUPPORTED_TP}, got {tp}")
    return SHARED_INTERMEDIATE // tp


def k_up_for_tp(tp: int) -> int:
    if tp not in SUPPORTED_TP:
        raise ValueError(f"tp must be one of {SUPPORTED_TP}, got {tp}")
    return LATENT // tp


def front_s_tiles(i_local: int, block_n: int = BLOCK_N) -> int:
    """Shared column tiles: ``block_n / 2`` output columns per pair tile (CTA0 the gate rows, CTA1 the up rows); the
    224-wide tile's last one holds the remainder (the kernel stores no column past ``I_local``)."""
    half = block_n // CTA_GROUP
    if block_n == BLOCK_N and i_local % half:
        raise ValueError(f"I_LOCAL must be a multiple of {half}, got {i_local}")
    return (i_local + half - 1) // half


def front_n_tiles(i_local: int, block_n: int = BLOCK_N) -> int:
    return (
        (NUM_EXPERTS + block_n - 1) // block_n
        + LATENT // block_n
        + front_s_tiles(i_local, block_n)
    )


def m_tiles_for(M: int) -> int:
    tiles = (M + BLOCK_M - 1) // BLOCK_M
    return tiles + (tiles % CTA_GROUP)


def front_grid(m_tiles: int, i_local: int) -> int:
    """One cluster (CTA pair) per output tile pair; CLC hands out the rest in raster order."""
    return (m_tiles // CTA_GROUP) * front_n_tiles(i_local) * CTA_GROUP


FRONT_BLOCK_N = 256
FRONT_NUM_K_ITERS = HIDDEN // BLOCK_K  # 112 K iterations per 256 x 256 pair tile
FRONT_SPLIT = 2  # trailing-wave tiles cut into two aligned K halves (round-8 rule)
FRONT_SK_EPI_ITERS = (
    18  # exposed stream-K epilogue of the last contributor, in K iterations
)
FRONT_SK_MIN_GAIN = (
    0.04  # split only when the modelled saving is >= 4 % of the whole-tile cost
)
FRONT_EVICT_FIRST_MAX_PAIR_ROWS = (
    2  # weight boxes re-read by <= 2 pair rows stream evict_first
)
#: Larger re-read bounds for ``evict_first`` (pair rows) where it measured faster; a re-route of the weights' L2
#: policy only, so the outputs stay bitwise.  Per shared-intermediate width on every qualified part: the TP8 rows of
#: three pair rows (T = 513-768).
FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_I_LOCAL = {768: 3}
#: Per (shared-intermediate width, SM count), above that: on 152-SM parts (GB300 / GB200) every TP8 row up to
#: T = 16384 and the TP1 rows up to T = 2048.  148-SM parts (B200 / B300) keep the bounds above.
FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_PART = {(768, 152): 64, (6144, 152): 8}


def front_split_plan(cluster_tiles: int, sm_count: int = SM_COUNT) -> dict[str, int]:
    """Round-8 front work plan (mirrors ``kimi_k3_latent_moe_front.front_split_plan``): the full
    waves run whole 256 x 256 pair tiles; the trailing partial wave (``rem`` tiles) is cut into two
    aligned K halves per tile when the halves fit one wave (``2 * rem <= resident``) and the modelled
    saving ``(56 - FRONT_SK_EPI_ITERS) / whole cost`` reaches ``FRONT_SK_MIN_GAIN``.  Aligned halves
    keep the pair rows of one weight column in lockstep (the column's weight boxes are fetched from
    HBM once); every other cut lost on both GPUs.  Keys as the tail's ``split_plan``."""
    resident = max(1, sm_count // CTA_GROUP)
    full_waves, rem = divmod(cluster_tiles, resident)
    whole = {
        "num_items": cluster_tiles,
        "full_items": cluster_tiles,
        "sk_ipc": FRONT_NUM_K_ITERS,
        "sk_max_seg": 1,
        "sk_total": 0,
        "sk_tiles": 0,
    }
    if rem == 0 or rem * FRONT_SPLIT > resident:
        return whole
    whole_cost = (full_waves + 1) * FRONT_NUM_K_ITERS
    saving = FRONT_NUM_K_ITERS - FRONT_NUM_K_ITERS // FRONT_SPLIT - FRONT_SK_EPI_ITERS
    if saving < FRONT_SK_MIN_GAIN * whole_cost:
        return whole
    return {
        "num_items": cluster_tiles - rem + rem * FRONT_SPLIT,
        "full_items": cluster_tiles - rem,
        "sk_ipc": FRONT_NUM_K_ITERS // FRONT_SPLIT,
        "sk_max_seg": FRONT_SPLIT,
        "sk_total": rem * FRONT_NUM_K_ITERS,
        "sk_tiles": rem,
    }


def front_evict_first(m_tiles: int, i_local: int, sm_count: int = SM_COUNT) -> bool:
    """Round-8 lever 13b: shapes whose weight boxes are re-read by at most two pair rows (T <= 512)
    stream the weights with the ``evict_first`` L2 policy (a separate kernel instance); the TP8
    front also at three pair rows (``FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_I_LOCAL``), and further rows
    on 152-SM parts (``FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_PART``).  The decision is the plan's
    ``evict_first`` field; the kernel key composes it (``front_kernel_key``)."""
    i_local = int(i_local)
    limit = FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_PART.get(
        (i_local, int(sm_count)),
        FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_I_LOCAL.get(
            i_local, FRONT_EVICT_FIRST_MAX_PAIR_ROWS
        ),
    )
    return (m_tiles // CTA_GROUP) <= limit


# Round-4 lever T1F stage 2: on 152-SM parts the TP1 front rows of more than 128 and at most 512 tokens (one and two
# pair rows, evict_first weights) run the 224-wide pair tile: 4 router + 16 latent + 55 shared tiles (the last one 96
# columns wide) give 75 per pair row, which fill 75 of the 76 resident clusters where the 256-wide tile's 66 leave 10
# idle (two whole waves at two pair rows).  Whole tiles only; the same K order and accumulation as the 256-wide tile,
# so the outputs are bit-identical.
FRONT_N224_I_LOCAL = SHARED_INTERMEDIATE
FRONT_N224_MAX_T = 512
FRONT_N224_SM_COUNT = 152
FRONT_N224_BLOCK_N = 224


def front_block_n(M: int, i_local: int, sm_count: int = SM_COUNT) -> int:
    """Pair-tile width of the prefill front GEMM instance for ``M`` tokens of one rank on a ``sm_count``-SM device."""
    if (
        int(i_local) == FRONT_N224_I_LOCAL
        and DECODE_MAX_T < int(M) <= FRONT_N224_MAX_T
        and int(sm_count) == FRONT_N224_SM_COUNT
    ):
        return FRONT_N224_BLOCK_N
    return FRONT_BLOCK_N


# Fold form of the prefill front GEMM (key suffix ``fold``, evict_first weights): a static schedule of one item per
# cluster.  Lanes of G clusters split the flattened K iterations of the pair tiles evenly (a unit is one weight column
# times the lane's G pair rows, which its clusters read in step); a lane boundary inside a tile carries the raw FP32
# accumulator to the next lane, and the split tiles of the trailing-wave plan keep its two K runs and their sum
# ``(0 + P0) + P1``, so the outputs are bitwise those of ``front_split_plan``.  Its CTAs wait for words other CTAs
# release, so it launches cooperatively: every CTA of its grid resident at once.  Taken on the parts where it measured
# faster: the TP8 rows of 16 row tiles (T = 1793-2048) with G = 8 on 152-SM parts (GB200 and GB300) and on sm_100a
# 148-SM parts (B200), and the TP8 rows of 8 row tiles (T = 769-1024) with G = 4 on sm_103a 152-SM parts (GB300);
# ``m_tiles_for`` rounds the row-tile count up to an even count.  sm_103a 148-SM parts (B300) keep the trailing-wave
# plan on every row.  The plan travels in the stream-K arguments: ``num_items`` = clusters, ``full_items`` = G,
# ``sk_ipc`` = lanes, ``sk_total`` = flattened K iterations and ``sk_max_seg`` = 2 * t0 + fold_all, t0 the
# trailing-wave plan's whole tiles.
FRONT_FOLD_I_LOCAL = SHARED_INTERMEDIATE // 8  # TP8 only
#: Pair rows per lane (G) by (row tiles, architecture, SM count): the fold form serves these parts only.
FRONT_FOLD_LANE_ROWS_BY_PART = {
    (16, "sm_100a", 152): 8,
    (16, "sm_100a", 148): 8,
    (16, "sm_103a", 152): 8,
    (8, "sm_103a", 152): 4,
}


def front_fold_plan(
    m_tiles: int, i_local: int, sm_count: int = SM_COUNT, arch: str = "sm_100a"
) -> Optional[dict[str, int]]:
    """Fold-form work plan of the prefill front GEMM, or ``None`` where the trailing-wave plan stays.

    ``lanes = sm_count // 2 // G`` lanes of G clusters, one item per cluster; the ``units x 112`` flattened K
    iterations split evenly over the lanes.  ``sk_max_seg`` selects the mode: ``2 * t0`` keeps the split tiles of
    ``front_split_plan`` (``t0`` its whole tiles; every tile when it splits none), plus one when a lane is shorter
    than a unit (``fold_all``).  The workspace holds two ``128 x 256`` FP32 slots and two self-resetting counters
    per CTA (``ws_slots``).  Keys as ``front_split_plan`` plus ``fold`` and ``ws_slots``.
    """
    m_tiles, i_local, sm_count = int(m_tiles), int(i_local), int(sm_count)
    if i_local != FRONT_FOLD_I_LOCAL:
        return None
    lane_rows = FRONT_FOLD_LANE_ROWS_BY_PART.get((m_tiles, str(arch), sm_count))
    if lane_rows is None:
        return None
    pair_rows = m_tiles // CTA_GROUP
    lanes = sm_count // CTA_GROUP // lane_rows
    if pair_rows % lane_rows or lanes < 2:
        return None
    n_tiles = front_n_tiles(i_local)
    total = pair_rows // lane_rows * n_tiles * FRONT_NUM_K_ITERS
    split = front_split_plan(pair_rows * n_tiles, sm_count)
    if split["sk_max_seg"] > 1:
        # The split tiles keep their two K runs: every lane spans one to one and a half units.
        if not lanes * FRONT_NUM_K_ITERS <= total <= lanes * FRONT_NUM_K_ITERS * 3 // 2:
            return None
        max_seg = 2 * split["full_items"]
    else:
        fold_all = total < lanes * FRONT_NUM_K_ITERS
        # A lane at least half a unit long: a unit has at most three parts.
        if fold_all and 2 * total < lanes * FRONT_NUM_K_ITERS:
            return None
        max_seg = 2 * pair_rows * n_tiles + int(fold_all)
    clusters = lanes * lane_rows
    return {
        "num_items": clusters,
        "full_items": lane_rows,
        "sk_ipc": lanes,
        "sk_max_seg": max_seg,
        "sk_total": total,
        "sk_tiles": 0,
        "fold": True,
        "ws_slots": 2 * clusters * CTA_GROUP,
    }


def front_kernel_key(
    i_local: int,
    evict_first: bool = False,
    block_n: int = FRONT_BLOCK_N,
    fold: bool = False,
) -> str:
    """``front:i<I_local>[e1][n224|fold]``: the ``e1`` suffix names the ``evict_first`` weight-policy instance,
    ``n224`` the 224-wide pair tile and ``fold`` the fold-form program."""
    return (
        f"front:i{int(i_local)}"
        + ("e1" if evict_first else "")
        + ("" if block_n == FRONT_BLOCK_N else f"n{int(block_n)}")
        + ("fold" if fold else "")
    )


@_cached_plan
def prefill_front_plan(
    M: int,
    i_local: int,
    sm_count: int = SM_COUNT,
    arch: str = "sm_100a",
    resident_clusters: Optional[int] = None,
) -> dict[str, Any]:
    """Host plan of the prefill front GEMM (one persistent launch) for ``M`` tokens of one rank on an ``arch``
    device of ``sm_count`` SMs.

    ``resident_clusters`` is the device's co-resident cluster count of the fold-form launch: the fold plan needs
    every one of its clusters resident and keeps the trailing-wave plan below that (``None``: the qualified parts'
    count, which holds every fold plan)."""
    m_tiles = m_tiles_for(M)
    block_n = front_block_n(M, i_local, sm_count)
    n_tiles = front_n_tiles(i_local, block_n)
    cluster_tiles = (m_tiles // CTA_GROUP) * n_tiles
    fold = (
        front_fold_plan(m_tiles, i_local, sm_count, arch)
        if block_n == FRONT_BLOCK_N
        else None
    )
    if fold is not None and (
        resident_clusters is None or int(resident_clusters) >= fold["num_items"]
    ):
        # The fold programs stream the weights evict_first.
        sk, evict_first = fold, True
    else:
        sk = dict(front_split_plan(cluster_tiles, sm_count), fold=False, ws_slots=0)
        evict_first = bool(front_evict_first(m_tiles, i_local, sm_count))
    if block_n != FRONT_BLOCK_N and (sk["sk_max_seg"] != 1 or not evict_first):
        raise AssertionError(
            f"the {block_n}-wide front instance runs whole evict_first tiles only (M={M}, plan {sk})"
        )
    return dict(
        M=M,
        i_local=i_local,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        block_n=block_n,
        cluster_tiles=cluster_tiles,
        evict_first=evict_first,
        grid=sk["num_items"] * CTA_GROUP,
        **sk,
    )


def prefill_front_kernel_key(plan: dict[str, Any]) -> str:
    """The logical kernel key of a ``prefill_front_plan``."""
    return front_kernel_key(
        plan["i_local"], plan["evict_first"], plan["block_n"], plan["fold"]
    )


def norm_kernel_key(early_trigger: bool, packed: bool = False) -> str:
    """``tail_norm:e<0|1>[pk]``; ``pk``: the one-partial instance that keeps the row as packed bf16x2 words."""
    return f"tail_norm:e{1 if early_trigger else 0}{'pk' if packed else ''}"


# Ring depth of the prefill tail GEMM per row (Cake round-6 lever ledger, mirrored from
# ``kimi_k3_latent_moe_tail.tail_gemm_config_for``): 6 stages for the TP1 rows at
# T <= 256 (+1.0..+1.2 % in five paired samples on B200 and B300), 7 everywhere else.
TAIL_NUM_STAGES = 7
TAIL_RING6_TP = 1
TAIL_RING6_MAX_T = 256
TAIL_RING6_STAGES = 6
# Round-7 lever 7 (TP8 T=512) and round-4 R128 (TP8 T=1024 on 152 SMs): the 128-wide pair tile (56 column tiles) with
# a 9-deep ring; every other row keeps the 256-wide tile.  T=1024 on 152 SMs: 224 pair tiles in 3 waves of 76 clusters
# (3 x 19 x 256 clk) instead of 112 in 2 waves (2 x 19 x 512 clk); on 148 SMs 224 / 74 = 3.03 -> 4 waves, the MMA path
# ties, so 148-SM parts keep the 256-wide tile there.
TAIL_N128_TP = 8
TAIL_N128_T = 512
TAIL_N128_BLOCK_N = 128
TAIL_N128_STAGES = 9
TAIL_N128_TOKENS_BY_SM_COUNT = {152: (512, 1024)}


def tail_n128_tokens(sm_count: int = SM_COUNT) -> tuple[int, ...]:
    """TP8 token counts of the 128-wide pair tile on a ``sm_count``-SM device."""
    return TAIL_N128_TOKENS_BY_SM_COUNT.get(int(sm_count), (TAIL_N128_T,))


# Round-10 lever L1c (Cake ``tail_gemm_config_for`` ``FINAL_TS`` rule): every 256-wide tail GEMM instance stages the
# CTA's final item (its last stream-K segment) through one TMA store per epilogue warp out of the drained TMA ring
# instead of the direct pointer stores -- a separate kernel instance (key suffix ``t1``) with the same accumulator
# values and the same round-to-nearest bf16 conversion, so ``out`` is bit-identical; every earlier item of a CTA keeps
# the direct stores.  The instance takes the bf16 tensor map of ``out`` as one extra argument (``out_map``, bound by
# value from the same tensor as the ``out`` pointer).  The 128-wide TP8 instance keeps the direct stores (one
# 64-column pass per warp: the slab's fixed cost exceeds the exposed store it replaces).
TAIL_FINAL_TS = True


# Round-4 lever T1F: on 152-SM parts the TP1 rows of more than 128 and at most 512 tokens (the single-wave fused rows)
# run the 192-wide pair tile with an 8-deep ring: 38 column tiles (the last one 64 columns wide) give 76 pair tiles at
# two pair rows, one per cluster, where the 256-wide tile fills 56 of the 76 clusters; one pair row cuts its 38 tiles
# into two K halves (76 items).  The same K order and accumulation as the 256-wide tile, so ``out`` is bit-identical.
TAIL_N192_TP = 1
TAIL_N192_MAX_T = 512
TAIL_N192_SM_COUNT = 152
TAIL_N192_STAGES = 8
TAIL_N192_BLOCK_N = 192


# TP1 prefill rows from ``tail_wide_min_tokens(sm_count)`` tokens run the 256 x 512 pair tile (a 4-stage ring,
# single-buffered TMEM): a third fewer operand bytes per FLOP through L2 and shared memory, which the power-capped large
# rows turn into time; below it the 512-wide items quantise into too few waves (2048 tokens: 112 items over 76 clusters).
TAIL_WIDE_TP = 1
TAIL_WIDE_STAGES = 4
TAIL_WIDE_BLOCK_N = 512
# Wide tile from 4096 tokens on 152 SMs (speedup at 4096: 1.03 wide vs 1.01-1.02 256-wide), 8192 on 148 (1.03 vs 1.06).
TAIL_WIDE_MIN_T = 8192
TAIL_WIDE_MIN_T_BY_SM_COUNT = {152: 4096}


def tail_wide_min_tokens(sm_count: int = SM_COUNT) -> int:
    """First TP1 token count of the 256 x 512 pair tile on a ``sm_count``-SM device."""
    return TAIL_WIDE_MIN_T_BY_SM_COUNT.get(int(sm_count), TAIL_WIDE_MIN_T)


def tail_gemm_config(
    M: int, tp: int, sm_count: int = SM_COUNT
) -> tuple[int, int, bool]:
    """``(num_stages, block_n, final_ts)`` of the production prefill tail GEMM instance for ``M`` tokens at ``tp`` on a
    ``sm_count``-SM device."""
    if (
        int(tp) == TAIL_N192_TP
        and DECODE_MAX_T < int(M) <= TAIL_N192_MAX_T
        and int(sm_count) == TAIL_N192_SM_COUNT
    ):
        return TAIL_N192_STAGES, TAIL_N192_BLOCK_N, TAIL_FINAL_TS
    if int(tp) == TAIL_RING6_TP and int(M) <= TAIL_RING6_MAX_T:
        return TAIL_RING6_STAGES, BLOCK_N, TAIL_FINAL_TS
    if int(tp) == TAIL_WIDE_TP and int(M) >= tail_wide_min_tokens(sm_count):
        return TAIL_WIDE_STAGES, TAIL_WIDE_BLOCK_N, TAIL_FINAL_TS
    if int(tp) == TAIL_N128_TP and int(M) in tail_n128_tokens(sm_count):
        return TAIL_N128_STAGES, TAIL_N128_BLOCK_N, False
    return TAIL_NUM_STAGES, BLOCK_N, TAIL_FINAL_TS


def tail_gemm_num_stages(M: int, tp: int, sm_count: int = SM_COUNT) -> int:
    return tail_gemm_config(M, tp, sm_count)[0]


def tail_gemm_final_ts(M: int, tp: int, sm_count: int = SM_COUNT) -> bool:
    return tail_gemm_config(M, tp, sm_count)[2]


def tail_gemm_kernel_key(
    tp: int,
    weights_evict_first: bool,
    fused_norm: bool = False,
    num_stages: int = TAIL_NUM_STAGES,
    block_n: int = BLOCK_N,
    final_ts: bool = False,
    claim_items: bool = False,
) -> str:
    """``tail_gemm:tp<tp>e<evict>f<fused>[s<stages>][n<block_n>][t1][ci]``; the ring-depth, tile-width, final-item
    staged-store and claimed-item suffixes appear only for a non-default instance (``ci``: the resident grid of one CTA
    pair per resident cluster, which claims the items past its first from a device counter)."""
    key = f"tail_gemm:tp{int(tp)}e{1 if weights_evict_first else 0}f{1 if fused_norm else 0}"
    if int(num_stages) != TAIL_NUM_STAGES:
        key += f"s{int(num_stages)}"
    if int(block_n) != BLOCK_N:
        key += f"n{int(block_n)}"
    if final_ts:
        key += "t1"
    if claim_items:
        key += "ci"
    return key


def _sk_max_seg(sk_tiles: int, num_k: int, ipc: int) -> int:
    return max(
        ((t * num_k + num_k - 1) // ipc) - ((t * num_k) // ipc) + 1
        for t in range(sk_tiles)
    )


@functools.lru_cache(maxsize=None)
def split_plan(
    cluster_tiles: int, num_k: int, sm_count: int, reuse_rows: int = 1
) -> dict[str, int]:
    """Persistent plan: whole pair tiles for the full waves, then the remainder tiles cut into stream-K ranges when
    that saves at least ``SK_MIN_SAVING_ITERS``; a near-whole remainder wave with partial B-tile reuse is not cut."""
    resident = max(1, sm_count // CTA_GROUP)
    full = (cluster_tiles // resident) * resident
    rem = cluster_tiles - full
    waves = (cluster_tiles + resident - 1) // resident
    whole_cost = float(waves * num_k)
    best_cost = whole_cost
    plan = {
        "num_items": cluster_tiles,
        "full_items": cluster_tiles,
        "sk_ipc": num_k,
        "sk_max_seg": 1,
        "sk_total": 0,
        "sk_tiles": 0,
    }
    partial_reuse = 1 < reuse_rows < SK_MIN_REUSE_ROWS
    if (
        rem == 0
        or num_k < SK_MIN_NUM_K
        or (
            partial_reuse
            and (full == 0 or rem * 100 > SK_PARTIAL_REUSE_MAX_REM_PCT * resident)
        )
    ):
        return plan
    # The last full wave stays whole: in stream-K each range pays two fp32 partial round trips (1-3 % slower, 148 SMs).
    full_i = cluster_tiles - rem
    total = rem * num_k
    for g in range(min(resident, total // SK_MIN_ITERS), 0, -1):
        ipc = (total + g - 1) // g
        g2 = (total + ipc - 1) // ipc
        max_seg = _sk_max_seg(rem, num_k, ipc)
        if max_seg > MAX_SEG:
            continue
        cost = (full_i // resident) * num_k + ipc + SK_FIXUP_ITERS * (max_seg - 1)
        if cost < best_cost and cost <= whole_cost - SK_MIN_SAVING_ITERS:
            best_cost = cost
            plan = {
                "num_items": full_i + g2,
                "full_items": full_i,
                "sk_ipc": ipc,
                "sk_max_seg": max_seg,
                "sk_total": total,
                "sk_tiles": rem,
            }
    return plan


def norm_early_trigger(gemm_ctas: int, norm_ctas: int, sm_count: int) -> bool:
    """Fire the norm's PDL trigger at kernel start when the GEMM's early launch cannot serialise items."""
    return gemm_ctas > sm_count or gemm_ctas + norm_ctas <= sm_count


def weights_evict_first(gemm_ctas: int, sm_count: int) -> bool:
    """Single-wave GEMM grids load the weight boxes with the ``evict_first`` L2 policy (every CTA
    streams its weight columns once; multi-wave grids re-read weight blocks from L2)."""
    return gemm_ctas <= sm_count


FUSED_MIN_K2_ITERS = 48
# u32 words of the fused-norm counter buffer: two row-release semaphores, the claim ticket and the
# retirement count of the claimed instances (per-runner control words, ``_control_words``).
NORM_COUNTER_WORDS = 4


def fused_norm_fits(M: int, gemm_ctas: int, sm_count: int) -> bool:
    """Single-wave grids whose epilogue warps can own every latent row (one row per warp)."""
    return gemm_ctas <= sm_count and TAIL_EPI_WARPS * gemm_ctas >= M


def use_resident_fused_norm(
    M: int, gemm_ctas: int, sm_count: int, k2_iters: int
) -> bool:
    """Rows with the long shared-expert window (TP1) that the single-wave fused norm does not cover (a multi-wave grid,
    or more rows than its epilogue warps) fuse the norm on the resident grid: the launch holds one CTA pair per
    resident cluster, which claims the items past its first from a device counter, and its epilogue warps normalise
    every row (the first raster group's rows while the first item's shared-expert K blocks stream)."""
    return k2_iters >= FUSED_MIN_K2_ITERS and not fused_norm_fits(
        M, gemm_ctas, sm_count
    )


def resident_grid(num_items: int, sm_count: int) -> int:
    """CTAs of the claimed launch: one CTA pair per resident cluster, at most one per item.  Every CTA of the claimed
    instances allocates all 512 TMEM columns and more than half of the SM's shared memory, so one CTA runs per SM and
    ``sm_count // 2`` pairs are resident at once, as the cooperative launch requires."""
    return min(int(num_items), max(1, int(sm_count) // CTA_GROUP)) * CTA_GROUP


def use_fused_norm(M: int, gemm_ctas: int, sm_count: int, k2_iters: int) -> bool:
    """One launch (the GEMM's epilogue warps normalise the latent while the shared-expert K blocks
    stream) when the grid is single-wave and the shared-expert window of ``k2_iters`` 64-wide K
    blocks hides the normalisation: TP1 (96 blocks) fuses, TP8 (12 blocks) keeps the norm launch.
    The single-wave fused instances run one work item per cluster (no cluster launch control) and
    launch cooperatively: their CTAs wait for latent rows that other CTAs normalise."""
    return fused_norm_fits(M, gemm_ctas, sm_count) and k2_iters >= FUSED_MIN_K2_ITERS


# The TP8 prefill tails of one routed partial from 2048 tokens run the packed norm instance (``tail_norm:e<0|1>pk``):
# the row stays in registers as packed bf16x2 words (half the registers per thread, twice the rows in flight per SM),
# and the statistics and output repeat the summing instance's FP32 arithmetic, flushes and round points in the same
# element order (``y`` is byte-identical for every input).
PACKED_NORM_TP = 8
PACKED_NORM_MIN_T = 2048


def use_packed_norm(M: int, tp: int, num_partials: int) -> bool:
    return (
        int(tp) == PACKED_NORM_TP
        and int(num_partials) == 1
        and int(M) >= PACKED_NORM_MIN_T
    )


@_cached_plan
def prefill_tail_plan(
    M: int, tp: int, sm_count: int = SM_COUNT, num_partials: int = 1
) -> dict[str, Any]:
    """Host plan of the prefill tail chain (norm launch + persistent GEMM) for ``M`` tokens and ``num_partials``
    routed partials (the norm instance depends on it)."""
    k_up = k_up_for_tp(tp)
    i_local = i_local_for_tp(tp)
    norm_grid = (M + NORM_ROWS_PER_CTA - 1) // NORM_ROWS_PER_CTA
    m_tiles = m_tiles_for(M)
    num_stages, block_n, final_ts = tail_gemm_config(M, tp, sm_count)
    n_tiles = (HIDDEN + block_n - 1) // block_n
    cluster_tiles = (m_tiles // CTA_GROUP) * n_tiles
    num_k = (k_up + i_local) // BLOCK_K
    sk = split_plan(cluster_tiles, num_k, sm_count, min(GROUP_M, m_tiles) // CTA_GROUP)
    gemm_grid = sk["num_items"] * CTA_GROUP
    early = norm_early_trigger(gemm_grid, norm_grid, sm_count)
    single_fused = use_fused_norm(M, gemm_grid, sm_count, i_local // BLOCK_K)
    claim_items = use_resident_fused_norm(M, gemm_grid, sm_count, i_local // BLOCK_K)
    launch_grid = resident_grid(sk["num_items"], sm_count) if claim_items else gemm_grid
    return dict(
        M=M,
        tp=tp,
        k_up=k_up,
        i_local=i_local,
        norm_grid=norm_grid,
        m_tiles=m_tiles,
        cluster_tiles=cluster_tiles,
        num_k=num_k,
        gemm_grid=launch_grid,
        early_trigger=bool(early),
        # The claimed instances load the weights with the default L2 policy.
        weights_evict_first=bool(
            weights_evict_first(gemm_grid, sm_count) and not claim_items
        ),
        fused_norm=bool(single_fused or claim_items),
        packed_norm=bool(
            not (single_fused or claim_items) and use_packed_norm(M, tp, num_partials)
        ),
        claim_items=bool(claim_items),
        num_stages=num_stages,
        block_n=block_n,
        final_ts=bool(final_ts),
        n_tiles=n_tiles,
        **sk,
    )


# ---------------------------------------------------------------------------
# Route resolution (logical kernel keys of every stage)
# ---------------------------------------------------------------------------


def route_kernel_keys(
    stage: str,
    tp: int,
    num_tokens: int,
    sm_count: int = SM_COUNT,
    num_partials: int = 1,
    arch: str = "sm_100a",
) -> tuple[str, ...]:
    """Logical kernel keys launched by the production route of ``(stage, tp, num_tokens)`` on an
    ``arch`` device of ``sm_count`` SMs; ``num_partials`` is the tail's routed-partial count ``P``
    (the decode tail instance and the prefill tail's norm instance depend on it; the front does not).
    The front's fold form is planned for a device that holds its clusters (``prefill_front_plan``)."""
    if tp not in SUPPORTED_TP:
        raise ValueError(f"tp must be one of {SUPPORTED_TP}, got {tp}")
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if int(num_partials) < 1:
        raise ValueError("num_partials must be positive")
    i_local = i_local_for_tp(tp)
    if stage == "front":
        if num_tokens <= DECODE_MAX_T:
            return (
                decode_kernel_key(decode_front_plan(num_tokens, i_local, sm_count)),
            )
        return (
            prefill_front_kernel_key(
                prefill_front_plan(num_tokens, i_local, sm_count, arch)
            ),
        )
    if stage == "tail":
        if num_tokens <= DECODE_MAX_T:
            return (
                decode_kernel_key(
                    decode_tail_plan(
                        num_tokens, i_local, tp, int(num_partials), sm_count
                    )
                ),
            )
        plan = prefill_tail_plan(num_tokens, tp, sm_count, int(num_partials))
        gemm = tail_gemm_kernel_key(
            tp,
            plan["weights_evict_first"],
            plan["fused_norm"],
            plan["num_stages"],
            plan["block_n"],
            plan["final_ts"],
            plan["claim_items"],
        )
        if plan["fused_norm"]:
            return (gemm,)
        return (norm_kernel_key(plan["early_trigger"], plan["packed_norm"]), gemm)
    raise ValueError(f"stage must be 'front' or 'tail', got {stage!r}")


def required_kernel_keys(
    sm_count: int = SM_COUNT, arch: str = "sm_100a"
) -> tuple[str, ...]:
    """Every logical kernel the validated route set can select on an ``arch`` device of ``sm_count`` SMs (both
    stages, TP 1 / 8, all row tokens)."""
    keys: list[str] = []
    for stage in ("front", "tail"):
        for tp in SUPPORTED_TP:
            for tokens in ROW_TOKENS:
                for key in route_kernel_keys(stage, tp, tokens, sm_count, arch=arch):
                    if key not in keys:
                        keys.append(key)
    return tuple(keys)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _device_facts(index: int) -> tuple[tuple[int, int], int]:
    """``(compute capability, SM count)`` of ``cuda:index``, queried once per process."""
    properties = torch.cuda.get_device_properties(index)
    return (int(properties.major), int(properties.minor)), int(
        properties.multi_processor_count
    )


def _device_arch(device: torch.device) -> str:
    capability, _ = _device_facts(_device_index(device))
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the Kimi-K3 LatentMoE front/tail programs require compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def _device_sm_count(device: torch.device) -> tuple[int, int]:
    """``(device index, SM count)`` of a qualified device.

    The plans are derived from the device's own SM count; a count outside
    ``QUALIFIED_SM_COUNTS`` is refused rather than planned (the decode programs rely on
    every CTA of their grid being resident and the plan rules were validated on the
    qualified parts only).
    """
    index = _device_index(device)
    _, count = _device_facts(index)
    if count not in QUALIFIED_SM_COUNTS:
        raise NotImplementedError(
            f"the generated Kimi-K3 LatentMoE programs are qualified for devices with "
            f"{' / '.join(map(str, QUALIFIED_SM_COUNTS))} SMs; device {index} has {count}"
        )
    return index, count


def generated_program_available(
    device: torch.device,
    stage: Optional[str] = None,
    tp: Optional[int] = None,
    num_tokens: Optional[int] = None,
    num_partials: int = 1,
) -> bool:
    """True when this checkout registers the programs for ``device`` (optionally: one exact route).

    The route is resolved with the device's architecture and SM count and, for the tail, the caller's
    routed-partial count ``num_partials`` (the decode tail instance and the TP8 prefill norm
    instance depend on it), so the answer is exactly what ``prepare_kimi_k3_latent_moe_*``
    would require.
    """
    capability, count = _device_facts(_device_index(device))
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None or not MODULES or count not in QUALIFIED_SM_COUNTS:
        return False
    if stage is None and tp is None and num_tokens is None:
        return cake_jit.route_available(arch, required_kernel_keys(count, arch))
    if stage is None or tp is None or num_tokens is None:
        raise ValueError("pass stage, tp and num_tokens together or none of them")
    try:
        keys = route_kernel_keys(
            stage, int(tp), int(num_tokens), count, int(num_partials), arch=arch
        )
    except ValueError:
        return False
    return cake_jit.route_available(arch, keys)


# ---------------------------------------------------------------------------
# Device scratch: per-runner control words, per-stream partial workspace
# ---------------------------------------------------------------------------
#
# The programs use two kinds of mutable device scratch, owned so that distinct runners never share any:
#
# * Control words: the self-resetting arrival words -- the decode grid counter (``counters``, ``atom.inc``
#   with limit grid - 1), the prefill stream-K arrival counters (one per split tile and CTA rank; the last
#   contributor zeroes its counter) and the fused-norm words (``norm_counter``: two row-release semaphores,
#   the claimed instances' item ticket and their retirement count, which the last CTA to retire resets).
#   They must read zero before a launch and every launch returns them to zero, so they are zeroed once.
#   Each prepared runner takes its own words from the device's ``_CtrlSlab``; a slot is never returned or
#   reused, so a CUDA Graph that captured a runner replays into valid private words after the runner is
#   dropped, and a third-party PDL kernel between two runners cannot make them share a word.
# * Partials: the FP32 stream-K workspace of split plans, written before it is read inside one launch.
#   An eager launch uses the workspace of its (device, stream) (``_WS_LANES``), grown to the largest split
#   plan launched there: launches on one stream are stream-ordered, so they reuse it as PyTorch's caching
#   allocator reuses a block on its stream, and distinct streams never share one.  A launch under CUDA Graph
#   capture allocates its block from the graph's memory pool instead (once, at capture).  A program that binds
#   a partial workspace must not trigger ``griddepcontrol.launch_dependents`` (a dependent launched early could
#   write the block while the program still uses it); the package test checks it.  Whole plans bind
#   ``_READ_NEVER``.

#: Control words are handed out in 128-byte slots of 32 four-byte words.
CTRL_SLOT_WORDS = 32
#: Four-byte words of one slab chunk (256 KiB): one zero fill and one device synchronisation for hundreds of runners.
CTRL_CHUNK_WORDS = 1 << 16
#: u32 words of the decode programs' grid counter buffer (only ``[0]`` is used).
DECODE_COUNTER_WORDS = 4


class _CtrlSlab:
    """Zero-initialised control words of one device, handed out in 128-byte slots and never reused.

    A chunk is allocated outside CUDA Graph capture, zeroed, and made visible to every stream by one
    device synchronisation at creation; every chunk stays referenced for the life of the process, so a slot
    stays valid (and private to its owner) for every launch and every graph that bound it.  ``take`` under
    capture with no free slot left zeroes new words inside the capture instead: one zero-fill kernel node in that
    graph, and the words stay pinned for good as well.
    """

    def __init__(self, index: int) -> None:
        self.index = index
        self._lock = threading.Lock()
        self._chunk: Optional[torch.Tensor] = None
        self._used = 0
        self._pinned: list[torch.Tensor] = []

    def take(self, words: int) -> torch.Tensor:
        """``words`` (a multiple of ``CTRL_SLOT_WORDS``) zero int32 words owned by the caller for good."""
        with self._lock:
            chunk = self._chunk
            if chunk is not None and self._used + words <= chunk.numel():
                slot = chunk[self._used : self._used + words]
                self._used += words
                return slot
            device = torch.device("cuda", self.index)
            if torch.cuda.is_current_stream_capturing():
                slot = torch.zeros(words, dtype=torch.int32, device=device)
                self._pinned.append(slot)
                return slot
            chunk = torch.zeros(
                max(CTRL_CHUNK_WORDS, words), dtype=torch.int32, device=device
            )
            torch.cuda.synchronize(self.index)
            self._pinned.append(chunk)
            self._chunk, self._used = chunk, words
            return chunk[:words]


_CTRL_LOCK = threading.Lock()
_CTRL_SLABS: dict[int, _CtrlSlab] = {}
# One-shot functional calls: the (int32, uint32) views of one slot per (device, stream handle, int32 words,
# uint32 words).
_CTRL_LANES: dict[
    tuple[int, int, int, int], tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
] = {}
_READ_NEVER: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
# Eager split-plan launches: ``(elements, FP32 workspace)`` per (device, stream handle).
_WS_LANES: dict[tuple[int, int], tuple[int, torch.Tensor]] = {}
_FFI_DEVICES: dict[int, Any] = {}

# Launch-path queries without building Python stream objects.
# The raw handle of the current stream of device ``index``.
_RAW_STREAM: Callable[[int], int] = getattr(
    torch._C,
    "_cuda_getCurrentRawStream",
    lambda index: torch.cuda.current_stream(index).cuda_stream,
)
# Whether the current stream of the current device is capturing a CUDA Graph.
_CAPTURING: Callable[[], bool] = getattr(
    torch._C, "_cuda_isCurrentStreamCapturing", torch.cuda.is_current_stream_capturing
)
_CURRENT_DEVICE: Callable[[], int] = getattr(
    torch._C, "_cuda_getDevice", torch.cuda.current_device
)
_use_raw_stream = tvm_ffi.use_raw_stream


def _ffi_device(index: int) -> Any:
    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    return device


def _ctrl_layout(n_i32: int, n_u32: int) -> tuple[int, int]:
    """``(uint32 offset, slot words)`` of a control-word slot: ``n_i32`` int32 words first, then ``n_u32``
    uint32 words from the next 128-byte boundary, rounded up to whole 128-byte slots."""
    u32_off = -(-int(n_i32) // CTRL_SLOT_WORDS) * CTRL_SLOT_WORDS
    words = -(-(u32_off + int(n_u32)) // CTRL_SLOT_WORDS) * CTRL_SLOT_WORDS
    return u32_off, max(words, CTRL_SLOT_WORDS)


def _ctrl_slab(index: int) -> _CtrlSlab:
    slab = _CTRL_SLABS.get(index)
    if slab is None:
        with _CTRL_LOCK:
            slab = _CTRL_SLABS.setdefault(index, _CtrlSlab(index))
    return slab


def _control_words(
    index: int, n_i32: int, n_u32: int, *, lane: bool = False
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Zero control words on ``cuda:index`` (the current device): an int32 view of ``n_i32`` words and a uint32
    view of ``n_u32`` words (``None`` for a zero count), in one slot.

    ``lane=False`` (``prepare_*``): a fresh slot owned by the runner.  ``lane=True`` (the one-shot functional
    entry points): eager calls share one slot per (device, current stream, word counts) -- calls on one stream
    are stream-ordered and every launch returns the words to zero -- while a call under CUDA Graph capture
    takes a fresh slot, so concurrently replayed graphs never share words.  A lane hit takes no lock.
    """
    if lane and not _CAPTURING():
        views = _CTRL_LANES.get((index, _RAW_STREAM(index), n_i32, n_u32))
        if views is not None:
            return views
        return _new_lane(index, n_i32, n_u32)
    return _slot_views(_ctrl_slab(index), n_i32, n_u32)


def _new_lane(
    index: int, n_i32: int, n_u32: int
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    slab = _ctrl_slab(index)  # outside ``_CTRL_LOCK``, which creating the slab takes
    key = (index, _RAW_STREAM(index), n_i32, n_u32)
    with _CTRL_LOCK:
        views = _CTRL_LANES.get(key)
        if views is None:
            views = _CTRL_LANES[key] = _slot_views(slab, n_i32, n_u32)
    return views


def _slot_views(
    slab: _CtrlSlab, n_i32: int, n_u32: int
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The int32 and uint32 views of a fresh control-word slot of ``slab``."""
    u32_off, words = _ctrl_layout(n_i32, n_u32)
    slot = slab.take(words)
    i32 = slot[: int(n_i32)] if n_i32 else None
    u32 = slot[u32_off : u32_off + int(n_u32)].view(torch.uint32) if n_u32 else None
    return i32, u32


def _read_never(index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(fp32 [1024], uint64 timeline)`` of ``cuda:index``, bound where no registered program reads or
    writes: the decode tail's ``out_r``, every decode ``tl`` (written only by diagnostic instances, which are
    never registered) and the ``ws`` of whole prefill plans (only multi-segment items touch it; a split plan's
    partial workspace replaces it at launch).  Never initialised and never written."""
    entry = _READ_NEVER.get(index)
    if entry is None:
        dev = torch.device("cuda", index)
        with _CTRL_LOCK:
            entry = _READ_NEVER.get(index)
            if entry is None:
                entry = (
                    torch.empty(1024, dtype=torch.float32, device=dev),
                    torch.empty(TL_MAX_GRID * TL_SLOTS, dtype=torch.uint64, device=dev),
                )
                _READ_NEVER[index] = entry
    return entry


def _partial_ws_numel(stage: str, plan: dict[str, Any]) -> int:
    """FP32 elements of the stream-K partial workspace one launch uses (0: a whole plan, which writes none).

    One ``128 x block_n`` slot per contributor CTA: ``sk_tiles`` split tiles of ``sk_max_seg`` segments of
    ``CTA_GROUP`` CTAs (the front's slot is 256 wide, the tail's follows the plan's tile).  A plan that
    carries partials between its items states its slot count in ``ws_slots`` (the front fold form: two
    ``128 x 256`` slots per CTA).
    """
    ws_slots = int(plan.get("ws_slots", 0))
    if ws_slots:
        return ws_slots * BLOCK_M * FRONT_BLOCK_N
    sk_tiles = int(plan.get("sk_tiles", 0))
    if sk_tiles == 0:
        return 0
    block_n = FRONT_BLOCK_N if stage == "front" else int(plan["block_n"])
    return sk_tiles * int(plan["sk_max_seg"]) * CTA_GROUP * BLOCK_M * block_n


def _arrival_counter_words(plan: dict[str, Any]) -> int:
    """int32 stream-K arrival counters of a prefill plan: one per split tile and CTA rank, at least two (the
    front fold form: one per workspace slot, ``ws_slots``; the reader of a slot resets its counter)."""
    ws_slots = int(plan.get("ws_slots", 0))
    if ws_slots:
        return ws_slots
    return max(2, int(plan.get("sk_tiles", 0)) * CTA_GROUP)


_RESIDENT_CLUSTERS: dict[tuple[int, int, int], int] = {}
# An empty kernel: the co-resident cluster count of a one-CTA-per-SM launch depends on the launch shape only.
_OCCUPANCY_PROBE = "cake_kimi_k3_latent_moe_occupancy_probe"
_OCCUPANCY_PROBE_PTX = (
    ".version 8.0\n.target sm_90\n.address_size 64\n"
    f".visible .entry {_OCCUPANCY_PROBE}()\n{{\n\tret;\n}}\n"
)


def _resident_clusters(device_index: int, block_threads: int, smem_bytes: int) -> int:
    """Co-resident two-CTA clusters of ``cuda:device_index`` for CTAs of ``block_threads`` threads and
    ``smem_bytes`` dynamic shared memory (``cuOccupancyMaxActiveClusters``, queried once per device and
    shape on an empty probe kernel launched with that shape).  The front programs take more than half
    of an SM's shared memory, so they run one CTA per SM and the count follows the part's GPC topology,
    not ``SM count / 2``."""
    key = (int(device_index), int(block_threads), int(smem_bytes))
    count = _RESIDENT_CLUSTERS.get(key)
    if count is not None:
        return count
    from cuda.bindings import driver

    def checked(result):
        status = result[0]
        if status != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cluster occupancy query: {status!r}")
        return result[1] if len(result) == 2 else None

    ctx = checked(driver.cuDevicePrimaryCtxRetain(int(device_index)))
    try:
        checked(driver.cuCtxSetCurrent(ctx))
        module = checked(driver.cuModuleLoadData(_OCCUPANCY_PROBE_PTX.encode() + b"\0"))
        try:
            func = checked(
                driver.cuModuleGetFunction(module, _OCCUPANCY_PROBE.encode())
            )
            checked(
                driver.cuFuncSetAttribute(
                    func,
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    int(smem_bytes),
                )
            )
            attr = driver.CUlaunchAttribute()
            attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attr.value.clusterDim.x = CTA_GROUP
            attr.value.clusterDim.y = 1
            attr.value.clusterDim.z = 1
            config = driver.CUlaunchConfig()
            config.gridDimX = CTA_GROUP
            config.gridDimY = 1
            config.gridDimZ = 1
            config.blockDimX = int(block_threads)
            config.blockDimY = 1
            config.blockDimZ = 1
            config.sharedMemBytes = int(smem_bytes)
            config.attrs = [attr]
            config.numAttrs = 1
            count = int(checked(driver.cuOccupancyMaxActiveClusters(func, config)))
        finally:
            checked(driver.cuModuleUnload(module))
    finally:
        checked(driver.cuDevicePrimaryCtxRelease(int(device_index)))
    _RESIDENT_CLUSTERS[key] = count
    return count


def _front_resident_clusters(device_index: int, arch: str, i_local: int) -> int:
    """Co-resident clusters of the fold-form front program's launch on ``cuda:device_index`` (the current
    device).  The query loads a module, which a CUDA Graph capture must not see: under capture, before the
    device's first query, it answers 0, so that prepare keeps the trailing-wave plan (the same outputs,
    bitwise)."""
    key = front_kernel_key(i_local, True, fold=True)
    launch = MODULES[kernel_module_name(arch, key)]["launch"]
    shape = (
        int(device_index),
        int(launch["block"][0]),
        int(launch["dynamic_smem_bytes"]),
    )
    if shape not in _RESIDENT_CLUSTERS and _CAPTURING():
        return 0
    return _resident_clusters(*shape)


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


# Per-program facts of the launch binding: the argument position of the ``ws`` buffer, and the programs that
# launch cooperatively.
_WS_POSITION: dict[str, int] = {
    name: position
    for name, record in MODULES.items()
    for position, (kind, arg) in enumerate(record["arg_plan"])
    if kind == "buffer" and arg == "ws"
}
_COOPERATIVE: frozenset[str] = frozenset(
    name
    for name, record in MODULES.items()
    if record.get("launch", {}).get("cooperative")
)


def _bind(
    arch: str, key: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple, int, int]:
    """Order ``kwargs`` by the argument plan of the program serving ``key`` and load its entry; also
    return the position of its ``ws`` buffer argument (-1 when it has none) and the CTA count of its
    cooperative launch (0 when the program does not launch cooperatively)."""
    module_name = kernel_module_name(arch, key)
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    cooperative_ctas = (
        grid["grid_x"] * grid["grid_y"] * grid["grid_z"]
        if module_name in _COOPERATIVE
        else 0
    )
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} ({kind}); "
                f"host binding provides {sorted(kwargs)}"
            )
    module = load_cake_kimi_k3_latent_moe_module(arch, key)
    return (
        getattr(module, record["ffi_entry"]),
        tuple(arguments),
        _WS_POSITION.get(module_name, -1),
        cooperative_ctas,
    )


# ``cudaGetErrorString(cudaErrorCooperativeLaunchTooLarge)``, which the bindings report when a cooperative grid
# does not fit the SMs the launch stream offers (a smaller green context, for one).
_COOPERATIVE_TOO_LARGE = "too many blocks in cooperative launch"


def _clear_cuda_last_error() -> None:
    """Reset the CUDA runtime's last error after a refused launch.  The refusal is not sticky, but the runtime
    keeps it as the calling thread's last error, which the next PyTorch launch check would raise once.  The
    bindings share the process's ``libcudart`` with PyTorch (cuda-python's runtime module may load another
    copy), so this calls ``cudaGetLastError`` in every ``libcudart`` the process has loaded."""
    paths = set()
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split(None, 5)
                if len(fields) == 6 and "/libcudart.so" in fields[5]:
                    paths.add(fields[5].strip())
    except OSError:
        return
    for path in sorted(paths):
        try:  # already loaded: the same instance, never a new copy
            lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError:
            continue
        lib.cudaGetLastError()


def _grow_ws_lane(index: int, stream: int, numel: int) -> tuple[int, torch.Tensor]:
    """The workspace of (``cuda:index``, ``stream``), (re)allocated for at least ``numel`` elements.

    ``stream`` is the current stream of ``cuda:index``, so the caching allocator ties the block to it; a
    replaced (smaller) block returns to that stream's cache, which reuses it only for work queued after the
    launches that used it."""
    with _CTRL_LOCK:
        lane = _WS_LANES.get((index, stream))
        if lane is None or lane[0] < numel:
            workspace = torch.empty(
                numel, dtype=torch.float32, device=torch.device("cuda", index)
            )
            lane = _WS_LANES[(index, stream)] = (numel, workspace)
    return lane


class _Launch:
    """One bound program launch of a runner."""

    __slots__ = (
        "stage",
        "key",
        "module",
        "kwargs",
        "entry",
        "arguments",
        "device_index",
        "ws_numel",
        "ws_head",
        "ws_tail",
        "cooperative_ctas",
    )

    def __init__(
        self,
        stage: str,
        key: str,
        module: str,
        kwargs: dict[str, Any],
        entry: Callable[..., Any],
        arguments: tuple,
        device_index: int,
        ws_slot: int = -1,
        ws_numel: int = 0,
        cooperative_ctas: int = 0,
    ) -> None:
        self.stage = stage
        self.key = key
        self.module = module
        self.kwargs = kwargs
        self.entry = entry
        self.arguments = arguments
        self.device_index = device_index
        # Split prefill plans: FP32 elements of the partial workspace that replaces the ``ws`` argument
        # (``kwargs["ws"]`` is then the never-read placeholder) between ``ws_head`` and ``ws_tail``.
        self.ws_numel = ws_numel
        self.ws_head = arguments[:ws_slot] if ws_numel else ()
        self.ws_tail = arguments[ws_slot + 1 :] if ws_numel else ()
        # CTAs of a cooperative launch (0: not cooperative); all of them must be resident at once.
        self.cooperative_ctas = cooperative_ctas

    def __repr__(self) -> str:
        return (
            f"_Launch(stage={self.stage!r}, key={self.key!r}, module={self.module!r})"
        )

    def __call__(self, stream: int) -> None:
        """Launch on ``stream``, the current stream of the launch's device (set as the FFI stream)."""
        try:
            if self.ws_numel:
                self.entry(*self.ws_head, self.partial_ws(stream), *self.ws_tail)
            else:
                self.entry(*self.arguments)
        except Exception as exc:
            if self.cooperative_ctas and _COOPERATIVE_TOO_LARGE in str(exc):
                self._refused()
            raise

    def partial_ws(self, stream: int) -> torch.Tensor:
        """The partial workspace of a launch on ``stream``: the stream's workspace when eager, a block of the
        graph's memory pool under capture (written before it is read inside the launch, so never initialised)."""
        index = self.device_index
        if _CURRENT_DEVICE() == index:
            capturing = _CAPTURING()
        else:
            with torch.cuda.device(index):
                capturing = _CAPTURING()
        if capturing:
            return torch.empty(
                self.ws_numel, dtype=torch.float32, device=torch.device("cuda", index)
            )
        lane = _WS_LANES.get((index, stream))
        if lane is None or lane[0] < self.ws_numel:
            lane = _grow_ws_lane(index, stream, self.ws_numel)
        return lane[1]

    def _refused(self) -> None:
        # Nothing ran.  Clear the runtime's last error, so the caller's next CUDA work does not fail on it.
        _clear_cuda_last_error()
        raise RuntimeError(
            f"{self.key} ({self.stage}): the cooperative launch of {self.cooperative_ctas} CTAs needs at "
            f"least {self.min_sms()} SMs resident at once, more than the launch stream offers (a smaller "
            "green context or a partitioned device); launch it on a stream with that many SMs "
            f"(CUDA: {_COOPERATIVE_TOO_LARGE})"
        ) from None

    def min_sms(self) -> int:
        """SMs a cooperative launch needs: its CTAs, as many per SM as the program's shared memory allows
        (one for every registered cooperative program)."""
        if not self.cooperative_ctas:
            return 0
        smem = int(MODULES[self.module]["launch"]["dynamic_smem_bytes"])
        per_sm_smem = getattr(
            torch.cuda.get_device_properties(self.device_index),
            "shared_memory_per_multiprocessor",
            0,
        )
        # 1 KiB of each SM's shared memory is reserved per resident CTA.
        per_sm = max(1, int(per_sm_smem) // (smem + 1024))
        return -(-self.cooperative_ctas // per_sm)


def _make_launch(
    stage: str,
    arch: str,
    key: str,
    kwargs: dict[str, Any],
    *,
    ws_numel: int = 0,
    device_index: int = 0,
) -> _Launch:
    entry, arguments, ws_slot, cooperative_ctas = _bind(arch, key, kwargs)
    if ws_numel and ws_slot < 0:
        raise RuntimeError(
            f"{key!r} has no 'ws' argument for its {ws_numel}-element partial workspace"
        )
    return _Launch(
        stage,
        key,
        kernel_module_name(arch, key),
        kwargs,
        entry,
        arguments,
        device_index,
        ws_slot,
        ws_numel,
        cooperative_ctas,
    )


@dataclass(frozen=True)
class KimiK3LatentMoeRunner:
    """The prepared launch sequence of one front or tail call.

    ``launch()`` submits the bound programs in order on the current torch
    stream of the runner's device with no host synchronisation; the kernels
    read every operand on device at launch, so the runner (or a CUDA Graph
    capturing it) replays for new values written into the same buffers.
    Prepare a new runner when a shape or a tensor binding changes.

    The runner owns its control words (never freed, so a graph that captured
    it stays valid after the runner is dropped).  An eager launch of a split
    prefill plan uses the FP32 partial workspace of its (device, stream),
    shared in stream order with the other launches on that stream; a launch
    under capture takes a block of the graph's memory pool.  Launches on
    distinct streams or graphs never mix control words or partials; a runner
    must not run concurrently with itself, and graphs that share one memory
    pool must not replay concurrently.
    """

    stage: str
    tp: int
    rank: int
    num_tokens: int
    arch: str
    sm_count: int  # SM count of the device the plan was derived for
    route: str  # "decode" or "prefill"
    plan: dict[str, Any] = field(repr=False)
    launches: tuple[_Launch, ...] = field(repr=False)
    outputs: tuple[torch.Tensor, ...] = field(repr=False)
    device_index: int = field(repr=False)

    @property
    def kernel_keys(self) -> tuple[str, ...]:
        return tuple(launch.key for launch in self.launches)

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(launch.module for launch in self.launches)

    @property
    def launch_count(self) -> int:
        return len(self.launches)

    def launch(self) -> tuple[torch.Tensor, ...]:
        # The current stream of the runner's device (whichever device is current) becomes the FFI stream the
        # bindings launch on; each binding enters the device of its operands.
        index = self.device_index
        stream = _RAW_STREAM(index)
        device = _FFI_DEVICES.get(index)
        if device is None:
            device = _ffi_device(index)
        with _use_raw_stream(device, stream):
            for launch in self.launches:
                launch(stream)
        return self.outputs

    __call__ = launch


def _check(
    t: torch.Tensor,
    shape: tuple[int, ...],
    name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}, got {tuple(t.shape)} {t.dtype}"
        )


def _same_device(tensors: dict[str, torch.Tensor]) -> torch.device:
    devices = {t.device for t in tensors.values()}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError(
            f"every operand must live on one CUDA device, got {sorted(map(str, devices))}"
        )
    return next(iter(devices))


# ---------------------------------------------------------------------------
# Front
# ---------------------------------------------------------------------------


def prepare_kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> KimiK3LatentMoeRunner:
    """Validate the front operands, plan the route and bind its launch(es).

    ``shared_gate_up_weight`` is the rank-local ``[2 * I_local, 7168]`` concatenation of the
    shared experts' gate rows followed by their up rows (``I_local = 6144 / TP``).  The JIT
    module(s) are built and loaded and the runner's control words are taken here: prepare
    before CUDA Graph capture, once per layer and shape (the words are never freed).
    """
    return _prepare_front(
        x,
        gate_weight,
        down_weight,
        shared_gate_up_weight,
        logits,
        latent,
        shared_act,
        lane=False,
    )


def _prepare_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
    *,
    lane: bool,
) -> KimiK3LatentMoeRunner:
    T = int(x.shape[0]) if x.dim() == 2 else -1
    _check(x, (T, HIDDEN), "x")
    i_local = (
        int(shared_gate_up_weight.shape[0]) // 2
        if shared_gate_up_weight.dim() == 2
        else 0
    )
    if i_local not in (SHARED_INTERMEDIATE // tp for tp in SUPPORTED_TP):
        raise ValueError(
            f"shared_gate_up_weight must have 2 * (6144 / TP) rows for TP in {SUPPORTED_TP}, "
            f"got {tuple(shared_gate_up_weight.shape)}"
        )
    tp = SHARED_INTERMEDIATE // i_local
    _check(gate_weight, (NUM_EXPERTS, HIDDEN), "gate_weight")
    _check(down_weight, (LATENT, HIDDEN), "down_weight")
    _check(shared_gate_up_weight, (2 * i_local, HIDDEN), "shared_gate_up_weight")
    _check(logits, (T, NUM_EXPERTS), "logits", torch.float32)
    _check(latent, (T, LATENT), "latent")
    _check(shared_act, (T, i_local), "shared_act")
    device = _same_device(
        dict(
            x=x,
            gate_weight=gate_weight,
            down_weight=down_weight,
            shared_gate_up_weight=shared_gate_up_weight,
            logits=logits,
            latent=latent,
            shared_act=shared_act,
        )
    )
    arch = _device_arch(device)
    index, sm_count = _device_sm_count(device)
    launches: tuple[_Launch, ...]
    plan: dict[str, Any]
    with torch.cuda.device(index):
        if T <= DECODE_MAX_T:
            plan = decode_front_plan(T, i_local, sm_count)
            key = decode_kernel_key(plan)
            _, counters = _control_words(index, 0, DECODE_COUNTER_WORDS, lane=lane)
            _, tl = _read_never(index)
            kwargs = dict(
                A_R=gate_weight,
                A_L=down_weight,
                A_S=shared_gate_up_weight,
                A_2=down_weight,
                B_1=x,
                B_2=x,
                out_r=logits,
                out_l=latent,
                out_s=shared_act,
                counters=counters,
                routed=latent,
                norm_w=latent,
                y_out=latent,
                tl=tl,
                num_tokens=T,
                k1_off=0,
                num_partials=0,
                eps=0.0,
                grid=(int(plan["grid"]), 1, 1),
            )
            assert tuple(kwargs) == DECODE_KWARGS
            launches = (
                _make_launch("front_decode", arch, key, kwargs, device_index=index),
            )
            route = "decode"
        else:
            resident = None
            if front_fold_plan(m_tiles_for(T), i_local, sm_count, arch) is not None:
                resident = _front_resident_clusters(index, arch, i_local)
            plan = prefill_front_plan(T, i_local, sm_count, arch, resident)
            key = prefill_front_kernel_key(plan)
            counters, _ = _control_words(
                index, _arrival_counter_words(plan), 0, lane=lane
            )
            # A split plan's partial workspace replaces it at launch.
            ws, _ = _read_never(index)
            kwargs = dict(
                A=x,
                WG=gate_weight,
                WD=down_weight,
                WS=shared_gate_up_weight,
                logits=logits,
                latent=latent,
                shared_act=shared_act,
                ws=ws,
                counters=counters,
                M=T,
                m_tiles=int(plan["m_tiles"]),
                num_items=int(plan["num_items"]),
                full_items=int(plan["full_items"]),
                sk_ipc=int(plan["sk_ipc"]),
                sk_max_seg=int(plan["sk_max_seg"]),
                sk_total=int(plan["sk_total"]),
                grid=(int(plan["grid"]), 1, 1),
            )
            assert tuple(kwargs) == FRONT_KWARGS
            launches = (
                _make_launch(
                    "front_gemm",
                    arch,
                    key,
                    kwargs,
                    ws_numel=_partial_ws_numel("front", plan),
                    device_index=index,
                ),
            )
            route = "prefill"
    return KimiK3LatentMoeRunner(
        "front",
        tp,
        0,
        T,
        arch,
        sm_count,
        route,
        plan,
        launches,
        (logits, latent, shared_act),
        index,
    )


def kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete front operator for one rank into the caller-owned ``logits`` / ``latent`` / ``shared_act``.

    Eager calls share one set of control words per device, stream and size and the stream's partial workspace
    (calls on one stream are stream-ordered); a call under CUDA Graph capture takes fresh words, which are never
    freed, and a block of the graph's memory pool.
    """
    return _prepare_front(
        x,
        gate_weight,
        down_weight,
        shared_gate_up_weight,
        logits,
        latent,
        shared_act,
        lane=True,
    )()


# ---------------------------------------------------------------------------
# Tail
# ---------------------------------------------------------------------------


def prepare_kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
) -> KimiK3LatentMoeRunner:
    """Validate the tail operands, plan the route and bind its launch(es).

    ``routed`` is ``[P, T, 3584]`` BF16 (``P`` un-reduced routed-expert partials, summed on
    device); ``y_workspace`` is a caller-owned BF16 ``[T, 3584]`` buffer that receives the
    normalised latent (the GEMM's B operand).  ``up_weight`` is the full ``[7168, 3584]``
    replicated up-projection; this rank multiplies its ``3584 / tp`` column slice.
    ``shared_down_weight`` is the rank-local ``[7168, 6144 / tp]`` shard.  Prepare before
    CUDA Graph capture, once per layer and shape (the runner's control words are never freed).
    """
    return _prepare_tail(
        routed,
        norm_weight,
        up_weight,
        shared_act,
        shared_down_weight,
        out,
        tp=tp,
        rank=rank,
        y_workspace=y_workspace,
        lane=False,
    )


def _prepare_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
    lane: bool,
) -> KimiK3LatentMoeRunner:
    tp, rank = int(tp), int(rank)
    if tp not in SUPPORTED_TP:
        raise ValueError(f"tp must be one of {SUPPORTED_TP}, got {tp}")
    if not 0 <= rank < tp:
        raise ValueError(f"rank {rank} out of range for tp={tp}")
    if not isinstance(routed, torch.Tensor) or routed.dim() != 3:
        raise ValueError("routed must be a contiguous bf16 [P, T, 3584] tensor")
    P, T = int(routed.shape[0]), int(routed.shape[1])
    _check(routed, (P, T, LATENT), "routed")
    i_local = i_local_for_tp(tp)
    _check(norm_weight, (LATENT,), "norm_weight")
    _check(up_weight, (HIDDEN, LATENT), "up_weight")
    _check(shared_act, (T, i_local), "shared_act")
    _check(shared_down_weight, (HIDDEN, i_local), "shared_down_weight")
    _check(out, (T, HIDDEN), "out")
    _check(y_workspace, (T, LATENT), "y_workspace")
    device = _same_device(
        dict(
            routed=routed,
            norm_weight=norm_weight,
            up_weight=up_weight,
            shared_act=shared_act,
            shared_down_weight=shared_down_weight,
            out=out,
            y_workspace=y_workspace,
        )
    )
    arch = _device_arch(device)
    index, sm_count = _device_sm_count(device)
    launches: tuple[_Launch, ...]
    plan: dict[str, Any]
    with torch.cuda.device(index):
        if T <= DECODE_MAX_T:
            plan = decode_tail_plan(T, i_local, tp, P, sm_count)
            key = decode_kernel_key(plan)
            _, counters = _control_words(index, 0, DECODE_COUNTER_WORDS, lane=lane)
            f32_dummy, tl = _read_never(index)
            kwargs = dict(
                A_R=up_weight,
                A_L=up_weight,
                A_S=up_weight,
                A_2=shared_down_weight,
                B_1=y_workspace,
                B_2=shared_act,
                out_r=f32_dummy,
                out_l=out,
                out_s=out,
                counters=counters,
                routed=routed,
                norm_w=norm_weight,
                y_out=y_workspace,
                tl=tl,
                num_tokens=T,
                k1_off=rank * int(plan["k1"]),
                num_partials=P,
                eps=float(RMS_EPS),
                grid=(int(plan["grid"]), 1, 1),
            )
            assert tuple(kwargs) == DECODE_KWARGS
            launches = (
                _make_launch(
                    "tail_decode_fused", arch, key, kwargs, device_index=index
                ),
            )
            route = "decode"
        else:
            plan = prefill_tail_plan(T, tp, sm_count, P)
            counters, norm_counter = _control_words(
                index, _arrival_counter_words(plan), NORM_COUNTER_WORDS, lane=lane
            )
            # A split plan's partial workspace replaces it at launch.
            ws, _ = _read_never(index)
            gemm_key = tail_gemm_kernel_key(
                tp,
                plan["weights_evict_first"],
                plan["fused_norm"],
                plan["num_stages"],
                plan["block_n"],
                plan["final_ts"],
                plan["claim_items"],
            )
            # ``out_map``: the bf16 tensor map of ``out`` for the final-item staged TMA store (``t1`` instances bind it
            # by value from the same tensor as the ``out`` pointer; the argument plan of the 128-wide instance does not
            # list it and ``_bind`` passes only what the generated program declares).
            gemm_kwargs = dict(
                A1=y_workspace,
                B1=up_weight,
                A2=shared_act,
                B2=shared_down_weight,
                out=out,
                out_map=out,
                ws=ws,
                counters=counters,
                M=T,
                m_tiles=int(plan["m_tiles"]),
                k0_blocks=rank * plan["k_up"] // BLOCK_K,
                num_items=int(plan["num_items"]),
                full_items=int(plan["full_items"]),
                sk_ipc=int(plan["sk_ipc"]),
                sk_max_seg=int(plan["sk_max_seg"]),
                sk_total=int(plan["sk_total"]),
                routed=routed,
                norm_weight=norm_weight,
                y_out=y_workspace,
                norm_counter=norm_counter,
                num_partials=P,
                eps=float(RMS_EPS),
                grid=(int(plan["gemm_grid"]), 1, 1),
            )
            assert tuple(gemm_kwargs) == TAIL_GEMM_KWARGS
            gemm_launch = _make_launch(
                "tail_gemm",
                arch,
                gemm_key,
                gemm_kwargs,
                ws_numel=_partial_ws_numel("tail", plan),
                device_index=index,
            )
            if plan["fused_norm"]:
                # One launch: the GEMM's epilogue warps normalise the latent rows into
                # ``y_workspace`` while its load warp streams the shared-expert K blocks.
                launches = (gemm_launch,)
            else:
                norm_key = norm_kernel_key(plan["early_trigger"], plan["packed_norm"])
                norm_kwargs = dict(
                    routed=routed,
                    norm_weight=norm_weight,
                    y_out=y_workspace,
                    M=T,
                    num_partials=P,
                    eps=float(RMS_EPS),
                    grid=(int(plan["norm_grid"]), 1, 1),
                )
                assert tuple(norm_kwargs) == NORM_KWARGS
                launches = (
                    _make_launch(
                        "tail_norm", arch, norm_key, norm_kwargs, device_index=index
                    ),
                    gemm_launch,
                )
            route = "prefill"
    return KimiK3LatentMoeRunner(
        "tail",
        tp,
        rank,
        T,
        arch,
        sm_count,
        route,
        plan,
        launches,
        (y_workspace, out),
        index,
    )


def kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
) -> torch.Tensor:
    """Complete tail for one rank into the caller-owned ``out`` (``y_workspace`` receives the normalised latent).

    Eager calls share one set of control words per device, stream and size and the stream's partial workspace
    (calls on one stream are stream-ordered); a call under CUDA Graph capture takes fresh words, which are never
    freed, and a block of the graph's memory pool.
    """
    _prepare_tail(
        routed,
        norm_weight,
        up_weight,
        shared_act,
        shared_down_weight,
        out,
        tp=tp,
        rank=rank,
        y_workspace=y_workspace,
        lane=True,
    )()
    return out


__all__ = [
    "DECODE_MAX_T",
    "HIDDEN",
    "LATENT",
    "NUM_EXPERTS",
    "RMS_EPS",
    "ROW_TOKENS",
    "SHARED_INTERMEDIATE",
    "SM_COUNT",
    "QUALIFIED_SM_COUNTS",
    "TAIL_ROWS_STAGGER",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "SUPPORTED_TP",
    "KimiK3LatentMoeRunner",
    "decode_front_plan",
    "decode_kernel_key",
    "decode_symbol",
    "decode_tail_plan",
    "front_kernel_key",
    "generated_program_available",
    "i_local_for_tp",
    "k_up_for_tp",
    "kimi_k3_latent_moe_front",
    "kimi_k3_latent_moe_tail",
    "norm_kernel_key",
    "prefill_tail_plan",
    "prepare_kimi_k3_latent_moe_front",
    "prepare_kimi_k3_latent_moe_tail",
    "required_kernel_keys",
    "route_kernel_keys",
    "split_fin_route",
    "split_plan",
    "tail_gemm_config",
    "tail_gemm_final_ts",
    "tail_gemm_kernel_key",
    "tail_partial_prefetch",
    "tail_sb_norm_n_pad",
    "tail_wide_min_tokens",
    "use_fused_norm",
    "use_packed_norm",
    "use_resident_fused_norm",
    "resident_grid",
    "fused_norm_fits",
    "FUSED_MIN_K2_ITERS",
    "weights_evict_first",
]
