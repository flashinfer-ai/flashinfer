"""NVFP4 SwiGLU MoE built from CuTeDSL kernels.

The input arrives quantized (FlashInfer's per-token NVFP4 activation format: packed E2M1 [M, H / 2], linear E4M3
block scales [M, H / 16], row scales [M]; or, ``CuteMoE(per_token_scale=False)``, its per-tensor format: the same
values and block scales with a unit global scale, no row scales). The kernel sequences:

- direct (M <= 4 at 512 <= I <= 768; M <= 4 of other shapes where ``direct_unit_stages`` finds room): route + copy
  the input rows (``RouteInput``) -> gate/up GEMM per route (``DirectGemm``, SwiGLU; reads the caller's ids to load
  weights before its PDL wait) -> down GEMM per route (weights loaded before its wait) quantizing the BF16
  rows in shared memory (512 <= I <= 768), or loading the NVFP4 rows the gate/up quantized with block scales only ->
  ``ReduceRoutes``;
- grouped, short batches (M * topk < 8192, and longer ones with fewer routes per expert than ``long_rpe``; PDL):
  ``SelectExperts`` (with route ranks; at block_m <= 32 it also lists each expert's routes, ``eroute``) ->
  [``Scatter``, block_m 64 only: copies the input rows to the routed rows] -> gate/up ``GroupedGemm`` (every CTA
  places the tiles itself from the expert counts, ``place_tiles``, and CTA 0 publishes them for the down GEMM; at
  block_m <= 32 it gathers the input rows by token; its epilogue quantizes the SwiGLU output to NVFP4 with block
  scales only, a unit second-level scale as FlashInfer prims_ts: no BF16 intermediate) -> down
  ``GroupedGemm`` (no activation row scales) -> ``ReduceRoutes`` (zeroes the expert route counts for the next call).
  Where the gate/up's tile table does not fit its shared memory (block_m 64 with 4 stages), the Scatter places the
  routes;
- grouped, long batches (M * topk >= 8192 with at least ``long_rpe`` routes per expert on average: 56 gated, 40 / 48
  non-gated; no PDL): ``SelectExperts``
  with per-chunk counts -> ``ScanCounts`` -> ``ScatterChunks`` with in-kernel placement (row-major scales; experts
  padded to 128 rows, 256 for CTA pairs) -> token-major gate/up ``GroupedGemm`` (block_m 128; its epilogue quantizes
  the SwiGLU output to NVFP4 with block scales only; CTA pairs where ``two_cta_gate_up``) -> down (weight-major, or
  token-major CTA pairs over ScatterChunks' pair tiles where ``down_pairs``) -> reduce.

The intermediate is quantized per token (row scale amax / 2688) on the direct path at 512 <= I <= 768, with block
scales only (a unit second-level scale) everywhere else (``CuteMoE.mid_unit``).

``use_fused_finalize`` (FlashInfer's ``MoEFinalizeConfig`` option of that name): the down GEMM adds each route's output
times its routing weight into the token's output row and there is no ``ReduceRoutes`` (``CuteMoE.fused_finalize``).
On the direct path and the short batches with BF16 atomic adds (``DirectGemm(fused_out)``, weight-major
``GroupedGemm(fused_out)``; the output zeroed by ``RouteInput`` / ``Scatter`` / the gate/up's gather warps); on the
long batches the down GEMM is token-major and adds each tile row with one bulk reduce (``GroupedGemm(token_major,
fused_out)``, as CTA pairs of 256 tokens x 256 weight rows where ``down_pairs``; the output zeroed by the gate/up's
load warp, ``GroupedGemm(zero_rows)``). The sum is then rounded to BF16 after every add, in a varying
order: the output is not deterministic and its error is larger (relative RMSE ~0.004 instead of ~0.0025).

``CuteMoE(E, topk, H, I, max_tokens)`` allocates one workspace for every batch size up to ``max_tokens``; a call
with the quantized input ``q`` [m, H / 2] uint8, ``sf`` [m, H / 16] (E4M3 as uint8 or float8_e4m3fn), ``gs`` [m]
FP32, the weights (``weights.KernelWeights``) and the pre-routed ``topk_ids`` / ``topk_weights`` launches the
sequence of that M (its ``Plan``, built on the first call of the M) on the current stream and returns the BF16 output [m, H]. Kernels with the same static configuration are compiled once per process; the batch
sizes of a power-of-two bucket (``token_bucket``) share them.

The routing is the caller's (FlashInfer's precomputed routing): the routing kernel (``SelectExperts`` / ``RouteInput``)
does no top-k and the kernels read the caller's ids (and our copy of its weights); the direct gate/up reads the ids
before its PDL wait. Ids outside this rank's experts are skipped as FlashInfer's TRT-LLM routing
skips them (no row, weight not read, a zero row in front of ``routed`` for ReduceRoutes); expert parallelism
(``local_expert_offset`` / ``local_num_experts``) only moves that range.

Per-tensor input (``per_token_scale=False``, FlashInfer's ``QuantConfig(per_token_scale=False)``): the same kernels
with ``unit_a_gs`` / ``row_scales=False``: no row scale is loaded, copied or multiplied, except by the short batches'
gate/up without Scatter (block_m <= 32), which keeps the per-token kernel and reads unit row scales (without its
row-scale loads that epilogue was 0.5-2% slower at H = 2048, M = 8-32). The outputs equal those of a per-token call
whose row scales are all 1. A calibrated activation scale is the caller's to fold into the gate/up's
per-expert ``w1_gs``, as FlashInfer's per-tensor backends fold it into their ``w1_alpha``.

Options matching FlashInfer's CuTe-DSL backend: ``activation=SiTU(...)``, ``enable_pdl=False`` (the same kernels
launched without PDL).
"""

import dataclasses
import enum
import functools
import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, NamedTuple

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from .activation import as_activation, is_gated
from .direct import DirectGemm, make_direct_descriptors
from .gemm import GroupedGemm, MaxnregPatchUnused
from .placement import ScanCounts
from .quantize import RouteInput
from .reduce import ReduceRoutes, reduce_shape
from .router import SelectExperts
from .scatter import Scatter, ScatterChunks
from .weights import check_weights, view_weights

# routing and path-selection constants
SORT_CHUNK = 8192  # routes from which routing counts and places chunks (no PDL)
ROUTE_TOKENS = 16  # tokens per routing CTA / counting chunk on long batches
# routes per expert from which a long batch's unfused down GEMM is bound by its MMAs rather than by the weights' DRAM
# traffic: from here it takes K 256 tiles with a K tail or as CTA pairs with a K loop of at least 8 stages of K 512
# (I >= 4096), and runs as CTA pairs at H < 4096 or I < 2048 (two_cta_down; at H < 2048 from twice as many)
DOWN_MMA_RPE = 128
# routes per expert from which the unfused down GEMM runs as CTA pairs at H >= 4096 and I >= 2048 (two_cta_down):
# below DOWN_MMA_RPE they are faster only at the GPU's power limit
DOWN_PAIR_RPE = 80
# the gate/up CTA pairs' padding limits (two_cta_gate_up): expected rows padded to 256 over those padded to 128, from
# GU_PAIR_EXP_RPE routes per expert and below
GU_PAIR_PAD = 1.12
GU_PAIR_EXP_RPE = 192
GU_PAIR_PAD_LOW = 1.28
# the long down GEMM's token-major CTA pairs' padding limits (down_pairs): expected rows paired (each expert's 128-row
# blocks two by two) over those padded to 128; fused at I > 1024 and at 768 <= I <= 1024, unfused at 1024 < I <= 4096
# and at I > 4096
FUSED_DOWN_PAIR_PAD = 1.25
FUSED_DOWN_PAIR_PAD_SHORT = 1.10
DOWN_PAIR_PAD = 1.12
DOWN_PAIR_PAD_LONG_K = 1.05


def long_rpe(gated, k):
    """Routes per expert from which M * topk >= SORT_CHUNK takes the long-batch path (select_schedule): 56 for a gated
    activation; non-gated (ReLU2) 40 at H <= 1024 (its short-batch gate/up, two K stages of 512, is slower there), else
    48. Measured in short bursts (high clocks) and in the steady state of continuous load at the GPU's power limit:
    non-gated at H = 2048 the short path is faster in both below 48; from 48 the unfused long path is faster in the
    steady state (from 52 by 3%) and slower in bursts, and the steady state decides."""
    return 56 if gated else 40 if k <= 1024 else 48


def down_l2_tiles(m, h, i):
    """Short batches (PDL): tiles past the first whose weights each down GEMM CTA L2-prefetches before its
    griddepcontrol.wait (``GroupedGemm(l2_tiles=)``; ``h``, ``i``: the MoE's H and I): 0 tiles at M <= 2, 1 at
    M <= 8, 2 at M <= 512 with H < 4096, else 0; none where a tile's weights (128 rows x I / 2 bytes) exceed 128 KB
    (I > 2048: Nemotron 3 Super and Ultra were 0.5-1.5% faster without). The gate/up GEMM prefetches none."""
    if i > 2048:
        return 0
    if m <= 8:
        return 0 if m <= 2 else 1
    return 2 if m <= 512 and h < 4096 else 0


def weight_evict(which, m, k512=0, long=False, bm=128):
    """Whether the GEMM ``which`` ("GU" gate/up, "DN" down) loads its weights with L2 EVICT_FIRST on their
    last use (``GroupedGemm(evict_weights=)``; not the direct path): gate/up at M <= 1024 and, on the short-batch path
    (``long`` False) with a K loop of at most 8 stages of K 512, at any M; down at M <= 2048. Not a grouped gate/up at
    M >= 8 with a token tile ``bm`` > 32 whose K loop has ``k512`` >= 12 stages of K 512 (0: another K tile): there it
    is 0.5-1.6% slower, with ``bm`` <= 32 0.6-2.8% faster. The direct path does not use it (``CuteMoE``)."""
    if which == "GU" and m >= 8 and k512 >= 12 and bm > 32:
        return False
    if which == "GU" and not long and 1 <= k512 <= 8:
        return True
    return m <= (1024 if which == "GU" else 2048)


def sf_gather_mode(m, bm, k):
    """Short batches: how the gate/up GEMM (K = k, token tile bm) gets the activation block scales
    (``GroupedGemm(x_sf_gather=)``; ``Scatter(rowtok=, sf_linear=)``):

    - 0 (M <= 4): the TMA loads the scatter's tile-packed copy (one 4-byte store per 512-byte block in the scatter);
    - 3 (bm 64 and K >= 4096, i.e. at least 8 K stages of 512): the scatter writes row-major scale rows, the TMA loads
      a bm x 512 / 16 box per stage and two scale warps rearrange it in shared memory;
    - 1 (else): the scatter writes each row's token, two scale warps load the scales of each stage straight from the
      MoE input (no copy).

    Modes 1 and 3 are bit-identical with mode 0."""
    if m <= 4:
        return 0
    return 3 if bm == 64 and k >= 4096 else 1


# FP4 weight bytes per SM from which a grouped GEMM launch claims its tiles (claim_tiles), about 1.8 MB
CLAIM_BYTES = 56 * 32768


def gate_up_4th_stage(gu_claim, bm, bk, k):
    """Whether the grouped gate/up takes a 4th K stage: block_m 64, K 512, a launch that claims its tiles and a K loop
    of at least 8 stages (CuteMoE._build)."""
    return bool(gu_claim) and bm == 64 and bk == 512 and k // 512 >= 8


def claim_tiles(bn, k, sms, max_tiles):
    """Short batches (PDL): the tile count (tiles of bn weight rows, K = k) from which a weight-major grouped GEMM
    launch takes each CTA's tiles after its first two from a global counter instead of round-robin
    (``GroupedGemm(claim_tiles=)``), i.e. CLAIM_BYTES of weights per SM; 0 (round-robin only) when no launch reaches
    it (``max_tiles``)."""
    t = cdiv(CLAIM_BYTES * sms, bn * k // 2)
    return t if max_tiles >= t else 0


def cdiv(a, b):
    return (a + b - 1) // b


def two_cta_down(m, topk, h, i, experts):
    """Long batches, unfused: whether the down GEMM (N = h = H, K = i = I) runs as weight-major CTA pairs
    (``GroupedGemm(two_cta=True)``, wm2: MMAs of 256 weight rows x 128 tokens, each CTA loading half of the tokens and
    storing its 128 weight rows' outputs), with a K loop longer than one stage of K 512 (I = 512, Qwen3.6-35B: no
    gain), from these routes per expert on average: DOWN_PAIR_RPE at H >= 4096 and I >= 2048, else DOWN_MMA_RPE at
    H >= 2048 and twice DOWN_MMA_RPE at H < 2048 (Nemotron 3 Super's H = 1024: the pairs are slower below that). From
    DOWN_MMA_RPE the MMAs rather than the weights' DRAM traffic bound the down GEMM and the pairs are faster at any
    clock. From DOWN_PAIR_RPE to DOWN_MMA_RPE, at H >= 4096 and I >= 2048, they are 1.3-3% faster in the steady state
    of continuous load at the GPU's power limit (they move fewer bytes) and 1-3.5% slower in short bursts at higher
    clocks; the steady state decides. At H = 2048 (I = 5120) they gain at most 1.3% there in the steady state and are
    3-4% slower in bursts; at I < 2048 the GPU stays below its power limit and they are slower. The thresholds are
    measured on the test set's nine shapes (H = 1024 only on one)."""
    rpe = m * topk / experts
    need = (
        2 * DOWN_MMA_RPE
        if h < 2048
        else DOWN_PAIR_RPE
        if h >= 4096 and i >= 2048
        else DOWN_MMA_RPE
    )
    return h % 256 == 0 and i > 512 and rpe >= need


def padded_rows(m, topk, experts, tile):
    """Expected rows of one expert's routes padded to a multiple of ``tile``: its route count binomial (M tokens, each
    routed to it with probability topk / experts, as evenly spread routes), in the normal approximation,
    tile * sum over j >= 0 of P(routes > j * tile)."""
    p = topk / experts
    mu = m * p
    sd = math.sqrt(max(mu * (1 - p), 1e-9))
    return tile * sum(
        0.5 * math.erfc((j * tile + 0.5 - mu) / (sd * math.sqrt(2)))
        for j in range(int((mu + 8 * sd) // tile) + 1)
    )


def two_cta_gate_up(m, topk, k, experts):
    """Long batches: whether the token-major gate/up runs as CTA pairs (``GroupedGemm(two_cta=)``, 5 stages; experts
    padded to 256 rows, the down GEMM skipping the empty 128-row halves): K = H >= 4096 and padding to 256 rows adds
    at most 15% to padding to 128 for the average routes per expert rpe = M * topk / E (ceil(rpe / 256) * 256 <=
    1.15 * ceil(rpe / 128) * 128), and the expected rows padded to 256 at most GU_PAIR_PAD (from GU_PAIR_EXP_RPE routes
    per expert) or GU_PAIR_PAD_LOW (below) times those padded to 128 (``padded_rows``, evenly spread routes). The
    average misjudges an average just above or below a multiple of 256 or 128, where about half of the experts pad to
    the next 256: at 256 routes per expert 20% more rows (the pairs 2-9% slower), at 384-390 11-13% (as fast or
    slower), at 128-130 28-35% (1-7% slower). Below GU_PAIR_EXP_RPE the pairs gain more and stay faster to an expected
    ratio of about 1.15-1.27 (130-138 routes per expert, the larger H the higher); at 512 (1.11) they are faster in
    some modes and shapes, slower in others. Measured at 100-512 routes per expert on five shapes; skewed routing pads
    more."""
    rpe = m * topk / experts
    return (
        k >= 4096
        and -(-rpe // 256) * 256 <= 1.15 * (-(-rpe // 128) * 128)
        and padded_rows(m, topk, experts, 256)
        <= (GU_PAIR_PAD if rpe >= GU_PAIR_EXP_RPE else GU_PAIR_PAD_LOW)
        * padded_rows(m, topk, experts, 128)
    )


def down_pairs(m, topk, i, experts, fused):
    """Long batches: whether the down GEMM (K = i = I) runs token-major as CTA pairs (``GroupedGemm(two_cta,
    pair_table)``: MMAs of 256 tokens x 256 weight rows over each expert's 128-row blocks two by two,
    ``ScatterChunks(pairs=True)``; ``fused``: each row added into its token's output, else stored into its routed
    row): where the expected rows paired (``padded_rows`` at 256) are at most FUSED_DOWN_PAIR_PAD (fused, I > 1024),
    FUSED_DOWN_PAIR_PAD_SHORT (fused, 768 <= I <= 1024), DOWN_PAIR_PAD (unfused, 1024 < I <= 4096) or
    DOWN_PAIR_PAD_LONG_K (unfused, I > 4096) times those padded to 128.
    The pairs load each weight row once per 256 tokens (and, against the tiles of 128 weight rows, each token row once
    per 256 weight rows): about half the bytes into the SMs of the fused 1-CTA token-major GEMM. They compute the
    second block of an expert's odd last pair (no output): one block per expert (ratio 2, below 128 routes per expert)
    is 4-20% slower.
    Fused (against the 1-CTA token-major GEMM): 4-12% faster end to end at I >= 2048 and ratios up to 1.25; at
    1.29-1.35 even to 4% faster on the gated shapes with I >= 2048 and up to 2% slower on Nemotron 3 Super. At
    I <= 1024 (a short K loop, the 1-CTA GEMM already at 256 weight rows) they gain less: at I = 768 2-4% to a ratio of
    1.10 and up to 3% slower at 1.15; at I = 512 1-8% slower at most ratios (I = 768 and 512 each measured on one
    shape).
    Unfused (against the weight-major GEMM, CTA pairs from two_cta_down): 1-9% faster at ratios up to 1.12, even at
    1.15-1.20, 1-5% slower from 1.22; at I <= 1024 3-13% slower. At I = 5120 (Nemotron 3 Ultra, its weight-major CTA
    pairs with K 256; measured on that shape only) 2-4% faster to a ratio of 1.02, even to 3% faster at 1.05, up to 1%
    slower in short bursts at 1.08 and 0.6-2% at 1.12."""
    if fused:
        limit = (
            FUSED_DOWN_PAIR_PAD
            if i > 1024
            else FUSED_DOWN_PAIR_PAD_SHORT
            if i >= 768
            else 0
        )
    else:
        limit = (
            DOWN_PAIR_PAD
            if 1024 < i <= 4096
            else DOWN_PAIR_PAD_LONG_K
            if i > 4096
            else 0
        )
    return limit > 0 and padded_rows(m, topk, experts, 256) <= limit * padded_rows(
        m, topk, experts, 128
    )


class Tiles(NamedTuple):
    block_n: int
    block_k: int
    stages: int


class Schedule(NamedTuple):
    direct: bool
    block_m: int
    gate_up: Tiles
    down: Tiles
    fuse_quantization: bool  # direct: the down GEMM quantizes the intermediate per token (else the gate/up, unit)
    launch_pdl: bool


def direct_unit_stages(m, n, topk, k, sms, sm_smem):
    """Gate/up stages of the unit direct GEMMs at M <= 4 (shapes outside 512 <= I <= 768; n = 2 I, k = H), else 0
    (the grouped path).

    The most stages (5, 4, 3; 22.5 KB each at BK 256, about 2 KB per CTA beyond them) at which the gate/up's
    m x topk x n / 128 CTAs (one per (route, 128 rows), not persistent) are all resident at once: every CTA then issues
    its weight loads before its PDL wait. The exception, M = 1 beyond one wave: 5 stages where the gate/up's CTAs fit
    two waves and the down GEMM's (topk x H / 128 at 4 stages) one."""
    slots = lambda st: sms * (sm_smem // (st * 22528 + 2048))  # noqa: E731
    gu, dn = m * topk * (n // 128), m * topk * (k // 128)
    for st in (5, 4, 3):
        if gu <= slots(st):
            return st
    return 5 if m == 1 and gu <= 2 * slots(5) and dn <= slots(4) else 0


def direct_down_stages(m, n, topk, k, sms, sm_smem):
    """Stages of the unit direct down GEMM (n = 2 I, k = H): of 4, 3, 2 (at most I / 256), the fewest waves of its
    m x topk x k / 128 CTAs (not persistent; at most one per (route, 128 rows)), then the most stages. CTAs past the
    first wave issue their weight loads only when an earlier CTA ends; one wave of 2-stage CTAs streams the weights
    better than more waves of 4-stage ones (448-480 CTAs at 2 stages: 1.3-1.8% faster), as do two waves of 3 stages
    against three of 4 (640 CTAs: 3.7% faster). One wave at 4 stages stays (fewer stages there are 2.5-17% slower)."""
    slots = lambda st: sms * (sm_smem // (st * 22528 + 2048))  # noqa: E731
    dn = m * topk * (k // 128)
    _waves, neg_st = min(
        (-(-dn // slots(st)), -st) for st in (4, 3, 2) if st <= n // 2 // 256
    )
    return -neg_st


def select_schedule(
    m, n, topk, k=2048, experts=None, direct=True, gated=True, *, sms, sm_smem
):
    """The schedule of a batch of ``m`` tokens (n = 2 I, k = H): the direct or grouped path, block_m and the gate/up's
    tiles. The grouped gate/up takes 4 stages at block_m <= 32 (M <= 256 at E <= 16 topk; one CTA per SM, 189-206 KB
    of shared memory), 3 at block_m 64; ``CuteMoE`` gives it 4 stages at block_m 64 too where its launch claims its
    tiles and its K loop has at least 8 stages of K 512. The down GEMM's K tile and stages are chosen in ``CuteMoE``
    (``GroupedGemm.pick_tiles``).

    M <= 4 takes the per-token direct GEMMs if 512 <= I <= 768 (the direct down GEMM needs >= 2 K stages and stages
    the quantized row in 512 bytes; their stage counts capped at K / 256), else the unit direct GEMMs where
    ``direct_unit_stages`` finds room (the gate/up in one wave, or M = 1 with the gate/up in two and the down in one;
    the down GEMM's stages from ``direct_down_stages``);
    the gate/up K tile is 256 when 512 does not divide H.

    block_m is the smaller of M's step and, given ``experts``, the same step for the routes per expert
    (M * topk / experts).

    Long batches (M * topk >= SORT_CHUNK, no PDL) take block_m 128 and the token-major gate/up (bn 256) unless the
    experts average fewer than ``long_rpe`` routes: then the short-batch path (PDL, weight-major, block_m <= 64; a
    128-token tile would be mostly padding).

    ``direct=False``: never the direct GEMMs.

    ``gated=False`` (a non-gated activation, ReLU2): n = I weight rows (no gate rows); never the direct GEMMs; the
    long batches' token-major gate/up takes bn 256 (256 outputs per tile; where 256 does not divide I, the last N
    tile's rows past I are the TMA's zeros: ``GroupedGemm`` n_tail); token tile 32 up to 24 routes per expert.

    ``sms``, ``sm_smem``: the GPU's SM count and shared memory per SM (the direct path's one-wave rule,
    ``direct_unit_stages``; B200: 148 SMs, 228 KB)."""
    if direct and m <= 4 and 512 <= n // 2 <= 768 and (n // 2) % 256 == 0 and k >= 512:
        stages = 4 if m <= 2 else 3
        # gate/up: the most stages (5 / 4 / 3) at which its CTAs are resident in one wave (direct_unit_stages, as the
        # unit direct path), else 5 at M=1 and the down GEMM's count
        gst = direct_unit_stages(m, n, topk, k, sms, sm_smem) or (
            5 if m == 1 else stages
        )
        return Schedule(
            True,
            1,
            Tiles(128, 256, min(gst, k // 256)),
            Tiles(128, 256, stages),
            m <= 8 and n // 2 <= 1024,
            True,
        )
    # M <= 4, other shapes: the direct GEMMs with the intermediate quantized by the gate/up (block scales only, as the
    # grouped paths: fuse_quantization False; bit-identical to them) where direct_unit_stages finds room (the gate/up's
    # CTAs resident in one wave, or M = 1 with the gate/up in two waves and the down GEMM in one); else the grouped path
    if (
        direct
        and m <= 4
        and k >= 512
        and k % 256 == 0
        and (n // 2) % 256 == 0
        and n // 2 >= 512
    ):  # down: >= 2 K stages
        gst = direct_unit_stages(m, n, topk, k, sms, sm_smem)
        if gst:
            return Schedule(
                True,
                1,
                Tiles(128, 256, min(gst, k // 256)),
                Tiles(128, 256, direct_down_stages(m, n, topk, k, sms, sm_smem)),
                False,
                True,
            )
    long = m * topk >= SORT_CHUNK and not (
        experts and m * topk < long_rpe(gated, k) * experts
    )
    bm = 16 if m <= 128 else 32 if m <= 256 else 64 if m <= 1024 else 128
    if long:
        bm = 128  # the token-major gate/up
    else:
        if experts:
            rpe = m * topk / experts  # routes per expert
            # token tile 16 up to 10 routes per expert, 32 up to 22 at M <= 1024 and 16 beyond (non-gated, ReLU2: 24):
            # at 16-22 routes per expert block_m 32 is 2-6% faster than 64 where M <= 1024 (at 22 1-5%); at 24 still on
            # four of the five gated shapes with M <= 1024 there but up to 2% slower on one (I = 4096), and from 26 64
            # is faster at the power limit; beyond M = 1024 (topk 6, E = 384) the gate/up has no evict_first there and
            # 32 is 1% slower
            r32 = (22 if m <= 1024 else 16) if gated else 24
            bm = min(
                bm, 16 if rpe <= 10 else 32 if rpe <= r32 else 64 if rpe <= 64 else 128
            )
        # the weight-major (short-batch) GEMMs take block_m <= 64 (128 is the token-major gate/up's): short batches
        # with more than 64 routes per expert (few experts, M * topk < SORT_CHUNK, e.g. Mixtral's E = 8 at M=1024)
        bm = min(bm, 64)
    gk = 512 if k % 512 == 0 else 256  # gate/up K tile
    if bm <= 32:
        gate_up = Tiles(128, gk, 4)
    elif bm <= 64:
        gate_up = Tiles(128, gk, 3)
    else:
        # the long batches' token-major gate/up: bn 256, 4 stages of K 256 (the epilogue's scratch at 9 KB makes room
        # for the 4th; CTA pairs take 5 in CuteMoE). Non-gated: 256 outputs per tile, the epilogue in two passes of 128
        # through the gated path's 9 KB scratch (the same shared memory); where 256 does not divide I the weights are
        # addressed per expert and the last N tile's rows past I are zeros (GroupedGemm n_tail)
        gate_up = Tiles(256, 256, 4)
    # the down GEMM's K tile and stages follow from its shared memory (GroupedGemm.pick_tiles, in CuteMoE, which knows
    # the exact configuration): here 128 weight rows only
    down = Tiles(128, 0, 0)
    assert not long or gate_up.block_n == 256 or not gated, (
        "long batches: the token-major gate/up needs 2 I % 256 == 0"
    )
    return Schedule(False, bm, gate_up, down, False, not long)


# The compute capabilities (major * 10 + minor, as FlashInfer's ``XxxConfig.supported(arch)``) the kernels are built for:
# datacenter Blackwell with tcgen05, sm_100a (B200 / GB200) and sm_103a (B300 / GB300). CuTeDSL compiles for the GPU it
# runs on; the SM count and the shared memory per SM and per CTA come from the device (CuteMoE). Tested on SM100
# (B200: 148 SMs, 228 KB per SM, 227 KB per CTA) only: SM103 is declared but not tested yet.
def _require(ok, message, exc=ValueError):
    """Raise ``exc(message)`` unless ``ok``: argument checks that hold under ``python -O``."""
    if not ok:
        raise exc(message)


def _check_tensor(name, t, shape, dtypes):
    """A caller's tensor of ``shape`` and one of ``dtypes`` (ValueError / TypeError)."""
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name}: shape {tuple(t.shape)}, expected {tuple(shape)}")
    if t.dtype not in dtypes:
        raise TypeError(f"{name}: dtype {t.dtype}, expected one of {dtypes}")


SUPPORTED_ARCHS = (100, 103)


def supported(arch):
    """Whether the kernels run on compute capability ``arch`` (major * 10 + minor, e.g. 100)."""
    return arch in SUPPORTED_ARCHS


def check_support(experts, topk, hidden, inter):
    """Raise ValueError for a shape the kernels do not support: the hidden size H a multiple of 256 (the gate/up's K
    tile), the intermediate size I a multiple of 128 (the token-major gate/up's 256 rows; the down GEMM's K tile 128
    where 256 does not divide I), 1 <= topk <= min(E, 32) (SelectExperts ranks a token's routes in one warp), E <= 1024
    (ScatterChunks: a CTA's threads cover max(chunk, E))."""
    if (
        hidden % 256
        or inter % 128
        or not 1 <= topk <= min(experts, 32)
        or experts > 1024
    ):
        raise ValueError(
            f"unsupported MoE shape: H={hidden} (needs H % 256 == 0), I={inter} (I % 128 == 0), "
            f"topk={topk} (1 <= topk <= min(E, 32)), E={experts} (E <= 1024)"
        )


def _value_key(v):
    """``v`` if it compares by value (numbers, strings, None, enums, types, frozen dataclasses, tuples or lists of
    these), else ``_NO_KEY``."""
    if v is None or isinstance(v, (bool, int, float, str, enum.Enum, type)):
        return v
    if isinstance(v, (tuple, list)):
        items = tuple(_value_key(x) for x in v)
        return _NO_KEY if _NO_KEY in items else (type(v).__name__,) + items
    if dataclasses.is_dataclass(v) and v.__dataclass_params__.frozen:
        return v
    return _NO_KEY


_NO_KEY = object()


def _config_key(op):
    """A kernel object's configuration for the compile cache: its class and every attribute its constructor set that
    compares by value, so that a compiled kernel is reused only for an identically configured object (two CuteMoE
    objects of different E or topk share no kernel built for the other)."""
    return (type(op).__name__,) + tuple(
        (k, kv)
        for k, v in sorted(vars(op).items())
        if (kv := _value_key(v)) is not _NO_KEY
    )


def _spec_key(specs):
    """The launch arguments' types for the compile cache: pointers' element types and alignments (``make_ptr``'s
    assumed_align, kept only privately by CuTeDSL's pointer), scalars' types."""
    return tuple(
        (s.dtype, s.align)
        if isinstance(s, _A)
        else "call"
        if isinstance(s, _V)
        else (s.dtype, getattr(s, "_assumed_align", None))
        if hasattr(s, "dtype")
        else type(s).__name__
        for s in specs
    )


_COMPILED = {}  # (device index, argument types, configuration) -> (op, compiled): a device's own module


def _compile(key, op, *args):
    if key not in _COMPILED:
        try:
            fn = cute.compile(op, *args)
        except (
            MaxnregPatchUnused
        ):  # gemm.py's register cap: the public launch bounds from now on
            fn = cute.compile(op, *args)
        _COMPILED[key] = (op, fn)
    return _COMPILED[key]


_PTRS: dict[
    tuple, object
] = {}  # (address, dtype, align) -> pointer argument (make_ptr): launch arguments by address, any object
_PTRS_MAX = 8192  # cleared when full (the arguments in use stay valid: a pointer is only its address and type)


def _ptr_at(addr, dtype, align=16):
    k = (addr, dtype, align)
    p = _PTRS.get(k)
    if p is None:
        if len(_PTRS) >= _PTRS_MAX:
            _PTRS.clear()
        p = _PTRS[k] = make_ptr(
            dtype, addr, cute.AddressSpace.gmem, assumed_align=align
        )
    return p


def _ptr(t, dtype, align=16):
    return _ptr_at(t.data_ptr(), dtype, align)


def _dense(t):
    """``t`` if contiguous at a 16-byte aligned address, else such a copy: FlashInfer's packs guarantee neither, the
    kernels assume both (16-byte loads; their pointer arguments are declared 16-byte aligned)."""
    return (
        t
        if t.is_contiguous() and t.data_ptr() % 16 == 0
        else t.clone(memory_format=torch.contiguous_format)
    )


I32, F32, BF16, U8 = cutlass.Int32, cutlass.Float32, cutlass.BFloat16, cutlass.Uint8
# the caller's topk_weights types (as FlashInfer's cute_dsl: FP32, BF16, FP16), converted to FP32 by the routing kernel
ROUTE_WEIGHT_TYPES = {
    torch.float32: F32,
    torch.bfloat16: BF16,
    torch.float16: cutlass.Float16,
}
FP4, FP8 = cutlass.Float4E2M1FN, cutlass.Float8E4M3FN


def token_bucket(m, max_tokens):
    """The batch size whose static sizes a batch of ``m`` tokens uses (the gate/up's tile table and eroute's stride,
    ``GroupedGemm(max_tiles=, tokens=)``): the next power of two, at most ``max_tokens``. Powers of two compile exactly
    as a batch of their own size; other M reuse their bucket's kernels."""
    return min(1 << (m - 1).bit_length(), max_tokens)


class _V:
    """A launch argument taken from the call: ``f(call) -> argument`` (``_Call``)."""

    __slots__ = ("f",)

    def __init__(self, f):
        self.f = f


class _A(_V):
    """A launch argument: the call's tensor ``name`` as a pointer (``_arg``); ``Plan.args`` updates it from the
    tensor's address alone."""

    __slots__ = ("name", "dtype", "align")

    def __init__(self, name, dtype, align):
        super().__init__(lambda c: _ptr(getattr(c, name), dtype, align))
        self.name, self.dtype, self.align = name, dtype, align


@functools.cache  # (immutable: one per name, type and alignment)
def _arg(name, dtype, align=16):
    """The call's tensor ``name`` as a pointer argument."""
    return _A(name, dtype, align)


# the call tensors whose addresses make a call's key (CuteMoE.__call__), in its order; _Call's "wts" is the plan's
_KEY_FIELDS = (
    "q",
    "sf",
    "gs",
    "win",
    "ids",
    "out",
    "w1",
    "w1_sf",
    "w1_gs",
    "w2",
    "w2_sf",
    "w2_gs",
)
_KEY_INDEX = {n: i for i, n in enumerate(_KEY_FIELDS)}


class _Call(NamedTuple):
    """The tensors of one call: the input (q, sf, gs), the caller's routing weights (win) and ids, the weights the
    kernels read (the routing kernel's copy of the caller's, skipped routes zeroed), the output and the six weight
    tensors. ``dummy``: placeholders for compiling."""

    q: Any
    sf: Any
    gs: Any
    win: Any
    ids: Any
    wts: Any
    out: Any
    w1: Any
    w1_sf: Any
    w1_gs: Any
    w2: Any
    w2_sf: Any
    w2_gs: Any
    dummy: bool = False


class _Build(NamedTuple):
    """Per-plan values the builders share (``CuteMoE._build``)."""

    m: int
    base: int  # expert parallelism: local expert = id - base
    wdt: object  # the caller's routing weights' type
    g: dict  # CuteMoE._geometry(m)
    stream: object


class _Choices(NamedTuple):
    """The grouped paths' launch choices for a batch (``CuteMoE._grouped_choices``)."""

    pdl: bool  # the short batches' PDL chain
    ranked: bool  # short batches: SelectExperts ranks the routes (ecount)
    gu_claim: int  # claim_tiles of the gate/up and down GEMMs (0: round-robin)
    dn_claim: int
    place: int  # the gate/up places its tiles (GroupedGemm place_tiles): 0 no (the Scatter), 1 no Scatter, 2 early
    ns: bool  # no Scatter (place 1)
    sf_gather: int  # the gate/up's activation block scales (GroupedGemm x_sf_gather)


class _Workspace:
    """Zeroed byte buffers by name, each as large as any batch size up to ``max_tokens`` needs (``CuteMoE._sizes``);
    plans of different M share them (one call at a time: the counts are left at zero for the next call)."""

    def __init__(self, sizes, device):
        self.bufs = {
            k: torch.zeros(v, dtype=torch.uint8, device=device)
            for k, v in sizes.items()
            if v
        }

    def view(self, name, shape, dtype=torch.uint8):
        n = math.prod(shape) * dtype.itemsize
        b = self.bufs[name]
        assert n <= b.numel(), f"workspace {name}: {n} > {b.numel()} bytes ({shape})"
        return b[:n].view(dtype).view(shape)


class Plan:
    """The kernels of one batch size and routing weights' type (``CuteMoE.plan``). ``sched``: the schedule (tiles and
    stages as launched); ``mid_unit``: the intermediate quantized with block scales only (else per token, the direct path at
    512 <= I <= 768); ``fused_finalize``."""

    def __init__(self, m, fused):
        self.m, self.fused_finalize = m, fused
        self.sched, self.mid_unit, self.rows = None, True, 0
        self.wts = None  # the routing kernel's copy of the caller's weights
        self.launches = []  # (compiled, argument specs: constants and _V)
        self.descs = {}  # direct path: TMA descriptors per weight tensors
        self.desc_of = {}  # which -> (gemm, a_q, a_sf, weight names)
        self._key = self._args = (
            None  # the last call's addresses and every launch's arguments (args)
        )
        self._by_key = self._other = None

    def desc(self, c, which, dummy):
        """The direct GEMM's TMA descriptors (make_direct_descriptors) for the call's weights, built on first use (so a
        CUDA graph capture needs ``CuteMoE.warmup(weights=)`` or a call with the same weight tensors first)."""
        if c.dummy:
            return _ptr(dummy, cutlass.Int64, 64)
        gemm, a_q, a_sf, wn, sfn = self.desc_of[which]
        w, w_sf = getattr(c, wn), getattr(c, sfn)
        key = (which, w.data_ptr(), w_sf.data_ptr())
        if key not in self.descs:
            if (
                torch.cuda.is_current_stream_capturing()
            ):  # (the copy would be captured as a graph node)
                raise RuntimeError(
                    "CuteMoE: the direct path's TMA descriptors of these weights are copied to the "
                    "device on their first call, which a CUDA graph capture cannot do: call "
                    "CuteMoE.warmup(ms, weights=) or call with these weights before capturing"
                )
            self.descs[key] = make_direct_descriptors(gemm, a_q, a_sf, w, w_sf)
        return _ptr(self.descs[key], cutlass.Int64, 64)

    def args(self, key, make_call):
        """Every launch's arguments for a call whose tensors are at the addresses ``key`` (``_KEY_FIELDS``): built on
        the first call; afterwards only the arguments of the tensors that moved are replaced (pointers cached by
        address, ``_ptr_at``), and the direct path's descriptors (other ``_V``) looked up again."""
        if key == self._key:
            return self._args
        if self._args is None:
            c = make_call()
            self._args = [
                [s.f(c) if isinstance(s, _V) else s for s in specs]
                for _, specs in self.launches
            ]
            self._by_key = [
                [] for _ in _KEY_FIELDS
            ]  # key index -> (launch, position, dtype, align)
            self._other = []  # (launch, position, _V): recomputed from the call when anything moved
            for li, (_, specs) in enumerate(self.launches):
                for pos, s in enumerate(specs):
                    if isinstance(s, _A):
                        if (
                            s.name in _KEY_INDEX
                        ):  # else the plan's own buffer (wts): fixed
                            self._by_key[_KEY_INDEX[s.name]].append(
                                (li, pos, s.dtype, s.align)
                            )
                    elif isinstance(s, _V):
                        self._other.append((li, pos, s))
        else:
            # the call-dependent arguments first (a direct descriptor lookup can raise, e.g. under a capture): nothing
            # is replaced unless all of them succeed, so the arguments always match self._key
            c = make_call() if self._other else None
            other = [(li, pos, s.f(c)) for li, pos, s in self._other or ()]
            args = self._args
            for i, (a, b) in enumerate(zip(key, self._key, strict=False)):
                if a != b:
                    for li, pos, dt, al in self._by_key[i]:
                        args[li][pos] = _ptr_at(a, dt, al)
            for li, pos, v in other:
                args[li][pos] = v
        self._key = key
        return self._args


_SIZES = {}


class CuteMoE:
    """CuTeDSL NVFP4 MoE for every batch size up to ``max_tokens`` (see the module docstring).

    Args:
        num_experts, topk, hidden_size, intermediate_size: E, routes per token, H, I (``check_support``).
        max_tokens: The largest batch; the workspace is allocated once for it (FlashInfer's tune_max_num_tokens).
        use_fused_finalize: The fused finalize (module docstring; every batch size).
        activation: The gated activation (``activation.py``: SwiGLU(alpha, beta, limit), SwiGLUStep, GeGLU, GeGLUTanh,
            SiTU(gate_scale, linear_scale), or FlashInfer's objects of these names, as ``MoEConfig(activation=)``);
            None: the plain SwiGLU.
        enable_pdl: Launch with PDL (FlashInfer's ``ExecutionConfig(enable_pdl=)``; None: True). False: the same
            kernels without the PDL launch attribute (griddepcontrol a no-op, the kernels serialized).
        per_token_scale: The input's scales (FlashInfer's ``QuantConfig(per_token_scale=)``): True, per-token (a row
            scale per token, ``gs``); False, per-tensor (unit global scale, no row scales: ``gs`` is None).
        local_expert_offset, local_num_experts: Expert parallelism (FlashInfer's ``ExpertConfig`` fields): this rank
            holds experts [offset, offset + local) of ``num_experts``; the weights are those local experts, the
            ids global; routes to other ranks' experts are skipped, so the output is this rank's part of the sum (the
            communication between ranks is the caller's, as in FlashInfer).
        device: The CUDA device (default: the current one): the plans are compiled and the kernels launched there (on
            its current stream), whatever the current device; the call's tensors must be on it.

    Calls: ``moe(q, sf, gs, weights, topk_ids=, topk_weights=)``, see ``__call__``. The first call of a batch size and
    routing weights' type builds its ``Plan`` (compiling kernels not compiled yet in this process, and on the direct path the
    weights' TMA descriptors): ``warmup`` (or ``plan``, or a call) before capturing a CUDA graph. One call at a time per
    object (its workspace; the expert counts are left at zero for the next call), as one ``MoELayer`` per stream in
    FlashInfer; the weights come with each call, so one object can serve every layer of a shape.
    """

    def __init__(
        self,
        num_experts,
        topk,
        hidden_size,
        intermediate_size,
        max_tokens=8192,
        *,
        use_fused_finalize=False,
        activation=None,
        enable_pdl=None,
        per_token_scale=True,
        local_expert_offset=0,
        local_num_experts=None,
        device=None,
    ):
        # E: this rank's experts (the weights, counts and buffers); E_total: all experts (the routes per expert that
        # select_schedule's rules use, the routing width)
        self.E_total = num_experts
        self.E = num_experts if local_num_experts is None else local_num_experts
        self.base = local_expert_offset
        _require(
            self.base >= 0 and self.E >= 1 and self.base + self.E <= num_experts,
            f"local experts [{self.base}, {self.base + self.E}) outside the {num_experts} experts",
        )
        self.topk, self.H, self.I = topk, hidden_size, intermediate_size
        self.activation = as_activation(activation)
        # gated: gate/up weights of 2I rows (gate and up interleaved per 8); non-gated (ReLU2): the I up rows
        self.gated = is_gated(self.activation)
        self.N1 = 2 * intermediate_size if self.gated else intermediate_size
        check_support(num_experts, topk, hidden_size, intermediate_size)
        _require(max_tokens >= 1, f"max_tokens={max_tokens}: at least 1")
        self.max_tokens = max_tokens
        self.fused_finalize = use_fused_finalize
        self.enable_pdl = enable_pdl is not False
        # per-tensor input: the kernels take no row scales (GroupedGemm / DirectGemm unit_a_gs, Scatter /
        # ScatterChunks / RouteInput row_scales=False). Except the gate/up without Scatter (block_m <= 32, route_rows): there the per-token kernel reads unit row
        # scales from ``_ones`` (_queue_gate_up)
        self.row_scales = bool(per_token_scale)
        dev = torch.device("cuda") if device is None else torch.device(device)
        # an index-less "cuda" is the current device, so the weights' devices compare equal
        self.dev = (
            torch.device(dev.type, torch.cuda.current_device())
            if dev.index is None
            else dev
        )
        props = torch.cuda.get_device_properties(self.dev)
        if not supported(props.major * 10 + props.minor):
            raise ValueError(
                f"CuteMoE: {props.name} (compute capability {props.major}.{props.minor}); the kernels need "
                f"datacenter Blackwell with tcgen05: {', '.join(f'SM{a}' for a in SUPPORTED_ARCHS)}"
            )
        # the SM count, shared memory per SM and the most a CTA can opt in to (B200: 148, 233472, 232448)
        self.sms, self.sm_bytes = (
            props.multi_processor_count,
            props.shared_memory_per_multiprocessor,
        )
        self.cta_smem = props.shared_memory_per_block_optin
        key = (
            num_experts,
            self.E,
            topk,
            hidden_size,
            intermediate_size,
            max_tokens,
            use_fused_finalize,
            self.sms,
            self.sm_bytes,
            self.cta_smem,
            self.gated,
        )
        if key not in _SIZES:
            sizes = {}
            for m in range(1, max_tokens + 1):
                for k, v in self._sizes(m).items():
                    sizes[k] = max(sizes.get(k, 0), v)
            _SIZES[key] = sizes
        self.ws = _Workspace(_SIZES[key], self.dev)
        self._dummy = torch.zeros(4, dtype=torch.int32, device=self.dev)
        self._tokens = {}  # M -> Int32 [1] (_tokens_arg)
        # per-tensor input: unit row scales for the gate/up without Scatter (_queue_gate_up)
        self._ones = (
            None
            if self.row_scales
            else torch.ones(max_tokens, dtype=torch.float32, device=self.dev)
        )
        d = self._dummy
        self._dc = _Call(*(d,) * 13, dummy=True)
        self._plans = {}
        self._wok = set()  # the weights checked: addresses and layouts (_weights)

    # ------------------------------------------------------------------------------------------------------ geometry

    def _geometry(self, m):
        """The schedule and the grouped paths' row geometry of a batch of ``m`` tokens."""
        E, N1, H, T = self.E, self.N1, self.H, self.topk
        # (routes per expert: of all experts)
        s = select_schedule(
            m,
            N1,
            T,
            H,
            self.E_total,
            direct=self.gated,
            gated=self.gated,
            sms=self.sms,
            sm_smem=self.sm_bytes,
        )
        g = dict(s=s, routes=m * T)
        if s.direct:
            return g
        bm, long = s.block_m, not s.launch_pdl
        # long batches: the token-major gate/up as CTA pairs where two_cta_gate_up, the experts padded to 256 rows;
        # non-gated with 256 not dividing I: 1-CTA (its last N tile's rows past I, GroupedGemm n_tail, 1-CTA only)
        two = (
            long
            and two_cta_gate_up(m, T, H, self.E_total)
            and (self.gated or self.I % 256 == 0)
        )
        # long batches: the down GEMM as token-major CTA pairs (down_pairs)
        dpair = long and down_pairs(m, T, self.I, self.E_total, self.fused_finalize)
        pad = 2 * bm if two else bm
        rows = cdiv(m * T + E * (pad - 1), pad) * pad
        mb = token_bucket(m, self.max_tokens)
        g.update(
            bm=bm,
            long=long,
            two=two,
            dpair=dpair,
            pad=pad,
            rows=rows,
            tiles=rows // bm,
            chunks=cdiv(m, ROUTE_TOKENS),
            mb=mb,
            # the gate/up's tile table (place_tiles) holds the bucket's tiles
            tiles_cap=cdiv(mb * T + E * (pad - 1), pad) * pad // bm,
        )
        return g

    def _sizes(self, m):
        """Workspace bytes by buffer for a batch of ``m`` tokens (an upper bound of what ``_build`` views)."""
        g = self._geometry(m)
        E, N1, H, fused = self.E, self.N1, self.H, self.fused_finalize
        s, routes = g["s"], g["routes"]
        sz = dict(ids=routes * 4, wts=routes * 4)
        if s.direct:
            unit = not s.fuse_quantization
            sz.update(
                q=m * H // 2,
                sf=m * (H // 64) * 512,
                gs=m * 4,
                mid=routes * (N1 // 4 if unit else N1),
                mid_sf=routes * (N1 // 128) * 512 if unit else 0,
                routed=0 if fused else (routes + 1) * H * 2,
            )
            return sz
        rows, tiles, bm, long = g["rows"], g["tiles"], g["bm"], g["long"]
        sfm = 3 if long else sf_gather_mode(m, bm, H)
        k2 = self.I  # the intermediate's width (the down GEMM's K)
        sz.update(
            count=4,
            experts=tiles * 4,
            limits=tiles * 4,
            tbase=tiles * 4,
            inverse=routes * 4,
            tmaps=2 * self.sms * 512,
            claims=8,
            partial=(g["chunks"] + 1) * E * 4 if long else 0,
            eroute=E * g["mb"] * 4 if not long and bm <= 32 else 0,
            ecount=0 if long else E * 4,
            q=rows * 4 if long else rows * H // 2,
            sf=rows * H // 16
            if sfm == 3
            else rows * 4
            if sfm == 1
            else tiles * (H // 64) * 512,
            gs=rows * 4,
            q2=rows * k2 // 2,
            sf2=tiles * (k2 // 64) * 512,
            routed=0 if fused else (rows + 1) * H * 2,
            rtok=rows * 4 if fused else 0,
            rwt=rows * 4 if fused else 0,
            # the down GEMM's pair tiles (count, then each pair's first block; at most tiles of them)
            pairs=(tiles + 1) * 4 if g["dpair"] else 0,
        )
        return sz

    # ---------------------------------------------------------------------------------------------------------- plans

    def plan(self, m, weights_dtype=torch.float32):
        """The ``Plan`` of a batch of ``m`` tokens whose routing weights (topk_weights) are of ``weights_dtype`` (FP32,
        BF16 or FP16), built on first use."""
        _require(
            1 <= m <= self.max_tokens,
            f"M={m} outside 1..{self.max_tokens} (max_tokens)",
        )
        _require(
            weights_dtype in ROUTE_WEIGHT_TYPES,
            f"routing weights of {weights_dtype}",
            TypeError,
        )
        key = (m, weights_dtype)
        if key not in self._plans:
            # compiled, and its buffers made, on this object's device
            with torch.cuda.device(self.dev):
                self._plans[key] = self._build(m, ROUTE_WEIGHT_TYPES[weights_dtype])
        return self._plans[key]

    def warmup(
        self,
        ms=None,
        *,
        weights_dtypes=(torch.float32,),
        weights=None,
    ):
        """Build the plans of the batch sizes ``ms`` (default: every M up to max_tokens) for the routing weights' types
        ``weights_dtypes`` before their first call, compiling the kernels this
        process has not compiled yet; with ``weights`` (``KernelWeights`` or a weight view), also the direct path's
        TMA descriptors of those weights. A first call does both otherwise, and a CUDA graph capture cannot: the
        descriptors are copied to the device. Returns the number of kernels compiled."""
        ms = range(1, self.max_tokens + 1) if ms is None else ms
        c = None
        if weights is not None:
            w, _ = self._weights(weights)
            d = self._dummy
            c = _Call(d, d, d, d, d, d, d, *w)
        n = len(_COMPILED)
        with torch.cuda.device(self.dev):
            for m in ms:
                for wd in weights_dtypes:
                    pl = self.plan(m, wd)
                    for which in pl.desc_of if c is not None else ():
                        pl.desc(c, which, self._dummy)
        return len(_COMPILED) - n

    def _add(self, pl, op, specs, stream):
        """Queue the kernel ``op`` with the argument ``specs`` (constants and ``_V``), compiled once per process per
        device and configuration (the device, the arguments' types and ``_config_key(op)``; its placeholder arguments
        built only then); returns the op compiled for it."""
        if not self.enable_pdl:  # the same kernel launched without PDL (gemm.use_pdl)
            op.no_pdl = True
        key = (self.dev.index, _spec_key(specs), _config_key(op))
        if key in _COMPILED:
            op, compiled = _COMPILED[key]
        else:
            args = [s.f(self._dc) if isinstance(s, _V) else s for s in specs]
            op, compiled = _compile(key, op, *args, stream)
        pl.launches.append((compiled, list(specs)))
        return op

    def _build(self, m, wdt=F32):
        """The ``Plan`` of a batch of ``m`` tokens: the direct path (``_build_direct``) or the grouped paths
        (``_build_grouped``), its kernels compiled and queued in launch order."""
        g = self._geometry(m)
        pl = Plan(m, self.fused_finalize)
        pl.sched = g["s"]
        # the routing kernel's copy of the caller's weights (0 for skipped routes)
        pl.wts = self.ws.view("wts", (m, self.topk), torch.float32)
        # the caller's weights' type, read by the routing kernel (SelectExperts / RouteInput); expert parallelism:
        # local = id - base
        b = _Build(
            m,
            self.base,
            wdt,
            g,
            cuda.CUstream(torch.cuda.current_stream(self.dev).cuda_stream),
        )
        if g["s"].direct:
            self._build_direct(pl, b)
        else:
            self._build_grouped(pl, b)
        return pl

    def _build_direct(self, pl, b):
        """M <= 4 (``select_schedule``): RouteInput -> DirectGemm gate/up -> DirectGemm down [-> ReduceRoutes]."""
        E, N1, H, T = self.E, self.N1, self.H, self.topk
        m, base, wdt, stream = b.m, b.base, b.wdt, b.stream
        act, fused, ws, dummy = (
            self.activation,
            self.fused_finalize,
            self.ws,
            self._dummy,
        )
        s, routes = pl.sched, b.g["routes"]
        f32, bf16 = torch.float32, torch.bfloat16
        # fuse_quantization (512 <= I <= 768): the down GEMM quantizes the BF16 intermediate per token; else
        # (direct_unit_stages, select_schedule) the gate/up quantizes it with block scales only (DirectGemm unit_out,
        # the grouped paths' numbers) and the down loads it through the TMA
        unit = not s.fuse_quantization
        pl.mid_unit = unit
        q, sf, gs = (
            ws.view("q", (m, H // 2)),
            ws.view("sf", (m, H // 64, 512)),
            ws.view("gs", (m,), f32),
        )
        mid = (
            ws.view("mid", (routes, N1 // 4))
            if unit
            else ws.view("mid", (routes, N1 // 2), bf16)
        )
        mid_sf = ws.view("mid_sf", (routes, N1 // 128, 512)) if unit else dummy
        routed = (
            dummy if fused else ws.view("routed", (routes + 1, H), bf16)[1:]
        )  # row -1: zeros (skipped routes)
        # copy the input rows (bound at call time) and the routing weights; every weight load before its GEMM's PDL
        # wait: both GEMMs read the caller's ids before their wait (the gate/up triggers after its wait)
        op = RouteInput(
            H,
            num_experts=E,
            topk=T,
            fused=fused,
            base=base,
            win_dtype=wdt,
            row_scales=self.row_scales,
        )
        self._add(
            pl,
            op,
            [
                _arg("q", U8),
                _arg("sf", U8, 4),
                _arg("gs", F32, 4),
                _ptr(q, U8),
                _ptr(sf, U8),
                _ptr(gs, F32),
                _arg("ids", I32),
                _arg("wts", F32),
                _arg("out", BF16),
                _arg("win", wdt),
                I32(m),
            ],
            stream,
        )
        # no L2 evict_first on the direct path (the grouped paths: weight_evict)
        gu = DirectGemm(
            N1,
            H,
            s.gate_up.block_k,
            s.gate_up.stages,
            True,
            T,
            trigger_late=True,
            unit_out=unit,
            unit_a_gs=not self.row_scales,
            activation=act,
            num_experts=E,
            base=base,
        )
        pl.desc_of["gu"] = (gu, q, sf, "w1", "w1_sf")
        mid_t = U8 if unit else BF16
        self._add(
            pl,
            gu,
            [
                _ptr(gs, F32),
                _arg("w1_gs", F32),
                _arg("ids", I32),
                _ptr(mid, mid_t),
                _V(lambda c: pl.desc(c, "gu", dummy)),
                _ptr(mid_sf, U8) if unit else _ptr(gs, BF16),
                _ptr(dummy, F32),  # (rw_ptr: the down GEMM's)
                I32(routes),
                I32(E),
            ],
            stream,
        )
        kd = N1 // 2
        # (fused: the routes added into the output with their routing weights, rw_ptr)
        dn = DirectGemm(
            H,
            kd,
            s.down.block_k,
            s.down.stages if unit else kd // s.down.block_k,
            False,
            1,
            x_quant=not unit,
            unit_a_gs=unit,
            fused_out=T if fused else 0,
            num_experts=E,
            base=base,
        )
        pl.desc_of["dn"] = (
            dn,
            mid if unit else None,
            mid_sf if unit else None,
            "w2",
            "w2_sf",
        )
        self._add(
            pl,
            dn,
            [
                _arg("w2_gs", F32),
                _arg("w2_gs", F32),
                _arg("ids", I32),
                _arg("out", BF16) if fused else _ptr(routed, BF16),
                _V(lambda c: pl.desc(c, "dn", dummy)),
                _ptr(dummy, BF16) if unit else _ptr(mid, BF16),
                _arg("wts", F32) if fused else _ptr(dummy, F32),
                I32(routes),
                I32(E),
            ],
            stream,
        )
        if fused:
            return
        # (a skipped route's row is zeros and its weight 0)
        red = ReduceRoutes(H, T, 128, 1, inverse=False, launch_pdl=s.launch_pdl)
        self._add(
            pl,
            red,
            [
                _ptr(routed, BF16),
                _arg("wts", F32),
                _arg("out", BF16),
                _ptr(dummy, I32),
                _ptr(dummy, I32),
                I32(m),
            ],
            stream,
        )

    def _build_grouped(self, pl, b):
        """The short (PDL) and long batches: SelectExperts [-> ScanCounts] [-> Scatter / ScatterChunks] -> gate/up
        GEMM -> down GEMM [-> ReduceRoutes] (the module docstring): the launch choices
        (``_grouped_choices``), the workspace views (``_grouped_views``), then the kernels in launch order.

        The gate/up epilogue quantizes its SwiGLU output with a unit second-level scale (block scales only, as
        FlashInfer prims_ts; GroupedGemm act); the down GEMM then neither loads nor multiplies row scales (unit_a_gs)."""
        c = self._grouped_choices(b)
        v = self._grouped_views(pl, b, c)
        self._queue_routing(pl, b, c, v)
        self._queue_gate_up(pl, b, c, v)
        self._queue_down(pl, b, c, v)

    def _grouped_choices(self, b):
        """The grouped paths' launch choices for a batch (``_Choices``): tile claiming, the gate/up's tile placement,
        the gate/up's activation block-scale source."""
        E, N1, H, T = self.E, self.N1, self.H, self.topk
        m, g = b.m, b.g
        s, routes, bm, long = g["s"], g["routes"], g["bm"], g["long"]
        pdl = s.launch_pdl
        # tiles from which the grouped GEMMs claim (claim_tiles), from an upper bound on the launch's row blocks. The
        # gate/up claims only while an expert's routes fit a token tile on average (routes < E * bm).
        sms = self.sms
        rb = routes // bm + min(routes, E)
        gu_claim = (
            claim_tiles(s.gate_up.block_n, H, sms, rb * cdiv(N1, s.gate_up.block_n))
            if pdl and s.gate_up.block_n == 128 and routes < self.E_total * bm
            else 0
        )
        dn_claim = (
            claim_tiles(s.down.block_n, self.I, sms, rb * cdiv(H, s.down.block_n))
            if pdl
            else 0
        )
        # short batches: SelectExperts ranks each route within its expert (atomics on ecount, zero on entry); the
        # Scatter (block_m 64) or the gate/up through eroute (block_m <= 32) turns ranks into rows; ReduceRoutes, or
        # the down GEMM with the fused finalize, zeroes ecount for the next call.
        ranked = not long
        # short batches: every gate/up CTA places the tiles itself from the expert counts (GroupedGemm place_tiles;
        # CTA 0 publishes the table for the down GEMM), instead of the Scatter. block_m <= 32: no Scatter
        # (SelectExperts lists each expert's routes in eroute; the gate/up gathers rows and block scales by token);
        # block_m 64: the Scatter still copies the rows but triggers right after its wait. Only where the CTA's table
        # fits next to the gate/up's stages without fewer CTAs per SM, else the Scatter places (block_m 64 with 4
        # stages fills the shared memory).
        # The table and eroute's stride are static: sized for the batch's token_bucket (tiles_cap tiles, stride mb).
        place = 0
        if ranked and pdl:
            gu0 = s.gate_up
            if gate_up_4th_stage(gu_claim, bm, gu0.block_k, H):
                gu0 = gu0._replace(stages=4)

            def gu_smem(**kw):
                return GroupedGemm(
                    N1,
                    H,
                    bm,
                    gu0.block_n,
                    gu0.block_k,
                    gu0.stages,
                    True,
                    launch_pdl=pdl,
                    activation=self.activation,
                    **({"unit_a_gs": not self.row_scales} | kw),
                ).smem_bytes()

            b0 = gu_smem(x_sf_gather=sf_gather_mode(m, bm, H))
            b1 = (
                gu_smem(
                    x_sf_gather=2,
                    x_gather=True,
                    route_rows=T,
                    place_tiles=1,
                    max_tiles=g["tiles_cap"],
                    tokens=g["mb"],
                    place_experts=E,
                    unit_a_gs=False,  # (the per-token kernel, _queue_gate_up)
                )
                if bm <= 32
                else gu_smem(
                    x_sf_gather=sf_gather_mode(m, bm, H),
                    place_tiles=2,
                    max_tiles=g["tiles_cap"],
                    place_experts=E,
                )
            )
            if b1 <= self.cta_smem and min(2, self.sm_bytes // b1) == min(
                2, self.sm_bytes // b0
            ):
                place = 1 if bm <= 32 else 2
        ns = place == 1  # no Scatter
        # gate/up activation block scales (sf_gather_mode): the scatter's row-major copy for the long batches' token-
        # major GEMM (its MMA's A scales, rearranged by the scale warps) and for mode 3, packed per tile for mode 0;
        # mode 1: each row's token
        sf_gather = 3 if long else 2 if ns else sf_gather_mode(m, bm, H)
        return _Choices(pdl, ranked, gu_claim, dn_claim, place, ns, sf_gather)

    def _grouped_views(self, pl, b, c):
        """The grouped paths' workspace views for a batch (``_Workspace.view``; the plan's rows in ``pl.rows``)."""
        E, H = self.E, self.H
        g, ws, dummy, fused = b.g, self.ws, self._dummy, self.fused_finalize
        rows, tiles, long = g["rows"], g["tiles"], g["long"]
        i32, f32, bf16 = torch.int32, torch.float32, torch.bfloat16
        pl.rows = rows
        k2 = self.I  # the intermediate's width (the down GEMM's K)
        v = SimpleNamespace(
            count=ws.view("count", (1,), i32),
            experts=ws.view("experts", (tiles,), i32),
            limits=ws.view("limits", (tiles,), i32),
            inverse=ws.view("inverse", (g["routes"],), i32),
            # the tile counters (gate/up, down), each left at zero by its launch
            claims=ws.view("claims", (2,), i32) if c.gu_claim or c.dn_claim else dummy,
            partial=ws.view("partial", ((g["chunks"] + 1) * E,), i32)
            if long
            else dummy,
            eroute=ws.view("eroute", (E * g["mb"],), i32) if c.ns else dummy,
            tbase=ws.view("tbase", (tiles,), i32) if c.place else dummy,
            pairs=ws.view("pairs", (tiles + 1,), i32) if g["dpair"] else dummy,
            ecount=ws.view("ecount", (E,), i32) if c.ranked else dummy,
            # long batches: the gate/up gathers the input rows by token (GroupedGemm x_gather) and ScatterChunks writes
            # each row's token (Int32, in q) instead of copying the FP4 row
            q=ws.view("q", (rows, 4)) if long else ws.view("q", (rows, H // 2)),
            sf=(
                ws.view("sf", (rows, H // 16))
                if c.sf_gather == 3
                else ws.view("sf", (rows,), i32)
                if c.sf_gather == 1
                else dummy
                if c.sf_gather == 2
                else ws.view("sf", (tiles, H // 64, 512))
            ),
            gs=ws.view("gs", (rows,), f32),
            # the NVFP4 intermediate (block scales packed per tile; no row scales)
            q2=ws.view("q2", (rows, k2 // 2)),
            sf2=ws.view("sf2", (tiles, k2 // 64, 512)),
            # row -1 (in front of the rows, never written): zeros, read by ReduceRoutes for skipped routes (inverse -1)
            routed=dummy if fused else ws.view("routed", (rows + 1, H), bf16)[1:],
            # fused finalize: each routed row's token and routing weight (Scatter(fused)), for the down GEMM's adds
            rtok=ws.view("rtok", (rows,), i32) if fused else dummy,
            rwt=ws.view("rwt", (rows,), f32) if fused else dummy,
        )
        v.counts = v.ecount if c.ranked else v.partial
        return v

    def _queue_routing(self, pl, b, c, v):
        """The routing kernel (the caller's ids), the long batches' ScanCounts and the scatter."""
        E, H, T = self.E, self.H, self.topk
        m, base, wdt, g, stream = b.m, b.base, b.wdt, b.g, b.stream
        bm, long, pad, mb, fused, pdl = (
            g["bm"],
            g["long"],
            g["pad"],
            g["mb"],
            self.fused_finalize,
            c.pdl,
        )
        rpc, warps = (ROUTE_TOKENS, 4) if long else (1, 1)
        # ids outside [0, E) are skipped (FlashInfer TRT-LLM's routing skips ids outside the rank's
        # experts): not counted or ranked, inverse -1 (ReduceRoutes reads the zero row in front of routed), their
        # weights copied as 0 (the caller's may be NaN); with the fused finalize and no Scatter, a token whose slot 0
        # is skipped has its output row zeroed here (the gate/up's gather warps zero the rows of slot-0 routes)
        zero = H if c.ns and fused else 0
        op = SelectExperts(
            E,
            T,
            rpc,
            warps,
            partial=long,
            launch_pdl=pdl,
            ranks=c.ranked,
            eroute=mb if c.ns else 0,
            zero_out=zero,
            base=base,
            win_dtype=wdt,
        )
        self._add(
            pl,
            op,
            [
                _arg("ids", I32),
                _arg("wts", F32),
                _ptr(v.count, I32),
                _ptr(v.counts, I32),
                _ptr(v.inverse, I32),
                _ptr(v.eroute, I32, 4),
                _arg("out", BF16),
                _arg("win", wdt),
                I32(m),
            ],
            stream,
        )
        if long:
            op = ScanCounts(E)
            self._add(pl, op, [_ptr(v.partial, I32), I32(g["chunks"])], stream)
        # scatter of the input rows
        if long:  # per chunk (ScatterChunks); max(chunk, E) threads per CTA
            ch = ROUTE_TOKENS * T  # a routing CTA's routes (128 at topk 8)
            op = ScatterChunks(
                H,
                T,
                num_experts=E,
                chunk=ch,
                pad_m=pad,
                tokens_per_cta=max(2, 1 << (-(-max(ch, E) // 64) - 1).bit_length()),
                fused=fused,
                base=base,
                row_scales=self.row_scales,
                pairs=g["dpair"],
            )
        else:
            # about 128 CTAs (up to 8 tokens per CTA)
            op = Scatter(
                H,
                T,
                bm,
                tokens_per_cta=min(8, max(1, m // 128)),
                num_experts=E,
                rowtok=c.sf_gather == 1,
                sf_linear=c.sf_gather == 3,
                fused=fused,
                early=c.place == 2,
                base=base,
                row_scales=self.row_scales,
            )
        if not c.ns:
            # the scatter also takes the routing weights and stores the rows' tokens and weights (fused finalize; the
            # Scatter also zeroes the output rows there, ScatterChunks leaves that to the gate/up: zero_rows)
            self._add(
                pl,
                op,
                [
                    _arg("q", U8),
                    _arg("sf", U8, 4),
                    _arg("gs", F32, 4),
                    _ptr(v.inverse, I32),
                    _ptr(v.q, U8),
                    _ptr(v.sf, U8),
                    _ptr(v.gs, F32),
                    _arg("ids", I32),
                    _ptr(v.counts, I32),
                    _ptr(v.experts, I32),
                    _ptr(v.limits, I32),
                    _ptr(v.count, I32),
                    _arg("wts", F32, 4),
                    _ptr(v.rtok, I32, 4),
                    _ptr(v.rwt, F32, 4),
                    _arg("out", BF16),
                ]
                + (
                    [_ptr(v.pairs, I32, 4)] if long else []
                )  # ScatterChunks: the pair tiles
                + [I32(m)],
                stream,
            )

    def _tokens_arg(self, m):
        """An Int32 [1] holding M, for the long batches' gate/up that zeroes the output's M rows (zero_rows)."""
        if m not in self._tokens:
            self._tokens[m] = torch.full((1,), m, dtype=torch.int32, device=self.dev)
        return self._tokens[m]

    def _queue_gate_up(self, pl, b, c, v):
        """The gate/up GEMM."""
        E, N1, H, T = self.E, self.N1, self.H, self.topk
        m, g, stream = b.m, b.g, b.stream
        bm, long, two, mb, tiles = g["bm"], g["long"], g["two"], g["mb"], g["tiles"]
        fused, pdl, ns, place, sf_gather = (
            self.fused_finalize,
            c.pdl,
            c.ns,
            c.place,
            c.sf_gather,
        )
        gu = pl.sched.gate_up
        if two:
            gu = gu._replace(
                stages=5
            )  # 2-CTA (half the weight rows per CTA): 5 stages fit
        if gate_up_4th_stage(c.gu_claim, bm, gu.block_k, H):
            # a 4th stage (it fits since the SwiGLU epilogue needs no scratch) where the launch claims its tiles and
            # the K loop has at least 8 stages of K 512 (gate_up_4th_stage)
            gu = gu._replace(stages=4)
        pl.sched = pl.sched._replace(gate_up=gu)
        gu_ev = weight_evict("GU", m, H // 512 if gu.block_k == 512 else 0, long, bm)
        # per-tensor input: the gate/up loads no row scales (unit_a_gs), except without Scatter (route_rows): there the
        # per-token kernel reads unit row scales (self._ones), the one compiled for per-token input. Its epilogue
        # without the row-scale loads made cohere-nmc1.0 / qwen3.6-35b M=8-32 0.5-2% slower (the per-tensor A/B)
        ones_gs = not self.row_scales and ns
        unit_gs = not self.row_scales and not ns
        # long batches with the fused finalize: the gate/up's load warp zeroes the output (ScatterChunks' serial pass
        # did before: dsv3 M=16384 54 instead of 17 us), not its epilogue (where H is small the epilogue limits)
        zero_rows = long and fused
        self._grouped(
            pl,
            GroupedGemm(
                N1,
                H,
                bm,
                gu.block_n,
                gu.block_k,
                gu.stages,
                True,
                launch_pdl=pdl,
                evict_weights=gu_ev,
                claim_tiles=c.gu_claim,
                x_sf_gather=sf_gather,
                two_cta=two,
                x_gather=long or ns,
                activation=self.activation,
                route_rows=T if ns else 0,
                place_tiles=place,
                max_tiles=g["tiles_cap"] if place else 0,
                tokens=mb if ns else 0,
                zero_out=H if ns and fused else 0,
                zero_rows=H if zero_rows else 0,
                place_experts=E if place else 0,
                token_major=long,
                unit_a_gs=unit_gs,
            ),
            # the activation rows: the MoE input's (x_gather; no Scatter) or the scatter's copies; their
            # block scales: the input's (modes 1, 2) or the scatter's copy; row scales likewise
            _arg("q", FP4, 16) if long or ns else _ptr(v.q, FP4, 32),
            _arg("sf", FP8, 16) if sf_gather in (1, 2) else _ptr(v.sf, FP8, 16),
            "w1",
            (_ptr(self._ones, F32) if ones_gs else _arg("gs", F32))
            if ns
            else _ptr(v.gs, F32),
            _ptr(v.q2, U8),
            v.claims[:1],
            tiles * cdiv(N1, gu.block_n),
            stream,
            tok=v.eroute if ns else v.q if long else v.sf if sf_gather == 1 else None,
            osf=v.sf2,
            tb=self._tokens_arg(m) if zero_rows else v.tbase if place else None,
            inv=(_arg("out", I32, 4) if fused else _ptr(v.inverse, I32, 4))
            if ns or zero_rows
            else None,
            clr=v.ecount if place else None,
        )

    def _queue_down(self, pl, b, c, v):
        """The down GEMM and, without the fused finalize, ReduceRoutes.

        The down GEMM's K tile, stages and accumulators follow from its shared memory (GroupedGemm.pick_tiles): two
        CTAs per SM where half the shared memory holds a whole tile's K loop, else one CTA with as many stages as fit
        (K 512 first), and two accumulators where the TMEM allows. Long batches' two-CTA launches keep just a tile's K
        loop in stages (short batches fill half the shared memory). Its activation row scales are all 1 (not read)
        except with the per-token intermediate. With the fused finalize (GroupedGemm fused_out) it takes the rows'
        routing weights as row scales, the rows' tokens and the output, and zeroes the expert counts (no ReduceRoutes);
        the same shared memory as without."""
        E, H, T = self.E, self.H, self.topk
        m, g, stream = b.m, b.g, b.stream
        bm, long, two, tiles = g["bm"], g["long"], g["two"], g["tiles"]
        fused, pdl, ns = self.fused_finalize, c.pdl, c.ns
        k2 = self.I
        # non-gated shapes with 256 not dividing I (e.g. Nemotron 3 Super): the down GEMM's K tiles of 512 / 256, the
        # last one past I filled with zeros by the TMA (GroupedGemm k_tail)
        ktk = {"k_tail": True} if not self.gated and k2 % 256 else {}
        # unfused long batches: the down GEMM as weight-major CTA pairs where its MMAs bound it (two_cta_down)
        wm2 = long and not fused and two_cta_down(m, T, H, k2, E)
        # weight-major (unfused) long batches with at least DOWN_MMA_RPE routes per expert: K tiles of 256 with a K
        # tail (less zero fill past I: Nemotron 3 Super from M=2979) or as CTA pairs with I >= 4096
        k256 = (
            long and m * T >= DOWN_MMA_RPE * E and (bool(ktk) or (wm2 and k2 >= 4096))
        )
        if long and (fused or g["dpair"]):
            # Long batches, fused: token-major (128 tokens x bn weight rows per tile, one CTA per SM, tiles from
            # GroupedGemm.pick_tiles_token_major), each tile row added into its token's output row with one bulk
            # reduce; as CTA pairs over ScatterChunks' pair tiles where down_pairs (else skipping the empty halves of
            # the gate/up pairs' padding). Unfused (down_pairs only): the same CTA pairs, each tile row stored into its
            # routed row with one bulk copy.
            dp = g["dpair"]
            pk = dict(two_cta=True, pair_table=True) if dp else dict(skip_empty=two)
            dbn, dk, dst, dn_acc = GroupedGemm.pick_tiles_token_major(
                H, k2, cta_smem=self.cta_smem, **pk, **ktk
            )
            dn = Tiles(dbn, dk, dst)
            pl.sched = pl.sched._replace(down=dn)
            dn_ev = (
                weight_evict("DN", m) and fused and not dp
            )  # (pairs: the policy follows row blocks, not pair tiles)
            self._grouped(
                pl,
                GroupedGemm(
                    H,
                    k2,
                    bm,
                    dn.block_n,
                    dn.block_k,
                    dn.stages,
                    False,
                    evict_weights=dn_ev,
                    acc_stages=dn_acc,
                    token_major=True,
                    fused_out=fused,
                    unit_a_gs=not fused,
                    **pk,
                    **ktk,
                ),
                _ptr(v.q2, FP4, 32),
                _ptr(v.sf2, FP8, 16),
                "w2",
                _ptr(v.rwt if fused else self._dummy, F32),
                _arg("out", BF16) if fused else _ptr(v.routed, BF16),
                v.claims[1:2],
                # pairs: at most (tiles + E) / 2 pair tiles of two CTAs
                (2 * cdiv(tiles + E, 2) if dp else tiles) * cdiv(H, dn.block_n),
                stream,
                tok=v.rtok if fused else None,
                count=v.pairs if dp else None,
                tb=v.pairs[1:] if dp else None,
            )
            if fused:
                return
        else:
            fkw = (
                dict(fused_out=True, clear=E, unit_a_gs=False)
                if fused
                else dict(unit_a_gs=True)
            )
            if wm2:
                fkw = dict(fkw, two_cta=True)
            dk, dst, dn_acc = GroupedGemm.pick_tiles(
                H,
                k2,
                bm,
                False,
                tile_stages=long,
                sm_smem=self.sm_bytes,
                cta_smem=self.cta_smem,
                claim_tiles=c.dn_claim,
                skip_empty=two,
                launch_pdl=pdl,
                k256=k256,
                **fkw,
                **ktk,
            )
            dn = Tiles(128, dk, dst)
            pl.sched = pl.sched._replace(down=dn)
            dn_l2 = down_l2_tiles(m, H, k2) if pdl else 0
            dn_ev = weight_evict("DN", m)
            nsf = (
                ns and fused
            )  # the fused down GEMM takes each row's token and routing weight through eroute
            self._grouped(
                pl,
                GroupedGemm(
                    H,
                    k2,
                    bm,
                    dn.block_n,
                    dn.block_k,
                    dn.stages,
                    False,
                    launch_pdl=pdl,
                    l2_tiles=dn_l2,
                    evict_weights=dn_ev,
                    claim_tiles=c.dn_claim,
                    skip_empty=two,
                    acc_stages=dn_acc,
                    route_rows=T if nsf else 0,
                    **fkw,
                    **ktk,
                ),
                _ptr(v.q2, FP4, 32),
                _ptr(v.sf2, FP8, 16),
                "w2",
                (_arg("wts", F32) if nsf else _ptr(v.rwt, F32))
                if fused
                else _ptr(self._dummy, F32),
                _arg("out", BF16) if fused else _ptr(v.routed, BF16),
                v.claims[1:2],
                tiles * cdiv(H, dn.block_n),
                stream,
                tok=(v.eroute if nsf else v.rtok) if fused else None,
                clr=v.ecount if fused else None,
                tb=v.tbase if nsf else None,
            )
        if fused:
            return
        rblock, rwarps, rvec = reduce_shape(m, H)
        # (a skipped route has inverse -1, the zero row in front of routed, and weight 0)
        clear = E if c.ranked else 0
        red = ReduceRoutes(
            H, T, rblock, rwarps, inverse=True, launch_pdl=pdl, vec=rvec, clear=clear
        )
        self._add(
            pl,
            red,
            [
                _ptr(v.routed, BF16),
                _arg("wts", F32),
                _arg("out", BF16),
                _ptr(v.inverse, I32),
                _ptr(v.ecount, I32),
                I32(m),
            ],
            stream,
        )

    def _grouped(
        self,
        pl,
        op,
        a,
        sfa,
        w,
        a_gs,
        out,
        claims,
        blocks,
        stream,
        tok=None,
        osf=None,
        clr=None,
        tb=None,
        inv=None,
        count=None,
    ):
        """Compile and queue a grouped GEMM. ``a``, ``sfa``, ``a_gs``, ``out``, ``inv``: pointer arguments (or the
        call's, ``_arg``); ``w``: "w1" or "w2", the call's weights. ``osf``: with act, the output's block scales;
        ``tok``, ``clr``: the rows' tokens (route_rows: eroute; x_gather / x_sf_gather 1: the scatter's row tokens)
        and the counts to zero; ``count``: the Int32 tile count the GEMM reads (default: the rows in use)."""
        d, ws, i32 = self._dummy, self.ws, torch.int32
        # (tmaps: per-CTA TMA descriptor copies of the grouped GEMMs, GroupedGemm l2_256: up to 2 CTAs per SM)
        specs = [
            a,
            sfa,
            _arg(w, FP4, 32),
            _arg(w + "_sf", FP8),
            a_gs,
            _arg(w + "_gs", F32),
            _ptr(ws.view("experts", (1,), i32), I32),
            _ptr(ws.view("count", (1,), i32) if count is None else count, I32),
            _ptr(ws.view("limits", (1,), i32), I32),
            out,
            _ptr(d if osf is None else osf, U8),
            _ptr(ws.view("tmaps", (2 * self.sms * 512,)), U8, 128),
            _ptr(claims, I32, 4),
            _ptr(d if tok is None else tok, I32, 4),
            _ptr(d if clr is None else clr, I32, 4),
            _ptr(d if tb is None else tb, I32, 4),
            _ptr(d, I32, 4) if inv is None else inv,
            I32(pl.rows),
            I32(self.E),
            I32(1),
        ]
        op = self._add(pl, op, specs, stream)
        # the grid: CTAs resident at once, at most 2 per SM by shared memory
        per_sm = self.sm_bytes // op.storage.size_in_bytes()
        grid = min(blocks, self.sms * min(per_sm, 2))
        assert not op.two_cta or grid % 2 == 0, "two_cta: whole CTA pairs"
        pl.launches[-1][1][-1] = I32(grid)
        return grid

    # ----------------------------------------------------------------------------------------------------------- call

    def _weights(self, weights):
        """A call's ``KernelWeights`` (also from a weight view) and their addresses; checked (``weights.check_weights``)
        the first time these tensors come (address, type, shape and strides: a view of the same storage with another
        layout is checked again): once per prepared weight set, not per call."""
        if isinstance(weights, Mapping):  # weights.prepare_view / weight_view
            if weights.get("gated", self.gated) != self.gated:
                raise ValueError(
                    f"a {'gated' if weights['gated'] else 'non-gated'} weight view for {self.activation!r}"
                )
            weights = view_weights(weights)
        waddr = tuple([t.data_ptr() for t in weights])
        wkey = (waddr, tuple([(t.dtype, t.shape, t.stride()) for t in weights]))
        if wkey not in self._wok:
            check_weights(weights, self.E, self.H, self.I, self.gated, self.dev)
            self._wok.add(wkey)
        return weights, waddr

    def __call__(
        self,
        q,
        sf,
        gs,
        weights,
        *,
        topk_ids=None,
        topk_weights=None,
        out=None,
    ):
        """Launch the kernels on the current stream of the object's device (whatever the current device) and return the
        BF16 output [m, H] (``out`` if given, else a new tensor; contiguous, 16-byte aligned). The input, routing and
        output tensors on that device (ValueError otherwise, before any launch); the kernels read them
        contiguous at 16-byte aligned addresses, others are copied first (FlashInfer's packs guarantee neither):

        - the NVFP4 input of m <= max_tokens tokens (m = 0: no launch, an empty output): q [m, H / 2] uint8 (two E2M1
          values per byte, the even element in the low nibble), sf [m, H / 16] E4M3 block scales (linear layout; uint8
          or float8_e4m3fn), gs [m] FP32 row scales (x ~ e2m1 * sf * gs); per-tensor input (``per_token_scale=False``):
          gs None (x ~ e2m1 * sf);
        - ``weights``: ``weights.KernelWeights`` (``weights.prepare_weights``) or our FlashInfer weight view
          (``weights.prepare_view`` / ``weight_view``), prepared once: a call passes their addresses, nothing is
          converted. Checked (``weights.check_weights``, ValueError) on the first call with each set of addresses;
          never copied;
        - the routing: ``topk_ids`` [m, topk] int32 expert ids and ``topk_weights`` [m, topk] FP32, BF16 or
          FP16 (FlashInfer's PackedPrecomputed pack fields; used as given, no normalization; BF16 / FP16 converted to
          FP32 exactly, a separate ``Plan``). A route whose id is outside
          [0, E) is skipped, as FlashInfer's TRT-LLM routing skips ids outside the rank's experts (e.g. -1 for CUDA
          graph padding): it adds nothing and its weight is not read; a token without valid routes gets a zero row.
          A token's ids must name distinct experts, as a top-k selection does (not checked: repeated ids cause no
          out-of-bounds access, but their result is unspecified)."""
        m = q.shape[0]
        H, T = self.H, self.topk
        # every tensor the kernels touch on this object's device, checked before anything is launched (a kernel would
        # read host memory or another GPU's through the addresses; the weights: check_weights in _weights)
        for name, t in (
            ("q", q),
            ("sf", sf),
            ("gs", gs),
            ("topk_ids", topk_ids),
            ("topk_weights", topk_weights),
            ("out", out),
        ):
            if t is not None and t.device != self.dev:
                raise ValueError(
                    f"{name}: on {t.device}, expected {self.dev} (the CuteMoE's device)"
                )
        _check_tensor("q", q, (m, H // 2), (torch.uint8,))
        _check_tensor("sf", sf, (m, H // 16), (torch.uint8, torch.float8_e4m3fn))
        if self.row_scales:
            _require(
                gs is not None, "gs: the per-token row scales (per_token_scale=True)"
            )
            _check_tensor("gs", gs, (m,), (torch.float32,))
            gs = _dense(gs)
        else:
            _require(
                gs is None, "gs: None for per-tensor input (per_token_scale=False)"
            )
            gs = self._dummy  # (a placeholder argument: no kernel reads it)
        q, sf = _dense(q), _dense(sf)
        _require(
            topk_ids is not None and topk_weights is not None,
            "topk_ids and topk_weights",
        )
        _check_tensor("topk_ids", topk_ids, (m, T), (torch.int32,))
        _check_tensor("topk_weights", topk_weights, (m, T), tuple(ROUTE_WEIGHT_TYPES))
        ids, win = _dense(topk_ids), _dense(topk_weights)
        weights, waddr = self._weights(weights)
        if (
            m == 0
        ):  # no tokens (e.g. an expert-parallel rank without routes): nothing to launch
            return (
                torch.empty((0, H), dtype=torch.bfloat16, device=q.device)
                if out is None
                else out
            )
        pl = self.plan(m, win.dtype)
        if out is None:
            out = torch.empty((m, H), dtype=torch.bfloat16, device=q.device)
        else:
            _check_tensor("out", out, (m, H), (torch.bfloat16,))
            _require(
                out.is_contiguous() and out.data_ptr() % 16 == 0,
                "out: contiguous and 16-byte aligned",
            )
        key = (
            q.data_ptr(),
            sf.data_ptr(),
            gs.data_ptr(),
            win.data_ptr(),
            ids.data_ptr(),
            out.data_ptr(),
        ) + waddr  # in _KEY_FIELDS' order
        # this object's device and its current stream, whatever the current device
        with torch.cuda.device(self.dev):
            args = pl.args(
                key, lambda: _Call(q, sf, gs, win, ids, pl.wts, out, *weights)
            )
            stream = cuda.CUstream(torch.cuda.current_stream(self.dev).cuda_stream)
            for (compiled, _), a in zip(pl.launches, args, strict=False):
                compiled(*a, stream)
        return out
