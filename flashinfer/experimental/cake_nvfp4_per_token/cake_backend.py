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

Cake backend: per-token NVFP4 activation quantization and the per-token-alpha
NVFP4 GEMM on SM100 / SM103.

The path is FlashInfer's per-token NVFP4 quantize + GEMM chain::

    fp4, sf, scale = nvfp4_quantize(x, 1 / (448 * 6), per_token_activation=True)
    out            = mm_fp4(fp4, w_fp4.T, sf, w_sf.T, scale * w_scale, out_dtype)

run as two generated Cake programs on the current stream:

* ``quant:...`` -- the per-token quantizer (one CTA per token row, the row
  held in registers).  Every token row is scaled independently
  (``per_token_scale = amax(row) * global_scale_inv``),
  its 16-element blocks get UE4M3 scale bytes in the swizzled 128x4 layout
  (128-row x 4-column tiles, zero padding rows / columns) and the values are
  packed as E2M1.  The recipe is instruction-for-instruction that of the
  CuTe-DSL per-token kernel, so ``fp4``, ``sf`` and ``per_token_scale`` are
  bitwise equal to ``backend="cute-dsl"``; ``out_scale`` multiplies the
  returned ``per_token_scale`` (the packed data is unchanged).
* ``gemm:...`` -- the persistent block-scaled tcgen05 GEMM
  ``out[m, n] = alpha[m] * sum_k deq(a[m, k]) deq(b[n, k])`` with one FP32
  ``alpha`` per token and a single bf16 / fp16 rounding.  ``M <= 32`` runs in
  the swapped orientation (tokens along the MMA N extent, 8 / 16 / 32 tokens
  per tile, optionally cluster split-K, two CTAs per SM or three mainloop
  stages on the rows that exceed or fill one wave); larger ``M`` runs
  128-token tiles on one CTA (a ``cta_group::2`` 256x64 pair on the
  single-token-tile narrow rows) or 256-token tiles on a ``cta_group::2`` CTA
  pair, with the grouped raster and the cluster-launch-control tile scheduler
  on the largest rows; two single-token-tile wide-``N`` rules depend on the SM
  count (128-wide tile on 148 SMs, no L2 promotion on 152 SMs).

The host rules of this module -- CTA width and occupancy of the quantizer,
the untuned GEMM tactic, tile / grid geometry and the
scale-tensor views -- are pure-Python ports of the Cake production launchers
(each function names its source); the generated-program export checks route
and bitwise output parity against those launchers on every validated shape.
Nothing is planned on the host at launch and nothing is allocated, so the
prepared runners are CUDA-graph capturable.  See ``README.md`` in this
package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional

import torch
import tvm_ffi

from .cake_jit import (
    MODULES,
    kernel_module_name,
    load_cake_nvfp4_per_token_module,
    route_available,
)

# ---------------------------------------------------------------------------
# Fixed geometry of the generated programs
# ---------------------------------------------------------------------------

SF_VEC = 16  # elements per E4M3 block scale
ROW_TILE = 128  # rows of one swizzled scale tile
BLOCK_M = 128  # tokens per GEMM CTA in the m orientation (256 per CTA pair)
K_TILE = 256  # mainloop K tile (512 with the deep-K variant)
# Token tiles of the swapped orientation stay inside one 32-row scale group.
N_ORIENTATION_MAX_M = 32
SUPPORTED_THREADS = (64, 128, 256, 512)
BASELINE_THREADS = (128, 256, 512)
# Row counts that launch as one wave of one-row CTAs on both supported parts (148 / 152 SMs).
SINGLE_WAVE_MAX_ROWS = 148
MAX_REG_BLOCKS = 8  # 16-element blocks one quantizer thread holds in registers
WIDE_MAX_M = 4096
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
ARCHES = tuple(sorted(set(SUPPORTED_COMPUTE_CAPABILITIES.values())))
# Streaming multiprocessors of the parts the dispatch was validated on
# (sm_100a: B200; sm_103a: GB300).  The GEMM tactic depends on the SM count;
# :func:`required_kernel_keys` enumerates the registered kernels for these
# counts.
ARCH_SM_COUNT = {"sm_100a": 148, "sm_103a": 152}
# Co-resident clusters per cluster size on each part (``cuOccupancyMaxActiveClusters`` of the split-K
# kernel at one CTA per SM), keyed by SM count.  A split-K launch is ``weight tiles x token tiles``
# clusters and must fit in one wave; the capacity is the GPC placement limit, not ``sm_count // size``.
CLUSTER_CAPACITY_BY_SM_COUNT = {
    148: {2: 74, 3: 45, 4: 33, 5: 26, 6: 22, 8: 15},
    152: {2: 76, 3: 46, 4: 36, 5: 28, 6: 23, 8: 15},
}
SPLITK3_CLUSTER = 3

# Validated problem matrix (kernel keys enumerated by :func:`required_kernel_keys`).
# GEMM families are ``(K, N)`` of the per-token quantize + GEMM chains the backend was
# measured on; rows are the token counts of the validated matrix (tails 17 /
# 130 / 257 included) and the ragged correctness rows.
VALIDATED_GEMM_FAMILIES: tuple[tuple[int, int], ...] = (
    (7168, 2112),
    (7168, 1536),
    (16384, 7168),
    (7168, 18432),
    (18432, 7168),
    (8192, 8192),
    (8192, 28672),
    (28672, 8192),
)
VALIDATED_ROWS: tuple[int, ...] = (1, 8, 32, 128, 512, 2048, 8192, 17, 130, 257)
VALIDATED_RAGGED_ROWS: tuple[int, ...] = (3, 1000, 4097)
VALIDATED_QUANTIZE_K: tuple[int, ...] = (7168, 8192, 16384, 18432, 28672)
# Correctness-only GEMM rows beyond the bf16 no-fold matrix: ragged M on two
# families, fp16 activations and a folded output scale on the first family.
VALIDATED_RAGGED_FAMILIES: tuple[tuple[int, int], ...] = ((7168, 2112), (8192, 8192))
VALIDATED_F16_INPUT_ROWS: tuple[int, ...] = (8, 257, 2048)
VALIDATED_FOLD_ROWS: tuple[int, ...] = (1, 130, 8192)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def padded_rows(m: int) -> int:
    """Rows of the swizzled scale tensor (``M`` rounded up to the 128-row tile)."""
    return (int(m) + ROW_TILE - 1) // ROW_TILE * ROW_TILE


def padded_sf_cols(k: int) -> int:
    """Columns of the swizzled scale tensor (``K / 16`` rounded up to 4)."""
    return (int(k) // SF_VEC + 3) // 4 * 4


def sf_logical_offsets(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """Byte offsets ``[rows, cols]`` of every scale in the swizzled 128x4 layout."""
    r = torch.arange(rows, device=device)[:, None]
    c = torch.arange(cols, device=device)[None, :]
    return (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * cols)
    )


def unswizzle_sf_128x4(sf: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Logical ``[rows, K/16]`` E4M3 scale bytes of a swizzled 128x4 scale tensor."""
    cols = padded_sf_cols(k)
    offsets = sf_logical_offsets(padded_rows(rows), cols, sf.device)
    return sf.reshape(-1)[offsets][:rows, : k // SF_VEC]


# ---------------------------------------------------------------------------
# Quantizer host rules (ports of the Cake production launcher)
# ---------------------------------------------------------------------------


def cta_threads(k: int, m: int) -> int:
    """CTA width of the baseline rule (port of ``cta_threads`` of the Cake quantizer).

    For ``M <= 4096`` the widest width not exceeding the block count; above it
    the narrowest width that keeps at most eight blocks per thread."""
    num_blocks = k // SF_VEC
    threads = BASELINE_THREADS[0]
    if m <= WIDE_MAX_M:
        while threads < BASELINE_THREADS[-1] and threads < num_blocks:
            threads *= 2
        return threads
    while (
        threads < BASELINE_THREADS[-1]
        and (num_blocks + threads - 1) // threads > MAX_REG_BLOCKS
    ):
        threads *= 2
    return threads


def min_blocks_for(k: int, threads: int) -> int:
    """Launch-bounds occupancy target (port of ``min_blocks_for``): as many CTAs per SM
    as the register-resident row allows (24 registers of state plus 8 per block)."""
    blocks_per_thread = (k // SF_VEC + threads - 1) // threads
    regs_needed = 24 + 8 * blocks_per_thread
    return max(1, min(1024 // threads, 65536 // (threads * regs_needed)))


def cta_config(k: int, m: int) -> tuple[int, int]:
    """``(threads, min_blocks)`` of the direct quantizer (port of ``cta_config``).

    Rows with few CTAs (``M < 512``) take the widest CTA with one or two CTAs
    per SM (two above 128 rows so the second wave's loads overlap the first
    wave's reduction); ``K <= 8192`` with ``2 <= M < 512`` prefers 256 threads,
    and ``148 < M < 512`` (more than one wave of one-row CTAs on the 148 / 152-SM
    parts) with at most four blocks per 256-thread lane (``K <= 16384``) the
    256-wide CTA at two per SM; a single wave (``128 < M <= 148``) keeps the
    widest CTA.
    Rows with many CTAs take narrow CTAs holding 4-8 blocks per thread."""
    num_blocks = k // SF_VEC

    def bpt(threads: int) -> int:
        return (num_blocks + threads - 1) // threads

    if m < 512:
        if m > 1 and bpt(256) <= 2:
            return 256, 3
        if m > SINGLE_WAVE_MAX_ROWS and bpt(256) <= 4:
            return 256, 2
        return cta_threads(k, m), (2 if m > 128 else 1)
    if m < 8192:
        threads = 256 if bpt(256) <= MAX_REG_BLOCKS else 512
    else:
        threads = (
            128
            if bpt(128) <= MAX_REG_BLOCKS
            else 256
            if bpt(256) <= MAX_REG_BLOCKS
            else 512
        )
    return threads, min_blocks_for(k, threads)


def quant_kernel_key(
    k: int, threads: int, min_blocks: int, is_bf16: bool, fold: bool
) -> str:
    """Logical kernel key of one quantizer instance."""
    dtype = "bf16" if is_bf16 else "f16"
    suffix = "_fold" if fold else ""
    return f"quant:k{k}_t{threads}_mb{min_blocks}_{dtype}{suffix}"


@dataclass(frozen=True)
class QuantizePlan:
    """Resolved quantizer launch of one ``(M, K, dtype, fold)`` on one architecture."""

    arch: str
    sm_count: int
    M: int
    K: int
    is_bf16: bool
    fold: bool
    threads: int
    min_blocks: int  # launch-bounds occupancy (CTAs per SM)
    grid: int  # one CTA per padded row
    padded_rows: int
    padded_cols: int
    kernel_key: str


def quant_plan(
    m: int, k: int, is_bf16: bool, fold: bool, arch: str, sm_count: int
) -> QuantizePlan:
    """Resolve the quantizer instance and grid exactly like the Cake production launcher
    (``quantize_per_token``): CTA width and occupancy from :func:`cta_config`, one CTA
    per padded row (the CTAs past ``M`` zero the padding rows of the scale tiles)."""
    m, k = int(m), int(k)
    if m < 1:
        raise ValueError("M must be positive")
    if k <= 0 or k % SF_VEC:
        raise ValueError(f"K must be a positive multiple of {SF_VEC}, got {k}")
    threads, min_blocks = cta_config(k, m)
    blocks_per_thread = (k // SF_VEC + threads - 1) // threads
    if blocks_per_thread > MAX_REG_BLOCKS:
        raise ValueError(
            f"K={k} at {threads} threads needs {blocks_per_thread} register blocks "
            f"per thread (max {MAX_REG_BLOCKS})"
        )
    pm, cols = padded_rows(m), padded_sf_cols(k)
    return QuantizePlan(
        arch=arch,
        sm_count=int(sm_count),
        M=m,
        K=k,
        is_bf16=bool(is_bf16),
        fold=bool(fold),
        threads=threads,
        min_blocks=min_blocks,
        grid=pm,
        padded_rows=pm,
        padded_cols=cols,
        kernel_key=quant_kernel_key(k, threads, min_blocks, is_bf16, fold),
    )


# ---------------------------------------------------------------------------
# GEMM host rules (ports of the Cake production launcher)
# ---------------------------------------------------------------------------

_M_BUCKETS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
# (tile_m, tile_n) candidates of the m orientation: 1-CTA 128-token tiles
# (64 / 128 / 192 / 256 weights) and 2-CTA 256-token tiles (cluster (2, 1),
# tcgen05.mma.cta_group::2).  The 128 x 192 tile fills a 152-SM part in one
# wave on wide-N M=128 rows where 256-wide tiles leave a quarter of it idle.
_M_TILE_CANDIDATES = (
    (128, 64),
    (256, 64),
    (128, 128),
    (256, 128),
    (128, 192),
    (128, 256),
    (256, 192),
    (256, 256),
)
L2_PROMO_DEFAULT = "l2_256b"
CLC_MIN_TILES_PER_PAIR = 10
# Weight-tile count below which a single 128-token tile takes the 2-CTA 256x64 pair.
TWO_CTA_SINGLE_TILE_MAX_W_TILES = 40
# SM count from which two persistent CTAs per SM win on the 32-token multi-wave swapped rows.
TWO_CTA_PER_SM_MIN_SMS = 152
# 16-token-tile raster groups when the group's fp4 operands (16 token tiles of A plus every
# weight tile it sweeps) fit the 126 MiB L2 and the group sweeps at least 24 weight tiles.
RASTER16_MIN_W_TILES = 24
RASTER16_L2_BYTES = 126 << 20
# Measured cost of one wave of 2-CTA 256 x tile_n tiles relative to a wave of 256 x 256
# tiles (paired sweeps on B200 and GB300); the tile width is re-picked on multi-wave
# rows when ceil(units / pairs) x cost improves by at least TWO_CTA_TILE_MIN_GAIN.
_TWO_CTA_TILE_WAVE_COST = {128: 0.575, 192: 0.787, 256: 1.0}
TWO_CTA_TILE_MIN_GAIN = 0.015
# Cluster split-K 2 for the unsplit swapped-orientation rows whose 128-row weight-tile
# grid fills at most half the SMs at K >= SPLITK2_MIN_K.
SPLITK2_MIN_K = 16384


def _two_cta_wave_time(m: int, n: int, sm_count: int, tile_n: int) -> float:
    """Waves of the persistent 2-CTA grid x the measured relative cost of one wave."""
    pairs = max(1, sm_count // 2)
    units = ((m + 255) // 256) * ((n + tile_n - 1) // tile_n)
    return -(-units // pairs) * _TWO_CTA_TILE_WAVE_COST[tile_n]


def _two_cta_tile_n(m: int, n: int, sm_count: int, tile_n: int) -> int:
    """Re-pick the 2-CTA tile width from the measured wave model on multi-wave rows
    (port of ``_two_cta_tile_n``)."""
    if tile_n not in _TWO_CTA_TILE_WAVE_COST:
        return tile_n
    pairs = max(1, sm_count // 2)
    if ((m + 255) // 256) * ((n + tile_n - 1) // tile_n) <= pairs:
        return tile_n
    base = _two_cta_wave_time(m, n, sm_count, tile_n)
    best_n, best_t = tile_n, base
    for cand in (256, 192, 128):
        t = _two_cta_wave_time(m, n, sm_count, cand)
        if t < best_t:
            best_n, best_t = cand, t
    return best_n if best_t <= base * (1.0 - TWO_CTA_TILE_MIN_GAIN) else tile_n


def _score_m_tactic(
    m: int, n: int, k: int, sm_count: int, tile_m: int, tile_n: int
) -> float:
    """Tile score of the m orientation (port of ``_score_m_tactic``)."""
    m_tiles = (m + tile_m - 1) // tile_m
    n_tiles = (n + tile_n - 1) // tile_n
    m_eff = m / (m_tiles * tile_m)
    n_eff = n / (n_tiles * tile_n)
    cta_group = 2 if tile_m == 256 else 1
    padded_ctas = m_tiles * cta_group * n_tiles
    num_waves = (padded_ctas + sm_count - 1) // sm_count
    throughput = (tile_n / 256) ** 0.5
    if tile_n > 128:
        if k <= 1024:
            throughput *= 0.50
        elif k <= 2048:
            throughput *= 0.80
    score = m_eff * n_eff * throughput / (num_waves * tile_n)
    if cta_group == 2:
        score *= 1.05
        if m > tile_m:
            score *= 0.99
    return score


def default_tactic(m: int, n: int, k: int, sm_count: int) -> dict[str, Any]:
    """Untuned GEMM tactic (port of ``default_tactic`` of the Cake GEMM launcher).

    ``M <= 32`` (``N % 8 == 0``) takes the swapped orientation with 8 / 16 / 32
    tokens per tile and cluster split-K while the tile grid leaves most SMs idle;
    otherwise the tile the scorer picks for the ``M`` bucket (next power of two):
    1-CTA 128 x {64, 128, 192, 256} or 2-CTA 256 x {64, 128, 192, 256}, the 2-CTA width
    re-picked on multi-wave rows by the measured wave model (``_two_cta_tile_n``).  2-CTA
    tiles use the grouped raster (``raster_group`` 8, or 16 when a 16-tile group's fp4
    operands fit L2 and the group sweeps at least 24 weight tiles) when the weight
    dimension has at most 32 tiles and the grid needs more than one wave, and the
    cluster-launch-control scheduler when the persistent pairs average at least
    :data:`CLC_MIN_TILES_PER_PAIR` tiles.  Unsplit deep-K swapped-orientation rows whose
    weight tiles fill at most half the SMs take cluster split-K 2.  Single-token-tile
    rows (``M <= 128``) override the scorer three times: a 2-CTA 256x64 pair over at
    most :data:`TWO_CTA_SINGLE_TILE_MAX_W_TILES` narrow weight tiles (the pair loads
    the 128 real token rows once), the 128-wide two-wave tile on the 148-SM part when
    more 128-wide weight tiles than SMs exist, and no 256 B L2 promotion on the
    152-SM part when one wave of 128-wide tiles covers the row.  The swapped
    orientation never splits the wide-``N`` rows (the persistent kernel is faster
    unsplit there), unlike the CuTe-DSL rule it otherwise mirrors.
    """
    if 1 <= m <= N_ORIENTATION_MAX_M and n % 8 == 0:
        tile_n = 8 if m <= 8 else 16 if m <= 16 else 32
        n_tiles = (n + 127) // 128
        split_k = 1
        if tile_n <= 16 and n_tiles <= 20:
            split_k = 4
        elif n_tiles <= 20:
            tile_n, split_k = 8, 2
            # Three K slices when every cluster of the launch is co-resident (7168x1536 M = 17:
            # 36 clusters, +6 % on both parts); 48 or more clusters of 3 need a second cluster
            # wave (-27..-30 %) and keep two slices.
            capacity = CLUSTER_CAPACITY_BY_SM_COUNT.get(sm_count, {}).get(
                SPLITK3_CLUSTER
            )
            token_tiles = (m + tile_n - 1) // tile_n
            if (
                capacity is not None
                and n_tiles * token_tiles <= capacity
                and k // K_TILE >= SPLITK3_CLUSTER
            ):
                split_k = SPLITK3_CLUSTER
        # Even K slices for split-K 2 / 4; the three-way split uses the kernel's owner-remainder partition.
        if (
            split_k > 1
            and k % K_TILE == 0
            and (split_k == SPLITK3_CLUSTER or k % (K_TILE * split_k) == 0)
        ):
            return {
                "tile_n": tile_n,
                "deep_k": False,
                "alpha_n": True,
                "a_hint": None,
                "b_hint": None,
                "split_k": split_k,
                "l2_promo": L2_PROMO_DEFAULT,
            }
        deep_k = n_tiles >= 128 and k % 512 == 0
        tactic: dict[str, Any] = {
            "tile_n": tile_n,
            "deep_k": deep_k,
            "alpha_n": True,
            "a_hint": "evict_first",
            "b_hint": None,
            "l2_promo": L2_PROMO_DEFAULT,
        }
        if n_tiles > sm_count:
            # More weight tiles than SMs: the shallow K tile, and two CTAs per SM on the
            # 8-token tile or on the 152-SM part (the 32-token tile on 148 SMs prefers one).
            tactic["deep_k"] = False
            if tile_n == 8 or sm_count >= TWO_CTA_PER_SM_MIN_SMS:
                tactic["blocks_per_sm"] = 2
        elif deep_k:
            # Single-wave deep-K rows run three mainloop stages.
            tactic["num_stages"] = 3
        elif (
            k >= SPLITK2_MIN_K
            and k % 512 == 0
            and 2 * n_tiles <= sm_count
            and (k == SPLITK2_MIN_K or tile_n == 32)
        ):
            # Unsplit deep-K rows that leave more than half the SMs idle: two K halves per
            # weight tile double the streaming CTAs (the 8 / 16-token tiles of the deeper
            # rows lose with the split and stay unsplit).
            tactic["split_k"] = 2
        return tactic
    bucket = m if m <= 0 else min(1 << (m - 1).bit_length(), _M_BUCKETS[-1])
    best = None
    for tile_m, tile_n in _M_TILE_CANDIDATES:
        score = _score_m_tactic(bucket, n, k, sm_count, tile_m, tile_n)
        if best is None or score > best[0]:
            best = (score, tile_m, tile_n)
    assert best is not None
    _, tile_m, tile_n = best
    if tile_m == 256:
        tile_n = _two_cta_tile_n(m, n, sm_count, tile_n)
    tactic = {
        "tile_n": tile_n,
        "deep_k": False,
        "alpha_n": False,
        "a_hint": "evict_last",
        "b_hint": "evict_last",
        "l2_promo": L2_PROMO_DEFAULT,
    }
    if (
        m <= BLOCK_M
        and tile_m == BLOCK_M
        and tile_n == 64
        and (n + 63) // 64 <= TWO_CTA_SINGLE_TILE_MAX_W_TILES
    ):
        # One 128-token tile over a few 64-wide weight tiles: the 2-CTA pair loads the
        # real token rows once (no grouped raster, no CLC on these short rows), as a
        # half-M pair (64 token rows per CTA, cta_group::2 M = 128).
        return {
            **tactic,
            "two_cta": True,
            "a_hint": None,
            "b_hint": "evict_first",
            "half_m": True,
        }
    if (
        m <= BLOCK_M
        and tile_m == BLOCK_M
        and tile_n > 128
        and (n + 127) // 128 > sm_count
    ):
        # One token tile over more 128-wide weight tiles than SMs: on the 148-SM part the
        # 128-wide persistent tile (two waves) beats the scorer's single-wave wide tile.
        if sm_count < TWO_CTA_PER_SM_MIN_SMS:
            tactic["tile_n"] = tile_n = 128
    elif m <= BLOCK_M and tile_m == BLOCK_M and tile_n == 128:
        # One token tile, one wave of 128-wide tiles (both parts): no L2 promotion.
        tactic["l2_promo"] = None
    if tile_m == 256:
        tactic["two_cta"] = True
        pairs = max(1, sm_count // 2)
        w_tiles = (n + tile_n - 1) // tile_n
        # The grouped raster costs 1-2 % on a grid that fits one wave (every tile runs
        # concurrently), so it applies to multi-wave grids only.
        if w_tiles <= 32 and ((m + 255) // 256) * w_tiles > pairs:
            tactic["raster_group"] = 8
            group16_bytes = (16 * 256 + w_tiles * tile_n) * (k // 2)
            if (
                w_tiles >= RASTER16_MIN_W_TILES
                and (m + 255) // 256 > 8
                and group16_bytes <= RASTER16_L2_BYTES
            ):
                tactic["raster_group"] = 16
        if ((m + 255) // 256) * (
            (n + tile_n - 1) // tile_n
        ) >= CLC_MIN_TILES_PER_PAIR * pairs:
            tactic["sched"] = "clc"
    tok_tiles = (m + tile_m - 1) // tile_m
    if tok_tiles <= 2:
        tactic["a_hint"], tactic["b_hint"] = None, "evict_first"
    if _half_m_pair_rows(m, n, tile_n, sm_count, tactic):
        tactic["half_m"] = True
    return tactic


def _half_m_pair_rows(
    m: int, n: int, tile_n: int, sm_count: int, tactic: dict[str, Any]
) -> bool:
    """The 64-wide 2-CTA tiles of the narrow-N rows run as half-M pairs (64 token rows per
    CTA) while the half-M grid fits 1.5 waves of pairs (1.3 waves measured faster alone and
    in the quantize + GEMM chain, 1.6 waves faster alone but slower in the chain, 2.2 waves
    slower); the half-M program has no CLC or grouped-raster variant."""
    if (
        tile_n != 64
        or not tactic.get("two_cta")
        or tactic.get("sched") == "clc"
        or tactic.get("raster_group")
    ):
        return False
    w_tiles = (n + 63) // 64
    pairs = max(1, sm_count // 2)
    return (
        w_tiles <= TWO_CTA_SINGLE_TILE_MAX_W_TILES
        and ((m + 63) // 64) * w_tiles <= pairs + pairs // 2
    )


def sched_is_clc(tactic: dict[str, Any]) -> bool:
    return str(tactic.get("sched", "static")) == "clc"


def gemm_kernel_key(tactic: dict[str, Any], out_f16: bool) -> str:
    """Logical kernel key of one GEMM tactic instance."""
    parts = [("n" if tactic["alpha_n"] else "m") + str(int(tactic["tile_n"]))]
    if tactic.get("deep_k"):
        parts.append("k512")
    if tactic.get("two_cta"):
        parts.append("2cta")
    if tactic.get("half_m"):
        parts.append("hm")
    if tactic.get("stream_k"):
        # The slice count is baked into the program; ``gemm_plan`` resolves it into the tactic.
        parts.append(f"skt{int(tactic['sk_split'])}")
    if int(tactic.get("split_k", 1)) > 1:
        parts.append(f"sk{int(tactic['split_k'])}")
    parts.append("f16" if out_f16 else "bf16")
    for hint, prefix in ((tactic.get("a_hint"), "a"), (tactic.get("b_hint"), "b")):
        if hint:
            parts.append(prefix + hint.split("_")[1][0].upper())
    if tactic.get("l2_promo"):
        parts.append("l2" + tactic["l2_promo"].split("_")[1])
    if tactic.get("raster_group"):
        parts.append(f"g{int(tactic['raster_group'])}")
    if tactic.get("sched", "static") == "clc":
        parts.append("clc")
    if int(tactic.get("amc", 1)) > 1:
        parts.append(f"amc{int(tactic['amc'])}")
    if tactic.get("num_stages") is not None:
        parts.append(f"s{int(tactic['num_stages'])}")
    if int(tactic.get("blocks_per_sm", 1)) > 1:
        parts.append(f"o{int(tactic['blocks_per_sm'])}")
    if int(tactic.get("k_skew", 0)):
        parts.append(f"ks{int(tactic['k_skew'])}")
    if tactic.get("mailbox") is not None:
        parts.append(f"mb{tactic['mailbox']}")
    return "gemm:" + "_".join(parts)


@dataclass(frozen=True)
class GemmPlan:
    """Resolved GEMM launch of one ``(M, N, K, out dtype)`` on one architecture."""

    arch: str
    sm_count: int
    M: int
    N: int
    K: int
    out_f16: bool
    tactic: dict[str, Any] = field(compare=False)
    kernel_key: str = ""
    k_tile: int = K_TILE
    tok_tile: int = BLOCK_M
    w_tile: int = BLOCK_M
    tok_tiles: int = 0
    w_tiles: int = 0
    num_tiles: int = 0
    grid: int = 0
    # Stream-K tail of the static 2-CTA schedule (``stream_k`` tactics): the partial last
    # wave's tiles, the equal K slices each is cut into and the pairs holding them.
    sk_tiles: int = 0
    sk_slices: int = 0
    sk_pairs: int = 0

    @property
    def alpha_n(self) -> bool:
        return bool(self.tactic["alpha_n"])

    @property
    def stream_k(self) -> bool:
        return bool(self.tactic.get("stream_k", False))

    @property
    def sk_workspace_floats(self) -> int:
        """FP32 partials: tail tiles x slices x two CTAs x 128 rows x tile_n columns."""
        return self.sk_tiles * self.sk_slices * 2 * BLOCK_M * int(self.tactic["tile_n"])

    @property
    def sk_flag_words(self) -> int:
        return self.sk_tiles * self.sk_slices * 2

    @property
    def two_cta(self) -> bool:
        return bool(self.tactic.get("two_cta", False))

    @property
    def split_k(self) -> int:
        return int(self.tactic.get("split_k", 1))

    @property
    def sched(self) -> str:
        return str(self.tactic.get("sched", "static"))

    @property
    def amc(self) -> int:
        return int(self.tactic.get("amc", 1))


# Largest slice count of a stream-K tail tile (host launch rule ``sk_split``; mirrors the Cake
# default).  The slice count actually used is bounded by the free pairs per tail tile, the
# tile's output subtiles and its K tiles, and is baked into the program (``skt{S}``).
SK_MAX_SPLIT = 8
EPI_TILE_N = 32


def stream_k_tail(
    num_tiles: int, pairs: int, k_tiles: int, split: int = SK_MAX_SPLIT, max_slices: int = 6
) -> tuple[int, int, int]:
    """``(tail_tiles, slices, pairs_used)`` of the stream-K tail of a static 2-CTA grid (the
    Cake ``stream_k_tail``): the partial last wave's ``R = num_tiles % pairs`` tiles are each
    cut into ``slices = min(split, pairs // R, max_slices, k_tiles)`` equal K slices held by
    ``R * slices`` pairs.  ``slices < 2`` means the tail is not run: ``(0, 0, 0)`` without a
    partial last wave, ``(R, 1, 0)`` when the pairs cannot share it."""
    if split < 1:
        raise ValueError(f"sk_split must be positive, got {split}")
    if num_tiles <= pairs or num_tiles % pairs == 0:
        return 0, 0, 0
    tail = num_tiles % pairs
    slices = min(split, pairs // tail, max_slices, k_tiles)
    if slices < 2:
        return tail, 1, 0
    return tail, slices, tail * slices


def gemm_plan(
    m: int,
    n: int,
    k: int,
    out_f16: bool,
    arch: str,
    sm_count: int,
    *,
    tactic: Optional[dict[str, Any]] = None,
) -> GemmPlan:
    """Resolve tactic, tile geometry and grid exactly like the Cake ``launch_gemm``."""
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1:
        raise ValueError("M and N must be positive")
    if tactic is None:
        tactic = default_tactic(m, n, k, int(sm_count))
    tile_n = int(tactic["tile_n"])
    deep_k = bool(tactic["deep_k"])
    alpha_n = bool(tactic["alpha_n"])
    k_tile = 512 if deep_k else K_TILE
    if k % k_tile:
        raise ValueError(f"K={k} must be a multiple of the K tile {k_tile}")
    if alpha_n and n % 8:
        raise ValueError("the swapped orientation needs N % 8 == 0")
    if alpha_n and m > N_ORIENTATION_MAX_M:
        raise ValueError(
            f"the swapped orientation covers M <= {N_ORIENTATION_MAX_M} tokens"
        )
    two_cta = bool(tactic.get("two_cta", False))
    split_k = int(tactic.get("split_k", 1))
    k_skew = int(tactic.get("k_skew", 0))
    if split_k > 1 and (k // k_tile - k_skew) // split_k < 1:
        raise ValueError(
            f"K={k} leaves no K tile for the split-K peers "
            f"(k_tile {k_tile}, split_k {split_k}, k_skew {k_skew})"
        )
    blocks_per_sm = int(tactic.get("blocks_per_sm", 1))
    if blocks_per_sm != 1 and (two_cta or split_k > 1 or int(tactic.get("amc", 1)) > 1):
        raise ValueError("blocks_per_sm is a knob of the 1-CTA persistent program")
    amc = int(tactic.get("amc", 1))
    if amc not in (1, 2, 4):
        raise ValueError(f"amc (cluster A-multicast width) must be 1, 2 or 4: {amc}")
    if amc > 1 and (two_cta or split_k > 1 or alpha_n):
        raise ValueError("the cluster A-multicast is a 1-CTA m-orientation tactic")
    if amc > 1 and m > BLOCK_M:
        raise ValueError("the cluster A-multicast tactic serves one 128-token tile")
    half_m = bool(tactic.get("half_m", False))
    if half_m and (not two_cta or sched_is_clc(tactic) or tactic.get("raster_group")):
        raise ValueError(
            "half_m is a 2-CTA program knob without CLC scheduler or grouped raster"
        )
    stream_k = bool(tactic.get("stream_k", False))
    if stream_k and (not two_cta or half_m or sched_is_clc(tactic)):
        raise ValueError(
            "stream_k is a knob of the static 2-CTA schedule (no half_m, no CLC scheduler)"
        )
    # 2-CTA pairs cover 256 token rows (128 per CTA), half-M pairs 128 (64 per CTA).
    tok_tile = (
        tile_n
        if alpha_n
        else ((BLOCK_M if half_m else 2 * BLOCK_M) if two_cta else BLOCK_M)
    )
    w_tile = BLOCK_M if alpha_n else tile_n
    tok_tiles = (m + tok_tile - 1) // tok_tile
    w_tiles = (n + w_tile - 1) // w_tile
    num_tiles = tok_tiles * w_tiles
    sched = str(tactic.get("sched", "static"))
    if two_cta:
        grid = (
            2 * num_tiles if sched == "clc" else 2 * min(int(sm_count) // 2, num_tiles)
        )
    elif split_k > 1:
        grid = split_k * num_tiles
    elif amc > 1:
        # Whole clusters of AMC consecutive weight tiles; the padding CTAs of the
        # last cluster repeat the last tile's loads and skip their stores.
        grid = min(int(sm_count) // amc, (num_tiles + amc - 1) // amc) * amc
    else:
        grid = min(int(sm_count) * blocks_per_sm, num_tiles)
    sk_tiles = sk_slices = sk_pairs = 0
    tactic = dict(tactic)
    if stream_k:
        sk_tiles, sk_slices, sk_pairs = stream_k_tail(
            num_tiles,
            int(sm_count) // 2,
            k // k_tile,
            int(tactic.get("sk_split", SK_MAX_SPLIT)),
            int(tactic["tile_n"]) // EPI_TILE_N,
        )
        if sk_slices < 2:
            # No partial last wave, or one the pairs cannot share: the plain static program.
            tactic.pop("stream_k", None)
            tactic.pop("sk_split", None)
            sk_tiles = sk_slices = sk_pairs = 0
        else:
            tactic["sk_split"] = sk_slices
    return GemmPlan(
        arch=arch,
        sm_count=int(sm_count),
        M=m,
        N=n,
        K=k,
        out_f16=bool(out_f16),
        tactic=dict(tactic),
        kernel_key=gemm_kernel_key(tactic, out_f16),
        k_tile=k_tile,
        tok_tile=tok_tile,
        w_tile=w_tile,
        tok_tiles=tok_tiles,
        w_tiles=w_tiles,
        num_tiles=num_tiles,
        grid=grid,
        sk_tiles=sk_tiles,
        sk_slices=sk_slices,
        sk_pairs=sk_pairs,
    )


def validated_problems() -> tuple[tuple[str, int, int, int, bool, bool, bool], ...]:
    """``(kind, K, N, M, x_is_bf16, out_f16, fold)`` rows of the validated matrix
    (``kind`` ``"gemm"`` = quantizer + GEMM chain, ``"quantize"`` = quantizer only)."""
    rows: list[tuple[str, int, int, int, bool, bool, bool]] = []
    for k, n in VALIDATED_GEMM_FAMILIES:
        for m in VALIDATED_ROWS:
            for out_f16 in (False, True):
                rows.append(("gemm", k, n, m, True, out_f16, False))
    for k, n in VALIDATED_RAGGED_FAMILIES:
        for m in VALIDATED_RAGGED_ROWS:
            rows.append(("gemm", k, n, m, True, False, False))
    k, n = VALIDATED_GEMM_FAMILIES[0]
    for m in VALIDATED_F16_INPUT_ROWS:
        rows.append(("gemm", k, n, m, False, False, False))
    for m in VALIDATED_FOLD_ROWS:
        rows.append(("gemm", k, n, m, True, False, True))
    for k in VALIDATED_QUANTIZE_K:
        for m in VALIDATED_ROWS:
            for fold in (False, True):
                rows.append(("quantize", k, 0, m, True, False, fold))
    return tuple(rows)


def required_kernel_keys(arch: str, sm_count: Optional[int] = None) -> tuple[str, ...]:
    """Every logical kernel the validated matrix launches on ``arch`` (``sm_count``
    defaults to :data:`ARCH_SM_COUNT`), quantizer keys first, in first-use order."""
    if sm_count is None:
        sm_count = ARCH_SM_COUNT[arch]
    keys: list[str] = []
    for kind, k, n, m, x_bf16, out_f16, fold in validated_problems():
        plan_keys = [quant_plan(m, k, x_bf16, fold, arch, sm_count).kernel_key]
        if kind == "gemm":
            plan_keys.append(gemm_plan(m, n, k, out_f16, arch, sm_count).kernel_key)
        for key in plan_keys:
            if key not in keys:
                keys.append(key)
    return tuple(keys)


# ---------------------------------------------------------------------------
# Device helpers and launch binding
# ---------------------------------------------------------------------------


def _device_arch(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the cake per-token NVFP4 backend requires compute capability 10.0 or "
            f"10.3 (got {capability[0]}.{capability[1]})"
        )
    return arch


def _sm_count(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def generated_program_available(device: torch.device) -> bool:
    """True when this checkout registers every validated program for ``device``."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        return False
    return bool(MODULES) and route_available(
        arch, required_kernel_keys(arch, _sm_count(device))
    )


def _bind(module_name: str, kwargs: dict[str, Any]) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the argument plan of ``module_name`` and load its entry."""
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} "
                f"({kind}); host binding provides {sorted(kwargs)}"
            )
    module = load_cake_nvfp4_per_token_module(module_name)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


def _scalar_f32(value: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.numel() != 1
        or value.device != device
    ):
        raise ValueError(f"{name} must be a one-element float32 tensor on x's device")
    return value.reshape(1)


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------


class PerTokenQuantizeOutputs(NamedTuple):
    """Caller-owned outputs of the quantizer: ``fp4 [M, K/2]`` uint8, ``sf
    [padded_rows(M), padded_sf_cols(K)]`` uint8 (swizzled 128x4 E4M3 bytes) and
    ``scale [M]`` float32."""

    fp4: torch.Tensor
    sf: torch.Tensor
    scale: torch.Tensor


def allocate_nvfp4_per_token_quantize_outputs(
    m: int, k: int, device: torch.device
) -> PerTokenQuantizeOutputs:
    """Allocate the quantizer outputs for ``x[M, K]`` (no launch)."""
    m, k = int(m), int(k)
    if k % SF_VEC:
        raise ValueError(f"K={k} must be a multiple of {SF_VEC}")
    return PerTokenQuantizeOutputs(
        torch.empty((m, k // 2), dtype=torch.uint8, device=device),
        torch.empty(
            (padded_rows(m), padded_sf_cols(k)), dtype=torch.uint8, device=device
        ),
        torch.empty((m,), dtype=torch.float32, device=device),
    )


@dataclass(frozen=True)
class NVFP4PerTokenQuantizeRunner:
    """The prepared quantizer launch of one ``(x, global_scale_inv, out_scale)``.

    ``launch()`` quantizes ``x`` on the current torch stream into the caller-owned
    outputs with no CUDA allocation and no host synchronisation and returns them;
    it is CUDA-graph capturable (capture belongs to the caller).  The program reads
    ``x`` and the scalars on device at every launch."""

    plan: QuantizePlan
    x: torch.Tensor
    outputs: PerTokenQuantizeOutputs
    launches: tuple[tuple[Callable[..., Any], tuple], ...] = field(repr=False)

    def launch(self) -> PerTokenQuantizeOutputs:
        with tvm_ffi.use_torch_stream():
            for entry, arguments in self.launches:
                entry(*arguments)
        return self.outputs

    __call__ = launch

    @property
    def launch_count(self) -> int:
        return len(self.launches)


def validate_quantize_inputs(
    x: torch.Tensor, outputs: PerTokenQuantizeOutputs
) -> tuple[int, int]:
    """Shape / dtype validation of one quantizer call; returns ``(M, K)``."""
    if x.dim() != 2 or not x.is_contiguous() or x.device.type != "cuda":
        raise ValueError("x must be a contiguous 2-D CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"x must be bf16 or fp16, got {x.dtype}")
    m, k = (int(v) for v in x.shape)
    if k % SF_VEC:
        raise ValueError(f"K={k} must be a multiple of {SF_VEC}")
    fp4, sf, scale = outputs
    if (
        tuple(fp4.shape) != (m, k // 2)
        or fp4.dtype != torch.uint8
        or not fp4.is_contiguous()
    ):
        raise ValueError(f"fp4 must be a contiguous uint8 [{m}, {k // 2}] tensor")
    if (
        sf.dtype != torch.uint8
        or sf.numel() != padded_rows(m) * padded_sf_cols(k)
        or not sf.is_contiguous()
    ):
        raise ValueError(
            f"sf must be a contiguous uint8 tensor of {padded_rows(m)} x "
            f"{padded_sf_cols(k)} bytes (allocate_nvfp4_per_token_quantize_outputs)"
        )
    if (
        tuple(scale.shape) != (m,)
        or scale.dtype != torch.float32
        or not scale.is_contiguous()
    ):
        raise ValueError(f"scale must be a contiguous float32 [{m}] tensor")
    if any(t.device != x.device for t in (fp4, sf, scale)):
        raise ValueError("the outputs must be on x's device")
    return m, k


def prepare_nvfp4_per_token_quantize(
    x: torch.Tensor,
    global_scale_inv: torch.Tensor,
    outputs: PerTokenQuantizeOutputs,
    *,
    out_scale: Optional[torch.Tensor] = None,
) -> NVFP4PerTokenQuantizeRunner:
    """Validate the binding, select the quantizer instance and bind the launch.

    ``global_scale_inv`` is the one-element float32 inverse base scale
    (typically ``1 / (448 * 6)``); ``out_scale`` an optional one-element
    float32 tensor the returned per-token scale is multiplied by (the packed
    data is unchanged).  The JIT module of the instance is built and loaded
    here, so prepare outside CUDA Graph capture."""
    m, k = validate_quantize_inputs(x, outputs)
    device = x.device
    arch = _device_arch(device)
    gs_inv = _scalar_f32(global_scale_inv, "global_scale_inv", device)
    fold = out_scale is not None
    scale_arg = _scalar_f32(out_scale, "out_scale", device) if fold else gs_inv
    plan = quant_plan(m, k, x.dtype == torch.bfloat16, fold, arch, _sm_count(device))
    kwargs: dict[str, Any] = dict(
        x=x,
        out_fp4=outputs.fp4,
        out_sf=outputs.sf.view(-1),
        per_token_scale=outputs.scale,
        global_scale_inv=gs_inv,
        out_scale=scale_arg,
        M=m,
        grid=(plan.grid, 1, 1),
    )
    with torch.cuda.device(_device_index(device)):
        launch = _bind(kernel_module_name(arch, plan.kernel_key), kwargs)
    return NVFP4PerTokenQuantizeRunner(plan, x, outputs, (launch,))


def nvfp4_quantize_per_token(
    x: torch.Tensor,
    global_scale_inv: Any,
    *,
    out_scale: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(fp4 [M, K/2] uint8, sf [padded rows, padded cols] uint8, scale [M] f32)``
    of ``x`` in one allocating call (the entry behind
    ``nvfp4_quantize(..., per_token_activation=True, backend="cake")``).

    ``global_scale_inv`` may be a Python float or a one-element tensor.  The
    programs are built with programmatic dependent launch; ``enable_pdl=False``
    is rejected."""
    if enable_pdl is False:
        raise ValueError(
            "the cake per-token quantizer builds its programs with programmatic "
            "dependent launch; enable_pdl=False is not available"
        )
    if x.dim() != 2 or x.device.type != "cuda":
        raise ValueError("x must be a 2-D CUDA tensor")
    x = x.contiguous()
    if isinstance(global_scale_inv, torch.Tensor):
        gs_inv = global_scale_inv.float().reshape(1).contiguous().to(x.device)
    else:
        gs_inv = torch.tensor(
            [float(global_scale_inv)], dtype=torch.float32, device=x.device
        )
    if out_scale is not None:
        out_scale = out_scale.float().reshape(1).contiguous().to(x.device)
    outputs = allocate_nvfp4_per_token_quantize_outputs(
        x.shape[0], x.shape[1], x.device
    )
    runner = prepare_nvfp4_per_token_quantize(x, gs_inv, outputs, out_scale=out_scale)
    fp4, sf, scale = runner.launch()
    return fp4, sf, scale


# ---------------------------------------------------------------------------
# GEMM
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVFP4PerTokenGemmRunner:
    """The prepared GEMM launch of one ``(a, a_sf, b, b_sf, alpha, out)`` binding.

    ``launch()`` runs the block-scaled GEMM on the current torch stream into the
    caller-owned ``out`` with no CUDA allocation and no host synchronisation and
    returns ``out``; it is CUDA-graph capturable.  The program reads every operand
    on device at every launch (re-quantized activations in the same buffers are
    picked up by a captured graph)."""

    plan: GemmPlan
    out: torch.Tensor
    launches: tuple[tuple[Callable[..., Any], tuple], ...] = field(repr=False)

    def launch(self) -> torch.Tensor:
        with tvm_ffi.use_torch_stream():
            for entry, arguments in self.launches:
                entry(*arguments)
        return self.out

    __call__ = launch

    @property
    def launch_count(self) -> int:
        return len(self.launches)


def _flat_u8_scales(sf: torch.Tensor, name: str, rows: int, cols: int) -> torch.Tensor:
    """Flat uint8 view of a swizzled 128x4 scale tensor of ``rows`` padded rows."""
    if sf.dtype == torch.float8_e4m3fn:
        sf = sf.view(torch.uint8)
    if sf.dtype != torch.uint8 or not sf.is_contiguous():
        raise ValueError(f"{name} must be a contiguous float8_e4m3fn / uint8 tensor")
    if sf.numel() != rows * cols:
        raise ValueError(
            f"{name} must hold {rows} x {cols} swizzled scale bytes, got {sf.numel()}"
        )
    return sf.reshape(-1)


_STREAM_K_WORKSPACE: dict[int, torch.Tensor] = {}
_STREAM_K_FLAGS: dict[tuple[int, int], torch.Tensor] = {}


def _stream_k_workspace(
    device: torch.device, floats: int, flags: int, slices: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-device FP32 partial workspace and per-(device, slice count) slot flags of the
    stream-K tail, grown on demand (never shrunk) and shared by the device's stream-K
    launches, which the stream orders.  The flags are free-running per-launch generations
    (every word of a tail tile advances once per launch in which the tile exists, so all
    words of a tile share one history for a fixed slice count); grown arrays restart from
    zero.  No per-launch initialisation and nothing is reset."""
    index = _device_index(device)
    ws = _STREAM_K_WORKSPACE.get(index)
    if ws is None or ws.numel() < max(floats, 1):
        ws = torch.empty(max(floats, 1), dtype=torch.float32, device=device)
        _STREAM_K_WORKSPACE[index] = ws
    fl = _STREAM_K_FLAGS.get((index, slices))
    if fl is None or fl.numel() < max(flags, 1):
        fl = torch.zeros(max(flags, 1), dtype=torch.uint32, device=device)
        _STREAM_K_FLAGS[(index, slices)] = fl
    return ws, fl


def prepare_mm_fp4_per_token(
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    *,
    tactic: Optional[dict[str, Any]] = None,
) -> NVFP4PerTokenGemmRunner:
    """Validate the binding, resolve the tactic and bind the GEMM launch.

    ``a_fp4 [M, K/2]`` and ``b_fp4 [N, K/2]`` are contiguous packed E2M1 (uint8),
    ``a_sf`` / ``b_sf`` their swizzled 128x4 E4M3 block scales
    (``padded_rows(M | N) x padded_sf_cols(K)`` bytes, uint8 or float8_e4m3fn),
    ``alpha [M]`` the FP32 per-token scale and ``out [M, N]`` a contiguous bf16 /
    fp16 tensor.  ``tactic`` overrides :func:`default_tactic`.  The JIT module is
    built and loaded here, so prepare outside CUDA Graph capture."""
    if a_fp4.dim() != 2 or a_fp4.dtype != torch.uint8 or not a_fp4.is_contiguous():
        raise ValueError("a_fp4 must be a contiguous uint8 [M, K/2] tensor")
    if b_fp4.dim() != 2 or b_fp4.dtype != torch.uint8 or not b_fp4.is_contiguous():
        raise ValueError("b_fp4 must be a contiguous uint8 [N, K/2] tensor")
    m, kh = (int(v) for v in a_fp4.shape)
    n, kh_b = (int(v) for v in b_fp4.shape)
    if kh != kh_b:
        raise ValueError(
            f"K mismatch: a_fp4 has {2 * kh} elements per row, b_fp4 {2 * kh_b}"
        )
    k = 2 * kh
    device = a_fp4.device
    if device.type != "cuda":
        raise ValueError("the operands must be CUDA tensors")
    if any(t.device != device for t in (a_sf, b_fp4, b_sf, alpha, out)):
        raise ValueError("every operand must be on one CUDA device")
    if out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"out must be bf16 or fp16, got {out.dtype}")
    if tuple(out.shape) != (m, n) or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous [{m}, {n}] tensor")
    if alpha.dtype != torch.float32 or alpha.numel() != m or not alpha.is_contiguous():
        raise ValueError(
            "alpha must be a contiguous float32 tensor with one value per token"
        )
    cols = padded_sf_cols(k)
    pm, pn = padded_rows(m), padded_rows(n)
    a_flat = _flat_u8_scales(a_sf, "a_sf", pm, cols)
    b_flat = _flat_u8_scales(b_sf, "b_sf", pn, cols)
    arch = _device_arch(device)
    plan = gemm_plan(
        m, n, k, out.dtype == torch.float16, arch, _sm_count(device), tactic=tactic
    )
    # Scale atoms as u32 words: (atoms, K/64 sets, 128 words) for whole-atom operands;
    # narrow token tiles of the swapped orientation address eight-row groups.
    a_view = a_flat.view(torch.uint32).view(pm // ROW_TILE, cols // 4, 128)
    b_view = b_flat.view(torch.uint32).view(pn // ROW_TILE, cols // 4, 128)
    if plan.alpha_n:
        op_a, sf_a, op_b, sf_b = b_fp4, b_view, a_fp4, a_view
        if int(plan.tactic["tile_n"]) < 32:
            sf_b = sf_b.view(pm // ROW_TILE, cols // 4, 4, 32)
    else:
        op_a, sf_a, op_b, sf_b = a_fp4, a_view, b_fp4, b_view
    kwargs: dict[str, Any] = dict(
        A=op_a,
        B=op_b,
        SFA=sf_a,
        SFB=sf_b,
        alpha=alpha,
        out=out,
        M=m,
        N=n,
        K_tiles=k // plan.k_tile,
        tok_tiles=plan.tok_tiles,
        num_tiles=plan.num_tiles,
        grid=(plan.grid, 1, 1),
    )
    if plan.tactic.get("two_cta"):
        # The 2-CTA programs also take the flat u8 scale tensors: the half-M pair gathers
        # the words of its 64-row halves from them with register-path cp.async.
        kwargs["SFA_RAW"], kwargs["SFB_RAW"] = a_flat, b_flat
    if plan.stream_k:
        # Stream-K tail: FP32 partial workspace, the generation flags (also bound as the plain
        # u32 pointer the holders read their own word through) and the tail geometry.
        red_ws, red_flags = _stream_k_workspace(
            device, plan.sk_workspace_floats, plan.sk_flag_words, plan.sk_slices
        )
        kwargs.update(
            red_ws=red_ws,
            red_flags=red_flags,
            red_gen=red_flags,
            sk_tiles=plan.sk_tiles,
            sk_pairs=plan.sk_pairs,
        )
    with torch.cuda.device(_device_index(device)):
        launch = _bind(kernel_module_name(arch, plan.kernel_key), kwargs)
    return NVFP4PerTokenGemmRunner(plan, out, (launch,))


def mm_fp4_per_token(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """The ``mm_fp4(backend="cake")`` entry: FlashInfer's operand convention
    (``a [M, K/2]`` row-major, ``b [K/2, N]`` column-major = ``b_fp4.T``,
    ``a_descale [padded M, K/16]``, ``b_descale`` = ``b_sf.T``) mapped onto
    :func:`prepare_mm_fp4_per_token`; ``out`` is written and returned."""
    b_nk = b.t()
    if not b_nk.is_contiguous():
        raise ValueError(
            "the cake mm_fp4 backend requires b as the column-major [K/2, N] view of a "
            "contiguous [N, K/2] weight (pass b_fp4.T)"
        )
    b_sf = b_descale.t()
    if not b_sf.is_contiguous():
        raise ValueError(
            "the cake mm_fp4 backend requires b_descale as the transposed view of the "
            "contiguous swizzled [padded N, K/16] weight scales (pass b_sf.T)"
        )
    a_u8 = a.view(torch.uint8) if a.dtype != torch.uint8 else a
    b_u8 = b_nk.view(torch.uint8) if b_nk.dtype != torch.uint8 else b_nk
    runner = prepare_mm_fp4_per_token(
        a_u8, a_descale, b_u8, b_sf, alpha.contiguous(), out
    )
    return runner.launch()


# ---------------------------------------------------------------------------
# Quantizer + GEMM chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVFP4PerTokenChainRunner:
    """Quantizer followed by the GEMM: one ``launch()`` runs both generated programs
    on the current stream into the caller-owned workspace and output (no allocation,
    CUDA-graph capturable)."""

    quantize: NVFP4PerTokenQuantizeRunner
    gemm: NVFP4PerTokenGemmRunner

    def launch(self) -> torch.Tensor:
        with tvm_ffi.use_torch_stream():
            for runner in (self.quantize, self.gemm):
                for entry, arguments in runner.launches:
                    entry(*arguments)
        return self.gemm.out

    __call__ = launch

    @property
    def launch_count(self) -> int:
        return self.quantize.launch_count + self.gemm.launch_count

    @property
    def kernels(self) -> tuple[str, ...]:
        return (self.quantize.plan.kernel_key, self.gemm.plan.kernel_key)

    @property
    def grids(self) -> tuple[int, ...]:
        return (self.quantize.plan.grid, self.gemm.plan.grid)


def prepare_nvfp4_per_token_chain(
    x: torch.Tensor,
    global_scale_inv: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf: torch.Tensor,
    out: torch.Tensor,
    workspace: PerTokenQuantizeOutputs,
    *,
    out_scale: Optional[torch.Tensor] = None,
) -> NVFP4PerTokenChainRunner:
    """Bind ``out = mm_fp4(quantize(x), b, ...)`` with the per-token scale (times
    ``out_scale`` when given) as the GEMM alpha; ``workspace`` holds the quantized
    activation (:func:`allocate_nvfp4_per_token_quantize_outputs` for ``x``)."""
    quantize = prepare_nvfp4_per_token_quantize(
        x, global_scale_inv, workspace, out_scale=out_scale
    )
    gemm = prepare_mm_fp4_per_token(
        workspace.fp4, workspace.sf, b_fp4, b_sf, workspace.scale, out
    )
    return NVFP4PerTokenChainRunner(quantize, gemm)
