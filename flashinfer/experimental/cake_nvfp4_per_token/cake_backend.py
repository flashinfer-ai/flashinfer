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
  held in registers; short launches of long bf16 rows split each row over a
  cluster of four CTAs).  Every token row is scaled independently
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

from dataclasses import dataclass, field, replace
from typing import Any, Callable, NamedTuple, Optional

import functools
import math

import torch
import tvm_ffi

from ...utils import get_compute_capability, get_device_sm_count
from .cake_jit import (
    ARG_PLANS,
    KERNELS,
    MODULES,
    kernel_definitions,
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
BASELINE_THREADS = (128, 256, 512)
# Row counts that launch as one wave of one-row CTAs on both supported parts (148 / 152 SMs).
SINGLE_WAVE_MAX_ROWS = 148
MAX_REG_BLOCKS = 8  # 16-element blocks one quantizer thread holds in registers
# Static-K quantizer tier (port of the production ``STATIC_M_LIMIT`` / ``STATIC_K``): contiguous row
# sets of fewer than :data:`STATIC_M_LIMIT` tokens at a row width in :data:`STATIC_K` take an instance
# with the row width compiled in (key suffix ``_k<K>``, ``-DK_STATIC``); every other row set, every
# other width and every row-strided view takes the runtime-geometry instance of the same CTA shape.
# The widths and the limit are the ones where the compiled-in build won or tied the runtime-geometry
# build of the same CTA shape on both parts (5 interleaved rounds per row): +1.0..3.9 % at 2688 and
# 4096, +0.0..2.7 % at 7168; from 8192 rows on the CTA shape changes and the compiled-in build loses
# at 4096 and 7168, and widths of 8192 and above lose at every row count.
STATIC_M_LIMIT = 8192
STATIC_K: tuple[int, ...] = (2688, 4096, 7168)
WIDE_MAX_M = 4096
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
ARCHES = tuple(sorted(set(SUPPORTED_COMPUTE_CAPABILITIES.values())))
# SM count each architecture's GEMM dispatch was measured on (sm_100a: B200, sm_103a: GB300).
# A part with another SM count (GB200: sm_100a with 152 SMs, B300: sm_103a with 148) runs the
# tactic of the measured count where its own tactic selects a kernel the architecture does not
# register (:func:`gemm_plan`).
MEASURED_SM_COUNT = {"sm_100a": 148, "sm_103a": 152}
# Co-resident clusters per cluster size on each part (``cuOccupancyMaxActiveClusters`` of the split-K
# kernel at one CTA per SM), keyed by SM count.  A split-K launch is ``weight tiles x token tiles``
# clusters and must fit in one wave; the capacity is the GPC placement limit, not ``sm_count // size``.
CLUSTER_CAPACITY_BY_SM_COUNT = {
    148: {2: 74, 3: 45, 4: 33, 5: 26, 6: 22, 8: 15},
    152: {2: 76, 3: 46, 4: 36, 5: 28, 6: 23, 8: 15},
}
SPLITK3_CLUSTER = 3
# The 16-token tile with three K slices on the 7168-deep narrow rows whose 8-token
# three-slice grid exceeds the co-resident clusters: (SM count, K, N, FP16 output) ->
# the token counts that take it.  These are the rows where it measured faster than two
# slices of the 8-token tile in two sessions.  A measured M = 17 row stands for
# M = 17..24 and M = 32 for M = 25..32 (the same cluster grids) where both output dtypes
# agree; otherwise only the measured token count takes it.
N16_SPLITK3_M: dict[tuple[int, int, int, bool], frozenset[int]] = {
    (148, 7168, 2112, False): frozenset(range(25, 33)),
    (148, 7168, 2112, True): frozenset((17, *range(25, 33))),
    (148, 7168, 1536, False): frozenset((32,)),
    (152, 7168, 2112, False): frozenset(range(17, 33)),
    (152, 7168, 2112, True): frozenset(range(17, 33)),
    (152, 7168, 1536, False): frozenset(range(25, 33)),
    (152, 7168, 1536, True): frozenset(range(25, 33)),
}


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


LARGE_ROW_MIN_M = 512
STORE_HINT_DEFAULT = "NONE"
STORE_HINT_LARGE_ROW = "L2_EVICT_LAST"
# Cluster row split: a launch of few long bf16 rows spreads every row over a cluster of
# QUANT_CLUSTER_SPLIT CTAs (rank r converts the r-th quarter of the row's blocks; the ranks
# exchange their maxima through distributed shared memory).  Each CTA takes the CTA shape
# :func:`cta_config` picks for a quarter row.  Largest token count that takes the split per
# row width (paired sweeps on B200 and GB300: the split beats one CTA per row by 3-26 % at
# these widths and token counts); every other width, token count and fp16 activations keep
# one CTA per row.
QUANT_CLUSTER_SPLIT = 4
CLUSTER_SPLIT_MAX_M: dict[int, int] = {
    7168: 1,
    8192: 32,
    16384: 32,
    18432: 32,
    28672: 32,
}
# sm_103a, K = 8192 with the out_scale folded: the split up to 17 tokens (at 32 tokens it
# ties one CTA per row on GB300).
CLUSTER_SPLIT_MAX_M_FOLD_SM103_K8192 = 17
# K = 8192, both folds: the token counts that keep one CTA per row on each architecture (in
# some timing states the split reads 2-4 % slower than one CTA per row there); 1 token keeps
# the split on both, and on sm_103a so do 17 tokens and up to the limits above.
CLUSTER_SPLIT_ONE_CTA_M_K8192: dict[str, tuple[int, int]] = {
    "sm_100a": (2, 32),
    "sm_103a": (2, 16),
}
# The quantize + GEMM chain at K = 8192 (the quantizer alone keeps the split): per architecture,
# the token counts that run one CTA per row at its measured SM count (:data:`MEASURED_SM_COUNT`).
# The row CTAs are few, so the GEMM's CTAs reach their entry L2 prefetch while the quantizer
# still runs.  Other SM counts keep :func:`quant_cluster_split`.
CHAIN_ONE_CTA_M_K8192: dict[str, tuple[int, int]] = {
    "sm_103a": (18, 32),
}
# The quantize + GEMM chain of one weight shape: per architecture, the ``(K, N)`` shapes and
# the token-count range whose quantizer runs one CTA per row at the measured SM count (paired
# timings; the chain's GEMM takes its CHAIN_EP_MEASURED_CELLS entry L2 prefetch program
# there).  Other shapes and SM counts keep :func:`chain_cluster_split`'s other rules.
CHAIN_ONE_CTA_MEASURED_CELLS: dict[str, dict[tuple[int, int], tuple[int, int]]] = {
    "sm_100a": {(7168, 18432): (1, 1)},
}
# The quantizer alone (the public per-token quantize entries; the quantize + GEMM chain keeps
# :func:`chain_cluster_split`): per architecture and row width, the token-count ranges
# ``(low, high, C)`` whose bf16 rows run as clusters of ``C`` CTAs instead.  Paired timings on
# B200 and GB300 at the SM counts of :data:`MEASURED_SM_COUNT`; other SM counts keep the chain's
# split.
QUANT_ALONE_CLUSTER_SPLIT: dict[str, dict[int, tuple[tuple[int, int, int], ...]]] = {
    "sm_100a": {
        16384: ((1, 1, 8), (130, 130, 2)),
        18432: ((1, 8, 8),),
        28672: ((1, 8, 8), (130, 130, 4)),
    },
    "sm_103a": {
        16384: ((1, 1, 8),),
        18432: ((1, 8, 8),),
        28672: ((1, 8, 8), (130, 130, 4)),
    },
}
# 16-byte chunks of the last scale tile's padding one padding-cluster CTA zeroes.
CLUSTER_SPLIT_PAD_CHUNKS = 1024
# The quantizer alone (the quantize + GEMM chain keeps one CTA per padded row): per
# architecture and row width, the token-count ranges ``(low, high)`` (within ROW_TILE <
# M <= SINGLE_WAVE_MAX_ROWS) whose bf16 rows take the one-CTA-per-row instance at launch
# bounds 1 that zeroes the last scale tile's padding rows the way the padding clusters do
# (``_pc``: CLUSTER_SPLIT_PAD_CHUNKS 16-byte chunks per CTA), so the grid is M plus a few
# CTAs instead of one CTA per padded row.  At the SM counts of :data:`MEASURED_SM_COUNT`.
QUANT_ALONE_PAD_CHUNKED: dict[str, dict[int, tuple[tuple[int, int], ...]]] = {
    "sm_100a": {
        18432: ((130, 130),),
    },
    "sm_103a": {
        16384: ((130, 130),),
        18432: ((130, 130),),
    },
}


# The quantizer alone (the quantize + GEMM chain keeps the entry read): per architecture and
# row width, the token-count ranges ``(low, high, read)`` whose contiguous folded bf16 rows on the
# large-row instance (256 threads, launch bounds 3, L2 evict-last stores) read ``out_scale``
# at its single use (``"use"``, ``_ru``) instead of with the row's scalar inputs.  At the SM
# counts of :data:`MEASURED_SM_COUNT`.
QUANT_ALONE_OUT_SCALE_READ: dict[str, dict[int, tuple[tuple[int, int, str], ...]]] = {
    "sm_103a": {28672: ((512, 512, "use"),)},
}
OUT_SCALE_READ_SUFFIX = {"use": "_ru"}


def blocks_per_thread_for(k: int, threads: int) -> int:
    """Register capacity of the compiled instance: 16-element blocks per thread of a ``K``-wide row."""
    return (k // SF_VEC + threads - 1) // threads


def large_row_config(k: int, threads: int, min_blocks: int) -> bool:
    """True for the CTA configurations :func:`cta_config` picks for row sets of
    :data:`LARGE_ROW_MIN_M` tokens or more (port of ``large_row_config``)."""
    return (threads, min_blocks) in {
        cta_config(k, LARGE_ROW_MIN_M),
        cta_config(k, 8192),
    }


def default_store_hint(k: int, threads: int, min_blocks: int) -> str:
    """Output-store cache policy of the ``(K, threads, min_blocks)`` configuration (port of
    ``default_store_hint``): L2 evict-last for the large-row configurations, whose 2-130 MB of
    outputs would otherwise be written back into their own read stream."""
    return (
        STORE_HINT_LARGE_ROW
        if large_row_config(k, threads, min_blocks)
        else STORE_HINT_DEFAULT
    )


def quant_kernel_key(
    threads: int,
    blocks_per_thread: int,
    min_blocks: int,
    is_bf16: bool,
    fold: bool,
    store_hint: str = STORE_HINT_DEFAULT,
    k_static: Optional[int] = None,
    cluster_split: int = 1,
    pad_chunked: bool = False,
    out_scale_read: str = "entry",
) -> str:
    """Logical kernel key of one quantizer instance: CTA width, register capacity (blocks per
    thread), launch-bounds occupancy, input dtype, folded scale, the large-row store policy
    (``_sl``), the cluster row split (``_cs<C>``), for the small-row tier the compiled-in
    row width (``_k<K>``) and the chunked padding of the scale tile (``_pc``).  ``K`` and
    the row stride are otherwise kernel arguments, not part of the key."""
    dtype = "bf16" if is_bf16 else "f16"
    return (
        f"quant:t{threads}_b{blocks_per_thread}_mb{min_blocks}_{dtype}"
        + ("_fold" if fold else "")
        + ("_sl" if store_hint == STORE_HINT_LARGE_ROW else "")
        + (f"_cs{cluster_split}" if cluster_split > 1 else "")
        + (f"_k{k_static}" if k_static is not None else "")
        + ("_pc" if pad_chunked else "")
        + OUT_SCALE_READ_SUFFIX.get(out_scale_read, "")
    )


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
    blocks_per_thread: (
        int  # register capacity of the instance (16-element blocks per thread)
    )
    min_blocks: int  # launch-bounds occupancy (CTAs per SM)
    store_hint: str  # output-store cache policy of the instance
    k_static: Optional[
        int
    ]  # compiled-in row width of the small-row tier (None: runtime geometry)
    grid: int  # CTAs: one per padded row, or (M + padding clusters) x cluster_split
    padded_rows: int
    padded_cols: int
    kernel_key: str
    cluster_split: int = 1  # CTAs per token row (one cluster of that many CTAs)
    pad_chunked: bool = False  # padding rows zeroed by chunk CTAs past the M row CTAs
    out_scale_read: str = "entry"  # out_scale load: entry / use
    # The key the planner selected when ``arch`` does not register it and the plan runs
    # the nearest registered program instead (None: the plan runs its own program).
    unregistered_key: Optional[str] = None


def static_k_for(k: int, m: int, x_row_stride: Optional[int] = None) -> Optional[int]:
    """``k`` when ``(M, K)`` takes the instance with the row width compiled in, else ``None``
    (runtime geometry): ``K`` in :data:`STATIC_K`, contiguous rows and fewer than
    :data:`STATIC_M_LIMIT` of them.  Port of the production ``static_k_for``."""
    if k not in STATIC_K or m >= STATIC_M_LIMIT:
        return None
    # A single row has no row stride: ``M == 1`` always takes the static-K instance.
    if m == 1 or x_row_stride is None or x_row_stride == k:
        return int(k)
    return None


def quant_cluster_split(m: int, k: int, is_bf16: bool, fold: bool, arch: str) -> int:
    """CTAs per token row: :data:`QUANT_CLUSTER_SPLIT` for bf16 rows of a width in
    :data:`CLUSTER_SPLIT_MAX_M` up to its token count, else 1 (at K = 8192, also 1 for the
    token counts :data:`CLUSTER_SPLIT_ONE_CTA_M_K8192` gives the architecture)."""
    if not is_bf16:
        return 1
    if int(k) == 8192 and arch in CLUSTER_SPLIT_ONE_CTA_M_K8192:
        low, high = CLUSTER_SPLIT_ONE_CTA_M_K8192[arch]
        if low <= int(m) <= high:
            return 1
    limit = CLUSTER_SPLIT_MAX_M.get(int(k), 0)
    if arch == "sm_103a" and int(k) == 8192 and fold:
        limit = CLUSTER_SPLIT_MAX_M_FOLD_SM103_K8192
    return QUANT_CLUSTER_SPLIT if int(m) <= limit else 1


def chain_cluster_split(
    m: int,
    k: int,
    is_bf16: bool,
    fold: bool,
    arch: str,
    sm_count: int,
    n: Optional[int] = None,
) -> int:
    """CTAs per token row of the quantize + GEMM chain: 1 on the token counts
    :data:`CHAIN_ONE_CTA_M_K8192` gives ``arch`` at K = 8192, and on the token counts
    :data:`CHAIN_ONE_CTA_MEASURED_CELLS` gives the GEMM's ``(K, N)`` (``n``: the weight
    rows; ``None`` = no weight shape), at its measured SM count, else
    :func:`quant_cluster_split`."""
    if MEASURED_SM_COUNT.get(arch) == int(sm_count):
        cell = CHAIN_ONE_CTA_M_K8192.get(arch)
        if cell is not None and int(k) == 8192 and cell[0] <= int(m) <= cell[1]:
            return 1
        if n is not None:
            cell = CHAIN_ONE_CTA_MEASURED_CELLS.get(arch, {}).get((int(k), int(n)))
            if cell is not None and cell[0] <= int(m) <= cell[1]:
                return 1
    return quant_cluster_split(m, k, is_bf16, fold, arch)


def quant_alone_cluster_split(
    m: int, k: int, is_bf16: bool, fold: bool, arch: str, sm_count: int
) -> int:
    """CTAs per token row of the quantizer alone: the range of
    :data:`QUANT_ALONE_CLUSTER_SPLIT` holding ``M`` at ``K`` on ``arch`` at its measured SM
    count, else :func:`quant_cluster_split`."""
    if is_bf16 and MEASURED_SM_COUNT.get(arch) == int(sm_count):
        for low, high, split in QUANT_ALONE_CLUSTER_SPLIT.get(arch, {}).get(int(k), ()):
            if low <= int(m) <= high:
                return split
    return quant_cluster_split(m, k, is_bf16, fold, arch)


def quant_alone_pad_chunked(
    m: int, k: int, is_bf16: bool, k_static: Optional[int], arch: str, sm_count: int
) -> bool:
    """True when the quantizer alone runs ``(M, K)`` on ``arch`` as the one-CTA-per-row
    instance with the chunked padding of the scale tile: bf16 rows of runtime geometry,
    ``M`` in a range :data:`QUANT_ALONE_PAD_CHUNKED` gives ``K`` on ``arch``, at its
    measured SM count."""
    return (
        is_bf16
        and k_static is None
        and MEASURED_SM_COUNT.get(arch) == int(sm_count)
        and any(
            low <= int(m) <= high
            for low, high in QUANT_ALONE_PAD_CHUNKED.get(arch, {}).get(int(k), ())
        )
    )


def quant_alone_out_scale_read(
    m: int,
    k: int,
    is_bf16: bool,
    fold: bool,
    instance: tuple[int, int, int, str, Optional[int]],
    arch: str,
    sm_count: int,
) -> str:
    """Where the quantizer alone reads ``out_scale`` for ``(M, K)`` on ``arch``: the read of the
    range :data:`QUANT_ALONE_OUT_SCALE_READ` gives ``K`` holding ``M`` for folded bf16 rows on the
    large-row instance ``instance`` = ``(threads, blocks_per_thread, min_blocks, store_hint,
    k_static)`` at the measured SM count, else ``"entry"``."""
    if not (
        is_bf16
        and fold
        and instance == (256, 7, 3, STORE_HINT_LARGE_ROW, None)
        and MEASURED_SM_COUNT.get(arch) == int(sm_count)
    ):
        return "entry"
    for low, high, read in QUANT_ALONE_OUT_SCALE_READ.get(arch, {}).get(int(k), ()):
        if low <= int(m) <= high:
            return read
    return "entry"


def quant_plan(
    m: int,
    k: int,
    is_bf16: bool,
    fold: bool,
    arch: str,
    sm_count: int,
    x_row_stride: Optional[int] = None,
    *,
    alone: bool = False,
    n: Optional[int] = None,
) -> QuantizePlan:
    """Resolve the quantizer instance and grid exactly like the Cake production launcher
    (``quantize_per_token``): CTA width and occupancy from :func:`cta_config`, the
    static-K instance for contiguous row sets of fewer than :data:`STATIC_M_LIMIT` tokens
    at a row width in :data:`STATIC_K`, one CTA per padded row (the CTAs past ``M`` zero
    the padding rows of the scale tiles).  ``x_row_stride`` (elements; ``None`` =
    contiguous) decides between the static-K and the runtime-geometry instance, and
    row-strided rows run one CTA per padded row with the entry out_scale read;
    ``alone`` selects the cluster split of the quantizer alone
    (:func:`quant_alone_cluster_split`) instead of the chain's (:func:`chain_cluster_split`,
    of the chain GEMM's weight rows ``n`` when given).
    A row the cluster split serves runs as a cluster of
    ``C`` CTAs, each with the CTA shape of a ``C``-th of the row; the grid is one cluster
    per row, then (``M`` not a multiple of 128) the padding clusters, each CTA zeroing
    :data:`CLUSTER_SPLIT_PAD_CHUNKS` 16-byte chunks of the last scale tile."""
    m, k = int(m), int(k)
    if m < 1:
        raise ValueError("M must be positive")
    if k <= 0 or k % SF_VEC:
        raise ValueError(f"K must be a positive multiple of {SF_VEC}, got {k}")
    threads, min_blocks = cta_config(k, m)
    blocks_per_thread = blocks_per_thread_for(k, threads)
    if blocks_per_thread > MAX_REG_BLOCKS:
        raise ValueError(
            f"K={k} at {threads} threads needs {blocks_per_thread} register blocks "
            f"per thread (max {MAX_REG_BLOCKS})"
        )
    store_hint = default_store_hint(k, threads, min_blocks)
    k_static = static_k_for(k, m, x_row_stride)
    pm, cols = padded_rows(m), padded_sf_cols(k)
    grid = pm
    # The bindings of the cluster-split, pad-chunked and ``_ru`` programs require
    # contiguous rows: row-strided activations run one CTA per padded row.
    contiguous = m == 1 or x_row_stride is None or x_row_stride == k
    if not contiguous:
        split = 1
    elif alone:
        split = quant_alone_cluster_split(m, k, is_bf16, fold, arch, sm_count)
    else:
        split = chain_cluster_split(m, k, is_bf16, fold, arch, sm_count, n)
    if split > 1:
        share = (k // SF_VEC + split - 1) // split * SF_VEC
        threads, min_blocks = cta_config(share, m)
        blocks_per_thread = blocks_per_thread_for(share, threads)
        pad_clusters = (
            -(-8 * cols // (CLUSTER_SPLIT_PAD_CHUNKS * split)) if m % ROW_TILE else 0
        )
        grid = (m + pad_clusters) * split
    pad_chunked = (
        alone
        and split == 1
        and contiguous
        and threads == 512
        and quant_alone_pad_chunked(m, k, is_bf16, k_static, arch, sm_count)
    )
    if pad_chunked:
        min_blocks = 1
        store_hint = default_store_hint(k, threads, min_blocks)
        grid = m + (-(-8 * cols // CLUSTER_SPLIT_PAD_CHUNKS) if m % ROW_TILE else 0)
    out_scale_read = (
        quant_alone_out_scale_read(
            m,
            k,
            is_bf16,
            fold,
            (threads, blocks_per_thread, min_blocks, store_hint, k_static),
            arch,
            sm_count,
        )
        if alone and split == 1 and not pad_chunked and contiguous
        else "entry"
    )
    plan = QuantizePlan(
        arch=arch,
        sm_count=int(sm_count),
        M=m,
        K=k,
        is_bf16=bool(is_bf16),
        fold=bool(fold),
        threads=threads,
        blocks_per_thread=blocks_per_thread,
        min_blocks=min_blocks,
        store_hint=store_hint,
        k_static=k_static,
        grid=grid,
        padded_rows=pm,
        padded_cols=cols,
        kernel_key=quant_kernel_key(
            threads,
            blocks_per_thread,
            min_blocks,
            is_bf16,
            fold,
            store_hint,
            k_static,
            split,
            pad_chunked,
            out_scale_read,
        ),
        cluster_split=split,
        pad_chunked=pad_chunked,
        out_scale_read=out_scale_read,
    )
    if plan.kernel_key in KERNELS.get(arch, {}):
        return plan
    return _registered_quant_plan(plan, contiguous)


def _quant_shape_of_key(
    key: str,
) -> Optional[tuple[int, int, int, bool, bool, str, Optional[int]]]:
    """``(threads, blocks_per_thread, min_blocks, is_bf16, fold, store_hint, k_static)`` of
    a one-CTA-per-row quantizer key of :func:`quant_kernel_key`, else ``None``."""
    head = key.removeprefix("quant:").split("_")
    if not key.startswith("quant:") or len(head) < 4:
        return None
    t, b, mb, dtype, *rest = head
    if not (t[1:].isdigit() and b[1:].isdigit() and mb[2:].isdigit()):
        return None
    k_static = next((int(p[1:]) for p in rest if p[1:].isdigit()), None)
    hint = STORE_HINT_LARGE_ROW if "sl" in rest else STORE_HINT_DEFAULT
    shape = (int(t[1:]), int(b[1:]), int(mb[2:]), dtype == "bf16", "fold" in rest)
    if quant_kernel_key(*shape, hint, k_static) != key:
        return None
    return (*shape, hint, k_static)


def _registered_quant_plan(plan: QuantizePlan, contiguous: bool) -> QuantizePlan:
    """The plan of the registered one-CTA-per-row quantizer program nearest to
    ``plan``'s unregistered instance that runs its rows: the same input dtype and folded
    scale, a CTA holding the row (threads x register blocks x 16 >= K) and a compiled-in
    width only for contiguous rows of that width; the same CTA width first, then the
    closest register capacity, occupancy, store policy, then the key.  The result records
    ``plan``'s key in ``unregistered_key``; ``plan`` itself when no registered program
    runs the rows."""
    best: Optional[tuple[tuple[Any, ...], str, tuple]] = None
    for key in KERNELS.get(plan.arch, {}):
        shape = _quant_shape_of_key(key)
        if shape is None:
            continue
        threads, blocks, min_blocks, is_bf16, fold, hint, k_static = shape
        if (is_bf16, fold) != (plan.is_bf16, plan.fold) or (
            threads * blocks * SF_VEC < plan.K
        ):
            continue
        if k_static is not None and (k_static != plan.K or not contiguous):
            continue
        cost = (
            threads != plan.threads,
            abs(blocks - plan.blocks_per_thread),
            min_blocks != plan.min_blocks,
            hint != plan.store_hint,
            k_static != plan.k_static,
            key,
        )
        if best is None or cost < best[0]:
            best = (cost, key, shape)
    if best is None:
        return plan
    threads, blocks, min_blocks, _, _, hint, k_static = best[2]
    return replace(
        plan,
        threads=threads,
        blocks_per_thread=blocks,
        min_blocks=min_blocks,
        store_hint=hint,
        k_static=k_static,
        grid=plan.padded_rows,
        kernel_key=best[1],
        cluster_split=1,
        unregistered_key=plan.kernel_key,
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
# The architecture whose loader registers the 64-row weight-tile programs (``_w64`` keys):
# other parts keep 128-row weight tiles on those rows, also with 152 SMs.
N8_WIDE_W_ROWS_ARCH = "sm_103a"
# Weight rows per tile of the 8-token swapped-orientation tile on the 152-SM part when
# more 128-row weight tiles than SMs exist: 64-row tiles at one CTA per SM (twice the
# tiles, about three waves) instead of 128-row tiles at two CTAs per SM.
N8_WIDE_W_ROWS = 64
# Width of the 1-CTA m-orientation tile whose output staging overlays the start of the
# mainloop ring (``ring_overlay``): one more mainloop stage (7 -> 8); the overlaid
# stages go back to the next tile's loads once the stores have read them.
RING_OVERLAY_TILE_N = 64
# The architecture whose loader registers the overlay programs of the 128-wide 1-CTA m tile
# (``_ro`` m128 keys): one token tile over more 128-wide weight tiles than SMs takes them
# (8192x28672 M = 128 on B200: 224 tiles on 148 CTAs; a CTA's later tiles reuse the overlaid
# stages once the previous tile's stores have read them).
M128_RING_OVERLAY_ARCH = "sm_100a"
# 16-token-tile raster groups when the group's fp4 operands (16 token tiles of A plus every
# weight tile it sweeps) fit the 126 MiB L2 and the group sweeps at least 24 weight tiles.
RASTER16_MIN_W_TILES = 24
RASTER16_L2_BYTES = 126 << 20
# Measured cost of one wave of 2-CTA 256 x tile_n tiles relative to a wave of 256 x 256
# tiles (paired sweeps on B200 and GB300); the tile width is re-picked on multi-wave
# rows when ceil(units / pairs) x cost improves by at least TWO_CTA_TILE_MIN_GAIN.
_TWO_CTA_TILE_WAVE_COST = {128: 0.575, 192: 0.787, 256: 1.0}
TWO_CTA_TILE_MIN_GAIN = 0.015
# sm_103a programs with the K = 96 mainloop: on sm_103a the loader routes these keys (the
# 256-token 2-CTA pair tiles of width 192 / 256 with evict-last operands, static or CLC
# schedule, no stream-K tail) to programs whose mainloop takes K = 96 MMA steps.  Their
# measured wave cost of the 192-wide tile relative to the 256-wide one is 0.82 (GB300); on
# the 152-SM part the width re-pick uses it when the re-picked plan is such a program, and
# keeps the K = 64 plan otherwise.
K96_ARCH = "sm_103a"
K96_WIDTH_SM_COUNT = 152
K96_KERNEL_KEYS = frozenset(
    f"gemm:m{tile_n}_2cta_{out}_aL_bL_l2256b{suffix}"
    for out in ("bf16", "f16")
    for tile_n, suffix in (
        (192, ""),
        (256, ""),
        (256, "_g8"),
        (256, "_clc"),
        (256, "_g8_clc"),
        (256, "_g16_clc"),
    )
)
_TWO_CTA_TILE_WAVE_COST_K96 = {128: 0.575, 192: 0.82, 256: 1.0}
# The scale-factor ring of the K = 96 2-CTA pair programs with 256-wide weight tiles
# (``sf_ring`` tactics, ``_sfr`` keys): the A / B ring holds six K256 stages of one
# weight panel, the block scales ride a separate five-stage ring with its own producer
# warp, and the epilogue stores each thread's row straight from registers.  The MMA
# sequence, the accumulation order and the per-token epilogue are those of the program
# without the knob, so are its outputs.  On SF_RING_ARCH at its measured SM count, the
# GEMM takes it on every row of the CLC programs (SF_RING_CLC_KEYS) and on the
# token-count ranges of SF_RING_G8_CELLS of the static programs (SF_RING_G8_KEYS): per
# kind, the GEMM alone (``"gemm"``) and the quantize + GEMM chain (``"chain"``), keyed
# by ``(K, N)``; the chain keeps the plan's program on a cell of the GEMM alone only.
SF_RING_ARCH = "sm_103a"
SF_RING_CLC_KEYS = frozenset(
    f"gemm:m256_2cta_{out}_aL_bL_l2256b_g16_clc" for out in ("bf16", "f16")
)
SF_RING_G8_KEYS = frozenset(
    f"gemm:m256_2cta_{out}_aL_bL_l2256b_g8" for out in ("bf16", "f16")
)
SF_RING_G8_CELLS: dict[str, dict[tuple[int, int], tuple[int, int]]] = {
    "gemm": {
        (16384, 7168): (1537, 2048),
        (18432, 7168): (1537, 2048),
        (8192, 8192): (1281, 2048),
        (28672, 8192): (1281, 2304),
        (7168, 1536): (7937, 9728),
    },
    "chain": {
        (16384, 7168): (1537, 2048),
        (18432, 7168): (1537, 2048),
        (8192, 8192): (1281, 2048),
        (28672, 8192): (1281, 2304),
    },
}
# The K = 96 2-CTA pair programs with 256-wide weight tiles that have no ring twin
# (SF_RING_FORM_KEYS: the static schedule without raster groups, the CLC schedule
# without or with raster groups of 8) take a ring program of SF_RING_FORMS on the
# measured cells of SF_RING_FORM_CELLS: per kind, keyed by ``(K, N)``, the token
# ranges (``None``: no upper bound) and the form, ``"g8"`` (static, raster groups
# of 8), ``"g16_clc"`` (CLC, raster groups of 16) or ``"clc"`` (CLC, no raster
# groups).  The tile order and the schedule may change; the MMA sequence of every
# tile and so the outputs do not.
SF_RING_FORM_KEYS = frozenset(
    f"gemm:m256_2cta_{out}_aL_bL_l2256b{suffix}"
    for out in ("bf16", "f16")
    for suffix in ("", "_clc", "_g8_clc")
)
SF_RING_FORMS: dict[str, dict[str, Any]] = {
    "g8": {"raster_group": 8},
    "g16_clc": {"raster_group": 16, "sched": "clc"},
    "clc": {"sched": "clc"},
}
SF_RING_FORM_CELLS: dict[
    str, dict[tuple[int, int], tuple[tuple[int, Optional[int], str], ...]]
] = {
    "gemm": {
        (7168, 18432): ((513, 2560, "g16_clc"),),
        (8192, 28672): ((1537, 2560, "g16_clc"), (2561, 8192, "clc")),
        (28672, 8192): ((5889, 8192, "g8"),),
    },
    "chain": {
        (7168, 18432): ((513, 2560, "g8"),),
        (8192, 28672): ((1537, 2560, "g16_clc"), (2561, 8192, "clc")),
        (28672, 8192): ((5889, 8192, "g8"),),
    },
}
# The cluster-of-8 multicast form of the CLC ring program without raster groups
# (``mc42`` tactics, ``_sfr_mc42`` keys): the launch asks for clusters of 8 CTAs over
# the required pairs and walks the pair tiles in 2 x 2 blocks (two token tiles by two
# weight tiles), so the pairs that share a token or weight tile split its operand and
# scale loads and multicast them.  It needs even token and weight tile counts.  The
# MMA sequence of every tile and so the outputs are those of the ring program.  On
# SF_RING_ARCH at its measured SM count, a K = 96 pair program with 256-wide weight
# tiles (SF_RING_FORM_KEYS, SF_RING_G8_KEYS, SF_RING_CLC_KEYS) takes it on the measured
# cells of SF_RING_MC42_CELLS before the other ring rules: per kind, keyed by
# ``(K, N)``, the token ranges (a row takes it when one of them holds its token count).
SF_RING_MC42_CELLS: dict[str, dict[tuple[int, int], tuple[tuple[int, int], ...]]] = {
    "gemm": {
        (8192, 8192): ((1793, 2048), (5889, 8192)),
        (7168, 18432): ((1793, 8192),),
        (8192, 28672): ((1793, 2048),),
    },
    "chain": {
        (8192, 8192): ((1793, 2048), (5889, 8192)),
        (7168, 18432): ((1793, 8192),),
        (8192, 28672): ((1793, 2048),),
    },
}
# Cluster split-K 2 for the unsplit swapped-orientation rows whose 128-row weight-tile
# grid fills at most half the SMs at K >= SPLITK2_MIN_K.
SPLITK2_MIN_K = 16384
# Entry L2 prefetch of the weight tiles in the quantize + GEMM chain (``ep_stages`` programs,
# ``_ep4`` keys): the GEMM's idle prefetch warp streams the weight operand and its block scales
# of the CTA's first tile, for its first CHAIN_EP_STAGES K tiles, into L2 before the grid
# dependency wait, so they load while the quantizer still runs.  Per architecture, the program
# families (kernel key without the output dtype) and the token-count range where the chain
# (:func:`prepare_nvfp4_per_token_chain`) takes them; the GEMM alone takes them only on the
# cells of GEMM_ALONE_EP_RULES.
CHAIN_EP_STAGES = 4
CHAIN_EP_RULES: dict[str, dict[str, tuple[int, int]]] = {
    "sm_100a": {
        "m128_bF": (128, 128),
        "m128_bF_fold": (129, 136),
        "m64_bF_l2256b_ro": (128, 128),
        "n16_sk3_l2256b": (17, 32),
        "n32_aF_l2256b": (17, 32),
        "n32_k512_aF_l2256b_s3": (17, 32),
        "n32_sk2_aF_l2256b": (17, 17),
        "n8_aF_l2256b": (1, 8),
        "n8_aF_l2256b_o2": (1, 8),
        "n8_k512_aF_l2256b_s3": (8, 8),
        "n8_sk2_aF_l2256b": (1, 32),
        "n8_sk2_l2256b": (1, 32),
        "n8_sk3_l2256b": (17, 17),
        "n8_sk4_l2256b": (1, 8),
    },
    "sm_103a": {
        "m128_bF": (128, 128),
        "m128_bF_fold": (129, 136),
        "m64_bF_l2256b_ro": (128, 128),
        "n16_sk3_l2256b": (17, 32),
        "n32_aF_l2256b": (17, 32),
        "n32_aF_l2256b_o2": (17, 17),
        "n32_k512_aF_l2256b_s3": (17, 32),
        "n32_sk2_aF_l2256b": (17, 17),
        "n8_aF_l2256b": (1, 8),
        "n8_aF_l2256b_w64": (1, 8),
        "n8_k512_aF_l2256b_s3": (8, 8),
        "n8_sk2_aF_l2256b": (1, 8),
        "n8_sk3_l2256b": (17, 17),
        "n8_sk4_l2256b": (1, 8),
    },
}
# Chain cells of the entry L2 prefetch on the SM count of :data:`MEASURED_SM_COUNT` only: per
# architecture, the program family, the token-count range and the ``(K, N)`` shapes (paired
# timings on B200 and GB300); the GEMM alone keeps its program there.
CHAIN_EP_MEASURED_CELLS: dict[
    str, dict[str, tuple[int, int, tuple[tuple[int, int], ...]]]
] = {
    "sm_100a": {
        "m128_bF_ro": (128, 128, ((8192, 28672),)),
        "n8_k512_aF_l2256b_s3": (1, 1, ((7168, 18432),)),
    },
    "sm_103a": {"n32_aF_l2256b_o2": (18, 32, ((8192, 28672),))},
}
# The GEMM alone (``mm_fp4``, and :func:`prepare_mm_fp4_per_token` without a tactic) takes
# the entry L2 prefetch program on these cells too: per architecture, the program families
# and the token-count range, each inside a CHAIN_EP_RULES cell, so the chain and the GEMM
# alone run one program there.
GEMM_ALONE_EP_RULES: dict[str, dict[str, tuple[int, int]]] = {
    "sm_100a": {"m64_bF_l2256b_ro": (128, 128)},
}


def _two_cta_wave_time(
    m: int,
    n: int,
    sm_count: int,
    tile_n: int,
    cost: Optional[dict[int, float]] = None,
) -> float:
    """Waves of the persistent 2-CTA grid x the measured relative cost of one wave."""
    cost = _TWO_CTA_TILE_WAVE_COST if cost is None else cost
    pairs = max(1, sm_count // 2)
    units = ((m + 255) // 256) * ((n + tile_n - 1) // tile_n)
    return -(-units // pairs) * cost[tile_n]


def _two_cta_tile_n(
    m: int,
    n: int,
    sm_count: int,
    tile_n: int,
    cost: Optional[dict[int, float]] = None,
) -> int:
    """Re-pick the 2-CTA tile width from the measured wave model on multi-wave rows
    (port of ``_two_cta_tile_n``); ``cost`` replaces the wave-cost table."""
    if tile_n not in _TWO_CTA_TILE_WAVE_COST:
        return tile_n
    pairs = max(1, sm_count // 2)
    if ((m + 255) // 256) * ((n + tile_n - 1) // tile_n) <= pairs:
        return tile_n
    base = _two_cta_wave_time(m, n, sm_count, tile_n, cost)
    best_n, best_t = tile_n, base
    for cand in (256, 192, 128):
        t = _two_cta_wave_time(m, n, sm_count, cand, cost)
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


def default_tactic(
    m: int,
    n: int,
    k: int,
    sm_count: int,
    *,
    out_f16: bool = False,
    two_cta_wave_cost: Optional[dict[int, float]] = None,
    n8_wide_w_rows: bool = True,
    m128_ring_overlay: bool = False,
) -> dict[str, Any]:
    """Untuned GEMM tactic (port of ``default_tactic`` of the Cake GEMM launcher).

    ``M <= 32`` (``N % 8 == 0``) takes the swapped orientation with 8 / 16 / 32
    tokens per tile and cluster split-K while the tile grid leaves most SMs idle (three
    K slices of the 8-token tile when the cluster grid is co-resident, else of the
    16-token tile on the :data:`N16_SPLITK3_M` rows of the output dtype ``out_f16``);
    otherwise the tile the scorer picks for the ``M`` bucket (next power of two):
    1-CTA 128 x {64, 128, 192, 256} or 2-CTA 256 x {64, 128, 192, 256}, the 2-CTA width
    re-picked on multi-wave rows by the measured wave model (``_two_cta_tile_n``).  2-CTA
    tiles use the grouped raster (``raster_group`` 8, or 16 when a 16-tile group's fp4
    operands fit L2 and the group sweeps at least 24 weight tiles) when the weight
    dimension has at most 32 tiles and the grid needs more than one wave, and the
    cluster-launch-control scheduler when the persistent pairs average at least
    :data:`CLC_MIN_TILES_PER_PAIR` tiles.  Unsplit deep-K swapped-orientation rows whose
    weight tiles fill at most half the SMs take cluster split-K 2; the 8-token tile
    on the 152-SM part with more weight tiles than SMs takes :data:`N8_WIDE_W_ROWS`-row
    weight tiles at one CTA per SM; the :data:`RING_OVERLAY_TILE_N`-wide 1-CTA m tile
    stages its output over the start of the mainloop ring (``ring_overlay``).
    Single-token-tile rows (``M <= 128``) override the scorer three times: a 2-CTA
    256x64 pair over at most :data:`TWO_CTA_SINGLE_TILE_MAX_W_TILES` narrow weight
    tiles (the pair loads the 128 real token rows once), the 128-wide two-wave tile on
    the 148-SM part when more 128-wide weight tiles than SMs exist, and no 256 B L2
    promotion on the 152-SM part when one wave of 128-wide tiles covers the row.  The
    swapped orientation never splits the wide-``N`` rows (the persistent kernel is
    faster unsplit there), unlike the CuTe-DSL rule it otherwise mirrors.
    ``two_cta_wave_cost`` replaces the wave-cost table of the 2-CTA width re-pick
    (:func:`gemm_plan` passes the K = 96 table on sm_103a).  ``n8_wide_w_rows`` allows
    the 64-row weight tiles (:func:`gemm_plan` passes it on :data:`N8_WIDE_W_ROWS_ARCH`
    only).  ``m128_ring_overlay`` allows the overlay on the 128-wide 1-CTA m tile of one
    token tile when its tiles outnumber the SMs (:func:`gemm_plan` passes it on
    :data:`M128_RING_OVERLAY_ARCH` only).
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
            # wave (-27..-30 %) and keep two slices, except on the N16_SPLITK3_M rows,
            # where the 16-token tile halves the token tiles of the three-slice grid.
            capacity = CLUSTER_CAPACITY_BY_SM_COUNT.get(sm_count, {}).get(
                SPLITK3_CLUSTER
            )
            token_tiles = (m + tile_n - 1) // tile_n
            if capacity is not None and k // K_TILE >= SPLITK3_CLUSTER:
                if n_tiles * token_tiles <= capacity:
                    split_k = SPLITK3_CLUSTER
                elif (
                    m in N16_SPLITK3_M.get((sm_count, k, n, bool(out_f16)), ())
                    and n_tiles * ((m + 15) // 16) <= capacity
                ):
                    tile_n, split_k = 16, SPLITK3_CLUSTER
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
            # 8-token tile or on the 152-SM part (the 32-token tile on 148 SMs prefers
            # one); the 8-token tile on the 152-SM part takes 64-row weight tiles at one
            # CTA per SM where their programs are registered (``n8_wide_w_rows``).
            tactic["deep_k"] = False
            if tile_n == 8 and sm_count >= TWO_CTA_PER_SM_MIN_SMS and n8_wide_w_rows:
                tactic["w_rows"] = N8_WIDE_W_ROWS
            elif tile_n == 8 or sm_count >= TWO_CTA_PER_SM_MIN_SMS:
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
        tile_n = _two_cta_tile_n(m, n, sm_count, tile_n, two_cta_wave_cost)
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
    if m <= BLOCK_M and tile_m == BLOCK_M and tile_n == 128:
        # One token tile over 128-wide tiles (one or two waves, both parts): no L2 promotion;
        # one generated program serves every such row.
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
    if tile_m == BLOCK_M and tile_n == RING_OVERLAY_TILE_N:
        # The 64-wide 1-CTA tile: the output staging overlays the mainloop ring, 8
        # stages instead of 7 (18432x7168 and 28672x8192 M = 128: 2-3 % faster on both
        # parts).
        tactic["ring_overlay"] = True
    if (
        m128_ring_overlay
        and m <= BLOCK_M
        and tile_m == BLOCK_M
        and tile_n == 128
        and (n + tile_n - 1) // tile_n > sm_count
    ):
        # One token tile over more 128-wide weight tiles than SMs: 6 mainloop stages instead
        # of 5, the overlaid stages handed back per tile by the epilogue (the program of the
        # single-token-tile 128-wide rows, no L2 promotion).
        tactic["ring_overlay"] = True
    if _half_m_pair_rows(m, n, tile_n, sm_count, tactic):
        tactic["half_m"] = True
    if (
        tactic.get("two_cta")
        and not tactic.get("half_m")
        and tactic.get("sched", "static") == "static"
    ):
        # Stream-K tail of the static 2-CTA schedule: the partial last wave's tiles are cut
        # into equal K slices over consecutive pairs (``stream_k_tail``); ``gemm_plan`` keeps
        # the knob only where at least SK_MIN_SLICES fit (B200 16384x7168 / 18432x7168
        # M=2048 1.089 / 1.098, 8192x28672 M=257 / 512 1.017 / 1.023 vs the plain program;
        # the GB300 perf rows have no shareable tail and run the plain program).
        tactic["stream_k"] = True
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
        if "sk_split" not in tactic:
            raise ValueError(
                "stream_k tactic without a resolved sk_split: build the key from gemm_plan(...).tactic"
            )
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
    if tactic.get("num_stages") is not None:
        parts.append(f"s{int(tactic['num_stages'])}")
    if int(tactic.get("blocks_per_sm", 1)) > 1:
        parts.append(f"o{int(tactic['blocks_per_sm'])}")
    if int(tactic.get("w_rows", BLOCK_M)) != BLOCK_M:
        parts.append(f"w{int(tactic['w_rows'])}")
    if tactic.get("ring_overlay"):
        parts.append("ro")
    if int(tactic.get("ep_stages", 0)):
        parts.append(f"ep{int(tactic['ep_stages'])}")
    if tactic.get("fold"):
        parts.append("fold")
    if tactic.get("b2"):
        parts.append("b2" + str(tactic["b2"]))
    if tactic.get("sf_ring"):
        parts.append("sfr")
    if tactic.get("mc42"):
        parts.append("mc42")
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
    # The key of the default tactic when ``arch`` does not register it and the plan runs
    # the nearest registered program instead (None: the plan runs its own program).
    unregistered_key: Optional[str] = None
    # The key of the registered program the default tactic resolves to when that program's
    # TMA boxes do not fit the operands (:func:`_tma_boxes_fit`) and the plan runs the
    # nearest registered program whose boxes fit instead (None: no such replacement).
    unfit_key: Optional[str] = None

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


# Largest slice count of a stream-K tail tile (host launch rule ``sk_split``; mirrors the Cake
# default).  The slice count actually used is bounded by the free pairs per tail tile, the
# tile's output subtiles and its K tiles, and is baked into the program (``skt{S}``).
SK_MAX_SPLIT = 8
# Fewest slices worth running: two slices keep half a tile plus the exchange on the critical
# path (1.005-1.008 where measured); three or more measured 1.017-1.098 on B200.
SK_MIN_SLICES = 3
EPI_TILE_N = 32


def stream_k_tail(
    num_tiles: int,
    pairs: int,
    k_tiles: int,
    split: int = SK_MAX_SPLIT,
    max_slices: int = 6,
    min_slices: int = SK_MIN_SLICES,
) -> tuple[int, int, int]:
    """``(tail_tiles, slices, pairs_used)`` of the stream-K tail of a static 2-CTA grid (the
    Cake ``stream_k_tail``): the partial last wave's ``R = num_tiles % pairs`` tiles are each
    cut into ``slices = min(split, pairs // R, max_slices, k_tiles)`` equal K slices held by
    ``R * slices`` pairs.  ``pairs_used == 0`` means the tail is not run: ``(0, 0, 0)`` without
    a partial last wave, ``(R, 1, 0)`` when fewer than ``min_slices`` fit."""
    if split < 1:
        raise ValueError(f"sk_split must be positive, got {split}")
    if num_tiles <= pairs or num_tiles % pairs == 0:
        return 0, 0, 0
    tail = num_tiles % pairs
    slices = min(split, pairs // tail, max_slices, k_tiles)
    if slices < max(2, min_slices):
        return tail, 1, 0
    return tail, slices, tail * slices


# Every key default_tactic emits, plus sk_split (the stream-K slice count
# gemm_plan resolves into the tactic and accepts as an override).  An
# explicit tactic= override with any other key is rejected by name.
KNOWN_TACTIC_KEYS = frozenset(
    {
        "a_hint",
        "alpha_n",
        "b2",
        "b_hint",
        "blocks_per_sm",
        "deep_k",
        "ep_stages",
        "fold",
        "half_m",
        "l2_promo",
        "mc42",
        "num_stages",
        "raster_group",
        "ring_overlay",
        "sched",
        "sf_ring",
        "sk_split",
        "split_k",
        "stream_k",
        "tile_n",
        "two_cta",
        "w_rows",
    }
)


# The token-remainder fold (``fold`` tactics, ``_fold`` keys): the 1-CTA 128-wide m
# tile whose remainder token rows (at most 8 past its 128-row tile) ride on its
# weight-tile stages as a swapped-role 128 x 8 MMA instead of a second token tile.
# Without an explicit tactic, :func:`gemm_plan` takes it for 128 < M <= 136 on these
# (arch, K, N) cells at the architecture's :data:`MEASURED_SM_COUNT`.
FOLD_CELLS = frozenset(
    {("sm_100a", 7168, 18432), ("sm_100a", 8192, 28672), ("sm_103a", 7168, 18432)}
)
FOLD_TACTIC = {
    "tile_n": 128,
    "deep_k": False,
    "alpha_n": False,
    "a_hint": None,
    "b_hint": "evict_first",
    "l2_promo": None,
    "fold": True,
}

# The 2-CTA token-remainder fold (``b2="fold"`` tactics, ``_b2fold`` keys): the static
# 2-CTA pair program with 256-wide weight tiles whose remainder token rows (at most 16
# past its 256-row pair tile) ride on its weight-tile stages as a swapped-role
# 256 x 16 pair MMA instead of a second token tile.  Without an explicit tactic,
# :func:`gemm_plan` takes it for 256 < M <= 272 on these (arch, K, N) cells at the
# architecture's :data:`MEASURED_SM_COUNT` when the default plan is that program.
B2_FOLD_CELLS = frozenset({("sm_100a", 7168, 18432), ("sm_103a", 7168, 18432)})
B2_FOLD_BASE_KEYS = frozenset(
    {"gemm:m256_2cta_bf16_bF_l2256b", "gemm:m256_2cta_f16_bF_l2256b"}
)


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
    """Resolve tactic, tile geometry and grid exactly like the Cake ``launch_gemm``.

    Without an explicit ``tactic``, on an SM count other than ``arch``'s
    :data:`MEASURED_SM_COUNT` a default tactic whose kernel ``arch`` does not
    register falls back to the tactic of the measured SM count, planned (grid and
    stream-K tail) for ``sm_count``.  Without an explicit ``tactic``, 128 < M <= 136
    on a :data:`FOLD_CELLS` cell at ``arch``'s measured SM count takes the
    token-remainder fold (:data:`FOLD_TACTIC`), and 256 < M <= 272 on a
    :data:`B2_FOLD_CELLS` cell at that SM count, whose default plan is a
    :data:`B2_FOLD_BASE_KEYS` program, takes its 2-CTA fold (``b2="fold"``)."""
    if (
        tactic is None
        and 128 < int(m) <= 136
        and int(sm_count) == MEASURED_SM_COUNT.get(arch)
        and (arch, int(k), int(n)) in FOLD_CELLS
    ):
        tactic = dict(FOLD_TACTIC)
    plan = _gemm_plan(m, n, k, out_f16, arch, sm_count, tactic=tactic)
    if (
        tactic is None
        and 256 < int(m) <= 272
        and int(sm_count) == MEASURED_SM_COUNT.get(arch)
        and (arch, int(k), int(n)) in B2_FOLD_CELLS
        and plan.kernel_key in B2_FOLD_BASE_KEYS
    ):
        plan = _gemm_plan(
            m, n, k, out_f16, arch, sm_count, tactic={**plan.tactic, "b2": "fold"}
        )
    measured = MEASURED_SM_COUNT.get(arch, int(sm_count))
    if tactic is not None:
        return plan
    if plan.kernel_key in KERNELS.get(arch, {}):
        return _tma_fitting_plan(plan)  # unless the binding refuses the shape
    if int(sm_count) != measured:
        reference = _gemm_plan(m, n, k, out_f16, arch, measured)
        fallback = _gemm_plan(m, n, k, out_f16, arch, sm_count, tactic=reference.tactic)
        if fallback.kernel_key in KERNELS.get(arch, {}):
            return _tma_fitting_plan(fallback)
        # Outside the registered matrix on the measured part too: the program the
        # measured part runs (:func:`_nearest_registered_plan`), when this grid reaches it.
        nearest = _nearest_registered_plan(reference)
        fallback = _gemm_plan(m, n, k, out_f16, arch, sm_count, tactic=nearest.tactic)
        if nearest.unregistered_key and fallback.kernel_key == nearest.kernel_key:
            return replace(fallback, unregistered_key=plan.kernel_key)
    # A default plan whose kernel is still not registered (a shape outside the
    # registered matrix) runs the nearest registered program.
    return _nearest_registered_plan(plan)


def _gemm_plan(
    m: int,
    n: int,
    k: int,
    out_f16: bool,
    arch: str,
    sm_count: int,
    *,
    tactic: Optional[dict[str, Any]] = None,
) -> GemmPlan:
    """The plan of ``tactic`` (default: :func:`default_tactic`) on ``sm_count`` SMs."""
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1:
        raise ValueError("M and N must be positive")
    if tactic is None:
        tactic = default_tactic(
            m,
            n,
            k,
            int(sm_count),
            out_f16=bool(out_f16),
            n8_wide_w_rows=arch == N8_WIDE_W_ROWS_ARCH,
            m128_ring_overlay=arch == M128_RING_OVERLAY_ARCH,
        )
        if arch == K96_ARCH and int(sm_count) == K96_WIDTH_SM_COUNT:
            # The 2-CTA width re-pick of the K = 96 programs (K96_KERNEL_KEYS).
            k96 = default_tactic(
                m,
                n,
                k,
                int(sm_count),
                out_f16=bool(out_f16),
                two_cta_wave_cost=_TWO_CTA_TILE_WAVE_COST_K96,
            )
            if k96 != tactic:
                plan = _gemm_plan(m, n, k, out_f16, arch, sm_count, tactic=k96)
                if plan.kernel_key in K96_KERNEL_KEYS:
                    return plan
    else:
        unknown = sorted(set(tactic) - KNOWN_TACTIC_KEYS)
        if unknown:
            raise ValueError(f"unknown tactic keys: {', '.join(unknown)}")
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
    if split_k > 1 and (k // k_tile) // split_k < 1:
        raise ValueError(
            f"K={k} leaves no K tile for the split-K peers "
            f"(k_tile {k_tile}, split_k {split_k})"
        )
    blocks_per_sm = int(tactic.get("blocks_per_sm", 1))
    if blocks_per_sm != 1 and (two_cta or split_k > 1):
        raise ValueError("blocks_per_sm is a knob of the 1-CTA persistent program")
    w_rows = int(tactic.get("w_rows", BLOCK_M))
    if w_rows not in (N8_WIDE_W_ROWS, BLOCK_M):
        raise ValueError(f"w_rows must be {N8_WIDE_W_ROWS} or {BLOCK_M}, got {w_rows}")
    if w_rows != BLOCK_M and (not alpha_n or two_cta or split_k > 1):
        raise ValueError("w_rows is a knob of the 1-CTA swapped orientation")
    if w_rows != BLOCK_M and blocks_per_sm != 1:
        raise ValueError("64-row weight tiles run at one CTA per SM")
    if tactic.get("ring_overlay") and (
        alpha_n or two_cta or split_k > 1 or blocks_per_sm != 1
    ):
        raise ValueError("ring_overlay is a knob of the 1-CTA m orientation")
    if int(tactic.get("ep_stages", 0)) and (two_cta or (split_k > 1 and not alpha_n)):
        raise ValueError(
            "ep_stages is a knob of the 1-CTA program and the swapped-orientation split-K program"
        )
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
    if tactic.get("sf_ring") and (
        arch != SF_RING_ARCH
        or alpha_n
        or not two_cta
        or tile_n != 256
        or deep_k
        or half_m
        or stream_k
        or split_k > 1
        or tactic.get("a_hint") != "evict_last"
        or tactic.get("b_hint") != "evict_last"
        or tactic.get("l2_promo") != "l2_256b"
        or tactic.get("num_stages") is not None
        or int(tactic.get("ep_stages", 0))
        or (int(tactic.get("raster_group") or 0), sched_is_clc(tactic))
        not in ((8, False), (16, True), (0, True))
    ):
        raise ValueError(
            "sf_ring is a knob of the sm_103a K = 96 2-CTA pair programs with 256-wide"
            " weight tiles and evict-last operands: the static schedule with raster"
            " groups of 8 or the CLC schedule with groups of 16 or without groups"
        )
    if tactic.get("mc42") and (
        not tactic.get("sf_ring")
        or not sched_is_clc(tactic)
        or int(tactic.get("raster_group") or 0)
    ):
        raise ValueError(
            "mc42 is a form of the sf_ring program on the CLC schedule without raster"
            " groups"
        )
    # 2-CTA pairs cover 256 token rows (128 per CTA), half-M pairs 128 (64 per CTA).
    tok_tile = (
        tile_n
        if alpha_n
        else ((BLOCK_M if half_m else 2 * BLOCK_M) if two_cta else BLOCK_M)
    )
    w_tile = w_rows if alpha_n else tile_n
    tok_tiles = (m + tok_tile - 1) // tok_tile
    if tactic.get("fold"):
        if (
            alpha_n
            or two_cta
            or split_k > 1
            or deep_k
            or tile_n != 128
            or blocks_per_sm != 1
            or tactic.get("ring_overlay")
            or int(tactic.get("ep_stages", 0)) not in (0, CHAIN_EP_STAGES)
            or not 128 < m <= 136
        ):
            raise ValueError(
                "fold is a knob of the 1-CTA 128-wide m tile for 128 < M <= 136"
            )
        # One token tile: its 128 rows and the remainder rows the fold carries.
        tok_tiles = 1
    if tactic.get("b2") is not None:
        if (
            tactic["b2"] != "fold"
            or alpha_n
            or not two_cta
            or tile_n != 256
            or deep_k
            or half_m
            or stream_k
            or str(tactic.get("sched", "static")) != "static"
            or tactic.get("raster_group")
            or tactic.get("num_stages") is not None
            or tactic.get("b_hint") != "evict_first"
            or tactic.get("a_hint")
            or tactic.get("l2_promo") != "l2_256b"
            or not 256 < m <= 272
        ):
            raise ValueError(
                'b2="fold" is a knob of the static 2-CTA pair program with 256-wide'
                " weight tiles (b_hint evict_first, l2_promo l2_256b) for 256 < M <= 272"
            )
        # One pair token tile: its 256 rows and the remainder rows the fold carries.
        tok_tiles = 1
    w_tiles = (n + w_tile - 1) // w_tile
    num_tiles = tok_tiles * w_tiles
    if tactic.get("mc42") and (tok_tiles % 2 or w_tiles % 2):
        raise ValueError(
            "mc42 walks the pair tiles in 2 x 2 blocks: it needs even token and"
            f" weight tile counts, got {tok_tiles} x {w_tiles}"
        )
    sched = str(tactic.get("sched", "static"))
    if two_cta:
        grid = (
            2 * num_tiles if sched == "clc" else 2 * min(int(sm_count) // 2, num_tiles)
        )
    elif split_k > 1:
        grid = split_k * num_tiles
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
        if sk_pairs == 0:
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


# Key parts of :func:`gemm_kernel_key` that set one tactic knob, and the count prefixes.
_KEY_FLAGS: dict[str, tuple[str, Any]] = {
    "k512": ("deep_k", True),
    "2cta": ("two_cta", True),
    "hm": ("half_m", True),
    "clc": ("sched", "clc"),
    "ro": ("ring_overlay", True),
    "l2256b": ("l2_promo", L2_PROMO_DEFAULT),
    "aF": ("a_hint", "evict_first"),
    "aL": ("a_hint", "evict_last"),
    "bF": ("b_hint", "evict_first"),
    "bL": ("b_hint", "evict_last"),
}
_KEY_COUNTS = {
    "skt": "sk_split",
    "sk": "split_k",
    "g": "raster_group",
    "s": "num_stages",
    "o": "blocks_per_sm",
    "w": "w_rows",
    "ep": "ep_stages",
}
# Knobs the registered-program fallback compares, with their unset values.
_FALLBACK_KNOBS: dict[str, Any] = {
    "deep_k": False,
    "split_k": 1,
    "half_m": False,
    "sched": "static",
    "raster_group": None,
    "sk_split": None,
    "blocks_per_sm": 1,
    "w_rows": BLOCK_M,
    "num_stages": None,
    "ring_overlay": False,
    "a_hint": None,
    "b_hint": None,
    "l2_promo": None,
}


def _tactic_of_key(key: str) -> Optional[tuple[dict[str, Any], bool]]:
    """``(tactic, out_f16)`` whose :func:`gemm_kernel_key` is ``key``, or ``None`` for a key
    of another form."""
    head, *parts = key.removeprefix("gemm:").split("_")
    if head[:1] not in ("m", "n") or not head[1:].isdigit():
        return None
    tactic: dict[str, Any] = {
        "tile_n": int(head[1:]),
        "deep_k": False,
        "alpha_n": head[0] == "n",
        "a_hint": None,
        "b_hint": None,
        "l2_promo": None,
    }
    out_f16 = None
    for part in parts:
        name = part.rstrip("0123456789")
        if part in ("bf16", "f16"):
            out_f16 = part == "f16"
        elif part in _KEY_FLAGS:
            tactic[_KEY_FLAGS[part][0]] = _KEY_FLAGS[part][1]
        elif name in _KEY_COUNTS and name != part:
            tactic[_KEY_COUNTS[name]] = int(part[len(name) :])
            if name == "skt":
                tactic["stream_k"] = True
        else:
            return None
    if out_f16 is None or gemm_kernel_key(tactic, out_f16) != key:
        return None
    return tactic, out_f16


def _knob_distance(a: dict[str, Any], b: dict[str, Any]) -> int:
    """Differing knobs of two tactics: one each, the raster group one per halving
    (16 -> 8 -> none) and the scheduler two (CLC launches one CTA pair per tile)."""
    distance = 0
    for name, unset in _FALLBACK_KNOBS.items():
        x, y = a.get(name) or unset, b.get(name) or unset
        if name == "raster_group":
            level = [int(g).bit_length() - 3 if g else 0 for g in (x, y)]
            distance += abs(level[0] - level[1])
        elif x != y:
            distance += 2 if name == "sched" else 1
    return distance


def _tma_boxes_fit(plan: GemmPlan) -> bool:
    """True when the host binding of ``plan``'s program accepts its operands: each
    program stores the output through a TMA box 32 weight columns wide (64 in the
    swapped orientation), the 2-CTA programs also check the box's token rows (128;
    half-M pair: 64) and the 2-CTA programs wider than 128 columns load the weight
    block scales in two-atom boxes (``N`` beyond one 128-row scale atom)."""
    m, n, tactic = plan.M, plan.N, plan.tactic
    if n < (64 if tactic["alpha_n"] else 32):
        return False
    if not tactic.get("two_cta"):
        return True
    if m < (BLOCK_M // 2 if tactic.get("half_m") else BLOCK_M):
        return False
    return int(tactic["tile_n"]) <= ROW_TILE or padded_rows(n) >= 2 * ROW_TILE


def _tma_fitting_plan(plan: GemmPlan) -> GemmPlan:
    """``plan`` of a registered program, or, when that program's host binding would
    refuse the shape (a TMA box larger than its operand, :func:`_tma_boxes_fit`), the
    plan of the nearest registered program whose boxes fit, which records ``plan``'s key
    in ``unfit_key``; ``plan`` itself when no registered program fits (an output
    narrower than 32 columns)."""
    if _tma_boxes_fit(plan):
        return plan
    nearest = _nearest_registered_plan(plan)
    if not _tma_boxes_fit(nearest):
        return plan
    return replace(nearest, unregistered_key=None, unfit_key=plan.kernel_key)


def _nearest_registered_plan(plan: GemmPlan) -> GemmPlan:
    """The plan, for ``plan``'s shape and SM count, of the registered program of its
    output dtype nearest to ``plan``'s tactic: one whose TMA boxes fit the operands
    (:func:`_tma_boxes_fit`) first, then the same orientation and CTA group, then the
    closest tile width, then the fewest differing knobs (:func:`_knob_distance`), then
    the key; programs with the entry prefetch are not candidates.  The result records
    ``plan``'s key in ``unregistered_key``; ``plan`` itself when no registered program
    runs the shape."""
    want = plan.tactic
    best: Optional[tuple[tuple[Any, ...], GemmPlan]] = None
    for key in KERNELS.get(plan.arch, {}):
        parsed = _tactic_of_key(key)
        if parsed is None or parsed[1] != plan.out_f16 or parsed[0].get("ep_stages"):
            continue
        try:
            candidate = _gemm_plan(
                plan.M,
                plan.N,
                plan.K,
                plan.out_f16,
                plan.arch,
                plan.sm_count,
                tactic=parsed[0],
            )
        except ValueError:
            continue
        if candidate.kernel_key != key:
            continue  # e.g. a stream-K slice count this grid does not reach
        have = candidate.tactic
        cost = (
            not _tma_boxes_fit(candidate),
            have["alpha_n"] != want["alpha_n"],
            bool(have.get("two_cta")) != bool(want.get("two_cta")),
            abs(math.log2(int(have["tile_n"]) / int(want["tile_n"]))),
            _knob_distance(have, want),
            key,
        )
        if best is None or cost < best[0]:
            best = (cost, candidate)
    if best is None:
        return plan
    return replace(best[1], unregistered_key=plan.kernel_key)


# ---------------------------------------------------------------------------
# Device helpers and launch binding
# ---------------------------------------------------------------------------


def _device_arch(device: torch.device) -> str:
    capability = get_compute_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the cake per-token NVFP4 backend requires compute capability 10.0 or "
            f"10.3 (got {capability[0]}.{capability[1]})"
        )
    return arch


def _sm_count(device: torch.device) -> int:
    return int(get_device_sm_count(device))


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def generated_program_available(device: torch.device) -> bool:
    """True when this checkout registers the generated programs of ``device``'s architecture.

    The tactic of a shape is resolved for the device's actual SM count; a shape
    whose kernel is not registered raises ``NotImplementedError`` naming the key
    at preparation (:func:`kernel_module_name`)."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(get_compute_capability(device))
    if arch is None:
        return False
    return bool(MODULES) and route_available(arch)


def _bind(
    arch: str, kernel_key: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the argument plan of the program behind ``kernel_key`` and load its ``arch`` build."""
    entry, arg_plan = _entry(arch, kernel_key)
    missing = [name for kind, name in arg_plan if kind != "grid" and name not in kwargs]
    if missing:
        raise KeyError(
            f"generated kernel {kernel_key!r} expects arguments {missing}; "
            f"host binding provides {sorted(kwargs)}"
        )
    return entry, _arguments(arg_plan, kwargs)


def _entry(
    arch: str, kernel_key: str
) -> tuple[Callable[..., Any], tuple[tuple[str, str], ...]]:
    """FFI entry and argument plan of the registered program behind ``kernel_key``: its
    module built with the key's compile-line definitions (the module load is cached)."""
    module_name = kernel_module_name(arch, kernel_key)
    record = MODULES[module_name]
    module = load_cake_nvfp4_per_token_module(
        module_name, arch, kernel_definitions(kernel_key)
    )
    plan = ARG_PLANS[record["arg_plan"]]
    return module.run, tuple((kind, name) for kind, name in plan)


def _arguments(arg_plan: tuple[tuple[str, str], ...], kwargs: dict[str, Any]) -> tuple:
    grid = kwargs["grid"]
    grid_by_name = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    return tuple(
        grid_by_name[name] if kind == "grid" else kwargs[name]
        for kind, name in arg_plan
    )


@functools.lru_cache(maxsize=1024)
def _cached_scalar_f32(value: float, device_index: int) -> torch.Tensor:
    """One device-resident float32 scalar per distinct Python value and device (the public
    entries take ``global_scale_inv`` as a float; the kernel reads it from device memory)."""
    return torch.full(
        (1,),
        float(value),
        dtype=torch.float32,
        device=torch.device("cuda", device_index),
    )


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


def row_stride_of(x: torch.Tensor) -> int:
    """Elements between consecutive rows of ``x[M, K]`` (unit stride along the row required)."""
    m, k = (int(v) for v in x.shape)
    if k > 1 and x.stride(1) != 1:
        raise ValueError("x must have unit stride along the row (K dimension)")
    stride = int(x.stride(0)) if m > 1 else k
    if stride < k:
        raise ValueError(f"x rows overlap: row stride {stride} < K={k}")
    return stride


def rows_aligned(x: torch.Tensor) -> bool:
    """Whether every row of ``x[M, K]`` starts on a 32-byte boundary.  The quantizer
    programs load ``x`` 32 bytes at a time from ``x + row * row_stride``."""
    stride_bytes = row_stride_of(x) * x.element_size()
    return x.data_ptr() % 32 == 0 and stride_bytes % 32 == 0


def validate_quantize_inputs(
    x: torch.Tensor, outputs: PerTokenQuantizeOutputs
) -> tuple[int, int]:
    """Shape / dtype validation of one quantizer call; returns ``(M, K)``.  ``x`` may be
    a row-strided view (unit stride along the row) whose rows start on 32-byte
    boundaries."""
    if x.dim() != 2 or x.device.type != "cuda":
        raise ValueError("x must be a 2-D CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"x must be bf16 or fp16, got {x.dtype}")
    m, k = (int(v) for v in x.shape)
    if k <= 0 or k % SF_VEC:
        raise ValueError(f"K={k} must be a positive multiple of {SF_VEC}")
    if not rows_aligned(x):
        raise ValueError(
            "x rows must start on 32-byte boundaries (data_ptr % 32 == 0 and a row "
            "stride that is a multiple of 16 elements); quantize a copy (x.clone())"
        )
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
    """Validate the binding, select the quantizer-alone instance
    (``quant_plan(..., alone=True)``) and bind the launch.

    ``global_scale_inv`` is the one-element float32 inverse base scale
    (typically ``1 / (448 * 6)``); ``out_scale`` an optional one-element
    float32 tensor the returned per-token scale is multiplied by (the packed
    data is unchanged).  The JIT module of the instance is built and loaded
    here, so prepare outside CUDA Graph capture."""
    return _prepare_quantize(
        x, global_scale_inv, outputs, out_scale=out_scale, alone=True
    )


def _prepare_quantize(
    x: torch.Tensor,
    global_scale_inv: torch.Tensor,
    outputs: PerTokenQuantizeOutputs,
    *,
    out_scale: Optional[torch.Tensor],
    alone: bool,
    n: Optional[int] = None,
) -> NVFP4PerTokenQuantizeRunner:
    """:func:`prepare_nvfp4_per_token_quantize` with the cluster split of the quantizer
    alone (``alone``) or of the quantize + GEMM chain (whose GEMM has ``n`` weight rows)."""
    m, k = validate_quantize_inputs(x, outputs)
    device = x.device
    arch = _device_arch(device)
    gs_inv = _scalar_f32(global_scale_inv, "global_scale_inv", device)
    fold = out_scale is not None
    scale_arg = _scalar_f32(out_scale, "out_scale", device) if fold else gs_inv
    plan = quant_plan(
        m,
        k,
        x.dtype == torch.bfloat16,
        fold,
        arch,
        _sm_count(device),
        row_stride_of(x),
        alone=alone,
        n=n,
    )
    kwargs = _quant_kwargs(plan, x, outputs, gs_inv, scale_arg)
    with torch.cuda.device(_device_index(device)):
        launch = _bind(arch, plan.kernel_key, kwargs)
    return NVFP4PerTokenQuantizeRunner(plan, x, outputs, (launch,))


def _quant_kwargs(
    plan: QuantizePlan,
    x: torch.Tensor,
    outputs: PerTokenQuantizeOutputs,
    gs_inv: torch.Tensor,
    scale_arg: torch.Tensor,
) -> dict[str, Any]:
    return dict(
        x=x,
        out_fp4=outputs.fp4,
        out_sf=outputs.sf.view(-1),
        per_token_scale=outputs.scale,
        global_scale_inv=gs_inv,
        out_scale=scale_arg,
        M=plan.M,
        K=plan.K,
        x_row_stride=row_stride_of(x),
        grid=(plan.grid, 1, 1),
    )


@functools.lru_cache(maxsize=4096)
def _quant_prepared(
    m: int, k: int, is_bf16: bool, fold: bool, device_index: int, x_row_stride: int
) -> tuple[QuantizePlan, Callable[..., Any], tuple[tuple[str, str], ...]]:
    """Plan, loaded FFI entry and argument plan of the quantizer launch of one shape on one device
    (``x_row_stride`` selects the static-K or the runtime-geometry instance and keeps
    row-strided rows on one CTA per padded row with the entry out_scale read)."""
    device = torch.device("cuda", device_index)
    arch = _device_arch(device)
    plan = quant_plan(
        m, k, is_bf16, fold, arch, _sm_count(device), x_row_stride, alone=True
    )
    with torch.cuda.device(device_index):
        entry, arg_plan = _entry(arch, plan.kernel_key)
    return plan, entry, arg_plan


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
    if x.shape[1] > 1 and x.stride(1) != 1:
        # Only a non-unit stride along the row needs a copy; row-strided views are read in place.
        x = x.contiguous()
    device = x.device
    device_index = _device_index(device)
    if isinstance(global_scale_inv, torch.Tensor):
        gs_inv = global_scale_inv
        if gs_inv.dtype != torch.float32 or gs_inv.device != device:
            gs_inv = gs_inv.to(device=device, dtype=torch.float32)
        gs_inv = gs_inv.reshape(1)
    else:
        gs_inv = _cached_scalar_f32(float(global_scale_inv), device_index)
    if out_scale is not None:
        if out_scale.dtype != torch.float32 or out_scale.device != device:
            out_scale = out_scale.to(device=device, dtype=torch.float32)
        out_scale = out_scale.reshape(1)
    m, k = (int(v) for v in x.shape)
    if k <= 0 or k % SF_VEC:
        raise ValueError(f"K={k} must be a positive multiple of {SF_VEC}")
    if not rows_aligned(x):
        # Rows that start off 32-byte boundaries are quantized from an aligned copy.
        x = x.clone(memory_format=torch.contiguous_format)
    outputs = allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    fold = out_scale is not None
    plan, entry, arg_plan = _quant_prepared(
        m, k, x.dtype == torch.bfloat16, fold, device_index, row_stride_of(x)
    )
    kwargs = _quant_kwargs(plan, x, outputs, gs_inv, out_scale if fold else gs_inv)
    with tvm_ffi.use_torch_stream():
        entry(*_arguments(arg_plan, kwargs))
    return outputs.fp4, outputs.sf, outputs.scale


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
    fp16 tensor.  ``tactic`` overrides :func:`default_tactic`; without it the GEMM runs the
    program of :func:`gemm_alone_tactic` on the cells of :data:`GEMM_ALONE_EP_RULES`.  The
    JIT module is built and loaded here, so prepare outside CUDA Graph capture."""
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
    out_f16 = out.dtype == torch.float16
    plan = gemm_plan(m, n, k, out_f16, arch, _sm_count(device), tactic=tactic)
    if tactic is None and (alone := gemm_alone_tactic(plan)) is not None:
        plan = gemm_plan(m, n, k, out_f16, arch, _sm_count(device), tactic=alone)
    kwargs = _gemm_kwargs(plan, a_fp4, a_flat, b_fp4, b_flat, alpha, out)
    with torch.cuda.device(_device_index(device)):
        launch = _bind(arch, plan.kernel_key, kwargs)
    return NVFP4PerTokenGemmRunner(plan, out, (launch,))


def _gemm_kwargs(
    plan: GemmPlan,
    a_fp4: torch.Tensor,
    a_flat: torch.Tensor,
    b_fp4: torch.Tensor,
    b_flat: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
) -> dict[str, Any]:
    """Kernel arguments of one GEMM launch (operand views of the resolved plan)."""
    m, n, k = plan.M, plan.N, plan.K
    cols = padded_sf_cols(k)
    pm, pn = padded_rows(m), padded_rows(n)
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
            out.device, plan.sk_workspace_floats, plan.sk_flag_words, plan.sk_slices
        )
        kwargs.update(
            red_ws=red_ws,
            red_flags=red_flags,
            red_gen=red_flags,
            sk_tiles=plan.sk_tiles,
            sk_pairs=plan.sk_pairs,
        )
    return kwargs


@functools.lru_cache(maxsize=4096)
def _gemm_prepared(
    m: int, n: int, k: int, out_f16: bool, device_index: int
) -> tuple[GemmPlan, Callable[..., Any], tuple[tuple[str, str], ...]]:
    """Plan, loaded FFI entry and argument plan of the default-tactic GEMM of one shape on one device."""
    device = torch.device("cuda", device_index)
    arch = _device_arch(device)
    plan = gemm_plan(m, n, k, out_f16, arch, _sm_count(device))
    if (alone := gemm_alone_tactic(plan)) is not None:
        plan = gemm_plan(m, n, k, out_f16, arch, _sm_count(device), tactic=alone)
    with torch.cuda.device(device_index):
        entry, arg_plan = _entry(arch, plan.kernel_key)
    return plan, entry, arg_plan


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
    if a_u8.dim() != 2 or not a_u8.is_contiguous():
        raise ValueError("a_fp4 must be a contiguous uint8 [M, K/2] tensor")
    m, kh = (int(v) for v in a_u8.shape)
    n, kh_b = (int(v) for v in b_u8.shape)
    if kh != kh_b:
        raise ValueError(
            f"K mismatch: a_fp4 has {2 * kh} elements per row, b_fp4 {2 * kh_b}"
        )
    k = 2 * kh
    device = a_u8.device
    if device.type != "cuda":
        raise ValueError("the operands must be CUDA tensors")
    if any(t.device != device for t in (a_descale, b_u8, b_sf, alpha, out)):
        raise ValueError("every operand must be on one CUDA device")
    if out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"out must be bf16 or fp16, got {out.dtype}")
    if tuple(out.shape) != (m, n) or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous [{m}, {n}] tensor")
    if alpha.dtype != torch.float32 or alpha.numel() != m:
        raise ValueError("alpha must be a float32 tensor with one value per token")
    if not alpha.is_contiguous():
        alpha = alpha.contiguous()
    cols = padded_sf_cols(k)
    a_flat = _flat_u8_scales(a_descale, "a_sf", padded_rows(m), cols)
    b_flat = _flat_u8_scales(b_sf, "b_sf", padded_rows(n), cols)
    plan, entry, arg_plan = _gemm_prepared(
        m, n, k, out.dtype == torch.float16, _device_index(device)
    )
    kwargs = _gemm_kwargs(plan, a_u8, a_flat, b_u8, b_flat, alpha, out)
    with tvm_ffi.use_torch_stream():
        entry(*_arguments(arg_plan, kwargs))
    return out


# ---------------------------------------------------------------------------
# Quantizer + GEMM chain
# ---------------------------------------------------------------------------


def _entry_prefetch_tactic(
    plan: GemmPlan, rules: dict[str, dict[str, tuple[int, int]]]
) -> Optional[dict[str, Any]]:
    """``plan``'s tactic with the entry L2 prefetch (``ep_stages`` = :data:`CHAIN_EP_STAGES`)
    when its program family (kernel key without the output dtype) and token count lie in a
    cell of ``rules`` on its architecture (for the chain, also of
    :data:`CHAIN_EP_MEASURED_CELLS`) and that program is registered, else ``None``."""
    family = "_".join(
        part
        for part in plan.kernel_key.removeprefix("gemm:").split("_")
        if part not in ("bf16", "f16")
    )
    cell = rules.get(plan.arch, {}).get(family)
    if (cell is None or not cell[0] <= plan.M <= cell[1]) and rules is CHAIN_EP_RULES:
        measured = CHAIN_EP_MEASURED_CELLS.get(plan.arch, {}).get(family)
        if (
            measured is not None
            and MEASURED_SM_COUNT.get(plan.arch) == plan.sm_count
            and (plan.K, plan.N) in measured[2]
        ):
            cell = measured[:2]
    if cell is None or not cell[0] <= plan.M <= cell[1]:
        return None
    tactic = {**plan.tactic, "ep_stages": CHAIN_EP_STAGES}
    if gemm_kernel_key(tactic, plan.out_f16) not in KERNELS.get(plan.arch, {}):
        return None
    return tactic


def _sf_ring_tactic(plan: GemmPlan, kind: str) -> Optional[dict[str, Any]]:
    """``plan``'s tactic with the scale-factor ring (``sf_ring``) when ``plan`` runs on
    :data:`SF_RING_ARCH` at its measured SM count and its program is a
    :data:`SF_RING_CLC_KEYS` program, or a :data:`SF_RING_G8_KEYS` program on a cell of
    ``SF_RING_G8_CELLS[kind]``, and the ring program is registered; for a
    :data:`SF_RING_FORM_KEYS` program on a cell of ``SF_RING_FORM_CELLS[kind]``, the
    ring tactic of the cell's form (:data:`SF_RING_FORMS`); first, for any of these
    programs on a token range of ``SF_RING_MC42_CELLS[kind]`` with even token and
    weight tile counts, the cluster-of-8 multicast form (``mc42``) of the CLC ring
    program without raster groups; else ``None``."""
    if plan.arch != SF_RING_ARCH or MEASURED_SM_COUNT.get(plan.arch) != plan.sm_count:
        return None
    mc42_ranges = SF_RING_MC42_CELLS[kind].get((plan.K, plan.N), ())
    if (
        any(lo <= plan.M <= hi for lo, hi in mc42_ranges)
        and plan.tok_tiles % 2 == 0
        and plan.w_tiles % 2 == 0
        and (
            plan.kernel_key in SF_RING_FORM_KEYS
            or plan.kernel_key in SF_RING_G8_KEYS
            or plan.kernel_key in SF_RING_CLC_KEYS
        )
    ):
        tactic = {
            name: value
            for name, value in plan.tactic.items()
            if name not in ("sched", "raster_group", "stream_k", "sk_split")
        }
        tactic.update(sched="clc", sf_ring=True, mc42=True)
    elif plan.kernel_key in SF_RING_FORM_KEYS:
        form = next(
            (
                form
                for lo, hi, form in SF_RING_FORM_CELLS[kind].get((plan.K, plan.N), ())
                if lo <= plan.M and (hi is None or hi >= plan.M)
            ),
            None,
        )
        if form is None:
            return None
        tactic = {
            name: value
            for name, value in plan.tactic.items()
            if name not in ("sched", "stream_k", "sk_split")
        }
        tactic.update(SF_RING_FORMS[form], sf_ring=True)
    else:
        if plan.kernel_key not in SF_RING_CLC_KEYS:
            cell = SF_RING_G8_CELLS[kind].get((plan.K, plan.N))
            if (
                plan.kernel_key not in SF_RING_G8_KEYS
                or cell is None
                or not cell[0] <= plan.M <= cell[1]
            ):
                return None
        tactic = {**plan.tactic, "sf_ring": True}
    if gemm_kernel_key(tactic, plan.out_f16) not in KERNELS.get(plan.arch, {}):
        return None
    return tactic


def gemm_alone_tactic(plan: GemmPlan) -> Optional[dict[str, Any]]:
    """The tactic the GEMM alone takes when no tactic is passed and the default-tactic plan
    is ``plan``: the entry L2 prefetch program on the cells of :data:`GEMM_ALONE_EP_RULES`,
    the scale-factor ring program (:func:`_sf_ring_tactic`, kind ``"gemm"``), else
    ``None`` (the plan's own program)."""
    tactic = _entry_prefetch_tactic(plan, GEMM_ALONE_EP_RULES)
    return tactic if tactic is not None else _sf_ring_tactic(plan, "gemm")


def chain_gemm_tactic(plan: GemmPlan) -> Optional[dict[str, Any]]:
    """The GEMM tactic of the quantize + GEMM chain whose default-tactic GEMM plan is
    ``plan``: the entry L2 prefetch program on the cells of :data:`CHAIN_EP_RULES`, the
    scale-factor ring program (:func:`_sf_ring_tactic`, kind ``"chain"``), the plan's own
    tactic on a scale-factor ring cell of the GEMM alone only, else ``None`` (the program
    the GEMM alone runs)."""
    tactic = _entry_prefetch_tactic(plan, CHAIN_EP_RULES)
    if tactic is None:
        tactic = _sf_ring_tactic(plan, "chain")
    if (
        tactic is None
        and _entry_prefetch_tactic(plan, GEMM_ALONE_EP_RULES) is None
        and _sf_ring_tactic(plan, "gemm") is not None
    ):
        tactic = dict(plan.tactic)
    return tactic


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
    activation (:func:`allocate_nvfp4_per_token_quantize_outputs` for ``x``).  The GEMM
    runs the program of :func:`chain_gemm_tactic` (the entry L2 prefetch of the weight
    tiles on the cells of :data:`CHAIN_EP_RULES`), else the one ``mm_fp4`` runs."""
    quantize = _prepare_quantize(
        x,
        global_scale_inv,
        workspace,
        out_scale=out_scale,
        alone=False,
        n=int(b_fp4.shape[0]),
    )
    device = workspace.fp4.device
    plan = gemm_plan(
        int(workspace.fp4.shape[0]),
        int(b_fp4.shape[0]),
        2 * int(workspace.fp4.shape[1]),
        out.dtype == torch.float16,
        _device_arch(device),
        _sm_count(device),
    )
    gemm = prepare_mm_fp4_per_token(
        workspace.fp4,
        workspace.sf,
        b_fp4,
        b_sf,
        workspace.scale,
        out,
        tactic=chain_gemm_tactic(plan),
    )
    return NVFP4PerTokenChainRunner(quantize, gemm)
