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

Host planner of the Cake SM90 (Hopper) variable block-sparse attention kernels.

Pure CPU code: turns a boolean ``[H, MB, NB]`` block mask into the by-value
plan of the small-selection kernels (:func:`plan_small`), the uploaded tile
metadata of the persistent kernel (:func:`plan_vsa_sm90`) and the route rule
that chooses between them (:func:`small_route`, :func:`split_kmax`,
:func:`cluster_variant_for`, :func:`split_cluster_variant_for`), plus the
device's co-resident cluster capacity
(:func:`cluster_capacity`, the one function here that touches the driver).
The plan layouts mirror the Cake kernel modules (``vsa_sm90_bf16`` and
``vsa_sm90_small``) and are validated by the kernels' own tests; both planners
are dependency-free ports of the Cake planners and must stay byte-identical to
them (the two are changed in the same round).  This module imports torch only;
the plan object that owns the device state and launches is
:class:`flashinfer.cake_vsa_sm90_backend.CakeVsaSm90Plan`.
"""

from __future__ import annotations

import functools
import heapq
from typing import Any, Mapping, Optional, Sequence

import torch

BLOCK = 64
HEAD_DIM = 128
NUM_K_STAGES = 3
NUM_V_STAGES = 3
NUM_CONSUMER_WGS = 2
MAX_SEQ = 136
MAX_SELECTED = 64
TILE_FIXED_COST = 0.3
# A tile's load time also carries a fixed part (Q lift, epilogue, inter-tile gap: ~ one
# position's load) and a split tile the merge of its two warpgroups' partials; in ring
# positions.  Mirrors ``vsa_sm90_bf16``.
TILE_FIXED_LOAD = 1.0
SPLIT_MERGE_LOAD = 1.0
LOAD_COST = 1.0
# By-value first-tile header of the persistent kernel: per CTA ``[head, qb0,
# qb1, mode, n_hdr, (blk_a, blk_b) x HDR_POS, pad]`` (int16) so the first Q
# and ring positions are issued at CTA start, ahead of the plan row's load.
HDR_POS = 3  # header slots (fixed kernel layout)
HDR_ISSUE = 2  # positions the planner puts in the header (two measured best; three hurt the k12 row)
HDR_WORDS = 12
HDR_CTAS = 144
HDR_HALFWORDS = HDR_CTAS * HDR_WORDS
# Ragged plans list-schedule costlier tiles first over all heads (LPT) when
# every head's K and V fit this many bytes of L2; larger working sets keep the
# head-major order so the CTAs running concurrently share one head's blocks.
TILE_ORDER_L2_BUDGET = 24 << 20
PAIRING_WINDOW = 16
LOG2E = 1.4426950408889634
_FP32_MAX = 3.4028234663852886e38

META_NSEQ = 4
META_NOWN = 5
META_NTILES = 7
# Queue kernel (``attention_queue``): every CTA keeps its first tile (the
# by-value header's) and takes the rest from a global queue at run time, in the
# planner's head-major tile order (full dynamic queue; the kernel's loop bound
# admits POOL_ROUNDS pooled tiles per CTA at most, so larger plans keep the
# surplus static).  Used for uniform plans whose every CTA has at least
# POOL_MIN_TILES tiles (or the tiny-tile exception), and for ragged plans with at
# least POOL_RAGGED_MIN_TILES tiles per CTA.  POOL_ROUNDS matches the kernel.
POOL_ROUNDS = 256
POOL_MIN_TILES = (
    12  # uniform plans; 2..8-tile rows measured -0.2..-2.2 % with the full queue
)
# Ragged plans (cost-descending within head) take the queue from this many tiles per
# CTA: measured +0.7..+1.7 % at >= 29 tiles per CTA (m >= 75648, h >= 7), -1.3..-3.3 %
# at 12..25 tiles per CTA (h3-m75648, h10/h12-m32768).
POOL_RAGGED_MIN_TILES = 28
# Ragged plans whose heads are long enough that the queue's window of one grid's worth of
# consecutive pool tiles stays inside one head's K/V (tiles per head >=
# POOL_RAGGED_LONG_HEAD_WINDOWS x grid) join from this many tiles per CTA.  Measured on
# H100 (5 groups x 128, two processes): h3-m109632-k64 ragged (857 tiles per head, 18..21
# per CTA) +1.46 % / +1.5 %; h12-m32768-k64 ragged (256 tiles per head, 18..25 per CTA)
# -2.8 %, h10-m32768 (256, 13..21) -3.0 %, h8-m32768 (256, 11..17) -5.5 %, h3-m75648 (591
# per head but 12..15 per CTA) -1.1 %.
POOL_RAGGED_LONG_HEAD_MIN_TILES = 18
POOL_RAGGED_LONG_HEAD_WINDOWS = 4
POOL_SMALL_MIN_TILES = (
    7  # ... or this many tiles of at most POOL_SMALL_TILE_POS positions
)
# Tiny-tile exception: plans of one-position tiles with >= 7 tiles per CTA measured
# +4.2 % / +4.3 % on h8-m8192-k2 (split tiles of one position, 8 per CTA).  Six-position
# pair tiles at 7..8 per CTA (h16-m8192-k5) measured -1.0 % against the static kernel
# (static 1.0131 vs queue 1.0028 against the previous Cake, same process); 16-position
# tiles at 7-8 per CTA -0.8 %, 8-position tiles at 4 per CTA -0.5 %.
POOL_SMALL_TILE_POS = 1
# Tile ORDER of static UNIFORM plans before the
# list schedule.  The queue kernel balances its pooled tiles at run time and ragged plans
# are already list-scheduled costliest first, so both keep the legacy order; the mode and
# the grid candidates are evaluated on the legacy order and the tiles themselves (hence
# the output bits) never change, only which CTA runs which tile.  Plans of one round (no
# more tiles than CTAs) take the LPT order over all heads: CTAs launch in index order, so
# the costliest tiles start first.  Plans of several rounds take the LPT order within
# each head (head-major kept: the CTAs streaming concurrently share one head's K/V in
# L2; the LPT order over all heads measured -1.2 % (h8-m4096-k16), -9.5 % (h32-m2048-k8)
# and -6.5 % (h8-m2048-n32768-k48-ragged1) against the legacy order).  Measured A:N on
# H100 (5 groups): h7-m16384-k32 +1.9 %, h32-m2048-k8 +1.7 %, h8-m2048-k16 (one round)
# +1.0 / +2.2 %, h8-m4096-k16 +0.3 %, h8-m4096-k32 +0.5 % (noise band +-0.7 %).
# Mirrors the kernel-tests reference planner.  False disables.
UNIFORM_LPT = True
# Ragged STATIC plans of at most
# RAGGED_LPT_MAX_TILES tiles per CTA (legacy assignment) are re-ordered like uniform plans
# before the list schedule: by -max(cost, load) over all heads when every head's K and V fit
# the L2 budget (the legacy ragged order already; no change), within each head beyond it
# (legacy: cost descending, i.e. the longer warpgroup's positions; max(cost, load) =
# positions loaded ranks the split tiles of h8-m2048-n32768-k48-ragged1 (1..4 tiles per
# CTA, 28 % list-schedule tail) finer: measured A:N +1.5 % / T:N +1.6 % on H100, 5 groups,
# all groups positive).  Plans of more tiles per CTA (the long ragged rows, 9..26 tiles,
# tails 1.3..3.5 %) measured -0.4..+0.5 % with the same re-order (noise) and keep the legacy
# order.  The mode and the grid are decided on the legacy order; the tile set (hence the
# output bits) never changes.  Mirrors the kernel-tests reference planner.  0 disables.
RAGGED_LPT_MAX_TILES = 8
META_SEQ_OFF = 8
META_SEQ_WORDS = MAX_SEQ // 2
MAX_OWN = MAX_SEQ // 2
OWN_WORDS = MAX_OWN // 2
META_OWN_OFF = META_SEQ_OFF + META_SEQ_WORDS
META_WORDS = META_OWN_OFF + NUM_CONSUMER_WGS * OWN_WORDS  # 144

# Small-selection route (port of the Cake kernel module ``vsa_sm90_small``).
SMALL_KMAX_VARIANTS = (1, 3, 4, 6)
# Split-KV variants that ship.  :func:`split_kmax` ranks variants by
# ``chain_blocks(kmax) + SPLIT_MERGE_COST * (max_nsplit - 1)``: KMAX 3 has the
# same two-block chain as KMAX 4 and never fewer slices, so KMAX 4 always wins
# the tie-or-better; KMAX 1 (one-block chain, one slice per block) only wins
# when KMAX 3/4/6 all exceed the one-wave item budget while KMAX 1 fits it,
# a band of ragged masks with ``2 * tiles`` within two of the SM count and one
# selection of 13..31 blocks.  Those masks run the persistent kernel instead.
SPLIT_KMAX_VARIANTS = (4, 6)
SMALL_OCCUPANCY = {
    1: 4,
    2: 1,
    3: 1,
    4: 1,
    6: 1,
}  # CTAs per SM on H100: KMAX 1 is one warpgroup (49 KB); KMAX > 1 run two warpgroups (256 threads)
PLAN_HALFWORDS = 1750  # int16 elements of the by-value plan parameter (3500 B)
PLAN_META_UNSPLIT = 1  # unsplit rows: [count, blk...] per query block
PLAN_META_SPLIT = 2  # split rows: [meta, qtile, blk...] per item
MAX_NSPLIT = 31  # nsplit field of ``meta`` (bits 10..14 keep the int16 sign clear)
SMALL_ITEM_ELEMS = BLOCK * HEAD_DIM  # FP32 partial accumulator per split item
SMALL_STATS_FLOATS = 2 * BLOCK  # (max, sum) per row per split item
# Split-KV cost model (relative units): one KV block through the chain vs. one
# extra slice merged by the last CTA (32 KB FP32 read + weights) plus the route's
# fixed workspace round trip (publish, fence, arrival atomic, last-CTA re-read).
# Same constants as the Cake planner (``vsa_sm90_small``): chain block 0.81 us,
# slice 0.25 us, fixed 1.6 us on H100.
SPLIT_BLOCK_COST = 1.0
SPLIT_MERGE_COST = 0.3
SPLIT_FIXED_COST = (
    2.0  # route comparison only (split_kmax ranks splits among themselves)
)
# Cluster / DSM-merge route (port of the Cake cluster variants):
# (kmax, cluster size) variants, SMs per GPC assumed for one-wave placement,
# and the cost model (merged rank, padded slot) in the same block units.
CLUSTER_VARIANTS = (
    (2, 4),
    (3, 3),
    (3, 6),
    (4, 2),
    (4, 4),
    (6, 2),
)  # (6, 3): receive buffers exceed the 227 KB SMEM
# One-wave cluster placement uses the device's co-resident cluster capacity
# ``{csize: clusters}`` resolved from ``cuOccupancyMaxActiveClusters`` at one
# CTA per SM (:func:`cluster_capacity`).  A per-GPC model (8 GPCs x 16 SMs)
# over-counted four-CTA clusters on the 132-SM H100 (32 vs 30) and sent
# 31-32-tile problems into a second cluster wave.
CLUSTER_MERGE_COST = 0.3
CLUSTER_PAD_COST = 0.15
# Split-cluster route: split-KV slices of a query block form a thread-block
# cluster; every slice publishes its FP32 partial to the workspace, one cluster
# rendezvous replaces the arrival counter, and up to four slices each merge one
# 32-column O set of every slice (``nsplit x 8 KB`` reads per merging CTA
# instead of ``nsplit x 32 KB`` by one).  ``(kmax, csize)`` variants
# (capacities 18 / 24 / 48); one unsplit-style row per query block with
# ``kmax * csize`` id slots.  ``SPLIT_CLUSTER_ROUTE`` enables the route in
# :func:`small_route` (the stages ``small_k6sc{3,4,8}``, a cluster launch).
SPLIT_CLUSTER_VARIANTS = ((6, 3), (6, 4), (6, 8))
SPLIT_CLUSTER_ROUTE = True


def _chain_blocks(kmax: int) -> int:
    """Blocks on the critical chain of one CTA: two warpgroups walk the even and odd blocks (KMAX > 1)."""
    return -(-int(kmax) // 2) if int(kmax) > 1 else int(kmax)


def plan_meta(split: bool) -> int:
    """Halfwords before the block ids in one plan row of the given kernel family."""
    return PLAN_META_SPLIT if split else PLAN_META_UNSPLIT


def small_kmax_for(capacity: int) -> int:
    """Smallest compiled small-kernel variant that holds ``capacity`` blocks per tile."""
    for kmax in SMALL_KMAX_VARIANTS:
        if capacity <= kmax:
            return kmax
    raise ValueError(
        f"small kernel supports at most {SMALL_KMAX_VARIANTS[-1]} KV blocks per query block, got {capacity}"
    )


# A problem whose persistent grid would fill at
# most ``SMs / SPLIT_UNDERFILL_SMS`` SMs and whose selection no one-wave split / cluster
# variant holds takes the split-KV variant in up to SPLIT_UNDERFILL_WAVES waves of items
# instead (the arrival-counter merge needs no co-residency; the by-value plan holds at
# most ~218 split items).  Measured on H100 (A:N, 5 groups): h8-m128-n32768-k64 (16 tiles
# of 64 blocks) 35.2 us persistent -> 28.8 / 29.4 us as KMAX 6 split (176 items, +20..22 %);
# h8-m256-n16384-k32 (32 tiles of 32 blocks) 23.0 us persistent -> 25.5 us KMAX 6 (192
# items) / 26.3 us KMAX 4 (256 items): slower, so the rule stops at 16 tiles.  0 disables.
SPLIT_UNDERFILL_SMS = 8
SPLIT_UNDERFILL_WAVES = 2


def split_kmax(counts: list[int], *, sms: Optional[int] = None, waves: int = 1) -> int:
    """KMAX of the split-KV variant for the per-query-block selection ``counts``.

    Variants whose item count fits ``waves`` waves (when ``sms`` is given) and whose
    plan fits the parameter bank are ranked by ``chain_blocks(kmax) * SPLIT_BLOCK_COST
    + (max_nsplit - 1) * SPLIT_MERGE_COST``; the cheapest wins (largest kmax on
    ties).  Raises when no variant fits.  Mirrors ``vsa_sm90_small.split_kmax``
    over the shipped :data:`SPLIT_KMAX_VARIANTS`.
    """
    best: Optional[tuple[tuple[int, float], int]] = None
    for kmax in SPLIT_KMAX_VARIANTS:
        nsplits = [max(1, -(-c // kmax)) for c in counts]
        items = sum(nsplits)
        if (
            max(nsplits) > MAX_NSPLIT
            or items * (kmax + PLAN_META_SPLIT) > PLAN_HALFWORDS
        ):
            continue
        if sms is not None and items > waves * SMALL_OCCUPANCY[kmax] * sms:
            continue
        cost = (
            _chain_blocks(kmax) * SPLIT_BLOCK_COST
            + (max(nsplits) - 1) * SPLIT_MERGE_COST
        )
        # Beyond one wave the item count sets the number of waves: fewest items
        # first (h8-m256-n16384-k32: KMAX 4 at 256 items 26.3 us, KMAX 6 at 192
        # items 25.5 us, both slower than the 23.0 us persistent plan); within
        # one wave every variant ranks by cost alone.
        key = (items if waves > 1 else 0, cost)
        if best is None or key <= best[0]:
            best = (key, kmax)
    if best is None:
        raise ValueError("no split-KV variant fits this selection in one wave")
    return best[1]


def split_cost(counts: list[int], kmax: int) -> float:
    """Modelled cost of the split-KV variant ``kmax``: fixed workspace round trip, chain, merged slices of the fullest query block."""
    return (
        SPLIT_FIXED_COST
        + _chain_blocks(kmax) * SPLIT_BLOCK_COST
        + (max(-(-c // kmax) for c in counts) - 1) * SPLIT_MERGE_COST
    )


def cluster_cost(counts: list[int], kmax: int, csize: int) -> float:
    """Modelled cost of cluster variant ``(kmax, csize)``: chain over the blocks one CTA actually holds, DSM-merged ranks, mean padded slots."""
    pad = kmax * csize - sum(counts) / len(counts)
    per_cta = min(int(kmax), -(-max(counts) // int(csize)))
    return (
        _chain_blocks(per_cta) * SPLIT_BLOCK_COST
        + (csize - 1) * CLUSTER_MERGE_COST
        + pad * CLUSTER_PAD_COST
    )


def cluster_variant_for(
    counts: list[int],
    *,
    sms: Optional[int] = None,
    cluster_capacity: Optional[Mapping[int, int]] = None,
) -> Optional[tuple[int, int]]:
    """``(kmax, csize)`` of the cheapest cluster variant for ``counts``, or ``None``.

    Candidates hold the largest selection (``kmax * csize >= capacity``), fit
    the plan parameter and, when ``sms`` is given, run as one wave of clusters:
    ``tiles <= cluster_capacity[csize]``, the device's co-resident cluster
    count at one CTA per SM (:func:`cluster_capacity`).  Ranked by
    :func:`cluster_cost`.  Mirrors ``vsa_sm90_small.cluster_variant_for``.
    """
    if sms is not None and cluster_capacity is None:
        raise ValueError(
            "[cluster_capacity_unresolved] cluster_variant_for(sms=...) needs "
            "cluster_capacity={csize: co-resident clusters} resolved from the device "
            "(cluster_capacity(device))"
        )
    capacity = max(counts)
    tiles = len(counts)
    best = None
    for kmax, csize in CLUSTER_VARIANTS:
        if (
            kmax * csize < capacity
            or tiles * csize * (kmax + PLAN_META_SPLIT) > PLAN_HALFWORDS
        ):
            continue
        if sms is not None:
            assert cluster_capacity is not None
            if csize not in cluster_capacity:
                raise ValueError(
                    f"[cluster_capacity_missing] no co-resident cluster count for cluster size {csize}"
                )
            if tiles > int(cluster_capacity[csize]):
                continue
        cost = cluster_cost(counts, kmax, csize)
        if best is None or cost < best[0]:
            best = (cost, kmax, csize)
    return None if best is None else (best[1], best[2])


_PROBE_KERNEL = "cake_vsa_sm90_cluster_capacity_probe"
_PROBE_SOURCE = f'extern "C" __global__ void {_PROBE_KERNEL}() {{}}\n'
CLUSTER_SIZES: tuple[int, ...] = tuple(
    sorted(
        {csize for _kmax, csize in CLUSTER_VARIANTS}
        | {csize for _kmax, csize in SPLIT_CLUSTER_VARIANTS}
    )
)


def _check_cuda(result):
    """Unwrap a ``cuda.bindings`` result ``(status, *values)``; a non-zero status raises with the error's name."""
    status = result[0]
    if status.value:
        from cuda.bindings import driver, nvrtc

        if isinstance(status, driver.CUresult):
            err, name = driver.cuGetErrorName(status)
            label = name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(status, nvrtc.nvrtcResult):
            label = nvrtc.nvrtcGetErrorString(status)[1]
        else:
            label = type(status).__name__
        raise RuntimeError(
            f"cluster capacity probe: CUDA error code={status.value}({label})"
        )
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


@functools.lru_cache(maxsize=None)
def cluster_capacity(device_index: int) -> dict[int, int]:
    """``{cluster size: max co-resident clusters}`` of ``device_index`` at one CTA per SM.

    ``cuOccupancyMaxActiveClusters`` on an empty probe kernel launched with the
    device's maximum opt-in dynamic shared memory (one CTA per SM).  The answer
    follows the part's GPC topology after floorsweeping, not ``SMs // csize``:
    a 132-SM H100 SXM holds 66 two-CTA clusters but 30 four-CTA clusters.
    """
    from cuda.bindings import driver, nvrtc

    checkCudaErrors = _check_cuda

    with torch.cuda.device(device_index):
        torch.empty(1, device="cuda")  # primary context
        major, minor = torch.cuda.get_device_capability(device_index)
        prog = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(_PROBE_SOURCE.encode(), b"probe.cu", 0, [], [])
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        size = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, cubin))
        checkCudaErrors(nvrtc.nvrtcDestroyProgram(prog))
        ctx = checkCudaErrors(driver.cuDevicePrimaryCtxRetain(device_index))
        try:
            checkCudaErrors(driver.cuCtxSetCurrent(ctx))
            module = checkCudaErrors(driver.cuModuleLoadData(cubin))
            try:
                func = checkCudaErrors(
                    driver.cuModuleGetFunction(module, _PROBE_KERNEL.encode())
                )
                smem = checkCudaErrors(
                    driver.cuDeviceGetAttribute(
                        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                        device_index,
                    )
                )
                checkCudaErrors(
                    driver.cuFuncSetAttribute(
                        func,
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem,
                    )
                )
                result = {}
                for csize in CLUSTER_SIZES:
                    attr = driver.CUlaunchAttribute()
                    attr.id = (
                        driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                    )
                    attr.value.clusterDim.x = csize
                    attr.value.clusterDim.y = 1
                    attr.value.clusterDim.z = 1
                    config = driver.CUlaunchConfig()
                    config.gridDimX = csize
                    config.gridDimY = 1
                    config.gridDimZ = 1
                    config.blockDimX = 128
                    config.blockDimY = 1
                    config.blockDimZ = 1
                    config.sharedMemBytes = smem
                    config.attrs = [attr]
                    config.numAttrs = 1
                    result[int(csize)] = int(
                        checkCudaErrors(
                            driver.cuOccupancyMaxActiveClusters(func, config)
                        )
                    )
            finally:
                checkCudaErrors(driver.cuModuleUnload(module))
        finally:
            checkCudaErrors(driver.cuDevicePrimaryCtxRelease(device_index))
    return result


def small_route(
    block_mask: torch.Tensor, *, sms: int, cluster_capacity: Mapping[int, int]
) -> Optional[tuple[int, bool, int]]:
    """``(kmax, split, cluster)`` when the problem should take a small-kernel route, else ``None``.

    Unsplit rule: every query block selects at most ``SMALL_KMAX_VARIANTS[-1]``
    KV blocks, the whole grid fits one wave of the chosen variant (``tiles <=
    occupancy * SMs``) and the plan fits the by-value parameter.  Sliced rules
    (tried when the unsplit rule fails): the grid is too small for the
    persistent kernel's two consumer warpgroups per CTA (``2 * tiles <= SMs``);
    the split-KV variant of :func:`split_kmax` and the cluster variant of
    :func:`cluster_variant_for` (one wave of clusters by the device's
    :func:`cluster_capacity`) are compared by their modelled costs and the
    cheaper one wins; a chosen split route is upgraded to its split-cluster
    variant (:func:`split_cluster_variant_for`) when one saves enough per-CTA
    ingest.  Everything larger goes to the persistent pair/split kernel.
    Mirrors ``vsa_sm90_small.small_route``.
    """
    mask = block_mask.to("cpu", torch.bool)
    h, mb, _nb = mask.shape
    counts = mask.sum(dim=-1).reshape(-1).tolist()
    capacity = max(counts)
    tiles = h * mb
    if capacity <= SMALL_KMAX_VARIANTS[-1]:
        kmax = small_kmax_for(capacity)
        if (
            tiles <= SMALL_OCCUPANCY[kmax] * sms
            and tiles * (kmax + PLAN_META_UNSPLIT) <= PLAN_HALFWORDS
        ):
            return kmax, False, 0
    if 2 * tiles <= sms:
        try:
            split: Optional[int] = split_kmax(counts, sms=sms)
        except ValueError:
            split = None
        cluster = cluster_variant_for(
            counts, sms=sms, cluster_capacity=cluster_capacity
        )
        if (
            split is None
            and cluster is None
            and SPLIT_UNDERFILL_SMS > 0
            and tiles * SPLIT_UNDERFILL_SMS <= sms
        ):
            try:
                split = split_kmax(counts, sms=sms, waves=SPLIT_UNDERFILL_WAVES)
            except ValueError:
                split = None
        if cluster is not None and (
            split is None or cluster_cost(counts, *cluster) < split_cost(counts, split)
        ):
            return cluster[0], False, cluster[1]
        if split is not None:
            if SPLIT_CLUSTER_ROUTE:
                sc = split_cluster_variant_for(
                    counts, split_kmax=split, cluster_capacity=cluster_capacity
                )
                if sc is not None:
                    return sc[0], True, sc[1]
            return split, True, 0
    return None


def _split_ingest_kb(counts: Sequence[int], kmax: int, cluster: Optional[int]) -> int:
    """Per-CTA L2 ingest (KB) of a split route: the longest slice's K/V bytes plus the merging CTA's partial reads
    (the small kernel is bound by the SM's ~60 GB/s L2 -> SM ingest)."""
    worst = 0
    for count in counts:
        nsplit = max(1, -(-count // kmax))
        blocks = -(-count // nsplit)
        if cluster is None:
            merge = 32 * (nsplit - 1)
        else:
            n_active = min(nsplit, 4)
            merge = 8 * nsplit * (-(-4 // n_active)) if nsplit > 1 else 0
        worst = max(worst, 32 * blocks + merge)
    return worst


SPLIT_CLUSTER_MIN_SAVING_KB = 48


def split_cluster_variant_for(
    counts: Sequence[int], *, split_kmax: int, cluster_capacity: Mapping[int, int]
) -> Optional[tuple[int, int]]:
    """The smallest split-cluster ``(kmax, csize)`` that holds the selection, fits the plan, places in one wave of
    clusters (``tiles <= cluster_capacity[csize]``) and saves at least ``SPLIT_CLUSTER_MIN_SAVING_KB`` of per-CTA
    ingest over the split-KV variant ``split_kmax``, else ``None``."""
    capacity, tiles = max(counts), len(counts)
    split_kb = _split_ingest_kb(counts, split_kmax, None)
    for kmax, csize in SPLIT_CLUSTER_VARIANTS:
        if kmax * csize < capacity:
            continue
        if tiles * (kmax * csize + PLAN_META_UNSPLIT) > PLAN_HALFWORDS:
            continue
        cap = cluster_capacity.get(csize)
        if cap is None or tiles > cap:
            continue
        if (
            split_kb - _split_ingest_kb(counts, kmax, csize)
            < SPLIT_CLUSTER_MIN_SAVING_KB
        ):
            continue
        return kmax, csize
    return None


def _balanced_chunks(ids: list[int], nsplit: int) -> list[list[int]]:
    """Split ``ids`` into ``nsplit`` contiguous chunks whose sizes differ by at most one."""
    n = len(ids)
    base, extra = divmod(n, nsplit)
    out, pos = [], 0
    for j in range(nsplit):
        size = base + (1 if j < extra else 0)
        out.append(ids[pos : pos + size])
        pos += size
    return out


def plan_small(
    block_mask: torch.Tensor,
    *,
    kmax: Optional[int] = None,
    split: bool = False,
    cluster: int = 0,
) -> dict:
    """Plan rows as int16 halfwords: ``[count, blk...]`` per tile (unsplit) or ``[meta, qtile, blk...]`` per item (split / cluster).

    Without ``split`` every query block is one item (``nsplit = 1``) and
    ``kmax`` must hold the largest selection.  With ``split`` a query block
    whose selection exceeds ``kmax`` becomes ``ceil(count / kmax)`` items of
    nearly equal size, merged on the device by the last-arriving CTA.  With
    ``cluster = csize`` every query block is exactly ``csize`` consecutive
    items (empty slices padded with ``count = 0``) forming one thread-block
    cluster that merges through distributed shared memory.
    Returns the CPU int16 ``plan`` of exactly ``PLAN_HALFWORDS`` elements (the
    by-value kernel parameter, ``-1`` padded), or raises when the problem does
    not fit.  Byte-identical to the Cake planner (``vsa_sm90_small.plan_small``).
    """
    mask = block_mask.to("cpu", torch.bool)
    if mask.ndim != 3:
        raise ValueError("block_mask must be (H, MB, NB)")
    h, mb, nb = mask.shape
    if nb > 32767:
        raise ValueError("KV block ids must fit int16")
    counts = mask.sum(dim=-1, dtype=torch.int32)
    if bool((counts == 0).any()):
        raise ValueError("every query block must select at least one KV block")
    capacity = int(counts.max())
    flat_counts = counts.reshape(-1).tolist()
    cluster = int(cluster or 0)
    if cluster and split:
        # Split-cluster: one row per query block, ``[count, blk...]`` with ``kmax * cluster`` id slots; every
        # query block is ``cluster`` consecutive items (slice = cluster rank; slice bounds computed in the kernel).
        if kmax is None:
            kmax = min(
                (
                    k
                    for k, c in SPLIT_CLUSTER_VARIANTS
                    if c == cluster and k * c >= capacity
                ),
                default=None,
            )
            if kmax is None:
                raise ValueError(
                    f"no split-cluster-{cluster} variant holds {capacity} blocks per query block"
                )
        kmax = int(kmax)
        if (kmax, cluster) not in SPLIT_CLUSTER_VARIANTS:
            raise ValueError(
                f"(kmax, split, cluster)=({kmax}, True, {cluster}) is not a compiled split-cluster variant {SPLIT_CLUSTER_VARIANTS}"
            )
        if kmax * cluster < capacity:
            raise ValueError(
                f"kmax={kmax} x {cluster} slices cannot hold {capacity} blocks per query block"
            )
        tiles = h * mb
        stride = kmax * cluster + PLAN_META_UNSPLIT
        if tiles * stride > PLAN_HALFWORDS:
            raise ValueError(
                f"{tiles} rows x {stride} halfwords exceed the {PLAN_HALFWORDS}-halfword plan parameter"
            )
        order = torch.argsort(~mask, dim=-1, stable=True).reshape(tiles, nb)
        rows = torch.full((tiles, stride), -1, dtype=torch.int16)
        for tile in range(tiles):
            count = flat_counts[tile]
            rows[tile, 0] = count
            rows[tile, 1 : 1 + count] = order[tile, :count].to(torch.int16)
        plan = torch.full((PLAN_HALFWORDS,), -1, dtype=torch.int16)
        plan[: tiles * stride] = rows.reshape(-1)
        nsplits = [max(1, -(-c // kmax)) for c in flat_counts]
        return {
            "plan": plan,
            "H": h,
            "MB": mb,
            "NB": nb,
            "capacity": capacity,
            "kmax": kmax,
            "num_tiles": tiles,
            "num_items": tiles * cluster,
            "max_nsplit": max(nsplits),
            "split": True,
            "cluster": cluster,
            "stride": stride,
            "meta_halfwords": PLAN_META_UNSPLIT,
        }
    if cluster:
        if kmax is None:
            kmax = min(
                (k for k, c in CLUSTER_VARIANTS if c == cluster and k * c >= capacity),
                default=None,
            )
            if kmax is None:
                raise ValueError(
                    f"no cluster-{cluster} variant holds {capacity} blocks per query block"
                )
        if (int(kmax), cluster) not in CLUSTER_VARIANTS:
            raise ValueError(
                f"(kmax, cluster)=({kmax}, {cluster}) is not a compiled cluster variant {CLUSTER_VARIANTS}"
            )
        if int(kmax) * cluster < capacity:
            raise ValueError(
                f"kmax={kmax} x {cluster} slices cannot hold {capacity} blocks per query block"
            )
    elif kmax is None:
        kmax = small_kmax_for(capacity) if not split else split_kmax(flat_counts)
    kmax = int(kmax)
    if kmax not in SMALL_KMAX_VARIANTS and not cluster:
        raise ValueError(
            f"kmax={kmax} is not a compiled small-kernel variant {SMALL_KMAX_VARIANTS}"
        )
    if split and kmax not in SPLIT_KMAX_VARIANTS:
        raise ValueError(
            f"kmax={kmax} is not a compiled split-KV variant {SPLIT_KMAX_VARIANTS}"
        )
    if not split and not cluster and kmax < capacity:
        raise ValueError(f"kmax={kmax} cannot hold {capacity} blocks per tile")
    tiles = h * mb
    if tiles > 32767:
        raise ValueError("query-block ids must fit int16")
    rowfmt = split or cluster > 0
    meta_halfwords = plan_meta(rowfmt)
    stride = kmax + meta_halfwords
    nsplits = (
        [cluster] * len(flat_counts)
        if cluster
        else [max(1, -(-c // kmax)) for c in flat_counts]
    )
    if max(nsplits) > MAX_NSPLIT:
        raise ValueError(
            f"a query block needs {max(nsplits)} slices; the plan encodes at most {MAX_NSPLIT}"
        )
    num_items = sum(nsplits)
    if num_items * stride > PLAN_HALFWORDS:
        raise ValueError(
            f"{num_items} items x {stride} halfwords exceed the {PLAN_HALFWORDS}-halfword plan parameter"
        )
    order = torch.argsort(~mask, dim=-1, stable=True).reshape(tiles, nb)
    rows = torch.full((num_items, stride), -1, dtype=torch.int16)
    item = 0
    for tile in range(tiles):
        ids = order[tile, : flat_counts[tile]].tolist()
        for j, chunk in enumerate(_balanced_chunks(ids, nsplits[tile])):
            if rowfmt:
                rows[item, 0] = len(chunk) | (j << 4) | (nsplits[tile] << 10)
                rows[item, 1] = tile
            else:
                rows[item, 0] = len(chunk)
            rows[item, meta_halfwords : meta_halfwords + len(chunk)] = torch.tensor(
                chunk, dtype=torch.int16
            )
            item += 1
    plan = torch.full((PLAN_HALFWORDS,), -1, dtype=torch.int16)
    plan[: num_items * stride] = rows.reshape(-1)
    return {
        "plan": plan,
        "H": h,
        "MB": mb,
        "NB": nb,
        "capacity": capacity,
        "kmax": kmax,
        "num_tiles": tiles,
        "num_items": num_items,
        "max_nsplit": max(nsplits),
        "split": bool(split) and max(nsplits) > 1,
        "cluster": cluster,
        "stride": stride,
        "meta_halfwords": meta_halfwords,
    }


def _order_union_rounds(a: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    sa, sb = set(a), set(b)
    i = j = 0
    c = [0, 0]
    emitted: set[int] = set()
    blocks: list[int] = []
    bits: list[int] = []
    while i < len(a) or j < len(b):
        while i < len(a) and a[i] in emitted:
            i += 1
        while j < len(b) and b[j] in emitted:
            j += 1
        if i >= len(a) and j >= len(b):
            break
        if (
            i < len(a)
            and j < len(b)
            and a[i] == b[j]
            or j >= len(b)
            or (i < len(a) and (c[0] < c[1] or (c[0] == c[1] and a[i] < b[j])))
        ):
            blk = a[i]
        else:
            blk = b[j]
        bt = (1 if blk in sa else 0) | (2 if blk in sb else 0)
        emitted.add(blk)
        blocks.append(blk)
        bits.append(bt)
        for w in (0, 1):
            if bt & (1 << w):
                c[w] += 1
    return blocks, bits


def _schedule_feasible(bits: list[int]) -> bool:
    """Exact simulation of the two consumer warpgroups and the two producer rings.

    ``bits[p]`` are the use bits of ring position ``p``.  Mirrors the kernel's
    program order: the K half of a position is released right after its QK
    completes (or, for a partner-only position, as its K tile lands), the V
    half after its PV completes, partner-only positions released in order,
    count-2 releases, and the two producer warps' in-order loads each gated on
    the previous use of its half of the stage.  False when the CTA would
    deadlock.
    """
    n_seq = len(bits)
    own = [
        [p for p, bt in enumerate(bits) if bt & (1 << w)]
        for w in range(NUM_CONSUMER_WGS)
    ]

    def program(w: int) -> list[tuple]:
        o = own[w]
        prog: list[tuple] = []
        prev = -1

        def visit(hi: int) -> None:
            nonlocal prev
            for p in range(prev + 1, hi):
                prog.append(("waitk", p))
                prog.append(("relk", p))
                prog.append(("waitv", p))
                prog.append(("relv", p))
            prev = max(prev, hi - 1)

        n = len(o)
        for j in range(n):
            visit(o[j])
            prog.append(("waitk", o[j]))
            if j >= 1:
                prog.append(("waitv", o[j - 1]))
            prog.append(("relk", o[j]))
            if j >= 1:
                prog.append(("relv", o[j - 1]))
            prev = o[j]
        if n >= 1:
            prog.append(("waitv", o[n - 1]))
            prog.append(("relv", o[n - 1]))
        visit(n_seq)
        prog.append(("done",))
        return prog

    progs = [program(w) for w in range(NUM_CONSUMER_WGS)]
    pc = [0] * NUM_CONSUMER_WGS
    relk = [0] * n_seq
    relv = [0] * n_seq
    loaded = {"k": 0, "v": 0}

    def step(w: int) -> bool:
        op = progs[w][pc[w]]
        if op[0] == "done":
            return False
        if op[0] == "waitk":
            if op[1] >= loaded["k"]:
                return False
        elif op[0] == "waitv":
            if op[1] >= loaded["v"]:
                return False
        elif op[0] == "relk":
            relk[op[1]] += 1
        elif op[0] == "relv":
            relv[op[1]] += 1
        pc[w] += 1
        return True

    while True:
        progressed = False
        for ring, rel in (("k", relk), ("v", relv)):
            depth = NUM_K_STAGES if ring == "k" else NUM_V_STAGES
            while loaded[ring] < n_seq and (
                loaded[ring] < depth or rel[loaded[ring] - depth] >= NUM_CONSUMER_WGS
            ):
                loaded[ring] += 1
                progressed = True
        for w in range(NUM_CONSUMER_WGS):
            while step(w):
                progressed = True
        if all(progs[w][pc[w]][0] == "done" for w in range(NUM_CONSUMER_WGS)):
            return True
        if not progressed:
            return False


def _tile_cost(owns_t) -> float:
    return max(len(o) for o in owns_t) + TILE_FIXED_COST


def _tile_load(kind: int, len_t: int) -> float:
    return len_t + TILE_FIXED_LOAD + (SPLIT_MERGE_LOAD if kind == 1 else 0.0)


GRID_MIN_FRACTION = (
    0.9  # a smaller, uniform persistent grid only while >= 90 % of the SMs stay busy
)
# Grid model (H100 SXM, forced-grid measurements on h7-m32768-k64): the per-SM
# stream rate is 3 % higher with at most 128 of the 132 SMs streaming and then
# rises only slowly (~(g/128) ** -0.25), so a CTA's load time is scaled
# accordingly and a uniform 128-CTA grid beats 132 CTAs carrying the same
# maximum tile count.  Mirrors ``vsa_sm90_bf16``.
GRID_MODEL = 1
# The refined load model (per-tile fixed load, split-merge load, grid knee credit) is
# calibrated on long tiles (>= 32 KV blocks per query block); on short tiles the pair plans
# it prefers measured 8-14 % slower than the split plans of the previous model, so below this
# mean number of KV blocks per query block the previous model (raw positions, flat grid) decides.
REFINED_LOAD_MIN_K = 28.0
GRID_RATE_FULL_PENALTY = 0.03
GRID_RATE_EXP = 0.25
GRID_RATE_KNEE = 128


def _grid_load_scale(g: int, sms: int) -> float:
    if GRID_MODEL == 0 or sms <= GRID_RATE_KNEE or g >= sms:
        return 1.0
    if g > GRID_RATE_KNEE:
        # Grids between the knee and the full machine measured 0.2-1.3 % slower
        # on the CuTe route when chosen for a sub-1 % model credit; only the
        # measured knee (<= 128 CTAs) takes the credit.
        return 1.0
    knee = 1.0 - GRID_RATE_FULL_PENALTY
    return knee * (g / GRID_RATE_KNEE) ** GRID_RATE_EXP


def _list_schedule(
    costs: list[float], lens: list[float], g: int, load_scale: float = 1.0
) -> tuple[list[list[int]], float]:
    """``load_scale`` is the grid's per-position load time relative to the full machine."""
    cons = [0.0] * g
    load = [0.0] * g
    heap = [(0.0, c) for c in range(g)]
    lists: list[list[int]] = [[] for _ in range(g)]
    for t in range(len(costs)):
        _, c = heapq.heappop(heap)
        lists[c].append(t)
        cons[c] += costs[t]
        load[c] += lens[t] * LOAD_COST * load_scale
        heapq.heappush(heap, (max(cons[c], load[c]), c))
    return lists, max(key for key, _ in heap)


def _grid_candidates(n_tiles: int, sms: int) -> list[int]:
    """Persistent grid sizes worth trying: the full machine and the largest
    grids that spread ``n_tiles`` uniformly (``ceil(n / k)`` CTAs for k tiles
    each) while keeping at least ``GRID_MIN_FRACTION`` of the SMs busy.
    Mirrors ``vsa_sm90_bf16._grid_candidates``."""
    g_max = max(1, min(n_tiles, sms))
    cands = {g_max}
    for k in range(1, n_tiles + 1):  # k tiles per CTA
        g = -(-n_tiles // k)
        if g < GRID_MIN_FRACTION * g_max:
            break
        if g <= g_max:
            cands.add(g)
        if g == 1:
            break
    return sorted(cands, reverse=True)


def _assign_tiles(
    costs: list[float], lens: list[float], sms: int, refined: bool = True
) -> tuple[list[list[int]], float]:
    """List-schedule the tiles onto ``g`` persistent CTAs.

    A CTA's key is the larger of its consumer time (sum of tile costs) and its
    load time (positions loaded x ``LOAD_COST``; the chip's memory bandwidth
    bounds a streaming iteration, so the term does not scale with the grid).
    The grid is chosen among :func:`_grid_candidates` by makespan; a tie goes
    to the grid with the most uniform tile counts (128 x 2 rather than
    124 x 2 + 8 x 1 for 256 tiles), then to the larger grid.  Mirrors
    ``vsa_sm90_bf16._assign_tiles``.
    """
    best = None
    max_full = None
    for g in _grid_candidates(len(costs), sms):  # descending: the full grid first
        lists, makespan = _list_schedule(
            costs, lens, g, _grid_load_scale(g, sms) if refined else 1.0
        )
        max_tiles = max(len(lst) for lst in lists)
        if max_full is None:
            max_full = max_tiles
        elif max_tiles > max_full:
            # only grids with the full grid's maximum tile count take the knee's rate credit
            continue
        spread = max_tiles - min(len(lst) for lst in lists)
        key = (round(makespan, 9), spread, -g)
        if best is None or key < best[0]:
            best = (key, lists, makespan)
    return best[1], best[2]


def _balance_static(
    infos,
    seqs,
    owns,
    lens,
    costs,
    loads,
    *,
    one_round: bool,
    sms: int,
    refined: bool,
    ragged: bool = False,
    fits_l2: bool = True,
):
    """A static plan's tiles in LPT order (by max(cost, load)) with their CTA lists and makespan, or
    None when the legacy order already is that order.  Uniform plans (UNIFORM_LPT): over all heads
    for a one-round plan, within each head otherwise.  Ragged plans (RAGGED_LPT_MAX_TILES): over all
    heads when every head's K and V fit the L2 budget, within each head beyond it.  Mirrors the
    kernel-tests reference planner (``_balance_static``)."""
    if (one_round and not ragged) or (ragged and fits_l2):
        order = sorted(
            range(len(infos)),
            key=lambda i: (
                -max(costs[i], loads[i] * LOAD_COST),
                infos[i][0],
                infos[i][1],
            ),
        )
    else:
        order = sorted(
            range(len(infos)),
            key=lambda i: (
                infos[i][0],
                -max(costs[i], loads[i] * LOAD_COST),
                infos[i][1],
            ),
        )
    if order == list(range(len(infos))):
        return None
    infos, seqs, owns, lens, costs, loads = (
        [x[i] for i in order] for x in (infos, seqs, owns, lens, costs, loads)
    )
    lists, makespan = _assign_tiles(costs, loads, sms, refined)
    return infos, seqs, owns, lens, costs, loads, lists, makespan


def _pairs_for_head(
    hh: int, mode: str, rows, counts, mb: int, ragged: bool
) -> list[tuple[int, int]]:
    if mode != "pair":
        return [(qb, qb) for qb in range(mb)]
    qbs = (
        sorted(range(mb), key=lambda i: -int(counts[hh, i]))
        if ragged
        else list(range(mb))
    )
    if mb > 1:
        sets = [set(rows[hh][qb]) for qb in range(mb)]
        used = [False] * mb
        pairs = []
        for i in range(mb):
            if used[i]:
                continue
            used[i] = True
            best, best_score = -1, None
            for j in range(i + 1, min(mb, i + 1 + PAIRING_WINDOW)):
                if used[j]:
                    continue
                score = (
                    len(sets[qbs[i]] & sets[qbs[j]]),
                    -abs(int(counts[hh, qbs[i]]) - int(counts[hh, qbs[j]])),
                    -j,
                )
                if best_score is None or score > best_score:
                    best, best_score = j, score
            if best < 0:
                pairs.append((qbs[i], qbs[i]))
            else:
                used[best] = True
                pairs.append((qbs[i], qbs[best]))
        return pairs
    return [
        (qbs[i], qbs[i + 1]) if i + 1 < mb else (qbs[i], qbs[i])
        for i in range(0, mb, 2)
    ]


def _pairs_of(lst):
    return [
        (lst[k], lst[k + 1] if k + 1 < len(lst) else -1) for k in range(0, len(lst), 2)
    ]


def _emit_tile(
    hh: int, qb: int, partner: int, plan_mode: str, rows, infos, seqs, owns, lens
) -> None:
    """Tile info word 3: 0 = pair tile, 1 = split tile, 2 = lone block of a pair plan."""
    if partner == qb:
        sps = _pairs_of(rows[hh][qb])
        if plan_mode == "pair":
            order = [(sp, 1) for sp in sps]
            infos.append((hh, qb, qb, 2))
        else:
            order = [(sp, 1 << (k % 2)) for k, sp in enumerate(sps)]
            infos.append((hh, qb, qb, 1))
    else:
        la, lb = rows[hh][qb], rows[hh][partner]
        order = []
        sset = set(la) & set(lb)
        ss = _pairs_of([b for b in la if b in sset])
        sa = _pairs_of([b for b in la if b not in sset])
        sb = _pairs_of([b for b in lb if b not in sset])
        for k in range(max(len(ss), len(sa), len(sb))):
            if k < len(ss):
                order.append((ss[k], 3))
            if k < len(sa):
                order.append((sa[k], 1))
            if k < len(sb):
                order.append((sb[k], 2))
        if not _schedule_feasible([bt for _, bt in order]):
            order = []
        if not order:
            sa, sb = _pairs_of(la), _pairs_of(lb)
            for k in range(max(len(sa), len(sb))):
                if k < len(sa):
                    order.append((sa[k], 1))
                if k < len(sb):
                    order.append((sb[k], 2))
        infos.append((hh, qb, partner, 0))
    if 2 * len(order) > MAX_SEQ:
        raise ValueError("KV sequence exceeds MAX_SEQ")
    bits = [bt for _, bt in order]
    if not _schedule_feasible(bits):
        raise AssertionError("ring schedule infeasible (kernel protocol bug)")
    seqs.append([b for (a, b2), _ in order for b in (a, b2)])
    owns.append(
        [
            [
                (pos << 3) | ((1 if sp[1] >= 0 else 0) << 2) | bt
                for pos, (sp, bt) in enumerate(order)
                if bt & (1 << w)
            ]
            for w in range(NUM_CONSUMER_WGS)
        ]
    )
    lens.append(len(order))


def _int32_words(words: torch.Tensor) -> torch.Tensor:
    """Unsigned 32-bit words held in int64 as the contiguous int32 storage the kernel reads (a bit reinterpretation)."""
    return (
        torch.where(words >= 1 << 31, words - (1 << 32), words)
        .to(torch.int32)
        .contiguous()
    )


def plan_vsa_sm90(
    block_mask: torch.Tensor, *, sms: int, mode: Optional[str] = None
) -> dict[str, Any]:
    """Turn a boolean ``[H, MB, NB]`` block mask into the kernel's per-CTA plan.

    Returns the CPU int32 plan ``meta[G * stride, META_WORDS]`` (row
    ``c * stride + i`` = CTA ``c``'s i-th tile; the first row of a CTA carries
    its tile count), the tile stride, the chosen mode, the persistent CTA count
    and the tile count.  ``sms`` is the multiprocessor count of the executing
    device (it sizes the persistent grid).
    """
    if block_mask.dtype != torch.bool or block_mask.ndim != 3:
        raise ValueError("block_mask must be a boolean [H, MB, NB] tensor")
    mask = block_mask.to(device="cpu").contiguous()
    h, mb, nb = mask.shape
    if h == 0 or mb == 0 or nb == 0:
        raise ValueError("block_mask must be non-empty")
    if nb > 0xFFFF:
        raise ValueError("at most 65535 KV blocks are supported")
    counts = mask.sum(dim=-1)
    if int(counts.min()) == 0:
        raise ValueError("empty sparse rows are not supported")
    if int(counts.max()) > MAX_SELECTED:
        raise ValueError(
            f"at most {MAX_SELECTED} KV blocks per query block are supported"
        )
    if mode is not None and mode not in {"pair", "split"}:
        raise ValueError(f"unknown mode {mode!r}")
    if sms < 1:
        raise ValueError("sms must be positive")
    rows = [[m.nonzero().flatten().tolist() for m in mask[hh]] for hh in range(h)]
    ragged = int(counts.min()) != int(counts.max())
    # The refined load model is calibrated on uniform plans; on ragged plans it
    # only re-orders and re-balances the list schedule and measured slower on
    # the 32k-token K=64 ragged rows, so ragged plans keep the previous model.
    refined = (not ragged) and float(counts.sum()) / (h * mb) >= REFINED_LOAD_MIN_K

    def build(mode: str):
        infos: list[tuple[int, int, int, int]] = []
        seqs: list[list[int]] = []
        owns: list[list[list[int]]] = []
        lens: list[int] = []
        for hh in range(h):
            pairs = _pairs_for_head(hh, mode, rows, counts, mb, ragged)
            for qb, partner in pairs:
                _emit_tile(hh, qb, partner, mode, rows, infos, seqs, owns, lens)
        costs = [_tile_cost(owns[t]) for t in range(len(infos))]
        loads = [
            _tile_load(infos[t][3], lens[t]) if refined else float(lens[t])
            for t in range(len(infos))
        ]
        if not ragged:
            order = list(range(len(infos)))
        elif h * nb * BLOCK * HEAD_DIM * 2 * 2 <= TILE_ORDER_L2_BUDGET:
            order = sorted(
                range(len(infos)),
                key=lambda i: (
                    -max(costs[i], loads[i] * LOAD_COST),
                    infos[i][0],
                    infos[i][1],
                ),
            )
        else:
            order = sorted(range(len(infos)), key=lambda i: (infos[i][0], -costs[i]))
        return tuple(
            [x[i] for i in order] for x in (infos, seqs, owns, lens, costs, loads)
        )

    if mode is None:
        best = None
        for cand in ("split", "pair"):  # ties go to split
            built = build(cand)
            _, makespan = _assign_tiles(built[4], built[5], sms, refined)
            if best is None or makespan < best[0]:
                best = (makespan, cand, built)
        _, mode, (infos, seqs, owns, lens, costs, loads) = best
    else:
        infos, seqs, owns, lens, costs, loads = build(mode)
    lists, makespan = _assign_tiles(costs, loads, sms, refined)
    g = len(lists)
    if g > HDR_CTAS:
        raise ValueError(
            f"persistent grid {g} exceeds the {HDR_CTAS}-CTA by-value header"
        )
    # Uniform plans with enough rounds run the queue kernel (``attention_queue``):
    # every CTA keeps its first tile and takes the rest from the queue in
    # head-major order, so the CTAs running concurrently stream one head's K/V
    # in lock step (the static lists drift apart) and the tail balances on
    # measured, not modelled, tile times.  Short lists keep the static kernel:
    # a round boundary's simultaneous fetches need several rounds to amortise.
    # Ragged plans join from POOL_RAGGED_MIN_TILES tiles per CTA, or from
    # POOL_RAGGED_LONG_HEAD_MIN_TILES when the plan holds POOL_RAGGED_LONG_HEAD_WINDOWS grids'
    # worth of tiles per head.
    min_tiles = min(len(lst) for lst in lists)
    if ragged:
        queue = min_tiles >= POOL_RAGGED_MIN_TILES or (
            min_tiles >= POOL_RAGGED_LONG_HEAD_MIN_TILES
            and len(infos) >= POOL_RAGGED_LONG_HEAD_WINDOWS * h * g
        )
    else:
        queue = min_tiles >= POOL_MIN_TILES or (
            POOL_SMALL_TILE_POS > 0
            and min_tiles >= POOL_SMALL_MIN_TILES
            and max(lens) <= POOL_SMALL_TILE_POS
        )
    if not queue and (
        max(len(lst) for lst in lists) <= RAGGED_LPT_MAX_TILES
        if ragged
        else UNIFORM_LPT
    ):
        # Static plan: LPT tile order; the queue
        # decision above rests on the legacy assignment, as in the reference planner.
        balanced = _balance_static(
            infos,
            seqs,
            owns,
            lens,
            costs,
            loads,
            one_round=len(infos) <= g,
            sms=sms,
            refined=refined,
            ragged=ragged,
            fits_l2=h * nb * BLOCK * HEAD_DIM * 2 * 2 <= TILE_ORDER_L2_BUDGET,
        )
        if balanced is not None:
            infos, seqs, owns, lens, costs, loads, lists, makespan = balanced
            g = len(lists)
            if g > HDR_CTAS:
                raise ValueError(
                    f"persistent grid {g} exceeds the {HDR_CTAS}-CTA by-value header"
                )
    # One static tile per CTA (the header's), the rest pooled (at most POOL_ROUNDS
    # per CTA) in the planner's tile order: head-major, query-block order within a
    # head (cost-descending within a head for ragged plans).
    pool_rounds = POOL_ROUNDS if queue else 0
    static = [lst[: max(1, len(lst) - pool_rounds)] for lst in lists]
    pool = sorted(
        t
        for c, lst in enumerate(lists)
        for i, t in enumerate(lst)
        if i >= len(static[c])
    )
    stride = max(len(st) for st in static)
    # Queue layout: static rows, the pool rows, then one all-zero sentinel row
    # per CTA (a fetch past the pool stages a slot whose word META_NTILES is 0).
    n_rows = g * stride + len(pool) + (g if queue else 0)
    hdr = torch.zeros((HDR_HALFWORDS,), dtype=torch.int16)
    for c, st in enumerate(static):
        t = st[0]
        n_hdr = min(HDR_ISSUE, lens[t])
        hdr[c * HDR_WORDS : c * HDR_WORDS + 5] = torch.tensor(
            (*infos[t], n_hdr), dtype=torch.int16
        )
        hdr[c * HDR_WORDS + 5 : c * HDR_WORDS + 5 + 2 * n_hdr] = torch.tensor(
            seqs[t][: 2 * n_hdr], dtype=torch.int16
        )
    # Plan rows laid out per CTA: row c * stride + i is CTA c's i-th (static)
    # tile; the first row of each CTA also carries its (static) tile count
    # (word META_NTILES).  Queue layout: the other static rows and the pool
    # rows carry 1 in that word (valid), the sentinel rows 0.
    # Packed 32-bit words are built in int64 (two 16-bit halves per word) and
    # reinterpreted as the kernel's int32 storage at the end (``_int32_words``).
    meta = torch.zeros((n_rows, META_WORDS), dtype=torch.int64)

    def fill(r, t, nt):
        meta[r, 0:4] = torch.tensor(infos[t], dtype=torch.int64)
        meta[r, META_NSEQ] = lens[t]
        sq = torch.tensor(seqs[t], dtype=torch.int64) & 0xFFFF
        meta[r, META_SEQ_OFF : META_SEQ_OFF + len(sq) // 2] = sq[0::2] | (
            sq[1::2] << 16
        )
        for w in range(NUM_CONSUMER_WGS):
            ol = owns[t][w]
            meta[r, META_NOWN + w] = len(ol)
            if len(ol) > MAX_OWN:
                raise ValueError("own list exceeds MAX_OWN")
            ow = torch.tensor(ol + [0] * (len(ol) & 1), dtype=torch.int64)
            base = META_OWN_OFF + w * OWN_WORDS
            meta[r, base : base + len(ow) // 2] = ow[0::2] | (ow[1::2] << 16)
        meta[r, META_NTILES] = nt

    for c, st in enumerate(static):
        for i, t in enumerate(st):
            fill(c * stride + i, t, len(st) if i == 0 else int(queue))
    for j, tp in enumerate(pool):
        fill(g * stride + j, tp, 1)
    return {
        "meta": _int32_words(meta),
        "hdr": hdr,
        "tile_stride": stride,
        "queue": queue,
        "pool": pool,
        "num_rows": n_rows,
        "mode": mode,
        "makespan": makespan,
        "num_ctas": g,
        "num_tiles": len(infos),
        "tile_lists": lists,
        "H": h,
        "MB": mb,
        "NB": nb,
    }
