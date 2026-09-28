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

Hopper (SM90) variable block-sparse attention on the generated Cake kernel.

``backend="cake"`` of :class:`flashinfer.sparse.VariableBlockSparseAttentionWrapper`
routes here on compute capability 9.0.  Supported scope: BF16 HND ``q/k/v`` with
shape ``(H, S, 128)``, equal query/KV head counts, noncausal 64-token query and
KV blocks, 1..64 selected KV blocks per query block (ragged counts allowed),
custom finite softmax scales (including 0 and negative), no LSE / PDL /
positional encoding / logits soft cap.

Two kernels serve the contract.  Small problems (at most 6 selected KV blocks
per query block, a grid that fits one wave, a plan that fits the kernel
parameter budget; :func:`small_route`) run the *small-selection* kernel: one
128-thread CTA per query block whose plan (``[meta, qtile, blk...]`` int16
halfwords, :func:`plan_small`) travels in the launch parameters, so no metadata
upload and no dependent global load precede the K/V loads.  Grids too small for
the persistent kernel's two warpgroups per CTA (``2 * tiles <= SMs``) take the
kernel's *split-KV* variant: a query block's selection is sliced across
``ceil(count / KMAX)`` CTAs (``meta = count | split << 4 | nsplit << 10``);
each slice publishes its FP32 accumulator and row statistics to a plan-owned
workspace and the last-arriving CTA merges them in slice order and resets the
arrival counter (:func:`split_kmax` picks the variant).  Uniform sliced
selections take the kernel's *cluster* variant instead when its modelled cost
is lower (:func:`cluster_variant_for`): the ``csize`` slices of a query block
form one thread-block cluster, every CTA publishes its FP32 partial and row
statistics into its own shared memory and, after one cluster barrier, each
rank merges one 32-column set through distributed shared memory (no
workspace, no atomics; padded slices carry ``count = 0``).  Everything else runs
the persistent kernel: the plan (:func:`plan_vsa_sm90`) is uploaded once as a
per-CTA list of *tiles* (``grid = min(tiles, SMs)``).  A tile is either one
64-row query block whose KV list alternates between the two consumer
warpgroups (``split``) or two query blocks of one head that share the KV ring
(``pair``); the mode with the smaller modelled makespan wins.  The plan layouts
mirror the Cake kernel modules (``vsa_sm90_bf16`` and
``vsa_sm90_small``) and are validated by the kernels' own tests; both
planners are dependency-free ports of the Cake planners and must stay
byte-identical to them.
"""

from __future__ import annotations

import functools
import heapq
import math
from typing import Any, Mapping, Optional

import numpy as np
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
META_SEQ_OFF = 8
META_SEQ_WORDS = MAX_SEQ // 2
MAX_OWN = MAX_SEQ // 2
OWN_WORDS = MAX_OWN // 2
META_OWN_OFF = META_SEQ_OFF + META_SEQ_WORDS
META_WORDS = META_OWN_OFF + NUM_CONSUMER_WGS * OWN_WORDS  # 144

# Small-selection route (port of the Cake kernel module ``vsa_sm90_small``).
SMALL_KMAX_VARIANTS = (1, 3, 4, 6)
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
# Cluster / DSM-merge route (port of the Cake cluster variants, CAKE-682):
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
# 31-32-tile problems into a second cluster wave.  The reference table below
# is that device's measured capacity, for tests only — planning resolves it.
REFERENCE_CLUSTER_CAPACITY_H100_SXM: dict[int, int] = {
    2: 66,
    3: 39,
    4: 30,
    6: 17,
    8: 15,
}
CLUSTER_MERGE_COST = 0.3
CLUSTER_PAD_COST = 0.15


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


def split_kmax(counts: list[int], *, sms: Optional[int] = None) -> int:
    """KMAX of the split-KV variant for the per-query-block selection ``counts``.

    Variants whose item count fits one wave (when ``sms`` is given) and whose
    plan fits the parameter bank are ranked by ``chain_blocks(kmax) * SPLIT_BLOCK_COST
    + (max_nsplit - 1) * SPLIT_MERGE_COST``; the cheapest wins (largest kmax on
    ties).  Raises when no variant fits.  Mirrors ``vsa_sm90_small.split_kmax``.
    """
    best = None
    for kmax in SMALL_KMAX_VARIANTS:
        nsplits = [max(1, -(-c // kmax)) for c in counts]
        items = sum(nsplits)
        if (
            max(nsplits) > MAX_NSPLIT
            or items * (kmax + PLAN_META_SPLIT) > PLAN_HALFWORDS
        ):
            continue
        if sms is not None and items > SMALL_OCCUPANCY[kmax] * sms:
            continue
        cost = (
            _chain_blocks(kmax) * SPLIT_BLOCK_COST
            + (max(nsplits) - 1) * SPLIT_MERGE_COST
        )
        if best is None or cost <= best[0]:
            best = (cost, kmax)
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
    sorted({csize for _kmax, csize in CLUSTER_VARIANTS})
)


@functools.lru_cache(maxsize=None)
def cluster_capacity(device_index: int) -> dict[int, int]:
    """``{cluster size: max co-resident clusters}`` of ``device_index`` at one CTA per SM.

    ``cuOccupancyMaxActiveClusters`` on an empty probe kernel launched with the
    device's maximum opt-in dynamic shared memory (one CTA per SM).  The answer
    follows the part's GPC topology after floorsweeping, not ``SMs // csize``:
    a 132-SM H100 SXM holds 66 two-CTA clusters but 30 four-CTA clusters.
    """
    from .cuda_utils import checkCudaErrors, driver, nvrtc

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
    cheaper one wins.  Everything larger goes to the persistent pair/split
    kernel.  Mirrors ``vsa_sm90_small.small_route``.
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
        if cluster is not None and (
            split is None or cluster_cost(counts, *cluster) < split_cost(counts, split)
        ):
            return cluster[0], False, cluster[1]
        if split is not None:
            return split, True, 0
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
        raise ValueError("cluster and split are exclusive routes")
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
GRID_RATE_FULL_PENALTY = 0.03
GRID_RATE_EXP = 0.25
GRID_RATE_KNEE = 128


def _grid_load_scale(g: int, sms: int) -> float:
    if GRID_MODEL == 0 or sms <= GRID_RATE_KNEE or g >= sms:
        return 1.0
    knee = 1.0 - GRID_RATE_FULL_PENALTY
    if g >= GRID_RATE_KNEE:
        return knee + (1.0 - knee) * (g - GRID_RATE_KNEE) / (sms - GRID_RATE_KNEE)
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
    costs: list[float], lens: list[float], sms: int
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
        lists, makespan = _list_schedule(costs, lens, g, _grid_load_scale(g, sms))
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
        loads = [_tile_load(infos[t][3], lens[t]) for t in range(len(infos))]
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
            _, makespan = _assign_tiles(built[4], built[5], sms)
            if best is None or makespan < best[0]:
                best = (makespan, cand, built)
        _, mode, (infos, seqs, owns, lens, costs, loads) = best
    else:
        infos, seqs, owns, lens, costs, loads = build(mode)
    lists, makespan = _assign_tiles(costs, loads, sms)
    g = len(lists)
    stride = max(len(lst) for lst in lists)
    if g > HDR_CTAS:
        raise ValueError(
            f"persistent grid {g} exceeds the {HDR_CTAS}-CTA by-value header"
        )
    hdr = np.zeros((HDR_HALFWORDS,), dtype=np.int16)
    for c, lst in enumerate(lists):
        t = lst[0]
        n_hdr = min(HDR_ISSUE, lens[t])
        hdr[c * HDR_WORDS : c * HDR_WORDS + 5] = (*infos[t], n_hdr)
        hdr[c * HDR_WORDS + 5 : c * HDR_WORDS + 5 + 2 * n_hdr] = seqs[t][: 2 * n_hdr]
    meta = np.zeros((g * stride, META_WORDS), dtype=np.uint32)
    for c, lst in enumerate(lists):
        for i, t in enumerate(lst):
            r = c * stride + i
            meta[r, 0:4] = infos[t]
            meta[r, META_NSEQ] = lens[t]
            sq = np.asarray(seqs[t], dtype=np.int64) & 0xFFFF
            meta[r, META_SEQ_OFF : META_SEQ_OFF + len(sq) // 2] = sq[0::2] | (
                sq[1::2] << 16
            )
            for w in range(NUM_CONSUMER_WGS):
                ol = owns[t][w]
                meta[r, META_NOWN + w] = len(ol)
                if len(ol) > MAX_OWN:
                    raise ValueError("own list exceeds MAX_OWN")
                ow = np.asarray(ol + [0] * (len(ol) & 1), dtype=np.int64)
                base = META_OWN_OFF + w * OWN_WORDS
                meta[r, base : base + len(ow) // 2] = ow[0::2] | (ow[1::2] << 16)
        meta[c * stride, META_NTILES] = len(lst)
    return {
        "meta": torch.from_numpy(meta.view(np.int32).copy()),
        "hdr": torch.from_numpy(hdr),
        "tile_stride": stride,
        "mode": mode,
        "makespan": makespan,
        "num_ctas": g,
        "num_tiles": len(infos),
        "tile_lists": lists,
        "H": h,
        "MB": mb,
        "NB": nb,
    }


# ---------------------------------------------------------------------------
# Wrapper-facing plan object
# ---------------------------------------------------------------------------


def _check_descriptors(block_mask_map, block_row_sz, block_col_sz, num_qo_heads):
    if (
        not isinstance(block_mask_map, torch.Tensor)
        or block_mask_map.dtype != torch.bool
    ):
        raise ValueError("cake (SM90) requires a boolean block_mask_map")
    if block_mask_map.ndim != 3:
        raise ValueError("block_mask_map must have shape (num_heads, MB, NB)")
    h, mb, nb = block_mask_map.shape
    if h != num_qo_heads:
        raise ValueError(
            "cake (SM90) requires one sparse pattern per head: block_mask_map.shape[0] "
            f"({h}) must equal num_qo_heads ({num_qo_heads})"
        )
    for name, sizes, count in (
        ("block_row_sz", block_row_sz, mb),
        ("block_col_sz", block_col_sz, nb),
    ):
        if not isinstance(sizes, torch.Tensor) or tuple(sizes.shape) != (h, count):
            raise ValueError(f"{name} must have shape ({h}, {count})")
        if not bool((sizes.to("cpu") == BLOCK).all()):
            raise ValueError(f"cake (SM90) requires {BLOCK}-token blocks in {name}")


class CakeVsaSm90Plan:
    """Wrapper-owned plan: uploaded tile metadata plus the launch geometry.

    ``run`` issues exactly one kernel launch on the current stream (after an
    event wait on the metadata upload), so a fixed plan is CUDA-graph
    capturable; ``plan()`` may not replace a plan that a graph has captured.
    """

    def __init__(
        self,
        device: torch.device,
        block_mask_map: torch.Tensor,
        block_row_sz: torch.Tensor,
        block_col_sz: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: Optional[torch.dtype] = None,
        non_blocking: bool = True,
        route: Optional[str] = None,
        engine: str = "cuda",
    ):
        """``route`` ``None`` applies :func:`small_route`; ``"small"`` / ``"smallsplit"`` / ``"smallcluster"`` / ``"persistent"`` force one kernel.

        ``engine`` selects the kernel build: ``"cuda"`` (generated CUDA C++,
        ``backend="cake"``) or ``"cute"`` (the same kernels rendered as CuTe DSL
        modules, ``backend="cake_cute"``; see
        :mod:`flashinfer.experimental.cake_vsa_sm90_cute`).  Plans, routes and
        launch geometry are identical; both builds take the plan / header
        tables by value (the CuTe build as 8-byte scalar launch arguments read
        from the kernel parameter space).
        """
        if engine not in ("cuda", "cute"):
            raise ValueError(f"unknown engine {engine!r}; expected 'cuda' or 'cute'")
        self.engine = engine
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("cake (SM90) requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (9, 0):
            raise ValueError("cake (SM90) VSA requires Hopper compute capability 9.0")
        if causal:
            raise ValueError("cake (SM90) VSA supports noncausal attention only")
        if pos_encoding_mode != "NONE":
            raise ValueError("cake (SM90) VSA does not apply positional encodings")
        if use_fp16_qk_reduction:
            raise ValueError("cake (SM90) VSA accumulates QK in FP32 only")
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise ValueError("cake (SM90) VSA does not support logits_soft_cap")
        if q_data_type != torch.bfloat16 or (
            kv_data_type is not None and kv_data_type != torch.bfloat16
        ):
            raise ValueError("cake (SM90) VSA supports BF16 q/k/v only")
        if num_qo_heads != num_kv_heads or num_qo_heads < 1:
            raise ValueError(
                "cake (SM90) VSA requires equal, positive query/KV head counts"
            )
        if head_dim != HEAD_DIM:
            raise ValueError(f"cake (SM90) VSA requires head_dim {HEAD_DIM}")
        _check_descriptors(block_mask_map, block_row_sz, block_col_sz, num_qo_heads)
        scale = 1.0 / math.sqrt(HEAD_DIM) if sm_scale is None else float(sm_scale)
        # The kernel consumes scale * log2(e) as FP32.
        if not math.isfinite(scale * LOG2E) or abs(scale * LOG2E) > _FP32_MAX:
            raise ValueError("sm_scale must be finite in FP32")
        self.sm_scale = scale
        self.scale_log2 = scale * LOG2E
        props = torch.cuda.get_device_properties(self.device)
        sms = int(props.multi_processor_count)
        if route not in (None, "small", "smallsplit", "smallcluster", "persistent"):
            raise ValueError(
                f"unknown route {route!r}; expected None, 'small', 'smallsplit', 'smallcluster' or 'persistent'"
            )
        small: Optional[tuple[int, bool, int]] = None
        if route is None:
            small = small_route(
                block_mask_map,
                sms=sms,
                cluster_capacity=cluster_capacity(self.device.index),
            )
        elif route == "small":
            small = (
                small_kmax_for(int(block_mask_map.to("cpu").sum(dim=-1).max())),
                False,
                0,
            )
        elif route == "smallsplit":
            counts = block_mask_map.to("cpu").sum(dim=-1).reshape(-1).tolist()
            small = (split_kmax(counts), True, 0)
        elif route == "smallcluster":
            counts = block_mask_map.to("cpu").sum(dim=-1).reshape(-1).tolist()
            variant = cluster_variant_for(counts)
            if variant is None:
                raise ValueError("no cluster variant holds this selection")
            small = (variant[0], False, variant[1])
        self.captured = False
        self.small_kmax: Optional[int] = None if small is None else small[0]
        self.small_split = False
        self.small_cluster = 0
        if small is not None:
            plan = plan_small(
                block_mask_map, kmax=small[0], split=small[1], cluster=small[2]
            )
            self.small_split = bool(plan["split"])
            self.small_cluster = int(plan["cluster"])
            self.mode = (
                "smallcluster"
                if self.small_cluster
                else ("smallsplit" if self.small_split else "small")
            )
            self.num_tiles = plan["num_tiles"]
            self.num_items = plan["num_items"]
            self.num_ctas = plan["num_items"]
            self.tile_stride = 0
            self.num_heads = plan["H"]
            self.qo_len = plan["MB"] * BLOCK
            self.kv_len = plan["NB"] * BLOCK
            # The plan is a launch parameter (CPU int16): nothing is uploaded,
            # so an unsplit or cluster plan has no stream dependency and a
            # captured graph replays it.  A split plan owns device workspace
            # whose arrival counters must be zero before the first run
            # (``ready``); the kernel leaves them zero, so replays need no host
            # reset.  One run per split plan may be in flight at a time.
            # Both builds take the plan by value (the CuTe DSL build as 8-byte
            # scalar launch arguments read from the kernel parameter space).
            self.plan_param = plan["plan"].contiguous()
            if self.small_split:
                with torch.cuda.device(self.device):
                    self.partial_o = torch.empty(
                        (self.num_items * SMALL_ITEM_ELEMS,),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.partial_stats = torch.empty(
                        (self.num_items * SMALL_STATS_FLOATS,),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.counters = torch.zeros(
                        (self.num_tiles,), dtype=torch.int32, device=self.device
                    )
                    # The rendered binding takes ``u32`` arrival counters; keep the
                    # int32 storage (uint32 lacks most reductions) and pass a view.
                    self.counters_u32 = self.counters.view(torch.uint32)
                    self.ready = torch.cuda.Event(external=True)
                    self.ready.record(torch.cuda.current_stream(self.device))
            return
        plan = plan_vsa_sm90(block_mask_map, sms=sms)
        self.mode = plan["mode"]
        self.num_ctas = plan["num_ctas"]
        self.num_tiles = plan["num_tiles"]
        self.tile_stride = plan["tile_stride"]
        self.num_heads = plan["H"]
        self.qo_len = plan["MB"] * BLOCK
        self.kv_len = plan["NB"] * BLOCK
        with torch.cuda.device(self.device):
            host_meta = plan["meta"]
            if non_blocking:
                host_meta = host_meta.pin_memory()
            self.meta = host_meta.to(self.device, non_blocking=non_blocking)
            # By-value kernel parameter in both builds: stays on the host.
            self.hdr = plan["hdr"].contiguous()
            # The kernel's debug/timeline buffers are inert in the exported
            # build; they only have to exist on the device.
            self.dbg = torch.zeros((64,), dtype=torch.int32, device=self.device)
            self.tl = torch.zeros((8,), dtype=torch.uint64, device=self.device)
            # Capture must retain an external wait node for the metadata
            # upload, which was recorded outside the captured stream.
            self.ready = torch.cuda.Event(external=True)
            self.ready.record(torch.cuda.current_stream(self.device))
        self._host_meta = host_meta  # keeps pinned storage alive for the copy

    def check_replan(self) -> None:
        if self.captured:
            raise RuntimeError(
                "A captured VSA plan cannot be replaced; create a new wrapper and "
                "keep the original wrapper alive while its graph may replay"
            )

    def _check_tensor(self, name, tensor, shape) -> None:
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"{name} must have shape {tuple(shape)} and dtype bfloat16"
            )
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous on the planned device {self.device}"
            )

    @staticmethod
    def _aligned(tensor: torch.Tensor) -> torch.Tensor:
        # TMA needs 16-byte aligned bases; an offset view is copied on the
        # current stream (graph-capturable).
        if tensor.data_ptr() % 16 == 0:
            return tensor
        return tensor.clone(memory_format=torch.contiguous_format)

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        if return_lse or lse is not None:
            raise ValueError("cake (SM90) VSA does not support log-sum-exp outputs")
        if enable_pdl:
            raise ValueError("cake (SM90) VSA does not support PDL")
        for name, tensor, length in (
            ("q", q, self.qo_len),
            ("k", k, self.kv_len),
            ("v", v, self.kv_len),
        ):
            self._check_tensor(name, tensor, (self.num_heads, length, HEAD_DIM))
        # Preserve VariableBlockSparseAttentionWrapper's existing DPS ABI: the
        # provided output is [H*M, 1, D], while the return value is HND.
        if out is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires a preallocated out buffer"
                )
            result = torch.empty_like(q)
        else:
            self._check_tensor("out", out, (self.num_heads * self.qo_len, 1, HEAD_DIM))
            result = out.view(self.num_heads, self.qo_len, HEAD_DIM)
            begin = out.data_ptr()
            end = begin + out.numel() * out.element_size()
            for tensor in (q, k, v):
                tensor_begin = tensor.data_ptr()
                tensor_end = tensor_begin + tensor.numel() * tensor.element_size()
                if begin < tensor_end and tensor_begin < end:
                    raise ValueError("out must not overlap Q/K/V storage")

        if self.small_kmax is None:
            stage = "attention"
        elif self.small_cluster:
            stage = f"small_k{self.small_kmax}c{self.small_cluster}"
        else:
            stage = f"small_k{self.small_kmax}{'s' if self.small_split else ''}"
        if self.engine == "cute":
            from .experimental.cake_vsa_sm90_cute import load_stage

            cute_stage = load_stage(stage)
            module = None
        else:
            from .jit.cake_vsa_sm90 import load_cake_vsa_sm90_module

            module, _record = load_cake_vsa_sm90_module(stage)
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if torch.cuda.is_current_stream_capturing():
                self.captured = True
            if self.small_kmax is None or self.small_split:
                # The device meta table (persistent route) or the split
                # workspace may have been built on another stream: an event
                # wait enqueues the dependency without a CPU synchronization.
                # Unsplit / cluster plans are by-value launch arguments in
                # both builds and need no wait.
                stream.wait_event(self.ready)
            aligned_q = self._aligned(q)
            aligned_k = self._aligned(k)
            aligned_v = self._aligned(v)
            target = result if result.data_ptr() % 16 == 0 else torch.empty_like(result)
            if self.engine == "cute":
                self._run_cute(cute_stage, aligned_q, aligned_k, aligned_v, target)
            else:
                self._run_cuda(module, aligned_q, aligned_k, aligned_v, target)
            if target is not result:
                result.copy_(target)
        return result

    def _run_cute(self, cute_stage, aligned_q, aligned_k, aligned_v, target) -> None:
        """Launch the CuTe DSL build with the same plan rows and grid as the CUDA build."""
        bindings = {
            "Q": aligned_q,
            "K": aligned_k,
            "Vt": aligned_v,
            "O": target,
            "seqlen_q": int(self.qo_len),
            "seqlen_k": int(self.kv_len),
            "scale_log2": float(self.scale_log2),
        }
        if self.small_kmax is not None:
            bindings["plan"] = self.plan_param
            if self.small_split:
                bindings["Wo"] = self.partial_o
                bindings["Ws"] = self.partial_stats
                bindings["Wc"] = self.counters_u32
            grid = (int(self.num_items), 1, 1)
        else:
            bindings["meta"] = self.meta
            bindings["hdr"] = self.hdr
            bindings["tile_stride"] = int(self.tile_stride)
            bindings["dbg"] = self.dbg
            bindings["tl"] = self.tl
            grid = (int(self.num_ctas), 1, 1)
        cute_stage.run(bindings, grid)

    def _run_cuda(self, module, aligned_q, aligned_k, aligned_v, target) -> None:
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            if self.small_split:
                module.run(
                    aligned_q,
                    aligned_k,
                    aligned_v,
                    target,
                    self.plan_param,
                    int(self.qo_len),
                    int(self.kv_len),
                    float(self.scale_log2),
                    self.partial_o,
                    self.partial_stats,
                    self.counters_u32,
                    int(self.num_items),
                    1,
                    1,
                )
            elif self.small_kmax is not None:
                module.run(
                    aligned_q,
                    aligned_k,
                    aligned_v,
                    target,
                    self.plan_param,
                    int(self.qo_len),
                    int(self.kv_len),
                    float(self.scale_log2),
                    int(self.num_items),
                    1,
                    1,
                )
            else:
                module.run(
                    aligned_q,
                    aligned_k,
                    aligned_v,
                    target,
                    self.meta,
                    self.hdr,
                    int(self.tile_stride),
                    int(self.qo_len),
                    int(self.kv_len),
                    float(self.scale_log2),
                    self.dbg,
                    self.tl,
                    int(self.num_ctas),
                    1,
                    1,
                )


def create_plan(device, *args, **kwargs) -> CakeVsaSm90Plan:
    """Validate the request and build the uploaded plan (outside graph capture)."""
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("cake (SM90) VSA requires Hopper compute capability 9.0")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Call plan() outside CUDA graph capture")
        return CakeVsaSm90Plan(device, *args, **kwargs)


__all__ = [
    "CakeVsaSm90Plan",
    "cluster_capacity",
    "cluster_variant_for",
    "create_plan",
    "plan_small",
    "plan_vsa_sm90",
    "small_route",
    "split_kmax",
]
