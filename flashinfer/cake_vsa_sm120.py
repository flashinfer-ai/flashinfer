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

Host side of ``BlockSparseAttentionWrapper(backend="vsa_sm120_blk64")``: the plan
of the persistent SM120 block-64 programs of ``flashinfer.jit.cake_vsa_sm120``
and their launch.  Two kernels run a plan over the same two virtual CTAs per
SM: ``vsa_sm120_blk64`` (two 128-thread CTAs per SM, a virtual CTA is a CTA)
and ``vsa_sm120_blk64_pair`` (one CTA per SM of two 128-thread groups, a
virtual CTA is a group); their outputs are bitwise equal.

The plan keeps every (head, query block) row's KV-block list in one flat int32
index (a BSR selection shared by every head is stored once) and turns the rows
into work items of eight int32: ``head, query block, first, end`` (the item's
range of index positions), then ``group, parts, part, slot`` of a row split into
``parts`` items (``group = -1`` for a row that is one item).  A row longer than
the part size is split into near-equal parts; the items are ordered longest
first (a split part before a whole row of its size), then by (KV head, query
block, head, part), so neighbouring query blocks of one KV head run together.
The part size and the kernel are the pair of least time under the plan's cost
model: the static grid-stride's longest virtual CTA when every one runs one
item, else its longest SM, two virtual CTAs sharing one, plus the merge of the
most-split row, at the kernel's rates; a split part costs more once the K and
V of all KV heads pass half the L2 (three eighths of it for a plan that may be
packed).  Below three rows per virtual CTA, and
while the K and V of all KV heads fit the L2 (a quarter of it below one row per
virtual CTA), the plan may instead be packed: the items of every full wave keep
their places and those of the last partial wave fill the virtual CTAs one after
another up to the least common load that holds them, a row cut where a virtual
CTA fills up (parts of any size), the lists then evened to two adjacent
lengths, longest first, for the grid-stride.  Each kernel's packed plan
replaces the uniform one when it is faster under the packing constants.  A
split row's parts leave their partial states in the plan's workspace and the
last one to arrive merges them in part order and rewinds its counter, so a plan
is replayed without a reset.  One plan's workspace serves one launch at a time.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Optional

import torch

HEAD_DIM = 128
BLOCK_SIZE = 64
# The programs' fixed shape: eight int32 per work item, two virtual CTAs (a
# 128-thread group each) per SM, and the partial (O, m, d) state of one split
# part: 64 + 2 + 2 floats per thread of a group.  Per kernel, the virtual CTAs
# of one of its CTAs.
_WORK_WORDS = 8
_GROUP_THREADS = 128
_PARTIAL_FLOATS = 68 * _GROUP_THREADS
_VIRTUAL_PER_SM = 2
_GROUPS = {"vsa_sm120_blk64": 1, "vsa_sm120_blk64_pair": 2}
_MIN_PART = 2
# The plan's cost model, in the time a virtual CTA of vsa_sm120_blk64 that
# shares its SM takes for one KV block.  Per kernel: an item's own cost (its Q
# tile, the first K / V latency and the epilogue), the rate of a virtual CTA
# alone on its SM in tenths of its own shared time per block, and that shared
# time per block in hundredths of the unit.  For both: a split part's extra
# when the items outnumber the virtual CTAs (its partial state out and the
# merge it may run before its next item) and the merge, per part of the
# most-split row.  The candidate part sizes are the longest row divided by
# _PART_DIVISORS, at least _MIN_PART.
_COST = {"vsa_sm120_blk64": (2, 6, 100), "vsa_sm120_blk64_pair": (3, 8, 94)}
_SPLIT_COST = 2
_MERGE_COST = 1
# Past _SPLIT_KV_BLOCKS blocks of K and V over all KV heads (64 MB in 16 bits,
# half of an RTX PRO 6000's L2) a split part's extra is _FAR_SPLIT_COST: the
# blocks then come from memory, and a row split to fill the virtual CTAs gains
# less than when they come from the L2.  For a plan that may be packed (below)
# the limit is _PACK_SPLIT_KV_BLOCKS (48 MB, three eighths of the L2): the
# packing fills the last partial wave itself, so splitting every row to fill
# the virtual CTAs gains less.
_SPLIT_KV_BLOCKS = 2048
_PACK_SPLIT_KV_BLOCKS = 1536
_FAR_SPLIT_COST = 4
_PART_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
# Packing, when there are fewer rows than _PACK_ROWS per virtual CTA and the K
# and V of all KV heads span at most _PACK_KV_BLOCKS blocks: the first limit
# with fewer rows than virtual CTAs (32 MB in 16 bits, a quarter of an RTX PRO
# 6000's L2), the second otherwise (128 MB, all of it); beyond them the packed
# plan, which cuts rows over every virtual CTA, runs slower than the uniform
# one.  An item's own cost and the alone rate in tenths per kernel, the split
# cost, no merge term, at _COST's rates, for comparing a packed plan with the
# uniform one.  Fit on 118 selections (shared BSR, per-head top-k, window and
# prefix, 2-46 heads, 8-128 blocks per row, 0.5-5.5 waves) on an RTX PRO 6000:
# the chosen plans run 6.7% faster than the uniform ones (geometric mean) and
# 0.24% slower than the best of the four (uniform or packed, on either kernel).
# _COST's item costs and _SPLIT_COST size the packing itself.
_PACK_COST = {"vsa_sm120_blk64": (3, 8), "vsa_sm120_blk64_pair": (5, 9)}
_PACK_SPLIT_COST = 1
_PACK_ROWS = 3
_PACK_KV_BLOCKS = (1024, 4096)
_MIN_PIECE = 2

# The generated binding's host parameters, in the kernel's order (then the grid).
_HOST_PARAMS = (
    "q_map",
    "k_map",
    "v_map",
    "out",
    "lse",
    "partial_ws",
    "split_counters",
    "q2k_index",
    "work",
    "num_heads",
    "group_size",
    "total_work",
    "write_lse",
    "softmax_scale_log2",
)


def _ceil_div(a: torch.Tensor, b: int) -> torch.Tensor:
    return torch.div(a + (b - 1), b, rounding_mode="floor")


def _split(row_count: torch.Tensor, part: int):
    """Rows of ``row_count`` blocks split at ``part``: each row's part count, and
    per item its row, its index in the row and its block count."""
    row_parts = torch.clamp(_ceil_div(row_count, part), min=1)
    item_row = torch.repeat_interleave(
        torch.arange(row_parts.numel(), device=row_count.device), row_parts
    )
    item_start = torch.cumsum(row_parts, 0) - row_parts
    item_part = (
        torch.arange(item_row.numel(), device=row_count.device) - item_start[item_row]
    )
    count = row_count[item_row]
    parts = row_parts[item_row]
    size = (
        torch.div(count, parts, rounding_mode="floor")
        + (item_part < count % parts).long()
    )
    return row_parts, item_row, item_part, size


def _makespan(
    sizes: torch.Tensor,
    rows: torch.Tensor,
    part: int,
    resident: int,
    kernel: str,
    split_cost: int = _SPLIT_COST,
) -> int:
    """The cost model's time on ``kernel``, in thousandths, of ``rows[i]`` rows
    of ``sizes[i]`` blocks (CPU int64) split at ``part``, a split part's extra
    ``split_cost`` when the items outnumber the virtual CTAs.  The items run
    longest first (a split part before a whole row of its size); virtual CTA c
    of G = min(items, resident) takes items c, c + G, ..., its load the sum of
    their costs.  One wave (an item per virtual CTA) takes its longest item;
    more items take the longest SM, which runs virtual CTAs s and
    s + resident / 2 together for the shorter load and the longer one alone for
    the rest.  Items of one size and kind cost the same wherever they sit in
    their run, so the loads need no list of the items."""
    item_cost, lone_rate, rate = _COST[kernel]
    parts = torch.clamp(-(-sizes // part), min=1)
    base = sizes // parts
    extra = sizes - base * parts
    split = (parts > 1).long()
    # The runs of one item size and kind, longest first: key 2 * size + split.
    key, run = torch.unique(
        torch.cat((2 * base + 2 + split, 2 * base + split)), return_inverse=True
    )
    number = key.new_zeros(key.numel()).index_add_(
        0, run, torch.cat((extra * rows, (parts - extra) * rows))
    )
    key, number = key.flip(0), number.flip(0)
    end = torch.cumsum(number, 0)
    start = end - number
    items = int(end[-1])
    grid = max(1, min(items, resident))
    wave = items <= resident
    cost = key // 2 + item_cost + (0 if wave else split_cost) * (key % 2)
    # CTA c holds end // G - start // G items of a run, one more where
    # start % G <= c < end % G, one less where end % G <= c < start % G.
    load = (
        cost.new_zeros(grid)
        .index_add_(0, start % grid, cost)
        .index_add_(0, end % grid, -cost)
    )
    load = torch.cumsum(load, 0) + int((cost * (end // grid - start // grid)).sum())
    if wave:
        span = 10 * int(load.max())
    else:
        sms = resident // _VIRTUAL_PER_SM
        both = torch.minimum(load[:sms], load[sms:])
        alone = torch.maximum(load[:sms], load[sms:]) - both
        span = int((10 * both + lone_rate * alone).max())
    most = int(parts.max())
    return rate * (span + (10 * _MERGE_COST * most if most > 1 else 0))


def _choose(
    row_count: torch.Tensor,
    resident: int,
    part: Optional[int] = None,
    kernel: Optional[str] = None,
    split_cost: int = _SPLIT_COST,
) -> tuple[int, str]:
    """The (part size, kernel) of least _makespan at ``split_cost``, the part
    among the longest row divided by _PART_DIVISORS (the larger part, then
    vsa_sm120_blk64, on a tie); a given ``part`` or ``kernel`` is kept."""
    sizes, rows = torch.unique(row_count.cpu().long(), return_counts=True)
    longest = int(sizes[-1]) if sizes.numel() else 0
    if part is not None:
        parts = [part]
    elif longest <= _MIN_PART:
        parts = [max(_MIN_PART, longest)]
    else:
        parts = sorted(
            {max(_MIN_PART, -(-longest // k)) for k in _PART_DIVISORS}, reverse=True
        )
    kernels = [kernel] if kernel is not None else list(_COST)
    if not sizes.numel():  # no row (no query block): nothing runs
        return parts[0], kernels[0]
    return min(
        ((p, k) for p in parts for k in kernels),
        key=lambda pk: (
            _makespan(sizes, rows, pk[0], resident, pk[1], split_cost),
            -pk[0],
            kernels.index(pk[1]),
        ),
    )


def _table(
    row_count: torch.Tensor,
    row_first: torch.Tensor,
    row_head: torch.Tensor,
    row_block: torch.Tensor,
    part: int,
) -> tuple[torch.Tensor, int, int]:
    """The uniform work table of the rows split at ``part`` (int64, in launch
    order), its split groups and its partial slots."""
    row_parts, item_row, item_part, size = _split(row_count, part)
    split = row_parts > 1
    row_group = torch.where(split, torch.cumsum(split.long(), 0) - 1, -1)
    split_parts = torch.where(split, row_parts, 0)
    row_slot = torch.cumsum(split_parts, 0) - split_parts
    parts = row_parts[item_row]
    count = row_count[item_row]
    base = torch.div(count, parts, rounding_mode="floor")
    extra = count % parts
    begin = row_first[item_row] + item_part * base + torch.minimum(item_part, extra)
    is_split = split[item_row]
    work = torch.stack(
        (
            row_head[item_row],
            row_block[item_row],
            begin,
            begin + size,
            row_group[item_row],
            parts,
            torch.where(is_split, item_part, 0),
            torch.where(is_split, row_slot[item_row] + item_part, 0),
        ),
        dim=1,
    )
    # Longest first, a split part before a whole row of its size.
    item_order = torch.sort(-(2 * size + is_split.long()), stable=True).indices
    return work[item_order], int(split.sum()), int(split_parts.sum())


def _pack(
    work: list[list[int]], resident: int, item_cost: int
) -> Optional[tuple[list[list[int]], int, int]]:
    """The packed plan of the uniform ``work`` (its rows of eight words in
    launch order): its rows, split groups and partial slots, or None.  The
    items of every full wave stay on their virtual CTAs (so that the rows in
    flight together stay those of the uniform plan); the last partial wave's
    fill virtual CTA 0, 1, ... up to the least capacity that holds them (an
    item's blocks, its own cost and, once its row is cut, the split cost), an
    item cut where a virtual CTA fills up when at least _MIN_PIECE blocks fit.
    The grid-stride needs list lengths L + 1 then L: a list two or more short
    has its largest piece halved, or takes the last item of the longest list."""
    held = len(work) // resident * resident
    lists: list[list[list]] = [[] for _ in range(resident)]
    for p in range(held):
        w = work[p]
        lists[p % resident].append([w[0], w[1], w[2], w[3], w[4] >= 0])
    tail = [[w[0], w[1], w[2], w[3], w[4] >= 0] for w in work[held:]]
    if not tail:
        return None

    def cost(e: list) -> int:
        return e[3] - e[2] + item_cost + _SPLIT_COST * e[4]

    base = [sum(cost(e) for e in lst) for lst in lists]

    def fill(capacity: int) -> Optional[list[list[list]]]:
        out: list[list[list]] = [[] for _ in range(resident)]
        used = base[:]
        v = 0
        for head, block, begin, end, was_split in tail:
            cut = False
            while True:
                if v >= resident:
                    return None
                rest = end - begin
                need = rest + item_cost + _SPLIT_COST * (was_split or cut)
                if used[v] + need <= capacity:
                    out[v].append([head, block, begin, end, was_split or cut])
                    used[v] += need
                    break
                piece = capacity - used[v] - item_cost - _SPLIT_COST
                if piece >= _MIN_PIECE and rest - piece >= 1:
                    out[v].append([head, block, begin, begin + piece, True])
                    begin += piece
                    cut = True
                v += 1
        return out

    total = sum(base) + sum(cost(e) for e in tail)
    low, high = max(max(base), -(-total // resident)), max(base) + total
    best = fill(high)
    if best is None:
        return None
    while low < high:
        mid = (low + high) // 2
        got = fill(mid)
        if got is None:
            low = mid + 1
        else:
            high, best = mid, got
    full = [lists[v] + best[v] for v in range(resident)]
    for _ in range(4 * resident):
        n = [len(lst) for lst in full]
        longest = max(n)
        short = [v for v in range(resident) if n[v] < longest - 1]
        if not short:
            break
        v = short[0]
        big = max(
            range(len(full[v])),
            key=lambda j: full[v][j][3] - full[v][j][2],
            default=None,
        )
        if big is not None and full[v][big][3] - full[v][big][2] >= 2:
            head, block, begin, end, _ = full[v][big]
            mid = begin + (end - begin) // 2
            full[v][big : big + 1] = [
                [head, block, begin, mid, True],
                [head, block, mid, end, True],
            ]
        else:
            donor = max(
                range(resident), key=lambda u: (n[u], sum(cost(e) for e in full[u]))
            )
            full[v].append(full[donor].pop())
    else:
        return None
    pieces: dict[tuple[int, int], list[list]] = {}
    for lst in full:
        for e in lst:
            pieces.setdefault((e[0], e[1]), []).append(e)
    full = [full[v] for v in sorted(range(resident), key=lambda v: -len(full[v]))]
    # Parts in index order; groups and slots in the order of the rows' first blocks.
    words: dict[int, list[int]] = {}
    groups = slots = 0
    for ps in sorted(pieces.values(), key=lambda ps: min(e[2] for e in ps)):
        ps.sort(key=lambda e: e[2])
        if len(ps) == 1:
            words[id(ps[0])] = [-1, 1, 0, 0]
            continue
        for j, e in enumerate(ps):
            words[id(e)] = [groups, len(ps), j, slots + j]
        groups += 1
        slots += len(ps)
    rows: list[list[int]] = [[]] * sum(len(lst) for lst in full)
    for v, lst in enumerate(full):
        for j, e in enumerate(lst):
            rows[v + j * resident] = e[:4] + words[id(e)]
    return rows, groups, slots


def _pack_time(work: list[list[int]], resident: int, kernel: str) -> int:
    """The time of the explicit ``work`` (rows in launch order) on ``kernel``
    under the packing constants, in _makespan's form and units."""
    item_cost, lone_rate = _PACK_COST[kernel]
    grid = max(1, min(len(work), resident))
    wave = len(work) <= resident
    load = [0] * resident
    for p, w in enumerate(work):
        split = 0 if wave else _PACK_SPLIT_COST * (w[4] >= 0)
        load[p % grid] += w[3] - w[2] + item_cost + split
    if wave:
        span = 10 * max(load)
    else:
        sms = resident // _VIRTUAL_PER_SM
        span = max(
            10 * min(a, b) + lone_rate * abs(a - b)
            for a, b in zip(load[:sms], load[sms:], strict=True)
        )
    return _COST[kernel][2] * span


def build_work(
    counts: torch.Tensor,
    first: torch.Tensor,
    group_size: int,
    num_sms: int,
    part: Optional[int] = None,
    kernel: Optional[str] = None,
    pack: Optional[bool] = None,
    num_kv_blocks: int = 0,
) -> dict[str, Any]:
    """The work table of the rows ``counts`` / ``first`` (``[num_qo_heads, MB]``,
    int64: each row's block count and first index position) over
    ``num_kv_blocks`` KV blocks per KV head, the kernel that runs it and its
    grid, and the plan's split workspace, on the rows' device.  ``part`` and
    ``kernel`` fix those choices and ``pack`` the packing (True: a packed plan
    whenever one fits, False: never; a fixed ``part`` alone turns it off)
    (tests)."""
    if kernel is not None and kernel not in _GROUPS:
        raise ValueError(f"kernel must be one of {list(_GROUPS)}, got {kernel!r}")
    num_heads, mb = counts.shape
    device = counts.device
    resident = _VIRTUAL_PER_SM * num_sms
    kv_heads = num_heads // group_size
    # Rows in (KV head, query block, head) order, so that a stable sort of the
    # items leaves those of one size and kind in that order.
    order = (
        torch.arange(num_heads * mb, device=device)
        .view(kv_heads, group_size, mb)
        .transpose(1, 2)
        .reshape(-1)
    )
    row_count = counts.reshape(-1)[order]
    row_first = first.reshape(-1)[order]
    row_head = torch.div(order, mb, rounding_mode="floor")
    row_block = order % mb
    row_data = (row_count, row_first, row_head, row_block)
    packing = pack if pack is not None else part is None
    admitted = (
        packing
        and 0 < row_count.numel() < _PACK_ROWS * resident
        and kv_heads * num_kv_blocks <= _PACK_KV_BLOCKS[row_count.numel() >= resident]
    )
    far = _PACK_SPLIT_KV_BLOCKS if admitted else _SPLIT_KV_BLOCKS
    split_cost = _FAR_SPLIT_COST if kv_heads * num_kv_blocks > far else _SPLIT_COST
    chosen_part, chosen = _choose(row_count, resident, part, kernel, split_cost)
    work, groups, slots = _table(*row_data, chosen_part)
    if admitted:
        uniform = work.tolist()
        best = None if pack else _pack_time(uniform, resident, chosen)
        for k in [kernel] if kernel is not None else list(_GROUPS):
            words = uniform
            if k != chosen:
                k_part = _choose(row_count, resident, part, k, split_cost)[0]
                words = _table(*row_data, k_part)[0].tolist()
            packed = _pack(words, resident, _COST[k][0])
            if packed is None:
                continue
            span = _pack_time(packed[0], resident, k)
            if best is None or span < best:
                best, chosen = span, k
                work = torch.tensor(packed[0], dtype=torch.int64, device=device)
                groups, slots = packed[1], packed[2]
    work = work.to(torch.int32).contiguous()
    total_work = int(work.shape[0])
    return {
        "work": work,
        "total_work": total_work,
        "kernel": chosen,
        # One CTA per item, at most one per _GROUPS[kernel] virtual CTAs of the
        # SMs: CTA c's groups are the virtual CTAs c, c + grid, ...
        "grid": (max(1, min(total_work, resident // _GROUPS[chosen])), 1, 1),
        "partial_ws": torch.empty(
            (max(1, slots) * _PARTIAL_FLOATS,), dtype=torch.float32, device=device
        ),
        "split_counters": torch.zeros(
            (max(1, groups),), dtype=torch.int32, device=device
        ),
    }


def _with_tail(index: torch.Tensor) -> torch.Tensor:
    # One trailing entry: the kernel reads a row's next block id one block
    # ahead, and an empty selection still has a buffer.  The work table holds
    # int32 positions into this list.
    if index.numel() >= 2**31 - 1:
        raise ValueError(
            f"vsa_sm120_blk64 indexes the selected KV blocks with int32 positions; "
            f"the selection holds {index.numel()} block ids"
        )
    return torch.cat((index.to(torch.int32), index.new_zeros(1, dtype=torch.int32)))


def plan_block_mask(
    block_mask: torch.Tensor,
    group_size: int,
    num_sms: int,
    device: torch.device,
    part: Optional[int] = None,
    kernel: Optional[str] = None,
    pack: Optional[bool] = None,
) -> dict[str, Any]:
    """The plan of a per-head ``block_mask`` ``[num_qo_heads, MB, NB]``: each
    row's selected KV blocks in ascending order (``part``, ``kernel`` and
    ``pack`` as in :func:`build_work`)."""
    mask = block_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    counts = mask.sum(dim=-1, dtype=torch.int64)
    first = torch.cumsum(counts.reshape(-1), 0) - counts.reshape(-1)
    index = _with_tail(torch.nonzero(mask.flatten(0, 1))[:, 1])
    num_qo_heads, num_q_blocks, num_kv_blocks = mask.shape
    plan = build_work(
        counts,
        first.view_as(counts),
        group_size,
        num_sms,
        part,
        kernel,
        pack,
        num_kv_blocks,
    )
    plan["q2k_index"] = index
    plan["shape"] = (num_q_blocks, num_kv_blocks, num_qo_heads, group_size)
    return plan


def plan_bsr(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    num_q_blocks: int,
    num_qo_heads: int,
    num_kv_blocks: int,
    group_size: int,
    num_sms: int,
    device: torch.device,
) -> dict[str, Any]:
    """The plan of a BSR selection shared by every head: every head's row of
    query block ``b`` reads ``indices[indptr[b]:indptr[b + 1]]`` in place."""
    if indptr.dim() != 1 or indptr.numel() != num_q_blocks + 1:
        raise ValueError(
            f"BSR indptr must have num_q_blocks + 1 = {num_q_blocks + 1} entries, "
            f"got shape {tuple(indptr.shape)}"
        )
    indptr = indptr.to(device=device, dtype=torch.int64, non_blocking=True)
    indices = indices.to(device=device, non_blocking=True)
    # The kernel stores every row a work item names and reads its block ids
    # without bounds checks, so the rows must lie inside ``indices``.
    start, end, falling = torch.stack(
        (indptr[0], indptr[-1], (indptr[1:] < indptr[:-1]).sum())
    ).tolist()
    if start < 0 or end > indices.numel() or falling:
        raise ValueError(
            f"BSR indptr must be non-decreasing within [0, {indices.numel()}]: "
            f"got indptr[0]={start}, indptr[-1]={end}, {falling} decreasing steps"
        )
    if indices.numel():
        low, high = int(indices.min()), int(indices.max())
        if low < 0 or high >= num_kv_blocks:
            raise ValueError(
                f"BSR indices out of range [0, {num_kv_blocks}): got min={low}, max={high}"
            )
    counts = (indptr[1:] - indptr[:-1]).unsqueeze(0).expand(num_qo_heads, -1)
    first = indptr[:-1].unsqueeze(0).expand(num_qo_heads, -1)
    plan = build_work(counts, first, group_size, num_sms, num_kv_blocks=num_kv_blocks)
    plan["q2k_index"] = _with_tail(indices)
    plan["shape"] = (num_q_blocks, num_kv_blocks, num_qo_heads, group_size)
    return plan


@functools.cache
def _entry(dtype: torch.dtype, kernel: str, target: str):
    """The generated program of ``kernel`` for ``dtype`` and its ``run`` entry on
    ``target``; the record's arguments are checked against the host launch
    once, at load."""
    from .jit import cake_vsa_sm120
    from .jit.cake_program import GRID_ARGS

    bf16, f16 = {
        "vsa_sm120_blk64": (
            cake_vsa_sm120.VSA_SM120_BLK64_BF16,
            cake_vsa_sm120.VSA_SM120_BLK64_F16,
        ),
        "vsa_sm120_blk64_pair": (
            cake_vsa_sm120.VSA_SM120_BLK64_PAIR_BF16,
            cake_vsa_sm120.VSA_SM120_BLK64_PAIR_F16,
        ),
    }[kernel]
    program = {torch.bfloat16: bf16, torch.float16: f16}[dtype]
    if tuple(program.args) != (*_HOST_PARAMS, *GRID_ARGS):
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {[*_HOST_PARAMS, *GRID_ARGS]}"
        )
    if target not in program.targets:
        raise RuntimeError(
            f"{program.name} ships no {target!r} build; its targets are {list(program.targets)}"
        )
    return program, program.entry(target)


@functools.cache
def _no_lse(device: torch.device) -> torch.Tensor:
    return torch.empty((1,), dtype=torch.float32, device=device)


def _overlaps(tensor: torch.Tensor, *others: torch.Tensor) -> bool:
    """Whether the bytes ``tensor`` spans meet those of one of ``others``."""

    def span(t: torch.Tensor) -> tuple[int, int]:
        steps = zip(t.shape, t.stride(), strict=True)
        extent = 1 + sum((n - 1) * s for n, s in steps)
        return t.data_ptr(), t.data_ptr() + extent * t.element_size()

    if tensor.numel() == 0:
        return False
    begin, end = span(tensor)
    return any(o.numel() and o.data_ptr() < end and begin < span(o)[1] for o in others)


def run(
    plan: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float],
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    return_lse: bool,
):
    """Launch the planned attention: ``q [M, Hq, 128]``, ``k`` / ``v``
    ``[N, Hkv, 128]`` (NHD); returns ``out`` and, with ``return_lse``, the
    natural-log ``lse [M, Hq]`` (float32)."""
    import tvm_ffi

    from .jit.cake_program import CAPABILITY_TARGET
    from .utils import check_shape_dtype_device

    num_q_blocks, num_kv_blocks, num_qo_heads, group_size = plan["shape"]
    if q.dtype not in (torch.float16, torch.bfloat16) or not (
        k.dtype == v.dtype == q.dtype
    ):
        raise ValueError(
            "run() takes q, k and v of one dtype, float16 or bfloat16; got q "
            f"{q.dtype}, k {k.dtype}, v {v.dtype}"
        )
    expected_q = (64 * num_q_blocks, num_qo_heads, HEAD_DIM)
    expected_kv = (64 * num_kv_blocks, num_qo_heads // group_size, HEAD_DIM)
    if (
        tuple(q.shape) != expected_q
        or tuple(k.shape) != expected_kv
        or tuple(v.shape) != expected_kv
    ):
        raise ValueError(
            f"run() takes q {list(expected_q)} and k / v {list(expected_kv)} as "
            f"planned; got q {tuple(q.shape)}, k {tuple(k.shape)}, v "
            f"{tuple(v.shape)}"
        )
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    caller_out, caller_lse = out is not None, lse is not None
    if out is None:
        out = torch.empty_like(q)
    else:
        check_shape_dtype_device(out, q.shape, q.dtype, q.device, "out")
    if return_lse:
        if lse is None:
            lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        else:
            check_shape_dtype_device(lse, q.shape[:2], torch.float32, q.device, "lse")
    if q.shape[0] == 0 or k.shape[0] == 0:
        # No query or no KV block: every row is empty (O = 0, lse = -inf), as
        # for an empty selection; the kernel cannot map a zero-length tensor.
        out.zero_()
        if return_lse:
            lse.fill_(-math.inf)
            return out, lse
        return out
    # The kernel stores whole rows of out (16-byte vectors) and lse while other
    # work items still read q, k and v: a caller's out / lse that is strided,
    # misaligned or shares memory with an input (lse: or with out) gets a
    # buffer of its own, copied back after the launch, out first (the result
    # of an out-of-place launch).
    out_buf = out
    if (
        not out.is_contiguous()
        or out.data_ptr() % 16
        or (caller_out and _overlaps(out, q, k, v))
    ):
        out_buf = torch.empty_like(q)
    lse_buf = _no_lse(q.device)
    if return_lse:
        lse_buf = lse
        if not lse.is_contiguous() or (caller_lse and _overlaps(lse, q, k, v, out)):
            lse_buf = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    index = (
        q.device.index if q.device.index is not None else torch.cuda.current_device()
    )
    program, entry = _entry(
        q.dtype,
        plan["kernel"],
        CAPABILITY_TARGET[torch.cuda.get_device_capability(index)],
    )
    grid = plan["grid"]
    values = {
        "q_map": q,
        "k_map": k,
        "v_map": v,
        "out": out_buf,
        "lse": lse_buf,
        "partial_ws": plan["partial_ws"],
        "split_counters": plan["split_counters"],
        "q2k_index": plan["q2k_index"],
        "work": plan["work"],
        "num_heads": q.shape[1],
        "group_size": q.shape[1] // k.shape[1],
        "total_work": plan["total_work"],
        "write_lse": int(return_lse),
        "softmax_scale_log2": float(sm_scale) * math.log2(math.e),
        "grid_x": grid[0],
        "grid_y": grid[1],
        "grid_z": grid[2],
    }
    with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
        entry(*[values[name] for name in program.args])
    if out_buf is not out:
        out.copy_(out_buf)
    if return_lse:
        if lse_buf is not lse:
            lse.copy_(lse_buf)
        return out, lse
    return out
