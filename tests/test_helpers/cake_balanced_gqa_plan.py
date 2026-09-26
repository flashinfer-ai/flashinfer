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

Host mirror of the on-device balanced paged-GQA decode work planner.

The decode kernel plans its own work on the GPU from the device ``seq_lens``
buffer: one scheduler warp derives the chunk length ``L`` (in KV block pairs of
256 tokens) and four length buckets, then decodes every ticket online while
the other warps run the attention pipeline.  Neither the kernel nor the host
layer calls this module; it is a test-only reference so the plan the kernel
publishes in its workspace counters can be checked exactly, and it documents
the scheduling rule.

Work unit: one KV block pair (``PAIR_TOKENS`` tokens).  Request ``b`` with KV
length ``S_b`` is chunked by the pair count of its *shortest* query row,
``P_b = ceil((S_b - q_len + 1) / PAIR_TOKENS)``, and ``total_work = sum_b P_b *
q_len * num_kv_heads``.  The chunk length ``L`` is ``max(ceil(total_work /
(k * num_ctas)), pairs_min)`` with the integer balance factor ``k =
clamp(ceil(total_work / num_ctas / TARGET_CHUNK_PAIRS), 1, MAX_BALANCE_FACTOR)``.
Splitting is only used when it can shorten the launch (``P_max > ideal + L`` or
``P_max - P_min > L / 2``); otherwise ``L`` is raised to ``P_max`` and every
tile is one item.  Near-uniform small batches whose tiles all fit the grid use
an even split instead (``floor(num_ctas / tiles)`` chunks per tile).

Ticket order is by length bucket (full chunks first, then remainders / whole
tiles in ``[L/2, L)``, ``[L/4, L/2)`` and shorter), and inside a bucket by
``(batch, chunk, q_row, kv_head)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from flashinfer.experimental.balanced_gqa_decode.cake_bounds import (
    BLOCK_N,
    DEFAULT_PAIRS_MIN,
    MAX_BALANCE_FACTOR,
    MAX_REQUESTS,
    NUM_BUCKETS,
    PAIR_TOKENS,
    TARGET_CHUNK_PAIRS,
)


# Multi-wave uniform-batch launch-time model (mirror of the cake planner's
# ``uniform_launch_cost``): quarter pairs of streaming at the per-CTA rate of a
# full wave, times CTA slots.  A partial wave with fewer than SATURATING_CTAS
# streaming CTAs runs at a proportionally lower aggregate rate; above it the
# HBM roofline is reached and idle CTAs cost nothing.
ITEM_OVERHEAD_QUARTER_PAIRS = 4
MERGE_QUARTER_PAIRS = 3
SATURATING_CTAS = 96
UNIFORM_SPLIT_MARGIN_DIV = 50


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def length_bucket(length_pairs: int, chunk_pairs: int) -> int:
    """Bucket of an item of ``length_pairs`` pairs given chunk length ``L``.

    Bucket 0 holds exactly-``L`` items.  Bucket ``i >= 1`` holds lengths in
    ``[L / 2**i, L / 2**(i-1))``; the last bucket collects everything shorter.
    """
    if length_pairs <= 0 or length_pairs > chunk_pairs:
        raise ValueError("item length must be in [1, chunk_pairs]")
    if length_pairs == chunk_pairs:
        return 0
    bucket = 1
    while (length_pairs << bucket) < chunk_pairs and bucket < NUM_BUCKETS - 1:
        bucket += 1
    return bucket


def balance_factor_for(total_work: int, num_ctas: int) -> int:
    """Integer balance factor ``k`` from the ideal per-CTA share in pairs."""
    ideal_pairs = _ceil_div(total_work, num_ctas)
    return max(1, min(MAX_BALANCE_FACTOR, _ceil_div(ideal_pairs, TARGET_CHUNK_PAIRS)))


def split_is_worthwhile(
    *, p_max: int, p_min: int, ideal_pairs: int, chunk_pairs: int
) -> bool:
    """Device guard: split only when it can shorten the launch."""
    return p_max > ideal_pairs + chunk_pairs or 2 * (p_max - p_min) > chunk_pairs


def even_split_chunk_pairs(
    pairs: Sequence[int], *, items_per_chunk: int, num_ctas: int, pairs_min: int
) -> Optional[int]:
    """Near-uniform batches with ``tiles <= num_ctas``: one chunk per CTA.

    Every tile is split into ``floor(num_ctas / tiles)`` chunks of
    ``ceil(p_max / n)`` pairs.  Returns ``None`` when the rule does not apply
    (ragged spread above 1/8 of ``p_max``, no room for a second chunk, or the
    chunk would not be shorter than the tile).
    """
    whole_items = len(pairs) * items_per_chunk
    if whole_items > num_ctas:
        return None
    n_even = num_ctas // whole_items
    p_max, p_min = max(pairs), min(pairs)
    if n_even <= 1 or 8 * (p_max - p_min) > p_max:
        return None
    l_even = max(_ceil_div(p_max, n_even), pairs_min)
    return l_even if l_even < p_max else None


def uniform_launch_cost(
    *, p_max: int, whole_items: int, num_ctas: int, chunk_pairs: int
) -> int:
    """Launch-time estimate for ``whole_items`` tiles of ``p_max`` pairs cut
    into chunks of ``chunk_pairs`` (integer; the device planner evaluates the
    same expression lane-parallel).  Full chunks run first in complete waves;
    the partial wave that follows carries the remainder chunks on its idle
    CTAs (each absorbs ``unit // rem_unit``); a wave with ``n`` streaming CTAs
    costs ``max(SATURATING_CTAS, n)`` slots per unit of work; leftover
    remainders run in rounds over all CTAs; a split tile adds one merge.
    """
    chunk = min(chunk_pairs, p_max)
    full_per_tile, remainder = divmod(p_max, chunk)
    unit = 4 * chunk + ITEM_OVERHEAD_QUARTER_PAIRS
    full_items = full_per_tile * whole_items
    full_waves, n_last = divmod(full_items, num_ctas)
    cost = full_waves * unit * num_ctas
    rem_items = whole_items if remainder else 0
    rem_unit = 4 * remainder + ITEM_OVERHEAD_QUARTER_PAIRS
    if n_last:
        idle = num_ctas - n_last
        absorbed = min(rem_items, idle * (unit // rem_unit))
        active = n_last + min(idle, rem_items)
        cost += unit * max(SATURATING_CTAS, active)
        rem_items -= absorbed
    if rem_items:
        rounds = _ceil_div(rem_items, num_ctas)
        cost += rounds * rem_unit * max(SATURATING_CTAS, min(rem_items, num_ctas))
    if full_per_tile > 1 or remainder:
        cost += MERGE_QUARTER_PAIRS * num_ctas
    return cost


def uniform_fill_chunk_pairs(
    pairs: Sequence[int], *, items_per_chunk: int, num_ctas: int, pairs_min: int
) -> Optional[int]:
    """Multi-wave near-uniform batches (``tiles > num_ctas``,
    ``8 * (p_max - p_min) <= p_max``): the chunk length whose launch-cost
    estimate beats whole tiles by ``1 / UNIFORM_SPLIT_MARGIN_DIV``.

    Candidates: ``ceil(total_work / (k * num_ctas))`` for ``k = 1..8`` and the
    divisors ``ceil(p_max / n)`` for ``n = 2..8`` no shorter than the ``k = 8``
    length (keeps the split-item count inside the workspace bound).  Ties keep
    the longer chunk.  ``None`` when the rule does not apply or whole tiles win.
    """
    whole_items = len(pairs) * items_per_chunk
    p_max, p_min = max(pairs), min(pairs)
    if whole_items <= num_ctas or 8 * (p_max - p_min) > p_max:
        return None
    total_work = sum(pairs) * items_per_chunk
    whole_cost = uniform_launch_cost(
        p_max=p_max, whole_items=whole_items, num_ctas=num_ctas, chunk_pairs=p_max
    )
    l_min = max(_ceil_div(total_work, MAX_BALANCE_FACTOR * num_ctas), pairs_min)
    candidates = [
        max(_ceil_div(total_work, k * num_ctas), pairs_min)
        for k in range(1, MAX_BALANCE_FACTOR + 1)
    ]
    candidates += [
        chunk
        for chunk in (
            max(_ceil_div(p_max, n), pairs_min)
            for n in range(2, MAX_BALANCE_FACTOR + 1)
        )
        if chunk >= l_min
    ]
    best: Optional[tuple[int, int]] = None
    for chunk in candidates:
        if chunk >= p_max:
            continue
        cost = uniform_launch_cost(
            p_max=p_max, whole_items=whole_items, num_ctas=num_ctas, chunk_pairs=chunk
        )
        if best is None or cost < best[0] or (cost == best[0] and chunk > best[1]):
            best = (cost, chunk)
    if best is None or whole_cost - best[0] < whole_cost // UNIFORM_SPLIT_MARGIN_DIV:
        return None
    return best[1]


def chunk_pairs_for(
    seq_lens: Sequence[int],
    *,
    q_len: int,
    num_kv_heads: int,
    num_ctas: int,
    balance_factor: Optional[int] = None,
    pairs_min: int = DEFAULT_PAIRS_MIN,
    fill_uniform: bool = True,
) -> tuple[int, int]:
    """``(L, k)``: chunk length in pairs and the balance factor used.

    ``k == 0`` marks a near-uniform batch decided outright: multi-wave batches
    by the launch-cost model (``uniform_fill_chunk_pairs``; ``fill_uniform=False``
    skips it, the packed-row MTP planner's base candidate), single-wave batches
    by the even-split rule.
    """
    pairs = [_ceil_div(int(s) - (q_len - 1), PAIR_TOKENS) for s in seq_lens]
    items_per_chunk = q_len * num_kv_heads
    total_work = sum(pairs) * items_per_chunk
    ideal_pairs = _ceil_div(total_work, num_ctas)
    k = (
        balance_factor_for(total_work, num_ctas)
        if balance_factor is None
        else balance_factor
    )
    if k <= 0 or k > MAX_BALANCE_FACTOR:
        raise ValueError(f"balance_factor must be in [1, {MAX_BALANCE_FACTOR}]")
    chunk_pairs = max(_ceil_div(total_work, k * num_ctas), pairs_min)
    p_max, p_min = max(pairs), min(pairs)
    if (
        fill_uniform
        and balance_factor is None
        and len(pairs) * items_per_chunk > num_ctas
        and 8 * (p_max - p_min) <= p_max
    ):
        fill = uniform_fill_chunk_pairs(
            pairs,
            items_per_chunk=items_per_chunk,
            num_ctas=num_ctas,
            pairs_min=pairs_min,
        )
        return (fill, 0) if fill is not None else (p_max, k)
    if chunk_pairs < p_max and not split_is_worthwhile(
        p_max=p_max, p_min=p_min, ideal_pairs=ideal_pairs, chunk_pairs=chunk_pairs
    ):
        chunk_pairs = p_max
    even = even_split_chunk_pairs(
        pairs, items_per_chunk=items_per_chunk, num_ctas=num_ctas, pairs_min=pairs_min
    )
    if even is not None and balance_factor is None:
        return even, 0
    return chunk_pairs, k


@dataclass(frozen=True)
class BalancedWorkItem:
    ticket: int
    bucket: int
    batch: int
    q_row: int
    kv_head: int
    chunk: int
    n_chunks: int
    block_begin: int
    block_end: int  # exclusive, clamped to the causal row length; always > begin
    seqlen_row: int
    slot: int  # partial slot, -1 when the tile is not split
    counter: int  # completion counter index, -1 when the tile is not split

    @property
    def is_split(self) -> bool:
        return self.n_chunks > 1


@dataclass(frozen=True)
class BalancedWorkPlan:
    chunk_pairs: int
    q_len: int
    num_kv_heads: int
    num_ctas: int
    balance_factor: int
    pairs_min: int
    items: tuple[BalancedWorkItem, ...]
    bucket_ends: tuple[int, ...]
    num_split_items: int
    num_split_tiles: int

    @property
    def num_items(self) -> int:
        return len(self.items)

    def row_tile(self, item: BalancedWorkItem) -> int:
        return item.batch * self.q_len + item.q_row


def plan_balanced_work(
    seq_lens: Sequence[int],
    *,
    q_len: int = 1,
    num_kv_heads: int,
    num_ctas: int,
    balance_factor: Optional[int] = None,
    pairs_min: int = DEFAULT_PAIRS_MIN,
    chunk_pairs: Optional[int] = None,
) -> BalancedWorkPlan:
    """Enumerate the device ticket space for one launch.

    ``balance_factor`` overrides the adaptive integer ``k`` (diagnostics only;
    the kernel always uses the adaptive rule).  ``chunk_pairs`` fixes the
    length outright (used with ``mtp_chunk_pairs`` for the packed-row program).
    """
    lens = [int(s) for s in seq_lens]
    if not lens or len(lens) > MAX_REQUESTS:
        raise ValueError(f"request count must be in [1, {MAX_REQUESTS}]")
    if min(lens) <= 0:
        raise ValueError("every KV length must be positive")
    if q_len <= 0 or num_kv_heads <= 0 or num_ctas <= 0:
        raise ValueError("q_len, num_kv_heads and num_ctas must be positive")
    if pairs_min <= 0:
        raise ValueError("pairs_min must be positive")
    if min(lens) < q_len:
        raise ValueError("every KV length must be at least q_len")

    items_per_chunk = q_len * num_kv_heads
    if chunk_pairs is None:
        chunk_pairs, balance_factor_used = chunk_pairs_for(
            lens,
            q_len=q_len,
            num_kv_heads=num_kv_heads,
            num_ctas=num_ctas,
            balance_factor=balance_factor,
            pairs_min=pairs_min,
        )
    else:
        # Fixed length (the packed-row MTP kernel refines the length itself;
        # see mtp_chunk_pairs).
        if chunk_pairs < pairs_min:
            raise ValueError("chunk_pairs must be at least pairs_min")
        balance_factor_used = 0

    pairs = [_ceil_div(s - (q_len - 1), PAIR_TOKENS) for s in lens]
    n_chunks = [_ceil_div(p, chunk_pairs) for p in pairs]
    full_chunks = [p // chunk_pairs for p in pairs]
    remainder = [p - f * chunk_pairs for p, f in zip(pairs, full_chunks, strict=True)]

    split_item_prefix: list[int] = []
    split_tile_prefix: list[int] = []
    acc_items = acc_tiles = 0
    for n in n_chunks:
        split_item_prefix.append(acc_items)
        split_tile_prefix.append(acc_tiles)
        if n > 1:
            acc_items += n * items_per_chunk
            acc_tiles += items_per_chunk
    num_split_items, num_split_tiles = acc_items, acc_tiles

    def make_item(ticket: int, bucket: int, b: int, c: int) -> list[BalancedWorkItem]:
        out: list[BalancedWorkItem] = []
        for j in range(q_len):
            seqlen_row = lens[b] - (q_len - 1 - j)
            n_blocks_row = _ceil_div(seqlen_row, BLOCK_N)
            block_begin = 2 * c * chunk_pairs
            # The last chunk runs to the row's causal end (a longer MTP row may
            # own one block past the shortest-row pair count); other chunks
            # are exactly 2 * L blocks.
            block_end = (
                n_blocks_row if c == n_chunks[b] - 1 else 2 * (c + 1) * chunk_pairs
            )
            if not block_begin < block_end <= n_blocks_row:
                raise AssertionError(
                    "shortest-row chunking must never produce an empty item"
                )
            for h in range(num_kv_heads):
                split = n_chunks[b] > 1
                slot = (
                    split_item_prefix[b] + (c * q_len + j) * num_kv_heads + h
                    if split
                    else -1
                )
                counter = split_tile_prefix[b] + j * num_kv_heads + h if split else -1
                out.append(
                    BalancedWorkItem(
                        ticket=ticket + len(out),
                        bucket=bucket,
                        batch=b,
                        q_row=j,
                        kv_head=h,
                        chunk=c,
                        n_chunks=n_chunks[b],
                        block_begin=block_begin,
                        block_end=block_end,
                        seqlen_row=seqlen_row,
                        slot=slot,
                        counter=counter,
                    )
                )
        return out

    items: list[BalancedWorkItem] = []
    bucket_ends: list[int] = []
    for bucket in range(NUM_BUCKETS):
        for b in range(len(lens)):
            if bucket == 0:
                for c in range(full_chunks[b]):
                    items.extend(make_item(len(items), bucket, b, c))
            elif (
                remainder[b] > 0 and length_bucket(remainder[b], chunk_pairs) == bucket
            ):
                items.extend(make_item(len(items), bucket, b, full_chunks[b]))
        bucket_ends.append(len(items))

    return BalancedWorkPlan(
        chunk_pairs=chunk_pairs,
        q_len=q_len,
        num_kv_heads=num_kv_heads,
        num_ctas=num_ctas,
        balance_factor=balance_factor_used,
        pairs_min=pairs_min,
        items=tuple(items),
        bucket_ends=tuple(bucket_ends),
        num_split_items=num_split_items,
        num_split_tiles=num_split_tiles,
    )


def simulate_greedy_makespan(
    plan: BalancedWorkPlan, *, item_overhead_pairs: float = 0.0
) -> float:
    """Greedy list-scheduling makespan in pairs (planning diagnostics only)."""
    import heapq

    finish = [0.0] * plan.num_ctas
    heapq.heapify(finish)
    for item in plan.items:
        start = heapq.heappop(finish)
        length = max(item.block_end - item.block_begin, 0) / 2.0
        heapq.heappush(finish, start + length + item_overhead_pairs)
    return max(finish)


__all__ = [
    "BalancedWorkItem",
    "BalancedWorkPlan",
    "balance_factor_for",
    "chunk_pairs_for",
    "even_split_chunk_pairs",
    "length_bucket",
    "plan_balanced_work",
    "simulate_greedy_makespan",
    "split_is_worthwhile",
]


# ---------------------------------------------------------------------------
# Packed-row MTP program (q_len_per_req 3..8)
# ---------------------------------------------------------------------------

# Planner cost model of the packed-row kernel, in quarter pairs (mirrors the kernel).
MTP_ITEM_OVERHEAD_PAIRS = 1
MTP_MERGE_WAVE_PAIRS = 1
MTP_MERGE_ROWCHUNKS_PER_QUARTER_PAIR = 6
MTP_INLINE_MERGE_ROWCHUNKS = 18  # in-place fold of a two-chunk tile ~ 3/4 pair
MTP_COARSE_CANDIDATES = 3  # the planner also evaluates 2, 3, 4 x ceil(total / CTAs)


def mtp_chunk_pairs(
    seq_lens: Sequence[int], *, q_len: int, num_kv_heads: int, num_ctas: int
) -> int:
    """Chunk length chosen by the packed-row MTP kernel's device planner.

    The packed kernel plans ``items_per_chunk = num_kv_heads`` from the longest
    query row.  It starts from ``chunk_pairs_for`` (``q_len = 1``) and then
    evaluates ``ceil(total / (k * CTAs))`` for ``k = 1..MAX_BALANCE_FACTOR``,
    that length, whole tiles and ``2, 3, 4 x ceil(total / CTAs)`` (below whole
    tiles), keeping the smallest cost in quarter pairs
    ``4 * waves * (L + 1) + 4 * merge_waves + ceil(max_rowchunks / 6)`` (ties go
    to the earlier candidate), where ``max_rowchunks`` is the longest fold: a
    merge ticket's rows per warp x chunks, or 18 for a two-chunk tile folded
    in place by its last chunk item.
    """
    from flashinfer.experimental.balanced_gqa_decode.cake_bounds import (
        MTP_INLINE_MERGE_CHUNKS,
        mtp_merge_slices_for,
    )

    pairs = [_ceil_div(int(s), PAIR_TOKENS) for s in seq_lens]
    total_work = sum(pairs) * num_kv_heads
    base, _ = chunk_pairs_for(
        seq_lens,
        q_len=1,
        num_kv_heads=num_kv_heads,
        num_ctas=num_ctas,
        fill_uniform=False,
    )
    best_cost, best = None, base
    prev = 0
    l_one = max(_ceil_div(total_work, num_ctas), DEFAULT_PAIRS_MIN)
    for kc in range(MAX_BALANCE_FACTOR + 2 + MTP_COARSE_CANDIDATES):
        if kc < MAX_BALANCE_FACTOR:
            cand = max(_ceil_div(total_work, (kc + 1) * num_ctas), DEFAULT_PAIRS_MIN)
        elif kc == MAX_BALANCE_FACTOR:
            cand = base
        elif kc == MAX_BALANCE_FACTOR + 1:
            cand = max(pairs)
        else:
            cand = l_one * (kc - MAX_BALANCE_FACTOR)
            if cand >= max(pairs):
                continue  # whole tiles already evaluated
        if cand == prev:
            continue
        prev = cand
        n_chunks = [_ceil_div(p, cand) for p in pairs]
        tickets = sum(n_chunks) * num_kv_heads
        merge_tickets = (
            sum(mtp_merge_slices_for(n, q_len) for n in n_chunks if n > 1)
            * num_kv_heads
        )
        waves = _ceil_div(tickets, num_ctas)
        merge_waves = _ceil_div(merge_tickets, num_ctas)
        max_rowchunks = max(
            (
                (2 * q_len // mtp_merge_slices_for(n, q_len)) * n
                if n > MTP_INLINE_MERGE_CHUNKS
                else MTP_INLINE_MERGE_ROWCHUNKS
                for n in n_chunks
                if n >= MTP_INLINE_MERGE_CHUNKS
            ),
            default=0,
        )
        cost = (
            4 * waves * (cand + MTP_ITEM_OVERHEAD_PAIRS)
            + 4 * merge_waves * MTP_MERGE_WAVE_PAIRS
            + _ceil_div(max_rowchunks, MTP_MERGE_ROWCHUNKS_PER_QUARTER_PAIR)
        )
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, cand
    return best


def mtp_device_plan(
    seq_lens: Sequence[int], *, q_len: int, num_kv_heads: int, num_ctas: int
) -> tuple[int, int]:
    """``(chunk_pairs, tickets)`` the packed-row kernel publishes in its queue counters."""
    from flashinfer.experimental.balanced_gqa_decode.cake_bounds import (
        mtp_merge_items,
    )

    chunk = mtp_chunk_pairs(
        seq_lens, q_len=q_len, num_kv_heads=num_kv_heads, num_ctas=num_ctas
    )
    plan = plan_balanced_work(
        seq_lens,
        q_len=1,
        num_kv_heads=num_kv_heads,
        num_ctas=num_ctas,
        chunk_pairs=chunk,
    )
    return chunk, plan.num_items + mtp_merge_items(
        seq_lens, q_len_per_req=q_len, num_kv_heads=num_kv_heads, chunk_pairs=chunk
    )
