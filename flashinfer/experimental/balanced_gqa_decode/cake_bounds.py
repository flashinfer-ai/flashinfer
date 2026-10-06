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


Shape-independent facts of the on-device balanced paged-GQA decode kernels.

The experimental API runs the ``csrc/cake_fmha`` balanced components:
``decode_balanced_bf16`` (the row-tile kernel, ``q_len_per_req`` 1) and
``decode_balanced_bf16_mtp_n32`` / ``decode_balanced_bf16_mtp_n64`` (the
packed-row MTP kernels, ``q_len_per_req`` 3..4 / 5..8).  Each kernel plans its
own work on the GPU from the device ``seq_lens`` buffer (one scheduler warp
derives the chunk length and four length buckets, then decodes every ticket
online); nothing here reads KV lengths.  The production adapter
(``csrc/cake_fmha/jit/cake_fmha_decode_balanced_jit_binding.cu``) sizes and
carves the partial workspace and the self-resetting counters and bounds the
kernel's ticket loop from the CTA count alone.  This module mirrors those
constants so the host can locate the plan facts the kernel publishes and the
tests can check the device plan against the planner mirror in
``tests/test_helpers/cake_balanced_gqa_plan.py``.
"""

from __future__ import annotations

from typing import Sequence

BLOCK_N = 128
PAIR_TOKENS = 2 * BLOCK_N
NUM_BUCKETS = 4  # full chunks, [L/2, L), [L/4, L/2), shorter
REQUEST_GROUP = 32  # requests per planning group (one per scheduler lane)
MAX_REQUEST_GROUPS = 32
MAX_REQUESTS = MAX_REQUEST_GROUPS * REQUEST_GROUP
TARGET_CHUNK_PAIRS = 64
MAX_BALANCE_FACTOR = 8
DEFAULT_PAIRS_MIN = 2

# Workspace geometry of the production adapter.  Per split item one FP32
# partial tile and one statistics slot (row maxima then row sums); the
# statistics region carries one extra slot, the last of the region, in which
# CTA 0 publishes the plan facts of every launch.  The tile counters are
# followed by the four 16-byte-aligned queue counters.
ROW_STATS_PER_SLOT = 16  # max[8] then sum[8]
ROW_COUNTERS_PER_TILE = 1  # chunk arrivals
QUEUE_COUNTERS = 4  # ticket, done CTAs (both reset in-kernel), two unused words
PLAN_FACTS_WORDS = 2  # (chunk pairs, total items), stored as float32

# Packed-row MTP kernels (q_len_per_req in [MTP_MIN_Q_LEN, MTP_MAX_Q_LEN]).
MTP_MIN_Q_LEN = 3
MTP_MAX_Q_LEN = 8
MTP_GROUP = 8  # query heads per KV head
MTP_MAX_N_ROWS = 64  # physical packed tile of both instances
MTP_STATS_PER_SLOT = 2 * MTP_MAX_N_ROWS  # max[64] then sum[64]
# Per split tile: chunk arrivals, reduce tickets that observed the completed
# tile, one unused word, the two-chunk published flag.
MTP_COUNTERS_PER_TILE = 4
# Tiles of exactly two chunks are folded in place by the last arriving chunk
# item and take no reduce ticket.
MTP_INLINE_MERGE_CHUNKS = 2
# Reduce tickets per split tile: 1 (longest split tile <= 4 chunks), 2 (<= 8),
# 4 (<= 16), 8 (more); the ticket-loop bound counts the maximum.
MTP_REDUCE_SLICES_MAX = 8


def uses_packed_mtp(q_len_per_req: int) -> bool:
    """True when ``q_len_per_req`` is served by a packed-row MTP program."""
    return MTP_MIN_Q_LEN <= q_len_per_req <= MTP_MAX_Q_LEN


def serves_q_len(q_len_per_req: int) -> bool:
    """True for the query lengths the balanced programs serve: 1 and 3..8."""
    return q_len_per_req == 1 or uses_packed_mtp(q_len_per_req)


def mtp_n_rows(q_len_per_req: int) -> int:
    """Packed tile rows (32 or 64) of the MTP instance serving ``q_len_per_req``."""
    if not uses_packed_mtp(q_len_per_req):
        raise ValueError(
            f"packed MTP serves q_len_per_req in [{MTP_MIN_Q_LEN}, {MTP_MAX_Q_LEN}]"
        )
    return 32 if q_len_per_req * MTP_GROUP <= 32 else 64


def mtp_reduce_shift(n_max: int) -> int:
    """log2 of the reduce tickets per split tile when the longest split tile has ``n_max`` chunks."""
    return int(n_max > 4) + int(n_max > 8) + int(n_max > 16)


def mtp_reduce_items(
    seq_lens: Sequence[int], *, num_kv_heads: int, chunk_pairs: int
) -> int:
    """Reduce tickets the device scheduler appends for chunk length ``chunk_pairs``.

    Every ``(request, kv head)`` tile of more than ``MTP_INLINE_MERGE_CHUNKS``
    chunks takes ``1 << mtp_reduce_shift(n_max)`` tickets, ``n_max`` being the
    most chunks any such tile has; two-chunk tiles fold in place.
    """
    n_chunks = [
        -(-((int(s) + PAIR_TOKENS - 1) // PAIR_TOKENS) // chunk_pairs) for s in seq_lens
    ]
    split = [n for n in n_chunks if n > MTP_INLINE_MERGE_CHUNKS]
    if not split:
        return 0
    return len(split) * num_kv_heads * (1 << mtp_reduce_shift(max(split)))


def mtp_max_items_bound(batch: int, num_kv_heads: int, num_ctas: int) -> int:
    """Packed kernels' ticket-loop bound (the adapter's ``MaxItems``).

    Whole tiles, every possible split item and the most reduce tickets per
    split tile; the plan does not depend on ``q_len_per_req``.
    """
    max_split_items, max_split_tiles = workspace_bounds(num_ctas)
    return (
        batch * num_kv_heads + max_split_items + max_split_tiles * MTP_REDUCE_SLICES_MAX
    )


def workspace_bounds(num_ctas: int) -> tuple[int, int]:
    """``(max_split_items, max_split_tiles)`` valid for every shape.

    Split requests have ``P_b > L`` so ``n_b < 2 P_b / L``; summing gives
    ``split_items < 2 * total_work / L <= 2 * k * num_ctas`` with
    ``k <= MAX_BALANCE_FACTOR``, and every split tile has at least two items.
    """
    if num_ctas <= 0:
        raise ValueError("num_ctas must be positive")
    return 2 * MAX_BALANCE_FACTOR * num_ctas, MAX_BALANCE_FACTOR * num_ctas


def max_items_bound(batch: int, q_len: int, num_kv_heads: int, num_ctas: int) -> int:
    """Row-tile kernel's ticket-loop bound: whole tiles plus every possible split item."""
    max_split_items, _ = workspace_bounds(num_ctas)
    return batch * q_len * num_kv_heads + max_split_items
