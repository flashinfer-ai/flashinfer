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


Shape-independent bounds of the on-device balanced paged-GQA decode planner.

The decode kernel plans its own work on the GPU from the device ``seq_lens``
buffer (one scheduler warp derives the chunk length and four length buckets,
then decodes every ticket online).  Nothing here reads KV lengths: the host
only needs these constants and bounds to carve the caller-owned workspace and
to bound the kernel's ticket loop, so one prepared launch replays for any
length distribution.  The exact host mirror of the planner used by the tests
lives in ``tests/test_helpers/cake_balanced_gqa_plan.py``.

Two generated programs share the workspace: the row-tile kernel (one 8-head
query row per item, any ``q_len_per_req``) and the packed-row MTP kernel
(``q_len_per_req`` 3..8: one ``8 * q_len``-row tile per ``(request, kv head)``
so each KV chunk is streamed once per request instead of once per draft row).
The packed kernel plans ``items_per_chunk = num_kv_heads`` from the longest
row and appends ``2 * q_len`` merge tickets per split tile; its partial slots
hold a 64 x 128 FP32 tile and 128 statistics words.
"""

from __future__ import annotations

BLOCK_N = 128
PAIR_TOKENS = 2 * BLOCK_N
NUM_BUCKETS = 4  # full chunks, [L/2, L), [L/4, L/2), shorter
REQUEST_GROUP = 32  # requests per planning group (one per scheduler lane)
MAX_REQUEST_GROUPS = 32
MAX_REQUESTS = MAX_REQUEST_GROUPS * REQUEST_GROUP
TARGET_CHUNK_PAIRS = 64
MAX_BALANCE_FACTOR = 8
DEFAULT_PAIRS_MIN = 2


# Packed-row MTP kernel (q_len_per_req in [MTP_MIN_Q_LEN, MTP_MAX_Q_LEN]).
MTP_MIN_Q_LEN = 3
MTP_MAX_Q_LEN = 8
MTP_GROUP = 8  # query heads per KV head
MTP_MERGE_ROWS = 4  # packed rows folded per merge ticket (one per correction warp)
MTP_MAX_N_ROWS = 64
MTP_PARTIAL_O_PER_SLOT = MTP_MAX_N_ROWS * 128  # FP32 O^T[64, 128] per split item
MTP_STATS_PER_SLOT = 2 * MTP_MAX_N_ROWS  # max[64] then sum[64]
MTP_COUNTERS_PER_TILE = 2  # arrivals, merges done (both reset by the last merge)


def uses_packed_mtp(q_len_per_req: int) -> bool:
    """True when ``q_len_per_req`` is served by the packed-row MTP program."""
    return MTP_MIN_Q_LEN <= q_len_per_req <= MTP_MAX_Q_LEN


def mtp_n_rows(q_len_per_req: int) -> int:
    """Packed tile rows (32 or 64) of the MTP instance serving ``q_len_per_req``."""
    if not uses_packed_mtp(q_len_per_req):
        raise ValueError(
            f"packed MTP serves q_len_per_req in [{MTP_MIN_Q_LEN}, {MTP_MAX_Q_LEN}]"
        )
    return 32 if q_len_per_req * MTP_GROUP <= 32 else 64


def mtp_merge_slices_per_tile(q_len_per_req: int) -> int:
    """Merge tickets appended per split ``(request, kv head)`` tile."""
    return q_len_per_req * MTP_GROUP // MTP_MERGE_ROWS


def mtp_max_items_bound(
    batch: int, q_len_per_req: int, num_kv_heads: int, num_ctas: int
) -> int:
    """Packed kernel ticket-loop bound: whole tiles, split items and merge tickets."""
    max_split_items, max_split_tiles = workspace_bounds(num_ctas)
    return (
        batch * num_kv_heads
        + max_split_items
        + max_split_tiles * mtp_merge_slices_per_tile(q_len_per_req)
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
    """Device ticket-loop bound: whole tiles plus every possible split item."""
    max_split_items, _ = workspace_bounds(num_ctas)
    return batch * q_len * num_kv_heads + max_split_items
