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
