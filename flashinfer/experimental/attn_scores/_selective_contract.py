# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared private publication contracts for selective planners and kernels."""

from enum import IntEnum

# The native FP16 sample histogram uses descending bins; the last represents
# the conservative all-key (-inf) threshold.
_SAMPLE_HISTOGRAM_BINS = 2048

_COUNT_SEGMENTS = (4, 8, 12, 16, 32, 64, 128)
_SPLIT_KV_CHOICES = (1, 2, 4, 8, 16, 32, 64)
_MAX_STRIPED_SPLIT_KV = 8
_CANDIDATE_CAPACITY = 6144
_REPAIR_CAPACITY = 8192


def _h64_single_pool_topology(
    num_heads: int,
    head_dim: int,
    query_tile: int,
    num_umma_stages: int,
    split_kv: int,
    count_segments: int,
    capacity: int,
) -> bool:
    """Recognize the bounded H64 split1 candidate ABI, not global repair."""
    return capacity in (_CANDIDATE_CAPACITY, _REPAIR_CAPACITY) and (
        num_heads,
        head_dim,
        query_tile,
        num_umma_stages,
        split_kv,
        count_segments,
    ) == (64, 128, 4, 1, 1, 1)


def _q1_owner_pool_topology(
    num_heads: int,
    head_dim: int,
    query_tile: int,
    num_umma_stages: int,
    split_kv: int,
    count_segments: int,
    capacity: int,
) -> bool:
    """One store warp owns each Q1 row in each disjoint split-KV segment.

    This only pools that warp's four vector cursors. Capacity is still divided
    among real split owners; no cursor or quota is shared across CTAs.
    """
    return (
        num_heads in (8, 64)
        and (head_dim, query_tile, num_umma_stages, capacity)
        == (128, 1, 2, _CANDIDATE_CAPACITY)
        and split_kv in (1, 2, 4, 8)
        and count_segments == split_kv
    )


class _SelectiveLogitsMode(IntEnum):
    """Output mode for selective FP8 MQA logits production.

    ``SAMPLE`` writes FP32 scores. ``CANDIDATE`` and ``REPAIR`` write packed
    score/index records for ``score >= threshold``. ``RADIX_HISTOGRAM`` and
    ``EXACT`` provide device-side exact TopK completion: ``EXACT`` writes two
    equal row banks of int32 indices, ``> T`` in the first and ``== T`` in the
    second (ordered FP32 keys), each with its own per-row count.
    """

    SAMPLE = 0
    CANDIDATE = 1
    REPAIR = 2
    RADIX_HISTOGRAM = 3
    EXACT = 4
