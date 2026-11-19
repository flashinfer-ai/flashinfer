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

"""Runtime nonempty-prefix split-KV helpers for throughput 2CTA MLA."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32


@cute.jit
def runtime_split_kv_cap(
    max_split_kv,
    is_var_split_kv: cutlass.Constexpr[bool],
    block_split_kvs,
    batch_idx,
):
    """Return a positive per-batch cap bounded by launch/workspace capacity."""

    max_split_kv = cute.math.max(Int32(max_split_kv), Int32(1))
    split_kv_cap = max_split_kv
    if cutlass.const_expr(is_var_split_kv):
        split_kv_cap = Int32(block_split_kvs[batch_idx])
    return cute.math.max(
        cute.math.min(split_kv_cap, max_split_kv),
        Int32(1),
    )


@cute.jit
def runtime_split_tile_range(k_tile_total, split_kv_cap, split_idx, split_divisor=None):
    """Device form of :func:`split_tile_range`."""

    k_tile_total = cute.math.max(Int32(k_tile_total), Int32(0))
    split_idx = Int32(split_idx)
    split_kv_cap = cute.math.max(Int32(split_kv_cap), Int32(1))
    divisor = split_kv_cap if split_divisor is None else split_divisor
    tiles_per_split = cute.math.max(
        (k_tile_total + split_kv_cap - Int32(1)) // divisor,
        Int32(1),
    )
    start = split_idx * tiles_per_split
    count = cute.math.min(
        tiles_per_split,
        cute.math.max(k_tile_total - start, Int32(0)),
    )
    start = start if count > Int32(0) else k_tile_total
    return start, count


@cute.jit
def runtime_row_prefix_active_split_count(
    row_k_tile_total,
    group_k_tile_total,
    split_kv_cap,
    split_divisor=None,
):
    """Device form of :func:`row_prefix_active_split_count`."""

    group_k_tile_total = cute.math.max(Int32(group_k_tile_total), Int32(0))
    row_k_tile_total = cute.math.max(
        cute.math.min(Int32(row_k_tile_total), group_k_tile_total),
        Int32(0),
    )
    split_kv_cap = cute.math.max(Int32(split_kv_cap), Int32(1))
    divisor = split_kv_cap if split_divisor is None else split_divisor
    tiles_per_split = cute.math.max(
        (group_k_tile_total + split_kv_cap - Int32(1)) // divisor,
        Int32(1),
    )
    row_active_splits = (
        row_k_tile_total + tiles_per_split - Int32(1)
    ) // tiles_per_split
    # The row K domain was clipped to the group's prefix, making the extra
    # group-active quotient and min redundant.
    return row_active_splits
