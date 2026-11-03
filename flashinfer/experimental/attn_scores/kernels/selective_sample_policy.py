# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device-keyed stratified SAMPLE budgets and input-independent positions."""

import cutlass
import cutlass.cute as cute


@cute.jit
def selective_sample_count(
    length, top_k: cutlass.Constexpr, capacity: cutlass.Constexpr
):
    # Reduced samples cover at least 32 expected TopK hits, with the same
    # device-keyed floor for prefill and decode.
    target = max(
        cutlass.Int32(256),
        cutlass.Int32(
            min(
                cutlass.Int64(capacity),
                cute.ceil_div(cutlass.Int64(length) * 32, top_k),
            )
        ),
    )
    count = cutlass.Int32(
        cutlass.Uint32(1)
        << (32 - cutlass.Int32(cute.arch.clz(cutlass.Uint32(target - 1))))
    )
    # Beyond this population/K ratio, use the full publication sample budget.
    if length > 32 * top_k:
        count = cutlass.Int32(capacity)
    return min(length, min(count, capacity))


@cute.jit
def stratified_sample_position(position, length, count, seed, limit):
    # Unsigned wrap and shifts give the same fixed seed in every specialization.
    position = min(position, max(count - 1, 0))
    hashed = (
        cutlass.Uint32(position)
        + cutlass.Uint32(seed) * cutlass.Uint32(0x9E3779B9)
        + cutlass.Uint32(0x243F6A88)
    )
    hashed = (hashed ^ (hashed >> 16)) * cutlass.Uint32(0x85EBCA6B)
    hashed = (hashed ^ (hashed >> 13)) * cutlass.Uint32(0xC2B2AE35)
    hashed = hashed ^ (hashed >> 16)
    begin = cutlass.Int64(position) * cutlass.Int64(length) // max(count, 1)
    end = cutlass.Int64(position + 1) * cutlass.Int64(length) // max(count, 1)
    offset = (cutlass.Uint64(hashed) * cutlass.Uint64(max(end - begin, 1))) >> 32
    return min(begin + cutlass.Int64(offset), cutlass.Int64(limit))
