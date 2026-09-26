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
"""

import triton
import triton.language as tl


@triton.jit
def _pack_query_value(
    query,
    value,
    order,
    packed_query,
    packed_value,
    elements: tl.constexpr,
    row_width: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    copy_value: tl.constexpr,
    block: tl.constexpr,
):
    program = tl.program_id(0)
    if elements >= 2**31:
        program = program.to(tl.int64)
    offset = program * block + tl.arange(0, block)
    valid = offset < elements
    token = (offset // row_width).to(tl.int64)
    head = (offset % row_width // 128).to(tl.int64)
    channel = offset % 128
    source_token = tl.load(order + token, valid, other=0).to(tl.int64)
    q = tl.load(
        query + source_token * q_stride_t + head * q_stride_h + channel,
        valid,
        other=0,
    )
    tl.store(packed_query + offset, q, valid)
    if copy_value:
        v = tl.load(
            value + token * v_stride_t + head * v_stride_h + channel,
            valid,
            other=0,
        )
        tl.store(packed_value + offset, v, valid)


@triton.jit
def _scatter_output(
    source,
    order,
    output,
    elements: tl.constexpr,
    row_width: tl.constexpr,
    block: tl.constexpr,
):
    program = tl.program_id(0)
    if elements >= 2**31:
        program = program.to(tl.int64)
    offset = program * block + tl.arange(0, block)
    valid = offset < elements
    token = tl.load(order + offset // row_width, valid, other=0).to(tl.int64)
    value = tl.load(source + offset, valid, other=0)
    tl.store(output + token * row_width + offset % row_width, value, valid)
