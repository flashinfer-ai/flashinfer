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

# Model/topology invariants stay specialized so index division can be folded.
# Resolution, head-band schedule, offsets, and tensor strides remain runtime
# arguments to avoid recompiling for each request shape or positive-strided view.


@triton.jit
def _pack_ulysses_qkv_head_chunk_kernel(
    output,
    query,
    key,
    value,
    total,
    batch,
    seq_len,
    world_size: tl.constexpr,
    local_heads: tl.constexpr,
    chunk_heads,
    head_offset,
    head_dim: tl.constexpr,
    q_batch_stride,
    q_seq_stride,
    q_head_stride,
    q_dim_stride,
    k_batch_stride,
    k_seq_stride,
    k_head_stride,
    k_dim_stride,
    v_batch_stride,
    v_seq_stride,
    v_head_stride,
    v_dim_stride,
    nccl_layout: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack Q/K/V into fused ``[..., 3 * D]`` Ulysses payloads.

    ``nccl_layout=False`` writes ``[B,S,W*HC,3D]``. ``True`` writes the
    send-major ``[W,B,S,HC,3D]`` representation consumed directly by
    ``all_to_all_single``.
    """
    # The compact payload is int32-sized, but a positive-strided source view
    # can address beyond that range. Keep all derived source offsets in i64.
    offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    valid = offsets < total
    dim = offsets % head_dim
    slot = offsets // head_dim
    head_in_chunk = slot % chunk_heads
    slot = slot // chunk_heads
    destination = slot % world_size
    slot = slot // world_size
    seq = slot % seq_len
    batch_idx = slot // seq_len

    global_head = destination * local_heads + head_offset + head_in_chunk
    q_offset = (
        batch_idx * q_batch_stride
        + seq * q_seq_stride
        + global_head * q_head_stride
        + dim * q_dim_stride
    )
    k_offset = (
        batch_idx * k_batch_stride
        + seq * k_seq_stride
        + global_head * k_head_stride
        + dim * k_dim_stride
    )
    v_offset = (
        batch_idx * v_batch_stride
        + seq * v_seq_stride
        + global_head * v_head_stride
        + dim * v_dim_stride
    )

    if nccl_layout:
        record = (
            (destination * batch + batch_idx) * seq_len + seq
        ) * chunk_heads + head_in_chunk
    else:
        record = (
            (batch_idx * seq_len + seq) * world_size + destination
        ) * chunk_heads + head_in_chunk
    output_base = record * (3 * head_dim) + dim
    tl.store(output + output_base, tl.load(query + q_offset, mask=valid), mask=valid)
    tl.store(
        output + output_base + head_dim,
        tl.load(key + k_offset, mask=valid),
        mask=valid,
    )
    tl.store(
        output + output_base + 2 * head_dim,
        tl.load(value + v_offset, mask=valid),
        mask=valid,
    )


@triton.jit
def _pack_ulysses_output_sequence_chunk_kernel(
    output,
    source,
    total,
    batch,
    local_seq,
    world_size: tl.constexpr,
    chunk_heads,
    head_dim: tl.constexpr,
    source_batch_stride,
    source_seq_stride,
    source_head_stride,
    source_dim_stride,
    BLOCK: tl.constexpr,
):
    """Pack ``[B,W*S,HC,D]`` into NCCL send-major ``[W,B,S,HC,D]``."""
    offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    valid = offsets < total
    dim = offsets % head_dim
    slot = offsets // head_dim
    head = slot % chunk_heads
    slot = slot // chunk_heads
    seq = slot % local_seq
    slot = slot // local_seq
    batch_idx = slot % batch
    destination = slot // batch
    global_seq = destination * local_seq + seq
    source_offset = (
        batch_idx * source_batch_stride
        + global_seq * source_seq_stride
        + head * source_head_stride
        + dim * source_dim_stride
    )
    tl.store(output + offsets, tl.load(source + source_offset, mask=valid), mask=valid)


@triton.jit
def _merge_ulysses_output_head_chunk_kernel(
    received,
    output,
    total,
    batch,
    local_seq,
    world_size: tl.constexpr,
    local_heads: tl.constexpr,
    chunk_heads,
    head_offset,
    head_dim: tl.constexpr,
    recv_rank_stride,
    recv_batch_stride,
    recv_seq_stride,
    recv_head_stride,
    recv_dim_stride,
    out_batch_stride,
    out_seq_stride,
    out_head_stride,
    out_dim_stride,
    BLOCK: tl.constexpr,
):
    """Merge logical ``[source_rank,B,S,HC,D]`` bands into full output."""
    offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    valid = offsets < total
    dim = offsets % head_dim
    slot = offsets // head_dim
    head_in_chunk = slot % chunk_heads
    slot = slot // chunk_heads
    seq = slot % local_seq
    slot = slot // local_seq
    batch_idx = slot % batch
    source_rank = slot // batch

    recv_offset = (
        source_rank * recv_rank_stride
        + batch_idx * recv_batch_stride
        + seq * recv_seq_stride
        + head_in_chunk * recv_head_stride
        + dim * recv_dim_stride
    )
    global_head = source_rank * local_heads + head_offset + head_in_chunk
    output_offset = (
        batch_idx * out_batch_stride
        + seq * out_seq_stride
        + global_head * out_head_stride
        + dim * out_dim_stride
    )
    tl.store(
        output + output_offset,
        tl.load(received + recv_offset, mask=valid),
        mask=valid,
    )


def _pack_ulysses_qkv_head_chunk(
    output,
    query,
    key,
    value,
    *,
    world_size,
    local_heads,
    head_offset,
    head_count,
    nccl_layout=False,
):
    batch, seq_len, _, head_dim = query.shape
    total = batch * seq_len * world_size * head_count * head_dim
    grid = ((total + 255) // 256,)
    _pack_ulysses_qkv_head_chunk_kernel[grid](
        output,
        query,
        key,
        value,
        total,
        batch,
        seq_len,
        world_size,
        local_heads,
        head_count,
        head_offset,
        head_dim,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        nccl_layout,
        BLOCK=256,
    )


def _pack_ulysses_output_sequence_chunk(
    output,
    source,
    *,
    world_size,
):
    batch, global_seq, chunk_heads, head_dim = source.shape
    local_seq = global_seq // world_size
    total = source.numel()
    grid = ((total + 255) // 256,)
    _pack_ulysses_output_sequence_chunk_kernel[grid](
        output,
        source,
        total,
        batch,
        local_seq,
        world_size,
        chunk_heads,
        head_dim,
        *source.stride(),
        BLOCK=256,
    )


def _merge_ulysses_output_head_chunk(
    received_rank_major,
    output,
    *,
    world_size,
    local_heads,
    head_offset,
):
    world, batch, local_seq, chunk_heads, head_dim = received_rank_major.shape
    assert world == world_size
    total = received_rank_major.numel()
    grid = ((total + 255) // 256,)
    _merge_ulysses_output_head_chunk_kernel[grid](
        received_rank_major,
        output,
        total,
        batch,
        local_seq,
        world_size,
        local_heads,
        chunk_heads,
        head_offset,
        head_dim,
        *received_rank_major.stride(),
        *output.stride(),
        BLOCK=256,
    )
