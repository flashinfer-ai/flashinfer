# Copyright (c) 2025 by FlashInfer team.
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
"""Triton MXFP8 q0 followed by route scatter."""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from ....utils import ceil_div
from ._moe_utils.sm12x_blockscaled_layout import (
    SF_M_ALIGN,
    UE8M0_PACK_NUM,
    compute_padded_offset,
)

TILE_K = 128
BLOCK_K = TILE_K * UE8M0_PACK_NUM
DIRECT_BLOCK_K = BLOCK_K * 2
DECODE_BLOCK_N = 256


@dataclass
class Mxfp8Q0RouteWorkspace:
    counts: torch.Tensor
    offsets: torch.Tensor
    expert_cursor: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    dst_rows: torch.Tensor
    scale_dst_rows: torch.Tensor
    q_out: torch.Tensor
    scale_out: torch.Tensor


def mxfp8_q0_route_workspace_shapes(
    num_tokens: int, hidden_size: int, top_k: int, num_experts: int
):
    total_pairs = num_tokens * top_k
    num_k_blocks = ceil_div(hidden_size, BLOCK_K)
    padded_rows = compute_padded_offset(total_pairs, num_experts, SF_M_ALIGN)
    fp8_elems = total_pairs * hidden_size
    cursor = 0
    cursor += num_experts
    cursor = _align_int32(cursor)
    cursor += num_experts + 1
    cursor += num_experts
    cursor = _align_int32(cursor)
    cursor += total_pairs
    cursor = _align_int32(cursor)
    cursor += total_pairs
    cursor += total_pairs
    cursor += total_pairs
    cursor = _align_int32(cursor)
    int32_elems = cursor + num_k_blocks * padded_rows
    return (ceil_div(fp8_elems, 2),), (int32_elems * 2,)


def _slice_view(flat: torch.Tensor, start: int, size: int, shape, dtype: torch.dtype):
    return flat[start : start + size].view(dtype=dtype).view(shape)


def _align_int32(cursor: int) -> int:
    return ceil_div(cursor, 4) * 4


def _record_workspace_stream(workspace: Mxfp8Q0RouteWorkspace) -> None:
    stream = torch.cuda.current_stream(workspace.q_out.device)
    for tensor in vars(workspace).values():
        tensor.record_stream(stream)


def make_mxfp8_q0_route_workspace(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    workspace13: Optional[torch.Tensor] = None,
    workspace2: Optional[torch.Tensor] = None,
) -> Mxfp8Q0RouteWorkspace:
    num_tokens, hidden_size = x.shape
    top_k = topk_ids.shape[1]
    total_pairs = num_tokens * top_k
    num_k_blocks = ceil_div(hidden_size, BLOCK_K)
    padded_rows = compute_padded_offset(total_pairs, num_experts, SF_M_ALIGN)

    if workspace13 is None:
        q_out = torch.empty(
            (total_pairs, hidden_size), dtype=torch.float8_e4m3fn, device=x.device
        )
    else:
        fp8_flat = workspace13.view(dtype=torch.float8_e4m3fn).flatten()
        q_out_elems = total_pairs * hidden_size
        assert fp8_flat.numel() >= q_out_elems
        q_out = fp8_flat[:q_out_elems].view(total_pairs, hidden_size)

    if workspace2 is None:
        counts = torch.empty((num_experts,), dtype=torch.int32, device=x.device)
        offsets = torch.empty((num_experts + 1,), dtype=torch.int32, device=x.device)
        expert_cursor = torch.empty((num_experts,), dtype=torch.int32, device=x.device)
        token_map = torch.empty((total_pairs,), dtype=torch.int32, device=x.device)
        token_weights = torch.empty(
            (total_pairs,), dtype=torch.float32, device=x.device
        )
        dst_rows = torch.empty_like(topk_ids, dtype=torch.int32)
        scale_dst_rows = torch.empty_like(topk_ids, dtype=torch.int32)
        scale_out = torch.empty(
            (num_k_blocks, padded_rows), dtype=torch.int32, device=x.device
        )
    else:
        int_flat = workspace2.view(dtype=torch.int32).flatten()
        required = (
            mxfp8_q0_route_workspace_shapes(
                num_tokens, hidden_size, top_k, num_experts
            )[1][0]
            // 2
        )
        assert int_flat.numel() >= required
        cursor = 0
        counts = _slice_view(int_flat, cursor, num_experts, (num_experts,), torch.int32)
        cursor += num_experts
        cursor = _align_int32(cursor)
        offsets = _slice_view(
            int_flat, cursor, num_experts + 1, (num_experts + 1,), torch.int32
        )
        cursor += num_experts + 1
        expert_cursor = _slice_view(
            int_flat, cursor, num_experts, (num_experts,), torch.int32
        )
        cursor += num_experts
        cursor = _align_int32(cursor)
        token_map = _slice_view(
            int_flat, cursor, total_pairs, (total_pairs,), torch.int32
        )
        cursor += total_pairs
        cursor = _align_int32(cursor)
        token_weights = _slice_view(
            int_flat, cursor, total_pairs, (total_pairs,), torch.float32
        )
        cursor += total_pairs
        dst_rows = _slice_view(
            int_flat, cursor, total_pairs, topk_ids.shape, torch.int32
        )
        cursor += total_pairs
        scale_dst_rows = _slice_view(
            int_flat, cursor, total_pairs, topk_ids.shape, torch.int32
        )
        cursor += total_pairs
        cursor = _align_int32(cursor)
        scale_out = _slice_view(
            int_flat,
            cursor,
            num_k_blocks * padded_rows,
            (num_k_blocks, padded_rows),
            torch.int32,
        )

    return Mxfp8Q0RouteWorkspace(
        counts,
        offsets,
        expert_cursor,
        token_map,
        token_weights,
        dst_rows,
        scale_dst_rows,
        q_out,
        scale_out,
    )


@triton.jit
def _count_expert_kernel(
    topk_ids, counts, total_pairs: tl.constexpr, block_n: tl.constexpr
):
    expert_idx = tl.program_id(0)
    offs = tl.arange(0, block_n)
    acc = tl.zeros((block_n,), dtype=tl.int32)
    for pair_begin in tl.range(0, total_pairs, block_n):
        pair_idx = pair_begin + offs
        expert = tl.load(topk_ids + pair_idx, mask=pair_idx < total_pairs, other=-1)
        acc += tl.where(expert == expert_idx, 1, 0)
    tl.store(counts + expert_idx, tl.sum(acc))


@triton.jit
def _route_assign_kernel(
    topk_ids,
    topk_weights,
    offsets,
    expert_cursor,
    token_map,
    token_weights,
    dst_rows,
    scale_dst_rows,
    total_pairs: tl.constexpr,
    top_k: tl.constexpr,
    scale_align: tl.constexpr,
    block_n: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, block_n)
    pair_idx = pid * block_n + offs
    valid = pair_idx < total_pairs
    expert = tl.load(topk_ids + pair_idx, mask=valid, other=0)
    routed_row = tl.atomic_add(expert_cursor + expert, 1, mask=valid, sem="relaxed")
    token_idx = pair_idx // top_k
    expert_begin = tl.load(offsets + expert, mask=valid, other=0)
    scale_begin = (
        (expert_begin + expert * (scale_align - 1)) // scale_align
    ) * scale_align
    tl.store(token_map + routed_row, token_idx, mask=valid)
    tl.store(
        token_weights + routed_row,
        tl.load(topk_weights + pair_idx, mask=valid, other=0.0),
        mask=valid,
    )
    tl.store(dst_rows + pair_idx, routed_row, mask=valid)
    tl.store(
        scale_dst_rows + pair_idx, scale_begin + routed_row - expert_begin, mask=valid
    )


@triton.jit
def _route_assign_decode_kernel(
    topk_ids,
    topk_weights,
    offsets,
    token_map,
    token_weights,
    dst_rows,
    scale_dst_rows,
    scale_out,
    total_pairs: tl.constexpr,
    top_k: tl.constexpr,
    num_experts: tl.constexpr,
    scale_align: tl.constexpr,
    block_n: tl.constexpr,
    total_scale: tl.constexpr,
    padded_rows: tl.constexpr,
    s_som: tl.constexpr,
    s_sok: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, block_n)
    valid = offs < total_pairs
    experts = tl.load(topk_ids + offs, mask=valid, other=num_experts)

    flat = pid * block_n + offs
    k_block = flat // padded_rows
    scale_row = flat - k_block * padded_rows
    tl.store(
        scale_out + k_block * s_som + scale_row * s_sok, 0, mask=flat < total_scale
    )

    expert_prefix = tl.sum(tl.where(valid & (experts < pid), 1, 0), 0)
    tl.store(offsets + pid, expert_prefix, mask=pid < num_experts)
    tl.store(offsets + num_experts, total_pairs, mask=pid == 0)

    pair_expert = tl.load(topk_ids + pid, mask=pid < total_pairs, other=0)
    expert_begin = tl.sum(tl.where(valid & (experts < pair_expert), 1, 0), 0)
    rank = tl.sum(tl.where(valid & (experts == pair_expert) & (offs < pid), 1, 0), 0)
    routed_row = expert_begin + rank
    scale_begin = (
        (expert_begin + pair_expert * (scale_align - 1)) // scale_align
    ) * scale_align
    token_idx = pid // top_k
    tl.store(token_map + routed_row, token_idx, mask=pid < total_pairs)
    tl.store(
        token_weights + routed_row,
        tl.load(topk_weights + pid, mask=pid < total_pairs, other=0.0),
        mask=pid < total_pairs,
    )
    tl.store(dst_rows + pid, routed_row, mask=pid < total_pairs)
    tl.store(scale_dst_rows + pid, scale_begin + rank, mask=pid < total_pairs)


@triton.jit
def _mxfp8_q0_route_direct_kernel(
    x,
    dst_rows,
    scale_dst_rows,
    q_out,
    scale_out,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    tile_k: tl.constexpr,
    direct_block_k: tl.constexpr,
    num_tile_per_pack_sf: tl.constexpr,
    num_scale_blocks: tl.constexpr,
    s_xm: tl.constexpr,
    s_xk: tl.constexpr,
    s_qom: tl.constexpr,
    s_qok: tl.constexpr,
    s_som: tl.constexpr,
    s_sok: tl.constexpr,
    use_gdc: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    token_idx = tl.program_id(0)
    k_block_1024 = tl.program_id(1)
    tile_ids = tl.arange(0, num_tile_per_pack_sf * 2)
    k_tile = tile_ids[:, None]
    k_lane = tl.arange(0, tile_k)[None, :]
    cols = k_block_1024 * direct_block_k + k_tile * tile_k + k_lane
    valid_cols = cols < hidden_size
    x_vals = tl.load(x + token_idx * s_xm + cols * s_xk, mask=valid_cols, other=0.0).to(
        tl.float32
    )
    amax = tl.maximum(tl.max(tl.abs(x_vals), axis=1), 1.0e-4)
    sf = amax / 448.0
    bits = sf.to(tl.int32, bitcast=True)
    exp = ((bits >> 23) & 0xFF) + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    sf_e8 = (exp << 23).to(tl.float32, bitcast=True)
    q_vals = (x_vals * (1.0 / sf_e8[:, None])).to(q_out.dtype.element_ty)
    bit_shift = (tile_ids & 3) * 8
    packed_sf0 = tl.sum(
        tl.where(tile_ids < num_tile_per_pack_sf, exp << bit_shift, 0), axis=0
    )
    packed_sf1 = tl.sum(
        tl.where(tile_ids >= num_tile_per_pack_sf, exp << bit_shift, 0), axis=0
    )
    scale_block0 = k_block_1024 * 2
    scale_block1 = scale_block0 + 1
    if use_gdc:
        tl.extra.cuda.gdc_launch_dependents()
    for slot_idx in tl.static_range(0, top_k):
        pair_idx = token_idx * top_k + slot_idx
        routed_row = tl.load(dst_rows + pair_idx)
        scale_row = tl.load(scale_dst_rows + pair_idx)
        tl.store(q_out + routed_row * s_qom + cols * s_qok, q_vals, mask=valid_cols)
        tl.store(
            scale_out + scale_block0 * s_som + scale_row * s_sok,
            packed_sf0,
            mask=scale_block0 < num_scale_blocks,
        )
        tl.store(
            scale_out + scale_block1 * s_som + scale_row * s_sok,
            packed_sf1,
            mask=scale_block1 < num_scale_blocks,
        )


def _validate(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> None:
    if x.device.type != "cuda":
        raise ValueError("x must be a CUDA tensor")
    if x.dtype is not torch.bfloat16:
        raise TypeError(f"x must be bfloat16, got {x.dtype}")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("routing tensors must be on the same CUDA device as x")
    if topk_ids.dtype is not torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if topk_weights.dtype is not torch.float32:
        raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("x, topk_ids, and topk_weights must be 2D")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have equal shape")
    if x.shape[0] != topk_ids.shape[0]:
        raise ValueError("routing tensors must have one row per token")
    assert num_experts > 0
    assert x.shape[0] > 0
    assert x.shape[1] % BLOCK_K == 0


def mxfp8_q0_route_triton(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    workspace: Optional[Mxfp8Q0RouteWorkspace] = None,
    workspace13: Optional[torch.Tensor] = None,
    workspace2: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
):
    _validate(x, topk_ids, topk_weights, num_experts)
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    num_tokens, hidden_size = x.shape
    top_k = topk_ids.shape[1]
    total_pairs = num_tokens * top_k
    num_k_blocks = ceil_div(hidden_size, BLOCK_K)
    num_direct_blocks = ceil_div(hidden_size, DIRECT_BLOCK_K)
    if workspace is None:
        workspace = make_mxfp8_q0_route_workspace(
            x, topk_ids, num_experts, workspace13=workspace13, workspace2=workspace2
        )
    if total_pairs <= DECODE_BLOCK_N:
        total_scale = num_k_blocks * workspace.scale_out.shape[1]
        decode_grid = max(
            num_experts, total_pairs, ceil_div(total_scale, DECODE_BLOCK_N)
        )
        _route_assign_decode_kernel[(decode_grid,)](
            topk_ids,
            topk_weights,
            workspace.offsets,
            workspace.token_map,
            workspace.token_weights,
            workspace.dst_rows,
            workspace.scale_dst_rows,
            workspace.scale_out,
            total_pairs,
            top_k,
            num_experts,
            SF_M_ALIGN,
            DECODE_BLOCK_N,
            total_scale,
            workspace.scale_out.shape[1],
            workspace.scale_out.stride(0),
            workspace.scale_out.stride(1),
        )
        _mxfp8_q0_route_direct_kernel[(num_tokens, num_direct_blocks)](
            x,
            workspace.dst_rows,
            workspace.scale_dst_rows,
            workspace.q_out,
            workspace.scale_out,
            hidden_size,
            top_k,
            TILE_K,
            DIRECT_BLOCK_K,
            UE8M0_PACK_NUM,
            num_k_blocks,
            x.stride(0),
            x.stride(1),
            workspace.q_out.stride(0),
            workspace.q_out.stride(1),
            workspace.scale_out.stride(0),
            workspace.scale_out.stride(1),
            use_gdc=enable_pdl,
            launch_pdl=False,
            num_warps=4,
        )
        _record_workspace_stream(workspace)
        return (
            workspace.offsets,
            workspace.token_map,
            workspace.token_weights,
            workspace.q_out,
            workspace.scale_out,
        )
    block_n = 256
    workspace.offsets[:1].zero_()
    workspace.scale_out.zero_()
    _count_expert_kernel[(num_experts,)](
        topk_ids, workspace.counts, total_pairs, block_n
    )
    workspace.offsets[1:] = workspace.counts.cumsum(0)
    workspace.expert_cursor.copy_(workspace.offsets[:-1])
    _route_assign_kernel[(ceil_div(total_pairs, block_n),)](
        topk_ids,
        topk_weights,
        workspace.offsets,
        workspace.expert_cursor,
        workspace.token_map,
        workspace.token_weights,
        workspace.dst_rows,
        workspace.scale_dst_rows,
        total_pairs,
        top_k,
        SF_M_ALIGN,
        block_n,
    )
    _mxfp8_q0_route_direct_kernel[(num_tokens, num_direct_blocks)](
        x,
        workspace.dst_rows,
        workspace.scale_dst_rows,
        workspace.q_out,
        workspace.scale_out,
        hidden_size,
        top_k,
        TILE_K,
        DIRECT_BLOCK_K,
        UE8M0_PACK_NUM,
        num_k_blocks,
        x.stride(0),
        x.stride(1),
        workspace.q_out.stride(0),
        workspace.q_out.stride(1),
        workspace.scale_out.stride(0),
        workspace.scale_out.stride(1),
        use_gdc=enable_pdl,
        launch_pdl=False,
        num_warps=4,
    )
    _record_workspace_stream(workspace)
    return (
        workspace.offsets,
        workspace.token_map,
        workspace.token_weights,
        workspace.q_out,
        workspace.scale_out,
    )


__all__ = [
    "Mxfp8Q0RouteWorkspace",
    "DIRECT_BLOCK_K",
    "make_mxfp8_q0_route_workspace",
    "mxfp8_q0_route_triton",
    "mxfp8_q0_route_workspace_shapes",
]
