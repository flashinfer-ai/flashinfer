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
"""Triton MXFP8 q0 followed by atomic route scatter."""

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


@triton.jit
def _mxfp8_q0_token_kernel(
    x,
    q_token,
    scale_token,
    hidden_size: tl.constexpr,
    tile_k: tl.constexpr,
    block_k: tl.constexpr,
    num_tile_per_pack_sf: tl.constexpr,
    s_xm: tl.constexpr,
    s_xk: tl.constexpr,
    s_qm: tl.constexpr,
    s_qk: tl.constexpr,
    s_sm: tl.constexpr,
    s_sk: tl.constexpr,
):
    token_idx = tl.program_id(0)
    k_block = tl.program_id(1)
    offs = tl.arange(0, block_k)
    cols = k_block * block_k + offs
    x_vals = tl.load(
        x + token_idx * s_xm + cols * s_xk, mask=cols < hidden_size, other=0.0
    ).to(tl.float32)
    packed_sf = tl.full((), 0, tl.int32)
    for k_tile_idx in tl.static_range(0, num_tile_per_pack_sf):
        tile_begin = k_tile_idx * tile_k
        in_tile = (offs >= tile_begin) & (offs < tile_begin + tile_k)
        amax = tl.maximum(tl.max(tl.where(in_tile, tl.abs(x_vals), 0.0)), 1.0e-4)
        sf = amax / 448.0
        bits = sf.to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 0xFF) + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        sf_e8 = (exp << 23).to(tl.float32, bitcast=True)
        q_vals = (x_vals * (1.0 / sf_e8)).to(q_token.dtype.element_ty)
        packed_sf = packed_sf | (exp << (k_tile_idx * 8))
        tl.store(
            q_token + token_idx * s_qm + cols * s_qk,
            q_vals,
            mask=(cols < hidden_size) & in_tile,
        )
    tl.store(scale_token + token_idx * s_sm + k_block * s_sk, packed_sf)


@triton.jit
def _count_expert_kernel(
    topk_ids,
    counts,
    total_pairs: tl.constexpr,
    block_n: tl.constexpr,
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
def _scatter_q0_kernel(
    q_token,
    scale_token,
    dst_rows,
    scale_dst_rows,
    q_out,
    scale_out,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    block_k: tl.constexpr,
    s_qtm: tl.constexpr,
    s_qtk: tl.constexpr,
    s_stm: tl.constexpr,
    s_stk: tl.constexpr,
    s_qom: tl.constexpr,
    s_qok: tl.constexpr,
    s_som: tl.constexpr,
    s_sok: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    k_block = tl.program_id(1)
    offs = tl.arange(0, block_k)
    cols = k_block * block_k + offs
    token_idx = pair_idx // top_k
    routed_row = tl.load(dst_rows + pair_idx)
    q_vals = tl.load(
        q_token + token_idx * s_qtm + cols * s_qtk,
        mask=cols < hidden_size,
        other=0.0,
    )
    tl.store(q_out + routed_row * s_qom + cols * s_qok, q_vals, mask=cols < hidden_size)
    packed_sf = tl.load(scale_token + token_idx * s_stm + k_block * s_stk)
    scale_row = tl.load(scale_dst_rows + pair_idx)
    tl.store(scale_out + k_block * s_som + scale_row * s_sok, packed_sf)


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
):
    _validate(x, topk_ids, topk_weights, num_experts)
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    num_tokens, hidden_size = x.shape
    top_k = topk_ids.shape[1]
    total_pairs = num_tokens * top_k
    num_k_blocks = ceil_div(hidden_size, BLOCK_K)
    padded_rows = compute_padded_offset(total_pairs, num_experts, SF_M_ALIGN)
    q_token = torch.empty(
        (num_tokens, hidden_size), dtype=torch.float8_e4m3fn, device=x.device
    )
    scale_token = torch.empty(
        (num_tokens, num_k_blocks), dtype=torch.int32, device=x.device
    )
    counts = torch.empty((num_experts,), dtype=torch.int32, device=x.device)
    offsets = torch.zeros((num_experts + 1,), dtype=torch.int32, device=x.device)
    token_map = torch.empty((total_pairs,), dtype=torch.int32, device=x.device)
    token_weights = torch.empty((total_pairs,), dtype=torch.float32, device=x.device)
    dst_rows = torch.empty_like(topk_ids, dtype=torch.int32)
    scale_dst_rows = torch.empty_like(topk_ids, dtype=torch.int32)
    q_out = torch.empty(
        (total_pairs, hidden_size), dtype=torch.float8_e4m3fn, device=x.device
    )
    scale_out = torch.zeros(
        (num_k_blocks, padded_rows), dtype=torch.int32, device=x.device
    )
    block_n = 256
    _mxfp8_q0_token_kernel[(num_tokens, num_k_blocks)](
        x,
        q_token,
        scale_token,
        hidden_size,
        TILE_K,
        BLOCK_K,
        UE8M0_PACK_NUM,
        x.stride(0),
        x.stride(1),
        q_token.stride(0),
        q_token.stride(1),
        scale_token.stride(0),
        scale_token.stride(1),
    )
    _count_expert_kernel[(num_experts,)](topk_ids, counts, total_pairs, block_n)
    offsets[1:] = counts.cumsum(0)
    expert_cursor = offsets[:-1].clone()
    _route_assign_kernel[(ceil_div(total_pairs, block_n),)](
        topk_ids,
        topk_weights,
        offsets,
        expert_cursor,
        token_map,
        token_weights,
        dst_rows,
        scale_dst_rows,
        total_pairs,
        top_k,
        SF_M_ALIGN,
        block_n,
    )
    _scatter_q0_kernel[(total_pairs, num_k_blocks)](
        q_token,
        scale_token,
        dst_rows,
        scale_dst_rows,
        q_out,
        scale_out,
        hidden_size,
        top_k,
        BLOCK_K,
        q_token.stride(0),
        q_token.stride(1),
        scale_token.stride(0),
        scale_token.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        scale_out.stride(0),
        scale_out.stride(1),
    )
    return offsets, token_map, token_weights, q_out, scale_out


__all__ = ["mxfp8_q0_route_triton"]
