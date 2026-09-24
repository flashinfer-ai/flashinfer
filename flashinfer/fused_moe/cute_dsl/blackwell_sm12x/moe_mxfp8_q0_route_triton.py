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
from ._moe_utils.moe_route_meta import (
    count_expert_kernel as _count_expert_kernel,
)
from ._moe_utils.moe_route_meta import (
    count_routes_kernel as _count_routes_kernel,
)
from ._moe_utils.moe_route_meta import (
    prefix_cursor_kernel as _prefix_cursor_kernel,
)
from ._moe_utils.moe_route_meta import (
    route_assign_decode_kernel as _route_assign_decode_kernel,
)
from ._moe_utils.moe_route_meta import (
    route_assign_kernel as _route_assign_kernel,
)
from ._moe_utils.sm12x_blockscaled_layout import (
    SF_M_ALIGN,
    UE8M0_PACK_NUM,
    compute_padded_offset,
)

TILE_K = 128
BLOCK_K = TILE_K * UE8M0_PACK_NUM
DIRECT_BLOCK_K = BLOCK_K * 2
DECODE_BLOCK_N = 256
PREFILL_FUSED_PREFIX_MAX_EXPERTS = 1024


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
    workspace.scale_out.zero_()
    if num_experts <= PREFILL_FUSED_PREFIX_MAX_EXPERTS:
        workspace.counts.zero_()
        _count_routes_kernel[(ceil_div(total_pairs, block_n),)](
            topk_ids, workspace.counts, total_pairs, block_n, num_warps=4
        )
        _prefix_cursor_kernel[(1,)](
            workspace.counts,
            workspace.offsets,
            workspace.expert_cursor,
            num_experts,
            triton.next_power_of_2(num_experts),
            num_warps=8,
        )
    else:
        workspace.offsets[:1].zero_()
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
