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
"""Triton true-FP32 FP8 Q0 followed by route scatter."""

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
    route_assign_decode_kernel as _route_assign_decode_kernel,
)
from ._moe_utils.moe_route_meta import (
    route_assign_kernel as _route_assign_kernel,
)
from ._moe_utils.sm12x_blockscaled_layout import SF_M_ALIGN, compute_padded_offset

TILE_K = 128
DIRECT_BLOCK_K = 1024
DECODE_BLOCK_N = 256
ALIGN_BYTES = 16


@dataclass
class Fp8Q0RouteWorkspace:
    counts: torch.Tensor
    offsets: torch.Tensor
    expert_cursor: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    dst_rows: torch.Tensor
    scale_dst_rows: torch.Tensor
    q_out: torch.Tensor
    scale_out: torch.Tensor


def _align_int32(cursor: int) -> int:
    return ceil_div(cursor, ALIGN_BYTES // 4) * (ALIGN_BYTES // 4)


def _workspace_layout(num_tokens: int, hidden_size: int, top_k: int, num_experts: int):
    total_pairs = num_tokens * top_k
    scale_blocks = ceil_div(hidden_size, TILE_K)
    padded_rows = compute_padded_offset(total_pairs, num_experts, SF_M_ALIGN)
    cursor = 0
    slices: dict[str, tuple[int, int]] = {}
    slices["counts"] = (cursor, num_experts)
    cursor += num_experts
    cursor = _align_int32(cursor)
    slices["offsets"] = (cursor, num_experts + 1)
    cursor += num_experts + 1
    slices["expert_cursor"] = (cursor, num_experts)
    cursor += num_experts
    cursor = _align_int32(cursor)
    slices["token_map"] = (cursor, total_pairs)
    cursor += total_pairs
    cursor = _align_int32(cursor)
    slices["token_weights"] = (cursor, total_pairs)
    cursor += total_pairs
    slices["dst_rows"] = (cursor, total_pairs)
    cursor += total_pairs
    slices["scale_dst_rows"] = (cursor, total_pairs)
    cursor += total_pairs
    cursor = _align_int32(cursor)
    slices["scale_out"] = (cursor, scale_blocks * padded_rows)
    cursor += scale_blocks * padded_rows
    return slices, cursor, scale_blocks, padded_rows


def fp8_q0_route_workspace_shapes(
    num_tokens: int, hidden_size: int, top_k: int, num_experts: int
):
    total_pairs = num_tokens * top_k
    _, workspace2_i32, _, _ = _workspace_layout(
        num_tokens, hidden_size, top_k, num_experts
    )
    return (ceil_div(total_pairs * hidden_size, 2),), (workspace2_i32 * 2,)


def _byte_range(tensor: torch.Tensor):
    return tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()


def _overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    l0, l1 = _byte_range(lhs)
    r0, r1 = _byte_range(rhs)
    return l0 < r1 and r0 < l1


def _validate_external(
    name: str, tensor: torch.Tensor, required: int, device: torch.device
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if tensor.dtype is not torch.bfloat16:
        raise TypeError(f"{name} must be bfloat16")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D buffer")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % ALIGN_BYTES:
        raise ValueError(f"{name} must be {ALIGN_BYTES}-byte aligned")
    if tensor.numel() < required:
        raise ValueError(f"{name} has {tensor.numel()} elements, needs {required}")


def _slice_view(flat: torch.Tensor, spec: tuple[int, int], shape, dtype: torch.dtype):
    start, size = spec
    return flat[start : start + size].view(dtype=dtype).view(shape)


def _record_workspace_stream(workspace: Fp8Q0RouteWorkspace) -> None:
    stream = torch.cuda.current_stream(workspace.q_out.device)
    for tensor in vars(workspace).values():
        tensor.record_stream(stream)


def make_fp8_q0_route_workspace(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    workspace13: Optional[torch.Tensor] = None,
    workspace2: Optional[torch.Tensor] = None,
) -> Fp8Q0RouteWorkspace:
    num_tokens, hidden_size = x.shape
    top_k = topk_ids.shape[1]
    total_pairs = num_tokens * top_k
    shape13, shape2 = fp8_q0_route_workspace_shapes(
        num_tokens, hidden_size, top_k, num_experts
    )
    slices, _, scale_blocks, padded_rows = _workspace_layout(
        num_tokens, hidden_size, top_k, num_experts
    )
    if workspace13 is not None:
        _validate_external("workspace13", workspace13, shape13[0], x.device)
    if workspace2 is not None:
        _validate_external("workspace2", workspace2, shape2[0], x.device)
    external = [t for t in (workspace13, workspace2) if t is not None]
    if len(external) == 2 and _overlaps(external[0], external[1]):
        raise ValueError("workspace13 and workspace2 must not alias")
    if any(
        _overlaps(workspace, source)
        for workspace in external
        for source in (x, topk_ids)
    ):
        raise ValueError("workspace must not alias an input")

    if workspace13 is None:
        q_out = torch.empty(
            (total_pairs, hidden_size), dtype=torch.float8_e4m3fn, device=x.device
        )
    else:
        q_out = workspace13[: shape13[0]].view(torch.float8_e4m3fn)
        q_out = q_out[: total_pairs * hidden_size].view(total_pairs, hidden_size)
    if workspace2 is None:
        raw = torch.empty((shape2[0] // 2,), dtype=torch.int32, device=x.device)
    else:
        raw = workspace2[: shape2[0]].view(torch.int32)
    counts = _slice_view(raw, slices["counts"], (num_experts,), torch.int32)
    offsets = _slice_view(raw, slices["offsets"], (num_experts + 1,), torch.int32)
    expert_cursor = _slice_view(
        raw, slices["expert_cursor"], (num_experts,), torch.int32
    )
    token_map = _slice_view(raw, slices["token_map"], (total_pairs,), torch.int32)
    token_weights = _slice_view(
        raw, slices["token_weights"], (total_pairs,), torch.float32
    )
    dst_rows = _slice_view(raw, slices["dst_rows"], topk_ids.shape, torch.int32)
    scale_dst_rows = _slice_view(
        raw, slices["scale_dst_rows"], topk_ids.shape, torch.int32
    )
    scale_out = _slice_view(
        raw, slices["scale_out"], (scale_blocks, padded_rows), torch.float32
    )
    return Fp8Q0RouteWorkspace(
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
def _fp8_q0_route_direct_kernel(
    x,
    dst_rows,
    scale_dst_rows,
    q_out,
    scale_out,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    tile_k: tl.constexpr,
    direct_block_k: tl.constexpr,
    num_tiles: tl.constexpr,
    num_scale_blocks: tl.constexpr,
    s_xm: tl.constexpr,
    s_xk: tl.constexpr,
    s_qom: tl.constexpr,
    s_qok: tl.constexpr,
    s_som: tl.constexpr,
    s_sok: tl.constexpr,
):
    token_idx = tl.program_id(0)
    direct_block = tl.program_id(1)
    tile_ids = tl.arange(0, num_tiles)
    k_lane = tl.arange(0, tile_k)
    cols = direct_block * direct_block_k + tile_ids[:, None] * tile_k + k_lane[None, :]
    valid_cols = cols < hidden_size
    values = tl.load(x + token_idx * s_xm + cols * s_xk, mask=valid_cols, other=0.0).to(
        tl.float32
    )
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), 1.0e-4)
    scales = amax / 448.0
    quant = (values * (1.0 / scales[:, None])).to(q_out.dtype.element_ty)
    scale_blocks = direct_block * num_tiles + tile_ids
    for slot_idx in tl.static_range(0, top_k):
        pair_idx = token_idx * top_k + slot_idx
        routed_row = tl.load(dst_rows + pair_idx)
        scale_row = tl.load(scale_dst_rows + pair_idx)
        tl.store(q_out + routed_row * s_qom + cols * s_qok, quant, mask=valid_cols)
        tl.store(
            scale_out + scale_blocks * s_som + scale_row * s_sok,
            scales,
            mask=scale_blocks < num_scale_blocks,
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
    if topk_ids.shape != topk_weights.shape or x.shape[0] != topk_ids.shape[0]:
        raise ValueError("routing shapes mismatch")
    if num_experts <= 0 or x.shape[0] <= 0 or x.shape[1] % TILE_K:
        raise ValueError("unsupported dimensions")


def fp8_q0_route_triton(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    workspace: Optional[Fp8Q0RouteWorkspace] = None,
    workspace13: Optional[torch.Tensor] = None,
    workspace2: Optional[torch.Tensor] = None,
):
    _validate(x, topk_ids, topk_weights, num_experts)
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    num_tokens, hidden_size = x.shape
    top_k = topk_ids.shape[1]
    total_pairs = num_tokens * top_k
    scale_blocks = ceil_div(hidden_size, TILE_K)
    direct_blocks = ceil_div(hidden_size, DIRECT_BLOCK_K)
    if workspace is None:
        workspace = make_fp8_q0_route_workspace(
            x, topk_ids, num_experts, workspace13=workspace13, workspace2=workspace2
        )
    if any(
        _overlaps(view, source)
        for view in vars(workspace).values()
        for source in (x, topk_ids, topk_weights)
    ):
        raise ValueError("workspace must not alias an input")
    if total_pairs <= DECODE_BLOCK_N:
        total_scale = scale_blocks * workspace.scale_out.shape[1]
        grid = max(num_experts, total_pairs, ceil_div(total_scale, DECODE_BLOCK_N))
        _route_assign_decode_kernel[(grid,)](
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
    else:
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
    _fp8_q0_route_direct_kernel[(num_tokens, direct_blocks)](
        x,
        workspace.dst_rows,
        workspace.scale_dst_rows,
        workspace.q_out,
        workspace.scale_out,
        hidden_size,
        top_k,
        TILE_K,
        DIRECT_BLOCK_K,
        DIRECT_BLOCK_K // TILE_K,
        scale_blocks,
        x.stride(0),
        x.stride(1),
        workspace.q_out.stride(0),
        workspace.q_out.stride(1),
        workspace.scale_out.stride(0),
        workspace.scale_out.stride(1),
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
    "Fp8Q0RouteWorkspace",
    "fp8_q0_route_triton",
    "fp8_q0_route_workspace_shapes",
    "make_fp8_q0_route_workspace",
]
