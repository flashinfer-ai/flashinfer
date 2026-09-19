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
"""Triton FP4 Q0 quantization followed by route scatter."""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from ....utils import ceil_div
from ....utils import round_up as align
from ._moe_utils.moe_route_meta import count_expert_kernel as _count_expert_kernel
from ._moe_utils.moe_route_meta import count_routes_kernel as _count_routes_kernel
from ._moe_utils.moe_route_meta import prefix_cursor_kernel as _prefix_cursor_kernel
from ._moe_utils.moe_route_meta import (
    route_assign_decode_kernel as _route_assign_decode_kernel,
)
from ._moe_utils.moe_route_meta import route_assign_kernel as _route_assign_kernel
from ._moe_utils.sm12x_blockscaled_layout import compute_padded_offset

GRAN_K = 16
DIRECT_BLOCK_K = 256
NVFP4_SF_M_ALIGN = 128
SF_COL_ALIGN = 4
DECODE_BLOCK_N = 256
PREFILL_FUSED_PREFIX_MAX_EXPERTS = 1024
ALIGN_BYTES = 16


@dataclass
class Nvfp4Q0RouteWorkspace:
    counts: torch.Tensor
    offsets: torch.Tensor
    expert_cursor: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    dst_rows: torch.Tensor
    scale_dst_rows: torch.Tensor
    q_out: torch.Tensor
    scale_out: torch.Tensor


def _align_bytes(cursor: int) -> int:
    return align(cursor, ALIGN_BYTES)


def _workspace_layout(num_tokens, hidden_size, top_k, num_experts):
    total_pairs = num_tokens * top_k
    padded_rows = compute_padded_offset(total_pairs, num_experts, NVFP4_SF_M_ALIGN)
    padded_cols = align(hidden_size // GRAN_K, SF_COL_ALIGN)
    cursor, slices = 0, {}
    for name, count in (
        ("counts", num_experts),
        ("offsets", num_experts + 1),
        ("expert_cursor", num_experts),
        ("token_map", total_pairs),
        ("token_weights", total_pairs),
        ("dst_rows", total_pairs),
        ("scale_dst_rows", total_pairs),
    ):
        cursor = _align_bytes(cursor)
        slices[name] = (cursor, count * 4)
        cursor += count * 4
    cursor = _align_bytes(cursor)
    slices["scale_out"] = (cursor, padded_rows * padded_cols)
    cursor += padded_rows * padded_cols
    return slices, cursor, padded_rows, padded_cols


def nvfp4_q0_route_workspace_shapes(num_tokens, hidden_size, top_k, num_experts):
    total_pairs = num_tokens * top_k
    _, workspace2_bytes, _, _ = _workspace_layout(
        num_tokens, hidden_size, top_k, num_experts
    )
    return (
        (ceil_div(total_pairs * hidden_size // 2, 2),),
        (ceil_div(workspace2_bytes, 2),),
    )


def _byte_range(tensor):
    return tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()


def _overlaps(lhs, rhs):
    l0, l1 = _byte_range(lhs)
    r0, r1 = _byte_range(rhs)
    return l0 < r1 and r0 < l1


def _validate_external(name, tensor, required, device):
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if tensor.dtype is not torch.bfloat16:
        raise TypeError(f"{name} must be bfloat16")
    if tensor.ndim != 1 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous 1D buffer")
    if tensor.data_ptr() % ALIGN_BYTES:
        raise ValueError(f"{name} must be {ALIGN_BYTES}-byte aligned")
    if tensor.numel() < required:
        raise ValueError(f"{name} has {tensor.numel()} elements, needs {required}")


def _view(raw, spec, shape, dtype):
    start, size = spec
    return raw[start : start + size].view(dtype).view(shape)


def make_nvfp4_q0_route_workspace(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    workspace13: Optional[torch.Tensor] = None,
    workspace2: Optional[torch.Tensor] = None,
) -> Nvfp4Q0RouteWorkspace:
    num_tokens, hidden_size = x.shape
    top_k = topk_ids.shape[1]
    total_pairs = num_tokens * top_k
    shape13, shape2 = nvfp4_q0_route_workspace_shapes(
        num_tokens, hidden_size, top_k, num_experts
    )
    slices, workspace2_bytes, padded_rows, padded_cols = _workspace_layout(
        num_tokens, hidden_size, top_k, num_experts
    )
    for name, value, required in (
        ("workspace13", workspace13, shape13[0]),
        ("workspace2", workspace2, shape2[0]),
    ):
        if value is not None:
            _validate_external(name, value, required, x.device)
    external = [value for value in (workspace13, workspace2) if value is not None]
    if len(external) == 2 and _overlaps(external[0], external[1]):
        raise ValueError("workspace13 and workspace2 must not alias")
    if any(
        _overlaps(workspace, source)
        for workspace in external
        for source in (x, topk_ids)
    ):
        raise ValueError("workspace must not alias an input")
    q_bytes = total_pairs * hidden_size // 2
    q_out = (
        torch.empty((total_pairs, hidden_size // 2), dtype=torch.uint8, device=x.device)
        if workspace13 is None
        else workspace13[: shape13[0]]
        .view(torch.uint8)[:q_bytes]
        .view(total_pairs, hidden_size // 2)
    )
    raw = (
        torch.empty((workspace2_bytes,), dtype=torch.uint8, device=x.device)
        if workspace2 is None
        else workspace2[: shape2[0]].view(torch.uint8)[:workspace2_bytes]
    )
    return Nvfp4Q0RouteWorkspace(
        _view(raw, slices["counts"], (num_experts,), torch.int32),
        _view(raw, slices["offsets"], (num_experts + 1,), torch.int32),
        _view(raw, slices["expert_cursor"], (num_experts,), torch.int32),
        _view(raw, slices["token_map"], (total_pairs,), torch.int32),
        _view(raw, slices["token_weights"], (total_pairs,), torch.float32),
        _view(raw, slices["dst_rows"], topk_ids.shape, torch.int32),
        _view(raw, slices["scale_dst_rows"], topk_ids.shape, torch.int32),
        q_out,
        _view(raw, slices["scale_out"], (padded_rows, padded_cols), torch.uint8),
    )


@triton.jit
def _nvfp4_q0_route_direct_kernel(
    x,
    global_scale,
    dst_rows,
    scale_dst_rows,
    q_out,
    scale_out_fp8,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    direct_block_k: tl.constexpr,
    gran_k: tl.constexpr,
    padded_sf_cols: tl.constexpr,
    s_xm: tl.constexpr,
    s_xk: tl.constexpr,
    s_qom: tl.constexpr,
    s_qok: tl.constexpr,
    use_gdc: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    token_idx = tl.program_id(0)
    direct_block = tl.program_id(1)
    sf_id = tl.arange(0, direct_block_k // gran_k)
    sf_lane = sf_id[:, None]
    value_lane = tl.arange(0, gran_k)[None, :]
    cols = direct_block * direct_block_k + sf_lane * gran_k + value_lane
    values = tl.load(
        x + token_idx * s_xm + cols * s_xk, mask=cols < hidden_size, other=0.0
    ).to(tl.float32)
    scale = tl.load(global_scale).to(tl.float32)
    sf = tl.minimum(tl.max(tl.abs(values), axis=1) * (scale / 6.0), 448.0).to(
        tl.float8e4nv
    )
    sf_f32 = sf.to(tl.float32)
    encode = tl.where(sf_f32 == 0.0, 0.0, scale / sf_f32)
    pair_lane = tl.arange(0, gran_k // 2)[None, :]
    pair_col = direct_block * direct_block_k + sf_lane * gran_k + pair_lane * 2
    pair_valid = pair_col < hidden_size
    low = tl.load(
        x + token_idx * s_xm + pair_col * s_xk, mask=pair_valid, other=0.0
    ).to(tl.float32)
    high = tl.load(
        x + token_idx * s_xm + (pair_col + 1) * s_xk, mask=pair_valid, other=0.0
    ).to(tl.float32)
    even = low * encode[:, None]
    odd = high * encode[:, None]
    packed = tl.inline_asm_elementwise(
        """
        {
            .reg .b8 r;
            cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
            mov.b32 $0, {r, r, r, r};
        }
        """,
        constraints="=r,f,f",
        args=[odd, even],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )
    sf_col = direct_block * (direct_block_k // gran_k) + sf_id
    sf_valid = sf_col < hidden_size // gran_k
    if use_gdc:
        tl.extra.cuda.gdc_launch_dependents()
    for slot_idx in tl.static_range(0, top_k):
        pair_idx = token_idx * top_k + slot_idx
        routed_row = tl.load(dst_rows + pair_idx)
        scale_row = tl.load(scale_dst_rows + pair_idx)
        tl.store(
            q_out + routed_row * s_qom + (pair_col >> 1) * s_qok,
            packed,
            mask=pair_valid,
        )
        sf_index = (
            (sf_col & 3)
            + (sf_col >> 2) * 512
            + (scale_row & 31) * 16
            + ((scale_row & 127) >> 5) * 4
            + (scale_row >> 7) * 128 * padded_sf_cols
        )
        tl.store(scale_out_fp8 + sf_index, sf, mask=sf_valid)


def _validate(x, topk_ids, topk_weights, num_experts, global_scale):
    if (
        any(
            value.device != x.device for value in (topk_ids, topk_weights, global_scale)
        )
        or x.device.type != "cuda"
    ):
        raise ValueError("inputs must be CUDA tensors on one device")
    if x.dtype is not torch.bfloat16:
        raise TypeError("x must be bfloat16")
    if topk_ids.dtype is not torch.int32 or topk_weights.dtype is not torch.float32:
        raise TypeError("routing dtypes must be int32 and float32")
    if global_scale.dtype is not torch.float32 or tuple(global_scale.shape) != (1,):
        raise TypeError("global_scale must be float32[1]")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("x and routing tensors must be 2D")
    if topk_ids.shape != topk_weights.shape or x.shape[0] != topk_ids.shape[0]:
        raise ValueError("routing shapes mismatch")
    if num_experts <= 0 or x.shape[0] <= 0 or x.shape[1] % GRAN_K:
        raise ValueError("unsupported dimensions")


def nvfp4_q0_route_triton(
    x,
    topk_ids,
    topk_weights,
    num_experts,
    global_scale,
    workspace=None,
    workspace13=None,
    workspace2=None,
    enable_pdl=False,
):
    _validate(x, topk_ids, topk_weights, num_experts, global_scale)
    topk_ids, topk_weights = topk_ids.contiguous(), topk_weights.contiguous()
    num_tokens, hidden_size = x.shape
    top_k, total_pairs = topk_ids.shape[1], topk_ids.numel()
    if workspace is None:
        workspace = make_nvfp4_q0_route_workspace(
            x, topk_ids, num_experts, workspace13, workspace2
        )
    if any(
        _overlaps(value, source)
        for value in vars(workspace).values()
        for source in (x, topk_ids, topk_weights, global_scale)
    ):
        raise ValueError("workspace must not alias an input")
    workspace.scale_out.zero_()
    if total_pairs <= DECODE_BLOCK_N:
        total_scale = workspace.scale_out.numel()
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
            NVFP4_SF_M_ALIGN,
            DECODE_BLOCK_N,
            total_scale,
            workspace.scale_out.shape[0],
            workspace.scale_out.stride(0),
            workspace.scale_out.stride(1),
        )
    else:
        block_n = 256
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
            NVFP4_SF_M_ALIGN,
            block_n,
        )
    _nvfp4_q0_route_direct_kernel[(num_tokens, ceil_div(hidden_size, DIRECT_BLOCK_K))](
        x,
        global_scale,
        workspace.dst_rows,
        workspace.scale_dst_rows,
        workspace.q_out,
        workspace.scale_out.view(torch.float8_e4m3fn),
        hidden_size,
        top_k,
        DIRECT_BLOCK_K,
        GRAN_K,
        workspace.scale_out.shape[1],
        x.stride(0),
        x.stride(1),
        workspace.q_out.stride(0),
        workspace.q_out.stride(1),
        use_gdc=enable_pdl,
        launch_pdl=False,
        num_warps=4,
    )
    stream = torch.cuda.current_stream(x.device)
    for tensor in vars(workspace).values():
        tensor.record_stream(stream)
    return (
        workspace.offsets,
        workspace.token_map,
        workspace.token_weights,
        workspace.q_out,
        workspace.scale_out,
    )


__all__ = [
    "Nvfp4Q0RouteWorkspace",
    "make_nvfp4_q0_route_workspace",
    "nvfp4_q0_route_triton",
    "nvfp4_q0_route_workspace_shapes",
]
