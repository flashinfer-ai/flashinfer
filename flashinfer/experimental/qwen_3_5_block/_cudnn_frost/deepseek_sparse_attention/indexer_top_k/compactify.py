# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL row-wise compactify for top-K index tensors.

This module provides one index utility: compactify(idxs) packs valid entries
(>= 0) to the front of each row and writes -1 padding at the tail. The input
tensor is expected to already be in the index space consumed by downstream
sparse attention kernels; use local_to_global before compactifying local top-K
indices. BSHD-shaped inputs are flattened batch-major before launch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.compiler import compile_options
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import (
    device_major,
    resolve_stream as _resolve_stream,
    torch_stream_context,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor as _to_cute

WARPS_PER_CTA = 4
ROWS_PER_CTA = WARPS_PER_CTA  # One warp per row.


def is_available() -> bool:
    return torch.cuda.is_available() and device_major() >= 9


@cute.jit
def _warp_inclusive_scan_i32(val: Int32, lane_id: Int32) -> Int32:
    """Warp-wide inclusive prefix sum over 32 lanes."""
    for i in cutlass.range(5, unroll_full=True):
        offset = 1 << i
        other = cute.arch.shuffle_sync_up(val, offset, mask=0xFFFFFFFF, mask_and_clamp=0)
        if lane_id >= offset:
            val = val + other
    return val


class CompactifyKernel:
    """Row compactify kernel for a single two-dimensional tensor."""

    def __init__(self, cols: int):
        self.cols = cols
        self.chunk = (cols + 31) // 32

    @cute.jit
    def __call__(
        self,
        mIn: cute.Tensor,
        mOut: cute.Tensor,
        mLen: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        num_ctas = (rows + ROWS_PER_CTA - 1) // ROWS_PER_CTA
        self.kernel(mIn, mOut, mLen, rows).launch(
            grid=(num_ctas, 1, 1),
            block=(WARPS_PER_CTA * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mOut: cute.Tensor, mLen: cute.Tensor, rows: Int32):
        cols = const_expr(self.cols)
        chunk = const_expr(self.chunk)
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        warp_id = tidx // 32
        lane_id = tidx % 32
        row = bidx * ROWS_PER_CTA + warp_id

        if row < rows:
            local_vals = cute.make_rmem_tensor((chunk,), Int32)
            local_is_valid = cute.make_rmem_tensor((chunk,), cutlass.Boolean)
            cnt_v = Int32(0)
            cnt_i = Int32(0)

            for i in cutlass.range(chunk, unroll_full=True):
                pos = lane_id * chunk + i
                v = Int32(-1)
                in_range = pos < cols
                if in_range:
                    v = mIn[row, pos]

                local_vals[i] = v
                is_valid = in_range and (v >= Int32(0))
                local_is_valid[i] = is_valid
                if is_valid:
                    cnt_v = cnt_v + Int32(1)
                elif in_range:
                    cnt_i = cnt_i + Int32(1)

            prefix_v_incl = _warp_inclusive_scan_i32(cnt_v, Int32(lane_id))
            prefix_i_incl = _warp_inclusive_scan_i32(cnt_i, Int32(lane_id))
            my_v_start = prefix_v_incl - cnt_v
            my_i_start = prefix_i_incl - cnt_i
            total_valid = cute.arch.shuffle_sync(prefix_v_incl, 31)

            v_pos = my_v_start
            i_pos = total_valid + my_i_start
            for i in cutlass.range(chunk, unroll_full=True):
                pos = lane_id * chunk + i
                if pos < cols:
                    if local_is_valid[i]:
                        mOut[row, v_pos] = local_vals[i]
                        v_pos = v_pos + Int32(1)
                    else:
                        mOut[row, i_pos] = Int32(-1)
                        i_pos = i_pos + Int32(1)

            if lane_id == 0:
                mLen[row] = total_valid


_compile_cache: dict[tuple, object] = {}


def _compile_or_fetch(key, kernel_obj, *args):
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(kernel_obj, *args, options=compile_options("--opt-level 3"))
    return _compile_cache[key]


def compactify(
    idxs: torch.Tensor,
    stream: Optional[cuda.CUstream] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compact a non-compact int32 tensor row-wise.

    The input may be shaped (M, K) or (B, S, K). The 3D form is flattened
    batch-major to (B*S, K), matching DSA sparse attention's flat top-K
    layout. Returns (indices, topk_length).
    """
    if not is_available():
        raise RuntimeError("compactify requires SM90+ CUDA device")
    if idxs.dtype != torch.int32:
        raise TypeError(f"idxs must be int32, got {idxs.dtype}")
    if not idxs.is_cuda or idxs.ndim not in (2, 3):
        raise ValueError("idxs must be a 2D or 3D CUDA tensor")

    launch_stream = _resolve_stream(stream)
    with torch_stream_context(stream):
        idxs = idxs.contiguous()
        if idxs.ndim == 3:
            idxs = idxs.reshape(-1, idxs.shape[-1])
        rows, cols = idxs.shape
        out = torch.empty_like(idxs)
        length = torch.empty(rows, dtype=torch.int32, device=idxs.device)

        key = (cols,)
        if key not in _compile_cache:
            _compile_or_fetch(
                key,
                CompactifyKernel(cols=cols),
                _to_cute(idxs),
                _to_cute(out),
                _to_cute(length),
                Int32(int(rows)),
                launch_stream,
            )
        compiled = _compile_cache[key]
        compiled(idxs, out, length, Int32(int(rows)), launch_stream)
        return out, length


__all__ = ["is_available", "compactify"]
