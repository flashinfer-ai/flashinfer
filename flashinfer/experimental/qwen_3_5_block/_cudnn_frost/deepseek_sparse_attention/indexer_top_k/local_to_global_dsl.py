# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL local-to-global topK index conversion.

The topK backends return local K ids.  This module converts them to the public
global id space with one small DSL kernel:

* BSHD: global = local + batch_idx * seqlen_k
* THD:  global = local + cu_seqlens_k[batch_idx]
"""

from __future__ import annotations

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.compiler import compile_options
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import (
    device_major as _get_device_capability,
    resolve_stream,
    torch_stream_context,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor as _to_cute_tensor


class LocalToGlobalTopK:
    def __init__(self, is_varlen: bool, threads_per_cta: int = 256, preserve_negative: bool = False):
        self.is_varlen = is_varlen
        self.preserve_negative = preserve_negative
        self.threads_per_cta = threads_per_cta

    @cute.jit
    def __call__(
        self,
        mLocal: cute.Tensor,
        mGlobal: cute.Tensor,
        seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        is_varlen = mCuSeqlensQ is not None
        if const_expr(is_varlen):
            assert self.is_varlen
            assert mCuSeqlensQ is not None and mCuSeqlensK is not None
            num_rows = cute.size(mLocal.shape[0])
            topk = cute.size(mLocal.shape[1])
        else:
            assert not self.is_varlen
            assert mCuSeqlensQ is None and mCuSeqlensK is None
            num_rows = cute.size(mLocal.shape[0]) * cute.size(mLocal.shape[1])
            topk = cute.size(mLocal.shape[2])

        topk_blocks = cute.ceil_div(topk, self.threads_per_cta)
        self.kernel(
            mLocal,
            mGlobal,
            seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
        ).launch(
            grid=(num_rows, topk_blocks, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mLocal,
        mGlobal,
        seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
    ):
        tidx = cute.arch.thread_idx()[0]
        row = cute.arch.block_idx()[0]
        topk_block = cute.arch.block_idx()[1]
        is_varlen = mCuSeqlensQ is not None

        topk = cute.size(mLocal.shape[1]) if const_expr(is_varlen) else cute.size(mLocal.shape[2])
        topk_idx = topk_block * self.threads_per_cta + tidx
        if topk_idx < topk:
            if const_expr(is_varlen):
                batch_idx = Int32(0)
                batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1
                lo = Int32(0)
                hi = Int32(batch_size)
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if row < mCuSeqlensQ[mid]:
                        hi = mid
                    else:
                        lo = mid
                batch_idx = lo
                offset = Int64(mCuSeqlensK[batch_idx])
                local = Int64(mLocal[row, topk_idx])
                out_coord = (row, topk_idx)
            else:
                seqlen_q = cute.size(mLocal.shape[1])
                batch_idx = row // seqlen_q
                q_idx = row - batch_idx * seqlen_q
                offset = Int64(batch_idx) * Int64(seqlen_k)
                local = Int64(mLocal[batch_idx, q_idx, topk_idx])
                out_coord = (batch_idx, q_idx, topk_idx)

            # Internal math uses Int64 to keep `batch_idx * seqlen_k` safe
            # against overflow on huge batched workloads. The final stored
            # value fits comfortably in int32 for any realistic shape (max
            # ~ B * max_seqlen_kv ≈ 10^7 ≪ 2^31), and every downstream
            # consumer (dsa-next sparse attn, indexer bwd) requires int32,
            # so we cast on store.
            global_idx = local if const_expr(self.preserve_negative) else Int64(-1)
            if local >= Int64(0):
                global_idx = offset + local
            mGlobal[out_coord] = Int32(global_idx)


_compile_cache: dict[tuple, object] = {}


def is_available() -> bool:
    # The kernel is arch-agnostic CuTe DSL: thread/block indexing, integer
    # ALU, and plain global mem load/store — no TMA / TMEM / tcgen05 / wgmma.
    # SM90+ is enough.
    return torch.cuda.is_available() and _get_device_capability() >= 9


def local_to_global(
    local_indices: torch.Tensor,
    seqlen_k: int,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    if not is_available():
        raise RuntimeError("CuTe DSL local_to_global requires an SM90+ CUDA device")
    if not local_indices.is_cuda or not local_indices.is_contiguous():
        raise ValueError("local_indices must be a contiguous CUDA tensor")
    if local_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("local_indices must be int32 or int64")

    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_varlen:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("THD local_to_global requires both cu_seqlens_q and cu_seqlens_k")
        if local_indices.ndim != 2:
            raise ValueError("THD local_indices must be 2D")
        for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
            if not t.is_cuda or t.dtype != torch.int32 or not t.is_contiguous() or t.ndim != 1:
                raise ValueError(f"{name} must be a contiguous 1D CUDA int32 tensor")
            if t.device != local_indices.device:
                raise ValueError(f"{name} must be on the same device as local_indices")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape")
    elif local_indices.ndim != 3:
        raise ValueError("BSHD local_indices must be 3D")

    launch_stream = resolve_stream(stream)
    if stream is None:
        global_indices = torch.empty_like(local_indices, dtype=torch.int32)
    else:
        with torch_stream_context(stream):
            global_indices = torch.empty_like(local_indices, dtype=torch.int32)
    return _launch_local_to_global(local_indices, global_indices, seqlen_k, cu_seqlens_q, cu_seqlens_k, launch_stream)


def _local_to_global_inplace(
    indices: torch.Tensor,
    seqlen_k: int,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    """Internal conversion of validated compressed top-k outputs.

    Dynamic layouts allow caller-owned strided buffers. Each thread updates
    its own slot, so aliasing the input and output is safe. Preserve every
    negative sentinel, matching the compressed wrapper's in-place contract.
    """
    return _launch_local_to_global(indices, indices, seqlen_k, cu_seqlens_q, cu_seqlens_k, resolve_stream(stream), preserve_negative=True)


def _launch_local_to_global(local_indices, global_indices, seqlen_k, cu_seqlens_q, cu_seqlens_k, stream, preserve_negative=False):
    is_varlen = cu_seqlens_q is not None
    compile_key = (
        local_indices.dtype,
        is_varlen,
        preserve_negative,
    )
    if compile_key not in _compile_cache:
        kernel_obj = LocalToGlobalTopK(is_varlen=is_varlen, preserve_negative=preserve_negative)
        _compile_cache[compile_key] = cute.compile(
            kernel_obj,
            _to_cute_tensor(local_indices),
            _to_cute_tensor(global_indices),
            cutlass.Int32(int(seqlen_k)),
            _to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None,
            _to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None,
            stream,
            options=compile_options("--opt-level 3"),
        )

    _compile_cache[compile_key](
        local_indices,
        global_indices,
        cutlass.Int32(int(seqlen_k)),
        cu_seqlens_q if is_varlen else None,
        cu_seqlens_k if is_varlen else None,
        stream,
    )
    return global_indices


__all__ = ["is_available", "local_to_global"]
