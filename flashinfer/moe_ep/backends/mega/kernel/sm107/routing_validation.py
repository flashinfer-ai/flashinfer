# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""One-pass routing checks for Rubin batches."""

from __future__ import annotations

from dataclasses import dataclass
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils
import torch


class _RoutingCheck:
    def __init__(self, topk: int):
        self.topk = topk

    @cute.jit
    def __call__(
        self,
        ids: cute.Tensor,
        weights: cute.Tensor,
        result: cute.Tensor,
        experts: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(ids, weights, result, experts).launch(
            grid=(result.shape[0], 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        ids: cute.Tensor,
        weights: cute.Tensor,
        result: cute.Tensor,
        experts: cutlass.Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        valid = cutlass.Boolean(True)
        for index in range(
            tid + block * 256, ids.shape[0] * self.topk, result.shape[0] * 256
        ):
            row = index // self.topk
            col = index % self.topk
            expert = cutlass.Int64(ids[row, col])
            score = cutlass.Float32(weights[row, col])
            valid = valid & (expert >= -1) & (expert < experts)
            finite = (score == score) & (score < float("inf")) & (score > -float("inf"))
            valid = valid & ((expert == -1) | finite)
            for other in cutlass.range_constexpr(self.topk):
                if other < col:
                    duplicate = (expert >= 0) & (
                        expert == cutlass.Int64(ids[row, other])
                    )
                    valid = valid & ~duplicate
        warp_valid = cute.arch.vote_all_sync(valid)
        smem = cutlass.utils.SmemAllocator()
        votes = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((8,)), byte_alignment=16
        )
        if tid % 32 == 0:
            votes[tid // 32] = cutlass.Int32(warp_valid)
        cute.arch.sync_threads()
        if tid < 32:
            block_valid = cutlass.Boolean(True)
            if tid < 8:
                block_valid = votes[tid] != 0
            block_valid = cute.arch.vote_all_sync(block_valid)
            if tid == 0:
                result[block] = block_valid


@dataclass
class _Entry:
    compiled: object
    result: torch.Tensor
    inputs: tuple
    arguments: tuple


_COMPILED: dict[tuple, object] = {}


def _dynamic_argument(tensor: torch.Tensor):
    return cutlass_torch.from_dlpack(tensor, assumed_align=1).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


def check_routing(
    ids: torch.Tensor, weights: torch.Tensor, experts: int, workspace=None
) -> None:
    """Keep the existing asynchronous device assertion, including graph replay."""
    stream = torch.cuda.current_stream().cuda_stream
    ffi = os.environ.get("FLASHINFER_SM107_PREP_TVM_FFI", "1") != "0"
    compile_key = (
        ids.device,
        ids.dtype,
        weights.dtype,
        ids.shape[1],
        tuple(ids.stride()),
        tuple(weights.stride()),
        ffi,
    )
    # Reuse the kernel across row counts; keep each graph's buffers separate.
    key = (*compile_key, tuple(ids.shape), stream)
    if workspace is not None and not hasattr(workspace, "_routing_checks"):
        workspace._routing_checks = {}
    entries = workspace._routing_checks if workspace is not None else {}
    entry = entries.get(key)
    if entry is None:
        entries_count = ids.numel()
        blocks = (entries_count + 255) // 256 if entries_count > 1024 else 1
        result = torch.empty(blocks, dtype=torch.bool, device=ids.device)
        compile_arguments = tuple(_dynamic_argument(t) for t in (ids, weights, result))
        arguments = (ids, weights, result) if ffi else compile_arguments
        compiled = _COMPILED.get(compile_key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm routing validation before CUDA graph capture")
            compiled = cute.compile(
                _RoutingCheck(ids.shape[1]),
                *compile_arguments,
                cutlass.Int64(experts),
                cuda.CUstream(stream),
                options="--enable-tvm-ffi" if ffi else "",
            )
            _COMPILED[compile_key] = compiled
        entry = _Entry(compiled, result, (ids, weights), arguments)
        # The workspace owns this assertion buffer for every captured profile.
        # Releasing it while a graph can replay would invalidate the graph.
        entries[key] = entry
    elif ids is not entry.inputs[0] or weights is not entry.inputs[1]:
        entry.inputs = (ids, weights)
        entry.arguments = (
            (ids, weights, entry.result)
            if ffi
            else tuple(_dynamic_argument(t) for t in (ids, weights, entry.result))
        )
    entry.compiled(
        *entry.arguments,
        experts if ffi else cutlass.Int64(experts),
        cuda.CUstream(stream),
    )
    torch._assert_async(
        entry.result.all() if entry.result.numel() > 1 else entry.result,
        "SM107 routes require unique expert IDs in [0, E), or -1, and finite scores for valid routes",
    )
