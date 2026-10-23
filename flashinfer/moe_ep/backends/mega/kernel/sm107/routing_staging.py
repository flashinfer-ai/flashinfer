# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Copy validated routes into the MegaMoE workspace without mask temporaries."""

from dataclasses import dataclass
from functools import cache
import os

import torch


@cache
def _kernel_class():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    class RouteStager:
        def __init__(self, topk: int):
            self.topk = topk

        @cute.jit
        def __call__(
            self,
            ids: cute.Tensor,
            scores: cute.Tensor,
            ids_out: cute.Tensor,
            scores_out: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.kernel(ids, scores, ids_out, scores_out).launch(
                grid=((ids.shape[0] * self.topk + 255) // 256, 1, 1),
                block=(256, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            ids: cute.Tensor,
            scores: cute.Tensor,
            ids_out: cute.Tensor,
            scores_out: cute.Tensor,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            index = block * 256 + tid
            if index < ids.shape[0] * self.topk:
                row, col = index // self.topk, index % self.topk
                expert = cutlass.Int32(ids[row, col])
                score = cutlass.Float32(0.0)
                if expert != -1:
                    score = cutlass.Float32(scores[row, col])
                ids_out[row, col] = expert
                scores_out[row, col] = score

    return RouteStager


@dataclass
class _Entry:
    key: tuple
    inputs: tuple
    arguments: tuple
    compiled: object


_COMPILED: dict[tuple, object] = {}


def stage_routes(ids, scores, ids_out, scores_out, *, workspace) -> None:
    """Stage live rows; the caller validates routes and clears the capacity tail."""
    if (
        not ids.is_cuda
        or os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") == "0"
        or scores.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        live = ids.shape[0]
        ids_out[:live].copy_(ids.to(torch.int32))
        scores_out[:live].copy_(scores.float().masked_fill(ids == -1, 0))
        return
    if ids.numel() == 0:
        return

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch

    tensors = (ids, scores, ids_out, scores_out)
    ffi = os.environ.get("FLASHINFER_SM107_PREP_TVM_FFI", "1") != "0"
    key = (
        ids.shape[1],
        tuple((t.device, t.dtype, tuple(t.stride())) for t in tensors),
        ffi,
    )
    entry = getattr(workspace, "_route_staging", None)
    if (
        entry is None
        or entry.key != key
        or any(a is not b for a, b in zip(entry.inputs, tensors, strict=True))
    ):
        compiled = _COMPILED.get(key)
        arguments = tensors
        if compiled is None or not ffi:
            arguments = tuple(
                cutlass_torch.from_dlpack(t, assumed_align=1).mark_layout_dynamic(
                    leading_dim=1 if t.stride(1) == 1 else None
                )
                for t in tensors
            )
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm route staging before CUDA graph capture")
            compiled = cute.compile(
                _kernel_class()(ids.shape[1]),
                *arguments,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                options="--enable-tvm-ffi" if ffi else "",
            )
            _COMPILED[key] = compiled
        entry = _Entry(key, tensors, tensors if ffi else arguments, compiled)
        workspace._route_staging = entry
    entry.compiled(
        *entry.arguments, cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )
