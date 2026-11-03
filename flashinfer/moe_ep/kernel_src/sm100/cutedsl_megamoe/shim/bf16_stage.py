# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Copy BF16 activations, routes and routing tails in one launch."""

from dataclasses import dataclass
from functools import cache
from threading import local
from typing import Any

import torch

from .comm import ensure_not_capturing


@cache
def _kernel_class():
    """Build the CuTeDSL stager lazily to keep CUDA imports out of discovery."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    class Bf16Stager:
        def __init__(self, hidden: int, topk: int):
            self.hidden = hidden
            self.topk = topk

        @cute.jit
        def __call__(
            self,
            activation: cute.Tensor,
            ids: cute.Tensor,
            scores: cute.Tensor,
            activation_out: cute.Tensor,
            ids_out: cute.Tensor,
            scores_out: cute.Tensor,
            stream: cuda.CUstream,
        ):
            """Launch enough CTAs for both live activations and the routing tail."""
            # Extra CTAs cover large tails even for an empty live batch.
            route_blocks = (cutlass.Int64(ids_out.shape[0]) * self.topk + 127) // 128
            blocks = cutlass.max(cutlass.Int64(activation.shape[0]), route_blocks)
            self.kernel(
                activation, ids, scores, activation_out, ids_out, scores_out
            ).launch(grid=(blocks, 1, 1), block=(128, 1, 1), stream=stream)

        @cute.jit
        def _aligned(self, tensor: cute.Tensor):
            """Expose the verified 16-byte alignment of an activation chunk."""
            ptr = tensor.iterator
            return cute.make_tensor(
                cute.make_ptr(ptr.dtype, ptr.toint(), ptr.memspace, assumed_align=16),
                tensor.layout,
            )

        @cute.kernel
        def kernel(
            self,
            activation: cute.Tensor,
            ids: cute.Tensor,
            scores: cute.Tensor,
            activation_out: cute.Tensor,
            ids_out: cute.Tensor,
            scores_out: cute.Tensor,
        ):
            """Copy independent activation vectors and the full routing plane."""
            tid, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            if row < activation.shape[0]:
                source = cute.zipped_divide(activation[row, None], (8,))
                target = cute.zipped_divide(activation_out[row, None], (8,))
                atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                    num_bits_per_copy=128,
                )
                values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                for chunk in cutlass.range_constexpr((self.hidden // 8 + 127) // 128):
                    index = tid + chunk * 128
                    if index < self.hidden // 8:
                        cute.copy(
                            atom, self._aligned(source[(None,), (index,)]), values
                        )
                        cute.copy(
                            atom, values, self._aligned(target[(None,), (index,)])
                        )

            index = cutlass.Int64(row) * 128 + tid
            if index < cutlass.Int64(ids_out.shape[0]) * self.topk:
                token, slot = index // self.topk, index % self.topk
                if token < activation.shape[0]:
                    ids_out[token, slot] = ids[token, slot].to(ids_out.element_type)
                    scores_out[token, slot] = scores[token, slot].to(
                        scores_out.element_type
                    )
                else:
                    # Every invocation masks the entire capacity tail. Host
                    # live-count memos cannot track other captured graph replays.
                    ids_out[token, slot] = ids_out.element_type(-1)

    return Bf16Stager


@dataclass(frozen=True)
class _PreparedStager:
    """One immutable binding owned by the calling host thread."""

    signature: tuple
    compiled: Any
    arguments: tuple


_STAGERS: dict[tuple, Any] = {}
_PREPARED = local()


def fused_bf16_stage(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_out: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> bool:
    """Stage supported CUDA layouts; return False for the Torch fallback.

    Preserve the original copy semantics, including masked live scores and
    untouched activation/score tails. Compilation must precede graph capture.
    """
    tensors = (
        hidden_states,
        topk_ids,
        topk_weights,
        x_out,
        topk_idx_out,
        topk_weights_out,
    )
    # Tensor storage and layouts may change in place. Snapshot all runtime
    # metadata once; an unchanged signature can reuse the prepared binding.
    signature = tuple(
        (t.data_ptr(), t.shape, t.stride(), t.dtype, t.device) for t in tensors
    )
    prepared = getattr(_PREPARED, "stager", None)
    if prepared is not None and prepared.signature == signature:
        import cuda.bindings.driver as cuda

        prepared.compiled(
            *prepared.arguments,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return True
    if not all(t.ndim == 2 for t in tensors):
        return False
    live, hidden = hidden_states.shape
    capacity = x_out.shape[0]
    if (
        live > capacity
        or x_out.shape[1] != hidden
        or topk_ids.shape[0] != live
        or topk_weights.shape != topk_ids.shape
        or topk_idx_out.shape != (capacity, topk_ids.shape[1])
        or topk_weights_out.shape != topk_idx_out.shape
        or topk_ids.shape[1] == 0
    ):
        return False
    if not all(
        t.is_cuda and t.device == hidden_states.device and t.is_contiguous()
        for t in tensors
    ):
        return False
    if (
        hidden_states.dtype != torch.bfloat16
        or x_out.dtype != torch.bfloat16
        or hidden_states.data_ptr() % 16
        or x_out.data_ptr() % 16
        or hidden_states.shape[1] % 8
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_idx_out.dtype not in (torch.int32, torch.int64)
        or topk_weights.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or topk_weights_out.dtype != torch.float32
    ):
        return False
    if x_out.shape[0] == 0:
        return True

    # The Torch copies run in sequence. Keep that path for overlapping planes
    # so this concurrently executing kernel cannot overwrite a later input.
    regions = [
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors
    ]
    for index in range(3, len(regions)):
        begin, end = regions[index]
        for other_begin, other_end in regions[:index]:
            if max(begin, other_begin) < min(end, other_end):
                return False

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    key = (
        hidden_states.shape[1],
        topk_ids.shape[1],
        tuple((item[4], item[3], item[2]) for item in signature),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _STAGERS.get(key)
    import cutlass.torch as cutlass_torch

    arguments = tuple(
        cutlass_torch.from_dlpack(
            t, assumed_align=16 if i in (0, 3) else 1
        ).mark_layout_dynamic(leading_dim=1)
        for i, t in enumerate(tensors)
    )
    if compiled is None:
        ensure_not_capturing("fused BF16 staging compilation")
        compiled = cute.compile(
            _kernel_class()(hidden_states.shape[1], topk_ids.shape[1]),
            *arguments,
            stream,
        )
        _STAGERS[key] = compiled
    # Publish a single immutable object and retain this call's local arguments.
    # Other host threads cannot replace its binding before the launch.
    _PREPARED.stager = _PreparedStager(signature, compiled, arguments)
    compiled(*arguments, stream)
    return True
