# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Single-launch BF16/routing staging, with the caller's torch fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..bf16_bf16_bf16_cutedsl.staging import (
    stage_mega_moe_inputs as _torch_stage_mega_moe_inputs,
)


@dataclass
class _CompiledStager:
    compiled: Any = None
    launch_key: tuple | None = None
    launch_args: tuple | None = None


_STAGERS: dict[tuple, _CompiledStager] = {}


def _supported(tensors: tuple[torch.Tensor, ...]) -> bool:
    hidden, ids = tensors[:2]
    # The backend already validates dtypes, dimensions, common device and
    # capacity. Gate only requirements introduced by this vector-copy path.
    if hidden.shape[0] == 0 or hidden.shape[1] % 8 or ids.shape[1] > 128:
        return False
    # Contiguous singleton columns may still have a non-unit inner stride.
    # Those legal torch views cannot use our explicit leading_dim=1 wrapper.
    if any(
        not t.is_cuda or not t.is_contiguous() or t.stride(1) != 1 or t.requires_grad
        for t in tensors
    ):
        return False
    # Row starts and each eight-BF16 chunk must be 16-byte aligned. Scalar
    # routing loads/stores need only their element alignment.
    alignments = (16, ids.element_size(), 4, 16, 8, 4)
    if any(t.data_ptr() % a for t, a in zip(tensors, alignments, strict=False)):
        return False
    # Preserve torch.copy_ semantics for aliased views, including overlap
    # errors. Sharing a storage is uncommon here and safely takes the fallback.
    output_storage = {t.untyped_storage().data_ptr() for t in tensors[3:]}
    return not any(
        t.untyped_storage().data_ptr() in output_storage for t in tensors[:3]
    )


def _to_cute(tensor: torch.Tensor, alignment: int):
    # All fast-path views are contiguous 2D. Fix only inner stride=1; both
    # shape modes, including n and capacity at size1, remain dynamic.
    import cutlass.torch as cutlass_torch

    return cutlass_torch.from_dlpack(
        tensor, assumed_align=alignment
    ).mark_layout_dynamic(leading_dim=1)


def _try_fused_bf16_stage(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> bool:
    """Stage eligible views and return True; False requests original torch copies.

    The fused kernel always masks the entire capacity tail, without a host
    live-count memo. Empty batches use the original single tail-fill operation.
    Strided, misaligned and aliased views keep their existing torch behavior.
    A new specialization first seen during capture also uses torch staging.
    """
    tensors = (hidden_states, topk_ids, topk_weights, x, topk_idx_out, topk_weights_out)
    if not _supported(tensors):
        return False
    from ......kernel_src.cutedsl_megamoe import ensure_not_capturing

    with torch.cuda.device(hidden_states.device):
        _, hidden = hidden_states.shape
        topk = topk_ids.shape[1]
        key = (hidden_states.device.index, hidden, topk, topk_ids.dtype)
        stager = _STAGERS.get(key)
        if stager is None:
            # Default MegaLayer warmup uses int64 IDs; real int32 routing can
            # first appear during capture. Preserve the original torch path
            # for that graph rather than compiling or requiring extra warmup.
            if torch.cuda.is_current_stream_capturing():
                return False
            ensure_not_capturing("BF16 staging construction")
            stager = _CompiledStager()

        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        launch_key = (
            tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride())) for t in tensors),
            stream,
        )
        if stager.launch_key != launch_key:
            import cuda.bindings.driver as cuda_driver

            alignments = (16, topk_ids.element_size(), 4, 16, 8, 4)
            args = tuple(
                _to_cute(t, a) for t, a in zip(tensors, alignments, strict=False)
            ) + (cuda_driver.CUstream(stream),)
            if stager.compiled is None:
                ensure_not_capturing("BF16 staging cute.compile")
                import cutlass.cute as cute

                from .staging_kernel import Bf16InputStage

                stager.compiled = cute.compile(Bf16InputStage(hidden, topk), *args)
                _STAGERS[key] = stager
            stager.launch_key = launch_key
            stager.launch_args = args
        stager.compiled(*stager.launch_args)
        return True


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    args = (hidden_states, topk_weights, topk_ids, x, topk_idx_out, topk_weights_out)
    if not _try_fused_bf16_stage(*args):
        _torch_stage_mega_moe_inputs(*args)
