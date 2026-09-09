"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Exact-SM110 FP16 GQA decode backend.

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .jit import load_sm110_gqa_decode_module

_NUM_Q_HEADS = 32
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_HEADS_PER_GROUP = _NUM_Q_HEADS // _NUM_KV_HEADS
_SHORT_CAPACITY_MAX = 64


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def sm110_gqa_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Run the fixed Hq32/Hkv8/D128 FP16 decode specialization."""

    if q.ndim != 3:
        raise ValueError("q must have shape [batch, 32, 128]")
    batch = int(q.shape[0])
    if batch <= 0:
        raise ValueError("batch must be positive")
    _require_tensor(
        q,
        name="q",
        shape=(batch, _NUM_Q_HEADS, _HEAD_DIM),
        dtype=torch.float16,
    )
    if kv.ndim != 5:
        raise ValueError("kv must have shape [batch, 2, 8, capacity, 128]")
    capacity = int(kv.shape[-2])
    if capacity <= 0:
        raise ValueError("kv capacity must be positive")
    _require_tensor(
        kv,
        name="kv",
        shape=(batch, 2, _NUM_KV_HEADS, capacity, _HEAD_DIM),
        dtype=torch.float16,
        device=q.device,
    )
    _require_tensor(
        sequence_lengths,
        name="sequence_lengths",
        shape=(batch,),
        dtype=torch.int32,
        device=q.device,
    )

    if out is None:
        out = torch.empty_like(q)
    else:
        _require_tensor(
            out,
            name="out",
            shape=q.shape,
            dtype=torch.float16,
            device=q.device,
        )
        if out.data_ptr() == q.data_ptr():
            raise ValueError("out must not alias q")

    module = load_sm110_gqa_decode_module(device=q.device)
    q_grouped = q.view(
        batch,
        _NUM_KV_HEADS,
        _HEADS_PER_GROUP,
        _HEAD_DIM,
    ).transpose(1, 2)
    k = kv[:, 0]
    v = kv[:, 1]
    softmax_scale_log2 = float(q_scale) / math.sqrt(_HEAD_DIM) / math.log(2.0)
    launch = module.run_short if capacity <= _SHORT_CAPACITY_MAX else module.run_long
    with torch.cuda.device(q.device):
        launch(
            q_grouped,
            k,
            v,
            out,
            sequence_lengths,
            softmax_scale_log2,
            batch * _NUM_KV_HEADS,
            1,
            1,
        )
    return out


__all__ = ["sm110_gqa_decode"]
