# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers shared by DSA Python wrappers."""

from functools import lru_cache
from typing import Optional

import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost._torch_stream import record_streams, stream_context
import cuda.bindings.driver as cuda


@lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def device_capability(device=None) -> tuple[int, int]:
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
    return _device_capability(index)


def device_major() -> int:
    return device_capability()[0]


def maybe_contiguous(
    x: torch.Tensor | None,
    stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor | None:
    if x is None or x.stride(-1) == 1:
        return x
    record_streams((x,), stream, x.device)  # R1 staging: the copy reads x asynchronously on `stream`
    with torch_stream_context(stream, x.device):
        return x.contiguous()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def validate_q_causal_offsets(
    q_causal_offsets: torch.Tensor | None,
    batch: int,
    device: torch.device,
    stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor | None:
    if q_causal_offsets is None:
        return None
    if q_causal_offsets.dtype != torch.int32:
        raise ValueError("q_causal_offsets must be int32")
    if q_causal_offsets.ndim != 1 or q_causal_offsets.shape[0] != batch:
        raise ValueError(f"q_causal_offsets must have shape ({batch},), got {tuple(q_causal_offsets.shape)}")
    if not q_causal_offsets.is_cuda:
        raise ValueError("q_causal_offsets must be a CUDA tensor")
    if q_causal_offsets.device != device:
        raise ValueError("q_causal_offsets must be on the same device as q")
    if q_causal_offsets.is_contiguous():
        return q_causal_offsets
    with torch_stream_context(stream, device):
        return q_causal_offsets.contiguous()


def resolve_stream(current_stream: Optional[cuda.CUstream] = None) -> cuda.CUstream:
    if current_stream is not None:
        return current_stream
    return cuda.CUstream(torch.cuda.current_stream(torch.cuda.current_device()).cuda_stream)


def torch_stream_context(current_stream: Optional[cuda.CUstream] = None, device=None):
    """Run torch work on ``current_stream`` (R1). ``device`` is the operand's
    device; it defaults to the current device, which is only right when the
    operands live there -- pass it whenever a tensor is at hand."""
    return stream_context(current_stream, device)
