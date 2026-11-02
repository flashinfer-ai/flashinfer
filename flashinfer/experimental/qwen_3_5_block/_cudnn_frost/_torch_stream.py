# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A raw CUDA stream handle as the torch stream torch work must run on (Hard Rule 5).

Every torch-facing engine receives the launch stream as a driver handle (the
execute-time handle's stream). Turning it into a torch stream has one trap: the
legacy default-stream sentinels ``0`` (cudaStream_t null) and ``1``
(cudaStreamLegacy) map to ``torch.cuda.default_stream``, never to
``torch.cuda.ExternalStream`` -- torch before pytorch/pytorch#183258 (v2.13.0)
returns a fresh NON-BLOCKING pool stream for ``ExternalStream(0)``, so torch work
issued in it never orders against a kernel launched on ``CUstream(0)``. Thirteen
call sites carried that guard as local copies and the fourteenth missed it
(cudnn-frontend #1165); this module is the one copy.

``2`` (cudaStreamPerThread) identifies a different stream in each host thread.
It cannot be mapped to torch's legacy default stream or safely used as a shared
allocator stream identity. Torch interop rejects it; use a concrete torch stream.

torch is imported lazily: ``python/cudnn`` must import without it.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, Optional

# cudaStream_t 0 and cudaStreamLegacy. The per-thread sentinel 2 is NOT legacy.
DEFAULT_STREAM_HANDLES = frozenset({0, 1})

_RAW_CURRENT: Any = None  # torch._C._cuda_getCurrentRawStream, resolved once


def _device_index(torch, device) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return device.index if device.index is not None else torch.cuda.current_device()


def _raw_current_stream(torch, device) -> Optional[int]:
    """torch's current stream handle on ``device`` via the private raw getter
    (~0.07 us) -- ``torch.cuda.current_stream()`` builds a Stream object (~3.4 us)."""
    global _RAW_CURRENT
    if _RAW_CURRENT is None:
        _RAW_CURRENT = getattr(torch._C, "_cuda_getCurrentRawStream", False)
    if not _RAW_CURRENT:
        return None
    return int(_RAW_CURRENT(_device_index(torch, device)))


def _reject_per_thread_stream(handle: int) -> None:
    if handle == 2:
        raise ValueError(
            "cudaStreamPerThread (CUstream(2)) cannot be represented safely by torch stream interop; "
            "pass a concrete torch.cuda.Stream or its cuda_stream handle instead"
        )


def as_torch_stream(stream, device=None):
    """The ``torch.cuda.Stream`` for ``stream``: a driver ``CUstream``, an int handle,
    or already a torch stream.

    Legacy sentinels and torch's own default stream -> ``torch.cuda.default_stream(device)``;
    torch's current stream -> itself; only a genuine side stream is wrapped in
    ``torch.cuda.ExternalStream(handle, device=device)``.
    """
    import torch

    if isinstance(stream, torch.cuda.Stream):
        if device is not None and stream.device != torch.device("cuda", _device_index(torch, device)):
            raise ValueError(f"stream must be on cuda:{_device_index(torch, device)}, got {stream.device}")
        _reject_per_thread_stream(stream.cuda_stream)
        return stream
    handle = int(stream)
    _reject_per_thread_stream(handle)
    default = torch.cuda.default_stream(device)
    if handle in DEFAULT_STREAM_HANDLES or handle == default.cuda_stream:
        return default
    current = torch.cuda.current_stream(device)
    if handle == current.cuda_stream:
        return current
    return torch.cuda.ExternalStream(handle, device=device)


@contextmanager
def stream_context(stream, device=None, *, verify_current: bool = False) -> Iterator[None]:
    """``torch.cuda.stream(as_torch_stream(stream, device))``; a no-op when ``stream``
    is None or already torch's current stream on ``device``.

    ``verify_current=False`` (default) takes the raw-handle fast path for the
    common case where the launch stream is the one torch is already on.
    """
    if stream is None:
        yield
        return
    import torch

    handle = stream.cuda_stream if isinstance(stream, torch.cuda.Stream) else int(stream)
    _reject_per_thread_stream(handle)
    if not verify_current:
        raw = _raw_current_stream(torch, device)
        # A default-stream sentinel maps to torch's default stream (see as_torch_stream), so it
        # is also a no-op while torch is on its default stream.
        if raw is not None and (handle == raw or (handle in DEFAULT_STREAM_HANDLES and raw in DEFAULT_STREAM_HANDLES)):
            yield
            return
    with torch.cuda.stream(as_torch_stream(stream, device)):
        yield


def launch_stream(stream, device=None):
    """The torch stream a launch on ``stream`` runs on: ``as_torch_stream(stream, device)``,
    or torch's current stream on ``device`` when ``stream`` is None."""
    import torch

    index = _device_index(torch, device)
    if stream is None:
        stream = _raw_current_stream(torch, index)
        if stream is None:
            return torch.cuda.current_stream(index)
    return as_torch_stream(stream, index)


def device_context(device):
    """``torch.cuda.device(device)``, or a no-op when ``device`` is already current."""
    import torch

    index = device if isinstance(device, int) else device.index
    if index is None or index == torch.cuda.current_device():
        return nullcontext()
    return torch.cuda.device(index)


def record_streams(tensors, stream, device=None) -> None:
    """Order the caching allocator's reuse of each tensor's block behind the work already
    enqueued on ``stream`` (R1): ``tensor.record_stream(as_torch_stream(stream, device))``
    for every CUDA tensor in ``tensors`` (``None`` entries skipped).

    ``None`` means torch's current stream on the tensor's device. That stream must
    still be recorded: the tensor may have been allocated on another stream before
    the caller entered a side-stream context.
    """
    consumer = None
    for tensor in tensors:
        if tensor is None or not tensor.is_cuda:
            continue
        if consumer is None:
            consumer = launch_stream(stream, device if device is not None else tensor.device)
        tensor.record_stream(consumer)


def contiguous_on_stream(tensor, stream, device=None):
    """``tensor`` itself when it is None or contiguous; otherwise a contiguous copy made on
    ``stream`` after ``record_streams((tensor,), stream, device)``.

    The recording is what makes the copy safe to rebind over: a wrapper doing
    ``t = t.contiguous()`` under ``stream_context(side)`` drops the caller's reference while
    the copy kernel is still queued on ``side``; if the caller also releases ``t``, its block
    returns to the allocator's pool for the caller's stream and the next same-size allocation
    there overwrites what the copy has yet to read.
    """
    if tensor is None or tensor.is_contiguous():
        return tensor
    record_streams((tensor,), stream, device)
    with stream_context(stream, device):
        return tensor.contiguous()


def copy_into_on_stream(dst, src, stream, device=None) -> None:
    """``dst.copy_(src)`` on ``stream`` with both tensors recorded there first: the wrapper copy-back
    into a caller-owned buffer is asynchronous too, so neither a ``dst`` the caller releases right
    after the call nor a ``src`` allocated on another stream may be reused under the pending copy."""
    record_streams((src, dst), stream, device)
    with stream_context(stream, device):
        dst.copy_(src)
