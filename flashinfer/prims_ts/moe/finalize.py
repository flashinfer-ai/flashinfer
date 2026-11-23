# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Overlap the native FC2 accumulation buffer's initialization with FC1."""

from contextlib import contextmanager
from threading import local

import cuda.bindings.driver as cuda
import torch

from flashinfer.cuda_utils import checkCudaErrors


_resources = local()


def _zero_stream(device):
    # One stream per host thread/device, independent of shapes and runner count.
    # No output tensors or captured graph buffers are retained in this cache.
    if not hasattr(_resources, "streams"):
        _resources.streams = {}
    if device not in _resources.streams:
        _resources.streams[device] = torch.cuda.Stream(device=device)
    return _resources.streams[device]


def _zero_output(output, stream):
    checkCudaErrors(
        cuda.cuMemsetD8Async(
            output.data_ptr(),
            0,
            output.numel() * output.element_size(),
            cuda.CUstream(stream.cuda_stream),
        )
    )


def _overlapping_bytes(left, right):
    if left.device != right.device or not left.numel() or not right.numel():
        return False

    def end(tensor):
        extent = 1 + sum(
            (n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True)
        )
        return tensor.data_ptr() + extent * tensor.element_size()

    return left.data_ptr() < end(right) and right.data_ptr() < end(left)


@contextmanager
def zero_output_during_fc1(output, *, read_input=None):
    """Fork zeroing alongside FC1 and join it before FC2, including graph capture.

    Each invocation owns its events. The output remains live until the auxiliary
    work rejoins the caller's stream, even if the FC1 launch raises. No tensor
    allocation or synchronization with the host is needed.
    If output overlaps FC1's input, enqueue zeroing after FC1 on the caller
    stream instead, so initialization cannot destroy unread activation bytes.
    """
    main = torch.cuda.current_stream(output.device)
    if read_input is not None and _overlapping_bytes(output, read_input):
        yield
        _zero_output(output, main)
        return
    auxiliary = _zero_stream(output.device)
    ready, done = torch.cuda.Event(), torch.cuda.Event()
    ready.record(main)
    try:
        yield
    finally:
        # Queue FC1 first so eager execution can overlap the memset too.
        # The auxiliary dependency is the event recorded *before* FC1.
        auxiliary.wait_event(ready)
        try:
            _zero_output(output, auxiliary)
        finally:
            done.record(auxiliary)
            main.wait_event(done)
