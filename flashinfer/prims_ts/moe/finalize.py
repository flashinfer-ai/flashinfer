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


@contextmanager
def zero_output_during_fc1(output):
    """Fork zeroing alongside FC1 and join it before FC2, including graph capture.

    Each invocation owns its events. The output remains live until the auxiliary
    work rejoins the caller's stream, even if the FC1 launch raises. No tensor
    allocation or synchronization with the host is needed.
    """
    main = torch.cuda.current_stream(output.device)
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
