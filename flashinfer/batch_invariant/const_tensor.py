"""Cached device constants for the batch-invariant APIs.

A kernel argument that is a constant filled from a Python scalar (a per-row filter value of the truncated samplers, the
global scale of the NVFP4 quantizer) was an allocation and a fill kernel on every call. The kernels only read such a tensor,
so one fill per (device, dtype, value, shape) serves every call with that value: :func:`const_fill` returns the cached
tensor and the call issues no kernel of its own for it.

Ordering: the fill runs on the creating call's stream and the entry carries its event; a call on another stream waits on
that event, device side, until the fill is known complete, after which the entry is used as it stands (no host
synchronisation; same-stream reuse is unchanged). Inside a CUDA-graph capture on the tensor's device the cache is not
consulted: the call fills its own tensor into the graph, as every call did before (an event recorded outside a capture
cannot be waited on inside it). The cache is bounded; past the bound a call fills a fresh tensor as before.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

__all__ = ["const_fill"]

#: (device, dtype, value, shape) -> [tensor, fill event or None once the fill is known complete]
_CACHE: Dict[tuple, list] = {}
_CACHE_MAX = 256


def _capturing(dev: torch.device) -> bool:
    if dev.index != torch.cuda.current_device():
        with torch.cuda.device(dev):
            return torch.cuda.is_current_stream_capturing()
    return torch.cuda.is_current_stream_capturing()


def const_fill(shape: Tuple[int, ...], value, dtype: torch.dtype, device) -> torch.Tensor:
    """The tensor ``torch.full(shape, value, dtype=dtype, device=device)``, cached per (device, dtype, value, shape) and
    ordered across streams (module docstring). The caller must only read it."""
    dev = torch.device(device)
    if dev.index is None:
        dev = torch.device(dev.type, torch.cuda.current_device())
    if _capturing(dev):
        return torch.full(shape, value, dtype=dtype, device=dev)
    key = (dev, dtype, value, shape)
    entry = _CACHE.get(key)
    if entry is None:
        t = torch.full(shape, value, dtype=dtype, device=dev)
        if len(_CACHE) < _CACHE_MAX:
            fill_done = torch.cuda.Event()
            fill_done.record(torch.cuda.current_stream(dev))
            _CACHE[key] = [t, fill_done]
        return t
    t, fill_done = entry
    if fill_done is not None:
        if fill_done.query():
            entry[1] = None  # the fill is complete: no ordering needed from here on
        else:
            torch.cuda.current_stream(dev).wait_event(fill_done)
    return t
