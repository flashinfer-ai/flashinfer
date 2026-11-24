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

import struct
from typing import Dict, Tuple

import torch

__all__ = ["const_fill"]

#: (kind, device, dtype, value key, shape) -> [tensor, fill event or None once the fill is known complete]
_CACHE: Dict[tuple, list] = {}
#: entries per ``kind`` (one user of the cache, e.g. the NVFP4 global scale or the samplers' per-row filters): past the bound
#: a call of that kind fills a fresh tensor; another kind's entries are unaffected
_CACHE_MAX = 256
_COUNTS: Dict[str, int] = {}


def _value_key(value):
    """A Python float by its IEEE-754 bits -- ``+0.0`` and ``-0.0`` are different tensors (``0.0 == -0.0`` would share a
    dict key and the result would depend on which was filled first), and a NaN is one entry instead of a new one per call
    (``nan != nan`` never finds its own key); an int (or bool) by its value."""
    if isinstance(value, float):
        return ("f", struct.pack("<d", value))
    if isinstance(value, (bool, int)):
        return ("i", int(value))
    return ("o", value)


def _capturing(dev: torch.device) -> bool:
    if dev.index != torch.cuda.current_device():
        with torch.cuda.device(dev):
            return torch.cuda.is_current_stream_capturing()
    return torch.cuda.is_current_stream_capturing()


def const_fill(
    shape: Tuple[int, ...], value, dtype: torch.dtype, device, kind: str = "const"
) -> torch.Tensor:
    """The tensor ``torch.full(shape, value, dtype=dtype, device=device)``, cached per (kind, device, dtype, value, shape)
    and ordered across streams (module docstring); ``kind`` names the user, each with its own bound. The caller must only
    read it. Entries are never evicted: a tensor handed out may still be read by a kernel in flight on another stream,
    and the bound-and-fallback keeps every such tensor alive."""
    dev = torch.device(device)
    if dev.index is None:
        dev = torch.device(dev.type, torch.cuda.current_device())
    if _capturing(dev):
        return torch.full(shape, value, dtype=dtype, device=dev)
    key = (kind, dev, dtype, _value_key(value), shape)
    entry = _CACHE.get(key)
    if entry is None:
        t = torch.full(shape, value, dtype=dtype, device=dev)
        if _COUNTS.get(kind, 0) >= _CACHE_MAX:
            return t
        fill_done = torch.cuda.Event()
        fill_done.record(torch.cuda.current_stream(dev))
        # one publication per key: two threads missing the same key together both reach this point, and a plain store
        # would replace the first entry -- whose tensor a reader on another stream may already hold and which only the
        # cache keeps alive (the allocator would hand its block to the first thread's stream while that reader's kernel
        # runs). `setdefault` is atomic under the GIL: the loser takes the hit path on the winner's entry and drops its
        # own tensor, never handed out (its fill is ordered on its own stream, so the block's reuse there is safe).
        entry = _CACHE.setdefault(key, [t, fill_done])
        if entry[0] is t:
            _COUNTS[kind] = _COUNTS.get(kind, 0) + 1
            return t
    t, fill_done = entry
    if fill_done is not None:
        if fill_done.query():
            entry[1] = None  # the fill is complete: no ordering needed from here on
        else:
            torch.cuda.current_stream(dev).wait_event(fill_done)
    return t
