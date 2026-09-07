"""Concrete paged-prefill backends and the factory the controller uses.

Each backend owns one complete implementation behind the same two calls:
``plan(meta, derived)`` and ``run(q, k_cache, v_cache, *, out, lse)``.
Nothing outside a backend knows its native dialect or LSE format.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

from ._capabilities import CAPABILITIES, MIN_DENSE_PAGE_SIZE, PagedAttentionCapabilities
from .cudnn_backend import _CudnnBackend
from .fa_backend import _FaBackend
from .trtllm_gen_backend import _TrtllmGenBackend

_FACTORIES: Dict[str, Callable] = {
    "fa2": lambda dev, layout, ws: _FaBackend(dev, layout, ws, "fa2"),
    "fa3": lambda dev, layout, ws: _FaBackend(dev, layout, ws, "fa3"),
    "cudnn": _CudnnBackend,
    "trtllm-gen": _TrtllmGenBackend,
}


def make_backend(
    name: str, device: torch.device, kv_layout: str, workspace: torch.Tensor
):
    """Construct the backend ``name`` (a key of ``CAPABILITIES``)."""
    return _FACTORIES[name](device, kv_layout, workspace)


__all__ = [
    "CAPABILITIES",
    "MIN_DENSE_PAGE_SIZE",
    "PagedAttentionCapabilities",
    "make_backend",
]
