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
    "fa2": lambda dev, layout, ws, cap: _FaBackend(dev, layout, ws, "fa2", cap),
    "fa3": lambda dev, layout, ws, cap: _FaBackend(dev, layout, ws, "fa3", cap),
    "cudnn": lambda dev, layout, ws, cap: _CudnnBackend(dev, layout, ws),
    "trtllm-gen": lambda dev, layout, ws, cap: _TrtllmGenBackend(dev, layout, ws),
}


def make_backend(
    name: str,
    device: torch.device,
    kv_layout: str,
    workspace: torch.Tensor,
    *,
    graph_capacity=None,
):
    """Construct the backend ``name`` (a key of ``CAPABILITIES``).

    ``graph_capacity`` (a ``_graph.GraphCapacity``) is set in CUDA-graph mode so
    backends that keep their own metadata storage (the generated-FA wrapper)
    can reserve it up front.
    """
    return _FACTORIES[name](device, kv_layout, workspace, graph_capacity)


__all__ = [
    "CAPABILITIES",
    "MIN_DENSE_PAGE_SIZE",
    "PagedAttentionCapabilities",
    "make_backend",
]
