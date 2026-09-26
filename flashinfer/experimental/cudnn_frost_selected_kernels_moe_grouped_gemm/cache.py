# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Bound preparation lookups independently of packed-call and graph ownership."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Any

import torch


class LRUCache(OrderedDict):
    """Keep recently used preparation records; owners may outlive eviction."""

    def __init__(self, maxsize: int = 32):
        super().__init__()
        if maxsize < 1:
            raise ValueError("cache capacity must be positive")
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        """Read an entry and refresh its position in the eviction order."""
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


class TensorCache:
    """Cache by tensor identity/version without retaining the source tensor.

    Values are bounded even when a packed view aliases its source. A live
    packed call or CUDA Graph, rather than this lookup, owns evicted resources.
    """

    def __init__(self, maxsize: int = 4):
        self._entries = LRUCache(maxsize)

    def __len__(self):
        return len(self._entries)

    def get(self, tensor: torch.Tensor, version: int | None, *, context=None) -> Any:
        """Return the value only for this tensor, version and numerical contract."""
        entry = self._entries.get((id(tensor), context))
        if entry is not None and entry[0]() is tensor and entry[1] == version:
            return entry[2]
        return None

    def put(
        self, tensor: torch.Tensor, version: int | None, value: Any, *, context=None
    ) -> None:
        """Remember preparation and drop it when an unaliased source dies."""
        key, owner = (id(tensor), context), weakref.ref(self)

        def discard(reference):
            cache = owner()
            if cache is not None:
                entry = cache._entries.get(key)
                if entry is not None and entry[0] is reference:
                    del cache._entries[key]

        self._entries[key] = (weakref.ref(tensor, discard), version, value)

    def clear(self) -> None:
        """Drop lookup ownership; active calls and graphs remain valid."""
        self._entries.clear()


def require_graph_resource_retention() -> None:
    """Require graph ownership before admitting a runner with evictable caches."""
    graph_type = torch.cuda.CUDAGraph
    for name in ("get_currently_capturing_graph", "retain_object"):
        if not callable(getattr(graph_type, name, None)):
            raise NotImplementedError(
                "cuDNN Frost MoE requires PyTorch CUDA Graph resource retention "
                f"(CUDAGraph.{name})"
            )


def retain_graph_resources(state, inputs) -> None:
    """Keep captured plans and tensors alive until their graph releases them.

    Retaining Python objects adds no GPU work. Synchronization happens only
    when the graph releases resources, after any outstanding replay finishes.
    Resources must not reference the graph, which would create a lifetime cycle.
    """
    with torch.cuda.device(state.workspace.device):
        if not torch.cuda.is_current_stream_capturing():
            return
        graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
        if graph is None:
            raise RuntimeError("Capture cuDNN Frost MoE with torch.cuda.CUDAGraph")
        graph.retain_object((state, *inputs), synchronize_before_release=True)
