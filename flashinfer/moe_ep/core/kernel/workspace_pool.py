"""Process-level pool of mega-MoE symm-buffer workspaces, shared across layers.

A model stacks many MoE layers with identical EP geometry; without sharing,
each ``MoEEpMegaLayer`` allocates its own symmetric-heap workspace (~GBs of
NVSHMEM heap and a separate compiled-kernel session per layer — 43x at
DeepSeek-scale). The workspace is stateless across forwards (staging
overwrites the inputs and the kernel tail-cleans its counters) and layers
execute sequentially on one stream, so all layers with the same pool key can
share one buffer — this also shares the frontend's compiled kernel.

Backends opt in by returning a hashable key from ``_workspace_pool_key()``
(``None`` keeps the old per-layer allocation, e.g. for ``knobs="auto"``
sessions whose autotune mutates the shared frontend). Entries are
refcounted: ``MegaKernelBackend.destroy`` releases and only the last release
actually frees the symmetric heap.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Tuple


@dataclasses.dataclass
class _PoolEntry:
    workspace: Any
    refcount: int


_POOL: Dict[Tuple, _PoolEntry] = {}
_KEY_BY_ID: Dict[int, Tuple] = {}


def acquire_workspace(key: Tuple, factory: Callable[[], Any]) -> Any:
    """Return the pooled workspace for ``key``, allocating on first use."""
    entry = _POOL.get(key)
    if entry is None:
        entry = _PoolEntry(workspace=factory(), refcount=0)
        _POOL[key] = entry
        _KEY_BY_ID[id(entry.workspace)] = key
    entry.refcount += 1
    return entry.workspace


def release_workspace(workspace: Any) -> bool:
    """Drop one reference; ``True`` when the caller should actually destroy.

    Unpooled workspaces (never acquired here) always return ``True`` so the
    caller's destroy path is unchanged for them.
    """
    key = _KEY_BY_ID.get(id(workspace))
    if key is None:
        return True
    entry = _POOL.get(key)
    if entry is None:
        return True
    entry.refcount -= 1
    if entry.refcount <= 0:
        del _POOL[key]
        del _KEY_BY_ID[id(workspace)]
        return True
    return False


def pooled_workspace_count() -> int:
    """Number of live pooled workspaces (introspection / tests)."""
    return len(_POOL)


def epilogue_pool_key(value: Any) -> Any:
    """Hashable identity for a per-expert epilogue scalar config value.

    Scalars share by value. Tensors share only by object identity — two
    layers holding different tensors (even equal ones) get separate buffers,
    because the values are baked into the buffer at creation and a shared
    buffer would silently apply one layer's scalars to the other. Per-forward
    scalars staged via ``MoEEpTensors`` are unaffected (re-copied before
    every compute).
    """
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return ("tensor", id(value))
    return ("scalar", float(value))


def knobs_pool_key(knobs: Any) -> Any:
    """Canonical hashable form of a knobs dict (order-insensitive)."""
    if knobs is None or isinstance(knobs, str):
        return knobs
    return tuple(
        sorted(
            (k, tuple(v) if isinstance(v, (list, tuple)) else v)
            for k, v in knobs.items()
        )
    )
