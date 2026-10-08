"""Checkpoint loading into ordinary CUDA tensors.

Importing this namespace does not initialize CUDA or build the native helper.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "capabilities": "._api",
    "DirectWeightSession": "._checkpoint",
    "SharedReadGroup": "._shared_checkpoint",
    "CheckpointDisplay": "._progress",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
