"""Checkpoint loading into ordinary CUDA tensors.

Importing this namespace does not initialize CUDA or build the native helper.
"""

from __future__ import annotations

__all__ = ["capabilities"]


def __getattr__(name):
    if name in __all__:
        from . import _api

        value = getattr(_api, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
