# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
"""Layout-keyed host caches for the public ``run`` path."""

from __future__ import annotations

import threading

import torch

RUN_LOCK = threading.RLock()

_CONST_LIMIT = 1024
_WORKSPACE_LIMIT = 16
_CONST: dict = {}
_WORKSPACES: dict = {}
_NUM_SMS: dict = {}
_GENERATION = 0


def device_index(device):
    device = torch.device(device)
    return torch.cuda.current_device() if device.index is None else device.index


def num_sms(device):
    index = device_index(device)
    count = _NUM_SMS.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _NUM_SMS[index] = count
    return count


def const_table(key, build, device, dtype=torch.int32):
    """Read-only device tensor memoized by a layout-only ``key``.

    :param key: Hashable layout key (never derived from tensor values).
    :param build: Zero-argument callable returning the host values.
    """
    full = (device_index(device), dtype, key)
    table = _CONST.get(full)
    if table is None:
        values = build()
        table = (
            values.to(device=device)
            if isinstance(values, torch.Tensor)
            else torch.tensor(values, dtype=dtype, device=device)
        )
        if len(_CONST) < _CONST_LIMIT:
            _CONST[full] = table
    return table


def stream_workspace(key, build, device):
    """Kernel-written workspace of ephemeral run-path launches, per current stream."""

    stream = torch.cuda.current_stream(device)
    full = (device_index(device), stream.stream_id, key)
    workspace = _WORKSPACES.get(full)
    if workspace is None:
        if len(_WORKSPACES) >= _WORKSPACE_LIMIT:
            _WORKSPACES.pop(next(iter(_WORKSPACES)))
            _bump_generation()
        workspace = build()
        _WORKSPACES[full] = workspace
    return workspace


def _bump_generation():
    global _GENERATION
    _GENERATION += 1


def workspace_generation():
    """Counter bumped whenever a run-path workspace is evicted or dropped."""
    return _GENERATION


def drop_stream_workspaces():
    """Forget every run-path workspace (after a failed launch sequence)."""

    _WORKSPACES.clear()
    _bump_generation()


__all__ = [
    "RUN_LOCK",
    "const_table",
    "device_index",
    "drop_stream_workspaces",
    "num_sms",
    "stream_workspace",
    "workspace_generation",
]
