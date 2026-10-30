# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Loader of the standalone MoK CUDA libraries.

``sources.json`` maps every kernel role to one record: the targets it is
generated for (``arches``), the argument names of its entry (``arguments``)
and the architecture-free stem of its library (``library``,
``cake_mok_<role>``).  ``build.json`` -- read by :mod:`cake_build` -- names
the per-architecture build ``<library>_<arch>`` (``cake_mok_<role>_<arch>``)
and is the only record of what to compile (translation units, digests, nvcc
flags; kernels with target-specific schedules, for example the MXFP8 tile
width that follows the tensor-memory capacity, list a different kernel source
per target there).  The build is selected by the device compute capability;
its library is the prebuilt one under ``aot/`` when the package ships it, else
FlashInfer's JIT build of the ``build.json`` entry.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import torch

from . import cake_build

_PACKAGE = Path(__file__).parent
_TARGETS = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}


class _Kernel:
    def __init__(self, module, arguments):
        self._run = module.run
        self._arguments = tuple(arguments)
        self._argument_set = frozenset(self._arguments)

    def launch(self, *, grid, **arguments):
        arguments.update(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
        if arguments.keys() != self._argument_set:
            raise ValueError("Kernel arguments do not match the exported ABI")
        return self._run(*(arguments[name] for name in self._arguments))


def _target(capability):
    if capability not in _TARGETS:
        raise RuntimeError("MoK requires an SM100a, SM103a or SM107a device")
    return _TARGETS[capability]


def target_arch(device=None):
    """Exact generated target (``sm_100a``, ``sm_103a`` or ``sm_107a``) for a device."""
    return _target(torch.cuda.get_device_capability(device))


@functools.cache
def _registry():
    return json.loads((_PACKAGE / "sources.json").read_text())


def variant(role, capability):
    """The ``sources.json`` record of kernel ``role``, checked to be generated
    for a device of ``capability``: ``{"library": "cake_mok_<role>",
    "arches": [...], "arguments": [name, ...]}``."""
    arch = _target(capability)
    record = _registry()[role]
    if arch not in record["arches"]:
        raise RuntimeError(f"MoK kernel {role!r} has no generated {arch} variant")
    return record


def library_name(role, capability):
    """``build.json`` entry of kernel ``role`` on a device of ``capability``:
    ``<library>_<arch>``, for example ``cake_mok_forward_sm_100a``."""
    return f"{variant(role, capability)['library']}_{_target(capability)}"


@functools.cache
def _load(role, device):
    with torch.cuda.device(device):
        capability = torch.cuda.get_device_capability(device)
        module = cake_build.load_library(library_name(role, capability))
        return _Kernel(module, variant(role, capability)["arguments"])


def load_kernel(role):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare the MoK backend before CUDA Graph capture")
    return _load(role, torch.cuda.current_device())
