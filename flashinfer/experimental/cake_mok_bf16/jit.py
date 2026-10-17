# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Loader of the standalone MoK CUDA libraries.

``sources.json`` maps every kernel role to one or more variants.  A variant
lists the exact targets it was generated for (``arches``), the argument plan of
its entry (``arguments``) and the architecture-free stem of its library
(``library``, ``cake_mok_<role>``).  ``build.json`` -- read by
:mod:`cake_build` -- names the per-architecture build ``<library>_<arch>``
(``cake_mok_<role>_<arch>``) and is the only record of what to compile
(translation units, digests, nvcc flags).  Kernels whose generated code
does not depend on the target share one variant, while kernels with
target-specific schedules (for example the MXFP8 tile width, which follows the
tensor-memory capacity) carry one variant per target group.  The variant is
selected by the device compute capability; its library is the prebuilt one
under ``aot/`` when the package ships it, else FlashInfer's JIT build of the
``build.json`` entry.
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
        self._arguments = tuple(name for _, name in arguments)

    def launch(self, *, grid, **arguments):
        arguments.update(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
        if set(arguments) != set(self._arguments):
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
    """The ``sources.json`` record of kernel ``role`` generated for a device of
    ``capability``: ``{"arches": [...], "arguments": [[kind, name], ...],
    "library": "cake_mok_<role>"}``."""
    arch = _target(capability)
    record = next(
        (v for v in _registry()[role]["variants"] if arch in v["arches"]), None
    )
    if record is None:
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


def specs_for_arch(arch):
    """JIT specs of every ``build.json`` entry of ``arch`` without a prebuilt library (the
    ``flashinfer.jit.build_jit_specs`` warm-up)."""
    return cake_build.jit_specs(arch)
