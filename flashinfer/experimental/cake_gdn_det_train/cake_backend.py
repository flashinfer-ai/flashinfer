"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Loader and launch glue of the Cake deterministic GDN chunked-training kernels
(SM100 / SM103): the device -> architecture map, the library loading through
the package's build manifest and the launch of one registry record.

The registry (:mod:`.cake_jit`) is data: ``MODULES`` holds one record per
program, each with the program's ``library`` stem (its ``build.json`` entries
are ``<library>_<arch>``), the architectures it is delivered for, the stage it
serves and the argument plan of its ``run`` entry.  What to compile -- sources,
digests, nvcc flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves a library from a prebuilt ``aot/`` file when a builder
compiled it and otherwise through FlashInfer's JIT from the same sources and
flags.  The prebuilt route imports nothing from ``flashinfer``: a process
without ``flashinfer.jit`` loads every library from ``aot/``
(``cake_build.load_library``).

Launches go through :func:`launch_stage` (the chain's positional call: the
kernel parameters in ``arg_plan`` order, then the grid) or :func:`run_module`
(the same call with the parameters named).  The kernel runs on the current
torch stream.  :func:`launch_stage` resolves a stage once per device -- its
record, the arity, the library and the FFI entry -- and launches from that
cached entry afterwards.  It takes PyTorch tensors, as the chain passes them,
and enters no stream context: a tvm-ffi call whose arguments include PyTorch
CUDA tensors sets the FFI stream of the first one's device to PyTorch's
current stream for the call (a CUDA graph's capture stream during capture),
and the binding launches there.  :func:`run_module` enters
``tvm_ffi.use_torch_stream()`` for any argument type.  Every launch is a
plain stream-ordered launch: nothing cooperative, no PDL.  Nothing here
reads an environment variable, synchronizes, reads device memory or
allocates.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import tvm_ffi

from . import cake_build, cake_jit

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "arch_for_capability",
    "device_arch",
    "is_prebuilt",
    "launch_stage",
    "library_name",
    "load",
    "parameters",
    "record",
    "run_module",
    "sm_count",
    "specs_for_arch",
    "stage_record",
]

# One verified build per architecture: B200 / GB200 (``sm_100a``) and B300 /
# GB300 (``sm_103a``).  Every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); no family fallback.
ARCHES: tuple[str, ...] = tuple(cake_jit.ARCHES)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = dict(cake_jit.ARCH_BY_CAPABILITY)


# ----------------------------------------------------------------------------------------------------------------
# Devices


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target); cached per device index."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake deterministic GDN training target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


# ----------------------------------------------------------------------------------------------------------------
# Records and libraries (``build.json``)


def record(name: str) -> dict[str, Any]:
    """The registry record ``name`` (a key of ``MODULES``)."""
    found = cake_jit.MODULES.get(name)
    if found is None:
        raise KeyError(
            f"{name!r} is not a record of the Cake deterministic GDN training registry"
        )
    return found


@functools.cache
def stage_record(stage: str, arch: str) -> str:
    """The record serving ``stage`` on ``arch`` (``KeyError`` when none does)."""
    return cake_jit.stage_record(stage, arch)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    found = record(name)
    if arch not in found["arches"]:
        raise RuntimeError(
            f"Cake deterministic GDN training record {name} is not delivered for {arch}; "
            f"its architectures: {found['arches']}"
        )
    return f"{found['library']}_{arch}"


def is_prebuilt(name: str, arch: str) -> bool:
    """Whether the library of record ``name`` for ``arch`` is served from ``aot/`` (a builder compiled it)."""
    return cake_build.prebuilt_path(library_name(name, arch)).is_file()


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the prebuilt ``aot/`` library when a
    builder compiled it, else FlashInfer's JIT build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library of ``build.json`` without a prebuilt file (for ahead-of-time
    warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


# ----------------------------------------------------------------------------------------------------------------
# Launching one record


@functools.cache
def parameters(name: str) -> tuple[str, ...]:
    """The kernel parameter names of record ``name`` in ``run`` order (the argument plan without the grid)."""
    found = record(name)
    names: list[str] = []
    axes: list[str] = []
    for kind, key in found["arg_plan"]:
        if kind == "grid":
            axes.append(key)
        else:
            if axes:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} follows the grid arguments"
                )
            names.append(key)
    if axes != ["grid_x", "grid_y", "grid_z"]:
        raise ValueError(
            f"record {name}: the argument plan must end with grid_x, grid_y, grid_z; got {axes}"
        )
    return tuple(names)


# (stage, device index) -> (FFI entry, kernel parameter count, record name), filled by the stage's first launch on
# the device; a device index keeps its architecture for the process (``device_arch``)
_ENTRIES: dict[tuple[str, int], tuple[Any, int, str]] = {}


def _arity_error(stage: str, name: str, count: int) -> ValueError:
    expected = parameters(name)
    return ValueError(
        f"stage {stage} ({name}) takes {len(expected)} arguments {expected}; got {count}"
    )


def _stage_entry(stage: str, device_index: int, count: int) -> tuple[Any, int, str]:
    """Resolve ``stage`` on the device once: the record serving it on the device's architecture, its parameter count
    (``count`` arguments must match it before the library loads), the loaded library and its FFI entry."""
    arch = device_arch(device_index)
    name = stage_record(stage, arch)
    arity = len(parameters(name))
    if count != arity:
        raise _arity_error(stage, name, count)
    entry = getattr(load(name, arch), record(name)["ffi_entry"])
    found = _ENTRIES[(stage, device_index)] = (entry, arity, name)
    return found


def launch_stage(
    stage: str,
    grid: tuple[int, int, int],
    values: Sequence[Any],
    *,
    device_index: int,
) -> None:
    """Launch the record serving ``stage`` on the device's architecture with ``values`` in ``arg_plan`` order (the
    kernel's parameter order) on ``grid``, on the current torch stream of the device.  The tensors of ``values`` are
    PyTorch tensors on the device: tvm-ffi passes that stream with them, so no stream context is entered here.  The
    first launch of a stage on a device resolves it (:func:`_stage_entry`, with its errors); later launches check the
    arity and call the cached FFI entry."""
    found = _ENTRIES.get((stage, device_index))
    if found is None:
        found = _stage_entry(stage, device_index, len(values))
    entry, arity, name = found
    if len(values) != arity:
        raise _arity_error(stage, name, len(values))
    entry(*values, int(grid[0]), int(grid[1]), int(grid[2]))


def run_module(
    name: str,
    values: Mapping[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
    module: Any | None = None,
) -> None:
    """Launch record ``name`` with its kernel parameters named in ``values`` on ``grid``.

    ``module`` overrides the loaded library; by default the record's library of the device's architecture runs.
    The kernel runs on the current torch stream.
    """
    found = record(name)
    args: list[Any] = []
    for key in parameters(name):
        try:
            args.append(values[key])
        except KeyError:
            raise ValueError(f"record {name} needs argument {key!r}") from None
    if module is None:
        module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, found["ffi_entry"])(
            *args, int(grid[0]), int(grid[1]), int(grid[2])
        )
