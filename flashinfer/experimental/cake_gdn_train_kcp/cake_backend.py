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

Loader and launch glue of the Cake packed GDN KCP training kernels
(SM100 / SM103 / SM107): the device -> architecture map, the library loading
through the package's build manifest and the generic launch of one registry
record with its logical arguments.

The registry (:mod:`.cake_jit`) is data: ``MODULES`` holds the shipped
records, ``TEST_MODULES`` the test-only ones, each with the program's
``library`` stem (its ``build.json`` entries are ``<library>_<arch>``), the
architectures it is delivered for, the ``select_rule`` / ``grid_rule`` the host
evaluates and the kernel-parameter -> logical-argument map of its ``run``
entry.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves a library from a
prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags.  The prebuilt route imports
nothing from ``flashinfer``: a process without ``flashinfer.jit`` loads every
library from ``aot/`` (``cake_build.load_library``).

Test-only programs (``build.test.json``, the same ``cake-build/1`` form) are
reachable through :func:`load_test` alone.  Nothing here reads an environment
variable: a test selects a test-only record through the keyword-only test
knobs of the public call (``OptionsKcp(test_knobs=...)``), which the
orchestration module validates and maps onto a ``TEST_MODULES`` record.

Launches go through :func:`run_module`: the record maps its kernel
parameters onto the package's logical vocabulary (``cake_jit.LOGICAL_TENSORS``
/ ``LOGICAL_SCALARS``), the grid comes from the host plan
(``cake_jit.launch_grid``) and the kernel runs on the current torch stream.
Every launch is a plain stream-ordered launch: nothing cooperative, no PDL.
Nothing here synchronizes, reads device memory or allocates.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tvm_ffi

from . import cake_build, cake_jit

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "TEST_MANIFEST",
    "arch_for_capability",
    "device_arch",
    "is_prebuilt",
    "library_name",
    "load",
    "load_test",
    "record",
    "run_module",
    "sm_count",
    "specs_for_arch",
    "test_jit_spec",
    "test_library_name",
    "test_manifest",
    "test_specs_for_arch",
]

# One verified build per architecture: B200 / GB200 (``sm_100a``), B300 / GB300
# (``sm_103a``) and R200 (``sm_107a``).  Every architecture is compiled with its
# exact ``-gencode`` flag set (``build.json`` ``arch_flags``); no family fallback.
ARCHES: tuple[str, ...] = tuple(cake_jit.ARCHES)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
TEST_MANIFEST = "build.test.json"
_PACKAGE = Path(__file__).resolve().parent
_TEST_MANIFEST_PATH = _PACKAGE / TEST_MANIFEST
_MANIFEST_SCHEMA = "cake-build/1"


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
            f"Cake GDN KCP training target; the exported architectures are {ARCHES}"
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
    """The registry record ``name`` (a key of ``MODULES`` or ``TEST_MODULES``)."""
    found = cake_jit.MODULES.get(name)
    if found is None:
        found = cake_jit.TEST_MODULES.get(name)
    if found is None:
        raise KeyError(f"{name!r} is not a record of the Cake GDN KCP registry")
    return found


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of shipped record ``name`` built for ``arch``: ``<library>_<arch>``."""
    found = cake_jit.MODULES[name]
    if arch not in found["arches"]:
        raise RuntimeError(
            f"Cake GDN KCP record {name} is not delivered for {arch}; its architectures: {found['arches']}"
        )
    return f"{found['library']}_{arch}"


def is_prebuilt(name: str, arch: str) -> bool:
    """Whether the library of shipped record ``name`` for ``arch`` is served from ``aot/`` (a builder compiled it)."""
    return cake_build.prebuilt_path(library_name(name, arch)).is_file()


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of shipped record ``name`` built for ``arch``: the prebuilt ``aot/`` library when a
    builder compiled it, else FlashInfer's JIT build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library of ``build.json`` without a prebuilt file (for ahead-of-time
    warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


# ----------------------------------------------------------------------------------------------------------------
# The test-only record set (``build.test.json``)


@functools.cache
def test_manifest() -> dict[str, Any]:
    """The parsed ``build.test.json`` (schema ``cake-build/1``, the test-only libraries), read on first use.  The
    shipped route never opens it: only :func:`load_test` (reached through the validated test knobs) reads it."""
    try:
        text = _TEST_MANIFEST_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{_TEST_MANIFEST_PATH} is missing: the test-only record set was not exported"
        ) from exc
    document = json.loads(text)
    if document.get("schema") != _MANIFEST_SCHEMA:
        raise RuntimeError(
            f"{_TEST_MANIFEST_PATH} has schema {document.get('schema')!r}; expected {_MANIFEST_SCHEMA!r}"
        )
    if document.get("package") != cake_build.manifest()["package"]:
        raise RuntimeError(
            f"{_TEST_MANIFEST_PATH} names package {document.get('package')!r}, not this package"
        )
    return document


def test_library_name(name: str, arch: str) -> str:
    """``build.test.json`` library of test-only record ``name`` built for ``arch``: ``<library>_<arch>``."""
    found = cake_jit.TEST_MODULES[name]
    if arch not in found["arches"]:
        raise RuntimeError(
            f"test-only record {name} is not delivered for {arch}; its architectures: {found['arches']}"
        )
    return f"{found['library']}_{arch}"


def _test_entry(library: str) -> dict[str, Any]:
    entries = {entry["name"]: entry for entry in test_manifest()["libraries"]}
    try:
        return entries[library]
    except KeyError:
        raise KeyError(
            f"{library!r} is not a library of {_TEST_MANIFEST_PATH}"
        ) from None


def test_jit_spec(library: str) -> Any:
    """FlashInfer's JIT spec of a ``build.test.json`` library, derived exactly as ``cake_build.jit_spec`` derives
    one from ``build.json``: digest-verified sources, ``arch_flags[arch]`` + the entry's ``nvcc_flags`` + the
    ``nvcc_flags_if`` items the local toolkit matches, the closure digest in the spec name, no fast math."""
    document = test_manifest()
    entry = _test_entry(library)
    gen_jit_spec, get_cuda_version = cake_build._jit_toolchain(library)
    cuda = cake_build._local_cuda_version(get_cuda_version)
    if not cuda >= cake_build._version(entry["min_cuda"]):
        raise RuntimeError(
            f"{library} needs CUDA {entry['min_cuda']} or newer; the local toolkit is {cuda[0]}.{cuda[1]}"
        )
    sources: list[Path] = []
    for relative in entry["sources"]:
        path = _PACKAGE / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"][relative]:
            raise RuntimeError(
                f"{path} does not match its {TEST_MANIFEST} digest for {library}: "
                f"{digest} != {entry['sha256'][relative]}"
            )
        sources.append(path)
    flags = [
        *document["arch_flags"][entry["arch"]],
        *entry["nvcc_flags"],
        *cake_build._conditional_flags(entry["nvcc_flags_if"], cuda),
    ]
    return gen_jit_spec(
        f"{library}_{entry['closure_sha256'][:20]}",
        sources,
        extra_cuda_cflags=flags,
        extra_ldflags=list(document["common"]["link_flags"]),
        extra_include_paths=[_PACKAGE / d for d in document["common"]["include_dirs"]],
        use_fast_math=False,
    )


@functools.cache
def load_test(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of test-only record ``name`` (a key of ``TEST_MODULES``) built for ``arch``,
    JIT-built from ``build.test.json`` (never prebuilt, never from ``build.json``)."""
    return test_jit_spec(test_library_name(name, arch)).build_and_load()


def test_specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of the ``arch`` libraries of ``build.test.json`` (test warm-up)."""
    return [
        test_jit_spec(entry["name"])
        for entry in test_manifest()["libraries"]
        if entry["arch"] == arch
    ]


# ----------------------------------------------------------------------------------------------------------------
# Launching one record


@functools.cache
def _argument_plan(name: str) -> tuple[tuple[str | None, int], ...]:
    """The record's plan as ``(logical name, -1)`` per kernel parameter and ``(None, axis)`` per grid argument, in
    ``run`` order; computed once per record."""
    found = record(name)
    logical = found["logical"]
    plan: list[tuple[str | None, int]] = []
    for kind, key in found["arg_plan"]:
        if kind == "grid":
            plan.append((None, "xyz".index(key[-1])))
        else:
            if key not in logical:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} has no logical name"
                )
            plan.append((logical[key], -1))
    return tuple(plan)


def run_module(
    name: str,
    values: Mapping[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
    module: Any | None = None,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid``.

    ``values`` maps the names of ``cake_jit.LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` to tensors and Python scalars;
    the record maps its kernel parameters onto that vocabulary and ``grid`` is the plan's launch grid
    (``cake_jit.launch_grid``).  ``module`` overrides the loaded library (a test-only build from :func:`load_test`);
    by default the shipped record's library of the device's architecture runs.  The kernel runs on the current
    torch stream.
    """
    found = record(name)
    args: list[Any] = []
    for logical_name, axis in _argument_plan(name):
        if logical_name is None:
            args.append(int(grid[axis]))
        else:
            try:
                args.append(values[logical_name])
            except KeyError:
                raise ValueError(
                    f"record {name} needs logical argument {logical_name!r}"
                ) from None
    if module is None:
        module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, found["ffi_entry"])(*args)
