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

Library loader of one Cake-generated FlashInfer package.

``build.json`` (schema ``cake-build/1``) next to this file is the only record
of what the package compiles: one entry per ``<program>[_<spec>]_<arch>``
library with its package-relative CUDA sources, nvcc flags, toolkit-conditional
flags and digests.  A library is served from ``aot/<name>.so`` when a builder
(a vendoring consumer's build rules, or FlashInfer's
``scripts/check_cake_build_manifests.py``) has compiled it from the manifest,
and otherwise through FlashInfer's JIT from the same sources and flags.

The prebuilt path never imports ``flashinfer``, so the package stays usable in
a process without a FlashInfer installation.  A builder that fills ``aot/``
also writes the sidecar ``aot/build_info.json`` (schema ``cake-build-info/1``:
``{"schema", "cuda": "13.0.88", "cudart_version": 13000, "arches": [...]}``,
``arches`` naming every architecture the builder compiled).  Every prebuilt
library exports ``build_cuda_version()``, the ``CUDART_VERSION`` it was
compiled against, and :func:`load_library` refuses a library whose value
disagrees with the sidecar.

Public surface (``__all__``): :func:`manifest`, :func:`libraries_by_name`,
:func:`prebuilt_path`, :func:`library_path`, :func:`load_library`,
:func:`jit_spec`, :func:`jit_specs` and :func:`record_cuda_version`.

The Cake exporter writes this file byte-identically into every package from
its loader template; do not edit it in place.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path
from typing import Any

__all__ = [
    "jit_spec",
    "jit_specs",
    "libraries_by_name",
    "library_path",
    "load_library",
    "manifest",
    "prebuilt_path",
    "record_cuda_version",
]

_PACKAGE = Path(__file__).parent
_MANIFEST_SCHEMA = "cake-build/1"
_BUILD_INFO_SCHEMA = "cake-build-info/1"
_MANIFEST_PATH = _PACKAGE / "build.json"
_BUILD_INFO_PATH = _PACKAGE / "aot" / "build_info.json"
_CUDA_CONSTRAINT = re.compile(r"(==|>=)([0-9]+)\.([0-9]+)")


@functools.cache
def manifest() -> dict[str, Any]:
    """The parsed ``build.json`` of this package (read on first use, not at import).

    One cached document shared by every caller: read it, do not modify it.
    """
    try:
        text = _MANIFEST_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{_MANIFEST_PATH} is missing: the package was not exported with its {_MANIFEST_SCHEMA} manifest",
        ) from exc
    document = json.loads(text)
    schema = document.get("schema")
    if schema != _MANIFEST_SCHEMA:
        raise RuntimeError(
            f"{_MANIFEST_PATH} has schema {schema!r}; this loader reads {_MANIFEST_SCHEMA!r}",
        )
    return document


@functools.cache
def libraries_by_name() -> dict[str, dict[str, Any]]:
    """``name -> entry`` over the ``libraries`` of ``build.json``."""
    index: dict[str, dict[str, Any]] = {}
    for entry in manifest()["libraries"]:
        if entry["name"] in index:
            raise RuntimeError(f"{_MANIFEST_PATH} lists {entry['name']!r} twice")
        index[entry["name"]] = entry
    return index


def _entry(name: str) -> dict[str, Any]:
    try:
        return libraries_by_name()[name]
    except KeyError:
        raise KeyError(f"{name!r} is not a library of {_MANIFEST_PATH}") from None


def prebuilt_path(name: str) -> Path:
    """Where a builder places the compiled library of ``name`` (``aot/<name>.so``)."""
    return _PACKAGE / _entry(name)["path"]


def _cuda_constraint_matches(constraint: str, cuda: tuple[int, int]) -> bool:
    """Does toolkit ``cuda`` (major, minor) satisfy ``"==X.Y"`` or ``">=X.Y"``?"""
    match = _CUDA_CONSTRAINT.fullmatch(constraint)
    if match is None:
        raise ValueError(
            f"unsupported CUDA constraint {constraint!r}; expected '==X.Y' or '>=X.Y'",
        )
    operator, major, minor = match.groups()
    version = (int(major), int(minor))
    if operator == "==":
        return tuple(cuda) == version
    return tuple(cuda) >= version


def _conditional_flags(items: list[Any], cuda: tuple[int, int]) -> list[str]:
    """The flags of every ``nvcc_flags_if`` item whose ``cuda`` constraint matches."""
    flags: list[str] = []
    for item in items:
        if _cuda_constraint_matches(item["cuda"], cuda):
            flags.extend(item["flags"])
    return flags


def _jit_toolchain(name: str) -> tuple[Any, Any]:
    """``(gen_jit_spec, get_cuda_version)`` of the local FlashInfer, or the error."""
    try:
        from flashinfer.jit.core import gen_jit_spec
        from flashinfer.jit.cpp_ext import get_cuda_version
    except ImportError as exc:
        raise RuntimeError(
            f"{name} is not prebuilt ({prebuilt_path(name)}) and flashinfer's JIT is not available",
        ) from exc
    return gen_jit_spec, get_cuda_version


def _local_cuda_version(get_cuda_version: Any) -> tuple[int, int]:
    version = get_cuda_version()
    return (int(version.major), int(version.minor))


def _version(text: str) -> tuple[int, int]:
    """``(major, minor)`` of a dotted version such as ``"13.0"`` or ``"13.0.88"``."""
    parts = str(text).split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise RuntimeError(f"malformed CUDA version {text!r}")
    return (int(parts[0]), int(parts[1]))


def _toolkit_builds(entry: dict[str, Any], cuda: tuple[int, int]) -> bool:
    """Does toolkit ``cuda`` reach the entry's ``min_cuda`` floor?"""
    return cuda >= _version(entry["min_cuda"])


def _verified_sources(entry: dict[str, Any]) -> list[Path]:
    """The entry's sources as package paths, each checked against its digest."""
    sources: list[Path] = []
    for relative in entry["sources"]:
        path = _PACKAGE / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = entry["sha256"][relative]
        if digest != expected:
            raise RuntimeError(
                f"{path} does not match its build.json digest for {entry['name']}: {digest} != {expected}",
            )
        sources.append(path)
    return sources


def jit_spec(name: str) -> Any:
    """FlashInfer's ``JitSpecNvcc`` of ``name``, built from the manifest entry.

    Sources are digest-verified here and only here, so a modified or partially
    delivered source tree fails closed before anything compiles.  The spec name
    carries the closure digest: a changed closure never reuses a stale JIT cache
    entry.  Flags are ``arch_flags[arch]`` + the entry's ``nvcc_flags`` + the
    ``nvcc_flags_if`` items matching the local toolkit; FlashInfer's JIT adds
    its own defaults (``-O3``, ``-fPIC``, ...), so ``common.nvcc_flags`` are not
    repeated here.  A toolkit below the entry's ``min_cuda`` raises.
    """
    entry = _entry(name)
    gen_jit_spec, get_cuda_version = _jit_toolchain(name)
    document = manifest()
    cuda = _local_cuda_version(get_cuda_version)
    if not _toolkit_builds(entry, cuda):
        raise RuntimeError(
            f"{name} needs CUDA {entry['min_cuda']} or newer; the local toolkit is {cuda[0]}.{cuda[1]}",
        )
    flags = [
        *document["arch_flags"][entry["arch"]],
        *entry["nvcc_flags"],
        *_conditional_flags(entry["nvcc_flags_if"], cuda),
    ]
    include_dirs = [_PACKAGE / d for d in document["common"]["include_dirs"]]
    return gen_jit_spec(
        f"{name}_{entry['closure_sha256'][:20]}",
        _verified_sources(entry),
        extra_cuda_cflags=flags,
        extra_ldflags=list(document["common"]["link_flags"]),
        extra_include_paths=include_dirs,
        use_fast_math=False,
    )


def _names_of(arch: str) -> list[str]:
    index = libraries_by_name()
    return [name for name in index if index[name]["arch"] == arch]


def jit_specs(arch: str) -> list[Any]:
    """JIT specifications of the ``arch`` libraries still to build locally.

    The entries without a prebuilt file that the local toolkit can build: the
    list feeds ``flashinfer.jit.build_jit_specs`` warm-ups, so an entry whose
    ``min_cuda`` exceeds the local toolkit is left out (the manifest check
    script's "skip min_cuda > toolkit" rule), while :func:`jit_spec` on that
    entry by name still raises.  When every ``arch`` library is prebuilt the
    list is empty and ``flashinfer`` is not imported.
    """
    pending = [name for name in _names_of(arch) if not prebuilt_path(name).is_file()]
    if not pending:
        return []
    _, get_cuda_version = _jit_toolchain(pending[0])
    cuda = _local_cuda_version(get_cuda_version)
    return [jit_spec(name) for name in pending if _toolkit_builds(_entry(name), cuda)]


def library_path(name: str) -> Path:
    """The prebuilt library of ``name`` when present, else its JIT cache path."""
    path = prebuilt_path(name)
    if path.is_file():
        return path
    return jit_spec(name).get_library_path()


@functools.cache
def _build_info() -> dict[str, Any]:
    """The builder-written ``aot/build_info.json`` sidecar (prebuilt path only)."""
    try:
        text = _BUILD_INFO_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{_BUILD_INFO_PATH} is missing although prebuilt libraries exist under {_PACKAGE / 'aot'}: the builder must write the {_BUILD_INFO_SCHEMA} sidecar",
        ) from exc
    info = json.loads(text)
    schema = info.get("schema")
    if schema != _BUILD_INFO_SCHEMA:
        raise RuntimeError(
            f"{_BUILD_INFO_PATH} has schema {schema!r}; this loader reads {_BUILD_INFO_SCHEMA!r}",
        )
    return info


def _prebuilt_info(path: Path, arch: str) -> dict[str, Any]:
    """The sidecar, which must record a build of ``arch`` for the prebuilt ``path``."""
    info = _build_info()
    if arch not in info["arches"]:
        raise RuntimeError(
            f"{path} exists but {_BUILD_INFO_PATH} records no {arch} build (arches {info['arches']})",
        )
    return info


def load_library(name: str) -> Any:
    """The loaded tvm-ffi module of ``name``: the prebuilt library, else a JIT build."""
    entry = _entry(name)
    path = _PACKAGE / entry["path"]
    if not path.is_file():
        return jit_spec(name).build_and_load()
    info = _prebuilt_info(path, entry["arch"])
    import tvm_ffi

    module = tvm_ffi.load_module(str(path))
    built = int(module.build_cuda_version())
    if built != int(info["cudart_version"]):
        raise RuntimeError(
            f"{path} was compiled against CUDART {built} but {_BUILD_INFO_PATH} records {info['cudart_version']}",
        )
    return module


def record_cuda_version(arch: str) -> tuple[int, int]:
    """``(major, minor)`` of the toolkit that builds the ``arch`` libraries.

    Prebuilt path, taken as soon as any library of the manifest exists under
    ``aot/``: the sidecar ``aot/build_info.json`` is mandatory (missing ->
    ``RuntimeError``; no local nvcc is consulted) and its ``cuda`` is the
    answer whenever its ``arches`` list ``arch`` -- the builder compiled that
    architecture, whether or not every one of its libraries is present.  An
    ``arch`` library under ``aot/`` that the sidecar does not list raises, as
    :func:`load_library` does.

    JIT path, when nothing is prebuilt or the builder did not compile
    ``arch``: the local toolkit of FlashInfer's JIT (``RuntimeError`` in a
    process without a FlashInfer installation).  Call it for the architecture
    you are about to load.
    """
    names = _names_of(arch)
    if not names:
        raise KeyError(f"{arch!r} has no libraries in {_MANIFEST_PATH}")
    index = libraries_by_name()
    prebuilt = [name for name in index if (_PACKAGE / index[name]["path"]).is_file()]
    if prebuilt:
        of_arch = [name for name in prebuilt if index[name]["arch"] == arch]
        if of_arch:
            return _version(_prebuilt_info(prebuilt_path(of_arch[0]), arch)["cuda"])
        info = _build_info()
        if arch in info["arches"]:
            return _version(info["cuda"])
    _, get_cuda_version = _jit_toolchain(names[0])
    return _local_cuda_version(get_cuda_version)
