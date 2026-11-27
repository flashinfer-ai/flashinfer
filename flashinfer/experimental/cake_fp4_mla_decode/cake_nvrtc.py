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

NVRTC builds of the package's decode libraries.

A ``build.json`` entry with an ``nvrtc`` record (``{"source", "sha256",
"options"}``) is a library whose device source NVRTC compiles, with the
recorded options, to a cubin that the library embeds: nvcc compiles the
entry's ``sources`` (the binding) alone, and FlashInfer's JIT embeds the
cubin under the program name (``<library>`` without ``_<arch>``), where the
binding's ``TVM_FFI_EMBED_CUBIN`` loads it once per process.  NVRTC lowers
the decode to the code it was validated and measured with; nvcc's front end
lowers the same source to different PTX, and the resulting SASS is slower on
``sm_103a``.  The reducer entries have no ``nvrtc`` record and build through
:mod:`.cake_build` as every other Cake package's do.

The options are the recorded ones after ``--gpu-architecture=<arch>``,
followed by ``-I`` for the toolkit's headers (``$CUDA_HOME/include`` and its
CCCL directory), the directories the source's ``#include`` lines resolve in.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import cake_build

_PACKAGE = Path(__file__).parent


def nvrtc_record(library: str) -> dict[str, Any] | None:
    """The ``nvrtc`` record of the ``build.json`` entry ``library``, or ``None``
    when nvcc builds all of its sources."""
    return cake_build.libraries_by_name()[library].get("nvrtc")


def _program(library: str) -> str:
    """The embedded cubin's name: the entry name without its ``_<arch>``."""
    entry = cake_build.libraries_by_name()[library]
    return library[: -len(entry["arch"]) - 1]


def _checked(relative: str, expected: str, library: str) -> Path:
    path = _PACKAGE / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"{path} does not match its build.json digest for {library}: {digest} != {expected}"
        )
    return path


@functools.cache
def _include_dirs() -> tuple[str, ...]:
    """The toolkit's header directories: ``$CUDA_HOME/include`` and, when the
    toolkit keeps CCCL beside it, ``include/cccl``."""
    from flashinfer.jit.cpp_ext import get_cuda_path

    include = (Path(get_cuda_path()) / "include").resolve()
    dirs = [include]
    if (include / "cccl" / "cuda" / "std").is_dir():
        dirs.append(include / "cccl")
    return tuple(str(d) for d in dirs)


def _nvrtc_version() -> tuple[int, int]:
    from cuda.bindings import nvrtc

    status, major, minor = nvrtc.nvrtcVersion()
    if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion failed: {status}")
    return int(major), int(minor)


def compile_options(library: str) -> list[str]:
    """NVRTC's options for ``library``'s device source: the target, the
    recorded options, then the toolkit's include directories."""
    entry = cake_build.libraries_by_name()[library]
    record = entry["nvrtc"]
    return [
        f"--gpu-architecture={entry['arch']}",
        *record["options"],
        *(f"-I{d}" for d in _include_dirs()),
    ]


def compile_cubin(library: str) -> bytes:
    """NVRTC's cubin of ``library``'s device source (digest-checked first;
    refused when the loaded NVRTC is older than the entry's ``min_cuda``)."""
    from cuda.bindings import nvrtc

    record = nvrtc_record(library)
    if record is None:
        raise KeyError(f"{library} has no NVRTC build in build.json")
    floor = cake_build.libraries_by_name()[library]["min_cuda"]
    if _nvrtc_version() < tuple(int(part) for part in floor.split(".")[:2]):
        major, minor = _nvrtc_version()
        raise RuntimeError(
            f"{library} needs NVRTC {floor} or newer (its options and PTX); "
            f"the loaded NVRTC is {major}.{minor}"
        )
    source = _checked(record["source"], record["sha256"], library)
    options = [option.encode() for option in compile_options(library)]
    status, program = nvrtc.nvrtcCreateProgram(
        source.read_bytes(), source.name.encode(), 0, [], []
    )
    if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram of {source.name} failed: {status}")
    try:
        (status,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, size = nvrtc.nvrtcGetProgramLogSize(program)
            log = bytearray(size)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(
                f"NVRTC failed to compile {source.name} for {library}:\n"
                + bytes(log).rstrip(b"\0").decode(errors="replace")
            )
        status, size = nvrtc.nvrtcGetCUBINSize(program)
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcGetCUBINSize of {source.name} failed: {status}")
        cubin = bytearray(size)
        (status,) = nvrtc.nvrtcGetCUBIN(program, cubin)
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcGetCUBIN of {source.name} failed: {status}")
    finally:
        nvrtc.nvrtcDestroyProgram(program)
    return bytes(cubin)


def _cubins(build_dir: Path, *, library: str) -> dict[str, Path]:
    """FlashInfer's ``embedded_cubin_factory`` of ``library``: ``{program:
    cubin path}``, the cubin compiled once per NVRTC version, option list and
    source digest and kept under the JIT build directory."""
    record = nvrtc_record(library)
    key = hashlib.sha256(
        json.dumps(
            {
                "nvrtc": _nvrtc_version(),
                "options": compile_options(library),
                "source": record["sha256"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    path = build_dir / f"{_program(library)}_{key}.cubin"
    if not path.is_file():
        build_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
        partial.write_bytes(compile_cubin(library))
        os.replace(partial, path)
    return {_program(library): path}


def jit_spec(library: str) -> Any:
    """FlashInfer's ``JitSpecNvcc`` of an NVRTC entry: the binding (digest-
    checked) built with the manifest's nvcc flags, NVRTC's cubin of the device
    source embedded; named after the entry's closure as :mod:`.cake_build`
    names every JIT library."""
    from flashinfer.jit.core import gen_jit_spec
    from flashinfer.jit.cpp_ext import get_cuda_version

    entry = cake_build.libraries_by_name()[library]
    if "nvrtc" not in entry:
        raise KeyError(f"{library} has no NVRTC build in build.json")
    document = cake_build.manifest()
    version = get_cuda_version()
    cuda = (int(version.major), int(version.minor))
    floor = tuple(int(part) for part in entry["min_cuda"].split(".")[:2])
    if cuda < floor:
        raise RuntimeError(
            f"{library} needs CUDA {entry['min_cuda']} or newer; the local toolkit is {cuda[0]}.{cuda[1]}"
        )
    conditional = [
        flag
        for item in entry["nvcc_flags_if"]
        if _matches(item["cuda"], cuda)
        for flag in item["flags"]
    ]
    return gen_jit_spec(
        f"{library}_{entry['closure_sha256'][:20]}",
        [
            _checked(source, entry["sha256"][source], library)
            for source in entry["sources"]
        ],
        extra_cuda_cflags=[
            *document["arch_flags"][entry["arch"]],
            *entry["nvcc_flags"],
            *conditional,
        ],
        extra_ldflags=list(document["common"]["link_flags"]),
        extra_include_paths=[_PACKAGE / d for d in document["common"]["include_dirs"]],
        embedded_cubin_factory=functools.partial(_cubins, library=library),
        use_fast_math=False,
    )


def _matches(constraint: str, cuda: tuple[int, int]) -> bool:
    operator, version = constraint[:2], constraint[2:]
    wanted = tuple(int(part) for part in version.split("."))
    if operator == "==":
        return cuda == wanted
    if operator == ">=":
        return cuda >= wanted
    raise ValueError(f"unsupported CUDA constraint {constraint!r}")


def load_library(library: str) -> Any:
    """The loaded tvm-ffi module of ``library``: a prebuilt ``aot/`` library
    (a builder embedded the cubin) through :mod:`.cake_build`, else the JIT
    build of :func:`jit_spec`; an entry without ``nvrtc`` loads through
    :mod:`.cake_build`."""
    if nvrtc_record(library) is None or cake_build.prebuilt_path(library).is_file():
        return cake_build.load_library(library)
    return jit_spec(library).build_and_load()
