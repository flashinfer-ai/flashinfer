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

Isolated, fail-closed NVRTC compiler for generated indexed FlashKDA sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Protocol


class GeneratedKDANVRTCWorkerError(RuntimeError):
    """The requested source cannot be rebuilt with the sealed toolchain."""


class _NVRTCModule(Protocol):
    def nvrtcVersion(self) -> tuple[object, int, int]: ...


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"name": path.name, "sha256": digest.hexdigest(), "size_bytes": size}


def _check(error: object, operation: str) -> None:
    code = getattr(error, "value", error)
    if code != 0:
        raise GeneratedKDANVRTCWorkerError(
            f"{operation} failed with NVRTC result {code}"
        )


def _loaded_toolchain_identity(nvrtc: _NVRTCModule) -> str:
    error, major, minor = nvrtc.nvrtcVersion()
    _check(error, "nvrtcVersion")
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise GeneratedKDANVRTCWorkerError("cannot resolve the loaded NVRTC binary")
    mapped: set[Path] = set()
    for line in maps.read_text(encoding="utf-8").splitlines():
        raw = line.rpartition(" ")[2]
        if "libnvrtc" in Path(raw).name:
            try:
                mapped.add(Path(raw).resolve(strict=True))
            except FileNotFoundError as error:
                raise GeneratedKDANVRTCWorkerError(
                    "a mapped NVRTC library is unavailable"
                ) from error
    if not mapped:
        raise GeneratedKDANVRTCWorkerError("cannot resolve the loaded NVRTC binary")
    libraries = [_file_identity(path) for path in sorted(mapped)]
    libraries.sort(key=_canonical_bytes)
    toolchain = {
        "kind": "flashinfer.nvrtc_toolchain_identity",
        "nvrtc_version": [int(major), int(minor)],
        "loaded_libraries": libraries,
    }
    return "sha256:" + _sha256(_canonical_bytes(toolchain))


def _compile(args: argparse.Namespace) -> None:
    from cuda.bindings import nvrtc

    observed_identity = _loaded_toolchain_identity(nvrtc)
    if observed_identity != args.toolchain_identity:
        raise GeneratedKDANVRTCWorkerError(
            "NVRTC toolchain identity differs: "
            f"expected {args.toolchain_identity}, observed {observed_identity}"
        )
    source = args.source.resolve(strict=True).read_bytes()
    if len(source) != args.source_size_bytes or _sha256(source) != args.source_sha256:
        raise GeneratedKDANVRTCWorkerError("CUDA source identity differs")
    error, program = nvrtc.nvrtcCreateProgram(source, b"kernel.cu", 0, [], [])
    _check(error, "nvrtcCreateProgram")
    try:
        options = [
            f"--gpu-architecture={args.architecture}",
            "-std=c++17",
            "-default-device",
        ]
        for include in args.include_dir:
            resolved = include.resolve(strict=True)
            options.append(f"-I{resolved}")
            cccl = resolved / "cccl"
            if (cccl / "cuda" / "std").exists():
                options.append(f"-I{cccl}")
        options.extend(args.compile_option)
        encoded = [option.encode() for option in options]
        (error,) = nvrtc.nvrtcCompileProgram(program, len(encoded), encoded)
        if getattr(error, "value", error) != 0:
            _, size = nvrtc.nvrtcGetProgramLogSize(program)
            log = b"\0" * size
            nvrtc.nvrtcGetProgramLog(program, log)
            raise GeneratedKDANVRTCWorkerError(
                log.decode(errors="replace").rstrip("\0")
            )
        error, size = nvrtc.nvrtcGetCUBINSize(program)
        _check(error, "nvrtcGetCUBINSize")
        cubin = b"\0" * size
        (error,) = nvrtc.nvrtcGetCUBIN(program, cubin)
        _check(error, "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(program)
    if len(cubin) != args.cubin_size_bytes or _sha256(cubin) != args.cubin_sha256:
        raise GeneratedKDANVRTCWorkerError("rebuilt cubin identity differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(cubin)
    os.replace(temporary, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=("sm_100a", "sm_103a"))
    parser.add_argument("--toolchain-identity", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-size-bytes", required=True, type=int)
    parser.add_argument("--cubin-sha256", required=True)
    parser.add_argument("--cubin-size-bytes", required=True, type=int)
    parser.add_argument("--include-dir", action="append", default=[], type=Path)
    parser.add_argument("--compile-option", action="append", default=[])
    args = parser.parse_args()
    try:
        _compile(args)
    except (GeneratedKDANVRTCWorkerError, ImportError, OSError) as error:
        print(f"generated-kda-nvrtc: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
