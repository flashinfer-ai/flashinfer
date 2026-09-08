"""Build-time NVRTC packaging for exact generated FlashKDA kernels."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, cast

from filelock import FileLock

from .cpp_ext import get_cuda_path


_BODY_DEFINE = re.compile(
    r'^#define FLASHKDA_GENERATED_BODY_FILE "([^"]+)"$', re.MULTILINE
)
_KERNEL_DEFINE = re.compile(
    r"^#define FLASHKDA_GENERATED_KERNEL ([A-Za-z_][A-Za-z0-9_]*)$", re.MULTILINE
)
_C_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_SCHEMA_VERSION = 1


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _result_ok(result: object) -> bool:
    try:
        return int(cast(Any, result)) == 0
    except TypeError:
        return getattr(result, "value", result) == 0


def _compile_log(nvrtc: object, program: object) -> str:
    nvrtc_api = cast(Any, nvrtc)
    result, size = nvrtc_api.nvrtcGetProgramLogSize(program)
    if not _result_ok(result) or size <= 1:
        return ""
    log = b"\0" * size
    (result,) = nvrtc_api.nvrtcGetProgramLog(program, log)
    if not _result_ok(result):
        return ""
    return log.decode(errors="replace").rstrip("\0")


def _cuda_include_dirs() -> tuple[Path, ...]:
    cuda_include = (Path(get_cuda_path()) / "include").resolve()
    if not (cuda_include / "cuda_bf16.h").is_file():
        raise FileNotFoundError(
            f"CUDA include directory lacks cuda_bf16.h: {cuda_include}"
        )
    paths = [cuda_include]
    cccl_include = cuda_include / "cccl"
    if (cccl_include / "cuda" / "std").is_dir():
        paths.append(cccl_include)
    return tuple(paths)


def _compile_cubin(
    source: bytes, *, source_name: str, options: tuple[str, ...]
) -> bytes:
    from cuda.bindings import nvrtc

    result, program = nvrtc.nvrtcCreateProgram(source, source_name.encode(), 0, [], [])
    if not _result_ok(result):
        raise RuntimeError(
            f"nvrtcCreateProgram failed for generated FlashKDA {source_name}: {result}"
        )
    try:
        encoded_options = [option.encode() for option in options]
        (result,) = nvrtc.nvrtcCompileProgram(
            program, len(encoded_options), encoded_options
        )
        if not _result_ok(result):
            raise RuntimeError(
                f"NVRTC compilation failed for generated FlashKDA {source_name}: "
                f"{result}\n{_compile_log(nvrtc, program)}"
            )
        result, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
        if not _result_ok(result):
            raise RuntimeError(
                f"nvrtcGetCUBINSize failed for generated FlashKDA {source_name}: "
                f"{result}"
            )
        cubin = b"\0" * cubin_size
        (result,) = nvrtc.nvrtcGetCUBIN(program, cubin)
        if not _result_ok(result):
            raise RuntimeError(
                f"nvrtcGetCUBIN failed for generated FlashKDA {source_name}: {result}"
            )
        return cubin
    finally:
        nvrtc.nvrtcDestroyProgram(program)


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def prepare_generated_flash_kda_cubin(
    build_dir: Path,
    *,
    selector_path: Path,
    body_path: Path,
    module_ident: str,
    target: str,
) -> Mapping[str, Path]:
    """Compile and return one content-addressed cubin for Ninja embedding."""

    if _C_IDENTIFIER.fullmatch(module_ident) is None:
        raise ValueError(
            f"invalid generated FlashKDA module identifier: {module_ident!r}"
        )
    try:
        arch = _TARGET_ARCH[target]
    except KeyError as error:
        raise ValueError(
            f"unsupported generated FlashKDA NVRTC target: {target}"
        ) from error

    selector = selector_path.read_text()
    body_match = _BODY_DEFINE.search(selector)
    kernel_match = _KERNEL_DEFINE.search(selector)
    if body_match is None or kernel_match is None:
        raise ValueError(
            f"generated FlashKDA selector lacks body or kernel identity: {selector_path}"
        )
    if body_match.group(1) != body_path.name:
        raise ValueError(
            f"generated FlashKDA selector/body mismatch: {body_match.group(1)!r} "
            f"!= {body_path.name!r}"
        )

    source = body_path.read_bytes()
    include_dirs = _cuda_include_dirs()
    options = (
        f"--gpu-architecture={arch}",
        "-std=c++17",
        "-default-device",
        *(f"-I{path}" for path in include_dirs),
        "--use_fast_math",
    )
    if any("o1" in option.lower() for option in options):
        raise RuntimeError(
            f"forbidden O1 option in generated FlashKDA NVRTC flags: {options}"
        )
    inputs = {
        "schema_version": _SCHEMA_VERSION,
        "source_name": body_path.name,
        "source_sha256": _sha256(source),
        "module_ident": module_ident,
        "kernel_name": kernel_match.group(1),
        "target": target,
        "arch": arch,
        "compile_options": list(options),
        "optimization_level_one_absent": True,
    }
    cache_ident = _sha256(_canonical_json(inputs))[:20]
    cache_dir = build_dir.parent / "_flash_kda_nvrtc_cubins"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = cache_dir / f"{cache_ident}.cubin"
    receipt_path = cache_dir / f"{cache_ident}.json"
    with FileLock(cache_dir / f"{cache_ident}.lock", thread_local=False):
        reusable = False
        if cubin_path.is_file() and receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text())
                reusable = (
                    receipt.get("inputs") == inputs
                    and receipt.get("cubin_sha256") == _sha256(cubin_path.read_bytes())
                    and receipt.get("cubin_size") == cubin_path.stat().st_size
                )
            except (OSError, TypeError, ValueError):
                reusable = False
        if not reusable:
            cubin = _compile_cubin(source, source_name=body_path.name, options=options)
            receipt = {
                "schema_version": _SCHEMA_VERSION,
                "inputs": inputs,
                "cubin_sha256": _sha256(cubin),
                "cubin_size": len(cubin),
            }
            _write_atomic(cubin_path, cubin)
            _write_atomic(
                receipt_path,
                json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n",
            )
    return {module_ident: cubin_path}


__all__ = ["prepare_generated_flash_kda_cubin"]
