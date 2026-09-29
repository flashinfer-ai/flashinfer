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
"""

# JIT loader for the generated exact-SM110 GQA decode kernels.

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec, logger, sm110a_nvcc_flags

_OPERATOR_DIR = "sm110_gqa_decode"
_MANIFEST = "manifest.json"


def _get_operator_dir() -> Path:
    operator_dir = Path(__file__).resolve().parent / "csrc" / _OPERATOR_DIR
    if (operator_dir / _MANIFEST).is_file():
        return operator_dir
    raise FileNotFoundError(
        "SM110 GQA decode sources were not found in the installed package or source checkout"
    )


def _read_manifest() -> tuple[Path, dict[str, Any]]:
    operator_dir = _get_operator_dir()
    manifest = json.loads((operator_dir / _MANIFEST).read_text())
    if manifest.get("architecture") != "sm_110a":
        raise RuntimeError("SM110 GQA decode manifest has an unexpected architecture")

    routes = manifest.get("routes")
    if not isinstance(routes, list) or {
        route.get("ffi_entry") for route in routes if isinstance(route, dict)
    } != {"run_short", "run_long"}:
        raise RuntimeError("SM110 GQA decode manifest has an unexpected launch ABI")

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 4:
        raise RuntimeError("SM110 GQA decode manifest must contain four source files")
    for artifact in files:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"invalid generated source path: {relative}")
        source = operator_dir / relative
        if not source.is_file():
            raise FileNotFoundError(f"generated source not found: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise RuntimeError(f"generated source digest mismatch: {source}")
    return operator_dir, manifest


def _get_flashinfer_header_dirs() -> list[Path]:
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer" / "layout.cuh"
    ).is_file():
        return installed

    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer" / "layout.cuh"
    ).is_file():
        return source

    raise FileNotFoundError("FlashInfer JIT headers were not found")


@functools.cache
def gen_sm110_gqa_decode_module() -> JitSpec:
    """Create the exact-SM110a JIT specification from the sealed sources."""

    operator_dir, manifest = _read_manifest()
    sources = [operator_dir / artifact["path"] for artifact in manifest["files"]]
    identity = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    spec = gen_jit_spec(
        name=f"sm110_gqa_decode_{identity}",
        sources=sources,
        extra_cuda_cflags=[*sm110a_nvcc_flags],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            operator_dir,
            operator_dir / "sm_110a",
            *_get_flashinfer_header_dirs(),
        ],
    )
    logger.info(f"Generated SM110 GQA decode JIT spec: {spec.name}")
    return spec


def _check_exact_sm110a(device: Any = None) -> None:
    import torch

    from ...utils import get_compute_capability, is_sm110a_supported

    resolved = torch.device("cuda") if device is None else torch.device(device)
    capability = get_compute_capability(resolved)
    if capability != (11, 0) or not is_sm110a_supported(resolved):
        raise RuntimeError(
            "SM110 GQA decode requires compute capability 11.0 and CUDA 13.0 or newer; "
            f"got compute capability {capability[0]}.{capability[1]} with CUDA {torch.version.cuda}"
        )


@functools.cache
def _build_and_load() -> Any:
    module = gen_sm110_gqa_decode_module().build_and_load()
    logger.info("Loaded SM110 GQA decode module")
    return module


def load_sm110_gqa_decode_module(*, device: Any = None) -> Any:
    """Build or load the generated module for an exact SM110a device."""

    _check_exact_sm110a(device)
    return _build_and_load()


__all__ = ["gen_sm110_gqa_decode_module", "load_sm110_gqa_decode_module"]
