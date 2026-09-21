"""Shared artifact verification, geometry materialization and JIT runtime.

Dtype-specific tensor contracts and launch ABIs live in sibling dtype packages.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import torch

_MANIFEST = "cudnn_frost_selected_kernels.json"


def artifact_root(dtype: str) -> Path:
    """Return the packaged artifact directory for an explicit dtype."""
    return _safe_child(Path(__file__).resolve().parent / "artifacts", dtype)


def _validate_abi(raw: dict[str, Any], op: str) -> bool:
    swap = raw.get("tactic", {}).get("swap_ab", False)
    expected = f"cudnn_frost_{op}{'_swap_ab' if swap else ''}_v1"
    if not isinstance(swap, bool) or raw.get("abi") != expected:
        raise RuntimeError(
            f"unsupported or inconsistent cuDNN Frost ABI: {raw.get('id')}"
        )
    return swap


def _safe_child(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"invalid cuDNN Frost artifact path {relative!r}")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"cuDNN Frost artifact escapes its root: {relative!r}"
        ) from exc
    return path


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _arch_for(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


def _dimension_matches(value: int, rule: Any) -> bool:
    if isinstance(rule, int):
        return value == rule
    if not isinstance(rule, dict):
        return False
    multiple = int(rule.get("multiple_of", 1))
    if multiple <= 0:
        raise RuntimeError("cuDNN Frost dimension 'multiple_of' must be positive")
    return (
        value >= int(rule.get("min", 0))
        and ("max" not in rule or value <= int(rule["max"]))
        and value % multiple == 0
    )


def _read_source(root: Path, raw: dict[str, Any]) -> tuple[Path, str]:
    source = raw.get("source", {})
    path = _safe_child(root, source.get("path", ""))
    if path.suffix != ".py" or not path.is_file():
        raise FileNotFoundError(
            f"cuDNN Frost generated Python source not found: {path}"
        )
    digest = _digest(path)
    if digest != source.get("sha256"):
        raise RuntimeError(f"cuDNN Frost generated source digest mismatch: {path}")
    if "parameters" in source:
        from .source_template import materialize_source

        return materialize_source(path, source["parameters"])
    return path, digest


@functools.cache
def _cached_tactic_digest(source_sha256: str, compiler_key: str) -> str:
    from ...jit.cute_dsl_core import _get_cute_dsl_version

    # Compiler upgrades can change tactic rankings as well as compiled code.
    return hashlib.sha256(
        (source_sha256 + _get_cute_dsl_version() + compiler_key).encode()
    ).hexdigest()[:20]


def _tactic_digest(source_sha256: str) -> str:
    from .compiler import identity_key

    return _cached_tactic_digest(source_sha256, identity_key())


def _load_kernel(kernel, device: torch.device) -> Any:
    with torch.cuda.device(device):
        arch = _arch_for(device)
        if kernel.arch != arch:
            raise ValueError(
                "cuDNN Frost source kernel architecture does not match the device"
            )
        target = os.environ.get("CUTE_DSL_ARCH", arch).replace("_", "")
        if target != arch.replace("_", ""):
            raise ValueError(
                "CUTE_DSL_ARCH must match the cuDNN Frost kernel architecture"
            )
        return _load_source(
            kernel.source_path, kernel.source_sha256, arch, torch.cuda.current_device()
        )


def _load_source(path: Path, digest: str, arch: str, device_index: int) -> Any:
    from .compiler import identity_key

    return _load_source_cached(path, digest, arch, device_index, identity_key())


@functools.lru_cache(maxsize=None)
def _load_source_cached(
    path: Path, digest: str, arch: str, device_index: int, compiler_key: str
) -> Any:
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from .capabilities import require_compiler
    from .compiler import compile_module

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "prepare cuDNN Frost source kernels outside CUDA Graph capture"
        )
    if _digest(path) != digest:
        raise RuntimeError(f"cuDNN Frost generated source digest mismatch: {path}")
    require_compiler(arch, ((path, digest),))

    def compile_kernel():
        # Each device owns its compiled launchable and CUDA module lifetime.
        compiler_suffix = (
            "_" + hashlib.sha256(compiler_key.encode()).hexdigest()[:20]
            if compiler_key
            else ""
        )
        name = (
            f"_flashinfer_cudnn_frost_{digest}_{arch}_{device_index}{compiler_suffix}"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load cuDNN Frost Python source: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            return compile_module(module, arch, compiler_key)
        except BaseException:
            sys.modules.pop(name, None)
            raise

    launch = build_and_load_cute_dsl_kernel(
        module_name=(
            f"cudnn_frost_{digest[:20]}"
            + (
                "_" + hashlib.sha256(compiler_key.encode()).hexdigest()[:20]
                if compiler_key
                else ""
            )
        ),
        kernel_name="kernel",
        compile_fn=compile_kernel,
        extra_key_files=(
            str(path),
            __file__,
            str(Path(__file__).with_name("compiler.py")),
        ),
    )
    # Cache-disabled and persistence-failure paths return CuTe's keyword wrapper;
    # the native MoE adapter needs its positional TVM-FFI function.
    if hasattr(launch, "__tvm_ffi_object__"):
        launch = launch.__tvm_ffi_object__()
    if launch is None:
        raise RuntimeError("cuDNN Frost compilation did not produce a TVM-FFI function")
    return launch


# Expose cache clearing while keying lookups by compiler.
_clear_source_cache = _load_source_cached.cache_clear


def _current_custream(device: torch.device) -> Any:
    from cuda.bindings import driver

    return driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
