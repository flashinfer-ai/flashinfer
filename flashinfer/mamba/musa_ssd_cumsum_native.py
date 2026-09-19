"""Compiled MUSA fast path for the regular H64/C128 SSD cumsum stage."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.machinery
import threading
from typing import Any

from filelock import FileLock

_LOCK = threading.Lock()
_EXT: Any | None = None
_LIBS: list[Any] = []
_REGISTERED = False


def _source_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "csrc" / "mamba" / "musa_ssd_chunk_cumsum.mu",
        here.parents[1] / "csrc" / "mamba" / "musa_ssd_chunk_cumsum.mu",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("musa_ssd_chunk_cumsum.mu is missing from FlashInfer")


def _cache_root() -> Path:
    root = os.environ.get("FLASHINFER_MUSA_CACHE_DIR")
    if root:
        return Path(root).expanduser() / "musa_ssd_chunk_cumsum"
    return Path.home() / ".cache" / "flashinfer" / "musa_ssd_chunk_cumsum"


def _settings(torch: Any, torch_musa: Any) -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "torch_musa": getattr(torch_musa, "__version__", "unknown"),
        "musa": getattr(torch.version, "musa", None),
        "arch": os.environ.get("MUSA_ARCH", "mp_31"),
        "flags": ["-O3", "--offload-arch=mp_31"],
    }


def _fingerprint(source: Path, settings: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()[:32]


def _module_file(directory: Path) -> Path | None:
    modules = sorted(directory.glob("flashinfer_musa_ssd_chunk_cumsum*.so"))
    return modules[-1] if modules else None


def _import_module(module_path: Path) -> Any:
    name = "flashinfer_musa_ssd_chunk_cumsum"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_active(torch: Any) -> bool:
    musa = getattr(torch, "musa", None)
    checker = getattr(musa, "is_current_stream_capturing", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return False


def _load_extension() -> Any:
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        import torchada  # noqa: F401
        import torch
        import torch_musa  # noqa: F401

        source = _source_path()
        fingerprint = _fingerprint(source, _settings(torch, torch_musa))
        artifact = _cache_root() / fingerprint
        artifact.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(artifact.parent / f"{fingerprint}.lock"), thread_local=False):
            module_path = _module_file(artifact)
            if module_path is None:
                if _capture_active(torch):
                    raise RuntimeError("cannot build SSD cumsum during MUSA graph capture")
                artifact.mkdir(parents=True, exist_ok=True)
                setup_py = artifact / "setup.py"
                setup_py.write_text(
                    "from setuptools import setup\n"
                    "import torchada\n"
                    "import torch_musa\n"
                    "from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
                    "setup(name='flashinfer_musa_ssd_chunk_cumsum', "
                    f"ext_modules=[MUSAExtension('flashinfer_musa_ssd_chunk_cumsum', [{str(source)!r}], "
                    "extra_compile_args={'cxx': ['-O3', '-std=c++17'], "
                    "'mcc': ['-O3', '--offload-arch=mp_31']})], "
                    "cmdclass={'build_ext': BuildExtension})\n"
                )
                env = os.environ.copy()
                subprocess.run(
                    [sys.executable, str(setup_py), "build_ext", "--inplace"],
                    cwd=artifact, env=env, check=True,
                )
                candidates = sorted(
                    p for suffix in importlib.machinery.EXTENSION_SUFFIXES
                    for p in artifact.glob(f"flashinfer_musa_ssd_chunk_cumsum*{suffix}")
                    if p.is_file()
                )
                if not candidates:
                    raise RuntimeError("SSD cumsum build produced no extension")
                module_path = candidates[0]
                (artifact / "manifest.json").write_text(
                    json.dumps({"fingerprint": fingerprint, "module": module_path.name})
                )
            _EXT = _import_module(module_path)
            return _EXT


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    import torch

    with _LOCK:
        if _REGISTERED:
            return
        lib = None
        try:
            lib = torch.library.Library("flashinfer_musa", "DEF")
        except RuntimeError:
            lib = torch.library.Library("flashinfer_musa", "FRAGMENT")
        try:
            lib.define(
                "ssd_chunk_cumsum(Tensor dt, Tensor A, Tensor bias) -> (Tensor, Tensor)"
            )
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise
        _LIBS.append(lib)

        def impl(dt: Any, A: Any, bias: Any) -> tuple[Any, Any]:
            return _load_extension().musa_ssd_chunk_cumsum(dt, A, bias)

        try:
            torch.library.impl(
                "flashinfer_musa::ssd_chunk_cumsum", "PrivateUse1", impl
            )
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise

        def fake(dt: Any, A: Any, bias: Any) -> tuple[Any, Any]:
            del A, bias
            shape = (64, dt.shape[0] // 128, 128)
            return dt.new_empty(shape), dt.new_empty(shape)

        try:
            torch.library.register_fake("flashinfer_musa::ssd_chunk_cumsum", fake)
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise
        _REGISTERED = True


def musa_ssd_chunk_cumsum_native(dt: Any, A: Any, bias: Any) -> tuple[Any, Any]:
    _ensure_registered()
    import torch

    return torch.ops.flashinfer_musa.ssd_chunk_cumsum(dt, A, bias)


def preload_musa_ssd_chunk_cumsum() -> None:
    _ensure_registered()
    _load_extension()


_ensure_registered()

__all__ = ["musa_ssd_chunk_cumsum_native", "preload_musa_ssd_chunk_cumsum"]
