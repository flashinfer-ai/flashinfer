"""Standalone exact-shape MUSA tensor-core SSD chunk scan.

The production packed SSD path still uses the Triton implementation. This
module exposes the dashboard TCE candidate with its full B/C/state contract so
it can be validated and then wired into a caller that has those tensors.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

from filelock import FileLock

_LOCK = threading.Lock()
_EXT: Any | None = None


def _source_path() -> Path:
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[2] / "csrc" / "mamba" / "musa_ssd_chunk_scan_tce.mu",
        here.parents[1] / "csrc" / "mamba" / "musa_ssd_chunk_scan_tce.mu",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("musa_ssd_chunk_scan_tce.mu is missing")


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
        fingerprint = hashlib.sha256(
            source.read_bytes()
            + str(torch.__version__).encode()
            + str(getattr(torch.version, "musa", None)).encode()
            + os.environ.get("TORCH_MUSA_ARCH_LIST", "").encode()
        ).hexdigest()[:32]
        root = Path(
            os.environ.get(
                "FLASHINFER_MUSA_CACHE_DIR",
                str(Path.home() / ".cache" / "flashinfer"),
            )
        ) / "musa_ssd_chunk_scan_tce" / fingerprint
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root.parent / f"{fingerprint}.lock"), thread_local=False):
            modules = sorted(
                p
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
                for p in root.glob(f"flashinfer_musa_ssd_chunk_scan_tce*{suffix}")
            )
            if not modules:
                setup = root / "setup.py"
                setup.write_text(
                    "from setuptools import setup\n"
                    "import torchada\n"
                    "import torch_musa\n"
                    "from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
                    "setup(name='flashinfer_musa_ssd_chunk_scan_tce', "
                    f"ext_modules=[MUSAExtension('flashinfer_musa_ssd_chunk_scan_tce', [{str(source)!r}], "
                    "extra_compile_args={'cxx': ['-O3', '-std=c++17'], "
                    "'mcc': ['-O3', '-std=c++17', '--offload-arch=mp_31', "
                    "'-Wno-error=address-of-temporary']})], "
                    "cmdclass={'build_ext': BuildExtension})\n"
                )
                subprocess.run(
                    [sys.executable, str(setup), "build_ext", "--inplace"],
                    cwd=root,
                    check=True,
                )
                modules = sorted(
                    p
                    for suffix in importlib.machinery.EXTENSION_SUFFIXES
                    for p in root.glob(f"flashinfer_musa_ssd_chunk_scan_tce*{suffix}")
                )
            if not modules:
                raise RuntimeError("TCE scan build produced no extension")
            module_path = modules[-1]
            spec = importlib.util.spec_from_file_location(
                "flashinfer_musa_ssd_chunk_scan_tce", module_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _EXT = module
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "torch": torch.__version__,
                        "musa": getattr(torch.version, "musa", None),
                        "cache_mode": "per-call",
                    },
                    sort_keys=True,
                )
            )
            return _EXT


def musa_ssd_chunk_scan_tce_native(
    state: Any,
    x: Any,
    dt: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
) -> Any:
    """Run the exact ``T=128,H=64,D=64,N=128,G=8`` TCE candidate.

    ``dt`` is the positive, transformed SSD step input. The candidate performs
    the cumulative sum internally; callers must apply any model-specific
    softplus and bounds before entering this standalone API.

    The extension returns an undefined tensor for a contract mismatch. Raising
    here makes accidental production dispatch visible to the caller.
    """

    import torch

    tensors = (state, x, dt, A, B, C, D)
    if any(t.device.type != "musa" for t in tensors):
        raise ValueError("MUSA SSD TCE scan requires all tensors on MUSA")
    if any(t.device != state.device for t in tensors):
        raise ValueError("MUSA SSD TCE scan requires one device")
    expected = (
        ((64, 64, 128), torch.float32),
        ((128, 64, 64), torch.bfloat16),
        ((128, 64), torch.float32),
        ((64,), torch.float32),
        ((128, 8, 128), torch.bfloat16),
        ((128, 8, 128), torch.bfloat16),
        ((64,), torch.float32),
    )
    for tensor, (shape, dtype) in zip(tensors, expected):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(
                "MUSA SSD TCE scan requires contiguous exact-shape inputs "
                "(state/x/dt/A/B/C/D)"
            )
        if tensor.data_ptr() % 16:
            raise ValueError("MUSA SSD TCE scan requires 16-byte aligned inputs")

    result = _load_extension().ssd_scan(state, x, dt, A, B, C, D)
    if result is None or not getattr(result, "defined", lambda: True)():
        raise ValueError(
            "MUSA SSD TCE scan requires contiguous T128/H64/D64/N128/G8 "
            "BF16 inputs, FP32 state/dt/A/D, and one MUSA device"
        )
    return result


def preload_musa_ssd_chunk_scan_tce() -> None:
    _load_extension()


__all__ = [
    "musa_ssd_chunk_scan_tce_native",
    "preload_musa_ssd_chunk_scan_tce",
]
