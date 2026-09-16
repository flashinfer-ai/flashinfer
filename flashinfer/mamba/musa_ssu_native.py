"""Lazy loader for the opt-in S5000 native Simple-STP MUSA kernel."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.resources
import importlib.util
import json
import os
import platform
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any

from filelock import FileLock

_EXT: Any | None = None
_TORCH_LIB: Any | None = None
_LOAD_LOCK = threading.RLock()
_MODULE_NAME = "flashinfer_musa_simple_stp"


def _source_path() -> Path:
    source = Path(__file__).resolve().parents[2] / "csrc" / "mamba" / "musa_simple_stp.mu"
    if source.is_file():
        return source
    try:
        source = Path(
            importlib.resources.files("flashinfer.data").joinpath("csrc/mamba/musa_simple_stp.mu")
        )
    except (ModuleNotFoundError, TypeError) as exc:
        raise FileNotFoundError("cannot locate flashinfer.data/csrc/mamba/musa_simple_stp.mu") from exc
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _cache_root() -> Path:
    configured = os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", Path.home()))
    return base / ".cache" / "flashinfer" / "musa_simple_stp"


def _package_version(module: Any) -> str | None:
    return str(getattr(module, "__version__", None)) if module is not None else None


def _build_settings(torch: Any, torch_musa: Any) -> dict[str, Any]:
    env_names = {
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "MAX_JOBS",
        "CUDA_HOME",
        "MUSA_HOME",
        "TORCH_CUDA_ARCH_LIST",
        "TORCH_MUSA_ARCH_LIST",
        "MUSA_ARCH_LIST",
    }
    env_names.update(name for name in os.environ if name.startswith(("MUSA_", "FLASHINFER_MUSA_")))
    return {
        "python": sys.version,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": _package_version(torch),
        "torch_musa": _package_version(torch_musa),
        "torch_musa_version": str(
            getattr(getattr(torch_musa, "version", None), "__version__", None)
        ),
        "torch_cxx11_abi": getattr(
            getattr(torch, "_C", None), "_GLIBCXX_USE_CXX11_ABI", None
        ),
        "torch_musa_cxx11_abi": getattr(torch_musa, "_GLIBCXX_USE_CXX11_ABI", None),
        "musa_version": str(getattr(getattr(torch, "version", None), "musa", None)),
        "compiler_env": {name: os.environ.get(name) for name in sorted(env_names)},
        "compile_args": {"cxx": ["-O3", "-std=c++17"], "mcc": ["-O3"]},
    }


def _fingerprint(source: Path, settings: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()[:32]


def _capture_active(torch: Any) -> bool:
    musa = getattr(torch, "musa", None)
    check = getattr(musa, "is_current_stream_capturing", None)
    return bool(check()) if callable(check) else False


def _module_file(artifact_dir: Path) -> Path | None:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        name = json.loads(manifest_path.read_text())["module"]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    module = artifact_dir / name
    return module if module.is_file() else None


def _write_setup(path: Path, source: Path) -> None:
    source_literal = repr(str(source))
    path.write_text(
        "from setuptools import setup\n"
        "import torchada\n"
        "import torch_musa\n"
        "from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
        f"setup(name={_MODULE_NAME!r}, ext_modules=[MUSAExtension({_MODULE_NAME!r}, "
        f"[{source_literal}], extra_compile_args={{'cxx': ['-O3', '-std=c++17'], 'mcc': ['-O3']}})], "
        "cmdclass={'build_ext': BuildExtension})\n"
    )


def _build_and_find(artifact_dir: Path, source: Path, env: dict[str, str]) -> Path:
    setup_py = artifact_dir / "setup.py"
    _write_setup(setup_py, source)
    subprocess.run(
        [sys.executable, str(setup_py), "build_ext", "--inplace"],
        cwd=artifact_dir,
        check=True,
        env=env,
    )
    candidates = sorted(
        path
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        for path in artifact_dir.glob(f"{_MODULE_NAME}*{suffix}")
        if path.is_file()
    )
    if not candidates:
        raise RuntimeError(f"MUSAExtension build produced no module in {artifact_dir}")
    return candidates[0]


def _import_module(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _register_torch_op(extension: Any) -> None:
    """Expose the mutating pybind kernel to Dynamo/MUSA graph capture."""
    global _TORCH_LIB
    if _TORCH_LIB is not None:
        return
    import torch

    lib = torch.library.Library("flashinfer_musa", "DEF")
    lib.define(
        "simple_stp(Tensor(a!) state, Tensor x, Tensor dt, Tensor A, Tensor B, "
        "Tensor C, Tensor D, Tensor src, Tensor dst, Tensor? dt_bias, Tensor? z, "
        "bool dt_softplus, int pad_slot_id, Tensor? out, Tensor? rand_seed, "
        "int philox_rounds) -> Tensor"
    )

    def impl(
        state: Any,
        x: Any,
        dt: Any,
        A: Any,
        B: Any,
        C: Any,
        D: Any,
        src: Any,
        dst: Any,
        dt_bias: Any,
        z: Any,
        dt_softplus: bool,
        pad_slot_id: int,
        out: Any,
        rand_seed: Any,
        philox_rounds: int,
    ) -> Any:
        return extension.musa_ssu_simple(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            src,
            dst,
            dt_bias,
            z,
            dt_softplus,
            pad_slot_id,
            out,
            rand_seed,
            philox_rounds,
        )

    lib.impl("simple_stp", impl, "PrivateUse1")
    _TORCH_LIB = lib


def _load_extension() -> Any:
    global _EXT
    with _LOAD_LOCK:
        if _EXT is not None:
            return _EXT
        if os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") != "1":
            raise RuntimeError("native Simple STP is opt-in")

        import torchada  # noqa: F401  # apply CUDA-to-MUSA mappings first
        import torch
        import torch_musa  # noqa: F401  # activates the MUSA PyTorch extension shim

        source = _source_path()
        settings = _build_settings(torch, torch_musa)
        fingerprint = _fingerprint(source, settings)
        artifact_dir = _cache_root() / fingerprint
        artifact_dir.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(artifact_dir.parent / f"{fingerprint}.lock"), thread_local=False)
        with lock:
            module_path = _module_file(artifact_dir)
            if module_path is None:
                if _capture_active(torch):
                    raise RuntimeError("cannot build native Simple STP during MUSA stream capture")
                artifact_dir.mkdir(parents=True, exist_ok=True)
                build_env = os.environ.copy()
                build_env.setdefault("MUSA_HOME", "/usr/local/musa")
                module_path = _build_and_find(artifact_dir, source, build_env)
                manifest = artifact_dir / "manifest.json.tmp"
                manifest.write_text(json.dumps({"fingerprint": fingerprint, "module": module_path.name}))
                manifest.replace(artifact_dir / "manifest.json")
            _EXT = _import_module(module_path)
            _register_torch_op(_EXT)
            return _EXT


def musa_ssu_one_token_native(*args: Any, **kwargs: Any) -> Any:
    """Invoke the cached native S5000 Simple-STP extension."""
    _load_extension()
    import torch

    return torch.ops.flashinfer_musa.simple_stp(*args, **kwargs)


def preload_musa_simple_stp() -> None:
    """Build and import the extension before graph capture starts."""
    _load_extension()


__all__ = ["musa_ssu_one_token_native", "preload_musa_simple_stp"]
