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

import functools
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

from filelock import FileLock

from . import env as jit_env
from .core import logger

_META_FILENAME = "meta.json"


def cute_dsl_cache_disabled() -> bool:
    """Whether the CuTe-DSL disk cache is disabled via environment variable."""
    return os.environ.get("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "0") == "1"


@functools.cache
def _get_cute_dsl_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _hash_source_files(paths: tuple) -> str:
    sha = hashlib.sha256()
    for path in sorted(str(p) for p in paths):
        sha.update(path.encode())
        with open(path, "rb") as f:
            sha.update(f.read())
    return sha.hexdigest()


def _get_compile_arch() -> str:
    """The CuTe-DSL compilation target, e.g. ``sm100a``.

    Mirrors the nvidia-cutlass-dsl's own resolution for
    ``CUTE_DSL_ARCH`` environment variable when set.

    If not set, detect from the current CUDA device.
    """
    arch = os.environ.get("CUTE_DSL_ARCH")
    if not arch:
        import torch

        major, minor = torch.cuda.get_device_capability()
        suffix = "a" if major >= 9 else ""
        arch = f"sm_{major}{minor}{suffix}"
    return sanitize_symbol_name(arch.replace("_", ""))


def sanitize_symbol_name(name: str) -> str:
    """Sanitize a name into a valid C symbol / filename component."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def build_and_load_cute_dsl_kernel(
    module_name: str,
    kernel_name: str,
    compile_fn: Callable[[], Any],
    extra_key_files: Sequence[str] = (),
) -> Any:
    """Compile a CuTe-DSL kernel with a persistent on-disk cache.

    On a cache hit the kernel is reloaded from the exported object file
    without recompilation; on a miss ``compile_fn`` runs, and the result is
    stored in the on-disk module directory.

    Parameters
    ----------
    module_name : str
        Op-family name (e.g. ``"nvfp4_quantize"``) for the on-disk module.
    kernel_name : str
        Specialization name of the kernel.
    compile_fn : Callable[[], Any]
        Zero-argument closure performing the
        ``cute.compile(..., options="--enable-tvm-ffi")`` call.
    extra_key_files : Sequence[str]
        Source files whose content participates in cache invalidation.
        Shared by all kernels of the module; a change wipes and lazily
        rebuilds the whole module directory.

    Returns
    -------
    A TVM-FFI callable taking the same positional arguments as the
    compiled kernel.
    """
    if cute_dsl_cache_disabled():
        return compile_fn()

    try:
        source_sha256 = _hash_source_files(tuple(extra_key_files))
    except (OSError, TypeError) as e:
        logger.warning(
            f"Cannot hash CuTe-DSL kernel sources for {module_name}/{kernel_name} "
            f"({e}); bypassing the disk cache for this kernel."
        )
        return compile_fn()

    kernel_name = sanitize_symbol_name(kernel_name)
    module_dir_name = sanitize_symbol_name(
        f"{module_name}_{_get_compile_arch()}_cute_dsl"
    )
    module_dir = jit_env.FLASHINFER_JIT_DIR / module_dir_name
    object_path = module_dir / f"{kernel_name}.o"
    symbol = f"{module_name}_{kernel_name}"
    expected_meta = {
        "module": module_dir_name,
        "arch": _get_compile_arch(),
        "cute_dsl_version": _get_cute_dsl_version(),
        "source_sha256": source_sha256,
    }

    kernel = _try_load_cached_kernel(module_dir, object_path, symbol, expected_meta)
    if kernel is not None:
        return kernel

    from .core import get_tmpdir

    with FileLock(get_tmpdir() / f"{module_dir_name}.lock", thread_local=False):
        # Another process may have built the artifact while we waited.
        kernel = _try_load_cached_kernel(module_dir, object_path, symbol, expected_meta)
        if kernel is not None:
            return kernel

        # Stale module (dsl version / source change): wipe it, like a ninja
        # rebuild of the whole module. Open .o files stay mapped (POSIX).
        # Also wipes a module with missing meta.json (crash before the meta
        # write): an orphaned .o must not be adopted by fresh metadata.
        meta_path = module_dir / _META_FILENAME
        if module_dir.exists() and _read_meta(meta_path) != expected_meta:
            logger.info(f"Invalidating stale CuTe-DSL module {module_dir_name}")
            shutil.rmtree(module_dir, ignore_errors=True)

        logger.info(f"Compiling CuTe-DSL kernel {module_dir_name}/{kernel_name}")
        compiled_kernel = compile_fn()
        try:
            _export_kernel(
                compiled_kernel, module_dir, object_path, symbol, expected_meta
            )
        except Exception as e:
            logger.warning(
                f"Failed to persist CuTe-DSL kernel {module_dir_name}/{kernel_name} "
                f"to {object_path}: {e}. The kernel will be recompiled next run."
            )
        return compiled_kernel


def _read_meta(meta_path: Path) -> Any:
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def _try_load_cached_kernel(
    module_dir: Path, object_path: Path, symbol: str, expected_meta: dict
) -> Any:
    if not object_path.exists():
        return None
    if _read_meta(module_dir / _META_FILENAME) != expected_meta:
        return None
    try:
        import cutlass.cute as cute

        module = cute.runtime.load_module(str(object_path), enable_tvm_ffi=True)
        kernel = getattr(module, symbol)
        logger.debug(f"Loaded cached CuTe-DSL kernel from {object_path}")
        return kernel
    except Exception as e:
        logger.warning(
            f"Failed to load cached CuTe-DSL kernel from {object_path}: {e}. "
            "Recompiling."
        )
        return None


def _export_kernel(
    compiled_kernel: Any,
    module_dir: Path,
    object_path: Path,
    symbol: str,
    meta: dict,
) -> None:
    module_dir.mkdir(parents=True, exist_ok=True)
    meta_path = module_dir / _META_FILENAME
    tmp_object_path = object_path.with_suffix(f".o.tmp.{os.getpid()}")
    tmp_meta_path = meta_path.with_suffix(f".json.tmp.{os.getpid()}")
    try:
        compiled_kernel.export_to_c(str(tmp_object_path), function_name=symbol)
        os.replace(tmp_object_path, object_path)
        # meta.json is the module's commit marker: written after the object
        # file is in place so a crash never leaves a loadable partial entry.
        # Kernels added to an already-committed module skip the rewrite.
        if _read_meta(meta_path) != meta:
            with open(tmp_meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp_meta_path, meta_path)
        logger.info(f"Persisted CuTe-DSL kernel to {object_path}")
    finally:
        for tmp in (tmp_object_path, tmp_meta_path):
            if tmp.exists():
                tmp.unlink()
