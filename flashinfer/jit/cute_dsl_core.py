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
from typing import Any, Callable, Optional, Sequence

from . import env as jit_env
from .core import JitSpec, get_tmpdir, logger

_META_FILENAME = "meta.json"


def cute_dsl_cache_disabled() -> bool:
    """Whether the CuTe-DSL disk cache is disabled via environment variable."""
    return os.environ.get("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "0") == "1"


@functools.cache
def _get_cute_dsl_version() -> str:
    """Fingerprint of the CuTe-DSL compiler stack.

    Covers every installed ``nvidia-cutlass-dsl*`` distribution, not just the
    main package: the codegen backend lives in the libs packages, and the
    CUDA-family variant is encoded in the package *name* (e.g.
    ``nvidia-cutlass-dsl-libs-cu13``), so two environments with the same DSL
    version but different libs variants must not share artifacts. The system
    CUDA toolkit/driver deliberately does not participate: ``cute.compile``
    uses this bundled stack, not ``CUDA_HOME``.
    """
    import importlib.metadata

    stack = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if (dist.metadata["Name"] or "").lower().startswith("nvidia-cutlass-dsl")
    )
    return ";".join(stack) if stack else "unknown"


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


class JitSpecCuteDsl(JitSpec):
    """JitSpec for one CuTe-DSL kernel specialization.

    One instance represents a single compiled kernel (nvcc specs represent a
    whole module); instances of the same op family share a module directory
    and its module-level ``meta.json``.
    """

    def __init__(
        self,
        module_name: str,
        kernel_name: str,
        compile_fn: Callable[[], Any],
        source_sha256: str,
    ):
        self.module_name = module_name
        self.kernel_name = sanitize_symbol_name(kernel_name)
        self.compile_fn = compile_fn
        self.module_dir_name = sanitize_symbol_name(
            f"{module_name}_{_get_compile_arch()}_cute_dsl"
        )
        self.name = f"{self.module_dir_name}/{self.kernel_name}"
        self.module_dir = jit_env.FLASHINFER_JIT_DIR / self.module_dir_name
        self.object_path = self.module_dir / f"{self.kernel_name}.o"
        self.symbol = f"{module_name}_{self.kernel_name}"
        self.expected_meta = {
            "arch": _get_compile_arch(),
            "cute_dsl_version": _get_cute_dsl_version(),
            "source_sha256": source_sha256,
        }
        # Set by build(): the freshly compiled in-process kernel, returned by
        # load() so a build is never followed by a redundant JITLink reload.
        self._compiled_kernel: Optional[Any] = None

    @property
    def lock_path(self) -> Path:
        # Module-level lock: kernels of one op family serialize their builds
        # (meta.json wipes/writes must not race sibling kernel exports).
        return get_tmpdir() / f"{self.module_dir_name}.lock"

    @property
    def meta_path(self) -> Path:
        return self.module_dir / _META_FILENAME

    @property
    def is_compiled(self) -> bool:
        return (
            self.object_path.exists()
            and _read_meta(self.meta_path) == self.expected_meta
        )

    def get_library_path(self) -> Path:
        return self.object_path

    def try_load(self) -> Optional[Any]:
        """The cached kernel iff its ``.o`` exists and meta.json matches."""
        if not self.object_path.exists():
            return None
        if _read_meta(self.meta_path) != self.expected_meta:
            return None
        try:
            kernel = self._load_from_disk()
            logger.debug(f"Loaded cached CuTe-DSL kernel from {self.object_path}")
            return kernel
        except Exception as e:
            logger.warning(
                f"Failed to load cached CuTe-DSL kernel from {self.object_path}: "
                f"{e}. Recompiling."
            )
            return None

    def build(self) -> None:
        """Compile via ``cute.compile`` and export the ``.o`` artifact.

        Runs under ``lock_path`` when invoked via ``build_and_load()``.
        Persistence failures degrade gracefully: the in-process kernel is
        kept for load(); only the disk write is lost.
        Stale modules (dsl version / source change) are wiped like a ninja rebuild.
        """
        if (
            self.module_dir.exists()
            and _read_meta(self.meta_path) != self.expected_meta
        ):
            logger.info(f"Invalidating stale CuTe-DSL module {self.module_dir_name}")
            shutil.rmtree(self.module_dir, ignore_errors=True)

        logger.info(f"Compiling CuTe-DSL kernel {self.name}")
        self._compiled_kernel = self.compile_fn()
        try:
            self._export()
        except Exception as e:
            logger.warning(
                f"Failed to persist CuTe-DSL kernel {self.name} to "
                f"{self.object_path}: {e}. The kernel will be recompiled next run."
            )

    def compile_and_persist(self) -> None:
        """Compile outside the module lock; commit the artifact under it.

        ``build_and_load()`` holds the module lock for the whole build.
        This variant runs ``compile_fn`` unlocked and takes the lock only
        for the stale-module wipe and the artifact export (~ms).

        Meant for parallel precompilation workers.
        """
        from filelock import FileLock

        self._compiled_kernel = self.compile_fn()
        with FileLock(self.lock_path, thread_local=False):
            if (
                self.module_dir.exists()
                and _read_meta(self.meta_path) != self.expected_meta
            ):
                logger.info(
                    f"Invalidating stale CuTe-DSL module {self.module_dir_name}"
                )
                shutil.rmtree(self.module_dir, ignore_errors=True)
            try:
                self._export()
            except Exception as e:  # noqa: BLE001 -- persistence is best-effort
                logger.warning(
                    f"Failed to persist CuTe-DSL kernel {self.name} to "
                    f"{self.object_path}: {e}. The kernel will be recompiled "
                    f"next run."
                )

    def load(self) -> Any:
        """The kernel compiled by build(), or the on-disk artifact."""
        if self._compiled_kernel is not None:
            return self._compiled_kernel
        return self._load_from_disk()

    def _load_from_disk(self) -> Any:
        import cutlass.cute as cute

        module = cute.runtime.load_module(str(self.object_path), enable_tvm_ffi=True)
        return getattr(module, self.symbol)

    def _export(self) -> None:
        self.module_dir.mkdir(parents=True, exist_ok=True)
        tmp_object_path = self.object_path.with_suffix(f".o.tmp.{os.getpid()}")
        tmp_meta_path = self.meta_path.with_suffix(f".json.tmp.{os.getpid()}")
        try:
            self._compiled_kernel.export_to_c(
                str(tmp_object_path), function_name=self.symbol
            )
            os.replace(tmp_object_path, self.object_path)
            # meta.json is the module's commit marker: written after the
            # object file is in place so a crash never leaves a loadable
            # partial entry. Kernels added to an already-committed module
            # skip the rewrite.
            if _read_meta(self.meta_path) != self.expected_meta:
                with open(tmp_meta_path, "w") as f:
                    json.dump(self.expected_meta, f, indent=2)
                os.replace(tmp_meta_path, self.meta_path)
            logger.info(f"Persisted CuTe-DSL kernel to {self.object_path}")
        finally:
            for tmp in (tmp_object_path, tmp_meta_path):
                if tmp.exists():
                    tmp.unlink()


def _read_meta(meta_path: Path) -> Any:
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def build_and_load_cute_dsl_kernel(
    module_name: str,
    kernel_name: str,
    compile_fn: Callable[[], Any],
    extra_key_files: Sequence[str] = (),
) -> Any:
    """Compile a CuTe-DSL kernel with a persistent on-disk cache.

    Convenience wrapper constructing a :class:`JitSpecCuteDsl` and running
    its ``build_and_load()``. On a cache hit the kernel is reloaded from the
    exported object file without recompilation; on a miss ``compile_fn``
    runs, and the result is stored in the on-disk module directory.

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

    # If any key source is unreadable (missing file, __file__ unavailable
    # under exotic packaging), bypass the disk cache rather than skipping the
    # file from the hash: a weakened key could serve stale artifacts.
    try:
        source_sha256 = _hash_source_files(tuple(extra_key_files))
    except (OSError, TypeError) as e:
        logger.warning(
            f"Cannot hash CuTe-DSL kernel sources for {module_name}/{kernel_name} "
            f"({e}); bypassing the disk cache for this kernel."
        )
        return compile_fn()

    spec = JitSpecCuteDsl(module_name, kernel_name, compile_fn, source_sha256)
    return spec.build_and_load()
