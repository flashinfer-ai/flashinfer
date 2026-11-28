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
import sys
import sysconfig
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Optional, Sequence, TextIO

from filelock import FileLock

from . import env as jit_env
from .core import JitSpec, MissingJITCacheError, get_tmpdir, logger
from ._cute_dsl_source import (
    _hash_source_files,
    cute_dsl_strict_cache_key as cute_dsl_strict_cache_key,
)

_META_FILENAME = "meta.json"


# CuTe DSL compiler options are process-global, even for distinct kernels.
_compile_lock = RLock()


def _reset_compile_lock() -> None:
    global _compile_lock
    _compile_lock = RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_compile_lock)


def _compile_in_process(compile_fn: Callable[[], Any]) -> Any:
    with _compile_lock:
        return compile_fn()


def cute_dsl_cache_disabled() -> bool:
    """Whether the CuTe-DSL disk cache is disabled via environment variable."""
    return os.environ.get("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "0") == "1"


@functools.cache
def _get_cute_dsl_version() -> str:
    """Fingerprint of the CuTe-DSL compiler stack.

    Covers TVM-FFI, Torch, Python's ABI, and every installed
    ``nvidia-cutlass-dsl*`` distribution, not just the
    main package: the codegen backend lives in the libs packages, and the
    CUDA-family variant is encoded in the package *name* (e.g.
    ``nvidia-cutlass-dsl-libs-cu13``), so two environments with the same DSL
    version but different libs variants must not share artifacts. The system
    CUDA toolkit does not participate: ``cute.compile`` uses the bundled stack,
    not ``CUDA_HOME``. The active backend is not fingerprinted separately: with
    multiple CUDA-family variants installed, a host driver change can switch
    the selected backend without changing this fingerprint. Persistent caches
    need a reset after such a switch.
    """
    import importlib.metadata

    stack = []
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name in {"torch", "apache-tvm-ffi"} or name.startswith("nvidia-cutlass-dsl"):
            # RECORD distinguishes rebuilt wheels with the same version;
            # direct_url carries the commit of a VCS dependency override.
            provenance = json.dumps(
                [dist.read_text("RECORD"), dist.read_text("direct_url.json")]
            ).encode()
            stack.append(
                f"{name}=={dist.version}:{hashlib.sha256(provenance).hexdigest()}"
            )
    return ";".join(
        [
            str(sys.implementation.cache_tag),
            str(sysconfig.get_config_var("SOABI")),
            *sorted(stack),
        ]
    )


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

    ``enable_tvm_ffi`` states how ``compile_fn`` compiles: True for a
    ``cute.compile(..., options="--enable-tvm-ffi")`` kernel, exported and
    reloaded as a TVM-FFI function; False for a kernel compiled without it,
    exported as a plain object file and reloaded as a CuTe-DSL compiled
    function that takes the same arguments as the in-process one.

    ``export_compile_fn``, when given, compiles the same kernel without the
    DSL's execution engine (``cute.compile(..., no_jit_engine=True)``). The
    disk-cache path exports the compiled object and reloads it, so the
    engine, its entry lookup and the TVM-FFI link of the in-process kernel
    are work it never uses; the exported object is the same. ``compile_fn``
    still runs when the export fails (an engine-less kernel cannot run) and
    when the cache is disabled.
    """

    toolchain = "cute_dsl"

    def __init__(
        self,
        module_name: str,
        kernel_name: str,
        compile_fn: Callable[[], Any],
        source_sha256: str,
        *,
        enable_tvm_ffi: bool = True,
        export_compile_fn: Optional[Callable[[], Any]] = None,
    ):
        self.module_name = module_name
        self.kernel_name = sanitize_symbol_name(kernel_name)
        self.compile_fn = compile_fn
        self.export_compile_fn = export_compile_fn
        self.enable_tvm_ffi = enable_tvm_ffi
        self.module_dir_name = sanitize_symbol_name(
            f"{module_name}_{_get_compile_arch()}_cute_dsl"
        )
        self.name = f"{self.module_dir_name}/{self.kernel_name}"
        self.module_dir = jit_env.FLASHINFER_JIT_DIR / self.module_dir_name
        self.object_path = self.module_dir / f"{self.kernel_name}.o"
        self.symbol = f"{module_name}_{self.kernel_name}"
        self.expected_meta: dict[str, Any] = {
            "arch": _get_compile_arch(),
            "cute_dsl_version": _get_cute_dsl_version(),
            "source_sha256": source_sha256,
        }
        if not enable_tvm_ffi:
            # The two export formats must never share a module directory.
            self.expected_meta["tvm_ffi"] = False
        # Retained by build() as a fallback when the artifact cannot be loaded
        # from the disk cache because persistence failed or it was invalidated.
        self._compiled_kernel: Optional[Any] = None
        self._exported_without_engine = False

    @property
    def lock_path(self) -> Path:
        # Keep locks outside module_dir: it can be replaced on invalidation,
        # and creating it before meta.json would make it appear stale.
        return get_tmpdir() / f"{self.module_dir_name}.{self.kernel_name}.lock"

    @property
    def module_lock_path(self) -> Path:
        # Module metadata and exports are shared by kernels of one op family.
        return get_tmpdir() / f"{self.module_dir_name}.lock"

    @property
    def meta_path(self) -> Path:
        return self.module_dir / _META_FILENAME

    def _same_cache_generation(self, marker: TextIO) -> Optional[bool]:
        """False if the marker was replaced or removed, None if it cannot be checked."""
        try:
            return os.path.samestat(os.fstat(marker.fileno()), self.meta_path.stat())
        except (FileNotFoundError, NotADirectoryError):
            return False
        except OSError:
            return None

    def _cached_entry(self, *, load: bool) -> Any:
        # meta.json is the generation's commit marker, published after the
        # object. Keep its inode alive across the lookup to detect replacement
        # without writing a lock file on cache hits.
        try:
            marker = self.meta_path.open()
        except OSError:
            return None
        with marker:
            try:
                if (
                    json.load(marker) != self.expected_meta
                    or not self.object_path.is_file()
                ):
                    return None
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return None

            try:
                value = self._load_from_disk() if load else True
            except Exception:
                # A sibling can invalidate the object after the existence check.
                # Preserve genuine loader errors for the same (or unknown) generation.
                if self._same_cache_generation(marker) is False:
                    return None
                raise
            return value if self._same_cache_generation(marker) else None

    @property
    def is_compiled(self) -> bool:
        return self._cached_entry(load=False) is True

    def get_library_path(self) -> Path:
        return self.object_path

    def try_load(self) -> Optional[Any]:
        """The cached kernel iff its object and metadata belong to the same generation."""
        try:
            return self._cached_entry(load=True)
        except Exception as e:
            logger.warning(
                f"Failed to load cached CuTe-DSL kernel {self.object_path}: {e}. Recompiling."
            )
            return None

    def build(self) -> None:
        """Compile via ``cute.compile`` and export the ``.o`` artifact.

        The caller holds the kernel lock. Compilation does not take the module
        lock; only invalidation and export serialize with sibling kernels.
        Persistence failures degrade gracefully: the in-process kernel is
        kept for load(); only the disk write is lost.
        Stale modules (dsl version / source change) are wiped like a ninja rebuild.
        """
        logger.info(f"Compiling CuTe-DSL kernel {self.name}")
        self._exported_without_engine = False
        self._compiled_kernel = _compile_in_process(
            self.export_compile_fn or self.compile_fn
        )
        exported = False
        with FileLock(self.module_lock_path, thread_local=False):
            try:
                if (
                    self.module_dir.exists()
                    and _read_meta(self.meta_path) != self.expected_meta
                ):
                    logger.info(
                        f"Invalidating stale CuTe-DSL module {self.module_dir_name}"
                    )
                    # Retire the generation before removing any of its objects.
                    self.meta_path.unlink(missing_ok=True)
                    shutil.rmtree(self.module_dir)
                self._export()
                exported = True
            except Exception as e:  # noqa: BLE001 -- persistence is best-effort
                logger.warning(
                    f"Failed to persist CuTe-DSL kernel {self.name} to "
                    f"{self.object_path}: {e}. The kernel will be recompiled next run."
                )

        if self.export_compile_fn is not None:
            self._exported_without_engine = exported
            # An engine-less kernel can only run after a successful export and reload.
            self._compiled_kernel = (
                None if exported else _compile_in_process(self.compile_fn)
            )

    def compile_and_persist(self, *, force: bool = True) -> None:
        """Build and publish the artifact. Meant for parallel precompilation workers.

        The default forces a rebuild. With force=False, workers filling a cache
        share a published specialization, including one built while waiting.
        Persistence is best-effort; load() can return an in-process fallback.
        """
        if not force and self.is_compiled:
            return
        with FileLock(self.lock_path, thread_local=False):
            if not force and self.is_compiled:
                return
            self._timed_build()

    def load(self) -> Any:
        """The kernel to hand back: the exported artifact whenever one exists.

        The artifact is preferred even immediately after ``build()``.
        ``cute.compile`` does not reject a ``--gpu-arch`` the current device
        cannot run: it returns a callable with no execution engine, which
        compiles cleanly and then raises ``DSLRuntimeError`` on first
        invocation (measured on sm_100a for sm_90a, sm_103a, sm_120a and
        sm_80).  Reloading through ``load_module`` instead either yields a
        properly loaded module or fails here, at build time, rather than
        deferring the failure to the first call in a user's hot path.

        This also removes a divergence: a cache *hit* already returned the
        reloaded artifact, so every second run was exercising that form
        anyway.  Miss and hit now hand back the same thing.

        Falls back to the in-process object when no valid artifact exists -- export
        is best-effort (``build()`` logs and continues on failure) and the
        on-disk cache can be disabled outright.
        """
        kernel = self._cached_entry(load=True)
        if kernel is not None:
            return kernel
        if self._compiled_kernel is None and self._exported_without_engine:
            if os.environ.get("FLASHINFER_DISABLE_JIT"):
                raise MissingJITCacheError(
                    f"JIT compilation is disabled and the export for {self.name} was invalidated",
                    spec=self,
                )
            self._compiled_kernel = _compile_in_process(self.compile_fn)
            self._exported_without_engine = False
        if self._compiled_kernel is not None:
            return self._compiled_kernel
        raise RuntimeError(f"No valid cached CuTe-DSL kernel for {self.name}")

    def _load_from_disk(self) -> Any:
        import cutlass.cute as cute

        module = cute.runtime.load_module(
            str(self.object_path), enable_tvm_ffi=self.enable_tvm_ffi
        )
        return getattr(module, self.symbol)

    def _export_object(self, object_path: Path) -> None:
        """Write the compiled kernel's object file to ``object_path``."""
        if self.enable_tvm_ffi:
            self._compiled_kernel.export_to_c(
                str(object_path), function_name=self.symbol
            )
            return
        # Without TVM-FFI only the object file is written: export_to_c would
        # also generate a C header, which needs every argument annotated.
        object_path.write_bytes(self._compiled_kernel.dump_to_object(self.symbol))

    def _export(self) -> None:
        self.module_dir.mkdir(parents=True, exist_ok=True)
        tmp_object_path = self.object_path.with_suffix(f".o.tmp.{os.getpid()}")
        tmp_meta_path = self.meta_path.with_suffix(f".json.tmp.{os.getpid()}")
        try:
            self._export_object(tmp_object_path)
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
    *,
    enable_tvm_ffi: bool = True,
    export_compile_fn: Optional[Callable[[], Any]] = None,
    source_sha256: Optional[str] = None,
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
        Zero-argument closure performing the ``cute.compile`` call, with
        ``options="--enable-tvm-ffi"`` unless ``enable_tvm_ffi`` is False.
    extra_key_files : Sequence[str]
        Source files whose content participates in cache invalidation.
        Shared by all kernels of the module; a change wipes and lazily
        rebuilds the whole module directory.
    enable_tvm_ffi : bool
        Whether ``compile_fn`` compiles with TVM-FFI (see
        :class:`JitSpecCuteDsl`).
    export_compile_fn : Optional[Callable[[], Any]]
        ``compile_fn`` without the DSL's execution engine, for the exported
        object (see :class:`JitSpecCuteDsl`).
    source_sha256 : Optional[str]
        The hash of ``extra_key_files`` when the caller keeps it across the
        kernels of a module (``_hash_source_files(extra_key_files)``);
        computed here when None.

    Returns
    -------
    A callable taking the same positional arguments as the compiled kernel:
    a TVM-FFI function, or with ``enable_tvm_ffi=False`` a CuTe-DSL compiled
    function.
    """
    if cute_dsl_cache_disabled():
        return _compile_in_process(compile_fn)

    # If any key source is unreadable (missing file, __file__ unavailable
    # under exotic packaging), bypass the disk cache rather than skipping the
    # file from the hash: a weakened key could serve stale artifacts.
    if source_sha256 is None:
        try:
            source_sha256 = _hash_source_files(tuple(extra_key_files))
        except (OSError, TypeError, SyntaxError, ValueError) as e:
            logger.warning(
                f"Cannot hash CuTe-DSL kernel sources for {module_name}/{kernel_name} "
                f"({e}); bypassing the disk cache for this kernel."
            )
            return _compile_in_process(compile_fn)

    spec = JitSpecCuteDsl(
        module_name,
        kernel_name,
        compile_fn,
        source_sha256,
        enable_tvm_ffi=enable_tvm_ffi,
        export_compile_fn=export_compile_fn,
    )
    return spec.build_and_load()
