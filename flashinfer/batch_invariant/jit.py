# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compiled kernels of the batch-invariant classes in FlashInfer's CuTe-DSL disk cache.

:func:`build_and_load` is :func:`flashinfer.jit.cute_dsl_core.build_and_load_cute_dsl_kernel` with the module's **source
closure** as the key files: the transitive closure of relative imports (module- and function-level) from the module's
root files (:data:`MODULE_ROOTS`) within the mode's directories (:data:`MODE_DIRS`) plus the ``__init__`` of every package on
the way. A change to any file a kernel's
generated code depends on therefore invalidates the module's cached objects, while the API module, the dispatcher (chunk
rule, ranking heuristic: they choose a kernel, they never enter one) and the other surfaces of the mode stay outside the
closure and can change freely: an import resolves to the file it names, and a package ``__init__`` enters the key without
its own imports being followed (:func:`_relative_imports`). Everything else (module directory, ``meta.json``, locking,
``FLASHINFER_DISABLE_JIT``, ``FLASHINFER_CUTE_DSL_DISABLE_CACHE``) is FlashInfer's.
"""

from __future__ import annotations

import ast
import functools
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..jit._cute_dsl_source import _hash_source_files
from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel, sanitize_symbol_name

__all__ = [
    "build_and_load",
    "module_source_sha256",
    "precompile",
    "module_sources",
    "relative_sources",
    "sanitize_symbol_name",
]

_FI_DIR = Path(__file__).resolve().parents[1]  # the flashinfer package
#: The directories of the mode: the engine here and the per-operator classes next to their FlashInfer operators.
MODE_DIRS: Tuple[Path, ...] = (
    _FI_DIR / "batch_invariant",
    _FI_DIR / "gemm" / "batch_invariant",
    _FI_DIR / "attention" / "batch_invariant",
    _FI_DIR / "fused_moe" / "batch_invariant",
)

#: Root files of every cache module, relative to the flashinfer package; the module's closure is the transitive
#: relative-import closure of these files within the mode's directories.
MODULE_ROOTS: Dict[str, Tuple[str, ...]] = {
    # the GEMM kernels (both operand formats) and the split-K fixup share one module: both roots, so an edit to either
    # invalidates its objects
    "batch_invariant_gemm": (
        "gemm/batch_invariant/blocks/compile.py",
        "gemm/batch_invariant/fixup.py",
    ),
    # the block-scaled kernels (the MXFP8 classes): their own module, so an edit to the block-scaled body leaves the dense
    # module's objects valid and vice versa (the split's fixup stays in the dense module)
    "batch_invariant_gemm_blockscaled": (
        "gemm/batch_invariant/blocks/compile_blockscaled.py",
    ),
    # the mixed MX classes (e2m1 weights x e4m3 activations) on the same body: their own module, so an edit to the mixed
    # plans leaves the one-format block-scaled module's objects valid (an edit to the shared body moves both)
    "batch_invariant_gemm_blockscaled_mixed": (
        "gemm/batch_invariant/blocks/compile_blockscaled_mixed.py",
    ),
    # the block-scale fp8 kernels (the DeepSeek-style class behind gemm_fp8_nt_groupwise): their own module
    "batch_invariant_gemm_fp8_blockscale": (
        "gemm/batch_invariant/blocks/compile_fp8bs.py",
    ),
    # the attention kernels (the row body, the Swaps-AB decode body) and the split-KV fixup share one module
    "batch_invariant_attention": (
        "attention/batch_invariant/blocks/compile.py",
        "attention/batch_invariant/swaps/compile.py",
        "attention/batch_invariant/fixup.py",
    ),
    # the MoE kernels of the bf16 and the fp8 per-tensor classes: the grouped and the swapped GEMM bodies (one source for
    # both operand formats, composing the dense GEMM blocks: they are in its closure), the routing, permutation and
    # combine kernels; the MoE API, dispatchers and runners stay outside
    "batch_invariant_moe": (
        "fused_moe/batch_invariant/blocks/compile.py",
        "fused_moe/batch_invariant/route.py",
        "fused_moe/batch_invariant/route_plan.py",
        "fused_moe/batch_invariant/permute.py",
        "fused_moe/batch_invariant/combine.py",
    ),
}

#: Sources outside the mode's directories that a module's kernels import absolutely, relative to the flashinfer
#: package: the relative-import closure cannot reach them, so they are keyed explicitly (a missing file is an error).
MODULE_EXTRA_SOURCES: Dict[str, Tuple[str, ...]] = {
    "batch_invariant_gemm": (
        "cute_dsl/attention/fmha/helpers/static_persistent_tile_scheduler.py",
    ),
    "batch_invariant_gemm_blockscaled": (
        "cute_dsl/attention/fmha/helpers/static_persistent_tile_scheduler.py",
    ),
    "batch_invariant_gemm_blockscaled_mixed": (
        "cute_dsl/attention/fmha/helpers/static_persistent_tile_scheduler.py",
    ),
    "batch_invariant_gemm_fp8_blockscale": (
        "cute_dsl/attention/fmha/helpers/static_persistent_tile_scheduler.py",
    ),
}


def _in_mode(path: Path) -> bool:
    return any(path == d or d in path.parents for d in MODE_DIRS)


def _relative_imports(path: Path) -> set:
    """Files named by the relative imports of ``path`` (module- and function-level).

    ``from .m import a`` names the module ``m.py``; from a package ``p``, ``from .p import a`` names the submodule
    ``p/a.py`` (or ``p/a/__init__.py``) when it exists and the package's ``__init__.py`` only when ``a`` is a symbol
    defined there. An ``__init__`` named by an import is followed in full (:func:`module_sources` walks it like any
    key file); one that is only on the path to a key file enters the key unfollowed, so the surfaces it re-exports
    are not part of a kernel the package's submodules build.
    """
    out: set = set()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.level):
            continue
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        parts = node.module.split(".") if node.module else []
        target = base.joinpath(*parts) if parts else base
        cands = []
        if parts and not (target / "__init__.py").exists():
            cands.append(target.with_suffix(".py"))
        else:
            for a in node.names:
                sub, sub_pkg = target / f"{a.name}.py", target / a.name / "__init__.py"
                if sub.exists():
                    cands.append(sub)
                elif sub_pkg.exists():
                    cands.append(sub_pkg)
                else:
                    cands.append(target / "__init__.py")
        for c in cands:
            if c.exists():
                r = c.resolve()
                if _in_mode(r):
                    out.add(r)
    return out


@functools.cache
def module_sources(module: str) -> Tuple[Path, ...]:
    """The source files a module's compiled objects depend on (see the module docstring)."""
    roots = MODULE_ROOTS.get(module)
    if roots is None:
        return tuple(sorted(p for d in MODE_DIRS for p in d.rglob("*.py")))
    seen: set = set()
    stack = [(_FI_DIR / r).resolve() for r in roots]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(_relative_imports(p) - seen)
    for p in list(seen):
        d = p.parent
        while True:
            init = d / "__init__.py"
            if init.exists():
                seen.add(init.resolve())
            if d in MODE_DIRS or not _in_mode(d):
                break
            d = d.parent
    for r in MODULE_EXTRA_SOURCES.get(module, ()):
        p = (_FI_DIR / r).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{module}: cache key source {r} is missing")
        seen.add(p)
    return tuple(sorted(seen))


def relative_sources(module: str) -> list:
    return [str(p.relative_to(_FI_DIR)) for p in module_sources(module)]


@functools.cache
def module_source_sha256(module: str) -> str:
    """The hash of the module's key files (:func:`module_sources`), computed once per process: every kernel of the module
    shares it, and an edit to a key file is seen by the next process."""
    return _hash_source_files(tuple(str(p) for p in module_sources(module)))


def _export_compile_fn(
    compile_fn: Callable[..., Any],
) -> Optional[Callable[[], Any]]:
    """``compile_fn`` with the DSL's ``no_jit_engine`` option when it forwards keyword options to ``cute.compile``
    (``def compile_fn(**compile_options)``), else None. The disk-cache path exports the compiled object and reloads it:
    the in-process execution engine, its entry lookup and the TVM-FFI link are work such a kernel never uses, and the
    exported object is the same without them."""
    try:
        params = inspect.signature(compile_fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return functools.partial(compile_fn, no_jit_engine=True)
    return None


def build_and_load(module: str, kernel: str, compile_fn: Callable[..., Any]) -> Any:
    """Return a TVM-FFI callable for ``kernel`` of ``module``, compiling only on a cache miss (under the op family's module
    lock: concurrent processes missing the same kernel compile it once). A ``compile_fn`` that forwards
    ``**compile_options`` to ``cute.compile`` is compiled without the execution engine for the exported object
    (:func:`_export_compile_fn`); one without them keeps the engine."""
    try:
        source_sha256: Optional[str] = module_source_sha256(module)
    except (OSError, TypeError, SyntaxError, ValueError):
        # FlashInfer's path hashes again, warns and bypasses the disk cache
        source_sha256 = None
    return build_and_load_cute_dsl_kernel(
        module,
        kernel,
        compile_fn,
        extra_key_files=[str(p) for p in module_sources(module)],
        source_sha256=source_sha256,
        export_compile_fn=_export_compile_fn(compile_fn),
    )


def precompile(module: str, kernel: str, compile_fn: Callable[..., Any]) -> None:
    """Compile ``kernel`` of ``module`` into the on-disk cache without loading it, OUTSIDE the module lock
    (``JitSpecCuteDsl.compile_and_persist``: the lock is taken only for the stale-module wipe and the artifact export, the
    path the GEMM and MoE precompilers use). The entry for precompile pools: ``build_and_load`` holds the lock for the whole
    build, which makes a pool of workers compiling DIFFERENT kernels of one family serial -- 579 truncated-sampler kernels on
    8 workers measured 1565 s through it against 259 s through this path (2026-11-18). The compile is the one
    ``build_and_load`` would run (the module's cached source hash, the engine-less export for a ``compile_fn`` that forwards
    ``**compile_options``: :func:`_export_compile_fn`). A kernel already in the cache is left alone; with the disk cache
    disabled (``FLASHINFER_CUTE_DSL_DISABLE_CACHE=1``) there is nothing to persist and nothing is compiled."""
    from ..jit.cute_dsl_core import JitSpecCuteDsl, cute_dsl_cache_disabled

    if cute_dsl_cache_disabled():
        return  # nothing to persist: with the disk cache disabled the serving path compiles in-process
    spec = JitSpecCuteDsl(
        module,
        kernel,
        compile_fn,
        module_source_sha256(module),
        export_compile_fn=_export_compile_fn(compile_fn),
    )
    if not spec.is_compiled:
        spec.compile_and_persist()
