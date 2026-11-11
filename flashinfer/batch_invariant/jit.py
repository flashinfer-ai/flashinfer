# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compiled kernels of the batch-invariant classes in FlashInfer's CuTe-DSL disk cache.

:func:`build_and_load` is :func:`flashinfer.jit.cute_dsl_core.build_and_load_cute_dsl_kernel` with the module's **source
closure** as the key files: the transitive closure of relative imports (module- and function-level) from the module's
root files (:data:`MODULE_ROOTS`) within the mode's directories (:data:`MODE_DIRS`) plus the ``__init__`` of every package on
the way. A change to any file a kernel's
generated code depends on therefore invalidates the module's cached objects, while the API module stays outside the
closure and can change freely. Sources the walk reaches that are host-side selection code, not kernel code (the dispatcher's
chunk rule and ranking heuristic: they choose a kernel, they never enter one), are excluded per module (:data:`MODULE_NON_KERNEL_SOURCES`)
so an edit to them keeps every compiled object valid. Everything else (module directory, ``meta.json``, locking,
``FLASHINFER_DISABLE_JIT``, ``FLASHINFER_CUTE_DSL_DISABLE_CACHE``) is FlashInfer's.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel, sanitize_symbol_name

__all__ = [
    "build_and_load",
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
    # the bf16 MoE kernels: the grouped GEMM (which composes the dense GEMM blocks: they are in its closure), the
    # routing, permutation and combine kernels; the MoE API, dispatcher and runner stay outside
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
    "batch_invariant_gemm_fp8_blockscale": (
        "cute_dsl/attention/fmha/helpers/static_persistent_tile_scheduler.py",
    ),
}

#: Sources within the mode's directories that the closure walk reaches but that no kernel binary depends on (host-side
#: selection code: the dispatcher's chunk rule and ranking heuristic), relative to the flashinfer package; removed from the key after
#: the walk, so an edit to them (or recalibrated constants) does not recompile the module's kernels.
MODULE_NON_KERNEL_SOURCES: Dict[str, Tuple[str, ...]] = {
    "batch_invariant_gemm": (
        "gemm/batch_invariant/schedule.py",
        "gemm/batch_invariant/heuristic.py",
    ),
    "batch_invariant_gemm_blockscaled": (
        "gemm/batch_invariant/schedule.py",
        "gemm/batch_invariant/heuristic.py",
    ),
    # the grouped kernels compose the dense GEMM blocks, whose closure reaches the GEMM dispatcher files
    "batch_invariant_moe": (
        "gemm/batch_invariant/schedule.py",
        "gemm/batch_invariant/heuristic.py",
    ),
    "batch_invariant_gemm_fp8_blockscale": (
        "gemm/batch_invariant/schedule.py",
        "gemm/batch_invariant/heuristic.py",
    ),
}


def _in_mode(path: Path) -> bool:
    return any(path == d or d in path.parents for d in MODE_DIRS)


def _relative_imports(path: Path) -> set:
    """Files named by the relative imports of ``path`` (module- and function-level)."""
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
        if parts:
            cands += [target.with_suffix(".py"), target / "__init__.py"]
        cands += [target / f"{a.name}.py" for a in node.names]
        cands += [target / a.name / "__init__.py" for a in node.names]
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
    for r in MODULE_NON_KERNEL_SOURCES.get(module, ()):
        p = (_FI_DIR / r).resolve()
        if p not in seen:
            # a listed exclusion that is not in the closure is a stale entry (the file moved, or stopped being imported):
            # silently ignoring it would let a renamed dispatcher file back into the cache key
            raise FileNotFoundError(
                f"{module}: non-kernel source {r} is not in the module's import closure; update MODULE_NON_KERNEL_SOURCES"
            )
        seen.discard(p)
    return tuple(sorted(seen))


def relative_sources(module: str) -> list:
    return [str(p.relative_to(_FI_DIR)) for p in module_sources(module)]


def build_and_load(module: str, kernel: str, compile_fn: Callable[[], Any]) -> Any:
    """Return a TVM-FFI callable for ``kernel`` of ``module``, compiling only on a cache miss."""
    return build_and_load_cute_dsl_kernel(
        module,
        kernel,
        compile_fn,
        extra_key_files=[str(p) for p in module_sources(module)],
    )
