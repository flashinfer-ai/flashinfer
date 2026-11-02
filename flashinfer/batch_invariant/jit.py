# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compiled kernels of the batch-invariant classes in FlashInfer's CuTe-DSL disk cache.

:func:`build_and_load` is :func:`flashinfer.jit.cute_dsl_core.build_and_load_cute_dsl_kernel` with the module's **source
closure** as the key files: the transitive closure of relative imports (module- and function-level) from the module's
root files (:data:`MODULE_ROOTS`) plus the ``__init__`` of every package on the way. A change to any file a kernel's
generated code depends on therefore invalidates the module's cached objects, while the API module stays outside the
closure and can change freely. Everything else (module directory, ``meta.json``, locking, ``FLASHINFER_DISABLE_JIT``,
``FLASHINFER_CUTE_DSL_DISABLE_CACHE``) is FlashInfer's.
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

_PKG_DIR = Path(__file__).resolve().parent

#: Root files of every cache module; the module's closure is the transitive relative-import closure of these files.
MODULE_ROOTS: Dict[str, Tuple[str, ...]] = {
    "batch_invariant_gemm_bf16": ("gemm/blocks/compile.py",),
}


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
                if r == _PKG_DIR or _PKG_DIR in r.parents:
                    out.add(r)
    return out


@functools.cache
def module_sources(module: str) -> Tuple[Path, ...]:
    """The source files a module's compiled objects depend on (see the module docstring)."""
    roots = MODULE_ROOTS.get(module)
    if roots is None:
        return tuple(sorted(_PKG_DIR.rglob("*.py")))
    seen: set = set()
    stack = [(_PKG_DIR / r).resolve() for r in roots]
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
            if d == _PKG_DIR or _PKG_DIR not in d.parents:
                break
            d = d.parent
    return tuple(sorted(seen))


def relative_sources(module: str) -> list:
    return [str(p.relative_to(_PKG_DIR)) for p in module_sources(module)]


def build_and_load(module: str, kernel: str, compile_fn: Callable[[], Any]) -> Any:
    """Return a TVM-FFI callable for ``kernel`` of ``module``, compiling only on a cache miss."""
    return build_and_load_cute_dsl_kernel(
        module,
        kernel,
        compile_fn,
        extra_key_files=[str(p) for p in module_sources(module)],
    )
