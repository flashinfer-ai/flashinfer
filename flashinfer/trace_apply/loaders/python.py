# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python-family solution loader (python / triton / cutedsl / tilelang).

All four are loaded the same way: materialize the solution's sources to a
per-solution cache dir, import the entry-point file as a module, and resolve the
entry-point function. They differ only in which libraries the source imports
(triton, nvidia-cutlass-dsl, …), not in how they are loaded.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Callable

from flashinfer.trace.solution import Solution


def _cache_base() -> Path:
    """Writable cache root for materialized solution sources.

    No custom env knob: ``~/.cache/flashinfer/trace_apply/solutions`` normally,
    falling back to the system temp dir only when ``$HOME`` is unresolvable
    or the home directory is read-only/not writable.
    """
    try:
        path = Path.home() / ".cache" / "flashinfer" / "trace_apply" / "solutions"
        path.mkdir(parents=True, exist_ok=True)
        return path
    except (RuntimeError, OSError):
        return Path(tempfile.gettempdir()) / "flashinfer" / "trace_apply" / "solutions"


def _cache_dir(solution: Solution) -> Path:
    return _cache_base() / solution.hash()[:16]


def _safe_relpath(out: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``out``, rejecting absolute paths and ``..`` escapes
    (path-traversal / arbitrary-write guard for untrusted solution sources).
    """
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe solution source path: {rel!r}")
    target = (out / p).resolve()
    if not target.is_relative_to(out.resolve()):
        raise ValueError(f"Solution source path escapes cache dir: {rel!r}")
    return target


def materialize(solution: Solution) -> Path:
    """Write the solution's sources to its cache dir (idempotent + atomic)."""
    out = _cache_dir(solution)
    out.mkdir(parents=True, exist_ok=True)
    for src in solution.sources:
        target = _safe_relpath(out, src.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Only (re)write when content differs, via a temp file + os.replace so
        # concurrent readers never observe a partial write.
        if not target.exists() or target.read_text(encoding="utf-8") != src.content:
            fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(src.content)
                os.replace(tmp, target)
            except BaseException:
                with contextlib.suppress(FileNotFoundError, OSError):
                    os.unlink(tmp)
                raise
    return out


def _import_file(module_name: str, path: Path) -> ModuleType:
    """Import ``path`` as ``module_name``, dropping it from sys.modules on failure."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Don't leave a partially-initialized module behind on import failure.
        sys.modules.pop(module_name, None)
        raise
    return module


def load(solution: Solution) -> Callable:
    """Materialize, import the entry-point file, and return its function.

    A Solution that declares ``setup_entry_point`` returns its function bound to
    the setup (and teardown) imported from the same materialized directory, a
    :class:`~flashinfer.trace_apply.setup.SetupBoundCallable`, so this path never
    serves a setup-bearing Solution without its setup.
    """
    sol_dir = materialize(solution)
    module_name = f"flashinfer_trace_apply_solution_{solution.hash()[:16]}"
    modules: dict[str, ModuleType] = {}

    def symbol(file_part: str, func_part: str, role: str) -> Callable:
        """Import ``file_part`` once per role; return its ``func_part`` attribute."""
        module = modules.get(file_part)
        if module is None:
            path = _safe_relpath(sol_dir, file_part)
            if not path.is_file():
                raise FileNotFoundError(f"Solution {role} file not found: {path}")
            suffix = "" if file_part == solution.entry_file() else f"_{role}"
            module = modules[file_part] = _import_file(module_name + suffix, path)
        if not hasattr(module, func_part):
            raise AttributeError(
                f"Solution {solution.name!r} {role} missing function {func_part!r}"
            )
        return getattr(module, func_part)

    run = symbol(solution.entry_file(), solution.entry_symbol(), "entry-point")
    if solution.spec.setup_entry_point is None:
        return run
    from flashinfer.trace_apply.setup import SetupBoundCallable  # noqa: PLC0415

    setup = symbol(solution.setup_file(), solution.setup_symbol(), "setup")
    teardown = (
        symbol(solution.teardown_file(), solution.teardown_symbol(), "teardown")
        if solution.spec.teardown_entry_point is not None
        else None
    )
    return SetupBoundCallable(
        run,
        setup,
        teardown,
        name=solution.definition,
        setup_inputs=tuple(solution.spec.setup_inputs),
        state_param=solution.spec.setup_state_param,
    )


__all__ = ["load", "materialize"]
