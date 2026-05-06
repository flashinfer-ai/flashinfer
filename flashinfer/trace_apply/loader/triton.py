from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable

from flashinfer.trace_apply.schema import Solution


def _solution_hash(solution: Solution) -> str:
    h = hashlib.sha256()
    h.update(solution.name.encode())
    h.update(solution.definition.encode())
    for src in solution.sources:
        h.update(src.path.encode())
        h.update(src.content.encode())
    return h.hexdigest()[:16]


def _cache_dir(solution: Solution) -> Path:
    base = Path.home() / ".cache" / "flashinfer" / "trace_apply" / "solutions"
    return base / _solution_hash(solution)


def _materialize(solution: Solution) -> Path:
    out = _cache_dir(solution)
    out.mkdir(parents=True, exist_ok=True)
    for src in solution.sources:
        target = out / src.path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent: only write if content changed.
        if not target.exists() or target.read_text() != src.content:
            target.write_text(src.content)
    return out


def load(solution: Solution) -> Callable:
    """Return the entry-point callable for a Python or Triton Solution.

    `Solution.spec.entry_point` is "<file>::<function>" — e.g. "main.py::run".
    """
    if "::" not in solution.spec.entry_point:
        raise ValueError(
            f"Expected entry_point of the form '<file>::<function>', got {solution.spec.entry_point!r}"
        )
    file_part, func_part = solution.spec.entry_point.split("::", 1)
    sol_dir = _materialize(solution)
    entry_path = sol_dir / file_part
    if not entry_path.is_file():
        raise FileNotFoundError(f"Solution entry-point file not found: {entry_path}")

    module_name = f"flashinfer_trace_apply_solution_{_solution_hash(solution)}"
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {entry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, func_part):
        raise AttributeError(f"Solution {solution.name!r} entry-point missing function {func_part!r}")
    return getattr(module, func_part)
