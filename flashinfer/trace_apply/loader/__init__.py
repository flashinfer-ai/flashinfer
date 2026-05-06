from __future__ import annotations

from typing import Callable

from flashinfer.trace_apply.loader import cutedsl, jit, triton
from flashinfer.trace_apply.schema import Solution

# Mapping from `Solution.spec.language` (lowercased) to a loader function.
LOADERS: dict[str, Callable[[Solution], Callable]] = {
    "python": triton.load,  # pure Python solutions go through the same path as Triton
    "triton": triton.load,
    "cuda": jit.load,
    "cutlass": jit.load,
    "cutile": jit.load,
    "cutedsl": cutedsl.load,
}


def load(solution: Solution) -> Callable:
    """Materialize a Solution into a callable. Raises if the language has no loader."""
    lang = solution.spec.language.lower()
    if lang not in LOADERS:
        raise NotImplementedError(f"No loader registered for Solution.spec.language={solution.spec.language!r}")
    return LOADERS[lang](solution)


__all__ = ["LOADERS", "load"]
