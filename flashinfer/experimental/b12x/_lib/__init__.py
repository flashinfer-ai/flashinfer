"""Shared runtime for b12x.

One infrastructure spine for every op: the CuTe-DSL compile/cache wrapper
(``compiler``), device PTX intrinsics and host quant helpers (``intrinsics``),
dtype/stream/arch utilities (``utils``), the caller-owned scratch contract
(``scratch``/``scratch_layout``), serving freeze guards (``runtime_control``),
CUTLASS DSL runtime patches (``runtime_patches``), plus the hand-written
``env``/``meta``/``gating`` conventions.

Importing this package is side-effect free and light (no cutlass, no torch
ops). Attribute access hydrates the aggregation surface once, which is when
the CUTLASS runtime patches apply (via ``compiler`` import).
"""

from __future__ import annotations

import importlib
from typing import Any

# Order matters: compiler first (applies runtime patches before intrinsics/utils
# import cutlass); later modules win name collisions in the aggregation surface.
_AGG_MODULES = ("compiler", "runtime_control", "scratch", "intrinsics", "utils")
_hydrated = False


def _hydrate() -> None:
    global _hydrated
    if _hydrated:
        return
    _hydrated = True
    for module_name in _AGG_MODULES:
        module = importlib.import_module(f".{module_name}", __name__)
        public = getattr(module, "__all__", None)
        if public is None:
            public = [name for name in vars(module) if not name.startswith("_")]
        for symbol in public:
            globals()[symbol] = getattr(module, symbol)


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _hydrate()
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
