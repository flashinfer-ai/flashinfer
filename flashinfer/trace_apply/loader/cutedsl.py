from __future__ import annotations

from typing import Callable

from flashinfer.trace_apply.loader import triton as _triton_loader
from flashinfer.trace_apply.schema import Solution


def load(solution: Solution) -> Callable:
    return _triton_loader.load(solution)
