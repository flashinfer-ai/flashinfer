"""Public surface for :mod:`b12x.sequence.kda_prefill`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported

from . import reference
from .._shared.delta_prefill._cute_kernels import clear_caches
from ._impl import Binding, Caps, Plan, bind, plan, run
from ._preparation import invocation_from_tensors
from ._tuning import KdaPrefillConfig, KdaPrefillQuery


def is_supported(device=None) -> bool:
    """True when the CuTe DSL kernels can run on ``device``."""
    return default_is_supported(device)


__all__ = [
    "Binding",
    "Caps",
    "KdaPrefillConfig",
    "KdaPrefillQuery",
    "Plan",
    "bind",
    "clear_caches",
    "is_supported",
    "plan",
    "invocation_from_tensors",
    "reference",
    "run",
]
