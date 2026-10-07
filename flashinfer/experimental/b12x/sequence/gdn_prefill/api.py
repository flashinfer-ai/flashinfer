"""Public surface for :mod:`b12x.sequence.gdn_prefill`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported

from . import reference
from .._shared.delta_prefill._cute_kernels import clear_caches
from ._impl import Binding, Caps, Plan, bind, plan, run, staging_memory
from ._preparation import invocation_from_tensors
from ._tuning import GdnPrefillConfig, GdnPrefillQuery


def is_supported(device=None) -> bool:
    """True when the CuTe DSL kernels can run on ``device``."""
    return default_is_supported(device)


__all__ = [
    "Binding",
    "Caps",
    "GdnPrefillConfig",
    "GdnPrefillQuery",
    "Plan",
    "bind",
    "clear_caches",
    "is_supported",
    "plan",
    "invocation_from_tensors",
    "reference",
    "run",
    "staging_memory",
]
