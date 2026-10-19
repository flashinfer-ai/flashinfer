"""Public surface for :mod:`b12x.sequence.ple`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from b12x.preparation import Plan
from ._contracts import (
    LayerBinding as Binding,
    LayerCaps as Caps,
    bind_layer as bind,
    plan_layer as plan,
    run_decode,
    run_mixed,
    run_prefill,
)
from ._preparation import invocation_from_tensors
from ._tuning import PleConfig, PleQuery


def is_supported(device=None) -> bool:
    """True on supported b12x devices with Triton available."""
    return default_is_supported(device, requires=("triton",))


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "PleConfig",
    "PleQuery",
    "plan",
    "invocation_from_tensors",
    "bind",
    "run_decode",
    "run_mixed",
    "run_prefill",
    "is_supported",
]
