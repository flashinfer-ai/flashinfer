"""Public surface for :mod:`b12x.sequence.ple_hash`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._contracts import Binding, Caps, Plan, bind, plan, run
from ._tuning import PleHashConfig, PleHashQuery
from ._preparation import invocation_from_tensors
from .geometry import Geometry, GeometryTensors, compute_geometry, allocate_geometry


def is_supported(device=None) -> bool:
    """True on supported b12x devices with Triton available."""
    return default_is_supported(device, requires=("triton",))


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "PleHashConfig",
    "PleHashQuery",
    "Geometry",
    "GeometryTensors",
    "compute_geometry",
    "allocate_geometry",
    "invocation_from_tensors",
    "plan",
    "bind",
    "run",
    "is_supported",
]
