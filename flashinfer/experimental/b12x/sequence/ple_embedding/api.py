"""Public surface for :mod:`b12x.sequence.ple_embedding`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from . import META
from ._contracts import (
    Binding,
    Caps,
    Plan,
    QuantMode,
    TableMemory,
    TableLayout,
    storage_layout,
    bind,
    plan,
    run,
)
from ._disk import DiskTable
from ._storage import TableStorage, allocate_storage
from ._tuning import PleEmbeddingConfig, PleEmbeddingQuery
from ._preparation import invocation_from_tensors
from b12x.sequence.ple_hash.geometry import Geometry, GeometryTensors, compute_geometry, allocate_geometry


def is_supported(device=None) -> bool:
    """True when the registered b12x architecture can run this Triton op."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "QuantMode",
    "TableMemory",
    "TableStorage",
    "DiskTable",
    "Caps",
    "Plan",
    "TableLayout",
    "storage_layout",
    "Geometry",
    "GeometryTensors",
    "compute_geometry",
    "allocate_geometry",
    "invocation_from_tensors",
    "Binding",
    "PleEmbeddingConfig",
    "PleEmbeddingQuery",
    "plan",
    "allocate_storage",
    "bind",
    "run",
    "is_supported",
]
