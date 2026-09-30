"""Prepared public API for native DeepSeek V4.1 Engram."""
from b12x.preparation import Plan
from b12x._lib.gating import has_triton
from ._disk import DiskTable
from ._impl import Binding, Caps, LookupBinding, bind, bind_lookup, run, run_lookup, run_lookups
from ._preparation import make_plan as plan
from ._tuning import EngramQuery, TUNING
from .geometry import Geometry, build_compressed_token_map, build_geometry

def is_supported(device=None):
    del device
    return has_triton()

__all__ = ["Caps", "Plan", "Binding", "LookupBinding", "DiskTable", "EngramQuery", "TUNING", "Geometry", "build_geometry", "build_compressed_token_map", "plan", "bind", "bind_lookup", "run", "run_lookup", "run_lookups", "is_supported"]
