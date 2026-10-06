"""Prepared public API for native CSA compression."""
from b12x.preparation import Plan
from b12x._lib.gating import has_cutlass_dsl
from . import reference
from ._impl import Binding, Caps, bind, run
from ._preparation import make_plan as plan
from ._tuning import MlaCompressQuery, TUNING

def is_supported(device=None):
    del device
    return has_cutlass_dsl()

__all__ = ["Caps", "Plan", "Binding", "MlaCompressQuery", "TUNING", "plan", "bind", "run", "reference", "is_supported"]
