"""Public surface for norm.mhc (docs in the op ``__init__``)."""

from __future__ import annotations

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared
from ..._lib.gating import default_is_supported
from ._impl import (
    MHC_DEFAULT_BLOCK_H as DEFAULT_BLOCK_H,
)
from ._tuning import MhcConfig, MhcQuery
from ._impl import (
    MHC_DEFAULT_BLOCK_K as DEFAULT_BLOCK_K,
)
from ._impl import (
    MHC_DEFAULT_SPLIT_K as DEFAULT_SPLIT_K,
)
from ._impl import (
    MHC_MIXES as MIXES,
)
from ._impl import (
    MHC_MULT as MULT,
)
from ._impl import (
    MHC_PARTIALS as PARTIALS,
)
from ._impl import (
    B12XMHCBinding as Binding,
)
from ._impl import (
    B12XMHCScratchCaps as Caps,
)
from ._preparation import plan_mhc as plan
from ._impl import run_collapse
from ._impl import (
    b12x_mhc_post as run_post,
)
from ._impl import (
    b12x_mhc_post_pre as run_post_pre,
)
from ._impl import (
    b12x_mhc_pre as run_pre,
)
from ._impl import run_collapse
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind caller-owned scratch to a prepared plan."""
    state = require_prepared(plan, "norm.mhc")
    return state.bind(plan=plan, **kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "MhcConfig",
    "MhcQuery",
    "plan",
    "bind",
    "run_pre",
    "run_post",
    "run_post_pre",
    "run_collapse",
    "MIXES",
    "MULT",
    "PARTIALS",
    "DEFAULT_SPLIT_K",
    "DEFAULT_BLOCK_K",
    "DEFAULT_BLOCK_H",
    "is_supported",
]
