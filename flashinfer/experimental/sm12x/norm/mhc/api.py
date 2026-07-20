# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for norm.mhc (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._impl import (
    MHC_DEFAULT_BLOCK_H as DEFAULT_BLOCK_H,
)
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
    SM12XMHCBinding as Binding,
)
from ._impl import (
    SM12XMHCScratchCaps as Caps,
)
from ._impl import (
    SM12XMHCScratchPlan as Plan,
)
from ._impl import (
    plan_mhc_scratch as plan,
)
from ._impl import (
    sm12x_mhc_post as run_post,
)
from ._impl import (
    sm12x_mhc_post_pre as run_post_pre,
)
from ._impl import (
    sm12x_mhc_pre as run_pre,
)
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "plan",
    "bind",
    "run_pre",
    "run_post",
    "run_post_pre",
    "MIXES",
    "MULT",
    "PARTIALS",
    "DEFAULT_SPLIT_K",
    "DEFAULT_BLOCK_K",
    "DEFAULT_BLOCK_H",
    "is_supported",
]
