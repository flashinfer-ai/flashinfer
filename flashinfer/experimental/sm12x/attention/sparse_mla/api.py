# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for attention.sparse_mla (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from .._shared.mla.api import (
    MLASparseDecodeMetadata as DecodeMetadata,
)
from .._shared.mla.api import (
    MLASparseExtendMetadata as ExtendMetadata,
)
from .._shared.mla.api import (
    clear_mla_caches as clear_caches,
)
from .._shared.mla.api import (
    sparse_mla_decode_forward as run_decode,
)
from .._shared.mla.api import (
    sparse_mla_extend_forward as run_extend,
)
from ._scratch import (
    SM12XSparseMLABinding as Binding,
)
from ._scratch import (
    SM12XSparseMLAScratch as Scratch,
)
from ._scratch import (
    SM12XSparseMLAScratchCaps as Caps,
)
from ._scratch import (
    SM12XSparseMLAScratchPlan as Plan,
)
from ._scratch import (
    plan_sparse_mla_scratch as plan,
)
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "Scratch",
    "DecodeMetadata",
    "ExtendMetadata",
    "plan",
    "bind",
    "run_decode",
    "run_extend",
    "is_supported",
    "clear_caches",
]
