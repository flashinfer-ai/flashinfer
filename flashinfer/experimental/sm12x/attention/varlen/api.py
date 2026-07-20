# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for attention.varlen (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from .._shared.contiguous.api import (
    AttentionBinding as BatchedBinding,
)
from .._shared.contiguous.api import (
    AttentionPlan as BatchedPlan,
)
from .._shared.contiguous.api import (
    AttentionScratchPlan as BatchedScratchPlan,
)
from .._shared.contiguous.api import (
    VarlenAttentionBinding as VarlenBinding,
)
from .._shared.contiguous.api import (
    VarlenAttentionPlan as VarlenPlan,
)
from .._shared.contiguous.api import (
    VarlenAttentionScratchPlan as VarlenScratchPlan,
)
from .._shared.contiguous.api import (
    clear_attention_caches as clear_caches,
)
from .._shared.contiguous.api import (
    create_attention_plan as create_plan_batched,
)
from .._shared.contiguous.api import (
    create_varlen_attention_plan as create_plan,
)
from .._shared.contiguous.api import (
    plan_attention_scratch as plan_batched,
)
from .._shared.contiguous.api import (
    plan_varlen_attention_scratch as plan,
)
from .._shared.contiguous.api import (
    sm12x_attention_forward as run_batched,
)
from .._shared.contiguous.api import (
    sm12x_varlen_attention_forward as run,
)
from . import META


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = list(META.entry_points)
