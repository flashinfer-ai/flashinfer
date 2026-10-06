"""Public surface for :mod:`b12x.gemm.bf16_vocab_projection`."""
from __future__ import annotations

from b12x.preparation import Plan

from ..._lib.gating import default_is_supported
from . import META
from ._contracts import Binding, Caps, bind, plan, run
from ._tuning import Bf16VocabProjectionConfig, Bf16VocabProjectionQuery


def is_supported(device=None) -> bool:
    """True on supported b12x devices with Triton available."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Binding",
    "Bf16VocabProjectionConfig",
    "Bf16VocabProjectionQuery",
    "Caps",
    "Plan",
    "bind",
    "is_supported",
    "plan",
    "run",
]
