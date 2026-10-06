"""Prepared batched and packed-varlen contiguous attention."""
from __future__ import annotations

from b12x.preparation import Plan

from ..._lib.gating import default_is_supported
from . import META
from ._preparation import BatchedBinding, VarlenBinding, bind, bind_batched, plan, plan_batched, run, run_batched
from ._tuning import VarlenAttentionConfig, VarlenAttentionQuery


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "BatchedBinding", "Plan", "VarlenAttentionConfig",
    "VarlenAttentionQuery", "VarlenBinding", "bind", "bind_batched", "is_supported",
    "plan", "plan_batched", "run", "run_batched",
]
