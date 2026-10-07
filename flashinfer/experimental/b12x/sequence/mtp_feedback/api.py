"""Public surface for :mod:`b12x.sequence.mtp_feedback`."""

from __future__ import annotations

from ..._lib.gating import has_cutlass_dsl, has_triton

from . import reference
from ._impl import Binding, Caps, Plan, bind, plan, run
from ._preparation import invocation_from_tensors
from ._tuning import MtpFeedbackConfig, MtpFeedbackQuery


def is_supported(device=None) -> bool:
    """True when mandatory CuTe projections and Triton auxiliaries are usable."""
    del device
    return has_cutlass_dsl() and has_triton()


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "MtpFeedbackConfig",
    "MtpFeedbackQuery",
    "plan",
    "invocation_from_tensors",
    "bind",
    "run",
    "reference",
    "is_supported",
]
