"""Public surface for :mod:`b12x.sequence.gdn_decode`."""

from __future__ import annotations

from ..._lib.gating import has_cutlass_dsl, has_triton
from b12x.preparation import Plan

from . import reference
from ._impl import (
    Binding,
    Caps,
    KdaBinding,
    bind,
    bind_kda,
    run,
    run_kda,
)
from ._preparation import plan, invocation_from_tensors
from ._tuning import GdnConfig, GdnQuery
from ._commit import KdaCommitBinding, bind_kda_commit, run_kda_commit


def is_supported(device=None) -> bool:
    """True when mandatory Qwen CuTe and its Triton auxiliaries are usable."""
    del device
    return has_cutlass_dsl() and has_triton()


__all__ = [
    "Binding",
    "Caps",
    "GdnConfig",
    "GdnQuery",
    "KdaBinding",
    "KdaCommitBinding",
    "Plan",
    "bind",
    "bind_kda",
    "bind_kda_commit",
    "is_supported",
    "plan",
    "invocation_from_tensors",
    "reference",
    "run",
    "run_kda",
    "run_kda_commit",
]
