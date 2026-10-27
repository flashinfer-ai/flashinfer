"""Prepared projection and stream-gating API with caller-owned storage."""

import torch

from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared
from ._preparation import plan
from ._tuning import Config, Query


@torch.library.custom_op(
    "b12x::hyperconnection_projection_gate", mutates_args=("out", "workspace")
)
def _run(
    source: torch.Tensor,
    weight: torch.Tensor,
    normalized: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    handle: int,
) -> None:
    state = require_prepared(
        plan_from_handle(handle), "gemm.hyperconnection_gate", source.device
    )
    state.run(source, weight, normalized, out, workspace)


@_run.register_fake
def _fake(source, weight, normalized, out, workspace, handle):
    pass


def run(source, weight, normalized, *, out, workspace, plan: Plan):
    """Project and gate BF16 streams into caller-owned ``out[T,H]``.

    ``workspace`` is contiguous BF16 ``[capacity,S*H]`` for separate projection
    candidates. Neither output may alias an input or the other output. The
    selected fused implementation leaves workspace contents unspecified.
    """
    _run(source, weight, normalized, out, workspace, plan.handle)
    return out


def reference(source, weight, normalized, *, streams):
    from ...norm.hyperconnection.reference import gate_mean

    return gate_mean(
        normalized, torch.nn.functional.linear(source, weight), streams=streams
    )


def is_supported(device=None):
    from ..._lib.gating import default_is_supported
    from . import META

    return default_is_supported(device, requires=META.requires)


__all__ = ["Query", "Config", "plan", "run", "reference", "is_supported"]
