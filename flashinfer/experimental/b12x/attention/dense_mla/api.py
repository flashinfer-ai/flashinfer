"""Prepared public API for paged dense MLA."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.gating import default_is_supported
from b12x.preparation import FrozenMapping, Plan

from . import META
from ._kernel import clear_dense_mla_kernel_caches
from ._preparation import (
    invocation_from_descriptors,
    invocation_from_tensors,
    plan as _plan,
    state as _state,
)
from ._reference import dense_mla_reference
from ._scratch import Binding as _RuntimeBinding, Caps, Scratch
from ._tuning import DenseMlaConfig, DenseMlaQuery
from .planner import Budget, infer_dense_mla_mode


@dataclass(frozen=True, kw_only=True)
class Binding:
    """Live caller-owned tensors bound to one prepared dense-MLA plan."""

    plan: Plan
    runtime: _RuntimeBinding

    def __getattr__(self, name):
        return getattr(self.runtime, name)


def plan(
    caps: Caps,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: DenseMlaConfig | None = None,
) -> Plan:
    """Declare a dense-MLA route for ``PreparationSession``."""
    return _plan(caps, invocation=invocation, override=override)


def bind(
    plan: Plan,
    **kwargs,
) -> Binding:
    """Bind caller-owned views to an already prepared dense-MLA plan."""
    if not isinstance(plan, Plan):
        raise TypeError("bind requires a session-prepared Plan")
    device = kwargs.get("q").device if isinstance(kwargs.get("q"), torch.Tensor) else None
    return Binding(
        plan=plan,
        runtime=_state(plan, device=device).bind(**kwargs),
    )


def run(
    *, plan: Plan | None = None, binding: Binding
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run only native entries retained by the prepared plan."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be dense_mla.Binding")
    if plan is None:
        plan = binding.plan
    if plan is not binding.plan:
        raise ValueError("plan must match binding.plan")
    return _state(plan, device=binding.runtime.q.device).run(binding.runtime)


def reference(*args, **kwargs):
    return dense_mla_reference(*args, **kwargs)


def infer_mode(cu_seqlens_q):
    return infer_dense_mla_mode(cu_seqlens_q)


def is_supported(device=None) -> bool:
    if not default_is_supported(device, requires=META.requires):
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    return tuple(torch.cuda.get_device_capability(device)) in ((12, 0), (12, 1))


def clear_caches() -> None:
    clear_dense_mla_kernel_caches()


__all__ = [
    "Binding", "Budget", "DenseMlaConfig", "DenseMlaQuery", "Caps", "Plan",
    "Scratch", "bind", "clear_caches", "infer_mode", "invocation_from_descriptors",
    "invocation_from_tensors", "is_supported", "plan", "reference", "run",
]
