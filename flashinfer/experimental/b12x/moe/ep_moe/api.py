"""Public surface for session-prepared replicated-input EP MoE."""
from __future__ import annotations

from b12x._lib.gating import default_is_supported
from b12x.preparation.types import Plan, require_prepared

from . import META
from ._impl import EPExpertMap as ExpertMap
from ._impl import EPMoEFP4Binding as Binding
from ._impl import EPMoEScratchCaps as Caps
from ._impl import prepare_ep_expert_map as prepare_expert_map
from ._preparation import invocation_from_tensors, plan
from ._tuning import EpMoeConfig, EpMoeQuery



def bind(plan: Plan, **kwargs) -> Binding:
    """Bind caller tensors to one session-prepared EP plan."""
    return require_prepared(plan, "moe.ep_moe").bind(
        _plan=plan, **kwargs
    )

def run(*, binding: Binding):
    """Run a bound EP partial; no compiler or selection path is available here."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be an EPMoEFP4Binding")
    return binding.run()


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps", "Plan", "Binding", "ExpertMap", "EpMoeConfig", "EpMoeQuery", "plan",
    "bind", "run", "prepare_expert_map", "invocation_from_tensors", "is_supported",
]
