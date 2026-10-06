"""HyperConnection declarations and prepared-only mathematical entry points."""
from __future__ import annotations

import torch

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared
from ..._lib.gating import has_cutlass_dsl, has_triton
from . import reference
from . import _kernels
from ._impl import HyperConnectionBinding as Binding
from ._impl import HyperConnectionCaps as Caps
from ._preparation import plan_hyperconnection as plan
from ._tuning import HyperConnectionConfig, HyperConnectionQuery


def bind(plan: Plan, **kwargs) -> Binding:
    state = require_prepared(plan, "norm.hyperconnection")
    return state.bind(plan=plan, **kwargs)


def _bound_plan_handle(binding):
    if binding.plan is None:
        raise TypeError("binding has not been installed from a prepared Plan")
    return binding.plan.handle


def run_grouped_rmsnorm(state, weight, *, eps, binding: Binding, zero_centered=True):
    if state.shape[0] != binding.tokens:
        raise ValueError("state rows differ from bound live rows")
    torch.ops.b12x.hyperconnection_grouped_rmsnorm(
        state, weight, binding.normalized_capacity, float(eps), _bound_plan_handle(binding), bool(zero_centered),
    )
    return binding.normalized


def run_scaled_silu(projected_down, *, binding: Binding):
    if projected_down.shape[0] != binding.tokens:
        raise ValueError("projection rows differ from bound live rows")
    torch.ops.b12x.hyperconnection_scaled_silu(
        projected_down, binding.bottleneck_capacity, _bound_plan_handle(binding),
    )
    return binding.bottleneck


def run_gate_mean(normalized, gate_logits, *, binding: Binding):
    if normalized.shape[0] != binding.tokens:
        raise ValueError("normalized rows differ from bound live rows")
    torch.ops.b12x.hyperconnection_gate_mean(
        normalized, gate_logits, binding.block_input_capacity, _bound_plan_handle(binding),
    )
    return binding.block_input


def run_combine(state, block_output, injection_logits, *, plan: Plan):
    return torch.ops.b12x.hyperconnection_combine(state, block_output, injection_logits, plan.handle)


def run_combine_norm(state, block_output, injection_logits, next_norm_weight, *, eps, plan: Plan):
    return torch.ops.b12x.hyperconnection_combine_norm(
        state, block_output, injection_logits, next_norm_weight, float(eps), plan.handle,
    )


def run_engram_mix(state, projected_kv, norm_weights, *, eps, plan: Plan, out, token_mask=None):
    torch.ops.b12x.hyperconnection_engram_mix(
        state, projected_kv, norm_weights, token_mask, out, float(eps), plan.handle,
    )
    return out


def run_swiglu(gate_up, *, limit, out, plan: Plan, round_silu=False):
    torch.ops.b12x.hyperconnection_swiglu(gate_up, out, float(limit), bool(round_silu), plan.handle)
    return out


def run_add(left, right, *, out, plan: Plan):
    torch.ops.b12x.hyperconnection_add(left, right, out, plan.handle)
    return out


def run_sigmoid(source, *, out, plan: Plan):
    torch.ops.b12x.hyperconnection_sigmoid(source, out, plan.handle)
    return out


def is_supported(device=None):
    del device
    return has_cutlass_dsl() and has_triton()


__all__ = [
    "Caps", "Plan", "Binding", "HyperConnectionConfig", "HyperConnectionQuery",
    "plan", "bind", "run_grouped_rmsnorm", "run_scaled_silu", "run_gate_mean",
    "run_combine", "run_combine_norm", "run_engram_mix", "run_swiglu", "run_add", "run_sigmoid",
    "reference", "is_supported",
]
