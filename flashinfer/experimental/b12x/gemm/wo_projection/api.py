"""Prepared public surface for native MXFP8 WO projection."""
from __future__ import annotations
from dataclasses import replace
from ..._lib.gating import default_is_supported
from ...preparation import Plan
from ...preparation.types import require_prepared
from .._shared.wo_mxfp8 import MXFP8Rows
from .._shared.wo_mxfp8 import WOProjectionBinding as Binding
from .._shared.wo_mxfp8 import WOProjectionInvRopeBinding as InvRopeBinding
from .._shared.wo_mxfp8 import WOProjectionMXFP8Weights as Weights
from .._shared.wo_mxfp8 import WOProjectionScratchCaps as Caps
from .._shared.wo_mxfp8 import pack_wo_projection_fp8_block_scaled_weights_mxfp8 as pack_weights
from .._shared import wo_mxfp8 as _shared
from ._preparation import plan
from ._tuning import WoProjectionConfig, WoProjectionQuery
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind caller-owned WO tensors and scratch to a ready plan."""
    source = kwargs.get("source_tgd")
    state = require_prepared(plan, "gemm.wo_projection", source.device if source is not None else None)
    return replace(state.bind(**kwargs), plan=plan)


def bind_inv_rope(plan: Plan, **kwargs) -> InvRopeBinding:
    """Bind inverse-RoPE tensors and caller-owned scratch to a ready plan."""
    source = kwargs.get("o")
    state = require_prepared(plan, "gemm.wo_projection", source.device if source is not None else None)
    return replace(state.bind_inv_rope(**kwargs), plan=plan)

def run(*, binding: Binding, plan: Plan, stream=None):
    """Run a prepared WO projection; declarations and raw bindings are rejected."""
    state = require_prepared(plan, "gemm.wo_projection", binding.source_tgd.device)
    if binding.plan is not plan:
        raise ValueError("WO projection binding belongs to a different prepared plan")
    return state.run(binding, stream=stream)

def run_inv_rope(*, binding: InvRopeBinding, plan: Plan, stream=None):
    """Run a prepared inverse-RoPE WO projection."""
    state = require_prepared(plan, "gemm.wo_projection", binding.o.device)
    if binding.plan is not plan:
        raise ValueError("WO projection inverse-RoPE binding belongs to a different prepared plan")
    return state.run_inv_rope(binding, stream=stream)


def quantize_input(source_tgd, *, plan: Plan, out=None):
    state = require_prepared(plan, "gemm.wo_projection", source_tgd.device)
    return state.quantize_a(source_tgd, out=out)


def quantize_input_inv_rope(o, positions, cos_sin_cache, *, groups, heads_per_group,
                             nope_dim=448, rope_dim=64, plan: Plan,
                             out=None):
    state = require_prepared(plan, "gemm.wo_projection", o.device)
    return state.quantize_a_inv_rope(
        o, positions, cos_sin_cache, groups=groups, heads_per_group=heads_per_group,
        nope_dim=nope_dim, rope_dim=rope_dim, out=out,
    )


def quantize_input_b(tmp_trg, *, plan: Plan, out=None):
    state = require_prepared(plan, "gemm.wo_projection", tmp_trg.device)
    return state.quantize_b(tmp_trg, out=out)


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps", "Plan", "Binding", "InvRopeBinding", "Weights", "MXFP8Rows",
    "WoProjectionConfig", "WoProjectionQuery", "plan", "bind", "bind_inv_rope",
    "run", "run_inv_rope", "pack_weights", "quantize_input",
    "quantize_input_inv_rope", "quantize_input_b", "is_supported",
]
