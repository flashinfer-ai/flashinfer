"""Prepared public surface for native paged attention."""
from __future__ import annotations

from b12x._lib.gating import default_is_supported
from b12x.preparation import Plan

from . import META
from ._forward import clear_paged_caches as clear_caches
from ._preparation import bind as _bind, plan as _plan, run as _run
from ._preparation import invocation_from_descriptors, invocation_from_tensors, memory_requirements
from ._forward import _compile_paged_attention, paged_attention_forward
from ._scratch import B12XPagedAttentionScratchPlan, plan_paged_attention_scratch
from ._scratch import (
    B12XPagedAttentionBinding as Binding,
    B12XPagedAttentionScratchCaps as Caps,
    B12XPagedDecodeGraphScratchEnvelope as DecodeGraphScratchEnvelope,
    plan_decode_graph_scratch_envelope as decode_graph_scratch_envelope,
)
from ._tuning import GqaConfig, GqaQuery
from .planner import (
    PagedDecodeGraphCapacity as DecodeGraphCapacity,
    PagedExtendGraphCapacity as ExtendGraphCapacity,
    PagedPlanBudget as Budget,
    PagedVerifyGraphCapacity as VerifyGraphCapacity,
    infer_paged_mode as infer_mode,
    plan_decode_graph_capacity as decode_graph_capacity,
    plan_extend_graph_capacity as extend_graph_capacity,
    plan_verify_graph_capacity as verify_graph_capacity,
)
from .workspace import PagedAttentionWorkspace as Workspace


def plan(caps, *, invocation=None, override=None):
    """Declare session preparation, or size heuristic scratch without metadata."""
    if invocation is not None:
        return _plan(caps, invocation=invocation, override=override)
    if override is not None:
        from dataclasses import replace
        caps = replace(caps, config=override)
    return plan_paged_attention_scratch(caps)


def bind(plan, **kwargs):
    if isinstance(plan, B12XPagedAttentionScratchPlan):
        return plan.bind(**kwargs)
    return _bind(plan, **kwargs)


def compile(*, binding):
    """Prime a heuristic binding before serving or CUDA graph capture."""
    if binding.plan is not None:
        from b12x.preparation import require_prepared
        require_prepared(binding.plan, "attention.gqa")
        return
    _compile_paged_attention(binding=binding)


def run(*, binding, plan=None):
    if binding.plan is None and plan is None:
        return paged_attention_forward(binding=binding)
    return _run(binding=binding, plan=plan)


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps", "Plan", "Binding", "Workspace", "Budget", "DecodeGraphCapacity",
    "GqaConfig", "GqaQuery", "ExtendGraphCapacity", "VerifyGraphCapacity",
    "DecodeGraphScratchEnvelope", "decode_graph_capacity", "extend_graph_capacity",
    "verify_graph_capacity", "decode_graph_scratch_envelope", "plan", "bind",
    "compile", "run", "invocation_from_descriptors", "invocation_from_tensors", "memory_requirements", "infer_mode", "is_supported", "clear_caches",
]
