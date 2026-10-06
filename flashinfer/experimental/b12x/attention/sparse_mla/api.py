"""Public surface for attention.sparse_mla (docs in the op ``__init__``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x._lib.gating import default_is_supported
from b12x.preparation import FrozenMapping, Plan
from .._shared.mla.traits import ModelType
from .._shared.mla.api import (
    MLASparseDecodeMetadata as DecodeMetadata,
)
from .._shared.mla.api import (
    MLASparseExtendMetadata as ExtendMetadata,
)
from .._shared.mla.api import (
    clear_mla_caches as clear_caches,
)
from .pooled_selection import (
    expand_pooled_topk_to_physical_slots,
    plan_pooled_selection,
)
from ._scratch import (
    B12XSparseMLABinding as _RuntimeBinding,
)
from ._tuning import SparseMlaConfig, SparseMlaQuery
from ._scratch import (
    B12XSparseMLAScratch as Scratch,
)
from ._scratch import (
    B12XSparseMLAScratchCaps as Caps,
)
from ._preparation import plan as _plan
from ._preparation import state as _state
from ._preparation import plan_cache_writer
from ._preparation import writer_state as _writer_state
from . import META


@dataclass(frozen=True, kw_only=True)
class Binding:
    """A complete sparse-MLA invocation bound to one immutable plan."""

    plan: Plan
    runtime: _RuntimeBinding
    kv_cache: torch.Tensor
    attention_sink: torch.Tensor | None = None


def plan(
    caps: Caps,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: SparseMlaConfig | None = None,
) -> Plan:
    """Declare a sparse-MLA route for PreparationSession."""
    return _plan(caps, invocation=invocation, override=override)


def bind(
    plan: Plan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    cache_lengths: torch.Tensor,
    selected_lengths: torch.Tensor,
    attention_sink: torch.Tensor | None = None,
) -> Binding:
    """Bind live tensors to an already prepared sparse-MLA plan."""

    state = _state(plan, device=kv_cache.device)
    caps = state.caps
    if kv_cache.ndim != 3:
        raise ValueError(f"kv_cache must be rank-3, got {tuple(kv_cache.shape)}")
    if kv_cache.device != caps.device:
        raise ValueError(f"kv_cache must be on {caps.device}, got {kv_cache.device}")
    if kv_cache.dtype != caps.kv_dtype:
        raise TypeError(
            f"kv_cache must have dtype {caps.kv_dtype}, got {kv_cache.dtype}"
        )
    if bool(attention_sink is not None) != bool(caps.has_attention_sink):
        raise ValueError(
            "attention_sink presence must match plan.caps.has_attention_sink"
        )
    if attention_sink is not None and caps.mode != "decode":
        raise ValueError("attention_sink is supported only by sparse MLA decode")
    if attention_sink is not None:
        if attention_sink.shape != (caps.num_q_heads,):
            raise ValueError(
                "attention_sink must have shape "
                f"({caps.num_q_heads},), got {tuple(attention_sink.shape)}"
            )
        if attention_sink.dtype != torch.float32:
            raise TypeError(
                f"attention_sink must have dtype torch.float32, got "
                f"{attention_sink.dtype}"
            )
        if attention_sink.device != caps.device or not attention_sink.is_contiguous():
            raise ValueError(f"attention_sink must be contiguous on {caps.device}")
    runtime = state.bind(
        scratch=scratch,
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_lengths,
        nsa_cache_seqlens_int32=selected_lengths,
        kv_cache=kv_cache,
    )
    return Binding(
        plan=plan,
        runtime=runtime,
        kv_cache=kv_cache,
        attention_sink=attention_sink,
    )


def run(binding: Binding) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute a complete binding through the route selected by its plan."""

    if not isinstance(binding, Binding):
        raise TypeError("binding must be sparse_mla.Binding")
    state = _state(binding.plan, device=binding.kv_cache.device)
    return state.run(
        binding.runtime,
        kv_cache=binding.kv_cache,
        attention_sink=binding.attention_sink,
    )


def concat_and_cache_glm_next_mla(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    plan: Plan,
) -> None:
    """Write GLM_NEXT cache records through a prepared native writer."""
    _writer_state(plan, device=kv_cache.device).run(kv_c, kv_cache, slot_mapping)


def concat_and_cache_glm_next_mla_fp8(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    plan: Plan,
) -> None:
    concat_and_cache_glm_next_mla(kv_c, kv_cache, slot_mapping, plan=plan)


def concat_and_cache_glm_next_mla_nvfp4(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    plan: Plan,
) -> None:
    concat_and_cache_glm_next_mla(kv_c, kv_cache, slot_mapping, plan=plan)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "ModelType",
    "Caps",
    "Binding",
    "Scratch",
    "DecodeMetadata",
    "ExtendMetadata",
    "SparseMlaConfig",
    "SparseMlaQuery",
    "plan",
    "bind",
    "run",
    "plan_cache_writer",
    "plan_pooled_selection",
    "concat_and_cache_glm_next_mla",
    "concat_and_cache_glm_next_mla_fp8",
    "concat_and_cache_glm_next_mla_nvfp4",
    "expand_pooled_topk_to_physical_slots",
    "is_supported",
    "clear_caches",
]
