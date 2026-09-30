"""Planned contract for chunked KDA prefill: caps, plan, bind, run."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from b12x.preparation.types import require_prepared
from .._shared.kda_math import KDA_HEAD_DIM
from .._shared.tensors import canonical_device, positive
from .._shared.delta_prefill.contract import (
    Binding as _SharedBinding, Layout as _SharedLayout, bind_tensors, materialize_layout,
)
from ._tuning import CHUNK_TOKENS, KdaPrefillConfig, KdaPrefillQuery, tiles_capacity


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static geometry and planned capacity of a KDA prefill plan."""

    device: torch.device | str
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    heads: int
    head_dim: int = KDA_HEAD_DIM
    model_dtype: torch.dtype = torch.bfloat16
    state_dtype: torch.dtype = torch.float32
    qk_l2norm: bool = True
    checkpoint_export: bool = False
    null_state_index: int | None = None
    chunk_tokens: int = 16

    def __post_init__(self) -> None:
        device = canonical_device(self.device)
        if device.type != "cuda":
            raise ValueError(f"KDA prefill requires a CUDA device, got {device}")
        object.__setattr__(self, "device", device)
        for name in ("max_tokens", "max_seqs", "max_state_slots", "heads"):
            object.__setattr__(self, name, positive(name, getattr(self, name)))
        if self.max_seqs > 4096:
            raise ValueError("max_seqs must be at most 4096")
        if int(self.head_dim) != KDA_HEAD_DIM:
            raise ValueError(f"head_dim must be {KDA_HEAD_DIM}, got {self.head_dim}")
        object.__setattr__(self, "head_dim", KDA_HEAD_DIM)
        if self.model_dtype != torch.bfloat16:
            raise ValueError("model_dtype must be torch.bfloat16")
        if self.state_dtype != torch.float32:
            raise ValueError("state_dtype must be torch.float32")
        if self.chunk_tokens != CHUNK_TOKENS:
            raise ValueError(f"chunk_tokens must be {CHUNK_TOKENS}")
        object.__setattr__(self, "qk_l2norm", bool(self.qk_l2norm))
        object.__setattr__(self, "checkpoint_export", bool(self.checkpoint_export))
        if self.null_state_index is not None:
            null = int(self.null_state_index)
            if null < 0 or null >= self.max_state_slots:
                raise ValueError("null_state_index must be a valid slot index")
            object.__setattr__(self, "null_state_index", null)

    @property
    def key_heads(self) -> int:
        return self.heads

    @property
    def is_gdn(self) -> bool:
        return False

    @property
    def op_name(self) -> str:
        return "kda_prefill"

    @property
    def tiles_capacity(self) -> int:
        """Upper bound on packed chunk tiles: one partial tile per sequence."""
        return tiles_capacity(self.max_tokens, self.max_seqs)


@dataclass(frozen=True)
class _Layout(_SharedLayout):
    """Fixed KDA launch geometry and caller-owned workspace contract."""

    caps: Caps
    _scratch_specs: tuple[ScratchBufferSpec, ...]



@dataclass(frozen=True)
class Binding(_SharedBinding):
    """Caller-owned tensors for lower-bounded KDA prefill."""

    _state: _Layout


def staging_memory(caps: Caps) -> MemoryRequirements:
    """KDA prefill binds its existing caller-owned capacity buffers directly."""
    del caps
    return MemoryRequirements()


def _query(caps: Caps, invocation: FrozenMapping) -> KdaPrefillQuery:
    return KdaPrefillQuery(
        heads=caps.heads,
        head_dim=caps.head_dim,
        model_dtype=str(caps.model_dtype).removeprefix("torch."),
        state_dtype=str(caps.state_dtype).removeprefix("torch."),
        qk_l2norm=caps.qk_l2norm,
        checkpoint_export=caps.checkpoint_export,
        max_tokens=caps.max_tokens,
        max_seqs=caps.max_seqs,
        max_state_slots=caps.max_state_slots, null_state_index=caps.null_state_index,
        **dict(invocation),
    )


def _materialize_layout(caps: Caps, config: KdaPrefillConfig) -> _Layout:
    return materialize_layout(
        caps, layout_type=_Layout, v_split=config.v_split, k_split=config.k_split,
        stages=config.stages, window_tiles=config.window_tiles,
    )


def plan(caps: Caps, *, invocation: FrozenMapping = FrozenMapping(), override: KdaPrefillConfig | None = None) -> Plan:
    """Declare KDA prefill without compiling or allocating resources."""
    from ._preparation import make_plan
    if not isinstance(caps, Caps):
        raise TypeError("caps must be kda_prefill.Caps")
    return make_plan(caps, invocation=invocation, override=override)


def bind(plan: Plan, **kwargs) -> Binding:
    state = require_prepared(plan, "sequence.kda_prefill")
    return state.bind(_plan=plan, **kwargs)


def _bind(
    plan: _Layout,
    *,
    _plan: Plan | None = None,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state_indices: torch.Tensor,
    final_state_indices: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
) -> Binding:
    """Bind live tensors to a plan without allocating or launching work.

    Live capacities come from the bound tensors: ``q.shape[0]`` tokens and
    ``cu_seqlens.numel() - 1`` sequences, each at most the planned capacity.
    """
    if not isinstance(plan, _Layout):
        raise TypeError("binding requires a KDA prefill layout")
    return bind_tensors(
        plan, binding_type=Binding, _plan=_plan, scratch=scratch, q=q, k=k, v=v,
        raw_g=raw_g, raw_beta=raw_beta, A_log=A_log, dt_bias=dt_bias,
        recurrent_state=recurrent_state, cu_seqlens=cu_seqlens,
        initial_state_indices=initial_state_indices, final_state_indices=final_state_indices,
        checkpoint_state_indices=checkpoint_state_indices, checkpoint_offsets=checkpoint_offsets,
        num_seqs=num_seqs, num_tokens=num_tokens, output=output,
    )


def _check_run_scalars(lower_bound: float, scale: float | None, eps: float) -> tuple[float, float, float]:
    lower_bound_value = float(lower_bound)
    if not math.isfinite(lower_bound_value) or not -5.0 <= lower_bound_value < 0.0:
        raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound_value}")
    scale_value = KDA_HEAD_DIM**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale_value}")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    return lower_bound_value, scale_value, eps_value


def run(
    binding: Binding,
    *,
    lower_bound: float,
    scale: float | None = None,
    eps: float = 1e-6,
    max_live_tokens: int | None = None,
    max_live_seqs: int | None = None,
) -> torch.Tensor:
    """Run the prologue, prepare, and recurrence kernels; capture safe.

    ``max_live_tokens`` and ``max_live_seqs`` are optional host-side upper
    bounds on the device counts; they only limit how many pipeline windows are
    launched, so the bounds are part of the caller's contract: live tiles
    beyond the launched windows are not processed.

    A sequence whose tiles span more than one pipeline window keeps its
    running state in its final state slot between windows, so such a
    sequence must have a non-null final slot. Packed metadata is not checked
    on the device; the caller supplies in-range, conflict-free slots.
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be kda_prefill.Binding")
    state = require_prepared(binding.plan, "sequence.kda_prefill", binding.output.device)
    return state.run(binding, lower_bound=lower_bound, scale=scale, eps=eps,
                     max_live_tokens=max_live_tokens, max_live_seqs=max_live_seqs)




__all__ = [
    "Binding",
    "Caps",
    "Plan",
    "bind",
    "plan",
    "run",
]
