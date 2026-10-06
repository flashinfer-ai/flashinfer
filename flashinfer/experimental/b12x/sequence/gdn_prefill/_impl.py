"""Planned contract for chunked GDN prefill: caps, plan, bind, run."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
)
from b12x.preparation.types import require_prepared
from .._shared.tensors import canonical_device, positive
from .._shared.delta_prefill.contract import (
    HEAD_DIM, Binding as _SharedBinding, Layout as _SharedLayout, bind_tensors, materialize_layout,
)
from ._tuning import CHUNK_TOKENS, GdnPrefillConfig, GdnPrefillQuery, tiles_capacity
from ._parallel import ParallelBinding, ParallelPlan


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static geometry and planned capacity of a GDN prefill plan."""

    device: torch.device | str
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    key_heads: int
    value_heads: int
    head_dim: int = HEAD_DIM
    model_dtype: torch.dtype = torch.bfloat16
    state_dtype: torch.dtype = torch.float32
    qk_l2norm: bool = True
    checkpoint_export: bool = False
    null_state_index: int | None = None
    chunk_tokens: int = 16
    staging_key: Hashable | None = None
    staging_resident_nbytes: int = 0

    def __post_init__(self) -> None:
        device = canonical_device(self.device)
        if device.type != "cuda":
            raise ValueError(f"GDN prefill requires a CUDA device, got {device}")
        object.__setattr__(self, "device", device)
        for name in ("max_tokens", "max_seqs", "max_state_slots", "key_heads", "value_heads"):
            object.__setattr__(self, name, positive(name, getattr(self, name)))
        if self.value_heads != 3 * self.key_heads:
            raise ValueError("GDN prefill requires three value heads per key head")
        if self.max_seqs > 4096:
            raise ValueError("max_seqs must be at most 4096")
        if int(self.head_dim) != HEAD_DIM:
            raise ValueError(f"head_dim must be {HEAD_DIM}, got {self.head_dim}")
        object.__setattr__(self, "head_dim", HEAD_DIM)
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
        if self.staging_key is not None:
            hash(self.staging_key)
        if type(self.staging_resident_nbytes) is not int or self.staging_resident_nbytes < 0:
            raise ValueError("staging_resident_nbytes must be a nonnegative integer")

    @property
    def heads(self) -> int:
        return self.value_heads

    @property
    def is_gdn(self) -> bool:
        return True

    @property
    def op_name(self) -> str:
        return "gdn_prefill"

    @property
    def tiles_capacity(self) -> int:
        """Upper bound on packed chunk tiles: one partial tile per sequence."""
        return tiles_capacity(self.max_tokens, self.max_seqs)


@dataclass(frozen=True)
class _Layout(_SharedLayout):
    """Fixed GDN launch geometry and caller-owned workspace contract."""

    caps: Caps
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    parallel: ParallelPlan | None = None



@dataclass(frozen=True)
class Binding(_SharedBinding):
    """Caller-owned GDN tensors; a and b are the raw scalar projections."""

    _state: _Layout
    parallel: ParallelBinding | None = None

    @property
    def a(self) -> torch.Tensor:
        return self.raw_g

    @property
    def b(self) -> torch.Tensor:
        return self.raw_beta


def staging_memory(caps: Caps) -> MemoryRequirements:
    """Return the reusable V-side capacity staging envelope for ``caps``.

    Callers that own those buffers provide a stable allocation key and their
    current resident size.  This keeps the native declaration metadata-only
    while allowing session reservation before the owner publishes buffers.
    """
    if caps.staging_key is None:
        return MemoryRequirements()
    element_size = torch.empty((), dtype=caps.model_dtype).element_size()
    int32_size = torch.empty((), dtype=torch.int32).element_size()
    required = (
        caps.max_tokens
        * (2 * caps.key_heads + caps.value_heads)
        * caps.head_dim
        * element_size
        + 2 * caps.max_tokens * caps.value_heads * element_size
        + caps.max_tokens * caps.value_heads * caps.head_dim * element_size
        + ((caps.max_seqs + 1) + 4 * caps.max_seqs + 2) * int32_size
    )
    return MemoryRequirements(persistent=(
        PersistentMemory(
            caps.staging_key,
            required,
            caps.staging_resident_nbytes,
        ),
    ))


def _query(caps: Caps, invocation: FrozenMapping) -> GdnPrefillQuery:
    return GdnPrefillQuery(
        key_heads=caps.key_heads,
        value_heads=caps.value_heads,
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


def _materialize_layout(caps: Caps, config: GdnPrefillConfig) -> _Layout:
    result = materialize_layout(
        caps, layout_type=_Layout, v_split=config.v_split, k_split=config.k_split,
        stages=config.stages, window_tiles=config.window_tiles,
        workspace_windows=0 if config.algorithm == "chunk_parallel" else 2,
    )
    if config.algorithm == "chunk_parallel":
        from ._parallel import materialize
        result = materialize(result, segment_tokens=config.segment_tokens)
    return result


def plan(caps: Caps, *, invocation: FrozenMapping = FrozenMapping(), override: GdnPrefillConfig | None = None) -> Plan:
    """Declare GDN prefill without compiling or allocating resources."""
    from ._preparation import make_plan
    if not isinstance(caps, Caps):
        raise TypeError("caps must be gdn_prefill.Caps")
    return make_plan(caps, invocation=invocation, override=override)


def bind(plan: Plan, **kwargs) -> Binding:
    state = require_prepared(plan, "sequence.gdn_prefill")
    return state.bind(_plan=plan, **kwargs)


def _bind(
    plan: _Layout,
    *,
    _plan: Plan | None = None,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
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
        raise TypeError("binding requires a GDN prefill layout")
    result = bind_tensors(
        plan, binding_type=Binding, _plan=_plan, scratch=scratch, q=q, k=k, v=v,
        raw_g=a, raw_beta=b, A_log=A_log, dt_bias=dt_bias,
        recurrent_state=recurrent_state, cu_seqlens=cu_seqlens,
        initial_state_indices=initial_state_indices, final_state_indices=final_state_indices,
        checkpoint_state_indices=checkpoint_state_indices, checkpoint_offsets=checkpoint_offsets,
        num_seqs=num_seqs, num_tokens=num_tokens, output=output,
    )
    if plan.parallel is not None:
        from ._parallel import bind as bind_parallel

        result = replace(result, parallel=bind_parallel(result))
    return result


def _check_run_scalars(scale: float | None, eps: float) -> tuple[float, float]:
    scale_value = HEAD_DIM**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale_value}")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    return scale_value, eps_value


def run(
    binding: Binding,
    *,
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
        raise TypeError("binding must be gdn_prefill.Binding")
    state = require_prepared(binding.plan, "sequence.gdn_prefill", binding.output.device)
    return state.run(binding, scale=scale, eps=eps,
                     max_live_tokens=max_live_tokens, max_live_seqs=max_live_seqs)




__all__ = [
    "Binding",
    "Caps",
    "Plan",
    "bind",
    "plan",
    "run",
]
