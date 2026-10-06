"""Capacity plans and runtime bindings for PLE hashing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
)
from b12x.preparation import FrozenMapping, Plan
from b12x.preparation.types import require_prepared

from ._tuning import PleHashConfig
from .geometry import Geometry, GeometryTensors



def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def _require_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


def _require_mutation_alias_contract(
    *,
    mutable: tuple[tuple[str, torch.Tensor], ...],
    read_only: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    for index, (left_name, left) in enumerate(mutable):
        for right_name, right in mutable[index + 1 :]:
            if _overlaps(left, right):
                raise ValueError(
                    f"mutable buffers {left_name} and {right_name} must not overlap"
                )
        for right_name, right in read_only:
            if _overlaps(left, right):
                raise ValueError(
                    f"mutable buffer {left_name} must not overlap read-only "
                    f"tensor {right_name}"
                )


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Capacity and immutable model inputs for PLE hashing.

    ``max_tokens`` and ``max_seqs`` are serving capacities and therefore part
    of the compile/cache identity. ``dense_layer_ordinal`` is the zero-based
    ordinal among PLE-enabled layers, not the decoder layer index.
    """

    device: torch.device | str
    max_tokens: int
    max_seqs: int
    vocab_size: int
    eos_token_id: int
    max_order: int
    heads_per_order: int
    dense_layer_ordinal: int
    base_table_size: int
    table_alignment: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        for name in (
            "max_tokens",
            "max_seqs",
            "vocab_size",
            "heads_per_order",
            "base_table_size",
            "table_alignment",
        ):
            object.__setattr__(self, name, _require_positive(name, getattr(self, name)))
        max_order = int(self.max_order)
        if max_order < 2:
            raise ValueError(f"max_order must be at least 2, got {max_order}")
        object.__setattr__(self, "max_order", max_order)
        dense_layer_ordinal = int(self.dense_layer_ordinal)
        if dense_layer_ordinal < 0:
            raise ValueError(
                f"dense_layer_ordinal must be nonnegative, got {dense_layer_ordinal}"
            )
        object.__setattr__(self, "dense_layer_ordinal", dense_layer_ordinal)
        eos_token_id = int(self.eos_token_id)
        if eos_token_id < 0 or eos_token_id >= int(self.vocab_size):
            raise ValueError(
                f"eos_token_id must be in [0, {self.vocab_size}), got {eos_token_id}"
            )
        object.__setattr__(self, "eos_token_id", eos_token_id)

    @property
    def head_count(self) -> int:
        return (self.max_order - 1) * self.heads_per_order


@dataclass(frozen=True, kw_only=True)
class Binding:
    """Caller-owned PLE hash inputs, output, and scratch views.

    Token and request metadata are read-only. ``out`` is the mutable result
    buffer; ``request_ids`` is an internal scratch view.
    """

    _state: _HashLayout
    geometry: GeometryTensors
    scratch: torch.Tensor
    token_ids: torch.Tensor
    query_start_loc: torch.Tensor
    committed_history: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor
    out: torch.Tensor
    request_ids: torch.Tensor
    plan: Plan | None = None


@dataclass(frozen=True)
class _ScratchLayout:
    nbytes: int
    request_ids_offset_bytes: int


@dataclass(frozen=True, kw_only=True)
class _HashLayout:
    """Immutable host hash geometry and caller-allocated scratch contract."""

    caps: Caps
    geometry: Geometry
    layout: _ScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    config: PleHashConfig

    @property
    def head_count(self) -> int:
        return self.caps.head_count

    @property
    def padded_vocab_size(self):
        return self.geometry.padded_vocab_size

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)





def _materialize_layout(caps: Caps, geometry: Geometry, config: PleHashConfig) -> _HashLayout:

    request_ids_offset_bytes = align_up(0, SCRATCH_ALIGN_BYTES)
    cursor = request_ids_offset_bytes + caps.max_tokens * dtype_nbytes(torch.int32)
    layout = _ScratchLayout(
        nbytes=cursor,
        request_ids_offset_bytes=request_ids_offset_bytes,
    )
    spec = scratch_buffer_spec("ple_hash", nbytes=cursor, device=caps.device)
    return _HashLayout(
        caps=caps,
        geometry=geometry,
        layout=layout,
        _scratch_specs=(spec,),
        config=config,
    )


def plan(
    caps: Caps, *, geometry: Geometry | None = None,
    prime_sizes: torch.Tensor | None = None, table_offsets: torch.Tensor | None = None,
    multipliers: torch.Tensor | None = None, invocation: FrozenMapping = FrozenMapping(),
    override: PleHashConfig | None = None,
) -> Plan:
    """Declare hashing while retaining the caller's checkpoint tensor references."""
    from ._preparation import make_plan
    if not isinstance(caps, Caps):
        raise TypeError("caps must be Caps")
    return make_plan(
        caps, geometry=geometry, prime_sizes=prime_sizes, table_offsets=table_offsets,
        multipliers=multipliers, invocation=invocation, override=override,
    )


def bind(plan: Plan, **kwargs) -> Binding:
    state = require_prepared(plan, "sequence.ple_hash")
    return state.bind(_plan=plan, **kwargs)


def _bind(
    plan: _HashLayout,
    *,
    _geometry: GeometryTensors,
    _plan: Plan | None = None,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
) -> Binding:
    """Bind fixed-capacity hash tensors without allocating."""
    if not isinstance(plan, _HashLayout):
        raise TypeError("hash binding requires its prepared layout")
    caps = plan.caps
    scratch_storage = scratch_tensor(scratch, plan.scratch_specs(), owner="PLE hash")
    request_ids, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.layout.request_ids_offset_bytes,
        shape=(caps.max_tokens,),
        dtype=torch.int32,
    )
    _require_tensor(
        "token_ids",
        token_ids,
        shape=(caps.max_tokens,),
        dtype=torch.int64,
        device=caps.device,
    )
    _require_tensor(
        "query_start_loc",
        query_start_loc,
        shape=(caps.max_seqs + 1,),
        dtype=torch.int32,
        device=caps.device,
    )
    _require_tensor(
        "committed_history",
        committed_history,
        shape=(caps.max_seqs, caps.max_order - 1),
        dtype=torch.int64,
        device=caps.device,
    )
    for name, tensor in (("num_seqs", num_seqs), ("num_tokens", num_tokens)):
        _require_tensor(
            name,
            tensor,
            shape=(1,),
            dtype=torch.int32,
            device=caps.device,
        )
    _require_tensor(
        "out",
        out,
        shape=(caps.max_tokens, caps.head_count),
        dtype=torch.int64,
        device=caps.device,
    )
    _require_mutation_alias_contract(
        mutable=(("scratch", scratch_storage), ("out", out)),
        read_only=(
            ("token_ids", token_ids),
            ("query_start_loc", query_start_loc),
            ("committed_history", committed_history),
            ("num_seqs", num_seqs),
            ("num_tokens", num_tokens),
            ("multipliers", _geometry.multipliers),
            ("prime_sizes", _geometry.prime_sizes),
            ("table_offsets", _geometry.table_offsets),
        ),
    )
    return Binding(
        _state=plan,
        geometry=_geometry,
        plan=_plan,
        scratch=scratch_storage,
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        out=out,
        request_ids=request_ids,
    )


def run(binding: Binding) -> torch.Tensor:
    """Hash packed live rows; padding is written as ``-1``."""
    if binding._state.caps.device.type != "cuda":
        raise ValueError(
            "PLE hash GPU run requires CUDA; use the explicit reference oracle"
        )
    from ._kernels import run_hash_kernel

    run_hash_kernel(binding)
    return binding.out


__all__ = ["Caps", "Plan", "Binding", "plan", "bind", "run"]
