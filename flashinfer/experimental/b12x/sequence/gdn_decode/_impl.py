"""Capacity planning and binding for GDN decode."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared
from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from ._tuning import GdnConfig


GateActivation = Literal["silu", "sigmoid"]


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def _positive(name: str, value: int) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Serving capacity, recurrent-state geometry, and gate policy."""

    device: torch.device | str
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    key_heads: int
    value_heads: int
    key_head_dim: int = 128
    value_head_dim: int = 128
    state_index_columns: int = 1
    model_dtype: torch.dtype = torch.bfloat16
    state_dtype: torch.dtype = torch.float32
    gate_activation: GateActivation = "silu"
    qk_l2norm: bool = True
    null_state_index: int | None = None
    recover_speculative_state: bool = False

    def __post_init__(self) -> None:
        device = _canonical_device(self.device)
        if device.type != "cuda":
            raise ValueError(f"GDN decode requires a CUDA device, got {device}")
        for name in (
            "max_tokens",
            "max_seqs",
            "max_state_slots",
            "key_heads",
            "value_heads",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        key_head_dim = _positive("key_head_dim", self.key_head_dim)
        value_head_dim = _positive("value_head_dim", self.value_head_dim)
        if key_head_dim != 128 or value_head_dim != 128:
            raise ValueError(
                "GDN decode requires key_head_dim=value_head_dim=128, got "
                f"{key_head_dim}/{value_head_dim}"
            )
        columns = _positive("state_index_columns", self.state_index_columns)
        if columns > 8:
            raise ValueError(f"state_index_columns must be at most 8, got {columns}")
        if self.max_tokens > self.max_seqs * columns:
            raise ValueError(
                "max_tokens must fit max_seqs * state_index_columns, got "
                f"{self.max_tokens} > {self.max_seqs} * {columns}"
            )
        if self.value_heads % self.key_heads:
            raise ValueError("value_heads must be divisible by key_heads")
        if self.recover_speculative_state and (
            self.key_heads != self.value_heads
            or self.state_dtype != torch.float32
            or not self.qk_l2norm
        ):
            raise ValueError("KDA recovery requires equal heads, FP32 state and Q/K normalization")
        if self.model_dtype != torch.bfloat16:
            raise TypeError(
                f"model_dtype must be torch.bfloat16, got {self.model_dtype}"
            )
        if self.state_dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(
                "state_dtype must be torch.bfloat16 or torch.float32, got "
                f"{self.state_dtype}"
            )
        if self.gate_activation not in ("silu", "sigmoid"):
            raise ValueError(
                "gate_activation must be 'silu' or 'sigmoid', got "
                f"{self.gate_activation!r}"
            )
        null_state_index = self.null_state_index
        if null_state_index is not None:
            if isinstance(null_state_index, bool):
                raise TypeError("null_state_index must be an integer or None")
            null_state_index = int(null_state_index)
            if not -(1 << 63) <= null_state_index < (1 << 63):
                raise ValueError("null_state_index must fit in a signed 64-bit integer")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "key_head_dim", key_head_dim)
        object.__setattr__(self, "value_head_dim", value_head_dim)
        object.__setattr__(self, "state_index_columns", columns)
        object.__setattr__(self, "qk_l2norm", bool(self.qk_l2norm))
        object.__setattr__(self, "null_state_index", null_state_index)

    @property
    def value_heads_per_key_head(self) -> int:
        return self.value_heads // self.key_heads

    @property
    def packed_qkv_width(self) -> int:
        return (
            2 * self.key_heads * self.key_head_dim
            + self.value_heads * self.value_head_dim
        )


@dataclass(frozen=True)
class _GdnLayout:
    """Immutable GDN geometry and caller-allocated scratch contract."""

    caps: Caps
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    config: GdnConfig
    recurrent_block_k: int = 128
    recurrent_num_warps: int = 1
    norm_block: int = 128
    norm_num_warps: int = 4

    @property
    def recurrent_block_v(self) -> int:
        return self.config.recurrent_block_v

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def output_shape(self, tokens: int | None = None) -> tuple[int, int, int]:
        live_tokens = self.caps.max_tokens if tokens is None else int(tokens)
        if live_tokens < 0 or live_tokens > self.caps.max_tokens:
            raise ValueError(
                f"tokens={live_tokens} exceeds capacity {self.caps.max_tokens}"
            )
        return (live_tokens, self.caps.value_heads, self.caps.value_head_dim)



@dataclass(frozen=True)
class Binding:
    """Caller-owned GDN inputs, recurrent state, output, and scratch.

    ``recurrent_state`` is updated in place. All projection and norm tensors
    are read-only; ``output`` is the caller-owned result buffer.
    """

    _state: _GdnLayout
    scratch: torch.Tensor
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    z: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    recurrent_state: torch.Tensor
    query_start_loc: torch.Tensor
    num_accepted_tokens: torch.Tensor
    state_indices: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor
    output: torch.Tensor
    plan: Plan | None = None


@dataclass(frozen=True)
class KdaBinding:
    """Complete caller-owned invocation for lower-bounded KDA decode.

    ``raw_g`` is the unactivated per-key-coordinate forget gate and
    ``raw_beta`` is the unactivated scalar update gate for each head.
    """

    _state: _GdnLayout
    scratch: torch.Tensor
    mixed_qkv: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    z: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    recurrent_state: torch.Tensor
    query_start_loc: torch.Tensor
    num_accepted_tokens: torch.Tensor
    state_indices: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor
    output: torch.Tensor
    plan: Plan | None = None
    correction_cache: torch.Tensor | None = None
    kg_cache: torch.Tensor | None = None


def _scratch_layout(caps: Caps, *, config: GdnConfig) -> _GdnLayout:
    # No kernel of the family writes scratch; the buffer is the plan's
    # caller-owned workspace contract at its minimum size.
    spec = scratch_buffer_spec("gdn_decode", nbytes=0, device=caps.device)
    return _GdnLayout(caps=caps, _scratch_specs=(spec,), config=config)


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind caller tensors to a fully prepared GDN plan."""
    state = require_prepared(plan, "attention.gdn")
    return state.bind(_plan=plan, **kwargs)


def bind_kda(plan: Plan, **kwargs) -> KdaBinding:
    """Bind caller tensors to a fully prepared KDA plan."""
    state = require_prepared(plan, "attention.gdn")
    return state.bind_kda(_plan=plan, **kwargs)


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
    contiguous: bool = True,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype not in dtypes:
        expected = " or ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must have dtype {expected}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_paged_recurrent_state(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _require_tensor(
        "recurrent_state",
        tensor,
        shape=shape,
        device=device,
        dtypes=(dtype,),
        contiguous=False,
    )
    _, value_heads, value_dim, key_dim = shape
    slot_elements = value_heads * value_dim * key_dim
    expected_inner_strides = (value_dim * key_dim, key_dim, 1)
    if tuple(tensor.stride()[1:]) != expected_inner_strides:
        raise ValueError(
            "recurrent_state must be contiguous within each state slot; "
            f"expected inner strides {expected_inner_strides}, got "
            f"{tuple(tensor.stride()[1:])}"
        )
    if tensor.stride(0) < slot_elements:
        raise ValueError(
            "recurrent_state slots must not overlap; expected outer stride at "
            f"least {slot_elements}, got {tensor.stride(0)}"
        )


def _require_row_contiguous(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
) -> None:
    _require_tensor(
        name,
        tensor,
        shape=shape,
        device=device,
        dtypes=dtypes,
        contiguous=False,
    )
    expected_inner_strides = []
    stride = 1
    for size in reversed(shape[1:]):
        expected_inner_strides.append(stride)
        stride *= size
    expected_inner = tuple(reversed(expected_inner_strides))
    if tuple(tensor.stride()[1:]) != expected_inner:
        raise ValueError(
            f"{name} must be contiguous within each token row; expected inner "
            f"strides {expected_inner}, got {tuple(tensor.stride()[1:])}"
        )
    if tensor.stride(0) < stride:
        raise ValueError(
            f"{name} token rows must not overlap; expected outer stride at "
            f"least {stride}, got {tensor.stride(0)}"
        )


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    element_size = int(tensor.element_size())
    min_element = max_element = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if size == 0:
            storage = int(tensor.untyped_storage().data_ptr())
            start = storage + min_element * element_size
            return start, start
        extent = (int(size) - 1) * int(stride)
        min_element += min(0, extent)
        max_element += max(0, extent)
    storage = int(tensor.untyped_storage().data_ptr())
    return (
        storage + min_element * element_size,
        storage + (max_element + 1) * element_size,
    )


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


def _paged_overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Recognize disjoint state/record fields in one interleaved page pool."""
    if not _overlaps(left, right):
        return False
    step = left.stride(0) * left.element_size()
    if left.shape[0] != right.shape[0] or step != right.stride(0) * right.element_size():
        return True
    left_bytes = math.prod(left.shape[1:]) * left.element_size()
    right_bytes = math.prod(right.shape[1:]) * right.element_size()
    delta = (right.data_ptr() - left.data_ptr()) % step
    return delta < left_bytes or step - delta < right_bytes


def _bind(
    plan: _GdnLayout,
    *,
    _plan: Plan | None = None,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
) -> Binding:
    """Bind Qwen GDN tensors without allocating runtime storage.

    ``recurrent_state`` uses the optimized physical layout
    ``[slot, value_head, value_dim, key_dim]``. Each slot must be contiguous,
    but the outer slot stride may include page-alignment padding. Binding uses
    that stride directly without copying the caller-owned cache. Slow
    mathematical references commonly use its transpose,
    ``[batch, head, key_dim, value_dim]``.

    Projection, metadata, and output tensors may use any positive leading
    capacity within the plan. Projection rows may be views into wider packed
    tensors; their innermost dimensions must remain contiguous and rows must
    not overlap. All live tensors for one invocation must be bound together
    before :func:`run`.
    """
    if not isinstance(plan, _GdnLayout):
        raise TypeError("GDN binding requires its materialized scratch layout")
    caps = plan.caps
    scratch_storage = scratch_tensor(scratch, plan.scratch_specs(), owner="GDN decode")
    model = (caps.model_dtype,)
    parameter = (torch.bfloat16, torch.float32)
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"mixed_qkv must have two dimensions, got shape {tuple(mixed_qkv.shape)}"
        )
    token_capacity = _positive("mixed_qkv token capacity", mixed_qkv.shape[0])
    if token_capacity > caps.max_tokens:
        raise ValueError(
            f"mixed_qkv token capacity {token_capacity} exceeds planned "
            f"capacity {caps.max_tokens}"
        )
    if state_indices.ndim != 2:
        raise ValueError(
            "state_indices must have two dimensions, got "
            f"shape {tuple(state_indices.shape)}"
        )
    sequence_capacity = _positive(
        "state_indices sequence capacity", state_indices.shape[0]
    )
    state_index_columns = _positive(
        "state_indices column capacity", state_indices.shape[1]
    )
    if sequence_capacity > caps.max_seqs:
        raise ValueError(
            f"state_indices sequence capacity {sequence_capacity} exceeds "
            f"planned capacity {caps.max_seqs}"
        )
    if state_index_columns > caps.state_index_columns:
        raise ValueError(
            f"state_indices column capacity {state_index_columns} exceeds "
            f"planned capacity {caps.state_index_columns}"
        )
    if token_capacity > sequence_capacity * state_index_columns:
        raise ValueError(
            "token capacity must fit the bound packed metadata geometry, got "
            f"{token_capacity} > {sequence_capacity} * {state_index_columns}"
        )
    _require_row_contiguous(
        "mixed_qkv",
        mixed_qkv,
        shape=(token_capacity, caps.packed_qkv_width),
        device=caps.device,
        dtypes=model,
    )
    for name, tensor in (("a", a), ("b", b)):
        _require_row_contiguous(
            name,
            tensor,
            shape=(token_capacity, caps.value_heads),
            device=caps.device,
            dtypes=model,
        )
    _require_row_contiguous(
        "z",
        z,
        shape=(token_capacity, caps.value_heads, caps.value_head_dim),
        device=caps.device,
        dtypes=model,
    )
    for name, tensor in (("A_log", A_log), ("dt_bias", dt_bias)):
        _require_tensor(
            name,
            tensor,
            shape=(caps.value_heads,),
            device=caps.device,
            dtypes=parameter,
        )
    _require_tensor(
        "norm_weight",
        norm_weight,
        shape=(caps.value_head_dim,),
        device=caps.device,
        dtypes=(torch.bfloat16, torch.float32),
    )
    _require_paged_recurrent_state(
        recurrent_state,
        shape=(
            caps.max_state_slots,
            caps.value_heads,
            caps.value_head_dim,
            caps.key_head_dim,
        ),
        device=caps.device,
        dtype=caps.state_dtype,
    )
    _require_tensor(
        "query_start_loc",
        query_start_loc,
        shape=(sequence_capacity + 1,),
        device=caps.device,
        dtypes=(torch.int32,),
    )
    _require_tensor(
        "num_accepted_tokens",
        num_accepted_tokens,
        shape=(sequence_capacity,),
        device=caps.device,
        dtypes=(torch.int32,),
    )
    _require_tensor(
        "state_indices",
        state_indices,
        shape=(sequence_capacity, state_index_columns),
        device=caps.device,
        dtypes=(torch.int32, torch.int64),
        contiguous=False,
    )
    for name, tensor in (("num_seqs", num_seqs), ("num_tokens", num_tokens)):
        _require_tensor(
            name,
            tensor,
            shape=(1,),
            device=caps.device,
            dtypes=(torch.int32,),
        )
    _require_row_contiguous(
        "output",
        output,
        shape=(token_capacity, caps.value_heads, caps.value_head_dim),
        device=caps.device,
        dtypes=model,
    )
    mutable = (
        ("scratch", scratch_storage),
        ("recurrent_state", recurrent_state),
        ("output", output),
    )
    read_only = (
        ("mixed_qkv", mixed_qkv),
        ("a", a),
        ("b", b),
        ("z", z),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
        ("norm_weight", norm_weight),
        ("query_start_loc", query_start_loc),
        ("num_accepted_tokens", num_accepted_tokens),
        ("state_indices", state_indices),
        ("num_seqs", num_seqs),
        ("num_tokens", num_tokens),
    )
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
    return Binding(
        _state=plan,
        plan=_plan,
        scratch=scratch_storage,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        z=z,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        recurrent_state=recurrent_state,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        state_indices=state_indices,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        output=output,
    )


def _bind_kda(
    plan: _GdnLayout,
    *,
    _plan: Plan | None = None,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
    correction_cache: torch.Tensor | None = None,
    kg_cache: torch.Tensor | None = None,
) -> KdaBinding:
    """Bind lower-bounded KDA tensors to a GDN decode plan.

    KDA uses one Q/K/V head per recurrent state head. Projection, metadata,
    and output tensors may use any positive leading capacity within the plan.
    All live tensors for one invocation must be bound together before
    :func:`run_kda`; binding neither copies nor stages their contents.
    """
    if not isinstance(plan, _GdnLayout):
        raise TypeError("KDA binding requires its materialized scratch layout")
    caps = plan.caps
    if caps.key_heads != caps.value_heads:
        raise ValueError(
            "KDA decode requires key_heads=value_heads, got "
            f"{caps.key_heads}/{caps.value_heads}"
        )
    if caps.gate_activation != "sigmoid":
        raise ValueError(
            "KDA decode requires gate_activation='sigmoid', got "
            f"{caps.gate_activation!r}"
        )

    scratch_storage = scratch_tensor(scratch, plan.scratch_specs(), owner="KDA decode")
    model = (caps.model_dtype,)
    parameter = (torch.bfloat16, torch.float32)
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"mixed_qkv must have two dimensions, got shape {tuple(mixed_qkv.shape)}"
        )
    token_capacity = _positive("mixed_qkv token capacity", mixed_qkv.shape[0])
    if token_capacity > caps.max_tokens:
        raise ValueError(
            f"mixed_qkv token capacity {token_capacity} exceeds planned "
            f"capacity {caps.max_tokens}"
        )
    if state_indices.ndim != 2:
        raise ValueError(
            "state_indices must have two dimensions, got "
            f"shape {tuple(state_indices.shape)}"
        )
    sequence_capacity = _positive(
        "state_indices sequence capacity", state_indices.shape[0]
    )
    state_index_columns = _positive(
        "state_indices column capacity", state_indices.shape[1]
    )
    if sequence_capacity > caps.max_seqs:
        raise ValueError(
            f"state_indices sequence capacity {sequence_capacity} exceeds "
            f"planned capacity {caps.max_seqs}"
        )
    if state_index_columns > caps.state_index_columns:
        raise ValueError(
            f"state_indices column capacity {state_index_columns} exceeds "
            f"planned capacity {caps.state_index_columns}"
        )
    query_columns = caps.state_index_columns if caps.recover_speculative_state else state_index_columns
    if token_capacity > sequence_capacity * query_columns:
        raise ValueError(
            "token capacity must fit the bound packed metadata geometry, got "
            f"{token_capacity} > {sequence_capacity} * {query_columns}"
        )
    _require_row_contiguous(
        "mixed_qkv",
        mixed_qkv,
        shape=(token_capacity, caps.packed_qkv_width),
        device=caps.device,
        dtypes=model,
    )
    _require_row_contiguous(
        "raw_g",
        raw_g,
        shape=(token_capacity, caps.value_heads, caps.key_head_dim),
        device=caps.device,
        dtypes=model,
    )
    _require_tensor(
        "raw_beta",
        raw_beta,
        shape=(token_capacity, caps.value_heads),
        device=caps.device,
        dtypes=model,
        contiguous=False,
    )
    _require_row_contiguous(
        "z",
        z,
        shape=(token_capacity, caps.value_heads, caps.value_head_dim),
        device=caps.device,
        dtypes=model,
    )
    _require_tensor(
        "A_log",
        A_log,
        shape=(caps.value_heads,),
        device=caps.device,
        dtypes=parameter,
    )
    _require_tensor(
        "dt_bias",
        dt_bias,
        shape=(caps.value_heads, caps.key_head_dim),
        device=caps.device,
        dtypes=parameter,
    )
    _require_tensor(
        "norm_weight",
        norm_weight,
        shape=(caps.value_head_dim,),
        device=caps.device,
        dtypes=parameter,
    )
    _require_paged_recurrent_state(
        recurrent_state,
        shape=(
            caps.max_state_slots,
            caps.value_heads,
            caps.value_head_dim,
            caps.key_head_dim,
        ),
        device=caps.device,
        dtype=caps.state_dtype,
    )
    _require_tensor(
        "query_start_loc",
        query_start_loc,
        shape=(sequence_capacity + 1,),
        device=caps.device,
        dtypes=(torch.int32,),
    )
    _require_tensor(
        "num_accepted_tokens",
        num_accepted_tokens,
        shape=(sequence_capacity,),
        device=caps.device,
        dtypes=(torch.int32,),
    )
    _require_tensor(
        "state_indices",
        state_indices,
        shape=(sequence_capacity, state_index_columns),
        device=caps.device,
        dtypes=(torch.int32, torch.int64),
        contiguous=False,
    )
    for name, tensor in (("num_seqs", num_seqs), ("num_tokens", num_tokens)):
        _require_tensor(
            name,
            tensor,
            shape=(1,),
            device=caps.device,
            dtypes=(torch.int32,),
        )
    _require_row_contiguous(
        "output",
        output,
        shape=(token_capacity, caps.value_heads, caps.value_head_dim),
        device=caps.device,
        dtypes=model,
    )
    mutable = (
        ("scratch", scratch_storage),
        ("recurrent_state", recurrent_state),
        ("output", output),
    )
    if caps.recover_speculative_state:
        for name, tensor, dtype, dim in (
            ("correction_cache", correction_cache, torch.float32, 128),
            ("kg_cache", kg_cache, caps.model_dtype, 256),
        ):
            if tensor is None:
                raise ValueError(f"KDA recovery requires {name}")
            _require_tensor(
                name, tensor,
                shape=(caps.max_state_slots, caps.value_heads, caps.state_index_columns, dim),
                device=caps.device, dtypes=(dtype,), contiguous=False,
            )
            if tensor.stride()[1:] != (caps.state_index_columns * dim, dim, 1):
                raise ValueError(f"{name} requires contiguous per-slot records")
            if tensor.stride(0) < caps.value_heads * caps.state_index_columns * dim:
                raise ValueError(f"{name} slots must not overlap")
            mutable += ((name, tensor),)
    elif correction_cache is not None or kg_cache is not None:
        raise ValueError("recovery records require recover_speculative_state=True")
    read_only = (
        ("mixed_qkv", mixed_qkv),
        ("raw_g", raw_g),
        ("raw_beta", raw_beta),
        ("z", z),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
        ("norm_weight", norm_weight),
        ("query_start_loc", query_start_loc),
        ("num_accepted_tokens", num_accepted_tokens),
        ("state_indices", state_indices),
        ("num_seqs", num_seqs),
        ("num_tokens", num_tokens),
    )
    paged_names = {"recurrent_state", "correction_cache", "kg_cache"}
    for index, (left_name, left) in enumerate(mutable):
        for right_name, right in mutable[index + 1 :]:
            overlap = _paged_overlaps if {left_name, right_name} <= paged_names else _overlaps
            if overlap(left, right):
                raise ValueError(
                    f"mutable buffers {left_name} and {right_name} must not overlap"
                )
        for right_name, right in read_only:
            if _overlaps(left, right):
                raise ValueError(
                    f"mutable buffer {left_name} must not overlap read-only "
                    f"tensor {right_name}"
                )
    return KdaBinding(
        _state=plan,
        plan=_plan,
        scratch=scratch_storage,
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        raw_beta=raw_beta,
        z=z,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        recurrent_state=recurrent_state,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        state_indices=state_indices,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        output=output,
        correction_cache=correction_cache,
        kg_cache=kg_cache,
    )


def run(
    binding: Binding,
    *,
    eps: float = 1e-6,
    scale: float | None = None,
) -> torch.Tensor:
    """Run packed per-request decode and return caller-owned output capacity.

    Request ``r`` consumes the packed token interval described by
    ``query_start_loc[r:r+2]``. Its initial recurrent checkpoint comes from
    ``state_indices[r, num_accepted_tokens[r] - 1]``. Each token executes
    sequentially and writes its post-token checkpoint to the corresponding
    state-index column. A one-column plan with one token per request is ordinary
    decode.

    Packed metadata is trusted: counts, sequence bounds, accepted-token counts,
    and state slots must be valid, and active state-index cells must be unique.
    If configured, a null initial checkpoint zeroes that request's output
    without reading recurrent state, and null destination cells are not
    written. Capacity rows beyond ``num_tokens`` are zeroed.
    """
    if not isinstance(binding, Binding):
        raise TypeError(f"binding must be Binding, got {type(binding)!r}")
    if binding.plan is None:
        raise TypeError("GDN run requires a session-prepared binding")
    caps = binding._state.caps
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    scale_value = caps.key_head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale_value}")
    from ._kernels import run_gdn_decode

    run_gdn_decode(
        binding.mixed_qkv,
        binding.a,
        binding.b,
        binding.z,
        binding.A_log,
        binding.dt_bias,
        binding.norm_weight,
        binding.recurrent_state,
        binding.query_start_loc,
        binding.num_accepted_tokens,
        binding.state_indices,
        binding.num_seqs,
        binding.num_tokens,
        binding.output,
        eps=eps_value,
        scale=scale_value,
        lower_bound=0.0,
        plan=binding.plan,
    )
    return binding.output


def run_kda(
    binding: KdaBinding,
    *,
    lower_bound: float = -5.0,
    eps: float = 1e-6,
    scale: float | None = None,
    apply_output_norm: bool = True,
) -> torch.Tensor:
    """Run packed lower-bounded KDA decode into caller-owned buffers."""
    if not isinstance(binding, KdaBinding):
        raise TypeError(f"binding must be KdaBinding, got {type(binding)!r}")
    lower_bound_value = float(lower_bound)
    if not math.isfinite(lower_bound_value) or lower_bound_value >= 0.0:
        raise ValueError(
            f"lower_bound must be finite and negative, got {lower_bound_value}"
        )
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    if binding.plan is None:
        raise TypeError("KDA run requires a session-prepared binding")
    if binding._state.caps.recover_speculative_state:
        state = require_prepared(binding.plan, "attention.gdn")
        return state.run(binding, eps=eps_value, scale=scale, lower_bound=lower_bound_value,
                         apply_output_norm=apply_output_norm)
    if not apply_output_norm:
        raise ValueError("raw output is supported only by KDA state recovery")
    caps = binding._state.caps
    scale_value = caps.key_head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale_value}")
    from ._kernels import run_gdn_decode

    run_gdn_decode(
        binding.mixed_qkv,
        binding.raw_g,
        binding.raw_beta,
        binding.z,
        binding.A_log,
        binding.dt_bias,
        binding.norm_weight,
        binding.recurrent_state,
        binding.query_start_loc,
        binding.num_accepted_tokens,
        binding.state_indices,
        binding.num_seqs,
        binding.num_tokens,
        binding.output,
        eps=eps_value,
        scale=scale_value,
        lower_bound=lower_bound_value,
        plan=binding.plan,
    )
    return binding.output


__all__ = [
    "Binding",
    "Caps",
    "KdaBinding",
    "bind",
    "bind_kda",
    "run",
    "run_kda",
]
