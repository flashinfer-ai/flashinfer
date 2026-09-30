"""Capacity planning and binding for fused PLE hash and embedding lookup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
from b12x.sequence import ple_hash

from b12x.sequence.ple_hash._contracts import _HashLayout, _materialize_layout as _hash_layout
from b12x.sequence.ple_hash._tuning import PleHashConfig
from b12x.sequence.ple_hash.geometry import Geometry, compute_geometry
from ._tuning import PleEmbeddingConfig

if TYPE_CHECKING:
    from ._disk import DiskTable
    from ._storage import TableStorage


_SIGNED_INT64_MAX = (1 << 63) - 1
QuantMode = Literal["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"]
TableMemory = Literal["device", "mapped_host", "io_uring"]
_BF16_MODE: QuantMode = "bf16"
_FP8_QUANT_MODE: QuantMode = "fp8_e4m3_per_tensor"
_NVFP4_QUANT_MODE: QuantMode = "nvfp4_group16"
_SUPPORTED_MODES: tuple[QuantMode, ...] = (
    _BF16_MODE,
    _FP8_QUANT_MODE,
    _NVFP4_QUANT_MODE,
)
_SUPPORTED_TABLE_MEMORY: tuple[TableMemory, ...] = (
    "device",
    "mapped_host",
    "io_uring",
)


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


def _require_mapped_host_tensor(
    name: str, tensor: torch.Tensor, *, device: torch.device
) -> None:
    from cuda.bindings import runtime as cudart

    with torch.cuda.device(device):
        error, attributes = cudart.cudaPointerGetAttributes(tensor.data_ptr())
    if error != cudart.cudaError_t.cudaSuccess:
        raise ValueError(
            f"{name} must use CUDA-mapped host storage; pointer query failed: {error}"
        )
    if attributes.type != cudart.cudaMemoryType.cudaMemoryTypeHost:
        raise ValueError(
            f"{name} must use CUDA-mapped host storage, got memory type "
            f"{attributes.type}"
        )
    device_pointer = int(attributes.devicePointer or 0)
    if device_pointer == 0 or device_pointer != tensor.data_ptr():
        raise ValueError(f"{name} does not expose the mapped device alias on {device}")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Serving capacity, hash geometry, and persistent table storage policy.

    ``quant_mode`` selects BF16 rows, FP8 E4M3 rows with one BF16
    ``weight_scale``, or packed group-16 NVFP4 rows with FP8 E4M3
    ``weight_scale`` blocks and one FP32 ``weight_scale_2`` global scale.
    ``scale_dtype`` is validated against the selected storage format.
    ``table_memory="mapped_host"`` places row payloads and row-associated
    scales in CUDA-mapped, write-combined host memory while scalar scales stay
    device-resident. ``table_memory="io_uring"`` reads selected rows with
    O_DIRECT into a batch-sized mapped-host cache owned by ``DiskTable``;
    its ``run`` must execute outside graph capture.
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
    embedding_dim: int
    tp_size: int
    tp_rank: int
    table_alignment: int = 128
    quant_mode: QuantMode = _FP8_QUANT_MODE
    table_memory: TableMemory = "device"
    scale_dtype: torch.dtype | None = None
    output_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        for name in (
            "max_tokens",
            "max_seqs",
            "vocab_size",
            "heads_per_order",
            "base_table_size",
            "embedding_dim",
            "tp_size",
            "table_alignment",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
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
        if eos_token_id < 0 or eos_token_id >= self.vocab_size:
            raise ValueError(
                f"eos_token_id must be in [0, {self.vocab_size}), got {eos_token_id}"
            )
        object.__setattr__(self, "eos_token_id", eos_token_id)
        tp_rank = int(self.tp_rank)
        if tp_rank < 0 or tp_rank >= self.tp_size:
            raise ValueError(f"tp_rank must be in [0, {self.tp_size}), got {tp_rank}")
        object.__setattr__(self, "tp_rank", tp_rank)
        if self.embedding_dim % self.head_count:
            raise ValueError(
                f"embedding_dim={self.embedding_dim} must be divisible by "
                f"head_count={self.head_count}"
            )
        quant_mode = str(self.quant_mode)
        if quant_mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"quant_mode must be one of {_SUPPORTED_MODES!r}, "
                f"got {self.quant_mode!r}"
            )
        object.__setattr__(self, "quant_mode", quant_mode)
        table_memory = str(self.table_memory)
        if table_memory not in _SUPPORTED_TABLE_MEMORY:
            raise ValueError(
                f"table_memory must be one of {_SUPPORTED_TABLE_MEMORY!r}, "
                f"got {self.table_memory!r}"
            )
        if table_memory != "device" and self.device.type != "cuda":
            raise ValueError(
                f"{table_memory} PLE table storage requires a CUDA device, "
                f"got {self.device}"
            )
        object.__setattr__(self, "table_memory", table_memory)
        scale_dtype = self.scale_dtype
        if quant_mode == _BF16_MODE:
            if scale_dtype is not None:
                raise TypeError(
                    "BF16 PLE storage has no dequantization scale; "
                    f"scale_dtype must be None, got {scale_dtype}"
                )
        elif quant_mode == _FP8_QUANT_MODE:
            scale_dtype = torch.bfloat16 if scale_dtype is None else scale_dtype
            if scale_dtype != torch.bfloat16:
                raise TypeError(
                    "FP8 per-table PLE scale must use torch.bfloat16, got "
                    f"{scale_dtype}"
                )
        else:
            scale_dtype = torch.float8_e4m3fn if scale_dtype is None else scale_dtype
            if scale_dtype != torch.float8_e4m3fn:
                raise TypeError(
                    "NVFP4 group-16 PLE scale must use torch.float8_e4m3fn, got "
                    f"{scale_dtype}"
                )
            if self.head_dim % 16:
                raise ValueError(
                    f"NVFP4 group-16 PLE head_dim={self.head_dim} must be "
                    "divisible by 16"
                )
        object.__setattr__(self, "scale_dtype", scale_dtype)
        if self.output_dtype != torch.bfloat16:
            raise TypeError(
                f"PLE embedding output must use torch.bfloat16, got {self.output_dtype}"
            )

    @property
    def head_count(self) -> int:
        return (self.max_order - 1) * self.heads_per_order

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count


@dataclass(frozen=True)
class _ScratchLayout:
    nbytes: int
    hash_scratch_offset_bytes: int
    hash_scratch_nbytes: int
    ids_offset_bytes: int


@dataclass(frozen=True, kw_only=True)
class TableLayout:
    """Host-only TP table geometry, storage shapes, and scratch contract.

    ``weight_scale`` and ``weight_scale_2`` retain checkpoint-compatible names.
    For FP8, ``weight_scale`` is the BF16 per-table scale and
    ``weight_scale_2`` is absent. For NVFP4, ``weight_scale`` contains one FP8
    E4M3 block scale per group of 16 values and ``weight_scale_2`` is the FP32
    global scale.
    """

    caps: Caps
    geometry: Geometry
    table_vocab_size: int
    padded_vocab_size: int
    shard_start: int
    shard_end: int
    weight_shape: tuple[int, int]
    weight_dtype: torch.dtype
    weight_scale_shape: tuple[int, ...] | None
    weight_scale_dtype: torch.dtype | None
    weight_scale_2_shape: tuple[int, ...] | None
    weight_scale_2_dtype: torch.dtype | None
    output_shape: tuple[int, int]
    output_dtype: torch.dtype
    _ids_shape: tuple[int, int]
    _layout: _ScratchLayout
    _hash_layout: _HashLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    @property
    def head_count(self) -> int:
        return self.caps.head_count

    @property
    def head_dim(self) -> int:
        return self.caps.head_dim

    @property
    def scale_shape(self) -> tuple[int, ...] | None:
        """Alias for the primary dequantization-scale storage shape."""
        return self.weight_scale_shape

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def allocate_storage(self) -> TableStorage:
        """Allocate persistent table tensors according to ``caps.table_memory``."""
        from ._storage import allocate_storage

        return allocate_storage(self)


@dataclass(frozen=True, kw_only=True)
class Binding:
    """Caller-owned fused hash, gather, and inline-dequantization tensors.

    Persistent weights and scales are read-only. ``out`` is the mutable
    caller-owned result buffer; hash IDs and hash scratch remain private
    implementation details.
    """

    _state: TableLayout
    scratch: torch.Tensor
    weight: torch.Tensor | None
    weight_scale: torch.Tensor | None
    weight_scale_2: torch.Tensor | None
    token_ids: torch.Tensor
    query_start_loc: torch.Tensor
    committed_history: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor
    out: torch.Tensor
    _ids: torch.Tensor
    _hash_scratch: torch.Tensor
    _hash_binding: ple_hash.Binding
    disk_table: DiskTable | None = None
    plan: Plan | None = None


def plan(
    caps: Caps,
    *,
    geometry: Geometry | None = None,
    prime_sizes: torch.Tensor | None = None,
    table_offsets: torch.Tensor | None = None,
    multipliers: torch.Tensor | None = None,
    invocation: FrozenMapping = FrozenMapping(),
    override: PleEmbeddingConfig | None = None,
) -> Plan:
    """Declare the plan, retaining the actual checkpoint geometry buffers."""
    from ._preparation import make_plan

    if not isinstance(caps, Caps):
        raise TypeError("caps must be Caps")
    return make_plan(
        caps, geometry=geometry, prime_sizes=prime_sizes, table_offsets=table_offsets,
        multipliers=multipliers, invocation=invocation, override=override,
    )


def storage_layout(caps: Caps, *, geometry: Geometry | None = None) -> TableLayout:
    """Calculate table storage before loading, without device allocation."""
    if not isinstance(caps, Caps):
        raise TypeError("caps must be Caps")
    geometry = compute_geometry(caps) if geometry is None else geometry
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be host Geometry")
    validated = compute_geometry(
        caps, prime_sizes=geometry.prime_sizes, table_offsets=geometry.table_offsets,
        multipliers=geometry.multipliers,
    )
    if geometry != validated:
        raise ValueError("inconsistent PLE table geometry")
    hash_caps = ple_hash.Caps(
        device=caps.device, max_tokens=caps.max_tokens, max_seqs=caps.max_seqs,
        vocab_size=caps.vocab_size, eos_token_id=caps.eos_token_id,
        max_order=caps.max_order, heads_per_order=caps.heads_per_order,
        dense_layer_ordinal=caps.dense_layer_ordinal, base_table_size=caps.base_table_size,
        table_alignment=caps.table_alignment,
    )
    hash_layout = _hash_layout(hash_caps, geometry, PleHashConfig(backend="triton"))
    padded_vocab_size = geometry.padded_vocab_size
    if padded_vocab_size % caps.tp_size:
        raise ValueError(
            f"padded_vocab_size={padded_vocab_size} must be divisible by "
            f"tp_size={caps.tp_size}"
        )
    shard_size = padded_vocab_size // caps.tp_size
    shard_start = caps.tp_rank * shard_size
    shard_end = shard_start + shard_size
    if shard_size * caps.head_dim > _SIGNED_INT64_MAX:
        raise ValueError("TP-local PLE weight extent must fit signed int64 indexing")

    if caps.quant_mode == _BF16_MODE:
        weight_shape = (shard_size, caps.head_dim)
        weight_dtype = torch.bfloat16
        weight_scale_shape = None
        weight_scale_dtype = None
        weight_scale_2_shape = None
        weight_scale_2_dtype = None
    elif caps.quant_mode == _FP8_QUANT_MODE:
        weight_shape = (shard_size, caps.head_dim)
        weight_dtype = torch.float8_e4m3fn
        weight_scale_shape = (1,)
        weight_scale_dtype = caps.scale_dtype
        weight_scale_2_shape = None
        weight_scale_2_dtype = None
    else:
        weight_shape = (shard_size, caps.head_dim // 2)
        weight_dtype = torch.uint8
        weight_scale_shape = (shard_size, caps.head_dim // 16)
        weight_scale_dtype = caps.scale_dtype
        weight_scale_2_shape = (1,)
        weight_scale_2_dtype = torch.float32

    hash_spec = hash_layout.scratch_specs()[0]
    hash_scratch_offset_bytes = align_up(0, SCRATCH_ALIGN_BYTES)
    cursor = hash_scratch_offset_bytes + hash_spec.nbytes
    ids_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = ids_offset_bytes + (
        caps.max_tokens * caps.head_count * dtype_nbytes(torch.int64)
    )
    layout = _ScratchLayout(
        nbytes=cursor,
        hash_scratch_offset_bytes=hash_scratch_offset_bytes,
        hash_scratch_nbytes=hash_spec.nbytes,
        ids_offset_bytes=ids_offset_bytes,
    )
    scratch_spec = scratch_buffer_spec(
        "ple_embedding", nbytes=layout.nbytes, device=caps.device
    )
    return TableLayout(
        caps=caps,
        geometry=geometry,
        table_vocab_size=geometry.table_vocab_size,
        padded_vocab_size=padded_vocab_size,
        shard_start=shard_start,
        shard_end=shard_end,
        weight_shape=weight_shape,
        weight_dtype=weight_dtype,
        weight_scale_shape=weight_scale_shape,
        weight_scale_dtype=weight_scale_dtype,
        weight_scale_2_shape=weight_scale_2_shape,
        weight_scale_2_dtype=weight_scale_2_dtype,
        output_shape=(caps.max_tokens, caps.embedding_dim),
        output_dtype=caps.output_dtype,
        _ids_shape=(caps.max_tokens, caps.head_count),
        _layout=layout,
        _hash_layout=hash_layout,
        _scratch_specs=(scratch_spec,),
    )


def bind(plan: Plan, **kwargs) -> Binding:
    state = require_prepared(plan, "sequence.ple_embedding")
    return state.bind(_plan=plan, **kwargs)


def _bind(
    plan: TableLayout,
    *,
    _hash_state,
    _plan: Plan | None = None,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    weight: torch.Tensor | None,
    weight_scale: torch.Tensor | None = None,
    weight_scale_2: torch.Tensor | None = None,
    disk_table: DiskTable | None = None,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
) -> Binding:
    """Bind fixed-capacity hashing and local lookup tensors without allocating."""
    if not isinstance(plan, TableLayout):
        raise TypeError("binding requires a materialized table layout")
    caps = plan.caps
    scratch_storage = scratch_tensor(
        scratch, plan.scratch_specs(), owner="PLE embedding"
    )
    hash_scratch, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan._layout.hash_scratch_offset_bytes,
        shape=(plan._layout.hash_scratch_nbytes,),
        dtype=torch.uint8,
    )
    ids, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan._layout.ids_offset_bytes,
        shape=plan._ids_shape,
        dtype=torch.int64,
    )
    if disk_table is not None and caps.table_memory != "io_uring":
        raise ValueError("disk_table requires table_memory='io_uring'")
    if caps.table_memory == "io_uring":
        from ._disk import DiskTable

        if not isinstance(disk_table, DiskTable) or disk_table.layout != plan:
            raise ValueError("disk binding requires the declared table layout")
        if weight is not None:
            raise ValueError("disk binding requires weight=None")
        disk_table._require_complete()
    else:
        if weight is None:
            raise ValueError("weight is required for resident table storage")
        _require_tensor(
            "weight",
            weight,
            shape=plan.weight_shape,
            dtype=plan.weight_dtype,
            device=caps.device,
        )
        if caps.table_memory == "mapped_host":
            _require_mapped_host_tensor("weight", weight, device=caps.device)
    if plan.weight_scale_shape is None or (
        caps.table_memory == "io_uring" and caps.quant_mode == _NVFP4_QUANT_MODE
    ):
        if weight_scale is not None:
            raise ValueError(
                f"weight_scale must be None for quant_mode={caps.quant_mode!r}"
            )
    else:
        if weight_scale is None:
            raise ValueError(
                f"weight_scale is required for quant_mode={caps.quant_mode!r}"
            )
        assert plan.weight_scale_dtype is not None
        _require_tensor(
            "weight_scale",
            weight_scale,
            shape=plan.weight_scale_shape,
            dtype=plan.weight_scale_dtype,
            device=caps.device,
        )
        if caps.table_memory == "mapped_host" and caps.quant_mode == _NVFP4_QUANT_MODE:
            _require_mapped_host_tensor(
                "weight_scale", weight_scale, device=caps.device
            )
    if plan.weight_scale_2_shape is None:
        if weight_scale_2 is not None:
            raise ValueError(
                f"weight_scale_2 must be None for quant_mode={caps.quant_mode!r}"
            )
    else:
        if weight_scale_2 is None:
            raise ValueError(
                f"weight_scale_2 is required for quant_mode={caps.quant_mode!r}"
            )
        assert plan.weight_scale_2_dtype is not None
        _require_tensor(
            "weight_scale_2",
            weight_scale_2,
            shape=plan.weight_scale_2_shape,
            dtype=plan.weight_scale_2_dtype,
            device=caps.device,
        )
    _require_tensor(
        "out",
        out,
        shape=plan.output_shape,
        dtype=plan.output_dtype,
        device=caps.device,
    )
    hash_binding = _hash_state.bind(
        scratch=hash_scratch,
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        out=ids,
    )
    read_tensors = [
        ("token_ids", token_ids),
        ("query_start_loc", query_start_loc),
        ("committed_history", committed_history),
        ("num_seqs", num_seqs),
        ("num_tokens", num_tokens),
        ("multipliers", hash_binding.geometry.multipliers),
        ("prime_sizes", hash_binding.geometry.prime_sizes),
        ("table_offsets", hash_binding.geometry.table_offsets),
    ]
    if weight is not None:
        read_tensors.append(("weight", weight))
    if disk_table is not None:
        read_tensors.append(("disk_weight", disk_table.weight))
        if disk_table.weight_scale is not None:
            read_tensors.append(("disk_weight_scale", disk_table.weight_scale))
    if weight_scale is not None:
        read_tensors.append(("weight_scale", weight_scale))
    if weight_scale_2 is not None:
        read_tensors.append(("weight_scale_2", weight_scale_2))
    for read_name, read_tensor in read_tensors:
        if _overlaps(scratch_storage, read_tensor):
            raise ValueError(
                f"mutable scratch must not overlap read-only tensor {read_name}"
            )
        if _overlaps(out, read_tensor):
            raise ValueError(
                f"mutable out must not overlap read-only tensor {read_name}"
            )
    if _overlaps(scratch_storage, out):
        raise ValueError("mutable scratch and out must not overlap")
    if disk_table is not None:
        disk_table._freeze()
    return Binding(
        _state=plan,
        plan=_plan,
        scratch=scratch_storage,
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        out=out,
        _ids=ids,
        _hash_scratch=hash_scratch,
        _hash_binding=hash_binding,
        disk_table=disk_table,
    )


def run(binding: Binding, *, token_count: int | None = None) -> torch.Tensor:
    """Hash and gather up to token_count rows from the planned storage.

    The device num_tokens must fit this host launch bound. Rows after the
    bound are untouched. The bound changes grids, not kernel specializations.
    """
    if binding._state.caps.device.type != "cuda":
        raise ValueError(
            "PLE embedding GPU run requires CUDA; use the explicit reference oracle"
        )
    from ._kernels import run_pipeline

    if token_count is None:
        token_count = binding._state.caps.max_tokens
    if not 0 <= token_count <= binding._state.caps.max_tokens:
        raise ValueError("token_count must fit the planned token capacity")
    if binding.disk_table is not None:
        state = require_prepared(binding.plan, "sequence.ple_embedding")
        state.run(binding, token_count=token_count)
    else:
        run_pipeline(binding, token_count=token_count)
    return binding.out[:token_count]


__all__ = [
    "QuantMode",
    "TableMemory",
    "Caps",
    "Plan",
    "TableLayout",
    "storage_layout",
    "Binding",
    "plan",
    "bind",
    "run",
]
