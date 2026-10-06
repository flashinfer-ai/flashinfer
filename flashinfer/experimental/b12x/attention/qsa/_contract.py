"""Capacity, binding, and runtime contract for QSA."""

from __future__ import annotations

from b12x._lib.program_cache import program_cache

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from ..._lib.compile_plan import (
    attach_programs, compile_only_launches, load_programs, program_keys,
)
from ..._lib.compile_pool import CompileJob
from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from ...preparation.types import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    PreparedCall,
    Selection,
    plan_from_handle,
    require_prepared,
)
from ._tuning import QsaConfig, QsaQuery, TUNING
from ._sparse_gqa_cute_config import MAX_SPLIT_ROWS as _MAX_SPLIT_ROWS

_ALIGN_BYTES = 256
_SCORE_WORKSPACE_LIMIT_BYTES = 128 * 1024 * 1024
_MIN_TOPK_WORKSPACE_BYTES = 1024 * 1024
_STABLE_TOPK_BLOCK = 512
_MAX_SCORE_CHUNK_GROUPS = 65536


def _align_up(value: int, alignment: int = _ALIGN_BYTES) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def _is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and value & (value - 1) == 0


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None and torch.cuda.is_available():
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@dataclass(frozen=True, kw_only=True)
class CacheRequirements:
    """Pure per-page shapes and byte layout required by QSA cache storage."""

    dtype: torch.dtype
    kv_dtype: torch.dtype
    main_k_page_shape: tuple[int, int, int]
    main_v_page_shape: tuple[int, int, int]
    compressed_page_shape: tuple[int, int]
    raw_k_ring_shape: tuple[int, int]
    raw_logical_positions_shape: tuple[int]
    raw_rope_positions_shape: tuple[int, int]
    raw_interval_start_positions_shape: tuple[int]
    main_k_page_nbytes: int
    main_v_page_nbytes: int
    main_kv_page_nbytes: int
    compressed_page_nbytes: int
    raw_k_ring_offset_bytes: int
    raw_logical_positions_offset_bytes: int
    raw_rope_positions_offset_bytes: int
    raw_interval_start_positions_offset_bytes: int
    raw_page_nbytes: int
    raw_ring_capacity: int
    selection_width: int
    alignment_bytes: int
    shared_compressed_raw_storage_legal: bool


def cache_requirements(
    *,
    main_page_size: int,
    kv_heads: int = 2,
    head_dim: int = 256,
    index_head_dim: int = 128,
    compress_ratio: int = 4,
    budget: int = 2048,
    position_axes: int = 1,
    max_speculative_tokens: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.bfloat16,
) -> CacheRequirements:
    """Describe QSA cache storage without requiring a device or pool capacity.

    The compressed page size is derived as ``main_page_size / compress_ratio``.
    Raw selector payload and metadata offsets describe their byte layout within
    a compressed/raw shared physical page.  A false
    ``shared_compressed_raw_storage_legal`` result lets an allocator reject or
    separately place a geometry before page counts and sequence limits exist.
    """

    positive = {
        "main_page_size": main_page_size,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "index_head_dim": index_head_dim,
        "compress_ratio": compress_ratio,
        "budget": budget,
    }
    for name, value in positive.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if dtype != torch.bfloat16:
        raise TypeError("QSA query and selector-cache dtype must be torch.bfloat16")
    if kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("QSA main KV cache dtype must be BF16 or FP8 E4M3FN")
    if not _is_power_of_two(head_dim) or int(head_dim) < 16:
        raise ValueError("head_dim must be a power of two at least 16")
    if not _is_power_of_two(index_head_dim):
        raise ValueError("index_head_dim must be a power of two")
    if int(main_page_size) % int(compress_ratio):
        raise ValueError("main_page_size must be divisible by compress_ratio")
    if int(budget) % int(compress_ratio):
        raise ValueError("budget must be divisible by compress_ratio")
    if int(budget) // int(compress_ratio) not in (512, 2048):
        raise ValueError("budget / compress_ratio must be 512 or 2048")
    if int(position_axes) not in (1, 3):
        raise ValueError("position_axes must be 1 or 3")
    if int(max_speculative_tokens) < 0:
        raise ValueError("max_speculative_tokens must be nonnegative")

    ratio = int(compress_ratio)
    raw_ring_capacity = ratio * math.ceil((ratio + int(max_speculative_tokens)) / ratio)
    if int(main_page_size) % raw_ring_capacity:
        raise ValueError("raw_ring_capacity must divide main_page_size")

    element_nbytes = dtype.itemsize
    kv_element_nbytes = kv_dtype.itemsize
    int64_nbytes = torch.int64.itemsize
    compressed_page_size = int(main_page_size) // ratio
    main_page_shape = (int(main_page_size), int(kv_heads), int(head_dim))
    main_page_nbytes = math.prod(main_page_shape) * kv_element_nbytes
    compressed_page_shape = (compressed_page_size, int(index_head_dim))
    compressed_page_nbytes = math.prod(compressed_page_shape) * element_nbytes
    raw_k_ring_shape = (raw_ring_capacity, int(index_head_dim))
    raw_k_ring_nbytes = math.prod(raw_k_ring_shape) * element_nbytes
    raw_logical_positions_shape = (raw_ring_capacity,)
    raw_logical_positions_offset_bytes = raw_k_ring_nbytes
    raw_rope_positions_shape = (raw_ring_capacity, int(position_axes))
    raw_rope_positions_offset_bytes = (
        raw_logical_positions_offset_bytes
        + math.prod(raw_logical_positions_shape) * int64_nbytes
    )
    raw_interval_start_positions_offset_bytes = (
        raw_rope_positions_offset_bytes
        + math.prod(raw_rope_positions_shape) * int64_nbytes
    )
    raw_page_nbytes = raw_interval_start_positions_offset_bytes + int64_nbytes

    return CacheRequirements(
        dtype=dtype,
        kv_dtype=kv_dtype,
        main_k_page_shape=main_page_shape,
        main_v_page_shape=main_page_shape,
        compressed_page_shape=compressed_page_shape,
        raw_k_ring_shape=raw_k_ring_shape,
        raw_logical_positions_shape=raw_logical_positions_shape,
        raw_rope_positions_shape=raw_rope_positions_shape,
        raw_interval_start_positions_shape=(1,),
        main_k_page_nbytes=main_page_nbytes,
        main_v_page_nbytes=main_page_nbytes,
        main_kv_page_nbytes=2 * main_page_nbytes,
        compressed_page_nbytes=compressed_page_nbytes,
        raw_k_ring_offset_bytes=0,
        raw_logical_positions_offset_bytes=raw_logical_positions_offset_bytes,
        raw_rope_positions_offset_bytes=raw_rope_positions_offset_bytes,
        raw_interval_start_positions_offset_bytes=(
            raw_interval_start_positions_offset_bytes
        ),
        raw_page_nbytes=raw_page_nbytes,
        raw_ring_capacity=raw_ring_capacity,
        selection_width=int(budget) + ratio - 1,
        alignment_bytes=_ALIGN_BYTES,
        shared_compressed_raw_storage_legal=(raw_page_nbytes <= compressed_page_nbytes),
    )


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static model geometry and serving capacities for one QSA plan."""

    device: torch.device | str
    max_batch: int
    max_raw_state_slots: int
    max_q_rows: int
    max_seq_len: int
    num_main_cache_pages: int
    num_compressed_cache_pages: int
    main_page_size: int
    compressed_page_size: int
    max_speculative_tokens: int = 0
    q_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    index_heads: int = 4
    index_kv_heads: int = 1
    index_head_dim: int = 128
    index_rotary_dim: int = 64
    compress_ratio: int = 4
    budget: int = 2048
    position_axes: int = 1
    mrope_sections: tuple[int, int, int] | None = None
    mrope_interleaved: bool = False
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.bfloat16
    dcp_size: int = 1
    dcp_rank: int = 0
    cp_kv_cache_interleave_size: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        if self.device.type != "cuda":
            raise ValueError(f"QSA decode requires a CUDA device, got {self.device}")
        positive = {
            "max_batch": self.max_batch,
            "max_raw_state_slots": self.max_raw_state_slots,
            "max_q_rows": self.max_q_rows,
            "max_seq_len": self.max_seq_len,
            "num_main_cache_pages": self.num_main_cache_pages,
            "num_compressed_cache_pages": self.num_compressed_cache_pages,
            "main_page_size": self.main_page_size,
            "compressed_page_size": self.compressed_page_size,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "index_heads": self.index_heads,
            "index_kv_heads": self.index_kv_heads,
            "index_head_dim": self.index_head_dim,
            "index_rotary_dim": self.index_rotary_dim,
            "compress_ratio": self.compress_ratio,
            "budget": self.budget,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.max_speculative_tokens) < 0:
            raise ValueError("max_speculative_tokens must be nonnegative")
        from ._dcp import validate_geometry

        validate_geometry(
            size=int(self.dcp_size),
            rank=int(self.dcp_rank),
            token_interleave=int(self.cp_kv_cache_interleave_size),
            compress_ratio=int(self.compress_ratio),
        )
        if not math.isfinite(float(self.rms_norm_eps)) or float(self.rms_norm_eps) <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if int(self.max_raw_state_slots) < int(self.max_batch):
            raise ValueError("max_raw_state_slots must be at least max_batch")
        if int(self.max_q_rows) < int(self.max_batch):
            raise ValueError("max_q_rows must be at least max_batch")
        if int(self.max_seq_len) < int(self.compress_ratio):
            raise ValueError("max_seq_len must be at least compress_ratio")
        if self.max_local_groups == 0:
            raise ValueError(
                "max_seq_len must provide at least one local compressed group "
                "on this DCP rank"
            )
        if int(self.max_seq_len) > torch.iinfo(torch.int32).max:
            raise ValueError("max_seq_len must fit in positive int32 positions")
        max_page_count = torch.iinfo(torch.int32).max + 1
        if int(self.num_main_cache_pages) > max_page_count:
            raise ValueError("num_main_cache_pages must fit in nonnegative int32 IDs")
        if int(self.num_compressed_cache_pages) > max_page_count:
            raise ValueError(
                "num_compressed_cache_pages must fit in nonnegative int32 IDs"
            )
        if self.dtype != torch.bfloat16:
            raise TypeError("QSA query and selector-cache dtype must be torch.bfloat16")
        if self.kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError("QSA main KV cache dtype must be BF16 or FP8 E4M3FN")
        if int(self.index_kv_heads) != 1:
            raise ValueError("QSA requires exactly one index KV head")
        if int(self.q_heads) % int(self.kv_heads):
            raise ValueError("q_heads must be divisible by kv_heads")
        from ._sparse_gqa_cute_config import (
            BLOCK_N as QWEN_CUTE_BLOCK_N,
            NUM_SPLITS as QWEN_CUTE_NUM_SPLITS,
            is_qwen_geometry,
        )

        if not is_qwen_geometry(
            q_heads=int(self.q_heads),
            kv_heads=int(self.kv_heads),
            head_dim=int(self.head_dim),
            selection_width=int(self.selection_width),
            block_n=QWEN_CUTE_BLOCK_N,
            splits=QWEN_CUTE_NUM_SPLITS,
        ):
            raise NotImplementedError(
                "QSA requires the CuTe Qwen sparse-GQA geometry "
                "with q_heads divisible by kv_heads, head_dim=256, and "
                "selection_width>=2051"
            )
        if not _is_power_of_two(self.head_dim) or int(self.head_dim) < 16:
            raise ValueError("head_dim must be a power of two at least 16")
        if not _is_power_of_two(self.index_head_dim):
            raise ValueError("index_head_dim must be a power of two")
        if int(self.index_rotary_dim) > int(self.index_head_dim):
            raise ValueError("index_rotary_dim cannot exceed index_head_dim")
        if int(self.index_rotary_dim) % 2:
            raise ValueError("index_rotary_dim must be even")
        if int(self.budget) % int(self.compress_ratio):
            raise ValueError("budget must be divisible by compress_ratio")
        if self.group_budget not in (512, 2048):
            raise ValueError("budget / compress_ratio must be 512 or 2048")
        if int(self.main_page_size) % int(self.compress_ratio):
            raise ValueError("main_page_size must be divisible by compress_ratio")
        if int(self.compressed_page_size) != (
            int(self.main_page_size) // int(self.compress_ratio)
        ):
            raise ValueError(
                "compressed_page_size must equal main_page_size / compress_ratio"
            )
        if int(self.main_page_size) % int(self.raw_ring_capacity):
            raise ValueError("raw_ring_capacity must divide main_page_size")
        if int(self.num_main_cache_pages) * int(self.main_page_size) < int(
            self.max_local_seq_len
        ):
            raise ValueError("main cache capacity cannot cover max_seq_len")
        if (
            int(self.num_compressed_cache_pages) * int(self.compressed_page_size)
            < self.max_groups
        ):
            raise ValueError("compressed cache capacity cannot cover max_seq_len")
        if int(self.position_axes) not in (1, 3):
            raise ValueError("position_axes must be 1 or 3")
        if int(self.position_axes) == 1:
            if self.mrope_sections is not None:
                raise ValueError("mrope_sections must be None for scalar positions")
            if self.mrope_interleaved:
                raise ValueError("mrope_interleaved must be false for scalar positions")
        else:
            if self.mrope_sections is None or len(self.mrope_sections) != 3:
                raise ValueError("three-axis positions require three mrope_sections")
            if any(int(section) <= 0 for section in self.mrope_sections):
                raise ValueError("mrope_sections must be positive")
            if sum(map(int, self.mrope_sections)) != int(self.index_rotary_dim) // 2:
                raise ValueError("mrope_sections must sum to half of index_rotary_dim")
            if self.mrope_interleaved:
                half = int(self.index_rotary_dim) // 2
                expected_sections = ((half + 2) // 3, (half + 1) // 3, half // 3)
                if tuple(map(int, self.mrope_sections)) != expected_sections:
                    raise ValueError(
                        "interleaved mrope_sections must match the round-robin "
                        f"axis counts {expected_sections}"
                    )

    @property
    def group_budget(self) -> int:
        return int(self.budget) // int(self.compress_ratio)

    @property
    def cache_requirements(self) -> CacheRequirements:
        return cache_requirements(
            main_page_size=int(self.main_page_size),
            kv_heads=int(self.kv_heads),
            head_dim=int(self.head_dim),
            index_head_dim=int(self.index_head_dim),
            compress_ratio=int(self.compress_ratio),
            budget=int(self.budget),
            position_axes=int(self.position_axes),
            max_speculative_tokens=int(self.max_speculative_tokens),
            dtype=self.dtype,
            kv_dtype=self.kv_dtype,
        )

    @property
    def selection_width(self) -> int:
        return self.cache_requirements.selection_width

    @property
    def raw_ring_capacity(self) -> int:
        return self.cache_requirements.raw_ring_capacity

    @property
    def compressed_page_nbytes(self) -> int:
        return self.cache_requirements.compressed_page_nbytes

    @property
    def raw_page_nbytes(self) -> int:
        return self.cache_requirements.raw_page_nbytes

    @property
    def max_groups(self) -> int:
        return self.max_local_groups

    @property
    def max_global_groups(self) -> int:
        return int(self.max_seq_len) // int(self.compress_ratio)

    @property
    def max_local_seq_len(self) -> int:
        from ._dcp import local_length

        return local_length(
            int(self.max_seq_len),
            size=int(self.dcp_size),
            rank=int(self.dcp_rank),
            interleave=int(self.cp_kv_cache_interleave_size),
        )

    @property
    def max_local_groups(self) -> int:
        if int(self.dcp_size) == 1:
            return self.max_global_groups
        from ._dcp import local_length

        return local_length(
            self.max_global_groups,
            size=int(self.dcp_size),
            rank=int(self.dcp_rank),
            interleave=int(self.cp_kv_cache_interleave_size)
            // int(self.compress_ratio),
        )

    @property
    def main_table_width(self) -> int:
        return math.ceil(self.max_local_seq_len / int(self.main_page_size))

    @property
    def compressed_table_width(self) -> int:
        return math.ceil(self.max_groups / int(self.compressed_page_size))


@dataclass(frozen=True)
class _ScratchLayout:
    prepared_query_offset_bytes: int
    prepared_query_nbytes: int
    score_offset_bytes: int
    score_nbytes: int
    eligible_counts_offset_bytes: int
    eligible_counts_nbytes: int
    merge_lengths_offset_bytes: int
    merge_lengths_nbytes: int
    topk_values_offset_bytes: int
    topk_values_nbytes: int
    topk_indices_offset_bytes: int
    topk_indices_nbytes: int
    topk_values_b_offset_bytes: int
    topk_values_b_nbytes: int
    topk_indices_b_offset_bytes: int
    topk_indices_b_nbytes: int
    topk_offset_bytes: int
    topk_nbytes: int
    partial_output_offset_bytes: int
    partial_output_nbytes: int
    partial_lse_offset_bytes: int
    partial_lse_nbytes: int
    output_lse_offset_bytes: int
    output_lse_nbytes: int
    draft_positions_offset_bytes: int
    total_nbytes: int


@dataclass(frozen=True)
class QsaPrograms:
    """Actual native executables retained by a prepared QSA plan.

    These are the objects returned by the QSA compiler transactions, rather
    than an observational bundle of their ProgramKeys.  Runtime adapters take
    them explicitly and never resolve a cache entry on their own.
    """

    support: Mapping[str, object]
    score: object
    sparse: Mapping[str, object]
    sparse_draft: Mapping[str, object]
    draft: Mapping[str, object]

    def __post_init__(self) -> None:
        objects = (
            *self.support.values(), self.score, *self.sparse.values(),
            *self.sparse_draft.values(),
            *self.draft.values(),
        )
        if any(item is None for item in objects):
            raise RuntimeError("QSA preparation did not retain every native executable")
        attach_programs(self, *objects)


@dataclass(frozen=True)
class _MaterializedPlan:
    """Resolved QSA layout and launch choices, owned by the prepared plan."""

    caps: Caps
    abi: FrozenMapping
    workspace_q_rows: int
    score_chunk_groups: int
    score_workspace_width: int
    num_score_chunks: int
    max_split_row_product: int
    config: QsaConfig
    programs: QsaPrograms | None
    _layout: _ScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(self, **kwargs: object) -> Binding:
        raise TypeError("bind requires the session-produced prepared plan")

    def bind_for_preparation(self, **kwargs: object) -> Binding:
        """Private session callback binding before plan publication."""
        return _bind_materialized(self, **kwargs)

    def run_for_preparation(self, binding: Binding, **dynamic: object) -> torch.Tensor:
        """Prime retained native programs before this state is published."""
        if binding.state is not self:
            raise ValueError("QSA preparation binding belongs to another state")
        if not isinstance(self.programs, QsaPrograms):
            raise RuntimeError("QSA preparation requires retained native programs")
        return _run(binding, programs=self.programs, **dynamic)

    def draft_selection_plan(
        self, *, max_source_rows: int | None = None
    ) -> DraftSelectionPlan:
        """Describe persistent anchor storage without allocating or initializing it."""
        if self.caps.max_speculative_tokens < 1:
            raise ValueError("draft selection reuse requires speculative capacity")
        rows = self.caps.max_q_rows if max_source_rows is None else int(max_source_rows)
        if rows < self.caps.max_q_rows:
            raise ValueError("draft selection state must cover planned query rows")
        return DraftSelectionPlan(self.caps.device, rows, self.caps.selection_width)


def draft_selection_plan(
    plan: Plan, *, max_source_rows: int | None = None,
) -> "DraftSelectionPlan":
    """Return the caller-owned anchor layout for a ready QSA plan."""
    state = require_prepared(plan, "attention.qsa")
    if not isinstance(state, _MaterializedPlan):
        raise TypeError("QSA plan has an invalid materialized state")
    return state.draft_selection_plan(max_source_rows=max_source_rows)


@dataclass(frozen=True)
class DraftSelectionPlan:
    """B12X-defined persistent storage layout for draft-selection anchors.

    Compatible bindings may share this storage when device and selection width
    match and source capacity covers each plan of the same model layer and
    request/cache assignment. Storage must outlive all bound
    operations and captured graphs. Planning and binding allocate no tensors.
    """

    device: torch.device
    max_source_rows: int
    selection_width: int

    def __post_init__(self) -> None:
        if self.max_source_rows < 1 or self.selection_width < 1:
            raise ValueError("draft selection capacity and width must be positive")
        object.__setattr__(self, "device", _canonical_device(self.device))

    def _regions(self) -> tuple[tuple[str, tuple[int, ...], torch.dtype, int], ...]:
        offset = 0
        regions = []
        for name, shape, dtype in (
            (
                "selected_positions",
                (self.max_source_rows, self.selection_width),
                torch.int32,
            ),
            ("logical_positions", (self.max_source_rows,), torch.int64),
            ("num_source_rows", (1,), torch.int32),
        ):
            offset = _align_up(offset)
            regions.append((name, shape, dtype, offset))
            offset += math.prod(shape) * dtype.itemsize
        return tuple(regions)

    def storage_specs(self) -> tuple[ScratchBufferSpec, ...]:
        """Return caller allocation requirements for persistent, non-scratch storage."""
        _, shape, dtype, offset = self._regions()[-1]
        return (
            scratch_buffer_spec(
                "qsa_draft_selection",
                nbytes=_align_up(offset + math.prod(shape) * dtype.itemsize),
                device=self.device,
            ),
        )

    def bind(
        self,
        *,
        storage: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    ) -> DraftSelectionState:
        """Create views only; call ``reset`` before the first use of uninitialized storage."""
        buffer = scratch_tensor(
            storage, self.storage_specs(), owner="qsa draft selection"
        )
        return DraftSelectionState(plan=self, _storage=buffer)


@dataclass(frozen=True)
class DraftSelectionState:
    """Persistent caller-owned anchors created by ``DraftSelectionPlan.bind``.

    Ordinary ``run`` replaces the valid anchor prefix. Reuse preserves it.
    The caller must reset before first use, at round boundaries, and before
    recycling requests; it must map each request to an anchor from that same
    request and round. A mapping outside the recorded row count or a query
    position outside the anchor's causal tail selects no positions for that
    row. Request identity and round provenance are caller invariants and
    cannot be inferred from logical positions.

    Reset, recording, map updates, and reuse must be ordered on the calling
    stream or joined with caller-established events. Concurrent operations
    must not share mutable storage or scratch. Captured reset and record calls
    execute on every replay, so capture must preserve the round lifecycle.
    Tensor properties expose inspection views; use ``reset`` and ``run`` for mutation.
    """

    plan: DraftSelectionPlan
    _storage: torch.Tensor

    def _view(self, index: int) -> torch.Tensor:
        _, shape, dtype, offset = self.plan._regions()[index]
        return _scratch_view(
            self._storage, offset_bytes=offset, shape=shape, dtype=dtype
        )

    @property
    def selected_positions(self) -> torch.Tensor:
        return self._view(0)

    @property
    def logical_positions(self) -> torch.Tensor:
        return self._view(1)

    @property
    def num_source_rows(self) -> torch.Tensor:
        return self._view(2)

    def reset(self) -> None:
        """Invalidate all anchors on the calling stream without allocating storage."""
        from ._draft_selection import reset_anchors

        reset_anchors(self._storage, self.plan._regions()[2][3])


@dataclass(frozen=True)
class DraftSelectionReuse:
    """Live request-to-anchor row map for one reuse transaction.

    ``source_rows[request_id]`` names a row of the last ordinary run. The caller
    supplies a contiguous int32/int64 vector of ``max_batch`` entries and keeps
    it stable until the operation completes. An active mapping outside the
    recorded rows selects no positions. The caller is responsible for request
    identity and round lifetime.
    """

    source_rows: torch.Tensor


@dataclass(frozen=True)
class Binding:
    """Caller-owned QSA cache, state, output, and scratch views.

    The main K/V cache and its block table are read-only. Compressed-cache and
    raw-ring tensors are mutable decode state; ``output`` and
    ``selected_positions`` are caller-owned result buffers.
    """

    state: _MaterializedPlan
    plan: Plan | None
    shared_compressed_raw_pool: bool
    scratch: torch.Tensor
    main_k_cache: torch.Tensor
    main_v_cache: torch.Tensor
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None
    main_block_table: torch.Tensor
    compressed_k_cache: torch.Tensor
    compressed_block_table: torch.Tensor
    raw_k_ring: torch.Tensor
    raw_logical_positions: torch.Tensor
    raw_rope_positions: torch.Tensor
    raw_interval_start_positions: torch.Tensor
    raw_state_slot_ids: torch.Tensor
    index_q_norm_weight: torch.Tensor
    index_k_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    output: torch.Tensor
    selected_positions: torch.Tensor
    prepared_index_query: torch.Tensor
    scores: torch.Tensor
    eligible_group_counts: torch.Tensor
    merge_lengths: torch.Tensor
    topk_values: torch.Tensor
    topk_group_ids: torch.Tensor
    topk_values_b: torch.Tensor
    topk_group_ids_b: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    output_lse: torch.Tensor
    selection_stream: torch.cuda.Stream | None = None
    _selection_done: torch.cuda.Event | None = None
    draft_selection: DraftSelectionState | None = None
    _draft_work_positions: torch.Tensor | None = None
    _record_draft_enabled: bool = True


@dataclass(frozen=True)
class LocalSelection:
    """Rank-local QSA group candidates for an exact DCP merge."""

    group_ids: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class _KernelCaps:
    """Scalar-only launch contract reconstructed inside the opaque op."""

    max_batch: int
    max_seq_len: int
    compressed_page_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    index_heads: int
    index_head_dim: int
    index_rotary_dim: int
    compress_ratio: int
    budget: int
    position_axes: int
    mrope_sections: tuple[int, int, int] | None
    mrope_interleaved: bool
    rms_norm_eps: float
    raw_ring_capacity: int
    max_speculative_tokens: int
    dcp_size: int
    dcp_rank: int
    cp_kv_cache_interleave_size: int

    @property
    def group_budget(self) -> int:
        return int(self.budget) // int(self.compress_ratio)

    @property
    def selection_width(self) -> int:
        return int(self.budget) + int(self.compress_ratio) - 1

    @property
    def max_groups(self) -> int:
        global_groups = int(self.max_seq_len) // int(self.compress_ratio)
        if int(self.dcp_size) == 1:
            return global_groups
        interleave = int(self.cp_kv_cache_interleave_size) // int(
            self.compress_ratio
        )
        round_width = int(self.dcp_size) * interleave
        full_rounds, remainder = divmod(global_groups, round_width)
        return full_rounds * interleave + min(
            max(remainder - int(self.dcp_rank) * interleave, 0), interleave
        )


def _qwen_row_splits(rows: int) -> int:
    return (
        64 if rows == 1 else 32 if rows <= 4 else 16 if rows <= _MAX_SPLIT_ROWS else 1
    )


def _target_splits(caps: Caps, rows: int) -> tuple[int, int]:
    from ._sparse_gqa_cute_config import (
        BLOCK_N as QWEN_CUTE_BLOCK_N,
        NUM_SPLITS as QWEN_CUTE_NUM_SPLITS,
        is_qwen_geometry,
    )

    if is_qwen_geometry(
        q_heads=int(caps.q_heads),
        kv_heads=int(caps.kv_heads),
        head_dim=int(caps.head_dim),
        selection_width=int(caps.selection_width),
        block_n=QWEN_CUTE_BLOCK_N,
        splits=QWEN_CUTE_NUM_SPLITS,
    ):
        return QWEN_CUTE_BLOCK_N, _qwen_row_splits(int(rows))
    raise NotImplementedError(
        "QSA requires the CuTe Qwen sparse-GQA geometry: q_heads divisible by "
        "kv_heads, head_dim=256, selection_width>=2051; "
        f"got q_heads={caps.q_heads}, "
        f"kv_heads={caps.kv_heads}, head_dim={caps.head_dim}, "
        f"selection_width={caps.selection_width}"
    )


def _scratch_layout(
    caps: Caps,
) -> tuple[_ScratchLayout, int, int, int, int, int]:
    workspace_q_rows = int(caps.max_q_rows)
    score_width_limit = max(
        1,
        _SCORE_WORKSPACE_LIMIT_BYTES // (workspace_q_rows * torch.float32.itemsize),
    )
    score_width_limit = min(
        score_width_limit,
        int(caps.group_budget) + _MAX_SCORE_CHUNK_GROUPS,
    )
    if int(caps.max_groups) <= score_width_limit:
        score_chunk_groups = int(caps.max_groups)
        score_workspace_width = score_chunk_groups
    else:
        score_chunk_groups = score_width_limit - int(caps.group_budget)
        if score_chunk_groups <= 0:
            raise ValueError(
                "QSA 128 MiB score workspace cannot hold one top-k carry row"
            )
        score_workspace_width = int(caps.group_budget) + score_chunk_groups
    num_score_chunks = math.ceil(int(caps.max_groups) / score_chunk_groups)
    prepared_query_nbytes = (
        workspace_q_rows
        * int(caps.index_heads)
        * int(caps.index_head_dim)
        * torch.bfloat16.itemsize
    )
    score_nbytes = workspace_q_rows * score_workspace_width * torch.float32.itemsize
    from ._sparse_gqa_cute_config import MAX_SPLIT_ROWS

    max_split_row_product = max(
        rows * _target_splits(caps, rows)[1]
        for rows in range(1, min(workspace_q_rows, MAX_SPLIT_ROWS) + 1)
    )
    partial_output_nbytes = (
        max_split_row_product
        * int(caps.q_heads)
        * int(caps.head_dim)
        * torch.float32.itemsize
    )
    partial_lse_nbytes = (
        max_split_row_product * int(caps.q_heads) * torch.float32.itemsize
    )
    output_lse_nbytes = (
        int(caps.max_q_rows) * int(caps.q_heads) * torch.float32.itemsize
    )
    eligible_counts_nbytes = workspace_q_rows * torch.int32.itemsize
    merge_lengths_nbytes = workspace_q_rows * torch.int32.itemsize
    topk_values_nbytes = (
        workspace_q_rows * int(caps.group_budget) * torch.float32.itemsize
    )
    topk_indices_nbytes = (
        workspace_q_rows * int(caps.group_budget) * torch.int32.itemsize
    )
    stable_topk_blocks = math.ceil(score_workspace_width / _STABLE_TOPK_BLOCK)
    stable_topk_nbytes = (
        2 * workspace_q_rows * stable_topk_blocks * torch.int32.itemsize
        + workspace_q_rows
        * int(caps.group_budget)
        * (torch.float32.itemsize + torch.int32.itemsize)
        + workspace_q_rows * (torch.float32.itemsize + torch.int32.itemsize)
    )
    topk_workspace_nbytes = _align_up(
        max(_MIN_TOPK_WORKSPACE_BYTES, stable_topk_nbytes)
    )

    offset = 0
    prepared_query_offset = _align_up(offset)
    offset = prepared_query_offset + prepared_query_nbytes
    score_offset = _align_up(offset)
    offset = score_offset + score_nbytes
    eligible_counts_offset = _align_up(offset)
    offset = eligible_counts_offset + eligible_counts_nbytes
    merge_lengths_offset = _align_up(offset)
    offset = merge_lengths_offset + merge_lengths_nbytes
    topk_values_offset = _align_up(offset)
    offset = topk_values_offset + topk_values_nbytes
    topk_indices_offset = _align_up(offset)
    offset = topk_indices_offset + topk_indices_nbytes
    topk_values_b_offset = _align_up(offset)
    offset = topk_values_b_offset + topk_values_nbytes
    topk_indices_b_offset = _align_up(offset)
    offset = topk_indices_b_offset + topk_indices_nbytes
    topk_offset = _align_up(offset)
    offset = topk_offset + topk_workspace_nbytes
    partial_output_offset = _align_up(offset)
    offset = partial_output_offset + partial_output_nbytes
    partial_lse_offset = _align_up(offset)
    offset = partial_lse_offset + partial_lse_nbytes
    output_lse_offset = _align_up(offset)
    offset = output_lse_offset + output_lse_nbytes
    draft_positions_offset = _align_up(offset)
    if caps.max_speculative_tokens > 0:
        offset = (
            draft_positions_offset
            + caps.max_batch
            * (caps.selection_width + caps.max_speculative_tokens)
            * torch.int32.itemsize
        )
    total_nbytes = _align_up(offset)
    return (
        _ScratchLayout(
            prepared_query_offset_bytes=prepared_query_offset,
            prepared_query_nbytes=prepared_query_nbytes,
            score_offset_bytes=score_offset,
            score_nbytes=score_nbytes,
            eligible_counts_offset_bytes=eligible_counts_offset,
            eligible_counts_nbytes=eligible_counts_nbytes,
            merge_lengths_offset_bytes=merge_lengths_offset,
            merge_lengths_nbytes=merge_lengths_nbytes,
            topk_values_offset_bytes=topk_values_offset,
            topk_values_nbytes=topk_values_nbytes,
            topk_indices_offset_bytes=topk_indices_offset,
            topk_indices_nbytes=topk_indices_nbytes,
            topk_values_b_offset_bytes=topk_values_b_offset,
            topk_values_b_nbytes=topk_values_nbytes,
            topk_indices_b_offset_bytes=topk_indices_b_offset,
            topk_indices_b_nbytes=topk_indices_nbytes,
            topk_offset_bytes=topk_offset,
            topk_nbytes=topk_workspace_nbytes,
            partial_output_offset_bytes=partial_output_offset,
            partial_output_nbytes=partial_output_nbytes,
            partial_lse_offset_bytes=partial_lse_offset,
            partial_lse_nbytes=partial_lse_nbytes,
            output_lse_offset_bytes=output_lse_offset,
            output_lse_nbytes=output_lse_nbytes,
            draft_positions_offset_bytes=draft_positions_offset,
            total_nbytes=total_nbytes,
        ),
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
        max_split_row_product,
        workspace_q_rows,
    )

_ABI_OPERANDS = (
    "request_ids",
    "rope_positions",
    "index_query",
    "raw_index_key",
    "main_k_cache",
    "main_v_cache",
    "main_block_table",
    "compressed_k_cache",
    "compressed_block_table",
    "raw_k_ring",
    "raw_logical_positions",
    "raw_rope_positions",
    "raw_interval_start_positions",
    "raw_state_slot_ids",
    "index_q_norm_weight",
    "index_k_norm_weight",
    "rope_cos",
    "rope_sin",
)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _descriptor(tensor: torch.Tensor) -> FrozenMapping:
    return FrozenMapping({
        "dtype": _dtype_name(tensor.dtype),
        "strides": tuple(int(value) for value in tensor.stride()),
    })


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for size in reversed(shape):
        result.append(stride)
        stride *= int(size)
    return tuple(reversed(result))


def _canonical_abi(caps: Caps) -> FrozenMapping:
    shapes = {
        "request_ids": (caps.max_q_rows,),
        "rope_positions": (caps.max_q_rows, caps.position_axes),
        "index_query": (caps.max_q_rows, caps.index_heads, caps.index_head_dim),
        "raw_index_key": (caps.max_q_rows, caps.index_head_dim),
        "main_k_cache": (
            caps.num_main_cache_pages, caps.main_page_size, caps.kv_heads, caps.head_dim,
        ),
        "main_v_cache": (
            caps.num_main_cache_pages, caps.main_page_size, caps.kv_heads, caps.head_dim,
        ),
        "main_block_table": (caps.max_batch, caps.main_table_width),
        "compressed_k_cache": (
            caps.num_compressed_cache_pages, caps.compressed_page_size, caps.index_head_dim,
        ),
        "compressed_block_table": (caps.max_batch, caps.compressed_table_width),
        "raw_k_ring": (caps.max_raw_state_slots, caps.raw_ring_capacity, caps.index_head_dim),
        "raw_logical_positions": (caps.max_raw_state_slots, caps.raw_ring_capacity),
        "raw_rope_positions": (caps.max_raw_state_slots, caps.raw_ring_capacity, caps.position_axes),
        "raw_interval_start_positions": (caps.max_raw_state_slots,),
        "raw_state_slot_ids": (caps.max_batch,),
        "index_q_norm_weight": (caps.index_head_dim,),
        "index_k_norm_weight": (caps.index_head_dim,),
        "rope_cos": (caps.max_seq_len, caps.index_rotary_dim // 2),
        "rope_sin": (caps.max_seq_len, caps.index_rotary_dim // 2),
    }
    dtypes = {
        "request_ids": torch.int64,
        "rope_positions": torch.int64,
        "index_query": caps.dtype,
        "raw_index_key": caps.dtype,
        "main_k_cache": caps.kv_dtype,
        "main_v_cache": caps.kv_dtype,
        "main_block_table": torch.int32,
        "compressed_k_cache": caps.dtype,
        "compressed_block_table": torch.int32,
        "raw_k_ring": torch.bfloat16,
        "raw_logical_positions": torch.int64,
        "raw_rope_positions": torch.int64,
        "raw_interval_start_positions": torch.int64,
        "raw_state_slot_ids": torch.int64,
        "index_q_norm_weight": torch.float32,
        "index_k_norm_weight": torch.float32,
        "rope_cos": torch.float32,
        "rope_sin": torch.float32,
    }
    return FrozenMapping({
        name: FrozenMapping({
            "dtype": _dtype_name(dtypes[name]),
            "strides": _contiguous_strides(shapes[name]),
        })
        for name in _ABI_OPERANDS
    })


def invocation_from_descriptors(
    caps: Caps, *, operands: Mapping[str, Mapping[str, object]],
) -> FrozenMapping:
    """Normalize immutable QSA native-ABI metadata without tensor allocation."""
    if not isinstance(caps, Caps):
        raise TypeError("caps must be qsa.Caps")
    if set(operands) != set(_ABI_OPERANDS):
        raise ValueError("QSA invocation requires every native ABI operand")
    canonical = _canonical_abi(caps)
    normalized = {}
    for name in _ABI_OPERANDS:
        descriptor = FrozenMapping(operands[name])
        if set(descriptor) != {"dtype", "strides"}:
            raise ValueError(f"QSA {name} ABI descriptor requires dtype and strides")
        dtype, strides = descriptor["dtype"], tuple(descriptor["strides"])
        expected_strides = tuple(canonical[name]["strides"])
        if (
            not isinstance(dtype, str)
            or len(strides) != len(expected_strides)
            or any(type(value) is not int or value <= 0 for value in strides)
        ):
            raise ValueError(f"QSA {name} ABI descriptor is invalid")
        normalized[name] = FrozenMapping({"dtype": dtype, "strides": strides})
    return FrozenMapping({"operands": FrozenMapping(normalized)})


def invocation_from_tensors(caps: Caps, **operands: torch.Tensor) -> FrozenMapping:
    """Capture the exact static native ABI from caller-owned QSA tensors."""
    if set(operands) != set(_ABI_OPERANDS):
        raise ValueError("QSA invocation requires every native ABI operand")
    if any(not isinstance(tensor, torch.Tensor) for tensor in operands.values()):
        raise TypeError("QSA ABI operands must be tensors")
    return invocation_from_descriptors(
        caps, operands={name: _descriptor(operands[name]) for name in _ABI_OPERANDS}
    )


def _abi_from_invocation(caps: Caps, invocation: FrozenMapping) -> FrozenMapping:
    if not invocation:
        return invocation_from_descriptors(caps, operands=_canonical_abi(caps))
    if set(invocation) != {"operands"} or not isinstance(invocation["operands"], FrozenMapping):
        raise ValueError("QSA invocation must come from invocation_from_tensors")
    return invocation_from_descriptors(caps, operands=invocation["operands"])


def _require_runtime_abi(
    expected: FrozenMapping,
    caps: Caps,
    *,
    request_ids: torch.Tensor,
    rope_positions: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    main_block_table: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    **binding_operands: torch.Tensor,
) -> None:
    del caps
    operands = dict(
        request_ids=request_ids,
        rope_positions=rope_positions,
        index_query=index_query,
        raw_index_key=raw_index_key,
        main_k_cache=main_k_cache,
        main_v_cache=main_v_cache,
        main_block_table=main_block_table,
        compressed_k_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        **binding_operands,
    )
    expected_operands = expected["operands"]
    if len(operands) != len(expected_operands):
        raise ValueError("QSA runtime ABI is missing declared operands")
    for name, tensor in operands.items():
        descriptor = expected_operands[name]
        strides = tensor.stride()
        expected_strides = descriptor["strides"]
        # A size-one axis contributes zero to every address. PyTorch may
        # canonicalize its stride when unflattening the M=1 fused projection.
        if (
            tensor.dtype != getattr(torch, descriptor["dtype"])
            or tensor.ndim != len(expected_strides)
            or any(size != 1 and stride != planned for size, stride, planned in
                   zip(tensor.shape, strides, expected_strides, strict=True))
        ):
            raise ValueError(
                f"QSA runtime {name} ABI ({tensor.dtype}, {strides}) differs "
                f"from prepared {descriptor}"
            )



def _query_from_caps(caps: Caps, invocation: FrozenMapping) -> QsaQuery:
    abi = _abi_from_invocation(caps, invocation)
    return QsaQuery(
        q_dtype=str(caps.dtype).removeprefix("torch."),
        kv_dtype=str(caps.kv_dtype).removeprefix("torch."),
        q_heads=caps.q_heads, kv_heads=caps.kv_heads, head_dim=caps.head_dim,
        index_heads=caps.index_heads, index_kv_heads=caps.index_kv_heads,
        index_head_dim=caps.index_head_dim, index_rotary_dim=caps.index_rotary_dim,
        main_page_size=caps.main_page_size, max_batch=caps.max_batch,
        max_q_rows=caps.max_q_rows, max_seq_len=caps.max_seq_len,
        max_speculative_tokens=caps.max_speculative_tokens,
        compress_ratio=caps.compress_ratio, budget=caps.budget,
        position_axes=caps.position_axes, mrope_interleaved=caps.mrope_interleaved,
        max_raw_state_slots=caps.max_raw_state_slots,
        num_main_cache_pages=caps.num_main_cache_pages,
        num_compressed_cache_pages=caps.num_compressed_cache_pages,
        compressed_page_size=caps.compressed_page_size,
        mrope_sections=caps.mrope_sections,
        rms_norm_eps=caps.rms_norm_eps,
        dcp_size=caps.dcp_size,
        dcp_rank=caps.dcp_rank,
        cp_kv_cache_interleave_size=caps.cp_kv_cache_interleave_size,
        abi=abi,
    )
def _materialize(caps: Caps, abi: FrozenMapping, config: QsaConfig) -> _MaterializedPlan:
    (
        layout, score_chunk_groups, score_workspace_width, num_score_chunks,
        max_split_row_product, workspace_q_rows,
    ) = _scratch_layout(caps)
    return _MaterializedPlan(
        caps=caps, abi=abi, workspace_q_rows=workspace_q_rows,
        score_chunk_groups=score_chunk_groups,
        score_workspace_width=score_workspace_width,
        num_score_chunks=num_score_chunks,
        max_split_row_product=max_split_row_product,
        config=config, programs=None, _layout=layout,
        _scratch_specs=(scratch_buffer_spec(
            "qsa.scratch", nbytes=layout.total_nbytes, device=caps.device,
        ),),
    )


def _caps_from_query(query: QsaQuery, *, ordinal: int) -> Caps:
    """Restore normalized declaration geometry without consulting runtime controls."""
    return Caps(
        device=torch.device("cuda", ordinal),
        max_batch=query.max_batch,
        max_raw_state_slots=query.max_raw_state_slots,
        max_q_rows=query.max_q_rows,
        max_seq_len=query.max_seq_len,
        num_main_cache_pages=query.num_main_cache_pages,
        num_compressed_cache_pages=query.num_compressed_cache_pages,
        main_page_size=query.main_page_size,
        compressed_page_size=query.compressed_page_size,
        max_speculative_tokens=query.max_speculative_tokens,
        q_heads=query.q_heads,
        kv_heads=query.kv_heads,
        head_dim=query.head_dim,
        index_heads=query.index_heads,
        index_kv_heads=query.index_kv_heads,
        index_head_dim=query.index_head_dim,
        index_rotary_dim=query.index_rotary_dim,
        compress_ratio=query.compress_ratio,
        budget=query.budget,
        position_axes=query.position_axes,
        mrope_sections=query.mrope_sections,
        mrope_interleaved=query.mrope_interleaved,
        rms_norm_eps=query.rms_norm_eps,
        dtype=getattr(torch, query.q_dtype),
        kv_dtype=getattr(torch, query.kv_dtype),
        dcp_size=query.dcp_size,
        dcp_rank=query.dcp_rank,
        cp_kv_cache_interleave_size=query.cp_kv_cache_interleave_size,
    )


def _compile_rows(caps: Caps) -> tuple[int, ...]:
    """Rows whose fixed split/direct native entries can execute at runtime."""
    candidates = (1, 2, 5, _MAX_SPLIT_ROWS, _MAX_SPLIT_ROWS + 1, caps.max_q_rows)
    return tuple(rows for rows in dict.fromkeys(candidates)
                 if 0 < rows <= int(caps.max_q_rows))


@program_cache(scope="preparation")
def compile_qsa(
    query_payload: FrozenMapping | Mapping[str, object],
    config_payload: FrozenMapping | Mapping[str, object],
    ordinal: int,
) -> QsaPrograms:
    """Compile and retain every exact QSA execution carrier.

    The metadata transaction has the serving ABI, but each returned value is
    the real compiler carrier consumed by the prepared host adapters.
    """
    from ._draft_selection import _prepare_kernel, _record_kernel
    from ._kernels import _support_context
    from ._score_cute import compile_score_representatives
    from ._sparse_gqa import compile_sparse_paged_gqa
    from ..._lib.compile_plan import launch_triton
    import triton

    query = QsaQuery(**dict(query_payload))
    config = QsaConfig.from_config(FrozenMapping(config_payload))
    TUNING.validate_query(query, None)
    TUNING.validate_config(query, config, None)
    caps = _caps_from_query(query, ordinal=int(ordinal))
    state = _materialize(caps, query.abi, config)
    layout = state._layout
    abi = query.abi["operands"]
    compile_device = torch.device("meta")
    with compile_only_launches():
        def empty(
            shape: tuple[int, ...], dtype: torch.dtype, name: str | None = None,
        ) -> torch.Tensor:
            if name is None:
                return torch.empty(shape, dtype=dtype, device=compile_device)
            descriptor = abi[name]
            if descriptor["dtype"] != _dtype_name(dtype):
                raise ValueError(
                    f"QSA {name} declaration dtype does not match its native operand"
                )
            return torch.empty_strided(
                shape, tuple(descriptor["strides"]), dtype=dtype,
                device=compile_device,
            )

        scratch = empty(state.scratch_specs()[0].shape, torch.uint8)
        main_k = empty(
            (caps.num_main_cache_pages, caps.main_page_size, caps.kv_heads, caps.head_dim),
            caps.kv_dtype, "main_k_cache",
        )
        main_v = empty(tuple(main_k.shape), caps.kv_dtype, "main_v_cache")
        main_table = empty(
            (caps.max_batch, caps.main_table_width), torch.int32, "main_block_table",
        )
        compressed = empty(
            (caps.num_compressed_cache_pages, caps.compressed_page_size, caps.index_head_dim),
            torch.bfloat16, "compressed_k_cache",
        )
        compressed_table = empty(
            (caps.max_batch, caps.compressed_table_width), torch.int32,
            "compressed_block_table",
        )
        raw_ring = empty(
            (caps.max_raw_state_slots, caps.raw_ring_capacity, caps.index_head_dim),
            torch.bfloat16, "raw_k_ring",
        )
        raw_logical = empty(
            (caps.max_raw_state_slots, caps.raw_ring_capacity), torch.int64, "raw_logical_positions",
        )
        raw_rope = empty(
            (caps.max_raw_state_slots, caps.raw_ring_capacity, caps.position_axes),
            torch.int64, "raw_rope_positions",
        )
        raw_interval = empty(
            (caps.max_raw_state_slots,), torch.int64, "raw_interval_start_positions",
        )
        raw_slots = empty(
            (caps.max_batch,), getattr(torch, abi["raw_state_slot_ids"]["dtype"]), "raw_state_slot_ids",
        )
        q_norm = empty(
            (caps.index_head_dim,), getattr(torch, abi["index_q_norm_weight"]["dtype"]), "index_q_norm_weight",
        )
        k_norm = empty(
            (caps.index_head_dim,), getattr(torch, abi["index_k_norm_weight"]["dtype"]), "index_k_norm_weight",
        )
        rope_cos = empty(
            (caps.max_seq_len, caps.index_rotary_dim // 2), getattr(torch, abi["rope_cos"]["dtype"]), "rope_cos",
        )
        rope_sin = empty(tuple(rope_cos.shape), getattr(torch, abi["rope_sin"]["dtype"]), "rope_sin")
        sequence_lengths = empty((caps.max_batch,), torch.int32)
        query_start = empty((caps.max_batch + 1,), torch.int32)
        accepted = empty((caps.max_batch,), torch.int32)
        prefilling = empty((caps.max_batch,), torch.bool)
        descale = empty((1,), torch.float32) if caps.kv_dtype == torch.float8_e4m3fn else None
        rows = max(_compile_rows(caps))
        q = empty((rows, caps.q_heads, caps.head_dim), torch.bfloat16)
        output = empty(tuple(q.shape), torch.bfloat16)
        selected = empty((rows, caps.selection_width), torch.int32)
        index_q = empty(
            (rows, caps.index_heads, caps.index_head_dim), torch.bfloat16, "index_query",
        )
        raw_key = empty((rows, caps.index_head_dim), torch.bfloat16, "raw_index_key")
        request_ids = empty(
            (rows,), getattr(torch, abi["request_ids"]["dtype"]), "request_ids",
        )
        positions = empty((rows,), torch.int64)
        rope_positions = empty(
            (rows, caps.position_axes), torch.int64, "rope_positions",
        )
        prepared = _scratch_view(scratch, offset_bytes=layout.prepared_query_offset_bytes,
                                 shape=(state.workspace_q_rows, caps.index_heads,
                                        caps.index_head_dim), dtype=torch.bfloat16)
        scores = _scratch_view(scratch, offset_bytes=layout.score_offset_bytes,
                               shape=(state.workspace_q_rows, state.score_workspace_width),
                               dtype=torch.float32)
        eligible = _scratch_view(scratch, offset_bytes=layout.eligible_counts_offset_bytes,
                                 shape=(state.workspace_q_rows,), dtype=torch.int32)
        merge = _scratch_view(scratch, offset_bytes=layout.merge_lengths_offset_bytes,
                              shape=(state.workspace_q_rows,), dtype=torch.int32)
        kernel_caps = _KernelCaps(
            max_batch=caps.max_batch, max_seq_len=caps.max_seq_len,
            compressed_page_size=caps.compressed_page_size, q_heads=caps.q_heads,
            kv_heads=caps.kv_heads, head_dim=caps.head_dim, index_heads=caps.index_heads,
            index_head_dim=caps.index_head_dim, index_rotary_dim=caps.index_rotary_dim,
            compress_ratio=caps.compress_ratio, budget=caps.budget,
            position_axes=caps.position_axes, mrope_sections=caps.mrope_sections,
            mrope_interleaved=caps.mrope_interleaved, rms_norm_eps=caps.rms_norm_eps,
            raw_ring_capacity=caps.raw_ring_capacity,
            max_speculative_tokens=caps.max_speculative_tokens,
            dcp_size=caps.dcp_size,
            dcp_rank=caps.dcp_rank,
            cp_kv_cache_interleave_size=caps.cp_kv_cache_interleave_size,
        )
        score = compile_score_representatives(
            prepared_query=prepared, query_positions=positions, request_ids=request_ids,
            sequence_lengths=sequence_lengths, compressed_cache=compressed,
            compressed_block_table=compressed_table,
            scores=scores, eligible_counts=eligible, merge_lengths=merge, caps=kernel_caps,
        )
        sparse = compile_sparse_paged_gqa(
            query=q, key_cache=main_k, value_cache=main_v, request_ids=request_ids,
            selected_positions=selected[:rows], direct_kv_warps=config.sparse_gqa_direct_kv_warps,
            return_lse=caps.dcp_size > 1,
        )
        if caps.max_speculative_tokens:
            draft_selected = empty(
                (rows, caps.selection_width + caps.max_speculative_tokens),
                torch.int32,
            )
            sparse_draft = compile_sparse_paged_gqa(
                query=q,
                key_cache=main_k,
                value_cache=main_v,
                request_ids=request_ids,
                selected_positions=draft_selected,
                direct_kv_warps=config.sparse_gqa_direct_kv_warps,
                return_lse=caps.dcp_size > 1,
            )
        else:
            sparse_draft = sparse
        draft = {}
        # One complete transaction under the compile context compiles every
        # support program the runtime launches, with the runtime ABI.
        from types import SimpleNamespace
        support: dict[str, object] = {}
        with _support_context(support, compiling=True):
            _qsa_decode_impl(
                q, index_q, raw_key, request_ids, positions, rope_positions,
                sequence_lengths, query_start, accepted, prefilling, scratch,
                main_k, main_v, descale, descale, main_table, compressed,
                compressed_table, raw_ring, raw_logical, raw_rope, raw_interval,
                raw_slots, q_norm, k_norm, rope_cos, rope_sin, output, selected,
                config.sparse_gqa_direct_kv_warps, caps.max_seq_len,
                caps.max_speculative_tokens, caps.compress_ratio, caps.budget,
                caps.index_rotary_dim, *(caps.mrope_sections or (0, 0, 0)),
                caps.mrope_interleaved, caps.rms_norm_eps, state.score_chunk_groups,
                state.score_workspace_width, state.num_score_chunks,
                state.max_split_row_product, state.workspace_q_rows,
                layout.prepared_query_offset_bytes, layout.score_offset_bytes,
                layout.eligible_counts_offset_bytes, layout.merge_lengths_offset_bytes,
                layout.topk_values_offset_bytes, layout.topk_indices_offset_bytes,
                layout.topk_values_b_offset_bytes, layout.topk_indices_b_offset_bytes,
                layout.topk_offset_bytes, layout.partial_output_offset_bytes,
                layout.partial_lse_offset_bytes,
                layout.output_lse_offset_bytes,
                dcp_size=caps.dcp_size,
                dcp_rank=caps.dcp_rank,
                cp_kv_cache_interleave_size=caps.cp_kv_cache_interleave_size,
                programs=SimpleNamespace(support=support, score=score, sparse=sparse),
            )
        if not support:
            raise RuntimeError("QSA support compilation produced no native programs")
        if caps.max_speculative_tokens:
            draft_plan = DraftSelectionPlan(
                compile_device, caps.max_q_rows, caps.selection_width,
            )
            storage = empty(draft_plan.storage_specs()[0].shape, torch.uint8)
            draft_state = draft_plan.bind(storage=storage)
            draft["record"] = launch_triton(
                _record_kernel, (rows,), positions,
                draft_state.logical_positions, draft_state.num_source_rows,
                selected[:rows], draft_state.selected_positions, rows, 1,
                WIDTH=caps.selection_width, BLOCK=triton.next_power_of_2(caps.selection_width),
            )
            draft["prepare"] = launch_triton(
                _prepare_kernel, (rows,), draft_state.logical_positions,
                draft_state.selected_positions, empty((caps.max_batch,), torch.int64),
                draft_state.num_source_rows, empty((rows,), torch.int32), positions,
                selected, rows, caps.max_q_rows, caps.max_batch,
                WIDTH=caps.selection_width, TAIL=caps.max_speculative_tokens,
                DCP_SIZE=caps.dcp_size, DCP_RANK=caps.dcp_rank,
                CP_INTERLEAVE=caps.cp_kv_cache_interleave_size,
                BLOCK=triton.next_power_of_2(caps.selection_width + caps.max_speculative_tokens),
                num_warps=4,
            )
        else:
            draft["disabled"] = score
    return QsaPrograms(
        support=support,
        score=score,
        sparse=sparse,
        sparse_draft=sparse_draft,
        draft=draft,
    )


def plan(
    caps: Caps, *, invocation: FrozenMapping = FrozenMapping(),
    override: QsaConfig | None = None,
) -> Plan[QsaConfig]:
    """Declare QSA preparation; it neither allocates nor resolves programs."""
    if not isinstance(caps, Caps):
        raise TypeError("caps must be qsa.Caps")
    invocation = FrozenMapping(invocation)
    query = _query_from_caps(caps, invocation)
    return Plan(
        contract=TUNING, query=query, invocation=invocation, override=override,
        _device=caps.device, shared=False,
        # Compile factories and materialization receive the full declared
        # query; the selection key omits the pool's page counts.
        _compile_jobs=lambda config, device: (CompileJob.create(
            "b12x.attention.qsa._contract:compile_qsa",
            query.to_dict(), TUNING.encode_config(config), device.ordinal,
        ),),
        _memory_requirements=lambda config, _device: MemoryRequirements(
            scratch=_materialize(caps, query.abi, config).scratch_specs(),
        ),
        _materialize=lambda selection, device: replace(
            _materialize(caps, query.abi, selection.config),
            programs=load_programs(compile_qsa(
                query.to_dict(),
                TUNING.encode_config(selection.config),
                device.ordinal,
            )),
        ),
    )

def _check_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | tuple[torch.dtype, ...] | None = None,
    contiguous: bool = False,
    unit_inner_stride: bool = False,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} device {tensor.device} does not match {device}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if dtype is not None:
        dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
        if tensor.dtype not in dtypes:
            raise TypeError(f"{name} must have dtype in {dtypes}, got {tensor.dtype}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if unit_inner_stride and (tensor.ndim == 0 or int(tensor.stride(-1)) != 1):
        raise ValueError(f"{name} must have unit inner stride")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the bounding byte interval of a validated non-overlapping view."""
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    extent = 0
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if int(size) > 1:
            extent += (int(size) - 1) * int(stride)
    return start, start + (extent + 1) * int(tensor.element_size())


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    from ..._lib.compile_plan import compile_only_launches_enabled

    if compile_only_launches_enabled():
        # Metadata extraction compiles the separate-storage transaction; the
        # shared-storage validator is enumerated explicitly by compile_qsa.
        return False
    if left.device != right.device:
        return False
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


def _slot_views_are_disjoint(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Prove disjoint live regions for two views embedded in shared pages."""
    if not _overlaps(left, right):
        return True
    if left.ndim == 0 or right.ndim == 0:
        return False
    left_stride = int(left.stride(0)) * int(left.element_size())
    right_stride = int(right.stride(0)) * int(right.element_size())
    if left_stride <= 0 or left_stride != right_stride:
        return False

    def slot_interval(tensor: torch.Tensor) -> tuple[int, int]:
        start = int(tensor.untyped_storage().data_ptr()) + int(
            tensor.storage_offset()
        ) * int(tensor.element_size())
        extent = 0
        for size, stride in zip(tensor.shape[1:], tensor.stride()[1:], strict=True):
            if int(size) > 1:
                extent += (int(size) - 1) * int(stride)
        return start, start + (extent + 1) * int(tensor.element_size())

    left_start, left_end = slot_interval(left)
    right_start, right_end = slot_interval(right)
    if left_end - left_start > left_stride or right_end - right_start > left_stride:
        return False
    delta = right_start - left_start
    center = (-delta) // left_stride
    left_count = int(left.shape[0])
    right_count = int(right.shape[0])
    for difference in range(center - 2, center + 3):
        if difference < -(left_count - 1) or difference > right_count - 1:
            continue
        shifted_start = right_start + difference * left_stride
        shifted_end = right_end + difference * left_stride
        if left_start < shifted_end and shifted_start < left_end:
            return False
    return True


def _require_non_overlapping_layout(name: str, tensor: torch.Tensor) -> None:
    """Reject internal aliases while permitting ordinary padded layouts."""
    span = 1
    dimensions = sorted(
        (int(stride), int(size))
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        if int(size) > 1
    )
    for stride, size in dimensions:
        if stride < span:
            raise ValueError(f"{name} must not have internal storage overlap")
        span += (size - 1) * stride


def _require_mutation_alias_contract(
    *,
    mutable: tuple[tuple[str, torch.Tensor], ...],
    read_only: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    for name, tensor in (*mutable, *read_only):
        _require_non_overlapping_layout(name, tensor)
    for index, (left_name, left) in enumerate(mutable):
        for right_name, right in mutable[index + 1 :]:
            if _overlaps(left, right):
                names = {left_name, right_name}
                raw_names = {
                    "raw_k_ring",
                    "raw_logical_positions",
                    "raw_rope_positions",
                    "raw_interval_start_positions",
                }
                if "compressed_k_cache" in names and names & raw_names:
                    continue
                if names <= raw_names and _slot_views_are_disjoint(left, right):
                    continue
                raise ValueError(
                    f"mutable buffers {left_name} and {right_name} must not overlap"
                )
        for right_name, right in read_only:
            if _overlaps(left, right):
                if _slot_views_are_disjoint(left, right):
                    continue
                raise ValueError(
                    f"mutable buffer {left_name} must not overlap read-only "
                    f"tensor {right_name}"
                )


def _validate_shared_compressed_raw_layout(
    *,
    caps: Caps,
    compressed_k_cache: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
) -> None:
    raw_tensors = (
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
    )
    shared = tuple(_overlaps(compressed_k_cache, tensor) for tensor in raw_tensors)
    if not any(shared):
        return
    if not all(shared):
        raise ValueError(
            "shared compressed/raw storage must include key, logical-position, "
            "RoPE-position, and interval-start-position views"
        )
    page_stride = int(compressed_k_cache.stride(0)) * int(
        compressed_k_cache.element_size()
    )
    if int(caps.raw_page_nbytes) > page_stride:
        raise ValueError(
            "raw ring page bytes, including int64 logical and RoPE metadata, "
            "must fit inside one aliased compressed cache page"
        )
    expected_compressed_strides = (
        int(caps.compressed_page_size) * int(caps.index_head_dim),
        int(caps.index_head_dim),
        1,
    )
    if (
        page_stride != int(caps.compressed_page_nbytes)
        or tuple(compressed_k_cache.stride()) != expected_compressed_strides
    ):
        raise ValueError(
            "shared compressed/raw storage requires dense contiguous compressed pages"
        )
    if int(raw_k_ring.shape[0]) > int(compressed_k_cache.shape[0]):
        raise ValueError(
            "shared raw state slots cannot exceed compressed physical pages"
        )
    for name, tensor in zip(
        (
            "raw_k_ring",
            "raw_logical_positions",
            "raw_rope_positions",
            "raw_interval_start_positions",
        ),
        raw_tensors,
        strict=True,
    ):
        tensor_page_stride = int(tensor.stride(0)) * int(tensor.element_size())
        if tensor_page_stride != page_stride:
            raise ValueError(
                f"{name} page stride must match compressed page stride in bytes"
            )

    expected_raw_strides = (
        page_stride // torch.bfloat16.itemsize,
        int(caps.index_head_dim),
        1,
    )
    expected_tag_strides = (page_stride // torch.int64.itemsize, 1)
    expected_rope_strides = (
        page_stride // torch.int64.itemsize,
        int(caps.position_axes),
        1,
    )
    expected_interval_start_strides = (page_stride // torch.int64.itemsize,)
    if tuple(raw_k_ring.stride()) != expected_raw_strides:
        raise ValueError("shared raw_k_ring must use the packed raw-page layout")
    if tuple(raw_logical_positions.stride()) != expected_tag_strides:
        raise ValueError(
            "shared raw_logical_positions must use the packed raw-page tail layout"
        )
    if tuple(raw_rope_positions.stride()) != expected_rope_strides:
        raise ValueError(
            "shared raw_rope_positions must use the packed raw-page tail layout"
        )
    if tuple(raw_interval_start_positions.stride()) != expected_interval_start_strides:
        raise ValueError(
            "shared raw_interval_start_positions must use the packed raw-page "
            "tail layout"
        )

    compressed_start, _ = _byte_interval(compressed_k_cache[:1])
    raw_start, _ = _byte_interval(raw_k_ring[:1])
    tags_start, _ = _byte_interval(raw_logical_positions[:1])
    rope_start, _ = _byte_interval(raw_rope_positions[:1])
    interval_start, _ = _byte_interval(raw_interval_start_positions[:1])
    payload_nbytes = (
        int(caps.raw_ring_capacity) * int(caps.index_head_dim) * torch.bfloat16.itemsize
    )
    tags_nbytes = int(caps.raw_ring_capacity) * torch.int64.itemsize
    rope_nbytes = (
        int(caps.raw_ring_capacity) * int(caps.position_axes) * torch.int64.itemsize
    )
    expected_starts = (
        compressed_start,
        compressed_start + payload_nbytes,
        compressed_start + payload_nbytes + tags_nbytes,
        compressed_start + payload_nbytes + tags_nbytes + rope_nbytes,
    )
    if (raw_start, tags_start, rope_start, interval_start) != expected_starts:
        raise ValueError(
            "shared raw key and int64 state views must use their named "
            "page-tail offsets"
        )


def _scratch_view(
    storage: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    elements = math.prod(shape)
    nbytes = elements * dtype.itemsize
    return storage.narrow(0, int(offset_bytes), int(nbytes)).view(dtype).view(shape)

def _bind_materialized(
    state: _MaterializedPlan,
    *,
    plan: Plan | None = None,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    main_block_table: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    selection_stream: torch.cuda.Stream | None = None,
    selection_done: torch.cuda.Event | None = None,
    draft_selection: DraftSelectionState | None = None,
) -> Binding:
    """Private binding for a session materialized state."""
    caps = state.caps
    scratch_storage = scratch_tensor(
        scratch,
        state.scratch_specs(),
        owner="qsa",
    )
    if main_k_cache.ndim != 4 or tuple(main_k_cache.shape[1:]) != (
        int(caps.main_page_size),
        int(caps.kv_heads),
        int(caps.head_dim),
    ):
        raise ValueError(
            "main_k_cache must have shape "
            f"[pages, {caps.main_page_size}, {caps.kv_heads}, {caps.head_dim}]"
        )
    if not 0 < int(main_k_cache.shape[0]) <= int(caps.num_main_cache_pages):
        raise ValueError("main_k_cache page count exceeds planned capacity")
    _check_tensor(
        main_k_cache,
        name="main_k_cache",
        device=caps.device,
        dtype=caps.kv_dtype,
        unit_inner_stride=True,
    )
    _check_tensor(
        main_v_cache,
        name="main_v_cache",
        device=caps.device,
        shape=tuple(main_k_cache.shape),
        dtype=caps.kv_dtype,
        unit_inner_stride=True,
    )
    fp8_kv = caps.kv_dtype == torch.float8_e4m3fn
    if fp8_kv and (k_descale is None or v_descale is None):
        raise ValueError("FP8 QSA main caches require k_descale and v_descale")
    for descale, name in ((k_descale, "k_descale"), (v_descale, "v_descale")):
        if descale is None:
            continue
        _check_tensor(
            descale,
            name=name,
            device=caps.device,
            dtype=torch.float32,
            contiguous=True,
        )
        if descale.numel() != 1:
            raise ValueError(f"{name} must contain exactly one per-layer scale")
    if main_block_table.ndim != 2 or tuple(main_block_table.shape) != (
        int(caps.max_batch),
        int(caps.main_table_width),
    ):
        raise ValueError(
            "main_block_table must have shape "
            f"({caps.max_batch}, {caps.main_table_width})"
        )
    _check_tensor(
        main_block_table,
        name="main_block_table",
        device=caps.device,
        dtype=torch.int32,
        contiguous=True,
    )
    expected_compressed_tail = (
        int(caps.compressed_page_size),
        int(caps.index_head_dim),
    )
    if compressed_k_cache.ndim != 3 or tuple(compressed_k_cache.shape[1:]) != (
        expected_compressed_tail
    ):
        raise ValueError(
            "compressed_k_cache must have shape "
            f"[pages, {caps.compressed_page_size}, {caps.index_head_dim}]"
        )
    if not 0 < int(compressed_k_cache.shape[0]) <= int(caps.num_compressed_cache_pages):
        raise ValueError("compressed_k_cache page count exceeds planned capacity")
    _check_tensor(
        compressed_k_cache,
        name="compressed_k_cache",
        device=caps.device,
        dtype=caps.dtype,
        unit_inner_stride=True,
    )
    _check_tensor(
        compressed_block_table,
        name="compressed_block_table",
        device=caps.device,
        shape=(int(caps.max_batch), int(caps.compressed_table_width)),
        dtype=torch.int32,
        contiguous=True,
    )
    raw_shape = (
        int(caps.max_raw_state_slots),
        int(caps.raw_ring_capacity),
    )
    _check_tensor(
        raw_k_ring,
        name="raw_k_ring",
        device=caps.device,
        shape=(*raw_shape, int(caps.index_head_dim)),
        dtype=caps.dtype,
        unit_inner_stride=True,
    )
    _check_tensor(
        raw_logical_positions,
        name="raw_logical_positions",
        device=caps.device,
        shape=raw_shape,
        dtype=torch.int64,
        unit_inner_stride=True,
    )
    _check_tensor(
        raw_rope_positions,
        name="raw_rope_positions",
        device=caps.device,
        shape=(*raw_shape, int(caps.position_axes)),
        dtype=torch.int64,
        unit_inner_stride=True,
    )
    _check_tensor(
        raw_interval_start_positions,
        name="raw_interval_start_positions",
        device=caps.device,
        shape=(int(caps.max_raw_state_slots),),
        dtype=torch.int64,
    )
    _check_tensor(
        raw_state_slot_ids,
        name="raw_state_slot_ids",
        device=caps.device,
        shape=(int(caps.max_batch),),
        dtype=(torch.int32, torch.int64),
    )
    if int(raw_state_slot_ids.stride(0)) <= 0:
        raise ValueError("raw_state_slot_ids must have a positive stride")
    norm_dtype = (torch.bfloat16, torch.float32)
    _check_tensor(
        index_q_norm_weight,
        name="index_q_norm_weight",
        device=caps.device,
        shape=(int(caps.index_head_dim),),
        dtype=norm_dtype,
        contiguous=True,
    )
    _check_tensor(
        index_k_norm_weight,
        name="index_k_norm_weight",
        device=caps.device,
        shape=(int(caps.index_head_dim),),
        dtype=norm_dtype,
        contiguous=True,
    )
    rope_width = int(caps.index_rotary_dim) // 2
    if (
        rope_cos.ndim != 2
        or int(rope_cos.shape[0]) <= 0
        or int(rope_cos.shape[1]) != rope_width
    ):
        raise ValueError(f"rope_cos must have shape [positions, {rope_width}]")
    _check_tensor(
        rope_cos,
        name="rope_cos",
        device=caps.device,
        dtype=(torch.bfloat16, torch.float32),
        unit_inner_stride=True,
    )
    if int(rope_cos.stride(0)) <= 0:
        raise ValueError("rope_cos must have a positive row stride")
    _check_tensor(
        rope_sin,
        name="rope_sin",
        device=caps.device,
        shape=tuple(rope_cos.shape),
        dtype=rope_cos.dtype,
        unit_inner_stride=True,
    )
    if int(rope_sin.stride(0)) <= 0:
        raise ValueError("rope_sin must have a positive row stride")
    if output.ndim != 3 or not 0 < int(output.shape[0]) <= int(caps.max_q_rows):
        raise ValueError("output rows must be within the planned QSA capacity")
    _check_tensor(
        output,
        name="output",
        device=caps.device,
        shape=(int(output.shape[0]), int(caps.q_heads), int(caps.head_dim)),
        dtype=caps.dtype,
        contiguous=True,
    )
    _check_tensor(
        selected_positions,
        name="selected_positions",
        device=caps.device,
        shape=(int(caps.max_q_rows), int(caps.selection_width)),
        dtype=torch.int32,
        contiguous=True,
    )
    _validate_shared_compressed_raw_layout(
        caps=caps,
        compressed_k_cache=compressed_k_cache,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        raw_interval_start_positions=raw_interval_start_positions,
    )
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch_storage),
            ("compressed_k_cache", compressed_k_cache),
            ("raw_k_ring", raw_k_ring),
            ("raw_logical_positions", raw_logical_positions),
            ("raw_rope_positions", raw_rope_positions),
            ("raw_interval_start_positions", raw_interval_start_positions),
            ("output", output),
            ("selected_positions", selected_positions),
        ),
        read_only=tuple(
            (name, tensor)
            for name, tensor in (
                ("main_k_cache", main_k_cache),
                ("main_v_cache", main_v_cache),
                ("k_descale", k_descale),
                ("v_descale", v_descale),
                ("main_block_table", main_block_table),
                ("compressed_block_table", compressed_block_table),
                ("raw_state_slot_ids", raw_state_slot_ids),
                ("index_q_norm_weight", index_q_norm_weight),
                ("index_k_norm_weight", index_k_norm_weight),
                ("rope_cos", rope_cos),
                ("rope_sin", rope_sin),
            )
            if tensor is not None
        ),
    )
    layout = state._layout
    workspace_q_rows = int(state.workspace_q_rows)
    prepared_index_query = _scratch_view(
        scratch_storage,
        offset_bytes=layout.prepared_query_offset_bytes,
        shape=(
            workspace_q_rows,
            int(caps.index_heads),
            int(caps.index_head_dim),
        ),
        dtype=torch.bfloat16,
    )
    scores = _scratch_view(
        scratch_storage,
        offset_bytes=layout.score_offset_bytes,
        shape=(workspace_q_rows, int(state.score_workspace_width)),
        dtype=torch.float32,
    )
    eligible_group_counts = _scratch_view(
        scratch_storage,
        offset_bytes=layout.eligible_counts_offset_bytes,
        shape=(workspace_q_rows,),
        dtype=torch.int32,
    )
    merge_lengths = _scratch_view(
        scratch_storage,
        offset_bytes=layout.merge_lengths_offset_bytes,
        shape=(workspace_q_rows,),
        dtype=torch.int32,
    )
    topk_values = _scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_values_offset_bytes,
        shape=(workspace_q_rows, int(caps.group_budget)),
        dtype=torch.float32,
    )
    topk_group_ids = _scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_indices_offset_bytes,
        shape=(workspace_q_rows, int(caps.group_budget)),
        dtype=torch.int32,
    )
    topk_values_b = _scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_values_b_offset_bytes,
        shape=(workspace_q_rows, int(caps.group_budget)),
        dtype=torch.float32,
    )
    topk_group_ids_b = _scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_indices_b_offset_bytes,
        shape=(workspace_q_rows, int(caps.group_budget)),
        dtype=torch.int32,
    )
    partial_output = _scratch_view(
        scratch_storage,
        offset_bytes=layout.partial_output_offset_bytes,
        shape=(
            int(state.max_split_row_product),
            int(caps.q_heads),
            int(caps.head_dim),
        ),
        dtype=torch.float32,
    )
    partial_lse = _scratch_view(
        scratch_storage,
        offset_bytes=layout.partial_lse_offset_bytes,
        shape=(int(state.max_split_row_product), int(caps.q_heads)),
        dtype=torch.float32,
    )
    output_lse = _scratch_view(
        scratch_storage,
        offset_bytes=layout.output_lse_offset_bytes,
        shape=(int(caps.max_q_rows), int(caps.q_heads)),
        dtype=torch.float32,
    )
    if draft_selection is not None:
        if not isinstance(draft_selection, DraftSelectionState):
            raise TypeError("draft_selection must be a qsa.DraftSelectionState")
        if caps.max_speculative_tokens < 1:
            raise ValueError("draft selection reuse requires speculative capacity")
        source_capacity = int(draft_selection.selected_positions.shape[0])
        if source_capacity < caps.max_q_rows:
            raise ValueError("draft selection state must cover planned query rows")
        for tensor, name, shape, dtype in (
            (
                draft_selection.selected_positions,
                "draft selected_positions",
                (source_capacity, caps.selection_width),
                torch.int32,
            ),
            (
                draft_selection.logical_positions,
                "draft logical_positions",
                (source_capacity,),
                torch.int64,
            ),
            (
                draft_selection.num_source_rows,
                "draft num_source_rows",
                (1,),
                torch.int32,
            ),
        ):
            _check_tensor(
                tensor,
                name=name,
                device=caps.device,
                shape=shape,
                dtype=dtype,
                contiguous=True,
            )
        _require_mutation_alias_contract(
            mutable=(
                ("draft selected_positions", draft_selection.selected_positions),
                ("draft logical_positions", draft_selection.logical_positions),
                ("draft num_source_rows", draft_selection.num_source_rows),
            ),
            read_only=tuple(
                (name, tensor)
                for name, tensor in (
                    ("scratch", scratch_storage),
                    ("output", output),
                    ("selected_positions", selected_positions),
                    ("main_k_cache", main_k_cache),
                    ("main_v_cache", main_v_cache),
                    ("k_descale", k_descale),
                    ("v_descale", v_descale),
                    ("main_block_table", main_block_table),
                    ("compressed_k_cache", compressed_k_cache),
                    ("compressed_block_table", compressed_block_table),
                    ("raw_k_ring", raw_k_ring),
                    ("raw_logical_positions", raw_logical_positions),
                    ("raw_rope_positions", raw_rope_positions),
                    ("raw_interval_start_positions", raw_interval_start_positions),
                    ("raw_state_slot_ids", raw_state_slot_ids),
                    ("index_q_norm_weight", index_q_norm_weight),
                    ("index_k_norm_weight", index_k_norm_weight),
                    ("rope_cos", rope_cos),
                    ("rope_sin", rope_sin),
                )
                if tensor is not None
            ),
        )
    if selection_stream is not None:
        if not isinstance(selection_stream, torch.cuda.Stream):
            raise TypeError("selection_stream must be a CUDA stream")
        if selection_stream.device != caps.device:
            raise ValueError("selection_stream must match the QSA plan device")
        if selection_done is None:
            with torch.cuda.device(caps.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "bind selection-stream resources before graph capture"
                    )
                selection_done = torch.cuda.Event()
                selection_done.record(selection_stream)
        elif not isinstance(selection_done, torch.cuda.Event):
            raise TypeError("selection_done must be a caller-established CUDA event")
        elif selection_done.device != caps.device:
            raise ValueError("selection_done must be recorded on the QSA plan device")
    elif selection_done is not None:
        raise ValueError("selection_done requires selection_stream")
    return Binding(
        selection_stream=selection_stream,
        _selection_done=selection_done,
        draft_selection=draft_selection,
        _draft_work_positions=_scratch_view(
            scratch_storage,
            offset_bytes=layout.draft_positions_offset_bytes,
            shape=(caps.max_batch, caps.selection_width + caps.max_speculative_tokens),
            dtype=torch.int32,
        )
        if caps.max_speculative_tokens > 0
        else None,
        state=state,
        plan=plan,
        shared_compressed_raw_pool=_overlaps(compressed_k_cache, raw_k_ring),
        scratch=scratch_storage,
        main_k_cache=main_k_cache,
        main_v_cache=main_v_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        main_block_table=main_block_table,
        compressed_k_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        raw_interval_start_positions=raw_interval_start_positions,
        raw_state_slot_ids=raw_state_slot_ids,
        index_q_norm_weight=index_q_norm_weight,
        index_k_norm_weight=index_k_norm_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        output=output,
        selected_positions=selected_positions,
        prepared_index_query=prepared_index_query,
        scores=scores,
        eligible_group_counts=eligible_group_counts,
        merge_lengths=merge_lengths,
        topk_values=topk_values,
        topk_group_ids=topk_group_ids,
        topk_values_b=topk_values_b,
        topk_group_ids_b=topk_group_ids_b,
        partial_output=partial_output,
        partial_lse=partial_lse,
        output_lse=output_lse,
    )


def bind(
    plan: Plan,
    **kwargs: object,
) -> Binding:
    """Bind caller-owned QSA buffers to a ready prepared plan."""
    state = require_prepared(plan, "attention.qsa")
    if not isinstance(state, _MaterializedPlan):
        raise TypeError("QSA plan has an invalid materialized state")
    return _bind_materialized(state, plan=plan, **kwargs)


def _qsa_decode_impl(
    query: torch.Tensor | None,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    output_lse_offset_bytes: int,
    *,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
    programs: QsaPrograms | None = None,
) -> None:
    """Launch selector state updates, optionally followed by sparse attention."""
    rows = int(index_query.shape[0])
    q_heads = int(output.shape[1])
    head_dim = int(output.shape[2])
    index_heads = int(index_query.shape[1])
    index_head_dim = int(index_query.shape[2])
    position_axes = int(rope_positions.shape[1])
    sections = (
        (int(mrope_section_0), int(mrope_section_1), int(mrope_section_2))
        if position_axes == 3
        else None
    )
    caps = _KernelCaps(
        max_batch=int(main_block_table.shape[0]),
        max_seq_len=int(max_seq_len),
        compressed_page_size=int(compressed_k_cache.shape[1]),
        q_heads=q_heads,
        kv_heads=int(main_k_cache.shape[2]),
        head_dim=head_dim,
        index_heads=index_heads,
        index_head_dim=index_head_dim,
        index_rotary_dim=int(index_rotary_dim),
        compress_ratio=int(compress_ratio),
        budget=int(budget),
        position_axes=position_axes,
        mrope_sections=sections,
        mrope_interleaved=bool(mrope_interleaved),
        rms_norm_eps=float(rms_norm_eps),
        raw_ring_capacity=int(raw_k_ring.shape[1]),
        max_speculative_tokens=int(max_speculative_tokens),
        dcp_size=int(dcp_size),
        dcp_rank=int(dcp_rank),
        cp_kv_cache_interleave_size=int(cp_kv_cache_interleave_size),
    )

    max_q_rows = int(selected_positions.shape[0])
    work_rows = int(workspace_q_rows)
    if not 0 < work_rows <= max_q_rows or rows > int(output.shape[0]):
        raise RuntimeError("invalid planned QSA workspace row capacity")
    group_budget = int(budget) // int(compress_ratio)
    prepared_query = _scratch_view(
        scratch,
        offset_bytes=int(prepared_query_offset_bytes),
        shape=(work_rows, index_heads, index_head_dim),
        dtype=torch.bfloat16,
    )
    scores = _scratch_view(
        scratch,
        offset_bytes=int(score_offset_bytes),
        shape=(work_rows, int(score_workspace_width)),
        dtype=torch.float32,
    )
    eligible_counts = _scratch_view(
        scratch,
        offset_bytes=int(eligible_counts_offset_bytes),
        shape=(work_rows,),
        dtype=torch.int32,
    )
    merge_lengths = _scratch_view(
        scratch,
        offset_bytes=int(merge_lengths_offset_bytes),
        shape=(work_rows,),
        dtype=torch.int32,
    )
    topk_values = _scratch_view(
        scratch,
        offset_bytes=int(topk_values_offset_bytes),
        shape=(work_rows, group_budget),
        dtype=torch.float32,
    )
    topk_ids = _scratch_view(
        scratch,
        offset_bytes=int(topk_indices_offset_bytes),
        shape=(work_rows, group_budget),
        dtype=torch.int32,
    )
    topk_values_b = _scratch_view(
        scratch,
        offset_bytes=int(topk_values_b_offset_bytes),
        shape=(work_rows, group_budget),
        dtype=torch.float32,
    )
    topk_ids_b = _scratch_view(
        scratch,
        offset_bytes=int(topk_indices_b_offset_bytes),
        shape=(work_rows, group_budget),
        dtype=torch.int32,
    )
    stable_topk_blocks = math.ceil(int(score_workspace_width) / _STABLE_TOPK_BLOCK)
    stable_offset = int(topk_offset_bytes)
    stable_count_nbytes = work_rows * stable_topk_blocks * torch.int32.itemsize
    tie_counts = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows, stable_topk_blocks),
        dtype=torch.int32,
    )
    stable_offset += stable_count_nbytes
    greater_counts = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows, stable_topk_blocks),
        dtype=torch.int32,
    )
    stable_offset += stable_count_nbytes
    stable_values = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows, group_budget),
        dtype=torch.float32,
    )
    stable_offset += work_rows * group_budget * torch.float32.itemsize
    stable_ids = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows, group_budget),
        dtype=torch.int32,
    )
    stable_offset += work_rows * group_budget * torch.int32.itemsize
    thresholds = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows,),
        dtype=torch.float32,
    )
    stable_offset += work_rows * torch.float32.itemsize
    greater_totals = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows,),
        dtype=torch.int32,
    )
    partial_output_storage = _scratch_view(
        scratch,
        offset_bytes=int(partial_output_offset_bytes),
        shape=(int(max_split_row_product), q_heads, head_dim),
        dtype=torch.float32,
    )
    partial_lse_storage = _scratch_view(
        scratch,
        offset_bytes=int(partial_lse_offset_bytes),
        shape=(int(max_split_row_product), q_heads),
        dtype=torch.float32,
    )
    output_lse_storage = _scratch_view(
        scratch,
        offset_bytes=int(output_lse_offset_bytes),
        shape=(max_q_rows, q_heads),
        dtype=torch.float32,
    )
    from ._kernels import (
        launch_compress_completed_groups,
        launch_expand_global_selected_groups,
        launch_expand_selected_groups,
        launch_prepare_index_query,
        launch_stabilize_topk,
        launch_stage_topk_carry,
        launch_topk_groups,
        launch_commit_raw_ring,
    )
    from ._score_cute import launch_score_representatives

    # Completion consumes the old ring before the current suffix can wrap it.
    launch_compress_completed_groups(
        raw_index_key=raw_index_key,
        _prepared=None if programs is None else programs.support,
        query_positions=query_positions,
        rope_positions=rope_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        key_norm_weight=index_k_norm_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        compressed_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        caps=caps,
    )
    launch_commit_raw_ring(
        raw_index_key=raw_index_key,
        query_positions=query_positions,
        rope_positions=rope_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        sequence_lengths=sequence_lengths,
        _prepared=None if programs is None else programs.support,
        is_prefilling=is_prefilling,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        raw_interval_start_positions=raw_interval_start_positions,
        caps=caps,
    )

    from ._sparse_gqa import launch_sparse_paged_gqa

    for row_offset in range(0, rows, work_rows):
        chunk_rows = min(work_rows, rows - row_offset)
        row_slice = slice(row_offset, row_offset + chunk_rows)
        chunk_request_ids = request_ids[row_slice]
        chunk_positions = query_positions[row_slice]
        chunk_rope = rope_positions[row_slice]
        chunk_prepared = prepared_query[:chunk_rows]
        chunk_scores = scores[:chunk_rows]
        chunk_eligible = eligible_counts[:chunk_rows]
        chunk_merge_lengths = merge_lengths[:chunk_rows]
        chunk_topk_values = topk_values[:chunk_rows]
        chunk_topk_ids = topk_ids[:chunk_rows]
        chunk_topk_values_b = topk_values_b[:chunk_rows]
        chunk_topk_ids_b = topk_ids_b[:chunk_rows]
        chunk_tie_counts = tie_counts[:chunk_rows]
        chunk_greater_counts = greater_counts[:chunk_rows]
        chunk_stable_values = stable_values[:chunk_rows]
        chunk_stable_ids = stable_ids[:chunk_rows]
        chunk_thresholds = thresholds[:chunk_rows]
        chunk_greater_totals = greater_totals[:chunk_rows]

        launch_prepare_index_query(
            index_query=index_query[row_slice],
            request_ids=chunk_request_ids,
            norm_weight=index_q_norm_weight,
            rope_positions=chunk_rope,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            prepared_query=chunk_prepared,
            caps=caps,
            _prepared=None if programs is None else programs.support,
        )
        prior_values = chunk_topk_values_b
        prior_ids = chunk_topk_ids_b
        final_ids = prior_ids
        for score_chunk in range(int(num_score_chunks)):
            group_offset = score_chunk * int(score_chunk_groups)
            group_count = min(
                int(score_chunk_groups), int(caps.max_groups) - group_offset
            )
            output_values = (
                chunk_topk_values if score_chunk % 2 == 0 else chunk_topk_values_b
            )
            output_ids = chunk_topk_ids if score_chunk % 2 == 0 else chunk_topk_ids_b
            if group_offset:
                launch_stage_topk_carry(
                    prior_values=prior_values,
                    eligible_counts=chunk_eligible,
                    scores=chunk_scores,
                    group_offset=group_offset,
                    group_budget=group_budget,
                    _prepared=None if programs is None else programs.support,
                )
            launch_score_representatives(
                prepared_query=chunk_prepared,
                query_positions=chunk_positions,
                request_ids=chunk_request_ids,
                sequence_lengths=sequence_lengths,
                compressed_cache=compressed_k_cache,
                compressed_block_table=compressed_block_table,
                scores=chunk_scores,
                eligible_counts=chunk_eligible,
                merge_lengths=chunk_merge_lengths,
                group_offset=group_offset,
                group_count=group_count,
                caps=caps,
                _prepared=None if programs is None else programs.score,
            )
            launch_topk_groups(
                scores=chunk_scores,
                eligible_counts=chunk_merge_lengths,
                topk_values=output_values,
                topk_group_ids=output_ids,
                group_budget=group_budget,
                _prepared=None if programs is None else programs.support,
            )
            launch_stabilize_topk(
                scores=chunk_scores,
                merge_lengths=chunk_merge_lengths,
                prior_ids=prior_ids,
                eligible_counts=chunk_eligible,
                topk_values=output_values,
                topk_group_ids=output_ids,
                tie_counts=chunk_tie_counts,
                greater_counts=chunk_greater_counts,
                stable_values=chunk_stable_values,
                stable_ids=chunk_stable_ids,
                thresholds=chunk_thresholds,
                greater_totals=chunk_greater_totals,
                group_offset=group_offset,
                group_budget=group_budget,
                _prepared=None if programs is None else programs.support,
            )
            prior_values, prior_ids = output_values, output_ids
            final_ids = output_ids

        selected = selected_positions[row_slice]
        if int(caps.dcp_size) == 1:
            launch_expand_selected_groups(
                topk_group_ids=final_ids,
                eligible_counts=chunk_eligible,
                query_positions=chunk_positions,
                selected_positions=selected,
                caps=caps,
                _prepared=None if programs is None else programs.support,
            )

        if query is None:
            continue

        if int(caps.dcp_size) > 1:
            launch_expand_global_selected_groups(
                topk_group_ids=final_ids,
                query_positions=chunk_positions,
                selected_positions=selected,
                caps=caps,
                _prepared=None if programs is None else programs.support,
            )
        block_n, splits = _target_splits(caps, chunk_rows)
        split_output = None
        split_lse = None
        if splits > 1:
            split_output = partial_output_storage[: chunk_rows * splits].view(
                chunk_rows, splits, q_heads, head_dim
            )
            split_lse = partial_lse_storage[: chunk_rows * splits].view(
                chunk_rows, splits, q_heads
            )
        active_output = output[row_slice]
        launch_sparse_paged_gqa(
            query=query[row_slice],
            key_cache=main_k_cache,
            value_cache=main_v_cache,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=main_block_table,
            request_ids=chunk_request_ids,
            selected_positions=selected,
            query_positions=chunk_positions,
            output=active_output,
            output_lse=(
                output_lse_storage[row_slice]
                if int(caps.dcp_size) > 1
                else None
            ),
            partial_output=split_output,
            partial_lse=split_lse,
            softmax_scale=1.0 / math.sqrt(head_dim),
            block_n=block_n,
            splits=splits,
            direct_kv_warps=int(sparse_gqa_direct_kv_warps),
            _prepared=None if programs is None else programs.sparse,
        )


_QSA_MUTATED_ARGUMENTS = (
    "scratch",
    "compressed_k_cache",
    "raw_k_ring",
    "raw_logical_positions",
    "raw_rope_positions",
    "raw_interval_start_positions",
    "output",
    "selected_positions",
)


@torch.library.custom_op(
    "b12x::qsa_decode",
    mutates_args=_QSA_MUTATED_ARGUMENTS,
)
def _qsa_decode_op(
    plan_handle: int,
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    output_lse_offset_bytes: int,
    selection_only: bool,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "attention.qsa", query.device)
    if not isinstance(state, _MaterializedPlan) or not isinstance(state.programs, QsaPrograms):
        raise RuntimeError("QSA custom op requires a retained native plan")
    _require_runtime_abi(
        state.abi, state.caps,
        request_ids=request_ids,
        rope_positions=rope_positions,
        index_query=index_query,
        raw_index_key=raw_index_key,
        main_k_cache=main_k_cache,
        main_v_cache=main_v_cache,
        main_block_table=main_block_table,
        compressed_k_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        raw_interval_start_positions=raw_interval_start_positions,
        raw_state_slot_ids=raw_state_slot_ids,
        index_q_norm_weight=index_q_norm_weight,
        index_k_norm_weight=index_k_norm_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch),
            ("compressed_k_cache", compressed_k_cache),
            ("raw_k_ring", raw_k_ring),
            ("raw_logical_positions", raw_logical_positions),
            ("raw_rope_positions", raw_rope_positions),
            ("raw_interval_start_positions", raw_interval_start_positions),
            ("output", output),
            ("selected_positions", selected_positions),
        ),
        read_only=(
            ("query", query),
            ("index_query", index_query),
            ("raw_index_key", raw_index_key),
            ("request_ids", request_ids),
            ("query_positions", query_positions),
            ("rope_positions", rope_positions),
            ("sequence_lengths", sequence_lengths),
            ("query_start_loc", query_start_loc),
            ("num_accepted_tokens", num_accepted_tokens),
            ("is_prefilling", is_prefilling),
        ),
    )
    _qsa_decode_impl(
        None if selection_only else query,
        index_query,
        raw_index_key,
        request_ids,
        query_positions,
        rope_positions,
        sequence_lengths,
        query_start_loc,
        num_accepted_tokens,
        is_prefilling,
        scratch,
        main_k_cache,
        main_v_cache,
        k_descale,
        v_descale,
        main_block_table,
        compressed_k_cache,
        compressed_block_table,
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
        raw_state_slot_ids,
        index_q_norm_weight,
        index_k_norm_weight,
        rope_cos,
        rope_sin,
        output,
        selected_positions,
        sparse_gqa_direct_kv_warps,
        max_seq_len,
        max_speculative_tokens,
        compress_ratio,
        budget,
        index_rotary_dim,
        mrope_section_0,
        mrope_section_1,
        mrope_section_2,
        mrope_interleaved,
        rms_norm_eps,
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
        max_split_row_product,
        workspace_q_rows,
        prepared_query_offset_bytes,
        score_offset_bytes,
        eligible_counts_offset_bytes,
        merge_lengths_offset_bytes,
        topk_values_offset_bytes,
        topk_indices_offset_bytes,
        topk_values_b_offset_bytes,
        topk_indices_b_offset_bytes,
        topk_offset_bytes,
        partial_output_offset_bytes,
        partial_lse_offset_bytes,
        output_lse_offset_bytes,
        dcp_size=state.caps.dcp_size,
        dcp_rank=state.caps.dcp_rank,
        cp_kv_cache_interleave_size=state.caps.cp_kv_cache_interleave_size,
        programs=state.programs,
    )


@_qsa_decode_op.register_fake
def _qsa_decode_fake(
    plan_handle: int,
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    output_lse_offset_bytes: int,
    selection_only: bool,
) -> None:
    return None


def _raw_views_from_compressed_pool(
    compressed_k_cache: torch.Tensor,
    *,
    max_raw_state_slots: int,
    raw_ring_capacity: int,
    index_head_dim: int,
    position_axes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    page_elements = int(compressed_k_cache.shape[1]) * int(compressed_k_cache.shape[2])
    raw_k_ring = compressed_k_cache.as_strided(
        (int(max_raw_state_slots), int(raw_ring_capacity), int(index_head_dim)),
        (page_elements, int(index_head_dim), 1),
        storage_offset=int(compressed_k_cache.storage_offset()),
    )
    pool_i64 = compressed_k_cache.view(torch.uint8).reshape(-1).view(torch.int64)
    page_i64 = (
        int(compressed_k_cache.stride(0))
        * int(compressed_k_cache.element_size())
        // torch.int64.itemsize
    )
    payload_i64 = (
        int(raw_ring_capacity)
        * int(index_head_dim)
        * torch.bfloat16.itemsize
        // torch.int64.itemsize
    )
    base_i64 = int(pool_i64.storage_offset())
    raw_logical_positions = pool_i64.as_strided(
        (int(max_raw_state_slots), int(raw_ring_capacity)),
        (page_i64, 1),
        storage_offset=base_i64 + payload_i64,
    )
    raw_rope_positions = pool_i64.as_strided(
        (
            int(max_raw_state_slots),
            int(raw_ring_capacity),
            int(position_axes),
        ),
        (page_i64, int(position_axes), 1),
        storage_offset=base_i64 + payload_i64 + int(raw_ring_capacity),
    )
    raw_interval_start_positions = pool_i64.as_strided(
        (int(max_raw_state_slots),),
        (page_i64,),
        storage_offset=(
            base_i64 + payload_i64 + int(raw_ring_capacity) * (1 + int(position_axes))
        ),
    )
    return (
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
    )


_QSA_SHARED_MUTATED_ARGUMENTS = (
    "scratch",
    "compressed_raw_pool",
    "output",
    "selected_positions",
)


@torch.library.custom_op(
    "b12x::qsa_decode_shared",
    mutates_args=_QSA_SHARED_MUTATED_ARGUMENTS,
)
def _qsa_decode_shared_op(
    plan_handle: int,
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_raw_pool: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_raw_state_slots: int,
    raw_ring_capacity: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    output_lse_offset_bytes: int,
    selection_only: bool,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "attention.qsa", query.device)
    if not isinstance(state, _MaterializedPlan) or not isinstance(state.programs, QsaPrograms):
        raise RuntimeError("QSA custom op requires a retained native plan")
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch),
            ("compressed_raw_pool", compressed_raw_pool),
            ("output", output),
            ("selected_positions", selected_positions),
        ),
        read_only=(
            ("query", query),
            ("index_query", index_query),
            ("raw_index_key", raw_index_key),
            ("request_ids", request_ids),
            ("query_positions", query_positions),
            ("rope_positions", rope_positions),
            ("sequence_lengths", sequence_lengths),
            ("query_start_loc", query_start_loc),
            ("num_accepted_tokens", num_accepted_tokens),
            ("is_prefilling", is_prefilling),
        ),
    )
    (
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
    ) = _raw_views_from_compressed_pool(
        compressed_raw_pool,
        max_raw_state_slots=int(max_raw_state_slots),
        raw_ring_capacity=int(raw_ring_capacity),
        index_head_dim=int(index_query.shape[2]),
        position_axes=int(rope_positions.shape[1]),
    )
    _require_runtime_abi(
        state.abi, state.caps,
        request_ids=request_ids,
        rope_positions=rope_positions,
        index_query=index_query,
        raw_index_key=raw_index_key,
        main_k_cache=main_k_cache,
        main_v_cache=main_v_cache,
        main_block_table=main_block_table,
        compressed_k_cache=compressed_raw_pool,
        compressed_block_table=compressed_block_table,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        raw_interval_start_positions=raw_interval_start_positions,
        raw_state_slot_ids=raw_state_slot_ids,
        index_q_norm_weight=index_q_norm_weight,
        index_k_norm_weight=index_k_norm_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    _qsa_decode_impl(
        None if selection_only else query,
        index_query,
        raw_index_key,
        request_ids,
        query_positions,
        rope_positions,
        sequence_lengths,
        query_start_loc,
        num_accepted_tokens,
        is_prefilling,
        scratch,
        main_k_cache,
        main_v_cache,
        k_descale,
        v_descale,
        main_block_table,
        compressed_raw_pool,
        compressed_block_table,
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
        raw_state_slot_ids,
        index_q_norm_weight,
        index_k_norm_weight,
        rope_cos,
        rope_sin,
        output,
        selected_positions,
        sparse_gqa_direct_kv_warps,
        max_seq_len,
        max_speculative_tokens,
        compress_ratio,
        budget,
        index_rotary_dim,
        mrope_section_0,
        mrope_section_1,
        mrope_section_2,
        mrope_interleaved,
        rms_norm_eps,
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
        max_split_row_product,
        workspace_q_rows,
        prepared_query_offset_bytes,
        score_offset_bytes,
        eligible_counts_offset_bytes,
        merge_lengths_offset_bytes,
        topk_values_offset_bytes,
        topk_indices_offset_bytes,
        topk_values_b_offset_bytes,
        topk_indices_b_offset_bytes,
        topk_offset_bytes,
        partial_output_offset_bytes,
        partial_lse_offset_bytes,
        output_lse_offset_bytes,
        dcp_size=state.caps.dcp_size,
        dcp_rank=state.caps.dcp_rank,
        cp_kv_cache_interleave_size=state.caps.cp_kv_cache_interleave_size,
        programs=state.programs,
    )


@_qsa_decode_shared_op.register_fake
def _qsa_decode_shared_fake(
    plan_handle: int,
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_raw_pool: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_raw_state_slots: int,
    raw_ring_capacity: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    output_lse_offset_bytes: int,
    selection_only: bool,
) -> None:
    return None



def _run(
    binding: Binding,
    *,
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    selection_only: bool = False,
    programs: QsaPrograms | None = None,
) -> torch.Tensor:
    """Validate a complete transaction and launch its combined or selector stage."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a qsa.Binding")
    caps = binding.state.caps
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a tensor")
    if index_query.device.type != "cuda":
        raise RuntimeError("QSA GPU decode requires a CUDA device")
    rows = int(index_query.shape[0])
    if not 0 < rows <= int(caps.max_q_rows):
        raise ValueError("query rows must be within the planned decode capacity")
    if rows > int(binding.output.shape[0]):
        raise ValueError("query rows exceed the bound QSA output capacity")
    dynamic_specs = (
        (
            index_query,
            "index_query",
            (rows, int(caps.index_heads), int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (
            raw_index_key,
            "raw_index_key",
            (rows, int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (request_ids, "request_ids", (rows,), (torch.int32, torch.int64)),
        (query_positions, "query_positions", (rows,), (torch.int64,)),
        (
            rope_positions,
            "rope_positions",
            (rows, int(caps.position_axes)),
            (torch.int64,),
        ),
        (
            sequence_lengths,
            "sequence_lengths",
            (int(caps.max_batch),),
            (torch.int32,),
        ),
        (
            query_start_loc,
            "query_start_loc",
            (int(caps.max_batch) + 1,),
            (torch.int32,),
        ),
        (
            num_accepted_tokens,
            "num_accepted_tokens",
            (int(caps.max_batch),),
            (torch.int32,),
        ),
        (
            is_prefilling,
            "is_prefilling",
            (int(caps.max_batch),),
            (torch.bool,),
        ),
    )
    dynamic_specs = (
        (
            query,
            "query",
            (rows, int(caps.q_heads), int(caps.head_dim)),
            (torch.bfloat16,),
        ),
        *dynamic_specs,
    )
    for tensor, name, shape, dtypes in dynamic_specs:
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=shape,
            dtype=dtypes,
            contiguous=name not in ("rope_positions", "index_query", "raw_index_key"),
        )
    for tensor, name in (
        (index_query, "index_query"),
        (raw_index_key, "raw_index_key"),
    ):
        tail_elements = math.prod(tensor.shape[1:])
        if (
            int(tensor.stride(0)) < tail_elements
            or int(tensor.stride(-1)) != 1
            or (tensor.ndim == 3 and int(tensor.stride(1)) != int(tensor.shape[2]))
        ):
            raise ValueError(f"{name} requires non-overlapping dense rows")
    if int(rope_positions.stride(0)) <= 0 or int(rope_positions.stride(1)) <= 0:
        raise ValueError("rope_positions must have positive row and axis strides")
    _require_non_overlapping_layout("rope_positions", rope_positions)
    if not torch.compiler.is_compiling():
        _require_runtime_abi(
            binding.state.abi, caps,
            request_ids=request_ids,
            rope_positions=rope_positions,
            index_query=index_query,
            raw_index_key=raw_index_key,
            main_k_cache=binding.main_k_cache,
            main_v_cache=binding.main_v_cache,
            main_block_table=binding.main_block_table,
            compressed_k_cache=binding.compressed_k_cache,
            compressed_block_table=binding.compressed_block_table,
            raw_k_ring=binding.raw_k_ring,
            raw_logical_positions=binding.raw_logical_positions,
            raw_rope_positions=binding.raw_rope_positions,
            raw_interval_start_positions=binding.raw_interval_start_positions,
            raw_state_slot_ids=binding.raw_state_slot_ids,
            index_q_norm_weight=binding.index_q_norm_weight,
            index_k_norm_weight=binding.index_k_norm_weight,
            rope_cos=binding.rope_cos,
            rope_sin=binding.rope_sin,
        )
        _require_mutation_alias_contract(
            mutable=(
                ("scratch", binding.scratch),
                ("compressed_k_cache", binding.compressed_k_cache),
                ("raw_k_ring", binding.raw_k_ring),
                ("raw_logical_positions", binding.raw_logical_positions),
                ("raw_rope_positions", binding.raw_rope_positions),
                (
                    "raw_interval_start_positions",
                    binding.raw_interval_start_positions,
                ),
                ("output", binding.output),
                ("selected_positions", binding.selected_positions),
            ),
            read_only=tuple(
                (name, tensor) for tensor, name, _shape, _dtypes in dynamic_specs
            ),
        )

    if programs is not None:
        _qsa_decode_impl(
            query, index_query, raw_index_key, request_ids, query_positions,
            rope_positions, sequence_lengths, query_start_loc, num_accepted_tokens,
            is_prefilling, binding.scratch, binding.main_k_cache, binding.main_v_cache,
            binding.k_descale, binding.v_descale, binding.main_block_table,
            binding.compressed_k_cache, binding.compressed_block_table,
            binding.raw_k_ring, binding.raw_logical_positions,
            binding.raw_rope_positions, binding.raw_interval_start_positions,
            binding.raw_state_slot_ids, binding.index_q_norm_weight,
            binding.index_k_norm_weight, binding.rope_cos, binding.rope_sin,
            binding.output, binding.selected_positions,
            int(binding.state.config.sparse_gqa_direct_kv_warps),
            int(caps.max_seq_len), int(caps.max_speculative_tokens),
            int(caps.compress_ratio), int(caps.budget), int(caps.index_rotary_dim),
            *(caps.mrope_sections or (0, 0, 0)), bool(caps.mrope_interleaved),
            float(caps.rms_norm_eps), int(binding.state.score_chunk_groups),
            int(binding.state.score_workspace_width), int(binding.state.num_score_chunks),
            int(binding.state.max_split_row_product), int(binding.state.workspace_q_rows),
            int(binding.state._layout.prepared_query_offset_bytes),
            int(binding.state._layout.score_offset_bytes),
            int(binding.state._layout.eligible_counts_offset_bytes),
            int(binding.state._layout.merge_lengths_offset_bytes),
            int(binding.state._layout.topk_values_offset_bytes),
            int(binding.state._layout.topk_indices_offset_bytes),
            int(binding.state._layout.topk_values_b_offset_bytes),
            int(binding.state._layout.topk_indices_b_offset_bytes),
            int(binding.state._layout.topk_offset_bytes),
            int(binding.state._layout.partial_output_offset_bytes),
            int(binding.state._layout.partial_lse_offset_bytes),
            int(binding.state._layout.output_lse_offset_bytes),
            dcp_size=caps.dcp_size,
            dcp_rank=caps.dcp_rank,
            cp_kv_cache_interleave_size=caps.cp_kv_cache_interleave_size,
            programs=programs,
        )
        return binding.output[:rows]

    sections = caps.mrope_sections or (0, 0, 0)
    layout = binding.state._layout
    if binding.shared_compressed_raw_pool:
        _qsa_decode_shared_op(
            binding.plan.handle,
            query,
            index_query,
            raw_index_key,
            request_ids,
            query_positions,
            rope_positions,
            sequence_lengths,
            query_start_loc,
            num_accepted_tokens,
            is_prefilling,
            binding.scratch,
            binding.main_k_cache,
            binding.main_v_cache,
            binding.k_descale,
            binding.v_descale,
            binding.main_block_table,
            binding.compressed_k_cache,
            binding.compressed_block_table,
            binding.raw_state_slot_ids,
            binding.index_q_norm_weight,
            binding.index_k_norm_weight,
            binding.rope_cos,
            binding.rope_sin,
            binding.output,
            binding.selected_positions,
            int(binding.state.config.sparse_gqa_direct_kv_warps),
            int(caps.max_raw_state_slots),
            int(caps.raw_ring_capacity),
            int(caps.max_seq_len),
            int(caps.max_speculative_tokens),
            int(caps.compress_ratio),
            int(caps.budget),
            int(caps.index_rotary_dim),
            int(sections[0]),
            int(sections[1]),
            int(sections[2]),
            bool(caps.mrope_interleaved),
            float(caps.rms_norm_eps),
            int(binding.state.score_chunk_groups),
            int(binding.state.score_workspace_width),
            int(binding.state.num_score_chunks),
            int(binding.state.max_split_row_product),
            int(binding.state.workspace_q_rows),
            int(layout.prepared_query_offset_bytes),
            int(layout.score_offset_bytes),
            int(layout.eligible_counts_offset_bytes),
            int(layout.merge_lengths_offset_bytes),
            int(layout.topk_values_offset_bytes),
            int(layout.topk_indices_offset_bytes),
            int(layout.topk_values_b_offset_bytes),
            int(layout.topk_indices_b_offset_bytes),
            int(layout.topk_offset_bytes),
            int(layout.partial_output_offset_bytes),
            int(layout.partial_lse_offset_bytes),
            int(layout.output_lse_offset_bytes),
            selection_only,
        )
        return binding.output[:rows]
    _qsa_decode_op(
        binding.plan.handle,
        query,
        index_query,
        raw_index_key,
        request_ids,
        query_positions,
        rope_positions,
        sequence_lengths,
        query_start_loc,
        num_accepted_tokens,
        is_prefilling,
        binding.scratch,
        binding.main_k_cache,
        binding.main_v_cache,
        binding.k_descale,
        binding.v_descale,
        binding.main_block_table,
        binding.compressed_k_cache,
        binding.compressed_block_table,
        binding.raw_k_ring,
        binding.raw_logical_positions,
        binding.raw_rope_positions,
        binding.raw_interval_start_positions,
        binding.raw_state_slot_ids,
        binding.index_q_norm_weight,
        binding.index_k_norm_weight,
        binding.rope_cos,
        binding.rope_sin,
        binding.output,
        binding.selected_positions,
        int(binding.state.config.sparse_gqa_direct_kv_warps),
        int(caps.max_seq_len),
        int(caps.max_speculative_tokens),
        int(caps.compress_ratio),
        int(caps.budget),
        int(caps.index_rotary_dim),
        int(sections[0]),
        int(sections[1]),
        int(sections[2]),
        bool(caps.mrope_interleaved),
        float(caps.rms_norm_eps),
        int(binding.state.score_chunk_groups),
        int(binding.state.score_workspace_width),
        int(binding.state.num_score_chunks),
        int(binding.state.max_split_row_product),
        int(binding.state.workspace_q_rows),
        int(layout.prepared_query_offset_bytes),
        int(layout.score_offset_bytes),
        int(layout.eligible_counts_offset_bytes),
        int(layout.merge_lengths_offset_bytes),
        int(layout.topk_values_offset_bytes),
        int(layout.topk_indices_offset_bytes),
        int(layout.topk_values_b_offset_bytes),
        int(layout.topk_indices_b_offset_bytes),
        int(layout.topk_offset_bytes),
        int(layout.partial_output_offset_bytes),
        int(layout.partial_lse_offset_bytes),
        int(layout.output_lse_offset_bytes),
        selection_only,
    )
    return binding.output[:rows]


def run(
    binding: Binding,
    *,
    query: torch.Tensor,
    index_query: torch.Tensor | None = None,
    raw_index_key: torch.Tensor | None = None,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    is_prefilling: torch.Tensor | None = None,
    index_ready: torch.cuda.Event | None = None,
    reuse: DraftSelectionReuse | None = None,
) -> torch.Tensor:
    """Run one packed QSA decode transaction after main K/V writes are enqueued.

    Main Q/K/V writes precede this call on the calling stream. By default all
    QSA work follows them on that stream. A binding with ``selection_stream``
    instead requires ``index_ready``, recorded after all selector inputs,
    metadata and prior state writes, before independent main Q/K/V work.
    Selection waits for that event and runs on the bound stream; attention
    joins it on the calling stream. Intermediate selected positions remain an
    internal handoff within this call. Bind and prepare before graph capture,
    and record the readiness event inside the graph for captured producers.

    A bound ``draft_selection`` records anchors during ordinary execution.
    ``reuse=DraftSelectionReuse(source_rows)`` uses the accepted anchors selected
    by the caller-owned request-to-source map and appends only the causal draft tail.
    This mode requires one query per request inside the same draft round;
    selector projections/metadata and ``index_ready`` must be omitted. It
    preserves selector caches and the anchor buffers. An anchor outside the
    recorded rows or a query position outside its causal tail selects no
    positions. Request identity and round lifetime are caller invariants; see
    ``DraftSelectionState``. Target layers
    must not use it.

    Active request intervals are the dense prefix encoded by
    ``query_start_loc``.  Remaining boundaries repeat the live-row count and
    remaining query rows contain ``-1`` metadata.  ``sequence_lengths`` includes
    every row in the corresponding current interval.  The read-only main K/V
    cache must contain every selected original-token K/V row before attention
    executes on the calling stream.

    ``num_accepted_tokens`` commits the preceding verification interval:
    each active value is in ``[1, 1 + max_speculative_tokens]`` and counts the
    accepted prefix including its guaranteed or recovered token.  The mutable
    ``raw_interval_start_positions`` entry for the request's persistent state
    slot records the preceding interval's first row before the call and the
    current interval's first row after it.  Candidate rows may
    remain physically resident after rejection, but exact logical tags prevent
    them from becoming eligible.  Dynamic metadata is consumed as given; the
    caller keeps it consistent with the persistent state.

    With shared compressed/raw backing, requests that have no rows in the
    current packed call remain live page owners until eviction.  Their
    ``sequence_lengths``, ``compressed_block_table`` entries, and
    ``raw_state_slot_ids`` mappings must remain valid.  Zero sequence lengths
    and ``-1`` table or slot entries describe only unused or evicted capacity.

    For the first decode interval beginning at logical position ``N``, seed
    the persistent interval anchor to ``N - num_accepted_tokens``.  If the
    interval can complete a group whose prefix came from prefill, the raw ring
    must already contain exact logical tags, raw index keys, and RoPE positions
    for the trailing open group.  An anchor of ``-1`` is valid only for the
    position-zero initialization with one accepted token.  ``rope_positions``
    may be any non-overlapping 2-D view with positive row and axis strides,
    including ``positions_by_axis.T``.
    Selector projections may be disjoint dense-row views of a fused Q/K
    projection; their positive row strides are consumed without staging copies.
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a qsa.Binding")
    if not isinstance(binding.plan, Plan):
        raise TypeError("QSA binding requires a prepared plan")
    if binding.state.caps.dcp_size > 1:
        raise ValueError("DCP QSA requires select() followed by attend()")
    if reuse is not None:
        if not isinstance(reuse, DraftSelectionReuse):
            raise TypeError("reuse must be a qsa.DraftSelectionReuse")
        state = binding.draft_selection
        if state is None:
            raise ValueError("draft selection reuse requires bound draft state")
        if any(
            value is not None
            for value in (
                index_query,
                raw_index_key,
                rope_positions,
                sequence_lengths,
                query_start_loc,
                num_accepted_tokens,
                is_prefilling,
                index_ready,
            )
        ):
            raise ValueError("draft selection reuse does not accept selector inputs")
        rows = int(query.shape[0])
        if not 0 < rows <= min(binding.state.caps.max_batch, binding.output.shape[0]):
            raise ValueError("draft selection reuse requires one row per request")
        caps = binding.state.caps
        _check_tensor(
            query,
            name="query",
            device=caps.device,
            shape=(rows, caps.q_heads, caps.head_dim),
            dtype=torch.bfloat16,
            contiguous=True,
        )
        for tensor, name, dtype in (
            (request_ids, "request_ids", (torch.int32, torch.int64)),
            (query_positions, "query_positions", torch.int64),
        ):
            _check_tensor(
                tensor,
                name=name,
                device=binding.state.caps.device,
                shape=(rows,),
                dtype=dtype,
                contiguous=True,
            )
        _check_tensor(
            reuse.source_rows,
            name="source_rows",
            device=caps.device,
            shape=(caps.max_batch,),
            dtype=(torch.int32, torch.int64),
            contiguous=True,
        )
        from ._draft_selection import prepare_selection, validate_buffers

        validate_buffers(
            [binding.scratch, binding.output],
            [query, request_ids, query_positions, reuse.source_rows],
        )
        prepare_selection(
            state._storage,
            state.plan.max_source_rows,
            state.plan.selection_width,
            reuse.source_rows,
            request_ids,
            query_positions,
            binding.scratch,
            binding.state._layout.draft_positions_offset_bytes,
            caps.max_batch,
            caps.max_speculative_tokens,
            caps.dcp_size,
            caps.dcp_rank,
            caps.cp_kv_cache_interleave_size,
        )
        return _run_attention(
            binding,
            query=query,
            request_ids=request_ids,
            query_positions=query_positions,
            reuse=True,
        )
    if any(
        value is None
        for value in (
            index_query,
            raw_index_key,
            rope_positions,
            sequence_lengths,
            query_start_loc,
            num_accepted_tokens,
            is_prefilling,
        )
    ):
        raise ValueError("ordinary QSA execution requires every selector input")
    inputs = dict(
        query=query,
        index_query=index_query,
        raw_index_key=raw_index_key,
        request_ids=request_ids,
        query_positions=query_positions,
        rope_positions=rope_positions,
        sequence_lengths=sequence_lengths,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        is_prefilling=is_prefilling,
    )
    if binding.draft_selection is not None:
        from ._draft_selection import validate_buffers

        state = binding.draft_selection
        validate_buffers(
            [
                state._storage,
                binding.selected_positions,
            ],
            list(inputs.values()),
        )
    stream = binding.selection_stream
    if stream is None:
        if index_ready is not None:
            raise ValueError("index_ready requires a binding with selection_stream")
        result = _run(binding, **inputs)
        _record_draft_anchors(binding, query_positions)
        return result
    if index_ready is None:
        raise ValueError("selection_stream requires an index_ready event")
    if not torch.compiler.is_compiling():
        if not isinstance(index_ready, torch.cuda.Event):
            raise TypeError("index_ready must be a recorded CUDA event")
        if index_ready.device != binding.state.caps.device:
            raise ValueError("index_ready must be recorded on the QSA plan device")
    with torch.cuda.device(binding.state.caps.device):
        main_stream = torch.cuda.current_stream()
        stream.wait_event(index_ready)
        with torch.cuda.stream(stream):
            _run(binding, **inputs, selection_only=True)
            binding._selection_done.record(stream)
        main_stream.wait_event(binding._selection_done)
        result = _run_attention(
            binding,
            query=query,
            request_ids=request_ids,
            query_positions=query_positions,
        )
        _record_draft_anchors(binding, query_positions)
        return result


def select(
    binding: Binding,
    *,
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
) -> LocalSelection:
    """Update selector state and return rank-local group IDs and scores."""
    if not isinstance(binding, Binding) or not isinstance(binding.plan, Plan):
        raise TypeError("QSA selection requires a prepared binding")
    if binding.state.caps.dcp_size <= 1:
        raise ValueError("select() is reserved for context-parallel QSA")
    if binding.selection_stream is not None:
        raise ValueError("DCP QSA selection currently requires the calling stream")
    _run(
        binding,
        query=query,
        index_query=index_query,
        raw_index_key=raw_index_key,
        request_ids=request_ids,
        query_positions=query_positions,
        rope_positions=rope_positions,
        sequence_lengths=sequence_lengths,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        is_prefilling=is_prefilling,
        selection_only=True,
    )
    if binding.state.num_score_chunks % 2:
        group_ids, scores = binding.topk_group_ids, binding.topk_values
    else:
        group_ids, scores = binding.topk_group_ids_b, binding.topk_values_b
    rows = int(query.shape[0])
    return LocalSelection(group_ids=group_ids[:rows], scores=scores[:rows])


def attend(
    binding: Binding,
    *,
    query: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    selection: LocalSelection,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand globally merged groups and return rank-local output and LSE."""
    if not isinstance(binding, Binding) or not isinstance(binding.plan, Plan):
        raise TypeError("QSA attention requires a prepared binding")
    caps = binding.state.caps
    if caps.dcp_size <= 1:
        raise ValueError("attend() is reserved for context-parallel QSA")
    if not isinstance(selection, LocalSelection):
        raise TypeError("selection must be a qsa.LocalSelection")
    rows = int(query.shape[0])
    if not 0 < rows <= min(caps.max_q_rows, binding.selected_positions.shape[0]):
        raise ValueError("query rows exceed the bound QSA selection capacity")
    _check_tensor(
        query_positions,
        name="query_positions",
        device=caps.device,
        shape=(rows,),
        dtype=torch.int64,
        contiguous=True,
    )
    _check_tensor(
        selection.group_ids,
        name="selection.group_ids",
        device=caps.device,
        shape=(rows, caps.group_budget),
        dtype=torch.int32,
        contiguous=True,
    )
    from ._kernels import launch_expand_global_selected_groups

    programs = binding.state.programs
    if not isinstance(programs, QsaPrograms):
        raise RuntimeError("QSA attention requires retained native programs")
    launch_expand_global_selected_groups(
        topk_group_ids=selection.group_ids,
        query_positions=query_positions,
        selected_positions=binding.selected_positions[:rows],
        caps=caps,
        _prepared=programs.support,
    )
    output = _run_attention(
        binding,
        query=query,
        request_ids=request_ids,
        query_positions=query_positions,
    )
    _record_draft_anchors(binding, query_positions)
    return output, binding.output_lse[:rows]


def attend_reuse(
    binding: Binding,
    *,
    query: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    reuse: DraftSelectionReuse,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend with a previously recorded rank-local draft selection."""
    if not isinstance(binding, Binding) or not isinstance(binding.plan, Plan):
        raise TypeError("QSA draft reuse requires a prepared binding")
    caps = binding.state.caps
    if caps.dcp_size <= 1:
        raise ValueError("attend_reuse() is reserved for context-parallel QSA")
    state = binding.draft_selection
    if state is None or not isinstance(reuse, DraftSelectionReuse):
        raise ValueError("QSA draft reuse requires bound draft state and mapping")
    rows = int(query.shape[0])
    if not 0 < rows <= min(caps.max_batch, binding.output.shape[0]):
        raise ValueError("draft selection reuse requires one row per request")
    for tensor, name, shape, dtype in (
        (query, "query", (rows, caps.q_heads, caps.head_dim), torch.bfloat16),
        (request_ids, "request_ids", (rows,), (torch.int32, torch.int64)),
        (query_positions, "query_positions", (rows,), torch.int64),
        (
            reuse.source_rows,
            "source_rows",
            (caps.max_batch,),
            (torch.int32, torch.int64),
        ),
    ):
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=shape,
            dtype=dtype,
            contiguous=True,
        )
    from ._draft_selection import prepare_selection, validate_buffers

    validate_buffers(
        [binding.scratch, binding.output],
        [query, request_ids, query_positions, reuse.source_rows],
    )
    prepare_selection(
        state._storage,
        state.plan.max_source_rows,
        state.plan.selection_width,
        reuse.source_rows,
        request_ids,
        query_positions,
        binding.scratch,
        binding.state._layout.draft_positions_offset_bytes,
        caps.max_batch,
        caps.max_speculative_tokens,
        caps.dcp_size,
        caps.dcp_rank,
        caps.cp_kv_cache_interleave_size,
    )
    output = _run_attention(
        binding,
        query=query,
        request_ids=request_ids,
        query_positions=query_positions,
        reuse=True,
    )
    return output, binding.output_lse[:rows]


def _record_draft_anchors(binding: Binding, positions: torch.Tensor) -> None:
    if binding.draft_selection is not None:
        from ._draft_selection import record_anchors

        state = binding.draft_selection
        record_anchors(
            positions,
            binding.selected_positions,
            state._storage,
            state.plan.max_source_rows,
            state.plan.selection_width,
            binding._record_draft_enabled,
        )


@torch.library.custom_op("b12x::qsa_attention", mutates_args=("scratch", "output"))
def _qsa_attention_op(
    plan_handle: int,
    query: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    selected_positions: torch.Tensor | None,
    scratch: torch.Tensor,
    output: torch.Tensor,
    work_rows: int,
    max_split_row_product: int,
    partial_output_offset: int,
    partial_lse_offset: int,
    output_lse_offset: int,
    direct_kv_warps: int,
    draft_positions_offset: int,
    draft_width: int,
    draft_capacity: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "attention.qsa", query.device)
    if not isinstance(state, _MaterializedPlan) or not isinstance(state.programs, QsaPrograms):
        raise RuntimeError("QSA attention custom op requires a retained native plan")
    from ._sparse_gqa import launch_sparse_paged_gqa
    from ._sparse_gqa_cute_config import BLOCK_N

    # Storage addresses exist at the custom-op runtime boundary, not in tracing.
    _require_mutation_alias_contract(
        mutable=(("scratch", scratch), ("output", output)),
        read_only=(
            ("query", query),
            ("request_ids", request_ids),
            ("query_positions", query_positions),
        )
        + (
            ()
            if selected_positions is None
            else (("selected_positions", selected_positions),)
        ),
    )
    rows, q_heads, head_dim = map(int, query.shape)
    draft_reuse = selected_positions is None
    if draft_reuse:
        selected_positions = _scratch_view(
            scratch,
            offset_bytes=draft_positions_offset,
            shape=(draft_capacity, draft_width),
            dtype=torch.int32,
        )
    partial_output = _scratch_view(
        scratch,
        offset_bytes=partial_output_offset,
        shape=(max_split_row_product, q_heads, head_dim),
        dtype=torch.float32,
    )
    partial_lse = _scratch_view(
        scratch,
        offset_bytes=partial_lse_offset,
        shape=(max_split_row_product, q_heads),
        dtype=torch.float32,
    )
    output_lse = _scratch_view(
        scratch,
        offset_bytes=output_lse_offset,
        shape=(state.caps.max_q_rows, q_heads),
        dtype=torch.float32,
    )
    # Use the same row-chunk and native CuTe split policy as the combined call.
    for start in range(0, rows, work_rows):
        count = min(work_rows, rows - start)
        row_slice = slice(start, start + count)
        splits = _qwen_row_splits(count)
        launch_sparse_paged_gqa(
            query=query[row_slice],
            key_cache=main_k_cache,
            value_cache=main_v_cache,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=main_block_table,
            request_ids=request_ids[row_slice],
            selected_positions=selected_positions[row_slice],
            query_positions=query_positions[row_slice],
            output=output[row_slice],
            output_lse=(
                output_lse[row_slice]
                if int(state.caps.dcp_size) > 1
                else None
            ),
            partial_output=(
                partial_output[: count * splits].view(count, splits, q_heads, head_dim)
                if splits > 1
                else None
            ),
            partial_lse=(
                partial_lse[: count * splits].view(count, splits, q_heads)
                if splits > 1
                else None
            ),
            softmax_scale=1.0 / math.sqrt(head_dim),
            block_n=BLOCK_N,
            splits=splits,
            direct_kv_warps=direct_kv_warps,
            _prepared=(
                state.programs.sparse_draft
                if draft_reuse
                else state.programs.sparse
            ),
        )


@_qsa_attention_op.register_fake
def _qsa_attention_fake(
    plan_handle: int,
    query: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    selected_positions: torch.Tensor | None,
    scratch: torch.Tensor,
    output: torch.Tensor,
    work_rows: int,
    max_split_row_product: int,
    partial_output_offset: int,
    partial_lse_offset: int,
    output_lse_offset: int,
    direct_kv_warps: int,
    draft_positions_offset: int,
    draft_width: int,
    draft_capacity: int,
) -> None:
    return None


def _run_attention(
    binding: Binding,
    *,
    query: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    reuse: bool = False,
) -> torch.Tensor:
    """Consume the bound selection within one run call."""

    if not isinstance(binding, Binding):
        raise TypeError("binding must be a qsa.Binding")
    caps = binding.state.caps
    rows = int(query.shape[0])
    if not 0 < rows <= int(caps.max_q_rows):
        raise ValueError("query rows must be within the planned decode capacity")
    if rows > int(binding.output.shape[0]):
        raise ValueError("query rows exceed the bound QSA output capacity")
    for tensor, name, shape, dtype in (
        (query, "query", (rows, caps.q_heads, caps.head_dim), torch.bfloat16),
        (request_ids, "request_ids", (rows,), (torch.int32, torch.int64)),
        (query_positions, "query_positions", (rows,), torch.int64),
    ):
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=shape,
            dtype=dtype,
            contiguous=True,
        )
    layout = binding.state._layout
    _qsa_attention_op(
        binding.plan.handle,
        query,
        request_ids,
        query_positions,
        binding.main_k_cache,
        binding.main_v_cache,
        binding.k_descale,
        binding.v_descale,
        binding.main_block_table,
        None if reuse else binding.selected_positions,
        binding.scratch,
        binding.output,
        binding.state.workspace_q_rows,
        binding.state.max_split_row_product,
        layout.partial_output_offset_bytes,
        layout.partial_lse_offset_bytes,
        layout.output_lse_offset_bytes,
        binding.state.config.sparse_gqa_direct_kv_warps,
        layout.draft_positions_offset_bytes if reuse else -1,
        caps.selection_width + caps.max_speculative_tokens,
        caps.max_batch,
    )
    return binding.output[:rows]
def _prime(binding: Binding, *, rows: int | None = None) -> None:
    """Compile a bound QSA transaction without mutating persistent state.

    Every synthetic row has an invalid request ID and position. The launch
    therefore warms the bound output capacity, cache-table stride, selector
    workspace, selection, and attention specializations. Cache accesses
    and persistent selector-state writes remain masked. Scratch, output, and
    selected-position buffers are transient and have unspecified contents
    after this call, except for a bound draft state's anchor buffers, which
    are preserved. The record kernel is warmed with persistent writes disabled.
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a qsa.Binding")

    caps = binding.state.caps
    output_capacity = int(binding.output.shape[0])
    requested_rows = output_capacity if rows is None else int(rows)
    if not 0 < requested_rows <= output_capacity:
        raise ValueError("priming rows must fit the bound QSA output capacity")
    device = caps.device
    binding = replace(binding, _record_draft_enabled=False)
    ready = torch.cuda.Event() if binding.selection_stream is not None else None
    sequence_lengths = torch.zeros(
        int(caps.max_batch), dtype=torch.int32, device=device
    )
    query_start_loc = torch.zeros(
        int(caps.max_batch) + 1, dtype=torch.int32, device=device
    )
    num_accepted_tokens = torch.ones(
        int(caps.max_batch), dtype=torch.int32, device=device
    )
    is_prefilling = torch.ones(int(caps.max_batch), dtype=torch.bool, device=device)
    warm_rows = {requested_rows}
    if output_capacity > _MAX_SPLIT_ROWS:
        warm_rows.add(_MAX_SPLIT_ROWS)
        warm_rows.add(_MAX_SPLIT_ROWS + 1)
    for warm_row_count in sorted(warm_rows):
        query = torch.empty(
            (warm_row_count, int(caps.q_heads), int(caps.head_dim)),
            dtype=caps.dtype,
            device=device,
        )
        index_query = torch.empty(
            (warm_row_count, int(caps.index_heads), int(caps.index_head_dim)),
            dtype=caps.dtype,
            device=device,
        )
        raw_index_key = torch.empty(
            (warm_row_count, int(caps.index_head_dim)),
            dtype=caps.dtype,
            device=device,
        )
        query_positions = torch.full(
            (warm_row_count,), -1, dtype=torch.int64, device=device
        )
        rope_positions = torch.full(
            (warm_row_count, int(caps.position_axes)),
            -1,
            dtype=torch.int64,
            device=device,
        )
        for request_id_dtype in (torch.int32, torch.int64):
            request_ids = torch.full(
                (warm_row_count,),
                -1,
                dtype=request_id_dtype,
                device=device,
            )
            if ready is not None:
                ready.record(torch.cuda.current_stream(device))
            inputs = dict(
                query=query,
                index_query=index_query,
                raw_index_key=raw_index_key,
                request_ids=request_ids,
                query_positions=query_positions,
                rope_positions=rope_positions,
                sequence_lengths=sequence_lengths,
                query_start_loc=query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                is_prefilling=is_prefilling,
            )
            if caps.dcp_size > 1:
                local = select(binding, **inputs)
                attend(
                    binding,
                    query=query,
                    request_ids=request_ids,
                    query_positions=query_positions,
                    selection=local,
                )
            else:
                run(binding, **inputs, index_ready=ready)
            if binding.draft_selection is not None:
                draft_rows = min(warm_row_count, int(caps.max_batch))
                for source_rows in (
                    sequence_lengths,
                    binding.draft_selection.logical_positions[: caps.max_batch],
                ):
                    reuse = DraftSelectionReuse(source_rows)
                    if caps.dcp_size > 1:
                        attend_reuse(
                            binding,
                            query=query[:draft_rows],
                            request_ids=request_ids[:draft_rows],
                            query_positions=query_positions[:draft_rows],
                            reuse=reuse,
                        )
                    else:
                        run(
                            binding,
                            query=query[:draft_rows],
                            request_ids=request_ids[:draft_rows],
                            query_positions=query_positions[:draft_rows],
                            reuse=reuse,
                        )


def is_supported(device: torch.device | str | None = None) -> bool:
    """Return whether the mandatory CuTe QSA dependencies are available."""
    from ..._lib.gating import has_cutlass_dsl, has_triton

    del device
    return has_cutlass_dsl() and has_triton()


__all__ = [
    "Caps",
    "Binding",
    "DraftSelectionState",
    "DraftSelectionPlan",
    "DraftSelectionReuse",
    "LocalSelection",
    "CacheRequirements",
    "QsaPrograms",
    "cache_requirements",
    "draft_selection_plan",
    "plan",
    "bind",
    "run",
    "select",
    "attend",
    "attend_reuse",
    "is_supported",
]
