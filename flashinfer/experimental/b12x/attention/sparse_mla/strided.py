"""Planned strided-record sparse-MLA API."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation.types import (
    FrozenMapping,
    MemoryRequirements,
    Plan as PreparationPlan,
    require_prepared,
)
from ..._lib.gating import default_is_supported
from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from ..._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    materialize_scratch_view,
)
from .._shared import static_fp8_quant
from ..dense_mla._kernel import (
    clear_dense_mla_kernel_caches,
    run_dense_mla,
)
from ..dense_mla._scratch import Binding as _NativeBinding
from ..dense_mla._scratch import Caps as _NativeCaps
from ..dense_mla._scratch import Plan as _NativePlan
from ..dense_mla._scratch import Scratch
from ..dense_mla._scratch import plan_dense_mla_scratch
from ..dense_mla.planner import Budget
from . import _paged_index_remap
from ._tuning import SparseMlaConfig, SparseMlaQuery, TUNING

FP8 = torch.float8_e4m3fn
BLOCK_SIZE = 64
TOTAL_HEADS = 128
QK_DIM = 576
VALUE_DIM = 512
PHYSICAL_RECORD_WIDTH = 1088
TOPK = 2048
SM_SCALE = 1.0 / math.sqrt(192)


@dataclass(frozen=True, kw_only=True)
class Caps:
    device: torch.device | str
    num_q_heads: int
    tp_size: int
    max_q_rows: int
    num_cache_blocks: int
    max_physical_records: int | None = None
    block_size: int = BLOCK_SIZE
    topk: int = TOPK
    kv_dtype: torch.dtype = FP8
    use_cuda_graph: bool = False
    budget: Budget | None = None

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        if int(self.tp_size) != 8:
            raise ValueError("strided sparse MLA is qualified only for TP8")
        if int(self.num_q_heads) != TOTAL_HEADS // int(self.tp_size):
            raise ValueError("num_q_heads must equal total_heads / tp_size")
        if int(self.block_size) != BLOCK_SIZE:
            raise ValueError(f"strided sparse MLA block_size must be {BLOCK_SIZE}")
        if int(self.topk) != TOPK:
            raise ValueError(f"strided sparse MLA topk must be {TOPK}")
        if self.kv_dtype != FP8:
            raise TypeError("strided sparse MLA requires E4M3 KV cache")
        if int(self.max_q_rows) <= 0:
            raise ValueError("max_q_rows must be positive")
        if int(self.num_cache_blocks) <= 0:
            raise ValueError("num_cache_blocks must be positive")
        if int(self.num_cache_blocks) > torch.iinfo(torch.int32).max // BLOCK_SIZE:
            raise ValueError("num_cache_blocks exceeds the physical-slot int32 range")
        max_physical_records = (
            int(self.num_cache_blocks) * BLOCK_SIZE
            if self.max_physical_records is None
            else int(self.max_physical_records)
        )
        if not int(self.num_cache_blocks) * BLOCK_SIZE <= max_physical_records:
            raise ValueError(
                "max_physical_records must cover all contiguous physical slots"
            )
        if max_physical_records > torch.iinfo(torch.int32).max:
            raise ValueError("max_physical_records exceeds the index int32 range")
        object.__setattr__(self, "max_physical_records", max_physical_records)

@dataclass(frozen=True)
class Binding:
    plan: PreparationPlan | None
    native: _NativeBinding
    query_quant: static_fp8_quant.Binding
    index_remap: _paged_index_remap.Binding | None
    kv_cache: torch.Tensor
    selected_indices: torch.Tensor
    selected_counts: torch.Tensor


@dataclass(frozen=True)
class _StridedLayout:
    caps: Caps
    native: _NativePlan
    query_offset_bytes: int
    physical_indices_offset_bytes: int
    selected_counts_offset_bytes: int
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self):
        return self._scratch_specs

    def shapes_and_dtypes(self):
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    @property
    def num_splits(self) -> int:
        return self.native.num_splits


def _materialize_layout(caps: Caps) -> _StridedLayout:
    native_caps = _NativeCaps(
        device=caps.device,
        mode="decode",
        kv_dtype=FP8,
        num_q_heads=int(caps.num_q_heads),
        page_size=1,
        max_total_q=int(caps.max_q_rows),
        max_batch=int(caps.max_q_rows),
        max_cache_tokens=TOPK,
        max_page_table_width=TOPK,
        num_cache_pages=int(caps.max_physical_records),
        head_dim=QK_DIM,
        v_head_dim=VALUE_DIM,
        physical_record_width=PHYSICAL_RECORD_WIDTH,
        use_cuda_graph=bool(caps.use_cuda_graph),
        budget=caps.budget,
    )
    native = plan_dense_mla_scratch(native_caps)
    native_spec = native.scratch_specs()[0]
    query_offset_bytes = align_up(native_spec.nbytes, SCRATCH_ALIGN_BYTES)
    query_nbytes = int(caps.max_q_rows) * int(caps.num_q_heads) * QK_DIM
    physical_indices_offset_bytes = align_up(
        query_offset_bytes + query_nbytes,
        SCRATCH_ALIGN_BYTES,
    )
    physical_indices_nbytes = int(caps.max_q_rows) * TOPK * torch.int32.itemsize
    selected_counts_offset_bytes = align_up(
        physical_indices_offset_bytes + physical_indices_nbytes,
        SCRATCH_ALIGN_BYTES,
    )
    selected_counts_nbytes = int(caps.max_q_rows) * torch.int32.itemsize
    total_nbytes = align_up(
        selected_counts_offset_bytes + selected_counts_nbytes,
        SCRATCH_ALIGN_BYTES,
    )
    return _StridedLayout(
        caps=caps,
        native=native,
        query_offset_bytes=query_offset_bytes,
        physical_indices_offset_bytes=physical_indices_offset_bytes,
        selected_counts_offset_bytes=selected_counts_offset_bytes,
        _scratch_specs=(
            scratch_buffer_spec(
                "sparse_mla.strided.scratch",
                nbytes=total_nbytes,
                device=native_spec.device,
            ),
        ),
    )

@program_cache(scope="preparation")
def compile_strided_sparse_mla(caps: Caps, ordinal: int):
    """Compile the real quantize/remap/decode/merge program set from metadata."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from b12x._lib.compile_plan import compile_only_launches
    from ..dense_mla import _kernel as dense_kernel

    if not isinstance(caps, Caps):
        raise TypeError("strided sparse MLA compiler requires Caps")
    device = torch.device("cuda", ordinal)
    caps = Caps(
        device=device,
        num_q_heads=caps.num_q_heads,
        tp_size=caps.tp_size,
        max_q_rows=caps.max_q_rows,
        num_cache_blocks=caps.num_cache_blocks,
        max_physical_records=caps.max_physical_records,
        block_size=caps.block_size,
        topk=caps.topk,
        kv_dtype=caps.kv_dtype,
        use_cuda_graph=caps.use_cuda_graph,
        budget=caps.budget,
    )
    layout = _materialize_layout(caps)
    with FakeTensorMode(), compile_only_launches():
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        (spec,) = layout.scratch_specs()
        storage = empty(spec.shape, spec.dtype)
        rows = caps.max_q_rows
        q = empty((rows, caps.num_q_heads, QK_DIM), torch.bfloat16)
        cache = empty(
            (caps.num_cache_blocks, BLOCK_SIZE, PHYSICAL_RECORD_WIDTH), FP8
        )
        output = empty((rows, caps.num_q_heads, VALUE_DIM), torch.bfloat16)
        indices = empty((rows, TOPK), torch.int32)
        counts = empty((rows,), torch.int32)
        cu = empty((rows + 1,), torch.int32)
        scale = empty((1,), torch.float32)
        flat_cache, _, _ = _physical_record_view(layout, cache)
        binding = _bind_native(
            layout,
            scratch_storage=storage,
            q=q,
            flat_cache=flat_cache,
            original_cache=cache,
            output=output,
            record_indices=indices,
            selected_counts=counts,
            cu_seqlens_q=cu,
            kv_scale=scale,
            q_scale=scale,
        )
        forward, merge = dense_kernel._compile_entries(binding.native)
        static_fp8_quant.compile(binding=binding.query_quant)
        physical_remap = _paged_index_remap.bind_physical_slots(
            physical_slots=indices,
            input_counts=counts,
            physical_indices=indices,
            selected_counts=counts,
            max_q_rows=rows,
            num_cache_blocks=caps.num_cache_blocks,
            block_stride_records=BLOCK_SIZE,
            token_stride_records=1,
        )
        indexed_remap = _paged_index_remap.bind(
            request_ids=counts,
            block_table=empty((rows, caps.num_cache_blocks), torch.int32),
            logical_indices=indices,
            physical_indices=indices,
            selected_counts=counts,
            max_q_rows=rows,
            num_cache_blocks=caps.num_cache_blocks,
            block_stride_records=BLOCK_SIZE,
            token_stride_records=1,
        )
        programs = {
            "quantizer": static_fp8_quant.resolve_static_fp8_quant_launcher(
                binding=binding.query_quant
            ).compiled,
            "decode": forward,
            "remap_physical": _paged_index_remap._compile(binding=physical_remap),
            "remap_indexed": _paged_index_remap._compile(binding=indexed_remap),
        }
        if merge is not None:
            programs["merge"] = merge
    return programs




@dataclass(frozen=True)
class _StridedState:
    layout: _StridedLayout
    _quant_launcher: object | None = None
    _remap_launcher: object | None = None
    _dense_launchers: object | None = None

    def bind(self, **kwargs) -> Binding:
        return _bind_physical(self, plan=None, **kwargs)

    def bind_indexed(self, **kwargs) -> Binding:
        return _bind_indexed(self, plan=None, **kwargs)

    def prime(self, binding: Binding):
        from b12x.attention._shared.static_fp8_quant import resolve_static_fp8_quant_launcher
        from b12x.attention.dense_mla._kernel import resolve_dense_mla_launchers
        object.__setattr__(
            self, "_quant_launcher",
            resolve_static_fp8_quant_launcher(binding=binding.query_quant),
        )
        object.__setattr__(
            self, "_dense_launchers",
            resolve_dense_mla_launchers(binding=binding.native),
        )
        if binding.index_remap is not None:
            object.__setattr__(
                self, "_remap_launcher",
                _paged_index_remap._resolve_launcher(binding=binding.index_remap),
            )

    def scratch_specs(self):
        return self.layout.scratch_specs()

    def run(self, binding: Binding):
        if self._quant_launcher is None or self._dense_launchers is None:
            # A plan materialized on first use resolves its launchers from its
            # first binding; under a kernel resolution guard this raises.
            self.prime(binding)
        if binding.index_remap is not None:
            if self._remap_launcher is None:
                raise RuntimeError("strided sparse MLA remap was not prepared")
            self._remap_launcher.run(binding.index_remap)
        self._quant_launcher.run(binding.query_quant)
        return self._dense_launchers.run(binding.native)
def plan(
    caps: Caps,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: SparseMlaConfig | None = None,
) -> PreparationPlan:
    if not isinstance(caps, Caps):
        raise TypeError("plan requires strided sparse MLA Caps")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("strided sparse MLA invocation is fixed by its Caps")
    query = SparseMlaQuery(
        mode="decode", dtype="bfloat16", kv_dtype=str(FP8).removeprefix("torch."),
        num_q_heads=int(caps.num_q_heads), qk_head_dim=QK_DIM, v_head_dim=VALUE_DIM,
        max_q_rows=int(caps.max_q_rows), max_width=TOPK, page_size=BLOCK_SIZE,
        model_type=None, head_major_output=False, scale_format=0,
        cache_record_bytes=PHYSICAL_RECORD_WIDTH, fp8_rope=False,
        latent_scale_per_token=False, has_attention_sink=False,
        cache_layout="strided_physical", operation="strided_attention", slot_dtype="int32",
        prefill_mg_enabled=False,
        max_batch=int(caps.max_q_rows),
        max_page_table_width=TOPK,
        physical_block_size=BLOCK_SIZE,
        physical_record_width=PHYSICAL_RECORD_WIDTH,
        num_cache_blocks=int(caps.num_cache_blocks),
        max_physical_records=int(caps.max_physical_records),
        tp_size=int(caps.tp_size),
        use_cuda_graph=bool(caps.use_cuda_graph),
        budget_max_splits=(
            None if caps.budget is None else caps.budget.max_splits
        ),
        budget_max_partial_rows=(
            None if caps.budget is None else caps.budget.max_partial_rows
        ),
    )
    layout = _materialize_layout(caps)
    return PreparationPlan(
        contract=TUNING, query=query, invocation=invocation, override=override,
        _compile_jobs=lambda config, device: (CompileJob.create(
            "b12x.attention.sparse_mla.strided:compile_strided_sparse_mla",
            caps,
            device.ordinal,
        ),),
        _memory_requirements=lambda config, device: MemoryRequirements(
            scratch=layout.scratch_specs()
        ),
        _materialize=lambda selection, device: _StridedState(layout),
        _device=caps.device,
    )


def _physical_record_view(
    layout: _StridedLayout,
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    expected_record_shape = (BLOCK_SIZE, PHYSICAL_RECORD_WIDTH)
    if (
        kv_cache.dtype != FP8
        or kv_cache.ndim != 3
        or tuple(kv_cache.shape[1:]) != expected_record_shape
        or not 0 < int(kv_cache.shape[0]) <= int(layout.caps.num_cache_blocks)
    ):
        raise ValueError(
            "kv_cache must be E4M3 with shape "
            f"[1..{layout.caps.num_cache_blocks}, {BLOCK_SIZE}, "
            f"{PHYSICAL_RECORD_WIDTH}], got "
            f"dtype={kv_cache.dtype}, shape={tuple(kv_cache.shape)}"
        )
    block_stride, token_stride, element_stride = map(int, kv_cache.stride())
    if (
        element_stride != 1
        or token_stride < PHYSICAL_RECORD_WIDTH
        or block_stride < BLOCK_SIZE * token_stride
        or token_stride % PHYSICAL_RECORD_WIDTH
        or block_stride % PHYSICAL_RECORD_WIDTH
    ):
        raise ValueError(
            "kv_cache must use non-overlapping, whole 1088-element physical "
            f"record strides, got stride={tuple(kv_cache.stride())}"
        )
    block_stride_records = block_stride // PHYSICAL_RECORD_WIDTH
    token_stride_records = token_stride // PHYSICAL_RECORD_WIDTH
    physical_records = (
        (int(kv_cache.shape[0]) - 1) * block_stride_records
        + (BLOCK_SIZE - 1) * token_stride_records
        + 1
    )
    if physical_records > int(layout.caps.max_physical_records):
        raise ValueError(
            "kv_cache physical record span exceeds planned capacity: "
            f"need {physical_records}, planned {layout.caps.max_physical_records}"
        )
    flat_cache = torch.as_strided(
        kv_cache,
        size=(physical_records, 1, PHYSICAL_RECORD_WIDTH),
        stride=(PHYSICAL_RECORD_WIDTH, PHYSICAL_RECORD_WIDTH, 1),
    )
    return flat_cache, block_stride_records, token_stride_records


def _bind_native(
    layout: _StridedLayout,
    *,
    scratch_storage: torch.Tensor,
    q: torch.Tensor,
    flat_cache: torch.Tensor,
    original_cache: torch.Tensor,
    output: torch.Tensor,
    record_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_scale: torch.Tensor,
    q_scale: torch.Tensor,
    active_splits: int | None = None,
) -> Binding:
    if q.dtype != torch.bfloat16:
        raise TypeError("strided sparse MLA absorbed query must be BF16")
    rows = int(q.shape[0])
    if record_indices.dtype != torch.int32 or tuple(record_indices.shape) != (
        rows,
        TOPK,
    ):
        raise ValueError(f"record_indices must be int32 with shape ({rows}, {TOPK})")
    if selected_counts.dtype != torch.int32 or tuple(selected_counts.shape) != (rows,):
        raise ValueError(f"selected_counts must be int32 with shape ({rows},)")
    if tuple(cu_seqlens_q.shape) != (rows + 1,):
        raise ValueError(f"cu_seqlens_q must have shape ({rows + 1},)")
    q_fp8, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.query_offset_bytes,
        shape=tuple(q.shape),
        dtype=FP8,
    )
    query_quant = static_fp8_quant.bind(
        source=q,
        output=q_fp8,
        scale=q_scale,
        max_numel=int(layout.caps.max_q_rows) * int(layout.caps.num_q_heads) * QK_DIM,
    )
    native = layout.native.bind(
        scratch=scratch_storage,
        q=q_fp8,
        kv_cache=flat_cache,
        output=output,
        page_table=record_indices,
        cache_seqlens=selected_counts,
        cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
        sm_scale=SM_SCALE,
        active_splits=active_splits,
    )
    return Binding(
        plan=None,
        native=native,
        query_quant=query_quant,
        index_remap=None,
        kv_cache=original_cache,
        selected_indices=record_indices,
        selected_counts=selected_counts,
    )


def _index_scratch(
    layout: _StridedLayout,
    scratch_storage: torch.Tensor,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical_indices, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.physical_indices_offset_bytes,
        shape=(rows, TOPK),
        dtype=torch.int32,
    )
    selected_counts, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.selected_counts_offset_bytes,
        shape=(rows,),
        dtype=torch.int32,
    )
    return physical_indices, selected_counts


def _bind_physical(
    state: _StridedState,
    *,
    plan: PreparationPlan | None,
    scratch,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_scale: torch.Tensor,
    q_scale: torch.Tensor,
    active_splits: int | None = None,
) -> Binding:
    layout = state.layout
    scratch_storage = scratch_tensor(scratch, layout.scratch_specs(), owner="strided sparse MLA")
    rows = int(q.shape[0])
    record_indices, remapped_counts = _index_scratch(layout, scratch_storage, rows)
    flat_cache, block_stride_records, token_stride_records = _physical_record_view(layout, kv_cache)
    binding = _bind_native(
        layout, scratch_storage=scratch_storage, q=q, flat_cache=flat_cache,
        original_cache=kv_cache, output=output, record_indices=record_indices,
        selected_counts=remapped_counts, cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale, q_scale=q_scale, active_splits=active_splits,
    )
    remap = _paged_index_remap.bind_physical_slots(
        physical_slots=selected_indices, input_counts=selected_counts,
        physical_indices=record_indices, selected_counts=remapped_counts,
        max_q_rows=int(layout.caps.max_q_rows), num_cache_blocks=int(kv_cache.shape[0]),
        block_stride_records=block_stride_records, token_stride_records=token_stride_records,
    )
    return Binding(
        plan=plan, native=binding.native, query_quant=binding.query_quant,
        index_remap=remap, kv_cache=binding.kv_cache,
        selected_indices=binding.selected_indices, selected_counts=binding.selected_counts,
    )


def _bind_indexed(
    state: _StridedState,
    *,
    plan: PreparationPlan | None,
    scratch,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    logical_indices: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_scale: torch.Tensor,
    q_scale: torch.Tensor,
    active_splits: int | None = None,
) -> Binding:
    layout = state.layout
    scratch_storage = scratch_tensor(scratch, layout.scratch_specs(), owner="strided sparse MLA")
    rows = int(q.shape[0])
    physical_indices, selected_counts = _index_scratch(layout, scratch_storage, rows)
    flat_cache, block_stride_records, token_stride_records = _physical_record_view(layout, kv_cache)
    binding = _bind_native(
        layout, scratch_storage=scratch_storage, q=q, flat_cache=flat_cache,
        original_cache=kv_cache, output=output, record_indices=physical_indices,
        selected_counts=selected_counts, cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale, q_scale=q_scale, active_splits=active_splits,
    )
    remap = _paged_index_remap.bind(
        request_ids=request_ids, block_table=block_table, logical_indices=logical_indices,
        physical_indices=physical_indices, selected_counts=selected_counts,
        max_q_rows=int(layout.caps.max_q_rows), num_cache_blocks=int(kv_cache.shape[0]),
        block_stride_records=block_stride_records, token_stride_records=token_stride_records,
    )
    return Binding(
        plan=plan, native=binding.native, query_quant=binding.query_quant,
        index_remap=remap, kv_cache=binding.kv_cache,
        selected_indices=binding.selected_indices, selected_counts=binding.selected_counts,
    )

def _state_for(plan: PreparationPlan, device: torch.device) -> _StridedState:
    state = require_prepared(plan, TUNING.component_id, device)
    if not isinstance(state, _StridedState):
        raise ValueError("plan is not a strided sparse-MLA plan")
    return state


def bind(
    plan: PreparationPlan,
    **kwargs,
) -> Binding:
    return _bind_physical(
        _state_for(plan, kwargs["kv_cache"].device),
        plan=plan,
        **kwargs,
    )


def bind_indexed(
    plan: PreparationPlan,
    **kwargs,
) -> Binding:
    return _bind_indexed(
        _state_for(plan, kwargs["kv_cache"].device),
        plan=plan,
        **kwargs,
    )




def run_decode(*, binding: Binding) -> tuple[torch.Tensor, torch.Tensor]:
    state = require_prepared(binding.plan, TUNING.component_id, binding.kv_cache.device)
    if not isinstance(state, _StridedState):
        raise ValueError("plan is not a strided sparse-MLA plan")
    return state.run(binding)


def run_extend(*, binding: Binding) -> tuple[torch.Tensor, torch.Tensor]:
    return run_decode(binding=binding)

def reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    *,
    kv_scale: torch.Tensor | float,
    q_scale: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    def scalar(value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    count_host = [int(value) for value in selected_counts.detach().cpu().tolist()]
    q_mul = scalar(q_scale)
    q_f32 = (q.float() / q_mul).to(FP8).float() * q_mul
    kv_mul = scalar(kv_scale)
    output = torch.empty(
        q.shape[0], q.shape[1], VALUE_DIM, dtype=torch.float32, device=q.device
    )
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    for row, count in enumerate(count_host):
        ids = selected_indices[row, :count].to(torch.long)
        blocks = torch.div(ids, BLOCK_SIZE, rounding_mode="floor")
        tokens = ids.remainder(BLOCK_SIZE)
        records = kv_cache[blocks, tokens, :QK_DIM].float() * kv_mul
        logits = torch.einsum("hd,kd->hk", q_f32[row], records) * SM_SCALE
        probability = torch.softmax(logits, dim=-1)
        output[row] = torch.einsum("hk,kd->hd", probability, records[:, :VALUE_DIM])
        lse[row] = torch.logsumexp(logits, dim=-1)
    return output.to(torch.bfloat16), lse


def is_supported(device=None) -> bool:
    if not default_is_supported(device, requires=()):
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    return tuple(torch.cuda.get_device_capability(device)) in ((12, 0), (12, 1))


def clear_caches() -> None:
    _paged_index_remap.clear_caches()
    static_fp8_quant.clear_caches()
    clear_dense_mla_kernel_caches()


__all__ = [
    "Binding",
    "Budget",
    "Caps",
    "Scratch",
    "bind",
    "bind_indexed",
    "clear_caches",
    "plan",
    "reference",
    "run_decode",
    "run_extend",
]
