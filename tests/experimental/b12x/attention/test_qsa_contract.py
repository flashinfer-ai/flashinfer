from __future__ import annotations

from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import replace

import pytest
import torch

import b12x.attention.qsa._contract as qsa_contract
from b12x.attention import qsa
from b12x.preparation import PreparationSession, PreparedCall
from b12x.attention.qsa._kernels import (
    launch_compress_completed_groups,
    launch_expand_selected_groups,
    launch_prepare_index_query,
    launch_score_representatives,
)
from b12x.attention.qsa.reference import (
    gemma_rmsnorm_reference,
    score_select_reference,
    sparse_paged_gqa_reference,
)

from ..conftest import require_b12x as require_sm120


_test_resources: ContextVar[ExitStack | None] = ContextVar(
    "qsa_test_resources", default=None,
)


@pytest.fixture(autouse=True)
def _retain_qsa_test_resources():
    with ExitStack() as resources:
        token = _test_resources.set(resources)
        try:
            yield
        finally:
            _test_resources.reset(token)


def _resources() -> ExitStack:
    resources = _test_resources.get()
    if resources is None:
        raise RuntimeError("QSA test resources require the test scope")
    return resources


def _cuda_graph() -> torch.cuda.CUDAGraph:
    graph = getattr(torch.cuda, "CUDAGraph")()
    _resources().callback(graph.reset)
    return graph

def _caps(
    device: torch.device | str | None = None,
    **changes: object,
) -> qsa.Caps:
    if device is None:
        device = require_sm120()
    values: dict[str, object] = {
        "device": device,
        "max_batch": 2,
        "max_raw_state_slots": 3,
        "max_q_rows": 2,
        "max_seq_len": 64,
        "num_main_cache_pages": 4,
        "num_compressed_cache_pages": 4,
        "main_page_size": 32,
        "compressed_page_size": 8,
        "q_heads": 6,
        "kv_heads": 1,
        "head_dim": 256,
        "index_heads": 1,
        "index_kv_heads": 1,
        "index_head_dim": 128,
        "index_rotary_dim": 64,
        "compress_ratio": 4,
        "budget": 2048,
    }
    values.update(changes)
    return qsa.Caps(**values)
def _invocation(
    caps: qsa.Caps, *, shared_pool: bool = False, **overrides: dict[str, object],
) -> object:
    operands = dict(qsa_contract._canonical_abi(caps))
    if shared_pool:
        page_elements = caps.compressed_page_size * caps.index_head_dim
        page_i64 = caps.compressed_page_nbytes // torch.int64.itemsize
        operands.update({
            "raw_k_ring": {
                "dtype": "bfloat16", "strides": (page_elements, caps.index_head_dim, 1),
            },
            "raw_logical_positions": {"dtype": "int64", "strides": (page_i64, 1)},
            "raw_rope_positions": {
                "dtype": "int64", "strides": (page_i64, caps.position_axes, 1),
            },
            "raw_interval_start_positions": {"dtype": "int64", "strides": (page_i64,)},
        })
    operands.update(overrides)
    return qsa.invocation_from_descriptors(caps, operands=operands)



def _allocate_binding(
    caps: qsa.Caps,
    *,
    plan: qsa.Plan | None = None,
) -> qsa.Binding:
    declaration = qsa.plan(caps, invocation=_invocation(caps)) if plan is None else plan
    device = caps.device
    main_k = torch.empty(
        (
            caps.num_main_cache_pages,
            caps.main_page_size,
            caps.kv_heads,
            caps.head_dim,
        ),
        dtype=caps.kv_dtype,
        device=device,
    )
    main_v = torch.empty_like(main_k)
    main_table = torch.full(
        (caps.max_batch, caps.main_table_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    compressed = torch.empty(
        (
            caps.num_compressed_cache_pages,
            caps.compressed_page_size,
            caps.index_head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    compressed_table = torch.full(
        (caps.max_batch, caps.compressed_table_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    raw_ring = torch.empty(
        (
            caps.max_raw_state_slots,
            caps.raw_ring_capacity,
            caps.index_head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    raw_tags = torch.full(
        (caps.max_raw_state_slots, caps.raw_ring_capacity),
        -1,
        dtype=torch.int64,
        device=device,
    )
    raw_rope = torch.full(
        (
            caps.max_raw_state_slots,
            caps.raw_ring_capacity,
            caps.position_axes,
        ),
        -1,
        dtype=torch.int64,
        device=device,
    )
    raw_interval_start = torch.full(
        (caps.max_raw_state_slots,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    slot_ids = torch.arange(
        caps.max_batch, dtype=torch.int64, device=device
    ).contiguous()
    q_weight = torch.zeros((caps.index_head_dim,), dtype=torch.float32, device=device)
    k_weight = torch.zeros_like(q_weight)
    rope_cos = torch.ones(
        (caps.max_seq_len, caps.index_rotary_dim // 2),
        dtype=torch.float32,
        device=device,
    )
    rope_sin = torch.zeros_like(rope_cos)
    output = torch.empty(
        (caps.max_q_rows, caps.q_heads, caps.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    selected = torch.empty(
        (caps.max_q_rows, caps.selection_width),
        dtype=torch.int32,
        device=device,
    )
    binding_kwargs = dict(
        main_k_cache=main_k, main_v_cache=main_v,
        k_descale=torch.ones(1, device=device)
        if caps.kv_dtype == torch.float8_e4m3fn else None,
        v_descale=torch.ones(1, device=device)
        if caps.kv_dtype == torch.float8_e4m3fn else None,
        main_block_table=main_table, compressed_k_cache=compressed,
        compressed_block_table=compressed_table, raw_k_ring=raw_ring,
        raw_logical_positions=raw_tags, raw_rope_positions=raw_rope,
        raw_interval_start_positions=raw_interval_start, raw_state_slot_ids=slot_ids,
        index_q_norm_weight=q_weight, index_k_norm_weight=k_weight,
        rope_cos=rope_cos, rope_sin=rope_sin, output=output,
        selected_positions=selected,
    )
    prepared: dict[str, object] = {}

    def prepare_call(state: object) -> PreparedCall:
        spec = state.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind_for_preparation(scratch=scratch, **binding_kwargs)
        restore_tensors = (
            binding.compressed_k_cache, binding.raw_k_ring,
            binding.raw_logical_positions, binding.raw_rope_positions,
            binding.raw_interval_start_positions, binding.output,
            binding.selected_positions,
        )
        snapshots = tuple(tensor.clone() for tensor in restore_tensors)
        dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))

        def restore() -> None:
            for tensor, snapshot in zip(restore_tensors, snapshots, strict=True):
                tensor.copy_(snapshot)

        prepared["spec"] = spec
        return PreparedCall(
            run=lambda: state.run_for_preparation(binding, **dynamic),
            reset=restore,
            restore=restore,
            owners=(scratch, binding, snapshots, dynamic),
        )

    resources = _resources()
    session = resources.enter_context(PreparationSession(device=device, autotune=False))
    result = session.prepare((declaration.request(
        name="qsa-test", prepare_call=prepare_call,
    ),))
    resources.callback(result.close)
    spec = prepared["spec"]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    return qsa.bind(declaration, scratch=scratch, **binding_kwargs)


def _rebind(binding: qsa.Binding, **changes: torch.Tensor) -> qsa.Binding:
    tensors = {
        "scratch": binding.scratch,
        "main_k_cache": binding.main_k_cache,
        "main_v_cache": binding.main_v_cache,
        "k_descale": binding.k_descale,
        "v_descale": binding.v_descale,
        "main_block_table": binding.main_block_table,
        "compressed_k_cache": binding.compressed_k_cache,
        "compressed_block_table": binding.compressed_block_table,
        "raw_k_ring": binding.raw_k_ring,
        "raw_logical_positions": binding.raw_logical_positions,
        "raw_rope_positions": binding.raw_rope_positions,
        "raw_interval_start_positions": binding.raw_interval_start_positions,
        "raw_state_slot_ids": binding.raw_state_slot_ids,
        "index_q_norm_weight": binding.index_q_norm_weight,
        "index_k_norm_weight": binding.index_k_norm_weight,
        "rope_cos": binding.rope_cos,
        "rope_sin": binding.rope_sin,
        "output": binding.output,
        "selected_positions": binding.selected_positions,
    }
    tensors.update(changes)
    return qsa.bind(binding.plan, **tensors)


def test_qsa_bind_accepts_a_runtime_pool_smaller_than_planned_capacity() -> None:
    binding = _allocate_binding(_caps())

    rebound = _rebind(
        binding,
        main_k_cache=binding.main_k_cache[:1],
        main_v_cache=binding.main_v_cache[:1],
        compressed_k_cache=binding.compressed_k_cache[:1],
    )

    assert rebound.state.caps.main_table_width == 2
    assert rebound.state.caps.compressed_table_width == 2
    assert rebound.main_k_cache.shape[0] == 1
    assert rebound.compressed_k_cache.shape[0] == 1


def _allocate_shared_compressed_raw_binding(
    caps: qsa.Caps,
    *,
    initialize_metadata: bool = True,
) -> qsa.Binding:
    declaration = qsa.plan(caps, invocation=_invocation(caps, shared_pool=True))
    layout = qsa_contract._scratch_layout(caps)[0]
    scratch = torch.empty((layout.total_nbytes,), dtype=torch.uint8, device=caps.device)
    device = caps.device
    main_k = torch.empty(
        (
            caps.num_main_cache_pages,
            caps.main_page_size,
            caps.kv_heads,
            caps.head_dim,
        ),
        dtype=caps.kv_dtype,
        device=device,
    )
    main_v = torch.empty_like(main_k)
    main_table = torch.full(
        (caps.max_batch, caps.main_table_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    page_elements = caps.compressed_page_size * caps.index_head_dim
    pool = torch.empty(
        (caps.num_compressed_cache_pages, page_elements),
        dtype=torch.bfloat16,
        device=device,
    )
    compressed = pool.view(
        caps.num_compressed_cache_pages,
        caps.compressed_page_size,
        caps.index_head_dim,
    )
    raw_ring = pool.as_strided(
        (
            caps.max_raw_state_slots,
            caps.raw_ring_capacity,
            caps.index_head_dim,
        ),
        (page_elements, caps.index_head_dim, 1),
    )
    pool_i64 = pool.view(torch.uint8).reshape(-1).view(torch.int64)
    page_i64 = caps.compressed_page_nbytes // torch.int64.itemsize
    payload_i64 = (
        caps.raw_ring_capacity
        * caps.index_head_dim
        * torch.bfloat16.itemsize
        // torch.int64.itemsize
    )
    raw_tags = pool_i64.as_strided(
        (caps.max_raw_state_slots, caps.raw_ring_capacity),
        (page_i64, 1),
        storage_offset=payload_i64,
    )
    raw_rope = pool_i64.as_strided(
        (
            caps.max_raw_state_slots,
            caps.raw_ring_capacity,
            caps.position_axes,
        ),
        (page_i64, caps.position_axes, 1),
        storage_offset=payload_i64 + caps.raw_ring_capacity,
    )
    raw_interval_start = pool_i64.as_strided(
        (caps.max_raw_state_slots,),
        (page_i64,),
        storage_offset=(
            payload_i64 + caps.raw_ring_capacity * (1 + caps.position_axes)
        ),
    )
    if initialize_metadata:
        raw_tags.fill_(-1)
        raw_rope.fill_(-1)
        raw_interval_start.fill_(-1)
    compressed_table = torch.full(
        (caps.max_batch, caps.compressed_table_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    slot_ids = torch.arange(
        caps.max_batch, dtype=torch.int64, device=device
    ).contiguous()
    q_weight = torch.zeros((caps.index_head_dim,), dtype=torch.float32, device=device)
    k_weight = torch.zeros_like(q_weight)
    rope_cos = torch.ones(
        (caps.max_seq_len, caps.index_rotary_dim // 2),
        dtype=torch.float32,
        device=device,
    )
    rope_sin = torch.zeros_like(rope_cos)
    output = torch.empty(
        (caps.max_q_rows, caps.q_heads, caps.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    selected = torch.empty(
        (caps.max_q_rows, caps.selection_width),
        dtype=torch.int32,
        device=device,
    )
    binding_kwargs = dict(
        main_k_cache=main_k, main_v_cache=main_v,
        k_descale=torch.ones(1, device=device)
        if caps.kv_dtype == torch.float8_e4m3fn else None,
        v_descale=torch.ones(1, device=device)
        if caps.kv_dtype == torch.float8_e4m3fn else None,
        main_block_table=main_table, compressed_k_cache=compressed,
        compressed_block_table=compressed_table, raw_k_ring=raw_ring,
        raw_logical_positions=raw_tags, raw_rope_positions=raw_rope,
        raw_interval_start_positions=raw_interval_start, raw_state_slot_ids=slot_ids,
        index_q_norm_weight=q_weight, index_k_norm_weight=k_weight,
        rope_cos=rope_cos, rope_sin=rope_sin, output=output,
        selected_positions=selected,
    )

    def prepare_call(state: object) -> PreparedCall:
        prime_scratch = torch.empty_like(scratch)
        binding = state.bind_for_preparation(scratch=prime_scratch, **binding_kwargs)
        restored = (
            compressed, raw_ring, raw_tags, raw_rope, raw_interval_start, output, selected,
        )
        snapshots = tuple(tensor.clone() for tensor in restored)
        dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))

        def restore() -> None:
            for tensor, snapshot in zip(restored, snapshots, strict=True):
                tensor.copy_(snapshot)

        return PreparedCall(
            run=lambda: state.run_for_preparation(binding, **dynamic),
            reset=restore,
            restore=restore,
            owners=(prime_scratch, binding, snapshots, dynamic),
        )

    resources = _resources()
    session = resources.enter_context(PreparationSession(device=device, autotune=False))
    result = session.prepare((
        declaration.request(
            name="qsa-shared-test",
            prepare_call=prepare_call,
        ),
    ))
    resources.callback(result.close)
    return qsa.bind(
        declaration,
        scratch=scratch,
        **binding_kwargs,
    )


def _dynamic_inputs(
    binding: qsa.Binding,
    *,
    positions: tuple[int, ...],
    request_ids: tuple[int, ...] | None = None,
    accepted_tokens: tuple[int, ...] | None = None,
    is_prefilling: tuple[bool, ...] | None = None,
) -> dict[str, torch.Tensor]:
    caps = binding.state.caps
    rows = len(positions)
    device = caps.device
    if request_ids is None:
        request_ids = tuple(range(rows))
    active_requests = [request for request in request_ids if request >= 0]
    active_count = max(active_requests, default=-1) + 1
    if accepted_tokens is None:
        accepted_tokens = (1,) * active_count
    if len(accepted_tokens) != active_count:
        raise ValueError("accepted_tokens must cover the dense active request prefix")
    if is_prefilling is None:
        is_prefilling = (False,) * active_count
    if len(is_prefilling) != active_count:
        raise ValueError("is_prefilling must cover the dense active request prefix")
    generator = torch.Generator(device="cpu").manual_seed(98123 + sum(positions))
    query = torch.randn(
        (rows, caps.q_heads, caps.head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=torch.bfloat16)
    index_query = torch.randn(
        (rows, caps.index_heads, caps.index_head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=torch.bfloat16)
    raw_key = torch.randn(
        (rows, caps.index_head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=torch.bfloat16)
    request_tensor = torch.tensor(request_ids, dtype=torch.int64, device=device)
    position_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
    rope_positions = (
        position_tensor[:, None].expand(rows, caps.position_axes).contiguous()
    )
    sequence_lengths = torch.zeros((caps.max_batch,), dtype=torch.int32, device=device)
    query_start_loc_values = [0]
    cursor = 0
    for request in range(caps.max_batch):
        while cursor < rows and request_ids[cursor] == request:
            cursor += 1
        query_start_loc_values.append(cursor)
    if any(request_id >= 0 for request_id in request_ids[cursor:]):
        raise ValueError("request rows must form dense contiguous request intervals")
    for request_id, position in zip(request_ids, positions, strict=True):
        if request_id >= 0:
            sequence_lengths[request_id] = position + 1
    accepted = torch.zeros((caps.max_batch,), dtype=torch.int32, device=device)
    if active_count:
        accepted[:active_count] = torch.tensor(
            accepted_tokens, dtype=torch.int32, device=device
        )
    for request in range(active_count):
        start = query_start_loc_values[request]
        state_slot = int(binding.raw_state_slot_ids[request].item())
        binding.raw_interval_start_positions[state_slot] = positions[start] - int(
            accepted_tokens[request]
        )
    return {
        "query": query,
        "index_query": index_query,
        "raw_index_key": raw_key,
        "request_ids": request_tensor,
        "query_positions": position_tensor,
        "rope_positions": rope_positions,
        "sequence_lengths": sequence_lengths,
        "query_start_loc": torch.tensor(
            query_start_loc_values, dtype=torch.int32, device=device
        ),
        "num_accepted_tokens": accepted,
        "is_prefilling": torch.tensor(
            (*is_prefilling, *((False,) * (caps.max_batch - active_count))),
            dtype=torch.bool,
            device=device,
        ),
    }


def _require_free_cuda_bytes(device: torch.device, required: int) -> None:
    free, _total = torch.cuda.mem_get_info(device)
    if int(free) < int(required):
        pytest.skip(
            f"requires {required / 2**30:.1f} GiB free CUDA memory for a true "
            "high-offset paged-pool regression"
        )


def _rope_reference(
    value: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
    sections: tuple[int, int, int] | None,
    interleaved: bool,
) -> torch.Tensor:
    half = rotary_dim // 2
    pairs = torch.arange(half, device=value.device)
    if sections is None:
        axes = torch.zeros((half,), dtype=torch.int64, device=value.device)
    elif interleaved:
        axes = torch.zeros((half,), dtype=torch.int64, device=value.device)
        axes[(pairs % 3 == 1) & (pairs < 3 * sections[1])] = 1
        axes[(pairs % 3 == 2) & (pairs < 3 * sections[2])] = 2
    else:
        axes = torch.where(
            pairs < sections[0],
            0,
            torch.where(pairs < sections[0] + sections[1], 1, 2),
        )
    row_positions = positions[:, axes]
    row_cos = cos[row_positions, pairs]
    row_sin = sin[row_positions, pairs]
    source = value.float()
    result = source.clone()
    first = source[..., :half]
    second = source[..., half:rotary_dim]
    while row_cos.ndim < first.ndim:
        row_cos = row_cos.unsqueeze(1)
        row_sin = row_sin.unsqueeze(1)
    result[..., :half] = first * row_cos - second * row_sin
    result[..., half:rotary_dim] = second * row_cos + first * row_sin
    return result.to(value.dtype)


def _bf16_eager_rope_reference(
    value: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
    sections: tuple[int, int, int] | None,
    interleaved: bool,
) -> torch.Tensor:
    assert value.dtype == cos.dtype == sin.dtype == torch.bfloat16
    half = rotary_dim // 2
    pairs = torch.arange(half, device=value.device)
    if sections is None:
        axes = torch.zeros((half,), dtype=torch.int64, device=value.device)
    elif interleaved:
        axes = pairs % 3
    else:
        axes = torch.where(
            pairs < sections[0],
            0,
            torch.where(pairs < sections[0] + sections[1], 1, 2),
        )
    row_positions = positions[:, axes]
    row_cos = cos[row_positions, pairs]
    row_sin = sin[row_positions, pairs]
    while row_cos.ndim < value.ndim:
        row_cos = row_cos.unsqueeze(1)
        row_sin = row_sin.unsqueeze(1)
    result = value.clone()
    first = value[..., :half]
    second = value[..., half:rotary_dim]
    result[..., :half] = first * row_cos - second * row_sin
    result[..., half:rotary_dim] = second * row_cos + first * row_sin
    return result






def test_qsa_cache_requirements_are_pure_and_describe_shared_page_layout() -> None:
    requirements = qsa.cache_requirements(
        main_page_size=64,
        kv_heads=2,
        head_dim=256,
        index_head_dim=128,
        compress_ratio=4,
        budget=2048,
        position_axes=3,
        max_speculative_tokens=4,
    )

    assert requirements.main_k_page_shape == (64, 2, 256)
    assert requirements.main_v_page_shape == (64, 2, 256)
    assert requirements.main_k_page_nbytes == 64 * 2 * 256 * 2
    assert requirements.main_v_page_nbytes == requirements.main_k_page_nbytes
    assert requirements.main_kv_page_nbytes == 2 * requirements.main_k_page_nbytes
    assert requirements.compressed_page_shape == (16, 128)
    assert requirements.compressed_page_nbytes == 16 * 128 * 2
    assert requirements.raw_ring_capacity == 8
    assert requirements.raw_k_ring_shape == (8, 128)
    assert requirements.raw_logical_positions_shape == (8,)
    assert requirements.raw_rope_positions_shape == (8, 3)
    assert requirements.raw_interval_start_positions_shape == (1,)
    assert requirements.raw_k_ring_offset_bytes == 0
    assert requirements.raw_logical_positions_offset_bytes == 8 * 128 * 2
    assert requirements.raw_rope_positions_offset_bytes == 8 * 128 * 2 + 8 * 8
    assert requirements.raw_interval_start_positions_offset_bytes == (
        8 * 128 * 2 + 8 * 8 + 8 * 3 * 8
    )
    assert requirements.raw_page_nbytes == (
        requirements.raw_interval_start_positions_offset_bytes + 8
    )
    assert requirements.selection_width == 2051
    assert requirements.alignment_bytes == 256
    assert requirements.shared_compressed_raw_storage_legal

    fp8_requirements = qsa.cache_requirements(
        main_page_size=64,
        kv_heads=2,
        head_dim=256,
        index_head_dim=128,
        compress_ratio=4,
        kv_dtype=torch.float8_e4m3fn,
    )
    assert fp8_requirements.main_k_page_nbytes == 64 * 2 * 256
    assert fp8_requirements.main_kv_page_nbytes == 2 * 64 * 2 * 256
    assert (
        fp8_requirements.compressed_page_nbytes == requirements.compressed_page_nbytes
    )

    too_small = qsa.cache_requirements(
        main_page_size=32,
        kv_heads=2,
        head_dim=16,
        index_head_dim=16,
        position_axes=3,
    )
    assert not too_small.shared_compressed_raw_storage_legal




def test_qsa_caps_reject_unmodeled_or_inconsistent_state() -> None:
    caps = _caps(
        main_page_size=16,
        compressed_page_size=4,
        q_heads=12,
        kv_heads=1,
        head_dim=256,
    )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        replace(caps, device="cpu")
    speculative = replace(
        caps,
        max_speculative_tokens=1,
    )
    assert speculative.raw_ring_capacity == 8
    with pytest.raises(ValueError, match="nonnegative"):
        replace(caps, max_speculative_tokens=-1)
    with pytest.raises(ValueError, match="max_q_rows must be at least max_batch"):
        replace(caps, max_q_rows=1)
    with pytest.raises(ValueError, match="max_seq_len must be at least"):
        replace(caps, max_seq_len=2)
    with pytest.raises(ValueError, match="positive int32 positions"):
        replace(caps, max_seq_len=torch.iinfo(torch.int32).max + 1)
    with pytest.raises(ValueError, match="num_main_cache_pages"):
        replace(caps, num_main_cache_pages=torch.iinfo(torch.int32).max + 2)
    with pytest.raises(ValueError, match="num_compressed_cache_pages"):
        replace(caps, num_compressed_cache_pages=torch.iinfo(torch.int32).max + 2)
    separate_only = replace(caps, main_page_size=16, compressed_page_size=4)
    assert not separate_only.cache_requirements.shared_compressed_raw_storage_legal
    with pytest.raises(ValueError, match="index_q_norm_weight"):
        binding = _allocate_binding(caps)
        qsa.bind(
            binding.plan,
            scratch=binding.scratch,
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
            index_q_norm_weight=torch.empty((2, 16)),
            index_k_norm_weight=binding.index_k_norm_weight,
            rope_cos=binding.rope_cos,
            rope_sin=binding.rope_sin,
            output=binding.output,
            selected_positions=binding.selected_positions,
        )
    with pytest.raises(ValueError, match="mrope_interleaved"):
        replace(caps, mrope_interleaved=True)
    with pytest.raises(ValueError, match="round-robin axis counts"):
        replace(
            caps,
            position_axes=3,
            mrope_sections=(10, 11, 11),
            mrope_interleaved=True,
        )
    target = replace(
        caps,
        position_axes=3,
        mrope_sections=(11, 11, 10),
        mrope_interleaved=True,
    )
    assert target.mrope_sections == (11, 11, 10)






@pytest.mark.parametrize("rows", [1, 16, 32, 257, 513])
def test_qsa_scaled_topk_scratch_executes_fixed_capacity_rows(rows: int) -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=rows,
        max_raw_state_slots=rows,
        max_q_rows=rows,
    )
    binding = _allocate_binding(caps)
    binding.main_block_table[:, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    dynamic = _dynamic_inputs(
        binding,
        positions=(0,) * rows,
        request_ids=tuple(range(rows)),
    )

    result = qsa.run(binding, **dynamic)

    if rows > 204:
        assert binding.plan._layout.topk_nbytes > 1024 * 1024
    assert torch.isfinite(result).all()
    assert torch.all(binding.selected_positions[:rows, 0] == 0)
    assert torch.all(binding.selected_positions[:rows, 1:] == -1)


def test_qsa_binding_exposes_views_into_one_scratch_allocation() -> None:
    binding = _allocate_binding(_caps())
    scratch_begin = binding.scratch.data_ptr()
    scratch_end = scratch_begin + binding.scratch.numel()
    views = (
        binding.prepared_index_query,
        binding.scores,
        binding.eligible_group_counts,
        binding.merge_lengths,
        binding.topk_values,
        binding.topk_group_ids,
        binding.topk_values_b,
        binding.topk_group_ids_b,
        binding.partial_output,
        binding.partial_lse,
    )
    assert all(scratch_begin <= view.data_ptr() < scratch_end for view in views)


def test_qsa_binds_strided_int32_raw_state_column_and_int32_request_ids() -> None:
    device = require_sm120()
    caps = _caps(device)
    binding = _allocate_binding(caps)
    raw_block_table = torch.tensor([[2, 91], [1, 92]], dtype=torch.int32, device=device)
    raw_state_slot_ids = raw_block_table[:, 0]
    assert raw_state_slot_ids.stride() == (2,)
    binding = _rebind(binding, raw_state_slot_ids=raw_state_slot_ids)
    binding.main_block_table[:, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.compressed_block_table[1, 0] = 1
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    for state_slot in (2, 1):
        binding.raw_k_ring[state_slot, :3].zero_()
        binding.raw_logical_positions[state_slot, :3].copy_(
            torch.arange(3, dtype=torch.int64, device=device)
        )
        binding.raw_rope_positions[state_slot, :3, 0].copy_(
            torch.arange(3, dtype=torch.int64, device=device)
        )
    dynamic = _dynamic_inputs(binding, positions=(3, 3), request_ids=(0, 1))
    dynamic["request_ids"] = dynamic["request_ids"].to(torch.int32)

    result = qsa.run(binding, **dynamic)

    assert torch.isfinite(result).all()
    assert int(binding.raw_logical_positions[2, 3]) == 3
    assert int(binding.raw_logical_positions[1, 3]) == 3
    assert torch.isfinite(binding.compressed_k_cache[0, 0]).all()
    assert torch.isfinite(binding.compressed_k_cache[1, 0]).all()


def test_qsa_separate_raw_storage_can_exceed_compressed_page_bytes() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        main_page_size=32,
        compressed_page_size=8,
        position_axes=3,
        mrope_sections=(2, 1, 1),
    )
    assert caps.raw_page_nbytes == 264
    assert caps.compressed_page_nbytes == 256
    assert not caps.cache_requirements.shared_compressed_raw_storage_legal
    binding = _allocate_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))

    result = qsa.run(binding, **dynamic)

    assert torch.isfinite(result).all()
    assert int(binding.raw_interval_start_positions[0]) == 0


def test_qsa_shared_raw_storage_rejects_page_tail_overflow() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        main_page_size=32,
        compressed_page_size=8,
        position_axes=3,
        mrope_sections=(2, 1, 1),
    )
    binding = _allocate_binding(caps)
    page_nbytes = caps.compressed_page_nbytes
    backing = torch.empty(
        caps.num_compressed_cache_pages * page_nbytes + torch.int64.itemsize,
        dtype=torch.uint8,
        device=device,
    )
    compressed = (
        backing[: caps.num_compressed_cache_pages * page_nbytes]
        .view(torch.bfloat16)
        .view(
            caps.num_compressed_cache_pages,
            caps.compressed_page_size,
            caps.index_head_dim,
        )
    )
    page_elements = page_nbytes // torch.bfloat16.itemsize
    raw_ring = compressed.view(caps.num_compressed_cache_pages, -1).as_strided(
        (1, caps.raw_ring_capacity, caps.index_head_dim),
        (page_elements, caps.index_head_dim, 1),
    )
    backing_i64 = backing.view(torch.int64)
    payload_i64 = (
        caps.raw_ring_capacity
        * caps.index_head_dim
        * torch.bfloat16.itemsize
        // torch.int64.itemsize
    )
    raw_tags = backing_i64.as_strided(
        (1, caps.raw_ring_capacity),
        (page_nbytes // torch.int64.itemsize, 1),
        storage_offset=payload_i64,
    )
    raw_rope = backing_i64.as_strided(
        (1, caps.raw_ring_capacity, caps.position_axes),
        (page_nbytes // torch.int64.itemsize, caps.position_axes, 1),
        storage_offset=payload_i64 + caps.raw_ring_capacity,
    )
    raw_interval_start = backing_i64.as_strided(
        (1,),
        (page_nbytes // torch.int64.itemsize,),
        storage_offset=(
            payload_i64 + caps.raw_ring_capacity * (1 + caps.position_axes)
        ),
    )

    with pytest.raises(ValueError, match="fit inside one aliased compressed"):
        _rebind(
            binding,
            compressed_k_cache=compressed,
            raw_k_ring=raw_ring,
            raw_logical_positions=raw_tags,
            raw_rope_positions=raw_rope,
            raw_interval_start_positions=raw_interval_start,
        )



def test_qsa_run_rejects_dynamic_dtype_drift_before_launch() -> None:
    device = require_sm120()
    binding = _allocate_binding(_caps(device))
    dynamic = _dynamic_inputs(binding, positions=(0,))
    dynamic["sequence_lengths"] = dynamic["sequence_lengths"].to(torch.int64)
    with pytest.raises(TypeError, match="sequence_lengths"):
        qsa.run(binding, **dynamic)


def test_qsa_bind_rejects_stride_and_unsafe_storage_aliases() -> None:
    caps = _caps()
    binding = _allocate_binding(caps)
    aliased_tags = torch.empty(
        (caps.max_raw_state_slots, 1), dtype=torch.int64, device=caps.device
    ).expand(caps.max_raw_state_slots, caps.raw_ring_capacity)
    with pytest.raises(ValueError, match="raw_logical_positions.*unit inner stride"):
        _rebind(binding, raw_logical_positions=aliased_tags)

    output_alias = binding.main_k_cache.reshape(-1)[: binding.output.numel()].view(
        binding.output.shape
    )
    with pytest.raises(ValueError, match="output.*main_k_cache"):
        _rebind(binding, output=output_alias)


def test_qsa_bind_accepts_disjoint_compressed_tail_in_main_cache_pages() -> None:
    caps = _caps()
    binding = _allocate_binding(caps)
    main_plane_elements = caps.main_page_size * caps.kv_heads * caps.head_dim
    compressed_elements = caps.compressed_page_size * caps.index_head_dim
    page_elements = 2 * main_plane_elements + compressed_elements
    backing = torch.empty(
        (caps.num_main_cache_pages, page_elements),
        dtype=torch.bfloat16,
        device=caps.device,
    )
    main_strides = (
        page_elements,
        caps.kv_heads * caps.head_dim,
        caps.head_dim,
        1,
    )
    main_shape = (
        caps.num_main_cache_pages,
        caps.main_page_size,
        caps.kv_heads,
        caps.head_dim,
    )
    main_k = backing.as_strided(main_shape, main_strides, storage_offset=0)
    main_v = backing.as_strided(
        main_shape,
        main_strides,
        storage_offset=main_plane_elements,
    )
    compressed = backing.as_strided(
        (
            caps.num_compressed_cache_pages,
            caps.compressed_page_size,
            caps.index_head_dim,
        ),
        (page_elements, caps.index_head_dim, 1),
        storage_offset=2 * main_plane_elements,
    )

    packed = _rebind(
        binding,
        main_k_cache=main_k,
        main_v_cache=main_v,
        compressed_k_cache=compressed,
    )
    assert packed.main_k_cache.stride(0) == page_elements
    assert packed.compressed_k_cache.stride(0) == page_elements

    overlapping = backing.as_strided(
        compressed.shape,
        compressed.stride(),
        storage_offset=main_plane_elements - compressed_elements,
    )
    with pytest.raises(ValueError, match="compressed_k_cache.*main_k_cache"):
        _rebind(
            binding,
            main_k_cache=main_k,
            main_v_cache=main_v,
            compressed_k_cache=overlapping,
        )


def test_qsa_bind_rejects_misaligned_shared_raw_page_view() -> None:
    caps = _caps()
    binding = _allocate_shared_compressed_raw_binding(caps)
    page_elements = caps.compressed_page_size * caps.index_head_dim
    misaligned_ring = binding.compressed_k_cache.as_strided(
        binding.raw_k_ring.shape,
        (page_elements, caps.index_head_dim, 1),
        storage_offset=1,
    )
    with pytest.raises(ValueError, match="named page-tail offsets"):
        _rebind(binding, raw_k_ring=misaligned_ring)


def test_qsa_bind_rejects_overlapping_compressed_rows_in_shared_pool() -> None:
    caps = _caps()
    binding = _allocate_shared_compressed_raw_binding(caps)
    page_elements = caps.compressed_page_size * caps.index_head_dim
    overlapping_rows = binding.compressed_k_cache.as_strided(
        binding.compressed_k_cache.shape,
        (page_elements, caps.index_head_dim // 2, 1),
    )

    with pytest.raises(ValueError, match="dense contiguous compressed pages"):
        _rebind(binding, compressed_k_cache=overlapping_rows)


def test_qsa_shared_compressed_raw_pool_runs() -> None:
    device = require_sm120()
    caps = _caps(device, max_batch=1, max_raw_state_slots=1)
    binding = _allocate_shared_compressed_raw_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 1
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    binding.raw_k_ring[0, :3].zero_()
    binding.raw_logical_positions[0, :3].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    binding.raw_rope_positions[0, :3, 0].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    dynamic = _dynamic_inputs(binding, positions=(3,), request_ids=(0,))
    result = qsa.run(binding, **dynamic)
    assert torch.isfinite(result).all()
    assert int(binding.raw_logical_positions[0, 3]) == 3
    assert torch.isfinite(binding.compressed_k_cache[1, 0]).all()


def test_qsa_shared_pool_runs_under_fullgraph_compile_and_checks_runtime_aliases() -> (
    None
):
    device = require_sm120()
    caps = _caps(device, max_batch=1, max_raw_state_slots=1, max_q_rows=1)
    binding = _allocate_shared_compressed_raw_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
    names = (
        "query",
        "index_query",
        "raw_index_key",
        "request_ids",
        "query_positions",
        "rope_positions",
        "sequence_lengths",
        "query_start_loc",
        "num_accepted_tokens",
        "is_prefilling",
    )

    def decode(
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
    ) -> torch.Tensor:
        return qsa.run(
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
        )

    qsa.run(binding, **dynamic)
    binding.raw_interval_start_positions[0] = -1
    compiled = torch.compile(decode, fullgraph=True)
    arguments = tuple(dynamic[name] for name in names)
    result = compiled(*arguments)
    torch.cuda.synchronize(device)

    assert result.data_ptr() == binding.output.data_ptr()
    assert int(binding.selected_positions[0, 0]) == 0
    with pytest.raises(ValueError, match="output.*query"):
        compiled(binding.output[:1], *arguments[1:])


def test_qsa_shared_pool_cuda_graph_replay_has_stable_addresses_and_allocation() -> (
    None
):
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=2,
        max_raw_state_slots=2,
        max_q_rows=5,
        max_speculative_tokens=1,
        main_page_size=64,
        compressed_page_size=16,
    )
    binding = _allocate_shared_compressed_raw_binding(caps)
    binding.main_block_table[:, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    dynamic = _dynamic_inputs(
        binding,
        positions=(0, 1, 0, -1, -1),
        request_ids=(0, 0, 1, -1, -1),
        accepted_tokens=(1, 1),
    )

    qsa.run(binding, **dynamic)
    torch.cuda.synchronize(device)
    output_ptr = binding.output.data_ptr()
    scratch_ptr = binding.scratch.data_ptr()
    pool_ptr = binding.compressed_k_cache.data_ptr()
    binding.raw_interval_start_positions[:2] = -1
    graph = _cuda_graph()
    with torch.cuda.graph(graph):
        captured = qsa.run(binding, **dynamic)
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)

    for _ in range(3):
        binding.raw_interval_start_positions[:2] = -1
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_allocated(device) == allocated
        assert binding.output.data_ptr() == output_ptr
        assert binding.scratch.data_ptr() == scratch_ptr
        assert binding.compressed_k_cache.data_ptr() == pool_ptr
        assert torch.all(binding.selected_positions[3:5] == -1)
    assert captured.data_ptr() == output_ptr


@pytest.mark.parametrize(
    ("position_axes", "sections", "interleaved", "positions"),
    [
        (1, None, False, (7,)),
        (3, (2, 1, 1), False, (3, 7, 11)),
        (3, (2, 1, 1), True, (3, 7, 11)),
    ],
)
def test_qsa_prepare_query_matches_scalar_and_mrope_reference(
    position_axes: int,
    sections: tuple[int, int, int] | None,
    interleaved: bool,
    positions: tuple[int, ...],
) -> None:
    device = require_sm120()
    caps = _caps(
        device,
        position_axes=position_axes,
        mrope_sections=sections,
        mrope_interleaved=interleaved,
        main_page_size=64 if position_axes == 3 else 32,
        compressed_page_size=16 if position_axes == 3 else 8,
    )
    binding = _allocate_binding(caps)
    generator = torch.Generator(device="cpu").manual_seed(98551 + position_axes)
    index_query = torch.randn(
        (1, caps.index_heads, caps.index_head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    weight = torch.randn(
        (caps.index_head_dim,), generator=generator, dtype=torch.float32
    ).to(device)
    rope_cos = torch.randn(
        binding.rope_cos.shape, generator=generator, dtype=torch.float32
    ).to(device)
    rope_sin = torch.randn(
        binding.rope_sin.shape, generator=generator, dtype=torch.float32
    ).to(device)
    rope_positions = torch.tensor([positions], dtype=torch.int64, device=device)
    prepared = binding.prepared_index_query[:1]

    launch_prepare_index_query(
        index_query=index_query,
        request_ids=torch.tensor([0], dtype=torch.int64, device=device),
        norm_weight=weight,
        rope_positions=rope_positions,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        prepared_query=prepared,
        caps=caps,
    )
    normalized = gemma_rmsnorm_reference(index_query, weight, caps.rms_norm_eps)
    expected = _rope_reference(
        normalized,
        rope_positions,
        rope_cos,
        rope_sin,
        rotary_dim=caps.index_rotary_dim,
        sections=sections,
        interleaved=interleaved,
    )

    torch.testing.assert_close(prepared, expected, rtol=0.0, atol=2e-2)


def test_qsa_prepare_query_matches_bf16_eager_rotary_cast_order() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
        position_axes=3,
        mrope_sections=(11, 11, 10),
        mrope_interleaved=True,
    )
    binding = _allocate_binding(caps)
    generator = torch.Generator(device="cpu").manual_seed(98581)
    index_query = torch.randn(
        (1, caps.index_heads, caps.index_head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    weight = torch.randn(
        (caps.index_head_dim,), generator=generator, dtype=torch.float32
    ).to(device)
    cos = torch.randn(
        binding.rope_cos.shape, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    sin = torch.randn(
        binding.rope_sin.shape, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    positions = torch.tensor([[3, 7, 11]], dtype=torch.int64, device=device)
    normalized_by_kernel = torch.empty_like(binding.prepared_index_query[:1])

    launch_prepare_index_query(
        index_query=index_query,
        request_ids=torch.tensor([0], dtype=torch.int64, device=device),
        norm_weight=weight,
        rope_positions=positions,
        rope_cos=torch.ones_like(cos),
        rope_sin=torch.zeros_like(sin),
        prepared_query=normalized_by_kernel,
        caps=caps,
    )

    launch_prepare_index_query(
        index_query=index_query,
        request_ids=torch.tensor([0], dtype=torch.int64, device=device),
        norm_weight=weight,
        rope_positions=positions,
        rope_cos=cos,
        rope_sin=sin,
        prepared_query=binding.prepared_index_query[:1],
        caps=caps,
    )
    expected = _bf16_eager_rope_reference(
        normalized_by_kernel,
        positions,
        cos,
        sin,
        rotary_dim=caps.index_rotary_dim,
        sections=caps.mrope_sections,
        interleaved=caps.mrope_interleaved,
    )

    assert torch.equal(binding.prepared_index_query[:1], expected)


@pytest.mark.parametrize("interleaved", [False, True])
def test_qsa_completed_group_compression_matches_bf16_rounding_and_rope(
    interleaved: bool,
) -> None:
    device = require_sm120()
    position_axes = 3 if interleaved else 1
    sections = (2, 1, 1) if interleaved else None
    caps = _caps(
        device,
        position_axes=position_axes,
        mrope_sections=sections,
        mrope_interleaved=interleaved,
        main_page_size=64 if interleaved else 32,
        compressed_page_size=16 if interleaved else 8,
    )
    binding = _allocate_binding(caps)
    binding.compressed_block_table[0].copy_(
        torch.arange(caps.compressed_table_width, device=device, dtype=torch.int32)
    )
    generator = torch.Generator(device="cpu").manual_seed(98661 + position_axes)
    all_keys = torch.randn(
        (4, caps.index_head_dim), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    weight = torch.randn(
        (caps.index_head_dim,), generator=generator, dtype=torch.float32
    ).to(device)
    cos = torch.randn(
        binding.rope_cos.shape, generator=generator, dtype=torch.float32
    ).to(device)
    sin = torch.randn(
        binding.rope_sin.shape, generator=generator, dtype=torch.float32
    ).to(device)
    binding.raw_k_ring[0, :3].copy_(all_keys[:3])
    binding.raw_logical_positions[0, :3].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    first_position = (2, 5, 8) if interleaved else (2,)
    prior_rope = torch.tensor(
        [first_position, first_position, first_position],
        dtype=torch.int64,
        device=device,
    )
    binding.raw_rope_positions[0, :3].copy_(prior_rope)
    current_rope = torch.tensor(
        [[3, 6, 9] if interleaved else [3]],
        dtype=torch.int64,
        device=device,
    )

    launch_compress_completed_groups(
        raw_index_key=all_keys[3:],
        query_positions=torch.tensor([3], dtype=torch.int64, device=device),
        rope_positions=current_rope,
        request_ids=torch.tensor([0], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        raw_state_slot_ids=binding.raw_state_slot_ids,
        raw_k_ring=binding.raw_k_ring,
        raw_logical_positions=binding.raw_logical_positions,
        raw_rope_positions=binding.raw_rope_positions,
        key_norm_weight=weight,
        rope_cos=cos,
        rope_sin=sin,
        compressed_cache=binding.compressed_k_cache,
        compressed_block_table=binding.compressed_block_table,
        caps=caps,
    )
    pooled = all_keys.float().mean(0).to(torch.bfloat16)
    normalized = gemma_rmsnorm_reference(pooled, weight, caps.rms_norm_eps)
    expected = _rope_reference(
        normalized[None, :],
        torch.tensor([first_position], dtype=torch.int64, device=device),
        cos,
        sin,
        rotary_dim=caps.index_rotary_dim,
        sections=sections,
        interleaved=interleaved,
    )[0]

    torch.testing.assert_close(
        binding.compressed_k_cache[0, 0], expected, rtol=0.0, atol=2e-2
    )


def test_qsa_packed_speculative_rejection_replaces_stale_groups_before_use() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=8,
        max_seq_len=96,
        num_main_cache_pages=1,
        num_compressed_cache_pages=1,
        main_page_size=96,
        compressed_page_size=24,
        max_speculative_tokens=7,
    )
    binding = _allocate_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    binding.compressed_k_cache.zero_()
    main_k_before = binding.main_k_cache.clone()
    main_v_before = binding.main_v_cache.clone()

    first = _dynamic_inputs(
        binding,
        positions=tuple(range(8)),
        request_ids=(0,) * 8,
        accepted_tokens=(1,),
    )
    first["raw_index_key"].zero_()
    qsa.run(binding, **first)
    assert int(binding.raw_interval_start_positions[0]) == 0
    stale_group_one = binding.compressed_k_cache[0, 1].clone()

    replacement = _dynamic_inputs(
        binding,
        positions=(2, 3, 4),
        request_ids=(0, 0, 0),
        accepted_tokens=(2,),
    )
    replacement_keys = (
        torch.arange(3 * caps.index_head_dim, dtype=torch.float32)
        .reshape(3, caps.index_head_dim)
        .add_(1)
        .to(device=device, dtype=torch.bfloat16)
    )
    replacement["raw_index_key"].copy_(replacement_keys)
    qsa.run(binding, **replacement)
    assert int(binding.raw_interval_start_positions[0]) == 2
    assert torch.equal(binding.compressed_k_cache[0, 1], stale_group_one)
    assert torch.equal(
        binding.selected_positions[2, :5],
        torch.arange(5, dtype=torch.int32, device=device),
    )
    assert torch.all(binding.selected_positions[2, 5:] == -1)

    final = _dynamic_inputs(
        binding,
        positions=(5, 6, 7),
        request_ids=(0, 0, 0),
        accepted_tokens=(3,),
    )
    final_keys = (
        torch.arange(3 * caps.index_head_dim, dtype=torch.float32)
        .reshape(3, caps.index_head_dim)
        .flip(-1)
        .add_(3)
        .to(device=device, dtype=torch.bfloat16)
    )
    final["raw_index_key"].copy_(final_keys)
    qsa.run(binding, **final)
    expected_group_one = gemma_rmsnorm_reference(
        torch.cat((replacement_keys[2:3], final_keys), dim=0)
        .float()
        .mean(0)
        .to(torch.bfloat16),
        binding.index_k_norm_weight,
        caps.rms_norm_eps,
    )

    assert int(binding.raw_interval_start_positions[0]) == 5
    assert not torch.equal(binding.compressed_k_cache[0, 1], stale_group_one)
    torch.testing.assert_close(
        binding.compressed_k_cache[0, 1], expected_group_one, rtol=0.0, atol=2e-2
    )
    assert torch.equal(binding.main_k_cache, main_k_before)
    assert torch.equal(binding.main_v_cache, main_v_before)


def test_qsa_production_page_boundary_after_speculative_rollback_graph_replay() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=4,
        max_raw_state_slots=4,
        max_q_rows=16,
        max_seq_len=2048,
        num_main_cache_pages=146,
        num_compressed_cache_pages=146,
        main_page_size=1504,
        compressed_page_size=376,
        max_speculative_tokens=3,
        q_heads=12,
        kv_heads=1,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
        position_axes=3,
        mrope_sections=(11, 11, 10),
        mrope_interleaved=True,
    )
    binding = _allocate_binding(caps)
    binding.main_block_table[0, :2].copy_(
        torch.tensor([14, 145], dtype=torch.int32, device=device)
    )
    binding.compressed_block_table[0, :2].copy_(
        torch.tensor([14, 145], dtype=torch.int32, device=device)
    )
    binding.main_k_cache[14].normal_()
    binding.main_v_cache[14].normal_()
    binding.main_k_cache[145].normal_()
    binding.main_v_cache[145].normal_()
    binding.compressed_k_cache[14].normal_()
    binding.compressed_k_cache[145].zero_()
    binding.raw_interval_start_positions[0] = 1503

    crossing = _dynamic_inputs(
        binding,
        positions=(1504, 1505, 1506, 1507),
        request_ids=(0, 0, 0, 0),
        accepted_tokens=(1,),
    )
    qsa.run(binding, **crossing)
    assert int(binding.raw_interval_start_positions[0]) == 1504

    replay = _dynamic_inputs(
        binding,
        positions=(1506, 1507, 1508, 1509),
        request_ids=(0, 0, 0, 0),
        accepted_tokens=(2,),
    )
    raw_before = binding.raw_k_ring.clone()
    tags_before = binding.raw_logical_positions.clone()
    rope_before = binding.raw_rope_positions.clone()
    anchor_before = binding.raw_interval_start_positions.clone()
    compressed_before = binding.compressed_k_cache[145, 0].clone()

    graph = _cuda_graph()
    with torch.cuda.graph(graph):
        result = qsa.run(binding, **replay)
    binding.raw_k_ring.copy_(raw_before)
    binding.raw_logical_positions.copy_(tags_before)
    binding.raw_rope_positions.copy_(rope_before)
    binding.raw_interval_start_positions.copy_(anchor_before)
    binding.compressed_k_cache[145, 0].copy_(compressed_before)
    binding.output.zero_()
    binding.selected_positions.fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)

    expected = sparse_paged_gqa_reference(
        replay["query"],
        binding.main_k_cache,
        binding.main_v_cache,
        binding.main_block_table,
        replay["request_ids"],
        binding.selected_positions[:4],
        replay["query_positions"],
    )
    assert torch.all(binding.selected_positions[:4, 0] >= 0)
    assert torch.count_nonzero(result) > 0
    torch.testing.assert_close(result, expected, rtol=0.0, atol=2e-2)


@pytest.mark.parametrize("high_page", [False, True])
@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
def test_qsa_cute_scores_preserve_paged_contract_under_graph_replay(
    high_page: bool,
    request_dtype: torch.dtype,
) -> None:
    """Live extents reuse compilation; recycled pages retain 64-bit addressing."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.qsa._score_cute import (
        launch_score_representatives as launch_cute,
    )

    device = require_sm120()
    page_size, dim, groups = 752, 128, 65536
    first_page = 2**31 // (page_size * dim) + 1 if high_page else 3
    page_count = first_page + (groups + page_size - 1) // page_size
    caps = qsa.Caps(
        device=device,
        max_batch=16,
        max_raw_state_slots=16,
        max_q_rows=16,
        max_seq_len=groups * 4,
        num_main_cache_pages=page_count,
        num_compressed_cache_pages=page_count,
        main_page_size=page_size * 4,
        compressed_page_size=page_size,
    )
    torch.manual_seed(98797)
    if high_page:
        _require_free_cuda_bytes(
            device, page_count * page_size * dim * torch.bfloat16.itemsize + 2**28
        )
    cache = torch.empty(
        (page_count, page_size, dim), device=device, dtype=torch.bfloat16
    )
    cache[first_page:].normal_()
    table = torch.arange(first_page, page_count, device=device, dtype=torch.int32)
    table = table.flip(0).expand(16, -1).contiguous()
    prepared = torch.randn((16, 4, dim), device=device, dtype=torch.bfloat16)
    positions = torch.full((16,), 1023, device=device, dtype=torch.int64)
    requests = torch.arange(16, device=device, dtype=request_dtype)
    lengths = torch.full((16,), groups * 4, device=device, dtype=torch.int32)
    output = torch.empty((16, groups + 512), device=device, dtype=torch.float32)
    expected = torch.empty_like(output)
    counts = torch.empty(16, device=device, dtype=torch.int32)
    merges = torch.empty_like(counts)
    expected_counts, expected_merges = (
        torch.empty_like(counts),
        torch.empty_like(counts),
    )

    def invoke(launch, rows, offset, count, actual):
        launch(
            prepared_query=prepared[:rows],
            query_positions=positions[:rows],
            request_ids=requests[:rows],
            sequence_lengths=lengths,
            compressed_cache=cache,
            compressed_block_table=table,
            scores=(output if actual else expected)[:rows],
            eligible_counts=(counts if actual else expected_counts)[:rows],
            merge_lengths=(merges if actual else expected_merges)[:rows],
            group_offset=offset,
            group_count=count,
            caps=caps,
        )

    invoke(launch_cute, 16, 0, groups, True)
    with kernel_resolution_guard('QSA score rows and chunks are runtime quantities'):
        for rows, offset, count in ((1, 0, 65), (4, 0, groups), (16, 32768, 32768)):
            graph = _cuda_graph()
            with torch.cuda.graph(graph):
                invoke(launch_cute, rows, offset, count, True)
            for position in (1023, groups * 4 - 1):
                positions.fill_(position)
                if rows > 1:
                    requests[rows - 1] = -1
                output.fill_(123.0)
                expected.fill_(123.0)
                graph.replay()
                invoke(launch_score_representatives, rows, offset, count, False)
                torch.cuda.synchronize()
                assert torch.equal(counts[:rows], expected_counts[:rows])
                assert torch.equal(merges[:rows], expected_merges[:rows])
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
                assert torch.isfinite(output[0, : min(count, 256)]).any()
                requests.copy_(torch.arange(16, device=device, dtype=request_dtype))
    if high_page:
        assert first_page * cache.stride(0) > 2**31


@pytest.mark.parametrize("heads,dim", [(1, 16), (4, 128), (8, 256), (9, 128)])
@pytest.mark.parametrize("odd_alignment", [False, True])
def test_qsa_cute_scores_preserve_strides_masks_and_null_pages(
    heads: int,
    dim: int,
    odd_alignment: bool,
) -> None:
    from b12x.attention.qsa._score_cute import (
        launch_score_representatives as launch_cute,
    )

    device = require_sm120()
    pages, page, groups, rows = 5, 16, 65, 3
    caps = qsa.Caps(
        device=device,
        max_batch=2,
        max_raw_state_slots=2,
        max_q_rows=rows,
        max_seq_len=groups * 4,
        num_main_cache_pages=pages,
        num_compressed_cache_pages=pages,
        main_page_size=page * 4,
        compressed_page_size=page,
        index_heads=heads,
        index_head_dim=dim,
        index_rotary_dim=min(dim, 64),
    )
    torch.manual_seed(98901)
    padding = int(odd_alignment)
    storage = torch.empty(
        1 + pages * page * (dim + padding),
        device=device,
        dtype=torch.bfloat16,
    )
    cache = storage.as_strided(
        (pages, page, dim),
        (page * (dim + padding), dim + padding, 1),
        storage_offset=padding,
    )
    cache.normal_()
    query_storage = torch.randn(
        1 + rows * heads * dim,
        device=device,
        dtype=torch.bfloat16,
    )
    query = query_storage[padding : padding + rows * heads * dim].view(rows, heads, dim)
    table = torch.arange(pages, device=device, dtype=torch.int32).flip(0)
    table = table.expand(2, -1).contiguous()
    positions = torch.full((rows,), 259, device=device, dtype=torch.int64)
    requests = torch.tensor([0, 1, -1], device=device, dtype=torch.int32)
    lengths = torch.tensor([260, 260], device=device, dtype=torch.int32)
    scores = torch.empty((rows, groups), device=device)
    counts = torch.empty(rows, device=device, dtype=torch.int32)
    merges = torch.empty_like(counts)
    kwargs = dict(
        prepared_query=query,
        query_positions=positions,
        request_ids=requests,
        sequence_lengths=lengths,
        compressed_cache=cache,
        compressed_block_table=table,
        scores=scores,
        eligible_counts=counts,
        merge_lengths=merges,
        group_offset=0,
        group_count=groups,
        caps=caps,
    )
    launch_score_representatives(**kwargs)
    expected = scores.clone()
    launch_cute(**kwargs)
    graph = _cuda_graph()
    with torch.cuda.graph(graph):
        launch_cute(**kwargs)
    scores.fill_(123)
    graph.replay()
    torch.testing.assert_close(scores, expected, rtol=1e-5, atol=1e-5)
    assert counts.tolist() == merges.tolist() == [groups, groups, 0]
    assert torch.isneginf(scores[-1]).all()
    expected_order = torch.argsort(expected, descending=True, stable=True)
    assert torch.equal(
        torch.argsort(scores, descending=True, stable=True), expected_order
    )

    # A one-page chunk isolates a null page-table entry: its groups score
    # negative infinity and the other rows are unaffected.
    kwargs["group_count"] = page
    table[0, 0] = -1
    scores.fill_(123)
    launch_cute(**kwargs)
    assert torch.isneginf(scores[0, :page]).all()
    torch.testing.assert_close(
        scores[1, :page],
        expected[1, :page],
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.all(scores[:, page:] == 123)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("keys_kind", ["random", "zero_ties", "ordered_ties"])
def test_qsa_cute_scores_preserve_chunked_selection_and_attention_graph_replay(
    monkeypatch,
    keys_kind: str,
    overlap: bool,
) -> None:
    """Score replacement preserves the complete state/selection/read transaction."""
    import b12x.attention.qsa._kernels as kernel_module
    import b12x.attention.qsa._score_cute as cute_module

    device = require_sm120()
    monkeypatch.setattr(
        qsa_contract, "_SCORE_WORKSPACE_LIMIT_BYTES", 4 * (512 + 256) * 4
    )
    caps = qsa.Caps(
        device=device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=4,
        max_seq_len=4096,
        num_main_cache_pages=64,
        num_compressed_cache_pages=64,
        main_page_size=64,
        compressed_page_size=16,
    )
    torch.manual_seed(98903)
    binding = _allocate_binding(caps)
    binding = _rebind(
        binding,
        selection_stream=torch.cuda.Stream(device=device) if overlap else None,
    )
    ready = torch.cuda.Event() if overlap else None

    def invoke():
        if ready is not None:
            ready.record()
        return qsa.run(binding, **dynamic, index_ready=ready)

    assert binding.state.num_score_chunks == 4
    binding.main_k_cache.normal_()
    binding.main_v_cache.normal_()
    binding.main_block_table[0].copy_(torch.arange(64, device=device))
    binding.compressed_block_table[0].copy_(torch.arange(64, device=device))
    binding.raw_k_ring.zero_()
    binding.compressed_k_cache.normal_()
    if keys_kind == "zero_ties":
        binding.compressed_k_cache.zero_()
    elif keys_kind == "ordered_ties":
        amplitude = (torch.arange(caps.max_groups, device=device) * 37 % 101).to(
            torch.bfloat16
        )
        binding.compressed_k_cache.copy_(
            amplitude[:, None]
            .expand(-1, caps.index_head_dim)
            .reshape_as(binding.compressed_k_cache)
        )
    dynamic = _dynamic_inputs(
        binding, positions=(4094, -1, -1, -1), request_ids=(0, -1, -1, -1)
    )
    if keys_kind != "random":
        dynamic["index_query"].fill_(1)
    state_names = (
        "compressed_k_cache",
        "raw_k_ring",
        "raw_logical_positions",
        "raw_rope_positions",
        "raw_interval_start_positions",
    )
    initial = {name: getattr(binding, name).clone() for name in state_names}

    def reset():
        for name, value in initial.items():
            getattr(binding, name).copy_(value)
        binding.scratch.fill_(173)
        binding.output.fill_(23)

    scalar_launch = kernel_module.launch_score_representatives
    cute_launch = cute_module.launch_score_representatives

    def scalar_reference(*, _prepared=None, **kwargs):
        # The independent scalar oracle resolves its own Triton programs;
        # the binding's prepared scorer belongs to the CuTe implementation.
        return scalar_launch(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cute_module, "launch_score_representatives", scalar_reference)
        reset()
        expected_output = invoke().clone()
        expected_positions = binding.selected_positions.clone()
        expected_state = {name: getattr(binding, name).clone() for name in state_names}
    assert torch.isfinite(expected_output[0]).all()
    assert expected_output[0].abs().max() > 0
    with monkeypatch.context() as patch:
        patch.setattr(cute_module, "launch_score_representatives", cute_launch)
        reset()
        invoke()
        reset()
        graph = _cuda_graph()
        with torch.cuda.graph(graph):
            actual = invoke()
        for _ in range(3):
            reset()
            graph.replay()
            assert torch.equal(binding.selected_positions, expected_positions)
            torch.testing.assert_close(
                actual, expected_output, rtol=0, atol=0, equal_nan=True
            )
            for name, value in expected_state.items():
                assert torch.equal(getattr(binding, name), value)


def test_qsa_paged_representative_scores_match_fp32_reference() -> None:
    device = require_sm120()
    caps = _caps(device)
    binding = _allocate_binding(caps)
    binding.compressed_block_table[0].copy_(
        torch.arange(caps.compressed_table_width, device=device, dtype=torch.int32)
    )
    generator = torch.Generator(device="cpu").manual_seed(98773)
    logical_keys = torch.randn(
        (5, caps.index_head_dim), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    for group in range(5):
        page, offset = divmod(group, caps.compressed_page_size)
        binding.compressed_k_cache[page, offset].copy_(logical_keys[group])
    prepared = torch.randn(
        (1, caps.index_heads, caps.index_head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    scores = torch.empty((1, 5), dtype=torch.float32, device=device)
    eligible = torch.empty((1,), dtype=torch.int32, device=device)
    lengths = torch.empty_like(eligible)
    query_position = torch.tensor([19], dtype=torch.int64, device=device)

    launch_score_representatives(
        prepared_query=prepared,
        query_positions=query_position,
        request_ids=torch.tensor([0], dtype=torch.int64, device=device),
        sequence_lengths=torch.tensor([20, 0], dtype=torch.int32, device=device),
        compressed_cache=binding.compressed_k_cache,
        compressed_block_table=binding.compressed_block_table,
        scores=scores,
        eligible_counts=eligible,
        merge_lengths=lengths,
        group_offset=0,
        group_count=5,
        caps=caps,
    )
    expected_scores, _ = score_select_reference(
        prepared, logical_keys, query_position, 20, 4, 2048
    )

    assert eligible.item() == 5 and lengths.item() == 5
    torch.testing.assert_close(scores, expected_scores, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    ("rank", "expected_eligible"), [(0, 1), (1, 0), (2, 0), (3, 0)]
)
def test_qsa_cute_score_respects_dcp_rank_eligibility(
    rank: int,
    expected_eligible: int,
) -> None:
    from b12x.attention.qsa._score_cute import launch_score_representatives

    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        dcp_size=4,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=4,
    )
    prepared = torch.ones(
        (1, caps.index_heads, caps.index_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    cache = torch.ones(
        (
            caps.num_compressed_cache_pages,
            caps.compressed_page_size,
            caps.index_head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    table = torch.arange(
        caps.compressed_table_width, dtype=torch.int32, device=device,
    ).unsqueeze(0)
    scores = torch.empty((1, caps.max_groups), dtype=torch.float32, device=device)
    eligible = torch.empty((1,), dtype=torch.int32, device=device)
    lengths = torch.empty_like(eligible)

    launch_score_representatives(
        prepared_query=prepared,
        query_positions=torch.tensor([3], dtype=torch.int64, device=device),
        request_ids=torch.tensor([0], dtype=torch.int64, device=device),
        sequence_lengths=torch.tensor([4], dtype=torch.int32, device=device),
        compressed_cache=cache,
        compressed_block_table=table,
        scores=scores,
        eligible_counts=eligible,
        merge_lengths=lengths,
        group_offset=0,
        group_count=caps.max_groups,
        caps=caps,
    )

    assert eligible.item() == expected_eligible
    assert lengths.item() == expected_eligible
    if expected_eligible == 0:
        assert torch.all(scores == -torch.inf)


def test_qsa_representative_scores_match_explicit_32_32_1_partition() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=260,
        num_main_cache_pages=9,
        num_compressed_cache_pages=9,
    )
    binding = _allocate_binding(caps)
    binding.compressed_block_table[0].copy_(
        torch.arange(9, dtype=torch.int32, device=device)
    )
    generator = torch.Generator(device="cpu").manual_seed(98791)
    logical_keys = torch.randn(
        (65, caps.index_head_dim), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    for group in range(65):
        page, offset = divmod(group, caps.compressed_page_size)
        binding.compressed_k_cache[page, offset].copy_(logical_keys[group])
    prepared = torch.randn(
        (1, caps.index_heads, caps.index_head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    query_position = torch.tensor([259], dtype=torch.int64, device=device)
    request_ids = torch.tensor([0], dtype=torch.int64, device=device)
    sequence_lengths = torch.tensor([260], dtype=torch.int32, device=device)
    partitioned = torch.full((1, 65), torch.nan, dtype=torch.float32, device=device)
    one_shot = torch.empty_like(partitioned)
    partitioned_eligible = torch.empty((1,), dtype=torch.int32, device=device)
    partitioned_lengths = torch.empty_like(partitioned_eligible)
    one_shot_eligible = torch.empty_like(partitioned_eligible)
    one_shot_lengths = torch.empty_like(partitioned_eligible)

    for group_offset, group_count in ((0, 32), (32, 32), (64, 1)):
        launch_score_representatives(
            prepared_query=prepared,
            query_positions=query_position,
            request_ids=request_ids,
            sequence_lengths=sequence_lengths,
            compressed_cache=binding.compressed_k_cache,
            compressed_block_table=binding.compressed_block_table,
            scores=partitioned,
            eligible_counts=partitioned_eligible,
            merge_lengths=partitioned_lengths,
            group_offset=group_offset,
            group_count=group_count,
            caps=caps,
        )
    launch_score_representatives(
        prepared_query=prepared,
        query_positions=query_position,
        request_ids=request_ids,
        sequence_lengths=sequence_lengths,
        compressed_cache=binding.compressed_k_cache,
        compressed_block_table=binding.compressed_block_table,
        scores=one_shot,
        eligible_counts=one_shot_eligible,
        merge_lengths=one_shot_lengths,
        group_offset=0,
        group_count=65,
        caps=caps,
    )
    expected_scores, _ = score_select_reference(
        prepared, logical_keys, query_position, 260, 4, 2048
    )

    assert partitioned_eligible.item() == one_shot_eligible.item() == 65
    assert partitioned_lengths.item() == one_shot_lengths.item() == 65
    torch.testing.assert_close(partitioned, one_shot, rtol=0.0, atol=0.0)
    torch.testing.assert_close(partitioned, expected_scores, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    ("position", "eligible", "tail"),
    [(0, 0, [0]), (2, 0, [0, 1, 2]), (3, 1, []), (4, 1, [4])],
)
def test_qsa_group_expansion_has_exact_boundary_tail(
    position: int,
    eligible: int,
    tail: list[int],
) -> None:
    device = require_sm120()
    caps = _caps(device, max_batch=1, max_raw_state_slots=1, max_q_rows=1)
    topk = torch.full((1, caps.group_budget), -1, dtype=torch.int32, device=device)
    if eligible:
        topk[0, 0] = 0
    selected = torch.empty((1, caps.selection_width), dtype=torch.int32, device=device)
    launch_expand_selected_groups(
        topk_group_ids=topk,
        eligible_counts=torch.tensor([eligible], dtype=torch.int32, device=device),
        query_positions=torch.tensor([position], dtype=torch.int64, device=device),
        selected_positions=selected,
        caps=caps,
    )
    prefix = list(range(4)) if eligible else []
    expected = prefix + tail
    assert selected[0, : len(expected)].tolist() == expected
    assert torch.all(selected[0, len(expected) :] == -1)


def test_qsa_forced_chunked_topk_is_exact_and_deterministic_on_ties(
    monkeypatch,
) -> None:
    device = require_sm120()
    # Three score chunks force one carry fold across the 512-group cutoff.
    monkeypatch.setattr(
        qsa_contract,
        "_SCORE_WORKSPACE_LIMIT_BYTES",
        (512 + 256) * torch.float32.itemsize,
    )
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=4096,
        num_main_cache_pages=256,
        num_compressed_cache_pages=256,
    )
    planned = qsa.plan(caps)
    binding = _allocate_binding(caps, plan=planned)
    assert binding.state.num_score_chunks == 4
    binding.main_block_table[0].copy_(
        torch.arange(caps.main_table_width, dtype=torch.int32, device=device)
    )
    binding.compressed_block_table[0].copy_(
        torch.arange(caps.compressed_table_width, dtype=torch.int32, device=device)
    )
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    binding.compressed_k_cache.zero_()
    binding.raw_k_ring.zero_()
    binding.raw_logical_positions[0, :3].copy_(
        torch.arange(4092, 4095, dtype=torch.int64, device=device)
    )
    binding.raw_rope_positions[0, :3, 0].copy_(
        torch.arange(4092, 4095, dtype=torch.int64, device=device)
    )
    dynamic = _dynamic_inputs(binding, positions=(4095,), request_ids=(0,))
    dynamic["raw_index_key"].zero_()

    qsa.run(binding, **dynamic)
    first = binding.selected_positions[0].clone()
    binding.raw_interval_start_positions[0] = 4094
    qsa.run(binding, **dynamic)
    second = binding.selected_positions[0].clone()

    expected = torch.full((caps.selection_width,), -1, dtype=torch.int32, device=device)
    expected[: caps.budget] = torch.arange(
        caps.budget, dtype=torch.int32, device=device
    )
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


def test_qsa_chunked_topk_orders_mixed_scores_by_score_then_lower_group_id(
    monkeypatch,
) -> None:
    device = require_sm120()
    monkeypatch.setattr(
        qsa_contract,
        "_SCORE_WORKSPACE_LIMIT_BYTES",
        (512 + 256) * torch.float32.itemsize,
    )
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=4096,
        num_main_cache_pages=128,
        num_compressed_cache_pages=128,
    )
    binding = _allocate_binding(caps)
    binding.main_block_table[0].copy_(
        torch.arange(caps.main_table_width, dtype=torch.int32, device=device)
    )
    binding.compressed_block_table[0].copy_(
        torch.arange(caps.compressed_table_width, dtype=torch.int32, device=device)
    )
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    group_ids = torch.arange(caps.max_groups, device=device)
    amplitudes = ((group_ids * 37) % 101).to(torch.bfloat16)
    logical_keys = amplitudes[:, None].expand(-1, caps.index_head_dim).contiguous()
    binding.compressed_k_cache.copy_(logical_keys.view_as(binding.compressed_k_cache))
    dynamic = _dynamic_inputs(binding, positions=(4094,), request_ids=(0,))
    dynamic["index_query"].fill_(1)

    qsa.run(binding, **dynamic)
    _, expected = score_select_reference(
        binding.prepared_index_query[:1],
        logical_keys,
        dynamic["query_positions"],
        sequence_length=4095,
        compress_ratio=caps.compress_ratio,
        budget=caps.budget,
    )
    assert torch.equal(binding.selected_positions[:1], expected)


def test_qsa_group_budget_2048_executes_full_decode() -> None:
    device = require_sm120()
    caps = _caps(device, budget=8192, max_batch=1, max_raw_state_slots=1)
    binding = _allocate_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    binding.raw_k_ring[0, :3].zero_()
    binding.raw_logical_positions[0, :3].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    binding.raw_rope_positions[0, :3, 0].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    dynamic = _dynamic_inputs(binding, positions=(3,), request_ids=(0,))

    result = qsa.run(binding, **dynamic)
    assert torch.isfinite(result).all()
    assert torch.equal(
        binding.selected_positions[0, :4],
        torch.arange(4, dtype=torch.int32, device=device),
    )
    assert torch.all(binding.selected_positions[0, 4:] == -1)


def test_qsa_compressed_page_addressing_crosses_signed_int32_product() -> None:
    device = require_sm120()
    page_elements = 8 * 16
    high_page = torch.iinfo(torch.int32).max // page_elements + 1
    page_count = high_page + 1
    pool_nbytes = page_count * page_elements * torch.bfloat16.itemsize
    _require_free_cuda_bytes(device, pool_nbytes + 2 * 2**30)
    binding: qsa.Binding | None = None
    dynamic: dict[str, torch.Tensor] | None = None
    try:
        caps = _caps(
            device,
            max_batch=1,
            max_raw_state_slots=1,
            max_q_rows=1,
            num_compressed_cache_pages=page_count,
        )
        try:
            binding = _allocate_binding(caps)
        except torch.OutOfMemoryError:
            pytest.skip("CUDA allocator could not reserve the high-page QSA pool")
        binding.main_block_table[0, 0] = 0
        binding.compressed_block_table[0, 0] = high_page
        binding.main_k_cache.zero_()
        binding.main_v_cache.zero_()
        binding.raw_k_ring[0, :3].normal_()
        binding.raw_logical_positions[0, :3].copy_(
            torch.arange(3, dtype=torch.int64, device=device)
        )
        binding.raw_rope_positions[0, :3, 0].copy_(
            torch.arange(3, dtype=torch.int64, device=device)
        )
        dynamic = _dynamic_inputs(binding, positions=(3,), request_ids=(0,))

        result = qsa.run(binding, **dynamic)

        assert high_page * int(binding.compressed_k_cache.stride(0)) > int(
            torch.iinfo(torch.int32).max
        )
        assert torch.isfinite(result).all()
        assert torch.isfinite(binding.compressed_k_cache[high_page, 0]).all()
        assert torch.count_nonzero(binding.compressed_k_cache[high_page, 0]) > 0
    finally:
        del dynamic
        del binding
        torch.cuda.empty_cache()


def test_qsa_raw_state_slot_addressing_crosses_signed_int32_products() -> None:
    device = require_sm120()
    page_elements = 8 * 16
    page_i64 = page_elements * torch.bfloat16.itemsize // torch.int64.itemsize
    high_slot = torch.iinfo(torch.int32).max // page_i64 + 1
    page_count = high_slot + 1
    pool_nbytes = page_count * page_elements * torch.bfloat16.itemsize
    _require_free_cuda_bytes(device, pool_nbytes + 3 * 2**30)
    binding: qsa.Binding | None = None
    dynamic: dict[str, torch.Tensor] | None = None
    try:
        caps = _caps(
            device,
            max_batch=1,
            max_raw_state_slots=page_count,
            max_q_rows=1,
            num_compressed_cache_pages=page_count,
        )
        try:
            binding = _allocate_shared_compressed_raw_binding(
                caps, initialize_metadata=False
            )
        except torch.OutOfMemoryError:
            pytest.skip("CUDA allocator could not reserve the high-slot QSA pool")
        raw_block_table = torch.full((1, 2), -1, dtype=torch.int32, device=device)
        raw_block_table[0, 0] = high_slot
        binding = _rebind(binding, raw_state_slot_ids=raw_block_table[:, 0])
        assert binding.raw_state_slot_ids.dtype == torch.int32
        assert binding.raw_state_slot_ids.stride() == (2,)
        binding.main_block_table[0, 0] = 0
        binding.main_k_cache.zero_()
        binding.main_v_cache.zero_()
        dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
        dynamic["request_ids"] = dynamic["request_ids"].to(torch.int32)

        result = qsa.run(binding, **dynamic)

        int32_max = int(torch.iinfo(torch.int32).max)
        assert high_slot * int(binding.raw_k_ring.stride(0)) > int32_max
        assert high_slot * int(binding.raw_logical_positions.stride(0)) > int32_max
        assert high_slot * int(binding.raw_rope_positions.stride(0)) > int32_max
        assert (
            high_slot * int(binding.raw_interval_start_positions.stride(0)) > int32_max
        )
        assert torch.isfinite(result).all()
        assert torch.equal(
            binding.raw_k_ring[high_slot, 0], dynamic["raw_index_key"][0]
        )
        assert int(binding.raw_logical_positions[high_slot, 0]) == 0
        assert int(binding.raw_rope_positions[high_slot, 0, 0]) == 0
        assert int(binding.raw_interval_start_positions[high_slot]) == 0
    finally:
        del dynamic
        del binding
        torch.cuda.empty_cache()


@pytest.mark.parametrize("budget", [2048, 8192])
def test_qsa_checkpoint_geometry_executes_integrated_interleaved_mrope_decode(
    budget: int,
) -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=64,
        num_main_cache_pages=1,
        num_compressed_cache_pages=1,
        main_page_size=64,
        compressed_page_size=16,
        q_heads=24,
        kv_heads=2,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
        position_axes=3,
        mrope_sections=(11, 11, 10),
        mrope_interleaved=True,
        budget=budget,
    )
    binding = _allocate_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.main_k_cache.normal_()
    binding.main_v_cache.normal_()
    binding.raw_k_ring[0, :3].normal_()
    binding.raw_logical_positions[0, :3].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    binding.raw_rope_positions[0, :3].copy_(
        torch.tensor(
            [[0, 4, 8], [1, 5, 9], [2, 6, 10]],
            dtype=torch.int64,
            device=device,
        )
    )
    dynamic = _dynamic_inputs(binding, positions=(3,), request_ids=(0,))
    dynamic["rope_positions"][0].copy_(
        torch.tensor([3, 7, 11], dtype=torch.int64, device=device)
    )
    main_k_before = binding.main_k_cache.clone()
    main_v_before = binding.main_v_cache.clone()

    actual = qsa.run(binding, **dynamic)
    expected = sparse_paged_gqa_reference(
        dynamic["query"],
        binding.main_k_cache,
        binding.main_v_cache,
        binding.main_block_table,
        dynamic["request_ids"],
        binding.selected_positions[:1],
        dynamic["query_positions"],
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-2)
    assert torch.equal(binding.main_k_cache, main_k_before)
    assert torch.equal(binding.main_v_cache, main_v_before)
    assert torch.isfinite(binding.compressed_k_cache[0, 0]).all()


def test_qsa_tp4_geometry_matches_reference_under_cuda_graph_replay() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=3,
        max_seq_len=64,
        num_main_cache_pages=1,
        num_compressed_cache_pages=1,
        main_page_size=64,
        compressed_page_size=16,
        q_heads=6,
        kv_heads=1,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
        position_axes=3,
        mrope_sections=(11, 11, 10),
        mrope_interleaved=True,
    )
    binding = _allocate_binding(
        caps,
        plan=qsa.plan(
            caps,
            invocation=_invocation(
                caps,
                rope_positions={"dtype": "int64", "strides": (1, 3)},
            ),
        ),
    )
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.main_k_cache.normal_()
    binding.main_v_cache.normal_()
    binding.raw_k_ring[0, :3].normal_()
    binding.raw_logical_positions[0, :3].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    binding.raw_rope_positions[0, :3].copy_(
        torch.tensor(
            [[0, 4, 8], [1, 5, 9], [2, 6, 10]],
            dtype=torch.int64,
            device=device,
        )
    )
    prior_raw_keys = binding.raw_k_ring[0, :3].clone()
    prior_rope_positions = binding.raw_rope_positions[0, :3].clone()
    generator = torch.Generator(device="cpu").manual_seed(189223)
    combined_rope_cache = torch.randn(
        (caps.max_seq_len, caps.index_rotary_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=torch.bfloat16)
    rotary_half = caps.index_rotary_dim // 2
    rope_cos = combined_rope_cache[:, :rotary_half]
    rope_sin = combined_rope_cache[:, rotary_half:]
    assert rope_cos.stride() == rope_sin.stride() == (caps.index_rotary_dim, 1)
    binding = _rebind(binding, rope_cos=rope_cos, rope_sin=rope_sin)

    dynamic = _dynamic_inputs(
        binding,
        positions=(3, -1, -1),
        request_ids=(0, -1, -1),
    )
    positions_by_axis = torch.tensor(
        [[3, -1, -1], [7, -1, -1], [11, -1, -1]],
        dtype=torch.int64,
        device=device,
    )
    dynamic["rope_positions"] = positions_by_axis.T
    assert dynamic["rope_positions"].stride() == (1, 3)
    main_k_before = binding.main_k_cache.clone()
    main_v_before = binding.main_v_cache.clone()

    qsa.run(binding, **dynamic)
    binding.raw_interval_start_positions[0] = 2
    graph = _cuda_graph()
    with torch.cuda.graph(graph):
        captured = qsa.run(binding, **dynamic)
    binding.raw_interval_start_positions[0] = 2
    dynamic["query"].normal_()
    dynamic["index_query"].normal_()
    dynamic["raw_index_key"].normal_()
    graph.replay()
    torch.cuda.synchronize(device)

    expected_prepared = _bf16_eager_rope_reference(
        gemma_rmsnorm_reference(
            dynamic["index_query"][:1],
            binding.index_q_norm_weight,
            caps.rms_norm_eps,
        ),
        dynamic["rope_positions"][:1],
        rope_cos,
        rope_sin,
        rotary_dim=caps.index_rotary_dim,
        sections=caps.mrope_sections,
        interleaved=caps.mrope_interleaved,
    )
    torch.testing.assert_close(
        binding.prepared_index_query[:1], expected_prepared, rtol=0.0, atol=2e-2
    )
    pooled = (
        torch.cat((prior_raw_keys, dynamic["raw_index_key"][:1]), dim=0)
        .float()
        .mean(dim=0)
        .to(torch.bfloat16)
    )
    expected_representative = _bf16_eager_rope_reference(
        gemma_rmsnorm_reference(
            pooled,
            binding.index_k_norm_weight,
            caps.rms_norm_eps,
        )[None, :],
        prior_rope_positions[:1],
        rope_cos,
        rope_sin,
        rotary_dim=caps.index_rotary_dim,
        sections=caps.mrope_sections,
        interleaved=caps.mrope_interleaved,
    )[0]
    torch.testing.assert_close(
        binding.compressed_k_cache[0, 0],
        expected_representative,
        rtol=0.0,
        atol=2e-2,
    )
    expected = sparse_paged_gqa_reference(
        dynamic["query"],
        binding.main_k_cache,
        binding.main_v_cache,
        binding.main_block_table,
        dynamic["request_ids"],
        binding.selected_positions[:3],
        dynamic["query_positions"],
    )

    assert captured.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(captured, expected, rtol=0.0, atol=2e-2)
    assert torch.equal(binding.main_k_cache, main_k_before)
    assert torch.equal(binding.main_v_cache, main_v_before)


@pytest.mark.parametrize(
    "rows,shared_pool,kv_dtype,high_page",
    [
        (rows, shared, torch.bfloat16, False)
        for rows in (1, 4, 16)
        for shared in (False, True)
    ]
    + [(rows, False, torch.float8_e4m3fn, False) for rows in (1, 4, 16)]
    + [
        (4, True, torch.float8_e4m3fn, False),
        (4, False, torch.float8_e4m3fn, True),
    ],
)
@pytest.mark.parametrize("direct_views", [False, True])
def test_qsa_run_streams_match_combined_across_graph_replay(
    rows: int,
    shared_pool: bool,
    kv_dtype: torch.dtype,
    high_page: bool,
    direct_views: bool,
) -> None:
    """One run owns the selection fork/join without changing the transaction."""
    device = require_sm120()
    first_page = (2**31 // (3008 * 2 * 256) + 1) if high_page else 8
    caps = _caps(
        device,
        max_batch=4,
        max_raw_state_slots=4,
        max_q_rows=16,
        max_seq_len=3008,
        main_page_size=3008,
        compressed_page_size=752,
        num_main_cache_pages=first_page + 4,
        num_compressed_cache_pages=12,
        q_heads=24,
        kv_heads=2,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
        max_speculative_tokens=3,
        kv_dtype=kv_dtype,
    )
    allocate = (
        _allocate_shared_compressed_raw_binding if shared_pool else _allocate_binding
    )
    if high_page:
        pool_bytes = (
            4
            * caps.num_main_cache_pages
            * caps.main_page_size
            * caps.kv_heads
            * caps.head_dim
            * caps.kv_dtype.itemsize
        )
        projection_width = (caps.index_heads + 1) * caps.index_head_dim
        projection_bytes = (
            ((rows - 1) * (2**31 + projection_width) + projection_width)
            * torch.bfloat16.itemsize
            if direct_views
            else 0
        )
        _require_free_cuda_bytes(device, pool_bytes + projection_bytes + 2**29)
    reference = allocate(caps)
    candidate = allocate(caps)
    live_reqs = max(1, rows // 4)
    for request in range(live_reqs):
        page = first_page + request
        reference.main_k_cache[page].copy_(
            torch.randn(
                reference.main_k_cache.shape[1:],
                dtype=torch.bfloat16,
                device=device,
            )
        )
        reference.main_v_cache[page].copy_(
            torch.randn(
                reference.main_v_cache.shape[1:],
                dtype=torch.bfloat16,
                device=device,
            )
        )
        candidate.main_k_cache[page].copy_(reference.main_k_cache[page])
        candidate.main_v_cache[page].copy_(reference.main_v_cache[page])
        for binding in (reference, candidate):
            binding.main_block_table[request, 0] = page
            binding.compressed_block_table[request, 0] = 8 + request
    for binding in (reference, candidate):
        binding.raw_k_ring.zero_()
        binding.compressed_k_cache[8:].zero_()
    positions = tuple(row % 4 for row in range(rows))
    requests = tuple(row // 4 for row in range(rows))
    dynamic = _dynamic_inputs(reference, positions=positions, request_ids=requests)
    candidate.raw_interval_start_positions.copy_(reference.raw_interval_start_positions)
    names = (
        "compressed_k_cache",
        "raw_k_ring",
        "raw_logical_positions",
        "raw_rope_positions",
        "raw_interval_start_positions",
    )
    # Include initialized and unused storage so unexpected writes are observable.
    for name in names:
        getattr(candidate, name).copy_(getattr(reference, name))
    initial = {name: getattr(reference, name).clone() for name in names}

    def reset(binding):
        for name, value in initial.items():
            getattr(binding, name).copy_(value)
        binding.output.fill_(23)
        binding.scratch.fill_(173)

    select_args = {name: value for name, value in dynamic.items() if name != "query"}
    if direct_views:
        query_width = caps.index_heads * caps.index_head_dim
        projection_width = query_width + caps.index_head_dim
        projection = torch.empty_strided(
            (rows, projection_width),
            (2**31 + projection_width if high_page else projection_width, 1),
            dtype=torch.bfloat16,
            device=device,
        )
        index_query = projection[:, :query_width].view(
            rows, caps.index_heads, caps.index_head_dim
        )
        raw_index_key = projection[:, query_width:]
        select_args.update(index_query=index_query, raw_index_key=raw_index_key)
        candidate = _rebind(candidate, output=candidate.output[:rows])

    def update_projection():
        if direct_views:
            index_query.copy_(dynamic["index_query"])
            raw_index_key.copy_(dynamic["raw_index_key"])

    update_projection()
    attention_args = {
        name: dynamic[name] for name in ("query", "request_ids", "query_positions")
    }
    projected_query = dynamic["query"].clone()
    attention_args["query"] = projected_query
    selector_stream = torch.cuda.Stream(device=device)
    candidate = _rebind(candidate, selection_stream=selector_stream)
    ready = torch.cuda.Event()

    def stream_call():
        ready.record()
        projected_query.copy_(dynamic["query"])
        return qsa.run(
            candidate,
            **attention_args,
            **{
                name: value
                for name, value in select_args.items()
                if name not in attention_args
            },
            index_ready=ready,
        )

    reset(reference)
    qsa.run(reference, **dynamic)
    reset(candidate)
    stream_call()
    reset(candidate)
    graph = _cuda_graph()
    with torch.cuda.graph(graph):
        result = stream_call()
    for _ in range(3):
        dynamic["query"].normal_()
        dynamic["index_query"].normal_()
        dynamic["raw_index_key"].normal_()
        update_projection()
        reset(reference)
        qsa.run(reference, **dynamic)
        reset(candidate)
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(result, reference.output[:rows], rtol=0, atol=0)
        assert torch.equal(
            candidate.selected_positions[:rows], reference.selected_positions[:rows]
        )
        for name in names:
            torch.testing.assert_close(
                getattr(candidate, name),
                getattr(reference, name),
                rtol=0,
                atol=0,
                equal_nan=True,
            )


def test_qsa_rejects_insufficient_live_output_before_mutation() -> None:
    caps = _caps(
        max_q_rows=4, q_heads=24, head_dim=256, index_heads=4, index_head_dim=128
    )
    binding = _allocate_binding(caps)
    binding = _rebind(binding, output=binding.output[:1])
    dynamic = _dynamic_inputs(binding, positions=(0, 1), request_ids=(0, 0))
    binding.scratch.fill_(173)
    binding.output.fill_(23)
    with pytest.raises(ValueError, match="output capacity"):
        qsa.run(binding, **dynamic)
    assert torch.all(binding.scratch == 173)
    assert torch.all(binding.output == 23)


def test_qsa_prefill_handoff_completes_the_trailing_open_group() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
    )
    binding = _allocate_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.main_k_cache.normal_()
    binding.main_v_cache.normal_()
    binding.compressed_k_cache.zero_()

    trailing_keys = (
        torch.arange(
            3 * caps.index_head_dim,
            dtype=torch.float32,
            device=device,
        )
        .reshape(3, caps.index_head_dim)
        .to(torch.bfloat16)
        / 64
    )
    binding.raw_k_ring[0, :3].copy_(trailing_keys)
    binding.raw_logical_positions[0, :3].copy_(
        torch.tensor([4, 5, 6], dtype=torch.int64, device=device)
    )
    binding.raw_rope_positions[0, :3, 0].copy_(
        torch.tensor([4, 5, 6], dtype=torch.int64, device=device)
    )
    dynamic = _dynamic_inputs(binding, positions=(7,), request_ids=(0,))
    assert int(binding.raw_interval_start_positions[0]) == 6
    main_k_before = binding.main_k_cache.clone()
    main_v_before = binding.main_v_cache.clone()

    actual = qsa.run(binding, **dynamic)

    pooled = (
        torch.cat((trailing_keys, dynamic["raw_index_key"]), dim=0)
        .float()
        .mean(dim=0)
        .to(torch.bfloat16)
    )
    expected_representative = gemma_rmsnorm_reference(
        pooled,
        binding.index_k_norm_weight,
        caps.rms_norm_eps,
    )
    torch.testing.assert_close(
        binding.compressed_k_cache[0, 1],
        expected_representative,
        rtol=0.0,
        atol=2e-2,
    )
    expected = sparse_paged_gqa_reference(
        dynamic["query"],
        binding.main_k_cache,
        binding.main_v_cache,
        binding.main_block_table,
        dynamic["request_ids"],
        binding.selected_positions[:1],
        dynamic["query_positions"],
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-2)
    assert int(binding.raw_interval_start_positions[0]) == 7
    assert int(binding.raw_logical_positions[0, 3]) == 7
    assert int(binding.raw_rope_positions[0, 3, 0]) == 7
    assert torch.equal(binding.main_k_cache, main_k_before)
    assert torch.equal(binding.main_v_cache, main_v_before)


def test_qsa_negative_one_anchor_initializes_position_zero() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
    )
    binding = _allocate_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.main_k_cache.normal_()
    binding.main_v_cache.normal_()
    binding.compressed_k_cache.zero_()

    initial = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
    assert int(binding.raw_interval_start_positions[0]) == -1
    assert torch.isfinite(qsa.run(binding, **initial)).all()
    assert int(binding.raw_interval_start_positions[0]) == 0


def test_qsa_checkpoint_geometry_matches_eight_step_mrope_ring_oracle() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=64,
        num_main_cache_pages=1,
        num_compressed_cache_pages=2,
        main_page_size=64,
        compressed_page_size=16,
        q_heads=24,
        kv_heads=2,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
        position_axes=3,
        mrope_sections=(11, 11, 10),
        mrope_interleaved=True,
    )
    binding = _allocate_shared_compressed_raw_binding(caps)
    generator = torch.Generator(device="cpu").manual_seed(99127)

    def random_bf16(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            shape, generator=generator, dtype=torch.float32, device="cpu"
        ).to(device=device, dtype=torch.bfloat16)

    main_k = random_bf16(tuple(binding.main_k_cache.shape))
    main_v = random_bf16(tuple(binding.main_v_cache.shape))
    queries = random_bf16((8, caps.q_heads, caps.head_dim))
    index_queries = random_bf16((8, caps.index_heads, caps.index_head_dim))
    raw_keys = random_bf16((8, caps.index_head_dim))
    q_weight = torch.randn(
        (caps.index_head_dim,), generator=generator, dtype=torch.float32
    ).to(device)
    k_weight = torch.randn(
        (caps.index_head_dim,), generator=generator, dtype=torch.float32
    ).to(device)
    rope_cos = random_bf16(tuple(binding.rope_cos.shape))
    rope_sin = random_bf16(tuple(binding.rope_sin.shape))
    rope_positions = torch.stack(
        (
            torch.arange(8, dtype=torch.int64),
            torch.arange(8, dtype=torch.int64) + 8,
            torch.arange(8, dtype=torch.int64) + 16,
        ),
        dim=1,
    ).to(device)
    binding = _rebind(
        binding,
        index_q_norm_weight=q_weight,
        index_k_norm_weight=k_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    binding.main_k_cache.copy_(main_k)
    binding.main_v_cache.copy_(main_v)
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 1
    main_k_before = binding.main_k_cache.clone()
    main_v_before = binding.main_v_cache.clone()
    expected_representatives: list[torch.Tensor] = []

    for position in range(8):
        position_tensor = torch.tensor([position], dtype=torch.int64, device=device)
        dynamic = {
            "query": queries[position : position + 1],
            "index_query": index_queries[position : position + 1],
            "raw_index_key": raw_keys[position : position + 1],
            "request_ids": torch.tensor([0], dtype=torch.int64, device=device),
            "query_positions": position_tensor,
            "rope_positions": rope_positions[position : position + 1],
            "sequence_lengths": torch.tensor(
                [position + 1], dtype=torch.int32, device=device
            ),
            "query_start_loc": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "num_accepted_tokens": torch.tensor([1], dtype=torch.int32, device=device),
            "is_prefilling": torch.tensor([False], dtype=torch.bool, device=device),
        }

        actual = qsa.run(binding, **dynamic).clone()

        expected_prepared = _bf16_eager_rope_reference(
            gemma_rmsnorm_reference(
                dynamic["index_query"], q_weight, caps.rms_norm_eps
            ),
            dynamic["rope_positions"],
            rope_cos,
            rope_sin,
            rotary_dim=caps.index_rotary_dim,
            sections=caps.mrope_sections,
            interleaved=caps.mrope_interleaved,
        )
        torch.testing.assert_close(
            binding.prepared_index_query[:1],
            expected_prepared,
            rtol=0.0,
            atol=2e-2,
        )

        if (position + 1) % caps.compress_ratio == 0:
            first = position - caps.compress_ratio + 1
            pooled = (
                raw_keys[first : position + 1].float().mean(dim=0).to(torch.bfloat16)
            )
            normalized = gemma_rmsnorm_reference(pooled, k_weight, caps.rms_norm_eps)
            representative = _bf16_eager_rope_reference(
                normalized[None, :],
                rope_positions[first : first + 1],
                rope_cos,
                rope_sin,
                rotary_dim=caps.index_rotary_dim,
                sections=caps.mrope_sections,
                interleaved=caps.mrope_interleaved,
            )[0]
            expected_representatives.append(representative)
            torch.testing.assert_close(
                binding.compressed_k_cache[1, len(expected_representatives) - 1],
                representative,
                rtol=0.0,
                atol=2e-2,
            )

        reference_representatives = (
            torch.stack(expected_representatives)
            if expected_representatives
            else torch.empty(
                (0, caps.index_head_dim), dtype=torch.bfloat16, device=device
            )
        )
        _, expected_selected = score_select_reference(
            expected_prepared,
            reference_representatives,
            position_tensor,
            position + 1,
            caps.compress_ratio,
            caps.budget,
        )
        assert torch.equal(binding.selected_positions[:1], expected_selected)
        expected_output = sparse_paged_gqa_reference(
            dynamic["query"],
            binding.main_k_cache,
            binding.main_v_cache,
            binding.main_block_table,
            dynamic["request_ids"],
            expected_selected,
            position_tensor,
        )
        torch.testing.assert_close(actual, expected_output, rtol=0.0, atol=2e-2)

        ring_slot = position % caps.raw_ring_capacity
        assert torch.equal(binding.raw_k_ring[0, ring_slot], raw_keys[position])
        assert int(binding.raw_logical_positions[0, ring_slot]) == position
        assert torch.equal(
            binding.raw_rope_positions[0, ring_slot], rope_positions[position]
        )

    assert binding.raw_logical_positions[0].tolist() == [4, 5, 6, 7]
    assert len(expected_representatives) == 2
    assert torch.equal(binding.main_k_cache, main_k_before)
    assert torch.equal(binding.main_v_cache, main_v_before)


def test_qsa_decode_full_path_matches_exact_gqa_and_keeps_main_cache_read_only() -> (
    None
):
    device = require_sm120()
    caps = _caps(device, q_heads=24, head_dim=256, index_head_dim=128)
    binding = _allocate_binding(caps)
    binding.main_block_table.copy_(
        torch.arange(caps.main_table_width, dtype=torch.int32, device=device)
        .expand(caps.max_batch, -1)
        .contiguous()
    )
    binding.compressed_block_table.copy_(
        torch.arange(caps.compressed_table_width, dtype=torch.int32, device=device)
        .expand(caps.max_batch, -1)
        .contiguous()
    )
    binding.main_k_cache.normal_()
    binding.main_v_cache.normal_()
    binding.compressed_k_cache.zero_()
    dynamic = _dynamic_inputs(binding, positions=(3, -1), request_ids=(0, -1))
    dynamic["rope_positions"][1].fill_(-1)
    binding.raw_k_ring[0, :3].copy_(
        torch.arange(3 * caps.index_head_dim, device=device)
        .reshape(3, caps.index_head_dim)
        .to(torch.bfloat16)
        / 128
    )
    binding.raw_logical_positions[0, :3].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    binding.raw_rope_positions[0, :3, 0].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    main_k_before = binding.main_k_cache.clone()
    main_v_before = binding.main_v_cache.clone()
    dummy_ring_before = binding.raw_k_ring[1].clone()

    actual = qsa.run(binding, **dynamic)
    expected = sparse_paged_gqa_reference(
        dynamic["query"],
        binding.main_k_cache,
        binding.main_v_cache,
        binding.main_block_table,
        dynamic["request_ids"],
        binding.selected_positions[:2],
        dynamic["query_positions"],
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=2e-2)
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.equal(binding.main_k_cache, main_k_before)
    assert torch.equal(binding.main_v_cache, main_v_before)
    assert torch.equal(binding.raw_k_ring[1], dummy_ring_before)
    assert torch.equal(
        binding.selected_positions[0, :4],
        torch.arange(4, dtype=torch.int32, device=device),
    )
    assert torch.all(binding.selected_positions[0, 4:] == -1)


def test_qsa_score_uses_tensor_device_and_stream() -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.qsa import _score_cute as implementation

    device = require_sm120()
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two explicitly assigned CUDA devices")
    device = torch.device("cuda", torch.cuda.current_device())
    other = (device.index + 1) % torch.cuda.device_count()
    caps = _caps(device, q_heads=24, head_dim=256, index_heads=4, index_head_dim=128)
    query = torch.ones(1, 4, 128, dtype=torch.bfloat16, device=device)
    cache = torch.ones(4, 8, 128, dtype=torch.bfloat16, device=device)
    output = torch.empty(1, 16, device=device)
    counts = torch.empty(1, dtype=torch.int32, device=device)
    kwargs = dict(
        prepared_query=query,
        query_positions=torch.full((1,), 63, dtype=torch.int64, device=device),
        request_ids=torch.zeros(1, dtype=torch.int32, device=device),
        sequence_lengths=torch.full((1,), 64, dtype=torch.int32, device=device),
        compressed_cache=cache,
        compressed_block_table=torch.arange(4, dtype=torch.int32, device=device).view(
            1, 4
        ),
        scores=output,
        eligible_counts=counts,
        merge_lengths=torch.empty_like(counts),
        group_offset=0,
        group_count=16,
        caps=caps,
    )
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    implementation._CACHE.clear()
    with torch.cuda.stream(stream), torch.cuda.device(other):
        implementation.launch_score_representatives(**kwargs)
        assert torch.cuda.current_device() == other
    stream.synchronize()
    warmed = tuple(implementation._CACHE.items())
    with kernel_resolution_guard('QSA scorer retains tensor device under ambient device changes'):
        with torch.cuda.stream(stream), torch.cuda.device(other):
            graph = _cuda_graph()
            with (
                torch.cuda.device(device),
                torch.cuda.graph(graph, stream=stream),
                torch.cuda.device(other),
            ):
                implementation.launch_score_representatives(**kwargs)
            for value in (2, 3):
                with torch.cuda.device(device):
                    cache.fill_(value)
                    output.fill_(float("nan"))
                graph.replay()
                stream.synchronize()
                torch.testing.assert_close(
                    output, torch.full_like(output, 4 * 128**0.5 * value)
                )
        assert tuple(implementation._CACHE.items()) == warmed


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("budget", [2048, 8192])
@pytest.mark.parametrize("output_rows", [4, 66])
def test_qsa_run_prewarm_covers_bound_capacity_and_graph_replay(
    overlap: bool,
    budget: int,
    output_rows: int,
) -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard

    device = require_sm120()
    caps = _caps(
        device,
        max_q_rows=128,
        q_heads=24,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
        max_speculative_tokens=3,
        budget=budget,
    )
    binding = _allocate_binding(caps)
    binding = _rebind(
        binding,
        output=binding.output[:output_rows],
        selection_stream=torch.cuda.Stream(device=device) if overlap else None,
    )
    binding.main_k_cache.zero_()
    binding.main_v_cache.fill_(1)
    binding.main_block_table[:, 0] = 0
    binding.compressed_k_cache.zero_()
    binding.raw_k_ring.zero_()
    persistent = (
        binding.main_k_cache,
        binding.main_v_cache,
        binding.compressed_k_cache,
        binding.raw_k_ring,
        binding.raw_logical_positions,
        binding.raw_rope_positions,
        binding.raw_interval_start_positions,
    )
    ready = torch.cuda.Event() if overlap else None
    buffers = (binding.scratch, binding.output, binding.selected_positions)
    pointers = tuple(t.data_ptr() for t in buffers)
    completion_event = binding._selection_done
    with kernel_resolution_guard('QSA run uses prewarmed capacity across live rows'):
        for dtype in (torch.int32, torch.int64):
            for rows in sorted({1, min(4, output_rows), output_rows}):
                dynamic = _dynamic_inputs(
                    binding,
                    positions=(0,) + (-1,) * (rows - 1),
                    request_ids=(0,) + (-1,) * (rows - 1),
                )
                dynamic["request_ids"] = dynamic["request_ids"].to(dtype)

                def invoke():
                    if ready is not None:
                        ready.record()
                    return qsa.run(binding, **dynamic, index_ready=ready)

                graph = _cuda_graph()
                with torch.cuda.graph(graph):
                    result = invoke()
                for value in (1, 2):
                    binding.raw_interval_start_positions.fill_(-1)
                    binding.main_v_cache.fill_(value)
                    binding.output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in (
                        "allocation.all.allocated",
                        "allocated_bytes.all.allocated",
                    ):
                        assert after[key] == before[key]
                    assert torch.all(result[0] == value)
                    assert torch.all(result[1:] == 0)
                    assert tuple(t.data_ptr() for t in buffers) == pointers
                    assert binding._selection_done is completion_event


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("shared_pool", [False, True])
def test_qsa_run_streams_fullgraph_preserves_state_and_alias_checks(
    overlap: bool,
    shared_pool: bool,
) -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        q_heads=24,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
    )
    allocate = (
        _allocate_shared_compressed_raw_binding if shared_pool else _allocate_binding
    )
    binding = allocate(caps)
    binding = _rebind(
        binding,
        selection_stream=torch.cuda.Stream(device=device) if overlap else None,
    )
    binding.main_block_table[0, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.fill_(2)
    binding.compressed_k_cache.zero_()
    binding.raw_k_ring.zero_()
    ready = torch.cuda.Event() if overlap else None
    dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
    if ready is not None:
        ready.record()

    def invoke(query):
        if ready is not None:
            ready.record()
        return qsa.run(binding, **{**dynamic, "query": query}, index_ready=ready)

    compiled = torch.compile(invoke, fullgraph=True)
    for launch in (invoke, compiled):
        for value in (2, 3):
            binding.raw_interval_start_positions.fill_(-1)
            binding.main_v_cache.fill_(value)
            result = launch(dynamic["query"])
            assert torch.all(result == value)
            assert result.data_ptr() == binding.output.data_ptr()
        binding.raw_interval_start_positions.fill_(-1)
        before = binding.raw_interval_start_positions.clone()
        with pytest.raises(ValueError, match="output.*query"):
            launch(binding.output)
        assert torch.equal(binding.raw_interval_start_positions, before)
    binding.raw_interval_start_positions.fill_(-1)
    graph = _cuda_graph()
    with torch.cuda.graph(graph):
        result = compiled(dynamic["query"])
    for value in (4, 5):
        binding.raw_interval_start_positions.fill_(-1)
        binding.main_v_cache.fill_(value)
        graph.replay()
        assert torch.all(result == value)


def test_qsa_run_stream_options_validate_before_mutation() -> None:
    device = require_sm120()
    binding = _allocate_binding(
        _caps(
            device,
            q_heads=24,
            head_dim=256,
            index_heads=4,
            index_head_dim=128,
            index_rotary_dim=64,
        )
    )
    dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
    binding.scratch.fill_(173)
    binding.output.fill_(23)
    ready = torch.cuda.Event()
    with pytest.raises(ValueError, match="requires a binding"):
        qsa.run(binding, **dynamic, index_ready=ready)
    binding = _rebind(binding, selection_stream=torch.cuda.Stream(device=device))
    with pytest.raises(ValueError, match="requires an index_ready"):
        qsa.run(binding, **dynamic)
    with pytest.raises(ValueError, match="must be recorded"):
        qsa.run(binding, **dynamic, index_ready=ready)
    ready.record()
    with pytest.raises(TypeError, match="query must be a tensor"):
        qsa.run(binding, **{**dynamic, "query": None}, index_ready=ready)
    assert torch.all(binding.scratch == 173)
    assert torch.all(binding.output == 23)
    assert not hasattr(qsa, "select")
    assert not hasattr(qsa, "run_selected")


@torch.library.custom_op(
    "b12x_tests::qsa_delayed_producer",
    mutates_args=("query", "value_cache"),
)
def _qsa_delayed_producer(query: torch.Tensor, value_cache: torch.Tensor) -> None:
    # Model an opaque cache writer; Torch functionalization must not defer writes.
    torch.cuda._sleep(100_000_000)
    query.zero_()
    value_cache.fill_(2)


@_qsa_delayed_producer.register_fake
def _qsa_delayed_producer_fake(query: torch.Tensor, value_cache: torch.Tensor) -> None:
    return None


@pytest.mark.parametrize("compiled", [False, True])
def test_qsa_run_selection_forks_before_main_producer_and_joins_before_attention(
    monkeypatch,
    compiled: bool,
) -> None:
    """A delayed producer proves stream ordering; this is not a speed benchmark."""
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        q_heads=24,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
    )
    binding = _allocate_binding(caps)
    binding = _rebind(binding, selection_stream=torch.cuda.Stream(device=device))
    binding.main_block_table[0, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    binding.compressed_k_cache.zero_()
    binding.raw_k_ring.zero_()
    dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
    ready = torch.cuda.Event()
    started = torch.cuda.Event(enable_timing=True, external=True)
    selected = torch.cuda.Event(enable_timing=True, external=True)
    produced = torch.cuda.Event(enable_timing=True, external=True)
    original = qsa_contract._qsa_decode_impl

    def observe(*args, **kwargs):
        original(*args, **kwargs)
        if args[0] is None:
            selected.record()

    monkeypatch.setattr(qsa_contract, "_qsa_decode_impl", observe)
    ready.record()

    def invoke():
        started.record()
        ready.record()
        _qsa_delayed_producer(dynamic["query"], binding.main_v_cache)
        produced.record()
        return qsa.run(binding, **dynamic, index_ready=ready)

    launch = torch.compile(invoke, fullgraph=True) if compiled else invoke
    launch()
    torch.cuda.synchronize()
    for graph_mode in (False, True):
        binding.raw_interval_start_positions.fill_(-1)
        if graph_mode:
            graph = _cuda_graph()
            with torch.cuda.graph(graph):
                result = launch()
        for _ in range(2):
            binding.raw_interval_start_positions.fill_(-1)
            dynamic["query"].fill_(float("nan"))
            binding.main_v_cache.fill_(float("nan"))
            if graph_mode:
                graph.replay()
            else:
                result = launch()
            torch.cuda.synchronize()
            assert torch.all(result == 2)
            assert 0 <= started.elapsed_time(selected) < started.elapsed_time(produced)


def _draft_selection_state(binding):
    storage_plan = qsa.draft_selection_plan(binding.plan)
    storage = {
        spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in storage_plan.storage_specs()
    }
    state = storage_plan.bind(storage=storage)
    state.reset()
    return state


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("high_page", [False, True])
def test_qsa_run_reuses_draft_anchors_without_mutating_selector_state(
    kv_dtype,
    request_dtype,
    high_page,
    monkeypatch,
):
    """Accepted-row mapping and causal tails survive changed-input replay."""
    from b12x._lib.runtime_control import kernel_resolution_guard

    device = require_sm120()
    first_page = 2**31 // (3008 * 2 * 256) + 1 if high_page else 0
    caps = _caps(
        device,
        max_batch=4,
        max_raw_state_slots=4,
        max_q_rows=16,
        max_seq_len=3008,
        main_page_size=3008,
        compressed_page_size=752,
        num_main_cache_pages=first_page + 4,
        num_compressed_cache_pages=4,
        q_heads=24,
        kv_heads=2,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
        max_speculative_tokens=3,
        kv_dtype=kv_dtype,
    )
    if high_page:
        pool_bytes = (
            2
            * caps.num_main_cache_pages
            * caps.main_page_size
            * caps.kv_heads
            * caps.head_dim
            * caps.kv_dtype.itemsize
        )
        _require_free_cuda_bytes(device, pool_bytes + 2**29)
    binding = _allocate_binding(caps)
    state = _draft_selection_state(binding)
    binding = _rebind(binding, draft_selection=state)
    for request in range(4):
        page = first_page + request
        binding.main_block_table[request, 0] = page
        binding.compressed_block_table[request, 0] = request
        binding.main_k_cache[page].zero_()
        binding.main_v_cache[page].fill_(request + 1)
    binding.raw_k_ring.zero_()
    binding.compressed_k_cache.zero_()
    initial = _dynamic_inputs(
        binding,
        positions=tuple(i % 4 for i in range(16)),
        request_ids=tuple(i // 4 for i in range(16)),
    )
    compiled = (
        not high_page and kv_dtype == torch.bfloat16 and request_dtype == torch.int64
    )
    if compiled:

        def record_round():
            state.reset()
            return qsa.run(binding, **initial)

        torch.compile(record_round, fullgraph=True)()
    else:
        qsa.run(binding, **initial)
    assert state.num_source_rows.item() == 16
    torch.testing.assert_close(state.logical_positions, initial["query_positions"])
    assert torch.equal(state.selected_positions, binding.selected_positions)
    persistent = (
        binding.compressed_k_cache,
        binding.raw_k_ring,
        binding.raw_logical_positions,
        binding.raw_rope_positions,
        binding.raw_interval_start_positions,
    )
    snapshots = tuple(tensor.clone() for tensor in persistent)
    if compiled:
        reuse_plan = qsa.plan(replace(caps, max_q_rows=4))
        binding = _rebind(
            replace(binding, plan=reuse_plan),
            selected_positions=binding.selected_positions[:4],
            output=binding.output[:4],
            draft_selection=state,
        )
    source_rows = torch.tensor([3, 7, 11, 15], dtype=request_dtype, device=device)
    query = initial["query"][:4].clone()
    requests = torch.tensor([2, 0, -1, 1], device=device, dtype=request_dtype)
    positions = torch.tensor([4, 5, -1, 4], device=device, dtype=torch.int64)
    pointers = tuple(
        t.data_ptr()
        for t in (
            binding.output,
            binding.scratch,
            binding._draft_work_positions,
        )
    )

    def invoke(rows=4):
        return qsa.run(
            binding,
            query=query[:rows],
            request_ids=requests[:rows],
            query_positions=positions[:rows],
            reuse=qsa.DraftSelectionReuse(source_rows),
        )

    if compiled:
        invoke = torch.compile(invoke, fullgraph=True)
    for rows in (1, 3, 4):
        invoke(rows)
    with kernel_resolution_guard('QSA draft reuse must retain prewarmed reader capacity'):
        for rows in (1, 3, 4):
            graph = _cuda_graph()
            with torch.cuda.graph(graph):
                result = invoke(rows)
            for tail_end in (5, 4):
                positions[0] = tail_end
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                )
                assert torch.all(result[0] == 3)
                if rows >= 3:
                    assert torch.all(result[1] == 1)
                    assert torch.all(result[2] == 0)
                if rows == 4:
                    assert torch.all(result[3] == 2)
                assert binding._draft_work_positions[
                    0, caps.selection_width :
                ].tolist() == ([4, 5, -1] if tail_end == 5 else [4, -1, -1])
            positions[0] = 7
            graph.replay()
            assert torch.all(binding._draft_work_positions[0] == -1)
            positions[0] = 4
            # A row outside the last ordinary run must not reuse older state.
            source_rows[2] = 16
            graph.replay()
            assert torch.all(binding._draft_work_positions[0] == -1)
            source_rows[2] = 11
            state.num_source_rows.fill_(11)
            graph.replay()
            assert torch.all(binding._draft_work_positions[0] == -1)
            state.num_source_rows.fill_(16)
            state.reset()
            graph.replay()
            assert torch.all(binding._draft_work_positions[0] == -1)
            state.num_source_rows.fill_(16)
            requests[0] = caps.max_batch
            graph.replay()
            assert torch.all(binding._draft_work_positions[0] == -1)
            requests[0] = 2
    for tensor, expected in zip(persistent, snapshots, strict=True):
        assert torch.equal(tensor, expected)
    assert pointers == tuple(
        t.data_ptr()
        for t in (
            binding.output,
            binding.scratch,
            binding._draft_work_positions,
        )
    )


def test_qsa_draft_reuse_attends_appended_tail_positions() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_speculative_tokens=3,
    )
    binding = _allocate_binding(caps)
    state = _draft_selection_state(binding)
    binding = _rebind(binding, draft_selection=state)
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.zero_()
    binding.main_v_cache[0, 4].fill_(1)
    binding.compressed_k_cache.zero_()
    binding.raw_k_ring.zero_()

    initial = _dynamic_inputs(binding, positions=(3,), request_ids=(0,))
    initial["query"].zero_()
    qsa.run(binding, **initial)

    output = qsa.run(
        binding,
        query=torch.zeros_like(initial["query"]),
        request_ids=initial["request_ids"],
        query_positions=torch.tensor([4], dtype=torch.int64, device=device),
        reuse=qsa.DraftSelectionReuse(
            torch.tensor([0], dtype=torch.int64, device=device)
        ),
    )

    assert binding._draft_work_positions[0, caps.selection_width].item() == 4
    torch.testing.assert_close(
        output,
        torch.full_like(output, 0.2),
        rtol=0.02,
        atol=0.02,
    )


@pytest.mark.parametrize("draft", [False, True])
def test_qsa_rebind_uses_caller_events_during_capture(monkeypatch, draft):
    """Fresh bindings may not construct synchronization resources inside capture."""
    device = require_sm120()
    binding = _allocate_binding(
        _caps(
            device,
            q_heads=24,
            head_dim=256,
            index_heads=4,
            index_head_dim=128,
            index_rotary_dim=64,
            max_speculative_tokens=3 if draft else 0,
        )
    )
    stream = torch.cuda.Stream(device=device)
    done = torch.cuda.Event()
    done.record(stream)
    ready = torch.cuda.Event()
    dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.fill_(3)
    binding.raw_k_ring.zero_()
    binding.compressed_k_cache.zero_()
    state = _draft_selection_state(binding) if draft else None
    warmed = _rebind(
        binding, selection_stream=stream, selection_done=done, draft_selection=state
    )
    before_event = torch.cuda.Event

    def forbid_event(*args, **kwargs):
        raise AssertionError("fresh QSA binding allocated a CUDA event")

    # Keep isinstance checks valid while failing any attempted construction.
    monkeypatch.setattr(before_event, "__new__", forbid_event)
    graph = _cuda_graph()
    with torch.cuda.graph(graph):
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        capture_state = state.plan.bind(storage=state._storage) if state else None
        rebound = _rebind(
            binding,
            selection_stream=stream,
            selection_done=done,
            draft_selection=capture_state,
        )
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        if capture_state is not None:
            capture_state.reset()
        assert rebound is not warmed
        assert rebound._selection_done is done
        ready.record()
        result = qsa.run(rebound, **dynamic, index_ready=ready)
    graph.replay()
    assert torch.all(result == 3)


def test_qsa_run_stream_resources_use_plan_device() -> None:
    device = require_sm120()
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two explicitly assigned CUDA devices")
    device = torch.device("cuda", torch.cuda.current_device())
    other = (device.index + 1) % torch.cuda.device_count()
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        q_heads=24,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
    )
    binding = _allocate_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.main_k_cache.zero_()
    binding.main_v_cache.fill_(2)
    binding.compressed_k_cache.zero_()
    binding.raw_k_ring.zero_()
    dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
    with torch.cuda.device(other):
        wrong_stream = torch.cuda.Stream()
        wrong_ready = torch.cuda.Event()
        wrong_ready.record()
    with pytest.raises(ValueError, match="selection_stream must match"):
        _rebind(binding, selection_stream=wrong_stream)
    binding = _rebind(binding, selection_stream=torch.cuda.Stream(device=device))
    with pytest.raises(ValueError, match="index_ready must be recorded on"):
        qsa.run(binding, **dynamic, index_ready=wrong_ready)
    with torch.cuda.device(other):
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
        result = qsa.run(binding, **dynamic, index_ready=ready)
        assert torch.cuda.current_device() == other
    assert torch.all(result == 2)


def test_qsa_draft_storage_binding_is_caller_owned_and_preserves_contents():
    device = require_sm120()
    caps = _caps(
        device,
        q_heads=24,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_rotary_dim=64,
        max_speculative_tokens=3,
    )
    binding = _allocate_binding(caps)
    storage_plan = qsa.draft_selection_plan(binding.plan, max_source_rows=8)
    (spec,) = storage_plan.storage_specs()
    storage = torch.full(spec.shape, 37, dtype=spec.dtype, device=spec.device)
    expected = storage.clone()
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    state = storage_plan.bind(storage=storage)
    rebound = _rebind(binding, draft_selection=state)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    assert torch.equal(storage, expected)
    assert rebound.draft_selection is state
    assert state.selected_positions.shape == (8, caps.selection_width)
    for tensor in (
        state.selected_positions,
        state.logical_positions,
        state.num_source_rows,
    ):
        assert tensor.untyped_storage().data_ptr() == storage.data_ptr()
    assert (
        rebound._draft_work_positions.untyped_storage().data_ptr()
        == binding.scratch.data_ptr()
    )
    with pytest.raises(ValueError, match="requires"):
        storage_plan.bind(storage=storage[:1])
    with pytest.raises(ValueError, match="cover planned query rows"):
        qsa.draft_selection_plan(binding.plan, max_source_rows=1)
    for aliased_storage in (binding.main_k_cache.view(torch.uint8), binding.scratch):
        with pytest.raises(ValueError, match="overlap"):
            _rebind(binding, draft_selection=storage_plan.bind(storage=aliased_storage))
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    state.reset()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    assert state.num_source_rows.item() == 0


def test_qsa_draft_reuse_validates_live_input_contract_before_mutation():
    device = require_sm120()
    binding = _allocate_binding(
        _caps(
            device,
            q_heads=24,
            head_dim=256,
            index_heads=4,
            index_head_dim=128,
            index_rotary_dim=64,
            max_speculative_tokens=3,
        )
    )
    state = _draft_selection_state(binding)
    binding = _rebind(binding, draft_selection=state)
    dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
    common = {
        name: dynamic[name] for name in ("query", "request_ids", "query_positions")
    }
    with pytest.raises(TypeError, match="DraftSelectionReuse"):
        qsa.run(binding, **common, reuse=True)
    with pytest.raises(ValueError, match="shape"):
        qsa.run(
            binding,
            **common,
            reuse=qsa.DraftSelectionReuse(
                dynamic["request_ids"][:1],
            ),
        )
    source_rows = torch.zeros(
        binding.state.caps.max_batch, dtype=torch.int64, device=device
    )
    with pytest.raises(ValueError, match="does not accept selector inputs"):
        qsa.run(binding, **dynamic, reuse=qsa.DraftSelectionReuse(source_rows))
    aliased_rows = binding.scratch[: 8 * binding.state.caps.max_batch].view(torch.int64)
    with pytest.raises(ValueError, match="overlap"):
        qsa.run(binding, **common, reuse=qsa.DraftSelectionReuse(aliased_rows))
    assert state.num_source_rows.item() == 0
