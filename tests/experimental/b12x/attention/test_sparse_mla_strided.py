"""Strided-record sparse MLA correctness gates."""

from __future__ import annotations

from dataclasses import replace

import torch

from b12x.attention.sparse_mla import strided as sparse_mla_strided
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x

FP8 = torch.float8_e4m3fn


def _prepared_binding(declaration, *, name: str, device, make_binding):
    """Prime a real strided binding through the preparation lifecycle."""
    prepared_binding = None

    def prepare_call(state):
        nonlocal prepared_binding
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        prepared_binding = make_binding(state, scratch)
        state.prime(prepared_binding)
        return PreparedCall(
            run=lambda: state.run(prepared_binding),
            output=prepared_binding.native.output,
            owners=(scratch, prepared_binding),
        )

    session = PreparationSession(device=device, autotune=False)
    result = session.prepare((
        declaration.request(
            name=name, prepare_call=prepare_call
        ),
    ))
    return result, replace(prepared_binding, plan=declaration)


def test_is_supported_accepts_implicit_current_device() -> None:
    require_b12x()
    assert sparse_mla_strided.is_supported()


@torch.inference_mode()
def test_fp8_physical_slots_ignore_padding_and_mask_invalid_tail() -> None:
    device = require_b12x()
    torch.manual_seed(20260813)
    rows = 4
    blocks = 40
    plan = sparse_mla_strided.plan(
        sparse_mla_strided.Caps(
            device=device,
            num_q_heads=16,
            tp_size=8,
            max_q_rows=rows,
            num_cache_blocks=blocks,
        )
    )
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    q = (torch.randn(rows, 16, 576, device=device) * 0.1).to(torch.bfloat16)
    cache = torch.empty(blocks, 64, 1088, dtype=FP8, device=device)
    cache[..., :576] = (torch.randn(blocks, 64, 576, device=device) * 10).to(FP8)
    cache[..., 576:] = (
        (torch.randn(blocks, 64, 512, device=device) * 100).clamp(-448, 448).to(FP8)
    )
    selected = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    counts = torch.tensor([1, 64, 513, 2048], dtype=torch.int32, device=device)
    for row, count in enumerate(counts.tolist()):
        selected[row, :count] = torch.randperm(
            blocks * 64, dtype=torch.int32, device=device
        )[:count]
    cu_seqlens_q = torch.arange(rows + 1, dtype=torch.int32, device=device)
    output = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).repeat(rows, 1)
    request_ids = torch.arange(rows, dtype=torch.int32, device=device)
    result, binding = _prepared_binding(
        plan, name="physical-slots", device=device,
        make_binding=lambda state, scratch: state.bind_indexed(
            scratch=scratch, q=q, kv_cache=cache, output=output,
            logical_indices=selected, request_ids=request_ids, block_table=block_table,
            cu_seqlens_q=cu_seqlens_q, kv_scale=kv_scale, q_scale=q_scale,
        ),
    )
    actual, actual_lse = sparse_mla_strided.run_decode(binding=binding)
    expected, expected_lse = sparse_mla_strided.reference(
        q,
        cache,
        selected,
        counts,
        kv_scale=kv_scale,
        q_scale=q_scale,
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual).all().item())
    assert bool(torch.isfinite(actual_lse).all().item())
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-5, atol=2e-5)


@torch.inference_mode()
def test_request_relative_indices_are_stably_compacted_and_remapped() -> None:
    device = require_b12x()
    torch.manual_seed(20260818)
    rows = 2
    blocks = 12
    plan = sparse_mla_strided.plan(
        sparse_mla_strided.Caps(
            device=device,
            num_q_heads=16,
            tp_size=8,
            max_q_rows=rows,
            num_cache_blocks=blocks,
        )
    )
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    q = (torch.randn(rows, 16, 576, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(blocks, 64, 1088, device=device) * 10).to(FP8)
    block_table = torch.tensor(
        [[7, 2, 10, -1], [4, 11, 1, 8]], dtype=torch.int32, device=device
    )
    request_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    logical = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    logical[0, :8] = torch.tensor(
        [65, -1, 3, 130, 999, 64, -1, 191], dtype=torch.int32, device=device
    )
    logical[1, :7] = torch.tensor(
        [255, 0, -1, 66, 129, -1, 67], dtype=torch.int32, device=device
    )
    expected_selected = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    expected_selected[0, :5] = torch.tensor(
        [2 * 64 + 1, 7 * 64 + 3, 10 * 64 + 2, 2 * 64, 10 * 64 + 63],
        dtype=torch.int32,
        device=device,
    )
    expected_selected[1, :5] = torch.tensor(
        [8 * 64 + 63, 4 * 64, 11 * 64 + 2, 1 * 64 + 1, 11 * 64 + 3],
        dtype=torch.int32,
        device=device,
    )
    expected_counts = torch.tensor([5, 5], dtype=torch.int32, device=device)
    output = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=device)
    result, binding = _prepared_binding(
        plan, name="request-relative", device=device,
        make_binding=lambda state, scratch: state.bind_indexed(
            scratch=scratch, q=q, kv_cache=cache, output=output,
            logical_indices=logical, request_ids=request_ids, block_table=block_table,
            cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device=device),
            kv_scale=kv_scale, q_scale=q_scale,
        ),
    )
    actual, actual_lse = sparse_mla_strided.run_extend(binding=binding)
    expected, expected_lse = sparse_mla_strided.reference(
        q,
        cache,
        expected_selected,
        expected_counts,
        kv_scale=kv_scale,
        q_scale=q_scale,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(binding.selected_counts, expected_counts)
    torch.testing.assert_close(
        binding.selected_indices[:, :5], expected_selected[:, :5]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-5, atol=2e-5)


@torch.inference_mode()
def test_request_relative_indices_address_layer_interleaved_records() -> None:
    device = require_b12x()
    torch.manual_seed(20260819)
    rows = 1
    blocks = 12
    layers = 3
    layer = 1
    plan = sparse_mla_strided.plan(
        sparse_mla_strided.Caps(
            device=device,
            num_q_heads=16,
            tp_size=8,
            max_q_rows=rows,
            num_cache_blocks=blocks,
            max_physical_records=blocks * layers * 64,
        )
    )
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    q = (torch.randn(rows, 16, 576, device=device) * 0.1).to(torch.bfloat16)
    backing = (torch.randn(blocks, layers, 64, 1088, device=device) * 10).to(FP8)
    cache = backing[:, layer]
    assert cache.stride() == (layers * 64 * 1088, 1088, 1)
    logical = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    selected = torch.tensor(
        [0, 63, 64, 65, 5 * 64 + 2], dtype=torch.int32, device=device
    )
    logical[0, : selected.numel()] = selected
    counts = torch.tensor([selected.numel()], dtype=torch.int32, device=device)
    output = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=device)
    result, binding = _prepared_binding(
        plan, name="interleaved", device=device,
        make_binding=lambda state, scratch: state.bind_indexed(
            scratch=scratch, q=q, kv_cache=cache, output=output,
            logical_indices=logical,
            request_ids=torch.zeros(rows, dtype=torch.int32, device=device),
            block_table=torch.arange(blocks, dtype=torch.int32, device=device)[None],
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
            kv_scale=kv_scale, q_scale=q_scale,
        ),
    )
    actual, actual_lse = sparse_mla_strided.run_decode(binding=binding)
    expected, expected_lse = sparse_mla_strided.reference(
        q,
        cache,
        logical,
        counts,
        kv_scale=kv_scale,
        q_scale=q_scale,
    )
    expected_records = torch.div(
        selected, 64, rounding_mode="floor"
    ) * layers * 64 + selected.remainder(64)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(binding.selected_counts, counts)
    torch.testing.assert_close(
        binding.selected_indices[0, : selected.numel()], expected_records
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-5, atol=2e-5)


@torch.inference_mode()
def test_fp8_sparse_replays_on_non_default_stream_without_allocation() -> None:
    device = require_b12x()
    torch.manual_seed(20260814)
    rows = 2
    blocks = 32
    layers = 3
    plan = sparse_mla_strided.plan(
        sparse_mla_strided.Caps(
            device=device,
            num_q_heads=16,
            tp_size=8,
            max_q_rows=rows,
            num_cache_blocks=blocks,
            max_physical_records=blocks * layers * 64,
            use_cuda_graph=True,
        )
    )
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    q = (torch.randn(rows, 16, 576, device=device) * 0.1).to(torch.bfloat16)
    backing = (torch.randn(blocks, layers, 64, 1088, device=device) * 10).to(FP8)
    cache = backing[:, 1]
    selected = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    counts = torch.tensor([64, 513], dtype=torch.int32, device=device)
    selected[0, :64] = torch.randperm(blocks * 64, device=device)[:64].to(torch.int32)
    selected[1, :513] = torch.randperm(blocks * 64, device=device)[:513].to(torch.int32)
    cu_seqlens_q = torch.arange(rows + 1, dtype=torch.int32, device=device)
    output = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).repeat(rows, 1)
    request_ids = torch.arange(rows, dtype=torch.int32, device=device)
    result, binding = _prepared_binding(
        plan, name="replay", device=device,
        make_binding=lambda state, scratch: state.bind_indexed(
            scratch=scratch, q=q, kv_cache=cache, output=output,
            logical_indices=selected, request_ids=request_ids, block_table=block_table,
            cu_seqlens_q=cu_seqlens_q, kv_scale=kv_scale, q_scale=q_scale,
        ),
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        sparse_mla_strided.run_decode(binding=binding)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_output, captured_lse = sparse_mla_strided.run_decode(binding=binding)
    assert captured_output.data_ptr() == output.data_ptr()

    for seed in (20260815, 20260816):
        generator = torch.Generator(device=device).manual_seed(seed)
        q.copy_(
            (torch.randn(q.shape, generator=generator, device=device) * 0.1).to(
                torch.bfloat16
            )
        )
        expected, expected_lse = sparse_mla_strided.reference(
            q,
            cache,
            selected,
            counts,
            kv_scale=kv_scale,
            q_scale=q_scale,
        )
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        graph.replay()
        stream.synchronize()
        assert torch.cuda.memory_allocated(device) == allocated
        assert torch.cuda.memory_reserved(device) == reserved
        torch.testing.assert_close(captured_output, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(captured_lse, expected_lse, rtol=2e-5, atol=2e-5)


@torch.inference_mode()
def test_fp8_physical_slot_offset_exceeds_signed_int32() -> None:
    device = require_b12x()
    torch.manual_seed(20260817)
    record_stride_bytes = 1088
    int32_max = torch.iinfo(torch.int32).max
    high_slot = int32_max // record_stride_bytes + 2
    blocks = high_slot // 64 + 1
    plan = sparse_mla_strided.plan(
        sparse_mla_strided.Caps(
            device=device,
            num_q_heads=16,
            tp_size=8,
            max_q_rows=1,
            num_cache_blocks=blocks,
        )
    )
    cache = torch.empty(blocks, 64, 1088, dtype=FP8, device=device)
    assert high_slot * cache.stride(1) * cache.element_size() > int32_max
    flat_cache = cache.view(-1, 1088)
    flat_cache[high_slot].copy_((torch.randn(1088, device=device) * 10).to(FP8))
    q = (torch.randn(1, 16, 576, device=device) * 0.1).to(torch.bfloat16)
    selected = torch.full((1, 2048), -1, dtype=torch.int32, device=device)
    selected[0, 0] = high_slot
    counts = torch.tensor([1], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    output = torch.empty(1, 16, 512, dtype=torch.bfloat16, device=device)
    result, binding = _prepared_binding(
        plan, name="high-pid", device=device,
        make_binding=lambda state, scratch: state.bind(
            scratch=scratch, q=q, kv_cache=cache, output=output,
            selected_indices=selected, selected_counts=counts,
            cu_seqlens_q=cu_seqlens_q, kv_scale=kv_scale, q_scale=q_scale,
        ),
    )
    actual, actual_lse = sparse_mla_strided.run_decode(binding=binding)
    expected, expected_lse = sparse_mla_strided.reference(
        q,
        cache,
        selected,
        counts,
        kv_scale=kv_scale,
        q_scale=q_scale,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-5, atol=2e-5)
