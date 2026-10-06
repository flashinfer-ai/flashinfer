"""Windowed dense-MLA correctness gates."""

from __future__ import annotations

import pytest
import torch

from b12x.attention import dense_mla

from .test_dense_mla import _bind, _prepared_scope
from ..conftest import require_b12x

FP8 = torch.float8_e4m3fn
WINDOW_SIZE = 513
SM_SCALE = 1.0 / (256**0.5)




@pytest.fixture(autouse=True)
def _prepared_dense_mla_window():
    with _prepared_scope():
        yield

@pytest.mark.parametrize(
    "cache_lens,query_lens",
    [
        ([1, 63, 64, 65], [1, 1, 1, 1]),
        ([512, 513, 514], [1, 2, 4]),
        ([2048, 2049], [2, 4]),
        ([65], [65]),
    ],
)
@torch.inference_mode()
def test_fp8_ragged_window_matches_reference(
    cache_lens: list[int], query_lens: list[int]
) -> None:
    device = require_b12x()
    torch.manual_seed(20260813 + max(cache_lens))
    batch = len(cache_lens)
    total_q = sum(query_lens)
    max_seq_len = max(cache_lens)
    table_width = (max_seq_len + 63) // 64
    num_pages = batch * table_width + 7
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            dtype=torch.bfloat16,
            q_dtype=torch.bfloat16,
            kv_dtype=FP8,
            num_q_heads=8,
            page_size=64,
            max_total_q=total_q,
            max_batch=batch,
            max_cache_tokens=max_seq_len,
            max_page_table_width=table_width,
            num_cache_pages=num_pages,
            head_dim=1088,
            v_head_dim=1024,
            physical_record_width=1088,
            window_size=WINDOW_SIZE,
        )
    )
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    q = (torch.randn(total_q, 8, 1088, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(num_pages, 64, 1088, device=device) * 10).to(FP8)
    page_table = torch.arange(
        batch * table_width, dtype=torch.int32, device=device
    ).reshape(batch, table_width)
    page_table = torch.flip(page_table, dims=(1,))
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(total_q, 8, 1024, dtype=torch.bfloat16, device=device)
    binding = _bind(
        plan,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
        sm_scale=SM_SCALE,
    )
    actual, actual_lse = dense_mla.run(binding=binding)
    expected, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
        sm_scale=SM_SCALE,
        v_head_dim=1024,
        window_size=WINDOW_SIZE,
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual).all().item())
    assert bool(torch.isfinite(actual_lse).all().item())
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-5, atol=2e-5)


@torch.inference_mode()
def test_fp8_decode_replays_on_non_default_stream_without_allocation() -> None:
    device = require_b12x()
    torch.manual_seed(20260814)
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            dtype=torch.bfloat16,
            q_dtype=torch.bfloat16,
            kv_dtype=FP8,
            num_q_heads=8,
            page_size=64,
            max_total_q=4,
            max_batch=1,
            max_cache_tokens=514,
            max_page_table_width=9,
            num_cache_pages=9,
            head_dim=1088,
            v_head_dim=1024,
            physical_record_width=1088,
            window_size=WINDOW_SIZE,
            use_cuda_graph=True,
        )
    )
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    q = (torch.randn(4, 8, 1088, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(9, 64, 1088, device=device) * 10).to(FP8)
    page_table = torch.arange(9, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([514], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 4], dtype=torch.int32, device=device)
    output = torch.empty(4, 8, 1024, dtype=torch.bfloat16, device=device)
    binding = _bind(
        plan,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
        sm_scale=SM_SCALE,
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        dense_mla.run(binding=binding)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    assert captured_output.data_ptr() == output.data_ptr()

    for seed in (20260815, 20260816):
        generator = torch.Generator(device=device).manual_seed(seed)
        q.copy_(
            (torch.randn(q.shape, generator=generator, device=device) * 0.1).to(
                torch.bfloat16
            )
        )
        expected, expected_lse = dense_mla.reference(
            q,
            cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            kv_scale=kv_scale,
            q_scale=q_scale,
            sm_scale=SM_SCALE,
            v_head_dim=1024,
            window_size=WINDOW_SIZE,
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
def test_fp8_page_offset_exceeds_signed_int32() -> None:
    device = require_b12x()
    torch.manual_seed(20260817)
    page_stride_bytes = 64 * 1088
    int32_max = torch.iinfo(torch.int32).max
    high_page_id = int32_max // page_stride_bytes + 2
    num_pages = high_page_id + 1
    caps = dense_mla.Caps(
        device=device,
        mode="decode",
        dtype=torch.bfloat16,
        q_dtype=torch.bfloat16,
        kv_dtype=FP8,
        num_q_heads=8,
        page_size=64,
        max_total_q=1,
        max_batch=1,
        max_cache_tokens=64,
        max_page_table_width=1,
        num_cache_pages=num_pages,
        head_dim=1088,
        v_head_dim=1024,
        physical_record_width=1088,
        window_size=WINDOW_SIZE,
    )
    cache = torch.empty(num_pages, 64, 1088, dtype=FP8, device=device)
    assert high_page_id * cache.stride(0) * cache.element_size() > int32_max
    cache[high_page_id].copy_((torch.randn(64, 1088, device=device) * 10).to(FP8))
    q = (torch.randn(1, 8, 1088, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor([[high_page_id]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    output = torch.empty(1, 8, 1024, dtype=torch.bfloat16, device=device)
    plan = dense_mla.plan(
        caps,
        invocation=dense_mla.invocation_from_tensors(
            caps,
            q=q,
            kv_cache=cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            kv_scale=kv_scale,
            q_scale=q_scale,
        ),
    )
    binding = _bind(
        plan,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
        sm_scale=SM_SCALE,
    )
    actual, actual_lse = dense_mla.run(binding=binding)
    expected, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
        sm_scale=SM_SCALE,
        v_head_dim=1024,
        window_size=WINDOW_SIZE,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-5, atol=2e-5)
