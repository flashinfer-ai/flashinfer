"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import gc
import pytest
import torch

try:
    import cutlass.cute as cute

    if not hasattr(cute, "FastDivmodDivisorV2") and hasattr(cute, "FastDivmodDivisor"):
        cute.FastDivmodDivisorV2 = cute.FastDivmodDivisor
except ImportError:
    pass

from flashinfer import BatchPrefillWithPagedKVCacheWrapper


def _make_batch(
    batch,
    q_len,
    kv_len,
    page_size,
    num_pages,
    num_qo_heads,
    head_dim,
    dev,
    page_offset=0,
):
    pages_per_req = (kv_len + page_size - 1) // page_size
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32) * q_len
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32) * pages_per_req
    kv_indices = (
        torch.arange(batch * pages_per_req, dtype=torch.int32) + page_offset
    ) % num_pages
    last_page_len = torch.full(
        (batch,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )
    q = torch.randn(
        batch * q_len, num_qo_heads, head_dim, dtype=torch.float16, device=dev
    )
    return q, (qo_indptr, kv_indptr, kv_indices.to(dev), last_page_len)


@pytest.mark.parametrize("backend", ["fa2"])
def test_plan_async_double_buffer_overlap(backend):
    dev = torch.device("cuda:0")
    num_qo_heads, num_kv_heads, head_dim, page_size = 16, 8, 128, 16
    num_pages = 2048
    kv_cache = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=dev
    )

    q_a, meta_a = _make_batch(
        batch=8,
        q_len=64,
        kv_len=1024,
        page_size=page_size,
        num_pages=num_pages,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        dev=dev,
        page_offset=0,
    )
    _, meta_b = _make_batch(
        batch=1,
        q_len=1,
        kv_len=16,
        page_size=page_size,
        num_pages=num_pages,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        dev=dev,
        page_offset=7,
    )

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend=backend)

    # Reference computation (isolated, synchronized)
    wrapper.plan(
        *meta_a,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.float16,
    )
    ref = wrapper.run(q_a, kv_cache)
    torch.cuda.synchronize()

    # In-flight overlap: subsequent plan() must not corrupt the pending copy.
    # A sleep kernel keeps the stream busy while plan(meta_a) queues its async H2D copy
    # and run(q_a) queues the kernel. The CPU immediately proceeds to plan(meta_b) without
    # synchronizing; double-buffering ensures meta_b does not overwrite meta_a's buffer.
    for _ in range(3):
        torch.cuda._sleep(200_000_000)
        wrapper.plan(
            *meta_a,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.float16,
        )
        out = wrapper.run(q_a, kv_cache)
        wrapper.plan(
            *meta_b,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.float16,
        )
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_plan_wrapper_deallocation_overlap():
    dev = torch.device("cuda:0")
    num_qo_heads, num_kv_heads, head_dim, page_size = 16, 8, 128, 16
    num_pages = 2048
    kv_cache = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=dev
    )

    q_a, meta_a = _make_batch(
        batch=8,
        q_len=64,
        kv_len=1024,
        page_size=page_size,
        num_pages=num_pages,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        dev=dev,
        page_offset=0,
    )
    _, meta_b = _make_batch(
        batch=1,
        q_len=1,
        kv_len=16,
        page_size=page_size,
        num_pages=num_pages,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        dev=dev,
        page_offset=7,
    )

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    w.plan(
        *meta_a,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.float16,
    )
    ref = w.run(q_a, kv_cache)
    torch.cuda.synchronize()
    del w

    # Overlapping plan followed immediately by wrapper deletion:
    # __del__ must synchronize in-flight copy before pinned buffers are released.
    w = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    torch.cuda._sleep(200_000_000)
    w.plan(
        *meta_a,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.float16,
    )
    out = w.run(q_a, kv_cache)
    del w
    gc.collect()

    w_new = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    w_new.plan(
        *meta_b,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.float16,
    )
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_wrapper_double_buffers():
    dev = torch.device("cuda:0")
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    # Double buffers are pre-allocated pinned CPU memory
    assert len(wrapper._pin_memory_int_workspace_buffers) == 2
    assert len(wrapper._pin_memory_copy_events) == 2
    for buf in wrapper._pin_memory_int_workspace_buffers:
        assert isinstance(buf, torch.Tensor)
        assert buf.device.type == "cpu"
        assert buf.is_pinned()

    # Active buffer points to slot 0 initially
    assert (
        wrapper._pin_memory_int_workspace_buffer
        is wrapper._pin_memory_int_workspace_buffers[0]
    )


if __name__ == "__main__":
    test_plan_async_double_buffer_overlap("fa2")
    test_plan_wrapper_deallocation_overlap()
    test_wrapper_double_buffers()
    print("All tests in test_batch_prefill_plan_race.py passed!")
