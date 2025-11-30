"""
Copyright (c) 2025 by FlashInfer team.

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

import math

import pytest
import torch
from tests.test_helpers.sink_attention_reference import sink_attention_unified

import flashinfer
from flashinfer.utils import has_flashinfer_jit_cache


def sink_attention_decode_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    sm_scale: float,
) -> torch.Tensor:
    """Reference implementation for decode mode sink attention."""
    return sink_attention_unified(
        q,
        k_cache,
        v_cache,
        sink,
        window_left,
        causal=True,
        sm_scale=sm_scale,
        mode="incremental",
    )


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    """Warmup JIT cache for decode kernels."""
    # This will be built on-demand during tests
    yield


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("kv_len", [32, 128, 512])
@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads",
    [
        (8, 8),  # MHA: equal heads
        (32, 8),  # GQA: 4:1 ratio
        (32, 32),  # MHA: equal heads
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("window_left", [-1])  # Only test without sliding window
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("kv_layout", ["NHD"])
def test_batch_decode_with_sink_attention(
    batch_size,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    window_left,
    page_size,
    kv_layout,
):
    """Test batch decode with sink attention support."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    sm_scale = 1.0 / math.sqrt(head_dim)

    # Create query tensor: [batch_size, num_qo_heads, head_dim]
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)

    # Create KV cache in paged format
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    if kv_layout == "NHD":
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    else:
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]

    kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device=device)
    kv_data = kv_data_fp32.to(dtype)

    # Create page indices and metadata
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    # Create sink tensor: [num_qo_heads] float32
    # Sink values should be on similar scale to logits (QK^T * sm_scale)
    # For typical logits, use smaller range to match expected scale
    sinks = torch.randn(num_qo_heads, device=device, dtype=torch.float32) * 0.5

    # Create workspace buffer
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    # Test with FlashInfer
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=dtype,
        q_data_type=dtype,
        sm_scale=sm_scale,
        window_left=window_left,
    )

    out = wrapper.run(q, kv_data, sinks=sinks)

    # Create reference implementation
    # Convert paged KV cache to regular format for reference
    k_cache_ref = torch.zeros(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache_ref = torch.zeros(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    for b in range(batch_size):
        page_start = b * num_pages_per_seq
        for p in range(num_pages_per_seq):
            page_idx = page_start + p
            token_start = p * page_size
            token_end = min(token_start + page_size, kv_len)
            actual_page_len = token_end - token_start

            if kv_layout == "NHD":
                k_cache_ref[b, token_start:token_end] = kv_data_fp32[
                    page_idx, 0, :actual_page_len
                ].to(dtype)
                v_cache_ref[b, token_start:token_end] = kv_data_fp32[
                    page_idx, 1, :actual_page_len
                ].to(dtype)
            else:
                k_cache_ref[b, token_start:token_end] = (
                    kv_data_fp32[page_idx, 0, :, :actual_page_len]
                    .transpose(0, 1)
                    .to(dtype)
                )
                v_cache_ref[b, token_start:token_end] = (
                    kv_data_fp32[page_idx, 1, :, :actual_page_len]
                    .transpose(0, 1)
                    .to(dtype)
                )

    # Compute reference output
    out_ref = sink_attention_decode_ref(
        q, k_cache_ref, v_cache_ref, sinks, window_left, sm_scale
    )

    # Compare results
    # bfloat16 may have slightly larger numerical differences due to lower precision,
    # differences in order of operations between reference and CUDA kernel, and
    # GQA scenarios where multiple query heads share KV heads
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=3.5e-2)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("kv_len", [128])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_decode_without_sink_attention(
    batch_size, kv_len, num_qo_heads, num_kv_heads, head_dim
):
    """Test that decode without sinks matches decode with zero sinks."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    sm_scale = 1.0 / math.sqrt(head_dim)
    page_size = 16
    kv_layout = "NHD"

    # Create query tensor
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)

    # Create KV cache
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data = torch.randn(*kv_shape, dtype=dtype, device=device)

    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=dtype,
        q_data_type=dtype,
        sm_scale=sm_scale,
    )

    # Test without sinks
    out_no_sinks = wrapper.run(q, kv_data, sinks=None)

    # Test with zero sinks (should match no sinks)
    zero_sinks = torch.zeros(num_qo_heads, device=device, dtype=torch.float32)
    out_zero_sinks = wrapper.run(q, kv_data, sinks=zero_sinks)

    # Results should be very close (zero sinks should be equivalent to no sinks)
    # Note: Even when skipping zero sinks, there may be small numerical differences
    # due to code path differences and floating point precision
    # bfloat16 has lower precision, so allow slightly larger tolerance
    torch.testing.assert_close(out_no_sinks, out_zero_sinks, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("kv_len", [64])
@pytest.mark.parametrize("num_qo_heads", [16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
def test_batch_decode_sink_attention_gqa(
    batch_size, kv_len, num_qo_heads, num_kv_heads, head_dim
):
    """Test sink attention with grouped query attention (GQA)."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    sm_scale = 1.0 / math.sqrt(head_dim)
    page_size = 16
    kv_layout = "NHD"

    # Create query tensor with more heads than KV
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)

    # Create KV cache with fewer heads
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data = torch.randn(*kv_shape, dtype=dtype, device=device)

    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    # Sink tensor should have num_qo_heads elements
    sinks = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5.0

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=dtype,
        q_data_type=dtype,
        sm_scale=sm_scale,
    )

    # This should work with GQA
    out = wrapper.run(q, kv_data, sinks=sinks)

    # Basic sanity check: output should have correct shape
    assert out.shape == (batch_size, num_qo_heads, head_dim)
    assert out.dtype == dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("kv_len", [32, 128, 512])
@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads",
    [
        (8, 8),  # MHA: equal heads
        (16, 8),  # GQA: 2:1 ratio
        (32, 8),  # GQA: 4:1 ratio
        (32, 32),  # MHA: equal heads
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_single_decode_sink_attention_tensor_cores(
    kv_len, num_qo_heads, num_kv_heads, head_dim, kv_layout
):
    """Test sink attention with single decode using tensor cores (prefill template)."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    sm_scale = 1.0 / math.sqrt(head_dim)
    window_left = -1  # No sliding window

    # Create query tensor
    q = torch.randn(num_qo_heads, head_dim, dtype=dtype, device=device)

    # Create KV cache based on layout
    if kv_layout == "NHD":
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    else:  # HND
        k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=dtype, device=device)
        v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=dtype, device=device)

    # Sink tensor should have num_qo_heads elements
    # Sink values should be on similar scale to logits (QK^T * sm_scale)
    sinks = torch.randn(num_qo_heads, device=device, dtype=torch.float32) * 0.5

    # Test with tensor cores enabled (uses prefill template)
    out = flashinfer.single_decode_with_kv_cache(
        q,
        k,
        v,
        kv_layout=kv_layout,
        pos_encoding_mode="NONE",
        use_tensor_cores=True,
        sm_scale=sm_scale,
        sinks=sinks,
    )

    # Basic sanity check: output should have correct shape
    assert out.shape == (num_qo_heads, head_dim)
    assert out.dtype == dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Validate against reference implementation
    # Convert to batch format for reference (add batch dimension)
    q_batch = q.unsqueeze(0)  # [1, num_qo_heads, head_dim]

    # Convert KV cache to reference format [batch_size, kv_len, num_kv_heads, head_dim]
    if kv_layout == "NHD":
        k_cache_ref = k.unsqueeze(0)  # [1, kv_len, num_kv_heads, head_dim]
        v_cache_ref = v.unsqueeze(0)  # [1, kv_len, num_kv_heads, head_dim]
    else:  # HND -> transpose to NHD
        k_cache_ref = k.transpose(0, 1).unsqueeze(
            0
        )  # [1, kv_len, num_kv_heads, head_dim]
        v_cache_ref = v.transpose(0, 1).unsqueeze(
            0
        )  # [1, kv_len, num_kv_heads, head_dim]

    # Compute reference output
    out_ref = sink_attention_decode_ref(
        q_batch, k_cache_ref, v_cache_ref, sinks, window_left, sm_scale
    )

    # Remove batch dimension from reference output
    out_ref = out_ref.squeeze(0)  # [num_qo_heads, head_dim]

    # Compare results
    # bfloat16 may have slightly larger numerical differences due to lower precision,
    # differences in order of operations between reference and CUDA kernel, and
    # GQA scenarios where multiple query heads share KV heads
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=3.5e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
