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
from sink_attention_reference import sink_attention_unified

import flashinfer
from flashinfer.jit.utils import filename_safe_dtype_map
from flashinfer.jit.attention import gen_batch_prefill_attention_sink_module
from flashinfer.jit.attention.variants import attention_sink_decl
from flashinfer.utils import is_sm90a_supported


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    jit_specs = []
    for dtype in [torch.float16, torch.bfloat16]:
        for backend in ["fa2", "fa3"]:
            for use_swa in [True, False]:
                for head_dim in [128]:
                    jit_specs.append(
                        gen_batch_prefill_attention_sink_module(
                            backend=backend,
                            dtype_q=dtype,
                            dtype_kv=dtype,
                            dtype_o=dtype,
                            dtype_idx=torch.int32,
                            head_dim_qk=head_dim,
                            head_dim_vo=head_dim,
                            pos_encoding_mode=0,
                            use_sliding_window=use_swa,
                        )
                    )

    flashinfer.jit.build_jit_specs(jit_specs)
    yield


# Wrapper functions for backward compatibility
def sink_attention_ref(
    batch_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Backward compatible wrapper for prefill mode."""
    return sink_attention_unified(
        q,
        k,
        v,
        sink,
        window_left,
        causal,
        sm_scale,
        batch_size=batch_size,
        mode="prefill",
    )


def sink_attention_incremental_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Backward compatible wrapper for incremental mode."""
    return sink_attention_unified(
        q, k_cache, v_cache, sink, window_left, causal, sm_scale, mode="incremental"
    )


def sink_attention_chunk_ref(
    batch_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Wrapper for chunk prefill mode."""
    return sink_attention_unified(
        q,
        k,
        v,
        sink,
        window_left,
        causal,
        sm_scale,
        batch_size=batch_size,
        mode="chunk",
    )


def sink_attention_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
) -> torch.Tensor:
    """Wrapper for variable length sequences mode."""
    return sink_attention_unified(
        q,
        k,
        v,
        sink,
        window_left,
        causal,
        sm_scale,
        mode="varlen",
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [1, 4, 16, 128])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_sink(
    dtype, batch_size, seq_len, num_qo_heads, num_kv_heads, window_left, causal, backend
):
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    jit_args = (
        f"batch_prefill_attention_sink_{filename_safe_dtype_map[dtype]}_swa_{window_left >= 0}_{backend}",  # uri
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        torch.int32,  # idtype
        128,  # hidden_dim_qk
        128,  # hidden_dim_vo
        ["sink"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["double"],  # additional_scalar_dtypes
        "AttentionSink",
        attention_sink_decl[backend],
    )
    jit_kwargs = {
        "use_sliding_window": window_left >= 0,
    }
    sm_scale = 1.0 / math.sqrt(128)
    torch.manual_seed(42)
    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend=backend,
        jit_args=jit_args,
        jit_kwargs=jit_kwargs,
    )
    qo_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )

    head_dim = 128

    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    q = torch.randn(
        batch_size * seq_len,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        batch_size * seq_len,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        batch_size * seq_len,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    o = wrapper.run(q, k, v, sink, sm_scale)
    o_ref = sink_attention_ref(batch_size, q, k, v, sink, window_left, causal, sm_scale)
    if dtype == torch.float16:
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend=backend,
        jit_args=jit_args,
        jit_kwargs=jit_kwargs,
    )
    kv_indices_host = torch.arange(
        0,
        batch_size * seq_len,
        dtype=torch.int32,
    )
    paged_kv_last_page_len_host = torch.full((batch_size,), 1, dtype=torch.int32)
    wrapper_paged.plan(
        qo_indptr_host,
        kv_indptr_host,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
        non_blocking=True,
    )
    o_paged = wrapper_paged.run(q, (k, v), sink, sm_scale)
    if dtype == torch.float16:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)

    # Test with non-contiguous KV indices (production scenario)
    total_pages = batch_size * seq_len
    if total_pages > 1:  # Only test fragmentation when we have multiple pages
        # Create a fragmented page allocation pattern
        import random

        random.seed(42 + total_pages)  # Deterministic but varied seed
        all_pages = list(range(0, total_pages * 2))  # Larger page pool
        occupied_pages = set(
            random.sample(all_pages, min(total_pages, len(all_pages) // 2))
        )
        available_pages = [p for p in all_pages if p not in occupied_pages]

        # Allocate non-contiguous pages
        kv_indices_fragmented = torch.tensor(
            available_pages[:total_pages], dtype=torch.int32, device=device
        )

        # Create new paged KV cache with larger capacity
        k_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Copy K,V data to fragmented pages
        for i, page_idx in enumerate(kv_indices_fragmented):
            k_paged_frag[page_idx, 0] = k[i]
            v_paged_frag[page_idx, 0] = v[i]

        # Test with fragmented indices
        wrapper_paged_frag = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer,
            kv_layout="NHD",
            backend=backend,
            jit_args=jit_args,
            jit_kwargs=jit_kwargs,
        )
        wrapper_paged_frag.plan(
            qo_indptr_host,
            kv_indptr_host,
            kv_indices_fragmented,
            paged_kv_last_page_len_host,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
            causal=causal,
            window_left=window_left,
            q_data_type=dtype,
            kv_data_type=dtype,
            non_blocking=True,
        )
        o_paged_frag = wrapper_paged_frag.run(
            q, (k_paged_frag, v_paged_frag), sink, sm_scale
        )

        # Verify fragmented result matches reference
        if dtype == torch.float16:
            torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("initial_seq_len", [32, 128])
@pytest.mark.parametrize("num_generation_steps", [1, 2, 4])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_sink_incremental_generation(
    dtype,
    batch_size,
    initial_seq_len,
    num_generation_steps,
    num_qo_heads,
    num_kv_heads,
    window_left,
    causal,
    backend,
):
    """
    Test incremental generation scenario: q_len=1, kv_len grows gradually
    Simulate the token-by-token generation process in real large model inference
    """
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)

    # Create JIT arguments
    jit_args = (
        f"batch_prefill_attention_sink_{filename_safe_dtype_map[dtype]}_swa_{window_left >= 0}_{backend}",
        dtype,
        dtype,
        dtype,
        torch.int32,
        head_dim,
        head_dim,
        ["sink"],
        ["float"],
        ["sm_scale"],
        ["double"],
        "AttentionSink",
        attention_sink_decl[backend],
    )
    jit_kwargs = {
        "use_sliding_window": window_left >= 0,
    }

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )

    # Initialize KV cache - simulate state after prefill phase
    k_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    k_accumulated = v_accumulated = None
    # Simulate incremental generation process
    for step in range(num_generation_steps):
        current_kv_len = initial_seq_len + step

        # Current generated new token (q_len=1)
        q_new = torch.randn(
            batch_size, num_qo_heads, head_dim, dtype=dtype, device=device
        )

        # K,V for newly generated token
        k_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Update KV cache
        if step == 0:
            k_cache_current = k_cache
            v_cache_current = v_cache
        else:
            k_cache_current = torch.cat([k_cache, k_accumulated], dim=1)
            v_cache_current = torch.cat([v_cache, v_accumulated], dim=1)

        # Calculate reference result
        o_ref = sink_attention_incremental_ref(
            q_new, k_cache_current, v_cache_current, sink, window_left, causal, sm_scale
        )

        # Use flashinfer to calculate result (need format conversion to adapt to existing API)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            float_workspace_buffer,
            kv_layout="NHD",
            backend=backend,
            jit_args=jit_args,
            jit_kwargs=jit_kwargs,
        )

        # Set correct indptr: q_len=1 for each batch, kv_len=current_kv_len for each batch
        qo_indptr_host = torch.arange(
            0, batch_size + 1, dtype=torch.int32
        )  # [0, 1, 2, ..., batch_size]
        kv_indptr_host = torch.arange(
            0, batch_size * current_kv_len + 1, current_kv_len, dtype=torch.int32
        )

        wrapper.plan(
            qo_indptr_host,
            kv_indptr_host,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=causal,
            window_left=window_left,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        # Convert to format expected by flashinfer [total_q_len, num_heads, head_dim]
        q_flashinfer = q_new.view(
            batch_size, num_qo_heads, head_dim
        )  # [batch_size, num_heads, head_dim]
        k_flashinfer = k_cache_current.view(
            batch_size * current_kv_len, num_kv_heads, head_dim
        )
        v_flashinfer = v_cache_current.view(
            batch_size * current_kv_len, num_kv_heads, head_dim
        )

        o = wrapper.run(q_flashinfer, k_flashinfer, v_flashinfer, sink, sm_scale)

        # Verify results
        if dtype == torch.float16:
            torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

        # Also test with BatchPrefillWithPagedKVCacheWrapper
        wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer,
            kv_layout="NHD",
            backend=backend,
            jit_args=jit_args,
            jit_kwargs=jit_kwargs,
        )
        kv_indices_host = torch.arange(
            0,
            batch_size * current_kv_len,
            dtype=torch.int32,
        )
        paged_kv_last_page_len_host = torch.full((batch_size,), 1, dtype=torch.int32)
        wrapper_paged.plan(
            qo_indptr_host,
            kv_indptr_host,
            kv_indices_host,
            paged_kv_last_page_len_host,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
            causal=causal,
            window_left=window_left,
            q_data_type=dtype,
            kv_data_type=dtype,
            non_blocking=True,
        )
        o_paged = wrapper_paged.run(
            q_flashinfer, (k_flashinfer, v_flashinfer), sink, sm_scale
        )
        if dtype == torch.float16:
            torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)

        # Test with non-contiguous KV indices for incremental generation
        total_pages = batch_size * current_kv_len
        if total_pages > 1:  # Only test fragmentation when we have multiple pages
            # Create fragmented page allocation pattern
            import random

            random.seed(42 + step + current_kv_len)  # Vary seed with step and length
            all_pages = list(range(0, total_pages * 2))
            occupied_pages = set(
                random.sample(all_pages, min(total_pages, len(all_pages) // 2))
            )
            available_pages = [p for p in all_pages if p not in occupied_pages]

            # Allocate non-contiguous pages
            kv_indices_fragmented = torch.tensor(
                available_pages[:total_pages], dtype=torch.int32, device=device
            )

            # Create fragmented paged KV cache
            k_paged_frag = torch.randn(
                total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
            )
            v_paged_frag = torch.randn(
                total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
            )

            # Copy K,V data to fragmented pages
            for i, page_idx in enumerate(kv_indices_fragmented):
                k_paged_frag[page_idx, 0] = k_flashinfer[i]
                v_paged_frag[page_idx, 0] = v_flashinfer[i]

            # Test with fragmented indices
            wrapper_paged_frag = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                float_workspace_buffer,
                kv_layout="NHD",
                backend=backend,
                jit_args=jit_args,
                jit_kwargs=jit_kwargs,
            )
            wrapper_paged_frag.plan(
                qo_indptr_host,
                kv_indptr_host,
                kv_indices_fragmented,
                paged_kv_last_page_len_host,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
                causal=causal,
                window_left=window_left,
                q_data_type=dtype,
                kv_data_type=dtype,
                non_blocking=True,
            )
            o_paged_frag = wrapper_paged_frag.run(
                q_flashinfer, (k_paged_frag, v_paged_frag), sink, sm_scale
            )

            # Verify fragmented result matches reference
            if dtype == torch.float16:
                torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-3, atol=1e-3)
            else:
                torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-2, atol=1e-2)

        # Accumulate new K,V for next step
        if step == 0:
            k_accumulated = k_new
            v_accumulated = v_new
        else:
            k_accumulated = torch.cat([k_accumulated, k_new], dim=1)
            v_accumulated = torch.cat([v_accumulated, v_new], dim=1)

        print(
            f"Step {step}: q_len=1, kv_len={current_kv_len}, both RaggedKV and PagedKV wrappers passed!"
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("historical_len", [256, 512])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_sink_chunk_prefill(
    dtype,
    batch_size,
    chunk_size,
    historical_len,
    num_qo_heads,
    num_kv_heads,
    window_left,
    causal,
    backend,
):
    """
    Test chunk prefill scenario: q_len != kv_len and q_len > 1
    Simulate chunk-based processing of long sequences where current chunk
    attends to all historical tokens plus current chunk tokens
    """
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    # Skip invalid combinations
    if chunk_size >= historical_len:
        pytest.skip(
            "chunk_size should be smaller than historical_len for meaningful chunk prefill test"
        )

    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    total_kv_len = historical_len + chunk_size

    # Create JIT arguments
    jit_args = (
        f"batch_prefill_attention_sink_{filename_safe_dtype_map[dtype]}_swa_{window_left >= 0}_{backend}",
        dtype,
        dtype,
        dtype,
        torch.int32,
        head_dim,
        head_dim,
        ["sink"],
        ["float"],
        ["sm_scale"],
        ["double"],
        "AttentionSink",
        attention_sink_decl[backend],
    )
    jit_kwargs = {
        "use_sliding_window": window_left >= 0,
    }

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )

    # Create input tensors for chunk prefill scenario
    # q represents current chunk: [batch_size * chunk_size, num_heads, head_dim]
    q_chunk = torch.randn(
        batch_size * chunk_size, num_qo_heads, head_dim, dtype=dtype, device=device
    )

    # k, v represent all tokens (historical + current chunk)
    k_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    # Calculate reference result using chunk prefill mode
    o_ref = sink_attention_chunk_ref(
        batch_size, q_chunk, k_all, v_all, sink, window_left, causal, sm_scale
    )

    # Test with flashinfer
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend=backend,
        jit_args=jit_args,
        jit_kwargs=jit_kwargs,
    )

    # Set up indices for chunk prefill
    qo_indptr_host = torch.arange(
        0, batch_size * chunk_size + 1, chunk_size, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * total_kv_len + 1, total_kv_len, dtype=torch.int32
    )

    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    o = wrapper.run(q_chunk, k_all, v_all, sink, sm_scale)

    # Verify results
    if dtype == torch.float16:
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

    # Also test with BatchPrefillWithPagedKVCacheWrapper
    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend=backend,
        jit_args=jit_args,
        jit_kwargs=jit_kwargs,
    )
    kv_indices_host = torch.arange(
        0,
        batch_size * total_kv_len,
        dtype=torch.int32,
    )
    paged_kv_last_page_len_host = torch.full((batch_size,), 1, dtype=torch.int32)
    wrapper_paged.plan(
        qo_indptr_host,
        kv_indptr_host,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
        non_blocking=True,
    )
    o_paged = wrapper_paged.run(q_chunk, (k_all, v_all), sink, sm_scale)
    if dtype == torch.float16:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)

    # Test with non-contiguous KV indices for chunk prefill
    total_pages = batch_size * total_kv_len
    if total_pages > 1:  # Only test fragmentation when we have multiple pages
        # Create fragmented page allocation pattern
        import random

        random.seed(
            42 + batch_size + total_kv_len
        )  # Vary seed with batch and total length
        all_pages = list(range(0, total_pages * 2))
        occupied_pages = set(
            random.sample(all_pages, min(total_pages, len(all_pages) // 2))
        )
        available_pages = [p for p in all_pages if p not in occupied_pages]

        # Allocate non-contiguous pages
        kv_indices_fragmented = torch.tensor(
            available_pages[:total_pages], dtype=torch.int32, device=device
        )

        # Create fragmented paged KV cache
        k_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Copy K,V data to fragmented pages
        for i, page_idx in enumerate(kv_indices_fragmented):
            k_paged_frag[page_idx, 0] = k_all[i]
            v_paged_frag[page_idx, 0] = v_all[i]

        # Test with fragmented indices
        wrapper_paged_frag = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer,
            kv_layout="NHD",
            backend=backend,
            jit_args=jit_args,
            jit_kwargs=jit_kwargs,
        )
        wrapper_paged_frag.plan(
            qo_indptr_host,
            kv_indptr_host,
            kv_indices_fragmented,
            paged_kv_last_page_len_host,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
            causal=causal,
            window_left=window_left,
            q_data_type=dtype,
            kv_data_type=dtype,
            non_blocking=True,
        )
        o_paged_frag = wrapper_paged_frag.run(
            q_chunk, (k_paged_frag, v_paged_frag), sink, sm_scale
        )

        # Verify fragmented result matches reference
        if dtype == torch.float16:
            torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-2, atol=1e-2)

    print(
        f"Chunk prefill test passed: q_len={chunk_size}, kv_len={total_kv_len}, "
        f"batch_size={batch_size}, causal={causal}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "indptr_config",
    [
        # (qo_indptr, kv_indptr, description)
        (
            [0, 32, 64, 128, 256],
            [0, 128, 256, 512, 1024],
            "4 requests: prefill-like scenarios",
        ),
        (
            [0, 1, 2, 3, 4],
            [0, 128, 256, 384, 512],
            "4 requests: incremental generation",
        ),
        ([0, 50, 150, 200], [0, 200, 600, 800], "3 requests: mixed lengths"),
        (
            [0, 100, 200, 400, 600, 1000],
            [0, 300, 600, 1200, 1800, 3000],
            "5 requests: large sequences",
        ),
        (
            [0, 16, 32, 96, 128],
            [0, 64, 128, 384, 512],
            "4 requests: chunk prefill-like",
        ),
    ],
)
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_sink_varlen(
    dtype, indptr_config, num_qo_heads, num_kv_heads, window_left, causal, backend
):
    """
    Test variable length sequences within a batch.
    Each request in the batch can have different query and key/value lengths.
    """
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    # Unpack the indptr configuration
    qo_indptr, kv_indptr, description = indptr_config

    # Validate that qo_indptr and kv_indptr have same batch size
    if len(qo_indptr) != len(kv_indptr):
        pytest.skip(
            f"qo_indptr and kv_indptr must have same batch size for {description}"
        )

    batch_size = len(qo_indptr) - 1
    total_qo_len = qo_indptr[-1]
    total_kv_len = kv_indptr[-1]
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)

    # Check if any request has qo_len > kv_len for causal case
    if causal:
        for i in range(batch_size):
            qo_len_i = qo_indptr[i + 1] - qo_indptr[i]
            kv_len_i = kv_indptr[i + 1] - kv_indptr[i]
            if qo_len_i > kv_len_i:
                pytest.skip(
                    "qo_len > kv_len not supported for causal attention in varlen mode"
                )

    # Create input tensors
    q = torch.randn(total_qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    qo_indptr_tensor = torch.tensor(qo_indptr, dtype=torch.int32, device=device)
    kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32, device=device)

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    # Test the variable length reference implementation
    o_ref = sink_attention_varlen_ref(
        q, k, v, sink, window_left, causal, sm_scale, qo_indptr_tensor, kv_indptr_tensor
    )

    # Verify output shape
    assert o_ref.shape == (
        total_qo_len,
        num_qo_heads,
        head_dim,
    ), f"Expected shape ({total_qo_len}, {num_qo_heads}, {head_dim}), got {o_ref.shape}"

    # Test against FlashInfer kernel for verification
    # Create JIT arguments for attention sink
    jit_args = (
        f"batch_prefill_attention_sink_{filename_safe_dtype_map[dtype]}_swa_{window_left >= 0}_{backend}",  # uri
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        torch.int32,  # idtype
        head_dim,  # hidden_dim_qk
        head_dim,  # hidden_dim_vo
        ["sink"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["double"],  # additional_scalar_dtypes
        "AttentionSink",
        attention_sink_decl[backend],
    )
    jit_kwargs = {
        "use_sliding_window": window_left >= 0,
    }

    # Create workspace buffer
    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )

    # Test with BatchPrefillWithRaggedKVCacheWrapper
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend=backend,
        jit_args=jit_args,
        jit_kwargs=jit_kwargs,
    )

    wrapper.plan(
        qo_indptr_tensor,
        kv_indptr_tensor,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    o = wrapper.run(q, k, v, sink, sm_scale)

    # Compare varlen reference result with FlashInfer kernel result
    if dtype == torch.float16:
        torch.testing.assert_close(o_ref, o, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_ref, o, rtol=1e-2, atol=1e-2)

    # Also test with BatchPrefillWithPagedKVCacheWrapper
    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend=backend,
        jit_args=jit_args,
        jit_kwargs=jit_kwargs,
    )
    kv_indices_host = torch.arange(0, total_kv_len, dtype=torch.int32, device=device)
    paged_kv_last_page_len_host = torch.full(
        (batch_size,), 1, dtype=torch.int32, device=device
    )
    wrapper_paged.plan(
        qo_indptr_tensor,
        kv_indptr_tensor,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
        non_blocking=True,
    )
    o_paged = wrapper_paged.run(q, (k, v), sink, sm_scale)
    if dtype == torch.float16:
        torch.testing.assert_close(o_ref, o_paged, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_ref, o_paged, rtol=1e-2, atol=1e-2)

    # Test with non-contiguous KV indices for variable length sequences
    total_pages = total_kv_len
    if total_pages > 1:  # Only test fragmentation when we have multiple pages
        # Create fragmented page allocation pattern
        import random

        random.seed(
            42 + batch_size + total_kv_len
        )  # Vary seed with batch and total length
        all_pages = list(range(0, total_pages * 2))
        occupied_pages = set(
            random.sample(all_pages, min(total_pages, len(all_pages) // 2))
        )
        available_pages = [p for p in all_pages if p not in occupied_pages]

        # Allocate non-contiguous pages
        kv_indices_fragmented = torch.tensor(
            available_pages[:total_pages], dtype=torch.int32, device=device
        )

        # Create fragmented paged KV cache
        k_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Copy K,V data to fragmented pages
        for i, page_idx in enumerate(kv_indices_fragmented):
            k_paged_frag[page_idx, 0] = k[i]
            v_paged_frag[page_idx, 0] = v[i]

        # Test with fragmented indices
        wrapper_paged_frag = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer, kv_layout="NHD", backend=backend, jit_args=jit_args
        )
        wrapper_paged_frag.plan(
            qo_indptr_tensor,
            kv_indptr_tensor,
            kv_indices_fragmented,
            paged_kv_last_page_len_host,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
            causal=causal,
            window_left=window_left,
            q_data_type=dtype,
            kv_data_type=dtype,
            non_blocking=True,
        )
        o_paged_frag = wrapper_paged_frag.run(
            q, (k_paged_frag, v_paged_frag), sink, sm_scale
        )

        # Verify fragmented result matches reference
        if dtype == torch.float16:
            torch.testing.assert_close(o_ref, o_paged_frag, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o_ref, o_paged_frag, rtol=1e-2, atol=1e-2)

    print(
        f"Variable length test passed: {description}, batch_size={batch_size}, "
        f"qo_lens={[qo_indptr[i + 1] - qo_indptr[i] for i in range(batch_size)]}, "
        f"kv_lens={[kv_indptr[i + 1] - kv_indptr[i] for i in range(batch_size)]}, "
        f"causal={causal}"
    )


if __name__ == "__main__":
    test_attention_sink(
        torch.float16,
        batch_size=128,
        seq_len=1024,
        num_qo_heads=32,
        num_kv_heads=32,
        window_left=128,
        causal=False,
        backend="fa2",
    )
