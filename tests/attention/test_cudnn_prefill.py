import pytest
import torch

import flashinfer
import cudnn

from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")

    # test set up basics
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    q_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device)
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]

    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )

    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                dim=0,
            ),
        ]
    ).int()

    # kv_indices
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )

    # kv_last_page_len
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim**0.5))

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_cudnn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    wrapper_cudnn.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
        seq_lens=actual_seq_lens_kv,
        seq_lens_q=actual_seq_lens_q,
        sm_scale=scale,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        block_tables=block_tables,
    )

    cache_nhd = (k_cache.transpose(1, 2), v_cache.transpose(1, 2))
    result = wrapper_cudnn.run(q, cache_nhd, return_lse=return_lse)
    output, stats = result if return_lse else (result, None)
    if is_cuda_graph_compatible:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper_cudnn.run(
                q, cache_nhd, out=output, lse=stats, return_lse=return_lse
            )
        output.fill_(torch.nan)
        if stats is not None:
            stats.fill_(torch.nan)
        graph.replay()

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # Workspace buffer
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "HND", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )

    result_ref = wrapper.run(q, kv_cache, return_lse=return_lse)
    output_ref, stats_ref = result_ref if return_lse else (result_ref, None)
    torch.testing.assert_close(output, output_ref, atol=3e-3, rtol=1e-2)
    if return_lse:
        torch.testing.assert_close(stats, stats_ref, atol=3e-3, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill_fp8(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    if cudnn.backend_version() < 91701:
        pytest.skip("cuDNN backend version is less than 9.17.1, skipping test")

    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")

    # test set up basics
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"

    major, _ = get_compute_capability(torch.device(device))

    if major != 10:
        pytest.skip(
            f"cuDNN FP8 prefill is not supported on compute capability {major}, skipping test"
        )

    if cudnn.backend_version() < 92400:
        pytest.xfail("Legacy Blackwell FP8 coverage is qualified from cuDNN 9.24")

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    q_scale = q.amax().item() / 256

    q_scale = torch.tensor(q_scale, device=device, dtype=torch.float32)
    q_fp8 = (q / q_scale).to(torch.float8_e4m3fn)

    q_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device) * 0.05
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]

    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )

    k_scale = k_cache.amax().item() / 256
    v_scale = v_cache.amax().item() / 256
    k_cache_fp8 = (k_cache / k_scale).to(torch.float8_e4m3fn)
    v_cache_fp8 = (v_cache / v_scale).to(torch.float8_e4m3fn)

    k_scale_tensor = torch.tensor(k_scale, device=device, dtype=torch.float32)
    v_scale_tensor = torch.tensor(v_scale, device=device, dtype=torch.float32)

    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                dim=0,
            ),
        ]
    ).int()

    # kv_indices
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )

    # kv_last_page_len
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim**0.5))

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_cudnn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    wrapper_cudnn.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        seq_lens=actual_seq_lens_kv,
        seq_lens_q=actual_seq_lens_q,
        sm_scale=scale,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        block_tables=block_tables,
    )

    cache_nhd = (k_cache_fp8.transpose(1, 2), v_cache_fp8.transpose(1, 2))
    run_kwargs = dict(
        q_scale=q_scale,
        k_scale=k_scale_tensor,
        v_scale=v_scale_tensor,
        return_lse=return_lse,
    )
    result = wrapper_cudnn.run(q_fp8, cache_nhd, **run_kwargs)
    output, stats = result if return_lse else (result, None)

    def check():
        q_start = 0
        for batch in range(batch_size):
            lq = int(actual_seq_lens_q[batch].item())
            lkv = int(actual_seq_lens_kv[batch].item())
            qs = slice(q_start, q_start + lq)
            qi = (q_fp8[qs].float() * q_scale).transpose(0, 1)
            pages = block_tables[batch].long()
            ki = (
                k_cache_fp8[pages]
                .float()
                .permute(1, 0, 2, 3)
                .reshape(num_kv_heads, -1, head_dim)[:, :lkv]
                * k_scale_tensor
            )
            vi = (
                v_cache_fp8[pages]
                .float()
                .permute(1, 0, 2, 3)
                .reshape(num_kv_heads, -1, head_dim)[:, :lkv]
                * v_scale_tensor
            )
            ki = ki.repeat_interleave(num_qo_heads // num_kv_heads, dim=0)
            vi = vi.repeat_interleave(num_qo_heads // num_kv_heads, dim=0)
            scores = qi @ ki.transpose(-1, -2) * scale
            if causal:
                row = torch.arange(lq, device=device)[:, None]
                col = torch.arange(lkv, device=device)[None, :]
                scores.masked_fill_(col > row + lkv - lq, -torch.inf)
            ref = (scores.softmax(-1) @ vi).transpose(0, 1)
            torch.testing.assert_close(output[qs].float(), ref, atol=1e-2, rtol=1e-2)
            if stats is not None:
                ref_lse = scores.logsumexp(-1).T * flashinfer.utils.log2e
                torch.testing.assert_close(stats[qs], ref_lse, atol=1e-2, rtol=1e-2)
            q_start += lq

    check()
    if is_cuda_graph_compatible:
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                wrapper_cudnn.run(q_fp8, cache_nhd, out=output, lse=stats, **run_kwargs)
            # Captured execution must read current inputs, not just reproduce the
            # values left by warmup. Poison every caller-visible output as well.
            q_fp8.copy_((-q_fp8.float()).to(q_fp8.dtype))
            output.fill_(torch.nan)
            if stats is not None:
                stats.fill_(torch.nan)
            graph.replay()
            check()
            # GPU scale contents can change without recapturing or requantizing
            # the payload. Check a scale-only replay separately from Q rebinding.
            q_scale.mul_(0.5)
            k_scale_tensor.mul_(0.75)
            v_scale_tensor.mul_(1.25)
            output.fill_(torch.nan)
            if stats is not None:
                stats.fill_(torch.nan)
            graph.replay()
            check()
        finally:
            graph.reset()
