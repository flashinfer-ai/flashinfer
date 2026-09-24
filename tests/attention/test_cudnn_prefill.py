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

    output = wrapper_cudnn.run(q, (k_cache, v_cache))

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

    output_ref = wrapper.run(q, kv_cache)
    torch.testing.assert_close(output, output_ref, atol=3e-3, rtol=1e-2)


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

    # TODO: Remove this xfail once cuDNN fixes FP8 prefill on Blackwell
    pytest.xfail(
        "cuDNN FP8 prefill has known issues on Blackwell; expected to be fixed in a subsequent cuDNN release"
    )

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

    output = wrapper_cudnn.run(
        q_fp8,
        (k_cache_fp8, v_cache_fp8),
        q_scale=q_scale,
        k_scale=k_scale_tensor,
        v_scale=v_scale_tensor,
    )

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

    output_ref = wrapper.run(q, kv_cache)

    torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=1e-2)


def _paged_lse_problem(q_lens, kv_lens, num_qo_heads, num_kv_heads, page_size=16):
    """Packed Q plus a token-major paged KV cache, with cuDNN and fa2 views of it."""
    head_dim = 128
    device = "cuda:0"
    batch_size = len(q_lens)
    pages = [(n + page_size - 1) // page_size for n in kv_lens]
    qo_indptr = torch.tensor([0] + q_lens, device=device).cumsum(0).int()
    kv_indptr = torch.tensor([0] + pages, device=device).cumsum(0).int()
    kv_indices = torch.arange(sum(pages), device=device, dtype=torch.int32)
    kv_last_page_len = torch.tensor(
        [(n - 1) % page_size + 1 for n in kv_lens], device=device, dtype=torch.int32
    )
    block_tables = torch.zeros(batch_size, max(pages), device=device, dtype=torch.int32)
    for i in range(batch_size):
        block_tables[i, : pages[i]] = kv_indices[kv_indptr[i] : kv_indptr[i + 1]]

    q = torch.randn(
        sum(q_lens), num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    # [pages, 2, page_size, H_kv, D] token-major memory (NHD).
    kv_cache = torch.randn(
        sum(pages), 2, page_size, num_kv_heads, head_dim, device=device
    ).to(torch.bfloat16)
    # cuDNN takes [pages, H_kv, page_size, D]-shaped views over that memory.
    cudnn_kv = (kv_cache[:, 0].transpose(1, 2), kv_cache[:, 1].transpose(1, 2))

    plan_args = (
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    cudnn_plan_kwargs = dict(
        seq_lens=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        seq_lens_q=torch.tensor(q_lens, device=device, dtype=torch.int32),
        max_token_per_sequence=max(q_lens),
        max_sequence_kv=max(kv_lens),
        block_tables=block_tables,
    )
    return q, kv_cache, cudnn_kv, plan_args, cudnn_plan_kwargs


@pytest.mark.parametrize("num_kv_heads", [2, 8])
@pytest.mark.parametrize("causal", [True, False])
def test_cudnn_paged_prefill_return_lse(num_kv_heads, causal):
    """return_lse=True yields the packed [tokens, heads] LSE, matching fa2.

    plan() gets no sm_scale, so this also covers the default softmax scale.
    Regression test for https://github.com/flashinfer-ai/flashinfer/issues/5258.
    """
    torch.manual_seed(0)
    q, kv_cache, cudnn_kv, plan_args, cudnn_plan_kwargs = _paged_lse_problem(
        [17, 1, 64, 33], [40, 90, 64, 70], 8, num_kv_heads
    )
    plan_kwargs = dict(causal=causal, q_data_type=torch.bfloat16)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=q.device)
    wrapper_cudnn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="cudnn"
    )
    wrapper_cudnn.plan(*plan_args, **plan_kwargs, **cudnn_plan_kwargs)
    out, lse = wrapper_cudnn.run(q, cudnn_kv, return_lse=True)
    assert lse.shape == (q.shape[0], q.shape[1])

    workspace_ref = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=q.device)
    wrapper_ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_ref, "NHD", backend="fa2"
    )
    wrapper_ref.plan(*plan_args, **plan_kwargs)
    out_ref, lse_ref = wrapper_ref.run(q, kv_cache, return_lse=True)

    torch.testing.assert_close(out, out_ref, atol=3e-3, rtol=1e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-3)

    # The LSE-free call must agree with the LSE-returning one.
    torch.testing.assert_close(wrapper_cudnn.run(q, cudnn_kv), out)


def test_cudnn_paged_prefill_single_token_gqa_lse_rejected():
    """cuDNN's single-token GQA kernel writes a partial LSE; refuse it loudly."""
    torch.manual_seed(0)
    q, _, cudnn_kv, plan_args, cudnn_plan_kwargs = _paged_lse_problem(
        [1, 1, 1, 1], [40, 90, 64, 70], 8, 2
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=q.device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="cudnn"
    )
    wrapper.plan(
        *plan_args, causal=True, q_data_type=torch.bfloat16, **cudnn_plan_kwargs
    )
    with pytest.raises(NotImplementedError, match="single-token"):
        wrapper.run(q, cudnn_kv, return_lse=True)
    # Without the LSE the same call is fine.
    assert wrapper.run(q, cudnn_kv).shape == q.shape
