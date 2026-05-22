import pytest
import torch

from flashinfer import BatchPrefillWithPagedKVCacheWrapper


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_effect(dtype):
    torch.manual_seed(42)

    H_QO, H_KV, N_CTX, HEAD_DIM, PAGE_SIZE = 1, 1, 8, 64, 16
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE

    # Create paged KV cache
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    paged_kv_cache = (k_cache, v_cache)

    # Create query tensor and indptrs
    q = torch.randn(N_CTX, H_QO, HEAD_DIM, dtype=dtype, device="cuda")
    qo_indptr = torch.tensor([0, N_CTX], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor(
        [N_CTX % PAGE_SIZE or PAGE_SIZE], dtype=torch.int32, device="cuda"
    )

    workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)

    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    out1, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=0.1, v_scale=0.1)
    out2, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=2.0, v_scale=2.0)

    assert not torch.allclose(out1, out2, atol=1e-3), (
        "Output should change when k_scale/v_scale values are different. "
        "This may indicate that the arguments are not passed correctly."
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_math_property(dtype: torch.dtype):
    torch.manual_seed(0)

    # ---------------- parameters ----------------
    N_CTX, PAGE_SIZE = 128, 16
    H_QO, H_KV, HEAD_DIM = 1, 1, 64  # Explicitly specify H_QO
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE

    # ---------------- paged KV cache ----------------
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn_like(k_cache)
    paged_kv_cache = (k_cache, v_cache)

    # ---------------- query and indptr ----------------
    q = torch.randn(N_CTX, H_QO, HEAD_DIM, dtype=dtype, device="cuda")
    qo_indptr = torch.tensor([0, N_CTX], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor(
        [N_CTX % PAGE_SIZE or PAGE_SIZE], dtype=torch.int32, device="cuda"
    )

    # ---------------- wrapper ----------------
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)

    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # ---------------- scale factors ----------------
    k_scale = 0.5
    v_scale = 2.0

    # -------- case 1: k_scale only ----------
    out1, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=k_scale)
    out1_ref, _ = wrapper.forward_return_lse(q * k_scale, paged_kv_cache)
    torch.testing.assert_close(out1, out1_ref, rtol=1e-2, atol=1e-3)

    # -------- case 2: v_scale only ----------
    out2, _ = wrapper.forward_return_lse(q, paged_kv_cache, v_scale=v_scale)
    out2_ref, _ = wrapper.forward_return_lse(q, paged_kv_cache)
    torch.testing.assert_close(out2, out2_ref * v_scale, rtol=1e-2, atol=1e-3)

    # -------- case 3: both k_scale and v_scale ----------
    out3, _ = wrapper.forward_return_lse(
        q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale
    )
    out3_ref, _ = wrapper.forward_return_lse(q * k_scale, paged_kv_cache)
    torch.testing.assert_close(out3, out3_ref * v_scale, rtol=1e-2, atol=1e-3)


def test_batch_prefill_invalid_fixed_cta_tile_q():
    batch_size = 2
    qo_len = 8
    kv_len = 128
    page_size = 16
    num_kv_heads = 2
    group_size = 2
    num_qo_heads = num_kv_heads * group_size
    head_dim = 64

    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    with pytest.raises(ValueError, match="fixed_cta_tile_q should be one of"):
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            fixed_split_size=2048,
            disable_split_kv=True,
            fixed_cta_tile_q=32,
        )


def test_batch_prefill_fixed_cta_tile_q_incompatible_head_dim():
    batch_size = 2
    qo_len = 8
    kv_len = 128
    page_size = 16
    num_kv_heads = 2
    group_size = 2
    num_qo_heads = num_kv_heads * group_size
    head_dim = 256  # fixed_cta_tile_q=128 is invalid for head_dim >= 256

    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    with pytest.raises(
        ValueError, match="fixed_cta_tile_q=128 is not supported with head_dim"
    ):
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            fixed_split_size=2048,
            disable_split_kv=True,
            fixed_cta_tile_q=128,
        )


def test_batch_prefill_fixed_cta_tile_q_rejected_for_non_fa2_backend():
    """fixed_cta_tile_q must raise when the resolved backend is not fa2."""
    batch_size, qo_len, kv_len, page_size, num_kv_heads, head_dim = (
        2,
        64,
        512,
        16,
        4,
        128,
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), page_size, dtype=torch.int32, device="cuda:0"
    )
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper._backend = "fa3"
    with pytest.raises(
        ValueError, match="fixed_cta_tile_q is only supported for the fa2 backend"
    ):
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            fixed_cta_tile_q=64,
        )
