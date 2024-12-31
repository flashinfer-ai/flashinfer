import pytest
import torch
from jit_utils import jit_decode_attention_func_args, jit_prefill_attention_func_args

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if flashinfer.jit.has_prebuilt_ops:
        yield
    else:
        try:
            flashinfer.jit.parallel_load_modules(
                jit_decode_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [64, 128, 256],  # head_dims
                    [0],  # pos_encoding_modes
                    [False],  # use_sliding_windows
                    [False],  # use_logits_soft_caps
                )
                + jit_prefill_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [64, 128, 256],  # head_dims
                    [0],  # pos_encoding_modes
                    [False],  # use_sliding_windows
                    [False],  # use_logits_soft_caps
                    [False],  # use_fp16_qk_reductions
                )
            )
        except Exception as e:
            # abort the test session if warmup fails
            pytest.exit(str(e))
        finally:
            yield


@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("page_size", [1, 5])
@pytest.mark.parametrize("seq_len", [1])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_batch_paged_decode_packed_input(
    batch_size,
    page_size,
    seq_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")
    nnz = batch_size * seq_len
    num_pages_per_req = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_req
    last_page_len = (seq_len - 1) % page_size + 1
    k_cache = torch.randn(
        size=(num_pages, page_size, num_kv_heads, head_dim),
        dtype=torch.float16,
        device="cuda:0",
    )
    v_cache = torch.randn_like(k_cache)
    paged_kv_cache = (k_cache, v_cache)
    workspace_buffer = torch.empty(
        (256 * 1024 * 1024,), dtype=torch.uint8, device="cuda:0"
    )
    paged_kv_indptr = torch.tensor(
        [i * num_pages_per_req for i in range(batch_size + 1)],
        dtype=torch.int32,
        device="cuda:0",
    )
    paged_kv_indices = torch.tensor(
        list(range(num_pages)), dtype=torch.int32, device="cuda:0"
    )
    paged_kv_last_page_len = torch.tensor(
        [last_page_len for _ in range(batch_size)], dtype=torch.int32, device="cuda:0"
    )

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer)
    wrapper.plan(
        indptr=paged_kv_indptr,
        indices=paged_kv_indices,
        last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )

    qkv_packed = torch.randn(
        size=(nnz, (num_qo_heads + 2 * num_kv_heads) * head_dim),
        dtype=torch.float16,
        device="cuda:0",
    )
    qkv_split_idx = (
        num_qo_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
    )
    q, _, _ = qkv_packed.split(qkv_split_idx, dim=-1)
    q = q.view(-1, num_qo_heads, head_dim)
    o_packed = wrapper.run(q, paged_kv_cache)
    o_contiguous = wrapper.run(q.contiguous(), paged_kv_cache)
    torch.testing.assert_close(o_packed, o_contiguous, rtol=1e-3, atol=1e-3)
