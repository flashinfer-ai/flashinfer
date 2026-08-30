import math

import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("s_kv", [512, 1000, 8192])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True, False])
def test_cudnn_decode(
    batch_size,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    is_cuda_graph_compatible,
):
    # test set up basics
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    s_qo = 1
    head_dim = 128

    # Initialize Q tensor
    # Since the number of tokens is 1, batch size is the token count
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

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

    # Actual sequence lengths (should be randomized across batches. )
    actual_seq_lens_kv = torch.randint(
        0, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    workspace_buffer_size = math.ceil(
        (
            batch_size * s_qo * num_qo_heads * head_dim * 4
            + batch_size * s_qo * num_qo_heads * 4
        )
        / (1024 * 1024)
    ) * (1024 * 1024)

    workspace_buffer_size = max(workspace_buffer_size, 128 * 1024 * 1024)

    workspace_buffer = torch.empty(
        workspace_buffer_size, dtype=torch.int8, device=device
    )

    output = flashinfer.decode.cudnn_batch_decode_with_kv_cache(
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        max_sequence_kv=s_kv,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_tables=block_tables,
        is_cuda_graph_compatible=is_cuda_graph_compatible,
    )

    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)

    kv_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(
                    (actual_seq_lens_kv_device.flatten() + page_size - 1) // page_size,
                    dim=0,
                ),
            ]
        )
        .int()
        .to(device)
    )

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
    kv_last_page_len = (
        torch.where(
            actual_seq_lens_kv_device.flatten() % page_size == 0,
            torch.full((batch_size,), page_size, device=device),
            actual_seq_lens_kv_device.flatten() % page_size,
        )
        .int()
        .to(device)
    )

    # Workspace buffer
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer_ref, "HND")
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
    )

    output_ref = wrapper.run(q, kv_cache)

    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)


def _decode_offsets_fixture(device="cuda:0"):
    batch_size, num_heads, head_dim, page_size, s_kv = 2, 8, 128, 16, 64
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    torch.manual_seed(0)
    q = torch.randn(
        batch_size, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        total_num_pages,
        num_heads,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.arange(total_num_pages, dtype=torch.int32, device=device).view(
        batch_size, num_pages_per_seq
    )
    actual_seq_lens_kv = torch.full(
        (batch_size, 1, 1, 1), s_kv, dtype=torch.int32, device=device
    )
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    packed = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * (
        num_heads * head_dim
    )
    return (
        dict(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            scale=float(1.0 / (head_dim**0.5)),
            workspace_buffer=workspace_buffer,
            max_sequence_kv=s_kv,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
        ),
        packed,
    )


def test_cudnn_decode_packed_batch_offsets_match_omitted():
    """The packed q/out indptr describes the same addressing as omitting the offsets,
    so both must produce identical output."""
    from flashinfer.cudnn import decode as cudnn_decode

    if not cudnn_decode.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")

    call, packed = _decode_offsets_fixture()
    out_omitted = flashinfer.decode.cudnn_batch_decode_with_kv_cache(**call)
    out_packed = flashinfer.decode.cudnn_batch_decode_with_kv_cache(
        **call, batch_offsets_q=packed, batch_offsets_o=packed
    )
    torch.testing.assert_close(out_packed, out_omitted, rtol=1e-2, atol=1e-2)


def test_cudnn_decode_rejects_kv_batch_offsets():
    """batch_offsets_k/v have no effect on decode (the KV cache is addressed through
    block_tables), so they must be rejected rather than silently ignored."""
    from flashinfer.cudnn import decode as cudnn_decode

    if not cudnn_decode.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")

    call, packed = _decode_offsets_fixture()
    with pytest.raises(NotImplementedError):
        flashinfer.decode.cudnn_batch_decode_with_kv_cache(
            **call, batch_offsets_k=packed
        )
