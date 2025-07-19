import math

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("s_qo", [2])
@pytest.mark.parametrize("s_kv", [2])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [1])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("is_cuda_graph_compatible", [False])
def test_cudnn_prefill_deepseek(
    batch_size,
    s_qo,
    s_kv,
    num_kv_heads,
    num_qo_heads,
    causal,
    is_cuda_graph_compatible,
):
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    head_dim_qk = 192
    head_dim_vo = 128

    return_lse = True

    # test set up basics
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = actual_seq_lens_q

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )

    qo_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(actual_seq_lens_q.view(-1), dim=0)
                * head_dim_qk
                * num_qo_heads,
            ]
        )
        .int()
        .to(device)
    )

    batch_offsets_stats = torch.cat(
        [
            torch.zeros(
                1, device=actual_seq_lens_q.device, dtype=actual_seq_lens_q.dtype
            ),
            torch.cumsum(actual_seq_lens_q.flatten(), dim=0) * num_qo_heads,
        ]
    ).cuda()

    k_cache = torch.randn(
        batch_size * s_kv,
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        batch_size * s_kv,
        num_kv_heads,
        head_dim_vo,
        device=device,
        dtype=torch.bfloat16,
    )

    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * s_kv
    )

    # Initialize scale
    scale = float(1.0 / (head_dim_qk**0.5))

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    output, lse = flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        causal=causal,
        return_lse=return_lse,
        batch_offsets_q=qo_indptr,
        batch_offsets_k=qo_indptr,
        batch_offsets_v=qo_indptr,
        batch_offsets_o=qo_indptr,
        batch_offsets_stats=batch_offsets_stats,
        is_cuda_graph_compatible=is_cuda_graph_compatible,
    )

    actual_seq_lens_q_device = actual_seq_lens_q.to(device)
    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)
    qo_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(actual_seq_lens_q_device.view(-1), dim=0),
            ]
        )
        .int()
        .to(device)
    )

    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device="cuda"
    )
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer_ref,
        "NHD",
    )

    print(f"head_dim_qk: {head_dim_qk}, head_dim_vo: {head_dim_vo}")

    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        q_data_type=torch.bfloat16,
    )
    output_ref, lse_ref = wrapper.run_return_lse(q, k_cache, v_cache)

    print(f"output: {output.flatten()[:10]}")
    print(f"output_ref: {output_ref.flatten()[:10]}")
    print()

    torch.testing.assert_close(output, output_ref, atol=1e-3, rtol=1e-2)
