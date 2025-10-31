import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [32, 64, 87])
@pytest.mark.parametrize("s_kv", [32, 64, 87])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [1, 16])
@pytest.mark.parametrize("causal", [True, False])
def test_cudnn_prefill_deepseek(
    batch_size, s_qo, s_kv, num_kv_heads, num_qo_heads, causal
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

    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)

    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )

    q_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0)
            * head_dim_qk
            * num_qo_heads,
        ]
    ).int()

    k_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_kv.view(-1), dim=0)
            * head_dim_qk
            * num_kv_heads,
        ]
    ).int()

    v_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_kv.view(-1), dim=0)
            * head_dim_vo
            * num_kv_heads,
        ]
    ).int()

    o_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0)
            * head_dim_vo
            * num_qo_heads,
        ]
    ).int()

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

    # Initialize scale
    scale = float(1.0 / (head_dim_qk**0.5))

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # output = torch.zeros_like(q)
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
        batch_offsets_q=q_indptr,
        batch_offsets_k=k_indptr,
        batch_offsets_v=v_indptr,
        batch_offsets_o=o_indptr,
        batch_offsets_stats=batch_offsets_stats,
        is_cuda_graph_compatible=True,
    )

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # kv_indptr = torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * s_kv

    # Create kv_indptr as cumulative sum of actual_seq_lens_kv
    kv_indptr = torch.cat(
        [
            torch.tensor(
                [0],
                device=device,
            ),
            torch.cumsum(actual_seq_lens_kv.view(-1), dim=0),
        ]
    ).int()

    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        kv_layout="NHD",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    output_ref, lse_ref = wrapper.run(q, k_cache, v_cache, return_lse=True)

    torch.testing.assert_close(
        output,
        output_ref,
        atol=1e-2,
        rtol=1e-2,
    )
