import math

import pytest
import torch

import flashinfer

global_workspace_buffer = None
workspace_size = 128 * 1024 * 1024


@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("s_qo", [32, 64, 87])
@pytest.mark.parametrize("s_kv", [32, 64, 87])
@pytest.mark.parametrize("num_kv_heads", [16, 32])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_gen_prefill_deepseek(
    batch_size, s_qo, s_kv, num_kv_heads, head_grp_size, causal
):
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    num_qo_heads = num_kv_heads * head_grp_size
    head_dim_qk = 192
    head_dim_vo = 128

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
    cumsum_s_kv = torch.sum(actual_seq_lens_kv)

    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )

    k_cache = torch.randn(
        (cumsum_s_kv, num_kv_heads, head_dim_qk),
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        (cumsum_s_kv, num_kv_heads, head_dim_vo),
        device=device,
        dtype=torch.bfloat16,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim_qk**0.5))

    workspace_buffer = torch.empty(workspace_size, dtype=torch.int8, device=device)

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
        torch.zeros(workspace_size, device="cuda", dtype=torch.uint8),
        kv_layout="NHD",
        backend="cutlass",
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
    output = torch.empty_like(output_ref)

    bmm1_scale = scale
    bmm2_scale = 1.0
    bmm1_scale_log2_tensor = torch.tensor([scale * math.log2(math.e)], device=device)
    bmm2_scale_tensor = torch.tensor([1.0], device=device)
    output_trtllm, lse_trtllm = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q,
        k_cache,
        v_cache,
        workspace_buffer,
        actual_seq_lens_kv,
        s_qo,
        s_kv,
        bmm1_scale,
        bmm2_scale,
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
        out=output,
        bmm1_scale_log2_tensor=bmm1_scale_log2_tensor,
        bmm2_scale_tensor=bmm2_scale_tensor,
    )
    torch.testing.assert_close(
        output_trtllm,
        output_ref,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        lse_trtllm,
        lse_ref,
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize(
    "q_len_per_request", [1, 2]
)  # todo(Yingyi): verify larger q_len_per_request
@pytest.mark.parametrize("dynamic_scale", [False])
@pytest.mark.parametrize("enable_pdl", [True, False, None])
def test_trtllm_batch_decode_mla(
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    page_size: int,
    q_len_per_request: int,
    dynamic_scale: bool,
    enable_pdl: bool,
):
    if dynamic_scale and dtype != torch.float8_e4m3fn:
        pytest.skip("Dynamic scale is not supported for non-fp8 dtype")

    torch.manual_seed(42)
    device = "cuda:0"

    # Fixed max sequence length
    MAX_SEQ_LEN = 1024

    # Deepseek attention config (decode-MLA)
    num_q_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512

    # Initialize tensors
    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
    ).to(dtype)

    num_tokens = MAX_SEQ_LEN * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size

    # Sequence lengths and block tables
    seq_lens = [torch.randint(1, MAX_SEQ_LEN, (1,)).item() for _ in range(batch_size)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq.max().item()

    # Generate random but unique block IDs for all sequences
    total_blocks_needed = sum(blocks_per_seq)
    all_block_ids = torch.randperm(
        total_blocks_needed, device=device
    )  # Random permutation

    # Generate unique block IDs for all sequences
    block_id = 0
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int, device=device
    )

    # Populate block tables and track block assignments
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    # Create interleaved KV cache
    # Allocate more than needed blocks, block_id is just enough, to mimick real-world cases
    kv_cache = torch.randn(
        size=(num_blocks, page_size, kv_lora_rank + qk_rope_head_dim), device=device
    ).to(dtype)
    # (num_blocks, 1, page_size, kv_lora_rank + qk_rope_head_dim)

    # Allocate workspace buffer
    # todo(Yingyi): calculate the actual size of workspace buffer
    global global_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    workspace_buffer = global_workspace_buffer

    bmm1_log2_scale_tensor = (
        torch.tensor(
            [scale / ((128 + 64) ** 0.5 * math.log2(math.e))],
            dtype=torch.float32,
            device=device,
        )
        if dynamic_scale
        else None
    )
    bmm2_scale_tensor = (
        torch.tensor([1.0], dtype=torch.float32, device=device)
        if dynamic_scale
        else None
    )

    # Run decode-MLA
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=scale / ((128 + 64) ** 0.5),
        bmm2_scale=1.0,
        bmm1_scale_log2_tensor=bmm1_log2_scale_tensor,
        bmm2_scale_tensor=bmm2_scale_tensor,
        enable_pdl=enable_pdl,
    )

    # Run reference attention and align output
    sm_scale = scale / (
        (128 + 64) ** 0.5
    )  # use head dimension before matrix absorption
    workspace_buffer_ref = torch.empty(workspace_size, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer_ref,
        backend="fa2",
    )

    if dtype == torch.float8_e4m3fn:
        # convert query and kv_cache to bfloat16
        query = query.to(torch.bfloat16)
        kv_cache = kv_cache.to(torch.bfloat16)

    q_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * q_len_per_request
    )
    kv_indptr = torch.zeros_like(q_indptr)
    kv_indptr[1:] = torch.cumsum(blocks_per_seq, dim=0)
    kv_indices = all_block_ids.int()

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        seq_lens_tensor,
        num_q_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        True,
        sm_scale,
        query.dtype,
        kv_cache.dtype,
    )
    q_nope = query[..., :kv_lora_rank].view(
        batch_size * q_len_per_request, num_q_heads, kv_lora_rank
    )
    q_pe = query[..., kv_lora_rank:].view(
        batch_size * q_len_per_request, num_q_heads, qk_rope_head_dim
    )

    # todo: fix kv_cache
    ckv = kv_cache[..., :kv_lora_rank]
    kpe = kv_cache[..., kv_lora_rank:]

    o_ref = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    # check is nan
    assert not torch.isnan(o_ref).any(), "o_ref is nan"
    assert not torch.isnan(output).any(), "output is nan"

    if dtype == torch.float8_e4m3fn:
        try:
            torch.testing.assert_close(
                output,
                o_ref.view(batch_size, q_len_per_request, num_q_heads, -1),
                rtol=1e-1,
                atol=1e-1,
            )  # todo: do reference with normal attention?
        except AssertionError as e:
            print("output:", output)
            print("o_ref:", o_ref)
            raise e
    else:
        try:
            torch.testing.assert_close(
                output,
                o_ref.view(batch_size, q_len_per_request, num_q_heads, -1),
                rtol=1e-2,
                atol=1e-2,
            )
        except AssertionError as e:
            print("output:", output)
            print("o_ref:", o_ref)
            raise e
