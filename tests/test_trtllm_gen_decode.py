import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flashinfer


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values


def reference_paged_attention(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, num_kv_heads, page_size, head_dim]
    block_tables: torch.Tensor,  # [batch_size, max_blocks_per_seq]
    seq_lens: torch.Tensor,  # [batch_size]
    page_size: int,
    scale: float,
    num_kv_heads: int,
    head_dim: int,
):
    batch_size, num_q_heads, _ = q.shape
    device = q.device
    dtype = q.dtype
    head_grp_size = num_q_heads // num_kv_heads

    # Initialize output tensor
    output = torch.zeros_like(q)

    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        num_blocks = (seq_len + page_size - 1) // page_size

        # Get the blocks for this sequence
        blocks = block_tables[b, :num_blocks]

        # Initialize K and V for this sequence
        k_seq = torch.zeros(
            (num_kv_heads, seq_len, head_dim), device=device, dtype=dtype
        )
        v_seq = torch.zeros(
            (num_kv_heads, seq_len, head_dim), device=device, dtype=dtype
        )

        # Gather K and V from kv_cache
        current_pos = 0
        for block_idx, block_id in enumerate(blocks):
            # Calculate how many tokens we can take from this block
            remaining_tokens = seq_len - current_pos
            tokens_to_take = min(page_size, remaining_tokens)

            if tokens_to_take <= 0:
                break

            # Get K and V from the block
            k_block = kv_cache[
                block_id, 0, :, :tokens_to_take, :
            ]  # [num_kv_heads, tokens_to_take, head_dim]
            v_block = kv_cache[
                block_id, 1, :, :tokens_to_take, :
            ]  # [num_kv_heads, tokens_to_take, head_dim]

            # Store in the sequence tensor
            k_seq[:, current_pos : current_pos + tokens_to_take, :] = k_block
            v_seq[:, current_pos : current_pos + tokens_to_take, :] = v_block

            current_pos += tokens_to_take

        q_b = q[b].unsqueeze(1)

        k_seq = torch.repeat_interleave(k_seq, head_grp_size, dim=0)
        v_seq = torch.repeat_interleave(v_seq, head_grp_size, dim=0)
        output[b] = scaled_dot_product(
            q_b.unsqueeze(0), k_seq.unsqueeze(0), v_seq.unsqueeze(0)
        ).squeeze()

    return output


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize("batch_size", [4, 256])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
@pytest.mark.parametrize("q_dtype", ["half", "bf16", "fp8"])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_trtllm_batch_decode_fmha(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    q_dtype,
    head_grp_size,
    kv_cache_dtype,
):
    if head_grp_size == 5 and kv_cache_dtype == "fp8":
        pytest.skip("No reference provided for head_grp_size=5 and fp8 kv_cache")
    if kv_cache_dtype == "auto" and q_dtype == "fp8":
        pytest.skip("duplicated test to fp8 kvcache type.")

    # Set up test parameters
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"
    head_dim = 128
    num_qo_heads = num_kv_heads * head_grp_size
    batch_size = batch_size
    MAX_SEQ_LEN = 110

    # Initialize tensors
    num_tokens = MAX_SEQ_LEN * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size

    dtype_map = {
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }

    sm_scale = float(1.0 / (head_dim**0.5))
    q = torch.randn(batch_size, num_qo_heads, head_dim, device=device).to(
        dtype_map[q_dtype]
    )

    # Sequence lengths and block tables
    seq_lens = [torch.randint(1, MAX_SEQ_LEN, (1,)).item() for _ in range(batch_size)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    blocks_per_seq = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]
    max_num_blocks_per_seq = max(blocks_per_seq)

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
    # kv_cache_shape = (block_id, 2, num_kv_heads, page_size, head_dim)
    # Allocate more than needed blocks, block_id is just enough, to mimick real-world cases
    kv_cache_shape = (num_blocks, 2, num_kv_heads, page_size, head_dim)
    q_scale = k_scale = v_scale = 1.0
    kv_cache = torch.randn(size=kv_cache_shape, device=device).to(dtype_map[q_dtype])

    # Output type is fp8 when q is fp8, set scale for it.
    o_scale = (
        1.0 if q_dtype != "fp8" else torch.rand(1).item() * 0.5 + 0.5
    )  # Scale range: 0.5 ~ 1.0
    if kv_cache_dtype.startswith("fp8") and q_dtype != "fp8":
        kv_cache, _ = to_float8(kv_cache)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q,
        kv_cache,
        workspace_buffer,
        num_qo_heads,
        num_kv_heads,
        block_tables,
        seq_lens_tensor,
        page_size,
        max_seq_len,
        kv_cache_dtype,
        q_scale * k_scale * sm_scale,  # bmm1_scale
        v_scale / o_scale,  # bmm2_scale
    )

    # Reference implementation have functional issue or low precision with fp8, use half instead.
    ref_q = q.half() if q_dtype == "fp8" else q
    if head_grp_size == 5:
        scale = float(1.0 / (head_dim**0.5))
        output_ref = reference_paged_attention(
            ref_q,
            kv_cache,
            block_tables,
            seq_lens_tensor,
            page_size,
            scale,
            num_kv_heads,
            head_dim,
        )
    else:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size

        # Compute kv_indptr as cumulative sum of blocks per sequence
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
        kv_indptr[1:] = torch.cumsum(blocks_per_seq, dim=0)
        # Create kv_indices with only the allocated blocks
        kv_indices = all_block_ids.int()

        # Calculate last page lengths
        kv_last_page_len = seq_lens_tensor % page_size
        kv_last_page_len[kv_last_page_len == 0] = page_size

        if kv_cache_dtype == "auto":
            kv_compute_dtype = dtype_map[q_dtype]
        elif kv_cache_dtype == "fp8":
            kv_compute_dtype = torch.float8_e4m3fn

        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            data_type=kv_compute_dtype,
            q_data_type=ref_q.dtype,
        )

        output_ref = wrapper.run(ref_q, kv_cache)

    rtol, atol = (1e-2, 5e-2) if q_dtype != "fp8" else (5e-2, 7e-2)

    # convert to float32 for fp8 is not supported by assert_close
    torch.testing.assert_close(
        output.float() * o_scale, output_ref.float(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("acc_q_len", [1, 2])
@pytest.mark.parametrize("dynamic_scale", [False, True])
def test_trtllm_batch_decode_mla(
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    page_size: int,
    acc_q_len: int,
    dynamic_scale: bool,
):
    if dynamic_scale and dtype != torch.float8_e4m3fn:
        pytest.skip("Dynamic scale is not supported for non-fp8 dtype")

    torch.manual_seed(42)
    device = "cuda:0"

    # Fixed max sequence length
    MAX_SEQ_LEN = 1024

    # Deepseek attention config (decode-MLA)
    num_q_heads = 128
    num_kv_heads = 1
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512

    # Initialize tensors
    query = torch.randn(
        batch_size,
        acc_q_len,
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
    # (num_blocks, 2, page_size, kv_lora_rank + qk_rope_head_dim)
    # todo(Yingyi): do not duplicate kv_cache for the next generated cubins
    kv_cache_duplicate = torch.stack([kv_cache, kv_cache], dim=1)
    # kv_cache_duplicate = torch.stack([kv_cache], dim=1)

    # Allocate workspace buffer
    # todo(Yingyi): calculate the actual size of workspace buffer
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    bmm1_scale_tensor = (
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
        kv_cache=kv_cache_duplicate,  # kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        block_size=page_size,
        max_seq_len=max_seq_len,
        bmm1_scale=scale / ((128 + 64) ** 0.5),
        bmm2_scale=1.0,
        bmm1_scale_tensor=bmm1_scale_tensor,
        bmm2_scale_tensor=bmm2_scale_tensor,
    )
    torch.cuda.synchronize()
    print("output shape", output.shape)
    print("output", output)

    # Run reference attention and align output
    sm_scale = scale / (
        (128 + 64) ** 0.5
    )  # use head dimension before matrix absorption
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer_ref,
        backend="fa2",
        # use_cuda_graph=True,
        # qo_indptr=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
        # kv_indptr=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
        # kv_indices=torch.empty(1048576, dtype=torch.int32, device=device),
        # kv_len_arr=torch.empty(batch_size, dtype=torch.int32, device=device),
    )

    if dtype == torch.float8_e4m3fn:
        # convert query and kv_cache to bfloat16
        query = query.to(torch.bfloat16)
        kv_cache = kv_cache.to(torch.bfloat16)

    q_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * acc_q_len
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
        batch_size * acc_q_len, num_q_heads, kv_lora_rank
    )
    q_pe = query[..., kv_lora_rank:].view(
        batch_size * acc_q_len, num_q_heads, qk_rope_head_dim
    )

    # todo: fix kv_cache
    ckv = kv_cache[..., :kv_lora_rank]
    kpe = kv_cache[..., kv_lora_rank:]

    o_ref = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
    torch.cuda.synchronize()
    print("o_ref shape", o_ref.shape)
    print("o_ref", o_ref)

    # check is nan
    assert not torch.isnan(o_ref).any(), "o_ref is nan"
    assert not torch.isnan(output).any(), "output is nan"

    if dtype == torch.float8_e4m3fn:
        try:
            torch.testing.assert_close(
                output,
                o_ref.view(batch_size, acc_q_len, num_q_heads, -1),
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
                o_ref.view(batch_size, acc_q_len, num_q_heads, -1),
                rtol=1e-2,
                atol=1e-2,
            )
        except AssertionError as e:
            print("output:", output)
            print("o_ref:", o_ref)
            raise e


if __name__ == "__main__":
    # run all tests in the order of pytest
    # test_trtllm_batch_decode_mla(16, 0.5, torch.float8_e4m3fn, 32, 1)
    test_trtllm_batch_decode_mla(1024, 1.0, torch.float8_e4m3fn, 32, 2, False)
