import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from utils_fp4 import cast_from_fp4, recover_swizzled_scales, ref_nvfp4_quant

import flashinfer
from flashinfer.utils import FP4Tensor


def flip_coin(*args, **kwargs):
    # Use any test parameters to deterministically decide branch
    # This makes test configurations go through different paths
    param_tuple = args + tuple(sorted(kwargs.items()))
    hash_value = hash(param_tuple)
    return (hash_value % 2) == 0


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
@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("window_left", [-1, 127])
@pytest.mark.parametrize(
    "q_dtype,kv_cache_dtype,o_dtype",
    [
        ("half", "half", "half"),
        ("half", "fp8", "half"),
        ("bf16", "bf16", "bf16"),
        ("bf16", "fp8", "bf16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
    ],
)
def test_trtllm_batch_decode_fmha(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_cache_dtype,
):

    # Set up test parameters
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"
    head_dim = 128
    MAX_SEQ_LEN = 110

    dtype_map = {
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
        "nvfp4": "nvfp4",
    }

    # Sequence lengths and block tables
    num_qo_heads = num_kv_heads * head_grp_size

    q = torch.randn(
        batch_size,
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16 if q_dtype == "fp8" else dtype_map[q_dtype],
        device=device,
    )
    if q_dtype == "fp8":
        q, q_scale = to_float8(q)
        # Reference implementation have functional issue or low precision with fp8, use bfloat16 and fake-quantization instead.
        ref_q = q.bfloat16() * q_scale
    else:
        q_scale = 1.0
        ref_q = q

    seq_lens = torch.randint(1, MAX_SEQ_LEN, (batch_size,), dtype=torch.int32)
    seq_lens[-1] = MAX_SEQ_LEN
    seq_lens_gpu = seq_lens.to(device)
    max_seq_len = torch.max(seq_lens).item()

    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = torch.max(blocks_per_seq).item()

    # Generate random but unique block IDs for all sequences
    total_blocks_needed = torch.sum(blocks_per_seq).item()
    all_block_ids = torch.randperm(
        total_blocks_needed, dtype=torch.int32, device=device
    )  # Random permutation

    # Generate unique block IDs for all sequences
    block_id = 0
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device
    )

    # Populate block tables and track block assignments
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    # Create separate K and V caches
    num_tokens = max_seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size

    kv_dtype = dtype_map[q_dtype] if q_dtype != "fp8" else torch.bfloat16
    k_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=kv_dtype, device=device
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=kv_dtype, device=device
    )
    # Convert K and V separately to fp8 if needed
    if kv_cache_dtype.startswith("fp8"):
        k_cache, k_scale = to_float8(k_cache)
        v_cache, v_scale = to_float8(v_cache)
        # use high precision for reference kv_cache to avoid precision/functional issue
        ref_kv_type = torch.bfloat16 if q_dtype == "fp8" else dtype_map[q_dtype]
        ref_kv_cache = torch.stack(
            [k_cache.to(ref_kv_type) * k_scale, v_cache.to(ref_kv_type) * v_scale],
            dim=1,
        )
    else:
        k_scale = v_scale = 1.0
        ref_kv_cache = torch.stack([k_cache, v_cache], dim=1)

    # Combine K and V into interleaved format for the API
    kv_cache = torch.stack(
        [k_cache, v_cache], dim=1
    )  # Shape: (num_blocks, 2, num_kv_heads, page_size, head_dim)

    if o_dtype == "fp8":
        o_scale = torch.rand(1).item() * 0.5 + 0.5  # Scale range: 0.5 ~ 1.0
    else:
        o_scale = 1.0
    o_sf_scale = (
        300 if o_dtype == "nvfp4" else None
    )  # choose a value to make error smaller by testing.
    o_sf_vec_size = 16 if o_dtype == "nvfp4" else None

    sm_scale = float(1.0 / (head_dim**0.5))

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Compute kv_indptr as cumulative sum of blocks per sequence
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(blocks_per_seq.to(device), dim=0, dtype=torch.int32),
        ]
    )

    if flip_coin(batch_size, page_size, num_kv_heads, head_grp_size, o_dtype):
        if o_dtype == "nvfp4":
            fp4_out_shape = q.shape[:-1] + (math.ceil(q.shape[-1] / 2),)

            fp4_out_scale_shape = (
                math.ceil(q.shape[0] / 128) * 128,
                math.ceil(q.shape[1] * q.shape[2] / o_sf_vec_size / 4) * 4,
            )

            out_scale_factor = torch.empty(
                fp4_out_scale_shape, dtype=torch.float8_e4m3fn, device=q.device
            )
            extra_size = fp4_out_scale_shape[0] - q.shape[0]
            o_sf_start_index = (
                torch.randint(0, extra_size, (1,)).item() if extra_size > 0 else 0
            )
            out_data = torch.empty(fp4_out_shape, dtype=torch.uint8, device=q.device)
            out = FP4Tensor(out_data, out_scale_factor, o_sf_start_index)
        else:
            out = torch.empty_like(q, dtype=dtype_map[o_dtype])
    else:
        out = None

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens_gpu,
        max_seq_len,
        q_scale * k_scale * sm_scale,  # bmm1_scale
        v_scale / o_scale,  # bmm2_scale
        window_left,  # window_left
        out=out,
        out_dtype=dtype_map[o_dtype],
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
    )

    # Handle different return types based on out_dtype
    if o_dtype == "nvfp4":
        out_scale_factor = output.scale  # FP4Tensor.scale
        o_sf_start_index = output.scale_start_index
        output = output.data  # FP4Tensor.data
    else:
        out_scale_factor = None

    output = output.squeeze(1)

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    blocks_per_seq = (seq_lens_gpu + page_size - 1) // page_size

    # Calculate last page lengths
    kv_last_page_len = seq_lens_gpu % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size

    wrapper.plan(
        kv_indptr,
        all_block_ids,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        window_left=window_left,
        data_type=ref_kv_cache.dtype,
        q_data_type=ref_q.dtype,
    )

    output_ref = wrapper.run(ref_q, ref_kv_cache)

    if q_dtype == "fp8" and o_dtype == "nvfp4":
        rtol, atol = 3e-1, 1e0
    elif q_dtype == "fp8" and o_dtype == "fp8":
        rtol, atol = 5e-2, 7e-2
    else:
        rtol, atol = 1e-2, 5e-2

    if o_dtype == "nvfp4":
        output = cast_from_fp4(output)
        output_ref, out_scale_factor_ref = ref_nvfp4_quant(output_ref, o_sf_scale, 16)
        out_scale_factor = recover_swizzled_scales(
            out_scale_factor,
            output.shape[0],
            output.shape[1] * output.shape[2],
            16,
            o_sf_start_index,
        )

        torch.testing.assert_close(
            out_scale_factor.float().reshape(out_scale_factor_ref.shape),
            out_scale_factor_ref.float(),
            rtol=2e-1,
            atol=2e-1,
        )
        rmse = torch.sqrt(
            torch.mean((output.float() * o_scale - output_ref.float()) ** 2)
        )
        assert rmse.item() < 0.3
    # convert to float32 for fp8 is not supported by assert_close
    torch.testing.assert_close(
        output.float() * o_scale, output_ref.float(), rtol=rtol, atol=atol
    )

    if o_dtype != "nvfp4":  # wrapper api does not support fp4 output yet.
        # test wrapper with trtllm-gen backend
        wrapper2 = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend="trtllm-gen"
        )
        wrapper2.plan(
            kv_indptr,
            all_block_ids,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            data_type=kv_cache.dtype,
            q_data_type=q.dtype,
            window_left=window_left,
        )
        output2 = wrapper2.run(
            q.contiguous(),
            kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale / o_scale,
        )
        # v_scale, o_scale is not supported in wrapper api yet.
        if v_scale == o_scale == 1.0:
            assert (output2 == output).all()


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("q_len_per_request", [1, 2])
@pytest.mark.parametrize("dynamic_scale", [False])
def test_trtllm_batch_decode_mla(
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    page_size: int,
    q_len_per_request: int,
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
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

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
    )

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
