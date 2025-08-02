import math

import pytest
import torch
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


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize("batch_size", [4, 8, 128])
@pytest.mark.parametrize("kv_len", [512, 2048])
@pytest.mark.parametrize("qo_len", [32, 16, 128, 512])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("q_dtype", ["half", "bf16", "fp8"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("window_left", [-1])  # todo(Siyuan): add 127 window_left
def test_trtllm_batch_context_wrapper(
    kv_layout,
    batch_size,
    qo_len,
    kv_len,
    num_qo_heads,
    head_dim,
    page_size,
    num_kv_heads,
    q_dtype,
    logits_soft_cap,
    window_left,
):
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    dtype_map = {
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
        "nvfp4": "nvfp4",
    }

    if q_dtype == "fp8":
        q = torch.randn(
            batch_size * qo_len,
            num_qo_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        q, q_scale = to_float8(q)
        # Reference implementation have functional issue or low precision with fp8, use bfloat16 and fake-quantization instead.
        ref_q = q.bfloat16() * q_scale
    else:
        q = torch.randn(
            batch_size * qo_len,
            num_qo_heads,
            head_dim,
            device=device,
            dtype=dtype_map[q_dtype],
        )
        q_scale = 1.0
        ref_q = q

    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
    kv_data = kv_data_fp32.to(dtype_map[q_dtype])
    ref_kv_data = kv_data.bfloat16() if q_dtype == "fp8" else kv_data
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    # reference
    q_indptr_gpu = q_indptr_cpu.to(device)
    kv_indptr_gpu = kv_indptr_cpu.to(device)
    kv_indices_gpu = kv_indices_cpu.to(device)
    kv_last_page_len_gpu = kv_last_page_len_cpu.to(device)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_last_page_len_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        logits_soft_cap=logits_soft_cap,
        q_data_type=ref_q.dtype,
        window_left=window_left,
    )
    reference_output = wrapper.run(ref_q, ref_kv_data)
    reference_kv_cache = kv_data.clone()

    # trtllm-gen
    wrapper2 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="trtllm-gen"
    )
    wrapper2.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_last_page_len_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        logits_soft_cap=logits_soft_cap,
        q_data_type=dtype_map[q_dtype],
        window_left=window_left,
    )
    output = wrapper2.run(q, kv_data, q_scale=q_scale)
    rmse = torch.sqrt(torch.mean((output.float() - reference_output.float()) ** 2))
    assert rmse.item() < (1e-2 if q_dtype == "fp8" else 1e-3)

    if q_dtype == "fp8":
        rtol, atol = 8e-2, 8e-2
    else:
        rtol, atol = 1e-2, 1e-2

    torch.testing.assert_close(
        output.float(), reference_output.float(), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        reference_kv_cache.float(), kv_data.float(), rtol=rtol, atol=atol
    )

    # Test trtllm_batch_context_with_kv_cache function
    seq_lens = flashinfer.page.get_seq_lens(
        kv_indptr_cpu, kv_last_page_len_cpu, page_size
    ).to(device)

    # Build block_tables using existing kv_indices_gpu
    blocks_per_seq = [
        (seq_len + page_size - 1) // page_size for seq_len in seq_lens.cpu()
    ]
    max_num_blocks_per_seq = max(blocks_per_seq)
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )
    block_id = kv_indptr_cpu[0]
    for i in range(batch_size):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = kv_indices_gpu[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    # Call trtllm_batch_context_with_kv_cache
    direct_output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=q,
        kv_cache=kv_data,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=qo_len,
        max_kv_len=kv_len,
        bmm1_scale=q_scale / math.sqrt(head_dim),
        bmm2_scale=1,
        batch_size=batch_size,
        cum_seq_lens_q=q_indptr_gpu,
        cum_seq_lens_kv=kv_indptr_gpu,
        window_left=window_left,
    )

    # Compare direct function output with wrapper output
    assert (direct_output == output).all()


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("window_left", [-1])  # todo(Siyuan): add 127 window_left
@pytest.mark.parametrize(
    "q_dtype,kv_cache_dtype,o_dtype",
    [
        ("half", "half", "half"),
        ("bf16", "bf16", "bf16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
    ],
)
def test_trtllm_batch_prefill(
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
    MAX_Q_LEN = 512
    MAX_IN_KV_LEN = 2048

    dtype_map = {
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
        "nvfp4": "nvfp4",
    }

    # Sequence lengths and block tables
    q_lens = torch.randint(1, MAX_Q_LEN, (batch_size,), dtype=torch.int32)
    q_lens[-1] = MAX_Q_LEN
    max_q_len = torch.max(q_lens).item()
    q_lens_tensor = q_lens.to(device)
    num_qo_heads = num_kv_heads * head_grp_size

    q = torch.randn(
        torch.sum(q_lens).item(),
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

    in_kv_lens = torch.randint(0, MAX_IN_KV_LEN, (batch_size,), dtype=torch.int)
    in_kv_lens[-1] = MAX_IN_KV_LEN
    max_in_kv_len = torch.max(in_kv_lens).item()
    seq_lens = in_kv_lens + q_lens
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
    q_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(q_lens_tensor, dim=0, dtype=torch.int32),
        ]
    )
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

    output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens_gpu,
        max_q_len,
        max_seq_len,
        q_scale * k_scale * sm_scale,  # bmm1_scale
        v_scale / o_scale,  # bmm2_scale
        batch_size,
        q_indptr,
        kv_indptr,
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

    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )

    # Calculate last page lengths
    kv_last_page_len = seq_lens_gpu % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size
    logits_soft_cap = 0.0

    wrapper.plan(
        q_indptr,
        kv_indptr,
        all_block_ids,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        logits_soft_cap=logits_soft_cap,
        q_data_type=ref_q.dtype,
        window_left=window_left,
    )
    output_ref = wrapper.run(ref_q, ref_kv_cache)

    if q_dtype == "fp8" and o_dtype == "nvfp4":
        rtol, atol = 4e-1, 1e0
    elif q_dtype == "fp8" and o_dtype == "fp8":
        rtol, atol = 5e-2, 7e-2
    else:
        rtol, atol = 1e-2, 1e-2

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
        wrapper2 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend="trtllm-gen"
        )
        wrapper2.plan(
            q_indptr,
            kv_indptr,
            all_block_ids,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=logits_soft_cap,
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
