import math

import pytest
import torch
from utils_fp4 import cast_from_fp4, recover_swizzled_scales, ref_fp4_quant

import flashinfer
from flashinfer.utils import FP4Tensor, ceil_div, round_up

DTYPE_MAP = {
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "nvfp4": "nvfp4",
}

GPU_DEVICE = "cuda:0"

global_workspace_buffer = None


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


def generate_seq_lens(batch_size, max_q_len, max_in_kv_len):
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    q_lens[-1] = max_q_len
    in_kv_lens = torch.randint(0, max_in_kv_len + 1, (batch_size,), dtype=torch.int)
    in_kv_lens[-1] = max_in_kv_len
    seq_lens = q_lens + in_kv_lens
    return q_lens, in_kv_lens, seq_lens


def generate_cumsum_lens(lens):
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=GPU_DEVICE),
            torch.cumsum(lens.to(GPU_DEVICE), dim=0, dtype=torch.int32),
        ]
    )


def create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype):
    q = torch.randn(
        torch.sum(q_lens).item(),
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16 if q_dtype == "fp8" else DTYPE_MAP[q_dtype],
        device=GPU_DEVICE,
    )
    if q_dtype == "fp8":
        q, q_scale = to_float8(q)
        # Reference implementation have functional issue or low precision with fp8, use bfloat16 and fake-quantization instead.
        ref_q = q.bfloat16() * q_scale
    else:
        q_scale = 1.0
        ref_q = q

    return q, q_scale, ref_q


def create_kv_cache(
    batch_size, seq_lens, page_size, num_kv_heads, head_dim, kv_dtype, ref_kv_dtype
):
    # Create separate K and V caches
    max_seq_len = torch.max(seq_lens).item()
    num_tokens = max_seq_len * batch_size
    num_pages = (num_tokens + page_size - 1) // page_size
    ref_kv_dtype_torch = DTYPE_MAP[ref_kv_dtype]
    if kv_dtype != "fp8":  # for fp8, create with high precision to generate scale.
        assert kv_dtype == ref_kv_dtype, (
            "kv_dtype and ref_kv_dtype must be the same for non-fp8 kv_cache"
        )

    k_cache = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=ref_kv_dtype_torch,
        device=GPU_DEVICE,
    )
    v_cache = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=ref_kv_dtype_torch,
        device=GPU_DEVICE,
    )

    # Convert K and V separately to fp8 if needed
    if kv_dtype == "fp8":
        k_cache, k_scale = to_float8(k_cache)
        v_cache, v_scale = to_float8(v_cache)
        # use high precision and fake-quantization for reference to avoid precision/functional issue
        ref_kv_cache = torch.stack(
            [
                k_cache.to(ref_kv_dtype_torch) * k_scale,
                v_cache.to(ref_kv_dtype_torch) * v_scale,
            ],
            dim=1,
        )
    else:
        k_scale = v_scale = 1.0
        ref_kv_cache = torch.stack([k_cache, v_cache], dim=1)
    # Combine K and V into interleaved format for the API
    kv_cache = torch.stack([k_cache, v_cache], dim=1)

    return kv_cache, k_scale, v_scale, ref_kv_cache


def create_page_table(batch_size, seq_lens, page_size):
    page_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_pages_per_seq = torch.max(page_per_seq).item()

    # Generate random but unique page IDs for all sequences
    total_pages_needed = torch.sum(page_per_seq).item()
    all_page_ids = torch.randperm(
        total_pages_needed, dtype=torch.int32, device=GPU_DEVICE
    )

    # Generate unique page IDs for all sequences
    page_tables = torch.zeros(
        (batch_size, max_num_pages_per_seq), dtype=torch.int32, device=GPU_DEVICE
    )

    # Populate page tables and track page assignments
    page_id = 0
    for i in range(batch_size):
        num_pages_needed = page_per_seq[i]
        page_tables[i, :num_pages_needed] = all_page_ids[
            page_id : page_id + num_pages_needed
        ]
        page_id += num_pages_needed
    return page_tables, all_page_ids, page_per_seq


def create_output(q, o_dtype, create_out_tensor):
    if o_dtype == "fp8":
        o_scale = torch.rand(1).item() * 0.5 + 0.5  # Scale range: 0.5 ~ 1.0
    else:
        o_scale = 1.0
    o_sf_scale = (
        300 if o_dtype == "nvfp4" else None
    )  # choose a value to make error smaller by testing.
    o_sf_vec_size = 16 if o_dtype == "nvfp4" else None

    if create_out_tensor:
        if o_dtype == "nvfp4":
            fp4_out_shape = q.shape[:-1] + (ceil_div(q.shape[-1], 2),)

            extra_size = torch.randint(0, 256, (1,)).item()

            fp4_out_scale_shape = (
                round_up(q.shape[0] + extra_size, 128),
                round_up(q.shape[1] * q.shape[2] // o_sf_vec_size, 4),
            )

            out_scale_factor = torch.empty(
                fp4_out_scale_shape, dtype=torch.float8_e4m3fn, device=q.device
            )
            rounded_extra_size = fp4_out_scale_shape[0] - q.shape[0]
            o_sf_start_index = (
                torch.randint(0, rounded_extra_size, (1,)).item()
                if rounded_extra_size > 0
                else 0
            )
            out_data = torch.empty(fp4_out_shape, dtype=torch.uint8, device=q.device)
            out = FP4Tensor(out_data, out_scale_factor, o_sf_start_index)
        else:
            out = torch.empty_like(q, dtype=DTYPE_MAP[o_dtype])
    else:
        out = None
    return out, o_scale, o_sf_scale, o_sf_vec_size


def get_last_page_len(seq_lens, page_size):
    kv_last_page_len = seq_lens % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size
    return kv_last_page_len


def unpack_compare_nvfp4(
    output: FP4Tensor,
    output_ref,
    o_sf_scale,
    o_sf_vec_size,
    sf_rtol=2e-1,
    sf_atol=2e-1,
    rmse_tol=0.3,
):
    output_ref, out_scale_factor_ref = ref_fp4_quant(
        output_ref, o_sf_scale, o_sf_vec_size
    )

    output_unpacked = cast_from_fp4(output.data)
    out_scale_factor = recover_swizzled_scales(
        output.scale,
        output_unpacked.shape[0],
        math.prod(list(output_unpacked.shape[1:])),
        o_sf_vec_size,
        output.scale_start_index,
    )

    torch.testing.assert_close(
        out_scale_factor.float().reshape(out_scale_factor_ref.shape),
        out_scale_factor_ref.float(),
        rtol=sf_rtol,
        atol=sf_atol,
    )
    rmse = torch.sqrt(torch.mean((output_unpacked.float() - output_ref.float()) ** 2))
    assert rmse.item() < rmse_tol
    return output_unpacked, output_ref


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("window_left", [-1])  # todo(Siyuan): add 127 window_left
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("half", "half", "half"),
        ("bf16", "bf16", "bf16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
def test_trtllm_batch_prefill(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
):
    # Set up test parameters
    torch.manual_seed(0)
    head_dim = 128
    MAX_Q_LEN = 511
    MAX_IN_KV_LEN = 2047

    # Generate random sequence lengths
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, in_kv_lens, seq_lens = generate_seq_lens(
        batch_size, MAX_Q_LEN, MAX_IN_KV_LEN
    )

    # Create query tensor and related data
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    q_indptr = generate_cumsum_lens(q_lens)

    # Create KV cache and related data
    kv_cache, k_scale, v_scale, ref_kv_cache = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" else q_dtype,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)

    # Create output tensor and related data
    create_out_tensor = flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype
    )
    out, o_scale, o_sf_scale, o_sf_vec_size = create_output(
        q, o_dtype, create_out_tensor
    )

    global global_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.int8, device=GPU_DEVICE
        )
    workspace_buffer = global_workspace_buffer

    # Run reference wrapper
    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    plan_params = {
        "qo_indptr": q_indptr,
        "paged_kv_indptr": kv_indptr,
        "paged_kv_indices": all_page_ids,
        "paged_kv_last_page_len": kv_last_page_len.to(GPU_DEVICE),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim_qk": head_dim,
        "page_size": page_size,
        "causal": True,
        "pos_encoding_mode": "NONE",
        "logits_soft_cap": 0.0,
        "q_data_type": ref_q.dtype,
        "kv_data_type": ref_kv_cache.dtype,
        "window_left": window_left,
    }
    wrapper_ref.plan(**plan_params)
    output_ref = wrapper_ref.run(ref_q, ref_kv_cache)

    # Run trtllm-gen function call
    sm_scale = float(1.0 / (head_dim**0.5))
    output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        page_table,
        seq_lens.to(GPU_DEVICE),
        torch.max(q_lens).item(),
        torch.max(seq_lens).item(),
        q_scale * k_scale * sm_scale,  # bmm1_scale
        v_scale / o_scale,  # bmm2_scale
        batch_size,
        q_indptr,
        kv_indptr,
        window_left,  # window_left
        out=out,
        out_dtype=DTYPE_MAP[o_dtype],
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        enable_pdl=enable_pdl,
    )

    if o_dtype == "nvfp4":
        output, output_ref = unpack_compare_nvfp4(
            output, output_ref, o_sf_scale, o_sf_vec_size
        )
        assert o_scale == 1.0
        rtol, atol = 4e-1, 1e0
    elif o_dtype == "fp8":
        rtol, atol = 5e-2, 7e-2
    else:
        rtol, atol = 1e-2, 1e-2

    # convert to float32 for fp8 is not supported by assert_close
    torch.testing.assert_close(
        output.float() * o_scale, output_ref.float(), rtol=rtol, atol=atol
    )

    if o_dtype != "nvfp4":  # wrapper api does not support fp4 output yet.
        # test wrapper with trtllm-gen backend
        wrapper_trtllm_gen = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend="trtllm-gen"
        )
        plan_params["q_data_type"] = q.dtype
        plan_params["kv_data_type"] = kv_cache.dtype
        wrapper_trtllm_gen.plan(**plan_params)
        output_wrapper = wrapper_trtllm_gen.run(
            q.contiguous(),
            kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale / o_scale,
            enable_pdl=enable_pdl,
        )
        # v_scale, o_scale in wrapper is emulated by multiplying output by v_scale instead of fused into kernel.
        if v_scale == o_scale == 1.0:
            assert (output_wrapper == output).all()
        else:
            torch.testing.assert_close(
                output.float(), output_wrapper.float(), rtol=1e-1, atol=1e-1
            )


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("window_left", [-1, 127])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("half", "half", "half"),
        ("half", "fp8", "half"),
        ("bf16", "bf16", "bf16"),
        ("bf16", "fp8", "bf16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
def test_trtllm_batch_decode(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
):
    # Set up test parameters
    torch.manual_seed(0)
    head_dim = 128
    MAX_Q_LEN = 1  # must be 1 for decode test
    MAX_IN_KV_LEN = 110

    # Generate random sequence lengths
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, in_kv_lens, seq_lens = generate_seq_lens(
        batch_size, MAX_Q_LEN, MAX_IN_KV_LEN
    )

    # Create query tensor and related data
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)

    # Create KV cache and related data
    kv_cache, k_scale, v_scale, ref_kv_cache = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" else q_dtype,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)

    # Create output tensor and related data
    create_out_tensor = flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype
    )
    out, o_scale, o_sf_scale, o_sf_vec_size = create_output(
        q, o_dtype, create_out_tensor
    )

    global global_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.int8, device=GPU_DEVICE
        )
    workspace_buffer = global_workspace_buffer

    # Run reference wrapper
    wrapper_ref = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    plan_params = {
        "indptr": kv_indptr,
        "indices": all_page_ids,
        "last_page_len": kv_last_page_len.to(GPU_DEVICE),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "pos_encoding_mode": "NONE",
        "kv_data_type": ref_kv_cache.dtype,
        "q_data_type": ref_q.dtype,
        "window_left": window_left,
    }
    wrapper_ref.plan(**plan_params)
    output_ref = wrapper_ref.run(ref_q, ref_kv_cache)

    # Run trtllm-gen function call
    sm_scale = float(1.0 / (head_dim**0.5))

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        page_table,
        seq_lens.to(GPU_DEVICE),
        torch.max(seq_lens).item(),
        q_scale * k_scale * sm_scale,  # bmm1_scale
        v_scale / o_scale,  # bmm2_scale
        window_left,  # window_left
        out=out,
        out_dtype=DTYPE_MAP[o_dtype],
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        enable_pdl=enable_pdl,
    )

    if o_dtype == "nvfp4":
        output, output_ref = unpack_compare_nvfp4(
            output, output_ref, o_sf_scale, o_sf_vec_size
        )
        assert o_scale == 1.0
        rtol, atol = 3e-1, 1e0
    elif o_dtype == "fp8":
        rtol, atol = 5e-2, 7e-2
    else:
        rtol, atol = 1e-2, 1e-2

    # convert to float32 for fp8 is not supported by assert_close
    torch.testing.assert_close(
        output.float() * o_scale, output_ref.float(), rtol=rtol, atol=atol
    )

    if o_dtype != "nvfp4":  # wrapper api does not support fp4 output yet.
        # test wrapper with trtllm-gen backend
        wrapper_trtllm_gen = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend="trtllm-gen"
        )
        plan_params["q_data_type"] = q.dtype
        plan_params["kv_data_type"] = kv_cache.dtype
        wrapper_trtllm_gen.plan(**plan_params)
        output_wrapper = wrapper_trtllm_gen.run(
            q.contiguous(),
            kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale / o_scale,
            enable_pdl=enable_pdl,
        )
        # v_scale, o_scale in wrapper is emulated by multiplying output by v_scale instead of fused into kernel.
        if v_scale == o_scale == 1.0:
            assert (output_wrapper == output).all()
        else:
            torch.testing.assert_close(
                output.float(), output_wrapper.float(), rtol=1e-1, atol=1e-1
            )


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

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

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
