import math

import numpy as np
import pytest
import torch

import flashinfer


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
@pytest.mark.parametrize("q_dtype", ["half", "bf16"])
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
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.bfloat16 if q_dtype == "bf16" else torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
    kv_data = kv_data_fp32.to(torch.bfloat16 if q_dtype == "bf16" else torch.float16)
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
        q_data_type=torch.bfloat16 if q_dtype == "bf16" else torch.float16,
        window_left=window_left,
    )
    reference_output = wrapper.run(q, kv_data)
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
        q_data_type=torch.bfloat16 if q_dtype == "bf16" else torch.float16,
        window_left=window_left,
    )
    output = wrapper2.run(q, kv_data)
    rmse = torch.sqrt(torch.mean((output - reference_output) ** 2))
    print(f"RMSE between output and reference_output: {rmse.item()}")
    rmse = torch.sqrt(torch.mean((reference_kv_cache - kv_data) ** 2))
    print(f"RMSE between reference_kv_cache and kv_data: {rmse.item()}")
    torch.testing.assert_close(output, reference_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(reference_kv_cache, kv_data, rtol=1e-2, atol=1e-2)
