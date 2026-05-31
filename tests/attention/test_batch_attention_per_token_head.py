"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

BatchAttention (persistent holistic attention) with FP8 per-token-head KV cache tests.
"""

import pytest
import torch
import flashinfer
from tests.utils_fp8 import get_cos_sim_threshold, to_float8_per_token_head


# ============================================================
# Helpers
# ============================================================


def _cc():
    return torch.cuda.get_device_capability(0)


def _skip_if_sm_below_75():
    if _cc()[0] < 7 or (_cc()[0] == 7 and _cc()[1] < 5):
        pytest.skip("Persistent batch attention requires SM75+")


def _skip_if_sm75_limits(head_dim, dtype: torch.dtype, kv_dtype: torch.dtype):
    if _cc()[0] > 7:
        return
    if dtype != torch.float16:
        pytest.skip(f"{dtype} skipped on SM75")
    if head_dim >= 256:
        pytest.skip(
            f"BatchAttention CTA128 exceeds SM75 64KiB shared memory limit for hd={head_dim}"
        )


def check_accuracy(
    o_ref: torch.Tensor,
    o: torch.Tensor,
    kv_dtype: torch.dtype,
    mode,
    label="",
):
    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.reshape(-1).float(), o.reshape(-1).float(), dim=0
    ).item()
    max_diff = (o_ref - o).abs().max().item()
    prefix = f"[{label}] " if label else ""
    threshold = get_cos_sim_threshold(kv_dtype, mode)
    print(f"{prefix}{kv_dtype} cos_sim={cos_sim:.8f} max_diff={max_diff:.8f}")
    assert cos_sim >= threshold, (
        f"{prefix}cos_sim={cos_sim:.8f} < {threshold} ({kv_dtype})"
    )
    return cos_sim, max_diff


def _alloc_paged_cache(shape, head_dim, dtype, device):
    max_pages, page_size, num_kv_heads = shape[0], shape[1], shape[2]
    total_tokens = max_pages * page_size * num_kv_heads
    stride = head_dim + 16
    buf_size = total_tokens * stride
    k_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)
    v_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)
    s = (page_size * num_kv_heads * stride, num_kv_heads * stride, stride, 1)
    k_cache = torch.as_strided(k_buf, shape, s).view(dtype)
    v_cache = torch.as_strided(v_buf, shape, s).view(dtype)
    return k_cache, v_cache, k_buf, v_buf


def _write_scales(cache_tensor, buf, scales, head_dim):
    assert head_dim % 4 == 0, (
        f"head_dim={head_dim} must be divisible by 4 for float32 offset"
    )
    stride = head_dim + 16
    assert stride % 4 == 0, f"stride={stride} must be divisible by 4 for float32 stride"
    scale_stride_f32 = stride // 4
    scale_offset_f32 = head_dim // 4
    max_pages, page_size, num_kv_heads = (
        cache_tensor.shape[0],
        cache_tensor.shape[1],
        cache_tensor.shape[2],
    )
    s = (
        page_size * num_kv_heads * scale_stride_f32,
        num_kv_heads * scale_stride_f32,
        scale_stride_f32,
    )
    scale_view = torch.as_strided(
        buf.view(torch.float32),
        (max_pages, page_size, num_kv_heads),
        s,
        storage_offset=scale_offset_f32,
    )
    scale_view.copy_(scales.to(torch.float32).reshape(scale_view.shape))


def _build_pth_data(
    k_ref, v_ref, total_num_pages, page_size, num_kv_heads, head_dim, kv_dtype, device
):
    k_flat = k_ref.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_ref.reshape(-1, num_kv_heads, head_dim)
    k_fp8, k_scales = to_float8_per_token_head(k_flat, kv_dtype)
    v_fp8, v_scales = to_float8_per_token_head(v_flat, kv_dtype)

    cache_shape = (total_num_pages, page_size, num_kv_heads, head_dim)
    k_cache, v_cache, k_buf, v_buf = _alloc_paged_cache(
        cache_shape, head_dim, kv_dtype, device
    )

    k_paged_fp8 = k_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    v_paged_fp8 = v_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    k_scales_paged = k_scales.reshape(total_num_pages, page_size, num_kv_heads)
    v_scales_paged = v_scales.reshape(total_num_pages, page_size, num_kv_heads)

    k_cache.copy_(k_paged_fp8)
    v_cache.copy_(v_paged_fp8)
    _write_scales(k_cache, k_buf, k_scales_paged, head_dim)
    _write_scales(v_cache, v_buf, v_scales_paged, head_dim)
    return k_cache, v_cache


# ============================================================
# Test
# ============================================================


@pytest.mark.parametrize("batch_size", [3, 4, 5])
@pytest.mark.parametrize(
    "qo_len,kv_len", [(8, 32), (128, 2048)], ids=["small", "large"]
)
@pytest.mark.parametrize(
    "num_kv_heads,num_qo_heads",
    [(1, 1), (2, 4), (1, 16), (4, 4), (4, 32)],
    ids=["1x1", "gqa4x2", "16x1", "mha4", "gqa32x4"],
)
@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=["hd64", "hd128", "hd256"])
@pytest.mark.parametrize("page_size", [16, 32], ids=["ps16", "ps32"])
@pytest.mark.parametrize("causal", [False, True], ids=["no-causal", "causal"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
def test_batch_attention_per_token_head(
    batch_size,
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
    causal,
    dtype,
    kv_dtype,
):
    _skip_if_sm_below_75()
    _skip_if_sm75_limits(head_dim, dtype, kv_dtype)
    device = "cuda:0"
    kv_layout = "NHD"

    total_qo = batch_size * qo_len
    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=dtype, device=device)
    k_ref = 0.1 * torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_ref = 0.1 * torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )
    kv_len_arr = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    # Baseline via BatchPrefill
    k_paged = k_ref.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    v_paged = v_ref.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    bp_base = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout, backend="fa2"
    )
    bp_base.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_baseline = bp_base.run(q, (k_paged, v_paged))

    # FP8 per-token-head via BatchAttention
    k_cache, v_cache = _build_pth_data(
        k_ref,
        v_ref,
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        device,
    )
    ba_pth = flashinfer.BatchAttention(
        kv_layout=kv_layout,
        use_per_token_head=True,
    )
    ba_pth.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=kv_dtype,
    )
    o_ba_pth, _ = ba_pth.run(q, (k_cache, v_cache))
    assert not torch.isnan(o_ba_pth).any(), (
        "BatchAttention FP8 per-token-head output contains NaN"
    )
    check_accuracy(o_baseline, o_ba_pth, kv_dtype, "prefill", label="batch attention")


# ============================================================
# Smoke tests
# ============================================================

if __name__ == "__main__":
    dtypes = [torch.float16]
    if _cc()[0] > 7:
        dtypes.append(torch.bfloat16)
    for dtype in dtypes:
        kv_dtype = torch.float8_e4m3fn

        test_batch_attention_per_token_head(
            batch_size=4,
            qo_len=8,
            kv_len=32,
            num_kv_heads=4,
            num_qo_heads=4,
            head_dim=128,
            page_size=16,
            causal=True,
            dtype=dtype,
            kv_dtype=kv_dtype,
        )
        print(f"MHA {dtype}/{kv_dtype} causal smoke passed")

        test_batch_attention_per_token_head(
            batch_size=3,
            qo_len=128,
            kv_len=2048,
            num_kv_heads=2,
            num_qo_heads=32,
            head_dim=64,
            page_size=32,
            causal=False,
            dtype=dtype,
            kv_dtype=kv_dtype,
        )
        print(f"GQA {dtype}/{kv_dtype} no-causal smoke passed")

    print("\nAll batch attention per-token-head smoke tests passed")
