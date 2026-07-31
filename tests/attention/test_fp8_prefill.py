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
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


def head_dim_512_supported() -> bool:
    # head_dim > 256 is only supported on SM100+.
    return get_compute_capability(torch.device("cuda:0"))[0] >= 10


def skip_if_head_dim_unsupported(head_dim: int):
    if head_dim > 256 and not head_dim_512_supported():
        pytest.skip("head_dim > 256 is only supported on SM100 or newer")


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("qo_len", [1, 7, 53])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    dtype,
):
    skip_if_head_dim_unsupported(head_dim)
    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16
    ).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        0.05
        * torch.randn(
            total_num_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.float16
        ).to(0)
        if kv_layout == "HND"
        else 0.05
        * torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
        ).to(0)
    )
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper_f16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_f16.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_fp16 = wrapper_f16.run(q, kv_data)
    k_data, v_data = torch.chunk(kv_data, 2, dim=1)
    k_scale = k_data.amax().item() / 256
    v_scale = v_data.amax().item() / 256

    k_fp8 = (k_data / k_scale).to(dtype)
    v_fp8 = (v_data / v_scale).to(dtype)
    kv_data_fp8 = torch.cat([k_fp8, v_fp8], dim=1)

    wrapper_f8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_f8.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=dtype,
    )
    o_fp8 = wrapper_f8.run(
        q,
        kv_data_fp8.to(dtype),
        k_scale=k_scale,
        v_scale=v_scale,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=1e-2, rtol=2e-1)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("qo_len", [7, 53])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_with_ragged_kv_cache_fp8(
    batch_size,
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    dtype,
):
    skip_if_head_dim_unsupported(head_dim)
    # Validates the ragged FP8 KV dequant kernel path (BF16 repack for hd128/256,
    # in-loop dequant for the k64B hd64 case) against the equivalent 16-bit kernel
    # run on the *same* dequantized values -- so no dependence on k/v scale
    # plumbing (the fa2 ragged wrapper does not apply k_scale/v_scale).
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16
    ).to(0)
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=torch.float16
    ).to(0)
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=torch.float16
    ).to(0)
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len

    # Quantize K/V to FP8 (values already fit the FP8 range; no scaling needed).
    k_fp8 = k.to(dtype)
    v_fp8 = v.to(dtype)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    # 16-bit reference on the dequantized FP8 values (same data, native path).
    wrapper_ref = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )
    wrapper_ref.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_ref = wrapper_ref.run(q, k_fp8.to(torch.float16), v_fp8.to(torch.float16))

    # FP8 KV path.
    wrapper_f8 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )
    wrapper_f8.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=torch.float16,
        kv_data_type=dtype,
    )
    o_fp8 = wrapper_f8.run(q, k_fp8, v_fp8)

    torch.testing.assert_close(o_fp8.to(torch.float16), o_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_decode_with_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    dtype,
):
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        0.1
        * torch.randn(
            total_num_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.float16
        ).to(0)
        if kv_layout == "HND"
        else 0.1
        * torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
        ).to(0)
    ).to(dtype)
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=dtype,
    )
    o_fp8 = wrapper.run(q, kv_data)

    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    decode_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=dtype,
    )
    o_decode_fp8 = decode_wrapper.run(q, kv_data)

    torch.testing.assert_close(o_decode_fp8, o_fp8, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# CTA_TILE_Q selection for FP8-KV head_dim=512 (gh #3843)
#
# FP8 h512 ragged/single prefill run on the non-VO-split SharedStorage; only
# the 2-Q x 2-KV-warp layout keeps its cross-warp merge buffer within the
# 101376B per-block limit of SM120/121-class GPUs at CTA_TILE_Q=32 (the 1x4
# layout needs 132176B and cannot launch there, and clamping to CTA_TILE_Q=16
# doubles the KV traversal).
# ---------------------------------------------------------------------------

_PLAN_INFO_CTA_TILE_Q_IDX = 3  # PrefillPlanInfo::ToVector layout (scheduler.cuh)


# 128: full Q tiles; 53: partial tail tile of 21 rows (the second Q-warp of
# the 2x2 layout gets a partial slice -- the failure mode of the old generic
# 2x2 layout removed in gh #523); 40: partial tail tile of 8 rows (the second
# Q-warp gets no rows).
@pytest.mark.parametrize("qo_len", [128, 53, 40])
def test_ragged_fp8_h512_long_q_keeps_cta32(qo_len):
    head_dim = 512
    skip_if_head_dim_unsupported(head_dim)
    torch.manual_seed(42)
    batch_size = 2
    kv_len = 128
    num_qo_heads = num_kv_heads = 4
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16
    ).to(0)
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=torch.float16
    ).to(0)
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=torch.float16
    ).to(0)
    k_fp8 = k.to(torch.float8_e4m3fn)
    v_fp8 = v.to(torch.float8_e4m3fn)
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper_ref = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )
    wrapper_ref.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_ref = wrapper_ref.run(q, k_fp8.to(torch.float16), v_fp8.to(torch.float16))

    wrapper_f8 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )
    wrapper_f8.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
    )
    # Long q must keep CTA_TILE_Q=32: 16 would double the KV traversal, and a
    # regression to the 1x4 layout at 32 fails to launch on 99KB-smem GPUs
    # ("Required shared memory (132176 bytes) exceeds ...").
    assert wrapper_f8._plan_info[_PLAN_INFO_CTA_TILE_Q_IDX] == 32
    o_fp8 = wrapper_f8.run(q, k_fp8, v_fp8)
    torch.testing.assert_close(o_fp8.to(torch.float16), o_ref, atol=1e-2, rtol=1e-2)


def test_paged_fp8_h512_long_q_keeps_cta32():
    head_dim = 512
    skip_if_head_dim_unsupported(head_dim)
    torch.manual_seed(42)
    batch_size = 2
    qo_len = kv_len = 128
    page_size = 16
    num_qo_heads = num_kv_heads = 4
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16
    ).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data_fp8 = (
        torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
        )
        .to(0)
        .to(torch.float8_e4m3fn)
    )
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
    )
    assert wrapper._plan_info[_PLAN_INFO_CTA_TILE_Q_IDX] == 32
    o = wrapper.run(q, kv_data_fp8)
    assert torch.isfinite(o).all()


def test_single_prefill_fp8_h512_long_q():
    head_dim = 512
    skip_if_head_dim_unsupported(head_dim)
    torch.manual_seed(42)
    qo_len = 100  # 3 full CTA_TILE_Q=32 tiles plus a 4-row partial tile
    kv_len = 128
    num_qo_heads = num_kv_heads = 4
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.float16).to(0)
    k_fp8 = (
        torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16)
        .to(0)
        .to(torch.float8_e4m3fn)
    )
    v_fp8 = (
        torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16)
        .to(0)
        .to(torch.float8_e4m3fn)
    )
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_fp8.to(torch.float16), v_fp8.to(torch.float16)
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(q, k_fp8, v_fp8)
    torch.testing.assert_close(o_fp8.to(torch.float16), o_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale(
        12, 7, 54, 1, 4, 4, 128, "NHD", torch.float8_e5m2
    )
    test_batch_decode_with_prefill_with_paged_kv_cache(
        12, 54, 1, 4, 4, 128, "NHD", torch.float8_e5m2
    )
