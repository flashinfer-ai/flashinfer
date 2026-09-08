"""
Copyright (c) 2023 by FlashInfer team.

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

"""
Regression test for the FA2 FP8 KV-cache + GQA decode crash.

Root cause (fixed in ``include/flashinfer/attention/decode.cuh``):
``sync_state`` (``bdz > 1``, i.e. GQA) reuses the K+V shared-memory buffer as
float storage for ``st.o``, which needs ``bdz * bdy * head_dim * sizeof(float)``
bytes. The kernels, however, placed ``smem_md`` (and the batch-decode
``kv_offset_smem``) at a fixed offset that assumed the K+V buffer is only
``2 * num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
sizeof(DTypeKV)`` bytes. When FP8 (``sizeof(DTypeKV) == 1``) + GQA
(``tile_size_per_bdx == 1``) + SM75 (``num_stages_smem == 1``) all hold, the
float ``st.o`` is twice as large as the K+V buffer, so ``st.o`` overwrites
``smem_md`` and runs past the allocation (illegal smem write -> crash).

These tests exercise exactly that combination (FP8 KV cache + GQA) through the
public per-tensor-scale API and check the result against an fp16 reference.
They run on every architecture (validating correctness); the crash itself only
manifests on SM75 where ``num_stages_smem == 1``.
"""

import pytest
import torch

import flashinfer


# GQA: group_size = num_qo_heads // num_kv_heads = 8, so bdz > 1 and sync_state
# takes the smem-reuse path. This is the combination that triggers the bug.
_NUM_QO_HEADS = 32
_NUM_KV_HEADS = 4


@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_single_decode_fp8_gqa(kv_len, head_dim, fp8_dtype):
    torch.manual_seed(42)
    q = torch.randn(_NUM_QO_HEADS, head_dim, dtype=torch.float16).to(0)
    k = torch.randn(kv_len, _NUM_KV_HEADS, head_dim, dtype=torch.float16).to(0)
    v = 0.1 * torch.randn(kv_len, _NUM_KV_HEADS, head_dim, dtype=torch.float16).to(0)

    o_fp16 = flashinfer.single_decode_with_kv_cache(q, k, v)

    k_scale = k.amax().item() / 256
    v_scale = v.amax().item() / 256
    k_fp8 = (k / k_scale).to(fp8_dtype)
    v_fp8 = (v / v_scale).to(fp8_dtype)

    o_fp8 = flashinfer.single_decode_with_kv_cache(
        q, k_fp8, v_fp8, k_scale=k_scale, v_scale=v_scale
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=1e-2, rtol=2e-2)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_decode_paged_fp8_gqa(batch_size, kv_len, page_size, head_dim, fp8_dtype):
    torch.manual_seed(42)
    q = torch.randn(batch_size, _NUM_QO_HEADS, head_dim, dtype=torch.float16).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = 0.1 * torch.randn(
        total_num_pages, 2, _NUM_KV_HEADS, page_size, head_dim, dtype=torch.float16
    ).to(0)
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "HND")
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    o_fp16 = wrapper.run(q, kv_data)

    k_data, v_data = torch.chunk(kv_data, 2, dim=1)
    k_scale = k_data.amax().item() / 256
    v_scale = v_data.amax().item() / 256

    k_fp8 = (k_data / k_scale).to(fp8_dtype)
    v_fp8 = (v_data / v_scale).to(fp8_dtype)
    kv_data_fp8 = torch.cat([k_fp8, v_fp8], dim=1)

    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=fp8_dtype,
        q_data_type=torch.float16,
    )
    o_fp8 = wrapper.run(q, kv_data_fp8.to(fp8_dtype), k_scale=k_scale, v_scale=v_scale)

    torch.testing.assert_close(o_fp16, o_fp8, atol=1e-2, rtol=2e-1)


if __name__ == "__main__":
    test_single_decode_fp8_gqa(97, 128, torch.float8_e4m3fn)
    test_batch_decode_paged_fp8_gqa(12, 54, 16, 128, torch.float8_e5m2)
