"""
Copyright (c) 2026 by FlashInfer team.

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
from flashinfer import utils as flashinfer_utils
from flashinfer.utils import PosEncodingMode, has_flashinfer_jit_cache
from tests.test_helpers.jit_utils import (
    gen_decode_attention_modules,
    gen_prefill_attention_modules,
)


def _require_sm80_or_newer() -> None:
    major, _ = torch.cuda.get_device_capability(0)
    if major < 8:
        pytest.skip("int8 paged-kv coverage requires sm80 or newer")


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    _require_sm80_or_newer()
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],
            [torch.int8],
            [128],
            [0],
            [False],
            [False],
        )
        + gen_prefill_attention_modules(
            [torch.float16],
            [torch.int8],
            [128],
            [0],
            [False],
            [False],
            [False],
        ),
        verbose=False,
    )
    yield


def test_append_paged_kv_cache_int8():
    _require_sm80_or_newer()

    nnz_kv = 12
    num_kv_heads = 4
    head_dim = 128
    page_size = 4

    k_append = torch.randint(
        -16, 16, (nnz_kv, num_kv_heads, head_dim), dtype=torch.int8, device="cuda:0"
    )
    v_append = torch.randint(
        -16, 16, (nnz_kv, num_kv_heads, head_dim), dtype=torch.int8, device="cuda:0"
    )

    kv_append_length = torch.tensor([3, 5, 4], dtype=torch.int32, device="cuda:0")
    kv_append_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda:0"),
            torch.cumsum(kv_append_length, dim=0),
        ]
    )

    num_pages_per_req = torch.tensor([1, 2, 1], dtype=torch.int32, device="cuda:0")
    kv_page_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda:0"),
            torch.cumsum(num_pages_per_req, dim=0),
        ]
    )
    kv_page_indices = torch.arange(4, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor([3, 1, 4], dtype=torch.int32, device="cuda:0")

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )

    paged_kv_cache = torch.empty(
        8, 2, page_size, num_kv_heads, head_dim, dtype=torch.int8, device="cuda:0"
    )
    flashinfer.append_paged_kv_cache(
        k_append,
        v_append,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )

    batch_indices_cpu = batch_indices.cpu()
    positions_cpu = positions.cpu()
    kv_page_indptr_cpu = kv_page_indptr.cpu()
    kv_page_indices_cpu = kv_page_indices.cpu()
    for i in range(nnz_kv):
        batch_idx = int(batch_indices_cpu[i])
        position = int(positions_cpu[i])
        page_slot = position // page_size
        offset = position % page_size
        page_idx = int(
            kv_page_indices_cpu[int(kv_page_indptr_cpu[batch_idx]) + page_slot]
        )
        torch.testing.assert_close(paged_kv_cache[page_idx, 0, offset], k_append[i])
        torch.testing.assert_close(paged_kv_cache[page_idx, 1, offset], v_append[i])


def test_batch_decode_with_paged_kv_cache_int8():
    _require_sm80_or_newer()

    batch_size = 3
    kv_len = 9
    page_size = 4
    num_kv_heads = 2
    num_qo_heads = 2
    head_dim = 128
    k_scale = 0.125
    v_scale = 0.25

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = torch.randint(
        -8,
        8,
        (total_num_pages, 2, page_size, num_kv_heads, head_dim),
        device="cuda:0",
        dtype=torch.int8,
    )
    kv_data_ref = kv_data.to(torch.float16)
    kv_data_ref[:, 0].mul_(k_scale)
    kv_data_ref[:, 1].mul_(v_scale)

    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=torch.int8,
        q_data_type=torch.float16,
    )
    out = wrapper.run(q, kv_data, k_scale=k_scale, v_scale=v_scale)

    wrapper_ref = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
    )
    wrapper_ref.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    out_ref = wrapper_ref.run(q, kv_data_ref)

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=2e-2)


def test_batch_prefill_with_paged_kv_cache_int8():
    _require_sm80_or_newer()

    batch_size = 2
    kv_len = 8
    qo_len = 3
    page_size = 4
    num_kv_heads = 2
    num_qo_heads = 2
    head_dim = 128
    k_scale = 0.125
    v_scale = 0.25

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = torch.randint(
        -8,
        8,
        (total_num_pages, 2, page_size, num_kv_heads, head_dim),
        device="cuda:0",
        dtype=torch.int8,
    )
    kv_data_ref = kv_data.to(torch.float16)
    kv_data_ref[:, 0].mul_(k_scale)
    kv_data_ref[:, 1].mul_(v_scale)

    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=torch.float16,
        kv_data_type=torch.int8,
    )
    out = wrapper.run(q, kv_data, k_scale=k_scale, v_scale=v_scale)

    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
    )
    wrapper_ref.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    out_ref = wrapper_ref.run(q, kv_data_ref)

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_single_decode_with_kv_cache_int8(use_tensor_cores: bool):
    _require_sm80_or_newer()

    kv_len = 9
    num_kv_heads = 2
    num_qo_heads = 2
    head_dim = 128
    k_scale = 0.125
    v_scale = 0.25

    q = torch.randn(num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randint(
        -8,
        8,
        (kv_len, num_kv_heads, head_dim),
        device="cuda:0",
        dtype=torch.int8,
    )
    v = torch.randint(
        -8,
        8,
        (kv_len, num_kv_heads, head_dim),
        device="cuda:0",
        dtype=torch.int8,
    )
    k_ref = k.to(torch.float16) * k_scale
    v_ref = v.to(torch.float16) * v_scale

    out = flashinfer.single_decode_with_kv_cache(
        q,
        k,
        v,
        use_tensor_cores=use_tensor_cores,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    out_ref = flashinfer.single_decode_with_kv_cache(
        q,
        k_ref,
        v_ref,
        use_tensor_cores=use_tensor_cores,
    )

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=2e-2)


def test_single_prefill_with_kv_cache_int8():
    _require_sm80_or_newer()

    qo_len = 3
    kv_len = 8
    num_kv_heads = 2
    num_qo_heads = 2
    head_dim = 128
    scale_k = 0.125
    scale_v = 0.25

    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    k = torch.randint(
        -8,
        8,
        (kv_len, num_kv_heads, head_dim),
        device="cuda:0",
        dtype=torch.int8,
    )
    v = torch.randint(
        -8,
        8,
        (kv_len, num_kv_heads, head_dim),
        device="cuda:0",
        dtype=torch.int8,
    )
    k_ref = k.to(torch.float16) * scale_k
    v_ref = v.to(torch.float16) * scale_v

    out = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=False,
        scale_k=scale_k,
        scale_v=scale_v,
    )
    out_ref = flashinfer.single_prefill_with_kv_cache(
        q,
        k_ref,
        v_ref,
        causal=False,
    )

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=2e-2)


def test_determine_attention_backend_int8_falls_back_to_fa2_on_sm90(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(flashinfer_utils, "is_sm90a_supported", lambda device: True)

    backend_int8 = flashinfer_utils.determine_attention_backend(
        torch.device("cuda:0"),
        PosEncodingMode.NONE.value,
        False,
        False,
        torch.float16,
        torch.int8,
    )
    backend_fp16 = flashinfer_utils.determine_attention_backend(
        torch.device("cuda:0"),
        PosEncodingMode.NONE.value,
        False,
        False,
        torch.float16,
        torch.float16,
    )

    assert backend_int8 == "fa2"
    assert backend_fp16 == "fa3"
