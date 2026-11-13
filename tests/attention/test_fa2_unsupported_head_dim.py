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


def _paged_kv_metadata(device, batch_size=4, pages_per_req=2, page_size=16):
    kv_indptr = torch.arange(
        0, batch_size * pages_per_req + 1, pages_per_req, dtype=torch.int32
    ).to(device)
    kv_indices = torch.arange(batch_size * pages_per_req, dtype=torch.int32).to(device)
    kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32).to(
        device
    )
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32).to(device) * 4
    return qo_indptr, kv_indptr, kv_indices, kv_last_page_len, page_size


# The FA2 prefill kernels copy each Q / K / V row in whole column blocks of 64
# (16-bit) or 128 (FP8 KV) elements. These head dims are not a multiple of the
# block, so part of each row was never loaded and the kernels returned NaN or
# faulted with an illegal address. They must be refused when the module is
# generated (at plan time) instead.
_UNSUPPORTED = [
    (torch.bfloat16, 80),
    (torch.bfloat16, 96),
    (torch.float8_e4m3fn, 80),
    (torch.float8_e4m3fn, 96),
    (torch.float8_e4m3fn, 192),
]


@pytest.mark.parametrize("kv_dtype, head_dim", _UNSUPPORTED)
def test_tensor_core_decode_refuses_head_dim(kv_dtype, head_dim):
    device = "cuda:0"
    _, kv_indptr, kv_indices, kv_last_page_len, page_size = _paged_kv_metadata(device)
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True
    )
    with pytest.raises(ValueError, match=f"head_dim_qk={head_dim}"):
        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            8,
            2,
            head_dim,
            page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dtype,
        )


# (kv_dtype, head_dim_qk, head_dim_vo, the dimension the error names)
_UNSUPPORTED_QK_VO = [
    (dtype, d, d, f"head_dim_qk={d}") for dtype, d in _UNSUPPORTED
] + [
    # FP8 K rows of 192 are loaded only up to column 128: wrong output, no error.
    (torch.float8_e4m3fn, 192, 128, "head_dim_qk=192"),
    # head_dim_vo == 64 selects the 64B swizzle, which only fits 64-element K rows.
    (torch.float8_e4m3fn, 192, 64, "head_dim_qk=192"),
    # A valid head_dim_qk must not hide an invalid head_dim_vo.
    (torch.bfloat16, 128, 96, "head_dim_vo=96"),
]


@pytest.mark.parametrize(
    "kv_dtype, head_dim_qk, head_dim_vo, named", _UNSUPPORTED_QK_VO
)
def test_fa2_paged_prefill_refuses_head_dim(kv_dtype, head_dim_qk, head_dim_vo, named):
    device = "cuda:0"
    qo_indptr, kv_indptr, kv_indices, kv_last_page_len, page_size = _paged_kv_metadata(
        device
    )
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match=named):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            8,
            2,
            head_dim_qk,
            page_size,
            head_dim_vo=head_dim_vo,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dtype,
        )
