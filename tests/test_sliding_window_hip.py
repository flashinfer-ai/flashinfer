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
from jit_utils import jit_decode_attention_func_args

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if flashinfer.jit.has_prebuilt_ops:
        yield
    else:
        try:
            flashinfer.jit.parallel_load_modules(
                jit_decode_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [64, 128, 256],  # head_dims
                    [0],  # pos_encoding_modes
                    [False, True],  # use_sliding_windows
                    [False],  # use_logits_soft_caps
                )
            )
        except Exception as e:
            # abort the test session if warmup fails
            pytest.exit(str(e))
        finally:
            yield


@pytest.mark.parametrize("seq_len", [1, 3, 19, 99, 199, 1999])
@pytest.mark.parametrize("window_left", [3, 13, 23, 43])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_single_decode_sliding_window(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    v = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )

    k_sliced = k[-(window_left + 1) :]
    v_sliced = v[-(window_left + 1) :]

    o_ref = flashinfer.single_decode_with_kv_cache(q, k_sliced, v_sliced)
    o = flashinfer.single_decode_with_kv_cache(q, k, v, window_left=window_left)

    torch.testing.assert_close(o.cpu(), o_ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3, 13, 32])
@pytest.mark.parametrize("kv_len", [1, 3, 99, 199, 1999])
@pytest.mark.parametrize("window_left", [33, 533])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("page_size", [1, 16])
def test_batch_decode_sliding_window(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size
):
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    k_data = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    v_data = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        window_left=window_left,
    )
    o = wrapper.run(q, (k_data, v_data))

    for i in range(batch_size):
        qi = q[i]
        ki = torch.cat(
            [
                k_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                k_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                v_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                v_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            dim=0,
        )
        o_ref_i = flashinfer.single_decode_with_kv_cache(
            qi,
            ki,
            vi,
            window_left=window_left,
        )
        torch.testing.assert_close(o[i], o_ref_i, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_single_decode_sliding_window(13, 20, 1, 4, 128)
    test_batch_decode_sliding_window(3, 199, 33, 1, 4, 128, 16)
