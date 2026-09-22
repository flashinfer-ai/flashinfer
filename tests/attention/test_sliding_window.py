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

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import flashinfer
from flashinfer.jit.attention.modules import _gen_batch_prefill_primary_module
from flashinfer.utils import (
    get_compute_capability,
    has_flashinfer_jit_cache,
    is_sm90a_supported,
)


def head_dim_512_supported() -> bool:
    # 16-bit FA2 head_dim > 256 uses the Ampere+ large-head path.
    return get_compute_capability(torch.device("cuda:0"))[0] >= 8


def skip_if_head_dim_unsupported(head_dim: int):
    if head_dim > 256 and not head_dim_512_supported():
        pytest.skip("16-bit FA2 head_dim > 256 is only supported on SM80 or newer")


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    """Prebuild, in parallel, exactly the modules this file loads.

    Per head_dim: single/batch decode with and without a sliding window, FA2
    single prefill and the FA2 primary (equal K/V stride) batch prefill module
    that the wrappers load, and, on SM90a, the FA3 single/batch prefill modules
    that backend="auto" resolves to (FA3 has no head_dim 512). Specs already in
    the AOT cache are skipped.

    Each spec is built with its own ninja (in parallel) rather than through
    flashinfer.jit.build_jit_specs: that helper's combined ninja logs to the
    cached_ops root, so the per-module ninja that build_and_load() later runs
    finds no record of the outputs and recompiles them.
    """
    f16, i32 = torch.float16, torch.int32
    NONE = 0
    head_dims = [64, 128, 256] + ([512] if head_dim_512_supported() else [])
    fa3 = is_sm90a_supported(torch.device("cuda:0"))
    specs = []
    for head_dim in head_dims:
        for swa in (False, True):
            specs.append(
                flashinfer.decode.gen_single_decode_module(
                    f16, f16, f16, head_dim, head_dim, NONE, swa, False
                )
            )
            specs.append(
                flashinfer.decode.gen_batch_decode_module(
                    f16, f16, f16, i32, head_dim, head_dim, NONE, swa, False
                )
            )
            specs.append(
                flashinfer.prefill.gen_single_prefill_module(
                    "fa2", f16, f16, f16, head_dim, head_dim, NONE, swa, False, False
                )
            )
            specs.append(
                _gen_batch_prefill_primary_module(
                    "fa2",
                    f16,
                    f16,
                    f16,
                    i32,
                    head_dim,
                    head_dim,
                    NONE,
                    swa,
                    False,
                    False,
                )
            )
            if fa3 and head_dim <= 256:
                specs.append(
                    flashinfer.prefill.gen_single_prefill_module(
                        "fa3",
                        f16,
                        f16,
                        f16,
                        head_dim,
                        head_dim,
                        NONE,
                        swa,
                        False,
                        False,
                    )
                )
                specs.append(
                    flashinfer.prefill.gen_batch_prefill_module(
                        "fa3",
                        f16,
                        f16,
                        f16,
                        i32,
                        head_dim,
                        head_dim,
                        NONE,
                        swa,
                        False,
                        False,
                    )
                )

    to_build = [spec for spec in specs if not spec.is_aot]
    if to_build:
        workers = min(len(to_build), max(1, (os.cpu_count() or 8) // 8))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda spec: spec.build(need_lock=True), to_build))
    yield


@pytest.mark.parametrize("seq_len", [1, 3, 19, 99, 199, 1177, 1999])
@pytest.mark.parametrize("window_left", [3, 13, 23, 37, 43])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
def test_single_decode_sliding_window(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    skip_if_head_dim_unsupported(head_dim)
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
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("backend", ["fa2", "auto"])
def test_batch_decode_sliding_window(
    batch_size,
    kv_len,
    window_left,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
    backend,
):
    skip_if_head_dim_unsupported(head_dim)
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
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend=backend
    )
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


@pytest.mark.parametrize("seq_len", [1, 3, 19, 99, 199, 1999])
@pytest.mark.parametrize("window_left", [3, 13, 23, 43])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_single_decode_prefill_sliding_window_match(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = torch.randn(1, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    v = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, window_left=window_left, causal=True
    )
    o_decoded = flashinfer.single_decode_with_kv_cache(
        q[0], k, v, window_left=window_left
    )
    torch.testing.assert_close(o.cpu()[0], o_decoded.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("seq_len", [99, 199, 1999])
@pytest.mark.parametrize("window_left", [43, 233])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
def test_single_prefill_sliding_window(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    skip_if_head_dim_unsupported(head_dim)
    q = torch.randn(
        seq_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    k = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    v = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )

    row_idx = torch.arange(seq_len, dtype=torch.int32, device="cuda:0")[:, None]
    col_idx = torch.arange(seq_len, dtype=torch.int32, device="cuda:0")[None, :]
    mask = (row_idx >= col_idx) & (row_idx - window_left <= col_idx)

    o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, window_left=window_left, causal=True
    )
    torch.testing.assert_close(o.cpu(), o_ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 30])
@pytest.mark.parametrize("kv_len", [54, 397, 1177])
@pytest.mark.parametrize("qo_len", [1, 37, 47])
@pytest.mark.parametrize("window_left", [13, 33, 111])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("backend", ["fa2", "auto"])
def test_batch_paged_prefill_sliding_window(
    batch_size,
    kv_len,
    qo_len,
    window_left,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
    backend,
):
    skip_if_head_dim_unsupported(head_dim)
    if num_qo_heads < num_kv_heads:
        pytest.skip("num_qo_heads < num_kv_heads is not supported")

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
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

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend=backend
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
        window_left=window_left,
        causal=True,
    )
    o = wrapper.run(
        q,
        (k_data, v_data),
    )

    for i in range(batch_size):
        qi = q[q_indptr[i] : q_indptr[i + 1]]
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
        o_ref_i = flashinfer.single_prefill_with_kv_cache(
            qi, ki, vi, window_left=window_left, causal=True, backend="fa2"
        )
        o_i = o[q_indptr[i] : q_indptr[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 397])
@pytest.mark.parametrize("qo_len", [37, 47])
@pytest.mark.parametrize("window_left", [13, 33])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("backend", ["fa2", "auto"])
def test_batch_ragged_prefill_sliding_window(
    batch_size,
    kv_len,
    qo_len,
    window_left,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    backend,
):
    skip_if_head_dim_unsupported(head_dim)
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    k = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    v = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, "NHD", backend=backend
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        window_left=window_left,
        causal=True,
    )
    o = wrapper.run(q, k, v)

    for i in range(batch_size):
        qi = q[q_indptr[i] : q_indptr[i + 1]]
        ki = k[kv_indptr[i] : kv_indptr[i + 1]]
        vi = v[kv_indptr[i] : kv_indptr[i + 1]]
        o_ref_i = flashinfer.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            window_left=window_left,
            causal=True,
        )
        o_i = o[q_indptr[i] : q_indptr[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_single_decode_sliding_window(13, 20, 1, 4, 128)
    test_single_prefill_sliding_window(13, 20, 1, 4, 128)
    test_batch_paged_prefill_sliding_window(12, 54, 37, 13, 1, 4, 128, 1)
    test_batch_ragged_prefill_sliding_window(12, 54, 37, 13, 1, 4, 128)
