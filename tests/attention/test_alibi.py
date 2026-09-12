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

import itertools

import pytest
import torch
from tests.test_helpers.alibi_reference import alibi_attention
from tests.test_helpers.jit_utils import (
    gen_decode_attention_modules,
    gen_prefill_attention_modules,
)

import flashinfer
from flashinfer.utils import get_alibi_slopes, has_flashinfer_jit_cache


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0, 2],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0, 2],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("seq_len", [1, 9, 81, 729])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_single_decode_alibi(
    seq_len,
    num_heads,
    head_dim,
):
    q = torch.randn(num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randn(seq_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn(seq_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)

    o = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ALIBI")
    mask = torch.ones(1, seq_len, dtype=torch.bool, device="cuda:0")
    o_ref = alibi_attention(q.unsqueeze(0), k, v, mask).squeeze(0)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("q_len", [1, 17, 81, 987])
@pytest.mark.parametrize("kv_len", [1, 17, 81, 987])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_alibi(
    q_len,
    kv_len,
    num_heads,
    head_dim,
    causal,
):
    if causal and q_len > kv_len:
        pytest.skip("Causal attention requires q_len <= kv_len")
    q = torch.randn(q_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)

    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, pos_encoding_mode="ALIBI"
    )
    mask = torch.ones(q_len, kv_len, dtype=torch.bool, device="cuda:0")
    if causal:
        mask = torch.tril(mask, diagonal=kv_len - q_len)
    o_ref = alibi_attention(q, k, v, mask)
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


def _tp_slopes(num_global_heads: int, tp_size: int, rank: int) -> torch.Tensor:
    """This rank's slice of the slopes computed for the global head count."""
    slopes = get_alibi_slopes(num_global_heads, device="cuda:0")
    num_local_heads = num_global_heads // tp_size
    return slopes[rank * num_local_heads : (rank + 1) * num_local_heads]


def _causal_mask(q_len: int, kv_len: int) -> torch.Tensor:
    mask = torch.ones(q_len, kv_len, dtype=torch.bool, device="cuda:0")
    return torch.tril(mask, diagonal=kv_len - q_len)


def _build_paged_kv(kv_lens, num_heads, head_dim, page_size):
    """Random paged K/V for ``kv_lens`` plus the dense per-request views."""
    num_pages = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    total_pages = sum(num_pages)
    k_cache = torch.randn(
        total_pages,
        page_size,
        num_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    v_cache = torch.randn_like(k_cache)
    indptr = [0] + list(itertools.accumulate(num_pages))
    kv_indptr = torch.tensor(indptr, dtype=torch.int32, device="cuda:0")
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor(
        [(kv_len - 1) % page_size + 1 for kv_len in kv_lens],
        dtype=torch.int32,
        device="cuda:0",
    )
    dense_k, dense_v = [], []
    for i, kv_len in enumerate(kv_lens):
        pages = slice(indptr[i], indptr[i + 1])
        dense_k.append(k_cache[pages].reshape(-1, num_heads, head_dim)[:kv_len])
        dense_v.append(v_cache[pages].reshape(-1, num_heads, head_dim)[:kv_len])
    return k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, dense_k, dense_v


@pytest.mark.parametrize("num_global_heads", [8, 12, 32])
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_single_decode_alibi_slopes_tp(
    num_global_heads, tp_size, head_dim, use_tensor_cores
):
    kv_len = 81
    num_local_heads = num_global_heads // tp_size
    for rank in range(tp_size):
        slopes = _tp_slopes(num_global_heads, tp_size, rank)
        q = torch.randn(num_local_heads, head_dim, device="cuda:0", dtype=torch.float16)
        k = torch.randn(
            kv_len, num_local_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn_like(k)
        o = flashinfer.single_decode_with_kv_cache(
            q,
            k,
            v,
            pos_encoding_mode="ALIBI",
            use_tensor_cores=use_tensor_cores,
            alibi_slopes=slopes,
        )
        mask = torch.ones(1, kv_len, dtype=torch.bool, device="cuda:0")
        o_ref = alibi_attention(q.unsqueeze(0), k, v, mask, slopes=slopes).squeeze(0)
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("q_len,kv_len", [(17, 81), (128, 128), (81, 17)])
@pytest.mark.parametrize("num_global_heads", [8, 12, 32])
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_alibi_slopes_tp(
    q_len, kv_len, num_global_heads, tp_size, causal
):
    if causal and q_len > kv_len:
        pytest.skip("Causal attention requires q_len <= kv_len")
    head_dim = 128
    num_local_heads = num_global_heads // tp_size
    for rank in range(tp_size):
        slopes = _tp_slopes(num_global_heads, tp_size, rank)
        q = torch.randn(
            q_len, num_local_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        k = torch.randn(
            kv_len, num_local_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn_like(k)
        o = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=causal, pos_encoding_mode="ALIBI", alibi_slopes=slopes
        )
        mask = (
            _causal_mask(q_len, kv_len)
            if causal
            else torch.ones(q_len, kv_len, dtype=torch.bool, device="cuda:0")
        )
        o_ref = alibi_attention(q, k, v, mask, slopes=slopes)
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("num_global_heads", [8, 12, 32])
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_batch_decode_alibi_slopes_tp(num_global_heads, tp_size, use_tensor_cores):
    head_dim, page_size = 128, 16
    kv_lens = [37, 5, 64]
    num_local_heads = num_global_heads // tp_size
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    for rank in range(tp_size):
        slopes = _tp_slopes(num_global_heads, tp_size, rank)
        k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, dense_k, dense_v = (
            _build_paged_kv(kv_lens, num_local_heads, head_dim, page_size)
        )
        q = torch.randn(
            len(kv_lens),
            num_local_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=use_tensor_cores
        )
        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_local_heads,
            num_local_heads,
            head_dim,
            page_size,
            pos_encoding_mode="ALIBI",
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            alibi_slopes=slopes,
        )
        o = wrapper.run(q, (k_cache, v_cache))
        for i, kv_len in enumerate(kv_lens):
            mask = torch.ones(1, kv_len, dtype=torch.bool, device="cuda:0")
            o_ref = alibi_attention(
                q[i : i + 1], dense_k[i], dense_v[i], mask, slopes=slopes
            )
            torch.testing.assert_close(o[i : i + 1], o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_global_heads", [8, 12, 32])
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_paged_alibi_slopes_tp(
    num_global_heads, tp_size, page_size, causal
):
    head_dim = 128
    qo_lens = [17, 1, 33]
    kv_lens = [37, 5, 64]
    num_local_heads = num_global_heads // tp_size
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    qo_indptr = torch.tensor(
        [0] + list(itertools.accumulate(qo_lens)), dtype=torch.int32, device="cuda:0"
    )
    for rank in range(tp_size):
        slopes = _tp_slopes(num_global_heads, tp_size, rank)
        k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, dense_k, dense_v = (
            _build_paged_kv(kv_lens, num_local_heads, head_dim, page_size)
        )
        q = torch.randn(
            sum(qo_lens),
            num_local_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_local_heads,
            num_local_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode="ALIBI",
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            alibi_slopes=slopes,
        )
        o = wrapper.run(q, (k_cache, v_cache))
        for i, (q_len, kv_len) in enumerate(zip(qo_lens, kv_lens, strict=True)):
            rows = slice(int(qo_indptr[i]), int(qo_indptr[i + 1]))
            mask = (
                _causal_mask(q_len, kv_len)
                if causal
                else torch.ones(q_len, kv_len, dtype=torch.bool, device="cuda:0")
            )
            o_ref = alibi_attention(
                q[rows], dense_k[i], dense_v[i], mask, slopes=slopes
            )
            torch.testing.assert_close(o[rows], o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("num_global_heads", [8, 12, 32])
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_ragged_alibi_slopes_tp(num_global_heads, tp_size, causal):
    head_dim = 128
    qo_lens = [17, 1, 33]
    kv_lens = [37, 5, 64]
    num_local_heads = num_global_heads // tp_size
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    qo_indptr = torch.tensor(
        [0] + list(itertools.accumulate(qo_lens)), dtype=torch.int32, device="cuda:0"
    )
    kv_indptr = torch.tensor(
        [0] + list(itertools.accumulate(kv_lens)), dtype=torch.int32, device="cuda:0"
    )
    for rank in range(tp_size):
        slopes = _tp_slopes(num_global_heads, tp_size, rank)
        q = torch.randn(
            sum(qo_lens),
            num_local_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        k = torch.randn(
            sum(kv_lens),
            num_local_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        v = torch.randn_like(k)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_local_heads,
            num_local_heads,
            head_dim,
            causal=causal,
            pos_encoding_mode="ALIBI",
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            alibi_slopes=slopes,
        )
        o = wrapper.run(q, k, v)
        for i, (q_len, kv_len) in enumerate(zip(qo_lens, kv_lens, strict=True)):
            rows = slice(int(qo_indptr[i]), int(qo_indptr[i + 1]))
            kv_rows = slice(int(kv_indptr[i]), int(kv_indptr[i + 1]))
            mask = (
                _causal_mask(q_len, kv_len)
                if causal
                else torch.ones(q_len, kv_len, dtype=torch.bool, device="cuda:0")
            )
            o_ref = alibi_attention(
                q[rows], k[kv_rows], v[kv_rows], mask, slopes=slopes
            )
            torch.testing.assert_close(o[rows], o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("num_heads", [8, 12])
def test_alibi_slopes_explicit_default_matches_builtin(num_heads):
    head_dim, q_len, kv_len = 128, 33, 81
    slopes = get_alibi_slopes(num_heads, device="cuda:0")
    q = torch.randn(q_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn_like(k)
    o_default = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=True, pos_encoding_mode="ALIBI"
    )
    o_explicit = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=True, pos_encoding_mode="ALIBI", alibi_slopes=slopes
    )
    assert torch.equal(o_default, o_explicit)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda:0")
    outputs = []
    for explicit in (None, slopes):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_heads,
            num_heads,
            head_dim,
            causal=True,
            pos_encoding_mode="ALIBI",
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            alibi_slopes=explicit,
        )
        outputs.append(wrapper.run(q, k, v))
    assert torch.equal(outputs[0], outputs[1])


def _alibi_entry_points(num_heads):
    """Callables ``f(pos_encoding_mode, alibi_slopes)`` for every public entry point."""
    head_dim, q_len, kv_len, page_size = 128, 17, 37, 16
    q_decode = torch.randn(num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    q_prefill = torch.randn(
        q_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    k = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn_like(k)
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, _, _ = _build_paged_kv(
        [kv_len], num_heads, head_dim, page_size
    )
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda:0")
    ragged_kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda:0")

    def single_decode(mode, slopes):
        return flashinfer.single_decode_with_kv_cache(
            q_decode, k, v, pos_encoding_mode=mode, alibi_slopes=slopes
        )

    def single_prefill(mode, slopes):
        return flashinfer.single_prefill_with_kv_cache(
            q_prefill, k, v, pos_encoding_mode=mode, alibi_slopes=slopes
        )

    def batch_decode(mode, slopes):
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
            pos_encoding_mode=mode,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            alibi_slopes=slopes,
        )

    def batch_prefill_paged(mode, slopes):
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
            pos_encoding_mode=mode,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            alibi_slopes=slopes,
        )

    def batch_prefill_ragged(mode, slopes):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            qo_indptr,
            ragged_kv_indptr,
            num_heads,
            num_heads,
            head_dim,
            pos_encoding_mode=mode,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            alibi_slopes=slopes,
        )

    return {
        "single_decode": single_decode,
        "single_prefill": single_prefill,
        "batch_decode": batch_decode,
        "batch_prefill_paged": batch_prefill_paged,
        "batch_prefill_ragged": batch_prefill_ragged,
    }


def _bad_slopes(num_heads):
    good = get_alibi_slopes(num_heads, device="cuda:0")
    return {
        "wrong_length": (good[:-1], "Invalid shape of alibi_slopes"),
        "wrong_dtype": (good.half(), "Invalid dtype of alibi_slopes"),
        "wrong_device": (good.cpu(), "Invalid device of alibi_slopes"),
        "non_contiguous": (good.repeat_interleave(2)[::2], "must be contiguous"),
        "not_a_tensor": (good.tolist(), "must be a torch.Tensor"),
    }


@pytest.mark.parametrize(
    "entry_point",
    [
        "single_decode",
        "single_prefill",
        "batch_decode",
        "batch_prefill_paged",
        "batch_prefill_ragged",
    ],
)
@pytest.mark.parametrize(
    "case",
    [
        "wrong_length",
        "wrong_dtype",
        "wrong_device",
        "non_contiguous",
        "not_a_tensor",
        "mode_none",
    ],
)
def test_alibi_slopes_validation(entry_point, case):
    num_heads = 8
    call = _alibi_entry_points(num_heads)[entry_point]
    expected = ValueError
    if case == "mode_none":
        slopes = get_alibi_slopes(num_heads, device="cuda:0")
        mode, match = "NONE", 'requires pos_encoding_mode="ALIBI"'
    else:
        slopes, match = _bad_slopes(num_heads)[case]
        mode = "ALIBI"
        if case == "not_a_tensor":
            expected = TypeError
    with pytest.raises(expected, match=match):
        call(mode, slopes)


@pytest.mark.parametrize(
    "entry_point",
    ["single_prefill", "batch_decode", "batch_prefill_paged", "batch_prefill_ragged"],
)
def test_alibi_slopes_rejected_on_non_fa2_backend(entry_point):
    """The rejection happens before any JIT lookup, so it needs no fa3-capable GPU."""
    num_heads, head_dim, q_len, kv_len, page_size = 8, 128, 17, 37, 16
    slopes = get_alibi_slopes(num_heads, device="cuda:0")
    q = torch.randn(q_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn_like(k)
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda:0")
    kv_indices = torch.arange(3, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor([5], dtype=torch.int32, device="cuda:0")
    paged_kv_indptr = torch.tensor([0, 3], dtype=torch.int32, device="cuda:0")
    common = dict(
        pos_encoding_mode="ALIBI",
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        alibi_slopes=slopes,
    )
    if entry_point == "single_prefill":

        def call():
            flashinfer.single_prefill_with_kv_cache(
                q, k, v, pos_encoding_mode="ALIBI", backend="fa3", alibi_slopes=slopes
            )

    elif entry_point == "batch_decode":
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa3"
        )

        def call():
            wrapper.plan(
                paged_kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_heads,
                num_heads,
                head_dim,
                page_size,
                **common,
            )

    elif entry_point == "batch_prefill_paged":
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa3"
        )

        def call():
            wrapper.plan(
                qo_indptr,
                paged_kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_heads,
                num_heads,
                head_dim,
                page_size,
                **common,
            )

    else:
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", backend="fa3"
        )

        def call():
            wrapper.plan(qo_indptr, kv_indptr, num_heads, num_heads, head_dim, **common)

    with pytest.raises(NotImplementedError, match="only supported by the fa2 backend"):
        call()


def test_alibi_slopes_run_rejects_head_count_mismatch():
    num_heads, head_dim, q_len, kv_len = 8, 128, 17, 37
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim,
        pos_encoding_mode="ALIBI",
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        alibi_slopes=get_alibi_slopes(num_heads, device="cuda:0"),
    )
    q = torch.randn(
        q_len, 2 * num_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    k = torch.randn(
        kv_len, 2 * num_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="entries but q has"):
        wrapper.run(q, k, v)


def test_alibi_slopes_cuda_graph_replan():
    """A captured graph must follow the slopes of a later plan(): the wrapper
    copies them into a buffer whose address the graph keeps reading."""
    num_heads, head_dim, q_len, kv_len = 8, 128, 33, 81
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr.clone(),
        kv_indptr_buf=kv_indptr.clone(),
    )
    q = torch.randn(q_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    mask = _causal_mask(q_len, kv_len)

    def plan(slopes):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_heads,
            num_heads,
            head_dim,
            causal=True,
            pos_encoding_mode="ALIBI",
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            alibi_slopes=slopes,
        )

    global_slopes = get_alibi_slopes(32, device="cuda:0")
    plan(global_slopes[8:16].clone())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            wrapper.run(q, k, v, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, out=out)
    graph.replay()
    torch.cuda.synchronize()
    o_ref = alibi_attention(q, k, v, mask, slopes=global_slopes[8:16])
    torch.testing.assert_close(out, o_ref, rtol=1e-2, atol=1e-2)

    # Replan from a temporary tensor without recapturing; the temporary is
    # freed before the replay.
    plan(global_slopes[24:32].clone())
    junk = torch.full((num_heads,), 1000.0, device="cuda:0")
    graph.replay()
    torch.cuda.synchronize()
    del junk
    o_ref = alibi_attention(q, k, v, mask, slopes=global_slopes[24:32])
    torch.testing.assert_close(out, o_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_single_decode_alibi(4096, 32, 128)
    test_single_prefill_alibi(128, 128, 8, 128, False)
    test_batch_prefill_paged_alibi_slopes_tp(12, 4, 16, True)
