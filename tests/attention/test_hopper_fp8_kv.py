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

# FA3 (SM90) attention with a 16-bit query and an FP8 KV cache.
#
# The kernels dequantize K/V to the query dtype while loading them and then run
# the 16-bit tensor-core mainloop, which is also what the FA2 mixed-precision
# kernels do. Every test therefore checks the fa3 output against fa2 and against
# an fp32 reference computed from the dequantized KV cache.

import functools
import math

import pytest
import torch

import flashinfer
from flashinfer.utils import determine_attention_backend, is_sm90a_supported

# (head_dim, q_dtype, kv_dtype). Every dtype pair exercises a different conversion path, so
# all four are covered at head_dim 128; the other head dims (different tile shapes and
# staging depths) use one pair to bound the number of JIT modules.
CONFIGS = [
    (128, torch.bfloat16, torch.float8_e4m3fn),
    (128, torch.float16, torch.float8_e4m3fn),
    (128, torch.bfloat16, torch.float8_e5m2),
    (128, torch.float16, torch.float8_e5m2),
    (64, torch.bfloat16, torch.float8_e4m3fn),
    (256, torch.bfloat16, torch.float8_e4m3fn),
]
# The single-prefill kernel shares the dense mainloop with ragged prefill.
SINGLE_CONFIGS = [
    (128, torch.bfloat16, torch.float8_e4m3fn),
    (128, torch.float16, torch.float8_e5m2),
    (64, torch.bfloat16, torch.float8_e4m3fn),
    (256, torch.bfloat16, torch.float8_e4m3fn),
]


def _skip_unless_sm90a():
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")


@functools.lru_cache(maxsize=1)
def _workspace():
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _tol(q_dtype):
    # fa3 and fa2 see the same dequantized operands and both accumulate in fp32, so they
    # only differ by accumulation order and the rounding of P and O to q_dtype.
    if q_dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=2e-2)
    return dict(rtol=1e-2, atol=5e-3)


def _lse_tol(q_dtype):
    if q_dtype == torch.bfloat16:
        return dict(rtol=5e-3, atol=5e-3)
    return dict(rtol=1e-3, atol=1e-3)


def _reference(
    q,
    k,
    v,
    qo_indptr,
    kv_indptr,
    causal,
    window_left=-1,
    logits_soft_cap=0.0,
):
    """fp32 attention over the dequantized KV cache, one request at a time.

    q: [total_q, H, D]; k, v: [total_kv, HKV, D] (any dtype). Query positions are aligned
    to the end of the KV sequence, as in the kernels.
    """
    H = q.shape[1]
    HKV = k.shape[1]
    D = q.shape[2]
    sm_scale = 1.0 / math.sqrt(D)
    outs = []
    for b in range(len(qo_indptr) - 1):
        qb = q[qo_indptr[b] : qo_indptr[b + 1]].float()
        kb = k[kv_indptr[b] : kv_indptr[b + 1]].float().repeat_interleave(H // HKV, 1)
        vb = v[kv_indptr[b] : kv_indptr[b + 1]].float().repeat_interleave(H // HKV, 1)
        qo_len, kv_len = qb.shape[0], kb.shape[0]
        s = torch.einsum("qhd,khd->hqk", qb, kb) * sm_scale
        if logits_soft_cap > 0:
            s = torch.tanh(s / logits_soft_cap) * logits_soft_cap
        q_pos = torch.arange(qo_len, device=q.device)[:, None] + kv_len - qo_len
        k_pos = torch.arange(kv_len, device=q.device)[None, :]
        mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=q.device)
        if causal:
            mask &= k_pos <= q_pos
        if window_left >= 0:
            mask &= k_pos >= q_pos - window_left
        s = s.masked_fill(~mask[None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vb))
    return torch.cat(outs, 0)


def _paged_cache(k, v, kv_indptr_tokens, page_size, kv_layout, generator):
    """Scatter per-request [tokens, HKV, D] K/V into a page table with shuffled pages.

    Returns (k_cache, v_cache, kv_indptr, kv_indices, last_page_len).
    """
    HKV, D = k.shape[1], k.shape[2]
    lens = [
        int(kv_indptr_tokens[b + 1] - kv_indptr_tokens[b])
        for b in range(len(kv_indptr_tokens) - 1)
    ]
    pages_per_req = [(n + page_size - 1) // page_size for n in lens]
    total_pages = sum(pages_per_req)
    perm = torch.randperm(total_pages, generator=generator)
    k_cache = torch.zeros(
        total_pages, page_size, HKV, D, dtype=k.dtype, device=k.device
    )
    v_cache = torch.zeros_like(k_cache)
    kv_indptr = [0]
    kv_indices = []
    last_page_len = []
    page = 0
    for b, (n, num_pages) in enumerate(zip(lens, pages_per_req, strict=True)):
        kb = k[kv_indptr_tokens[b] : kv_indptr_tokens[b + 1]]
        vb = v[kv_indptr_tokens[b] : kv_indptr_tokens[b + 1]]
        for p in range(num_pages):
            phys = int(perm[page + p])
            rows = slice(p * page_size, min((p + 1) * page_size, n))
            k_cache[phys, : rows.stop - rows.start] = kb[rows]
            v_cache[phys, : rows.stop - rows.start] = vb[rows]
            kv_indices.append(phys)
        page += num_pages
        kv_indptr.append(page)
        last_page_len.append(n - (num_pages - 1) * page_size)
    if kv_layout == "HND":
        k_cache = k_cache.transpose(1, 2).contiguous()
        v_cache = v_cache.transpose(1, 2).contiguous()
    to_i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=k.device)
    return (
        k_cache,
        v_cache,
        to_i32(kv_indptr),
        to_i32(kv_indices),
        to_i32(last_page_len),
    )


@pytest.mark.parametrize("seq_len", [11, 99, 1763, 4097])
@pytest.mark.parametrize("num_qo_heads, num_kv_heads", [(1, 1), (8, 2)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim, q_dtype, kv_dtype", SINGLE_CONFIGS)
def test_single_prefill_fp8_kv(
    seq_len, num_qo_heads, num_kv_heads, causal, head_dim, q_dtype, kv_dtype
):
    _skip_unless_sm90a()
    torch.manual_seed(0)
    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=q_dtype, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )

    o_fa3, lse_fa3 = flashinfer.single_prefill_with_kv_cache_return_lse(
        q, k, v, causal=causal, backend="fa3"
    )
    o_fa2, lse_fa2 = flashinfer.single_prefill_with_kv_cache_return_lse(
        q, k, v, causal=causal, backend="fa2"
    )
    indptr = [0, seq_len]
    o_ref = _reference(q, k, v, indptr, indptr, causal)

    assert o_fa3.dtype == q_dtype
    torch.testing.assert_close(lse_fa3, lse_fa2, **_lse_tol(q_dtype))
    torch.testing.assert_close(o_fa3, o_fa2, **_tol(q_dtype))
    torch.testing.assert_close(o_fa3.float(), o_ref, **_tol(q_dtype))


@pytest.mark.parametrize("batch_size", [1, 7])
@pytest.mark.parametrize("seq_len", [12, 99, 1763])
@pytest.mark.parametrize("num_qo_heads, num_kv_heads", [(4, 4), (8, 1)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim, q_dtype, kv_dtype", CONFIGS)
def test_batch_ragged_prefill_fp8_kv(
    batch_size, seq_len, num_qo_heads, num_kv_heads, causal, head_dim, q_dtype, kv_dtype
):
    _skip_unless_sm90a()
    torch.manual_seed(0)
    total = batch_size * seq_len
    q = torch.randn(total, num_qo_heads, head_dim, dtype=q_dtype, device="cuda")
    k = torch.randn(total, num_kv_heads, head_dim, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )
    v = torch.randn(total, num_kv_heads, head_dim, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )
    indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len

    outs = {}
    for backend in ["fa3", "fa2"]:
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _workspace(), "NHD", backend=backend
        )
        wrapper.plan(
            indptr,
            indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=causal,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
        )
        outs[backend] = wrapper.run_return_lse(q, k, v)
    o_ref = _reference(q, k, v, indptr.tolist(), indptr.tolist(), causal)

    torch.testing.assert_close(outs["fa3"][1], outs["fa2"][1], **_lse_tol(q_dtype))
    torch.testing.assert_close(outs["fa3"][0], outs["fa2"][0], **_tol(q_dtype))
    torch.testing.assert_close(outs["fa3"][0].float(), o_ref, **_tol(q_dtype))


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("qo_len, kv_len", [(1, 54), (37, 37), (300, 4097)])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("num_qo_heads, num_kv_heads", [(8, 2)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim, q_dtype, kv_dtype", CONFIGS)
def test_batch_paged_prefill_fp8_kv(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    kv_layout,
    num_qo_heads,
    num_kv_heads,
    causal,
    head_dim,
    q_dtype,
    kv_dtype,
):
    _skip_unless_sm90a()
    if causal and qo_len > kv_len:
        pytest.skip("causal requires qo_len <= kv_len")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    # Requests of different lengths, with the last one at the parametrized size.
    qo_lens = [max(1, qo_len - 3 * b) for b in range(batch_size)]
    kv_lens = [max(n, kv_len - 5 * b) for b, n in enumerate(qo_lens)]
    qo_indptr = [0]
    kv_indptr_tokens = [0]
    for a, b in zip(qo_lens, kv_lens, strict=True):
        qo_indptr.append(qo_indptr[-1] + a)
        kv_indptr_tokens.append(kv_indptr_tokens[-1] + b)
    q = torch.randn(qo_indptr[-1], num_qo_heads, head_dim, dtype=q_dtype, device="cuda")
    k = torch.randn(
        kv_indptr_tokens[-1], num_kv_heads, head_dim, dtype=q_dtype, device="cuda"
    ).to(kv_dtype)
    v = torch.randn(
        kv_indptr_tokens[-1], num_kv_heads, head_dim, dtype=q_dtype, device="cuda"
    ).to(kv_dtype)
    k_cache, v_cache, kv_indptr, kv_indices, last_page_len = _paged_cache(
        k, v, kv_indptr_tokens, page_size, kv_layout, gen
    )
    qo_indptr_t = torch.tensor(qo_indptr, dtype=torch.int32, device="cuda")

    outs = {}
    for backend in ["fa3", "fa2"]:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace(), kv_layout, backend=backend
        )
        wrapper.plan(
            qo_indptr_t,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
        )
        outs[backend] = wrapper.run_return_lse(q, (k_cache, v_cache))
    o_ref = _reference(q, k, v, qo_indptr, kv_indptr_tokens, causal)

    torch.testing.assert_close(outs["fa3"][1], outs["fa2"][1], **_lse_tol(q_dtype))
    torch.testing.assert_close(outs["fa3"][0], outs["fa2"][0], **_tol(q_dtype))
    torch.testing.assert_close(outs["fa3"][0].float(), o_ref, **_tol(q_dtype))


@pytest.mark.parametrize("seq_len", [99, 1763])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_dtype, kv_dtype", [(torch.bfloat16, torch.float8_e4m3fn)])
def test_deepseek_prefill_fp8_kv(seq_len, causal, q_dtype, kv_dtype):
    """The (192, 128) head-dim pair of the DeepSeek prefill kernel, ragged KV.

    The fa2 mixed-precision kernel returns wrong results for this head-dim pair with an
    fp8 KV cache (issue #4976), so the comparison is against the 16-bit fa3 kernel on the
    dequantized cache (bit-identical operands) and the fp32 reference instead."""
    _skip_unless_sm90a()
    torch.manual_seed(0)
    num_heads, head_dim_qk, head_dim_vo = 16, 192, 128
    q = torch.randn(seq_len, num_heads, head_dim_qk, dtype=q_dtype, device="cuda")
    k = torch.randn(seq_len, num_heads, head_dim_qk, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )
    v = torch.randn(seq_len, num_heads, head_dim_vo, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )
    indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    outs = {}
    for name, (kk, vv) in {
        "fp8": (k, v),
        "16bit": (k.to(q_dtype), v.to(q_dtype)),
    }.items():
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _workspace(), "NHD", backend="fa3"
        )
        wrapper.plan(
            indptr,
            indptr,
            num_heads,
            num_heads,
            head_dim_qk,
            head_dim_vo=head_dim_vo,
            causal=causal,
            q_data_type=q_dtype,
            kv_data_type=kk.dtype,
        )
        outs[name] = wrapper.run_return_lse(q, kk, vv)
    o_ref = _reference(q, k, v, [0, seq_len], [0, seq_len], causal)

    torch.testing.assert_close(outs["fp8"][1], outs["16bit"][1], **_lse_tol(q_dtype))
    torch.testing.assert_close(outs["fp8"][0], outs["16bit"][0], **_tol(q_dtype))
    torch.testing.assert_close(outs["fp8"][0].float(), o_ref, **_tol(q_dtype))


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("qo_len, kv_len", [(300, 4097), (1024, 1024)])
@pytest.mark.parametrize("window_left", [64, 333])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("q_dtype, kv_dtype", [(torch.bfloat16, torch.float8_e4m3fn)])
def test_fp8_kv_sliding_window_and_soft_cap(
    paged, qo_len, kv_len, window_left, causal, logits_soft_cap, q_dtype, kv_dtype
):
    """window_left and logits_soft_cap through the fp8-KV mainloops, head_dim 128."""
    _skip_unless_sm90a()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    num_qo_heads, num_kv_heads, head_dim, page_size = 8, 2, 128, 16
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype, device="cuda")
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device="cuda")
    plan_kwargs = dict(
        causal=causal,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    outs = {}
    for backend in ["fa3", "fa2"]:
        if paged:
            k_cache, v_cache, kv_indptr, kv_indices, last_page_len = _paged_cache(
                k, v, [0, kv_len], page_size, "NHD", gen
            )
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                _workspace(), "NHD", backend=backend
            )
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                **plan_kwargs,
            )
            outs[backend] = wrapper.run(q, (k_cache, v_cache))
        else:
            kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                _workspace(), "NHD", backend=backend
            )
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                **plan_kwargs,
            )
            outs[backend] = wrapper.run(q, k, v)
    o_ref = _reference(
        q, k, v, [0, qo_len], [0, kv_len], causal, window_left, logits_soft_cap
    )

    torch.testing.assert_close(outs["fa3"].float(), o_ref, **_tol(q_dtype))
    if causal:
        # The fa2 non-causal sliding window is wrong (issue #4972), so fa2 is only a
        # reference for the causal cases.
        torch.testing.assert_close(outs["fa3"], outs["fa2"], **_tol(q_dtype))


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_kv_calibration_scale(paged, kv_dtype):
    """k_scale / v_scale keep their fa2 semantics: fp8 attention over a scaled cache
    matches 16-bit attention over the unscaled one."""
    _skip_unless_sm90a()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    q_dtype = torch.float16
    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, page_size = (
        53,
        97,
        32,
        4,
        128,
        8,
    )
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype, device="cuda")
    k16 = 0.05 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda"
    )
    v16 = 0.05 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda"
    )
    k_scale = k16.abs().amax().item() / 256
    v_scale = v16.abs().amax().item() / 256
    k8 = (k16 / k_scale).to(kv_dtype)
    v8 = (v16 / v_scale).to(kv_dtype)

    outs = {}
    for name, (kk, vv, scales) in {
        "f16": (k16, v16, {}),
        "fp8": (k8, v8, dict(k_scale=k_scale, v_scale=v_scale)),
    }.items():
        if paged:
            k_cache, v_cache, kv_indptr, kv_indices, last_page_len = _paged_cache(
                kk, vv, [0, kv_len], page_size, "NHD", gen
            )
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                _workspace(), "NHD", backend="fa3"
            )
            wrapper.plan(
                torch.tensor([0, qo_len], dtype=torch.int32, device="cuda"),
                kv_indptr,
                kv_indices,
                last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                causal=True,
                q_data_type=q_dtype,
                kv_data_type=kk.dtype,
            )
            outs[name] = wrapper.run(q, (k_cache, v_cache), **scales)
        else:
            outs[name] = flashinfer.single_prefill_with_kv_cache(
                q, kk, vv, causal=True, backend="fa3", **scales
            )

    torch.testing.assert_close(outs["f16"], outs["fp8"], atol=1e-2, rtol=2e-1)


@pytest.mark.parametrize(
    "q_dtype, kv_dtype",
    [(torch.bfloat16, torch.float8_e4m3fn), (torch.float16, torch.float8_e5m2)],
)
def test_fp8_kv_backend_selection(q_dtype, kv_dtype):
    """backend="auto" routes 16-bit-query / fp8-KV prefill to fa3 on SM90, while
    tensor-core decode stays on fa2; an explicit fa3 decode still works."""
    _skip_unless_sm90a()
    device = torch.device("cuda")
    assert (
        determine_attention_backend(device, 0, False, False, q_dtype, kv_dtype) == "fa3"
    )

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    batch_size, kv_len, num_qo_heads, num_kv_heads, head_dim, page_size = (
        4,
        333,
        8,
        2,
        128,
        16,
    )
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=q_dtype, device="cuda")
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda"
    ).to(kv_dtype)
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda"
    ).to(kv_dtype)
    kv_indptr_tokens = [b * kv_len for b in range(batch_size + 1)]
    k_cache, v_cache, kv_indptr, kv_indices, last_page_len = _paged_cache(
        k, v, kv_indptr_tokens, page_size, "NHD", gen
    )
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")

    prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace(), "NHD", backend="auto"
    )
    prefill.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    assert prefill._backend == "fa3"
    o_prefill = prefill.run(q, (k_cache, v_cache))

    outs = {}
    for backend in ["auto", "fa3"]:
        decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _workspace(), "NHD", use_tensor_cores=True, backend=backend
        )
        decode.plan(
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
        )
        outs[backend] = decode.run(q, (k_cache, v_cache))
        assert decode._backend == ("fa2" if backend == "auto" else "fa3")
    o_ref = _reference(q, k, v, qo_indptr.tolist(), kv_indptr_tokens, causal=False)

    for o in [o_prefill, outs["auto"], outs["fa3"]]:
        torch.testing.assert_close(o.float(), o_ref, **_tol(q_dtype))


@pytest.mark.parametrize("q_dtype, kv_dtype", [(torch.bfloat16, torch.float8_e4m3fn)])
def test_fp8_kv_block_sparse_wrapper(q_dtype, kv_dtype):
    """BlockSparseAttentionWrapper resolves backend="auto" through the same selector and
    runs the paged kernel with page_size = C."""
    _skip_unless_sm90a()
    torch.manual_seed(0)
    M, N, R, C, num_qo_heads, num_kv_heads, head_dim = 512, 2048, 16, 32, 8, 2, 128
    MB, NB = M // R, N // C
    block_mask = torch.rand(MB, NB) < 0.3
    indptr = [0]
    indices = []
    for r in range(MB):
        indices += block_mask[r].nonzero().flatten().tolist()
        indptr.append(len(indices))
    indptr = torch.tensor(indptr, dtype=torch.int32, device="cuda")
    indices = torch.tensor(indices, dtype=torch.int32, device="cuda")
    q = torch.randn(M, num_qo_heads, head_dim, dtype=q_dtype, device="cuda")
    k = torch.randn(N, num_kv_heads, head_dim, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )
    v = torch.randn(N, num_kv_heads, head_dim, dtype=q_dtype, device="cuda").to(
        kv_dtype
    )

    outs = {}
    for backend in ["fa3", "fa2", "auto"]:
        wrapper = flashinfer.sparse.BlockSparseAttentionWrapper(
            _workspace(), backend=backend
        )
        wrapper.plan(
            indptr,
            indices,
            M,
            N,
            R,
            C,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=q_dtype,
        )
        outs[backend] = wrapper.run(q, k, v)
        assert wrapper._backend == ("fa2" if backend == "fa2" else "fa3")

    torch.testing.assert_close(outs["fa3"], outs["fa2"], **_tol(q_dtype))
    torch.testing.assert_close(outs["auto"], outs["fa3"], rtol=0, atol=0)


@pytest.mark.parametrize("q_dtype, kv_dtype", [(torch.bfloat16, torch.float8_e4m3fn)])
def test_fp8_kv_attention_sink_wrapper(q_dtype, kv_dtype):
    """The attention-sink variant of the paged kernel with an fp8 KV cache."""
    _skip_unless_sm90a()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    batch_size, kv_len, num_qo_heads, num_kv_heads, head_dim, page_size = (
        3,
        333,
        8,
        2,
        128,
        16,
    )
    q = torch.randn(
        batch_size * kv_len, num_qo_heads, head_dim, dtype=q_dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda"
    ).to(kv_dtype)
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=q_dtype, device="cuda"
    ).to(kv_dtype)
    kv_indptr_tokens = [b * kv_len for b in range(batch_size + 1)]
    k_cache, v_cache, kv_indptr, kv_indices, last_page_len = _paged_cache(
        k, v, kv_indptr_tokens, page_size, "NHD", gen
    )
    qo_indptr = torch.tensor(kv_indptr_tokens, dtype=torch.int32, device="cuda")
    sink = torch.rand(num_qo_heads, dtype=torch.float32, device="cuda") * 5

    outs = {}
    for backend in ["fa3", "fa2", "auto"]:
        wrapper = flashinfer.BatchAttentionWithAttentionSinkWrapper(
            _workspace(),
            kv_layout="NHD",
            backend=backend,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
        )
        outs[backend] = wrapper.run(
            q, (k_cache, v_cache), sink, 1.0 / math.sqrt(head_dim)
        )
        assert wrapper._backend == ("fa2" if backend == "fa2" else "fa3")

    torch.testing.assert_close(outs["fa3"], outs["fa2"], **_tol(q_dtype))
    torch.testing.assert_close(outs["auto"], outs["fa3"], rtol=0, atol=0)


@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("q_dtype, kv_dtype", [(torch.float16, torch.float8_e4m3fn)])
def test_fp8_kv_multi_item_scoring(page_size, num_qo_heads, q_dtype, kv_dtype):
    """Multi-item scoring skips KV tiles; fa3 must stage and dequantize the same tiles
    the consumer visits."""
    _skip_unless_sm90a()
    torch.manual_seed(0)
    kv_len, qo_len, num_kv_heads, head_dim = 97, 81, 4, 128
    prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr = (
        16,
        list(range(80)) + [0],
        97,
        79,
    )
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype, device="cuda")
    num_pages = (kv_len + page_size - 1) // page_size
    kv_data = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim, dtype=q_dtype, device="cuda"
    ).to(kv_dtype)
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(0, num_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.tensor(
        [(kv_len - 1) % page_size + 1], dtype=torch.int32, device="cuda"
    )
    mis_kwargs = dict(
        prefix_len_ptr=torch.tensor(
            [prefix_len_ptr], dtype=torch.uint32, device="cuda"
        ),
        token_pos_in_items_ptr=torch.tensor(
            token_pos_in_items_ptr, dtype=torch.uint16, device="cuda"
        ),
        token_pos_in_items_len=token_pos_in_items_len,
        max_item_len_ptr=torch.tensor(
            [max_item_len_ptr], dtype=torch.uint16, device="cuda"
        ),
    )

    outs = {}
    for backend in ["fa3", "fa2"]:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace(), "NHD", backend=backend
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            **mis_kwargs,
        )
        outs[backend] = wrapper.run_return_lse(q, kv_data)

    torch.testing.assert_close(outs["fa3"][1], outs["fa2"][1], **_lse_tol(q_dtype))
    torch.testing.assert_close(outs["fa3"][0], outs["fa2"][0], **_tol(q_dtype))
