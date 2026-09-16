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

"""``lse_base`` on the prefill wrappers.

The default (``"log2"``) must stay bit-identical to what the wrappers returned
before the option existed; ``"ln"`` must equal ``torch.logsumexp`` of the
scaled scores on every backend, and equal the base-2 result times ``ln 2``.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm90a_supported, is_sm100a_supported, ln2, log2e

DT = torch.bfloat16
H_QO, H_KV, D = 8, 2, 128
Q_LENS = [5, 64, 17]
KV_LENS = [37, 64, 90]
PAGE_SIZE = 16


def _ragged_backends():
    dev = torch.device("cuda")
    backends = ["fa2"]
    if is_sm90a_supported(dev):
        backends.append("fa3")
    if is_sm100a_supported(dev):
        backends += ["cutlass", "cudnn"]
    return backends


def _paged_backends():
    dev = torch.device("cuda")
    backends = ["fa2"]
    if is_sm90a_supported(dev):
        backends.append("fa3")
    if is_sm100a_supported(dev):
        # Pre-existing, independent of lse_base: the paged cuDNN path insists on a
        # padded (num_sequences, max_q_len, h) LSE buffer while the wrapper
        # allocates the [total_tokens, h] one every other backend writes, so
        # return_lse=True fails before any base handling. Strict xfail so the
        # fix is noticed.
        backends.append(
            pytest.param(
                "cudnn",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=ValueError,
                    reason="paged cuDNN backend rejects the wrapper's [tokens, h] LSE buffer",
                ),
            )
        )
    return backends


def _indptr(lens):
    return torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lens), 0)),
        dtype=torch.int32,
        device="cuda",
    )


def _ref_lse_ln(q, k, qo_indptr, kv_indptr, causal):
    """Natural-log LSE of the softmax scores, fp32, per request; bottom-right causal."""
    scale = 1.0 / D**0.5
    rows = []
    for i in range(qo_indptr.numel() - 1):
        qi = q[qo_indptr[i] : qo_indptr[i + 1]].float()  # [sq, hq, d]
        ki = k[kv_indptr[i] : kv_indptr[i + 1]].float()  # [skv, hkv, d]
        ki = ki.repeat_interleave(H_QO // H_KV, dim=1)
        s = torch.einsum("qhd,khd->hqk", qi, ki) * scale
        if causal:
            sq, skv = qi.shape[0], ki.shape[0]
            qpos = torch.arange(sq, device=q.device)[:, None] + (skv - sq)
            kpos = torch.arange(skv, device=q.device)[None, :]
            s = s.masked_fill(kpos > qpos, float("-inf"))
        rows.append(torch.logsumexp(s, dim=-1).transpose(0, 1))  # [sq, hq]
    return torch.cat(rows, 0)


def _check(lse_default, lse_log2, lse_ln, ref_ln):
    assert torch.equal(lse_default, lse_log2), "lse_base='log2' must equal the default"
    torch.testing.assert_close(lse_ln, ref_ln, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(lse_log2, ref_ln * log2e, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(lse_ln, lse_log2 * ln2, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("backend", _ragged_backends())
@pytest.mark.parametrize("causal", [True, False])
def test_ragged_lse_base(backend, causal):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    qo_indptr, kv_indptr = _indptr(Q_LENS), _indptr(KV_LENS)
    q = torch.randn(int(qo_indptr[-1]), H_QO, D, dtype=DT, device=dev)
    k = torch.randn(int(kv_indptr[-1]), H_KV, D, dtype=DT, device=dev)
    v = torch.randn_like(k)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend=backend)
    w.plan(
        qo_indptr,
        kv_indptr,
        H_QO,
        H_KV,
        D,
        causal=causal,
        q_data_type=DT,
        kv_data_type=DT,
    )
    assert w._backend == backend

    out_default, lse_default = w.run(q, k, v, return_lse=True)
    out_log2, lse_log2 = w.run(q, k, v, return_lse=True, lse_base="log2")
    out_ln, lse_ln = w.run(q, k, v, return_lse=True, lse_base="ln")
    assert torch.equal(out_default, out_log2) and torch.equal(out_default, out_ln)
    _check(
        lse_default, lse_log2, lse_ln, _ref_lse_ln(q, k, qo_indptr, kv_indptr, causal)
    )

    # a caller-provided buffer is filled in place, in the requested base
    buf = torch.empty_like(lse_ln)
    _, lse_buf = w.run(q, k, v, return_lse=True, lse=buf, lse_base="ln")
    assert lse_buf is buf
    torch.testing.assert_close(buf, lse_ln, atol=0, rtol=0)

    with pytest.raises(ValueError, match="lse_base"):
        w.run(q, k, v, return_lse=True, lse_base="e")


def _paged_kv(k, v, kv_indptr, lens):
    """Scatter ragged k/v into NHD pages; one contiguous page run per request."""
    dev = k.device
    pages_per_req = [(n + PAGE_SIZE - 1) // PAGE_SIZE for n in lens]
    total_pages = sum(pages_per_req)
    k_cache = torch.zeros(total_pages, PAGE_SIZE, H_KV, D, dtype=k.dtype, device=dev)
    v_cache = torch.zeros_like(k_cache)
    page_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(pages_per_req), 0)),
        dtype=torch.int32,
        device=dev,
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=dev)
    last_page_len = torch.tensor(
        [n - (p - 1) * PAGE_SIZE for n, p in zip(lens, pages_per_req, strict=True)],
        dtype=torch.int32,
        device=dev,
    )
    for i, n in enumerate(lens):
        flat_k = k_cache[page_indptr[i] : page_indptr[i + 1]].view(-1, H_KV, D)
        flat_v = v_cache[page_indptr[i] : page_indptr[i + 1]].view(-1, H_KV, D)
        flat_k[:n] = k[kv_indptr[i] : kv_indptr[i + 1]]
        flat_v[:n] = v[kv_indptr[i] : kv_indptr[i + 1]]
    max_pages = max(pages_per_req)
    block_tables = torch.zeros(len(lens), max_pages, dtype=torch.int32, device=dev)
    for i, p in enumerate(pages_per_req):
        block_tables[i, :p] = torch.arange(
            page_indptr[i], page_indptr[i + 1], device=dev
        )
    return k_cache, v_cache, page_indptr, kv_indices, last_page_len, block_tables


@pytest.mark.parametrize("backend", _paged_backends())
@pytest.mark.parametrize("causal", [True, False])
def test_paged_lse_base(backend, causal):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    qo_indptr, kv_indptr = _indptr(Q_LENS), _indptr(KV_LENS)
    q = torch.randn(int(qo_indptr[-1]), H_QO, D, dtype=DT, device=dev)
    k = torch.randn(int(kv_indptr[-1]), H_KV, D, dtype=DT, device=dev)
    v = torch.randn_like(k)
    k_cache, v_cache, page_indptr, kv_indices, last_page_len, block_tables = _paged_kv(
        k, v, kv_indptr, KV_LENS
    )
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
    plan_kwargs = {}
    if backend == "cudnn":
        plan_kwargs = dict(
            seq_lens=torch.tensor(KV_LENS, dtype=torch.int32, device=dev),
            seq_lens_q=torch.tensor(Q_LENS, dtype=torch.int32, device=dev),
            max_token_per_sequence=max(Q_LENS),
            max_sequence_kv=max(KV_LENS),
            block_tables=block_tables,
        )
    w.plan(
        qo_indptr,
        page_indptr,
        kv_indices,
        last_page_len,
        H_QO,
        H_KV,
        D,
        PAGE_SIZE,
        causal=causal,
        q_data_type=DT,
        kv_data_type=DT,
        **plan_kwargs,
    )
    assert w._backend == backend

    out_default, lse_default = w.run(q, (k_cache, v_cache), return_lse=True)
    out_log2, lse_log2 = w.run(q, (k_cache, v_cache), return_lse=True, lse_base="log2")
    out_ln, lse_ln = w.run(q, (k_cache, v_cache), return_lse=True, lse_base="ln")
    assert torch.equal(out_default, out_log2) and torch.equal(out_default, out_ln)
    _check(
        lse_default, lse_log2, lse_ln, _ref_lse_ln(q, k, qo_indptr, kv_indptr, causal)
    )

    with pytest.raises(ValueError, match="lse_base"):
        w.run(q, (k_cache, v_cache), return_lse=True, lse_base="e")
