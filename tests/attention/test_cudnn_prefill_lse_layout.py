"""The cuDNN prefill returns a packed LSE, ``(total_qo_tokens, num_qo_heads)``.

That is FlashInfer's LSE contract (what every other backend returns and what
the prefill wrappers allocate).  cuDNN writes it through a ragged Stats tensor
over the query token indptr, the same packing as ``q``.  The historical padded
``(batch, max_token_per_sequence, num_qo_heads)`` buffer is still accepted when
passed in explicitly.
"""

import pytest
import torch

import flashinfer
from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.utils import log2e


def _skip_unless_cudnn():
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")


def _ragged_case(batch_size, s_qo, s_kv, num_qo_heads, num_kv_heads, head_dim, device):
    torch.manual_seed(0)
    q_lens = torch.randint(1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device)
    kv_lens = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(q_lens, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(kv_lens, 0)]).int()
    q = torch.randn(
        int(q_lens.sum()), num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        int(kv_lens.sum()), num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    return q_lens, kv_lens, qo_indptr, kv_indptr, q, k, v


def _reference_lse(q, k, q_lens, kv_lens, qo_indptr, kv_indptr, scale, causal):
    """Base-2 LSE per query token, ``[total_q, h_qo]``, in float."""
    h_qo, h_kv = q.shape[1], k.shape[1]
    kf = k.float().repeat_interleave(h_qo // h_kv, dim=1)
    out = torch.empty(q.shape[0], h_qo, device=q.device, dtype=torch.float32)
    for b in range(q_lens.numel()):
        qs = slice(int(qo_indptr[b]), int(qo_indptr[b + 1]))
        ks = slice(int(kv_indptr[b]), int(kv_indptr[b + 1]))
        scores = torch.einsum("qhd,khd->qhk", q[qs].float(), kf[ks]) * scale
        if causal:
            lq, lk = int(q_lens[b]), int(kv_lens[b])
            col = torch.arange(lk, device=q.device)[None, :]
            row = torch.arange(lq, device=q.device)[:, None]
            scores = scores.masked_fill(
                (col > row + (lk - lq))[:, None, :], float("-inf")
            )
        out[qs] = torch.logsumexp(scores, dim=-1) * log2e
    return out


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("units", ["tokens", "elements"])
@pytest.mark.parametrize("causal", [True, False])
def test_cudnn_prefill_lse_packed_by_default(batch_size, units, causal):
    """``return_lse=True`` without a buffer returns the packed
    ``(total_qo_tokens, h_qo)`` LSE, correct per token, on both offset units
    (the ragged Stats offsets are derived from the q offsets)."""
    _skip_unless_cudnn()
    device = "cuda:0"
    s_qo, s_kv, h_qo, h_kv, d = 87, 512, 8, 4, 128
    q_lens, kv_lens, qo_indptr, kv_indptr, q, k, v = _ragged_case(
        batch_size, s_qo, s_kv, h_qo, h_kv, d, device
    )
    scale = float(d**-0.5)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    if units == "tokens":
        offsets = dict(
            batch_offsets_q=qo_indptr,
            batch_offsets_k=kv_indptr,
            batch_offsets_units="tokens",
        )
    else:
        offsets = dict(
            batch_offsets_q=qo_indptr * (h_qo * d),
            batch_offsets_o=qo_indptr * (h_qo * d),
            batch_offsets_k=kv_indptr * (h_kv * d),
            batch_offsets_v=kv_indptr * (h_kv * d),
        )
    _, lse = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale,
        ws,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
        causal=causal,
        return_lse=True,
        **offsets,
    )
    assert lse.shape == (q.shape[0], h_qo)
    ref = _reference_lse(q, k, q_lens, kv_lens, qo_indptr, kv_indptr, scale, causal)
    torch.testing.assert_close(lse, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("units", ["tokens", "elements"])
def test_cudnn_prefill_lse_padded_buffer_still_accepted(units):
    """An explicitly passed padded ``(batch, max_token_per_sequence, h_qo)``
    buffer keeps working and holds the same values as the packed LSE, request
    by request."""
    _skip_unless_cudnn()
    device = "cuda:0"
    batch_size, s_qo, s_kv, h_qo, h_kv, d = 4, 87, 512, 8, 4, 128
    q_lens, kv_lens, qo_indptr, kv_indptr, q, k, v = _ragged_case(
        batch_size, s_qo, s_kv, h_qo, h_kv, d, device
    )
    scale = float(d**-0.5)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    if units == "tokens":
        offsets = dict(
            batch_offsets_q=qo_indptr,
            batch_offsets_k=kv_indptr,
            batch_offsets_units="tokens",
        )
    else:
        offsets = dict(
            batch_offsets_q=qo_indptr * (h_qo * d),
            batch_offsets_o=qo_indptr * (h_qo * d),
            batch_offsets_k=kv_indptr * (h_kv * d),
            batch_offsets_v=kv_indptr * (h_kv * d),
        )
    common = dict(
        scale=scale,
        workspace_buffer=ws,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
        causal=True,
        return_lse=True,
        **offsets,
    )
    padded = torch.full(
        (batch_size, s_qo, h_qo), float("nan"), device=device, dtype=torch.float32
    )
    try:
        _, lse_padded = cudnn_batch_prefill_with_kv_cache(q, k, v, **common, lse=padded)
    except cudnn_prefill.cudnn.cudnnGraphNotSupportedError as exc:
        # The padded buffer is a non-ragged Stats tensor that is not packed
        # BHSD, which cudnn-frontend >= 1.28 refuses below backend 9.26.
        if "non-ragged Stats" in str(exc):
            pytest.skip(f"padded LSE needs cuDNN backend 9.26+ on this frontend: {exc}")
        raise
    assert lse_padded is padded
    _, lse_packed = cudnn_batch_prefill_with_kv_cache(q, k, v, **common)
    assert lse_packed.shape == (q.shape[0], h_qo)
    for b in range(batch_size):
        n = int(q_lens[b])
        torch.testing.assert_close(
            lse_packed[int(qo_indptr[b]) : int(qo_indptr[b + 1])],
            lse_padded[b, :n],
            atol=1e-3,
            rtol=1e-3,
        )


def test_cudnn_prefill_lse_rejects_other_shapes():
    _skip_unless_cudnn()
    device = "cuda:0"
    batch_size, s_qo, s_kv, h_qo, h_kv, d = 2, 32, 64, 4, 4, 128
    q_lens, kv_lens, qo_indptr, kv_indptr, q, k, v = _ragged_case(
        batch_size, s_qo, s_kv, h_qo, h_kv, d, device
    )
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device=device)
    with pytest.raises(ValueError, match="lse must have shape"):
        cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            float(d**-0.5),
            ws,
            max_token_per_sequence=s_qo,
            max_sequence_kv=s_kv,
            actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
            actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
            causal=False,
            return_lse=True,
            batch_offsets_q=qo_indptr,
            batch_offsets_k=kv_indptr,
            batch_offsets_units="tokens",
            lse=torch.empty(h_qo, q.shape[0], device=device, dtype=torch.float32),
        )


@pytest.mark.parametrize("units", ["tokens", "elements"])
def test_cudnn_prefill_lse_packed_with_noncontiguous_q(units):
    """Element-unit q offsets count elements of q's own storage, so for a
    non-contiguous q (here a T3HD-style view, token stride 3 * h * d) the
    derived Stats offsets must use q.stride(0), not h * d. Both unit paths
    must match the contiguous-q result."""
    _skip_unless_cudnn()
    device = "cuda:0"
    batch_size, s_qo, s_kv, h_qo, h_kv, d = 4, 64, 128, 8, 4, 128
    q_lens, kv_lens, qo_indptr, kv_indptr, q, k, v = _ragged_case(
        batch_size, s_qo, s_kv, h_qo, h_kv, d, device
    )
    qkv = torch.randn(q.shape[0], 3, h_qo, d, device=device, dtype=torch.bfloat16)
    q_view = qkv[:, 0]
    assert not q_view.is_contiguous() and q_view.stride(0) == 3 * h_qo * d
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    common = dict(
        scale=float(d**-0.5),
        workspace_buffer=ws,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
        causal=True,
        return_lse=True,
    )
    if units == "tokens":
        offs = dict(batch_offsets_q=qo_indptr, batch_offsets_k=kv_indptr)
    else:
        offs = dict(
            batch_offsets_q=qo_indptr * q_view.stride(0),
            batch_offsets_o=qo_indptr * (h_qo * d),
            batch_offsets_k=kv_indptr * (h_kv * d),
            batch_offsets_v=kv_indptr * (h_kv * d),
        )
    out, lse = cudnn_batch_prefill_with_kv_cache(
        q_view, k, v, **common, **offs, batch_offsets_units=units
    )
    out_ref, lse_ref = cudnn_batch_prefill_with_kv_cache(
        q_view.contiguous(),
        k,
        v,
        **common,
        batch_offsets_q=qo_indptr,
        batch_offsets_k=kv_indptr,
        batch_offsets_units="tokens",
    )
    assert lse.shape == (q.shape[0], h_qo)
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("h_qo,h_kv", [(8, 2), (8, 8)])
def test_cudnn_prefill_lse_packed_single_token_rows(paged, h_qo, h_kv):
    """Single-token requests (max_token_per_sequence == 1, the decode /
    verify shape). cuDNN < 9.27 mis-stores a ragged Stats tensor for GQA here
    (NVBug 6783545), so the packed LSE is served through the padded (b, 1, h)
    declaration instead. The buffer is poisoned with NaN so an unwritten head
    cannot pass by accident."""
    _skip_unless_cudnn()
    device = "cuda:0"
    batch_size, d = 3, 128
    kv_lens = torch.tensor([16, 31, 65], dtype=torch.int32, device=device)
    q_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(q_lens, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(kv_lens, 0)]).int()
    torch.manual_seed(0)
    q = torch.randn(batch_size, h_qo, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(int(kv_lens.sum()), h_kv, d, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = float(d**-0.5)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    lse = torch.full((batch_size, h_qo), float("nan"), device=device)
    common = dict(
        max_token_per_sequence=1,
        max_sequence_kv=int(kv_lens.max()),
        actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
        causal=False,
        return_lse=True,
        batch_offsets_q=qo_indptr,
        batch_offsets_units="tokens",
        lse=lse,
    )
    if paged:
        page_size = 16
        pages_per_req = [(int(n) + page_size - 1) // page_size for n in kv_lens]
        k_cache = torch.zeros(
            sum(pages_per_req), h_kv, page_size, d, device=device, dtype=k.dtype
        )
        v_cache = torch.zeros_like(k_cache)
        block_tables = torch.zeros(
            batch_size, max(pages_per_req), dtype=torch.int32, device=device
        )
        page = 0
        for b in range(batch_size):
            for j in range(pages_per_req[b]):
                start = int(kv_indptr[b]) + j * page_size
                n = min(page_size, int(kv_lens[b]) - j * page_size)
                k_cache[page, :, :n] = k[start : start + n].transpose(0, 1)
                v_cache[page, :, :n] = v[start : start + n].transpose(0, 1)
                block_tables[b, j] = page
                page += 1
        _, lse_out = cudnn_batch_prefill_with_kv_cache(
            q, k_cache, v_cache, scale, ws, **common, block_tables=block_tables
        )
    else:
        _, lse_out = cudnn_batch_prefill_with_kv_cache(
            q, k, v, scale, ws, **common, batch_offsets_k=kv_indptr
        )
    assert lse_out.shape == (batch_size, h_qo)
    assert not lse_out.isnan().any(), "some LSE heads were never written"
    lse_ref = _reference_lse(q, k, q_lens, kv_lens, qo_indptr, kv_indptr, scale, False)
    torch.testing.assert_close(lse_out, lse_ref, atol=1e-2, rtol=1e-2)


def test_cudnn_prefill_lse_single_request_no_offsets_cuda_graph():
    """A single request may omit every batch offset; the implicit Stats
    offsets must then be built without a host-to-device copy so the call can
    be captured into a CUDA graph."""
    _skip_unless_cudnn()
    device = "cuda:0"
    h_qo, h_kv, d = 8, 2, 128
    torch.manual_seed(0)
    q = torch.randn(37, h_qo, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(64, h_kv, d, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    kwargs = dict(
        max_token_per_sequence=37,
        max_sequence_kv=64,
        actual_seq_lens_q=torch.tensor([37], dtype=torch.int32, device=device).view(
            1, 1, 1, 1
        ),
        actual_seq_lens_kv=torch.tensor([64], dtype=torch.int32, device=device).view(
            1, 1, 1, 1
        ),
        causal=False,
        return_lse=True,
        is_cuda_graph_compatible=True,
    )
    scale = float(d**-0.5)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            cudnn_batch_prefill_with_kv_cache(q, k, v, scale, ws, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, lse = cudnn_batch_prefill_with_kv_cache(q, k, v, scale, ws, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    lse_ref = _reference_lse(
        q,
        k,
        torch.tensor([37], dtype=torch.int32, device=device),
        torch.tensor([64], dtype=torch.int32, device=device),
        torch.cat([zero, torch.tensor([37], dtype=torch.int32, device=device)]),
        torch.cat([zero, torch.tensor([64], dtype=torch.int32, device=device)]),
        scale,
        False,
    )
    assert lse.shape == (37, h_qo)
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("flaw", ["dtype", "contiguity"])
def test_cudnn_prefill_lse_rejects_unbindable_buffers(flaw):
    """The packed LSE is bound to cuDNN's Stats as contiguous float32; other
    buffers are rejected up front instead of at graph execution."""
    _skip_unless_cudnn()
    device = "cuda:0"
    batch_size, s_qo, s_kv, h_qo, h_kv, d = 2, 32, 64, 4, 4, 128
    q_lens, kv_lens, qo_indptr, kv_indptr, q, k, v = _ragged_case(
        batch_size, s_qo, s_kv, h_qo, h_kv, d, device
    )
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device=device)
    if flaw == "dtype":
        lse = torch.empty(q.shape[0], h_qo, device=device, dtype=torch.float16)
        match = "lse must have dtype torch.float32"
    else:
        lse = torch.empty(h_qo, q.shape[0], device=device, dtype=torch.float32).T
        match = "lse must be contiguous"
    with pytest.raises(ValueError, match=match):
        cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            float(d**-0.5),
            ws,
            max_token_per_sequence=s_qo,
            max_sequence_kv=s_kv,
            actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
            actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
            causal=False,
            return_lse=True,
            batch_offsets_q=qo_indptr,
            batch_offsets_k=kv_indptr,
            batch_offsets_units="tokens",
            lse=lse,
        )


def test_cudnn_prefill_lse_rejects_padded_with_stats_offsets():
    """Stats offsets address packed rows, so they cannot accompany a padded
    buffer: cuDNN would write every request after the first to the wrong
    rows."""
    _skip_unless_cudnn()
    device = "cuda:0"
    batch_size, s_qo, s_kv, h_qo, h_kv, d = 2, 32, 64, 4, 4, 128
    q_lens, kv_lens, qo_indptr, kv_indptr, q, k, v = _ragged_case(
        batch_size, s_qo, s_kv, h_qo, h_kv, d, device
    )
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device=device)
    with pytest.raises(ValueError, match="batch_offsets_stats addresses a packed"):
        cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            float(d**-0.5),
            ws,
            max_token_per_sequence=s_qo,
            max_sequence_kv=s_kv,
            actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
            actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
            causal=False,
            return_lse=True,
            batch_offsets_q=qo_indptr,
            batch_offsets_k=kv_indptr,
            batch_offsets_stats=qo_indptr,
            batch_offsets_units="tokens",
            lse=torch.empty(batch_size, s_qo, h_qo, device=device, dtype=torch.float32),
        )


@pytest.mark.parametrize("lse_layout", ["NH", "HN"])
@pytest.mark.parametrize("paged", [False, True])
def test_wrapper_cudnn_single_token_gqa_lse(paged, lse_layout):
    """The wrappers' prepared-plan path for single-token GQA rows (decode /
    verify steps). On cuDNN < 9.27 (NVBug 6783545) the plan drops the Stats
    ragged offset so the unragged (b, h, 1, 1) declaration writes the packed
    LSE; on fixed versions it keeps the ragged form. Either way every head is
    written: the caller's LSE buffer is NaN-poisoned."""
    _skip_unless_cudnn()
    from flashinfer.cudnn.prefill import _cudnn_single_token_gqa_ragged_stats_broken

    device = "cuda:0"
    batch_size, h_qo, h_kv, d, page_size = 3, 8, 2, 128, 16
    kv_lens = torch.tensor([16, 31, 65], dtype=torch.int32, device=device)
    q_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(q_lens, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(kv_lens, 0)]).int()
    torch.manual_seed(0)
    q = torch.randn(batch_size, h_qo, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(int(kv_lens.sum()), h_kv, d, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = float(d**-0.5)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    if paged:
        pages_per_req = [(int(n) + page_size - 1) // page_size for n in kv_lens]
        num_pages = sum(pages_per_req)
        k_cache = torch.zeros(
            num_pages, h_kv, page_size, d, device=device, dtype=k.dtype
        )
        v_cache = torch.zeros_like(k_cache)
        block_tables = torch.zeros(
            batch_size, max(pages_per_req), dtype=torch.int32, device=device
        )
        page = 0
        for b in range(batch_size):
            for j in range(pages_per_req[b]):
                start = int(kv_indptr[b]) + j * page_size
                n = min(page_size, int(kv_lens[b]) - j * page_size)
                k_cache[page, :, :n] = k[start : start + n].transpose(0, 1)
                v_cache[page, :, :n] = v[start : start + n].transpose(0, 1)
                block_tables[b, j] = page
                page += 1
        page_indptr = torch.cat(
            [zero, torch.cumsum(torch.tensor(pages_per_req, device=device), 0)]
        ).int()
        last_page_len = (kv_lens - 1) % page_size + 1
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="cudnn")
        w.plan(
            qo_indptr,
            page_indptr,
            torch.arange(num_pages, dtype=torch.int32, device=device),
            last_page_len.int(),
            h_qo,
            h_kv,
            d,
            page_size,
            causal=False,
            q_data_type=torch.bfloat16,
            seq_lens=kv_lens,
            seq_lens_q=q_lens,
            sm_scale=scale,
            max_token_per_sequence=1,
            max_sequence_kv=int(kv_lens.max()),
            block_tables=block_tables,
        )
        kv_args = ((k_cache.transpose(1, 2), v_cache.transpose(1, 2)),)
    else:
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="cudnn")
        w.plan(
            qo_indptr,
            kv_indptr,
            h_qo,
            h_kv,
            d,
            causal=False,
            sm_scale=scale,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        kv_args = (k, v)
    # Below the direct path's cuDNN minimum (paged: 9.25) the wrapper has no
    # prepared plan and runs through cudnn_batch_prefill_with_kv_cache, which
    # applies the same workaround; check the plan only where there is one.
    plan = w._cudnn_plan
    if plan is not None:
        assert (plan.metadata.batch_offsets_stats is None) == (
            _cudnn_single_token_gqa_ragged_stats_broken()
        )
    shape = (batch_size, h_qo) if lse_layout == "NH" else (h_qo, batch_size)
    lse = torch.full(shape, float("nan"), device=device)
    _, lse_out = w.run(q, *kv_args, lse=lse, return_lse=True, lse_layout=lse_layout)
    assert not lse_out.isnan().any(), "some LSE heads were never written"
    lse_ref = _reference_lse(q, k, q_lens, kv_lens, qo_indptr, kv_indptr, scale, False)
    if lse_layout == "HN":
        lse_ref = lse_ref.t()
    torch.testing.assert_close(lse_out, lse_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("causal", [True, False])
def test_ragged_wrapper_cudnn_return_lse(causal):
    """``BatchPrefillWithRaggedKVCacheWrapper(backend="cudnn").run(...,
    return_lse=True)`` returns the ``[qo_len, num_qo_heads]`` LSE the wrapper
    documents, checked against independent Torch O and LSE references."""
    _skip_unless_cudnn()
    device = "cuda:0"
    batch_size, s_qo, s_kv, h_qo, h_kv, d = 4, 87, 512, 8, 4, 128
    q_lens, kv_lens, qo_indptr, kv_indptr, q, k, v = _ragged_case(
        batch_size, s_qo, s_kv, h_qo, h_kv, d, device
    )
    scale = float(d**-0.5)
    ws = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)

    # Same token-unit indptrs for both backends; the cuDNN wrapper derives
    # everything else (seq lens, max lengths, Stats offsets) from them.
    w_cudnn = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, "NHD", backend="cudnn"
    )
    w_cudnn.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=h_qo,
        num_kv_heads=h_kv,
        head_dim_qk=d,
        head_dim_vo=d,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
    )
    out, lse = w_cudnn.run(q, k, v, return_lse=True)

    # A default wrapper may also select cuDNN. Use an independent reference
    # so a dispatch change cannot turn this into a comparison with itself.
    ref_lse = _reference_lse(q, k, q_lens, kv_lens, qo_indptr, kv_indptr, scale, causal)
    ref_out = torch.empty_like(out, dtype=torch.float32)
    for batch in range(batch_size):
        qs = slice(int(qo_indptr[batch]), int(qo_indptr[batch + 1]))
        ks = slice(int(kv_indptr[batch]), int(kv_indptr[batch + 1]))
        qi = q[qs].float().transpose(0, 1)
        ki = k[ks].float().repeat_interleave(h_qo // h_kv, dim=1).transpose(0, 1)
        vi = v[ks].float().repeat_interleave(h_qo // h_kv, dim=1).transpose(0, 1)
        scores = qi @ ki.transpose(-1, -2) * scale
        if causal:
            lq, lkv = qi.size(1), ki.size(1)
            row = torch.arange(lq, device=q.device)[:, None]
            col = torch.arange(lkv, device=q.device)[None, :]
            scores.masked_fill_(col > row + lkv - lq, -torch.inf)
        ref_out[qs] = (scores.softmax(-1) @ vi).transpose(0, 1)

    assert lse.shape == (q.shape[0], h_qo) == ref_lse.shape
    torch.testing.assert_close(out.float(), ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("hk", [2, 4])
@pytest.mark.parametrize("maxq", [1, 17])
@pytest.mark.parametrize("packed_lse", [False, True])
def test_cudnn_prefill_empty_requests_changed_lengths_replay(
    dtype, paged, hk, maxq, packed_lse
):
    """Move the empty request on replay, checking live O and packed/padded LSE.

    In particular, run this on the consumer cuDNN 9.24 stack: do not waive
    zero-length requests just because newer backend tests require 9.25.
    """
    _skip_unless_cudnn()
    if cudnn_prefill.cudnn.backend_version() < 92400:
        pytest.skip("requires cuDNN 9.24 or later")
    b, h, d, page, capacity = 3, 4, 128, 16, 48
    first_q = [1, 0, 1] if maxq == 1 else [17, 0, 9]
    second_q = [0, 1, 1] if maxq == 1 else [0, 9, 17]
    first_k, second_k = [33, 0, 17], [0, 17, 33]
    gen = torch.Generator().manual_seed(841)

    def tensor(shape):
        return torch.randn(shape, generator=gen, dtype=torch.float32).to(
            device="cuda", dtype=dtype
        )

    q = tensor((sum(first_q), h, d))
    if paged:
        k, v = tensor((9, hk, page, d)), tensor((9, hk, page, d))
        tables = torch.arange(9, dtype=torch.int32, device="cuda").reshape(b, 3)
    else:
        k, v = tensor((sum(first_k), hk, d)), tensor((sum(first_k), hk, d))
        tables = None
    out = torch.full(q.shape, torch.nan, dtype=dtype, device="cuda")
    lse = torch.full(
        (len(q), h) if packed_lse else (b, maxq, h), torch.nan, device="cuda"
    )
    ws = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    qlens = torch.tensor(first_q, dtype=torch.int32, device="cuda").reshape(b, 1, 1, 1)
    klens = torch.tensor(first_k, dtype=torch.int32, device="cuda").reshape(b, 1, 1, 1)
    qo = torch.tensor(
        [0, *torch.tensor(first_q).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    ko = torch.tensor(
        [0, *torch.tensor(first_k).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )

    def run():
        return cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            d**-0.5,
            ws,
            max_token_per_sequence=maxq,
            max_sequence_kv=capacity if paged else 33,
            actual_seq_lens_q=qlens,
            actual_seq_lens_kv=klens,
            batch_offsets_q=qo,
            batch_offsets_k=None if paged else ko,
            batch_offsets_units="tokens",
            block_tables=tables,
            causal=True,
            return_lse=True,
            lse_base="ln",
            out=out,
            lse=lse,
        )

    if (
        maxq == 1
        and hk != h
        and packed_lse
        and cudnn_prefill._cudnn_single_token_gqa_ragged_stats_broken()
    ):
        # Existing public guard documents this unrepresentable packed Stats recipe.
        with pytest.raises(ValueError, match="zero-length requests.*packed LSE"):
            run()
        return

    def check(qsizes, ksizes):
        qs = ks = 0
        for i, (nq, nk) in enumerate(zip(qsizes, ksizes, strict=True)):
            if nq:
                qb = q[qs : qs + nq].double().transpose(0, 1)
                if paged:
                    kb = (
                        k[i * 3 : (i + 1) * 3]
                        .permute(1, 0, 2, 3)
                        .reshape(hk, -1, d)[:, :nk]
                    )
                    vb = (
                        v[i * 3 : (i + 1) * 3]
                        .permute(1, 0, 2, 3)
                        .reshape(hk, -1, d)[:, :nk]
                    )
                else:
                    kb, vb = (
                        k[ks : ks + nk].transpose(0, 1),
                        v[ks : ks + nk].transpose(0, 1),
                    )
                kb, vb = (
                    kb.double().repeat_interleave(h // hk, 0),
                    vb.double().repeat_interleave(h // hk, 0),
                )
                scores = (qb @ kb.transpose(1, 2)) * d**-0.5
                row, col = (
                    torch.arange(nq, device="cuda"),
                    torch.arange(nk, device="cuda"),
                )
                scores.masked_fill_(col[None, :] > row[:, None] + nk - nq, -torch.inf)
                ref = (scores.softmax(-1) @ vb).transpose(0, 1)
                stat = scores.logsumexp(-1).transpose(0, 1)
                torch.testing.assert_close(
                    out[qs : qs + nq].double(), ref, atol=0.015, rtol=0.015
                )
                got = lse[qs : qs + nq] if packed_lse else lse[i, :nq]
                torch.testing.assert_close(got.double(), stat, atol=0.005, rtol=0.005)
            qs += nq
            ks += nk

    run()
    check(first_q, first_k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.mul_(-0.5)
    qlens.copy_(
        torch.tensor(second_q, device="cuda", dtype=torch.int32).reshape_as(qlens)
    )
    klens.copy_(
        torch.tensor(second_k, device="cuda", dtype=torch.int32).reshape_as(klens)
    )
    qo.copy_(
        torch.tensor(
            [0, *torch.tensor(second_q).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
    )
    ko.copy_(
        torch.tensor(
            [0, *torch.tensor(second_k).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
    )
    out.fill_(torch.nan)
    lse.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    check(second_q, second_k)
