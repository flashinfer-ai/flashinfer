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


@pytest.mark.parametrize("causal", [True, False])
def test_ragged_wrapper_cudnn_return_lse(causal):
    """``BatchPrefillWithRaggedKVCacheWrapper(backend="cudnn").run(...,
    return_lse=True)`` returns the ``[qo_len, num_qo_heads]`` LSE the wrapper
    documents (it allocates that shape and hands it to the cuDNN path), matching
    the default backend's."""
    _skip_unless_cudnn()
    device = "cuda:0"
    batch_size, s_qo, s_kv, h_qo, h_kv, d = 4, 87, 512, 8, 4, 128
    q_lens, kv_lens, qo_indptr, kv_indptr, q, k, v = _ragged_case(
        batch_size, s_qo, s_kv, h_qo, h_kv, d, device
    )
    scale = float(d**-0.5)
    ws = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)

    # The cuDNN ragged wrapper takes element-unit indptrs (see
    # test_cudnn_prefill_deepseek.py); the default backend takes token units.
    w_cudnn = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, "NHD", backend="cudnn"
    )
    w_cudnn.plan(
        qo_indptr=qo_indptr * (h_qo * d),
        kv_indptr=kv_indptr * (h_kv * d),
        num_qo_heads=h_qo,
        num_kv_heads=h_kv,
        head_dim_qk=d,
        head_dim_vo=d,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        seq_lens=kv_lens,
        seq_lens_q=q_lens,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        v_indptr=kv_indptr * (h_kv * d),
        o_indptr=qo_indptr * (h_qo * d),
    )
    out, lse = w_cudnn.run(q, k, v, return_lse=True)

    w_ref = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        ws.clone(), kv_layout="NHD"
    )
    w_ref.plan(
        qo_indptr,
        kv_indptr,
        h_qo,
        h_kv,
        d,
        head_dim_vo=d,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out_ref, lse_ref = w_ref.run(q, k, v, return_lse=True)

    assert lse.shape == (q.shape[0], h_qo) == lse_ref.shape
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-2)
