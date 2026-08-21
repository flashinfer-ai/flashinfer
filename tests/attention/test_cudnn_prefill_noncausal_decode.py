"""Evidence tests: cuDNN non-causal multi-token decode with paged KV cache.

FlashInfer issues #3570 (DFlash / diffusion-LM block decoding) and #3335
(non-causal speculative decode) ask for non-causal / bidirectional paged GQA
generation kernels for multi-token decode (q_len per request <= 16).
``cudnn_batch_prefill_with_kv_cache`` already exposes ``causal: bool`` and
handles short packed queries against long paged KV, so these tests demonstrate
that capability via the cuDNN backend today:

- ``causal=False`` decode-like batches: short varied q (1..16 tokens per
  request) against long varied KV contexts (hundreds to 2048 tokens), every q
  token attending the full valid KV prefix (bidirectional block-decode
  semantics). Checked against a dense fp32 torch reference that applies only
  the KV padding mask (no causal mask).
- ``causal=True`` at q_len <= 16: cuDNN's causal flag on this API is
  bottom-right-aligned -- the diagonal ends at the last *valid* KV token of
  each request -- which is the masking speculative decode needs. The reference
  aligns the diagonal to the END of the KV sequence, and the test additionally
  asserts a top-left-aligned reference would NOT match, so the alignment claim
  is load-bearing.
- ``return_lse=True``: LSE shape and finiteness on the valid rows
  (merging-readiness for speculative decode).
"""

import pytest
import torch

from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.cudnn import prefill as cudnn_prefill

MAX_Q_LEN = 16  # multi-token decode: q_len per request <= 16
MAX_KV_LEN = 2048
MIN_KV_LEN = 256
HEAD_DIM = 128


def _make_decode_like_inputs(batch_size, page_size, num_qo_heads, num_kv_heads, device):
    """Decode-like batch: short packed q, long paged KV, varied per request."""
    torch.manual_seed(42)
    seq_q = torch.randint(
        1, MAX_Q_LEN + 1, (batch_size,), dtype=torch.int32, device=device
    )
    # Guarantee the q-length extremes the block-decode scenario cares about.
    seq_q[0] = 1
    seq_q[1] = MAX_Q_LEN
    seq_kv = torch.randint(
        MIN_KV_LEN, MAX_KV_LEN + 1, (batch_size,), dtype=torch.int32, device=device
    )
    seq_kv[-1] = MAX_KV_LEN  # one full-length context

    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(seq_q, 0)]).int()  # token-unit

    q = torch.randn(
        int(seq_q.sum()), num_qo_heads, HEAD_DIM, device=device, dtype=torch.bfloat16
    )

    num_pages_per_seq = (MAX_KV_LEN + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_cache = torch.randn(
        total_num_pages,
        2,
        num_kv_heads,
        page_size,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    strides = (
        2 * page_size * num_kv_heads * HEAD_DIM,
        HEAD_DIM,
        num_kv_heads * HEAD_DIM,
        1,
    )
    k_cache = kv_cache[:, 0].as_strided(kv_cache[:, 0].shape, strides)
    v_cache = kv_cache[:, 1].as_strided(kv_cache[:, 1].shape, strides)
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int32,
        device=device,
    )
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        qo_indptr=qo_indptr,
        seq_q=seq_q,
        seq_kv=seq_kv,
        page_size=page_size,
    )


def _dense_reference(inp, scale, causal_alignment=None):
    """Dense fp32 torch attention over the valid KV prefix of each request.

    Gathering only the valid KV tokens applies exactly the padding mask (and
    nothing else); ``causal_alignment`` adds a causal mask on top:
    ``None`` (bidirectional), ``"bottom_right"`` (diagonal aligned to the end
    of the valid KV sequence, cuDNN's convention on this API), or
    ``"top_left"`` (diagonal at kv position 0, used only to prove the
    alignments differ). Returns packed output ``(total_q_tokens, h_qo, d)``.
    """
    q = inp["q"]
    k_cache, v_cache = inp["k_cache"], inp["v_cache"]
    block_tables, page_size = inp["block_tables"], inp["page_size"]
    seq_q, seq_kv = inp["seq_q"].tolist(), inp["seq_kv"].tolist()
    num_qo_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    outs = []
    q_start = 0
    for i, (len_q, len_kv) in enumerate(zip(seq_q, seq_kv, strict=False)):
        q_i = q[q_start : q_start + len_q].float()  # (len_q, h_qo, d)
        q_start += len_q
        num_pages = (len_kv + page_size - 1) // page_size
        pages = block_tables[i, :num_pages]
        # (num_pages, h_kv, page_size, d) -> (len_kv, h_kv, d), valid prefix only
        k_i = k_cache[pages].permute(0, 2, 1, 3).reshape(-1, num_kv_heads, HEAD_DIM)
        v_i = v_cache[pages].permute(0, 2, 1, 3).reshape(-1, num_kv_heads, HEAD_DIM)
        k_i = k_i[:len_kv].float()
        v_i = v_i[:len_kv].float()
        # GQA: expand kv heads to qo heads
        rep = num_qo_heads // num_kv_heads
        k_i = k_i.repeat_interleave(rep, dim=1)
        v_i = v_i.repeat_interleave(rep, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
        if causal_alignment is not None:
            q_pos = torch.arange(len_q, device=q.device)
            kv_pos = torch.arange(len_kv, device=q.device)
            if causal_alignment == "bottom_right":
                # q token j attends kv positions <= len_kv - len_q + j
                allowed = kv_pos[None, :] <= (len_kv - len_q) + q_pos[:, None]
            elif causal_alignment == "top_left":
                allowed = kv_pos[None, :] <= q_pos[:, None]
            else:
                raise ValueError(f"unknown causal_alignment: {causal_alignment}")
            scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v_i))
    return torch.cat(outs, dim=0)


def _run_cudnn(inp, scale, causal, return_lse, lse=None):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=inp["q"].device)
    batch_size = inp["seq_q"].shape[0]
    return cudnn_batch_prefill_with_kv_cache(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        scale,
        ws,
        max_token_per_sequence=MAX_Q_LEN,
        max_sequence_kv=MAX_KV_LEN,
        actual_seq_lens_q=inp["seq_q"].view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=inp["seq_kv"].view(batch_size, 1, 1, 1),
        block_tables=inp["block_tables"],
        causal=causal,
        return_lse=return_lse,
        batch_offsets_q=inp["qo_indptr"],
        batch_offsets_units="tokens",
        lse=lse,
    )


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 2), (32, 8)])
def test_cudnn_noncausal_multi_token_decode(
    batch_size, page_size, num_qo_heads, num_kv_heads
):
    """causal=False, q_len 1..16 per request, long paged KV: every q token
    attends the full valid KV prefix (bidirectional block-decode semantics)."""
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")

    device = "cuda:0"
    inp = _make_decode_like_inputs(
        batch_size, page_size, num_qo_heads, num_kv_heads, device
    )
    scale = float(HEAD_DIM**-0.5)

    output, _ = _run_cudnn(inp, scale, causal=False, return_lse=False)

    output_ref = _dense_reference(inp, scale, causal_alignment=None)
    torch.testing.assert_close(output.float(), output_ref, atol=1e-2, rtol=1e-2)


def test_cudnn_causal_bottom_right_multi_token_decode():
    """causal=True at q_len <= 16 (speculative-decode masking sanity): the
    causal flag is bottom-right-aligned, i.e. the diagonal is aligned to the
    end of each request's valid KV sequence."""
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")

    device = "cuda:0"
    inp = _make_decode_like_inputs(
        batch_size=4, page_size=16, num_qo_heads=8, num_kv_heads=2, device=device
    )
    scale = float(HEAD_DIM**-0.5)

    output, _ = _run_cudnn(inp, scale, causal=True, return_lse=False)

    output_ref = _dense_reference(inp, scale, causal_alignment="bottom_right")
    torch.testing.assert_close(output.float(), output_ref, atol=1e-2, rtol=1e-2)

    # The alignment assertion is only meaningful if the two conventions
    # actually disagree on this data: top-left would confine each q token to
    # the first <=16 KV tokens instead of the full context.
    output_ref_top_left = _dense_reference(inp, scale, causal_alignment="top_left")
    assert not torch.allclose(output_ref, output_ref_top_left, atol=1e-2, rtol=1e-2), (
        "top-left and bottom-right references coincide; alignment check is vacuous"
    )


def test_cudnn_noncausal_multi_token_decode_return_lse():
    """return_lse=True on the non-causal decode-like batch: correct output plus
    an LSE of the documented shape whose valid rows are finite and written
    (merging-readiness for speculative decode)."""
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")

    device = "cuda:0"
    batch_size = 4
    num_qo_heads = 8
    inp = _make_decode_like_inputs(
        batch_size,
        page_size=32,
        num_qo_heads=num_qo_heads,
        num_kv_heads=2,
        device=device,
    )
    scale = float(HEAD_DIM**-0.5)

    # Pre-fill with NaN so finiteness on valid rows also proves the kernel
    # wrote them (rows past each request's q length stay unspecified).
    lse_buf = torch.full(
        (batch_size, MAX_Q_LEN, num_qo_heads),
        float("nan"),
        device=device,
        dtype=torch.float32,
    )
    output, lse = _run_cudnn(inp, scale, causal=False, return_lse=True, lse=lse_buf)

    output_ref = _dense_reference(inp, scale, causal_alignment=None)
    torch.testing.assert_close(output.float(), output_ref, atol=1e-2, rtol=1e-2)

    assert lse is not None
    assert lse.shape == (batch_size, MAX_Q_LEN, num_qo_heads)
    for i, len_q in enumerate(inp["seq_q"].tolist()):
        assert torch.isfinite(lse[i, :len_q, :]).all(), (
            f"non-finite LSE in valid rows of request {i}"
        )
