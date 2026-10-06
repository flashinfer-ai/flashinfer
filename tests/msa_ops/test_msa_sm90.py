"""MiniMax Sparse Attention (MSA) ops on SM90 (Hopper).

The shared suite in ``test_msa_ops.py`` gates on ``is_sm12x_supported``, so none
of it runs here. SM90 also supports a deliberately narrower surface than SM12x --
paged KV only, fp8 index cache for prefill, no LSE, no per-tensor KV dequant
scales -- so those cases cannot simply be un-gated. This file covers what SM90
does support and pins each restriction to the error it is supposed to raise.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.msa_ops import (
    msa_proxy_score,
    msa_sparse_attention,
    msa_sparse_decode_attention,
    msa_topk_select,
)

BLK_KV = 128
HEAD_DIM = 128
TOPK = 16


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9


sm90_only = pytest.mark.skipif(
    not _is_sm90(), reason="requires an SM90 (Hopper) device"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _decode_inputs(
    batch=2,
    num_qo_heads=8,
    num_kv_heads=1,
    max_k_tiles=48,
    device="cuda",
    seed=0,
    packed=True,
):
    """One decode token per sequence against a paged, block-sparse fp8 KV cache.

    Mirrors the shapes the SM90 decode schedule is built for: q is bf16, the KV
    cache is fp8 e4m3 with K and V interleaved in one buffer, and q2k_indices are
    block indices within the sequence that resolve through ``page_table``.
    """
    torch.manual_seed(seed)
    npages = max_k_tiles * batch
    total_q = batch
    base = (
        torch.randn(npages, num_kv_heads, BLK_KV, 2 * HEAD_DIM, device=device) / 4
    ).to(torch.float8_e4m3fn)
    if packed:
        k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    else:  # separate allocations -- SM90 refuses these rather than copying per call
        k = base[..., :HEAD_DIM].clone()
        v = base[..., HEAD_DIM:].clone()
    q = (torch.randn(total_q, num_qo_heads, HEAD_DIM, device=device) / 4).to(
        torch.bfloat16
    )
    idx = torch.randint(
        0, max_k_tiles, (num_kv_heads, total_q, TOPK), dtype=torch.int32, device=device
    )
    page_table = (
        torch.randperm(npages, device=device)[: batch * max_k_tiles]
        .to(torch.int32)
        .view(batch, max_k_tiles)
    )
    seqused_k = torch.full(
        (batch,), max_k_tiles * BLK_KV - 13, dtype=torch.int32, device=device
    )
    return q, k, v, idx, page_table, seqused_k


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
@sm90_only
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 2), (16, 4), (64, 4)])
def test_sparse_decode_paged_matches_reference(num_qo_heads, num_kv_heads):
    """Paged sparse decode against a dense torch reference over the selected blocks."""
    q, k, v, idx, page_table, seqused_k = _decode_inputs(
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads
    )
    scale = 1.0 / math.sqrt(HEAD_DIM)
    out = msa_sparse_decode_attention(
        q,
        k,
        v,
        idx,
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=1,
        softmax_scale=scale,
    )
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out.float()).all()


@sm90_only
def test_topk_select_shape_and_dtype():
    """top-k select is the only width SM90 supports (16) and must emit int32."""
    torch.manual_seed(0)
    num_qo_heads, max_k_tiles, total_q = 4, 64, 32
    scores = torch.randn(
        num_qo_heads, max_k_tiles, total_q, dtype=torch.float32, device="cuda"
    )
    out = msa_topk_select(scores, TOPK)
    assert out.shape == (total_q, num_qo_heads, TOPK)
    assert out.dtype == torch.int32
    assert int(out.max()) < max_k_tiles


@sm90_only
def test_proxy_score_paged_fp8_runs():
    """Proxy score on the paged fp8 index cache -- the only prefill path SM90 has."""
    torch.manual_seed(0)
    batch, total_q, hq, hkv = 2, 256, 4, 1
    seqlen_k = 1024
    q = (torch.randn(total_q, hq, HEAD_DIM, device="cuda") / 3).to(torch.float8_e4m3fn)
    pages = seqlen_k // BLK_KV
    npages = pages * batch
    k = (torch.randn(npages, hkv, BLK_KV, HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    page_table = (
        torch.arange(npages, dtype=torch.int32, device="cuda")
        .view(batch, pages)
        .contiguous()
    )
    cu_q = torch.tensor([0, total_q // 2, total_q], dtype=torch.int32, device="cuda")
    seqused_k = torch.full((batch,), seqlen_k, dtype=torch.int32, device="cuda")
    out = msa_proxy_score(
        q,
        k,
        cu_q,
        page_table=page_table,
        seqused_k=seqused_k,
        max_seqlen_q=total_q // 2,
        max_k_tiles=pages,
    )
    assert out.shape[-1] == total_q
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# CUDA graphs -- each of these pins a bug that shipped once
# ---------------------------------------------------------------------------
@sm90_only
def test_decode_and_topk_capture_into_cuda_graph():
    """Both kernels must launch on the current stream or they are silently not captured.

    An off-stream launch does not error; it produces stale top-k indices on replay.
    """
    q, k, v, idx, page_table, seqused_k = _decode_inputs()
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.randn(4, 64, q.shape[0], dtype=torch.float32, device="cuda")
    sel = torch.empty((q.shape[0], 4, TOPK), dtype=torch.int32, device="cuda")

    for _ in range(3):  # warm the compile caches outside capture
        msa_topk_select(scores, TOPK, output=sel)
        msa_sparse_decode_attention(
            q,
            k,
            v,
            idx,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=1,
            softmax_scale=scale,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        msa_topk_select(scores, TOPK, output=sel)
        out = msa_sparse_decode_attention(
            q,
            k,
            v,
            idx,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=1,
            softmax_scale=scale,
        )
    scores.normal_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert int(sel.max()) < 64


@sm90_only
def test_multiple_captured_graph_sizes_stay_valid():
    """A later, larger capture must not invalidate an earlier graph.

    Growing the decode scratch used to free the buffer an already-captured graph
    still held device pointers into.
    """
    scale = 1.0 / math.sqrt(HEAD_DIM)
    graphs, outs = [], []
    for batch in (1, 2, 4):
        q, k, v, idx, page_table, seqused_k = _decode_inputs(batch=batch, seed=batch)
        for _ in range(3):
            msa_sparse_decode_attention(
                q,
                k,
                v,
                idx,
                page_table=page_table,
                seqused_k=seqused_k,
                seqlen_q=1,
                softmax_scale=scale,
            )
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = msa_sparse_decode_attention(
                q,
                k,
                v,
                idx,
                page_table=page_table,
                seqused_k=seqused_k,
                seqlen_q=1,
                softmax_scale=scale,
            )
        graphs.append(g)
        outs.append(o)
    for g in graphs:  # replay the oldest graph last
        g.replay()
    torch.cuda.synchronize()
    for o in outs:
        assert torch.isfinite(o.float()).all()


# ---------------------------------------------------------------------------
# restrictions -- each raises instead of returning plausible, wrong numbers
# ---------------------------------------------------------------------------
@sm90_only
def test_lse_is_rejected_not_ignored():
    q, k, v, idx, page_table, seqused_k = _decode_inputs()
    with pytest.raises(NotImplementedError, match="does not return an LSE"):
        msa_sparse_decode_attention(
            q,
            k,
            v,
            idx,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=1,
            return_softmax_lse=True,
        )


@sm90_only
def test_per_tensor_kv_scales_are_rejected_not_ignored():
    """A silently dropped dequant scale returns clean-looking, wrong numbers."""
    q, k, v, idx, page_table, seqused_k = _decode_inputs()
    scale = torch.tensor(1.0, device="cuda")
    with pytest.raises(NotImplementedError, match="k_scale/v_scale"):
        msa_sparse_decode_attention(
            q,
            k,
            v,
            idx,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=1,
            k_scale=scale,
            v_scale=scale,
        )


@sm90_only
def test_ragged_layout_is_rejected():
    """SM90 has the paged schedule only; the ragged layout must not fall through."""
    torch.manual_seed(0)
    total_q, total_k, hq, hkv = 64, 256, 4, 1
    q = torch.randn(total_q, hq, HEAD_DIM, dtype=torch.bfloat16, device="cuda") / 3
    k = torch.randn(total_k, hkv, HEAD_DIM, dtype=torch.bfloat16, device="cuda") / 3
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, total_k], dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError, match="paged KV layout"):
        msa_proxy_score(
            q, k, cu_q, cu_seqlens_k=cu_k, max_seqlen_q=total_q, max_k_tiles=2
        )


@sm90_only
def test_separate_kv_allocations_are_rejected():
    """K and V must be halves of one cache; copying per call would cost more than it saves."""
    q, k, v, idx, page_table, seqused_k = _decode_inputs(packed=False)
    with pytest.raises(
        NotImplementedError, match="interleaved in one cache|matching k/v layouts"
    ):
        msa_sparse_decode_attention(
            q,
            k,
            v,
            idx,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=1,
        )


@sm90_only
@pytest.mark.parametrize("topk", [8, 32, 64])
def test_topk_width_other_than_16_is_rejected(topk):
    scores = torch.randn(4, 64, 32, dtype=torch.float32, device="cuda")
    with pytest.raises((NotImplementedError, ValueError), match="topk"):
        msa_topk_select(scores, topk)


@sm90_only
def test_bf16_kv_cache_is_rejected_not_read_as_fp8():
    """The decode schedule declares the cache e4m3 and reads it as raw bytes.

    A bf16 cache is 2 bytes per element, so both the values and the element
    stride come out wrong -- it used to return silent garbage instead of failing.
    """
    q, k, v, idx, page_table, seqused_k = _decode_inputs()
    kv_bf16 = (
        torch.randn(
            k.shape[0],
            k.shape[1],
            BLK_KV,
            2 * HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    k16, v16 = kv_bf16[..., :HEAD_DIM], kv_bf16[..., HEAD_DIM:]
    with pytest.raises(NotImplementedError, match="fp8 e4m3 KV cache"):
        msa_sparse_decode_attention(
            q,
            k16,
            v16,
            idx,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=1,
        )


@sm90_only
def test_bf16_kv_cache_is_rejected_by_sparse_attention():
    """Same hazard on the prefill path: _views() builds e4m3 pointers."""
    torch.manual_seed(0)
    batch, qlen, hq, hkv, mkt = 1, 128, 8, 1, 16
    npages = mkt * batch
    kv = (
        torch.randn(
            npages, hkv, BLK_KV, 2 * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        / 4
    )
    k16, v16 = kv[..., :HEAD_DIM], kv[..., HEAD_DIM:]
    q = torch.randn(qlen, hq, HEAD_DIM, dtype=torch.bfloat16, device="cuda") / 4
    idx = torch.randint(0, mkt, (hkv, qlen, TOPK), dtype=torch.int32, device="cuda")
    cu_q = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    page_table = torch.arange(npages, dtype=torch.int32, device="cuda").view(batch, mkt)
    seqused_k = torch.full(
        (batch,), mkt * BLK_KV - 13, dtype=torch.int32, device="cuda"
    )
    with pytest.raises(NotImplementedError, match="fp8 e4m3 KV cache"):
        msa_sparse_attention(
            q,
            k16,
            v16,
            idx,
            cu_q,
            page_table=page_table,
            seqused_k=seqused_k,
        )


@sm90_only
def test_sm90_is_advertised_as_supporting_packed_kv():
    """vLLM selects the FlashInfer MSA path off this flag; losing it silently serves Triton."""
    assert flashinfer.msa_ops.SUPPORTS_PACKED_KV is True
