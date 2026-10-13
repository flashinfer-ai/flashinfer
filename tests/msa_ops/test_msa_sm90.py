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


def _ref_sparse_decode(q, kv, idx, page_table, seqused_k, scale):
    """Dense torch reference over the selected blocks, in fp32."""
    total_q, hq, d = q.shape
    hkv = kv.shape[1]
    g = hq // hkv
    out = torch.zeros(total_q, hq, d, dtype=torch.float32, device=q.device)
    k_all = kv[..., :d].float()
    v_all = kv[..., d:].float()
    for b in range(total_q):
        qpos = int(seqused_k[b]) - 1
        for h in range(hq):
            kh = h // g
            keys, vals = [], []
            for blk in idx[kh, b].tolist():
                if blk < 0 or blk >= page_table.shape[1]:
                    continue
                pg = int(page_table[b, blk])
                base = blk * BLK_KV
                for t in range(BLK_KV):
                    if base + t <= qpos:  # causal
                        keys.append(k_all[pg, kh, t])
                        vals.append(v_all[pg, kh, t])
            if not keys:
                continue
            kmat = torch.stack(keys)
            vmat = torch.stack(vals)
            logits = (q[b, h].float() @ kmat.T) * scale
            p = torch.softmax(logits, dim=-1)
            out[b, h] = p @ vmat
    return out


@sm90_only
def test_sparse_decode_matches_dense_reference():
    """Numerical check against a dense softmax over the selected blocks."""
    q, k, v, idx, page_table, seqused_k = _decode_inputs(
        batch=2, num_qo_heads=8, num_kv_heads=1, max_k_tiles=8, seed=7
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
    # rebuild the packed cache the reference walks
    kv = torch.cat([k, v], dim=-1)
    ref = _ref_sparse_decode(q, kv, idx, page_table, seqused_k, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@sm90_only
@pytest.mark.parametrize("max_k_tiles", [16, 1024])
def test_sparse_prefill_runs_both_backends(max_k_tiles):
    """Exercise the union backend (mpg <= 800) and the single-token one above it."""
    torch.manual_seed(max_k_tiles)
    batch, qlen, hq, hkv = 1, 256, 8, 1
    npages = max_k_tiles * batch
    base = (torch.randn(npages, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    q = (torch.randn(qlen, hq, HEAD_DIM, device="cuda") / 4).to(torch.bfloat16)
    idx = torch.randint(
        0, max_k_tiles, (hkv, qlen, TOPK), dtype=torch.int32, device="cuda"
    )
    cu_q = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    page_table = (
        torch.randperm(npages, device="cuda")[: batch * max_k_tiles]
        .to(torch.int32)
        .view(batch, max_k_tiles)
    )
    seqused_k = torch.full(
        (batch,), max_k_tiles * BLK_KV - 13, dtype=torch.int32, device="cuda"
    )
    out = msa_sparse_attention(
        q,
        k,
        v,
        idx,
        cu_q,
        page_table=page_table,
        seqused_k=seqused_k,
        causal=True,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
    )
    assert out.shape == q.shape
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
@pytest.mark.parametrize("fill", ["nan", "nan_k_only", "nan_v_only", "garbage", "zeros"])
def test_prefill_ignores_page_bytes_past_seqused_k(fill):
    """Page memory beyond seqused_k must not reach the result.

    vLLM hands out pages whose tail is uninitialized. The kernel reads whole
    128-token pages, so without a causal bound on the selected blocks -- and with
    an additive mask, which cannot clear a NaN -- that tail leaks into the
    softmax. NaN there poisons the whole row.
    """
    torch.manual_seed(0)
    batch, qlen, hq, hkv, mkt = 1, 128, 8, 1, 8
    npages = mkt * batch
    kv = (torch.randn(npages, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4)
    # sequence ends mid-page; everything after it is "uninitialized"
    seqlen = mkt * BLK_KV - 50
    last_pg, off = seqlen // BLK_KV, seqlen % BLK_KV
    if fill == "nan":
        kv[last_pg, :, off:, :] = float("nan")
    elif fill == "nan_k_only":          # K tail NaN, V tail finite
        kv[last_pg, :, off:, :HEAD_DIM] = float("nan")
    elif fill == "nan_v_only":          # V tail NaN, K tail finite
        kv[last_pg, :, off:, HEAD_DIM:] = float("nan")
    elif fill == "garbage":
        kv[last_pg, :, off:, :] = 1e4
    else:
        kv[last_pg, :, off:, :] = 0.0
    kv = kv.to(torch.float8_e4m3fn)
    k, v = kv[..., :HEAD_DIM], kv[..., HEAD_DIM:]
    q = (torch.randn(qlen, hq, HEAD_DIM, device="cuda") / 4).to(torch.bfloat16)
    idx = torch.randint(0, mkt, (hkv, qlen, TOPK), dtype=torch.int32, device="cuda")
    cu_q = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    page_table = torch.arange(npages, dtype=torch.int32, device="cuda").view(batch, mkt)
    seqused_k = torch.full((batch,), seqlen, dtype=torch.int32, device="cuda")
    out = msa_sparse_attention(
        q, k, v, idx, cu_q, page_table=page_table, seqused_k=seqused_k,
        causal=True, softmax_scale=1.0 / math.sqrt(HEAD_DIM),
    )
    assert torch.isfinite(out.float()).all(), f"{fill} tail leaked into the output"


@sm90_only
@pytest.mark.parametrize("pattern", ["all_equal", "few_distinct", "tied_top16", "ties_with_pad"])
def test_topk_select_returns_distinct_blocks_under_ties(pattern):
    """Exact score ties must not yield duplicate block indices.

    The final scatter folds the lane into its sort key so it is tie-safe, but the
    cascade merge keeps (score, index) pairs with a plain >= on the score; a tile
    reachable from two partial lists can therefore be emitted twice, leaving
    fewer than topk distinct blocks.
    """
    torch.manual_seed(0)
    num_qo_heads, max_k_tiles, total_q = 4, 64, 32
    if pattern == "all_equal":
        scores = torch.full((num_qo_heads, max_k_tiles, total_q), 1.0,
                            dtype=torch.float32, device="cuda")
    elif pattern == "few_distinct":                     # 4 values over 64 tiles
        base = torch.tensor([3.0, 2.0, 1.0, 0.0], device="cuda")
        scores = base.repeat(max_k_tiles // 4)[None, :, None].expand(
            num_qo_heads, max_k_tiles, total_q).contiguous()
    elif pattern == "tied_top16":                       # 16 tied winners, rest lower
        scores = torch.zeros(num_qo_heads, max_k_tiles, total_q,
                             dtype=torch.float32, device="cuda")
        scores[:, :16, :] = 5.0
    else:                                               # ties plus padding
        scores = torch.full((num_qo_heads, max_k_tiles, total_q), 2.0,
                            dtype=torch.float32, device="cuda")
        scores[:, 8:, :] = float("-inf")                # only 8 valid tiles
    out = msa_topk_select(scores, TOPK)
    bad = []
    for t in range(total_q):
        for h in range(num_qo_heads):
            sel = out[t, h].tolist()
            real = [b for b in sel if b >= 0]
            if len(real) != len(set(real)):
                dup = {b for b in real if real.count(b) > 1}
                bad.append((t, h, sorted(dup), len(set(real))))
    assert not bad, (
        f"{len(bad)} (token, head) slots have duplicate blocks; "
        f"first: token={bad[0][0]} head={bad[0][1]} dups={bad[0][2]} "
        f"distinct={bad[0][3]}"
    )


@sm90_only
@pytest.mark.parametrize("hq,tiles,nq", [(1, 64, 1536), (1, 512, 8192)])
def test_topk_select_20way_tie_reviewer_repro(hq, tiles, nq):
    """Reviewer repro for #6140: a 20-way tie at the top score.

    q = 22 everywhere; the first 20 blocks carry k = 22, giving an exact block
    score of 128 * 22^2 = 61952, and every other block k = 11 -> 30976. Twenty
    blocks tie for sixteen slots. Observed before the fix: duplicated indices and
    fewer than sixteen distinct blocks, e.g. [0, 0, 16 x8, 17 x6]. The failure is
    plan dependent, hence both shapes.
    """
    hi, lo = 128.0 * 22.0 * 22.0, 128.0 * 11.0 * 11.0
    scores = torch.full((hq, tiles, nq), lo, dtype=torch.float32, device="cuda")
    scores[:, :20, :] = hi
    out = msa_topk_select(scores, TOPK)
    assert out.shape == (nq, hq, TOPK)
    sel = out.reshape(-1, TOPK)
    # every slot must be a real block (20 candidates >= 16 slots) and distinct
    assert int(sel.min()) >= 0, "padding emitted despite 20 valid candidates"
    srt, _ = sel.sort(dim=-1)
    dup_rows = (srt[:, 1:] == srt[:, :-1]).any(dim=-1)
    if bool(dup_rows.any()):
        i = int(dup_rows.nonzero()[0])
        row = sel[i].tolist()
        raise AssertionError(
            f"{int(dup_rows.sum())}/{sel.shape[0]} rows duplicate a block; "
            f"row {i} = {row} ({len(set(row))} distinct)"
        )
    # all winners must come from the tied top-20
    assert int(sel.max()) < 20, f"selected a non-tied block: max={int(sel.max())}"


@sm90_only
def test_sm90_is_advertised_as_supporting_packed_kv():
    """vLLM selects the FlashInfer MSA path off this flag; losing it silently serves Triton."""
    assert flashinfer.msa_ops.SUPPORTS_PACKED_KV is True
