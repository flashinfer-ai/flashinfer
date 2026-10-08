"""MiniMax Sparse Attention (MSA) ops on SM90 (Hopper).

The shared suite in ``test_msa_ops.py`` gates on ``is_sm12x_supported``, so none
of it runs here. SM90 also supports a deliberately narrower surface than SM12x --
paged KV only, fp8 index cache for prefill, no LSE, no per-tensor KV dequant
scales -- so those cases cannot simply be un-gated. This file covers what SM90
does support and pins each restriction to the error it is supposed to raise.

The reference checks compare every SM90 operation against an FP32 torch oracle
over the same selected blocks (bf16 outputs at atol = rtol = 1e-2, top-k
indices exactly), including both accumulation precisions of the proxy-score
prefill regime (``use_fp32_acc``).
"""

import itertools
import math
import warnings

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
    distinct ascending block indices within the sequence (the format
    ``msa_topk_select`` produces) that resolve through ``page_table``.
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
    idx = (
        torch.rand(num_kv_heads, total_q, max_k_tiles, device=device)
        .argsort(dim=-1)[..., :TOPK]
        .sort(dim=-1)
        .values.to(torch.int32)
        .contiguous()
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


def _proxy_prefill_inputs(batch=2, chunk=512, ctx=2048, hq=4, seed=0):
    """A chunked-prefill scoring call: ``chunk`` fp8 query tokens per sequence
    sitting at the end of a ``ctx``-token paged fp8 index cache (one index head),
    permuted page table, a page tail of 37 tokens."""
    torch.manual_seed(seed)
    pages = ctx // BLK_KV
    npages = pages * batch
    total_q = chunk * batch
    q = (torch.randn(total_q, hq, HEAD_DIM, device="cuda") / 3).to(torch.float8_e4m3fn)
    k = (torch.randn(npages, 1, BLK_KV, HEAD_DIM, device="cuda") / 3).to(
        torch.float8_e4m3fn
    )
    page_table = (
        torch.randperm(npages, device="cuda").to(torch.int32).view(batch, pages)
    )
    cu_q = torch.arange(0, total_q + 1, chunk, dtype=torch.int32, device="cuda")
    seqused_k = torch.full((batch,), ctx - 37, dtype=torch.int32, device="cuda")
    return q, k, cu_q, page_table, seqused_k, pages


def _proxy_prefill_reference(q, k, cu_q, page_table, seqused_k, pages):
    """FP32 per-block causal max of ``Q K^T`` (query ``i`` of sequence ``b`` at
    position ``seqused_k[b] - qlen_b + i``); blocks without a visible key are -inf."""
    hq = q.shape[1]
    batch = seqused_k.numel()
    ref = torch.full((hq, pages, q.shape[0]), float("-inf"), device=q.device)
    qf = q.float()
    kf = k.float()[:, 0]
    for b in range(batch):
        q0, q1 = int(cu_q[b]), int(cu_q[b + 1])
        kv_len = int(seqused_k[b])
        qpos = torch.arange(kv_len - (q1 - q0), kv_len, device=q.device)
        for t in range(pages):
            kpos = t * BLK_KV + torch.arange(BLK_KV, device=q.device)
            visible = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] < kv_len)
            if not bool(visible.any()):
                continue
            logits = torch.einsum("qhd,kd->hqk", qf[q0:q1], kf[int(page_table[b, t])])
            logits = logits.masked_fill(~visible[None], float("-inf"))
            ref[:, t, q0:q1] = logits.amax(dim=-1)
    return ref


def _topk_reference(scores, nvp, fb, fe):
    """Exact top-k over the valid blocks of every (token, head): forced sink /
    window blocks first, the largest remaining scores, ascending, -1 padded."""
    hq, tiles, total_q = scores.shape
    out = torch.full((total_q, hq, TOPK), -1, dtype=torch.int32, device=scores.device)
    for t in range(total_q):
        valid = tiles if nvp is None else int(nvp[t])
        forced = sorted(set(range(fb)) | set(range(max(valid - fe, 0), valid)))
        for h in range(hq):
            column = scores[h, :valid, t]
            finite = torch.isfinite(column)
            cands = [i for i in range(valid) if bool(finite[i]) and i not in forced]
            chosen = list(forced)
            take = min(TOPK - len(chosen), len(cands))
            if take > 0:
                picks = torch.topk(column[cands], take).indices.tolist()
                chosen.extend(cands[i] for i in picks)
            chosen.sort()
            out[t, h, : len(chosen)] = torch.tensor(
                chosen, dtype=torch.int32, device=scores.device
            )
    return out


def _prefill_inputs(hq, hkv, pages, q_lens, seed=0):
    """Ragged sparse-prefill batch against a paged, interleaved fp8 KV cache:
    every token selects block 0 plus 15 distinct other blocks of its sequence."""
    torch.manual_seed(seed)
    batch = len(q_lens)
    npages = pages * batch
    total_q = sum(q_lens)
    base = (torch.randn(npages, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 3).to(
        torch.float8_e4m3fn
    )
    k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    q = (torch.randn(total_q, hq, HEAD_DIM, device="cuda") / 3).to(torch.bfloat16)
    page_table = (
        torch.randperm(npages, device="cuda").to(torch.int32).view(batch, pages)
    )
    seqused_k = torch.tensor(
        [pages * BLK_KV - 13 * (b + 1) for b in range(batch)],
        dtype=torch.int32,
        device="cuda",
    )
    cu_q = torch.tensor(
        [0] + list(itertools.accumulate(q_lens)), dtype=torch.int32, device="cuda"
    )
    idx = torch.empty(hkv, total_q, TOPK, dtype=torch.int32, device="cuda")
    for b in range(batch):
        nblk = (int(seqused_k[b]) + BLK_KV - 1) // BLK_KV
        q0, q1 = int(cu_q[b]), int(cu_q[b + 1])
        others = torch.rand(hkv, q1 - q0, nblk - 1, device="cuda").argsort(dim=-1)
        idx[:, q0:q1, 0] = 0
        idx[:, q0:q1, 1:] = (others[..., : TOPK - 1] + 1).to(torch.int32)
    return q, k, v, idx.contiguous(), cu_q, page_table, seqused_k


def _prefill_reference(q, k, v, idx, cu_q, page_table, seqused_k):
    """FP32 causal softmax attention of every token over its selected blocks."""
    total_q, hq, _ = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    kf, vf = k.float(), v.float()
    scale = 1.0 / math.sqrt(HEAD_DIM)
    ref = torch.empty(total_q, hq, HEAD_DIM, device=q.device)
    offsets = torch.arange(BLK_KV, device=q.device)
    for b in range(seqused_k.numel()):
        q0, q1 = int(cu_q[b]), int(cu_q[b + 1])
        kv_len = int(seqused_k[b])
        for i in range(q1 - q0):
            qpos = kv_len - (q1 - q0) + i
            for h in range(hkv):
                blocks = idx[h, q0 + i].long()
                phys = page_table[b, blocks].long()
                kpos = blocks[:, None] * BLK_KV + offsets[None, :]
                visible = (kpos <= qpos) & (kpos < kv_len)
                qh = q[q0 + i, h * group : (h + 1) * group].float()
                logits = torch.einsum("gd,bkd->gbk", qh, kf[phys, h]) * scale
                logits = logits.masked_fill(~visible[None], float("-inf"))
                probs = torch.softmax(logits.reshape(group, -1), dim=-1)
                ref[q0 + i, h * group : (h + 1) * group] = probs @ vf[phys, h].reshape(
                    -1, HEAD_DIM
                )
    return ref


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
@sm90_only
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 2), (16, 4), (64, 4)])
def test_sparse_decode_paged_matches_reference(num_qo_heads, num_kv_heads):
    """Paged sparse decode against the FP32 oracle over the selected blocks."""
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
    # one query per sequence: the prefill oracle with unit query lengths
    cu_q = torch.arange(q.shape[0] + 1, dtype=torch.int32, device="cuda")
    ref = _prefill_reference(q, k, v, idx, cu_q, page_table, seqused_k)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


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


@sm90_only
@pytest.mark.parametrize("use_fp32_acc", [True, False])
def test_proxy_score_prefill_matches_fp32_reference(use_fp32_acc):
    """Chunked-prefill scoring against the FP32 oracle in both accumulation
    precisions: f32 (default) and f16 (the CuTe DSL kernel's numerics); -inf
    blocks must match exactly."""
    q, k, cu_q, page_table, seqused_k, pages = _proxy_prefill_inputs()
    out = msa_proxy_score(
        q,
        k,
        cu_q,
        page_table=page_table,
        seqused_k=seqused_k,
        max_seqlen_q=512,
        max_k_tiles=pages,
        use_fp32_acc=use_fp32_acc,
    )
    ref = _proxy_prefill_reference(q, k, cu_q, page_table, seqused_k, pages)
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(out), finite)
    torch.testing.assert_close(out[finite], ref[finite], atol=1e-2, rtol=1e-2)


@sm90_only
def test_proxy_score_prefill_defaults_to_fp32_accumulation():
    """The keyword's default is the exact f32 path, bitwise."""
    q, k, cu_q, page_table, seqused_k, pages = _proxy_prefill_inputs(seed=1)
    common = dict(
        page_table=page_table, seqused_k=seqused_k, max_seqlen_q=512, max_k_tiles=pages
    )
    default = msa_proxy_score(q, k, cu_q, **common)
    fp32 = msa_proxy_score(q, k, cu_q, use_fp32_acc=True, **common)
    assert torch.equal(default, fp32)


@sm90_only
def test_f16_accumulation_warns_exactly_once_per_process():
    """``use_fp32_acc=False`` is an opt-in away from the required numerics: one
    RuntimeWarning per process, none for the default."""
    from flashinfer.msa_ops import proxy_score as proxy_score_module

    q, k, cu_q, page_table, seqused_k, pages = _proxy_prefill_inputs(seed=6)
    common = dict(
        page_table=page_table, seqused_k=seqused_k, max_seqlen_q=512, max_k_tiles=pages
    )
    proxy_score_module._F16_ACC_WARNED = False  # earlier tests may have tripped it
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msa_proxy_score(q, k, cu_q, use_fp32_acc=False, **common)
        msa_proxy_score(q, k, cu_q, use_fp32_acc=False, **common)
        msa_proxy_score(q, k, cu_q, use_fp32_acc=False, **common)
    f16 = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "use_fp32_acc=False" in str(w.message)
    ]
    assert len(f16) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msa_proxy_score(q, k, cu_q, **common)
        msa_proxy_score(q, k, cu_q, use_fp32_acc=True, **common)
    assert not [w for w in caught if "use_fp32_acc" in str(w.message)]


@sm90_only
def test_proxy_score_prefill_honours_q_offset():
    """An explicit per-sequence query offset replaces the end-of-sequence alignment."""
    q, k, cu_q, page_table, seqused_k, pages = _proxy_prefill_inputs(seed=2)
    batch = seqused_k.numel()
    offset = torch.full((batch,), 300, dtype=torch.int32, device="cuda")
    out = msa_proxy_score(
        q,
        k,
        cu_q,
        page_table=page_table,
        seqused_k=seqused_k,
        max_seqlen_q=512,
        max_k_tiles=pages,
        q_offset=offset,
    )
    # the reference aligns on seqused_k - qlen: emulate the offset by shortening
    # the sequence so that position 300 + i is where query i sits
    shortened = offset + 512
    ref = _proxy_prefill_reference(q, k, cu_q, page_table, shortened, pages)
    # keys beyond seqused_k are invalid for the kernel but not for the shortened
    # reference: restrict the comparison to the blocks both consider valid
    valid_pages = (int(seqused_k[0]) + BLK_KV - 1) // BLK_KV
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(out), finite)
    torch.testing.assert_close(
        out[:, :valid_pages][finite[:, :valid_pages]],
        ref[:, :valid_pages][finite[:, :valid_pages]],
        atol=1e-2,
        rtol=1e-2,
    )


@sm90_only
def test_proxy_score_prefill_int_q_offset_matches_tensor():
    """The documented integer ``q_offset`` is one offset for every sequence: it
    must score exactly like the equivalent int32 tensor, not be dropped."""
    q, k, cu_q, page_table, seqused_k, pages = _proxy_prefill_inputs(seed=7)
    common = dict(
        page_table=page_table, seqused_k=seqused_k, max_seqlen_q=512, max_k_tiles=pages
    )
    offset = torch.full((seqused_k.numel(),), 300, dtype=torch.int32, device="cuda")
    as_tensor = msa_proxy_score(q, k, cu_q, q_offset=offset, **common)
    as_int = msa_proxy_score(q, k, cu_q, q_offset=300, **common)
    assert torch.equal(as_int, as_tensor)
    assert not torch.equal(as_int, msa_proxy_score(q, k, cu_q, **common))


@sm90_only
def test_topk_select_matches_reference_from_inf_validity():
    """Validity read from the -inf tiles the proxy pass writes, including a token
    with fewer than 16 valid blocks (-1 padded)."""
    torch.manual_seed(0)
    hq, tiles, total_q = 2, 48, 37
    scores = torch.randn(hq, tiles, total_q, dtype=torch.float32, device="cuda")
    valid = torch.randint(16, tiles + 1, (total_q,), device="cuda")
    valid[5] = 10
    for t in range(total_q):
        scores[:, int(valid[t]) :, t] = float("-inf")
    out = msa_topk_select(scores, TOPK)
    assert torch.equal(out, _topk_reference(scores, None, 0, 0))


@sm90_only
def test_topk_select_matches_reference_with_valid_pages_and_forced_blocks():
    """Per-token num_valid_pages clamps the candidates; forced sink / window
    blocks are selected regardless of their scores."""
    torch.manual_seed(1)
    hq, tiles, total_q = 1, 64, 53
    scores = torch.randn(hq, tiles, total_q, dtype=torch.float32, device="cuda")
    nvp = torch.randint(20, tiles + 1, (total_q,), dtype=torch.int32, device="cuda")
    out = msa_topk_select(
        scores, TOPK, num_valid_pages=nvp, force_begin_blocks=1, force_end_blocks=2
    )
    assert torch.equal(out, _topk_reference(scores, nvp, 1, 2))


@sm90_only
@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads,pages,q_lens",
    [
        (8, 1, 16, (96, 160)),  # GQA 8, 16 pages per sequence
        (8, 2, 32, (64, 100)),  # GQA 4, two KV heads
        (16, 1, 80, (64, 100)),  # GQA 16, 80 pages per sequence
    ],
)
def test_sparse_prefill_matches_fp32_reference(
    num_qo_heads, num_kv_heads, pages, q_lens
):
    """Ragged paged sparse prefill against the FP32 oracle over the selected blocks."""
    q, k, v, idx, cu_q, page_table, seqused_k = _prefill_inputs(
        num_qo_heads, num_kv_heads, pages, q_lens
    )
    out = msa_sparse_attention(
        q, k, v, idx, cu_q, causal=True, page_table=page_table, seqused_k=seqused_k
    )
    assert out.shape == q.shape and out.dtype == q.dtype
    ref = _prefill_reference(q, k, v, idx, cu_q, page_table, seqused_k)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


@sm90_only
def test_sparse_prefill_writes_out_and_scales_v():
    """``out=`` is written in place (no allocation) and ``v_global_scale`` scales it."""
    q, k, v, idx, cu_q, page_table, seqused_k = _prefill_inputs(
        8, 1, 16, (40, 72), seed=3
    )
    plain = msa_sparse_attention(
        q, k, v, idx, cu_q, causal=True, page_table=page_table, seqused_k=seqused_k
    )
    out = torch.empty_like(q)
    returned = msa_sparse_attention(
        q,
        k,
        v,
        idx,
        cu_q,
        causal=True,
        page_table=page_table,
        seqused_k=seqused_k,
        out=out,
    )
    assert returned is out
    assert torch.equal(out, plain)
    scaled = msa_sparse_attention(
        q,
        k,
        v,
        idx,
        cu_q,
        causal=True,
        page_table=page_table,
        seqused_k=seqused_k,
        v_global_scale=0.5,
    )
    torch.testing.assert_close(
        scaled.float(), plain.float() * 0.5, atol=1e-2, rtol=1e-2
    )


@sm90_only
def test_sparse_prefill_int_q_offset_matches_tensor():
    """An integer ``q_offset`` places query ``i`` of every sequence at
    ``q_offset + i``: identical to the equivalent int32 tensor and to the FP32
    oracle aligned on that position, not silently right-aligned."""
    q, k, v, idx, cu_q, page_table, seqused_k = _prefill_inputs(
        8, 1, 16, (40, 72), seed=8
    )
    common = dict(causal=True, page_table=page_table, seqused_k=seqused_k)
    # 1950 keeps every selected block partially visible for every query of
    # both sequences and moves the first sequence away from its end-of-
    # sequence alignment (seqused_k - 40 = 1995)
    offset = torch.full((seqused_k.numel(),), 1950, dtype=torch.int32, device="cuda")
    as_tensor = msa_sparse_attention(q, k, v, idx, cu_q, q_offset=offset, **common)
    as_int = msa_sparse_attention(q, k, v, idx, cu_q, q_offset=1950, **common)
    assert torch.equal(as_int, as_tensor)
    assert not torch.equal(as_int, msa_sparse_attention(q, k, v, idx, cu_q, **common))
    # the oracle aligns on seqused_k - qlen: shorten the sequences so that
    # position 1950 + i is where query i sits (the keys it drops are above the
    # causal limit for the kernel too)
    shortened = offset + (cu_q[1:] - cu_q[:-1])
    ref = _prefill_reference(q, k, v, idx, cu_q, page_table, shortened)
    torch.testing.assert_close(as_int.float(), ref, atol=1e-2, rtol=1e-2)


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


@sm90_only
def test_prefill_pipeline_replays_bitwise_from_cuda_graph():
    """Proxy score (prefill regime), top-k select and sparse prefill capture into
    one graph and replay bitwise against their eager results."""
    # 32 pages so that every query of the 256-token chunk sees at least 16
    # finite blocks: the selection never carries -1 padding into the prefill
    q, k, cu_q, page_table, seqused_k, pages = _proxy_prefill_inputs(
        batch=2, chunk=256, ctx=4096, hq=1, seed=4
    )
    scores = torch.empty((1, pages, q.shape[0]), dtype=torch.float32, device="cuda")
    sel = torch.empty((q.shape[0], 1, TOPK), dtype=torch.int32, device="cuda")
    nvp = torch.randint(16, pages + 1, (q.shape[0],), dtype=torch.int32, device="cuda")
    aq, ak, av, _idx, acu, apt, ask = _prefill_inputs(8, 1, 32, (256, 256), seed=4)
    aout = torch.empty_like(aq)

    def pipeline():
        msa_proxy_score(
            q,
            k,
            cu_q,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_q=256,
            max_k_tiles=pages,
            output=scores,
        )
        msa_topk_select(
            scores, TOPK, num_valid_pages=nvp, output=sel, force_begin_blocks=1
        )
        msa_sparse_attention(
            aq,
            ak,
            av,
            sel.permute(1, 0, 2).contiguous(),
            acu,
            causal=True,
            page_table=apt,
            seqused_k=ask,
            out=aout,
        )

    for _ in range(2):  # warm the compile caches outside capture
        pipeline()
    torch.cuda.synchronize()
    eager = (scores.clone(), sel.clone(), aout.clone())
    scores.zero_()
    sel.fill_(-7)
    aout.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline()
    scores.zero_()
    sel.fill_(-7)
    aout.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(scores, eager[0])
    assert torch.equal(sel, eager[1])
    assert torch.equal(aout, eager[2])


@sm90_only
def test_program_first_needed_under_capture_raises_instead_of_building(monkeypatch):
    """A program no eager call has loaded is not JIT-built inside a CUDA graph
    capture: the launch raises, naming the program and the remedies, and the
    loader is not called.  An eager call of the geometry then loads it and the
    same capture replays bitwise."""
    from flashinfer.jit import cake_hopper_msa as jm
    from flashinfer.msa_ops import cake_hopper_sm90 as ch

    nsm = flashinfer.utils.get_device_sm_count(torch.device("cuda"))
    plan = ch.plan_topk_select(
        num_heads=1, tiles=8, total_q=256, num_sms=nsm, masked=True, nvp=True
    )
    name = jm.ROUTES.get(plan.route)
    if name is None:
        pytest.skip(f"no Cake top-k program for {plan.route} on this tree")
    scores = torch.randn(1, 8, 256, dtype=torch.float32, device="cuda")
    nvp = torch.randint(1, 9, (256,), dtype=torch.int32, device="cuda")
    sel = torch.empty((256, 1, TOPK), dtype=torch.int32, device="cuda")

    def topk():
        msa_topk_select(
            scores, TOPK, num_valid_pages=nvp, output=sel, force_begin_blocks=1
        )

    # the geometry's program leaves the process registry; the templates that cached it are reset
    monkeypatch.delitem(ch._loaded_modules, name, raising=False)
    ch._program.cache_clear()
    ch._topk_template.cache_clear()
    misses = jm.load_hopper_msa_module.cache_info().misses
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="is not loaded") as info,
        torch.cuda.graph(graph),
    ):
        topk()
    assert name in str(info.value)
    assert "preload_programs" in str(info.value)
    assert jm.load_hopper_msa_module.cache_info().misses == misses
    topk()  # the eager call loads it (outside capture)
    torch.cuda.synchronize()
    eager = sel.clone()
    sel.fill_(-7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk()
    sel.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(sel, eager)


@sm90_only
def test_preload_loads_every_program_of_a_kind_outside_capture():
    """The explicit opt-in loads every delivered program of the named kinds,
    reports what it loaded, and refuses unknown kinds and capture."""
    from flashinfer.msa_ops import cake_hopper_sm90 as ch
    from flashinfer.msa_ops._sm90_dispatch import preload_sm90_programs

    names = ch._programs_of_kind("proxy_prefill")
    assert names
    loaded = preload_sm90_programs(("proxy_prefill",))
    assert set(loaded) == {"proxy_prefill"}
    assert 0 <= loaded["proxy_prefill"] <= len(names)
    assert all(name in ch._loaded_modules for name in names)
    assert preload_sm90_programs(("proxy_prefill",)) == {"proxy_prefill": 0}
    with pytest.raises(ValueError, match="unknown"):
        preload_sm90_programs(("no_such_kind",))
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="must not run under CUDA graph capture"),
        torch.cuda.graph(graph),
    ):
        preload_sm90_programs(("proxy_prefill",))


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
def test_fp8_proxy_decode_rejects_multi_head_index_cache():
    """The fp8 decode schedule re-views the index cache with a one-head page
    stride and selects no KV head; several index heads must raise, not score
    the wrong pages."""
    torch.manual_seed(0)
    batch, hq, hkv, pages = 2, 4, 2, 8
    npages = pages * batch
    q = (torch.randn(batch, hq, HEAD_DIM, device="cuda") / 3).to(torch.bfloat16)
    k = (torch.randn(npages, hkv, BLK_KV, HEAD_DIM, device="cuda") / 3).to(
        torch.float8_e4m3fn
    )
    page_table = torch.arange(npages, dtype=torch.int32, device="cuda").view(
        batch, pages
    )
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    seqused_k = torch.full(
        (batch,), pages * BLK_KV - 13, dtype=torch.int32, device="cuda"
    )
    with pytest.raises(NotImplementedError, match="one-head index cache"):
        msa_proxy_score(
            q,
            k,
            cu_q,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_k_tiles=pages,
        )


@sm90_only
@pytest.mark.parametrize("use_fp32_acc", [True, False])
def test_fp8_proxy_prefill_rejects_multi_head_index_cache(use_fp32_acc):
    """Both accumulation precisions of the prefill regime serve a one-head fp8
    index cache; the f16 kernel also assumes a one-head page stride."""
    torch.manual_seed(0)
    batch, chunk, pages, hq, hkv = 2, 128, 8, 4, 2
    npages = pages * batch
    total_q = chunk * batch
    q = (torch.randn(total_q, hq, HEAD_DIM, device="cuda") / 3).to(torch.float8_e4m3fn)
    k = (torch.randn(npages, hkv, BLK_KV, HEAD_DIM, device="cuda") / 3).to(
        torch.float8_e4m3fn
    )
    page_table = torch.arange(npages, dtype=torch.int32, device="cuda").view(
        batch, pages
    )
    cu_q = torch.arange(0, total_q + 1, chunk, dtype=torch.int32, device="cuda")
    seqused_k = torch.full(
        (batch,), pages * BLK_KV - 13, dtype=torch.int32, device="cuda"
    )
    with pytest.raises(NotImplementedError, match="one-head index cache"):
        msa_proxy_score(
            q,
            k,
            cu_q,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_q=chunk,
            max_k_tiles=pages,
            use_fp32_acc=use_fp32_acc,
        )


@sm90_only
def test_sparse_decode_rejects_mismatched_paged_metadata():
    """``seqused_k`` / ``page_table`` sized for another batch must raise: the
    decode program derives every query position from them."""
    q, k, v, idx, page_table, seqused_k = _decode_inputs(batch=2)
    with pytest.raises(ValueError, match="seqused_k must have batch_size"):
        msa_sparse_decode_attention(
            q, k, v, idx, page_table=page_table, seqused_k=seqused_k[:1], seqlen_q=1
        )
    with pytest.raises(ValueError, match="page_table batch dimension"):
        msa_sparse_decode_attention(
            q, k, v, idx, page_table=page_table[:1], seqused_k=seqused_k, seqlen_q=1
        )
    with pytest.raises(ValueError, match="1D int32"):
        msa_sparse_decode_attention(
            q,
            k,
            v,
            idx,
            page_table=page_table,
            seqused_k=seqused_k.to(torch.int64),
            seqlen_q=1,
        )


@sm90_only
def test_sparse_prefill_rejects_out_on_another_device():
    """``out`` reaches the kernel as a raw pointer on q's device."""
    q, k, v, idx, cu_q, page_table, seqused_k = _prefill_inputs(
        8, 1, 16, (32, 32), seed=9
    )
    out = torch.empty(q.shape, dtype=q.dtype)  # host memory
    with pytest.raises(ValueError, match="on q's device"):
        msa_sparse_attention(
            q,
            k,
            v,
            idx,
            cu_q,
            causal=True,
            page_table=page_table,
            seqused_k=seqused_k,
            out=out,
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
def test_sparse_attention_rejects_causal_false():
    """Every SM90 sparse-prefill schedule is causal; the flag used to be ignored."""
    q, k, v, idx, cu_q, page_table, seqused_k = _prefill_inputs(
        8, 1, 16, (32, 32), seed=5
    )
    with pytest.raises(NotImplementedError, match="causal"):
        msa_sparse_attention(
            q, k, v, idx, cu_q, page_table=page_table, seqused_k=seqused_k
        )


@sm90_only
def test_sm90_is_advertised_as_supporting_packed_kv():
    """vLLM selects the FlashInfer MSA path off this flag; losing it silently serves Triton."""
    assert flashinfer.msa_ops.SUPPORTS_PACKED_KV is True
