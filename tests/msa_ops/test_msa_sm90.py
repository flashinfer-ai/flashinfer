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


def test_sm90_predicate_is_false_for_a_host_tensor():
    """A CPU device is "not SM90", not an error: the public MSA surfaces consult
    this predicate before their own raise rules (``out=``, dtype), which the host
    tests of ``test_msa_nvfp4_decode_sm100`` reach without a GPU."""
    from flashinfer.msa_ops._sm90_dispatch import is_sm90a_device

    assert is_sm90a_device(torch.device("cpu")) is False


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
        # the token's extent is clamp(num_valid_pages[t], 0, tiles) and both forced regions are fitted inside it
        valid = tiles if nvp is None else max(0, min(int(nvp[t]), tiles))
        forced = sorted(
            set(range(min(fb, valid))) | set(range(max(valid - fe, 0), valid))
        )
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
@pytest.mark.parametrize(
    "use_fp32_acc,batch,hq,chunk,ctx",
    [
        pytest.param(True, 2, 4, 512, 2048, id="fp32"),
        pytest.param(False, 2, 4, 512, 2048, id="f16"),
        # one index head, 1536-token chunk, 64K context: the fold-free plain-grid
        # form of the 128-row f16-accumulation kernel (the CuTe DSL kernel itself
        # launches the plain grid there and the plain layout adds no SM-tile step)
        pytest.param(False, 1, 1, 1536, 65536, id="f16-fold-free-h1-chunk1536-ctx64K"),
    ],
)
def test_proxy_score_prefill_matches_fp32_reference(
    use_fp32_acc, batch, hq, chunk, ctx
):
    """Chunked-prefill scoring against the FP32 oracle in both accumulation
    precisions: f32 (default) and f16 (the CuTe DSL kernel's numerics), including
    the fold-free plain-grid f16 form at its one-head 1536-token 64K-context
    coordinates; -inf blocks must match exactly."""
    q, k, cu_q, page_table, seqused_k, pages = _proxy_prefill_inputs(
        batch=batch, chunk=chunk, ctx=ctx, hq=hq
    )
    out = msa_proxy_score(
        q,
        k,
        cu_q,
        page_table=page_table,
        seqused_k=seqused_k,
        max_seqlen_q=chunk,
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
@pytest.mark.parametrize("tiles,total_q", [(64, 256), (512, 256), (2048, 32)])
def test_topk_select_per_token_extent_below_forced_region(tiles, total_q):
    """Tokens shorter than the forced regions select exactly their own extent:
    the forced prefix and suffix are fitted inside clamp(num_valid_pages[t], 0,
    max_k_tiles), so no selected index exceeds the token's extent and -1 is only
    a tail pad (the docstring's contract; the CuTe SM90 kernels and the Cake
    programs agree on it)."""
    hq = 1
    g = torch.Generator(device="cuda").manual_seed(tiles + total_q)
    scores = (
        torch.randperm(hq * tiles * total_q, generator=g, device="cuda")
        .to(torch.float32)
        .reshape(hq, tiles, total_q)
    )
    for fb, fe in ((2, 2), (3, 0), (0, 2)):
        short = torch.randint(
            0, fb + fe + 2, (total_q,), generator=g, device="cuda", dtype=torch.int32
        )
        full = torch.randint(
            17, tiles + 1, (total_q,), generator=g, device="cuda", dtype=torch.int32
        )
        nvp = torch.where(
            torch.rand(total_q, generator=g, device="cuda") < 0.5, short, full
        )
        nvp[0] = 0  # an empty extent yields an all -1 row
        out = msa_topk_select(
            scores,
            TOPK,
            num_valid_pages=nvp,
            force_begin_blocks=fb,
            force_end_blocks=fe,
        )
        assert torch.equal(out, _topk_reference(scores, nvp, fb, fe))
        rows, extents = out.cpu(), nvp.cpu()
        for t in range(total_q):
            n = int(extents[t])
            for h in range(hq):
                row = rows[t, h]
                valid = row[row >= 0]
                assert len(valid) == min(TOPK, n), (fb, fe, t, h, row.tolist())
                assert bool((valid < n).all()), (fb, fe, t, h, row.tolist())
                assert bool((valid[1:] > valid[:-1]).all()), (
                    fb,
                    fe,
                    t,
                    h,
                    row.tolist(),
                )
                assert bool((row[len(valid) :] == -1).all()), (
                    fb,
                    fe,
                    t,
                    h,
                    row.tolist(),
                )


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
    q = (torch.randn(batch, hq, HEAD_DIM, device="cuda") / 3).to(torch.float8_e4m3fn)
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
def test_topk_select_scalar_num_valid_pages():
    """A batch-wide scalar count is served at the full score width only (no
    clamping, identical to the default); neither SM90 top-k schedule reads a
    scalar, so a smaller one raises instead of being dropped."""
    torch.manual_seed(2)
    hq, tiles, total_q = 2, 48, 37
    scores = torch.randn(hq, tiles, total_q, dtype=torch.float32, device="cuda")
    full = msa_topk_select(scores, TOPK, num_valid_pages=tiles)
    assert torch.equal(full, msa_topk_select(scores, TOPK))
    with pytest.raises(NotImplementedError, match="per-token"):
        msa_topk_select(scores, TOPK, num_valid_pages=tiles - 8)


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
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (2, 1)])
def test_fp16_q_is_rejected_by_sm90_sparse_attention(num_qo_heads, num_kv_heads):
    """Every SM90 sparse-prefill schedule is bf16-only: the programs reject
    fp16 q and the CuTe DSL kernels (GQA 2 here) would read it as bf16 bytes,
    so both classes must raise at the dispatch."""
    q, k, v, idx, cu_q, page_table, seqused_k = _prefill_inputs(
        num_qo_heads, num_kv_heads, 16, (32, 32), seed=10
    )
    with pytest.raises(NotImplementedError, match="bf16 q"):
        msa_sparse_attention(
            q.to(torch.float16),
            k,
            v,
            idx,
            cu_q,
            causal=True,
            page_table=page_table,
            seqused_k=seqused_k,
        )


@sm90_only
def test_fp16_q_is_rejected_by_sm90_sparse_decode():
    """The SM90 decode program consumes bf16 q and has no fallback schedule."""
    q, k, v, idx, page_table, seqused_k = _decode_inputs(seed=11)
    with pytest.raises(NotImplementedError, match="bf16 q"):
        msa_sparse_decode_attention(
            q.to(torch.float16),
            k,
            v,
            idx,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=1,
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


# PR regressions: also exercise the retained CuTe implementations.


@pytest.fixture(params=["cake", "cute"])
def sm90_regression_backend(request, monkeypatch):
    """Exercise the public Cake route and the retained CuTe fallback separately."""
    from flashinfer.msa_ops import _sm90_dispatch, cake_hopper_sm90

    if request.param == "cute":
        for route in (
            "prefill_route_available",
            "topk_route_available",
            "proxy_prefill_route_available",
        ):
            monkeypatch.setattr(cake_hopper_sm90, route, lambda **kwargs: False)
    else:
        # sparse prefill now prefers the CuTe schedules, so the Cake arm has to
        # turn that preference off or this fixture would run CuTe on both arms.
        for pref in ("_cute_prefill_preferred", "_cute_proxy_decode_preferred"):
            monkeypatch.setattr(_sm90_dispatch, pref, lambda *a, **k: False)
    return request.param


def _ref_attn_rows(q, kv, idx, page_table, seqused_k, scale, rows, qpos_of):
    """Dense fp32 reference for selected query rows, vectorised per KV head.

    ``idx`` may repeat a block (the kernels walk the *union* of a tile's selected
    blocks), so the block list is deduplicated before the softmax -- a reference
    that counted a repeat twice would disagree with a correct kernel.
    ``qpos_of(row)`` returns the row's absolute key position, which is
    right-aligned for prefill: seqused_k - qlen + offset_in_sequence.
    """
    total_q, hq, d = q.shape
    hkv = kv.shape[1]
    g = hq // hkv
    out = torch.zeros(len(rows), hq, d, dtype=torch.float32, device=q.device)
    for i, row in enumerate(rows):
        b, qpos = qpos_of(row)
        limit = min(qpos, int(seqused_k[b]) - 1)
        for kh in range(hkv):
            blocks = torch.unique(idx[kh, row])
            blocks = blocks[(blocks >= 0) & (blocks < page_table.shape[1])]
            if blocks.numel() == 0:
                continue
            pages = page_table[b, blocks.long()].long()
            keys = kv[pages, kh, :, :d].reshape(-1, d).float()
            vals = kv[pages, kh, :, d:].reshape(-1, d).float()
            pos = (
                blocks.long().unsqueeze(1) * BLK_KV
                + torch.arange(BLK_KV, device=q.device).unsqueeze(0)
            ).reshape(-1)
            keep = pos <= limit
            if not bool(keep.any()):
                continue
            heads = list(range(kh * g, (kh + 1) * g))
            logits = (q[row, heads].float() @ keys.T) * scale
            logits = logits.masked_fill(~keep.unsqueeze(0), float("-inf"))
            out[i, heads] = torch.softmax(logits, dim=-1) @ vals
    return out


@sm90_only
@pytest.mark.parametrize("max_k_tiles", [16, 1024])
def test_sparse_prefill_runs_both_backends(sm90_regression_backend, max_k_tiles):
    """Exercise the union backend (mpg <= 800) and the single-token one above it."""
    torch.manual_seed(max_k_tiles)
    batch, qlen, hq, hkv = 1, 256, 8, 1
    npages = max_k_tiles * batch
    base = (torch.randn(npages, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    q = (torch.randn(qlen, hq, HEAD_DIM, device="cuda") / 4).to(torch.bfloat16)
    idx = (
        torch.rand(hkv, qlen, max_k_tiles, device="cuda")
        .argsort(dim=-1)[..., :TOPK]
        .sort(dim=-1)
        .values.to(torch.int32)
        .contiguous()
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
    kv = torch.cat([k, v], dim=-1)
    prefix = int(seqused_k[0]) - qlen
    rows = [0, 1, qlen // 2, qlen - 2, qlen - 1]
    ref = _ref_attn_rows(
        q,
        kv,
        idx,
        page_table,
        seqused_k,
        1.0 / math.sqrt(HEAD_DIM),
        rows,
        lambda r: (0, prefix + r),
    )
    torch.testing.assert_close(out[rows].float(), ref, atol=3e-2, rtol=3e-2)


@sm90_only
@pytest.mark.parametrize(
    "fill", ["nan", "nan_k_only", "nan_v_only", "garbage", "zeros"]
)
def test_prefill_ignores_page_bytes_past_seqused_k(sm90_regression_backend, fill):
    """Page memory beyond seqused_k must not reach the result.

    vLLM hands out pages whose tail is uninitialized. The kernel reads whole
    128-token pages, so without a causal bound on the selected blocks -- and with
    an additive mask, which cannot clear a NaN -- that tail leaks into the
    softmax. NaN there poisons the whole row.
    """
    torch.manual_seed(0)
    batch, qlen, hq, hkv, mkt = 1, 128, 8, 1, 16
    npages = mkt * batch
    kv = torch.randn(npages, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4
    # sequence ends mid-page; everything after it is "uninitialized"
    seqlen = mkt * BLK_KV - 50
    last_pg, off = seqlen // BLK_KV, seqlen % BLK_KV
    if fill == "nan":
        kv[last_pg, :, off:, :] = float("nan")
    elif fill == "nan_k_only":  # K tail NaN, V tail finite
        kv[last_pg, :, off:, :HEAD_DIM] = float("nan")
    elif fill == "nan_v_only":  # V tail NaN, K tail finite
        kv[last_pg, :, off:, HEAD_DIM:] = float("nan")
    elif fill == "garbage":
        kv[last_pg, :, off:, :] = 1e4
    else:
        kv[last_pg, :, off:, :] = 0.0
    kv = kv.to(torch.float8_e4m3fn)
    k, v = kv[..., :HEAD_DIM], kv[..., HEAD_DIM:]
    q = (torch.randn(qlen, hq, HEAD_DIM, device="cuda") / 4).to(torch.bfloat16)
    idx = (
        torch.arange(mkt, dtype=torch.int32, device="cuda")
        .view(1, 1, TOPK)
        .expand(hkv, qlen, TOPK)
        .contiguous()
    )
    cu_q = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    page_table = torch.arange(npages, dtype=torch.int32, device="cuda").view(batch, mkt)
    seqused_k = torch.full((batch,), seqlen, dtype=torch.int32, device="cuda")
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
    assert torch.isfinite(out.float()).all(), f"{fill} tail leaked into the output"


@sm90_only
@pytest.mark.parametrize(
    "pattern", ["all_equal", "few_distinct", "tied_top16", "ties_with_pad"]
)
def test_topk_select_returns_distinct_blocks_under_ties(
    sm90_regression_backend, pattern
):
    """Exact score ties must not yield duplicate block indices.

    The final scatter folds the lane into its sort key so it is tie-safe, but the
    cascade merge keeps (score, index) pairs with a plain >= on the score; a tile
    reachable from two partial lists can therefore be emitted twice, leaving
    fewer than topk distinct blocks.
    """
    torch.manual_seed(0)
    num_qo_heads, max_k_tiles, total_q = 4, 64, 32
    if pattern == "all_equal":
        scores = torch.full(
            (num_qo_heads, max_k_tiles, total_q),
            1.0,
            dtype=torch.float32,
            device="cuda",
        )
    elif pattern == "few_distinct":  # 4 values over 64 tiles
        base = torch.tensor([3.0, 2.0, 1.0, 0.0], device="cuda")
        scores = (
            base.repeat(max_k_tiles // 4)[None, :, None]
            .expand(num_qo_heads, max_k_tiles, total_q)
            .contiguous()
        )
    elif pattern == "tied_top16":  # 16 tied winners, rest lower
        scores = torch.zeros(
            num_qo_heads, max_k_tiles, total_q, dtype=torch.float32, device="cuda"
        )
        scores[:, :16, :] = 5.0
    else:  # ties plus padding
        scores = torch.full(
            (num_qo_heads, max_k_tiles, total_q),
            2.0,
            dtype=torch.float32,
            device="cuda",
        )
        scores[:, 8:, :] = float("-inf")  # only 8 valid tiles
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
def test_topk_select_20way_tie_reviewer_repro(sm90_regression_backend, hq, tiles, nq):
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
@pytest.mark.parametrize("hq,hkv", [(8, 1), (16, 1)])
def test_prefill_single_key_window_returns_v_exactly(sm90_regression_backend, hq, hkv):
    """Token 0 attends exactly one key, so the output must be V[0] bit-exactly.

    softmax over a one-element window is 1.0, which leaves no room for rounding:
    any permutation of P, any double-counting, and any mismatch between the fp8
    operand the PV gemm consumes and the f32 sum used to normalise it all show up
    here as a scale or a wrong row. Longer windows average those away.
    """
    dev = "cuda"
    torch.manual_seed(0)
    kv = (torch.randn(1, hkv, BLK_KV, 2 * HEAD_DIM, device=dev) / 4).to(
        torch.float8_e4m3fn
    )
    k, v = kv[..., :HEAD_DIM], kv[..., HEAD_DIM:]
    n = 32
    q = (torch.randn(n, hq, HEAD_DIM, device=dev) * 0.25).to(torch.bfloat16)
    idx = torch.full((hkv, n, TOPK), -1, dtype=torch.int32, device=dev)
    idx[:, :, 0] = 0
    out = msa_sparse_attention(
        q,
        k,
        v,
        idx,
        torch.tensor([0, n], dtype=torch.int32, device=dev),
        page_table=torch.zeros(1, 1, dtype=torch.int32, device=dev),
        seqused_k=torch.full((1,), n, dtype=torch.int32, device=dev),
        causal=True,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
    ).float()
    v0 = kv[0, 0, 0, HEAD_DIM:].float()
    for h in range(hq):
        torch.testing.assert_close(out[0, h], v0, atol=0.0, rtol=0.0)


@sm90_only
@pytest.mark.parametrize("lens", [[128, 128], [128, 128, 128]])
def test_sparse_prefill_uniform_batch_matches_reference(sm90_regression_backend, lens):
    """Equal-length sequences in one batch, against a dense reference.

    seqused_k equals the query length, so the first rows of every sequence see
    one- and two-key windows -- the regime where a mis-masked or mis-weighted P
    is visible before longer windows average it away.
    """
    torch.manual_seed(11)
    mkt, hq, hkv = 8, 8, 1
    total_q, batch = sum(lens), len(lens)
    npages = mkt * batch
    base = (torch.randn(npages, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    q = (torch.randn(total_q, hq, HEAD_DIM, device="cuda") / 4).to(torch.bfloat16)
    idx = torch.full((hkv, total_q, TOPK), -1, dtype=torch.int32, device="cuda")
    idx[..., :mkt] = torch.arange(mkt, dtype=torch.int32, device="cuda")
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    cu_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    page_table = torch.randperm(npages, device="cuda").to(torch.int32).view(batch, mkt)
    seqused_k = torch.tensor(lens, dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(HEAD_DIM)
    out = msa_sparse_attention(
        q,
        k,
        v,
        idx,
        cu_q,
        page_table=page_table,
        seqused_k=seqused_k,
        causal=True,
        softmax_scale=scale,
    )
    assert torch.isfinite(out.float()).all()
    rows = [cu[b] + off for b in range(batch) for off in (0, 1, lens[b] - 1)]

    def qpos_of(row):
        b = max(i for i in range(batch) if cu[i] <= row)
        return b, row - cu[b]

    kv = torch.cat([k, v], dim=-1)
    ref = _ref_attn_rows(q, kv, idx, page_table, seqused_k, scale, rows, qpos_of)
    torch.testing.assert_close(out[rows].float(), ref, atol=3e-2, rtol=3e-2)


@sm90_only
@pytest.mark.parametrize("lens", [[96, 160], [64, 192, 96], [17, 256, 33, 128]])
def test_sparse_prefill_ragged_cu_seqlens_matches_reference(
    sm90_regression_backend, lens
):
    """Unequal query lengths in one batch, against a dense reference.

    The union schedule groups nt consecutive tokens of ONE sequence per CTA, so
    it has to recover the sequence from the CTA index. Dividing by a uniform
    groups-per-sequence put the first token of sequence 1 on sequence 0's page
    table at the global row index; walking cu_seqlens_q is what makes unequal
    lengths land on the right sequence and the right causal position.
    """
    torch.manual_seed(11)
    mkt, hq, hkv = 8, 8, 1
    total_q, batch = sum(lens), len(lens)
    npages = mkt * batch
    base = (torch.randn(npages, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    q = (torch.randn(total_q, hq, HEAD_DIM, device="cuda") / 4).to(torch.bfloat16)
    idx = torch.full((hkv, total_q, TOPK), -1, dtype=torch.int32, device="cuda")
    idx[..., :mkt] = torch.arange(mkt, dtype=torch.int32, device="cuda")
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    cu_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    page_table = torch.randperm(npages, device="cuda").to(torch.int32).view(batch, mkt)
    seqused_k = torch.tensor(lens, dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(HEAD_DIM)
    out = msa_sparse_attention(
        q,
        k,
        v,
        idx,
        cu_q,
        page_table=page_table,
        seqused_k=seqused_k,
        causal=True,
        softmax_scale=scale,
    )
    assert torch.isfinite(out.float()).all()
    # boundary rows of every sequence: first token of a sequence is where a
    # misattributed CTA shows up before longer windows average it away
    rows = [cu[b] + off for b in range(batch) for off in (0, 1, lens[b] - 1)]

    def qpos_of(row):
        b = max(i for i in range(batch) if cu[i] <= row)
        return b, row - cu[b]

    kv = torch.cat([k, v], dim=-1)
    ref = _ref_attn_rows(q, kv, idx, page_table, seqused_k, scale, rows, qpos_of)
    torch.testing.assert_close(out[rows].float(), ref, atol=3e-2, rtol=3e-2)


@sm90_only
def test_packed_kv_rejects_unsupported_paged_layouts():
    """The packed view is rebuilt with raw strides, so a wrong shape reads garbage.

    Both routes require interleaved K/V. The CuTe _views() helper additionally
    reconstructs dense strides, while Cake reads strides from each tensor and
    can serve a strided pool view.
    """
    from flashinfer.msa_ops._sm90_dispatch import _as_packed_kv

    base = (torch.randn(4, 1, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    # the supported layout is accepted
    _as_packed_kv(k, v, HEAD_DIM)

    with pytest.raises(NotImplementedError, match="4D paged KV cache"):
        _as_packed_kv(k[0], v[0], HEAD_DIM)

    short = (torch.randn(4, 1, 64, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    with pytest.raises(NotImplementedError, match="page size"):
        _as_packed_kv(short[..., :HEAD_DIM], short[..., HEAD_DIM:], HEAD_DIM)

    with pytest.raises(NotImplementedError, match="head_dim"):
        _as_packed_kv(k, v, HEAD_DIM // 2)

    # separate allocations: tokens are head_dim apart, not 2*head_dim
    with pytest.raises(NotImplementedError):
        _as_packed_kv(k.contiguous(), v.contiguous(), HEAD_DIM)

    # Cake admits a pool view with a larger page stride. CuTe computes that
    # stride from the shape, so its stricter layout check must refuse it.
    pool = (torch.randn(8, 1, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    sparse = pool[::2]
    assert sparse.shape == base.shape and sparse.stride(0) == 2 * base.stride(0)
    restored = _as_packed_kv(sparse[..., :HEAD_DIM], sparse[..., HEAD_DIM:], HEAD_DIM)
    assert restored.data_ptr() == sparse.data_ptr()
    assert restored.stride() == sparse.stride()
    with pytest.raises(NotImplementedError, match="dense packed layout"):
        _as_packed_kv(
            sparse[..., :HEAD_DIM], sparse[..., HEAD_DIM:], HEAD_DIM, require_dense=True
        )


@sm90_only
def test_decode_scratch_is_not_shared_across_streams():
    """Split-K partials are live for a whole launch.

    Two streams running decode concurrently would otherwise write each other's
    partial outputs and combine counters, so the stream has to be part of the
    cache key rather than only the device and tile geometry.
    """
    from flashinfer.msa_ops.cute_dsl import sparse_decode_sm90 as sd

    saved = dict(sd._WS)
    sd._WS.clear()
    try:
        dev = torch.device("cuda:0")
        sd._scratch(4, 8, HEAD_DIM, dev, 1)
        sd._scratch(4, 8, HEAD_DIM, dev, 2)
        streams = {key[1] for key in sd._WS}
        assert streams == {1, 2}, f"scratch key ignores the stream: {list(sd._WS)}"
    finally:
        sd._WS.clear()
        sd._WS.update(saved)


@sm90_only
def test_proxy_score_fp32_acc_survives_large_logits(sm90_regression_backend):
    """The fp16 score accumulator saturates on unscaled fp8 dot products.

    max_score holds raw Q K^T logits. Two fp8 e4m3 operands give a single
    product of up to 448*448 = 200704, and a 128-term row runs far past the
    65504 fp16 ceiling with nothing clamping the partial sums, so the row
    becomes ``inf`` and outranks every real block. Ranking is what the output
    is for, so the failure is silent: top-k picks among infinities.
    """
    torch.manual_seed(0)
    total_q, hq, hkv, pages = 128, 4, 1, 32
    mag = 64.0
    q = (torch.randn(total_q, hq, HEAD_DIM, device="cuda") * mag).clamp(-448, 448)
    q = q.to(torch.float8_e4m3fn)
    k = (torch.randn(pages, hkv, BLK_KV, HEAD_DIM, device="cuda") * mag).clamp(
        -448, 448
    )
    k = k.to(torch.float8_e4m3fn)
    page_table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, pages)
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device="cuda")
    seqused_k = torch.full((1,), pages * BLK_KV, dtype=torch.int32, device="cuda")

    def score(use_fp32_acc):
        return msa_proxy_score(
            q,
            k,
            cu_q,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_q=total_q,
            max_k_tiles=pages,
            use_fp32_acc=use_fp32_acc,
        ).float()

    narrow = score(False)
    wide = score(True)
    # the fp16 accumulator loses most of the grid; fp32 keeps all of it. Only
    # causally dead blocks may be -inf, and those are identical either way.
    assert torch.isfinite(wide).sum() > 4 * torch.isfinite(narrow).sum()
    dead = torch.isneginf(wide)
    assert bool((torch.isfinite(wide) | dead).all())


@sm90_only
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"causal": False}, "causal=False is not supported"),
        ({"q_offset": 3}, "q_offset is not supported"),
    ],
)
def test_sparse_decode_rejects_dropped_arguments(kwargs, match):
    """causal=False and q_offset reach no SM90 kernel argument, so they must raise.

    The decode schedule masks against ``seqused_k[b] - seqlen_q + i`` and takes
    neither flag, so honouring the signature while ignoring the values would
    return right-aligned causal numbers for a request that asked for something
    else. Other backends do apply both, which is what makes silence here a
    portability trap rather than a documented narrowing.
    """
    q, k, v, idx, page_table, seqused_k = _decode_inputs()
    with pytest.raises(NotImplementedError, match=match):
        msa_sparse_decode_attention(
            q,
            k,
            v,
            idx,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=1,
            softmax_scale=1.0 / math.sqrt(HEAD_DIM),
            **kwargs,
        )


@sm90_only
def test_single_token_prefill_rejects_unsupported_gqa(monkeypatch):
    """g < 8 leaves the single-token copy-out with zero iterations.

    The loop is sized g * (HEAD_DIM // 8) // 128, so Hq/Hkv of 4 moves nothing
    and the caller would get its uninitialised output buffer back. Cake supports
    this GQA group, so force the CuTe fallback to check its narrower contract.
    """
    from flashinfer.msa_ops import cake_hopper_sm90

    monkeypatch.setattr(
        cake_hopper_sm90, "prefill_route_available", lambda **kwargs: False
    )
    torch.manual_seed(0)
    qlen, hq, hkv, mkt = 8, 4, 1, 1024
    base = (torch.randn(mkt, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    q = (torch.randn(qlen, hq, HEAD_DIM, device="cuda") / 4).to(torch.bfloat16)
    idx = torch.full((hkv, qlen, TOPK), -1, dtype=torch.int32, device="cuda")
    idx[..., 0] = 0
    cu_q = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    page_table = torch.arange(mkt, dtype=torch.int32, device="cuda").view(1, mkt)
    seqused_k = torch.full((1,), BLK_KV, dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError, match="Hq/Hkv of 8 or 16"):
        msa_sparse_attention(
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


@sm90_only
@pytest.mark.parametrize("backend", ["cake", "cute"])
def test_sparse_decode_ignores_poisoned_page_tail(backend):
    """Bytes past seqused_k are uninitialised in vLLM and must not reach the output.

    The causal/length mask only sanitises *scores*; V is a matmul operand, and
    0 * NaN = NaN in the PV gemm, so V itself has to be clamped. Decode feeds V
    from smem through a register conversion, which is where the clamp lives.
    """
    q, k, v, idx, page_table, seqused_k = _decode_inputs(
        batch=2, num_qo_heads=8, num_kv_heads=1, max_k_tiles=16, seed=11
    )
    # All sixteen blocks are distinct and selected, including the boundary block.
    scale = 1.0 / math.sqrt(HEAD_DIM)
    kv = torch.cat([k, v], dim=-1)  # copy: the reference keeps the clean bytes
    ref = _ref_attn_rows(
        q,
        kv,
        idx,
        page_table,
        seqused_k,
        scale,
        list(range(q.shape[0])),
        lambda r: (r, int(seqused_k[r]) - 1),
    )

    tail = int(seqused_k[0]) % BLK_KV
    nan8 = torch.tensor(float("nan"), device=q.device).to(torch.float8_e4m3fn)
    for b in range(page_table.shape[0]):
        pg = int(page_table[b, -1])
        k[pg, :, tail:, :] = nan8
        v[pg, :, tail:, :] = nan8

    if backend == "cute":
        from flashinfer.msa_ops._sm90_dispatch import _as_packed_kv
        from flashinfer.msa_ops.cute_dsl.sparse_decode_sm90 import run

        out = torch.empty_like(q)
        run(
            q,
            _as_packed_kv(k, v, HEAD_DIM, require_dense=True),
            idx,
            page_table,
            seqused_k,
            out,
        )
    else:
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
    assert torch.isfinite(out.float()).all(), "a NaN past seqused_k reached the output"
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@sm90_only
@pytest.mark.parametrize("hq,hkv", [(64, 4), (32, 4)])
def test_sparse_prefill_multi_kv_head_matches_reference(
    sm90_regression_backend, hq, hkv
):
    """More than one KV head: each head must resolve its own block list.

    Exercise the union-path kv-head stride with different selected blocks
    for each KV head, as well as the public Cake route.
    """
    torch.manual_seed(hq * 100 + hkv)
    qlen, mkt = 128, 32
    base = (torch.randn(mkt, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    q = (torch.randn(qlen, hq, HEAD_DIM, device="cuda") / 4).to(torch.bfloat16)
    idx = (
        torch.rand(hkv, qlen, mkt, device="cuda")
        .argsort(dim=-1)[..., :TOPK]
        .sort(dim=-1)
        .values.to(torch.int32)
        .contiguous()
    )
    cu_q = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    page_table = torch.randperm(mkt, device="cuda").to(torch.int32).view(1, mkt)
    seqused_k = torch.full((1,), mkt * BLK_KV - 13, dtype=torch.int32, device="cuda")
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
    assert torch.isfinite(out.float()).all()
    prefix = int(seqused_k[0]) - qlen
    rows = [0, 1, qlen // 2, qlen - 2, qlen - 1]
    kv = torch.cat([k, v], dim=-1)
    ref = _ref_attn_rows(
        q,
        kv,
        idx,
        page_table,
        seqused_k,
        1.0 / math.sqrt(HEAD_DIM),
        rows,
        lambda r: (0, prefix + r),
    )
    torch.testing.assert_close(out[rows].float(), ref, atol=3e-2, rtol=3e-2)


@sm90_only
@pytest.mark.parametrize("mult", [0.5, 1.0, 2.0])
def test_sparse_prefill_honours_non_default_softmax_scale(
    sm90_regression_backend, mult
):
    """An fp8 KV cache folds k_scale into softmax_scale, so it is not 1/sqrt(D).

    The CuTe schedules bake the scale in at compile time rather than pre-scaling
    q, so a non-default value has to reach the kernel exactly on both backends.
    """
    torch.manual_seed(int(mult * 10))
    qlen, hq, hkv, mkt = 128, 8, 1, 16
    base = (torch.randn(mkt, hkv, BLK_KV, 2 * HEAD_DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    k, v = base[..., :HEAD_DIM], base[..., HEAD_DIM:]
    q = (torch.randn(qlen, hq, HEAD_DIM, device="cuda") / 4).to(torch.bfloat16)
    idx = torch.randint(0, mkt, (hkv, qlen, TOPK), dtype=torch.int32, device="cuda")
    idx, _ = idx.sort(dim=-1)
    cu_q = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    page_table = torch.randperm(mkt, device="cuda").to(torch.int32).view(1, mkt)
    seqused_k = torch.full((1,), mkt * BLK_KV - 13, dtype=torch.int32, device="cuda")
    scale = mult / math.sqrt(HEAD_DIM)
    out = msa_sparse_attention(
        q,
        k,
        v,
        idx,
        cu_q,
        page_table=page_table,
        seqused_k=seqused_k,
        causal=True,
        softmax_scale=scale,
    )
    assert torch.isfinite(out.float()).all()
    prefix = int(seqused_k[0]) - qlen
    rows = [0, 1, qlen // 2, qlen - 1]
    kv = torch.cat([k, v], dim=-1)
    ref = _ref_attn_rows(
        q, kv, idx, page_table, seqused_k, scale, rows, lambda r: (0, prefix + r)
    )
    # Halving the scale flattens the softmax, so the outputs shrink toward the
    # mean of V and the fp8 V quantisation dominates: the Cake route lands ~1%
    # of elements just past a 3e-2 bound at mult=0.5. The point of this test is
    # that the scale reaches the kernel at all, so bound it loosely.
    torch.testing.assert_close(out[rows].float(), ref, atol=6e-2, rtol=6e-2)


@sm90_only
@pytest.mark.parametrize("mult", [1.0, 2.0])
def test_sparse_decode_cute_fallback_matches_reference(monkeypatch, mult):
    """Sparse decode must not be single-sourced on the Cake program.

    Force the CuTe DSL schedule the way an unadmitted geometry would, and check
    it still matches the dense reference -- including a non-default
    softmax_scale, which it now takes as a compile-time constant rather than by
    pre-scaling q.
    """
    from flashinfer.msa_ops import _sm90_dispatch

    monkeypatch.setattr(_sm90_dispatch, "_cake_decode_available", lambda *a, **k: False)
    q, k, v, idx, page_table, seqused_k = _decode_inputs(
        batch=2, num_qo_heads=8, num_kv_heads=1, max_k_tiles=32, seed=int(mult) + 3
    )
    scale = mult / math.sqrt(HEAD_DIM)
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
    assert torch.isfinite(out.float()).all()
    kv = torch.cat([k, v], dim=-1)
    rows = list(range(q.shape[0]))
    ref = _ref_attn_rows(
        q,
        kv,
        idx,
        page_table,
        seqused_k,
        scale,
        rows,
        lambda r: (r, int(seqused_k[r]) - 1),
    )
    torch.testing.assert_close(out[rows].float(), ref, atol=3e-2, rtol=3e-2)
