# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""QSA stage-1 token top-k block indexer (SM107) against an exact PyTorch
reference, including the deterministic lower-block-index tie-break."""

import itertools

import pytest
import torch

from flashinfer import qsa_topk_indexer
from flashinfer.qsa import (
    QSA_INDEXER_BLOCK_TOPK,
    QSA_INDEXER_BLOCK_SIZE,
    QSA_INDEXER_HEAD_DIM,
    QSA_INDEXER_NUM_HEADS,
    QSA_INDEXER_PACKED_WIDTH,
    QSA_INDEXER_PAGE_BLOCKS,
)

CR = QSA_INDEXER_BLOCK_SIZE
BLOCK_TOPK = QSA_INDEXER_BLOCK_TOPK
PWIDTH = QSA_INDEXER_PACKED_WIDTH
CPAGE = QSA_INDEXER_PAGE_BLOCKS

SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(not SM107, reason="requires SM107 (Rubin)")


def ref_qsa_topk_indexer(
    qi: torch.Tensor,
    ck_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cpage_base: torch.Tensor,
) -> torch.Tensor:
    """Exact reference: per-block relu(q.k) scores summed over the index
    heads, stable descending top-k (ties: lower block index wins), x4 token
    expansion, causal tail, ascending packed output with the count in the
    last column."""
    dev = qi.device
    T = qi.shape[0]
    cu = cu_seqlens.long()
    lens = cu[1:] - cu[:-1]
    req = torch.repeat_interleave(torch.arange(lens.numel(), device=dev), lens)
    pos = torch.arange(int(cu[-1]), device=dev) - cu[req]
    vis = (pos + 1) // CR
    vmax = max(int(vis.max().item()), 1)
    flat = ck_cache.reshape(-1, ck_cache.shape[-1]).float()
    B = cu.numel() - 1
    logits = torch.full((T, vmax), float("-inf"), device=dev)
    q = qi.float()
    for r in range(B):
        t0, t1 = int(cu[r]), int(cu[r + 1])
        if t1 == t0:
            continue
        vr = max(int(vis[t0:t1].max().item()), 1)
        base = int(cpage_base[r]) * CPAGE
        ck_r = flat[base : base + vr]
        s = torch.relu(torch.einsum("thd,vd->thv", q[t0:t1], ck_r)).sum(1)
        logits[t0:t1, :vr] = s
    cols = torch.arange(vmax, device=dev)
    logits = logits.masked_fill(cols.unsqueeze(0) >= vis.unsqueeze(1), float("-inf"))
    k = min(BLOCK_TOPK, vmax)
    order = torch.argsort(-logits, dim=1, stable=True)[:, :k]
    rank = torch.arange(k, device=dev)
    nb = torch.minimum(vis, torch.tensor(BLOCK_TOPK, device=dev))
    blocks = torch.where(
        rank.unsqueeze(0) < nb.unsqueeze(1), order, torch.full_like(order, -1)
    )
    off = torch.arange(CR, device=dev)
    toks = torch.where(
        blocks.unsqueeze(-1) >= 0,
        blocks.unsqueeze(-1) * CR + off,
        torch.full((1, 1, CR), -1, dtype=torch.long, device=dev),
    ).reshape(T, k * CR)
    tail_start = (pos + 1) // CR * CR
    tcol = torch.arange(CR - 1, device=dev)
    tail = tail_start.unsqueeze(1) + tcol
    tail = torch.where(
        tcol.unsqueeze(0) < (pos + 1 - tail_start).unsqueeze(1),
        tail,
        torch.full_like(tail, -1),
    )
    allt = torch.cat([toks, tail], 1)
    big = torch.where(allt >= 0, allt, torch.full_like(allt, 2**31 - 1))
    srt = torch.sort(big, dim=1)[0]
    counts = (allt >= 0).sum(1, dtype=torch.int32)
    out = torch.full((T, PWIDTH), -1, dtype=torch.int32, device=dev)
    w = srt.shape[1]
    out[:, :w] = torch.where(srt < 2**31 - 1, srt, torch.full_like(srt, -1)).int()
    out[:, PWIDTH - 1] = counts
    return out


def make_inputs(lengths, seed=0, device="cuda", identical_keys=False):
    """Packed requests of the given lengths over a contiguous paged
    compressed-key cache (one spare page per request)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cu = torch.tensor([0] + list(itertools.accumulate(lengths)), dtype=torch.int32)
    T = int(cu[-1])
    npages = [(length // CR + CPAGE - 1) // CPAGE + 1 for length in lengths]
    cpage_base = torch.tensor(
        [0] + list(itertools.accumulate(npages))[:-1], dtype=torch.int32
    )
    num_pages = sum(npages)
    qi = torch.randn(T, QSA_INDEXER_NUM_HEADS, QSA_INDEXER_HEAD_DIM, generator=gen).to(
        torch.float8_e4m3fn
    )
    if identical_keys:
        row = (torch.randn(1, 1, 1, QSA_INDEXER_HEAD_DIM, generator=gen) * 0.5).to(
            torch.float8_e4m3fn
        )
        ck = row.expand(num_pages, CPAGE, 1, QSA_INDEXER_HEAD_DIM).contiguous()
    else:
        ck = (
            torch.randn(num_pages, CPAGE, 1, QSA_INDEXER_HEAD_DIM, generator=gen) * 0.5
        ).to(torch.float8_e4m3fn)
    return (
        qi.to(device),
        ck.to(device),
        cu.to(device),
        cpage_base.to(device),
    )


@pytest.mark.parametrize(
    "lengths",
    [
        (24,),  # trivial rows only (every row keeps its full prefix)
        (96, 160),  # batch > 1, still within the selection budget
        (2600,),  # crosses the 2051 trivial-prefix bound: real selection
        (3000, 1057),  # batch > 1 with one long and one mid request
        (7, 2129, 33),  # uneven mix, one empty-ish short request
    ],
)
def test_correctness(lengths):
    qi, ck, cu, cpage_base = make_inputs(lengths, seed=sum(lengths))
    out = qsa_topk_indexer(qi, ck, cu, cpage_base)
    ref = ref_qsa_topk_indexer(qi, ck, cu, cpage_base)
    assert out.shape == (qi.shape[0], PWIDTH) and out.dtype == torch.int32
    assert torch.equal(out, ref)


def test_preallocated_out_and_determinism():
    qi, ck, cu, cpage_base = make_inputs((2600,), seed=7)
    out = torch.full((qi.shape[0], PWIDTH), 123, dtype=torch.int32, device=qi.device)
    ret = qsa_topk_indexer(qi, ck, cu, cpage_base, out=out)
    assert ret.data_ptr() == out.data_ptr()
    again = qsa_topk_indexer(qi, ck, cu, cpage_base)
    assert torch.equal(out, again)  # bitwise repeatable


def test_tie_break_lower_block_wins():
    """All keys identical: every visible block of a row scores the same, so
    the deterministic tie-break must select the lowest block indices."""
    lengths = (2600,)  # vis reaches 650 > BLOCK_TOPK: the budget binds
    qi, ck, cu, cpage_base = make_inputs(lengths, seed=3, identical_keys=True)
    out = qsa_topk_indexer(qi, ck, cu, cpage_base)
    ref = ref_qsa_topk_indexer(qi, ck, cu, cpage_base)
    assert torch.equal(out, ref)
    # Spot-check the pinned semantics directly on the last row: top-512 of
    # 650 equal-scored blocks must be blocks 0..511, plus the causal tail.
    pos = lengths[0] - 1
    vis = (pos + 1) // CR
    assert vis > BLOCK_TOPK
    expected_tokens = list(range(BLOCK_TOPK * CR))  # blocks 0..511 expanded
    tail_start = (pos + 1) // CR * CR
    expected_tokens += list(range(tail_start, pos + 1))
    expected_tokens.sort()
    row = out[-1]
    count = int(row[PWIDTH - 1])
    assert count == len(expected_tokens)
    assert row[:count].tolist() == expected_tokens
    assert bool((row[count : PWIDTH - 1] == -1).all())


def test_zero_scores_rank_deterministically():
    """relu-sum scores that are exactly zero must still rank by block index."""
    lengths = (2404,)
    qi, ck, cu, cpage_base = make_inputs(lengths, seed=11)
    qi = (-torch.abs(qi.float())).to(torch.float8_e4m3fn)  # qi <= 0
    ck = torch.abs(ck.float()).to(torch.float8_e4m3fn)  # ck >= 0: relu == 0
    out = qsa_topk_indexer(qi, ck, cu, cpage_base)
    ref = ref_qsa_topk_indexer(qi, ck, cu, cpage_base)
    assert torch.equal(out, ref)


def test_backend_kwarg_explicit_matches_auto():
    qi, ck, cu, cpage_base = make_inputs((300, 212), seed=5)
    auto = qsa_topk_indexer(qi, ck, cu, cpage_base, backend="auto")
    spec = qsa_topk_indexer(qi, ck, cu, cpage_base, backend="specialized")
    assert torch.equal(auto, spec)
    with pytest.raises(ValueError, match="backend"):
        qsa_topk_indexer(qi, ck, cu, cpage_base, backend="fa2")


def test_kill_switch_forces_non_fused_path(monkeypatch):
    """FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1 must never dispatch the
    specialized kernel; with no composable fallback for this op, the call
    raises NotImplementedError and records the reason."""
    from flashinfer import qsa as qsa_mod

    qi, ck, cu, cpage_base = make_inputs((64,), seed=9)
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    before_reasons = len(qsa_mod.qsa_topk_indexer_fallback_reasons)
    before_stats = qsa_mod.qsa_topk_indexer_dispatch_stats["specialized"]
    for backend in ("auto", "specialized"):
        with pytest.raises(NotImplementedError, match="DISABLE"):
            qsa_topk_indexer(qi, ck, cu, cpage_base, backend=backend)
    assert len(qsa_mod.qsa_topk_indexer_fallback_reasons) == before_reasons + 2
    assert qsa_mod.qsa_topk_indexer_dispatch_stats["specialized"] == before_stats


def test_dispatch_stats_recorded():
    from flashinfer import qsa as qsa_mod

    qi, ck, cu, cpage_base = make_inputs((64,), seed=10)
    before = qsa_mod.qsa_topk_indexer_dispatch_stats["specialized"]
    qsa_topk_indexer(qi, ck, cu, cpage_base)
    assert qsa_mod.qsa_topk_indexer_dispatch_stats["specialized"] == before + 1
