"""Correctness of ``msa_blockscore_topk_indexer`` (SM107) against a CPU reference.

The reference is computed on the CPU in float32 and treated as ground truth:
torch's GPU matmul can legitimately flip a near-boundary top-k pick on large
random inputs (fp fusion/contraction order differs from the CPU path), so the
large-shape tests assert a matched-row ratio of at least 0.999 rather than
bitwise equality, while small shapes and the constructed-tie test (where every
score gap is exact) assert exact equality.
"""

import pytest
import torch

from flashinfer.msa_ops import (
    is_msa_blockscore_indexer_supported,
    msa_blockscore_indexer_stats,
    msa_blockscore_topk_indexer,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_msa_blockscore_indexer_supported(torch.device("cuda")),
    reason="msa_blockscore_topk_indexer requires SM107",
)

BLOCK = 128
TOPK = 16
D = 128
QCHUNK = 2048


def _ref_cpu(qi, ki, cu_seqlens, kv_base, seq_lens, prefix_lens):
    """CPU float32 ground-truth reference (block MAX score, forced own block,
    top-15 strictly-prior with lower-id tie-break, ascending, -1 tail)."""
    cu = cu_seqlens.cpu().long()
    B = cu.numel() - 1
    TQ = qi.shape[0]
    qf = qi.cpu().reshape(-1, D).float()
    k_flt = ki.cpu().reshape(-1, D).float()
    kv_base = kv_base.cpu()
    seq_lens = seq_lens.cpu()
    prefix_lens = prefix_lens.cpu()
    sel = torch.full((TQ, TOPK), -1, dtype=torch.int32)
    for r in range(B):
        q0, q1 = int(cu[r]), int(cu[r + 1])
        if q1 == q0:
            continue
        s_len = int(seq_lens[r])
        pg = (s_len + BLOCK - 1) // BLOCK
        base = int(kv_base[r]) * BLOCK
        kseg = k_flt[base : base + pg * BLOCK].reshape(pg, BLOCK, D)
        kpos = torch.arange(pg).unsqueeze(1) * BLOCK + torch.arange(BLOCK).unsqueeze(0)
        pos_r = int(prefix_lens[r]) + torch.arange(q1 - q0)
        for c0 in range(0, q1 - q0, QCHUNK):
            c1 = min(c0 + QCHUNK, q1 - q0)
            qc = qf[q0 + c0 : q0 + c1]
            lg = torch.einsum("qd,pkd->qpk", qc, kseg)
            vis = (kpos.unsqueeze(0) <= pos_r[c0:c1].view(-1, 1, 1)) & (
                kpos.unsqueeze(0) < s_len
            )
            scores = lg.masked_fill(~vis, float("-inf")).amax(-1)
            own = (pos_r[c0:c1] // BLOCK).long()
            blk = torch.arange(pg)
            s = scores.masked_fill(blk.unsqueeze(0) > own.unsqueeze(1), float("-inf"))
            s = s.scatter(1, own.unsqueeze(1), float("-inf"))
            k = min(TOPK - 1, pg)
            order = torch.argsort(-s, dim=1, stable=True)[:, :k]
            vals = s.gather(1, order)
            picks = torch.where(torch.isinf(vals), torch.full_like(order, -1), order)
            full = torch.cat([picks, own.unsqueeze(1)], dim=1).int()
            big = torch.where(full < 0, torch.full_like(full, 2**30), full)
            srt, _ = torch.sort(big, dim=1)
            srt = torch.where(srt >= 2**30, torch.full_like(srt, -1), srt)
            sel[q0 + c0 : q0 + c1, : srt.shape[1]] = srt
    return sel


def _make_inputs(TQ, B, prefix, seed, device="cuda"):
    """Variable-length batch in the kernel's layout: per-request keys stored in
    contiguous whole 128-token blocks starting at block ``kv_base[r]``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = TQ // B
    qlens = [base + (1 if i < TQ - base * B else 0) for i in range(B)]
    cu = [0]
    for ln in qlens:
        cu.append(cu[-1] + ln)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)
    seq_lens = torch.tensor([prefix + ln for ln in qlens], dtype=torch.int32)
    pgs = [(int(s) + BLOCK - 1) // BLOCK for s in seq_lens]
    kv_base = torch.tensor(
        [sum(pgs[:i]) for i in range(B)],
        dtype=torch.int32,
    )
    tkpad = sum(pgs) * BLOCK
    qi = torch.randn(TQ, 1, D, generator=gen).to(torch.bfloat16)
    ki = torch.randn(tkpad, 1, D, generator=gen).to(torch.bfloat16)
    prefix_lens = torch.full((B,), prefix, dtype=torch.int32)
    return (
        qi.to(device),
        ki.to(device),
        cu_seqlens.to(device),
        kv_base.to(device),
        seq_lens.to(device),
        prefix_lens.to(device),
    )


def _matched_ratio(got, ref):
    return (got.cpu() == ref).all(dim=1).float().mean().item()


@pytest.mark.parametrize(
    "TQ,B,prefix",
    [
        (48, 1, 0),  # fewer blocks than TOPK: -1 tail padding
        (300, 2, 0),  # uneven qlens across requests
        (257, 3, 200),  # prefix not a multiple of BLOCK
        (640, 2, 384),
    ],
)
def test_small_shapes_exact(TQ, B, prefix):
    args = _make_inputs(TQ, B, prefix, seed=TQ * 7 + B + prefix)
    got = msa_blockscore_topk_indexer(*args)
    ref = _ref_cpu(*args)
    assert got.shape == (TQ, TOPK) and got.dtype == torch.int32
    torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "TQ,B,prefix",
    [
        (4096, 1, 0),
        (4096, 2, 8192),
        (2048, 4, 4096),
    ],
)
def test_large_shapes_matched_ratio(TQ, B, prefix):
    """Large random shapes: the row-match ratio bounds reference fp noise, and
    every mismatching row must still be a valid selection (ascending, own
    block present, ids in range)."""
    args = _make_inputs(TQ, B, prefix, seed=101 + TQ + B)
    got = msa_blockscore_topk_indexer(*args)
    ref = _ref_cpu(*args)
    ratio = _matched_ratio(got, ref)
    assert ratio >= 0.999, f"matched-row ratio {ratio} < 0.999"

    # Structural invariants hold on every row, matched or not: valid ids are
    # strictly ascending and -1 entries form a contiguous tail.
    g = got.cpu().long()
    valid = g >= 0
    for row in range(g.shape[0]):
        n = int(valid[row].sum())
        assert valid[row, :n].all() and not valid[row, n:].any(), (
            f"row {row}: -1 entries must be a contiguous tail: {g[row].tolist()}"
        )
        ids = g[row, :n]
        assert (ids[1:] > ids[:-1]).all(), f"row {row} not strictly ascending"


def test_constructed_tie_deterministic():
    """All keys identical: every strictly-prior block ties exactly, so the
    deterministic tie-break (lower block id wins) fully determines the output:
    blocks 0..min(14, own-1) plus the own block, ascending, -1 tail."""
    TQ, B, prefix = 512, 1, 2048  # own block index = 16.. for every query
    qi, ki, cu_seqlens, kv_base, seq_lens, prefix_lens = _make_inputs(
        TQ, B, prefix, seed=5
    )
    ki = torch.ones_like(ki)  # exact ties across all blocks
    args = (qi, ki, cu_seqlens, kv_base, seq_lens, prefix_lens)
    got = msa_blockscore_topk_indexer(*args).cpu()

    pos = prefix + torch.arange(TQ)
    own = pos // BLOCK
    for t in range(TQ):
        o = int(own[t])
        prior = list(range(min(TOPK - 1, o)))
        expect = sorted(prior + [o])
        expect = expect + [-1] * (TOPK - len(expect))
        assert got[t].tolist() == expect, f"token {t}: {got[t].tolist()} != {expect}"

    # And the CPU reference agrees bitwise (gaps are exact, no fp ambiguity).
    ref = _ref_cpu(*args)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_preallocated_output_and_rejects():
    args = _make_inputs(256, 2, 0, seed=9)
    out = torch.empty((256, TOPK), dtype=torch.int32, device="cuda")
    got = msa_blockscore_topk_indexer(*args, output=out)
    assert got.data_ptr() == out.data_ptr()

    with pytest.raises(ValueError):
        msa_blockscore_topk_indexer(*args, topk=32)
    with pytest.raises(ValueError):
        msa_blockscore_topk_indexer(*args, block_size=64)
    qi, ki, cu, kvb, sl, pl = args
    with pytest.raises(ValueError):
        msa_blockscore_topk_indexer(qi.float(), ki, cu, kvb, sl, pl)
    with pytest.raises(ValueError):
        msa_blockscore_topk_indexer(qi, ki[: BLOCK - 1], cu, kvb, sl, pl)


def test_backend_kwarg_semantics(monkeypatch):
    """backend="auto" and backend="specialized" serve identically (the fused
    kernel is the only implementation); an unknown backend is rejected and the
    kill switch raises NotImplementedError (no composable fallback exists)."""
    args = _make_inputs(300, 2, 256, seed=11)
    ref = _ref_cpu(*args)
    auto = msa_blockscore_topk_indexer(*args, backend="auto")
    spec = msa_blockscore_topk_indexer(*args, backend="specialized")
    torch.testing.assert_close(auto.cpu(), ref, rtol=0, atol=0)
    torch.testing.assert_close(spec.cpu(), auto.cpu(), rtol=0, atol=0)

    with pytest.raises(ValueError, match="backend"):
        msa_blockscore_topk_indexer(*args, backend="fa2")

    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    for b in ("auto", "specialized"):
        with pytest.raises(NotImplementedError, match="SPECIALIZED_KERNEL_DISABLE"):
            msa_blockscore_topk_indexer(*args, backend=b)
    monkeypatch.delenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE")

    stats = msa_blockscore_indexer_stats()
    assert stats["backends"] == ("auto", "specialized")
    assert stats["backend_dispatches"]["specialized"] >= 2
    assert stats["fallback_reasons"].get("kill_switch", 0) >= 2
