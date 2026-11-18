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


# ---------------------------------------------------------------------------
# Analytic-score harness: q = [128, 1, 0, ...] and every key of request-local
# block b = [b >> 7, b & 127, 0, ...], so the block score is exactly b (all
# addends are BF16-exact and the f32 accumulation is exact), ascending in b.
# The expected selection is then analytic -- own <= 15 takes every prior
# block, otherwise the 15 highest-scoring prior blocks are own-15 .. own-1 --
# which keeps the TQ=16384 cases free of the O(TQ x blocks) CPU reference.
# ---------------------------------------------------------------------------


def _analytic_inputs(qlens, prefixes, device="cuda"):
    B = len(qlens)
    TQ = sum(qlens)
    cu = [0]
    for ln in qlens:
        cu.append(cu[-1] + ln)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)
    seq_lens = torch.tensor(
        [p + ln for p, ln in zip(prefixes, qlens, strict=True)], dtype=torch.int32
    )
    pgs = [(int(s) + BLOCK - 1) // BLOCK for s in seq_lens]
    kv_base = torch.tensor([sum(pgs[:i]) for i in range(B)], dtype=torch.int32)
    tkpad = sum(pgs) * BLOCK

    qi = torch.zeros(TQ, 1, D, dtype=torch.bfloat16)
    qi[:, 0, 0] = 128.0
    qi[:, 0, 1] = 1.0
    ki = torch.zeros(tkpad, 1, D, dtype=torch.bfloat16)
    for r in range(B):
        base = int(kv_base[r]) * BLOCK
        blk = torch.arange(pgs[r]).repeat_interleave(BLOCK).float()
        ki[base : base + pgs[r] * BLOCK, 0, 0] = (blk // 128).to(torch.bfloat16)
        ki[base : base + pgs[r] * BLOCK, 0, 1] = (blk % 128).to(torch.bfloat16)
    prefix_lens = torch.tensor(prefixes, dtype=torch.int32)
    return (
        qi.to(device),
        ki.to(device),
        cu_seqlens.to(device),
        kv_base.to(device),
        seq_lens.to(device),
        prefix_lens.to(device),
    )


def _analytic_expected(qlens, prefixes):
    rows = []
    for ln, p in zip(qlens, prefixes, strict=True):
        for i in range(ln):
            own = (p + i) >> 7
            if own <= 15:
                sel = list(range(own + 1)) + [-1] * (15 - own)
            else:
                sel = list(range(own - 15, own + 1))
            rows.append(sel)
    return torch.tensor(rows, dtype=torch.int32)


def _ties_expected(qlens, prefixes):
    rows = []
    for ln, p in zip(qlens, prefixes, strict=True):
        for i in range(ln):
            own = (p + i) >> 7
            prior = list(range(min(15, own)))
            sel = sorted(prior + [own])
            rows.append(sel + [-1] * (TOPK - len(sel)))
    return torch.tensor(rows, dtype=torch.int32)


# One case per TQ=16384 specialization the dispatcher owns, plus the ragged
# corners the specialized schedules must survive: unequal and empty request
# lengths and prefixes that are not a multiple of the 128-token block.
@pytest.mark.parametrize(
    "name,qlens,prefixes",
    [
        # B=1, 128 blocks: fixed-22 W1 persistent schedule + exact W1 selector.
        ("w1_persistent", [16384], [0]),
        # B=1, 640 blocks: W5 persistent schedule + exact W5 selector.
        ("w5_persistent", [16384], [65536]),
        # W5 with a non-block-aligned prefix.
        ("w5_unaligned_prefix", [16384], [65443]),
        # B=2, 1152 blocks: fused-selection persistent path (captured table).
        ("w6_fused_sel", [8192, 8192], [65536, 65536]),
        # Same specialization, unequal lengths + non-aligned prefixes: the
        # dynamic equal partition and per-request metadata path.
        ("w6_unequal", [12000, 4384], [65500, 65450]),
        # B=4, 2176 blocks: the fully fused single-pass kernel family.
        ("w7_fused", [4096, 4096, 4096, 4096], [65536] * 4),
        # Short-context TQ=16384 persistent dispatches (B=2 / B=4).
        ("persistent_b2_short", [8192, 8192], [0, 0]),
        ("persistent_b4_short", [4096] * 4, [0] * 4),
        # An empty request between two live ones.
        ("empty_request", [9000, 0, 7384], [4096, 4096, 4099]),
    ],
)
def test_tq16384_dispatch_analytic(name, qlens, prefixes):
    args = _analytic_inputs(qlens, prefixes)
    got = msa_blockscore_topk_indexer(*args)
    ref = _analytic_expected(qlens, prefixes)
    torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=0)


# Exact ties on the TQ=16384 specializations: all keys identical makes every
# strictly-prior block tie bitwise, so the lower-id contract fully determines
# the output on the persistent scorers and their exact selectors.
@pytest.mark.parametrize(
    "name,qlens,prefixes",
    [
        ("w5_ties", [16384], [65536]),
        ("w6_ties", [8192, 8192], [65536, 65536]),
        ("w7_ties", [4096, 4096, 4096, 4096], [65536] * 4),
    ],
)
def test_tq16384_exact_ties(name, qlens, prefixes):
    qi, ki, cu, kvb, sl, pl = _analytic_inputs(qlens, prefixes)
    qi = torch.ones_like(qi)
    ki = torch.ones_like(ki)
    got = msa_blockscore_topk_indexer(qi, ki, cu, kvb, sl, pl)
    ref = _ties_expected(qlens, prefixes)
    torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=0)


def test_fallback_inf_tie_keeps_lowest_ids():
    """All-tied +inf scores in the direct fallback: finite BF16 inputs whose
    f32 dot products overflow make every prior block score +inf. The top-15
    must still be the lowest-id prior blocks (strict tie contract), i.e.
    [0..14, own] -- not [1..15, own]."""
    TQ, prefix = 1, 2176  # own = 17: direct fallback path (TQ < 2)
    qi = torch.full((TQ, 1, D), 2.0**120, dtype=torch.bfloat16, device="cuda")
    pg = (prefix + TQ + BLOCK - 1) // BLOCK
    ki = torch.full((pg * BLOCK, 1, D), 2.0**120, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, TQ], dtype=torch.int32, device="cuda")
    kvb = torch.tensor([0], dtype=torch.int32, device="cuda")
    sl = torch.tensor([prefix + TQ], dtype=torch.int32, device="cuda")
    pl = torch.tensor([prefix], dtype=torch.int32, device="cuda")
    got = msa_blockscore_topk_indexer(qi, ki, cu, kvb, sl, pl)
    expect = list(range(15)) + [17]
    assert got[0].tolist() == expect, f"{got[0].tolist()} != {expect}"


def test_graph_replay_same_capture_stream():
    """Capture one call into a CUDA graph and replay it 20 times on the
    capture stream (the documented replay contract), checking the output
    after every replay."""
    qlens, prefixes = [1024, 1024], [4096, 4096]
    # Claim the primary workspace slot for the current stream first so the
    # capture below exercises the guarded (overflow-slot) call path too.
    warm = _analytic_inputs(qlens, prefixes)
    msa_blockscore_topk_indexer(*warm)

    s = torch.cuda.Stream()
    args = _analytic_inputs(qlens, prefixes)
    out = torch.empty((sum(qlens), TOPK), dtype=torch.int32, device="cuda")
    ref = _analytic_expected(qlens, prefixes)

    with torch.cuda.stream(s):
        msa_blockscore_topk_indexer(*args, output=out)  # build + warm up
    s.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        msa_blockscore_topk_indexer(*args, output=out)

    for replay in range(20):
        out.zero_()
        with torch.cuda.stream(s):
            g.replay()
        s.synchronize()
        torch.testing.assert_close(
            out.cpu(), ref, rtol=0, atol=0, msg=f"replay {replay} mismatched"
        )


def test_concurrent_eager_overflow_streams():
    """Concurrent eager calls on two non-primary streams share the guarded
    overflow workspace; the release/acquire handoff must keep every call's
    scratch state private to its lock tenure. One stream drives the staged
    scorer, the other the fused single-pass kernel, 30 interleaved calls
    each with no intervening synchronization."""
    # Ensure the primary slot is owned by the current stream, so both side
    # streams below take the guarded overflow slot.
    warm = _analytic_inputs([256], [4096])
    msa_blockscore_topk_indexer(*warm)

    qlens_a, prefixes_a = [4096], [65536]  # staged W5-tier scorer+selector
    qlens_b, prefixes_b = [4096] * 4, [65536] * 4  # fused single-pass kernel
    args_a = _analytic_inputs(qlens_a, prefixes_a)
    args_b = _analytic_inputs(qlens_b, prefixes_b)
    ref_a = _analytic_expected(qlens_a, prefixes_a)
    ref_b = _analytic_expected(qlens_b, prefixes_b)

    n_iters = 30
    outs_a = [
        torch.empty((sum(qlens_a), TOPK), dtype=torch.int32, device="cuda")
        for _ in range(n_iters)
    ]
    outs_b = [
        torch.empty((sum(qlens_b), TOPK), dtype=torch.int32, device="cuda")
        for _ in range(n_iters)
    ]
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    # Build both module paths before the contended loop.
    with torch.cuda.stream(s1):
        msa_blockscore_topk_indexer(*args_a)
    with torch.cuda.stream(s2):
        msa_blockscore_topk_indexer(*args_b)
    torch.cuda.synchronize()

    for i in range(n_iters):
        with torch.cuda.stream(s1):
            msa_blockscore_topk_indexer(*args_a, output=outs_a[i])
        with torch.cuda.stream(s2):
            msa_blockscore_topk_indexer(*args_b, output=outs_b[i])
    torch.cuda.synchronize()

    for i in range(n_iters):
        torch.testing.assert_close(
            outs_a[i].cpu(), ref_a, rtol=0, atol=0, msg=f"staged call {i} mismatched"
        )
        torch.testing.assert_close(
            outs_b[i].cpu(), ref_b, rtol=0, atol=0, msg=f"fused call {i} mismatched"
        )
