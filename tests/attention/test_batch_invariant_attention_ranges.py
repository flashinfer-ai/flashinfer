# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The ranged batch-invariant attention class behind ``batch_invariant=True`` on
``BatchPrefillWithCausalBidirectionalRangesWrapper``: a query token sees its causal keys and one key range of its request.
Bitwise tests (a token without a range has the causal class's bits; a token's bits depend on its own row alone), an fp32
reference for the mask itself, the device-free checks of the term, the planner and the receipt, and the typed refusals."""

import importlib.util
import math
import os

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.attention.batch_invariant.wrappers import ATTENTION_ARCHS

_base_spec = importlib.util.spec_from_file_location(
    "_bi_attention_base",
    os.path.join(os.path.dirname(__file__), "test_batch_invariant_attention.py"),
)
assert _base_spec is not None and _base_spec.loader is not None
_base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base)
Problem, bits, workspace = _base.Problem, _base.bits, _base.workspace

needs_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bi.is_supported_device(torch.device("cuda"), ATTENTION_ARCHS),
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

RANGES_WRAPPER = "BatchPrefillWithCausalBidirectionalRangesWrapper"
FINGERPRINTS = {
    64: "attn-bf16-d64-br-79871238fea8278a",
    128: "attn-bf16-d128-br-d9243bef6256a86d",
    256: "attn-bf16-d256-br-22d7e4213307d326",
}


# ------------------------------------------------------------------------------------------------------ problems
def span_rows(pr, spans):
    """The wrapper's range tensor for ``pr``: ``spans[r]`` lists request ``r``'s disjoint inclusive ``(start, end)`` spans in
    absolute key positions; a query token inside a span gets that span's row, every other token ``(-1, -1)``."""
    rows = []
    for ql, kl, sp in zip(pr.q_lens, pr.kv_lens, spans, strict=True):
        for i in range(ql):
            p = kl - ql + i
            hit = [se for se in sp if se[0] <= p <= se[1]]
            rows.append(list(hit[0]) if hit else [-1, -1])
    return torch.tensor(rows, dtype=torch.int32, device=pr.device).reshape(-1, 2)


def make_wrapper(pr, layout="NHD", ws=None, **plan_kw):
    w = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        ws if ws is not None else workspace(pr.device),
        layout,
        head_dim_qk=pr.D,
        head_dim_vo=pr.D,
        batch_invariant=True,
    )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        **plan_kw,
    )
    return w


def run_ranges(pr, ranges, causal_window=-1, range_window=-1, layout="NHD", ws=None):
    return make_wrapper(pr, layout, ws).run(
        pr.q,
        pr.cache(layout),
        ranges,
        causal_window_left=causal_window,
        range_window_left=range_window,
    )


def run_causal(pr, window_left=-1, layout="NHD"):
    return _base.run_paged_prefill(pr, layout, window_left)


def reference(pr, ranges, causal_window=-1, range_window=-1):
    """fp32 attention under the wrapper's mask, ``(causal AND causal window) OR (in range AND range window)``: the yardstick
    for the mask and the accuracy (never for bits)."""
    outs = []
    tok_q = tok_kv = 0
    g = pr.Hq // pr.Hkv
    for ql, kl in zip(pr.q_lens, pr.kv_lens, strict=True):
        q = pr.q[tok_q : tok_q + ql].float()
        k = pr.k_rag[tok_kv : tok_kv + kl].float().repeat_interleave(g, dim=1)
        v = pr.v_rag[tok_kv : tok_kv + kl].float().repeat_interleave(g, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(pr.D)
        pos = torch.arange(kl - ql, kl, device=pr.device)[:, None]
        key = torch.arange(kl, device=pr.device)[None, :]
        mask = key <= pos
        if causal_window >= 0:
            mask &= pos - key <= causal_window
        r = ranges[tok_q : tok_q + ql].to(torch.int64)
        in_range = (key >= r[:, :1]) & (key <= r[:, 1:])
        if range_window > 0:
            in_range &= pos - key < range_window
        mask |= in_range
        s = s.masked_fill(~mask[None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), v))
        tok_q += ql
        tok_kv += kl
    return torch.cat(outs)


def close(out, ref):
    err = (out.float() - ref).abs().max().item()
    return err < 2e-2 + 1e-2 * ref.abs().max().item()


def offsets(pr):
    return [0] + torch.cumsum(torch.tensor(pr.q_lens), 0).tolist()


# ------------------------------------------------------------------------------------------------------ device-free
@pytest.mark.parametrize("D", [64, 128, 256])
def test_receipt_of_the_ranged_class(D):
    """The ranged class has its own receipt per head dim, next to the causal one (which does not move)."""
    from flashinfer.attention.batch_invariant import spec

    fp = bi.fingerprint(RANGES_WRAPPER, head_dim=D)
    assert fp == FINGERPRINTS[D]
    causal = bi.fingerprint("BatchPrefillWithPagedKVCacheWrapper", head_dim=D)
    assert causal == spec.FINGERPRINTS[D] and fp != causal
    assert bi.class_name(RANGES_WRAPPER) == "attention.bf16"
    with pytest.raises(TypeError):
        bi.fingerprint(RANGES_WRAPPER, head_dim=D, batch=8)


def test_a_row_without_a_range_has_the_causal_term():
    """The ranged row with no range, an empty range or a range inside its causal keys is the causal row, node for node (the
    same digest); a range that adds a key changes the term."""
    from flashinfer.attention.batch_invariant.spec import AttentionSpec, RowSpec
    from flashinfer.batch_invariant.fp_interface.term import fingerprint

    spec = AttentionSpec(64, T=16, C=32)
    for p, win in ((40, -1), (40, 20), (7, -1), (100, 0)):
        causal = fingerprint(spec.output(RowSpec(p=p, window_left=win), 0))
        lo = 0 if win < 0 else max(0, p - win)
        for same in ((0, 0), (9, 9), (50, 3), (lo, p + 1), (p, p + 1)):
            row = RowSpec(p=p, window_left=win, brange=same)
            assert fingerprint(spec.output(row, 0)) == causal, (p, win, same)
        for other in ((p + 1, p + 2), (p - 3, p + 40), (0, 200)):
            row = RowSpec(p=p, window_left=win, brange=other)
            assert fingerprint(spec.output(row, 0)) != causal, (p, win, other)
    # a row inside its range with no window sees a prefix: the causal row at the range's last key, for the same query row
    a = spec.output(RowSpec(p=20, brange=(10, 71)), 1)
    b = spec.output(RowSpec(p=70), 1)
    assert fingerprint(a) == fingerprint(b)
    # a range behind the window leaves tiles and a chunk without a key between the two parts
    holed = RowSpec(p=100, window_left=3, brange=(5, 20))
    assert [k for k in range(101) if holed.visible(k)] == list(range(5, 20)) + list(
        range(97, 101)
    )
    from flashinfer.attention.batch_invariant.spec import present_chunks, chunk_len

    C = chunk_len(64)
    wide = RowSpec(p=4 * C + 5, window_left=2, brange=(C - 1, C + 1))
    assert present_chunks(wide, 64) == [0, 1, 4]


def test_planner_takes_the_hull_of_the_visible_keys():
    """A row block's tiles and chunks cover its tokens' causal keys and ranges; without a range the plan is the causal one."""
    from flashinfer.attention.batch_invariant import plan as P

    qo, kv_indptr, last = [0, 300, 1000], [0, 44, 144], [12, 16]  # kv 700 and 1600
    kw = dict(Hq=2, Hkv=2, D=128, page=16, sm_count=148)
    none = [[0, 0]] * 1000
    causal = P.build_plan(qo, kv_indptr, last, **kw)
    assert P.build_plan(qo, kv_indptr, last, ranges=none, **kw).desc == causal.desc
    import numpy as np

    assert (
        P.build_plan(
            qo, kv_indptr, last, ranges=np.zeros((1000, 2), np.int32), **kw
        ).desc
        == causal.desc
    )
    ranges = [list(r) for r in none]
    for i in range(
        300, 364
    ):  # request 1, tokens 0..63 (positions 900..963): keys [1000, 1500)
        ranges[i] = [1000, 1500]
    for i in range(
        0, 4
    ):  # request 0, tokens 0..3 (positions 400..403) under a window: keys [10, 20) behind it
        ranges[i] = [10, 20]
    plan = P.build_plan(
        qo,
        kv_indptr,
        last,
        window_left=100,
        split_policy="fused",
        ranges=ranges,
        **kw,
    )
    by_block = {(d[P.F_REQ], d[P.F_QTOK0]): d for d in plan.desc if d[P.F_KVHEAD] == 0}
    assert by_block[(1, 0)][P.F_TILE_HI] == -(-1500 // 128)
    assert by_block[(1, 0)][P.F_CHUNK_HI] == -(-1500 // 512)
    assert by_block[(1, 0)][P.F_TILE_LO] == (900 - 100) // 128
    assert by_block[(1, 128)][P.F_TILE_HI] == -(-(900 + 256) // 128)
    assert by_block[(0, 0)][P.F_TILE_LO] == 0 and by_block[(0, 0)][P.F_CHUNK_LO] == 0
    assert by_block[(0, 128)][P.F_TILE_LO] == (400 + 128 - 100) // 128
    as_array = P.build_plan(
        qo,
        kv_indptr,
        last,
        window_left=100,
        split_policy="fused",
        ranges=np.array(ranges, dtype=np.int32),
        **kw,
    )
    assert as_array.desc == plan.desc
    with pytest.raises(AssertionError):
        bad = [list(r) for r in none]
        bad[0] = [0, 701]  # past the request's keys
        P.build_plan(qo, kv_indptr, last, ranges=bad, **kw)
    with pytest.raises(AssertionError):
        P.build_plan(qo, kv_indptr, last, ranges=np.array(bad, dtype=np.int32), **kw)


def test_descriptor_table_carries_the_ranges():
    """The range table follows the unit rows: eight tokens per row, a zero tail covering a Q tile past the last token."""
    import numpy as np

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.plan import (
        DESC_FIELDS,
        F_RANGES,
        RANGE_ROW_TOKENS,
    )

    units = [[7] * DESC_FIELDS for _ in range(3)]
    ranges = np.array([[i, i + 5] for i in range(19)], dtype=np.int32)
    table = engine._desc_table(units, ranges).tolist()
    assert all(r[F_RANGES] == 3 for r in table[:3]) and units[0][F_RANGES] == 7
    assert len(table) == 3 + -(-(19 + 128) // RANGE_ROW_TOKENS)
    for t, (a, b) in enumerate(ranges.tolist()):
        row = table[3 + t // RANGE_ROW_TOKENS]
        col = 2 * (t % RANGE_ROW_TOKENS)
        assert row[col : col + 2] == [a, b]
    assert all(x == 0 for r in table[6:] for x in r)
    assert engine._desc_table(units, None).tolist() == units
    assert engine._desc_table([], None).tolist() == [[0] * DESC_FIELDS]


def test_kernel_identity_carries_the_mask_mode():
    from flashinfer.attention.batch_invariant.blocks.compile import kernel_name

    causal = kernel_name(128, 4, 16, 4, 1, False, False, True, 228 << 10)
    ranged = kernel_name(128, 4, 16, 4, 1, False, False, True, 228 << 10, True)
    assert ranged == causal + "_br"


# ------------------------------------------------------------------------------------------------------ on the device
SPANS = [
    [(100, 139), (600, 1100)],  # inside a tile; across tiles and the first chunk end
    [(1600, 2200), (2300, 2301)],  # partly before the scheduled tokens
    [],
    [(0, 2047)],  # the whole request
]


def span_problem(D, Hq=32, Hkv=8, page=16, seed=21):
    return Problem(
        [1200, 800, 64, 2048], [1200, 2400, 3000, 2048], Hq, Hkv, D, page, seed=seed
    )


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
def test_matches_reference(D, Hq, Hkv):
    pr = span_problem(D, Hq, Hkv)
    ranges = span_rows(pr, SPANS)
    assert close(run_ranges(pr, ranges), reference(pr, ranges))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_random_ranges_match_reference(D):
    """Arbitrary rows (the wrapper does not require a token to lie inside its range): ranges ahead of, behind and around the
    token, negative starts, ends past the request, inverted rows, with and without the two windows."""
    pr = Problem([700, 257, 33], [1500, 257, 2100], 16, 4, D, 16, seed=22)
    g = torch.Generator().manual_seed(5)
    rows = []
    for ql, kl in zip(pr.q_lens, pr.kv_lens, strict=True):
        start = torch.randint(-50, kl + 50, (ql,), generator=g)
        length = torch.randint(-20, 900, (ql,), generator=g)
        keep = torch.rand(ql, generator=g) < 0.7
        rows.append(
            torch.where(
                keep[:, None],
                torch.stack((start, start + length), dim=1),
                torch.full((ql, 2), -1),
            )
        )
    ranges = torch.cat(rows).to(torch.int32).to(pr.device)
    for cw, rw in ((-1, -1), (300, 0), (0, 64), (700, 200)):
        out = run_ranges(pr, ranges, cw, rw)
        assert close(out, reference(pr, ranges, cw, rw)), (cw, rw)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_rows_without_a_range_have_the_causal_class_bits(D):
    """No range anywhere: the causal batch-invariant wrapper's bits, with and without a window. In a batch with spans, every
    token outside the spans (a whole request, or the tokens between two spans) keeps those bits."""
    pr = span_problem(D)
    none = torch.full((sum(pr.q_lens), 2), -1, dtype=torch.int32, device=pr.device)
    causal = run_causal(pr)
    assert torch.equal(bits(run_ranges(pr, none)), bits(causal))
    assert torch.equal(bits(run_ranges(pr, none, 600)), bits(run_causal(pr, 600)))
    assert torch.equal(bits(run_ranges(pr, none, 0, 77)), bits(run_causal(pr, 0)))
    ranges = span_rows(pr, SPANS)
    out = run_ranges(pr, ranges)
    plain = (ranges[:, 0] < 0).nonzero().flatten()
    assert 0 < plain.numel() < ranges.shape[0]
    assert torch.equal(bits(out[plain]), bits(causal[plain]))
    spanned = (ranges[:, 0] >= 0).nonzero().flatten()
    assert not torch.equal(bits(out[spanned]), bits(causal[spanned]))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_a_row_inside_a_span_sees_the_whole_span(D):
    """A token inside its span with no window sees the keys ``[0, end]``: the term of the causal row at the span's last key,
    so its bits are those of the causal class for the same query row placed at ``end`` -- for spans inside a tile, across
    tiles, across chunks and up to the last key."""
    pr = Problem([1800], [1800], 16, 4, D, 16, seed=23)
    spans = [(40, 90), (100, 139), (500, 520), (600, 1100), (1500, 1799)]
    ranges = span_rows(pr, [spans])
    out = run_ranges(pr, ranges)
    for s, e in spans:
        picks = sorted({s, (s + e) // 2, e})
        # the same keys [0, e] under the causal class: one request per picked token, each alone at position e
        cz = Problem([1] * len(picks), [e + 1] * len(picks), 16, 4, D, 16, seed=24)
        for j, p in enumerate(picks):
            cz.q[j] = pr.q[p]
        cz.k_rag[:] = pr.k_rag[: e + 1].repeat(len(picks), 1, 1)
        cz.v_rag[:] = pr.v_rag[: e + 1].repeat(len(picks), 1, 1)
        k, v = cz.ragged("NHD")
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace(pr.device), "NHD", batch_invariant=True
        )
        w.plan(
            cz.qo_indptr,
            cz.kv_indptr_rag,
            16,
            4,
            D,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        at_end = w.run(cz.q, k, v)
        assert torch.equal(bits(out[picks]), bits(at_end)), (s, e)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_batch_permutation_and_subbatch_invariance(D):
    """R2: a request's tokens have the same bits alone, in the batch, in a permuted batch and in a sub-batch."""
    pr = span_problem(D, seed=25)
    ranges = span_rows(pr, SPANS)
    out = run_ranges(pr, ranges)
    off = offsets(pr)
    kv_off = [0] + torch.cumsum(torch.tensor(pr.kv_lens), 0).tolist()

    def sub(order):
        s = Problem(
            [pr.q_lens[r] for r in order],
            [pr.kv_lens[r] for r in order],
            pr.Hq,
            pr.Hkv,
            D,
            pr.page,
            seed=26,
        )
        s.q = torch.cat([pr.q[off[r] : off[r + 1]] for r in order])
        k = torch.cat([pr.k_rag[kv_off[r] : kv_off[r + 1]] for r in order])
        v = torch.cat([pr.v_rag[kv_off[r] : kv_off[r + 1]] for r in order])
        s = _base._repack(_with_keys(s, k, v))
        rg = torch.cat([ranges[off[r] : off[r + 1]] for r in order]).contiguous()
        return run_ranges(s, rg), torch.cat([out[off[r] : off[r + 1]] for r in order])

    for order in ([0], [1], [3], [2, 0], [3, 1, 0, 2], [1, 3]):
        got, want = sub(order)
        assert torch.equal(bits(got), bits(want)), order


def _with_keys(pr, k, v):
    pr.k_rag, pr.v_rag = k, v
    return pr


@needs_device
@pytest.mark.parametrize("D", [64, 128])
def test_a_request_is_independent_of_the_other_requests_spans(D):
    """A token's bits depend on its own range row: changing, adding or removing the spans of the other requests of the batch
    (which moves the planner's tile ranges, unit kinds and order) leaves them unchanged."""
    pr = span_problem(D, seed=27)
    base = run_ranges(pr, span_rows(pr, SPANS))
    off = offsets(pr)
    variants = [
        [SPANS[0], [], [(2940, 2999)], []],
        [SPANS[0], [(1700, 2399)], [], [(5, 9)]],
        [[], SPANS[1], [], SPANS[3]],
    ]
    for spans in variants:
        out = run_ranges(pr, span_rows(pr, spans))
        for r in range(4):
            if spans[r] == SPANS[r]:
                assert torch.equal(
                    bits(out[off[r] : off[r + 1]]), bits(base[off[r] : off[r + 1]])
                ), (spans, r)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_page_sizes_and_layouts(D):
    """The page size and the KV layout are placements: the same keys at the same positions give the same bits."""
    ref = None
    for page in (1, 16, 128):
        pr = Problem([900, 300], [1400, 2100], 32, 8, D, page, seed=28)
        ranges = span_rows(pr, [[(450, 1399)], [(1790, 1900), (2000, 2099)]])
        for layout in ("NHD", "HND"):
            out = bits(run_ranges(pr, ranges, 1000, 300, layout))
            if ref is None:
                ref = out
            assert torch.equal(out, ref), (page, layout)


@needs_device
def test_range_window_is_a_clamp_of_the_range():
    """``range_window_left = N`` keeps the range keys with ``pos - key < N``: the bits of the rows clamped by hand; ``0`` and
    negative values leave the ranges alone. ``causal_window_left`` is FlashInfer's ``window_left``."""
    pr = Problem([1000], [1300], 16, 4, 128, 16, seed=29)
    ranges = span_rows(pr, [[(350, 800), (900, 1299)]])
    clamped = ranges.clone()
    pos = torch.arange(300, 1300, device=pr.device, dtype=torch.int32)
    has = ranges[:, 0] >= 0
    clamped[has, 0] = torch.maximum(ranges[has, 0], pos[has] - 99)
    out = run_ranges(pr, ranges, 50, 100)
    assert torch.equal(bits(out), bits(run_ranges(pr, clamped, 50, -1)))
    assert close(out, reference(pr, ranges, 50, 100))
    free = run_ranges(pr, ranges, 50, 0)
    assert torch.equal(bits(free), bits(run_ranges(pr, ranges, 50, -7)))
    assert not torch.equal(bits(free), bits(out))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_config_agreement(D):
    """R3: the unit kinds (fused, split), the fold placements (in-kernel, the fixup kernel), the ring depths, the softmax
    warpgroups, the paired placement and the SM count the planner sees all give the wrapper's bits -- on a batch whose rows
    have ranges around them, ranges behind a window (tiles and whole chunks without a key between the two parts), ranges a few
    keys ahead (a gap inside a tile) and no range."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.swaps.operands import arch_smem_cap

    pr = Problem([1400, 2048, 3, 40], [2600, 2048, 3000, 2300], 32, 8, D, 16, seed=30)
    ranges = span_rows(
        pr, [[(1300, 2100), (2400, 2599)], [(0, 700), (1500, 2047)], [], [(2270, 2290)]]
    )
    # tokens of request 0 before its first span: a range far behind the window
    ranges[0:100] = torch.tensor([20, 600], dtype=torch.int32)
    # ... and tokens with a short range a few keys ahead: the causal part and the range part in one tile, apart
    ahead = torch.arange(1300, 1400, dtype=torch.int32, device=pr.device)
    ranges[100:200] = torch.stack((ahead + 10, ahead + 30), dim=1)
    # ... and the last tokens of request 3 with a range at the start of the request: whole chunks without a key in between
    ranges[-8:] = torch.tensor([10, 40], dtype=torch.int32)
    window = 300
    w = make_wrapper(pr)
    ref = bits(w.run(pr.q, pr.cache("NHD"), ranges, causal_window_left=window))
    assert close(ref.view(torch.bfloat16), reference(pr, ranges, window))
    delegate = w._batch_invariant
    class_ranges = delegate._class_ranges(ranges.cpu(), 0)
    base = engine.default_config(D=D)
    cfgs = [
        replace(base, split_policy=p, merge=m, softmax_wgs=wg, kv_slots=s)
        for p, m in (
            ("fused", "auto"),
            ("split", "fixup"),
            ("split", "inkernel"),
            ("auto", "auto"),
        )
        for wg in (1, 2)
        for s in (2, engine.DEFAULT_RING[D])
    ]
    cfgs += [replace(base, small="rep"), replace(base, grid=3), replace(base, grid=40)]
    q2_sm = None
    if D == 64 or (D == 128 and arch_smem_cap(pr.device) >= (300 << 10)):
        two = replace(base, softmax_wgs=2, q2=True)
        q2_sm = 1
        cfgs += [
            replace(two, kv_slots=2, split_policy=p, merge=m)
            for p, m in (("auto", "auto"), ("split", "auto"), ("split", "inkernel"))
        ]
    seen = set()
    for cfg in cfgs:
        for sm in (None, 1, 7):
            if cfg.q2:
                sm = q2_sm
            if (cfg, sm) in seen:
                continue
            seen.add((cfg, sm))
            planned = engine.plan_attention(
                pr.qo_indptr,
                pr.kv_indptr,
                pr.kv_last,
                32,
                8,
                D,
                16,
                window,
                cfg,
                None,
                pr.device,
                sm_count=sm,
                ranges=class_ranges,
            )
            if cfg.q2:
                assert planned.plan.num_launch is not None, cfg
            out = engine.run_attention(
                planned,
                pr.q,
                pr.k_nhd,
                pr.v_nhd,
                pr.kv_indptr,
                pr.kv_indices,
                1.0 / math.sqrt(D),
                window,
                "NHD",
                None,
                cfg,
            )
            assert torch.equal(bits(out), ref), (cfg, sm)


@needs_device
def test_small_units_take_the_row_body():
    """Decode-sized units (which the causal class runs on the Swaps-AB kernel) take the row body under the ranged class:
    single tokens and short tails with a range ahead of them, behind them and none."""
    pr = Problem([1, 1, 3, 1, 8], [900, 1700, 1300, 5, 4000], 8, 8, 128, 16, seed=31)
    ranges = span_rows(pr, [[(880, 899)], [], [(1200, 1299)], [(0, 4)], [(3990, 3996)]])
    ranges[1] = torch.tensor([100, 400], dtype=torch.int32)
    ranges[6] = torch.tensor(
        [3996, 3997], dtype=torch.int32
    )  # position 3992: a gap inside its tile
    out = run_ranges(pr, ranges, 200)
    assert close(out, reference(pr, ranges, 200))
    none = torch.full_like(ranges, -1)
    assert torch.equal(bits(run_ranges(pr, none, 200)), bits(run_causal(pr, 200)))


@needs_device
def test_chunked_prefill_equals_full():
    """R4: the tokens of a later prefill chunk (``q_len < kv_len``) have the bits of the same tokens of the one-shot prefill
    when their range rows are the same."""
    full = Problem([1500], [1500], 16, 4, 128, 16, seed=32)
    spans = [[(200, 420), (700, 1300)]]
    rows = span_rows(full, spans)
    one = run_ranges(full, rows, 500, 64)
    for start in (1, 512, 800, 1499):
        tail = Problem([1500 - start], [1500], 16, 4, 128, 16, seed=33)
        tail.q = full.q[start:].contiguous()
        tail = _base._repack(_with_keys(tail, full.k_rag, full.v_rag))
        got = run_ranges(tail, rows[start:].contiguous(), 500, 64)
        assert torch.equal(bits(got), bits(one[start:])), start


@needs_device
def test_plans_are_kept_per_range_tensor_and_windows():
    """``run()`` builds the unit table of a (ranges, windows) triple once and reuses it; a new ``plan()`` drops the tables; a
    change of the caller's range tensor in place is seen (the table is keyed by the values)."""
    pr = span_problem(128, seed=34)
    ranges = span_rows(pr, SPANS)
    w = make_wrapper(pr)
    kv = pr.cache("NHD")
    d = w._batch_invariant
    a = w.run(pr.q, kv, ranges)
    assert len(d._range_plans) == 1
    first = next(iter(d._range_plans.values()))
    assert torch.equal(bits(w.run(pr.q, kv, ranges.clone())), bits(a))
    assert len(d._range_plans) == 1 and next(iter(d._range_plans.values())) is first
    b = w.run(pr.q, kv, ranges, causal_window_left=100)
    assert len(d._range_plans) == 2 and not torch.equal(bits(a), bits(b))
    changed = ranges.clone()
    changed[:] = -1
    ranges.copy_(changed)
    assert torch.equal(bits(w.run(pr.q, kv, ranges)), bits(run_causal(pr)))
    assert len(d._range_plans) == 3
    for i in range(12):
        w.run(pr.q, kv, ranges, causal_window_left=1000 + i)
    from flashinfer.attention.batch_invariant.wrappers import RANGE_PLANS_KEPT

    assert len(d._range_plans) == RANGE_PLANS_KEPT
    w.plan(
        pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, pr.Hq, pr.Hkv, 128, 16
    )
    assert len(d._range_plans) == 0
    assert torch.equal(bits(w.run(pr.q, kv, ranges)), bits(run_causal(pr)))


@needs_device
def test_small_workspace_falls_back_to_fused_units():
    """A workspace too small for the split partials plans fused units: the same bits; ``workspace_size`` bounds the need for
    any ranges."""
    pr = Problem([1, 1], [9000, 7000], 32, 8, 128, 16, seed=35)
    ranges = torch.tensor([[100, 8999], [-1, -1]], dtype=torch.int32, device=pr.device)
    big = run_ranges(pr, ranges)
    small = run_ranges(pr, ranges, ws=workspace(pr.device, mb=1)[:4096])
    assert torch.equal(bits(big), bits(small))
    w = make_wrapper(pr)
    need, _ = w.workspace_size(
        pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, pr.Hq, pr.Hkv, 128, 16
    )
    w.run(pr.q, pr.cache("NHD"), ranges)
    planned = next(iter(w._batch_invariant._range_plans.values()))
    assert planned.plan.ws_rows > 0
    assert planned.plan.ws_rows * planned.plan.ws_row * 4 <= need


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    E = bi.BatchInvariantUnsupportedError
    pr = Problem([64], [200], 8, 4, 128, 16, seed=36)
    ws = workspace(pr.device, mb=8)
    W = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper
    ranges = span_rows(pr, [[(150, 180)]])
    for kw in (
        dict(kv_data_type=torch.uint8),  # packed NVFP4
        dict(kv_data_type=torch.float8_e4m3fn),
        dict(q_data_type=torch.float16, kv_data_type=torch.float16),
        dict(o_data_type=torch.float16),
        dict(head_dim_qk=512, head_dim_vo=512),
        dict(head_dim_qk=128, head_dim_vo=64),
        dict(kv_data_type=torch.uint8, v_sf_swizzled=True),
        dict(pos_encoding_mode="ROPE_LLAMA"),
        dict(use_cuda_graph=True),
        dict(kv_layout="XYZ"),
        dict(qo_indptr_buf=torch.zeros(4, dtype=torch.int32, device=pr.device)),
    ):
        with pytest.raises(E):
            W(ws, batch_invariant=True, **kw)
    plan_args = (
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
    )
    w = W(ws, batch_invariant=True)
    with pytest.raises(RuntimeError):
        w.run(pr.q, pr.cache("NHD"), ranges)
    for kw in (
        dict(fixed_split_size=4096),
        dict(disable_split_kv=True),
        dict(sm_scale=float("nan")),
        dict(sm_scale=-1.0),
    ):
        with pytest.raises(E):
            w.plan(*plan_args, **kw)
    # the wrapper's own argument contract is unchanged under the mode
    for kw in (
        dict(causal=True),
        dict(window_left=5),
        dict(logits_soft_cap=30.0),
        dict(kv_data_type=torch.float16),
        dict(head_dim_vo=64),
    ):
        with pytest.raises(ValueError):
            w.plan(*plan_args, **kw)
    with pytest.raises(E):  # more query tokens than keys
        w.plan(
            torch.tensor([0, 300], dtype=torch.int32),
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            pr.Hq,
            pr.Hkv,
            pr.D,
            pr.page,
        )
    w.plan(*plan_args)
    kv = pr.cache("NHD")
    good = w.run(pr.q, kv, ranges)
    for kw in (
        dict(return_lse=True),
        dict(lse=torch.empty(64, 8, device=pr.device)),
        dict(q_scale=1.0),
        dict(k_scale=0.5),
        dict(v_scale=2.0),
        dict(kv_cache_sf=torch.zeros(1, dtype=torch.uint8, device=pr.device)),
        dict(out=torch.empty(64, 8, 128, dtype=torch.float16, device=pr.device)),
    ):
        with pytest.raises(E):
            w.run(pr.q, kv, ranges, **kw)
    with pytest.raises(E):
        w.run_return_lse(pr.q, kv, ranges)
    with pytest.raises(E):
        w.run(pr.q[:32], kv, ranges[:32].contiguous())
    with pytest.raises(E):
        w.run(pr.q, pr.cache("HND"), ranges)
    with pytest.raises(ValueError):  # the wrapper's own range-tensor contract
        w.run(pr.q, kv, ranges.to(torch.int64))
    with pytest.raises(ValueError):
        w.run(pr.q, kv, ranges[:32].contiguous())
    with pytest.raises(ValueError):
        w.run(pr.q, kv, ranges.cpu())
    # run() reads the ranges back to the host: it cannot be captured in a CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(torch.cuda.Stream()):
        graph.capture_begin()
        try:
            with pytest.raises(E):
                w.run(pr.q, kv, ranges)
        finally:
            graph.capture_end()
    torch.cuda.synchronize()
    # a refused call leaves the plan usable
    assert torch.equal(bits(w.run(pr.q, kv, ranges)), bits(good))
    assert torch.equal(bits(w.forward(pr.q, kv, ranges)), bits(good))
