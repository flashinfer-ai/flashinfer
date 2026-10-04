"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

QSA kernels: pre-indexer, scorer, route expansion and output gate.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import flashinfer.qsa_ops as qsa
from flashinfer.utils import get_compute_capability

DEV = "cuda"
BF16, F16, FP8, I64 = torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.long
INT = torch.int32
I32 = 2**31 - 1
PAST_I32 = [2**31, 2**32, 2**63 - 1, -(2**31) - 1]


def _need_sm80():
    if get_compute_capability(torch.device(DEV))[0] < 8:
        pytest.skip("the scorer needs SM80")


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def Z(*shape, dtype=INT):
    return torch.zeros(shape, dtype=dtype, device=DEV)


def _exact(got, want):
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


# --- pre-indexer -------------------------------------------------------------


def _rope_rows(x, w, table, pos3, mrope, eps, dtype):
    """Norm with a stored-plus-one weight, then a partial NeoX rope, in float64."""
    n = x.shape[-1] // 4
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (w.double() + 1)
    x = x.to(dtype).double()
    i = torch.arange(n, device=DEV)
    axis = torch.where((i % 3 > 0) & (i <= 33), i % 3, 0) if mrope else 0 * i
    p = pos3.gather(1, axis.expand(len(x), n)).clamp(max=len(table) - 1)
    cos, sin = table[p, 2 * i].double(), table[p, 2 * i + 1].double()
    lo, hi = x[:, :n].clone(), x[:, n : 2 * n].clone()
    x[:, :n], x[:, n : 2 * n] = lo * cos - hi * sin, hi * cos + lo * sin
    return x


def _pre_ref(c):
    T, H, D, r, R = len(c.q), c.H, c.D, c.ratio, c.R
    pos3 = c.positions.T if c.mrope else c.positions[:, None].expand(T, 3)
    rope = lambda x, w, p, m: _rope_rows(x, w, c.table, p, m, c.eps, c.dtype)
    x, pos3_q = c.q.view(T * H, D).double(), pos3.repeat_interleave(H, 0)
    q_out = rope(x, c.qw, pos3_q, c.mrope)
    ring, comp, old = c.state.clone(), c.comp.clone(), c.state
    flat = ring.view(-1, ring.shape[-1])
    qo, lp = c.qo_indptr.tolist(), c.logical.tolist()
    for req, g in c.work.tolist():
        s, e = qo[req : req + 2] if 0 <= req < len(c.ring_table) else (0, 0)
        if s < 0 or e > T or e <= s:
            continue
        start, blk = lp[e - 1] - (e - s) + 1, int(c.ring_table[req, 0])
        last = (start + r) // r * r - 1 + g * r
        slot = int(c.comp_slots[s + last - start]) if last <= lp[e - 1] else -1
        if 0 <= slot < comp.shape[0] * comp.shape[1]:
            key = lambda p: c.k[s + p - start] if p >= start else old[blk, p % R, 0, :D]
            acc = sum(key(p).double() for p in range(last - r + 1, last + 1)) / r
            p3, f = [last - r + 1] * 3, last - r + 1
            if c.cache_pos:
                p3 = pos3[s + f - start] if f >= start else old[blk, f % R, 0, D:]
                p3 = p3.view(torch.int64)[:3] if f < start else p3
            p3 = torch.as_tensor(p3, device=DEV)[None]
            row = rope(acc.to(c.dtype).double()[None], c.kw, p3, c.mrope_k)
            comp.view(-1, D)[slot] = row[0].to(comp.dtype)
        for tok in range(e - min(e - s, R), e) if g == 0 else ():
            if 0 <= (slot := int(c.ring_slots[tok])) < len(flat):
                flat[slot, :D] = c.k[tok]
                if c.cache_pos:
                    flat[slot, D:].view(torch.int64)[:3] = pos3[tok]
    return q_out.view(T, H, D).to(c.out), ring, comp


def _pcase(
    T=16, H=4, D=128, ratio=4, R=8, dtype=BF16, mrope=0, cache_pos=0, split=1, history=0
):
    """`split` requests (a count, or their qo_indptr) over a ring of R rows each."""
    torch.manual_seed(0)
    if not isinstance(split, list):
        split = [T // split * i for i in range(split)] + [T]
    lens = [b - a for a, b in zip(split, split[1:], strict=False)]
    off = R * history
    logical = torch.cat([torch.arange(off, off + n) for n in lens]).to(DEV)
    req = torch.cat([torch.full((n,), i) for i, n in enumerate(lens)]).to(DEV)
    three_axis = torch.stack([logical, logical // 7, logical // 13])
    positions = three_axis if mrope else logical
    q = torch.randn(T, H * D, dtype=dtype, device=DEV)
    k = torch.randn(T, D, dtype=dtype, device=DEV)
    qw, kw = torch.randn(2, D, dtype=dtype, device=DEV) * 0.2
    theta = torch.rand(512, D // 4) * 6
    table = torch.stack([theta.cos(), theta.sin()], -1).flatten(1).to(DEV, dtype)
    state = torch.randn(len(lens), R, 1, D + 12 * cache_pos, dtype=dtype, device=DEV)
    if cache_pos:  # each ring row keeps its rotary coordinates after it
        i = torch.arange(R, device=DEV)[:, None]
        state[..., D:].view(I64)[..., :3] = torch.cat([i, i // 7, i // 13], 1)[:, None]
    ring_slots = req * R + logical % R
    ring_table = torch.arange(len(lens), dtype=INT, device=DEV)[:, None]
    qo_indptr = torch.tensor(split, dtype=INT, device=DEV)
    per = (off + max(lens)) // ratio + 1  # the compressed slots each request owns
    blocks = -(-len(lens) * per // ratio)
    comp = torch.zeros(blocks, ratio, 1, D, dtype=dtype, device=DEV)
    comp_slots = (req * per + logical // ratio).clamp(max=comp[..., 0, 0].numel() - 1)
    groups = [max(1, (off + n) // ratio - off // ratio) for n in lens]
    work = [[i, g] for i, n in enumerate(groups) for g in range(n)]
    work = torch.tensor(work, dtype=INT, device=DEV)
    q_out = torch.zeros(T, H, D, dtype=dtype, device=DEV)
    eps, out, mrope_h = 1e-6, dtype, 11
    mrope_k, cache_pos = bool(mrope), bool(cache_pos)
    return SimpleNamespace(**locals())


_PRE_ARGS = (  # noqa: SIM905
    "q k positions table qw kw eps q_out state ring_slots ring_table qo_indptr "
    "logical comp comp_slots work ratio mrope_h mrope_h mrope_k cache_pos"
).split()


def _pre_run(c, outs=None):
    """Run into copies of the case's outputs, or into `outs`."""
    outs = outs or [getattr(c, n).clone() for n in ("q_out", "state", "comp")]
    env = {**vars(c), **dict(zip(("q_out", "state", "comp"), outs, strict=True))}
    qsa.qsa_pre_indexer(*[env[n] for n in _PRE_ARGS])
    return outs


def _pre_check(c, got=None):
    got = got or _pre_run(c)
    assert got[1].dtype == c.dtype  # the ring is never narrowed
    for a, b in zip(got, _pre_ref(c), strict=True):
        if a.dtype != FP8:
            torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)
        else:  # one e4m3 code apart, or within its denormal step
            codes = (a.view(torch.uint8).int() - b.view(torch.uint8).int()).abs()
            assert ((codes <= 1) | ((a.float() - b.float()).abs() <= 2**-8)).all()


def _put(**f):
    return lambda c: [setattr(c, n, v(c) if callable(v) else v) for n, v in f.items()]


_with = lambda name, fn: lambda c: setattr(c, name, fn(getattr(c, name)).contiguous())
_poke = lambda name, at, value: lambda c: getattr(c, name).__setitem__(at, value)


def _narrow(dt):
    return _put(out=dt, q_out=lambda c: c.q_out.to(dt), comp=lambda c: c.comp.to(dt))


_PRE_CASES = {
    "fp16-d256-h8": (dict(dtype=F16, D=256, H=8), None),
    "d256-h1": (dict(D=256, H=1), None),
    **{f"tokens{t}": (dict(T=t), None) for t in (1, 3, 9, 33, 129)},
    **{
        f"ratio{r}-history{h}": (dict(T=32, ratio=r, history=h), None)
        for r in (2, 4, 8)
        for h in (0, 1)
    },
    "mrope": (dict(T=32, mrope=1, history=1), None),
    "mrope-ring-coords": (dict(T=32, mrope=1, cache_pos=1, history=1), None),
    "key-mrope-one-axis-positions": ({}, _put(mrope_k=True)),
    "requests4": (dict(T=32, split=4, history=1), None),
    "non-pow2": (dict(T=24, ratio=3, R=6), None),
    "empty-request": (dict(split=[0, 0, 16]), None),
    **{
        f"ring{R}-ratio{r}": (dict(T=32, R=R, ratio=r, history=1, cache_pos=1), None)
        for R, r in ((7, 4), (6, 4), (5, 2))
    },
    "fp8": ({}, _narrow(FP8)),
    "fp8-fp16-d256": (dict(dtype=F16, D=256), _narrow(FP8)),
    "dead-work-item": ({}, _poke("work", (0, 0), -1)),
    "missing-request": ({}, _poke("work", (0, 0), 1)),
    "past-the-end": ({}, _poke("qo_indptr", 1, 64)),
    "reversed": ({}, _poke("qo_indptr", slice(None), torch.tensor([8, 4]))),
    "negative": ({}, _poke("qo_indptr", 0, -8)),
    "wraps": ({}, _poke("qo_indptr", slice(None), torch.tensor([I32, -(2**31)]))),
    "coordinate-past-table": (
        dict(mrope=1),
        _poke("positions", ([1, 2, 0], [0, 1, 2]), 999),
    ),
}


@pytest.mark.parametrize("name", _PRE_CASES)
def test_pre_indexer_matches_reference(name):
    kwargs, mutate = _PRE_CASES[name]
    c = _pcase(**kwargs)
    if mutate:
        mutate(c)
    _pre_check(c)


def test_pre_indexer_graph_replay_and_dispatch():
    mask = qsa.qsa_pre_indexer_dispatch_mask()
    assert mask == qsa.QSA_PRE_INDEXER_SAME_AS_COMPUTE | qsa.QSA_PRE_INDEXER_NARROW_E4M3
    c = _pcase(T=32, history=1)
    outs = [c.q_out, c.state.clone(), c.comp.clone()]
    graph = _capture(lambda: _pre_run(c, outs))
    for _ in range(2):
        c.q.normal_()
        outs[1].copy_(c.state)
        outs[2].copy_(c.comp)
        graph.replay()
        _pre_check(c, outs)
    before = [t.clone() for t in outs]
    _put(**{n: getattr(c, n)[:0] for n in ("q", "k", "positions", "ring_slots")})(c)
    _pre_run(c, outs)  # no tokens is a no-op
    _exact(outs, before)


@pytest.mark.parametrize(
    "fill,low,high",
    [(1e-2, 0.9, 1.5), (1.0, 0.9, 1.5), (1e18, 0.9, 1.5), (1e-20, 0.9e-17, 2e-17)]
    + [(1e19, 0, 0), (float("inf"), None, None), (float("nan"), None, None)],
)
def test_pre_indexer_norm_edges(fill, low, high):
    """A constant row normalises to one; its float32 sum of squares overflows at 1e19."""
    c = _pcase(T=8, H=1)
    c.q.fill_(fill)
    c.qw.zero_()
    out = _pre_run(c)[0][0, 0].float()
    assert out.isnan().all() if low is None else low <= out.abs().max() <= high
    if fill == 1.0:  # the weight is the offset from one, so -1 is a zero gain
        c.qw.fill_(-1)
        assert _pre_run(c)[0].abs().max() == 0


@pytest.mark.parametrize(
    "kwargs,mutate,match",
    [
        (dict(mrope=1), _put(mrope_k=False), "three-axis"),
        ({}, _with("q_out", lambda v: v[:, :1, :64]), "head_dim"),
        ({}, _with("table", lambda v: v[:, :32]), "pair-major"),
        ({}, _with("state", lambda v: v.repeat(1, 1, 2, 1)), "one KV head"),
        (dict(cache_pos=1), _with("state", lambda v: v[..., :132]), "coordinates"),
        ({}, _with("qo_indptr", lambda v: v[:-1]), "one entry past"),
        ({}, _with("qw", lambda v: v[:-8]), "one weight per feature"),
        ({}, _with("kw", lambda v: v[:-8]), "one weight per feature"),
        ({}, _with("ring_table", lambda v: v[:, :0]), "one ring block"),
        ({}, _with("ring_table", lambda v: v.repeat(1, 2)), "one ring block"),
        ({}, _with("k", lambda v: v[:-1]), "per token"),
        ({}, _with("positions", lambda v: v[:-1]), "per token"),
        ({}, _with("state", lambda v: v[:, :0]), "at least one row"),
        ({}, _with("comp", lambda v: v[:, :0]), "at least one row"),
        ({}, _with("positions", lambda v: v[0]), "one axis or three"),
        ({}, _with("table", lambda v: v[0, 0]), "at least one axis"),
        ({}, _with("table", lambda v: v[:0]), "at least one row"),
        ({}, _put(ratio=2**32 + 1), "fit in 32 bits"),
        ({}, _with("q_out", lambda v: v.to(FP8)), r"compressed_cache\.dtype"),
        ({}, _narrow(F16), "or float8_e4m3fn"),
        ({}, _narrow(torch.float8_e5m2), "or float8_e4m3fn"),
        ({}, _narrow(torch.float8_e4m3fnuz), "or float8_e4m3fn"),
    ],
)
def test_pre_indexer_refuses(kwargs, mutate, match):
    c = _pcase(T=8, **kwargs)
    mutate(c)
    with pytest.raises(Exception, match=match):
        _pre_run(c)


# --- scorer ------------------------------------------------------------------


def _scores_ref(q, k, table, t2r, pos, lens, cols=None, fill=-7.0, ratio=1):
    cols = cols or table.shape[1] * k.shape[1]
    logits = torch.full((len(q), cols), fill, device=DEV)
    visible = torch.zeros(len(q), dtype=table.dtype, device=DEV)
    pages, size = k.shape[:2]
    for row in range(len(q)):
        r, p = int(t2r[row]), int(pos[row])
        if not (0 <= r < len(table) and 0 <= p <= I32):
            continue
        seen = visible[row] = min((p + 1) // ratio, int(lens[r]) // ratio, cols)
        col = torch.arange(seen, device=DEV)
        page = table[r].long()[col // size]
        score = torch.full((seen,), -torch.inf, device=DEV)
        if pages:
            keys = k[page.clamp(0, pages - 1), col % size].float()
            dots = (q[row].float() @ keys.T).clamp(min=0).sum(0) / q.shape[2] ** 0.5
            score = torch.where((page >= 0) & (page < pages), dots, score)
        logits[row, :seen] = score
    return logits, visible


def _scase(D=128, H=4, rows=8, pages=6, dtype=BF16, idx=torch.int32):
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn(rows, H, D, dtype=dtype, device=DEV, generator=g)
    k = torch.randn(pages, 8, D, dtype=dtype, device=DEV, generator=g)
    i, n = torch.arange(max(rows, pages), device=DEV).to(idx), 8 * pages
    return [q, k, i[None, :pages], 0 * i[:rows], i[:rows] + n, 0 * i[:1] + n]


def _check_scores(q, k, table, t2r, pos, lens, cols=None):
    """Columns past what a row sees keep the caller's -7."""
    want = _scores_ref(q, k, table, t2r, pos, lens, cols)
    logits = torch.full_like(want[0], -7.0)
    args = (q, k, table, t2r, pos, lens, 1, q.shape[2] ** 0.5)
    got = qsa.qsa_paged_scores(*args, num_columns=logits.shape[1], logits=logits)
    torch.testing.assert_close(got[1], want[1], rtol=0, atol=0)
    torch.testing.assert_close(got[0], want[0], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [BF16, F16])
@pytest.mark.parametrize("D", [64, 128, 192, 256])
@pytest.mark.parametrize("H", [1, 4, 8, 9, 16])  # 8 and 9 straddle the two mma widths
def test_scores_match_reference(dtype, D, H):
    _need_sm80()
    _check_scores(*_scase(D, H, dtype=dtype))


def _edit(index, at, value):
    def edit(case):
        case[index] = case[index].clone()
        case[index][at] = value

    return edit


def _offset(index):  # a view whose base is not 16-byte aligned
    def edit(case):
        t = case[index]
        case[index] = torch.empty(
            *t.shape[:-1], t.shape[-1] + 8, dtype=t.dtype, device=DEV
        )
        case[index][..., 1:-7] = t
        case[index] = case[index][..., 1:-7]

    return edit


_SCORE_CASES = {
    "rows1": (dict(rows=1),),
    "rows96": (dict(rows=96, pages=8),),
    "pipelined": (dict(rows=-1, pages=64),),  # rows set from the SM count
    "unmapped-page": ({}, _edit(2, (0, 1), -1)),
    "row-without-request": ({}, _edit(3, 0, -1)),
    "narrow": ({}, lambda c: c.append(8)),
    "rows-see-less": ({}, _edit(4, slice(None), torch.arange(8))),
    "int32-max": (dict(idx=I64), _edit(4, 0, I32), _edit(5, 0, I32)),
    "no-pages": ({}, lambda c: c.__setitem__(1, c[1][:0])),
    "q-offset": ({}, _offset(0)),
    "k-offset": ({}, _offset(1)),
    **{
        f"{name}-past-int32-{bad}": (dict(idx=I64), _edit(i, at, bad))
        for bad in PAST_I32
        for name, i, at in (("table", 2, (0, 0)), ("t2r", 3, 0), ("pos", 4, 0))
    },
}


@pytest.mark.parametrize("name", _SCORE_CASES)
def test_scores_cases(name):
    _need_sm80()
    kwargs, *edits = _SCORE_CASES[name]
    if kwargs.get("rows") == -1:  # past 24 blocks per SM: eight tiles per block
        sms = torch.cuda.get_device_properties(DEV).multi_processor_count
        kwargs = {**kwargs, "rows": 4 * sms}
    case = _scase(**kwargs)
    for edit in edits:
        edit(case)
    _check_scores(*case)


@pytest.mark.parametrize(
    "index,fn,kwargs,match",
    [
        (1, lambda k: k[:, :0], dict(num_columns=48), "a page holds at least one"),
        (0, None, dict(num_columns=8, logits=Z(8, 12, dtype=torch.float)), "wide"),
        (0, None, dict(compress_ratio=2**32), "fit in 32 bits"),
        (3, lambda t: t[:, None].repeat(1, 2), {}, "has one axis"),
        (4, lambda t: t[:, None].repeat(1, 2), {}, "has one axis"),
        (5, lambda t: t[:, None].repeat(1, 2), {}, "has one axis"),
        (0, None, dict(visible_blocks=Z(8, 2)), "has one axis"),
    ],
)
def test_scores_refuse(index, fn, kwargs, match):
    _need_sm80()
    case = _scase()
    case[index] = (fn or (lambda t: t))(case[index])
    kwargs = {"compress_ratio": 1, "divisor": 1.0, **kwargs}
    with pytest.raises(Exception, match=match):
        qsa.qsa_paged_scores(*case, **kwargs)


# --- route -------------------------------------------------------------------


def _expand_ref(blocks, pos, lens, t2r, cr):
    ok = lambda v: 0 <= v <= I32
    lens = lens.tolist()
    out = torch.full(
        (len(blocks), blocks.shape[1] * cr + cr - 1), -1, dtype=blocks.dtype
    )
    rows = zip(blocks.tolist(), pos.tolist(), t2r.tolist(), strict=True)
    for row, (bl, p, r) in enumerate(rows):
        seq = lens[r] if ok(r) and r < len(lens) and ok(lens[r]) else 0
        p = p if ok(p) else -1
        past = min((p + 1) // cr, seq // cr)  # the blocks wholly behind the query
        toks = [b * cr + o if ok(b) and b < past else -1 for b in bl for o in range(cr)]
        start = (p + 1) // cr * cr  # the query's own block, up to the query
        toks += range(start, start + min(p + 1 - start, cr - 1))
        for col, t in enumerate(toks):
            if 0 <= t <= p and t < seq:
                out[row, col] = t
    return out.to(DEV)


def _slot_ref(logical, t2r, table, page, slots, valid):
    nbytes = -(-logical.shape[1] // 8)
    route = torch.zeros_like(logical, device="cpu")
    mask = torch.zeros(len(logical) * nbytes, dtype=torch.uint8)
    tab, reqs = table.tolist(), t2r.tolist()
    for row, toks in enumerate(logical.tolist()[:valid]):
        r, ok = reqs[row], torch.zeros(len(toks), dtype=torch.bool)
        for col, t in enumerate(toks if 0 <= r < len(tab) else ()):
            mapped = tab[r][t // page] if t >= 0 and t // page < len(tab[r]) else -1
            if mapped >= 0 and mapped * page + t % page < slots:
                route[row, col], ok[col] = mapped * page + t % page, True
                mask[row * nbytes + col // 8] |= 1 << col % 8
        if ok.any():  # invalid entries read the row's first valid entry
            route[row][~ok] = route[row][ok][0]
    return route.to(DEV), mask.to(DEV)


def _check_route(
    blocks, pos, lens, t2r, table, ratio, page, slots, valid=None, logical=None
):
    """Both entry points against the reference, in every output they write."""
    rows, valid = len(blocks), len(blocks) if valid is None else valid
    want = _expand_ref(blocks, pos, lens, t2r, ratio)
    logical = want if logical is None else logical
    mask = torch.empty(rows * -(-want.shape[1] // 8), dtype=torch.uint8, device=DEV)
    two_step = [torch.empty_like(want), mask]
    indptr = torch.full((rows + 1,), -1, dtype=INT, device=DEV)
    qsa.qsa_route_from_logical(
        logical, t2r, table, *two_step, valid, page, slots, out_indptr=indptr
    )
    got = [qsa.qsa_expand_block_route(blocks, pos, lens, t2r, ratio), *two_step]
    ref = lambda lg, n: _slot_ref(lg, t2r, table, page, slots, n)
    rows_in = (
        torch.arange(rows + 1, device=DEV).clamp(max=valid) * want.shape[1]
    ).int()
    _exact(got + [indptr], [want, *ref(logical, valid), rows_in])


def _rcase(rows, topk, cr, page=16, seq=512, nreq=4, valid=None, fill=None, t2r=None):
    g = torch.Generator(device=DEV).manual_seed(rows + topk)
    rand = lambda hi, *shape: torch.randint(0, hi, shape, device=DEV, generator=g)
    pages = -(-seq // page)
    table = torch.randperm(pages * nreq, device=DEV, generator=g).view(nreq, pages)
    table[:, ::7] = -1  # unmapped pages, the first among them
    if fill is not None:
        table.fill_(fill)
    t2r = torch.tensor(t2r, device=DEV) if t2r else rand(nreq, rows)
    lens = torch.full((nreq,), seq, device=DEV)
    args = [rand(max(1, seq // cr), rows, topk), rand(seq, rows), lens, t2r, table]
    return [a.int() for a in args] + [cr, page, pages * nreq * page, valid]


def _one(blocks, pos, lens=16, ratio=4, table=None, page=4, slots=None, dtype=INT):
    t = lambda v: torch.tensor(v, dtype=dtype, device=DEV)
    table = t(table or [list(range(8))])
    return [t([blocks]), t([pos]), t([lens]), t([0]), table, ratio, page, slots or 32]


def _past_i32(index, bad):
    case = _one([0, 1, 2, 3], 31, 512, table=[[0] * 32], page=16, slots=512, dtype=I64)
    case[index][(0, 0)[: case[index].ndim]] = bad
    return case


def _grid(rows):  # past the 65535 rows a grid's y dimension holds
    z = torch.zeros(rows, dtype=INT, device=DEV)
    table = torch.arange(8, dtype=INT, device=DEV)[None]
    return [z[:, None].repeat(1, 2), z + 31, z[:1] + 64, z, table, 4, 8, 64]


_ROUTE_CASES = {
    **{
        f"ratio{r}-rows{n}-topk{k}-page{p}": lambda a=(n, k, r, p): _rcase(*a)
        for r, n, k, p in [(1, 1, 1, 1), (2, 7, 16, 16), (4, 64, 128, 64)]
        + [(8, 300, 16, 16), (2, 128, 32, 1), (4, 9, 32, 64)]
    },
    "padding-rows": lambda: _rcase(128, 32, 4, 64, seq=1024, valid=100),
    "no-valid-rows": lambda: _rcase(64, 32, 4, seq=1024, valid=0),
    "unmapped": lambda: _rcase(32, 16, 4, seq=256, nreq=2, fill=-1),
    "mapped-past-cache": lambda: _rcase(32, 16, 4, seq=256, nreq=2, fill=32),
    "row-without-request": lambda: _rcase(2, 4, 4, t2r=[0, -1]),
    "block-ahead-of-query": lambda: _one([2, 0, 0, 0], 3, 512),
    "own-block": lambda: _one([2], 9, 512),
    "every-rank": lambda: _one([-1, -1, 0, 1], 7),
    # The only valid entries are the tail, past every block's first pass, on page 1.
    "tail-only": lambda: _one([-1] * 128, 6, 512, table=[[-1, 5, 1, 2, 3, 4, 6, 7]]),
    "pos-int32-max": lambda: _one([0, 1, 2, 3], I32, 512, dtype=I64),
    "len-int32-max": lambda: _one([0, 1, 2, 3], 31, I32, dtype=I64),
    "slot-wraps": lambda: _one([0] * 4, 7, 8, 1, [[2**16]], 2**16, 2**32 - 1, I64),
    "int64-page": lambda: _one([0] * 4, 7, 8, 1, [[2**31] * 8], 1, 2**31 + 1, I64),
    "rows-past-grid": lambda: _grid(70000),
    **{
        f"{name}-past-int32-{bad}": lambda a=(i, bad): _past_i32(*a)
        for bad in PAST_I32
        for i, name in enumerate(["blocks", "pos", "lens", "t2r", "table"])
    },
}


@pytest.mark.parametrize("name", _ROUTE_CASES)
def test_route_matches_reference(name):
    _check_route(*_ROUTE_CASES[name]())


@pytest.mark.parametrize("bad", PAST_I32)
def test_route_from_logical_does_not_wrap_a_token(bad):
    logical = torch.arange(19, dtype=I64, device=DEV)[None]
    logical[0, 0] = bad
    _check_route(*_past_i32(0, 0), logical=logical)


def test_route_from_logical_writes_the_row_pointers_of_an_empty_step():
    z = lambda *shape: torch.zeros(shape, dtype=INT, device=DEV)
    indptr = z(1) - 1  # a sentinel the call has to overwrite
    route, mask = z(0, 4), z(0).byte()
    qsa.qsa_route_from_logical(
        route, z(0), z(1, 1), route, mask, 0, 16, 256, out_indptr=indptr
    )
    _exact([indptr], [z(1)])


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda a, lg: qsa.qsa_expand_block_route(*a[:4], 0), "compress_ratio"),
        (lambda a, lg: qsa.qsa_expand_block_route(a[0][0], *a[1:4], 4), "2D"),
        (lambda a, lg: qsa.qsa_expand_block_route(*a[:4], 4, out=a[5]), "shape"),
        (lambda a, lg: qsa.qsa_expand_block_route(*a[:4], 3), "compress_ratio"),
        (lambda a, lg: qsa.qsa_route_from_logical(*lg, 1, 2**16, 2**31 + 1), "dtype"),
        (lambda a, lg: qsa.qsa_route_from_logical(*lg, 1, 16, 256, a[3]), "rows \\+ 1"),
    ],
)
def test_route_refuses(call, match):
    z = lambda *shape: torch.zeros(shape, dtype=INT, device=DEV)
    a = [z(1, 4), z(1), z(1) + 16, z(1), z(1, 1), z(1, 4), z(1, 4), z(1).byte()]
    lg = [a[5], a[3], a[4], *a[6:]]  # logical, token_to_request, block_table, outputs
    qsa.qsa_route_from_logical(*lg, 1, 2**16, 2**31)  # a slot space the route holds
    with pytest.raises(Exception, match=match):
        call(a, lg)


# --- output gate -------------------------------------------------------------


def _gate_ref(a, g):
    return (a.float() * torch.sigmoid(g.float())).to(a.dtype)


def _ulps(x, y):
    o = [t.view(torch.int16).long() for t in (x, y)]
    o = [torch.where(b < 0, -0x8000 - b, b) for b in o]
    return int((o[0] - o[1]).abs().max())


@pytest.mark.parametrize("dtype", [BF16, F16])
@pytest.mark.parametrize("shape", [(1, 1, 64), (7, 4, 100), (2048, 16, 128)])
@pytest.mark.parametrize("layout", ["plain", "padded", "in-place", "fused-gate"])
def test_gate_matches_the_chain_it_replaces(dtype, shape, layout):
    rows, heads, dim = shape
    pad = 32 * (layout == "padded")
    a = torch.randn(rows + pad, heads, dim, dtype=dtype, device=DEV)
    a[rows:] = 1000  # padding rows are never read
    fused = torch.randn(rows, 2, heads, dim, dtype=dtype, device=DEV)
    g = fused[:, 1] if layout == "fused-gate" else fused[:, 1].contiguous()
    want, before = _gate_ref(a[:rows], g), fused.clone()
    out = qsa.qsa_output_gate(a, g, out=a[:rows] if layout == "in-place" else None)
    assert out.shape == g.shape
    assert layout != "in-place" or out.data_ptr() == a.data_ptr()
    assert _ulps(out, want) <= 1
    _exact([fused], [before])


def test_gate_over_every_bfloat16():
    values = [0.0, -0.0, 2.0**-133, -(2.0**-133), 2.0**-126, 1.0, -1.0, 0.5, -3.75]
    values += [2.0**8, 2.0**64, -(2.0**64), 2.0**127, torch.inf, -torch.inf, torch.nan]
    g = torch.arange(1 << 16, dtype=INT, device=DEV).short().view(BF16)
    g = g.expand(len(values), 1, -1).contiguous()
    a = torch.tensor(values, dtype=BF16, device=DEV)[:, None, None].expand_as(g)
    out, want = qsa.qsa_output_gate(a.contiguous(), g), _gate_ref(a, g)
    real, fin = a.isfinite() & g.isfinite(), g[0, 0].isfinite()
    assert _ulps(out[real], want[real]) <= 1
    _exact([out[:2, :, fin].view(torch.int16)], [want[:2, :, fin].view(torch.int16)])
    assert (out[12][(g[12] > -100) & (g[12] < -88)] != 0).any()  # no flush to zero
    assert out[15].isnan().all() and out[..., g[0, 0].isnan()].isnan().all()
    zero = torch.sigmoid(g.float()) == 0
    for row in (13, 14):  # inf * sigmoid: NaN where the logistic is zero
        assert out[row][zero[row] & g[row].isfinite()].isnan().all()
        live = out[row][~zero[row] & ~g[row].isnan()]
        assert live.isinf().all() and ((live > 0) == (values[row] > 0)).all()
    rows = a[:, 0, 0].isfinite()
    _exact([out[rows][..., g[0, 0] == torch.inf]], [a[rows][..., g[0, 0] == torch.inf]])
    assert (out[rows][..., g[0, 0] == -torch.inf] == 0).all()
    assert (out[5][g[5] > 20] == 1).all() and (out[5][g[5] < -200] == 0).all()


def test_gate_under_graph_capture_and_its_rounding_order():
    a, g = torch.randn(2, 512, 16, 128, dtype=BF16, device=DEV)
    out = qsa.qsa_output_gate(a, g)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    qsa.qsa_output_gate(a, g, out=out)
    assert torch.cuda.max_memory_allocated() == before
    eager = out.clone()
    graph = _capture(lambda: qsa.qsa_output_gate(a, g, out=out))
    out.zero_()
    graph.replay()
    _exact([out], [eager])
    # The logistic is not rounded to the output dtype before the product.
    exact = a.double() * torch.sigmoid(g.double())
    shortcut = (a * torch.sigmoid(g)).double()
    assert (out.double() - exact).abs().sum() < (shortcut - exact).abs().sum()


def test_gate_refuses_aliasing_and_short_attention():
    a, both, g = (torch.randn(n, 4, 64, dtype=BF16, device=DEV) for n in (8, 10, 8))
    shifted = both.flatten()[1 : 1 + 8 * 256].view(8, 4, 64)
    for attention, gate, out, match in [
        (a[:4], g, None, "at least the output's rows"),
        (a, both[:8], both[:8], "overlap"),
        (a, both[1:9], both[:8], "overlap"),
        (both[1:9], g, both[:8], "attention itself"),
        (both[:8], g, shifted, "attention itself"),
    ]:
        with pytest.raises(RuntimeError, match=match):
            qsa.qsa_output_gate(attention, gate, out=out)
    qsa.qsa_output_gate(both, g, out=both[:8])  # the padded prefix, in place
    for shape in ((4, 0, 64), (4, 2, 0), (0, 2, 64)):
        empty = torch.empty(shape, dtype=BF16, device=DEV)
        assert qsa.qsa_output_gate(empty, empty.clone()).shape == shape


# --- build -------------------------------------------------------------------


@pytest.mark.parametrize(
    "archs,misc,expected",
    [({(7, "5")}, 1, 0), ({(8, "0")}, 1, 1), ({(9, "0a")}, 1, 1), ({(10, "0a")}, 1, 1)]
    + [({(12, "0f")}, 1, 1), ({(7, "5"), (9, "0a")}, 1, 1), ({(8, "0")}, 0, 0)],
)
def test_aot_registers_the_qsa_modules_together(monkeypatch, archs, misc, expected):
    """Scorer, route and gate need SM8+ and go in together; the pre-indexer always."""
    from flashinfer import aot
    from flashinfer.jit import core, qsa_ops

    fast_math = lambda spec: [f for f in spec.extra_cuda_cflags if "fast_math" in f]
    assert fast_math(qsa_ops.gen_qsa_route_module())
    assert not fast_math(qsa_ops.gen_qsa_output_gate_module())
    context = SimpleNamespace(TARGET_CUDA_ARCHS=archs)
    monkeypatch.setattr(core, "current_compilation_context", context)
    names = ("qsa_output_gate", "qsa_route", "qsa_scores", "qsa_pre_indexer")
    stub = lambda name: lambda *args: SimpleNamespace(name=name)
    for name in (*names, "spdlog", "cudnn_fmha"):
        monkeypatch.setattr(aot, f"gen_{name}_module", stub(name))
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    built = [s.name for s in aot.gen_all_modules(*[[]] * 6, {}, *[0] * 5, misc, 0)]
    assert [built.count(n) for n in names] == [expected] * 3 + [misc]


_CP_ASYNC_CALLS = """
#include <flashinfer/cp_async.cuh>
using namespace flashinfer::cp_async;
using P = PrefetchMode;
using F = SharedMemFillMode;
constexpr CacheMode ca = CacheMode::kCacheAll;
__global__ void calls(const float* g) {
  __shared__ float s[8];
  load_128b<P::kNoPrefetch, float>(s, g);
  load_128b<P::kPrefetch>(s, g);
  load_128b<P::kNoPrefetch, ca>(s, g);
  pred_load_128b<P::kNoPrefetch, F::kFillZero, float>(s, g, true);
  pred_load_128b<P::kPrefetch, F::kNoFill>(s, g, true);
  pred_load_128b<P::kNoPrefetch, F::kFillZero, ca>(s, g, true);
  load<256, P::kNoPrefetch, float>(s, g);
  load<128, P::kPrefetch>(s, g);
  load<256, P::kNoPrefetch, ca>(s, g);
  pred_load<256, P::kNoPrefetch, F::kFillZero, float>(s, g, true);
  pred_load<128, P::kPrefetch, F::kNoFill>(s, g, true);
  pred_load<256, P::kNoPrefetch, F::kFillZero, ca>(s, g, true);
}
"""


def test_cp_async_keeps_the_template_arguments_it_had(tmp_path):
    """Explicit-type, deduced and cache-mode calls of the four copies compile."""
    if os.environ.get("FLASHINFER_DISABLE_JIT"):
        pytest.skip("this compiles a kernel")
    from flashinfer.jit.core import gen_jit_spec

    (tmp_path / "calls.cu").write_text(_CP_ASYNC_CALLS)
    gen_jit_spec("qsa_cp_async_calls", [tmp_path / "calls.cu"]).build()
