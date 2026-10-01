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

QSA runtime: selection, paged attention, the step they make, and its capabilities.
"""

import gc
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

import flashinfer
import flashinfer.qsa_ops as qsa
from flashinfer.qsa_ops import capabilities

from .test_kernels import BF16, DEV, FP8, Z, _capture, _exact, _expand_ref, _need_sm80
from .test_kernels import INT, _scores_ref

RATIO, TOPK, QO, KV, D, PAGE, SLOTS = 4, 32, 4, 1, 128, 16, 512
WIDTH = TOPK + RATIO - 1  # the selected blocks, then the query's own block
U8 = torch.uint8


@pytest.fixture(autouse=True)
def _sm80():
    _need_sm80()


def _fail(*args, **kwargs):
    raise AssertionError("reached")


def _replay_on_streams(graphs):
    streams = [torch.cuda.Stream() for _ in graphs]
    for stream, graph in zip(streams, graphs, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)


# --- selection ---------------------------------------------------------------


def _batch(rows=8, nreq=2, seed=0, pos=None, length=4 * PAGE, unmapped=False):
    """Requests over the pages: an index cache, a KV cache and a route into it."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    pages = max(32, length // RATIO // PAGE)  # enough to index every visible column
    rand = lambda *shape: torch.randn(shape, dtype=BF16, device=DEV, generator=g)
    b = SimpleNamespace(
        table=torch.randperm(pages, device=DEV, generator=g).view(nreq, -1)
    )
    b.kc, b.k, b.v = (rand(pages, PAGE, *shape) for shape in ((D,), (KV, D), (KV, D)))
    b.q, b.gate, i = (
        rand(rows, QO, D),
        rand(rows, QO * D),
        torch.arange(rows, device=DEV),
    )
    b.table, b.t2r = (
        b.table.int(),
        (i // max(1, rows // nreq)).clamp(max=nreq - 1).int(),
    )
    if unmapped:
        b.table[0, 2:] = -1
    b.pos = (i % 256 if pos is None else 0 * i + pos).int()
    b.lens = 0 * b.table[:, 0] + length
    b.sel = [b.q, b.kc, b.table, b.t2r, b.pos, b.lens]
    b.route = Z(rows, WIDTH) - 1
    for row in range(rows):  # a live prefix of distinct tokens, then padding
        b.route[row, : WIDTH - row % 3] = torch.randperm(256, device=DEV, generator=g)[
            : WIDTH - row % 3
        ]
    return b


def _select_ref(q, kc, table, t2r, pos, lens, cols=64):
    """Score, keep the top blocks (the smaller index on a tie) and expand them."""
    scores, seen = _scores_ref(q, kc, table, t2r, pos, lens, cols, -torch.inf, RATIO)
    blocks = torch.full((len(q), TOPK // RATIO), -1, dtype=INT)
    for row, n in enumerate(seen.tolist()):
        top = torch.sort(-scores[row, :n], stable=True).indices[: TOPK // RATIO]
        blocks[row, : len(top)] = top
    return _expand_ref(blocks.to(DEV), pos, lens, t2r, RATIO)


def _same_route(got, want):
    """The top-k promises no order, so the expanded blocks compare as a set."""
    n = WIDTH - RATIO + 1
    _exact(*([torch.cat([t[:, :n].sort().values, t[:, n:]], 1)] for t in (got, want)))


def _selection(rows=8, cols=64, **kw):
    kw = dict(compress_ratio=RATIO, token_topk=TOPK, num_heads=QO, head_dim=D) | kw
    return qsa.QSASelection(max_rows=rows, max_columns=cols, device=DEV, **kw)


def _select(sel, b, ws):
    return sel.run(*b.sel, out_route=Z(len(b.q), WIDTH), workspace=ws)


_SEL_CASES = {
    **{f"seed{s}": (dict(seed=s), {}) for s in (1, 2, 3)},
    "chunked": (dict(rows=14, seed=4), dict(score_budget_bytes=1024)),
    **{f"position{p}": (dict(rows=4, nreq=1, pos=p), {}) for p in (0, 3, 4, 7, 37, 63)},
    **{f"length{n}": (dict(rows=4, nreq=1, pos=63, length=n), {}) for n in (0, 8, 64)},
    "unmapped": (dict(rows=4, nreq=1, pos=63, length=2 * PAGE, unmapped=True), {}),
    "long-rows": (  # multi-CTA radix, which reads its counters out of the scratch
        dict(rows=4, nreq=1, pos=(1 << 18) - 1, length=1 << 18),
        dict(cols=1 << 16, tie_break=0, dsa_graph_safe=False),
    ),
}


@pytest.mark.parametrize("name", _SEL_CASES)
def test_selection_matches_oracle(name):
    b = _batch(**_SEL_CASES[name][0])
    sel = _selection(len(b.q), **_SEL_CASES[name][1])
    assert name != "chunked" or sel.rows_per_chunk < len(b.q)
    ws = torch.full((sel.workspace_size(),), 0xA5, dtype=U8, device=DEV)  # dirty
    route = _select(sel, b, ws).clone()
    _same_route(route, _select_ref(*b.sel, sel.max_columns))
    _exact([_select(sel, b, ws)], [route])  # the same answer from reused scratch


def test_selection_allocates_nothing(monkeypatch):
    b, sel = _batch(), _selection()
    ws = Z(sel.workspace_size(), dtype=U8)
    route = _select(sel, b, ws)
    for name in ("empty", "zeros", "full", "empty_strided", "randn"):
        monkeypatch.setattr(torch, name, _fail)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    sel.run(*b.sel, out_route=route, workspace=ws)
    assert torch.cuda.max_memory_allocated() == before
    assert _selection(2048, 1 << 16).workspace_size() > 0  # nor does sizing


def test_selections_replay_from_slices_of_one_arena():
    sels, batches = [_selection(), _selection()], [_batch(seed=7), _batch(seed=8)]
    size = sels[0].workspace_size()
    arena, routes = Z(2, size, dtype=U8), Z(2, 8, WIDTH)
    run = lambda i: sels[i].run(
        *batches[i].sel, out_route=routes[i], workspace=arena[i]
    )
    graphs = [_capture(lambda i=i: run(i)) for i in range(2)]
    routes.zero_()
    _replay_on_streams(graphs)
    for route, b in zip(routes, batches, strict=True):
        _same_route(route, _select_ref(*b.sel))


def test_selection_refuses():
    for kw, match in [
        (dict(num_heads=17), "query heads"),
        (dict(head_dim=96), "head dimensions"),
        (dict(token_topk=0), "token_topk"),
        (dict(compress_ratio=0), "compress_ratio"),
        (dict(score_budget_bytes=0), "score_budget_bytes"),
        (dict(token_topk=30), "whole number of blocks"),
    ]:
        with pytest.raises(ValueError, match=match):
            _selection(**kw)
    sel, b = _selection(), _batch().sel
    arena = Z(sel.workspace_size() + 16, dtype=U8)
    ws, route, wide = arena[:-16], Z(8, WIDTH), Z(8, WIDTH + 1)
    for batch, out, w, match in [
        (b, route, ws[:-1], "workspace needs"),
        (b, route, arena[1:-15], "aligned"),
        (b, route, ws.view(torch.int32), "contiguous uint8"),
        (b, wide, ws, "out_route must be"),
        (b[:2] + [t.long() for t in b[2:]], route, ws, "int32"),
        ([b[0].float(), b[1].float(), *b[2:]], route, ws, "q must be one of"),
    ]:
        with pytest.raises(ValueError, match=match):
            sel.run(*batch, out_route=out, workspace=w)


# --- attention ---------------------------------------------------------------

_GEOMETRY = dict(num_qo_heads=QO, num_kv_heads=KV, head_dim=D, route_width=WIDTH)
_FORMATS = {
    "dense": dict(kv_data_type=BF16, kv_cache_format="dense"),
    "fp8": dict(kv_data_type=FP8, kv_cache_format="fp8_e4m3"),
    "nvfp4": dict(kv_data_type=U8, kv_cache_format="nvfp4", kv_layout="HND"),
}


def _kwargs(fmt="dense", rows=16, **kw):
    dtypes = dict(q_data_type=BF16, o_data_type=BF16)
    return _GEOMETRY | dtypes | _FORMATS[fmt[:5]] | dict(max_rows=rows) | kw


def _attention(fmt="dense", rows=16, bind=True, slots=SLOTS, fill=0):
    kw = _kwargs(fmt, rows)
    sizes = qsa.QSAAttention.workspace_bytes(device=DEV, **kw)
    buffers = [torch.full((n,), fill, dtype=U8, device=DEV) for n in sizes]
    att = qsa.QSAAttention(buffers[0], **kw)
    att.buffers = buffers
    if bind:
        att.bind_transient_workspace(buffers[1])
        att.plan_cache(slots, PAGE)
    return att


def _decode_nvfp4(data, sf, scale):
    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64, device=DEV)
    fields = torch.stack([data & 15, data >> 4], -1).flatten(-2).long()
    values = e2m1[fields & 7] * (1 - 2 * (fields >> 3))
    return values * sf.float().double().repeat_interleave(16, -1) * scale


def _cache(fmt, b, ks=1.5, vs=0.75):
    """The run arguments for a cache of this format, and its values decoded."""
    if fmt == "dense":
        return dict(k_data=b.k, v_data=b.v), (b.k, b.v)
    if fmt == "fp8":
        k, v = (b.k.float() / ks).to(FP8), (b.v.float() / vs).to(FP8)
        kw = dict(k_data=k, v_data=v, k_scale=ks, v_scale=vs)
        return kw, (k.double() * ks, v.double() * vs)
    data = torch.zeros(2, 32, KV, PAGE, D // 2, dtype=U8, device=DEV)
    sf = torch.zeros(2, 32, KV, PAGE, D // 16, dtype=FP8, device=DEV)
    kv = [t.view(-1, KV, D) for t in (b.k, b.v)]
    slots = torch.arange(SLOTS, dtype=INT, device=DEV)
    write = flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping
    write(*kv, slots, (data[0], data[1]), (sf[0], sf[1]), ks, vs, kv_layout="HND")
    decoded = [
        _decode_nvfp4(data[i], sf[i], s).transpose(1, 2) for i, s in ((0, ks), (1, vs))
    ]
    if fmt == "nvfp4-interleaved":  # one allocation per slot, [data | scales]
        full = torch.cat([data, sf.view(U8)], -1).transpose(0, 1).flatten(1, 2)
        data = [full[:, i::2, ..., : D // 2] for i in (0, 1)]
        sf = [full[:, i::2, ..., D // 2 :].view(FP8) for i in (0, 1)]
        assert not data[0].is_contiguous()
    kw = dict(k_data=data[0], v_data=data[1], k_sf=sf[0], v_sf=sf[1])
    return kw | dict(k_scale=ks, v_scale=vs), decoded


def _run(att, b, **kw):
    """`att.run` on the batch; an argument given as ... is left out."""
    args = dict(q=b.q, k_data=b.k, v_data=b.v, route=b.route, block_table=b.table)
    args |= dict(token_to_request=b.t2r, output_gate=b.gate) | kw
    return att.run(**{n: v for n, v in args.items() if v is not ...})


def _attend_ref(b, k, v):
    """Attention over the route's live entries in float64, then the gate."""
    k, v = (t.reshape(-1, KV, D)[:, 0].double() for t in (k, v))
    out = torch.zeros(len(b.q), QO, D, dtype=torch.float64, device=DEV)
    for row in range(len(b.q)):
        toks = torch.tensor([t for t in b.route[row].tolist() if t >= 0], device=DEV)
        if len(toks):
            slots = b.table[b.t2r[row], toks // PAGE] * PAGE + toks % PAGE
            w = torch.softmax(k[slots] @ b.q[row].double().T / D**0.5, 0)
            out[row] = w.T @ v[slots]
    gate = torch.sigmoid(b.gate.view(-1, QO, D).float())
    return (out.to(BF16).float() * gate).to(BF16)


def _close(got, want):
    """Within 4 bf16 steps of the largest output, and 3% where an output is over 0.05."""
    got, want = got.double(), want.double()
    err = (got - want).abs()
    assert err.max() <= 4 * 2**-8 * max(1.0, want.abs().max())
    assert torch.where(want.abs() >= 5e-2, err / want.abs(), 0).max() <= 3e-2


@pytest.mark.parametrize(
    "fmt,rows,null_page",
    [("dense", 16, 0), ("dense", 3, 0), ("fp8", 16, 0), ("nvfp4", 16, 0)]
    + [("nvfp4-interleaved", 16, 0), ("dense", 16, 1)],
)
def test_attention_matches_oracle(fmt, rows, null_page):
    b = _batch(rows, 1 if rows < 4 else 2, seed=rows)
    if null_page:  # page 0 is the caller's padding: no request maps it, and it is NaN
        b.table += 1
        b.k, b.v = (torch.cat([t[:1] * float("nan"), t]) for t in (b.k, b.v))
    kw, decoded = _cache(fmt, b)
    att = _attention(fmt, slots=SLOTS + null_page * PAGE)
    out = _run(att, b, **kw)
    _close(out, _attend_ref(b, *decoded))
    _exact([_run(att, b, **kw)], [out])  # deterministic
    ungated = _run(att, b, **kw, output_gate=torch.full_like(b.gate, 40.0))
    gate = torch.sigmoid(b.gate.view(-1, QO, D).float())
    _exact([out], [(ungated.float() * gate).to(BF16)])  # the gate is one rounding


def test_attention_plan_lifecycle():
    """Plans are rebuilt in place until the first run, and frozen after it."""
    att, b = _attention(bind=False), _batch(4, 1)
    with pytest.raises(RuntimeError, match="bind_transient_workspace"):
        att.plan_cache(SLOTS, PAGE)
    att.bind_transient_workspace(att.buffers[1])
    with pytest.raises(RuntimeError, match="plan_cache"):
        _run(att, b)
    with pytest.raises(ValueError, match="multiple of page_size"):
        att.plan_cache(SLOTS + 1, PAGE)
    assert att._staging is None
    att.plan_cache(PAGE, PAGE)  # a memory profile's minimal cache, then the real one
    replaced = [weakref.ref(w) for w in att._wrappers.values()]
    staging = att._staging
    assert staging.is_pinned() and staging.numel() == max(att._plan_bytes)
    att.plan_cache(SLOTS, PAGE)
    gc.collect()
    before = torch.cuda.memory_allocated()
    for slots in [PAGE, SLOTS] * 5:
        att.plan_cache(slots, PAGE)
    gc.collect()
    assert torch.cuda.memory_allocated() == before and att._staging is staging
    assert all(ref() is None for ref in replaced)
    held = dict(att._wrappers)
    att.plan_cache(SLOTS, PAGE)  # every layer asks for the cache it already has
    assert all(att._wrappers[rows] is w for rows, w in held.items())
    _run(att, b)
    att.plan_cache(SLOTS, PAGE)
    with pytest.raises(RuntimeError, match="has run"):
        att.plan_cache(2 * SLOTS, PAGE)


def test_attention_runs_from_dirty_buffers_without_allocating(monkeypatch):
    b = _batch(16)
    want, seen = _run(_attention(), b), []
    plan = flashinfer.sparse.BlockSparseAttentionWrapper.plan
    spy = lambda self, *a, **k: seen.append((a[1].clone(), k["packed_mask"].clone()))
    wrapper = flashinfer.sparse.BlockSparseAttentionWrapper
    monkeypatch.setattr(
        wrapper, "plan", lambda *a, **k: (spy(*a, **k), plan(*a, **k))[1]
    )
    att = _attention(fill=0xA5)  # the scratch arrives holding another tenant's bytes
    assert len(seen) == len(att.row_buckets)
    assert all(i.abs().max() == m.max() == 0 for i, m in seen)  # planned from zeros
    out = torch.empty_like(want)
    _run(att, b, out=out)
    for name in ("empty", "zeros", "full", "empty_strided"):
        monkeypatch.setattr(torch, name, _fail)
    monkeypatch.setattr(wrapper, "plan", _fail)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    _run(att, b, out=out)
    assert torch.cuda.max_memory_allocated() == before
    monkeypatch.undo()
    _exact([out], [want])


def test_attention_graphs_replay_together():
    """Two rungs of one attention take turns; two attentions replay at once."""
    atts = [_attention(rows=256), _attention(rows=256)]
    assert len(atts[0].row_buckets) > 1
    calls = []
    for att, rows in ((atts[0], 16), (atts[0], 200), (atts[1], 200)):
        b = _batch(rows, seed=rows)
        out = _run(att, b)
        call = lambda att=att, b=b, out=out: _run(att, b, out=out)
        calls.append((call, out, out.clone()))
    graphs = [_capture(call) for call, _, _ in calls]
    for graph, (_, out, want) in [*zip(graphs[:2], calls, strict=False)] * 2:
        out.zero_()
        graph.replay()
        _exact([out], [want])
    for _, out, _ in calls:
        out.zero_()
    _replay_on_streams(graphs[1:])
    _exact([c[1] for c in calls[1:]], [c[2] for c in calls[1:]])


_BAD_BUILDS = [
    (dict(kv_data_type=BF16, kv_cache_format="nvfp4"), "packed NVFP4"),
    (dict(kv_data_type=U8, kv_cache_format="dense"), "not a dense cache"),
    (dict(kv_cache_format="fp4"), "kv_cache_format"),
    (dict(kv_data_type=U8, kv_cache_format="fp8_e4m3"), "float8_e4m3fn"),
    (dict(num_qo_heads=3, num_kv_heads=2), "whole number of query heads"),
    (dict(num_kv_heads=0), "positive"),
    (dict(kv_data_type=U8, kv_cache_format="nvfp4", head_dim=100), "sixteen"),
]
_BAD_RUNS = [
    ("dense", lambda c: dict(k_data=c["k_data"][:16]), "k_data must be"),
    ("dense", lambda c: dict(k_data=c["k_data"][..., :64]), "k_data must be"),
    ("dense", lambda c: dict(k_scale=2.0), "no global scale"),
    ("dense", lambda c: dict(route=Z(16, WIDTH + 1)), "count column"),
    ("dense", lambda c: dict(output_gate=Z(16, 64, dtype=BF16)), "output_gate"),
    ("dense", lambda c: dict(output_gate=...), "output_gate"),
    ("dense", lambda c: dict(out=Z(16, QO, 64, dtype=BF16)), "out must be"),
    ("fp8", lambda c: dict(k_scale=None), "k_scale"),
    ("fp8", lambda c: dict(v_scale=None), "v_scale"),
    ("fp8", lambda c: dict(k_sf=c["k_data"].view(U8)[..., :8]), "no scale planes"),
    ("nvfp4", lambda c: dict(v_sf=None), "scale plane"),
    ("nvfp4", lambda c: dict(k_sf=c["k_sf"][..., :1]), "k_sf must be"),
    (
        "nvfp4",
        lambda c: dict(k_data=Z(32, KV, PAGE, D, dtype=U8)[..., ::2]),
        "innermost",
    ),
    *[
        ("fp8", lambda c, s=s: dict(k_scale=s), match)
        for s, match in [(torch.tensor(2.0), "host float"), (True, "real number")]
        + [("two", "real number"), (float("nan"), "finite"), (float("inf"), "finite")]
        + [(0.0, "positive"), (-1.0, "positive")]
    ],
]


def test_attention_refuses_before_launching(monkeypatch):
    for kw, match in _BAD_BUILDS:
        with pytest.raises(ValueError, match=match):
            qsa.QSAAttention(Z(1 << 20, dtype=U8), **_kwargs(**kw))
    sizes = qsa.QSAAttention.workspace_bytes(device=DEV, **_kwargs())
    for buffer, match in [
        (torch.empty(sizes[0], dtype=U8), "CUDA"),
        (torch.empty(sizes[0], dtype=INT, device=DEV), "raw bytes"),
        (torch.empty(2 * sizes[0], dtype=U8, device=DEV)[::2], "contiguous"),
        (torch.empty(sizes[0] + 16, dtype=U8, device=DEV)[1:], "aligned"),
        (torch.empty(sizes[0] - 1, dtype=U8, device=DEV), "needs"),
    ]:
        with pytest.raises(ValueError, match=match):
            qsa.QSAAttention(buffer, **_kwargs())
    with pytest.raises(ValueError, match="needs"):
        _attention(bind=False).bind_transient_workspace(Z(sizes[1] - 1, dtype=U8))
    b = _batch(16)
    atts = {fmt: _attention(fmt) for fmt in _FORMATS}
    caches = {fmt: _cache(fmt, b)[0] for fmt in _FORMATS}
    for name in ("qsa_route_from_logical", "qsa_output_gate"):
        monkeypatch.setattr(sys.modules["flashinfer.qsa_ops.attention"], name, _fail)
    monkeypatch.setattr(flashinfer.sparse.BlockSparseAttentionWrapper, "run", _fail)
    for fmt, edit, match in _BAD_RUNS:
        with pytest.raises((ValueError, TypeError), match=match):
            _run(atts[fmt], b, **caches[fmt] | edit(caches[fmt]))


# --- the step ----------------------------------------------------------------


def _runtime(page=PAGE):
    cfg = qsa.QSAConfig(QO, KV, D, 8, BF16, BF16, BF16, "dense", 64, RATIO, TOPK, QO, D)
    buffers = [
        Z(n, dtype=U8) for n in qsa.QSA.workspace_requirements(cfg, device=DEV)[:2]
    ]
    rt = qsa.QSA(cfg, buffers[0])
    rt.bind_transient_workspace(buffers[1])
    rt.plan_cache(SLOTS, page)
    rt.buffers = buffers
    return rt


def _step(rt, b, route=None, out=None):
    route = Z(8, WIDTH) if route is None else route
    rt.run_selection(*b.sel, out_route=route)
    args = dict(
        block_table=b.table, token_to_request=b.t2r, output_gate=b.gate, out=out
    )
    return route, rt.run_attention(b.q, b.k, b.v, route=route, **args)


_world = lambda seed: _batch(
    8, 1, seed, pos=SLOTS - 1, length=SLOTS
)  # at the last token


def test_qsa_step_matches_oracle():
    b = _world(1)
    route, out = _step(_runtime(), b)
    b.route = _select_ref(*b.sel)
    _same_route(route, b.route)
    want = _attend_ref(b, b.k, b.v)
    _close(out, want)
    with pytest.raises(AssertionError):  # the bound does see a dropped gate
        _close(want.double() / torch.sigmoid(b.gate.view(-1, QO, D).double()), want)
    with pytest.raises(ValueError, match="k_data must be"):  # entries read as pages
        _step(_runtime(page=1), b)


def test_qsa_workspace_lifetimes():
    """Sizing and building take nothing; the plans are not in the shared scratch."""
    config = _runtime().config
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    needs = {qsa.QSA.workspace_requirements(config, device=DEV) for _ in range(4)}
    assert len(needs) == 1 and torch.cuda.max_memory_allocated() == before
    persistent, transient = (Z(n, dtype=U8) for n in [*needs][0][:2])
    before = torch.cuda.memory_allocated()
    rt = qsa.QSA(config, persistent)
    rt.bind_transient_workspace(transient)
    assert torch.cuda.memory_allocated() == before
    rt.plan_cache(SLOTS, PAGE)
    b, a = _world(2), rt._attention
    route, out = _step(rt, b)
    want, plans = out.clone(), persistent.clone()
    regions = [transient, a._padded_q, a._padded_out, a._route_base, a._mask_base]
    for region in regions + [a._float_workspace, rt._selection._bound_workspace]:
        region.view(U8).fill_(0xA5)  # what another consumer of the step leaves
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        _step(rt, b, route, out)
        assert torch.cuda.max_memory_allocated() == before
        _exact([out, persistent], [want, plans])
    graph = _capture(lambda: _step(rt, b, route, out))
    transient.fill_(0xA5)
    out.zero_()
    graph.replay()
    _exact([out], [want])


def test_a_layer_reuses_its_own_route():
    """The route is the caller's: a speculative step attends with it again."""
    rt, b = _runtime(), _world(3)
    route, out = _step(rt, b)
    kept, first = route.clone(), out.clone()
    _step(rt, SimpleNamespace(**vars(b) | {"sel": [b.q.flip(0), *b.sel[1:]]}))
    _exact([route], [kept])
    args = dict(block_table=b.table, token_to_request=b.t2r, output_gate=b.gate)
    _exact([rt.run_attention(b.q, b.k, b.v, route=route, **args)], [first])


# --- capabilities ------------------------------------------------------------

ALL = 0b11111
SEL, GATED = qsa.QSA_CAP_SELECTION, ALL & ~qsa.QSA_CAP_SELECTION


def test_capabilities_report_this_build(monkeypatch):
    current = torch.device(DEV, torch.cuda.current_device())
    assert qsa.qsa_capabilities() == qsa.qsa_capabilities(current) == ALL
    names = {"selection", "attention_paged", "nvfp4", "fp8", "output_gate"}
    assert qsa.qsa_capability_names() == names
    assert not qsa.qsa_paged_scores.is_compute_capability_supported(75)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    capabilities._capabilities.cache_clear()
    try:  # a module built for SM80 that loads on an SM75 device has no scorer
        assert qsa.qsa_capabilities() == ALL & ~SEL
    finally:
        capabilities._capabilities.cache_clear()


@pytest.mark.parametrize(
    "module,symbol,lost",
    [
        ("qsa_ops.scores", None, SEL),
        ("qsa_ops.route", "qsa_route_from_logical", SEL),
        ("topk", None, SEL),
        ("topk", "cub_topk_ragged_transform_workspace_size_for", SEL),
        ("qsa_ops.output_gate", None, GATED),
        ("qsa_ops.output_gate", "qsa_output_gate_capabilities", GATED),
    ],
)
def test_a_missing_piece_takes_its_capability_down(monkeypatch, module, symbol, lost):
    owner = sys.modules[f"flashinfer.{module}"]
    loader = f"get_{module.replace('qsa_ops.', 'qsa_')}_module"
    loaded = getattr(owner, loader)() if symbol else None

    class Without:  # the module loads, without one symbol
        def __getattr__(self, name):
            if name == symbol:
                raise AttributeError(name)
            return getattr(loaded, name)

    monkeypatch.setattr(owner, loader, Without if symbol else _fail)
    capabilities._capabilities.cache_clear()
    try:
        assert qsa.qsa_capabilities() == ALL & ~lost
    finally:
        capabilities._capabilities.cache_clear()
