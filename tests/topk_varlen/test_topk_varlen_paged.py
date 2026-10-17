# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Paged output of ``top_k_varlen`` (gvr_2, decode mode): every selected column
``c`` of request ``q`` is returned as the physical KV slot
``page_table[q, c // page_size] * page_size + c % page_size``, fused into the
kernels' index emit (the SGLang DSA decode contract; same mapping as
``top_k_page_table_transform``).

Contract under test: paged output == page-table transform of the window-local
output, row by row as a set (the engines' emit order is not fixed), on every
engine family (register, clustered-register, cluster, streaming slab), with
mixed / short / empty rows, ``next_n > 1`` (one table row per request),
``compress_ratio = 4`` (compressed column units), page sizes 1 / 64 / 4096,
``warmup_varlen(page_size=)`` + CUDA-graph replay, the reference engine, and
the API admission rules.
"""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.topk_varlen.kernels import gvr2_topk_host as _host
    from flashinfer.utils import get_compute_capability

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (_FLASHINFER_AVAILABLE and torch.cuda.is_available()),
    reason="flashinfer + CUDA required",
)
_DEV = "cuda"


def _cc() -> int:
    major, minor = get_compute_capability(torch.device(_DEV))
    return major * 10 + minor


requires_gvr2 = pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE
    or not torch.cuda.is_available()
    or not flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()),
    reason="gvr_2 unsupported on this device",
)


def _page_table(batch, ncols, page_size, seed):
    """One distinct random physical page per (request, page): injective per row,
    so set equality of transformed rows is a faithful check."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    pages = (ncols + page_size - 1) // page_size
    pool = 4 * pages + 7
    pt = torch.stack(
        [torch.randperm(pool, generator=gen, device=_DEV)[:pages] for _ in range(batch)]
    ).to(torch.int32)
    return pt.contiguous()


def _expected(local, pt, next_n, page_size):
    """Page-table transform of window-local indices (pads stay -1)."""
    rows = local.shape[0]
    shift = page_size.bit_length() - 1
    req = (torch.arange(rows, device=_DEV) // next_n).unsqueeze(1)
    l64 = local.to(torch.int64)
    phys = (pt.to(torch.int64)[req, (l64 >> shift).clamp_min(0)] << shift) | (
        l64 & (page_size - 1)
    )
    return torch.where(local >= 0, phys.to(torch.int32), local)


def _same_rows(a, b):
    return torch.equal(torch.sort(a, dim=1).values, torch.sort(b, dim=1).values)


def _mixed_lens(batch, ncols, top_k, next_n, seed):
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    lens = torch.randint(top_k + 1, ncols + 1, (batch,), generator=gen, device=_DEV)
    # short / boundary / empty requests (identity path, -1 rows)
    if batch >= 4:
        lens[0] = top_k // 2
        lens[1] = top_k
        lens[2] = max(next_n - 1, 0)  # every row of the request empty
        lens[3] = top_k + 1
    return lens.to(torch.int32)


def _run_pair(lg, lens, top_k, pt, page_size, next_n=1, cr=1):
    loc, _ = flashinfer.top_k_varlen(
        lg, lens, top_k, next_n=next_n, compress_ratio=cr, backend="gvr_2"
    )
    pg, _ = flashinfer.top_k_varlen(
        lg,
        lens,
        top_k,
        next_n=next_n,
        compress_ratio=cr,
        backend="gvr_2",
        page_table=pt,
        page_size=page_size,
    )
    torch.cuda.synchronize()
    return loc, pg


# rows x columns shapes spanning the four engine families of the varlen router
_SHAPES = [(4, 4096), (8, 32768), (96, 65536), (640, 16384)]


@requires_gvr2
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"r{s[0]}n{s[1]}")
@pytest.mark.parametrize("top_k", [512, 2048], ids=lambda k: f"k{k}")
def test_paged_matches_transformed_local(top_k, shape):
    rows, ncols = shape
    page_size = 64
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = _mixed_lens(rows, ncols, top_k, 1, seed=rows + ncols)
    pt = _page_table(rows, ncols, page_size, seed=7)
    loc, pg = _run_pair(lg, lens, top_k, pt, page_size)
    assert torch.equal(pg < 0, loc < 0) or _same_rows(pg < 0, loc < 0)
    assert _same_rows(pg, _expected(loc, pt, 1, page_size))


@requires_gvr2
def test_paged_engine_families_covered():
    """The shape grid above reaches at least three of the four engine families."""
    fams = set()
    for rows, ncols in _SHAPES:
        lc = _host._varlen_launcher(rows, ncols, 512, ncols, 1, 1, True, 6)
        fams.add(lc[0])
    assert len(fams) >= 3, fams


@requires_gvr2
@pytest.mark.parametrize("page_size", [1, 4096], ids=lambda p: f"ps{p}")
def test_paged_page_sizes(page_size):
    rows, ncols, top_k = 16, 8192, 512
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = _mixed_lens(rows, ncols, top_k, 1, seed=3)
    pt = _page_table(rows, ncols, page_size, seed=11)
    loc, pg = _run_pair(lg, lens, top_k, pt, page_size)
    assert _same_rows(pg, _expected(loc, pt, 1, page_size))


@requires_gvr2
@pytest.mark.parametrize(
    "rows,ncols,top_k",
    [
        (9, 131136, 512),
        (9, 131136, 2048),
        (16, 131136, 512),
        (32, 131136, 2048),
        (128, 131136, 512),
        (160, 65600, 2048),
    ],
    ids=[
        "split9-k512",
        "split9-k2048",
        "split16-k512",
        "split32-k2048",
        "cta1024-k512",
        "cta512-k2048",
    ],
)
@pytest.mark.parametrize("page_size", [1, 64, 128], ids=lambda p: f"ps{p}")
def test_paged_main_register_rows(rows, ncols, top_k, page_size):
    """Main-engine launches (capacity-wide logits) whose rows mostly fit the
    in-kernel register path: that path stages its own page-table slice, sized
    from the register capacity (none at page size 1), and must map through the
    request's row like the streaming path."""
    plan = _host.route_streaming(rows, ncols, ncols, top_k, force_main=True)
    cap = (
        plan["tpl"][0] * _host._main_reg_vpt(plan["tpl"][0], plan["tpl"][5], ncols) * 4
    )
    assert cap > 0
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    gen = torch.Generator(device=_DEV).manual_seed(rows + page_size)
    lens = torch.randint(top_k + 1, cap + 1, (rows,), generator=gen, device=_DEV)
    # band edges, the VPT 2 / VPT 4 boundary, a short row, a streaming row, and
    # short register rows (k + 1, 2048, 2049)
    lens[0], lens[1], lens[2], lens[3] = cap, cap + 1, top_k, ncols
    lens[4], lens[5] = cap // 2, cap // 2 + 1
    lens[6], lens[7], lens[8] = top_k + 1, 2048, 2049
    lens = lens.to(torch.int32)
    pt = _page_table(rows, ncols, page_size, seed=41)
    loc, pg = _run_pair(lg, lens, top_k, pt, page_size)
    page_shift = page_size.bit_length() - 1
    assert (
        _host._varlen_launcher(rows, ncols, top_k, ncols, 1, 1, True, page_shift)[0]
        == "main"
    )
    for r in range(rows):  # the local output itself is the exact top-k
        n = int(lens[r])
        if n > top_k:
            idx = loc[r].to(torch.int64)
            assert (
                int(idx.min()) >= 0
                and int(idx.max()) < n
                and idx.unique().numel() == top_k
            )
            ref = torch.topk(lg[r, :n], top_k).values.sort().values
            assert torch.equal(lg[r, idx].sort().values, ref), f"row {r} inexact"
    assert _same_rows(pg, _expected(loc, pt, 1, page_size))


@requires_gvr2
def test_paged_next_n_shares_the_request_table():
    """next_n rows per request all map through the request's page-table row."""
    batch, next_n, ncols, top_k = 12, 3, 8192, 1024
    rows = batch * next_n
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = _mixed_lens(batch, ncols, top_k, next_n, seed=5)
    pt = _page_table(batch, ncols, 64, seed=13)
    loc, pg = _run_pair(lg, lens, top_k, pt, 64, next_n=next_n)
    assert _same_rows(pg, _expected(loc, pt, next_n, 64))


@requires_gvr2
def test_paged_compressed_columns():
    """compress_ratio=4: columns and page_size are in compressed units."""
    rows, ncols, top_k, cr = 8, 4096, 512, 4
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    gen = torch.Generator(device=_DEV).manual_seed(9)
    lens = torch.randint(
        (top_k + 1) * cr, ncols * cr + 1, (rows,), generator=gen, device=_DEV
    ).to(torch.int32)
    pt = _page_table(rows, ncols, 64, seed=17)
    loc, pg = _run_pair(lg, lens, top_k, pt, 64, cr=cr)
    assert _same_rows(pg, _expected(loc, pt, 1, 64))


@requires_gvr2
def test_paged_reference_engine_parity():
    rows, ncols, top_k = 6, 4096, 512
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = _mixed_lens(rows, ncols, top_k, 1, seed=21)
    pt = _page_table(rows, ncols, 64, seed=23)
    o1 = torch.empty(rows, top_k, dtype=torch.int32, device=_DEV)
    o2 = torch.empty_like(o1)
    _host.run_varlen(
        lg, None, lens, o1, top_k=top_k, max_seq_len=ncols, page_table=pt, page_size=64
    )
    _host.run_varlen(
        lg,
        None,
        lens,
        o2,
        top_k=top_k,
        max_seq_len=ncols,
        engine="reference",
        page_table=pt,
        page_size=64,
    )
    torch.cuda.synchronize()
    # both are exact top-k of the same rows through the same injective table
    assert _same_rows(o1, o2)


@requires_gvr2
def test_paged_warmup_and_cuda_graph_replay():
    rows, ncols, top_k, page_size = 32, 8192, 512, 64
    _host.warmup_varlen(top_k, ncols, num_rows_list=(rows,), page_size=page_size)
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = torch.full((rows,), ncols, dtype=torch.int32, device=_DEV)
    pt = _page_table(rows, ncols, page_size, seed=29)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        flashinfer.top_k_varlen(
            lg,
            lens,
            top_k,
            out_indices=out,
            backend="gvr_2",
            page_table=pt,
            page_size=page_size,
        )
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(
            lg,
            lens,
            top_k,
            out_indices=out,
            backend="gvr_2",
            page_table=pt,
            page_size=page_size,
        )
    torch.cuda.current_stream().wait_stream(s)
    # new scores, lengths AND page table between replays
    lg.normal_()
    lens.copy_(_mixed_lens(rows, ncols, top_k, 1, seed=31))
    pt.copy_(_page_table(rows, ncols, page_size, seed=37))
    g.replay()
    torch.cuda.synchronize()
    loc, _ = flashinfer.top_k_varlen(lg, lens, top_k, backend="gvr_2")
    torch.cuda.synchronize()
    assert _same_rows(out, _expected(loc, pt, 1, page_size))


@requires_gvr2
@pytest.mark.parametrize("bound", [None, 16384], ids=["capacity", "bound16k"])
def test_paged_capacity_buffer_graph_replay(bound):
    """V4.1 c1 shape: a 1M-column buffer (page 128) whose rows stay at or below
    16K, captured once with the capacity envelope or with the 16K-token graph
    variant's bound; replays rewrite lengths, scores and the page table."""
    rows, ncols, top_k, page_size, live = 32, 1048576, 512, 128, 16384
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = _mixed_lens(rows, live, top_k, 1, seed=43)
    pt = _page_table(rows, ncols, page_size, seed=47)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    kw = dict(out_indices=out, backend="gvr_2", page_table=pt, page_size=page_size)
    if bound is not None:
        kw["max_window_len"] = bound
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        flashinfer.top_k_varlen(lg, lens, top_k, **kw)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(lg, lens, top_k, **kw)
    torch.cuda.current_stream().wait_stream(s)
    for seed in (53, 59, 61):
        lg.normal_()
        lens.copy_(_mixed_lens(rows, live, top_k, 1, seed=seed))
        pt.copy_(_page_table(rows, ncols, page_size, seed=seed + 1))
        out.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        loc, _ = flashinfer.top_k_varlen(lg, lens, top_k, backend="gvr_2")
        torch.cuda.synchronize()
        assert _same_rows(out, _expected(loc, pt, 1, page_size))


def test_paged_api_validation():
    rows, ncols, k = 8, 4096, 512
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = torch.full((rows,), ncols, dtype=torch.int32, device=_DEV)
    pt = torch.zeros((rows, ncols // 64), dtype=torch.int32, device=_DEV)
    # page_size is a paged-output argument: any value without page_table is
    # refused by the API body, on every GPU (decode admission is universal)
    for ps in (64, 1):
        with pytest.raises(ValueError, match="page_size requires page_table"):
            flashinfer.top_k_varlen(lg, lens, k, page_size=ps)
    if not flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()):
        from flashinfer.utils import BackendSupportedError

        with pytest.raises((BackendSupportedError, ValueError)):
            flashinfer.top_k_varlen(lg, lens, k, page_table=pt, page_size=64)
        return
    from flashinfer.utils import BackendSupportedError

    bad = [
        dict(page_table=pt, page_size=48),  # not a power of two
        dict(page_table=pt, page_size=64, return_values=True),
        dict(page_table=pt[:4], page_size=64),  # wrong request count
        dict(page_table=pt.to(torch.int64), page_size=64),
        dict(page_table=pt[:, :10], page_size=64),  # too few pages for the width
    ]
    for kw in bad:
        with pytest.raises(ValueError, match="page_table|page_size"):
            flashinfer.top_k_varlen(lg, lens, k, **kw)
    # None is the omitted form: the same paged output as page_size=1
    pt1 = _page_table(rows, ncols, 1, seed=3)
    out_none, _ = flashinfer.top_k_varlen(lg, lens, k, page_table=pt1)
    out_one, _ = flashinfer.top_k_varlen(lg, lens, k, page_table=pt1, page_size=1)
    torch.cuda.synchronize()
    assert _same_rows(out_none, out_one)
    # mode conflicts are refused by the API body with a ValueError, under auto
    # and an explicit gvr_2 alike (the checker admits the call for that)
    for kw in (
        dict(row_starts=torch.zeros(rows, dtype=torch.int32, device=_DEV)),
        dict(pre_idx=torch.zeros((rows, k), dtype=torch.int32, device=_DEV)),
    ):
        for extra in ({}, {"backend": "gvr_2"}):
            with pytest.raises(ValueError, match="page_table"):
                flashinfer.top_k_varlen(
                    lg, lens, k, page_table=pt, page_size=64, **kw, **extra
                )
    # other backends refuse it explicitly
    with pytest.raises((BackendSupportedError, ValueError)):
        flashinfer.top_k_varlen(
            lg, lens, k, backend="radix", page_table=pt, page_size=64
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_paged_on_non_current_device():
    """logits, lengths and page table on a device that is not the current one:
    the host re-enters itself under that device and must forward page_table
    and page_size, otherwise the call silently returns raw columns."""
    other = torch.device("cuda:1")
    cc = get_compute_capability(other)
    if not flashinfer.top_k_varlen.is_backend_supported("gvr_2", cc[0] * 10 + cc[1]):
        pytest.skip("gvr_2 unsupported on cuda:1")
    torch.cuda.set_device(0)
    rows, ncols, top_k, page_size = 8, 4096, 512, 64
    gen = torch.Generator(device=other).manual_seed(9)
    lg = torch.randn((rows, ncols), generator=gen, dtype=torch.float32, device=other)
    lens = torch.randint(top_k + 1, ncols + 1, (rows,), generator=gen, device=other).to(
        torch.int32
    )
    pages = (ncols + page_size - 1) // page_size
    pt = torch.stack(
        [
            torch.randperm(4 * pages + 7, generator=gen, device=other)[:pages]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    loc, _ = flashinfer.top_k_varlen(lg, lens, top_k, backend="gvr_2")
    pg, _ = flashinfer.top_k_varlen(
        lg, lens, top_k, backend="gvr_2", page_table=pt, page_size=page_size
    )
    torch.cuda.synchronize(other)
    assert pg.device == other and loc.device == other
    shift = page_size.bit_length() - 1
    l64 = loc.to(torch.int64)
    phys = (
        pt.to(torch.int64)[
            torch.arange(rows, device=other).unsqueeze(1), (l64 >> shift).clamp_min(0)
        ]
        << shift
    ) | (l64 & (page_size - 1))
    expected = torch.where(loc >= 0, phys.to(torch.int32), loc)
    assert torch.equal(torch.sort(pg, dim=1).values, torch.sort(expected, dim=1).values)


def test_paged_boundary_ties_tie_aware_check():
    """Boundary ties under paged output (reviewer's case): 1024 equal scores,
    K=512, an identity page table and page_size=1 make every 512-subset a correct
    answer. The kernel's choice is valid, the trace checker accepts it and any
    other tied set through the page map, and still rejects a slot outside the
    request's pages or a selection below the K-th value; a non-identity table
    exercises the inversion, and a table padded past a shorter request's pages
    with a live page id inverts only the spanned pages."""
    if not flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()):
        pytest.skip("gvr_2 not supported here")
    from flashinfer.trace.templates.topk import _top_k_varlen_check as check

    rows, n, k = 2, 1024, 512
    lg = torch.ones((rows, n), dtype=torch.float32, device=_DEV)
    lens = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    pt = torch.arange(n, dtype=torch.int32, device=_DEV).repeat(rows, 1).contiguous()
    out, _ = flashinfer.top_k_varlen(
        lg, lens, k, page_table=pt, page_size=1, backend="gvr_2"
    )
    torch.cuda.synchronize()
    for r in range(rows):  # 512 distinct slots inside the row
        v = out[r][out[r] >= 0]
        assert v.numel() == k and v.unique().numel() == k and int(v.max()) < n
    kw = dict(logits=lg, seq_lens=lens, top_k=k, page_table=pt, page_size=1)
    lo = torch.arange(k, dtype=torch.int32, device=_DEV).repeat(rows, 1)
    hi = (torch.arange(k, dtype=torch.int32, device=_DEV) + k).repeat(rows, 1)
    assert check(
        lo, out, **kw
    )  # the kernel's tied set against a reference that chose 0..511
    assert check(lo, hi, **kw)  # two different tied sets are both correct
    bad = hi.clone()
    bad[0, 0] = n + 5  # a slot outside the request's pages
    assert not check(lo, bad, **kw)
    lg2 = lg.clone()
    lg2[:, k:] = 0.0  # now 0..511 are the only winners: 512..1023 is wrong
    assert not check(
        lo, hi, logits=lg2, seq_lens=lens, top_k=k, page_table=pt, page_size=1
    )
    # a non-identity page map with 64-column pages: the kernel's slots invert
    # back to columns through the map and pass the value check
    pt64 = _page_table(rows, n, 64, seed=5)
    out64, _ = flashinfer.top_k_varlen(
        lg, lens, k, page_table=pt64, page_size=64, backend="gvr_2"
    )
    torch.cuda.synchronize()
    assert check(
        out64, out64, logits=lg, seq_lens=lens, top_k=k, page_table=pt64, page_size=64
    )
    # a shorter request whose table is padded past its live pages with one of
    # them (serving tables carry such padding): only the spanned pages invert
    lens2 = torch.tensor([n, 640], dtype=torch.int32, device=_DEV)
    pt_pad = pt64.clone()
    pt_pad[1, 10:] = pt_pad[
        1, 0
    ]  # ceil(640 / 64) = 10 live pages, the rest repeat page 0
    out_pad, _ = flashinfer.top_k_varlen(
        lg, lens2, k, page_table=pt_pad, page_size=64, backend="gvr_2"
    )
    torch.cuda.synchronize()
    assert check(
        out_pad,
        out_pad,
        logits=lg,
        seq_lens=lens2,
        top_k=k,
        page_table=pt_pad,
        page_size=64,
    )


def test_paged_empty_row_tie_aware_check():
    """A request with seq_len 0 under paged output (reviewer's case: width 1024,
    K=512, identity table, page_size=1) spans no pages and comes back all -1.
    The trace checker accepts that row without touching the page map, and
    rejects a slot in it instead of raising on an empty reduction; the same
    through a random 64-column map."""
    if not flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()):
        pytest.skip("gvr_2 not supported here")
    from flashinfer.trace.templates.topk import _top_k_varlen_check as check

    rows, n, k = 2, 1024, 512
    lg = torch.randn((rows, n), dtype=torch.float32, device=_DEV)
    lens = torch.tensor(
        [0, n], dtype=torch.int32, device=_DEV
    )  # empty request, then a full one
    identity = (
        torch.arange(n, dtype=torch.int32, device=_DEV).repeat(rows, 1).contiguous()
    )
    for page_size, pt in ((1, identity), (64, _page_table(rows, n, 64, seed=7))):
        out, _ = flashinfer.top_k_varlen(
            lg, lens, k, page_table=pt, page_size=page_size, backend="gvr_2"
        )
        torch.cuda.synchronize()
        assert bool((out[0] == -1).all()), "an empty request must come back all -1"
        kw = dict(logits=lg, seq_lens=lens, top_k=k, page_table=pt, page_size=page_size)
        assert check(out, out, **kw)
        bad = out.clone()
        bad[0, 0] = 0  # a slot for a request that spans no pages
        assert not check(out, bad, **kw)
