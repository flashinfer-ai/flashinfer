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


def test_paged_api_validation():
    rows, ncols, k = 8, 4096, 512
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = torch.full((rows,), ncols, dtype=torch.int32, device=_DEV)
    pt = torch.zeros((rows, ncols // 64), dtype=torch.int32, device=_DEV)
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
    # combinations no backend admits: gvr_2's checker refuses them, so `auto`
    # reports no suitable backend; an explicit gvr_2 call raises at validation
    for kw in (
        dict(row_starts=torch.zeros(rows, dtype=torch.int32, device=_DEV)),
        dict(pre_idx=torch.zeros((rows, k), dtype=torch.int32, device=_DEV)),
    ):
        with pytest.raises((BackendSupportedError, ValueError)):
            flashinfer.top_k_varlen(lg, lens, k, page_table=pt, page_size=64, **kw)
    # other backends refuse it explicitly
    with pytest.raises((BackendSupportedError, ValueError)):
        flashinfer.top_k_varlen(
            lg, lens, k, backend="radix", page_table=pt, page_size=64
        )
