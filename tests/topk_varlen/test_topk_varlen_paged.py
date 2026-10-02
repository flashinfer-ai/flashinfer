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
"""Paged output of ``top_k_varlen`` (decode mode; backends ``walkfirst_primitives``,
``sglang`` and ``radix_primitives``): every selected column ``c`` of request
``q`` is returned as the physical KV slot
``page_table[q, c // page_size] * page_size + c % page_size``, fused into the
kernel's one launch (the SGLang DSA decode contract; the same mapping as
``top_k_page_table_transform``).

Contract under test: the paged output is an exact top-k of every row in physical
space -- mapped back through the row's (injective) table, the slots are valid,
distinct columns inside the row length whose values are the row's top-k value
multiset, and the -1 padding sits where the raw output pads.  Index SETS are
never compared between runs: under ties any subset of the tied columns is exact
and a kernel's pick is not fixed.  Covered: every arm of the walk-first kernel
(identity rows with length <= k, the census arm <= 4K, the register arm 4K-16K,
the single-CTA walk, the cluster epilogue, the gmem last-arriver epilogue) and
SGLang's Register2 / Register4 / Streaming families, every tie-resolution tail
(warp-ballot select on few ties, byte-radix select over hundreds of distinct
staged keys, the uniform-tie copy, walk-first's exact fallback on a flooded bin),
page sizes 1 / 64 / 4096, a table wider or narrower than the width, grouped rows
(``next_n > 1``) and compressed columns (also together, with token counts that
are not multiples of the ratio) for the grouped backends, CUDA-graph replay
with a table change, the empty batch, and the API admission rules.
"""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.cute_dsl.utils import is_cute_dsl_available
    from flashinfer.utils import BackendSupportedError, get_compute_capability

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (_FLASHINFER_AVAILABLE and torch.cuda.is_available()),
    reason="flashinfer + CUDA required",
)
_DEV = "cuda"
_BACKENDS = ["walkfirst_primitives", "sglang", "radix_primitives"]
# backends that accept next_n > 1 / compress_ratio > 1 (walk-first is next_n == 1)
_GROUPED_BACKENDS = ["sglang", "radix_primitives"]


def _cc() -> int:
    major, minor = get_compute_capability(torch.device(_DEV))
    return major * 10 + minor


def _require(backend):
    if not flashinfer.top_k_varlen.is_backend_supported(backend, _cc()):
        pytest.skip(f"{backend} unsupported on this device")
    if backend != "sglang" and not is_cute_dsl_available():  # CuTe-DSL kernels
        pytest.skip(f"{backend} needs nvidia-cutlass-dsl")


def _page_table(batch, ncols, page_size, seed):
    """One distinct random physical page per (request, page): injective per row,
    so a physical slot maps back to exactly one column."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    pages = (ncols + page_size - 1) // page_size
    pool = 4 * pages + 7
    pt = torch.stack(
        [torch.randperm(pool, generator=gen, device=_DEV)[:pages] for _ in range(batch)]
    ).to(torch.int32)
    return pt.contiguous()


def _unmap(phys, pt, page_size):
    """Physical slots -> columns through each row's injective table.  -1 stays
    -1; a slot whose page is not in the row's table becomes -2."""
    rows, pages = pt.shape
    shift = page_size.bit_length() - 1
    pool = int(pt.max()) + 1
    inv = torch.full((rows, pool), -1, dtype=torch.int64, device=pt.device)
    inv.scatter_(
        1, pt.long(), torch.arange(pages, device=pt.device).expand(rows, pages)
    )
    p64 = phys.to(torch.int64)
    pidx = p64 >> shift
    # a page outside the pool is flagged, not clamped onto a real page (the
    # clamp mapped a stray slot to a plausible column of the last page)
    in_pool = (pidx >= 0) & (pidx < pool)
    page = inv.gather(1, torch.where(in_pool, pidx, torch.zeros_like(pidx)))
    page = torch.where(in_pool, page, torch.full_like(page, -1))
    col = (page << shift) | (p64 & (page_size - 1))
    col = torch.where(page < 0, torch.full_like(col, -2), col)
    return torch.where(phys >= 0, col, p64)


def _check_exact_paged(lg, lens, top_k, pg, pt, page_size):
    """The paged output is an exact top-k of every row in physical space.
    ``lens`` are the per-ROW effective lengths; ``pt`` has one row per output
    row (expand a per-request table with ``repeat_interleave`` first)."""
    cols = _unmap(pg, pt, page_size)
    for r in range(lg.shape[0]):
        L = int(lens[r])
        kk = min(top_k, L)
        row = pg[r]
        assert int((cols[r] == -2).sum()) == 0, (r, L, "slot outside the table")
        assert bool((row[:kk] >= 0).all()) and bool((row[kk:] == -1).all()), (
            r,
            L,
            "pads",
        )
        if kk == 0:
            continue
        valid = cols[r][:kk]
        assert int(valid.max()) < L and valid.unique().numel() == kk, (r, L, "columns")
        got = torch.sort(lg[r, :L][valid], descending=True).values
        assert torch.equal(got, torch.topk(lg[r, :L], kk).values), (r, L, "values")


def _mixed_lens(batch, ncols, top_k, seed):
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    lens = torch.randint(top_k + 1, ncols + 1, (batch,), generator=gen, device=_DEV)
    # short / boundary / empty requests (identity path, -1 padding)
    if batch >= 4:
        lens[0] = top_k // 2
        lens[1] = top_k
        lens[2] = 0
        lens[3] = top_k + 1
    return lens.to(torch.int32)


def _logits(pattern, rows, ncols, seed):
    """Row patterns that steer the crossing bin into each tie-resolution tail."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    x = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV, generator=gen)
    if pattern == "randn":
        return x  # tens of ties in the crossing bin: warp-ballot select
    if pattern == "narrow":
        # 600 distinct values inside ONE coarse bin ([5.0, 5.0625) for fp32),
        # all ranked above the Gaussian bulk: byte-radix select over the ties
        x[:, :600] = 5.0 + torch.arange(600, device=_DEV, dtype=torch.float32) * 1e-4
        return x
    if pattern == "dups":
        x[:, ::2] = 3.0  # half the row is one repeated key: uniform-tie copy
        return x
    if pattern == "onebin":
        # the whole row inside one coarse bin (2048 distinct fp32 values, each
        # repeated): walk-first's exact fallback; SGLang truncates here
        return 1.0 + torch.rand((rows, ncols), device=_DEV, generator=gen) * 2.0**-12
    raise ValueError(pattern)


def _paged(backend, lg, lens, top_k, pt, page_size, **kw):
    pg, _ = flashinfer.top_k_varlen(
        lg, lens, top_k, backend=backend, page_table=pt, page_size=page_size, **kw
    )
    torch.cuda.synchronize()
    return pg


# rows x columns shapes spanning the kernels' arms: 16K (identity, census and
# register arms / Register2-4 under mixed lengths), 64K (single-CTA walk /
# Streaming), 256K x 16 rows (cluster epilogue on SM90+), 1M x 1 row (gmem
# last-arriver epilogue)
_SHAPES = [(8, 16384), (8, 65536), (16, 262144), (1, 1048576)]


@pytest.mark.parametrize("backend", ["radix_primitives", "walkfirst_primitives"])
def test_paged_signed_zero_crossing(backend):
    """Regression: the rank-k value is -0.0 while +0.0 values rank above it.

    radix_primitives' fp32 float-boundary collect tested ``v <= T`` with T =
    -0.0 (the ordered-key predecessor of the +0.0 bin's bound), which holds
    for +0.0 as well, while the fp16 coarse histogram kept the two zeros in
    different bins: every +0.0 winner vanished and the row came back
    underfilled.  Both zeros are one bin and one key now, and a zero
    threshold steps down to the largest negative float.
    """
    _require(backend)
    rows, ncols, top_k, page_size = 6, 16384, 2048, 64
    gen = torch.Generator(device=_DEV).manual_seed(5)
    lg = torch.relu(torch.randn((rows, ncols), device=_DEV, generator=gen) - 1.2816)
    lg[:, 1::2] = lg[:, 1::2] * -1.0  # odd columns: -0.0 zeros and negatives
    lens = torch.tensor(
        [3000, 2600, 4096, 8192, 12000, 16384], dtype=torch.int32, device=_DEV
    )
    pt = _page_table(rows, ncols, page_size, seed=41)
    pg = _paged(backend, lg, lens, top_k, pt, page_size)
    _check_exact_paged(lg, lens.tolist(), top_k, pg, pt, page_size)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"r{s[0]}n{s[1]}")
@pytest.mark.parametrize("top_k", [512, 2048], ids=lambda k: f"k{k}")
def test_paged_exact_in_physical_space(backend, top_k, shape):
    _require(backend)
    rows, ncols = shape
    page_size = 64
    lg = _logits("randn", rows, ncols, seed=rows + ncols)
    lens = _mixed_lens(rows, ncols, top_k, seed=rows + ncols)
    pt = _page_table(rows, ncols, page_size, seed=7)
    pg = _paged(backend, lg, lens, top_k, pt, page_size)
    _check_exact_paged(lg, lens, top_k, pg, pt, page_size)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("pattern", ["narrow", "dups", "onebin"])
@pytest.mark.parametrize("ncols", [4096, 16384, 65536], ids=lambda n: f"n{n}")
def test_paged_tie_tails(backend, pattern, ncols):
    """The tails that re-map after an inherited helper (byte-radix select,
    exact fallback) and the uniform copy, on every short-row arm and the
    single-CTA walk.  ``onebin`` (thousands of distinct keys in one coarse
    bin) and ``dups`` (a flood of one key sharing its coarse bin with a few
    larger Gaussian values) are exact for walk-first only: SGLang keeps the
    first 2048 arrivals of a boundary bin and may drop one of the larger
    values -- its documented approximation."""
    _require(backend)
    if backend == "sglang" and pattern in ("onebin", "dups"):
        pytest.skip("sglang truncates boundary bins with > 2048 candidates")
    rows, top_k = 6, 512
    lg = _logits(pattern, rows, ncols, seed=ncols)
    lens = torch.full((rows,), ncols, dtype=torch.int32, device=_DEV)
    lens[1] = ncols - 5  # a scalar tail (< one vector) at the row end
    lens[2] = 3000
    pt = _page_table(rows, ncols, 64, seed=11)
    _check_exact_paged(
        lg, lens, top_k, _paged(backend, lg, lens, top_k, pt, 64), pt, 64
    )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("page_size", [1, 4096], ids=lambda p: f"ps{p}")
def test_paged_page_sizes(backend, page_size):
    _require(backend)
    rows, ncols, top_k = 16, 16384, 512
    lg = _logits("randn", rows, ncols, seed=3)
    lens = _mixed_lens(rows, ncols, top_k, seed=3)
    pt = _page_table(rows, ncols, page_size, seed=11)
    pg = _paged(backend, lg, lens, top_k, pt, page_size)
    _check_exact_paged(lg, lens, top_k, pg, pt, page_size)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_paged_wider_table_than_width(backend):
    """The table may cover more pages than the logits width needs (a KV pool
    sized for the model's max context): only the leading pages are read."""
    _require(backend)
    rows, ncols, top_k, page_size = 8, 16384, 512, 64
    lg = _logits("randn", rows, ncols, seed=5)
    lens = _mixed_lens(rows, ncols, top_k, seed=5)
    pt = _page_table(rows, 4 * ncols, page_size, seed=13)
    pg = _paged(backend, lg, lens, top_k, pt, page_size)
    _check_exact_paged(lg, lens, top_k, pg, pt, page_size)


@pytest.mark.parametrize("backend", _GROUPED_BACKENDS)
def test_paged_grouped_rows(backend):
    """``next_n`` rows per request all map through the request's page-table row;
    row r of request q sees (seq_lens[q] - next_n + r % next_n + 1) tokens."""
    _require(backend)
    batch, next_n, ncols, top_k, page_size = 12, 3, 8192, 1024, 64
    rows = batch * next_n
    lg = _logits("randn", rows, ncols, seed=17)
    gen = torch.Generator(device=_DEV).manual_seed(18)
    seq = torch.randint(top_k + 1, ncols + 1, (batch,), generator=gen, device=_DEV)
    seq[0] = top_k // 2  # every row of this request empty or trivial
    seq[1] = top_k + next_n
    seq = seq.to(torch.int32)
    pt = _page_table(batch, ncols, page_size, seed=19)
    pg = _paged(backend, lg, seq, top_k, pt, page_size, next_n=next_n)
    t = torch.arange(rows, device=_DEV) % next_n
    row_lens = (seq.long().repeat_interleave(next_n) - next_n + t + 1).clamp(0, ncols)
    _check_exact_paged(
        lg, row_lens, top_k, pg, pt.repeat_interleave(next_n, dim=0), page_size
    )


@pytest.mark.parametrize("backend", _GROUPED_BACKENDS)
def test_paged_compressed_columns(backend):
    """``compress_ratio``: columns and ``page_size`` are in compressed units;
    ``seq_lens`` are in tokens."""
    _require(backend)
    rows, ncols, top_k, cr, page_size = 8, 4096, 512, 4, 64
    lg = _logits("randn", rows, ncols, seed=23)
    gen = torch.Generator(device=_DEV).manual_seed(29)
    seq = torch.randint(
        (top_k + 1) * cr, ncols * cr + 1, (rows,), generator=gen, device=_DEV
    )
    seq = seq.to(torch.int32)
    pt = _page_table(rows, ncols, page_size, seed=31)
    pg = _paged(backend, lg, seq, top_k, pt, page_size, compress_ratio=cr)
    row_lens = (seq.long() // cr).clamp(0, ncols)
    _check_exact_paged(lg, row_lens, top_k, pg, pt, page_size)


@pytest.mark.parametrize("backend", _GROUPED_BACKENDS)
def test_paged_grouped_compressed_rows(backend):
    """``next_n`` and ``compress_ratio`` together, with token counts that are
    not multiples of the ratio: row t of request q ranks exactly
    max(0, (seq_lens[q] - next_n + t + 1) // cr) compressed columns (adjust by
    next_n first, then floor-divide; negative numerators are empty rows), all
    through the request's one table row."""
    _require(backend)
    batch, next_n, cr, ncols, top_k, page_size = 8, 3, 4, 4096, 512, 64
    rows = batch * next_n
    lg = _logits("randn", rows, ncols, seed=61)
    seq = torch.tensor(
        [0, 1, 5, 4 * 2000 + 1, 4 * top_k + 1, 4 * top_k + 6, 4 * ncols - 3, 4 * ncols],
        dtype=torch.int32,
        device=_DEV,
    )
    pt = _page_table(batch, ncols, page_size, seed=67)
    pg = _paged(
        backend, lg, seq, top_k, pt, page_size, next_n=next_n, compress_ratio=cr
    )
    t = torch.arange(rows, device=_DEV) % next_n
    numer = seq.long().repeat_interleave(next_n) - next_n + t + 1
    row_lens = torch.div(numer, cr, rounding_mode="floor").clamp(0, ncols)
    _check_exact_paged(
        lg, row_lens, top_k, pg, pt.repeat_interleave(next_n, dim=0), page_size
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_paged_empty_batch(backend):
    """B == 0 with a (0, pages) table: a (0, top_k) int32 result and no
    zero-block launch."""
    _require(backend)
    lg = torch.empty((0, 4096), dtype=torch.float32, device=_DEV)
    lens = torch.empty((0,), dtype=torch.int32, device=_DEV)
    pt = torch.empty((0, 64), dtype=torch.int32, device=_DEV)
    pg = _paged(backend, lg, lens, 512, pt, 64)
    assert tuple(pg.shape) == (0, 512) and pg.dtype == torch.int32


@pytest.mark.parametrize("backend", _BACKENDS)
def test_paged_cuda_graph_replay(backend):
    _require(backend)
    rows, ncols, top_k, page_size = 32, 16384, 512, 64
    lg = _logits("randn", rows, ncols, seed=29)
    lens = torch.full((rows,), ncols, dtype=torch.int32, device=_DEV)
    pt = _page_table(rows, ncols, page_size, seed=29)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    kw = dict(out_indices=out, backend=backend, page_table=pt, page_size=page_size)
    s = torch.cuda.Stream()
    s.wait_stream(
        torch.cuda.current_stream()
    )  # the inputs come from the default stream
    with torch.cuda.stream(s):  # warm-up: compiles the paged variant
        flashinfer.top_k_varlen(lg, lens, top_k, **kw)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(lg, lens, top_k, **kw)
    torch.cuda.current_stream().wait_stream(s)
    # new scores, lengths AND page table between replays
    lg.normal_()
    lens.copy_(_mixed_lens(rows, ncols, top_k, seed=31))
    pt.copy_(_page_table(rows, ncols, page_size, seed=37))
    g.replay()
    torch.cuda.synchronize()
    _check_exact_paged(lg, lens, top_k, out, pt, page_size)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_paged_narrower_table_than_width(backend):
    """SGLang's shape: the table covers the batch's longest row (129 pages of 64
    = 8256 columns) while the logits buffer is padded wider (8448 columns).
    Rows never exceed the covered prefix, so the result is the exact top-k;
    the kernels must not read the table past its row."""
    _require(backend)
    rows, ncols, top_k, page_size, pages = 8, 8448, 512, 64, 129
    lg = _logits("randn", rows, ncols, seed=43)
    lens = _mixed_lens(rows, pages * page_size, top_k, seed=43)  # <= 8256
    pt = _page_table(rows, pages * page_size, page_size, seed=47)  # 129 pages
    assert pt.shape[1] == pages < (ncols + page_size - 1) // page_size
    pg = _paged(backend, lg, lens, top_k, pt, page_size)
    _check_exact_paged(lg, lens, top_k, pg, pt, page_size)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_paged_row_beyond_table_coverage(backend):
    """A row longer than the pages the table covers is clamped to the covered
    prefix: the exact top-k of that prefix, never a page the table lacks."""
    _require(backend)
    rows, ncols, top_k, page_size, pages = 8, 16384, 512, 64, 10
    lg = _logits("randn", rows, ncols, seed=53)
    lens = torch.full((rows,), ncols, dtype=torch.int32, device=_DEV)
    lens[0] = 300  # inside the covered prefix and shorter than k
    lens[1] = pages * page_size  # exactly the coverage
    pt = _page_table(rows, pages * page_size, page_size, seed=59)
    pg = _paged(backend, lg, lens, top_k, pt, page_size)
    _check_exact_paged(lg, lens.clamp(max=pages * page_size), top_k, pg, pt, page_size)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_paged_api_validation(backend):
    _require(backend)
    rows, ncols, k = 8, 16384, 512
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = torch.full((rows,), ncols, dtype=torch.int32, device=_DEV)
    pages = ncols // 64
    pt = torch.zeros((rows, pages), dtype=torch.int32, device=_DEV)
    # a contiguous table whose base pointer is 4 bytes off the 16-byte grain
    # the DSL kernels assume
    shifted = torch.zeros(rows * pages + 4, dtype=torch.int32, device=_DEV)
    shifted = shifted[1 : 1 + rows * pages].view(rows, pages)
    assert shifted.is_contiguous() and shifted.data_ptr() & 15
    # malformed paged arguments: the API body's ValueError
    for kw in (
        dict(page_table=pt, page_size=48),  # not a power of two
        dict(page_table=pt[:4], page_size=64),  # wrong request count
        dict(page_table=pt.to(torch.int64), page_size=64),
        dict(page_table=pt[:, :0], page_size=64),  # no pages at all
        dict(page_table=pt.cpu(), page_size=64),  # not on the logits device
        dict(page_table=pt[:, ::2], page_size=64),  # non-contiguous
        dict(page_table=pt.view(-1), page_size=64),  # 1-D
        dict(page_table=pt, page_size=True),  # bool is not a page size
        dict(page_table=pt, page_size=0),
        dict(page_table=pt, page_size=1 << 31),
        dict(page_table=shifted, page_size=64),  # 16-byte misaligned
        # the covered width must fit int32: 4 pages * 2**30 overflows the
        # kernels' Int32 coverage clamp (rows would silently come back all -1)
        dict(page_table=pt[:, :4].contiguous(), page_size=1 << 30),
    ):
        with pytest.raises(ValueError, match="page_table|page_size"):
            flashinfer.top_k_varlen(lg, lens, k, backend=backend, **kw)
    # combinations no backend admits (the checkers refuse them first; an
    # explicit call raises at validation either way)
    for kw in (
        dict(page_table=pt, page_size=64, return_values=True),
        dict(
            page_table=pt,
            page_size=64,
            pre_idx=torch.zeros((rows, k), dtype=torch.int32, device=_DEV),
        ),
    ):
        with pytest.raises((BackendSupportedError, ValueError)):
            flashinfer.top_k_varlen(lg, lens, k, backend=backend, **kw)


def test_paged_refused_elsewhere():
    rows, ncols, k = 8, 16384, 512
    lg = torch.randn((rows, ncols), dtype=torch.float32, device=_DEV)
    lens = torch.full((rows,), ncols, dtype=torch.int32, device=_DEV)
    pt = torch.zeros((rows, ncols // 64), dtype=torch.int32, device=_DEV)
    # other backends refuse paged output explicitly (radix_cutlass runs on
    # every GPU, so the refusal itself is what fails here, not a CC gate)
    with pytest.raises((BackendSupportedError, ValueError)):
        flashinfer.top_k_varlen(
            lg, lens, k, backend="radix_cutlass", page_table=pt, page_size=64
        )
    # the vectorized kernels need rows of whole 16-byte vectors (fp32: N % 4)
    n_odd = 16382
    lg_odd = torch.randn((rows, n_odd), dtype=torch.float32, device=_DEV)
    lens_odd = torch.full((rows,), n_odd, dtype=torch.int32, device=_DEV)
    pt_odd = torch.zeros((rows, -(-n_odd // 64)), dtype=torch.int32, device=_DEV)
    for backend in ("walkfirst_primitives", "sglang"):
        if not flashinfer.top_k_varlen.is_backend_supported(backend, _cc()):
            continue
        with pytest.raises(
            (BackendSupportedError, ValueError), match="Problem size|not support"
        ):
            flashinfer.top_k_varlen(
                lg_odd, lens_odd, k, backend=backend, page_table=pt_odd, page_size=64
            )
    # walk-first serves next_n == 1 only (its table is indexed by row)
    if flashinfer.top_k_varlen.is_backend_supported("walkfirst_primitives", _cc()):
        with pytest.raises((BackendSupportedError, ValueError)):
            flashinfer.top_k_varlen(
                lg,
                lens[: rows // 2],
                k,
                next_n=2,
                backend="walkfirst_primitives",
                page_table=pt[: rows // 2],
                page_size=64,
            )


def test_paged_auto_routes_to_primitives():
    """``backend="auto"`` with a page table must pick a paged-capable backend
    (walk-first first, radix_primitives behind it) instead of raising: the
    primitives backends are explicit-only in the unpaged ranking."""
    _require("walkfirst_primitives")
    batch, ncols, top_k, page_size = 8, 4096, 512, 64
    lg = torch.randn(batch, ncols, device=_DEV)
    lens = _mixed_lens(batch, ncols, top_k, seed=11)
    pt = _page_table(batch, ncols, page_size, seed=3)
    pg, _ = flashinfer.top_k_varlen(lg, lens, top_k, page_table=pt, page_size=page_size)
    _check_exact_paged(lg, lens, top_k, pg, pt, page_size)
