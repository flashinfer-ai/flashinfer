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
"""Serving-pattern harness for ``top_k_varlen``.

The shape-level suites check exactness one launch at a time. A DeepSeek-style
sparse-attention indexer in a serving engine (the SGLang DSA indexer is the
model here) drives the top-k differently, and several review findings on this
API were only visible under those patterns:

* several CUDA streams / CUDA graphs in flight on one device (the gvr_2
  default workspace is one slab per device);
* a top-k issued on a side stream with ``wait_stream`` hand-offs (dual-stream
  indexer overlap);
* CUDA graphs captured at a padded batch and replayed with fewer live rows,
  padded rows carrying a zero length and a static caller-owned output buffer
  whose padded rows must read back as ``-1``;
* the MLA "select all" path: scores of width ``top_k`` that are all zero, rows
  no longer than ``top_k``, expecting ``[0 .. len-1, -1, ...]``;
* the previous step's indices reused as the hint one token later;
* the engine's own top-k contract (mask beyond the row length, ``-1`` for
  masked picks) versus this API's contract for rows with fewer than
  ``top_k`` finite values.

Each test runs on every backend that the device and inputs admit.
"""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.topk_varlen import topk_varlen as _tv
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
_FMIN = -3.4028234663852886e38
_ALL = ("radix", "gvr", "gvr_2", "radix_cutlass", "radix_filter")
_CHECKERS = {
    "radix": "_radix_top_k_varlen_check",
    "gvr": "_gvr_top_k_varlen_check",
    "gvr_2": "_gvr2_top_k_varlen_check",
    "radix_cutlass": "_radix_cutlass_top_k_varlen_check",
    "radix_filter": "_radix_filter_top_k_varlen_check",
}


def _cc() -> int:
    major, minor = get_compute_capability(torch.device(_DEV))
    return major * 10 + minor


def _available(logits, seq_lens, top_k, pre_idx=None, **kw):
    """Backends whose checker admits this exact call on this device."""
    out = []
    for name in _ALL:
        if not flashinfer.top_k_varlen.is_backend_supported(name, _cc()):
            continue
        if getattr(_tv, _CHECKERS[name])(
            logits, seq_lens, top_k, pre_idx=pre_idx, **kw
        ):
            out.append(name)
    return out


def _hint(logits, top_k, seed):
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    return torch.randint(
        0,
        logits.shape[1],
        (logits.shape[0], top_k),
        generator=gen,
        device=_DEV,
        dtype=torch.int32,
    )


def _check_rows(logits, indices, lens, top_k, values=None, who=""):
    """Per-row oracle: short rows (len <= top_k) are identity + -1 tail; long
    rows are unique in-range indices whose value multiset equals torch.topk."""
    for r in range(indices.shape[0]):
        n = int(lens[r])
        row = indices[r].to(torch.int64)
        tag = f"{who} row {r} (len {n})"
        if n <= top_k:
            if n > 0:
                assert torch.equal(
                    torch.sort(row[:n]).values, torch.arange(n, device=_DEV)
                ), f"{tag}: short-row head is not a permutation of arange({n})"
            assert bool((row[n:] == -1).all()), f"{tag}: short-row tail not -1"
            if values is not None:
                assert bool((values[r, n:] == _FMIN).all()), f"{tag}: values tail"
            continue
        assert int(row.min()) >= 0 and int(row.max()) < n, (
            f"{tag}: index range (min {int(row.min())}, max {int(row.max())})"
        )
        assert int(torch.unique(row).numel()) == top_k, f"{tag}: duplicate indices"
        got = torch.sort(logits[r].gather(0, row) + 0.0, descending=True).values
        ref = torch.sort(
            torch.topk(logits[r, :n], top_k).values + 0.0, descending=True
        ).values
        assert torch.equal(got, ref), f"{tag}: value multiset differs from torch.topk"


# ---------------------------------------------------------------------------
# 1. concurrency on one device
# ---------------------------------------------------------------------------


def _split_case(streams, launches):
    """Inputs for the gvr_2 streaming SPLIT family (the only family that touches
    the workspace); skips if this device routes the shape elsewhere."""
    rows, n, k = 16, 131072, 512
    lc = _host._varlen_launcher(rows, n, k, n, 1, 1)
    if not (lc[0] == "main" and lc[2][5] > 1):
        pytest.skip(f"shape routes to {lc[0]} (no workspace use) on this device")
    gen = torch.Generator(device=_DEV).manual_seed(0)
    work = []
    for _ in range(streams):
        logits = torch.randn(rows, n, generator=gen, device=_DEV)
        seq = torch.randint(
            40000, n, (rows,), generator=gen, device=_DEV, dtype=torch.int32
        )
        pre = torch.full((rows, k), -1, dtype=torch.int32, device=_DEV)
        outs = [
            torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
            for _ in range(launches)
        ]
        ws = torch.zeros(_host.workspace_bytes(), dtype=torch.uint8, device=_DEV)
        work.append((logits, seq, pre, outs, ws))
    return rows, k, work


def _concurrent_graph_replays(private_workspace, streams=4, launches=8, rounds=6):
    rows, k, work = _split_case(streams, launches)
    cuda_streams = [torch.cuda.Stream() for _ in range(streams)]

    def launch(w, out):
        logits, seq, pre, _, ws = w
        flashinfer.top_k_varlen(
            logits,
            seq,
            k,
            pre_idx=pre,
            out_indices=out,
            backend="gvr_2",
            workspace={"gvr2_workspace": ws} if private_workspace else None,
        )

    for w in work:  # compile eagerly before capture
        launch(w, w[3][0])
    torch.cuda.synchronize()
    graphs = []
    for s, w in enumerate(work):
        g = torch.cuda.CUDAGraph()
        with (
            torch.cuda.stream(cuda_streams[s]),
            torch.cuda.graph(g, stream=cuda_streams[s]),
        ):
            for out in w[3]:
                launch(w, out)
        graphs.append(g)
    torch.cuda.synchronize()
    bad = 0
    for _ in range(rounds):
        for w in work:
            for out in w[3]:
                out.fill_(-7)
        torch.cuda.synchronize()
        for s, g in enumerate(graphs):
            with torch.cuda.stream(cuda_streams[s]):
                g.replay()
        torch.cuda.synchronize()
        for logits, seq, _, outs, _ in work:
            for out in outs:
                for r in range(rows):
                    idx = out[r].long()
                    n = int(seq[r])
                    ok = bool(((idx >= 0) & (idx < n)).all()) and torch.equal(
                        torch.sort(logits[r][idx]).values,
                        torch.sort(torch.topk(logits[r, :n], k).values).values,
                    )
                    bad += 0 if ok else 1
    return bad, rounds * streams * launches * rows


@pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE
    or not flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()),
    reason="gvr_2 unsupported on this device",
)
def test_gvr2_concurrent_graph_replays_private_workspaces():
    """The documented contract: every concurrently replayed graph carries its
    own gvr2_workspace, so four graphs of eight SPLIT launches each, replayed
    concurrently on four streams, stay exact."""
    bad, total = _concurrent_graph_replays(private_workspace=True)
    assert bad == 0, f"{bad}/{total} rows corrupted with private workspaces"


@pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE
    or not flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()),
    reason="gvr_2 unsupported on this device",
)
@pytest.mark.xfail(
    strict=False,
    reason="DOCUMENTED HAZARD: workspace=None resolves to one gvr_2 slab per "
    "device, so concurrently replayed graphs race on its counters and candidate "
    "buffer (measured 17/5120 corrupted rows on B100). Flips to XPASS if a "
    "per-stream/per-graph default ever lands; callers must pass a workspace.",
)
def test_gvr2_default_workspace_shared_across_concurrent_graphs():
    bad, total = _concurrent_graph_replays(private_workspace=False)
    assert bad == 0, f"{bad}/{total} rows corrupted through the shared default slab"


def test_side_stream_handoff_every_backend():
    """Dual-stream indexer pattern: the top-k is issued on a side stream after a
    wait_stream hand-off from the producer and the consumer waits on it again.
    One launch in flight at a time, so every backend must be exact with its
    default workspace."""
    rows, n, top_k = 8, 16384, 1024
    gen = torch.Generator(device=_DEV).manual_seed(21)
    logits = torch.randn(rows, n, generator=gen, device=_DEV)
    lens = torch.randint(
        top_k + 1, n, (rows,), generator=gen, device=_DEV, dtype=torch.int32
    )
    pre = _hint(logits, top_k, 22)
    side = torch.cuda.Stream()
    ran = []
    for backend in _available(logits, lens, top_k, pre_idx=pre):
        out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            flashinfer.top_k_varlen(
                logits, lens, top_k, pre_idx=pre, out_indices=out, backend=backend
            )
        main.wait_stream(side)
        consumed = out.clone()  # consumer on the main stream
        torch.cuda.synchronize()
        _check_rows(logits, consumed, lens.tolist(), top_k)
        ran.append(backend)
    assert ran, "no backend available"


# ---------------------------------------------------------------------------
# 2. CUDA graphs the way a serving engine replays them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["radix", "gvr", "gvr_2", "radix_cutlass"])
def test_graph_padded_rows_static_out_buffer(backend):
    """Capture at a padded batch with a caller-owned static output buffer, then
    replay with 8, 5, 2 and 8 live rows: padded rows carry seq_len 0 and must
    read back as -1 (the engine's sentinel), live rows stay exact, and the
    buffer identity never changes. Lengths also grow and shrink between
    replays."""
    if not flashinfer.top_k_varlen.is_backend_supported(backend, _cc()):
        pytest.skip(f"{backend} unsupported on this device")
    rows_pad, n, top_k = 8, 8192, 512
    gen = torch.Generator(device=_DEV).manual_seed(31)
    logits = torch.randn(rows_pad, n, generator=gen, device=_DEV)
    base = torch.randint(
        top_k + 1, n - 600, (rows_pad,), generator=gen, device=_DEV, dtype=torch.int32
    )
    seq_lens = base.clone()
    pre = _hint(logits, top_k, 32)
    if backend not in _available(logits, seq_lens, top_k, pre_idx=pre):
        pytest.skip(f"{backend} checker rejects this configuration here")
    out = torch.full((rows_pad, top_k), -7, dtype=torch.int32, device=_DEV)
    kw = dict(pre_idx=pre, out_indices=out, backend=backend)
    flashinfer.top_k_varlen(logits, seq_lens, top_k, **kw)  # warm-up / compile
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            ret, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, **kw)
    assert ret.data_ptr() == out.data_ptr()
    for live, delta in ((8, 0), (5, 300), (2, -250), (8, 500), (0, 0)):
        lens = base + delta
        lens[live:] = 0
        seq_lens.copy_(lens)
        out.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        _check_rows(logits, out, lens.tolist(), top_k)


# ---------------------------------------------------------------------------
# 3. indexer-specific inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [512, 1024, 2048])
def test_select_all_dummy_logits(top_k):
    """MLA path of the indexer: scores of width exactly top_k, all zero (every
    value tied), rows no longer than top_k. Every backend must return
    [0 .. len-1] followed by -1, for lengths 0, 1, top_k // 2 and top_k."""
    lens_list = [0, 1, top_k // 2, top_k]
    rows = len(lens_list)
    logits = torch.zeros(rows, top_k, device=_DEV)
    lens = torch.tensor(lens_list, dtype=torch.int32, device=_DEV)
    pre = torch.full((rows, top_k), -1, dtype=torch.int32, device=_DEV)
    ran = []
    for backend in _available(logits, lens, top_k, pre_idx=pre):
        out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
        flashinfer.top_k_varlen(
            logits, lens, top_k, pre_idx=pre, out_indices=out, backend=backend
        )
        torch.cuda.synchronize()
        _check_rows(logits, out, lens_list, top_k)
        ran.append(backend)
    assert ran


@pytest.mark.parametrize("backend", ["gvr", "gvr_2"])
def test_previous_step_indices_as_hint(backend):
    """Decode step t+1 reuses step t's indices as the hint while every request
    grew by one token and the scores changed: exactness must not depend on how
    stale the hint is."""
    if not flashinfer.top_k_varlen.is_backend_supported(backend, _cc()):
        pytest.skip(f"{backend} unsupported on this device")
    rows, n, top_k = 6, 32768, 1024
    gen = torch.Generator(device=_DEV).manual_seed(41)
    lens = torch.randint(
        top_k + 1, n - 8, (rows,), generator=gen, device=_DEV, dtype=torch.int32
    )
    logits = torch.randn(rows, n, generator=gen, device=_DEV)
    pre = _hint(logits, top_k, 42)
    if backend not in _available(logits, lens, top_k, pre_idx=pre):
        pytest.skip(f"{backend} checker rejects this configuration here")
    idx, _ = flashinfer.top_k_varlen(logits, lens, top_k, pre_idx=pre, backend=backend)
    _check_rows(logits, idx, lens.tolist(), top_k)
    for _ in range(4):
        lens = lens + 1
        logits = torch.randn(rows, n, generator=gen, device=_DEV)
        idx, _ = flashinfer.top_k_varlen(
            logits, lens, top_k, pre_idx=idx.contiguous(), backend=backend
        )
        torch.cuda.synchronize()
        _check_rows(logits, idx, lens.tolist(), top_k)


def _engine_reference(logits, lens, top_k):
    """The serving engine's unfused top-k contract (SGLang `_topk_unfused`):
    mask columns at or beyond the row length to -inf, torch.topk, and -1 for
    every pick whose score is -inf."""
    rows, n = logits.shape
    cols = torch.arange(n, device=_DEV).unsqueeze(0)
    masked = logits.masked_fill(cols >= lens.unsqueeze(1), float("-inf"))
    scores, idx = torch.topk(masked, top_k, dim=1)
    return idx.masked_fill(scores == float("-inf"), -1)


def test_engine_reference_parity_and_masked_token_boundary():
    """Decode-shaped batch compared with the engine's own reference. Rows with
    at least top_k finite scores must agree as value multisets. Rows where the
    engine masked tokens to -inf so that fewer than top_k finite scores remain
    expose an integration boundary: the engine's contract returns -1 for the
    -inf picks, this API returns top_k valid indices that include -inf columns
    (they are the largest remaining values). The finite picks must agree; an
    adapter that needs the engine's -1 semantics has to post-process, and this
    test pins both halves so the boundary stays visible."""
    rows, n, top_k = 6, 16384, 1024
    gen = torch.Generator(device=_DEV).manual_seed(51)
    logits = torch.randn(rows, n, generator=gen, device=_DEV)
    lens = torch.randint(
        top_k + 200, n, (rows,), generator=gen, device=_DEV, dtype=torch.int32
    )
    # engine-style masking of "init/local" tokens inside the window
    logits[:, :100] = float("-inf")
    # row 5: only columns 100..899 stay finite inside its 4096-wide window
    lens[5] = 4096
    logits[5, 900:] = float("-inf")
    n_finite_5 = int(torch.isfinite(logits[5, : int(lens[5])]).sum())
    assert n_finite_5 == 800 < top_k
    pre = _hint(logits, top_k, 52)
    ref = _engine_reference(logits, lens, top_k)
    ran = []
    for backend in _available(logits, lens, top_k, pre_idx=pre):
        out, _ = flashinfer.top_k_varlen(
            logits, lens, top_k, pre_idx=pre, backend=backend
        )
        torch.cuda.synchronize()
        for r in range(rows):
            n_r = int(lens[r])
            row = out[r].long()
            assert int(row.min()) >= 0 and int(row.max()) < n_r, (
                f"{backend} row {r}: range"
            )
            assert int(torch.unique(row).numel()) == top_k, f"{backend} row {r}: dups"
            got = logits[r].gather(0, row)
            if r < 5:
                exp = logits[r].gather(0, ref[r].long())
                assert torch.equal(
                    torch.sort(got, descending=True).values,
                    torch.sort(exp, descending=True).values,
                ), f"{backend} row {r}: differs from the engine reference"
            else:
                finite_got = set(row[torch.isfinite(got)].tolist())
                finite_ref = set(ref[r][ref[r] >= 0].tolist())
                assert finite_got == finite_ref, f"{backend} row 5: finite picks differ"
                assert (
                    int((ref[r] == -1).sum()) == top_k - n_finite_5
                )  # the engine pads with -1 ...
                assert bool((row >= 0).all())  # ... this API fills with -inf columns
        ran.append(backend)
    assert ran


# ---------------------------------------------------------------------------
# 4. metadata and scale the engine actually produces
# ---------------------------------------------------------------------------


def _check_long_rows_vectorized(logits, out, lens, top_k):
    """Batched oracle for rows that are all longer than top_k: unique in-range
    indices and a per-row value multiset equal to torch.topk over the valid
    prefix (columns at or beyond the row length masked to -inf)."""
    rows, n = logits.shape
    idx = out.long()
    lens_col = lens.to(torch.int64).unsqueeze(1)
    assert bool(((idx >= 0) & (idx < lens_col)).all()), "index out of range"
    assert bool(
        (
            torch.sort(idx, dim=1).values[:, 1:]
            != torch.sort(idx, dim=1).values[:, :-1]
        ).all()
    ), "duplicate indices"
    cols = torch.arange(n, device=_DEV).unsqueeze(0)
    masked = logits.masked_fill(cols >= lens_col, float("-inf"))
    ref = torch.sort(torch.topk(masked, top_k, dim=1).values, dim=1).values
    got = torch.sort(logits.gather(1, idx), dim=1).values
    assert torch.equal(got, ref), "value multiset differs from torch.topk"


@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize(
    "backend", ["radix", "gvr", "gvr_2", "radix_cutlass", "radix_filter"]
)
def test_unclipped_lengths(backend, next_n):
    """The engine hands the top-k its unclipped per-request cache lengths, which
    can exceed the width of the score buffer for that step (the dynamic length
    outgrows the static envelope, e.g. under CUDA-graph replay). Every backend
    must clamp each row to the buffer width, after the next_n row adjustment,
    and stay exact; rows within the width are unaffected. This harness found
    gvr (both load-balance modes) and radix_filter reading past the row here
    (compute-sanitizer: invalid global reads, indices >= width); both now
    clamp in the kernel."""
    if not flashinfer.top_k_varlen.is_backend_supported(backend, _cc()):
        pytest.skip(f"{backend} unsupported on this device")
    batch, n, top_k = 6, 8192, 1024
    rows = batch * next_n
    gen = torch.Generator(device=_DEV).manual_seed(61)
    logits = torch.randn(rows, n, generator=gen, device=_DEV)
    lens = torch.tensor(
        [100000, 8192, 8193, 5000, 200000, 1500], dtype=torch.int32, device=_DEV
    )
    pre = _hint(logits, top_k, 62)[:batch]
    if backend not in _available(logits, lens, top_k, pre_idx=pre, next_n=next_n):
        pytest.skip(f"{backend} checker rejects this configuration here")
    row_lens = [
        min(int(lens[r // next_n]) - next_n + (r % next_n) + 1, n) for r in range(rows)
    ]
    variants = (
        [{"load_balance": True}, {"load_balance": False}] if backend == "gvr" else [{}]
    )
    for kw in variants:
        out, _ = flashinfer.top_k_varlen(
            logits, lens, top_k, pre_idx=pre, next_n=next_n, backend=backend, **kw
        )
        torch.cuda.synchronize()
        _check_rows(logits, out, row_lens, top_k, who=f"{backend} {kw} next_n={next_n}")


@pytest.mark.parametrize("rows,n,top_k", [(1024, 8192, 512), (4096, 8192, 2048)])
def test_large_batch_every_backend(rows, n, top_k):
    """Serving batches reach thousands of requests (the engine's own kernel
    tests go to 4096 rows). Every backend that admits the shape must be exact
    there: this crosses gvr's load-balance cap and gvr_2's wide/narrow routing
    split (b > 148)."""
    gen = torch.Generator(device=_DEV).manual_seed(71)
    logits = torch.randn(rows, n, generator=gen, device=_DEV)
    lens = torch.randint(
        top_k + 1, n + 1, (rows,), generator=gen, device=_DEV, dtype=torch.int32
    )
    pre = _hint(logits, top_k, 72)
    ran = []
    for backend in _available(logits, lens, top_k, pre_idx=pre):
        out, _ = flashinfer.top_k_varlen(
            logits, lens, top_k, pre_idx=pre, backend=backend
        )
        torch.cuda.synchronize()
        _check_long_rows_vectorized(logits, out, lens, top_k)
        ran.append(backend)
    assert ran


def test_heavily_tied_scores_every_backend():
    """Quantized scores with a few distinct values (dense ties at the k-th
    value) must still give an exact value multiset on every backend; which of
    the tied columns is picked is unspecified."""
    rows, n, top_k = 8, 16384, 1024
    gen = torch.Generator(device=_DEV).manual_seed(81)
    logits = torch.randint(0, 16, (rows, n), generator=gen, device=_DEV).float()
    lens = torch.randint(
        top_k + 1, n + 1, (rows,), generator=gen, device=_DEV, dtype=torch.int32
    )
    pre = _hint(logits, top_k, 82)
    ran = []
    for backend in _available(logits, lens, top_k, pre_idx=pre):
        out, _ = flashinfer.top_k_varlen(
            logits, lens, top_k, pre_idx=pre, backend=backend
        )
        torch.cuda.synchronize()
        _check_long_rows_vectorized(logits, out, lens, top_k)
        ran.append(backend)
    assert ran
