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
"""Adversarial concurrency tests for ``top_k_varlen`` (gvr_2-centred).

The shape suites run one launch at a time on one stream from one thread, so
they cannot see the class of defect that review round 1 of PR #4986 found:
lazily created resources shared across launches that may overlap (the gvr_2
SPLIT workspace slab and the hint-free anchor table). Every test here drives
the API the way a serving engine does — several host threads, several CUDA
streams, CUDA graphs captured and replayed while other work is in flight —
and asserts exactness per row, so a race shows up as a wrong result rather
than as a note in a docstring.

Inputs are chosen so gvr_2 lands on the streaming ``main`` family with a
multi-CTA SPLIT (the only family that touches the slab); tests skip on parts
that route the shape elsewhere.
"""

import faulthandler
import functools
import sys
import threading
import time

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


def _barrier(parties):
    """Barrier with a timeout: a thread that fails before reaching it must not
    hang the whole test (the others then see BrokenBarrierError and report)."""
    return threading.Barrier(parties, timeout=120)


def _forget_table(k):
    """Drop the cached hint-free anchor table for k (cold start for a test)."""
    key = (torch.cuda.current_device(), k)
    lock = getattr(_host, "_HINT_FREE_LOCK", None) or threading.Lock()
    with lock:
        _host._HINT_FREE.pop(key, None)


def _cc() -> int:
    major, minor = get_compute_capability(torch.device(_DEV))
    return major * 10 + minor


requires_gvr2 = pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE
    or not torch.cuda.is_available()
    or not flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()),
    reason="gvr_2 unsupported on this device",
)


# ---------------------------------------------------------------------------
# deadline: a hung concurrency test must fail, not stall the suite
# ---------------------------------------------------------------------------

_CALIBRATION = {}


def _per_launch_seconds():
    """Wall time of one eager slab-family launch + sync on this device (warm),
    measured once per process; the deadline budgets scale with it so a slow
    part or a cold JIT cache does not produce false timeouts."""
    if "t" not in _CALIBRATION:
        rows, n, k = 16, 131072, 512
        gen = torch.Generator(device=_DEV).manual_seed(999)
        logits = torch.randn(rows, n, generator=gen, device=_DEV)
        seq = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
        pre = torch.full((rows, k), -1, dtype=torch.int32, device=_DEV)
        out = torch.empty(rows, k, dtype=torch.int32, device=_DEV)
        for _ in range(3):  # compile + warm
            flashinfer.top_k_varlen(
                logits, seq, k, pre_idx=pre, out_indices=out, backend="gvr_2"
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            flashinfer.top_k_varlen(
                logits, seq, k, pre_idx=pre, out_indices=out, backend="gvr_2"
            )
        torch.cuda.synchronize()
        _CALIBRATION["t"] = max((time.perf_counter() - t0) / 20, 1e-4)
    return _CALIBRATION["t"]


def _deadline(launches, slack_s=60.0, factor=200.0):
    """Run the test body in a worker thread and fail if it exceeds
    ``slack_s + factor * launches * per_launch_seconds`` — generous enough
    (200x the measured eager launch cost per launch, plus a minute for JIT,
    checks and graph capture) that a healthy run never trips it, small enough
    that a deadlock surfaces as a failure with every thread's stack printed.
    The body runs in a thread so the main thread can enforce the budget; CUDA
    graph capture is per-thread and unaffected."""

    def wrap(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            budget = slack_s + factor * launches * _per_launch_seconds()
            result = {}

            def body():
                try:
                    fn(*args, **kwargs)
                except BaseException as e:  # noqa: BLE001
                    result["error"] = e

            t = threading.Thread(target=body, name=f"{fn.__name__}-body", daemon=True)
            t.start()
            t.join(budget)
            if t.is_alive():
                sys.stderr.write(
                    f"\n=== {fn.__name__}: deadline {budget:.0f}s exceeded; thread stacks: ===\n"
                )
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                pytest.fail(
                    f"{fn.__name__} exceeded its deadline of {budget:.0f}s "
                    f"({launches} launches x {_per_launch_seconds() * 1e3:.2f} ms + {slack_s:.0f}s slack): probable hang"
                )
            if "error" in result:
                raise result["error"]

        return run

    return wrap


def _exact(logits, seq, out, k, who=""):
    for r in range(out.shape[0]):
        idx = out[r].long()
        n = int(seq[r])
        assert bool(((idx >= 0) & (idx < n)).all()), f"{who} row {r}: index range"
        assert torch.equal(
            torch.sort(logits[r][idx]).values,
            torch.sort(torch.topk(logits[r, :n], k).values).values,
        ), f"{who} row {r}: value multiset differs from torch.topk"


def _split_inputs(copies, rows=16, n=131072, k=512, seed=0):
    """`copies` independent problems that gvr_2 routes to the SPLIT slab family
    on this device (skip otherwise); each gets its own output buffer."""
    lc = _host._varlen_launcher(rows, n, k, n, 1, 1)
    if not (lc[0] == "main" and lc[2][5] > 1):
        pytest.skip(f"shape routes to {lc[0]} (no workspace use) on this device")
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    work = []
    for _ in range(copies):
        logits = torch.randn(rows, n, generator=gen, device=_DEV)
        seq = torch.randint(
            40000, n, (rows,), generator=gen, device=_DEV, dtype=torch.int32
        )
        pre = torch.full((rows, k), -1, dtype=torch.int32, device=_DEV)
        out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
        work.append((logits, seq, pre, out))
    return k, work


def _launch(w, k, hint=True, out=None, workspace=None):
    logits, seq, pre, o = w
    flashinfer.top_k_varlen(
        logits,
        seq,
        k,
        pre_idx=pre if hint else None,
        out_indices=out if out is not None else o,
        backend="gvr_2",
        workspace=workspace,
    )


# ---------------------------------------------------------------------------
# 1. eager launches overlapping across streams and host threads
# ---------------------------------------------------------------------------


@requires_gvr2
@pytest.mark.parametrize("hint", [True, False], ids=["hinted", "hint_free"])
@_deadline(launches=328)
def test_gvr2_eager_slab_launches_overlap_across_threads_and_streams(hint):
    """Eight host threads, each on its own stream, hammer the SPLIT family with
    workspace=None for 40 launches each while the others do the same: the
    host-thread-safety exercise of the per-stream slab lookup/creation path
    (GIL-serialized launches give little GPU overlap; the graph tests below
    cover that). Every row must be exact and no launch may raise."""
    threads_n, launches = 8, 40
    k, work = _split_inputs(threads_n, seed=11)
    streams = [torch.cuda.Stream() for _ in range(threads_n)]
    outs = [
        [torch.full_like(w[3], -7) for _ in range(launches)] for w in work
    ]  # one buffer per launch so every result is checked
    barrier = _barrier(threads_n)
    errors = []

    def worker(i):
        try:
            with torch.cuda.stream(streams[i]):
                _launch(work[i], k, hint)  # compile / create this stream's slab
                barrier.wait()
                for j in range(launches):
                    _launch(work[i], k, hint, out=outs[i][j])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    torch.cuda.synchronize()
    assert not errors, errors
    for i, w in enumerate(work):
        for j in range(launches):
            _exact(w[0], w[1], outs[i][j], k, who=f"stream {i} launch {j}")


@requires_gvr2
@_deadline(launches=524)
def test_gvr2_graph_replay_races_eager_launches_on_other_streams():
    """Four graphs (eight slab-using launches each, one output buffer per
    launch) captured on streams A..D are replayed back to back while stream E
    receives eight eager slab-using launches (default workspace), all issued
    from ONE host thread with no synchronization until the round ends —
    captured decode steps overlapping with eager work. Issuing from one thread
    is what produces real GPU-side overlap (Python threads serialize on the
    GIL); the four-graph replay is the pattern that measured 2-17 corrupted
    rows per run through a device-wide slab. Each stream owns a slab, so
    every replayed and eager result stays exact."""
    n_graphs, launches, rounds = 4, 8, 12
    k, work = _split_inputs(n_graphs + 1, seed=21)
    streams = [torch.cuda.Stream() for _ in range(n_graphs + 1)]
    outs = [[torch.full_like(w[3], -7) for _ in range(launches)] for w in work]
    graphs = []
    for i in range(n_graphs):
        with torch.cuda.stream(streams[i]):
            _launch(work[i], k)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(streams[i]), torch.cuda.graph(g, stream=streams[i]):
            for o in outs[i]:
                _launch(work[i], k, out=o)
        graphs.append(g)
    ew = work[n_graphs]
    with torch.cuda.stream(streams[n_graphs]):
        _launch(ew, k)  # slab for the eager stream
    torch.cuda.synchronize()
    for _ in range(rounds):
        for os_ in outs:
            for o in os_:
                o.fill_(-7)
        torch.cuda.synchronize()
        for i in range(n_graphs):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
        with torch.cuda.stream(streams[n_graphs]):
            for o in outs[n_graphs]:
                _launch(ew, k, out=o)
        torch.cuda.synchronize()
        for i, w in enumerate(work):
            for j, o in enumerate(outs[i]):
                who = f"graph {i} launch {j}" if i < n_graphs else f"eager launch {j}"
                _exact(w[0], w[1], o, k, who=who)


@requires_gvr2
@_deadline(launches=84)
def test_gvr2_graphs_captured_concurrently_from_threads_replay_exact():
    """Four threads capture graphs on four streams at the same time (a serving
    engine capturing per-batch-size graphs in parallel), after a per-stream
    eager warm-up; every graph replays exact, including when all four replay
    together."""
    k, work = _split_inputs(4, seed=31)
    streams = [torch.cuda.Stream() for _ in range(4)]
    for i in range(4):
        with torch.cuda.stream(streams[i]):
            _launch(work[i], k)
    torch.cuda.synchronize()
    graphs = [None] * 4
    barrier = _barrier(4)
    errors = []

    def capture(i):
        try:
            g = torch.cuda.CUDAGraph()
            barrier.wait()
            # thread-local capture mode: other threads' CUDA calls must not
            # invalidate this thread's capture (the default "global" mode does)
            with (
                torch.cuda.stream(streams[i]),
                torch.cuda.graph(
                    g, stream=streams[i], capture_error_mode="thread_local"
                ),
            ):
                for _ in range(4):
                    _launch(work[i], k)
            graphs[i] = g
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=capture, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    torch.cuda.synchronize()
    assert not errors, errors
    for _ in range(4):
        for w in work:
            w[3].fill_(-7)
        torch.cuda.synchronize()
        for i in range(4):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
        torch.cuda.synchronize()
        for i, w in enumerate(work):
            _exact(w[0], w[1], w[3], k, who=f"graph {i}")


@requires_gvr2
@_deadline(launches=58)
def test_gvr2_same_stream_graphs_share_a_slab_and_replay_sequentially():
    """Two graphs captured on the SAME stream share that stream's slab. That is
    the documented allowed pattern as long as they replay in stream order —
    alternating replays on the one stream must be exact (the slab is restored
    to zeros by every launch)."""
    k, work = _split_inputs(2, seed=41)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for w in work:
            _launch(w, k)
    torch.cuda.synchronize()
    graphs = []
    for w in work:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
            for _ in range(4):
                _launch(w, k)
        graphs.append(g)
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        for _ in range(6):
            for w, g in zip(work, graphs, strict=True):
                w[3].fill_(-7)
                g.replay()
    torch.cuda.synchronize()
    for i, w in enumerate(work):
        _exact(w[0], w[1], w[3], k, who=f"same-stream graph {i}")


# ---------------------------------------------------------------------------
# 2. the hint-free anchor table under growth, cross-stream use and capture
# ---------------------------------------------------------------------------


@requires_gvr2
@_deadline(launches=8)
def test_gvr2_hint_free_graph_survives_table_growth_by_others():
    """A hint-free graph captured against the anchor table at capacity C keeps
    replaying exactly after other callers grow the table past C (the graph
    holds the OLD table's address, which must stay alive and intact)."""
    k, n = 512, 8192
    key = (torch.cuda.current_device(), k)
    _forget_table(k)
    gen = torch.Generator(device=_DEV).manual_seed(51)
    logits = torch.randn(8, n, generator=gen, device=_DEV)
    seq = torch.randint(600, n, (8,), generator=gen, device=_DEV, dtype=torch.int32)
    out = torch.full((8, k), -7, dtype=torch.int32, device=_DEV)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        _launch((logits, seq, None, out), k, hint=False)
    torch.cuda.synchronize()
    old = _host._HINT_FREE[key]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        _launch((logits, seq, None, out), k, hint=False)
    torch.cuda.synchronize()
    # grow the table well past the captured capacity, several times, from
    # another stream, and scribble over freshly grown tables' *views*
    big_logits = torch.randn(1024, n, generator=gen, device=_DEV)
    big_seq = torch.full((1024,), n, dtype=torch.int32, device=_DEV)
    big_out = torch.empty(1024, k, dtype=torch.int32, device=_DEV)
    other = torch.cuda.Stream()
    for b in (100, 300, 1024):
        with torch.cuda.stream(other):
            _launch((big_logits[:b], big_seq[:b], None, big_out[:b]), k, hint=False)
    torch.cuda.synchronize()
    assert _host._HINT_FREE[key] is not old and _host._HINT_FREE[key].shape[0] >= 1024
    assert old in _host._HINT_FREE_KEEP
    assert bool((old == torch.arange(k, dtype=torch.int32, device=_DEV)).all())
    for _ in range(3):
        out.fill_(-7)
        with torch.cuda.stream(s):
            g.replay()
        torch.cuda.synchronize()
        _exact(logits, seq, out, k, who="graph on the superseded table")


@requires_gvr2
@_deadline(launches=3)
def test_gvr2_hint_free_table_grown_on_one_stream_used_at_once_on_another():
    """Grow the table on stream A and, without any host synchronization,
    launch hint-free on stream B at the new capacity from another thread; the
    published table must already be complete (the producer stream is
    synchronized before publication)."""
    k, n = 1024, 8192
    _forget_table(k)
    gen = torch.Generator(device=_DEV).manual_seed(61)
    logits = torch.randn(512, n, generator=gen, device=_DEV)
    seq = torch.full((512,), n, dtype=torch.int32, device=_DEV)
    outs = [torch.full((512, k), -7, dtype=torch.int32, device=_DEV) for _ in range(2)]
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(
        b
    ):  # compile the launcher for 512 rows on B, hinted (table untouched)
        _launch(
            (logits, seq, torch.zeros(512, k, dtype=torch.int32, device=_DEV), outs[1]),
            k,
        )
    torch.cuda.synchronize()
    _forget_table(k)
    grown = threading.Event()
    errors = []

    def producer():
        try:
            with torch.cuda.stream(a):
                _host._hint_free_pre_idx(512, k, torch.device(_DEV))  # grows 0 -> 512
                grown.set()
                _launch((logits, seq, None, outs[0]), k, hint=False)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def consumer():
        try:
            grown.wait()
            with torch.cuda.stream(b):
                _launch((logits, seq, None, outs[1]), k, hint=False)  # no sync with A
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    torch.cuda.synchronize()
    assert not errors, errors
    _exact(logits, seq, outs[0], k, who="producer stream")
    _exact(logits, seq, outs[1], k, who="consumer stream")


@requires_gvr2
@_deadline(launches=12)
def test_gvr2_hint_free_tables_for_different_k_grow_independently():
    """Interleaved growth of the k=512 and k=2048 tables from two threads keeps
    each table's rows equal to arange(k) and its capacity monotonic."""
    dev = torch.cuda.current_device()
    for k in (512, 2048):
        _forget_table(k)
    seen = {512: [], 2048: []}
    errors = []
    barrier = _barrier(2)

    def grow(k):
        try:
            barrier.wait()
            for b in (8, 70, 20, 300, 9, 600):
                t = _host._hint_free_pre_idx(b, k, torch.device(_DEV))
                seen[k].append(_host._HINT_FREE[(dev, k)].shape[0])
                assert t.shape == (b, k)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=grow, args=(k,)) for k in (512, 2048)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors
    for k in (512, 2048):
        caps = seen[k]
        assert caps == sorted(caps), f"k={k}: capacity shrank: {caps}"
        table = _host._HINT_FREE[(dev, k)]
        assert table.shape[0] >= 600
        assert bool((table == torch.arange(k, dtype=torch.int32, device=_DEV)).all())


# ---------------------------------------------------------------------------
# 3. warm-up on one stream, capture on another: loud, not silent
# ---------------------------------------------------------------------------


@requires_gvr2
@_deadline(launches=8)
def test_gvr2_warmup_varlen_on_stream_enables_capture_on_that_stream_only():
    """warmup_varlen run on stream A creates A's slab and sizes the anchor
    table; a hint-free slab-using capture on A then succeeds, while the same
    capture on a stream that never saw an eager launch raises (never allocates
    from the graph pool)."""
    k, work = _split_inputs(1, seed=71)
    logits, seq, _, out = work[0]
    rows, n = logits.shape
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(a):
        _host.warmup_varlen(k, n, num_rows_list=(rows,))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(a), torch.cuda.graph(g, stream=a):
        _launch((logits, seq, None, out), k, hint=False)
    out.fill_(-7)
    torch.cuda.synchronize()
    with torch.cuda.stream(a):
        g.replay()
    torch.cuda.synchronize()
    _exact(logits, seq, out, k, who="captured after warmup on A")
    g2 = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="no slab for this stream"),
        torch.cuda.stream(b),
        torch.cuda.graph(g2, stream=b),
    ):
        _launch((logits, seq, None, out), k, hint=False)
    torch.cuda.synchronize()


@requires_gvr2
@_deadline(launches=5)
def test_gvr2_register_family_capture_needs_no_slab_on_a_fresh_stream():
    """The rule is scoped to slab-using plans: a register-family shape warmed
    on the default stream captures on a fresh stream without any eager launch
    there (hinted or hint-free) and replays exact."""
    k, n, rows = 512, 8192, 4
    lc = _host._varlen_launcher(rows, n, k, n, 1, 1)
    assert lc[0] in ("reg", "reg_clus"), lc[0]
    gen = torch.Generator(device=_DEV).manual_seed(81)
    logits = torch.randn(rows, n, generator=gen, device=_DEV)
    seq = torch.randint(600, n, (rows,), generator=gen, device=_DEV, dtype=torch.int32)
    pre = torch.zeros(rows, k, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    _launch((logits, seq, pre, out), k)  # default stream, hinted (also sizes the table)
    torch.cuda.synchronize()
    for hint in (True, False):
        fresh = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(fresh), torch.cuda.graph(g, stream=fresh):
            _launch((logits, seq, pre, out), k, hint=hint)
        out.fill_(-7)
        torch.cuda.synchronize()
        with torch.cuda.stream(fresh):
            g.replay()
        torch.cuda.synchronize()
        _exact(logits, seq, out, k, who=f"register family, hint={hint}")
