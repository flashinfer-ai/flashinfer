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
SPLIT workspace slab; the hint-free anchor table this file also covered was
removed when the hint-free compiled engines of TRT-LLM #18410 were ported —
hint-free launches now hold no per-batch host state). Every test here drives
the API the way a serving engine does — several host threads, several CUDA
streams, CUDA graphs replayed while other work is in flight — and asserts the
invariant directly (exactness per row, or the contents of the shared object),
so a race shows up as a wrong result rather than as a note in a docstring.

Ground rules the tests themselves obey (each was a review finding):

* CUDA-graph captures are serialized — PyTorch allows one capture per process
  at a time; only the REPLAYS are concurrent;
* GPU-side overlap comes from graph replays and single-thread multi-stream
  issue; Python threads serialize on the GIL and exercise host thread-safety;
* every host write to a buffer a stream will read is issued on that stream or
  ordered before it with a synchronize;
* ``torch.cuda.Stream()`` hands out one of 32 pooled raw streams, so a "fresh"
  stream may carry a slab cached by an earlier test — the tests clear the
  cache entry for that raw handle before relying on its absence;
* every test carries ``pytest.mark.timeout(300, method="thread")`` (the
  repository's pytest-timeout convention): a hang dumps every thread's stack
  and terminates the process instead of stalling the suite. Healthy runs take
  1-6 s per test on every part measured.

Inputs are chosen so gvr_2 lands on the streaming ``main`` family with a
multi-CTA SPLIT (the only family that touches the slab); tests skip on parts
that route the shape elsewhere.
"""

import threading

import pytest
import torch

try:
    import flashinfer
    from flashinfer.topk_varlen.kernels import gvr2_topk_host as _host
    from flashinfer.utils import get_compute_capability

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(
        not (_FLASHINFER_AVAILABLE and torch.cuda.is_available()),
        reason="flashinfer + CUDA required",
    ),
    # > cold-cache JIT compile + the longest test by 50x; a real hang is
    # immediate. "thread" mode dumps all stacks and terminates the process.
    pytest.mark.timeout(300, method="thread"),
]
_DEV = "cuda"


def _barrier(parties):
    """Barrier with a timeout: a thread that fails before reaching it must not
    hang the whole test (the others then see BrokenBarrierError and report)."""
    return threading.Barrier(parties, timeout=120)


def _slab_key(stream):
    return (torch.cuda.current_device(), stream.cuda_stream)


def _forget_slab(stream):
    """Drop the cached default workspace slab of ``stream``'s RAW handle.
    torch.cuda.Stream() recycles 32 pooled streams, so a stream object that is
    new to this test may map to a handle an earlier test already gave a slab;
    the tests below establish 'no slab yet' explicitly instead of assuming it."""
    torch.cuda.synchronize()
    with _host._mu:
        _host._ws_keep.pop(_slab_key(stream), None)


def _cc() -> int:
    major, minor = get_compute_capability(torch.device(_DEV))
    return major * 10 + minor


requires_gvr2 = pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE
    or not torch.cuda.is_available()
    or not flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()),
    reason="gvr_2 unsupported on this device",
)


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
    torch.cuda.synchronize()  # inputs are consumed on side streams: complete them first
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
# 1. launches overlapping across streams (and host threads)
# ---------------------------------------------------------------------------


@requires_gvr2
@pytest.mark.parametrize("hint", [True, False], ids=["hinted", "hint_free"])
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
    torch.cuda.synchronize()  # the -7 fills complete before any side stream writes
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
def test_gvr2_graph_replay_races_eager_launches_on_other_streams():
    """Four graphs (eight slab-using launches each, one output buffer per
    launch) captured one after another on streams A..D are replayed back to
    back while stream E receives eight eager slab-using launches (default
    workspace), all issued from ONE host thread with no synchronization until
    the round ends — captured decode steps overlapping with eager work.
    Issuing from one thread is what produces real GPU-side overlap (Python
    threads serialize on the GIL); the four-graph replay is the pattern that
    measured 2-17 corrupted rows per run through a device-wide slab. Each
    stream owns a slab, so every replayed and eager result stays exact."""
    n_graphs, launches, rounds = 4, 8, 12
    k, work = _split_inputs(n_graphs + 1, seed=21)
    streams = [torch.cuda.Stream() for _ in range(n_graphs + 1)]
    outs = [[torch.full_like(w[3], -7) for _ in range(launches)] for w in work]
    torch.cuda.synchronize()  # the -7 fills complete before any side stream writes
    graphs = []
    for i in range(n_graphs):  # captures are serialized (one per process at a time)
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
        torch.cuda.synchronize()  # fills ordered before every stream's work
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
def test_gvr2_graphs_captured_sequentially_replay_concurrently_exact():
    """A serving engine captures its per-batch-size graphs one at a time (only
    one CUDA-graph capture may be underway per process) and later replays them
    concurrently on their streams. Four graphs, each captured on its own
    stream after a per-stream eager warm-up, replay together for several
    rounds; every result stays exact."""
    k, work = _split_inputs(4, seed=31)
    streams = [torch.cuda.Stream() for _ in range(4)]
    graphs = []
    for i in range(4):
        with torch.cuda.stream(streams[i]):
            _launch(work[i], k)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(streams[i]), torch.cuda.graph(g, stream=streams[i]):
            for _ in range(4):
                _launch(work[i], k)
        graphs.append(g)
    torch.cuda.synchronize()
    for _ in range(6):
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
# 3. warm-up on one stream, capture on another: loud, not silent
# ---------------------------------------------------------------------------


@requires_gvr2
def test_gvr2_warmup_varlen_on_stream_enables_capture_on_that_stream_only():
    """warmup_varlen run on stream A creates A's slab and compiles both hint
    modes; a hint-free slab-using capture on A then succeeds, while the same
    capture on a stream with no slab raises (never allocates from the graph
    pool). The 'no slab' precondition is established explicitly, since
    stream B's pooled raw handle may have been given a slab by an earlier
    test."""
    k, work = _split_inputs(1, seed=71)
    logits, seq, _, out = work[0]
    rows, n = logits.shape
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    assert a.cuda_stream != b.cuda_stream
    with torch.cuda.stream(a):
        _host.warmup_varlen(k, n, num_rows_list=(rows,))
    torch.cuda.synchronize()
    assert _slab_key(a) in _host._ws_keep, "warmup_varlen did not create A's slab"
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(a), torch.cuda.graph(g, stream=a):
        _launch((logits, seq, None, out), k, hint=False)
    with torch.cuda.stream(a):
        out.fill_(-7)
        g.replay()
    torch.cuda.synchronize()
    _exact(logits, seq, out, k, who="captured after warmup on A")
    _forget_slab(b)
    g2 = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="no slab for this stream"),
        torch.cuda.stream(b),
        torch.cuda.graph(g2, stream=b),
    ):
        _launch((logits, seq, None, out), k, hint=False)
    torch.cuda.synchronize()
    assert _slab_key(b) not in _host._ws_keep, "a failed capture must not leave a slab"


@requires_gvr2
def test_gvr2_release_between_quiescent_multithreaded_phases_is_exact():
    """`release_gvr2_resources` under its documented contract in a threaded
    program: eight threads launch on eight streams, all of them park at a
    barrier (quiescent: nothing in flight, nothing being issued), the main
    thread releases the device's slabs, and the threads resume with hint-free
    launches on the same streams. Every result stays exact, every slab is
    recreated, and a second release frees at least the same number of slabs."""
    from flashinfer.topk_varlen import release_gvr2_resources

    threads_n = 8
    k, work = _split_inputs(threads_n, seed=91)
    dev = torch.cuda.current_device()
    streams = [torch.cuda.Stream() for _ in range(threads_n)]
    outs = [[torch.full_like(w[3], -7) for _ in range(2)] for w in work]
    torch.cuda.synchronize()
    b_quiet, b_resume = _barrier(threads_n + 1), _barrier(threads_n + 1)
    errors = []

    def worker(i):
        try:
            with torch.cuda.stream(streams[i]):
                _launch(work[i], k, False, out=outs[i][0])  # creates the slab
                streams[i].synchronize()
                b_quiet.wait()  # main releases while everyone is parked here
                b_resume.wait()
                _launch(work[i], k, False, out=outs[i][1])  # recreates it
                streams[i].synchronize()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
    for t in ts:
        t.start()
    b_quiet.wait()
    n_slabs = sum(1 for key in _host._ws_keep if key[0] == dev)
    assert n_slabs >= threads_n
    freed = release_gvr2_resources()
    assert freed >= threads_n * _host.workspace_bytes(), (freed, n_slabs)
    assert not any(key[0] == dev for key in _host._ws_keep)
    b_resume.wait()
    for t in ts:
        t.join()
    torch.cuda.synchronize()
    assert not errors, errors
    assert sum(1 for key in _host._ws_keep if key[0] == dev) >= threads_n
    for i, w in enumerate(work):
        for j in range(2):
            _exact(w[0], w[1], outs[i][j], k, who=f"stream {i} phase {j}")
    assert release_gvr2_resources() >= threads_n * _host.workspace_bytes()


@requires_gvr2
def test_gvr2_register_family_capture_needs_no_slab_on_a_fresh_stream():
    """The rule is scoped to slab-using plans: a register-family shape warmed
    on the default stream captures on a stream WITHOUT a slab (cleared
    explicitly) with no eager launch there, hinted or hint-free, replays
    exact, and the capture does not create a slab for that stream — so a
    regression that made register plans consult the workspace would fail
    here instead of being masked by a slab cached on a recycled stream."""
    k, n, rows = 512, 8192, 4
    lc = _host._varlen_launcher(rows, n, k, n, 1, 1)
    assert lc[0] in ("reg", "reg_clus"), lc[0]
    gen = torch.Generator(device=_DEV).manual_seed(81)
    logits = torch.randn(rows, n, generator=gen, device=_DEV)
    seq = torch.randint(600, n, (rows,), generator=gen, device=_DEV, dtype=torch.int32)
    pre = torch.zeros(rows, k, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    for hint in (True, False):  # default stream: compile both hint modes' launchers
        _launch((logits, seq, pre, out), k, hint=hint)
    torch.cuda.synchronize()
    for hint in (True, False):
        fresh = torch.cuda.Stream()
        _forget_slab(fresh)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(fresh), torch.cuda.graph(g, stream=fresh):
            _launch((logits, seq, pre, out), k, hint=hint)
        assert _slab_key(fresh) not in _host._ws_keep, (
            "a register-family capture must not touch the workspace slab"
        )
        with torch.cuda.stream(fresh):
            out.fill_(-7)
            g.replay()
        torch.cuda.synchronize()
        _exact(logits, seq, out, k, who=f"register family, hint={hint}")
        assert _slab_key(fresh) not in _host._ws_keep
