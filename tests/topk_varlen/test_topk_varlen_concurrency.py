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


def _forget_table(k):
    """Drop the cached hint-free anchor table for k (cold start for a test)."""
    key = (torch.cuda.current_device(), k)
    # getattr: lets the file run against a host without the growth lock (the
    # teeth check runs these tests on the pre-fix tree)
    with getattr(_host, "_HINT_FREE_LOCK", None) or threading.Lock():
        _host._HINT_FREE.pop(key, None)


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
# 2. the hint-free anchor table under growth, cross-stream use and capture
# ---------------------------------------------------------------------------


@requires_gvr2
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
    torch.cuda.synchronize()  # default-stream initialisation before the side-stream launch
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
    # another stream
    big_logits = torch.randn(1024, n, generator=gen, device=_DEV)
    big_seq = torch.full((1024,), n, dtype=torch.int32, device=_DEV)
    big_out = torch.empty(1024, k, dtype=torch.int32, device=_DEV)
    torch.cuda.synchronize()  # default-stream initialisation before the side-stream launches
    other = torch.cuda.Stream()
    for b in (100, 300, 1024):
        with torch.cuda.stream(other):
            _launch((big_logits[:b], big_seq[:b], None, big_out[:b]), k, hint=False)
    torch.cuda.synchronize()
    assert _host._HINT_FREE[key] is not old and _host._HINT_FREE[key].shape[0] >= 1024
    assert any(t is old for t in _host._HINT_FREE_KEEP), (
        "superseded table not kept alive"
    )
    assert bool((old == torch.arange(k, dtype=torch.int32, device=_DEV)).all())
    for _ in range(3):
        with torch.cuda.stream(s):  # the fill is ordered before the replay on s
            out.fill_(-7)
            g.replay()
        torch.cuda.synchronize()
        _exact(logits, seq, out, k, who="graph on the superseded table")


@requires_gvr2
def test_gvr2_hint_free_table_grown_on_one_stream_is_complete_when_published():
    """Publication ordering of the anchor table: the producer on stream A queues
    a long busy-wait kernel and then grows the table (the arange fill sits
    behind the busy-wait in A's queue); the consumer on stream B, without any
    synchronization with A, snapshots the published table the moment the
    pointer is visible. The snapshot must equal arange(k): the host must
    synchronize the producing stream before publishing. (Exactness of the
    consumer's top-k is NOT the observable — gvr_2 is exact for any hint
    contents — so the table itself is compared.) Fails on a host that
    publishes before the fill completes."""
    k, n = 1024, 8192
    key = (torch.cuda.current_device(), k)
    gen = torch.Generator(device=_DEV).manual_seed(61)
    logits = torch.randn(512, n, generator=gen, device=_DEV)
    seq = torch.full((512,), n, dtype=torch.int32, device=_DEV)
    outs = [torch.full((512, k), -7, dtype=torch.int32, device=_DEV) for _ in range(2)]
    torch.cuda.synchronize()  # default-stream initialisation before the side-stream launch
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(b):  # compile the launcher for 512 rows on B, hinted
        _launch(
            (logits, seq, torch.zeros(512, k, dtype=torch.int32, device=_DEV), outs[1]),
            k,
        )
    torch.cuda.synchronize()
    _forget_table(k)  # the hinted call pre-sized it; go cold again
    # the freed table's block would be handed back by the caching allocator
    # with arange(k) still in it, which would mask a publish-before-fill bug:
    # release cached blocks, then poison a same-sized block ON STREAM A (the
    # allocator reuses blocks per stream) so the new table's memory does not
    # start out holding the right answer
    torch.cuda.empty_cache()
    with torch.cuda.stream(a):
        # Walk the growth's exact allocation path once on stream A and free the
        # result, so the real growth reuses cached blocks: any cudaMalloc
        # inside the growth would block the host behind the busy-wait and make
        # even a publish-before-fill host look ordered (measured). The block
        # is left holding -1, so a table published before its fill reads -1.
        warm = (
            torch.arange(k, dtype=torch.int32, device=_DEV)
            .unsqueeze(0)
            .expand(512, k)
            .contiguous()
        )
        warm.fill_(-1)
    torch.cuda.synchronize()
    del warm
    ref = torch.arange(k, dtype=torch.int32, device=_DEV)
    grown = threading.Event()
    snapshot = {}
    errors = []

    orig_arange = torch.arange

    def slow_arange(*args, **kwargs):
        # The growth builds the table as arange(k).expand(...).contiguous():
        # queue a ~2 s GPU busy-wait on the producing stream right AFTER the
        # arange and BEFORE the expand/contiguous copy, so the fill sits behind
        # the busy-wait in stream A's queue. (Queuing the busy-wait before the
        # growth does not work: torch.arange itself blocks the host until the
        # stream drains, which closes the window even on a publish-early host.)
        t = orig_arange(*args, **kwargs)
        torch.cuda._sleep(4_000_000_000)
        return t

    def producer():
        try:
            with torch.cuda.stream(a):
                torch.arange = slow_arange
                try:
                    _host._hint_free_pre_idx(
                        512, k, torch.device(_DEV)
                    )  # grows 0 -> 512
                finally:
                    torch.arange = orig_arange
                grown.set()
                _launch((logits, seq, None, outs[0]), k, hint=False)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            grown.set()

    def consumer():
        try:
            grown.wait()
            with torch.cuda.stream(b):  # no synchronization with A
                snapshot["table"] = _host._HINT_FREE[key].clone()
                _launch((logits, seq, None, outs[1]), k, hint=False)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    torch.cuda.synchronize()
    assert not errors, errors
    snap = snapshot["table"]
    bad_rows = int((snap != ref).any(dim=1).sum())
    assert bad_rows == 0, (
        f"{bad_rows}/{snap.shape[0]} table rows were not arange(k) when published"
    )
    _exact(logits, seq, outs[0], k, who="producer stream")
    _exact(logits, seq, outs[1], k, who="consumer stream")


@requires_gvr2
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
def test_gvr2_warmup_varlen_on_stream_enables_capture_on_that_stream_only():
    """warmup_varlen run on stream A creates A's slab and sizes the anchor
    table; a hint-free slab-using capture on A then succeeds, while the same
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
    thread releases the device's slabs and anchor tables, and the threads
    resume with hint-free launches on the same streams. Every result stays
    exact, every slab and the table are recreated, and a second release frees
    at least the same number of slabs."""
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
                _launch(work[i], k, False, out=outs[i][0])  # slab + table
                streams[i].synchronize()
                b_quiet.wait()  # main releases while everyone is parked here
                b_resume.wait()
                _launch(work[i], k, False, out=outs[i][1])  # recreates them
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
    assert (dev, k) not in _host._HINT_FREE
    b_resume.wait()
    for t in ts:
        t.join()
    torch.cuda.synchronize()
    assert not errors, errors
    assert sum(1 for key in _host._ws_keep if key[0] == dev) >= threads_n
    assert (dev, k) in _host._HINT_FREE
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
    _launch((logits, seq, pre, out), k)  # default stream, hinted (also sizes the table)
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
