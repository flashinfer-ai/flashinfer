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

Host synchronization of the KCP route (``cake_kda_det_train.cake_kcp`` / ``cake_kcp_plan``): pack preparation and the
KDA / conv halo calls never copy a table from pageable memory and never wait for queued device work, with or without
a caller-owned workspace.  The signals of a window are those of ``tests/test_helpers/cake_host_sync.py``: an upload
counter that needs no device tracing (equal to the profiler's host-to-device copies by kind where the profiler
traces the device), the caching allocator's device allocations, the synchronizing operations
``torch.cuda.set_sync_debug_mode("warn")`` reports, and a queue probe (``torch.cuda._sleep`` enqueued before the
window must still run after it, and the host must return well before it ends).  The negative control is the previous
path (context test knobs ``upload="blocking"`` + ``table_cache="workspace"``): the same signals must report the copies
and the waits there.  Results of the new path are bitwise those of the previous one.
"""

import gc
import threading
import weakref

import pytest
import torch

from tests.test_helpers.cake_host_sync import (
    SLEEP_MS,
    allocator_counts,
    census,
    counts_agree,
    nonblocking,
    sleep_cycles,
    summary,
)
from tests.test_helpers.cake_package import cake_package

kp = cake_package("cake_kda_det_train", "cake_kcp_plan")
cake_kcp = cake_package("cake_kda_det_train", "cake_kcp")
cake_kcp_jit = cake_package("cake_kda_det_train", "cake_kcp_jit")
cake_jit = cake_package("cake_kda_det_train", "cake_jit")

# the previous path (A/B, negative control)
OLD = {"upload": "blocking", "table_cache": "workspace"}
CU = (0, 300, 1024, 1500)  # one rank, three documents (boundaries inside chunks)
# the call rows: a row whose plan splits the token axis (P > 1) and one whose plan does not (P = 1).  P depends on the
# device's SM count, so each class takes the first (documents, H) of its search range whose plan on the device is of
# that class (_row_of); the kernels take 1..64 heads.  A class no candidate reaches skips, naming the range searched.
ROW_CUS = (
    CU,  # three documents, 1500 tokens (the row of the other tests)
    tuple(range(0, 2401, 300)),  # eight documents of 300 tokens (occupancy rule)
    (0, 1000, 2000),  # two documents
    (0, 4096),  # one long document (one-wave fill)
    (0, 64),  # one chunk (P = 1 by the chunk count)
)
P_CLASSES = {"P>1": (2, 4, 8), "P=1": (64, 48, 32, 16)}
C_CONV = 256
ALLOC_SKIP = (
    "the caching allocator's statistics do not count device allocations here (allocator_counts); "
    "the copy, synchronization and queue-probe gates of this test passed"
)


@pytest.fixture
def knobs(monkeypatch):
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {tuple(cake_jit.ARCHES)}"
        )
    return torch.device("cuda", index)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _assert_nonblocking(c: dict, what) -> None:
    assert nonblocking(c), (what, summary(c))


def _assert_no_alloc(c: dict, alloc_ok: bool, what) -> None:
    """No device allocation, free or out-of-memory retry in the window (allocator statistics; the profiler's
    ``cudaMalloc`` count too where it traces the device).  Unchecked when ``alloc_ok`` is False (the test then
    skips at its end with :data:`ALLOC_SKIP`)."""
    if alloc_ok:
        assert (c["device_alloc"], c["device_free"], c["alloc_retries"]) == (
            0,
            0,
            0,
        ), (what, summary(c))
        assert c["prof_malloc"] in (None, 0), (what, summary(c))


def _clear_content_caches() -> None:
    """Empty the process caches keyed by table content (the untrimmed path's): a measured first call then finds
    nothing an earlier call of the same partition uploaded."""
    cake_kcp_jit._device_table.cache_clear()
    cake_kcp_jit._chain_tables_cached.cache_clear()
    dev = getattr(cake_kcp.get_backend(), "_dev", None)
    if dev is not None:
        dev.clear()


def _ops(ctx, H: int, device: torch.device, seed: int = 0x7815) -> dict:
    dt = cake_kcp.get_backend().dtypes
    T = int(ctx.T_r)
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=gen, dtype=torch.float32)

    def unit(x):  # the delta rule's operand contract: q / k L2-normalized per head
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    return {
        "q": unit(randn(T, H, cake_kcp.D)).to(dt["q"]),
        "k": unit(randn(T, H, cake_kcp.D)).to(dt["k"]),
        "v": (0.5 * randn(T, H, cake_kcp.D)).to(dt["v"]),
        "g": (-5.0 * torch.sigmoid(randn(T, H, cake_kcp.D))).to(dt["g"]),
        "beta": torch.sigmoid(randn(T, H)).to(dt["beta"]),
        "dout": (0.5 * randn(T, H, cake_kcp.D)).to(dt["dout"]),
    }


def _new_ws():
    backend = cake_kcp.get_backend()
    return cake_kcp.KcpWorkspace(
        state_dtype=backend.dtypes["state"], local=backend.local_stages
    )


def _passes(ctx, o: dict, ws, **opts) -> tuple:
    """fwd + bwd in saved mode, in replay mode (``boundary_mode="replay"``), and the autograd entry under
    non-reentrant checkpointing: ``out`` and the five gradients of each."""
    q, k, v, g, beta, dout = (o[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    res = []
    for mode in ("saved", "replay"):
        out, saved = cake_kcp.chunk_kda_kcp_fwd(
            q, k, v, g, beta, ctx, boundary_mode=mode, workspace=ws, **opts
        )
        grads = cake_kcp.chunk_kda_kcp_bwd(
            q, k, v, g, beta, ctx, dout, saved=saved, workspace=ws, **opts
        )
        res += [out, *grads]
    leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
    out = torch.utils.checkpoint.checkpoint(
        lambda *a: cake_kcp.chunk_kda_kcp(*a, ctx, None, "light", "saved", ws),
        *leaves,
        use_reentrant=False,
    )
    out.backward(dout)
    res += [out.detach()] + [lf.grad for lf in leaves]
    return tuple(res)


# --------------------------------------------------------------------------- CPU


def test_upload_tables_layout_and_refusals(monkeypatch) -> None:
    """``_upload_tables``: the values, dtype and shape of ``torch.tensor(values, dtype)`` (flat, nested, empty and
    ``(0, k)`` tables); a value count that does not match the shape is a ``ValueError``; on a CPU device no upload
    record.  On CUDA (when present) every table is a view at a 256-byte offset of one blob, made by one pinned copy;
    under CUDA-graph capture the call is refused before any copy (its own refusal, not a CUDA capture error)."""
    specs = {
        "a": ([3, -1, 7], torch.int32, None),
        "b": ([[1, 2], [3, 4], [5, 6]], torch.int32, (3, 2)),
        "c": ([], torch.int64, None),
        "d": ([], torch.int32, (0, 4)),
        "e": ([2**40, -5], torch.int64, None),
    }
    out, rec = kp._upload_tables(specs, "cpu")
    assert rec is None
    for name, (values, dtype, shape) in specs.items():
        ref = torch.tensor(values, dtype=dtype)
        ref = ref if shape is None else ref.reshape(shape)
        assert out[name].dtype == dtype and torch.equal(out[name], ref), name
    with pytest.raises(ValueError, match="holds 3 values"):
        kp._upload_tables({"x": ([1, 2, 3], torch.int32, (2, 2))}, "cpu")
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda", torch.cuda.current_device())
    out, rec = kp._upload_tables(specs, dev)
    torch.cuda.synchronize(dev)
    blob = rec.blob
    assert blob is not None and blob.dtype == torch.uint8
    for name, (values, dtype, shape) in specs.items():
        t = out[name]
        ref = torch.tensor(values, dtype=dtype)
        ref = ref if shape is None else ref.reshape(shape)
        assert t.is_contiguous() and t.dtype == dtype
        assert torch.equal(t.cpu(), ref), name
        assert t.untyped_storage().data_ptr() == blob.untyped_storage().data_ptr()
        assert (t.data_ptr() - blob.data_ptr()) % kp.ALIGN == 0, name
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="re-read host memory"):
        kp._upload_tables(specs, dev)


def test_upload_knobs_are_gated_context_knobs(monkeypatch) -> None:
    """``upload`` / ``table_cache`` are private context knobs of ``build_kcp_context``: refused without
    ``enable_test_knobs()``, validated, recorded on the context; ``table_cache="workspace"`` (the previous path) only
    with ``upload="blocking"`` (its table sets are not recorded on the context); the KDA entry points refuse them (they
    are no call option); ``heads=`` takes an int-like scalar (0-d integer tensor) or a sequence; the head plan's device
    tables and the ``frag_table`` are uploaded on first read (CPU device here)."""
    part = kp.kcp_partition(list(CU), 1)
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(ValueError, match="enable_test_knobs"):
        kp.build_kcp_context(part, None, 0, "cpu", _kcp_test_knobs=dict(OLD))
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    with pytest.raises(ValueError, match="upload must be one of"):
        kp.build_kcp_context(part, None, 0, "cpu", _kcp_test_knobs={"upload": "x"})
    with pytest.raises(ValueError, match="table_cache must be one of"):
        kp.build_kcp_context(part, None, 0, "cpu", _kcp_test_knobs={"table_cache": "x"})
    with pytest.raises(ValueError, match="needs upload='blocking'"):
        kp.build_kcp_context(
            part, None, 0, "cpu", _kcp_test_knobs={"table_cache": "workspace"}
        )
    blocking = kp.build_kcp_context(
        part, None, 0, "cpu", _kcp_test_knobs={"upload": "blocking"}
    )
    assert (blocking.upload, blocking.table_cache) == ("blocking", "context")
    scalar = kp.build_kcp_context(part, None, 0, "cpu", heads=torch.tensor(64))
    assert set(scalar._head_plans) == {64}
    with pytest.raises(ValueError, match="unknown knob"):
        cake_kcp._options({"_kcp_test_knobs": {"upload": "blocking"}})
    old = kp.build_kcp_context(part, None, 0, "cpu", _kcp_test_knobs=dict(OLD))
    new = kp.build_kcp_context(part, None, 0, "cpu", heads=(4, 64))
    assert (old.upload, old.table_cache) == ("blocking", "workspace")
    assert (new.upload, new.table_cache) == (
        kp.UPLOAD_DEFAULT,
        kp.TABLE_CACHE_DEFAULT,
    )
    assert set(new._head_plans) == {4, 64}  # heads= derived the head plans
    hp = new.heads(64)
    assert (
        isinstance(hp.device_tables, kp._LazyTables) and not hp.device_tables.uploaded
    )
    assert set(hp.device_tables) == set(old.heads(64).device_tables)
    for name in hp.device_tables:
        a, b = hp.device_tables[name], old.heads(64).device_tables[name]
        assert a.dtype == b.dtype and torch.equal(a, b), name
    assert hp.device_tables.uploaded
    assert torch.equal(new.frag_table, old.frag_table) and new.frag_table.shape == (
        new.n_frag,
        4,
    )
    # the table set lives on the context (shared by workspaces); the knob keeps it per workspace
    ws1, ws2 = cake_kcp.KcpWorkspace(local=False), cake_kcp.KcpWorkspace(local=False)
    assert ws1.tables(new, 4) is ws2.tables(new, 4) and not ws1._tables
    assert ws1.tables(old, 4) is not ws2.tables(old, 4) and ws1._tables


# --------------------------------------------------------------------------- GPU


def test_gpu_prepare_does_not_wait_for_queued_work(device, knobs) -> None:
    """B T1: ``build_kcp_context(..., heads=H)`` behind ~1 s of queued device work: no synchronizing operation,
    no pageable copy, exactly one pinned copy (the context's tables; the head plans are host work), the host returns
    while the queued work still runs and the context's ready event is still pending.  After the queue drains every
    device table equals the CPU-built context's.  The previous path (knob) waits: pageable copies and synchronizing
    operations are counted and the queue has drained when it returns.  Rows: every rank of C / Z / D at W = 4, H = 64.
    Copy counts: the upload counter (and the profiler where it traces the device, equal to the counter)."""
    H = 64
    warm = kp.build_kcp_context(kp.kcp_partition(list(CU), 1), None, 0, device, heads=H)
    del warm  # the pinned staging size class and the allocator blocks are warm
    seen = 0
    for layout in kp.LAYOUTS:
        part = kp.kcp_partition([0, 700, 701, 2001, 3000], 4, layout=layout)
        for r in range(4):

            def prepare():
                ctx = kp.build_kcp_context(part, None, r, device, heads=H)
                return ctx, not ctx.ready.query()

            c = census(prepare, device)
            ctx, pending = c["out"]
            _assert_nonblocking(c, (layout, r))
            assert c["pinned"] == 1 and pending, (layout, r, summary(c))
            ref = kp.build_kcp_context(part, None, r, "cpu", sm_count=ctx.sm_count)
            for name in ref.tables:
                assert torch.equal(ctx.tables[name].cpu(), ref.tables[name]), name
            for name in ref.halo_tables:
                a, b = ctx.halo_tables[name], ref.halo_tables[name]
                assert a.dtype == b.dtype and torch.equal(a.cpu(), b), name
            assert torch.equal(ctx.cu_seqlens_local.cpu(), ref.cu_seqlens_local)
            assert torch.equal(ctx.frag_table.cpu(), ref.frag_table)
            hp, hp_ref = ctx.heads(H), ref.heads(H)
            assert not hp.device_tables.uploaded
            for name in hp_ref.device_tables:
                assert torch.equal(
                    hp.device_tables[name].cpu(), hp_ref.device_tables[name]
                ), name
            seen += 1
            old = census(
                lambda: kp.build_kcp_context(
                    part, None, r, device, heads=H, _kcp_test_knobs=dict(OLD)
                ),
                device,
            )
            assert old["pageable"] > 0 and counts_agree(old), (layout, r, summary(old))
            assert old["sync_warn"] > 0 and not old["busy"], (layout, r, summary(old))
    assert seen == 12


def test_gpu_prepare_inside_a_capture_is_refused(device) -> None:
    """A context is never prepared under CUDA-graph capture (the copy would re-read host memory the graph does not
    own): the upload's own ``RuntimeError``, not a CUDA capture error of an operation that ran under the capture."""
    part = kp.kcp_partition(list(CU), 1)
    g = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="re-read host memory"), torch.cuda.graph(g):
        kp.build_kcp_context(part, None, 0, device)


def test_gpu_byte_queries_and_head_plans_are_host_only(device) -> None:
    """The byte queries and ``heads=`` touch no device: no upload is recorded on the context and the head plan's
    device tables stay unread."""
    part = kp.kcp_partition([0, 700, 701, 2001, 3000], 4, layout="zigzag_per_document")
    ctx = kp.build_kcp_context(part, None, 1, device, heads=(4, 64))
    n = len(ctx._uploads)
    assert n == 1
    for H in (4, 64):
        kp.kcp_workspace_bytes(ctx, H=H)
        kp.kcp_saved_bytes(ctx, H=H)
        kp.kcp_comm_bytes(ctx, H=H, C=1024)
        assert not ctx.heads(H).device_tables.uploaded
    assert len(ctx._uploads) == n


def _row_of(device: torch.device, p_class: str) -> tuple[tuple, int, int]:
    """The first (documents, H) of ``ROW_CUS`` x ``P_CLASSES[p_class]`` whose plan on this device is of that class,
    and its P."""
    sms = None
    for cu in ROW_CUS:
        probe = kp.build_kcp_context(kp.kcp_partition(list(cu), 1), None, 0, device)
        sms = probe.sm_count
        for H in P_CLASSES[p_class]:
            P = cake_kcp.KcpWorkspace(local=False).tables(probe, H).P
            if (P > 1) == (p_class == "P>1"):
                return cu, H, P
    pytest.skip(
        f"no row of documents {ROW_CUS} x H {P_CLASSES[p_class]} gives {p_class} on {sms} SMs"
    )


def _other_of_class(cu: tuple, H: int, P: int, device: torch.device) -> tuple[int, ...]:
    """A partition of the class of ``cu`` (same number of documents, same P for ``H`` on this device) whose tables
    differ from ``cu``'s: the documents in reverse order, else boundaries moved by whole chunks."""
    lens = [b - a for a, b in zip(cu, cu[1:], strict=False)]
    cands = [lens[::-1]]
    for d in (64, -64, 128, -128, 192, -192):
        cands.append(
            [lens[0] + d, *lens[1:-1], lens[-1] - d] if len(lens) > 1 else [lens[0] + d]
        )
    for cand in cands:
        if cand == lens or min(cand) <= 0:
            continue
        other = [0]
        for n in cand:
            other.append(other[-1] + n)
        probe = kp.build_kcp_context(kp.kcp_partition(other, 1), None, 0, device)
        if cake_kcp.KcpWorkspace(local=False).tables(probe, H).P == P:
            return tuple(other)
    raise AssertionError(f"no partition of the class of {cu} (H {H}, P {P}) found")


@pytest.mark.parametrize("p_class", list(P_CLASSES))
@pytest.mark.parametrize("ws_mode", ["none", "reused"])
def test_gpu_calls_issue_no_blocking_upload(device, knobs, p_class, ws_mode) -> None:
    """A T1 / T2: the first fwd + bwd (saved, replay, autograd under checkpoint) of a fresh context of a partition no
    earlier call has seen (the layout class is warmed by another partition of the same class, and the process caches
    keyed by table content are emptied) issues only pinned asynchronous copies -- at least one: the table sets of the
    call -- and no pageable copy, no synchronizing operation, the queue still runs when the host returns; in steady
    state the same passes issue no copy at all and no device allocation or free, with ``workspace=None`` and with a
    reused workspace.  The previous path (knob) on the same passes with ``workspace=None`` is the negative control:
    pageable copies, synchronizing operations, a drained queue; bitwise the new path's results."""
    alloc_ok = allocator_counts(device)
    cu, H, P = _row_of(device, p_class)
    other = _other_of_class(cu, H, P, device)
    print(f"ROW {p_class}: documents {cu} H {H} P {P}; warmed by {other}")
    part = kp.kcp_partition(list(cu), 1)

    def run(ctx, o, ws):
        return _passes(ctx, o, ws if ws_mode == "reused" else None)

    warm = kp.build_kcp_context(kp.kcp_partition(list(other), 1), None, 0, device)
    # the class: modules loaded, allocator blocks cached
    run(warm, _ops(warm, H, device), _new_ws())
    del warm
    _clear_content_caches()
    torch.cuda.synchronize(device)
    ctx = kp.build_kcp_context(part, None, 0, device)
    o = _ops(ctx, H, device)
    ws = _new_ws()
    torch.cuda.synchronize(device)
    first = census(lambda: run(ctx, o, ws), device)
    _assert_nonblocking(first, ("first", cu, H, P, ws_mode))
    assert first["pinned"] >= 1, ("first", summary(first))
    for _ in range(2):
        run(ctx, o, ws)
    steady = census(lambda: run(ctx, o, ws), device)
    _assert_nonblocking(steady, ("steady", cu, H, P, ws_mode))
    assert steady["pinned"] == 0, ("steady", summary(steady))
    _assert_no_alloc(steady, alloc_ok, ("steady", cu, H, P, ws_mode))
    print(f"CENSUS first {summary(first)} steady {summary(steady)}")
    for a, b in zip(first["out"], steady["out"], strict=True):
        assert torch.equal(a, b)
    if ws_mode == "none":
        old = kp.build_kcp_context(part, None, 0, device, _kcp_test_knobs=dict(OLD))
        for _ in range(2):
            run(old, o, None)
        neg = census(lambda: run(old, o, None), device)
        assert neg["pageable"] > 0 and counts_agree(neg), ("old", summary(neg))
        assert neg["sync_warn"] > 0 and not neg["busy"], ("old", summary(neg))
        print(f"CENSUS old {summary(neg)}")
        for a, b in zip(neg["out"], steady["out"], strict=True):
            assert torch.equal(a, b)
    if not alloc_ok:
        pytest.skip(ALLOC_SKIP)


@pytest.mark.parametrize("p_class", list(P_CLASSES))
def test_gpu_new_path_is_bitwise_the_previous_path(device, knobs, p_class) -> None:
    """A T5 (one rank), a row of each P class on this device (:func:`_row_of`): out and the five gradients of the
    saved, replay and checkpointed passes, and the conv halo's ``y`` / ``dx`` / ``dweight``, are bitwise equal between
    the new path and the previous one (knob), with ``workspace=None`` and with a reused workspace."""
    cu, H, P = _row_of(device, p_class)
    print(f"ROW {p_class}: documents {cu} H {H} P {P}")
    part = kp.kcp_partition(list(cu), 1)
    new = kp.build_kcp_context(part, None, 0, device)
    old = kp.build_kcp_context(part, None, 0, device, _kcp_test_knobs=dict(OLD))
    o = _ops(new, H, device)
    for ws_new, ws_old in ((None, None), (_new_ws(), _new_ws())):
        a, b = _passes(new, o, ws_new), _passes(old, o, ws_old)
        assert len(a) == 18
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert torch.equal(x, y), (i, ws_new is None)
    from flashinfer.cake_causal_conv1d_det_train_backend import kcp as conv

    gen = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(new.T_r, C_CONV, device=device, generator=gen).to(torch.bfloat16)
    w = (0.5 * torch.randn(C_CONV, 4, device=device, generator=gen)).contiguous()
    dy = torch.randn(new.T_r, C_CONV, device=device, generator=gen).to(torch.bfloat16)
    res = []
    for ctx in (new, old):
        y, hist = conv.cake_causal_conv1d_kcp_forward(x, w, ctx)
        dx, dw = conv.cake_causal_conv1d_kcp_backward(x, w, ctx, dy, hist)
        res.append((y, dx, dw))
    for x_, y_ in zip(*res, strict=True):
        assert torch.equal(x_, y_)


def test_gpu_graph_replay_issues_no_copy(device) -> None:
    """A T4: a reused workspace under ``graph="replay"``: the first call of a key is eager (it uploads the table
    set), the forward is captured before the backward, and once both replay the replayed calls issue no copy, no
    allocator call and no synchronizing operation, bitwise equal to the eager call; no capture falls back to eager
    (``GRAPH_STATS["fallback"]`` unchanged).  The calls that capture are not measured: a capture allocates the graph's
    private pool and synchronizes by design."""
    alloc_ok = allocator_counts(device)
    H = 2
    part = kp.kcp_partition(list(CU), 1)
    ctx = kp.build_kcp_context(part, None, 0, device)
    o = _ops(ctx, H, device)
    ws = _new_ws()
    q, k, v, g, beta, dout = (o[n] for n in ("q", "k", "v", "g", "beta", "dout"))

    def call(clone: bool = True):
        out, saved = cake_kcp.chunk_kda_kcp_fwd(
            q, k, v, g, beta, ctx, workspace=ws, graph="replay"
        )
        grads = cake_kcp.chunk_kda_kcp_bwd(
            q, k, v, g, beta, ctx, dout, saved=saved, workspace=ws, graph="replay"
        )
        events = (
            cake_kcp.LAST_CALL["fwd"]["graph_event"],
            cake_kcp.LAST_CALL["bwd"]["graph_event"],
        )
        res = (out, *grads)
        return (tuple(t.clone() for t in res) if clone else res), events

    cake_kcp.graph_cache_clear()
    fallback = cake_kcp.GRAPH_STATS["fallback"]
    try:
        ref, ev = call()
        events = [ev]
        while ev != ("replay", "replay") and len(events) < 8:
            ev = call()[1]
            events.append(ev)
        assert events[0] == ("eager", "eager") and ev == ("replay", "replay"), events
        fwd, bwd = [e[0] for e in events], [e[1] for e in events]
        assert fwd.count("capture") == 1 and bwd.count("capture") == 1, events
        assert fwd.index("capture") < bwd.index("capture"), events
        for i in range(2):
            # the replayed call alone in the window: its static outputs are compared right after it
            # (the test's own clones would grow the allocator)
            c = census(lambda: call(clone=False), device)
            res, ev = c["out"]
            assert ev == ("replay", "replay"), (i, ev)
            _assert_nonblocking(c, ("replay", i))
            assert c["pinned"] == 0, (i, summary(c))
            _assert_no_alloc(c, alloc_ok, ("replay", i))
            for a, b in zip(ref, res, strict=True):
                assert torch.equal(a, b)
            del c, res
        assert cake_kcp.GRAPH_STATS["fallback"] == fallback
    finally:
        cake_kcp.graph_cache_clear()
    if not alloc_ok:
        pytest.skip(ALLOC_SKIP)


@pytest.mark.parametrize("hook", ["kda", "conv"])
@pytest.mark.parametrize("form", ["stream", "thread"])
def test_gpu_side_stream_preparation_is_ordered(device, form, hook) -> None:
    """B T4 (one rank), each entry point's hook alone: the next pack's context is prepared on a side stream whose
    copy is held back behind ~1 s of queued work there (``stream=side``; ``form="thread"`` prepares it from another
    thread, which must pass the stream), and the pack runs at once on the compute stream through ONE entry point:
    the KDA passes (``hook="kda"``) or the conv halo forward + backward (``hook="conv"``).  That entry point orders the
    copy itself (``ctx.wait_ready()``, an event wait): the context's wait record for the compute stream is empty
    before the pack and set after it, and the results are bitwise those of a context prepared on an idle device.
    Without that ordering the kernels would read the blob before its copy."""
    from flashinfer.cake_causal_conv1d_det_train_backend import kcp as conv

    H = 2
    part = kp.kcp_partition(list(CU), 1)
    ref_ctx = kp.build_kcp_context(part, None, 0, device)
    o = _ops(ref_ctx, H, device)
    gen = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(ref_ctx.T_r, C_CONV, device=device, generator=gen).to(
        torch.bfloat16
    )
    w = (0.5 * torch.randn(C_CONV, 4, device=device, generator=gen)).contiguous()
    dy = torch.randn(ref_ctx.T_r, C_CONV, device=device, generator=gen).to(
        torch.bfloat16
    )

    def pack(ctx):
        if hook == "conv":
            y, hist = conv.cake_causal_conv1d_kcp_forward(x, w, ctx)
            return (y, *conv.cake_causal_conv1d_kcp_backward(x, w, ctx, dy, hist))
        return _passes(ctx, o, None)

    ref = pack(ref_ctx)
    torch.cuda.synchronize(device)
    side = torch.cuda.Stream(device)
    holder = {}

    def prepare():
        holder["ctx"] = kp.build_kcp_context(part, None, 0, device, stream=side)

    with torch.cuda.stream(side):
        torch.cuda._sleep(sleep_cycles(device, SLEEP_MS))
    if form == "thread":
        th = threading.Thread(target=prepare)
        th.start()
        th.join()
    else:
        prepare()
    ctx = holder["ctx"]
    assert not ctx.ready.query()  # the copy is still held back on the side stream
    key = torch.cuda.current_stream(device).cuda_stream
    assert ctx._uploads[0].stream.cuda_stream == side.cuda_stream
    assert ctx._waited.get(key, 0) == 0
    res = pack(ctx)
    # the side-stream copy was ordered on the compute stream by this entry point
    assert ctx._waited.get(key, 0) >= 1, hook
    torch.cuda.synchronize(device)
    for i, (a, b) in enumerate(zip(ref, res, strict=True)):
        assert torch.equal(a, b), (hook, i)


def test_gpu_conv_halo_orders_a_cached_context_on_every_stream(device) -> None:
    """The conv halo's trimmed host path orders the context on every call, not only on a host-cache miss.  A
    context is prepared on a side stream P held back by ~1 s; its first forward runs on stream A (a host-cache miss),
    the second forward of the same context and shape on stream B (a host-cache hit).  B's call must wait for P's
    copy: the blob reuses a block of P's pool that holds zeros (a call reading the tables before the copy sees empty
    fragments and leaves its sentinel-filled output unwritten).  Both outputs are bitwise the reference, and the
    context's wait record names A and B (``wait_event`` plus ``record_stream`` of the blob on each)."""
    from flashinfer.cake_causal_conv1d_det_train_backend import kcp as conv

    part = kp.kcp_partition(list(CU), 1)
    ref_ctx = kp.build_kcp_context(part, None, 0, device)
    nbytes = ref_ctx._uploads[0].blob.numel()
    gen = torch.Generator(device=device).manual_seed(13)
    x = torch.randn(ref_ctx.T_r, C_CONV, device=device, generator=gen).to(
        torch.bfloat16
    )
    w = (0.5 * torch.randn(C_CONV, 4, device=device, generator=gen)).contiguous()
    y_ref, _ = conv.cake_causal_conv1d_kcp_forward(x, w, ref_ctx)
    torch.cuda.synchronize(device)
    prep, a_stream, b_stream = (torch.cuda.Stream(device) for _ in range(3))
    with torch.cuda.stream(prep):
        poison = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        poison_ptr = poison.data_ptr()
        torch.cuda._sleep(sleep_cycles(device, SLEEP_MS))
        del poison  # its block returns to the side stream's pool, zeroed
    ctx = kp.build_kcp_context(part, None, 0, device, stream=prep)
    on_zeros = ctx._uploads[0].blob.data_ptr() == poison_ptr
    print(f"blob on the zeroed block: {on_zeros}")
    assert not ctx.ready.query()
    outs = []
    for s in (a_stream, b_stream):
        with torch.cuda.stream(s):
            out = torch.full(
                tuple(y_ref.shape), 7.0, dtype=torch.bfloat16, device=device
            )
            y, _hist = conv.cake_causal_conv1d_kcp_forward(x, w, ctx, out=out)
        outs.append(y)
    for s in (a_stream, b_stream):
        assert ctx._waited.get(s.cuda_stream, 0) >= 1, s
    torch.cuda.synchronize(device)
    for i, y in enumerate(outs):
        assert torch.equal(y, y_ref), i


def _holders(refs) -> list:
    """For each object of ``refs`` still alive: its type and the types of its referrers (a retention diagnosis)."""
    out = []
    for r in refs:
        obj = r()
        if obj is not None:
            out.append(
                (type(obj).__name__, [type(x).__name__ for x in gc.get_referrers(obj)])
            )
    return out


def test_gpu_fresh_context_with_conv_halo_releases_everything(device) -> None:
    """A T7 / critique #3: a loop that builds a new context per step and runs the conv halo and the KDA step with
    ``workspace=None`` keeps nothing once the conv module's host cache is cleared: the context, its table sets and
    its blobs are freed by reference counting alone (the cycle collector disabled; the table set holds its context
    through a weak proxy) and ``memory_allocated`` returns to its level after the first step."""
    from flashinfer.cake_causal_conv1d_det_train_backend import kcp as conv

    H = 2
    part = kp.kcp_partition(list(CU), 1)
    o = _ops(kp.build_kcp_context(part, None, 0, device), H, device)
    q, k, v, g, beta, dout = (o[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    gen = torch.Generator(device=device).manual_seed(11)
    T = int(CU[-1])
    x = torch.randn(T, C_CONV, device=device, generator=gen).to(torch.bfloat16)
    w = (0.5 * torch.randn(C_CONV, 4, device=device, generator=gen)).contiguous()

    def step(checkpointed: bool):
        ctx = kp.build_kcp_context(part, None, 0, device, heads=H)
        y, _hist = conv.cake_causal_conv1d_kcp_forward(x, w, ctx)
        if checkpointed:
            res = tuple(t.detach().clone() for t in (y, *_passes(ctx, o, None)))
        else:
            res = [y.detach().clone()]
            for mode in ("saved", "replay"):
                out, saved = cake_kcp.chunk_kda_kcp_fwd(
                    q, k, v, g, beta, ctx, boundary_mode=mode, workspace=None
                )
                grads = cake_kcp.chunk_kda_kcp_bwd(
                    q, k, v, g, beta, ctx, dout, saved=saved, workspace=None
                )
                res += [out, *grads]
                del out, saved, grads
            res = tuple(res)
        tb = ctx._table_sets[next(iter(ctx._table_sets))]
        assert isinstance(tb.ctx, weakref.ProxyType)
        return res, [weakref.ref(ctx), weakref.ref(tb)]

    conv.host_cache_clear()
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:  # reference counting alone (a strong context <-> table set cycle would survive here)
        for i in range(2):
            _res, refs = step(checkpointed=False)
            del _res
            conv.host_cache_clear()
            torch.cuda.synchronize(device)
            assert all(r() is None for r in refs), (i, _holders(refs))
    finally:
        if enabled:
            gc.enable()
    first, refs = step(checkpointed=True)
    conv.host_cache_clear()
    gc.collect()
    torch.cuda.synchronize(device)
    assert all(r() is None for r in refs), "the first step's context is retained"
    base = torch.cuda.memory_allocated(device)
    for i in range(3):
        res, refs = step(checkpointed=True)
        assert all(torch.equal(a, b) for a, b in zip(first, res, strict=True)), i
        del res
        conv.host_cache_clear()
        gc.collect()
        torch.cuda.synchronize(device)
        assert all(r() is None for r in refs), (i, "a context is retained")
        assert torch.cuda.memory_allocated(device) == base, i
