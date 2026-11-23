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

Host-side checks of the specialized backend of ``allreduce_fusion`` (no GPU): the
allowlist, the static creation envelope, the guard order, the factory's backend selection,
the buffer check of the fallback, the creation votes (two simulated ranks) and the
dispatch path with a stand-in launcher (omitted outputs, misaligned views). The GPU tests
are in ``tests/comm/test_trtllm_allreduce_fusion.py``.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

import flashinfer.comm as comm
from flashinfer.comm import _allreduce_fusion_ws2_specialized as ws2

_HIDDEN = 6144
# The workspace vLLM creates for this hidden size (64 MiB of bf16 per rank).
_MAX_TOKENS = 5461
_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"


@pytest.fixture(autouse=True)
def _no_inherited_kill_switch(monkeypatch):
    """The tests set the kill switch themselves; an inherited value would reroute every call."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)


def _state(capacity=None, device_idx=0, handle=None):
    allowlist = ws2._load_allowlist()
    return ws2._State(
        handle=handle,
        own_ptr=1,
        peer_ptr=2,
        epoch=torch.zeros(1, dtype=torch.int32),
        capacity=allowlist.t_max if capacity is None else capacity,
        grid=1,
        device_idx=device_idx,
        rank=0,
        region_bytes=0,
    )


def test_ws2_specialized_guard_order(monkeypatch):
    """The allowlist content, the static creation envelope (no collective outside it) and the
    guard order (token count, semantics, shape/layout, device last)."""
    allowlist = ws2._load_allowlist()
    assert ws2._allowlist_error is None, ws2._allowlist_error
    assert 1.0 in allowlist.weight_bias
    assert allowlist.num_tokens and allowlist.t_min >= 1
    assert allowlist.t_max <= _MAX_TOKENS

    assert (
        ws2.creation_envelope_reason(2, 6144, torch.bfloat16, _MAX_TOKENS, True) is None
    )
    for args, needle in [
        ((4, 6144, torch.bfloat16, 4096, True), "world_size=4"),
        ((2, 7168, torch.bfloat16, 4096, True), "hidden_dim=7168"),
        ((2, 6144, torch.float16, 4096, True), "dtype"),
        ((2, 6144, torch.bfloat16, 4096, False), "symmetric device memory"),
        ((2, 6144, torch.bfloat16, allowlist.t_min - 1, True), "max_token_num"),
        ((2, 6144, torch.bfloat16, None, True), "max_token_num"),
    ]:
        reason = ws2.creation_envelope_reason(*args)
        assert reason is not None and needle in reason, (args, reason)

    state = _state()
    t = allowlist.t_min
    x = torch.zeros(t, _HIDDEN, dtype=torch.bfloat16)
    g = torch.zeros(_HIDDEN, dtype=torch.bfloat16)
    given = object()

    def reason(state=state, x=x, r=given, ro=given, no=given, g=g, wb=1.0, fp32=True):
        r = torch.zeros_like(x) if r is given else r
        ro = torch.zeros_like(x) if ro is given else ro
        no = torch.zeros_like(x) if no is given else no
        return ws2._reject_reason(state, x, r, ro, no, g, wb, fp32)

    monkeypatch.setenv(_DISABLE_ENV, "1")
    assert reason() == "kill_switch"
    monkeypatch.delenv(_DISABLE_ENV)
    assert reason(state=None) == "no_state"
    for t_out in (allowlist.t_min - 1, allowlist.t_max + 1):
        if t_out >= 1:
            xo = torch.zeros(t_out, _HIDDEN, dtype=torch.bfloat16)
            # the token count is checked before the semantics
            assert reason(x=xo, fp32=False) == "num_tokens"
    assert reason(x=torch.zeros(1, t, _HIDDEN, dtype=torch.bfloat16)) == "shape"
    assert reason(fp32=False) == "fp32_acc"
    assert reason(wb=0.5) == "weight_bias"
    # The pattern's required inputs.
    assert reason(r=None) == "missing_tensor"
    assert reason(g=None) == "missing_tensor"
    assert reason(x=x.half()) == "dtype"
    assert reason(no=torch.zeros_like(x, dtype=torch.float16)) == "dtype"
    assert reason(g=torch.zeros(_HIDDEN + 8, dtype=torch.bfloat16)) == "shape"
    assert reason(ro=torch.zeros(t, _HIDDEN + 8, dtype=torch.bfloat16)) == "shape"
    assert reason(ro=torch.zeros(t, 2 * _HIDDEN, dtype=torch.bfloat16)[:, ::2]) == (
        "non_contiguous"
    )
    # CPU tensors pass every check but the device one, which comes last.
    assert reason() == "device"
    # Omitted outputs never decide the route: a rank that omits them takes the path of a
    # rank that passes them (they are allocated at dispatch).
    for ro, no in ((None, given), (given, None), (None, None)):
        assert reason(ro=ro, no=no) == "device", (ro, no)
    on_cpu = _state(device_idx=-1)
    for ro, no in ((None, given), (given, None), (None, None), (given, given)):
        assert reason(state=on_cpu, ro=ro, no=no) is None, (ro, no)

    # The cheap pre-check of allreduce_fusion: token count against the state's ranges only.
    class _Ws:
        _ar_ws2 = state

    assert state.token_ranges == tuple(
        (lo, min(hi, state.capacity)) for lo, hi in allowlist.num_tokens
    )
    before = ws2._allreduce_fusion_ws2_specialized_stats()["fallback_reasons"]
    for t_out in (allowlist.t_min - 1, allowlist.t_max + 1):
        if t_out >= 1:
            xo = torch.zeros(t_out, _HIDDEN, dtype=torch.bfloat16)
            assert ws2.outside_token_ranges(_Ws(), xo)
    after = ws2._allreduce_fusion_ws2_specialized_stats()["fallback_reasons"]
    assert after.get("num_tokens", 0) > before.get("num_tokens", 0)
    assert not ws2.outside_token_ranges(_Ws(), x)
    assert not ws2.outside_token_ranges(_Ws(), torch.zeros(1, t, _HIDDEN))

    class _NoState:
        _ar_ws2 = None

    assert not ws2.outside_token_ranges(_NoState(), x)


_ENVELOPE_CASES = [
    ((2, 6144, torch.bfloat16, _MAX_TOKENS, True), None),
    ((2, 6144, torch.bfloat16, ws2._MIN_TOKEN_NUM, True), None),
    ((4, 6144, torch.bfloat16, 4096, True), "world_size=4"),
    ((2, 7168, torch.bfloat16, 4096, True), "hidden_dim=7168"),
    ((2, 6144, torch.float16, 4096, True), "dtype"),
    ((2, 6144, torch.bfloat16, 4096, False), "symmetric device memory"),
    ((2, 6144, torch.bfloat16, ws2._MIN_TOKEN_NUM - 1, True), "max_token_num"),
    ((2, 6144, torch.bfloat16, None, True), "max_token_num"),
]


def test_ws2_specialized_static_envelope_reads_no_file(monkeypatch):
    """The static envelope compares ``max_token_num`` with a module constant (the shipped
    allowlist's smallest token count) and never reads the allowlist, so a rank that cannot read
    it still reaches the vote with its peer; the allowlist is read in each rank's vote."""
    assert ws2._load_allowlist().t_min == ws2._MIN_TOKEN_NUM

    def unreadable():
        raise PermissionError("injected: the allowlist cannot be read")

    monkeypatch.setattr(ws2, "_load_allowlist", unreadable)
    for args, needle in _ENVELOPE_CASES:
        reason = ws2.creation_envelope_reason(*args)
        if needle is None:
            assert reason is None, (args, reason)
        else:
            assert reason is not None and needle in reason, (args, reason)


def test_ws2_specialized_unreadable_allowlist_never_raises(monkeypatch):
    """Any failure to read the allowlist gives an empty one (never dispatch, a veto in the
    vote) instead of an exception on one rank."""
    import importlib.resources

    def unreadable(package):
        raise PermissionError(f"injected: {package} cannot be read")

    monkeypatch.setattr(ws2, "_allowlist_error", ws2._allowlist_error)
    monkeypatch.setattr(importlib.resources, "files", unreadable)
    allowlist = ws2._load_allowlist.__wrapped__()
    assert allowlist.num_tokens == () and allowlist.weight_bias == ()
    assert ws2._allowlist_error.startswith("PermissionError")


@pytest.mark.parametrize(
    "allowlist, max_token_num, expected",
    [
        (
            ws2._Allowlist(weight_bias=(), num_tokens=()),
            _MAX_TOKENS,
            "allowlist_unavailable",
        ),
        (
            ws2._Allowlist(weight_bias=(1.0,), num_tokens=((1024, 2048),)),
            600,
            "capacity",
        ),
    ],
)
def test_ws2_specialized_vote_reads_the_allowlist(
    monkeypatch, allowlist, max_token_num, expected
):
    """Each rank's vote reads the allowlist: an unavailable one, or one whose smallest token
    count exceeds ``max_token_num``, is a veto (on every rank alike, through the vote)."""
    monkeypatch.setattr(ws2, "_load_allowlist", lambda: allowlist)
    monkeypatch.setattr(ws2, "get_compute_capability", lambda device: (10, 3))
    backend = SimpleNamespace(Get_size=lambda: 2, Get_rank=lambda: 0)
    reason, *_ = ws2._local_probe(torch.device("cpu"), max_token_num, backend, 0)
    assert reason == expected


# ---------------------------------------------------------------------------
# The dispatch path with a stand-in launcher (CPU tensors, state on "device" -1).
# ---------------------------------------------------------------------------


class _FakeLauncher:
    """Stands in for the JIT module: checks the launcher's alignment contract (returns 1,
    without "launching", when a tensor is not 16-byte aligned) and computes a stand-in result
    with the kernel's aliasing semantics: every input is read before any output is written."""

    def __init__(self):
        self.launches = []
        self.refusals = 0

    def allreduce_fusion_ws2_ar_rmsnorm(self, x, r, g, ro, no, *args):
        if any(t.data_ptr() % 16 for t in (x, r, g, ro, no)):
            self.refusals += 1
            return 1
        h = x.float() + r.float()
        res, norm = h.to(x.dtype), (2.0 * h + g.float()).to(x.dtype)
        ro.copy_(res)
        no.copy_(norm)
        self.launches.append(tuple(t.data_ptr() for t in (x, r, g, ro, no)))
        return 0


@pytest.fixture
def fake_launcher(monkeypatch):
    launcher = _FakeLauncher()
    monkeypatch.setattr(ws2, "_module", launcher)
    return launcher


def _dispatch_workspace():
    return SimpleNamespace(
        _ar_ws2=_state(device_idx=-1, handle=SimpleNamespace(mapped=True))
    )


def _misaligned(t):
    """A contiguous view of ``t``'s values starting one element (2 bytes) into an aligned
    allocation (e.g. ``torch.narrow`` of a larger buffer)."""
    flat = torch.empty(t.numel() + 8, dtype=t.dtype)
    view = flat[1 : 1 + t.numel()].view(t.shape)
    view.copy_(t)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    return view


def _inputs(seed, t=None):
    t = ws2._load_allowlist().t_min if t is None else t
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(t, _HIDDEN, generator=gen).bfloat16()
    r = torch.randn(t, _HIDDEN, generator=gen).bfloat16()
    g = torch.randn(_HIDDEN, generator=gen).bfloat16()
    return x, r, g


def _expected(x, r, g):
    h = x.float() + r.float()
    return h.to(x.dtype), (2.0 * h + g.float()).to(x.dtype)


def _dispatch(ws, x, r, g, ro, no):
    return ws2.try_dispatch(ws, x, r, ro, no, g, 1e-6, 1.0, True, True)


def test_ws2_specialized_dispatch_allocates_omitted_outputs(fake_launcher):
    """An omitted ``residual_out`` / ``norm_out`` is allocated and the call dispatches, so a
    rank that omits them runs the kernel its peer runs."""
    ws = _dispatch_workspace()
    x, r, g = _inputs(1)
    res, norm = _expected(x, r, g)
    for omit_ro, omit_no in ((True, False), (False, True), (True, True)):
        ro = None if omit_ro else torch.empty_like(x)
        no = None if omit_no else torch.empty_like(x)
        before = ws2._allreduce_fusion_ws2_specialized_stats()["dispatch_count"]
        out = _dispatch(ws, x, r, g, ro, no)
        assert out is not None, (omit_ro, omit_no)
        assert ws2._allreduce_fusion_ws2_specialized_stats()["dispatch_count"] == (
            before + 1
        )
        assert torch.equal(out, norm)
        if no is not None:
            assert out is no
        if ro is not None:
            assert torch.equal(ro, res)


@pytest.mark.parametrize(
    "misaligned",
    [("g",), ("x",), ("r",), ("ro",), ("no",), ("x", "r", "g", "ro", "no")],
)
@pytest.mark.parametrize(
    "aliasing", ["separate", "residual_out_is_input", "norm_out_is_input"]
)
def test_ws2_specialized_dispatch_stages_misaligned_views(
    fake_launcher, misaligned, aliasing
):
    """A contiguous view that is not 16-byte aligned still runs the kernel (through aligned
    copies on its rank), never the fallback; results land in the caller's tensors, also when
    an output aliases an input."""
    ws = _dispatch_workspace()
    x0, r0, g0 = _inputs(2)
    res, norm = _expected(x0, r0, g0)
    t = {"x": x0.clone(), "r": r0.clone(), "g": g0.clone()}
    for name in ("x", "r", "g"):
        if name in misaligned:
            t[name] = _misaligned(t[name])
    if aliasing == "separate":
        t["ro"], t["no"] = torch.empty_like(x0), torch.empty_like(x0)
        for name in ("ro", "no"):
            if name in misaligned:
                t[name] = _misaligned(t[name])
    elif aliasing == "residual_out_is_input":
        t["ro"], t["no"] = t["x"], torch.empty_like(x0)
        if "no" in misaligned:
            t["no"] = _misaligned(t["no"])
    else:
        t["ro"], t["no"] = t["r"], t["x"]
    before = ws2._allreduce_fusion_ws2_specialized_stats()
    out = _dispatch(ws, t["x"], t["r"], t["g"], t["ro"], t["no"])
    after = ws2._allreduce_fusion_ws2_specialized_stats()
    assert out is t["no"]
    assert after["dispatch_count"] == before["dispatch_count"] + 1
    any_misaligned = any(t[n].data_ptr() % 16 for n in t)
    assert after["staged_launches"] == before["staged_launches"] + int(any_misaligned)
    assert fake_launcher.refusals == int(any_misaligned)
    assert len(fake_launcher.launches) == 1
    assert all(p % 16 == 0 for p in fake_launcher.launches[0])
    assert torch.equal(t["ro"], res) and torch.equal(t["no"], norm)


def test_ws2_specialized_dispatch_fallbacks_do_not_launch(fake_launcher, monkeypatch):
    """Calls the guard rejects reach no launch (the fallback backend serves them)."""
    ws = _dispatch_workspace()
    x, r, g = _inputs(3)
    assert ws2.try_dispatch(ws, x, r, None, None, g, 1e-6, 1.0, False, True) is None
    assert ws2.try_dispatch(ws, x, None, None, None, g, 1e-6, 1.0, True, True) is None
    monkeypatch.setenv(_DISABLE_ENV, "1")
    assert _dispatch(ws, x, r, g, None, None) is None
    monkeypatch.delenv(_DISABLE_ENV)
    assert not fake_launcher.launches and not fake_launcher.refusals


@pytest.mark.parametrize("aliasing", ["separate", "norm_out_is_input"])
def test_ws2_specialized_staged_outputs_may_be_inference_tensors(
    fake_launcher, aliasing
):
    """Misaligned outputs allocated under ``torch.inference_mode()`` and dispatched outside it
    get the staged results (the copy-back runs under inference mode)."""
    ws = _dispatch_workspace()
    x0, r0, g0 = _inputs(4)
    res, norm = _expected(x0, r0, g0)
    with torch.inference_mode():
        if aliasing == "separate":
            x, r, g = x0.clone(), r0.clone(), g0.clone()
            ro, no = (
                _misaligned(torch.empty_like(x0)),
                _misaligned(torch.empty_like(x0)),
            )
        else:
            x, r, g = _misaligned(x0), _misaligned(r0), g0.clone()
            ro, no = r, x
    assert ro.is_inference() and no.is_inference()
    before = ws2._allreduce_fusion_ws2_specialized_stats()["staged_launches"]
    out = _dispatch(ws, x, r, g, ro, no)
    assert out is no
    assert (
        ws2._allreduce_fusion_ws2_specialized_stats()["staged_launches"] == before + 1
    )
    assert torch.equal(ro, res) and torch.equal(no, norm)


# ---------------------------------------------------------------------------
# The creation votes on two simulated ranks (threads and an in-process allgather).
# ---------------------------------------------------------------------------


class _ThreadGroup:
    """An in-process two-rank allgather. A rank that makes fewer or more collective calls than
    its peer breaks the barrier (BrokenBarrierError) instead of hanging the test."""

    def __init__(self, size=2, timeout=20.0):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=timeout)
        self.slots = [None] * size
        self.calls = [0] * size

    def backend(self, rank):
        group = self

        class _Backend:
            def Get_rank(self):
                return rank

            def Get_size(self):
                return group.size

            def allgather(self, obj):
                group.calls[rank] += 1
                group.slots[rank] = obj
                group.barrier.wait()
                out = list(group.slots)
                group.barrier.wait()
                return out

            def barrier(self):
                self.allgather(None)

        return _Backend()


class _FakeRegion:
    """Stands in for ``SymmDeviceMemory``: one collective (the handle exchange) and a local
    failure after it on the ranks in ``fail_after_exchange``; ``live`` counts the regions not
    yet released."""

    live = 0
    fail_after_exchange: set = set()

    def __init__(self, buf_size, group_size, group_rank, device_idx, **kwargs):
        kwargs["comm_backend_for_handle_transfer"].allgather(("handle", group_rank))
        if group_rank in _FakeRegion.fail_after_exchange:
            raise RuntimeError(f"injected local failure on rank {group_rank}")
        type(self).live += 1
        self.mapped = True

    def __del__(self):
        if getattr(self, "mapped", False):
            type(self).live -= 1


def _run_create_state(monkeypatch, fail_after_exchange=(), fail_init=()):
    _FakeRegion.live = 0
    _FakeRegion.fail_after_exchange = set(fail_after_exchange)
    monkeypatch.setattr(ws2, "SymmDeviceMemory", _FakeRegion)
    monkeypatch.setattr(
        ws2,
        "_module",
        SimpleNamespace(allreduce_fusion_ws2_region_bytes=lambda capacity: 1024),
    )
    monkeypatch.setattr(
        ws2,
        "_local_probe",
        lambda device, max_token_num, backend, tp_rank: (None, "", 4096, 8, 1024, True),
    )
    monkeypatch.setattr(ws2, "_node_id", lambda: "node")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def init_state(handle, tp_rank, device, capacity, grid, region_bytes):
        if tp_rank in fail_init:
            raise RuntimeError(f"injected init failure on rank {tp_rank}")
        return SimpleNamespace(
            handle=handle, rank=tp_rank, capacity=capacity, grid=grid
        )

    monkeypatch.setattr(ws2, "_init_state", init_state)
    group = _ThreadGroup()
    results = [None, None]
    errors = []

    def rank_main(rank):
        try:
            results[rank] = ws2.create_state(
                tp_rank=rank,
                max_token_num=_MAX_TOKENS,
                comm_backend=group.backend(rank),
                group=None,
            )
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append((rank, exc))

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(60)
    assert not any(th.is_alive() for th in threads), "a rank hung in create_state"
    assert not errors, errors
    return results, group


def test_ws2_specialized_create_state_success(monkeypatch):
    results, group = _run_create_state(monkeypatch)
    assert group.calls[0] == group.calls[1]
    assert [rejection for _, rejection in results] == [None, None]
    assert [state.rank for state, _ in results] == [0, 1]
    assert _FakeRegion.live == 2
    results.clear()
    assert _FakeRegion.live == 0


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_ws2_specialized_create_state_local_allocation_failure(
    monkeypatch, failing_rank
):
    """A local failure after the handle exchange on one rank (e.g. mapping or the pointer
    table): every rank reaches the same vote, rejects alike and releases its region; no rank
    returns while its peer still waits in a collective."""
    results, group = _run_create_state(monkeypatch, fail_after_exchange={failing_rank})
    assert group.calls[0] == group.calls[1]
    for state, rejection in results:
        assert state is None
        assert rejection[0] == "allocation"
        assert f"rank {failing_rank}" in rejection[1]
    assert _FakeRegion.live == 0, "a rank kept its region after the rejection"


def test_ws2_specialized_create_state_local_init_failure(monkeypatch):
    results, group = _run_create_state(monkeypatch, fail_init={0})
    assert group.calls[0] == group.calls[1]
    assert [rej for _, rej in results] == [("init", results[0][1][1])] * 2
    assert all(state is None for state, _ in results)
    assert "rank 0" in results[0][1][1]
    assert _FakeRegion.live == 0


# ---------------------------------------------------------------------------
# The factory: backend selection with stand-ins for the stock workspaces.
# ---------------------------------------------------------------------------


class _FakeStockWorkspace(comm.AllReduceFusionWorkspace):
    """Stands in for a stock workspace in the selection tests."""

    created: list = []

    def __init__(self, name, **kwargs):
        world_size = kwargs.get("tp_size") or kwargs["mapping"].world_size
        rank = kwargs.get("tp_rank")
        if rank is None:
            rank = kwargs["mapping"].rank
        super().__init__(world_size, rank)
        self.name = name
        self.destroyed = False
        _FakeStockWorkspace.created.append(self)

    @property
    def backend(self):
        return self.name

    def is_buffer_size_sufficient(self, *args, **kwargs):
        return True

    def destroy(self):
        self.destroyed = True
        self._destroyed = True


def _releasable_state():
    """A stand-in for the kernel state that ``release_state`` can release."""
    return SimpleNamespace(handle=object(), device_idx=0)


def _patch_factory(monkeypatch, multicast, state_result):
    """Patch the stock workspaces, the multicast probe and the specialized vote so that
    create_allreduce_fusion_workspace runs without GPUs. Returns the list of vote calls."""
    import flashinfer.comm.allreduce as ar

    _FakeStockWorkspace.created = []
    monkeypatch.setattr(
        ar,
        "TRTLLMAllReduceFusionWorkspace",
        lambda **kw: _FakeStockWorkspace("trtllm", **kw),
    )

    class _Mnnvl:
        get_required_buffer_size_bytes = staticmethod(lambda *a, **k: 1 << 20)

        def __new__(cls, **kw):
            return _FakeStockWorkspace("mnnvl", **kw)

    monkeypatch.setattr(ar, "MNNVLAllReduceFusionWorkspace", _Mnnvl)
    monkeypatch.setattr(ar, "is_multicast_supported", lambda device: multicast)
    monkeypatch.setattr(
        ar, "all_ranks_support_mnnvl", lambda local, ws, cb, group: local
    )
    monkeypatch.setattr(ar, "is_confidential_compute", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    votes = []

    def create_state(**kwargs):
        votes.append(kwargs)
        return state_result

    monkeypatch.setattr(ws2, "create_state", create_state)
    return votes


def _factory(backend, world_size=2, hidden_dim=6144, dtype=torch.bfloat16, **kw):
    return comm.create_allreduce_fusion_workspace(
        backend=backend,
        world_size=world_size,
        rank=0,
        max_token_num=kw.pop("max_token_num", _MAX_TOKENS),
        hidden_dim=hidden_dim,
        dtype=dtype,
        gpus_per_node=4,
        comm_backend=object(),
        **kw,
    )


def _destroy_and_check(ws):
    """Destroy a successful specialized workspace: its state is released and its fallback
    destroyed."""
    released = ws2._allreduce_fusion_ws2_specialized_stats()["states_released"]
    had_state = ws.has_specialized_kernels
    ws.destroy()
    assert ws.fallback.destroyed and ws._ar_ws2 is None
    stats = ws2._allreduce_fusion_ws2_specialized_stats()
    assert stats["states_released"] == released + int(had_state)
    ws.destroy()  # idempotent


@pytest.mark.parametrize("multicast", [True, False])
def test_ws2_specialized_auto_selection(monkeypatch, multicast):
    """auto picks the specialized backend first inside the creation envelope, with the stock
    pick as its fallback; outside the envelope or on a veto, auto returns exactly the stock
    pick (mnnvl with multicast, else trtllm); trtllm / mnnvl never consult the specialized
    backend."""
    stock = "mnnvl" if multicast else "trtllm"

    votes = _patch_factory(monkeypatch, multicast, (_releasable_state(), None))
    ws = _factory("auto")
    assert isinstance(ws, comm.SpecializedAllReduceFusionWorkspace)
    assert ws.backend == "specialized" and ws.fallback_backend == stock
    assert ws.has_specialized_kernels and len(votes) == 1
    assert [w.name for w in _FakeStockWorkspace.created] == [stock]
    _destroy_and_check(ws)

    for backend in ("trtllm", "mnnvl"):
        votes = _patch_factory(monkeypatch, multicast, (_releasable_state(), None))
        ws = _factory(backend)
        assert ws.backend == backend and not votes
        ws.destroy()

    # Outside the static envelope: no vote (no collective) and the stock pick.
    for kwargs in (
        dict(world_size=4),
        dict(hidden_dim=7168),
        dict(dtype=torch.float16),
        dict(max_token_num=512),
    ):
        votes = _patch_factory(monkeypatch, multicast, (_releasable_state(), None))
        ws = _factory("auto", **kwargs)
        assert isinstance(ws, _FakeStockWorkspace) and ws.backend == stock, kwargs
        assert not votes, kwargs
        ws.destroy()

    # A veto (e.g. another compute capability, the kill switch): the stock pick itself.
    for rejection in (("compute_capability", "rank 0"), ("kill_switch", "rank 1")):
        votes = _patch_factory(monkeypatch, multicast, (None, rejection))
        ws = _factory("auto")
        assert isinstance(ws, _FakeStockWorkspace) and ws.backend == stock
        assert len(votes) == 1 and not ws.destroyed
        ws.destroy()


@pytest.mark.parametrize("multicast", [True, False])
def test_ws2_specialized_explicit_selection(monkeypatch, multicast):
    """An explicit specialized workspace has the stock pick as its fallback, has no kernel
    state with the kill switch, and raises (after releasing its fallback) when the kernel
    cannot serve the workspace."""
    stock = "mnnvl" if multicast else "trtllm"

    _patch_factory(monkeypatch, multicast, (_releasable_state(), None))
    ws = _factory("specialized")
    assert isinstance(ws, comm.SpecializedAllReduceFusionWorkspace)
    assert ws.fallback_backend == stock and ws.has_specialized_kernels
    _destroy_and_check(ws)

    _patch_factory(monkeypatch, multicast, (None, ("kill_switch", "rank 0")))
    ws = _factory("specialized")
    assert isinstance(ws, comm.SpecializedAllReduceFusionWorkspace)
    assert ws.fallback_backend == stock and not ws.has_specialized_kernels
    _destroy_and_check(ws)

    _patch_factory(monkeypatch, multicast, (None, ("compute_capability", "rank 1")))
    with pytest.raises(ValueError, match="compute_capability"):
        _factory("specialized")
    assert [w.destroyed for w in _FakeStockWorkspace.created] == [True]

    votes = _patch_factory(monkeypatch, multicast, (_releasable_state(), None))
    with pytest.raises(ValueError, match="world_size=4"):
        _factory("specialized", world_size=4)
    assert not votes and not _FakeStockWorkspace.created


# ---------------------------------------------------------------------------
# The buffer check: use_oneshot as allreduce_fusion passes it to the fallback backend.
# ---------------------------------------------------------------------------


class _RecordingFallback(comm.AllReduceFusionWorkspace):
    def __init__(self):
        super().__init__(2, 0)
        self.args = None

    @property
    def backend(self):
        return "trtllm"

    def is_buffer_size_sufficient(self, *args):
        self.args = args
        return True

    def destroy(self):
        self._destroyed = True


def test_ws2_specialized_buffer_check_translates_use_oneshot():
    """An MNNVL fallback checks the strategy a call with ``use_oneshot`` runs (one-shot /
    two-shot), as ``allreduce_fusion`` translates it; ``None`` keeps the backend's default
    and an explicit strategy passes through; other fallbacks get ``use_oneshot`` as is."""
    from flashinfer.comm.trtllm_mnnvl_ar import (
        MNNVLAllReduceFusionWorkspace,
        MNNVLAllreduceFusionStrategy as Strategy,
    )

    tp, h, dt = 2, _HIDDEN, torch.bfloat16
    # Three tokens: one-shot needs 3 * tp rows, two-shot 2 * ceil(3 / tp) * tp = 8 rows.
    n = 3
    oneshot = MNNVLAllReduceFusionWorkspace.get_required_buffer_size_bytes(
        tp, n, h, dt, Strategy.ONESHOT
    )
    twoshot = MNNVLAllReduceFusionWorkspace.get_required_buffer_size_bytes(
        tp, n, h, dt, Strategy.TWOSHOT
    )
    assert oneshot < twoshot
    mnnvl = object.__new__(MNNVLAllReduceFusionWorkspace)
    mnnvl.world_size, mnnvl.rank, mnnvl._destroyed = tp, 0, True
    mnnvl.buffer_size_bytes = oneshot
    ws = comm.SpecializedAllReduceFusionWorkspace(mnnvl, None)
    assert ws.is_buffer_size_sufficient(tp, n, h, dt, use_oneshot=True)
    assert not ws.is_buffer_size_sufficient(tp, n, h, dt, use_oneshot=False)
    assert ws.is_buffer_size_sufficient(
        tp, n, h, dt, use_oneshot=Strategy.ONESHOT
    ) == mnnvl.is_buffer_size_sufficient(tp, n, h, dt, Strategy.ONESHOT)
    assert ws.is_buffer_size_sufficient(
        tp, n, h, dt
    ) == mnnvl.is_buffer_size_sufficient(tp, n, h, dt)
    ws.destroy()

    recording = _RecordingFallback()
    ws = comm.SpecializedAllReduceFusionWorkspace(recording, None)
    for use_oneshot in (True, False):
        assert ws.is_buffer_size_sufficient(tp, n, h, dt, use_oneshot=use_oneshot)
        assert recording.args == (tp, n, h, dt, use_oneshot)
    assert ws.is_buffer_size_sufficient(tp, n, h, dt)
    assert recording.args == (tp, n, h, dt)
    ws.destroy()
