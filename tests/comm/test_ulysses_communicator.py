# flashinfer: UlyssesCommunicator public-API tests.
# Single-rank (gloo) tests cover construction, validation, lifecycle, and the
# "fallback never touches IPC/JIT" guarantee at the real constructor entry.
# Multi-rank (spawn) tests cover both collectives against independent
# references with the actual backend asserted, staged-init fault injection
# with resource accounting, retryable close, and the device contract — all
# with timeout + terminate + natural-exit assertions.

import contextlib
import datetime
import importlib
import multiprocessing as std_mp
import os
import queue as queue_mod
import tempfile
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator
from flashinfer.comm.ulysses import missing_ulysses_pcie_dependencies
from flashinfer.comm.ulysses_topology import UlyssesBackendError, UlyssesRankTopology


def _full_mesh(world_size, hostname="hostA"):
    uuids = [f"GPU-fake-{i}" for i in range(world_size)]
    return [
        UlyssesRankTopology(
            rank=r,
            hostname=hostname,
            device_index=r,
            device_uuid=uuids[r],
            pci_bus_id=f"0000:{r:02x}:00.0",
            peer_p2p={uuids[p]: True for p in range(world_size) if p != r},
            peer_nvlink={uuids[p]: True for p in range(world_size) if p != r},
        )
        for r in range(world_size)
    ]


def _patch_probe_mesh_module(world_size, break_nvlink=False, error_rank=None):
    """Patch the probe inside a worker process (no monkeypatch fixture)."""
    topo_mod = importlib.import_module("flashinfer.comm.ulysses_topology")

    def fake_probe(device, r, *, probe_pcie=True):
        if error_rank is not None and r == error_rank:
            raise RuntimeError("injected probe failure")
        topos = _full_mesh(world_size)
        if break_nvlink:
            topos[1].peer_nvlink[topos[0].device_uuid] = False
        return topos[r]

    topo_mod.probe_ulysses_rank_topology = fake_probe


# ---- independent references (do not use the kernel or each other) ------------


def _ref_scatter_heads(x_local, world_size, rank, group):
    """out_r[b, j*S_local + s, hl, d] = x_j[b, s, r*H_local + hl, d]"""
    H = x_local.shape[2]
    H_local = H // world_size
    gathered = [torch.empty_like(x_local) for _ in range(world_size)]
    dist.all_gather(gathered, x_local.contiguous(), group=group)
    slabs = [xj[:, :, rank * H_local : (rank + 1) * H_local, :] for xj in gathered]
    return torch.cat(slabs, dim=1).contiguous()


def _ref_gather_heads(y_local, world_size, rank, group):
    """out_r[b, s, p*H_local + hl, d] = y_p[b, r*S_local + s, hl, d]"""
    S_global = y_local.shape[1]
    S_local = S_global // world_size
    gathered = [torch.empty_like(y_local) for _ in range(world_size)]
    dist.all_gather(gathered, y_local.contiguous(), group=group)
    blocks = [yp[:, rank * S_local : (rank + 1) * S_local, :, :] for yp in gathered]
    return torch.cat(blocks, dim=2).contiguous()


# ---- rendezvous ---------------------------------------------------------------


def _fresh_rendezvous_path():
    """A unique, unused path for a FileStore rendezvous.

    Not a TCP port: binding and closing a socket to pick one leaves a race,
    and each test starts eight workers.
    """
    handle, path = tempfile.mkstemp(prefix="flashinfer_ulysses_pg_")
    os.close(handle)
    # torch's FileStore wants to create the file itself.
    os.unlink(path)
    return path


@contextlib.contextmanager
def _rendezvous_path():
    path = _fresh_rendezvous_path()
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


# ---- single-rank fixtures -----------------------------------------------------


@pytest.fixture
def gloo_pg():
    with _rendezvous_path() as rendezvous:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous}",
            rank=0,
            world_size=1,
        )
        try:
            yield dist.group.WORLD
        finally:
            dist.destroy_process_group()


def _forbid_ipc_and_jit(monkeypatch):
    cuda_ipc_mod = importlib.import_module("flashinfer.comm.cuda_ipc")
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    vllm_ar_mod = importlib.import_module("flashinfer.comm.vllm_ar")

    def _boom(*args, **kwargs):
        raise AssertionError("IPC/JIT entry point must not be touched")

    monkeypatch.setattr(cuda_ipc_mod, "create_shared_buffer", _boom)
    monkeypatch.setattr(cuda_ipc_mod.cudart, "cudaMalloc", _boom, raising=False)
    monkeypatch.setattr(ulysses_mod, "get_ulysses_a2a_module", _boom)
    monkeypatch.setattr(ulysses_mod, "init_ulysses_a2a", _boom)
    monkeypatch.setattr(vllm_ar_mod, "meta_size", _boom)
    # merged module binds gen at import: patch the local binding
    monkeypatch.setattr(ulysses_mod, "gen_ulysses_a2a_module", _boom)


def _patch_probe_mesh(monkeypatch, world_size):
    monkeypatch.setattr(
        "flashinfer.comm.ulysses_topology.probe_ulysses_rank_topology",
        lambda device, rank, *, probe_pcie=True: _full_mesh(world_size)[rank],
    )


def _make_w1(gloo_pg, monkeypatch, backend="auto", max_bytes=1 << 21):
    _patch_probe_mesh(monkeypatch, 1)
    return UlyssesCommunicator(
        gloo_pg, max_bytes=max_bytes, dtype=torch.float16, backend=backend
    )


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _fake_operand(shape, *, ptr=None):
    """Host-only stand-in for what ``_pcie_exchange`` touches: shape, data_ptr."""
    return SimpleNamespace(shape=shape, data_ptr=lambda: ptr)


def _make_mock_pcie_comm():
    """Minimal host-only communicator for PCIe transaction unit tests."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = object.__new__(UlyssesCommunicator)
    comm._state = ulysses_mod._OPEN
    comm._broken_reason = None
    comm._pcie = 123
    comm._pcie_armed = True
    comm._pcie_outputs = {}
    comm._pcie_stream = None
    comm._pcie_python_teardown_safe = True
    comm._nvlink_armed = False
    comm._fa = None
    comm.device = torch.device("cpu")
    comm.dtype = torch.float16
    comm.max_bytes = 1 << 21
    comm.backend = "pcie"
    comm.transport = "hybrid"
    comm.rank = 0
    comm.world_size = 2
    comm.group = object()
    comm._gather = lambda payload: [payload, payload]
    comm.decision = SimpleNamespace(
        reason="explicit PCIe backend",
        pcie_plan=SimpleNamespace(
            transport="hybrid",
            numa_nodes=(0, 1),
            nic_names=("mlx5_0", "mlx5_1"),
            gid_indices=(2, 3),
        ),
    )
    comm.fallback_reason = None
    return comm


# ---- PCIe host/mock regressions -----------------------------------------------


def test_out_storage_overlap_is_rejected_before_backend_launch(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    comm.backend = "nvlink"
    comm._fa = 1
    x = torch.empty((1, 2, 4, 2), dtype=torch.float16)
    overlapping = x.view(1, 4, 2, 2)
    monkeypatch.setattr(
        ulysses_mod,
        "ulysses_a2a",
        lambda *args, **kwargs: pytest.fail("backend must not be launched"),
    )
    with pytest.raises(ValueError, match="must not overlap input storage"):
        comm.scatter_heads(x, out=overlapping)


def test_pcie_explicit_output_uses_native_owned_allocation(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    calls = []
    x = torch.empty((1, 2, 4, 2), dtype=torch.float16)
    # The native slot is allocated flat at max_bytes; the caller gets a view.
    storage = torch.empty(1 << 20, dtype=torch.float16)
    module = SimpleNamespace(
        allocate_output=lambda *_args: (calls.append("allocate"), (storage, [1, 2, 3]))[
            1
        ],
        register_output=lambda *_args: pytest.fail(
            "native allocation is already registered"
        ),
        connect_output=lambda *_args: calls.append("connect"),
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)

    shape = (1, 4, 2, 2)
    result = comm.allocate_output(x, "scatter_heads")
    assert tuple(result.shape) == shape
    assert result.data_ptr() == storage.data_ptr()
    assert calls == ["allocate", "connect"]
    assert comm._pcie_outputs[storage.data_ptr()] == 0


@pytest.mark.parametrize(
    ("op", "shape"),
    [
        ("scatter_heads", (1, 2, 4, 2)),
        ("gather_heads", (1, 4, 2, 2)),
    ],
)
def test_multirank_pcie_requires_explicit_output(monkeypatch, op, shape):
    comm = _make_mock_pcie_comm()
    x = torch.empty(shape, dtype=torch.float16)
    outputs_before = dict(comm._pcie_outputs)
    monkeypatch.setattr(
        comm,
        "_pcie_exchange",
        lambda *_args: pytest.fail("native exchange must not run without out="),
    )

    with pytest.raises(ValueError, match="requires out= from allocate_output"):
        getattr(comm, op)(x)
    assert comm._pcie_outputs == outputs_before


def test_pcie_native_allocation_failure_participates_in_rollback(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    module = SimpleNamespace(
        allocate_output=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("injected allocation failure")
        ),
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())

    gather_count = {"value": 0}

    def gather(payload):
        gather_count["value"] += 1
        # This rank failed before it obtained a Tensor; another rank may
        # already own a native registration and will roll it back locally.
        peer_ok = (("ok", ("scatter_heads", (1, 2, 4, 2), "torch.float16")), [1, 2, 3])
        return [payload, peer_ok if gather_count["value"] == 1 else payload]

    comm._gather = gather
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    x = torch.empty((1, 2, 4, 2), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="BROKEN state"):
        comm.allocate_output(x, "scatter_heads")
    assert comm._pcie_outputs == {}
    assert comm._state == ulysses_mod._BROKEN


@pytest.mark.parametrize("failure_stage", ["connect", "connect_outcome"])
def test_pcie_registration_failure_poisons_and_defers_cleanup(
    monkeypatch, failure_stage
):
    """A failed registration poisons and leaves the pointer for close().

    Rolling back in place would need a ledger that survives a rollback whose own
    outcome exchange fails; close() already walks every registered pointer and
    calls the idempotent native disconnect/dispose, so the pointer only has to
    be recorded before anything can fail.
    """
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    calls = []
    block = torch.empty(comm.max_bytes // 2, dtype=torch.float16)

    def boom(name):
        def fail(*_args):
            raise RuntimeError(f"injected {name}")

        return fail

    module = SimpleNamespace(
        allocate_output=lambda *_args: (block, [1, 2, 3]),
        connect_output=(
            boom("connect") if failure_stage == "connect" else (lambda *_args: None)
        ),
        disconnect_output_ptr=lambda *_args: calls.append("disconnect_ptr"),
        dispose_output_ptr=lambda *_args: calls.append("dispose_ptr"),
        dispose=lambda *_args: calls.append("dispose_transport"),
        teardown_safe=lambda *_args: 1,
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)

    if failure_stage == "connect_outcome":
        seen = {"count": 0}

        def gather(payload):
            if payload == ("ok",):
                seen["count"] += 1
                if seen["count"] == 1:
                    raise RuntimeError("injected connect_outcome")
            return [payload, payload]

        comm._gather = gather

    x = torch.empty((1, 2, 4, 2), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="BROKEN state"):
        comm.allocate_output(x, "scatter_heads")
    assert comm._state == ulysses_mod._BROKEN
    # The pointer is on the books even though the registration never completed.
    assert block.data_ptr() in comm._pcie_outputs

    comm._gather = lambda payload: [payload, payload]
    comm.close()
    assert calls == ["disconnect_ptr", "dispose_ptr", "dispose_transport"]
    assert comm._pcie_outputs == {}
    assert comm._state == ulysses_mod._CLOSED


@pytest.mark.parametrize(
    "failure_stage", ["missing_output", "capture", "current_stream"]
)
def test_pcie_p2p_pre_enqueue_failure_poisons_collective_close(
    monkeypatch, failure_stage
):
    """A local pre-enqueue failure may strand a peer's asynchronous barrier."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    comm.transport = "p2p"
    module = SimpleNamespace(
        teardown_safe=lambda *_args: pytest.fail(
            "Python poison must reject close without trusting a native query"
        ),
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *_args: pytest.fail("poisoned close must not synchronize"),
    )

    x = torch.empty((1, 2, 4, 2), dtype=torch.float16)
    out = torch.empty((1, 4, 2, 2), dtype=torch.float16)
    comm._pcie_outputs[out.data_ptr()] = 0
    if failure_stage == "missing_output":
        out = None
    elif failure_stage == "capture":
        monkeypatch.setattr(
            torch.cuda,
            "is_current_stream_capturing",
            lambda: (_ for _ in ()).throw(RuntimeError("injected capture failure")),
        )
    else:
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda *_args: (_ for _ in ()).throw(
                RuntimeError("injected current-stream failure")
            ),
        )

    with pytest.raises(RuntimeError, match="BROKEN state"):
        comm.scatter_heads(x, out=out)

    assert comm._state == ulysses_mod._BROKEN
    assert comm._pcie_python_teardown_safe is False

    with pytest.raises(RuntimeError, match="process termination required"):
        comm.close()
    assert comm._state == ulysses_mod._CLOSING


def test_pcie_p2p_wrapper_failure_poison_is_sticky(monkeypatch):
    """Wrapper dispatch can fail after stream bookkeeping but before native enqueue."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    comm.transport = "p2p"
    calls = []

    def fail_exchange(*_args):
        calls.append("exchange")
        raise RuntimeError("injected wrapper failure")

    module = SimpleNamespace(exchange=fail_exchange)
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: object())

    x = _fake_operand((1, 2, 4, 2))
    out = SimpleNamespace(data_ptr=lambda: 456)
    comm._pcie_outputs[456] = 0

    with pytest.raises(RuntimeError, match="BROKEN state"):
        comm._pcie_exchange(x, out, 0)

    assert calls == ["exchange"]
    assert comm._state == ulysses_mod._BROKEN
    assert comm._pcie_python_teardown_safe is False


def test_pcie_unsafe_native_work_blocks_every_teardown_stage(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    comm._pcie_outputs[456] = object()
    calls = []
    module = SimpleNamespace(
        teardown_safe=lambda *_args: (calls.append("teardown_safe"), 0)[1],
        disconnect_output_ptr=lambda *_args: pytest.fail("must not disconnect"),
        dispose_output_ptr=lambda *_args: pytest.fail("must not dispose output"),
        dispose=lambda *_args: pytest.fail("must not dispose transport"),
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *_args: pytest.fail("unsafe native work must be checked before sync"),
    )

    with pytest.raises(RuntimeError, match="process termination required"):
        comm.close()

    assert calls == ["teardown_safe"] * comm._TEARDOWN_ATTEMPTS
    assert 456 in comm._pcie_outputs
    assert comm._pcie == 123
    assert comm._state == ulysses_mod._CLOSING


def test_pcie_close_retry_after_peer_dispose_failure(monkeypatch):
    """A failed group close must leave ``_pcie_armed`` set: the documented
    recovery is retrying ``close()`` on all ranks, and a rank that disarmed
    after its own dispose would skip ``_pcie_close()`` on the retry and leave
    its peers gathering against a missing rank."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    module = SimpleNamespace(
        teardown_safe=lambda *_args: True,
        disconnect_output_ptr=lambda *_args: None,
        dispose_output_ptr=lambda *_args: None,
        dispose=lambda *_args: None,
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)

    # A peer's transport dispose keeps failing while every local step
    # succeeds. Rank-locally the dispose stage is the first gather issued
    # after this rank's transport handle went away.
    state = {"gathers": 0, "peer_broken": True}

    def gather(payload):
        state["gathers"] += 1
        if state["peer_broken"] and comm._pcie is None:
            return [payload, (1, "rank 1 PCIe transport teardown: peer failed")]
        return [payload, (0, None)]

    comm._gather = gather

    with pytest.raises(RuntimeError, match=r"retry close\(\) on all ranks"):
        comm.close()
    assert comm._pcie is None  # local dispose succeeded...
    assert comm._pcie_armed is True  # ...but the rank must stay enrolled
    assert comm._state == ulysses_mod._CLOSING

    state["peer_broken"] = False
    gathers_before = state["gathers"]
    comm.close()
    assert state["gathers"] > gathers_before  # rejoined the collective stages
    assert comm._pcie_armed is False
    assert comm._state == ulysses_mod._CLOSED


def test_pcie_rotating_input_has_no_python_control_collective(monkeypatch):
    """Native pre-registers the landing path; pointer changes stay local."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    calls = []

    module = SimpleNamespace(
        exchange=lambda *_args: calls.append("exchange"),
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    comm._gather = lambda _payload: pytest.fail(
        "a rotating input must not add a Python control collective"
    )

    out = SimpleNamespace(data_ptr=lambda: 456)
    comm._pcie_outputs[456] = 0

    assert comm._pcie_exchange(_fake_operand((1, 2, 4, 2), ptr=111), out, 0) is out
    assert comm._pcie_exchange(_fake_operand((1, 2, 4, 2), ptr=222), out, 0) is out
    assert calls == ["exchange", "exchange"]


def test_pcie_collectives_bind_to_the_first_caller_stream(monkeypatch):
    """A second stream must raise: peers order copies against the bound stream."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    calls = []

    first = object()
    second = object()
    streams = iter((first, second))
    module = SimpleNamespace(
        exchange=lambda *_args: calls.append("exchange"),
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: next(streams))

    x = torch.empty((1, 2, 4, 2), dtype=torch.float16)
    out = torch.empty((1, 4, 2, 2), dtype=torch.float16)
    comm._pcie_outputs[out.data_ptr()] = 0

    assert comm.scatter_heads(x, out=out) is out
    assert comm._pcie_stream is first
    with pytest.raises(RuntimeError, match="bound to the stream"):
        comm.scatter_heads(x, out=out)
    assert calls == ["exchange"]


def test_pcie_per_call_dtype_is_held_to_the_registered_output(monkeypatch):
    """A per-call dtype cannot drift from the width the group registered.

    ``allocate_output`` is where an element type becomes group-wide: every rank
    all-gathers ``(op, shape, dtype)`` there and a disagreement breaks the group
    before any transfer exists. Each later call is then held to that agreement
    without a collective of its own -- the operand must be what ``dtype=`` says,
    ``out=`` must carry the dtype it was registered with, and native re-checks
    both against ``buffer->dtype``. So a rank cannot reach the transport with an
    element width its peers did not agree to, which is what would let the copy
    and RDMA paths derive different transfer widths from the same exchange.
    """
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    module = SimpleNamespace(
        exchange=lambda *_args: pytest.fail(
            "a dtype disagreement must not reach the transport"
        ),
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: object())

    packed = torch.empty((1, 2, 4, 2), dtype=torch.uint8)

    # An output registered for the construction dtype, reached by a narrower
    # per-call one: the call names a width the group never registered.
    wide_out = torch.empty((1, 4, 2, 2), dtype=torch.float16)
    comm._pcie_outputs[wide_out.data_ptr()] = 0
    with pytest.raises(ValueError, match="out dtype .* does not match the expected"):
        comm.scatter_heads(packed, out=wide_out, dtype=torch.uint8)

    # And the mirror: an output registered narrow, called without dtype= so the
    # communicator's own width applies. Neither direction is silently widened.
    packed_out = torch.empty((1, 4, 2, 2), dtype=torch.uint8)
    comm._pcie_outputs[packed_out.data_ptr()] = 0
    with pytest.raises(ValueError, match="tensor dtype .* does not match the expected"):
        comm.scatter_heads(packed, out=packed_out)


def test_pcie_batch_validation_precedes_capture_and_native(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    x = torch.empty((2, 2, 4, 2), dtype=torch.float16)
    out = torch.empty((2, 4, 2, 2), dtype=torch.float16)
    gather_x = torch.empty((2, 4, 2, 2), dtype=torch.float16)
    gather_out = torch.empty((2, 2, 4, 2), dtype=torch.float16)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: pytest.fail("capture check must follow operand validation"),
    )
    monkeypatch.setattr(
        ulysses_mod,
        "get_ulysses_pcie_module",
        lambda: pytest.fail("native must not be reached"),
    )
    for call in (
        lambda: comm.allocate_output(x, "scatter_heads"),
        lambda: comm.scatter_heads(x),
        lambda: comm.scatter_heads(x, out=out),
        lambda: comm.allocate_output(gather_x, "gather_heads"),
        lambda: comm.gather_heads(gather_x),
        lambda: comm.gather_heads(gather_x, out=gather_out),
    ):
        with pytest.raises(ValueError, match="batch=1"):
            call()
    assert comm._state == ulysses_mod._OPEN


def test_pcie_hybrid_mkey_pitch_validation_precedes_native(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    scatter_x = torch.empty((1, 1, 128, 256), dtype=torch.float16)
    gather_x = torch.empty((1, 2, 64, 256), dtype=torch.float16)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: pytest.fail("capture check must follow mlx5 geometry validation"),
    )
    monkeypatch.setattr(
        ulysses_mod,
        "get_ulysses_pcie_module",
        lambda: pytest.fail("native must not be reached"),
    )
    for call in (
        lambda: comm.allocate_output(scatter_x, "scatter_heads"),
        lambda: comm.scatter_heads(scatter_x),
        lambda: comm.allocate_output(gather_x, "gather_heads"),
        lambda: comm.gather_heads(gather_x),
    ):
        with pytest.raises(ValueError, match=r"element_size <= 65535.*got 65536"):
            call()
    assert comm._state == ulysses_mod._OPEN

    # The limit belongs to mlx5 interleaved MKeys, not the CUDA P2P route.
    comm.transport = "p2p"
    comm._validate(scatter_x, "scatter_heads")
    comm._validate(gather_x, "gather_heads")

    # The largest two-byte row with H=2 below the provider limit remains valid.
    comm.transport = "hybrid"
    comm._validate(
        torch.empty((1, 1, 2, 16_383), dtype=torch.float16),
        "scatter_heads",
    )


def test_chunk_exchange_geometry_is_identity_at_mode_two():
    comm = _make_mock_pcie_comm()
    x = torch.empty((1, 1, 2, 1024), dtype=torch.float16)
    assert comm._output_geometry(x, "exchange_chunks") == ((1, 1, 2, 1024), 2)

    # One chunk per peer, and nothing to iterate over on the leading axes --
    # those two premises are what let the descriptor collapse to a single row.
    for bad, match in (
        (
            torch.empty((2, 1, 2, 8), dtype=torch.float16),
            r"\[1, 1, world_size, chunk\]",
        ),
        (
            torch.empty((1, 3, 2, 8), dtype=torch.float16),
            r"\[1, 1, world_size, chunk\]",
        ),
        (torch.empty((1, 1, 4, 8), dtype=torch.float16), r"one chunk per peer"),
    ):
        with pytest.raises(ValueError, match=match):
            comm._output_geometry(bad, "exchange_chunks")


def test_chunk_exchange_is_exempt_from_the_mkey_pitch_limit():
    """The 65535 limit bounds a stride; a single-row descriptor has none.

    The exemption must not leak to the interleaved transforms, which is the
    half of this that would fail silently -- an over-long row would reach the
    provider and be rejected there instead of here.
    """
    comm = _make_mock_pcie_comm()
    over = torch.empty((1, 1, 2, 65536), dtype=torch.uint8)
    comm._validate(over, "exchange_chunks", torch.uint8)
    with pytest.raises(ValueError, match=r"element_size <= 65535"):
        comm._validate(over, "scatter_heads", torch.uint8)


def test_chunk_exchange_world_size_one_is_identity():
    comm = _make_mock_pcie_comm()
    comm.world_size = 1
    x = torch.arange(8, dtype=torch.float16).reshape(1, 1, 1, 8)
    assert comm.exchange_chunks(x) is x
    out = torch.empty_like(x)
    assert comm.exchange_chunks(x, out) is out
    assert torch.equal(out, x)


def test_chunk_exchange_is_a_registered_output_op(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    x = torch.empty((1, 1, 2, 64), dtype=torch.float16)
    storage = torch.empty(1 << 20, dtype=torch.float16)
    seen = {}
    module = SimpleNamespace(
        allocate_output=lambda _h, _t, mode, _c: (
            seen.setdefault("mode", mode),
            (storage, [1, 2, 3]),
        )[1],
        connect_output=lambda *_args: None,
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)

    out = comm.allocate_output(x, "exchange_chunks")
    assert seen["mode"] == 2
    assert tuple(out.shape) == (1, 1, 2, 64)
    assert comm._pcie_outputs[storage.data_ptr()] == 2

    with pytest.raises(ValueError, match="exchange_chunks"):
        comm.allocate_output(x, "scatter_chunks")


def test_pcie_allocate_output_rejects_capture_before_allocation(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    x = torch.empty((1, 2, 4, 2), dtype=torch.float16)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        ulysses_mod,
        "get_ulysses_pcie_module",
        lambda: pytest.fail("allocation must not run during capture"),
    )
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        comm.allocate_output(x, "scatter_heads")


def test_pcie_init_cleanup_disposes_without_synchronize(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    calls = []
    monkeypatch.setattr(
        ulysses_mod,
        "get_ulysses_pcie_module",
        lambda: SimpleNamespace(dispose=lambda *_args: calls.append("dispose")),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *_args: pytest.fail("init cleanup has no exchange work to synchronize"),
    )
    assert comm._pcie_init_cleanup("injected init failure") == "injected init failure"
    assert calls == ["dispose"]
    assert comm._pcie is None


@pytest.mark.parametrize(
    ("transport", "gid_indices", "expected_tail"),
    [
        ("hybrid", tuple(range(2, 10)), (1, 2)),
        ("p2p", (), (0, -1)),
    ],
)
def test_pcie_init_passes_exact_local_gid_index(
    monkeypatch, transport, gid_indices, expected_tail
):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    comm.transport = transport
    comm.decision.pcie_plan.transport = transport
    if transport == "hybrid":
        comm.world_size = 8
        comm.decision.pcie_plan.numa_nodes = (0, 0, 0, 0, 1, 1, 1, 1)
        comm.decision.pcie_plan.nic_names = tuple(f"mlx5_{rank}" for rank in range(8))
        comm._gather = lambda payload: [payload] * comm.world_size
    comm.decision.pcie_plan.gid_indices = gid_indices
    init_args = []
    module = SimpleNamespace(
        init=lambda *args: (init_args.append(args), (456, [1, 2, 3]))[1],
        connect=lambda *_args: None,
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: module)

    assert comm._pcie_init_transaction() is None
    assert comm._pcie == 456
    # (use_rdma, gid_index) tail of the native init call
    assert init_args[0][-2:] == expected_tail
    assert init_args[0][4] == comm.decision.pcie_plan.nic_names[comm.rank]


# ---- constructor: backend selection before IPC/JIT ---------------------------


@requires_cuda
def test_ctor_pcie_world_size_one_is_identity_without_native(gloo_pg, monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    _patch_probe_mesh(monkeypatch, 1)

    def forbid_native(*_args, **_kwargs):
        pytest.fail("world-size-one PCIe identity must not JIT or initialize native")

    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", forbid_native)
    monkeypatch.setattr(ulysses_mod, "gen_ulysses_pcie_module", forbid_native)
    comm = UlyssesCommunicator(
        gloo_pg,
        max_bytes=1 << 17,
        dtype=torch.float16,
        backend="pcie",
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    assert comm.backend == "pcie"
    assert comm.transport == "p2p"
    assert comm._pcie_armed is False
    assert comm._pcie is None

    x = torch.randn((1, 2, 4, 2), dtype=torch.float16, device=comm.device)
    assert comm.scatter_heads(x) is x
    assert comm.gather_heads(x) is x
    out = comm.allocate_output(x, "scatter_heads")
    assert out.data_ptr() != x.data_ptr()
    assert comm.scatter_heads(x, out=out) is out
    torch.cuda.synchronize(comm.device)
    assert torch.equal(out, x)
    comm.close()
    assert comm._state == ulysses_mod._CLOSED


@requires_cuda
def test_ctor_pcie_dtype_gate(gloo_pg, monkeypatch):
    """PCIe accepts the 1/2/4-byte element types; other backends keep the
    narrow set, and unsupported dtypes raise the joint config error."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    _patch_probe_mesh(monkeypatch, 1)

    def forbid_native(*_args, **_kwargs):
        pytest.fail("world-size-one PCIe identity must not JIT or initialize native")

    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", forbid_native)
    monkeypatch.setattr(ulysses_mod, "gen_ulysses_pcie_module", forbid_native)
    device = torch.device("cuda", torch.cuda.current_device())
    for dtype in (
        torch.float32,
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        comm = UlyssesCommunicator(
            gloo_pg,
            max_bytes=(1 << 10) * dtype.itemsize,
            dtype=dtype,
            backend="pcie",
            device=device,
        )
        x = torch.zeros((1, 2, 4, 2), dtype=dtype, device=device)
        assert comm.scatter_heads(x) is x
        comm.close()
    with pytest.raises(ValueError, match="dtype must be one of"):
        UlyssesCommunicator(
            gloo_pg,
            max_bytes=1 << 11,
            dtype=torch.float64,
            backend="pcie",
            device=device,
        )
    with pytest.raises(ValueError, match="dtype must be one of"):
        UlyssesCommunicator(
            gloo_pg, max_bytes=1 << 10, dtype=torch.int8, backend="nccl", device=device
        )


@requires_cuda
def test_input_buffer_is_gated_to_routes_that_stage(gloo_pg, monkeypatch):
    """Only a multi-rank RDMA PCIe route copies input into a landing buffer.

    Everything else reads the caller's operand in place, so there is no buffer
    to hand out and nothing to save; saying so beats returning a tensor whose
    only property -- being the thing the NIC reads -- does not hold.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    shape = (1, 2, 4, 2)

    pcie_w1 = _make_w1(gloo_pg, monkeypatch, backend="pcie")
    try:
        out = torch.zeros(shape, dtype=torch.float16, device=device)
        with pytest.raises(ValueError, match="multi-rank pcie backend"):
            pcie_w1.input_buffer(out, shape)
    finally:
        pcie_w1.close()

    nccl_w1 = _make_w1(gloo_pg, monkeypatch, backend="nccl")
    try:
        out = torch.zeros(shape, dtype=torch.float16, device=device)
        with pytest.raises(ValueError, match="multi-rank pcie backend"):
            nccl_w1.input_buffer(out, shape)
    finally:
        nccl_w1.close()


@requires_cuda
def test_per_call_dtype_reprices_capacity_in_bytes(gloo_pg, monkeypatch):
    """``max_bytes`` is exactly that: a byte budget, whatever the dtype.

    A narrower per-call dtype therefore fits proportionally more elements in the
    same bytes. This is the only thing holding ``allocate_output``'s capacity
    rescale honest: native sizes its allocation from the operand's own element
    size, so if Python admitted an operand in elements while native allocated
    for it in bytes, the mismatch would surface as an ICHECK part-way through a
    collective rather than as a rejected argument here.
    """
    _patch_probe_mesh(monkeypatch, 1)
    device = torch.device("cuda", torch.cuda.current_device())
    # 12 BF16 elements = 24 bytes.
    comm = UlyssesCommunicator(
        gloo_pg, max_bytes=24, dtype=torch.bfloat16, backend="pcie", device=device
    )
    try:
        fits = torch.zeros((1, 2, 3, 4), dtype=torch.uint8, device=device)  # 24 B
        assert comm.scatter_heads(fits, dtype=torch.uint8) is fits
        over = torch.zeros((1, 1, 5, 5), dtype=torch.uint8, device=device)  # 25 B
        with pytest.raises(ValueError, match="capacity max_bytes"):
            comm.scatter_heads(over, dtype=torch.uint8)
        # 24 elements of uint8 pass; 24 of the construction dtype are 48 bytes
        # and must not. The budget did not grow, only its unit changed.
        with pytest.raises(ValueError, match="capacity max_bytes"):
            comm.scatter_heads(
                torch.zeros((1, 2, 3, 4), dtype=torch.bfloat16, device=device)
            )
    finally:
        comm.close()


@requires_cuda
def test_per_call_dtype_is_pcie_only_and_whitelisted(gloo_pg, monkeypatch):
    """A per-call dtype skips the constructor's cross-rank config check, so the
    checks that check would have made are re-run locally: the PCIe transport is
    the only one that moves opaque bytes, and its element-size whitelist still
    applies."""
    _patch_probe_mesh(monkeypatch, 1)
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.zeros((1, 2, 4, 2), dtype=torch.float16, device=device)

    nccl = UlyssesCommunicator(
        gloo_pg, max_bytes=1 << 11, dtype=torch.float16, backend="nccl", device=device
    )
    try:
        with pytest.raises(ValueError, match="only supported on the pcie"):
            nccl.scatter_heads(x, dtype=torch.float16)
        with pytest.raises(ValueError, match="only supported on the pcie"):
            nccl.gather_heads(x, dtype=torch.float16)
    finally:
        nccl.close()

    pcie = UlyssesCommunicator(
        gloo_pg, max_bytes=1 << 11, dtype=torch.float16, backend="pcie", device=device
    )
    try:
        with pytest.raises(ValueError, match="is not one of"):
            pcie.scatter_heads(x.double(), dtype=torch.float64)
        # The operand still has to be what the call says it is.
        with pytest.raises(ValueError, match="does not match the expected"):
            pcie.scatter_heads(x, dtype=torch.uint8)
    finally:
        pcie.close()


def test_joint_config_validation_uses_gathered_backends():
    """The verdict must be a pure function of the gathered configs, so every
    rank raises (or passes) together whatever backend it requested locally."""
    encode = UlyssesCommunicator._encode_config
    pcie_int8 = encode(1 << 10, torch.int8, "cuda:0", "pcie")
    UlyssesCommunicator._validate_configs_jointly(
        None, [pcie_int8, encode(1 << 10, torch.int8, "cuda:0", "pcie")]
    )
    with pytest.raises(ValueError, match="dtype must be one of"):
        UlyssesCommunicator._validate_configs_jointly(
            None, [pcie_int8, encode(1 << 10, torch.int8, "cuda:0", "nccl")]
        )


@requires_cuda
def test_ctor_nccl_backend_never_touches_ipc_jit(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    comm = UlyssesCommunicator(
        gloo_pg, max_bytes=2048, dtype=torch.float16, backend="nccl"
    )
    assert comm.backend == "nccl"
    assert comm.fallback_reason is None  # explicitly requested, not a fallback
    comm.close()


@requires_cuda
def test_ctor_auto_fallback_never_touches_ipc_jit(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    _patch_probe_mesh(monkeypatch, 1)
    comm = UlyssesCommunicator(
        gloo_pg, max_bytes=2048, dtype=torch.float16, backend="auto"
    )
    assert comm.backend == "nccl"
    assert comm.fallback_reason is not None and "world size 1" in comm.fallback_reason
    comm.close()


@requires_cuda
def test_ctor_forced_nvlink_fails_before_ipc_jit(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    _patch_probe_mesh(monkeypatch, 1)
    with pytest.raises(UlyssesBackendError, match="world size 1"):
        UlyssesCommunicator(
            gloo_pg, max_bytes=2048, dtype=torch.float16, backend="nvlink"
        )


# ---- constructor: config validation -------------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(max_bytes=0, dtype=torch.float16), "max_bytes"),
        (dict(max_bytes=-4, dtype=torch.float16), "max_bytes"),
        (dict(max_bytes="big", dtype=torch.float16), "max_bytes"),
        (dict(max_bytes=True, dtype=torch.float16), "max_bytes"),
        (dict(max_bytes=2**33, dtype=torch.float16), "int32"),
        (dict(max_bytes=2048, dtype=torch.int32), "dtype"),
        (dict(max_bytes=2048, dtype="float16"), "dtype"),
        (dict(max_bytes=2048, dtype=torch.float16, device="cpu"), "CUDA device"),
        (dict(max_bytes=2048, dtype=torch.float16, device="cuda:999"), "device count"),
        # torch.device would silently wrap these into valid ordinals
        # (cuda:256 == cuda:0); the raw string/int must be validated first
        (dict(max_bytes=2048, dtype=torch.float16, device="cuda:256"), "device count"),
        (dict(max_bytes=2048, dtype=torch.float16, device=256), "device count"),
    ],
)
def test_ctor_invalid_config(gloo_pg, monkeypatch, kwargs, match):
    _forbid_ipc_and_jit(monkeypatch)  # invalid config must fail before IPC/JIT too
    with pytest.raises(ValueError, match=match):
        UlyssesCommunicator(gloo_pg, backend="nccl", **kwargs)


@requires_cuda
def test_ctor_invalid_backend(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    with pytest.raises(ValueError, match="backend must be one of"):
        UlyssesCommunicator(
            gloo_pg, max_bytes=2048, dtype=torch.float16, backend="magic"
        )


@requires_cuda
def test_ctor_bare_cuda_device_normalized(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    comm = UlyssesCommunicator(
        gloo_pg, max_bytes=2048, dtype=torch.float16, backend="nccl", device="cuda"
    )
    # bare "cuda" must be bound to the *current indexed* device so legitimate
    # cuda:<current> tensors are accepted
    assert comm.device == torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(1, 2, 2, 4, dtype=torch.float16, device="cuda")
    assert comm.scatter_heads(x) is x  # W=1 passthrough, validation passed
    comm.close()


# ---- world_size == 1 passthrough ----------------------------------------------


@requires_cuda
def test_w1_passthrough_no_copy(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    comm = _make_w1(gloo_pg, monkeypatch)
    x = torch.randn(2, 8, 4, 16, dtype=torch.float16, device="cuda")
    assert comm.scatter_heads(x) is x
    assert comm.gather_heads(x) is x
    comm.close()


# ---- operand validation --------------------------------------------------------


@requires_cuda
def test_op_validation_negatives(gloo_pg, monkeypatch):
    comm = _make_w1(gloo_pg, monkeypatch)
    ok = torch.randn(1, 4, 4, 8, dtype=torch.float16, device="cuda")

    with pytest.raises(TypeError, match="torch.Tensor"):
        comm.scatter_heads([1, 2, 3])
    with pytest.raises(ValueError, match="4-D"):
        comm.scatter_heads(ok[0])
    with pytest.raises(ValueError, match="bound to"):
        comm.scatter_heads(ok.cpu())
    with pytest.raises(ValueError, match="dtype"):
        comm.gather_heads(ok.float())
    with pytest.raises(ValueError, match="contiguous"):
        comm.scatter_heads(ok.transpose(1, 2))
    with pytest.raises(ValueError, match="positive"):
        comm.scatter_heads(torch.empty(1, 0, 4, 8, dtype=torch.float16, device="cuda"))
    with pytest.raises(ValueError, match="capacity max_bytes"):
        comm.scatter_heads(
            torch.randn(2, 1 << 15, 4, 8, dtype=torch.float16, device="cuda")
        )
    # validation errors must mention the offending values
    try:
        comm.gather_heads(ok.float())
    except ValueError as e:
        assert "torch.float32" in str(e) and "torch.float16" in str(e)
    comm.close()


# ---- raw (advanced) API hardening ----------------------------------------------


def test_raw_init_rejects_full_nvlink_false():
    from flashinfer.comm import init_ulysses_a2a

    with pytest.raises(ValueError, match="full_nvlink=False is not supported"):
        init_ulysses_a2a([0, 0], [0, 0], 0, 2, False)


@requires_cuda
def test_raw_init_synchronizes_device(monkeypatch):
    # the raw wrapper must fence the async signal memset before returning
    from types import SimpleNamespace

    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    monkeypatch.setattr(
        ulysses_mod,
        "get_ulysses_a2a_module",
        lambda: SimpleNamespace(init_ulysses_a2a=lambda *a: 42),
    )
    calls = {"n": 0}
    orig = torch.cuda.synchronize

    def counting_sync(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(torch.cuda, "synchronize", counting_sync)
    from flashinfer.comm import init_ulysses_a2a

    assert init_ulysses_a2a([0, 0], [0, 0], 0, 2, True) == 42
    assert calls["n"] == 1, "init wrapper must synchronize the device"


@requires_cuda
def test_raw_init_sync_failure_disposes_handle(monkeypatch):
    # if the wrapper's own fence fails, the caller never receives fa and
    # cannot dispose it: the wrapper owns the handle and must release it
    from types import SimpleNamespace

    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    disposed = []
    monkeypatch.setattr(
        ulysses_mod,
        "get_ulysses_a2a_module",
        lambda: SimpleNamespace(
            init_ulysses_a2a=lambda *a: 42,
            dispose_ulysses_a2a=lambda fa: disposed.append(fa),
        ),
    )

    def failing_sync(*a, **k):
        raise RuntimeError("injected async CUDA error")

    monkeypatch.setattr(torch.cuda, "synchronize", failing_sync)
    from flashinfer.comm import init_ulysses_a2a

    with pytest.raises(RuntimeError, match="injected async CUDA error"):
        init_ulysses_a2a([0, 0], [0, 0], 0, 2, True)
    assert disposed == [42], f"handle must be disposed exactly once: {disposed}"


@requires_cuda
def test_raw_a2a_validation(monkeypatch):
    # validation fires before any module lookup: forbid JIT to prove it
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")

    def _boom(*args, **kwargs):
        raise AssertionError("JIT must not be touched by invalid raw calls")

    monkeypatch.setattr(ulysses_mod, "get_ulysses_a2a_module", _boom)
    from flashinfer.comm import ulysses_a2a

    good = torch.randn(1, 4, 4, 8, dtype=torch.float16, device="cuda")
    fa = 12345  # nonzero placeholder; validation fires before any module use
    with pytest.raises(ValueError, match="nonzero handle"):
        ulysses_a2a(0, good, good.clone(), 1, 4, 4, 8, 0)
    with pytest.raises(ValueError, match="nonzero handle"):
        ulysses_a2a("fa", good, good.clone(), 1, 4, 4, 8, 0)
    with pytest.raises(ValueError, match="must be an int"):
        ulysses_a2a(fa, good, good.clone(), 1.0, 4, 4, 8, 0)
    with pytest.raises(ValueError, match="must be an int"):
        ulysses_a2a(fa, good, good.clone(), True, 4, 4, 8, 0)
    with pytest.raises(ValueError, match="CUDA tensor"):
        ulysses_a2a(fa, good.cpu(), good, 1, 4, 4, 8, 0)
    with pytest.raises(ValueError, match="contiguous"):
        ulysses_a2a(fa, good.transpose(1, 2), good, 1, 4, 4, 8, 0)
    with pytest.raises(ValueError, match="4-D"):
        ulysses_a2a(fa, good[0], good, 1, 4, 4, 8, 0)
    with pytest.raises(ValueError, match="dtype"):
        ulysses_a2a(fa, good, good.float(), 1, 4, 4, 8, 0)
    bad_dtype = torch.zeros(1, 4, 4, 8, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="float16/bfloat16/float32"):
        ulysses_a2a(fa, bad_dtype, bad_dtype.clone(), 1, 4, 4, 8, 0)
    with pytest.raises(ValueError, match="mode"):
        ulysses_a2a(fa, good, good.clone(), 1, 4, 4, 8, 2)
    with pytest.raises(ValueError, match="positive"):
        ulysses_a2a(fa, good, good.clone(), 1, -4, 4, 8, 0)
    # same numel but wrong exact shape for the mode-checked operand
    with pytest.raises(ValueError, match="does not match"):
        ulysses_a2a(fa, good.reshape(1, 4, 8, 4), good.clone(), 1, 4, 4, 8, 0)
    with pytest.raises(ValueError, match="does not match"):
        ulysses_a2a(fa, good, good.reshape(1, 4, 8, 4), 1, 4, 4, 8, 1)
    with pytest.raises(ValueError, match="inconsistent"):
        ulysses_a2a(
            fa,
            good,
            torch.randn(2, 4, 4, 4, dtype=torch.float16, device="cuda"),
            1,
            4,
            4,
            8,
            0,
        )


# ---- lifecycle (single rank) ----------------------------------------------------


@requires_cuda
def test_lifecycle_idempotent_close_and_use_after_close(gloo_pg, monkeypatch):
    comm = _make_w1(gloo_pg, monkeypatch)
    x = torch.randn(1, 4, 4, 8, dtype=torch.float16, device="cuda")
    assert comm.scatter_heads(x) is x
    comm.close()
    comm.close()  # idempotent
    with pytest.raises(RuntimeError, match="use-after-close"):
        comm.scatter_heads(x)
    with pytest.raises(RuntimeError, match="use-after-close"):
        comm.gather_heads(x)


@requires_cuda
def test_lifecycle_context_manager(gloo_pg, monkeypatch):
    x = torch.randn(1, 4, 4, 8, dtype=torch.float16, device="cuda")
    with _make_w1(gloo_pg, monkeypatch) as comm:
        assert comm.scatter_heads(x) is x
    with pytest.raises(RuntimeError, match="use-after-close"):
        comm.scatter_heads(x)


@requires_cuda
def test_lifecycle_repeated_init_close(gloo_pg, monkeypatch):
    for _ in range(3):
        comm = _make_w1(gloo_pg, monkeypatch)
        comm.close()


# ================= multi-rank workers (spawn; all top-level) ====================

# H=24 is divisible by every supported world size (2/4/6/8)
CORRECTNESS_SHAPES = [
    (1, 8, 24, 128),  # vec-aligned fast path
    (2, 16, 24, 64),  # batch > 1
    (1, 5, 24, 3),  # unaligned row -> scalar fallback path
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


# bodies that need a specific process-group backend (None = torch default
# multi-backend group, used to prove capability detection handles "undefined")
_PG_BACKEND_OVERRIDES = {"_none_backend_body": None}


def _init_pg(rank, world_size, rendezvous, pg_backend="nccl"):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend=pg_backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        # Bounded so a rendezvous that never completes fails as this rank's
        # pg-error instead of stalling on torch's default timeout.
        timeout=datetime.timedelta(seconds=90),
    )
    return dist.group.WORLD


def _worker_main(rank, world_size, rendezvous, body_name, arg, allow_skip, q):
    """Common worker skeleton (top-level: spawn must pickle it by name):
    outcome computed, teardown finished, then a single q.put. The *topology*
    rejection class (UlyssesBackendError) becomes ('skip', ...) only for
    tests that explicitly opted in (real-probe forced-NVLink tests on
    non-NVLink machines); fake-topology and fault-injection tests must FAIL
    on it — a regressed resolver rejecting a fake full mesh is a bug, not a
    hardware limitation. Runtime init/JIT/IPC failures always FAIL, except
    that PCIe bodies pre-probe the verbs/mlx5 toolchain and re-raise a missing
    one as UlyssesBackendError so a machine without RDMA headers skips."""
    body = globals()[body_name]
    outcome = None
    try:
        group = _init_pg(
            rank, world_size, rendezvous, _PG_BACKEND_OVERRIDES.get(body_name, "nccl")
        )
        try:
            outcome = body(rank, world_size, group, arg)
        except UlyssesBackendError as e:
            if allow_skip:
                outcome = ("skip", str(e)[:500])
            else:
                outcome = ("UlyssesBackendError", str(e)[:2000])
        except Exception as e:  # noqa: BLE001
            outcome = (type(e).__name__, str(e)[:2000])
        finally:
            dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        outcome = ("pg-error", str(e)[:2000])
    q.put((rank, outcome))


def _correctness_body(rank, world_size, group, backend):
    peak_elems = max(B * S * H * D for (B, S, H, D) in CORRECTNESS_SHAPES)
    for dtype in DTYPES:
        comm = UlyssesCommunicator(
            group,
            max_bytes=peak_elems * dtype.itemsize,
            dtype=dtype,
            backend=backend,
        )
        # no fake coverage: the requested backend must actually be in use
        assert comm.backend == backend, (
            f"expected backend {backend}, got {comm.backend} ({comm.fallback_reason})"
        )
        for B, S_local, H, D in CORRECTNESS_SHAPES:
            torch.manual_seed(1234 + rank)
            x = torch.randn(B, S_local, H, D, dtype=dtype, device="cuda")
            out = comm.scatter_heads(x)
            ref = _ref_scatter_heads(x, world_size, rank, group)
            torch.cuda.synchronize()
            assert torch.equal(out, ref), (
                f"scatter_heads mismatch ws={world_size} rank={rank} "
                f"dtype={dtype} shape={(B, S_local, H, D)}"
            )
            # independent input for the gather direction (NOT the scatter
            # output): gather must hold on its own
            torch.manual_seed(4321 + rank)
            y = torch.randn(
                B,
                S_local * world_size,
                H // world_size,
                D,
                dtype=dtype,
                device="cuda",
            )
            out2 = comm.gather_heads(y)
            ref2 = _ref_gather_heads(y, world_size, rank, group)
            torch.cuda.synchronize()
            assert torch.equal(out2, ref2), (
                f"gather_heads mismatch ws={world_size} rank={rank} "
                f"dtype={dtype} shape={(B, S_local, H, D)}"
            )
        comm.close()
    return ("ok", "correct")


def _api_body(rank, world_size, group, backend):
    comm = UlyssesCommunicator(
        group, max_bytes=1 << 17, dtype=torch.bfloat16, backend=backend
    )
    x = torch.randn(1, 4, 6, 8, dtype=torch.bfloat16, device="cuda")
    out = comm.scatter_heads(x)
    ref = _ref_scatter_heads(x, world_size, rank, group)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    info = (comm.backend, comm.fallback_reason)
    comm.close()
    comm.close()  # idempotent across ranks
    try:
        comm.scatter_heads(x)
        raise AssertionError("use-after-close must raise")
    except RuntimeError as e:
        assert "use-after-close" in str(e)
    return ("ok", info)


def _stream_body(rank, world_size, group, backend):
    comm = UlyssesCommunicator(
        group, max_bytes=1 << 17, dtype=torch.float16, backend=backend
    )
    assert comm.backend == backend
    x = torch.randn(1, 8, 24, 32, dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = comm.scatter_heads(x)
        back = comm.gather_heads(out)
    stream.synchronize()
    ref = _ref_scatter_heads(x, world_size, rank, group)
    assert torch.equal(out, ref), "scatter on non-default stream mismatch"
    assert torch.equal(back, x), "round-trip on non-default stream mismatch"
    comm.close()
    return ("ok", comm.backend)


# One attention layer's two geometries: a fused-QKV scatter operand carrying
# 3 * heads, and the attention output carrying heads / world_size. The
# sequence length is a multiple of 64 so it divides every supported world
# size.
_LAYER_HEADS = 56
_LAYER_HEAD_DIM = 128
_LAYER_SEQ_LENS = (37888,)


def _pcie_layer_shape_body(rank, world_size, group, seq):
    """Both operands of one attention layer, at production scale.

    The toy shapes elsewhere in this file pin the layout algebra; this pins the
    two geometries a real layer asks for, where the operand is hundreds of
    MiB and the hybrid route's head-row pitch starts to matter.
    """
    dtype = torch.bfloat16
    device = torch.device("cuda", rank)
    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise UlyssesBackendError(
            f"PCIe transport needs {', '.join(missing)}, missing on this machine"
        )

    local_seq = seq // world_size
    packed_heads = 3 * _LAYER_HEADS
    # Both ops keep numel across the exchange, and the fused-QKV scatter is the
    # larger of the two, so it alone sizes every registration.
    capacity = local_seq * packed_heads * _LAYER_HEAD_DIM * dtype.itemsize

    qkv = torch.randn(
        (1, local_seq, packed_heads, _LAYER_HEAD_DIM), dtype=dtype, device=device
    )
    attn_out = torch.randn(
        (1, local_seq, _LAYER_HEADS, _LAYER_HEAD_DIM), dtype=dtype, device=device
    )
    reference_qkv = _ref_scatter_heads(qkv, world_size, rank, group)

    with UlyssesCommunicator(
        group, max_bytes=capacity, dtype=dtype, backend="pcie", device=device
    ) as comm:
        assert comm.backend == "pcie"
        if world_size == 8 and comm.transport != "rdma":
            raise UlyssesBackendError(
                f"eight-rank all-RDMA route unavailable: {comm.decision.reason}"
            )

        qkv_out = comm.allocate_output(qkv, "scatter_heads")
        exchanged = comm.scatter_heads(qkv, out=qkv_out)
        torch.cuda.synchronize(device)
        assert exchanged.shape == (
            1,
            seq,
            packed_heads // world_size,
            _LAYER_HEAD_DIM,
        )
        assert torch.equal(exchanged, reference_qkv), seq

        # The DiT re-views that result as [seq, H/ws, 3, D] and hands attention
        # three strided head slices. Check the view is the one it expects.
        merged = exchanged.view(seq, _LAYER_HEADS // world_size, 3, _LAYER_HEAD_DIM)
        for index in range(3):
            assert merged[:, :, index].shape == (
                seq,
                _LAYER_HEADS // world_size,
                _LAYER_HEAD_DIM,
            )

        # The output side is a separate geometry, not the inverse of the fused
        # scatter: attention returns 56/ws heads of the full sequence.
        sharded_out = comm.allocate_output(attn_out, "scatter_heads")
        sharded = comm.scatter_heads(attn_out, out=sharded_out)
        torch.cuda.synchronize(device)
        assert sharded.shape == (
            1,
            seq,
            _LAYER_HEADS // world_size,
            _LAYER_HEAD_DIM,
        )
        restored_out = comm.allocate_output(sharded, "gather_heads")
        restored = comm.gather_heads(sharded, out=restored_out)
        torch.cuda.synchronize(device)
        assert torch.equal(restored, attn_out), seq

    return ("ok", f"seq={seq}")


def _pcie_correctness_body(rank, world_size, group, test_case):
    scenario, dtype_name = test_case
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda", rank)
    # The module links libibverbs/libmlx5 even for an all-P2P route, so report
    # a missing toolchain as the skip class. A build failure with the
    # toolchain present stays a hard failure.
    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise UlyssesBackendError(
            f"PCIe transport needs {', '.join(missing)}, missing on this machine"
        )
    if scenario == "p2p":
        os.environ["FLASHINFER_ULYSSES_PCIE_ROUTE"] = "p2p"
    if scenario == "rdma":
        os.environ["FLASHINFER_ULYSSES_PCIE_ROUTE"] = "rdma"
    if scenario == "hybrid":
        os.environ["FLASHINFER_ULYSSES_PCIE_ROUTE"] = "hybrid"
    if scenario == "wrong-current-device":
        torch.cuda.set_device((rank + 1) % world_size)
    caller_device = torch.cuda.current_device()

    shape = (1, 4, world_size * 2, 16)
    q, k, v = (torch.randn(shape, dtype=dtype, device=device) for _ in range(3))
    # The varying-shape sweep below goes up to 8x the base sequence length, and
    # max_bytes is the capacity every registration is sized from.
    varying_steps = 8
    capacity = varying_steps * 4 * world_size * 2 * 16 * dtype.itemsize
    reference_q = _ref_scatter_heads(q, world_size, rank, group)
    reference_k = _ref_scatter_heads(k, world_size, rank, group)
    reference_v = _ref_scatter_heads(v, world_size, rank, group)

    with UlyssesCommunicator(
        group,
        max_bytes=capacity,
        dtype=q.dtype,
        backend="pcie",
        device=device,
    ) as comm:
        assert comm.backend == "pcie"
        if scenario == "rdma":
            # A host without a usable per-rank NIC plans all-P2P instead;
            # that is a hardware property, not a regression, so skip.
            if comm.transport != "rdma":
                raise UlyssesBackendError(
                    f"all-RDMA route unavailable: {comm.decision.reason}"
                )
        elif scenario == "hybrid":
            if comm.transport != "hybrid":
                raise UlyssesBackendError(
                    f"hybrid route unavailable: {comm.decision.reason}"
                )
        elif world_size == 2 or scenario == "p2p":
            # These routes are pure CUDA P2P by construction, so the transport
            # is fully determined by the requested configuration.
            assert comm.transport == "p2p", (comm.transport, comm.decision.reason)
        elif comm.transport != "rdma":
            # auto from four ranks up prefers all-RDMA. The toolchain probe
            # does not prove a usable NIC or RoCE v2 GID exists; planning
            # all-P2P there is correct, so skip.
            raise UlyssesBackendError(
                f"{world_size}-rank all-RDMA route unavailable: {comm.decision.reason}"
            )
        assert torch.cuda.current_device() == caller_device

        explicit_q = comm.allocate_output(q, "scatter_heads")
        explicit_k = comm.allocate_output(k, "scatter_heads")
        explicit_v = comm.allocate_output(v, "scatter_heads")
        explicit_gather = comm.allocate_output(explicit_q, "gather_heads")
        q_out = comm.scatter_heads(q, out=explicit_q)
        k_out = comm.scatter_heads(k, out=explicit_k)
        v_out = comm.scatter_heads(v, out=explicit_v)
        assert len({q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr()}) == 3
        torch.cuda.synchronize(device)
        assert torch.equal(q_out, reference_q)
        assert torch.equal(k_out, reference_k)
        assert torch.equal(v_out, reference_v)
        assert torch.cuda.current_device() == caller_device

        gather_out = comm.gather_heads(q_out, out=explicit_gather)
        torch.cuda.synchronize(device)
        assert torch.equal(gather_out, q)

        if comm.transport == "p2p":
            # Only the RDMA routes' interleaved MKeys force batch=1; the
            # peer copy loops over the batch axis, so the P2P route must
            # accept batch > 1.
            batched = torch.randn((2,) + tuple(shape[1:]), dtype=dtype, device=device)
            batched_reference = _ref_scatter_heads(batched, world_size, rank, group)
            batched_scatter = comm.allocate_output(batched, "scatter_heads")
            batched_gather = comm.allocate_output(batched_scatter, "gather_heads")
            batched_out = comm.scatter_heads(batched, out=batched_scatter)
            torch.cuda.synchronize(device)
            assert torch.equal(batched_out, batched_reference)
            assert torch.equal(
                comm.gather_heads(batched_out, out=batched_gather), batched
            )
            torch.cuda.synchronize(device)
            del batched_out
        else:
            # Not pytest.raises: its Failed derives from BaseException and
            # would skip the worker's outcome queue; keep the bare-assert
            # shape every other worker body uses.
            batch_rejected = False
            try:
                comm.scatter_heads(
                    torch.randn((2,) + tuple(shape[1:]), dtype=dtype, device=device)
                )
            except ValueError as e:
                batch_rejected = "batch=1" in str(e)
            assert batch_rejected, "hybrid route accepted batch > 1"

        assert (
            len({explicit_q.data_ptr(), explicit_k.data_ptr(), explicit_v.data_ptr()})
            == 3
        )
        assert comm.scatter_heads(q, out=explicit_q) is explicit_q
        assert comm.scatter_heads(k, out=explicit_k) is explicit_k
        assert comm.scatter_heads(v, out=explicit_v) is explicit_v
        assert comm.gather_heads(explicit_q, out=explicit_gather) is explicit_gather
        torch.cuda.synchronize(device)
        assert torch.equal(explicit_q, reference_q)
        assert torch.equal(explicit_k, reference_k)
        assert torch.equal(explicit_v, reference_v)
        assert torch.equal(explicit_gather, q)

        # Capture for real: the all-P2P route must replay correctly, the
        # hybrid one must refuse. Every rank takes the same branch, so the
        # collective call sequence stays aligned.
        graph_in = torch.empty_like(q)
        graph_inputs = [torch.randn_like(q) for _ in range(3)]
        graph_references = [
            _ref_scatter_heads(candidate, world_size, rank, group)
            for candidate in graph_inputs
        ]
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        refused = [
            ("allocate_output", lambda: comm.allocate_output(q, "scatter_heads"))
        ]
        if comm.transport in ("hybrid", "rdma"):
            refused.append(
                ("scatter_heads", lambda: comm.scatter_heads(q, out=explicit_q))
            )
        with torch.cuda.stream(capture_stream):
            for name, call in refused:
                # A fresh graph per attempt: the context manager still ends the
                # capture when the body raises, so the instance would already
                # own a graph on a second use.
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        call()
                except RuntimeError as error:
                    message = str(error)
                    assert "cuda graph" in message.lower(), message
                else:
                    raise AssertionError(f"PCIe {name} must refuse graph capture")
                del graph

            if comm.transport == "p2p":
                # The epoch advances in device memory, so a replay advances it
                # for real. Replay against changing input: a graph that
                # re-published a stale epoch would sail through both barriers
                # and hand back the previous replay's bytes.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    comm.scatter_heads(graph_in, out=explicit_q)
                for candidate, expected in zip(
                    graph_inputs, graph_references, strict=False
                ):
                    graph_in.copy_(candidate)
                    graph.replay()
                    capture_stream.synchronize()
                    assert torch.equal(explicit_q, expected), (
                        "a replayed PCIe graph must exchange the current input"
                    )
                del graph
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        torch.cuda.synchronize(device)

        # Whatever the capture did -- refused, or captured and replayed -- the
        # communicator must still work eagerly afterwards.
        assert torch.equal(comm.scatter_heads(q, out=explicit_q), reference_q), (
            "a refused capture must not disturb the next exchange"
        )
        torch.cuda.synchronize(device)

        # Pre-register each sequence geometry once, then reuse it. A serving
        # integration applies the same bounded geometry cache policy.
        varying_cases = []
        for step in range(1, varying_steps + 1):
            varying = torch.randn(
                (1, 4 * step, world_size * 2, 16), dtype=dtype, device=device
            )
            varying_out = comm.allocate_output(varying, "scatter_heads")
            varied = comm.scatter_heads(varying, out=varying_out)
            reference_varied = _ref_scatter_heads(varying, world_size, rank, group)
            torch.cuda.synchronize(device)
            assert torch.equal(varied, reference_varied), step
            varying_cases.append((varying, varying_out, reference_varied))
        registrations = len(comm._pcie_outputs)
        for varying, varying_out, reference_varied in varying_cases:
            assert torch.equal(
                comm.scatter_heads(varying, out=varying_out), reference_varied
            )
        torch.cuda.synchronize(device)
        assert len(comm._pcie_outputs) == registrations

        # Reuse one registered output while ranks arrive at deliberately skewed
        # times. Each clone consumes the previous epoch on the caller stream;
        # the next opening barrier must prevent a faster rank from overwriting
        # that output remotely before every peer has consumed it.
        snapshots = []
        expected = []
        skew_input = torch.empty_like(q)
        for step in range(12):
            skew_input.fill_(step * 32 + rank)
            result = comm.scatter_heads(skew_input, out=explicit_q)
            snapshots.append(result.clone())
            blocks = [
                torch.full_like(result[:, : shape[1]], step * 32 + peer)
                for peer in range(world_size)
            ]
            expected.append(torch.cat(blocks, dim=1))
            if rank == 0 and step % 3 == 0:
                time.sleep(0.01)
        torch.cuda.synchronize(device)
        for actual, wanted in zip(snapshots, expected, strict=True):
            assert torch.equal(actual, wanted)

        # Collectives are bound to the stream of their first call; the
        # second-stream refusal is covered by the host-only binding test.
        assert torch.cuda.current_device() == caller_device

    assert torch.cuda.current_device() == caller_device
    return ("ok", test_case)


def _divisibility_body(rank, world_size, group, _arg):
    comm = UlyssesCommunicator(
        group, max_bytes=1 << 17, dtype=torch.float16, backend="nccl"
    )
    bad_h = torch.randn(1, 4, 5, 8, dtype=torch.float16, device="cuda")  # 5 % 2 != 0
    try:
        comm.scatter_heads(bad_h)
        raise AssertionError("scatter_heads must reject H % W != 0")
    except ValueError as e:
        assert "divisible" in str(e) and "world size 2" in str(e)
    bad_s = torch.randn(1, 5, 4, 8, dtype=torch.float16, device="cuda")  # 5 % 2 != 0
    try:
        comm.gather_heads(bad_s)
        raise AssertionError("gather_heads must reject S_global % W != 0")
    except ValueError as e:
        assert "divisible" in str(e) and "world size 2" in str(e)
    # and a valid call still works after the rejected ones
    x = torch.randn(1, 4, 6, 8, dtype=torch.float16, device="cuda")
    out = comm.scatter_heads(x)
    ref = _ref_scatter_heads(x, world_size, rank, group)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    comm.close()
    return ("ok", "divisibility enforced")


def _none_backend_body(rank, world_size, group, _arg):
    # process group created with backend=None (multi-backend); the capability
    # check must detect the CUDA-bound ProcessGroupNCCL behind "undefined"
    comm = UlyssesCommunicator(
        group, max_bytes=1 << 17, dtype=torch.float16, backend="nccl"
    )
    assert comm.backend == "nccl"
    x = torch.randn(1, 4, 6, 8, dtype=torch.float16, device="cuda")
    out = comm.scatter_heads(x)
    ref = _ref_scatter_heads(x, world_size, rank, group)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    comm.close()
    return ("ok", "none-backend group accepted")


def _topology_fallback_body(rank, world_size, group, kind):
    # topology-driven fallback at a *supported* world size must not touch
    # IPC/JIT: boom every entry point, then construct through the real public
    # constructor with a broken-mesh / erroring probe.
    _patch_probe_mesh_module(
        world_size,
        break_nvlink=(kind == "missing_nvlink"),
        error_rank=(0 if kind == "probe_error" else None),
    )
    cuda_ipc_mod = importlib.import_module("flashinfer.comm.cuda_ipc")
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    vllm_ar_mod = importlib.import_module("flashinfer.comm.vllm_ar")

    def _boom(*args, **kwargs):
        raise AssertionError("IPC/JIT entry point must not be touched")

    cuda_ipc_mod.create_shared_buffer = _boom
    cuda_ipc_mod.cudart.cudaMalloc = _boom
    ulysses_mod.get_ulysses_a2a_module = _boom
    ulysses_mod.init_ulysses_a2a = _boom
    vllm_ar_mod.meta_size = _boom
    # merged module binds gen at import: patch the local binding
    ulysses_mod.gen_ulysses_a2a_module = _boom

    comm = UlyssesCommunicator(
        group, max_bytes=1 << 17, dtype=torch.float16, backend="auto"
    )
    assert comm.backend == "nccl", comm.backend
    assert comm.decision.backend == "nccl", comm.decision
    expect = "no NVLink" if kind == "missing_nvlink" else "probe failed"
    assert expect in comm.fallback_reason, comm.fallback_reason
    x = torch.randn(1, 4, 6, 8, dtype=torch.float16, device="cuda")
    out = comm.scatter_heads(x)
    ref = _ref_scatter_heads(x, world_size, rank, group)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    # gather direction verified independently on the fallback path too
    y = torch.randn(1, 4 * world_size, 3, 8, dtype=torch.float16, device="cuda")
    out2 = comm.gather_heads(y)
    ref2 = _ref_gather_heads(y, world_size, rank, group)
    torch.cuda.synchronize()
    assert torch.equal(out2, ref2)
    comm.close()
    return ("ok", comm.fallback_reason)


class _ResourceLedger:
    """Wrap the lazy cudart with counters and injected faults.

    ``faults`` maps op name ("malloc" | "free" | "open" | "close" |
    "get_handle") to how many calls should fail: an int fails the first N
    calls, True fails every call.
    """

    def __init__(self, faults=None):
        self.cuda_ipc = importlib.import_module("flashinfer.comm.cuda_ipc")
        self.counts = dict(malloc=0, free=0, open=0, close=0)
        self.faults = dict(faults or {})
        self.fired = {k: 0 for k in self.faults}
        real = self.cuda_ipc.cudart
        ledger = self

        def wrap(name, counter, key):
            orig = getattr(real, name)

            def wrapped(*a, **k):
                times = ledger.faults.get(key)
                if times is not None and (times is True or ledger.fired[key] < times):
                    ledger.fired[key] += 1
                    raise RuntimeError(f"injected {key} failure")
                out = orig(*a, **k)
                ledger.counts[counter] += 1
                return out

            setattr(real, name, wrapped)

        wrap("cudaMalloc", "malloc", "malloc")
        wrap("cudaFree", "free", "free")
        wrap("cudaIpcOpenMemHandle", "open", "open")
        wrap("cudaIpcCloseMemHandle", "close", "close")
        if "get_handle" in self.faults:

            def bad_handle(*a, **k):
                raise RuntimeError("injected get_handle failure")

            real.cudaIpcGetMemHandle = bad_handle

    def balanced(self):
        return (
            self.counts["malloc"] == self.counts["free"]
            and self.counts["open"] == self.counts["close"]
        )


def _init_fault_body(rank, world_size, group, arg):
    fault, requested = arg
    _patch_probe_mesh_module(world_size)  # decision: nvlink
    cudart_faults = fault if fault in ("malloc", "get_handle", "open") else None
    ledger = _ResourceLedger(
        faults={cudart_faults: True} if (cudart_faults and rank == 0) else None
    )
    if fault == "init" and rank == 0:
        ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")

        def bad_init(*a, **k):
            raise RuntimeError("injected init failure")

        ulysses_mod.init_ulysses_a2a = bad_init
    elif fault == "jit" and rank == 0:
        # rank-local stage-J failure: nothing may be allocated anywhere
        vllm_ar_mod = importlib.import_module("flashinfer.comm.vllm_ar")

        def bad_meta(*a, **k):
            raise RuntimeError("injected JIT/meta failure")

        vllm_ar_mod.meta_size = bad_meta
    elif fault == "sync_after_init" and rank == 0:
        # stage-C synchronize fails AFTER init returned a live handle: the
        # cleanup must dispose it and drain everything (ledger balanced)
        # The raw init wrapper itself synchronizes once internally; let that
        # first call pass so self._fa is really set, and fail the stage-C one.
        orig_sync = torch.cuda.synchronize
        state = {"calls": 0}

        def flaky_sync(*a, **k):
            state["calls"] += 1
            if state["calls"] == 2:
                raise RuntimeError("injected post-init synchronize failure")
            return orig_sync(*a, **k)

        torch.cuda.synchronize = flaky_sync

    if requested == "nvlink":
        try:
            UlyssesCommunicator(
                group, max_bytes=1 << 13, dtype=torch.float16, backend="nvlink"
            )
            raise AssertionError("forced nvlink must fail when init faults")
        except RuntimeError as e:
            assert "NVLink backend initialization failed" in str(e), str(e)
            assert "injected" in str(e), str(e)
        assert ledger.balanced(), f"leaked resources: {ledger.counts}"
        if fault == "jit":
            assert ledger.counts["malloc"] == 0, (
                f"stage-J failure must precede any allocation: {ledger.counts}"
            )
        return ("ok", ("raised", ledger.counts["malloc"], ledger.counts["free"]))

    comm = UlyssesCommunicator(
        group, max_bytes=1 << 13, dtype=torch.float16, backend="auto"
    )
    assert comm.backend == "nccl", comm.backend
    assert comm.decision.backend == "nccl", comm.decision  # effective decision
    assert comm.topology_decision.backend == "nvlink"  # what the probe said
    assert "nvlink init failed" in comm.fallback_reason, comm.fallback_reason
    assert "injected" in comm.fallback_reason, comm.fallback_reason
    assert ledger.balanced(), f"leaked resources after fallback: {ledger.counts}"
    x = torch.randn(1, 4, 6, 8, dtype=torch.float16, device="cuda")
    out = comm.scatter_heads(x)
    ref = _ref_scatter_heads(x, world_size, rank, group)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    comm.close()
    return ("ok", ("fell back", comm.backend))


def _init_cleanup_fault_body(rank, world_size, group, arg):
    # main init failure (rank0 IPC open) PLUS a cleanup fault; cleanup
    # completion must be verified group-wide, with drain-retry healing
    # transient faults and a deterministic joint failure otherwise.
    cleanup_fault, requested = arg
    _patch_probe_mesh_module(world_size)
    if cleanup_fault == "oneshot_close_rank1":
        # rank1's first import-close fails once during cleanup: drain retry
        # heals it, so the constructor reports only the main init error
        faults = {"open": True} if rank == 0 else {"close": 1}
        ledger = _ResourceLedger(faults=faults)
        try:
            UlyssesCommunicator(
                group, max_bytes=1 << 13, dtype=torch.float16, backend=requested
            )
            if requested == "nvlink":
                raise AssertionError("forced nvlink must fail")
            # auto: cleanup completed -> fallback allowed
        except RuntimeError as e:
            assert requested == "nvlink", str(e)
            assert "NVLink backend initialization failed" in str(e), str(e)
            assert "cleanup could not be completed" not in str(e), str(e)
        assert ledger.balanced(), f"leaked resources: {ledger.counts}"
        return ("ok", ledger.counts)
    else:  # persistent_free_rank1
        # rank1 cannot free its exports at all: cleanup is incomplete, so the
        # constructor must fail JOINTLY on every rank — auto must NOT fall
        # back to NCCL while NVLink resources may linger
        faults = {"open": True} if rank == 0 else {"free": True}
        _ResourceLedger(faults=faults)
        try:
            UlyssesCommunicator(
                group, max_bytes=1 << 13, dtype=torch.float16, backend=requested
            )
            raise AssertionError("constructor must fail jointly")
        except RuntimeError as e:
            assert "cleanup could not be completed" in str(e), str(e)
        return ("ok", "joint cleanup failure")


def _close_fault_body(rank, world_size, group, scenario):
    # real probe (forced nvlink -> genuine topology skip on non-NVLink boxes)
    ledger = _ResourceLedger()
    comm = UlyssesCommunicator(
        group, max_bytes=1 << 13, dtype=torch.float16, backend="nvlink"
    )
    x = torch.randn(1, 4, 6, 8, dtype=torch.float16, device="cuda")
    comm.scatter_heads(x)
    torch.cuda.synchronize()

    if scenario == "persistent_close":
        # rank0 cannot close its imports through all in-stage retries: close()
        # must raise on every rank and — crucially — NO rank may have freed
        # any export while imports were still open anywhere in the group
        if rank == 0:
            # 2 imports x 3 in-stage attempts
            ledger.faults["close"] = 6
            ledger.fired["close"] = 0
        try:
            comm.close()
            raise AssertionError("close must raise on every rank")
        except RuntimeError as e:
            assert "retry close()" in str(e), str(e)
        assert ledger.counts["free"] == 0, (
            f"exports were freed while imports were still open: {ledger.counts}"
        )
        comm.close()  # fault exhausted: converges
    elif scenario == "helper_transient":
        # the WHOLE step helper raising (not a per-pointer failure) must be
        # enveloped at the protocol layer and heal on the drain retry
        if rank == 0:
            orig_step = UlyssesCommunicator._try_close_imports
            state = {"left": 1}

            def flaky_step(self_):
                if state["left"] > 0:
                    state["left"] -= 1
                    raise RuntimeError("injected helper failure")
                return orig_step(self_)

            UlyssesCommunicator._try_close_imports = flaky_step
        comm.close()
        if rank == 0:
            UlyssesCommunicator._try_close_imports = orig_step
    elif scenario == "helper_persistent":
        if rank == 0:
            orig_step = UlyssesCommunicator._try_close_imports
            state = {"left": 3}

            def flaky_step(self_):
                if state["left"] > 0:
                    state["left"] -= 1
                    raise RuntimeError("injected helper failure")
                return orig_step(self_)

            UlyssesCommunicator._try_close_imports = flaky_step
        try:
            comm.close()
            raise AssertionError("close must raise on every rank")
        except RuntimeError as e:
            assert "retry close()" in str(e), str(e)
        comm.close()  # fault exhausted: full protocol re-run converges
        if rank == 0:
            UlyssesCommunicator._try_close_imports = orig_step
    elif scenario == "oneshot_close":
        # a single transient import-close failure is healed by the in-stage
        # drain retry: close() succeeds on the first call
        if rank == 0:
            ledger.faults["close"] = 1
            ledger.fired["close"] = 0
        comm.close()
    elif scenario == "persistent_free":
        # rank0's cudaFree fails through all in-stage retries: close() must
        # raise on EVERY rank (including rank1, which by then holds no
        # resources — this is exactly the retry-deadlock scenario), then a
        # retry after the fault clears must succeed
        if rank == 0:
            # 2 exports x 3 in-stage attempts: 6 failures exhaust the drain
            ledger.faults["free"] = 6
            ledger.fired["free"] = 0
        try:
            comm.close()
            raise AssertionError("close must raise on every rank")
        except RuntimeError as e:
            assert "retry close()" in str(e), str(e)
        comm.close()  # fault exhausted: full protocol re-run succeeds
    elif scenario == "sync_fault":
        if rank == 0:
            orig_sync = torch.cuda.synchronize
            state = {"left": 3}

            def flaky_sync(*a, **k):
                if state["left"] > 0:
                    state["left"] -= 1
                    raise RuntimeError("injected synchronize failure")
                return orig_sync(*a, **k)

            torch.cuda.synchronize = flaky_sync
        try:
            comm.close()
            raise AssertionError("close must raise on every rank")
        except RuntimeError as e:
            assert "retry close()" in str(e), str(e)
            assert rank != 0 or "synchronize" in str(e), str(e)
        comm.close()
    elif scenario == "dispose_guard_exit":
        # dispose succeeds but the device-guard __exit__ raises: the ledger
        # was already cleared inside the guard, so the drain retry must NOT
        # delete the handle a second time
        ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
        dispose_calls = {"n": 0}
        orig_dispose = ulysses_mod.dispose_ulysses_a2a

        def counting_dispose(fa):
            dispose_calls["n"] += 1
            return orig_dispose(fa)

        ulysses_mod.dispose_ulysses_a2a = counting_dispose
        if rank == 0:
            orig_device = torch.cuda.device
            state = {"raise_after_dispose": True}

            class FlakyGuard:
                def __init__(self, dev):
                    self._inner = orig_device(dev)

                def __enter__(self):
                    return self._inner.__enter__()

                def __exit__(self, *args):
                    r = self._inner.__exit__(*args)
                    if state["raise_after_dispose"] and dispose_calls["n"] > 0:
                        state["raise_after_dispose"] = False
                        raise RuntimeError("injected guard-exit failure")
                    return r

            torch.cuda.device = FlakyGuard
        # the one-shot guard-exit raise is enveloped by the protocol and the
        # drain retry converges without a second dispose
        comm.close()
        if rank == 0:
            torch.cuda.device = orig_device
        assert dispose_calls["n"] == 1, (
            f"handle must be disposed exactly once, got {dispose_calls['n']}"
        )
    elif scenario == "dispose_fault":
        # one-shot dispose failure heals within the drain retry
        if rank == 0:
            ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
            orig_dispose = ulysses_mod.dispose_ulysses_a2a
            state = {"left": 1}

            def flaky_dispose(fa):
                if state["left"] > 0:
                    state["left"] -= 1
                    raise RuntimeError("injected dispose failure")
                return orig_dispose(fa)

            ulysses_mod.dispose_ulysses_a2a = flaky_dispose
        comm.close()

    comm.close()  # CLOSED and idempotent
    assert ledger.balanced(), f"leaked resources: {ledger.counts}"
    try:
        comm.scatter_heads(x)
        raise AssertionError("use-after-close must raise")
    except RuntimeError as e:
        assert "use-after-close" in str(e)
    return ("ok", scenario)


def _lifecycle_nvlink_body(rank, world_size, group, scenario):
    # real probe: forced nvlink -> topology skip on non-NVLink machines
    mk = lambda: UlyssesCommunicator(  # noqa: E731
        group, max_bytes=1 << 13, dtype=torch.float16, backend="nvlink"
    )
    x = torch.randn(1, 4, 6, 8, dtype=torch.float16, device="cuda")

    if scenario == "ctx_exit":
        # immediate context exit right after an async collective, with NO
        # explicit synchronize by the user: close must sync internally
        with mk() as comm:
            comm.scatter_heads(x)
    elif scenario == "ctx_body_raises":

        class Boom(Exception):
            pass

        try:
            with mk() as comm:
                comm.scatter_heads(x)
                raise Boom()
        except Boom:
            pass
        try:
            comm.scatter_heads(x)
            raise AssertionError("must be closed after context exit")
        except RuntimeError as e:
            assert "use-after-close" in str(e)
    elif scenario == "repeat":
        for _ in range(2):
            comm = mk()
            out = comm.scatter_heads(x)
            ref = _ref_scatter_heads(x, world_size, rank, group)
            torch.cuda.synchronize()
            assert torch.equal(out, ref)
            comm.close()
    elif scenario == "double_close":
        comm = mk()
        comm.close()
        comm.close()
    elif scenario == "raw_shape_reject":
        # C++ global-operand exact-shape rejection with a REAL handle: the
        # Python wrapper cannot know W, so a same-numel-wrong-shape global
        # operand passes Python and must be rejected by the binding
        from flashinfer.comm import ulysses_a2a as raw_a2a

        comm = mk()
        inp = torch.randn(1, 4, 6, 8, dtype=torch.float16, device="cuda")
        # legal global operand for W=2 is (1, 8, 3, 8); this one matches
        # batch/D/numel only
        bad_out = torch.empty(1, 4, 6, 8, dtype=torch.float16, device="cuda")
        try:
            raw_a2a(comm._fa, inp, bad_out, 1, 4, 6, 8, 0)
            raise AssertionError("C++ binding must reject the global shape")
        except RuntimeError as e:
            assert "expected" in str(e), str(e)
        # a correct call still works afterwards
        out = comm.scatter_heads(inp)
        ref = _ref_scatter_heads(inp, world_size, rank, group)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
        comm.close()
    return ("ok", scenario)


def _config_fault_body(rank, world_size, group, kind):
    if kind == "invalid_one_rank":
        max_bytes = -1 if rank == 0 else 2048
        expect = "invalid UlyssesCommunicator config"
    else:  # inconsistent
        max_bytes = 2048 if rank == 0 else 4096
        expect = "inconsistent UlyssesCommunicator config"
    try:
        UlyssesCommunicator(
            group, max_bytes=max_bytes, dtype=torch.float16, backend="nccl"
        )
        raise AssertionError("constructor must reject the config")
    except ValueError as e:
        assert expect in str(e), str(e)
        return ("ok", str(e)[:200])


def _device_contract_body(rank, world_size, group, mode):
    if mode == "explicit":
        comm = UlyssesCommunicator(
            group,
            max_bytes=1 << 17,
            dtype=torch.float16,
            backend="nccl",
            device=f"cuda:{rank}",
        )
        assert comm.device == torch.device(f"cuda:{rank}")
    elif mode == "bare":
        comm = UlyssesCommunicator(
            group,
            max_bytes=1 << 17,
            dtype=torch.float16,
            backend="nccl",
            device="cuda",
        )
        assert comm.device == torch.device(f"cuda:{rank}"), comm.device
    elif mode == "unset_current":
        # the current device is deliberately NOT the explicit device= on any
        # rank: metadata collectives must still run bound to the explicit
        # device (NCCL object collectives stage on the current device without
        # the guard, landing every rank on GPU 0)
        torch.cuda.set_device(0)
        comm = UlyssesCommunicator(
            group,
            max_bytes=1 << 17,
            dtype=torch.float16,
            backend="nvlink",
            device=f"cuda:{rank}",
        )
        assert comm.device == torch.device(f"cuda:{rank}")
        assert torch.cuda.current_device() == 0, "guard must not leak set_device"
    else:  # switch: current device changed between construction and use/close
        # real probe: forced nvlink -> topology skip on non-NVLink machines
        comm = UlyssesCommunicator(
            group,
            max_bytes=1 << 17,
            dtype=torch.float16,
            backend="nvlink",
            device=f"cuda:{rank}",
        )
        torch.cuda.set_device((rank + 1) % torch.cuda.device_count())
        # metadata collectives (teardown etc.) must run bound to the
        # communicator device, not whatever device is current
        gather_devices = []
        orig_gather = dist.all_gather_object

        def recording_gather(obj_list, obj, group=None):
            gather_devices.append(torch.cuda.current_device())
            return orig_gather(obj_list, obj, group=group)

        ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
        ulysses_mod.dist.all_gather_object = recording_gather
        try:
            x = torch.randn(1, 4, 6, 8, dtype=torch.float16, device=f"cuda:{rank}")
            out = comm.scatter_heads(x)
            ref = _ref_scatter_heads(x, world_size, rank, group)
            torch.cuda.synchronize(f"cuda:{rank}")
            assert torch.equal(out, ref)
            comm.close()
        finally:
            ulysses_mod.dist.all_gather_object = orig_gather
        assert gather_devices, "close must have exchanged outcomes"
        assert all(d == rank for d in gather_devices), (
            f"metadata collectives ran on devices {set(gather_devices)}, "
            f"expected the bound device {rank}"
        )
        return ("ok", str(comm.device))
    x = torch.randn(1, 4, 6, 8, dtype=torch.float16, device=f"cuda:{rank}")
    out = comm.scatter_heads(x)
    ref = _ref_scatter_heads(x, world_size, rank, group)
    torch.cuda.synchronize(f"cuda:{rank}")
    assert torch.equal(out, ref)
    back = comm.gather_heads(out.contiguous())
    torch.cuda.synchronize(f"cuda:{rank}")
    assert torch.equal(back, x)
    comm.close()
    return ("ok", str(comm.device))


# ---- multi-rank runner ----------------------------------------------------------


def _run_multi_rank(body_name, world_size, arg, timeout=300, allow_skip=False):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs, have {torch.cuda.device_count()}")

    import os
    import sys

    # spawn starts fresh interpreters that need to re-import this module by its
    # dotted name; ensure the repo root is on sys.path so 'tests.comm' is
    # findable (bare `pytest` does not add it, unlike `python -m pytest`).
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    ctx = std_mp.get_context("spawn")
    q = ctx.Queue()
    rendezvous = _fresh_rendezvous_path()
    procs = [
        ctx.Process(
            target=_worker_main,
            args=(r, world_size, rendezvous, body_name, arg, allow_skip, q),
        )
        for r in range(world_size)
    ]
    results = {}
    try:
        for p in procs:
            p.start()
        deadline = time.time() + timeout
        while len(results) < world_size and time.time() < deadline:
            try:
                rank, outcome = q.get(timeout=1)
                results[rank] = outcome
            except queue_mod.Empty:
                pass
        for p in procs:
            p.join(timeout=max(1.0, deadline - time.time()))
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join(timeout=10)
        with contextlib.suppress(OSError):
            os.unlink(rendezvous)
    assert len(results) == world_size, (
        f"only ranks {sorted(results)} reported within {timeout}s: {results}"
    )
    exitcodes = [p.exitcode for p in procs]
    assert all(code == 0 for code in exitcodes), (
        f"workers must exit naturally with code 0, got {exitcodes} (results: {results})"
    )
    if all(kind == "skip" for kind, _ in results.values()):
        pytest.skip(f"backend unavailable on this machine: {results[0][1]}")
    for rank, (kind, payload) in results.items():
        assert kind == "ok", f"rank {rank} failed: {kind}: {payload}"
    return results


# ---- multi-rank tests ------------------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_correctness_forced_nvlink(world_size):
    # forced NVLink: skips (not silently passes via NCCL) on non-NVLink boxes
    _run_multi_rank("_correctness_body", world_size, "nvlink", allow_skip=True)


@pytest.mark.parametrize("world_size", [2, 3])
def test_correctness_forced_nccl(world_size):
    # W=3 also proves the NCCL backend covers world sizes the fused kernel
    # does not support.
    _run_multi_rank("_correctness_body", world_size, "nccl")


@pytest.mark.parametrize(
    ("world_size", "dtype_name"),
    [
        (2, "float16"),
        (2, "bfloat16"),
        (4, "bfloat16"),
        (4, "float32"),
        (8, "bfloat16"),
    ],
)
def test_correctness_forced_pcie(world_size, dtype_name):
    _run_multi_rank(
        "_pcie_correctness_body",
        world_size,
        ("explicit-output", dtype_name),
        timeout=600,
        allow_skip=True,
    )


def _pcie_dtype_body(rank, world_size, group, dtype_name):
    """Byte-exact round trip for element types the reference collectives
    cannot carry: the data is random bytes viewed as the dtype, and both the
    reference and the comparison run in the uint8 view."""
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda", rank)
    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise UlyssesBackendError(
            f"PCIe transport needs {', '.join(missing)}, missing on this machine"
        )
    esize = torch.empty((), dtype=dtype).element_size()
    torch.manual_seed(2000 + rank)
    raw = torch.randint(
        0, 256, (1, 16, world_size * 2, 16 * esize), dtype=torch.uint8, device=device
    )
    x = raw.view(dtype)
    with UlyssesCommunicator(
        max_bytes=x.nbytes, dtype=dtype, backend="pcie", device=device
    ) as comm:
        scatter_out = comm.allocate_output(x, "scatter_heads")
        gather_out = comm.allocate_output(scatter_out, "gather_heads")
        comm.scatter_heads(x, out=scatter_out)
        torch.cuda.synchronize(device)
        assert torch.equal(
            scatter_out.view(torch.uint8),
            _ref_scatter_heads(raw, world_size, rank, group),
        )
        comm.gather_heads(scatter_out, out=gather_out)
        torch.cuda.synchronize(device)
        assert torch.equal(gather_out.view(torch.uint8), raw)
    return ("ok", dtype_name)


def _pcie_p2p_fail_stop_body(rank, world_size, group, bad_kind):
    """A validation failure on the all-P2P route must fail stop, group-wide.

    The p2p barrier has no abort protocol, so a rank that fails before or
    during enqueue poisons teardown rather than risking a peer spinning
    forever. Every rank feeds the same bad operand, so no rank enqueues and
    process exit stays safe: the operand is rejected in Python, the
    communicator goes BROKEN, later collectives are refused, and close()
    demands process termination.
    """
    device = torch.device("cuda", rank)
    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise UlyssesBackendError(
            f"PCIe transport needs {', '.join(missing)}, missing on this machine"
        )
    os.environ["FLASHINFER_ULYSSES_PCIE_ROUTE"] = "p2p"
    torch.manual_seed(4500 + rank)
    good = torch.randn((1, 8, world_size * 2, 16), dtype=torch.bfloat16, device=device)
    comm = UlyssesCommunicator(
        max_bytes=good.nbytes, dtype=torch.bfloat16, backend="pcie", device=device
    )
    assert comm.transport == "p2p"
    out = comm.allocate_output(good, "scatter_heads")
    comm.scatter_heads(good, out=out)
    torch.cuda.synchronize(device)

    if bad_kind == "noncontiguous":
        bad = good.transpose(1, 2)
        expected = "contiguous"
    else:
        # Fewer rows than `good` so the capacity check cannot fire first.
        bad = torch.randn(
            (1, 4, world_size * 2 + 1, 16), dtype=torch.bfloat16, device=device
        )
        expected = "divisible"
    try:
        comm.scatter_heads(bad, out=out)
        raise AssertionError(f"pcie scatter_heads must reject a {bad_kind} operand")
    except RuntimeError as e:
        assert "BROKEN" in str(e) and expected in str(e), e

    try:
        comm.scatter_heads(good, out=out)
        raise AssertionError("a BROKEN communicator must refuse further collectives")
    except RuntimeError as e:
        assert "BROKEN" in str(e), e

    try:
        comm.close()
        raise AssertionError("close() after a p2p Python-side failure must refuse")
    except RuntimeError as e:
        assert "process termination required" in str(e), e
    # By design nothing is released here; spawn process exit reclaims the GPU.
    return ("ok", bad_kind)


def _pcie_input_landing_body(rank, world_size, group, _arg):
    """Producing the operand in the landing buffer must change nothing but the copy.

    The RDMA routes stage every input into a transport-owned landing buffer
    because the NIC will not read caller memory. ``input_buffer`` hands that
    buffer out so the producer can write it directly; the wire format, the
    barrier and the MKeys are untouched, so the result has to be bit-identical
    to the staged one -- and the staged path has to keep working on the same
    slot afterwards.
    """
    device = torch.device("cuda", rank)
    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise UlyssesBackendError(
            f"PCIe transport needs {', '.join(missing)}, missing on this machine"
        )
    # Force the all-RDMA route: auto never selects it below
    # PCIE_AUTO_RDMA_WORLD_SIZES, so the two-rank case would otherwise plan
    # p2p and skip forever on any machine.
    os.environ["FLASHINFER_ULYSSES_PCIE_ROUTE"] = "rdma"
    torch.manual_seed(4000 + rank)
    shape = (1, 16, world_size * 2, 16)
    x = torch.randn(shape, dtype=torch.bfloat16, device=device)
    y = torch.randn(shape, dtype=torch.bfloat16, device=device)
    with UlyssesCommunicator(
        max_bytes=x.nbytes, dtype=torch.bfloat16, backend="pcie", device=device
    ) as comm:
        if comm.transport not in ("hybrid", "rdma"):
            raise UlyssesBackendError(
                f"the {comm.transport} route reads the operand in place and has "
                "no landing buffer"
            )
        out = comm.allocate_output(x, "scatter_heads")
        comm.scatter_heads(x, out=out)
        torch.cuda.synchronize(device)
        staged_x = out.clone()
        assert torch.equal(staged_x, _ref_scatter_heads(x, world_size, rank, group))

        direct = comm.input_buffer(out, shape)
        assert direct.dtype == x.dtype and tuple(direct.shape) == shape
        assert direct.data_ptr() not in (x.data_ptr(), out.data_ptr())
        assert direct.data_ptr() == comm.input_buffer(out, shape).data_ptr()

        # The landing holds x from the staged run above. Write y over it, so a
        # fast path that skipped the copy *and* the caller's write -- or that
        # re-sent whatever was already there -- fails here rather than passing
        # on a coincidence.
        direct.copy_(y)
        comm.scatter_heads(direct, out=out)
        torch.cuda.synchronize(device)
        assert torch.equal(out, _ref_scatter_heads(y, world_size, rank, group))

        # The staging copy is still there for operands built anywhere else.
        comm.scatter_heads(x, out=out)
        torch.cuda.synchronize(device)
        assert torch.equal(out, staged_x)
    return ("ok", world_size)


def _pcie_exchange_chunks_body(rank, world_size, group, route):
    """Chunk exchange on the wire, against all_to_all_single at zero tolerance.

    The chunk is deliberately far past the 65535-byte interleaved-stride limit
    that bounds the head-axis transforms. That limit is on mlx5's bytes_skip,
    and this op registers no MKey at all -- each peer's bytes are contiguous on
    both ends, so the NIC reads them through the plain memory region at a
    linear offset. Reaching the same result as NCCL is what shows the linear
    addressing lands where the interleaved one would have.
    """
    device = torch.device("cuda", rank)
    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise UlyssesBackendError(
            f"PCIe transport needs {', '.join(missing)}, missing on this machine"
        )
    os.environ["FLASHINFER_ULYSSES_PCIE_ROUTE"] = route
    torch.manual_seed(6000 + rank)
    chunk = 1 << 18  # 256 KiB per peer
    packed = torch.randint(
        0, 256, (1, 1, world_size, chunk), dtype=torch.uint8, device=device
    )
    with UlyssesCommunicator(
        max_bytes=packed.nbytes, dtype=torch.uint8, backend="pcie", device=device
    ) as comm:
        if route == "rdma" and comm.transport not in ("hybrid", "rdma"):
            raise UlyssesBackendError(
                f"forced rdma fell back to {comm.transport}; this case is about "
                "the RDMA descriptor"
            )
        out = comm.allocate_output(packed, "exchange_chunks", dtype=torch.uint8)
        comm.exchange_chunks(packed, out=out, dtype=torch.uint8)
        torch.cuda.synchronize(device)

        reference = torch.empty_like(packed)
        dist.all_to_all_single(reference.view(-1), packed.reshape(-1), group=group)
        torch.cuda.synchronize(device)
        assert torch.equal(out, reference)

        # A second call on the same registration: the geometry rebind that the
        # head-axis transforms do must stay skipped rather than half-applied.
        again = torch.randint(
            0, 256, (1, 1, world_size, chunk), dtype=torch.uint8, device=device
        )
        comm.exchange_chunks(again, out=out, dtype=torch.uint8)
        dist.all_to_all_single(reference.view(-1), again.reshape(-1), group=group)
        torch.cuda.synchronize(device)
        assert torch.equal(out, reference)
    return ("ok", world_size)


@pytest.mark.parametrize("route", ["rdma", "p2p"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_correctness_forced_pcie_exchange_chunks(world_size, route):
    """A packed payload past the stride limit crosses both routes unchanged."""
    _run_multi_rank(
        "_pcie_exchange_chunks_body",
        world_size,
        route,
        timeout=600,
        allow_skip=True,
    )


@pytest.mark.parametrize("world_size", [2, 8])
def test_correctness_forced_pcie_input_landing(world_size):
    """Direct-written input is bit-identical to the staged path it replaces."""
    _run_multi_rank(
        "_pcie_input_landing_body",
        world_size,
        None,
        timeout=600,
        allow_skip=True,
    )


@pytest.mark.parametrize("bad_kind", ["noncontiguous", "indivisible_heads"])
def test_pcie_p2p_validation_fail_stop_two_ranks(bad_kind):
    """The shared validators reject bad operands on the pcie path too, and the
    rejection lands as the all-P2P fail-stop contract (BROKEN, close refused),
    not as a recoverable error."""
    _run_multi_rank("_pcie_p2p_fail_stop_body", 2, bad_kind, allow_skip=True)


def _pcie_mixed_dtype_body(rank, world_size, group, _arg):
    """One communicator, two element types, both registrations live at once.

    This is the case the transport is actually asked for: a layer builds its
    communicator for BF16 QKV, then sends a quantized payload of unrelated
    element size over the same buffers. The construction dtype stays the unit
    capacity is priced in; a per-call dtype only says what this operand is.
    Nothing here may re-enter the constructor.

    The packed record is 25 bytes on purpose -- odd, so it is not a whole
    number of BF16 elements. The old reinterpret-as-communicator-dtype route
    could not express it at all.
    """
    device = torch.device("cuda", rank)
    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise UlyssesBackendError(
            f"PCIe transport needs {', '.join(missing)}, missing on this machine"
        )
    torch.manual_seed(3000 + rank)
    heads = world_size * 2
    wide = torch.randn((1, 16, heads, 16), dtype=torch.bfloat16, device=device)
    packed = torch.randint(0, 256, (1, 16, heads, 25), dtype=torch.uint8, device=device)
    with UlyssesCommunicator(
        max_bytes=wide.nbytes, dtype=torch.bfloat16, backend="pcie", device=device
    ) as comm:
        wide_out = comm.allocate_output(wide, "scatter_heads")
        packed_out = comm.allocate_output(packed, "scatter_heads", dtype=torch.uint8)
        assert wide_out.data_ptr() != packed_out.data_ptr()

        comm.scatter_heads(wide, out=wide_out)
        comm.scatter_heads(packed, out=packed_out, dtype=torch.uint8)
        torch.cuda.synchronize(device)
        assert torch.equal(wide_out, _ref_scatter_heads(wide, world_size, rank, group))
        assert torch.equal(
            packed_out, _ref_scatter_heads(packed, world_size, rank, group)
        )

        # Interleave once more: neither registration may have been disturbed by
        # the other's geometry rebind.
        comm.scatter_heads(wide, out=wide_out)
        torch.cuda.synchronize(device)
        assert torch.equal(wide_out, _ref_scatter_heads(wide, world_size, rank, group))
    return ("ok", world_size)


@pytest.mark.parametrize("world_size", [2, 8])
def test_correctness_forced_pcie_mixed_dtypes(world_size):
    """A BF16 communicator carrying a uint8 payload alongside its BF16 one."""
    _run_multi_rank(
        "_pcie_mixed_dtype_body",
        world_size,
        None,
        timeout=600,
        allow_skip=True,
    )


@pytest.mark.parametrize(
    ("world_size", "dtype_name"),
    [
        (2, "uint8"),
        (4, "int8"),
        (4, "float8_e4m3fn"),
        (8, "float8_e5m2"),
    ],
)
def test_correctness_forced_pcie_dtypes(world_size, dtype_name):
    """1-byte element types ride the byte-exact round trip; float32 runs the
    full correctness family above."""
    _run_multi_rank(
        "_pcie_dtype_body",
        world_size,
        dtype_name,
        timeout=600,
        allow_skip=True,
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_correctness_forced_pcie_rdma(world_size):
    """All-RDMA route forced by FLASHINFER_ULYSSES_PCIE_ROUTE=rdma."""
    _run_multi_rank(
        "_pcie_correctness_body",
        world_size,
        ("rdma", "bfloat16"),
        timeout=600,
        allow_skip=True,
    )


@pytest.mark.parametrize("seq", _LAYER_SEQ_LENS)
def test_correctness_forced_pcie_layer_shapes(seq):
    """Layer-scale operands on the eight-rank preferred (all-RDMA) route."""
    _run_multi_rank(
        "_pcie_layer_shape_body",
        8,
        seq,
        timeout=900,
        allow_skip=True,
    )


def test_correctness_forced_pcie_ws8_all_p2p():
    _run_multi_rank(
        "_pcie_correctness_body",
        8,
        ("p2p", "bfloat16"),
        timeout=600,
        allow_skip=True,
    )


def test_correctness_forced_pcie_ws8_hybrid():
    _run_multi_rank(
        "_pcie_correctness_body",
        8,
        ("hybrid", "bfloat16"),
        timeout=600,
        allow_skip=True,
    )


def test_pcie_explicit_device_does_not_depend_on_current_device():
    _run_multi_rank(
        "_pcie_correctness_body",
        2,
        ("wrong-current-device", "bfloat16"),
        timeout=600,
        allow_skip=True,
    )


def test_api_auto_ws3_falls_back_to_nccl():
    # 3 is not a fused-kernel world size: auto must fall back and say why.
    results = _run_multi_rank("_api_body", 3, "auto")
    for _rank, (_kind, (backend, reason)) in results.items():
        assert backend == "nccl"
        assert reason is not None and "world size 3" in reason


def test_api_forced_nccl_reason_is_none():
    results = _run_multi_rank("_api_body", 2, "nccl")
    for _rank, (_kind, (backend, reason)) in results.items():
        assert backend == "nccl"
        assert reason is None


def test_nondefault_stream_forced_nvlink():
    _run_multi_rank("_stream_body", 2, "nvlink", allow_skip=True)


def test_nondefault_stream_forced_nccl():
    _run_multi_rank("_stream_body", 2, "nccl")


def test_op_divisibility_enforced_two_ranks():
    _run_multi_rank("_divisibility_body", 2, None)


@pytest.mark.parametrize("kind", ["missing_nvlink", "probe_error"])
def test_topology_fallback_supported_ws_never_touches_ipc_jit(kind):
    # fallback driven by *topology* (not an unsupported world size) at W=2,
    # through the real public constructor, with all IPC/JIT entries booby-trapped
    _run_multi_rank("_topology_fallback_body", 2, kind)


@pytest.mark.parametrize(
    "fault", ["malloc", "get_handle", "open", "init", "jit", "sync_after_init"]
)
@pytest.mark.parametrize("requested", ["nvlink", "auto"])
def test_init_fault_one_rank(fault, requested):
    # a single rank failing at any init stage: all ranks exit the constructor
    # together (joint raise for forced, joint NCCL fallback for auto) with
    # rank-local resource counters balanced (malloc==free, open==close)
    _run_multi_rank("_init_fault_body", 2, (fault, requested))


@pytest.mark.parametrize(
    "scenario",
    [
        "oneshot_close",
        "persistent_close",
        "persistent_free",
        "sync_fault",
        "dispose_fault",
        "dispose_guard_exit",
        "helper_transient",
        "helper_persistent",
    ],
)
def test_close_fault_scenarios(scenario):
    # oneshot faults heal inside the drain retry; persistent free / sync
    # faults raise the same error on EVERY rank (a resource-less rank still
    # runs the full stage sequence, so the retry cannot deadlock) and a
    # subsequent close() succeeds
    _run_multi_rank("_close_fault_body", 2, scenario, allow_skip=True)


@pytest.mark.parametrize("requested", ["nvlink", "auto"])
@pytest.mark.parametrize(
    "cleanup_fault", ["oneshot_close_rank1", "persistent_free_rank1"]
)
def test_init_cleanup_fault(cleanup_fault, requested):
    # main init failure + cleanup fault: transient cleanup faults drain to
    # zero (ledger balanced, forced raises the init error / auto falls back);
    # a cleanup that cannot complete is a deterministic JOINT constructor
    # failure on every rank — auto never falls back with lingering resources
    _run_multi_rank("_init_cleanup_fault_body", 2, (cleanup_fault, requested))


@pytest.mark.parametrize(
    "scenario",
    ["ctx_exit", "ctx_body_raises", "repeat", "double_close", "raw_shape_reject"],
)
def test_lifecycle_nvlink_two_ranks(scenario):
    _run_multi_rank("_lifecycle_nvlink_body", 2, scenario, allow_skip=True)


@pytest.mark.parametrize("kind", ["invalid_one_rank", "inconsistent"])
def test_config_fault_collective_safe(kind):
    _run_multi_rank("_config_fault_body", 2, kind)


@pytest.mark.parametrize("mode", ["explicit", "bare", "switch", "unset_current"])
def test_device_contract(mode):
    # per-rank cuda:rank devices are legitimate and must not be rejected by
    # the cross-rank config check; bare "cuda" binds to the current device;
    # switching the current device after construction must not break ops/close;
    # an explicit device= must be honored even when the current device was
    # never set to it (metadata collectives bound to the explicit device)
    _run_multi_rank(
        "_device_contract_body",
        2,
        mode,
        allow_skip=(mode in ("switch", "unset_current")),
    )


def test_group_backend_none_supports_cuda_alltoall():
    # init_process_group(backend=None) reports "undefined" from get_backend
    # but carries a ProcessGroupNCCL for CUDA: the capability check must
    # accept it and the NCCL backend must work on it
    _run_multi_rank("_none_backend_body", 2, None)


def _ipc_gather_count_body(rank, world_size, group, _arg):
    cuda_ipc = importlib.import_module("flashinfer.comm.cuda_ipc")
    calls = {"n": 0}
    orig = dist.all_gather_object

    def counting(obj_list, obj, group=None):
        calls["n"] += 1
        return orig(obj_list, obj, group=group)

    cuda_ipc.dist.all_gather_object = counting
    try:
        ptrs = cuda_ipc.create_shared_buffer(4096, group=group)
        n_after_create = calls["n"]
        cuda_ipc.free_shared_buffer(ptrs, group=group)
    finally:
        cuda_ipc.dist.all_gather_object = orig
    assert n_after_create == 1, (
        f"create_shared_buffer performed {n_after_create} handle all-gathers, "
        "expected exactly 1"
    )
    return ("ok", n_after_create)


def test_ipc_create_gathers_once():
    _run_multi_rank("_ipc_gather_count_body", 2, None)
