# flashinfer: UlyssesCommunicator public-API tests.
# Single-rank (gloo) tests cover construction, validation, lifecycle, and the
# "fallback never touches IPC/JIT" guarantee at the real constructor entry.
# Multi-rank (spawn) tests cover both collectives against independent
# references with the actual backend asserted, staged-init fault injection
# with resource accounting, retryable close, and the device contract — all
# with timeout + terminate + natural-exit assertions.

import importlib
import multiprocessing as std_mp
import queue as queue_mod
import socket
import time

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator
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

    def fake_probe(device, r):
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


# ---- single-rank fixtures -----------------------------------------------------


@pytest.fixture
def gloo_pg():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
    )
    yield dist.group.WORLD
    dist.destroy_process_group()


def _forbid_ipc_and_jit(monkeypatch):
    cuda_ipc_mod = importlib.import_module("flashinfer.comm.cuda_ipc")
    ulysses_a2a_mod = importlib.import_module("flashinfer.comm.ulysses_a2a")
    vllm_ar_mod = importlib.import_module("flashinfer.comm.vllm_ar")
    jit_comm_mod = importlib.import_module("flashinfer.jit.comm")

    def _boom(*args, **kwargs):
        raise AssertionError("IPC/JIT entry point must not be touched")

    monkeypatch.setattr(cuda_ipc_mod, "create_shared_buffer", _boom)
    monkeypatch.setattr(cuda_ipc_mod.cudart, "cudaMalloc", _boom, raising=False)
    monkeypatch.setattr(ulysses_a2a_mod, "get_ulysses_a2a_module", _boom)
    monkeypatch.setattr(ulysses_a2a_mod, "init_ulysses_a2a", _boom)
    monkeypatch.setattr(vllm_ar_mod, "meta_size", _boom)
    monkeypatch.setattr(jit_comm_mod, "gen_ulysses_a2a_module", _boom)


def _patch_probe_mesh(monkeypatch, world_size):
    monkeypatch.setattr(
        "flashinfer.comm.ulysses_topology.probe_ulysses_rank_topology",
        lambda device, rank: _full_mesh(world_size)[rank],
    )


def _make_w1(gloo_pg, monkeypatch, backend="auto", max_elems=1 << 20):
    _patch_probe_mesh(monkeypatch, 1)
    return UlyssesCommunicator(
        gloo_pg, max_elems=max_elems, dtype=torch.float16, backend=backend
    )


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


# ---- constructor: backend selection before IPC/JIT ---------------------------


@requires_cuda
def test_ctor_nccl_backend_never_touches_ipc_jit(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    comm = UlyssesCommunicator(
        gloo_pg, max_elems=1024, dtype=torch.float16, backend="nccl"
    )
    assert comm.backend == "nccl"
    assert comm.fallback_reason is None  # explicitly requested, not a fallback
    comm.close()


@requires_cuda
def test_ctor_auto_fallback_never_touches_ipc_jit(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    _patch_probe_mesh(monkeypatch, 1)
    comm = UlyssesCommunicator(
        gloo_pg, max_elems=1024, dtype=torch.float16, backend="auto"
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
            gloo_pg, max_elems=1024, dtype=torch.float16, backend="nvlink"
        )


# ---- constructor: config validation -------------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(max_elems=0, dtype=torch.float16), "max_elems"),
        (dict(max_elems=-4, dtype=torch.float16), "max_elems"),
        (dict(max_elems="big", dtype=torch.float16), "max_elems"),
        (dict(max_elems=True, dtype=torch.float16), "max_elems"),
        (dict(max_elems=2**31, dtype=torch.float16), "int32"),
        (dict(max_elems=1024, dtype=torch.int32), "dtype"),
        (dict(max_elems=1024, dtype="float16"), "dtype"),
        (dict(max_elems=1024, dtype=torch.float16, device="cpu"), "CUDA device"),
        (dict(max_elems=1024, dtype=torch.float16, device="cuda:999"), "device count"),
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
            gloo_pg, max_elems=1024, dtype=torch.float16, backend="magic"
        )


@requires_cuda
def test_ctor_bare_cuda_device_normalized(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    comm = UlyssesCommunicator(
        gloo_pg, max_elems=1024, dtype=torch.float16, backend="nccl", device="cuda"
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
    with pytest.raises(ValueError, match="capacity max_elems"):
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
def test_raw_a2a_validation(monkeypatch):
    # validation fires before any module lookup: forbid JIT to prove it
    ulysses_a2a_mod = importlib.import_module("flashinfer.comm.ulysses_a2a")

    def _boom(*args, **kwargs):
        raise AssertionError("JIT must not be touched by invalid raw calls")

    monkeypatch.setattr(ulysses_a2a_mod, "get_ulysses_a2a_module", _boom)
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


def _init_pg(rank, world_size, port, pg_backend="nccl"):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend=pg_backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    return dist.group.WORLD


def _worker_main(rank, world_size, port, body_name, arg, q):
    """Common worker skeleton (top-level: spawn must pickle it by name):
    outcome computed, teardown finished, then a single q.put. Only the
    *topology* rejection class (UlyssesBackendError) becomes ('skip', ...)
    for non-NVLink machines; runtime init/JIT/IPC failures must FAIL, not
    skip, or real regressions get silently swallowed."""
    body = globals()[body_name]
    outcome = None
    try:
        group = _init_pg(rank, world_size, port)
        try:
            outcome = body(rank, world_size, group, arg)
        except UlyssesBackendError as e:
            outcome = ("skip", str(e)[:500])
        except Exception as e:  # noqa: BLE001
            outcome = (type(e).__name__, str(e)[:2000])
        finally:
            dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        outcome = ("pg-error", str(e)[:2000])
    q.put((rank, outcome))


def _correctness_body(rank, world_size, group, backend):
    max_elems = max(B * S * H * D for (B, S, H, D) in CORRECTNESS_SHAPES)
    for dtype in DTYPES:
        comm = UlyssesCommunicator(
            group, max_elems=max_elems, dtype=dtype, backend=backend
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
        group, max_elems=1 << 16, dtype=torch.bfloat16, backend=backend
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
        group, max_elems=1 << 16, dtype=torch.float16, backend=backend
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


def _divisibility_body(rank, world_size, group, _arg):
    comm = UlyssesCommunicator(
        group, max_elems=1 << 16, dtype=torch.float16, backend="nccl"
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
    ulysses_a2a_mod = importlib.import_module("flashinfer.comm.ulysses_a2a")
    vllm_ar_mod = importlib.import_module("flashinfer.comm.vllm_ar")
    jit_comm_mod = importlib.import_module("flashinfer.jit.comm")

    def _boom(*args, **kwargs):
        raise AssertionError("IPC/JIT entry point must not be touched")

    cuda_ipc_mod.create_shared_buffer = _boom
    cuda_ipc_mod.cudart.cudaMalloc = _boom
    ulysses_a2a_mod.get_ulysses_a2a_module = _boom
    ulysses_a2a_mod.init_ulysses_a2a = _boom
    vllm_ar_mod.meta_size = _boom
    jit_comm_mod.gen_ulysses_a2a_module = _boom

    comm = UlyssesCommunicator(
        group, max_elems=1 << 16, dtype=torch.float16, backend="auto"
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
    ledger = _ResourceLedger(faults={fault: True} if rank == 0 else None)
    if fault == "init" and rank == 0:
        ulysses_a2a_mod = importlib.import_module("flashinfer.comm.ulysses_a2a")

        def bad_init(*a, **k):
            raise RuntimeError("injected init failure")

        ulysses_a2a_mod.init_ulysses_a2a = bad_init

    if requested == "nvlink":
        try:
            UlyssesCommunicator(
                group, max_elems=1 << 12, dtype=torch.float16, backend="nvlink"
            )
            raise AssertionError("forced nvlink must fail when init faults")
        except RuntimeError as e:
            assert "NVLink backend initialization failed" in str(e), str(e)
            assert "injected" in str(e), str(e)
        assert ledger.balanced(), f"leaked resources: {ledger.counts}"
        return ("ok", ("raised", ledger.counts["malloc"], ledger.counts["free"]))

    comm = UlyssesCommunicator(
        group, max_elems=1 << 12, dtype=torch.float16, backend="auto"
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
                group, max_elems=1 << 12, dtype=torch.float16, backend=requested
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
                group, max_elems=1 << 12, dtype=torch.float16, backend=requested
            )
            raise AssertionError("constructor must fail jointly")
        except RuntimeError as e:
            assert "cleanup could not be completed" in str(e), str(e)
        return ("ok", "joint cleanup failure")


def _close_fault_body(rank, world_size, group, scenario):
    # real probe (forced nvlink -> genuine topology skip on non-NVLink boxes)
    ledger = _ResourceLedger()
    comm = UlyssesCommunicator(
        group, max_elems=1 << 12, dtype=torch.float16, backend="nvlink"
    )
    x = torch.randn(1, 4, 6, 8, dtype=torch.float16, device="cuda")
    comm.scatter_heads(x)
    torch.cuda.synchronize()

    if scenario == "oneshot_close":
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
    elif scenario == "dispose_fault":
        # one-shot dispose failure heals within the drain retry
        if rank == 0:
            ulysses_a2a_mod = importlib.import_module("flashinfer.comm.ulysses_a2a")
            orig_dispose = ulysses_a2a_mod.dispose_ulysses_a2a
            state = {"left": 1}

            def flaky_dispose(fa):
                if state["left"] > 0:
                    state["left"] -= 1
                    raise RuntimeError("injected dispose failure")
                return orig_dispose(fa)

            ulysses_a2a_mod.dispose_ulysses_a2a = flaky_dispose
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
        group, max_elems=1 << 12, dtype=torch.float16, backend="nvlink"
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
    return ("ok", scenario)


def _config_fault_body(rank, world_size, group, kind):
    if kind == "invalid_one_rank":
        max_elems = -1 if rank == 0 else 1024
        expect = "invalid UlyssesCommunicator config"
    else:  # inconsistent
        max_elems = 1024 if rank == 0 else 2048
        expect = "inconsistent UlyssesCommunicator config"
    try:
        UlyssesCommunicator(
            group, max_elems=max_elems, dtype=torch.float16, backend="nccl"
        )
        raise AssertionError("constructor must reject the config")
    except ValueError as e:
        assert expect in str(e), str(e)
        return ("ok", str(e)[:200])


def _device_contract_body(rank, world_size, group, mode):
    if mode == "explicit":
        comm = UlyssesCommunicator(
            group,
            max_elems=1 << 16,
            dtype=torch.float16,
            backend="nccl",
            device=f"cuda:{rank}",
        )
        assert comm.device == torch.device(f"cuda:{rank}")
    elif mode == "bare":
        comm = UlyssesCommunicator(
            group,
            max_elems=1 << 16,
            dtype=torch.float16,
            backend="nccl",
            device="cuda",
        )
        assert comm.device == torch.device(f"cuda:{rank}"), comm.device
    else:  # switch: current device changed between construction and use/close
        # real probe: forced nvlink -> topology skip on non-NVLink machines
        comm = UlyssesCommunicator(
            group,
            max_elems=1 << 16,
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


def _run_multi_rank(body_name, world_size, arg, timeout=300):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs, have {torch.cuda.device_count()}")
    ctx = std_mp.get_context("spawn")
    q = ctx.Queue()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    procs = [
        ctx.Process(target=_worker_main, args=(r, world_size, port, body_name, arg, q))
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
    assert len(results) == world_size, (
        f"only ranks {sorted(results)} reported within {timeout}s: {results}"
    )
    exitcodes = [p.exitcode for p in procs]
    assert all(code == 0 for code in exitcodes), (
        f"workers must exit naturally with code 0, got {exitcodes} (results: {results})"
    )
    if all(kind == "skip" for kind, _ in results.values()):
        pytest.skip(f"NVLink path unavailable: {results[0][1]}")
    for rank, (kind, payload) in results.items():
        assert kind == "ok", f"rank {rank} failed: {kind}: {payload}"
    return results


# ---- multi-rank tests ------------------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_correctness_forced_nvlink(world_size):
    # forced NVLink: skips (not silently passes via NCCL) on non-NVLink boxes
    _run_multi_rank("_correctness_body", world_size, "nvlink")


@pytest.mark.parametrize("world_size", [2, 3])
def test_correctness_forced_nccl(world_size):
    # W=3 also proves the NCCL backend covers world sizes the fused kernel
    # does not support.
    _run_multi_rank("_correctness_body", world_size, "nccl")


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
    _run_multi_rank("_stream_body", 2, "nvlink")


def test_nondefault_stream_forced_nccl():
    _run_multi_rank("_stream_body", 2, "nccl")


def test_op_divisibility_enforced_two_ranks():
    _run_multi_rank("_divisibility_body", 2, None)


@pytest.mark.parametrize("kind", ["missing_nvlink", "probe_error"])
def test_topology_fallback_supported_ws_never_touches_ipc_jit(kind):
    # fallback driven by *topology* (not an unsupported world size) at W=2,
    # through the real public constructor, with all IPC/JIT entries booby-trapped
    _run_multi_rank("_topology_fallback_body", 2, kind)


@pytest.mark.parametrize("fault", ["malloc", "get_handle", "open", "init"])
@pytest.mark.parametrize("requested", ["nvlink", "auto"])
def test_init_fault_one_rank(fault, requested):
    # a single rank failing at any init stage: all ranks exit the constructor
    # together (joint raise for forced, joint NCCL fallback for auto) with
    # rank-local resource counters balanced (malloc==free, open==close)
    _run_multi_rank("_init_fault_body", 2, (fault, requested))


@pytest.mark.parametrize(
    "scenario", ["oneshot_close", "persistent_free", "sync_fault", "dispose_fault"]
)
def test_close_fault_scenarios(scenario):
    # oneshot faults heal inside the drain retry; persistent free / sync
    # faults raise the same error on EVERY rank (a resource-less rank still
    # runs the full stage sequence, so the retry cannot deadlock) and a
    # subsequent close() succeeds
    _run_multi_rank("_close_fault_body", 2, scenario)


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
    "scenario", ["ctx_exit", "ctx_body_raises", "repeat", "double_close"]
)
def test_lifecycle_nvlink_two_ranks(scenario):
    _run_multi_rank("_lifecycle_nvlink_body", 2, scenario)


@pytest.mark.parametrize("kind", ["invalid_one_rank", "inconsistent"])
def test_config_fault_collective_safe(kind):
    _run_multi_rank("_config_fault_body", 2, kind)


@pytest.mark.parametrize("mode", ["explicit", "bare", "switch"])
def test_device_contract(mode):
    # per-rank cuda:rank devices are legitimate and must not be rejected by
    # the cross-rank config check; bare "cuda" binds to the current device;
    # switching the current device after construction must not break ops/close
    _run_multi_rank("_device_contract_body", 2, mode)


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
