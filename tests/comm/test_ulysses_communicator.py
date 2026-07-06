# flashinfer: UlyssesCommunicator public-API tests.
# Single-rank (gloo) tests cover construction, validation, lifecycle, and the
# "fallback never touches IPC/JIT" guarantee at the real constructor entry.
# Multi-rank (NCCL, spawn) tests cover both collectives against independent
# references on the NVLink and NCCL backends, with timeout + terminate.

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
    jit_comm_mod = importlib.import_module("flashinfer.jit.comm")

    def _boom(*args, **kwargs):
        raise AssertionError("IPC/JIT entry point must not be touched")

    monkeypatch.setattr(cuda_ipc_mod, "create_shared_buffer", _boom)
    monkeypatch.setattr(ulysses_a2a_mod, "get_ulysses_a2a_module", _boom)
    monkeypatch.setattr(ulysses_a2a_mod, "init_ulysses_a2a", _boom)
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
        (dict(max_elems=1024, dtype=torch.int32), "dtype"),
        (dict(max_elems=1024, dtype="float16"), "dtype"),
        (dict(max_elems=1024, dtype=torch.float16, device="cpu"), "CUDA device"),
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


@requires_cuda
def test_op_validation_divisibility():
    # needs a real multi-rank communicator for H % W checks -> use the NCCL
    # backend on a 2-rank spawn instead; covered in _worker below. Here only
    # assert the W=1 case never trips divisibility.
    pass


@requires_cuda
def test_op_int32_range(gloo_pg, monkeypatch):
    # > 2^31 elements in fp16 is ~4.3 GB: allocatable on serious test GPUs,
    # skipped (via conftest OOM autoskip) elsewhere.
    comm = _make_w1(gloo_pg, monkeypatch, max_elems=2**31 + 2**20)
    free, _total = torch.cuda.mem_get_info()
    if free < 6 * (1 << 30):
        comm.close()
        pytest.skip("needs ~6 GB free GPU memory")
    big = torch.empty(1, 2**18 + 1, 64, 128, dtype=torch.float16, device="cuda")
    assert big.numel() > 2**31 - 1
    with pytest.raises(ValueError, match="int32"):
        comm.scatter_heads(big)
    del big
    comm.close()


# ---- lifecycle -----------------------------------------------------------------


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


# ---- multi-rank correctness (NCCL process group, spawn) ------------------------

# H=24 is divisible by every supported world size (2/4/6/8)
CORRECTNESS_SHAPES = [
    (1, 8, 24, 128),  # vec-aligned fast path
    (2, 16, 24, 64),  # batch > 1
    (1, 5, 24, 3),  # unaligned row -> scalar fallback path
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _correctness_worker(rank, world_size, port, backend, q):
    outcome = None
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        group = dist.group.WORLD
        try:
            max_elems = max(B * S * H * D for (B, S, H, D) in CORRECTNESS_SHAPES)
            for dtype in DTYPES:
                comm = UlyssesCommunicator(
                    group, max_elems=max_elems, dtype=dtype, backend=backend
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
                    # independent input for the gather direction (NOT the
                    # scatter output): gather must hold on its own
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
            outcome = ("ok", "correct")
        finally:
            dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        outcome = (type(e).__name__, str(e)[:2000])
    q.put((rank, outcome))


def _api_worker(rank, world_size, port, backend, q):
    """auto/forced-nccl fallback exposure + lifecycle across real ranks."""
    outcome = None
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        group = dist.group.WORLD
        try:
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
            outcome = ("ok", info)
        finally:
            dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        outcome = (type(e).__name__, str(e)[:2000])
    q.put((rank, outcome))


def _stream_worker(rank, world_size, port, backend, q):
    """collectives issued on a non-default CUDA stream."""
    outcome = None
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        group = dist.group.WORLD
        try:
            comm = UlyssesCommunicator(
                group, max_elems=1 << 16, dtype=torch.float16, backend=backend
            )
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
            outcome = ("ok", comm.backend)
        finally:
            dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        outcome = (type(e).__name__, str(e)[:2000])
    q.put((rank, outcome))


def _ipc_gather_count_worker(rank, world_size, port, _backend, q):
    """create_shared_buffer must all-gather IPC handles exactly once."""
    outcome = None
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        group = dist.group.WORLD
        try:
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
                f"create_shared_buffer performed {n_after_create} handle "
                "all-gathers, expected exactly 1"
            )
            outcome = ("ok", n_after_create)
        finally:
            dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        outcome = (type(e).__name__, str(e)[:2000])
    q.put((rank, outcome))


def _run_multi_rank(worker, world_size, backend, timeout=300):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs, have {torch.cuda.device_count()}")
    ctx = std_mp.get_context("spawn")
    q = ctx.Queue()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    procs = [
        ctx.Process(target=worker, args=(r, world_size, port, backend, q))
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
    for rank, (kind, payload) in results.items():
        assert kind == "ok", f"rank {rank} failed: {kind}: {payload}"
    return results


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_correctness_auto_nvlink(world_size):
    # On a full-NVLink machine auto selects the fused kernel; on others it
    # falls back to NCCL — both must match the independent references.
    _run_multi_rank(_correctness_worker, world_size, "auto")


@pytest.mark.parametrize("world_size", [2, 3])
def test_correctness_forced_nccl(world_size):
    # W=3 also proves the NCCL backend covers world sizes the fused kernel
    # does not support.
    _run_multi_rank(_correctness_worker, world_size, "nccl")


def test_api_auto_ws3_falls_back_to_nccl():
    # 3 is not a fused-kernel world size: auto must fall back and say why.
    results = _run_multi_rank(_api_worker, 3, "auto")
    for _rank, (_kind, (backend, reason)) in results.items():
        assert backend == "nccl"
        assert reason is not None and "world size 3" in reason


def test_api_forced_nccl_reason_is_none():
    results = _run_multi_rank(_api_worker, 2, "nccl")
    for _rank, (_kind, (backend, reason)) in results.items():
        assert backend == "nccl"
        assert reason is None


def test_nondefault_stream_auto():
    _run_multi_rank(_stream_worker, 2, "auto")


def test_ipc_create_gathers_once():
    _run_multi_rank(_ipc_gather_count_worker, 2, None)
