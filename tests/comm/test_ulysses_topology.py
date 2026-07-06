# flashinfer: Ulysses backend selection / topology decision layer tests.
# The decision function is pure (probe results injected), so most of this file
# runs without any GPU; the last test probes the real machine when it can.

import importlib
import multiprocessing as std_mp
import os
import queue as queue_mod
import socket
import time

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.ulysses_topology import (
    UlyssesBackendError,
    UlyssesRankTopology,
    decide_ulysses_backend,
    probe_ulysses_rank_topology,
    resolve_ulysses_backend,
)


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


# ---- pure decision layer ---------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_full_mesh_selects_nvlink(world_size):
    d = decide_ulysses_backend("auto", _full_mesh(world_size))
    assert d.backend == "nvlink"
    assert f"{world_size} ranks" in d.reason


@pytest.mark.parametrize("world_size", [1, 3, 5, 7, 9, 16])
def test_unsupported_world_size_falls_back(world_size):
    d = decide_ulysses_backend("auto", _full_mesh(world_size))
    assert d.backend == "nccl"
    assert f"world size {world_size}" in d.reason


def test_requested_nccl_short_circuits():
    # Explicit NCCL must not even look at probe results.
    broken = _full_mesh(4)
    broken[0].probe_error = "boom"
    d = decide_ulysses_backend("nccl", broken)
    assert d.backend == "nccl"
    assert "requested" in d.reason


def test_multi_node_falls_back():
    topos = _full_mesh(4)
    topos[3].hostname = "hostB"
    d = decide_ulysses_backend("auto", topos)
    assert d.backend == "nccl"
    assert "multiple hosts" in d.reason


def test_probe_error_falls_back():
    topos = _full_mesh(4)
    topos[2].probe_error = "NVMLError: Driver Not Loaded"
    d = decide_ulysses_backend("auto", topos)
    assert d.backend == "nccl"
    assert "rank 2" in d.reason and "Driver Not Loaded" in d.reason


def test_asymmetric_p2p_falls_back():
    topos = _full_mesh(4)
    # one missing direction (3 -> 1) breaks the full mesh
    del topos[3].peer_p2p[topos[1].device_uuid]
    d = decide_ulysses_backend("auto", topos)
    assert d.backend == "nccl"
    assert "no P2P access from rank 3" in d.reason


def test_missing_nvlink_pair_falls_back():
    topos = _full_mesh(4)
    # P2P reachable (e.g. over PCIe) but the concrete pair has no NVLink
    topos[0].peer_nvlink[topos[2].device_uuid] = False
    d = decide_ulysses_backend("auto", topos)
    assert d.backend == "nccl"
    assert "no NVLink between rank 0" in d.reason


def test_pair_probe_error_reported_as_diagnostic():
    topos = _full_mesh(4)
    # NVML broke for this concrete pair: must surface the diagnostic, not
    # masquerade as a verified missing physical link
    topos[0].peer_nvlink[topos[2].device_uuid] = False
    topos[0].pair_errors[topos[2].device_uuid] = "NVML unknown error"
    d = decide_ulysses_backend("auto", topos)
    assert d.backend == "nccl"
    assert "NVLink probe failed between rank 0 and rank 2" in d.reason
    assert "NVML unknown error" in d.reason


def test_unknown_identity_falls_back():
    topos = _full_mesh(2)
    topos[1].device_uuid = ""
    d = decide_ulysses_backend("auto", topos)
    assert d.backend == "nccl"
    assert "identity unknown" in d.reason


def test_duplicate_physical_gpu_falls_back():
    topos = _full_mesh(2)
    topos[1].device_uuid = topos[0].device_uuid
    d = decide_ulysses_backend("auto", topos)
    assert d.backend == "nccl"
    assert "same physical GPU" in d.reason


def test_malformed_ranks_fall_back():
    topos = _full_mesh(2)
    topos[1].rank = 5
    d = decide_ulysses_backend("auto", topos)
    assert d.backend == "nccl"
    assert "malformed" in d.reason


def test_forced_nvlink_ok_on_full_mesh():
    d = decide_ulysses_backend("nvlink", _full_mesh(8))
    assert d.backend == "nvlink"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: setattr(t[0], "probe_error", "boom"),
        lambda t: setattr(t[1], "hostname", "hostB"),
        lambda t: t[1].peer_nvlink.update({t[0].device_uuid: False}),
        lambda t: setattr(t[0], "device_uuid", ""),
    ],
)
def test_forced_nvlink_raises_with_reason(mutate):
    topos = _full_mesh(4)
    mutate(topos)
    with pytest.raises(UlyssesBackendError, match="backend='nvlink' requested but"):
        decide_ulysses_backend("nvlink", topos)


def test_forced_nvlink_raises_on_unsupported_world_size():
    with pytest.raises(UlyssesBackendError, match="world size 3"):
        decide_ulysses_backend("nvlink", _full_mesh(3))


def test_invalid_backend_value():
    with pytest.raises(ValueError, match="backend must be one of"):
        decide_ulysses_backend("magic", _full_mesh(2))


# ---- resolve (collective wrapper) -------------------------------------------
# Single-process gloo group: no GPU or NCCL needed, proves the forced-NVLink
# failure fires before any IPC allocation or JIT compilation.


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
    import importlib

    # importlib, not `import a.b as x`: the function `ulysses_a2a` re-exported
    # from the package shadows the submodule of the same name.
    cuda_ipc_mod = importlib.import_module("flashinfer.comm.cuda_ipc")
    ulysses_a2a_mod = importlib.import_module("flashinfer.comm.ulysses_a2a")
    jit_comm_mod = importlib.import_module("flashinfer.jit.comm")

    def _boom(*args, **kwargs):
        raise AssertionError("IPC/JIT entry point must not be touched")

    monkeypatch.setattr(cuda_ipc_mod, "create_shared_buffer", _boom)
    monkeypatch.setattr(ulysses_a2a_mod, "get_ulysses_a2a_module", _boom)
    monkeypatch.setattr(jit_comm_mod, "gen_ulysses_a2a_module", _boom)


def test_resolve_auto_world_size_1_no_ipc_jit(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    monkeypatch.setattr(
        "flashinfer.comm.ulysses_topology.probe_ulysses_rank_topology",
        lambda device, rank: _full_mesh(1)[0],
    )
    d = resolve_ulysses_backend("auto", group=gloo_pg, device=torch.device("cpu"))
    assert d.backend == "nccl"
    assert "world size 1" in d.reason


def test_resolve_forced_nvlink_fails_before_ipc_jit(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    monkeypatch.setattr(
        "flashinfer.comm.ulysses_topology.probe_ulysses_rank_topology",
        lambda device, rank: _full_mesh(1)[0],
    )
    with pytest.raises(UlyssesBackendError, match="world size 1"):
        resolve_ulysses_backend("nvlink", group=gloo_pg, device=torch.device("cpu"))


def test_resolve_invalid_backend(gloo_pg):
    with pytest.raises(ValueError, match="backend must be one of"):
        resolve_ulysses_backend("magic", group=gloo_pg)


def test_exports():
    assert comm.resolve_ulysses_backend is resolve_ulysses_backend
    assert comm.UlyssesBackendError is UlyssesBackendError


# ---- collective safety (2-rank gloo, timeout + terminate) --------------------
# Every rank must reach the same outcome (same exception class or same
# decision) within the time limit no matter which single rank misbehaves; a
# hung worker means a rank left the collective sequence early.


def _resolve_case_worker(rank, world_size, port, backends, patch, marker_path, q):
    mod = importlib.import_module("flashinfer.comm.ulysses_topology")

    def mesh_probe(device, r):
        return _full_mesh(world_size)[r]

    if patch == "nvlink_pair_missing":

        def broken_probe(device, r):
            topos = _full_mesh(world_size)
            topos[1].peer_nvlink[topos[0].device_uuid] = False
            return topos[r]

        mod.probe_ulysses_rank_topology = broken_probe
    elif patch == "probe_raises_rank0":
        if rank == 0:

            def raising_probe(device, r):
                raise RuntimeError("probe exploded")

            mod.probe_ulysses_rank_topology = raising_probe
        else:
            mod.probe_ulysses_rank_topology = mesh_probe
    elif patch == "decide_raises_rank1":
        mod.probe_ulysses_rank_topology = mesh_probe
        if rank == 1:

            def raising_decide(*args, **kwargs):
                raise RuntimeError("decision exploded")

            mod.decide_ulysses_backend = raising_decide
    elif patch == "probe_marker":

        def marker_probe(device, r):
            with open(marker_path, "w") as f:
                f.write(f"probe touched by rank {r}")
            return mesh_probe(device, r)

        mod.probe_ulysses_rank_topology = marker_probe
    elif patch == "mesh":
        mod.probe_ulysses_rank_topology = mesh_probe

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        try:
            d = mod.resolve_ulysses_backend(backends[rank])
            q.put((rank, "ok", (d.backend, d.reason)))
        except mod.UlyssesBackendError as e:
            q.put((rank, "UlyssesBackendError", str(e)))
        except ValueError as e:
            q.put((rank, "ValueError", str(e)))
        except Exception as e:  # noqa: BLE001
            q.put((rank, type(e).__name__, str(e)))
        finally:
            dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — process-group setup failure
        q.put((rank, "pg-error", str(e)))


def _run_resolve_case(backends, patch=None, marker_path=None, timeout=120):
    world_size = len(backends)
    ctx = std_mp.get_context("spawn")
    q = ctx.Queue()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    procs = [
        ctx.Process(
            target=_resolve_case_worker,
            args=(r, world_size, port, backends, patch, marker_path, q),
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    deadline = time.time() + timeout
    while len(results) < world_size and time.time() < deadline:
        try:
            rank, kind, payload = q.get(timeout=1)
            results[rank] = (kind, payload)
        except queue_mod.Empty:
            pass
    hung = [p for p in procs if p.is_alive() and len(results) < world_size]
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)
    assert len(results) == world_size, (
        f"only ranks {sorted(results)} finished within {timeout}s "
        f"(hung={bool(hung)}, likely a rank left the collective sequence early); "
        f"results so far: {results}"
    )
    return results


def test_resolve_2rank_invalid_backend_one_rank():
    results = _run_resolve_case(["magic", "auto"])
    for rank in (0, 1):
        kind, payload = results[rank]
        assert kind == "ValueError", results
        assert "invalid request" in payload


def test_resolve_2rank_inconsistent_requests():
    results = _run_resolve_case(["nvlink", "auto"], patch="mesh")
    for rank in (0, 1):
        kind, payload = results[rank]
        assert kind == "ValueError", results
        assert "inconsistent backend requests" in payload


def test_resolve_2rank_forced_nvlink_unsatisfied():
    results = _run_resolve_case(["nvlink", "nvlink"], patch="nvlink_pair_missing")
    for rank in (0, 1):
        kind, payload = results[rank]
        assert kind == "UlyssesBackendError", results
        assert "no NVLink" in payload


def test_resolve_2rank_probe_raises_one_rank():
    results = _run_resolve_case(["auto", "auto"], patch="probe_raises_rank0")
    for rank in (0, 1):
        kind, payload = results[rank]
        assert kind == "ok", results
        backend, reason = payload
        assert backend == "nccl"
        assert "rank 0" in reason and "probe exploded" in reason


def test_resolve_2rank_decision_raises_one_rank():
    results = _run_resolve_case(["auto", "auto"], patch="decide_raises_rank1")
    for rank in (0, 1):
        kind, payload = results[rank]
        assert kind == "ok", results
        backend, reason = payload
        assert backend == "nccl"
        assert "decision failed on rank(s)" in reason and "decision exploded" in reason


def test_resolve_2rank_explicit_nccl_skips_probe(tmp_path):
    marker = str(tmp_path / "probe_touched")
    results = _run_resolve_case(
        ["nccl", "nccl"], patch="probe_marker", marker_path=marker
    )
    for rank in (0, 1):
        kind, payload = results[rank]
        assert kind == "ok", results
        assert payload[0] == "nccl" and "requested" in payload[1]
    assert not os.path.exists(marker), "explicit NCCL must not touch the probe"


def test_resolve_2rank_auto_full_mesh_selects_nvlink():
    results = _run_resolve_case(["auto", "auto"], patch="mesh")
    for rank in (0, 1):
        kind, payload = results[rank]
        assert kind == "ok", results
        assert payload[0] == "nvlink"


# ---- probe device handling ---------------------------------------------------


def test_probe_cpu_device_records_error():
    t = probe_ulysses_rank_topology(torch.device("cpu"), 0)
    assert t.probe_error is not None
    assert "CUDA device" in t.probe_error


def test_probe_default_device_uses_current():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 GPUs")
    prev = torch.cuda.current_device()
    try:
        torch.cuda.set_device(1)
        # device=None and bare torch.device("cuda") must mean the *current*
        # device, not GPU 0
        assert probe_ulysses_rank_topology(None, 0).device_index == 1
        assert probe_ulysses_rank_topology(torch.device("cuda"), 0).device_index == 1
    finally:
        torch.cuda.set_device(prev)


# ---- real-machine probe ------------------------------------------------------


def test_probe_real_topology_two_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 GPUs")
    topos = []
    for r, dev in enumerate([0, 1]):
        t = probe_ulysses_rank_topology(torch.device("cuda", dev), r)
        assert t.probe_error is None, t.probe_error
        assert t.device_uuid.startswith("GPU-")
        assert t.pci_bus_id
        topos.append(t)
    d = decide_ulysses_backend("auto", topos)
    # Expectation depends on the actual machine: NVLink only if both concrete
    # pair directions are P2P-reachable *and* NVML reports pair-wise NVLink.
    pair_ok = all(
        t.peer_p2p.get(o.device_uuid) and t.peer_nvlink.get(o.device_uuid)
        for t, o in [(topos[0], topos[1]), (topos[1], topos[0])]
    )
    assert d.backend == ("nvlink" if pair_ok else "nccl"), d.reason
