# flashinfer: Ulysses backend selection / topology decision layer tests.
# The decision function is pure (probe results injected), so most of this file
# runs without any GPU; the last test probes the real machine when it can.

import socket

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
