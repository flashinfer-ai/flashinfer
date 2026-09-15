"""CPU coverage of UUID discovery and the distinct AR / AG-RS decisions."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import pcie_ipc_topology as ar_topology
from flashinfer.comm.pcie_ipc_collectives import _ag_rs_topology as ag_rs_topology
from flashinfer.comm.pcie_ipc_collectives import _topology as shared_topology


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid="physical-7"),
    )
    monkeypatch.setattr(shared_topology.socket, "gethostname", lambda: "host")
    return SimpleNamespace(
        shared=shared_topology,
        ar=ar_topology,
        ag_rs=ag_rs_topology,
        torch=torch,
        dist=dist,
    )


def _nvml(monkeypatch, ancestor):
    nvml = ModuleType("pynvml")
    nvml.NVMLError = type("NVMLError", (Exception,), {})
    nvml.NVML_TOPOLOGY_HOSTBRIDGE = 30
    nvml.NVML_TOPOLOGY_SYSTEM = 50
    nvml.events = []
    nvml.nvmlInit = lambda: nvml.events.append("init")
    nvml.nvmlShutdown = lambda: nvml.events.append("shutdown")
    nvml.nvmlDeviceGetHandleByUUID = lambda uuid: uuid.decode()

    def query(source, peer):
        nvml.events.append((source, peer))
        return ancestor(source, peer)

    nvml.nvmlDeviceGetTopologyCommonAncestor = query
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    return nvml


@pytest.mark.parametrize("resolver", ["raw", "ar"])
def test_group_probe_uses_peer_uuids_with_one_visible_gpu(
    modules, monkeypatch, resolver
):
    m = modules
    uuids = ["GPU-physical-2", "GPU-physical-5", "GPU-physical-7", "GPU-physical-0"]
    identities = [
        m.shared.PcieIpcTopologyEvidence(i, "host", 0, uuid)
        for i, uuid in enumerate(uuids)
    ]
    group = object()
    monkeypatch.setattr(m.dist, "get_rank", lambda group: 2)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group: 4)
    nvml = _nvml(monkeypatch, lambda source, peer: 10 if peer == uuids[0] else 50)
    exchanges = []

    def gather(output, local, group):
        exchanges.append(local)
        if len(exchanges) == 1:
            assert nvml.events == []
            assert local.device_uuid == uuids[2]
            output[:] = identities
        else:
            output[:] = [local if i == 2 else item for i, item in enumerate(identities)]

    def forbidden_visible_scan():
        raise AssertionError("group peers must come from UUID exchange")

    monkeypatch.setattr(m.dist, "all_gather_object", gather)
    monkeypatch.setattr(m.torch.cuda, "device_count", forbidden_visible_scan)
    if resolver == "raw":
        evidence = m.shared.collect_pcie_ipc_topology(group, m.torch.device("cuda:0"))[
            2
        ]
        assert evidence.peer_ancestors == {uuids[0]: 10, uuids[1]: 50, uuids[3]: 50}
    else:
        assert m.ar.resolve_pcie_ipc_profile(group).profile == m.ar.PROFILE_SWITCHPAIR
    assert len(exchanges) == 2
    assert nvml.events == [
        "init",
        (uuids[2], uuids[0]),
        (uuids[2], uuids[1]),
        (uuids[2], uuids[3]),
        "shutdown",
    ]


def test_standalone_ar_probe_preserves_visible_device_api(modules, monkeypatch):
    m = modules
    visible = ["physical-7", "physical-2", "physical-5"]
    monkeypatch.setattr(m.torch.cuda, "device_count", lambda: len(visible))
    monkeypatch.setattr(
        m.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid=visible[index]),
    )
    nvml = _nvml(monkeypatch, lambda source, peer: 20 if peer.endswith("7") else 30)
    result = m.ar.probe_pcie_ipc_rank_topology(9, m.torch.device("cuda:1"))
    assert (result.rank, result.device_index, result.device_uuid) == (
        9,
        1,
        "GPU-physical-2",
    )
    assert result.peer_switch_local == {"GPU-physical-7": True, "GPU-physical-5": False}
    assert nvml.events[-1] == "shutdown"


def test_ag_rs_non_tp8_collects_identity_without_nvml(modules, monkeypatch):
    m = modules
    monkeypatch.setattr(m.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group: 2)
    exchanges = []
    monkeypatch.setitem(sys.modules, "pynvml", None)

    def gather(output, local, group):
        exchanges.append(local)
        output[:] = [local, m.shared.PcieIpcTopologyEvidence(1, "host", 0, "GPU-peer")]

    monkeypatch.setattr(m.dist, "all_gather_object", gather)
    result = m.ag_rs.resolve_pcie_ipc_ag_rs_topology(object(), m.torch.device("cuda:0"))
    assert len(exchanges) == 1
    assert not result.ordered_4plus4
    assert result.reason == "requires 8 ranks, got 2"
    assert exchanges[0].probe_error is None


def test_pair_error_keeps_other_raw_links_and_shuts_down_nvml(modules, monkeypatch):
    m = modules

    def ancestor(source, peer):
        if peer == "GPU-bad":
            raise nvml.NVMLError("pair unavailable")
        return 40

    nvml = _nvml(monkeypatch, ancestor)
    identity = m.shared.probe_pcie_ipc_identity(0)
    evidence = m.shared.probe_pcie_ipc_links(identity, ["GPU-bad", "GPU-good"])
    assert evidence.peer_ancestors == {"GPU-good": 40}
    assert evidence.pair_errors == {"GPU-bad": "pair unavailable"}
    assert evidence.probe_error is None
    assert evidence.hostbridge_level == 30
    assert evidence.system_level == 50
    assert identity.peer_ancestors == {}
    assert nvml.events[-1] == "shutdown"


@pytest.mark.parametrize("failure", ["identity", "nvml"])
def test_probe_failure_is_exchanged_instead_of_stranding_peers(
    modules, monkeypatch, failure
):
    m = modules
    monkeypatch.setattr(m.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group: 2)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    if failure == "identity":
        monkeypatch.setattr(
            m.torch.cuda, "get_device_properties", lambda index: SimpleNamespace()
        )
    exchanges = []

    def gather(output, local, group):
        exchanges.append(local)
        output[:] = [local, m.shared.PcieIpcTopologyEvidence(1, "host", 0, "GPU-peer")]

    monkeypatch.setattr(m.dist, "all_gather_object", gather)
    result = m.ar.resolve_pcie_ipc_profile(object())
    assert len(exchanges) == 2
    assert exchanges[1].probe_error
    assert result.profile == m.ar.PROFILE_ROOTCPLX
    assert "probe failed" in result.reason


@pytest.mark.parametrize("failure", ["init", "query"])
def test_nvml_failure_retains_identity_and_releases_initialized_session(
    modules, monkeypatch, failure
):
    m = modules
    nvml = _nvml(monkeypatch, lambda source, peer: 30)

    def fail(*args):
        raise RuntimeError("NVML unavailable")

    if failure == "init":
        nvml.nvmlInit = fail
    else:
        nvml.nvmlDeviceGetTopologyCommonAncestor = fail
    evidence = m.shared.probe_pcie_ipc_links(
        m.shared.probe_pcie_ipc_identity(3), ["GPU-peer"]
    )
    assert (evidence.rank, evidence.device_uuid) == (3, "GPU-physical-7")
    assert evidence.probe_error == "RuntimeError: NVML unavailable"
    assert ("shutdown" in nvml.events) == (failure == "query")


def _ordered_evidence(m, islands):
    return [
        m.shared.PcieIpcTopologyEvidence(
            rank=rank,
            hostname="host",
            device_uuid=f"GPU-{rank}",
            peer_ancestors={
                f"GPU-{peer}": 50 if islands[rank] != islands[peer] else 30
                for peer in range(8)
                if peer != rank
            },
            hostbridge_level=30,
            system_level=50,
        )
        for rank in range(8)
    ]


@pytest.mark.parametrize(
    "case",
    ["ordered", "interleaved", "local", "missing", "asymmetric", "error", "unknown"],
)
def test_raw_topology_preserves_strict_ag_rs_admission(modules, case):
    m = modules
    islands = (0, 1) * 4 if case == "interleaved" else (0,) * 4 + (1,) * 4
    if case == "local":
        islands = (0,) * 8
    evidence = _ordered_evidence(m, islands)
    if case == "missing":
        del evidence[0].peer_ancestors["GPU-1"]
    elif case == "asymmetric":
        evidence[0].peer_ancestors["GPU-1"] = 50
    elif case == "error":
        evidence[0].pair_errors["GPU-1"] = "unavailable"
    elif case == "unknown":
        evidence[0].system_level = None
    decision = m.ag_rs.decide_pcie_ipc_ag_rs_topology(
        [m.ag_rs._rank_links(t) for t in evidence]
    )
    assert decision.ordered_4plus4 == (case == "ordered")
    if case == "ordered":
        # The pre-refactor fingerprint for these exact ranks and SYS links.
        assert decision.placement_fingerprint == "68aa2c5eebfc2c32"


def test_ar_profile_does_not_claim_ordered_four_plus_four(modules):
    m = modules
    evidence = _ordered_evidence(m, (0, 1) * 4)
    profiles = [m.ar._profile_links(t) for t in evidence]
    assert m.ar.decide_pcie_ipc_profile(None, profiles).profile == m.ar.PROFILE_ROOTCPLX
    assert not m.ag_rs.decide_pcie_ipc_ag_rs_topology(
        [m.ag_rs._rank_links(t) for t in evidence]
    ).ordered_4plus4


def test_ar_explicit_profile_and_legacy_evidence_api(modules):
    m = modules
    evidence = [m.ar.PcieIpcRankTopology(0, hostname="host", probe_error="unknown")]
    assert (
        m.ar.decide_pcie_ipc_profile("pcieswitch", evidence).profile
        == m.ar.PROFILE_SWITCHPAIR
    )
    assert m.ar.decide_pcie_ipc_profile(None, evidence).profile == m.ar.PROFILE_ROOTCPLX
    evidence.append(m.ar.PcieIpcRankTopology(1, hostname="other-host"))
    with pytest.raises(ValueError, match="intra-node only"):
        m.ar.decide_pcie_ipc_profile("pcieswitch", evidence)
