"""Regression tests for helpers that accept an unindexed CUDA device."""

from types import SimpleNamespace

import pytest
import torch

import flashinfer.utils as utils


_CACHED_DEVICE_HELPERS = (
    utils._get_device_properties,
    utils._get_gpu_memory_bandwidth,
    utils._get_default_generator,
)


def _clear_device_caches() -> None:
    for helper in _CACHED_DEVICE_HELPERS:
        helper.cache_clear()


@pytest.fixture(autouse=True)
def clear_device_caches():
    _clear_device_caches()
    yield
    _clear_device_caches()


def test_get_compute_capability_tracks_current_device(monkeypatch):
    current_device = 0
    capabilities = {0: (8, 0), 1: (10, 3)}

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            major=capabilities[index][0], minor=capabilities[index][1]
        ),
    )

    device = torch.device("cuda")
    assert utils.get_compute_capability(device) == (8, 0)

    current_device = 1
    assert utils.get_compute_capability(device) == (10, 3)
    assert utils.get_compute_capability(torch.device("cuda:0")) == (8, 0)
    assert utils._get_device_properties.cache_info().currsize == 2


def test_cached_device_properties_track_current_device(monkeypatch):
    current_device = 0
    properties = {
        0: SimpleNamespace(
            major=8,
            minor=0,
            name="GPU 0",
            multi_processor_count=80,
            shared_memory_per_block_optin=64 * 1024,
        ),
        1: SimpleNamespace(
            major=10,
            minor=3,
            name="GPU 1",
            multi_processor_count=160,
            shared_memory_per_block_optin=128 * 1024,
        ),
    }

    def resolve_index(device=None):
        if isinstance(device, torch.device):
            return current_device if device.index is None else device.index
        return current_device if device is None else device

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: properties[resolve_index(device)],
    )
    device = torch.device("cuda")
    assert utils.get_device_sm_count(device) == 80
    assert utils.get_shared_bytes_per_block_optin(device) == 64 * 1024
    assert utils.get_device_name(device) == "GPU 0"
    assert not utils.device_support_pdl(device)

    current_device = 1
    assert utils.get_device_sm_count(device) == 160
    assert utils.get_shared_bytes_per_block_optin(device) == 128 * 1024
    assert utils.get_device_name(device) == "GPU 1"
    assert utils.device_support_pdl(device)
    assert utils._get_device_properties.cache_info().currsize == 2


def test_gpu_memory_bandwidth_tracks_current_device(monkeypatch):
    current_device = 0
    device_uuids = {0: "uuid-0", 1: "MIG-uuid-1"}
    bus_widths = {"GPU-uuid-0": 256, "MIG-uuid-1": 512}
    memory_clocks = {"GPU-uuid-0": 1000, "MIG-uuid-1": 2000}

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid=device_uuids[index]),
    )
    monkeypatch.setattr(utils.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(utils.pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(utils.pynvml, "nvmlDeviceGetHandleByUUID", lambda uuid: uuid)
    monkeypatch.setattr(
        utils.pynvml,
        "nvmlDeviceGetMemoryBusWidth",
        lambda handle: bus_widths[handle],
    )
    monkeypatch.setattr(
        utils.pynvml,
        "nvmlDeviceGetMaxClockInfo",
        lambda handle, _: memory_clocks[handle],
    )

    device = torch.device("cuda")
    assert utils.get_gpu_memory_bandwidth(device) == 64.0

    current_device = 1
    assert utils.get_gpu_memory_bandwidth(device) == 256.0
    assert utils.get_gpu_memory_bandwidth("cuda:0") == 64.0
    assert utils._get_gpu_memory_bandwidth.cache_info().currsize == 2


def test_default_generator_uses_current_device(monkeypatch):
    current_device = 0

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
    monkeypatch.setattr(
        torch.cuda, "default_generators", ("generator-0", "generator-1")
    )
    monkeypatch.setattr(torch.cuda, "init", lambda: None)

    device = torch.device("cuda")
    assert utils.get_default_generators(device) == "generator-0"

    current_device = 1
    assert utils.get_default_generators(device) == "generator-1"
    assert utils.get_default_generators(torch.device("cuda:0")) == "generator-0"
    assert utils._get_default_generator.cache_info().currsize == 2


@pytest.mark.parametrize(
    "helper",
    (
        utils.get_compute_capability,
        utils.get_gpu_memory_bandwidth,
        utils.get_shared_bytes_per_block_optin,
        utils.get_device_sm_count,
        utils.get_default_generators,
    ),
)
def test_cuda_only_device_helpers_reject_cpu(helper):
    with pytest.raises(ValueError, match="device must be a cuda device"):
        helper(torch.device("cpu"))


def test_device_support_pdl_returns_false_for_cpu():
    assert not utils.device_support_pdl(torch.device("cpu"))
