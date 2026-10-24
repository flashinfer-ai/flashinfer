import pytest
import torch

import flashinfer.cute_dsl.availability as availability


@pytest.fixture
def capability_queries(monkeypatch):
    """Fake two devices: index 1 is sm_120, which the DSL does not support, and
    every other device is sm_100, which it does. Yields the device argument of
    each capability query."""
    queries = []

    def fake_get_device_capability(device=None):
        queries.append(device)
        return (12, 0) if device == 1 else (10, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_get_device_capability)
    monkeypatch.setattr(
        availability,
        "is_cute_dsl_arch_supported",
        lambda major, minor, native_only=False: (major, minor) == (10, 0),
    )
    availability._device_capability.cache_clear()
    yield queries
    availability._device_capability.cache_clear()


def test_require_cute_dsl_arch_queries_each_device_index_once(capability_queries):
    for _ in range(2):
        availability.require_cute_dsl_arch(torch.device("cuda", 0))
        with pytest.raises(NotImplementedError, match="sm_120"):
            availability.require_cute_dsl_arch(torch.device("cuda", 1))

    assert capability_queries == [0, 1]


@pytest.mark.parametrize("device", [torch.device("cuda"), None, torch.device("cpu", 0)])
def test_require_cute_dsl_arch_queries_devices_without_a_cuda_index_on_every_call(
    capability_queries, device
):
    availability.require_cute_dsl_arch(device)
    availability.require_cute_dsl_arch(device)

    assert capability_queries == [device, device]
