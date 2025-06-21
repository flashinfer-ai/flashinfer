import pytest
import torch

import flashinfer.green_ctx as green_ctx


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("num_groups", [1, 2, 3])
@pytest.mark.parametrize("min_count", [16, 32])
def test_green_ctx_creation(
    device: str,
    num_groups: int,
    min_count: int,
):
    streams, resources = green_ctx.split_device_green_ctx(
        torch.device(device), num_groups, min_count
    )

    assert len(resources) == num_groups + 1
    for resource in resources[:-1]:
        sm_count = resource.sm.smCount
        assert sm_count >= min_count


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("num_groups", [1, 2, 3])
@pytest.mark.parametrize("min_count", [16, 32])
def test_green_ctx_kernel_execution(
    device: str,
    num_groups: int,
    min_count: int,
):
    streams, resources = green_ctx.split_device_green_ctx(
        torch.device(device), num_groups, min_count
    )
    num_partitions = num_groups + 1
    assert len(streams) == num_partitions
    assert len(resources) == num_partitions

    for stream in streams:
        with torch.cuda.stream(stream):
            x = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
            y = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
            z = x @ y
            print(z.shape)
