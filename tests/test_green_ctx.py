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


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize(
    "sm_counts",
    [
        [16, 16, 16],
        [8, 16, 24],
        [32],
        [8, 8, 8, 8],
    ],
)
def test_split_device_green_ctx_by_sm_count_creation(
    device: str,
    sm_counts: list,
):
    streams, resources = green_ctx.split_device_green_ctx_by_sm_count(
        torch.device(device), sm_counts
    )
    num_partitions = len(sm_counts) + 1
    assert len(resources) == num_partitions
    assert len(streams) == num_partitions

    # Check that each partition has the expected SM count
    for i, expected_sm_count in enumerate(sm_counts):
        actual_sm_count = resources[i].sm.smCount
        assert actual_sm_count >= expected_sm_count


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize(
    "sm_counts",
    [
        [16, 16, 16],
        [8, 16, 24],
        [32],
    ],
)
def test_split_device_green_ctx_by_sm_count_kernel_execution(
    device: str,
    sm_counts: list,
):
    streams, resources = green_ctx.split_device_green_ctx_by_sm_count(
        torch.device(device), sm_counts
    )
    num_partitions = len(sm_counts) + 1
    assert len(streams) == num_partitions
    assert len(resources) == num_partitions

    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
            y = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
            z = x @ y
            print(f"Partition {i}: {z.shape}")


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize(
    "sm_counts",
    [
        [1, 2, 3, 4],  # Should be aligned to minimum requirements
        [7, 8, 9, 10],  # Should be aligned to 8 for compute capability 9+
        [15, 16, 17, 18],  # Should be aligned to 8 for compute capability 9+
    ],
)
def test_split_device_green_ctx_by_sm_count_alignment(
    device: str,
    sm_counts: list,
):
    _, resources = green_ctx.split_device_green_ctx_by_sm_count(
        torch.device(device), sm_counts
    )

    for resource in resources[:-1]:  # Exclude remaining SMs
        sm_count = resource.sm.smCount
        assert sm_count > 0

        min_sm_count, sm_alignment = green_ctx.get_sm_count_constraint(
            *green_ctx.get_compute_capability(torch.device(device))
        )
        assert sm_count >= min_sm_count
        assert sm_count % sm_alignment == 0
