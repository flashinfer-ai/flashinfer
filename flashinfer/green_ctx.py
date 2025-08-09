"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Tuple

import cuda.bindings.driver as driver
import cuda.bindings.runtime as runtime
import torch
from cuda.bindings.driver import CUdevice, CUdevResource

from .cuda_utils import checkCudaErrors
from .utils import get_compute_capability, round_up


def get_sm_count_constraint(major: int, minor: int) -> Tuple[int, int]:
    if major == 6:
        return (1, 1)
    elif major == 7:
        return (2, 2)
    elif major == 8:
        return (4, 2)
    elif major >= 9:
        return (8, 8)
    else:
        raise ValueError(f"Unsupported CUDA capability: {major}.{minor}")


def get_cudevice(dev: torch.device) -> CUdevice:
    try:
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    except RuntimeError:
        runtime.cudaInitDevice(dev.index, 0, 0)
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    return cu_dev


def get_device_resource(cu_dev: CUdevice) -> CUdevResource:
    return checkCudaErrors(
        driver.cuDeviceGetDevResource(
            cu_dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )


def split_resource(
    resource: CUdevResource,
    num_groups: int,
    min_count: int,
) -> Tuple[CUdevResource, CUdevResource]:
    results, _, remaining = checkCudaErrors(
        driver.cuDevSmResourceSplitByCount(
            num_groups,
            resource,
            0,  # useFlags
            min_count,
        )
    )
    return results, remaining


def split_resource_by_sm_count(
    cu_dev: CUdevice, resource: CUdevResource, sm_counts: List[int]
) -> Tuple[List[CUdevResource], CUdevResource]:
    results = []
    for sm_count in sm_counts:
        result, remaining = split_resource(resource, 1, sm_count)
        results.extend(result)
        # Refresh the remaining resource for the next iteration
        desc = checkCudaErrors(driver.cuDevResourceGenerateDesc([remaining], 1))
        green_ctx = checkCudaErrors(
            driver.cuGreenCtxCreate(
                desc, cu_dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            )
        )
        resource = checkCudaErrors(
            driver.cuGreenCtxGetDevResource(
                green_ctx, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
        )

    return results, resource


def create_green_ctx_streams(
    cu_dev: CUdevResource, resources: List[CUdevResource]
) -> List[torch.Stream]:
    streams = []
    for split in resources:
        desc = checkCudaErrors(driver.cuDevResourceGenerateDesc([split], 1))
        green_ctx = checkCudaErrors(
            driver.cuGreenCtxCreate(
                desc, cu_dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            )
        )
        stream = checkCudaErrors(
            driver.cuGreenCtxStreamCreate(
                green_ctx,
                driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
                0,  # priority
            )
        )
        streams.append(torch.cuda.get_stream_from_external(stream))

    return streams


def split_device_green_ctx(
    dev: torch.device, num_groups: int, min_count: int
) -> Tuple[List[torch.Stream], List[CUdevResource]]:
    r"""
    Split the device into multiple `green contexts <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html>`_,
    return the corresponding streams and `CUdevResource` for each group and the remaining SMs.
    Green contexts allow concurrent execution of multiple kernels on different SM partitions.

    Args:
        dev: The device to split.
        num_groups: The number of groups to split the device into.
        min_count: Minimum number of SMs required for each group, it will be adjusted to meet the
            alignment and granularity requirements.

    Returns:
        streams: The list of torch.Streams objects corresponding to the green contexts.
        resources: The list of CUdevResource objects corresponding to the green contexts.

    Example:
        >>> from flashinfer.green_ctx import split_device_green_ctx
        >>> import torch
        >>> dev = torch.device("cuda:0")
        >>> streams, resources = split_device_green_ctx(dev, 2, 16)
        >>> print([r.sm.smCount for r in resources])
        [16, 16, 100]
        >>> with torch.cuda.stream(streams[0]):
        ...     x = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        ...     y = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        ...     z = x @ y
        ...     print(z.shape)
        ...
        torch.Size([8192, 8192])

    Note:
        The length of the returned streams and resources is ``num_groups + 1``,
        where the last one is the remaining SMs.

        The following examples show how the SM count is rounded up to meet the alignment and granularity requirements:
        - Requested 7 SMs → Allocated 8 SMs (rounded up to minimum)
        - Requested 10 SMs → Allocated 16 SMs (rounded up to multiple of 8)
        - Requested 16 SMs → Allocated 16 SMs (no rounding needed)
        - Requested 17 SMs → Allocated 24 SMs (rounded up to multiple of 8)

    Raises:
        RuntimeError: when requested SM allocation exceeds device capacity:
        ``num_groups * rounded_min_count > total_device_sms``
    """
    cu_dev = get_cudevice(dev)
    resource = get_device_resource(cu_dev)
    results, remaining = split_resource(resource, num_groups, min_count)
    resources = results + [remaining]
    streams = create_green_ctx_streams(cu_dev, resources)
    return streams, resources


def split_device_green_ctx_by_sm_count(
    dev: torch.device, sm_counts: List[int]
) -> Tuple[List[torch.Stream], List[CUdevResource]]:
    r"""
    Split the device into multiple green contexts, each with a fixed number of SMs,
    return the corresponding streams and `CUdevResource` for each group and the remaining SMs.
    Green contexts allow concurrent execution of multiple kernels on different SM partitions.

    Args:
        dev: The device to split.
        sm_counts: List of SM counts for each partition. Each count will be rounded up
                   to meet the minimum and alignment requirements.

    Returns:
        streams: The list of torch.Streams objects corresponding to the green contexts.
        resources: The list of CUdevResource objects corresponding to the green contexts.

    Raises:
        RuntimeError: If the requested SM allocation exceeds device capacity:
            - When sum(rounded_sm_counts) > total_device_sms
            - When CUDA operations fail due to invalid resource types
            - When the device is not properly initialized
        ValueError: If sm_counts is empty or contains invalid values (e.g., negative values).

    Example:
        >>> from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
        >>> import torch
        >>> dev = torch.device("cuda:0")
        >>>
        >>> # Create three partitions with specific SM counts
        >>> streams, resources = split_device_green_ctx_by_sm_count(dev, [8, 16, 24])
        >>> print([r.sm.smCount for r in resources])
        [8, 16, 24, 84]  # Last value is remaining SMs
        >>>
        >>> # Execute kernels on different partitions
        >>> with torch.cuda.stream(streams[0]):
        ...     x = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        ...     y = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        ...     z = x @ y
        ...     print(f"Partition 0 result: {z.shape}")
        ...
        >>> with torch.cuda.stream(streams[1]):
        ...     # Different computation on partition 1
        ...     a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
        ...     b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
        ...     c = a @ b
        ...     print(f"Partition 1 result: {c.shape}")

    Note:
        The length of the returned streams and resources is ``len(sm_counts) + 1``,
        where the last one contains the remaining SMs that were not allocated.

        SM count alignment examples for Compute Capability 9.0+:
        - Requested 7 SMs → Allocated 8 SMs (rounded up to minimum)
        - Requested 10 SMs → Allocated 16 SMs (rounded up to multiple of 8)
        - Requested 16 SMs → Allocated 16 SMs (no rounding needed)
        - Requested 17 SMs → Allocated 24 SMs (rounded up to multiple of 8)

        The actual SM count can be obtained from the ``.sm.smCount`` field of the returned resources.

        See `CUDA Green Contexts <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html>`_
        for more details.
    """
    cu_dev = get_cudevice(dev)
    resource = get_device_resource(cu_dev)

    # Round sm counts to meet the alignment and granularity requirements
    rounded_sm_counts = []
    for sm_count in sm_counts:
        min_sm_count, sm_alignment = get_sm_count_constraint(
            *get_compute_capability(dev)
        )
        if sm_count <= 0:
            raise ValueError(f"SM count must be positive, got {sm_count}")
        rounded_sm_counts.append(round_up(max(sm_count, min_sm_count), sm_alignment))

    # Split the device into multiple green contexts
    results, remaining = split_resource_by_sm_count(cu_dev, resource, rounded_sm_counts)
    resources = results + [remaining]
    streams = create_green_ctx_streams(cu_dev, resources)
    return streams, resources
