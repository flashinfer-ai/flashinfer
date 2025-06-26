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


def get_cudevice(dev: torch.device) -> CUdevice:
    try:
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    except RuntimeError as e:
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

    Raises:
        RuntimeError: when requested SM allocation exceeds device capacity:
        ``num_groups * round_up(min_count, 8) > num_sm``
    """
    cu_dev = get_cudevice(dev)
    resource = get_device_resource(cu_dev)
    results, remaining = split_resource(resource, num_groups, min_count)
    resources = results + [remaining]
    streams = create_green_ctx_streams(cu_dev, resources)
    return streams, resources
