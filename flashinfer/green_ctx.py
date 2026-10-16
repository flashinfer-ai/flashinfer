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

import math
import weakref
from typing import List, Tuple

import torch

try:
    import cuda.bindings.driver as driver
    import cuda.bindings.runtime as runtime
    from cuda.bindings.driver import CUdevice, CUdevResource
except ImportError as e:
    raise ImportError(
        "Could not import the 'cuda' module. "
        "Please install cuda-python that matches your CUDA version."
    ) from e

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
    Split the device into multiple `green contexts <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html>`_
    and return the corresponding streams and `CUdevResource` for each group and the remaining SMs.
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
    try:
        cu_dev = get_cudevice(dev)
        resource = get_device_resource(cu_dev)
        results, remaining = split_resource(resource, num_groups, min_count)
        resources = results + [remaining]
        streams = create_green_ctx_streams(cu_dev, resources)
        return streams, resources
    except RuntimeError as e:
        if (
            "CUDA error code=914" in str(e)
            or "CUDA_ERROR_INVALID_RESOURCE_TYPE" in str(e)
            or "CUDA error code=915" in str(e)
            or "CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION" in str(e)
        ):
            raise RuntimeError(
                f"{e}\n"
                f"Failed to split device into {num_groups} groups with min_count={min_count}. "
                f"This is likely due to insufficient number of SMs available on the device. "
                f"Please reduce the number of groups or the minimum SM count per group."
            ) from e
        raise


def split_device_green_ctx_by_sm_count(
    dev: torch.device, sm_counts: List[int]
) -> Tuple[List[torch.Stream], List[CUdevResource]]:
    r"""
    Split the device into multiple green contexts, each with a fixed number of SMs,
    and return the corresponding streams and `CUdevResource` for each group and the remaining SMs.
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
    try:
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
            rounded_sm_counts.append(
                round_up(max(sm_count, min_sm_count), sm_alignment)
            )

        # Split the device into multiple green contexts
        results, remaining = split_resource_by_sm_count(
            cu_dev, resource, rounded_sm_counts
        )
        resources = results + [remaining]
        streams = create_green_ctx_streams(cu_dev, resources)
        return streams, resources
    except RuntimeError as e:
        if (
            "CUDA error code=914" in str(e)
            or "CUDA_ERROR_INVALID_RESOURCE_TYPE" in str(e)
            or "CUDA error code=915" in str(e)
            or "CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION" in str(e)
        ):
            raise RuntimeError(
                f"{e}\n"
                f"Failed to split device with SM counts {sm_counts} (rounded to {rounded_sm_counts}). "
                f"This is likely due to insufficient number of SMs available on the device. "
                f"Please reduce the requested SM counts or use fewer partitions."
            ) from e
        raise


# ---------------------------------------------------------------------------
# uGPU localization: locality domains (dies) of multi-die GPUs such as Rubin
# ---------------------------------------------------------------------------


def get_locality_domain_count(dev: torch.device) -> int:
    r"""Number of locality domains (uGPUs, one per die) of ``dev``.

    Returns 1 when the device or the installed driver / cuda-python does not expose
    locality domains.
    """
    attr = getattr(
        driver.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT", None
    )
    if attr is None:
        return 1
    try:
        return max(
            1,
            int(checkCudaErrors(driver.cuDeviceGetAttribute(attr, get_cudevice(dev)))),
        )
    except RuntimeError:
        return 1


def split_device_green_ctx_by_locality_domain(
    dev: torch.device, backfill: bool = True
) -> Tuple[List[torch.Stream], List[CUdevResource]]:
    r"""
    Create one green context per locality domain (uGPU) of ``dev`` and return its stream
    and ``CUdevResource``, in domain order. A kernel launched on stream ``i`` runs only on
    domain ``i``'s SMs, so together with memory placed in that domain
    (:func:`empty_by_locality_domain`) its DRAM traffic stays on the die.

    Args:
        dev: The device to split.
        backfill: If ``True`` (default), each group gets ``num_SMs // num_domains`` SMs and
            the driver completes a domain's group with SMs that belong to no domain (on
            Rubin VR200: 100 native SMs + 6 of the 12 remaining ones per die, so all 212
            SMs work). If ``False``, each group holds its domain's native SMs only.

    Returns:
        streams: One ``torch.Stream`` per locality domain.
        resources: The matching ``CUdevResource`` objects (``.sm.smCount`` is the SM count).

    Raises:
        RuntimeError: if the device has fewer than two locality domains.
    """
    num_domains = get_locality_domain_count(dev)
    if num_domains < 2:
        raise RuntimeError(
            f"{dev} exposes {num_domains} locality domain(s); uGPU localization needs at "
            "least two (e.g. Rubin) and a driver / cuda-python with the locality-domain API"
        )
    cu_dev = get_cudevice(dev)
    resource = get_device_resource(cu_dev)
    params = []
    for i in range(num_domains):
        p = driver.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
        p.smCount = resource.sm.smCount // num_domains if backfill else 0
        p.coscheduledSmCount = 2
        flags = int(
            driver.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
        )
        if backfill:
            flags |= int(
                driver.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_BACKFILL
            )
        p.flags = flags
        p.localityDomainId = i
        params.append(p)
    groups = checkCudaErrors(
        driver.cuDevSmResourceSplit(num_domains, resource, 0, params)
    )[0]
    groups = list(groups)
    return create_green_ctx_streams(cu_dev, groups), groups


class _LocalizedAllocation:
    """Owner of an :func:`empty_by_locality_domain` mapping; torch keeps it alive."""

    def __init__(self, ptr: int, nbytes: int, typestr: str):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": typestr,
            "data": (ptr, False),
            "version": 3,
        }


def _unmap_localized(dev_index: int, ptr: int, va_bytes: int, chunks) -> None:
    torch.cuda.synchronize(dev_index)
    for offset, size, handle in chunks:
        checkCudaErrors(driver.cuMemUnmap(ptr + offset, size))
        checkCudaErrors(driver.cuMemRelease(handle))
    checkCudaErrors(driver.cuMemAddressFree(ptr, va_bytes))


def empty_by_locality_domain(
    shape: Tuple[int, ...], dtype: torch.dtype, dev: torch.device
) -> Tuple[torch.Tensor, List[int]]:
    r"""
    Allocate an uninitialized, contiguous tensor whose rows (dim 0) are split between the
    locality domains of ``dev``: rows ``[row_starts[i], row_starts[i + 1])`` live in domain
    ``i``'s memory. One virtual address range is mapped onto per-domain physical memory, so
    the result is an ordinary tensor (e.g. a state pool indexed by slot).

    The split points are ``i * shape[0] // num_domains`` rounded down to the allocation
    granularity. The memory is released when the tensor and all its views are freed.

    Args:
        shape: Tensor shape; dim 0 is split between the domains.
        dtype: Element type.
        dev: CUDA device with at least two locality domains.

    Returns:
        tensor: The tensor.
        row_starts: ``num_domains + 1`` row boundaries (first is 0, last is ``shape[0]``).
    """
    num_domains = get_locality_domain_count(dev)
    if num_domains < 2:
        raise RuntimeError(
            f"{dev} exposes {num_domains} locality domain(s); empty_by_locality_domain "
            "needs at least two"
        )
    dev = torch.device(
        "cuda", dev.index if dev.index is not None else torch.cuda.current_device()
    )
    cu_dev = get_cudevice(dev)
    rows = shape[0]
    row_bytes = torch.tensor([], dtype=dtype).element_size()
    for s in shape[1:]:
        row_bytes *= s

    def prop_for(domain: int):
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = (
            driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        )
        prop.location.localized.deviceId = int(cu_dev)
        prop.location.localized.localityDomainId = domain
        return prop

    gran = int(
        checkCudaErrors(
            driver.cuMemGetAllocationGranularity(
                prop_for(0),
                driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
    )
    # rows per granularity-aligned chunk, so every domain except the last starts aligned
    step = gran // math.gcd(row_bytes, gran) if row_bytes else 1
    row_starts = [
        ((i * rows) // num_domains) // step * step for i in range(num_domains)
    ]
    row_starts.append(rows)
    sizes = [
        round_up((row_starts[i + 1] - row_starts[i]) * row_bytes, gran)
        for i in range(num_domains)
    ]
    va_bytes = max(sum(sizes), gran)
    ptr = int(checkCudaErrors(driver.cuMemAddressReserve(va_bytes, 0, 0, 0)))
    chunks = []
    offset = 0
    for i, size in enumerate(sizes):
        if size == 0:
            continue
        handle = checkCudaErrors(driver.cuMemCreate(size, prop_for(i), 0))
        checkCudaErrors(driver.cuMemMap(ptr + offset, size, 0, handle, 0))
        chunks.append((offset, size, handle))
        offset += size
    access = driver.CUmemAccessDesc()
    access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = int(cu_dev)
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    if offset:
        checkCudaErrors(driver.cuMemSetAccess(ptr, offset, [access], 1))
    nbytes = rows * row_bytes
    owner = _LocalizedAllocation(ptr, nbytes, "|u1")
    weakref.finalize(
        owner, _unmap_localized, dev.index, ptr, va_bytes, chunks
    ).atexit = False
    flat = torch.as_tensor(owner, device=dev)
    return flat.view(dtype).view(shape), row_starts
