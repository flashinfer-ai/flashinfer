# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Code imported from TensorRT-LLM/tensorrt_llm/_mnnvl_utils.py
import ctypes
import logging
import os
import socket
import array
import random

import contextlib

from abc import ABC, abstractmethod
from dataclasses import dataclass
import platform
import sys
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import pynvml

logger = logging.getLogger(__name__)

import torch

try:
    # cuda-python >= 12.9 (has cuda.bindings.driver)
    from cuda.bindings import driver as cuda
except ImportError:
    try:
        # cuda-python < 12.9 (no cuda.bindings.driver, use cuda as driver)
        # from cuda import cuda is not available in cuda-python >= 13.0
        from cuda import cuda
    except ImportError as e:
        raise ImportError(
            "Could not import the 'cuda' module. "
            "Please install cuda-python that matches your CUDA version."
        ) from e

from ..cuda_utils import checkCudaErrors
from .dlpack_utils import create_dlpack_capsule, pack_strided_memory
from .mapping import Mapping

# mpi4py only exports MPI_COMM_TYPE_SHARED, so we define OMPI_COMM_TYPE_HOST here
OMPI_COMM_TYPE_HOST = 9

# Constants from C++ header
SIGNAL_PAD_SIZE = 2048  # kSIGNAL_PAD_SIZE from header

MNNVL_DEBUG = False


def round_up(val: int, gran: int) -> int:
    """Efficient implementation assuming gran is a power of 2"""
    return (val + gran - 1) & ~(gran - 1)


def create_tensor_from_cuda_memory(
    ptr: int, shape: tuple, dtype: torch.dtype, device_id: int
) -> torch.Tensor:
    r"""Wrap a CUDA memory allocation as a PyTorch tensor via DLPack.

    Parameters
    ----------
    ptr : int
        CUDA memory pointer (device address) as an integer.
    shape : tuple
        Desired tensor shape.
    dtype : torch.dtype
        Element dtype of the resulting tensor.
    device_id : int
        CUDA device ID hosting ``ptr``.

    Returns
    -------
    torch.Tensor
        A tensor that views the provided device memory.
    """
    # Calculate total size in elements
    numel = 1
    for dim in shape:
        numel *= dim

    # Get element size in bytes
    element_size = torch.tensor([], dtype=dtype).element_size()

    # Create DLPack capsule for contiguous memory (stride = element_size, num_segments = numel)
    capsule_wrapper = create_dlpack_capsule(
        ptr, element_size, element_size, numel, dtype, device_id
    )

    # Convert to tensor and reshape
    tensor = torch.utils.dlpack.from_dlpack(capsule_wrapper.capsule)
    tensor._capsule_wrapper = capsule_wrapper  # Keep reference to prevent GC

    # Reshape to desired shape
    return tensor.view(shape)


def test_cuda_memory_access(ptr: int, size: int, device_id: int) -> bool:
    """
    Test if CUDA memory at ptr is accessible by trying to read/write a small amount.

    Args:
        ptr: CUDA memory pointer
        size: Size of memory region
        device_id: CUDA device ID

    Returns:
        True if memory is accessible, False otherwise
    """
    try:
        # Test with a small 4-byte read/write
        test_size = min(4, size)
        host_data = bytearray(test_size)

        # Try to copy from device to host
        checkCudaErrors(cuda.cuMemcpyDtoH(host_data, ptr, test_size))

        # Try to copy back from host to device
        checkCudaErrors(cuda.cuMemcpyHtoD(ptr, host_data, test_size))

        logger.debug("Memory access test PASSED for ptr=0x%x", ptr)
        return True
    except Exception as e:
        logger.debug("Memory access test FAILED for ptr=0x%x: %s", ptr, e)
        return False


def alloc_and_copy_to_cuda(host_ptr_array: List[int]) -> Optional[int]:
    r"""Allocate a device buffer holding the supplied host pointer array.

    The host pointers are packed into a ``uint64`` array, copied to device,
    and the resulting device pointer is returned.

    Parameters
    ----------
    host_ptr_array : list[int]
        Sequence of host-side pointer values (interpreted as ``uint64``).

    Returns
    -------
    Optional[int]
        Device pointer to the packed array, or ``None`` if
        ``host_ptr_array`` is empty.
    """
    if not host_ptr_array:
        return None

    ArrayType = ctypes.c_uint64 * len(host_ptr_array)
    c_array = ArrayType(*host_ptr_array)
    size_in_bytes = ctypes.sizeof(c_array)

    device_ptr: cuda.CUdeviceptr = checkCudaErrors(cuda.cuMemAlloc(size_in_bytes))
    checkCudaErrors(
        cuda.cuMemcpyHtoD(device_ptr, ctypes.addressof(c_array), size_in_bytes)
    )
    # c_array should be freed by GC

    return int(device_ptr)


class CommBackend(ABC):
    """Abstract communication backend interface"""

    @abstractmethod
    def Get_rank(self) -> int: ...

    @abstractmethod
    def Get_size(self) -> int: ...

    @abstractmethod
    def allgather(self, data: int) -> List[int]: ...

    @abstractmethod
    def bcast(self, data: Any, root: int) -> Any: ...

    @abstractmethod
    def barrier(self) -> None: ...

    @abstractmethod
    def Split(self, color: int, key: int) -> "CommBackend": ...


if TYPE_CHECKING:
    from mpi4py import MPI  # noqa: F401


def lazy_import_mpi():
    """Lazy import for mpi4py"""
    try:
        from mpi4py import MPI

        return MPI
    except ImportError as err:
        raise ImportError("mpi4py is not installed") from err  # type: ignore[no-redef]


class MpiComm:  # type: ignore[no-redef]
    _comm: Any = None
    _MPI: Any = None

    @classmethod
    def _get_mpi(cls):
        if cls._MPI is None:
            cls._MPI = lazy_import_mpi()
            cls._comm = cls._MPI.COMM_WORLD
        return cls._MPI

    @classmethod
    def set_mpi_comm(cls, new_comm: Any):
        cls._get_mpi()
        # Optional: add type checking here
        cls._comm = new_comm

    def __getattr__(self, name):
        if self._comm is None:
            self._get_mpi()
        return getattr(self._comm, name)


class MPIBackend(CommBackend):
    def __init__(self):
        self._mpicomm = MpiComm()

    def Get_rank(self) -> int:
        return self._mpicomm.Get_rank()

    def Get_size(self) -> int:
        return self._mpicomm.Get_size()

    def allgather(self, data: int) -> List[int]:
        return self._mpicomm.allgather(data)

    def bcast(self, data: Any, root: int) -> Any:
        return self._mpicomm.bcast(data, root)

    def barrier(self):
        self._mpicomm.Barrier()

    def Split(self, color: int, key: int) -> CommBackend:
        self._mpicomm = self._mpicomm.Split(color, key)
        return MPIBackend()  # Returns new adapter


class TorchDistBackend(CommBackend):
    """Communication backend using torch.distributed"""

    def __init__(self, group: Optional[Any] = None):
        """
        Initialize TorchDistBackend.

        Args:
            group: Optional process group. If None, uses the default process group.
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Please call torch.distributed.init_process_group() first."
            )
        self._group = group
        self._dist = dist

    def Get_rank(self) -> int:
        return self._dist.get_rank(self._group)

    def Get_size(self) -> int:
        return self._dist.get_world_size(self._group)

    def allgather(self, data: Any) -> List[Any]:
        """All-gather arbitrary Python objects across all ranks."""
        output_list = [None] * self.Get_size()
        self._dist.all_gather_object(output_list, data, group=self._group)
        return output_list

    def bcast(self, data: Any, root: int) -> Any:
        """Broadcast a Python object from root to all ranks.

        Args:
            data: object to broadcast (only used on the root rank).
            root: group-local rank of the sender (consistent with MPI).
        """
        object_list = [data]
        global_root = (
            self._dist.get_global_rank(self._group, root)
            if self._group is not None
            else root
        )
        self._dist.broadcast_object_list(
            object_list, src=global_root, group=self._group
        )
        return object_list[0]

    def barrier(self) -> None:
        self._dist.barrier(group=self._group)

    def Split(self, color: int, key: int) -> "TorchDistBackend":
        """
        Split the communicator into sub-groups based on color.

        All processes with the same color will be in the same new group.
        The key determines the rank ordering within the new group.

        Args:
            color: Processes with the same color are placed in the same group
            key: Determines rank ordering within the new group (lower key = lower rank)

        Returns:
            New TorchDistBackend with the split process group
        """
        # Gather (color, key, global_rank) from all processes
        global_rank = self.Get_rank()

        all_info = self.allgather((color, key, global_rank))

        # Group ranks by color, sort by key within each group
        color_groups: Dict[int, List[tuple]] = {}
        for c, k, r in all_info:
            if c not in color_groups:
                color_groups[c] = []
            color_groups[c].append((k, r))

        # Sort each group by key to determine rank ordering
        for c in color_groups:
            color_groups[c].sort(key=lambda x: x[0])

        # Find my new group's ranks (in sorted order by key)
        my_group_ranks = [r for _, r in color_groups[color]]

        # Create new process group with the ranks in my color group
        new_group = self._dist.new_group(ranks=my_group_ranks)

        return TorchDistBackend(group=new_group)


@dataclass
class MnnvlConfig:
    """Configuration for MNNVL memory management"""

    comm_backend: Optional[CommBackend] = None
    allocation_granularity: int = 0
    fabric_page_size: int = 1 << 29  # 512MB


@dataclass
class _MnnvlAllocationRecord:
    mapping: Mapping
    comm: CommBackend
    comm_size: int
    comm_rank: int
    aligned_size: int
    mem_handles: List[Any]
    start_address: int
    rank_stride: int
    address_offset: int
    mapped: bool = True


class MnnvlMemory:  # type: ignore[no-redef]
    initialized: bool = False

    current_mem_offset: int = 0
    current_rank_stride: int = 0  # stride for ranks and also address space size.
    current_start_address: int = 0

    # allocation granularity
    allocation_granularity: int = 0

    # fabric address page size (512 MB)
    fabric_page_size: int = 1 << 29

    # MPI communicator
    comm: Optional[CommBackend] = None

    dev_id: int = None

    allocated_map: Dict[int, Any] = {}
    address_refcnt: Dict[int, Any] = {}

    config: Optional[MnnvlConfig] = None

    def __init__(self, mapping: Mapping, size: int):
        self.mapping = mapping
        self.segment_size = size
        self.ptr, self.rank_stride = MnnvlMemory.open_mnnvl_memory(self.mapping, size)

    def __del__(self):
        if not sys.is_finalizing():
            # When open_mnnvl_memory fails, self.ptr may not be set. In that case, we should not call close_mnnvl_memory.
            if hasattr(self, "ptr"):
                MnnvlMemory.close_mnnvl_memory(self.ptr)

    def as_torch_strided_tensor(self, dtype):
        record = MnnvlMemory.allocated_map[self.ptr]
        num_segments = record.comm.Get_size()
        return pack_strided_memory(
            self.ptr,
            self.segment_size,
            self.rank_stride,
            num_segments,
            dtype,
            MnnvlMemory.dev_id,
        )

    def detach_physical_keep_va(
        self, *, synchronize: bool = True, barrier: bool = True
    ) -> None:
        """Release mapped MNNVL handles while keeping graph-visible VAs alive."""
        MnnvlMemory.detach_mnnvl_memory_keep_va(
            self.ptr, synchronize=synchronize, barrier=barrier
        )

    def remap_physical_same_va(
        self,
        *,
        config: Optional[MnnvlConfig] = None,
        synchronize: bool = True,
        barrier: bool = True,
        zero_local: bool = True,
    ) -> None:
        """Recreate MNNVL handles and map them at the original virtual addresses."""
        MnnvlMemory.remap_mnnvl_memory_same_va(
            self.ptr,
            config=config,
            synchronize=synchronize,
            barrier=barrier,
            zero_local=zero_local,
        )

    def get_graph_visible_addresses(self) -> Dict[str, Any]:
        """Return the VA/layout state captured by CUDA graph-visible tensors."""
        record = MnnvlMemory.allocated_map[self.ptr]
        return {
            "ptr": self.ptr,
            "segment_size": self.segment_size,
            "rank_stride": self.rank_stride,
            "start_address": record.start_address,
            "address_offset": record.address_offset,
            "aligned_size": record.aligned_size,
            "comm_size": record.comm_size,
            "comm_rank": record.comm_rank,
            "rank_ptrs": [
                record.start_address + i * record.rank_stride + record.address_offset
                for i in range(record.comm_size)
            ],
        }

    def validate_graph_visible_addresses(
        self, expected: Optional[Dict[str, Any]], tensor: Optional[torch.Tensor] = None
    ) -> None:
        """Validate that graph-visible VAs and optional tensor metadata are stable."""
        if expected is None:
            raise RuntimeError("Missing captured MNNVL graph-visible address metadata")

        current = self.get_graph_visible_addresses()
        for key in (
            "ptr",
            "segment_size",
            "rank_stride",
            "start_address",
            "address_offset",
            "aligned_size",
            "comm_size",
            "comm_rank",
            "rank_ptrs",
        ):
            if current[key] != expected[key]:
                raise RuntimeError(
                    f"MNNVL graph-visible address changed for {key}: "
                    f"{current[key]!r} != {expected[key]!r}"
                )

        if tensor is not None:
            element_size = tensor.element_size()
            if tensor.data_ptr() != current["ptr"]:
                raise RuntimeError(
                    f"MNNVL tensor data_ptr changed: {tensor.data_ptr()} "
                    f"!= {current['ptr']}"
                )
            if tensor.size(0) != current["comm_size"]:
                raise RuntimeError(
                    f"MNNVL tensor rank dimension changed: {tensor.size(0)} "
                    f"!= {current['comm_size']}"
                )
            expected_segment_elements = current["segment_size"] // element_size
            if tensor.size(1) != expected_segment_elements:
                raise RuntimeError(
                    "MNNVL tensor segment dimension changed: "
                    f"{tensor.size(1)} != {expected_segment_elements}"
                )
            if tensor.stride(0) * element_size != current["rank_stride"]:
                raise RuntimeError(
                    "MNNVL tensor rank stride changed: "
                    f"{tensor.stride(0) * element_size} != "
                    f"{current['rank_stride']}"
                )

    @staticmethod
    def initialize():
        if not MnnvlMemory.initialized:
            # use a dummy torch CUDA tensor to trigger CUDA context initialization
            _ = torch.empty(1, device="cuda")
            # ensure nvml is initialized.
            try:
                pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError_Uninitialized:
                pynvml.nvmlInit()
            MnnvlMemory.initialized = True

    @staticmethod
    def set_comm_from_config(mapping: Mapping, config: MnnvlConfig = None):
        MnnvlMemory.config = config or MnnvlConfig(comm_backend=MPIBackend())
        if MnnvlMemory.config.comm_backend is None:
            raise RuntimeError("MNNVL config must provide a communication backend")
        comm = MnnvlMemory.config.comm_backend.Split(
            mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank
        )
        MnnvlMemory.comm = comm  # type: ignore[assignment]

    @staticmethod
    def get_comm(mapping: Mapping):
        if MnnvlMemory.comm is not None:
            return MnnvlMemory.comm
        if MnnvlMemory.config is not None:
            config = MnnvlMemory.config
            if config.comm_backend is not None:
                comm = config.comm_backend.Split(
                    mapping.pp_rank * mapping.cp_size + mapping.cp_rank,
                    mapping.tp_rank,
                )
                MnnvlMemory.comm = comm
                return comm
        comm = MpiComm().Split(
            mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank
        )
        MnnvlMemory.comm = comm
        return comm

    @staticmethod
    def refresh_comm_from_config(mapping: Mapping, config: MnnvlConfig) -> CommBackend:
        if config.comm_backend is None:
            raise RuntimeError("MNNVL remap config must provide a communication backend")
        MnnvlMemory.config = config
        comm = config.comm_backend.Split(
            mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank
        )
        MnnvlMemory.comm = comm
        return comm

    @staticmethod
    def get_allocation_prop(dev_id: int):
        location = cuda.CUmemLocation()
        location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        location.id = dev_id
        allocation_prop = cuda.CUmemAllocationProp()
        allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        # TODO: We differentiate FABRIC for GB200 (aarch64) and POSIX_FILE_DESCRIPTOR for B200 (x86_64).
        # May need to find a better way to handle this.
        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        if is_on_aarch64:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
        else:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
        allocation_prop.location = location
        return allocation_prop

    @staticmethod
    def get_allocation_granularity(dev_id: int):
        if MnnvlMemory.allocation_granularity != 0:
            return MnnvlMemory.allocation_granularity
        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        option = cuda.CUmemAllocationGranularity_flags(
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        granularity = checkCudaErrors(
            cuda.cuMemGetAllocationGranularity(prop=allocation_prop, option=option)
        )
        MnnvlMemory.allocation_granularity = granularity
        return MnnvlMemory.allocation_granularity

    @staticmethod
    def _exchange_shareable_handles(
        comm: CommBackend,
        allocation_prop: cuda.CUmemAllocationProp,
        allocated_mem_handle: Any,
    ) -> List[Any]:
        shareable_handle = checkCudaErrors(
            cuda.cuMemExportToShareableHandle(
                allocated_mem_handle, allocation_prop.requestedHandleTypes, 0
            )
        )
        if (
            allocation_prop.requestedHandleTypes
            == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        ):
            return comm.allgather(shareable_handle.data)

        remote_fds = []
        try:
            all_handles_data = comm.allgather(shareable_handle)
            all_pids = comm.allgather(os.getpid())
            libc = ctypes.CDLL(None, use_errno=True)
            syscall = libc.syscall
            SYS_pidfd_open = 434
            SYS_pidfd_getfd = 438

            for pid, fd in zip(all_pids, all_handles_data, strict=True):
                pidfd = syscall(SYS_pidfd_open, pid, 0)
                if pidfd < 0:
                    err = ctypes.get_errno()
                    raise RuntimeError(
                        f"pidfd_open({pid}) failed with errno {err}: "
                        f"{os.strerror(err)}"
                    )
                try:
                    remote_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0)
                    if remote_fd < 0:
                        err = ctypes.get_errno()
                        error_msg = (
                            f"pidfd_getfd(pidfd={pidfd}, fd={fd}) failed with "
                            f"errno {err}: {os.strerror(err)}."
                        )
                        if err == 1:  # EPERM
                            error_msg += (
                                " Permission denied. If running in a container, "
                                "try adding --cap-add=SYS_PTRACE to your docker "
                                "run command."
                            )
                        else:
                            error_msg += (
                                " This may be due to kernel version "
                                "(requires Linux 5.6+)."
                            )
                        raise RuntimeError(error_msg)
                    remote_fds.append(remote_fd)
                finally:
                    os.close(pidfd)

            # Keep exported fds alive until all ranks have duplicated them.
            comm.barrier()
            exchanged_fds = remote_fds
            remote_fds = []
            return exchanged_fds
        finally:
            for fd in remote_fds:
                os.close(fd)
            os.close(shareable_handle)

    @staticmethod
    def _create_and_map_mnnvl_handles(
        mapping: Mapping,
        aligned_size: int,
        start_address: int,
        rank_stride: int,
        address_offset: int,
        *,
        comm: Optional[CommBackend] = None,
        zero_local: bool = False,
    ) -> List[Any]:
        dev = checkCudaErrors(cuda.cuCtxGetDevice())
        dev_id = int(dev)
        assert dev_id == MnnvlMemory.dev_id, (
            f"Different dev_id found dev_id={dev_id} but "
            f"MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
        )
        comm = comm or MnnvlMemory.get_comm(mapping)
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        is_posix_fd = (
            allocation_prop.requestedHandleTypes
            == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        allocated_mem_handle = checkCudaErrors(
            cuda.cuMemCreate(aligned_size, allocation_prop, flags=0)
        )
        created_mem_handles = [allocated_mem_handle]
        mapped_ptrs: List[int] = []
        posix_fds: List[Optional[int]] = []

        def close_posix_fd(index: int) -> None:
            fd = posix_fds[index]
            if fd is not None:
                os.close(fd)
                posix_fds[index] = None

        try:
            all_handles_data = MnnvlMemory._exchange_shareable_handles(
                comm, allocation_prop, allocated_mem_handle
            )
            posix_fds = list(all_handles_data) if is_posix_fd else []

            madesc = cuda.CUmemAccessDesc()
            madesc.location = allocation_prop.location
            madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

            mem_handles = [None] * comm_size

            for i, remote_handle_data in enumerate(all_handles_data):
                rank_ptr = start_address + rank_stride * i + address_offset
                try:
                    if i == comm_rank:
                        mem_handle = allocated_mem_handle
                    else:
                        mem_handle = checkCudaErrors(
                            cuda.cuMemImportFromShareableHandle(
                                remote_handle_data,
                                allocation_prop.requestedHandleTypes,
                            )
                        )
                        created_mem_handles.append(mem_handle)
                    mem_handles[i] = mem_handle
                    checkCudaErrors(
                        cuda.cuMemMap(rank_ptr, aligned_size, 0, mem_handle, 0)
                    )
                    mapped_ptrs.append(rank_ptr)
                finally:
                    if is_posix_fd:
                        close_posix_fd(i)

                checkCudaErrors(
                    cuda.cuMemSetAccess(rank_ptr, aligned_size, [madesc], 1)
                )

            if zero_local:
                local_ptr = start_address + rank_stride * comm_rank + address_offset
                checkCudaErrors(cuda.cuMemsetD8(local_ptr, 0, aligned_size))

            return mem_handles
        except Exception as exc:
            cleanup_errors = []
            for i in range(len(posix_fds)):
                try:
                    close_posix_fd(i)
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            for rank_ptr in reversed(mapped_ptrs):
                try:
                    checkCudaErrors(cuda.cuMemUnmap(rank_ptr, aligned_size))
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            for mem_handle in reversed(created_mem_handles):
                try:
                    checkCudaErrors(cuda.cuMemRelease(mem_handle))
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise RuntimeError(
                    "Failed to roll back partial MNNVL handle mapping after "
                    f"{exc!r}: {cleanup_errors!r}"
                ) from exc
            raise

    @staticmethod
    def new_mnnvl_memory_address(mapping: Mapping, size: int):
        page_count = (
            size + MnnvlMemory.fabric_page_size - 1
        ) // MnnvlMemory.fabric_page_size
        current_rank_stride = page_count * MnnvlMemory.fabric_page_size
        logging.info(
            f"[MnnvlMemory] creating address with stride={current_rank_stride}"
        )
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        address_size = current_rank_stride * comm_size
        ptr = checkCudaErrors(
            cuda.cuMemAddressReserve(address_size, MnnvlMemory.fabric_page_size, 0, 0)
        )
        MnnvlMemory.current_start_address = int(ptr)
        MnnvlMemory.current_rank_stride = current_rank_stride
        MnnvlMemory.current_mem_offset = 0

    @staticmethod
    def open_mnnvl_memory(mapping: Mapping, size: int):
        dev = checkCudaErrors(cuda.cuCtxGetDevice())
        dev_id = int(dev)
        if MnnvlMemory.dev_id is None:
            MnnvlMemory.dev_id = dev_id
        assert dev_id == MnnvlMemory.dev_id, (
            f"Different dev_id found dev_id={dev_id} but MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
        )
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        all_rank_allocate_sizes = comm.allgather(size)
        assert len(all_rank_allocate_sizes) == comm_size
        assert all(x == size for x in all_rank_allocate_sizes), (
            "Not all rank allocating same size."
        )
        granularity = MnnvlMemory.get_allocation_granularity(dev_id)
        aligned_size = (size + granularity - 1) // granularity * granularity

        if (
            MnnvlMemory.current_mem_offset + aligned_size
            > MnnvlMemory.current_rank_stride
        ):
            MnnvlMemory.new_mnnvl_memory_address(mapping, aligned_size)

        assert (
            MnnvlMemory.current_mem_offset + aligned_size
            <= MnnvlMemory.current_rank_stride
        )

        ptr = MnnvlMemory.current_start_address + MnnvlMemory.current_mem_offset
        stride = MnnvlMemory.current_rank_stride
        comm = MnnvlMemory.get_comm(mapping)
        mem_handles = MnnvlMemory._create_and_map_mnnvl_handles(
            mapping,
            aligned_size,
            MnnvlMemory.current_start_address,
            MnnvlMemory.current_rank_stride,
            MnnvlMemory.current_mem_offset,
            comm=comm,
        )
        MnnvlMemory.allocated_map[ptr] = _MnnvlAllocationRecord(
            mapping=mapping,
            comm=comm,
            comm_size=comm.Get_size(),
            comm_rank=comm.Get_rank(),
            aligned_size=aligned_size,
            mem_handles=mem_handles,
            start_address=MnnvlMemory.current_start_address,
            rank_stride=MnnvlMemory.current_rank_stride,
            address_offset=MnnvlMemory.current_mem_offset,
        )
        MnnvlMemory.address_refcnt[MnnvlMemory.current_start_address] = (
            MnnvlMemory.address_refcnt.get(MnnvlMemory.current_start_address, 0) + 1
        )

        MnnvlMemory.current_mem_offset += aligned_size
        return ptr, stride

    @staticmethod
    def _validate_remap_comm(
        record: _MnnvlAllocationRecord, comm: CommBackend
    ) -> None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()
        if comm_size != record.comm_size or comm_rank != record.comm_rank:
            raise RuntimeError(
                "Restored MNNVL communicator does not match the graph-visible "
                "allocation layout: "
                f"rank/size {comm_rank}/{comm_size} != "
                f"{record.comm_rank}/{record.comm_size}"
            )

    @staticmethod
    def close_mnnvl_memory(ptr: int):
        record = MnnvlMemory.allocated_map.pop(ptr)
        comm = record.comm
        comm_size = comm.Get_size()
        if record.mapped:
            for i in range(comm_size):
                rank_ptr = (
                    record.start_address
                    + i * record.rank_stride
                    + record.address_offset
                )
                checkCudaErrors(cuda.cuMemUnmap(rank_ptr, record.aligned_size))
                checkCudaErrors(cuda.cuMemRelease(record.mem_handles[i]))
            record.mapped = False
        MnnvlMemory.address_refcnt[record.start_address] -= 1

        if MnnvlMemory.address_refcnt[record.start_address] == 0:
            MnnvlMemory.address_refcnt.pop(record.start_address)
            device_ptr = cuda.CUdeviceptr(record.start_address)
            checkCudaErrors(
                cuda.cuMemAddressFree(
                    device_ptr, comm_size * record.rank_stride
                )
            )
            if record.start_address == MnnvlMemory.current_start_address:
                MnnvlMemory.current_start_address = 0
                MnnvlMemory.current_rank_stride = 0
                MnnvlMemory.current_mem_offset = 0

    @staticmethod
    def detach_mnnvl_memory_keep_va(
        ptr: int, *, synchronize: bool = True, barrier: bool = True
    ) -> None:
        record = MnnvlMemory.allocated_map[ptr]
        comm = record.comm
        mapped_states = comm.allgather(record.mapped)
        if any(mapped_states) and not all(mapped_states):
            raise RuntimeError("Inconsistent MNNVL mapped state across ranks")
        if not any(mapped_states):
            if barrier:
                comm.barrier()
                comm.barrier()
            return
        comm_size = comm.Get_size()

        if synchronize:
            checkCudaErrors(cuda.cuCtxSynchronize())
        if barrier:
            comm.barrier()

        for i in range(comm_size):
            rank_ptr = (
                record.start_address + i * record.rank_stride + record.address_offset
            )
            checkCudaErrors(cuda.cuMemUnmap(rank_ptr, record.aligned_size))
            checkCudaErrors(cuda.cuMemRelease(record.mem_handles[i]))

        record.mem_handles = [None] * comm_size
        record.mapped = False

        if barrier:
            comm.barrier()

    @staticmethod
    def remap_mnnvl_memory_same_va(
        ptr: int,
        *,
        config: Optional[MnnvlConfig] = None,
        synchronize: bool = True,
        barrier: bool = True,
        zero_local: bool = True,
    ) -> None:
        record = MnnvlMemory.allocated_map[ptr]
        if config is not None and record.mapped:
            raise RuntimeError(
                "Cannot refresh MNNVL communicator while allocation is still mapped; "
                "call detach_physical_keep_va before checkpoint remap"
            )
        if config is not None:
            comm = MnnvlMemory.refresh_comm_from_config(record.mapping, config)
            MnnvlMemory._validate_remap_comm(record, comm)
            record.comm = comm
        else:
            comm = record.comm
            MnnvlMemory._validate_remap_comm(record, comm)
        mapped_states = comm.allgather(record.mapped)
        if any(mapped_states) and not all(mapped_states):
            raise RuntimeError("Inconsistent MNNVL mapped state across ranks")
        if all(mapped_states):
            if barrier:
                comm.barrier()
                comm.barrier()
            return

        if synchronize:
            checkCudaErrors(cuda.cuCtxSynchronize())
        if barrier:
            comm.barrier()

        record.mem_handles = MnnvlMemory._create_and_map_mnnvl_handles(
            record.mapping,
            record.aligned_size,
            record.start_address,
            record.rank_stride,
            record.address_offset,
            comm=comm,
            zero_local=zero_local,
        )
        record.mapped = True

        if barrier:
            comm.barrier()

    @staticmethod
    def support_nvlink(need_all_up: bool = True):
        dev_id = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        link_count = pynvml.NVML_NVLINK_MAX_LINKS
        active_links = 0
        available_links = 0
        for link_idx in range(link_count):
            try:
                if pynvml.nvmlDeviceGetNvLinkCapability(
                    handle, link_idx, pynvml.NVML_NVLINK_CAP_P2P_SUPPORTED
                ):
                    available_links += 1
                    is_active = pynvml.nvmlDeviceGetNvLinkState(handle, link_idx)
                    if is_active:
                        active_links += 1
            except pynvml.NVMLError_NotSupported:
                continue
        return (
            active_links == available_links and available_links > 0
            if need_all_up
            else available_links > 0
        )

    @staticmethod
    def supports_mnnvl() -> bool:
        # TODO:
        # We check if it has all NVLink up now.
        # But it is not equivalent to MNNVL support.
        # May need better support check.
        support_nvlink_and_all_up = MnnvlMemory.support_nvlink(True)
        return support_nvlink_and_all_up


# The helper class for passing the FD handle over the socket.
class IpcSocket:
    """Unix Domain Socket for IPC file descriptor passing"""

    def __init__(self, rank: int, op_id: int, use_abstract=True):
        """
        Initialize IPC socket

        Args:
            rank: Process rank
            op_id: Unique operation ID (hash)
            use_abstract: Use Linux abstract socket namespace
        """
        self.rank = rank
        self.op_id = op_id
        self.use_abstract = use_abstract

        # Create Unix domain socket (DGRAM for compatibility with C code)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

        # Create unique socket name
        socket_name = f"/tmp/mcastmem-socket-{rank}-{op_id:x}"

        if use_abstract:
            # Linux abstract socket: prepend null byte
            self.socket_path = "\0" + socket_name
        else:
            self.socket_path = socket_name
            # Remove existing socket file if it exists
            with contextlib.suppress(FileNotFoundError):
                os.unlink(socket_name)

        # Bind socket
        self.sock.bind(self.socket_path)

    def send_fd(self, fd: int, dest_rank: int, dest_op_id: Optional[int] = None):
        """
        Send a file descriptor to another process

        Args:
            fd: File descriptor to send
            dest_rank: Destination process rank
            dest_op_id: Destination operation ID
        """
        # Construct destination socket path
        dest_op_id = dest_op_id or self.op_id
        dest_socket_name = f"/tmp/mcastmem-socket-{dest_rank}-{dest_op_id:x}"

        if self.use_abstract:
            dest_path = "\0" + dest_socket_name
        else:
            dest_path = dest_socket_name

        # Prepare message with file descriptor
        # Send dummy byte as data (required)
        dummy_data = b"\x00"

        # Pack file descriptor in ancillary data (SCM_RIGHTS)
        fds = array.array("i", [fd])
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())]

        # Send message with file descriptor
        self.sock.sendmsg([dummy_data], ancillary, 0, dest_path)

    def recv_fd(self):
        """
        Receive a file descriptor from another process

        Returns:
            int: Received file descriptor
        """
        # Receive message with ancillary data
        # Maximum size for ancillary data containing one fd
        fds = array.array("i")
        msg, ancdata, flags, addr = self.sock.recvmsg(
            1,
            socket.CMSG_SPACE(
                fds.itemsize
            ),  # Buffer size for dummy data  # Ancillary data size
        )

        # Extract file descriptor from ancillary data
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                fds = array.array("i")
                fds.frombytes(
                    cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)]
                )
                return fds[0]

        raise RuntimeError("No file descriptor received")

    def close(self):
        """Close the socket"""
        self.sock.close()
        if not self.use_abstract and self.socket_path:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.socket_path)


class HandleExchanger(ABC):
    """Abstract interface for exchanging CUDA shareable handles across ranks."""

    def __init__(self, comm_backend: "CommBackend", group_rank: int, group_size: int):
        self.comm = comm_backend
        self.rank = group_rank
        self.size = group_size

    @property
    @abstractmethod
    def handle_type(self) -> cuda.CUmemAllocationHandleType:
        """The CUDA handle type this exchanger works with."""
        ...

    @abstractmethod
    def allgather(self, local_handle) -> List:
        """All-gather shareable handles from all ranks."""
        ...

    @abstractmethod
    def broadcast(self, handle, root: int):
        """Broadcast a handle from root to all ranks."""
        ...

    @abstractmethod
    def cleanup(self, handle) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class FabricHandleExchanger(HandleExchanger):
    """Handle exchange using CUDA Fabric handles via MPI/collective backend."""

    @property
    def handle_type(self) -> cuda.CUmemAllocationHandleType:
        return cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC

    def allgather(self, local_handle) -> List:
        return self.comm.allgather(local_handle.data)

    def broadcast(self, handle, root: int):
        return self.comm.bcast(handle.data if handle else None, root=root)

    def cleanup(self, handle) -> None:
        pass  # No cleanup needed for Fabric handles.

    def close(self) -> None:
        pass  # No close needed for Fabric handles.


class PosixFDHandleExchanger(HandleExchanger):
    """Handle exchange using POSIX file descriptors via IPC sockets."""

    def __init__(self, comm_backend: "CommBackend", group_rank: int, group_size: int):
        super().__init__(comm_backend, group_rank, group_size)
        self._socket = self._init_ipc_socket()

    def _init_ipc_socket(self) -> IpcSocket:
        if self.rank == 0:
            opId = random.Random().randint(0, 2**64 - 1)
        else:
            opId = None
        opId = self.comm.bcast(opId, root=0)
        return IpcSocket(self.rank, opId)

    @property
    def handle_type(self) -> cuda.CUmemAllocationHandleType:
        return cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    def allgather(self, local_handle) -> List:
        result = [None] * self.size
        for i in range(self.size):
            self.comm.barrier()
            self._socket.send_fd(local_handle, (self.rank + i) % self.size)
            src = (self.rank + self.size - i) % self.size
            result[src] = self._socket.recv_fd()
        return result

    def broadcast(self, handle, root: int):
        if self.rank == root:
            for p in range(1, self.size):
                self.comm.barrier()
                self._socket.send_fd(handle, p)
            return handle
        else:
            # Ordered receive to avoid race condition
            for _ in range(self.rank):
                self.comm.barrier()
            result = self._socket.recv_fd()
            for _ in range(self.size - self.rank - 1):
                self.comm.barrier()
            return result

    def cleanup(self, handle) -> None:
        os.close(handle)

    def close(self) -> None:
        self._socket.close()


def is_mnnvl_fabric_supported(device_idx: int) -> bool:
    fabric_handle_supported = checkCudaErrors(
        cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
            device_idx,
        )
    )
    if fabric_handle_supported == 0:
        return False

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        fabric_info = pynvml.c_nvmlGpuFabricInfoV_t()
        pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(fabric_info))
        return (
            fabric_info.state >= pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
            and fabric_info.clusterUuid
            and fabric_info.clusterUuid[0] != 0
        )
    finally:
        pynvml.nvmlShutdown()


# TODO: This class follows similar logic with MnnvlMemory, but the latter use single instance mode to manage the memory allocation.
class SymmDeviceMemory:
    """Python port of SymmDeviceMemory from TensorRT-LLM"""

    def __init__(
        self,
        buf_size: int,
        group_size: int,
        group_rank: int,
        device_idx: int,
        comm_backend_for_handle_transfer: Optional[CommBackend] = None,
        enable_multicast: bool = True,
        allocate_signal_pads: bool = True,
    ):
        cu_device = checkCudaErrors(cuda.cuDeviceGet(device_idx))

        primary_ctx = checkCudaErrors(cuda.cuDevicePrimaryCtxRetain(cu_device))
        checkCudaErrors(cuda.cuCtxSetCurrent(primary_ctx))

        # Set CUDA device
        # Check if cuda.cudart is available and import accordingly
        from flashinfer.utils import has_cuda_cudart

        if has_cuda_cudart():
            # cuda-python <= 12.9
            import cuda.cudart as cudart
        else:
            # cuda-python >= 13.0
            import cuda.bindings.runtime as cudart

        checkCudaErrors(cudart.cudaSetDevice(device_idx))

        self.device_idx = device_idx
        self.group_size = group_size
        self.group_rank = group_rank
        self.buf_size = buf_size
        self.signal_pad_offset = 0
        self.allocation_size = 0
        self.comm_backend = comm_backend_for_handle_transfer or MPIBackend()

        # CUDA memory handles and pointers
        self.mc_ptr = 0  # CUdeviceptr mMcPtr
        self.uc_ptrs: List[int] = []  # std::vector<CUdeviceptr> mUcPtrs
        self.signal_pads: List[int] = []  # mSignalPads
        self.signal_pads_dev = 0  # std::vector<CUdeviceptr> mSignalPadsDev
        self.uc_ptrs_dev = 0
        self.mc_handle = 0  # CUmemGenericAllocationHandle mMcHandle
        self.uc_handles: List[
            int
        ] = []  # std::vector<CUmemGenericAllocationHandle> mUcHandles
        self._graph_visible_addresses: Optional[Dict[str, Any]] = None
        self._mapped = False

        # Signal pad constants
        self.SIGNAL_PAD_ALIGNMENT = 16
        self.SIGNAL_PAD_SIZE = SIGNAL_PAD_SIZE

        # Check if device supports multicasting
        multicast_supported = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                device_idx,
            )
        )
        if multicast_supported == 0:
            raise RuntimeError(
                "[SymmDeviceMemory] Device does not support multicasting."
            )

        # Calculate signal pad offset with alignment (matching C++ exactly)
        self.signal_pad_offset = round_up(buf_size, self.SIGNAL_PAD_ALIGNMENT)

        logging.info(
            f"[SymmDeviceMemory] Rank: {group_rank}, Group size: {group_size}, "
            f"device_idx: {device_idx}, "
            f"Signal pad offset: {self.signal_pad_offset}"
        )

        # Create handle exchanger
        self._exchanger: Optional[HandleExchanger] = self._create_handle_exchanger()
        self._alloc_mn_mcast_mem(buf_size, enable_multicast)
        self._mapped = True

        if allocate_signal_pads:
            # Initialize signal pads
            self.signal_pads = [0] * self.group_size
            for i in range(self.group_size):
                self.signal_pads[i] = self.uc_ptrs[i] + self.signal_pad_offset
                if i == self.group_rank:
                    checkCudaErrors(
                        cuda.cuMemsetD8(self.signal_pads[i], 0, self.SIGNAL_PAD_SIZE)
                    )

            self.signal_pads_dev = alloc_and_copy_to_cuda(self.signal_pads)
        self.uc_ptrs_dev = alloc_and_copy_to_cuda(self.uc_ptrs)
        self._graph_visible_addresses = self.get_graph_visible_addresses()

    def __del__(self):
        """Destructor - cleanup allocated memory"""

        if getattr(self, "_exchanger", None) is not None:
            self._exchanger.close()

        # Skip cleanup during Python finalization to avoid segfaults
        # Especially cause the CUDA context could be destroyed at this point.
        if sys.is_finalizing():
            return

        # Verify CUDA context is still valid
        try:
            cuda.cuCtxGetCurrent()
        except Exception as e:
            logger.warning("Destructor: CUDA context invalid, skipping cleanup: %s", e)
            return

        # Free device pointers
        if self.signal_pads_dev:
            checkCudaErrors(cuda.cuMemFree(self.signal_pads_dev))
            self.signal_pads_dev = 0
        if self.uc_ptrs_dev:
            checkCudaErrors(cuda.cuMemFree(self.uc_ptrs_dev))
            self.uc_ptrs_dev = 0

        # Unmap UC regions and release their handles
        if hasattr(self, "uc_handles") and self.uc_handles:
            for rank in range(self.group_size):
                if self.uc_handles[rank] != 0:
                    try:
                        # Release the handle
                        checkCudaErrors(cuda.cuMemRelease(self.uc_handles[rank]))
                        # Unmap the vmem
                        if rank < len(self.uc_ptrs) and self.uc_ptrs[rank]:
                            checkCudaErrors(
                                cuda.cuMemUnmap(
                                    self.uc_ptrs[rank], self.allocation_size
                                )
                            )
                    except Exception as e:
                        logger.warning(
                            "Destructor: Failed to release UC handle for rank %d: %s",
                            rank,
                            e,
                        )

            # Free the UC address space
            if hasattr(self, "uc_base_ptr") and self.uc_base_ptr:
                checkCudaErrors(
                    cuda.cuMemAddressFree(self.uc_base_ptr, self.total_uc_size)
                )
                self.uc_base_ptr = 0

        # Release MC handle
        if hasattr(self, "mc_ptr") and self.mc_ptr:
            if hasattr(self, "mc_handle") and self.mc_handle and self.mc_handle != 0:
                try:
                    checkCudaErrors(cuda.cuMemUnmap(self.mc_ptr, self.allocation_size))
                    checkCudaErrors(cuda.cuMemRelease(self.mc_handle))
                except Exception as e:
                    logger.warning("Destructor: Failed to release MC handle: %s", e)
            try:
                checkCudaErrors(
                    cuda.cuMemAddressFree(self.mc_ptr, self.allocation_size)
                )
            except Exception as e:
                logger.warning("Destructor: Failed to free MC VA: %s", e)
            self.mc_ptr = 0
            self.mc_handle = 0

    def _create_handle_exchanger(self) -> HandleExchanger:
        if is_mnnvl_fabric_supported(self.device_idx):
            return FabricHandleExchanger(
                self.comm_backend, self.group_rank, self.group_size
            )
        return PosixFDHandleExchanger(
            self.comm_backend, self.group_rank, self.group_size
        )

    def _close_handle_exchanger(self) -> None:
        exchanger = getattr(self, "_exchanger", None)
        if exchanger is not None:
            exchanger.close()
            self._exchanger = None

    def _validate_comm(self, comm_backend: CommBackend) -> None:
        comm_size = comm_backend.Get_size()
        comm_rank = comm_backend.Get_rank()
        if comm_size != self.group_size or comm_rank != self.group_rank:
            raise RuntimeError(
                "Restored symmetric-memory communicator does not match the "
                "graph-visible allocation layout: "
                f"rank/size {comm_rank}/{comm_size} != "
                f"{self.group_rank}/{self.group_size}"
            )

    def _collective_mapped_states(self, comm_backend: CommBackend) -> List[bool]:
        mapped_states = comm_backend.allgather(self._mapped)
        if len(mapped_states) != self.group_size:
            raise RuntimeError(
                "Symmetric-memory mapped-state allgather returned "
                f"{len(mapped_states)} ranks, expected {self.group_size}"
            )
        if any(mapped_states) and not all(mapped_states):
            raise RuntimeError("Inconsistent symmetric-memory mapped state across ranks")
        return mapped_states

    def _collective_allocation_metadata(self, comm_backend: CommBackend) -> None:
        local_metadata = {
            "group_rank": self.group_rank,
            "group_size": self.group_size,
            "buf_size": self.buf_size,
            "allocation_size": self.allocation_size,
            "signal_pad_offset": self.signal_pad_offset,
            "total_uc_size": getattr(self, "total_uc_size", 0),
            "has_multicast": bool(self.mc_ptr),
        }
        all_metadata = comm_backend.allgather(local_metadata)
        if len(all_metadata) != self.group_size:
            raise RuntimeError(
                "Symmetric-memory metadata allgather returned "
                f"{len(all_metadata)} ranks, expected {self.group_size}"
            )
        for rank, metadata in enumerate(all_metadata):
            expected = {**local_metadata, "group_rank": rank}
            if metadata != expected:
                raise RuntimeError(
                    "Inconsistent symmetric-memory allocation metadata across "
                    f"ranks: rank {rank} has {metadata!r}, expected {expected!r}"
                )

    def get_graph_visible_addresses(self) -> Dict[str, Any]:
        """Return the VA/layout state captured by graph-visible tensors."""
        return {
            "buf_size": self.buf_size,
            "group_size": self.group_size,
            "group_rank": self.group_rank,
            "device_idx": self.device_idx,
            "allocation_size": self.allocation_size,
            "signal_pad_offset": self.signal_pad_offset,
            "total_uc_size": getattr(self, "total_uc_size", 0),
            "uc_base_ptr": int(getattr(self, "uc_base_ptr", 0)),
            "uc_ptrs": list(self.uc_ptrs),
            "uc_ptrs_dev": int(self.uc_ptrs_dev),
            "signal_pads": list(self.signal_pads),
            "signal_pads_dev": int(self.signal_pads_dev),
            "mc_ptr": int(self.mc_ptr),
            "has_multicast": bool(self.mc_ptr),
        }

    def validate_graph_visible_addresses(
        self, expected: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate that graph-visible VAs and pointer arrays are stable."""
        expected = expected or self._graph_visible_addresses
        if expected is None:
            raise RuntimeError("Missing captured symmetric-memory address metadata")
        current = self.get_graph_visible_addresses()
        for key, expected_value in expected.items():
            if current.get(key) != expected_value:
                raise RuntimeError(
                    f"SymmDeviceMemory graph-visible address changed for {key}: "
                    f"{current.get(key)!r} != {expected_value!r}"
                )

    def detach_physical_keep_va(
        self, *, synchronize: bool = True, barrier: bool = True
    ) -> None:
        """Release UC/MC physical mappings while preserving graph-visible VAs."""
        self._validate_comm(self.comm_backend)
        mapped_states = self._collective_mapped_states(self.comm_backend)
        if not any(mapped_states):
            if barrier:
                self.comm_backend.barrier()
                self.comm_backend.barrier()
            return
        self._collective_allocation_metadata(self.comm_backend)
        self.validate_graph_visible_addresses()
        if synchronize:
            checkCudaErrors(cuda.cuCtxSynchronize())
        if barrier:
            self.comm_backend.barrier()

        if self.mc_handle:
            checkCudaErrors(cuda.cuMemUnmap(self.mc_ptr, self.allocation_size))
            checkCudaErrors(cuda.cuMemRelease(self.mc_handle))
            self.mc_handle = 0

        for peer, handle in enumerate(self.uc_handles):
            if handle:
                checkCudaErrors(
                    cuda.cuMemUnmap(self.uc_ptrs[peer], self.allocation_size)
                )
                checkCudaErrors(cuda.cuMemRelease(handle))
                self.uc_handles[peer] = 0
        self._mapped = False
        self._close_handle_exchanger()

        if barrier:
            self.comm_backend.barrier()

    def remap_physical_same_va(
        self,
        *,
        comm_backend: Optional[CommBackend] = None,
        synchronize: bool = True,
        barrier: bool = True,
        zero_local: bool = True,
    ) -> None:
        """Create fresh UC/MC backing and map it into the original VAs."""
        comm_backend = comm_backend or self.comm_backend
        self._validate_comm(comm_backend)
        mapped_states = self._collective_mapped_states(comm_backend)
        if comm_backend is not self.comm_backend and any(mapped_states):
            raise RuntimeError(
                "Cannot refresh symmetric-memory communicator while allocation "
                "is still mapped; call detach_physical_keep_va before checkpoint "
                "remap"
            )
        if all(mapped_states):
            if barrier:
                comm_backend.barrier()
                comm_backend.barrier()
            return
        self._collective_allocation_metadata(comm_backend)
        self.validate_graph_visible_addresses()
        if synchronize:
            checkCudaErrors(cuda.cuCtxSynchronize())
        if barrier:
            comm_backend.barrier()

        enable_multicast = bool(self.mc_ptr)
        fresh: Optional[SymmDeviceMemory] = None
        mapped_uc_peers: List[int] = []
        mapped_mc = False
        try:
            fresh = SymmDeviceMemory(
                buf_size=self.buf_size,
                group_size=self.group_size,
                group_rank=self.group_rank,
                device_idx=self.device_idx,
                comm_backend_for_handle_transfer=comm_backend,
                enable_multicast=enable_multicast,
                allocate_signal_pads=False,
            )
            if fresh.allocation_size != self.allocation_size:
                raise RuntimeError(
                    "Restored symmetric-memory allocation size changed: "
                    f"{fresh.allocation_size} != {self.allocation_size}"
                )

            for peer_ptr in fresh.uc_ptrs:
                checkCudaErrors(cuda.cuMemUnmap(peer_ptr, fresh.allocation_size))
            checkCudaErrors(
                cuda.cuMemAddressFree(fresh.uc_base_ptr, fresh.total_uc_size)
            )
            fresh.uc_ptrs = []
            fresh.uc_base_ptr = 0
            fresh.total_uc_size = 0

            for peer, handle in enumerate(fresh.uc_handles):
                checkCudaErrors(
                    cuda.cuMemMap(
                        self.uc_ptrs[peer], self.allocation_size, 0, handle, 0
                    )
                )
                mapped_uc_peers.append(peer)
            checkCudaErrors(
                cuda.cuMemSetAccess(
                    self.uc_base_ptr,
                    self.total_uc_size,
                    [self._get_mem_access_desc()],
                    1,
                )
            )

            if enable_multicast:
                checkCudaErrors(cuda.cuMemUnmap(fresh.mc_ptr, fresh.allocation_size))
                checkCudaErrors(
                    cuda.cuMemAddressFree(fresh.mc_ptr, fresh.allocation_size)
                )
                fresh.mc_ptr = 0
                checkCudaErrors(
                    cuda.cuMemMap(
                        self.mc_ptr,
                        self.allocation_size,
                        0,
                        fresh.mc_handle,
                        0,
                    )
                )
                mapped_mc = True
                checkCudaErrors(
                    cuda.cuMemSetAccess(
                        self.mc_ptr,
                        self.allocation_size,
                        [self._get_mem_access_desc()],
                        1,
                    )
                )

            if fresh.uc_ptrs_dev:
                checkCudaErrors(cuda.cuMemFree(fresh.uc_ptrs_dev))
                fresh.uc_ptrs_dev = 0

            if zero_local:
                checkCudaErrors(
                    cuda.cuMemsetD8(
                        self.uc_ptrs[self.group_rank], 0, self.allocation_size
                    )
                )

            self.validate_graph_visible_addresses()
            self._close_handle_exchanger()
            self.uc_handles = fresh.uc_handles
            self.mc_handle = fresh.mc_handle
            self.comm_backend = comm_backend
            self._exchanger = fresh._exchanger
            self._mapped = True

            fresh.uc_handles = []
            fresh.uc_ptrs = []
            fresh.uc_base_ptr = 0
            fresh.total_uc_size = 0
            fresh.mc_handle = 0
            fresh.mc_ptr = 0
            fresh._exchanger = None
        except Exception as exc:
            cleanup_errors = []
            if mapped_mc:
                try:
                    checkCudaErrors(cuda.cuMemUnmap(self.mc_ptr, self.allocation_size))
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if enable_multicast and not fresh.mc_ptr and fresh.mc_handle:
                try:
                    checkCudaErrors(cuda.cuMemRelease(fresh.mc_handle))
                    fresh.mc_handle = 0
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            for peer in reversed(mapped_uc_peers):
                try:
                    checkCudaErrors(
                        cuda.cuMemUnmap(self.uc_ptrs[peer], self.allocation_size)
                    )
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise RuntimeError(
                    "Failed to roll back partial symmetric-memory remap after "
                    f"{exc!r}: {cleanup_errors!r}"
                ) from exc
            raise
        finally:
            if fresh is not None:
                del fresh

        if barrier:
            self.comm_backend.barrier()

    def get_signal_pad_ptrs_host(self) -> List[int]:
        """Get the raw array of signal pad pointers to all ranks (including self)"""
        return self.signal_pads

    def get_buffer_ptrs_host(self) -> List[int]:
        """Get the raw array of unicast pointers to all ranks (including self)"""
        return self.uc_ptrs

    def get_signal_pad_ptrs_dev(self) -> int:
        """Get the raw array of signal pad pointers to all ranks (including self)"""
        return self.signal_pads_dev

    def get_buffer_ptrs_dev(self) -> int:
        """Get the raw array of unicast pointers to all ranks (including self)"""
        return self.uc_ptrs_dev

    def get_unicast_ptr(self, rank: int) -> int:
        """Get the raw unicast pointer to a given rank"""
        if rank >= len(self.uc_ptrs):
            raise ValueError(f"Rank {rank} out of range (0-{len(self.uc_ptrs) - 1})")

        data_ptr = self.uc_ptrs[rank]
        # Note: In C++, this would call tensorrt_llm::common::registerMcastDevMemBuffer
        # For Python port, we skip this registration for now
        return data_ptr

    def get_multicast_ptr(self) -> int:
        """Get the raw multicast pointer"""
        # Note: In C++, this would call tensorrt_llm::common::registerMcastDevMemBuffer
        # For Python port, we skip this registration for now
        return int(self.mc_ptr)

    def get_rank(self) -> int:
        """Get the rank of this device in the group"""
        return self.group_rank

    def get_world_size(self) -> int:
        """Get the total number of devices in the group"""
        return self.group_size

    def get_allocation_size(self) -> int:
        """Get the total allocation size (including signal pad)"""
        return self.allocation_size

    def get_usable_buffer_size(self) -> int:
        """Get the usable buffer size (excluding signal pad)"""
        return self.allocation_size - self.SIGNAL_PAD_SIZE

    def _alloc_mn_mcast_mem(self, buf_size: int, enable_multicast: bool):
        """Allocate multi-node multicast memory using MNNVL"""
        self._verify_cuda_context()

        # Compute allocation size and get allocation properties
        allocation_prop, mc_prop = self._get_allocation_prop(buf_size)

        # Allocate, exchange, and map unicast buffers
        self._allocate_unicast_buffers(allocation_prop)

        # Setup multicast object, exchange handles, map and bind memory
        if enable_multicast:
            self._setup_multicast(mc_prop)

    def _verify_cuda_context(self):
        """Verify CUDA context is set to the correct device."""
        try:
            current_device = checkCudaErrors(cuda.cuCtxGetDevice())
            if int(current_device) != self.device_idx:
                logger.warning(
                    "CUDA context device mismatch! Current: %s, Expected: %s",
                    current_device,
                    self.device_idx,
                )
        except Exception as e:
            logger.warning("Error checking CUDA context: %s", e)

    def _get_allocation_prop(self, buf_size: int):
        """Compute allocation size and return allocation/multicast properties."""
        allocation_prop = cuda.CUmemAllocationProp()
        allocation_prop.requestedHandleTypes = self._exchanger.handle_type
        allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        allocation_prop.location = cuda.CUmemLocation()
        allocation_prop.location.type = (
            cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        )
        allocation_prop.location.id = self.device_idx
        allocation_prop.allocFlags.gpuDirectRDMACapable = 1

        # Get allocation granularity
        alloc_granularity = checkCudaErrors(
            cuda.cuMemGetAllocationGranularity(
                allocation_prop,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )

        self.allocation_size = round_up(
            buf_size + self.SIGNAL_PAD_SIZE, alloc_granularity
        )

        # Set up multicast properties
        mc_prop = cuda.CUmulticastObjectProp()
        mc_prop.numDevices = self.group_size
        mc_prop.size = self.allocation_size
        mc_prop.handleTypes = self._exchanger.handle_type

        # Get multicast granularity and adjust allocation size
        self._mc_granularity = checkCudaErrors(
            cuda.cuMulticastGetGranularity(
                mc_prop,
                cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
            )
        )
        self.allocation_size = round_up(self.allocation_size, self._mc_granularity)

        return allocation_prop, mc_prop

    def _allocate_unicast_buffers(self, allocation_prop):
        """Allocate local UC memory, exchange handles with peers, and map memory."""
        # Initialize UC handles list
        self.uc_handles = [0] * self.group_size

        # Allocate local GPU memory
        self.uc_handles[self.group_rank] = checkCudaErrors(
            cuda.cuMemCreate(self.allocation_size, allocation_prop, 0)
        )

        # Export local handle to shareable handle
        local_shareable_uc_handle = checkCudaErrors(
            cuda.cuMemExportToShareableHandle(
                self.uc_handles[self.group_rank],
                self._exchanger.handle_type,
                0,
            )
        )

        all_shareable_uc_handles = []
        try:
            # All-gather shareable handles
            all_shareable_uc_handles = self._exchanger.allgather(
                local_shareable_uc_handle
            )
            cuda.cuCtxSynchronize()

            # Import remote handles
            for p in range(self.group_size):
                if p != self.group_rank:
                    self.uc_handles[p] = checkCudaErrors(
                        cuda.cuMemImportFromShareableHandle(
                            all_shareable_uc_handles[p],
                            self._exchanger.handle_type,
                        )
                    )
        finally:
            self._exchanger.cleanup(local_shareable_uc_handle)
            for handle in all_shareable_uc_handles:
                self._exchanger.cleanup(handle)

        # Reserve address space for UC pointers
        self.uc_ptrs = [0] * self.group_size
        total_uc_size = self.allocation_size * self.group_size
        self.total_uc_size = total_uc_size
        uc_base_ptr = checkCudaErrors(
            cuda.cuMemAddressReserve(total_uc_size, self._mc_granularity, 0, 0)
        )
        self.uc_base_ptr = uc_base_ptr

        # Map UC memory
        for i in range(self.group_size):
            offset = self.allocation_size * i
            self.uc_ptrs[i] = int(uc_base_ptr) + offset
            checkCudaErrors(
                cuda.cuMemMap(
                    self.uc_ptrs[i], self.allocation_size, 0, self.uc_handles[i], 0
                )
            )

        # Set memory access permissions for UC
        access_desc = self._get_mem_access_desc()
        checkCudaErrors(
            cuda.cuMemSetAccess(uc_base_ptr, total_uc_size, [access_desc], 1)
        )

    def _setup_multicast(self, mc_prop):
        """Create multicast object, exchange handle, map memory, and bind."""
        # Rank 0 creates the multicast object
        if self.group_rank == 0:
            self.mc_handle = checkCudaErrors(cuda.cuMulticastCreate(mc_prop))
            shareable_mc_handle = checkCudaErrors(
                cuda.cuMemExportToShareableHandle(
                    self.mc_handle,
                    self._exchanger.handle_type,
                    0,
                )
            )
        else:
            shareable_mc_handle = None

        try:
            # Broadcast multicast handle from rank 0
            shareable_mc_handle = self._exchanger.broadcast(shareable_mc_handle, root=0)
            cuda.cuCtxSynchronize()

            # Import multicast handle for non-root ranks
            if self.group_rank != 0:
                self.mc_handle = checkCudaErrors(
                    cuda.cuMemImportFromShareableHandle(
                        shareable_mc_handle,
                        self._exchanger.handle_type,
                    )
                )
        finally:
            if shareable_mc_handle is not None:
                self._exchanger.cleanup(shareable_mc_handle)

        # Add device to multicast
        checkCudaErrors(cuda.cuMulticastAddDevice(self.mc_handle, self.device_idx))

        # Reserve and map MC pointer
        self.mc_ptr = checkCudaErrors(
            cuda.cuMemAddressReserve(self.allocation_size, self._mc_granularity, 0, 0)
        )
        checkCudaErrors(
            cuda.cuMemMap(self.mc_ptr, self.allocation_size, 0, self.mc_handle, 0)
        )
        access_desc = self._get_mem_access_desc()
        checkCudaErrors(
            cuda.cuMemSetAccess(self.mc_ptr, self.allocation_size, [access_desc], 1)
        )

        # Bind local memory to multicast
        checkCudaErrors(
            cuda.cuMulticastBindMem(
                self.mc_handle,
                0,  # mcOffset
                self.uc_handles[self.group_rank],
                0,  # memOffset
                self.allocation_size,
                0,  # flags
            )
        )

    def _get_mem_access_desc(self):
        """Create memory access descriptor for this device."""
        access_desc = cuda.CUmemAccessDesc()
        access_desc.location = cuda.CUmemLocation()
        access_desc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = self.device_idx
        access_desc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        return access_desc

    def lamport_initialize(self, rank: int, dtype: torch.dtype):
        if dtype == torch.bfloat16 or dtype == torch.float16:
            neg_zero = 0x8000
            dsize = 2
            memset_func = cuda.cuMemsetD16
        elif dtype == torch.float32:
            neg_zero = 0x80000000
            dsize = 4
            memset_func = cuda.cuMemsetD32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Calculate number of elements that fit in allocation_size; We don't want to include the signal pad.
        num_elements = (self.allocation_size - self.SIGNAL_PAD_SIZE) // dsize

        checkCudaErrors(
            memset_func(int(self.uc_ptrs[self.group_rank]), neg_zero, num_elements)
        )


class McastGPUBuffer:
    """
    Wrapper class for SymmDeviceMemory to facilitate PyTorch tensor creation.
    It manages a buffer accessible via unicast or multicast for multi-node communication.

    Python port of McastGPUBuffer from TensorRT-LLM
    """

    def __init__(
        self,
        buf_size: int,
        group_size: int,
        group_rank: int,
        device: torch.device,
        comm_backend_for_handle_transfer: Optional[CommBackend] = None,
    ):
        """
        Constructor for McastGpuBuffer.

        Args:
            buf_size: The requested size of the buffer in bytes. The actual usable size may differ due to alignment requirements.
            group_size: The number of ranks in the communication group
            group_rank: The rank of the local process within the group
            device: The CUDA device for buffer allocation
            mn_nvlink: Flag indicating if multi-node NVLink is used
            comm_backend_for_handle_transfer: Communication backend for handle transfer
        """
        self.mcast_device_memory = SymmDeviceMemory(
            buf_size,
            group_size,
            group_rank,
            device.index,
            comm_backend_for_handle_transfer,
        )
        # Update buf_size to reflect the actual usable buffer size after allocation
        self.buf_size = self.mcast_device_memory.get_usable_buffer_size()
        self.local_device = device

    def lamport_initialize(self, rank: int, dtype: torch.dtype):
        self.mcast_device_memory.lamport_initialize(rank, dtype)

    def get_multicast_buffer(
        self, sizes: tuple, dtype: torch.dtype, storage_offset: int = 0
    ) -> torch.Tensor:
        """
        Returns a PyTorch tensor view of the multicast buffer portion.

        Args:
            sizes: The desired shape (dimensions) of the tensor
            dtype: The data type of the tensor elements
            storage_offset: The offset in elements from the start of the buffer

        Returns:
            A PyTorch tensor wrapping the multicast buffer section
        """

        # FIXME: Is this needed? As the behavior of reading from mc_ptr is undefined.
        raise NotImplementedError("Not implemented yet")

    def get_unicast_buffer(
        self, sizes: tuple, dtype: torch.dtype, storage_offset: int = 0
    ) -> torch.Tensor:
        """
        Returns a PyTorch tensor view of the unicast buffer portion.
        """

        # TODO: How can I warp a raw pointer to a tensor in python level?
        raise NotImplementedError("Not implemented yet")

    def get_multicast_ptr(self) -> int:
        """Get the raw multicast pointer"""
        return self.mcast_device_memory.get_multicast_ptr()

    def get_unicast_ptr(self, rank: int) -> int:
        """Get the raw unicast pointer to a given rank"""
        return self.mcast_device_memory.get_unicast_ptr(rank)

    def get_buffer_ptrs_dev(self) -> int:
        """Get the buffer pointers device array"""
        return self.mcast_device_memory.get_buffer_ptrs_dev()

    @property
    def buffer_size(self) -> int:
        """Return the usable local buffer size, excluding signal padding."""
        return self.buf_size

    @property
    def buffer_ptrs(self) -> List[int]:
        """Return host unicast pointers for all ranks."""
        return self.mcast_device_memory.get_buffer_ptrs_host()

    @property
    def buffer_ptrs_dev(self) -> int:
        """Return the device pointer array of unicast pointers."""
        return self.mcast_device_memory.get_buffer_ptrs_dev()

    @property
    def multicast_ptr(self) -> int:
        """Return the multicast pointer."""
        return self.mcast_device_memory.get_multicast_ptr()

    def get_graph_visible_addresses(self) -> Dict[str, Any]:
        """Return graph-visible pointer metadata for this buffer."""
        return self.mcast_device_memory.get_graph_visible_addresses()

    def validate_graph_visible_addresses(self) -> None:
        """Validate that graph-visible buffer pointers are stable."""
        self.mcast_device_memory.validate_graph_visible_addresses()

    def detach_physical_keep_va(
        self, *, synchronize: bool = True, barrier: bool = True
    ) -> None:
        """Detach physical backing while preserving graph-visible VAs."""
        self.mcast_device_memory.detach_physical_keep_va(
            synchronize=synchronize, barrier=barrier
        )

    def remap_physical_same_va(
        self,
        *,
        comm_backend: Optional[CommBackend] = None,
        synchronize: bool = True,
        barrier: bool = True,
        zero_local: bool = True,
    ) -> None:
        """Remap physical backing at the original graph-visible VAs."""
        self.mcast_device_memory.remap_physical_same_va(
            comm_backend=comm_backend,
            synchronize=synchronize,
            barrier=barrier,
            zero_local=zero_local,
        )
