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
    """
    Create a PyTorch tensor from a CUDA memory pointer using DLPack.

    Args:
        ptr: CUDA memory pointer address as integer
        shape: Desired tensor shape
        dtype: PyTorch data type
        device_id: CUDA device ID

    Returns:
        PyTorch tensor that wraps the CUDA memory
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

        print(f"DEBUG: Memory access test PASSED for ptr=0x{ptr:x}")
        return True
    except Exception as e:
        print(f"DEBUG: Memory access test FAILED for ptr=0x{ptr:x}: {e}")
        return False


def alloc_and_copy_to_cuda(host_ptr_array: List[int]) -> int:
    """
    A helper function that allocates memory on cuda and copies the data from the host to the device.
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
        """Broadcast a Python object from root to all ranks."""
        object_list = [data]
        self._dist.broadcast_object_list(object_list, src=root, group=self._group)
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
            MnnvlMemory.close_mnnvl_memory(self.ptr)

    def as_torch_strided_tensor(self, dtype):
        num_segments = MnnvlMemory.comm.Get_size()
        return pack_strided_memory(
            self.ptr,
            self.segment_size,
            self.rank_stride,
            num_segments,
            dtype,
            MnnvlMemory.dev_id,
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
        MnnvlMemory.config = config or MnnvlConfig(comm_backend=MPIBackend())  # type: ignore[attr-defined]
        comm = config.comm_backend.Split(
            mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank
        )
        MnnvlMemory.comm = comm  # type: ignore[assignment]

    @staticmethod
    def get_comm(mapping: Mapping):
        if MnnvlMemory.comm is not None:
            return MnnvlMemory.comm
        comm = MpiComm().Split(
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
        comm_rank = comm.Get_rank()
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

        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        allocated_mem_handle = checkCudaErrors(
            cuda.cuMemCreate(aligned_size, allocation_prop, flags=0)
        )
        exported_fabric_handle = checkCudaErrors(
            cuda.cuMemExportToShareableHandle(
                allocated_mem_handle, allocation_prop.requestedHandleTypes, 0
            )
        )
        if (
            allocation_prop.requestedHandleTypes
            == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        ):
            all_handles_data = comm.allgather(exported_fabric_handle.data)
        else:
            all_handles_data = comm.allgather(exported_fabric_handle)
            all_pids = comm.allgather(os.getpid())
            libc = ctypes.CDLL(None, use_errno=True)
            syscall = libc.syscall
            SYS_pidfd_open = 434
            SYS_pidfd_getfd = 438
            pidfds = []
            for pid in all_pids:
                pidfd = syscall(SYS_pidfd_open, pid, 0)
                if pidfd < 0:
                    err = ctypes.get_errno()
                    raise RuntimeError(
                        f"pidfd_open({pid}) failed with errno {err}: {os.strerror(err)}"
                    )
                pidfds.append(pidfd)

            remote_fds = []
            for pidfd, fd in zip(pidfds, all_handles_data, strict=True):
                remote_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0)
                if remote_fd < 0:
                    err = ctypes.get_errno()
                    error_msg = f"pidfd_getfd(pidfd={pidfd}, fd={fd}) failed with errno {err}: {os.strerror(err)}."
                    if err == 1:  # EPERM
                        error_msg += (
                            " Permission denied. If running in a container, try adding --cap-add=SYS_PTRACE "
                            "to your docker run command."
                        )
                    else:
                        error_msg += (
                            " This may be due to kernel version (requires Linux 5.6+)."
                        )
                    raise RuntimeError(error_msg)
                remote_fds.append(remote_fd)

            all_handles_data = remote_fds
        # all_handles_data like b'\x00\x00\x00 \x00\x00\x00\x00\x8f\xec\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'  # noqa: E501
        # can use buf = memoryview(data) to import if using plain buffer for data.

        madesc = cuda.CUmemAccessDesc()
        madesc.location = allocation_prop.location
        madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        mem_handles = [None] * comm_size

        for i, remote_handle_data in enumerate(all_handles_data):
            rank_ptr = (
                MnnvlMemory.current_start_address
                + MnnvlMemory.current_rank_stride * i
                + MnnvlMemory.current_mem_offset
            )
            if i == comm_rank:
                # Local memory mapping
                mem_handles[i] = allocated_mem_handle
                checkCudaErrors(
                    cuda.cuMemMap(rank_ptr, aligned_size, 0, allocated_mem_handle, 0)
                )
            else:
                # Fabric memory mapping
                imported_mem_handle = checkCudaErrors(
                    cuda.cuMemImportFromShareableHandle(
                        remote_handle_data, allocation_prop.requestedHandleTypes
                    )
                )
                mem_handles[i] = imported_mem_handle
                checkCudaErrors(
                    cuda.cuMemMap(rank_ptr, aligned_size, 0, imported_mem_handle, 0)
                )

            checkCudaErrors(cuda.cuMemSetAccess(rank_ptr, aligned_size, [madesc], 1))

        ptr = MnnvlMemory.current_start_address + MnnvlMemory.current_mem_offset
        stride = MnnvlMemory.current_rank_stride
        MnnvlMemory.allocated_map[ptr] = (
            mapping,
            aligned_size,
            mem_handles,
            MnnvlMemory.current_start_address,
            MnnvlMemory.current_rank_stride,
            MnnvlMemory.current_mem_offset,
        )
        MnnvlMemory.address_refcnt[MnnvlMemory.current_start_address] = (
            MnnvlMemory.address_refcnt.get(MnnvlMemory.current_start_address, 0) + 1
        )

        MnnvlMemory.current_mem_offset += aligned_size
        return ptr, stride

    @staticmethod
    def close_mnnvl_memory(ptr: int):
        (
            mapping,
            aligned_size,
            mem_handles,
            start_address,
            rank_stride,
            address_offset,
        ) = MnnvlMemory.allocated_map.pop(ptr)
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        for i in range(comm_size):
            rank_ptr = start_address + i * rank_stride + address_offset
            checkCudaErrors(cuda.cuMemUnmap(rank_ptr, aligned_size))
            checkCudaErrors(cuda.cuMemRelease(mem_handles[i]))
        MnnvlMemory.address_refcnt[start_address] -= 1

        if MnnvlMemory.address_refcnt[start_address] == 0:
            MnnvlMemory.address_refcnt.pop(start_address)
            device_ptr = cuda.CUdeviceptr(start_address)
            checkCudaErrors(cuda.cuMemAddressFree(device_ptr, comm_size * rank_stride))
            if start_address == MnnvlMemory.current_start_address:
                MnnvlMemory.current_start_address = 0
                MnnvlMemory.current_rank_stride = 0
                MnnvlMemory.current_mem_offset = 0

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
            opId = random.randint(0, 2**64 - 1)
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
        if (
            fabric_info.state >= pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
            and fabric_info.clusterUuid[0] != 0
        ):
            return True
        return False
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
        if is_mnnvl_fabric_supported(device_idx):
            self._exchanger: HandleExchanger = FabricHandleExchanger(
                self.comm_backend, self.group_rank, self.group_size
            )
        else:
            self._exchanger = PosixFDHandleExchanger(
                self.comm_backend, self.group_rank, self.group_size
            )
        self._alloc_mn_mcast_mem(buf_size, enable_multicast)

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

    def __del__(self):
        """Destructor - cleanup allocated memory"""

        if hasattr(self, "_exchanger"):
            self._exchanger.close()

        # Skip cleanup during Python finalization to avoid segfaults
        # Especially cause the CUDA context could be destroyed at this point.
        if sys.is_finalizing():
            return

        # Verify CUDA context is still valid
        try:
            cuda.cuCtxGetCurrent()
        except Exception as e:
            print(f"Destructor: CUDA context invalid, skipping cleanup: {e}")
            return

        # Free device pointers
        if self.signal_pads_dev:
            checkCudaErrors(cuda.cuMemFree(self.signal_pads_dev))
        if self.uc_ptrs_dev:
            checkCudaErrors(cuda.cuMemFree(self.uc_ptrs_dev))

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
                        print(
                            f"Destructor: Failed to release UC handle for rank {rank}: {e}"
                        )

            # Free the UC address space
            if hasattr(self, "uc_base_ptr") and self.uc_base_ptr:
                checkCudaErrors(
                    cuda.cuMemAddressFree(self.uc_base_ptr, self.total_uc_size)
                )

        # Release MC handle
        if hasattr(self, "mc_handle") and self.mc_handle and self.mc_handle != 0:
            try:
                checkCudaErrors(cuda.cuMemUnmap(self.mc_ptr, self.allocation_size))
                checkCudaErrors(
                    cuda.cuMemAddressFree(self.mc_ptr, self.allocation_size)
                )
                checkCudaErrors(cuda.cuMemRelease(self.mc_handle))
            except Exception as e:
                print(f"Destructor: Failed to release MC handle: {e}")

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
                print(
                    f"CUDA context device mismatch! Current: {current_device}, Expected: {self.device_idx}"
                )
        except Exception as e:
            print(f"Error checking CUDA context: {e}")

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

        # All-gather shareable handles
        all_shareable_uc_handles = self._exchanger.allgather(local_shareable_uc_handle)
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
                self._exchanger.cleanup(all_shareable_uc_handles[p])

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
