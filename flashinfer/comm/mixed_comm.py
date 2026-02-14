"""
Copyright (c) 2026 by FlashInfer team.

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

import ctypes
import enum
import functools
import math
import os
import socket
import struct
from typing import Tuple

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

from .allreduce import (
    create_allreduce_fusion_workspace,
    allreduce_fusion,
    AllReduceFusionWorkspace,
)
from .mnnvl import TorchDistBackend
from ..cuda_utils import checkCudaErrors
from ..jit import env as jit_env
from ..jit.comm import gen_mixed_comm_module


@functools.cache
def get_mixed_comm_module():
    # Try to find libnvshmem_host.so first, fallback to libnvshmem_host.so.3
    lib_dirs = jit_env.get_nvshmem_lib_dirs()
    lib_path = None

    lib_names = ["libnvshmem_host.so", "libnvshmem_host.so.3"]
    for lib_dir in lib_dirs:
        for lib_name in lib_names:
            candidate_path = lib_dir / lib_name
            if candidate_path.exists():
                lib_path = candidate_path
                break
        if lib_path is not None:
            break

    if lib_path is None:
        raise FileNotFoundError(
            f"Could not find libnvshmem_host.so or libnvshmem_host.so.3 in {lib_dirs}"
        )

    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    module = gen_mixed_comm_module().build_and_load()

    return module


def get_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def round_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def trtllm_allreduce(x_in: torch.Tensor, workspace: AllReduceFusionWorkspace):
    x_out = torch.empty_like(x_in)
    allreduce_fusion(
        x_in,
        workspace,
        pattern=0,
        output=x_out,
        fp32_acc=True,
    )
    return x_out


class MixedCommMode(enum.IntEnum):
    # Run a fused kernel (correspond to OPT_WAITS, OPT_BYTES, MIXED in fused_comm.cuh)
    FUSED_OPT_WAITS = 0
    FUSED_OPT_BYTES = enum.auto()
    FUSED_MIXED = enum.auto()
    # Run different NCCL kernels for TP and DP
    NCCL_TP_DP = enum.auto()
    # Run a NCCL all-reduce kernel with preprocessing or postprocessing
    NCCL_AR = enum.auto()
    # Run different NCCL kernels for intra-node and inter-node communications
    NCCL_LOCAL_INTER = enum.auto()
    # Run a TRT-LLM kernel for intra-node communication and a NCCL kernel for inter-node communication
    TRTLLM_LOCAL_INTER = enum.auto()


class ParallelInfo:
    def __init__(
        self,
        world_rank: int,
        world_size: int,
        local_rank: int,
        local_size: int,
        node_id: int,
        num_nodes: int,
        local_tp_size: int | None,
        local_dp_size: int | None,
        inter_tp_size: int | None,
        inter_dp_size: int | None,
    ):
        self._world_rank = world_rank
        self._world_size = world_size
        self._local_rank = local_rank
        self._local_size = local_size
        self._node_id = node_id
        self._num_nodes = num_nodes
        assert self.world_rank == self.node_id * self.local_size + self.local_rank
        assert self.world_size == self.num_nodes * self.local_size
        if local_tp_size is None and local_dp_size is None:
            self._local_tp_size = self.local_size
            self._local_dp_size = 1
        elif local_tp_size is None:
            assert self.local_size % local_dp_size == 0
            self._local_tp_size = self.local_size // local_dp_size
            self._local_dp_size = local_dp_size
        elif local_dp_size is None:
            assert self.local_size % local_tp_size == 0
            self._local_tp_size = local_tp_size
            self._local_dp_size = self.local_size // local_tp_size
        else:
            assert self.local_size == local_tp_size * local_dp_size
            self._local_tp_size = local_tp_size
            self._local_dp_size = local_dp_size
        if inter_tp_size is None and inter_dp_size is None:
            self._inter_tp_size = self.num_nodes
            self._inter_dp_size = 1
        elif inter_tp_size is None:
            assert self.num_nodes % inter_dp_size == 0
            self._inter_tp_size = self.num_nodes // inter_dp_size
            self._inter_dp_size = inter_dp_size
        elif inter_dp_size is None:
            assert self.num_nodes % inter_tp_size == 0
            self._inter_tp_size = inter_tp_size
            self._inter_dp_size = self.num_nodes // inter_tp_size
        else:
            assert self.num_nodes == inter_tp_size * inter_dp_size
            self._inter_tp_size = inter_tp_size
            self._inter_dp_size = inter_dp_size
        self._local_tp_rank = self.local_rank % self.local_tp_size
        self._local_dp_rank = self.local_rank // self.local_tp_size
        self._inter_tp_rank = self.node_id % self.inter_tp_size
        self._inter_dp_rank = self.node_id // self.inter_tp_size

    @property
    def world_rank(self):
        return self._world_rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def local_size(self):
        return self._local_size

    @property
    def node_id(self):
        return self._node_id

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def local_tp_rank(self):
        return self._local_tp_rank

    @property
    def local_tp_size(self):
        return self._local_tp_size

    @property
    def local_dp_rank(self):
        return self._local_dp_rank

    @property
    def local_dp_size(self):
        return self._local_dp_size

    @property
    def inter_tp_rank(self):
        return self._inter_tp_rank

    @property
    def inter_tp_size(self):
        return self._inter_tp_size

    @property
    def inter_dp_rank(self):
        return self._inter_dp_rank

    @property
    def inter_dp_size(self):
        return self._inter_dp_size

    @property
    def tp_rank(self):
        return self.local_tp_rank + self.inter_tp_rank * self.local_tp_size

    @property
    def tp_size(self):
        return self.local_tp_size * self.inter_tp_size

    @property
    def dp_rank(self):
        return self.local_dp_rank + self.inter_dp_rank * self.local_dp_size

    @property
    def dp_size(self):
        return self.local_dp_size * self.inter_dp_size

    def get_local_comm_group(self):
        world_rank_list_all = [
            list(range(node_id * self.local_size, (node_id + 1) * self.local_size))
            for node_id in range(self.num_nodes)
        ]
        local_comm_group_list = [
            torch.distributed.new_group(world_rank_list)
            for world_rank_list in world_rank_list_all
        ]
        return local_comm_group_list[self.node_id]

    def get_inter_comm_group(self):
        world_rank_list_all = [
            list(range(local_rank, self.world_size, self.local_size))
            for local_rank in range(self.local_size)
        ]
        inter_comm_group_list = [
            torch.distributed.new_group(world_rank_list)
            for world_rank_list in world_rank_list_all
        ]
        return inter_comm_group_list[self.local_rank]

    def get_tp_comm_group(self):
        world_rank_list_all = []
        for inter_dp_rank in range(self.inter_dp_size):
            for local_dp_rank in range(self.local_dp_size):
                local_rank_list = self.get_local_tp_group_local_ranks(local_dp_rank)
                world_rank_list = []
                for inter_tp_rank in range(self.inter_tp_size):
                    node_id = inter_dp_rank * self.inter_tp_size + inter_tp_rank
                    world_rank_list += [
                        val + node_id * self.local_size for val in local_rank_list
                    ]
                world_rank_list_all.append(world_rank_list)
        tp_comm_group_list = [
            torch.distributed.new_group(world_rank_list)
            for world_rank_list in world_rank_list_all
        ]
        return tp_comm_group_list[self.dp_rank]

    def get_dp_comm_group(self):
        world_rank_list_all = []
        for inter_tp_rank in range(self.inter_tp_size):
            for local_tp_rank in range(self.local_tp_size):
                local_rank_list = self.get_local_dp_group_local_ranks(local_tp_rank)
                world_rank_list = []
                for inter_dp_rank in range(self.inter_dp_size):
                    node_id = inter_dp_rank * self.inter_tp_size + inter_tp_rank
                    world_rank_list += [
                        val + node_id * self.local_size for val in local_rank_list
                    ]
                world_rank_list_all.append(world_rank_list)
        dp_comm_group_list = [
            torch.distributed.new_group(world_rank_list)
            for world_rank_list in world_rank_list_all
        ]
        return dp_comm_group_list[self.tp_rank]

    def get_local_full_group_local_ranks(self):
        return list(range(self.local_size))

    def get_local_tp_group_local_ranks(self, local_dp_rank: int | None = None):
        if local_dp_rank is None:
            local_dp_rank = self.local_dp_rank
        return list(
            range(
                local_dp_rank * self.local_tp_size,
                (local_dp_rank + 1) * self.local_tp_size,
            )
        )

    def get_local_dp_group_local_ranks(self, local_tp_rank: int | None = None):
        if local_tp_rank is None:
            local_tp_rank = self.local_tp_rank
        return list(range(local_tp_rank, self.local_size, self.local_tp_size))


class MixedComm:
    """
    An implementation for the combinations of all-reduce + all-gather and reduce-scatter + all-reduce.
    The fused kernels use virtual memory for intra-node communication and nvshmem for inter-node communication.
    Currently, only float16 and bfloat16 data types are supported.
    Note: An active torch.distributed process group should be initialized before creating an instance of this class.

    Args:
        world_rank (int): The world rank of the current process (local_rank + node_id * local_size).
        world_size (int): The total number of processes in the distributed group (num_nodes * local_size).
        local_rank (int): The local rank of the current process (local_tp_rank + local_dp_rank * local_tp_size).
        local_size (int): The total number of processes in the current node (local_dp_size * local_tp_size).
        node_id (int): The index of the current node (inter_tp_rank + inter_dp_rank * inter_tp_size).
        num_nodes (int): The total number of nodes in the distributed group (inter_dp_size * inter_tp_size).
        local_tp_size (int | None): TP size in the intra-node group. Use the default value if not provided.
        local_dp_size (int | None): DP size in the intra-node group. Use the default value if not provided.
        inter_tp_size (int | None): TP size in the inter-node group. Use the default value if not provided.
        inter_dp_size (int | None): DP size in the inter-node group. Use the default value if not provided.
        max_local_bs (int): The maximum local batch size.
        hidden_size (int): The hidden size.
        dtype (torch.dtype): The data type.
        device (torch.device): The device on which the tensors are located.
        should_init_nvshmem (bool, optional): Whether to initialize nvshmem. The default value is True.
        maybe_use_trtllm_comm (bool, optional): Whether to use kernels from TRT-LLM. The default value is False.
    Raises:
        RuntimeError: If nvshmem fails to initialize.
    """

    def __init__(
        self,
        world_rank: int,
        world_size: int,
        local_rank: int,
        local_size: int,
        node_id: int,
        num_nodes: int,
        local_tp_size: int | None,
        local_dp_size: int | None,
        inter_tp_size: int | None,
        inter_dp_size: int | None,
        max_local_bs: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        should_init_nvshmem: bool = True,
        maybe_use_trtllm_comm: bool = False,
    ):
        assert torch.distributed.is_initialized()
        assert local_size > 1
        access_bytes = 16
        self.para_info = ParallelInfo(
            world_rank=world_rank,
            world_size=world_size,
            local_rank=local_rank,
            local_size=local_size,
            node_id=node_id,
            num_nodes=num_nodes,
            local_tp_size=local_tp_size,
            local_dp_size=local_dp_size,
            inter_tp_size=inter_tp_size,
            inter_dp_size=inter_dp_size,
        )
        self.max_local_bs = max_local_bs
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.should_init_nvshmem = should_init_nvshmem
        self.local_comm_group = self.para_info.get_local_comm_group()
        self.inter_comm_group = (
            self.para_info.get_inter_comm_group()
            if self.para_info.num_nodes > 1
            else None
        )
        self.tp_comm_group = (
            self.para_info.get_tp_comm_group() if self.para_info.tp_size > 1 else None
        )
        self.dp_comm_group = (
            self.para_info.get_dp_comm_group() if self.para_info.dp_size > 1 else None
        )
        self.grid_size = torch.cuda.get_device_properties(device).multi_processor_count

        max_local_bytes = round_up(
            max_local_bs * hidden_size * get_element_size(dtype),
            access_bytes * self.para_info.tp_size * self.grid_size,
        )
        self.flip_flag = 0
        self.vm_data_bytes_base = (
            self.para_info.inter_dp_size * self.para_info.local_size * max_local_bytes
        )
        self.vm_signal_bytes_base = round_up(
            2 * self.grid_size * get_element_size(torch.uint32), access_bytes
        )
        self.ns_data_bytes_base = max(
            (self.para_info.num_nodes + self.para_info.inter_dp_size) * max_local_bytes,
            min(self.para_info.num_nodes * 2, self.para_info.inter_dp_size * 2 + 1)
            * (max_local_bytes / self.para_info.local_tp_size),
        )
        self.ns_signal_bytes_base = round_up(
            2 * self.grid_size * get_element_size(torch.uint64), access_bytes
        )
        self.vm_data_bytes = 2 * self.vm_data_bytes_base
        self.vm_signal_bytes = 2 * self.vm_signal_bytes_base
        self.ns_data_bytes = 2 * self.ns_data_bytes_base
        self.ns_signal_bytes = 2 * self.ns_signal_bytes_base

        if "NVSHMEM_IB_ENABLE_IBGDA" not in os.environ:
            os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "true"
        if "NVSHMEM_IBGDA_NUM_RC_PER_PE" not in os.environ:
            os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] = f"{self.grid_size}"
        self.mixed_comm_module = get_mixed_comm_module()

        self.vm_handle_type = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        self.vm_buffer_size = None
        self.uc_handle_list = None
        self.uc_ptr_list = None
        void_p_null = ctypes.c_void_p(None)
        self.mem_data = [void_p_null for _ in range(2)]
        self.mem_signal = [void_p_null for _ in range(2)]
        uc_name_list = ["raw", "void_p"]
        self.uc_data_dict = [
            {key: void_p_null for key in uc_name_list} for _ in range(2)
        ]
        mc_name_list = ["full", "tp"]
        self.mc_handle_dict = {key: None for key in mc_name_list}
        self.mc_ptr_dict = {key: None for key in mc_name_list}
        self.mc_data_dict = [
            {key: void_p_null for key in mc_name_list} for _ in range(2)
        ]
        self.mc_signal_dict = [
            {key: void_p_null for key in mc_name_list} for _ in range(2)
        ]
        self.init_virtual_memory()

        if self.para_info.num_nodes > 1:
            self.init_nvshmem()
            self.ns_data = [
                self.mixed_comm_module.nvshmem_malloc(
                    [self.ns_data_bytes],
                    torch.uint8,
                    self.device.index,
                )
                for _ in range(2)
            ]
            self.ns_signal = [
                self.mixed_comm_module.nvshmem_malloc_with_init(
                    [self.ns_signal_bytes],
                    torch.uint8,
                    self.device.index,
                )
                for _ in range(2)
            ]
        else:
            self.ns_data = [None for _ in range(2)]
            self.ns_signal = [None for _ in range(2)]

        self.valid_block_size_dict = self.get_valid_block_size_dict()
        self.valid_mode_list = [val for val in MixedCommMode if self.check_mode(val)[0]]

        if (
            maybe_use_trtllm_comm
            and MixedCommMode.TRTLLM_LOCAL_INTER in self.valid_mode_list
        ):
            self.trtllm_workspace = create_allreduce_fusion_workspace(
                backend="trtllm",
                world_size=self.para_info.local_size,
                rank=self.para_info.local_rank,
                max_token_num=max_local_bs * self.para_info.dp_size,
                hidden_dim=hidden_size,
                dtype=dtype,
                comm_backend=TorchDistBackend(group=self.local_comm_group),
            )
        else:
            self.trtllm_workspace = None

        torch.cuda.synchronize()
        torch.distributed.barrier()

    def init_virtual_memory(self):
        def get_socket_path(pid):
            return f"/tmp/{pid}"

        def send_fd(sock, fd, rank):
            sock.sendmsg(
                [b"\x00"],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))],
                0,
                get_socket_path(local_pid_list[rank]),
            )

        def recv_fd(sock):
            ancdata = sock.recvmsg(
                1,
                socket.CMSG_SPACE(struct.calcsize("i")),
            )[1]
            for cmsg_level, cmsg_type, cmsg_data in ancdata:
                if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                    fd = struct.unpack("i", cmsg_data)[0]
                    return fd
            raise RuntimeError("Failed to receive file descriptor")

        def create_and_allgather_uc_handle(sock, uc_prop):
            local_rank = self.para_info.local_rank
            local_size = self.para_info.local_size
            uc_handle_list = [None for _ in range(local_size)]
            uc_handle_list[local_rank] = checkCudaErrors(
                cuda.cuMemCreate(self.vm_buffer_size, uc_prop, 0)
            )
            uc_fd_send = checkCudaErrors(
                cuda.cuMemExportToShareableHandle(
                    uc_handle_list[local_rank], self.vm_handle_type, 0
                )
            )
            for rel_rank in range(1, local_size):
                send_fd(sock, uc_fd_send, (local_rank + rel_rank) % local_size)
                uc_fd_recv = recv_fd(sock)
                uc_handle_list[(local_rank - rel_rank) % local_size] = checkCudaErrors(
                    cuda.cuMemImportFromShareableHandle(uc_fd_recv, self.vm_handle_type)
                )
                torch.distributed.barrier(group=self.local_comm_group)
            return uc_handle_list

        def create_and_send_mc_handle(sock, mc_prop, rank_list):
            mc_handle = checkCudaErrors(cuda.cuMulticastCreate(mc_prop))
            mc_fd = checkCudaErrors(
                cuda.cuMemExportToShareableHandle(mc_handle, self.vm_handle_type, 0)
            )
            for rank in rank_list[1:]:
                send_fd(sock, mc_fd, rank)
            checkCudaErrors(cuda.cuMulticastAddDevice(mc_handle, self.device.index))
            return mc_handle

        def recv_and_create_mc_handle(sock):
            mc_fd = recv_fd(sock)
            mc_handle = checkCudaErrors(
                cuda.cuMemImportFromShareableHandle(mc_fd, self.vm_handle_type)
            )
            checkCudaErrors(cuda.cuMulticastAddDevice(mc_handle, self.device.index))
            return mc_handle

        def map_handle(handle, access_desc, vm_granularity):
            ptr = checkCudaErrors(
                cuda.cuMemAddressReserve(self.vm_buffer_size, vm_granularity, 0, 0)
            )
            checkCudaErrors(cuda.cuMemMap(ptr, self.vm_buffer_size, 0, handle, 0))
            checkCudaErrors(
                cuda.cuMemSetAccess(ptr, self.vm_buffer_size, [access_desc], 1)
            )
            return ptr

        def create_gpu_array(ptr_list):
            ArrayType = ctypes.c_void_p * len(ptr_list)
            cpu_array = ArrayType(*ptr_list)
            array_bytes = ctypes.sizeof(cpu_array)
            gpu_array = checkCudaErrors(cuda.cuMemAlloc(array_bytes))
            checkCudaErrors(
                cuda.cuMemcpyHtoD(gpu_array, ctypes.addressof(cpu_array), array_bytes)
            )
            return {"raw": gpu_array, "void_p": ctypes.c_void_p(int(gpu_array))}

        # Create socket for broadcasting multicast and unicast handles
        local_pid_list = [None for _ in range(self.para_info.local_size)]
        torch.distributed.all_gather_object(
            local_pid_list,
            os.getpid(),
            group=self.local_comm_group,
        )
        socket_path = get_socket_path(local_pid_list[self.para_info.local_rank])
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(socket_path)
        torch.distributed.barrier(group=self.local_comm_group)

        # Create multicast and unicast handles
        uc_prop = cuda.CUmemAllocationProp()
        uc_prop.requestedHandleTypes = self.vm_handle_type
        uc_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        uc_prop.location = cuda.CUmemLocation()
        uc_prop.location.id = self.device.index
        uc_prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        uc_prop.allocFlags.gpuDirectRDMACapable = 1
        uc_granularity = checkCudaErrors(
            cuda.cuMemGetAllocationGranularity(
                uc_prop,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
        mc_prop = cuda.CUmulticastObjectProp()
        mc_prop.handleTypes = self.vm_handle_type
        mc_granularity = checkCudaErrors(
            cuda.cuMulticastGetGranularity(
                mc_prop,
                cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
            )
        )
        vm_granularity = math.lcm(mc_granularity, uc_granularity)
        self.vm_buffer_size = round_up(
            self.vm_data_bytes + self.vm_signal_bytes, vm_granularity
        )
        self.uc_handle_list = create_and_allgather_uc_handle(sock, uc_prop)
        mc_prop.size = self.vm_buffer_size
        if self.para_info.local_size > 1:
            if self.para_info.local_rank == 0:
                mc_prop.numDevices = self.para_info.local_size
                self.mc_handle_dict["full"] = create_and_send_mc_handle(
                    sock,
                    mc_prop,
                    self.para_info.get_local_full_group_local_ranks(),
                )
            else:
                self.mc_handle_dict["full"] = recv_and_create_mc_handle(sock)
            torch.distributed.barrier(group=self.local_comm_group)
        if self.para_info.local_tp_size > 1:
            if self.para_info.local_tp_rank == 0:
                mc_prop.numDevices = self.para_info.local_tp_size
                self.mc_handle_dict["tp"] = create_and_send_mc_handle(
                    sock,
                    mc_prop,
                    self.para_info.get_local_tp_group_local_ranks(),
                )
            else:
                self.mc_handle_dict["tp"] = recv_and_create_mc_handle(sock)
            torch.distributed.barrier(group=self.local_comm_group)
        sock.close()
        os.unlink(socket_path)

        # Bind and map memory
        access_desc = cuda.CUmemAccessDesc()
        access_desc.location = cuda.CUmemLocation()
        access_desc.location.id = self.device.index
        access_desc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.uc_ptr_list = [
            map_handle(val, access_desc, vm_granularity) for val in self.uc_handle_list
        ]
        ptr_list = [int(val) for val in self.uc_ptr_list]
        self.mem_data[0] = ctypes.c_void_p(ptr_list[self.para_info.local_rank])
        self.uc_data_dict[0] = create_gpu_array(ptr_list)
        ptr_list = [val + self.vm_data_bytes_base for val in ptr_list]
        self.mem_data[1] = ctypes.c_void_p(ptr_list[self.para_info.local_rank])
        self.uc_data_dict[1] = create_gpu_array(ptr_list)
        ptr_list = [val + self.vm_data_bytes_base for val in ptr_list]
        checkCudaErrors(
            cuda.cuMemsetD8(
                ptr_list[self.para_info.local_rank], 0, self.vm_signal_bytes
            )
        )
        self.mem_signal[0] = ctypes.c_void_p(ptr_list[self.para_info.local_rank])
        ptr_list = [val + self.vm_signal_bytes_base for val in ptr_list]
        self.mem_signal[1] = ctypes.c_void_p(ptr_list[self.para_info.local_rank])
        uc_handle_self = self.uc_handle_list[self.para_info.local_rank]
        for key, val in self.mc_handle_dict.items():
            if val is not None:
                checkCudaErrors(
                    cuda.cuMulticastBindMem(
                        val, 0, uc_handle_self, 0, self.vm_buffer_size, 0
                    )
                )
                self.mc_ptr_dict[key] = map_handle(val, access_desc, vm_granularity)
                ptr = int(self.mc_ptr_dict[key])
                self.mc_data_dict[0][key] = ctypes.c_void_p(ptr)
                ptr += self.vm_data_bytes_base
                self.mc_data_dict[1][key] = ctypes.c_void_p(ptr)
                ptr += self.vm_data_bytes_base
                self.mc_signal_dict[0][key] = ctypes.c_void_p(ptr)
                ptr += self.vm_signal_bytes_base
                self.mc_signal_dict[1][key] = ctypes.c_void_p(ptr)

    def init_nvshmem(self):
        if self.should_init_nvshmem:
            uid = torch.zeros(
                self.mixed_comm_module.nvshmem_unique_id_size(),
                dtype=torch.uint8,
                device="cpu",
            )
            if self.para_info.world_rank == 0:
                self.mixed_comm_module.nvshmem_get_unique_id(uid)
            torch.distributed.broadcast(uid, src=0)
            init_status = self.mixed_comm_module.nvshmem_init(
                uid,
                self.para_info.world_rank,
                self.para_info.world_size,
            )
            if init_status != 0:
                raise RuntimeError("Failed to initialize nvshmem")
        my_pe = self.mixed_comm_module.nvshmem_my_pe()
        n_pes = self.mixed_comm_module.nvshmem_n_pes()
        local_my_pe = self.mixed_comm_module.nvshmem_local_my_pe()
        local_n_pes = self.mixed_comm_module.nvshmem_local_n_pes()
        assert my_pe == self.para_info.world_rank, (
            f"Rank {self.para_info.world_rank}: nvshmem world_rank mismatch. "
            f"Expected world_rank {self.para_info.world_rank}, got world_rank {my_pe}."
        )
        assert n_pes == self.para_info.world_size, (
            f"Rank {self.para_info.world_rank}: nvshmem world_size mismatch. "
            f"Expected world_size {self.para_info.world_size}, got world_size {n_pes}."
        )
        assert local_my_pe == self.para_info.local_rank, (
            f"Rank {self.para_info.world_rank}: nvshmem local_rank mismatch. "
            f"Expected local_rank {self.para_info.local_rank}, got local_rank {local_my_pe}."
        )
        assert local_n_pes == self.para_info.local_size, (
            f"Rank {self.para_info.world_rank}: nvshmem local_size mismatch. "
            f"Expected local_size {self.para_info.local_size}, got local_size {local_n_pes}."
        )

    def __del__(self):
        def unmap_handle(ptr, handle):
            checkCudaErrors(cuda.cuMemUnmap(ptr, self.vm_buffer_size))
            checkCudaErrors(cuda.cuMemRelease(handle))
            checkCudaErrors(cuda.cuMemAddressFree(ptr, self.vm_buffer_size))

        torch.cuda.synchronize()
        torch.distributed.barrier()

        for key, val in self.mc_handle_dict.items():
            if val is not None:
                checkCudaErrors(
                    cuda.cuMulticastUnbind(
                        val, self.device.index, 0, self.vm_buffer_size
                    )
                )
                unmap_handle(self.mc_ptr_dict[key], val)
        for handle, ptr in zip(self.uc_handle_list, self.uc_ptr_list, strict=True):
            unmap_handle(ptr, handle)
        for val in self.uc_data_dict:
            checkCudaErrors(cuda.cuMemFree(val["raw"]))

        if self.para_info.num_nodes > 1:
            del self.ns_data
            del self.ns_signal
            if self.should_init_nvshmem:
                self.mixed_comm_module.nvshmem_finalize()

        if self.trtllm_workspace is not None:
            self.trtllm_workspace.destroy()

    def get_valid_block_size_dict(self):
        warp_size = 32
        block_size_range = self.mixed_comm_module.get_block_size_range(
            self.dtype,
            self.para_info.local_tp_rank,
            self.para_info.local_tp_size,
            self.para_info.local_dp_rank,
            self.para_info.local_dp_size,
            self.para_info.inter_tp_rank,
            self.para_info.inter_tp_size,
            self.para_info.inter_dp_rank,
            self.para_info.inter_dp_size,
        )
        outputs = {
            op: {mode: [None] for mode in MixedCommMode} for op in ["ar_ag", "rs_ar"]
        }
        mode_list = [MixedCommMode.FUSED_OPT_WAITS, MixedCommMode.FUSED_OPT_BYTES]
        if self.para_info.num_nodes > 1:
            mode_list.append(MixedCommMode.FUSED_MIXED)
        for op, sub_outputs in outputs.items():
            for mode in mode_list:
                min_val = block_size_range["min_val"]
                max_val = block_size_range[f"max_val_{op}_{mode}"]
                assert min_val <= max_val
                assert min_val % warp_size == 0
                assert max_val % warp_size == 0
                sub_outputs[mode] += list(
                    range(min_val, max_val + warp_size, warp_size)
                )
        return outputs

    def check_mode(self, mode: MixedCommMode) -> Tuple[bool, str | None]:
        if mode in [
            MixedCommMode.FUSED_OPT_WAITS,
            MixedCommMode.FUSED_OPT_BYTES,
            MixedCommMode.NCCL_TP_DP,
        ]:
            return True, None
        if mode == MixedCommMode.FUSED_MIXED:
            if self.para_info.num_nodes == 1:
                info = (
                    f"Rank {self.para_info.world_rank}: too small number of nodes. "
                    f"Expected at least 2, got {self.para_info.num_nodes}."
                )
                return False, info
            return True, None
        if mode == MixedCommMode.NCCL_AR:
            if self.para_info.tp_size == 1:
                info = (
                    f"Rank {self.para_info.world_rank}: {mode.name} is not in candidate list. "
                    f"{mode.name} is known to be inferior to NCCL_TP_DP in this case."
                )
                return False, info
            if self.para_info.dp_size == 1:
                info = (
                    f"Rank {self.para_info.world_rank}: {mode.name} is equivalent to NCCL_TP_DP. "
                    "Should use NCCL_TP_DP in this case."
                )
                return False, info
            return True, None
        if mode == MixedCommMode.NCCL_LOCAL_INTER:
            if self.para_info.num_nodes == 1:
                info = (
                    f"Rank {self.para_info.world_rank}: too small number of nodes. "
                    f"Expected at least 2, got {self.para_info.num_nodes}."
                )
                return False, info
            if self.para_info.tp_size == 1 or self.para_info.dp_size == 1:
                info = (
                    f"Rank {self.para_info.world_rank}: {mode.name} is not in candidate list. "
                    f"{mode.name} is known to be inferior to NCCL_TP_DP in this case."
                )
                return False, info
            if (
                min(self.para_info.local_tp_size, self.para_info.local_dp_size) == 1
                and min(self.para_info.inter_tp_size, self.para_info.inter_dp_size) == 1
            ):
                info = (
                    f"Rank {self.para_info.world_rank}: {mode.name} is equivalent to NCCL_TP_DP. "
                    "Should use NCCL_TP_DP in this case."
                )
                return False, info
            if (
                self.para_info.local_tp_size > 1 and self.para_info.local_dp_size > 2
            ) or (
                self.para_info.inter_tp_size > 1 and self.para_info.inter_dp_size > 2
            ):
                info = (
                    f"Rank {self.para_info.world_rank}: {mode.name} is not in candidate list. "
                    f"{mode.name} is expected to be suboptimal in this case."
                )
                return False, info
            return True, None
        if mode == MixedCommMode.TRTLLM_LOCAL_INTER:
            if self.para_info.local_dp_size > 2 or (
                self.para_info.inter_tp_size > 1 and self.para_info.inter_dp_size > 2
            ):
                info = (
                    f"Rank {self.para_info.world_rank}: {mode.name} is not in candidate list. "
                    f"{mode.name} is expected to be suboptimal in this case."
                )
                return False, info
            return True, None
        return False, f"Rank {self.para_info.world_rank}: {mode.name} is not supported."

    def allreduce_allgather(
        self,
        x_in: torch.Tensor,
        mode: MixedCommMode,
        block_size: int | None = None,
    ) -> torch.Tensor:
        assert x_in.ndim == 2
        assert x_in.shape[0] <= self.max_local_bs
        assert x_in.shape[1] == self.hidden_size
        assert x_in.dtype == self.dtype
        assert x_in.device == self.device
        is_valid_mode, info_valid_mode = self.check_mode(mode)
        assert is_valid_mode, info_valid_mode
        assert block_size in self.valid_block_size_dict["ar_ag"][mode], (
            f"Invalid block size: {block_size}. "
            f"Expected one of {self.valid_block_size_dict['ar_ag'][mode]}."
        )

        def create_tensor(
            x_in,
            dp_rank=self.para_info.dp_rank,
            dp_size=self.para_info.dp_size,
            copy_data=False,
        ):
            shape = [x_in.shape[0] * dp_size, *x_in.shape[1:]]
            if not copy_data:
                x_out = torch.empty(shape, dtype=x_in.dtype, device=x_in.device)
            else:
                x_out = torch.zeros(shape, dtype=x_in.dtype, device=x_in.device)
                x_out.unflatten(0, [dp_size, x_in.shape[0]])[dp_rank].copy_(x_in)
            return x_out

        def transpose_local_inter(
            x,
            local_dp_size=self.para_info.local_dp_size,
            inter_dp_size=self.para_info.inter_dp_size,
        ):
            if min(local_dp_size, inter_dp_size) == 1:
                return x
            else:
                return (
                    x.unflatten(0, [local_dp_size, inter_dp_size, -1])
                    .transpose(0, 1)
                    .flatten(0, 2)
                )

        if mode in [
            MixedCommMode.FUSED_OPT_WAITS,
            MixedCommMode.FUSED_OPT_BYTES,
            MixedCommMode.FUSED_MIXED,
        ]:
            x_out = create_tensor(x_in)
            self.mixed_comm_module.fused_allreduce_allgather(
                x_out,
                x_in,
                self.mem_data[self.flip_flag],
                self.mem_signal[self.flip_flag],
                self.mc_data_dict[self.flip_flag]["full"],
                self.mc_data_dict[self.flip_flag]["tp"],
                self.mc_signal_dict[self.flip_flag]["full"],
                self.mc_signal_dict[self.flip_flag]["tp"],
                self.ns_data[self.flip_flag],
                self.ns_signal[self.flip_flag],
                self.para_info.local_tp_rank,
                self.para_info.local_tp_size,
                self.para_info.local_dp_rank,
                self.para_info.local_dp_size,
                self.para_info.inter_tp_rank,
                self.para_info.inter_tp_size,
                self.para_info.inter_dp_rank,
                self.para_info.inter_dp_size,
                self.grid_size,
                mode,
                block_size,
            )
            self.flip_flag ^= 1
        elif mode == MixedCommMode.NCCL_TP_DP:
            if self.para_info.tp_size == 1:
                x_out = create_tensor(x_in)
                torch.distributed.all_gather_into_tensor(x_out, x_in)
            elif self.para_info.dp_size == 1:
                x_out = x_in.clone()
                torch.distributed.all_reduce(x_out)
            else:
                x_mid = x_in.clone()
                torch.distributed.all_reduce(x_mid, group=self.tp_comm_group)
                x_out = create_tensor(x_in)
                torch.distributed.all_gather_into_tensor(
                    x_out, x_mid, group=self.dp_comm_group
                )
        elif mode == MixedCommMode.NCCL_AR:
            x_out = create_tensor(x_in, copy_data=True)
            torch.distributed.all_reduce(x_out)
        elif mode == MixedCommMode.NCCL_LOCAL_INTER:
            if self.para_info.local_dp_size == 1:
                x_mid = x_in.clone()
                torch.distributed.all_reduce(x_mid, group=self.local_comm_group)
                x_out = create_tensor(
                    x_mid,
                    dp_rank=self.para_info.inter_dp_rank,
                    dp_size=self.para_info.inter_dp_size,
                    copy_data=True,
                )
                torch.distributed.all_reduce(x_out, group=self.inter_comm_group)
            else:
                if self.para_info.inter_tp_size == 1:
                    x_mid = create_tensor(
                        x_in,
                        dp_rank=self.para_info.inter_dp_rank,
                        dp_size=self.para_info.inter_dp_size,
                    )
                    torch.distributed.all_gather_into_tensor(
                        x_mid, x_in, group=self.inter_comm_group
                    )
                else:
                    x_mid = create_tensor(
                        x_in,
                        dp_rank=self.para_info.inter_dp_rank,
                        dp_size=self.para_info.inter_dp_size,
                        copy_data=True,
                    )
                    torch.distributed.all_reduce(x_mid, group=self.inter_comm_group)
                if self.para_info.local_tp_size == 1:
                    x_out = create_tensor(
                        x_mid,
                        dp_rank=self.para_info.local_dp_rank,
                        dp_size=self.para_info.local_dp_size,
                    )
                    torch.distributed.all_gather_into_tensor(
                        x_out, x_mid, group=self.local_comm_group
                    )
                else:
                    if self.para_info.local_dp_size == 1:
                        x_out = x_mid
                    else:
                        x_out = create_tensor(
                            x_mid,
                            dp_rank=self.para_info.local_dp_rank,
                            dp_size=self.para_info.local_dp_size,
                            copy_data=True,
                        )
                    torch.distributed.all_reduce(x_out, group=self.local_comm_group)
                x_out = transpose_local_inter(x_out)
        elif mode == MixedCommMode.TRTLLM_LOCAL_INTER:
            if self.para_info.num_nodes == 1:
                if self.para_info.local_dp_size == 1:
                    x_in_padded = x_in
                else:
                    x_in_padded = create_tensor(
                        x_in,
                        dp_rank=self.para_info.local_dp_rank,
                        dp_size=self.para_info.local_dp_size,
                        copy_data=True,
                    )
                x_out = trtllm_allreduce(x_in_padded, self.trtllm_workspace)
            else:
                if self.para_info.local_dp_size == 1:
                    x_mid = trtllm_allreduce(x_in, self.trtllm_workspace)
                    if self.para_info.inter_tp_size == 1:
                        x_out = create_tensor(x_in)
                        torch.distributed.all_gather_into_tensor(
                            x_out, x_mid, group=self.inter_comm_group
                        )
                    else:
                        x_out = create_tensor(
                            x_mid,
                            dp_rank=self.para_info.inter_dp_rank,
                            dp_size=self.para_info.inter_dp_size,
                            copy_data=True,
                        )
                        torch.distributed.all_reduce(x_out, group=self.inter_comm_group)
                else:
                    if self.para_info.inter_tp_size == 1:
                        x_mid = create_tensor(
                            x_in,
                            dp_rank=self.para_info.inter_dp_rank,
                            dp_size=self.para_info.inter_dp_size,
                        )
                        torch.distributed.all_gather_into_tensor(
                            x_mid, x_in, group=self.inter_comm_group
                        )
                    else:
                        x_mid = create_tensor(
                            x_in,
                            dp_rank=self.para_info.inter_dp_rank,
                            dp_size=self.para_info.inter_dp_size,
                            copy_data=True,
                        )
                        torch.distributed.all_reduce(x_mid, group=self.inter_comm_group)
                    if self.para_info.local_dp_size == 1:
                        x_mid_padded = x_mid
                    else:
                        x_mid_padded = create_tensor(
                            x_mid,
                            dp_rank=self.para_info.local_dp_rank,
                            dp_size=self.para_info.local_dp_size,
                            copy_data=True,
                        )
                    x_out = trtllm_allreduce(x_mid_padded, self.trtllm_workspace)
                    x_out = transpose_local_inter(x_out)
        else:
            raise ValueError(f"Invalid mode: {mode.name}")
        return x_out

    def reducescatter_allreduce(
        self,
        x_in: torch.Tensor,
        mode: MixedCommMode,
        block_size: int | None = None,
    ) -> torch.Tensor:
        assert x_in.ndim == 2
        assert x_in.shape[0] % self.para_info.dp_size == 0
        assert x_in.shape[0] <= self.max_local_bs * self.para_info.dp_size
        assert x_in.shape[1] == self.hidden_size
        assert x_in.dtype == self.dtype
        assert x_in.device == self.device
        is_valid_mode, info_valid_mode = self.check_mode(mode)
        assert is_valid_mode, info_valid_mode
        assert block_size in self.valid_block_size_dict["rs_ar"][mode], (
            f"Invalid block size: {block_size}. "
            f"Expected one of {self.valid_block_size_dict['rs_ar'][mode]}."
        )

        x_out_shape = [x_in.shape[0] // self.para_info.dp_size, *x_in.shape[1:]]
        if mode in [
            MixedCommMode.FUSED_OPT_WAITS,
            MixedCommMode.FUSED_OPT_BYTES,
            MixedCommMode.FUSED_MIXED,
        ]:
            x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
            self.mixed_comm_module.fused_reducescatter_allreduce(
                x_out,
                x_in,
                self.mem_data[self.flip_flag],
                self.mem_signal[self.flip_flag],
                self.uc_data_dict[self.flip_flag]["void_p"],
                self.mc_data_dict[self.flip_flag]["full"],
                self.mc_data_dict[self.flip_flag]["tp"],
                self.mc_signal_dict[self.flip_flag]["full"],
                self.ns_data[self.flip_flag],
                self.ns_signal[self.flip_flag],
                self.para_info.local_tp_rank,
                self.para_info.local_tp_size,
                self.para_info.local_dp_rank,
                self.para_info.local_dp_size,
                self.para_info.inter_tp_rank,
                self.para_info.inter_tp_size,
                self.para_info.inter_dp_rank,
                self.para_info.inter_dp_size,
                self.grid_size,
                mode,
                block_size,
            )
            self.flip_flag ^= 1
        elif mode == MixedCommMode.NCCL_TP_DP:
            if self.para_info.tp_size == 1:
                x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
                torch.distributed.reduce_scatter_tensor(x_out, x_in)
            elif self.para_info.dp_size == 1:
                x_out = x_in.clone()
                torch.distributed.all_reduce(x_out)
            else:
                x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
                torch.distributed.reduce_scatter_tensor(
                    x_out, x_in, group=self.dp_comm_group
                )
                torch.distributed.all_reduce(x_out, group=self.tp_comm_group)
        elif mode == MixedCommMode.NCCL_AR:
            x_out = x_in.clone()
            torch.distributed.all_reduce(x_out)
            x_out = x_out.unflatten(0, [self.para_info.dp_size, -1])[
                self.para_info.dp_rank
            ]
        elif mode == MixedCommMode.NCCL_LOCAL_INTER:
            if self.para_info.local_dp_size == 1:
                x_mid = x_in.clone()
                torch.distributed.all_reduce(x_mid, group=self.inter_comm_group)
                x_out = x_mid.unflatten(0, [self.para_info.inter_dp_size, -1])[
                    self.para_info.inter_dp_rank
                ]
                torch.distributed.all_reduce(x_out, group=self.local_comm_group)
            else:
                if self.para_info.local_tp_size == 1:
                    x_mid = torch.empty(
                        [
                            x_in.shape[0] // self.para_info.local_dp_size,
                            *x_in.shape[1:],
                        ],
                        dtype=x_in.dtype,
                        device=x_in.device,
                    )
                    x_in_transposed = (
                        x_in.unflatten(
                            0,
                            [
                                self.para_info.inter_dp_size,
                                self.para_info.local_dp_size,
                                -1,
                            ],
                        )
                        .transpose(0, 1)
                        .flatten(0, 2)
                    )
                    torch.distributed.reduce_scatter_tensor(
                        x_mid, x_in_transposed, group=self.local_comm_group
                    )
                    x_mid = x_mid.unflatten(0, [self.para_info.inter_dp_size, -1])
                else:
                    x_mid = x_in.clone()
                    torch.distributed.all_reduce(x_mid, group=self.local_comm_group)
                    x_mid = x_mid.unflatten(
                        0,
                        [
                            self.para_info.inter_dp_size,
                            self.para_info.local_dp_size,
                            -1,
                        ],
                    )[:, self.para_info.local_dp_rank].contiguous()
                if self.para_info.inter_tp_size == 1:
                    x_out = torch.empty(
                        x_out_shape, dtype=x_in.dtype, device=x_in.device
                    )
                    torch.distributed.reduce_scatter_tensor(
                        x_out, x_mid, group=self.inter_comm_group
                    )
                else:
                    torch.distributed.all_reduce(x_mid, group=self.inter_comm_group)
                    x_out = x_mid[self.para_info.inter_dp_rank]
        elif mode == MixedCommMode.TRTLLM_LOCAL_INTER:
            if self.para_info.num_nodes == 1:
                x_out = trtllm_allreduce(x_in, self.trtllm_workspace)
                x_out = x_out.unflatten(0, [self.para_info.local_dp_size, -1])[
                    self.para_info.local_dp_rank
                ]
            else:
                if self.para_info.local_dp_size == 1:
                    if self.para_info.inter_tp_size == 1:
                        x_mid = torch.empty(
                            [
                                x_in.shape[0] // self.para_info.inter_dp_size,
                                *x_in.shape[1:],
                            ],
                            dtype=x_in.dtype,
                            device=x_in.device,
                        )
                        torch.distributed.reduce_scatter_tensor(
                            x_mid, x_in, group=self.inter_comm_group
                        )
                    else:
                        x_mid = x_in.clone()
                        torch.distributed.all_reduce(x_mid, group=self.inter_comm_group)
                        x_mid = x_mid.unflatten(0, [self.para_info.inter_dp_size, -1])[
                            self.para_info.inter_dp_rank
                        ]
                    x_out = trtllm_allreduce(x_mid, self.trtllm_workspace)
                    x_out = x_out.unflatten(0, [self.para_info.local_dp_size, -1])[
                        self.para_info.local_dp_rank
                    ]
                else:
                    x_mid = trtllm_allreduce(x_in, self.trtllm_workspace)
                    x_mid = x_mid.unflatten(
                        0,
                        [
                            self.para_info.inter_dp_size,
                            self.para_info.local_dp_size,
                            -1,
                        ],
                    )[:, self.para_info.local_dp_rank].contiguous()
                    if self.para_info.inter_tp_size == 1:
                        x_out = torch.empty(
                            x_out_shape, dtype=x_in.dtype, device=x_in.device
                        )
                        torch.distributed.reduce_scatter_tensor(
                            x_out, x_mid, group=self.inter_comm_group
                        )
                    else:
                        torch.distributed.all_reduce(x_mid, group=self.inter_comm_group)
                        x_out = x_mid[self.para_info.inter_dp_rank]
        else:
            raise ValueError(f"Invalid mode: {mode.name}")
        return x_out
