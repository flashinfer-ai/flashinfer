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
import pathlib
import socket
import statistics
import struct
from itertools import product
from typing import Dict, List

import torch
from flashinfer.testing.utils import bench_gpu_time

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
from ..jit.comm import gen_mixed_comm_module


@functools.cache
def get_mixed_comm_module():
    import nvidia.nvshmem

    # Try to find libnvshmem_host.so first, fallback to libnvshmem_host.so.3
    lib_dir = pathlib.Path(nvidia.nvshmem.__path__[0]) / "lib"
    lib_path = None

    lib_names = ["libnvshmem_host.so", "libnvshmem_host.so.3"]
    for lib_name in lib_names:
        candidate_path = lib_dir / lib_name
        if candidate_path.exists():
            lib_path = candidate_path
            break

    if lib_path is None:
        raise FileNotFoundError(
            f"Could not find libnvshmem_host.so or libnvshmem_host.so.3 in {lib_dir}"
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


class MixedCommOp(enum.IntEnum):
    # Should be aligned with MixedCommOp in mixed_comm.cuh
    ALLREDUCE = 0
    ALLGATHER = enum.auto()
    REDUCESCATTER = enum.auto()
    ALLREDUCE_ALLGATHER = enum.auto()
    REDUCESCATTER_ALLREDUCE = enum.auto()


class MixedCommMode(enum.IntEnum):
    # Run a fused kernel (should be aligned with MixedCommMode in mixed_comm.cuh)
    FUSED_OPT_WAITS_MC = 0
    FUSED_OPT_WAITS_UC = enum.auto()
    FUSED_OPT_BYTES1_MC = enum.auto()
    FUSED_OPT_BYTES1_UC = enum.auto()
    FUSED_OPT_BYTES2_MC = enum.auto()
    FUSED_OPT_BYTES2_UC = enum.auto()
    # Run only one NCCL kernel (preprocessing or postprocessing may be applied)
    NCCL_ONE = enum.auto()
    # Run different NCCL kernels for TP and DP
    NCCL_TP_DP = enum.auto()
    # Choose the best mode from autotune
    AUTOTUNE = enum.auto()


class ParallelInfo:
    def __init__(
        self,
        world_rank: int,
        world_size: int,
        local_rank: int,
        local_size: int,
        inter_rank: int,
        inter_size: int,
        local_tp_size: int | None,
        local_dp_size: int | None,
        inter_tp_size: int | None,
        inter_dp_size: int | None,
    ):
        self._world_rank = world_rank
        self._world_size = world_size
        self._local_rank = local_rank
        self._local_size = local_size
        self._inter_rank = inter_rank
        self._inter_size = inter_size
        assert self.world_rank == self.inter_rank * self.local_size + self.local_rank
        assert self.world_size == self.inter_size * self.local_size
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
            self._inter_tp_size = self.inter_size
            self._inter_dp_size = 1
        elif inter_tp_size is None:
            assert self.inter_size % inter_dp_size == 0
            self._inter_tp_size = self.inter_size // inter_dp_size
            self._inter_dp_size = inter_dp_size
        elif inter_dp_size is None:
            assert self.inter_size % inter_tp_size == 0
            self._inter_tp_size = inter_tp_size
            self._inter_dp_size = self.inter_size // inter_tp_size
        else:
            assert self.inter_size == inter_tp_size * inter_dp_size
            self._inter_tp_size = inter_tp_size
            self._inter_dp_size = inter_dp_size
        self._local_tp_rank = self.local_rank % self.local_tp_size
        self._local_dp_rank = self.local_rank // self.local_tp_size
        self._inter_tp_rank = self.inter_rank % self.inter_tp_size
        self._inter_dp_rank = self.inter_rank // self.inter_tp_size

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
    def inter_rank(self):
        return self._inter_rank

    @property
    def inter_size(self):
        return self._inter_size

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

    @property
    def use_local_tp(self):
        return self.local_tp_size > 1

    @property
    def use_inter_tp(self):
        return self.inter_tp_size > 1

    @property
    def use_tp(self):
        return self.tp_size > 1

    @property
    def use_dp(self):
        return self.dp_size > 1

    @property
    def use_inter(self):
        return self.inter_size > 1

    @property
    def use_mixed(self):
        return self.use_tp and self.use_dp

    def get_local_comm_group(self):
        world_rank_list_all = [
            list(
                range(inter_rank * self.local_size, (inter_rank + 1) * self.local_size)
            )
            for inter_rank in range(self.inter_size)
        ]
        local_comm_group_list = [
            torch.distributed.new_group(world_rank_list)
            for world_rank_list in world_rank_list_all
        ]
        return local_comm_group_list[self.inter_rank]

    def get_tp_comm_group(self):
        world_rank_list_all = []
        for inter_dp_rank in range(self.inter_dp_size):
            for local_dp_rank in range(self.local_dp_size):
                local_rank_list = self.get_local_tp_group_local_ranks(local_dp_rank)
                world_rank_list = []
                for inter_tp_rank in range(self.inter_tp_size):
                    inter_rank = inter_dp_rank * self.inter_tp_size + inter_tp_rank
                    world_rank_list += [
                        val + inter_rank * self.local_size for val in local_rank_list
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
                    inter_rank = inter_dp_rank * self.inter_tp_size + inter_tp_rank
                    world_rank_list += [
                        val + inter_rank * self.local_size for val in local_rank_list
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


def nccl_allreduce(
    x_in: torch.Tensor,
    group: torch.distributed.ProcessGroup = None,
    inplace: bool = False,
) -> torch.Tensor:
    x_out = x_in if inplace else x_in.clone()
    torch.distributed.all_reduce(x_out, group=group)
    return x_out


def nccl_allgather(
    x_in: torch.Tensor,
    group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    dp_size = torch.distributed.get_world_size(group)
    x_out = torch.empty([dp_size, *x_in.shape], dtype=x_in.dtype, device=x_in.device)
    torch.distributed.all_gather_into_tensor(x_out, x_in, group=group)
    return x_out.flatten(0, 1)


def nccl_reducescatter(
    x_in: torch.Tensor,
    group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    dp_size = torch.distributed.get_world_size(group)
    x_in = x_in.unflatten(0, [dp_size, -1])
    x_out = torch.empty(x_in.shape[1:], dtype=x_in.dtype, device=x_in.device)
    torch.distributed.reduce_scatter_tensor(x_out, x_in, group=group)
    return x_out


def nccl_allreduce_allgather(
    x_in: torch.Tensor,
    para_info: ParallelInfo,
) -> torch.Tensor:
    x_out = torch.empty(
        [
            para_info.inter_tp_size,
            para_info.local_tp_size,
            para_info.inter_dp_size,
            para_info.local_dp_size,
            *x_in.shape,
        ],
        dtype=x_in.dtype,
        device=x_in.device,
    )
    x_out_list = [
        x_out[inter_tp_rank][local_tp_rank][inter_dp_rank][local_dp_rank]
        for inter_dp_rank, inter_tp_rank, local_dp_rank, local_tp_rank in product(
            range(para_info.inter_dp_size),
            range(para_info.inter_tp_size),
            range(para_info.local_dp_size),
            range(para_info.local_tp_size),
        )
    ]
    torch.distributed.all_gather(x_out_list, x_in)
    x_out = x_out.view([para_info.tp_size, para_info.dp_size, *x_in.shape]).sum(0)
    return x_out.flatten(0, 1)


def nccl_reducescatter_allreduce(
    x_in: torch.Tensor,
    para_info: ParallelInfo,
) -> torch.Tensor:
    x_in = x_in.unflatten(0, [para_info.inter_dp_size, para_info.local_dp_size, -1])
    x_in_list = [
        x_in[inter_dp_rank][local_dp_rank]
        for inter_dp_rank, _, local_dp_rank, _ in product(
            range(para_info.inter_dp_size),
            range(para_info.inter_tp_size),
            range(para_info.local_dp_size),
            range(para_info.local_tp_size),
        )
    ]
    x_out = torch.empty(
        [para_info.world_size, *x_in.shape[2:]], dtype=x_in.dtype, device=x_in.device
    )
    x_out_list = [x_out[world_rank] for world_rank in range(para_info.world_size)]
    torch.distributed.all_to_all(x_out_list, x_in_list)
    return x_out.sum(0)


class MixedComm:
    """
    An implementation for the combinations of all-reduce + all-gather and reduce-scatter + all-reduce.
    The fused kernels use virtual memory for intra-node communication and nvshmem for inter-node communication.
    Currently, only float16 and bfloat16 data types are supported.
    Note: An active torch.distributed process group should be initialized before creating an instance of this class.

    Args:
        world_rank (int): The world rank of the current process (local_rank + inter_rank * local_size).
        world_size (int): The total number of processes in the distributed group (inter_size * local_size).
        local_rank (int): The local rank of the current process (local_tp_rank + local_dp_rank * local_tp_size).
        local_size (int): The total number of processes in the current node (local_dp_size * local_tp_size).
        inter_rank (int): The index of the current node (inter_tp_rank + inter_dp_rank * inter_tp_size).
        inter_size (int): The total number of nodes in the distributed group (inter_dp_size * inter_tp_size).
        local_tp_size (int | None): TP size in the intra-node group. Use the default value if None is provided.
        local_dp_size (int | None): DP size in the intra-node group. Use the default value if None is provided.
        inter_tp_size (int | None): TP size in the inter-node group. Use the default value if None is provided.
        inter_dp_size (int | None): DP size in the inter-node group. Use the default value if None is provided.
        dtype (torch.dtype): The data type.
        device (torch.device): The device on which the tensors are located.
        grid_size (int | None, optional): The number of CTAs per GPU. The default behavior is to use the number of SMs.
        max_block_size (int | None, optional): The maximum limit of block size. The default behavior is not to set a limit.
        min_block_size (int, optional): The minimum block size if using multiple steps.
        min_num_steps (int, optional): The minimum number of steps if the maximum possible value of block size is chosen.
        ib_enable_ibgda (bool, optional): Whether to enable IBGDA.
        should_init_nvshmem (bool, optional): Whether to initialize nvshmem.
    Raises:
        RuntimeError: If nvshmem fails to initialize.
    """

    def __init__(
        self,
        world_rank: int,
        world_size: int,
        local_rank: int,
        local_size: int,
        inter_rank: int,
        inter_size: int,
        local_tp_size: int | None,
        local_dp_size: int | None,
        inter_tp_size: int | None,
        inter_dp_size: int | None,
        dtype: torch.dtype,
        device: torch.device,
        grid_size: int | None = None,
        max_block_size: int | None = None,
        min_block_size: int = 256,
        min_num_steps: int = 4,
        ib_enable_ibgda: bool = True,
        should_init_nvshmem: bool = True,
        use_autotune: bool = True,
    ):
        assert torch.distributed.is_initialized()
        assert local_size > 1
        self.is_running = True
        self.warp_size = 32
        self.access_bytes = 16
        self.num_buffers = 4
        self.neg_zero_uint16 = 0x8000
        self.para_info = ParallelInfo(
            world_rank=world_rank,
            world_size=world_size,
            local_rank=local_rank,
            local_size=local_size,
            inter_rank=inter_rank,
            inter_size=inter_size,
            local_tp_size=local_tp_size,
            local_dp_size=local_dp_size,
            inter_tp_size=inter_tp_size,
            inter_dp_size=inter_dp_size,
        )
        self.dtype = dtype
        self.device = device
        max_grid_size = torch.cuda.get_device_properties(device).multi_processor_count
        self.grid_size = max_grid_size if grid_size is None else grid_size
        assert self.grid_size <= max_grid_size
        assert min_block_size % self.warp_size == 0
        self.min_block_size = min_block_size
        self.min_num_steps = min_num_steps
        self.ib_enable_ibgda = ib_enable_ibgda
        self.should_init_nvshmem = should_init_nvshmem
        self.use_autotune = use_autotune
        self.autotune_base_bytes = 8192
        self.autotune_max_coef = max(15 - int(math.log2(self.para_info.dp_size)), 1)
        self.autotune_map: Dict[MixedCommOp, List[MixedCommMode]] = {}
        self.local_comm_group = (
            self.para_info.get_local_comm_group() if self.para_info.use_inter else None
        )
        self.tp_comm_group = (
            self.para_info.get_tp_comm_group() if self.para_info.use_mixed else None
        )
        self.dp_comm_group = (
            self.para_info.get_dp_comm_group() if self.para_info.use_mixed else None
        )

        if "NVSHMEM_IB_ENABLE_IBGDA" not in os.environ:
            os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = (
                "true" if self.ib_enable_ibgda else "false"
            )
        if "NVSHMEM_IBGDA_NUM_RC_PER_PE" not in os.environ:
            os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] = f"{self.grid_size}"
        self.mixed_comm_module = get_mixed_comm_module()

        self.valid_op_list = self.get_valid_op_list()
        self.valid_mode_list = self.get_valid_mode_list()
        self.op_dict = {
            MixedCommOp.ALLREDUCE: self.allreduce,
            MixedCommOp.ALLGATHER: self.allgather,
            MixedCommOp.REDUCESCATTER: self.reducescatter,
            MixedCommOp.ALLREDUCE_ALLGATHER: self.allreduce_allgather,
            MixedCommOp.REDUCESCATTER_ALLREDUCE: self.reducescatter_allreduce,
        }
        self.max_block_size_dict = {
            (op, mode): self.mixed_comm_module.get_max_block_size(
                self.dtype,
                local_tp_size,
                local_dp_size,
                inter_tp_size,
                inter_dp_size,
                op,
                mode,
            )
            for op in self.valid_op_list
            for mode in self.valid_mode_list
            if mode.name.startswith("FUSED_")
        }
        if max_block_size is not None:
            assert max_block_size % self.warp_size == 0
            assert max_block_size >= self.min_block_size
            for val in self.max_block_size_dict.values():
                val = {
                    sub_key: min(sub_val, max_block_size)
                    for sub_key, sub_val in val.items()
                }
        max_block_size = max(
            max(val.values()) for val in self.max_block_size_dict.values()
        )

        workspace_alignment = 16384
        assert workspace_alignment % self.access_bytes == 0
        data_bytes_base = max_block_size * self.access_bytes
        self.vm_buffer_bytes_base = round_up(
            self.para_info.tp_size * (self.para_info.dp_size + 1) * data_bytes_base
            + get_element_size(torch.uint32),
            workspace_alignment,
        )
        self.ns_data_bytes_base = round_up(
            2 * inter_size * data_bytes_base, workspace_alignment
        )
        self.ns_signal_bytes_base = 2 * get_element_size(torch.uint64)
        self.vm_buffer_bytes = self.grid_size * self.vm_buffer_bytes_base
        self.ns_data_bytes = self.grid_size * self.ns_data_bytes_base
        self.ns_signal_bytes = round_up(
            self.grid_size * self.ns_signal_bytes_base, workspace_alignment
        )
        self.vm_buffer_bytes_all = self.num_buffers * self.vm_buffer_bytes
        self.ns_data_bytes_all = self.num_buffers * self.ns_data_bytes
        self.ns_signal_bytes_all = self.num_buffers * self.ns_signal_bytes
        self.buffer_info_bytes = round_up(
            self.grid_size * get_element_size(torch.uint64), workspace_alignment
        )

        self.vm_handle_type = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        self.vm_workspace_bytes = None
        self.uc_handle_list = None
        self.uc_ptr_list = None
        self.ns_workspace = None
        void_p_null = ctypes.c_void_p(None)
        mc_name_list = ["full", "tp"]
        self.uc_buffer_dict = {key: void_p_null for key in ["raw", "void_p"]}
        self.mem_buffer = void_p_null
        self.mc_handle_dict = {name: None for name in mc_name_list}
        self.mc_ptr_dict = {name: None for name in mc_name_list}
        self.mc_buffer_dict = {name: void_p_null for name in mc_name_list}
        self.ns_buffer = void_p_null
        self.init_virtual_memory()
        if self.para_info.use_inter:
            self.init_nvshmem()

        torch.cuda.synchronize()
        torch.distributed.barrier()

        if self.use_autotune:
            self.run_autotune()

    def get_valid_op_list(self):
        # Should be aligned with is_valid_op in mixed_comm.cuh
        if self.para_info.use_mixed:
            op_list = [
                MixedCommOp.ALLREDUCE_ALLGATHER,
                MixedCommOp.REDUCESCATTER_ALLREDUCE,
            ]
        elif self.para_info.use_tp:
            op_list = [MixedCommOp.ALLREDUCE]
        else:
            assert self.para_info.use_dp
            op_list = [MixedCommOp.ALLGATHER, MixedCommOp.REDUCESCATTER]
        return op_list

    def get_valid_mode_list(self):
        # The fused modes in valid_mode_list should be aligned with is_valid_mode in mixed_comm.cuh
        valid_mode_list = [
            MixedCommMode.FUSED_OPT_WAITS_MC,
            MixedCommMode.FUSED_OPT_WAITS_UC,
        ]
        if self.para_info.use_local_tp:
            valid_mode_list += [
                MixedCommMode.FUSED_OPT_BYTES1_MC,
                MixedCommMode.FUSED_OPT_BYTES1_UC,
            ]
        if self.para_info.use_inter_tp:
            valid_mode_list += [
                MixedCommMode.FUSED_OPT_BYTES2_MC,
                MixedCommMode.FUSED_OPT_BYTES2_UC,
            ]
        valid_mode_list.append(MixedCommMode.NCCL_ONE)
        if self.para_info.use_mixed:
            valid_mode_list.append(MixedCommMode.NCCL_TP_DP)
        if self.use_autotune:
            valid_mode_list.append(MixedCommMode.AUTOTUNE)
        return valid_mode_list

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
                cuda.cuMemCreate(self.vm_workspace_bytes, uc_prop, 0)
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
                cuda.cuMemAddressReserve(self.vm_workspace_bytes, vm_granularity, 0, 0)
            )
            checkCudaErrors(cuda.cuMemMap(ptr, self.vm_workspace_bytes, 0, handle, 0))
            checkCudaErrors(
                cuda.cuMemSetAccess(ptr, self.vm_workspace_bytes, [access_desc], 1)
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
        self.vm_workspace_bytes = round_up(
            self.vm_buffer_bytes_all + self.buffer_info_bytes, vm_granularity
        )
        self.uc_handle_list = create_and_allgather_uc_handle(sock, uc_prop)
        mc_prop.size = self.vm_workspace_bytes
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
        if self.para_info.use_local_tp:
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
        uc_buffer_list = []
        local_tp_rank = self.para_info.local_tp_rank
        local_dp_rank = self.para_info.local_dp_rank
        local_tp_size = self.para_info.local_tp_size
        local_dp_size = self.para_info.local_dp_size
        for i in range(local_dp_size):
            peer_local_dp_rank = (local_dp_rank + i + 1) % local_dp_size
            for j in range(local_tp_size):
                peer_local_tp_rank = (local_tp_rank + j + 1) % local_tp_size
                uc_buffer_list.append(
                    ptr_list[peer_local_tp_rank + peer_local_dp_rank * local_tp_size]
                )
        self.uc_buffer_dict = create_gpu_array(uc_buffer_list)
        ptr = ptr_list[self.para_info.local_rank]
        self.mem_buffer = ctypes.c_void_p(ptr)
        checkCudaErrors(
            cuda.cuMemsetD16(ptr, self.neg_zero_uint16, self.vm_buffer_bytes_all // 2)
        )
        checkCudaErrors(
            cuda.cuMemsetD8(ptr + self.vm_buffer_bytes_all, 0, self.buffer_info_bytes)
        )
        uc_handle_self = self.uc_handle_list[self.para_info.local_rank]
        for key, val in self.mc_handle_dict.items():
            if val is None:
                continue
            checkCudaErrors(
                cuda.cuMulticastBindMem(
                    val, 0, uc_handle_self, 0, self.vm_workspace_bytes, 0
                )
            )
            self.mc_ptr_dict[key] = map_handle(val, access_desc, vm_granularity)
            self.mc_buffer_dict[key] = ctypes.c_void_p(int(self.mc_ptr_dict[key]))

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
        self.ns_workspace = self.mixed_comm_module.nvshmem_malloc(
            [self.ns_data_bytes_all + self.ns_signal_bytes_all],
            torch.uint8,
            self.device.index,
        )
        ns_tensor = torch.from_dlpack(self.ns_workspace)
        ptr = int(ns_tensor.data_ptr())
        self.ns_buffer = ctypes.c_void_p(ptr)
        checkCudaErrors(
            cuda.cuMemsetD8(ptr + self.ns_data_bytes_all, 0, self.ns_signal_bytes_all)
        )

    def shutdown(self):
        def unmap_handle(ptr, handle):
            checkCudaErrors(cuda.cuMemUnmap(ptr, self.vm_workspace_bytes))
            checkCudaErrors(cuda.cuMemRelease(handle))
            checkCudaErrors(cuda.cuMemAddressFree(ptr, self.vm_workspace_bytes))

        assert self.is_running, "Should not call shutdown() more than once."
        self.is_running = False
        torch.cuda.synchronize()
        torch.distributed.barrier()

        for key, val in self.mc_handle_dict.items():
            if val is not None:
                checkCudaErrors(
                    cuda.cuMulticastUnbind(
                        val, self.device.index, 0, self.vm_workspace_bytes
                    )
                )
                unmap_handle(self.mc_ptr_dict[key], val)
        for handle, ptr in zip(self.uc_handle_list, self.uc_ptr_list, strict=True):
            unmap_handle(ptr, handle)
        checkCudaErrors(cuda.cuMemFree(self.uc_buffer_dict["raw"]))

        if self.para_info.use_inter:
            del self.ns_workspace
            if self.should_init_nvshmem:
                self.mixed_comm_module.nvshmem_finalize()

        for val in [self.local_comm_group, self.tp_comm_group, self.dp_comm_group]:
            if val is not None:
                torch.distributed.destroy_process_group(val)

    def __del__(self):
        if self.is_running:
            self.shutdown()

    def run_autotune(self):
        max_local_bs = pow(2, self.autotune_max_coef - 1)
        hidden_size = self.autotune_base_bytes // get_element_size(self.dtype)
        data = torch.empty(
            [max_local_bs * self.para_info.dp_size, hidden_size],
            dtype=self.dtype,
            device=self.device,
        ).uniform_(-0.5, 0.5)
        for op in self.valid_op_list:
            self.autotune_map[op] = []
            for log_local_bs in range(self.autotune_max_coef):
                local_bs = pow(2, log_local_bs)
                if op in [
                    MixedCommOp.REDUCESCATTER,
                    MixedCommOp.REDUCESCATTER_ALLREDUCE,
                ]:
                    x_in = data[: local_bs * self.para_info.dp_size]
                else:
                    x_in = data[:local_bs]
                best_duration = float("inf")
                best_mode = None
                for mode in self.valid_mode_list:
                    if mode == MixedCommMode.AUTOTUNE:
                        continue
                    duration_list = bench_gpu_time(
                        self.run_op,
                        input_args=(op, x_in, mode),
                        input_kwargs=None,
                        dry_run_time_ms=10,
                        repeat_time_ms=100,
                        use_cuda_graph=False,
                        num_iters_within_graph=10,
                    )
                    duration = statistics.mean(duration_list)
                    if duration < best_duration:
                        best_duration = duration
                        best_mode = mode
                self.autotune_map[op].append(best_mode)

    def select_autotune_mode(
        self,
        op: MixedCommOp,
        x_in: torch.Tensor,
    ) -> MixedCommMode:
        num_local_bytes = x_in.numel() * get_element_size(self.dtype)
        if op in [MixedCommOp.REDUCESCATTER, MixedCommOp.REDUCESCATTER_ALLREDUCE]:
            num_local_bytes //= self.para_info.dp_size
        coef = math.log2(num_local_bytes / self.autotune_base_bytes)
        if coef <= 0:
            mode = self.autotune_map[op][0]
        elif coef >= self.autotune_max_coef - 1:
            mode = self.autotune_map[op][-1]
        else:
            mode = self.autotune_map[op][round(coef)]
        return mode

    def allreduce(
        self,
        x_in: torch.Tensor,
        mode: MixedCommMode,
    ) -> torch.Tensor:
        op = MixedCommOp.ALLREDUCE
        assert op in self.valid_op_list
        assert x_in.ndim >= 2
        assert x_in.dtype == self.dtype
        assert x_in.device == self.device
        assert mode in self.valid_mode_list
        if mode == MixedCommMode.AUTOTUNE:
            mode = self.select_autotune_mode(op, x_in)
        if mode.name.startswith("FUSED_"):
            x_out = torch.empty_like(x_in)
            self.mixed_comm_module.allreduce(
                x_out,
                x_in,
                self.uc_buffer_dict["void_p"],
                self.mem_buffer,
                self.mc_buffer_dict["full"],
                self.ns_buffer,
                self.vm_buffer_bytes_base,
                self.ns_data_bytes,
                self.ns_signal_bytes,
                self.grid_size,
                self.max_block_size_dict[(op, mode)],
                self.min_block_size,
                self.min_num_steps,
                self.para_info.local_rank,
                self.para_info.local_size,
                self.para_info.inter_rank,
                self.para_info.inter_size,
                mode,
            )
        elif mode == MixedCommMode.NCCL_ONE:
            x_out = nccl_allreduce(x_in)
        else:
            raise ValueError(f"Invalid mode: {mode.name}")
        return x_out

    def allgather(
        self,
        x_in: torch.Tensor,
        mode: MixedCommMode,
    ) -> torch.Tensor:
        op = MixedCommOp.ALLGATHER
        assert op in self.valid_op_list
        assert x_in.ndim >= 2
        assert x_in.dtype == self.dtype
        assert x_in.device == self.device
        assert mode in self.valid_mode_list
        if mode == MixedCommMode.AUTOTUNE:
            mode = self.select_autotune_mode(op, x_in)
        if mode.name.startswith("FUSED_"):
            x_out_shape = [x_in.shape[0] * self.para_info.dp_size, *x_in.shape[1:]]
            x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
            self.mixed_comm_module.allgather(
                x_out,
                x_in,
                self.uc_buffer_dict["void_p"],
                self.mem_buffer,
                self.mc_buffer_dict["full"],
                self.ns_buffer,
                self.vm_buffer_bytes_base,
                self.ns_data_bytes,
                self.ns_signal_bytes,
                self.grid_size,
                self.max_block_size_dict[(op, mode)],
                self.min_block_size,
                self.min_num_steps,
                self.para_info.local_rank,
                self.para_info.local_size,
                self.para_info.inter_rank,
                self.para_info.inter_size,
                mode,
            )
        elif mode == MixedCommMode.NCCL_ONE:
            x_out = nccl_allgather(x_in)
        else:
            raise ValueError(f"Invalid mode: {mode.name}")
        return x_out

    def reducescatter(
        self,
        x_in: torch.Tensor,
        mode: MixedCommMode,
    ) -> torch.Tensor:
        op = MixedCommOp.REDUCESCATTER
        assert op in self.valid_op_list
        assert x_in.ndim >= 2
        assert x_in.shape[0] % self.para_info.dp_size == 0
        assert x_in.dtype == self.dtype
        assert x_in.device == self.device
        assert mode in self.valid_mode_list
        if mode == MixedCommMode.AUTOTUNE:
            mode = self.select_autotune_mode(op, x_in)
        if mode.name.startswith("FUSED_"):
            x_out_shape = [x_in.shape[0] // self.para_info.dp_size, *x_in.shape[1:]]
            x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
            self.mixed_comm_module.reducescatter(
                x_out,
                x_in,
                self.uc_buffer_dict["void_p"],
                self.mem_buffer,
                self.mc_buffer_dict["full"],
                self.ns_buffer,
                self.vm_buffer_bytes_base,
                self.ns_data_bytes,
                self.ns_signal_bytes,
                self.grid_size,
                self.max_block_size_dict[(op, mode)],
                self.min_block_size,
                self.min_num_steps,
                self.para_info.local_rank,
                self.para_info.local_size,
                self.para_info.inter_rank,
                self.para_info.inter_size,
                mode,
            )
        elif mode == MixedCommMode.NCCL_ONE:
            x_out = nccl_reducescatter(x_in)
        else:
            raise ValueError(f"Invalid mode: {mode.name}")
        return x_out

    def allreduce_allgather(
        self,
        x_in: torch.Tensor,
        mode: MixedCommMode,
    ) -> torch.Tensor:
        op = MixedCommOp.ALLREDUCE_ALLGATHER
        assert op in self.valid_op_list
        assert x_in.ndim >= 2
        assert x_in.dtype == self.dtype
        assert x_in.device == self.device
        assert mode in self.valid_mode_list
        if mode == MixedCommMode.AUTOTUNE:
            mode = self.select_autotune_mode(op, x_in)
        if mode.name.startswith("FUSED_"):
            x_out_shape = [x_in.shape[0] * self.para_info.dp_size, *x_in.shape[1:]]
            x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
            self.mixed_comm_module.fused_allreduce_allgather(
                x_out,
                x_in,
                self.uc_buffer_dict["void_p"],
                self.mem_buffer,
                self.mc_buffer_dict["full"],
                self.mc_buffer_dict["tp"],
                self.ns_buffer,
                self.vm_buffer_bytes_base,
                self.ns_data_bytes,
                self.ns_signal_bytes,
                self.grid_size,
                self.max_block_size_dict[(op, mode)],
                self.min_block_size,
                self.min_num_steps,
                self.para_info.local_tp_rank,
                self.para_info.local_tp_size,
                self.para_info.local_dp_rank,
                self.para_info.local_dp_size,
                self.para_info.inter_tp_rank,
                self.para_info.inter_tp_size,
                self.para_info.inter_dp_rank,
                self.para_info.inter_dp_size,
                mode,
            )
        elif mode == MixedCommMode.NCCL_ONE:
            x_out = nccl_allreduce_allgather(x_in, self.para_info)
        elif mode == MixedCommMode.NCCL_TP_DP:
            x_mid = nccl_allreduce(x_in, group=self.tp_comm_group)
            x_out = nccl_allgather(x_mid, group=self.dp_comm_group)
        else:
            raise ValueError(f"Invalid mode: {mode.name}")
        return x_out

    def reducescatter_allreduce(
        self,
        x_in: torch.Tensor,
        mode: MixedCommMode,
    ) -> torch.Tensor:
        op = MixedCommOp.REDUCESCATTER_ALLREDUCE
        assert op in self.valid_op_list
        assert x_in.ndim >= 2
        assert x_in.shape[0] % self.para_info.dp_size == 0
        assert x_in.dtype == self.dtype
        assert x_in.device == self.device
        assert mode in self.valid_mode_list
        if mode == MixedCommMode.AUTOTUNE:
            mode = self.select_autotune_mode(op, x_in)
        if mode.name.startswith("FUSED_"):
            x_out_shape = [x_in.shape[0] // self.para_info.dp_size, *x_in.shape[1:]]
            x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
            self.mixed_comm_module.fused_reducescatter_allreduce(
                x_out,
                x_in,
                self.uc_buffer_dict["void_p"],
                self.mem_buffer,
                self.mc_buffer_dict["full"],
                self.mc_buffer_dict["tp"],
                self.ns_buffer,
                self.vm_buffer_bytes_base,
                self.ns_data_bytes,
                self.ns_signal_bytes,
                self.grid_size,
                self.max_block_size_dict[(op, mode)],
                self.min_block_size,
                self.min_num_steps,
                self.para_info.local_tp_rank,
                self.para_info.local_tp_size,
                self.para_info.local_dp_rank,
                self.para_info.local_dp_size,
                self.para_info.inter_tp_rank,
                self.para_info.inter_tp_size,
                self.para_info.inter_dp_rank,
                self.para_info.inter_dp_size,
                mode,
            )
        elif mode == MixedCommMode.NCCL_ONE:
            x_out = nccl_reducescatter_allreduce(x_in, self.para_info)
        elif mode == MixedCommMode.NCCL_TP_DP:
            x_mid = nccl_reducescatter(x_in, group=self.dp_comm_group)
            x_out = nccl_allreduce(x_mid, group=self.tp_comm_group, inplace=True)
        else:
            raise ValueError(f"Invalid mode: {mode.name}")
        return x_out

    def run_op(
        self,
        op: MixedCommOp,
        x_in: torch.Tensor,
        mode: MixedCommMode,
    ) -> torch.Tensor:
        return self.op_dict[op](x_in, mode)
