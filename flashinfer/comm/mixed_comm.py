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
import tempfile
from itertools import product
from typing import Dict, List

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

from ..api_logging import flashinfer_api
from ..cuda_utils import checkCudaErrors
from ..jit.comm import gen_mixed_comm_module
from ..testing.utils import bench_gpu_time
from ..utils import backend_requirement, supported_compute_capability


@functools.cache
def get_mixed_comm_module():
    """Load and return the JIT-compiled mixed communication module.

    Locates the nvshmem host library, loads it globally, and builds/loads
    the mixed communication CUDA module via JIT compilation. The result
    is cached so the module is compiled only once per process.

    Returns:
        The compiled and loaded mixed communication module.

    Raises:
        FileNotFoundError: If libnvshmem_host.so cannot be found.
    """
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


def _get_element_size(dtype: torch.dtype) -> int:
    """Return the size in bytes of a single element of the given dtype."""
    return torch.empty((), dtype=dtype).element_size()


def _ceil_div(x: int, y: int) -> int:
    """Return the ceiling of integer division x / y."""
    return (x + y - 1) // y


def _round_up(x: int, y: int) -> int:
    """Round x up to the nearest multiple of y."""
    return _ceil_div(x, y) * y


class MixedCommOp(enum.IntEnum):
    """Enumeration of mixed communication operation types.

    Values must be aligned with ``MixedCommOp`` in ``mixed_comm.cuh``.
    """

    ALLREDUCE = 0
    ALLGATHER = enum.auto()
    REDUCESCATTER = enum.auto()
    ALLREDUCE_ALLGATHER = enum.auto()
    REDUCESCATTER_ALLREDUCE = enum.auto()


class MixedCommMode(enum.IntEnum):
    """Enumeration of mixed communication execution modes.

    Fused modes run a single fused kernel using virtual memory (intra-node)
    and nvshmem (inter-node). NCCL modes delegate to one or more NCCL
    collective calls. AUTOTUNE selects the best mode based on profiling.

    Fused values must be aligned with ``MixedCommMode`` in ``mixed_comm.cuh``.
    """

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
    """Describes the parallel topology for mixed communication.

    Encodes the hierarchical decomposition of ranks into local (intra-node)
    and inter (inter-node) groups, further split into tensor-parallel (TP)
    and data-parallel (DP) sub-groups. Provides convenience properties and
    helpers to create ``torch.distributed`` process groups for each axis.

    Args:
        world_rank: Global rank of the current process.
        world_size: Total number of processes.
        local_rank: Rank within the current node.
        local_size: Number of processes per node.
        inter_rank: Index of the current node.
        inter_size: Total number of nodes.
        local_tp_size: Intra-node TP group size. Defaults to ``local_size``.
        local_dp_size: Intra-node DP group size. Defaults to 1.
        inter_tp_size: Inter-node TP group size. Defaults to ``inter_size``.
        inter_dp_size: Inter-node DP group size. Defaults to 1.
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
    ):
        """Initialize parallel topology information and compute derived ranks."""
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
        """Global rank of the current process."""
        return self._world_rank

    @property
    def world_size(self):
        """Total number of processes in the distributed group."""
        return self._world_size

    @property
    def local_rank(self):
        """Rank of the current process within its node."""
        return self._local_rank

    @property
    def local_size(self):
        """Number of processes per node."""
        return self._local_size

    @property
    def inter_rank(self):
        """Index of the current node."""
        return self._inter_rank

    @property
    def inter_size(self):
        """Total number of nodes."""
        return self._inter_size

    @property
    def local_tp_rank(self):
        """Tensor-parallel rank within the intra-node group."""
        return self._local_tp_rank

    @property
    def local_tp_size(self):
        """Tensor-parallel group size within a node."""
        return self._local_tp_size

    @property
    def local_dp_rank(self):
        """Data-parallel rank within the intra-node group."""
        return self._local_dp_rank

    @property
    def local_dp_size(self):
        """Data-parallel group size within a node."""
        return self._local_dp_size

    @property
    def inter_tp_rank(self):
        """Tensor-parallel rank across nodes."""
        return self._inter_tp_rank

    @property
    def inter_tp_size(self):
        """Tensor-parallel group size across nodes."""
        return self._inter_tp_size

    @property
    def inter_dp_rank(self):
        """Data-parallel rank across nodes."""
        return self._inter_dp_rank

    @property
    def inter_dp_size(self):
        """Data-parallel group size across nodes."""
        return self._inter_dp_size

    @property
    def tp_rank(self):
        """Global tensor-parallel rank."""
        return self.local_tp_rank + self.inter_tp_rank * self.local_tp_size

    @property
    def tp_size(self):
        """Global tensor-parallel group size."""
        return self.local_tp_size * self.inter_tp_size

    @property
    def dp_rank(self):
        """Global data-parallel rank."""
        return self.local_dp_rank + self.inter_dp_rank * self.local_dp_size

    @property
    def dp_size(self):
        """Global data-parallel group size."""
        return self.local_dp_size * self.inter_dp_size

    @property
    def use_local_tp(self):
        """Whether intra-node tensor parallelism is active."""
        return self.local_tp_size > 1

    @property
    def use_inter_tp(self):
        """Whether inter-node tensor parallelism is active."""
        return self.inter_tp_size > 1

    @property
    def use_tp(self):
        """Whether tensor parallelism is active (local or inter)."""
        return self.tp_size > 1

    @property
    def use_dp(self):
        """Whether data parallelism is active."""
        return self.dp_size > 1

    @property
    def use_inter(self):
        """Whether multi-node communication is needed."""
        return self.inter_size > 1

    @property
    def use_mixed(self):
        """Whether both TP and DP are active (mixed parallelism)."""
        return self.use_tp and self.use_dp

    def get_local_comm_group(self):
        """Create and return a ``torch.distributed`` process group for all ranks on this node."""
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
        """Create and return a ``torch.distributed`` process group for the TP axis."""
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
        """Create and return a ``torch.distributed`` process group for the DP axis."""
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
        """Return the list of all local ranks on the current node."""
        return list(range(self.local_size))

    def get_local_tp_group_local_ranks(self, local_dp_rank: int | None = None):
        """Return local ranks belonging to a TP sub-group on this node.

        Args:
            local_dp_rank: The DP rank whose TP group to query. Defaults to this process's DP rank.
        """
        if local_dp_rank is None:
            local_dp_rank = self.local_dp_rank
        return list(
            range(
                local_dp_rank * self.local_tp_size,
                (local_dp_rank + 1) * self.local_tp_size,
            )
        )

    def get_local_dp_group_local_ranks(self, local_tp_rank: int | None = None):
        """Return local ranks belonging to a DP sub-group on this node.

        Args:
            local_tp_rank: The TP rank whose DP group to query. Defaults to this process's TP rank.
        """
        if local_tp_rank is None:
            local_tp_rank = self.local_tp_rank
        return list(range(local_tp_rank, self.local_size, self.local_tp_size))


class MixedCommHandler:
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
        """Initialize the handler, set up virtual memory, nvshmem, and run autotune."""
        assert torch.distributed.is_initialized()
        assert local_size > 1
        assert dtype in [torch.float16, torch.bfloat16]
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
        self.max_block_size_dict = {
            (op, mode): self.mixed_comm_module.get_max_block_size(
                self.dtype,
                self.para_info.local_tp_size,
                self.para_info.local_dp_size,
                self.para_info.inter_tp_size,
                self.para_info.inter_dp_size,
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
                for sub_key, sub_val in val.items():
                    val[sub_key] = min(sub_val, max_block_size)
        max_block_size = max(
            max(val.values()) for val in self.max_block_size_dict.values()
        )

        workspace_alignment = 16384
        assert workspace_alignment % self.access_bytes == 0
        data_bytes_base = max_block_size * self.access_bytes
        self.vm_buffer_bytes_base = _round_up(
            self.para_info.tp_size * (self.para_info.dp_size + 1) * data_bytes_base
            + _get_element_size(torch.uint32),
            workspace_alignment,
        )
        self.ns_data_bytes_base = _round_up(
            2 * inter_size * data_bytes_base, workspace_alignment
        )
        self.ns_signal_bytes_base = 2 * _get_element_size(torch.uint64)
        self.vm_buffer_bytes = self.grid_size * self.vm_buffer_bytes_base
        self.ns_data_bytes = self.grid_size * self.ns_data_bytes_base
        self.ns_signal_bytes = _round_up(
            self.grid_size * self.ns_signal_bytes_base, workspace_alignment
        )
        self.vm_buffer_bytes_all = self.num_buffers * self.vm_buffer_bytes
        self.ns_data_bytes_all = self.num_buffers * self.ns_data_bytes
        self.ns_signal_bytes_all = self.num_buffers * self.ns_signal_bytes
        self.buffer_info_bytes = _round_up(
            self.grid_size * _get_element_size(torch.uint64), workspace_alignment
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
        """Return the list of valid communication operations for the current parallel topology."""
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
        """Return the list of valid execution modes for the current parallel topology."""
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
        """Initialize CUDA virtual memory for intra-node communication.

        Allocates unicast and multicast memory handles, exchanges them between
        local ranks via Unix domain sockets, and maps them into the GPU virtual
        address space so that all ranks on a node can directly read/write each
        other's buffers.
        """

        def get_socket_path(rank):
            """Return the Unix socket path for the given local rank."""
            return f"{socket_folder}/rank_{rank}"

        def send_fd(sock, fd, rank):
            """Send a file descriptor to the specified rank via Unix socket."""
            sock.sendmsg(
                [b"\x00"],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))],
                0,
                get_socket_path(rank),
            )

        def recv_fd(sock):
            """Receive a file descriptor from a Unix socket."""
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
            """Create a unicast memory handle and exchange it with all local ranks."""
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
                os.close(uc_fd_recv)
                torch.distributed.barrier(group=self.local_comm_group)
            os.close(uc_fd_send)
            return uc_handle_list

        def create_and_send_mc_handle(sock, mc_prop, rank_list):
            """Create a multicast handle and send it to all other ranks in the group."""
            mc_handle = checkCudaErrors(cuda.cuMulticastCreate(mc_prop))
            mc_fd = checkCudaErrors(
                cuda.cuMemExportToShareableHandle(mc_handle, self.vm_handle_type, 0)
            )
            for rank in rank_list[1:]:
                send_fd(sock, mc_fd, rank)
            os.close(mc_fd)
            checkCudaErrors(cuda.cuMulticastAddDevice(mc_handle, self.device.index))
            return mc_handle

        def recv_and_create_mc_handle(sock):
            """Receive a multicast handle from the group leader and register the local device."""
            mc_fd = recv_fd(sock)
            mc_handle = checkCudaErrors(
                cuda.cuMemImportFromShareableHandle(mc_fd, self.vm_handle_type)
            )
            os.close(mc_fd)
            checkCudaErrors(cuda.cuMulticastAddDevice(mc_handle, self.device.index))
            return mc_handle

        def map_handle(handle, access_desc, vm_granularity):
            """Reserve virtual address space and map a memory handle into it."""
            ptr = checkCudaErrors(
                cuda.cuMemAddressReserve(self.vm_workspace_bytes, vm_granularity, 0, 0)
            )
            checkCudaErrors(cuda.cuMemMap(ptr, self.vm_workspace_bytes, 0, handle, 0))
            checkCudaErrors(
                cuda.cuMemSetAccess(ptr, self.vm_workspace_bytes, [access_desc], 1)
            )
            return ptr

        def create_gpu_array(ptr_list):
            """Copy a list of pointers into a GPU-resident array."""
            ArrayType = ctypes.c_void_p * len(ptr_list)
            cpu_array = ArrayType(*ptr_list)
            array_bytes = ctypes.sizeof(cpu_array)
            gpu_array = checkCudaErrors(cuda.cuMemAlloc(array_bytes))
            checkCudaErrors(
                cuda.cuMemcpyHtoD(gpu_array, ctypes.addressof(cpu_array), array_bytes)
            )
            return {"raw": gpu_array, "void_p": ctypes.c_void_p(int(gpu_array))}

        # Create socket for broadcasting multicast and unicast handles
        local_ranks = self.para_info.get_local_full_group_local_ranks()
        if self.para_info.local_rank == 0:
            socket_folder_list = [tempfile.mkdtemp(prefix="flashinfer_mixed_comm_")]
            os.chmod(socket_folder_list[0], 0o700)
        else:
            socket_folder_list = [None]
        torch.distributed.broadcast_object_list(
            socket_folder_list,
            src=self.para_info.inter_rank * self.para_info.local_size,
            group=self.local_comm_group,
        )
        socket_folder = socket_folder_list[0]
        socket_path = get_socket_path(self.para_info.local_rank)
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
        self.vm_workspace_bytes = _round_up(
            self.vm_buffer_bytes_all + self.buffer_info_bytes, vm_granularity
        )
        self.uc_handle_list = create_and_allgather_uc_handle(sock, uc_prop)
        mc_prop.size = self.vm_workspace_bytes
        if self.para_info.local_rank == 0:
            mc_prop.numDevices = self.para_info.local_size
            self.mc_handle_dict["full"] = create_and_send_mc_handle(
                sock, mc_prop, local_ranks
            )
        else:
            self.mc_handle_dict["full"] = recv_and_create_mc_handle(sock)
        torch.distributed.barrier(group=self.local_comm_group)
        if self.para_info.use_local_tp:
            if self.para_info.local_tp_rank == 0:
                mc_prop.numDevices = self.para_info.local_tp_size
                self.mc_handle_dict["tp"] = create_and_send_mc_handle(
                    sock, mc_prop, self.para_info.get_local_tp_group_local_ranks()
                )
            else:
                self.mc_handle_dict["tp"] = recv_and_create_mc_handle(sock)
            torch.distributed.barrier(group=self.local_comm_group)
        torch.distributed.barrier(group=self.local_comm_group)
        sock.close()
        os.unlink(socket_path)
        torch.distributed.barrier(group=self.local_comm_group)
        if self.para_info.local_rank == 0:
            os.rmdir(socket_folder)

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
        """Initialize nvshmem for inter-node communication.

        Broadcasts a unique ID from rank 0, initializes the nvshmem library,
        verifies rank consistency, and allocates symmetric nvshmem workspace
        for data and signal buffers.

        Raises:
            RuntimeError: If nvshmem fails to initialize.
        """
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
        """Tear down all communication resources.

        Unmaps and releases virtual memory handles, frees nvshmem workspace,
        and destroys process groups. Must be called at most once.
        """

        def unmap_handle(ptr, handle):
            """Unmap, release, and free a virtual memory handle and its address range."""
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
        """Ensure shutdown is called when the handler is garbage collected."""
        if self.is_running:
            self.shutdown()

    def run_autotune(self):
        """Profile all valid (op, mode) combinations and populate the autotune map.

        For each valid operation and a range of input sizes (powers of 2), benchmarks
        every non-AUTOTUNE mode and records the fastest one in ``self.autotune_map``.
        """
        max_local_bs = pow(2, self.autotune_max_coef - 1)
        hidden_size = self.autotune_base_bytes // _get_element_size(self.dtype)
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
                        run_mixed_comm,
                        input_args=(op, self, x_in),
                        input_kwargs={"mode": mode},
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
        """Select the best execution mode for the given operation and input tensor.

        Looks up the profiled autotune map using a log2-scaled index derived from
        the input tensor's byte size.

        Args:
            op: The communication operation to perform.
            x_in: The input tensor.

        Returns:
            The autotuned :class:`MixedCommMode`.
        """
        num_local_bytes = x_in.numel() * x_in.element_size()
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


def _nccl_allreduce(
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    group: torch.distributed.ProcessGroup = None,
    inplace: bool = False,
) -> torch.Tensor:
    """Perform all-reduce using NCCL via ``torch.distributed.all_reduce``."""
    if inplace:
        x_out = x_in
    else:
        if x_out is None:
            x_out = x_in.clone()
        else:
            x_out.copy_(x_in)
    torch.distributed.all_reduce(x_out, group=group)
    return x_out


def _nccl_allgather(
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """Perform all-gather using NCCL via ``torch.distributed.all_gather``."""
    dp_size = torch.distributed.get_world_size(group)
    if x_out is None:
        x_out = torch.empty(
            [dp_size * x_in.shape[0], *x_in.shape[1:]],
            dtype=x_in.dtype,
            device=x_in.device,
        )
    torch.distributed.all_gather_into_tensor(x_out, x_in, group=group)
    return x_out


def _nccl_reducescatter(
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """Perform reduce-scatter using NCCL via ``torch.distributed.reduce_scatter_tensor``."""
    dp_size = torch.distributed.get_world_size(group)
    x_in = x_in.unflatten(0, [dp_size, -1])
    if x_out is None:
        x_out = torch.empty(x_in.shape[1:], dtype=x_in.dtype, device=x_in.device)
    torch.distributed.reduce_scatter_tensor(x_out, x_in, group=group)
    return x_out


def _nccl_allreduce_allgather(
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    para_info: ParallelInfo,
) -> torch.Tensor:
    """Perform all-reduce + all-gather using NCCL (single global collective)."""
    x_tmp = torch.empty(
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
    x_tmp_list = [
        x_tmp[inter_tp_rank][local_tp_rank][inter_dp_rank][local_dp_rank]
        for inter_dp_rank, inter_tp_rank, local_dp_rank, local_tp_rank in product(
            range(para_info.inter_dp_size),
            range(para_info.inter_tp_size),
            range(para_info.local_dp_size),
            range(para_info.local_tp_size),
        )
    ]
    torch.distributed.all_gather(x_tmp_list, x_in)
    x_tmp = x_tmp.view([para_info.tp_size, para_info.dp_size, *x_in.shape]).sum(0)
    x_tmp = x_tmp.flatten(0, 1)
    if x_out is None:
        x_out = x_tmp
    else:
        x_out.copy_(x_tmp)
    return x_out


def _nccl_reducescatter_allreduce(
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    para_info: ParallelInfo,
) -> torch.Tensor:
    """Perform reduce-scatter + all-reduce using NCCL (single global collective)."""
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
    x_tmp = torch.empty(
        [para_info.world_size, *x_in.shape[2:]], dtype=x_in.dtype, device=x_in.device
    )
    x_tmp_list = [x_tmp[world_rank] for world_rank in range(para_info.world_size)]
    torch.distributed.all_to_all(x_tmp_list, x_in_list)
    x_tmp = x_tmp.sum(0)
    if x_out is None:
        x_out = x_tmp
    else:
        x_out.copy_(x_tmp)
    return x_out


def _allreduce(
    handler: MixedCommHandler,
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    mode: MixedCommMode,
) -> torch.Tensor:
    """Dispatch all-reduce using the specified mode (fused or NCCL)."""
    op = MixedCommOp.ALLREDUCE
    if mode.name.startswith("FUSED_"):
        if x_out is None:
            x_out = torch.empty_like(x_in)
        handler.mixed_comm_module.allreduce(
            x_out,
            x_in,
            handler.uc_buffer_dict["void_p"],
            handler.mem_buffer,
            handler.mc_buffer_dict["full"],
            handler.ns_buffer,
            handler.vm_buffer_bytes_base,
            handler.ns_data_bytes,
            handler.ns_signal_bytes,
            handler.grid_size,
            handler.max_block_size_dict[(op, mode)],
            handler.min_block_size,
            handler.min_num_steps,
            handler.para_info.local_rank,
            handler.para_info.local_size,
            handler.para_info.inter_rank,
            handler.para_info.inter_size,
            mode,
        )
    elif mode == MixedCommMode.NCCL_ONE:
        x_out = _nccl_allreduce(x_in, x_out)
    else:
        raise ValueError(f"Invalid mode: {mode.name}")
    return x_out


def _allgather(
    handler: MixedCommHandler,
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    mode: MixedCommMode,
) -> torch.Tensor:
    """Dispatch all-gather using the specified mode (fused or NCCL)."""
    op = MixedCommOp.ALLGATHER
    if mode.name.startswith("FUSED_"):
        if x_out is None:
            x_out_shape = [x_in.shape[0] * handler.para_info.dp_size, *x_in.shape[1:]]
            x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
        handler.mixed_comm_module.allgather(
            x_out,
            x_in,
            handler.uc_buffer_dict["void_p"],
            handler.mem_buffer,
            handler.mc_buffer_dict["full"],
            handler.ns_buffer,
            handler.vm_buffer_bytes_base,
            handler.ns_data_bytes,
            handler.ns_signal_bytes,
            handler.grid_size,
            handler.max_block_size_dict[(op, mode)],
            handler.min_block_size,
            handler.min_num_steps,
            handler.para_info.local_rank,
            handler.para_info.local_size,
            handler.para_info.inter_rank,
            handler.para_info.inter_size,
            mode,
        )
    elif mode == MixedCommMode.NCCL_ONE:
        x_out = _nccl_allgather(x_in, x_out)
    else:
        raise ValueError(f"Invalid mode: {mode.name}")
    return x_out


def _reducescatter(
    handler: MixedCommHandler,
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    mode: MixedCommMode,
) -> torch.Tensor:
    """Dispatch reduce-scatter using the specified mode (fused or NCCL)."""
    op = MixedCommOp.REDUCESCATTER
    if mode.name.startswith("FUSED_"):
        if x_out is None:
            x_out_shape = [x_in.shape[0] // handler.para_info.dp_size, *x_in.shape[1:]]
            x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
        handler.mixed_comm_module.reducescatter(
            x_out,
            x_in,
            handler.uc_buffer_dict["void_p"],
            handler.mem_buffer,
            handler.mc_buffer_dict["full"],
            handler.ns_buffer,
            handler.vm_buffer_bytes_base,
            handler.ns_data_bytes,
            handler.ns_signal_bytes,
            handler.grid_size,
            handler.max_block_size_dict[(op, mode)],
            handler.min_block_size,
            handler.min_num_steps,
            handler.para_info.local_rank,
            handler.para_info.local_size,
            handler.para_info.inter_rank,
            handler.para_info.inter_size,
            mode,
        )
    elif mode == MixedCommMode.NCCL_ONE:
        x_out = _nccl_reducescatter(x_in, x_out)
    else:
        raise ValueError(f"Invalid mode: {mode.name}")
    return x_out


def _allreduce_allgather(
    handler: MixedCommHandler,
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    mode: MixedCommMode,
) -> torch.Tensor:
    """Dispatch all-reduce + all-gather using the specified mode."""
    op = MixedCommOp.ALLREDUCE_ALLGATHER
    if mode.name.startswith("FUSED_"):
        if x_out is None:
            x_out_shape = [x_in.shape[0] * handler.para_info.dp_size, *x_in.shape[1:]]
            x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
        handler.mixed_comm_module.fused_allreduce_allgather(
            x_out,
            x_in,
            handler.uc_buffer_dict["void_p"],
            handler.mem_buffer,
            handler.mc_buffer_dict["full"],
            handler.mc_buffer_dict["tp"],
            handler.ns_buffer,
            handler.vm_buffer_bytes_base,
            handler.ns_data_bytes,
            handler.ns_signal_bytes,
            handler.grid_size,
            handler.max_block_size_dict[(op, mode)],
            handler.min_block_size,
            handler.min_num_steps,
            handler.para_info.local_tp_rank,
            handler.para_info.local_tp_size,
            handler.para_info.local_dp_rank,
            handler.para_info.local_dp_size,
            handler.para_info.inter_tp_rank,
            handler.para_info.inter_tp_size,
            handler.para_info.inter_dp_rank,
            handler.para_info.inter_dp_size,
            mode,
        )
    elif mode == MixedCommMode.NCCL_ONE:
        x_out = _nccl_allreduce_allgather(x_in, x_out, handler.para_info)
    elif mode == MixedCommMode.NCCL_TP_DP:
        x_tmp = _nccl_allreduce(x_in, None, group=handler.tp_comm_group)
        x_out = _nccl_allgather(x_tmp, x_out, group=handler.dp_comm_group)
    else:
        raise ValueError(f"Invalid mode: {mode.name}")
    return x_out


def _reducescatter_allreduce(
    handler: MixedCommHandler,
    x_in: torch.Tensor,
    x_out: torch.Tensor | None,
    mode: MixedCommMode,
) -> torch.Tensor:
    """Dispatch reduce-scatter + all-reduce using the specified mode."""
    op = MixedCommOp.REDUCESCATTER_ALLREDUCE
    if mode.name.startswith("FUSED_"):
        if x_out is None:
            x_out_shape = [x_in.shape[0] // handler.para_info.dp_size, *x_in.shape[1:]]
            x_out = torch.empty(x_out_shape, dtype=x_in.dtype, device=x_in.device)
        handler.mixed_comm_module.fused_reducescatter_allreduce(
            x_out,
            x_in,
            handler.uc_buffer_dict["void_p"],
            handler.mem_buffer,
            handler.mc_buffer_dict["full"],
            handler.mc_buffer_dict["tp"],
            handler.ns_buffer,
            handler.vm_buffer_bytes_base,
            handler.ns_data_bytes,
            handler.ns_signal_bytes,
            handler.grid_size,
            handler.max_block_size_dict[(op, mode)],
            handler.min_block_size,
            handler.min_num_steps,
            handler.para_info.local_tp_rank,
            handler.para_info.local_tp_size,
            handler.para_info.local_dp_rank,
            handler.para_info.local_dp_size,
            handler.para_info.inter_tp_rank,
            handler.para_info.inter_tp_size,
            handler.para_info.inter_dp_rank,
            handler.para_info.inter_dp_size,
            mode,
        )
    elif mode == MixedCommMode.NCCL_ONE:
        x_out = _nccl_reducescatter_allreduce(x_in, x_out, handler.para_info)
    elif mode == MixedCommMode.NCCL_TP_DP:
        x_out = _nccl_reducescatter(x_in, x_out, group=handler.dp_comm_group)
        x_out = _nccl_allreduce(x_out, None, group=handler.tp_comm_group, inplace=True)
    else:
        raise ValueError(f"Invalid mode: {mode.name}")
    return x_out


@supported_compute_capability([90, 100])
def _common_check(
    op: MixedCommOp,
    handler: MixedCommHandler,
    x_in: torch.Tensor,
    x_out: torch.Tensor | None = None,
    mode: MixedCommMode | None = None,
) -> bool:
    """Validate inputs for a mixed communication operation.

    Checks that the operation, mode, tensor dimensions, dtypes, devices,
    and shapes are consistent with the handler's configuration.

    Raises:
        ValueError: If any validation check fails.
    """
    if op not in handler.valid_op_list:
        raise ValueError(f"Invalid op: {op.name}")
    if mode is not None and mode not in handler.valid_mode_list:
        raise ValueError(f"Invalid mode: {mode.name}")
    if x_in.ndim < 2:
        raise ValueError("x_in.ndim should be at least 2")
    if x_in.dtype != handler.dtype:
        raise ValueError(f"x_in.dtype should be {handler.dtype}")
    if x_in.device != handler.device:
        raise ValueError(f"x_in.device should be {handler.device}")
    if op in [MixedCommOp.REDUCESCATTER, MixedCommOp.REDUCESCATTER_ALLREDUCE]:
        if x_in.shape[0] % handler.para_info.dp_size != 0:
            raise ValueError("x_in.shape[0] should be divisible by dp_size")
    if x_out is not None:
        if x_out.ndim != x_in.ndim:
            raise ValueError("x_out.ndim should be equal to x_in.ndim")
        if x_out.dtype != x_in.dtype:
            raise ValueError("x_out.dtype should be equal to x_in.dtype")
        if x_out.device != x_in.device:
            raise ValueError("x_out.device should be equal to x_in.device")
        if x_out.shape[1:] != x_in.shape[1:]:
            raise ValueError("x_out.shape[1:] should be equal to x_in.shape[1:]")
        if op in [MixedCommOp.REDUCESCATTER, MixedCommOp.REDUCESCATTER_ALLREDUCE]:
            if x_out.shape[0] * handler.para_info.dp_size != x_in.shape[0]:
                raise ValueError(
                    "x_out.shape[0] * dp_size should be equal to x_in.shape[0]"
                )
        else:
            if x_out.shape[0] != x_in.shape[0] * handler.para_info.dp_size:
                raise ValueError(
                    "x_out.shape[0] should be equal to x_in.shape[0] * dp_size"
                )
    return True


_mixed_comm_op_dict = {
    MixedCommOp.ALLREDUCE: _allreduce,
    MixedCommOp.ALLGATHER: _allgather,
    MixedCommOp.REDUCESCATTER: _reducescatter,
    MixedCommOp.ALLREDUCE_ALLGATHER: _allreduce_allgather,
    MixedCommOp.REDUCESCATTER_ALLREDUCE: _reducescatter_allreduce,
}


@flashinfer_api
@backend_requirement(
    backend_checks={},
    common_check=_common_check,
)
def run_mixed_comm(
    op: MixedCommOp,
    handler: MixedCommHandler,
    x_in: torch.Tensor,
    x_out: torch.Tensor | None = None,
    mode: MixedCommMode | None = None,
) -> torch.Tensor:
    """Execute a mixed communication operation.

    This is the main entry point for running communication collectives
    through the mixed communication handler. It supports fused GPU kernels
    (using virtual memory intra-node and nvshmem inter-node), NCCL-based
    fallbacks, and autotuned mode selection.

    Args:
        op: The communication operation to perform.
        handler: An initialized :class:`MixedCommHandler`.
        x_in: The input tensor. Must be at least 2-D and match the handler's dtype/device.
        x_out: Optional pre-allocated output tensor. Allocated automatically if ``None``.
        mode: The execution mode. If ``None``, uses autotune (if enabled) or falls back
            to an NCCL mode.

    Returns:
        The output tensor containing the result of the collective operation.
    """
    if mode is None:
        if handler.use_autotune:
            mode = MixedCommMode.AUTOTUNE
        else:
            if MixedCommMode.NCCL_TP_DP in handler.valid_mode_list:
                mode = MixedCommMode.NCCL_TP_DP
            else:
                mode = MixedCommMode.NCCL_ONE
    if mode == MixedCommMode.AUTOTUNE:
        mode = handler.select_autotune_mode(op, x_in)
    return _mixed_comm_op_dict[op](handler, x_in, x_out, mode)
