import ctypes
import threading
import functools
from dataclasses import dataclass
from enum import IntEnum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from typing import Optional, Tuple, Union
from flashinfer.mapping import Mapping
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from cuda import cuda, cudart
from cuda.cudart import cudaError_t

import torch.nn as nn

import os

from flashinfer.comm import allreduce, moe_allreduce

'''
Some functions used for trtllm allreduce params
'''

class AllReduceStrategy(IntEnum):
    NCCL = 0
    MIN_LATENCY = 1
    UB = 2
    AUTO = 3
    ONESHOT = 4
    TWOSHOT = 5


class AllReduceFusionOp(IntEnum):
    NONE = 0
    RESIDUAL_RMS_NORM = 1
    LAST_PROCESS_FOR_UB = 2
    RESIDUAL_RMS_PREPOST_NORM = 3
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7
    MOE_ALLREDUCE_RESIDUAL_RMS_NORM = 8

def _raise_if_error(error: cudaError_t | cuda.CUresult):
    if isinstance(error, cudaError_t):
        if error != cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Runtime API error: {repr(error)}")
    if isinstance(error, cuda.CUresult):
        if error != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Driver API error: {repr(error)}")

def can_access_peer(mapping: Mapping) -> bool:
    src_node = mapping.local_rank

    for rank in mapping.tp_group:
        dest_node = mapping.get_local_rank(rank)

        # Early exit if devices are on different nodes
        if mapping.get_node_rank(rank) != mapping.node_rank:
            # logger.info(
            #     f"Detect inter-node TP between rank {mapping.rank} and rank {rank}"
            # )
            return False

        # Skip if same device
        if dest_node == src_node:
            continue

        error, result = cudart.cudaDeviceCanAccessPeer(src_node, dest_node)
        _raise_if_error(error)

        if result == 0:
            # logger.info(
            #     f"cudaDeviceCanAccessPeer failed for device: {src_node} peerDevice: {dest_node}"
            # )
            return False

    return True

class IpcMemory():

    # WARNING: Must in sync with FLAGS_SIZE in cpp/include/tensorrt_llm/runtime/ipcUtils.h
    # (Max all reduce blocks + 1) * sizeof(int)
    IPC_BARRIERS_SIZE_PER_GPU = (24 + 1) * 4

    def __init__(self, mapping: Mapping, size: int, open_ipc: bool = True):
        self.mapping = mapping
        self.open_ipc = open_ipc and mapping.tp_size <= mapping.gpus_per_node
        if self.open_ipc:
            self.peer_ptrs, self.local_ptr = IpcMemory.open_ipc_memory(
                self.mapping, size, True)
        else:
            self.peer_ptrs = [0] * mapping.tp_size
            self.local_ptr = 0

    def __del__(self):
        if not sys.is_finalizing() and self.open_ipc:
            IpcMemory.close_ipc_memory(self.mapping, self.peer_ptrs)

    def serialize(self) -> List[int]:
        buffer = bytes(0)
        for ptr in self.peer_ptrs:
            buffer += struct.pack("P", ptr)

        return array.array("Q", buffer).tolist()

    @staticmethod
    def open_ipc_memory(mapping: Mapping,
                        size: int,
                        set_to_zero: bool = False) -> Tuple[List[int], int]:
        """ Allocates a buffer with the given *size* on each GPU. Then, enables IPC communication between TP groups.
        Returns a list of buffer pointers, buffers[i] is a handle to the corresponding buffer residing on GPU #i.
        Call close_ipc_handle with the *buffer*.
        """

        def align_size(size, alignment):
            if (size % alignment) != 0:
                size += alignment - (size % alignment)
            return size

        comm = mpi_comm().Split(
            mapping.pp_rank * mapping.cp_size + mapping.cp_rank,
            mapping.tp_rank)

        # see allocateIpcMemory in cpp/tensorrt_llm/runtime/ipcUtils.cpp for alignment reason
        # 1 << 21 is 2MB
        aligned_size = align_size(size, 1 << 21)
        error, local_ptr = cudart.cudaMalloc(aligned_size)
        _raise_if_error(error)
        if set_to_zero:
            _raise_if_error(cudart.cudaMemset(local_ptr, 0, aligned_size)[0])
        error, local_handle = cudart.cudaIpcGetMemHandle(local_ptr)
        _raise_if_error(error)

        handles_reserved = comm.allgather(local_handle.reserved)
        handles = []
        for reserved in handles_reserved:
            handle = cudart.cudaIpcMemHandle_t()
            handle.reserved = reserved
            handles.append(handle)

        peer_ptrs = []
        for node, handle in enumerate(handles):
            if node == mapping.tp_rank:
                peer_ptrs.append(local_ptr)
            else:
                error, ptr = cudart.cudaIpcOpenMemHandle(
                    handle, cudart.cudaIpcMemLazyEnablePeerAccess)
                _raise_if_error(error)
                peer_ptrs.append(ptr)

        return peer_ptrs, local_ptr

    @staticmethod
    def close_ipc_memory(mapping: Mapping, peer_ptrs: List[int]):
        for node, ptr in enumerate(peer_ptrs):
            if node == mapping.tp_rank:
                _raise_if_error(cudart.cudaFree(ptr)[0])
            else:
                _raise_if_error(cudart.cudaIpcCloseMemHandle(ptr)[0])


def force_all_reduce_deterministic():
    return os.getenv("FORCE_DETERMINISTIC", "0") == "1" or os.getenv(
        "FORCE_ALL_REDUCE_DETERMINISTIC", "0") == "1"

class CustomAllReduceHelper:
    """
        Globally visible class to help usage of custom_all_reduce plugin.
        Provides the following utilities:

        workspace: Tensor
            When using CUSTOM or AUTO mode, a tensor containing pointers to memory
            visible to all GPUs. It should be 3 pointers per TP rank -
            ptr to data buffer, ptr to barriers in, ptr to barriers out.
            It must be initialized using IpcMemory class.

        Usage:
            - Set custom_all_reduce_helper.workspace with the required tensor.
              Then, each instance of allreduce will reference that tensor automatically.
    """
    POINTERS_PER_RANK = 7
    POINTERS_OF_COUNTER = 2

    def __init__(self) -> None:
        self.workspace: Optional[torch.Tensor] = None

    def set_workspace_tensor(self,
                             mapping: Mapping,
                             num_profiles: Optional[int] = None):
        workspace_size = self.POINTERS_PER_RANK * mapping.tp_size + self.POINTERS_OF_COUNTER

        dim_range = None
        if num_profiles is not None:
            dim_range = OrderedDict([('all_reduce_size',
                                      [workspace_size] * num_profiles)])

        self.workspace = torch.Tensor(
            name='all_reduce_workspace',
            dtype=torch.int64,
            shape=[workspace_size],
            dim_range=dim_range,
        )

    @staticmethod
    def max_workspace_size_auto(tp_size: int,
                                support_deterministic=True) -> int:
        if force_all_reduce_deterministic() and support_deterministic:
            workspace_size = os.getenv("FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE",
                                       "1000000000")
            return int(workspace_size)
        if tp_size <= 2:
            return 16_000_000
        return 8_000_000

    @staticmethod
    def allocate_workspace(mapping: Mapping,
                           size: int) -> Tuple[List[IpcMemory], "torch.tensor"]:
        import torch

        # Force pull mode and disable lamport when force deterministic is enabled, for reducing device memory usage.
        force_deterministic = force_all_reduce_deterministic()
        is_p2p_supported = can_access_peer(mapping)
        ipc_buffers_size = size if force_deterministic else size * mapping.tp_size
        ipc_buffers_ping = IpcMemory(mapping, ipc_buffers_size,
                                     is_p2p_supported)
        ipc_buffers_pong = IpcMemory(mapping, ipc_buffers_size,
                                     is_p2p_supported)
        ipc_barriers_in = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size * 2,
            is_p2p_supported)
        ipc_barriers_out = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size * 2,
            is_p2p_supported)
        lamport_buffers_size = 1 if force_deterministic else size * mapping.tp_size
        lamport_buffers_0 = IpcMemory(mapping, lamport_buffers_size,
                                      is_p2p_supported)
        lamport_buffers_1 = IpcMemory(mapping, lamport_buffers_size,
                                      is_p2p_supported)
        lamport_buffers_2 = IpcMemory(mapping, lamport_buffers_size,
                                      is_p2p_supported)
        # TODO: it seems we may need to initialize lamport buffers for all tp groups
        # just like its cpp counterpart (AllReduceBuffers::AllReduceBuffers()) does.
        if is_p2p_supported:
            lamport_initialize_all(
                lamport_buffers_0.local_ptr,
                lamport_buffers_1.local_ptr,
                lamport_buffers_2.local_ptr,
                lamport_buffers_size,
            )
        buffers = [
            ipc_buffers_ping, ipc_buffers_pong, ipc_barriers_in,
            ipc_barriers_out, lamport_buffers_0, lamport_buffers_1,
            lamport_buffers_2
        ]

        return buffers, torch.tensor(
            ipc_buffers_ping.serialize() + ipc_buffers_pong.serialize() +
            ipc_barriers_in.serialize() + ipc_barriers_out.serialize() +
            lamport_buffers_0.serialize() + lamport_buffers_1.serialize() +
            lamport_buffers_2.serialize() + [0] + [0],
            dtype=torch.int64,
            device="cpu")

    @staticmethod
    def allocate_allreduce_fusion_workspace(
            mapping: Mapping,
            size: int) -> Tuple[List[IpcMemory], "torch.tensor"]:
        import torch
        is_p2p_supported = can_access_peer(mapping)
        ipc_buffers_size = size * mapping.tp_size
        ipc_buffers = IpcMemory(mapping, ipc_buffers_size, is_p2p_supported)
        ipc_barriers = IpcMemory(mapping, 256 * mapping.tp_size,
                                 is_p2p_supported)
        lamport_buffers_size = size * mapping.tp_size
        lamport_buffers = IpcMemory(mapping, 3 * lamport_buffers_size,
                                    is_p2p_supported)
        if is_p2p_supported:
            lamport_initialize(
                lamport_buffers.local_ptr,
                3 * lamport_buffers_size,
            )
        flag_buffer = torch.tensor([0, 0, 0, lamport_buffers_size, 0],
                                   dtype=torch.int,
                                   device="cuda")
        buffers = [ipc_buffers, ipc_barriers, lamport_buffers, flag_buffer]

        return buffers, torch.tensor(
            ipc_buffers.serialize() + ipc_barriers.serialize() +
            lamport_buffers.serialize() + [flag_buffer.data_ptr()],
            dtype=torch.int64,
            device="cuda")


custom_all_reduce_helper = None


def init_all_reduce_helper():
    global custom_all_reduce_helper
    custom_all_reduce_helper = CustomAllReduceHelper()


def current_all_reduce_helper():
    global custom_all_reduce_helper
    assert custom_all_reduce_helper is not None, "You must call `init_all_reduce_helper` first"
    return custom_all_reduce_helper


class AllReduceParams():

    def __init__(self,
                 strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
                 fusion_op: AllReduceFusionOp = AllReduceFusionOp.NONE,
                 bias: Optional[torch.Tensor] = None,
                 residual: Optional[torch.Tensor] = None,
                 norm_weight: Optional[torch.Tensor] = None,
                 scale: Optional[torch.Tensor] = None,
                 norm_pre_residual_weight: Optional[torch.Tensor] = None,
                 eps: float = 1e-06,
                 enable_allreduce: bool = True):
        self.strategy = strategy
        self.fusion_op = fusion_op
        self.bias = bias
        self.residual = residual
        self.norm_weight = norm_weight
        self.scale = scale
        self.norm_pre_residual_weight = norm_pre_residual_weight
        self.eps = eps
        # For torch path only, has no effect on TRT path
        self.enable_allreduce = enable_allreduce
        assert fusion_op == AllReduceFusionOp.NONE.value or (residual
                                                             is not None)

    def has_affine(self):
        return 1 if self.norm_weight is not None else 0

    def has_bias(self):
        return 1 if self.bias is not None else 0

    def has_scale(self):
        return 1 if self.scale is not None else 0


# layer might be removed later
_thread_local = threading.local()

def get_allreduce_workspace(mapping: Mapping) -> torch.LongTensor:
    if not hasattr(_thread_local, f'allreduce_workspaces_{mapping.pp_rank}'):
        setattr(_thread_local, f'allreduce_workspaces_{mapping.pp_rank}', {})

    allreduce_workspaces = getattr(_thread_local,
                                   f'allreduce_workspaces_{mapping.pp_rank}')
    if mapping not in allreduce_workspaces:
        ipc_buffers, workspace = CustomAllReduceHelper.allocate_allreduce_fusion_workspace(
            mapping,
            CustomAllReduceHelper.max_workspace_size_auto(
                mapping.tp_size, support_deterministic=False),
        )
        allreduce_workspaces[mapping] = (ipc_buffers, workspace)
    return allreduce_workspaces[mapping][1]


def userbuffers_allreduce_finalize(
        input: torch.Tensor,
        force_applying_finalize: bool = False) -> torch.Tensor:
    output = torch.ops.trtllm.userbuffers_allreduce_finalize(
        input, force_applying_finalize)
    return output

class AllReduce(nn.module):

    def __init__(self,
                 mapping: Mapping,
                 strategy: AllReduceStrategy = AllReduceStrategy.AUTO):
        super().__init__()
        """
        AllReduce is a module that performs an all-reduce operation on a tensor.

        Args:
            mapping (Mapping):  The parallel mapping config.
            strategy (AllReduceStrategy):
                Three types of all-reduce strategies are supported:
                - UB: AllReduce uses user-buffer based all-reduce kernel. Supported ops:
                    - RESIDUAL_RMS_NORM
                    - RESIDUAL_RMS_NORM_QUANT_FP8
                    - RESIDUAL_RMS_NORM_QUANT_NVFP4

                - NCCL: AllReduce delegates all-reduce to NCCL MIN_LATENCY mode kernel. Supported ops:
                    - NONE (AllReduce only)
                    - RESIDUAL_RMS_NORM

                - MIN_LATENCY: AllReduce uses MIN_LATENCY mode kernel. Supported ops:
                    - NONE (AllReduce only)
                    - RESIDUAL_RMS_NORM
                    - RESIDUAL_RMS_NORM_QUANT_FP8
                    - RESIDUAL_RMS_NORM_QUANT_NVFP4
                    - RESIDUAL_RMS_NORM_OUT_QUANT_FP8
                    - RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4

                - AUTO: AUTO chooses between NCCL and MIN_LATENCY mode based on a heuristic policy.

        Note:
            For the reference implementation for each pattern, please refer to the following unit test:
            https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/unittest/_torch/multi_gpu/test_allreduce.py
        """

        self.mapping = mapping
        self.workspace = None
        self.strategy = strategy

        if self.mapping.tp_size > 1:
            self.workspace = get_allreduce_workspace(self.mapping)

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        '''
        The input tensors in the different ranks must have the same shape.
        The output tensor will have that same shape with the input tensor.
        The output tensor will be replicated among the TP group.
        Note that it is not an in-place operation like torch.distributed.all_reduce.

        That operation is implemented using a torch op that wraps the NCCL all-reduce
        collective operation and custom one-shot/two-shot allreduce kernels. See
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
        for details.

        Args:
            input (Tensor): The input tensor.
            all_reduce_params (AllReduceParams): The parameters for the fused ops into the allreduce op.
        Returns:
            A tensor lists with different tensor outptus according to the fusion_op.
            NONE: [hidden_states]
            RESIDUAL_RMS_NORM: [hidden_states, residual]
            RESIDUAL_RMS_NORM_QUANT_FP8: [norm_quant, residual]
            RESIDUAL_RMS_NORM_OUT_QUANT_FP8: [norm, norm_quant, residual]
            RESIDUAL_RMS_NORM_QUANT_NVFP4: [norm_quant_fp4, scale_factor, residual]
            RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4: [norm, norm_quant_fp4, scale_factor, residual]
        '''
        if self.mapping.tp_size == 1 or (all_reduce_params is not None
                                         and all_reduce_params.enable_allreduce
                                         == False):
            return input

        # Assume using no fusion allreduce here
        if all_reduce_params is None:
            all_reduce_params = AllReduceParams()

        output = torch.ops.trtllm.allreduce(
            input=input,
            residual=all_reduce_params.residual,
            norm_weight=all_reduce_params.norm_weight,
            scale=all_reduce_params.scale,
            bias=all_reduce_params.bias,
            workspace=self.workspace,
            group=self.mapping.tp_group,
            strategy=self.strategy,
            op=all_reduce_params.fusion_op,
            eps=all_reduce_params.eps,
        )

        return output if len(output) > 1 else output[0]


class MoEAllReduce(nn.module):

    def __init__(self, mapping: Mapping):
        """
        MoEAllReduce is a module that performs a specific fused MoE reduction
        followed by a regular AR + RMS norm.

        Args:
            mapping (Mapping):  The parallel mapping config.

        Notes:
            Support pattern: MoE Reduction + Add + AR + ADD_RMS, see this torch reference implementation:
            expert_reduction = torch.sum(active_experts_token_input *
                                        scale.unsqueeze(-1),
                                        dim=0)
            output_add = expert_reduction + shared_expert_output
            output_residual = output_add + residual
            output_hidden_states = rms_norm(output_residual, norm_weight, eps)
        """
        super().__init__()
        self.mapping = mapping
        self.workspace = get_allreduce_workspace(self.mapping)

    def forward(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        device_num_experts: torch.Tensor,
        scale_input: torch.Tensor,
        active_experts_token_input: torch.Tensor,
        token_input: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Args:
            residual: residual tensor
            norm_weight: RMS norm weight
            device_num_experts: number of experts per device
            scale_input: experts to token score
            active_experts_token_input: per token per expert input
            token_input: per token input, shared expert output
            eps: epsilon for RMSNorm

        Output:
            hidden_states: hidden_states of the model
            residual: residual tensor
        """
        return torch.ops.trtllm.moe_allreduce(
            residual=residual,
            norm_weight=norm_weight,
            device_num_experts=device_num_experts,
            scale_input=scale_input,
            active_experts_token_input=active_experts_token_input,
            token_input=token_input,
            workspace=self.workspace,
            rank=self.mapping.tp_rank,
            nranks=self.mapping.tp_size,
            eps=eps,
        )