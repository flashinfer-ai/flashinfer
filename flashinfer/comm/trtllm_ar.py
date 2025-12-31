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

import functools
import logging
from ctypes import c_void_p, cast
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union
from typing_extensions import deprecated

from flashinfer.comm.mnnvl import CommBackend, SymmDeviceMemory, TorchDistBackend
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ..jit.comm import gen_trtllm_comm_module
from ..utils import register_custom_op, round_up
from .cuda_ipc import create_shared_buffer, cudart, free_shared_buffer


class AllReduceStrategyType:
    # NOTE: for trtllm_custom_all_reduce
    NCCL = 0
    MIN_LATENCY = 1
    UB = 2
    AUTO = 3
    ONESHOT = 4
    TWOSHOT = 5
    LOWPRECISION = 6


class AllReduceStrategyConfig:
    # NOTE: for trtllm_custom_all_reduce
    USE_MEMCPY = 1 << 0
    PUSH_MODE = 1 << 1


class AllReduceFusionOp:
    # NOTE: for trtllm_custom_all_reduce
    NONE = 0
    RESIDUAL_RMS_NORM = 1
    LAST_PROCESS_FOR_UB = 2
    RESIDUAL_RMS_PREPOST_NORM = 3
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7
    MOE_ALLREDUCE_RESIDUAL_RMS_NORM = 8
    MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM = 9


class AllReduceFusionPattern:
    # NOTE: for trtllm_allreduce_fusion
    # Basic all-reduce pattern
    kAllReduce = 0
    # All-reduce followed by residual add and RMS norm
    kARResidualRMSNorm = 1
    # All-reduce followed by residual add, RMS norm and FP8 quantization
    kARResidualRMSNormFP8Quant = 2
    # All-reduce followed by residual add, RMS norm and FP4 quantization
    kARResidualRMSNormFP4Quant = 3
    # All-reduce followed by residual add, RMS norm and FP8 quantization, with norm output
    kARResidualRMSNormOutFP8Quant = 4
    # All-reduce followed by residual add, RMS norm and FP4 quantization, with norm output
    kARResidualRMSNormOutFP4Quant = 5


class QuantizationSFLayout:
    # Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
    # blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8
    # values. The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and
    # columns to the nearest multiple of 4.
    #
    # The scale factor block rows map to data block rows in an interleaved pattern:
    # For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
    # Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
    #
    # Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
    SWIZZLED_128x4 = 0
    SWIZZLED_8x4 = 1
    # Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen
    # kernels standard.
    LINEAR = 2


@functools.cache
def get_trtllm_comm_module():
    module = gen_trtllm_comm_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_lamport_initialize", mutates_args=["buffer"]
    )
    def trtllm_lamport_initialize(
        buffer_ptr: int, size: int, dtype: torch.dtype
    ) -> None:
        module.trtllm_lamport_initialize(buffer_ptr, size, dtype)

    @register_custom_op(
        "flashinfer::trtllm_lamport_initialize_all",
        mutates_args=["buffer_0_ptr", "buffer_1_ptr", "buffer_2_ptr", "size", "dtype"],
    )
    def trtllm_lamport_initialize_all(
        buffer_0_ptr: int,
        buffer_1_ptr: int,
        buffer_2_ptr: int,
        size: int,
        dtype: torch.dtype,
    ) -> None:
        module.trtllm_lamport_initialize_all(
            buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype
        )

    @deprecated(
        "trtllm_create_ipc_workspace_for_all_reduce and trtllm_custom_all_reduce are deprecated and will be removed in the next major bump, use allreduce.py instead."
    )
    @register_custom_op(
        "flashinfer::trtllm_custom_all_reduce",
        mutates_args=[
            "inp",
            "out",
            "tp_size",
            "tp_rank",
            "token_num",
            "fusion_op_code",
            "strategy_code",
            "config_code",
            "launch_with_pdl",
            "flag_value",
            "peer_comm_buffer_ptrs",
            "peer_barrier_ptrs_in",
            "peer_barrier_ptrs_out",
            "bias",
            "residual",
            "weight",
            "weight_pre_residual_norm",
            "eps",
            "intermediate_buffer",
            "lamport_peer_comm_buffer_ptrs_0",
            "lamport_peer_comm_buffer_ptrs_1",
            "lamport_peer_comm_buffer_ptrs_2",
        ],
    )
    def trtllm_custom_all_reduce(
        inp: torch.Tensor,
        out: torch.Tensor,
        tp_size: int,
        tp_rank: int,
        token_num: int,
        fusion_op_code: AllReduceFusionOp,
        strategy_code: AllReduceStrategyType,
        config_code: AllReduceStrategyConfig,
        launch_with_pdl: bool,
        flag_value: int,
        peer_comm_buffer_ptrs: torch.Tensor,
        peer_barrier_ptrs_in: torch.Tensor,
        peer_barrier_ptrs_out: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        weight_pre_residual_norm: Optional[torch.Tensor],
        eps: Optional[float],
        intermediate_buffer: Optional[torch.Tensor],
        lamport_peer_comm_buffer_ptrs_0: Optional[torch.Tensor],
        lamport_peer_comm_buffer_ptrs_1: Optional[torch.Tensor],
        lamport_peer_comm_buffer_ptrs_2: Optional[torch.Tensor],
    ) -> None:
        module.trtllm_custom_all_reduce(
            inp,
            out,
            tp_size,
            tp_rank,
            token_num,
            fusion_op_code,
            strategy_code,
            config_code,
            launch_with_pdl,
            flag_value,
            peer_comm_buffer_ptrs,
            peer_barrier_ptrs_in,
            peer_barrier_ptrs_out,
            bias,
            residual,
            weight,
            weight_pre_residual_norm,
            eps,
            intermediate_buffer,
            lamport_peer_comm_buffer_ptrs_0,
            lamport_peer_comm_buffer_ptrs_1,
            lamport_peer_comm_buffer_ptrs_2,
        )

    @register_custom_op(
        "flashinfer::trtllm_allreduce_fusion",
        mutates_args=[
            "allreduce_in",
            "world_size",
            "world_rank",
            "token_num",
            "hidden_dim",
            "workspace_ptrs",
            "launch_with_pdl",
            "use_oneshot",
            "trigger_completion_at_end",
            "fp32_acc",
            "pattern_code",
            "allreduce_out",
            "residual_in",
            "residual_out",
            "norm_out",
            "quant_out",
            "scale_out",
            "rms_gamma",
            "rms_eps",
            "scale_factor",
            "layout_code",
        ],
    )
    def trtllm_allreduce_fusion(
        allreduce_in: torch.Tensor,
        world_size: int,
        world_rank: int,
        token_num: int,
        hidden_dim: int,
        workspace_ptrs: torch.Tensor,
        launch_with_pdl: bool,
        use_oneshot: bool,
        trigger_completion_at_end: bool,
        fp32_acc: bool,
        pattern_code: AllReduceFusionPattern,
        allreduce_out: Optional[torch.Tensor],
        residual_in: Optional[torch.Tensor],
        residual_out: Optional[torch.Tensor],
        norm_out: Optional[torch.Tensor],
        quant_out: Optional[torch.Tensor],
        scale_out: Optional[torch.Tensor],
        rms_gamma: Optional[torch.Tensor],
        rms_eps: Optional[float],
        scale_factor: Optional[Union[torch.Tensor, float]],
        layout_code: Optional[QuantizationSFLayout],
    ) -> None:
        module.trtllm_allreduce_fusion(
            allreduce_in,
            world_size,
            world_rank,
            token_num,
            hidden_dim,
            workspace_ptrs,
            launch_with_pdl,
            use_oneshot,
            trigger_completion_at_end,
            fp32_acc,
            pattern_code,
            allreduce_out,
            residual_in,
            residual_out,
            norm_out,
            quant_out,
            scale_out,
            rms_gamma,
            rms_eps,
            scale_factor,
            layout_code,
        )

    @register_custom_op(
        "flashinfer::trtllm_moe_allreduce_fusion",
        mutates_args=[
            "out",
            "tp_size",
            "tp_rank",
            "token_num",
            "hidden_dim",
            "workspace_ptrs",
            "launch_with_pdl",
            "residual_in",
            "rms_gamma",
            "rms_eps",
            "scale_factor",
            "moe_reduction_device_num_experts",
            "moe_reduction_scale_input",
            "moe_reduction_active_experts_token_input",
            "moe_reduction_token_input",
            "layout_code",
            "allreduce_out",
            "residual_out",
            "norm_out",
            "quant_out",
            "scale_out",
        ],
    )
    def trtllm_moe_allreduce_fusion(
        world_size: int,
        world_rank: int,
        token_num: int,
        hidden_dim: int,
        workspace_ptrs: torch.Tensor,
        launch_with_pdl: bool,
        residual_in: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float,
        scale_factor: float,
        moe_reduction_device_num_experts: int,
        moe_reduction_scale_input: torch.Tensor,
        moe_reduction_active_experts_token_input: torch.Tensor,
        moe_reduction_token_input: torch.Tensor,
        layout_code: Optional[QuantizationSFLayout],
        moe_allreduce_out: Optional[torch.Tensor],
        residual_out: Optional[torch.Tensor],
        norm_out: Optional[torch.Tensor],
        quant_out: Optional[torch.Tensor],
        scale_out: Optional[torch.Tensor],
    ) -> None:
        module.trtllm_moe_allreduce_fusion(
            world_size,
            world_rank,
            token_num,
            hidden_dim,
            workspace_ptrs,
            launch_with_pdl,
            residual_in,
            rms_gamma,
            rms_eps,
            scale_factor,
            moe_reduction_device_num_experts,
            moe_reduction_scale_input,
            moe_reduction_active_experts_token_input,
            moe_reduction_token_input,
            layout_code,
            moe_allreduce_out,
            residual_out,
            norm_out,
            quant_out,
            scale_out,
        )

    @register_custom_op(
        "flashinfer::trtllm_moe_finalize_allreduce_fusion",
        mutates_args=["residual_out", "norm_out"],
    )
    def trtllm_moe_finalize_allreduce_fusion(
        allreduce_in: torch.Tensor,
        residual_in: torch.Tensor,
        norm_weight: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        norm_out: torch.Tensor,
        residual_out: torch.Tensor,
        launch_with_pdl: bool,
        workspace: torch.Tensor,
        world_rank: int,
        world_size: int,
        eps: float,
        shared_expert_output: Optional[torch.Tensor],
        expert_scale_factor: Optional[torch.Tensor],
    ) -> None:
        module.trtllm_moe_finalize_allreduce_fusion(
            allreduce_in,
            residual_in,
            norm_weight,
            expanded_idx_to_permuted_idx,
            norm_out,
            residual_out,
            launch_with_pdl,
            workspace,
            world_rank,
            world_size,
            eps,
            shared_expert_output,
            expert_scale_factor,
        )

    return SimpleNamespace(
        trtllm_lamport_initialize=trtllm_lamport_initialize,
        trtllm_lamport_initialize_all=trtllm_lamport_initialize_all,
        trtllm_custom_all_reduce=trtllm_custom_all_reduce,
        trtllm_allreduce_fusion=trtllm_allreduce_fusion,
        trtllm_moe_allreduce_fusion=trtllm_moe_allreduce_fusion,
        trtllm_moe_finalize_allreduce_fusion=trtllm_moe_finalize_allreduce_fusion,
    )


# NOTE(Yingyi): The customAllReduce and allReduceFusion require different buffer size
# since allreduceFusion kernels are an improved implementation
OneShotMaxToken = 128
MAX_ALL_REDUCE_BLOCKS = 24
LamportTokenNumThreshold = 16


@deprecated(
    "trtllm_create_ipc_workspace_for_all_reduce and trtllm_custom_all_reduce are deprecated and will be removed in the next major bump, use allreduce.py instead."
)
def trtllm_create_ipc_workspace_for_all_reduce(
    rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    group: Optional[ProcessGroup] = None,
) -> List[List[int]]:
    """
    Parameters:
    - rank: the rank of the current process.
    - tp_size: the size of the process group.
    - max_token_num: the maximum number of tokens in a sequence.
    - hidden_dim: the dimension of the hidden states.
    - group: the process group to use.

    Note:
    This function is used to create a workspace for all reduce.
    The workspace is a list of IPC handles.
    The workspace should be initialized before calling trtllm_custom_all_reduce.
    The workspace should be destroyed after calling trtllm_custom_all_reduce.
    The workspace can be reused for multiple all reduce calls under the same configuration.

    We would init 7 IPC buffers for trtllm_custom_all_reduce.
    They are sized as follows:
    [buffer_size, buffer_size, flag_size, flag_size, lamport_buffer_size, lamport_buffer_size, lamport_buffer_size]
    where:
    - buffer_size: tp_size * max_token_num * hidden_dim * sizeof(float) * (maxBeamWidth)
    - flag_size: (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t) * tp_size * 2
    - lamport_buffer_size: tp_size * LamportTokenNumThreshold * tp_size * hidden_dim * sizeof(half)

    They are for:
    ipcHandles[0] - peer_comm_buffer_ptrs
    ipcHandles[2] - peer_barrier_ptrs_in
    ipcHandles[3] - peer_barrier_ptrs_out
    ipcHandles[4] - lamport_peer_comm_buffer_ptrs[0:tp_size]
    ipcHandles[5] - lamport_peer_comm_buffer_ptrs[tp_size:tp_size * 2]
    ipcHandles[6] - lamport_peer_comm_buffer_ptrs[tp_size * 2:tp_size * 3]

    We use tp_size and world_size here interchangeably (customAllReduce).

    Reference: trtllm, cpp/tests/unit_tests/kernels/allReduce/allReduceKernelTest.cu, Workspace init
    """

    buffer_size = tp_size * max_token_num * hidden_dim * 4
    FLAG_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * 4
    flag_size = FLAG_SIZE * tp_size * 2
    lamport_buffer_size = tp_size * LamportTokenNumThreshold * tp_size * hidden_dim * 2

    ipc_handles = list()

    for size in [
        buffer_size,
        buffer_size,
        flag_size,
        flag_size,
        lamport_buffer_size,
        lamport_buffer_size,
        lamport_buffer_size,
    ]:
        # all sizes should be aligned to 1LU << 21 bytes (2MB)
        aligned_size = round_up(size, 1 << 21)
        ipc_handles.append(create_shared_buffer(aligned_size, group))

    print(
        f"rank {rank} allocated ipc_handles: {[[hex(handle) for handle in sublist] for sublist in ipc_handles]}"
    )

    trtllm_lamport_initialize_all(
        ipc_handles[4][rank],
        ipc_handles[5][rank],
        ipc_handles[6][rank],
        lamport_buffer_size // 2,
        torch.float16,
    )

    dist.barrier(group=group)  # must sync after create_workspace

    return ipc_handles


def trtllm_destroy_ipc_workspace_for_all_reduce(
    workspace: List[List[int]], group: Optional[ProcessGroup] = None
) -> None:
    """
    Note:
    This function is used to destroy a workspace for all reduce.
    The workspace is a list of IPC handles.
    The workspace should be destroyed after calling trtllm_custom_all_reduce.
    The workspace can be reused for multiple all reduce calls under the same configuration.
    """

    for ipc_handle in workspace:
        free_shared_buffer(ipc_handle, group)


BarrierFlagCount = 256

MAX_COMM_SIZE = 2147483647 & ~((1 << 21) - 1)  # MAX_INT32 rounded down to 2MB


@deprecated(
    "use the unified API allreduce.py instead. It will internally call trtllm_create_ipc_workspace_for_all_reduce_fusion."
)
def trtllm_create_ipc_workspace_for_all_reduce_fusion(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    use_fp32_lamport: bool = False,
    group: Optional[ProcessGroup] = None,
    create_metadata: bool = False,
    comm_backend: Optional[CommBackend] = None,
    use_symm_dev_mem: bool = False,
) -> Union[
    Tuple[List[List[int]], torch.Tensor],
    Tuple[List[List[int]], torch.Tensor, dict],
    Tuple[List[List[int]], torch.Tensor, List[SymmDeviceMemory], dict],
]:
    """
    Parameters:
    - tp_rank: the rank of the current process.
    - tp_size: the size of the process group.
    - max_token_num: the maximum number of tokens in a sequence.
    - hidden_dim: the dimension of the hidden states.
    - use_fp32_lamport: if True, we will use fp32 datatype in allreduce fusion.
    - group: the process group to use.
    - create_metadata: if True, return metadata dict as third element (default: False).
    - comm_backend: the communication backend to use.
    - use_symm_dev_mem: if True, we will use symmetric device memory for the workspace.

    Returns:
    - If create_metadata=False: (ipc_handles, workspace_tensor)
    - If create_metadata=True: and use_symm_dev_mem=False: (ipc_handles, workspace_tensor, metadata)
      where metadata contains: tp_rank, tp_size, max_token_num, hidden_dim,
      use_fp32_lamport, buffer_size, flag_size, lamport_comm_size, lamport_buffer_size
    - If create_metadata=True: and use_symm_dev_mem=True: (ipc_handles, workspace_tensor, mem_handles,metadata)
      where metadata contains: tp_rank, tp_size, max_token_num, hidden_dim,
      use_fp32_lamport, buffer_size, flag_size, lamport_comm_size, lamport_buffer_size
      and mem_handles is a list of SymmDeviceMemory objects.

    Note: The optional parameters make the API clunky at this time. This will be refactored in the future, at the cost of backward compatibility, where the default behavior will be
    create_metadata=True and use_symm_dev_mem=True.

    Note:
    We would init 3 IPC buffers for trtllm_custom_all_reduce_fusion.
    They are sized as follows:
    [buffer_size, flag_size, lamport_buffer_size * 3]
    where:
    - buffer_size: tp_size * max_token_num * hidden_dim * sizeof(half)
    - flag_size: tp_size * BarrierFlagCount * sizeof(int)
    - lamport_buffer_size: tp_size * max_token_num * tp_size * hidden_dim * sizeof(half)
      where sizeof(elem) = 2 (fp16/bf16) or 4 (fp32 when use_fp32_lamport=True)
    The workspace is passed as workspace field in AllReduceFusionParams.

    We use tp_size and world_size here interchangeably (allReduceFusion).

    Reference: trtllm, cpp/tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.cu, Workspace init
    """

    if comm_backend is None and use_symm_dev_mem:
        comm_backend = TorchDistBackend(group=group)

    # No need to support all variations. In the future we only support create_metadata=True and use_symm_dev_mem=True.
    if use_symm_dev_mem and not create_metadata:
        raise ValueError("use_symm_dev_mem is only supported when create_metadata=True")

    buffer_size = tp_size * max_token_num * hidden_dim * 2
    flag_size = tp_size * BarrierFlagCount * 4
    # lamport_comm_size = tp_size * max(max_token_num, OneShotMaxToken) * hidden_dim * 2
    # enable larger workspace for cases > OneShotMaxToken
    lamport_comm_size = (
        tp_size * max_token_num * hidden_dim * 2
        if not use_fp32_lamport
        else tp_size * max_token_num * hidden_dim * 4
    )
    if lamport_comm_size > MAX_COMM_SIZE:
        logging.warning(
            f"warning: lamport_comm_size {lamport_comm_size} is greater than MAX_COMM_SIZE {MAX_COMM_SIZE}, set to MAX_COMM_SIZE"
        )
        lamport_comm_size = MAX_COMM_SIZE

    lamport_buffer_size = lamport_comm_size * 3

    # we should init 3 buffers for all reduce fusion:
    # [buffer_size, flag_size, lamport_buffer_size]

    ipc_handles: List[List[int]] = list()
    mem_handles: List[SymmDeviceMemory] = list()
    for size in [buffer_size, flag_size, lamport_buffer_size]:
        # todo(review): confirm we need this alignment
        # all sizes should be aligned to 1LU << 21 bytes (2MB)
        aligned_size = round_up(size, 1 << 21)

        if not use_symm_dev_mem:
            ipc_handles.append(create_shared_buffer(aligned_size, group))
        else:
            symm_mem = SymmDeviceMemory(
                aligned_size,
                tp_size,
                tp_rank,
                torch.device("cuda", tp_rank).index,
                comm_backend,
                enable_multicast=False,
                allocate_signal_pads=False,
            )
            ipc_handles.append(symm_mem.uc_ptrs)
            mem_handles.append(symm_mem)

    print(
        f"rank {tp_rank} allocated ipc_handles: {[[hex(handle) for handle in sublist] for sublist in ipc_handles]}"
    )

    # Initialize lamport buffer
    aligned_lamport_buffer_size = round_up(lamport_buffer_size, 1 << 21)
    if use_fp32_lamport:
        trtllm_lamport_initialize(
            ipc_handles[2][tp_rank], aligned_lamport_buffer_size // 4, torch.float32
        )
    else:
        trtllm_lamport_initialize(
            ipc_handles[2][tp_rank], aligned_lamport_buffer_size // 2, torch.float16
        )

    # initialize workspace
    workspace = list()
    # add ipc handles to workspace
    for ipc_handle in ipc_handles:
        for rank in range(tp_size):
            workspace.append(ipc_handle[rank])

    # add flags to workspace
    """
    NOTE:
    The flags are for the lamport communication states.
    atomic flag read counter: kernel_flag_ptr[0] = 0;
    non-lamport flag: kernel_flag_ptr[1] = 0;
    lamport flag: kernel_flag_ptr[2] = 0;
    lamport triple buffer offset: kernel_flag_ptr[3] = lamport_comm_size;
    lamport clear size: kernel_flag_ptr[4] = 0;
    """
    # malloc cuda memory of int32_t * 5
    flag_ptr = cudart.cudaMalloc(5 * 4)
    # initialize the flag to [0,0,0,lamport_comm_size,0]
    cudart.cudaMemset(flag_ptr, 0, 5 * 4)
    # Set flag_ptr[3] = lamport_comm_size
    lamport_comm_size_bytes = lamport_comm_size.to_bytes(4, byteorder="little")
    cudart.cudaMemcpy(
        c_void_p(flag_ptr.value + 3 * 4), cast(lamport_comm_size_bytes, c_void_p), 4
    )
    print("set flag_ptr[3] = lamport_comm_size: ", lamport_comm_size)
    # add flag_ptr to workspace
    workspace.append(flag_ptr.value)

    for i in range(len(workspace)):
        print(f"Rank {tp_rank} workspace[{i}] {hex(workspace[i])}")

    # Store workspace pointers in device tensor
    workspace_tensor = torch.tensor(
        workspace, dtype=torch.int64, device=torch.device("cuda")
    )

    if use_symm_dev_mem:
        comm_backend.barrier()  # must sync after create_workspace
    else:
        dist.barrier(group=group)

    if create_metadata:
        metadata = {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "max_token_num": max_token_num,
            "hidden_dim": hidden_dim,
            "use_fp32_lamport": use_fp32_lamport,
            "buffer_size": buffer_size,
            "flag_size": flag_size,
            "lamport_comm_size": lamport_comm_size,
            "lamport_buffer_size": lamport_buffer_size,
        }
        if use_symm_dev_mem:
            return ipc_handles, workspace_tensor, mem_handles, metadata
        else:
            return ipc_handles, workspace_tensor, metadata

    else:
        return ipc_handles, workspace_tensor


def trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
    workspace: List[List[int]], group: Optional[ProcessGroup] = None
) -> None:
    """
    Parameters:
    - workspace: the workspace to destroy.
    - group: the process group to use.

    Note:
    This function is used to destroy a workspace for all reduce fusion.
    The workspace is a list of IPC handles.
    The workspace should be destroyed after calling trtllm_custom_all_reduce_fusion.
    The workspace can be reused for multiple all reduce fusion calls under the same configuration.
    """

    for ipc_handle in workspace:
        free_shared_buffer(ipc_handle, group)


# allReduce fused quant utils
def compute_fp4_swizzled_layout_sf_size(total_row, total_column):
    """
    Helper function to compute the padded size of the fp4 swizzled layout.

    Parameters:
    - total_row: the total number of rows.
    - total_column: the total number of columns.
    """

    def pad_up(x, y):
        return ((x + y - 1) // y) * y

    padded_row = pad_up(total_row, 128)
    padded_column = pad_up(total_column, 4)
    return padded_row * padded_column


def trtllm_lamport_initialize(buffer_ptr: int, size: int, dtype: torch.dtype) -> None:
    get_trtllm_comm_module().trtllm_lamport_initialize(buffer_ptr, size, dtype)


def trtllm_lamport_initialize_all(
    buffer_0_ptr: int,
    buffer_1_ptr: int,
    buffer_2_ptr: int,
    size: int,
    dtype: torch.dtype,
) -> None:
    """
    Initialize 3 lamport buffers by negative zero.

    Parameters:
    - buffer_0_ptr: the pointer to the first buffer.
    - buffer_1_ptr: the pointer to the second buffer.
    - buffer_2_ptr: the pointer to the third buffer.
    - size: the size of the buffer.
    - dtype: the data type of the buffer.
    """

    get_trtllm_comm_module().trtllm_lamport_initialize_all(
        buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype
    )


@deprecated(
    "trtllm_create_ipc_workspace_for_all_reduce and trtllm_custom_all_reduce are deprecated, use trtllm_create_ipc_workspace_for_all_reduce_fusion and trtllm_allreduce_fusion instead"
)
def trtllm_custom_all_reduce(
    inp: torch.Tensor,
    out: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    token_num: int,
    fusion_op_code: AllReduceFusionOp,
    strategy_code: AllReduceStrategyType,
    config_code: AllReduceStrategyConfig,
    launch_with_pdl: bool,
    flag_value: int,
    peer_comm_buffer_ptrs: torch.Tensor,
    peer_barrier_ptrs_in: torch.Tensor,
    peer_barrier_ptrs_out: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    weight_pre_residual_norm: Optional[torch.Tensor],
    eps: Optional[float],
    intermediate_buffer: Optional[torch.Tensor],
    lamport_peer_comm_buffer_ptrs_0: Optional[torch.Tensor],
    lamport_peer_comm_buffer_ptrs_1: Optional[torch.Tensor],
    lamport_peer_comm_buffer_ptrs_2: Optional[torch.Tensor],
) -> None:
    """
    Parameters:
    - inp: the input tensor. [token_num, hidden_dim]
    - out: the output tensor. [token_num, hidden_dim]
    - tp_size: the size of the process group.
    - tp_rank: the rank of the current process.
    - token_num: the number of tokens in the sequence.
    - fusion_op_code: the fusion operation code.
    - strategy_code: the strategy code.
    - config_code: the config code.
    - launch_with_pdl: whether to launch with pdl.
    - flag_value: the flag value.
    - peer_comm_buffer_ptrs: the peer communication buffer pointers.
    - peer_barrier_ptrs_in: the peer barrier pointers in.
    - peer_barrier_ptrs_out: the peer barrier pointers out.
    - bias: the bias tensor. [hidden_dim]
    - residual: the residual tensor. [token_num, hidden_dim]
    - weight: the weight tensor. [hidden_dim]
    - weight_pre_residual_norm: the weight pre residual norm tensor. [hidden_dim]
    - eps: the epsilon value.
    - intermediate_buffer: the intermediate buffer tensor.
    - lamport_peer_comm_buffer_ptrs_0: the lamport peer communication buffer pointers 0.
    - lamport_peer_comm_buffer_ptrs_1: the lamport peer communication buffer pointers 1.
    - lamport_peer_comm_buffer_ptrs_2: the lamport peer communication buffer pointers 2.
    """

    get_trtllm_comm_module().trtllm_custom_all_reduce(
        inp,
        out,
        tp_size,
        tp_rank,
        token_num,
        fusion_op_code,
        strategy_code,
        config_code,
        launch_with_pdl,
        flag_value,
        peer_comm_buffer_ptrs,
        peer_barrier_ptrs_in,
        peer_barrier_ptrs_out,
        bias,
        residual,
        weight,
        weight_pre_residual_norm,
        eps,
        intermediate_buffer,
        lamport_peer_comm_buffer_ptrs_0,
        lamport_peer_comm_buffer_ptrs_1,
        lamport_peer_comm_buffer_ptrs_2,
    )


# Heuristics based on all configs of trtllm_allreduce_fusion on B200.
# Empirically, the fusion pattern and fp32_acc are irrelevant to the decision.
_use_oneshot_heuristics: dict[int, int] = {
    2: 512,
    4: 64,
    8: 42,
}


def _should_use_oneshot(
    token_num: int, hidden_dim: int, dtype: torch.dtype, world_size: int
) -> bool:
    comm_size_mb = (
        token_num * hidden_dim * 2 * world_size * dtype.itemsize / 1024 / 1024
    )
    return comm_size_mb <= _use_oneshot_heuristics[world_size]


def check_trtllm_allreduce_fusion_workspace_metadata(
    token_num: int,
    hidden_dim: int,
    world_size: int,
    dtype: torch.dtype,
    metadata: dict,
) -> None:
    errors = []
    required_keys = ["max_token_num", "tp_size", "hidden_dim", "use_fp32_lamport"]
    for key in required_keys:
        if key not in metadata:
            errors.append(f"Workspace metadata is missing required key: {key}")
    if errors:
        error_msg = "Workspace metadata validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(error_msg)

    # world_size must match tp_size (flag size depends on it)
    if world_size != metadata["tp_size"]:
        errors.append(
            f"world_size ({world_size}) does not match workspace tp_size ({metadata['tp_size']}). "
            f"Workspace was created for tp_size={metadata['tp_size']}."
        )

    # token_num * hidden_dim must not exceed max_token_num * hidden_dim
    if token_num * hidden_dim > metadata["max_token_num"] * metadata["hidden_dim"]:
        errors.append(
            f"token_num ({token_num}) * hidden_dim ({hidden_dim}) exceeds workspace max_token_num ({metadata['max_token_num']}) * hidden_dim ({metadata['hidden_dim']}). "
            f"This may cause Illegal Memory Access."
        )

    # use_fp32_lamport must match
    if metadata["use_fp32_lamport"] != (dtype == torch.float32):
        errors.append(
            f"use_fp32_lamport ({metadata['use_fp32_lamport']}) does not match allreduce_in.dtype ({dtype}). "
            f"Workspace was created for use_fp32_lamport={metadata['use_fp32_lamport']}."
        )
    if errors:
        error_msg = "Workspace validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(error_msg)


@deprecated(
    "use the unified API allreduce.py instead. It will internally call trtllm_allreduce_fusion."
)
def trtllm_allreduce_fusion(
    allreduce_in: torch.Tensor,
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    trigger_completion_at_end: bool,
    fp32_acc: bool,
    pattern_code: AllReduceFusionPattern,
    use_oneshot: Optional[bool],
    allreduce_out: Optional[torch.Tensor],
    residual_in: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
    rms_gamma: Optional[torch.Tensor],
    rms_eps: Optional[float],
    scale_factor: Optional[Union[torch.Tensor, float]],
    layout_code: Optional[QuantizationSFLayout],
    metadata: Optional[dict] = None,
) -> None:
    """
    Parameters:
    - allreduce_in: the input tensor. [token_num, hidden_dim]
    - world_size: the size of the process group.
    - world_rank: the rank of the current process.
    - token_num: the number of tokens in the sequence.
    - hidden_dim: the dimension of the hidden states.
    - workspace_ptrs: the workspace pointers.
    - launch_with_pdl: whether to launch with pdl.
    - use_oneshot: whether to use oneshot. If None, internal heuristics will be used.
    - trigger_completion_at_end: whether to trigger completion at the end.
    - fp32_acc: whether to use fp32 accumulation.
    - pattern_code: the pattern code.
    - allreduce_out: the output tensor. [token_num, hidden_dim]
    - residual_in: the residual input tensor. [token_num, hidden_dim]
    - residual_out: the residual output tensor. [token_num, hidden_dim]
    - norm_out: the norm output tensor. [token_num, hidden_dim]
    - quant_out: the quant output tensor. [token_num, hidden_dim]
    - scale_out: the scale output tensor. Initialization referece: tests/comm/test_trtllm_allreduce_fusion.py
    - rms_gamma: the rms gamma tensor. [hidden_dim]
    - rms_eps: the rms epsilon value.
    - scale_factor: the scale factor. For cudaGraphs safety, it should be a tensor.
    - layout_code: the layout code.
    - metadata: optional workspace metadata dict from create_ipc_workspace_for_all_reduce_fusion.
                If provided, validates that token_num <= max_token_num, world_size == tp_size,
                and hidden_dim == workspace hidden_dim. Raises ValueError if validation fails.
    """

    # Validate against workspace metadata if provided
    if metadata is not None:
        check_trtllm_allreduce_fusion_workspace_metadata(
            token_num, hidden_dim, world_size, allreduce_in.dtype, metadata
        )

    if use_oneshot is None:
        use_oneshot = _should_use_oneshot(
            token_num, hidden_dim, allreduce_in.dtype, world_size
        )

    if not use_oneshot:
        assert token_num > world_size, "sequence length should be larger than tp_size"

    required_lamport_comm_size = (
        token_num * hidden_dim * 2 * world_size
        if allreduce_in.dtype != torch.float32
        else token_num * hidden_dim * 4 * world_size
    )

    if required_lamport_comm_size > MAX_COMM_SIZE and use_oneshot:
        logging.warning(
            f"required_lamport_comm_size {required_lamport_comm_size} is greater than MAX_COMM_SIZE {MAX_COMM_SIZE}. Cannot use oneshot in this case."
        )
        use_oneshot = False
    if scale_factor is not None:
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(torch.float32)
        else:
            scale_factor = torch.tensor(
                [scale_factor], dtype=torch.float32, device=allreduce_in.device
            )
    get_trtllm_comm_module().trtllm_allreduce_fusion(
        allreduce_in=allreduce_in,
        world_size=world_size,
        world_rank=world_rank,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=workspace_ptrs,
        launch_with_pdl=launch_with_pdl,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=pattern_code,
        allreduce_out=allreduce_out,
        residual_in=residual_in,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        scale_factor=scale_factor,
        layout_code=layout_code,
    )


def trtllm_moe_allreduce_fusion(
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: float,
    moe_reduction_device_num_experts: int,
    moe_reduction_scale_input: torch.Tensor,
    moe_reduction_active_experts_token_input: torch.Tensor,
    moe_reduction_token_input: torch.Tensor,
    layout_code: Optional[QuantizationSFLayout],
    moe_allreduce_out: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
) -> None:
    """
    Parameters:
    - world_size: the size of the process group.
    - world_rank: the rank of the current process.
    - token_num: the number of tokens in the sequence.
    - hidden_dim: the dimension of the hidden states.
    - workspace_ptrs: the workspace pointers.
    - launch_with_pdl: whether to launch with pdl.
    - residual_in: the residual input tensor. [token_num, hidden_dim]
    - rms_gamma: the rms gamma tensor. [hidden_dim]
    - rms_eps: the rms epsilon value.
    - scale_factor: the scale factor.
    - moe_reduction_device_num_experts: the number of experts.
    - moe_reduction_scale_input: the scale input tensor. [token_num, hidden_dim]
    - moe_reduction_active_experts_token_input: the active experts token input tensor. [token_num, hidden_dim]
    - moe_reduction_token_input: the token input tensor. [token_num, hidden_dim]
    - layout_code: the layout code.
    - moe_allreduce_out: the moe allreduce output tensor. [token_num, hidden_dim]
    - residual_out: the residual output tensor. [token_num, hidden_dim]
    - norm_out: the norm output tensor. [token_num, hidden_dim]
    - quant_out: the quant output tensor. [token_num // 4, hidden_dim], fp16/bf16 -> fp4
    - scale_out: the scale output tensor. Initialization referece: tests/comm/test_trtllm_moe_allreduce_fusion.py
    """

    required_lamport_comm_size = moe_reduction_token_input.numel() * 2 * world_size

    # Note: only one-shot is supported for moe allreduce fusion.
    if required_lamport_comm_size > MAX_COMM_SIZE:
        raise ValueError(
            f"required_lamport_comm_size {required_lamport_comm_size} is greater than MAX_COMM_SIZE {MAX_COMM_SIZE}. Cannot use oneshot in this case."
        )

    get_trtllm_comm_module().trtllm_moe_allreduce_fusion(
        world_size=world_size,
        world_rank=world_rank,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=workspace_ptrs,
        launch_with_pdl=launch_with_pdl,
        residual_in=residual_in,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        scale_factor=scale_factor,
        moe_reduction_device_num_experts=moe_reduction_device_num_experts,
        moe_reduction_scale_input=moe_reduction_scale_input,
        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
        moe_reduction_token_input=moe_reduction_token_input,
        layout_code=layout_code,
        moe_allreduce_out=moe_allreduce_out,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
    )


def trtllm_moe_finalize_allreduce_fusion(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    norm_weight: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    world_rank: int,
    world_size: int,
    eps: float,
    shared_expert_output: Optional[torch.Tensor],
    expert_scale_factor: Optional[torch.Tensor],
) -> None:
    """
    Parameters:
    - allreduce_in: the input tensor. [token_num, top_k, hidden_dim]
    - residual_in: the residual input tensor. [token_num, hidden_dim]
    - norm_weight: the norm weight tensor. [hidden_dim]
    - expanded_idx_to_permuted_idx: the expanded index to permuted index tensor. [token_num, top_k]
    - norm_out: the norm output tensor. [token_num, hidden_dim]
    - residual_out: the residual output tensor. [token_num, hidden_dim]
    - workspace_ptrs: the workspace pointers.
    - launch_with_pdl: whether to launch with pdl.
    - world_rank: the rank of the current process.
    - world_size: the size of the process group.
    - eps: the epsilon value.
    - shared_expert_output: the shared expert output tensor. [token_num, hidden_dim]
    - expert_scale_factor: the expert scale factor tensor. [token_num, top_k]
    """

    required_lamport_comm_size = allreduce_in.numel() * 2 * world_size

    # Note: only one-shot is supported for moe allreduce fusion.
    if required_lamport_comm_size > MAX_COMM_SIZE:
        raise ValueError(
            f"required_lamport_comm_size {required_lamport_comm_size} is greater than MAX_COMM_SIZE {MAX_COMM_SIZE}. Cannot use oneshot in this case."
        )

    get_trtllm_comm_module().trtllm_moe_finalize_allreduce_fusion(
        allreduce_in=allreduce_in,
        residual_in=residual_in,
        norm_weight=norm_weight,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        norm_out=norm_out,
        residual_out=residual_out,
        workspace=workspace_ptrs,
        launch_with_pdl=launch_with_pdl,
        world_rank=world_rank,
        world_size=world_size,
        eps=eps,
        shared_expert_output=shared_expert_output,
        expert_scale_factor=expert_scale_factor,
    )
