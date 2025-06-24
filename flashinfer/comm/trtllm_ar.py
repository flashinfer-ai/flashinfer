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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec, sm100a_nvcc_flags
from ..utils import register_custom_op
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


class FP4QuantizationSFLayout:
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
    SWIZZLED = 0
    # Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen
    # kernels standard.
    LINEAR = 1


def gen_trtllm_comm_module() -> JitSpec:
    return gen_jit_spec(
        "trtllm_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce_fusion.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_moe_allreduce_fusion.cu",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags,
    )


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
        scale_factor: Optional[float],
        layout_code: Optional[FP4QuantizationSFLayout],
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
        layout_code: Optional[FP4QuantizationSFLayout],
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


def trtllm_create_ipc_workspace_for_all_reduce(
    rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    group: Optional[ProcessGroup] = None,
) -> List[int]:
    """
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
        aligned_size = ((size + (1 << 21) - 1) >> 21) << 21
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
    workspace: List[int], group: Optional[ProcessGroup] = None
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


def trtllm_create_ipc_workspace_for_all_reduce_fusion(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    use_fp32_lamport: bool = False,
    group: Optional[ProcessGroup] = None,
) -> List[int]:
    """
    Note:
    We would init 3 IPC buffers for trtllm_custom_all_reduce_fusion.
    They are sized as follows:
    [buffer_size, flag_size, lamport_buffer_size * 3]
    where:
    - buffer_size: tp_size * max_token_num * hidden_dim * sizeof(half)
    - flag_size: tp_size * BarrierFlagCount * sizeof(int)
    - lamport_buffer_size: tp_size * max(max_token_num, OneShotMaxToken) * tp_size * hidden_dim * sizeof(half)

    The workspace is passed as workspace field in AllReduceFusionParams.

    We use tp_size and world_size here interchangeably (allReduceFusion).

    Reference: trtllm, cpp/tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.cu, Workspace init
    """

    buffer_size = tp_size * max_token_num * hidden_dim * 2
    flag_size = tp_size * BarrierFlagCount * 4
    # lamport_comm_size = tp_size * max(max_token_num, OneShotMaxToken) * hidden_dim * 2
    # enable larger workspace for cases > OneShotMaxToken
    lamport_comm_size = (
        tp_size * max_token_num * hidden_dim * 2
        if not use_fp32_lamport
        else tp_size * max_token_num * hidden_dim * 4
    )
    lamport_buffer_size = lamport_comm_size * 3

    # we should init 3 buffers for all reduce fusion:
    # [buffer_size, flag_size, lamport_buffer_size]

    ipc_handles = list()
    for size in [buffer_size, flag_size, lamport_buffer_size]:
        # todo(review): confirm we need this alignment
        # all sizes should be aligned to 1LU << 21 bytes (2MB)
        aligned_size = ((size + (1 << 21) - 1) >> 21) << 21
        ipc_handles.append(create_shared_buffer(aligned_size, group))

    print(
        f"rank {tp_rank} allocated ipc_handles: {[[hex(handle) for handle in sublist] for sublist in ipc_handles]}"
    )

    # Initialize lamport buffer
    if use_fp32_lamport:
        trtllm_lamport_initialize(
            ipc_handles[2][tp_rank], lamport_buffer_size // 4, torch.float32
        )
    else:
        trtllm_lamport_initialize(
            ipc_handles[2][tp_rank], lamport_buffer_size // 2, torch.float16
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
    cudart.cudaMemcpy(flag_ptr.value + 3 * 4, lamport_comm_size_bytes, 4)
    print("set flag_ptr[3] = lamport_comm_size: ", lamport_comm_size)
    # add flag_ptr to workspace
    workspace.append(flag_ptr.value)

    for i in range(len(workspace)):
        print(f"Rank {tp_rank} workspace[{i}] {hex(workspace[i])}")

    # Store workspace pointers in device tensor
    workspace_tensor = torch.tensor(
        workspace, dtype=torch.int64, device=torch.device("cuda")
    )

    dist.barrier(group=group)  # must sync after create_workspace

    return ipc_handles, workspace_tensor


def trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
    workspace: List[int], group: Optional[ProcessGroup] = None
) -> None:
    """
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
    get_trtllm_comm_module().trtllm_lamport_initialize_all(
        buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype
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
    scale_factor: Optional[float],
    layout_code: Optional[FP4QuantizationSFLayout],
) -> None:
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
    layout_code: Optional[FP4QuantizationSFLayout],
    moe_allreduce_out: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
) -> None:
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
