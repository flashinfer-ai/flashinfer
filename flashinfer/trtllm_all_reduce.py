"""
Copyright (c) 2024 by FlashInfer team.

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
import functools
from dataclasses import dataclass
from enum import IntEnum
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm90a_nvcc_flags, sm100a_nvcc_flags
from .utils import register_custom_op


class DataType(IntEnum):
    BOOL = 0
    UINT8 = 1
    INT8 = 2
    INT32 = 3
    INT64 = 4
    BF16 = 5
    FP8 = 6
    FP16 = 7
    FP32 = 8
    UNKNOWN = 9


class AllReduceFusionPattern(IntEnum):
    ALL_REDUCE = 0
    AR_RESIDUAL_RMS_NORM = 1
    AR_RESIDUAL_RMS_NORM_FP8_QUANT = 2
    AR_RESIDUAL_RMS_NORM_FP4_QUANT = 3
    AR_RESIDUAL_RMS_NORM_OUT_FP8_QUANT = 4
    AR_RESIDUAL_RMS_NORM_OUT_FP4_QUANT = 5


class FP4QuantizationSFLayout(IntEnum):
    SWIZZLED = 0
    LINEAR = 1


@dataclass
class AllReduceFusionParams:
    nranks: int
    rank: int
    dtype: DataType
    size: int
    hidden_dim: int
    workspace: torch.Tensor  # void**
    allreduce_in: torch.Tensor  # void*
    residual_in: Optional[torch.Tensor] = None  # void*
    allreduce_out: Optional[torch.Tensor] = None  # void*
    residual_out: Optional[torch.Tensor] = None  # void*
    norm_out: Optional[torch.Tensor] = None  # void*
    quant_out: Optional[torch.Tensor] = None  # void*
    scale_out: Optional[torch.Tensor] = None  # void*
    rms_gamma: Optional[torch.Tensor] = None  # void*
    rms_eps: float = 1e-6
    scale_factor: Optional[torch.Tensor] = None  # float*
    use_oneshot: bool = True
    layout: FP4QuantizationSFLayout = FP4QuantizationSFLayout.SWIZZLED
    stream: Optional[torch.cuda.Stream] = None
    pattern: AllReduceFusionPattern = AllReduceFusionPattern.ALL_REDUCE

    def __post_init__(self):
        if self.stream is None:
            self.stream = torch.cuda.current_stream()


@dataclass
class MoeReductionAllReduceFusionParams(AllReduceFusionParams):
    """Parameters for MoE reduction + all-reduce fusion operation.

    This extends AllReduceFusionParams with MoE-specific parameters.
    """

    moe_reduction_device_num_experts: Optional[torch.Tensor] = None  # int*
    moe_reduction_scale_input: Optional[torch.Tensor] = None  # float*
    moe_reduction_active_experts_token_input: Optional[torch.Tensor] = None  # void*
    moe_reduction_token_input: Optional[torch.Tensor] = None  # void*


def gen_trtllm_comm_module() -> JitSpec:
    return gen_jit_spec(
        "trtllm_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "trtllm_comm/allReduceFusionKernels.cu",  # allreduce_fusion_kernel_oneshot_lamport
            jit_env.FLASHINFER_CSRC_DIR
            / "trtllm_comm/moeAllReduceFusionKernels.cu",  # moereduce_allreduce_fusion_kernel_oneshot_lamport
        ],
        extra_cuda_cflags=sm100a_nvcc_flags,
    )


@functools.cache
def get_trtllm_comm_module():
    module = gen_trtllm_comm_module().build_and_load()

    # torch library for all
    @register_custom_op(
        "flashinfer::allreduce_fusion_op",
        mutates_args=["params"],
    )
    def allreduce_fusion_op(params: AllReduceFusionParams) -> None:
        """Performs an all-reduce operation with optional fusion patterns.

        Args:
            params: AllReduceFusionParams object for the all-reduce operation.
        """
        # Convert Python params to C++ struct format
        tensors = {
            "workspace": params.workspace,
            "allreduce_in": params.allreduce_in,
            "residual_in": params.residual_in,
            "allreduce_out": params.allreduce_out,
            "residual_out": params.residual_out,
            "norm_out": params.norm_out,
            "quant_out": params.quant_out,
            "scale_out": params.scale_out,
            "rms_gamma": params.rms_gamma,
            "scale_factor": params.scale_factor,
        }

        # Filter out None tensors and get their data pointers
        tensor_ptrs = {
            k: (v.data_ptr() if v is not None else 0) for k, v in tensors.items()
        }

        return module.allreduce_fusion_op(
            params.nranks,
            params.rank,
            int(params.dtype),
            params.size,
            params.hidden_dim,
            tensor_ptrs["workspace"],
            tensor_ptrs["allreduce_in"],
            tensor_ptrs["residual_in"],
            tensor_ptrs["allreduce_out"],
            tensor_ptrs["residual_out"],
            tensor_ptrs["norm_out"],
            tensor_ptrs["quant_out"],
            tensor_ptrs["scale_out"],
            tensor_ptrs["rms_gamma"],
            params.rms_eps,
            tensor_ptrs["scale_factor"],
            params.use_oneshot,
            int(params.layout),
            (
                params.stream.cuda_stream
                if params.stream
                else torch.cuda.current_stream().cuda_stream
            ),
            int(params.pattern),
        )

    @register_custom_op(
        "flashinfer::moereduction_allreduce_fusion_op",
        mutates_args=["params"],
    )
    def moereduction_allreduce_fusion_op(
        params: MoeReductionAllReduceFusionParams,
    ) -> None:
        """Performs a MoE reduction + all-reduce operation with optional fusion patterns.

        Args:
            params: MoeReductionAllReduceFusionParams object for the MoE reduction + all-reduce operation.
        """
        # Convert Python params to C++ struct format
        tensors = {
            "workspace": params.workspace,
            "allreduce_in": params.allreduce_in,
            "residual_in": params.residual_in,
            "residual_out": params.residual_out,
            "norm_out": params.norm_out,
            "quant_out": params.quant_out,
            "scale_out": params.scale_out,
            "rms_gamma": params.rms_gamma,
            "scale_factor": params.scale_factor,
            "moe_reduction_device_num_experts": params.moe_reduction_device_num_experts,
            "moe_reduction_scale_input": params.moe_reduction_scale_input,
            "moe_reduction_active_experts_token_input": params.moe_reduction_active_experts_token_input,
            "moe_reduction_token_input": params.moe_reduction_token_input,
        }

        # Filter out None tensors and get their data pointers
        tensor_ptrs = {
            k: (v.data_ptr() if v is not None else 0) for k, v in tensors.items()
        }

        return module.moereduction_allreduce_fusion_op(
            params.nranks,
            params.rank,
            int(params.dtype),
            params.size,
            params.hidden_dim,
            tensor_ptrs["workspace"],
            tensor_ptrs["allreduce_in"],
            tensor_ptrs["residual_in"],
            tensor_ptrs["residual_out"],
            tensor_ptrs["norm_out"],
            tensor_ptrs["quant_out"],
            tensor_ptrs["scale_out"],
            tensor_ptrs["rms_gamma"],
            params.rms_eps,
            tensor_ptrs["scale_factor"],
            params.use_oneshot,
            int(params.layout),
            (
                params.stream.cuda_stream
                if params.stream
                else torch.cuda.current_stream().cuda_stream
            ),
            tensor_ptrs["moe_reduction_device_num_experts"],
            tensor_ptrs["moe_reduction_scale_input"],
            tensor_ptrs["moe_reduction_active_experts_token_input"],
            tensor_ptrs["moe_reduction_token_input"],
        )

    return SimpleNamespace(
        allreduce_fusion_op=allreduce_fusion_op,
        moereduction_allreduce_fusion_op=moereduction_allreduce_fusion_op,
    )


def allreduce_fusion_op(params: AllReduceFusionParams) -> None:
    get_trtllm_comm_module().allreduce_fusion_op(params)


def moereduction_allreduce_fusion_op(params: MoeReductionAllReduceFusionParams) -> None:
    get_trtllm_comm_module().moereduction_allreduce_fusion_op(params)
