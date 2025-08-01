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
from enum import IntEnum
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fp4_quantization import nvfp4_block_scale_interleave
from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec, setup_cubin_loader, sm100a_nvcc_flags
from ..jit.cutlass_gemm.generate_kernels import generate_gemm_operations
from ..utils import _check_shape_dtype_device, register_custom_op, register_fake_op
from .utils import (
    compute_swizzled_sf_shape,
    fp4_scale_infer_shape,
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)


# The type of method in top-K routing, for use in torch custom op
# Please keep this in sync with the counterpart defined in cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h
class RoutingMethodType(IntEnum):
    # Default: Softmax -> TopK
    Default = (0,)
    # Renormalize: TopK -> Softmax
    Renormalize = (1,)
    # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups
    DeepSeekV3 = (2,)
    # Llama4: Top1 -> Sigmoid
    Llama4 = (3,)
    # Qwen3: Softmax -> TopK -> Renormalize
    RenormalizeNaive = (4,)
    # Unspecified
    Unspecified = 5


# See MatrixLayout from include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/Enums.h
class WeightLayout(IntEnum):
    # K-major layout (default). [Mn, K]
    MajorK = 0
    # M-major for A and N-major for B. [K, Mn]
    MajorMn = 1
    # Layout is blocked along the K dimension. [K / blockK, Mn, blockK]
    # where blockK is fixed at 128B
    BlockMajorK = 2


def get_reorder_rows_for_gated_act_gemm_row_indices(x) -> torch.Tensor:
    """
    Reorders rows in the gemm/MOE_gemm weight matrix for min-latency
    [r0, r1, r2, r3, ..., rN/2, r(N/2+1), .. r(N-1)]
    to
    [r0, rN/2, r1, rN/2+1, ..., r(N/2-1), r(N-1)]
    """
    assert x.dim() == 2, f"x should be a 2D tensor, not {x.dim()}"
    M, K = x.shape
    assert M % 2 == 0, f"x.shape[0] must be even, not {M}"

    row_indices = torch.arange(M, dtype=torch.long)

    # We split into top half and bottom half, but if M is odd,
    # the bottom half is one row larger.
    top = row_indices[: (M + 1) // 2]  # round up
    bot = row_indices[(M + 1) // 2 :]  # remainder

    # Create the output
    permuted_row_indices = torch.empty_like(row_indices)

    # We'll place rows of `top` and `bot` in alternation
    permuted_row_indices[0::2] = top
    permuted_row_indices[1::2] = bot

    return permuted_row_indices


def reorder_rows_for_gated_act_gemm(x):
    """
    PyTorch implementation of trt-llm gen `reorderRowsForGatedActGemm`
    """
    row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(x)

    permute = lambda x: x[row_indices]

    return permute(x)


def convert_to_block_layout(input_tensor: torch.Tensor, blockK: int) -> torch.Tensor:
    M, K = input_tensor.shape
    assert K % blockK == 0, "K must be divisible by blockK"
    return input_tensor.view(M, K // blockK, blockK).permute(1, 0, 2).contiguous()


def gen_cutlass_fused_moe_sm100_module(use_fast_build: bool = False) -> JitSpec:
    output_dir = (
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal/tensorrt_llm/cutlass_instantiations/"
    )

    required_kernels_sm100 = [
        # M128 kernels
        "cutlass_kernel_file_gemm_grouped_sm100_M128_BS_group0.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_BS_group1.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_BS_group2.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_group0.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_group1.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_group2.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_group3.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_group4.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_group5.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_group6.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_group7.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M128_group8.generated.cu",
        # M256 kernels
        "cutlass_kernel_file_gemm_grouped_sm100_M256_BS_group0.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M256_BS_group1.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M256_BS_group2.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M256_group0.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M256_group1.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M256_group2.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M256_group3.generated.cu",
        # M64 kernels
        "cutlass_kernel_file_gemm_grouped_sm100_M64_group0.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M64_group1.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M64_group2.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M64_group3.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M64_group4.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm100_M64_group5.generated.cu",
    ]
    required_kernels_sm80 = [
        # M128 kernels
        "cutlass_kernel_file_gemm_grouped_sm80_M128_group0.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm80_M128_group1.generated.cu",
        # M16 kernels
        "cutlass_kernel_file_gemm_grouped_sm80_M16_group0.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm80_M16_group1.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm80_M16_group2.generated.cu",
        # M32 kernels
        "cutlass_kernel_file_gemm_grouped_sm80_M32_group0.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm80_M32_group1.generated.cu",
        # M64 kernels
        "cutlass_kernel_file_gemm_grouped_sm80_M64_group0.generated.cu",
        "cutlass_kernel_file_gemm_grouped_sm80_M64_group1.generated.cu",
    ]
    group_gemm_sm100_dir = output_dir / "gemm_grouped/100"
    group_gemm_sm80_dir = output_dir / "gemm_grouped/80"

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        generate_gemm_operations(
            output_dir,
            "100;100-real",
        )

    except Exception as e:
        raise RuntimeError(f"Failed to generate Cutlass kernels: {e}")

    return gen_jit_spec(
        "fused_moe_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp4_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp32_fp32.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp16.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_bf16.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_stub.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_sm100_ops.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
            # Add all generated kernels
            *(group_gemm_sm100_dir / kernel for kernel in required_kernels_sm100),
            *(group_gemm_sm80_dir / kernel for kernel in required_kernels_sm80),
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/lora/lora.cpp",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
            "-DCOMPILE_BLACKWELL_TMA_GEMMS",
            "-DCOMPILE_BLACKWELL_TMA_GROUPED_GEMMS",
            "-DCOMPILE_HOPPER_TMA_GEMMS",
            "-DUSING_OSS_CUTLASS_MOE_GEMM",
        ],
        extra_cflags=["-DFAST_BUILD"] if use_fast_build else [],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "cutlass_extensions"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels",
        ],
    )


@functools.cache
def get_cutlass_fused_moe_sm100_module(use_fast_build: bool = False):
    module = gen_cutlass_fused_moe_sm100_module(use_fast_build).build_and_load(
        class_name="FusedMoeRunner"
    )

    class MoERunner(TunableRunner):
        # avoid overhead of creating a new runner in forward pass
        runner_dict = dict()
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    0,
                    0,
                    get_last_power_of_2_num_tokens_buckets(8192),
                    lambda x: min(last_positive_power_of_2(x), 8192),
                ),
            )
        )

        def __init__(
            self,
            x_dtype: torch.dtype,
            weight_dtype: torch.dtype,
            output_dtype: torch.dtype,
            top_k: int,
            tp_size: int,
            tp_rank: int,
            ep_size: int,
            ep_rank: int,
            cluster_size: int,
            cluster_rank: int,
            enable_alltoall: bool,
            use_deepseek_fp8_block_scale: bool,
            use_w4a8_group_scaling: bool,
            use_mxfp8_act_scaling: bool,
            min_latency_mode: bool,
        ):
            self.x_dtype = x_dtype
            self.weight_dtype = weight_dtype
            self.output_dtype = output_dtype
            self.top_k = top_k
            self.tp_size = tp_size
            self.tp_rank = tp_rank
            self.ep_size = ep_size
            self.ep_rank = ep_rank
            self.cluster_size = cluster_size
            self.cluster_rank = cluster_rank
            self.enable_alltoall = enable_alltoall
            self.use_deepseek_fp8_block_scale = use_deepseek_fp8_block_scale
            self.use_w4a8_group_scaling = use_w4a8_group_scaling
            self.use_mxfp8_act_scaling = use_mxfp8_act_scaling
            self.min_latency_mode = min_latency_mode
            instance_key = (
                x_dtype,
                weight_dtype,
                output_dtype,
                use_deepseek_fp8_block_scale,
                use_w4a8_group_scaling,
                use_mxfp8_act_scaling,
            )

            if instance_key not in MoERunner.runner_dict:
                MoERunner.runner_dict[instance_key] = (
                    torch.classes.fused_moe_sm100.FusedMoeRunner(
                        x_dtype,
                        weight_dtype,
                        output_dtype,
                        use_deepseek_fp8_block_scale,
                        use_w4a8_group_scaling,
                        use_mxfp8_act_scaling,
                    )
                )
            self.fused_moe_runner = MoERunner.runner_dict[instance_key]

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            return range(self.fused_moe_runner.get_tactic_num())

        def forward(
            self,
            inputs: List[torch.Tensor],
            gemm_idx: int = 0,
            tactic: int = -1,
            do_preparation: bool = False,
        ):
            (
                x,
                fc1_expert_weights,
                fc1_expert_biases,
                fc2_expert_weights,
                fc2_expert_biases,
            ) = inputs
            self.fused_moe_runner.run_gemm_profile(
                x,
                fc1_expert_weights,
                fc1_expert_biases,
                fc2_expert_weights,
                fc2_expert_biases,
                self.top_k,
                self.tp_size,
                self.tp_rank,
                self.ep_size,
                self.ep_rank,
                self.cluster_size,
                self.cluster_rank,
                self.enable_alltoall,
                self.min_latency_mode,
                gemm_idx,
                tactic,
                do_preparation,
            )

        @classmethod
        @functools.lru_cache(maxsize=None)
        def refine_tuning_config(cls, tune_max_num_tokens: int):
            cls.tuning_config = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        0,
                        0,
                        get_last_power_of_2_num_tokens_buckets(tune_max_num_tokens),
                        lambda x: min(last_positive_power_of_2(x), tune_max_num_tokens),
                    ),
                )
            )

    @register_custom_op(
        "flashinfer::cutlass_fused_moe_sm100",
        mutates_args=(""),
    )
    def cutlass_fused_moe_sm100(
        output: torch.Tensor,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor],
        fc2_expert_weights: torch.Tensor,
        fc2_expert_biases: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        enable_alltoall: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4a8_group_scaling: bool = False,
        use_mxfp8_act_scaling: bool = False,
        min_latency_mode: bool = False,
        tune_max_num_tokens: int = 8192,
    ) -> List[torch.Tensor]:
        tuner = AutoTuner.get()
        MoERunner.refine_tuning_config(tune_max_num_tokens)

        # allocate workspace for profiling
        moe_runner = MoERunner(
            x_dtype=input.dtype,
            weight_dtype=fc1_expert_weights.dtype,
            output_dtype=output_dtype,
            top_k=token_selected_experts.size(1),
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=cluster_size,
            cluster_rank=cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4a8_group_scaling=use_w4a8_group_scaling,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            min_latency_mode=min_latency_mode,
        )

        _, gemm_tactic_1 = tuner.choose_one(
            "trtllm::fused_moe::gemm1",
            [moe_runner],
            MoERunner.tuning_config,
            [
                input,
                fc1_expert_weights,
                fc1_expert_biases,
                fc2_expert_weights,
                fc2_expert_biases,
            ],
            gemm_idx=1,
        )

        _, gemm_tactic_2 = tuner.choose_one(
            "trtllm::fused_moe::gemm2",
            [moe_runner],
            MoERunner.tuning_config,
            [
                input,
                fc1_expert_weights,
                fc1_expert_biases,
                fc2_expert_weights,
                fc2_expert_biases,
            ],
            gemm_idx=2,
        )

        run_moe = (
            moe_runner.fused_moe_runner.run_moe_min_latency
            if min_latency_mode
            else moe_runner.fused_moe_runner.run_moe
        )
        result = run_moe(
            output,
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc1_expert_biases,
            fc2_expert_weights,
            fc2_expert_biases,
            quant_scales,
            input_sf,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            cluster_size,
            cluster_rank,
            enable_alltoall,
            min_latency_mode,
            [gemm_tactic_1, gemm_tactic_2],
        )

        return result if min_latency_mode else [result]

    @register_fake_op("flashinfer::cutlass_fused_moe_sm100")
    def _fake_cutlass_fused_moe_sm100(
        output: torch.Tensor,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor],
        fc2_expert_weights: torch.Tensor,
        fc2_expert_biases: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        enable_alltoall: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4a8_group_scaling: bool = False,
        use_mxfp8_act_scaling: bool = False,
        min_latency_mode: bool = False,
        tune_max_num_tokens: int = 8192,
    ):
        seq_len = input.shape[0]
        hidden_size = fc2_expert_weights.shape[1]

        if min_latency_mode:
            num_experts_on_rank = fc2_expert_weights.shape[0]
            output_shape = [seq_len * num_experts_on_rank, hidden_size]
            experts_to_token_score_shape = [num_experts_on_rank, seq_len]
            active_expert_global_ids_shape = [num_experts_on_rank]
            return [
                input.new_empty(output_shape, dtype=output_dtype),
                input.new_empty([1], dtype=torch.int32),
                input.new_empty(experts_to_token_score_shape, dtype=torch.float32),
                input.new_empty(active_expert_global_ids_shape, dtype=torch.int32),
            ]
        else:
            return [input.new_empty([seq_len, hidden_size], dtype=output_dtype)]

    # Register the module
    return SimpleNamespace(
        cutlass_fused_moe_sm100=cutlass_fused_moe_sm100,
    )


# ref: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L121
def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    fc1_expert_biases: Optional[torch.Tensor] = None,
    fc2_expert_biases: Optional[torch.Tensor] = None,
    input_sf: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4a8_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    """Compute a Mixture of Experts (MoE) layer using CUTLASS backend.

    This function implements a fused MoE layer that combines expert selection, expert computation,
    and output combination into a single operation. It uses CUTLASS for efficient matrix multiplication
    and supports various data types and parallelism strategies.

    Args:
        input (torch.Tensor): Input tensor of shape [num_tokens, hidden_size].
            Support float, float16, bfloat16, float8_e4m3fn and nvfp4.
            For FP8, the input must be quantized.
            For NVFP4, both quantized and non-quantized inputs are supported.
        token_selected_experts (torch.Tensor): Indices of selected experts for each token.
        token_final_scales (torch.Tensor): Scaling factors for each token's expert outputs.
        fc1_expert_weights (torch.Tensor): GEMM1 weights for each expert.
        fc1_expert_biases (torch.Tensor, optional): GEMM1 biases for each expert.
        fc2_expert_weights (torch.Tensor): GEMM2 weights for each expert.
        fc2_expert_biases (torch.Tensor, optional): GEMM1 biases for each expert.
        output_dtype (torch.dtype): Desired output data type.
        quant_scales (List[torch.Tensor]): Quantization scales for the operation.
            NVFP4:
                - gemm1 activation global scale
                - gemm1 weights block scales
                - gemm1 dequant scale
                - gemm2 activation global scale
                - gemm2 weights block scales
                - gemm2 dequant scale
            FP8:
               - gemm1 dequant scale
               - gemm2 activation quant scale
               - gemm2 dequant scale
               - gemm1 input dequant scale
        input_sf (Optional[torch.Tensor]): Input scaling factor for quantization.
        tp_size (int, optional): Tensor parallelism size. Defaults to 1.
        tp_rank (int, optional): Tensor parallelism rank. Defaults to 0.
        ep_size (int, optional): Expert parallelism size. Defaults to 1.
        ep_rank (int, optional): Expert parallelism rank. Defaults to 0.
        cluster_size (int, optional): Cluster size. Defaults to 1.
        cluster_rank (int, optional): Cluster rank. Defaults to 0.
        output (torch.Tensor, optional): The output tensor, if not provided, will be allocated internally.
        enable_alltoall (bool, False): Whether to enable all-to-all communication for expert outputs. Defaults to False.
        use_deepseek_fp8_block_scale (bool, optional): Whether to use FP8 block scaling. Defaults to False.
        use_w4a8_group_scaling (bool, optional): Whether to use W4A8 group scaling. Defaults to False.
        use_mxfp8_act_scaling (bool, optional): Whether to use MXFP8 activation scaling. Defaults to False.
        min_latency_mode (bool, optional): Whether to use minimum latency mode. Defaults to False.
        tune_max_num_tokens (int, optional): Maximum number of tokens for tuning. Defaults to 8192.

    Returns:
        torch.Tensor: Output tensor of shape [seq_len, hidden_size].

    Raises:
        NotImplementedError: If any of the following features are requested but not implemented:
            - FP8 Block Scaling
            - W4A8 Group Scaling
            - Minimum Latency Mode

    Note:
        - The function supports various data types including FP32, FP16, BF16, FP8, and NVFP4.
        - It implements both tensor parallelism and expert parallelism.
        - Currently, some advanced features like FP8 block scaling and minimum latency mode
          are not implemented for Blackwell architecture.
    """
    if use_deepseek_fp8_block_scale:
        raise NotImplementedError(
            "DeepSeek FP8 Block Scaling is not yet implemented in CUTLASS for Blackwell."
        )
    if use_w4a8_group_scaling:
        raise NotImplementedError(
            "W4A8 Group Scaling is not yet implemented for Blackwell."
        )
    if min_latency_mode:
        raise NotImplementedError("min latency mode not yet implemented for Blackwell.")
    if use_mxfp8_act_scaling:
        raise NotImplementedError("mxfp8 not yet implemented for Blackwell.")

    num_rows = input.shape[0]
    if min_latency_mode:
        num_rows *= fc2_expert_weights.shape[0]
    hidden_size = fc2_expert_weights.shape[1]
    output_shape = (num_rows, hidden_size)

    if output is None:
        output = torch.empty(output_shape, dtype=output_dtype, device=input.device)
    else:
        _check_shape_dtype_device(
            output, output_shape, output_dtype, input.device, "output"
        )

    return get_cutlass_fused_moe_sm100_module().cutlass_fused_moe_sm100(
        output,
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc1_expert_biases,
        fc2_expert_weights,
        fc2_expert_biases,
        output_dtype,
        quant_scales,
        input_sf,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        cluster_size,
        cluster_rank,
        enable_alltoall=enable_alltoall,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        use_w4a8_group_scaling=use_w4a8_group_scaling,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        min_latency_mode=min_latency_mode,
        tune_max_num_tokens=tune_max_num_tokens,
    )


# trtllmgen-moe-fp8


def trtllm_gen_fused_moe_sm100_module() -> JitSpec:
    return gen_jit_spec(
        "fused_moe_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_kernel_launcher.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_runner.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_kernel.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_dev_kernel.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_batched_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_ENABLE_CUDA",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
        ]
        + sm100a_nvcc_flags,
        extra_ldflags=["-lcuda"],
    )


@functools.cache
def get_trtllm_moe_sm100_module():
    module = trtllm_gen_fused_moe_sm100_module()
    moe_op = module.build_and_load()
    setup_cubin_loader(str(module.get_library_path()))

    @register_custom_op(
        "flashinfer::trtllm_fp8_per_tensor_scale_moe",
        mutates_args=(""),
    )
    def trtllm_fp8_per_tensor_scale_moe_op(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        output1_scales_scalar: torch.Tensor,
        output1_scales_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        output2_scales_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: float,
        use_routing_scales_on_input: bool,
        tile_tokens_dim: int = 8,
        routing_method_type: int = 0,
    ) -> torch.Tensor:

        # Call the C++ function
        output = moe_op.trtllm_fp8_per_tensor_scale_moe(
            routing_logits,
            routing_bias,
            hidden_states,
            gemm1_weights,
            output1_scales_scalar,
            output1_scales_gate_scalar,
            gemm2_weights,
            output2_scales_scalar,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            use_routing_scales_on_input,
            tile_tokens_dim,
            routing_method_type,
        )
        return output

    @register_fake_op("flashinfer::trtllm_fp8_per_tensor_scale_moe")
    def _fake_trtllm_fp8_per_tensor_scale_moe(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        output1_scales_scalar: torch.Tensor,
        output1_scales_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        output2_scales_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: float,
        use_routing_scales_on_input: bool,
        tile_tokens_dim: int = 8,
        routing_method_type: int = 0,
    ):
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]

        return [hidden_states.new_empty([seq_len, hidden_size], dtype=torch.bfloat16)]

    @register_custom_op(
        "flashinfer::trtllm_fp8_block_scale_moe",
        mutates_args=(""),
    )
    def trtllm_fp8_block_scale_moe_op(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: float,
        tile_tokens_dim: int,
        routing_method_type: int,
        use_shuffled_weight: bool = False,
        weight_layout: int = 0,
    ) -> torch.Tensor:

        # Call the C++ function for block scale MoE
        output = moe_op.trtllm_fp8_block_scale_moe(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            tile_tokens_dim,
            routing_method_type,
            use_shuffled_weight,
            weight_layout,
        )

        return output

    @register_fake_op("flashinfer::trtllm_fp8_block_scale_moe")
    def _fake_trtllm_fp8_block_scale_moe(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: float,
        tile_tokens_dim: int = 8,
        routing_method_type: int = 0,
        use_shuffled_weight: bool = False,
        weight_layout: int = 0,
    ):
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]

        return [hidden_states.new_empty([seq_len, hidden_size], dtype=torch.bfloat16)]

    @register_custom_op(
        "flashinfer::trtllm_fp4_block_scale_moe",
        mutates_args=(""),
    )
    def trtllm_fp4_block_scale_moe_op(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        tile_tokens_dim: int,
        routing_method_type: int,
        do_finalize: bool,
    ) -> List[torch.Tensor]:

        # Call the C++ function for block scale MoE
        output = moe_op.trtllm_fp4_block_scale_moe(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            tile_tokens_dim,
            routing_method_type,
            do_finalize,
        )

        return output

    @register_fake_op("flashinfer::trtllm_fp4_block_scale_moe")
    def _fake_trtllm_fp4_block_scale_moe(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        tile_tokens_dim: int,
        routing_method_type: int,
        do_finalize: bool,
    ):
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]

        return [hidden_states.new_empty([seq_len, hidden_size], dtype=torch.bfloat16)]

    return SimpleNamespace(
        trtllm_fp8_per_tensor_scale_moe=trtllm_fp8_per_tensor_scale_moe_op,
        trtllm_fp8_block_scale_moe=trtllm_fp8_block_scale_moe_op,
        trtllm_fp4_block_scale_moe=trtllm_fp4_block_scale_moe_op,
    )


def trtllm_fp8_per_tensor_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    use_routing_scales_on_input: bool,
    tile_tokens_dim: int = 8,
    routing_method_type: int = 0,
) -> torch.Tensor:
    """FP8 per tensor scale MoE operation.

    Args:
        routing_logits: [seq_len, num_experts] tensor of routing logits
        routing_bias: [num_experts] tensor of routing bias
        hidden_states: [seq_len, hidden_size] tensor of input hidden states
        gemm1_weights: [num_experts, 2*intermediate_size, hidden_size] tensor of first layer weights
        output1_scales_scalar: [local_num_experts] tensor of first layer output scales
        output1_scales_gate_scalar: [local_num_experts] tensor of first layer gate scales
        gemm2_weights: [num_experts, hidden_size, intermediate_size] tensor of second layer weights
        output2_scales_scalar: [local_num_experts] tensor of second layer output scales
        num_experts: Total number of experts
        top_k: Number of experts to route to per token
        n_group: Number of expert groups
        topk_group: Number of groups to consider for top-k routing
        intermediate_size: Size of intermediate layer
        local_expert_offset: Offset of local experts in global expert space
        local_num_experts: Number of experts handled by this device
        routed_scaling_factor: Scaling factor for routing
        use_routing_scales_on_input: Whether to use routing scales on input
        tile_tokens_dim: Tile dimension for tokens (default: 8)
        routing_method_type: Type of routing method to use (default: 0)

    Returns:
        torch.Tensor: Output tensor of shape [seq_len, hidden_size]
    """
    return get_trtllm_moe_sm100_module().trtllm_fp8_per_tensor_scale_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        output1_scales_scalar,
        output1_scales_gate_scalar,
        gemm2_weights,
        output2_scales_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        use_routing_scales_on_input,
        tile_tokens_dim,
        routing_method_type,
    )


def trtllm_fp8_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    tile_tokens_dim: int = 8,
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
) -> torch.Tensor:
    """FP8 block scale MoE operation.

    Args:
        routing_logits: [seq_len, num_experts] tensor of routing logits
        routing_bias: [num_experts] tensor of routing bias
        hidden_states: [seq_len, hidden_size] tensor of input hidden states
        hidden_states_scale: [hidden_size//128, seq_len] tensor of hidden states block scales
        gemm1_weights: [num_experts, 2*intermediate_size, hidden_size] tensor of first layer weights
        gemm1_weights_scale: [num_experts, 2*intermediate_size//128, hidden_size//128] tensor of first layer block scales
        gemm2_weights: [num_experts, hidden_size, intermediate_size] tensor of second layer weights
        gemm2_weights_scale: [num_experts, hidden_size//128, intermediate_size//128] tensor of second layer block scales
        num_experts: Total number of experts
        top_k: Number of experts to route to per token
        n_group: Number of expert groups
        topk_group: Number of groups to consider for top-k routing
        intermediate_size: Size of intermediate layer
        local_expert_offset: Offset of local experts in global expert space
        local_num_experts: Number of experts handled by this device
        routed_scaling_factor: Scaling factor for routing
        tile_tokens_dim: Tile dimension for tokens (default: 8)
        routing_method_type: Type of routing method to use (default: 0)

    Returns:
        torch.Tensor: Output tensor of shape [seq_len, hidden_size]
    """
    return get_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        tile_tokens_dim,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
    )


def trtllm_fp4_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    tile_tokens_dim: int = 8,
    routing_method_type: int = 0,
    do_finalize: bool = False,
) -> List[torch.Tensor]:
    """FP4 block scale MoE operation.

    Args:
        routing_logits: [seq_len, num_experts] tensor of routing logits
        routing_bias: [num_experts] tensor of routing bias (can be None for some routing methods)
        hidden_states: [seq_len, hidden_size] tensor of input hidden states
        hidden_states_scale: [hidden_size//128, seq_len] tensor of hidden states block scales
        gemm1_weights: [num_experts, 2*intermediate_size, hidden_size] tensor of first layer weights
        gemm1_weights_scale: [num_experts, 2*intermediate_size//128, hidden_size//128] tensor of first layer block scales
        gemm2_weights: [num_experts, hidden_size, intermediate_size] tensor of second layer weights
        gemm2_weights_scale: [num_experts, hidden_size//128, intermediate_size//128] tensor of second layer block scales
        output1_scale_scalar: [local_num_experts] tensor of scaling factors for first layer activation output
        output1_scale_gate_scalar: [local_num_experts] tensor of scaling factors for first layer gate output
        output2_scale_scalar: [local_num_experts] tensor of scaling factors for second layer output
        num_experts: Total number of experts
        top_k: Number of experts to route to per token
        n_group: Number of expert groups (can be None for some routing methods)
        topk_group: Number of groups to consider for top-k routing (can be None for some routing methods)
        intermediate_size: Size of intermediate layer
        local_expert_offset: Offset of local experts in global expert space
        local_num_experts: Number of experts handled by this device
        routed_scaling_factor: Scaling factor for routing (can be None for some routing methods)
        tile_tokens_dim: Tile dimension for tokens (default: 8)
        routing_method_type: Type of routing method to use (default: 0)
            - 0: Default (Softmax -> TopK)
            - 1: Renormalize (TopK -> Softmax)
            - 2: DeepSeekV3 (Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts)
            - 3: Llama4 (Top1 -> Sigmoid)
            - 4: RenormalizeNaive (Softmax -> TopK -> Renormalize)
        do_finalize: Whether to finalize the output (default: False)

    Returns:
        List[torch.Tensor]: List of output tensors. If do_finalize=True, returns the final MoE output.
            Otherwise, returns intermediate results that need further processing.
    """
    return get_trtllm_moe_sm100_module().trtllm_fp4_block_scale_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        tile_tokens_dim,
        routing_method_type,
        do_finalize,
    )
