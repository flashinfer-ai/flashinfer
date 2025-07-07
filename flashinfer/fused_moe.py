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
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

from .autotuner import AutoTuner, TunableRunner, TuningConfig
from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm100a_nvcc_flags
from .utils import _check_shape_dtype_device, register_custom_op, register_fake_op


def gen_fused_moe_sm100_module() -> JitSpec:
    return gen_jit_spec(
        "fused_moe_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_tma_warp_specialized_input.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_fp8_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_fp8_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_fp4_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_fp32_fp32.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_fp16_uint8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_fp16_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_fp16_fp16.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_bf16_uint8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_bf16_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_bf16_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_bf16_bf16.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_sm100_ops.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_1.generated.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_2.generated.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_3.generated.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_4.generated.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_5.generated.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_6.generated.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_7.generated.cu",
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
        ],
        extra_cflags=[
            "-DFAST_BUILD",
        ],
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
            / "internal_cutlass_kernels"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "internal_cutlass_kernels",
        ],
    )


@functools.cache
def get_fused_moe_sm100_module():
    module = gen_fused_moe_sm100_module().build_and_load(class_name="FusedMoeRunner")

    class MoERunner(TunableRunner):
        # avoid overhead of creating a new runner in forward pass
        _runner_dict: Dict[str, module] = dict()

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
            use_fp8_block_scaling: bool,
            use_w4a8_group_scaling: bool,
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
            self.use_fp8_block_scaling = use_fp8_block_scaling
            self.use_w4a8_group_scaling = use_w4a8_group_scaling

            instance_key = (
                x_dtype,
                weight_dtype,
                output_dtype,
                use_fp8_block_scaling,
                use_w4a8_group_scaling,
            )

            if instance_key not in MoERunner._runner_dict:
                MoERunner._runner_dict[instance_key] = (
                    torch.classes.fused_moe_sm100.FusedMoeRunner(
                        x_dtype,
                        weight_dtype,
                        output_dtype,
                        use_fp8_block_scaling,
                        use_w4a8_group_scaling,
                    )
                )
            self._fused_moe_runner = MoERunner._runner_dict[instance_key]
            self._is_nvfp4 = weight_dtype == torch.int64

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
        ) -> List[int]:
            x, _, _, min_latency_mode_tensor = inputs
            min_latency_mode = min_latency_mode_tensor.size(0) == 1
            m = x.shape[0]

            # Only profile m <= 128 for min latency mode = True
            # Profile all valid buckets for min latency mode = False
            # TODO: min_latency_mode = True will cause the following error:
            # Cannot profile configuration 4: Cutlass GEMM Tactic
            # [TensorRT-LLM][ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm.
            # Should be fixed in the moe_kernels in the future.
            invalid = (m > 128 and min_latency_mode) or (
                m <= 128 and min_latency_mode and (not self._is_nvfp4)
            )

            return (
                [] if invalid else list(range(self._fused_moe_runner.get_tactic_num()))
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            gemm_idx: int = 0,
            tactic: int = -1,
            do_preparation: bool = False,
        ):
            x, fc1_expert_weights, fc2_expert_weights, min_latency_mode_tensor = inputs
            min_latency_mode = min_latency_mode_tensor.size(0) == 1
            # determine if we should use min latency mode according to the profiled seq len
            self._fused_moe_runner.run_gemm_profile(
                x,
                fc1_expert_weights,
                fc2_expert_weights,
                self.top_k,
                self.tp_size,
                self.tp_rank,
                self.ep_size,
                self.ep_rank,
                self.cluster_size,
                self.cluster_rank,
                min_latency_mode,
                gemm_idx,
                tactic,
                do_preparation,
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
        fc2_expert_weights: torch.Tensor,
        output_dtype: torch.dtype,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        use_fp8_block_scaling: bool = False,
        use_w4a8_group_scaling: bool = False,
        min_latency_mode: bool = False,
        tune_max_num_tokens: int = 8192,
    ) -> List[torch.Tensor]:
        tuner = AutoTuner.get()

        def next_positive_power_of_2(x: int) -> int:
            if x < 1:
                return 1

            return 1 << (x - 1).bit_length()

        tune_num_tokens_list = []
        tune_num_tokens = next_positive_power_of_2(tune_max_num_tokens)
        while tune_num_tokens > 0:
            tune_num_tokens_list.append(tune_num_tokens)
            tune_num_tokens //= 2
        # TODO: only profile for min_latency_mode = False due to the error in the moe_kernels
        tuning_config = TuningConfig(
            dynamic_tensors=(
                # input, dim 0, all valid buckets, map a seq_len to power of 2 bucket index
                (0, 0, (tuple(tune_num_tokens_list), next_positive_power_of_2)),
                # min_latency_tensor, dim 0, (0 for False, 1 for True), map to it self
                (3, 0, ((0,), lambda x: x)),
            )
        )

        # TODO: set min_latency_mode always to False due to the error in the moe_kernels
        min_latency_tensor = torch.empty(0)

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
            use_fp8_block_scaling=use_fp8_block_scaling,
            use_w4a8_group_scaling=use_w4a8_group_scaling,
        )

        _, gemm_tactic_1 = tuner.choose_one(
            "trtllm::fused_moe::gemm1",
            [moe_runner],
            tuning_config,
            [input, fc1_expert_weights, fc2_expert_weights, min_latency_tensor],
            gemm_idx=1,
        )

        _, gemm_tactic_2 = tuner.choose_one(
            "trtllm::fused_moe::gemm2",
            [moe_runner],
            tuning_config,
            [input, fc1_expert_weights, fc2_expert_weights, min_latency_tensor],
            gemm_idx=2,
        )

        run_moe = (
            moe_runner._fused_moe_runner.run_moe_min_latency
            if min_latency_mode
            else moe_runner._fused_moe_runner.run_moe
        )
        result = run_moe(
            output,
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc2_expert_weights,
            quant_scales,
            input_sf,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            cluster_size,
            cluster_rank,
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
        fc2_expert_weights: torch.Tensor,
        output_dtype: torch.dtype,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        use_fp8_block_scaling: bool = False,
        use_w4a8_group_scaling: bool = False,
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


# TODO(shuw): wrap into a FusedMoeModule once trtllm-gen is readly.
# ref: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/fused_moe.py#L827
def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    input_sf: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    output: Optional[torch.Tensor] = None,
    use_fp8_block_scaling: bool = False,
    use_w4a8_group_scaling: bool = False,
    min_latency_mode: bool = False,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    """Compute a Mixture of Experts (MoE) layer using CUTLASS backend.

    This function implements a fused MoE layer that combines expert selection, expert computation,
    and output combination into a single operation. It uses CUTLASS for efficient matrix multiplication
    and supports various data types and parallelism strategies.

    Args:
        input (torch.Tensor): Input tensor of shape [seq_len, hidden_size].
            Support float, float16, bfloat16, float8_e4m3fn and nvfp4.
            For FP8, the input must be quantized.
            For NVFP4, both quantized and non-quantized inputs are supported.
        token_selected_experts (torch.Tensor): Indices of selected experts for each token.
        token_final_scales (torch.Tensor): Scaling factors for each token's expert outputs.
        fc1_expert_weights (torch.Tensor): GEMM1 weights for each expert.
        fc2_expert_weights (torch.Tensor): GEMM2 weights for each expert.
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
        use_fp8_block_scaling (bool, optional): Whether to use FP8 block scaling. Defaults to False.
        use_w4a8_group_scaling (bool, optional): Whether to use W4A8 group scaling. Defaults to False.
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
    if use_fp8_block_scaling:
        raise NotImplementedError(
            "FP8 Block Scaling is not yet implemented for Blackwell."
        )
    if use_w4a8_group_scaling:
        raise NotImplementedError(
            "W4A8 Group Scaling is not yet implemented for Blackwell."
        )
    if min_latency_mode:
        raise NotImplementedError("min latency mode not yet implemented for Blackwell.")

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

    return get_fused_moe_sm100_module().cutlass_fused_moe_sm100(
        output,
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        output_dtype,
        quant_scales,
        input_sf,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        cluster_size,
        cluster_rank,
        use_fp8_block_scaling,
        use_w4a8_group_scaling,
        min_latency_mode,
        tune_max_num_tokens,
    )
