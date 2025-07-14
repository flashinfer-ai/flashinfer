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
from .jit import gen_jit_spec, setup_cubin_loader, sm100a_nvcc_flags
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
        ]
        + sm100a_nvcc_flags,
        extra_ldflags=["-lcuda"],
    )


def _calculate_fp8_per_tensor_scale_workspace_size(
    seq_len: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    tile_tokens_dim: int = 8,
) -> tuple[int, dict]:
    """Calculate the required workspace size for FP8 per tensor scale MoE computation"""

    # Calculate maximum number of padded tokens
    expanded_row_count = seq_len * top_k
    max_padding_required = (tile_tokens_dim - 1) * num_experts
    max_padded_tokens = (
        (expanded_row_count + max_padding_required + tile_tokens_dim - 1)
        // tile_tokens_dim
    ) * tile_tokens_dim

    # Calculate maximum number of CTAs in batch dimension
    max_ctas_in_batch_dim_per_expert = (
        seq_len + tile_tokens_dim - 1
    ) // tile_tokens_dim
    max_enabled_experts = min(seq_len * top_k, num_experts)
    max_num_ctas_in_batch_dim = max_enabled_experts * max_ctas_in_batch_dim_per_expert

    # For large token counts, bound by permuted buffer size
    tiles_for_permuted_buffer = (
        max_padded_tokens + tile_tokens_dim - 1
    ) // tile_tokens_dim
    max_num_ctas_in_batch_dim = min(
        max_num_ctas_in_batch_dim, tiles_for_permuted_buffer
    )

    # Calculate workspace component sizes with debug output
    offset = 0
    workspace_info = {
        "parameters": {
            "seq_len": seq_len,
            "num_experts": num_experts,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "top_k": top_k,
            "tile_tokens_dim": tile_tokens_dim,
            "max_padded_tokens": max_padded_tokens,
            "max_num_ctas_in_batch_dim": max_num_ctas_in_batch_dim,
        },
        "components": {},
    }

    def align_offset(alignment: int) -> int:
        nonlocal offset
        offset = ((offset + alignment - 1) // alignment) * alignment
        return offset

    # Helper function to add component info
    def add_component(name: str, size: int, offset: int, data_type: str = ""):
        workspace_info["components"][name] = {
            "size": size,
            "offset": offset,
            "data_type": data_type,
        }
        # print(f"workspace({name}) = {size} @ {offset}")

    # Routing workspace tensors
    num_tokens_per_expert_size = num_experts * 4  # int32_t
    align_offset(4)
    num_tokens_per_expert_offset = offset
    offset += num_tokens_per_expert_size
    add_component(
        "num_tokens_per_expert",
        num_tokens_per_expert_size,
        num_tokens_per_expert_offset,
        "int32_t",
    )

    total_num_padded_tokens_size = 4  # int32_t
    align_offset(4)
    total_num_padded_tokens_offset = offset
    offset += total_num_padded_tokens_size
    add_component(
        "total_num_padded_tokens",
        total_num_padded_tokens_size,
        total_num_padded_tokens_offset,
        "int32_t",
    )

    expanded_idx_to_permuted_idx_size = seq_len * top_k * 4  # int32_t
    align_offset(4)
    expanded_idx_to_permuted_idx_offset = offset
    offset += expanded_idx_to_permuted_idx_size
    add_component(
        "expanded_idx_to_permuted_idx",
        expanded_idx_to_permuted_idx_size,
        expanded_idx_to_permuted_idx_offset,
        "int32_t",
    )

    permuted_idx_to_token_idx_size = max_padded_tokens * 4  # int32_t
    align_offset(4)
    permuted_idx_to_token_idx_offset = offset
    offset += permuted_idx_to_token_idx_size
    add_component(
        "permuted_idx_to_token_idx",
        permuted_idx_to_token_idx_size,
        permuted_idx_to_token_idx_offset,
        "int32_t",
    )

    expert_weights_size = seq_len * top_k * 2  # BFloat16 (uint16_t)
    align_offset(2)
    expert_weights_offset = offset
    offset += expert_weights_size
    add_component(
        "expert_weights", expert_weights_size, expert_weights_offset, "bfloat16"
    )

    expert_indexes_size = seq_len * top_k * 4  # int32_t
    align_offset(4)
    expert_indexes_offset = offset
    offset += expert_indexes_size
    add_component(
        "expert_indexes", expert_indexes_size, expert_indexes_offset, "int32_t"
    )

    expert_count_histogram_size = 2 * 256 * 4  # int32_t
    align_offset(4)
    expert_count_histogram_offset = offset
    offset += expert_count_histogram_size
    add_component(
        "expert_count_histogram",
        expert_count_histogram_size,
        expert_count_histogram_offset,
        "int32_t",
    )

    # CTA workspace tensors
    cta_idx_xy_to_batch_idx_size = max_num_ctas_in_batch_dim * 4  # int32_t
    align_offset(4)
    cta_idx_xy_to_batch_idx_offset = offset
    offset += cta_idx_xy_to_batch_idx_size
    add_component(
        "cta_idx_xy_to_batch_idx",
        cta_idx_xy_to_batch_idx_size,
        cta_idx_xy_to_batch_idx_offset,
        "int32_t",
    )

    cta_idx_xy_to_mn_limit_size = max_num_ctas_in_batch_dim * 4  # int32_t
    align_offset(4)
    cta_idx_xy_to_mn_limit_offset = offset
    offset += cta_idx_xy_to_mn_limit_size
    add_component(
        "cta_idx_xy_to_mn_limit",
        cta_idx_xy_to_mn_limit_size,
        cta_idx_xy_to_mn_limit_offset,
        "int32_t",
    )

    num_non_exiting_ctas_size = 4  # int32_t
    align_offset(4)
    num_non_exiting_ctas_offset = offset
    offset += num_non_exiting_ctas_size
    add_component(
        "num_non_exiting_ctas",
        num_non_exiting_ctas_size,
        num_non_exiting_ctas_offset,
        "int32_t",
    )

    # Intermediate computation tensors
    gemm1_output_size = max_padded_tokens * 2 * intermediate_size * 1  # fp8 (uint8_t)
    align_offset(16)  # 16 bytes alignment for TMA
    gemm1_output_offset = offset
    offset += gemm1_output_size
    add_component("gemm1_output", gemm1_output_size, gemm1_output_offset, "fp8_e4m3fn")

    gemm1_output_scale_size = (
        (2 * intermediate_size // 128) * max_padded_tokens * 4
    )  # float
    align_offset(4)
    gemm1_output_scale_offset = offset
    offset += gemm1_output_scale_size
    add_component(
        "gemm1_output_scale",
        gemm1_output_scale_size,
        gemm1_output_scale_offset,
        "float32",
    )

    activation_output_size = max_padded_tokens * intermediate_size * 1  # fp8 (uint8_t)
    align_offset(16)  # 16 bytes alignment for TMA
    activation_output_offset = offset
    offset += activation_output_size
    add_component(
        "activation_output",
        activation_output_size,
        activation_output_offset,
        "fp8_e4m3fn",
    )

    activation_output_scale_size = (
        (intermediate_size // 128) * max_padded_tokens * 4
    )  # float
    align_offset(4)
    activation_output_scale_offset = offset
    offset += activation_output_scale_size
    add_component(
        "activation_output_scale",
        activation_output_scale_size,
        activation_output_scale_offset,
        "float32",
    )

    gemm2_output_size = max_padded_tokens * hidden_size * 2  # BFloat16 (uint16_t)
    align_offset(16)  # 16 bytes alignment for TMA
    gemm2_output_offset = offset
    offset += gemm2_output_size
    add_component("gemm2_output", gemm2_output_size, gemm2_output_offset, "bfloat16")

    # Add estimated BMM workspace sizes
    align_offset(256)  # 256 bytes alignment for BMM workspaces
    bmm1_workspace_size = (
        max_padded_tokens * intermediate_size * 4
    )  # Rough estimate for BMM1
    bmm1_workspace_offset = offset
    offset += bmm1_workspace_size
    add_component("bmm1_workspace", bmm1_workspace_size, bmm1_workspace_offset, "char")

    align_offset(256)
    bmm2_workspace_size = max_padded_tokens * hidden_size * 4  # Rough estimate for BMM2
    bmm2_workspace_offset = offset
    offset += bmm2_workspace_size
    add_component("bmm2_workspace", bmm2_workspace_size, bmm2_workspace_offset, "char")

    total_workspace_size = offset

    # Add some safety margin (20%)
    total_workspace_size_with_margin = int(total_workspace_size * 1.2)

    # Add summary info to workspace_info
    workspace_info["summary"] = {
        "total_size_before_margin": total_workspace_size,
        "total_size_with_margin": total_workspace_size_with_margin,
        "safety_margin_percent": 20,
    }

    return total_workspace_size_with_margin, workspace_info


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
        routing_bias: torch.Tensor,
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        output1_scales_scalar: torch.Tensor,
        output1_scales_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        output2_scales_scalar: torch.Tensor,
        output: torch.Tensor,
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
        store_workspace_info: bool = False,
    ) -> None:
        # Calculate workspace size
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        workspace_size, workspace_info = _calculate_fp8_per_tensor_scale_workspace_size(
            seq_len, num_experts, hidden_size, intermediate_size, top_k, tile_tokens_dim
        )
        workspace_buffer = torch.empty(
            workspace_size, dtype=torch.uint8, device=hidden_states.device
        )

        # Call the C++ function
        moe_op.trtllm_fp8_per_tensor_scale_moe(
            routing_logits,
            routing_bias,
            hidden_states,
            gemm1_weights,
            output1_scales_scalar,
            output1_scales_gate_scalar,
            gemm2_weights,
            output2_scales_scalar,
            output,
            workspace_buffer,
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

        # Store workspace info on output tensor for debugging if requested
        if store_workspace_info:
            output._workspace_info = workspace_info
            output._workspace_buffer = workspace_buffer

    @register_fake_op("flashinfer::trtllm_fp8_per_tensor_scale_moe")
    def _fake_trtllm_fp8_per_tensor_scale_moe(
        routing_logits: torch.Tensor,
        routing_bias: torch.Tensor,
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        output1_scales_scalar: torch.Tensor,
        output1_scales_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        output2_scales_scalar: torch.Tensor,
        output: torch.Tensor,
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
        store_workspace_info: bool = False,
    ):
        # No-op for fake op since output is provided
        pass

    @register_custom_op(
        "flashinfer::trtllm_fp8_block_scale_moe",
        mutates_args=(""),
    )
    def trtllm_fp8_block_scale_moe_op(
        expert_logits: torch.Tensor,
        routing_bias: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_scales: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_scales: torch.Tensor,
        output: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_groups: int,
        top_k_groups: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling: float,
        tile_tokens_dim: int = 8,
        routing_method_type: int = 0,
    ) -> None:
        # Calculate workspace size
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        workspace_size, workspace_info = _calculate_fp8_per_tensor_scale_workspace_size(
            seq_len, num_experts, hidden_size, intermediate_size, top_k, tile_tokens_dim
        )
        workspace_buffer = torch.empty(
            workspace_size, dtype=torch.uint8, device=hidden_states.device
        )

        # Call the C++ function for block scale MoE
        moe_op.trtllm_fp8_block_scale_moe(
            expert_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_scales,
            gemm2_weights,
            gemm2_scales,
            output,
            workspace_buffer,
            num_experts,
            top_k,
            n_groups,
            top_k_groups,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling,
            tile_tokens_dim,
            routing_method_type,
        )

    @register_fake_op("flashinfer::trtllm_fp8_block_scale_moe")
    def _fake_trtllm_fp8_block_scale_moe(
        expert_logits: torch.Tensor,
        routing_bias: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_scales: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_scales: torch.Tensor,
        output: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_groups: int,
        top_k_groups: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling: float,
        tile_tokens_dim: int = 8,
        routing_method_type: int = 0,
    ):
        # No-op for fake op since output is provided
        pass

    return SimpleNamespace(
        trtllm_fp8_per_tensor_scale_moe=trtllm_fp8_per_tensor_scale_moe_op,
        trtllm_fp8_block_scale_moe=trtllm_fp8_block_scale_moe_op,
    )


def trtllm_fp8_per_tensor_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    output: torch.Tensor,
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
    store_workspace_info: bool = False,
) -> None:
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
        output: [seq_len, hidden_size] tensor to store the output
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
        store_workspace_info: Whether to store workspace info on output tensor for debugging (default: False)
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
        output,
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
        store_workspace_info,
    )


def trtllm_fp8_block_scale_moe(
    expert_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
    output: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_groups: int,
    top_k_groups: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling: float,
    tile_tokens_dim: int = 8,
    routing_method_type: int = 0,
) -> None:
    """FP8 block scale MoE operation.

    Args:
        expert_logits: [seq_len, num_experts] tensor of routing logits
        routing_bias: [num_experts] tensor of routing bias
        hidden_states: [seq_len, hidden_size] tensor of input hidden states
        hidden_states_scale: [hidden_size//128, seq_len] tensor of hidden states block scales
        gemm1_weights: [num_experts, 2*intermediate_size, hidden_size] tensor of first layer weights
        gemm1_scales: [num_experts, 2*intermediate_size//128, hidden_size//128] tensor of first layer block scales
        gemm2_weights: [num_experts, hidden_size, intermediate_size] tensor of second layer weights
        gemm2_scales: [num_experts, hidden_size//128, intermediate_size//128] tensor of second layer block scales
        output: [seq_len, hidden_size] tensor to store the output
        num_experts: Total number of experts
        top_k: Number of experts to route to per token
        n_groups: Number of expert groups
        top_k_groups: Number of groups to consider for top-k routing
        intermediate_size: Size of intermediate layer
        local_expert_offset: Offset of local experts in global expert space
        local_num_experts: Number of experts handled by this device
        routed_scaling: Scaling factor for routing
        tile_tokens_dim: Tile dimension for tokens (default: 8)
        routing_method_type: Type of routing method to use (default: 0)
    """
    return get_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(
        expert_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_scales,
        gemm2_weights,
        gemm2_scales,
        output,
        num_experts,
        top_k,
        n_groups,
        top_k_groups,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling,
        tile_tokens_dim,
        routing_method_type,
    )


# trtllmgen-moe-nvfp4
