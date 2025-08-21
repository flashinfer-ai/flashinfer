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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..artifacts import ArtifactPath, MetaInfoHash
from ..autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec, setup_cubin_loader, sm100a_nvcc_flags, sm90a_nvcc_flags
from ..jit.cubin_loader import get_cubin
from ..jit.cutlass_gemm.generate_kernels import generate_gemm_operations
from ..utils import (
    _check_shape_dtype_device,
    device_support_pdl,
    get_shuffle_matrix_a_row_indices,
    get_shuffle_matrix_sf_a_row_indices,
    register_custom_op,
    register_fake_op,
)
from .utils import (
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
    next_positive_power_of_2,
)


# The type of method in top-K routing, for use in torch custom op
# Please keep this in sync with the counterpart defined in include/flashinfer/trtllm/fused_moe/runner.h
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
    # TopK only (no softmax)
    TopK = (5,)
    # Unspecified
    Unspecified = 6


class DtypeTrtllmGen(IntEnum):
    def __new__(cls, block_format_bit, signed_bit, integer_bit, num_bits, uid):
        value = (
            (block_format_bit << 24)
            | (signed_bit << 20)
            | (integer_bit << 16)
            | (num_bits << 8)
            | uid
        )
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj

    # keep the values in sync with include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h
    Bfloat16 = (0, 1, 0, 16, 0)
    Bool = (0, 0, 1, 1, 1)
    E2m1 = (1, 1, 0, 4, 2)
    E2m3 = (1, 1, 0, 6, 3)
    E3m2 = (1, 1, 0, 6, 4)
    E4m3 = (0, 1, 0, 8, 5)
    E5m2 = (0, 1, 0, 8, 6)
    Fp16 = (0, 1, 0, 16, 7)
    Fp32 = (0, 1, 0, 32, 8)
    Int8 = (0, 1, 1, 8, 9)
    Int32 = (0, 1, 1, 32, 10)
    Int64 = (0, 1, 1, 64, 11)
    MxE2m1 = (1, 1, 0, 4, 12)
    MxE4m3 = (1, 1, 0, 8, 13)
    UE8m0 = (0, 0, 0, 8, 14)
    UInt8 = (0, 0, 1, 8, 15)
    UInt16 = (0, 0, 1, 16, 16)
    UInt32 = (0, 0, 1, 32, 17)
    UInt64 = (0, 0, 1, 64, 18)
    UInt128 = (0, 0, 1, 128, 19)
    Void = (0, 1, 0, 0, 20)


def trtllm_gen_dtype_has_scale(dtype: DtypeTrtllmGen) -> bool:
    if dtype in [
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.MxE2m1,
        DtypeTrtllmGen.MxE4m3,
    ]:
        return True
    else:
        return False


def deduce_trtllm_gen_tensor_dtype(
    x: torch.Tensor, scale: Optional[torch.Tensor]
) -> DtypeTrtllmGen:
    hidden_size = x.shape[-1]
    if x.dtype == torch.uint8:  # FIXME(siyuan): use torch.float4_e2m1x2 after torch 2.8
        hidden_size *= 2
    if x.dtype == torch.bfloat16:
        dtype = DtypeTrtllmGen.Bfloat16
    elif x.dtype == torch.float8_e4m3fn:
        dtype = DtypeTrtllmGen.E4m3 if scale is None else DtypeTrtllmGen.MxE4m3
    elif (
        x.dtype == torch.uint8
    ):  # FIXME(siyuan): use torch.float4_e2m1x2 after torch 2.8
        assert scale is not None, "Scale tensor must be provided for float4x2 input"
        if scale.shape[-1] == hidden_size // 16:
            dtype = DtypeTrtllmGen.E2m1
        else:
            dtype = DtypeTrtllmGen.MxE2m1
    else:
        raise ValueError("Unsupported trtllm-gen input tensor.")
    return dtype


# See MatrixLayout from include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/Enums.h
class WeightLayout(IntEnum):
    # K-major layout (default). [Mn, K]
    MajorK = 0
    # M-major for A and N-major for B. [K, Mn]
    MajorMn = 1
    # Layout is blocked along the K dimension. [K / blockK, Mn, blockK]
    # where blockK is fixed at 128B
    BlockMajorK = 2


# The type of gated activation function
# Please keep this in sync with the counterpart defined in include/flashinfer/trtllm/fused_moe/runner.h
class GatedActType(IntEnum):
    # SwiGlu
    SwiGlu = 0
    # GeGlu
    GeGlu = 1


def _maybe_get_cached_w3_w1_permute_indices(
    _cache_permute_indices,
    dst_w3_w1_weight: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: Union[None, int] = None,
) -> torch.Tensor:
    if dst_w3_w1_weight.shape not in _cache_permute_indices:
        # Get permute indices and chain them together
        permute0 = get_reorder_rows_for_gated_act_gemm_row_indices(dst_w3_w1_weight)
        if num_elts_per_sf is None:
            permute1 = get_shuffle_matrix_a_row_indices(
                dst_w3_w1_weight, epilogue_tile_m=epilogue_tile_m
            )
        else:
            permute1 = get_shuffle_matrix_sf_a_row_indices(
                dst_w3_w1_weight,
                epilogue_tile_m=epilogue_tile_m,
                num_elts_per_sf=num_elts_per_sf,
            )
        # Memoize permute indices as recompute is **very** costly
        _cache_permute_indices[dst_w3_w1_weight.shape] = permute0[permute1].to(
            dst_w3_w1_weight.device
        )
    permute_indices = _cache_permute_indices[dst_w3_w1_weight.shape]
    return permute_indices


def _maybe_get_cached_w2_permute_indices(
    _cache_permute_indices,
    dst_w2_weight: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: Union[None, int] = None,
) -> torch.Tensor:
    if dst_w2_weight.shape not in _cache_permute_indices:
        if num_elts_per_sf is None:
            permute_indices = get_shuffle_matrix_a_row_indices(
                dst_w2_weight, epilogue_tile_m
            ).to(dst_w2_weight.device)
        else:
            permute_indices = get_shuffle_matrix_sf_a_row_indices(
                dst_w2_weight,
                epilogue_tile_m=epilogue_tile_m,
                num_elts_per_sf=num_elts_per_sf,
            ).to(dst_w2_weight.device)
        # Memoize permute indices as recompute is **very** costly
        _cache_permute_indices[dst_w2_weight.shape] = permute_indices
    permute_indices = _cache_permute_indices[dst_w2_weight.shape]
    return permute_indices


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
    nvcc_flags = sm100a_nvcc_flags + [
        "-DCOMPILE_BLACKWELL_TMA_GEMMS",
        "-DCOMPILE_BLACKWELL_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
    ]
    return gen_cutlass_fused_moe_module(nvcc_flags, "100", use_fast_build)


def gen_cutlass_fused_moe_sm90_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = sm90a_nvcc_flags + [
        "-DCOMPILE_HOPPER_TMA_GEMMS",
        "-DCOMPILE_HOPPER_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
    ]
    return gen_cutlass_fused_moe_module(nvcc_flags, "90", use_fast_build)


def gen_cutlass_fused_moe_module(
    nvcc_flags: List[str], device_arch: str, use_fast_build: bool = False
) -> JitSpec:
    """
    Generate a JitSpec for the cutlass fused moe module.
    """
    output_dir = (
        jit_env.FLASHINFER_CSRC_DIR
        / f"nv_internal/tensorrt_llm/cutlass_instantiations/{device_arch}"
    )

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        generate_gemm_operations(
            output_dir,
            f"{device_arch};{device_arch}-real",
        )

    except Exception as e:
        raise RuntimeError(f"Failed to generate Cutlass kernels: {e}") from e

    return gen_jit_spec(
        f"fused_moe_{device_arch}",
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
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_stub.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_sm100_ops.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
            # Add all generated kernels
            *(output_dir / kernel for kernel in output_dir.rglob("*.generated.cu")),
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
        extra_cuda_cflags=nvcc_flags,
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
def get_cutlass_fused_moe_module(backend: str = "100", use_fast_build: bool = False):
    if backend == "100":
        FusedMoeRunner = gen_cutlass_fused_moe_sm100_module(
            use_fast_build
        ).build_and_load(class_name="FusedMoeRunner")
    elif backend == "90":
        FusedMoeRunner = gen_cutlass_fused_moe_sm90_module(
            use_fast_build
        ).build_and_load(class_name="FusedMoeRunner")
    else:
        raise ValueError(f"Invalid backend: {backend}")

    class MoERunner(TunableRunner):
        # avoid overhead of creating a new runner in forward pass
        runner_dict: Dict[
            Tuple[torch.dtype, torch.dtype, torch.dtype, bool, bool, bool], Any
        ] = dict()
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    (0,),
                    (0,),
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
            use_w4_group_scaling: bool,
            use_mxfp8_act_scaling: bool,
            min_latency_mode: bool,
            enable_pdl: bool,
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
            self.use_w4_group_scaling = use_w4_group_scaling
            self.use_mxfp8_act_scaling = use_mxfp8_act_scaling
            self.min_latency_mode = min_latency_mode
            self.enable_pdl = enable_pdl
            instance_key = (
                x_dtype,
                weight_dtype,
                output_dtype,
                use_deepseek_fp8_block_scale,
                use_w4_group_scaling,
                use_mxfp8_act_scaling,
            )

            if instance_key not in MoERunner.runner_dict:
                MoERunner.runner_dict[instance_key] = FusedMoeRunner(
                    x_dtype,
                    weight_dtype,
                    output_dtype,
                    use_deepseek_fp8_block_scale,
                    use_w4_group_scaling,
                    use_mxfp8_act_scaling,
                )

            self.fused_moe_runner = MoERunner.runner_dict[instance_key]

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            return list(range(self.fused_moe_runner.get_tactic_num()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
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
                kwargs["gemm_idx"],
                tactic,
                do_preparation,
                self.enable_pdl,
            )

        @classmethod
        @functools.lru_cache(maxsize=None)
        def refine_tuning_config(cls, tune_max_num_tokens: int):
            cls.tuning_config = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        (0,),
                        (0,),
                        get_last_power_of_2_num_tokens_buckets(tune_max_num_tokens),
                        lambda x: min(last_positive_power_of_2(x), tune_max_num_tokens),
                    ),
                )
            )

    @register_custom_op(
        "flashinfer::cutlass_fused_moe",
        mutates_args=(""),
    )
    def cutlass_fused_moe(
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
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        enable_alltoall: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4_group_scaling: bool = False,
        use_mxfp8_act_scaling: bool = False,
        min_latency_mode: bool = False,
        tune_max_num_tokens: int = 8192,
        enable_pdl: Optional[bool] = None,
    ) -> List[torch.Tensor]:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(input.device)
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
            use_w4_group_scaling=use_w4_group_scaling,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            min_latency_mode=min_latency_mode,
            enable_pdl=enable_pdl,
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
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            cluster_size,
            cluster_rank,
            enable_alltoall,
            min_latency_mode,
            [gemm_tactic_1, gemm_tactic_2],
            enable_pdl,
        )

        return result if min_latency_mode else [result]

    @register_fake_op("flashinfer::cutlass_fused_moe")
    def _fake_cutlass_fused_moe(
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
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        enable_alltoall: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4_group_scaling: bool = False,
        use_mxfp8_act_scaling: bool = False,
        min_latency_mode: bool = False,
        tune_max_num_tokens: int = 8192,
        enable_pdl: Optional[bool] = None,
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
        cutlass_fused_moe=cutlass_fused_moe,
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
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    tune_max_num_tokens: int = 8192,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Compute a Mixture of Experts (MoE) layer using CUTLASS backend.

    This function implements a fused MoE layer that combines expert selection, expert computation,
    and output combination into a single operation. It uses CUTLASS for efficient matrix multiplication
    and supports various data types and parallelism strategies.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape [num_tokens, hidden_size].
        Support float, float16, bfloat16, float8_e4m3fn and nvfp4.
        For FP8, the input must be quantized.
        For NVFP4, both quantized and non-quantized inputs are supported.

    token_selected_experts : torch.Tensor
        Indices of selected experts for each token.

    token_final_scales : torch.Tensor
        Scaling factors for each token's expert outputs.

    fc1_expert_weights : torch.Tensor
        GEMM1 weights for each expert.

    fc2_expert_weights : torch.Tensor
        GEMM2 weights for each expert.

    output_dtype : torch.dtype
        Desired output data type.

    quant_scales : List[torch.Tensor]
        Quantization scales for the operation.

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

    fc1_expert_biases : Optional[torch.Tensor]
        GEMM1 biases for each expert.

    fc2_expert_biases : Optional[torch.Tensor]
        GEMM1 biases for each expert.

    input_sf : Optional[torch.Tensor]
        Input scaling factor for quantization.

    swiglu_alpha : Optional[torch.Tensor]
        Swiglu alpha for swiglu activation.

    swiglu_beta : Optional[torch.Tensor]
        Swiglu beta for swiglu activation.

    swiglu_limit : Optional[torch.Tensor]
        Swiglu limit for swiglu activation.

    tp_size : int = 1
        Tensor parallelism size. Defaults to 1.

    tp_rank : int = 0
        Tensor parallelism rank. Defaults to 0.

    ep_size : int = 1
        Expert parallelism size. Defaults to 1.

    ep_rank : int = 0
        Expert parallelism rank. Defaults to 0.

    cluster_size : int = 1
        Cluster size. Defaults to 1.

    cluster_rank : int = 0
        Cluster rank. Defaults to 0.

    output : Optional[torch.Tensor] = None
        The output tensor, if not provided, will be allocated internally.

    enable_alltoall : bool = False
        Whether to enable all-to-all communication for expert outputs. Defaults to False.

    use_deepseek_fp8_block_scale : bool = False
        Whether to use FP8 block scaling. Defaults to False.

    use_w4_group_scaling : bool = False
        Whether to use W4A8 group scaling. Defaults to False.

    use_mxfp8_act_scaling : bool = False
        Whether to use MXFP8 activation scaling. Defaults to False.

    min_latency_mode : bool = False
        Whether to use minimum latency mode. Defaults to False.

    tune_max_num_tokens : int = 8192
        Maximum number of tokens for tuning. Defaults to 8192.

    Returns
    -------
    out: torch.Tensor
        Output tensor of shape [seq_len, hidden_size].


    Raises
    ------
    NotImplementedError:
        If any of the following features are requested but not implemented:
            - FP8 Block Scaling
            - W4A8 Group Scaling
            - Minimum Latency Mode

    Note
    ----
    - The function supports various data types including FP32, FP16, BF16, FP8, and NVFP4.
    - It implements both tensor parallelism and expert parallelism.
    - Currently, some advanced features like FP8 block scaling and minimum latency mode
        are not implemented for Blackwell architecture.
    """
    if use_deepseek_fp8_block_scale:
        raise NotImplementedError(
            "DeepSeek FP8 Block Scaling is not yet implemented in CUTLASS for Blackwell."
        )
    if min_latency_mode:
        raise NotImplementedError("min latency mode not yet implemented for Blackwell.")

    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)

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

    major, minor = torch.cuda.get_device_capability()
    device_arch = f"{major * 10 + minor}"

    return get_cutlass_fused_moe_module(device_arch).cutlass_fused_moe(
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
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        cluster_size,
        cluster_rank,
        enable_alltoall=enable_alltoall,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        use_w4_group_scaling=use_w4_group_scaling,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        min_latency_mode=min_latency_mode,
        tune_max_num_tokens=tune_max_num_tokens,
        enable_pdl=enable_pdl,
    )


# trtllmgen-moe-fp8


def trtllm_gen_fused_moe_sm100_module() -> JitSpec:
    # Fetch "flashinferMetaInfo.h" from the online kernel cache. This file
    # contains the `tllmGenBatchedGemmList` as the list of available kernels
    # online. It is included when compiling `trtllm_fused_moe_runner.cu`, etc.
    include_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/include"
    header_name = "flashinferMetaInfo"

    # use `get_cubin` to get "flashinferMetaInfo.h"
    metainfo = get_cubin(
        f"{include_path}/{header_name}", MetaInfoHash.TRTLLM_GEN_BMM, ".h"
    )
    # make sure "flashinferMetaInfo.h" is downloaded or cached
    assert metainfo, f"{header_name}.h not found"

    return gen_jit_spec(
        "fused_moe_trtllm_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_kernel_launcher.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_runner.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_deepseek.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_llama4.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_renormalize.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_dev_kernel.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_batched_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_ENABLE_CUDA",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
            f'-DTLLM_GEN_BMM_CUBIN_PATH=\\"{ArtifactPath.TRTLLM_GEN_BMM}\\"',
        ]
        + sm100a_nvcc_flags,
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            # link "include" sub-directory in cache
            jit_env.FLASHINFER_CUBIN_DIR / include_path,
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/include",
        ],
    )


@functools.cache
def get_trtllm_moe_sm100_module():
    module = trtllm_gen_fused_moe_sm100_module()
    moe_op = module.build_and_load()
    setup_cubin_loader(str(module.get_library_path()))

    class MoERunner(TunableRunner):
        dynamic_tensor_initializers = [
            lambda shapes, dtype, device: torch.empty(
                shapes, device=device, dtype=dtype
            ),  # output buffer, [num_tokens, hidden_size]
            lambda shapes, dtype, device: torch.rand(
                shapes, device=device, dtype=dtype
            ),  # routing_logits, [num_tokens, num_experts]
            lambda shapes, dtype, device: torch.empty(
                shapes, device=device, dtype=dtype
            ),  # topk_ids buffer. empty since routing_logits is used. [num_tokens, topk]
            lambda shapes, dtype, device: torch.empty(
                shapes, device=device, dtype=dtype
            ),  # expert_weights buffer. empty since routing_logits is used. [num_tokens, topk]
            lambda shapes, dtype, device: torch.randn(shapes, device=device).to(
                dtype
            ),  # hidden_states, [num_tokens, hidden_size]
            lambda shapes, dtype, device: torch.ones(shapes, device=device).to(
                dtype
            ),  # hidden_states_scale, [num_tokens, hidden_size // sf_vec_size]
        ]
        # their first dimension is num_tokens which will be tuned
        tuning_config_with_hidden_states_scales = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    (0, 1, 2, 3, 4, 5),
                    (0, 0, 0, 0, 0, 0),
                    get_last_power_of_2_num_tokens_buckets(1024, 8),
                    lambda x: min(last_positive_power_of_2(x), 1024),
                    dynamic_tensor_initializers,
                ),
            )
        )
        tuning_config_no_hidden_states_scales = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    (0, 1, 2, 3, 4),
                    (0, 0, 0, 0, 0),
                    get_last_power_of_2_num_tokens_buckets(1024, 8),
                    lambda x: min(last_positive_power_of_2(x), 1024),
                    dynamic_tensor_initializers[:5],
                ),
            ),
        )
        # cache the valid tactics to reduce the overhead of instantiating the runner
        # TODO(siyuan): directly cache the runners
        valid_tactics_dict = dict()

        def __init__(
            self,
            top_k: int,
            num_experts: int,
            dtype_act: DtypeTrtllmGen,
            dtype_weights: DtypeTrtllmGen,
            use_deepseek_fp8: bool,
            hidden_size: int,
            intermediate_size: int,
            gated_act_type: int,
            tile_tokens_dim: Optional[int] = None,
        ):
            self.num_experts = num_experts
            self.top_k = top_k
            self.dtype_act = dtype_act
            self.dtype_weights = dtype_weights
            self.use_deepseek_fp8 = use_deepseek_fp8
            self.top_k = top_k
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.gated_act_type = gated_act_type
            self.tile_tokens_dim = tile_tokens_dim

        def get_tile_tokens_dim(self, num_tokens: int, top_k: int):
            # Factor to account for the imbalance of the experts.
            # factor equals to the
            # max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
            # - 1.0 means perfect expert distribution.
            # - > 1.0 means some experts have more
            #     tokens than the perfect distribution.
            # - < 1.0 does not make sense.
            imbalance_factor = 1.3
            # Calculate the number of tokens per expert
            # assuming perfect distribution.
            num_tokens_per_expert = (num_tokens * top_k) // self.num_experts
            # Apply the imbalance factor.
            num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
            # And pad the number to the next power of 2.
            tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
            # Cap to 8-64 tokens per CTA tile
            # as it's the range supported by the kernel.
            tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

            return tile_tokens_dim

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            (
                output,
                routing_logits,
                topk_ids,
                expert_weights,
                hidden_states,
                *extra_inputs,
            ) = inputs
            num_tokens = routing_logits.shape[0]
            tile_tokens_dim = (
                self.get_tile_tokens_dim(num_tokens, self.top_k)
                if self.tile_tokens_dim is None
                else self.tile_tokens_dim
            )
            instance_key = (
                tile_tokens_dim,
                self.dtype_act,
                self.dtype_weights,
                self.use_deepseek_fp8,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                self.num_experts,
                self.gated_act_type,
                num_tokens,
            )
            if instance_key not in MoERunner.valid_tactics_dict:
                MoERunner.valid_tactics_dict[instance_key] = (
                    moe_op.trtllm_get_valid_moe_configs(*instance_key)
                )
            return MoERunner.valid_tactics_dict[instance_key]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            (
                output,
                routing_logits,
                topk_ids,
                expert_weights,
                hidden_states,
                *extra_inputs,
            ) = inputs
            num_tokens = routing_logits.shape[0]
            tile_tokens_dim = (
                self.get_tile_tokens_dim(num_tokens, self.top_k)
                if self.tile_tokens_dim is None
                else self.tile_tokens_dim
            )

            extra_input_idx = 0
            if trtllm_gen_dtype_has_scale(self.dtype_act):
                hidden_states_scale = extra_inputs[extra_input_idx]
                extra_input_idx += 1
            else:
                hidden_states_scale = None
            # sanity checks to ensure that dynamic tensors have the correct shapes
            assert output.shape[0] == num_tokens, (
                "output's first dimension must be batch size."
            )
            assert topk_ids.shape[0] == num_tokens, (
                "topk_ids's first dimension must be batch size."
            )
            assert expert_weights.shape[0] == num_tokens, (
                "expert_weights's first dimension must be batch size."
            )
            assert hidden_states.shape[0] == num_tokens, (
                "hidden_states's first dimension must be batch size."
            )
            assert hidden_states_scale is None or (
                hidden_states_scale.dim() == 2
                and hidden_states_scale.shape[0] == num_tokens
            ), "hidden_states_scale's first dimension must be batch size"

            # TODO(siyuan): support fp8
            moe_op.trtllm_fp4_block_scale_moe(
                routing_logits.to(torch.bfloat16),
                topk_ids,
                expert_weights,
                kwargs["routing_bias"],
                hidden_states,
                hidden_states_scale,  # hidden_states_scale
                kwargs["gemm1_weights"],
                kwargs["gemm1_weights_scale"],
                kwargs["gemm1_bias"],
                kwargs["gemm1_alpha"],
                kwargs["gemm1_beta"],
                kwargs["gemm1_clamp_limit"],
                kwargs["gemm2_weights"],
                kwargs["gemm2_weights_scale"],
                kwargs["gemm2_bias"],
                kwargs["output1_scale_scalar"],
                kwargs["output1_scale_gate_scalar"],
                kwargs["output2_scale_scalar"],
                self.num_experts,
                self.top_k,
                kwargs["n_group"],
                kwargs["topk_group"],
                self.intermediate_size,
                kwargs["local_expert_offset"],
                kwargs["num_local_experts"],
                kwargs["routed_scaling_factor"],
                tile_tokens_dim,
                kwargs["routing_method_type"],
                kwargs["enable_pdl"],
                kwargs["do_finalize"],
                self.gated_act_type,
                output,
                tactic,
            )

        @classmethod
        @functools.lru_cache(maxsize=None)
        def refine_tuning_config(cls, tune_max_num_tokens: int):
            cls.tuning_config_with_hidden_states_scales = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        (0, 1, 2, 3, 4, 5),
                        (0, 0, 0, 0, 0, 0),
                        get_last_power_of_2_num_tokens_buckets(tune_max_num_tokens, 8),
                        lambda x: min(last_positive_power_of_2(x), tune_max_num_tokens),
                        cls.dynamic_tensor_initializers,
                    ),
                )
            )
            cls.tuning_config_no_hidden_states_scales = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        (0, 1, 2, 3, 4),
                        (0, 0, 0, 0, 0),
                        get_last_power_of_2_num_tokens_buckets(tune_max_num_tokens, 8),
                        lambda x: min(last_positive_power_of_2(x), tune_max_num_tokens),
                        cls.dynamic_tensor_initializers[:5],
                    ),
                ),
            )

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
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)
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
            enable_pdl,
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
        enable_pdl: Optional[bool] = None,
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
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)
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
            enable_pdl,
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
        enable_pdl: Optional[bool] = None,
    ):
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]

        return [hidden_states.new_empty([seq_len, hidden_size], dtype=torch.bfloat16)]

    @register_custom_op(
        "flashinfer::trtllm_fp4_block_scale_moe",
        mutates_args=(""),
    )
    def trtllm_fp4_block_scale_moe_op(
        routing_logits: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: Optional[torch.Tensor],
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: Optional[torch.Tensor],
        output1_scale_gate_scalar: Optional[torch.Tensor],
        output2_scale_scalar: Optional[torch.Tensor],
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
        enable_pdl: Optional[bool] = None,
        gated_act_type: int = 0,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 1024,
    ) -> List[torch.Tensor]:
        if routing_logits is None:
            assert topk_ids is not None, (
                "either topk_ids or routing_logits must be provided."
            )
            assert topk_ids.dtype == torch.int32, "topk_ids must be an int32 tensor."
            routing_dtype = torch.bfloat16
        else:
            routing_dtype = routing_logits.dtype
        hidden_size = hidden_states.shape[-1]
        if hidden_states.dtype == torch.uint8:
            hidden_size = hidden_size * 2
        num_tokens = hidden_states.shape[0]

        # workspace buffers required by trtllm-gen
        if topk_ids is None:
            topk_ids = torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=hidden_states.device
            )
        if expert_weights is None:
            expert_weights = torch.empty(
                num_tokens, top_k, dtype=routing_dtype, device=hidden_states.device
            )
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)
        if output is None:
            output = torch.empty(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )

        tuner = AutoTuner.get()
        MoERunner.refine_tuning_config(tune_max_num_tokens)
        dtype_act = deduce_trtllm_gen_tensor_dtype(hidden_states, hidden_states_scale)
        dtype_weights = deduce_trtllm_gen_tensor_dtype(
            gemm1_weights, gemm1_weights_scale
        )
        moe_runner = MoERunner(
            top_k=top_k,
            num_experts=num_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            use_deepseek_fp8=False,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gated_act_type=gated_act_type,
            # NOTE(siyuan): do not fix the tile_tokens_dim to let tunnable runner decide the tile_tokens_dim itself.
            # however, when the user chooses a different heuristic for tile_tokens_dim, the autotuner will fail to find the correct cached tactics.
            # tile_tokens_dim=tile_tokens_dim,
        )
        tunning_config = (
            MoERunner.tuning_config_no_hidden_states_scales
            if hidden_states_scale is None
            else MoERunner.tuning_config_with_hidden_states_scales
        )
        inputs = [
            output,
            routing_logits,
            topk_ids,
            expert_weights,
            hidden_states,
        ]
        if hidden_states_scale is not None:
            inputs.append(hidden_states_scale)

        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_fp4_block_scale_moe",
            [moe_runner],
            tunning_config,
            inputs,
            num_local_experts=num_experts,
            routing_bias=routing_bias,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_bias=gemm1_bias,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm2_bias=gemm2_bias,
            output1_scale_scalar=output1_scale_scalar,
            output1_scale_gate_scalar=output1_scale_gate_scalar,
            output2_scale_scalar=output2_scale_scalar,
            n_group=n_group,
            topk_group=topk_group,
            local_expert_offset=local_expert_offset,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=routing_method_type,
            enable_pdl=enable_pdl,
            do_finalize=do_finalize,
            gated_act_type=gated_act_type,
        )

        # Call the C++ function for block scale MoE
        output = moe_op.trtllm_fp4_block_scale_moe(
            routing_logits,
            topk_ids,
            expert_weights,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_bias,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            gemm2_weights,
            gemm2_weights_scale,
            gemm2_bias,
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
            enable_pdl,
            gated_act_type,
            output,
            tactic,
        )

        return output

    @register_fake_op("flashinfer::trtllm_fp4_block_scale_moe")
    def _fake_trtllm_fp4_block_scale_moe(
        routing_logits: torch.Tensor,
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: Optional[torch.Tensor],
        output1_scale_gate_scalar: Optional[torch.Tensor],
        output2_scale_scalar: Optional[torch.Tensor],
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
        enable_pdl: bool,
        gated_act_type: int,
        output: Optional[torch.Tensor],
        tune_max_num_tokens: int,
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
    enable_pdl: Optional[bool] = None,
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
        enable_pdl: Whether to enable Programmatic Dependent Launch (PDL). Auto-enabled for >= sm90.

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
        enable_pdl,
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
    enable_pdl: Optional[bool] = None,
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
        enable_pdl: Whether to enable Programmatic Dependent Launch (PDL). Auto-enabled for >= sm90.
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
        enable_pdl,
    )


def trtllm_fp4_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
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
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    gated_act_type: int = 0,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 1024,
) -> List[torch.Tensor]:
    """FP4 block scale MoE operation.

    Args:
        routing_logits (torch.Tensor): shape [seq_len, num_experts]
            Input tensor of routing logits. Supports float32, bfloat16.
        routing_bias (Optional[torch.Tensor]): shape [num_experts]
            Tensor of routing bias. Can be None for some routing methods. Must be the same type as routing logits.
        hidden_states (torch.Tensor): shape [seq_len, hidden_size // 2 if nvfp4 else hidden_size]
            Tensor of input hidden states. Supports bfloat16, mxfp8, and nvfp4 (packed into uint8)
        hidden_states_scale (Optional[torch.Tensor]): shape [seq_len, hidden_size // (32 if mxfp8, 16 if mxfp4)]
            Scale tensor of mxfp8 / nvfp4 hidden states. Dtype must be float8.
        gemm1_weights (torch.Tensor): shape [num_experts, 2 * intermediate_size, hidden_size // 2]
            Tensor of FC1 weights. Dtype must be uint8 (packed fp4)
        gemm1_weights_scale (torch.Tensor): shape [num_experts, 2 * intermediate_size, hidden_size // (32 if mxfp4 else 16)]
            Scale tensor of FC1 weights. Dtype must be float8.
        gemm1_bias (Optional[torch.Tensor]): shape [num_experts, 2 * intermediate_size]
            Tensor of FC1 biases. Dtype is float32.
        gemm1_alpha (Optional[torch.Tensor]): shape [num_experts]
            Tensor of swiglu alpha. Dtype is float32.
        gemm1_beta (Optional[torch.Tensor]): shape [num_experts]
            Tensor of swiglu beta. Dtype is float32.
        gemm1_clamp_limit (Optional[torch.Tensor]): shape [num_experts]
            Tensor of swiglu clamp limit. Dtype is float32.
        gemm2_weights (torch.Tensor): shape [num_experts, hidden_size, intermediate_size]
            Tensor of FC2 weights. Dtype must be uint8 (packed fp4)
        gemm2_weights_scale (torch.Tensor): shape [num_experts, hidden_size, intermediate_size // (32 if mxfp4 else 16)]
            Scale tensor of FC2 weights. Dtype must be float8.
        gemm2_bias (Optional[torch.Tensor]): shape [num_experts, hidden_size]
            Tensor of FC2 biases. Dtype is float32.
        output1_scale_scalar (Optional[torch.Tensor]): shape [local_num_experts]
            Tensor of scaling factors for first layer activation output
        output1_scale_gate_scalar (Optional[torch.Tensor]): shape [local_num_experts]
            Tensor of scaling factors for first layer gate output
        output2_scale_scalar (Optional[torch.Tensor]): shape [local_num_experts]
            Tensor of scaling factors for second layer output
        num_experts (int): Total number of experts
        top_k (int): Number of experts to route to per token
        n_group (Optional[int]): Number of expert groups (can be None for some routing methods)
        topk_group (Optional[int]): Number of groups to consider for top-k routing (can be None for some routing methods)
        intermediate_size (int): Size of intermediate layer
        local_expert_offset (int): Offset of local experts in global expert space
        local_num_experts (int): Number of experts handled by this device
        routed_scaling_factor (Optional[float]): Scaling factor for routing (can be None for some routing methods)
        tile_tokens_dim (int): Tile dimension for tokens (default: 8)
        routing_method_type (int): Type of routing method to use (default: 0)
            - 0: Default (Softmax -> TopK)
            - 1: Renormalize (TopK -> Softmax)
            - 2: DeepSeekV3 (Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts)
            - 3: Llama4 (Top1 -> Sigmoid)
            - 4: RenormalizeNaive (Softmax -> TopK -> Renormalize)
        do_finalize (bool): Whether to finalize the output (default: False)
        enable_pdl (Optional[bool]): Whether to enable Programmatic Dependent Launch (PDL). Auto-enabled for >= sm90.
        gated_act_type (int): Type of gated activation function (default: 0)
            - 0: SwiGlu
            - 1: GeGlu
        tune_max_num_tokens(int): Maximum number of tokens for tuning. (default: 1024)
        output (Optional[torch.Tensor]): shape [seq_len, hidden_size]
            Optional inplace output tensor.
    Returns:
        List[torch.Tensor]: List of output tensors. If do_finalize=True, returns the final MoE output.
            Otherwise, returns intermediate results (gemm2_output, expert_weights, expanded_idx_to_permuted_idx) that need further processing.
    """
    return get_trtllm_moe_sm100_module().trtllm_fp4_block_scale_moe(
        routing_logits,
        None,
        None,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
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
        enable_pdl,
        gated_act_type,
        output,
        tune_max_num_tokens,
    )


def trtllm_fp4_block_scale_routed_moe(
    topk_ids: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
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
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    gated_act_type: int = 0,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 1024,
) -> List[torch.Tensor]:
    """FP4 block scale MoE operation.

    Args:
        topk_ids (torch.Tensor): shape [seq_len, top_k]
            Tensor of top-k indices and expert weights. Dtype must be int32.
            It must represent a packed value. The most significant 16/32 bits represent the score and
            the least significant 16 bits represent the index of the chosen expert (unsigned).
        routing_bias (Optional[torch.Tensor]): shape [num_experts]
            Tensor of routing bias. Can be None for some routing methods. Must be the same type as routing logits.
        hidden_states (torch.Tensor): shape [seq_len, hidden_size // 2 if nvfp4 else hidden_size]
            Tensor of input hidden states. Supports bfloat16, mxfp8, and nvfp4 (packed into uint8)
        hidden_states_scale (Optional[torch.Tensor]): shape [seq_len, hidden_size // (32 if mxfp8, 16 if mxfp4)]
            Scale tensor of mxfp8 / nvfp4 hidden states. Dtype must be float8.
        gemm1_weights (torch.Tensor): shape [num_experts, 2 * intermediate_size, hidden_size // 2]
            Tensor of FC1 weights. Dtype must be uint8 (packed fp4)
        gemm1_weights_scale (torch.Tensor): shape [num_experts, 2 * intermediate_size, hidden_size // (32 if mxfp4 else 16)]
            Scale tensor of FC1 weights. Dtype must be float8.
        gemm2_weights (torch.Tensor): shape [num_experts, hidden_size, intermediate_size]
            Tensor of FC2 weights. Dtype must be uint8 (packed fp4)
        gemm2_weights_scale (torch.Tensor): shape [num_experts, hidden_size//128, intermediate_size//128]
            Scale tensor of FC2 weights. Dtype must be float8.
        output1_scale_scalar (Optional[torch.Tensor]): shape [local_num_experts]
            Tensor of scaling factors for first layer activation output
        output1_scale_gate_scalar (Optional[torch.Tensor]): shape [local_num_experts]
            Tensor of scaling factors for first layer gate output
        output2_scale_scalar (Optional[torch.Tensor]): shape [local_num_experts]
            Tensor of scaling factors for second layer output
        num_experts (int): Total number of experts
        top_k (int): Number of experts to route to per token
        n_group (Optional[int]): Number of expert groups (can be None for some routing methods)
        topk_group (Optional[int]): Number of groups to consider for top-k routing (can be None for some routing methods)
        intermediate_size (int): Size of intermediate layer
        local_expert_offset (int): Offset of local experts in global expert space
        local_num_experts (int): Number of experts handled by this device
        routed_scaling_factor (Optional[float]): Scaling factor for routing (can be None for some routing methods)
        tile_tokens_dim (int): Tile dimension for tokens (default: 8)
        routing_method_type (int): Type of routing method to use (default: 0)
            - 0: Default (Softmax -> TopK)
            - 1: Renormalize (TopK -> Softmax)
            - 2: DeepSeekV3 (Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts)
            - 3: Llama4 (Top1 -> Sigmoid)
            - 4: RenormalizeNaive (Softmax -> TopK -> Renormalize)
        do_finalize (bool): Whether to finalize the output (default: False)
        gated_act_type (int): Type of gated activation function (default: 0)
            - 0: SwiGlu
            - 1: GeGlu
        tune_max_num_tokens(int): Maximum number of tokens for tuning. (default: 1024)
        output (Optional[torch.Tensor]): shape [seq_len, hidden_size]
            Optional inplace output tensor.

    Returns:
        List[torch.Tensor]: List of output tensors. If do_finalize=True, returns the final MoE output.
            Otherwise, returns intermediate results (gemm2_output, expert_weights, expanded_idx_to_permuted_idx) that need further processing.
    """
    return get_trtllm_moe_sm100_module().trtllm_fp4_block_scale_moe(
        None,
        topk_ids,
        None,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
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
        enable_pdl,
        gated_act_type,
        output,
        tune_max_num_tokens,
    )
