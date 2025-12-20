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

from ..api_logging import flashinfer_api
from ..autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..jit.cpp_ext import is_cuda_version_at_least
from ..jit.core import logger
from ..jit import (
    setup_cubin_loader,
)
from ..jit.fused_moe import (
    gen_cutlass_fused_moe_sm120_module,
    gen_cutlass_fused_moe_sm103_module,
    gen_cutlass_fused_moe_sm100_module,
    gen_cutlass_fused_moe_sm90_module,
    gen_cutlass_fused_moe_sm89_module,
    gen_trtllm_gen_fused_moe_sm100_module,
)
from ..utils import (
    check_shape_dtype_device,
    device_support_pdl,
    get_shuffle_matrix_a_row_indices,
    get_shuffle_matrix_sf_a_row_indices,
    register_custom_op,
    register_fake_op,
    get_compute_capability,
)
from .utils import (
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
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


# Copied from csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/common.h
class ActivationType(IntEnum):
    Gelu = 0
    Relu = 1
    Silu = 2
    Swiglu = 3
    Geglu = 4
    SwigluBias = 5
    Relu2 = 6
    Identity = 7
    InvalidType = 8


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
    MxInt4 = (1, 1, 1, 4, 14)
    UE8m0 = (0, 0, 0, 8, 15)
    UInt8 = (0, 0, 1, 8, 16)
    UInt16 = (0, 0, 1, 16, 17)
    UInt32 = (0, 0, 1, 32, 18)
    UInt64 = (0, 0, 1, 64, 19)
    UInt128 = (0, 0, 1, 128, 20)
    Void = (0, 1, 0, 0, 21)


def trtllm_gen_dtype_has_scale(dtype: DtypeTrtllmGen) -> bool:
    if dtype in [
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.MxE2m1,
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.MxInt4,
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


@functools.cache
def is_trtllm_moe_supported(
    dtype_weights: DtypeTrtllmGen,
    dtype_act: DtypeTrtllmGen,
    quant_method: Optional[str] = None,
) -> bool:
    arch = get_compute_capability(torch.cuda.current_device())
    if arch[0] < 10:
        return False
    if dtype_weights not in [
        DtypeTrtllmGen.Bfloat16,
        DtypeTrtllmGen.E4m3,
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.MxE2m1,
    ]:
        return False
    if (
        dtype_weights == DtypeTrtllmGen.Bfloat16
        and dtype_act != DtypeTrtllmGen.Bfloat16
    ):
        return False
    if dtype_weights == DtypeTrtllmGen.E4m3 and dtype_act != DtypeTrtllmGen.E4m3:
        return False
    if dtype_weights == DtypeTrtllmGen.E2m1 and dtype_act != DtypeTrtllmGen.E2m1:
        return False
    if dtype_weights == DtypeTrtllmGen.MxE2m1 and dtype_act not in [
        DtypeTrtllmGen.MxE2m1,
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.Bfloat16,
    ]:
        return False
    return True


def _maybe_get_cached_w3_w1_permute_indices(
    _cache_permute_indices,
    dst_w3_w1_weight: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: Union[None, int] = None,
) -> torch.Tensor:
    # Create a unique cache key (weight_type, weight_shape)
    cache_key = ("w3_w1", dst_w3_w1_weight.shape)
    if cache_key not in _cache_permute_indices:
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
        _cache_permute_indices[cache_key] = permute0[permute1].to(
            dst_w3_w1_weight.device
        )
    permute_indices = _cache_permute_indices[cache_key]
    return permute_indices


def get_w2_permute_indices_with_cache(
    _cache_permute_indices,
    dst_w2_weight: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: Union[None, int] = None,
) -> torch.Tensor:
    # Create a unique cache key (weight_type, weight_shape)
    cache_key = ("w2", dst_w2_weight.shape)
    if cache_key not in _cache_permute_indices:
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
        _cache_permute_indices[cache_key] = permute_indices
    permute_indices = _cache_permute_indices[cache_key]
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


@functools.cache
def get_cutlass_fused_moe_module(backend: str = "100", use_fast_build: bool = False):
    if backend in ("120", "121"):
        module = gen_cutlass_fused_moe_sm120_module(use_fast_build).build_and_load()
    elif backend == "103":
        module = gen_cutlass_fused_moe_sm103_module(use_fast_build).build_and_load()
    elif backend in ("100", "110"):
        module = gen_cutlass_fused_moe_sm100_module(use_fast_build).build_and_load()
    elif backend == "90":
        module = gen_cutlass_fused_moe_sm90_module(use_fast_build).build_and_load()
    elif backend == "89":
        module = gen_cutlass_fused_moe_sm89_module(use_fast_build).build_and_load()
    else:
        raise ValueError(f"Invalid backend: {backend}")

    # Set DeepGEMM JIT include directories after module is loaded
    from ..jit import env as jit_env

    deepgemm_include_dir = str(
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "tensorrt_llm"
    )
    module.set_deepgemm_jit_include_dirs([deepgemm_include_dir])

    class MoERunner(TunableRunner):
        # avoid overhead of creating a new runner in forward pass
        runner_dict: Dict[
            Tuple[torch.dtype, torch.dtype, torch.dtype, bool, bool, bool, bool], Any
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
            activation_type: ActivationType,
            use_packed_weights: bool,
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
            self.use_packed_weights = use_packed_weights
            instance_key = (
                x_dtype,
                weight_dtype,
                output_dtype,
                use_deepseek_fp8_block_scale,
                use_w4_group_scaling,
                use_mxfp8_act_scaling,
                use_packed_weights,
            )
            self.activation_type = activation_type
            # Set by tuning flow to indicate which GEMM stage (1 or 2) to filter tactics for
            self.gemm_idx_for_tuning: Optional[int] = None

            if instance_key not in MoERunner.runner_dict:
                MoERunner.runner_dict[instance_key] = module.init(
                    x_dtype,
                    weight_dtype,
                    output_dtype,
                    use_deepseek_fp8_block_scale,
                    use_w4_group_scaling,
                    use_mxfp8_act_scaling,
                    use_packed_weights,
                )

            self.fused_moe_runner = MoERunner.runner_dict[instance_key]

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            # Prefer filtering tactics by GEMM stage to avoid invalid combos during tuning
            try:
                gemm1_count = self.fused_moe_runner.get_gemm1_tactic_count()
                gemm2_count = self.fused_moe_runner.get_gemm2_tactic_count()
                total = gemm1_count + gemm2_count
            except Exception:
                return list(range(self.fused_moe_runner.get_tactic_num()))

            stage = getattr(self, "gemm_idx_for_tuning", None)
            if stage == 1:
                return list(range(gemm1_count))
            if stage == 2:
                return list(range(gemm1_count, gemm1_count + gemm2_count))
            return list(range(total))

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
                self.activation_type,
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
        activation_type: ActivationType = ActivationType.Swiglu,
        use_packed_weights: bool = False,
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
            activation_type=activation_type,
            use_packed_weights=use_packed_weights,
        )

        # Limit tactics to GEMM1 during tuning
        moe_runner.gemm_idx_for_tuning = 1
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

        # Limit tactics to GEMM2 during tuning
        moe_runner.gemm_idx_for_tuning = 2
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
        num_active_experts_per_node = torch.empty(
            (1,), dtype=torch.int32, device=input.device
        )
        experts_to_token_score = torch.empty(
            (fc2_expert_weights.shape[0], input.shape[0]),
            dtype=torch.float32,
            device=input.device,
        )
        active_expert_global_ids = torch.empty(
            (fc2_expert_weights.shape[0],),
            dtype=torch.int32,
            device=input.device,
        )
        min_latency_output = (
            [
                num_active_experts_per_node,
                experts_to_token_score,
                active_expert_global_ids,
            ]
            if min_latency_mode
            else []
        )
        run_moe(
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
            *min_latency_output,
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
            activation_type,
        )

        return (
            output
            if min_latency_mode
            else [
                output,
                num_active_experts_per_node,
                experts_to_token_score,
                active_expert_global_ids,
            ]
        )

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
        use_packed_weights: bool = False,
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
@flashinfer_api
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
    use_packed_weights: bool = False,
    tune_max_num_tokens: int = 8192,
    enable_pdl: Optional[bool] = None,
    activation_type: ActivationType = ActivationType.Swiglu,
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

    use_packed_weights : bool = False
        Whether to use packed uint4x2 weights passed as packed uint8 values. Defaults to False.

    tune_max_num_tokens : int = 8192
        Maximum number of tokens for tuning. Defaults to 8192.

    activation_type: ActivationType = ActivationType.Swiglu
        Activation to apply on for GEMM1, note that Relu2 means non-gated GEMM1

    Returns
    -------
    out: torch.Tensor
        Output tensor of shape [seq_len, hidden_size].


    Raises
    ------
    NotImplementedError:
        If any of the following features are requested but not implemented:
            - Minimum Latency Mode

    Note
    ----
    - The function supports various data types including FP32, FP16, BF16, FP8, and NVFP4.
    - It implements both tensor parallelism and expert parallelism.
    - Currently, some advanced features like FP8 block scaling and minimum latency mode
        are not implemented for Blackwell architecture.
    """
    major, minor = torch.cuda.get_device_capability()
    device_arch = f"{major * 10 + minor}"

    if min_latency_mode:
        raise NotImplementedError("min latency mode not yet implemented for Blackwell.")

    if use_deepseek_fp8_block_scale:
        if device_arch != "90":
            raise NotImplementedError(
                "FP8 block scaling not yet implemented for Blackwell."
            )
        elif not is_cuda_version_at_least("12.8"):
            raise NotImplementedError(
                "FP8 block scaling not implemented for CUDA 12.6 or lower."
            )

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
        check_shape_dtype_device(
            output, output_shape, output_dtype, input.device, "output"
        )

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
        use_packed_weights=use_packed_weights,
        enable_alltoall=enable_alltoall,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        use_w4_group_scaling=use_w4_group_scaling,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        min_latency_mode=min_latency_mode,
        tune_max_num_tokens=tune_max_num_tokens,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
    )


# trtllmgen-moe-fp8


@functools.cache
def get_trtllm_moe_sm100_module():
    module = gen_trtllm_gen_fused_moe_sm100_module()
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
                    get_last_power_of_2_num_tokens_buckets(8192, 1),
                    lambda x: min(last_positive_power_of_2(x), 8192),
                    dynamic_tensor_initializers,
                ),
            )
        )
        tuning_config_no_hidden_states_scales = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    (0, 1, 2, 3, 4),
                    (0, 0, 0, 0, 0),
                    get_last_power_of_2_num_tokens_buckets(8192, 1),
                    lambda x: min(last_positive_power_of_2(x), 8192),
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
            num_local_experts: int,
            dtype_act: DtypeTrtllmGen,
            dtype_weights: DtypeTrtllmGen,
            use_deepseek_fp8: bool,
            hidden_size: int,
            intermediate_size: int,
            gated_act_type: int = GatedActType.SwiGlu,
            use_shuffled_weight: bool = False,
            weight_layout: int = WeightLayout.MajorK,
            use_packed_weights: bool = False,
        ):
            self.num_local_experts = num_local_experts
            self.top_k = top_k
            self.dtype_act = dtype_act
            self.dtype_weights = dtype_weights
            self.use_deepseek_fp8 = use_deepseek_fp8
            self.top_k = top_k
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.gated_act_type = GatedActType(gated_act_type)
            self.use_shuffled_weight = use_shuffled_weight
            self.weight_layout = WeightLayout(weight_layout)
            self.use_packed_weights = use_packed_weights

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

            instance_key = (
                self.dtype_act,
                self.dtype_weights,
                self.use_deepseek_fp8,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                self.num_local_experts,
                self.gated_act_type,
                self.use_shuffled_weight,
                self.weight_layout,
                num_tokens,
            )
            if instance_key not in MoERunner.valid_tactics_dict:
                try:
                    valid_tactics = moe_op.trtllm_get_valid_moe_configs(*instance_key)
                except Exception as e:
                    logger.debug(
                        f"[Autotuner]: Failed to get valid tactics for {instance_key}. Error occurred: {e}"
                    )
                    return []
                MoERunner.valid_tactics_dict[instance_key] = valid_tactics
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
            # Choose the appropriate operation based on data types
            if self.dtype_weights == DtypeTrtllmGen.Bfloat16:
                # BF16 operations
                moe_op.trtllm_bf16_moe(
                    routing_logits,
                    kwargs["routing_bias"],
                    hidden_states,
                    kwargs["gemm1_weights"],
                    kwargs["gemm2_weights"],
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["use_shuffled_weight"],
                    kwargs["weight_layout"],
                    kwargs["enable_pdl"],
                    [-1, -1] if tactic == -1 else tactic,
                )
            elif (
                self.dtype_act == DtypeTrtllmGen.E4m3
                and self.dtype_weights == DtypeTrtllmGen.E4m3
            ):
                # FP8 operations
                if self.use_deepseek_fp8:
                    # FP8 block scale
                    current_num_tokens = hidden_states.shape[0]
                    current_hidden_size = hidden_states.shape[1]
                    current_hidden_states_scale = torch.full(
                        (current_hidden_size // 128, current_num_tokens),
                        2.0,
                        dtype=torch.float,
                        device=hidden_states.device,
                    )
                    moe_op.trtllm_fp8_block_scale_moe(
                        routing_logits,
                        kwargs["routing_bias"],
                        hidden_states,
                        current_hidden_states_scale,
                        kwargs["gemm1_weights"],
                        kwargs["gemm1_weights_scale"],
                        kwargs["gemm2_weights"],
                        kwargs["gemm2_weights_scale"],
                        output,
                        kwargs["num_experts"],
                        self.top_k,
                        kwargs["n_group"],
                        kwargs["topk_group"],
                        self.intermediate_size,
                        kwargs["local_expert_offset"],
                        self.num_local_experts,
                        kwargs["routed_scaling_factor"],
                        kwargs["routing_method_type"],
                        kwargs["use_shuffled_weight"],
                        kwargs["weight_layout"],
                        kwargs["enable_pdl"],
                        [-1, -1] if tactic == -1 else tactic,
                    )
                else:
                    # FP8 per tensor scale
                    moe_op.trtllm_fp8_per_tensor_scale_moe(
                        routing_logits,
                        kwargs["routing_bias"],
                        hidden_states,
                        kwargs["gemm1_weights"],
                        kwargs["output1_scales_scalar"],
                        kwargs["output1_scales_gate_scalar"],
                        kwargs["gemm2_weights"],
                        kwargs["output2_scales_scalar"],
                        output,
                        kwargs["num_experts"],
                        self.top_k,
                        kwargs["n_group"],
                        kwargs["topk_group"],
                        self.intermediate_size,
                        kwargs["local_expert_offset"],
                        self.num_local_experts,
                        kwargs["routed_scaling_factor"],
                        kwargs["use_routing_scales_on_input"],
                        kwargs["routing_method_type"],
                        kwargs["enable_pdl"],
                        [-1, -1] if tactic == -1 else tactic,
                    )
            elif (
                self.dtype_act == DtypeTrtllmGen.Bfloat16
                and self.dtype_weights == DtypeTrtllmGen.MxInt4
            ):
                moe_op.trtllm_mxint4_block_scale_moe(
                    routing_logits,
                    kwargs["routing_bias"],
                    hidden_states,
                    kwargs["gemm1_weights"],
                    kwargs["gemm1_weights_scale"],
                    kwargs["gemm1_alpha"],
                    kwargs["gemm1_beta"],
                    kwargs["gemm1_clamp_limit"],
                    kwargs["gemm2_weights"],
                    kwargs["gemm2_weights_scale"],
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["enable_pdl"],
                    output,
                    [-1, -1] if tactic == -1 else tactic,
                )
            else:
                moe_op.trtllm_fp4_block_scale_moe(
                    routing_logits,
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
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["enable_pdl"],
                    kwargs["do_finalize"],
                    self.gated_act_type,
                    output,
                    [-1, -1] if tactic == -1 else tactic,
                )

        @classmethod
        @functools.lru_cache(maxsize=None)
        def refine_tuning_config(cls, tune_max_num_tokens: int):
            cls.tuning_config_with_hidden_states_scales = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        (0, 1, 2, 3, 4, 5),
                        (0, 0, 0, 0, 0, 0),
                        get_last_power_of_2_num_tokens_buckets(tune_max_num_tokens, 1),
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
                        get_last_power_of_2_num_tokens_buckets(tune_max_num_tokens, 1),
                        lambda x: min(last_positive_power_of_2(x), tune_max_num_tokens),
                        cls.dynamic_tensor_initializers[:5],
                    ),
                ),
            )

    @register_custom_op(
        "flashinfer::trtllm_bf16_moe",
        mutates_args=(""),
    )
    def trtllm_bf16_moe_op(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        use_shuffled_weight: bool,
        weight_layout: int,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
    ) -> torch.Tensor:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)

        # Use AutoTuner to select the best tactic
        tuner = AutoTuner.get()
        MoERunner.refine_tuning_config(tune_max_num_tokens)

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]

        # Create workspace buffers
        output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
        )
        topk_ids = torch.empty(
            num_tokens, top_k, dtype=torch.int32, device=hidden_states.device
        )
        expert_weights = torch.empty(
            num_tokens, top_k, dtype=routing_logits.dtype, device=hidden_states.device
        )

        dtype_act = DtypeTrtllmGen.Bfloat16
        dtype_weights = DtypeTrtllmGen.Bfloat16

        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=local_num_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            use_deepseek_fp8=False,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            weight_layout=weight_layout,
            use_shuffled_weight=use_shuffled_weight,
            gated_act_type=GatedActType.SwiGlu,  # Default for BF16
        )

        inputs = [output, routing_logits, topk_ids, expert_weights, hidden_states]

        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_bf16_moe",
            [moe_runner],
            MoERunner.tuning_config_no_hidden_states_scales,
            inputs,
            routing_bias=routing_bias,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=routing_method_type,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            enable_pdl=enable_pdl,
        )

        # Call the C++ function with the selected tactic
        result = moe_op.trtllm_bf16_moe(
            routing_logits,
            routing_bias,
            hidden_states,
            gemm1_weights,
            gemm2_weights,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            use_shuffled_weight,
            weight_layout,
            enable_pdl,
            [-1, -1] if tactic == -1 else tactic,
        )
        return result

    @register_fake_op("flashinfer::trtllm_bf16_moe")
    def _fake_trtllm_bf16_moe(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        use_shuffled_weight: bool,
        weight_layout: int,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
    ):
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]

        return [hidden_states.new_empty([seq_len, hidden_size], dtype=torch.bfloat16)]

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
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        use_routing_scales_on_input: bool,
        routing_method_type: int = 0,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
    ) -> torch.Tensor:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)
        # Use AutoTuner to select the best tactic
        tuner = AutoTuner.get()
        MoERunner.refine_tuning_config(tune_max_num_tokens)

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]

        # Create workspace buffers
        output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
        )
        topk_ids = torch.empty(
            num_tokens, top_k, dtype=torch.int32, device=hidden_states.device
        )
        expert_weights = torch.empty(
            num_tokens, top_k, dtype=routing_logits.dtype, device=hidden_states.device
        )

        dtype_act = DtypeTrtllmGen.E4m3  # FP8 activation
        dtype_weights = DtypeTrtllmGen.E4m3  # FP8 weights

        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=local_num_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            use_deepseek_fp8=False,  # per_tensor mode
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            weight_layout=WeightLayout.MajorK,
            use_shuffled_weight=True,
        )

        inputs = [output, routing_logits, topk_ids, expert_weights, hidden_states]

        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_fp8_per_tensor_scale_moe",
            [moe_runner],
            MoERunner.tuning_config_no_hidden_states_scales,  # FP8 per-tensor doesn't use hidden_states_scale
            inputs,
            routing_bias=routing_bias,
            gemm1_weights=gemm1_weights,
            output1_scales_scalar=output1_scales_scalar,
            output1_scales_gate_scalar=output1_scales_gate_scalar,
            gemm2_weights=gemm2_weights,
            output2_scales_scalar=output2_scales_scalar,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            use_routing_scales_on_input=use_routing_scales_on_input,
            routing_method_type=routing_method_type,
            enable_pdl=enable_pdl,
        )
        # Call the C++ function
        result = moe_op.trtllm_fp8_per_tensor_scale_moe(
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
            routing_method_type,
            enable_pdl,
            [-1, -1] if tactic == -1 else tactic,
        )

        return result

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
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        use_routing_scales_on_input: bool,
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
        output: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        use_shuffled_weight: bool = False,
        weight_layout: int = 0,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
    ) -> torch.Tensor:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)

        # Use AutoTuner to select the best tactic - follow FP4 pattern exactly
        tuner = AutoTuner.get()
        MoERunner.refine_tuning_config(tune_max_num_tokens)

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]

        # Create workspace buffers
        output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
        )
        topk_ids = torch.empty(
            num_tokens, top_k, dtype=torch.int32, device=hidden_states.device
        )
        expert_weights = torch.empty(
            num_tokens, top_k, dtype=routing_logits.dtype, device=hidden_states.device
        )

        dtype_act = DtypeTrtllmGen.E4m3  # FP8 activation
        dtype_weights = DtypeTrtllmGen.E4m3  # FP8 weights

        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=local_num_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            use_deepseek_fp8=True,  # block_scale mode
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            weight_layout=weight_layout,
            use_shuffled_weight=use_shuffled_weight,
        )

        inputs = [
            output,
            routing_logits,
            topk_ids,
            expert_weights,
            hidden_states,
            hidden_states_scale,
        ]

        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_fp8_block_scale_moe",
            [moe_runner],
            MoERunner.tuning_config_with_hidden_states_scales,  # FP8 block-scale uses hidden_states_scale
            inputs,
            routing_bias=routing_bias,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=routing_method_type,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            enable_pdl=enable_pdl,
        )
        # Call the C++ function for block scale MoE
        result = moe_op.trtllm_fp8_block_scale_moe(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            use_shuffled_weight,
            weight_layout,
            enable_pdl,
            [-1, -1] if tactic == -1 else tactic,
        )

        return result

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
        output: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
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
        num_local_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        enable_pdl: Optional[bool] = None,
        gated_act_type: int = 0,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
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
        else:
            check_shape_dtype_device(
                output, None, torch.bfloat16, hidden_states.device, "output"
            )
            assert output.shape[0] == num_tokens, (
                f"output.shape[0]={output.shape[0]} must be equal to {num_tokens}"
            )
            assert output.shape[1] <= hidden_size, (
                f"output.shape[1]={output.shape[1]} must be less than or equal to {hidden_size}"
            )

        tuner = AutoTuner.get()
        MoERunner.refine_tuning_config(tune_max_num_tokens)
        dtype_act = deduce_trtllm_gen_tensor_dtype(hidden_states, hidden_states_scale)
        dtype_weights = deduce_trtllm_gen_tensor_dtype(
            gemm1_weights, gemm1_weights_scale
        )
        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=num_local_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            use_deepseek_fp8=False,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gated_act_type=gated_act_type,
            weight_layout=WeightLayout.MajorK,
            use_shuffled_weight=True,
        )
        tunning_config = (
            MoERunner.tuning_config_no_hidden_states_scales
            if hidden_states_scale is None
            else MoERunner.tuning_config_with_hidden_states_scales
        )
        inputs = [
            output,
            torch.empty(num_tokens, num_experts, dtype=routing_dtype, device="meta")
            if routing_logits is None
            else routing_logits,
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
            num_experts=num_experts,
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
        intermediate_output = moe_op.trtllm_fp4_block_scale_moe(
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
            num_local_experts,
            routed_scaling_factor,
            routing_method_type,
            do_finalize,
            enable_pdl,
            gated_act_type,
            output,
            [-1, -1] if tactic == -1 else tactic,
        )
        if do_finalize:
            return [output]
        else:
            gemm2_output, expanded_idx_to_permuted_idx = intermediate_output
            return [
                torch.from_dlpack(gemm2_output),
                expert_weights,
                torch.from_dlpack(expanded_idx_to_permuted_idx),
            ]

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
        routing_method_type: int,
        do_finalize: bool,
        enable_pdl: bool,
        gated_act_type: int,
        output: Optional[torch.Tensor],
        tune_max_num_tokens: int,
    ):
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1] if output is None else output.shape[1]

        return [hidden_states.new_empty([seq_len, hidden_size], dtype=torch.bfloat16)]

    @register_custom_op(
        "flashinfer::trtllm_mxint4_block_scale_moe",
        mutates_args=(""),
    )
    def trtllm_mxint4_block_scale_moe_op(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        num_local_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        enable_pdl: Optional[bool] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
    ) -> List[torch.Tensor]:
        routing_dtype = routing_logits.dtype
        hidden_size = hidden_states.shape[-1]
        if hidden_states.dtype == torch.uint8:
            hidden_size = hidden_size * 2
        num_tokens = hidden_states.shape[0]

        # workspace buffers required by trtllm-gen
        topk_ids = torch.empty(
            num_tokens, top_k, dtype=torch.int32, device=hidden_states.device
        )
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
        dtype_act = DtypeTrtllmGen.Bfloat16
        dtype_weights = DtypeTrtllmGen.MxInt4
        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=num_local_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            use_deepseek_fp8=False,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gated_act_type=GatedActType.SwiGlu,
            weight_layout=WeightLayout.BlockMajorK,
            use_shuffled_weight=True,
        )
        tunning_config = MoERunner.tuning_config_no_hidden_states_scales
        inputs = [
            output,
            routing_logits,
            topk_ids,
            expert_weights,
            hidden_states,
        ]

        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_mxint4_block_scale_moe",
            [moe_runner],
            tunning_config,
            inputs,
            num_experts=num_experts,
            routing_bias=routing_bias,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            n_group=n_group,
            topk_group=topk_group,
            local_expert_offset=local_expert_offset,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=routing_method_type,
            enable_pdl=enable_pdl,
        )

        # Call the C++ function for block scale MoE
        moe_op.trtllm_mxint4_block_scale_moe(
            routing_logits,
            routing_bias,
            hidden_states,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            gemm2_weights,
            gemm2_weights_scale,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            num_local_experts,
            routed_scaling_factor,
            routing_method_type,
            enable_pdl,
            output,
            [-1, -1] if tactic == -1 else tactic,
        )
        return output

    @register_fake_op("flashinfer::trtllm_mxint4_block_scale_moe")
    def _fake_trtllm_mxint4_block_scale_moe(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        enable_pdl: bool,
        output: Optional[torch.Tensor],
        tune_max_num_tokens: int,
    ):
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]

        return [hidden_states.new_empty([seq_len, hidden_size], dtype=torch.bfloat16)]

    return SimpleNamespace(
        trtllm_bf16_moe=trtllm_bf16_moe_op,
        trtllm_fp8_per_tensor_scale_moe=trtllm_fp8_per_tensor_scale_moe_op,
        trtllm_fp8_block_scale_moe=trtllm_fp8_block_scale_moe_op,
        trtllm_fp4_block_scale_moe=trtllm_fp4_block_scale_moe_op,
        trtllm_mxint4_block_scale_moe=trtllm_mxint4_block_scale_moe_op,
    )


@flashinfer_api
def trtllm_bf16_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float] = None,
    routing_method_type: int = 0,
    use_shuffled_weight: bool = True,
    weight_layout: int = WeightLayout.BlockMajorK,
    enable_pdl: bool = True,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    """BF16 MoE operation with autotuning support.

    This function implements a bfloat16 Mixture of Experts layer using the TensorRT-LLM backend
    with automatic performance tuning for optimal tile size selection.

    Args:
        routing_logits: [seq_len, num_experts] tensor of routing logits.
            Supports float32 or bfloat16.
        routing_bias: Optional [num_experts] tensor of routing bias.
            Must be bfloat16 if provided.
        hidden_states: [seq_len, hidden_size] tensor of input hidden states.
            Must be bfloat16.
        gemm1_weights: [num_experts, 2*intermediate_size, hidden_size] tensor of first layer weights.
            Must be bfloat16.
        gemm2_weights: [num_experts, hidden_size, intermediate_size] tensor of second layer weights.
            Must be bfloat16.
        num_experts: Total number of experts.
        top_k: Number of experts to route to per token.
        n_group: Number of expert groups.
        topk_group: Number of groups to consider for top-k routing.
        intermediate_size: Size of intermediate layer.
        local_expert_offset: Offset of local experts in global expert space.
        local_num_experts: Number of experts handled by this device.
        routed_scaling_factor (Optional[float]): Scaling factor for routing (can be None for some routing methods)
        routing_method_type: Type of routing method to use (default: 0).
            - 0: Default (Softmax -> TopK)
            - 1: Renormalize (TopK -> Softmax)
            - 2: DeepSeekV3 (Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts)
            - 3: Llama4 (Top1 -> Sigmoid)
            - 4: RenormalizeNaive (Softmax -> TopK -> Renormalize)
        use_shuffled_weight: Whether to use shuffled weight layout for optimization (default: True).
        weight_layout: Weight layout format (default: WeightLayout.BlockMajorK).
            - 0: MajorK - K-major layout [Mn, K]
            - 1: MajorMn - M-major for A and N-major for B [K, Mn]
            - 2: BlockMajorK - Blocked along K dimension [K/blockK, Mn, blockK]
        enable_pdl: Whether to enable Programmatic Dependent Launch. Auto-enabled for >= sm90.
        tune_max_num_tokens: Maximum number of tokens for autotuning (default: 8192).

    Returns:
        torch.Tensor: Output tensor of shape [seq_len, hidden_size].
    """
    return get_trtllm_moe_sm100_module().trtllm_bf16_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        enable_pdl,
        tune_max_num_tokens,
    )


@flashinfer_api
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
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
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
        routing_method_type: Type of routing method to use (default: 0)
        enable_pdl: Whether to enable Programmatic Dependent Launch (PDL). Auto-enabled for >= sm90.
        tune_max_num_tokens(int): Maximum number of tokens for tuning. (default: 8192)

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
        routing_method_type,
        enable_pdl,
        tune_max_num_tokens,
    )


@flashinfer_api
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
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
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
        routing_method_type: Type of routing method to use (default: 0)
        enable_pdl: Whether to enable Programmatic Dependent Launch (PDL). Auto-enabled for >= sm90.
        tune_max_num_tokens(int): Maximum number of tokens for tuning. (default: 8192)
    Returns:
        torch.Tensor: Output tensor of shape [seq_len, hidden_size]
    """
    output = torch.empty(
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
    )
    return get_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        enable_pdl,
        tune_max_num_tokens,
    )


@flashinfer_api
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
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    gated_act_type: int = 0,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
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
        tune_max_num_tokens(int): Maximum number of tokens for tuning. (default: 8192)
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
        routing_method_type,
        do_finalize,
        enable_pdl,
        gated_act_type,
        output,
        tune_max_num_tokens,
    )


@flashinfer_api
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
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    gated_act_type: int = 0,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
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
        tune_max_num_tokens(int): Maximum number of tokens for tuning. (default: 8192)
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
        routing_method_type,
        do_finalize,
        enable_pdl,
        gated_act_type,
        output,
        tune_max_num_tokens,
    )


@flashinfer_api
def trtllm_mxint4_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    enable_pdl: Optional[bool] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
) -> List[torch.Tensor]:
    """MxInt4 block scale MoE operation.

    Args:
        routing_logits (torch.Tensor): shape [seq_len, num_experts]
            Input tensor of routing logits. Supports float32, bfloat16.
        routing_bias: Optional [num_experts] tensor of routing bias.
            Must be bfloat16 if provided.
        hidden_states (torch.Tensor): shape [seq_len, hidden_size]
            Tensor of input hidden states. Supports bfloat16.
        gemm1_weights (torch.Tensor): shape [num_experts, 2 * intermediate_size, hidden_size // 2]
            Tensor of FC1 weights. Dtype must be uint8 (packed mxint4)
        gemm1_weights_scale (torch.Tensor): shape [num_experts, 2 * intermediate_size, hidden_size // 32]
            Scale tensor of FC1 weights. Dtype must be bfloat16.
        gemm1_alpha (Optional[torch.Tensor]): shape [num_experts]
            Tensor of swiglu alpha. Dtype is float32.
        gemm1_beta (Optional[torch.Tensor]): shape [num_experts]
            Tensor of swiglu beta. Dtype is float32.
        gemm1_clamp_limit (Optional[torch.Tensor]): shape [num_experts]
            Tensor of swiglu clamp limit. Dtype is float32.
        gemm2_weights (torch.Tensor): shape [num_experts, hidden_size, intermediate_size]
            Tensor of FC2 weights. Dtype must be uint8 (packed mxint4)
        gemm2_weights_scale (torch.Tensor): shape [num_experts, hidden_size, intermediate_size // 32]
            Scale tensor of FC2 weights. Dtype must be bfloat16.
        num_experts (int): Total number of experts
        top_k (int): Number of experts to route to per token
        n_group (Optional[int]): Number of expert groups (can be None for some routing methods)
        topk_group (Optional[int]): Number of groups to consider for top-k routing (can be None for some routing methods)
        intermediate_size (int): Size of intermediate layer
        local_expert_offset (int): Offset of local experts in global expert space
        local_num_experts (int): Number of experts handled by this device
        routed_scaling_factor (Optional[float]): Scaling factor for routing (can be None for some routing methods)
        routing_method_type (int): Type of routing method to use (default: 0)
            - 0: Default (Softmax -> TopK)
            - 1: Renormalize (TopK -> Softmax)
            - 2: DeepSeekV3 (Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts)
            - 3: Llama4 (Top1 -> Sigmoid)
            - 4: RenormalizeNaive (Softmax -> TopK -> Renormalize)
        enable_pdl (Optional[bool]): Whether to enable Programmatic Dependent Launch (PDL). Auto-enabled for >= sm90.
        tune_max_num_tokens(int): Maximum number of tokens for tuning. (default: 8192)
        output (Optional[torch.Tensor]): shape [seq_len, hidden_size]
            Optional inplace output tensor.
    Returns:
        torch.Tensor: returns the final MoE output.
    """
    return get_trtllm_moe_sm100_module().trtllm_mxint4_block_scale_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
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
        routing_method_type,
        enable_pdl,
        output,
        tune_max_num_tokens,
    )
