# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Auto-tuner support for the MXFP8 activation x MXFP4 weight MoE path.

This module intentionally has a runner, tactic space, and cache identity that
are separate from the homogeneous NVFP4 path in :mod:`.tuner`.  Although the
two pipelines share the routing operation, their MMA operand types, scale
layouts, FC1 output format, and useful FC2 tile shapes are different.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, List, Tuple

import torch

from ...autotuner import (
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ...tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from ...utils import get_compute_capability
from ..utils import get_hybrid_num_tokens_buckets, map_to_hybrid_bucket_uncapped
from ._inputs_helper import CuteDslMoEInputsHelper
from .moe_utils import normalize_cute_dsl_moe_activation_type

logger = logging.getLogger(__name__)


def get_mxfp8_mxfp4_moe_valid_tactics() -> List[Tuple[Any, ...]]:
    """Return the 32 production MXFP8 x MXFP4 full-pipeline tactics.

    The mixed finalize kernel additionally supports N tiles 64 and 192.  They
    are kept out of ``ALL_MOE_TACTICS`` so adding this path cannot change an
    NVFP4 autotune result or invalidate an NVFP4 cache entry.
    """

    tactics: List[Tuple[Any, ...]] = []
    for tile_size in (128, 256):
        cluster_m = tile_size // 128
        gemm1_tactics = [
            ((tile_size, tile_n), (cluster_m, 1), False) for tile_n in (128, 256)
        ]
        gemm2_tactics = [
            ((tile_size, tile_n), (cluster_m, cluster_n), False)
            for tile_n, cluster_n in itertools.product((64, 128, 192, 256), (1, 2))
        ]
        tactics.extend(
            (tile_size, gemm1_tactic, gemm2_tactic)
            for gemm1_tactic, gemm2_tactic in itertools.product(
                gemm1_tactics, gemm2_tactics
            )
        )
    return tactics


ALL_MXFP8_MXFP4_MOE_TACTICS = get_mxfp8_mxfp4_moe_valid_tactics()

DEFAULT_MXFP8_MXFP4_MOE_TACTIC = (
    128,
    ((128, 128), (1, 1), False),
    ((128, 128), (1, 1), False),
)

# Current B200 EP8/DP8 winners.  These are only used as the no-autotune
# fallback for the exact target shape; every other shape uses the conservative
# N=128 default above.  Autotuning remains authoritative whenever enabled.
_TARGET_SMALL_TACTIC = (
    128,
    # NCU shows that N=256 halves the FC1 grid and redundant A/SFA LDGSTS
    # work at the exact DP256 target.  Two randomized paired B200 runs
    # measured a 0.59-0.94% full-pipeline GPU improvement over N=128.
    ((128, 256), (1, 1), False),
    ((128, 192), (1, 2), False),
)
_TARGET_MEDIUM_TACTIC = (
    128,
    ((128, 256), (1, 1), False),
    ((128, 192), (1, 1), False),
)
_TARGET_LARGE_TACTIC = (
    256,
    ((256, 256), (2, 1), False),
    ((256, 256), (2, 1), False),
)


def canonicalize_mxfp8_mxfp4_tactic(tactic: Any) -> Tuple[Any, ...]:
    """Recursively turn a JSON/list tactic representation into tuples."""

    if not isinstance(tactic, (tuple, list)) or len(tactic) != 3:
        raise ValueError("tactic must be (tile_size, gemm1_tactic, gemm2_tactic)")

    def as_tuple(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return tuple(as_tuple(item) for item in value)
        return value

    result = as_tuple(tactic)
    if result not in ALL_MXFP8_MXFP4_MOE_TACTICS:
        raise ValueError(f"unsupported MXFP8 x MXFP4 MoE tactic: {result!r}")
    return result


def _extract_tactic_params(tactic: Tuple[Any, ...]) -> Dict[str, Any]:
    tile_size, gemm1_tactic, gemm2_tactic = tactic
    return {
        "tile_size": tile_size,
        "gemm1_mma_tiler_mn": gemm1_tactic[0],
        "gemm1_cluster_shape_mn": gemm1_tactic[1],
        "gemm2_mma_tiler_mn": gemm2_tactic[0],
        "gemm2_cluster_shape_mn": gemm2_tactic[1],
    }


def _seeded_randn(shapes, dtype, device):
    generator = torch.Generator(device=device).manual_seed(515)
    return torch.randn(shapes, device=device, generator=generator).to(dtype)


class CuteDslFusedMoEMxfp8Mxfp4Runner(TunableRunner):
    """Tunable runner for the mixed MXFP8 activation x MXFP4 weight pipeline."""

    def __init__(
        self,
        forward_impl: Callable,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
        output_dtype: torch.dtype = torch.bfloat16,
        enable_pdl: bool = True,
        activation_type: int = ActivationType.Swiglu.value,
        swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
        swiglu_beta: float = DEFAULT_SWIGLU_BETA,
        swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    ) -> None:
        if output_dtype is not torch.bfloat16:
            raise ValueError(
                "MXFP8 x MXFP4 fused MoE only supports torch.bfloat16 output"
            )
        activation_type, gated = normalize_cute_dsl_moe_activation_type(activation_type)
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.output_dtype = output_dtype
        self.enable_pdl = enable_pdl
        self.activation_type = activation_type
        self.gated = gated
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit

        self._inputs_helper = CuteDslMoEInputsHelper(
            num_experts, top_k, num_local_experts, local_expert_offset
        )
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=(0, 1, 2, 3, 10),
                    dim_idx=(0, 0, 0, 0, 0),
                    gen_tuning_buckets=get_hybrid_num_tokens_buckets,
                    map_to_tuning_buckets=map_to_hybrid_bucket_uncapped,
                ),
            ),
            # Per-input initializer closures, keyed by input index. All seeds
            # are fixed so autotune picks are reproducible across processes
            # (matches trt-llm's seed=515 convention).
            tensor_initializers=(
                # 0: x — MXFP8 E4M3 activations.
                (0, _seeded_randn),
                # 1: x_sf — linear E8M0 scale bytes; keep away from 0/255 so
                # the dequantized magnitudes stay in a representable range.
                (
                    1,
                    lambda shapes, dtype, device: torch.randint(
                        1,
                        128,
                        shapes,
                        dtype=torch.uint8,
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(515),
                    ),
                ),
                # 2: token_selected_experts — overwritten by inputs_pre_hook
                # with a balanced distribution; this is only the fallback.
                (
                    2,
                    lambda shapes, dtype, device: torch.randint(
                        0,
                        max(num_experts, 1),
                        shapes,
                        dtype=torch.int32,
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(515),
                    ),
                ),
                # 3: token_final_scales — router weights, normalized per token.
                (
                    3,
                    lambda shapes, dtype, device: torch.softmax(
                        torch.randn(
                            shapes,
                            device=device,
                            generator=torch.Generator(device=device).manual_seed(515),
                        ),
                        dim=-1,
                    ).to(torch.float32),
                ),
                # 10: moe_output — written by the kernel, contents irrelevant.
                (
                    10,
                    lambda shapes, dtype, device: torch.empty(
                        shapes, dtype=dtype, device=device
                    ),
                ),
            ),
            inputs_pre_hook=self._inputs_helper.inputs_pre_hook,
            use_cold_l2_cache=True,
        )

    def __hash__(self) -> int:
        return hash(
            (
                "mxfp8_mxfp4",
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.output_dtype,
                self.enable_pdl,
                int(self.activation_type),
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
            )
        )

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        device = inputs[0].device
        compute_capability = (
            get_compute_capability(device) if device.type == "cuda" else None
        )
        return (
            "mxfp8_mxfp4_v2",
            self.enable_pdl,
            device.type,
            device.index,
            compute_capability,
            int(self.activation_type),
            self.swiglu_alpha,
            self.swiglu_beta,
            self.swiglu_limit,
        )

    def _fallback_tactic(self, inputs: List[torch.Tensor]) -> Tuple[Any, ...]:
        x, w1_weight, w2_weight = inputs[0], inputs[4], inputs[7]
        is_target_shape = (
            self.gated
            and self.num_experts == 256
            and self.top_k == 8
            and self.num_local_experts == 32
            and tuple(x.shape[1:]) == (6144,)
            and tuple(w1_weight.shape) == (32, 4096, 3072)
            and tuple(w2_weight.shape) == (32, 6144, 1024)
        )
        if not is_target_shape:
            return DEFAULT_MXFP8_MXFP4_MOE_TACTIC
        if x.shape[0] <= 2048:
            return _TARGET_SMALL_TACTIC
        if x.shape[0] <= 4096:
            return _TARGET_MEDIUM_TACTIC
        return _TARGET_LARGE_TACTIC

    def get_valid_tactics(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[Tuple[Any, ...]]:
        import cutlass

        from .blackwell import (
            BlockScaledContiguousGatherGroupedGemmKernel,
            Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel as FinalizeKernel,
        )
        from .moe_utils import get_max_num_permuted_tokens

        x = inputs[0]
        w1_weight = inputs[4]
        w2_weight = inputs[7]
        num_tokens = x.shape[0]
        hidden_size = x.shape[1]
        num_local_experts = w1_weight.shape[0]
        gemm1_n = w1_weight.shape[1]
        intermediate_size = gemm1_n // 2 if self.gated else gemm1_n

        valid_tactics: List[Tuple[Any, ...]] = []
        for tactic in ALL_MXFP8_MXFP4_MOE_TACTICS:
            params = _extract_tactic_params(tactic)
            permuted_m = get_max_num_permuted_tokens(
                num_tokens,
                self.top_k,
                self.num_local_experts,
                params["tile_size"],
            )
            gemm1_ok = BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
                a_dtype=cutlass.Float8E4M3FN,
                b_dtype=cutlass.Float4E2M1FN,
                sf_dtype=cutlass.Float8E8M0FNU,
                sf_vec_size=32,
                c_dtype=cutlass.Float8E4M3FN,
                mma_tiler_mn=params["gemm1_mma_tiler_mn"],
                cluster_shape_mn=params["gemm1_cluster_shape_mn"],
                m=permuted_m,
                n=gemm1_n,
                k=hidden_size,
                l=num_local_experts,
                a_major="k",
                b_major="k",
                c_major="n",
            )
            gemm2_ok = (
                w2_weight.shape[0] == num_local_experts
                and w2_weight.shape[1] == hidden_size
                and w2_weight.shape[2] * 2 == intermediate_size
                and FinalizeKernel.can_implement(
                    a_dtype=cutlass.Float8E4M3FN,
                    b_dtype=cutlass.Float4E2M1FN,
                    sf_dtype=cutlass.Float8E8M0FNU,
                    sf_vec_size=32,
                    out_dtype=cutlass.BFloat16,
                    final_scale_dtype=cutlass.Float32,
                    mma_tiler_mn=params["gemm2_mma_tiler_mn"],
                    cluster_shape_mn=params["gemm2_cluster_shape_mn"],
                    m=permuted_m,
                    n=hidden_size,
                    k=intermediate_size,
                    l=num_local_experts,
                    a_major="k",
                    b_major="k",
                    out_major="n",
                )
            )
            if gemm1_ok and gemm2_ok:
                valid_tactics.append(tactic)

        if not valid_tactics:
            logger.warning(
                "No valid MXFP8 x MXFP4 tactics for tokens=%d, hidden=%d, "
                "intermediate=%d, experts=%d; falling back to the default tactic",
                num_tokens,
                hidden_size,
                intermediate_size,
                num_local_experts,
            )
            return [DEFAULT_MXFP8_MXFP4_MOE_TACTIC]
        return valid_tactics

    def forward(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        tactic: Tuple[Any, ...] = None,  # type: ignore[assignment]
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if tactic is None or tactic == -1:
            tactic = self._fallback_tactic(inputs)
        else:
            tactic = canonicalize_mxfp8_mxfp4_tactic(tactic)
        params = _extract_tactic_params(tactic)

        (
            x,
            x_sf,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            *optional_inputs,
        ) = inputs
        moe_output = optional_inputs[0] if optional_inputs else None
        return self.forward_impl(
            x=x,
            x_sf=x_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            tile_size=params["tile_size"],
            gemm1_mma_tiler_mn=params["gemm1_mma_tiler_mn"],
            gemm1_cluster_shape_mn=params["gemm1_cluster_shape_mn"],
            gemm2_mma_tiler_mn=params["gemm2_mma_tiler_mn"],
            gemm2_cluster_shape_mn=params["gemm2_cluster_shape_mn"],
            output_dtype=self.output_dtype,
            moe_output=moe_output,
            enable_pdl=self.enable_pdl,
            activation_type=int(self.activation_type),
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            **kwargs,
        )


__all__ = [
    "ALL_MXFP8_MXFP4_MOE_TACTICS",
    "DEFAULT_MXFP8_MXFP4_MOE_TACTIC",
    "CuteDslFusedMoEMxfp8Mxfp4Runner",
    "canonicalize_mxfp8_mxfp4_tactic",
    "get_mxfp8_mxfp4_moe_valid_tactics",
]
