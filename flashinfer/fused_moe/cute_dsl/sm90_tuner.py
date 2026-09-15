"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SM90 (Hopper) CuTe-DSL fused-MoE tuning: the tactic list, the fixed default
and the :class:`TunableRunner` used by
:func:`~.sm90_fused_moe.cute_dsl_fused_moe_bf16`.

Tactic format: ``(tile_size, gemm1_tactic, gemm2_tactic)`` where

- ``tile_size``: 64 or 128 -- ``moe_sort``'s tile and both GEMMs' M tile;
- ``gemm1_tactic``: ``(tile_shape_mn, swizzle_size)`` -- the gather+SwiGLU
  GEMM's CTA tile and its persistent-walk swizzle (M-tile blocks per pass
  over an expert's B; 1 is the plain N-fast walk);
- ``gemm2_tactic``: ``(tile_shape_mn, cluster_shape_mn, raster_along_m)`` --
  the finalize GEMM's CTA tile, its CTA cluster ((1, 2) multicasts the
  intermediate between two N-tile CTAs) and its tile raster order.
"""

import itertools
from typing import Any, Callable, Dict, List, Tuple

import torch

from ...autotuner import (
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from .hopper.contiguous_gather_grouped_gemm_act_fusion import (
    Sm90ContiguousGatherGroupedGemmActFusionKernel,
)
from .hopper.contiguous_grouped_gemm_finalize_fusion import (
    Sm90ContiguousGroupedGemmFinalizeFusionKernel,
)
from .hopper.utils import TORCH_TO_CUTLASS_DTYPE
from .moe_utils import get_max_num_permuted_tokens


TILE_SIZES: Tuple[int, ...] = (64, 128)
GEMM1_TILE_NS: Tuple[int, ...] = (256, 192, 128, 64)
GEMM1_SWIZZLE_SIZES: Tuple[int, ...] = (1, 8, 16)
GEMM2_TILE_NS: Tuple[int, ...] = (256, 128, 64)
GEMM2_CLUSTER_SHAPES: Tuple[Tuple[int, int], ...] = ((1, 1), (1, 2))
GEMM2_RASTER_ALONG_MS: Tuple[bool, ...] = (False, True)

DEFAULT_SM90_MOE_TACTIC = (
    128,
    ((128, 64), 1),
    ((128, 64), (1, 1), False),
)


def _extract_tactic_params(tactic: Any) -> Dict[str, Any]:
    """Extract the kernel parameters from an SM90 MoE tactic tuple."""
    try:
        (
            tile_size,
            (gemm1_tile_shape_mn, gemm1_swizzle_size),
            (gemm2_tile_shape_mn, gemm2_cluster_shape_mn, gemm2_raster_along_m),
        ) = tactic
        return {
            "tile_size": int(tile_size),
            "gemm1_tile_shape_mn": tuple(gemm1_tile_shape_mn),
            "gemm1_swizzle_size": int(gemm1_swizzle_size),
            "gemm2_tile_shape_mn": tuple(gemm2_tile_shape_mn),
            "gemm2_cluster_shape_mn": tuple(gemm2_cluster_shape_mn),
            "gemm2_raster_along_m": bool(gemm2_raster_along_m),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SM90 MoE tactic must be (tile_size, (tile_shape_mn, swizzle_size), "
            f"(tile_shape_mn, cluster_shape_mn, raster_along_m)); got {tactic!r}"
        ) from exc


def is_valid_tactic(
    tactic: Tuple,
    *,
    dtype: torch.dtype,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_local_experts: int,
) -> bool:
    """Whether the tactic is valid for the problem."""
    params = _extract_tactic_params(tactic)
    cutlass_dtype = TORCH_TO_CUTLASS_DTYPE[dtype]
    permuted_m = get_max_num_permuted_tokens(
        num_tokens, top_k, num_local_experts, params["tile_size"]
    )
    gemm1_ok = Sm90ContiguousGatherGroupedGemmActFusionKernel.can_implement(
        cutlass_dtype,
        cutlass_dtype,
        cutlass_dtype,
        params["gemm1_tile_shape_mn"],
        (1, 1),
        permuted_m,
        2 * intermediate_size,
        hidden_size,
        num_local_experts,
        swizzle_size=params["gemm1_swizzle_size"],
    )
    gemm2_ok = Sm90ContiguousGroupedGemmFinalizeFusionKernel.can_implement(
        cutlass_dtype,
        cutlass_dtype,
        cutlass_dtype,
        params["gemm2_tile_shape_mn"],
        params["gemm2_cluster_shape_mn"],
        permuted_m,
        hidden_size,
        intermediate_size,
        num_local_experts,
    )
    return gemm1_ok and gemm2_ok


class CuteDslFusedMoESm90Runner(TunableRunner):
    """TunableRunner for the SM90 CuTe-DSL fused MoE.

    Tactic format: (tile_size, gemm1_tactic, gemm2_tactic), see the module
    docstring; ``None``/``-1`` selects :data:`DEFAULT_SM90_MOE_TACTIC`.
    Args:
        forward_impl: The actual MoE implementation function.

    Input tensor indices follow
    :func:`~.sm90_fused_moe.cute_dsl_fused_moe_bf16`'s signature order
    (the profiling pre-hook replaces index 1 with a seeded uniform top-k
    routing draw, see :meth:`_profile_routing_pre_hook`):
        0: x (num_tokens, hidden) bf16/fp16
        1: token_selected_experts (num_tokens, top_k) int32
        2: token_final_scales (num_tokens, top_k) fp32
        3: w1_weight (E_local, 2I, hidden) — 32-col up/gate interleaved
        4: w2_weight (E_local, hidden, I)
        5: moe_output (num_tokens, hidden)
    """

    def __init__(
        self,
        forward_impl: Callable,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
        use_fused_finalize: bool = True,
        enable_pdl: bool = True,
    ):
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.use_fused_finalize = use_fused_finalize
        self.enable_pdl = enable_pdl

        seeded = lambda device: torch.Generator(device=device).manual_seed(  # noqa: E731
            515
        )
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=(0, 1, 2, 5),
                    dim_idx=(0, 0, 0, 0),
                    gen_tuning_buckets=get_hybrid_num_tokens_buckets,
                    map_to_tuning_buckets=map_to_hybrid_bucket_uncapped,
                ),
            ),
            tensor_initializers=(
                # 0: x — bf16/fp16 activations (seeded).
                (
                    0,
                    lambda shapes, dtype, device: torch.randn(
                        shapes, device=device, generator=seeded(device)
                    ).to(dtype),
                ),
                # 1: token_selected_experts — overwritten by the
                # pre-hook's routing draw; seeded fallback.
                (
                    1,
                    lambda shapes, dtype, device: torch.randint(
                        0,
                        max(self.num_experts, 1),
                        shapes,
                        dtype=torch.int32,
                        device=device,
                        generator=seeded(device),
                    ),
                ),
                # 2: token_final_scales — softmax-normalized fp32.
                (
                    2,
                    lambda shapes, dtype, device: torch.softmax(
                        torch.randn(shapes, device=device, generator=seeded(device)),
                        dim=-1,
                    ).to(torch.float32),
                ),
                # 5: moe_output — kernel-owned buffer.
                (
                    5,
                    lambda shapes, dtype, device: torch.empty(
                        shapes, dtype=dtype, device=device
                    ),
                ),
            ),
            inputs_pre_hook=self._profile_routing_pre_hook,
            use_cold_l2_cache=True,
            value_aware_input_indices=(1, 2),
            profile_arena_input_indices=(0, 1, 2, 5),
            # Graph replay excludes host launch overhead from short-kernel
            # measurements.
            use_cuda_graph=True,
        )

    def _profile_routing_pre_hook(
        self, inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Replace ``token_selected_experts`` with a seeded uniform top-k draw
        over the global experts (distinct experts per token).

        Real routing spreads the rows unevenly over the experts, and that
        imbalance decides the padding of each row tile; a perfectly balanced
        assignment makes the 64-row tile look best at exactly 64 rows per expert
        and picks tactics that lose 15-20% at run time. A uniform draw
        reproduces the imbalance of unskewed routing and gives each local shard
        ``num_tokens * top_k * num_local_experts / num_experts`` rows on
        average, as at run time.
        """
        routing = inputs[1]
        scores = torch.rand(
            routing.shape[0],
            self.num_experts,
            device=routing.device,
            generator=torch.Generator(device=routing.device).manual_seed(515),
        )
        out = list(inputs)
        out[1] = scores.topk(self.top_k, dim=-1).indices.to(routing.dtype)
        return out

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[Any]:
        """Return the default followed by pruned legal GEMM combinations.

        Every legal N tile of both GEMMs is a candidate. Swizzling is
        limited to at least eight routed rows per expert and weight matrices
        below 48 MiB when more than 16 experts are local. GEMM2 multicast
        needs I >= 192; M-major raster is considered only at M=128, I <= 384.

        The list depends only on the problem shapes and the global expert
        count, never on the local shard, so expert-parallel ranks tuning
        under ``set_autotune_process_group`` profile identical sequences.
        """
        x, w2_weight = inputs[0], inputs[4]
        num_tokens, hidden_size = x.shape
        intermediate_size = w2_weight.shape[2]
        dtype = TORCH_TO_CUTLASS_DTYPE[x.dtype]
        routed_rows = num_tokens * self.top_k
        expert_b_bytes = 2 * intermediate_size * hidden_size * x.element_size()
        tactics: List[Any] = [-1]
        for tile_size in TILE_SIZES:
            permuted_m = get_max_num_permuted_tokens(
                num_tokens, self.top_k, self.num_local_experts, tile_size
            )
            swizzles = GEMM1_SWIZZLE_SIZES
            if (
                routed_rows <= tile_size
                or routed_rows < 8 * self.num_experts
                or (expert_b_bytes >= 48 * 1024 * 1024 and self.num_experts > 16)
            ):
                swizzles = (1,)

            gemm1_ns = [
                tile_n
                for tile_n in GEMM1_TILE_NS
                if Sm90ContiguousGatherGroupedGemmActFusionKernel.can_implement(
                    dtype,
                    dtype,
                    dtype,
                    (tile_size, tile_n),
                    (1, 1),
                    permuted_m,
                    2 * intermediate_size,
                    hidden_size,
                    self.num_local_experts,
                )
            ]
            gemm1_tactics = [
                ((tile_size, tile_n), s) for tile_n in gemm1_ns for s in swizzles
            ]

            gemm2_raster_along_ms = (
                GEMM2_RASTER_ALONG_MS
                if tile_size == 128 and intermediate_size <= 384
                else (False,)
            )
            gemm2_tactics: List[Tuple[Any, ...]] = []
            for cluster in GEMM2_CLUSTER_SHAPES:
                if cluster != (1, 1) and intermediate_size < 192:
                    continue
                gemm2_ns = [
                    tile_n
                    for tile_n in GEMM2_TILE_NS
                    if Sm90ContiguousGroupedGemmFinalizeFusionKernel.can_implement(
                        dtype,
                        dtype,
                        dtype,
                        (tile_size, tile_n),
                        cluster,
                        permuted_m,
                        hidden_size,
                        intermediate_size,
                        self.num_local_experts,
                    )
                ]
                gemm2_tactics.extend(
                    ((tile_size, tile_n), cluster, raster)
                    for tile_n in gemm2_ns
                    for raster in gemm2_raster_along_ms
                )

            tactics.extend(
                (tile_size, gemm1, gemm2)
                for gemm1, gemm2 in itertools.product(gemm1_tactics, gemm2_tactics)
                if (tile_size, gemm1, gemm2) != DEFAULT_SM90_MOE_TACTIC
                and is_valid_tactic(
                    (tile_size, gemm1, gemm2),
                    dtype=x.dtype,
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
                )
            )
        return tactics

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> Tuple[Any, ...]:
        return (
            "input_dtype",
            str(inputs[0].dtype),
            "use_fused_finalize",
            self.use_fused_finalize,
            "enable_pdl",
            self.enable_pdl,
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = None,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the MoE forward with ``tactic`` (``None``/``-1``: the default)."""
        if tactic is None or tactic == -1:
            tactic = DEFAULT_SM90_MOE_TACTIC
        params = _extract_tactic_params(tactic)
        (
            x,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w2_weight,
            moe_output,
        ) = inputs
        return self.forward_impl(
            x,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w2_weight,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            moe_output=moe_output,
            use_fused_finalize=self.use_fused_finalize,
            enable_pdl=self.enable_pdl,
            **params,
            **kwargs,
        )

    def __hash__(self):
        return hash(
            (
                "cute_dsl_fused_moe_bf16",
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.use_fused_finalize,
                self.enable_pdl,
            )
        )
