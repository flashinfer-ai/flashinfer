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

SM90 (Hopper) CuTe-DSL fused-MoE tuning: the tactic schema, the shape
heuristics shared with the untuned dispatch, candidate enumeration, and the
:class:`TunableRunner` used by :func:`~.sm90_fused_moe.cute_dsl_fused_moe_bf16`.
"""

from typing import Any, Callable, List, NamedTuple, Optional, Tuple

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
from ._inputs_helper import CuteDslMoEInputsHelper

# Candidate tables shared by the heuristic auto-selection and the tuner's
# tactic space.
_TILE_SIZES = (64, 128)
_GEMM1_TILE_N_BY_TILE_SIZE = {64: (128, 64), 128: (256, 192, 128, 64)}
_GEMM2_TILE_N_BY_TILE_SIZE = {64: (128, 64), 128: (256, 128, 64)}
_GEMM2_CLUSTER_SHAPES = ((1, 1), (1, 2))


def _default_gemm2_tile_k(inter_per_rank: int, tile_size: int) -> int:
    """Reduction K-tile for GEMM2.

    Use 32 when 64 cannot divide the per-rank reduction dimension. When both
    divide, 32 allows a deeper pipeline for large tile-M reductions; 64 avoids
    the extra pipeline steps for shorter reductions.
    """
    if inter_per_rank % 64 != 0:
        return 32
    if tile_size >= 128 and inter_per_rank >= 384:
        return 32
    return 64


def _gemm2_tactic_can_implement(
    hidden_size: int,
    intermediate_size: int,
    tile_shape_mn: Tuple[int, int],
    tile_k: int,
    cluster_shape_mn: Tuple[int, int],
) -> bool:
    """Whether a GEMM2 topology is legal without host-side fallback."""
    tile_m, tile_n = tile_shape_mn
    return (
        tile_m in _TILE_SIZES
        and tile_n % 8 == 0
        and 8 <= tile_n <= 256
        and hidden_size % tile_n == 0
        and tile_k in (32, 64)
        and intermediate_size % tile_k == 0
        and cluster_shape_mn in _GEMM2_CLUSTER_SHAPES
        and (hidden_size // tile_n) % cluster_shape_mn[1] == 0
    )


class Sm90MoeTactic(NamedTuple):
    """One SM90 CuTe-DSL MoE tactic; the AutoTuner passes ``-1`` for the
    default (the heuristic auto-selection). ``gemm2_tile_k`` is pinned to
    the :func:`_default_gemm2_tile_k` heuristic in the tuned tactic space;
    explicitly constructed tactics may force either legal value. GEMM2
    cluster multicast and raster order are independent tuned axes."""

    tile_size: int
    gemm1_tile_n: int
    gemm2_tile_n: int
    gemm2_tile_k: int
    gemm2_cluster_shape_mn: Tuple[int, int]
    gemm2_raster_along_m: bool


class _Sm90MoeTacticOverride(NamedTuple):
    tile_size: Optional[int]
    gemm1_tile_n: Optional[int]
    gemm2_tile_n: Optional[int]
    gemm2_tile_k: Optional[int]
    gemm2_cluster_shape_mn: Optional[Tuple[int, int]]
    gemm2_raster_along_m: Optional[bool]


def _decode_sm90_moe_tactic(tactic: Any) -> _Sm90MoeTacticOverride:
    """Validate a tactic and normalize it to per-field overrides."""
    if tactic is None or tactic == -1:
        return _Sm90MoeTacticOverride(None, None, None, None, None, None)
    if not isinstance(tactic, (tuple, list)):
        raise TypeError("SM90 MoE tactic must be -1, None, a tuple, or a list")
    if len(tactic) != 6:
        raise ValueError(f"SM90 MoE tactic has {len(tactic)} fields; expected 6")
    current = Sm90MoeTactic(*tactic)
    cluster_shape_mn = (
        int(current.gemm2_cluster_shape_mn[0]),
        int(current.gemm2_cluster_shape_mn[1]),
    )
    if cluster_shape_mn not in _GEMM2_CLUSTER_SHAPES:
        raise ValueError(
            f"unsupported GEMM2 cluster shape {current.gemm2_cluster_shape_mn}"
        )
    if not isinstance(current.gemm2_raster_along_m, bool):
        raise ValueError("GEMM2 raster tactic must be bool")
    return _Sm90MoeTacticOverride(
        current.tile_size,
        current.gemm1_tile_n,
        current.gemm2_tile_n,
        current.gemm2_tile_k,
        cluster_shape_mn,
        current.gemm2_raster_along_m,
    )


def _enumerate_sm90_moe_tactics(
    gemm1_n_size: int, hidden_size: int, intermediate_size: int
) -> List[Sm90MoeTactic]:
    """Enumerate the legal tuned cross-product for one static geometry.

    The sweep is capped at the top-2 legal N tiles per GEMM to bound
    compile-cached kernel specializations.
    """
    tactics: List[Sm90MoeTactic] = []
    for tile_size in _TILE_SIZES:
        g2_tile_k = _default_gemm2_tile_k(intermediate_size, tile_size)
        g1_top2 = [
            n for n in _GEMM1_TILE_N_BY_TILE_SIZE[tile_size] if gemm1_n_size % n == 0
        ][:2]
        g2_top2 = [
            n for n in _GEMM2_TILE_N_BY_TILE_SIZE[tile_size] if hidden_size % n == 0
        ][:2]
        for g1_n in g1_top2:
            for g2_n in g2_top2:
                for g2_cluster in _GEMM2_CLUSTER_SHAPES:
                    if not _gemm2_tactic_can_implement(
                        hidden_size,
                        intermediate_size,
                        (tile_size, g2_n),
                        g2_tile_k,
                        g2_cluster,
                    ):
                        continue
                    for g2_raster in (False, True):
                        tactics.append(
                            Sm90MoeTactic(
                                tile_size,
                                g1_n,
                                g2_n,
                                g2_tile_k,
                                g2_cluster,
                                g2_raster,
                            )
                        )
    return tactics


class CuteDslFusedMoESm90Runner(TunableRunner):
    """TunableRunner for the SM90 CuTe-DSL fused MoE.

    Tactic format: :class:`Sm90MoeTactic` ``(tile_size, gemm1_tile_n,
    gemm2_tile_n, gemm2_tile_k, gemm2_cluster_shape_mn,
    gemm2_raster_along_m)``; ``None``/``-1`` selects the fallback heuristic.

    Args:
        forward_impl: The actual MoE implementation function.

    Input tensor indices follow
    :func:`~.sm90_fused_moe.cute_dsl_fused_moe_bf16`'s signature order
    (:class:`CuteDslMoEInputsHelper`'s pre-hook replaces index 1 with
    a balanced expert assignment during autotune profiling):
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

        # Balanced approx-max-load expert assignment for profiling inputs
        # (a random assignment biases autotune picks at marginal cells).
        self._inputs_helper = CuteDslMoEInputsHelper(
            num_experts, top_k, num_local_experts, local_expert_offset, tse_idx=1
        )
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
                # pre-hook's balanced assignment; seeded fallback.
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
            inputs_pre_hook=self._inputs_helper.inputs_pre_hook,
            use_cold_l2_cache=True,
            value_aware_input_indices=(1, 2),
            profile_arena_input_indices=(0, 1, 2, 5),
            # Graph replay excludes host launch overhead from short-kernel
            # measurements.
            use_cuda_graph=True,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[Any]:
        gemm1_n = inputs[3].shape[1]
        hidden = inputs[4].shape[1]
        inter = inputs[4].shape[2]
        # The heuristic auto-selection competes as an explicit candidate:
        # a tuned winner then never ranks below the default dispatch in the
        # same measurement session (no-worse-than-heuristic by construction).
        return [-1, *_enumerate_sm90_moe_tactics(gemm1_n, hidden, inter)]

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
        (
            x,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w2_weight,
            moe_output,
        ) = inputs
        tactic_override = _decode_sm90_moe_tactic(tactic)
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
            tile_size=tactic_override.tile_size,
            gemm1_tile_n=tactic_override.gemm1_tile_n,
            gemm2_tile_n=tactic_override.gemm2_tile_n,
            gemm2_tile_k=tactic_override.gemm2_tile_k,
            gemm2_cluster_shape_mn=tactic_override.gemm2_cluster_shape_mn,
            gemm2_raster_along_m=tactic_override.gemm2_raster_along_m,
            use_fused_finalize=self.use_fused_finalize,
            enable_pdl=self.enable_pdl,
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
