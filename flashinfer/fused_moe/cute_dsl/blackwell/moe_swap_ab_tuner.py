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

"""Tactics for Blackwell MoE with weights on A and routed activations on B."""

import itertools
from typing import Any, List, Tuple

import torch


SWAP_AB_ROUTE_TILES = (16, 32, 64, 128, 256)


def get_swap_ab_gemm_valid_tactics(tile_size: int) -> List[Tuple]:
    if tile_size not in SWAP_AB_ROUTE_TILES:
        raise ValueError(f"Unsupported swap-AB routing tile: {tile_size}")
    return [
        (
            (128 * cta_group_size * channel_tiles, tile_size),
            (cta_group_size, 1),
            False,
        )
        for cta_group_size, channel_tiles in ((1, 1), (2, 1), (2, 2))
    ]


ALL_SWAP_AB_MOE_TACTICS = [
    (route_tile, gemm1, gemm2, k_steps, use_clc)
    for route_tile in SWAP_AB_ROUTE_TILES
    for gemm1 in get_swap_ab_gemm_valid_tactics(route_tile)
    for gemm2 in get_swap_ab_gemm_valid_tactics(route_tile)
    for k_steps in itertools.product((4, 8), repeat=2)
    for use_clc in (False, True)
]

DEFAULT_SWAP_AB_MOE_TACTIC = (
    128,
    ((128, 128), (1, 1), False),
    ((128, 128), (1, 1), False),
)


def _split_swap_ab_tactic(tactic: Any) -> Tuple:
    """Accept automatic K depth or explicit FC1/FC2 depths and shared CLC."""
    if not isinstance(tactic, (tuple, list)) or len(tactic) not in (3, 4, 5):
        raise ValueError(
            "swap-AB tactic must be (tile_size, gemm1, gemm2[, k_steps_pair[, use_clc]])"
        )

    tile_size, gemm1, gemm2 = tactic[:3]
    if not isinstance(gemm1, (tuple, list)) or not isinstance(gemm2, (tuple, list)):
        raise ValueError("GEMM sub-tactics must be tuples or lists")
    if len(gemm1) != 3 or len(gemm2) != 3:
        raise ValueError("swap-AB GEMM sub-tactics must use the Blackwell format")
    if len(tactic) == 3:
        k_steps = (None, None)
    else:
        k_steps = tactic[3]
        if not isinstance(k_steps, (tuple, list)) or len(k_steps) != 2:
            raise ValueError("k_steps_pair must contain FC1 and FC2 depths")
        if any(type(depth) is not int or depth not in (4, 8) for depth in k_steps):
            raise ValueError("Blackwell K-step depths must be 4 or 8")
    if len(tactic) == 5:
        use_clc = tactic[4]
        if type(use_clc) is not bool:
            raise ValueError("CLC scheduling choice must be a boolean")
    else:
        use_clc = True
    return tile_size, gemm1, gemm2, k_steps, use_clc


def canonicalize_swap_ab_tactic(tactic: Any) -> Tuple:
    """Canonicalize an MX-activation tactic from Python or JSON."""

    def as_tuple(value):
        if isinstance(value, (tuple, list)):
            return tuple(as_tuple(item) for item in value)
        return value

    result = as_tuple(tactic)
    _split_swap_ab_tactic(result)
    if not any(
        candidate[: len(result)] == result for candidate in ALL_SWAP_AB_MOE_TACTICS
    ):
        raise ValueError(f"unsupported swap-AB MoE tactic: {result!r}")
    return result


def extract_swap_ab_tactic_params(tactic: Any) -> dict:
    tile_size, gemm1, gemm2, k_steps, use_clc = _split_swap_ab_tactic(tactic)
    params = {"tile_size": tile_size, "is_rubin": False}
    for phase, gemm, depth in (
        ("gemm1", gemm1, k_steps[0]),
        ("gemm2", gemm2, k_steps[1]),
    ):
        mma_tiler_mn, cluster_shape_mn, raster_along_m = gemm
        params.update(
            {
                f"{phase}_mma_tiler_mn": mma_tiler_mn,
                f"{phase}_cluster_shape_mn": cluster_shape_mn,
                f"{phase}_raster_along_m": raster_along_m,
                f"{phase}_mma_tiler": None,
                f"{phase}_mma_inst_shape": None,
                f"{phase}_k_steps": depth,
                f"{phase}_use_clc_scheduler": use_clc,
            }
        )
    return params


def get_valid_swap_ab_tactics(runner, inputs: List[torch.Tensor]) -> List[Tuple]:
    import cutlass

    from ....cute_dsl.utils import torch_to_cutlass_dtype
    from ..moe_utils import get_max_num_permuted_tokens
    from .moe_swap_ab_kernel import Sm100BlockScaledGroupedGemmKernel

    x = inputs[0]
    w1_weight, w2_weight = runner._weights_for_tuning(inputs)
    is_mxfp8 = runner.quant_mode in ("w4a8", "mxfp8")
    weight_packing = 1 if runner.quant_mode == "mxfp8" else 2
    activation_dtype = cutlass.Float8E4M3FN if is_mxfp8 else cutlass.Float4E2M1FN
    weight_dtype = (
        cutlass.Float8E4M3FN if runner.quant_mode == "mxfp8" else cutlass.Float4E2M1FN
    )
    sf_dtype = cutlass.Float8E8M0FNU if is_mxfp8 else cutlass.Float8E4M3FN
    sf_vec_size = 32 if is_mxfp8 else 16
    if runner.use_per_token_activation:
        if runner.output_dtype not in (torch.float16, torch.bfloat16):
            return []
        gemm1_c_dtype = torch_to_cutlass_dtype(runner.output_dtype)
    else:
        gemm1_c_dtype = activation_dtype
    gemm2_c_dtype = torch_to_cutlass_dtype(runner.output_dtype)

    def tactic_ok(tactic):
        tile_size, gemm1, gemm2, k_steps, _ = tactic
        route_slots = get_max_num_permuted_tokens(
            x.shape[0], runner.top_k, runner.num_local_experts, tile_size
        )
        for weight, gemm, output_dtype, depth in (
            (w1_weight, gemm1, gemm1_c_dtype, k_steps[0]),
            (w2_weight, gemm2, gemm2_c_dtype, k_steps[1]),
        ):
            mma_tiler_mn, cluster_shape_mn, _ = gemm
            if not Sm100BlockScaledGroupedGemmKernel.can_implement(
                m=weight.shape[1],
                n=route_slots,
                k=weight.shape[2] * weight_packing,
                output_dtype=output_dtype,
                activation_dtype=activation_dtype,
                weight_dtype=weight_dtype,
                sf_dtype=sf_dtype,
                sf_vec_size=sf_vec_size,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                k_steps=depth,
            ):
                return False
        return True

    return [tactic for tactic in ALL_SWAP_AB_MOE_TACTICS if tactic_ok(tactic)]
