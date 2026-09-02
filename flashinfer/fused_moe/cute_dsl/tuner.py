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

"""
Auto-tuner for CuteDSL block-scaled MoE kernels.

This module provides TunableRunner implementations for CuteDSL MoE kernels,
enabling automatic performance tuning across different GEMM tactics.

Tactic format follows TRT-LLM's style and is architecture-dependent:
- Blackwell (SM100): tactic = (mma_tiler_mn, cluster_shape_mn, raster_along_m)
  for both GEMM1 (Gather + FC1 activation) and GEMM2 (Finalize)
- Rubin (SM107): tactic = (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)

Reference: TensorRT-LLM/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
"""

import itertools
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from ..utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ._inputs_helper import CuteDslMoEInputsHelper
from .blackwell.moe_w4a16 import launch_w4a16_moe
from .blackwell.moe_w4a16_kernel import Sm100W4A16GroupedGemmKernel
from .moe_utils import (
    normalize_cute_dsl_moe_activation_type,
    validate_cute_dsl_moe_situ_config,
)

logger = logging.getLogger(__name__)


def _seeded_activation(shapes, dtype, device):
    generator = torch.Generator(device=device).manual_seed(515)
    if dtype is torch.float8_e4m3fn:
        return torch.randn(shapes, device=device, generator=generator).to(dtype)
    return torch.randint(
        0, 256, shapes, dtype=dtype, device=device, generator=generator
    )


# =============================================================================
# Blackwell (SM100) Tactics
# =============================================================================


def get_blackwell_gemm1_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid Blackwell tactics for GEMM1 (Gather + SwiGLU Fusion).

    Format: (mma_tiler_mn, cluster_shape_mn, raster_along_m)
    """
    mma_tiler_mn_candidates = [(tile_size, 128), (tile_size, 256)]
    cluster_shape_mn_candidates = [(tile_size // 128, 1)]
    raster_along_m_candidates = [False]

    return [
        (mma_tiler_mn, cluster_shape_mn, raster_along_m)
        for mma_tiler_mn, cluster_shape_mn, raster_along_m in itertools.product(
            mma_tiler_mn_candidates,
            cluster_shape_mn_candidates,
            raster_along_m_candidates,
        )
    ]


def get_blackwell_gemm2_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid Blackwell tactics for GEMM2 (Finalize Fusion).

    The finalize kernel's MMA shape must match tile_size because it consumes
    the upstream gemm1 output layout. At tile_size=128 it uses 1-CTA mma_m=128;
    at tile_size=256 it uses 2-CTA mma_m=256 (use_2cta_instrs=True). Returning a
    1-CTA tactic at tile_size=256 yields a layout mismatch and incorrect output
    (bug #3067, fixed upstream by #3171).

    Format: (mma_tiler_mn, cluster_shape_mn, raster_along_m)
    """
    mma_tiler_mn_candidates = [(tile_size, 128), (tile_size, 256)]
    cluster_shape_mn_candidates = [
        (tile_size // 128, 1),
        (tile_size // 128, 2),
    ]
    raster_along_m_candidates = [False]

    return [
        (mma_tiler_mn, cluster_shape_mn, raster_along_m)
        for mma_tiler_mn, cluster_shape_mn, raster_along_m in itertools.product(
            mma_tiler_mn_candidates,
            cluster_shape_mn_candidates,
            raster_along_m_candidates,
        )
    ]


def get_blackwell_moe_valid_tactics() -> List[Tuple]:
    """Get all valid Blackwell MoE tactic combinations.

    Returns: List of (tile_size, gemm1_tactic, gemm2_tactic)
    """
    tactics = []
    # tile_size=256 (2-CTA) is enabled: the gemm1(2-CTA)/gemm2(1-CTA) layout
    # mismatch that caused incorrect results (#3067) is fixed by parameterizing
    # get_blackwell_gemm2_valid_tactics on tile_size (#3171). Mirrors main's
    # get_moe_valid_tactics over VALID_TILE_SIZES.
    for tile_size in VALID_TILE_SIZES:
        gemm1_tactics = get_blackwell_gemm1_valid_tactics(tile_size)
        gemm2_tactics = get_blackwell_gemm2_valid_tactics(tile_size)
        for gemm1_tactic, gemm2_tactic in itertools.product(
            gemm1_tactics, gemm2_tactics
        ):
            tactics.append((tile_size, gemm1_tactic, gemm2_tactic))
    return tactics


def get_w4a8_moe_valid_tactics() -> List[Tuple]:
    """Get the Blackwell tactics for W4A8 mixed-format MoE.

    The mixed finalize kernel also supports N tiles 64 and 192. Keep its
    tactic space distinct so W4A8 does not change W4A4 tuning results.
    """
    return [
        (
            tile,
            ((tile, gemm1_n), (tile // 128, 1), False),
            ((tile, gemm2_n), (tile // 128, cluster_n), False),
        )
        for tile, gemm1_n, gemm2_n, cluster_n in itertools.product(
            VALID_TILE_SIZES, (128, 256), (64, 128, 192, 256), (1, 2)
        )
    ]


# Canonical list of tile_sizes the autotuner is allowed to pick.  Used by
# ``CuteDslMoEWrapper`` to size its preallocated kernel-output buffers so
# every tactic in the arch-specific tactic lists can reuse the prealloc,
# regardless of which tile_size the autotuner picks at runtime.  Adding a
# new tile_size here automatically widens the prealloc.
VALID_TILE_SIZES: Tuple[int, ...] = (128, 256)


# =============================================================================
# Rubin (SM107) Tactics
# =============================================================================
# Rubin tactics use 3-tuple mma_tiler/mma_inst_shape and support B-reuse.
# Fixed K dimensions for FP4: mma_tiler_k=256, mma_inst_k=128
#
# Format: (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
# where mma_tiler = (M, N, K) and mma_inst_shape = (M', N, K')
def get_rubin_gemm1_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid Rubin tactics for GEMM1 (Gather + SwiGLU Fusion).

    Format: (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
    """
    mma_tiler_k = 256
    mma_inst_k = 128

    # (mma_tiler_m, mma_inst_m) candidates — B-reuse when tiler_m = 2 * inst_m
    #
    # NOTE: while tile_size is restricted to 128 (see get_rubin_moe_valid_tactics),
    # the mma_tiler_m == tile_size and cluster_shape_m * mma_tiler_m <= tile_size
    # constraints below force mma_tiler_m = 128 and cluster_shape_m = 1.  Every
    # 2CTA and B-reuse entry here is therefore currently unreachable, and only
    # the two 1CTA (128, 128) tactics are actually tuned.  The entries are kept
    # because they become reachable again once tile_size=256 is re-enabled.
    mma_m_candidates = [
        (128, 128),  # no B-reuse, 1CTA
        (256, 256),  # no B-reuse, 2CTA   (unreachable at tile_size=128)
        (256, 128),  # B-reuse, 1CTA      (unreachable at tile_size=128)
        (512, 256),  # B-reuse, 2CTA      (unreachable at tile_size=128)
    ]
    mma_n_candidates = [128, 256]
    # (2, 1) is unreachable at tile_size=128 for the same reason.
    cluster_shape_mn_candidates = [(1, 1), (2, 1)]
    raster_along_m_candidates = [False]

    valid_tactics = []
    for (
        mma_tiler_m,
        mma_inst_m,
    ), mma_n, cluster_shape_mn, raster_along_m in itertools.product(
        mma_m_candidates,
        mma_n_candidates,
        cluster_shape_mn_candidates,
        raster_along_m_candidates,
    ):
        # GEMM1 is a gather GEMM: mma_tiler_m must equal tile_size so that
        # each CTA's tile exactly covers one moe_sort tile in the M dimension.
        # Smaller mma_tiler_m causes incorrect gather indexing.  (This also
        # subsumes the mma_tiler_m > tile_size case.)
        if mma_tiler_m != tile_size:
            continue
        if cluster_shape_mn[0] * mma_tiler_m > tile_size:
            continue
        # 2CTA (mma_inst_m=256) requires even cluster_shape_m
        if mma_inst_m == 256 and cluster_shape_mn[0] % 2 != 0:
            continue

        mma_tiler = (mma_tiler_m, mma_n, mma_tiler_k)
        mma_inst_shape = (mma_inst_m, mma_n, mma_inst_k)
        valid_tactics.append(
            (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
        )

    return valid_tactics


def get_rubin_gemm2_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid Rubin tactics for GEMM2 (Finalize Fusion).

    Format: (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
    """
    mma_tiler_k = 256
    mma_inst_k = 128

    # As in get_rubin_gemm1_valid_tactics, the mma_tiler_m == tile_size and
    # cluster_shape_m * mma_tiler_m <= tile_size constraints below make every
    # 2CTA and B-reuse entry unreachable while tile_size is restricted to 128.
    mma_m_candidates = [
        (128, 128),
        (256, 256),  # unreachable at tile_size=128
        (256, 128),  # unreachable at tile_size=128
        (512, 256),  # unreachable at tile_size=128
    ]
    mma_n_candidates = [128, 256]
    # cluster_shape_n is pinned to 1: the Rubin finalize kernel triggers
    # illegal memory accesses with cluster_shape_n>1 at larger token counts
    # (non-deterministic, routing-dependent).  cluster_shape_m varies here,
    # but (2, 1) is unreachable at tile_size=128 per the note above.
    cluster_shape_mn_candidates = [(1, 1), (2, 1)]
    raster_along_m_candidates = [False]

    valid_tactics = []
    for (
        mma_tiler_m,
        mma_inst_m,
    ), mma_n, cluster_shape_mn, raster_along_m in itertools.product(
        mma_m_candidates,
        mma_n_candidates,
        cluster_shape_mn_candidates,
        raster_along_m_candidates,
    ):
        # tile_idx_to_expert_idx has one entry per routing tile (tile_size
        # rows). mma_tiler_m must equal tile_size so that the CTA tile
        # aligns with the routing tile — matching TRT-LLM's enforcement.
        if mma_tiler_m != tile_size:
            continue
        if cluster_shape_mn[0] * mma_tiler_m > tile_size:
            continue
        # 2CTA (mma_inst_m=256) requires even cluster_shape_m
        if mma_inst_m == 256 and cluster_shape_mn[0] % 2 != 0:
            continue

        mma_tiler = (mma_tiler_m, mma_n, mma_tiler_k)
        mma_inst_shape = (mma_inst_m, mma_n, mma_inst_k)
        valid_tactics.append(
            (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
        )

    return valid_tactics


def get_rubin_moe_valid_tactics() -> List[Tuple]:
    """Get all valid Rubin MoE tactic combinations.

    Returns: List of (tile_size, gemm1_tactic, gemm2_tactic)
    """
    tactics = []
    # Only tile_size=128 is enabled. tile_size=256 with B-reuse causes
    # illegal memory accesses for certain GEMM2 tactic configurations and
    # is disabled until the kernel bug is fixed (mirrors the Blackwell
    # restriction in get_blackwell_moe_valid_tactics).
    for tile_size in [128]:
        gemm1_tactics = get_rubin_gemm1_valid_tactics(tile_size)
        gemm2_tactics = get_rubin_gemm2_valid_tactics(tile_size)
        for gemm1_tactic, gemm2_tactic in itertools.product(
            gemm1_tactics, gemm2_tactics
        ):
            tactics.append((tile_size, gemm1_tactic, gemm2_tactic))
    return tactics


# =============================================================================
# Pre-generated tactic sets
# =============================================================================

ALL_BLACKWELL_MOE_TACTICS = get_blackwell_moe_valid_tactics()
ALL_W4A8_MOE_TACTICS = get_w4a8_moe_valid_tactics()
ALL_RUBIN_MOE_TACTICS = get_rubin_moe_valid_tactics()

# Backwards-compatible alias.  Before the tactic space became
# architecture-dependent, the Blackwell list was simply ALL_MOE_TACTICS.
# Use _get_arch_tactics() for new code.
ALL_MOE_TACTICS = ALL_BLACKWELL_MOE_TACTICS


DEFAULT_BLACKWELL_MOE_TACTIC = (
    128,
    ((128, 128), (1, 1), False),
    ((128, 128), (1, 1), False),
)

DEFAULT_RUBIN_MOE_TACTIC = (
    128,
    ((128, 128, 256), (128, 128, 128), (1, 1), False),
    ((128, 128, 256), (128, 128, 128), (1, 1), False),
)


def canonicalize_w4a8_tactic(tactic: Any) -> Tuple:
    """Canonicalize and validate a W4A8 tactic from Python or JSON."""
    if not isinstance(tactic, (tuple, list)) or len(tactic) != 3:
        raise ValueError("tactic must be (tile_size, gemm1_tactic, gemm2_tactic)")

    def as_tuple(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return tuple(as_tuple(item) for item in value)
        return value

    result = as_tuple(tactic)
    if result not in ALL_W4A8_MOE_TACTICS:
        raise ValueError(f"unsupported W4A8 MoE tactic: {result!r}")
    return result


# =============================================================================
# Tactic parameter extraction
# =============================================================================


def _is_rubin_tactic(tactic: Tuple) -> bool:
    """Detect whether a tactic is Rubin format by checking sub-tactic length.

    Blackwell sub-tactic: (mma_tiler_mn, cluster_shape_mn, raster_along_m) — 3 elements
    Rubin sub-tactic: (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m) — 4 elements
    """
    _, gemm1_tactic, _ = tactic
    return len(gemm1_tactic) == 4


def _extract_tactic_params(tactic: Tuple) -> Dict[str, Any]:
    """Extract parameters from a MoE tactic tuple.

    Handles both Blackwell and Rubin formats transparently.

    Returns:
        Dictionary with all tactic parameters. For Rubin tactics, includes
        'gemm1_mma_tiler', 'gemm1_mma_inst_shape', etc. in addition to
        the standard keys.
    """
    tile_size, gemm1_tactic, gemm2_tactic = tactic

    if _is_rubin_tactic(tactic):
        (
            gemm1_mma_tiler,
            gemm1_mma_inst_shape,
            gemm1_cluster_shape_mn,
            gemm1_raster_along_m,
        ) = gemm1_tactic
        (
            gemm2_mma_tiler,
            gemm2_mma_inst_shape,
            gemm2_cluster_shape_mn,
            gemm2_raster_along_m,
        ) = gemm2_tactic
        return {
            "tile_size": tile_size,
            "is_rubin": True,
            "gemm1_mma_tiler_mn": (gemm1_mma_tiler[0], gemm1_mma_tiler[1]),
            "gemm1_cluster_shape_mn": gemm1_cluster_shape_mn,
            "gemm1_raster_along_m": gemm1_raster_along_m,
            "gemm1_mma_tiler": gemm1_mma_tiler,
            "gemm1_mma_inst_shape": gemm1_mma_inst_shape,
            "gemm2_mma_tiler_mn": (gemm2_mma_tiler[0], gemm2_mma_tiler[1]),
            "gemm2_cluster_shape_mn": gemm2_cluster_shape_mn,
            "gemm2_raster_along_m": gemm2_raster_along_m,
            "gemm2_mma_tiler": gemm2_mma_tiler,
            "gemm2_mma_inst_shape": gemm2_mma_inst_shape,
        }
    else:
        gemm1_mma_tiler_mn, gemm1_cluster_shape_mn, gemm1_raster_along_m = gemm1_tactic
        gemm2_mma_tiler_mn, gemm2_cluster_shape_mn, gemm2_raster_along_m = gemm2_tactic
        return {
            "tile_size": tile_size,
            "is_rubin": False,
            "gemm1_mma_tiler_mn": gemm1_mma_tiler_mn,
            "gemm1_cluster_shape_mn": gemm1_cluster_shape_mn,
            "gemm1_raster_along_m": gemm1_raster_along_m,
            "gemm1_mma_tiler": None,
            "gemm1_mma_inst_shape": None,
            "gemm2_mma_tiler_mn": gemm2_mma_tiler_mn,
            "gemm2_cluster_shape_mn": gemm2_cluster_shape_mn,
            "gemm2_raster_along_m": gemm2_raster_along_m,
            "gemm2_mma_tiler": None,
            "gemm2_mma_inst_shape": None,
        }


def _get_arch_tactics() -> List[Tuple]:
    """Return the tactic list appropriate for the current GPU architecture."""
    if not torch.cuda.is_available():
        return ALL_BLACKWELL_MOE_TACTICS
    major, minor = get_compute_capability(torch.device("cuda"))
    if major == 10 and minor == 7:
        return ALL_RUBIN_MOE_TACTICS
    return ALL_BLACKWELL_MOE_TACTICS


def _get_default_tactic() -> Tuple:
    """Return the default tactic for the current GPU architecture."""
    if not torch.cuda.is_available():
        return DEFAULT_BLACKWELL_MOE_TACTIC
    major, minor = get_compute_capability(torch.device("cuda"))
    if major == 10 and minor == 7:
        return DEFAULT_RUBIN_MOE_TACTIC
    return DEFAULT_BLACKWELL_MOE_TACTIC


# =============================================================================
# TunableRunner
# =============================================================================


class CuteDslFusedMoERunner(TunableRunner):
    """TunableRunner for CuteDSL W4A4 and W4A8 MoE kernels.

    This runner enables auto-tuning of the W4A4 and W4A8 MoE pipelines by
    trying different combinations of GEMM tactics.

    Tactic format follows TRT-LLM style:
        (tile_size, gemm1_tactic, gemm2_tactic)
    where:
        - tile_size: 128 or 256
        - gemm1_tactic: (mma_tiler_mn, cluster_shape_mn, raster_along_m)
        - gemm2_tactic: (mma_tiler_mn, cluster_shape_mn, raster_along_m)

    Input tensor indices (for dynamic_tensor_specs):
        0: x (num_tokens, hidden_size//2) - FP4 packed input
        1: x_sf (num_tokens, hidden_size//sf_vec_size) - input scale factors
        2: token_selected_experts (num_tokens, top_k) - expert assignments
        3: token_final_scales (num_tokens, top_k) - routing weights
        4-10: weight tensors (fixed size, don't depend on num_tokens)
        11: moe_output, or per_token_scale when per-token activation is enabled
        12: moe_output when per-token activation is enabled

    Args:
        forward_impl: The actual MoE implementation function.
        num_experts: Total number of experts.
        top_k: Number of experts selected per token.
        num_local_experts: Number of local experts (for expert parallelism).
        local_expert_offset: Starting expert index for this partition.
        use_fused_finalize: Whether to use fused finalize (default: True).
        output_dtype: Output data type (default: torch.bfloat16).
        use_per_token_activation: Whether inputs include per-token row scales
            for GEMM1.
        situ_beta: When set with ActivationType.Swiglu, use the SiTU gate.
        situ_linear_beta: Optional SiTU tanh clamp for the up branch.

    Also supports Rubin (SM107): tactic format is architecture-dependent —
    see _extract_tactic_params.
    """

    def __init__(
        self,
        forward_impl: Callable,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
        use_fused_finalize: bool = True,
        output_dtype: torch.dtype = torch.bfloat16,
        enable_pdl: bool = True,
        activation_type: int = ActivationType.Swiglu.value,
        swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
        swiglu_beta: float = DEFAULT_SWIGLU_BETA,
        swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        use_per_token_activation: bool = False,
        quant_mode: str = "w4a4",
    ):
        activation_type, gated = normalize_cute_dsl_moe_activation_type(activation_type)
        validate_cute_dsl_moe_situ_config(activation_type, situ_beta, situ_linear_beta)
        quant_mode = quant_mode.lower()
        if quant_mode not in ("w4a4", "w4a8"):
            raise ValueError(f"unsupported CuTe-DSL quant_mode {quant_mode!r}")
        if quant_mode == "w4a8" and use_per_token_activation:
            raise ValueError("per-token activation scaling is not supported for W4A8")
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.use_fused_finalize = use_fused_finalize
        self.output_dtype = output_dtype
        self.enable_pdl = enable_pdl
        self.activation_type = activation_type
        self.gated = gated
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.use_per_token_activation = use_per_token_activation
        self.quant_mode = quant_mode

        # Helper that builds a deterministic balanced approx-max-load
        # assignment for token_selected_experts during autotune profiling.
        # See _inputs_helper.py for rationale -- the random tensor_initializer
        # for input #2 produces non-deterministic and unrealistic per-expert
        # load distributions, biasing autotune picks at marginal cells.
        self._inputs_helper = CuteDslMoEInputsHelper(
            num_experts, top_k, num_local_experts, local_expert_offset
        )

        # Instance-level so dummy expert IDs span all local experts
        # (randint(0, num_experts)) for realistic profiling.
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=(0, 1, 2, 3, 11)
                    + ((12,) if use_per_token_activation else ()),
                    dim_idx=(0,) * (6 if use_per_token_activation else 5),
                    # Bare callables: autotuner adapts the bucket set to
                    # the actual input dim (matches the
                    # _FP8_GEMM_SM100_TUNING_CONFIG pattern in
                    # `gemm/gemm_base.py`).
                    gen_tuning_buckets=get_hybrid_num_tokens_buckets,
                    map_to_tuning_buckets=map_to_hybrid_bucket_uncapped,
                ),
            ),
            # Per-input initializer closures, keyed by input index. Indices
            # match input_idx above: 11 is per_token_scale when per-token
            # activation is enabled (else moe_output), 12 is moe_output.
            tensor_initializers=(
                # 0: x — packed FP4 or MXFP8 activation. Seeded
                # for cross-process determinism of autotune picks
                # (matches trt-llm's seed=515 convention).
                (
                    0,
                    _seeded_activation,
                ),
                # 1: x_sf — FP8 scale factors (uint8). Seeded.
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
                # 2: token_selected_experts — output is overwritten
                # by inputs_pre_hook (CuteDslMoEInputsHelper), but
                # seed the initializer too in case the hook is ever
                # disabled.
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
                # 3: token_final_scales — softmax-normalized. Seeded.
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
                *(
                    [
                        # 11: per_token_scale — ones.
                        (
                            11,
                            lambda shapes, dtype, device: torch.ones(
                                shapes, dtype=torch.float32, device=device
                            ),
                        ),
                        # 12: moe_output — empty.
                        (
                            12,
                            lambda shapes, dtype, device: torch.empty(
                                shapes, dtype=dtype, device=device
                            ),
                        ),
                    ]
                    if use_per_token_activation
                    else [
                        # 11: moe_output — empty.
                        (
                            11,
                            lambda shapes, dtype, device: torch.empty(
                                shapes, dtype=dtype, device=device
                            ),
                        )
                    ]
                ),
            ),
            inputs_pre_hook=self._inputs_helper.inputs_pre_hook,
            # Cold-L2 measurement matches TRT-LLM's
            # CuteDslFusedMoERunner.tuning_config; flushing L2
            # between profile iterations yields autotune timings
            # representative of production cold-cache conditions.
            use_cold_l2_cache=True,
        )

    def __hash__(self):
        return hash(
            (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.use_fused_finalize,
                self.output_dtype,
                int(self.activation_type),
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.situ_beta,
                self.situ_linear_beta,
                self.use_per_token_activation,
                self.quant_mode,
            )
        )

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        return (
            self.quant_mode,
            int(self.activation_type),
            self.swiglu_alpha,
            self.swiglu_beta,
            self.swiglu_limit,
            self.situ_beta,
            self.situ_linear_beta,
        )

    def get_valid_tactics(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[Tuple[Any, ...]]:
        """Return valid tactics filtered by can_implement checks.

        Validates each candidate tactic against both GEMM1 and GEMM2 kernel
        can_implement methods using the actual problem dimensions from inputs.
        Supports both Blackwell and Rubin architectures.
        """
        import cutlass
        from .moe_utils import get_max_num_permuted_tokens

        x = inputs[0]
        w1_weight = inputs[4]

        gated = self.gated
        num_tokens = x.shape[0]
        is_mxfp8 = self.quant_mode == "w4a8"
        hidden_size = x.shape[1] * (1 if is_mxfp8 else 2)
        num_local_experts = w1_weight.shape[0]
        # Gated SwiGLU fuses gate+up (2*intermediate rows); non-gated ReLU^2
        # has a single intermediate-row projection.
        gemm1_n = w1_weight.shape[1]
        intermediate_size = gemm1_n // 2 if gated else gemm1_n

        a_dtype = cutlass.Float8E4M3FN if is_mxfp8 else cutlass.Float4E2M1FN
        b_dtype = cutlass.Float4E2M1FN
        sf_dtype = cutlass.Float8E8M0FNU if is_mxfp8 else cutlass.Float8E4M3FN
        sf_vec_size = 32 if is_mxfp8 else 16

        if self.use_per_token_activation:
            if self.output_dtype == torch.float16:
                gemm1_c_dtype = cutlass.Float16
            elif self.output_dtype == torch.bfloat16:
                gemm1_c_dtype = cutlass.BFloat16
            else:
                return []
        else:
            gemm1_c_dtype = cutlass.Float8E4M3FN if is_mxfp8 else cutlass.Float4E2M1FN
        gemm2_out_dtype = cutlass.BFloat16

        all_tactics = ALL_W4A8_MOE_TACTICS if is_mxfp8 else _get_arch_tactics()

        token_final_scales = inputs[3]
        if token_final_scales.dtype == torch.float32:
            final_scale_dtype = cutlass.Float32
        elif token_final_scales.dtype == torch.bfloat16:
            final_scale_dtype = cutlass.BFloat16
        else:
            final_scale_dtype = cutlass.Float16

        def _tactic_ok(tactic):
            tile_size, gemm1_tactic, gemm2_tactic = tactic
            permuted_m = get_max_num_permuted_tokens(
                num_tokens, self.top_k, self.num_local_experts, tile_size
            )

            if _is_rubin_tactic(tactic):
                # The Rubin (SM107) kernels only implement the gated (SwiGLU)
                # activation path; skip Rubin tactics for non-gated activations.
                if not gated:
                    return False

                # The SM107 kernels need cutlass.utils.rubin_helpers, which only
                # exists from CuTe DSL 4.8. Without this probe the import below
                # raises ModuleNotFoundError instead of merely declining the
                # tactic, aborting autotuning rather than falling back.
                #
                # Imported inside the function because ``cute_dsl/utils`` pulls in
                # cutlass at module scope and this module deliberately does not.
                from ...cute_dsl.utils import is_rubin_cute_dsl_available

                if not is_rubin_cute_dsl_available():
                    return False

                from .rubin import (
                    Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel,
                    Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
                )

                gemm1_mma_tiler, gemm1_mma_inst_shape, gemm1_cluster_shape_mn, _ = (
                    gemm1_tactic
                )
                gemm2_mma_tiler, gemm2_mma_inst_shape, gemm2_cluster_shape_mn, _ = (
                    gemm2_tactic
                )

                gemm1_ok = Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel.can_implement(
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    c_dtype=gemm1_c_dtype,
                    mma_inst_shape=gemm1_mma_inst_shape,
                    mma_tiler=gemm1_mma_tiler,
                    cluster_shape_mn=gemm1_cluster_shape_mn,
                    m=permuted_m,
                    n=2 * intermediate_size,
                    k=hidden_size,
                    l=num_local_experts,
                    a_major="k",
                    b_major="k",
                    c_major="n",
                )
                gemm2_ok = Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    c_dtype=gemm2_out_dtype,
                    mma_inst_shape=gemm2_mma_inst_shape,
                    mma_tiler=gemm2_mma_tiler,
                    cluster_shape_mn=gemm2_cluster_shape_mn,
                    m=permuted_m,
                    n=hidden_size,
                    k=intermediate_size,
                    l=num_local_experts,
                    a_major="k",
                    b_major="k",
                    c_major="n",
                )
            else:
                from .blackwell import (
                    BlockScaledContiguousGatherGroupedGemmKernel,
                    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
                )

                gemm1_mma_tiler_mn, gemm1_cluster_shape_mn, _ = gemm1_tactic
                gemm2_mma_tiler_mn, gemm2_cluster_shape_mn, _ = gemm2_tactic

                gemm1_ok = BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    c_dtype=gemm1_c_dtype,
                    mma_tiler_mn=gemm1_mma_tiler_mn,
                    cluster_shape_mn=gemm1_cluster_shape_mn,
                    m=permuted_m,
                    n=gemm1_n,
                    k=hidden_size,
                    l=num_local_experts,
                    a_major="k",
                    b_major="k",
                    c_major="n",
                )
                gemm2_ok = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    out_dtype=gemm2_out_dtype,
                    final_scale_dtype=final_scale_dtype,
                    mma_tiler_mn=gemm2_mma_tiler_mn,
                    cluster_shape_mn=gemm2_cluster_shape_mn,
                    m=permuted_m,
                    n=hidden_size,
                    k=intermediate_size,
                    l=num_local_experts,
                    a_major="k",
                    b_major="k",
                    out_major="n",
                )

            return gemm1_ok and gemm2_ok

        valid_tactics = [t for t in all_tactics if _tactic_ok(t)]

        if not valid_tactics:
            # The default tactic is a member of the arch tactic list, so an empty
            # list means even the default fails can_implement -- do not fall
            # back to it unvalidated (gh #3957). This early refusal is
            # diagnostics/defense-in-depth: the kernel wrappers re-validate
            # can_implement at launch and raise, so an unvalidated tactic
            # cannot reach the device -- but refusing here avoids pointless
            # profiling of a tactic that can only throw, and says why.
            logger.warning(
                "No valid tactics found for problem dims "
                "(tokens=%d, hidden=%d, intermediate=%d, experts=%d, top_k=%d).",
                num_tokens,
                hidden_size,
                intermediate_size,
                num_local_experts,
                self.top_k,
            )

        return valid_tactics

    def forward(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        tactic: Tuple[Any, ...] = None,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute the MoE forward pass with the specified tactic.

        Args:
            inputs: List of input tensors:
                [x, x_sf, token_selected_experts, token_final_scales,
                 w1_weight, w1_weight_sf, w1_alpha, fc2_input_scale,
                 w2_weight, w2_weight_sf, w2_alpha, per_token_scale (optional),
                 moe_output (optional)]
            tactic: Tactic tuple (tile_size, gemm1_tactic, gemm2_tactic) or None for default.
            do_preparation: If True, perform one-time setup (not used).
            **kwargs: Additional keyword arguments passed to forward_impl.

        Returns:
            Output tensor from the MoE computation.
        """
        if tactic is None or tactic == -1:
            tactic = (
                DEFAULT_BLACKWELL_MOE_TACTIC
                if self.quant_mode == "w4a8"
                else _get_default_tactic()
            )
        elif self.quant_mode == "w4a8":
            tactic = canonicalize_w4a8_tactic(tactic)

        params = _extract_tactic_params(tactic)

        (
            x,
            x_sf,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            fc2_input_scale,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            *optional_inputs,
        ) = inputs

        if self.use_per_token_activation:
            if not optional_inputs:
                raise ValueError(
                    "per_token_scale is required when use_per_token_activation=True"
                )
            per_token_scale = optional_inputs[0]
            moe_output = optional_inputs[1] if len(optional_inputs) > 1 else None
        else:
            per_token_scale = None
            moe_output = optional_inputs[0] if optional_inputs else None

        return self.forward_impl(
            x=x,
            x_sf=x_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
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
            gemm1_mma_tiler=params["gemm1_mma_tiler"],
            gemm1_mma_inst_shape=params["gemm1_mma_inst_shape"],
            gemm2_mma_tiler=params["gemm2_mma_tiler"],
            gemm2_mma_inst_shape=params["gemm2_mma_inst_shape"],
            output_dtype=self.output_dtype,
            use_fused_finalize=self.use_fused_finalize,
            moe_output=moe_output,
            per_token_scale=per_token_scale,
            enable_pdl=self.enable_pdl,
            activation_type=int(self.activation_type),
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
            **kwargs,
        )


_W4A16_ROUTE_TILES = (8, 16, 32, 64, 128, 192)
_W4A16_K_TILE = 256
# W4A16 maps output channels to M and routed rows to N. Preserve the current
# M-major static scheduler order; CLC scheduling owns its traversal order.
_W4A16_RASTER_ALONG_M = True
# Grouped expert scheduling requires cluster N=1 when multiple routed rows
# target the same expert.
_W4A16_GEMM_TOPOLOGY_PAIRS = (
    ((128, (1, 1)), (128, (1, 1))),
    ((128, (2, 1)), (128, (2, 1))),
    ((256, (2, 1)), (256, (2, 1))),
    ((128, (2, 1)), (256, (2, 1))),
    ((256, (2, 1)), (128, (2, 1))),
)


def get_w4a16_moe_valid_tactics() -> List[Tuple]:
    """Get valid W4A16 GEMM1/GEMM2 tactic pairs."""
    tactics: List[Tuple] = []
    for route_tile in _W4A16_ROUTE_TILES:
        for gemm1_topology, gemm2_topology in _W4A16_GEMM_TOPOLOGY_PAIRS:
            gemm1_m, gemm1_cluster_shape = gemm1_topology
            gemm2_m, gemm2_cluster_shape = gemm2_topology
            if route_tile < 16 and (gemm1_m == 256 or gemm2_m == 256):
                continue
            # A 128x192x256 tile cannot retain the two load and transform
            # stages required by the warp-specialized pipeline.
            if route_tile == 192 and (gemm1_m == 128 or gemm2_m == 128):
                continue
            gemm1_tactic = (
                (gemm1_m, route_tile, _W4A16_K_TILE),
                gemm1_cluster_shape,
                _W4A16_RASTER_ALONG_M,
            )
            gemm2_tactic = (
                (gemm2_m, route_tile, _W4A16_K_TILE),
                gemm2_cluster_shape,
                _W4A16_RASTER_ALONG_M,
            )
            tactics.append((gemm1_tactic, gemm2_tactic))
    return tactics


W4A16_MOE_TACTICS = tuple(get_w4a16_moe_valid_tactics())


class CuteDslFusedMoEW4A16Runner(TunableRunner):
    """Tunable runner for the BF16-activation, NVFP4-weight MoE pipeline.

    Inputs:
        [x, token_selected_experts, token_final_scales,
         w1_weight, w1_weight_sf, w1_alpha,
         w2_weight, w2_weight_sf, w2_alpha, moe_output]
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
        use_fused_finalize: bool = True,
        output_dtype: torch.dtype = torch.bfloat16,
        enable_pdl: bool = True,
        activation_type: int = ActivationType.Swiglu.value,
        swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
        swiglu_beta: float = DEFAULT_SWIGLU_BETA,
        swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
    ):
        activation_type, _ = normalize_cute_dsl_moe_activation_type(activation_type)
        validate_cute_dsl_moe_situ_config(activation_type, situ_beta, situ_linear_beta)
        if output_dtype != torch.bfloat16:
            raise ValueError("W4A16 only supports BF16 output")
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.use_fused_finalize = use_fused_finalize
        self.output_dtype = output_dtype
        self.enable_pdl = enable_pdl
        self.activation_type = activation_type
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        # Match production EP routing density while retaining seeded load
        # variance around route-tile boundaries.
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=(0, 1, 2, 9),
                    dim_idx=(0, 0, 0, 0),
                    gen_tuning_buckets=get_hybrid_num_tokens_buckets,
                    map_to_tuning_buckets=map_to_hybrid_bucket_uncapped,
                ),
            ),
            tensor_initializers=(
                (
                    0,
                    lambda shapes, dtype, device: torch.randn(
                        shapes,
                        dtype=dtype,
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(515),
                    ),
                ),
                (
                    1,
                    lambda shapes, dtype, device: torch.randint(
                        0,
                        max(num_experts, 1),
                        shapes,
                        dtype=torch.int32,
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(515),
                    ),
                ),
                (
                    2,
                    lambda shapes, dtype, device: torch.softmax(
                        torch.randn(
                            shapes,
                            device=device,
                            generator=torch.Generator(device=device).manual_seed(515),
                        ),
                        dim=-1,
                    ).to(torch.float32),
                ),
                (
                    9,
                    lambda shapes, dtype, device: torch.empty(
                        shapes, dtype=dtype, device=device
                    ),
                ),
            ),
            use_cold_l2_cache=True,
        )

    def __hash__(self):
        return hash(
            (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.use_fused_finalize,
                self.output_dtype,
                self.enable_pdl,
                int(self.activation_type),
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.situ_beta,
                self.situ_linear_beta,
            )
        )

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        return (
            self.num_experts,
            self.top_k,
            self.num_local_experts,
            self.local_expert_offset,
            self.use_fused_finalize,
            self.output_dtype,
            self.enable_pdl,
            int(self.activation_type),
            self.swiglu_alpha,
            self.swiglu_beta,
            self.swiglu_limit,
            self.situ_beta,
            self.situ_linear_beta,
        )

    def get_valid_tactics(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[Tuple[Any, ...]]:
        import cutlass

        from .moe_utils import get_max_num_permuted_tokens

        w1_weight = inputs[3]
        w2_weight = inputs[6]
        num_tokens = inputs[0].shape[0]

        def can_implement(
            weight: torch.Tensor,
            route_slots: int,
            gemm_tactic: Tuple,
        ) -> bool:
            m = weight.shape[1]
            k = weight.shape[2] * 2
            mma_tiler_mnk, cluster_shape_mn, _ = gemm_tactic
            mma_tiler_m = mma_tiler_mnk[0]
            return Sm100W4A16GroupedGemmKernel.can_implement(
                mnkl=(m, route_slots, k, self.num_local_experts),
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.BFloat16,
                c_dtype=cutlass.BFloat16,
                a_major="k",
                b_major="k",
                c_major="m",
                mma_tiler=mma_tiler_mnk,
                cluster_shape_mn=cluster_shape_mn,
                use_2cta_instrs=mma_tiler_m == 256,
            )

        valid_tactics = []
        for tactic in W4A16_MOE_TACTICS:
            gemm1_tactic, gemm2_tactic = tactic
            route_tile = gemm1_tactic[0][1]
            route_slots = get_max_num_permuted_tokens(
                num_tokens,
                self.top_k,
                self.num_local_experts,
                route_tile,
            )
            if can_implement(w1_weight, route_slots, gemm1_tactic) and can_implement(
                w2_weight, route_slots, gemm2_tactic
            ):
                valid_tactics.append(tactic)
        return valid_tactics

    def forward(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        tactic: Optional[Tuple] = None,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        (
            x,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            moe_output,
        ) = inputs
        if x.dtype != torch.bfloat16:
            raise TypeError(f"W4A16 requires x.dtype=torch.bfloat16, got {x.dtype}")
        if token_final_scales.dtype != torch.float32:
            raise TypeError(
                "W4A16 requires token_final_scales.dtype=torch.float32, "
                f"got {token_final_scales.dtype}"
            )
        num_tokens = int(token_selected_experts.size(0))
        hidden_size = int(w2_weight.size(1))
        if tuple(x.shape) != (num_tokens, hidden_size):
            raise ValueError(
                f"x must have shape {(num_tokens, hidden_size)} for W4A16, "
                f"got {tuple(x.shape)}"
            )
        if tuple(moe_output.shape) != (num_tokens, hidden_size):
            raise ValueError(
                f"moe_output must have shape {(num_tokens, hidden_size)}, "
                f"got {tuple(moe_output.shape)}"
            )
        return launch_w4a16_moe(
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            use_fused_finalize=self.use_fused_finalize,
            moe_output=moe_output,
            enable_pdl=self.enable_pdl,
            activation_type=self.activation_type,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
            tactic=None if tactic is None or tactic == -1 else tactic,
        )


# =============================================================================
# Utility Functions
# =============================================================================


def print_all_tactics():
    """Print all valid MoE tactics for debugging."""
    for label, tactics in [
        ("Blackwell", ALL_BLACKWELL_MOE_TACTICS),
        ("Rubin", ALL_RUBIN_MOE_TACTICS),
    ]:
        logger.info("%s MoE tactics: %d", label, len(tactics))
        for i, tactic in enumerate(tactics):
            tile_size, gemm1_tactic, gemm2_tactic = tactic
            logger.info(
                "  Tactic %d: tile_size=%s, gemm1=%s, gemm2=%s",
                i,
                tile_size,
                gemm1_tactic,
                gemm2_tactic,
            )


def __getattr__(name: str):
    if name == "CuteDslFusedMoENvfp4Runner":
        warnings.warn(
            "CuteDslFusedMoENvfp4Runner is deprecated; use "
            "CuteDslFusedMoERunner instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return CuteDslFusedMoERunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
