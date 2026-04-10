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
Shared type definitions for AllReduce fusion operations.

This module contains the canonical definitions for:
- AllReduceFusionPattern: Fusion pattern enumeration
- QuantFusionType: Quantization type enumeration (FP8, FP4, etc.)
- QuantizationSFLayout: Scale factor layout enumeration
- FusionPatternTraits: Traits dataclass for pattern capabilities

These types are shared across different backend implementations (TensorRT-LLM, MNNVL).
"""

from dataclasses import dataclass


# ============================================================================
# FUSION PATTERNS
# ============================================================================


class AllReduceFusionPattern:
    """
    Fusion patterns for AllReduce operations.

    Matches the C++ AllReduceFusionPattern enum in trtllm_allreduce_fusion.cuh.
    """

    # Basic all-reduce pattern
    kAllReduce = 0
    # All-reduce followed by residual add and RMS norm
    kARResidualRMSNorm = 1
    # All-reduce followed by residual add, RMS norm and FP8 quantization
    kARResidualRMSNormFP8Quant = 2
    # All-reduce followed by residual add, RMS norm and FP4 quantization
    kARResidualRMSNormFP4Quant = 3
    # All-reduce followed by residual add, RMS norm and FP8 quantization, with norm output
    kARResidualRMSNormOutFP8Quant = 4
    # All-reduce followed by residual add, RMS norm and FP4 quantization, with norm output
    kARResidualRMSNormOutFP4Quant = 5


# ============================================================================
# QUANTIZATION TYPES
# ============================================================================


class QuantFusionType:
    """
    Quantization types for fused AllReduce operations.

    Matches the C++ QuantType enum in trtllm_allreduce_fusion.cuh.
    """

    NONE = 0
    FP8 = 1
    NVFP4 = 2


class QuantizationSFLayout:
    """
    Scale factor layout for quantization.

    Matches the C++ QuantizationSFLayout enum in trtllm_allreduce_fusion.cuh.
    """

    # Block scale factors are stored in swizzled layout for cutlass FP4 kernel.
    # Scale factor blocks are organized in 512-byte blocks in global memory,
    # with each block having 128x4 FP8 values.
    SWIZZLED_128x4 = 0
    SWIZZLED_8x4 = 1
    # Block scale factors are stored in linear layout (row-major).
    LINEAR = 2


# ============================================================================
# FUSION PATTERN TRAITS
# ============================================================================


@dataclass(frozen=True)
class FusionPatternTraits:
    """
    Traits for AllReduceFusionPattern.

    Mirrors the C++ FusionPatternTraits template in trtllm_allreduce_fusion.cuh.
    Provides a single source of truth for pattern capabilities.
    """

    has_allreduce_out: bool
    has_residual: bool
    has_residual_out: bool
    has_rmsnorm: bool
    has_norm_out: bool
    quant_type: int  # QuantFusionType constant

    @property
    def has_quant(self) -> bool:
        """Returns True if this pattern includes quantization."""
        return self.quant_type != QuantFusionType.NONE


# Single source of truth - mirrors the C++ DEFINE_FUSION_PATTERN_TRAITS macros
_PATTERN_TRAITS: dict[int, FusionPatternTraits] = {
    AllReduceFusionPattern.kAllReduce: FusionPatternTraits(
        has_allreduce_out=True,
        has_residual=False,
        has_residual_out=False,
        has_rmsnorm=False,
        has_norm_out=False,
        quant_type=QuantFusionType.NONE,
    ),
    AllReduceFusionPattern.kARResidualRMSNorm: FusionPatternTraits(
        has_allreduce_out=False,
        has_residual=True,
        has_residual_out=True,
        has_rmsnorm=True,
        has_norm_out=True,
        quant_type=QuantFusionType.NONE,
    ),
    AllReduceFusionPattern.kARResidualRMSNormFP8Quant: FusionPatternTraits(
        has_allreduce_out=False,
        has_residual=True,
        has_residual_out=True,
        has_rmsnorm=True,
        has_norm_out=False,
        quant_type=QuantFusionType.FP8,
    ),
    AllReduceFusionPattern.kARResidualRMSNormFP4Quant: FusionPatternTraits(
        has_allreduce_out=False,
        has_residual=True,
        has_residual_out=True,
        has_rmsnorm=True,
        has_norm_out=False,
        quant_type=QuantFusionType.NVFP4,
    ),
    AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant: FusionPatternTraits(
        has_allreduce_out=False,
        has_residual=True,
        has_residual_out=True,
        has_rmsnorm=True,
        has_norm_out=True,
        quant_type=QuantFusionType.FP8,
    ),
    AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant: FusionPatternTraits(
        has_allreduce_out=False,
        has_residual=True,
        has_residual_out=True,
        has_rmsnorm=True,
        has_norm_out=True,
        quant_type=QuantFusionType.NVFP4,
    ),
}


def get_pattern_traits(pattern: int) -> FusionPatternTraits:
    """
    Get traits for an AllReduceFusionPattern.

    Args:
        pattern: AllReduceFusionPattern constant (0-5)

    Returns:
        FusionPatternTraits with all trait flags for the pattern

    Example:
        >>> traits = get_pattern_traits(AllReduceFusionPattern.kARResidualRMSNormFP8Quant)
        >>> traits.has_quant  # True
        >>> traits.has_rmsnorm  # True
        >>> traits.quant_type  # QuantFusionType.FP8
    """
    return _PATTERN_TRAITS[pattern]
