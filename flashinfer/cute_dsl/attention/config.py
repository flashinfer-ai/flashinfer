# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""AttentionConfig and AttentionFusion — single source of truth for attention kernel parameters.

AttentionConfig holds all the configuration needed by the kernel: dtypes, tile shapes,
execution mode, and feature flags. Derived properties (cta_tiler, pv_mma_tiler) are
computed from the base parameters.

AttentionFusion bundles an AttentionVariant (the customization point for logits
transform, softmax statistics, and output normalization) into a single object
that the kernel consumes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Tuple, Any

from .fusion.mask import MaskType
from .fusion.variant import AttentionVariant, StandardAttention


class HeadMapping(enum.Enum):
    """How attention heads map to MMA tile dimensions.

    GRID: Heads in grid/loop dimension (prefill, training). Any head count works.
    MMA_M: All heads packed into MMA M-dimension (decode).
    MMA_N: GQA group packed into MMA N-dimension (standard GQA decode).
    """

    GRID = "grid"
    MMA_M = "mma_m"
    MMA_N = "mma_n"


@dataclass
class TileBounds:
    """Handles partial MMA tile filling when logical data < physical tile size.

    Reserved for decode — unused by prefill. Most critical for decode where
    num_heads < 128 (the MMA M-tile width). Compiles away via
    cutlass.const_expr when no masking is needed.
    """

    m_bound: int | None = None
    n_bound: int | None = None

    def needs_m_masking(self, tile_m: int) -> bool:
        return self.m_bound is not None and self.m_bound < tile_m

    def needs_n_masking(self, tile_n: int) -> bool:
        return self.n_bound is not None and self.n_bound < tile_n


@dataclass
class AttentionConfig:
    """Single source of truth for attention kernel configuration.

    Replaces the scattered self.xxx attributes in the kernel's __init__.
    Derived properties (cta_tiler, pv_mma_tiler, etc.) are computed from
    the base parameters.
    """

    # Core parameters
    qk_acc_dtype: Any  # Type[cutlass.Numeric] — using Any to avoid import dependency
    pv_acc_dtype: Any
    mma_tiler: Tuple[int, int, int]
    is_persistent: bool
    mask_type: MaskType
    num_repeat_kv_heads: int = 1
    window_left: int = -1

    # Reserved for decode — unused by prefill. Decode kernels pack heads into
    # MMA M/N dimensions; prefill maps heads via the grid (HeadMapping.GRID).
    head_mapping: HeadMapping = HeadMapping.GRID
    num_heads: int = 0
    num_kv_heads: int = 0

    SUPPORTED_MMA_TILE_MN = (128, 128)
    MMA_K_GRANULARITY = {
        16: 16,
        8: 32,
    }  # {dtype_width_bits: K-tile element granularity}

    def can_implement(self, dtype_width: int = 16) -> None:
        """Validate that this config is implementable on Blackwell SM100.

        Checks hardware-level constraints that are independent of the target GPU's
        SMEM capacity. SMEM overruns are caught at kernel launch time by CUDA.

        :param dtype_width: Bit width of the input element type (16 for fp16/bf16, 8 for fp8).
        :raises ValueError: If validation fails, with a descriptive message.
        """
        mma_mn = self.mma_tiler[:2]
        if mma_mn != self.SUPPORTED_MMA_TILE_MN:
            raise ValueError(
                f"mma_tiler_mn={mma_mn} is not supported. "
                f"Must be {self.SUPPORTED_MMA_TILE_MN} for Blackwell SM100 tcgen05"
            )
        head_dim = self.mma_tiler[2]
        k_gran = self.MMA_K_GRANULARITY.get(dtype_width, 16)
        if head_dim == 0 or head_dim % k_gran != 0:
            raise ValueError(
                f"head_dim={head_dim} must be a positive multiple of {k_gran} "
                f"(MMA K-dimension granularity for {dtype_width}-bit dtype)"
            )
        if self.num_repeat_kv_heads < 1:
            raise ValueError(
                f"num_repeat_kv_heads={self.num_repeat_kv_heads} must be >= 1"
            )

    @property
    def cta_tiler(self) -> Tuple[int, int, int]:
        """CTA tile: 2 Q tiles per CTA in M-dimension."""
        return (
            2 * self.mma_tiler[0],
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

    @property
    def qk_mma_tiler(self) -> Tuple[int, int, int]:
        """MMA tile for Q*K^T computation."""
        return self.mma_tiler

    @property
    def pv_mma_tiler(self) -> Tuple[int, int, int]:
        """MMA tile for P*V computation (transposed)."""
        return (
            self.mma_tiler[0],
            self.mma_tiler[2],
            self.mma_tiler[1],
        )

    @property
    def cluster_shape_mn(self) -> Tuple[int, int]:
        """Cluster shape (always (1,1) for prefill)."""
        return (1, 1)

    @property
    def tile_bounds(self) -> TileBounds:
        """Derive tile bounds from head mapping."""
        if self.head_mapping == HeadMapping.MMA_M and self.num_heads > 0:
            return TileBounds(m_bound=self.num_heads)
        return TileBounds()


@dataclass
class AttentionFusion:
    """Bundles an AttentionVariant with the kernel.

    The variant object defines all customization hooks (logits transform,
    statistics update, output transform) as co-defined methods.  Compile-time
    flags on the variant drive dead-code elimination via ``cutlass.const_expr``.

    See :class:`~flashinfer.cute_dsl.attention.fusion.variant.AttentionVariant`
    for the full API and execution-order documentation.
    """

    variant: AttentionVariant = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.variant is None:
            self.variant = StandardAttention()

    @property
    def has_params(self) -> bool:
        """Whether the variant needs runtime tensor data."""
        return self.variant.extra_params is not None

    @property
    def params_shape(self) -> tuple | None:
        """Shape of the variant's runtime tensor, or None."""
        ep = self.variant.extra_params
        return tuple(ep.shape) if ep is not None else None

    @property
    def params_strides(self) -> tuple | None:
        """Element strides of the variant's runtime tensor, or None.

        Derived from the PyTorch tensor's actual strides so the CuTe layout
        in the kernel matches the source memory layout.  CuTe defaults to
        column-major; PyTorch is row-major — using explicit strides avoids
        a silent layout mismatch.
        """
        ep = self.variant.extra_params
        return tuple(ep.stride()) if ep is not None else None
