# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""AttentionConfig and AttentionFusion — single source of truth for attention kernel parameters.

AttentionConfig holds all the configuration needed by the kernel: dtypes, tile shapes,
execution mode, and feature flags. Derived properties (cta_tiler, pv_mma_tiler) are
computed from the base parameters.

AttentionFusion bundles the optional customization callbacks (logits transform, output
transform, attention sinks) into a single object.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Tuple, Type, Any

from types import SimpleNamespace

from .fusion.mask import MaskType


class HeadMapping(enum.Enum):
    """How attention heads map to MMA tile dimensions.

    GRID: Heads in grid/loop dimension (prefill, training). Any head count works.
    MMA_M: All heads packed into MMA M-dimension (MLA decode).
    MMA_N: GQA group packed into MMA N-dimension (standard GQA decode).
    """

    GRID = "grid"
    MMA_M = "mma_m"
    MMA_N = "mma_n"


@dataclass
class TileBounds:
    """Handles partial MMA tile filling when logical data < physical tile size.

    Most critical for MLA decode where num_heads < 128 (the MMA M-tile width).
    Compiles away via cutlass.const_expr when no masking is needed.
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

    # Future extensions for decode/MLA
    head_mapping: HeadMapping = HeadMapping.GRID
    num_heads: int = 0
    num_kv_heads: int = 0

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
    """Bundles optional customization callbacks for attention variants.

    Each field resolves at JIT time — None values compile away to zero overhead.
    """

    logits_transform: Callable | None = None
    output_transform: Callable | None = None
    M_D_update: Callable | None = None
    use_attention_sink: bool = False
    custom_params: SimpleNamespace | None = None

    def __post_init__(self):
        if self.use_attention_sink and self.M_D_update is None:
            raise ValueError(
                "M_D_update is required when use_attention_sink is True"
            )
        if self.custom_params is None:
            self.custom_params = SimpleNamespace()
