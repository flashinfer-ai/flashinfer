# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAConfig — configuration dataclass for Multi-Head Latent Attention decode kernels.

Separate concrete type from AttentionConfig, following the C++ CUTLASS pattern where
each mainloop variant has its own config type. The problem shapes, tile sizes, and
feature flags are fundamentally different between FMHA prefill and MLA decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Type

if TYPE_CHECKING:
    import cutlass


@dataclass(frozen=True)
class MLAConfig:
    """Configuration for MLA decode kernels.

    Encapsulates all parameters from the monolithic kernel's __init__ and
    _setup_attributes into a single immutable object.  Supports both FP16/BF16
    and FP8 variants via the ``is_fp8`` flag, which adjusts MMA tiler
    dimensions, pipeline stage counts, and register budgets.
    """

    # Problem shape
    latent_dim: int = 512
    rope_dim: int = 64

    # Data types
    acc_dtype: Type[cutlass.Numeric] = None  # type: ignore[assignment]
    lse_dtype: Type[cutlass.Numeric] = None  # type: ignore[assignment]

    # MMA tile shapes
    mma_qk_tiler_mn: Tuple[int, int] = (128, 128)
    mma_pv_tiler_mn: Tuple[int, int] = (128, 256)

    # Execution parameters
    max_active_clusters: int = 1
    page_size: int = 1
    skip_correction_threshold: float = 0.0
    is_persistent: bool = False
    is_var_seq: bool = False
    is_var_split_kv: bool = False
    enable_pdl: bool = False

    # Cluster configuration
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)
    use_2cta_instrs: bool = True
    warps_in_n: int = 2

    # FP8 flag — selects FP8-specific tiler, stages, and register budgets
    is_fp8: bool = False

    # Pipeline stage counts — shared between FP16 and FP8
    load_q_stage: int = 1
    mma_s_stage: int = 2
    p_mma_stage: int = 2
    p_cor_stage: int = 2
    mma_o_stage: int = 1  # FP16: 1, FP8: 2

    # Pipeline stage counts — FP16 only (unified K+V + page table pipelines)
    load_kv_stage: int = 15
    load_pt_stage: int = 4

    # Pipeline stage counts — FP8 only (separate K and V pipelines, no page table)
    load_k_stage: int = 3
    load_v_stage: int = 2

    # --- Derived properties ---

    @property
    def mma_qk_tiler(self) -> Tuple[int, int, int]:
        # FP8 doubles K-dim to pack latent into wider tiles
        k = self.rope_dim * 2 if self.is_fp8 else self.rope_dim
        return (self.mma_qk_tiler_mn[0], self.mma_qk_tiler_mn[1], k)

    @property
    def mma_qk_rope_tiler(self) -> Tuple[int, int, int]:
        # Rope tiler always uses rope_dim as K-dim (not doubled for FP8)
        return (self.mma_qk_tiler_mn[0], self.mma_qk_tiler_mn[1], self.rope_dim)

    @property
    def mma_pv_tiler(self) -> Tuple[int, int, int]:
        return (
            self.mma_pv_tiler_mn[0],
            self.mma_pv_tiler_mn[1],
            self.mma_qk_tiler[1] * self.mma_qk_tiler[2] // self.mma_pv_tiler_mn[1],
        )

    @property
    def iterations_qk_latent(self) -> int:
        return self.latent_dim // self.mma_qk_tiler[2]

    @property
    def iterations_qk_rope(self) -> int:
        # FP8: rope fits in one iteration due to doubled K-dim
        return 1 if self.is_fp8 else self.rope_dim // self.mma_qk_tiler[2]

    @property
    def iterations_qk(self) -> int:
        return self.iterations_qk_latent + self.iterations_qk_rope

    @property
    def iterations_pv_k(self) -> int:
        return self.mma_qk_tiler[1] // self.mma_pv_tiler[2]

    @property
    def iterations_pv_n(self) -> int:
        return self.latent_dim // self.mma_pv_tiler[1]

    @property
    def tmem_o_offset(self) -> int:
        return self.mma_s_stage * self.mma_qk_tiler[1] // self.warps_in_n

    @property
    def correction_factor_offset(self) -> int:
        return self.tmem_o_offset + self.latent_dim // self.warps_in_n

    @property
    def num_compute_warps(self) -> int:
        return 4

    @property
    def per_iteration_mma_o(self) -> bool:
        """FP8 uses mma_o_stage=2 with per-iteration pipeline wait/release."""
        return self.mma_o_stage > 1

    @property
    def correction_reg_num(self) -> int:
        return 256 if self.is_fp8 else 208

    @property
    def other_reg_num(self) -> int:
        return 48 if self.is_fp8 else 96

    @staticmethod
    def can_implement(
        B: int,
        S: int,
        K: int,
        H: int,
        L: int,
        R: int,
        in_dtype,
        out_dtype,
        acc_dtype,
        lse_dtype,
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        split_kv: int,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        page_size: int,
    ) -> bool:
        """Check if the FP16/BF16 MLA kernel can be implemented."""
        import cutlass as _cutlass

        if L != 512 or R != 64:
            return False
        if in_dtype not in [_cutlass.Float16, _cutlass.BFloat16]:
            return False
        if out_dtype not in [_cutlass.Float16, _cutlass.BFloat16]:
            return False
        if acc_dtype != _cutlass.Float32 or lse_dtype != _cutlass.Float32:
            return False
        if mma_qk_tiler_mn[1] % page_size != 0 or page_size == 1:
            return False
        if mma_qk_tiler_mn[0] != mma_pv_tiler_mn[0] or mma_qk_tiler_mn[0] != 128:
            return False
        if is_var_split_kv and not is_var_seq:
            return False
        if H > 128 or (H < 128 and split_kv != 1):
            return False
        if S < 1 or S > 4:
            return False
        if K <= 0:
            return False
        return True

    @staticmethod
    def can_implement_fp8(
        B: int,
        S: int,
        K: int,
        H: int,
        L: int,
        R: int,
        in_dtype,
        out_dtype,
        acc_dtype,
        lse_dtype,
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        split_kv: int,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        page_size: int,
    ) -> bool:
        """Check if the FP8 MLA kernel can be implemented."""
        import cutlass as _cutlass

        if L != 512 or R != 64:
            return False
        if in_dtype not in [_cutlass.Float8E4M3FN]:
            return False
        if out_dtype not in [_cutlass.Float8E4M3FN, _cutlass.BFloat16]:
            return False
        if acc_dtype != _cutlass.Float32 or lse_dtype != _cutlass.Float32:
            return False
        if mma_qk_tiler_mn[1] % page_size != 0 or page_size == 1:
            return False
        if mma_qk_tiler_mn[0] != mma_pv_tiler_mn[0] or mma_qk_tiler_mn[0] != 128:
            return False
        if is_var_split_kv and not is_var_seq:
            return False
        if H > 128 or (H < 128 and split_kv != 1):
            return False
        if S <= 0 or S > 4:
            return False
        if K <= 0:
            return False
        return True
