# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAConfig — configuration dataclass for Multi-Head Latent Attention decode kernels.

Separate concrete type from AttentionConfig, following the C++ CUTLASS pattern where
each mainloop variant has its own config type. The problem shapes, tile sizes, and
feature flags are fundamentally different between FMHA prefill and MLA decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Type


@dataclass(frozen=True)
class MLAConfig:
    """Configuration for MLA decode kernels.

    Encapsulates all parameters from the monolithic kernel's __init__ and
    _setup_attributes into a single immutable object.
    """

    # Problem shape
    latent_dim: int = 512
    rope_dim: int = 64

    # Data types (Any to avoid import dependency on cutlass at module level)
    acc_dtype: object = None  # Type[cutlass.Numeric], e.g. cutlass.Float32
    lse_dtype: object = None  # Type[cutlass.Numeric], e.g. cutlass.Float32

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

    # Pipeline stage counts
    load_q_stage: int = 1
    load_kv_stage: int = 15
    mma_s_stage: int = 2
    p_mma_stage: int = 2
    p_cor_stage: int = 2
    mma_o_stage: int = 1
    load_pt_stage: int = 4

    # --- Derived properties ---

    @property
    def mma_qk_tiler(self) -> Tuple[int, int, int]:
        return (self.mma_qk_tiler_mn[0], self.mma_qk_tiler_mn[1], self.rope_dim)

    @property
    def mma_qk_rope_tiler(self) -> Tuple[int, int, int]:
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
        return self.rope_dim // self.mma_qk_tiler[2]

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
        """Check if the MLA kernel can be implemented with the given parameters."""
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
