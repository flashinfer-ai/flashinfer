# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAConfig — configuration for Multi-Latent Attention decode kernels.

Sibling of AttentionConfig (not extending it), following the C++ CUTLASS pattern
of separate concrete types per kernel variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Type

import cutlass


@dataclass(frozen=True)
class MLAConfig:
    """Core configuration for MLA decode kernels.

    Groups constructor parameters that define the problem shape, data types,
    tile shapes, and execution mode. Derived iteration counts are computed
    as properties.
    """

    latent_dim: int
    rope_dim: int
    num_heads: int

    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32
    lse_dtype: Type[cutlass.Numeric] = cutlass.Float32

    mma_qk_tiler_mn: Tuple[int, int] = (128, 128)
    mma_pv_tiler_mn: Tuple[int, int] = (128, 256)

    max_active_clusters: int = 1
    is_persistent: bool = True
    is_cpasync: bool = False
    use_page_table: bool = True
    is_var_seq: bool = False
    is_var_split_kv: bool = False
    use_2cta_instrs: bool = True
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)

    warps_in_n: int = 2

    @property
    def mma_qk_tiler(self) -> Tuple[int, int, int]:
        return (self.mma_qk_tiler_mn[0], self.mma_qk_tiler_mn[1], self.rope_dim)

    @property
    def mma_pv_tiler(self) -> Tuple[int, int, int]:
        return (self.mma_pv_tiler_mn[0], self.mma_pv_tiler_mn[1], 32)

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


def mla_can_implement(
    B: int,
    K: int,
    H: int,
    L: int,
    R: int,
    in_dtype: Type[cutlass.Numeric],
    out_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    lse_dtype: Type[cutlass.Numeric],
    mma_qk_tiler_mn: Tuple[int, int],
    mma_pv_tiler_mn: Tuple[int, int],
    split_kv: int,
    is_persistent: bool,
    is_cpasync: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    use_page_table: bool,
    page_size: int,
) -> bool:
    """Check if the MLA kernel can be implemented with the given parameters.

    :param B: Batch size
    :param K: Sequence length
    :param H: Number of heads
    :param L: Latent dimension (must be 512)
    :param R: RoPE dimension (must be 64)
    :param in_dtype: Input data type
    :param out_dtype: Output data type
    :param acc_dtype: Accumulator data type
    :param lse_dtype: Log-sum-exp data type
    :param mma_qk_tiler_mn: QK MMA tile shape (M, N)
    :param mma_pv_tiler_mn: PV MMA tile shape (M, N)
    :param split_kv: Split-KV factor
    :param is_persistent: Whether to use persistent kernel
    :param is_cpasync: Whether to use cpasync
    :param is_var_seq: Whether to use variable sequence length
    :param is_var_split_kv: Whether to use variable split_kv
    :param use_page_table: Whether to use page table
    :param page_size: Page size for paged KV cache
    :return: True if the configuration is supported
    """
    if L != 512 or R != 64:
        return False
    if in_dtype not in [cutlass.Float8E4M3FN, cutlass.Float16, cutlass.BFloat16]:
        return False
    if out_dtype not in [cutlass.Float16, cutlass.BFloat16]:
        return False
    if acc_dtype != cutlass.Float32 or lse_dtype != cutlass.Float32:
        return False
    if is_cpasync:
        if not use_page_table:
            return False
        if page_size & (page_size - 1) != 0:
            return False
        if page_size > mma_qk_tiler_mn[1]:
            return False
    else:
        if use_page_table and page_size != mma_qk_tiler_mn[1]:
            return False
    if mma_qk_tiler_mn[0] != 128 or mma_pv_tiler_mn[0] != 128:
        return False
    if mma_pv_tiler_mn[1] * 32 != mma_qk_tiler_mn[1] * R:
        return False
    if is_var_split_kv and (not use_page_table or not is_var_seq):
        return False
    if is_var_seq and not use_page_table:
        return False
    if K <= 0:
        return False
    return True
