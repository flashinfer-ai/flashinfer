# Copyright (c) 2025 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packed FP6 tile geometry."""

from typing import Tuple
import cutlass


# MX-FP6 (W6A6/W6A8) uses sf_vec_size=32 and m16n8k32 MMA (see b12x._lib.fp6).
MXFP6_SF_VEC_SIZE = 32

MXFP6_MMA_K = 32

MXFP6_SF_DTYPE_STR = "float8_e8m0fnu"

MXFP6_AB_DTYPE_STRINGS = frozenset({"float6_e3m2fn", "float6_e2m3fn"})


def is_mxfp6_ab_dtype(ab_dtype) -> bool:
    """True when ``ab_dtype`` is an MX-FP6 operand element type."""
    return ab_dtype in (cutlass.Float6E3M2FN, cutlass.Float6E2M3FN)


def is_mxfp6_ab_dtype_string(dtype: str) -> bool:
    return dtype in MXFP6_AB_DTYPE_STRINGS


def mxfp6_tile_k(sf_vec_size: int = MXFP6_SF_VEC_SIZE) -> int:
    """K tile size for one MX-FP6 pipeline stage (four ``m16n8k32`` MMA slices)."""
    if sf_vec_size != MXFP6_SF_VEC_SIZE:
        raise ValueError(
            f"MX-FP6 expects sf_vec_size={MXFP6_SF_VEC_SIZE}, got {sf_vec_size}"
        )
    return sf_vec_size * 4


def mxfp6_num_k_blocks(tile_k: int, mma_k: int = MXFP6_MMA_K) -> int:
    """Number of ``m16n8k32`` MMA K-blocks covered by ``tile_k``."""
    if tile_k % mma_k != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by mma_k={mma_k}")
    return tile_k // mma_k


def mxfp6_packed_k_bytes(k: int) -> int:
    """Packed storage bytes along K (4 FP6 values per 3 bytes)."""
    if k % 4 != 0:
        raise ValueError(f"k must be divisible by 4 for MX-FP6 packing, got {k}")
    return (3 * k) // 4


def mxfp6_logical_k_from_packed_bytes(packed_k_bytes: int) -> int:
    """Logical K element count from packed byte width along K."""
    if packed_k_bytes % 3 != 0:
        raise ValueError(f"packed_k_bytes must be divisible by 3, got {packed_k_bytes}")
    return (packed_k_bytes * 4) // 3


def mxfp6_tile_shape_mnk(
    tile_m: int,
    tile_n: int,
    sf_vec_size: int = MXFP6_SF_VEC_SIZE,
) -> Tuple[int, int, int]:
    """Return ``(tile_m, tile_n, tile_k)`` for MX-FP6 block-scaled kernels."""
    return tile_m, tile_n, mxfp6_tile_k(sf_vec_size)


def verify_mxfp6_smem_tile_k(
    tile_k: int,
    sf_vec_size: int = MXFP6_SF_VEC_SIZE,
) -> int:
    """Check ``tile_k`` satisfies ``sm120_make_smem_layout_sfa/sfb`` divisibility for MX-FP6.

    Returns ``mma_nsf = tile_k // sf_vec_size`` (4 when ``tile_k=128``).
    """
    if sf_vec_size != MXFP6_SF_VEC_SIZE:
        raise ValueError(
            f"MX-FP6 expects sf_vec_size={MXFP6_SF_VEC_SIZE}, got {sf_vec_size}"
        )
    if tile_k % sf_vec_size != 0:
        raise ValueError(
            f"tile_k={tile_k} must be divisible by sf_vec_size={sf_vec_size}"
        )
    mma_nsf = tile_k // sf_vec_size
    blk_sf = 4
    if tile_k % (blk_sf * mma_nsf) != 0:
        raise ValueError(
            f"tile_k={tile_k} must be divisible by blk_sf*mma_nsf={blk_sf * mma_nsf} "
            f"for sm120 SF smem (mma_nsf={mma_nsf})"
        )
    return mma_nsf
