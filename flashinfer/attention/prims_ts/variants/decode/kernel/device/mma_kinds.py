# Copyright (c) 2026 by FlashInfer team.
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

"""tcgen05 MMA kinds, K steps and SMEM operand swizzles of the decode kernel."""

from cutlass import Float32, Int32
from cutlass.experimental import primitives as prims
from ..config import FmhaDecodeConfig


def kv_dtype_elements(cfg: FmhaDecodeConfig) -> int:
    """Return element count in one staged K/V SMEM allocation."""
    return cfg.smem_kv_tile_elements


def kv_smem_swizzle(
    cfg: FmhaDecodeConfig,
) -> prims.Tcgen05SmemSwizzle:
    """Select the tcgen05 SMEM swizzle for raw staged K/V tiles."""
    if cfg.k_dtype.width == 8 and cfg.v_dtype.width == 8 and cfg.headdim == 64:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_64B
    return prims.Tcgen05SmemSwizzle.SWIZZLE_128B


def major_k_stride_bytes(dtype_bytes: int, headdim: int) -> int:
    """Return the descriptor K-major stride in bytes for one swizzle block."""
    # The descriptor swizzle shape depends on head dim and operand type, not
    # on the number of Q rows in the tile.
    num_smem_cols = 128 // dtype_bytes
    rows_per_smem_row = max(1, num_smem_cols // headdim)
    if rows_per_smem_row == 1:
        rows_per_swizzle_blk = 8
    elif rows_per_smem_row == 2:
        rows_per_swizzle_blk = 4
    elif rows_per_smem_row == 4:
        rows_per_swizzle_blk = 2
    else:
        rows_per_swizzle_blk = 1
    return 128 * rows_per_swizzle_blk


def mma_k_step_pv(cfg: FmhaDecodeConfig) -> int:
    """Return the K dimension advanced by one PV-GEMM MMA instruction."""
    return cfg.pv_mma_k_step


def mma_k_step_qk(cfg: FmhaDecodeConfig) -> int:
    """Return the K dimension advanced by one QK-GEMM MMA instruction."""
    return 32 if cfg.qk_dtype_bytes == 1 else 16


def mma_kind_for_pv(cfg: FmhaDecodeConfig) -> prims.Tcgen05MMAKind:
    """Select the tcgen05 MMA opcode family for the PV GEMM."""
    if cfg.use_fp8_pv:
        return prims.Tcgen05MMAKind.F8F6F4
    return prims.Tcgen05MMAKind.F16


def mma_kind_for_qk(cfg: FmhaDecodeConfig) -> prims.Tcgen05MMAKind:
    """Select the tcgen05 MMA kind of the QK GEMM from ``qk_dtype``.

    Int8 accumulates INT32 scores; E4M3 and the 16-bit types accumulate FP32.
    The INT8 kind exists on SM100 alone, so planning admits Int8 Q/K only on
    that architecture (``_validate_int8_qk_device``).
    """
    if cfg.uses_int32_scores:
        return prims.Tcgen05MMAKind.INT8
    if cfg.use_fp8_qk:
        return prims.Tcgen05MMAKind.F8F6F4
    return prims.Tcgen05MMAKind.F16


def q_dtype_elements(cfg: FmhaDecodeConfig) -> int:
    """Return element count in one staged Q SMEM allocation."""
    return cfg.smem_q_tile_bytes // cfg.q_dtype_bytes


def qk_accumulator_dtype(cfg: FmhaDecodeConfig) -> type:
    """Return the type the QK GEMM accumulates one score as: Int32 for Int8 Q/K.

    INT32 scores are accumulated on top of ``INT32_SCORE_BIAS``, so the
    softmax reads every score tile as FP32.
    """
    return Int32 if cfg.uses_int32_scores else Float32


def qkv_smem_swizzle(
    cfg: FmhaDecodeConfig,
) -> prims.Tcgen05SmemSwizzle:
    """Select the tcgen05 SMEM swizzle for staged Q tiles."""
    if cfg.q_dtype_bytes == 1 and cfg.headdim == 64:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_64B
    return prims.Tcgen05SmemSwizzle.SWIZZLE_128B
