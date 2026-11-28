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

"""Types, SMEM layouts and helpers shared by the context kernel resources."""

import cutlass.cute as cute
import cutlass
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import Constexpr
from ..config import FmhaConfig
from typing import Any, TypeAlias


SUPPORTED_CONTEXT_PAGE_SIZES = (4, 8, 16, 32, 64, 128)

SmemDescOffsets: TypeAlias = tuple[int, int]

# MXFP8 Q/K. One E8M0 scale per 32 channels of a token, so a 128-token tile of
# head dimension 128 carries 512 scale bytes, the 128x4 atom the quantizer
# writes. The load warp copies each atom behind its Q tile or K stage with the
# tile's own barrier. The tensor core reads the scales from tensor memory, four
# columns per operand, which the MMA warp fills with one 32x128b copy broadcast
# to the four lane quarters. The eight columns of a QK step live in the tail of
# the peer S tile, which is dead between the peer softmax's load and the peer's
# next QK.
MX_SCALE_TILE_BYTES = 512


TmemAddr: TypeAlias = int | Int32
TmemPtr: TypeAlias = cutlass.Array

# Keep masked scores finite: an all-masked tile must not form inf - inf.
SOFTMAX_MASK_VALUE = -3.4028234663852886e38  # -FLT_MAX, bits 0xff7fffff
# Open right bound of a row's visible keys in LOOP coordinates.
UNBOUNDED_K = 0x3FFFFFFF
# Warps in one softmax warp group; each owns one packed LOOP mask range.
NUM_SOFTMAX_WARPS = 4

SoftmaxScalar: TypeAlias = Float32
tmem_sp_tile_sum: dict[int, Any] = {}


@cute.jit
def mx_scale_slot(smem_base: cutlass.Array, byte_offset: int) -> cutlass.Array:
    """The shared-memory slot of one scale atom, the destination of its bulk copy."""
    return cutlass.Array(
        smem_base.data_ptr() + byte_offset,
        dtype=cutlass.Uint8,
        shape=(MX_SCALE_TILE_BYTES,),
        addrspace=3,
    )


def qk_inner_dim_size_bytes(cfg: FmhaConfig) -> int:
    """Return the byte width of one Q/K tile inner dimension."""
    return cfg.qk_mma_tiler[2] * cfg.q_dtype.width // 8


def qk_smem_desc_offsets(cfg: FmhaConfig) -> SmemDescOffsets:
    """Return Q/K descriptor leading and stride byte offsets."""
    leading_byte_offset = 0 if cfg.head_paired else 16
    stride_byte_offset = cfg.tma_copy_q_granu_inner * cfg.q_dtype.width
    return leading_byte_offset, stride_byte_offset


def qk_smem_layout(cfg: FmhaConfig) -> int:
    """Return the tcgen05 descriptor layout selector for Q/K SMEM tiles."""
    return smem_layout_for_inner_bytes(qk_inner_dim_size_bytes(cfg))


@cute.jit
def resolve_work_tile_coords(
    cfg: Constexpr[FmhaConfig],
    tile_idx: cute.Coord,
) -> tuple[Int32, Int32, Int32]:
    """Return ``(seq, head, batch)`` for the configured tile order."""
    seq_idx, head_idx, batch_idx = cfg.work_tile_coord_indices
    seq_coord = tile_idx[seq_idx]
    head_coord = tile_idx[head_idx]
    batch_coord = tile_idx[batch_idx]
    if cutlass.const_expr(cfg.uses_causal_reversed_head_batch_seq_tile_order):
        seq_coord = cfg.num_seq_tiles - seq_coord - Int32(1)
    return seq_coord, head_coord, batch_coord


def smem_layout_for_inner_bytes(inner_dim_size: int) -> int:
    """Return the tcgen05 descriptor swizzle selector for an SMEM row of this many bytes."""
    if inner_dim_size % 128 == 0:
        return 2
    if inner_dim_size == 64:
        return 4
    if inner_dim_size == 32:
        return 6
    raise RuntimeError(f"Unsupported inner dimension size: {inner_dim_size}")
