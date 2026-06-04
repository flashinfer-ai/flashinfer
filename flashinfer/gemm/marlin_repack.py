# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Host-side weight repack for the W4A16 cute-DSL kernel
# (``dense_gemm_w4a16_blackwell.py``).
#
# Layout is fully driven by the kernel's MMA partition.  For the atom config
#   atom_layout      = (2, 2, 1)        # 4 MMA warps: 2 M-warps x 2 N-warps
#   mma_inst_mnk     = (16, 8, 16)
#   permutation_mnk  = (32, 32, 16)     # *2 trick on N
#   tile_shape_mnk   = (M_tile, 64,  64)
#
# ``thr_mma.partition_B(identity)`` gives every thread 16 fp16 slots per
# K-block, arranged as ``(mma_size=4, mma_n=4)`` covering:
#   * K = { tc_row, tc_row+1, tc_row+8, tc_row+9 }
#         where tc_row = (lane % 4) * 2  in {0, 2, 4, 6}
#   * N = { base_n, base_n+16, base_n+32, base_n+48 }
#         where base_n = (warp_idx // 2) * 8 + (lane // 4)  in [0, 16)
#
# (Warps 0/1 share B; warps 2/3 share B with N shifted by +8.  See
#  scratch_partition_mapping.py for the full diagnostic.)
#
# Per thread per K-block: 16 fp16 = 8 packed FP4 bytes = 2 int32.  We
# pack each int32 such that the four bytes carry K-pairs (even/odd) at
# the same N, so ``cvt.rn.f16x2.e2m1x2`` decodes 2 fp16 with one shared
# scale.  Slot mapping per int32:
#
#     u32_idx=0 (nn=0 and nn=1):
#       byte 0: K=tc_row, tc_row+1 at N=base_n        (mma_i=0,1 nn=0)
#       byte 1: K=tc_row+8, tc_row+9 at N=base_n      (mma_i=2,3 nn=0)
#       byte 2: K=tc_row, tc_row+1 at N=base_n+16     (mma_i=0,1 nn=1)
#       byte 3: K=tc_row+8, tc_row+9 at N=base_n+16   (mma_i=2,3 nn=1)
#     u32_idx=1 (nn=2 and nn=3):
#       byte 0: K=tc_row, tc_row+1 at N=base_n+32     (mma_i=0,1 nn=2)
#       byte 1: K=tc_row+8, tc_row+9 at N=base_n+32   (mma_i=2,3 nn=2)
#       byte 2: K=tc_row, tc_row+1 at N=base_n+48     (mma_i=0,1 nn=3)
#       byte 3: K=tc_row+8, tc_row+9 at N=base_n+48   (mma_i=2,3 nn=3)
#
# Each byte uses the standard nibble convention: low nibble = even K
# (= K=tc_row or K=tc_row+8), high nibble = odd K (+1).
#
# Smem layout (per K-block): 128 int32 ordered by (n_warp_idx, lane,
# u32_idx).  n_warp_idx = warp_idx // 2; thread (warp, lane) reads
# ``sB[k_block, n_warp_idx * 64 + lane * 2 + u32_idx, stage]``.

import torch


_TILE_K: int = 16  # Marlin K-tile size = MMA K-block size
_TILE_N: int = 64  # Marlin N-tile size = kernel tile_N
_INTS_PER_TILE: int = 128  # int32s per (16K x 64N) repack block


def prepare_fp4_w4a16_weight(b: torch.Tensor) -> torch.Tensor:
    """Repack a packed FP4 weight for the W4A16 cute-DSL kernel.

    Args:
        b: ``(K // 2, N)`` ``uint8`` (or ``torch.float4_e2m1fn_x2``) tensor
           of packed FP4 values.  Byte ``b[k_half, n]`` carries the FP4
           value at K-index ``2 * k_half`` in its low nibble and
           ``2 * k_half + 1`` in its high nibble.  ``K`` must be a
           multiple of 16; ``N`` a multiple of 64.

    Returns:
        ``(K // 16, N * 2)`` ``int32`` tensor in the layout the kernel
        expects (128 int32 per (16K x 64N) block).
    """
    if b.dtype != torch.uint8:
        b = b.view(torch.uint8)

    k_half, n = b.shape
    k = k_half * 2
    if k % _TILE_K != 0:
        raise ValueError(f"K must be a multiple of {_TILE_K} (got K={k})")
    if n % _TILE_N != 0:
        raise ValueError(f"N must be a multiple of {_TILE_N} (got N={n})")

    device = b.device
    k_tiles = k // _TILE_K
    n_tiles = n // _TILE_N

    # Per-position byte-source maps (one entry per int32 in a tile).
    u32_pos = torch.arange(_INTS_PER_TILE, device=device, dtype=torch.long)
    # u32_pos layout: n_warp_idx in [0, 2) outer, lane in [0, 32) middle,
    # u32_idx in [0, 2) inner.
    u32_idx_local = u32_pos % 2
    lane = (u32_pos // 2) % 32
    n_warp_idx = u32_pos // 64

    tc_col = lane // 4  # in [0, 8)
    tc_row_half = lane % 4  # tc_row = tc_row_half * 2 in {0, 2, 4, 6}
    base_n = n_warp_idx * 8 + tc_col  # in [0, 16)

    # Each int32 packs 4 bytes; each byte packs 2 FP4 values (low/high nibble).
    # Byte k-offset (K within the 16-row tile) is in K-half units (since the
    # source ``b`` is K/2 rows).  tc_row_half is already K-half.
    # Bytes 0,2 hit K=tc_row,tc_row+1 -> source row tc_row_half.
    # Bytes 1,3 hit K=tc_row+8,tc_row+9 -> source row tc_row_half + 4.
    byte_k_half_offset = torch.tensor([0, 4, 0, 4], device=device, dtype=torch.long)

    # N offset per byte (within the 64-col tile).  u32_idx 0 covers
    # N=base_n+{0, 16}; u32_idx 1 covers N=base_n+{32, 48}.
    byte_n_offset_for_u32_0 = torch.tensor(
        [0, 0, 16, 16], device=device, dtype=torch.long
    )
    byte_n_offset_for_u32_1 = torch.tensor(
        [32, 32, 48, 48], device=device, dtype=torch.long
    )
    # Select per-u32 offsets via gather.
    n_offset_stack = torch.stack(
        [byte_n_offset_for_u32_0, byte_n_offset_for_u32_1], dim=0
    )
    # (u32_pos, 4)
    byte_n_offset = n_offset_stack[u32_idx_local]

    # Full source indices: (K_tiles, N_tiles, _INTS_PER_TILE, 4) -> (k_half, n)
    kt = torch.arange(k_tiles, device=device, dtype=torch.long)
    nt = torch.arange(n_tiles, device=device, dtype=torch.long)
    k_half_global = (
        kt[:, None, None, None] * (_TILE_K // 2)
        + tc_row_half[None, None, :, None]
        + byte_k_half_offset[None, None, None, :]
    )
    n_global = (
        nt[None, :, None, None] * _TILE_N
        + base_n[None, None, :, None]
        + byte_n_offset[None, None, :, :]
    )

    bytes_gathered = b[k_half_global, n_global]  # (K_tiles, N_tiles, 128, 4)

    # Pack 4 bytes (little-endian: byte_idx 0 in bits 0-7) into one int32.
    out64 = torch.zeros(
        (k_tiles, n_tiles, _INTS_PER_TILE), dtype=torch.int64, device=device
    )
    for byte_idx in range(4):
        out64 |= bytes_gathered[..., byte_idx].to(torch.int64) << (byte_idx * 8)

    return out64.to(torch.int32).reshape(k_tiles, n_tiles * _INTS_PER_TILE)


def unpack_fp4_w4a16_weight(b_prepared: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Reverse ``prepare_fp4_w4a16_weight``.

    Recovers the per-element FP4 codes from the kernel-format prepared
    tensor.  Used by the ``torch`` fallback backend in ``mm_fp4_w4a16``
    and by tests for round-trip validation.

    Args:
        b_prepared: ``(K // 16, N * 2)`` ``int32`` tensor produced by
            ``prepare_fp4_w4a16_weight``.
        k: Original ``K`` dimension.
        n: Original ``N`` dimension.

    Returns:
        ``(K, N)`` ``uint8`` tensor of FP4 codes (each value in [0, 16)).
    """
    if b_prepared.dtype != torch.int32:
        raise TypeError(f"b_prepared must be int32 (got {b_prepared.dtype})")
    if k % _TILE_K != 0:
        raise ValueError(f"K must be a multiple of {_TILE_K} (got K={k})")
    if n % _TILE_N != 0:
        raise ValueError(f"N must be a multiple of {_TILE_N} (got N={n})")

    device = b_prepared.device
    k_tiles = k // _TILE_K
    n_tiles = n // _TILE_N

    # Same per-position maps as prepare_fp4_w4a16_weight.
    u32_pos = torch.arange(_INTS_PER_TILE, device=device, dtype=torch.long)
    u32_idx_local = u32_pos % 2
    lane = (u32_pos // 2) % 32
    n_warp_idx = u32_pos // 64
    tc_col = lane // 4
    tc_row_half = lane % 4
    base_n = n_warp_idx * 8 + tc_col

    byte_k_half_offset = torch.tensor([0, 4, 0, 4], device=device, dtype=torch.long)
    n_offset_stack = torch.stack(
        [
            torch.tensor([0, 0, 16, 16], device=device, dtype=torch.long),
            torch.tensor([32, 32, 48, 48], device=device, dtype=torch.long),
        ],
        dim=0,
    )
    byte_n_offset = n_offset_stack[u32_idx_local]  # (u32_pos, 4)

    # Where each (k_tile, n_tile, u32_pos, byte_idx) byte lands in the
    # output (K // 2, N) packed view -- i.e. its source K-half row and
    # N column.
    kt = torch.arange(k_tiles, device=device, dtype=torch.long)
    nt = torch.arange(n_tiles, device=device, dtype=torch.long)
    k_half_global = (
        kt[:, None, None, None] * (_TILE_K // 2)
        + tc_row_half[None, None, :, None]
        + byte_k_half_offset[None, None, None, :]
    )
    n_global = (
        nt[None, :, None, None] * _TILE_N
        + base_n[None, None, :, None]
        + byte_n_offset[None, None, :, :]
    )

    # Extract the four bytes from each int32 (little-endian).
    b_view = b_prepared.reshape(k_tiles, n_tiles, _INTS_PER_TILE).to(torch.int64)
    bytes_per_u32 = torch.empty(
        (k_tiles, n_tiles, _INTS_PER_TILE, 4), dtype=torch.uint8, device=device
    )
    for byte_idx in range(4):
        bytes_per_u32[..., byte_idx] = ((b_view >> (byte_idx * 8)) & 0xFF).to(
            torch.uint8
        )

    # Scatter bytes back to (K // 2, N) packed layout.
    packed = torch.empty((k // 2, n), dtype=torch.uint8, device=device)
    packed[k_half_global, n_global] = bytes_per_u32

    # Split each packed byte into low nibble (= even K) and high nibble (= odd K).
    codes = torch.empty((k, n), dtype=torch.uint8, device=device)
    codes[0::2, :] = packed & 0x0F
    codes[1::2, :] = (packed >> 4) & 0x0F
    return codes
