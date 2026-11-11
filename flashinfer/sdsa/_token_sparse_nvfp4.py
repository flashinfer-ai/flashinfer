# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.
"""Gather selected NVFP4 MQA tokens into FP8 attention workspace."""

import triton
import triton.language as tl


# Triton requires explicit constexpr objects for globals; host code reads .value.
# Storage format: two E2M1 values per byte and one E4M3 scale per block.
FP4_BITS = tl.constexpr(4)
FP4_VALUES_PER_BYTE = tl.constexpr(2)
FP4_CODE_MASK = tl.constexpr((1 << FP4_BITS.value) - 1)
FP4_SIGN_BIT = tl.constexpr(1 << (FP4_BITS.value - 1))
FP4_MAGNITUDE_MASK = tl.constexpr(FP4_SIGN_BIT.value - 1)
E2M1_MAGNITUDES = tl.constexpr((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
NVFP4_SCALE_BLOCK_SIZE = tl.constexpr(16)

# TRT-LLM HND V scales transpose the token and scale-group axes within a tile.
V_SCALE_SWIZZLE_TOKENS = tl.constexpr(4)
V_SCALE_SWIZZLE_GROUPS = tl.constexpr(4)
HEAD_DIM_ALIGNMENT = tl.constexpr(
    NVFP4_SCALE_BLOCK_SIZE.value * V_SCALE_SWIZZLE_GROUPS.value
)
KV_COMPONENTS = tl.constexpr(2)

# E2M1's largest magnitude expands the E4M3 block-scale range by this factor.
# Divide before casting to FP8; callers restore it in the attention BMM scales.
FP8_NORMALIZATION_FACTOR = tl.constexpr(E2M1_MAGNITUDES.value[-1])

# Launch tuning, independent of the NVFP4 quantization block size.
GATHER_TOKENS_PER_TILE = tl.constexpr(16)


@triton.jit
def gather_kernel(
    K,
    V,
    KS,
    VS,
    Slots,
    Lengths,
    Out,
    Tables,
    PAGE_STRIDE,
    SF_PAGE_STRIDE,
    PAGE: tl.constexpr,
    DIM: tl.constexpr,
    TOP_K,
    SLOT_ROW,
    SLOT_SIDE,
    BLOCK_DIM: tl.constexpr,
    V_SF_SWIZZLED: tl.constexpr,
):
    """Decode valid token/head lanes and scatter normalized K/V to FP8 rows.

    Strides and ``TOP_K`` are runtime values, so new cache layouts or Top-K
    sizes reuse the compiled kernel; page size, head dim and the V-scale
    layout specialize it.
    """
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    col = tile * GATHER_TOKENS_PER_TILE + tl.arange(0, GATHER_TOKENS_PER_TILE)
    dim = tl.arange(0, BLOCK_DIM)
    length = tl.load(Lengths + row)
    scales_per_token: tl.constexpr = DIM // NVFP4_SCALE_BLOCK_SIZE
    scales_per_group: tl.constexpr = scales_per_token // V_SCALE_SWIZZLE_GROUPS
    for side in tl.static_range(KV_COMPONENTS):
        slot = tl.load(
            Slots + row * SLOT_ROW + side * SLOT_SIDE + col, col < TOP_K, other=-1
        ).to(tl.int64)
        live = (col < length) & (slot >= 0)
        page, token = tl.maximum(slot, 0) // PAGE, tl.maximum(slot, 0) % PAGE
        data_ptr = K if side == 0 else V
        sf_ptr = KS if side == 0 else VS
        packed = tl.load(
            data_ptr
            + page[:, None] * PAGE_STRIDE
            + token[:, None] * (DIM // FP4_VALUES_PER_BYTE)
            + dim[None, :] // FP4_VALUES_PER_BYTE,
            live[:, None] & (dim[None, :] < DIM),
            other=0,
        ).to(tl.int32)
        code = (
            packed >> (FP4_BITS * (dim[None, :] % FP4_VALUES_PER_BYTE))
        ) & FP4_CODE_MASK
        mag = code & FP4_MAGNITUDE_MASK
        # Decode the format's eight magnitude codes using the E2M1 value table.
        value = tl.where(
            mag == 0,
            E2M1_MAGNITUDES[0],
            tl.where(
                mag == 1,
                E2M1_MAGNITUDES[1],
                tl.where(
                    mag == 2,
                    E2M1_MAGNITUDES[2],
                    tl.where(
                        mag == 3,
                        E2M1_MAGNITUDES[3],
                        tl.where(
                            mag == 4,
                            E2M1_MAGNITUDES[4],
                            tl.where(
                                mag == 5,
                                E2M1_MAGNITUDES[5],
                                tl.where(
                                    mag == 6, E2M1_MAGNITUDES[6], E2M1_MAGNITUDES[7]
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        value = tl.where(code >= FP4_SIGN_BIT, -value, value)
        # Swizzled V: [token tile, scale group, scale within group, token
        # within tile]. K scales, and V scales when not swizzled, are linear
        # [token, scale].
        sf_dim = dim // NVFP4_SCALE_BLOCK_SIZE
        sf_offset = (
            (
                token[:, None] // V_SCALE_SWIZZLE_TOKENS * V_SCALE_SWIZZLE_GROUPS
                + sf_dim[None, :] // scales_per_group
            )
            * scales_per_token
            + sf_dim[None, :] % scales_per_group * V_SCALE_SWIZZLE_TOKENS
            + token[:, None] % V_SCALE_SWIZZLE_TOKENS
        )
        if side == 0 or not V_SF_SWIZZLED:
            sf_offset = token[:, None] * scales_per_token + sf_dim[None, :]
        sf = tl.load(
            sf_ptr + page[:, None] * SF_PAGE_STRIDE + sf_offset,
            live[:, None] & (dim[None, :] < DIM),
            other=0.0,
        ).to(tl.float32)
        # Normalized NVFP4 reaches 6*448, while FP8 reaches 448. The caller
        # restores the factor 6 and the global K/V scales in BMM1/BMM2.
        value = (value * sf / FP8_NORMALIZATION_FACTOR).to(Out.dtype.element_ty)
        tl.store(
            Out
            + (row * TOP_K + col[:, None]) * (KV_COMPONENTS * DIM)
            + side * DIM
            + dim[None, :],
            value,
            (col[:, None] < TOP_K) & (dim[None, :] < DIM),
        )
        tl.store(
            Tables + (row * KV_COMPONENTS + side) * TOP_K + col,
            row * TOP_K + col,
            col < TOP_K,
        )
