"""Shared MX-FP6 helpers for the fused MoE dynamic kernels (w6a8_mx recipe)."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint8, Uint64
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.cute.nvgpu.warp.mma import Field as WarpField

from b12x._lib.intrinsics import get_ptr_as_int64, st_global_u64
from b12x._lib.fp6 import (
    quantize_block_fp6_e2m3,
    quantize_block_fp6_e2m3_containers,
    quantize_block_fp6_e2m3_fast,
    quantize_block_fp6_e3m2,
    quantize_block_fp6_e3m2_containers,
    quantize_block_fp6_e3m2_fast,
    quantize_block_fp8_e4m3_containers,
)
from b12x._lib.dense_gemm_mxfp6 import emit_mxfp6_dense_mma_k_block


@cute.jit
def moe_mxfp6_quantize_input_block(
    values: cute.Tensor,
    block_max: Float32,
    gs_value: Float32,
    a_dtype,
    fast_math: bool,
) -> tuple[Uint64, Uint64, Uint64, Uint8]:
    """Quantize one 32-element MX-FP6 block for route-pack / FC2 requant."""
    if cutlass.const_expr(a_dtype == cutlass.Float6E3M2FN):
        if fast_math:
            return quantize_block_fp6_e3m2_fast(values, block_max, gs_value)
        return quantize_block_fp6_e3m2(values, block_max, gs_value)
    if fast_math:
        return quantize_block_fp6_e2m3_fast(values, block_max, gs_value)
    return quantize_block_fp6_e2m3(values, block_max, gs_value)


@cute.jit
def moe_mxfp6_quantize_input_block_containers(
    values: cute.Tensor,
    block_max: Float32,
    gs_value: Float32,
    fmt_a: cutlass.Constexpr,
) -> tuple[cute.Tensor, Uint8]:
    """Quantize one 32-element MX-FP6 block to 32 byte-containers + scale byte.

    Byte-container analogue of :func:`moe_mxfp6_quantize_input_block`: each of the
    32 returned ``Uint8`` elements carries one FP6 code in bits ``[5:0]`` (the
    ``mxf8f6f4`` smem/ldmatrix layout). ``fmt_a`` selects the sub-format
    (``"e3m2"`` / ``"e2m3"`` / ``"e4m3"`` for W6A8 activations); the operands
    are otherwise indistinguishable from FP8 by dtype alone, so the format is
    passed explicitly.
    """
    if cutlass.const_expr(fmt_a == "e3m2"):
        return quantize_block_fp6_e3m2_containers(values, block_max, gs_value)
    if cutlass.const_expr(fmt_a == "e4m3"):
        return quantize_block_fp8_e4m3_containers(values, block_max, gs_value)
    return quantize_block_fp6_e2m3_containers(values, block_max, gs_value)


@cute.jit
def moe_mxfp6_store_packed_global(
    storage: cute.Tensor,
    byte_offset: Int32,
    lo: Uint64,
    mid: Uint64,
    hi: Uint64,
) -> None:
    """Store 24 packed MX-FP6 bytes (three u64 lanes) to global uint8 storage."""
    base = get_ptr_as_int64(storage, byte_offset)
    st_global_u64(base, lo)
    st_global_u64(base + Int64(8), mid)
    st_global_u64(base + Int64(16), hi)


@cute.jit
def moe_mxfp6_store_bytes_u64_global(
    storage: cute.Tensor,
    byte_offset: Int32,
    q0: Uint64,
    q1: Uint64,
    q2: Uint64,
    q3: Uint64,
) -> None:
    """Store 32 FP6 byte-containers (four u64 lanes) to global uint8 storage.

    Byte-container counterpart of :func:`moe_mxfp6_store_packed_global`: 32
    bytes (one code each) instead of 24 packed bytes, same vectorized
    ``st.global.u64`` path. ``byte_offset`` must stay 8-byte aligned (the
    bytes-mode block stride of 32 guarantees it).
    """
    base = get_ptr_as_int64(storage, byte_offset)
    st_global_u64(base, q0)
    st_global_u64(base + Int64(8), q1)
    st_global_u64(base + Int64(16), q2)
    st_global_u64(base + Int64(24), q3)


@cute.jit
def moe_mxfp6_store_expanded_global(
    storage: cute.Tensor,
    byte_offset: Int32,
    containers: cute.Tensor,
) -> None:
    """Store 32 expanded MX-FP6 byte-containers to flat global uint8 storage.

    Byte-container analogue of :func:`moe_mxfp6_store_packed_global`: writes one
    FP6 code per byte (K bytes per 32-element block) so the activation scratch is
    the Float8E4M3FN byte-container layout the migrated ``mxf8f6f4`` A path / TMA
    consumes (cutlass has no working 6-bit smem layout).
    """
    for byte_idx in cutlass.range_constexpr(32):
        storage[byte_offset + Int32(byte_idx)] = containers[byte_idx]


@cute.jit
def moe_mxfp6_store_packed_smem_swizzled(
    sA_u8: cute.Tensor,
    row: Int32,
    sf_block: Int32,
    packed_cols: Int32,
    lo: Uint64,
    mid: Uint64,
    hi: Uint64,
) -> None:
    """Scatter 24 packed bytes into swizzled activation smem (matches FP4 layout)."""
    packed_base = sf_block * Int32(24)
    dst_pcol = row & Int32(63)
    xor_bits = ((dst_pcol >> Int32(1)) & Int32(0x3)) << Int32(4)
    row_high = row >> Int32(6)
    for byte_idx in cutlass.range_constexpr(24):
        src_pcol = packed_base + Int32(byte_idx)
        dst_row = ((src_pcol ^ xor_bits) << Int32(1)) + row_high
        dst_flat = dst_row * packed_cols + dst_pcol
        if byte_idx < 8:
            packed_u64 = lo
            shift = byte_idx
        elif byte_idx < 16:
            packed_u64 = mid
            shift = byte_idx - 8
        else:
            packed_u64 = hi
            shift = byte_idx - 16
        byte_val = Uint8((packed_u64 >> Uint64(shift * 8)) & Uint64(0xFF))
        sA_u8[dst_flat] = byte_val


@cute.jit
def moe_mxfp6_store_packed_smem_logical(
    sA_u8: cute.Tensor,
    row: Int32,
    sf_block: Int32,
    lo: Uint64,
    mid: Uint64,
    hi: Uint64,
) -> None:
    """Store 24 packed FP6 bytes (one 32-elem block) into a 2-D packed-byte smem.

    Unlike :func:`moe_mxfp6_store_packed_smem_swizzled`, this writes logical
    ``sA_u8[row, packed_base + byte_idx]`` indices and relies on the smem tensor's
    own layout (and the matching TMA descriptor) to place bytes physically. Used by
    the standalone BF16->FP6 quantizer, whose FP6 smem is a plain packed-Uint8
    layout (no MMA consumes it, so no FP4-style swizzle is required).
    """
    packed_base = sf_block * Int32(24)
    for byte_idx in cutlass.range_constexpr(24):
        # byte_idx is a compile-time int, so select the source u64 / shift in pure
        # Python (no dynamic control flow) to avoid the DSL's "variable defined in
        # control flow" restriction.
        packed_u64 = (lo, mid, hi)[byte_idx // 8]
        shift = byte_idx % 8
        byte_val = Uint8((packed_u64 >> Uint64(shift * 8)) & Uint64(0xFF))
        sA_u8[row, packed_base + Int32(byte_idx)] = byte_val


def moe_emit_fp4_mma(
    mma_op,
    accumulators: cute.Tensor,
    tCrA: cute.Tensor,
    tCrSFA: cute.Tensor,
    tCrB: cute.Tensor,
    tCrSFB: cute.Tensor,
    mt: int,
    nt: int,
    k_block_idx: int,
) -> None:
    """One FP4 block-scaled warp MMA (SFA/SFB via a freshly created atom).

    Takes the MMA *op descriptor* (``self.mma_op``), not a pre-made atom.
    ``mma_atom.set`` produces a new MLIR atom SSA value; if a single
    kernel-lifetime atom is chained through ``set`` calls across dynamic scf
    regions (k-tile mainloops, persistent while-loops), the CUTLASS DSL fails
    to thread the updated value through the region boundaries and the IR
    verifier rejects the module with "operand does not dominate this use"
    (observed on the FP4 gated static MoE kernel with >1 M-tile). Creating a
    fresh atom per emission keeps every atom value local to one straight-line
    block, so dominance holds by construction. The atom is trace-time
    metadata only — no runtime cost.
    """
    mma_atom = cute.make_mma_atom(mma_op)
    mma_atom.set(WarpField.SFA, tCrSFA[None, mt, k_block_idx].iterator)
    mma_atom.set(WarpField.SFB, tCrSFB[None, nt, k_block_idx].iterator)
    cute.gemm(
        mma_atom,
        accumulators[None, mt, nt],
        tCrA[None, mt, k_block_idx],
        tCrB[None, nt, k_block_idx],
        accumulators[None, mt, nt],
    )


def moe_emit_mma_k_block(
    mma_op,
    accumulators: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCrSFA: cute.Tensor,
    tCrSFB: cute.Tensor,
    mt: int,
    nt: int,
    k_block_idx: int,
    fmt_a,
    fmt_b,
) -> None:
    """Dispatch one K-block MMA for MoE (inline MX-FP6 or native FP4).

    ``fmt_a`` / ``fmt_b`` accept either FP6/FP8 sub-format strings (``"e3m2"``
    / ``"e2m3"`` / ``"e4m3"`` — the byte-container ``mxf8f6f4`` path) or the
    legacy ``Float6E3M2FN`` / ``Float6E2M3FN`` dtypes; both emit the inline
    ``mxf8f6f4`` MMA. ``None`` or a non-FP6 dtype falls back to the native FP4
    block-scaled MMA. The byte-container FP6 operands are carried in
    ``Float8E4M3FN``, indistinguishable from FP8 by dtype, so the sub-format
    must be passed explicitly.

    Plain Python (not ``@cute.jit``) and takes the MMA *op descriptor*
    (``self.mma_op``) rather than a shared atom — the FP4 branch creates a
    fresh atom per emission; see :func:`moe_emit_fp4_mma`.
    """
    if fmt_a == "e3m2" or fmt_a == "e2m3" or fmt_a == "e4m3":
        emit_mxfp6_dense_mma_k_block(
            accumulators,
            tCrA,
            tCrB,
            tCrSFA,
            tCrSFB,
            mt,
            nt,
            k_block_idx,
            fmt_a,
            fmt_b,
        )
    elif fmt_a == cutlass.Float6E3M2FN or fmt_a == cutlass.Float6E2M3FN:
        # Legacy dynamic path passes Float6 dtypes; translate to fmt strings.
        emit_mxfp6_dense_mma_k_block(
            accumulators,
            tCrA,
            tCrB,
            tCrSFA,
            tCrSFB,
            mt,
            nt,
            k_block_idx,
            "e3m2" if fmt_a == cutlass.Float6E3M2FN else "e2m3",
            "e3m2" if fmt_b == cutlass.Float6E3M2FN else "e2m3",
        )
    else:
        moe_emit_fp4_mma(
            mma_op,
            accumulators,
            tCrA,
            tCrSFA,
            tCrB,
            tCrSFB,
            mt,
            nt,
            k_block_idx,
        )
