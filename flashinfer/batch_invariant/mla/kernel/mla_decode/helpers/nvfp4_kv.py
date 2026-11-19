# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Storage and loader helpers shared by the 1CTA/2CTA NVFP4 MLA paths.

External records are token-interleaved uint8 bytes, with no global NVFP4 scale:

    D512 NVFP4: [ packed E2M1: 256 B ][ E4M3 block16 scales: 32 B ]
    D576 NVFP4: [ packed E2M1: 256 B ][ E4M3 RoPE: 64 B ][ scales: 32 B ]
    D512 FP8:   [ E4M3 K/V: 512 B ]   (primary SWA pool only)

D512 scale block b is at byte 256+b. D576 scale block b is at
320+8*(b%4)+b//4; its RoPE bytes are unscaled. Each NVFP4 byte stores the
even-indexed value in its low nibble. FP8 primary storage uses one separate
runtime tensor scale; QK/output arithmetic applies it without changing bytes.

Supported pairs are native/native, NVFP4/NVFP4, D576 NVFP4+FP8-RoPE in both
pools, and D512 FP8 primary/NVFP4 extra. External physical-page strides are
preserved when callers expose the internal [1, record_bytes, storage_rows]
view. Source-tagged indices select either descriptor: bit31 means extra,
low31 bits select a storage row, and all-one low bits represent an invalid row.
Both families gather the same records, then consume ordinary FP8 MMA tiles.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL
from cutlass.experimental import cuda, primitives as prims

from ......cute_dsl.availability import is_rubin_cute_dsl_available
from ...tensor_map import create_tensor_map_tiled_from_view
from .gather import _address
from ...fmha_decode.fmha_decode_resources.tmem_transformed_kv import (
    _has_packed_e4m3_e2m1_mul,
)
from ...fmha_decode.fmha_decode_resources.smem_transformed_kv import (
    _cvt_e2m1_word_to_e4m3_words,
)


# ===========================================================================
# Storage formats and configuration
# ===========================================================================


def cache_record_size(cache_format: str, rope_dim: int) -> int:
    """Elements per token; quantized formats use uint8 elements."""
    return {
        "native": 512 + rope_dim,
        "nvfp4": 288,
        "nvfp4.fp8rope": 352,
        "fp8": 512,
    }[cache_format]


def validate_nvfp4_kv_formats(
    primary_format, extra_format, rope_dim, qkv_dtype, page_size
):
    """Reject unsupported storage/compute pairs before constructing tensor maps.

    Page size one is the internal physical-row view, not an external cache-page
    restriction. Native/native retains the caller's ordinary paged-cache path.
    A mixed pair has FP8 only in the primary SWA pool and NVFP4 in the extra.
    """
    formats = (primary_format, extra_format)
    allowed = (
        ("native", "native"),
        ("nvfp4", "nvfp4"),
        ("nvfp4.fp8rope", "nvfp4.fp8rope"),
        ("fp8", "nvfp4"),
    )
    if formats not in allowed:
        raise ValueError(
            "cache formats must be native/native, nvfp4/nvfp4, nvfp4.fp8rope/nvfp4.fp8rope, or fp8 primary/nvfp4 extra"
        )
    if primary_format == "native":
        return
    if qkv_dtype != "e4m3":
        raise ValueError("NVFP4 KV formats require E4M3 Q")
    if page_size != 1:
        raise ValueError("NVFP4 KV requires an internal page-size-one storage-row view")
    if primary_format == "nvfp4.fp8rope":
        if rope_dim != 64:
            raise ValueError("nvfp4.fp8rope requires 512 NoPE and 64 RoPE channels")
    elif rope_dim != 0:
        raise ValueError("nvfp4/fp8 require D512 QK/V without an extra RoPE tail")


class Nvfp4KVConfig:
    """Common format predicates for both attention families."""

    @property
    def has_nvfp4_kv(self) -> bool:
        """Select row-record loaders for NVFP4 pools or the FP8-primary/NVFP4-extra pair."""
        return self.kv_cache_format != "native"

    @property
    def has_fp8_primary(self) -> bool:
        """Return whether primary K/V bytes need a separate per-tensor descale."""
        return self.kv_cache_format == "fp8"


# ===========================================================================
# Shared-memory addressing and publication
# ===========================================================================


@cute.jit
def publish_nvfp4_kv(barrier, count, *, cta_group: cutlass.Constexpr[int] = 1):
    """Publish ordinary shared stores before signaling the MMA pipeline."""
    cute.arch.fence_view_async_shared()
    # The elected publisher must observe every lane's converted stores.
    # Proxy fencing alone does not make a relaxed manual complete_tx a
    # release operation. Join the warp, then publish through a release fence.
    cute.arch.sync_warp()
    if prims.elect_sync():
        if cutlass.const_expr(cta_group == 2):
            prims.fence_sync_restrict(prims.MemOrder.RELEASE)
            # Use the cluster address mapping API, not encoded CTA-rank bits.
            leader = prims.mapa(barrier, Int32(0))
            prims.mbarrier_complete_tx(
                leader, Int32(count), scope=prims.MemScope.CLUSTER
            )
        else:
            cute.arch.inline_ptx("fence.acq_rel.cta;")
            prims.mbarrier_complete_tx(barrier, Int32(count))


@cute.jit
def swizzle_byte_offset(offset, width: cutlass.Constexpr[int]):
    """Map an unswizzled byte offset into the matching 32/64/128-byte TMA swizzle.

    The selected XOR mask changes the 16-byte bank group using row bits;
    callers must use the same swizzle width as the tensor map and MMA view."""
    mask = {32: 0x10, 64: 0x30, 128: 0x70}[width]
    return offset ^ ((offset >> Int32(3)) & Int32(mask))


@cute.jit
def store_shared_u32x4(dst, offset, a, b, c, d):
    """Store four packed 32-bit words at dst + byte offset with one 16-byte SMEM store.

    The destination must be 16-byte aligned. Publication is a separate
    operation so a loader can finish its complete tile before signaling MMA."""
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[_address(dst, Int32) + offset, a, b, c, d],
    )


# ===========================================================================
# Primary-pool FP8 selection
# ===========================================================================


# ===========================================================================
# NVFP4 conversion
# ===========================================================================

NVFP4_DATA_BYTES = 256


NVFP4_SCALE_BYTES = 32


@cute.jit
def nvfp4_scale_byte_offset(block, rope_dim: cutlass.Constexpr[int] = 64):
    """Offset within the scale region; only the mixed format permutes it."""
    if cutlass.const_expr(rope_dim == 0):
        return block
    return 8 * (block % 4) + block // 4


def nvfp4_row_bytes(rope_dim: int) -> int:
    """Packed D512, optional unscaled FP8 tail, and block16 E4M3 scales."""
    return NVFP4_DATA_BYTES + rope_dim + NVFP4_SCALE_BYTES


def is_sm107_target() -> bool:
    """Whether the CuTe DSL compile target is SM107 itself (``sm_107``
    through ``sm_107f``), so SM107-only resources such as its 327 KiB of SMEM
    per CTA (SM100: 227 KiB) can be assumed. The ``sm_100f`` family target
    is excluded: it also builds for SM100/SM103.
    """
    arch = BaseDSL._get_dsl().get_arch_enum()
    return is_rubin_cute_dsl_available() and Arch.sm_107 <= arch <= Arch.sm_107f


def use_sm107_nvfp4_converter() -> bool:
    """Whether the compile target is SM107, which uses the f16x2 converter.

    ptxas rejects the packed E4M3 x E2M1 multiply for SM107 and for the
    sm_100f family target, which SM107 compiles through on CuTe DSL < 4.8
    (``CUTE_DSL_ARCH=sm_100f``). Every other target keeps the original
    converter.
    """
    arch = BaseDSL._get_dsl().get_arch_enum()
    return arch == Arch.sm_100f or is_sm107_target()


@cute.jit
def prepare_nvfp4_scale(scale: Int32) -> Int32:
    """Return the ``convert_nvfp4x8`` scale operand for one block16 scale.

    ``scale`` is a zero-extended E4M3 byte; prepare it once per sixteen-value
    block. Other targets get the byte unchanged. On SM107 it becomes the f16x2
    pair (scale, scale), built on the FMA pipe: a non-negative E4M3 byte at
    bit 7 of an f16 lane encodes scale * 2^-8 exactly, E4M3 subnormals
    included, and one HMUL2 by 256 restores it. The per-value conversions are
    ALU-pipe F2FP instructions, so a per-word E4M3 -> f16 scale convert would
    compete with them. CuTe DSL has no plain f16x2 multiply primitive, hence
    the inline PTX.
    """
    if cutlass.const_expr(not use_sm107_nvfp4_converter()):
        return scale
    return cute.arch.inline_ptx(
        "{ .reg .b32 k;\n  mov.b32 k, 0x5C005C00;\n  mul.rn.f16x2 {$w0}, {$r0}, k;\n}",
        write_only_types=[Int32],
        read_only_args=[scale * Int32(0x00800080)],
    )


@cute.jit
def convert_nvfp4x8(word: Int32, scale: Int32):
    """Convert eight packed E2M1 values with one E4M3 block-scale byte.

    ``word`` contains eight consecutive four-bit values; ``scale`` is the
    ``prepare_nvfp4_scale`` operand (the E4M3 byte in its low eight bits except
    on SM107). Return two words containing eight E4M3 bytes, rounded once with
    saturation for ordinary FP8 MMA. No tensor-wide NVFP4 scale is applied.
    """
    if cutlass.const_expr(use_sm107_nvfp4_converter()):
        # The f16x2 sequence of _cvt_e2m1_word_to_e4m3_words, with the scale
        # pair prepared once per block by prepare_nvfp4_scale instead of once
        # per word. E2M1 x E4M3 has at most six significant bits and stays
        # inside the f16 range, so the only rounding is the final E4M3
        # convert, matching the packed multiply.
        return cute.arch.inline_ptx(
            "{ .reg .b8 b0, b1, b2, b3;\n"
            "  .reg .b32 h0, h1, h2, h3;\n"
            "  .reg .b16 e0, e1, e2, e3;\n"
            "  mov.b32 {b0, b1, b2, b3}, {$r0};\n"
            "  cvt.rn.f16x2.e2m1x2 h0, b0;\n"
            "  cvt.rn.f16x2.e2m1x2 h1, b1;\n"
            "  cvt.rn.f16x2.e2m1x2 h2, b2;\n"
            "  cvt.rn.f16x2.e2m1x2 h3, b3;\n"
            "  mul.rn.f16x2 h0, h0, {$r1};\n"
            "  mul.rn.f16x2 h1, h1, {$r1};\n"
            "  mul.rn.f16x2 h2, h2, {$r1};\n"
            "  mul.rn.f16x2 h3, h3, {$r1};\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n"
            "  cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
            "  mov.b32 {$w0}, {e0, e1};\n"
            "  mov.b32 {$w1}, {e2, e3};\n"
            "}",
            write_only_types=[Int32, Int32],
            read_only_args=[word, scale],
        )
    if cutlass.const_expr(not _has_packed_e4m3_e2m1_mul()):
        # Older DSL builds lack packed multiply even with a newer system
        # toolkit. Convert dense nibbles directly instead of unpacking to
        # bytes only to repack them inside the multiply fallback.
        return _cvt_e2m1_word_to_e4m3_words(word, scale)
    scales = (scale & Int32(255)) * Int32(0x01010101)
    # Each 16-bit half holds four contiguous E2M1 nibbles. CUDA 13.4's
    # dense packed operand avoids padded-nibble masks and output permutations.
    values = cute.arch.inline_ptx(
        "{ .reg .b16 lo, hi;\n"
        "  mov.b32 {lo, hi}, {$r0};\n"
        "  mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite {$w0}, lo, {$r1};\n"
        "  mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite {$w1}, hi, {$r1};\n"
        "}",
        write_only_types=[Int32, Int32],
        read_only_args=[word, scales],
    )
    return values[0], values[1]


# ===========================================================================
# TMA descriptors, gathering and staging
# ===========================================================================


@cute.jit
def make_whole_record_map(pool, record_bytes: cutlass.Constexpr[int]):
    """Describe a complete byte record as Uint16 values for unswizzled gather4.

    All supported records fit the 256-element TMA box limit using Uint16.
    A 32-byte raw row pitch makes each four-row destination 128-byte aligned;
    global rows retain their original record stride and are not repacked.
    """
    word_bytes = 2
    raw_bytes = (record_bytes + 31) // 32 * 32
    packed = cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint16,
            pool.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        cute.make_layout(
            (record_bytes // word_bytes, pool.shape[2]),
            stride=(1, pool.stride[2] // word_bytes),
        ),
    )
    return create_tensor_map_tiled_from_view(
        packed,
        box_dims=(raw_bytes // word_bytes, 1),
        stride_order=(0, 1),
        swizzle=cuda.TensorMapSwizzle.none,
        l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
    )


@cute.jit
def prefetch_nvfp4_kv_row(primary, extra, route, cfg):
    """Warm the complete selected record while metadata runs ahead of conversion."""
    row = route & Int32(0x7FFFFFFF)
    address = primary.iterator.toint()
    capacity = Int32(primary.shape[2])
    stride = Int32(primary.stride[2])
    width = Int32(cache_record_size(cfg.kv_cache_format, cfg.rope_dim))
    if route < Int32(0):
        address = extra.iterator.toint()
        capacity = Int32(extra.shape[2])
        stride = Int32(extra.stride[2])
        width = Int32(cache_record_size(cfg.extra_kv_cache_format, cfg.rope_dim))
    if row < capacity:
        address += Int64(row) * Int64(stride)
        for line in cutlass.range_constexpr(3):
            cute.arch.inline_ptx(
                "prefetch.global.L2 [{$r0}];",
                read_only_args=[address + Int64(line * 128)],
            )
        if cutlass.const_expr(cfg.rope_dim):
            cute.arch.inline_ptx(
                "prefetch.global.L2 [{$r0}];", read_only_args=[address + Int64(351)]
            )
        if cutlass.const_expr(cfg.has_fp8_primary):
            if width == Int32(512):
                for line in cutlass.range_constexpr(3, 4):
                    cute.arch.inline_ptx(
                        "prefetch.global.L2 [{$r0}];",
                        read_only_args=[address + Int64(line * 128)],
                    )
