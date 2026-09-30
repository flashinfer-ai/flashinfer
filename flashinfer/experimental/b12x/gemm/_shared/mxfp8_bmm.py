"""Shared rowwise-MXFP8 BMM lowering for GEMM-family operations.

The generic public operation is a conventional batched matrix multiply over a
BF16 left-hand side and a rowwise-MXFP8 right-hand side::

    C[g, m, n] = sum_k A[g, m, k] * dequant(W[g, k, n])

The RHS stays in its native E4M3-value + per-32 E8M0-scale representation.
Two physical layouts are supported without repacking:

``b_major="n", sf_axis="n"``
    values ``[G, K, N]`` and scales ``[G, K, N/32]``.  The scale axis is the
    GEMM N axis and the native tile is fed to the MMA transposed.
``b_major="k", sf_axis="k"``
    values ``[G, N, K]`` and scales ``[G, N, K/32]``.  The scale axis is the
    GEMM K axis and the native tile is the usual row-major RHS.

Both forms may be strided views into a larger packed tensor.  Runtime strides
describe that storage.

``dequant(value, scale)`` is one rounded BF16 multiply and is bitwise equal to
the reference ``value.to(bf16) * scale.view(e8m0).to(bf16)``.  Viewing scale
bytes as E8M0 is essential: converting raw uint8 values numerically would turn
the exponent encoding into unrelated integer magnitudes.

Numerics: fp8->bf16 conversion is exact (e4m3 is a subset of bf16, via the
hardware cvt e4m3x2->f16x2->f32->bf16 chain, all steps exact); e8m0->bf16
matches CUDA for every byte, including byte 0's subnormal and byte 255's NaN.
The value*scale product is a single RN bf16 multiply, the same operation torch
performs.  The FP4-path dequant primitives in ``b12x._lib.intrinsics``
embed sign in the scale and compensate a 2^7 bias in the epilogue, so this
specialization uses unbiased conversions to preserve the bitwise contract.

The backend accepts BF16 A/C, ``1 <= M <= 32``, and the two geometries
declared in ``_QUALIFIED_GEOMETRIES``.  The launch is deterministic, K-unsplit,
workspace free, allocation free, and CUDA-graph safe after explicit
prewarming.  A JIT miss during capture raises instead of recording a corrupt
graph.

``gemm.mla_query_projection`` reuses the same tiled BMM core with a
query-assembly and optional static-FP8 epilogue. Keeping both lowerings here
avoids either public operation reaching through the other's private package.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from collections.abc import Iterable
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cudart
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    bfloat2_mul,
    bf16_mma_m16n8k16_f32,
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    ld_global_b16,
    ld_shared_u8_offset,
    ldmatrix_m8n8x4_b16,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_u16,
    st_shared_v4_u32,
)
from b12x.preparation.types import plan_from_handle, require_prepared
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import register_program_cache

# ---------------------------------------------------------------------------
# Qualified tile configuration: tiny-M tiles and a 3-stage cp.async pipeline.
# ---------------------------------------------------------------------------
_TILE_N = 128
_TILE_K = 64
_STAGES = 3
_CTA_THREADS = 128  # 4 warps; each warp owns a 32-wide N slice of the tile.
_MAX_M = 32
_KERNEL_ID = "gemm.bmm.mxfp8"
_KERNEL_VERSION = 1
_MLA_QUERY_KERNEL_ID = "gemm.mla_query_projection.mxfp8"
_MLA_QUERY_KERNEL_VERSION = 1
_MLA_QUERY_ROPE_DIM = 64


class _BMajor(IntEnum):
    """Contiguous physical axis of the right-hand operand."""

    K = 0
    N = 1


# ---------------------------------------------------------------------------
# Exact-value device conversions (this module's own primitives).
# ---------------------------------------------------------------------------


@dsl_user_op
def ld_shared_u16_zx(smem_addr: Int32, *, loc=None, ip=None) -> Uint32:
    """Load a shared halfword zero-extended into a 32-bit register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u16 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def packed_cvt_e4m3x4_to_bfloat2x2_exact(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Exact e4m3 -> bf16 for 4 packed bytes, sequential pairing.

    Returns (lo, hi): lo = bf16x2(byte0, byte1), hi = bf16x2(byte2, byte3),
    with byte N in the low half of its pair.  Chain: cvt.rn.f16x2.e4m3x2
    (exact: every e4m3 value incl. subnormals and +-0 is representable in
    f16) -> cvt.f32.f16 (exact) -> cvt.rn.bf16x2.f32 (exact for e4m3-derived
    values: 4-bit mantissa fits bf16's 8).  This is the true-value conversion
    torch's ``.to(torch.bfloat16)`` performs, unlike the biased fp4-path
    dequant helpers.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e01, e23, h0, h1, h2, h3;
            .reg .b32 p01, p23;
            .reg .f32 f0, f1, f2, f3;
            mov.b32 {e01, e23}, $2;
            cvt.rn.f16x2.e4m3x2 p01, e01;
            cvt.rn.f16x2.e4m3x2 p23, e23;
            mov.b32 {h0, h1}, p01;
            mov.b32 {h2, h3}, p23;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_cvt_e8m0x4_to_bfloat2x2_exact(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Exact e8m0 -> bf16 for 4 packed bytes, sequential pairing.

    byte e -> bf16(2^(e-127)): bit pattern e<<7 (exponent field = e, mantissa
    0) for e in [1, 254]; e == 0 -> the exact bf16 subnormal 2^-127 (0x0040);
    e == 255 -> bf16 NaN (0x7fff), matching CUDA's e8m0 conversion.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred z0, z1, z2, z3, n0, n1, n2, n3;
            .reg .b32 b0, b1, b2, b3, h0, h1, h2, h3, t;
            and.b32 b0, $2, 0x000000ff;
            shr.u32 b1, $2, 8;
            and.b32 b1, b1, 0x000000ff;
            shr.u32 b2, $2, 16;
            and.b32 b2, b2, 0x000000ff;
            shr.u32 b3, $2, 24;
            shl.b32 h0, b0, 7;
            shl.b32 h1, b1, 7;
            shl.b32 h2, b2, 7;
            shl.b32 h3, b3, 7;
            setp.eq.u32 z0, b0, 0;
            setp.eq.u32 z1, b1, 0;
            setp.eq.u32 z2, b2, 0;
            setp.eq.u32 z3, b3, 0;
            setp.eq.u32 n0, b0, 255;
            setp.eq.u32 n1, b1, 255;
            setp.eq.u32 n2, b2, 255;
            setp.eq.u32 n3, b3, 255;
            @z0 mov.b32 h0, 0x00000040;
            @z1 mov.b32 h1, 0x00000040;
            @z2 mov.b32 h2, 0x00000040;
            @z3 mov.b32 h3, 0x00000040;
            @n0 mov.b32 h0, 0x00007fff;
            @n1 mov.b32 h1, 0x00007fff;
            @n2 mov.b32 h2, 0x00007fff;
            @n3 mov.b32 h3, 0x00007fff;
            shl.b32 t, h1, 16;
            or.b32 $0, h0, t;
            shl.b32 t, h3, 16;
            or.b32 $1, h2, t;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def cvt_e8m0_byte_to_bfloat2_exact(byte: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert one e8m0 byte and broadcast it into both bf16x2 lanes."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(byte).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred pz, pn;
                .reg .b32 e, t, s;
                and.b32 e, $1, 0x000000ff;
                shl.b32 t, e, 7;
                setp.eq.u32 pz, e, 0;
                setp.eq.u32 pn, e, 255;
                @pz mov.b32 t, 0x00000040;
                @pn mov.b32 t, 0x00007fff;
                shl.b32 s, t, 16;
                or.b32 $0, s, t;
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_f32x2_to_bfloat2_rn(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack 2 f32 into bf16x2 with plain RN (no satfinite), x0 in the low half.

    Matches the cuBLAS/torch epilogue conversion semantics (overflow -> Inf),
    unlike the existing satfinite packer.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_u32(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load one naturally aligned uint32 from global memory."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_f32(base_ptr: Int64, *, loc=None, ip=None) -> Float32:
    """Load one naturally aligned float32 from global memory."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_u16(base_ptr: Int64, value: Uint32, *, loc=None, ip=None):
    """Store the low 16 bits of ``value`` to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u16 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def reciprocal_rn_f32(value: Float32, *, loc=None, ip=None) -> Float32:
    """Compute the IEEE round-to-nearest float32 reciprocal used by vLLM."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 one; mov.f32 one, 0f3f800000; div.rn.f32 $0, one, $1; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def scale_bfloat2_to_e4m3x2(
    packed: Uint32, inv_scale: Float32, *, loc=None, ip=None
) -> Uint32:
    """BF16 -> FP32 multiply -> saturated E4M3, matching static FP8 quant."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(packed).ir_value(loc=loc, ip=ip),
                Float32(inv_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 b0, b1, e;
                .reg .f32 f0, f1;
                mov.b32 {b0, b1}, $1;
                cvt.f32.bf16 f0, b0;
                cvt.f32.bf16 f1, b1;
                mul.rn.f32 f0, f0, $2;
                mul.rn.f32 f1, f1, $2;
                min.f32 f0, f0, 0f43e00000;
                max.f32 f0, f0, 0fc3e00000;
                min.f32 f1, f1, 0f43e00000;
                max.f32 f1, f1, 0fc3e00000;
                cvt.rn.satfinite.e4m3x2.f32 e, f1, f0;
                cvt.u32.u16 $0, e;
            }
            """,
            "=r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def round_f32x2_to_bf16_then_scale_e4m3x2(
    x0: Float32, x1: Float32, inv_scale: Float32, *, loc=None, ip=None
) -> Uint32:
    """Preserve the existing BF16 BMM epilogue before static FP8 quantization."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
                Float32(inv_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 packed;
                .reg .b16 b0, b1, e;
                .reg .f32 f0, f1;
                cvt.rn.bf16x2.f32 packed, $2, $1;
                mov.b32 {b0, b1}, packed;
                cvt.f32.bf16 f0, b0;
                cvt.f32.bf16 f1, b1;
                mul.rn.f32 f0, f0, $3;
                mul.rn.f32 f1, f1, $3;
                min.f32 f0, f0, 0f43e00000;
                max.f32 f0, f0, 0fc3e00000;
                min.f32 f1, f1, 0f43e00000;
                max.f32 f1, f1, 0fc3e00000;
                cvt.rn.satfinite.e4m3x2.f32 e, f1, f0;
                cvt.u32.u16 $0, e;
            }
            """,
            "=r,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# The kernel template.
# ---------------------------------------------------------------------------


class _Mxfp8Kernel:
    """One CTA per (batch item, n-tile); 128 threads; cp.async pipeline.

    Shared memory per stage (16B-aligned regions, no dynamic smem):
      [B tile: 8 KiB fp8 bytes] [A tile: tile_m*128 B bf16] [scales: 256 B]
    B-tile staging keeps the RHS's native physical orientation.  B-major N and
    B-major K differ only in which physical axis is the shared-memory row and
    in per-fragment scale indexing; neither performs a data transform.
    """

    def __init__(
        self,
        *,
        b_major: _BMajor | int,
        groups: int,
        m: int,
        n: int,
        k: int,
        tile_n: int = _TILE_N,
        tile_k: int = _TILE_K,
        stages: int = _STAGES,
    ):
        try:
            b_major = _BMajor(int(b_major))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported MXFP8 B-major axis {b_major!r}") from exc
        if int(groups) < 1:
            raise ValueError(f"BMM batch count must be positive, got {groups}")
        if not (1 <= int(m) <= _MAX_M):
            raise ValueError(f"MXFP8 BMM supports 1 <= M <= {_MAX_M}, got {m}")
        if tile_n != 128 or tile_k != 64:
            raise ValueError("MXFP8 BMM tile config requires tile_n=128, tile_k=64")
        if stages not in (2, 3):
            raise ValueError("stages must be 2 or 3")

        self.b_major = b_major
        self.b_major_n = b_major is _BMajor.N
        self.groups = int(groups)
        self.m = int(m)
        self.gemm_n = int(n)
        self.gemm_k = int(k)
        self.tile_n = int(tile_n)
        self.tile_k = int(tile_k)
        self.stages = int(stages)
        self.cta_threads = _CTA_THREADS

        if self.gemm_n % self.tile_n != 0:
            raise ValueError(f"gemm_n {self.gemm_n} not divisible by tile_n")
        if self.gemm_k % self.tile_k != 0:
            raise ValueError(f"gemm_k {self.gemm_k} not divisible by tile_k")

        self.n_tiles = self.gemm_n // self.tile_n
        self.k_tiles = self.gemm_k // self.tile_k
        self.k_steps = self.tile_k // 16  # MMA k16 steps per k-tile
        if self.stages - 1 > self.k_tiles:
            raise ValueError("stages-1 must not exceed k_tiles")
        self.grid_x = self.groups * self.n_tiles

        self.tile_m = 16 if self.m <= 16 else 32
        self.m_blocks = self.tile_m // 16
        self.n_warps = self.cta_threads // 32  # 4: each owns 32 N columns
        if self.n_warps * 32 != self.tile_n:
            raise ValueError("tile_n must equal 32 * n_warps")

        # RHS-tile shared-memory geometry in native physical orientation.
        # B-major K rows follow N; B-major N rows follow K.
        self.b_rows = self.tile_k if self.b_major_n else self.tile_n
        self.b_row_bytes = self.tile_n if self.b_major_n else self.tile_k
        self.b_units_per_row = self.b_row_bytes // 16  # int4 units
        self.b_swz_mask = min(self.b_units_per_row, 8) - 1
        self.b_stage_bytes = self.b_rows * self.b_row_bytes  # 8192 either way
        self.b_units = self.b_stage_bytes // 16  # 512

        # A tile: tile_m rows x tile_k bf16 cols = tile_m x 128 bytes.
        self.a_units_per_row = (self.tile_k * 2) // 16  # 8
        self.a_swz_mask = min(self.a_units_per_row, 8) - 1
        self.a_stage_bytes = self.tile_m * self.tile_k * 2
        self.a_units = self.a_stage_bytes // 16

        # Scale region: B-major K = tile_n rows x (tile_k/32) bytes;
        #               B-major N = tile_k rows x (tile_n/32) bytes.
        if not self.b_major_n:
            self.s_rows = self.tile_n
            self.s_row_bytes = self.tile_k // 32  # 2
        else:
            self.s_rows = self.tile_k
            self.s_row_bytes = self.tile_n // 32  # 4
        self.s_stage_bytes = self.s_rows * self.s_row_bytes  # 256 either way

        self.b_off = 0
        self.a_off = self.b_stage_bytes
        self.s_off = self.b_stage_bytes + self.a_stage_bytes
        stage_bytes = self.b_stage_bytes + self.a_stage_bytes + self.s_stage_bytes
        self.stage_bytes = (stage_bytes + 15) // 16 * 16
        self.smem_bytes = self.stages * self.stage_bytes
        if self.smem_bytes > 48 * 1024:
            raise ValueError(f"smem {self.smem_bytes} exceeds 48 KiB static limit")
        self.smem_words = self.smem_bytes // 4

        self.physical_rows = self.gemm_k if self.b_major_n else self.gemm_n
        self.physical_cols = self.gemm_n if self.b_major_n else self.gemm_k
        self.scale_cols = self.physical_cols // 32

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            int(self.b_major),
            self.groups,
            self.m,
            self.gemm_n,
            self.gemm_k,
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.stages,
            self.cta_threads,
        )

    # -- host-side launch entry (compiled once per cache key) ---------------

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,  # bf16 A [G, M, K]
        b_ptr: cute.Pointer,  # u8 values [G,K,N] or [G,N,K]
        s_ptr: cute.Pointer,  # u8 scales, physical inner axis / 32
        c_ptr: cute.Pointer,  # bf16 C [G, M, N]
        a_stride_g: cutlass.Int64,
        a_stride_m: cutlass.Int64,
        b_stride_g: cutlass.Int64,
        b_stride_row: cutlass.Int64,
        s_stride_g: cutlass.Int64,
        s_stride_row: cutlass.Int64,
        c_stride_g: cutlass.Int64,
        c_stride_m: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        a_span = (
            Int64(self.groups - 1) * a_stride_g
            + Int64(self.m - 1) * a_stride_m
            + Int64(self.gemm_k)
        )
        b_span = (
            Int64(self.groups - 1) * b_stride_g
            + Int64(self.physical_rows - 1) * b_stride_row
            + Int64(self.physical_cols)
        )
        s_span = (
            Int64(self.groups - 1) * s_stride_g
            + Int64(self.physical_rows - 1) * s_stride_row
            + Int64(self.scale_cols)
        )
        c_span = (
            Int64(self.groups - 1) * c_stride_g
            + Int64(self.m - 1) * c_stride_m
            + Int64(self.gemm_n)
        )
        a_flat = cute.make_tensor(
            a_ptr, layout=cute.make_layout((a_span,), stride=(1,))
        )
        c_flat = cute.make_tensor(
            c_ptr, layout=cute.make_layout((c_span,), stride=(1,))
        )
        b_flat = cute.make_tensor(
            b_ptr, layout=cute.make_layout((b_span,), stride=(1,))
        )
        s_flat = cute.make_tensor(
            s_ptr, layout=cute.make_layout((s_span,), stride=(1,))
        )
        self.kernel(
            a_flat,
            b_flat,
            s_flat,
            c_flat,
            a_stride_g,
            a_stride_m,
            b_stride_g,
            b_stride_row,
            s_stride_g,
            s_stride_row,
            c_stride_g,
            c_stride_m,
        ).launch(
            grid=(self.grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    # -- device code --------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        a_flat: cute.Tensor,
        b_flat: cute.Tensor,
        s_flat: cute.Tensor,
        c_flat: cute.Tensor,
        a_stride_g: cutlass.Int64,
        a_stride_m: cutlass.Int64,
        b_stride_g: cutlass.Int64,
        b_stride_row: cutlass.Int64,
        s_stride_g: cutlass.Int64,
        s_stride_row: cutlass.Int64,
        c_stride_g: cutlass.Int64,
        c_stride_m: cutlass.Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.smem_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        group = cta // Int32(self.n_tiles)
        jt = cta - group * Int32(self.n_tiles)

        acc = cute.make_rmem_tensor((self.m_blocks, 4, 4), cutlass.Float32)
        acc.fill(0.0)
        a_regs = cute.make_rmem_tensor((self.m_blocks, 4), Uint32)

        # Pipeline prologue: fill stages-1 buffers (one commit per stage).
        for s in cutlass.range_constexpr(self.stages - 1):
            self._stage_tile(
                a_flat,
                b_flat,
                s_flat,
                smem_base,
                tid,
                group,
                jt,
                Int32(s),
                Int32(s),
                a_stride_g,
                a_stride_m,
                b_stride_g,
                b_stride_row,
                s_stride_g,
                s_stride_row,
            )

        # Main loop: [wait][sync][consume][stage next or empty commit].
        # Exactly one commit per iteration keeps wait_group(stages-2) exact:
        # before the wait at iteration kt, stages-1+kt groups have been
        # committed and group kt must have landed once <= stages-2 remain.
        kt = Int32(0)
        while kt < Int32(self.k_tiles):
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
            slot = kt % Int32(self.stages)
            self._consume_tile(smem_base, tid, slot, acc, a_regs)
            nkt = kt + Int32(self.stages - 1)
            if nkt < Int32(self.k_tiles):
                self._stage_tile(
                    a_flat,
                    b_flat,
                    s_flat,
                    smem_base,
                    tid,
                    group,
                    jt,
                    nkt,
                    nkt % Int32(self.stages),
                    a_stride_g,
                    a_stride_m,
                    b_stride_g,
                    b_stride_row,
                    s_stride_g,
                    s_stride_row,
                )
            else:
                cute.arch.cp_async_commit_group()
            kt += Int32(1)

        self._store_output(c_flat, tid, group, jt, acc, c_stride_g, c_stride_m)

    @cute.jit
    def _stage_tile(
        self,
        a_flat: cute.Tensor,
        b_flat: cute.Tensor,
        s_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        group: Int32,
        jt: Int32,
        kt: Int32,
        slot: Int32,
        a_stride_g: Int64,
        a_stride_m: Int64,
        b_stride_g: Int64,
        b_stride_row: Int64,
        s_stride_g: Int64,
        s_stride_row: Int64,
    ):
        """Async-copy one K tile of the native RHS layout, A, and scales.

        Ends with cp_async_commit_group() -- exactly one group per call.
        XOR swizzles (unit_col ^= row & mask) are applied identically on the
        store side here and the read side in _consume_tile, so they are
        correctness-neutral by construction.
        """
        st_base = smem_base + slot * Int32(self.stage_bytes)
        group64 = Int64(group)
        jt64 = Int64(jt)
        kt64 = Int64(kt)

        # ---- B tile: 512 int4 units, 4 per thread. ----
        for i in cutlass.range_constexpr(self.b_units // self.cta_threads):
            u = Int32(i * self.cta_threads) + tid
            row = u // Int32(self.b_units_per_row)
            col = u - row * Int32(self.b_units_per_row)
            col_sw = col ^ (row & Int32(self.b_swz_mask))
            dst = (
                st_base
                + Int32(self.b_off)
                + (row * Int32(self.b_units_per_row) + col_sw) * Int32(16)
            )
            if cutlass.const_expr(not self.b_major_n):
                # B-major K: physical row is N; contiguous bytes are K.
                physical_row = jt64 * Int64(self.tile_n) + Int64(row)
                src = (
                    group64 * b_stride_g
                    + physical_row * b_stride_row
                    + kt64 * Int64(self.tile_k)
                    + Int64(col) * Int64(16)
                )
            else:
                # B-major N: physical row is K; contiguous bytes are N.
                physical_row = kt64 * Int64(self.tile_k) + Int64(row)
                src = (
                    group64 * b_stride_g
                    + physical_row * b_stride_row
                    + jt64 * Int64(self.tile_n)
                    + Int64(col) * Int64(16)
                )
            cp_async4_shared_global(dst, get_ptr_as_int64(b_flat, src))

        # ---- A tile: tile_m rows x tile_k bf16 = a_units int4 units. ----
        for i in cutlass.range_constexpr(self.a_units // self.cta_threads):
            u = Int32(i * self.cta_threads) + tid
            row = u // Int32(self.a_units_per_row)
            col = u - row * Int32(self.a_units_per_row)
            col_sw = col ^ (row & Int32(self.a_swz_mask))
            dst = (
                st_base
                + Int32(self.a_off)
                + (row * Int32(self.a_units_per_row) + col_sw) * Int32(16)
            )
            if cutlass.const_expr(self.m < self.tile_m):
                if row < Int32(self.m):
                    src = (
                        group64 * a_stride_g
                        + Int64(row) * a_stride_m
                        + kt64 * Int64(self.tile_k)
                        + Int64(col) * Int64(8)
                    )
                    cp_async4_shared_global(dst, get_ptr_as_int64(a_flat, src))
                else:
                    st_shared_v4_u32(dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            else:
                src = (
                    group64 * a_stride_g
                    + Int64(row) * a_stride_m
                    + kt64 * Int64(self.tile_k)
                    + Int64(col) * Int64(8)
                )
                cp_async4_shared_global(dst, get_ptr_as_int64(a_flat, src))

        # ---- Scales. ----
        if cutlass.const_expr(not self.b_major_n):
            # B-major K: one u16 per N row for two K scale groups.
            # 2-byte granularity is below cp.async's 4B minimum, so this is a
            # synchronous ld.global + st.shared (visibility ordered by the
            # consumer-side syncthreads).
            if tid < Int32(self.s_rows):
                scale_row = jt64 * Int64(self.tile_n) + Int64(tid)
                src = (
                    group64 * s_stride_g
                    + scale_row * s_stride_row
                    + kt64 * Int64(self.tile_k // 32)
                )
                v = ld_global_b16(get_ptr_as_int64(s_flat, src))
                st_shared_u16(st_base + Int32(self.s_off) + tid * Int32(2), v)
        else:
            # B-major N: one u32 per K row for four N scale groups.
            if tid < Int32(self.s_rows):
                scale_row = kt64 * Int64(self.tile_k) + Int64(tid)
                src = (
                    group64 * s_stride_g
                    + scale_row * s_stride_row
                    + jt64 * Int64(self.tile_n // 32)
                )
                cp_async_u32_shared_global(
                    st_base + Int32(self.s_off) + tid * Int32(4),
                    get_ptr_as_int64(s_flat, src),
                )

        cute.arch.cp_async_commit_group()

    @cute.jit
    def _consume_tile(
        self,
        smem_base: Int32,
        tid: Int32,
        slot: Int32,
        acc: cute.Tensor,
        a_regs: cute.Tensor,
    ):
        """Dequant + MMA over one staged k-tile (4 k16 steps).

        m16n8k16 fragment mapping (canonical PTX, lane = 4*g + t, g = lane>>2,
        t = lane&3):
          A: a0=(row g, k 2t..2t+1) a1=(g+8, same) a2=(g, k+8) a3=(g+8, k+8)
          B: b0=(k 2t..2t+1, n g)   b1=(k 2t+8..2t+9, n g)
          C: c0=(g, 2t) c1=(g, 2t+1) c2=(g+8, 2t) c3=(g+8, 2t+1)
        Each warp owns N columns [warp*32, warp*32+32) of the CTA tile as 4
        jj-blocks of 8.
        """
        st_base = smem_base + slot * Int32(self.stage_bytes)
        lane = tid & Int32(31)
        warp = tid // Int32(32)
        g = lane // Int32(4)
        t = lane - g * Int32(4)
        t2 = t * Int32(2)

        a_base = st_base + Int32(self.a_off)
        b_base = st_base + Int32(self.b_off)
        s_base = st_base + Int32(self.s_off)

        # ldmatrix lane addressing: row = lane&15 (+16 per m-block), int4
        # column half = lane>>4, same xor swizzle as the store side.
        lm_row_in_blk = lane & Int32(15)
        lm_half = lane // Int32(16)

        for kk in cutlass.range_constexpr(self.k_steps):
            # A fragments for every m-block (shared across the 4 jj blocks).
            for mb in cutlass.range_constexpr(self.m_blocks):
                lm_row = Int32(16 * mb) + lm_row_in_blk
                lm_col = Int32(2 * kk) + lm_half
                lm_col_sw = lm_col ^ (lm_row & Int32(self.a_swz_mask))
                a_addr = a_base + (
                    lm_row * Int32(self.a_units_per_row) + lm_col_sw
                ) * Int32(16)
                r0, r1, r2, r3 = ldmatrix_m8n8x4_b16(a_addr)
                a_regs[mb, 0] = r0
                a_regs[mb, 1] = r1
                a_regs[mb, 2] = r2
                a_regs[mb, 3] = r3

            if cutlass.const_expr(self.b_major_n):
                # Per-K-row scales, shared by all jj (l-group = this warp's
                # 32-wide N window).  k rows: kk*16 + {2t, 2t+1, 2t+8, 2t+9}.
                kd0 = Int32(16 * kk) + t2
                s0 = ld_shared_u8_offset(
                    s_base + kd0 * Int32(self.s_row_bytes) + warp, 0
                )
                s1 = ld_shared_u8_offset(
                    s_base + (kd0 + Int32(1)) * Int32(self.s_row_bytes) + warp, 0
                )
                s2 = ld_shared_u8_offset(
                    s_base + (kd0 + Int32(8)) * Int32(self.s_row_bytes) + warp, 0
                )
                s3 = ld_shared_u8_offset(
                    s_base + (kd0 + Int32(9)) * Int32(self.s_row_bytes) + warp, 0
                )
                s_word = (
                    s0 + (s1 << Uint32(8)) + (s2 << Uint32(16)) + (s3 << Uint32(24))
                )
                sc_lo, sc_hi = packed_cvt_e8m0x4_to_bfloat2x2_exact(s_word)

            for jj in cutlass.range_constexpr(4):
                n_local = warp * Int32(32) + Int32(8 * jj) + g

                if cutlass.const_expr(not self.b_major_n):
                    # B-major K values: row n_local, K bytes contiguous.
                    unit = Int32(kk) ^ (n_local & Int32(self.b_swz_mask))
                    v_addr = (
                        b_base
                        + (n_local * Int32(self.b_units_per_row) + unit) * Int32(16)
                        + t2
                    )
                    v_lo16 = ld_shared_u16_zx(v_addr)
                    v_hi16 = ld_shared_u16_zx(v_addr + Int32(8))
                    v_word = v_lo16 + (v_hi16 << Uint32(16))
                    v_lo, v_hi = packed_cvt_e4m3x4_to_bfloat2x2_exact(v_word)
                    # One e8m0 group covers the whole k16 step: byte kk>>1 of
                    # this row's staged u16.
                    s_byte = ld_shared_u8_offset(
                        s_base + n_local * Int32(2) + Int32(kk // 2), 0
                    )
                    sc = cvt_e8m0_byte_to_bfloat2_exact(s_byte)
                    b0 = bfloat2_mul(v_lo, sc)
                    b1 = bfloat2_mul(v_hi, sc)
                else:
                    # B-major N values: K rows kk*16 + {2t,2t+1,2t+8,2t+9} (= kd0
                    # from the scale hoist above), byte column n_local;
                    # per-row xor swizzle on the int4 unit.
                    n_unit = n_local // Int32(16)
                    n_in_unit = n_local & Int32(15)
                    b0v = self._kn_value_byte(b_base, kd0, n_unit, n_in_unit)
                    b1v = self._kn_value_byte(b_base, kd0 + Int32(1), n_unit, n_in_unit)
                    b2v = self._kn_value_byte(b_base, kd0 + Int32(8), n_unit, n_in_unit)
                    b3v = self._kn_value_byte(b_base, kd0 + Int32(9), n_unit, n_in_unit)
                    v_word = (
                        b0v
                        + (b1v << Uint32(8))
                        + (b2v << Uint32(16))
                        + (b3v << Uint32(24))
                    )
                    v_lo, v_hi = packed_cvt_e4m3x4_to_bfloat2x2_exact(v_word)
                    b0 = bfloat2_mul(v_lo, sc_lo)
                    b1 = bfloat2_mul(v_hi, sc_hi)

                for mb in cutlass.range_constexpr(self.m_blocks):
                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                        acc[mb, jj, 0],
                        acc[mb, jj, 1],
                        acc[mb, jj, 2],
                        acc[mb, jj, 3],
                        a_regs[mb, 0],
                        a_regs[mb, 1],
                        a_regs[mb, 2],
                        a_regs[mb, 3],
                        b0,
                        b1,
                    )
                    acc[mb, jj, 0] = d0
                    acc[mb, jj, 1] = d1
                    acc[mb, jj, 2] = d2
                    acc[mb, jj, 3] = d3

    @cute.jit
    def _kn_value_byte(
        self, b_base: Int32, kd: Int32, n_unit: Int32, n_in_unit: Int32
    ) -> Uint32:
        unit_sw = n_unit ^ (kd & Int32(self.b_swz_mask))
        addr = (
            b_base
            + (kd * Int32(self.b_units_per_row) + unit_sw) * Int32(16)
            + n_in_unit
        )
        return ld_shared_u8_offset(addr, 0)

    @cute.jit
    def _store_output(
        self,
        c_flat: cute.Tensor,
        tid: Int32,
        group: Int32,
        jt: Int32,
        acc: cute.Tensor,
        c_stride_g: Int64,
        c_stride_m: Int64,
    ):
        """f32 acc -> bf16 pairs -> u32 stores into the strided C view.

        C rows beyond M are never written; with compile-time M the
        predicates fold away for full tiles.  Column pairs are contiguous (C
        inner dim contiguous by contract) so every store is one aligned u32.
        """
        lane = tid & Int32(31)
        warp = tid // Int32(32)
        g = lane // Int32(4)
        t = lane - g * Int32(4)
        col0 = jt * Int32(self.tile_n) + warp * Int32(32) + t * Int32(2)
        c_group = Int64(group) * c_stride_g

        for mb in cutlass.range_constexpr(self.m_blocks):
            row0 = Int32(16 * mb) + g
            row1 = row0 + Int32(8)
            for jj in cutlass.range_constexpr(4):
                col = col0 + Int32(8 * jj)
                if row0 < Int32(self.m):
                    off = c_group + Int64(row0) * c_stride_m + Int64(col)
                    st_global_u32(
                        get_ptr_as_int64(c_flat, off),
                        pack_f32x2_to_bfloat2_rn(acc[mb, jj, 0], acc[mb, jj, 1]),
                    )
                if row1 < Int32(self.m):
                    off = c_group + Int64(row1) * c_stride_m + Int64(col)
                    st_global_u32(
                        get_ptr_as_int64(c_flat, off),
                        pack_f32x2_to_bfloat2_rn(acc[mb, jj, 2], acc[mb, jj, 3]),
                    )


class _Mxfp8MlaQueryKernel(_Mxfp8Kernel):
    """MXFP8 absorbed-query BMM with a query-assembly epilogue.

    The GEMM accumulator is rounded through BF16 exactly as in ``bmm``.  The
    BF16 output mode then writes that result and the existing RoPE query into a
    caller-owned token-major ``[M,H,576]`` workspace.  The FP8 mode additionally
    applies vLLM's static per-tensor E4M3 quantization in the same launch.
    """

    def __init__(self, *, output_fp8: bool, **kwargs):
        super().__init__(**kwargs)
        self.output_fp8 = bool(output_fp8)

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (*super().__cache_key__, self.output_fp8)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,  # bf16 A [H, M, K]
        b_ptr: cute.Pointer,  # u8 values [H,K,N]
        s_ptr: cute.Pointer,  # u8 scales [H,K,N/32]
        q_pe_ptr: cute.Pointer,  # bf16 RoPE query [M,H,64]
        q_scale_ptr: cute.Pointer,  # f32 scalar; ignored for BF16 output
        out_ptr: cute.Pointer,  # bf16/fp8 assembled query [M,H,N+64]
        a_stride_g: cutlass.Int64,
        a_stride_m: cutlass.Int64,
        b_stride_g: cutlass.Int64,
        b_stride_row: cutlass.Int64,
        s_stride_g: cutlass.Int64,
        s_stride_row: cutlass.Int64,
        q_pe_stride_m: cutlass.Int64,
        q_pe_stride_g: cutlass.Int64,
        out_stride_m: cutlass.Int64,
        out_stride_g: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        a_span = (
            Int64(self.groups - 1) * a_stride_g
            + Int64(self.m - 1) * a_stride_m
            + Int64(self.gemm_k)
        )
        b_span = (
            Int64(self.groups - 1) * b_stride_g
            + Int64(self.physical_rows - 1) * b_stride_row
            + Int64(self.physical_cols)
        )
        s_span = (
            Int64(self.groups - 1) * s_stride_g
            + Int64(self.physical_rows - 1) * s_stride_row
            + Int64(self.scale_cols)
        )
        q_pe_span = (
            Int64(self.m - 1) * q_pe_stride_m
            + Int64(self.groups - 1) * q_pe_stride_g
            + Int64(_MLA_QUERY_ROPE_DIM)
        )
        out_span = (
            Int64(self.m - 1) * out_stride_m
            + Int64(self.groups - 1) * out_stride_g
            + Int64(self.gemm_n + _MLA_QUERY_ROPE_DIM)
        )
        a_flat = cute.make_tensor(
            a_ptr, layout=cute.make_layout((a_span,), stride=(1,))
        )
        b_flat = cute.make_tensor(
            b_ptr, layout=cute.make_layout((b_span,), stride=(1,))
        )
        s_flat = cute.make_tensor(
            s_ptr, layout=cute.make_layout((s_span,), stride=(1,))
        )
        q_pe_flat = cute.make_tensor(
            q_pe_ptr, layout=cute.make_layout((q_pe_span,), stride=(1,))
        )
        q_scale = cute.make_tensor(
            q_scale_ptr, layout=cute.make_layout((1,), stride=(1,))
        )
        out_flat = cute.make_tensor(
            out_ptr, layout=cute.make_layout((out_span,), stride=(1,))
        )
        self.kernel_mla_query(
            a_flat,
            b_flat,
            s_flat,
            q_pe_flat,
            q_scale,
            out_flat,
            a_stride_g,
            a_stride_m,
            b_stride_g,
            b_stride_row,
            s_stride_g,
            s_stride_row,
            q_pe_stride_m,
            q_pe_stride_g,
            out_stride_m,
            out_stride_g,
        ).launch(
            grid=(self.grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel_mla_query(
        self,
        a_flat: cute.Tensor,
        b_flat: cute.Tensor,
        s_flat: cute.Tensor,
        q_pe_flat: cute.Tensor,
        q_scale: cute.Tensor,
        out_flat: cute.Tensor,
        a_stride_g: cutlass.Int64,
        a_stride_m: cutlass.Int64,
        b_stride_g: cutlass.Int64,
        b_stride_row: cutlass.Int64,
        s_stride_g: cutlass.Int64,
        s_stride_row: cutlass.Int64,
        q_pe_stride_m: cutlass.Int64,
        q_pe_stride_g: cutlass.Int64,
        out_stride_m: cutlass.Int64,
        out_stride_g: cutlass.Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.smem_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        group = cta // Int32(self.n_tiles)
        jt = cta - group * Int32(self.n_tiles)

        acc = cute.make_rmem_tensor((self.m_blocks, 4, 4), cutlass.Float32)
        acc.fill(0.0)
        a_regs = cute.make_rmem_tensor((self.m_blocks, 4), Uint32)

        for stage in cutlass.range_constexpr(self.stages - 1):
            self._stage_tile(
                a_flat,
                b_flat,
                s_flat,
                smem_base,
                tid,
                group,
                jt,
                Int32(stage),
                Int32(stage),
                a_stride_g,
                a_stride_m,
                b_stride_g,
                b_stride_row,
                s_stride_g,
                s_stride_row,
            )

        kt = Int32(0)
        while kt < Int32(self.k_tiles):
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
            slot = kt % Int32(self.stages)
            self._consume_tile(smem_base, tid, slot, acc, a_regs)
            next_kt = kt + Int32(self.stages - 1)
            if next_kt < Int32(self.k_tiles):
                self._stage_tile(
                    a_flat,
                    b_flat,
                    s_flat,
                    smem_base,
                    tid,
                    group,
                    jt,
                    next_kt,
                    next_kt % Int32(self.stages),
                    a_stride_g,
                    a_stride_m,
                    b_stride_g,
                    b_stride_row,
                    s_stride_g,
                    s_stride_row,
                )
            else:
                cute.arch.cp_async_commit_group()
            kt += Int32(1)

        self._store_mla_query_output(
            q_pe_flat,
            q_scale,
            out_flat,
            tid,
            group,
            jt,
            acc,
            q_pe_stride_m,
            q_pe_stride_g,
            out_stride_m,
            out_stride_g,
        )

    @cute.jit
    def _store_mla_query_output(
        self,
        q_pe_flat: cute.Tensor,
        q_scale: cute.Tensor,
        out_flat: cute.Tensor,
        tid: Int32,
        group: Int32,
        jt: Int32,
        acc: cute.Tensor,
        q_pe_stride_m: Int64,
        q_pe_stride_g: Int64,
        out_stride_m: Int64,
        out_stride_g: Int64,
    ):
        lane = tid & Int32(31)
        warp = tid // Int32(32)
        lane_group = lane // Int32(4)
        lane_in_group = lane - lane_group * Int32(4)
        col0 = jt * Int32(self.tile_n) + warp * Int32(32) + lane_in_group * Int32(2)
        out_group = Int64(group) * out_stride_g

        inv_scale = Float32(1.0)
        if cutlass.const_expr(self.output_fp8):
            inv_scale = reciprocal_rn_f32(
                ld_global_f32(get_ptr_as_int64(q_scale, Int64(0)))
            )

        for mb in cutlass.range_constexpr(self.m_blocks):
            row0 = Int32(16 * mb) + lane_group
            row1 = row0 + Int32(8)
            for jj in cutlass.range_constexpr(4):
                col = col0 + Int32(8 * jj)
                if row0 < Int32(self.m):
                    off = Int64(row0) * out_stride_m + out_group + Int64(col)
                    if cutlass.const_expr(self.output_fp8):
                        st_global_u16(
                            get_ptr_as_int64(out_flat, off),
                            round_f32x2_to_bf16_then_scale_e4m3x2(
                                acc[mb, jj, 0], acc[mb, jj, 1], inv_scale
                            ),
                        )
                    else:
                        st_global_u32(
                            get_ptr_as_int64(out_flat, off),
                            pack_f32x2_to_bfloat2_rn(acc[mb, jj, 0], acc[mb, jj, 1]),
                        )
                if row1 < Int32(self.m):
                    off = Int64(row1) * out_stride_m + out_group + Int64(col)
                    if cutlass.const_expr(self.output_fp8):
                        st_global_u16(
                            get_ptr_as_int64(out_flat, off),
                            round_f32x2_to_bf16_then_scale_e4m3x2(
                                acc[mb, jj, 2], acc[mb, jj, 3], inv_scale
                            ),
                        )
                    else:
                        st_global_u32(
                            get_ptr_as_int64(out_flat, off),
                            pack_f32x2_to_bfloat2_rn(acc[mb, jj, 2], acc[mb, jj, 3]),
                        )

        # Exactly one N tile copies the 64-d RoPE suffix for this head.
        if jt == Int32(0):
            pair_count = self.m * (_MLA_QUERY_ROPE_DIM // 2)
            for iteration in cutlass.range_constexpr(
                (pair_count + self.cta_threads - 1) // self.cta_threads
            ):
                pair_idx = Int32(iteration * self.cta_threads) + tid
                if pair_idx < Int32(pair_count):
                    row = pair_idx // Int32(_MLA_QUERY_ROPE_DIM // 2)
                    pair = pair_idx - row * Int32(_MLA_QUERY_ROPE_DIM // 2)
                    q_off = (
                        Int64(row) * q_pe_stride_m
                        + Int64(group) * q_pe_stride_g
                        + Int64(pair) * Int64(2)
                    )
                    out_off = (
                        Int64(row) * out_stride_m
                        + out_group
                        + Int64(self.gemm_n)
                        + Int64(pair) * Int64(2)
                    )
                    packed = ld_global_u32(get_ptr_as_int64(q_pe_flat, q_off))
                    if cutlass.const_expr(self.output_fp8):
                        st_global_u16(
                            get_ptr_as_int64(out_flat, out_off),
                            scale_bfloat2_to_e4m3x2(packed, inv_scale),
                        )
                    else:
                        st_global_u32(get_ptr_as_int64(out_flat, out_off), packed)


# ---------------------------------------------------------------------------
# Host-side compile cache + launch.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Mxfp8Launch:
    compiled: object
    kernel: _Mxfp8Kernel


_LAUNCH_CACHE: dict[tuple[object, ...], _Mxfp8Launch] = {}
_MLA_QUERY_LAUNCH_CACHE: dict[tuple[object, ...], _Mxfp8Launch] = {}
register_program_cache(_LAUNCH_CACHE)
register_program_cache(_MLA_QUERY_LAUNCH_CACHE)
_QUALIFIED_BATCHES = frozenset((8, 16))
_QUALIFIED_GEOMETRIES = {
    _BMajor.N: (192, 512),
    _BMajor.K: (512, 256),
}


def _tile_m_for_m(m: int) -> int:
    return 16 if int(m) <= 16 else 32


def _compile(
    *,
    b_major: _BMajor,
    groups: int,
    m: int,
    n: int,
    k: int,
    device: torch.device,
) -> _Mxfp8Launch:
    device_index = int(device.index if device.index is not None else 0)
    tile_m = _tile_m_for_m(m)
    cache_key = (
        _KERNEL_ID,
        int(b_major),
        int(groups),
        int(m),
        int(n),
        int(k),
        device_index,
        (tile_m, _TILE_N, _TILE_K, _STAGES, _CTA_THREADS),
    )
    cached = _LAUNCH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "BMM MXFP8 compile miss during CUDA-graph capture for "
            f"b_major={b_major.name.lower()}, B={groups}, M={m}, N={n}, K={k}. "
            "Precompile every graph-visible M with prewarm(..., m_values=...) "
            "before capture."
        )

    kernel = _Mxfp8Kernel(
        b_major=b_major,
        groups=groups,
        m=m,
        n=n,
        k=k,
        tile_n=_TILE_N,
        tile_k=_TILE_K,
        stages=_STAGES,
    )

    def dummy(dt, align=16):
        return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=align)

    # b12x_compile guards frozen resolution on true cache misses itself
    # (memory/disk hits bypass the freeze -- warm boots stay legal).
    # Pointer alignment is part of the compiled signature.  A and B values
    # feed 16-byte cp.async loads; scales need 4-byte alignment for B-major N
    # and C needs 4-byte alignment for paired BF16 stores.
    compiled = b12x_compile(
        kernel,
        dummy(cutlass.BFloat16),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8, align=4),
        dummy(cutlass.BFloat16, align=4),
        Int64(kernel.gemm_k),
        Int64(kernel.gemm_k),
        Int64(kernel.physical_cols),
        Int64(kernel.physical_cols),
        Int64(kernel.scale_cols),
        Int64(kernel.scale_cols),
        Int64(kernel.gemm_n),
        Int64(kernel.gemm_n),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_facts(
            _KERNEL_ID,
            _KERNEL_VERSION,
            ("device_index", device_index),
            ("a_dtype", "bfloat16"),
            ("b_dtype", "float8_e4m3fn"),
            ("sf_dtype", "float8_e8m0fnu"),
            ("c_dtype", "bfloat16"),
            ("sf_vec_size", 32),
            ("b_major", b_major.name.lower()),
            ("sf_axis", b_major.name.lower()),
            ("groups", int(groups)),
            ("m", int(m)),
            ("n", int(n)),
            ("k", int(k)),
            ("tile_m", int(tile_m)),
            ("tile_n", int(_TILE_N)),
            ("tile_k", int(_TILE_K)),
            ("stages", int(_STAGES)),
            ("cta_threads", int(_CTA_THREADS)),
            ("grid_x", int(kernel.grid_x)),
        ),
    )
    launch = _Mxfp8Launch(compiled=compiled, kernel=kernel)
    attach_programs(launch, compiled)
    _LAUNCH_CACHE[cache_key] = launch
    return launch


def _compile_mla_query_projection(
    *,
    b_major: _BMajor,
    groups: int,
    m: int,
    n: int,
    k: int,
    output_fp8: bool,
    device: torch.device,
) -> _Mxfp8Launch:
    device_index = int(device.index if device.index is not None else 0)
    tile_m = _tile_m_for_m(m)
    cache_key = (
        _MLA_QUERY_KERNEL_ID,
        int(b_major),
        int(groups),
        int(m),
        int(n),
        int(k),
        bool(output_fp8),
        device_index,
        (tile_m, _TILE_N, _TILE_K, _STAGES, _CTA_THREADS),
    )
    cached = _MLA_QUERY_LAUNCH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MLA query compile miss during CUDA-graph capture for "
            f"B={groups}, M={m}, N={n}, K={k}, output_fp8={output_fp8}. "
            "Prepare every graph-visible M before capture."
        )

    kernel = _Mxfp8MlaQueryKernel(
        output_fp8=output_fp8,
        b_major=b_major,
        groups=groups,
        m=m,
        n=n,
        k=k,
        tile_n=_TILE_N,
        tile_k=_TILE_K,
        stages=_STAGES,
    )

    def dummy(dt, align=16):
        return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=align)

    output_dtype = cutlass.Float8E4M3FN if output_fp8 else cutlass.BFloat16
    output_align = 2 if output_fp8 else 4
    compiled = b12x_compile(
        kernel,
        dummy(cutlass.BFloat16),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8, align=4),
        dummy(cutlass.BFloat16, align=4),
        dummy(cutlass.Float32, align=4),
        dummy(output_dtype, align=output_align),
        Int64(kernel.gemm_k),
        Int64(kernel.gemm_k),
        Int64(kernel.physical_cols),
        Int64(kernel.physical_cols),
        Int64(kernel.scale_cols),
        Int64(kernel.scale_cols),
        Int64(kernel.groups * _MLA_QUERY_ROPE_DIM),
        Int64(_MLA_QUERY_ROPE_DIM),
        Int64(kernel.groups * (kernel.gemm_n + _MLA_QUERY_ROPE_DIM)),
        Int64(kernel.gemm_n + _MLA_QUERY_ROPE_DIM),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_facts(
            _MLA_QUERY_KERNEL_ID,
            _MLA_QUERY_KERNEL_VERSION,
            ("device_index", device_index),
            ("a_dtype", "bfloat16"),
            ("b_dtype", "float8_e4m3fn"),
            ("sf_dtype", "float8_e8m0fnu"),
            ("output_dtype", "float8_e4m3fn" if output_fp8 else "bfloat16"),
            ("b_major", b_major.name.lower()),
            ("groups", int(groups)),
            ("m", int(m)),
            ("n", int(n)),
            ("k", int(k)),
            ("rope_dim", int(_MLA_QUERY_ROPE_DIM)),
            ("tile_m", int(tile_m)),
            ("tile_n", int(_TILE_N)),
            ("tile_k", int(_TILE_K)),
            ("stages", int(_STAGES)),
            ("cta_threads", int(_CTA_THREADS)),
            ("grid_x", int(kernel.grid_x)),
        ),
    )
    launch = _Mxfp8Launch(compiled=compiled, kernel=kernel)
    attach_programs(launch, compiled)
    _MLA_QUERY_LAUNCH_CACHE[cache_key] = launch
    return launch


def _check_operand(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    stride_mod: int,
    ptr_align: int,
) -> None:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.stride(2) != 1:
        raise ValueError(f"{name} inner dim must be contiguous")
    if tensor.stride(0) <= 0 or tensor.stride(1) <= 0:
        raise ValueError(
            f"{name} outer strides must be positive, got {tensor.stride()}"
        )
    if tensor.stride(0) % stride_mod or tensor.stride(1) % stride_mod:
        raise ValueError(
            f"{name} outer strides {tensor.stride(0)}/{tensor.stride(1)} must be "
            f"multiples of {stride_mod} elements"
        )
    if tensor.data_ptr() % ptr_align:
        raise ValueError(f"{name} base pointer must be {ptr_align}-byte aligned")


def _has_internal_overlap(tensor: torch.Tensor) -> bool:
    required_span = 1
    dimensions = sorted(
        zip(tensor.shape, tensor.stride(), strict=True), key=lambda item: item[1]
    )
    for size, stride in dimensions:
        size = int(size)
        stride = int(stride)
        if size <= 1:
            continue
        if stride < required_span:
            return True
        required_span += (size - 1) * stride
    return False


def _coerce_b_major(b_major: str | _BMajor | int) -> _BMajor:
    if isinstance(b_major, str):
        try:
            return {"k": _BMajor.K, "n": _BMajor.N}[b_major.lower()]
        except KeyError as exc:
            raise ValueError(f"b_major must be 'k' or 'n', got {b_major!r}") from exc
    try:
        return _BMajor(int(b_major))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"b_major must be 'k' or 'n', got {b_major!r}") from exc


def _rhs_tensors(
    rhs: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(rhs, tuple) or len(rhs) != 2:
        raise TypeError("rhs must be a (values, scales) tensor tuple")
    values, scales = rhs
    if not isinstance(values, torch.Tensor) or not isinstance(scales, torch.Tensor):
        raise TypeError("rhs must be a (values, scales) tensor tuple")
    return values, scales


def _rhs_geometry(
    values: torch.Tensor,
    scales: torch.Tensor,
    *,
    b_major: str | _BMajor | int,
    sf_axis: str,
) -> tuple[_BMajor, int, int, int]:
    major = _coerce_b_major(b_major)
    if not isinstance(sf_axis, str) or sf_axis.lower() != major.name.lower():
        raise NotImplementedError(
            "the BMM MXFP8 specialization requires sf_axis to match "
            f"b_major, got b_major={major.name.lower()!r}, sf_axis={sf_axis!r}"
        )
    if values.ndim != 3:
        raise ValueError(f"rhs values must be 3-D, got shape {tuple(values.shape)}")
    if major is _BMajor.N:
        groups, k, n = map(int, values.shape)
        scale_shape = (groups, k, n // 32)
    else:
        groups, n, k = map(int, values.shape)
        scale_shape = (groups, n, k // 32)
    if n <= 0 or k <= 0:
        raise ValueError(f"rhs dimensions must be positive, got N={n}, K={k}")
    if int(values.shape[-1]) % 32:
        raise ValueError(
            "rhs physical inner dimension must be divisible by sf_vec_size=32, "
            f"got {values.shape[-1]}"
        )
    if tuple(scales.shape) != scale_shape:
        raise ValueError(
            f"rhs scales must have shape {scale_shape} for b_major="
            f"{major.name.lower()!r}, sf_axis={sf_axis!r}; got "
            f"{tuple(scales.shape)}"
        )
    return major, groups, n, k


def _geometry_is_qualified(
    *, b_major: _BMajor, groups: int, m: int, n: int, k: int
) -> bool:
    return (
        1 <= int(m) <= _MAX_M
        and int(groups) in _QUALIFIED_BATCHES
        and (int(k), int(n)) == _QUALIFIED_GEOMETRIES[b_major]
    )


def _check_rhs_tensor(name: str, tensor: torch.Tensor, *, ptr_align: int) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be 3-D, got shape {tuple(tensor.shape)}")
    if tensor.stride(2) != 1:
        raise ValueError(f"{name} physical inner dimension must be contiguous")
    if tensor.stride(0) <= 0 or tensor.stride(1) <= 0:
        raise ValueError(
            f"{name} outer strides must be positive, got {tensor.stride()}"
        )
    if tensor.stride(0) % ptr_align or tensor.stride(1) % ptr_align:
        raise ValueError(
            f"{name} outer strides {tensor.stride(0)}/{tensor.stride(1)} must "
            f"be multiples of {ptr_align} elements"
        )
    if tensor.data_ptr() % ptr_align:
        raise ValueError(f"{name} base pointer must be {ptr_align}-byte aligned")


def _validate_rhs_storage(b_values: torch.Tensor, b_scales: torch.Tensor) -> None:
    if b_values.dtype != torch.float8_e4m3fn:
        raise ValueError(f"rhs values must be float8_e4m3fn, got {b_values.dtype}")
    if b_scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError(f"rhs scales must be uint8/e8m0, got {b_scales.dtype}")
    if not b_values.is_cuda or not b_scales.is_cuda:
        raise ValueError("BMM RHS tensors must be CUDA tensors")
    if b_values.device != b_scales.device:
        raise ValueError("BMM RHS tensors must be on the same CUDA device")
    _check_rhs_tensor("rhs values", b_values, ptr_align=16)
    _check_rhs_tensor("rhs scales", b_scales, ptr_align=4)


def _storage_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    extent = 1
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if int(size) > 0:
            extent += (int(size) - 1) * int(stride)
    return start, start + extent * int(tensor.element_size())


def _overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.device != rhs.device:
        return False
    lhs_start, lhs_end = _storage_interval(lhs)
    rhs_start, rhs_end = _storage_interval(rhs)
    return max(lhs_start, rhs_start) < min(lhs_end, rhs_end)


def _validate_launch(
    a: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    out: torch.Tensor,
    *,
    b_major: str | _BMajor | int,
    sf_axis: str,
) -> tuple[_BMajor, int, int, int, int]:
    major, b_groups, n, b_k = _rhs_geometry(
        b_values, b_scales, b_major=b_major, sf_axis=sf_axis
    )
    if a.ndim != 3:
        raise ValueError(f"lhs must have shape [B,M,K], got {tuple(a.shape)}")
    groups, m, k = map(int, a.shape)
    if groups != b_groups or k != b_k:
        raise ValueError(
            "lhs and rhs batch/K dimensions must match: "
            f"lhs is [B,M,K]=[{groups},{m},{k}], rhs implies "
            f"B={b_groups}, K={b_k}"
        )
    _check_operand("lhs", a, shape=(groups, m, k), stride_mod=8, ptr_align=16)
    _check_operand("out", out, shape=(groups, m, n), stride_mod=2, ptr_align=4)
    if _has_internal_overlap(out):
        raise ValueError("out must not have internal storage overlap")
    _validate_rhs_storage(b_values, b_scales)
    tensors = (a, b_values, b_scales, out)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("BMM operands must be CUDA tensors")
    if any(tensor.device != a.device for tensor in tensors[1:]):
        raise ValueError("BMM operands must be on the same CUDA device")
    if not _geometry_is_qualified(b_major=major, groups=groups, m=m, n=n, k=k):
        raise NotImplementedError(
            "the BMM MXFP8 specialization is qualified for b_major='n' "
            "[B,K,N]=[16,192,512] or b_major='k' "
            "[B,N,K]=[16,256,512], with 1<=M<=32; "
            f"got b_major={major.name.lower()!r}, B={groups}, M={m}, "
            f"N={n}, K={k}"
        )
    for source_name, source in (
        ("lhs", a),
        ("rhs values", b_values),
        ("rhs scales", b_scales),
    ):
        if _overlaps(out, source):
            raise ValueError(f"out must not overlap {source_name}")
    return major, groups, m, n, k


def _run(
    a: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    out: torch.Tensor,
    *,
    b_major: str | _BMajor | int,
    sf_axis: str,
    stream: Optional[int] = None,
) -> None:
    """Validate and launch the allocation-free backend."""
    major, groups, m, n, k = _validate_launch(
        a,
        b_values,
        b_scales,
        out,
        b_major=b_major,
        sf_axis=sf_axis,
    )
    launch = _compile(b_major=major, groups=groups, m=m, n=n, k=k, device=a.device)
    _launch_prepared(a, b_values, b_scales, out, launch=launch, stream=stream)


def _launch_prepared(
    a: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    out: torch.Tensor,
    *,
    launch: _Mxfp8Launch,
    stream: Optional[int] = None,
) -> None:
    """Launch an already selected and compiled BMM specialization."""

    if stream is not None:
        launch_stream = (
            torch.cuda.default_stream(a.device)
            if int(stream) == 0
            else torch.cuda.ExternalStream(int(stream), device=a.device)
        )
        for tensor in (a, b_values, b_scales, out):
            tensor.record_stream(launch_stream)
    stream_int = (
        int(stream)
        if stream is not None
        else torch.cuda.current_stream(a.device).cuda_stream
    )
    launch.compiled(
        make_ptr(
            cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            cutlass.Uint8,
            b_values.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint8,
            b_scales.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        ),
        Int64(int(a.stride(0))),
        Int64(int(a.stride(1))),
        Int64(int(b_values.stride(0))),
        Int64(int(b_values.stride(1))),
        Int64(int(b_scales.stride(0))),
        Int64(int(b_scales.stride(1))),
        Int64(int(out.stride(0))),
        Int64(int(out.stride(1))),
        cuda.CUstream(stream_int),
    )


def _run_prepared(
    a: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    out: torch.Tensor,
    *,
    b_major: str | _BMajor | int,
    sf_axis: str,
    launch: _Mxfp8Launch,
    stream: Optional[object] = None,
) -> torch.Tensor:
    _validate_launch(a, b_values, b_scales, out, b_major=b_major, sf_axis=sf_axis)
    stream_int = (
        None if stream is None else int(stream)
        if isinstance(stream, int) else int(_torch_stream(stream, a.device).cuda_stream)
    )
    _launch_prepared(a, b_values, b_scales, out, launch=launch, stream=stream_int)
    return out


def _check_mla_query_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    ptr_align: int,
    stride_mod: int,
) -> None:
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.stride(2) != 1:
        raise ValueError(f"{name} inner dim must be contiguous")
    if tensor.stride(0) <= 0 or tensor.stride(1) <= 0:
        raise ValueError(
            f"{name} outer strides must be positive, got {tensor.stride()}"
        )
    if tensor.stride(0) % stride_mod or tensor.stride(1) % stride_mod:
        raise ValueError(
            f"{name} outer strides {tensor.stride(0)}/{tensor.stride(1)} must "
            f"be multiples of {stride_mod} elements"
        )
    if tensor.data_ptr() % ptr_align:
        raise ValueError(f"{name} base pointer must be {ptr_align}-byte aligned")
    if _has_internal_overlap(tensor):
        raise ValueError(f"{name} must not have internal storage overlap")


def _validate_mla_query_launch(
    a: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    b_major: str | _BMajor | int,
    sf_axis: str,
) -> tuple[_BMajor, int, int, int, int, bool]:
    major, b_groups, n, b_k = _rhs_geometry(
        b_values, b_scales, b_major=b_major, sf_axis=sf_axis
    )
    if major is not _BMajor.N:
        raise NotImplementedError(
            "the fused MLA query epilogue requires b_major='n', "
            f"got {major.name.lower()!r}"
        )
    if a.ndim != 3:
        raise ValueError(f"lhs must have shape [H,M,K], got {tuple(a.shape)}")
    groups, m, k = map(int, a.shape)
    if groups != b_groups or k != b_k:
        raise ValueError(
            "lhs and rhs head/K dimensions must match: "
            f"lhs is [H,M,K]=[{groups},{m},{k}], rhs implies "
            f"H={b_groups}, K={b_k}"
        )
    if n != 512:
        raise NotImplementedError(
            f"the fused MLA query epilogue requires N=512, got {n}"
        )

    output_fp8 = out.dtype == torch.float8_e4m3fn
    if not output_fp8 and out.dtype != torch.bfloat16:
        raise ValueError(
            f"assembled query output must be bfloat16 or float8_e4m3fn, got {out.dtype}"
        )
    _check_operand("lhs", a, shape=(groups, m, k), stride_mod=8, ptr_align=16)
    _check_mla_query_tensor(
        "q_pe",
        q_pe,
        shape=(m, groups, _MLA_QUERY_ROPE_DIM),
        dtype=torch.bfloat16,
        ptr_align=4,
        stride_mod=2,
    )
    _check_mla_query_tensor(
        "out",
        out,
        shape=(m, groups, n + _MLA_QUERY_ROPE_DIM),
        dtype=out.dtype,
        ptr_align=2 if output_fp8 else 4,
        stride_mod=2,
    )
    _validate_rhs_storage(b_values, b_scales)
    if output_fp8:
        if q_scale is None:
            raise ValueError("q_scale is required for float8_e4m3fn output")
        if q_scale.dtype != torch.float32 or q_scale.numel() != 1:
            raise ValueError(
                "q_scale must be a scalar float32 tensor, "
                f"got shape={tuple(q_scale.shape)}, dtype={q_scale.dtype}"
            )
        if q_scale.data_ptr() % 4:
            raise ValueError("q_scale base pointer must be 4-byte aligned")

    tensors = [a, b_values, b_scales, q_pe, out]
    if output_fp8:
        assert q_scale is not None
        tensors.append(q_scale)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("MLA query operands must be CUDA tensors")
    if any(tensor.device != a.device for tensor in tensors[1:]):
        raise ValueError("MLA query operands must be on the same CUDA device")
    if not _geometry_is_qualified(b_major=major, groups=groups, m=m, n=n, k=k):
        raise NotImplementedError(
            "the fused MLA query specialization is qualified for "
            "[H,K,N]=[8|16,192,512] with 1<=M<=32; "
            f"got H={groups}, M={m}, N={n}, K={k}"
        )
    sources = [
        ("lhs", a),
        ("rhs values", b_values),
        ("rhs scales", b_scales),
        ("q_pe", q_pe),
    ]
    if output_fp8:
        assert q_scale is not None
        sources.append(("q_scale", q_scale))
    for source_name, source in sources:
        if _overlaps(out, source):
            raise ValueError(f"out must not overlap {source_name}")
    return major, groups, m, n, k, output_fp8


def _run_mla_query_projection(
    a: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    b_major: str | _BMajor | int,
    sf_axis: str,
    stream: Optional[int] = None,
) -> None:
    major, groups, m, n, k, output_fp8 = _validate_mla_query_launch(
        a,
        b_values,
        b_scales,
        q_pe,
        q_scale,
        out,
        b_major=b_major,
        sf_axis=sf_axis,
    )
    launch = _compile_mla_query_projection(
        b_major=major, groups=groups, m=m, n=n, k=k, output_fp8=output_fp8, device=a.device,
    )
    _launch_mla_query_prepared(a, b_values, b_scales, q_pe, q_scale, out,
                               launch=launch, output_fp8=output_fp8, stream=stream)


def _launch_mla_query_prepared(
    a: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    launch: _Mxfp8Launch,
    output_fp8: bool,
    stream: Optional[int] = None,
) -> None:

    """Launch an already selected fused MLA query specialization."""

    if stream is not None:
        launch_stream = (
            torch.cuda.default_stream(a.device)
            if int(stream) == 0
            else torch.cuda.ExternalStream(int(stream), device=a.device)
        )
        tensors = [a, b_values, b_scales, q_pe, out]
        if q_scale is not None:
            tensors.append(q_scale)
        for tensor in tensors:
            tensor.record_stream(launch_stream)
    stream_int = (
        int(stream)
        if stream is not None
        else torch.cuda.current_stream(a.device).cuda_stream
    )
    output_dtype = cutlass.Float8E4M3FN if output_fp8 else cutlass.BFloat16
    launch.compiled(
        make_ptr(
            cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            cutlass.Uint8,
            b_values.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint8,
            b_scales.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.BFloat16,
            q_pe.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Float32,
            q_scale.data_ptr() if q_scale is not None else out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            output_dtype,
            out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=2 if output_fp8 else 4,
        ),
        Int64(int(a.stride(0))),
        Int64(int(a.stride(1))),
        Int64(int(b_values.stride(0))),
        Int64(int(b_values.stride(1))),
        Int64(int(b_scales.stride(0))),
        Int64(int(b_scales.stride(1))),
        Int64(int(q_pe.stride(0))),
        Int64(int(q_pe.stride(1))),
        Int64(int(out.stride(0))),
        Int64(int(out.stride(1))),
        cuda.CUstream(stream_int),
    )


def _run_mla_query_prepared(
    a: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    b_major: str | _BMajor | int,
    sf_axis: str,
    launch: _Mxfp8Launch,
    stream: Optional[object] = None,
) -> torch.Tensor:
    _, _, _, _, _, output_fp8 = _validate_mla_query_launch(
        a, b_values, b_scales, q_pe, q_scale, out, b_major=b_major, sf_axis=sf_axis,
    )
    stream_int = (
        None if stream is None else int(stream)
        if isinstance(stream, int) else int(_torch_stream(stream, a.device).cuda_stream)
    )
    _launch_mla_query_prepared(a, b_values, b_scales, q_pe, q_scale, out,
                               launch=launch, output_fp8=output_fp8, stream=stream_int)
    return out


@torch.library.custom_op("b12x::bmm_mxfp8", mutates_args=("out",))
def _op(
    lhs: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    out: torch.Tensor,
    b_major: int,
    plan_handle: int,
    stream_int: Optional[int] = None,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "gemm.bmm", lhs.device)
    major = _coerce_b_major(b_major).name.lower()
    state.run(lhs, (b_values, b_scales), out, b_major=major, sf_axis=major, stream=stream_int)


@_op.register_fake
def _fake(
    lhs: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    out: torch.Tensor,
    b_major: int,
    plan_handle: int,
    stream_int: Optional[int] = None,
) -> None:
    del lhs, b_values, b_scales, out, b_major, plan_handle, stream_int


@torch.library.custom_op(
    "b12x::mla_query_projection_mxfp8", mutates_args=("out",)
)
def _mla_query_projection_op(
    lhs: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
    b_major: int,
    plan_handle: int,
    stream_int: Optional[int] = None,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "gemm.mla_query_projection", lhs.device)
    del b_major
    state.run(lhs, (b_values, b_scales), q_pe, out, q_scale=q_scale, stream=stream_int)


@_mla_query_projection_op.register_fake
def _mla_query_projection_fake(
    lhs: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
    b_major: int,
    plan_handle: int,
    stream_int: Optional[int] = None,
) -> None:
    del lhs, b_values, b_scales, q_pe, q_scale, out, b_major, plan_handle, stream_int



def _stream_to_int(stream: Optional[object]) -> Optional[int]:
    if stream is None:
        return None
    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is not None:
        return int(cuda_stream)
    return int(stream)


def _torch_stream(stream: Optional[object], device: torch.device) -> torch.cuda.Stream:
    device_index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    if stream is None:
        target = torch.cuda.current_stream(device)
    elif isinstance(stream, torch.cuda.Stream):
        target = stream
    else:
        stream_int = _stream_to_int(stream)
        if stream_int == 0:
            target = torch.cuda.default_stream(device)
        else:
            error, stream_device = cudart.cudaStreamGetDevice(
                cudart.cudaStream_t(stream_int)
            )
            if error != cudart.cudaError_t.cudaSuccess:
                raise ValueError(f"invalid CUDA stream handle: {error.name}")
            if int(stream_device) != device_index:
                raise ValueError(
                    f"stream is on cuda:{stream_device}, but BMM operands are on "
                    f"cuda:{device_index}"
                )
            target = torch.cuda.ExternalStream(stream_int, device=device)
    if target.device.index != device_index:
        raise ValueError(
            f"stream is on {target.device}, but BMM operands are on cuda:{device_index}"
        )
    return target


# ---------------------------------------------------------------------------
# Precompile every caller-declared graph-visible M before capture.
# ---------------------------------------------------------------------------



def can_implement(
    *,
    batch: int,
    max_m: int,
    n: int,
    k: int,
    b_major: str,
    sf_axis: str,
) -> bool:
    """Return whether the qualified MXFP8 backend covers a geometry."""
    try:
        major = _coerce_b_major(b_major)
    except ValueError:
        return False
    if not isinstance(sf_axis, str) or sf_axis.lower() != major.name.lower():
        return False
    return _geometry_is_qualified(
        b_major=major,
        groups=int(batch),
        m=int(max_m),
        n=int(n),
        k=int(k),
    )


def can_implement_mla_query_projection(
    *,
    batch: int,
    max_m: int,
    n: int,
    k: int,
    output_dtype: torch.dtype,
    b_major: str = "n",
    sf_axis: str = "n",
) -> bool:
    """Return whether the fused absorbed-query epilogue covers a geometry."""
    try:
        major = _coerce_b_major(b_major)
    except ValueError:
        return False
    return (
        major is _BMajor.N
        and isinstance(sf_axis, str)
        and sf_axis.lower() == "n"
        and int(n) == 512
        and output_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and _geometry_is_qualified(
            b_major=major,
            groups=int(batch),
            m=int(max_m),
            n=int(n),
            k=int(k),
        )
    )


def clear_caches() -> None:
    _LAUNCH_CACHE.clear()


def clear_mla_query_projection_caches() -> None:
    _MLA_QUERY_LAUNCH_CACHE.clear()


__all__ = [
    "can_implement",
    "can_implement_mla_query_projection",
    "clear_caches",
    "clear_mla_query_projection_caches",
]
