"""CuTe DSL device primitives for the CE-driven PCIe DMA collective.

The copy engine itself is driven from Python through ``cudaMemcpyAsync``.
This module owns only the small SM kernels that publish/wait on peer flags,
add lossless ring payloads, and encode/decode the compressed wire formats.

All byte/element offsets are widened to ``Int64`` before pointer arithmetic.
The protocol operations use inline PTX deliberately: their system scope and
ordering are part of the collective ABI, not compiler hints.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as b12x_compile,
)
from b12x._lib.intrinsics import (
    cvt_e4m3x4_to_f32x4,
    cvt_f32x4_to_e4m3x4,
    div_rn_f32,
    fabs_f32,
    f16x2_to_f32x2,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_v4_f32,
    ld_global_v4_u32,
    pack_f32x2_to_half2,
    st_global_v4_f32,
    st_global_v4_u32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


_THREADS = 256
_MAX_GRID = 64
_QUANT_BLOCK = 128
_WARPS_PER_CTA = _THREADS // 32
_MX_SCALES_PER_BLOCK = 4
_FP8_MAX = 448.0
_INT8_MAX = 127.0
_CODECS = ("e4m3", "i8", "mx")
_DTYPES = ("bf16", "fp16", "fp32")


def _one_tensor(ptr: cute.Pointer) -> cute.Tensor:
    return cute.make_tensor(ptr, cute.make_layout((1,)))


@dsl_user_op
def _ld_global_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
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
def _ld_global_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
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
def _ld_global_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u8 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_u32(addr: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_f32(addr: Int64, value: Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_u8(addr: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u8 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_global_v2_u32(
    addr: Int64, *, loc=None, ip=None
) -> tuple[Uint32, Uint32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _st_global_v2_u32(
    addr: Int64, v0: Uint32, v1: Uint32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _publish_flag_sys(
    peer_flag_addr: Int64, counter_addr: Int64, *, loc=None, ip=None
) -> None:
    """Increment the local counter and publish it with native CUDA semantics."""

    llvm.inline_asm(
        None,
        [
            Int64(peer_flag_addr).ir_value(loc=loc, ip=ip),
            Int64(counter_addr).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .u32 value;
            ld.global.u32 value, [$1];
            add.u32 value, value, 1;
            st.global.u32 [$1], value;
            fence.sc.sys;
            st.relaxed.sys.global.u32 [$0], value;
        }
        """,
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _wait_flag_sys(
    flag_addr: Int64, counter_addr: Int64, *, loc=None, ip=None
) -> None:
    """Increment the expected value and wait with acquire-system loads."""

    llvm.inline_asm(
        None,
        [
            Int64(flag_addr).ir_value(loc=loc, ip=ip),
            Int64(counter_addr).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred pending;
            .reg .u32 expected, observed;
            .reg .s32 delta;
            ld.global.u32 expected, [$1];
            add.u32 expected, expected, 1;
            st.global.u32 [$1], expected;
        dma_wait_again:
            ld.acquire.sys.global.u32 observed, [$0];
            sub.s32 delta, observed, expected;
            setp.lt.s32 pending, delta, 0;
            @pending bra dma_wait_again;
        }
        """,
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _unpack_bf16x2(
    packed: Uint32, *, loc=None, ip=None
) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.bf16 $0, lo;
            cvt.f32.bf16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _pack_bf16x2(
    v0: Float32, v1: Float32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
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
def _pack_i8x4(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .s32 q0, q1, q2, q3;
                .reg .u32 u0, u1, u2, u3;
                cvt.rni.s32.f32 q0, $1;
                cvt.rni.s32.f32 q1, $2;
                cvt.rni.s32.f32 q2, $3;
                cvt.rni.s32.f32 q3, $4;
                max.s32 q0, q0, -127; min.s32 q0, q0, 127;
                max.s32 q1, q1, -127; min.s32 q1, q1, 127;
                max.s32 q2, q2, -127; min.s32 q2, q2, 127;
                max.s32 q3, q3, -127; min.s32 q3, q3, 127;
                and.b32 u0, q0, 255;
                and.b32 u1, q1, 255; shl.b32 u1, u1, 8;
                and.b32 u2, q2, 255; shl.b32 u2, u2, 16;
                and.b32 u3, q3, 255; shl.b32 u3, u3, 24;
                or.b32 u0, u0, u1;
                or.b32 u2, u2, u3;
                or.b32 $0, u0, u2;
            }
            """,
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _unpack_i8x4(
    packed: Uint32, *, loc=None, ip=None
) -> tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .s32 q0, q1, q2, q3;
            bfe.s32 q0, $4, 0, 8;
            bfe.s32 q1, $4, 8, 8;
            bfe.s32 q2, $4, 16, 8;
            bfe.s32 q3, $4, 24, 8;
            cvt.rn.f32.s32 $0, q0;
            cvt.rn.f32.s32 $1, q1;
            cvt.rn.f32.s32 $2, q2;
            cvt.rn.f32.s32 $3, q3;
        }
        """,
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [idx], loc=loc, ip=ip))
        for idx in range(4)
    )


@dsl_user_op
def _mx_scale_byte(amax: Float32, *, loc=None, ip=None) -> Uint32:
    """Match ``__nv_cvt_float_to_e8m0(..., cudaRoundPosInf)``."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(amax).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred is_zero, is_underflow, is_inf, is_nan, has_mantissa;
                .reg .u32 original, bits, mantissa, byte;
                mov.b32 original, $1;
                and.b32 original, original, 0x7fffffff;
                setp.eq.u32 is_zero, original, 0;
                setp.eq.u32 is_inf, original, 0x7f800000;
                setp.gt.u32 is_nan, original, 0x7f800000;
                mul.rn.f32 $1, $1, 0f3B124925;
                mov.b32 bits, $1;
                and.b32 mantissa, bits, 0x007fffff;
                setp.ne.u32 has_mantissa, mantissa, 0;
                setp.le.u32 is_underflow, bits, 0x00400000;
                @has_mantissa add.u32 bits, bits, 0x00800000;
                bfe.u32 byte, bits, 23, 8;
                @is_underflow mov.u32 byte, 0;
                @is_zero mov.u32 byte, 127;
                @is_inf mov.u32 byte, 254;
                @is_nan mov.u32 byte, 255;
                mov.u32 $0, byte;
            }
            """,
            "=r,+f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mx_scale(scale_byte: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(scale_byte).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred is_zero, is_nan;
                .reg .u32 bits;
                setp.eq.u32 is_zero, $1, 0;
                setp.eq.u32 is_nan, $1, 255;
                shl.b32 bits, $1, 23;
                @is_zero mov.u32 bits, 0x00400000;
                @is_nan mov.u32 bits, 0x7fffffff;
                mov.b32 $0, bits;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mx_inv_scale(scale_byte: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(scale_byte).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred is_min, is_nan;
                .reg .u32 exponent, bits;
                setp.eq.u32 is_min, $1, 254;
                setp.eq.u32 is_nan, $1, 255;
                sub.u32 exponent, 254, $1;
                shl.b32 bits, exponent, 23;
                @is_min mov.u32 bits, 0x00400000;
                @is_nan mov.u32 bits, 0x7fffffff;
                mov.b32 $0, bits;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _load_bf16x4(
    src: cute.Tensor, elem0: Int64
) -> tuple[Float32, Float32, Float32, Float32]:
    lo, hi = _ld_global_v2_u32(get_ptr_as_int64(src, elem0))
    v0, v1 = _unpack_bf16x2(lo)
    v2, v3 = _unpack_bf16x2(hi)
    return v0, v1, v2, v3


@cute.jit
def _store_bf16x4(
    dst: cute.Tensor,
    elem0: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
) -> None:
    _st_global_v2_u32(
        get_ptr_as_int64(dst, elem0),
        _pack_bf16x2(v0, v1),
        _pack_bf16x2(v2, v3),
    )


@cute.jit
def _warp_amax_32(
    v0: Float32, v1: Float32, v2: Float32, v3: Float32
) -> Float32:
    amax = fmax_f32(
        fmax_f32(fabs_f32(v0), fabs_f32(v1)),
        fmax_f32(fabs_f32(v2), fabs_f32(v3)),
    )
    for offset in cutlass.range_constexpr(5):
        amax = fmax_f32(
            amax, cute.arch.shuffle_sync_bfly(amax, offset=16 >> offset)
        )
    return amax


@cute.jit
def _warp_amax_8(
    v0: Float32, v1: Float32, v2: Float32, v3: Float32
) -> Float32:
    amax = fmax_f32(
        fmax_f32(fabs_f32(v0), fabs_f32(v1)),
        fmax_f32(fabs_f32(v2), fabs_f32(v3)),
    )
    for offset in cutlass.range_constexpr(3):
        amax = fmax_f32(
            amax,
            cute.arch.shuffle_sync_bfly(
                # CUDA's ``width=8`` encodes both the 8-lane clamp and the
                # segment mask in the PTX ``c`` operand: ((32-8)<<8)|7.
                amax, offset=4 >> offset, mask_and_clamp=0x1807
            ),
        )
    return amax


class _FlagLaunch:
    def __init__(self, wait: bool) -> None:
        self._wait = bool(wait)

    @cute.jit
    def __call__(
        self,
        flag_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(_one_tensor(flag_ptr), _one_tensor(counter_ptr)).launch(
            grid=(1, 1, 1),
            block=(1, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, flag: cute.Tensor, counter: cute.Tensor) -> None:
        flag_addr = get_ptr_as_int64(flag, Int64(0))
        counter_addr = get_ptr_as_int64(counter, Int64(0))
        if cutlass.const_expr(self._wait):
            _wait_flag_sys(flag_addr, counter_addr)
        else:
            _publish_flag_sys(flag_addr, counter_addr)


class _AddLaunch:
    def __init__(self, dtype_name: str) -> None:
        self._dtype_name = dtype_name

    @cute.jit
    def __call__(
        self,
        dst_ptr: cute.Pointer,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        packs: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _one_tensor(dst_ptr),
            _one_tensor(a_ptr),
            _one_tensor(b_ptr),
            packs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dst: cute.Tensor,
        a_src: cute.Tensor,
        b_src: cute.Tensor,
        packs: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        idx = Int64(bidx) * Int64(_THREADS) + Int64(tidx)
        step = Int64(gdim) * Int64(_THREADS)
        while idx < packs:
            if cutlass.const_expr(self._dtype_name == "fp32"):
                elem0 = idx * Int64(4)
                av = ld_global_v4_f32(get_ptr_as_int64(a_src, elem0))
                bv = ld_global_v4_f32(get_ptr_as_int64(b_src, elem0))
                st_global_v4_f32(
                    get_ptr_as_int64(dst, elem0),
                    av[0] + bv[0],
                    av[1] + bv[1],
                    av[2] + bv[2],
                    av[3] + bv[3],
                )
            else:
                elem0 = idx * Int64(8)
                aw = ld_global_v4_u32(get_ptr_as_int64(a_src, elem0))
                bw = ld_global_v4_u32(get_ptr_as_int64(b_src, elem0))
                out = cute.make_rmem_tensor((4,), Uint32)
                for word in cutlass.range_constexpr(4):
                    if cutlass.const_expr(self._dtype_name == "bf16"):
                        a0, a1 = _unpack_bf16x2(aw[word])
                        b0, b1 = _unpack_bf16x2(bw[word])
                        out[word] = _pack_bf16x2(a0 + b0, a1 + b1)
                    else:
                        a0, a1 = f16x2_to_f32x2(aw[word])
                        b0, b1 = f16x2_to_f32x2(bw[word])
                        out[word] = pack_f32x2_to_half2(a0 + b0, a1 + b1)
                st_global_v4_u32(
                    get_ptr_as_int64(dst, elem0),
                    out[0],
                    out[1],
                    out[2],
                    out[3],
                )
            idx += step


class _QuantLaunch:
    def __init__(self, codec: str) -> None:
        self._codec = codec

    @cute.jit
    def __call__(
        self,
        src_ptr: cute.Pointer,
        payload_ptr: cute.Pointer,
        scales_ptr: cute.Pointer,
        blocks: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _one_tensor(src_ptr),
            _one_tensor(payload_ptr),
            _one_tensor(scales_ptr),
            blocks,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        blocks: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = Int32(tidx) & Int32(31)
        warp = Int32(tidx) >> Int32(5)
        block = Int64(bidx) * Int64(_WARPS_PER_CTA) + Int64(warp)
        step = Int64(gdim) * Int64(_WARPS_PER_CTA)
        while block < blocks:
            elem0 = block * Int64(_QUANT_BLOCK) + Int64(lane) * Int64(4)
            v0, v1, v2, v3 = _load_bf16x4(src, elem0)
            if cutlass.const_expr(self._codec == "mx"):
                amax = _warp_amax_8(v0, v1, v2, v3)
                scale_byte = _mx_scale_byte(amax)
                scale_group = lane >> Int32(3)
                scale_lane = lane & Int32(7)
                if scale_lane == Int32(0):
                    scale_idx = block * Int64(_MX_SCALES_PER_BLOCK) + Int64(
                        scale_group
                    )
                    _st_global_u8(get_ptr_as_int64(scales, scale_idx), scale_byte)
                inv_scale = _mx_inv_scale(scale_byte)
                packed = cvt_f32x4_to_e4m3x4(
                    v0 * inv_scale,
                    v1 * inv_scale,
                    v2 * inv_scale,
                    v3 * inv_scale,
                )
            else:
                amax = _warp_amax_32(v0, v1, v2, v3)
                maximum = _INT8_MAX if self._codec == "i8" else _FP8_MAX
                scale = div_rn_f32(amax, Float32(maximum))
                # Match the native ``amax > 0 ? amax / maximum : 1``
                # predicate exactly.  In particular, unordered (NaN) and
                # non-positive blocks use the unit scale instead of
                # propagating a NaN scale into the wire representation.
                if not (amax > Float32(0.0)):
                    scale = Float32(1.0)
                if lane == Int32(0):
                    _st_global_f32(get_ptr_as_int64(scales, block), scale)
                inv_scale = div_rn_f32(Float32(1.0), scale)
                if cutlass.const_expr(self._codec == "i8"):
                    packed = _pack_i8x4(
                        v0 * inv_scale,
                        v1 * inv_scale,
                        v2 * inv_scale,
                        v3 * inv_scale,
                    )
                else:
                    packed = cvt_f32x4_to_e4m3x4(
                        v0 * inv_scale,
                        v1 * inv_scale,
                        v2 * inv_scale,
                        v3 * inv_scale,
                    )
            _st_global_u32(get_ptr_as_int64(payload, elem0 // Int64(4)), packed)
            block += step


@cute.jit
def _decode_payload(
    codec: cutlass.Constexpr,
    payload: cute.Tensor,
    scales: cute.Tensor,
    block: Int64,
    lane: Int32,
) -> tuple[Float32, Float32, Float32, Float32]:
    elem0 = block * Int64(_QUANT_BLOCK) + Int64(lane) * Int64(4)
    packed = _ld_global_u32(get_ptr_as_int64(payload, elem0 // Int64(4)))
    if cutlass.const_expr(codec == "i8"):
        v0, v1, v2, v3 = _unpack_i8x4(packed)
        scale = _ld_global_f32(get_ptr_as_int64(scales, block))
    elif cutlass.const_expr(codec == "mx"):
        v0, v1, v2, v3 = cvt_e4m3x4_to_f32x4(packed)
        scale_group = lane >> Int32(3)
        scale_idx = block * Int64(_MX_SCALES_PER_BLOCK) + Int64(scale_group)
        scale = _mx_scale(_ld_global_u8(get_ptr_as_int64(scales, scale_idx)))
    else:
        v0, v1, v2, v3 = cvt_e4m3x4_to_f32x4(packed)
        scale = _ld_global_f32(get_ptr_as_int64(scales, block))
    return v0 * scale, v1 * scale, v2 * scale, v3 * scale


class _DequantStoreLaunch:
    def __init__(self, codec: str) -> None:
        self._codec = codec

    @cute.jit
    def __call__(
        self,
        out_ptr: cute.Pointer,
        payload_ptr: cute.Pointer,
        scales_ptr: cute.Pointer,
        blocks: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _one_tensor(out_ptr),
            _one_tensor(payload_ptr),
            _one_tensor(scales_ptr),
            blocks,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        blocks: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = Int32(tidx) & Int32(31)
        warp = Int32(tidx) >> Int32(5)
        block = Int64(bidx) * Int64(_WARPS_PER_CTA) + Int64(warp)
        step = Int64(gdim) * Int64(_WARPS_PER_CTA)
        while block < blocks:
            v0, v1, v2, v3 = _decode_payload(
                self._codec, payload, scales, block, lane
            )
            elem0 = block * Int64(_QUANT_BLOCK) + Int64(lane) * Int64(4)
            _store_bf16x4(out, elem0, v0, v1, v2, v3)
            block += step


class _DequantAccumLaunch:
    def __init__(self, codec: str, nsrc: int) -> None:
        self._codec = codec
        self._nsrc = int(nsrc)

    @cute.jit
    def __call__(
        self,
        out_ptr: cute.Pointer,
        inp_ptr: cute.Pointer,
        p0: cute.Pointer,
        p1: cute.Pointer,
        p2: cute.Pointer,
        p3: cute.Pointer,
        p4: cute.Pointer,
        p5: cute.Pointer,
        p6: cute.Pointer,
        p7: cute.Pointer,
        s0: cute.Pointer,
        s1: cute.Pointer,
        s2: cute.Pointer,
        s3: cute.Pointer,
        s4: cute.Pointer,
        s5: cute.Pointer,
        s6: cute.Pointer,
        s7: cute.Pointer,
        blocks: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _one_tensor(out_ptr),
            _one_tensor(inp_ptr),
            _one_tensor(p0),
            _one_tensor(p1),
            _one_tensor(p2),
            _one_tensor(p3),
            _one_tensor(p4),
            _one_tensor(p5),
            _one_tensor(p6),
            _one_tensor(p7),
            _one_tensor(s0),
            _one_tensor(s1),
            _one_tensor(s2),
            _one_tensor(s3),
            _one_tensor(s4),
            _one_tensor(s5),
            _one_tensor(s6),
            _one_tensor(s7),
            blocks,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,
        inp: cute.Tensor,
        p0: cute.Tensor,
        p1: cute.Tensor,
        p2: cute.Tensor,
        p3: cute.Tensor,
        p4: cute.Tensor,
        p5: cute.Tensor,
        p6: cute.Tensor,
        p7: cute.Tensor,
        s0: cute.Tensor,
        s1: cute.Tensor,
        s2: cute.Tensor,
        s3: cute.Tensor,
        s4: cute.Tensor,
        s5: cute.Tensor,
        s6: cute.Tensor,
        s7: cute.Tensor,
        blocks: Int64,
    ) -> None:
        payloads = (p0, p1, p2, p3, p4, p5, p6, p7)
        scales = (s0, s1, s2, s3, s4, s5, s6, s7)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = Int32(tidx) & Int32(31)
        warp = Int32(tidx) >> Int32(5)
        block = Int64(bidx) * Int64(_WARPS_PER_CTA) + Int64(warp)
        step = Int64(gdim) * Int64(_WARPS_PER_CTA)
        while block < blocks:
            elem0 = block * Int64(_QUANT_BLOCK) + Int64(lane) * Int64(4)
            a0, a1, a2, a3 = _load_bf16x4(inp, elem0)
            for source in cutlass.range_constexpr(self._nsrc):
                v0, v1, v2, v3 = _decode_payload(
                    self._codec,
                    payloads[source],
                    scales[source],
                    block,
                    lane,
                )
                a0 += v0
                a1 += v1
                a2 += v2
                a3 += v3
            _store_bf16x4(out, elem0, a0, a1, a2, a3)
            block += step


class _DequantAccum9Launch:
    """TP10-only ABI with one pointer pair for every remote source.

    Keep this separate from ``_DequantAccumLaunch``: its eight payload and
    eight scale arguments are the frozen ABI for the existing 1..8-source
    compile keys.  Widening that launcher would perturb every smaller-world
    object solely to make room for TP10's ninth peer.
    """

    def __init__(self, codec: str) -> None:
        self._codec = codec

    @cute.jit
    def __call__(
        self,
        out_ptr: cute.Pointer,
        inp_ptr: cute.Pointer,
        p0: cute.Pointer,
        p1: cute.Pointer,
        p2: cute.Pointer,
        p3: cute.Pointer,
        p4: cute.Pointer,
        p5: cute.Pointer,
        p6: cute.Pointer,
        p7: cute.Pointer,
        p8: cute.Pointer,
        s0: cute.Pointer,
        s1: cute.Pointer,
        s2: cute.Pointer,
        s3: cute.Pointer,
        s4: cute.Pointer,
        s5: cute.Pointer,
        s6: cute.Pointer,
        s7: cute.Pointer,
        s8: cute.Pointer,
        blocks: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _one_tensor(out_ptr),
            _one_tensor(inp_ptr),
            _one_tensor(p0),
            _one_tensor(p1),
            _one_tensor(p2),
            _one_tensor(p3),
            _one_tensor(p4),
            _one_tensor(p5),
            _one_tensor(p6),
            _one_tensor(p7),
            _one_tensor(p8),
            _one_tensor(s0),
            _one_tensor(s1),
            _one_tensor(s2),
            _one_tensor(s3),
            _one_tensor(s4),
            _one_tensor(s5),
            _one_tensor(s6),
            _one_tensor(s7),
            _one_tensor(s8),
            blocks,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,
        inp: cute.Tensor,
        p0: cute.Tensor,
        p1: cute.Tensor,
        p2: cute.Tensor,
        p3: cute.Tensor,
        p4: cute.Tensor,
        p5: cute.Tensor,
        p6: cute.Tensor,
        p7: cute.Tensor,
        p8: cute.Tensor,
        s0: cute.Tensor,
        s1: cute.Tensor,
        s2: cute.Tensor,
        s3: cute.Tensor,
        s4: cute.Tensor,
        s5: cute.Tensor,
        s6: cute.Tensor,
        s7: cute.Tensor,
        s8: cute.Tensor,
        blocks: Int64,
    ) -> None:
        payloads = (p0, p1, p2, p3, p4, p5, p6, p7, p8)
        scales = (s0, s1, s2, s3, s4, s5, s6, s7, s8)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = Int32(tidx) & Int32(31)
        warp = Int32(tidx) >> Int32(5)
        block = Int64(bidx) * Int64(_WARPS_PER_CTA) + Int64(warp)
        step = Int64(gdim) * Int64(_WARPS_PER_CTA)
        while block < blocks:
            elem0 = block * Int64(_QUANT_BLOCK) + Int64(lane) * Int64(4)
            a0, a1, a2, a3 = _load_bf16x4(inp, elem0)
            for source in cutlass.range_constexpr(9):
                v0, v1, v2, v3 = _decode_payload(
                    self._codec,
                    payloads[source],
                    scales[source],
                    block,
                    lane,
                )
                a0 += v0
                a1 += v1
                a2 += v2
                a3 += v3
            _store_bf16x4(out, elem0, a0, a1, a2, a3)
            block += step


class _DequantAddQuantLaunch:
    def __init__(self, codec: str, store_bf16: bool) -> None:
        self._codec = codec
        self._store_bf16 = bool(store_bf16)

    @cute.jit
    def __call__(
        self,
        out_ptr: cute.Pointer,
        local_ptr: cute.Pointer,
        payload_in_ptr: cute.Pointer,
        scales_in_ptr: cute.Pointer,
        payload_out_ptr: cute.Pointer,
        scales_out_ptr: cute.Pointer,
        blocks: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _one_tensor(out_ptr),
            _one_tensor(local_ptr),
            _one_tensor(payload_in_ptr),
            _one_tensor(scales_in_ptr),
            _one_tensor(payload_out_ptr),
            _one_tensor(scales_out_ptr),
            blocks,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,
        local: cute.Tensor,
        payload_in: cute.Tensor,
        scales_in: cute.Tensor,
        payload_out: cute.Tensor,
        scales_out: cute.Tensor,
        blocks: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = Int32(tidx) & Int32(31)
        warp = Int32(tidx) >> Int32(5)
        block = Int64(bidx) * Int64(_WARPS_PER_CTA) + Int64(warp)
        step = Int64(gdim) * Int64(_WARPS_PER_CTA)
        while block < blocks:
            elem0 = block * Int64(_QUANT_BLOCK) + Int64(lane) * Int64(4)
            a0, a1, a2, a3 = _load_bf16x4(local, elem0)
            b0, b1, b2, b3 = _decode_payload(
                self._codec, payload_in, scales_in, block, lane
            )
            v0 = a0 + b0
            v1 = a1 + b1
            v2 = a2 + b2
            v3 = a3 + b3
            if cutlass.const_expr(self._codec == "mx"):
                amax = _warp_amax_8(v0, v1, v2, v3)
                scale = _mx_scale_byte(amax)
                scale_group = lane >> Int32(3)
                scale_lane = lane & Int32(7)
                if scale_lane == Int32(0):
                    scale_idx = block * Int64(_MX_SCALES_PER_BLOCK) + Int64(
                        scale_group
                    )
                    _st_global_u8(get_ptr_as_int64(scales_out, scale_idx), scale)
                inv_scale = _mx_inv_scale(scale)
                packed = cvt_f32x4_to_e4m3x4(
                    v0 * inv_scale,
                    v1 * inv_scale,
                    v2 * inv_scale,
                    v3 * inv_scale,
                )
            else:
                amax = _warp_amax_32(v0, v1, v2, v3)
                maximum = _INT8_MAX if self._codec == "i8" else _FP8_MAX
                scale = div_rn_f32(amax, Float32(maximum))
                if not (amax > Float32(0.0)):
                    scale = Float32(1.0)
                if lane == Int32(0):
                    _st_global_f32(get_ptr_as_int64(scales_out, block), scale)
                inv_scale = div_rn_f32(Float32(1.0), scale)
                if cutlass.const_expr(self._codec == "i8"):
                    packed = _pack_i8x4(
                        v0 * inv_scale,
                        v1 * inv_scale,
                        v2 * inv_scale,
                        v3 * inv_scale,
                    )
                else:
                    packed = cvt_f32x4_to_e4m3x4(
                        v0 * inv_scale,
                        v1 * inv_scale,
                        v2 * inv_scale,
                        v3 * inv_scale,
                    )
            _st_global_u32(
                get_ptr_as_int64(payload_out, elem0 // Int64(4)), packed
            )
            if cutlass.const_expr(self._store_bf16):
                _store_bf16x4(out, elem0, v0, v1, v2, v3)
            block += step


def _dtype_type(dtype_name: str) -> type[cutlass.Numeric]:
    if dtype_name == "bf16":
        return cutlass.BFloat16
    if dtype_name == "fp16":
        return cutlass.Float16
    if dtype_name == "fp32":
        return cutlass.Float32
    raise ValueError(f"unknown DMA dtype {dtype_name!r}")


def _payload_type(codec: str) -> type[cutlass.Numeric]:
    if codec not in _CODECS:
        raise ValueError(f"unknown DMA codec {codec!r}")
    return Uint32


def _scale_type(codec: str) -> type[cutlass.Numeric]:
    return Uint8 if codec == "mx" else Float32


def _fake_ptr(dtype: type[cutlass.Numeric], align: int) -> cute.Pointer:
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)


def _compile(
    kernel_id: str,
    key: tuple[object, ...],
    launch: object,
    *args: object,
) -> Callable:
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        *args,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(kernel_id, 1, key),
    )


@functools.cache
def _compiled_flag(wait: bool) -> Callable:
    launch = _FlagLaunch(wait)
    return _compile(
        "comm.pcie_dma.flag",
        ("wait" if wait else "set",),
        launch,
        _fake_ptr(Uint32, 4),
        _fake_ptr(Uint32, 4),
    )


@functools.cache
def _compiled_add(dtype_name: str) -> Callable:
    dtype = _dtype_type(dtype_name)
    launch = _AddLaunch(dtype_name)
    return _compile(
        "comm.pcie_dma.add",
        (dtype_name,),
        launch,
        _fake_ptr(dtype, 16),
        _fake_ptr(dtype, 16),
        _fake_ptr(dtype, 16),
        Int64(1),
        Int32(1),
    )


@functools.cache
def _compiled_quant(codec: str) -> Callable:
    launch = _QuantLaunch(codec)
    return _compile(
        "comm.pcie_dma.quant",
        (codec,),
        launch,
        _fake_ptr(cutlass.BFloat16, 8),
        _fake_ptr(_payload_type(codec), 4),
        _fake_ptr(_scale_type(codec), 4),
        Int64(1),
        Int32(1),
    )


@functools.cache
def _compiled_dequant_store(codec: str) -> Callable:
    launch = _DequantStoreLaunch(codec)
    return _compile(
        "comm.pcie_dma.dequant_store",
        (codec,),
        launch,
        _fake_ptr(cutlass.BFloat16, 8),
        _fake_ptr(_payload_type(codec), 4),
        _fake_ptr(_scale_type(codec), 4),
        Int64(1),
        Int32(1),
    )


@functools.cache
def _compiled_dequant_accum(codec: str, nsrc: int) -> Callable:
    if not 1 <= nsrc <= 9:
        raise ValueError(f"DMA dequant accumulate requires 1..9 sources, got {nsrc}")
    if nsrc == 9:
        launch = _DequantAccum9Launch(codec)
        abi_sources = 9
    else:
        launch = _DequantAccumLaunch(codec, nsrc)
        abi_sources = 8
    payload = _fake_ptr(_payload_type(codec), 4)
    scale = _fake_ptr(_scale_type(codec), 4)
    return _compile(
        "comm.pcie_dma.dequant_accum",
        (codec, nsrc),
        launch,
        _fake_ptr(cutlass.BFloat16, 8),
        _fake_ptr(cutlass.BFloat16, 8),
        *(payload for _ in range(abi_sources)),
        *(scale for _ in range(abi_sources)),
        Int64(1),
        Int32(1),
    )


@functools.cache
def _compiled_dequant_add_quant(codec: str, store_bf16: bool) -> Callable:
    launch = _DequantAddQuantLaunch(codec, store_bf16)
    return _compile(
        "comm.pcie_dma.dequant_add_quant",
        (codec, bool(store_bf16)),
        launch,
        _fake_ptr(cutlass.BFloat16, 8),
        _fake_ptr(cutlass.BFloat16, 8),
        _fake_ptr(_payload_type(codec), 4),
        _fake_ptr(_scale_type(codec), 4),
        _fake_ptr(_payload_type(codec), 4),
        _fake_ptr(_scale_type(codec), 4),
        Int64(1),
        Int32(1),
    )


def _grid_for_packs(packs: int) -> int:
    return max(1, min(_MAX_GRID, (int(packs) + _THREADS - 1) // _THREADS))


def _grid_for_quant_blocks(blocks: int) -> int:
    return max(
        1,
        min(_MAX_GRID, (int(blocks) + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA),
    )


def _ptr(dtype: type[cutlass.Numeric], address: int, align: int) -> cute.Pointer:
    return make_ptr(dtype, int(address), cute.AddressSpace.gmem, assumed_align=align)


@dataclass(frozen=True)
class DmaLaunchers:
    """Resolved CuTe entry points required by one fixed DMA wire protocol."""

    set_flag: Callable
    wait_flag: Callable
    add: tuple[Callable, ...]
    quant: Callable | None = None
    dequant_store: Callable | None = None
    dequant_accum: Callable | None = None
    dequant_add_quant_false: Callable | None = None
    dequant_add_quant_true: Callable | None = None

    @property
    def __b12x_dependencies__(self):
        return (
            self.set_flag, self.wait_flag, self.add, self.quant,
            self.dequant_store, self.dequant_accum,
            self.dequant_add_quant_false, self.dequant_add_quant_true,
        )

    @property
    def __b12x_programs__(self):
        from b12x._lib.compile_plan import program_keys
        return program_keys(self.__b12x_dependencies__)


def compile_launchers(*, world_size: int, wire_mode: str) -> DmaLaunchers:
    """Resolve the primitive programs reachable by declaration metadata."""

    if int(world_size) not in (2, 4, 6, 8, 10):
        raise ValueError(f"unsupported DMA world size {world_size}")
    mode = str(wire_mode)
    codec = "i8" if mode.startswith("i8") else "mx" if mode.startswith("mx") else "e4m3"
    compressed = bool(mode)
    a2a = mode == "a2a" or mode.endswith("a2a")
    return DmaLaunchers(
        set_flag=_compiled_flag(False),
        wait_flag=_compiled_flag(True),
        add=tuple(_compiled_add(dtype) for dtype in _DTYPES),
        quant=_compiled_quant(codec) if compressed else None,
        dequant_store=_compiled_dequant_store(codec) if compressed else None,
        dequant_accum=_compiled_dequant_accum(codec, int(world_size) - 1) if a2a else None,
        dequant_add_quant_false=_compiled_dequant_add_quant(codec, False) if compressed else None,
        dequant_add_quant_true=_compiled_dequant_add_quant(codec, True) if compressed else None,
    )


class DmaKernels:
    """Python-callable façade bound to preparation-resolved launchers."""

    def __init__(self, ipc) -> None:
        self._ipc = ipc
        self._launchers: DmaLaunchers | None = None

    def install(self, launchers: DmaLaunchers) -> None:
        if not isinstance(launchers, DmaLaunchers):
            raise TypeError("DMA kernels require preparation-resolved launchers")
        self._launchers = launchers

    def _require_launchers(self) -> DmaLaunchers:
        if self._launchers is None:
            raise RuntimeError("DMA kernels have not been prepared")
        return self._launchers

    def dma_copy(self, dst_ptr: int, src_ptr: int, bytes_: int) -> None:
        stream = int(current_cuda_stream())
        self._ipc.cudaMemcpyAsync(int(dst_ptr), int(src_ptr), int(bytes_), stream)

    def dma_set_flag(self, peer_flag_ptr: int, counter_ptr: int) -> None:
        raw = self._require_launchers().set_flag
        raw(
            _ptr(Uint32, peer_flag_ptr, 4),
            _ptr(Uint32, counter_ptr, 4),
            current_cuda_stream(),
        )

    def dma_wait_flag(self, flag_ptr: int, counter_ptr: int) -> None:
        raw = self._require_launchers().wait_flag
        raw(
            _ptr(Uint32, flag_ptr, 4),
            _ptr(Uint32, counter_ptr, 4),
            current_cuda_stream(),
        )

    def dma_add(
        self, dst_ptr: int, a_ptr: int, b_ptr: int, elems: int, dtype_code: int
    ) -> None:
        dtype_name = _DTYPES[int(dtype_code)]
        dtype = _dtype_type(dtype_name)
        elems_per_pack = 4 if dtype_name == "fp32" else 8
        packs = int(elems) // elems_per_pack
        raw = self._require_launchers().add[int(dtype_code)]
        raw(
            _ptr(dtype, dst_ptr, 16),
            _ptr(dtype, a_ptr, 16),
            _ptr(dtype, b_ptr, 16),
            packs,
            _grid_for_packs(packs),
            current_cuda_stream(),
        )

    def _quant(
        self, codec: str, src_ptr: int, payload_ptr: int, scales_ptr: int, elems: int
    ) -> None:
        blocks = int(elems) // _QUANT_BLOCK
        raw = self._require_launchers().quant
        if raw is None:
            raise RuntimeError("DMA quantization launcher was not prepared")
        raw(
            _ptr(cutlass.BFloat16, src_ptr, 8),
            _ptr(_payload_type(codec), payload_ptr, 4),
            _ptr(_scale_type(codec), scales_ptr, 4),
            blocks,
            _grid_for_quant_blocks(blocks),
            current_cuda_stream(),
        )

    def _dequant_store(
        self, codec: str, out_ptr: int, payload_ptr: int, scales_ptr: int, elems: int
    ) -> None:
        blocks = int(elems) // _QUANT_BLOCK
        raw = self._require_launchers().dequant_store
        if raw is None:
            raise RuntimeError("DMA dequantization launcher was not prepared")
        raw(
            _ptr(cutlass.BFloat16, out_ptr, 8),
            _ptr(_payload_type(codec), payload_ptr, 4),
            _ptr(_scale_type(codec), scales_ptr, 4),
            blocks,
            _grid_for_quant_blocks(blocks),
            current_cuda_stream(),
        )

    def _dequant_accum(
        self,
        codec: str,
        out_ptr: int,
        inp_ptr: int,
        payload_ptrs: Sequence[int],
        scale_ptrs: Sequence[int],
        elems: int,
    ) -> None:
        if len(payload_ptrs) != len(scale_ptrs):
            raise ValueError("DMA payload and scale source counts must match")
        if not 1 <= len(payload_ptrs) <= 9:
            raise ValueError(
                f"DMA dequant accumulate requires 1..9 sources, got {len(payload_ptrs)}"
            )
        payloads = list(map(int, payload_ptrs))
        scales = list(map(int, scale_ptrs))
        abi_sources = 9 if len(payloads) == 9 else 8
        payloads.extend([payloads[0]] * (abi_sources - len(payloads)))
        scales.extend([scales[0]] * (abi_sources - len(scales)))
        blocks = int(elems) // _QUANT_BLOCK
        raw = self._require_launchers().dequant_accum
        if raw is None:
            raise RuntimeError("DMA all-to-all accumulation launcher was not prepared")
        raw(
            _ptr(cutlass.BFloat16, out_ptr, 8),
            _ptr(cutlass.BFloat16, inp_ptr, 8),
            *(_ptr(_payload_type(codec), address, 4) for address in payloads),
            *(_ptr(_scale_type(codec), address, 4) for address in scales),
            blocks,
            _grid_for_quant_blocks(blocks),
            current_cuda_stream(),
        )

    def _dequant_add_quant(
        self,
        codec: str,
        out_ptr: int,
        local_ptr: int,
        payload_in_ptr: int,
        scales_in_ptr: int,
        payload_out_ptr: int,
        scales_out_ptr: int,
        elems: int,
        store_bf16: bool,
    ) -> None:
        blocks = int(elems) // _QUANT_BLOCK
        launchers = self._require_launchers()
        raw = (
            launchers.dequant_add_quant_true
            if store_bf16
            else launchers.dequant_add_quant_false
        )
        if raw is None:
            raise RuntimeError("DMA quantized reduction launcher was not prepared")
        raw(
            _ptr(cutlass.BFloat16, out_ptr, 8),
            _ptr(cutlass.BFloat16, local_ptr, 8),
            _ptr(_payload_type(codec), payload_in_ptr, 4),
            _ptr(_scale_type(codec), scales_in_ptr, 4),
            _ptr(_payload_type(codec), payload_out_ptr, 4),
            _ptr(_scale_type(codec), scales_out_ptr, 4),
            blocks,
            _grid_for_quant_blocks(blocks),
            current_cuda_stream(),
        )


def _install_codec_methods() -> None:
    for suffix, codec in (("", "e4m3"), ("_i8", "i8"), ("_mx", "mx")):
        setattr(
            DmaKernels,
            f"dma_quant{suffix}",
            lambda self, src, payload, scales, elems, _codec=codec: self._quant(
                _codec, src, payload, scales, elems
            ),
        )
        setattr(
            DmaKernels,
            f"dma_dequant_store{suffix}",
            lambda self, out, payload, scales, elems, _codec=codec: self._dequant_store(
                _codec, out, payload, scales, elems
            ),
        )
        setattr(
            DmaKernels,
            f"dma_dequant_accum{suffix}",
            lambda self, out, inp, payloads, scales, elems, _codec=codec: self._dequant_accum(
                _codec, out, inp, payloads, scales, elems
            ),
        )
        setattr(
            DmaKernels,
            f"dma_dequant_add_quant{suffix}",
            lambda self,
            out,
            local,
            payload_in,
            scales_in,
            payload_out,
            scales_out,
            elems,
            store_bf16,
            _codec=codec: self._dequant_add_quant(
                _codec,
                out,
                local,
                payload_in,
                scales_in,
                payload_out,
                scales_out,
                elems,
                store_bf16,
            ),
        )

_install_codec_methods()


__all__ = ["DmaKernels", "DmaLaunchers", "compile_launchers"]
