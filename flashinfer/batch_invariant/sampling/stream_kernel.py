# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The device side of the stream (``stream.py``): Philox4x32-10 on the request key, the fp32 and fp64 uniform
conversions, as CuTe-DSL device functions built from inline PTX integer ops (``mul.lo`` / ``mul.hi`` / ``xor`` / ``add``
on 32- and 64-bit words, the two declared ``cvt`` forms, ``mul.rn.f64`` for the fp64 lane) so the compiler cannot
reassociate or fuse them, and one conformance kernel that writes every block's words and both uniforms for a batch of
keys. The draw kernel (``draw_kernel.py``) calls the same device functions."""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import jit
from .cute_ext import CuteF32
from .stream import (
    PHILOX_M0,
    PHILOX_M1,
    PHILOX_ROUNDS,
    PHILOX_W0,
    PHILOX_W1,
    SPLITMIX_GOLDEN,
    SPLITMIX_M1,
    SPLITMIX_M2,
    TWO_POW_M32,
    TWO_POW_M33,
    TWO_POW_M53,
)

THREADS = 128
#: The on-disk cache module of the stream conformance kernel and the draw kernels (``sampling/__init__.py``).
DRAW_MODULE = "batch_invariant_sampling_draw"


def _asm2(ty, fmt, cons, a, b):
    """One inline-PTX instruction of two operands (``op d, a, b``): the bits move as PTX states, beyond the
    compiler's reach."""
    return llvm.inline_asm(
        ty,
        [a, b],
        fmt,
        cons,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _asm1(ty, fmt, cons, a):
    """One inline-PTX instruction of one operand (``op d, a``)."""
    return llvm.inline_asm(
        ty,
        [a],
        fmt,
        cons,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def u32_mul_lo(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """``mul.lo.u32``: the low 32 bits of the product."""
    return Int32(
        _asm2(
            T.i32(),
            "mul.lo.u32 $0, $1, $2;",
            "=r,r,r",
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(b).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u32_mul_hi(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """``mul.hi.u32``: the high 32 bits of the product."""
    return Int32(
        _asm2(
            T.i32(),
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(b).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u32_xor(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """``xor.b32``."""
    return Int32(
        _asm2(
            T.i32(),
            "xor.b32 $0, $1, $2;",
            "=r,r,r",
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(b).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u32_add(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """``add.u32`` (modulo 2^32)."""
    return Int32(
        _asm2(
            T.i32(),
            "add.u32 $0, $1, $2;",
            "=r,r,r",
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(b).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u64_mul_lo(a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    """``mul.lo.u64``: the low 64 bits of the product."""
    return Int64(
        _asm2(
            T.i64(),
            "mul.lo.u64 $0, $1, $2;",
            "=l,l,l",
            Int64(a).ir_value(loc=loc, ip=ip),
            Int64(b).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u64_xor(a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    """``xor.b64``."""
    return Int64(
        _asm2(
            T.i64(),
            "xor.b64 $0, $1, $2;",
            "=l,l,l",
            Int64(a).ir_value(loc=loc, ip=ip),
            Int64(b).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u64_add(a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    """``add.u64`` (modulo 2^64)."""
    return Int64(
        _asm2(
            T.i64(),
            "add.u64 $0, $1, $2;",
            "=l,l,l",
            Int64(a).ir_value(loc=loc, ip=ip),
            Int64(b).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u64_shr(a: Int64, n: Int32, *, loc=None, ip=None) -> Int64:
    """Logical right shift of a 64-bit word."""
    return Int64(
        _asm2(
            T.i64(),
            "shr.b64 $0, $1, $2;",
            "=l,l,r",
            Int64(a).ir_value(loc=loc, ip=ip),
            Int32(n).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u64_shl(a: Int64, n: Int32, *, loc=None, ip=None) -> Int64:
    """``shl.b64``: the logical left shift of a 64-bit word."""
    return Int64(
        _asm2(
            T.i64(),
            "shl.b64 $0, $1, $2;",
            "=l,l,r",
            Int64(a).ir_value(loc=loc, ip=ip),
            Int32(n).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u64_or(a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    """``or.b64``."""
    return Int64(
        _asm2(
            T.i64(),
            "or.b64 $0, $1, $2;",
            "=l,l,l",
            Int64(a).ir_value(loc=loc, ip=ip),
            Int64(b).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def u64_lo32(a: Int64, *, loc=None, ip=None) -> Int32:
    """``cvt.u32.u64``: the low 32 bits of a 64-bit word."""
    return Int32(
        _asm1(T.i32(), "cvt.u32.u64 $0, $1;", "=r,l", Int64(a).ir_value(loc=loc, ip=ip))
    )


@dsl_user_op
def u32_to_u64(a: Int32, *, loc=None, ip=None) -> Int64:
    """``cvt.u64.u32``: a 32-bit word widened to 64 bits."""
    return Int64(
        _asm1(T.i64(), "cvt.u64.u32 $0, $1;", "=l,r", Int32(a).ir_value(loc=loc, ip=ip))
    )


@dsl_user_op
def cvt_f32_u32(a: Int32, *, loc=None, ip=None) -> Float32:
    """``cvt.rn.f32.u32``: the declared integer-to-binary32 conversion of a word."""
    return Float32(
        _asm1(
            T.f32(),
            "cvt.rn.f32.u32 $0, $1;",
            "=f,r",
            Int32(a).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def cvt_f64_u64(a: Int64, *, loc=None, ip=None) -> Float64:
    """``cvt.rn.f64.u64``: exact below 2^53 (the fp64 lane feeds it ``2 z + 1 < 2^53``)."""
    return Float64(
        _asm1(
            T.f64(),
            "cvt.rn.f64.u64 $0, $1;",
            "=d,l",
            Int64(a).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def f64_mul_rn(a: Float64, b: Float64, *, loc=None, ip=None) -> Float64:
    """``mul.rn.f64``."""
    return Float64(
        _asm2(
            T.f64(),
            "mul.rn.f64 $0, $1, $2;",
            "=d,d,d",
            Float64(a).ir_value(loc=loc, ip=ip),
            Float64(b).ir_value(loc=loc, ip=ip),
        )
    )


def _i32(v: int) -> Int32:
    """A 32-bit constant as its two's-complement Int32 bit pattern."""
    v &= 0xFFFFFFFF
    return Int32(v - (1 << 32) if v >= 1 << 31 else v)


def _i64(v: int) -> Int64:
    """A 64-bit constant as its two's-complement Int64 bit pattern."""
    v &= 0xFFFFFFFFFFFFFFFF
    return Int64(v - (1 << 64) if v >= 1 << 63 else v)


@cute.jit
def mix64(v: Int64) -> Int64:
    """SplitMix64's finalizer on the device (``stream.mix64``)."""
    v = u64_add(v, _i64(SPLITMIX_GOLDEN))
    v = u64_mul_lo(u64_xor(v, u64_shr(v, Int32(30))), _i64(SPLITMIX_M1))
    v = u64_mul_lo(u64_xor(v, u64_shr(v, Int32(27))), _i64(SPLITMIX_M2))
    return u64_xor(v, u64_shr(v, Int32(31)))


@cute.jit
def request_key(seed: Int64, offset: Int64) -> Int64:
    """The row's 64-bit Philox key ``mix64(seed ^ mix64(offset))`` on the device (``stream.request_key``)."""
    return mix64(u64_xor(seed, mix64(offset)))


@cute.jit
def philox_block(n_lo: Int32, n_hi: Int32, k0: Int32, k1: Int32):
    """Philox4x32-10 of the counter ``(n_lo, n_hi, 0, 0)`` under key ``(k0, k1)``: the four words."""
    c0, c1, c2, c3 = n_lo, n_hi, Int32(0), Int32(0)
    for r in cutlass.range_constexpr(PHILOX_ROUNDS):
        lo0 = u32_mul_lo(_i32(PHILOX_M0), c0)
        hi0 = u32_mul_hi(_i32(PHILOX_M0), c0)
        lo1 = u32_mul_lo(_i32(PHILOX_M1), c2)
        hi1 = u32_mul_hi(_i32(PHILOX_M1), c2)
        n0 = u32_xor(u32_xor(hi1, c1), k0)
        n2 = u32_xor(u32_xor(hi0, c3), k1)
        c0, c1, c2, c3 = n0, lo1, n2, lo0
        if cutlass.const_expr(r != PHILOX_ROUNDS - 1):
            k0 = u32_add(k0, _i32(PHILOX_W0))
            k1 = u32_add(k1, _i32(PHILOX_W1))
    return c0, c1, c2, c3


@cute.jit
def uniform32(w: Int32) -> Float32:
    """``add(mul(cvt.rn.f32.u32(w), 2^-32), 2^-33)``: ``curand_uniform``'s conversion."""
    return CuteF32.add(
        CuteF32.mul(cvt_f32_u32(w), Float32(TWO_POW_M32)), Float32(TWO_POW_M33)
    )


@cute.jit
def uniform64(w1: Int32, w2: Int32) -> Float64:
    """``(2 z + 1) * 2^-53`` with ``z`` the top 52 bits of ``(w2 << 32) | w1``."""
    z = u64_shr(u64_or(u64_shl(u32_to_u64(w2), Int32(32)), u32_to_u64(w1)), Int32(12))
    odd = u64_add(u64_shl(z, Int32(1)), Int64(1))
    return f64_mul_rn(cvt_f64_u64(odd), Float64(TWO_POW_M53))


class StreamWordsKernel:
    """Conformance kernel: for row ``b`` and draw ``d``, the block's four words (int32 bit patterns), the fp32 uniform
    of word 0 and the fp64 uniform of words 1 and 2."""

    @cute.jit
    def __call__(
        self,
        mSeed: cute.Tensor,
        mOff: cute.Tensor,
        mW: cute.Tensor,
        mU32: cute.Tensor,
        mU64: cute.Tensor,
        B: Int32,
        draws: Int32,
        stream,
    ):
        self.kernel(mSeed, mOff, mW, mU32, mU64, B, draws).launch(
            grid=[cute.ceil_div(B * draws, THREADS), 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSeed: cute.Tensor,
        mOff: cute.Tensor,
        mW: cute.Tensor,
        mU32: cute.Tensor,
        mU64: cute.Tensor,
        B: Int32,
        draws: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * THREADS + tidx
        if i < B * draws:
            b = i // draws
            d = i - b * draws
            key = request_key(mSeed[b], mOff[b])
            k0 = u64_lo32(key)
            k1 = u64_lo32(u64_shr(key, Int32(32)))
            w0, w1, w2, w3 = philox_block(d, Int32(0), k0, k1)
            mW[b, d, 0] = w0
            mW[b, d, 1] = w1
            mW[b, d, 2] = w2
            mW[b, d, 3] = w3
            mU32[b, d] = uniform32(w0)
            mU64[b, d] = uniform64(w1, w2)


@functools.cache
def get_stream_words_kernel():
    """The compiled conformance kernel (built once, cached in the draw's module)."""

    def compile_fn():
        # Distinct symbolic sizes per axis: the TVM-FFI entry checks the call's shapes against the compiled ones, so a
        # shared symbol would demand B == draws at every call.
        sb, sd = cute.sym_int(32), cute.sym_int(32)
        seeds = cute.runtime.make_fake_tensor(Int64, (sb,), (1,), assumed_align=8)
        offs = cute.runtime.make_fake_tensor(Int64, (sb,), (1,), assumed_align=8)
        wds = cute.runtime.make_fake_tensor(
            Int32, (sb, sd, 4), (cute.sym_int64(), 4, 1), assumed_align=16
        )
        u32 = cute.runtime.make_fake_tensor(
            Float32, (sb, sd), (cute.sym_int64(), 1), assumed_align=4
        )
        u64 = cute.runtime.make_fake_tensor(
            Float64, (sb, sd), (cute.sym_int64(), 1), assumed_align=8
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            StreamWordsKernel(),
            seeds,
            offs,
            wds,
            u32,
            u64,
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(DRAW_MODULE, "stream_words", compile_fn)


def stream_words(seeds, offsets, draws: int):
    """``(words int32 [B, draws, 4], uniform32 [B, draws], uniform64 [B, draws])`` on the device for int64 key tensors."""
    import torch

    seeds = seeds.to(torch.int64).contiguous()
    offsets = offsets.to(torch.int64).contiguous()
    B = seeds.numel()
    w = torch.empty(B, draws, 4, dtype=torch.int32, device=seeds.device)
    u32 = torch.empty(B, draws, dtype=torch.float32, device=seeds.device)
    u64 = torch.empty(B, draws, dtype=torch.float64, device=seeds.device)
    get_stream_words_kernel()(seeds, offsets, w, u32, u64, B, draws)
    return w, u32, u64
