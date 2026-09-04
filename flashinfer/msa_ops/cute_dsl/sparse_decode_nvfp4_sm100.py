"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MSA sparse DECODE attention over a packed paged NVFP4 KV cache (sm100/sm103).

Storage is NVFP4 (packed e2m1 nibbles + e4m3 block scales, K scale linear,
V scale (4,4)-swizzled, all four planes strided views of ONE packed page which
is consumed in place).  Compute is the BF16/FP16 tensor-core arm: K and V are
dequantised in-kernel to fp16 -- e2m1 x e4m3 needs at most five mantissa bits,
so the dequantised operand is EXACT -- and fed to `mma.sync ... f32.f16.f16.f32`
with fp32 accumulation.  No native FP4 MMA is issued anywhere.

Softmax uses a running per-row maximum carried through the streaming KV loop;
the row max is never elided or replaced by a fixed shift.  Partials, the
split-K combine and every softmax statistic are fp32.

THIS MODULE IS NOT A COMPLETE DECODE IMPLEMENTATION AND IS NOT REACHED DIRECTLY.
It is the SPECIALISED HALF of the compute-capability 10.0/10.3 NVFP4 decode
route. :mod:`flashinfer.msa_ops._nvfp4_decode_sm100` owns the public surface,
validates every call, and routes here only when :func:`specialised_reason`
returns ``None`` -- i.e. only on the geometries whose instantiations this file
actually specialises. Every other call is served by the route's own kernel.

Two consequences of that split are load-bearing and are enforced here rather
than assumed:

* The fully dynamic "generalized" instantiation this file's dispatch arithmetic
  would otherwise select is NEVER COMPILED (see :func:`_get_kernels`). It is the
  only instantiation whose split-K partials are published through global memory,
  and its publication protocol is not sound: the partial and LSE stores are
  plain ``st.global`` and the only device-scope release is a single ``tid==0``
  ``atom.add.release.gpu``, so the stores of threads 1..255 are not ordered
  before the arrival that advertises them. Not compiling it makes that
  unreachable by construction instead of by dispatch discipline, and
  :func:`run` refuses a second time if the arithmetic ever selects it.
* Because that instantiation is the only consumer of the split-K arena, the
  ~65 MiB/device persistent scratch it would need is NOT ALLOCATED. See
  :data:`ARENA_BYTES_IF_GENERALIZED_WERE_REACHABLE`.

:func:`run` is lookup-only: it never compiles and never allocates, so it is safe
inside a CUDA-graph capture. :func:`warmup` must be called for a device first;
the route's ``warm`` entry point does that.
"""
import threading

import torch
import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Int32, Int64, Float32
# --------------------------------------------------------------------------
# Low-level PTX helpers.
# --------------------------------------------------------------------------
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

_ATT = llvm.AsmDialect.AD_ATT


def _iv(x, ty, loc, ip):
    return ty(x).ir_value(loc=loc, ip=ip)


@dsl_user_op
def fmax3(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """Blackwell three-input FMNMX3 through NVVM's ternary fmax form."""
    from cutlass import CUDA_VERSION
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        return Float32(nvvm.fmax(
            T.f32(), _iv(a, Float32, loc, ip), _iv(b, Float32, loc, ip),
            c=_iv(c, Float32, loc, ip), loc=loc, ip=ip))
    return Float32(nvvm.fmax(
        _iv(a, Float32, loc, ip), _iv(b, Float32, loc, ip),
        c=_iv(c, Float32, loc, ip), loc=loc, ip=ip))


@dsl_user_op
def ld_global_v4_b32(addr: Int64, off: int = 0, *, loc=None, ip=None):
    """128-bit read-only global load -> 4 x b32, at a CONSTANT byte offset.

    Every KV load of a chunk sits at a compile-time displacement from the same
    per-plane base, so the displacement belongs in the instruction's own offset
    field rather than in a 64-bit add that ptxas has to materialise into a
    register pair.  The whole chunk then costs ONE address chain instead of
    twelve.
    """
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_iv(addr, Int64, loc, ip)],
        "ld.global.nc.v4.b32 {$0,$1,$2,$3}, [$4+%d];" % off,
        "=r,=r,=r,=r,l", has_side_effects=False, is_align_stack=False,
        asm_dialect=_ATT)
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def ld_global_v2_b32(addr: Int64, off: int = 0, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [_iv(addr, Int64, loc, ip)],
        "ld.global.nc.v2.b32 {$0,$1}, [$2+%d];" % off, "=r,=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT)
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
                 for i in range(2))


@dsl_user_op
def ld_global_u8(addr: Int64, off: int = 0, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(addr, Int64, loc, ip)],
        "ld.global.nc.u8 $0, [$1+%d];" % off, "=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def ld_global_u16(addr: Int64, off: int = 0, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(addr, Int64, loc, ip)],
        "ld.global.nc.u16 $0, [$1+%d];" % off, "=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def prefetch_l2(addr: Int64, *, loc=None, ip=None):
    """Pull one global line into L2 WITHOUT occupying a destination register.

    The register pipeline can only run as deep as the shadow register sets it
    can afford.  An L2 prefetch runs one stage deeper for free: the line lands
    in the cache instead of in a register, so the load that does land in a
    register finds it already resident.
    """
    llvm.inline_asm(
        None, [_iv(addr, Int64, loc, ip)],
        "prefetch.global.L2 [$0];", "l",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)


@dsl_user_op
def st_shared_b32(addr: Int32, r0, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [_iv(addr, Int32, loc, ip), _iv(r0, Int32, loc, ip)],
        "st.shared.b32 [$0], $1;", "r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)


@dsl_user_op
def st_shared_v4_b32(addr: Int32, r0, r1, r2, r3, *, loc=None, ip=None):
    llvm.inline_asm(
        llvm.StructType.get_literal([]),
        [_iv(addr, Int32, loc, ip), _iv(r0, Int32, loc, ip),
         _iv(r1, Int32, loc, ip), _iv(r2, Int32, loc, ip),
         _iv(r3, Int32, loc, ip)],
        "st.shared.v4.b32 [$0], {$1,$2,$3,$4};",
        "r,r,r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=_ATT)


@dsl_user_op
def ldmatrix_x4(addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_iv(addr, Int32, loc, ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=_ATT)
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def ldmatrix_x4_trans(addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_iv(addr, Int32, loc, ip)],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=_ATT)
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def mma_m16n8k16_f16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """Warp MMA, fp16 operands, fp32 accumulate (tensor-core `kind::f16` arm)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [_iv(a0, Int32, loc, ip), _iv(a1, Int32, loc, ip),
         _iv(a2, Int32, loc, ip), _iv(a3, Int32, loc, ip),
         _iv(b0, Int32, loc, ip), _iv(b1, Int32, loc, ip),
         _iv(c0, Float32, loc, ip), _iv(c1, Float32, loc, ip),
         _iv(c2, Float32, loc, ip), _iv(c3, Float32, loc, ip)],
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def e4m3_byte_to_f16x2(b: Int32, sel: int = 0, *, loc=None, ip=None) -> Int32:
    """Decode e4m3 block-scale byte `sel` of a packed word into an f16x2.

    Four block scales ride in one register, so the double-buffered K and V
    prefetches cost four scale registers instead of twelve.
    """
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(b, Int32, loc, ip)],
        "{\n\t.reg .b32 t;\n\t.reg .b16 p;\n\t"
        "prmt.b32 t, $1, 0, %d;\n\t"
        "mov.b32 {p,_}, t;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, p;\n\t}\n" % (sel * 0x1111),
        "=r,r", has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def e4m3x2_to_f16x2(b: Int32, *, loc=None, ip=None) -> Int32:
    """Decode two adjacent e4m3 scale bytes into one packed f16x2."""
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(b, Int32, loc, ip)],
        "{\n\t.reg .b16 p;\n\t"
        "mov.b32 {p,_}, $1;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, p;\n\t}\n",
        "=r,r", has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def e4m3x4_to_dup_f16x2(b: Int32, *, loc=None, ip=None):
    """Decode four e4m3 bytes into four scalar-broadcast f16x2 words."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_iv(b, Int32, loc, ip)],
        "{\n\t.reg .b16 p0,p1;\n\t.reg .b32 h0,h1;\n\t"
        "mov.b32 {p0,p1}, $4;\n\t"
        "cvt.rn.f16x2.e4m3x2 h0, p0;\n\t"
        "cvt.rn.f16x2.e4m3x2 h1, p1;\n\t"
        "prmt.b32 $0, h0, 0, 0x1010;\n\t"
        "prmt.b32 $1, h0, 0, 0x3232;\n\t"
        "prmt.b32 $2, h1, 0, 0x1010;\n\t"
        "prmt.b32 $3, h1, 0, 0x3232;\n\t}\n",
        "=r,=r,=r,=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=_ATT)
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def mul_f16x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(a, Int32, loc, ip), _iv(b, Int32, loc, ip)],
        "mul.rn.f16x2 $0, $1, $2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def dequant_fp4x8_f16x8(src: Int32, sf_f16x2: Int32, *, loc=None, ip=None):
    """8 packed e2m1 nibbles * one e4m3 block scale -> 8 fp16 (4 x b32).

    e2m1 x e4m3 needs at most 5 mantissa bits, so the product is EXACT in
    fp16: the dequantised operand carries no extra error into the MMA.
    """
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_iv(src, Int32, loc, ip), _iv(sf_f16x2, Int32, loc, ip)],
        "{\n\t.reg .b8 b0,b1,b2,b3;\n\t.reg .b32 h0,h1,h2,h3;\n\t"
        "mov.b32 {b0,b1,b2,b3}, $4;\n\t"
        "cvt.rn.f16x2.e2m1x2 h0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 h1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 h2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 h3, b3;\n\t"
        "mul.rn.f16x2 $0, h0, $5;\n\t"
        "mul.rn.f16x2 $1, h1, $5;\n\t"
        "mul.rn.f16x2 $2, h2, $5;\n\t"
        "mul.rn.f16x2 $3, h3, $5;\n\t}\n",
        "=r,=r,=r,=r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=_ATT)
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


_PV_TMPL = ("{\n\t.reg .b8 a0,a1,a2,a3,b0,b1,b2,b3;\n\t"
            ".reg .b32 ha,hb,hc,hd,t0,t1,t2,t3;\n\t"
            "mov.b32 {a0,a1,a2,a3}, $4;\n\t"
            "mov.b32 {b0,b1,b2,b3}, $5;\n\t"
            "cvt.rn.f16x2.e2m1x2 ha, a%d;\n\t"
            "cvt.rn.f16x2.e2m1x2 hb, b%d;\n\t"
            "cvt.rn.f16x2.e2m1x2 hc, a%d;\n\t"
            "cvt.rn.f16x2.e2m1x2 hd, b%d;\n\t"
            "prmt.b32 t0, ha, hb, 0x5410;\n\t"
            "prmt.b32 t1, ha, hb, 0x7632;\n\t"
            "prmt.b32 t2, hc, hd, 0x5410;\n\t"
            "prmt.b32 t3, hc, hd, 0x7632;\n\t"
            "mul.rn.f16x2 $0, t0, $6;\n\t"
            "mul.rn.f16x2 $1, t1, $6;\n\t"
            "mul.rn.f16x2 $2, t2, $6;\n\t"
            "mul.rn.f16x2 $3, t3, $6;\n\t}\n")


@dsl_user_op
def pv_nibble_pair(a: Int32, b: Int32, *, loc=None, ip=None):
    """Interleave two tokens' packed nibbles so `cvt` emits PV B-fragments.

    Byte k of `a` holds dims (2k, 2k+1) of token j; byte k of `b` holds the
    same two dims of token j+1.  The PV B operand needs the two TOKENS in one
    f16x2, which the fp16 path otherwise reaches with a prmt PER OUTPUT
    REGISTER, after the conversion.  Doing the interleave on the NIBBLES
    instead costs two shifts and two lop3 selects for ALL SIXTEEN values of
    the pair, and makes cvt.rn.f16x2.e2m1x2 land the transposed pair directly:
      X byte k = { a_k low  | b_k low  << 4 }  -> cvt -> {tok j, tok j+1} @ 2k
      Y byte k = { a_k high | b_k high << 4 }  -> cvt -> {tok j, tok j+1} @ 2k+1
    Both halves of the pair reuse X and Y, so eight prmt become four ALU ops.
    """
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [_iv(a, Int32, loc, ip), _iv(b, Int32, loc, ip)],
        "{\n\t.reg .b32 t,u;\n\t"
        "shl.b32 t, $3, 4;\n\t"
        "lop3.b32 $0, $2, t, 0x0f0f0f0f, 0xe4;\n\t"
        "shr.b32 u, $2, 4;\n\t"
        "lop3.b32 $1, u, $3, 0x0f0f0f0f, 0xe4;\n\t}\n",
        "=r,=r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=_ATT)
    return (Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)))


_PV_XY_TMPL = ("{\n\t.reg .b8 x0,x1,x2,x3,y0,y1,y2,y3;\n\t"
               ".reg .b32 h0,h1,h2,h3;\n\t"
               "mov.b32 {x0,x1,x2,x3}, $4;\n\t"
               "mov.b32 {y0,y1,y2,y3}, $5;\n\t"
               "cvt.rn.f16x2.e2m1x2 h0, x%d;\n\t"
               "cvt.rn.f16x2.e2m1x2 h1, y%d;\n\t"
               "cvt.rn.f16x2.e2m1x2 h2, x%d;\n\t"
               "cvt.rn.f16x2.e2m1x2 h3, y%d;\n\t"
               "mul.rn.f16x2 $0, h0, $6;\n\t"
               "mul.rn.f16x2 $1, h1, $6;\n\t"
               "mul.rn.f16x2 $2, h2, $6;\n\t"
               "mul.rn.f16x2 $3, h3, $6;\n\t}\n")


@dsl_user_op
def pv_dequant4_xy(x: Int32, y: Int32, sf: Int32, half, *, loc=None, ip=None):
    """Four PV B-fragments out of an already nibble-transposed token pair."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_iv(x, Int32, loc, ip), _iv(y, Int32, loc, ip),
         _iv(sf, Int32, loc, ip)],
        _PV_XY_TMPL % (2 * half, 2 * half, 2 * half + 1, 2 * half + 1),
        "=r,=r,=r,=r,r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=_ATT)
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def pv_dequant4(a: Int32, b: Int32, sf: Int32, half, *, loc=None, ip=None):
    """Build four PV B-fragments by transposing two packed tokens in registers."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_iv(a, Int32, loc, ip), _iv(b, Int32, loc, ip),
         _iv(sf, Int32, loc, ip)],
        _PV_TMPL % (2 * half, 2 * half, 2 * half + 1, 2 * half + 1),
        "=r,=r,=r,=r,r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=_ATT)
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def prmt_b32(a: Int32, b: Int32, c: Int32, *, loc=None, ip=None) -> Int32:
    """Byte-select out of the {a,b} octet with a RUNTIME selector."""
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(a, Int32, loc, ip), _iv(b, Int32, loc, ip),
                  _iv(c, Int32, loc, ip)],
        "prmt.b32 $0, $1, $2, $3;", "=r,r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def bf16x2_to_f16x2(x: Int32, *, loc=None, ip=None) -> Int32:
    """Widen a packed bf16x2 to f32 then narrow to f16x2 (exact for |x| < 65504)."""
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(x, Int32, loc, ip)],
        "{\n\t.reg .b32 a,b;\n\t"
        "shl.b32 a, $1, 16;\n\t"
        "and.b32 b, $1, -65536;\n\t"
        "cvt.rn.f16x2.f32 $0, b, a;\n\t}\n",
        "=r,r", has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def pack_f16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(lo, Float32, loc, ip), _iv(hi, Float32, loc, ip)],
        "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def log2_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [_iv(x, Float32, loc, ip)],
        "lg2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def rcp_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [_iv(x, Float32, loc, ip)],
        "rcp.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def fma_rn_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [_iv(a, Float32, loc, ip), _iv(b, Float32, loc, ip),
                  _iv(c, Float32, loc, ip)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def ld_global_s32(addr: Int64, off: int = 0, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(addr, Int64, loc, ip)],
        "ld.global.nc.b32 $0, [$1+%d];" % off, "=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def ld_global_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [_iv(addr, Int64, loc, ip)],
        "ld.global.nc.f32 $0, [$1];", "=f,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def match_any_b32(v: Int32, *, loc=None, ip=None) -> Int32:
    """Mask of the lanes of the full warp whose `v` equals this lane's."""
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(v, Int32, loc, ip)],
        "match.any.sync.b32 $0, $1, -1;", "=r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def ld_global_v4_f32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [_iv(addr, Int64, loc, ip)],
        "ld.global.nc.v4.f32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def membar_gl(*, loc=None, ip=None):
    """Device-scope release/acquire fence for the cross-CTA split-K handoff."""
    llvm.inline_asm(llvm.StructType.get_literal([]), [], "membar.gl;", "",
                    has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)


@dsl_user_op
def atom_add_u32(addr: Int64, v: Int32, *, loc=None, ip=None) -> Int32:
    """Arrival counter with RELEASE semantics.

    The release ordering on the increment itself publishes this CTA's partial
    and LSE stores; a standalone `membar.gl` before it would fence every
    outstanding access of every CTA in the grid and sat on the critical path of
    all 256 of them.  The matching acquire lives in `fence_acq_gpu`.
    """
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(addr, Int64, loc, ip), _iv(v, Int32, loc, ip)],
        "atom.add.release.gpu.global.u32 $0, [$1], $2;", "=r,l,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def fence_acq_gpu(*, loc=None, ip=None):
    """Acquire half of the split-K handoff (pairs with the release above)."""
    llvm.inline_asm(llvm.StructType.get_literal([]), [], "fence.acquire.gpu;", "",
                    has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)


@dsl_user_op
def mapa_shared(addr: Int32, rank: Int32, *, loc=None, ip=None) -> Int32:
    """Translate one local shared address into cluster rank `rank`'s window."""
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(addr, Int32, loc, ip), _iv(rank, Int32, loc, ip)],
        "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def ld_dsmem_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [_iv(addr, Int32, loc, ip)],
        "ld.shared::cluster.f32 $0, [$1];", "=f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def ld_dsmem_v4_f32(addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [_iv(addr, Int32, loc, ip)],
        "ld.shared::cluster.v4.f32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def cluster_arrive(*, loc=None, ip=None):
    llvm.inline_asm(llvm.StructType.get_literal([]), [],
                    "barrier.cluster.arrive.aligned;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=_ATT)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None):
    llvm.inline_asm(llvm.StructType.get_literal([]), [],
                    "barrier.cluster.wait.aligned;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=_ATT)


@dsl_user_op
def st_global_u32(addr: Int64, v: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        llvm.StructType.get_literal([]),
        [_iv(addr, Int64, loc, ip), _iv(v, Int32, loc, ip)],
        "st.global.cg.b32 [$0], $1;", "l,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)


@dsl_user_op
def ld_global_cg_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """L1-bypassing load: the producer of these bytes is a DIFFERENT CTA."""
    return Float32(llvm.inline_asm(
        T.f32(), [_iv(addr, Int64, loc, ip)],
        "ld.global.cg.f32 $0, [$1];", "=f,l",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT))


@dsl_user_op
def ld_global_cg_v4_f32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [_iv(addr, Int64, loc, ip)],
        "ld.global.cg.v4.f32 {$0,$1,$2,$3}, [$4];",
        "=f,=f,=f,=f,l", has_side_effects=True, is_align_stack=False,
        asm_dialect=_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def st_global_v4_b32(addr: Int64, r0, r1, r2, r3, *, loc=None, ip=None):
    llvm.inline_asm(
        llvm.StructType.get_literal([]),
        [_iv(addr, Int64, loc, ip), _iv(r0, Int32, loc, ip),
         _iv(r1, Int32, loc, ip), _iv(r2, Int32, loc, ip),
         _iv(r3, Int32, loc, ip)],
        "st.global.v4.b32 [$0], {$1,$2,$3,$4};", "l,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)


@dsl_user_op
def st_global_v2_b32(addr: Int64, r0, r1, *, loc=None, ip=None):
    llvm.inline_asm(
        llvm.StructType.get_literal([]),
        [_iv(addr, Int64, loc, ip), _iv(r0, Int32, loc, ip),
         _iv(r1, Int32, loc, ip)],
        "st.global.v2.b32 [$0], {$1,$2};", "l,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)


@dsl_user_op
def st_global_v4_f32(addr: Int64, f0, f1, f2, f3, *, loc=None, ip=None):
    llvm.inline_asm(
        llvm.StructType.get_literal([]),
        [_iv(addr, Int64, loc, ip), _iv(f0, Float32, loc, ip),
         _iv(f1, Float32, loc, ip), _iv(f2, Float32, loc, ip),
         _iv(f3, Float32, loc, ip)],
        "st.global.v4.f32 [$0], {$1,$2,$3,$4};", "l,f,f,f,f",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)


@dsl_user_op
def st_global_f32(addr: Int64, f0, *, loc=None, ip=None):
    llvm.inline_asm(
        llvm.StructType.get_literal([]),
        [_iv(addr, Int64, loc, ip), _iv(f0, Float32, loc, ip)],
        "st.global.f32 [$0], $1;", "l,f",
        has_side_effects=True, is_align_stack=False, asm_dialect=_ATT)


@dsl_user_op
def pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [_iv(lo, Float32, loc, ip), _iv(hi, Float32, loc, ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=_ATT))


# ==========================================================================
# Geometry.  head_dim is the ONE compile-time constant the op allows
# (MSA is head_dim-128 only).  Everything else -- head counts, top-k,
# page_size, block-table width, batch, seqlen_q and every byte stride or
# page offset derived from them -- is a RUNTIME argument.
# ==========================================================================
HEAD_DIM = 128                    # the MSA API contract fixes head_dim at 128
HDP = HEAD_DIM // 2               # packed e2m1 bytes per (token, kv head)
SVEC = 16                         # NVFP4 scale vector length
SD = HEAD_DIM // SVEC             # e4m3 block scales per (token, kv head)
SGRP = SD // 4                    # (4,4) V-scale swizzle group

# Tile shape / unroll factors (allowed to be compile-time).
NTOK = 64                         # KV tokens staged in SMEM per pass
NWARP = 4
NTHREAD = NWARP * 32
TOKW = NTOK // NWARP              # 16 KV tokens per warp == MMA K-step
MMA_M = 16                        # m16n8k16 M
ND = HEAD_DIM // 8                # 16 N-tiles of the PV MMA
NK = HEAD_DIM // MMA_M            # 8 K-tiles of the QK MMA
SRS = HEAD_DIM + 8                # SMEM row stride in halves (pad kills LDS conflicts)
SRB = SRS * 2                     # ... in bytes
OUT_SKEW = 4                      # fp32 words inserted after each 32 dims
OUT_SRS = HEAD_DIM + (HEAD_DIM // 32) * OUT_SKEW + OUT_SKEW
OUT_SKEW_U = 1                    # scored-unsplit scalar epilogue only
OUT_SRS_U = HEAD_DIM + (HEAD_DIM // 32) * OUT_SKEW_U
# ---- cross-warp epilogue reduction, and the SHARED-MEMORY BUDGET ----------
# The four warps' fp32 output partials are reduced through a staging buffer in
# `redh` passes, and that buffer is the largest single SMEM consumer.  SMEM is
# what decides residency here: a 128-thread CTA holds one warp per
# sub-partition, so CTAs/SM = min(512 // regs_per_thread, carveout // smem).
# The driver hands this kernel the 64 KiB carveout, so a 31 KiB CTA is capped
# at TWO however few registers it uses -- which is why a 168-register build
# alone never bought its third CTA.  Staging in four passes instead of two
# halves the buffer, brings a CTA under 21 KiB and lets THREE be resident
# (12 warps/SM against 8), for twelve extra barriers -- ~1% of a CTA.
#   redh passes => RED_HALF = ND*4//redh values staged per pass.
# RED_HALF // NWARP must stay a MULTIPLE OF 4, so that the value a warp picks
# up keeps its position inside its own 4-value accumulator group and the `rr`
# / `dim` decode below stays a compile-time constant.
REDH_LOW = 2                      # staged-V builds keep the two-pass buffer
REDH_REG = 2                      # register-V builds stage in two
RED_RS_LOW = ND * 4 // REDH_LOW + 1
RED_RS_REG = ND * 4 // REDH_REG + 1
# SMEM rows of the fp16 KV slab the reduction buffer is overlaid on.
KV_ROWS_LOW = NWARP * TOKW
KV_ROWS_REG = (NWARP * 32 * RED_RS_REG * 4 + SRS * 2 - 1) // (SRS * 2)
# UNREACHABLE ON THE ROUTED PATH, and stated so rather than left to be
# rediscovered.  `_validate` refuses above this, but `plan()` only returns a
# compiled instantiation when `scored_geom` holds, and `scored_geom` pins
# `topk == 16`; every other top-k is declined here and served by the parametric
# CUDA family, whose own ceiling is `general::kSelectedCapacity == 32` (one
# selection slot per lane of warp 0's compaction ballot).  So the binding
# top-k constraint of the ROUTE is 32, not this constant.  128 is what this
# kernel body could address if a non-scored instantiation were ever compiled:
# the SMEM budget below reserves `2 * MAX_TOPK * 4` bytes for the compacted
# page/block lists, which is the only place it is load-bearing.
MAX_TOPK = 128                    # hard supported bound; the entry point REFUSES above it
SCORED_TOPK = 16
WIDE_NTG = 4                      # 8-token MMA groups per streaming chunk on
                                  # the register-rich 2-CTA/SM split binaries
# BOUNDED block-table prefix staged in SMEM, one 512-byte cooperative round per
# CTA at EVERY deployment width.  Entries beyond it -- reachable only on a
# context deep enough to select them -- resolve through a per-entry global read.
# Neither the stage nor the loop trip count is a function of `max_blocks`,
# and no predicate over the width picks between them: the block-table row
# width is a KV-cache-manager constant of the deployment, so a path chosen
# on it would be a path for exactly one max_model_len.
PT_CAP = 128
SCORED_PT_CAP = 128
NDPART = 4                        # head_dim slices per combine CTA
NDCHUNK = HEAD_DIM // NDPART // 4  # threads per row in the combine CTA
NO_LIMIT = 1 << 24                # in-page bound the fastpage stream cannot reach
NEG_BIG = -1.0e30
NEG_CLAMP = -5.0e29
NEG_INF = float("-inf")


@cute.jit
def _oc(d: Int32) -> Int32:
    """Skew by one aligned float4 per 32 dimensions.

    Dims 32 apart land in different banks, while every logical float4 keeps
    the natural 16-byte alignment needed by DSMEM vector loads.
    """
    return d + (d >> Int32(5)) * Int32(OUT_SKEW)


@cute.jit
def _ceil_div(a: Int32, b: Int32) -> Int32:
    return (a + b - Int32(1)) // b


@cute.kernel
def _msa_partial_kernel(
    q_addr: Int64,            # (total_q, num_qo_heads, head_dim) bf16
    q2k_addr: Int64,          # (num_kv_heads, total_q, topk) i32, outer strides
                              # passed separately; only the topk dim is dense
    pt_addr: Int64,           # (batch, max_blocks) i32
    sk_addr: Int64,           # (batch,) i32
    out_addr: Int64,          # (total_q, num_qo_heads, head_dim) bf16
    op_addr: Int64,           # (n_ctas, MMA_M, head_dim) f32  split-K partials
    lse_addr: Int64,          # (n_ctas, MMA_M) f32            split-K partials
    cnt_addr: Int64,          # (n_base,) u32   split-K arrival counters
    kd_addr: Int64, ks_addr: Int64, vd_addr: Int64, vs_addr: Int64,
    page_stride: Int32,       # bytes between pages   (read off the tensor)
    dhead_stride: Int32,      # bytes between kv heads in a data plane
    shead_stride: Int32,      # bytes between kv heads in a scale plane
    total_q: Int32, num_qo_heads: Int32, num_kv_heads: Int32, grp: Int32,
    topk: Int32, page_size: Int32, max_blocks: Int32, seqlen_q: Int32,
    causal: Int32, nsplit: Int32, bpc: Int32,
    q2k_hs: Int32,            # i32 elements between kv-head planes of q2k
    q2k_ts: Int32,            # i32 elements between q rows of q2k
    qk_scale: Float32,        # softmax_scale * k_global_scale * log2(e)
    v_gs: Float32,            # v_global_scale
    lowreg: cutlass.Constexpr,
    static_nsplit: cutlass.Constexpr,
    scored_geom: cutlass.Constexpr,
    cluster_c: cutlass.Constexpr,
    ntg: cutlass.Constexpr,
    vsb: cutlass.Constexpr,
    pf: cutlass.Constexpr,
    qreg: cutlass.Constexpr,
    hoist: cutlass.Constexpr,
):
    # `ntg` is the number of 8-token MMA groups a warp consumes per streaming
    # iteration, i.e. the chunk width is TOKW_L = 8 * ntg tokens.  Widening it
    # doubles the KV bytes a warp keeps IN FLIGHT for the same double-buffer
    # depth and halves both the loop trip count and the per-token ldmatrix
    # count, because one Q A-fragment now feeds twice as many QK n-tiles.
    TOKW_L = 8 * ntg
    NSFW = (ntg + 1) // 2
    # page_size 128 against a 32-token chunk and four warps means a page holds
    # EXACTLY one chunk per warp.  The stream then never carries inside a page:
    # the token base is warp-constant for the whole loop, the next block is
    # always cur+1, and every in-page bound is satisfied by construction (the
    # highest token any lane touches is 96 + 24 + 7 = 127).  The carry walk,
    # the twelve load predicates and the token half of every address all go.
    fastpage = scored_geom and 8 * ntg * NWARP == 128
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    # Pin the scored split counts so div/mod become shifts and the unsplit
    # publication/combine arm is erased. A static_nsplit of zero preserves the
    # fully dynamic generalized path.
    if cutlass.const_expr(static_nsplit > 0):
        nsplit = Int32(static_nsplit)
    # The production model geometry (heads, GQA group, top-k, page size, one
    # decode token per request, causal) is folded to constants in this
    # instantiation; the dispatch below guards every one of those axes and
    # falls back to the fully dynamic binary otherwise, so the kernel remains
    # CALLABLE at every geometry.  `max_blocks` is DELIBERATELY ABSENT: the
    # block-table row width is a deployment constant of the KV-cache manager,
    # never a property of the call, so no path is selected on it.  It stays a
    # runtime argument on every path, and the block-table
    # row is read through a BOUNDED prefix stage with a global fallback so
    # neither the work nor the shared memory scales with it.
    if cutlass.const_expr(scored_geom):
        num_qo_heads = Int32(64)
        num_kv_heads = Int32(4)
        grp = Int32(16)
        topk = Int32(16)
        page_size = Int32(128)
        seqlen_q = Int32(1)
        causal = Int32(1)
    warp = tid // Int32(32)
    lane = tid % Int32(32)
    lane_row = lane // Int32(4)          # MMA row group  (grp index)
    lane_col = lane % Int32(4)           # MMA column pair

    # The staged-V builds need the full warp-private fp16 KV slab; the
    # register-V builds never write it and use the allocation only as the
    # epilogue reduction buffer, which four-pass staging shrinks to 8.5 KiB.
    redh = REDH_LOW
    red_rs = RED_RS_LOW
    kv_rows = KV_ROWS_LOW
    if cutlass.const_expr(not lowreg):
        redh = REDH_REG
        red_rs = RED_RS_REG
        kv_rows = KV_ROWS_REG
    red_half = ND * 4 // redh

    smem = cutlass_utils.SmemAllocator()
    # ONE fp16 KV staging tile, partitioned into NWARP warp-private slabs of
    # TOKW tokens and reused for K then V inside each token chunk.  Nothing in
    # the KV loop is shared between warps, so the loop carries no CTA barrier.
    sKV = smem.allocate_tensor(cutlass.Float16,
                               cute.make_layout((kv_rows, SRS),
                                                stride=(SRS, 1)),
                               byte_alignment=512)
    # Q is consumed by ldmatrix inside the KV loop and is dead by the time the
    # epilogue writes output, so the fp16 Q tile is overlaid on the fp32 output
    # tile.  Keeping Q in SMEM rather than in 32 live registers is what buys the
    # register budget for the DOUBLE-BUFFERED packed-K prefetch below.
    out_srs = SRS
    if cutlass.const_expr(not lowreg):
        out_srs = OUT_SRS
        if cutlass.const_expr(static_nsplit == 1 and scored_geom
                              and cluster_c == 0):
            out_srs = OUT_SRS_U
    sOut = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((MMA_M, out_srs), stride=(out_srs, 1)),
        byte_alignment=512)
    sQ = cute.make_tensor(
        cute.make_ptr(cutlass.Float16, sOut.iterator.toint(),
                      mem_space=sOut.iterator.memspace, assumed_align=16),
        cute.make_layout((MMA_M, SRS), stride=(SRS, 1)))
    id_cap = MAX_TOPK
    pt_cap = PT_CAP
    if cutlass.const_expr(scored_geom):
        id_cap = SCORED_TOPK
        pt_cap = SCORED_PT_CAP
    sPg = smem.allocate_tensor(cutlass.Int32, cute.make_layout((id_cap,)),
                               byte_alignment=16)
    sId = smem.allocate_tensor(cutlass.Int32, cute.make_layout((id_cap,)),
                               byte_alignment=16)
    sMisc = smem.allocate_tensor(cutlass.Float32,
                                 cute.make_layout((2 * NWARP * MMA_M + 4,)),
                                 byte_alignment=16)
    sCnt = smem.allocate_tensor(cutlass.Int32, cute.make_layout((4,)),
                                byte_alignment=16)
    # This request's whole block-table row, staged cooperatively so that the
    # q2k read and the page-table read are issued in the SAME shadow instead
    # of forming a dependent DRAM pair on the CTA-serial critical path.
    sPT = smem.allocate_tensor(cutlass.Int32, cute.make_layout((pt_cap,)),
                               byte_alignment=16)

    kv_base32 = sKV.iterator.toint()
    q_base32 = sQ.iterator.toint()
    warp_kv = kv_base32 + warp * Int32(TOKW * SRB)

    # ---- decode the tile id ------------------------------------------------
    split = bid % nsplit
    rest = bid // nsplit
    h = rest % num_kv_heads
    qi = rest // num_kv_heads
    b = qi // seqlen_q
    sq = qi - b * seqlen_q

    seq = ld_global_s32(sk_addr + Int64(4) * Int64(b))
    # This request's OWN block count, additionally held inside the row the
    # caller actually gave us.  `max_blocks >= cdiv(seqused_k, page_size)` is a
    # property of a well-formed call, not a deployment fingerprint, so bounding
    # by it is legal -- and it means a caller whose row is too short for its own
    # seqused_k can never make a selection index off the end of that row.
    nblk = _ceil_div(seq, page_size)
    if nblk > max_blocks:
        nblk = max_blocks
    col_limit = seq - Int32(1)
    if causal != Int32(0):
        col_limit = sq + seq - seqlen_q

    # ---- co-issued selection list + block-table row ------------------------
    # The old form read q2k and then dereferenced page_table with the value it
    # had just loaded: two DEPENDENT DRAM round trips, both on the CTA-serial
    # critical path, and the other three warps idled through both of them.
    # The block-table row is a small request-local array whose address does NOT
    # depend on the selection, so it is staged in the SAME shadow as the q2k
    # read and the chain collapses to a single round trip.
    # Strides, not `total_q * topk`: the selection tensor's two OUTER extents
    # are whatever the caller's view has.  This is what lets a consumer hand in
    # `topk[:nd].transpose(0, 1)` instead of a contiguous copy of it.
    q2k_base = h * q2k_hs + qi * q2k_ts
    rIdv = cute.make_rmem_tensor((1,), cutlass.Int32)
    rIdv[0] = Int32(-1)
    if lane < topk:
        # Every warp reads the same <= 128 B of q2k, so the redundant lines
        # coalesce behind one L1 miss -- and the compaction below then needs no
        # barrier of its own, because each warp consumes only what it wrote.
        rIdv[0] = ld_global_s32(q2k_addr + Int64(4) * Int64(q2k_base + lane))
    # BOUNDED PREFIX STAGE.  At most `pt_cap` entries of the
    # row are staged whatever the deployment width is -- the shared memory, the
    # register footprint and the number of loads are all independent of
    # `max_blocks`.  Selections that land past the staged prefix (only possible
    # on a context deep enough to need them) resolve through a per-entry global
    # read, so a 2048-wide table costs exactly what a 128-wide one costs for
    # the same useful work.
    pt_lim = max_blocks if max_blocks < Int32(pt_cap) else Int32(pt_cap)
    pt_row = pt_addr + Int64(4) * Int64(b) * Int64(max_blocks)
    for c in cutlass.range_constexpr(pt_cap // NTHREAD):
        j = tid + Int32(c * NTHREAD)
        if j < pt_lim:
            sPT[j] = ld_global_s32(pt_row + Int64(4) * Int64(j))

    # ---- Q global read ------------------------------------------------------
    # Issued AFTER the selection list and the block-table row, and consumed
    # only once the first KV chunk is already in flight.  All three are in the
    # same shadow, but the CTA-serial chain that gates every later load is
    # q2k / page_table -> validity scan -> KV, and Q is 4 KiB per CTA against
    # the block-table row's 512 B.  Letting Q enter the queue first put the
    # BIGGEST request ahead of the one the whole prologue waits on.
    q_row_base = (qi * num_qo_heads + h * grp) * Int32(HEAD_DIM)
    rQr = cute.make_rmem_tensor((8,), cutlass.Int32)
    for it in cutlass.range_constexpr(2):
        lin = tid + Int32(it * NTHREAD)
        qrow = lin // Int32(16)
        dchunk = lin % Int32(16)
        w0 = Int32(0)
        w1 = Int32(0)
        w2 = Int32(0)
        w3 = Int32(0)
        if qrow < grp:
            r0, r1, r2, r3 = ld_global_v4_b32(q_addr + Int64(2) * Int64(
                q_row_base + qrow * Int32(HEAD_DIM) + dchunk * Int32(8)))
            w0 = bf16x2_to_f16x2(r0)
            w1 = bf16x2_to_f16x2(r1)
            w2 = bf16x2_to_f16x2(r2)
            w3 = bf16x2_to_f16x2(r3)
        rQr[it * 4 + 0] = w0
        rQr[it * 4 + 1] = w1
        rQr[it * 4 + 2] = w2
        rQr[it * 4 + 3] = w3

    # ---- Q -> fp16 in SMEM.  The GLOBAL half is issued above, before the
    # CTA-serial validity scan, so warp 0 pays one DRAM round trip for the
    # q2k -> page_table chain instead of that chain PLUS its own Q fetch.
    # K never reaches shared memory: every lane pulls its OWN 16 packed bytes
    # straight from the page into the QK MMA's B fragment.  That is legal
    # because the QK contraction index is head_dim, so ANY bijection of the
    # 128 dims onto the (k-step, k-slot) grid computes the same dot product as
    # long as Q uses the SAME bijection.  Q is written to SMEM under that
    # permutation here -- once per CTA, in the prologue -- and the K side then
    # costs zero shared-memory stores and zero ldmatrix.
    #   lane holds dims 32c..32c+31 (c = lane % 4); slot s of k-step j carries
    #   dim  32*(s>>1) + 4j + (s&1)                      for s < 8
    #        32*((s-8)>>1) + 4j + 2 + (s&1)              for s >= 8
    # so logical dim d = 32c + 4j + m sits at SMEM half position
    #   p = 16j + 2c + (m&1) + 8*(m>>1).
    # Only the block-table row has to be published before the scan; the Q
    # permutation is deferred until the first KV chunk is in flight.
    cute.arch.sync_threads()

    # ---- validity scan + de-duplication + compaction -----------------------
    # A selected id is usable only when it is inside THIS request's own block
    # count AND its page-table entry is non-negative.  `id >= 0` is not a
    # sufficient test: a CUDA-graph padding slot keeps a stale, in-range id
    # whose whole block-table row is -1, and an evicted block leaves -1 at a
    # STRICTLY INTERIOR position.  Entries that fail are DROPPED (compacted
    # away), never clamped and never used to terminate the scan -- the list is
    # not required to be sorted or -1-tail-packed.
    # Every warp runs the identical scan on identical inputs and writes the
    # identical result, so the compacted list needs no publication barrier.
    rcnt = cute.make_rmem_tensor((2,), cutlass.Int32)
    rcnt[0] = Int32(0)
    nchunk = _ceil_div(topk, Int32(32))
    for c in cutlass.range(nchunk):
        j = c * Int32(32) + lane
        idv = rIdv[0]
        if c != Int32(0):
            idv = Int32(-1)
            if j < topk:
                idv = ld_global_s32(q2k_addr + Int64(4) * Int64(q2k_base + j))
        ok0 = (idv >= Int32(0)) and (idv < nblk)
        pg = Int32(-1)
        if ok0:
            if idv < pt_lim:
                pg = sPT[idv]
            else:
                pg = ld_global_s32(pt_row + Int64(4) * Int64(idv))
        rok = cute.make_rmem_tensor((2,), cutlass.Int32)
        rok[0] = Int32(0)
        if ok0 and (pg >= Int32(0)):
            rok[0] = Int32(1)
        # A duplicate id must be counted ONCE, as the reference does.  The
        # filter runs on EVERY binary, including the ones the timed rows take:
        # a correctness path that is dead on the evaluated shapes is not a
        # correctness path.  It costs one `match.any.sync` plus a scan of the
        # already-accepted prefix, once per CTA in the prologue and never in
        # the streaming loop; at topk <= 32 the prior-chunk scan is provably
        # empty (one chunk) and folds away.
        okmask = cute.arch.vote_ballot_sync(rok[0] != Int32(0))
        same = match_any_b32(idv)
        if (same & okmask & cute.arch.lanemask_lt()) != Int32(0):
            rok[0] = Int32(0)
        for t in cutlass.range(rcnt[0]):
            if sId[t] == idv:
                rok[0] = Int32(0)
        okmask = cute.arch.vote_ballot_sync(rok[0] != Int32(0))
        pos = rcnt[0] + cute.arch.popc(okmask & cute.arch.lanemask_lt())
        if rok[0] != Int32(0):
            sId[pos] = idv
            sPg[pos] = pg
        rcnt[0] = rcnt[0] + cute.arch.popc(okmask)

    n_valid = rcnt[0]
    # Balanced split ranges over the ACTUAL valid block count: split s owns
    # blocks [s*n_valid/nsplit, (s+1)*n_valid/nsplit).  A static `split*bpc`
    # stride leaves the late splits completely empty whenever the request is
    # shorter than top-k, or entries were dropped by the validity scan.
    blk_lo = (split * n_valid) // nsplit
    blk_hi = ((split + Int32(1)) * n_valid) // nsplit

    # The normal path keeps Q fragments live across the full KV stream.  The
    # q128 specialization reloads them per chunk, trading eight ldmatrix ops
    # for the register headroom required by a fourth resident CTA.
    qldm0 = q_base32 + (lane % Int32(16)) * Int32(SRB) \
        + (lane // Int32(16)) * Int32(16)
    # Q is invariant across the KV stream.  The register-rich two-CTA family
    # materialises its eight A fragments once; this removes eight ldmatrix.x4
    # instructions (4 KiB of shared traffic) from every streaming iteration.
    # The 128-register four-CTA family continues to reload from shared memory.
    rQa = cute.make_rmem_tensor((NK * 4,), cutlass.Int32)
    rO = cute.make_rmem_tensor((ND * 4,), cutlass.Float32)
    for i in cutlass.range_constexpr(ND * 4):
        rO[i] = Float32(0.0)
    rM = cute.make_rmem_tensor((2,), cutlass.Float32)
    rL = cute.make_rmem_tensor((2,), cutlass.Float32)
    for i in cutlass.range_constexpr(2):
        rM[i] = Float32(NEG_BIG)
        rL[i] = Float32(0.0)
    rS = cute.make_rmem_tensor((4 * ntg,), cutlass.Float32)
    rK = cute.make_rmem_tensor((4 * ntg,), cutlass.Int32)
    rKn = cute.make_rmem_tensor((4 * ntg,), cutlass.Int32)
    rV = cute.make_rmem_tensor((4 * ntg,), cutlass.Int32)
    rVn = cute.make_rmem_tensor((4 * ntg,), cutlass.Int32)
    # K packs four scale bytes per word; V keeps two adjacent token scales in
    # each word so the PV fragment can be transposed entirely in registers.
    rSf = cute.make_rmem_tensor((2 * NSFW,), cutlass.Int32)  # K current / next
    rVs = cute.make_rmem_tensor((ntg,), cutlass.Int32)
    rVsn = cute.make_rmem_tensor((ntg,), cutlass.Int32)

    # The four-CTA/SM binaries retain the lower-register staged-V path.  Other
    # binaries use the direct paired-token fragments below.
    dq_tok = lane % Int32(8)
    dq_c = lane // Int32(8)
    # K's fragment-direct coordinates: lane l owns token l/4 of the 8-token
    # half and its 16 CONTIGUOUS packed bytes (dims 32c .. 32c+31, c = l%4).
    # Address = base + 16*lane, i.e. one perfectly coalesced 512-byte warp
    # transaction per half -- strictly better coalescing than the staged path.
    # ---- token PERMUTATION inside each 8-token MMA group -------------------
    # The PV B fragment forces lane l to own head-dim slice (l>>2) of the token
    # PAIR (l&3), so one V load instruction always touches four tokens at once
    # and the only freedom left is WHICH four.  With the natural pairing
    # (2t, 2t+1) those four are tokens {0,2,4,6} -- 64 useful bytes out of each
    # of FOUR 128-byte lines, i.e. four half-empty L1 wavefronts per load.
    # The QK contraction fixes only the SET of tokens in an n-tile, not their
    # order, so pairing k-index j with token perm(j) = 4*(j&1) + (j>>1) makes
    # each V load land on four CONSECUTIVE tokens: 256 contiguous bytes, TWO
    # full wavefronts.  V data wavefronts per chunk halve (32 -> 16); the only
    # cost is that the token pair's two e4m3 V scales are now 32 bytes apart,
    # so one u16 becomes two broadcast u32 reads plus a prmt.
    kf_tok = ((lane // Int32(4)) % Int32(2)) * Int32(4) + lane // Int32(8)
    kf_boff = (lane % Int32(4)) * Int32(16)
    kf_soff = (lane % Int32(4)) * Int32(2)
    # V's PV-B fragment pairs tokens (t, t+4) of the group. lane_row selects an
    # 8-byte head-dimension slice; lane_col selects one of four token pairs.
    vd_off = lane_row * Int32(8)
    # (4,4)-swizzled V scale: byte(t, d) = (t>>2)*32 + (d>>1)*8 + (d&1)*4 + (t&3).
    # Tokens 0..3 of a group therefore occupy FOUR CONTIGUOUS bytes at this
    # offset, and tokens 4..7 the same four bytes 32 further on.
    vs_boff = (lane_row // Int32(2)) * Int32(SD) + (lane_row % Int32(2)) * Int32(4)
    vs_sel = lane_col | ((lane_col + Int32(4)) << Int32(4))
    # ---- invariant halves of every KV address ------------------------------
    # A chunk's twelve loads differ from each other only by CONSTANTS, and from
    # the previous chunk only by (page, token base).  Folding the lane's own
    # displacement into the plane pointer ONCE leaves exactly one 64-bit add
    # per plane per chunk; the per-load displacement rides in the instruction.
    # The lane's whole displacement inside a plane is loop-invariant, but it is
    # kept as a 32-bit addend of the EXISTING plane pointer rather than folded
    # into four extra 64-bit bases: at the 128-register/4-CTA cliff those eight
    # permanently live registers cost more occupancy than the arithmetic they save.
    h_dh = h * dhead_stride
    h_sh = h * shead_stride
    kd_lane = kf_tok * Int32(HDP) + kf_boff
    ks_lane = kf_tok * Int32(SD) + kf_soff
    vd_lane = lane_col * Int32(HDP) + vd_off
    vs_lane = vs_boff
    lv_lane = dq_tok * Int32(HDP) + dq_c * Int32(16)
    # (t>>2)*4 is token-base + (dq_tok & ~3) because every chunk and task base
    # is a multiple of four, so the staged path's swizzled scale offset and its
    # byte shift are both lane-invariant too.
    ls_lane = ((dq_tok // Int32(4)) * Int32(4) + dq_c) * Int32(SD)
    lv_sh = (dq_tok % Int32(4)) * Int32(8)
    w_tok0 = warp * Int32(TOKW_L)
    if cutlass.const_expr(fastpage and hoist):
        # ... and the warp's own token base, being loop-invariant, joins them.
        kd_lane = kd_lane + w_tok0 * Int32(HDP)
        ks_lane = ks_lane + w_tok0 * Int32(SD)
        vd_lane = vd_lane + w_tok0 * Int32(HDP)
        vs_lane = vs_lane + w_tok0 * Int32(SD)
        lv_lane = lv_lane + w_tok0 * Int32(HDP)
        ls_lane = ls_lane + w_tok0 * Int32(SD)
    if cutlass.const_expr(hoist):
        # Register-rich tiers fold the lane displacement AND the kv-head plane
        # offset into four permanent 64-bit bases, so a chunk costs one add per
        # plane.  The four-resident-CTA tier cannot: it is exactly at the
        # 128-register cliff and those eight permanent registers cost a whole CTA.
        kd_hl = kd_addr + Int64(h_dh + kd_lane)
        ks_hl = ks_addr + Int64(h_sh + ks_lane)
        vd_hl = vd_addr + Int64(h_dh + vd_lane)
        vs_hl = vs_addr + Int64(h_sh + vs_lane)
        lv_hl = vd_addr + Int64(h_dh + lv_lane)
        ls_hl = vs_addr + Int64(h_sh + ls_lane)
    # ... and the guards become one subtract per chunk plus an immediate
    # compare, instead of an add and a compare on every one of the twelve.
    pk_lim = page_size - kf_tok
    pv_lim = page_size - lane_col
    pd_lim = page_size - dq_tok
    cpp = _ceil_div(page_size, Int32(TOKW_L))     # token chunks per page
    # The staged-V path writes SMEM row perm^{-1}(token) so its ldmatrix.trans
    # fragment sees the same k-index order the QK accumulator produced.
    smem_row0 = warp_kv \
        + (Int32(2) * (dq_tok % Int32(4)) + dq_tok // Int32(4)) * Int32(SRB) \
        + dq_c * Int32(64)
    ldm0 = warp_kv + (lane % Int32(16)) * Int32(SRB) \
        + (lane // Int32(16)) * Int32(16)

    # ---- flatten (block, token chunk) into ONE warp-strided chunk stream ---
    # so that the global loads of chunk i+1 are issued in the middle of chunk
    # i's tensor-core work and their DRAM latency is paid while the MMAs run.
    # The prefetch reuses the SAME registers the current chunk just vacated,
    # so the pipeline costs no extra register pressure.
    nch_tot = (blk_hi - blk_lo) * cpp
    my_n = Int32(0)
    if nch_tot > warp:
        my_n = (nch_tot - warp + Int32(NWARP - 1)) // Int32(NWARP)
    it_blk = blk_lo
    it_ch = warp
    if cutlass.const_expr(not fastpage):
        for _ in cutlass.range_constexpr(NWARP):
            if it_ch >= cpp:
                it_ch = it_ch - cpp
                it_blk = it_blk + Int32(1)
    rIt = cute.make_rmem_tensor((2,), cutlass.Int32)
    rIt[0] = it_blk
    rIt[1] = it_ch

    if my_n > Int32(0):
        pg = sPg[it_blk]
        tok0 = it_ch * Int32(TOKW_L)
        pg_off = Int64(pg) * Int64(page_stride)
        if cutlass.const_expr(hoist):
            if cutlass.const_expr(fastpage):
                tdo = pg_off
                tso = pg_off
            else:
                tdo = pg_off + Int64(tok0 * Int32(HDP))
                tso = pg_off + Int64(tok0 * Int32(SD))
            kd_p = kd_hl + tdo
            ks_p = ks_hl + tso
            vd_p = vd_hl + tdo
            vs_p = vs_hl + tso
            lv_p = lv_hl + tdo
            ls_p = ls_hl + tso
        else:
            td = h_dh + tok0 * Int32(HDP)
            ts = h_sh + tok0 * Int32(SD)
            kd_p = kd_addr + pg_off + Int64(td + kd_lane)
            ks_p = ks_addr + pg_off + Int64(ts + ks_lane)
            vd_p = vd_addr + pg_off + Int64(td + vd_lane)
            vs_p = vs_addr + pg_off + Int64(ts + vs_lane)
            lv_p = vd_addr + pg_off + Int64(td + lv_lane)
            ls_p = vs_addr + pg_off + Int64(ts + ls_lane)
        if cutlass.const_expr(fastpage):
            kl = Int32(NO_LIMIT)
            vl = Int32(NO_LIMIT)
            dl = Int32(NO_LIMIT)
        else:
            kl = pk_lim - tok0
            vl = pv_lim - tok0
            dl = pd_lim - tok0
        for wsf in cutlass.range_constexpr(NSFW):
            ksfp = Int32(0)
            for sub in cutlass.range_constexpr(2):
                task = wsf * 2 + sub
                kw0 = Int32(0)
                kw1 = Int32(0)
                kw2 = Int32(0)
                kw3 = Int32(0)
                ksf = Int32(0)
                if Int32(task * 8) < kl:
                    kw0, kw1, kw2, kw3 = ld_global_v4_b32(
                        kd_p, task * 8 * HDP)
                    ksf = ld_global_u16(ks_p, task * 8 * SD)
                rK[task * 4 + 0] = kw0
                rK[task * 4 + 1] = kw1
                rK[task * 4 + 2] = kw2
                rK[task * 4 + 3] = kw3
                ksfp = ksfp | (ksf << Int32(sub * 16))
            rSf[wsf] = ksfp
        if cutlass.const_expr(lowreg):
            vsfp = Int32(0)
            for task in cutlass.range_constexpr(2):
                vw0 = Int32(0)
                vw1 = Int32(0)
                vw2 = Int32(0)
                vw3 = Int32(0)
                vs0 = Int32(0)
                vs1 = Int32(0)
                if Int32(task * 8) < dl:
                    vw0, vw1, vw2, vw3 = ld_global_v4_b32(
                        lv_p, task * 8 * HDP)
                    vlo, vhi = ld_global_v2_b32(ls_p, task * 8 * SD)
                    vs0 = vlo >> lv_sh
                    vs1 = vhi >> lv_sh
                rV[task * 4 + 0] = vw0
                rV[task * 4 + 1] = vw1
                rV[task * 4 + 2] = vw2
                rV[task * 4 + 3] = vw3
                vsfp = vsfp | ((vs0 & Int32(255)) << Int32(task * 16)) \
                    | ((vs1 & Int32(255)) << Int32(task * 16 + 8))
            rVs[0] = vsfp
        else:
            for p in cutlass.range_constexpr(2 * ntg):
                pt = (p // 2) * 8 + (p % 2) * 4
                v0 = Int32(0)
                v1 = Int32(0)
                if Int32(pt) < vl:
                    v0, v1 = ld_global_v2_b32(vd_p, pt * HDP)
                rV[p * 2 + 0] = v0
                rV[p * 2 + 1] = v1
            for p in cutlass.range_constexpr(ntg):
                sa = Int32(0)
                sb = Int32(0)
                if Int32(p * 8) < vl:
                    sa = ld_global_s32(vs_p, p * 8 * SD)
                if Int32(p * 8 + 4) < vl:
                    sb = ld_global_s32(vs_p, p * 8 * SD + 32)
                rVs[p] = prmt_b32(sa, sb, vs_sel)

        # ---- warm L2 for the chunk the register pipeline cannot reach ------
        # The register shadow is one chunk deep and the in-loop L2 prefetch
        # runs one chunk BEYOND it, so its first target is chunk 2.  Chunk 1
        # is issued into the shadow at the top of iteration 0 with only that
        # one loop body to hide a COLD round trip behind -- the harness
        # flushes L2 before every call, so nothing in the whole prologue is
        # resident.  A warp that streams only two or four chunks (every split
        # row: b8 streams two) therefore pays a fully exposed DRAM miss over
        # a quarter to a half of its loop.  Warming chunks 1 and 2 here is
        # eight instructions once per CTA and carries no register state.
        if cutlass.const_expr(pf):
            pwh = Int64(h * dhead_stride)
            pwj = Int64(h * shead_stride)
            # ONE chunk only.  Chunk 2 onward is already covered by the loop's
            # own prefetch from iteration 0; adding it here only enlarges the
            # start-up burst that the real chunk-0 and chunk-1 loads are
            # queued behind, which measured as a loss on the long streams and
            # no extra gain on the short ones.
            for pre in cutlass.range_constexpr(1):
                if cutlass.const_expr(fastpage):
                    # One chunk per warp per page: chunk k is simply block k
                    # further on at this warp's own, fixed, token base.
                    pw_blk = it_blk + Int32(pre + 1)
                    if pw_blk < blk_hi:
                        pwo = Int64(sPg[pw_blk]) * Int64(page_stride)
                        pwd = Int64(tok0 * Int32(HDP)
                                    + lane * Int32(TOKW_L * HDP // 32))
                        pws = Int64(tok0 * Int32(SD)
                                    + lane * Int32(TOKW_L * SD // 32))
                        prefetch_l2(kd_addr + pwo + pwh + pwd)
                        prefetch_l2(vd_addr + pwo + pwh + pwd)
                        prefetch_l2(ks_addr + pwo + pwj + pws)
                        prefetch_l2(vs_addr + pwo + pwj + pws)
                else:
                    pw_blk = it_blk
                    pw_ch = it_ch + Int32((pre + 1) * NWARP)
                    for _ in cutlass.range_constexpr(2 * NWARP):
                        if pw_ch >= cpp:
                            pw_ch = pw_ch - cpp
                            pw_blk = pw_blk + Int32(1)
                    pw_tok = pw_ch * Int32(TOKW_L)
                    # The lane stride tiles the chunk's own byte range exactly,
                    # so a prefetch inside the page can never reach past this
                    # kv head's plane.
                    if (pw_blk < blk_hi) and \
                            (pw_tok + Int32(TOKW_L) <= page_size):
                        pwo = Int64(sPg[pw_blk]) * Int64(page_stride)
                        pwd = Int64(pw_tok * Int32(HDP)
                                    + lane * Int32(TOKW_L * HDP // 32))
                        pws = Int64(pw_tok * Int32(SD)
                                    + lane * Int32(TOKW_L * SD // 32))
                        prefetch_l2(kd_addr + pwo + pwh + pwd)
                        prefetch_l2(vd_addr + pwo + pwh + pwd)
                        prefetch_l2(ks_addr + pwo + pwj + pws)
                        prefetch_l2(vs_addr + pwo + pwj + pws)

    # ---- Q -> fp16 in SMEM, under the QK B-fragment permutation -----------
    # Deferred to HERE, after the first KV chunk has been issued, so the 4 KiB
    # Q fetch is retired off the CTA-serial critical path: the scan now waits
    # only on the block-table row, and the first KV round trip is issued a
    # whole Q round trip earlier than before.
    # K never reaches shared memory: every lane pulls its OWN 16 packed bytes
    # straight from the page into the QK MMA's B fragment.  That is legal
    # because the QK contraction index is head_dim, so ANY bijection of the
    # 128 dims onto the (k-step, k-slot) grid computes the same dot product as
    # long as Q uses the SAME bijection.
    #   lane holds dims 32c..32c+31 (c = lane % 4); slot s of k-step j carries
    #   dim  32*(s>>1) + 4j + (s&1)                      for s < 8
    #        32*((s-8)>>1) + 4j + 2 + (s&1)              for s >= 8
    # so logical dim d = 32c + 4j + m sits at SMEM half position
    #   p = 16j + 2c + (m&1) + 8*(m>>1).
    for it in cutlass.range_constexpr(2):
        lin = tid + Int32(it * NTHREAD)
        qrow = lin // Int32(16)
        dchunk = lin % Int32(16)
        qpb = qrow * Int32(SRB) + (dchunk % Int32(4)) * Int32(64) \
            + (dchunk // Int32(4)) * Int32(4)
        st_shared_b32(q_base32 + qpb, rQr[it * 4 + 0])
        st_shared_b32(q_base32 + qpb + Int32(16), rQr[it * 4 + 1])
        st_shared_b32(q_base32 + qpb + Int32(32), rQr[it * 4 + 2])
        st_shared_b32(q_base32 + qpb + Int32(48), rQr[it * 4 + 3])
    cute.arch.sync_threads()
    # Q is invariant across the KV stream.  The register-rich family
    # materialises its eight A fragments once; this removes eight ldmatrix.x4
    # instructions (4 KiB of shared traffic) from every streaming iteration.
    if cutlass.const_expr(qreg):
        for kt in cutlass.range_constexpr(NK):
            qa0, qa1, qa2, qa3 = ldmatrix_x4(
                qldm0 + Int32(kt * MMA_M * 2))
            rQa[kt * 4 + 0] = qa0
            rQa[kt * 4 + 1] = qa1
            rQa[kt * 4 + 2] = qa2
            rQa[kt * 4 + 3] = qa3

    for _it in cutlass.range(my_n):
        cur_blk = rIt[0]
        if cutlass.const_expr(fastpage):
            # One chunk per warp per page: the stream advances a whole block
            # every iteration and the token base never moves.
            tok0 = w_tok0
            nx_ch = warp
            nx_blk = cur_blk + Int32(1)
        else:
            cur_ch = rIt[1]
            tok0 = cur_ch * Int32(TOKW_L)
            # advance the stream and build the NEXT chunk's page pointers
            nx_ch = cur_ch + Int32(NWARP)
            nx_blk = cur_blk
            # One carry step always suffices when a page holds at least NWARP
            # chunks, which every production page size does; the walk survives
            # for the narrow pages only, behind a warp-uniform predicate.
            for _ in cutlass.range_constexpr(NWARP):
                if nx_ch >= cpp:
                    nx_ch = nx_ch - cpp
                    nx_blk = nx_blk + Int32(1)
            rIt[1] = nx_ch
        col0 = sId[cur_blk] * page_size
        rIt[0] = nx_blk
        pb = nx_blk
        ntok0 = nx_ch * Int32(TOKW_L)
        if pb >= blk_hi:
            # past the end of this warp's stream: keep the page pointer live
            # but push the token base out of range, so every load of the tail
            # iteration is predicated off instead of fetching a junk chunk.
            pb = blk_hi - Int32(1)
            if cutlass.const_expr(not fastpage):
                ntok0 = page_size
            # fastpage instead re-reads the LAST block's own chunk: an in-page,
            # already-resident address whose result the loop never consumes,
            # which is cheaper than predicating twelve loads off.
        # ---- L2 prefetch, one chunk BEYOND the register pipeline -----------
        if cutlass.const_expr(pf):
            # The register pipeline is exactly one chunk deep, so chunk i+1's DRAM
            # round trip has only this one loop body to hide behind -- and with a
            # cold L2 (the harness flushes it before every iteration) that round
            # trip is longer than the body, so the warp stalls at the top of every
            # iteration.  Deepening the register pipeline costs a whole shadow
            # register set per stage, which the 128-register tier cannot buy.  An
            # L2 prefetch of chunk i+2 buys the same stage for FOUR instructions
            # and ZERO registers: the lines land in the cache, so the load that
            # does land in registers next iteration hits L2 instead of DRAM.
            if cutlass.const_expr(fastpage):
                p2_ch = nx_ch
                p2_blk = nx_blk + Int32(1)
            else:
                p2_ch = nx_ch + Int32(NWARP)
                p2_blk = nx_blk
                for _ in cutlass.range_constexpr(NWARP):
                    if p2_ch >= cpp:
                        p2_ch = p2_ch - cpp
                        p2_blk = p2_blk + Int32(1)
            # One stage only: a second prefetched stage was measured (w0 16 -> 17
            # us) to cost more in stream-coordinate arithmetic than it recovers in
            # latency -- the carry walk and the page deref, not the four prefetch
            # instructions, are what the extra stage actually buys.
            p2tok = p2_ch * Int32(TOKW_L)
            # The lane stride tiles the chunk's own byte range exactly, so the
            # prefetch can never reach past this kv head's plane in the page.
            if cutlass.const_expr(fastpage):
                do_pf = p2_blk < blk_hi
            else:
                do_pf = (p2_blk < blk_hi) \
                    and (p2tok + Int32(TOKW_L) <= page_size)
            if do_pf:
                p2off = Int64(sPg[p2_blk]) * Int64(page_stride)
                p2d = Int64(p2tok * Int32(HDP)
                            + lane * Int32(TOKW_L * HDP // 32))
                p2s = Int64(p2tok * Int32(SD) + lane * Int32(TOKW_L * SD // 32))
                hd = Int64(h * dhead_stride)
                hs = Int64(h * shead_stride)
                prefetch_l2(kd_addr + p2off + hd + p2d)
                prefetch_l2(vd_addr + p2off + hd + p2d)
                prefetch_l2(ks_addr + p2off + hs + p2s)
                prefetch_l2(vs_addr + p2off + hs + p2s)

        npg = sPg[pb]
        npg_off = Int64(npg) * Int64(page_stride)
        if cutlass.const_expr(hoist):
            if cutlass.const_expr(fastpage):
                ntdo = npg_off
                ntso = npg_off
            else:
                ntdo = npg_off + Int64(ntok0 * Int32(HDP))
                ntso = npg_off + Int64(ntok0 * Int32(SD))
            nkd_p = kd_hl + ntdo
            nks_p = ks_hl + ntso
            nvd_p = vd_hl + ntdo
            nvs_p = vs_hl + ntso
            nlv_p = lv_hl + ntdo
            nls_p = ls_hl + ntso
        else:
            ntd = h_dh + ntok0 * Int32(HDP)
            nts = h_sh + ntok0 * Int32(SD)
            nkd_p = kd_addr + npg_off + Int64(ntd + kd_lane)
            nks_p = ks_addr + npg_off + Int64(nts + ks_lane)
            nvd_p = vd_addr + npg_off + Int64(ntd + vd_lane)
            nvs_p = vs_addr + npg_off + Int64(nts + vs_lane)
            nlv_p = vd_addr + npg_off + Int64(ntd + lv_lane)
            nls_p = vs_addr + npg_off + Int64(nts + ls_lane)
        if cutlass.const_expr(fastpage):
            # Every in-page bound is satisfied by construction, so the only
            # thing left for the twelve predicates to express is "this warp
            # has run off the end of its block range".  That is WARP-UNIFORM
            # and identical for all twelve, so ptxas collapses them into one
            # uniform branch -- and the tail iteration stops re-reading a
            # chunk it will never consume, which on the split rows (two to
            # four iterations) was half the load traffic of the whole loop.
            nlim = Int32(NO_LIMIT)
            if nx_blk >= blk_hi:
                nlim = Int32(-1)
            nkl = nlim
            nvl = nlim
            ndl = nlim
        else:
            nkl = pk_lim - ntok0
            nvl = pv_lim - ntok0
            ndl = pd_lim - ntok0

        # ---- K[chunk+1] is issued HERE, at the very top of the body, into a
        # shadow register set.  It is not consumed until the QK MMA of the NEXT
        # iteration, so the DRAM round trip is covered by a full loop body plus
        # this one's V dequant and V prefetch -- roughly 1.5x the exposure the
        # single-buffered form could hide.  Holding Q in SMEM instead of in 32
        # registers is what pays for the extra eight live registers.
        for wsf in cutlass.range_constexpr(NSFW):
            ksfp = Int32(0)
            for sub in cutlass.range_constexpr(2):
                task = wsf * 2 + sub
                kw0 = Int32(0)
                kw1 = Int32(0)
                kw2 = Int32(0)
                kw3 = Int32(0)
                ksf = Int32(0)
                if Int32(task * 8) < nkl:
                    kw0, kw1, kw2, kw3 = ld_global_v4_b32(
                        nkd_p, task * 8 * HDP)
                    ksf = ld_global_u16(nks_p, task * 8 * SD)
                rKn[task * 4 + 0] = kw0
                rKn[task * 4 + 1] = kw1
                rKn[task * 4 + 2] = kw2
                rKn[task * 4 + 3] = kw3
                ksfp = ksfp | (ksf << Int32(sub * 16))
            rSf[NSFW + wsf] = ksfp

        if cutlass.const_expr(lowreg):
            vsfp = Int32(0)
            for task in cutlass.range_constexpr(2):
                vw0 = Int32(0)
                vw1 = Int32(0)
                vw2 = Int32(0)
                vw3 = Int32(0)
                vs0 = Int32(0)
                vs1 = Int32(0)
                if Int32(task * 8) < ndl:
                    vw0, vw1, vw2, vw3 = ld_global_v4_b32(
                        nlv_p, task * 8 * HDP)
                    vlo, vhi = ld_global_v2_b32(nls_p, task * 8 * SD)
                    vs0 = vlo >> lv_sh
                    vs1 = vhi >> lv_sh
                rVn[task * 4 + 0] = vw0
                rVn[task * 4 + 1] = vw1
                rVn[task * 4 + 2] = vw2
                rVn[task * 4 + 3] = vw3
                vsfp = vsfp | ((vs0 & Int32(255)) << Int32(task * 16)) \
                    | ((vs1 & Int32(255)) << Int32(task * 16 + 8))
            rVsn[0] = vsfp
        elif cutlass.const_expr(not vsb):
            # Direct-V arrives arranged by token pairs for the PV fragments.
            for p in cutlass.range_constexpr(2 * ntg):
                pt = (p // 2) * 8 + (p % 2) * 4
                v0 = Int32(0)
                v1 = Int32(0)
                if Int32(pt) < nvl:
                    v0, v1 = ld_global_v2_b32(nvd_p, pt * HDP)
                rVn[p * 2 + 0] = v0
                rVn[p * 2 + 1] = v1
            for p in cutlass.range_constexpr(ntg):
                sa = Int32(0)
                sb = Int32(0)
                if Int32(p * 8) < nvl:
                    sa = ld_global_s32(nvs_p, p * 8 * SD)
                if Int32(p * 8 + 4) < nvl:
                    sb = ld_global_s32(nvs_p, p * 8 * SD + 32)
                rVsn[p] = prmt_b32(sa, sb, vs_sel)

        if cutlass.const_expr(lowreg):
            # The occupancy-critical path dequantises V into its warp-private
            # shared slab and uses ldmatrix.trans for the PV operand.
            for task in cutlass.range_constexpr(2):
                smem_off = smem_row0 + Int32(task * 8 * SRB)
                sfa = e4m3_byte_to_f16x2(rVs[0], 2 * task)
                sfb = e4m3_byte_to_f16x2(rVs[0], 2 * task + 1)
                o0, o1, o2, o3 = dequant_fp4x8_f16x8(rV[task * 4 + 0], sfa)
                st_shared_v4_b32(smem_off, o0, o1, o2, o3)
                o0, o1, o2, o3 = dequant_fp4x8_f16x8(rV[task * 4 + 1], sfa)
                st_shared_v4_b32(smem_off + Int32(16), o0, o1, o2, o3)
                o0, o1, o2, o3 = dequant_fp4x8_f16x8(rV[task * 4 + 2], sfb)
                st_shared_v4_b32(smem_off + Int32(32), o0, o1, o2, o3)
                o0, o1, o2, o3 = dequant_fp4x8_f16x8(rV[task * 4 + 3], sfb)
                st_shared_v4_b32(smem_off + Int32(48), o0, o1, o2, o3)


        # ---------------- QK^T, entirely out of registers --------------------
        # Every lane already holds exactly the 32 head_dim values its B
        # fragment needs (see the Q permutation above), so the sixteen QK MMAs
        # run with no shared-memory store and no ldmatrix at all: the packed
        # nibbles are dequantised straight into the operand registers.
        for i in cutlass.range_constexpr(4 * ntg):
            rS[i] = Float32(0.0)
        # Decode the four packed QK scale bytes together, sharing unpack and
        # conversion before broadcasting the resulting fp16 scalars.
        sfv = []
        for w in cutlass.range_constexpr(NSFW):
            s0, s1, s2, s3 = e4m3x4_to_dup_f16x2(rSf[w])
            sfv.append((s0, s1))
            sfv.append((s2, s3))
        for q in cutlass.range_constexpr(4):
            # packed word q covers dims 32c+8q .. +7, i.e. scale group 2c+q/2
            dqv = []
            for t in cutlass.range_constexpr(ntg):
                dqv.append(dequant_fp4x8_f16x8(
                    rK[t * 4 + q], (sfv[t][0] if q < 2 else sfv[t][1])))
            for sub in cutlass.range_constexpr(2):
                kt = 2 * q + sub
                # One Q A-fragment now drives ntg n-tiles instead of two, so
                # the ldmatrix count per KV token falls by ntg/2.
                if cutlass.const_expr(qreg):
                    a0 = rQa[kt * 4 + 0]
                    a1 = rQa[kt * 4 + 1]
                    a2 = rQa[kt * 4 + 2]
                    a3 = rQa[kt * 4 + 3]
                else:
                    a0, a1, a2, a3 = ldmatrix_x4(
                        qldm0 + Int32(kt * MMA_M * 2))
                for t in cutlass.range_constexpr(ntg):
                    d0, d1, d2, d3 = mma_m16n8k16_f16(
                        a0, a1, a2, a3,
                        (dqv[t][0] if sub == 0 else dqv[t][2]),
                        (dqv[t][1] if sub == 0 else dqv[t][3]),
                        rS[t * 4 + 0], rS[t * 4 + 1],
                        rS[t * 4 + 2], rS[t * 4 + 3])
                    rS[t * 4 + 0] = d0
                    rS[t * 4 + 1] = d1
                    rS[t * 4 + 2] = d2
                    rS[t * 4 + 3] = d3

        # ---------------- masked online softmax -----------------------------
        # k-index (2*lane_col + cc) of n-tile nt carries token perm(...) =
        # tok0 + 8*nt + 4*cc + lane_col.
        tbase = tok0 + lane_col
        # The in-page bound and the causal/length bound are BOTH monotone in
        # the column index, so they collapse into ONE limit computed once per
        # chunk -- warp-uniform work replacing a compare and an AND on every
        # one of the chunk's columns.  Nothing is dropped: `clim` is the exact
        # last legal column of this chunk's page.
        rLim = cute.make_rmem_tensor((1,), cutlass.Int32)
        rLim[0] = page_size - Int32(1)
        crem = col_limit - col0
        if crem < rLim[0]:
            rLim[0] = crem
        clim = rLim[0]
        mloc0 = Float32(NEG_BIG)
        mloc1 = Float32(NEG_BIG)
        if cutlass.const_expr(scored_geom):
            # An IEEE -inf sentinel remains masked after positive QK scaling;
            # a large finite raw sentinel would shrink toward zero and could
            # become a false maximum on a wholly masked chunk.
            mloc0 = Float32(NEG_INF)
            mloc1 = Float32(NEG_INF)
        for nt in cutlass.range_constexpr(ntg):
            for cc in cutlass.range_constexpr(2):
                tk = tbase + Int32(nt * 8 + cc * 4)
                v0 = Float32(NEG_BIG)
                v1 = Float32(NEG_BIG)
                if tk <= clim:
                    if cutlass.const_expr(scored_geom):
                        v0 = rS[nt * 4 + cc]
                        v1 = rS[nt * 4 + 2 + cc]
                    else:
                        v0 = rS[nt * 4 + cc] * qk_scale
                        v1 = rS[nt * 4 + 2 + cc] * qk_scale
                elif cutlass.const_expr(scored_geom):
                    v0 = Float32(NEG_INF)
                    v1 = Float32(NEG_INF)
                rS[nt * 4 + cc] = v0
                rS[nt * 4 + 2 + cc] = v1
                if cutlass.const_expr(lowreg):
                    mloc0 = cute.arch.fmax(mloc0, v0)
                    mloc1 = cute.arch.fmax(mloc1, v1)
        if cutlass.const_expr(not lowreg and ntg == 4):
            a00 = fmax3(rS[0], rS[1], rS[4])
            a01 = fmax3(rS[5], rS[8], rS[9])
            a02 = fmax3(rS[12], rS[13], Float32(NEG_INF))
            a10 = fmax3(rS[2], rS[3], rS[6])
            a11 = fmax3(rS[7], rS[10], rS[11])
            a12 = fmax3(rS[14], rS[15], Float32(NEG_INF))
            mloc0 = fmax3(a00, a01, a02)
            mloc1 = fmax3(a10, a11, a12)
        elif cutlass.const_expr(not lowreg):
            mloc0 = cute.arch.fmax(cute.arch.fmax(rS[0], rS[1]),
                                   cute.arch.fmax(rS[4], rS[5]))
            mloc1 = cute.arch.fmax(cute.arch.fmax(rS[2], rS[3]),
                                   cute.arch.fmax(rS[6], rS[7]))
        for off in cutlass.range_constexpr(2):
            mloc0 = cute.arch.fmax(
                mloc0, cute.arch.shuffle_sync_bfly(mloc0, Int32(1 << off)))
            mloc1 = cute.arch.fmax(
                mloc1, cute.arch.shuffle_sync_bfly(mloc1, Int32(1 << off)))
        if cutlass.const_expr(scored_geom):
            # The runtime QK scale is positive.  Max-reduce raw scores first,
            # then scale only the two row maxima; probability formation below
            # fuses each remaining scale and subtract into one FFMA.
            mloc0 = mloc0 * qk_scale
            mloc1 = mloc1 * qk_scale
        mnew0 = cute.arch.fmax(rM[0], mloc0)
        mnew1 = cute.arch.fmax(rM[1], mloc1)
        al0 = Float32(0.0)
        al1 = Float32(0.0)
        if _it != Int32(0):
            al0 = cute.arch.exp2(rM[0] - mnew0)
            al1 = cute.arch.exp2(rM[1] - mnew1)
        rM[0] = mnew0
        rM[1] = mnew1
        me0 = cute.arch.fmax(mnew0, Float32(NEG_CLAMP))
        me1 = cute.arch.fmax(mnew1, Float32(NEG_CLAMP))
        ps0 = Float32(0.0)
        ps1 = Float32(0.0)
        for nt in cutlass.range_constexpr(ntg):
            for cc in cutlass.range_constexpr(2):
                if cutlass.const_expr(scored_geom):
                    p0 = cute.arch.exp2(fma_rn_f32(
                        rS[nt * 4 + cc], qk_scale, -me0))
                    p1 = cute.arch.exp2(fma_rn_f32(
                        rS[nt * 4 + 2 + cc], qk_scale, -me1))
                else:
                    p0 = cute.arch.exp2(rS[nt * 4 + cc] - me0)
                    p1 = cute.arch.exp2(rS[nt * 4 + 2 + cc] - me1)
                rS[nt * 4 + cc] = p0
                rS[nt * 4 + 2 + cc] = p1
                ps0 = ps0 + p0
                ps1 = ps1 + p1
        # The denominator is kept as a PER-LANE partial and reduced once in the
        # epilogue.  The rescale factor `al` is warp-uniform across the four
        # lanes of a row (it comes from the max butterfly), so
        # sum_lane(L_lane*al + p_lane) == (sum_lane L_lane)*al + sum_lane p_lane
        # exactly -- four shuffles and their dependent latency leave the inner
        # loop at zero cost to the arithmetic.
        # The first streaming iteration has no previous numerator to correct:
        # rO is exact zero, so all 64 accumulator multiplies are no-ops.
        need = (_it != Int32(0)) and \
            ((al0 < Float32(1.0)) or (al1 < Float32(1.0)))
        rL[0] = rL[0] * al0 + ps0
        rL[1] = rL[1] * al1 + ps1
        # The running max only ever RISES, so `al == 1` exactly on every chunk
        # that does not move it; a warp-uniform ballot then skips the 64
        # accumulator rescales.  The max itself is computed and applied on
        # every chunk -- this is a pure no-op elision, not a shortcut.
        if cute.arch.vote_ballot_sync(need) != Int32(0):
            for nd in cutlass.range_constexpr(ND):
                rO[nd * 4 + 0] = rO[nd * 4 + 0] * al0
                rO[nd * 4 + 1] = rO[nd * 4 + 1] * al0
                rO[nd * 4 + 2] = rO[nd * 4 + 2] * al1
                rO[nd * 4 + 3] = rO[nd * 4 + 3] * al1

        # ---------------- P @ V ---------------------------------------------
        pav = []
        for ks in cutlass.range_constexpr(ntg // 2):
            pav.append((pack_f16x2(rS[ks * 8 + 0], rS[ks * 8 + 1]),
                        pack_f16x2(rS[ks * 8 + 2], rS[ks * 8 + 3]),
                        pack_f16x2(rS[ks * 8 + 4], rS[ks * 8 + 5]),
                        pack_f16x2(rS[ks * 8 + 6], rS[ks * 8 + 7])))
        pa0, pa1, pa2, pa3 = pav[0]
        if cutlass.const_expr(lowreg):
            cute.arch.sync_warp()
            for dn in cutlass.range_constexpr(ND // 2):
                b0, b1, b2, b3 = ldmatrix_x4_trans(
                    ldm0 + Int32(dn * MMA_M * 2))
                d0, d1, d2, d3 = mma_m16n8k16_f16(
                    pa0, pa1, pa2, pa3, b0, b1,
                    rO[(2 * dn) * 4], rO[(2 * dn) * 4 + 1],
                    rO[(2 * dn) * 4 + 2], rO[(2 * dn) * 4 + 3])
                rO[(2 * dn) * 4] = d0
                rO[(2 * dn) * 4 + 1] = d1
                rO[(2 * dn) * 4 + 2] = d2
                rO[(2 * dn) * 4 + 3] = d3
                d0, d1, d2, d3 = mma_m16n8k16_f16(
                    pa0, pa1, pa2, pa3, b2, b3,
                    rO[(2 * dn + 1) * 4], rO[(2 * dn + 1) * 4 + 1],
                    rO[(2 * dn + 1) * 4 + 2], rO[(2 * dn + 1) * 4 + 3])
                rO[(2 * dn + 1) * 4] = d0
                rO[(2 * dn + 1) * 4 + 1] = d1
                rO[(2 * dn + 1) * 4 + 2] = d2
                rO[(2 * dn + 1) * 4 + 3] = d3
            cute.arch.sync_warp()
        else:
          for ks in cutlass.range_constexpr(ntg // 2):
            # k-step ks contracts token groups 2ks / 2ks+1 against P columns
            # 16ks .. 16ks+15; every k-step folds into the SAME rO tile.
            pb0, pb1, pb2, pb3 = pav[ks]
            vb = ks * 8
            sv0 = e4m3x2_to_f16x2(rVs[2 * ks])
            sv1 = e4m3x2_to_f16x2(rVs[2 * ks + 1])
            for u in cutlass.range_constexpr(2):
                # One nibble transpose per token pair, shared by both halves.
                px, py = pv_nibble_pair(rV[vb + u], rV[vb + 2 + u])
                qx, qy = pv_nibble_pair(rV[vb + 4 + u], rV[vb + 6 + u])
                for hh in cutlass.range_constexpr(2):
                    p0, p1, p2, p3 = pv_dequant4_xy(px, py, sv0, hh)
                    q0, q1, q2, q3 = pv_dequant4_xy(qx, qy, sv1, hh)
                    base = (u * 8 + hh * 4) * 4
                    d0, d1, d2, d3 = mma_m16n8k16_f16(
                        pb0, pb1, pb2, pb3, p0, q0,
                        rO[base + 0], rO[base + 1],
                        rO[base + 2], rO[base + 3])
                    rO[base + 0] = d0
                    rO[base + 1] = d1
                    rO[base + 2] = d2
                    rO[base + 3] = d3
                    d0, d1, d2, d3 = mma_m16n8k16_f16(
                        pb0, pb1, pb2, pb3, p1, q1,
                        rO[base + 4], rO[base + 5],
                        rO[base + 6], rO[base + 7])
                    rO[base + 4] = d0
                    rO[base + 5] = d1
                    rO[base + 6] = d2
                    rO[base + 7] = d3
                    d0, d1, d2, d3 = mma_m16n8k16_f16(
                        pb0, pb1, pb2, pb3, p2, q2,
                        rO[base + 8], rO[base + 9],
                        rO[base + 10], rO[base + 11])
                    rO[base + 8] = d0
                    rO[base + 9] = d1
                    rO[base + 10] = d2
                    rO[base + 11] = d3
                    d0, d1, d2, d3 = mma_m16n8k16_f16(
                        pb0, pb1, pb2, pb3, p3, q3,
                        rO[base + 12], rO[base + 13],
                        rO[base + 14], rO[base + 15])
                    rO[base + 12] = d0
                    rO[base + 13] = d1
                    rO[base + 14] = d2
                    rO[base + 15] = d3
        for i in cutlass.range_constexpr(4 * ntg):
            rK[i] = rKn[i]
            rV[i] = rVn[i]
        for wsf in cutlass.range_constexpr(NSFW):
            rSf[wsf] = rSf[NSFW + wsf]
        if cutlass.const_expr(lowreg):
            rVs[0] = rVsn[0]
        elif cutlass.const_expr(not vsb):
            for p in cutlass.range_constexpr(ntg):
                rVs[p] = rVsn[p]
        if cutlass.const_expr(vsb):
            # V for the NEXT chunk is fetched only now, into the registers the
            # PV MMA above has just freed.  Dropping the V shadow set is what
            # brings the 512-CTA tier under 128 registers (four resident CTAs
            # per SM); its DRAM latency is still covered, by the next
            # iteration's K prefetch plus the whole QK MMA and softmax.
            for p in cutlass.range_constexpr(2 * ntg):
                pt = (p // 2) * 8 + (p % 2) * 4
                v0 = Int32(0)
                v1 = Int32(0)
                if Int32(pt) < nvl:
                    v0, v1 = ld_global_v2_b32(nvd_p, pt * HDP)
                rV[p * 2 + 0] = v0
                rV[p * 2 + 1] = v1
            for p in cutlass.range_constexpr(ntg):
                sa = Int32(0)
                sb = Int32(0)
                if Int32(p * 8) < nvl:
                    sa = ld_global_s32(nvs_p, p * 8 * SD)
                if Int32(p * 8 + 4) < nvl:
                    sb = ld_global_s32(nvs_p, p * 8 * SD + 32)
                rVs[p] = prmt_b32(sa, sb, vs_sel)


    # ======================= epilogue ======================================
    # Reduce the four warps' independent token ranges, then emit either the
    # final bf16 output (nsplit == 1) or an fp32 split-K partial + LSE.
    for off in cutlass.range_constexpr(2):
        rL[0] = rL[0] + cute.arch.shuffle_sync_bfly(rL[0], Int32(1 << off))
        rL[1] = rL[1] + cute.arch.shuffle_sync_bfly(rL[1], Int32(1 << off))
    cute.arch.sync_threads()
    red_ptr = cute.make_ptr(cutlass.Float32, kv_base32,
                            mem_space=sKV.iterator.memspace, assumed_align=16)
    sRed = cute.make_tensor(red_ptr, cute.make_layout(
        (NWARP, 32, red_rs), stride=(32 * red_rs, red_rs, 1)))
    sRedF = cute.make_tensor(red_ptr, cute.make_layout((kv_rows * SRS // 2,)))
    if lane_col == Int32(0):
        sMisc[warp * Int32(MMA_M) + lane_row] = rM[0]
        sMisc[warp * Int32(MMA_M) + lane_row + Int32(8)] = rM[1]
        sMisc[Int32(NWARP * MMA_M) + warp * Int32(MMA_M) + lane_row] = rL[0]
        sMisc[Int32(NWARP * MMA_M) + warp * Int32(MMA_M) + lane_row + Int32(8)] = rL[1]

    # The cross-warp O reduction runs in two halves so its fp32 staging buffer
    # fits inside the fp16 KV staging tile, which is dead by then.
    #
    # The 8 values a thread reduces in each half land on exactly TWO output
    # rows (lane_row and lane_row+8), so the cross-warp max, the four rescale
    # weights and the denominator are built ONCE PER ROW and reused across all
    # dimensions instead of being rebuilt for every one of the 64 values.
    vgrp = tid // Int32(32)
    cute.arch.sync_threads()
    lse_col = Int32(HEAD_DIM)
    if cutlass.const_expr(not lowreg):
        lse_col = Int32(HEAD_DIM + (HEAD_DIM // 32) * OUT_SKEW)
    rWt = cute.make_rmem_tensor((2 * NWARP,), cutlass.Float32)
    rInv = cute.make_rmem_tensor((2,), cutlass.Float32)
    sMerge = cute.make_tensor(
        sOut.iterator,
        cute.make_layout((MMA_M, NWARP + 1), stride=(1, MMA_M)))
    # All four output warps need identical merge weights for a row.  Warp 0
    # computes the 16 row tables once in the dead Q region of sOut.
    if tid < Int32(MMA_M):
        rw = tid
        m0 = sMisc[rw]
        m1 = sMisc[Int32(MMA_M) + rw]
        m2 = sMisc[Int32(2 * MMA_M) + rw]
        m3 = sMisc[Int32(3 * MMA_M) + rw]
        mmax = cute.arch.fmax(fmax3(m0, m1, m2), m3)
        me = cute.arch.fmax(mmax, Float32(NEG_CLAMP))
        den = Float32(0.0)
        for w in cutlass.range_constexpr(NWARP):
            wt = cute.arch.exp2(sMisc[Int32(w * MMA_M) + rw] - me)
            sMerge[(rw, Int32(w))] = wt
            den = den + wt * sMisc[Int32(NWARP * MMA_M + w * MMA_M) + rw]
        iv = Float32(0.0)
        if den > Float32(0.0):
            iv = v_gs * rcp_approx(den)
        sMerge[(rw, Int32(NWARP))] = iv
        # The row maximum and denominator above are exactly the quantities
        # needed for the split-K LSE.  Publish them once from warp 0 instead
        # of recomputing a second max + four exp2 values after the O reduction.
        # The unsplit path consumes no LSE, so it skips this work entirely.
        if cutlass.const_expr(static_nsplit != 1):
            if nsplit != Int32(1):
                lv = Float32(NEG_BIG)
                if den > Float32(0.0):
                    lv = mmax + log2_approx(den)
                sOut[(rw, lse_col)] = lv
    cute.arch.sync_threads()
    for rr in cutlass.range_constexpr(2):
        rw = lane_row + Int32(rr * 8)
        for w in cutlass.range_constexpr(NWARP):
            rWt[rr * NWARP + w] = sMerge[(rw, Int32(w))]
        rInv[rr] = sMerge[(rw, Int32(NWARP))]
    for half in cutlass.range_constexpr(redh):
        if cutlass.const_expr(half > 0):
            cute.arch.sync_threads()
        for i in cutlass.range_constexpr(red_half):
            sRed[(warp, lane, i)] = rO[half * red_half + i]
        cute.arch.sync_threads()
        for vv in cutlass.range_constexpr(red_half // NWARP):
            v = vgrp * Int32(red_half // NWARP) + Int32(vv)
            vg = v + Int32(half * red_half)
            rr = (vv % 4) // 2                 # constexpr: 0,0,1,1,0,0,1,1
            nd = vg // Int32(4)
            row = lane_row + Int32(rr * 8)
            dim = nd * Int32(8) + lane_col * Int32(2) + Int32(vv % 2)
            if cutlass.const_expr(not lowreg):
                # Inverse of the direct-V fragment's head-dim permutation.
                dim = lane_col * Int32(32) + Int32((vv % 2) * 16) + nd
            if cutlass.const_expr(not lowreg and static_nsplit != 1):
                n01 = rWt[rr * NWARP] * sRed[(Int32(0), lane, v)] \
                    + rWt[rr * NWARP + 1] * sRed[(Int32(1), lane, v)]
                n23 = rWt[rr * NWARP + 2] * sRed[(Int32(2), lane, v)] \
                    + rWt[rr * NWARP + 3] * sRed[(Int32(3), lane, v)]
                num = n01 + n23
            else:
                num = Float32(0.0)
                for w in cutlass.range_constexpr(NWARP):
                    num = num + rWt[rr * NWARP + w] \
                        * sRed[(Int32(w), lane, v)]
            out_dim = dim
            if cutlass.const_expr(not lowreg):
                if cutlass.const_expr(static_nsplit == 1 and scored_geom
                                      and cluster_c == 0):
                    out_dim = dim + (dim >> Int32(5)) * Int32(OUT_SKEW_U)
                else:
                    out_dim = _oc(dim)
            sOut[(row, out_dim)] = num * rInv[rr]
    cute.arch.sync_threads()

    orow = tid // Int32(8)
    ocol = (tid % Int32(8)) * Int32(16)
    oskew = ocol
    if cutlass.const_expr(not lowreg):
        # Sixteen consecutive head-dims never straddle a 32-wide skew group.
        if cutlass.const_expr(static_nsplit == 1 and scored_geom
                              and cluster_c == 0):
            oskew = ocol + (ocol // Int32(32)) * Int32(OUT_SKEW_U)
        else:
            oskew = ocol + (ocol // Int32(32)) * Int32(OUT_SKEW)
    # Scored split binaries launch all splits of one tile as a cluster.  Each
    # rank reads peer partials directly from DSMEM and owns a disjoint row set.
    #
    # The row set is the ONLY thing in this combine that cares how many splits
    # there are.  The peer scan below is a linear range over CC -- not a tree,
    # no shuffle strides, no pairwise pass -- and every peer address is
    # `mapa.shared::cluster` at the SAME offset in each rank's window, so the
    # arithmetic is identical for any CC in [2, 8].  What does care is the
    # partition of MMA_M output rows across CC ranks: a CC that divides MMA_M
    # gives every rank the same RPC rows, and one that does not gives the last
    # rank a short set.  `RPC` is therefore the CEILING, and `lim` cuts the
    # rows a ceil-sized range would name past the end of the tile.  With
    # MMA_M % CC == 0 that cut cannot bind, `lim` is the constant NQ, and this
    # is the same loop it was before.
    if cutlass.const_expr(cluster_c > 1):
        CC = cluster_c
        RPC = (MMA_M + CC - 1) // CC
        NQ = RPC * (HEAD_DIM // 4)
        NIT = (NQ + NTHREAD - 1) // NTHREAD
        sout32 = sOut.iterator.toint()
        rank = split
        r0 = rank * Int32(RPC)
        lim = Int32(NQ)
        if cutlass.const_expr(MMA_M % CC != 0):
            tail = (Int32(MMA_M) - r0) * Int32(HEAD_DIM // 4)
            if tail < lim:
                lim = tail
        cluster_arrive()
        cluster_wait()
        rLs = cute.make_rmem_tensor((CC,), cutlass.Float32)
        for it in cutlass.range_constexpr(NIT):
            qq = tid + Int32(it * NTHREAD)
            if qq < lim:
                row = r0 + qq // Int32(HEAD_DIM // 4)
                d0 = (qq % Int32(HEAD_DIM // 4)) * Int32(4)
                pcol = d0
                if cutlass.const_expr(not lowreg):
                    pcol = _oc(d0)
                lbyte = row * Int32(out_srs * 4) + lse_col * Int32(4)
                pbyte = row * Int32(out_srs * 4) + pcol * Int32(4)
                cmax = Float32(NEG_BIG)
                for s in cutlass.range_constexpr(CC):
                    lv = ld_dsmem_f32(mapa_shared(sout32 + lbyte, Int32(s)))
                    rLs[s] = lv
                    cmax = cute.arch.fmax(cmax, lv)
                cme = cute.arch.fmax(cmax, Float32(NEG_CLAMP))
                a0 = Float32(0.0)
                a1 = Float32(0.0)
                a2 = Float32(0.0)
                a3 = Float32(0.0)
                cden = Float32(0.0)
                for s in cutlass.range_constexpr(CC):
                    cw = cute.arch.exp2(rLs[s] - cme)
                    cden = cden + cw
                    f0, f1, f2, f3 = ld_dsmem_v4_f32(
                        mapa_shared(sout32 + pbyte, Int32(s)))
                    a0 = a0 + cw * f0
                    a1 = a1 + cw * f1
                    a2 = a2 + cw * f2
                    a3 = a3 + cw * f3
                cinv = Float32(0.0)
                if cden > Float32(0.0):
                    cinv = rcp_approx(cden)
                if row < grp:
                    ob = out_addr + Int64(2) * Int64(
                        (qi * num_qo_heads + h * grp + row) * Int32(HEAD_DIM)
                        + d0)
                    st_global_v2_b32(
                        ob, pack_bf16x2(a0 * cinv, a1 * cinv),
                        pack_bf16x2(a2 * cinv, a3 * cinv))
        # A rank cannot retire while a peer is still reading its SMEM window.
        cluster_arrive()
        cluster_wait()
    # Compile-time unsplit builds omit the integrated combine's 16-float
    # accumulator and address state. The q64 build retains Q; only q128 also
    # enables the low-register Q-reload specialization.
    elif cutlass.const_expr(static_nsplit == 1):
        if orow < grp:
            obase = out_addr + Int64(2) * Int64(
                (qi * num_qo_heads + h * grp + orow) * Int32(HEAD_DIM) + ocol)
            for v in cutlass.range_constexpr(2):
                p0 = pack_bf16x2(sOut[(orow, oskew + Int32(8 * v + 0))],
                                 sOut[(orow, oskew + Int32(8 * v + 1))])
                p1 = pack_bf16x2(sOut[(orow, oskew + Int32(8 * v + 2))],
                                 sOut[(orow, oskew + Int32(8 * v + 3))])
                p2 = pack_bf16x2(sOut[(orow, oskew + Int32(8 * v + 4))],
                                 sOut[(orow, oskew + Int32(8 * v + 5))])
                p3 = pack_bf16x2(sOut[(orow, oskew + Int32(8 * v + 6))],
                                 sOut[(orow, oskew + Int32(8 * v + 7))])
                st_global_v4_b32(obase + Int64(16 * v), p0, p1, p2, p3)
    elif nsplit == Int32(1):
        if orow < grp:
            obase = out_addr + Int64(2) * Int64(
                (qi * num_qo_heads + h * grp + orow) * Int32(HEAD_DIM) + ocol)
            for v in cutlass.range_constexpr(2):
                p0 = pack_bf16x2(sOut[(orow, oskew + Int32(8 * v + 0))],
                                 sOut[(orow, oskew + Int32(8 * v + 1))])
                p1 = pack_bf16x2(sOut[(orow, oskew + Int32(8 * v + 2))],
                                 sOut[(orow, oskew + Int32(8 * v + 3))])
                p2 = pack_bf16x2(sOut[(orow, oskew + Int32(8 * v + 4))],
                                 sOut[(orow, oskew + Int32(8 * v + 5))])
                p3 = pack_bf16x2(sOut[(orow, oskew + Int32(8 * v + 6))],
                                 sOut[(orow, oskew + Int32(8 * v + 7))])
                st_global_v4_b32(obase + Int64(16 * v), p0, p1, p2, p3)
    else:
        pbase = op_addr + Int64(4) * Int64(
            bid * Int32(MMA_M * HEAD_DIM) + orow * Int32(HEAD_DIM) + ocol)
        for v in cutlass.range_constexpr(4):
            st_global_v4_f32(pbase + Int64(16 * v),
                             sOut[(orow, oskew + Int32(4 * v + 0))],
                             sOut[(orow, oskew + Int32(4 * v + 1))],
                             sOut[(orow, oskew + Int32(4 * v + 2))],
                             sOut[(orow, oskew + Int32(4 * v + 3))])
        if tid < Int32(MMA_M):
            st_global_f32(lse_addr + Int64(4) * Int64(bid * Int32(MMA_M) + tid),
                          sOut[(tid, lse_col)])

        # ---- in-kernel split-K fixup ---------------------------------------
        # A separate combine LAUNCH measured 7.07 us of a 26.8 us decode step
        # at 10% occupancy and 4% of DRAM: it is pure latency, not work.  So
        # the LAST of this tile's `nsplit` CTAs to arrive does the combine
        # itself, right here, while the other tiles are still streaming KV --
        # the partials it reads were written moments ago and are L2-hot, and
        # the counter is re-armed to 0 by that same CTA so the call path stays
        # allocation-free and capture-safe.
        cute.arch.sync_threads()
        if tid == Int32(0):
            # release-ordered arrival: publishes the partial + LSE stores
            sCnt[1] = atom_add_u32(cnt_addr + Int64(4) * Int64(rest), Int32(1))
        cute.arch.sync_threads()
        if sCnt[1] == (nsplit - Int32(1)):
            fence_acq_gpu()               # acquire every other CTA's partial
            if tid == Int32(0):
                st_global_u32(cnt_addr + Int64(4) * Int64(rest), Int32(0))
            sbase = rest * nsplit
            nlse = nsplit * Int32(MMA_M)
            for c in cutlass.range((nlse + Int32(NTHREAD - 1)) // Int32(NTHREAD)):
                jj = c * Int32(NTHREAD) + tid
                if jj < nlse:
                    sRedF[jj] = ld_global_cg_f32(
                        lse_addr + Int64(4) * Int64(sbase * Int32(MMA_M) + jj))
            cute.arch.sync_threads()
            crow = tid // Int32(8)
            ccol = (tid % Int32(8)) * Int32(16)
            cmax = Float32(NEG_BIG)
            for s in cutlass.range(nsplit):
                cmax = cute.arch.fmax(cmax, sRedF[s * Int32(MMA_M) + crow])
            cme = cute.arch.fmax(cmax, Float32(NEG_CLAMP))
            rAcc = cute.make_rmem_tensor((16,), cutlass.Float32)
            for u in cutlass.range_constexpr(16):
                rAcc[u] = Float32(0.0)
            cden = Float32(0.0)
            cbase = op_addr + Int64(4) * Int64(
                sbase * Int32(MMA_M * HEAD_DIM) + crow * Int32(HEAD_DIM) + ccol)
            cstep = Int64(4) * Int64(MMA_M * HEAD_DIM)
            # Two splits per trip: the eight v4 reads of a pair are mutually
            # independent, so the pair costs ONE round trip instead of two.
            # FOUR splits per trip: the sixteen v4 reads of a quad are mutually
            # independent, so a quad costs ONE DRAM round trip.  The combine
            # runs with only `n_base` CTAs resident -- it is a pure drain tail,
            # and its length is set by how many DEPENDENT round trips it makes,
            # not by the bytes it moves.
            nquad = nsplit // Int32(4)
            for p in cutlass.range(nquad):
                s0 = p * Int32(4)
                cw0 = cute.arch.exp2(sRedF[s0 * Int32(MMA_M) + crow] - cme)
                cw1 = cute.arch.exp2(
                    sRedF[(s0 + Int32(1)) * Int32(MMA_M) + crow] - cme)
                cw2 = cute.arch.exp2(
                    sRedF[(s0 + Int32(2)) * Int32(MMA_M) + crow] - cme)
                cw3 = cute.arch.exp2(
                    sRedF[(s0 + Int32(3)) * Int32(MMA_M) + crow] - cme)
                cden = cden + cw0 + cw1 + cw2 + cw3
                for u in cutlass.range_constexpr(4):
                    f0, f1, f2, f3 = ld_global_cg_v4_f32(
                        cbase + Int64(s0) * cstep + Int64(16 * u))
                    g0, g1, g2, g3 = ld_global_cg_v4_f32(
                        cbase + Int64(s0 + Int32(1)) * cstep + Int64(16 * u))
                    h0, h1, h2, h3 = ld_global_cg_v4_f32(
                        cbase + Int64(s0 + Int32(2)) * cstep + Int64(16 * u))
                    k0, k1, k2, k3 = ld_global_cg_v4_f32(
                        cbase + Int64(s0 + Int32(3)) * cstep + Int64(16 * u))
                    rAcc[u * 4 + 0] = rAcc[u * 4 + 0] + cw0 * f0 + cw1 * g0 \
                        + cw2 * h0 + cw3 * k0
                    rAcc[u * 4 + 1] = rAcc[u * 4 + 1] + cw0 * f1 + cw1 * g1 \
                        + cw2 * h1 + cw3 * k1
                    rAcc[u * 4 + 2] = rAcc[u * 4 + 2] + cw0 * f2 + cw1 * g2 \
                        + cw2 * h2 + cw3 * k2
                    rAcc[u * 4 + 3] = rAcc[u * 4 + 3] + cw0 * f3 + cw1 * g3 \
                        + cw2 * h3 + cw3 * k3
            nrem = nsplit - nquad * Int32(4)
            for pr in cutlass.range(nrem):
                s0 = nquad * Int32(4) + pr
                cw0 = cute.arch.exp2(sRedF[s0 * Int32(MMA_M) + crow] - cme)
                cden = cden + cw0
                for u in cutlass.range_constexpr(4):
                    f0, f1, f2, f3 = ld_global_cg_v4_f32(
                        cbase + Int64(s0) * cstep + Int64(16 * u))
                    rAcc[u * 4 + 0] = rAcc[u * 4 + 0] + cw0 * f0
                    rAcc[u * 4 + 1] = rAcc[u * 4 + 1] + cw0 * f1
                    rAcc[u * 4 + 2] = rAcc[u * 4 + 2] + cw0 * f2
                    rAcc[u * 4 + 3] = rAcc[u * 4 + 3] + cw0 * f3
            cinv = Float32(0.0)
            if cden > Float32(0.0):
                cinv = rcp_approx(cden)
            if crow < grp:
                cob = out_addr + Int64(2) * Int64(
                    (qi * num_qo_heads + h * grp + crow) * Int32(HEAD_DIM) + ccol)
                for u in cutlass.range_constexpr(2):
                    q0 = pack_bf16x2(rAcc[u * 8 + 0] * cinv, rAcc[u * 8 + 1] * cinv)
                    q1 = pack_bf16x2(rAcc[u * 8 + 2] * cinv, rAcc[u * 8 + 3] * cinv)
                    q2 = pack_bf16x2(rAcc[u * 8 + 4] * cinv, rAcc[u * 8 + 5] * cinv)
                    q3 = pack_bf16x2(rAcc[u * 8 + 6] * cinv, rAcc[u * 8 + 7] * cinv)
                    st_global_v4_b32(cob + Int64(16 * u), q0, q1, q2, q3)


_SMEM_LOWREG_BYTES = (KV_ROWS_LOW * SRS * 2
               + MMA_M * SRS * 4
               + 2 * MAX_TOPK * 4
               + (2 * NWARP * MMA_M + 4) * 4
               + PT_CAP * 4
               + 64 + 512)
# Register-V builds: 8.5 KiB reduction buffer + 9.3 KiB sOut + the small
# request-local arrays == just under 20 KiB, so THREE CTAs fit the 64 KiB
# carveout with room to spare (3 x 20.3 KiB + 3 x 1 KiB driver = 64 KiB).
_SMEM_BYTES = (KV_ROWS_REG * SRS * 2       # sKV (register-V builds use this
               #                            only as the epilogue reduction)
               + MMA_M * OUT_SRS * 4       # sOut, with the fp16 Q tile overlaid
               + 2 * MAX_TOPK * 4          # compacted page / block id lists
               + (2 * NWARP * MMA_M + 4) * 4
               + PT_CAP * 4                # staged block-table row
               + 64 + 512)                 # counters + allocator slack


@cute.jit
def _msa_launch(
    q_addr: Int64, q2k_addr: Int64, pt_addr: Int64, sk_addr: Int64,
    out_addr: Int64, op_addr: Int64, lse_addr: Int64, cnt_addr: Int64,
    kd_addr: Int64, ks_addr: Int64, vd_addr: Int64, vs_addr: Int64,
    page_stride: Int32, dhead_stride: Int32, shead_stride: Int32,
    total_q: Int32, num_qo_heads: Int32, num_kv_heads: Int32, grp: Int32,
    topk: Int32, page_size: Int32, max_blocks: Int32, seqlen_q: Int32,
    causal: Int32, nsplit: Int32, bpc: Int32, n_ctas: Int32, n_base: Int32,
    q2k_hs: Int32, q2k_ts: Int32,
    qk_scale: Float32, v_gs: Float32, stream,
    lowreg: cutlass.Constexpr,
    static_nsplit: cutlass.Constexpr,
    scored_geom: cutlass.Constexpr,
    cluster_c: cutlass.Constexpr,
    ntg: cutlass.Constexpr,
    vsb: cutlass.Constexpr,
    pf: cutlass.Constexpr,
    hoist: cutlass.Constexpr,
    mbpm: cutlass.Constexpr,
    qreg: cutlass.Constexpr,
):
    op = _msa_partial_kernel(
        q_addr, q2k_addr, pt_addr, sk_addr, out_addr, op_addr, lse_addr,
        cnt_addr, kd_addr, ks_addr, vd_addr, vs_addr,
        page_stride, dhead_stride, shead_stride,
        total_q, num_qo_heads, num_kv_heads, grp, topk, page_size,
        max_blocks, seqlen_q, causal, nsplit, bpc, q2k_hs, q2k_ts,
        qk_scale, v_gs,
        lowreg, static_nsplit, scored_geom, cluster_c, ntg, vsb, pf, qreg,
        hoist,
    )
    if cutlass.const_expr(cluster_c > 1):
        op.launch(grid=[n_ctas, 1, 1], block=[NTHREAD, 1, 1],
                  cluster=[cluster_c, 1, 1],
                  smem=(_SMEM_LOWREG_BYTES if lowreg else _SMEM_BYTES),
                  stream=stream, min_blocks_per_mp=mbpm)
    else:
        op.launch(grid=[n_ctas, 1, 1], block=[NTHREAD, 1, 1],
                  smem=(_SMEM_LOWREG_BYTES if lowreg else _SMEM_BYTES),
                  stream=stream, min_blocks_per_mp=mbpm)


_KERNELS = {}          # device -> compiled variant table (lock-protected)
_SCRATCH = {}
_SCRATCH_STREAMS = {}
_LOCK = threading.Lock()
# A 512-CTA grid at FOUR resident CTAs/SM is what this machine wants.  Holding
# the grid at 256 leaves 1.7 CTAs (6.9 of 64 possible warps) on each of 148 SMs
# and the kernel is latency-bound there.  Measured on the b128 row, which is
# already a 512-CTA grid: capping it at three resident CTAs/SM (168 registers,
# 444-CTA capacity, so 68 CTAs spill into a second wave) costs 110.96 us against
# 82.46 us at four.  Occupancy, not split cost, is the first-order term.
_TARGET_CTAS = 512          # base grid above this needs no split at all
# Per-row cost fits on this machine: the ntg=4 / two-resident-CTA family runs
# 9.3 us fixed + 1.83 us per 32-token chunk against 12 us + 1.875 us per
# 16-token chunk for the ntg=2 / four-resident-CTA family -- the wide-chunk
# binary amortises the invariant Q ldmatrix traffic and every other per-chunk
# cost over twice the tokens, and the narrow-chunk family buys its extra
# residency back only in fixed cost.  So a 512-CTA grid earns its combine only
# once the tile count alone overflows the 296-CTA capacity of the wide-chunk
# binaries; below that the wide-chunk family runs UNSPLIT and pays no partial.
_WIDE_MIN = 512             # tile count above which a 512-CTA grid pays for its
                            # own combine
_WIDE_CTAS = 512            # ... and the grid it then targets
_BASE_CTAS = 256            # CTA target below that
# nsplit -> _KERNELS index.  _WIDE_K is the 512-CTA / 4-resident-CTA family
# (ntg=2, single V buffer, 128 registers); _BASE_K is the 256-CTA family
# (ntg=4 streaming, 2 resident CTAs, no register cap worth paying for).
_WIDE_K = {2: 2}
_BASE_K = {2: 3, 3: 8, 4: 4, 8: 5}
# A SPLIT IS LAUNCHED AS A CLUSTER OF THAT MANY CTAs, AND A CLUSTER IS RESIDENT
# OR IT IS NOT.  So the wave a split grid has to fit inside is not
# gridDim/SM_count: a cluster occupies whole CTA slots in ONE scheduling
# domain, and a cluster size that does not divide that domain's slot count
# strands the remainder.  The three-CTA cluster's wave measures 288 CTAs on
# this machine, and the boundary is a cliff rather than a slope:
#
#   n_ctas 264 (b22)  11.680 us      n_ctas 300 (b25)  18.880 us
#   n_ctas 288 (b24)  11.616 us      n_ctas 336 (b28)  20.832 us
#                                    n_ctas 372 (b31)  21.338 us
#
# and 17.7-17.8 us is what the route's other kernel costs on those same rows,
# so above the wave the three-way split is a REGRESSION, not a smaller win.
# The next split count down is not: 176-248 CTAs of twice-as-fat CTAs, one
# wave, 13.7-14.0 us flat across the whole overflow range.  So a split count
# with an entry here is taken only while its grid fits its own wave, and
# otherwise steps down to the next count that has an instantiation.
#
# ONLY COUNTS WITH AN ENTRY ARE CAPPED.  1, 2, 4 and 8 have none, so every
# selection this file made before is the selection it makes now.
_WAVE_CTAS = {3: 288}
# THE SPLIT COUNT THE TARGET ASKS FOR IS NOT ALWAYS ONE THAT HAS A BINARY, AND
# THAT IS NOT A REASON TO DECLINE THE CALL.  `ceil(target / n_base)` takes every
# value in its range, so at small `n_base` it names counts nothing instantiates:
# batches 1..7 name 10..16 (the top-k cap) and 10..15 name 5..7.  Before this
# rule those batches fell out of the dispatch entirely and ran the route's other
# kernel; the loop in `plan()` now steps DOWN to the largest count that has a
# compiled instantiation.  Stepping down only ever SHRINKS the grid, so it can
# never walk into a wave the finer count already fitted, and it cannot move a
# batch whose own count is already instantiated -- every batch covered before
# this rule keeps the exact instantiation it had, which a test asserts over
# 1..512 with every cap cleared.
#
# THE COUNT IT LANDS ON IS MEASURED, NOT ASSUMED.  GB300 sm103, one process,
# one device, one set of tensors per row, device us (min of three alternated
# passes), control = the same build with the route forced to the ping-pong
# kernel.  Every instantiated count was forced on every row by moving the CTA
# target above, so this is a comparison of binaries and not of arguments:
#
#   batch  n_base | nsplit=1  2       4      8    | ping-pong | taken  ratio
#       1       4 |   14.376  9.978   7.034  5.708 |    6.560 | 8      1.149
#       2       8 |   14.400 10.132   7.174  5.766 |    6.604 | 8      1.145
#       3      12 |   14.298 10.490   7.200  5.862 |    6.644 | 8      1.133
#       4      16 |   14.284 10.382   7.290  7.104 |    7.820 | 8      1.101
#       5      20 |   14.272 10.318   7.360  7.206 |    8.128 | 8      1.128
#       6      24 |   14.304 10.284   7.380  7.200 |    8.160 | 8      1.133
#       7      28 |   14.432 10.240   7.366  7.290 |    8.332 | 8      1.143
#       9      36 |   14.310 10.356   7.488  7.532 |    8.186 | 8      1.087
#      10      40 |   14.304 10.580   9.306 12.492 |   10.502 | 4      1.129
#      12      48 |   14.316 10.516   9.458 13.408 |   10.636 | 4      1.125
#
# The step-down never asks for a grid the eight-way split has not been measured
# at: `ceil(_BASE_CTAS / n_base) >= 8` requires `n_base <= 36`, so its largest
# grid is batch 9's 288 CTAs -- which this build already selected before the
# rule existed, and which is measured here for the first time (7.532 against
# 8.186).  Batch 10 would be 320 and is 12.492, so the eight-CTA cluster's wave
# is in [288, 320); the dispatch cannot reach it and no cap is needed.
#
# Split counts with NO binary were compiled at runtime and priced on the same
# rows before this was settled.  Cluster sizes 5, 6, 7, 11 and 16 all compile,
# launch and are numerically correct here -- CuTe DSL sets
# `non_portable_cluster_size_allowed` on every kernel it emits, so 8 is not a
# hard bound on this toolchain.  They buy nothing worth a binary: 16 is 4.908 at
# batch 1 (14% under the eight-way split) and over its own wave by batch 2, and
# 6 is 7.5-10% better at batches 4, 5 and 10 and over its wave by batch 6.  One
# more binary each, one more machine-specific cap each, for one or three batches.
_LOG2E = 1.4426950408889634


def _compile_variant(lowreg, static_nsplit, scored_geom, cluster_c=0, ntg=2,
                     mbpm=2, vsb=False, qreg=False, pf=None, hoist=None):
    if pf is None:
        pf = mbpm < 4
    if hoist is None:
        hoist = mbpm < 4
    return cute.compile(
        _msa_launch, *([Int64(0)] * 12), *([Int32(0)] * 18),
        Float32(0.0), Float32(0.0), cuda_driver.CUstream(0),
        lowreg=lowreg, static_nsplit=static_nsplit,
        scored_geom=scored_geom, cluster_c=cluster_c, ntg=ntg, vsb=vsb,
        # The L2 prefetch pays only where the grid, not the register file, is
        # what limits the warps in flight.  A four-resident-CTA binary already
        # has 13.8 warps/SM hiding its DRAM, sits exactly on the 128-register
        # cliff, and streams a working set far larger than L2 -- there the
        # prefetch's stream-coordinate arithmetic and extra live registers cost
        # more than the latency it removes (measured b64 44 -> 48 us, b128
        # 75 -> 84 us).  Two-resident-CTA binaries are grid-limited with 128
        # spare registers and gain (b32 27 -> 24 us).
        pf=pf, hoist=hoist, mbpm=mbpm, qreg=qreg,
    )


# The eight instantiations this file's dispatch arithmetic can name, by index.
# Each entry is a THUNK, so an index that is never selected is never compiled.
_VARIANTS = (
    # 0 -- fully dynamic "generalized" instantiation.  UNREACHABLE by design:
    # it is the only one that publishes split-K partials through global memory,
    # and that publication is not correctly ordered (see the module docstring).
    lambda: _compile_variant(False, 0, False),
    # 1 -- dynamic geometry, nsplit pinned to 1.  Correct, but untimed, and the
    # route serves every non-specialised geometry with its own kernel instead.
    lambda: _compile_variant(True, 1, False, 0, 2, 4),
    # 2 -- scored nsplit=2 on the 512-CTA tier.  UNREACHABLE: see _WIDE_K.
    # The 512-CTA tier must hold FOUR resident CTAs/SM (444 < 512 spills a
    # whole second wave), so it stays at 128 registers -- but it takes the
    # DIRECT V path anyway: NCU puts that tier at 73% L1/TEX against 23% DRAM
    # and 49% issue, and staging V costs a shared store plus an ldmatrix.trans
    # per token that register fragments do not.
    lambda: _compile_variant(False, 2, True, 2, 2, 4, True),
    # 3/4/5 -- scored split, 256-CTA tier, 32-token streaming chunks.
    lambda: _compile_variant(False, 2, True, 2, WIDE_NTG, 2, qreg=True),
    lambda: _compile_variant(False, 4, True, 4, WIDE_NTG, 2, qreg=True),
    lambda: _compile_variant(False, 8, True, 8, WIDE_NTG, 2, qreg=True),
    # 6 -- scored unsplit, narrow-chunk.  UNREACHABLE: the unsplit arm below
    # always selects 7.  Folding the geometry to constants frees the registers
    # the direct-V path needs to stay inside the 128-register / 4-CTA-per-SM
    # budget.
    lambda: _compile_variant(False, 1, True, 0, 2, 4, True),
    # 7 -- scored unsplit on the WIDE-CHUNK family: at 256 and 512 tiles the
    # two-resident-CTA binaries stream 32-token chunks, hold the invariant Q
    # A-fragments in registers, and pay no split partial at all.
    lambda: _compile_variant(False, 1, True, 0, WIDE_NTG, 2, qreg=True),
    # 8 -- scored THREE-way split, same 256-CTA / 32-token-streaming family as
    # 3/4/5.  It exists because the split count is ceil(256 / n_base) and
    # therefore takes every value in its range, not only the powers of two:
    # without it, n_base 88..124 -- batch 22..31 at the deployment geometry --
    # names no instantiation and falls out of the MIDDLE of the covered range.
    # A three-CTA cluster is the same construct as a two- or four-CTA one; the
    # only asymmetry is that 3 does not divide the 16 output rows, which the
    # combine's ceiling partition handles.
    #
    # It serves the part of that range where its grid still fits one wave --
    # n_base 88..96, batch 22..24 -- and `_WAVE_CTAS` hands the rest to the
    # two-way split, which is 1.26x there while this one would be 0.83-0.94x.
    lambda: _compile_variant(False, 3, True, 3, WIDE_NTG, 2, qreg=True),
)

# The instantiations that are COMPILED, and therefore the only ones a call can
# reach.  Deliberately a subset of `_VARIANTS`:
#
#   0  publishes split-K partials through global memory with an unsound
#      release protocol -- excluded so the protocol is unreachable rather
#      than merely unselected;
#   1  is correct but untimed, and the route has its own kernel for every
#      geometry this one would serve;
#   2  and 6 are unreachable from the dispatch arithmetic itself.  For 2:
#      `nsplit > 1` requires `n_base < _TARGET_CTAS`, and `_WIDE_MIN ==
#      _TARGET_CTAS`, so `n_base >= _WIDE_MIN` is unsatisfiable whenever a
#      split happens and `_WIDE_K` is never consulted.  For 6: the `nsplit ==
#      1` arm names 7 unconditionally.
#
# `plan()` computes the index; `run()` refuses one that is not in this set, so
# the two facts (not compiled, not routed) are independent.
SPECIALISED_KERNEL_IDS = frozenset({3, 4, 5, 7, 8})


class _NotCompiled:
    """Placeholder for an instantiation this build deliberately does not hold.

    Calling it raises.  A silent slow path is what this replaces: an unselected
    instantiation and an uncompiled one are the same thing only until somebody
    changes the arithmetic, and then the difference is a 40%-slower kernel with
    an unsound global publication protocol running in production.
    """

    __slots__ = ("idx",)

    def __init__(self, idx):
        self.idx = idx

    def __call__(self, *args, **kwargs):
        _reject(
            "instantiation %d is not compiled in this build (compiled: %s); "
            "this call must be served by the route's own kernel, not here"
            % (self.idx, sorted(SPECIALISED_KERNEL_IDS))
        )


def _get_kernels(dev, _compiling_ok=False):
    """Look up one device's compiled variant table; build it only under warmup().

    The CALL path is lookup-only.  `_compiling_ok` is set only by warmup(),
    which the route's `warm` entry point invokes once per device.  A lookup
    miss on the call path raises rather than
    dropping a lazy `cute.compile` onto the caller's stream -- which, inside a
    CUDA-graph capture, would break the capture.  The table is keyed on the
    DEVICE and built under `_LOCK`: a table (and the per-kernel launch
    attributes it carries) built against one GPU's context is never reused on
    the second GPU in the process.
    """
    got = _KERNELS.get(dev)
    if got is not None:
        return got
    if not _compiling_ok:
        _reject("kernel variants are not compiled for %s; call warmup(device) "
                "before the first call (compilation is never done on the call "
                "path)" % dev)
    with _LOCK:
        got = _KERNELS.get(dev)
        if got is None:
            got = tuple(
                _VARIANTS[idx]() if idx in SPECIALISED_KERNEL_IDS
                else _NotCompiled(idx)
                for idx in range(len(_VARIANTS))
            )
            _KERNELS[dev] = got
    return got


# ==========================================================================
# Persistent device scratch -- NOT ALLOCATED, AND THAT IS THE POINT.
#
# The split-K arena exists for exactly one instantiation: the fully dynamic
# index 0, which writes its 16 x head_dim fp32 partial and its LSE to global
# memory for a sibling CTA to read back.  Index 0 is not compiled here (see
# `_VARIANTS`) and `plan()` never names a compiled index for a call that would
# use it, so nothing in this build reads or writes the arena.  It is therefore
# not allocated: the honest accounting of a buffer nothing can touch is zero
# bytes, not "65 MiB warmed outside the timed window".
#
# The sizing that WOULD have been needed is retained because it is the thing a
# reader has to check to believe the paragraph above, and because restoring
# index 0 (with its publication protocol fixed) means restoring this:
#
#   nsplit > 1  =>  n_base < _TARGET_CTAS,  nsplit = ceil(target / n_base)
#               =>  n_ctas = n_base * nsplit < target + n_base
#                          < _TARGET_CTAS + max(_BASE_CTAS, _WIDE_CTAS)
#
# and nsplit is additionally capped at topk <= MAX_TOPK.  MAX_SPLIT_CTAS is
# that bound rounded up.  Eight fixed MAX_SPLIT_CTAS*16*129 fp32 slots plus
# MAX_COUNTERS int32 counters each is 8*(1024*16*129*4 + 2048*4) B =
# 65.0 MiB per device, independent of num_pages, batch_size and seq_len.
#
# The compiled instantiations pass the caller-owned output pointer in the two
# dead arena arguments, so the launch signature is unchanged.
# ==========================================================================
MAX_SPLIT_CTAS = 1024             # >= _TARGET_CTAS + max(_BASE_CTAS,_WIDE_CTAS)
MAX_COUNTERS = 2048               # >= _TARGET_CTAS; only n_base < it can split
_WS_ELEMS = MAX_SPLIT_CTAS * MMA_M * (HEAD_DIM + 1)
_SCRATCH_SLOTS = 8                # concurrent streams the arena would serve
# What the arena would cost per device if index 0 were compiled again.  Reported
# by the route's stats() so the number is auditable rather than folklore.
ARENA_BYTES_IF_GENERALIZED_WERE_REACHABLE = _SCRATCH_SLOTS * (
    _WS_ELEMS * 4 + MAX_COUNTERS * 4)
# Bytes this build actually holds per device, for the same reason.
PERSISTENT_DEVICE_BYTES = 0


def _scratch(dev, stream_handle):
    """Refuse: the only instantiation that used the arena is not compiled.

    Reached only if `_VARIANTS`/`SPECIALISED_KERNEL_IDS`/`plan()` are changed
    so that a global-partial instantiation becomes selectable again.  Restoring
    that path means restoring an allocation AND fixing the publication protocol
    the module docstring describes; raising here keeps "we forgot one of those"
    from being a silent correctness bug.

    The stream-slot bookkeeping is retained because it is the other half of the
    contract: at most `_SCRATCH_SLOTS` concurrent CUDA streams per device could
    ever be served, and exceeding that was -- and would again be -- a refusal
    rather than aliased scratch.
    """
    del stream_handle
    _reject(
        "the split-K scratch arena is not allocated on %s: the only "
        "instantiation that uses it (index 0) is not compiled in this build. "
        "This call must be served by the route's own kernel." % dev)


def warmup(device=None):
    """Compile this device's instantiations.  The only place that compiles.

    After it returns for a device, run() on that device is pure lookup: no
    compilation, no allocation, nothing that a CUDA-graph capture region
    forbids.  The table is keyed on the device, so a multi-GPU process warms
    each one; nothing here is allocated on the device (see the scratch note
    above), so warming costs compile time and no HBM.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.index is None:
        dev = torch.device(dev.type, torch.cuda.current_device())
    _get_kernels(dev, _compiling_ok=True)


def is_warm(device) -> bool:
    """Whether :func:`warmup` has completed for ``device``."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.index is None:
        dev = torch.device(dev.type, torch.cuda.current_device())
    return dev in _KERNELS


def _stream(stream_handle):
    return cuda_driver.CUstream(stream_handle)


def _reject(msg):
    raise ValueError("msa_sparse_decode_attention: " + msg)


# ==========================================================================
# THE DISPATCH ARITHMETIC -- ONE COPY.
#
# `run()` selects its instantiation from this function and the route decides
# whether to call `run()` at all from the same function.  Two copies of a
# dispatch predicate is how a route stops matching the kernel it routes to in
# silence, so there is exactly one, and the caller-facing form
# (`specialised_reason`) is a thin wrapper over it rather than a restatement.
# It is pure host-side integer arithmetic over shapes: no tensors, no device,
# so a GPU-free test can execute it over every coordinate a serving run
# reaches.
# ==========================================================================
def plan(*, total_q, num_qo_heads, num_kv_heads, grp, topk, page_size,
         seqlen_q, causal):
    """Resolve the instantiation this call would take, and why.

    Returns a dict with `scored_geom`, `n_base`, `nsplit`, `bpc`, `n_ctas`,
    `kernel_idx`, `specialised` and, when not specialised, a `reason`.
    """
    n_base = total_q * num_kv_heads
    causal_i = int(causal)
    # Runtime-selected instantiation on the MODEL geometry only.  `max_blocks`
    # is NOT part of this predicate and never can be: it is
    # cdiv(max_model_len, page_size), a KV-cache-manager constant of the
    # deployment rather than a property of the call, and a fast path gated on
    # it would be a fast path for exactly one max_model_len.
    scored_geom = (num_qo_heads == 64 and num_kv_heads == 4 and grp == 16
                   and topk == 16 and page_size == 128
                   and seqlen_q == 1 and causal_i == 1)
    # Splitting at all is what costs: an unsplit tile writes bf16 straight out,
    # a split one writes a full fp32 16 x head_dim partial that has to be read
    # back.  So do not split once the base grid already covers the machine --
    # and once splitting is unavoidable, split far enough to actually FILL it
    # (the profile showed 0.58 waves and 7 of 12 possible warps per SM).
    if n_base >= _TARGET_CTAS:
        nsplit = 1
    else:
        target = _WIDE_CTAS if n_base >= _WIDE_MIN else _BASE_CTAS
        nsplit = (target + n_base - 1) // n_base
        if nsplit > topk:
            nsplit = topk
        # Reaching the CTA target is the first-order term, but only up to the
        # point where the grid stops fitting one wave of its own clusters --
        # past that the overflow costs a whole second pass and a coarser split
        # is strictly better.  See `_WAVE_CTAS`.  And a count with no compiled
        # instantiation is not a reason to DECLINE the call; it is a reason to
        # take the next count DOWN that has one -- the note above `_LOG2E`
        # carries the measurement that settles which count that is.  Both
        # conditions step in the same direction and neither can raise `nsplit`,
        # so one loop settles them: stop at the largest count that has a binary
        # AND fits its own wave.  Restricted to the specialised geometry because
        # a wave capacity is a property of a compiled binary, measured on one,
        # and there is no binary anywhere else.
        if scored_geom:
            while nsplit > 1 and (
                _BASE_K.get(nsplit, 0) not in SPECIALISED_KERNEL_IDS
                or n_base * nsplit > _WAVE_CTAS.get(nsplit, 0) > 0
            ):
                nsplit -= 1
    bpc = (topk + nsplit - 1) // nsplit
    n_ctas = n_base * nsplit
    # Split production rows get exact geometry/count binaries. Unsplit rows
    # keep geometry dynamic (constant-folding it regresses q64) while pinning
    # nsplit=1; every other shape falls to the dynamic index the table's
    # `.get` default names -- which this build does not compile.
    if nsplit == 1:
        kernel_idx = 7 if scored_geom else 1
    elif scored_geom:
        # A 512-CTA grid holds four resident CTAs/SM only inside the
        # 128-register budget, which is the ntg=2 / single-V-buffer binary.
        # Grids that stay at 256 CTAs are grid-limited, not register-limited,
        # so they keep the register-rich 32-token streaming binary instead.
        table = _WIDE_K if n_ctas >= _WIDE_CTAS else _BASE_K
        kernel_idx = table.get(nsplit, 0)
    else:
        kernel_idx = 0
    out = {"scored_geom": bool(scored_geom), "n_base": n_base,
           "nsplit": nsplit, "bpc": bpc, "n_ctas": n_ctas,
           "kernel_idx": kernel_idx,
           "specialised": kernel_idx in SPECIALISED_KERNEL_IDS}
    if not out["specialised"]:
        if not scored_geom:
            out["reason"] = (
                "geometry (%d qo heads, %d kv heads, group %d, top-k %d, page "
                "%d, seqlen_q %d, causal %d) is not the specialised geometry "
                "(64/4/16/16/128/1/1)"
                % (num_qo_heads, num_kv_heads, grp, topk, page_size,
                   seqlen_q, causal_i))
        else:
            out["reason"] = (
                "split count %d (from %d base CTAs) has no specialised "
                "instantiation; specialised split counts are %s"
                % (nsplit, n_base, sorted(set(_BASE_K) | set(_WIDE_K))))
    # The arena bound the standalone form enforced here.  It cannot bind on a
    # specialised call -- nsplit > 1 implies n_base < _TARGET_CTAS, so
    # n_ctas < _TARGET_CTAS + _BASE_CTAS = 768 <= MAX_SPLIT_CTAS and
    # n_base < 512 <= MAX_COUNTERS -- so it is recorded, not raised: a
    # non-specialised call is declined for the reason above, not for this one.
    out["exceeds_arena_bound"] = bool(
        nsplit > 1 and (n_ctas > MAX_SPLIT_CTAS or n_base > MAX_COUNTERS))
    return out


def specialised_reason(*, total_q, num_qo_heads, num_kv_heads, grp, topk,
                       page_size, seqlen_q, causal, softmax_scale,
                       k_global_scale):
    """``None`` when :func:`run` will serve this call, else why it will not.

    The route calls this BEFORE it decides which kernel to launch, so every
    condition `run` would raise on has to be represented here -- including the
    scale positivity, which is a numerical precondition of the specialised
    binaries (they scale the row maximum after the reduction, which is exact
    only for a positive scale) and not a geometry fact.
    """
    if not (float(softmax_scale) > 0.0 and float(k_global_scale) > 0.0):
        return ("softmax_scale %r and k_global_scale %r must both be positive"
                % (softmax_scale, k_global_scale))
    sel = plan(total_q=total_q, num_qo_heads=num_qo_heads,
               num_kv_heads=num_kv_heads, grp=grp, topk=topk,
               page_size=page_size, seqlen_q=seqlen_q, causal=causal)
    if not sel["specialised"]:
        return sel["reason"]
    if not (0 < topk <= MAX_TOPK):
        return "top-k %d is outside (0, %d]" % (topk, MAX_TOPK)
    return None


def _validate(q, k_data, v_data, k_scale, v_scale, q2k_indices, page_table,
              seqused_k, output, sq, total_q, num_qo_heads, num_kv_heads,
              page_size, topk, batch, grp, page_stride, dhead_stride,
              shead_stride):
    """REFUSE, never compute silently, outside the supported envelope."""
    if q.dim() != 3 or output.dim() != 3 or q.shape != output.shape:
        _reject("q and output must be 3-D and the same shape")
    if q.dtype is not torch.bfloat16 or output.dtype is not torch.bfloat16:
        _reject("q/output must be bfloat16")
    if not q.is_cuda:
        _reject("inputs must live on a CUDA device")
    if q.shape[2] != HEAD_DIM:
        _reject("head_dim must be 128 (MSA is head_dim-128 only)")
    if not (q.is_contiguous() and output.is_contiguous()):
        _reject("q and output must be contiguous")
    if k_data.dtype is not torch.uint8 or v_data.dtype is not torch.uint8:
        _reject("k_data/v_data must be packed uint8 e2m1 nibbles")
    if k_data.dim() != 4 or v_data.dim() != 4 or k_scale.dim() != 4 \
            or v_scale.dim() != 4:
        _reject("K/V data and scale views must be 4-D")
    if k_data.shape != v_data.shape or k_scale.shape != v_scale.shape:
        _reject("K and V views must share a geometry")
    if k_data.shape[3] != HEAD_DIM // 2 or k_scale.shape[3] != HEAD_DIM // SVEC:
        _reject("packed/scale inner extents must be head_dim//2 and head_dim//16")
    if k_data.shape[0] != k_scale.shape[0] or k_data.shape[1] != k_scale.shape[1] \
            or k_data.shape[2] != k_scale.shape[2]:
        _reject("scale views must match the data views")
    if page_size % 4 != 0:
        _reject("page_size must be a multiple of 4 ((4,4) V-scale swizzle)")
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads != 0 or not (0 < grp <= MMA_M):
        _reject("GQA group size must divide num_qo_heads and lie in (0, 16]")
    # Only the innermost dimension has to be dense; the outer two strides are
    # kernel arguments.
    if q2k_indices.dtype is not torch.int32 or q2k_indices.dim() != 3 \
            or q2k_indices.shape[0] != num_kv_heads \
            or q2k_indices.shape[1] != total_q:
        _reject("q2k_indices must be int32 (num_kv_heads, total_q, topk)")
    if q2k_indices.stride(2) != 1:
        _reject("q2k_indices must be dense in its innermost (top-k) dimension")
    if q2k_indices.stride(0) < 0 or q2k_indices.stride(1) < 0:
        _reject("q2k_indices must not be negatively strided")
    if ((num_kv_heads - 1) * q2k_indices.stride(0)
            + (total_q - 1) * q2k_indices.stride(1) + topk) > 0x7FFFFFFF:
        _reject("the q2k_indices view is too large for 32-bit addressing")
    if not (0 < topk <= MAX_TOPK):
        _reject("top-k above %d is not supported by this build" % MAX_TOPK)
    if page_table.dtype is not torch.int32 or not page_table.is_contiguous() \
            or page_table.dim() != 2:
        _reject("page_table must be a contiguous int32 (batch, max_blocks) table")
    if seqused_k.dtype is not torch.int32 or not seqused_k.is_contiguous() \
            or seqused_k.dim() != 1 or seqused_k.shape[0] != batch:
        _reject("seqused_k must be a contiguous int32 vector of length batch_size")
    if sq <= 0 or total_q != batch * sq:
        _reject("total_q must equal batch_size * seqlen_q")
    # The strides are READ off the tensors; they are only cross-checked against
    # the packed-page formula, never re-derived from the shapes.
    ds = k_data.stride()
    ss = k_scale.stride()
    hd_packed = HEAD_DIM // 2
    sc_dim = HEAD_DIM // SVEC
    if ds != v_data.stride() or ss != v_scale.stride():
        _reject("K and V views must share strides")
    if ds[2] != hd_packed or ds[3] != 1 or ss[2] != sc_dim or ss[3] != 1:
        _reject("data/scale views must be token-major with unit innermost stride")
    if dhead_stride != page_size * hd_packed or shead_stride != page_size * sc_dim \
            or page_stride != 2 * num_kv_heads * page_size * (hd_packed + sc_dim):
        _reject("K/V views do not match the packed NVFP4 page layout "
                "[K_data | K_scale | V_data | V_scale]")


@torch.no_grad()
def run(q, k_data, v_data, k_scale, v_scale, q2k_indices, page_table,
        seqused_k, seqlen_q, causal, softmax_scale, k_global_scale,
        v_global_scale, output):
    qsh = q.shape
    ksh = k_data.shape
    total_q = qsh[0]
    num_qo_heads = qsh[1]
    num_kv_heads = ksh[1]
    page_size = ksh[2]
    topk = q2k_indices.shape[2]
    ptsh = page_table.shape
    batch = ptsh[0]
    max_blocks = ptsh[1]
    grp = num_qo_heads // num_kv_heads if num_kv_heads else 0
    sq = int(seqlen_q)
    page_stride = k_data.stride(0)
    dhead_stride = k_data.stride(1)
    shead_stride = k_scale.stride(1)
    _validate(q, k_data, v_data, k_scale, v_scale, q2k_indices, page_table,
              seqused_k, output, sq, total_q, num_qo_heads, num_kv_heads,
              page_size, topk, batch, grp, page_stride, dhead_stride,
              shead_stride)

    # The raw-score maximum is scaled AFTER the reduction on the specialised
    # binaries, which is exact only for a POSITIVE scale.  `specialised_reason`
    # states the same requirement so the route can decline instead of raising;
    # this is the copy that cannot be bypassed.
    if not (float(softmax_scale) > 0.0 and float(k_global_scale) > 0.0):
        _reject("softmax_scale and k_global_scale must be positive")

    causal_i = int(causal)
    sel = plan(total_q=total_q, num_qo_heads=num_qo_heads,
               num_kv_heads=num_kv_heads, grp=grp, topk=topk,
               page_size=page_size, seqlen_q=sq, causal=causal_i)
    kernel_idx = sel["kernel_idx"]
    nsplit = sel["nsplit"]
    bpc = sel["bpc"]
    n_ctas = sel["n_ctas"]
    n_base = sel["n_base"]
    # SECOND, INDEPENDENT REFUSAL.  `_get_kernels` does not hold the
    # non-specialised instantiations, so this cannot silently succeed; saying
    # so here names the reason instead of surfacing it as "not compiled".
    if kernel_idx not in SPECIALISED_KERNEL_IDS:
        _reject(sel["reason"])
    kernels = _get_kernels(q.device)  # lookup-only; raises on a cold table
    kernel = kernels[kernel_idx]
    stream_handle = torch.cuda.current_stream(q.device).cuda_stream

    # Every compiled instantiation is either clustered-split or unsplit, and
    # both compile out global partials entirely, so the two arena pointers are
    # dead arguments.  They are given the caller-owned output rather than a
    # null so that a stray dereference would fault inside a buffer this call
    # already owns.
    ws_ptr = output.data_ptr()
    cnt_ptr = ws_ptr
    kernel(
        q.data_ptr(), q2k_indices.data_ptr(), page_table.data_ptr(),
        seqused_k.data_ptr(), output.data_ptr(),
        ws_ptr, ws_ptr + n_ctas * MMA_M * HEAD_DIM * 4, cnt_ptr,
        k_data.data_ptr(), k_scale.data_ptr(),
        v_data.data_ptr(), v_scale.data_ptr(),
        page_stride, dhead_stride, shead_stride,
        total_q, num_qo_heads, num_kv_heads, grp, topk, page_size, max_blocks,
        sq, causal_i, nsplit, bpc, n_ctas, n_base,
        q2k_indices.stride(0), q2k_indices.stride(1),
        float(softmax_scale) * float(k_global_scale) * _LOG2E,
        float(v_global_scale), _stream(stream_handle),
    )


# NO SELF-WARM AT IMPORT.  The standalone form of this kernel compiled its
# whole variant table as an import side effect, on whichever device happened to
# be current.  Inside FlashInfer that is wrong three ways: importing a module
# must not cost minutes of ptxas, must not bind a device the caller has not
# chosen, and must not run at all on a machine whose compute capability this
# route does not serve.  `flashinfer.msa_ops._nvfp4_decode_sm100.warm` is the
# lifecycle hook instead -- it imports this module, calls `warmup(device)` and
# takes the first eager launch of every compiled instantiation, all before a
# CUDA-graph capture region can open.
