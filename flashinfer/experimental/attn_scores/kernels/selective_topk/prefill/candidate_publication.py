# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""FP16/BF16 block publication, adapted from TensorRT-LLM GVR V2.

Uses per-row valid lengths, mandatory latest blocks and sorted block IDs;
FP32 publication uses the shared GVR implementation.
"""

import functools
import sys
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import llvm as mlir_llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator
from .candidate_publication_sort import publication_sort

C = sys.modules[__name__]
QUADC_CLUS = 288
SENT_LO = -3e38
SENT_HI = 3e38
RES_B = 0
RES_M = 1
RES_ABOVE = 2
RES_TOT = 3
RES_B2 = 4
RES_B3 = 5


def u32_of_f32(v):
    return cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, v.ir_value()))


def f32_of_u32(u):
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, u.ir_value()))


def fkey_bits(u):
    neg = Uint32(0) - (u >> Uint32(31))
    return u ^ (neg | cutlass.Uint32(2147483648))


def fkey(x):
    return fkey_bits(u32_of_f32(x))


def invkey_bits(K):
    s = K >> Uint32(31)
    m = s - cutlass.Uint32(1) | cutlass.Uint32(2147483648)
    return K ^ m


def invkey(K):
    return f32_of_u32(invkey_bits(K))


def warp_min_u32(v):
    return cute.arch.warp_redux_sync(v, "min")


def warp_max_u32(v):
    return cute.arch.warp_redux_sync(v, "max")


def fmin_f32(a, b):
    return cute.arch.fmin(a, b)


def fmax_f32(a, b):
    return cute.arch.fmax(a, b)


def ballot(pred):
    return cute.arch.vote_ballot_sync(pred)


def popc(x):
    return cute.arch.popc(x)


def clz_i32(x):
    return cutlass.Int32(cute.arch.clz(x))


def ffs_m1(x):
    return cutlass.Int32(cute.arch.popc((x & cutlass.Int32(0) - x) - cutlass.Int32(1)))


@cute.jit
def _shfl_up_add(val, lane, offset: cutlass.Constexpr):
    other = cute.arch.shuffle_sync_up(val, offset, mask_and_clamp=0)
    if lane >= cutlass.Int32(offset):
        val = val + other
    return val


@cute.jit
def warp_incl_scan_add(val, lane):
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_up_add(val, lane, o)
    return val


def atomic_add_cta(ptr, val):
    return cutlass.Int32(cute.arch.atomic_add(ptr, val, sem="relaxed", scope="cta"))


def atomic_min_cta(ptr, val):
    return cute.arch.atomic_min(ptr, val, sem="relaxed", scope="cta")


def atomic_max_cta(ptr, val):
    return cute.arch.atomic_max(ptr, val, sem="relaxed", scope="cta")


def f2u_rz(v):
    return cutlass.Uint32(v)


def f2s_rz(v):
    return cutlass.Int32(v)


@dsl_user_op
def _prefetch_l2(gaddr, *, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[gaddr.ir_value(loc=loc, ip=ip)],
        asm_string="prefetch.global.L2 [$0];",
        constraints="l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def g2r_atom_f32(bits: int, invariant: bool = True):
    return cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        cutlass.Float32,
        num_bits_per_copy=bits,
        invariant=invariant,
    )


def smem_atom_i32_128():
    return cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
    )


def _smem_v4_tensor(base_addr, byte_off):
    p = cute.make_ptr(
        cutlass.Int32, base_addr + byte_off, cute.AddressSpace.smem, assumed_align=16
    )
    return cute.make_tensor(p, cute.make_layout((4,)))


def lds128_i32(copy_atom, base_addr, byte_off, frag):
    cute.copy(copy_atom, _smem_v4_tensor(base_addr, byte_off), frag)


def sts128_i32(copy_atom, frag, base_addr, byte_off):
    cute.copy(copy_atom, frag, _smem_v4_tensor(base_addr, byte_off))


@dsl_user_op
def _lds_v2_u64(saddr, *, loc=None, ip=None):
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i64(), T.i64()]),
        [saddr.ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u64 {$0, $1}, [$2];",
        "=l,=l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Uint64(llvm.extractvalue(T.i64(), ret, [0])),
        cutlass.Uint64(llvm.extractvalue(T.i64(), ret, [1])),
    )


@cute.jit
def scan_cross0(
    s_hist,
    target,
    tidx,
    s_res,
    target2,
    target3,
    s_addv,
    nb: cutlass.Constexpr,
    zero: cutlass.Constexpr,
    two: cutlass.Constexpr = False,
    three: cutlass.Constexpr = False,
    addf: cutlass.Constexpr = False,
):
    BPT = nb // 32
    NV = BPT // 4
    HOLD = NV <= 2
    if tidx < cutlass.Int32(32):
        lane = tidx
        atom = smem_atom_i32_128()
        hbase = s_hist.iterator.toint()
        frags = [cute.make_rmem_tensor((4,), cutlass.Int32) for _ in range(NV)]
        sm = cutlass.Int32(0)
        for q in cutlass.range_constexpr(NV):
            boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
            lds128_i32(atom, hbase, boff, frags[q])
            sm = sm + frags[q][0] + frags[q][1] + frags[q][2] + frags[q][3]
        w = warp_incl_scan_add(sm, lane)
        tot = cute.arch.shuffle_sync(w, cutlass.Int32(31))
        after = tot - w
        if lane == cutlass.Int32(0):
            s_res[RES_TOT] = tot
        base = lane * cutlass.Int32(BPT)
        for q in cutlass.range_constexpr(NV - 1, -1, -1):
            if cutlass.const_expr(HOLD):
                vv = frags[q]
            else:
                vv = cute.make_rmem_tensor((4,), cutlass.Int32)
                boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
                lds128_i32(atom, hbase, boff, vv)
            o4 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for j in cutlass.range_constexpr(3, -1, -1):
                cq = vv[j]
                if cutlass.const_expr(zero):
                    o4[j] = cutlass.Int32(0)
                else:
                    o4[j] = after
                gb = base + cutlass.Int32(4 * q + j)
                cross = cutlass.Int32(0)
                if after < target:
                    if after + cq >= target:
                        cross = cutlass.Int32(1)
                    if gb == cutlass.Int32(0):
                        cross = cutlass.Int32(1)
                if cross != cutlass.Int32(0):
                    s_res[RES_B] = gb
                    s_res[RES_ABOVE] = after
                    s_res[RES_M] = cq
                if cutlass.const_expr(two):
                    cross2 = cutlass.Int32(0)
                    if after < target2:
                        if after + cq >= target2:
                            cross2 = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross2 = cutlass.Int32(1)
                    if cross2 != cutlass.Int32(0):
                        s_res[RES_B2] = gb
                if cutlass.const_expr(three):
                    cross3 = cutlass.Int32(0)
                    if after < target3:
                        if after + cq >= target3:
                            cross3 = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross3 = cutlass.Int32(1)
                    if cross3 != cutlass.Int32(0):
                        s_res[RES_B3] = gb
                after = after + cq
            if cutlass.const_expr(addf):
                av = cute.make_rmem_tensor((4,), cutlass.Int32)
                aoff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
                lds128_i32(atom, s_addv.iterator.toint(), aoff, av)
                for j in cutlass.range_constexpr(4):
                    o4[j] = o4[j] + av[j]
            boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
            sts128_i32(atom, o4, hbase, boff)


QUADC_CLUS__main = C.QUADC_CLUS
_NEG_INF = float("-inf")


@dsl_user_op
def _fmaf(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        mlir_math.fma(
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            fastmath=mlir_arith.FastMathFlags.none,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pin_i64(v, *, loc=None, ip=None):
    return cutlass.Int64(
        mlir_llvm.inline_asm(
            T.i64(),
            [v.ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, $1;",
            "=l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pin_i32(v, *, loc=None, ip=None):
    return cutlass.Int32(
        mlir_llvm.inline_asm(
            T.i32(),
            [v.ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ge16x2_or_asm(m, pair, threshold_pair, *, asm_str, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                m.ir_value(loc=loc, ip=ip),
                pair.ir_value(loc=loc, ip=ip),
                threshold_pair.ir_value(loc=loc, ip=ip),
            ],
            asm_str,
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def bf16x2_ge_or(m, pair, threshold_pair, shift):
    asm_str = (
        "{\n\t.reg .pred p, q;\n\tsetp.ge.bf16x2 p|q, $2, $3;\n\tmov.b32 $0, $1;\n\t@p or.b32 $0, $0, 0x%08x;\n\t@q or.b32 $0, $0, 0x%08x;\n}"
        % (1 << shift, 2 << shift)
    )
    return _ge16x2_or_asm(m, pair, threshold_pair, asm_str=asm_str)


@dsl_user_op
def _ld_g_nc_u16(gaddr, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [gaddr.ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u16 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_g_nc_v4_b32(gaddr, *, loc=None, ip=None):
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [gaddr.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (cutlass.Uint32(llvm.extractvalue(T.i32(), ret, [i])) for i in range(4))
    )


def ld_g_b16x8_pk(copy_atom, base_addr, v_idx, pk):
    a, b, c, d = _ld_g_nc_v4_b32(base_addr + cutlass.Int64(v_idx) * cutlass.Int64(16))
    pk[0] = a
    pk[1] = b
    pk[2] = c
    pk[3] = d


@cute.jit
def bf16_ceilx2_from_f32(x):
    bits = u32_of_f32(x)
    hi = bits >> cutlass.Uint32(16)
    if bits >> cutlass.Uint32(31) == cutlass.Uint32(0):
        if bits & cutlass.Uint32(65535) != cutlass.Uint32(0):
            hi = hi + cutlass.Uint32(1)
    return hi | hi << cutlass.Uint32(16)


def ld_g_bf16x8_pair(copy_atom, base_addr, v_idx, fa, fb):
    a, b, c, d = _ld_g_nc_v4_b32(base_addr + cutlass.Int64(v_idx) * cutlass.Int64(16))
    fa[0] = f32_of_u32(a << cutlass.Uint32(16))
    fa[1] = f32_of_u32(a & cutlass.Uint32(4294901760))
    fa[2] = f32_of_u32(b << cutlass.Uint32(16))
    fa[3] = f32_of_u32(b & cutlass.Uint32(4294901760))
    fb[0] = f32_of_u32(c << cutlass.Uint32(16))
    fb[1] = f32_of_u32(c & cutlass.Uint32(4294901760))
    fb[2] = f32_of_u32(d << cutlass.Uint32(16))
    fb[3] = f32_of_u32(d & cutlass.Uint32(4294901760))


def ldg_bf16(base_addr, idx):
    u = _ld_g_nc_u16(base_addr + cutlass.Int64(idx) * cutlass.Int64(2))
    return f32_of_u32(u << cutlass.Uint32(16))


@dsl_user_op
def widen_f16(u, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [u.ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 h; cvt.u16.u32 h, $1; cvt.f32.f16 $0, h; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def widen_f16_hi(u, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [u.ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 l, h; mov.b32 {l, h}, $1; cvt.f32.f16 $0, h; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ldg_f16_widen(gaddr, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [gaddr.ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 h; ld.global.nc.b16 h, [$1]; cvt.f32.f16 $0, h; }",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f16_bits_rn(value, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 h; cvt.rn.f16.f32 h, $1; cvt.u32.u16 $0, h; }",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f16_ceilx2_bits(x, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [x.ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 h; cvt.rp.f16.f32 h, $1; mov.b32 $0, {h, h}; }",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def ld_g_f16x8_pair(copy_atom, base_addr, v_idx, fa, fb):
    a, b, c, d = _ld_g_nc_v4_b32(base_addr + cutlass.Int64(v_idx) * cutlass.Int64(16))
    fa[0] = widen_f16(a)
    fa[1] = widen_f16_hi(a)
    fa[2] = widen_f16(b)
    fa[3] = widen_f16_hi(b)
    fb[0] = widen_f16(c)
    fb[1] = widen_f16_hi(c)
    fb[2] = widen_f16(d)
    fb[3] = widen_f16_hi(d)


def ldg_f16(base_addr, idx):
    return _ldg_f16_widen(base_addr + cutlass.Int64(idx) * cutlass.Int64(2))


def ldg_f16_code(base_addr, idx):
    u = _ld_g_nc_u16(base_addr + cutlass.Int64(idx) * cutlass.Int64(2))
    return (widen_f16(u), u)


def _is_f16(dtype) -> bool:
    assert dtype in (cutlass.BFloat16, cutlass.Float16), dtype
    return dtype is cutlass.Float16


def ld_g_16x8_pair(dtype, copy_atom, base_addr, v_idx, fa, fb):
    if _is_f16(dtype):
        ld_g_f16x8_pair(copy_atom, base_addr, v_idx, fa, fb)
    else:
        ld_g_bf16x8_pair(copy_atom, base_addr, v_idx, fa, fb)


def ldg_16(dtype, base_addr, idx):
    if _is_f16(dtype):
        return ldg_f16(base_addr, idx)
    return ldg_bf16(base_addr, idx)


def ldg_16_code(dtype, base_addr, idx):
    if _is_f16(dtype):
        return ldg_f16_code(base_addr, idx)
    return (ldg_bf16(base_addr, idx), None)


def fkey16_of_code16(c):
    sign = c >> cutlass.Uint32(15)
    return (c ^ (cutlass.Uint32(0) - sign | cutlass.Uint32(32768))) & cutlass.Uint32(
        65535
    )


def pack_val16_idx(dtype, v, idx, code=None):
    if _is_f16(dtype):
        if code is None:
            return f16_bits_rn(v) << cutlass.Uint32(16) | cutlass.Uint32(idx)
        return code << cutlass.Uint32(16) | cutlass.Uint32(idx)
    return u32_of_f32(v) | cutlass.Uint32(idx)


def val_of_hi16(dtype, w):
    if _is_f16(dtype):
        return widen_f16_hi(w)
    return f32_of_u32(w & cutlass.Uint32(4294901760))


def ceil16x2_of_f32(dtype, x):
    if _is_f16(dtype):
        return f16_ceilx2_bits(x)
    return bf16_ceilx2_from_f32(x)


# Oversample beyond 2048 to bound crossing-bucket refinement.
PUBLICATION_TOP_K = 2048
SAMPLE_TARGET = 2816
SAMPLE_FACTOR = 64


class BlockPublicationKernel:
    def __init__(self, columns, dtype=cutlass.BFloat16, parallel_rows=0):
        self.parallel_rows = parallel_rows
        self.dtype = dtype
        self.radix_key_shift = 32 if dtype == cutlass.Float16 else 48
        self.v16 = columns <= 65536
        self.publication_words = (columns + 8191) // 8192

    @cute.jit
    def _emitc(self, xv, idx, pos, TF, SC, s_hist, s_cbuf, code):
        SCPB = 8192
        NBS = 256
        bn_u = C.f2u_rz((xv - TF) * SC)
        if bn_u > cutlass.Uint32(NBS - 1):
            bn_u = cutlass.Uint32(NBS - 1)
        bn = cutlass.Int32(bn_u)
        C.atomic_add_cta(s_hist.iterator + bn, cutlass.Int32(1))
        ps = pos
        if ps > cutlass.Int32(SCPB):
            ps = Int32(SCPB)
        if cutlass.const_expr(self.v16):
            s_cbuf[ps] = cutlass.Int32(pack_val16_idx(self.dtype, xv, idx, code))
        else:
            s_cbuf[ps] = cutlass.Int32(bn_u << cutlass.Uint32(22) | cutlass.Uint32(idx))
        return pos + cutlass.Int32(1)

    @cute.jit
    def _ballot_pair_emit(
        self, p1, p2, idv, base1, cap1, base2, cap2, out_row, s_scal, lane
    ):
        n1 = C.ballot(p1 != cutlass.Int32(0))
        n2 = C.ballot(p2 != cutlass.Int32(0))
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        if lane == cutlass.Int32(0):
            if n1 != cutlass.Int32(0):
                b1 = C.atomic_add_cta(s_scal.iterator + 1, cutlass.Int32(C.popc(n1)))
            if n2 != cutlass.Int32(0):
                b2 = C.atomic_add_cta(s_scal.iterator + 2, cutlass.Int32(C.popc(n2)))
        b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
        b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
        lm = cutlass.Int32(cute.arch.lanemask_lt())
        if p1 != cutlass.Int32(0):
            p = b1 + cutlass.Int32(C.popc(n1 & lm))
            if p < cap1:
                out_row[base1 + p] = idv
        if p2 != cutlass.Int32(0):
            p = b2 + cutlass.Int32(C.popc(n2 & lm))
            if p < cap2:
                out_row[base2 + p] = idv

    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        ends: cute.Tensor,
        out: cute.Tensor,
        valid_lengths: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        starts: cute.Tensor,
        aim_base: cutlass.Int32,
        sfac: cutlass.Int32,
    ):
        if cutlass.const_expr(self.parallel_rows):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            first_row = cute.arch.block_idx()[1]
            for row in cutlass.range(
                first_row, logits.shape[0], cute.arch.grid_dim()[1]
            ):
                self._row(
                    logits,
                    ends,
                    out,
                    valid_lengths,
                    n,
                    npad,
                    k,
                    starts,
                    aim_base,
                    sfac,
                    row,
                )
                cute.arch.barrier()
        else:
            self._row(
                logits,
                ends,
                out,
                valid_lengths,
                n,
                npad,
                k,
                starts,
                aim_base,
                sfac,
                cute.arch.block_idx()[1],
            )

    @cute.jit
    def _row(
        self,
        logits: cute.Tensor,
        ends: cute.Tensor,
        out: cute.Tensor,
        valid_lengths: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        starts: cute.Tensor,
        aim_base: cutlass.Int32,
        sfac: cutlass.Int32,
        row: cutlass.Int32,
    ):
        SMP = cutlass.Int32(0)
        SS2 = cutlass.Int32(1)
        TGT = cutlass.Int32(0)
        TGT2 = cutlass.Int32(0)
        xv = cutlass.Float32(0.0)
        bq = cutlass.Int32(0)
        idv = cutlass.Int32(0)
        BLK = 256
        U = 4
        NBS = 256
        NBD = 256
        SCPB = 8192
        CMPB = 448 if self.parallel_rows else 1024
        PFD = 0
        NATT = 3
        NW = BLK // 32
        tidx, _, _ = cute.arch.thread_idx()
        part = cutlass.Int32(0)
        lane = tidx & cutlass.Int32(31)
        short = Int32(0)
        n_row = Int32(0)
        # Raw publication requires zero-based rows.
        raw_end = ends[row]
        ke = (raw_end + 7) // 8 if raw_end > starts[row] else cutlass.Int32(0)
        nv = max(0, min(ke, npad))
        n_row = nv
        if nv <= k:
            short = cutlass.Int32(1)
            n = cutlass.Int32(0)
            SMP = cutlass.Int32(0)
            SS2 = cutlass.Int32(1)
            TGT = cutlass.Int32(1073741823)
            TGT2 = cutlass.Int32(1073741823)
        if short == cutlass.Int32(0):
            n = nv
            SMP = cutlass.Int32(0)
            SS2 = cutlass.Int32(1)
            TGT = cutlass.Int32(0)
            TGT2 = cutlass.Int32(0)
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_ordered_layout((256,), order=(0,)),
            byte_alignment=128,
        )
        # Preserve padding allocations to keep the shared histogram offsets stable.
        _s_ws = smem.allocate_tensor(
            cutlass.Uint32,
            cute.make_ordered_layout((NW,), order=(0,)),
            byte_alignment=16,
        )
        s_wmn = smem.allocate_tensor(
            cutlass.Uint32,
            cute.make_ordered_layout((NW,), order=(0,)),
            byte_alignment=16,
        )
        s_wmx = smem.allocate_tensor(
            cutlass.Uint32,
            cute.make_ordered_layout((NW,), order=(0,)),
            byte_alignment=16,
        )
        s_res = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((8,), order=(0,)), byte_alignment=16
        )
        s_scal = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        _s_pk = smem.allocate_tensor(
            cutlass.Int64, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=8
        )
        s_tsh = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=4,
        )
        s_kmm = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((2,), order=(0,)), byte_alignment=8
        )
        s_lad = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        # Keep the padding word so the following shared buffer retains its offset.
        _s_lead = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
        )
        blob = smem.allocate_tensor(
            cutlass.Int8,
            cute.make_ordered_layout(
                (36376 if self.parallel_rows else 40984,), order=(0,)
            ),
            byte_alignment=16,
        )
        sbase = blob.iterator.toint()
        s_cbuf = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int32, sbase, cute.AddressSpace.smem, assumed_align=16
            ),
            cute.make_layout((SCPB + 4,)),
        )
        ck_addr = sbase + cutlass.Int32(32784)
        s_ck64 = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint64, ck_addr, cute.AddressSpace.smem, assumed_align=16
            ),
            cute.make_layout((CMPB + 1,)),
        )
        row64 = cutlass.Int64(row)
        x_addr = _pin_i64(
            logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(2)
        )
        out_row = out[row, None]
        n4 = n >> Int32(3)
        c0 = cutlass.Int32(0)
        c1 = n4
        tail0 = n4 << cutlass.Int32(3)
        tailn = cutlass.Int32(0)
        if part == cutlass.Int32(0):
            tailn = n - tail0
        if tidx == cutlass.Int32(0):
            s_scal[0] = Int32(0)
            s_res[C.RES_B2] = cutlass.Int32(-1)
            s_res[C.RES_B3] = cutlass.Int32(-1)
        if tidx < Int32(256):
            s_hist[tidx] = cutlass.Int32(0)
        plim4 = cutlass.Int32(0)
        pic = cutlass.Int32(0)
        if tidx < cutlass.Int32(32):
            if short == cutlass.Int32(0):
                x6 = cutlass.Int32(6) * n
                ri = cutlass.Int32(cmath.sqrt(cutlass.Float32(x6)))
                if ri * ri > x6:
                    ri = ri - cutlass.Int32(1)
                if (ri + cutlass.Int32(1)) * (ri + cutlass.Int32(1)) <= x6:
                    ri = ri + cutlass.Int32(1)
                r6 = ri
                if x6 - ri * ri > ri:
                    r6 = ri + cutlass.Int32(1)
                aim = aim_base
                if r6 > aim:
                    aim = r6
                scap_c = cutlass.Int32(8192)
                if aim > scap_c >> cutlass.Int32(1):
                    aim = scap_c >> cutlass.Int32(1)
                if aim < k:
                    aim = k
                n4w = n >> cutlass.Int32(2)
                gate = cutlass.Int32(0)
                if n > scap_c:
                    gate = cutlass.Int32(1)
                if n <= scap_c:
                    if n > k << cutlass.Int32(1):
                        gate = cutlass.Int32(1)
                if n4w < cutlass.Int32(4):
                    gate = cutlass.Int32(0)
                if gate != cutlass.Int32(0):
                    sel = cutlass.Int32(
                        cutlass.Float32(sfac * n)
                        * cute.arch.rcp_approx(cutlass.Float32(aim))
                    )
                    if sel < cutlass.Int32(256):
                        sel = cutlass.Int32(256)
                    nh = n >> cutlass.Int32(1)
                    if sel > nh:
                        sel = nh
                    pairs = sel >> cutlass.Int32(3)
                    if pairs < cutlass.Int32(1):
                        pairs = cutlass.Int32(1)
                    half = n4w >> cutlass.Int32(1)
                    if half < cutlass.Int32(1):
                        half = cutlass.Int32(1)
                    if pairs > half:
                        pairs = half
                    SS2 = cutlass.Int32(
                        cutlass.Float32(half)
                        * cute.arch.rcp_approx(cutlass.Float32(pairs))
                    )
                    if cutlass.Int32(1) > SS2:
                        SS2 = cutlass.Int32(1)
                    SMP = cutlass.Int32(
                        cutlass.Float32(half)
                        * cute.arch.rcp_approx(cutlass.Float32(SS2))
                    )
                    if half < SMP * SS2:
                        SMP = SMP - cutlass.Int32(1)
                    if cutlass.Int32(1) > SMP:
                        SMP = cutlass.Int32(1)
                    rn_ = cute.arch.rcp_approx(cutlass.Float32(n))
                    smp8f = cutlass.Float32(SMP) * cutlass.Float32(8.0)
                    TGT = cutlass.Int32(cutlass.Float32(aim) * smp8f * rn_)
                    if cutlass.Int32(1) > TGT:
                        TGT = cutlass.Int32(1)
                    TGT2 = cutlass.Int32(cutlass.Float32(k) * smp8f * rn_)
                    if cutlass.Int32(1) > TGT2:
                        TGT2 = cutlass.Int32(1)
            if tidx == cutlass.Int32(0):
                s_lad[0] = SMP
                s_lad[1] = SS2
                s_lad[2] = TGT
                s_lad[3] = TGT2
        plim4 = n4 - cutlass.Int32(1)
        if plim4 < cutlass.Int32(0):
            plim4 = cutlass.Int32(0)
        for uu in cutlass.range_constexpr(U):
            pic = c0 + tidx + Int32(uu * BLK)
            if pic >= c1:
                pic = plim4
            C._prefetch_l2(x_addr + Int64(pic) * Int64(16))
        cute.arch.barrier()
        SMP = s_lad[0]
        SS2 = s_lad[1]
        TGT = s_lad[2]
        TGT2 = s_lad[3]
        atom128 = C.g2r_atom_f32(128, invariant=True)
        fsa = cute.make_rmem_tensor((4,), cutlass.Float32)
        fsb = cute.make_rmem_tensor((4,), cutlass.Float32)
        shas = Int32(0)
        if tidx < SMP:
            shas = Int32(1)
        if shas != Int32(0):
            ld_g_16x8_pair(self.dtype, atom128, x_addr, tidx * SS2, fsa, fsb)
        smn = cutlass.Float32(float("inf"))
        smx = cutlass.Float32(float("-inf"))
        if shas != Int32(0):
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fsa[t])
                smx = C.fmax_f32(smx, fsa[t])
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fsb[t])
                smx = C.fmax_f32(smx, fsb[t])
        fma_ = cute.make_rmem_tensor((4,), cutlass.Float32)
        fmb_ = cute.make_rmem_tensor((4,), cutlass.Float32)
        j = tidx + Int32(BLK)
        while j < SMP:
            ld_g_16x8_pair(self.dtype, atom128, x_addr, j * SS2, fma_, fmb_)
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fma_[t])
                smx = C.fmax_f32(smx, fma_[t])
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fmb_[t])
                smx = C.fmax_f32(smx, fmb_[t])
            j = j + Int32(BLK)
        a0 = C.warp_min_u32(C.fkey(smn))
        c0m = C.warp_max_u32(C.fkey(smx))
        if lane == Int32(0):
            s_wmn[tidx >> Int32(5)] = a0
            s_wmx[tidx >> Int32(5)] = c0m
        cute.arch.barrier()
        lim4 = n4 - cutlass.Int32(1)
        if lim4 < cutlass.Int32(0):
            lim4 = cutlass.Int32(0)
        pf = [cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(max(PFD, 1))]
        fullsl = cutlass.Int32(0)
        if c1 - c0 >= cutlass.Int32(BLK * U):
            fullsl = cutlass.Int32(1)
        if fullsl != Int32(0):
            for uu in cutlass.range_constexpr(U):
                C._prefetch_l2(
                    x_addr
                    + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK))
                    * cutlass.Int64(16)
                )
        elif Int32(0) < SMP:
            for uu in cutlass.range_constexpr(U):
                i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                ic = i_
                if ic >= c1:
                    ic = lim4
                C._prefetch_l2(x_addr + cutlass.Int64(ic) * cutlass.Int64(16))
        av = cutlass.Uint32(4294967295)
        cv = cutlass.Uint32(0)
        if lane < cutlass.Int32(NW):
            av = s_wmn[lane]
            cv = s_wmx[lane]
        SMIN = C.invkey(C.warp_min_u32(av))
        SMAX = C.invkey(C.warp_max_u32(cv))
        GMIN = cutlass.Float32(C.SENT_LO)
        GMAX = cutlass.Float32(C.SENT_HI)
        T = cutlass.Float32(_NEG_INF)
        HIC = cutlass.Float32(_NEG_INF)
        w = cutlass.Float32(0.0)
        sok = cutlass.Int32(0)
        if cutlass.Int32(0) < SMP:
            if SMAX > SMIN:
                sok = cutlass.Int32(1)
        if sok != Int32(0):
            w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
            sc_s = cute.arch.rcp_approx(w)
            if shas != Int32(0):
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fsa[t] - SMIN) * sc_s)
                    if bq > Int32(NBS - 1):
                        bq = Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, Int32(1))
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fsb[t] - SMIN) * sc_s)
                    if bq > Int32(NBS - 1):
                        bq = Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, Int32(1))
            j = tidx + Int32(BLK)
            while j < SMP:
                ld_g_16x8_pair(self.dtype, atom128, x_addr, j * SS2, fma_, fmb_)
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fma_[t] - SMIN) * sc_s)
                    if bq > Int32(NBS - 1):
                        bq = Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, Int32(1))
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fmb_[t] - SMIN) * sc_s)
                    if bq > Int32(NBS - 1):
                        bq = Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, Int32(1))
                j = j + Int32(BLK)
        cute.arch.barrier()
        C.scan_cross0(
            s_hist,
            TGT,
            tidx,
            s_res,
            TGT2,
            TGT * cutlass.Int32(2),
            s_hist,
            nb=NBS,
            zero=True,
            two=True,
            three=True,
        )
        cute.arch.barrier()
        tot0 = s_res[C.RES_TOT]
        b1v = s_res[C.RES_B]
        if sok != cutlass.Int32(0):
            if tot0 >= TGT:
                T = _fmaf(cutlass.Float32(b1v), w, SMIN)
        Trung = T
        _tkc = cutlass.Int32(-1)
        needg = Int32(1)
        if cutlass.Float32(_NEG_INF) < T:
            needg = cutlass.Int32(0)
        if needg != cutlass.Int32(0):
            T = GMIN
        if sok != Int32(0):
            if tot0 >= TGT:
                b2v = s_res[C.RES_B2]
                if b2v >= cutlass.Int32(0):
                    Tk = _fmaf(cutlass.Float32(b2v), w, SMIN)
                    anch = T
                    anch = C.fmin_f32(T, Trung)
                    d_ = C.fmax_f32(Tk - anch, cutlass.Float32(0.0))
                    HIC = C.fmax_f32(
                        _fmaf(cutlass.Float32(4.0), d_, T),
                        _fmaf(cutlass.Float32(8.0), w, T),
                    )
        if tidx == cutlass.Int32(0):
            t5 = cutlass.Float32(_NEG_INF)
            if sok != cutlass.Int32(0):
                if tot0 >= TGT * cutlass.Int32(2):
                    b3v = s_res[C.RES_B3]
                    if b3v >= cutlass.Int32(0):
                        if T > GMIN:
                            T3 = _fmaf(cutlass.Float32(b3v), w, SMIN)
                            if T3 < T:
                                t5 = T3
            s_tsh[0] = t5
        listN = cutlass.Int32(0)
        above = cutlass.Int32(0)
        m = cutlass.Int32(0)
        need = cutlass.Int32(0)
        B = cutlass.Int32(0)
        SC = cutlass.Float32(1.0)
        TF = T
        complete = cutlass.Int32(0)
        valid = cutlass.Int32(0)
        _fromg = cutlass.Int32(0)
        anch = Int32(0)
        alive = cutlass.Int32(1)
        fr = [
            cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(max(U - PFD, 1))
        ]
        att = cutlass.Int32(0)
        running = cutlass.Int32(1)
        while running != cutlass.Int32(0):
            if att > Int32(0):
                if tidx < cutlass.Int32(NBS):
                    s_hist[tidx] = cutlass.Int32(0)
                if tidx == cutlass.Int32(0):
                    s_scal[0] = cutlass.Int32(0)
                cute.arch.barrier()
            TF = T
            tf_pair16 = ceil16x2_of_f32(self.dtype, TF)
            hi = C.fmax_f32(GMAX, T)
            if HIC > T:
                if hi > HIC:
                    hi = HIC
            WD = (hi - T) * cutlass.Float32(1.0 / 256.0)
            wdok = cutlass.Int32(0)
            if cutlass.Float32(0.0) < WD:
                wdok = cutlass.Int32(1)
            if wdok == cutlass.Int32(0):
                WD = cutlass.Float32(1e-30)
            SC = cutlass.Float32(1.0) / WD
            if wdok != cutlass.Int32(0):
                TF = T - WD
                tf_pair16 = ceil16x2_of_f32(self.dtype, TF)
            span = c1 - c0
            step = cutlass.Int32(BLK * U)
            nFull = cutlass.Int32(0)
            rem = cutlass.Int32(0)
            if span > Int32(0):
                nFull = span // step
                rem = span - nFull * step
            nFull = _pin_i32(nFull)
            nIt = nFull
            if rem > cutlass.Int32(0):
                nIt = nIt + cutlass.Int32(1)
            if rem == cutlass.Int32(1):
                nIt = nFull
            nIt = _pin_i32(nIt)
            it = cutlass.Int32(0)
            while it < nIt:
                i0 = c0 + it * step + tidx
                M = cutlass.Int32(0)
                isfull = cutlass.Int32(0)
                if it < nFull:
                    isfull = cutlass.Int32(1)
                if isfull != Int32(0):
                    for uu in cutlass.range_constexpr(PFD, U):
                        ld_g_b16x8_pk(
                            atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD]
                        )
                    for uu in cutlass.range_constexpr(U):
                        if cutlass.const_expr(uu < PFD):
                            vv = pf[uu]
                        else:
                            vv = fr[uu - PFD]
                        for q in cutlass.range_constexpr(4):
                            M = bf16x2_ge_or(M, vv[q], tf_pair16, uu * 8 + q * 2)
                else:
                    for uu in cutlass.range_constexpr(PFD, U):
                        i_ = i0 + cutlass.Int32(uu * BLK)
                        ic = i_
                        if ic >= c1:
                            ic = lim4
                        ld_g_b16x8_pk(atom128, x_addr, ic, fr[uu - PFD])
                    for uu in cutlass.range_constexpr(U):
                        if cutlass.const_expr(uu < PFD):
                            vv = pf[uu]
                        else:
                            vv = fr[uu - PFD]
                        i_ = i0 + cutlass.Int32(uu * BLK)
                        okq = cutlass.Int32(0)
                        if i_ < c1:
                            okq = cutlass.Int32(1)
                        if okq != Int32(0):
                            for q in cutlass.range_constexpr(4):
                                M = bf16x2_ge_or(M, vv[q], tf_pair16, uu * 8 + q * 2)
                cnt = cutlass.Int32(C.popc(M))
                inc = _bf16_emit_prefix_agg(cnt)
                bpos = cutlass.Int32(0)
                if lane == cutlass.Int32(31):
                    if inc != cutlass.Int32(0):
                        bpos = C.atomic_add_cta(s_scal.iterator + 0, inc)
                pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                if cutlass.Int32(0) != M:
                    bp = C.ffs_m1(M)
                    M = M & M - cutlass.Int32(1)
                    idx = (
                        i0 + (bp >> cutlass.Int32(3)) * cutlass.Int32(BLK)
                        << cutlass.Int32(3)
                    ) + (bp & cutlass.Int32(7))
                    xv, xc = ldg_16_code(self.dtype, x_addr, idx)
                    while cutlass.Int32(0) != M:
                        bp2 = C.ffs_m1(M)
                        M = M & M - cutlass.Int32(1)
                        idx2 = (
                            i0 + (bp2 >> cutlass.Int32(3)) * cutlass.Int32(BLK)
                            << cutlass.Int32(3)
                        ) + (bp2 & cutlass.Int32(7))
                        xv2, xc2 = ldg_16_code(self.dtype, x_addr, idx2)
                        pos = self._emitc(xv, idx, pos, TF, SC, s_hist, s_cbuf, xc)
                        idx = idx2
                        xv = xv2
                        xc = xc2
                    pos = self._emitc(xv, idx, pos, TF, SC, s_hist, s_cbuf, xc)
                it = it + cutlass.Int32(1)
            i = tidx
            if rem == cutlass.Int32(1):
                i = i - cutlass.Int32(8)
            while i < tailn:
                x, xtc = ldg_16_code(self.dtype, x_addr, tail0 + i)
                if x >= TF:
                    post = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                    post = self._emitc(x, tail0 + i, post, TF, SC, s_hist, s_cbuf, xtc)
                i = i + cutlass.Int32(BLK)
            cute.arch.barrier()
            myn = s_scal[0]
            C.scan_cross0(
                s_hist,
                k,
                tidx,
                s_res,
                cutlass.Int32(0),
                cutlass.Int32(0),
                s_hist,
                nb=NBD,
                zero=False,
            )
            cute.arch.barrier()
            tot = s_res[C.RES_TOT]
            acc = cutlass.Int32(0)
            if tot >= k:
                acc = cutlass.Int32(1)
            if acc != Int32(0):
                valid = cutlass.Int32(1)
                complete = cutlass.Int32(0)
                if myn <= cutlass.Int32(SCPB):
                    complete = cutlass.Int32(1)
                listN = myn
                above = s_res[C.RES_ABOVE]
                m = s_res[C.RES_M]
                need = k - above
                B = s_res[C.RES_B]
                running = cutlass.Int32(0)
            elif att == Int32(NATT - 1):
                running = cutlass.Int32(0)
            else:
                tshtaken = Int32(0)
                if att == cutlass.Int32(0):
                    T5 = s_tsh[0]
                    if cutlass.Float32(_NEG_INF) < T5:
                        if T5 < TF:
                            T = T5
                            tshtaken = cutlass.Int32(1)
                if tshtaken != cutlass.Int32(0):
                    cute.arch.barrier()
                else:
                    floorhit = cutlass.Int32(1)
                    if T > GMIN:
                        floorhit = cutlass.Int32(0)
                    if floorhit != cutlass.Int32(0):
                        running = cutlass.Int32(0)
                    else:
                        T = GMIN
                        cute.arch.barrier()
            att = att + cutlass.Int32(1)
        if alive != cutlass.Int32(0):
            whole = cutlass.Int32(0)
            if valid != cutlass.Int32(0):
                if need >= m:
                    whole = cutlass.Int32(1)
            lim1 = above
            if whole != cutlass.Int32(0):
                lim1 = above + m
            degen = cutlass.Int32(0)
            if valid == cutlass.Int32(0):
                degen = cutlass.Int32(1)
            if m > cutlass.Int32(CMPB):
                degen = cutlass.Int32(1)
            mc = cutlass.Int32(0)
            if degen == cutlass.Int32(0):
                mc = m
            if degen == cutlass.Int32(0):
                if complete != cutlass.Int32(0):
                    i = tidx
                    while i < listN:
                        idv = cutlass.Int32(0)
                        bq = cutlass.Int32(0)
                        xv = cutlass.Float32(0.0)
                        if cutlass.const_expr(self.v16):
                            wpk = cutlass.Uint32(s_cbuf[i])
                            idv = cutlass.Int32(wpk & cutlass.Uint32(65535))
                            xv = val_of_hi16(self.dtype, wpk)
                            bq = C.f2s_rz((xv - TF) * SC)
                            if bq > cutlass.Int32(NBD - 1):
                                bq = cutlass.Int32(NBD - 1)
                        else:
                            wpk = cutlass.Uint32(s_cbuf[i])
                            idv = cutlass.Int32(wpk & cutlass.Uint32(4194303))
                            bq = cutlass.Int32(wpk >> cutlass.Uint32(22))
                        if bq >= B:
                            p = C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                            if p < lim1:
                                out_row[p] = idv
                            elif whole == cutlass.Int32(0):
                                q2 = p - above
                                if q2 < cutlass.Int32(CMPB):
                                    if cutlass.const_expr(self.v16):
                                        kk = C.fkey(xv)
                                    else:
                                        kk = C.fkey(ldg_16(self.dtype, x_addr, idv))
                                    s_ck64[q2] = cutlass.Uint64(kk) << cutlass.Uint64(
                                        32
                                    ) | cutlass.Uint64(cutlass.Uint32(idv))
                        i = i + cutlass.Int32(BLK)
                else:
                    lo2 = c0 << cutlass.Int32(3)
                    hi2 = c1 << cutlass.Int32(3)
                    i0_ = lo2 + tidx
                    while i0_ < hi2 + tailn:
                        i_ = i0_
                        if i0_ >= hi2:
                            i_ = tail0 + (i0_ - hi2)
                        x, xtc = ldg_16_code(self.dtype, x_addr, i_)
                        if x >= TF:
                            bq = C.f2s_rz((x - TF) * SC)
                            if bq > cutlass.Int32(NBD - 1):
                                bq = cutlass.Int32(NBD - 1)
                            if bq >= B:
                                p = C.atomic_add_cta(
                                    s_hist.iterator + bq, cutlass.Int32(1)
                                )
                                if p < lim1:
                                    out_row[p] = i_
                                elif whole == cutlass.Int32(0):
                                    q2 = p - above
                                    if q2 < cutlass.Int32(CMPB):
                                        s_ck64[q2] = cutlass.Uint64(
                                            C.fkey(x)
                                        ) << cutlass.Uint64(32) | cutlass.Uint64(
                                            cutlass.Uint32(i_)
                                        )
                        i0_ = i0_ + cutlass.Int32(BLK)
                if whole == cutlass.Int32(0):
                    cute.arch.barrier()
                    if mc <= Int32(QUADC_CLUS__main):
                        mc2 = mc & cutlass.Int32(~1)
                        i = tidx
                        while i < mc:
                            u64v = s_ck64[i]
                            r_ = cutlass.Int32(0)
                            jq = cutlass.Int32(0)
                            while jq < mc2:
                                vlo, vhi = C._lds_v2_u64(
                                    ck_addr + jq * cutlass.Int32(8)
                                )
                                r_ = (
                                    r_
                                    + cutlass.Int32(vlo > cutlass.Uint64(u64v))
                                    + cutlass.Int32(vhi > cutlass.Uint64(u64v))
                                )
                                jq = jq + cutlass.Int32(2)
                            if mc2 < mc:
                                r_ = r_ + cutlass.Int32(
                                    cutlass.Uint64(s_ck64[mc2]) > cutlass.Uint64(u64v)
                                )
                            if r_ < need:
                                idv6 = cutlass.Int32(
                                    cutlass.Uint32(
                                        cutlass.Uint64(u64v)
                                        & cutlass.Uint64(4294967295)
                                    )
                                )

                                out_row[above + r_] = idv6
                            i = i + cutlass.Int32(BLK)
                    else:
                        if tidx == cutlass.Int32(0):
                            s_kmm[0] = cutlass.Uint32(4294967295)
                            s_kmm[1] = cutlass.Uint32(0)
                        if tidx < Int32(NBS):
                            s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()
                        i = tidx
                        while i < mc:
                            kk = cutlass.Uint32(
                                s_ck64[i] >> cutlass.Uint64(self.radix_key_shift)
                            )
                            C.atomic_min_cta(s_kmm.iterator + 0, kk)
                            C.atomic_max_cta(s_kmm.iterator + 1, kk)
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()
                        rlo = s_kmm[0]
                        rhi = s_kmm[1]
                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                        aboveC = cutlass.Int32(0)
                        needC = need
                        mm = mc
                        brk = cutlass.Int32(0)
                        lev = cutlass.Int32(0)
                        while brk == Int32(0):
                            if needC == mm:
                                ethr = cutlass.Int64(
                                    cutlass.Uint32(rlo)
                                ) - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                brk = cutlass.Int32(1)
                            elif cutlass.Uint32(rlo) >= cutlass.Uint32(
                                rhi
                            ) or lev >= cutlass.Int32(6):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            else:
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2_ = cutlass.Int32(32) - C.clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = b2_ - cutlass.Int32(8)
                                if sh2 < cutlass.Int32(0):
                                    sh2 = cutlass.Int32(0)
                                sh2u = cutlass.Uint32(sh2)
                                i = tidx
                                while i < mc:
                                    uq = cutlass.Uint32(
                                        s_ck64[i]
                                        >> cutlass.Uint64(self.radix_key_shift)
                                    )
                                    if uq >= cutlass.Uint32(rlo):
                                        if uq <= cutlass.Uint32(rhi):
                                            du = uq - cutlass.Uint32(rlo) >> sh2u
                                            if du > cutlass.Uint32(NBS - 1):
                                                du = cutlass.Uint32(NBS - 1)
                                            C.atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(du),
                                                cutlass.Int32(1),
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()
                                C.scan_cross0(
                                    s_hist,
                                    needC,
                                    tidx,
                                    s_res,
                                    cutlass.Int32(0),
                                    cutlass.Int32(0),
                                    s_hist,
                                    nb=NBS,
                                    zero=True,
                                )
                                cute.arch.barrier()
                                aboveC = aboveC + s_res[C.RES_ABOVE]
                                needC = needC - s_res[C.RES_ABOVE]
                                mm = s_res[C.RES_M]
                                sB = s_res[C.RES_B]
                                nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                                if sB != cutlass.Int32(NBS - 1):
                                    rhi = nlo + (
                                        (cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1)
                                    )
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        if tidx == cutlass.Int32(0):
                            s_scal[1] = cutlass.Int32(0)
                            s_scal[2] = cutlass.Int32(0)
                        cute.arch.barrier()
                        it2 = (mc + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tidx
                            p1 = cutlass.Int32(0)
                            p2 = cutlass.Int32(0)
                            idv = cutlass.Int32(0)
                            if i < mc:
                                w64 = s_ck64[i]
                                iu = cutlass.Int64(
                                    cutlass.Uint32(
                                        w64 >> cutlass.Uint64(self.radix_key_shift)
                                    )
                                )
                                idv = cutlass.Int32(
                                    cutlass.Uint32(w64 & cutlass.Uint64(4294967295))
                                )
                                if iu > ethr:
                                    p1 = cutlass.Int32(1)
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)

                            self._ballot_pair_emit(
                                p1,
                                p2,
                                idv,
                                above,
                                aboveC,
                                above + aboveC,
                                needC,
                                out_row,
                                s_scal,
                                lane,
                            )
                            it = it + cutlass.Int32(1)
            else:
                dga = Int32(0)
                if valid != cutlass.Int32(0):
                    if complete != cutlass.Int32(0):
                        dga = cutlass.Int32(1)
                if dga != cutlass.Int32(0):
                    rlo = Uint32(0)
                    rhi = Uint32(65535)
                    above2 = Int32(0)
                    need2 = k
                    m2 = listN
                    ethr = cutlass.Int64(0)
                    tie_m = cutlass.Int32(1)
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == Int32(0):
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                            above2 = above2 + m2
                            need2 = cutlass.Int32(0)
                            tie_m = cutlass.Int32(0)
                            brk = cutlass.Int32(1)
                        elif cutlass.Uint32(rlo) >= cutlass.Uint32(
                            rhi
                        ) or lev >= cutlass.Int32(8):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        else:
                            d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                            b2_ = cutlass.Int32(32) - C.clz_i32(
                                cutlass.Int32(d2 | cutlass.Uint32(1))
                            )
                            sh2 = b2_ - cutlass.Int32(8)
                            if sh2 < cutlass.Int32(0):
                                sh2 = cutlass.Int32(0)
                            sh2u = cutlass.Uint32(sh2)
                            i = tidx
                            while i < listN:
                                uq = cutlass.Uint32(0)
                                if cutlass.const_expr(self.v16):
                                    wq = Uint32(s_cbuf[i])
                                    uq = fkey16_of_code16(wq >> Uint32(16))
                                else:
                                    id0 = cutlass.Int32(
                                        cutlass.Uint32(s_cbuf[i])
                                        & cutlass.Uint32(4194303)
                                    )
                                    uq = fkey16_of_code16(
                                        _ld_g_nc_u16(x_addr + Int64(id0) * Int64(2))
                                    )
                                if uq >= cutlass.Uint32(rlo):
                                    if uq <= cutlass.Uint32(rhi):
                                        du = uq - cutlass.Uint32(rlo) >> sh2u
                                        if du > cutlass.Uint32(NBS - 1):
                                            du = cutlass.Uint32(NBS - 1)
                                        C.atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(du),
                                            cutlass.Int32(1),
                                        )
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()
                            C.scan_cross0(
                                s_hist,
                                need2,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                s_hist,
                                nb=NBS,
                                zero=True,
                            )
                            cute.arch.barrier()
                            above2 = above2 + s_res[C.RES_ABOVE]
                            need2 = need2 - s_res[C.RES_ABOVE]
                            m2 = s_res[C.RES_M]
                            sB = s_res[C.RES_B]
                            nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                            if sB != cutlass.Int32(NBS - 1):
                                rhi = nlo + (
                                    (cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1)
                                )
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()
                    nA = k
                    nT = cutlass.Int32(0)
                    if tie_m != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (listN + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    it = cutlass.Int32(0)
                    while it < it2:
                        i = it * cutlass.Int32(BLK) + tidx
                        p1 = cutlass.Int32(0)
                        p2 = cutlass.Int32(0)
                        idv = cutlass.Int32(0)
                        if i < listN:
                            uq = cutlass.Uint32(0)
                            if cutlass.const_expr(self.v16):
                                wq = Uint32(s_cbuf[i])
                                idv = Int32(wq & Uint32(65535))
                                uq = fkey16_of_code16(wq >> Uint32(16))
                            else:
                                idv = cutlass.Int32(
                                    cutlass.Uint32(s_cbuf[i]) & cutlass.Uint32(4194303)
                                )
                                uq = fkey16_of_code16(
                                    _ld_g_nc_u16(x_addr + Int64(idv) * Int64(2))
                                )
                            iu = cutlass.Int64(uq)
                            if iu > ethr:
                                p1 = cutlass.Int32(1)
                            if tie_m != cutlass.Int32(0):
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)

                        self._ballot_pair_emit(
                            p1,
                            p2,
                            idv,
                            cutlass.Int32(0),
                            nA,
                            nA,
                            nT,
                            out_row,
                            s_scal,
                            lane,
                        )
                        it = it + cutlass.Int32(1)
                else:
                    rlo = Uint32(0)
                    rhi = Uint32(65535)
                    above2 = Int32(0)
                    need2 = k
                    m2 = n
                    ethr = cutlass.Int64(0)
                    tie_m = cutlass.Int32(1)
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == Int32(0):
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                            above2 = above2 + m2
                            need2 = cutlass.Int32(0)
                            tie_m = cutlass.Int32(0)
                            brk = cutlass.Int32(1)
                        elif cutlass.Uint32(rlo) >= cutlass.Uint32(
                            rhi
                        ) or lev >= cutlass.Int32(8):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        else:
                            d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                            b2_ = cutlass.Int32(32) - C.clz_i32(
                                cutlass.Int32(d2 | cutlass.Uint32(1))
                            )
                            sh2 = b2_ - cutlass.Int32(8)
                            if sh2 < cutlass.Int32(0):
                                sh2 = cutlass.Int32(0)
                            sh2u = cutlass.Uint32(sh2)
                            i = tidx
                            while i < n:
                                uq = fkey16_of_code16(
                                    _ld_g_nc_u16(x_addr + Int64(i) * Int64(2))
                                )
                                if uq >= cutlass.Uint32(rlo):
                                    if uq <= cutlass.Uint32(rhi):
                                        du = uq - cutlass.Uint32(rlo) >> sh2u
                                        if du > cutlass.Uint32(NBS - 1):
                                            du = cutlass.Uint32(NBS - 1)
                                        C.atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(du),
                                            cutlass.Int32(1),
                                        )
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()
                            C.scan_cross0(
                                s_hist,
                                need2,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                s_hist,
                                nb=NBS,
                                zero=True,
                            )
                            cute.arch.barrier()
                            above2 = above2 + s_res[C.RES_ABOVE]
                            need2 = need2 - s_res[C.RES_ABOVE]
                            m2 = s_res[C.RES_M]
                            sB = s_res[C.RES_B]
                            nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                            if sB != cutlass.Int32(NBS - 1):
                                rhi = nlo + (
                                    (cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1)
                                )
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()
                    nA = k
                    nT = cutlass.Int32(0)
                    if tie_m != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (n + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    it = cutlass.Int32(0)
                    while it < it2:
                        i = it * cutlass.Int32(BLK) + tidx
                        p1 = cutlass.Int32(0)
                        p2 = cutlass.Int32(0)
                        if i < n:
                            uq = fkey16_of_code16(
                                _ld_g_nc_u16(x_addr + Int64(i) * Int64(2))
                            )
                            iu = cutlass.Int64(uq)
                            if iu > ethr:
                                p1 = cutlass.Int32(1)
                            if tie_m != cutlass.Int32(0):
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)
                        self._ballot_pair_emit(
                            p1,
                            p2,
                            i,
                            cutlass.Int32(0),
                            nA,
                            nA,
                            nT,
                            out_row,
                            s_scal,
                            lane,
                        )
                        it = it + cutlass.Int32(1)
        if short != cutlass.Int32(0):
            if part == cutlass.Int32(0):
                i = tidx
                while i < n_row:
                    out_row[i] = i
                    i = i + cutlass.Int32(BLK)
                j = n_row + tidx
                while j < k:
                    out_row[j] = cutlass.Int32(-1)
                    j = j + cutlass.Int32(BLK)
        cute.arch.barrier()
        bitmap = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32, sbase, cute.AddressSpace.smem, assumed_align=16
            ),
            cute.make_layout((256 * self.publication_words,)),
        )
        prefix = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int32,
                sbase + cutlass.Int32(1024 * self.publication_words),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((8,)),
        )
        raw_end = ends[row]
        live_end = (
            min(raw_end, logits.shape[1] * 8)
            if raw_end > starts[row]
            else cutlass.Int32(0)
        )
        block_count = (live_end + 7) // 8
        publication_sort(
            row, tidx, out, bitmap, prefix, self.publication_words, block_count - 1
        )
        if tidx == 0:
            valid_lengths[row] = max(
                0, (min(block_count, 2048) - 1) * 8 + (live_end - 1) % 8 + 1
            )


@dsl_user_op
def _bf16_emit_prefix_agg(packed, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [packed.ir_value(loc=loc, ip=ip)],
            "{ .reg .pred valid; .reg .b32 other; mov.b32 $0, $1; shfl.sync.up.b32 other|valid, $0, 1, 0, 0xffffffff; @valid add.u32 $0, $0, other; shfl.sync.up.b32 other|valid, $0, 2, 0, 0xffffffff; @valid add.u32 $0, $0, other; shfl.sync.up.b32 other|valid, $0, 4, 0, 0xffffffff; @valid add.u32 $0, $0, other; shfl.sync.up.b32 other|valid, $0, 8, 0, 0xffffffff; @valid add.u32 $0, $0, other; shfl.sync.up.b32 other|valid, $0, 16, 0, 0xffffffff; @valid add.u32 $0, $0, other; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class Publication:
    def __init__(self, columns, dtype=cutlass.BFloat16, parallel_rows=0):
        self.parallel_rows = parallel_rows
        self.selector = BlockPublicationKernel(columns, dtype, parallel_rows)

    @cute.jit
    def __call__(self, scores, ends, blocks, workspace, starts, stream):
        self.selector.kern(
            scores,
            ends,
            blocks,
            workspace,
            cutlass.Int32(0),
            cutlass.Int32(scores.shape[1]),
            cutlass.Int32(PUBLICATION_TOP_K),
            starts,
            cutlass.Int32(SAMPLE_TARGET),
            cutlass.Int32(SAMPLE_FACTOR),
        ).launch(
            grid=(
                1,
                min(scores.shape[0], self.parallel_rows)
                if self.parallel_rows
                else scores.shape[0],
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
            min_blocks_per_mp=6 if self.parallel_rows else 4,
            use_pdl=bool(self.parallel_rows),
        )


@functools.cache
def compile_block_selector(dtype, columns, parallel_rows=0):
    args = [
        cute.runtime.make_fake_compact_tensor(
            dtype,
            tuple(cute.sym_int() for _ in range(ndim)),
            stride_order=tuple(reversed(range(ndim))),
            assumed_align=16,
        )
        for dtype, ndim in [
            (dtype, 2),
            (cutlass.Int32, 1),
            (cutlass.Int32, 2),
            (cutlass.Int32, 1),
            (cutlass.Int32, 1),
        ]
    ]
    if dtype in (cutlass.BFloat16, cutlass.Float16):
        kernel = Publication(columns, dtype, parallel_rows)
    else:
        from .dtype_selection import Float32BlockSelector

        kernel = Float32BlockSelector(columns)
    return cute.compile(
        kernel,
        *args,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
