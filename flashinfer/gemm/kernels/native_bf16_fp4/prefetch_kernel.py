# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Overlap packed-weight loads with BF16 tensor-core computation."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def _dq2(p0, p1, sw, sh, *, loc=None, ip=None):
    """Two packed E2M1 byte pairs + the E4M3 scale in bits [sh, sh+8) of sw."""
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [
            Uint32(p0).ir_value(loc=loc, ip=ip),
            Uint32(p1).ir_value(loc=loc, ip=ip),
            Uint32(sw).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b8 q0, q1; .reg .b16 sq; .reg .b32 v0, v1, s, e; cvt.u8.u32 q0, $2; cvt.u8.u32 q1, $3; prmt.b32.rc8 e, $4, $4, "
        + str(sh // 8)
        + "; cvt.u16.u32 sq, e; cvt.rn.bf16x2.e4m3x2 s, sq; cvt.rn.bf16x2.e2m1x2 v0, q0; cvt.rn.bf16x2.e2m1x2 v1, q1; mul.rn.bf16x2 $0, v0, s; mul.rn.bf16x2 $1, v1, s; }",
        "=r,=r,r,r,r",
        has_side_effects=False,
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
def _ld_global_stream_u32(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None):
    """One-shot weight/scale load: bypass cache residency with PTX .cs."""
    elem_ptr = tensor.iterator + Int32(offset)
    addr = Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip)],
            "ld.global.cs.u32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_v4_cs(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None):
    """128-bit streaming load: four consecutive u32 in one request."""
    elem_ptr = tensor.iterator + Int32(offset)
    addr = Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.cs.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
            for i in range(4)
        )
    )


@dsl_user_op
def _ld_global_v4(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None):
    """128-bit cached load (L2/L1 resident policy)."""
    elem_ptr = tensor.iterator + Int32(offset)
    addr = Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
            for i in range(4)
        )
    )


@dsl_user_op
def _ld_global_v4_nc(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None):
    """128-bit load through the read-only data path."""
    elem_ptr = tensor.iterator + Int32(offset)
    addr = Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
            for i in range(4)
        )
    )


@dsl_user_op
def _st_shared_v4(addr: Int32, v0, v1, v2, v3, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _smem_ptr(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None):
    elem_ptr = tensor.iterator + Int32(offset)
    return elem_ptr.toint(loc=loc, ip=ip)


@dsl_user_op
def _ld_shared_v2(addr: Int32, *, loc=None, ip=None):
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
            for i in range(2)
        )
    )


@dsl_user_op
def _ldmatrix_a(addr: Int32, *, loc=None, ip=None):
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
            for i in range(4)
        )
    )


@dsl_user_op
def _mma(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
            for i in range(4)
        )
    )


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
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


class NativeBf16Fp4PrefetchKernel:
    def __init__(
        self,
        bm,
        bk,
        warps,
        splits,
        n,
        k,
        arows=None,
        buf=2,
        minb=1,
        vec=0,
        cacheb=-1,
        alpha_one=False,
        compact_b=False,
        bn=128,
    ):
        self.minb = minb
        self.buf = buf
        self.bm = bm
        self.bk = bk
        self.warps = warps
        self.splits = splits
        self.n = n
        self.k = k
        self.bn = bn
        self.threads = 32 * warps
        self.m_tiles = bm // 16
        self.n_tiles = bn // (8 * warps)
        self.ac = bk // 2
        self.bc = bk // 8
        self.sa_s = self.ac + 4
        self.compact_b = compact_b
        self.sb_s = self.bc if compact_b else self.bc + 4
        self.nsw = bk // 64 * bn
        self.sf_chunk_words = bk // 64 * 128
        self.ar = bm if arows is None else min(arows, bm)
        if bn == 128:
            self.na = self.ar * self.ac // self.threads
        else:
            self.na = (self.ar * self.ac + self.threads - 1) // self.threads
        self.nb = bn * self.bc // self.threads
        self.ns = max((self.nsw + self.threads - 1) // self.threads, 1)
        self.nq = max(bk // 64, 1)
        self.fr = min(bk // 16, 4)
        self.kt = k // bk
        self.tps = self.kt // splits
        self.rsp = self.kt % splits
        self.ragged = splits > 1 and self.rsp != 0
        self.cache_b = n == 248320 and bm == 32 if cacheb < 0 else cacheb == 1
        self.nc = cacheb == 2
        self.alpha_one = alpha_one
        self.vec = bool(vec) and self.nb % 4 == 0 and (self.bc % 4 == 0)
        self.veca = self.vec and self.ac % 4 == 0 and (bn != 128 or self.na % 4 == 0)
        self.vecs = self.vec and self.ns % 4 == 0 and (bn == 128)
        self.nbv = self.nb // 4
        if bn == 128:
            self.nav = self.na // 4
        else:
            self.nav = (self.ar * self.ac + self.threads * 4 - 1) // (self.threads * 4)
        self.nsv = self.ns // 4
        self.vecst = self.vec and vec >= 2
        self.pack_output = True

    @cute.jit
    def __call__(
        self, a, b, sf, alpha: cute.Tensor, out, partial, stream: cuda.CUstream
    ):
        a32 = cute.recast_tensor(a, Uint32)
        b32 = cute.recast_tensor(b, Uint32)
        sf32 = cute.recast_tensor(sf, Uint32)
        m, _ = out.shape
        self.kernel(a32, b32, sf32, alpha, out, partial).launch(
            grid=(
                cute.ceil_div(self.n, self.bn),
                cute.ceil_div(m, self.bm),
                self.splits,
            ),
            block=(self.threads, 1, 1),
            min_blocks_per_mp=self.minb,
            stream=stream,
        )
        if cutlass.const_expr(self.splits > 1):
            self.reduce(partial, alpha, out).launch(
                grid=(cute.ceil_div(self.n, self.bn), m, 1),
                block=(self.bn, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def reduce(self, partial, alpha: cute.Tensor, out):
        alpha = alpha[0]
        tid, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        m = out.shape[0]
        ni = bx * self.bn + tid
        acc = Float32(0.0)
        for z in cutlass.range_constexpr(self.splits):
            acc = acc + partial[z * m + by, ni]
        if cutlass.const_expr(not self.alpha_one):
            acc = acc * alpha
        out[by, ni] = out.element_type(acc)

    @cute.kernel
    def kernel(self, a, b, sf, alpha: cute.Tensor, out, partial):
        alpha = alpha[0]
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        bx, by, bz = cute.arch.block_idx()
        g = lane // 4
        p = lane % 4
        m = out.shape[0]
        out32 = cute.recast_tensor(out, Uint32)
        if cutlass.const_expr(self.ragged):
            extra = Int32(self.rsp)
            lead = bz
            if lead > extra:
                lead = extra
            t0 = bz * cutlass.const_expr(self.tps) + lead
            tps = Int32(self.tps)
            if bz < extra:
                tps = Int32(self.tps + 1)
        else:
            tps = cutlass.const_expr(self.tps)
            t0 = bz * tps
        smem = cutlass.utils.SmemAllocator()
        sa = smem.allocate_tensor(
            Uint32,
            cute.make_layout(
                (self.buf, self.ar, self.sa_s),
                stride=(self.ar * self.sa_s, self.sa_s, 1),
            ),
            byte_alignment=16,
        )
        sb = smem.allocate_tensor(
            Uint32,
            cute.make_layout(
                (self.buf, self.bn, self.sb_s),
                stride=(self.bn * self.sb_s, self.sb_s, 1),
            ),
            byte_alignment=16,
        )
        ssf = smem.allocate_tensor(
            Uint32,
            cute.make_layout((self.buf, self.nsw), stride=(self.nsw, 1)),
            byte_alignment=16,
        )
        acc = cute.make_rmem_tensor((self.m_tiles, self.n_tiles, 4), Float32)
        acc.fill(0.0)
        sf_base = bx * cutlass.const_expr(self.k // 64 * 128)
        nai = cutlass.const_expr(self.nav if self.veca else self.na)
        va = cutlass.const_expr(4 if self.veca else 1)
        nbi = cutlass.const_expr(self.nbv if self.vec else self.nb)
        vb = cutlass.const_expr(4 if self.vec else 1)
        nsi = cutlass.const_expr(self.nsv if self.vecs else self.ns)
        vs = cutlass.const_expr(4 if self.vecs else 1)
        rows = [0] * nai
        cols = [0] * nai
        asr = [0] * nai
        avalid = [None] * nai
        for i in cutlass.range_constexpr(nai):
            idx = (i * self.threads + tid) * va
            avalid[i] = idx < self.ar * self.ac
            if cutlass.const_expr(self.bn != 128):
                if idx >= self.ar * self.ac:
                    idx = self.ar * self.ac - 4
            lr = idx // self.ac
            mg = by * self.bm + lr
            if mg >= m:
                mg = m - 1
            rows[i] = mg
            cols[i] = idx % self.ac
            asr[i] = lr
        brow = [0] * nbi
        bcol = [0] * nbi
        bsr = [0] * nbi
        for i in cutlass.range_constexpr(nbi):
            idx = (i * self.threads + tid) * vb
            bsr[i] = idx // self.bc
            brow[i] = bx * self.bn + bsr[i]
            if cutlass.const_expr(self.bn != 128):
                if brow[i] >= self.n:
                    brow[i] = self.n - 1
            bcol[i] = idx % self.bc
        sidx = [0] * nsi
        sbase = [sf_base] * nsi
        soff = [0] * nsi
        for i in cutlass.range_constexpr(nsi):
            sidx[i] = (i * self.threads + tid) * vs % self.nsw
            soff[i] = sidx[i]
            if cutlass.const_expr(self.bn != 128):
                sni = sidx[i] % self.bn
                gni = bx * self.bn + sni
                if gni >= self.n:
                    gni = self.n - 1
                sf_ni = gni % 128
                sbase[i] = gni // 128 * cutlass.const_expr(self.k // 64 * 128)
                soff[i] = sidx[i] // self.bn * 128 + sf_ni % 32 * 4 + sf_ni // 32
        ra = [None] * nai
        rb = [None] * nbi
        rs = [None] * nsi
        for i in cutlass.range_constexpr(nai):
            if cutlass.const_expr(self.veca):
                ra[i] = _ld_global_v4(
                    a, rows[i] * (self.k // 2) + t0 * self.ac + cols[i]
                )
            else:
                ra[i] = a[rows[i], t0 * self.ac + cols[i]]
        for i in cutlass.range_constexpr(nbi):
            if cutlass.const_expr(self.vec and self.cache_b):
                rb[i] = _ld_global_v4(
                    b, brow[i] * (self.k // 8) + t0 * self.bc + bcol[i]
                )
            elif cutlass.const_expr(self.vec and self.nc):
                rb[i] = _ld_global_v4_nc(
                    b, brow[i] * (self.k // 8) + t0 * self.bc + bcol[i]
                )
            elif cutlass.const_expr(self.vec):
                rb[i] = _ld_global_v4_cs(
                    b, brow[i] * (self.k // 8) + t0 * self.bc + bcol[i]
                )
            elif cutlass.const_expr(self.cache_b):
                rb[i] = b[brow[i], t0 * self.bc + bcol[i]]
            else:
                rb[i] = _ld_global_stream_u32(
                    b, brow[i] * (self.k // 8) + t0 * self.bc + bcol[i]
                )
        for i in cutlass.range_constexpr(nsi):
            if cutlass.const_expr(self.vecs and self.cache_b):
                rs[i] = _ld_global_v4(sf, sbase[i] + t0 * self.sf_chunk_words + soff[i])
            elif cutlass.const_expr(self.vecs and self.nc):
                rs[i] = _ld_global_v4_nc(
                    sf, sbase[i] + t0 * self.sf_chunk_words + soff[i]
                )
            elif cutlass.const_expr(self.vecs):
                rs[i] = _ld_global_v4_cs(
                    sf, sbase[i] + t0 * self.sf_chunk_words + soff[i]
                )
            elif cutlass.const_expr(self.cache_b):
                rs[i] = sf[sbase[i] + t0 * self.sf_chunk_words + soff[i]]
            else:
                rs[i] = _ld_global_stream_u32(
                    sf, sbase[i] + t0 * self.sf_chunk_words + soff[i]
                )
        for i in cutlass.range_constexpr(nai):
            if cutlass.const_expr(self.bn != 128):
                if avalid[i]:
                    _st_shared_v4(
                        _smem_ptr(sa, sa.layout((0, asr[i], cols[i]))),
                        ra[i][0],
                        ra[i][1],
                        ra[i][2],
                        ra[i][3],
                    )
            elif cutlass.const_expr(self.veca and self.vecst):
                _st_shared_v4(
                    _smem_ptr(sa, sa.layout((0, asr[i], cols[i]))),
                    ra[i][0],
                    ra[i][1],
                    ra[i][2],
                    ra[i][3],
                )
            elif cutlass.const_expr(self.veca):
                for j in cutlass.range_constexpr(4):
                    sa[0, asr[i], cols[i] + j] = ra[i][j]
            else:
                sa[0, asr[i], cols[i]] = ra[i]
        for i in cutlass.range_constexpr(nbi):
            sbc = bcol[i]
            if cutlass.const_expr(self.compact_b):
                sbc = ((bcol[i] >> 2 ^ bsr[i] & 7) << 2) + (bcol[i] & 3)
            if cutlass.const_expr(self.vecst):
                _st_shared_v4(
                    _smem_ptr(sb, sb.layout((0, bsr[i], sbc))),
                    rb[i][0],
                    rb[i][1],
                    rb[i][2],
                    rb[i][3],
                )
            elif cutlass.const_expr(self.vec):
                for j in cutlass.range_constexpr(4):
                    sb[0, bsr[i], sbc + j] = rb[i][j]
            else:
                sb[0, bsr[i], sbc] = rb[i]
        for i in cutlass.range_constexpr(nsi):
            if cutlass.const_expr(self.vecs and self.vecst):
                _st_shared_v4(
                    _smem_ptr(ssf, ssf.layout((0, sidx[i]))),
                    rs[i][0],
                    rs[i][1],
                    rs[i][2],
                    rs[i][3],
                )
            elif cutlass.const_expr(self.vecs):
                for j in cutlass.range_constexpr(4):
                    ssf[0, sidx[i] + j] = rs[i][j]
            else:
                ssf[0, sidx[i]] = rs[i]
        cute.arch.sync_threads()
        for t in cutlass.range(tps, unroll=1):
            src = t0 + t + 1
            if t + 1 < tps:
                for i in cutlass.range_constexpr(nai):
                    if cutlass.const_expr(self.veca):
                        ra[i] = _ld_global_v4(
                            a, rows[i] * (self.k // 2) + src * self.ac + cols[i]
                        )
                    else:
                        ra[i] = a[rows[i], src * self.ac + cols[i]]
                for i in cutlass.range_constexpr(nbi):
                    if cutlass.const_expr(self.vec and self.cache_b):
                        rb[i] = _ld_global_v4(
                            b, brow[i] * (self.k // 8) + src * self.bc + bcol[i]
                        )
                    elif cutlass.const_expr(self.vec and self.nc):
                        rb[i] = _ld_global_v4_nc(
                            b, brow[i] * (self.k // 8) + src * self.bc + bcol[i]
                        )
                    elif cutlass.const_expr(self.vec):
                        rb[i] = _ld_global_v4_cs(
                            b, brow[i] * (self.k // 8) + src * self.bc + bcol[i]
                        )
                    elif cutlass.const_expr(self.cache_b):
                        rb[i] = b[brow[i], src * self.bc + bcol[i]]
                    else:
                        rb[i] = _ld_global_stream_u32(
                            b, brow[i] * (self.k // 8) + src * self.bc + bcol[i]
                        )
                for i in cutlass.range_constexpr(nsi):
                    if cutlass.const_expr(self.vecs and self.cache_b):
                        rs[i] = _ld_global_v4(
                            sf, sbase[i] + src * self.sf_chunk_words + soff[i]
                        )
                    elif cutlass.const_expr(self.vecs and self.nc):
                        rs[i] = _ld_global_v4_nc(
                            sf, sbase[i] + src * self.sf_chunk_words + soff[i]
                        )
                    elif cutlass.const_expr(self.vecs):
                        rs[i] = _ld_global_v4_cs(
                            sf, sbase[i] + src * self.sf_chunk_words + soff[i]
                        )
                    elif cutlass.const_expr(self.cache_b):
                        rs[i] = sf[sbase[i] + src * self.sf_chunk_words + soff[i]]
                    else:
                        rs[i] = _ld_global_stream_u32(
                            sf, sbase[i] + src * self.sf_chunk_words + soff[i]
                        )
            st = Int32(0)
            if cutlass.const_expr(self.buf == 2):
                st = t % 2
            for fq in cutlass.range_constexpr(self.nq):
                sw = [None] * self.n_tiles
                for nt in cutlass.range_constexpr(self.n_tiles):
                    ni = warp * (8 * self.n_tiles) + nt * 8 + g
                    sfi = ni % 32 * 4 + ni // 32 + fq * 128
                    if cutlass.const_expr(self.bn != 128):
                        sfi = fq * self.bn + ni
                    sw[nt] = ssf[st, sfi]
                for fi in cutlass.range_constexpr(self.fr):
                    f = fq * 4 + fi
                    bf0 = [None] * self.n_tiles
                    bf1 = [None] * self.n_tiles
                    for nt in cutlass.range_constexpr(self.n_tiles):
                        ni = warp * (8 * self.n_tiles) + nt * 8 + g
                        bcol_s = Int32(f * 2)
                        if cutlass.const_expr(self.compact_b):
                            bcol_s = ((bcol_s >> 2 ^ ni & 7) << 2) + (bcol_s & 3)
                        baddr = _smem_ptr(sb, sb.layout((st, ni, bcol_s)))
                        w0, w1 = _ld_shared_v2(baddr)
                        bf0[nt], bf1[nt] = _dq2(
                            w0 >> p * 8, w1 >> p * 8, sw[nt], fi * 8
                        )
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        arow = Int32(mt * 16 + lane % 16)
                        if cutlass.const_expr(self.ar < self.bm):
                            lim = Int32(self.ar - 1)
                            if arow > lim:
                                arow = lim
                        acol = f * 8 + lane // 16 * 4
                        aaddr = _smem_ptr(sa, sa.layout((st, arow, acol)))
                        a0, a1, a2, a3 = _ldmatrix_a(aaddr)
                        for nt in cutlass.range_constexpr(self.n_tiles):
                            c0, c1, c2, c3 = _mma(
                                a0,
                                a1,
                                a2,
                                a3,
                                bf0[nt],
                                bf1[nt],
                                acc[mt, nt, 0],
                                acc[mt, nt, 1],
                                acc[mt, nt, 2],
                                acc[mt, nt, 3],
                            )
                            acc[mt, nt, 0] = c0
                            acc[mt, nt, 1] = c1
                            acc[mt, nt, 2] = c2
                            acc[mt, nt, 3] = c3
            ns_ = Int32(0)
            if cutlass.const_expr(self.buf == 2):
                ns_ = 1 - st
            elif t + 1 < tps:
                cute.arch.sync_threads()
            if t + 1 < tps:
                for i in cutlass.range_constexpr(nai):
                    if cutlass.const_expr(self.bn != 128):
                        if avalid[i]:
                            _st_shared_v4(
                                _smem_ptr(sa, sa.layout((ns_, asr[i], cols[i]))),
                                ra[i][0],
                                ra[i][1],
                                ra[i][2],
                                ra[i][3],
                            )
                    elif cutlass.const_expr(self.veca and self.vecst):
                        _st_shared_v4(
                            _smem_ptr(sa, sa.layout((ns_, asr[i], cols[i]))),
                            ra[i][0],
                            ra[i][1],
                            ra[i][2],
                            ra[i][3],
                        )
                    elif cutlass.const_expr(self.veca):
                        for j in cutlass.range_constexpr(4):
                            sa[ns_, asr[i], cols[i] + j] = ra[i][j]
                    else:
                        sa[ns_, asr[i], cols[i]] = ra[i]
                for i in cutlass.range_constexpr(nbi):
                    sbc = bcol[i]
                    if cutlass.const_expr(self.compact_b):
                        sbc = ((bcol[i] >> 2 ^ bsr[i] & 7) << 2) + (bcol[i] & 3)
                    if cutlass.const_expr(self.vecst):
                        _st_shared_v4(
                            _smem_ptr(sb, sb.layout((ns_, bsr[i], sbc))),
                            rb[i][0],
                            rb[i][1],
                            rb[i][2],
                            rb[i][3],
                        )
                    elif cutlass.const_expr(self.vec):
                        for j in cutlass.range_constexpr(4):
                            sb[ns_, bsr[i], sbc + j] = rb[i][j]
                    else:
                        sb[ns_, bsr[i], sbc] = rb[i]
                for i in cutlass.range_constexpr(nsi):
                    if cutlass.const_expr(self.vecs and self.vecst):
                        _st_shared_v4(
                            _smem_ptr(ssf, ssf.layout((ns_, sidx[i]))),
                            rs[i][0],
                            rs[i][1],
                            rs[i][2],
                            rs[i][3],
                        )
                    elif cutlass.const_expr(self.vecs):
                        for j in cutlass.range_constexpr(4):
                            ssf[ns_, sidx[i] + j] = rs[i][j]
                    else:
                        ssf[ns_, sidx[i]] = rs[i]
                cute.arch.sync_threads()
        nbase = bx * self.bn + warp * (8 * self.n_tiles) + p * 2
        if cutlass.const_expr(self.splits > 1):
            base = bz * m + by * self.bm
            for mt in cutlass.range_constexpr(self.m_tiles):
                for row in cutlass.range_constexpr(2):
                    mi = mt * 16 + row * 8 + g
                    if by * self.bm + mi < m:
                        for nt in cutlass.range_constexpr(self.n_tiles):
                            for col in cutlass.range_constexpr(2):
                                oi = nbase + nt * 8 + col
                                if cutlass.const_expr(self.bn != 128):
                                    if oi < self.n:
                                        partial[base + mi, oi] = acc[
                                            mt, nt, row * 2 + col
                                        ]
                                else:
                                    partial[base + mi, oi] = acc[mt, nt, row * 2 + col]
        else:
            for mt in cutlass.range_constexpr(self.m_tiles):
                for row in cutlass.range_constexpr(2):
                    mi = by * self.bm + mt * 16 + row * 8 + g
                    if mi < m:
                        for nt in cutlass.range_constexpr(self.n_tiles):
                            if cutlass.const_expr(self.pack_output):
                                lo = acc[mt, nt, row * 2]
                                hi = acc[mt, nt, row * 2 + 1]
                                if cutlass.const_expr(not self.alpha_one):
                                    lo = lo * alpha
                                    hi = hi * alpha
                                oi = nbase + nt * 8
                                if cutlass.const_expr(self.bn != 128):
                                    if oi + 1 < self.n:
                                        out32[mi, oi // 2] = _pack_bf16x2(lo, hi)
                                else:
                                    out32[mi, oi // 2] = _pack_bf16x2(lo, hi)
                            else:
                                for col in cutlass.range_constexpr(2):
                                    v = acc[mt, nt, row * 2 + col]
                                    if cutlass.const_expr(not self.alpha_one):
                                        v = v * alpha
                                    oi = nbase + nt * 8 + col
                                    if cutlass.const_expr(self.bn != 128):
                                        if oi < self.n:
                                            out[mi, oi] = out.element_type(v)
                                    else:
                                        out[mi, oi] = out.element_type(v)
