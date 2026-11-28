# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
# Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
# See flashinfer/experimental/deepgemm_fp8_gemm/DEEPGEMM_NOTICE.txt for the MIT license.
# Include that notice when redistributing this module.
"""Warp-specialized MXFP4 scoring over per-query blocks of eight keys."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from ... import memory
from .validation import validate_candidate_row


@dsl_user_op
def instruction(code, constraints, operands=(), dtype=None, *, loc=None, ip=None):
    result = llvm.inline_asm(
        None if dtype is None else dtype.mlir_type,
        [x.ir_value(loc=loc, ip=ip) for x in operands],
        code,
        constraints,
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    if dtype is not None:
        return dtype(result)


@cute.jit
def load_shared(addr):
    return instruction(
        "ld.shared.u32 $0, [$1];",
        "=r,r,~{memory}",
        (cutlass.Int32(addr),),
        cutlass.Uint32,
    )


@cute.jit
def store_shared(addr, value):
    instruction(
        "st.shared.u32 [$0], $1;",
        "r,r,~{memory}",
        (cutlass.Int32(addr), cutlass.Uint32(value)),
    )


@cute.jit
def wait(addr, phase):
    instruction(
        "{ .reg .pred p; AGAIN: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [$0], $1, 32; @!p bra AGAIN; }",
        "r,r,~{memory}",
        (cutlass.Int32(addr), cutlass.Int32(phase)),
    )


@cute.jit
def arrive(addr, count: cutlass.Constexpr = 1):
    instruction(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0], $1;",
        "r,r,~{memory}",
        (cutlass.Int32(addr), cutlass.Int32(count)),
    )


@cute.jit
def async_arrive(addr):
    instruction(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r,~{memory}",
        (cutlass.Int32(addr),),
    )


@cute.jit
def copy16(dst, src):
    instruction(
        "cp.async.cg.shared::cta.global [$0], [$1], 16;",
        "r,l,~{memory}",
        (cutlass.Int32(dst), cutlass.Uint64(src)),
    )


@cute.jit
def commit(addr):
    instruction(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];",
        "r,~{memory}",
        (cutlass.Int32(addr),),
    )


@cute.jit
def copy_scale(taddr, addr):
    desc = cutlass.Uint64(cutlass.Uint32(addr) >> 4) | cutlass.Uint64(0x400800000000)
    instruction(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [$0], $1;",
        "r,l,~{memory}",
        (cutlass.Int32(taddr), desc),
    )


@cute.jit
def mma(taddr, a, b, sa, sb):
    ad = cutlass.Uint64((cutlass.Uint32(a) >> 4) & 0x3FFF) | cutlass.Uint64(
        0x8000402000000000
    )
    bd = cutlass.Uint64((cutlass.Uint32(b) >> 4) & 0x3FFF) | cutlass.Uint64(
        0x8000402000000000
    )
    instruction(
        "{ .reg .pred p; setp.ne.u32 p, 0, 0; tcgen05.mma.cta_group::1.kind::mxf4.block_scale [$0], $1, $2, $3, [$4], [$5], p; setp.ne.u32 p, 1, 0; tcgen05.mma.cta_group::1.kind::mxf4.block_scale [$0], $6, $7, $8, [$4], [$5], p; }",
        "r,l,l,r,r,r,l,l,r,~{memory}",
        (
            cutlass.Int32(taddr),
            ad,
            bd,
            cutlass.Uint32(0x8880480),
            cutlass.Int32(sa),
            cutlass.Int32(sb),
            ad + 2,
            bd + 2,
            cutlass.Uint32(0x488804A0),
        ),
    )


# Keep the four native-BF16 FMA chains in registers across the two TMEM loads.
_REDUCE = "{ .reg .f32 v<16>; .reg .b32 a,b,x,y; mov.b64 {a,b}, $2; "
_REDUCE += (
    "tcgen05.ld.sync.aligned.32x32b.x16.b32 {"
    + ",".join(f"v{i}" for i in range(16))
    + "}, [$1]; tcgen05.wait::ld.sync.aligned; "
)
for _i in range(4):
    _REDUCE += f"cvt.rn.relu.bf16x2.f32 x,v{4 * _i + 1},v{4 * _i}; cvt.rn.relu.bf16x2.f32 y,v{4 * _i + 3},v{4 * _i + 2}; fma.rn.bf16x2 a,x,${3 + 2 * _i},a; fma.rn.bf16x2 b,y,${4 + 2 * _i},b; "
_REDUCE += "mov.b64 $0,{a,b}; }"


@cute.jit
def reduce16(addr, accum, weights):
    return instruction(
        _REDUCE,
        "=l,r,l," + ",".join(["r"] * 8) + ",~{memory}",
        (cutlass.Int32(addr), cutlass.Uint64(accum), *weights),
        cutlass.Uint64,
    )


@cute.jit
def reduce16_release(addr, accum, weights, barrier):
    code = _REDUCE.replace(
        "tcgen05.wait::ld.sync.aligned;",
        "tcgen05.wait::ld.sync.aligned; tcgen05.fence::before_thread_sync; mbarrier.arrive.release.cta.shared::cta.b64 _, [$11];",
    )
    return instruction(
        code,
        "=l,r,l," + ",".join(["r"] * 9) + ",~{memory}",
        (cutlass.Int32(addr), cutlass.Uint64(accum), *weights, cutlass.Int32(barrier)),
        cutlass.Uint64,
    )


@cute.jit
def warp_extreme(value, op: cutlass.Constexpr):
    code = (
        "{ .reg .b32 b, s; mov.b32 b,$1; shr.s32 s,b,31; and.b32 s,s,0x7fffffff; xor.b32 b,b,s; redux.sync."
        + op
        + ".s32 b,b,0xffffffff; shr.s32 s,b,31; and.b32 s,s,0x7fffffff; xor.b32 b,b,s; mov.b32 $0,b; }"
    )
    return instruction(code, "=f,f", (value,), cutlass.Float32)


@cute.jit
def fast_div(a, b):
    return instruction(
        "div.approx.ftz.f32 $0,$1,$2;",
        "=f,f,f",
        (cutlass.Float32(a), cutlass.Float32(b)),
        cutlass.Float32,
    )


@cute.jit
def histogram_edge(high, bucket, scale):
    return instruction(
        "{ .reg .f32 r,n; rcp.approx.ftz.f32 r,$3; neg.f32 n,$2; fma.rn.ftz.f32 $0,n,r,$1; }",
        "=f,f,f,f",
        (high, bucket, scale),
        cutlass.Float32,
    )


@cute.jit
def finish_sum(accum):
    return instruction(
        "{ .reg .b32 a,b,c; .reg .b16 lo,hi,s; mov.b64 {a,b},$1; add.rn.bf16x2 c,a,b; mov.b32 {lo,hi},c; add.rn.bf16 s,lo,hi; cvt.f32.bf16 $0,s; }",
        "=f,l",
        (cutlass.Uint64(accum),),
        cutlass.Float32,
    )


@cute.jit
def publish(ptr, score, col, length, cursor):
    return instruction(
        "{ .reg .pred hit,empty,store; .reg .b32 votes,lower,rank,n,pos,bits; .reg .b64 addr; setp.ge.f32 hit,$2,$3; setp.lt.and.u32 hit,$4,$5,hit; vote.ballot.sync.b32 votes,hit,0xffffffff; mov.u32 $0,$6; setp.eq.u32 empty,votes,0; @empty bra DONE; mov.u32 lower,%lanemask_lt; and.b32 lower,lower,votes; popc.b32 rank,lower; popc.b32 n,votes; add.u32 pos,$6,rank; add.u32 $0,$6,n; setp.lt.and.u32 store,pos,256,hit; mov.b32 bits,$2; or.b32 bits,bits,$4; mad.wide.u32 addr,pos,4,$1; @store st.global.cs.b32 [addr],bits; DONE: }",
        "=&r,l,f,f,r,r,r,~{memory}",
        (
            cutlass.Uint64(ptr),
            score[0],
            score[1],
            cutlass.Uint32(col),
            cutlass.Uint32(length),
            cutlass.Uint32(cursor),
        ),
        cutlass.Uint32,
    )


def typed_reduce_code(mode, weight_bytes):
    code = "{ .reg .f32 v<32>,a<4>,w,t; .reg .b16 x,y,z; .reg .b32 wa,ta; "
    for i in range(4):
        code += f"mov.f32 a{i},0f00000000; "
    for offset in (0, 16):
        code += f"add.u32 ta,$1,{offset}; "
        code += (
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 {"
            + ",".join(f"v{i}" for i in range(offset, offset + 16))
            + "}, [ta]; tcgen05.wait::ld.sync.aligned; "
        )
    code += "tcgen05.fence::before_thread_sync; mbarrier.arrive.release.cta.shared::cta.b64 _,[$3]; "
    native = {"native_bf16": "bf16", "native_fp16": "f16"}.get(mode)
    for h in range(32):
        code += f"add.u32 wa,$2,{h * weight_bytes}; "
        if weight_bytes == 4:
            code += "ld.shared.f32 w,[wa]; "
        else:
            code += "ld.shared.b16 y,[wa]; cvt.f32.bf16 w,y; "
        code += f"max.f32 t,v{h},0f00000000; "
        a = h % 4
        if native:
            code += (
                f"cvt.rn.{native}.f32 x,t; cvt.rn.{native}.f32 y,w; "
                f"cvt.rn.{native}.f32 z,a{a}; fma.rn.{native} z,x,y,z; cvt.f32.{native} a{a},z; "
            )
        else:
            code += f"fma.rn.f32 a{a},t,w,a{a}; "
    if native:
        code += f"add.f32 t,a0,a2; cvt.rn.{native}.f32 x,t; cvt.f32.{native} a0,x; "
        code += f"add.f32 t,a1,a3; cvt.rn.{native}.f32 y,t; cvt.f32.{native} a1,y; "
        code += f"add.f32 t,a0,a1; cvt.rn.{native}.f32 z,t; cvt.f32.{native} $0,z; "
    else:
        code += "add.f32 a0,a0,a1; add.f32 a0,a0,a2; add.f32 $0,a0,a3; "
    return code + "}"


@cute.jit
def publish_typed(ptr, count_ptr, score, threshold, col, length):
    instruction(
        "{ .reg .pred hit,empty,leader,store; .reg .b32 votes,lower,rank,n,base,pos,bits; .reg .b64 addr,record,idx; "
        "setp.ge.f32 hit,$2,$3; setp.lt.and.u32 hit,$4,$5,hit; vote.ballot.sync.b32 votes,hit,0xffffffff; "
        "setp.eq.u32 empty,votes,0; @empty bra DONE; mov.u32 lower,%lanemask_lt; and.b32 lower,lower,votes; "
        "popc.b32 rank,lower; popc.b32 n,votes; mov.u32 base,0; mov.u32 pos,%laneid; setp.eq.u32 leader,pos,0; "
        "@leader atom.global.add.u32 base,[$1],n; shfl.sync.idx.b32 base,base,0,31,0xffffffff; add.u32 pos,base,rank; "
        "setp.lt.and.u32 store,pos,4096,hit; mov.b32 bits,$2; cvt.u64.u32 record,bits; shl.b64 record,record,32; "
        "cvt.u64.u32 idx,$4; or.b64 record,record,idx; mad.wide.u32 addr,pos,8,$0; @store st.global.b64 [addr],record; DONE: }",
        "l,l,f,f,r,r,~{memory}",
        (ptr, count_ptr, score, threshold, cutlass.Uint32(col), cutlass.Uint32(length)),
    )


class SparseScore:
    """Persistent sampling, candidate filtering and failed-row scoring."""

    def __init__(self, sms, pdl=True, stage=2, math_mode=None, weight_bytes=2):
        self.typed = math_mode is not None
        self.weight_bytes = weight_bytes
        self.reduce_code = (
            typed_reduce_code(math_mode, weight_bytes) if self.typed else ""
        )
        self.unroll = 8
        self.sms = sms * (2 if stage == 2 else 1)
        self.pdl = pdl
        self.stage = stage
        self.groups = {1: 2, 2: 4, 4: 5}[stage]
        self.stages = {1: 3, 2: 3, 4: 5}[stage]
        self.math_warps = 4 * self.groups
        self.threads = ((self.math_warps + 3 + self.groups + 3) // 4) * 128
        self.acc_stride = 32
        self.sfq_tmem = self.groups * self.acc_stride
        self.sfk_tmem = self.sfq_tmem + 8
        self.tmem_cols = (
            1 << (self.sfk_tmem + self.stages * self.groups * 4 - 1).bit_length()
        )
        self.math_regs = {1: 56, 2: 48, 4: 80}[stage]
        self.other_regs = 48 if stage == 4 else 24
        if self.typed:
            self.math_regs = 80
        self.k_stride = self.groups * 8192
        self.s_stride = self.groups * 512
        self.sfq_offset = 8192 + self.stages * self.k_stride
        self.sfk_offset = self.sfq_offset + 1024
        self.w_offset = self.sfk_offset + self.stages * self.s_stride
        self.qctrl_offset = self.w_offset + 256
        self.info_offset = self.qctrl_offset + 32
        self.hdr_offset = self.info_offset + self.stages * self.groups * 64
        self.bar_offset = (self.hdr_offset + self.stages * 16 + 7) // 8 * 8
        self.bar_count = 6 + self.stages * 4 + self.groups * 2
        self.alloc_offset = self.bar_offset + self.bar_count * 8
        self.temp_offset = (self.alloc_offset + 4 + 127) // 128 * 128
        self.smem_bytes = self.temp_offset + (2176 if stage == 1 else 0)

    @cute.jit
    def __call__(
        self,
        q,
        qs,
        k,
        ks,
        weights,
        blocks,
        lengths,
        thresholds,
        candidates,
        info,
        queue,
        dense,
        row_ends,
        error,
        stream: cuda.CUstream,
        bounds=None,
        enabled: cutlass.Int32 = 1,
    ):
        self.kernel(
            q,
            qs,
            k,
            ks,
            weights,
            blocks,
            lengths,
            thresholds,
            candidates,
            info,
            queue,
            dense,
            row_ends,
            error,
            bounds,
            enabled,
        ).launch(
            grid=(self.sms, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=1 if self.typed else {1: 3, 2: 2, 4: 1}[self.stage],
        )

    @cute.kernel
    def kernel(
        self,
        q,
        qs,
        k,
        ks,
        weights,
        blocks,
        lengths,
        thresholds,
        candidates,
        info,
        queue,
        dense,
        row_ends,
        error,
        bounds,
        enabled,
    ):
        tid = cute.arch.thread_idx()[0]
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        bid = cute.arch.block_idx()[0]
        mem = memory.SmemAllocator()
        storage = mem.allocate_tensor(
            cutlass.Uint32,
            cute.make_layout(cute.ceil_div(self.smem_bytes, 4)),
            byte_alignment=1024,
        )
        sm = cutlass.Int32(storage.iterator.toint())
        qaddr = sm
        kaddr = sm + 8192
        qsaddr = sm + self.sfq_offset
        ksaddr = sm + self.sfk_offset
        waddr = sm + self.w_offset
        infos = sm + self.info_offset
        qctrl = sm + self.qctrl_offset
        hdr = sm + self.hdr_offset
        qfull = sm + self.bar_offset
        sfqfull = qfull + 16
        qempty = qfull + 32
        metafull = qfull + 48
        sfcopied = metafull + self.stages * 8
        kvfull = sfcopied + self.stages * 8
        kvempty = kvfull + self.stages * 8
        tfull = kvempty + self.stages * 8
        tempty = tfull + self.groups * 8
        alloc = sm + self.alloc_offset
        if warp == self.math_warps:
            with cute.arch.elect_one():
                for index in cutlass.range_constexpr(9):
                    off, count, n = (
                        (0, 32, 2),
                        (16, 1, 2),
                        (32, self.math_warps * 32 + 64, 2),
                        (48, 32, self.stages),
                        (48 + self.stages * 8, self.groups * 32, self.stages),
                        (48 + self.stages * 16, self.groups * 32 + 1, self.stages),
                        (48 + self.stages * 24, 1, self.stages),
                        (48 + self.stages * 32, 1, self.groups),
                        (48 + self.stages * 32 + self.groups * 8, 128, self.groups),
                    )[index]
                    for j in cutlass.range_constexpr(n):
                        instruction(
                            "mbarrier.init.shared::cta.b64 [$0], $1;",
                            "r,r,~{memory}",
                            (qfull + off + j * 8, cutlass.Int32(count)),
                        )
                instruction("fence.mbarrier_init.release.cluster;", "~{memory}")
        if warp == self.math_warps + 1:
            instruction(
                f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {self.tmem_cols}; tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
                "r,~{memory}",
                (alloc,),
            )
        cute.arch.sync_threads()
        instruction("tcgen05.fence::after_thread_sync;", "~{memory}")
        ta = load_shared(alloc)
        if cutlass.const_expr(self.pdl):
            instruction(
                "griddepcontrol.wait; griddepcontrol.launch_dependents;", "~{memory}"
            )
        if cutlass.const_expr(self.stage == 1):
            # Each CTA validates its own rows before any warp gathers their KV.
            for row in cutlass.range(
                bid + warp * self.sms, q.shape[0], (self.threads // 32) * self.sms
            ):
                raw = row_ends[row]
                if cutlass.const_expr(bounds is not None):
                    live = bounds[1][0]
                    end = bounds[0][min(row, max(live - 1, 0))]
                    start = cutlass.Int32(0) if row < live else end
                    raw = end if (end > start and enabled != 0) else cutlass.Int32(0)
                    if lane == 0:
                        bounds[2][row] = start
                        bounds[3][row] = end
                        bounds[4][row] = end if end - start > 512 else start
                        bounds[5][row] = raw
                invalid = validate_candidate_row(blocks, row, raw, k.shape[0])
                kept = min(2048, (max(0, raw) + 7) // 8)
                length = (
                    (kept - 1) * 8 + (raw - 1) % 8 + 1
                    if raw > 0 and invalid == 0
                    else cutlass.Int32(0)
                )
                if lane == 0:
                    lengths[row] = length
                    if invalid != 0:
                        cute.arch.atomic_or(
                            error.iterator.llvm_ptr,
                            cutlass.Int32(32),
                            sem="relaxed",
                            scope="gpu",
                        )
            cute.arch.sync_threads()
        if cutlass.const_expr(self.stage == 1 and not self.typed):
            if bid == 0 and tid == 0:
                queue[0] = 0
        if warp >= self.math_warps:
            instruction(f"setmaxnreg.dec.sync.aligned.u32 {self.other_regs};", "")
        if warp < self.math_warps:
            instruction(f"setmaxnreg.inc.sync.aligned.u32 {self.math_regs};", "")
            qi = cutlass.Int32(0)
            qp = cutlass.Int32(0)
            tp = cutlass.Int32(0)
            weights_r = cute.make_rmem_tensor(16, cutlass.Uint32)
            running = True
            while running:
                wait(qfull + qi * 8, qp)
                row = load_shared(qctrl + qi * 16)
                ns = load_shared(qctrl + qi * 16 + 8)
                running = load_shared(qctrl + qi * 16 + 4) != 0
                if running:
                    if cutlass.const_expr(not self.typed):
                        for j in cutlass.range_constexpr(16):
                            weights_r[j] = load_shared(waddr + qi * 128 + j * 4)
                    length = lengths[row]
                    begin = load_shared(qctrl + qi * 16 + 12)
                    tau = thresholds[row]
                    accepted = cutlass.Uint32(0)
                    dst = (
                        cutlass.Uint64(candidates.iterator.toint())
                        + (cutlass.Uint64(row) * candidates.shape[1] + 32 + warp * 256)
                        * 4
                    )
                    for split in cutlass.range(ns, unroll=self.unroll):
                        wait(tfull + (warp // 4) * 8, tp)
                        instruction("tcgen05.fence::after_thread_sync;", "~{memory}")
                        addr = (
                            ta + (warp // 4) * self.acc_stride + ((warp % 4) * 32 << 16)
                        )
                        if cutlass.const_expr(self.typed):
                            score = instruction(
                                self.reduce_code,
                                "=f,r,r,r,~{memory}",
                                (
                                    cutlass.Int32(addr),
                                    waddr + qi * 128,
                                    tempty + (warp // 4) * 8,
                                ),
                                cutlass.Float32,
                            )
                            score = cutlass.Float32(dense.element_type(score))
                            if score != score:
                                score = cutlass.Float32(float("-inf"))
                        else:
                            acc = reduce16(
                                addr,
                                cutlass.Uint64(0),
                                tuple(weights_r[j] for j in range(8)),
                            )
                            acc = reduce16_release(
                                addr + 16,
                                acc,
                                tuple(weights_r[j + 8] for j in range(8)),
                                tempty + (warp // 4) * 8,
                            )
                            score = finish_sum(acc)
                            if score != score:
                                score = cutlass.Float32(float("-inf"))
                        col = (begin + split) * self.groups * 128 + warp * 32 + lane
                        if cutlass.const_expr(self.stage == 2):
                            if cutlass.const_expr(self.typed):
                                publish_typed(
                                    cutlass.Uint64(candidates.iterator.toint())
                                    + cutlass.Uint64(row) * 4096 * 8,
                                    cutlass.Uint64(queue.iterator.toint())
                                    + cutlass.Uint64(row) * 4,
                                    score,
                                    tau,
                                    col,
                                    length,
                                )
                            else:
                                accepted = publish(
                                    dst, (score, tau), col, length, accepted
                                )
                        elif cutlass.const_expr(self.stage == 4):
                            if col < dense.shape[1]:
                                dense[row, col] = dense.element_type(score)
                        else:
                            sample_length = max(
                                0,
                                (min(64, cute.ceil_div(length, 8)) - 1) * 8
                                + (length - 1) % 8
                                + 1,
                            )
                            value = (
                                score
                                if col < sample_length
                                else cutlass.Float32(float("-inf"))
                            )
                            if cutlass.const_expr(self.typed):
                                info[row, col] = info.element_type(value)
                            else:
                                instruction(
                                    "{ .reg .b16 v; cvt.rn.bf16.f32 v,$1; st.shared.b16 [$0],v; }",
                                    "r,f,~{memory}",
                                    (sm + self.temp_offset + col * 2, value),
                                )
                        tp = tp ^ 1
                    if cutlass.const_expr(self.stage == 2 and not self.typed):
                        if lane == 0:
                            candidates[row, warp] = cutlass.Int32(accepted)
                        if warp == 0 and lane >= self.math_warps:
                            candidates[row, lane] = 0
                    elif cutlass.const_expr(self.stage == 1 and not self.typed):
                        self.propose(
                            sm + self.temp_offset,
                            tid,
                            lane,
                            warp,
                            row,
                            length,
                            thresholds,
                            candidates,
                            info,
                        )
                    arrive(qempty + qi * 8)
                    qi = qi + 1
                    if qi == 2:
                        qi = 0
                        qp = qp ^ 1
            cute.arch.barrier(barrier_id=0, number_of_threads=self.math_warps * 32)
            if warp == 0:
                instruction(
                    f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {self.tmem_cols};",
                    "r,~{memory}",
                    (cutlass.Int32(ta),),
                )
        elif warp == self.math_warps:
            qi = cutlass.Int32(0)
            qp = cutlass.Int32(1)
            ki = cutlass.Int32(0)
            kp = cutlass.Int32(1)
            task_count = cutlass.Int32(q.shape[0])
            parts = cutlass.Int32(1)
            if cutlass.const_expr(self.stage == 4 and not self.typed):
                parts = min(16, max(1, self.sms // max(1, queue[0])))
                task_count = queue[0] * parts
            for task in cutlass.range(bid, task_count, self.sms):
                row = task
                if cutlass.const_expr(self.stage == 4 and not self.typed):
                    row = queue[task // parts + 1]
                wait(qempty + qi * 8, qp)
                length = lengths[row]
                total = cute.ceil_div(length, self.groups * 128)
                if cutlass.const_expr(self.stage == 1):
                    total = cutlass.Int32(2)
                begin = cutlass.Int32(0)
                ns = total
                if cutlass.const_expr(self.stage == 4):
                    begin = (total * (task % parts) + parts - 1) // parts
                    finish = (total * (task % parts + 1) + parts - 1) // parts
                    ns = finish - begin
                if cutlass.const_expr(self.stage == 4 and self.typed):
                    count = queue[row]
                    if count >= min(512, length) and count <= 4096:
                        ns = 0
                if lane == 0:
                    store_shared(qctrl + qi * 16, row)
                    store_shared(qctrl + qi * 16 + 4, 1)
                    store_shared(qctrl + qi * 16 + 8, ns)
                    store_shared(qctrl + qi * 16 + 12, begin)
                if lane < 8:
                    copy16(
                        qsaddr + qi * 512 + lane * 16,
                        cutlass.Uint64(qs.iterator.toint())
                        + cutlass.Uint64(row) * 128
                        + lane * 16,
                    )
                if lane < 2 * self.weight_bytes:
                    copy16(
                        waddr + qi * 128 + lane * 16,
                        cutlass.Uint64(weights.iterator.toint())
                        + cutlass.Uint64(row) * (32 * self.weight_bytes)
                        + lane * 16,
                    )
                for j in cutlass.range_constexpr(4):
                    chunk = lane + j * 32
                    sw = (chunk & ~3) | ((chunk & 3) ^ ((chunk >> 3) & 3))
                    copy16(
                        qaddr + qi * 4096 + sw * 16,
                        cutlass.Uint64(q.iterator.toint())
                        + cutlass.Uint64(row) * 2048
                        + chunk * 16,
                    )
                instruction("fence.proxy.async.shared::cta;", "~{memory}")
                async_arrive(qfull + qi * 8)
                for split in cutlass.range(ns):
                    wait(kvempty + ki * 8, kp)
                    nb = min(
                        self.groups * 16,
                        cute.ceil_div(length, 8) - (split + begin) * self.groups * 16,
                    )
                    if lane == 0:
                        store_shared(hdr + ki * 16 + 4, nb)
                    if cutlass.const_expr(self.stage == 1):
                        index = split * self.groups * 16 + lane
                        kept = min(2048, (length + 7) // 8)
                        h = (
                            cutlass.Uint32(index) + cutlass.Uint32(row) * 2654435761
                        ) * 2246822519
                        slot = cutlass.Int32(0)
                        if kept == 2048:
                            slot = index * 32 + cutlass.Int32((h ^ (h >> 13)) & 31)
                        else:
                            lo = index * kept // 64
                            hi = (index + 1) * kept // 64
                            slot = lo + cutlass.Int32(
                                (h ^ (h >> 13)) % cutlass.Uint32(max(hi - lo, 1))
                            )
                        if index == 63:
                            slot = kept - 1
                        if kept <= 64:
                            slot = index
                        value = cutlass.Int32(0)
                        if slot < kept and slot >= 0:
                            instruction(
                                "cp.async.ca.shared::cta.global [$0],[$1],4;",
                                "r,l,~{memory}",
                                (
                                    infos + ki * self.groups * 64 + lane * 4,
                                    cutlass.Uint64(blocks.iterator.toint())
                                    + (cutlass.Uint64(row) * blocks.shape[1] + slot)
                                    * 4,
                                ),
                            )
                        else:
                            store_shared(infos + ki * self.groups * 64 + lane * 4, 0)
                    else:
                        offset = (split + begin) * self.groups * 16 + lane * 4
                        if lane < self.groups * 4 and offset < blocks.shape[1]:
                            copy16(
                                infos + ki * self.groups * 64 + lane * 16,
                                cutlass.Uint64(blocks.iterator.toint())
                                + (cutlass.Uint64(row) * blocks.shape[1] + offset) * 4,
                            )
                    async_arrive(metafull + ki * 8)
                    ki = ki + 1
                    if ki == self.stages:
                        ki = 0
                        kp = kp ^ 1
                arrive(qempty + qi * 8)
                qi = qi + 1
                if qi == 2:
                    qi = 0
                    qp = qp ^ 1
            wait(qempty + qi * 8, qp)
            wait(kvempty + ki * 8, kp)
            if lane == 0:
                store_shared(qctrl + qi * 16 + 4, 0)
                store_shared(hdr + ki * 16 + 4, 0xFFFFFFFF)
            cute.arch.sync_warp()
            arrive(qfull + qi * 8)
            arrive(metafull + ki * 8)
        elif warp == self.math_warps + 1:
            qi = cutlass.Int32(0)
            qp = cutlass.Int32(0)
            ki = cutlass.Int32(0)
            kp = cutlass.Int32(0)
            running = True
            while running:
                wait(qfull + qi * 8, qp)
                running = load_shared(qctrl + qi * 16 + 4) != 0
                if running:
                    ns = load_shared(qctrl + qi * 16 + 8)
                    word = load_shared(qsaddr + qi * 512 + lane * 4)
                    cute.arch.sync_warp()
                    instruction(
                        "st.shared.v4.b32 [$0],{$1,0,0,0};",
                        "r,r,~{memory}",
                        (qsaddr + qi * 512 + lane * 16, word),
                    )
                    instruction("fence.proxy.async.shared::cta;", "~{memory}")
                    with cute.arch.elect_one():
                        copy_scale(ta + self.sfq_tmem + qi * 4, qsaddr + qi * 512)
                        commit(sfqfull + qi * 8)
                    for _split in cutlass.range(ns, unroll=self.unroll):
                        wait(sfcopied + ki * 8, kp)
                        for j in cutlass.range_constexpr(self.groups):
                            a = ksaddr + ki * self.s_stride + j * 512
                            r0 = load_shared(a + lane * 4)
                            r1 = load_shared(a + 128 + lane * 4)
                            r2 = load_shared(a + 256 + lane * 4)
                            r3 = load_shared(a + 384 + lane * 4)
                            cute.arch.sync_warp()
                            instruction(
                                "st.shared.v4.b32 [$0],{$1,$2,$3,$4};",
                                "r,r,r,r,r,~{memory}",
                                (a + lane * 16, r0, r1, r2, r3),
                            )
                        instruction("fence.proxy.async.shared::cta;", "~{memory}")
                        with cute.arch.elect_one():
                            for j in cutlass.range_constexpr(self.groups):
                                copy_scale(
                                    ta + self.sfk_tmem + ki * self.groups * 4 + j * 4,
                                    ksaddr + ki * self.s_stride + j * 512,
                                )
                            commit(kvfull + ki * 8)
                        ki = ki + 1
                        if ki == self.stages:
                            ki = 0
                            kp = kp ^ 1
                    qi = qi + 1
                    if qi == 2:
                        qi = 0
                        qp = qp ^ 1
        elif warp == self.math_warps + 2:
            with cute.arch.elect_one():
                qi = cutlass.Int32(0)
                qp = cutlass.Int32(0)
                ki = cutlass.Int32(0)
                kp = cutlass.Int32(0)
                tp = cutlass.Int32(1)
                running = True
                while running:
                    wait(qfull + qi * 8, qp)
                    running = load_shared(qctrl + qi * 16 + 4) != 0
                    if running:
                        ns = load_shared(qctrl + qi * 16 + 8)
                        wait(sfqfull + qi * 8, qp)
                        for _split in cutlass.range(ns):
                            wait(kvfull + ki * 8, kp)
                            instruction(
                                "tcgen05.fence::after_thread_sync;", "~{memory}"
                            )
                            for j in cutlass.range_constexpr(self.groups):
                                wait(tempty + j * 8, tp)
                                instruction(
                                    "tcgen05.fence::after_thread_sync;", "~{memory}"
                                )
                                mma(
                                    ta + j * self.acc_stride,
                                    kaddr + ki * self.k_stride + j * 8192,
                                    qaddr + qi * 4096,
                                    ta + self.sfk_tmem + ki * self.groups * 4 + j * 4,
                                    ta + self.sfq_tmem + qi * 4,
                                )
                                commit(tfull + j * 8)
                            commit(kvempty + ki * 8)
                            tp = tp ^ 1
                            ki = ki + 1
                            if ki == self.stages:
                                ki = 0
                                kp = kp ^ 1
                        arrive(qempty + qi * 8, 32)
                        qi = qi + 1
                        if qi == 2:
                            qi = 0
                            qp = qp ^ 1
        elif warp < self.math_warps + 3 + self.groups:
            ki = cutlass.Int32(0)
            kp = cutlass.Int32(0)
            cw = warp - self.math_warps - 3
            running = True
            while running:
                for _copy_step in cutlass.range_constexpr(self.stages):
                    if running:
                        wait(metafull + ki * 8, kp)
                        running = load_shared(hdr + ki * 16 + 4) != cutlass.Uint32(
                            0xFFFFFFFF
                        )
                        if running:
                            physical = cutlass.Uint32(0)
                            if lane < 16 and cw * 16 + lane < load_shared(
                                hdr + ki * 16 + 4
                            ):
                                physical = (
                                    load_shared(
                                        infos
                                        + ki * self.groups * 64
                                        + (cw * 16 + lane) * 4
                                    )
                                    * 8
                                )
                            key = cute.arch.shuffle_sync(physical, lane // 2)
                            instruction(
                                "cp.async.cg.shared::cta.global.L2::64B [$0],[$1],16;",
                                "r,l,~{memory}",
                                (
                                    ksaddr
                                    + ki * self.s_stride
                                    + (cw * 16 + lane // 2) * 32
                                    + (lane % 2) * 16,
                                    cutlass.Uint64(ks.iterator.toint())
                                    + cutlass.Uint64(key) * 4
                                    + (lane % 2) * 16,
                                ),
                            )
                            async_arrive(sfcopied + ki * 8)
                            for pair in cutlass.range_constexpr(0, 16, 4):
                                bo = pair + lane // 8
                                key = cute.arch.shuffle_sync(physical, bo)
                                for cb in cutlass.range_constexpr(0, 32, 8):
                                    chunk = cb + lane % 8
                                    sw = (chunk & ~3) | (
                                        (chunk & 3) ^ ((chunk >> 3) & 3)
                                    )
                                    copy16(
                                        kaddr
                                        + ki * self.k_stride
                                        + ((cw * 16 + bo) * 32 + sw) * 16,
                                        cutlass.Uint64(k.iterator.toint())
                                        + cutlass.Uint64(key) * 64
                                        + chunk * 16,
                                    )
                            async_arrive(kvfull + ki * 8)
                            ki = ki + 1
                            if ki == self.stages:
                                ki = 0
                                kp = kp ^ 1

    @cute.jit
    def propose(self, temp, tid, lane, warp, row, length, thresholds, candidates, info):
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        lo = cutlass.Float32(float("inf"))
        hi = cutlass.Float32(float("-inf"))
        values = cute.make_rmem_tensor(2, cutlass.Float32)
        for i in cutlass.range_constexpr(2):
            v = instruction(
                "{ .reg .b16 v; ld.shared.b16 v,[$1]; cvt.f32.bf16 $0,v; }",
                "=f,r,~{memory}",
                (temp + (tid + i * 256) * 2,),
                cutlass.Float32,
            )
            values[i] = v
            if v != float("-inf"):
                lo = min(lo, v)
                hi = max(hi, v)
        lo = warp_extreme(lo, "min")
        hi = warp_extreme(hi, "max")
        if lane == 0:
            instruction(
                "st.shared.f32 [$0],$1; st.shared.f32 [$2],$3;",
                "r,f,r,f,~{memory}",
                (temp + 2048 + warp * 4, lo, temp + 2080 + warp * 4, hi),
            )
        store_shared(temp + 1024 + tid * 4, 0)
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        lo = cutlass.Float32(float("inf"))
        hi = cutlass.Float32(float("-inf"))
        if lane < 8:
            lo = instruction(
                "ld.shared.f32 $0,[$1];",
                "=f,r,~{memory}",
                (temp + 2048 + lane * 4,),
                cutlass.Float32,
            )
            hi = instruction(
                "ld.shared.f32 $0,[$1];",
                "=f,r,~{memory}",
                (temp + 2080 + lane * 4,),
                cutlass.Float32,
            )
        lo = warp_extreme(lo, "min")
        hi = warp_extreme(hi, "max")
        scale = cutlass.Float32(0)
        if hi > lo and hi < float("inf") and lo > float("-inf"):
            scale = fast_div(255.0, hi - lo)
        for i in cutlass.range_constexpr(2):
            v = values[i]
            bucket = cutlass.Int32(0)
            if scale > 0:
                bucket = cutlass.Int32(min(255.0, max(0.0, (hi - v) * scale)))
            if v == float("-inf") or v != v:
                bucket = 255
            instruction(
                "red.relaxed.cta.shared.add.u32 [$0],1;",
                "r,~{memory}",
                (temp + 1024 + bucket * 4,),
            )
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        amount = load_shared(temp + 1024 + tid * 4)
        prefix = amount
        for i in cutlass.range_constexpr(5):
            v = cute.arch.shuffle_sync_up(prefix, 1 << i)
            if lane >= (1 << i):
                prefix = prefix + v
        if lane == 31:
            store_shared(temp + 2048 + warp * 4, prefix)
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        for w in cutlass.range_constexpr(8):
            if w < warp:
                prefix = prefix + load_shared(temp + 2048 + w * 4)
        sample_valid = max(
            0, (min(64, cute.ceil_div(length, 8)) - 1) * 8 + (length - 1) % 8 + 1
        )
        expected = fast_div(
            512.0 * cutlass.Float32(sample_valid), cutlass.Float32(max(1, length))
        )
        rank = min(
            sample_valid,
            max(1, cutlass.Int32(expected + 5.0 * cute.sqrt(expected) + 1.0)),
        )
        if prefix >= rank and prefix - amount < rank:
            threshold = lo
            if scale > 0:
                threshold = max(
                    lo, histogram_edge(hi, cutlass.Float32(tid) + 1.0, scale)
                )
            if length <= 512:
                threshold = cutlass.Float32(float("-inf"))
            thresholds[row] = threshold
            info[row, 256] = cutlass.BFloat16(hi)
            info[row, 257] = cutlass.BFloat16(threshold)
        rank = min(sample_valid, max(1, cutlass.Int32(expected / 4.0)))
        if prefix >= rank and prefix - amount < rank:
            upper = hi
            if scale > 0:
                upper = histogram_edge(hi, cutlass.Float32(tid), scale)
            info[row, 258] = cutlass.BFloat16(upper)
        if tid < 32:
            candidates[row, tid] = 0
        if tid == 0 and length == 0:
            thresholds[row] = cutlass.Float32(float("-inf"))
            for i in cutlass.range_constexpr(256, 259):
                info[row, i] = cutlass.BFloat16(float("-inf"))
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
