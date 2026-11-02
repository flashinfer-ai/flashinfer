# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, SIM102, B905
"""FROST decode GDN block for Qwen3.8-27B (TP1): BA projection, causal conv + SiLU,
gated delta rule state update, gated RMSNorm and static FP8 output quantization
in one launch, after the FP8 QKVZ projection.

Grid: 48 V heads x 4-CTA clusters x request rows; each cluster owns one V head's
128 state rows. Q/K convolution history is shared by the three V heads of a K
head: every reader CTA recomputes it, and the last of the 12 readers (a relaxed
per-request ticket) shifts the history. Request rows with state slot <= 0 (a
padding row) write zeros and leave all state untouched. Rounding follows the
unfused op chain: BA in BF16, convolution SiLU in the ex2.approx PTX form, BF16
core output, BF16 normed value, then FP32-reciprocal FP8 quantization.

Shapes are fixed: 16 K heads, 48 V heads, head dim 128, conv width 4, hidden
5120.
"""

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.cute_dsl.fp4_common import (
    bfloat2_to_float2_scaled,
    get_ptr_as_int64,
    get_smem_ptr_as_int32,
    ld_global_v4_u32,
    st_global_v4_u32,
)


@dsl_user_op
def as_float(v: cutlass.Uint32, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.bitcast(T.f32(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def as_bits(v: cutlass.Float32, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.bitcast(T.i32(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def cluster_sync(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def peer_float(addr: cutlass.Int32, rank: cutlass.Int32, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [addr.ir_value(loc=loc, ip=ip), rank.ir_value(loc=loc, ip=ip)],
            "{ .reg .u32 p; mapa.shared::cluster.u32 p, $1, $2; ld.shared::cluster.f32 $0, [p]; }",
            "=f,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def state_copy_16(dst: cutlass.Int32, src: cutlass.Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [dst.ir_value(loc=loc, ip=ip), src.ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def state_load_16(addr: cutlass.Int32, *, loc=None, ip=None):
    # Memory clobber keeps the subsequent same-slice asynchronous overwrite
    # after these loads. Only this thread copies to or reads these addresses.
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@cute.jit
def prefetch_tile(
    state,
    stage,
    slot,
    head,
    hw,
    kcol,
    row_base,
    SS: cutlass.Constexpr,
    VPER: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
):
    for v in cutlass.range_constexpr(4):
        off = (
            slot * SS + (head * 128 + row_base + hw * VPER + TILE * 4 + v) * 128 + kcol
        )
        shared = (hw * 4 + v) * 128 + kcol
        state_copy_16(
            get_smem_ptr_as_int32(stage, shared), get_ptr_as_int64(state, off)
        )
        state_copy_16(
            get_smem_ptr_as_int32(stage, shared + 4), get_ptr_as_int64(state, off + 4)
        )
    cute.arch.cp_async_commit_group()


@cute.jit
def warp_sum(x):
    for step in cutlass.range_constexpr(4):
        x = x + cute.arch.shuffle_sync_bfly(x, offset=1 << step)
    return x


@dsl_user_op
def reader_done(addr: cutlass.Int64, limit: cutlass.Int32, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip), limit.ir_value(loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.inc.u32 $0, [$1], $2;",
            "=r,l,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def stock_conv_silu(value: cutlass.Float32, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [value.ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 t, e, d; mul.f32 t, $1, 0fBFB8AA3B; ex2.approx.f32 e, t; add.f32 d, e, 0f3F800000; div.full.f32 $0, $1, d; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def quant_fp8(value: cutlass.Float32, scale: cutlass.Float32, *, loc=None, ip=None):
    # The per-tensor quant multiplies by the FP32 reciprocal before SATFINITE.
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip), scale.ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 inv, v; .reg .b16 packed; div.rn.f32 inv, 0f3F800000, $2; mul.f32 v, $1, inv; cvt.rn.satfinite.e4m3x2.f32 packed, 0f00000000, v; cvt.u32.u16 $0, packed; }",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    ).to(cutlass.Uint8)


@cute.jit
def conv_channel(
    x,
    cs,
    cw,
    channel,
    slot,
    CSS: cutlass.Constexpr,
    CSD: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    PROUND: cutlass.Constexpr,
    WRITE: cutlass.Constexpr,
):
    base = slot * CSS + channel * CSD
    c0 = cs[base].to(cutlass.Float32)
    c1 = cs[base + CST].to(cutlass.Float32)
    c2 = cs[base + 2 * CST].to(cutlass.Float32)
    current = x[channel].to(cutlass.Float32)
    acc = cutlass.Float32(0)
    vals = cute.make_rmem_tensor((4,), cutlass.Float32)
    vals[0] = c0
    vals[1] = c1
    vals[2] = c2
    vals[3] = current
    for j in cutlass.range_constexpr(4):
        prod = vals[j] * cw[channel * 4 + j].to(cutlass.Float32)
        if cutlass.const_expr(PROUND):
            prod = prod.to(cutlass.BFloat16).to(cutlass.Float32)
        acc = acc + prod
    # All historical values have been read by their unique channel owner.
    if cutlass.const_expr(WRITE):
        cs[base] = c1.to(cutlass.BFloat16)
        cs[base + CST] = c2.to(cutlass.BFloat16)
        cs[base + 2 * CST] = current.to(cutlass.BFloat16)
    return stock_conv_silu(acc).to(cutlass.BFloat16).to(cutlass.Float32)


@cute.kernel
def kernel(
    x: cute.Tensor,
    cs: cute.Tensor,
    cw: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    al: cute.Tensor,
    dt: cute.Tensor,
    state: cute.Tensor,
    indices: cute.Tensor,
    z: cute.Tensor,
    w: cute.Tensor,
    counts: cute.Tensor,
    out: cute.Tensor,
    raw: cute.Tensor,
    convout: cute.Tensor,
    scale: cute.Tensor,
    qout: cute.Tensor,
    hidden: cute.Tensor,
    baweight: cute.Tensor,
    ba_out: cute.Tensor,
    BATCH: cutlass.Constexpr,
    XS: cutlass.Constexpr,
    AS: cutlass.Constexpr,
    BS: cutlass.Constexpr,
    ZS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CSD: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    CTAS: cutlass.Constexpr,
    WARPS: cutlass.Constexpr,
    PROUND: cutlass.Constexpr,
    DEBUG: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, row, _ = cute.arch.block_idx()
    rank = bid % CTAS
    head = bid // CTAS
    kh = head // 3
    # Independent request rows own separate inputs, output rows and read-completion counters.
    # All CTAs in a four-CTA cluster have the same row/state slot.
    x = cute.make_tensor(x.iterator + row * XS, cute.make_layout(10240))
    a = cute.make_tensor(a.iterator + row * AS, cute.make_layout(48))
    b = cute.make_tensor(b.iterator + row * BS, cute.make_layout(48))
    z = cute.make_tensor(z.iterator + row * ZS, cute.make_layout(6144))
    out = cute.make_tensor(out.iterator + row * 6144, cute.make_layout(6144))
    raw = cute.make_tensor(raw.iterator + row * 6144, cute.make_layout(6144))
    convout = cute.make_tensor(convout.iterator + row * 10240, cute.make_layout(10240))
    qout = cute.make_tensor(qout.iterator + row * 6144, cute.make_layout(6144))
    counts = cute.make_tensor(counts.iterator + row * 512, cute.make_layout(512))
    lane = tid % 16
    warp = tid // 16
    kcol = lane * 8
    slot = indices[row].to(cutlass.Int64)
    ROWS: cutlass.Constexpr = 128 // CTAS
    HP: cutlass.Constexpr = WARPS * 2
    VPER: cutlass.Constexpr = ROWS // HP
    smem = cutlass.memory.SmemAllocator()
    ticket = smem.allocate_tensor(cutlass.Int32, cute.make_layout(1), byte_alignment=16)
    qks = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(256), byte_alignment=16
    )
    vs = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(ROWS), byte_alignment=16
    )
    invs = smem.allocate_tensor(cutlass.Float32, cute.make_layout(3), byte_alignment=16)
    partial = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(3 * (HP + 1)), byte_alignment=16
    )
    ba_partials = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(2 * WARPS), byte_alignment=16
    )
    state_stage = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(HP * 4 * 128), byte_alignment=128
    )
    if slot > 0:
        prefetch_tile(
            state, state_stage, slot, head, warp, kcol, rank * ROWS, SS, VPER, 0
        )
        bdot = cutlass.Float32(0)
        adot = cutlass.Float32(0)
        for chunk in cutlass.range_constexpr((5120 + WARPS * 256 - 1) // (WARPS * 256)):
            kk = chunk * WARPS * 256 + tid * 8
            if kk < 5120:
                hx = ld_global_v4_u32(get_ptr_as_int64(hidden, row * 5120 + kk))
                bw = ld_global_v4_u32(get_ptr_as_int64(baweight, head * 5120 + kk))
                aw = ld_global_v4_u32(
                    get_ptr_as_int64(baweight, (head + 48) * 5120 + kk)
                )
                for j in cutlass.range_constexpr(4):
                    h0, h1 = bfloat2_to_float2_scaled(hx[j], cutlass.Float32(1))
                    b0, b1 = bfloat2_to_float2_scaled(bw[j], cutlass.Float32(1))
                    a0, a1 = bfloat2_to_float2_scaled(aw[j], cutlass.Float32(1))
                    bdot = bdot + h0 * b0
                    bdot = bdot + h1 * b1
                    adot = adot + h0 * a0
                    adot = adot + h1 * a1
        for step in cutlass.range_constexpr(5):
            bdot = bdot + cute.arch.shuffle_sync_bfly(bdot, offset=1 << step)
            adot = adot + cute.arch.shuffle_sync_bfly(adot, offset=1 << step)
        if tid % 32 == 0:
            ba_partials[tid // 32] = bdot
            ba_partials[WARPS + tid // 32] = adot
        # Every CTA recomputes small Q/K convolution into local shared memory.
        # No Q/K state writes until every CTA has finished reading old history.
        for chunk in cutlass.range_constexpr((256 + WARPS * 32 - 1) // (WARPS * 32)):
            ix = chunk * WARPS * 32 + tid
            if ix < 256:
                channel = kh * 128 + ix % 128 + (ix // 128) * 2048
                value = conv_channel(
                    x, cs, cw, channel, slot, CSS, CSD, CST, PROUND, False
                )
                qks[ix] = value
                if cutlass.const_expr(DEBUG):
                    if rank == 0 and head % 3 == 0:
                        convout[channel] = value.to(cutlass.BFloat16)
        if tid < ROWS:
            channel = 4096 + head * 128 + rank * ROWS + tid
            value = conv_channel(x, cs, cw, channel, slot, CSS, CSD, CST, PROUND, True)
            vs[tid] = value
            if cutlass.const_expr(DEBUG):
                convout[channel] = value.to(cutlass.BFloat16)
        # Every CTA publishes completion of its old Q/K-history reads. Nobody
        # waits for another cluster. The last of12CTAs performs the state shift.
        cute.arch.barrier()
        if tid == 0:
            ticket[0] = reader_done(
                get_ptr_as_int64(counts, kh * 32), cutlass.Int32(3 * CTAS - 1)
            )
        cute.arch.barrier()
        # Ticket11 proves all twelve readers completed; exactly one CTA writes.
        if ticket[0] == 3 * CTAS - 1:
            for chunk in cutlass.range_constexpr(
                (256 + WARPS * 32 - 1) // (WARPS * 32)
            ):
                ix = chunk * WARPS * 32 + tid
                if ix < 256:
                    channel = kh * 128 + ix % 128 + (ix // 128) * 2048
                    off = slot * CSS + channel * CSD
                    c1 = cs[off + CST]
                    c2 = cs[off + 2 * CST]
                    cs[off] = c1
                    cs[off + CST] = c2
                    cs[off + 2 * CST] = x[channel]
        q = cute.make_rmem_tensor((8,), cutlass.Float32)
        k = cute.make_rmem_tensor((8,), cutlass.Float32)
        qs = cutlass.Float32(0)
        ks = cutlass.Float32(0)
        for j in cutlass.range_constexpr(8):
            q[j] = qks[kcol + j]
            k[j] = qks[128 + kcol + j]
            qs = qs + q[j] * q[j]
            ks = ks + k[j] * k[j]
        qscale = cutlass.Float32(0.08838834764831845) / cute.math.sqrt(
            warp_sum(qs) + cutlass.Float32(1e-6)
        )
        kscale = cutlass.Float32(1) / cute.math.sqrt(
            warp_sum(ks) + cutlass.Float32(1e-6)
        )
        for j in cutlass.range_constexpr(8):
            q[j] = q[j] * qscale
            k[j] = k[j] * kscale
        # Three disjoint warp groups update the three V heads concurrently.
        hh = cutlass.Int32(0)
        hw = warp
        btotal = cutlass.Float32(0)
        atotal = cutlass.Float32(0)
        if tid % 32 < WARPS:
            btotal = ba_partials[tid % 32]
            atotal = ba_partials[WARPS + tid % 32]
        for step in cutlass.range_constexpr(5):
            btotal = btotal + cute.arch.shuffle_sync_bfly(btotal, offset=1 << step)
            atotal = atotal + cute.arch.shuffle_sync_bfly(atotal, offset=1 << step)
        bvalue = btotal.to(cutlass.BFloat16).to(cutlass.Float32)
        avalue = atotal.to(cutlass.BFloat16).to(cutlass.Float32)
        if cutlass.const_expr(DEBUG):
            if tid == 0 and rank == 0:
                ba_out[row * 96 + head] = bvalue.to(cutlass.BFloat16)
                ba_out[row * 96 + head + 48] = avalue.to(cutlass.BFloat16)
        av = avalue + dt[head].to(cutlass.Float32)
        soft = av
        if av <= cutlass.Float32(20):
            soft = cute.math.log(cutlass.Float32(1) + cute.math.exp(av))
        decay = cute.math.exp(-cute.math.exp(al[head].to(cutlass.Float32)) * soft)
        beta = cutlass.Float32(1) / (cutlass.Float32(1) + cute.math.exp(-bvalue))
        # Each lane only needs its own V row for norm/output; select it early.
        value = cutlass.Float32(0)
        # Interleave 4 independent V rows. Per-row arithmetic and each
        # butterfly reduction retain their original order; token/state ownership unchanged.
        for tile in cutlass.range_constexpr(VPER // 4):
            cute.arch.cp_async_wait_group(0)
            histories = cute.make_rmem_tensor((4, 8), cutlass.Float32)
            dots = cute.make_rmem_tensor((4,), cutlass.Float32)
            outputs = cute.make_rmem_tensor((4,), cutlass.Float32)
            for v in cutlass.range_constexpr(4):
                vc = rank * ROWS + hw * VPER + tile * 4 + v
                off = slot * SS + (head * 128 + vc) * 128 + kcol
                shared = (hw * 4 + v) * 128 + kcol
                packed0 = state_load_16(get_smem_ptr_as_int32(state_stage, shared))
                packed1 = state_load_16(get_smem_ptr_as_int32(state_stage, shared + 4))
                for j in cutlass.range_constexpr(4):
                    histories[v, j] = as_float(packed0[j]) * decay
                    histories[v, j + 4] = as_float(packed1[j]) * decay
            # All four rows have been consumed into registers before reuse.
            if cutlass.const_expr(tile + 1 < VPER // 4):
                prefetch_tile(
                    state,
                    state_stage,
                    slot,
                    head,
                    hw,
                    kcol,
                    rank * ROWS,
                    SS,
                    VPER,
                    tile + 1,
                )
            for v in cutlass.range_constexpr(4):
                dots[v] = cutlass.Float32(0)
                for j in cutlass.range_constexpr(8):
                    dots[v] = dots[v] + histories[v, j] * k[j]
            for step in cutlass.range_constexpr(4):
                for v in cutlass.range_constexpr(4):
                    dots[v] = dots[v] + cute.arch.shuffle_sync_bfly(
                        dots[v], offset=1 << step
                    )
            for v in cutlass.range_constexpr(4):
                innovation = (vs[hh * ROWS + hw * VPER + tile * 4 + v] - dots[v]) * beta
                outputs[v] = cutlass.Float32(0)
                for j in cutlass.range_constexpr(8):
                    histories[v, j] = histories[v, j] + innovation * k[j]
                    outputs[v] = outputs[v] + histories[v, j] * q[j]
            for step in cutlass.range_constexpr(4):
                for v in cutlass.range_constexpr(4):
                    outputs[v] = outputs[v] + cute.arch.shuffle_sync_bfly(
                        outputs[v], offset=1 << step
                    )
            for v in cutlass.range_constexpr(4):
                vc = rank * ROWS + hw * VPER + tile * 4 + v
                off = slot * SS + (head * 128 + vc) * 128 + kcol
                st_global_v4_u32(
                    get_ptr_as_int64(state, off),
                    as_bits(histories[v, 0]),
                    as_bits(histories[v, 1]),
                    as_bits(histories[v, 2]),
                    as_bits(histories[v, 3]),
                )
                st_global_v4_u32(
                    get_ptr_as_int64(state, off + 4),
                    as_bits(histories[v, 4]),
                    as_bits(histories[v, 5]),
                    as_bits(histories[v, 6]),
                    as_bits(histories[v, 7]),
                )
                if lane == tile * 4 + v:
                    value = outputs[v].to(cutlass.BFloat16).to(cutlass.Float32)
        ss = warp_sum(value * value)
        if lane == 0:
            partial[hh * (HP + 1) + hw] = ss
        cute.arch.barrier()
        total = cutlass.Float32(0)
        if lane < HP:
            total = partial[hh * (HP + 1) + lane]
        total = warp_sum(total)
        if hw == 0 and lane == 0:
            partial[hh * (HP + 1) + HP] = total
        cute.arch.barrier()
        if cutlass.const_expr(CTAS > 1):
            cluster_sync()
        # Every physical-warp lane named by shuffle's full member mask must
        # participate. Both half-warps compute the same norm; only one writes.
        if hw < 2:
            peer = cutlass.Float32(0)
            if cutlass.const_expr(CTAS > 1):
                if lane < CTAS:
                    peer = peer_float(
                        (partial.iterator + hh * (HP + 1) + HP)
                        .toint()
                        .to(cutlass.Int32),
                        lane,
                    )
            else:
                if lane == 0:
                    peer = partial[hh * (HP + 1) + HP]
            inv = cute.math.rsqrt(
                warp_sum(peer) / cutlass.Float32(128) + cutlass.Float32(EPS)
            )
            if hw == 0 and lane == 0:
                invs[hh] = inv
        cute.arch.barrier()
        inv = invs[hh]
        if lane < VPER:
            vc = rank * ROWS + hw * VPER + lane
            oi = head * 128 + vc
            gate = z[oi].to(cutlass.Float32)
            rounded = (
                ((value * inv) * w[vc].to(cutlass.Float32))
                * (gate / (cutlass.Float32(1) + cute.math.exp(-gate)))
            ).to(cutlass.BFloat16)
            qout[oi] = quant_fp8(rounded.to(cutlass.Float32), scale[0])
            out[oi] = rounded
            if cutlass.const_expr(DEBUG):
                raw[oi] = value.to(cutlass.BFloat16)
        # This protects both Q/K publication storage and every norm-partial peer read.
        if cutlass.const_expr(CTAS > 1):
            cluster_sync()
    else:
        if tid < ROWS:
            oi = head * 128 + rank * ROWS + tid
            qout[oi] = cutlass.Uint8(0)
            out[oi] = cutlass.BFloat16(0)
            if cutlass.const_expr(DEBUG):
                raw[oi] = cutlass.BFloat16(0)


@cute.jit
def launch(
    x,
    cs,
    cw,
    a,
    b,
    al,
    dt,
    state,
    indices,
    z,
    w,
    counts,
    out,
    raw,
    convout,
    scale,
    qout,
    hidden,
    baweight,
    ba_out,
    BATCH: cutlass.Constexpr,
    XS: cutlass.Constexpr,
    AS: cutlass.Constexpr,
    BS: cutlass.Constexpr,
    ZS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CSD: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    CTAS: cutlass.Constexpr,
    WARPS: cutlass.Constexpr,
    PROUND: cutlass.Constexpr,
    DEBUG: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    kernel(
        x,
        cs,
        cw,
        a,
        b,
        al,
        dt,
        state,
        indices,
        z,
        w,
        counts,
        out,
        raw,
        convout,
        scale,
        qout,
        hidden,
        baweight,
        ba_out,
        BATCH,
        XS,
        AS,
        BS,
        ZS,
        CSS,
        CSD,
        CST,
        SS,
        EPS,
        CTAS,
        WARPS,
        PROUND,
        DEBUG,
    ).launch(
        grid=(48 * CTAS, BATCH, 1),
        block=(32 * WARPS, 1, 1),
        cluster=(CTAS, 1, 1),
        stream=stream,
    )


def _flat(t: torch.Tensor):
    """A 1-D view covering a (possibly strided) tensor's whole footprint."""
    n = 1 + sum((d - 1) * s for d, s in zip(t.shape, t.stride()))
    return from_dlpack(
        t.detach().as_strided((n,), (1,)), assumed_align=16 if n > 1 else 4
    )


class FrostGdnConsumer:
    """Compiled per (request rows, cache/input strides), shared by every GDN
    layer: all per-layer tensors are launch arguments.

    Inputs per launch: ``x`` the QKV part of the projected QKVZ rows (BF16,
    row-strided view), ``z`` its Z part viewed [rows, 48, 128], ``hidden`` the
    BF16 normed layer input (BA projection source), the layer's conv weight
    [10240, 4], A_log, dt_bias, gated-norm weight, BA weight [96, 5120], conv
    and SSM state pools, per-row state slots and the static FP8 output scale.
    Outputs: ``qout`` FP8 [rows, 48, 128] (and the BF16 norm in ``out``)."""

    WARPS = 4

    def __init__(
        self,
        device: torch.device,
        eps: float,
        debug: bool = False,
        ctas: int | None = None,
        warps: int | None = None,
    ):
        self.device = device
        self.eps = float(eps)
        self.debug = debug
        assert ctas in (None, 1, 2, 4)
        self.fixed_ctas = ctas
        self.fixed_warps = warps
        self.compiled: dict[tuple, Any] = {}
        self.buffers: dict[int, dict[str, torch.Tensor]] = {}

    def ctas(self, rows: int) -> int:
        """CTAs per V head: two are faster at one or two rows, one from four rows
        on."""
        if self.fixed_ctas is not None:
            return self.fixed_ctas
        return 2 if rows == 1 else 1

    def warps(self, rows: int) -> int:
        return self.fixed_warps or self.WARPS

    def outputs(self, rows: int):
        if rows not in self.buffers:
            dev = self.device
            self.buffers[rows] = dict(
                qout=torch.empty(rows, 48, 128, dtype=torch.float8_e4m3fn, device=dev),
                out=torch.empty(rows, 48, 128, dtype=torch.bfloat16, device=dev),
                counts=torch.zeros(rows * 16 * 32, dtype=torch.int32, device=dev),
                dummy_ab=torch.zeros(rows, 96, dtype=torch.bfloat16, device=dev),
                dummy_x=torch.zeros(rows, 10240, dtype=torch.bfloat16, device=dev),
                raw=torch.zeros(rows, 48, 128, dtype=torch.bfloat16, device=dev),
            )
        return self.buffers[rows]

    def key(self, rows, x, cs, state, z):
        return (
            rows,
            self.ctas(rows),
            self.warps(rows),
            x.stride(0),
            z.stride(0),
            *cs.stride(),
            state.stride(0),
        )

    def _args(self, rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw):
        # The kernel reads indices[row] densely.
        assert idx.is_contiguous() and idx.numel() >= rows
        b = self.outputs(rows)
        ab = b["dummy_ab"]
        return [
            _flat(x),
            _flat(cs),
            _flat(cw),
            _flat(ab[:, 48:]),
            _flat(ab[:, :48]),
            _flat(al),
            _flat(dt),
            _flat(state),
            _flat(idx),
            _flat(z),
            _flat(w),
            _flat(b["counts"]),
            _flat(b["out"]),
            _flat(b["raw"]),
            _flat(b["dummy_x"]),
            _flat(scale),
            _flat(b["qout"].view(torch.uint8)),
            _flat(hidden),
            _flat(baw),
            _flat(ab),
        ]

    def compile(self, rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw):
        key = self.key(rows, x, cs, state, z)
        if key in self.compiled:
            return
        assert cs.shape[1:] == (10240, 3) and state.shape[1:] == (48, 128, 128)
        assert state.stride()[1:] == (16384, 128, 1) and state.dtype == torch.float32
        args = self._args(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)
        ab_stride = self.outputs(rows)["dummy_ab"].stride(0)
        cstr = cs.stride()
        self.compiled[key] = cute.compile(
            launch,
            *args,
            rows,
            x.stride(0),
            ab_stride,
            ab_stride,
            z.stride(0),
            *cstr,
            state.stride(0),
            self.eps,
            self.ctas(rows),
            self.warps(rows),
            True,
            self.debug,
            make_fake_stream(),
        )

    def __call__(self, rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw):
        """Returns the FP8 [rows, 48, 128] output buffer."""
        key = self.key(rows, x, cs, state, z)
        args = self._args(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)
        self.compiled[key](
            *args, cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        )
        return self.outputs(rows)["qout"]
