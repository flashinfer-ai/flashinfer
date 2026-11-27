"""SM100 NVFP4 direct GEMM (M <= 4) for the per-token MoE.

One CTA of 4 warps per (route, 128 weight rows), no warp specialization. The MMA is 128 weight rows
x 16 activation rows (the TMA box starts at the route's input row, so only accumulator column 0 is used).

Structure:
- prologue: warp 0 prefetches the TMA descriptors (they are read from global memory) and allocates
  64 TMEM columns; ``griddepcontrol.launch_dependents`` (PDL; after the wait with ``trigger_late``), the
  expert and the weight loads before ``griddepcontrol.wait`` (see ``DirectGemm``), the row scale after it;
  mbarrier init by warp 0 without ``fence.mbarrier_init``;
- shared memory: W stages, W_SF stages, A stages, A_SF stages, ready[S], mma_done[2] in that
  order (offsets 0 / 65536 / 73728 / 81920 / 90112 / 90144 for 4 stages of BK 256);
  TMEM: accumulator 128x16 at column 0, weight scales at 16, activation scales at 32;
- the software pipeline: all stages are filled up front; per K tile every thread waits
  ``ready[slot]``, warp 0 copies the weight then (after ``bar.sync``) the activation scales to
  TMEM and issues 4 MMAs + a commit to ``mma_done[ki % 2]``; from the second tile on, every
  thread waits for the previous MMA and refills its slot. Each TMA is preceded by a CTA-wide
  ``bar.sync`` and the expect_tx is issued by thread 0;
- epilogue (SwiGLU): ``tcgen05.ld.32x32b.x16``, column 0 (+0.0), times a_gs * w_gs, the 128
  values go through shared memory (the rows are interleaved per 8 and a plain ``st.shared`` /
  ``ld.shared.v2`` pairs them) so that thread j < 64 holds gate j and up j, then one predicated
  ``st.global.b16`` (``unit_out``: warps 0 and 1 quantize their 32 outputs each to NVFP4 with block scales only).

Inline PTX is used only for the predicated issue of the TMA loads, block-scale copies and MMAs
(``[P1]``, ``[P2]``); see ``HANDWRITTEN_PTX.md``.
"""

import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass import Float32, Int32, Int64, Uint16, Uint32
from cutlass._mlir import ir
from cutlass.cute.nvgpu import warp as nvwarp
from cutlass.cutlass_dsl import dsl_user_op

from .activation import as_activation
from .gemm import (
    LOG2E,
    OPERAND_DESC,
    SCALE_DESC,
    _asm,
    _encode,
    cvt_e2m1x2,
    cvt_e4m3x2,
    sel_if_pos,
    div_full,
    ex2_approx,
    fp4_unit_word,
    mma_idesc,
    prefetch_tensormap,
    gmem_bf16,
    smem_ptr,
    st_global,
    tmem_load,
    ld_shared_v2_f32,
    use_pdl,
)


# --------------------------------------------------------------------------------------------
# Inline PTX (cases and evidence: HANDWRITTEN_PTX.md, ids in brackets)
# --------------------------------------------------------------------------------------------
# [P1] The refill predicate (k tile < K) is evaluated inside the asm (``setp`` + ``@p``):
# an ``Int32(nk < K)`` computed in the DSL was folded to true by CuTeDSL's MLIR pipeline inside the
# ``if ki > 0`` region, which issued out-of-range TMAs whose zero fill raced with the epilogue's
# use of stage 0. [P2] The elected lane of warp 0 issues TMA, tcgen05.cp and MMA under a predicate;
# ``cute.arch.elect_one()`` branches instead.
@dsl_user_op
def expect_tx_if_lt(bar, nbytes: int, kk, k: int, *, loc=None, ip=None):
    """``@p mbarrier.arrive.expect_tx`` with p = kk < k [P1]."""
    _asm(
        None,
        [Int32(bar).ir_value(loc=loc, ip=ip), Int32(kk).ir_value(loc=loc, ip=ip)],
        f"{{ .reg .pred p; setp.lt.s32 p, $1, {k}; "
        f"@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {nbytes}; }}",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def tma_load_if_lt(dst, desc, bar, coords, tid, kk, k: int, *, loc=None, ip=None):
    """``@p cp.async.bulk.tensor.{n}d`` global -> shared (mbarrier completion) by the elected lane of warp 0,
    p = elect && tid < 32 && kk < k [P1, P2]. No L2 cache policy."""
    n = len(coords)
    cs = ", ".join(f"${3 + i}" for i in range(n))
    ops = [
        Int32(dst).ir_value(loc=loc, ip=ip),
        Int64(desc).ir_value(loc=loc, ip=ip),
        Int32(bar).ir_value(loc=loc, ip=ip),
    ]
    ops += [Int32(c).ir_value(loc=loc, ip=ip) for c in coords]
    ops += [Int32(tid).ir_value(loc=loc, ip=ip), Int32(kk).ir_value(loc=loc, ip=ip)]
    t, kv = 3 + n, 4 + n
    _asm(
        None,
        ops,
        f"{{ .reg .pred p, e, w; .reg .b32 r; elect.sync r|e, -1; setp.lt.u32 w, ${t}, 32; "
        f"setp.lt.s32 p, ${kv}, {k}; and.pred p, p, e; and.pred p, p, w; "
        f"@p cp.async.bulk.tensor.{n}d.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1, {{{cs}}}], "
        f"[$2]; }}",
        "r,l,r" + ",r" * n + ",r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def tcgen05_cp_x4_elect(tid, taddrs, sdescs, *, loc=None, ip=None):
    """Elected lane of warp 0 (predicated, no branch): ``tcgen05.cp.cta_group::1.warpx4.32x128b`` per pair [P2]."""
    n = len(taddrs)
    ops = [Int32(tid).ir_value(loc=loc, ip=ip)] + [
        Int32(a).ir_value(loc=loc, ip=ip) for a in taddrs
    ]
    ops += [Int64(d).ir_value(loc=loc, ip=ip) for d in sdescs]
    body = " ".join(
        f"@p tcgen05.cp.cta_group::1.warpx4.32x128b [${1 + i}], ${1 + n + i};"
        for i in range(n)
    )
    _asm(
        None,
        ops,
        "{ .reg .pred p, w; .reg .b32 r; elect.sync r|p, -1; setp.lt.u32 w, $0, 32; and.pred p, p, w; "
        + body
        + " }",
        "r" + ",r" * n + ",l" * n,
        True,
        loc,
        ip,
    )


@dsl_user_op
def tcgen05_mma_nvf4_elect(
    d_taddr,
    adescs,
    bdescs,
    idesc: int,
    asf_taddrs,
    bsf_taddrs,
    first,
    commit_bar,
    *,
    loc=None,
    ip=None,
):
    """One elected lane (predicated): the K block's MMAs (the first accumulates iff ``first`` != 0),
    then ``tcgen05.commit`` to ``commit_bar`` [P2]."""
    n = len(adescs)
    ops = [
        Int32(d_taddr).ir_value(loc=loc, ip=ip),
        Int32(first).ir_value(loc=loc, ip=ip),
        Int32(commit_bar).ir_value(loc=loc, ip=ip),
    ]
    ops += [Int64(x).ir_value(loc=loc, ip=ip) for x in adescs] + [
        Int64(x).ir_value(loc=loc, ip=ip) for x in bdescs
    ]
    ops += [Int32(x).ir_value(loc=loc, ip=ip) for x in asf_taddrs] + [
        Int32(x).ir_value(loc=loc, ip=ip) for x in bsf_taddrs
    ]
    a0, b0, sa0, sb0 = 3, 3 + n, 3 + 2 * n, 3 + 3 * n
    body = []
    for i in range(n):
        acc = "q" if i == 0 else "1"
        body.append(
            f"@e tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [$0], ${a0 + i}, ${b0 + i}, "
            f"{idesc}, [${sa0 + i}], [${sb0 + i}], {acc};"
        )
    body.append(
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$2];"
    )
    _asm(
        None,
        ops,
        "{ .reg .pred e, q; .reg .b32 r; elect.sync r|e, -1; setp.ne.s32 q, $1, 0; "
        + " ".join(body)
        + " }",
        "r,r,r" + ",l" * (2 * n) + ",r" * (2 * n),
        True,
        loc,
        ip,
    )


# --------------------------------------------------------------------------------------------
# Conversions through CuTeDSL / NVVM ops
# --------------------------------------------------------------------------------------------
def max3_f32(a, b, c):
    """max(a, b, c): two ``max.f32``, which ptxas fuses into one FMNMX3."""
    return cute.arch.fmax(cute.arch.fmax(a, b), c)


# --------------------------------------------------------------------------------------------
# CuTeDSL building blocks (integer shared-memory addresses)
# --------------------------------------------------------------------------------------------
def stmatrix_x4(addr, words):
    """``stmatrix.m8n8.x4.b16`` of 4 packed words with this thread's row address."""
    src = cute.make_rmem_tensor((4,), Uint32)
    for i in range(4):
        src[i] = words[i]
    dst = cute.make_tensor(smem_ptr(Uint16, addr), cute.make_layout(8))
    cute.copy(
        cute.make_copy_atom(nvwarp.StMatrix8x8x16bOp(False, 4), Uint16),
        cute.recast_tensor(src, Uint16),
        dst,
    )


def ld_bf16x4(addr):
    """``ld.global.v2.b32`` of 4 BF16 at an 8-byte aligned address."""
    p = cute.make_ptr(cutlass.BFloat16, addr, cute.AddressSpace.gmem, assumed_align=8)
    return cute.TensorSSA(
        cute.arch.load(p, ir.VectorType.get([4], cutlass.BFloat16.mlir_type)),
        (4,),
        cutlass.BFloat16,
    )


GS_SCALE = 1.0 / 2688.0  # 0f39C30C31: global scale = amax / (448 * 6)


def make_direct_descriptors(gemm: "DirectGemm", a_q, a_sf, w_q, w_sf):
    """[W, A, W_SF, A_SF] descriptors (128 B each): W boxes of 128 rows, A boxes of 16 rows starting at
    the route's input row, block scales as 5-D [256, 2, K/64, tiles, 1] boxes of one 128-row tile (A: one
    tile per input row)."""
    import torch

    k, bk = gemm.k, gemm.bk
    row_bytes = k // 64 * 512

    def operand(x, rows_per_box):
        cols = x.shape[-1]
        return _encode(
            x.data_ptr(),
            [cols, x.numel() // cols],
            [cols],
            [bk // 2, rows_per_box],
            True,
        )

    def scales(x):
        tiles = x.numel() // row_bytes
        return _encode(
            x.data_ptr(),
            [256, 2, k // 64, tiles, 1],
            [256, 512, row_bytes, tiles * row_bytes],
            [256, 2, bk // 64, 1, 1],
            False,
        )

    descs = [
        operand(w_q, 128),
        operand(a_q, 16) if a_q is not None else [0] * 16,
        scales(w_sf),
        scales(a_sf) if a_sf is not None else [0] * 16,
    ]
    flat = [v - (1 << 64) if v >= (1 << 63) else v for d in descs for v in d]
    return torch.tensor(flat, dtype=torch.int64, device=w_q.device)


class DirectGemm:
    """One CTA per (route, 128 weight rows): ``W[expert rows] @ A[input row]``.

    Args:
        n: Weight rows per expert (2I for gate/up with SwiGLU, H for down).
        k: Reduction size.
        bk: K per stage (256).
        stages: Stages in flight.
        act: SwiGLU of the gate/up rows interleaved per 8 (output N / 2 columns), else BF16 [routes, N].
        routes_per_row: Routes sharing an input row (topk for gate/up, 1 for down).
        trigger_late: PDL (always on; every weight load before the wait, see ``__init__``) with launch_dependents after
            the wait instead of before. The weights are loaded without an L2 cache policy.
    """

    def __init__(
        self,
        n,
        k,
        bk,
        stages,
        act,
        routes_per_row,
        *,
        x_quant=False,
        trigger_late=False,
        fused_out=0,
        unit_out=False,
        unit_a_gs=False,
        activation=None,
        num_experts=1,
        base=0,
    ):
        # num_experts = E: a route whose id is outside [0, E) is computed with expert 0; the down GEMM stores a zero
        # row for it (ReduceRoutes then adds 0 times its zeroed weight) or, with fused_out, adds nothing
        self.E = num_experts
        self.base = base  # expert parallelism: local expert = id - base
        self.activation = as_activation(
            activation
        )  # gate/up (act): activation.py; None: the plain SwiGLU
        # unit_out (gate/up; M <= 4 of shapes outside 512 <= I <= 768): the SwiGLU output quantized to NVFP4 with block scales only (a unit second-level
        # scale: the grouped paths' intermediate, gemm.fp4_unit_word), FP4 [routes, N / 4] at out_ptr and the block
        # scales in the direct A layout [routes, N / 128, 512] (one 1-row tile per route) at x_ptr, for a down GEMM
        # that loads its A through the TMA (not x_quant) with unit_a_gs (no row scale). unit_a_gs on the gate/up: the
        # per-tensor input (moe.CuteMoE(per_token_scale=False)), no row scale either
        assert not unit_out or act
        assert not unit_a_gs or not x_quant
        self.unit_out, self.unit_a_gs = unit_out, unit_a_gs
        # fused_out (down; the topk, ``CuteMoE(use_fused_finalize=True)``): the CTA adds its route's output times the
        # route's routing weight (FP32 [routes] at rw_ptr) into the zeroed output row of its token (route //
        # fused_out) instead of storing a routed row (no ReduceRoutes): lane pairs combine two adjacent columns into one
        # ``red.global.add.noftz.bf16x2`` (cute.arch.atomic_add on a bf16x2 vector; a scalar BF16 atomic add compiles to a
        # CAS loop). Outputs: BF16 atomics in a varying order, not deterministic.
        assert not fused_out or not act
        self.fused_out = fused_out
        # PDL: read the route's expert (the caller's id) and issue every stage's weight loads before the wait (the
        # producer triggers after its wait, so the ids it read are final); trigger_late: launch_dependents after the
        # wait instead of before.
        self.trigger_late = trigger_late
        assert n % 128 == 0 and k % bk == 0 and bk == 256
        self.n, self.k, self.bk, self.stages, self.act = n, k, bk, stages, act
        # x_quant: quantize the BF16 input row into every K stage in shared memory (K <= 768; all
        # stages hold the whole row, so stages = K / bk) instead of loading prequantized FP4.
        self.x_quant = x_quant
        assert not x_quant or k <= 768, (
            f"intermediate size {k} > 768: the direct down GEMM stages the whole quantized row and its block scales "
            "in a 512-byte shared-memory scratch (row at 0, scales at 384), which a longer row overflows; "
            "moe.select_schedule uses the grouped path instead"
        )
        assert not x_quant or stages == k // bk
        self.rpr = routes_per_row
        self.kt = k // bk
        assert stages > 1 and stages <= self.kt
        # row layout: thread t holds columns 4t..4t+3 and 512+4t..512+4t+3
        self.x_half = 512
        self.w_bytes = 128 * bk // 2
        self.a_bytes = 16 * bk // 2
        self.sf_bytes = bk // 64 * 512
        self.stage_bytes = (
            self.w_bytes
            + self.sf_bytes
            + (0 if x_quant else self.a_bytes + self.sf_bytes)
        )
        self.tmem_cols = 64
        self.threads = 128

    @cute.jit
    def __call__(
        self,
        a_gs_ptr: cute.Pointer,
        w_gs_ptr: cute.Pointer,
        ids_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        desc_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        rw_ptr: cute.Pointer,
        routes: Int32,
        num_experts: Int32,
        stream,
    ):
        S = self.stages

        @cute.struct
        class Storage:
            sW: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, S * self.w_bytes], 1024
            ]
            sWSF: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, S * self.sf_bytes], 1024
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, S * self.a_bytes], 1024
            ]
            sASF: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, S * self.sf_bytes], 1024
            ]
            ready: cute.struct.MemRange[cutlass.Int64, S]
            mma_done: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]
            pad: cute.struct.MemRange[cutlass.Uint8, 512]  # unused (keeps the layout)
            # x_quant: the barriers are live while the row is quantized, so its scratch is separate
            qscr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, 512 if self.x_quant else 16],
                16,
            ]

        self.storage = Storage
        rows = (routes + self.rpr - 1) // self.rpr
        a_gs = cute.make_tensor(a_gs_ptr, cute.make_layout(rows))
        w_gs = cute.make_tensor(w_gs_ptr, cute.make_layout(num_experts))
        ids = cute.make_tensor(ids_ptr, cute.make_layout(routes))
        descs = cute.make_tensor(desc_ptr, cute.make_layout(64))
        out_cols = self.n // 2 if self.act else self.n
        out = cute.make_tensor(
            out_ptr, cute.make_layout((routes, out_cols), stride=(out_cols, 1))
        )
        x = cute.make_tensor(
            x_ptr, cute.make_layout((rows, self.k), stride=(self.k, 1))
        )
        self.kernel(a_gs, w_gs, ids, out, descs, x, rw_ptr).launch(
            grid=[routes, self.n // 128, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=use_pdl(self, True),
        )

    @cute.jit
    def _load(
        self, st, desc, slot, kk, w_row, row, ready, part: cutlass.Constexpr = "all"
    ):
        """One stage: thread 0 expects the bytes, then warp 0 issues W, W_SF, A, A_SF, each after a
        CTA-wide bar.sync. Everything is predicated on kk < K inside the PTX.
        part "w": expect + weights only; "a": the activation loads only."""
        tidx, _, _ = cute.arch.thread_idx()
        bar = ready + slot
        if cutlass.const_expr(part != "a"):
            if tidx == 0:
                expect_tx_if_lt(bar.toint(), self.stage_bytes, kk, self.k)
        sW = st.sW.data_ptr().toint()
        sWSF = st.sWSF.data_ptr().toint()
        sA = st.sA.data_ptr().toint()
        sASF = st.sASF.data_ptr().toint()
        barv = bar.toint()
        if cutlass.const_expr(part != "a"):
            cute.arch.barrier()
            tma_load_if_lt(
                sW + slot * self.w_bytes, desc, barv, [kk // 2, w_row], tidx, kk, self.k
            )
            cute.arch.barrier()
            tma_load_if_lt(
                sWSF + slot * self.sf_bytes,
                desc + 256,
                barv,
                [0, 0, kk // 64, w_row // 128, 0],
                tidx,
                kk,
                self.k,
            )
        if cutlass.const_expr(not self.x_quant and part != "w"):
            cute.arch.barrier()
            tma_load_if_lt(
                sA + slot * self.a_bytes,
                desc + 128,
                barv,
                [kk // 2, row],
                tidx,
                kk,
                self.k,
            )
            cute.arch.barrier()
            tma_load_if_lt(
                sASF + slot * self.sf_bytes,
                desc + 384,
                barv,
                [0, 0, kk // 64, row, 0],
                tidx,
                kk,
                self.k,
            )

    @cute.kernel
    def kernel(
        self,
        a_gs: cute.Tensor,
        w_gs: cute.Tensor,
        ids: cute.Tensor,
        out: cute.Tensor,
        descs: cute.Tensor,
        x: cute.Tensor,
        rw_ptr: cute.Pointer,
    ):
        S, KT, BK = self.stages, self.kt, self.bk
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        route, nb, _ = cute.arch.block_idx()
        smem = cutlass.memory.SmemAllocator()
        st = smem.allocate(self.storage)
        ready = st.ready.data_ptr()
        mma_done = st.mma_done.data_ptr()
        holding = st.tmem_holding.data_ptr()
        desc = descs.iterator.toint()
        if warp == 0:
            with cute.arch.elect_one():
                for d in cutlass.range_constexpr(4):
                    prefetch_tensormap(desc + 128 * d)
        if warp == 0:
            cute.arch.alloc_tmem(self.tmem_cols, holding)
        cute.arch.barrier()
        tmem_base = st.tmem_holding.get_tensor(cute.make_layout(1))[0]
        cute.arch.barrier()
        if warp == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        row = route // self.rpr
        if cutlass.const_expr(not self.trigger_late):
            cute.arch.griddepcontrol_launch_dependents()
        expert = Int32(0)
        expert = ids[route] - self.base  # the caller's id, skipped outside [0, E)
        expert = Int32(
            cutlass.select_((expert >= 0) & (expert < self.E), expert, Int32(0))
        )
        if warp == 0:
            for s in cutlass.range_constexpr(S):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(ready + s, 1)
            for s in cutlass.range_constexpr(2):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(mma_done + s, 1)
        for s in cutlass.range_constexpr(S):
            self._load(
                st, desc, s, s * BK, expert * self.n + nb * 128, row, ready, part="w"
            )
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.trigger_late):
            cute.arch.griddepcontrol_launch_dependents()
        a_scale = Float32(0.0)
        if cutlass.const_expr(self.x_quant):
            a_scale = self._quantize_row(st, x, row, tidx)
        elif cutlass.const_expr(self.unit_a_gs):
            a_scale = Float32(1.0)
        else:
            a_scale = a_gs[row]
        w_row = expert * self.n + nb * 128
        if cutlass.const_expr(not self.x_quant):
            for s in cutlass.range_constexpr(S):
                self._load(st, desc, s, s * BK, w_row, row, ready, part="a")

        sW = st.sW.data_ptr().toint()
        sWSF = st.sWSF.data_ptr().toint()
        sA = st.sA.data_ptr().toint()
        sASF = st.sASF.data_ptr().toint()
        wsf_col = tmem_base + 16
        asf_col = tmem_base + 32
        idesc = mma_idesc(128, 16)
        KC = BK // 64
        for ki in cutlass.range(KT, unroll=1):
            slot = ki % S
            cute.arch.mbarrier_wait(ready + slot, (ki // S) % 2)
            wsf_desc = Int64(((sWSF + slot * self.sf_bytes) >> 4) & 0x3FFF) | SCALE_DESC
            asf_desc = Int64(((sASF + slot * self.sf_bytes) >> 4) & 0x3FFF) | SCALE_DESC
            offs = [
                (kc * 32) | ((4 * kc) & 7) << 49 for kc in range(KC)
            ]  # 512 B per 64 K: (off >> 4) | (off >> 7 & 7) << 49
            tcgen05_cp_x4_elect(
                tidx,
                [wsf_col + 4 * kc for kc in range(KC)],
                [wsf_desc + o for o in offs],
            )
            cute.arch.barrier()
            tcgen05_cp_x4_elect(
                tidx,
                [asf_col + 4 * kc for kc in range(KC)],
                [asf_desc + o for o in offs],
            )
            if warp == 0:
                w_desc = (
                    Int64(((sW + slot * self.w_bytes) >> 4) & 0x3FFF) | OPERAND_DESC
                )
                a_desc = (
                    Int64(((sA + slot * self.a_bytes) >> 4) & 0x3FFF) | OPERAND_DESC
                )
                tcgen05_mma_nvf4_elect(
                    tmem_base,
                    [w_desc + 2 * kb for kb in range(KC)],
                    [a_desc + 2 * kb for kb in range(KC)],
                    idesc,
                    [wsf_col + 4 * kb for kb in range(KC)],
                    [asf_col + 4 * kb for kb in range(KC)],
                    ki,
                    (mma_done + ki % 2).toint(),
                )
            if ki > 0:
                prev = ki - 1
                cute.arch.mbarrier_wait(mma_done + prev % 2, (prev // 2) % 2)
                nk = (prev + S) * BK
                self._load(st, desc, prev % S, nk, w_row, row, ready)
        last = KT - 1
        cute.arch.mbarrier_wait(mma_done + last % 2, (last // 2) % 2)
        cute.arch.barrier()

        lane = tidx % 32
        acc = tmem_load(tmem_base, 16, 1, 16, tidx)
        v = (acc[0] + Float32(0.0)) * (a_scale * w_gs[expert])
        out_base = out.iterator.toint()
        if cutlass.const_expr(self.act):
            # 128 values in TMEM-lane order through shared memory (rows interleaved per 8: row t is output
            # 8 (t / 16) + t % 8, its up row if t / 8 is odd): thread j < 64 gets gate j, up j as one ld.shared.v2
            cute.arch.store(
                smem_ptr(
                    Float32,
                    sW + 4 * (2 * (8 * (tidx // 16) + tidx % 8) + (tidx // 8) % 2),
                    4,
                ),
                v,
            )
            cute.arch.barrier()
            gu = ld_shared_v2_f32(sW + 8 * (tidx % 64))
            if cutlass.const_expr(self.activation.plain):
                o = (
                    div_full(
                        gu[0],
                        ex2_approx((Float32(0.0) - gu[0]) * Float32(LOG2E))
                        + Float32(1.0),
                    )
                    * gu[1]
                )
            else:
                o = self.activation.apply(gu[0], gu[1])
            half = self.n // 2
            if cutlass.const_expr(self.unit_out):
                # warps 0 / 1: outputs nb * 64 + 32 warp + lane, two 16-value blocks per warp
                if warp < 2:
                    word, sfw = fp4_unit_word(o, lane)
                    fb = nb * 64 + 32 * warp
                    if (lane & 7) == 0:
                        st_global(
                            out_base + Int64(route) * (half // 2) + (fb + lane) // 2,
                            word,
                            Uint32,
                            4,
                        )
                    if (lane & 15) == 0:
                        st_global(
                            x.iterator.toint()
                            + (Int64(route) * (half // 64) + nb) * 512
                            + 2 * warp
                            + lane // 16,
                            cutlass.Uint8(sfw & 0xFF),
                            cutlass.Uint8,
                            1,
                        )
            else:
                n_out = nb * 64 + (tidx & 63)
                ob = o.to(cutlass.BFloat16)
                po = gmem_bf16(out_base + (Int64(route) * half + n_out) * 2)
                if (nb < half // 64) & ((tidx & 64) == 0):
                    po[0] = ob
        elif cutlass.const_expr(self.fused_out):
            n_out = nb * 128 + warp * 32 + lane
            vw = v * cute.arch.load(
                cute.make_ptr(
                    Float32,
                    rw_ptr.toint() + Int64(route) * 4,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                Float32,
            )
            f = cute.make_rmem_tensor((2,), Float32)
            f[0] = vw
            f[1] = cute.arch.shuffle_sync_bfly(
                vw, offset=1
            )  # the odd neighbour's column
            p = cute.make_ptr(
                cutlass.BFloat16,
                out_base + (Int64(route // self.fused_out) * self.n + n_out) * 2,
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            add = (lane & 1) == 0
            e = ids[route] - self.base
            add = add & (e >= 0) & (e < self.E)
            if add:
                cute.arch.atomic_add(p, f.load().to(cutlass.BFloat16).ir_value())
        else:
            n_out = nb * 128 + warp * 32 + lane
            e = ids[route] - self.base
            v = Float32(cutlass.select_((e >= 0) & (e < self.E), v, Float32(0.0)))
            vb = v.to(cutlass.BFloat16)
            po = gmem_bf16(out_base + (Int64(route) * self.n + n_out) * 2)
            n_cols = self.n  # (``self`` cannot be referenced inside a dynamic ``if``)
            if n_out < n_cols:
                po[0] = vb
        cute.arch.barrier()
        if warp == 0:
            cute.arch.dealloc_tmem(
                cute.make_ptr(
                    Float32, tmem_base, cute.AddressSpace.tmem, assumed_align=16
                ),
                self.tmem_cols,
            )

    @cute.jit
    def _quantize_row(self, st, x, row, tidx):
        """Quantize BF16 row ``row`` to per-token NVFP4 and store it into every stage of A / A_SF
        (16 identical FP4 rows, 128 identical scale rows); returns the row's global scale.

        Math and per-thread layout: thread t loads columns 4t..4t+3 and 512+4t..+3, amax over the CTA
        (redux.sync + shared memory), block max over 4 threads (shfl bfly 2, 1), E4M3 scales from
        rcp.approx(amax / 2688), FP4 with cvt.rn.satfinite.e2m1x2; each stage is written with two
        stmatrix.x4 (replicated registers, same addresses). The row is first gathered, in one pass, in a
        512-byte scratch (``qscr``).
        """
        K = self.k
        warp = tidx // 32
        lane = tidx % 32
        scratch = st.qscr.data_ptr().toint()
        sA = st.sA.data_ptr().toint()
        sASF = st.sASF.data_ptr().toint()
        xrow = x.iterator.toint() + Int64(row) * (K * 2)
        lo_t = ld_bf16x4(xrow + tidx * 8).to(Float32)
        hi_b = cute.make_rmem_tensor((4,), cutlass.BFloat16)
        hi_b.fill(0.0)
        if (4 * tidx + self.x_half) < K:
            hi_b.store(ld_bf16x4(xrow + (self.x_half + 4 * tidx) * 2))
        hi_t = hi_b.load().to(Float32)
        lo = [lo_t[i] for i in range(4)]
        hi = [hi_t[i] for i in range(4)]
        a = [abs(v) for v in lo + hi]
        blk_lo = max3_f32(cute.arch.fmax(a[0], a[1]), a[2], a[3])
        tmax = max3_f32(max3_f32(blk_lo, a[4], a[5]), a[6], a[7])
        wmax = cute.arch.warp_redux_sync(tmax, "fmax")
        if lane == 0:
            cute.arch.store(smem_ptr(Float32, scratch + 4 * warp, 4), wmax)
        cute.arch.barrier()
        m = Float32(0.0)
        if tidx < 4:
            m = cute.arch.load(smem_ptr(Float32, scratch + 4 * tidx, 4), Float32)
        m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=2))
        m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1))
        if tidx == 0:
            cute.arch.store(smem_ptr(Float32, scratch, 4), m)
        cute.arch.barrier()
        amax = cute.arch.load(smem_ptr(Float32, scratch, 4), Float32)
        gs = amax * Float32(GS_SCALE)
        enc = sel_if_pos(amax, cute.arch.rcp_approx(gs))
        blk_hi = max3_f32(cute.arch.fmax(a[4], a[5]), a[6], a[7])
        b_lo = cute.arch.fmax(blk_lo, cute.arch.shuffle_sync_bfly(blk_lo, offset=2))
        b_lo = cute.arch.fmax(b_lo, cute.arch.shuffle_sync_bfly(b_lo, offset=1))
        b_hi = cute.arch.fmax(blk_hi, cute.arch.shuffle_sync_bfly(blk_hi, offset=2))
        b_hi = cute.arch.fmax(b_hi, cute.arch.shuffle_sync_bfly(b_hi, offset=1))
        sixth = Float32(1.0 / 6.0)
        sf2, sf_lo, sf_hi = cvt_e4m3x2(enc * (b_lo * sixth), enc * (b_hi * sixth))
        inv_lo = sel_if_pos(b_lo, cute.arch.rcp_approx(gs * sf_lo))
        inv_hi = sel_if_pos(b_hi, cute.arch.rcp_approx(gs * sf_hi))
        q = [
            cvt_e2m1x2(inv_lo * lo[0], inv_lo * lo[1]),
            cvt_e2m1x2(inv_lo * lo[2], inv_lo * lo[3]),
            cvt_e2m1x2(inv_hi * hi[0], inv_hi * hi[1]),
            cvt_e2m1x2(inv_hi * hi[2], inv_hi * hi[3]),
        ]
        cute.arch.barrier()  # everyone has read amax from the scratch
        # Packed row (K / 2 bytes) at scratch, block scales (K / 16) at scratch + 384.
        cute.arch.store(
            smem_ptr(Uint16, scratch + 2 * tidx, 2), Uint16(q[0] | (q[1] << 8))
        )
        if (4 * tidx + self.x_half) < K:
            cute.arch.store(
                smem_ptr(Uint16, scratch + 256 + 2 * tidx, 2),
                Uint16(q[2] | (q[3] << 8)),
            )
        if tidx % 4 == 0:
            cute.arch.store(
                smem_ptr(cutlass.Uint8, scratch + 384 + tidx // 4, 1),
                cutlass.Uint8(sf2 & 0xFF),
            )
            if (4 * tidx + self.x_half) < K:
                cute.arch.store(
                    smem_ptr(cutlass.Uint8, scratch + 384 + 32 + tidx // 4, 1),
                    cutlass.Uint8(sf2 >> 8),
                )
        cute.arch.barrier()
        t = tidx
        a_addr = (
            (((t << 4) & 112) | ((t & 96) << 2))
            ^ ((((t >> 3) & 1) * 576) | ((t & 96) >> 1))
        ) | ((t << 6) & 1024)
        s_addr = ((t << 4) & 112) | ((t & 96) << 2) | ((t << 6) & 1536)
        for s in cutlass.range_constexpr(self.kt):
            v = cute.arch.load(
                smem_ptr(Uint32, scratch + 128 * s + 4 * lane, 4), Uint32
            )
            w = [
                cute.arch.load(
                    smem_ptr(Uint32, scratch + 384 + 16 * s + 4 * i, 4), Uint32
                )
                for i in range(4)
            ]
            stmatrix_x4(sA + s * self.a_bytes + a_addr, [v, v, v, v])
            stmatrix_x4(sASF + s * self.sf_bytes + s_addr, w)
        cute.arch.barrier()
        cute.arch.fence_proxy("async.shared", space="cta")
        return gs
