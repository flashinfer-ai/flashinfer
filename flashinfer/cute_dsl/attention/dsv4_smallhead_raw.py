# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-head DSv4 decode with raw-index gather and CUTLASS warp MMA atoms.

Index translation is fused into the partial-attention kernel. The qualified
schedule uses FP8 QK, BF16 PV and a separate split merge with PDL dependencies.
"""

from functools import cache
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, BFloat16, Float8E4M3FN


class SparseSmallHead:
    def __init__(
        self,
        rows,
        heads,
        width,
        nw,
        nc,
        scale1,
        scale2,
        pv_tile=512,
        fastmath=False,
        pointer_swizzle=False,
        parallel_merge=False,
        vector_pv=False,
        vector_store=False,
        pv_serial_tiles=False,
        copy_warps=4,
        swa_page=32,
        swa_span=364,
        compressed_page=128,
        compressed_span=364,
    ):
        self.rows = rows
        self.heads = heads
        self.width = width
        self.nw = nw
        self.nc = nc
        self.ns = (width + 31) // 32
        self.scale1 = scale1
        self.scale2 = scale2
        self.pv_tile = pv_tile
        self.fastmath = fastmath
        self.pointer_swizzle = pointer_swizzle
        self.parallel_merge = parallel_merge
        self.vector_pv = vector_pv
        self.vector_store = vector_store
        self.pv_serial_tiles = pv_serial_tiles
        self.copy_warps = copy_warps
        self.swa_page = swa_page
        self.swa_span = swa_span
        self.compressed_page = compressed_page
        self.compressed_span = compressed_span
        assert (
            width == 640
            and copy_warps == 16
            and (pv_tile == 512)
            and (not pv_serial_tiles)
            and (not parallel_merge)
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        w: cute.Tensor,
        c: cute.Tensor,
        ids: cute.Tensor,
        lens: cute.Tensor,
        window: cute.Tensor,
        request_ids: cute.Tensor,
        block_table: cute.Tensor,
        token_valid: cute.Tensor,
        sink: cute.Tensor,
        po: cute.Tensor,
        pl: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
        stream: cuda.CUstream,
    ):
        mqk = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaFP8Op(Float8E4M3FN, Float32, (16, 8, 32)),
            (1, 4, 1),
            permutation_mnk=(16, 32, 32),
        )
        mpv = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (1, 4, 1),
            permutation_mnk=(16, self.pv_tile, 16),
        )
        self.partial(
            q,
            w,
            c,
            ids,
            lens,
            window,
            request_ids,
            block_table,
            token_valid,
            po,
            pl,
            mqk,
            mpv,
        ).launch(
            grid=(
                self.rows,
                self.heads // 16,
                self.ns * (1 if self.pv_serial_tiles else 512 // self.pv_tile),
            ),
            block=(self.copy_warps * 32, 1, 1),
            stream=stream,
            use_pdl=True,
        )
        if cutlass.const_expr(self.parallel_merge):
            self.merge_parallel(po, pl, sink, out, lse).launch(
                grid=(self.rows * self.heads, 4, 1),
                block=(128, 1, 1),
                stream=stream,
                use_pdl=True,
            )
        else:
            self.merge(po, pl, sink, out, lse).launch(
                grid=(self.rows * self.heads, 4, 1),
                block=(128, 1, 1),
                stream=stream,
                use_pdl=True,
            )

    @cute.kernel
    def partial(
        self,
        q: cute.Tensor,
        w: cute.Tensor,
        c: cute.Tensor,
        ids: cute.Tensor,
        lens: cute.Tensor,
        window: cute.Tensor,
        request_ids: cute.Tensor,
        block_table: cute.Tensor,
        token_valid: cute.Tensor,
        po: cute.Tensor,
        pl: cute.Tensor,
        mqk: cute.TiledMma,
        mpv: cute.TiledMma,
    ):
        row, hb, part_and_n = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.pv_serial_tiles):
            part = part_and_n
            nb = 0
        else:
            part = part_and_n // (512 // self.pv_tile)
            nb = part_and_n % (512 // self.pv_tile)
        lane = tid % 32
        warp = tid // 32
        allocator = utils.SmemAllocator()
        staged_ids = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), byte_alignment=128
        )
        length_parts = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((16,), stride=(1,)), byte_alignment=128
        )
        local_count = cutlass.Int32(0)
        if ids[row, tid] >= 0 and token_valid[row]:
            local_count = cutlass.Int32(1)
        for k in cutlass.range_constexpr(5):
            local_count += cute.arch.shuffle_sync_bfly(
                local_count, offset=1 << k, mask=-1, mask_and_clamp=31
            )
        if lane == 0:
            length_parts[warp] = local_count
        cute.arch.sync_threads()
        raw_length = cutlass.Int32(128)
        for source_warp in cutlass.range_constexpr(16):
            raw_length += length_parts[source_warp]
        scores = allocator.allocate_tensor(
            Float32, cute.make_layout((16, 32), stride=(32, 1)), byte_alignment=128
        )
        prob = allocator.allocate_tensor(
            BFloat16, cute.make_layout((16, 32), stride=(32, 1)), byte_alignment=128
        )
        qlayout = cute.make_layout((16, 512), stride=(512, 1))
        klayout = cute.make_layout((32, 512), stride=(512, 1))
        swizzle = None
        if cutlass.const_expr(self.pointer_swizzle):
            swizzle = cute.make_swizzle(3, 4, 3)
        sq = allocator.allocate_tensor(
            Float8E4M3FN, qlayout, byte_alignment=128, swizzle=swizzle
        )
        sk = allocator.allocate_tensor(
            Float8E4M3FN, klayout, byte_alignment=128, swizzle=swizzle
        )
        g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Float8E4M3FN, num_bits_per_copy=128
        )
        qrows = min(self.copy_warps * 2, 16)
        qcols = self.copy_warps * 32 // qrows
        qcopy = cute.make_tiled_copy_tv(
            g2s,
            cute.make_layout((qrows, qcols), stride=(qcols, 1)),
            cute.make_layout((1, 16)),
        )
        qt = qcopy.get_slice(tid)
        gq = cute.local_tile(q[row, None, None], (16, 512), (hb, 0))
        cute.copy(qcopy, qt.partition_S(gq), qt.partition_D(sq))
        zero = cute.make_rmem_tensor((16,), Float8E4M3FN)
        zero.fill(0.0)
        store_zero = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float8E4M3FN, num_bits_per_copy=128
        )
        for it in cutlass.range_constexpr(32 // self.copy_warps):
            key = warp + self.copy_warps * it
            pos = part * 32 + key
            ix = cutlass.Int32(-1)
            if pos < self.width:
                ix = self.raw_index(lens, ids, request_ids, block_table, row, pos)
                if lane == 0:
                    staged_ids[key] = ix
            valid = pos < self.width and pos < raw_length and (ix >= 0)
            dst = cute.local_tile(sk[key, None], (16,), (lane,))
            if pos < 128:
                if valid and pos < window[row] and (ix < self.nw):
                    src = cute.local_tile(w[cutlass.Int64(ix), None], (16,), (lane,))
                    cute.copy(g2s, src, dst)
                else:
                    cute.copy(store_zero, zero, dst)
            elif valid and ix < self.nc:
                src = cute.local_tile(c[cutlass.Int64(ix), None], (16,), (lane,))
                cute.copy(g2s, src, dst)
            else:
                cute.copy(store_zero, zero, dst)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if tid < 128:
            self.qk_fragment(
                sq, sk, scores, staged_ids, raw_length, window, mqk, tid, row, part
            )
        cute.arch.sync_threads()
        for it in cutlass.range_constexpr(16 // self.copy_warps):
            h = warp + self.copy_warps * it
            score = scores[h, lane]
            maximum = score
            for k in cutlass.range_constexpr(5):
                maximum = cute.arch.fmax(
                    maximum,
                    cute.arch.shuffle_sync_bfly(
                        maximum, offset=1 << k, mask=-1, mask_and_clamp=31
                    ),
                )
            safe = maximum
            if maximum == Float32(-float("inf")):
                safe = Float32(0.0)
            p = cute.math.exp(score - safe, fastmath=self.fastmath)
            total = p
            for k in cutlass.range_constexpr(5):
                total += cute.arch.shuffle_sync_bfly(
                    total, offset=1 << k, mask=-1, mask_and_clamp=31
                )
            prob[h, lane] = (p / cute.arch.fmax(total, Float32(1e-30))).to(BFloat16)
            if lane == 0 and nb == 0:
                l = Float32(-float("inf"))
                if total > 0:
                    l = cute.math.log(total, fastmath=self.fastmath) + safe
                pl[row, hb * 16 + h, part] = l
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_launch_dependents()
        if tid < 128:
            if cutlass.const_expr(self.pv_serial_tiles):
                for pv_chunk in cutlass.range_constexpr(512 // self.pv_tile):
                    self.pv_fragment(sk, prob, po, mpv, tid, row, hb, part, pv_chunk)
            else:
                tv = mpv.get_slice(tid)
                vt = cute.make_tensor(
                    sk.iterator,
                    cute.composition(
                        sk.layout, cute.make_layout((512, 32), stride=(32, 1))
                    ),
                )
                vt = cute.local_tile(vt, (self.pv_tile, 32), (nb, 0))
                pa = tv.partition_A(prob)
                pb = tv.partition_B(vt)
                pc = tv.partition_C(cute.make_identity_tensor((16, self.pv_tile)))
                a = cute.make_rmem_tensor(pa.shape, BFloat16)
                b = cute.make_rmem_tensor(pb.shape, BFloat16)
                o = cute.make_rmem_tensor(pc.shape, Float32)
                o.fill(0.0)
                cute.copy(
                    cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16), pa, a
                )
                if cutlass.const_expr(self.vector_pv):
                    raw = cute.make_rmem_tensor(pb.shape, Float8E4M3FN)
                    cute.copy(
                        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float8E4M3FN),
                        pb,
                        raw,
                    )
                    b.store(raw.load().to(BFloat16))
                else:
                    for i in cutlass.range_constexpr(cute.size(b)):
                        b[i] = pb[i].to(BFloat16)
                for k in cutlass.range_constexpr(cute.size(a, mode=[2])):
                    cute.gemm(mpv, o, a[None, None, k], b[None, None, k], o)
                if cutlass.const_expr(self.vector_store):
                    output_tile = cute.local_tile(
                        po[row, None, part, None], (16, self.pv_tile), (hb, nb)
                    )
                    cute.autovec_copy(o, tv.partition_C(output_tile))
                else:
                    for i in cutlass.range_constexpr(cute.size(o)):
                        xy = pc[i]
                        po[row, hb * 16 + xy[0], part, nb * self.pv_tile + xy[1]] = o[i]

    @cute.jit
    def qk_fragment(
        self,
        sq: cute.Tensor,
        sk: cute.Tensor,
        scores: cute.Tensor,
        ids: cute.Tensor,
        lens: cutlass.Int32,
        window: cute.Tensor,
        mqk: cute.TiledMma,
        tid: cutlass.Int32,
        row: cutlass.Int32,
        part: cutlass.Int32,
    ):
        th = mqk.get_slice(tid)
        ca = th.partition_A(cute.make_identity_tensor((16, 512)))
        cb = th.partition_B(cute.make_identity_tensor((32, 512)))
        cc = th.partition_C(cute.make_identity_tensor((16, 32)))
        ra = cute.make_rmem_tensor(ca.shape, Float8E4M3FN)
        rb = cute.make_rmem_tensor(cb.shape, Float8E4M3FN)
        acc = cute.make_rmem_tensor(cc.shape, Float32)
        acc.fill(0.0)
        ldm = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            Float8E4M3FN,
        )
        acopy = cute.make_tiled_copy_A(ldm, mqk)
        bcopy = cute.make_tiled_copy_B(ldm, mqk)
        at = acopy.get_slice(tid)
        bt = bcopy.get_slice(tid)
        cute.copy(acopy, at.partition_S(sq), at.retile(ra))
        cute.copy(bcopy, bt.partition_S(sk), bt.retile(rb))
        for k in cutlass.range_constexpr(cute.size(ra, mode=[2])):
            cute.gemm(mqk, acc, ra[None, None, k], rb[None, None, k], acc)
        for i in cutlass.range_constexpr(cute.size(acc)):
            xy = cc[i]
            pos = part * 32 + xy[1]
            ix = cutlass.Int32(-1)
            if pos < self.width:
                ix = ids[xy[1]]
            valid = pos < self.width and pos < lens and (ix >= 0)
            if pos < 128:
                valid = valid and pos < window[row] and (ix < self.nw)
            else:
                valid = valid and ix < self.nc
            score = Float32(-float("inf"))
            if valid:
                score = acc[i] * Float32(self.scale1)
            scores[xy[0], xy[1]] = score

    @cute.jit
    def pv_fragment(
        self,
        sk: cute.Tensor,
        prob: cute.Tensor,
        po: cute.Tensor,
        mpv: cute.TiledMma,
        tid: cutlass.Int32,
        row: cutlass.Int32,
        hb: cutlass.Int32,
        part: cutlass.Int32,
        nb: cutlass.Constexpr,
    ):
        tv = mpv.get_slice(tid)
        vt = cute.make_tensor(
            sk.iterator,
            cute.composition(sk.layout, cute.make_layout((512, 32), stride=(32, 1))),
        )
        vt = cute.local_tile(vt, (self.pv_tile, 32), (nb, 0))
        pa = tv.partition_A(prob)
        pb = tv.partition_B(vt)
        pc = tv.partition_C(cute.make_identity_tensor((16, self.pv_tile)))
        a = cute.make_rmem_tensor(pa.shape, BFloat16)
        b = cute.make_rmem_tensor(pb.shape, BFloat16)
        o = cute.make_rmem_tensor(pc.shape, Float32)
        o.fill(0.0)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16), pa, a)
        if cutlass.const_expr(self.vector_pv):
            raw = cute.make_rmem_tensor(pb.shape, Float8E4M3FN)
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float8E4M3FN), pb, raw
            )
            b.store(raw.load().to(BFloat16))
        else:
            for i in cutlass.range_constexpr(cute.size(b)):
                b[i] = pb[i].to(BFloat16)
        for k in cutlass.range_constexpr(cute.size(a, mode=[2])):
            cute.gemm(mpv, o, a[None, None, k], b[None, None, k], o)
        if cutlass.const_expr(self.vector_store):
            output_tile = cute.local_tile(
                po[row, None, part, None], (16, self.pv_tile), (hb, nb)
            )
            cute.autovec_copy(o, tv.partition_C(output_tile))
        else:
            for i in cutlass.range_constexpr(cute.size(o)):
                xy = pc[i]
                po[row, hb * 16 + xy[0], part, nb * self.pv_tile + xy[1]] = o[i]

    @cute.kernel
    def merge(
        self,
        po: cute.Tensor,
        pl: cute.Tensor,
        sink: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
    ):
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        rh, block, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = rh // self.heads
        head = rh % self.heads
        d = block * 128 + tid
        lane = tid % 32
        partial = Float32(-float("inf"))
        if lane < self.ns:
            partial = pl[row, head, lane]
        top = partial
        for k in cutlass.range_constexpr(5):
            top = cute.arch.fmax(
                top,
                cute.arch.shuffle_sync_bfly(
                    top, offset=1 << k, mask=-1, mask_and_clamp=31
                ),
            )
        maximum = cute.arch.fmax(top, sink[head])
        if maximum == Float32(-float("inf")):
            maximum = Float32(0.0)
        local_weight = cute.math.exp(partial - maximum, fastmath=self.fastmath)
        denominator = local_weight
        for k in cutlass.range_constexpr(5):
            denominator += cute.arch.shuffle_sync_bfly(
                denominator, offset=1 << k, mask=-1, mask_and_clamp=31
            )
        denominator += cute.math.exp(sink[head] - maximum, fastmath=self.fastmath)
        accum = Float32(0.0)
        for s in cutlass.range_constexpr(self.ns):
            weight = cute.arch.shuffle_sync(
                local_weight, offset=s, mask=-1, mask_and_clamp=31
            )
            accum += weight * po[row, head, s, d]
        out[row, 0, head, d] = (
            accum / cute.arch.fmax(denominator, Float32(1e-30)) * Float32(self.scale2)
        ).to(BFloat16)
        if block == 0 and tid == 0:
            safe = top
            if top == Float32(-float("inf")):
                safe = Float32(0.0)
            total = Float32(0.0)
            for s in cutlass.range_constexpr(self.ns):
                total += cute.math.exp(pl[row, head, s] - safe, fastmath=self.fastmath)
            result = Float32(-float("inf"))
            if total > 0:
                result = cute.math.log(total, fastmath=self.fastmath) + safe
            lse[row, 0, head] = result

    @cute.kernel
    def merge_parallel(
        self,
        po: cute.Tensor,
        pl: cute.Tensor,
        sink: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
    ):
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        rh, block, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = rh // self.heads
        head = rh % self.heads
        lane = tid % 32
        warp = tid // 32
        allocator = utils.SmemAllocator()
        partial = allocator.allocate_tensor(
            Float32, cute.make_layout((4, 128), stride=(128, 1)), byte_alignment=128
        )
        value = Float32(-float("inf"))
        if lane < self.ns:
            value = pl[row, head, lane]
        top = value
        for k in cutlass.range_constexpr(5):
            top = cute.arch.fmax(
                top,
                cute.arch.shuffle_sync_bfly(
                    top, offset=1 << k, mask=-1, mask_and_clamp=31
                ),
            )
        maximum = cute.arch.fmax(top, sink[head])
        if maximum == Float32(-float("inf")):
            maximum = Float32(0.0)
        weight = cute.math.exp(value - maximum, fastmath=self.fastmath)
        denominator = weight
        for k in cutlass.range_constexpr(5):
            denominator += cute.arch.shuffle_sync_bfly(
                denominator, offset=1 << k, mask=-1, mask_and_clamp=31
            )
        denominator += cute.math.exp(sink[head] - maximum, fastmath=self.fastmath)
        if block == 0 and warp == 0:
            safe = top
            if top == Float32(-float("inf")):
                safe = Float32(0.0)
            total = cute.math.exp(value - safe, fastmath=self.fastmath)
            for k in cutlass.range_constexpr(5):
                total += cute.arch.shuffle_sync_bfly(
                    total, offset=1 << k, mask=-1, mask_and_clamp=31
                )
            if lane == 0:
                result = Float32(-float("inf"))
                if total > 0:
                    result = cute.math.log(total, fastmath=self.fastmath) + safe
                lse[row, 0, head] = result
        accum = cute.make_rmem_tensor((4,), Float32)
        accum.fill(0.0)
        for it in cutlass.range_constexpr((self.ns + 3) // 4):
            s = warp + 4 * it
            if s < self.ns:
                split_weight = cute.arch.shuffle_sync(
                    weight, offset=s, mask=-1, mask_and_clamp=31
                )
                for j in cutlass.range_constexpr(4):
                    accum[j] += (
                        split_weight * po[row, head, s, block * 128 + j * 32 + lane]
                    )
        for j in cutlass.range_constexpr(4):
            partial[warp, j * 32 + lane] = accum[j]
        cute.arch.sync_threads()
        combined = Float32(0.0)
        for source_warp in cutlass.range_constexpr(4):
            combined += partial[source_warp, tid]
        out[row, 0, head, block * 128 + tid] = (
            combined
            / cute.arch.fmax(denominator, Float32(1e-30))
            * Float32(self.scale2)
        ).to(BFloat16)

    @cute.jit
    def raw_index(
        self,
        swa_ids: cute.Tensor,
        compressed_ids: cute.Tensor,
        request_ids: cute.Tensor,
        block_table: cute.Tensor,
        row: cutlass.Int32,
        pos: cutlass.Int32,
    ):
        result = cutlass.Int32(-1)
        if pos < 128:
            logical = swa_ids[row, pos]
            if logical >= 0:
                result = (
                    logical // self.swa_page * self.swa_span + logical % self.swa_page
                )
        else:
            logical = compressed_ids[row, pos - 128]
            if logical >= 0:
                request = request_ids[row]
                page = block_table[request, logical // self.compressed_page]
                if page >= 0:
                    result = (
                        page * self.compressed_span + logical % self.compressed_page
                    )
        return result


@cache
def compile_raw(
    rows,
    heads,
    nw,
    nc,
    scale1,
    scale2,
    swa_page,
    swa_span,
    compressed_page,
    compressed_span,
    table_columns,
    device,
):
    f = cute.runtime.make_fake_compact_tensor
    specs = [
        (Float8E4M3FN, (rows, heads, 512)),
        (Float8E4M3FN, (nw, 512)),
        (Float8E4M3FN, (nc, 512)),
        (cutlass.Int32, (rows, 512)),
        (cutlass.Int32, (rows, 128)),
        (cutlass.Int32, (rows,)),
        (cutlass.Int32, (rows,)),
        (cutlass.Int32, (rows, table_columns)),
        (cutlass.Boolean, (rows,)),
        (Float32, (heads,)),
        (Float32, (rows, heads, 20, 512)),
        (Float32, (rows, heads, 20)),
        (BFloat16, (rows, 1, heads, 512)),
        (Float32, (rows, 1, heads)),
    ]
    tensors = [
        f(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=16,
        )
        for (dtype, shape) in specs
    ]
    kernel = SparseSmallHead(
        rows,
        heads,
        640,
        nw,
        nc,
        scale1,
        scale2,
        pointer_swizzle=True,
        fastmath=True,
        vector_store=True,
        copy_warps=16,
        swa_page=swa_page,
        swa_span=swa_span,
        compressed_page=compressed_page,
        compressed_span=compressed_span,
    )
    return cute.compile(
        kernel,
        *tensors,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 2",
    )
