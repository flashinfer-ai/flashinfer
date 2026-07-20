# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/tiny_decode.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""MoETinyDecodeKernelBackend — tiny-decode (M<=4) W4A8-MX MoE for SM120.

Reads the N256/K128 in-place-REPACKED weight storage (the "rp" layout produced
by _logical_weight_to_w4a8_rp_inplace) directly.

Consumes the N256/K128 in-place-repacked FP4 weights and e8m0 sfb grids
directly (inverse mappings verified in tests/test_w4a8_rp_inverse_mapping.py),
with BF16 activations (no input quantization) and f32 accumulation. Two plain
(non-cooperative) launches: FC1 dots gate+up rows into an fp32 intermediate,
FC2 applies SiLU inline and folds router-weighted partials into the bf16
output via scatter-add. The wrapper zeroes the intermediate and output first;
there are no grid barriers and no CTA co-residency assumptions, so the
kernels are safe on busy serving streams.

Thread mapping (256 threads/CTA), the core of the rp coalescing story: one rp
(nt, kt) tile is 4096 contiguous int32 words whose flat index decomposes as
``k32<<10 | n8c<<7 | r8<<4 | cgrp<<2 | n8i``. Thread t = (n8c=t>>5, r8=(t>>2)&7,
cgrp=t&3) issues one 16 B ``ld.global.nc.v4`` per k32 covering n8i=0..3 — four
8-apart logical rows at one k window — so a warp (fixed n8c) covers a fully
coalesced 512 B run per k32, and the 4-lane cgrp butterfly folds the k dim.

The prep rotation normalizes both declared w13 layouts to the same rp tile
order (tiles [0, N/256) hold the "up" rows, [N/256, 2N/256) the "gate" rows of
the same channels), so FC1 stores inter rows through ``(p + n) % 2n`` and FC2
reads gate at [0, n), up at [n, 2n) unconditionally.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cutlass_dsl import Int32, Int64

from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream, make_ptr
from flashinfer.experimental.sm12x._lib.intrinsics import (
    cvt_bf16x2_to_f16x2,
    cvt_e8m0x4_to_f32x4,
    fp4_dot4_sum_f32acc,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    pack_f32x2_to_f16x2,
    red_add_global_f32,
    scatter_add_bf16x2,
    warp_reduce,
)

_BLOCK_THREADS = 256
_FC1_KT_PER_TASK = 4
_FC2_KT_PER_TASK = 2


class MoETinyDecodeKernelBackend:
    """Tiny-M (decode) W4A8-MX kernel reading the repacked weight layout."""

    def __init__(
        self,
        *,
        activation: str = "silu",
        w13_layout: str = "w31",
        compile_time_phase: int = 1,
    ):
        if activation != "silu":
            raise ValueError(f"tiny_decode supports silu only, got {activation!r}")
        if int(compile_time_phase) not in (1, 2):
            raise ValueError(f"unsupported tiny_decode phase {compile_time_phase!r}")
        self.compile_time_phase = int(compile_time_phase)
        self.activation = activation
        self.w13_layout = w13_layout
        self._cfg_key = None
        self._c = None
        self.grid_x = 0

    def configure(
        self,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        del device
        # Weights arrive in the ceil-tiled rp layout: partial 256-row / 128-col
        # tiles are stored zero-filled, so any n % 32 shard works. FC1's
        # zero rows contribute exact +0.0 through the rotated scatter-add;
        # FC2 bounds its intermediate loads by k2_tail_g32 (see the kernel).
        if k % 256 != 0 or n % 32 != 0:
            raise ValueError("tiny_decode requires k % 256 == 0 and n % 32 == 0")
        if m < 1 or m > 4:
            raise ValueError("tiny_decode supports 1 <= m <= 4")
        rt = m * num_topk
        kt13 = k // 128
        kt2 = -(-n // 128)
        nt13 = -(-(2 * n) // 256)
        if kt13 % _FC1_KT_PER_TASK != 0:
            raise ValueError("tiny_decode k-tile counts not divisible by task sizes")
        # FC2 walks the intermediate dim in per-task groups of K tiles. Odd
        # tile counts (e.g. n=384 -> 3 tiles from GLM 2048/TP6 padded shards)
        # drop to one tile per task; partials are scatter-added, so the task
        # split does not change results.
        fc2_kt_per_task = _FC2_KT_PER_TASK if kt2 % _FC2_KT_PER_TASK == 0 else 1
        cfg = dict(
            m=m,
            k=k,
            n=n,
            two_n=2 * n,
            num_topk=num_topk,
            weight_E=weight_E,
            rt=rt,
            nt13=nt13,
            kt13=kt13,
            fc1_ktg=kt13 // _FC1_KT_PER_TASK,
            nt2=k // 256,
            kt2=kt2,
            # Index of the partial FC2 K tile (== kt2 when none) and how many
            # 32-value groups of it are logically valid.
            kt2_full=n // 128,
            k2_tail_g32=(n % 128) // 32,
            # Half-aligned rp storage: each gated half is zero-padded to a
            # 128-row boundary (up at [0, n_pad128), gate at [n_pad128, ...)).
            n_pad128=kt2 * 128,
            fc2_kt_per_task=fc2_kt_per_task,
            fc2_ktg=kt2 // fc2_kt_per_task,
            w13_words=nt13 * kt13 * 4096,
            w2_words=(k // 256) * kt2 * 4096,
            sfb13_bytes=nt13 * kt13 * 1024,
            sfb2_bytes=(k // 256) * kt2 * 1024,
            fc1_tasks=rt * nt13 * (kt13 // _FC1_KT_PER_TASK),
            fc2_tasks=rt * (k // 256) * (kt2 // fc2_kt_per_task),
        )
        self._c = cfg
        self._cfg_key = tuple(sorted(cfg.items()))
        self.grid_x = (
            cfg["fc1_tasks"] if self.compile_time_phase == 1 else cfg["fc2_tasks"]
        )

    @property
    def __cache_key__(self):
        return (
            self.activation,
            self.w13_layout,
            self.compile_time_phase,
            self._cfg_key,
            self.grid_x,
        )

    @cute.jit
    def _row_block_dot(
        self,
        tile_word: Int64,
        srow: Int64,
        n8c: Int32,
        r8: Int32,
        cgrp: Int32,
        x0_0: cutlass.Uint32,
        x1_0: cutlass.Uint32,
        x2_0: cutlass.Uint32,
        x3_0: cutlass.Uint32,
        x0_1: cutlass.Uint32,
        x1_1: cutlass.Uint32,
        x2_1: cutlass.Uint32,
        x3_1: cutlass.Uint32,
        x0_2: cutlass.Uint32,
        x1_2: cutlass.Uint32,
        x2_2: cutlass.Uint32,
        x3_2: cutlass.Uint32,
        x0_3: cutlass.Uint32,
        x1_3: cutlass.Uint32,
        x2_3: cutlass.Uint32,
        x3_3: cutlass.Uint32,
    ):
        """Dot 4 n8i rows (v=0..3) x 128 k window against packed activations.

        Activation f16x2 quads are per k32 (x*_<k32>). Returns 4 row partials.
        """
        word_off = Int64(n8c * Int32(128) + r8 * Int32(16) + cgrp * Int32(4)) * Int64(4)
        acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
        xq = (
            (x0_0, x1_0, x2_0, x3_0),
            (x0_1, x1_1, x2_1, x3_1),
            (x0_2, x1_2, x2_2, x3_2),
            (x0_3, x1_3, x2_3, x3_3),
        )
        for v in cutlass.range_constexpr(4):
            sv = ld_global_nc_u32(srow + Int64(v * 32))
            sk = cvt_e8m0x4_to_f32x4(sv)
            accv = Float32(0.0)
            for k32 in cutlass.range_constexpr(4):
                words = ld_global_nc_v4_u32(tile_word + Int64(k32 * 4096) + word_off)
                x0, x1, x2, x3 = xq[k32]
                accv += sk[k32] * fp4_dot4_sum_f32acc(words[v], x0, x1, x2, x3)
            acc[v] += accv
        return acc[0], acc[1], acc[2], acc[3]

    @cute.kernel
    def kernel(
        self,
        a_input: cute.Tensor,  # bf16 [m*k]
        w13: cute.Tensor,  # u8 rp bytes, expert-major
        sfb13: cute.Tensor,  # u8 sfb bytes, expert-major
        inter: cute.Tensor,  # f32 [rt * 2n] (pre-zeroed for phase 1)
        w2: cute.Tensor,  # u8 rp bytes
        sfb2: cute.Tensor,  # u8 sfb bytes
        topk_ids: cute.Tensor,  # i32 [rt]
        topk_weights: cute.Tensor,  # f32 [rt]
        out: cute.Tensor,  # bf16 [m*k] (pre-zeroed for phase 2)
    ):
        c = self._c
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n8c = tidx // Int32(32)
        r8 = (tidx // Int32(4)) % Int32(8)
        cgrp = tidx % Int32(4)

        x_base = get_ptr_as_int64(a_input, Int32(0))
        w13_base = get_ptr_as_int64(w13, Int32(0))
        sfb13_base = get_ptr_as_int64(sfb13, Int32(0))
        inter_base = get_ptr_as_int64(inter, Int32(0))
        w2_base = get_ptr_as_int64(w2, Int32(0))
        sfb2_base = get_ptr_as_int64(sfb2, Int32(0))
        out_base = get_ptr_as_int64(out, Int32(0))

        if cutlass.const_expr(self.compile_time_phase == 1):
            # ---- FC1: block = (rt, nt13, ktg) ----
            fc1_per_rt = Int32(c["nt13"] * c["fc1_ktg"])
            rt_idx = bidx // fc1_per_rt
            rem = bidx % fc1_per_rt
            nt = rem // Int32(c["fc1_ktg"])
            ktg = rem % Int32(c["fc1_ktg"])
            eid = Int64(Int32(topk_ids[rt_idx]))
            tok = rt_idx // Int32(c["num_topk"])
            we_base = w13_base + eid * Int64(c["w13_words"] * 4)
            se_base = sfb13_base + eid * Int64(c["sfb13_bytes"])
            xt_base = x_base + Int64(tok) * Int64(c["k"] * 2)
            srow_base = se_base + Int64(r8 * Int32(4) + n8c * Int32(128))

            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            acc2 = Float32(0.0)
            acc3 = Float32(0.0)
            for kt_i in cutlass.range_constexpr(_FC1_KT_PER_TASK):
                kt = ktg * Int32(_FC1_KT_PER_TASK) + Int32(kt_i)
                col_tile = nt * Int32(c["kt13"]) + kt
                tile_word = we_base + Int64(col_tile) * Int64(4096 * 4)
                srow = srow_base + Int64(col_tile) * Int64(1024)
                xs = []
                for k32 in cutlass.range_constexpr(4):
                    ax = xt_base + (
                        Int64(kt * Int32(128) + cgrp * Int32(8)) + Int64(k32 * 32)
                    ) * Int64(2)
                    bq = ld_global_nc_v4_u32(ax)
                    xs.append(
                        (
                            cvt_bf16x2_to_f16x2(bq[0]),
                            cvt_bf16x2_to_f16x2(bq[1]),
                            cvt_bf16x2_to_f16x2(bq[2]),
                            cvt_bf16x2_to_f16x2(bq[3]),
                        )
                    )
                d0, d1, d2, d3 = self._row_block_dot(
                    tile_word,
                    srow,
                    n8c,
                    r8,
                    cgrp,
                    xs[0][0],
                    xs[0][1],
                    xs[0][2],
                    xs[0][3],
                    xs[1][0],
                    xs[1][1],
                    xs[1][2],
                    xs[1][3],
                    xs[2][0],
                    xs[2][1],
                    xs[2][2],
                    xs[2][3],
                    xs[3][0],
                    xs[3][1],
                    xs[3][2],
                    xs[3][3],
                )
                acc0 += d0
                acc1 += d1
                acc2 += d2
                acc3 += d3
            acc0 = warp_reduce(acc0, lambda a, b: a + b, width=4)
            acc1 = warp_reduce(acc1, lambda a, b: a + b, width=4)
            acc2 = warp_reduce(acc2, lambda a, b: a + b, width=4)
            acc3 = warp_reduce(acc3, lambda a, b: a + b, width=4)
            if cgrp == Int32(0):
                ibase_rt = inter_base + Int64(rt_idx) * Int64(c["two_n"] * 4)
                accs = (acc0, acc1, acc2, acc3)
                for v in cutlass.range_constexpr(4):
                    p = nt * Int32(256) + n8c * Int32(32) + Int32(v * 8) + r8
                    if cutlass.const_expr(c["k2_tail_g32"] > 0):
                        # Half-aligned tail storage: up rows at
                        # [0, n_pad128), gate rows at [n_pad128, 2*n_pad128),
                        # each half zero-padded past the logical n. Skip the
                        # dead rows' stores (their accs are exact zeros, but
                        # the mapped inter rows would go out of bounds).
                        is_gate = p >= Int32(c["n_pad128"])
                        half_dst = p
                        r_log = p + Int32(c["n"])
                        if is_gate:
                            half_dst = p - Int32(c["n_pad128"])
                            r_log = half_dst
                        if half_dst < Int32(c["n"]):
                            red_add_global_f32(
                                ibase_rt + Int64(r_log) * Int64(4), accs[v]
                            )
                    else:
                        r_log = p + Int32(c["n"])
                        if r_log >= Int32(c["two_n"]):
                            r_log -= Int32(c["two_n"])
                        red_add_global_f32(ibase_rt + Int64(r_log) * Int64(4), accs[v])

        if cutlass.const_expr(self.compile_time_phase == 2):
            # ---- FC2: block = (rt, nt2, ktg2) ----
            fc2_per_rt = Int32(c["nt2"] * c["fc2_ktg"])
            rt_idx = bidx // fc2_per_rt
            rem = bidx % fc2_per_rt
            nt = rem // Int32(c["fc2_ktg"])
            ktg = rem % Int32(c["fc2_ktg"])
            eid = Int64(Int32(topk_ids[rt_idx]))
            tok = rt_idx // Int32(c["num_topk"])
            rw = Float32(topk_weights[rt_idx])
            we_base = w2_base + eid * Int64(c["w2_words"] * 4)
            se_base = sfb2_base + eid * Int64(c["sfb2_bytes"])
            srow_base = se_base + Int64(r8 * Int32(4) + n8c * Int32(128))
            ibase = rt_idx * Int32(c["two_n"])

            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            acc2 = Float32(0.0)
            acc3 = Float32(0.0)
            for kt_i in cutlass.range_constexpr(c["fc2_kt_per_task"]):
                kt = ktg * Int32(c["fc2_kt_per_task"]) + Int32(kt_i)
                col_tile = nt * Int32(c["kt2"]) + kt
                tile_word = we_base + Int64(col_tile) * Int64(4096 * 4)
                srow = srow_base + Int64(col_tile) * Int64(1024)
                xs = []
                if cutlass.const_expr(c["k2_tail_g32"] > 0):
                    # Ceil-tiled FC2 K tail: 32-groups past the logical n get
                    # zero activations (their rp weights/scales are zero-filled
                    # too), which also keeps the intermediate loads in bounds.
                    kt_valid_g32 = Int32(4)
                    if kt == Int32(c["kt2_full"]):
                        kt_valid_g32 = Int32(c["k2_tail_g32"])
                    for k32 in cutlass.range_constexpr(4):
                        ich = (
                            ibase + kt * Int32(128) + cgrp * Int32(8) + Int32(k32 * 32)
                        )
                        quads = []
                        for jp in cutlass.range_constexpr(4):
                            a0 = Float32(0.0)
                            a1 = Float32(0.0)
                            if Int32(k32) < kt_valid_g32:
                                g0 = Float32(inter[ich + Int32(2 * jp)])
                                g1 = Float32(inter[ich + Int32(2 * jp + 1)])
                                u0 = Float32(inter[ich + Int32(c["n"]) + Int32(2 * jp)])
                                u1 = Float32(
                                    inter[ich + Int32(c["n"]) + Int32(2 * jp + 1)]
                                )
                                s0 = Float32(1.0) / (
                                    Float32(1.0) + cute.math.exp(-g0, fastmath=False)
                                )
                                s1 = Float32(1.0) / (
                                    Float32(1.0) + cute.math.exp(-g1, fastmath=False)
                                )
                                a0 = s0 * g0 * u0 * rw
                                a1 = s1 * g1 * u1 * rw
                            quads.append(pack_f32x2_to_f16x2(a0, a1))
                        xs.append((quads[0], quads[1], quads[2], quads[3]))
                else:
                    for k32 in cutlass.range_constexpr(4):
                        ich = (
                            ibase + kt * Int32(128) + cgrp * Int32(8) + Int32(k32 * 32)
                        )
                        quads = []
                        for jp in cutlass.range_constexpr(4):
                            g0 = Float32(inter[ich + Int32(2 * jp)])
                            g1 = Float32(inter[ich + Int32(2 * jp + 1)])
                            u0 = Float32(inter[ich + Int32(c["n"]) + Int32(2 * jp)])
                            u1 = Float32(inter[ich + Int32(c["n"]) + Int32(2 * jp + 1)])
                            s0 = Float32(1.0) / (
                                Float32(1.0) + cute.math.exp(-g0, fastmath=False)
                            )
                            s1 = Float32(1.0) / (
                                Float32(1.0) + cute.math.exp(-g1, fastmath=False)
                            )
                            a0 = s0 * g0 * u0 * rw
                            a1 = s1 * g1 * u1 * rw
                            quads.append(pack_f32x2_to_f16x2(a0, a1))
                        xs.append((quads[0], quads[1], quads[2], quads[3]))
                d0, d1, d2, d3 = self._row_block_dot(
                    tile_word,
                    srow,
                    n8c,
                    r8,
                    cgrp,
                    xs[0][0],
                    xs[0][1],
                    xs[0][2],
                    xs[0][3],
                    xs[1][0],
                    xs[1][1],
                    xs[1][2],
                    xs[1][3],
                    xs[2][0],
                    xs[2][1],
                    xs[2][2],
                    xs[2][3],
                    xs[3][0],
                    xs[3][1],
                    xs[3][2],
                    xs[3][3],
                )
                acc0 += d0
                acc1 += d1
                acc2 += d2
                acc3 += d3
            acc0 = warp_reduce(acc0, lambda a, b: a + b, width=4)
            acc1 = warp_reduce(acc1, lambda a, b: a + b, width=4)
            acc2 = warp_reduce(acc2, lambda a, b: a + b, width=4)
            acc3 = warp_reduce(acc3, lambda a, b: a + b, width=4)
            # pair consecutive output rows: partner lane differs in r8 bit0 (lane^4)
            o0 = cute.arch.shuffle_sync_bfly(acc0, offset=4)
            o1 = cute.arch.shuffle_sync_bfly(acc1, offset=4)
            o2 = cute.arch.shuffle_sync_bfly(acc2, offset=4)
            o3 = cute.arch.shuffle_sync_bfly(acc3, offset=4)
            if cgrp == Int32(0):
                if (r8 % Int32(2)) == Int32(0):
                    ob = out_base + Int64(tok) * Int64(c["k"] * 2)
                    accs = (acc0, acc1, acc2, acc3)
                    others = (o0, o1, o2, o3)
                    for v in cutlass.range_constexpr(4):
                        p2 = nt * Int32(256) + n8c * Int32(32) + Int32(v * 8) + r8
                        scatter_add_bf16x2(
                            ob + Int64(p2) * Int64(2), accs[v], others[v]
                        )

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w13_ptr: cute.Pointer,
        sfb13_ptr: cute.Pointer,
        inter_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        tid_ptr: cute.Pointer,
        tw_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        stream,
    ):
        c = self._c
        a_input = cute.make_tensor(x_ptr, cute.make_layout(Int32(c["m"] * c["k"])))
        w13 = cute.make_tensor(
            w13_ptr, cute.make_layout(Int64(c["weight_E"] * c["w13_words"] * 4))
        )
        sfb13 = cute.make_tensor(
            sfb13_ptr, cute.make_layout(Int64(c["weight_E"] * c["sfb13_bytes"]))
        )
        inter = cute.make_tensor(
            inter_ptr, cute.make_layout(Int32(c["rt"] * c["two_n"]))
        )
        w2 = cute.make_tensor(
            w2_ptr, cute.make_layout(Int64(c["weight_E"] * c["w2_words"] * 4))
        )
        sfb2 = cute.make_tensor(
            sfb2_ptr, cute.make_layout(Int64(c["weight_E"] * c["sfb2_bytes"]))
        )
        topk_ids = cute.make_tensor(tid_ptr, cute.make_layout(Int32(c["rt"])))
        topk_weights = cute.make_tensor(tw_ptr, cute.make_layout(Int32(c["rt"])))
        out = cute.make_tensor(out_ptr, cute.make_layout(Int32(c["m"] * c["k"])))

        self.kernel(
            a_input,
            w13,
            sfb13,
            inter,
            w2,
            sfb2,
            topk_ids,
            topk_weights,
            out,
        ).launch(
            grid=(Int32(self.grid_x), Int32(1), Int32(1)),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    @staticmethod
    def launch(
        compiled_fc1,
        compiled_fc2,
        *,
        x: torch.Tensor,
        w13_rp: torch.Tensor,
        sfb13: torch.Tensor,
        inter_fp32: torch.Tensor,
        w2_rp: torch.Tensor,
        sfb2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor,
    ):
        def ptr(dt, t, align=16):
            return make_ptr(
                dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
            )

        inter_fp32.zero_()
        out.zero_()
        stream = current_cuda_stream()
        args = (
            ptr(cutlass.BFloat16, x),
            ptr(cutlass.Uint8, w13_rp.view(torch.uint8)),
            ptr(cutlass.Uint8, sfb13.view(torch.uint8)),
            ptr(cutlass.Float32, inter_fp32),
            ptr(cutlass.Uint8, w2_rp.view(torch.uint8)),
            ptr(cutlass.Uint8, sfb2.view(torch.uint8)),
            ptr(cutlass.Int32, topk_ids, 4),
            ptr(cutlass.Float32, topk_weights, 4),
            ptr(cutlass.BFloat16, out),
            stream,
        )
        compiled_fc1(*args)
        compiled_fc2(*args)


class MoETinyDecodeKernelBackendPhase1(MoETinyDecodeKernelBackend):
    """Phase-1 compile identity for exact launch/resource attribution."""

    def __init__(self, *, activation: str = "silu", w13_layout: str = "w31"):
        super().__init__(
            activation=activation,
            w13_layout=w13_layout,
            compile_time_phase=1,
        )


class MoETinyDecodeKernelBackendPhase2(MoETinyDecodeKernelBackend):
    """Phase-2 compile identity for exact launch/resource attribution."""

    def __init__(self, *, activation: str = "silu", w13_layout: str = "w31"):
        super().__init__(
            activation=activation,
            w13_layout=w13_layout,
            compile_time_phase=2,
        )
