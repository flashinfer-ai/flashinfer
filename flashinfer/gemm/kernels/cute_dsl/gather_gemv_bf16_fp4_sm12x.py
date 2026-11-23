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

Gathered-row W4A16 GEMV with a scatter epilogue for SM12x.

For every token ``m`` and slot ``j``::

    out[m, col_offset + row_ids[m, j]] = alpha * dot(x[m], dequant(b[row_ids[m, j]]))

``b`` is row-major NVFP4 (``(R, K/2)`` packed E2M1, low nibble first, and
``(R, K/16)`` E4M3 block scales), so any row can be read on its own.  This is
the dynamic-row part of a sparse LM head: a speculative drafter picks a few
thousand vocabulary rows per draft token and needs their exact logits written
into a full-vocabulary logits tensor.

Each half-warp owns one row at a time and streams it GMEM -> registers with
16B non-coherent loads (16 lanes x 16B = 256B per step); no SMEM, no tensor
cores.  The token's activation row is loaded once per CTA into registers:
lane ``l`` always meets the same K slice, ``32 * (16 * i + l) + [0, 32)``.
A ``row_ids`` entry below zero is skipped, so callers can pad.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ....cute_dsl.fp4_common import get_ptr_as_int64, ld_global_nc_v4_u32


@dsl_user_op
def _ld_global_nc_u16(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load 16 bits (zero-extended) through the non-coherent cache."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u16 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _dot32_asm() -> str:
    """PTX for one 16B chunk: 32 E2M1 weights (4 x b32) against 32 bf16
    activations (16 x bf16x2), two E4M3 block scales in the low 16 bits of
    the last operand.  f32 math throughout; four accumulators for ILP."""
    lines = [
        "{",
        ".reg .b8 wb<16>;",
        ".reg .b32 h<16>;",
        ".reg .b16 hl, hh, sp, sj;",
        ".reg .f32 a<4>, wf, xf, s0, s1;",
        ".reg .b32 xt, sh;",
        "mov.f32 a0, 0f00000000;",
        "mov.f32 a1, 0f00000000;",
        "mov.f32 a2, 0f00000000;",
        "mov.f32 a3, 0f00000000;",
    ]
    for j in range(4):
        lines.append(
            f"mov.b32 {{wb{4 * j}, wb{4 * j + 1}, wb{4 * j + 2}, wb{4 * j + 3}}}, ${1 + j};"
        )
    for p in range(16):
        lines.append(f"cvt.rn.f16x2.e2m1x2 h{p}, wb{p};")
    # Byte p of the chunk holds elements 2p (low nibble) and 2p+1; x operand
    # p holds the same two elements as bf16x2 (low half first).  Elements
    # 0..15 share scale 0 (accumulators a0/a1), 16..31 scale 1 (a2/a3).
    for p in range(16):
        acc = (0 if p < 8 else 2) + (p & 1)
        xo = 5 + p
        lines += [
            f"mov.b32 {{hl, hh}}, h{p};",
            "cvt.f32.f16 wf, hl;",
            f"shl.b32 xt, ${xo}, 16;",
            "mov.b32 xf, xt;",
            f"fma.rn.f32 a{acc}, wf, xf, a{acc};",
            "cvt.f32.f16 wf, hh;",
            f"and.b32 xt, ${xo}, 0xFFFF0000;",
            "mov.b32 xf, xt;",
            f"fma.rn.f32 a{acc}, wf, xf, a{acc};",
        ]
    lines += [
        "mov.b32 {sp, sj}, $21;",
        "cvt.rn.f16x2.e4m3x2 sh, sp;",
        "mov.b32 {hl, hh}, sh;",
        "cvt.f32.f16 s0, hl;",
        "cvt.f32.f16 s1, hh;",
        "add.f32 a0, a0, a1;",
        "add.f32 a2, a2, a3;",
        "mul.f32 a0, a0, s0;",
        "fma.rn.f32 $0, a2, s1, a0;",
        "}",
    ]
    return "\n".join(lines)


_DOT32_ASM = _dot32_asm()


@dsl_user_op
def _nvfp4x32_dot_bf16(w, xs, sf: Uint32, *, loc=None, ip=None) -> Float32:
    """Scaled dot product of one 16B NVFP4 chunk with 32 bf16 activations."""
    operands = [Uint32(v).ir_value(loc=loc, ip=ip) for v in w]
    operands += [Uint32(v).ir_value(loc=loc, ip=ip) for v in xs]
    operands.append(Uint32(sf).ir_value(loc=loc, ip=ip))
    return Float32(
        llvm.inline_asm(
            T.f32(),
            operands,
            _DOT32_ASM,
            "=f," + ",".join(["r"] * 21),
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class GatherGemvBf16Fp4Sm12x:
    """``out[m, col_offset + ids[m, j]] = alpha * x[m] . dequant(b[ids[m, j]])``.

    Grid: ``(M, ceil(J / rows_per_cta))``; each CTA serves one token, and
    tokens vary fastest so that CTAs resident together cover the same slot
    range of every token: when the ids are ascending (as unsorted top-k
    returns them) rows shared between tokens are then served from L2.  A
    half-warp handles ``rows_per_hw`` rows, ``unroll`` of them with their
    loads in flight together.

    ``hidden`` is compile-time (it sizes the register-resident activation
    slice) and must be a multiple of 32.
    """

    def __init__(
        self,
        hidden: int,
        rows_per_hw: int = 2,
        unroll: int = 2,
        num_warps: int = 4,
        enable_pdl: bool = True,
    ):
        if hidden % 32 != 0:
            raise ValueError(f"hidden must be a multiple of 32, got {hidden}")
        if rows_per_hw % unroll != 0:
            raise ValueError("rows_per_hw must be a multiple of unroll")
        self.hidden = hidden
        self.chunks = hidden // 32  # 16B packed chunks per row
        self.iters = (self.chunks + 15) // 16  # chunks per lane
        self.full = self.chunks % 16 == 0
        self.rows_per_hw = rows_per_hw
        self.unroll = unroll
        self.num_threads = num_warps * 32
        self.num_hw = num_warps * 2
        self.rows_per_cta = self.num_hw * rows_per_hw
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (M, hidden) bf16
        mB: cute.Tensor,  # (R, hidden/2) uint8, compact rows
        mSF: cute.Tensor,  # (R, hidden/16) uint8 E4M3, compact rows
        mAlpha: cute.Tensor,  # (1,) f32
        mIds: cute.Tensor,  # (M, J) int32/int64
        mOut: cute.Tensor,  # (M, N) bf16/fp16/fp32
        col_offset: Int32,
        stream: cuda.CUstream,
    ):
        num_tokens = cute.size(mIds, mode=[0])
        num_slots = cute.size(mIds, mode=[1])
        self.kernel(mX, mB, mSF, mAlpha, mIds, mOut, col_offset).launch(
            grid=(
                num_tokens,
                (num_slots + self.rows_per_cta - 1) // self.rows_per_cta,
                1,
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mB: cute.Tensor,
        mSF: cute.Tensor,
        mAlpha: cute.Tensor,
        mIds: cute.Tensor,
        mOut: cute.Tensor,
        col_offset: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token, bidx, _ = cute.arch.block_idx()
        hw = tidx // 16
        lane = tidx % 16
        num_slots = cute.size(mIds, mode=[1])
        row_bytes = self.hidden // 2
        sf_bytes = self.hidden // 16

        cute.arch.griddepcontrol_wait()

        # This lane's activation slice: chunk c = 16 * i + lane covers
        # elements [32c, 32c + 32) = 64 bytes = 16 bf16x2 words.
        x_row = get_ptr_as_int64(mX, cute.crd2idx((token, 0), mX.layout))
        xr = []
        for i in cutlass.range_constexpr(self.iters):
            c = Int32(16 * i) + lane
            if cutlass.const_expr(not self.full):
                c = cutlass.min(c, Int32(self.chunks - 1))
            for q in cutlass.range_constexpr(4):
                v = ld_global_nc_v4_u32(x_row + Int64(c) * Int64(64) + Int64(16 * q))
                xr += list(v)

        alpha = mAlpha[0]
        b_base = get_ptr_as_int64(mB, Int32(0))
        sf_base = get_ptr_as_int64(mSF, Int32(0))
        slot0 = bidx * self.rows_per_cta + hw * self.rows_per_hw

        for r0 in cutlass.range_constexpr(0, self.rows_per_hw, self.unroll):
            slots = []
            ids = []
            for u in cutlass.range_constexpr(self.unroll):
                slots.append(slot0 + Int32(r0 + u))
                s = cutlass.min(slots[u], num_slots - 1)
                ids.append(Int64(mIds[token, s]))
            # Issue every load of the unrolled rows before any math.
            ws = []
            sfs = []
            for u in cutlass.range_constexpr(self.unroll):
                rid = cutlass.max(ids[u], Int64(0))
                b_row = b_base + rid * Int64(row_bytes)
                sf_row = sf_base + rid * Int64(sf_bytes)
                for i in cutlass.range_constexpr(self.iters):
                    c = Int32(16 * i) + lane
                    if cutlass.const_expr(not self.full):
                        c = cutlass.min(c, Int32(self.chunks - 1))
                    ws.append(ld_global_nc_v4_u32(b_row + Int64(c) * Int64(16)))
                    sfs.append(_ld_global_nc_u16(sf_row + Int64(c) * Int64(2)))
            for u in cutlass.range_constexpr(self.unroll):
                acc = Float32(0.0)
                for i in cutlass.range_constexpr(self.iters):
                    d = _nvfp4x32_dot_bf16(
                        ws[u * self.iters + i],
                        xr[16 * i : 16 * i + 16],
                        sfs[u * self.iters + i],
                    )
                    if cutlass.const_expr(not self.full and i == self.iters - 1):
                        if Int32(16 * i) + lane >= Int32(self.chunks):
                            d = Float32(0.0)
                    acc = acc + d
                for off in cutlass.range_constexpr(4):
                    acc = acc + cute.arch.shuffle_sync_bfly(acc, 8 >> off)
                if lane == 0:
                    if slots[u] < num_slots:
                        if ids[u] >= Int64(0):
                            col = col_offset + Int32(ids[u])
                            mOut[token, col] = mOut.element_type(acc * alpha)

        cute.arch.griddepcontrol_launch_dependents()
