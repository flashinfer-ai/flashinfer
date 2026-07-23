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

Prototype streaming W4A16 GEMV for SM12x: bf16 activations x NVFP4 weights at
tiny M.  Weights stream GMEM -> registers with no SMEM staging and no tensor
cores (each weight byte is used exactly once, so staging is pure overhead);
one output row per warp, 32 dim-parallel lanes, warp-shuffle reduction.  The
schedule mirrors MsaProxyScoreDecodeStreamSm12x, whose pack_q measurements
bound where this shape stops paying: expect wins at M <= 4, parity-at-best by
M ~ 8 as the FMA phase starts scaling with M while the loads do not.
"""

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ....fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_fp4_helpers import (
    SF_VEC_SIZE,
    packed_dequant_e2m1x4_to_bfloat2x2,
)

# The register dequant emits raw-bit bf16 = true_value * 2^-126 (no bias
# multiply), and the E4M3 scale-byte bit-place decode below emits
# true_scale * 2^-120.  Folding both ratios into one constant (2^246) is not
# f32-representable, so each is unfolded separately per scale block.
_TWO_POW_126 = 2.0**126
_TWO_POW_120 = 2.0**120


@dsl_user_op
def _bf16x2_to_f32x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Widen one packed bf16x2 register into two f32 values exactly."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.bf16 $0, lo;
            cvt.f32.bf16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(lo), Float32(hi)


@dsl_user_op
def _u32_bitcast_f32(bits: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(bits).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class GemvBf16Fp4Sm12x:
    """Streaming bf16 x nvfp4 GEMV, one output row per warp.

    Consumes the *canonical* nvfp4 weight layout directly -- ``(N, K/2)``
    uint8 viewed as ``(N, K/8)`` int32 (two codes per byte, low nibble =
    even K) plus linear ``(N, K/16)`` E4M3 scale bytes -- so unlike the
    MMA kernel no weight repack pass is needed.

    Per iteration a lane owns 32 consecutive K codes (one 16B load, two
    scale blocks); the warp's 32 lanes cover 1024 K.  Dequant reuses the
    e2m1 bit-place primitive from the SM12x MoE W4A16 kernel and the dot
    products accumulate in f32, so the only bf16 rounding on the B side is
    the exact e2m1 magnitude set -- slightly *tighter* than the MMA
    kernel's bf16 operand path.

    Constraints (prototype): K % 1024 == 0, M in [1, 4] (compile-time; Q
    and the accumulators live in registers per lane -- the same footprint
    argument as the MSA stream kernel's group_size cap), non-negative
    scales (guaranteed by nvfp4 quantization; the shipped MMA kernel's
    S0E5M3 reformat already bakes in the same assumption).
    """

    _NUM_WARPS = 8  # one output row per warp -> 8 rows per CTA
    _I32_PER_LANE = 4  # one 16B weight load = 4 int32 = 32 fp4 codes
    _CODES_PER_I32 = 8  # fp4 codes per packed int32
    # Scale blocks touched per lane per iteration (2) and int32s covered by
    # one scale block (2).
    _SF_PER_LANE = _I32_PER_LANE * _CODES_PER_I32 // SF_VEC_SIZE
    _I32_PER_SF = SF_VEC_SIZE // _CODES_PER_I32

    def __init__(self, m: int):
        if not 1 <= m <= 4:
            raise ValueError("m must be in [1, 4]")
        self._m = m
        self._num_threads = self._NUM_WARPS * 32
        # K codes consumed per warp per iteration (= 1024).
        self._k_per_iter = 32 * self._I32_PER_LANE * self._CODES_PER_I32

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M, K) bf16
        mB: cute.Tensor,  # (N, K/8) int32: canonical packed fp4
        mSF: cute.Tensor,  # (N, K/16) uint8 E4M3 per-block scales, linear
        mC: cute.Tensor,  # (M, N) bf16
        mAlpha: cute.Tensor,  # (1,) f32
        n: cutlass.Int32,
        num_iters: cutlass.Int32,  # K // self._k_per_iter (= K // 1024)
        stream: cuda.CUstream,
    ):
        self.kernel(mA, mB, mSF, mC, mAlpha, n, num_iters).launch(
            grid=((n + self._NUM_WARPS - 1) // self._NUM_WARPS, 1, 1),
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mSF: cute.Tensor,
        mC: cute.Tensor,
        mAlpha: cute.Tensor,
        n: cutlass.Int32,
        num_iters: cutlass.Int32,
    ):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * self._NUM_WARPS + warp
        M = self._m

        # No SMEM and no CTA-wide sync anywhere below, so a tail warp with no
        # row can retire early without deadlocking its CTA.
        if row < n:
            alpha = mAlpha[0]
            mB_row = mB[row, None]
            mSF_row = mSF[row, None]

            acc = cute.make_rmem_tensor((M,), Float32)
            acc.fill(0.0)
            wfrag = cute.make_rmem_tensor(
                cute.make_layout(self._I32_PER_LANE), Int32
            )
            afrags = [
                [
                    cute.make_rmem_tensor(
                        cute.make_layout(self._CODES_PER_I32), mA.element_type
                    )
                    for _ in range(self._I32_PER_LANE)
                ]
                for _ in range(M)
            ]

            for it in cutlass.range(num_iters):
                base_i32 = it * (32 * self._I32_PER_LANE) + lane * self._I32_PER_LANE

                # Issue every load of the iteration before any consumption;
                # cross-iteration overlap comes from warp parallelism (this
                # shape runs tens of warps per SM), not a SMEM pipeline.
                w_chunk = cute.local_tile(
                    mB_row, (self._I32_PER_LANE,), (it * 32 + lane,)
                )
                cute.autovec_copy(w_chunk, wfrag)
                for g in cutlass.range_constexpr(M):
                    for j in cutlass.range_constexpr(self._I32_PER_LANE):
                        a_chunk = cute.local_tile(
                            mA[g, None], (self._CODES_PER_I32,), (base_i32 + j,)
                        )
                        cute.autovec_copy(a_chunk, afrags[g][j])
                sblk = it * (32 * self._SF_PER_LANE) + lane * self._SF_PER_LANE

                for h in cutlass.range_constexpr(self._SF_PER_LANE):
                    # E4M3 bit-place decode: exp+mant land in the f32 fields
                    # at ratio 2^-120 (sign dropped: nvfp4 scales are
                    # non-negative).
                    sb = Uint32(mSF_row[sblk + h])
                    s_true = _u32_bitcast_f32((sb & 0x7F) << 20) * _TWO_POW_120
                    dot = [Float32(0.0) for _ in range(M)]
                    for j in cutlass.range_constexpr(self._I32_PER_SF):
                        wu = Uint32(wfrag[h * self._I32_PER_SF + j])
                        # Nibble order per int32 (little-endian codes k0..k7):
                        # the primitive returns (shifted, unshifted) pairs, so
                        # dequant(w) yields (k2,k6),(k3,k7) and dequant(w<<8)
                        # yields (k0,k4),(k1,k5).  The dot is order-invariant,
                        # so index A to match instead of shuffling W.
                        p26, p37 = packed_dequant_e2m1x4_to_bfloat2x2(wu)
                        p04, p15 = packed_dequant_e2m1x4_to_bfloat2x2(wu << 8)
                        f3, f7 = _bf16x2_to_f32x2(p37)
                        f2, f6 = _bf16x2_to_f32x2(p26)
                        f1, f5 = _bf16x2_to_f32x2(p15)
                        f0, f4 = _bf16x2_to_f32x2(p04)
                        fs = (f0, f1, f2, f3, f4, f5, f6, f7)
                        for g in cutlass.range_constexpr(M):
                            af = afrags[g][h * self._I32_PER_SF + j]
                            dot[g] = (
                                dot[g]
                                + fs[0] * Float32(af[0])
                                + fs[1] * Float32(af[1])
                                + fs[2] * Float32(af[2])
                                + fs[3] * Float32(af[3])
                                + fs[4] * Float32(af[4])
                                + fs[5] * Float32(af[5])
                                + fs[6] * Float32(af[6])
                                + fs[7] * Float32(af[7])
                            )
                    # Unfold the two power-of-two ratios separately (their
                    # product is not f32-representable).  dot's raw terms sit
                    # near 2^-126 where f32 subnormals lose a few mantissa
                    # bits, but only on terms ~2^-22 below the block dot --
                    # far under the e2m1 quantization noise.
                    for g in cutlass.range_constexpr(M):
                        acc[g] = acc[g] + (dot[g] * _TWO_POW_126) * s_true

            for g in cutlass.range_constexpr(M):
                r = acc[g]
                for s in cutlass.range_constexpr(5):  # log2(32-lane warp)
                    r = r + cute.arch.shuffle_sync_bfly(r, 1 << s)
                if lane == 0:
                    mC[g, row] = mC.element_type(r * alpha)
