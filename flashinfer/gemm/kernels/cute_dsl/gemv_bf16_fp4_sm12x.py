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

Streaming W4A16 GEMV for SM12x: bf16 activation row x NVFP4 weights at m=1.

The MMA kernel's design point (SMEM pipeline feeding tensor cores) is wrong
for m=1: the problem is DRAM-bound and the tensor-core math is nearly free,
so pipeline depth buys nothing while its SMEM cost caps resident warps.
This kernel inverts the trade. Weights stream GMEM -> registers with no
SMEM and no tensor cores, and latency is hidden by warp count alone
(~48 resident warps/SM), which holds the DRAM-bandwidth ceiling at m=1
where the MMA path falls short.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ....cute_dsl.fp4_common import f16x2_to_f32x2, fp4_decode_4bytes


@dsl_user_op
def _s0e5m3_to_f32(byte: Uint32, *, loc=None, ip=None) -> Float32:
    """Decode one S0E5M3 scale byte to f32: ``f16(byte << 7)``. Scalar-f32
    form of fp4_common's ``cvt_s0e5m3_to_f16x2_broadcast``."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(byte).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 t;
                .reg .b16 h;
                shl.b32 t, $1, 7;
                cvt.u16.u32 h, t;
                cvt.f32.f16 $0, h;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class GemvBf16Fp4Sm12x:
    """Streaming bf16 x nvfp4 GEMV over the cute-DSL backend's prepared
    tensors: the tile-packed ``(K/16, N*2)`` int32 weight and the S0E5M3
    ``(K/16, N)`` scales, the same operands the MMA kernel takes, so the
    two dispatch freely per call.

    Each warp owns one 64-wide N tile and streams its (16K x 64N) packed
    tiles along K: 4 int32 per lane per tile is one contiguous 16B load,
    so the warp consumes each 512B tile fully coalesced.  The pack's MMA
    thread mapping then pins per lane which (k, n) each byte holds: lane
    ``l`` covers n = base + {0,16,32,48} (base drawn from l) over half the
    tile's K, and its xor-1 partner lane covers the other half, so four
    f32 partials per lane plus one butterfly step complete each dot
    product.  fp4 codes and scales decode with single hardware ops
    (``cvt.f16x2.e2m1x2``, f16 bit-place) and accumulate in f32.

    ``splits`` shards K across grid.y for grid fill (a 64-wide-tile grid
    alone underfills large GPUs); split partials reuse the MMA kernel's
    fp32-workspace + fixed-order reduce scheme, so results stay
    deterministic.  m == 1 only: at m >= 2 the serial FMA chain scales
    with m while the loads do not, and the MMA path wins.
    """

    _NUM_WARPS = 8  # one 64-wide N tile per warp
    _TILE_N = 64  # fixed by the weight pack (_CUTE_DSL_PACK_TILE_N)
    _TILE_K = 16  # fixed by the weight pack; also the scale-block size
    _I32_PER_LANE = 4  # 128 int32 per packed tile / 32 lanes

    def __init__(self, splits: int = 1, enable_pdl: bool = True):
        if splits < 1:
            raise ValueError("splits must be >= 1")
        self._splits = splits
        self._enable_pdl = enable_pdl
        self._num_threads = self._NUM_WARPS * 32
        self._reduce_threads = 128

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M, K) bf16, row 0 used
        mB: cute.Tensor,  # (K/16, N*2) int32 tile-packed weight
        mSF: cute.Tensor,  # (K/16, N) uint8 S0E5M3 scales
        mC: cute.Tensor,  # (M, N)
        mPartial: cute.Tensor,  # (splits * N,) f32, dummy when splits == 1
        mAlpha: cute.Tensor,  # (1,) f32
        stream: cuda.CUstream,
    ):
        # M == 1 only: partials write at stride N but the reduce indexes at
        # stride M*N, so they agree only at M == 1. Enforced host-side by the
        # runner (the m != 1 guard in forward); M is symbolic here.
        n_tiles = cute.size(mSF, mode=[1]) // self._TILE_N
        self.kernel(mA, mB, mSF, mC, mPartial, mAlpha).launch(
            grid=(
                (n_tiles + self._NUM_WARPS - 1) // self._NUM_WARPS,
                self._splits,
                1,
            ),
            block=[self._num_threads, 1, 1],
            stream=stream,
            use_pdl=self._enable_pdl,
        )
        if cutlass.const_expr(self._splits > 1):
            total = cute.size(mC, mode=[0]) * cute.size(mC, mode=[1])
            reduce_grid = (total + Int32(self._reduce_threads) - Int32(1)) // Int32(
                self._reduce_threads
            )
            self.kernel_partial_reduce(mPartial, mC).launch(
                grid=[reduce_grid, 1, 1],
                block=[self._reduce_threads, 1, 1],
                stream=stream,
                use_pdl=self._enable_pdl,
            )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mSF: cute.Tensor,
        mC: cute.Tensor,
        mPartial: cute.Tensor,
        mAlpha: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        bidx, split, _ = cute.arch.block_idx()
        nt = bidx * self._NUM_WARPS + warp

        n = cute.size(mSF, mode=[1])
        n_tiles = n // self._TILE_N
        k_tiles = cute.size(mB, mode=[0])

        cute.arch.griddepcontrol_wait()

        # No SMEM and no CTA-wide sync, so a tail warp with no tile can
        # retire early without deadlocking its CTA.
        if nt < n_tiles:
            kt_per_split = (k_tiles + Int32(self._splits) - Int32(1)) // Int32(
                self._splits
            )
            kt0 = split * kt_per_split
            kt1 = cutlass.min(kt0 + kt_per_split, k_tiles)

            # Lane-fixed slice of the pack mapping (see the class docstring):
            # this lane's 16B load holds codes for n = base_n + {0,16,32,48}
            # at k_half in 2*(lane%2) + {0,1,4,5}.
            base_n = (lane // 16) * 8 + (lane % 16) // 2

            acc = cute.make_rmem_tensor((4,), Float32)
            acc.fill(0.0)
            wfrag = cute.make_rmem_tensor(cute.make_layout(self._I32_PER_LANE), Int32)
            afrags = [
                cute.make_rmem_tensor(cute.make_layout(4), mA.element_type)
                for _ in range(2)
            ]

            for kt in cutlass.range(kt0, kt1, 1):
                w_chunk = cute.local_tile(
                    mB[kt, None], (self._I32_PER_LANE,), (nt * 32 + lane,)
                )
                cute.autovec_copy(w_chunk, wfrag)
                # Each lane loads only its own 8 A values: k in
                # 2*trh0 + {0..3} and 2*trh0 + {8..11} of the tile.  The
                # lane parity lives in the address so the fragment indices
                # below stay compile-time.
                for h in cutlass.range_constexpr(2):
                    a_chunk = cute.local_tile(
                        mA[0, None], (4,), (kt * 4 + 2 * h + (lane % 2),)
                    )
                    cute.autovec_copy(a_chunk, afrags[h])
                af = [[Float32(afrags[h][i]) for i in range(4)] for h in range(2)]

                dot = [Float32(0.0) for _ in range(4)]
                for j in cutlass.range_constexpr(self._I32_PER_LANE):
                    b0, b1, b2, b3 = fp4_decode_4bytes(Uint32(wfrag[j]))
                    for b, reg in enumerate((b0, b1, b2, b3)):
                        p = 2 * (j & 1) + (b >> 1)
                        lo, hi = f16x2_to_f32x2(reg)
                        dot[p] = (
                            dot[p]
                            + lo * af[b & 1][2 * (j >> 1)]
                            + hi * af[b & 1][2 * (j >> 1) + 1]
                        )

                for p in cutlass.range_constexpr(4):
                    sb = Uint32(mSF[kt, nt * self._TILE_N + base_n + 16 * p])
                    acc[p] = acc[p] + dot[p] * _s0e5m3_to_f32(sb)

            # xor-1 partners cover complementary k_half sets of the same
            # four n's; one butterfly completes each dot product.
            for p in cutlass.range_constexpr(4):
                acc[p] = acc[p] + cute.arch.shuffle_sync_bfly(acc[p], 1)

            alpha = mAlpha[0]
            if lane % 2 == 0:
                for p in cutlass.range_constexpr(4):
                    n_idx = nt * self._TILE_N + lane // 2 + 16 * p
                    if cutlass.const_expr(self._splits == 1):
                        mC[0, n_idx] = mC.element_type(acc[p] * alpha)
                    else:
                        mPartial[split * n + n_idx] = acc[p] * alpha

        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel_partial_reduce(
        self,
        mPartial: cute.Tensor,
        mC_mn: cute.Tensor,
    ):
        """Sum the fp32 partials into C, one element per thread, in fixed
        split order for determinism (same scheme as the MMA kernel)."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        cute.arch.griddepcontrol_wait()

        total = cute.size(mC_mn, mode=[0]) * cute.size(mC_mn, mode=[1])
        mC_flat = cute.make_tensor(mC_mn.iterator, cute.make_layout(total))

        idx = Int32(bidx) * Int32(self._reduce_threads) + tidx
        if idx < total:
            acc_sum = Float32(0.0)
            # Dynamic loop: fill-derived splits reach ~100+, too many to unroll.
            for s in cutlass.range(self._splits):
                acc_sum = acc_sum + Float32(mPartial[Int32(s) * total + idx])
            mC_flat[idx] = mC_flat.element_type(acc_sum)

        cute.arch.griddepcontrol_launch_dependents()
