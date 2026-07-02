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

---

MSA split-KV combine kernel for SM120/SM121: LSE-weighted merge of the
sparse-decode split partials into the final output.
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

_LN2 = 0.6931471805599453
_FLT_MAX = 3.4028234663852886e38


class SparseCombineSm12x:
    def __init__(
        self,
        head_dim: int,
        topk: int,
        partial_is_fp8: bool,
        has_lse_out: bool,
        has_lse_t: bool,
        num_threads: int = 128,
    ):
        if head_dim % num_threads != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be a multiple of num_threads "
                f"({num_threads})"
            )
        self._head_dim = head_dim
        self._topk = topk
        self._partial_is_fp8 = partial_is_fp8
        self._has_lse_out = has_lse_out
        self._has_lse_t = has_lse_t
        self._num_threads = num_threads
        self._channels_per_thread = head_dim // num_threads

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,  # (topk, total_q, Hq, d)
        mLse2: cute.Tensor,  # (topk, total_q, Hq) f32, log2 domain
        mSplitCounts: cute.Tensor,  # (total_q, Hkv) int32
        mOut: cute.Tensor,  # (total_q, Hq, d)
        mLseOut: cute.Tensor,  # (total_q, Hq) f32 or dummy
        mLseT2: cute.Tensor,  # (topk, total_q, Hq) f32 or dummy
        mLseTOut: cute.Tensor,  # (total_q, Hq) f32 or dummy
        out_scale: cutlass.Float32,
        total_q: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        group_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._partial_dtype: Type[cutlass.Numeric] = mO_partial.element_type
        self.kernel(
            mO_partial,
            mLse2,
            mSplitCounts,
            mOut,
            mLseOut,
            mLseT2,
            mLseTOut,
            out_scale,
            group_size,
        ).launch(
            grid=(total_q, num_qo_heads, 1),
            block=[self._num_threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLse2: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mOut: cute.Tensor,
        mLseOut: cute.Tensor,
        mLseT2: cute.Tensor,
        mLseTOut: cute.Tensor,
        out_scale: cutlass.Float32,
        group_size: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q, h, _ = cute.arch.block_idx()

        # PDL: launched early; wait for the forward's partials before reading.
        cute.arch.griddepcontrol_wait()
        hkv = h // group_size
        count = mSplitCounts[q, hkv]
        if count > self._topk:
            count = cutlass.Int32(self._topk)

        lse_t_slots = self._topk if self._has_lse_t else 1

        @cute.struct
        class SharedStorage:
            s_lse: cute.struct.MemRange[cutlass.Float32, self._topk]
            s_lse_t: cute.struct.MemRange[cutlass.Float32, lse_t_slots]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_lse = storage.s_lse.get_tensor(cute.make_layout(self._topk))
        s_lse_t = storage.s_lse_t.get_tensor(cute.make_layout(lse_t_slots))

        neg_inf = -cutlass.Float32.inf

        # count <= 0 needs no special case: no slot passes s < count -> weight 0
        for it in cutlass.range_constexpr(cute.ceil_div(self._topk, self._num_threads)):
            slot = tidx + it * self._num_threads
            if slot < self._topk:
                v = neg_inf
                if slot < count:
                    v = mLse2[slot, q, h]
                s_lse[slot] = v
                if cutlass.const_expr(self._has_lse_t):
                    vt = neg_inf
                    if slot < count:
                        vt = mLseT2[slot, q, h]
                    s_lse_t[slot] = vt
        cute.arch.sync_threads()

        # weights rebuilt redundantly per thread: at low batch the grid is a
        # sub-wave, so serializing the build on one thread dominates the kernel
        m = neg_inf
        for s in cutlass.range_constexpr(self._topk):
            if s < count:
                m = cute.arch.fmax(m, s_lse[s])
        m_finite = m > neg_inf
        w_frag = cute.make_rmem_tensor(cute.make_layout(self._topk), cutlass.Float32)
        denom = cutlass.Float32(0.0)
        for s in cutlass.range_constexpr(self._topk):
            w = cutlass.Float32(0.0)
            if s < count and m_finite:
                w = cute.math.exp2(s_lse[s] - m, fastmath=True)
            w_frag[s] = w
            denom += w
        inv = cutlass.Float32(0.0)
        if denom > 0.0:
            inv = cutlass.Float32(1.0) / denom

        # Branch-free: all topk slots load unconditionally (they pipeline); invalid
        # slots get weight 0, but 0 * NaN = NaN, so clamp the garbage to finite first
        # (fmax drops a NaN operand). cutlass-dsl 4.5.2 has no fmin: use -fmax(-x, -c).
        for i in cutlass.range_constexpr(self._channels_per_thread):
            c = tidx + i * self._num_threads
            acc = cutlass.Float32(0.0)
            for s in cutlass.range_constexpr(self._topk):
                e = mO_partial[s, q, h, c]
                if cutlass.const_expr(self._partial_is_fp8):
                    ef = e.to(cutlass.Float16).to(cutlass.Float32)
                else:
                    ef = e.to(cutlass.Float32)
                ef = cute.arch.fmax(-cute.arch.fmax(-ef, -_FLT_MAX), -_FLT_MAX)
                acc += w_frag[s] * ef
            mOut[q, h, c] = (acc * inv * out_scale).to(mOut.element_type)

        if cutlass.const_expr(self._has_lse_out):
            if tidx == 0:
                lse = neg_inf
                if denom > 0.0:
                    lse = (m + cute.math.log2(denom)) * _LN2
                mLseOut[q, h] = lse
        if cutlass.const_expr(self._has_lse_t):
            if tidx == 0:
                mt = neg_inf
                for s in cutlass.range_constexpr(self._topk):
                    if s < count:
                        mt = cute.arch.fmax(mt, s_lse_t[s])
                dt = cutlass.Float32(0.0)
                if mt > neg_inf:
                    for s in cutlass.range_constexpr(self._topk):
                        if s < count:
                            dt += cute.math.exp2(s_lse_t[s] - mt, fastmath=True)
                lse_t = neg_inf
                if dt > 0.0:
                    lse_t = (mt + cute.math.log2(dt)) * _LN2
                mLseTOut[q, h] = lse_t
