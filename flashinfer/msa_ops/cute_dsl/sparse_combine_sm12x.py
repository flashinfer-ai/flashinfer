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

MSA fused split-KV combine for SM120/SM121.

Reduces per-KV-block partial outputs into the final attention output with
log2-domain LSE weighting:

    out[q, h, :] = sum_s w_s * o_partial[s, q, h, :] / sum_s w_s
    w_s          = exp2(lse2[s, q, h] - max_s lse2[s, q, h])

where ``s`` ranges over the query's valid split slots
``[0, split_counts[q, h // group_size])``. Slots >= count hold uninitialized
memory and are never read. Optionally writes the combined natural-log LSE and a
separately-accumulated temperature LSE (both converted log2 -> ln at the end).

One CTA per (query token, query head); a single thread builds the slot weights
in SMEM (count <= topk <= 256), then all threads cooperatively reduce the
head_dim channels.
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

_LN2 = 0.6931471805599453


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

        hkv = h // group_size
        count = mSplitCounts[q, hkv]
        if count > self._topk:
            count = cutlass.Int32(self._topk)

        @cute.struct
        class SharedStorage:
            s_w: cute.struct.MemRange[cutlass.Float32, self._topk]
            s_denom: cute.struct.MemRange[cutlass.Float32, 1]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_w = storage.s_w.get_tensor(cute.make_layout(self._topk))
        s_denom = storage.s_denom.get_tensor(cute.make_layout(1))

        neg_inf = -cutlass.Float32.inf

        if count <= 0:
            # No valid splits: zero the output, -inf LSE.
            zero = cutlass.Float32(0.0).to(mOut.element_type)
            for i in cutlass.range_constexpr(self._channels_per_thread):
                c = tidx + i * self._num_threads
                mOut[q, h, c] = zero
            if cutlass.const_expr(self._has_lse_out):
                if tidx == 0:
                    mLseOut[q, h] = neg_inf
            if cutlass.const_expr(self._has_lse_t):
                if tidx == 0:
                    mLseTOut[q, h] = neg_inf
        else:
            # Thread 0 builds the slot weights (count <= topk is small).
            if tidx == 0:
                m = neg_inf
                for s in cutlass.range_constexpr(self._topk):
                    if s < count:
                        v = mLse2[s, q, h]
                        s_w[s] = v
                        m = cute.arch.fmax(m, v)
                m_finite = m > neg_inf
                denom = cutlass.Float32(0.0)
                for s in cutlass.range_constexpr(self._topk):
                    if s < count:
                        w = cutlass.Float32(0.0)
                        if m_finite:
                            w = cute.math.exp2(s_w[s] - m, fastmath=True)
                        s_w[s] = w
                        denom += w
                s_denom[0] = denom

                if cutlass.const_expr(self._has_lse_t):
                    mt = neg_inf
                    for s in cutlass.range_constexpr(self._topk):
                        if s < count:
                            mt = cute.arch.fmax(mt, mLseT2[s, q, h])
                    mt_finite = mt > neg_inf
                    dt = cutlass.Float32(0.0)
                    if mt_finite:
                        for s in cutlass.range_constexpr(self._topk):
                            if s < count:
                                dt += cute.math.exp2(
                                    mLseT2[s, q, h] - mt, fastmath=True
                                )
                    lse_t = neg_inf
                    if dt > 0.0:
                        lse_t = (mt + cute.math.log2(dt)) * _LN2
                    mLseTOut[q, h] = lse_t

                if cutlass.const_expr(self._has_lse_out):
                    lse = neg_inf
                    if denom > 0.0:
                        lse = (m + cute.math.log2(denom)) * _LN2
                    mLseOut[q, h] = lse

            cute.arch.sync_threads()

            denom = s_denom[0]
            inv = cutlass.Float32(0.0)
            if denom > 0.0:
                inv = cutlass.Float32(1.0) / denom

            for i in cutlass.range_constexpr(self._channels_per_thread):
                c = tidx + i * self._num_threads
                acc = cutlass.Float32(0.0)
                for s in cutlass.range_constexpr(self._topk):
                    if s < count:
                        w = s_w[s]
                        if w > 0.0:
                            e = mO_partial[s, q, h, c]
                            if cutlass.const_expr(self._partial_is_fp8):
                                ef = e.to(cutlass.Float16).to(cutlass.Float32)
                            else:
                                ef = e.to(cutlass.Float32)
                            acc += w * ef
                mOut[q, h, c] = (acc * inv * out_scale).to(mOut.element_type)
