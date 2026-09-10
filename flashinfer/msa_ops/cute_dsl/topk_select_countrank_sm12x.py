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

Count-rank MSA top-K KV-block selection for SM120/SM121: O(N^2) rank count,
dispatched below ``_MAX_BLOCKS`` where it beats the radix kernel's fixed pass cost.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .topk_select_radix_sm12x import _atomic_add_i32, _radix_key

# SMEM-resident score cap and dispatch threshold. Below this many candidate blocks
# (128 blocks = 16k context) the O(N^2) rank count beats the radix kernel's fixed
# multi-pass cost; the crossover was measured empirically, and 128 is conservative
# for parts with more SMs.
_MAX_BLOCKS = 128
_NTHREADS = 256
_SENTINEL = 0x7FFFFFFF  # INT32_MAX: empty slots sort to the tail, unlike -1


class TopKSelectCountRankSm12x:
    """O(N^2) count-rank top-K selection for small candidate counts."""

    def __init__(self, topk: int, per_token_nvp: bool = False):
        if topk != 16:
            raise ValueError(f"topk must be 16, got {topk}")
        self._topk = topk
        # Per-token valid-page counts: each query token carries its own causal
        # KV extent, so the forced local window and the ranked middle range are
        # token-relative instead of batch-uniform.
        self._per_token_nvp = per_token_nvp

    @cute.jit
    def __call__(
        self,
        mMaxScore: cute.Tensor,  # (H, P, S) f32  (P = max_k_tiles)
        mOut: cute.Tensor,  # (S, H, topk) int32
        mNumValidPages: cute.Tensor,  # (S,) int32; dummy when not per-token
        num_valid_pages: cutlass.Int32,
        force_begin: cutlass.Int32,
        force_end: cutlass.Int32,
        total_qo_len: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mBits = cute.recast_tensor(mMaxScore, cutlass.Uint32)
        self.kernel(
            mBits, mOut, mNumValidPages, num_valid_pages, force_begin, force_end
        ).launch(
            grid=(total_qo_len, num_qo_heads, 1),
            block=(_NTHREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mScore: cute.Tensor,  # (H, P, S) f32 recast to u32 bits
        mOut: cute.Tensor,  # (S, H, topk) int32
        mNumValidPages: cute.Tensor,  # (S,) int32
        num_valid_pages: cutlass.Int32,
        force_begin: cutlass.Int32,
        force_end: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        q, h, _ = cute.arch.block_idx()

        @cute.struct
        class SharedStorage:
            score: cute.struct.MemRange[cutlass.Uint32, _MAX_BLOCKS]
            sel: cute.struct.MemRange[cutlass.Int32, 16]
            cnt: cute.struct.MemRange[cutlass.Int32, 1]

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(SharedStorage)
        score = st.score.get_tensor(cute.make_layout(_MAX_BLOCKS))
        sel = st.sel.get_tensor(cute.make_layout(16))
        cnt = st.cnt.get_tensor(cute.make_layout(1))

        if cutlass.const_expr(self._per_token_nvp):
            # Per-token counts arrive on device, so nothing on the host bounds
            # them (reading them would sync). Clamp to the score tensor's own
            # block extent, which is also the `score` staging bound because the
            # host only dispatches here when max_k_tiles fits _MAX_BLOCKS: an
            # out-of-range entry would otherwise read mScore past its end,
            # overrun smem, and emit block indices the attend kernel cannot
            # address. An over-large count degrades to the full block range.
            cap = cutlass.min(
                cutlass.Int32(mScore.shape[1]), cutlass.Int32(_MAX_BLOCKS)
            )
            nvp = cutlass.max(cutlass.Int32(0), cutlass.min(mNumValidPages[q], cap))
        else:
            nvp = num_valid_pages

        # Per-token nvp can be smaller than the forced region (a token whose
        # causal extent is shorter than the local window), which the host-side
        # scalar check cannot rule out. Shrink the forced region to fit rather
        # than emitting negative block indices.
        fb = cutlass.min(force_begin, nvp)
        fe = cutlass.min(force_end, nvp - fb)

        mid_lo = fb
        mid_hi = nvp - fe
        n_forced = fb + fe
        target = cutlass.Int32(self._topk) - n_forced

        # Stage the middle scores' radix keys in SMEM: the rank loop rereads each
        # one N times, and the bit key preserves the exact radix-kernel order,
        # with deterministic NaN placement.
        b = mid_lo + tid
        while b < mid_hi:
            score[b] = _radix_key(mScore[h, b, q])
            b += _NTHREADS

        # Forced sink/window blocks bypass ranking; emit slots start after them
        if tid == 0:
            w = cutlass.Int32(0)
            i = cutlass.Int32(0)
            while i < fb:
                sel[w] = i
                w += 1
                i += 1
            j = cutlass.Int32(0)
            while j < fe:
                sel[w] = mid_hi + j
                w += 1
                j += 1
            k = w
            while k < self._topk:
                sel[k] = cutlass.Int32(_SENTINEL)
                k += 1
            cnt[0] = cutlass.Int32(0)
        cute.arch.barrier()

        # rank = count of strictly-better blocks (lower key = higher score, ties
        # broken toward the lower index); a block is selected iff rank < target.
        b = mid_lo + tid
        while b < mid_hi:
            kb = score[b]
            rank = cutlass.Int32(0)
            j = mid_lo
            while j < mid_hi:
                kj = score[j]
                if (kj < kb) or ((kj == kb) and (j < b)):
                    rank += 1
                j += 1
            if rank < target:
                slot = _atomic_add_i32(1, cnt.iterator + 0)
                if n_forced + slot < cutlass.Int32(self._topk):
                    sel[n_forced + slot] = b
            b += _NTHREADS
        cute.arch.barrier()

        # Ascending-by-index sort, then write.
        if tid == 0:
            a = cutlass.Int32(1)
            while a < self._topk:
                key2 = sel[a]
                p = a - 1
                while (p >= 0) and (sel[p] > key2):
                    sel[p + 1] = sel[p]
                    p -= 1
                sel[p + 1] = key2
                a += 1
            kk = cutlass.Int32(0)
            while kk < self._topk:
                v = sel[kk]
                if v == cutlass.Int32(_SENTINEL):
                    v = cutlass.Int32(-1)
                mOut[q, h, kk] = v
                kk += 1
