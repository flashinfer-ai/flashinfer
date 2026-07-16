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

from .topk_select_radix_sm12x import _atomic_add_i32

# SMEM-resident score cap and dispatch threshold. Below this many candidate blocks
# (128 blocks = 16k context) the O(N^2) rank count beats the radix kernel's fixed
# multi-pass cost; the crossover was measured empirically, and 128 is conservative
# for parts with more SMs.
_MAX_BLOCKS = 128
_NTHREADS = 256
_SENTINEL = 0x7FFFFFFF  # INT32_MAX: empty slots sort to the tail, unlike -1


class TopKSelectCountRankSm12x:
    """O(N^2) count-rank top-K selection for small candidate counts."""

    def __init__(self, topk: int):
        if topk != 16:
            raise ValueError(f"topk must be 16, got {topk}")
        self._topk = topk

    @cute.jit
    def __call__(
        self,
        mMaxScore: cute.Tensor,  # (H, P, S) f32  (P = max_k_tiles)
        mOut: cute.Tensor,  # (S, H, topk) int32
        num_valid_pages: cutlass.Int32,
        force_begin: cutlass.Int32,
        force_end: cutlass.Int32,
        total_qo_len: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mBits = cute.recast_tensor(mMaxScore, cutlass.Uint32)
        self.kernel(mBits, mOut, num_valid_pages, force_begin, force_end).launch(
            grid=(total_qo_len, num_qo_heads, 1),
            block=(_NTHREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _radix_key(self, bits):
        """Order-preserving transform: ascending key == descending float score."""
        key = (~bits) & cutlass.Uint32(0x7FFFFFFF)
        if (bits & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
            key = bits
        return key

    @cute.kernel
    def kernel(
        self,
        mScore: cute.Tensor,  # (H, P, S) f32 recast to u32 bits
        mOut: cute.Tensor,  # (S, H, topk) int32
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

        nvp = num_valid_pages
        mid_lo = force_begin
        mid_hi = num_valid_pages - force_end
        n_forced = force_begin + force_end
        target = cutlass.Int32(self._topk) - n_forced

        # Stage the middle scores' radix keys in SMEM: the rank loop rereads each
        # one N times, and the bit key preserves the exact radix-kernel order,
        # with deterministic NaN placement.
        b = mid_lo + tid
        while b < mid_hi:
            score[b] = self._radix_key(mScore[h, b, q])
            b += _NTHREADS

        # Forced sink/window blocks bypass ranking; emit slots start after them
        if tid == 0:
            w = cutlass.Int32(0)
            i = cutlass.Int32(0)
            while i < force_begin:
                sel[w] = i
                w += 1
                i += 1
            j = cutlass.Int32(0)
            while j < force_end:
                sel[w] = (nvp - force_end) + j
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
