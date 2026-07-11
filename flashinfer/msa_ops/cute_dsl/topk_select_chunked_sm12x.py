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

Chunked count-rank MSA top-K KV-block selection for SM120/SM121: candidate
blocks are split into chunks ranked by independent CTAs, and a second kernel
merges the per-chunk survivors. Recovers parallelism when the (query, head)
grid alone leaves the GPU idle — the single-CTA-per-row kernels serialize the
whole candidate scan on a handful of SMs there.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

# Per-chunk staging cap: matches the count-rank kernel's SMEM score cap, so a
# chunk is never bigger than what the single-kernel path already handles well.
_MAX_CHUNK_BLOCKS = 128
# Merge capacity is _MAX_CHUNKS * topk candidate slots; together with the chunk
# cap this bounds the middle region the chunked path can cover.
_MAX_CHUNKS = 16
# Target blocks per chunk. Small enough to spread one row's scan across many
# CTAs, large enough that the merge's candidate count (topk per chunk) stays
# low — measured crossover: halving the chunk count beats halving the scan.
_CHUNK_BLOCKS = 64
# Below two chunks the split is pure overhead over the single-CTA count-rank.
_MIN_CHUNKS = 2
# Dispatch floor: under this many middle blocks the single-CTA count-rank's
# scan is short enough that the second launch never pays for itself.
_MIN_BLOCKS = 32
_NTHREADS_PARTIAL = 128
_NTHREADS_MERGE = 256
_SENTINEL = 0x7FFFFFFF  # INT32_MAX: empty slots sort to the tail, unlike -1
# Empty candidate slots also carry the worst possible key; detection uses the
# index (a real NaN score can legitimately produce key 0xFFFFFFFF).
_SENTINEL_KEY = 0xFFFFFFFF


class TopKSelectChunkedSm12x:
    """Two-phase (per-chunk rank + merge) top-K selection for small grids.

    Ranks by the same (radix bit-key, block index) total order as the
    single-kernel count-rank path, so the two produce identical selections.
    """

    def __init__(self, topk: int):
        if topk != 16:
            raise ValueError(f"topk must be 16, got {topk}")
        self._topk = topk

    @cute.jit
    def __call__(
        self,
        mMaxScore: cute.Tensor,  # (H, P, S) f32  (P = max_k_tiles)
        mCandKey: cute.Tensor,  # (S, H, C*topk) int32 scratch, recast to u32
        mCandIdx: cute.Tensor,  # (S, H, C*topk) int32 scratch
        mOut: cute.Tensor,  # (S, H, topk) int32
        num_valid_pages: cutlass.Int32,
        force_begin: cutlass.Int32,
        force_end: cutlass.Int32,
        num_chunks: cutlass.Int32,
        chunk_len: cutlass.Int32,
        total_qo_len: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mBits = cute.recast_tensor(mMaxScore, cutlass.Uint32)
        mKey = cute.recast_tensor(mCandKey, cutlass.Uint32)
        self.kernel_partial(
            mBits, mKey, mCandIdx, num_valid_pages, force_begin, force_end, chunk_len
        ).launch(
            grid=(total_qo_len, num_qo_heads, num_chunks),
            block=(_NTHREADS_PARTIAL, 1, 1),
            stream=stream,
        )
        self.kernel_merge(
            mKey, mCandIdx, mOut, num_valid_pages, force_begin, force_end, num_chunks
        ).launch(
            grid=(total_qo_len, num_qo_heads, 1),
            block=(_NTHREADS_MERGE, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _radix_key(self, bits):
        """Order-preserving transform: ascending key == descending float score."""
        key = (~bits) & cutlass.Uint32(0x7FFFFFFF)
        if (bits & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
            key = bits
        return key

    @cute.kernel
    def kernel_partial(
        self,
        mBits: cute.Tensor,  # (H, P, S) uint32, raw bits of max_score
        mKey: cute.Tensor,  # (S, H, C*topk) uint32 candidate keys
        mIdx: cute.Tensor,  # (S, H, C*topk) int32 candidate block indices
        num_valid_pages: cutlass.Int32,
        force_begin: cutlass.Int32,
        force_end: cutlass.Int32,
        chunk_len: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        q, h, c = cute.arch.block_idx()

        @cute.struct
        class SharedStorage:
            score: cute.struct.MemRange[cutlass.Uint32, _MAX_CHUNK_BLOCKS]

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(SharedStorage)
        score = st.score.get_tensor(cute.make_layout(_MAX_CHUNK_BLOCKS))

        mid_hi = num_valid_pages - force_end
        n_forced = force_begin + force_end
        target = cutlass.Int32(self._topk) - n_forced

        # This chunk's slice of the middle region (tail chunk may be short or,
        # past the end, empty; the sentinel fill below still runs for those).
        c_lo = force_begin + c * chunk_len
        c_hi = c_lo + chunk_len
        if c_hi > mid_hi:
            c_hi = mid_hi
        n_local = c_hi - c_lo
        # Padding the staged keys to the unroll width keeps the rank loop
        # branch-free; padded entries sit at local indices >= any real block,
        # so neither the key compare nor the lower-index tie-break counts them.
        n_pad = (n_local + 3) & ~cutlass.Int32(3)
        cand_base = c * cutlass.Int32(self._topk)

        # Sentinel-fill this chunk's candidate slots; survivors overwrite after
        # the barrier below, so unfilled slots read as empty in the merge.
        if tid < self._topk:
            mKey[q, h, cand_base + tid] = cutlass.Uint32(_SENTINEL_KEY)
            mIdx[q, h, cand_base + tid] = cutlass.Int32(_SENTINEL)

        # Stage the chunk's radix keys in SMEM: the rank loop rereads each one
        # n_local times, and the bit key preserves the exact radix-kernel order,
        # with deterministic NaN placement.
        l = cutlass.Int32(tid)
        while l < n_pad:
            k = cutlass.Uint32(_SENTINEL_KEY)
            if l < n_local:
                k = self._radix_key(mBits[h, c_lo + l, q])
            score[l] = k
            l += _NTHREADS_PARTIAL
        cute.arch.barrier()

        # rank = count of strictly-better blocks within the chunk (lower key =
        # higher score, ties broken toward the lower index). Every block in the
        # global top-target ranks < target inside its own chunk too, so the
        # per-chunk survivors are a superset of the final selection. Ranks are
        # distinct, so a survivor's rank IS its candidate slot — no atomics.
        # The scan is 4-way unrolled: the serial chain of dependent SMEM loads
        # is what a near-empty grid can't hide, so shortening it is the win.
        l = cutlass.Int32(tid)
        while l < n_local:
            kb = score[l]
            rank = cutlass.Int32(0)
            j = cutlass.Int32(0)
            while j < n_pad:
                k0 = score[j]
                k1 = score[j + 1]
                k2 = score[j + 2]
                k3 = score[j + 3]
                if (k0 < kb) or ((k0 == kb) and (j < l)):
                    rank += 1
                if (k1 < kb) or ((k1 == kb) and (j + 1 < l)):
                    rank += 1
                if (k2 < kb) or ((k2 == kb) and (j + 2 < l)):
                    rank += 1
                if (k3 < kb) or ((k3 == kb) and (j + 3 < l)):
                    rank += 1
                j += 4
            if rank < target:
                mKey[q, h, cand_base + rank] = kb
                mIdx[q, h, cand_base + rank] = c_lo + l
            l += _NTHREADS_PARTIAL

        # PDL: let the merge grid start its preamble now; its griddepcontrol
        # wait holds off the candidate reads until these stores land.
        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel_merge(
        self,
        mKey: cute.Tensor,  # (S, H, C*topk) uint32 candidate keys
        mIdx: cute.Tensor,  # (S, H, C*topk) int32 candidate block indices
        mOut: cute.Tensor,  # (S, H, topk) int32
        num_valid_pages: cutlass.Int32,
        force_begin: cutlass.Int32,
        force_end: cutlass.Int32,
        num_chunks: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        q, h, _ = cute.arch.block_idx()

        @cute.struct
        class SharedStorage:
            key: cute.struct.MemRange[cutlass.Uint32, _MAX_CHUNKS * 16]
            idx: cute.struct.MemRange[cutlass.Int32, _MAX_CHUNKS * 16]
            sel: cute.struct.MemRange[cutlass.Int32, 16]

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(SharedStorage)
        key = st.key.get_tensor(cute.make_layout(_MAX_CHUNKS * 16))
        idx = st.idx.get_tensor(cute.make_layout(_MAX_CHUNKS * 16))
        sel = st.sel.get_tensor(cute.make_layout(16))

        nvp = num_valid_pages
        n_forced = force_begin + force_end
        target = cutlass.Int32(self._topk) - n_forced
        n_cand = num_chunks * cutlass.Int32(self._topk)

        # Forced sink/window blocks bypass ranking; emit slots start after
        # them. SMEM-only, so it runs in the PDL preamble before the wait.
        if tid == 0:
            w = cutlass.Int32(0)
            fi = cutlass.Int32(0)
            while fi < force_begin:
                sel[w] = fi
                w += 1
                fi += 1
            fj = cutlass.Int32(0)
            while fj < force_end:
                sel[w] = (nvp - force_end) + fj
                w += 1
                fj += 1
            k = w
            while k < self._topk:
                sel[k] = cutlass.Int32(_SENTINEL)
                k += 1

        # PDL: this kernel launches while the partial ranker is still running;
        # wait for its candidate stores before reading them.
        cute.arch.griddepcontrol_wait()

        i = cutlass.Int32(tid)
        while i < n_cand:
            key[i] = mKey[q, h, i]
            idx[i] = mIdx[q, h, i]
            i += _NTHREADS_MERGE
        cute.arch.barrier()

        # Global rank over the union of chunk survivors, by the same (key,
        # block index) order the chunks used; empty slots never rank or count
        # (their key/tie-break can never win, so no guard is needed on them).
        # Distinct ranks double as emit slots; the scan is 4-way unrolled
        # (n_cand is a multiple of topk) to shorten the serial SMEM chain a
        # near-empty grid can't hide.
        i = cutlass.Int32(tid)
        while i < n_cand:
            ib = idx[i]
            kb = key[i]
            rank = cutlass.Int32(0)
            j = cutlass.Int32(0)
            while j < n_cand:
                k0 = key[j]
                k1 = key[j + 1]
                k2 = key[j + 2]
                k3 = key[j + 3]
                if (k0 < kb) or ((k0 == kb) and (idx[j] < ib)):
                    rank += 1
                if (k1 < kb) or ((k1 == kb) and (idx[j + 1] < ib)):
                    rank += 1
                if (k2 < kb) or ((k2 == kb) and (idx[j + 2] < ib)):
                    rank += 1
                if (k3 < kb) or ((k3 == kb) and (idx[j + 3] < ib)):
                    rank += 1
                j += 4
            if (ib != cutlass.Int32(_SENTINEL)) and (rank < target):
                sel[n_forced + rank] = ib
            i += _NTHREADS_MERGE
        cute.arch.barrier()

        # Parallel rank-order write: each of topk threads places one selected
        # index (sentinels tie on value, broken by slot, and become -1).
        if tid < self._topk:
            v = sel[tid]
            pos = cutlass.Int32(0)
            jj = cutlass.Int32(0)
            while jj < self._topk:
                u = sel[jj]
                if (u < v) or ((u == v) and (jj < tid)):
                    pos += 1
                jj += 1
            if v == cutlass.Int32(_SENTINEL):
                v = cutlass.Int32(-1)
            mOut[q, h, pos] = v
