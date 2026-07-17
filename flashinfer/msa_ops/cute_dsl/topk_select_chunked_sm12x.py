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

Chunked count-rank MSA top-K KV-block selection for SM120/SM121: independent
CTAs rank chunks of candidate blocks, then a second kernel merges the
survivors. Serves grids too small to fill the GPU, where the
single-CTA-per-row kernels serialize each row's whole scan, and full grids,
where reading the scores exactly once beats the radix kernel's pass per
stage and shortens the count-rank kernel's per-candidate scan.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .topk_select_radix_sm12x import _radix_key

# Per-chunk SMEM staging cap (same as the count-rank kernel's score cap).
_MAX_CHUNK_BLOCKS = 128
# The merge stages _MAX_CHUNKS * topk candidate slots in SMEM.
_MAX_CHUNKS = 16
# Target blocks per chunk: balances CTA parallelism against the merge's
# candidate count. The crossover was measured empirically.
_CHUNK_BLOCKS = 64
# Below two chunks the split is pure overhead over the single-CTA count-rank.
_MIN_CHUNKS = 2
# Dispatch floor: under this many middle blocks the second launch never pays
# for itself.
_MIN_BLOCKS = 32
# Partial-kernel boundary: at or below this many queries the grid needs
# row-per-CTA fan-out to fill the GPU; above it the q-tiled partial takes
# over for coalesced score reads.
_TILED_MIN_QUERIES = 2048
# Candidate-scratch ceiling (rows * num_chunks * topk * 8 bytes per call).
# Sized to cover the benchmarked range (32k-token prefill needs ~67MB) with
# margin; larger grids keep the allocation-free single-kernel paths so a
# memory-tight serving setup never sees a surprise multi-hundred-MB
# allocation from a top-k call.
_MAX_CHUNKED_SCRATCH_BYTES = 128 << 20
_NTHREADS_PARTIAL = 128
_NTHREADS_MERGE = 256
# Queries per CTA in the full-grid partial kernel: one warp lane per query
# turns the (H, P, S) score reads coalesced (S is the contiguous axis, so a
# single row's block scores are S floats apart).
_QTILE = 32
_SENTINEL = 0x7FFFFFFF  # INT32_MAX: empty slots sort to the tail, unlike -1
# Empty-slot key. Detection must use the index: a NaN score can legitimately
# produce this key.
_SENTINEL_KEY = 0xFFFFFFFF


class TopKSelectChunkedSm12x:
    """Two-phase (per-chunk rank + merge) top-K selection for small grids.
    Uses the count-rank kernel's (bit-key, block index) order, so the two
    produce identical selections."""

    def __init__(self, topk: int, tiled: bool = False):
        if topk != 16:
            raise ValueError(f"topk must be 16, got {topk}")
        self._topk = topk
        # Tiled = full-grid variant: the partial kernel covers _QTILE queries
        # per CTA for coalesced score reads. The row-per-CTA variant stays for
        # small grids, where those extra CTAs are the needed parallelism.
        self._tiled = tiled

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
        if cutlass.const_expr(self._tiled):
            self.kernel_partial_tiled(
                mBits,
                mKey,
                mCandIdx,
                num_valid_pages,
                force_begin,
                force_end,
                chunk_len,
                total_qo_len,
            ).launch(
                grid=(
                    (total_qo_len + _QTILE - 1) // _QTILE,
                    num_qo_heads,
                    num_chunks,
                ),
                block=(_NTHREADS_PARTIAL, 1, 1),
                stream=stream,
            )
        else:
            self.kernel_partial(
                mBits,
                mKey,
                mCandIdx,
                num_valid_pages,
                force_begin,
                force_end,
                chunk_len,
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

        # Tail chunks may be short or empty; their sentinel fill still runs.
        c_lo = force_begin + c * chunk_len
        c_hi = c_lo + chunk_len
        if c_hi > mid_hi:
            c_hi = mid_hi
        n_local = c_hi - c_lo
        # Pad to the unroll width; padded keys sit at indices above every real
        # block, so neither the compare nor the tie-break can count them.
        n_pad = (n_local + 3) & ~cutlass.Int32(3)
        cand_base = c * cutlass.Int32(self._topk)

        # Sentinel-fill the candidate slots; survivors overwrite them after
        # the barrier below.
        if tid < self._topk:
            mKey[q, h, cand_base + tid] = cutlass.Uint32(_SENTINEL_KEY)
            mIdx[q, h, cand_base + tid] = cutlass.Int32(_SENTINEL)

        # The bit keys preserve the radix-kernel order, with deterministic NaN
        # placement.
        l = cutlass.Int32(tid)
        while l < n_pad:
            k = cutlass.Uint32(_SENTINEL_KEY)
            if l < n_local:
                k = _radix_key(mBits[h, c_lo + l, q])
            score[l] = k
            l += _NTHREADS_PARTIAL
        cute.arch.barrier()

        # A block in the global top-target also ranks below target within its
        # chunk, so the per-chunk survivors are a superset of the final
        # selection. Ranks are distinct, so a survivor's rank is its slot (no
        # atomics). The 4-way unroll shortens the dependent-SMEM-load chain,
        # which these small grids have too few warps to hide.
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
    def kernel_partial_tiled(
        self,
        mBits: cute.Tensor,  # (H, P, S) uint32, raw bits of max_score
        mKey: cute.Tensor,  # (S, H, C*topk) uint32 candidate keys
        mIdx: cute.Tensor,  # (S, H, C*topk) int32 candidate block indices
        num_valid_pages: cutlass.Int32,
        force_begin: cutlass.Int32,
        force_end: cutlass.Int32,
        chunk_len: cutlass.Int32,
        total_qo_len: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        qb, h, c = cute.arch.block_idx()

        # Row-major (block, query) staging: a warp stages one block for 32
        # consecutive queries in a single coalesced line, and the rank scan
        # walks blocks down a bank-aligned column.
        @cute.struct
        class SharedStorage:
            score: cute.struct.MemRange[cutlass.Uint32, _MAX_CHUNK_BLOCKS * _QTILE]

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(SharedStorage)
        score = st.score.get_tensor(cute.make_layout(_MAX_CHUNK_BLOCKS * _QTILE))

        q0 = qb * _QTILE
        lane = tid % _QTILE
        grp = tid // _QTILE
        n_grp = _NTHREADS_PARTIAL // _QTILE
        q = q0 + lane
        q_ok = q < total_qo_len

        mid_hi = num_valid_pages - force_end
        n_forced = force_begin + force_end
        target = cutlass.Int32(self._topk) - n_forced

        c_lo = force_begin + c * chunk_len
        c_hi = c_lo + chunk_len
        if c_hi > mid_hi:
            c_hi = mid_hi
        n_local = c_hi - c_lo
        n_pad = (n_local + 3) & ~cutlass.Int32(3)
        cand_base = c * cutlass.Int32(self._topk)

        # Sentinel-fill every covered row's candidate slots; the barrier below
        # orders the fill before any thread's survivor stores.
        i = cutlass.Int32(tid)
        while i < cutlass.Int32(_QTILE * self._topk):
            qq = q0 + i // self._topk
            if qq < total_qo_len:
                mKey[qq, h, cand_base + i % self._topk] = cutlass.Uint32(_SENTINEL_KEY)
                mIdx[qq, h, cand_base + i % self._topk] = cutlass.Int32(_SENTINEL)
            i += _NTHREADS_PARTIAL

        # Out-of-range queries stage sentinels so no lane reads out of bounds.
        l = cutlass.Int32(grp)
        while l < n_pad:
            k = cutlass.Uint32(_SENTINEL_KEY)
            if (l < n_local) and q_ok:
                k = _radix_key(mBits[h, c_lo + l, q])
            score[l * _QTILE + lane] = k
            l += n_grp
        cute.arch.barrier()

        # Same rank-as-slot emit as the row-per-CTA kernel, per lane's query.
        l = cutlass.Int32(grp)
        while l < n_local:
            kb = score[l * _QTILE + lane]
            rank = cutlass.Int32(0)
            j = cutlass.Int32(0)
            while j < n_pad:
                k0 = score[j * _QTILE + lane]
                k1 = score[(j + 1) * _QTILE + lane]
                k2 = score[(j + 2) * _QTILE + lane]
                k3 = score[(j + 3) * _QTILE + lane]
                if (k0 < kb) or ((k0 == kb) and (j < l)):
                    rank += 1
                if (k1 < kb) or ((k1 == kb) and (j + 1 < l)):
                    rank += 1
                if (k2 < kb) or ((k2 == kb) and (j + 2 < l)):
                    rank += 1
                if (k3 < kb) or ((k3 == kb) and (j + 3 < l)):
                    rank += 1
                j += 4
            if q_ok and (rank < target):
                mKey[q, h, cand_base + rank] = kb
                mIdx[q, h, cand_base + rank] = c_lo + l
            l += n_grp

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

        # Rank-as-slot emit and unrolled scan as in the partial kernel, over
        # the union of survivors (n_cand is a multiple of the unroll width).
        # Empty slots need no ranker guard: a sentinel key and index can never
        # beat a real candidate.
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
