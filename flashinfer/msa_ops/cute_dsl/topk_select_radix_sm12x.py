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

Faithful radix-histogram MSA top-K KV-block selection for SM120/SM121 — the
complexity-parity replacement for the O(topk * max_k_tiles) iterative-argmax
port in ``topk_select_sm12x.py``. CuTe-DSL port of the CUDA
``IndexerTopKWithSortKernel`` (``csrc/include/sparse_topk_select.cuh``), itself
derived from TensorRT-LLM's ``indexerTopK.cu``.

Algorithm (one CTA per (head, query) row, ``kThreads`` threads):

1. **Forced blocks** ``[0, fb)`` (sink) and ``[nvp - fe, nvp)`` (window) are
   pre-placed into the selection and excluded from ranking — so the histogram
   ranks only the contiguous middle region ``[fb, nvp - fe)`` and picks the
   ``n_rest = topk - fb - fe`` highest scorers there.
2. **Radix histogram** (single fp32 pass, 1024 bins = high 10 bits of the
   order-preserving float-bit transform): each thread bins its slice of the
   middle region with SMEM atomics. A **2-level scan** then finds the threshold
   bin where the running count crosses ``n_rest`` (replacing ``cub::BlockScan``):
   one warp sums the 32 contiguous 32-bin groups in parallel, then thread 0
   walks the 32 group sums and refines within the one group that straddles the
   threshold — a ~64-step critical path instead of a 1024-step serial scan
   (which profiled as the dominant cost of an earlier single-thread version).
3. **Classify**: ``bin < threshold`` ⇒ emit directly; ``bin == threshold`` ⇒
   stage (key, index) for a tie-break sort.
4. **Insertion sort** the staged threshold-bin items by score (ties by staging
   index, matching the CUDA reference) and emit the remaining ``n_rest - base``.
5. Sort the (≤ topk) selected indices **ascending by index** and write, ``-1``
   tail-padding empty slots; indices ``>= num_valid_pages`` never enter the
   ranking (the middle region ends at ``nvp - fe``).

Because the staging buffer (``_STAGE_CAP``) holds the whole threshold bin, the
CUDA fp16 stage-0 + fp32 stages 1-3 collapse to this single exact pass whenever
``max_k_tiles <= _STAGE_CAP`` (true for MSA's <=128K-context envelope). The
``max_score`` input is read column-strided directly (its raw bits via a uint32
recast), so no transpose / workspace tensor is needed.

The final ascending-by-index sort is a single-thread insertion sort over the
<= topk selected slots (vs the CUDA reference's warp bitonic sort): for topk=16
the result is identical and the cost negligible, and it avoids a separate
bitonic primitive.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm

_NUM_BINS = 1024  # high 10 bits of the float-bit radix key
_RADIX_SHIFT = 22  # 32 - 10
_STAGE_CAP = 2048  # threshold-bin staging capacity (== max supported max_k_tiles)
_KTHREADS = 256
_GROUP = 32  # bins per group for the 2-level threshold scan (_NUM_BINS / _GROUP groups)
_SENTINEL = 0x7FFFFFFF  # sorts to the tail; written out as -1


def _atomic_add_i32(a, ptr: cute.Pointer) -> cutlass.Int32:
    """int32 atomic add (SMEM/global generic ptr); returns the OLD value."""
    return nvvm.atomicrmw(
        op=nvvm.AtomicOpKind.ADD, ptr=ptr.llvm_ptr, a=cutlass.Int32(a).ir_value()
    )


class TopKSelectRadixSm12x:
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
        max_k_tiles: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mBits = cute.recast_tensor(mMaxScore, cutlass.Uint32)
        self.kernel(
            mBits, mOut, num_valid_pages, force_begin, force_end, max_k_tiles
        ).launch(
            grid=(total_qo_len, num_qo_heads, 1),
            block=(_KTHREADS, 1, 1),
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
        mBits: cute.Tensor,  # (H, P, S) uint32 — raw bits of max_score
        mOut: cute.Tensor,  # (S, H, topk) int32
        num_valid_pages: cutlass.Int32,
        force_begin: cutlass.Int32,
        force_end: cutlass.Int32,
        max_k_tiles: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        q, h, _ = cute.arch.block_idx()

        @cute.struct
        class SharedStorage:
            hist: cute.struct.MemRange[cutlass.Int32, _NUM_BINS]
            grpsum: cute.struct.MemRange[cutlass.Int32, _NUM_BINS // _GROUP]
            stage_key: cute.struct.MemRange[cutlass.Uint32, _STAGE_CAP]
            stage_idx: cute.struct.MemRange[cutlass.Int32, _STAGE_CAP]
            sel: cute.struct.MemRange[cutlass.Int32, 16]
            scal: cute.struct.MemRange[cutlass.Int32, 8]

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(SharedStorage)
        hist = st.hist.get_tensor(cute.make_layout(_NUM_BINS))
        grpsum = st.grpsum.get_tensor(cute.make_layout(_NUM_BINS // _GROUP))
        stage_key = st.stage_key.get_tensor(cute.make_layout(_STAGE_CAP))
        stage_idx = st.stage_idx.get_tensor(cute.make_layout(_STAGE_CAP))
        sel = st.sel.get_tensor(cute.make_layout(16))
        scal = st.scal.get_tensor(cute.make_layout(8))
        # scal slots: 0=found, 1=stage_count, 2=threshold, 3=base

        nvp = num_valid_pages
        mid_lo = force_begin
        mid_hi = num_valid_pages - force_end
        n_forced = force_begin + force_end
        target = cutlass.Int32(self._topk) - n_forced

        # clear histogram (strided over the block)
        bb = tid
        while bb < _NUM_BINS:
            hist[bb] = cutlass.Int32(0)
            bb += _KTHREADS

        # thread 0: seed forced indices + sentinel-fill the rest + init counters
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
            scal[0] = cutlass.Int32(0)  # found
            scal[1] = cutlass.Int32(0)  # stage_count
        cute.arch.barrier()

        # ---- histogram over the middle region [mid_lo, mid_hi) --------------
        b = mid_lo + tid
        while b < mid_hi:
            key = self._radix_key(mBits[h, b, q])
            binv = cutlass.Int32(key >> _RADIX_SHIFT)
            _atomic_add_i32(1, hist.iterator + binv)
            b += _KTHREADS
        cute.arch.barrier()

        # ---- 2-level threshold scan (replaces a 1024-iter serial scan) ------
        # Phase A: first _NUM_BINS/_GROUP threads each sum one contiguous group
        # of _GROUP bins (parallel). Phase B/C: thread 0 walks the (small) group
        # sums to find the group holding the threshold, then refines within it.
        # Critical serial path ~ 2*_GROUP instead of _NUM_BINS.
        n_groups = _NUM_BINS // _GROUP
        if tid < n_groups:
            gs = cutlass.Int32(0)
            gi = cutlass.Int32(0)
            gbase = tid * _GROUP
            while gi < _GROUP:
                gs += hist[gbase + gi]
                gi += 1
            grpsum[tid] = gs
        cute.arch.barrier()

        if tid == 0:
            # Phase B: find the group whose running count crosses `target`.
            running = cutlass.Int32(0)
            grp = cutlass.Int32(n_groups)
            base_grp = cutlass.Int32(0)
            gg = cutlass.Int32(0)
            while gg < n_groups:
                s = grpsum[gg]
                if (grp == n_groups) and (running < target) and (running + s >= target):
                    grp = gg
                    base_grp = running
                running += s
                gg += 1
            # Default (target unreachable): all bins selected, base = total count.
            threshold = cutlass.Int32(_NUM_BINS)
            base = running
            # Phase C: refine within the found group.
            if grp < n_groups:
                run2 = base_grp
                bi = grp * _GROUP
                end = bi + _GROUP
                while bi < end:
                    c = hist[bi]
                    if (
                        (threshold == _NUM_BINS)
                        and (run2 < target)
                        and (run2 + c >= target)
                    ):
                        threshold = bi
                        base = run2
                    run2 += c
                    bi += 1
            scal[2] = threshold
            scal[3] = base
        cute.arch.barrier()

        threshold = scal[2]
        base = scal[3]
        do_middle = target > cutlass.Int32(0)

        # ---- classify: direct-emit (bin<threshold) or stage (bin==threshold)-
        if do_middle:
            b = mid_lo + tid
            while b < mid_hi:
                key = self._radix_key(mBits[h, b, q])
                binv = cutlass.Int32(key >> _RADIX_SHIFT)
                if binv < threshold:
                    slot = _atomic_add_i32(1, scal.iterator + 0)
                    sel[n_forced + slot] = b
                elif binv == threshold:
                    s = _atomic_add_i32(1, scal.iterator + 1)
                    if s < _STAGE_CAP:
                        stage_key[s] = key
                        stage_idx[s] = b
                b += _KTHREADS
        cute.arch.barrier()

        # ---- insertion-sort the staged threshold-bin items, emit the rest ---
        if do_middle:
            stage_count = scal[1]
            ii = tid
            while ii < stage_count:
                ti = stage_key[ii]
                rank = cutlass.Int32(0)
                jj = cutlass.Int32(0)
                while jj < stage_count:
                    tj = stage_key[jj]
                    # higher score == smaller key; ties broken by staging index
                    if (ti > tj) or ((ti == tj) and (ii < jj)):
                        rank += 1
                    jj += 1
                if base + rank < target:
                    sel[n_forced + base + rank] = stage_idx[ii]
                ii += _KTHREADS
        cute.arch.barrier()

        # ---- thread 0: ascending-by-index sort (<= topk slots) + write ------
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
