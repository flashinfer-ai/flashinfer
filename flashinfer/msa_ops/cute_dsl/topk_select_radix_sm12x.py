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

Faithful multi-stage radix-select MSA top-K KV-block selection for SM120/SM121.
CuTe-DSL port of the CUDA ``IndexerTopKWithSortKernel``
(``csrc/include/sparse_topk_select.cuh``), itself derived from TensorRT-LLM's
``indexerTopK.cu``. This is the sole CuTe-DSL top-k algorithm: it matches the
CUDA kernel's MSD-radix structure and complexity (O(max_k_tiles) per row), so no
separate fallback path is needed.

Algorithm (one CTA per (head, query) row, ``kThreads`` threads):

1. **Forced blocks** ``[0, fb)`` (sink) and ``[nvp - fe, nvp)`` (window) are
   pre-placed and excluded from ranking, so the radix ranks only the contiguous
   middle region ``[fb, nvp - fe)`` and picks the ``target = topk - fb - fe``
   highest scorers there.
2. MSD radix-select on the order-preserving float-bit key (ascending key ==
   descending score), 10 bits per stage:
     * stage 1: high 10 bits (``key >> 22``), ranks the whole middle region;
     * stage 2: mid 10 bits (``(key >> 12) & 0x3ff``), only stage-1 threshold-bin elements;
     * stage 3: low 10 bits (``(key >> 2) & 0x3ff``), only stage-2 threshold-bin elements.
   Each stage histograms its candidates, then a 2-level scan finds the threshold
   bin where the running count crosses the remaining ``need``: one warp sums the
   32 contiguous 32-bin groups, then thread 0 walks the 32 sums and refines the
   straddling group (a ~64-step critical path, not a 1024-step serial scan).
3. Classify each candidate: ``bin < threshold`` emits directly; ``bin ==
   threshold`` either stages it for a tie-break sort (if the bin fits
   ``_STAGE_CAP``) or defers it to the next finer stage.
4. When the threshold bin fits ``_STAGE_CAP``, insertion-sort the staged items by
   score (ties by staging index, as in CUDA) and emit the remaining ``need``.
   Stage 3 is terminal: a residual threshold bin holds true ties (equal in bits
   31..2), emitted directly up to ``need``, so the buffer never grows with
   ``max_k_tiles``.
5. Sort the selected indices ascending and write, ``-1``-padding empty slots.

Because the threshold bin shrinks by ~10 bits each stage and stage 3 emits
residual ties directly, the fixed ``_STAGE_CAP`` staging buffer is always
sufficient: there is no ``max_k_tiles`` ceiling from staging. The supported
range matches the CUDA dispatcher (``max_k_tiles < 12288``); the Python wrapper
raises above it.

Two output-equivalent simplifications vs the CUDA reference, validated bit-exact
on distinct-score inputs. First, no fp16 pre-pass: the three fp32 stages select
fully on their own (the fp16 pass is only a register micro-opt). Second, no
transpose workspace: ``max_score`` is read column-strided via a uint32 recast,
since the transpose only aids float4 coalescing. The final ascending-by-index
sort is a single-thread insertion sort over the <= topk slots (bit-identical to
the CUDA warp-bitonic sort at topk=16).
"""

import inspect

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm

# nvvm.atomicrmw's signature comes from the cutlass-dsl libs wheel, not the pinned DSL
# version: libs-base needs a leading ``res`` (result type) arg, libs-cu13 infers it.
_ATOMICRMW_NEEDS_RES = "res" in inspect.signature(nvvm.atomicrmw).parameters

_NUM_BINS = 1024  # 10-bit radix digit per stage
_STAGE_CAP = 2048  # threshold-bin staging capacity (per stage, not per row)
_KTHREADS = 256
_GROUP = 32  # bins per group for the 2-level threshold scan (_NUM_BINS / _GROUP groups)
_SENTINEL = 0x7FFFFFFF  # sorts to the tail; written out as -1

# Per-stage radix-digit shift of the 32-bit key: stage 1 bits 31..22, stage 2
# bits 21..12, stage 3 bits 11..2 (bits 0-1 dropped, they are true ties).
_STAGE_SHIFT = {1: 22, 2: 12, 3: 2}
# Shift to filter candidates to the previous stage's threshold bin (stage 1 ranks
# everything, so no filter).
_MATCH_SHIFT = {2: 22, 3: 12}


def _atomic_add_i32(a, ptr: cute.Pointer) -> cutlass.Int32:
    """int32 atomic add (SMEM/global generic ptr); returns the OLD value."""
    av = cutlass.Int32(a).ir_value()
    if _ATOMICRMW_NEEDS_RES:
        return nvvm.atomicrmw(
            res=av.type, op=nvvm.AtomicOpKind.ADD, ptr=ptr.llvm_ptr, a=av
        )
    return nvvm.atomicrmw(op=nvvm.AtomicOpKind.ADD, ptr=ptr.llvm_ptr, a=av)


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
        mBits: cute.Tensor,  # (H, P, S) uint32, raw bits of max_score
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
            pat: cute.struct.MemRange[cutlass.Uint32, 1]

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(SharedStorage)
        hist = st.hist.get_tensor(cute.make_layout(_NUM_BINS))
        grpsum = st.grpsum.get_tensor(cute.make_layout(_NUM_BINS // _GROUP))
        stage_key = st.stage_key.get_tensor(cute.make_layout(_STAGE_CAP))
        stage_idx = st.stage_idx.get_tensor(cute.make_layout(_STAGE_CAP))
        sel = st.sel.get_tensor(cute.make_layout(16))
        scal = st.scal.get_tensor(cute.make_layout(8))
        pat = st.pat.get_tensor(cute.make_layout(1))
        # scal slots: 0=found, 1=stage_count, 2=threshold, 3=base,
        #             4=finalBinSize, 5=done, 6=need

        nvp = num_valid_pages
        mid_lo = force_begin
        mid_hi = num_valid_pages - force_end
        n_forced = force_begin + force_end
        target = cutlass.Int32(self._topk) - n_forced

        n_groups = _NUM_BINS // _GROUP

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
            pat[0] = cutlass.Uint32(0)  # accumulated radix pattern
            # If every slot is forced (target <= 0) there is no middle to rank.
            scal[5] = (
                cutlass.Int32(0) if target > cutlass.Int32(0) else cutlass.Int32(1)
            )
        cute.arch.barrier()

        # ---- MSD radix-select stages (high -> mid -> low 10 bits) -----------
        # The Python loop is unrolled at trace time, so `step` is a constant and
        # the per-stage shift / pattern logic specializes statically. `done` is
        # block-uniform (thread 0 writes it, all read after a barrier), so every
        # `if dn == 0:` guard is uniform and the unconditional barriers between
        # guarded blocks stay balanced across the CTA.
        for step in (1, 2, 3):
            shift = _STAGE_SHIFT[step]

            # --- setup: refine pattern, reset stage_count, recompute need ----
            if tid == 0:
                if scal[5] == cutlass.Int32(0):
                    if cutlass.const_expr(step == 2):
                        pat[0] = cutlass.Uint32(scal[2]) << cutlass.Uint32(22)
                    elif cutlass.const_expr(step == 3):
                        pat[0] = pat[0] | (
                            cutlass.Uint32(scal[2]) << cutlass.Uint32(12)
                        )
                    scal[1] = cutlass.Int32(0)  # stage_count
                    scal[6] = target - scal[0]  # need (remaining to select)
            cute.arch.barrier()

            pattern = pat[0]

            # --- clear histogram ---------------------------------------------
            dn = scal[5]
            if dn == cutlass.Int32(0):
                bb = tid
                while bb < _NUM_BINS:
                    hist[bb] = cutlass.Int32(0)
                    bb += _KTHREADS
            cute.arch.barrier()

            # --- histogram over the pattern-matched candidates ---------------
            dn = scal[5]
            if dn == cutlass.Int32(0):
                b = mid_lo + tid
                while b < mid_hi:
                    okey = self._radix_key(mBits[h, b, q])
                    binv = cutlass.Int32((okey >> shift) & cutlass.Uint32(0x3FF))
                    if cutlass.const_expr(step == 1):
                        _atomic_add_i32(1, hist.iterator + binv)
                    else:
                        mshift = _MATCH_SHIFT[step]
                        if (
                            (okey ^ pattern) >> cutlass.Uint32(mshift)
                        ) == cutlass.Uint32(0):
                            _atomic_add_i32(1, hist.iterator + binv)
                    b += _KTHREADS
            cute.arch.barrier()

            # --- 2-level threshold scan ---
            # Phase A: first n_groups threads each sum one contiguous _GROUP-bin
            # group. Phase B/C: thread 0 finds the group holding the threshold and
            # refines within it. Critical serial path ~ 2*_GROUP, not _NUM_BINS.
            dn = scal[5]
            if dn == cutlass.Int32(0):
                if tid < n_groups:
                    gs = cutlass.Int32(0)
                    gi = cutlass.Int32(0)
                    gbase = tid * _GROUP
                    while gi < _GROUP:
                        gs += hist[gbase + gi]
                        gi += 1
                    grpsum[tid] = gs
            cute.arch.barrier()

            dn = scal[5]
            if dn == cutlass.Int32(0):
                if tid == 0:
                    need = scal[6]
                    # Phase B: find the group whose running count crosses `need`.
                    running = cutlass.Int32(0)
                    grp = cutlass.Int32(n_groups)
                    base_grp = cutlass.Int32(0)
                    gg = cutlass.Int32(0)
                    while gg < n_groups:
                        s = grpsum[gg]
                        if (
                            (grp == n_groups)
                            and (running < need)
                            and (running + s >= need)
                        ):
                            grp = gg
                            base_grp = running
                        running += s
                        gg += 1
                    # Default (need unreachable): all candidates selected.
                    threshold = cutlass.Int32(_NUM_BINS)
                    base = running
                    fbs = cutlass.Int32(0)
                    # Phase C: refine within the found group.
                    if grp < n_groups:
                        run2 = base_grp
                        bi = grp * _GROUP
                        end = bi + _GROUP
                        while bi < end:
                            c = hist[bi]
                            if (
                                (threshold == _NUM_BINS)
                                and (run2 < need)
                                and (run2 + c >= need)
                            ):
                                threshold = bi
                                base = run2
                                fbs = c
                            run2 += c
                            bi += 1
                    scal[2] = threshold
                    scal[3] = base
                    scal[4] = fbs
                    if cutlass.const_expr(step == 3):
                        # Terminal stage emits sub-threshold and threshold-bin
                        # candidates concurrently, so they need disjoint output
                        # ranges: sub-threshold via `found` [found, found+base),
                        # threshold-bin via this counter [found+base, target).
                        scal[1] = scal[0] + base
            cute.arch.barrier()

            # --- classify: direct-emit (bin<threshold) / stage / refine ------
            dn = scal[5]
            if dn == cutlass.Int32(0):
                threshold = scal[2]
                fbs = scal[4]
                fits = fbs <= cutlass.Int32(_STAGE_CAP)
                b = mid_lo + tid
                while b < mid_hi:
                    okey = self._radix_key(mBits[h, b, q])
                    # `proceed` is always a dynamic (uniform-true for stage 1)
                    # bool so the body below traces once, not per static branch.
                    if cutlass.const_expr(step == 1):
                        proceed = okey == okey
                    else:
                        mshift = _MATCH_SHIFT[step]
                        proceed = (
                            (okey ^ pattern) >> cutlass.Uint32(mshift)
                        ) == cutlass.Uint32(0)
                    if proceed:
                        binv = cutlass.Int32((okey >> shift) & cutlass.Uint32(0x3FF))
                        if binv < threshold:
                            slot = _atomic_add_i32(1, scal.iterator + 0)
                            if n_forced + slot < cutlass.Int32(self._topk):
                                sel[n_forced + slot] = b
                        elif binv == threshold:
                            if cutlass.const_expr(step == 3):
                                # Terminal stage: residual ties (equal in bits
                                # 31..2) emit directly via the separate counter
                                # (slot 1), capped at the topk slots.
                                slot = _atomic_add_i32(1, scal.iterator + 1)
                                if n_forced + slot < cutlass.Int32(self._topk):
                                    sel[n_forced + slot] = b
                            else:
                                if fits:
                                    s = _atomic_add_i32(1, scal.iterator + 1)
                                    if s < cutlass.Int32(_STAGE_CAP):
                                        stage_key[s] = okey
                                        stage_idx[s] = b
                    b += _KTHREADS
            cute.arch.barrier()

            if cutlass.const_expr(step != 3):
                # --- insertion-sort the staged threshold-bin items, emit rest
                dn = scal[5]
                if dn == cutlass.Int32(0):
                    fbs = scal[4]
                    if fbs <= cutlass.Int32(_STAGE_CAP):
                        stage_count = scal[1]
                        base = scal[3]
                        need = scal[6]
                        ii = tid
                        while ii < stage_count:
                            ti = stage_key[ii]
                            rank = cutlass.Int32(0)
                            jj = cutlass.Int32(0)
                            while jj < stage_count:
                                tj = stage_key[jj]
                                # higher score == smaller key; ties by stage index
                                if (ti > tj) or ((ti == tj) and (ii < jj)):
                                    rank += 1
                                jj += 1
                            if base + rank < need:
                                slot = _atomic_add_i32(1, scal.iterator + 0)
                                if n_forced + slot < cutlass.Int32(self._topk):
                                    sel[n_forced + slot] = stage_idx[ii]
                            ii += _KTHREADS
                cute.arch.barrier()

                # Mark done if this stage's threshold bin fit the buffer. Thread 0
                # only: guarding all threads on scal[5] would race this write
                # (read vs write of the same slot, no barrier between).
                if tid == 0:
                    if scal[5] == cutlass.Int32(0):
                        if scal[4] <= cutlass.Int32(_STAGE_CAP):
                            scal[5] = cutlass.Int32(1)
                cute.arch.barrier()
            else:
                # stage 3 is terminal regardless of bin size
                if tid == 0:
                    scal[5] = cutlass.Int32(1)
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
