"""Fused exact fallback for the walk-first selector (substrate, not a backend).

WalkFirstTopK's last-arriver epilogue inlines ``_fallback_row`` whenever a
row cannot be finished from its candidate slab (candidate overflow, a slot
mismatch, duplicate floods): the row is re-solved from scratch, exactly, on
device, inside the same kernel.  No host status readback (a device->host
sync every call, and a CUDA-graph capture breaker) and no second launch --
the always-launched gate kernel this replaced measured a flat ~1.65us of
pure launch latency.  The whole call stays sync-free and CUDA-graph
capturable.

Algorithm for a failed row: MSD radix-select with 12-bit digits over the
ordered fp32 key space (the DIGIT family -- certainty by construction):

  1. up to 3 histogram passes: 4096 shift-binned key buckets over the
     current candidate range, descending crossing-find at the remaining
     rank, recurse into the crossing bucket.  32-bit keys / 12 bits per
     round means round 3's buckets are single keys, so the loop PROVABLY
     terminates at either a bucket of <= GCAP candidates or a width-1
     (single-key) bucket -- no data-dependent escape hatch needed.
  2. one harvest pass with KEY-space compares (total order, so +/-inf and
     NaN are unambiguous; NaN keys sort above +inf, matching the torch
     ordering the other backends produce): key > bucket_hi -> emit via
     ticket cursor, key in bucket -> (key, idx) into the gmem slab.
  3. finish: single-key bucket -> fill by count (ties value-equal);
     else exact byte-radix rank-select over the <= GCAP slab entries
     (``_slab_select``).

Worst case is 4 row passes, data-independent.  Requires length > top_k,
k <= TIE_CAP, and a whole CTA executing under a CTA-uniform condition
(block barriers inside).

Nothing here compiles or launches on its own: GatedExactFallback is mixed
into ProdWalkFirstTopK (walkfirst_topk_primitives.py) and both methods run
only inline.  GCAP -- the per-row gmem candidate-slab capacity the harvest
pass stores into -- lives here because the walk-first slab row layout
(WF_ROW_INTS) and the dispatcher's slab allocation are sized from it.
"""

import cutlass
import cutlass.cute as cute

from .radix_topk_primitives import CoarseHistTopKPrimitivesKernel, smem_atomic_add

GCAP = 16384  # per-row gmem candidate-slab capacity (gvr_2's GCAP)


class GatedExactFallback(CoarseHistTopKPrimitivesKernel):
    """See module docstring.  Reuses the substrate's scan, histogram-find
    and key helpers; the row re-solve and the slab select live here."""

    @cute.jit
    def _fallback_row(
        self,
        row_in,
        length,
        out_idx_row,
        slab_k,
        slab_i,
        s_h4k,
        s_h256,
        s_warp_sums,
        s_misc,
        s_count_gt,
        s_count_eq,
        tidx,
    ):
        """Exact MSD radix-select re-solve of one failed row (see module
        docstring).  Requires length > top_k and a whole CTA executing
        under a CTA-uniform condition (block barriers inside).  s_h4k
        needs hist_size ints (4096 for fp32, 8192 for the 16-bit dtypes)
        -- callers may alias any dead smem array of at least that size
        (the fused walk-first epilogue passes its LCAP+4 candidate stage).
        Inlined by wf_topk_kernel's failure paths so a failed row costs no
        second kernel launch."""
        top_k = cutlass.const_expr(self.top_k)
        # ---- MSD radix-select: <= 3 key-histogram passes ----
        lo_k = cutlass.Uint32(0)
        hi_k = cutlass.Uint32(0xFFFFFFFF)
        need = cutlass.Int32(top_k)  # descending rank within range
        done = cutlass.Int32(0)
        for _round in cutlass.range_constexpr(3):
            if done == 0:  # block-uniform: barriers inside are safe
                # Histogram geometry MUST follow the substrate's hist_size:
                # find_threshold_wide scans hist_items = hist_size / nt bins
                # per thread (fp32: 4096 bins; 16-bit dtypes: 8192).  A
                # hardcoded 4096 here left bins 4096..8191 un-zeroed for
                # 16-bit rows -- and s_h4k aliases the walk's candidate
                # stage, so any row that staged > 4096 survivors fed the
                # find stale survivor VALUES as counts (garbage crossing,
                # empty harvest, zero-filled output).
                for zz in cutlass.range_constexpr(self.hist_size // self.nt):
                    s_h4k[tidx + cutlass.Int32(zz * self.nt)] = cutlass.Int32(0)
                cute.arch.barrier()
                # shift binning, max bin index bounded (span=2^k+1
                # would otherwise write one past the histogram)
                span = cutlass.Int64(hi_k - lo_k) + cutlass.Int64(1)
                shift = cutlass.Uint32(0)
                spn = span - cutlass.Int64(1)
                while spn > cutlass.Int64(self.hist_size - 1):
                    spn = spn >> 1
                    shift = shift + 1
                # DSL signedness landmine: lo_k/hi_k are loop-carried
                # across dynamic regions and get re-wrapped SIGNED, so
                # `kk <= hi_k` with hi_k = 0xFFFFFFFF lowered as `kk <= -1`
                # -- true only for keys with the top bit set (fp32 keys of
                # POSITIVE floats), silently dropping every fp16/bf16 key
                # and every fp32 negative-float key from the histogram
                # (nothing published, stale s_misc, garbage bucket).
                # Re-assert unsignedness at every use site.
                # Range test + bin in Int64: unsigned 32-bit compares on
                # loop-carried keys were STILL lowered signed in some
                # kernel instantiations even after Uint32 re-assertion
                # (S==1 fp16 landed on bucket 4095); Int64 zero-extension
                # is proven reliable here (the shift derivation above
                # depends on it), so the compares become unambiguous.
                lo64 = cutlass.Int64(cutlass.Uint32(lo_k))
                span64 = cutlass.Int64(cutlass.Uint32(hi_k)) - lo64
                sh64 = cutlass.Int64(shift)
                # unroll=1: the compiler bounds this loop by the row width and
                # unrolled it N / nt times per round per inlined copy (three
                # copies in the walk-first kernel) -- the 64K binary carried
                # ~700 SASS lines of this cold path; a rolled loop costs the
                # flood rows nothing measurable
                for i in cutlass.range(tidx, length, self.nt, unroll=1):
                    d = (
                        cutlass.Int64(
                            cutlass.Uint32(self.exact_key(self.load_scalar(row_in, i)))
                        )
                        - lo64
                    )
                    if d >= cutlass.Int64(0):
                        if d <= span64:
                            bb = cutlass.Int32(d >> sh64)
                            smem_atomic_add(s_h4k + bb, 1)
                cute.arch.barrier()
                # descending crossing at rank ``need`` (ends with a
                # block barrier; publishes bin/above/cnt)
                self.find_threshold_wide(
                    s_h4k,
                    cutlass.Int32(0),  # advisory; walk uses own sum
                    need,
                    s_warp_sums,
                    s_misc,
                    tidx,
                )
                bkt = s_misc[0]
                above = s_misc[1]
                cnt = s_misc[2]
                cute.arch.barrier()
                lo_k2 = lo_k + cutlass.Uint32(
                    cutlass.Int64(bkt) << cutlass.Int64(shift)
                )
                hi_k2 = lo_k + cutlass.Uint32(
                    ((cutlass.Int64(bkt) + 1) << cutlass.Int64(shift))
                    - cutlass.Int64(1)
                )
                need = need - above
                lo_k = lo_k2
                hi_k = hi_k2
                if cnt <= GCAP:
                    done = cutlass.Int32(1)
                if lo_k == hi_k:  # single key: fill-by-count cures it
                    done = cutlass.Int32(1)
        # round 3 buckets are single keys (32-bit key, 12+12+8 bits),
        # so done == 1 here unconditionally

        # ---- harvest pass: KEY-space compares (NaN-unambiguous) ----
        if tidx == 0:
            s_count_gt[0] = cutlass.Int32(0)
            s_count_eq[0] = cutlass.Int32(0)
        cute.arch.barrier()
        lo64 = cutlass.Int64(cutlass.Uint32(lo_k))
        hi64 = cutlass.Int64(cutlass.Uint32(hi_k))
        for i in cutlass.range(tidx, length, self.nt, unroll=1):  # see the round loop
            kk = cutlass.Uint32(self.exact_key(self.load_scalar(row_in, i)))
            k64 = cutlass.Int64(kk)
            if k64 > hi64:
                pos = smem_atomic_add(s_count_gt, 1)
                if pos < top_k:
                    out_idx_row[pos] = cutlass.Int32(i)
            else:
                if k64 >= lo64:
                    c = smem_atomic_add(s_count_eq, 1)
                    if c < GCAP:
                        slab_k[c] = kk.bitcast(cutlass.Int32)
                        slab_i[c] = cutlass.Int32(i)
        cute.arch.barrier()

        # ---- finish ----
        gt = s_count_gt[0]
        eq = s_count_eq[0]
        remaining = top_k - gt
        if lo_k == hi_k:
            # single-key bucket: candidates are key-equal, any
            # subset is exact; remaining <= k <= GCAP <= stored
            for t in range(tidx, remaining, self.nt):
                out_idx_row[gt + t] = slab_i[t]
        else:
            # cnt <= GCAP guaranteed by the loop's exit condition
            self._slab_select(
                slab_k,
                slab_i,
                eq,
                gt,
                remaining,
                s_h256,
                s_warp_sums,
                s_misc,
                out_idx_row,
                tidx,
            )

    @cute.jit
    def _slab_select(
        self,
        slab_k,
        slab_i,
        eq,
        gt,
        remaining,
        s_h256,
        s_warp_sums,
        s_misc,
        out_idx_row,
        tidx,
    ):
        """Exact selection of ``remaining`` winners among the eq slab
        candidates: 4-round byte-radix rank-find of the pivot key at
        descending rank ``remaining``, then one emit pass over the slab
        (strictly-above -> ticketed slots right after the gt block;
        pivot-equal fill the rest by count -- equal keys, any subset
        exact)."""
        prefix = cutlass.Uint32(0)
        pmask = cutlass.Uint32(0)
        need = cutlass.Int32(remaining)
        total = cutlass.Int32(eq)
        for r_ in cutlass.range_constexpr(4):
            sh = cutlass.const_expr(24 - 8 * r_)
            if tidx < 256:
                s_h256[tidx] = cutlass.Int32(0)
            cute.arch.barrier()
            for i in range(tidx, eq, self.nt):
                kk = cutlass.Uint32(slab_k[i])
                if (kk & pmask) == prefix:
                    smem_atomic_add(
                        s_h256
                        + cutlass.Int32(
                            (kk >> cutlass.Uint32(sh)) & cutlass.Uint32(0xFF)
                        ),
                        1,
                    )
            cute.arch.barrier()
            self.scan256_and_find(s_h256, total, need, s_warp_sums, s_misc, tidx)
            bucket = s_misc[0]
            above = s_misc[1]
            cnt = s_misc[2]
            cute.arch.barrier()
            prefix = prefix | (cutlass.Uint32(bucket) << cutlass.Uint32(sh))
            pmask = pmask | (cutlass.Uint32(0xFF) << cutlass.Uint32(sh))
            need = need - above
            total = cnt
        # winners strictly above the pivot take (remaining - need) slots
        # right after the gt block; pivot-equal candidates fill the last
        # ``need`` by count.
        if tidx == 0:
            s_misc[10] = cutlass.Int32(0)
            s_misc[11] = cutlass.Int32(0)
        cute.arch.barrier()
        nab = remaining - need
        pfx64 = cutlass.Int64(cutlass.Uint32(prefix))  # unambiguous compare
        for i in range(tidx, eq, self.nt):
            kk = cutlass.Uint32(slab_k[i])
            if cutlass.Int64(kk) > pfx64:
                p = smem_atomic_add(s_misc + 10, 1)
                if p < nab:  # bounds the store even if a round disagreed
                    out_idx_row[gt + p] = slab_i[i]
            else:
                if cutlass.Int64(kk) == pfx64:
                    e = smem_atomic_add(s_misc + 11, 1)
                    if e < need:
                        out_idx_row[gt + nab + e] = slab_i[i]
