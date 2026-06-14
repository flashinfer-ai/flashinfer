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

MSA top-K KV-block selection for SM120/SM121 (CuTe-DSL port of
``csrc/sparse_topk_select.cu(h)``).

For each (query head, query token) row of ``max_score`` (the per-128-block proxy
maxima), selects the ``topk`` most important KV blocks and returns their
**ascending** indices, with these MSA semantics:

* ``force_begin_blocks`` sink blocks ``[0, fb)`` and ``force_end_blocks`` local
  window blocks ``[nvp - fe, nvp)`` are always included (within the topk
  budget);
* the remaining ``topk - fb - fe`` slots are the top scorers of the middle
  region ``[fb, nvp - fe)``;
* only indices ``< num_valid_pages`` are eligible; rows with fewer than ``topk``
  valid blocks are ``-1`` tail-padded.

One warp handles one (head, query) row. The remaining slots are filled by
iterative warp arg-max (k passes, k = topk - fb - fe <= 16), which gives a
deterministic lowest-index-on-tie order matching the CUDA reference for the
distinct-score inputs MSA produces. ``max_score`` is read column-strided
directly, so no transpose / workspace tensor is needed.

This is selection (not the latency-critical path); a histogram or bitonic core
would cut the O(k * max_k_tiles) scan to O(max_k_tiles) if it ever matters.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class TopKSelectSm12x:
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
        self.kernel(mMaxScore, mOut, num_valid_pages, force_begin, force_end).launch(
            grid=(total_qo_len, num_qo_heads, 1),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _warp_argmax(self, score, idx):
        """Reduce (score, idx) across the warp: max score, lowest idx on tie."""
        for off in cutlass.range_constexpr(5):
            shift = 1 << (4 - off)
            o_s = cute.arch.shuffle_sync_bfly(
                score, offset=shift, mask=-1, mask_and_clamp=31
            )
            o_i = cute.arch.shuffle_sync_bfly(
                idx, offset=shift, mask=-1, mask_and_clamp=31
            )
            take = (o_s > score) or ((o_s == score) and (o_i < idx))
            if take:
                score = o_s
                idx = o_i
        return score, idx

    @cute.kernel
    def kernel(
        self,
        mMaxScore: cute.Tensor,
        mOut: cute.Tensor,
        num_valid_pages: cutlass.Int32,
        force_begin: cutlass.Int32,
        force_end: cutlass.Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        q, h, _ = cute.arch.block_idx()

        @cute.struct
        class SharedStorage:
            sel: cute.struct.MemRange[cutlass.Int32, self._topk]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sel = storage.sel.get_tensor(cute.make_layout(self._topk))

        nvp = num_valid_pages
        mid_lo = force_begin
        mid_hi = num_valid_pages - force_end
        n_forced = force_begin + force_end
        n_rest = cutlass.Int32(self._topk) - n_forced

        neg_inf = -cutlass.Float32.inf

        # lane 0 seeds the forced (sink + window) indices, ascending regions.
        if lane == 0:
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
        cute.arch.sync_warp()

        # iterative warp arg-max over the middle region, excluding picks so far.
        picked = cutlass.Int32(0)
        for it in cutlass.range_constexpr(self._topk):
            if it < n_rest:
                best_s = neg_inf
                best_i = cutlass.Int32(-1)
                b = mid_lo + lane
                while b < mid_hi:
                    s = mMaxScore[h, b, q]
                    # exclude blocks already chosen this row (middle picks only)
                    dup = False
                    for j in cutlass.range_constexpr(self._topk):
                        if j < picked:
                            if sel[n_forced + j] == b:
                                dup = True
                    if not dup:
                        if (s > best_s) or ((s == best_s) and (b < best_i)):
                            best_s = s
                            best_i = b
                    b += 32
                red_s, red_i = self._warp_argmax(best_s, best_i)
                if lane == 0:
                    sel[n_forced + picked] = red_i
                cute.arch.sync_warp()
                # red_i == -1 means the middle region is exhausted.
                if red_i >= 0:
                    picked += 1

        total = n_forced + picked

        # lane 0 sorts the (<= topk) selected indices ascending and writes out,
        # padding the tail with -1.
        if lane == 0:
            a = cutlass.Int32(1)
            while a < total:
                key = sel[a]
                b2 = a - 1
                while b2 >= 0 and sel[b2] > key:
                    sel[b2 + 1] = sel[b2]
                    b2 -= 1
                sel[b2 + 1] = key
                a += 1
            k = cutlass.Int32(0)
            while k < self._topk:
                v = sel[k] if k < total else cutlass.Int32(-1)
                mOut[q, h, k] = v
                k += 1
