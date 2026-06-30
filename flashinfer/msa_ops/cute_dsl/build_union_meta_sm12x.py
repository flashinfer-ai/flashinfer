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

On-device union-tile metadata builder for SM120/SM121.

Production replacement for the host Python reference
(:func:`..._union_metadata.build_msa_union_metadata`): for each
(query-tile, kv-head) work item it forms the **union** of the KV blocks the
tile's ``tokens_per_tile`` queries selected, together with a per-(union-block)
``tokens_per_tile``-bit membership mask (bit ``i`` set iff tile-token ``i``
selected that block). The union-tile forward (:mod:`sparse_fwd_union_sm12x`)
then walks this list with online softmax.

One warp owns one work item; lane ``i`` owns tile-token ``i``'s sorted (ascending,
``-1`` trailing-padded) top-k list. The warp does a k-way merge by repeatedly
warp-reducing the minimum head-of-list block id, OR-ing the membership bits of
every lane sitting on that minimum, emitting one union entry, and advancing the
matched lanes. The output blocks come out ascending -- identical to the host
reference's ``sorted()`` -- but order is immaterial: the forward's online softmax
and per-block causal mask are order-independent.

The work-item layout (which (batch, q-tile, kv-head) and the global query offset
of each tile) is precomputed host-side from ``cu_seqlens_q`` alone -- a tiny
``(batch + 1,)`` tensor -- so the expensive ``(Hkv, total_q, topk)`` selection is
never copied to the host (the whole point vs. the reference builder). The work
count is therefore statically known (``total_tiles * Hkv``); unlike the CSR
builder there is no compaction and no atomic work counter.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

# Sentinel head value for an exhausted / inactive lane in the k-way merge. KV
# block ids are bounded by seqlen/128, far below this, so it never collides.
_SENTINEL = (1 << 31) - 1


class BuildUnionMetaSm12x:
    def __init__(self, topk: int, tokens_per_tile: int):
        if topk not in (4, 8, 16, 32):
            raise ValueError(f"topk must be 4, 8, 16, or 32, got {topk}")
        if not 1 <= tokens_per_tile <= 32:
            raise ValueError(
                f"tokens_per_tile must be in [1, 32], got {tokens_per_tile}"
            )
        self._topk = topk
        self._tpt = tokens_per_tile
        # distinct blocks in a union are bounded by the total list length
        self._max_union = tokens_per_tile * topk

    @cute.jit
    def __call__(
        self,
        mQ2k: cute.Tensor,  # (Hkv, total_q, topk) int32, ascending / -1 trailing
        mTileBatch: cute.Tensor,  # (total_tiles,) int32, batch of each global tile
        mTileT: cute.Tensor,  # (total_tiles,) int32, within-batch tile index
        mTileQBase: cute.Tensor,  # (total_tiles,) int32, global query idx of token 0
        mTileNtok: cute.Tensor,  # (total_tiles,) int32, valid tokens in the tile
        mUnionBlocks: cute.Tensor,  # (work_items, max_union) int32 (out)
        mUnionMasks: cute.Tensor,  # (work_items, max_union) int32 (out)
        mUnionCount: cute.Tensor,  # (work_items,) int32 (out)
        mWorkMeta: cute.Tensor,  # (work_items, 3) int32 {batch, q_tile, kv_head} (out)
        H: cutlass.Int32,
        total_tiles: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # one warp per (tile, kv-head) work item
        self._k_build(
            mQ2k,
            mTileBatch,
            mTileT,
            mTileQBase,
            mTileNtok,
            mUnionBlocks,
            mUnionMasks,
            mUnionCount,
            mWorkMeta,
            H,
        ).launch(
            grid=(total_tiles, H, 1),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _k_build(
        self,
        mQ2k: cute.Tensor,
        mTileBatch: cute.Tensor,
        mTileT: cute.Tensor,
        mTileQBase: cute.Tensor,
        mTileNtok: cute.Tensor,
        mUnionBlocks: cute.Tensor,
        mUnionMasks: cute.Tensor,
        mUnionCount: cute.Tensor,
        mWorkMeta: cute.Tensor,
        H: cutlass.Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        gt, h, _ = cute.arch.block_idx()
        work_idx = gt * H + h

        batch_idx = mTileBatch[gt]
        q_tile = mTileT[gt]
        qbase = mTileQBase[gt]
        ntok = mTileNtok[gt]

        # lane i owns tile-token i; lanes >= ntok are inactive (head == SENTINEL)
        # and only participate in the warp shuffles / ballots.
        active = lane < ntok
        sentinel = cutlass.Int32(_SENTINEL)
        p = cutlass.Int32(0)  # cursor into this lane's top-k list

        u = cutlass.Int32(0)
        # at most max_union distinct blocks (one consumed per iteration); a rolled
        # loop over this fixed bound keeps the warp trip count uniform without
        # unrolling 128-256 bodies. Exhausted iterations (all lanes sentinel) no-op.
        for _ in cutlass.range(self._max_union):
            head = sentinel
            if active:
                if p < self._topk:
                    head = mQ2k[h, qbase + lane, p]
            # ascending + trailing -1 padding => a negative head means this lane
            # is exhausted; pin it to the sentinel so it never wins the min.
            if head < 0:
                head = sentinel

            # warp all-reduce of the minimum head (butterfly over 32 lanes).
            m = head
            off = 16
            while off >= 1:
                nbr = cute.arch.shuffle_sync(m, lane ^ off)
                m = cutlass.min(m, nbr)
                off >>= 1

            # m is uniform across the warp, so this branch is uniform: every lane
            # either emits this iteration or skips it together (ballot is safe).
            if m != sentinel:
                eq = active and head == m
                mask = cute.arch.vote_ballot_sync(eq)
                if lane == 0:
                    mUnionBlocks[work_idx, u] = m
                    mUnionMasks[work_idx, u] = mask
                if eq:
                    p += 1
                u += 1

        if lane == 0:
            mUnionCount[work_idx] = u
            mWorkMeta[work_idx, 0] = batch_idx
            mWorkMeta[work_idx, 1] = q_tile
            mWorkMeta[work_idx, 2] = h
