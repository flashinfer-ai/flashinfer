# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Continuous K ingress for the v1 selective scorer; no page lookup or gather."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from ._fp8_mqa_selective import (
    FP8MQASelectiveLogitsKernel,
    FP8MQASparseGather4SampleKernel,
    FP8MQASelectiveH64CandidateKernel,
)


class FP8MQAContinuousSelectiveLogitsKernel(FP8MQASelectiveLogitsKernel):
    uses_contiguous_kv = True

    @cute.jit
    def __call__(
        self,
        keys,
        q,
        weights,
        values,
        offsets,
        counts,
        contexts,
        schedule,
        tile_meta,
        policy,
        sample_tiles,
        scales,
        num_keys: cutlass.Int32,
        num_tiles: cutlass.Int32,
        stream,
    ):
        self._launch(
            keys,
            q,
            weights,
            values,
            offsets,
            counts,
            contexts,
            schedule,
            scales,
            tile_meta,
            policy,
            num_keys,
            num_tiles,
            stream,
            sample_tiles,
        )

    @cute.jit
    def _run_tma_role(
        self,
        mContextLens,
        batch_size,
        block_kv_val,
        end_q_idx,
        end_kv_idx,
        next_q_idx,
        next_kv_idx,
        next_num_kv,
        q_idx,
        has_work,
        tidx,
        kv_pipeline,
        kv_prod_state,
        mBlockTable,
        mSamplePages,
        mTileMeta,
        tma_atom_a,
        tAgA,
        tAsA,
        tma_atom_s,
        tSgS,
        tSsS,
        a_mcast_mask,
        group: cutlass.Constexpr,
        owns_q: cutlass.Constexpr,
        request_tiled: cutlass.Constexpr,
        guarded_first_q: cutlass.Constexpr,
        guarded_prefetch: cutlass.Constexpr,
        sample_internal: cutlass.Constexpr,
        q_pipeline=None,
        q_prod_state=None,
        mB_nkl=None,
        mW_tma=None,
        thr_mma=None,
        sQ=None,
        sW=None,
        tma_atom_b=None,
        tma_atom_w=None,
        tBgB=None,
        tWgW=None,
        tBsB=None,
        tWsW=None,
        b_mcast_mask=None,
        continuous_views=None,
    ):
        """Own a continuous TMA role's task/Q lookahead and producer rings.

        Entry wrappers retain register budgeting and initial state placement.
        Group 1 statically has no Q/view access. Ordinary keeps unconditional
        first-Q behavior; SAMPLE guards idle Q and tiles.
        """
        lane_idx = tidx % 32

        if cutlass.const_expr(owns_q):
            initial_q = True
            if cutlass.const_expr(guarded_first_q):
                initial_q = has_work
            # Idle SAMPLE CTAs have no valid packed query tile: construct
            # neither its metadata-derived view nor its transaction there.
            if initial_q:
                q_pipeline.producer_acquire(q_prod_state)
                q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                q_source, w_source = self._tma_q_source_views(
                    next_q_idx,
                    request_tiled,
                    mTileMeta,
                    mB_nkl,
                    mW_tma,
                    thr_mma,
                    sQ,
                    sW,
                    tma_atom_b,
                    tma_atom_w,
                    tBgB,
                    tWgW,
                )
                cute.copy(
                    tma_atom_b,
                    q_source,
                    tBsB[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_w,
                    w_source,
                    tWsW[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                )
                q_prod_state.advance()

        while has_work:
            q_idx_old = q_idx
            q_idx = next_q_idx
            kv_idx = next_kv_idx
            num_kv = next_num_kv
            if q_idx != q_idx_old:
                if cutlass.const_expr(owns_q):
                    prefetch_next = q_idx + 1
                    # Task traversal skips zero rows. Lookahead must stage
                    # the SAME next live Q/W; staging a skipped row silently
                    # gives its successor wrong logits without a barrier hang.
                    peek_ctx = cutlass.Int32(1)
                    if prefetch_next < end_q_idx:
                        peek_ctx = mContextLens[prefetch_next]
                    while (peek_ctx == 0) & (prefetch_next < end_q_idx):
                        prefetch_next = prefetch_next + 1
                        if prefetch_next < end_q_idx:
                            peek_ctx = mContextLens[prefetch_next]
                        else:
                            peek_ctx = cutlass.Int32(1)
                    if (prefetch_next < end_q_idx) | (
                        (prefetch_next == end_q_idx) & (end_kv_idx > 0)
                    ):
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        q_source, w_source = self._tma_q_source_views(
                            prefetch_next,
                            request_tiled,
                            mTileMeta,
                            mB_nkl,
                            mW_tma,
                            thr_mma,
                            sQ,
                            sW,
                            tma_atom_b,
                            tma_atom_w,
                            tBgB,
                            tWgW,
                        )
                        # Q and weights share the original transaction barrier
                        # and stage; there is no separate producer commit/tail.
                        cute.copy(
                            tma_atom_b,
                            q_source,
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            w_source,
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                        q_prod_state.advance()

            request = mTileMeta[(q_idx, 1)]
            tile = kv_idx + group
            if cutlass.const_expr(sample_internal):
                sampled = cutlass.Int32(0)
                if tile < mSamplePages.shape[1]:
                    sampled = cutlass.Int32(mSamplePages[(request, tile)])
                tile = sampled
            offset = mBlockTable[(request, 0)] + tile * self.block_kv
            mA, mS, kv_thr_mma, sKV, sScales, raw_scales = continuous_views
            shifted_a = cute.domain_offset((offset, 0, 0), mA)
            gA = cute.local_tile(
                shifted_a,
                cute.slice_(self.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            _, source_a = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sKV, 0, 3),
                cute.group_modes(kv_thr_mma.partition_A(gA), 0, 3),
            )
            kv_pipeline.producer_acquire(kv_prod_state)
            bar = kv_pipeline.producer_get_barrier(kv_prod_state)
            stage = kv_prod_state.index
            # Ragged request starts need not align FP32 scales to 16 bytes.
            # Scalar ingress avoids TMA's innermost-coordinate alignment rule.
            # Publish stores before issuing the K transaction that releases the
            # shared consumer barrier; the pipeline counts only K bytes.
            for item in cutlass.range_constexpr(4):
                row = lane_idx + item * 32
                value = cutlass.Float32(0)
                if offset + row < raw_scales.shape[0]:
                    value = raw_scales[offset + row]
                sScales[(row, stage)] = value
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            cute.copy(
                tma_atom_a,
                source_a[(None, 0, 0, 0)],
                tAsA[(None, stage)],
                tma_bar_ptr=bar,
                mcast_mask=a_mcast_mask,
            )
            kv_prod_state.advance()

            next_kv_idx = kv_idx + 2
            if next_kv_idx >= num_kv:
                next_q_idx = q_idx + 1
                next_kv_idx = 0
                if next_q_idx < batch_size:
                    next_num_kv = (
                        mContextLens[next_q_idx] + block_kv_val - 1
                    ) // block_kv_val
                # Stop at the exact CTA end pair when skipping empty rows;
                # otherwise the end coordinate could become unreachable.
                while (
                    (next_num_kv == 0)
                    & (next_q_idx < batch_size)
                    & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                ):
                    next_q_idx = next_q_idx + 1
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
        if cutlass.const_expr(owns_q):
            return q_prod_state, kv_prod_state
        else:
            return kv_prod_state


class FP8MQAContinuousSparseSampleKernel(FP8MQASparseGather4SampleKernel):
    uses_contiguous_kv = True
    __call__ = FP8MQAContinuousSelectiveLogitsKernel.__call__


class FP8MQAContinuousH64CandidateKernel(FP8MQASelectiveH64CandidateKernel):
    uses_contiguous_kv = True
    __call__ = FP8MQAContinuousSelectiveLogitsKernel.__call__

    def __init__(self, *, aligned_keys=False, **kwargs):
        super().__init__(**kwargs)
        # With one request starting at zero and tile-aligned storage, every
        # scale read is in bounds, including the masked causal tail. Keep the
        # original unconditional load so it can overlap the pending MMA.
        self.aligned_contiguous_keys = aligned_keys
