# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Pipelines shared by the gathering grouped kernels: plumbing only, no arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineOp,
    agent_sync,
)
from cutlass.pipeline.helpers import MbarrierArray


@dataclass(frozen=True)
class PipelineCpAsyncUmma(PipelineAsync):
    """Asynchronous-copy producers (the gather warps) and the UMMA consumer (its release is the ``tcgen05`` commit of the
    MMA that read the stage; on a 2-CTA record the leader's commit releases both CTAs' empty barriers). The DSL's own
    ``PipelineCpAsync`` has an async-thread consumer; this is its UMMA-consumer sibling."""

    cta_group: tcgen05.CtaGroup = tcgen05.CtaGroup.ONE

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        barrier_storage: cute.Pointer,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        defer_sync: bool = False,
        producer_op: PipelineOp = PipelineOp.AsyncLoad,
    ):
        """``producer_op``: ``AsyncLoad`` (the default) makes the producer's commit the ``cp.async.mbarrier.arrive.noinc``
        that fires when the thread's ``cp.async`` copies land; ``AsyncThread`` makes it a plain ``mbarrier.arrive`` -- for a
        producer whose bytes are counted on another barrier (the TMA gather's transaction count) and whose commit only marks
        the issue (an asynchronous arrive with no ``cp.async`` to track is not reliably performed)."""
        producer = (producer_op, producer_group)
        consumer = (PipelineOp.TCGen05Mma, consumer_group)
        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer
        )
        sync_object_empty = MbarrierArray(
            barrier_storage=barrier_storage.align(min_align=8) + num_stages,
            num_stages=num_stages,
            agent=consumer,
        )
        if not defer_sync:
            agent_sync(Agent.ThreadBlock)
        consumer_mask, cta_group = None, tcgen05.CtaGroup.ONE
        if cta_layout_vmnk is not None and cute.size(cta_layout_vmnk, mode=[0]) == 2:
            consumer_mask = pipeline.PipelineAsyncUmma._compute_peer_cta_mask(
                cta_layout_vmnk
            )  # the pair: both empty barriers
            cta_group = tcgen05.CtaGroup.TWO
        return PipelineCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            None,
            consumer_mask,
            cta_group,
        )

    def consumer_release(self, state):
        self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)
