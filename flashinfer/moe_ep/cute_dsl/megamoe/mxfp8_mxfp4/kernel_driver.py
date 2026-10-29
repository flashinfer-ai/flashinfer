# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp specialization and lifetime ordering for persistent MXFP8 x MXFP4 MegaMoE.

The caller supplies :mod:`kernel_pipelines` bundles and partitioned TMA
views. Supports M128 single-CTA or M256 paired-CTA MMA, four/eight epilogue
warps, dispatch warps, and a collective tail. Multi-rank transport shares
the same MMA and epilogue implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import L2_EVICT_FIRST

from .device_epilogue import EPILOGUE_WARPS
from .device_scheduler import DevicePhase
from .dynamic_mainloop import (
    SF_VEC_SIZE,
    compute_non_leader_cta_load_shift,
    issue_dynamic_block_scaled_mma_tile,
)
from .fc2_epilogue import Fc2RouteStore
from .kernel_pipelines import WARP_SIZE, KernelWarpIds


@dataclass(frozen=True)
class TmaOperandView:
    """TMA descriptor and tensor views for one operand."""

    atom: cute.CopyAtom
    tensor: cute.Tensor
    global_partition: cute.Tensor
    descriptor_ptr: cute.Pointer | None = None
    multicast_mask: cutlass.Int16 | None = None


@dataclass(frozen=True)
class PhaseTmaViews:
    """Pre-partitioned TMA operands for one FC phase."""

    a: TmaOperandView
    b: TmaOperandView
    sfa: TmaOperandView
    sfb: TmaOperandView


@dataclass(frozen=True)
class DriverViews:
    """Device views consumed by the control loop but owned by its caller."""

    fc1: PhaseTmaViews
    fc2: PhaseTmaViews
    fc1_output: cute.Tensor
    fc1_output_scales: cute.Tensor
    route_store: Fc2RouteStore
    fc1_done_counter: cute.Tensor
    token_metadata: cute.Tensor | None = None


class PersistentM128DeviceDriver:
    """Run persistent FC1/FC2 with token dispatch and the collective tail."""

    def __init__(
        self,
        *,
        kernel_pipelines,
        epilogue_registers: int | None = None,
        task_registers: int = 72,
        weight_l2_evict_first: bool = False,
    ) -> None:
        staging = kernel_pipelines.staging
        dispatch_phase = kernel_pipelines.dispatch_phase
        epilogue = kernel_pipelines.epilogue
        # M128 is one CTA. M256 is a 2-CTA pair, each CTA owning 128 rows.
        two_cta = staging.cta_group == tcgen05.CtaGroup.TWO
        if staging.mma_tiler_mn[0] != (256 if two_cta else 128):
            raise ValueError("MMA M must be 128 (1-CTA) or 256 (2-CTA)")
        if epilogue.config.cta_tile_features != 128:
            raise ValueError("driver and epilogue feature tiles must both be M128")
        if epilogue.config.cta_tile_tokens != staging.cta_tile_shape_mnk[1]:
            raise ValueError("driver staging and epilogue token tiles disagree")
        if epilogue_registers is None:
            # Mega's 512-thread launch has 128 registers/thread initially.
            # Two task warpgroups donate down to 72, allowing 184 in each
            # epilogue warpgroup.
            epilogue_registers = (
                184 if len(kernel_pipelines.warp_ids.epilogue) > EPILOGUE_WARPS else 256
            )
        if epilogue_registers <= 0 or task_registers <= 0:
            raise ValueError("warp-group register counts must be positive")
        epilogue_warps = len(kernel_pipelines.warp_ids.epilogue)
        if epilogue_warps > EPILOGUE_WARPS:
            task_warps = (
                kernel_pipelines.warp_ids.threads_per_cta // WARP_SIZE - epilogue_warps
            )
            if (
                WARP_SIZE
                * (epilogue_warps * epilogue_registers + task_warps * task_registers)
                > 65536
            ):
                raise ValueError(
                    "warp-group register counts exceed the SM register file"
                )
        self.two_cta = two_cta
        self.kernel_pipelines = kernel_pipelines
        self.staging = staging
        self.dispatch_phase = dispatch_phase
        self.epilogue = epilogue
        self.warp_ids = kernel_pipelines.warp_ids
        self.epilogue_registers = epilogue_registers
        self.task_registers = task_registers
        # Weights (A) and their scales (SFA) are read once per forward. Tokens,
        # token scales and the FC1 intermediate are re-read per feature tile.
        # EVICT_FIRST on the A/SFA loads lets the re-read streams keep L2.
        # The other copies pass no policy.
        self.weight_l2_evict_first = weight_l2_evict_first

    @staticmethod
    @cute.jit
    def _wait_for_counter(counter_ptr, threshold) -> None:
        """Acquire-spin on a monotonic device counter."""

        observed = cute.arch.load(
            counter_ptr, counter_ptr.dtype, sem="acquire", scope="gpu"
        )
        while observed < threshold:
            observed = cute.arch.load(
                counter_ptr, counter_ptr.dtype, sem="acquire", scope="gpu"
            )

    @staticmethod
    @cute.jit
    def _copy_tma(
        atom, source, destination, barrier, descriptor_ptr, mask, cache_policy=None
    ) -> None:
        """Issue a static or rewritten-descriptor TMA load, optionally with an
        L2 cache-policy word (``cache_policy``, an ``Int64``)."""

        cute.copy(
            atom,
            source,
            destination,
            tma_bar_ptr=barrier,
            tma_desc_ptr=descriptor_ptr,
            mcast_mask=mask,
            cache_policy=cache_policy,
        )

    @cute.jit
    def _run_scheduler(self, scheduler) -> None:
        scheduler.gen_next_work()
        while scheduler.current_work.is_valid_tile:
            scheduler.publish_work()
            scheduler.gen_next_work()
        # Every consumer needs one terminal record.
        scheduler.publish_work()
        scheduler.produce_tail()

    @cute.jit
    def _partition_tma_b_work(self, work, phase, operand_smem):
        # Activations from all experts are packed into one row pool.
        # Apply the separate data and scale row offsets before forming tiles,
        # so a 128-row tile can start at a 64-row offset.
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord = self.staging.cluster_layout_vmnk.get_flat_coord(rank)
        sf_coord = self.staging.cluster_layout_sfb_vmnk.get_flat_coord(rank)
        bidx, _, _ = cute.arch.block_idx()
        mma_v = bidx % cute.size(self.staging.tiled_mma.thr_id.shape)
        data_row = work.cumulative_data_physical_row
        if cutlass.const_expr(self.two_cta):
            # The hardware splits a dynamic-N 2-CTA MMA at align16(valid)/2
            # rows, while the TMA partition splits the static tile at N/2:
            # shift the non-leader's token rows to the hardware split.
            if mma_v != Int32(0):
                valid = work.valid_tokens_in_cta_tile
                if valid < Int32(1):
                    valid = Int32(1)
                data_row = data_row + compute_non_leader_cta_load_shift(
                    valid_tokens_in_tile=valid,
                    mma_tiler_n=self.staging.mma_tiler[1],
                )
        sf_row = work.cumulative_sf_physical_row
        if cutlass.const_expr(self.staging.mma_tiler[1] == 192):
            # An N192 tile's scales start at the 128-row atom holding its first
            # token (64 rows in for odd tiles, as in make_mma_tmem_sfb) and the
            # 256-row SFB tile loads from there (sfb_atom_tile is 0).
            sf_row = sf_row + (work.tile_n_idx * Int32(192)) // Int32(128) * Int32(128)
        parts = self.staging.partition_tma_loads(
            tma_atom_a=phase.a.atom,
            tma_tensor_a=phase.a.tensor,
            tma_atom_b=phase.b.atom,
            tma_tensor_b=cute.domain_offset((data_row, 0, 0), phase.b.tensor),
            tma_atom_sfa=phase.sfa.atom,
            tma_tensor_sfa=phase.sfa.tensor,
            tma_atom_sfb=phase.sfb.atom,
            tma_tensor_sfb=cute.domain_offset((sf_row, 0, 0), phase.sfb.tensor),
            smem_a=operand_smem.weight,
            smem_b=operand_smem.activation,
            smem_sfa=operand_smem.weight_scale,
            smem_sfb=operand_smem.activation_scale,
            mma_tile_coord_v=mma_v,
            block_in_cluster_coord_vmnk=coord,
            block_in_cluster_coord_sfb_vmnk=sf_coord,
        )
        return parts

    @cute.jit
    def _issue_tma_a_work(self, work, phase, ab_producer, operand_smem, k_tiles):
        """Issue one A/SFA record and return the advanced staged producer."""
        sources = self.staging.slice_tma_sources(
            global_a=phase.a.global_partition,
            global_b=phase.b.global_partition,
            global_sfa=phase.sfa.global_partition,
            global_sfb=phase.sfb.global_partition,
            tile_mnl=(work.tile_m_idx, work.tile_n_idx, work.expert_idx),
        )
        weight_policy = None
        if cutlass.const_expr(self.weight_l2_evict_first):
            weight_policy = Int64(L2_EVICT_FIRST)
        ab_producer.reset()
        ready = ab_producer.try_acquire()
        # Keep the TMA and MMA loops rolled. Unrolling would duplicate
        # asynchronous operations and add remainder handling around pipeline waits.
        for _ in cutlass.range(k_tiles, unroll=1):
            handle = ab_producer.acquire_and_advance(
                ready,
                expected_tx=Int32(self.staging.tma_transaction_bytes_a()),
            )
            ready = cutlass.Boolean(1)
            if handle.count + 1 < k_tiles:
                ready = ab_producer.try_acquire()
            self._copy_tma(
                phase.a.atom,
                sources.a[(None, handle.count)],
                operand_smem.weight[(None, handle.index)],
                handle.barrier,
                phase.a.descriptor_ptr,
                phase.a.multicast_mask,
                weight_policy,
            )
            self._copy_tma(
                phase.sfa.atom,
                sources.sfa[(None, handle.count)],
                operand_smem.weight_scale[(None, handle.index)],
                handle.barrier,
                phase.sfa.descriptor_ptr,
                phase.sfa.multicast_mask,
                weight_policy,
            )
        return ab_producer

    @cute.jit
    def _run_tma_a(
        self,
        *,
        consumer,
        ab_producer,
        operand_smem,
        views,
        fc1_k_tiles,
        fc2_k_tiles,
    ) -> None:
        work = consumer.consume_work()
        while work.is_valid_tile:
            # FC1 and FC2 views contain different numbers of K tiles. Keep them
            # in separate branches so CuTe can preserve each view's shape.
            if work.phase == Int32(DevicePhase.FC1):
                ab_producer = self._issue_tma_a_work(
                    work,
                    views.fc1,
                    ab_producer,
                    operand_smem,
                    Int32(fc1_k_tiles),
                )
            else:
                ab_producer = self._issue_tma_a_work(
                    work,
                    views.fc2,
                    ab_producer,
                    operand_smem,
                    Int32(fc2_k_tiles),
                )
            work = consumer.consume_work()
        ab_producer.tail()

    @cute.jit
    def _issue_tma_b_work(self, work, phase, ab_producer, operand_smem, k_tiles):
        """Issue one B/SFB record and return the advanced staged producer."""

        parts = self._partition_tma_b_work(work, phase, operand_smem)
        source_b = parts.global_b[(None, work.tile_n_idx, None, 0)]
        # SFB tiles are whole 128-row atoms. Two N64 token tiles share one.
        source_sfb = parts.global_sfb[
            (None, self.staging.sfb_atom_tile(work.tile_n_idx), None, 0)
        ]
        ab_producer.reset()
        ready = ab_producer.try_acquire()
        for _ in cutlass.range(k_tiles, unroll=1):
            handle = ab_producer.acquire_and_advance(
                ready,
                expected_tx=Int32(self.staging.tma_transaction_bytes_b()),
            )
            ready = cutlass.Boolean(1)
            if handle.count + 1 < k_tiles:
                ready = ab_producer.try_acquire()
            self._copy_tma(
                phase.b.atom,
                source_b[(None, handle.count)],
                parts.smem_b[(None, handle.index)],
                handle.barrier,
                phase.b.descriptor_ptr,
                phase.b.multicast_mask,
            )
            self._copy_tma(
                phase.sfb.atom,
                source_sfb[(None, handle.count)],
                parts.smem_sfb[(None, handle.index)],
                handle.barrier,
                phase.sfb.descriptor_ptr,
                phase.sfb.multicast_mask,
            )
        return ab_producer

    @cute.jit
    def _run_tma_b(
        self,
        *,
        consumer,
        ab_producer,
        operand_smem,
        views,
        scheduler_params,
        fc1_k_tiles,
        fc2_k_tiles,
    ) -> None:
        work = consumer.consume_work()
        while work.is_valid_tile:
            if not work.peek_ready:
                if work.phase == Int32(DevicePhase.FC1):
                    self._wait_for_counter(
                        scheduler_params.fc1_ready_counter_ptr + work.counter_slot,
                        work.valid_tokens_in_cta_tile,
                    )
                else:
                    self._wait_for_counter(
                        scheduler_params.fc1_done_counter_ptr + work.counter_slot,
                        scheduler_params.fc2_ready_threshold,
                    )
            if work.phase == Int32(DevicePhase.FC2):
                # A scheduler readiness peek can skip the acquire spin, but
                # it cannot replace this producer's generic-to-TMA proxy fence.
                cute.arch.fence_proxy("async")
                cute.arch.fence_proxy("async.global")
            if work.phase == Int32(DevicePhase.FC1):
                ab_producer = self._issue_tma_b_work(
                    work,
                    views.fc1,
                    ab_producer,
                    operand_smem,
                    Int32(fc1_k_tiles),
                )
            else:
                ab_producer = self._issue_tma_b_work(
                    work,
                    views.fc2,
                    ab_producer,
                    operand_smem,
                    Int32(fc2_k_tiles),
                )
            work = consumer.consume_work()
        ab_producer.tail()

    @cute.jit
    def _run_mma(
        self,
        *,
        consumer,
        ab_consumer,
        acc_pipeline,
        operand_smem,
        tmem_allocator,
        fc1_k_tiles,
        fc2_k_tiles,
    ) -> None:
        tiled_mma = self.staging.tiled_mma
        frag_a = tiled_mma.make_fragment_A(operand_smem.weight)
        frag_b = tiled_mma.make_fragment_B(operand_smem.activation)
        tmem_allocator.wait_for_alloc()
        tmem_base = tmem_allocator.retrieve_ptr(cutlass.Float32)
        tmem_acc = self.kernel_pipelines.make_accumulator_tensor(tmem_base)
        tmem_sfa, tmem_sfb = self.kernel_pipelines.make_scale_tmem_tensors(tmem_base)
        copy_sfa, src_sfa, dst_sfa = self.staging.make_s2t_copy_and_partition(
            operand_smem.weight_scale, tmem_sfa
        )
        copy_sfb, src_sfb, dst_sfb = self.staging.make_s2t_copy_and_partition(
            operand_smem.activation_scale, tmem_sfb
        )
        acc_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.epilogue.num_acc_pipeline_stages,
        )

        work = consumer.consume_work()
        leader = True
        if cutlass.const_expr(self.two_cta):
            bidx, _, _ = cute.arch.block_idx()
            leader = bidx % cute.size(tiled_mma.thr_id.shape) == Int32(0)
        while work.is_valid_tile:
            # A 2-CTA MMA is issued by the leader CTA for the pair. The peer's
            # MMA warp only walks the schedule (every consumer needs its
            # terminal record).
            if leader:
                acc_pipeline.producer_acquire(acc_state)
                acc = tmem_acc[(None, None, None, acc_state.index)]
                mma_sfb = self.kernel_pipelines.make_mma_tmem_sfb(
                    tmem_base, work.tile_n_idx, tmem_sfb
                )
                ab_consumer.reset()
                ab_full = ab_consumer.try_wait()
                # Both producers advance the shared stage once per K tile. Derive
                # K from the work phase and scheduler's static shapes without
                # carrying phase_views into this warp.
                k_tiles = Int32(0)
                if work.phase == Int32(DevicePhase.FC1):
                    k_tiles = Int32(fc1_k_tiles)
                else:
                    k_tiles = Int32(fc2_k_tiles)
                # The issuer's contract is 1..N valid tokens.
                mma_tokens = work.valid_tokens_in_cta_tile
                if mma_tokens < Int32(1):
                    mma_tokens = Int32(1)
                for k_tile in cutlass.range(k_tiles, unroll=1):
                    ab_handle = ab_consumer.wait_and_advance(ab_full)
                    ab_full = cutlass.Boolean(1)
                    if k_tile + 1 < k_tiles:
                        ab_full = ab_consumer.try_wait()
                    self.staging.copy_scale_stage(
                        copy_sfa, src_sfa, dst_sfa, ab_handle.index
                    )
                    self.staging.copy_scale_stage(
                        copy_sfb, src_sfb, dst_sfb, ab_handle.index
                    )
                    tile_a = (None, None, None, ab_handle.index)
                    tile_b = (None, None, None, ab_handle.index)
                    # Token columns are independent, and the epilogue masks
                    # stores using valid_tokens_in_cta_tile, so columns beyond the
                    # valid tokens never affect live results. Each MMA computes
                    # only the valid columns, rounded up to 16.
                    issue_dynamic_block_scaled_mma_tile(
                        acc_tensor=acc,
                        a_frag_tile=frag_a[tile_a],
                        b_frag_tile=frag_b[tile_b],
                        sfa_tensor=tmem_sfa,
                        sfb_tensor=mma_sfb,
                        k_tile_idx=k_tile,
                        valid_tokens_in_tile=mma_tokens,
                        a_dtype=self.staging.weight_dtype,
                        b_dtype=self.staging.activation_dtype,
                        sf_dtype=self.staging.scale_dtype,
                        sf_vec_size=SF_VEC_SIZE,
                        mma_tiler_mnk=self.staging.mma_tiler,
                    )
                    ab_handle.release()
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
            work = consumer.consume_work()
        if leader:
            acc_pipeline.producer_tail(acc_state)

    @cute.jit
    def __call__(
        self,
        *,
        mainloop_pipelines,
        scheduler_bundle,
        operand_smem,
        tma_operand_smem,
        tmem_allocator,
        views: DriverViews,
        dispatch_args,
        dispatch_storage,
        epilogue_exchange,
        fc1_k_tiles: int,
        fc2_k_tiles: int,
    ) -> None:
        """Execute the all-warp persistent control flow.

        Pipeline and tensor allocation have already happened in the enclosing
        ``@cute.kernel``. This callable only consumes those objects and never
        constructs or launches a second kernel.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()

        pipeline_init_arrive(
            cluster_shape_mn=self.staging.cluster_shape_mn, is_relaxed=True
        )
        pipeline_init_wait(cluster_shape_mn=self.staging.cluster_shape_mn)

        # Both schedulers prime their state on the scheduler warp, after
        # dispatch publishes counts and resets the atomic claim counter.
        # Their pipelines were constructed before the init fence above.
        if warp_idx == self.warp_ids.scheduler:
            self.dispatch_phase.scheduler_wait(dispatch_args)
        scheduler_bundle.scheduler.internal_init(
            warp_idx=warp_idx,
            sched_warp_id=self.warp_ids.scheduler,
        )

        # Save consumer parameters before entering the scheduler branch.
        # Scheduler initialization replaces them with values that other
        # warp branches cannot access.
        scheduler_params = scheduler_bundle.scheduler.params
        if warp_idx == self.warp_ids.scheduler:
            cute.arch.warpgroup_reg_dealloc(self.task_registers)
            self._run_scheduler(scheduler_bundle.scheduler)
        elif warp_idx == self.warp_ids.tma_a:
            cute.arch.warpgroup_reg_dealloc(self.task_registers)
            self._run_tma_a(
                consumer=scheduler_bundle.scheduler.make_consumer(),
                ab_producer=mainloop_pipelines.ab_producer,
                operand_smem=tma_operand_smem,
                views=views,
                fc1_k_tiles=fc1_k_tiles,
                fc2_k_tiles=fc2_k_tiles,
            )
        elif warp_idx == self.warp_ids.tma_b:
            cute.arch.warpgroup_reg_dealloc(self.task_registers)
            self._run_tma_b(
                consumer=scheduler_bundle.scheduler.make_consumer(),
                ab_producer=mainloop_pipelines.ab_producer,
                operand_smem=operand_smem,
                views=views,
                scheduler_params=scheduler_params,
                fc1_k_tiles=fc1_k_tiles,
                fc2_k_tiles=fc2_k_tiles,
            )
        elif warp_idx == self.warp_ids.mma:
            cute.arch.warpgroup_reg_dealloc(self.task_registers)
            self._run_mma(
                consumer=scheduler_bundle.scheduler.make_consumer(),
                ab_consumer=mainloop_pipelines.ab_consumer,
                acc_pipeline=mainloop_pipelines.accumulator,
                operand_smem=operand_smem,
                tmem_allocator=tmem_allocator,
                fc1_k_tiles=fc1_k_tiles,
                fc2_k_tiles=fc2_k_tiles,
            )
        elif warp_idx < Int32(EPILOGUE_WARPS) or (
            cutlass.const_expr(len(self.warp_ids.epilogue) > EPILOGUE_WARPS)
            and warp_idx >= Int32(self.warp_ids.dispatch[-1] + 1)
        ):
            cute.arch.warpgroup_reg_alloc(self.epilogue_registers)
            tmem_allocator.allocate(self.kernel_pipelines.num_tmem_cols)
            tmem_allocator.wait_for_alloc()
            tmem_base = tmem_allocator.retrieve_ptr(cutlass.Float32)
            tmem_acc = self.kernel_pipelines.make_accumulator_tensor(tmem_base)
            epilogue_warp_idx = warp_idx
            if cutlass.const_expr(len(self.warp_ids.epilogue) > EPILOGUE_WARPS):
                if warp_idx >= Int32(self.warp_ids.epilogue[EPILOGUE_WARPS]):
                    epilogue_warp_idx = warp_idx - Int32(
                        self.warp_ids.epilogue[EPILOGUE_WARPS] - EPILOGUE_WARPS
                    )
            self.kernel_pipelines.run_epilogue(
                tmem_acc_tensor=tmem_acc,
                acc_pipeline=mainloop_pipelines.accumulator,
                scheduler_consumer=scheduler_bundle.scheduler.make_consumer(),
                fc1_output=views.fc1_output,
                fc1_output_scales=views.fc1_output_scales,
                route_store=views.route_store,
                fc1_done_counter=views.fc1_done_counter,
                epilogue_exchange=epilogue_exchange,
                warp_idx=epilogue_warp_idx,
                tidx=tidx,
                token_metadata=views.token_metadata,
            )
            cute.arch.fence_acq_rel_sys()
            tmem_allocator.relinquish_alloc_permit()
            tmem_allocator.free(tmem_base, self.kernel_pipelines.num_tmem_cols)
        else:
            cute.arch.warpgroup_reg_dealloc(self.task_registers)
            self.dispatch_phase(
                dispatch_args,
                dispatch_storage,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )

        # This is an all-warp rendezvous. In particular, no role may return
        # after publishing its terminal scheduler/pipeline state.
        self.dispatch_phase.kernel_tail(
            dispatch_args, warp_idx=warp_idx, lane_idx=lane_idx, tidx=tidx
        )


__all__ = [
    "DriverViews",
    "KernelWarpIds",
    "PersistentM128DeviceDriver",
    "PhaseTmaViews",
    "TmaOperandView",
]
