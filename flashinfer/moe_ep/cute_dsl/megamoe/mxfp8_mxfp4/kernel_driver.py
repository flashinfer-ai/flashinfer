# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent device control loop for the SM100 MXFP8 x MXFP4 MegaMoE.

This module owns warp specialization and lifetime ordering only.  Shared-memory
construction, TMA descriptor construction, and the host launch adapter are
deliberately outside this boundary.  The caller supplies the bundles produced
by :mod:`kernel_pipelines` and the already partitioned TMA views.

The compute geometry supports M128 single-CTA and M256 paired-CTA MMA.
The epilogue uses four or eight warps; mega execution adds dispatch warps
and a collective tail.
Multi-rank transport never changes the native MMA/epilogue implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .device_scheduler import DevicePhase
from .dynamic_mainloop import (
    SF_VEC_SIZE,
    compute_non_leader_cta_load_shift,
    issue_dynamic_block_scaled_mma_tile,
)
from .kernel_pipelines import KernelWarpIds

# SM100 L2 cache-policy word for TMA loads (CUTLASS ``CacheHintSm100``
# EVICT_FIRST; the vendored comm helpers' ``_TMA_CACHE_HINT_EVICT_FIRST``).
TMA_CACHE_HINT_EVICT_FIRST = 0x12F0000000000000


@dataclass(frozen=True)
class PhaseTmaViews:
    """Pre-partitioned TMA operands for one FC phase.

    Tuple order is always ``(A, B, SFA, SFB)``.  Each global tensor retains
    its RestK mode; the driver selects the work-record M/N/L tile and then the
    monotonically increasing producer count.
    """

    atoms: tuple[Any, Any, Any, Any]
    global_partitions: tuple[Any, Any, Any, Any]
    tensors: tuple[Any, Any, Any, Any] = (None, None, None, None)
    descriptor_ptrs: tuple[Any, Any, Any, Any] = (None, None, None, None)
    multicast_masks: tuple[Any, Any, Any, Any] = (None, None, None, None)


@dataclass(frozen=True)
class DriverViews:
    """Device views consumed by the control loop but owned by its caller."""

    fc1: PhaseTmaViews
    fc2: PhaseTmaViews
    fc1_output: Any
    fc1_output_scales: Any
    route_store: Any
    fc1_done_counter: Any
    token_metadata: Any = None


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
        # M128 is one CTA; M256 is a 2-CTA pair, each CTA owning 128 rows.
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
                184 if len(kernel_pipelines.warp_ids.epilogue) == 8 else 256
            )
        if epilogue_registers <= 0 or task_registers <= 0:
            raise ValueError("warp-group register counts must be positive")
        if len(kernel_pipelines.warp_ids.epilogue) == 8:
            task_warps = 4 + len(kernel_pipelines.warp_ids.dispatch)
            if 32 * (8 * epilogue_registers + task_warps * task_registers) > 65536:
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
        # Weights (A) and their scales (SFA) are read once per forward; tokens,
        # token scales and the FC1 intermediate are re-read per feature tile.
        # EVICT_FIRST on the A/SFA loads lets the re-read streams keep L2;
        # the other copies pass no policy.
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

        if cutlass.const_expr(descriptor_ptr is None):
            if cutlass.const_expr(cache_policy is None):
                cute.copy(
                    atom,
                    source,
                    destination,
                    tma_bar_ptr=barrier,
                    mcast_mask=mask,
                )
            else:
                cute.copy(
                    atom,
                    source,
                    destination,
                    tma_bar_ptr=barrier,
                    mcast_mask=mask,
                    cache_policy=cache_policy,
                )
        else:
            if cutlass.const_expr(cache_policy is None):
                cute.copy(
                    atom,
                    source,
                    destination,
                    tma_bar_ptr=barrier,
                    tma_desc_ptr=descriptor_ptr,
                    mcast_mask=mask,
                )
            else:
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
        # Activation pools have one L plane, unlike the expert-indexed
        # weights. Apply the separately padded row origins before partitioning
        # so a 64-row data origin remains representable with an N128 tile.
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
            # token (64 rows in for odd tiles; see make_mma_tmem_sfb) and the
            # 256-row SFB tile loads from there (sfb_atom_tile is 0).
            sf_row = sf_row + (work.tile_n_idx * Int32(192)) // Int32(128) * Int32(128)
        parts = self.staging.partition_tma_loads(
            tma_atom_a=phase.atoms[0],
            tma_tensor_a=phase.tensors[0],
            tma_atom_b=phase.atoms[1],
            tma_tensor_b=cute.domain_offset((data_row, 0, 0), phase.tensors[1]),
            tma_atom_sfa=phase.atoms[2],
            tma_tensor_sfa=phase.tensors[2],
            tma_atom_sfb=phase.atoms[3],
            tma_tensor_sfb=cute.domain_offset((sf_row, 0, 0), phase.tensors[3]),
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
            global_a=phase.global_partitions[0],
            global_b=phase.global_partitions[1],
            global_sfa=phase.global_partitions[2],
            global_sfb=phase.global_partitions[3],
            tile_mnl=(work.tile_m_idx, work.tile_n_idx, work.expert_idx),
        )
        weight_policy = None
        if cutlass.const_expr(self.weight_l2_evict_first):
            weight_policy = Int64(TMA_CACHE_HINT_EVICT_FIRST)
        ab_producer.reset()
        ready = ab_producer.try_acquire()
        for _ in cutlass.range(0, k_tiles, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(
                ready,
                expected_tx=Int32(self.staging.tma_transaction_bytes_a()),
            )
            ready = cutlass.Boolean(1)
            if handle.count + 1 < k_tiles:
                ready = ab_producer.try_acquire()
            self._copy_tma(
                phase.atoms[0],
                sources[0][(None, handle.count)],
                operand_smem.weight[(None, handle.index)],
                handle.barrier,
                phase.descriptor_ptrs[0],
                phase.multicast_masks[0],
                weight_policy,
            )
            self._copy_tma(
                phase.atoms[2],
                sources[2][(None, handle.count)],
                operand_smem.weight_scale[(None, handle.index)],
                handle.barrier,
                phase.descriptor_ptrs[2],
                phase.multicast_masks[2],
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
            # FC1 and FC2 RestK extents differ.  Keep their view containers
            # branch-local so CuTe never has to merge heterogeneous SSA types.
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
        source_b = parts[3][(None, work.tile_n_idx, None, 0)]
        # SFB tiles are whole 128-row atoms; two N64 token tiles share one.
        source_sfb = parts[7][
            (None, self.staging.sfb_atom_tile(work.tile_n_idx), None, 0)
        ]
        ab_producer.reset()
        ready = ab_producer.try_acquire()
        for _ in cutlass.range(0, k_tiles, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(
                ready,
                expected_tx=Int32(self.staging.tma_transaction_bytes_b()),
            )
            ready = cutlass.Boolean(1)
            if handle.count + 1 < k_tiles:
                ready = ab_producer.try_acquire()
            self._copy_tma(
                phase.atoms[1],
                source_b[(None, handle.count)],
                parts[2][(None, handle.index)],
                handle.barrier,
                phase.descriptor_ptrs[1],
                phase.multicast_masks[1],
            )
            self._copy_tma(
                phase.atoms[3],
                source_sfb[(None, handle.count)],
                parts[6][(None, handle.index)],
                handle.barrier,
                phase.descriptor_ptrs[3],
                phase.multicast_masks[3],
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
                    if cutlass.const_expr(
                        scheduler_params.fc1_ready_counter_ptr is not None
                    ):
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
            # A 2-CTA MMA is issued by the leader CTA for the pair; the peer's
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
                # Both producers advance the shared stage once per K tile.  Derive
                # the common K extent from the work phase.
                # ``phase_views`` is intentionally not carried into this warp; the
                # scheduler's static problem shapes determine this loop at trace.
                k_tiles = Int32(0)
                if work.phase == Int32(DevicePhase.FC1):
                    k_tiles = Int32(fc1_k_tiles)
                else:
                    k_tiles = Int32(fc2_k_tiles)
                # The issuer's contract is 1..N valid tokens.
                mma_tokens = work.valid_tokens_in_cta_tile
                if mma_tokens < Int32(1):
                    mma_tokens = Int32(1)
                for k_tile in cutlass.range(0, k_tiles, 1, unroll=1):
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
                    # valid tokens never affect live results.  Each MMA computes
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
        ``@cute.kernel``.  This callable only consumes those objects and never
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
        if cutlass.const_expr(self.dispatch_phase is not None):
            if warp_idx == self.warp_ids.scheduler:
                self.dispatch_phase.scheduler_wait(dispatch_args)
        scheduler_bundle.scheduler.internal_init(
            warp_idx=warp_idx,
            sched_warp_id=self.warp_ids.scheduler,
        )

        # Capture immutable consumer parameters before tracing the scheduler
        # branch. Its first atomic decode rebinds scheduler.params through
        # SCF results; those branch-local values cannot feed a sibling warp.
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
        elif warp_idx < Int32(4) or (
            cutlass.const_expr(len(self.warp_ids.epilogue) == 8)
            and warp_idx >= Int32(8 + len(self.warp_ids.dispatch))
        ):
            cute.arch.warpgroup_reg_alloc(self.epilogue_registers)
            tmem_allocator.allocate(self.kernel_pipelines.num_tmem_cols)
            tmem_allocator.wait_for_alloc()
            tmem_base = tmem_allocator.retrieve_ptr(cutlass.Float32)
            tmem_acc = self.kernel_pipelines.make_accumulator_tensor(tmem_base)
            epilogue_warp_idx = warp_idx
            if cutlass.const_expr(len(self.warp_ids.epilogue) == 8):
                if warp_idx >= Int32(self.warp_ids.epilogue[4]):
                    epilogue_warp_idx = warp_idx - Int32(self.warp_ids.epilogue[4] - 4)
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
            if cutlass.const_expr(self.dispatch_phase is not None):
                cute.arch.warpgroup_reg_dealloc(self.task_registers)
                self.dispatch_phase(
                    dispatch_args,
                    dispatch_storage,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                    tidx=tidx,
                )

        # This is an all-warp rendezvous.  In particular, no role may return
        # after publishing its terminal scheduler/pipeline state.
        if cutlass.const_expr(self.dispatch_phase is not None):
            self.dispatch_phase.kernel_tail(
                dispatch_args, warp_idx=warp_idx, lane_idx=lane_idx, tidx=tidx
            )


__all__ = [
    "DriverViews",
    "KernelWarpIds",
    "PersistentM128DeviceDriver",
    "PhaseTmaViews",
]
