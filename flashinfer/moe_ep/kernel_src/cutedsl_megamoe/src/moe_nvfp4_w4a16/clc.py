# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Elastic FC1 CLC ownership and a completion-published FC2 ready queue."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import (
    T,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)

from .fc1_fc2_fuse_sched import BlockPhase
from moe_nvfp4_swapab.moe_utils import (
    mbarrier_arrive_expect_tx_on_peer,
    store_i32_to_peer_cluster_smem_async,
)


def workspace_shape(row_bound, fc1_features, fc2_features, bundle):
    bundles = (row_bound * fc1_features + bundle - 1) // bundle
    queue_offset = 4 + (bundles + 31) // 32
    return queue_offset, queue_offset + row_bound, bundles


def launch_grid(bundles, cluster):
    if bundles <= 65535:
        return (cluster[0], cluster[1], max(1, bundles))
    return (cluster[0] * bundles, cluster[1], 1)


@cute.jit
def active_cluster(
    params,
    fc1_features: cutlass.Constexpr,
    fc2_features: cutlass.Constexpr,
    bundle: cutlass.Constexpr,
    helper_bound: cutlass.Constexpr,
    cluster_x: cutlass.Constexpr,
):
    # Dispatch completed before this launch. Counts stay immutable until finish,
    # so every thread and both CTAs take the same branch without a new barrier.
    rows = cutlass.Int32(0)
    for expert in cutlass.range(params.expert_cnt):
        rows += cute.ceil_div(params.expert_token_sizes[expert], params.cluster_tile_m)
    bundles = cute.ceil_div(rows * fc1_features, bundle)
    helpers = cutlass.min(helper_bound, rows * fc2_features)
    x, _, z = cute.arch.block_idx()
    return x // cluster_x + z < cutlass.max(bundles, helpers)


class Workspace:
    """Rank-local zero-prefix view; reset only after the complete compute grid.

    Counters: scan cursor, queue head, reservation tail, published row count.
    Bitmap owns FC1 bundles. Queue slot zero means not yet published; a positive
    value is one plus a ready row index, stored with release semantics. Queue
    head counts feature tasks; tail counts reserved rows.
    """

    def __init__(self, data, queue_offset, fc2_features):
        self.data = data
        self.queue_offset = queue_offset
        self.fc2_features = fc2_features

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.data)

    def __new_from_mlir_values__(self, values):
        return Workspace(
            new_from_mlir_values(self.data, values),
            self.queue_offset,
            self.fc2_features,
        )

    @cute.jit
    def claim(self, bundle):
        mask = cutlass.Int32(1) << (bundle % 32)
        old = cute.arch.atomic_or(
            self.data.iterator + 4 + bundle // 32,
            mask,
            sem="relaxed",
            scope="gpu",
        )
        return (old & mask) == 0

    @cute.jit
    def pop(self):
        task = cutlass.Int32(-1)
        retry = True
        while retry:
            head = cute.arch.load(
                self.data.iterator + 1, cutlass.Int32, sem="acquire", scope="gpu"
            )
            tail = cute.arch.load(
                self.data.iterator + 2, cutlass.Int32, sem="acquire", scope="gpu"
            )
            retry = False
            if head < tail * self.fc2_features:
                encoded = cute.arch.load(
                    self.data.iterator + self.queue_offset + head // self.fc2_features,
                    cutlass.Int32,
                    sem="acquire",
                    scope="gpu",
                )
                if encoded != 0:
                    old = cute.arch.atomic_cas(
                        self.data.iterator + 1,
                        cmp=head,
                        val=head + 1,
                        sem="acq_rel",
                        scope="gpu",
                    )
                    if old == head:
                        task = (
                            encoded - 1
                        ) * self.fc2_features + head % self.fc2_features
                    else:
                        retry = True
        return task

    @cute.jit
    def publish_row(self, row):
        start = cute.arch.atomic_add(
            self.data.iterator + 2, cutlass.Int32(1), sem="relaxed", scope="gpu"
        )
        # The host bound covers every actual row, each of which publishes once.
        # No circular reuse or wait for a scheduler to free queue capacity.
        cute.arch.store(
            self.data.iterator + self.queue_offset + start,
            row + 1,
            sem="release",
            scope="gpu",
        )
        cute.arch.atomic_add(
            self.data.iterator + 3, cutlass.Int32(1), sem="acq_rel", scope="gpu"
        )


def make_storage(params, ext):
    stages = params.num_sched_stages
    fields = ext.WorkTileInfo.TotalFields

    @cute.struct
    class Storage:
        sched_mbar: cute.struct.MemRange[cutlass.Int64, stages * 2]
        sched_buf: cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, fields * stages], 16
        ]
        cluster_pipeline_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        cluster_broadcast_slot: cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, 1], 16
        ]
        clc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        clc_response: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
        completed: cutlass.Int32
        claim_mbar: cute.struct.MemRange[cutlass.Int64, 4]
        claim_buffer: cute.struct.MemRange[cutlass.Int32, 2]

    return Storage


@cute.jit
def initialize(scheduler, storage):
    size = scheduler.params.cluster_shape_mn[0] * scheduler.params.cluster_shape_mn[1]
    scheduler._cluster_pipeline = pipeline.PipelineAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * size),
        barrier_storage=storage.cluster_pipeline_mbar.data_ptr(),
        defer_sync=True,
    )
    if cute.arch.thread_idx()[0] == 0:
        storage.completed = cutlass.Int32(0)


@cute.jit
def broadcast(pipe, ds, value, size: cutlass.Constexpr):
    """Existing cluster scheduler broadcast protocol, without an implicit fetch."""
    slot = cute.make_tensor(ds.broadcast_ptr, cute.make_layout((1,)))
    lane = cute.arch.lane_idx()
    if ds.is_leader_cta:
        pipe.producer_acquire(ds.producer_state)
        barrier = pipe.sync_object_full.get_barrier(ds.producer_state.index)
        value = cute.arch.shuffle_sync(
            value, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31
        )
        if lane < size:
            store_i32_to_peer_cluster_smem_async(ds.broadcast_ptr, value, barrier, lane)
            mbarrier_arrive_expect_tx_on_peer(barrier, cutlass.Int32(4), lane)
    ds.producer_state.advance()
    pipe.consumer_wait(ds.consumer_state)
    result = slot[0]
    cute.arch.fence_acq_rel_cta()
    pipe.sync_object_empty.arrive(ds.consumer_state.index, cutlass.Int32(0))
    ds.consumer_state.advance()
    return result, ds


@dsl_user_op
def cancelled_bundle(response, cluster_x: int, *, loc=None, ip=None):
    """Only decode CTA coordinates under the success predicate (PTX 8.6+).

    The installed generic clc_response helper extracts coordinates even for a
    failed result. Keep this private response decoder explicitly predicated.
    A negative result permanently disables subsequent requests by this worker.
    """
    shift = 0 if cluster_x == 1 else 1
    assert cluster_x in (1, 2)
    result = llvm.inline_asm(
        T.i32(),
        [llvm.ptrtoint(T.i32(), response.llvm_ptr, loc=loc, ip=ip)],
        "{ .reg .b128 r; .reg .pred p; .reg .b32 x, y, z; "
        "ld.shared.b128 r, [$1]; "
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r; "
        "mov.b32 $0, -1; "
        "@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x,y,z,_}, r; "
        f"@p shr.u32 x, x, {shift}; "
        "@p add.u32 $0, x, z; }",
        "=r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@cute.jit
def phase_work(scheduler, index, fc1: cutlass.Constexpr):
    """Existing work-record layout, decoded without a forward-only cursor."""
    params = scheduler.params
    if cutlass.const_expr(fc1):
        phase = BlockPhase.Linear1
        features = scheduler._num_fc1_intermediate_blocks
    else:
        phase = BlockPhase.Linear2
        features = scheduler._num_fc2_hidden_blocks
    expert = cutlass.Int32(0)
    begin = cutlass.Int32(0)
    data = cutlass.Int32(0)
    scale = cutlass.Int32(0)
    blocks = cutlass.Int32(0)
    count = params.expert_token_sizes[expert]
    token_blocks = cute.ceil_div(count, params.cluster_tile_m)
    end = token_blocks * features
    while index >= end and expert < params.expert_cnt:
        data += (
            cute.ceil_div(count, params.token_padding_block)
            * params.token_padding_block
        )
        scale += cute.ceil_div(count, params.sf_padding_block) * params.sf_padding_block
        blocks += token_blocks
        expert += 1
        begin = end
        count = cutlass.Int32(0)
        if expert < params.expert_cnt:
            count = params.expert_token_sizes[expert]
        token_blocks = cute.ceil_div(count, params.cluster_tile_m)
        end += token_blocks * features
    work = scheduler._ext.WorkTileInfo(
        expert_idx=cutlass.Int32(-1),
        tile_m_idx=cutlass.Int32(0),
        tile_n_idx=cutlass.Int32(0),
        cumulative_data_physical_row=cutlass.Int32(0),
        cumulative_sf_physical_row=cutlass.Int32(0),
        cumulative_token_block_count=cutlass.Int32(0),
        valid_tokens_in_cta_tile=cutlass.Int32(0),
        phase_and_peek=cutlass.Int32(BlockPhase.None_),
    )
    if expert < params.expert_cnt:
        local = index - begin
        token_block = local // features
        feature_block = local - token_block * features
        tile_token = (
            token_block * params.cluster_shape_mn[0] + scheduler.cta_id_in_cluster[0]
        )
        tile_feature = (
            feature_block * params.cluster_shape_mn[1] + scheduler.cta_id_in_cluster[1]
        )
        valid = cutlass.min(
            cutlass.max(count - tile_token * params.cta_tile_shape_mnk[0], 0),
            params.cta_tile_shape_mnk[0],
        )
        work = scheduler._ext.WorkTileInfo(
            expert_idx=expert,
            tile_m_idx=tile_feature,
            tile_n_idx=tile_token,
            cumulative_data_physical_row=data,
            cumulative_sf_physical_row=scale,
            cumulative_token_block_count=blocks,
            valid_tokens_in_cta_tile=valid,
            phase_and_peek=cutlass.Int32(phase),
        )
    return scheduler._ext.enrich_work_tile_info(work)


@cute.jit
def complete_fc1(workspace, counter, work, threshold: cutlass.Constexpr, in_bound):
    if in_bound:
        row = work.cumulative_token_block_count + work.tile_n_idx
        old = cute.arch.atomic_add(
            counter.iterator + row, cutlass.Int32(1), sem="acq_rel", scope="gpu"
        )
        if old + 1 == threshold:
            workspace.publish_row(row)


@cute.jit
def claim_and_publish_bundle(
    cluster_pipeline,
    storage,
    workspace,
    claim_pipe,
    ds,
    claim_state,
    bundle,
    bundles,
    fc1_tasks,
    cluster_x: cutlass.Constexpr,
    bundle_size: cutlass.Constexpr,
):
    owned = cutlass.Int32(0)
    if (
        ds.is_leader_cta
        and cute.arch.lane_idx() == 0
        and bundle >= 0
        and bundle < bundles
    ):
        owned = cutlass.Int32(workspace.claim(bundle))
    owned, ds = broadcast(cluster_pipeline, ds, owned, cluster_x)
    if owned != 0:
        for sub in cutlass.range_constexpr(bundle_size):
            index = bundle * bundle_size + sub
            if index < fc1_tasks:
                claim_state = publish_claim(storage, claim_pipe, claim_state, index)
    return ds, claim_state


@cute.jit
def run(
    scheduler,
    storage,
    workspace,
    clc_pipe,
    claim_pipe,
    cluster_x: cutlass.Constexpr,
    bundle_size: cutlass.Constexpr,
):
    ds = scheduler._dynamic_state
    leader = ds.is_leader_cta
    lane = cute.arch.lane_idx()
    elected = leader and lane == 0
    rows = cutlass.Int32(0)
    for expert in cutlass.range(scheduler.params.expert_cnt):
        rows += cute.ceil_div(
            scheduler.params.expert_token_sizes[expert], scheduler.params.cluster_tile_m
        )
    fc1_tasks = rows * scheduler._num_fc1_intermediate_blocks
    fc2_tasks = rows * scheduler._num_fc2_hidden_blocks
    bundles = cute.ceil_div(fc1_tasks, bundle_size)
    producer = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.ProducerConsumer, 1
    )
    consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
    response = storage.clc_response.data_ptr()
    x, _, z = cute.arch.block_idx()
    initial_bundle = x // cluster_x + z
    clc_enabled = True
    scan_exhausted = False
    claim_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
    ds, claim_state = claim_and_publish_bundle(
        scheduler._cluster_pipeline,
        storage,
        workspace,
        claim_pipe,
        ds,
        claim_state,
        initial_bundle,
        bundles,
        fc1_tasks,
        cluster_x,
        bundle_size,
    )
    done = False
    while not done:
        task = cutlass.Int32(-1)
        if elected:
            task = workspace.pop()
        task, ds = broadcast(scheduler._cluster_pipeline, ds, task, cluster_x)
        if task >= 0:
            claim_state = publish_claim(
                storage, claim_pipe, claim_state, fc1_tasks + task
            )

        terminal = cutlass.Int32(0)
        if elected:
            published = cute.arch.load(
                workspace.data.iterator + 3, cutlass.Int32, sem="acquire", scope="gpu"
            )
            head = cute.arch.load(
                workspace.data.iterator + 1, cutlass.Int32, sem="acquire", scope="gpu"
            )
            terminal = cutlass.Int32(published == rows and head == fc2_tasks)
        terminal, ds = broadcast(scheduler._cluster_pipeline, ds, terminal, cluster_x)
        if terminal != 0:
            done = True
        else:
            bundle = cutlass.Int32(-1)
            if clc_enabled:
                if leader:
                    clc_pipe.producer_acquire(producer)
                    with cute.arch.elect_one():
                        cute.arch.issue_clc_query(
                            clc_pipe.producer_get_barrier(producer), response
                        )
                    producer.advance()
                clc_pipe.consumer_wait(consumer)
                bundle = cancelled_bundle(response, cluster_x)
                cute.arch.fence_acq_rel_cta()
                clc_pipe.consumer_release(consumer)
                consumer.advance()
                if bundle < 0:
                    clc_enabled = False
            if not clc_enabled and not scan_exhausted:
                next_bundle = cutlass.Int32(-1)
                if elected:
                    next_bundle = cute.arch.atomic_add(
                        workspace.data.iterator,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                next_bundle, ds = broadcast(
                    scheduler._cluster_pipeline, ds, next_bundle, cluster_x
                )
                if next_bundle < bundles:
                    bundle = next_bundle
                else:
                    scan_exhausted = True
            # The bundle dies after publication instead of crossing the loop header.
            ds, claim_state = claim_and_publish_bundle(
                scheduler._cluster_pipeline,
                storage,
                workspace,
                claim_pipe,
                ds,
                claim_state,
                bundle,
                bundles,
                fc1_tasks,
                cluster_x,
                bundle_size,
            )

    if leader:
        clc_pipe.producer_tail(producer)
        scheduler._cluster_pipeline.producer_tail(ds.producer_state)
    claim_state = publish_claim(storage, claim_pipe, claim_state, cutlass.Int32(-1))
    claim_pipe.producer_tail(claim_state)


@cute.jit
def publish_claim(storage, pipe, state, index):
    pipe.producer_acquire(state)
    if cute.arch.lane_idx() == 0:
        (storage.claim_buffer.data_ptr() + state.index).store(index)
    cute.arch.sync_warp()
    pipe.producer_commit(state)
    state.advance()
    return state


@cute.jit
def schedule(scheduler, storage, claim_pipe):
    """Warp 7 maps claims while warp 8 handles CLC/global queue latency."""
    rows = cutlass.Int32(0)
    for expert in cutlass.range(scheduler.params.expert_cnt):
        rows += cute.ceil_div(
            scheduler.params.expert_token_sizes[expert], scheduler.params.cluster_tile_m
        )
    fc1_tasks = rows * scheduler._num_fc1_intermediate_blocks
    state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
    submitted = cutlass.Int32(0)
    done = False
    work = scheduler.current_work
    while not done:
        claim_pipe.consumer_wait(state)
        encoded = (storage.claim_buffer.data_ptr() + state.index).load()
        cute.arch.fence_acq_rel_cta()
        claim_pipe.consumer_release(state)
        state.advance()
        if encoded < 0:
            done = True
        else:
            if encoded < fc1_tasks:
                work = phase_work(scheduler, encoded, True)
            else:
                work = phase_work(scheduler, encoded - fc1_tasks, False)
            scheduler._ext.prefetch_for_expert(work.expert_idx)
            scheduler.current_work = work
            scheduler.publish_work()
            submitted += 1
    # Every intermediate claim has now been mapped. Each CTA waits for its
    # own epilogue, including queue publication/flags, before publishing DONE.
    while (
        cute.arch.load(storage.completed.ptr, cutlass.Int32, sem="acquire", scope="cta")
        < submitted
    ):
        pass
    scheduler.current_work = phase_work(scheduler, fc1_tasks, True)
    scheduler.publish_work()
    scheduler.produce_tail()
