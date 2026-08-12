"""Blackwell persistent work-ID claim backends."""

import dataclasses
from typing import List, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Boolean, Int32, extract_mlir_values, new_from_mlir_values

from ...helpers.ptx_helpers import mbarrier_arrive_expect_tx_on_peer, store_i32_to_peer_cluster_smem_async


class GridStrideWorkIdState:
    """Register state for one monotonic grid-stride work-ID stream."""

    def __init__(self, next_work_id: Int32, work_id_stride: Int32) -> None:
        self.next_work_id = next_work_id
        self.work_id_stride = work_id_stride

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        values.extend(extract_mlir_values(self.next_work_id))
        values.extend(extract_mlir_values(self.work_id_stride))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "GridStrideWorkIdState":
        next_work_id_value_count = len(extract_mlir_values(self.next_work_id))
        stride_value_count = len(extract_mlir_values(self.work_id_stride))
        expected_value_count = next_work_id_value_count + stride_value_count
        if len(values) != expected_value_count:
            raise ValueError(
                f"GridStrideWorkIdState MLIR value count mismatch: expected {expected_value_count}, got {len(values)}."
            )
        return type(self)(
            next_work_id=new_from_mlir_values(self.next_work_id, values[:next_work_id_value_count]),
            work_id_stride=new_from_mlir_values(self.work_id_stride, values[next_work_id_value_count:]),
        )


class AtomicCounterWorkIdState:
    """Cluster-wide state for one of several contiguous atomic work-ID streams."""

    def __init__(
        self,
        counter_pointer: cute.Pointer,
        counter_count: int,
        broadcast_pointer: cute.Pointer,
        is_leader_cta: Boolean,
        cluster_pipeline: pipeline.PipelineAsync,
        producer_state,
        consumer_state,
        cluster_size: int,
    ) -> None:
        if isinstance(counter_count, bool) or not isinstance(counter_count, int) or counter_count <= 0:
            raise ValueError("counter_count must be a positive Python int.")
        self.counter_pointer = counter_pointer
        self.counter_count = counter_count
        self.broadcast_pointer = broadcast_pointer
        self.is_leader_cta = is_leader_cta
        self.cluster_pipeline = cluster_pipeline
        self.producer_state = producer_state
        self.consumer_state = consumer_state
        self.cluster_size = cluster_size

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in (
            self.counter_pointer,
            self.broadcast_pointer,
            self.is_leader_cta,
            self.producer_state,
            self.consumer_state,
        ):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "AtomicCounterWorkIdState":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(field, values[value_index : value_index + field_value_count])
            value_index += field_value_count
            return result

        result = type(self)(
            counter_pointer=rebuild(self.counter_pointer),
            counter_count=self.counter_count,
            broadcast_pointer=rebuild(self.broadcast_pointer),
            is_leader_cta=rebuild(self.is_leader_cta),
            cluster_pipeline=self.cluster_pipeline,
            producer_state=rebuild(self.producer_state),
            consumer_state=rebuild(self.consumer_state),
            cluster_size=self.cluster_size,
        )
        if value_index != len(values):
            raise ValueError(
                f"AtomicCounterWorkIdState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return result


@dataclasses.dataclass(frozen=True)
class GridWorkId:
    """One CTA-specific coordinate claimed from a three-dimensional grid."""

    grid_m: Int32
    grid_n: Int32
    grid_l: Int32
    is_valid: Boolean

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in (self.grid_m, self.grid_n, self.grid_l, self.is_valid):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "GridWorkId":
        if len(values) != 4:
            raise ValueError(f"GridWorkId expects four MLIR values, got {len(values)}.")
        fields = (self.grid_m, self.grid_n, self.grid_l, self.is_valid)
        return type(self)(*(new_from_mlir_values(field, [value]) for field, value in zip(fields, values)))


class ClusterLaunchControlWorkIdState:
    """Cluster-wide state for hardware-assisted grid-coordinate claims."""

    def __init__(
        self,
        response_pending: Boolean,
        grid_m: Int32,
        grid_n: Int32,
        grid_l: Int32,
        response_is_valid: Boolean,
        cta_coord_in_cluster: cute.Coord,
        cluster_pipeline: pipeline.PipelineClcFetchAsync,
        producer_state,
        consumer_state,
        is_leader_cta: Boolean,
        response_pointer: cute.Pointer,
    ) -> None:
        self.response_pending = response_pending
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.grid_l = grid_l
        self.response_is_valid = response_is_valid
        self.cta_coord_in_cluster = cta_coord_in_cluster
        self.cluster_pipeline = cluster_pipeline
        self.producer_state = producer_state
        self.consumer_state = consumer_state
        self.is_leader_cta = is_leader_cta
        self.response_pointer = response_pointer

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in (
            self.response_pending,
            self.grid_m,
            self.grid_n,
            self.grid_l,
            self.response_is_valid,
            self.cta_coord_in_cluster,
            self.producer_state,
            self.consumer_state,
            self.is_leader_cta,
            self.response_pointer,
        ):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "ClusterLaunchControlWorkIdState":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            rebuilt_field = new_from_mlir_values(field, values[value_index : value_index + field_value_count])
            value_index += field_value_count
            return rebuilt_field

        result = type(self)(
            response_pending=rebuild(self.response_pending),
            grid_m=rebuild(self.grid_m),
            grid_n=rebuild(self.grid_n),
            grid_l=rebuild(self.grid_l),
            response_is_valid=rebuild(self.response_is_valid),
            cta_coord_in_cluster=rebuild(self.cta_coord_in_cluster),
            cluster_pipeline=self.cluster_pipeline,
            producer_state=rebuild(self.producer_state),
            consumer_state=rebuild(self.consumer_state),
            is_leader_cta=rebuild(self.is_leader_cta),
            response_pointer=rebuild(self.response_pointer),
        )
        if value_index != len(values):
            raise ValueError(
                f"ClusterLaunchControlWorkIdState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return result


@cute.jit
def _claim_grid_stride_work_id(work_id_state: GridStrideWorkIdState) -> Tuple[Int32, GridStrideWorkIdState]:
    """Claim the next ID from one monotonic grid-stride stream."""
    linear_work_id = work_id_state.next_work_id
    work_id_state.next_work_id = linear_work_id + work_id_state.work_id_stride
    return linear_work_id, work_id_state


@cute.jit
def _claim_atomic_counter_work_id(
    work_id_state: AtomicCounterWorkIdState, atomic_counter_index=0
) -> Tuple[Int32, AtomicCounterWorkIdState]:
    """Claim from one selected counter and broadcast the ID within the cluster."""
    invalid_static_index = isinstance(atomic_counter_index, int) and (
        atomic_counter_index < 0 or atomic_counter_index >= work_id_state.counter_count
    )
    if cutlass.const_expr(invalid_static_index):
        raise ValueError(
            f"atomic_counter_index must be in [0, {work_id_state.counter_count}), got {atomic_counter_index}."
        )
    broadcast_tensor = cute.make_tensor(work_id_state.broadcast_pointer, cute.make_layout((1,)))
    cluster_pipeline = work_id_state.cluster_pipeline
    selected_counter_pointer = work_id_state.counter_pointer + Int32(atomic_counter_index)

    if work_id_state.is_leader_cta:
        cluster_pipeline.producer_acquire(work_id_state.producer_state)
        full_barrier_pointer = cluster_pipeline.sync_object_full.get_barrier(work_id_state.producer_state.index)
        thread_idx, _, _ = cute.arch.thread_idx()
        lane_idx = thread_idx % Int32(32)
        atomic_work_id = Int32(0)
        if lane_idx == Int32(0):
            atomic_work_id = cute.arch.atomic_add(selected_counter_pointer, Int32(1))
        atomic_work_id = cute.arch.shuffle_sync(atomic_work_id, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31)
        if lane_idx < Int32(work_id_state.cluster_size):
            store_i32_to_peer_cluster_smem_async(
                work_id_state.broadcast_pointer, atomic_work_id, full_barrier_pointer, lane_idx
            )
            mbarrier_arrive_expect_tx_on_peer(full_barrier_pointer, Int32(4), lane_idx)
    work_id_state.producer_state.advance()

    cluster_pipeline.consumer_wait(work_id_state.consumer_state)
    linear_work_id = broadcast_tensor[0]
    cute.arch.fence_acq_rel_cta()
    cluster_pipeline.sync_object_empty.arrive(work_id_state.consumer_state.index, Int32(0))
    work_id_state.consumer_state.advance()
    return linear_work_id, work_id_state


@cute.jit
def _claim_cluster_launch_control_work_id(
    work_id_state: ClusterLaunchControlWorkIdState,
) -> Tuple[GridWorkId, ClusterLaunchControlWorkIdState]:
    """Claim the next canceled cluster and return this CTA's grid coordinate."""
    use_bootstrap = work_id_state.response_pending
    state_before_bootstrap = work_id_state
    if use_bootstrap:
        work_id_state.response_pending = Boolean(False)
    else:
        work_id_state = state_before_bootstrap

    state_before_query = work_id_state
    if not use_bootstrap:
        state_before_leader_query = work_id_state
        if work_id_state.is_leader_cta:
            work_id_state.cluster_pipeline.producer_acquire(work_id_state.producer_state)
            response_barrier = work_id_state.cluster_pipeline.producer_get_barrier(work_id_state.producer_state)
            with cute.arch.elect_one():
                cute.arch.issue_clc_query(response_barrier, work_id_state.response_pointer)
        else:
            work_id_state = state_before_leader_query
        work_id_state.producer_state.advance()

        work_id_state.cluster_pipeline.consumer_wait(work_id_state.consumer_state)
        (cluster_origin_m, cluster_origin_n, grid_l, response_is_valid) = cute.arch.clc_response(
            work_id_state.response_pointer
        )
        cute.arch.fence_acq_rel_cta()
        work_id_state.cluster_pipeline.consumer_release(work_id_state.consumer_state)
        work_id_state.consumer_state.advance()

        work_id_state.grid_m = cluster_origin_m + work_id_state.cta_coord_in_cluster[0]
        work_id_state.grid_n = cluster_origin_n + work_id_state.cta_coord_in_cluster[1]
        work_id_state.grid_l = grid_l
        work_id_state.response_is_valid = response_is_valid != Int32(0)
    else:
        work_id_state = state_before_query

    return (
        GridWorkId(
            grid_m=work_id_state.grid_m,
            grid_n=work_id_state.grid_n,
            grid_l=work_id_state.grid_l,
            is_valid=work_id_state.response_is_valid,
        ),
        work_id_state,
    )


@cute.jit
def claim_work_id(work_id_state, atomic_counter_index=0):
    """Claim the next work ID using the backend encoded by the state type."""
    if cutlass.const_expr(isinstance(work_id_state, GridStrideWorkIdState)):
        return _claim_grid_stride_work_id(work_id_state)
    if cutlass.const_expr(isinstance(work_id_state, AtomicCounterWorkIdState)):
        return _claim_atomic_counter_work_id(work_id_state, atomic_counter_index)
    if cutlass.const_expr(isinstance(work_id_state, ClusterLaunchControlWorkIdState)):
        return _claim_cluster_launch_control_work_id(work_id_state)
    raise TypeError(f"Unsupported work-ID state: {type(work_id_state).__name__}.")


__all__ = [
    "AtomicCounterWorkIdState",
    "ClusterLaunchControlWorkIdState",
    "GridStrideWorkIdState",
    "GridWorkId",
    "claim_work_id",
]
