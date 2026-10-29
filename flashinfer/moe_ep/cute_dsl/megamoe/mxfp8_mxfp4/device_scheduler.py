# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device implementation of the FC1/FC2 contract in :mod:`scheduler`.

The scheduler warp keeps group/phase/expert state in registers and publishes
an eight-int32 ``WorkRecord`` through an async SMEM pipeline. It has no
vendored MegaMoE dependency.

In swap-AB coordinates, ``tile_m_idx`` selects features and ``tile_n_idx``
selects tokens. Scheduler parameters use logical token/feature axis names.
"""

from enum import IntEnum
from typing import List, Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    Integer,
    T,
    const_expr,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)


WORK_RECORD_WORDS = 8
PHASE_BITS = 16
PHASE_MASK = (1 << PHASE_BITS) - 1
PEEK_READY_BIT = 1 << PHASE_BITS
DONE_EXPERT = -1
SCALE_VECTOR_SIZE = 32


class DevicePhase(IntEnum):
    NONE = 0
    FC1 = 1
    FC2 = 2


class DeviceWorkRecord:
    """Device form of ``scheduler.WorkRecord`` with identical wire order."""

    TotalFields = WORK_RECORD_WORDS
    TotalBytes = WORK_RECORD_WORDS * 4

    def __init__(
        self,
        expert_idx: Int32,
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        cumulative_data_physical_row: Int32,
        cumulative_sf_physical_row: Int32,
        cumulative_token_block_count: Int32,
        valid_tokens_in_cta_tile: Int32,
        phase_and_peek: Int32,
    ):
        self.expert_idx = expert_idx
        self.tile_m_idx = tile_m_idx
        self.tile_n_idx = tile_n_idx
        self.cumulative_data_physical_row = cumulative_data_physical_row
        self.cumulative_sf_physical_row = cumulative_sf_physical_row
        self.cumulative_token_block_count = cumulative_token_block_count
        self.valid_tokens_in_cta_tile = valid_tokens_in_cta_tile
        self.phase_and_peek = phase_and_peek

    @property
    def is_valid_tile(self) -> Boolean:
        return self.expert_idx >= Int32(0)

    @property
    def phase(self) -> Int32:
        return self.phase_and_peek & Int32(PHASE_MASK)

    @property
    def peek_ready(self) -> Boolean:
        return (self.phase_and_peek & Int32(PEEK_READY_BIT)) != Int32(0)

    @property
    def counter_slot(self) -> Int32:
        return self.cumulative_token_block_count + self.tile_n_idx

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for value in (
            self.expert_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.cumulative_data_physical_row,
            self.cumulative_sf_physical_row,
            self.cumulative_token_block_count,
            self.valid_tokens_in_cta_tile,
            self.phase_and_peek,
        ):
            values.extend(extract_mlir_values(value))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "DeviceWorkRecord":
        assert len(values) == WORK_RECORD_WORDS
        prototypes = (
            self.expert_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.cumulative_data_physical_row,
            self.cumulative_sf_physical_row,
            self.cumulative_token_block_count,
            self.valid_tokens_in_cta_tile,
            self.phase_and_peek,
        )
        fields = [
            new_from_mlir_values(p, [v])
            for p, v in zip(prototypes, values, strict=True)
        ]
        return DeviceWorkRecord(*fields)

    def to_rmem(self) -> cute.Tensor:
        rmem = cute.make_rmem_tensor((WORK_RECORD_WORDS,), Int32)
        rmem[0] = self.expert_idx
        rmem[1] = self.tile_m_idx
        rmem[2] = self.tile_n_idx
        rmem[3] = self.cumulative_data_physical_row
        rmem[4] = self.cumulative_sf_physical_row
        rmem[5] = self.cumulative_token_block_count
        rmem[6] = self.valid_tokens_in_cta_tile
        rmem[7] = self.phase_and_peek
        return rmem

    @classmethod
    def from_rmem(cls, rmem: cute.Tensor) -> "DeviceWorkRecord":
        return cls(*(rmem[i] for i in range(WORK_RECORD_WORDS)))

    @dsl_user_op
    @cute.jit
    def write_to_smem(self, smem, dependency, *, loc=None, ip=None) -> None:
        pipe, state = dependency
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
        )
        pipe.producer_acquire(state)
        cute.copy(copy_atom, self.to_rmem(), smem[(None, state.index)])
        cute.arch.fence_proxy("async.shared", space="cta")
        pipe.producer_commit(state)
        state.advance()

    @classmethod
    @dsl_user_op
    @cute.jit
    def read_from_smem(cls, smem, dependency, *, loc=None, ip=None):
        pipe, state = dependency
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
        )
        pipe.consumer_wait(state)
        rmem = cute.make_rmem_tensor((WORK_RECORD_WORDS,), Int32)
        cute.copy(copy_atom, smem[(None, state.index)], rmem)
        result = cls.from_rmem(rmem)
        cute.arch.fence_acq_rel_cta()
        pipe.consumer_release(state)
        state.advance()
        return result


class DeviceSchedulerParams:
    """Host-created values carried into the device scheduler.

    ``expert_count`` must be compile-time when allocating scheduler storage.
    ``expert_token_sizes`` contains one int32 count per expert. Data rows and
    scale rows have separate padding. Both operands use one scale factor
    per 32 values along K.
    """

    # load_balance_counter_ptr addresses one int32 for atomic_counter, or
    # two consecutive int32s (FC1, FC2) for alternating. Reset before launch.
    DEFAULT_NUM_STAGES = 2

    def __init__(
        self,
        *,
        expert_token_sizes: cute.Tensor,
        expert_count: int | Int32,
        gate_up_size: int | Int32,
        hidden_size: int | Int32,
        cta_tile_tokens: int,
        cta_tile_features: int,
        cluster_shape_token_feature: Tuple[int, int],
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        fc1_ready_counter_ptr,
        fc1_done_counter_ptr,
        fc2_ready_threshold: int | Int32,
        load_balance_mode: Literal[
            "static", "atomic_counter", "alternating"
        ] = "static",
        load_balance_counter_ptr=None,
        num_stages: int = DEFAULT_NUM_STAGES,
        claim_order: Literal["expert", "longest_first"] = "expert",
    ):
        if claim_order not in ("expert", "longest_first"):
            raise ValueError("claim_order must be expert or longest_first")
        if load_balance_mode not in ("static", "atomic_counter", "alternating"):
            raise ValueError(
                "load_balance_mode must be static, atomic_counter, or alternating"
            )
        if load_balance_mode != "static" and load_balance_counter_ptr is None:
            raise ValueError("dynamic claiming requires load_balance_counter_ptr")
        if cta_tile_tokens <= 0 or cta_tile_features <= 0:
            raise ValueError("CTA tile extents must be positive")
        if any(v <= 0 for v in cluster_shape_token_feature):
            raise ValueError("cluster extents must be positive")
        if group_hint <= 0 or token_padding_block <= 0 or sf_padding_block <= 0:
            raise ValueError("group and padding values must be positive")
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if isinstance(gate_up_size, int) and gate_up_size % SCALE_VECTOR_SIZE:
            raise ValueError("gate_up_size must preserve K32 scale grouping")
        if isinstance(hidden_size, int) and hidden_size % SCALE_VECTOR_SIZE:
            raise ValueError("hidden_size must preserve K32 scale grouping")

        self.expert_token_sizes = expert_token_sizes
        self.expert_count = expert_count
        self.gate_up_size = gate_up_size
        self.hidden_size = hidden_size
        self.cta_tile_tokens = cta_tile_tokens
        self.cta_tile_features = cta_tile_features
        self.cluster_shape_token_feature = cluster_shape_token_feature
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.fc1_ready_counter_ptr = fc1_ready_counter_ptr
        self.fc1_done_counter_ptr = fc1_done_counter_ptr
        self.fc2_ready_threshold = Int32(fc2_ready_threshold)
        self.claim_order = claim_order
        self.load_balance_mode = load_balance_mode
        self.load_balance_counter_ptr = load_balance_counter_ptr
        self.num_stages = num_stages

    @property
    def cluster_tile_tokens(self) -> int:
        return self.cta_tile_tokens * self.cluster_shape_token_feature[0]

    @property
    def cluster_tile_features(self) -> int:
        return self.cta_tile_features * self.cluster_shape_token_feature[1]

    def get_grid_shape(self, max_active_clusters: int) -> Tuple[int, int, int]:
        token_ctas, feature_ctas = self.cluster_shape_token_feature
        return feature_ctas, token_ctas, max_active_clusters

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for value in (self.expert_count, self.gate_up_size, self.hidden_size):
            if isinstance(value, Int32):
                values.extend(extract_mlir_values(value))
        for value in (
            self.expert_token_sizes,
            self.fc1_ready_counter_ptr,
            self.fc1_done_counter_ptr,
            self.fc2_ready_threshold,
        ):
            if value is not None:
                values.extend(extract_mlir_values(value))
        if self.load_balance_mode != "static":
            values.extend(extract_mlir_values(self.load_balance_counter_ptr))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "DeviceSchedulerParams":
        result = DeviceSchedulerParams.__new__(DeviceSchedulerParams)
        idx = 0

        def take(prototype):
            nonlocal idx
            if prototype is None:
                return None
            count = len(extract_mlir_values(prototype))
            value = new_from_mlir_values(prototype, values[idx : idx + count])
            idx += count
            return value

        for name in ("expert_count", "gate_up_size", "hidden_size"):
            prototype = getattr(self, name)
            rebound = take(prototype) if isinstance(prototype, Int32) else prototype
            setattr(result, name, rebound)
        result.expert_token_sizes = take(self.expert_token_sizes)
        result.fc1_ready_counter_ptr = take(self.fc1_ready_counter_ptr)
        result.fc1_done_counter_ptr = take(self.fc1_done_counter_ptr)
        result.fc2_ready_threshold = take(self.fc2_ready_threshold)
        result.load_balance_counter_ptr = (
            take(self.load_balance_counter_ptr)
            if self.load_balance_mode != "static"
            else None
        )
        for name in (
            "cta_tile_tokens",
            "cta_tile_features",
            "cluster_shape_token_feature",
            "group_hint",
            "token_padding_block",
            "sf_padding_block",
            "load_balance_mode",
            "claim_order",
            "num_stages",
        ):
            setattr(result, name, getattr(self, name))
        assert idx == len(values)
        return result


class _SchedulerState:
    _FIELDS: tuple[str, ...] = (
        "group_first_expert",
        "group_last_expert",
        "phase",
        "expert_idx",
        "expert_pos",
        "expert_tile_start",
        "expert_tile_end",
        "group_fc1_end",
        "group_end",
        "cumulative_fc1_tiles",
        "cumulative_fc2_tiles",
        "data_row",
        "sf_row",
        "token_block_row",
        "token_count",
        "token_blocks",
        "static_tile_idx",
    )

    def __init__(self, *values):
        assert len(values) == len(self._FIELDS)
        for name, value in zip(self._FIELDS, values, strict=True):
            setattr(self, name, value)

    @classmethod
    def initial(cls, *, static_tile_idx: Int32) -> "_SchedulerState":
        defaults = dict(
            phase=Int32(DevicePhase.FC1),
            expert_idx=Int32(-1),
            expert_pos=Int32(-1),
            static_tile_idx=static_tile_idx,
        )
        return cls(
            *(defaults[name] if name in defaults else Int32(0) for name in cls._FIELDS)
        )

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for name in self._FIELDS:
            values.extend(extract_mlir_values(getattr(self, name)))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "_SchedulerState":
        assert len(values) == len(self._FIELDS)
        rebound = [
            new_from_mlir_values(getattr(self, name), [value])
            for name, value in zip(self._FIELDS, values, strict=True)
        ]
        return type(self)(*rebound)


class _AlternatingState(_SchedulerState):
    _FIELDS: tuple[str, ...] = (
        "pending",
        "total_fc1",
        "total_fc2",
        "fc1_exhausted",
        "fc2_exhausted",
    )


ALTERNATING_DONE = (1 << 31) - 1


class _AtomicClaimState:
    def __init__(
        self, counter_ptr, broadcast_ptr, is_leader, producer, consumer, cached
    ):
        self.counter_ptr = counter_ptr
        self.broadcast_ptr = broadcast_ptr
        self.is_leader = is_leader
        self.producer = producer
        self.consumer = consumer
        self.cached = cached

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for value in (
            self.counter_ptr,
            self.broadcast_ptr,
            self.is_leader,
            self.producer,
            self.consumer,
            self.cached,
        ):
            values.extend(extract_mlir_values(value))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "_AtomicClaimState":
        idx = 0

        def take(prototype):
            nonlocal idx
            count = len(extract_mlir_values(prototype))
            value = new_from_mlir_values(prototype, values[idx : idx + count])
            idx += count
            return value

        result = _AtomicClaimState(
            take(self.counter_ptr),
            take(self.broadcast_ptr),
            take(self.is_leader),
            take(self.producer),
            take(self.consumer),
            take(self.cached),
        )
        assert idx == len(values)
        return result


@dsl_user_op
def _store_peer_i32(smem_ptr, value, barrier_ptr, peer_rank, *, loc=None, ip=None):
    smem_addr = llvm.ptrtoint(T.i32(), smem_ptr.llvm_ptr, loc=loc, ip=ip)
    barrier_addr = llvm.ptrtoint(T.i32(), barrier_ptr.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            smem_addr,
            Int32(value).ir_value(loc=loc, ip=ip),
            barrier_addr,
            Int32(peer_rank).ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_data;
            .reg .u32 remote_barrier;
            mapa.shared::cluster.u32 remote_data, $0, $3;
            mapa.shared::cluster.u32 remote_barrier, $2, $3;
            st.async.shared::cluster.mbarrier::complete_tx::bytes.u32
                [remote_data], $1, [remote_barrier];
        }}""",
        constraints="r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _expect_peer_i32(barrier_ptr, peer_rank, *, loc=None, ip=None):
    barrier_addr = llvm.ptrtoint(T.i32(), barrier_ptr.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            barrier_addr,
            Int32(peer_rank).ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_barrier;
            mapa.shared::cluster.u32 remote_barrier, $0, $1;
            mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remote_barrier], 4;
        }}""",
        constraints="r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class DeviceSchedulerConsumer:
    def __init__(self, pipe, smem, num_stages: int):
        self._pipeline = pipe
        self._smem = smem
        self._state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_stages
        )

    def __extract_mlir_values__(self) -> List[ir.Value]:
        return list(extract_mlir_values(self._state))

    def __new_from_mlir_values__(self, values: List[ir.Value]):
        result = DeviceSchedulerConsumer.__new__(DeviceSchedulerConsumer)
        result._pipeline = self._pipeline
        result._smem = self._smem
        result._state = new_from_mlir_values(self._state, values)
        return result

    @dsl_user_op
    @cute.jit
    def consume_work(self, *, loc=None, ip=None) -> DeviceWorkRecord:
        return DeviceWorkRecord.read_from_smem(
            self._smem, (self._pipeline, self._state), loc=loc, ip=ip
        )


class PersistentDeviceScheduler:
    """CuTe-DSL scheduler with static, atomic, and alternating cluster claims."""

    def __init__(
        self,
        params,
        num_persistent_clusters,
        cta_coord,
        current_work,
        state,
        atomic_state,
        fc1_feature_blocks,
        fc2_feature_blocks,
        sched_pipeline,
        smem,
        producer_state,
        storage,
        cluster_pipeline,
    ):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self.cta_coord = cta_coord
        self.current_work = current_work
        self._state = state
        self._atomic_state = atomic_state
        if params.load_balance_mode == "alternating":
            self._fc2_state = _SchedulerState(
                *(getattr(state, name) for name in state._FIELDS)
            )
            self._fc2_state.phase = Int32(DevicePhase.FC2)
            self._alternating = _AlternatingState(
                *(Int32(0) for _ in _AlternatingState._FIELDS)
            )
        self._fc1_feature_blocks = fc1_feature_blocks
        self._fc2_feature_blocks = fc2_feature_blocks
        self._pipeline = sched_pipeline
        self._smem = smem
        self._producer_state = producer_state
        self._storage = storage
        self._order = storage.order.get_tensor(cute.make_layout(params.expert_count))
        self._pool_data_row = storage.pool_data_row.get_tensor(
            cute.make_layout(params.expert_count)
        )
        self._pool_sf_row = storage.pool_sf_row.get_tensor(
            cute.make_layout(params.expert_count)
        )
        self._pool_token_block_row = storage.pool_token_block_row.get_tensor(
            cute.make_layout(params.expert_count)
        )
        self._cluster_pipeline = cluster_pipeline
        self._first_claim_pending = False

    @staticmethod
    def make_storage_struct(params: DeviceSchedulerParams):
        num_stages = params.num_stages
        expert_count = params.expert_count

        @cute.struct
        class StaticSchedulerStorage:
            order: cute.struct.MemRange[cutlass.Int32, expert_count]
            pool_data_row: cute.struct.MemRange[cutlass.Int32, expert_count]
            pool_sf_row: cute.struct.MemRange[cutlass.Int32, expert_count]
            pool_token_block_row: cute.struct.MemRange[cutlass.Int32, expert_count]
            sched_mbar: cute.struct.MemRange[cutlass.Int64, num_stages * 2]
            sched_buf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, WORK_RECORD_WORDS * num_stages],
                16,
            ]

        @cute.struct
        class AtomicSchedulerStorage:
            order: cute.struct.MemRange[cutlass.Int32, expert_count]
            pool_data_row: cute.struct.MemRange[cutlass.Int32, expert_count]
            pool_sf_row: cute.struct.MemRange[cutlass.Int32, expert_count]
            pool_token_block_row: cute.struct.MemRange[cutlass.Int32, expert_count]
            sched_mbar: cute.struct.MemRange[cutlass.Int64, num_stages * 2]
            sched_buf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, WORK_RECORD_WORDS * num_stages],
                16,
            ]
            cluster_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            cluster_claim: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 16]

        if params.load_balance_mode == "static":
            return StaticSchedulerStorage
        return AtomicSchedulerStorage

    @staticmethod
    @dsl_user_op
    def create(
        params: DeviceSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        storage,
        num_consumer_threads: int,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "PersistentDeviceScheduler":
        if num_consumer_threads <= 0:
            raise ValueError("num_consumer_threads must be positive")

        feature_ctas = params.cluster_shape_token_feature[1]
        token_ctas = params.cluster_shape_token_feature[0]
        bidx, bidy, bidz = block_idx
        # Launch grid is (feature CTA, token CTA, persistent cluster).
        cta_coord = (
            Int32(bidy % token_ctas),
            Int32(bidx % feature_ctas),
        )
        cluster_size = token_ctas * feature_ctas
        num_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cluster_size

        state = _SchedulerState.initial(static_tile_idx=Int32(bidz))
        current_work = DeviceWorkRecord(
            Int32(DONE_EXPERT),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(DevicePhase.NONE),
        )

        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=params.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_consumer_threads
            ),
            barrier_storage=storage.sched_mbar.data_ptr(),
            defer_sync=True,
        )
        smem = cute.make_tensor(
            storage.sched_buf.data_ptr(),
            cute.make_layout(
                (WORK_RECORD_WORDS, params.num_stages),
                stride=(1, WORK_RECORD_WORDS),
            ),
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, params.num_stages
        )

        atomic_state = None
        cluster_pipeline = None
        if const_expr(params.load_balance_mode != "static"):
            # Initialize with the other pipelines, before the driver's
            # pipeline_init_arrive/wait publishes the mbarriers to peers.
            cluster_pipeline = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, cute.arch.WARP_SIZE * cluster_size
                ),
                barrier_storage=storage.cluster_mbar.data_ptr(),
                defer_sync=True,
            )
            atomic_state = _AtomicClaimState(
                params.load_balance_counter_ptr,
                storage.cluster_claim.data_ptr(),
                cta_coord[0] + cta_coord[1] == Int32(0),
                pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1),
                pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1),
                Int32(0),
            )

        fc1_feature_blocks = (
            params.gate_up_size + params.cluster_tile_features - 1
        ) // params.cluster_tile_features
        fc2_feature_blocks = (
            params.hidden_size + params.cluster_tile_features - 1
        ) // params.cluster_tile_features
        return PersistentDeviceScheduler(
            params,
            num_clusters,
            cta_coord,
            current_work,
            state,
            atomic_state,
            fc1_feature_blocks,
            fc2_feature_blocks,
            sched_pipeline,
            smem,
            producer_state,
            storage,
            cluster_pipeline,
        )

    @property
    def expert_count(self):
        return self.params.expert_count

    def make_consumer(self) -> DeviceSchedulerConsumer:
        return DeviceSchedulerConsumer(
            self._pipeline, self._smem, self.params.num_stages
        )

    @dsl_user_op
    @cute.jit
    def publish_work(self, *, loc=None, ip=None) -> None:
        self.current_work.write_to_smem(
            self._smem, (self._pipeline, self._producer_state), loc=loc, ip=ip
        )

    @dsl_user_op
    @cute.jit
    def produce_tail(self, *, loc=None, ip=None) -> None:
        self._pipeline.producer_tail(self._producer_state)

    @dsl_user_op
    @cute.jit
    def _init_expert_tables(self, *, loc=None, ip=None):
        """Warp-owned permutation and index-order prefixes. No pool data moves.

        Each lane handles one expert, so a chunk contains WARP_SIZE experts.
        Warp scans compute offsets, carrying totals into the next chunk.
        Masked lanes contribute zero in the final partial chunk. Stable
        ranking uses warp shuffles and breaks ties by expert index.
        """
        params = self.params
        lane = cute.arch.lane_idx()
        data_base = Int32(0)
        sf_base = Int32(0)
        block_base = Int32(0)
        for base in cutlass.range_constexpr(
            (params.expert_count + (cute.arch.WARP_SIZE - 1)) // cute.arch.WARP_SIZE
        ):
            expert = Int32(base * cute.arch.WARP_SIZE) + lane
            count = Int32(0)
            if expert < params.expert_count:
                count = params.expert_token_sizes[expert]
            data = (
                (count + params.token_padding_block - 1) // params.token_padding_block
            ) * params.token_padding_block
            sf = (
                (count + params.sf_padding_block - 1) // params.sf_padding_block
            ) * params.sf_padding_block
            blocks = (
                count + params.cluster_tile_tokens - 1
            ) // params.cluster_tile_tokens
            data_scan, sf_scan, block_scan = data, sf, blocks
            for step in cutlass.range_constexpr(cute.arch.WARP_SIZE.bit_length() - 1):
                delta = 1 << step
                d = cute.arch.shuffle_sync_up(data_scan, offset=delta, mask_and_clamp=0)
                f = cute.arch.shuffle_sync_up(sf_scan, offset=delta, mask_and_clamp=0)
                b = cute.arch.shuffle_sync_up(
                    block_scan, offset=delta, mask_and_clamp=0
                )
                if lane >= delta:
                    data_scan += d
                    sf_scan += f
                    block_scan += b
            rank = expert
            if const_expr(params.claim_order == "longest_first"):
                rank = Int32(0)
                for other_base in cutlass.range_constexpr(
                    (params.expert_count + (cute.arch.WARP_SIZE - 1))
                    // cute.arch.WARP_SIZE
                ):
                    other_expert = Int32(other_base * cute.arch.WARP_SIZE) + lane
                    other_count = Int32(0)
                    if other_expert < params.expert_count:
                        other_count = params.expert_token_sizes[other_expert]
                    for source in cutlass.range(cute.arch.WARP_SIZE, unroll=1):
                        peer = Int32(other_base * cute.arch.WARP_SIZE) + source
                        peer_count = cute.arch.shuffle_sync(
                            other_count,
                            offset=source,
                            mask=0xFFFFFFFF,
                            mask_and_clamp=cute.arch.WARP_SIZE - 1,
                        )
                        if peer < params.expert_count:
                            if peer_count > count or (
                                peer_count == count and peer < expert
                            ):
                                rank += Int32(1)
            if expert < params.expert_count:
                self._order[rank] = expert
                data_row = data_base + data_scan - data
                sf_row = sf_base + sf_scan - sf
                self._pool_data_row[expert] = data_row
                self._pool_sf_row[expert] = sf_row
                self._pool_token_block_row[expert] = block_base + block_scan - blocks
            data_base += cute.arch.shuffle_sync(
                data_scan,
                offset=cute.arch.WARP_SIZE - 1,
                mask_and_clamp=cute.arch.WARP_SIZE - 1,
            )
            sf_base += cute.arch.shuffle_sync(
                sf_scan,
                offset=cute.arch.WARP_SIZE - 1,
                mask_and_clamp=cute.arch.WARP_SIZE - 1,
            )
            block_base += cute.arch.shuffle_sync(
                block_scan,
                offset=cute.arch.WARP_SIZE - 1,
                mask_and_clamp=cute.arch.WARP_SIZE - 1,
            )
        cute.arch.sync_warp()

    @dsl_user_op
    @cute.jit
    def internal_init(self, warp_idx, sched_warp_id: int, *, loc=None, ip=None):
        """Prime the scheduler after dispatch has reset counters and counts."""
        if warp_idx == sched_warp_id:
            self._init_expert_tables()
            # Carry the helper's rebound count view across the warp branch.
            self.params = self.params
        else:
            self.params = self.params
        if const_expr(self.params.load_balance_mode == "atomic_counter"):
            if warp_idx == sched_warp_id:
                tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
                claimed = Int32(0)
                if self._atomic_state.is_leader and tidx % cute.arch.WARP_SIZE == Int32(
                    0
                ):
                    claimed = cute.arch.atomic_add(
                        self._atomic_state.counter_ptr, Int32(1), loc=loc, ip=ip
                    )
                self._atomic_state.cached = cute.arch.shuffle_sync(
                    claimed,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=cute.arch.WARP_SIZE - 1,
                )
                self._atomic_state = self._atomic_state
            else:
                self._atomic_state = self._atomic_state
        elif const_expr(self.params.load_balance_mode == "alternating"):
            if warp_idx == sched_warp_id:
                blocks = Int32(0)
                for expert in cutlass.range(0, self.expert_count, 1, unroll=1):
                    count = self.params.expert_token_sizes[expert]
                    blocks += (
                        count + self.params.cluster_tile_tokens - 1
                    ) // self.params.cluster_tile_tokens
                n1 = self._fc1_feature_blocks
                n2 = self._fc2_feature_blocks
                clusters = self.num_persistent_clusters
                total_waves = (blocks * n1 + clusters - 1) // clusters
                # Cover every block touched by the first FC2 wave.
                first_blocks = cutlass.min(blocks, (clusters + n2 - 1) // n2)
                cover = (first_blocks * n1 + clusters - 1) // clusters
                # Keep FC1 ahead even when FC2 consumes token blocks faster.
                ahead = (
                    n1 + (blocks - 1) * cutlass.max(n1 - n2, Int32(0)) + clusters - 1
                ) // clusters + 1
                self._alternating.pending = cutlass.min(
                    total_waves, cutlass.max(cover, ahead)
                )
                self._alternating.total_fc1 = blocks * n1
                self._alternating.total_fc2 = blocks * n2
                self._alternating = self._alternating
            else:
                self._alternating = self._alternating
        elif const_expr(self.params.load_balance_mode == "static"):
            if warp_idx == sched_warp_id:
                claimed = self._claim_static()
                self._decode_claim(claimed)
                self._state = self._state
                self.current_work = self.current_work
            else:
                self._state = self._state
                self.current_work = self.current_work
        self._first_claim_pending = True

    @dsl_user_op
    @cute.jit
    def _claim_static(self, *, loc=None, ip=None) -> Int32:
        result = self._state.static_tile_idx
        self._state.static_tile_idx = result + self.num_persistent_clusters
        return result

    @dsl_user_op
    @cute.jit
    def _claim_atomic(self, *, loc=None, ip=None) -> Int32:
        atomic = self._atomic_state
        broadcast = cute.make_tensor(atomic.broadcast_ptr, cute.make_layout((1,)))
        cluster_size = (
            self.params.cluster_shape_token_feature[0]
            * self.params.cluster_shape_token_feature[1]
        )
        if atomic.is_leader:
            self._cluster_pipeline.producer_acquire(atomic.producer)
            barrier = self._cluster_pipeline.sync_object_full.get_barrier(
                atomic.producer.index, loc=loc, ip=ip
            )
            tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
            lane = tidx % Int32(cute.arch.WARP_SIZE)
            if const_expr(self.params.load_balance_mode == "alternating"):
                claimed = Int32(ALTERNATING_DONE)
                if lane == Int32(0):
                    claimed = self._claim_alternating()
                    self._alternating = self._alternating
                else:
                    self._alternating = self._alternating
                claimed = cute.arch.shuffle_sync(
                    claimed,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=cute.arch.WARP_SIZE - 1,
                )
            elif const_expr(self._first_claim_pending):
                claimed = atomic.cached
            else:
                claimed = Int32(0)
                if lane == Int32(0):
                    claimed = cute.arch.atomic_add(
                        atomic.counter_ptr, Int32(1), loc=loc, ip=ip
                    )
                claimed = cute.arch.shuffle_sync(
                    claimed,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=cute.arch.WARP_SIZE - 1,
                )
            if lane < Int32(cluster_size):
                _store_peer_i32(
                    atomic.broadcast_ptr, claimed, barrier, lane, loc=loc, ip=ip
                )
                _expect_peer_i32(barrier, lane, loc=loc, ip=ip)
        atomic.producer.advance()
        self._cluster_pipeline.consumer_wait(atomic.consumer)
        claimed = broadcast[0]
        cute.arch.fence_acq_rel_cta()
        self._cluster_pipeline.sync_object_empty.arrive(atomic.consumer.index, Int32(0))
        atomic.consumer.advance()
        # The alternating claim loop rebinds self across MLIR regions. Keep
        # the advanced broadcast pipeline state, rather than its old alias.
        self._atomic_state = atomic
        return claimed

    @dsl_user_op
    @cute.jit
    def _claim_alternating(self, *, loc=None, ip=None) -> Int32:
        # Only the cluster leader lane enters this routine. Other lanes park
        # at the shuffle. FC2 waits for *claims*, never for completion here.
        a = self._alternating
        claimed = Int32(ALTERNATING_DONE)
        searching = Int32(1)
        while searching != Int32(0):
            if a.fc1_exhausted != Int32(0) and a.fc2_exhausted != Int32(0):
                searching = Int32(0)
            elif a.pending > Int32(0) or a.fc2_exhausted != Int32(0):
                index = cute.arch.atomic_add(
                    self.params.load_balance_counter_ptr, Int32(1)
                )
                if index < a.total_fc1:
                    a.pending = cutlass.max(a.pending - Int32(1), Int32(0))
                    claimed = index
                    searching = Int32(0)
                else:
                    a.fc1_exhausted = Int32(1)
                    a.pending = Int32(0)
            else:
                index = cute.arch.atomic_add(
                    self.params.load_balance_counter_ptr + Int32(1), Int32(1)
                )
                a.pending = Int32(1) - a.fc1_exhausted
                if index < a.total_fc2:
                    threshold = (
                        index // self._fc2_feature_blocks + Int32(1)
                    ) * self._fc1_feature_blocks
                    observed = Int32(0)
                    while observed < threshold:
                        observed = cute.arch.load(
                            self.params.load_balance_counter_ptr,
                            Int32,
                            sem="acquire",
                            scope="gpu",
                        )
                    claimed = -index - Int32(1)
                    searching = Int32(0)
                else:
                    a.fc2_exhausted = Int32(1)
            a = a
        self._alternating = a
        return claimed

    @dsl_user_op
    @cute.jit
    def _decode_alternating(self, claimed, *, loc=None, ip=None):
        record = DeviceWorkRecord(Int32(DONE_EXPERT), *(Int32(0) for _ in range(7)))
        if claimed != Int32(ALTERNATING_DONE):
            # Each phase has its own monotone cursor over the full expert list.
            if claimed < Int32(0):
                saved_fc1 = self._state
                self._state = self._fc2_state
                index = -claimed - Int32(1)
                end = self._state.expert_tile_end
                while index >= end:
                    self._advance_expert()
                    self._state = self._state
                    end = self._state.expert_tile_end
                record = self._decode_inside_expert(index)
                self._fc2_state = self._state
                self._state = saved_fc1
            else:
                end = self._state.expert_tile_end
                while claimed >= end:
                    self._advance_expert()
                    self._state = self._state
                    end = self._state.expert_tile_end
                record = self._decode_inside_expert(claimed)
                self._fc2_state = self._fc2_state
            self._state = self._state
            self._fc2_state = self._fc2_state
        else:
            self._state = self._state
            self._fc2_state = self._fc2_state
        self.current_work = self._peek_readiness(record)

    @dsl_user_op
    @cute.jit
    def _advance_expert(self, *, loc=None, ip=None) -> None:
        state = self._state
        params = self.params
        state.expert_pos += Int32(1)
        state.expert_idx = self._order[state.expert_pos]
        state.data_row = self._pool_data_row[state.expert_idx]
        state.sf_row = self._pool_sf_row[state.expert_idx]
        state.token_block_row = self._pool_token_block_row[state.expert_idx]
        state.token_count = params.expert_token_sizes[state.expert_idx]
        state.token_blocks = (
            state.token_count + Int32(params.cluster_tile_tokens - 1)
        ) // Int32(params.cluster_tile_tokens)
        state.expert_tile_start = state.expert_tile_end
        tiles = Int32(0)
        if state.phase == Int32(DevicePhase.FC1):
            tiles = state.token_blocks * self._fc1_feature_blocks
        else:
            tiles = state.token_blocks * self._fc2_feature_blocks
        state.expert_tile_end = state.expert_tile_start + tiles

    @dsl_user_op
    @cute.jit
    def _switch_to_fc2(self, *, loc=None, ip=None) -> None:
        state = self._state
        state.phase = Int32(DevicePhase.FC2)
        state.expert_pos = state.group_first_expert - Int32(1)
        state.expert_tile_end = state.group_fc1_end
        state.token_count = Int32(0)
        state.token_blocks = Int32(0)
        self._advance_expert(loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def _advance_group(self, *, loc=None, ip=None) -> None:
        state = self._state
        params = self.params

        base_fc1 = state.cumulative_fc1_tiles
        base_fc2 = state.cumulative_fc2_tiles
        state.group_first_expert = state.group_last_expert
        threshold = base_fc1 + Int32(params.group_hint)
        cumulative_fc1 = base_fc1
        cumulative_fc2 = base_fc2
        expert_cursor = state.group_first_expert
        while expert_cursor < self.expert_count and cumulative_fc1 < threshold:
            token_count = params.expert_token_sizes[self._order[expert_cursor]]
            token_blocks = (
                token_count + Int32(params.cluster_tile_tokens - 1)
            ) // Int32(params.cluster_tile_tokens)
            cumulative_fc1 = cumulative_fc1 + token_blocks * self._fc1_feature_blocks
            cumulative_fc2 = cumulative_fc2 + token_blocks * self._fc2_feature_blocks
            expert_cursor = expert_cursor + Int32(1)

        state.group_last_expert = expert_cursor
        state.cumulative_fc1_tiles = cumulative_fc1
        state.cumulative_fc2_tiles = cumulative_fc2
        group_start_tile = state.group_end
        state.group_fc1_end = group_start_tile + cumulative_fc1 - base_fc1
        state.group_end = state.group_fc1_end + cumulative_fc2 - base_fc2
        state.phase = Int32(DevicePhase.FC1)
        state.expert_pos = state.group_first_expert - Int32(1)
        state.expert_tile_end = group_start_tile
        state.token_count = Int32(0)
        state.token_blocks = Int32(0)

        # A zero-token suffix yields an empty group. Do not dereference its
        # first expert after the greedy loop has reached expert_count.
        if state.group_first_expert < self.expert_count:
            self._advance_expert(loc=loc, ip=ip)
        else:
            self._state = self._state

    @dsl_user_op
    @cute.jit
    def _decode_inside_expert(self, claimed, *, loc=None, ip=None):
        state = self._state
        local_id = claimed - state.expert_tile_start
        feature_blocks = self._fc2_feature_blocks
        if state.phase == Int32(DevicePhase.FC1):
            feature_blocks = self._fc1_feature_blocks
        cluster_token_idx = local_id // feature_blocks
        cluster_feature_idx = local_id - cluster_token_idx * feature_blocks
        cta_token_idx = (
            cluster_token_idx * self.params.cluster_shape_token_feature[0]
            + self.cta_coord[0]
        )
        cta_feature_idx = (
            cluster_feature_idx * self.params.cluster_shape_token_feature[1]
            + self.cta_coord[1]
        )
        token_start = cta_token_idx * Int32(self.params.cta_tile_tokens)
        valid = cutlass.max(state.token_count - token_start, Int32(0))
        valid = cutlass.min(valid, Int32(self.params.cta_tile_tokens))
        data_row = state.data_row
        sf_row = state.sf_row
        return DeviceWorkRecord(
            state.expert_idx,
            cta_feature_idx,
            cta_token_idx,
            data_row,
            sf_row,
            state.token_block_row,
            valid,
            state.phase,
        )

    @dsl_user_op
    @cute.jit
    def _peek_readiness(self, record, *, loc=None, ip=None):
        """Peek once at dispatch readiness for FC1 or FC1 completion for FC2."""
        phase_and_peek = record.phase_and_peek
        if record.is_valid_tile:
            if record.phase == Int32(DevicePhase.FC1):
                ptr = self.params.fc1_ready_counter_ptr + record.counter_slot
                observed = cute.arch.load(ptr, ptr.dtype, sem="acquire", scope="gpu")
                if observed >= record.valid_tokens_in_cta_tile:
                    phase_and_peek = phase_and_peek | Int32(PEEK_READY_BIT)
            else:
                ptr = self.params.fc1_done_counter_ptr + record.counter_slot
                observed = cute.arch.load(ptr, ptr.dtype, sem="acquire", scope="gpu")
                if observed >= self.params.fc2_ready_threshold:
                    phase_and_peek = phase_and_peek | Int32(PEEK_READY_BIT)
        return DeviceWorkRecord(
            record.expert_idx,
            record.tile_m_idx,
            record.tile_n_idx,
            record.cumulative_data_physical_row,
            record.cumulative_sf_physical_row,
            record.cumulative_token_block_count,
            record.valid_tokens_in_cta_tile,
            phase_and_peek,
        )

    @dsl_user_op
    @cute.jit
    def _decode_claim(self, claimed, *, loc=None, ip=None) -> None:
        state = self._state
        record = DeviceWorkRecord(
            Int32(DONE_EXPERT),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(DevicePhase.NONE),
        )

        group_end = state.group_end
        last_expert = state.group_last_expert
        while claimed >= group_end and last_expert < self.expert_count:
            self._advance_group(loc=loc, ip=ip)
            self._state = self._state
            state = self._state
            group_end = state.group_end
            last_expert = state.group_last_expert
        state = self._state

        if claimed < state.group_end:
            if state.phase == Int32(DevicePhase.FC1) and claimed >= state.group_fc1_end:
                self._switch_to_fc2(loc=loc, ip=ip)
                self._state = self._state
            else:
                self._state = self._state
            state = self._state
            expert_end = state.expert_tile_end
            while claimed >= expert_end:
                self._advance_expert(loc=loc, ip=ip)
                self._state = self._state
                state = self._state
                expert_end = state.expert_tile_end
            record = self._decode_inside_expert(claimed, loc=loc, ip=ip)
        else:
            self._state = self._state
        self.current_work = self._peek_readiness(record, loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def gen_next_work(self, *, loc=None, ip=None) -> None:
        """Claim and decode one cluster tile for the persistent loop."""
        if const_expr(
            self._first_claim_pending and self.params.load_balance_mode == "static"
        ):
            pass
        else:
            if const_expr(self.params.load_balance_mode != "static"):
                claimed = self._claim_atomic(loc=loc, ip=ip)
            else:
                claimed = self._claim_static(loc=loc, ip=ip)
            if const_expr(self.params.load_balance_mode == "alternating"):
                self._decode_alternating(claimed, loc=loc, ip=ip)
            else:
                self._decode_claim(claimed, loc=loc, ip=ip)
        if const_expr(self._first_claim_pending):
            self._first_claim_pending = False

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for value in (
            self.params,
            self.num_persistent_clusters,
            self.cta_coord,
            self.current_work,
            self._state,
            self._fc1_feature_blocks,
            self._fc2_feature_blocks,
        ):
            values.extend(extract_mlir_values(value))
        if self.params.load_balance_mode != "static":
            values.extend(extract_mlir_values(self._atomic_state))
        if self.params.load_balance_mode == "alternating":
            values.extend(extract_mlir_values(self._fc2_state))
            values.extend(extract_mlir_values(self._alternating))
        values.extend(extract_mlir_values(self._producer_state))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]):
        idx = 0

        def take(prototype):
            nonlocal idx
            count = len(extract_mlir_values(prototype))
            result = new_from_mlir_values(prototype, values[idx : idx + count])
            idx += count
            return result

        result = PersistentDeviceScheduler.__new__(PersistentDeviceScheduler)
        result.params = take(self.params)
        result.num_persistent_clusters = take(self.num_persistent_clusters)
        result.cta_coord = take(self.cta_coord)
        result.current_work = take(self.current_work)
        result._state = take(self._state)
        result._fc1_feature_blocks = take(self._fc1_feature_blocks)
        result._fc2_feature_blocks = take(self._fc2_feature_blocks)
        result._atomic_state = (
            take(self._atomic_state)
            if self.params.load_balance_mode != "static"
            else None
        )
        if self.params.load_balance_mode == "alternating":
            result._fc2_state = take(self._fc2_state)
            result._alternating = take(self._alternating)
        result._producer_state = take(self._producer_state)
        assert idx == len(values)
        result._pipeline = self._pipeline
        result._smem = self._smem
        result._storage = self._storage
        result._order = self._order
        result._pool_data_row = self._pool_data_row
        result._pool_sf_row = self._pool_sf_row
        result._pool_token_block_row = self._pool_token_block_row
        result._cluster_pipeline = self._cluster_pipeline
        result._first_claim_pending = self._first_claim_pending
        return result


__all__ = [
    "ALTERNATING_DONE",
    "DONE_EXPERT",
    "DevicePhase",
    "DeviceSchedulerConsumer",
    "DeviceSchedulerParams",
    "DeviceWorkRecord",
    "PEEK_READY_BIT",
    "PersistentDeviceScheduler",
    "SCALE_VECTOR_SIZE",
    "WORK_RECORD_WORDS",
]
