# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared-memory metadata resources for raw-BSR K/V and Softmax work.

Each KV instruction owns two independent resources. During HEAD, the load task
resolves a route, stores it for the matching K/V pair, issues K, then stages a
copy for Softmax. During LOOP, V consumes the previous K/V metadata before the
load task overwrites it with the next route and issues K. Softmax waits for its
staged copy, moves all seven scalars to registers, and releases the stage before
masking; the producer may therefore reuse SMEM while score processing continues.
"""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...._block_sparse.common import _validate_sparse_block_size
from ...placeholder_helpers import _placeholder_smem_array
from ...stage import FmhaStage
from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_block_sparse import resolve_block_sparse_route_origins
from .helpers_common import (
    _TASK_CACHE_KV_PAGE_IDX_UB,
    _TASK_CACHE_KV_REQUEST_BEGIN,
    _TASK_CACHE_SEQ_LEN_KV,
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    _decode_gen_task_cache,
    _keeps_col_base,
    _logical_head_batch,
)


_ALL_TOKEN_WORDS_VALID_FLAG = 1 << 2


@dataclass(frozen=True)
class _BlockSparseKvMetadataLayout:
    """One K/V route padded to four words for 16-byte alignment."""

    origin0_word_offset: int = 0
    origin1_word_offset: int = 1
    route_flags_word_offset: int = 2
    total_words: int = 4

    @property
    def size_bytes(self) -> int:
        """Return the fixed 16-byte K/V metadata allocation size."""

        return self.total_words * 4


@dataclass(frozen=True)
class _BlockSparseSoftmaxMetadataLayout:
    """Layout of the staged cross-warp Softmax metadata payload."""

    origin0_word_offset: int
    origin1_word_offset: int
    route_flags_word_offset: int
    token_words_word_offset: int | None
    stage_stride_words: int
    num_stages: int
    total_words: int

    @property
    def size_bytes(self) -> int:
        """Return the 16-byte-aligned staged allocation size."""

        return self.total_words * 4

    @staticmethod
    def create(
        *, has_token_bits: bool, num_stages: int
    ) -> "_BlockSparseSoftmaxMetadataLayout":
        """Build a stage-count-dependent layout for Softmax metadata."""

        if not isinstance(has_token_bits, bool):
            raise TypeError("has_token_bits must be a bool")
        if (
            isinstance(num_stages, bool)
            or not isinstance(num_stages, int)
            or num_stages <= 0
        ):
            raise ValueError("num_stages must be a positive integer")
        stage_stride_words = 8 if has_token_bits else 4
        return _BlockSparseSoftmaxMetadataLayout(
            origin0_word_offset=0,
            origin1_word_offset=1,
            route_flags_word_offset=2,
            token_words_word_offset=4 if has_token_bits else None,
            stage_stride_words=stage_stride_words,
            num_stages=num_stages,
            total_words=num_stages * stage_stride_words,
        )


@cute.jit
def _block_sparse_k_local_route_idx(
    cfg: Constexpr[FmhaDecodeConfig],
    stage_info: StageInfo,
    inst_id: Constexpr[int],
    section: Constexpr[FmhaStage],
) -> Int32:
    """Map a K publication to its instruction-local KV128 route ordinal."""

    if cutlass.const_expr(section == FmhaStage.Head):
        return Int32(inst_id)
    return (stage_info.loop_offset + Int32(1)) * Int32(cfg.num_insts_kv) + Int32(
        inst_id
    )


@cute.jit
def _warp_broadcast_lane0_i32(value: Int32) -> Int32:
    """Broadcast one lane-zero scalar as a warp-uniform Int32 value."""

    return cute.arch.make_warp_uniform(
        Int32(
            prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=value,
                offset=0,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.IDX,
            )
        )
    )


@cute.jit
def _resolve_block_sparse_route_metadata(
    block_sparse_indices: cute.Pointer,
    row_begin: Int32,
    row_end: Int32,
    route_idx: Int32,
    kv_block_size: Constexpr[int],
    seq_len_kv: Int32,
    lane_idx: Int32,
) -> tuple[Int32, Int32, Int32]:
    """Resolve one route into two warp-uniform KV64 origins and validity bits."""

    origin0 = Int32(0)
    origin1 = Int32(0)
    valid_mask = Int32(0)
    if lane_idx == Int32(0):
        origin0, valid0, origin1, valid1 = resolve_block_sparse_route_origins(
            block_sparse_indices,
            row_begin,
            row_end,
            route_idx,
            kv_block_size,
            seq_len_kv,
        )
        valid_mask = Int32(valid0) | (Int32(valid1) << Int32(1))

    origin0 = _warp_broadcast_lane0_i32(origin0)
    origin1 = _warp_broadcast_lane0_i32(origin1)
    valid_mask = _warp_broadcast_lane0_i32(valid_mask)
    return origin0, origin1, valid_mask


@cute.jit
def _load_block_sparse_token_word_from_route(
    kv_valid_bits: cute.Pointer,
    origin0: Int32,
    origin1: Int32,
    valid_mask: Int32,
    seq_len_kv: Int32,
    batch_idx: Int32,
    num_kv_valid_words: Constexpr[int],
    lane_idx: Int32,
    *,
    detect_all_token_words_valid: Constexpr[bool],
) -> tuple[Uint32, cutlass.Boolean]:
    """Load one lane-distributed token word from an already resolved route.

    Lanes 0/1 load origin0 words 0/1, and lanes 2/3 load origin1 words 0/1.
    Other lanes contribute zero and are neutral in the optional warp-wide
    all-valid vote. Keeping this separate from BSR resolution lets the load warp
    issue both K TensorMaps before it waits for a free Softmax metadata stage;
    staging then reuses the resolved route scalars instead of parsing the row
    again.
    """

    token_word = Uint32(0)
    if lane_idx < Int32(4):
        fragment_idx = lane_idx >> Int32(1)
        word_in_fragment = lane_idx & Int32(1)
        fragment_origin = origin0
        fragment_valid = valid_mask & Int32(1)
        if fragment_idx == Int32(1):
            fragment_origin = origin1
            fragment_valid = (valid_mask >> Int32(1)) & Int32(1)
        word_token_begin = fragment_origin + word_in_fragment * Int32(32)
        word_idx = word_token_begin >> Int32(5)
        if (
            fragment_valid != Int32(0)
            and word_token_begin < seq_len_kv
            and word_idx < Int32(num_kv_valid_words)
        ):
            bitset_offset = batch_idx * Int32(num_kv_valid_words) + word_idx
            token_word = Uint32(kv_valid_bits[bitset_offset])

    all_token_words_valid = cutlass.Boolean(False)
    if cutlass.const_expr(detect_all_token_words_valid):
        all_token_words_valid = cute.arch.vote_all_sync(
            lane_idx >= Int32(4) or token_word == Uint32(0xFFFFFFFF)
        )
    return token_word, all_token_words_valid


@dataclass(kw_only=True)
class SmemBlockSparseKvMetadataResource(DecodeGenResourceBase):
    """Pipeline-free route metadata retained from one K issue through V."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("resolved_origin0_slot", Int32, Int32(0), "Resolved first KV64 origin."),
        ("resolved_origin1_slot", Int32, Int32(0), "Resolved second KV64 origin."),
        (
            "resolved_route_flags_slot",
            Int32,
            Int32(0),
            "Resolved route flags for the two KV64 fragments.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    inst_id: Constexpr[int] = 0
    block_sparse_indices: cute.Pointer | None = None
    layout: Constexpr[_BlockSparseKvMetadataLayout | None] = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _smem_words: cutlass.Array = None
    resolved_origin0_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    resolved_origin1_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    resolved_route_flags_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self) -> None:
        """Validate the raw profile and create its fixed four-word layout."""

        if self.cfg is None:
            raise ValueError("cfg is required")
        if not self.cfg.use_block_sparse:
            raise ValueError(
                "SmemBlockSparseKvMetadataResource requires block-sparse mode"
            )
        if self.pipeline_config is not None:
            raise ValueError("K/V metadata resource must not own a pipeline")
        _validate_sparse_block_size(self.cfg.q_block_size, "q_block_size")
        _validate_sparse_block_size(self.cfg.kv_block_size, "kv_block_size")
        if self.cfg.num_insts_kv != 2:
            raise ValueError("raw sparse K/V metadata requires two KV instructions")
        if self.inst_id not in (0, 1):
            raise ValueError("inst_id must be 0 or 1")
        self.layout = _BlockSparseKvMetadataLayout()
        super().__post_init__()

    def _init_placeholder_state(self) -> None:
        """Create shape-correct K/V metadata SMEM for task-graph tracing."""

        self._smem_words = _placeholder_smem_array(Int32, self.layout.total_words)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate exactly one aligned four-word K/V metadata slot."""

        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=self.name,
                size_bytes=self.layout.size_bytes,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """K/V route metadata uses SMEM only."""

        return []

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the fixed K/V metadata allocation on the load warp."""

        if cutlass.const_expr(context is not None and context.smem_base is not None):
            self._smem_words = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=Int32,
                shape=(self.layout.total_words,),
                addrspace=3,
            )
        return {}

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Bind producer-local K/V metadata before the first resolution."""

        self._create_initial_task_locals(stage_info.context)

    @consumer_work(
        returns=(
            resolved_origin0_slot,
            resolved_origin1_slot,
            resolved_route_flags_slot,
        )
    )
    @cute.jit
    def resolve_route(
        self, stage_info: StageInfo, *, section: Constexpr[FmhaStage]
    ) -> tuple[Int32, Int32, Int32]:
        """Resolve this resource instance's real or dummy KV128 route."""

        assert self.block_sparse_indices is not None
        task_cache = _decode_gen_task_cache(stage_info)
        row_begin = Int32(task_cache[_TASK_CACHE_KV_REQUEST_BEGIN])
        row_nnz = Int32(task_cache[_TASK_CACHE_KV_PAGE_IDX_UB])
        seq_len_kv = Int32(task_cache[_TASK_CACHE_SEQ_LEN_KV])
        route_idx = _block_sparse_k_local_route_idx(
            self.cfg, stage_info, self.inst_id, section
        )
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        return _resolve_block_sparse_route_metadata(
            self.block_sparse_indices,
            row_begin,
            row_begin + row_nnz,
            route_idx,
            self.cfg.kv_block_size,
            seq_len_kv,
            lane_idx,
        )

    @producer_work
    @cute.jit
    def store_route(
        self,
        stage_info: StageInfo,
        *,
        resolved_origin0: Int32,
        resolved_origin1: Int32,
        resolved_route_flags: Int32,
    ) -> None:
        """Store resolved scalars in the slot retained from K through V."""

        del stage_info
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        if lane_idx == Int32(0):
            self._smem_words[Int32(self.layout.origin0_word_offset)] = resolved_origin0
            self._smem_words[Int32(self.layout.origin1_word_offset)] = resolved_origin1
            self._smem_words[Int32(self.layout.route_flags_word_offset)] = (
                resolved_route_flags
            )
        # K consumes this slot immediately, while V consumes it at the start
        # of the next cadence.  Both execute in this warp, so a warp fence is
        # sufficient; no cross-warp mbarrier belongs here.
        cute.arch.sync_warp()

    @cute.jit
    def route_origin(self, fragment_idx: Int32) -> Int32:
        """Load one KV64 origin from the route retained for K and V."""

        word_offset = Int32(self.layout.origin0_word_offset)
        if fragment_idx == Int32(1):
            word_offset = Int32(self.layout.origin1_word_offset)
        return Int32(self._smem_words[word_offset])

    @cute.jit
    def route_flags(self) -> Int32:
        """Load the retained route's two KV64 fragment-validity flags."""

        return Int32(self._smem_words[Int32(self.layout.route_flags_word_offset)])


@dataclass(kw_only=True)
class SmemBlockSparseSoftmaxMetadataResource(DecodeGenResourceBase):
    """Staged route and token metadata consumed by one Softmax group.

    ``inst_id`` identifies which of the two Softmax pipelines owns this
    resource. Route resolution belongs to the paired K/V resource, so the
    producer passes the resolved payload explicitly instead of recomputing it.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("softmax_origin0_slot", Int32, Int32(0), "Loaded first KV64 origin."),
        ("softmax_origin1_slot", Int32, Int32(0), "Loaded second KV64 origin."),
        (
            "softmax_route_flags_slot",
            Int32,
            Int32(0),
            "Loaded fragment and optional all-token-words-valid flags.",
        ),
        (
            "softmax_token_word0_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Loaded first token-validity word.",
        ),
        (
            "softmax_token_word1_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Loaded second token-validity word.",
        ),
        (
            "softmax_token_word2_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Loaded third token-validity word.",
        ),
        (
            "softmax_token_word3_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Loaded fourth token-validity word.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    inst_id: Constexpr[int] = 0
    kv_valid_bits: cute.Pointer | None = None
    h_k_idx: Int32 = None
    b_idx: Int32 = None
    layout: Constexpr[_BlockSparseSoftmaxMetadataLayout | None] = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _smem_words: cutlass.Array = None
    softmax_origin0_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_origin1_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_route_flags_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_token_word0_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_token_word1_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_token_word2_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_token_word3_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self) -> None:
        """Validate the profile and derive its staged metadata layout."""

        if self.cfg is None:
            raise ValueError("cfg is required")
        if not self.cfg.use_block_sparse:
            raise ValueError(
                "SmemBlockSparseSoftmaxMetadataResource requires block-sparse mode"
            )
        if self.pipeline_config is None:
            raise ValueError("Softmax metadata resource requires a pipeline")
        _validate_sparse_block_size(self.cfg.q_block_size, "q_block_size")
        _validate_sparse_block_size(self.cfg.kv_block_size, "kv_block_size")
        if self.cfg.num_insts_kv != 2:
            raise ValueError("raw sparse Softmax metadata requires two KV instructions")
        if self.inst_id not in (0, 1):
            raise ValueError("inst_id must be 0 or 1")
        self.layout = _BlockSparseSoftmaxMetadataLayout.create(
            has_token_bits=self.cfg.use_kv_valid_bits,
            num_stages=self.pipeline_config.num_stages,
        )
        super().__post_init__()

    def _init_placeholder_state(self) -> None:
        """Create shape-correct staged SMEM for task-graph tracing."""

        self._smem_words = _placeholder_smem_array(Int32, self.layout.total_words)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate one metadata payload per configured pipeline stage."""

        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=self.name,
                size_bytes=self.layout.size_bytes,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Softmax route metadata uses SMEM and registers only."""

        return []

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the staged allocation on producer and consumer tasks."""

        if cutlass.const_expr(context is not None and context.smem_base is not None):
            self._smem_words = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=Int32,
                shape=(self.layout.total_words,),
                addrspace=3,
            )
        return {}

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Bind producer-side metadata storage before the first route."""

        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo) -> None:
        """Bind consumer-side metadata storage before the first wait."""

        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _producer_stage_base(self, stage_info: StageInfo) -> Int32:
        """Return the producer stage selected by the task scheduler."""

        return stage_info.stage_idx * Int32(self.layout.stage_stride_words)

    @cute.jit
    def _consumer_stage_base(self) -> Int32:
        """Return the consumer stage selected by the latest wait."""

        return self.consumer_work_stage * Int32(self.layout.stage_stride_words)

    @producer_work
    @cute.jit
    def store_route(
        self,
        stage_info: StageInfo,
        *,
        resolved_origin0: Int32,
        resolved_origin1: Int32,
        resolved_route_flags: Int32,
    ) -> None:
        """Store one resolved route and its optional token words in a stage."""

        origin0 = Int32(resolved_origin0)
        origin1 = Int32(resolved_origin1)
        route_flags = Int32(resolved_route_flags)
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        token_word = Uint32(0)
        all_token_words_valid = cutlass.Boolean(False)
        if cutlass.const_expr(self.cfg.use_kv_valid_bits):
            assert self.kv_valid_bits is not None
            task_cache = _decode_gen_task_cache(stage_info)
            seq_len_kv = Int32(task_cache[_TASK_CACHE_SEQ_LEN_KV])
            _, logical_b_idx = _logical_head_batch(stage_info, self.h_k_idx, self.b_idx)
            token_word, all_token_words_valid = (
                _load_block_sparse_token_word_from_route(
                    self.kv_valid_bits,
                    origin0,
                    origin1,
                    route_flags,
                    seq_len_kv,
                    logical_b_idx,
                    self.cfg.num_kv_valid_words,
                    lane_idx,
                    detect_all_token_words_valid=self.cfg.use_q128_token_route_full_guard,
                )
            )

        stage_base = self._producer_stage_base(stage_info)
        if lane_idx == Int32(0):
            self._smem_words[stage_base + Int32(self.layout.origin0_word_offset)] = (
                origin0
            )
            self._smem_words[stage_base + Int32(self.layout.origin1_word_offset)] = (
                origin1
            )
            if cutlass.const_expr(self.cfg.use_q128_token_route_full_guard):
                route_flags = route_flags | (
                    Int32(all_token_words_valid) * Int32(_ALL_TOKEN_WORDS_VALID_FLAG)
                )
            self._smem_words[
                stage_base + Int32(self.layout.route_flags_word_offset)
            ] = route_flags
        if cutlass.const_expr(self.cfg.use_kv_valid_bits):
            assert self.layout.token_words_word_offset is not None
            if cutlass.const_expr(self.cfg.use_q128_token_route_full_guard):
                if lane_idx < Int32(4) and not all_token_words_valid:
                    self._smem_words[
                        stage_base
                        + Int32(self.layout.token_words_word_offset)
                        + lane_idx
                    ] = Int32(token_word)
            else:
                if lane_idx < Int32(4):
                    self._smem_words[
                        stage_base
                        + Int32(self.layout.token_words_word_offset)
                        + lane_idx
                    ] = Int32(token_word)
        cute.arch.sync_warp()

    @consumer_work(
        returns=(
            softmax_origin0_slot,
            softmax_origin1_slot,
            softmax_route_flags_slot,
            softmax_token_word0_slot,
            softmax_token_word1_slot,
            softmax_token_word2_slot,
            softmax_token_word3_slot,
        )
    )
    @cute.jit
    def load_route(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32, Uint32, Uint32, Uint32, Uint32]:
        """Copy the waited stage to task-local registers before release."""

        _ = stage_info
        stage_base = self._consumer_stage_base()
        origin0 = Int32(
            self._smem_words[stage_base + Int32(self.layout.origin0_word_offset)]
        )
        origin1 = Int32(
            self._smem_words[stage_base + Int32(self.layout.origin1_word_offset)]
        )
        route_flags = Int32(
            self._smem_words[stage_base + Int32(self.layout.route_flags_word_offset)]
        )
        token_word0 = Uint32(0xFFFFFFFF)
        token_word1 = Uint32(0xFFFFFFFF)
        token_word2 = Uint32(0xFFFFFFFF)
        token_word3 = Uint32(0xFFFFFFFF)
        if cutlass.const_expr(self.cfg.use_kv_valid_bits):
            assert self.layout.token_words_word_offset is not None
            if cutlass.const_expr(self.cfg.tile_size_q == 64):
                lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
                local_word_base = _keeps_col_base(
                    self.cfg,
                    lane_idx,
                    self.cfg.num_s_regs_per_thread,
                ) >> Int32(5)
                token_word0 = Uint32(
                    self._smem_words[
                        stage_base
                        + Int32(self.layout.token_words_word_offset)
                        + local_word_base
                    ]
                )
                token_word1 = Uint32(
                    self._smem_words[
                        stage_base
                        + Int32(self.layout.token_words_word_offset)
                        + local_word_base
                        + Int32(1)
                    ]
                )
            elif cutlass.const_expr(self.cfg.use_q128_token_route_full_guard):
                all_token_words_valid = cutlass.Boolean(
                    (route_flags & Int32(_ALL_TOKEN_WORDS_VALID_FLAG)) != Int32(0)
                )
                if not all_token_words_valid:
                    token_word0 = Uint32(
                        self._smem_words[
                            stage_base + Int32(self.layout.token_words_word_offset)
                        ]
                    )
                    token_word1 = Uint32(
                        self._smem_words[
                            stage_base + Int32(self.layout.token_words_word_offset + 1)
                        ]
                    )
                    token_word2 = Uint32(
                        self._smem_words[
                            stage_base + Int32(self.layout.token_words_word_offset + 2)
                        ]
                    )
                    token_word3 = Uint32(
                        self._smem_words[
                            stage_base + Int32(self.layout.token_words_word_offset + 3)
                        ]
                    )
            else:
                token_word0 = Uint32(
                    self._smem_words[
                        stage_base + Int32(self.layout.token_words_word_offset)
                    ]
                )
                token_word1 = Uint32(
                    self._smem_words[
                        stage_base + Int32(self.layout.token_words_word_offset + 1)
                    ]
                )
                token_word2 = Uint32(
                    self._smem_words[
                        stage_base + Int32(self.layout.token_words_word_offset + 2)
                    ]
                )
                token_word3 = Uint32(
                    self._smem_words[
                        stage_base + Int32(self.layout.token_words_word_offset + 3)
                    ]
                )
        return (
            origin0,
            origin1,
            route_flags,
            token_word0,
            token_word1,
            token_word2,
            token_word3,
        )


__all__ = [
    "SmemBlockSparseKvMetadataResource",
    "SmemBlockSparseSoftmaxMetadataResource",
    "_BlockSparseKvMetadataLayout",
    "_BlockSparseSoftmaxMetadataLayout",
    "_resolve_block_sparse_route_metadata",
    "_load_block_sparse_token_word_from_route",
]
