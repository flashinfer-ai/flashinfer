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
staged copy, moves the seven-slot task payload to registers, and releases the
stage before masking. SWAP reuses those slots for four origins and one token
word; the producer may therefore reuse SMEM while score processing continues.
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

from ...._block_sparse.common import (
    _KV_ROUTE_SIZE,
    _block_sparse_kv_atom_size,
    _block_sparse_kv_routes_are_block_aligned,
    _validate_sparse_block_size,
)
from ...placeholder_helpers import _placeholder_smem_array
from ...stage import FmhaStage
from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_block_sparse import (
    resolve_block_sparse_aligned_route_origin,
    resolve_block_sparse_coarse_route_fragments,
    resolve_block_sparse_route_atom_origin,
)
from .helpers_common import (
    _TASK_CACHE_KV_PAGE_IDX_UB,
    _TASK_CACHE_KV_REQUEST_BEGIN,
    _TASK_CACHE_SEQ_LEN_KV,
    _TASK_CACHE_WARP_IDX,
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    _decode_gen_task_cache,
    _keeps_col_base,
    _logical_head_batch,
)


# Coarse route flags use bits 0/1 for structural KV64 validity. Bit 2 only
# states that all four current token words are full; structural/tail/causal
# validity remains independent.
_TOKEN_MASK_ROUTE_IS_FULL_FLAG = 1 << 2


@dataclass(frozen=True)
class _BlockSparseKvMetadataLayout:
    """Instruction-local physical origins retained from K through V.

    Fine profiles use a dense origin array. Coarse profiles retain the
    ``origin0, origin1, flags, padding`` ABI; for block-aligned B128/B256/...
    ``origin1`` is a mask view derived from one resolved KV128 base.
    """

    atom_size: int
    num_origin_words: int
    origin0_word_offset: int
    origin1_word_offset: int
    route_flags_word_offset: int | None
    total_words: int

    @property
    def size_bytes(self) -> int:
        """Return the 16-byte-aligned K/V metadata allocation size."""

        return self.total_words * 4

    @staticmethod
    def create(*, kv_block_size: int) -> "_BlockSparseKvMetadataLayout":
        """Build the compact origin array for one fixed KV128 route."""

        atom_size = _block_sparse_kv_atom_size(kv_block_size)
        num_origin_words = _KV_ROUTE_SIZE // atom_size
        if atom_size == 64:
            # Preserve the coarse layout and lane-zero validity flags exactly.
            return _BlockSparseKvMetadataLayout(
                atom_size=atom_size,
                num_origin_words=num_origin_words,
                origin0_word_offset=0,
                origin1_word_offset=1,
                route_flags_word_offset=2,
                total_words=4,
            )
        return _BlockSparseKvMetadataLayout(
            atom_size=atom_size,
            num_origin_words=num_origin_words,
            origin0_word_offset=0,
            origin1_word_offset=1,
            route_flags_word_offset=None,
            total_words=((num_origin_words + 3) // 4) * 4,
        )


@dataclass(frozen=True)
class _BlockSparseSoftmaxMetadataLayout:
    """Layout of the staged cross-warp Softmax metadata payload.

    Keeps retains ``origin0, origin1, flags, padding[, token_words[4]]``.
    Block-aligned coarse routes derive both KV64 mask views from one KV128
    route base. Q128 leaves the four word slots unused for a route-full stage;
    Q64 stages them unconditionally for better codegen. SWAP stores
    execution-ordered origins followed by optional logical K32 token words,
    one for each consumer warp.
    """

    softmax_atom_size: int
    num_origin_words: int
    origins_per_warp: int
    origin0_word_offset: int
    origin1_word_offset: int
    route_flags_word_offset: int | None
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
        *,
        use_keeps_mma_ab: bool,
        kv_block_size: int,
        has_token_bits: bool,
        num_stages: int,
    ) -> "_BlockSparseSoftmaxMetadataLayout":
        """Build a stage-count-dependent layout for Softmax metadata."""

        if not isinstance(use_keeps_mma_ab, bool):
            raise TypeError("use_keeps_mma_ab must be a bool")
        if not isinstance(has_token_bits, bool):
            raise TypeError("has_token_bits must be a bool")
        if (
            isinstance(num_stages, bool)
            or not isinstance(num_stages, int)
            or num_stages <= 0
        ):
            raise ValueError("num_stages must be a positive integer")
        atom_size = _block_sparse_kv_atom_size(kv_block_size)
        if use_keeps_mma_ab:
            softmax_atom_size = 64
            num_origin_words = 2
            origins_per_warp = 2
            route_flags_word_offset = 2
            token_words_word_offset = 4 if has_token_bits else None
            stage_stride_words = 8 if has_token_bits else 4
        else:
            softmax_atom_size = min(atom_size, 32)
            num_origin_words = _KV_ROUTE_SIZE // softmax_atom_size
            origins_per_warp = 32 // softmax_atom_size
            route_flags_word_offset = None
            token_words_word_offset = num_origin_words if has_token_bits else None
            stage_stride_words = num_origin_words + (4 if has_token_bits else 0)
        return _BlockSparseSoftmaxMetadataLayout(
            softmax_atom_size=softmax_atom_size,
            num_origin_words=num_origin_words,
            origins_per_warp=origins_per_warp,
            origin0_word_offset=0,
            origin1_word_offset=1,
            route_flags_word_offset=route_flags_word_offset,
            token_words_word_offset=token_words_word_offset,
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
    """Resolve one KV128 route and derive its two KV64 mask views."""

    origin0 = Int32(0)
    origin1 = Int32(0)
    valid_mask = Int32(0)
    if lane_idx == Int32(0):
        if cutlass.const_expr(_block_sparse_kv_routes_are_block_aligned(kv_block_size)):
            origin0, valid0 = resolve_block_sparse_aligned_route_origin(
                block_sparse_indices,
                row_begin,
                row_end,
                route_idx,
                kv_block_size,
                seq_len_kv,
            )
            valid1 = cutlass.Boolean(False)
            if valid0:
                remaining_tokens = seq_len_kv - origin0
                valid1 = cutlass.Boolean(remaining_tokens > Int32(_KV_ROUTE_SIZE // 2))
                if valid1:
                    origin1 = origin0 + Int32(_KV_ROUTE_SIZE // 2)
        else:
            origin0, valid0, origin1, valid1 = (
                resolve_block_sparse_coarse_route_fragments(
                    block_sparse_indices,
                    row_begin,
                    row_end,
                    route_idx,
                    kv_block_size,
                    seq_len_kv,
                )
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
) -> tuple[Uint32, cutlass.Boolean]:
    """Load one lane-distributed token word from an already resolved route.

    Lanes 0/1 load origin0 words 0/1, and lanes 2/3 load origin1 words 0/1.
    Other lanes are neutral in the warp-wide all-valid vote. Invalid or
    out-of-range fragments contribute zero, so the returned route-full flag is
    conservative. Keeping this separate from BSR resolution lets the load warp
    issue both K TensorMaps before it waits for a free Softmax metadata stage.
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

    route_token_mask_is_full = cute.arch.vote_all_sync(
        lane_idx >= Int32(4) or token_word == Uint32(0xFFFFFFFF)
    )
    return token_word, route_token_mask_is_full


@dataclass(kw_only=True)
class SmemBlockSparseKvMetadataResource(DecodeGenResourceBase):
    """Pipeline-free route metadata retained from one K issue through V."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("resolved_origin0_slot", Int32, Int32(0), "Primary routed origin."),
        ("resolved_origin1_slot", Int32, Int32(0), "Coarse second KV64 origin."),
        (
            "resolved_route_flags_slot",
            Int32,
            Int32(0),
            "Fine lane validity or coarse two-fragment flags.",
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
        """Validate the profile and create its execution-atom origin layout."""

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
        self.layout = _BlockSparseKvMetadataLayout.create(
            kv_block_size=self.cfg.kv_block_size
        )
        super().__post_init__()

    def _init_placeholder_state(self) -> None:
        """Create shape-correct K/V metadata SMEM for task-graph tracing."""

        self._smem_words = _placeholder_smem_array(Int32, self.layout.total_words)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate one aligned instruction-local K/V metadata slot."""

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
        """Bind the K/V metadata allocation on the load warp."""

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
        if cutlass.const_expr(self.layout.atom_size == 64):
            return _resolve_block_sparse_route_metadata(
                self.block_sparse_indices,
                row_begin,
                row_begin + row_nnz,
                route_idx,
                self.cfg.kv_block_size,
                seq_len_kv,
                lane_idx,
            )

        # Fine routes stay lane-distributed: each active lane carries only its
        # own atom origin through the existing three-scalar task ABI.
        origin = Int32(0)
        valid = cutlass.Boolean(False)
        if lane_idx < Int32(self.layout.num_origin_words):
            origin, valid = resolve_block_sparse_route_atom_origin(
                self.block_sparse_indices,
                row_begin,
                row_begin + row_nnz,
                route_idx,
                lane_idx,
                self.cfg.kv_block_size,
                seq_len_kv,
            )
        return origin, Int32(0), Int32(valid)

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
        if cutlass.const_expr(self.layout.atom_size == 64):
            assert self.layout.route_flags_word_offset is not None
            if lane_idx == Int32(0):
                self._smem_words[Int32(self.layout.origin0_word_offset)] = (
                    resolved_origin0
                )
                self._smem_words[Int32(self.layout.origin1_word_offset)] = (
                    resolved_origin1
                )
                self._smem_words[Int32(self.layout.route_flags_word_offset)] = (
                    resolved_route_flags
                )
        else:
            if lane_idx < Int32(self.layout.num_origin_words):
                origin = Int32(resolved_origin0)
                if resolved_route_flags == Int32(0):
                    origin = Int32(-1)
                self._smem_words[lane_idx] = origin
        # K consumes this slot immediately, while V consumes it at the start
        # of the next cadence.  Both execute in this warp, so a warp fence is
        # sufficient; no cross-warp mbarrier belongs here.
        cute.arch.sync_warp()

    @cute.jit
    def route_origin(self, atom_idx: Int32) -> Int32:
        """Load one execution-atom origin retained for K and V."""

        if cutlass.const_expr(self.layout.atom_size < 64):
            return Int32(self._smem_words[atom_idx])

        word_offset = Int32(self.layout.origin0_word_offset)
        if atom_idx == Int32(1):
            word_offset = Int32(self.layout.origin1_word_offset)
        return Int32(self._smem_words[word_offset])

    @cute.jit
    def route_flags(self) -> Int32:
        """Load the retained route's two KV64 fragment-validity flags."""

        assert self.layout.route_flags_word_offset is not None
        return Int32(self._smem_words[Int32(self.layout.route_flags_word_offset)])


@dataclass(kw_only=True)
class SmemBlockSparseSoftmaxMetadataResource(DecodeGenResourceBase):
    """Staged route and token metadata consumed by one Softmax group.

    ``inst_id`` identifies which of the two Softmax pipelines owns this
    resource. Route resolution belongs to the paired K/V resource, so the
    producer passes the resolved payload explicitly instead of recomputing it.
    For Keeps, a runtime route-full bit always skips the consumer predicate
    pass while leaving structural masking independent. Q128 also skips
    token-word SMEM staging/loads; Q64 retains branch-free word traffic because
    that schedule benchmarks faster.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("softmax_origin0_slot", Int32, Int32(0), "Loaded first physical origin."),
        ("softmax_origin1_slot", Int32, Int32(0), "Loaded second physical origin."),
        (
            "softmax_route_flags_slot",
            Int32,
            Int32(0),
            "Keeps route flags or SWAP's third physical origin.",
        ),
        (
            "softmax_token_word0_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Keeps token word 0 or SWAP's fourth origin as unsigned bits.",
        ),
        (
            "softmax_token_word1_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Keeps token word 1 or SWAP's packed logical K32 token word.",
        ),
        (
            "softmax_token_word2_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Loaded third Keeps token-validity word; unused by SWAP.",
        ),
        (
            "softmax_token_word3_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Loaded fourth Keeps token-validity word; unused by SWAP.",
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
            use_keeps_mma_ab=self.cfg.use_keeps_mma_ab,
            kv_block_size=self.cfg.kv_block_size,
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

    @cute.jit
    def _store_route_swaps(
        self,
        stage_info: StageInfo,
        resolved_origin0: Int32,
        resolved_origin1: Int32,
        resolved_route_flags: Int32,
    ) -> None:
        """Stage SWAP origins and optional logical-K32 token-mask words."""

        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        stage_base = self._producer_stage_base(stage_info)
        task_cache = _decode_gen_task_cache(stage_info)
        seq_len_kv = Int32(task_cache[_TASK_CACHE_SEQ_LEN_KV])

        # Keep each lane's resolved origin in a register for both the staged
        # origin store and its optional token-bit load. The final warp fence
        # publishes both payloads together before the producer commits.
        softmax_origin = Int32(-1)
        if cutlass.const_expr(self.cfg.kv_block_size < 64):
            if lane_idx < Int32(self.layout.num_origin_words):
                softmax_origin = Int32(resolved_origin0)
                if resolved_route_flags == Int32(0):
                    softmax_origin = Int32(-1)
                self._smem_words[stage_base + lane_idx] = softmax_origin
        else:
            # SWAP with a coarse KV atom expands the two resolved KV64
            # fragments into the four physical K32 origins consumed by its
            # four softmax warps.
            if lane_idx < Int32(4):
                fragment_idx = lane_idx >> Int32(1)
                softmax_origin = Int32(resolved_origin0)
                valid = (resolved_route_flags & Int32(1)) != Int32(0)
                if fragment_idx == Int32(1):
                    softmax_origin = Int32(resolved_origin1)
                    valid = (resolved_route_flags & Int32(2)) != Int32(0)
                softmax_origin = softmax_origin + (lane_idx & Int32(1)) * Int32(32)
                if not valid or softmax_origin >= seq_len_kv:
                    softmax_origin = Int32(-1)
                self._smem_words[stage_base + lane_idx] = softmax_origin

        if cutlass.const_expr(self.cfg.use_kv_valid_bits):
            assert self.kv_valid_bits is not None
            assert self.layout.token_words_word_offset is not None
            _, logical_b_idx = _logical_head_batch(stage_info, self.h_k_idx, self.b_idx)
            token_chunk = Uint32(0)
            if lane_idx < Int32(self.layout.num_origin_words):
                if softmax_origin >= Int32(0) and softmax_origin < seq_len_kv:
                    physical_word_idx = softmax_origin >> Int32(5)
                    if physical_word_idx < Int32(self.cfg.num_kv_valid_words):
                        bitset_offset = (
                            logical_b_idx * Int32(self.cfg.num_kv_valid_words)
                            + physical_word_idx
                        )
                        physical_word = Uint32(self.kv_valid_bits[bitset_offset])
                        token_chunk_mask = Uint32(0xFFFFFFFF)
                        if cutlass.const_expr(self.layout.softmax_atom_size < 32):
                            token_chunk_mask = Uint32(
                                (1 << self.layout.softmax_atom_size) - 1
                            )
                        remaining_tokens = seq_len_kv - softmax_origin
                        if remaining_tokens < Int32(self.layout.softmax_atom_size):
                            # This implies 0 < remaining_tokens < atom_size <= 32,
                            # so the runtime shift is never the undefined width 32.
                            token_chunk_mask = (Uint32(1) << remaining_tokens) - Uint32(
                                1
                            )
                        token_chunk = (
                            physical_word >> (softmax_origin & Int32(0x1F))
                        ) & token_chunk_mask

            # Pack each naturally aligned power-of-two subgroup into one K32
            # word. All lanes execute each butterfly; only subgroup leaders
            # publish the four logical words.
            origins_per_word = 32 // self.layout.softmax_atom_size
            logical_word = token_chunk << (
                (lane_idx & Int32(origins_per_word - 1))
                * Int32(self.layout.softmax_atom_size)
            )
            if cutlass.const_expr(origins_per_word >= 2):
                logical_word = logical_word | Uint32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=logical_word,
                        offset=1,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                )
            if cutlass.const_expr(origins_per_word >= 4):
                logical_word = logical_word | Uint32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=logical_word,
                        offset=2,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                )
            is_subgroup_leader = (lane_idx & Int32(origins_per_word - 1)) == Int32(0)
            if lane_idx < Int32(self.layout.num_origin_words) and is_subgroup_leader:
                logical_word_idx = lane_idx // Int32(origins_per_word)
                self._smem_words[
                    stage_base
                    + Int32(self.layout.token_words_word_offset)
                    + logical_word_idx
                ] = Int32(logical_word)
        cute.arch.sync_warp()

    @cute.jit
    def _store_route_keeps(
        self,
        stage_info: StageInfo,
        resolved_origin0: Int32,
        resolved_origin1: Int32,
        resolved_route_flags: Int32,
    ) -> None:
        """Stage a Keeps route, its token words, and the runtime full-route bit.

        Q64 always stages all four words to keep metadata movement branch-free;
        Q128 omits them when the full-route bit says token masking can be
        skipped. The consumer follows the same tile-specific convention.
        """

        assert self.layout.route_flags_word_offset is not None
        origin0 = Int32(resolved_origin0)
        origin1 = Int32(resolved_origin1)
        route_flags = Int32(resolved_route_flags)
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        token_word = Uint32(0)
        route_token_mask_is_full = cutlass.Boolean(False)
        if cutlass.const_expr(self.cfg.use_kv_valid_bits):
            assert self.kv_valid_bits is not None
            task_cache = _decode_gen_task_cache(stage_info)
            seq_len_kv = Int32(task_cache[_TASK_CACHE_SEQ_LEN_KV])
            _, logical_b_idx = _logical_head_batch(stage_info, self.h_k_idx, self.b_idx)
            token_word, route_token_mask_is_full = (
                _load_block_sparse_token_word_from_route(
                    self.kv_valid_bits,
                    origin0,
                    origin1,
                    route_flags,
                    seq_len_kv,
                    logical_b_idx,
                    self.cfg.num_kv_valid_words,
                    lane_idx,
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
            if cutlass.const_expr(self.cfg.use_kv_valid_bits):
                route_flags = route_flags | (
                    Int32(route_token_mask_is_full)
                    * Int32(_TOKEN_MASK_ROUTE_IS_FULL_FLAG)
                )
            self._smem_words[
                stage_base + Int32(self.layout.route_flags_word_offset)
            ] = route_flags
        if cutlass.const_expr(self.cfg.use_kv_valid_bits):
            assert self.layout.token_words_word_offset is not None
            if lane_idx < Int32(4):
                if cutlass.const_expr(self.cfg.tile_size_q == 64):
                    # Q64 benchmarks better with unconditional word staging;
                    # its route flag only controls the consumer predicate.
                    self._smem_words[
                        stage_base
                        + Int32(self.layout.token_words_word_offset)
                        + lane_idx
                    ] = Int32(token_word)
                elif not route_token_mask_is_full:
                    self._smem_words[
                        stage_base
                        + Int32(self.layout.token_words_word_offset)
                        + lane_idx
                    ] = Int32(token_word)
        cute.arch.sync_warp()

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

        if cutlass.const_expr(self.cfg.use_keeps_mma_ab):
            self._store_route_keeps(
                stage_info,
                resolved_origin0,
                resolved_origin1,
                resolved_route_flags,
            )
        else:
            self._store_route_swaps(
                stage_info,
                resolved_origin0,
                resolved_origin1,
                resolved_route_flags,
            )

    @cute.jit
    def _load_route_swaps_values(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32, Uint32, Uint32]:
        """Load one SWAP warp's view through the original seven-slot ABI.

        The five values encode origins 0/1 directly, origin 2 in the route-flags
        slot, origin 3 as unsigned bits in token-word 0, and the packed logical
        K32 mask in token-word 1. Token-word slots 2/3 remain unused by SWAP.
        """

        stage_base = self._consumer_stage_base()
        local_warp_idx = Int32(_decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_IDX])
        warp_origin_base = stage_base + local_warp_idx * Int32(
            self.layout.origins_per_warp
        )
        origin0 = Int32(self._smem_words[warp_origin_base])
        origin1 = Int32(-1)
        origin2 = Int32(-1)
        origin3 = Int32(-1)
        if cutlass.const_expr(self.layout.origins_per_warp >= 2):
            origin1 = Int32(self._smem_words[warp_origin_base + Int32(1)])
        if cutlass.const_expr(self.layout.origins_per_warp >= 3):
            origin2 = Int32(self._smem_words[warp_origin_base + Int32(2)])
        if cutlass.const_expr(self.layout.origins_per_warp >= 4):
            origin3 = Int32(self._smem_words[warp_origin_base + Int32(3)])

        token_word = Uint32(0xFFFFFFFF)
        if cutlass.const_expr(self.cfg.use_kv_valid_bits):
            assert self.layout.token_words_word_offset is not None
            token_word = Uint32(
                self._smem_words[
                    stage_base
                    + Int32(self.layout.token_words_word_offset)
                    + local_warp_idx
                ]
            )
        return origin0, origin1, origin2, origin3.bitcast(Uint32), token_word

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

        if cutlass.const_expr(not self.cfg.use_keeps_mma_ab):
            # Reuse the original seven-slot task ABI: Task7 interprets the
            # middle fields as origin2, origin3 bits, and the logical K32 mask.
            origin0, origin1, origin2, origin3_bits, token_word = (
                self._load_route_swaps_values(stage_info)
            )
            return (
                origin0,
                origin1,
                origin2,
                origin3_bits,
                token_word,
                Uint32(0xFFFFFFFF),
                Uint32(0xFFFFFFFF),
            )

        assert self.layout.route_flags_word_offset is not None
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
            route_token_mask_is_full = cutlass.Boolean(
                (route_flags & Int32(_TOKEN_MASK_ROUTE_IS_FULL_FLAG)) != Int32(0)
            )
            if cutlass.const_expr(self.cfg.tile_size_q == 64):
                # Keep Q64's metadata load branch-free. The full-route flag
                # still skips the token predicate after the score load.
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
            elif not route_token_mask_is_full:
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
