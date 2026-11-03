"""Token communication implementations for MegaMoE-style kernels.

Current implementation: token-in pull with token-back push.  The standalone
``dispatch_kernel`` uses the same object methods as the fused MegaMoE kernel.
"""

import dataclasses
import os
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import (
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass.base_dsl.dsl import extract_mlir_attributes

try:
    from cutlass.cute import iket as _iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from .iket_compat import iket as _iket
from .megamoe_config import MegaMoEDiagnostics

from .grid_sync import software_grid_sync
from .nccl_gin_layout import build_megamoe_gin_window_layout
from .ptx_helpers import (
    copy_b128_raw,
    copy_b128x4_strided_raw,
    cp_reduce_async_bulk_add_noftz_bf16_s2g,
    device_trap,
    fns_b32,
    ldg_b128_to_smem_raw,
    ldg_b128x4_to_smem_predicated_raw,
    ldg_b32_predicated_raw,
    ldg_b64_predicated_raw,
    ldg_b32_raw,
    ldg_b64_raw,
    ldg_f32_raw,
    ldg_route_experts_b128_raw,
    ldg_route_match_b128_raw,
    ldsm_b32_raw,
    pack_f32x2_to_bf16x2,
    read_clock64,
    read_globaltimer,
    red_add_relaxed_sys_s32_raw,
    cp_async_16_smem_raw,
    cp_async_commit_group_raw,
    cp_async_wait_group_raw,
    fence_proxy_async_shared_cta_raw,
    st_shared_b32_raw,
    st_shared_release_cta_b32_raw,
    ld_shared_acquire_cta_b32_raw,
    lds_b32_addr_raw,
    red_add_release_gpu_s32,
    red_add_relaxed_sys_u64_raw,
    red_add_release_sys_s32_raw,
    red_add_release_sys_u64_raw,
    stg_b128_words_raw,
    stg_b32_raw,
    stg_b64_raw,
    tma_load_1d_raw,
    tma_store_1d,
    unpack_bf16x2_to_f32x2,
    _fence_rel_sys,
)
from .flag_batch import GpuReleaseFlagBatchTracker, SysReleaseFlagBatchTracker
from common.direct_meta import (
    DIRECT_META_COUNT_LIMIT,
    DIRECT_META_COUNT_SHIFT,
    DIRECT_META_FIELD_MASK,
    DIRECT_META_SF_SHIFT,
)
from .token_segments import merged_token_segment, expert_token_tiles
from .sf_swizzle import sf_atom_int32_offset
from cutlass._mlir import ir
from moe_nvfp4_swapab.moe_utils import _nanosleep, spin_wait
from common.pipeline_timing import (
    COMBINE_TIMING_ARRIVAL_DONE,
    COMBINE_TIMING_FC2_WAIT_NS,
    COMBINE_TIMING_FC2_WAIT_START,
    COMBINE_TIMING_FLUSH_CREDIT_DONE,
    COMBINE_TIMING_ISSUE_DONE,
    COMBINE_TIMING_MAX_FC2_WAIT_NS,
    COMBINE_TIMING_MESSAGE_COUNT,
    COMBINE_TIMING_PUBLISH_ISSUE_NS,
    COMBINE_TIMING_REDUCE_DONE,
    COMBINE_TIMING_REDUCE_NS,
    COMBINE_TIMING_ROUTE_DONE,
    COMBINE_TIMING_ROUTE_INCLUSIVE_NS,
    COMBINE_TIMING_SYNC_DONE,
    COMBINE_TIMING_WAITED_HIT_COUNT,
    PIPELINE_STAGE_COMBINE,
    PIPELINE_STAGE_COMBINE_REDUCE_RING,
    PIPELINE_STAGE_COMBINE_REDUCE_SCRATCH,
    PIPELINE_STAGE_FC_EPI_EXIT,
    PIPELINE_STAGE_FC_RING_RELEASE,
    PIPELINE_STAGE_DISPATCH,
    PIPELINE_STAGE_FANOUT,
    PIPELINE_STAGE_FANOUT_COPY,
    PIPELINE_STAGE_FANOUT_PUBLISH,
    PIPELINE_STAGE_FANOUT_ROUTE,
    PIPELINE_STAGE_FANOUT_SYNC,
    PIPELINE_STAGE_FANOUT_WAIT,
    PIPELINE_STAGE_META,
    PIPELINE_STAGE_META_LAYOUT,
    PIPELINE_STAGE_META_PUBLISH,
    PIPELINE_STAGE_META_REDIST,
    PIPELINE_STAGE_META_STEP3,
    PIPELINE_STAGE_META_WAIT,
    PIPELINE_STAGE_PUSH,
    PIPELINE_STAGE_TOKEN_BACK,
    PIPELINE_STAGE_PUSH_FLUSH,
    PIPELINE_STAGE_PUSH_ISSUE,
    PIPELINE_STAGE_PUSH_SYNC,
    PIPELINE_STAGE_TAIL,
    PIPELINE_STAGE_FANOUT_WAIT_SLOT0,
    PIPELINE_STAGE_META_REDIST_ISSUE,
    PIPELINE_STAGE_META_REDIST_SYNC,
    PIPELINE_STAGE_PUSH_PROLOGUE,
    PIPELINE_STAGE_PUSH_PACK,
    PIPELINE_STAGE_PUSH_PACK_ROUTE,
    PIPELINE_STAGE_PUSH_PACK_COPY,
    PIPELINE_STAGE_PUSH_PROLOGUE_SETUP,
    PIPELINE_STAGE_SF_MATERIALIZE,
)


# PIC-C supports at most 32 event types per compilation unit. Keep optional
# profiling under three transport-neutral event names and distinguish inclusive
# envelopes, active work, and explicit waits with NativeDump payloads. No range
# is emitted inside a polling iteration, so trace volume is independent of the
# number of spins. Explicit SSA range_start/range_end tokens prevent nested,
# predicated ranges from being closed by an unrelated stack pop.
#
# The long combine envelope retains a dedicated no-payload event because
# CuTeDSL 4.6.2 corrupts its payload across nested ranges. The summarizer maps
# that compatibility event back to megamoe_envelope/combine_return. Retire the
# special event when the nested-payload compiler defect is fixed.
IKET_MEGA_ENVELOPE_META = 1
IKET_MEGA_ENVELOPE_DISPATCH_PUT = 2
IKET_MEGA_ENVELOPE_RELAY_FANOUT = 3
# No emitter since the staged Stage 3 path was removed (2026-09-21); the code
# stays reserved so older traces decode unchanged.
IKET_MEGA_ENVELOPE_RELAY_COPY = 4
# Logical decoder code for the dedicated no-payload internode_combine event.
IKET_MEGA_ENVELOPE_COMBINE_RETURN = 5

IKET_MEGA_WORK_DISPATCH_MESSAGE = 1
IKET_MEGA_WORK_FANOUT_MESSAGE = 2
# No emitter since 2026-09-21 (staged Stage 3 path removed); reserved code.
IKET_MEGA_WORK_RELAY_COPY_MESSAGE = 3
IKET_MEGA_WORK_COMBINE_MESSAGE = 4
IKET_MEGA_WORK_PULL_EXPERT_SCAN = 5
IKET_MEGA_WORK_PULL_SOURCE_ROUTE = 6
IKET_MEGA_WORK_PULL_TMA_LOAD_ISSUE = 7
IKET_MEGA_WORK_PULL_SF_WEIGHT_LOAD = 8
IKET_MEGA_WORK_PULL_SF_WEIGHT_STORE = 9
IKET_MEGA_WORK_PULL_TMA_STORE_ISSUE = 10
IKET_MEGA_WORK_PULL_ARRIVAL_PUBLISH = 11

IKET_MEGA_COMBINE_WORK_ROUTE_LOOKUP = 1
IKET_MEGA_COMBINE_WORK_REDUCE = 2
IKET_MEGA_COMBINE_WORK_PUBLISH_ISSUE = 3
IKET_MEGA_COMBINE_WORK_FLUSH_SIGNAL = 4

IKET_MEGA_WAIT_DISPATCH_SIGNAL = 1
IKET_MEGA_WAIT_RELAY_LANDING = 2
IKET_MEGA_WAIT_FC2_DONE = 3
IKET_MEGA_WAIT_COMBINE_SIGNAL = 4
# Payload 5 remains reserved for decoding historical traces; current NVLink
# barrier waits use the fixed-name, no-payload nvlink_barrier_wait event.
IKET_MEGA_WAIT_NVLINK_BARRIER = 5
IKET_MEGA_WAIT_RAIL_BARRIER = 6
IKET_MEGA_WAIT_PULL_TMA_LOAD = 7
IKET_MEGA_WAIT_PULL_TMA_STORE = 8
IKET_MEGA_WAIT_COMBINE_CHUNKS = 9
IKET_MEGA_GRID_SYNC_WAIT_DISPATCH = 1
IKET_MEGA_GRID_SYNC_WAIT_RELAY_FANOUT = 2
IKET_MEGA_GRID_SYNC_WAIT_COMBINE = 3
IKET_MEGA_GRID_SYNC_WAIT_NVLINK_PROLOGUE = 4
IKET_MEGA_GRID_SYNC_WAIT_NVLINK_EPILOGUE = 5
IKET_MEGA_GRID_SYNC_WAIT_RAIL_PROLOGUE = 6
IKET_MEGA_GRID_SYNC_WAIT_RAIL_EPILOGUE = 7
IKET_MEGA_GRID_SYNC_WAIT_DISPATCH_COALESCE = 8
IKET_MEGA_GRID_SYNC_WAIT_COMBINE_COALESCE = 9
# Early Stage 1 (field 60): relay-only publication of every CTA's reservations
# and tokens_to_node entries before the flattened Stage 1 pack.
IKET_MEGA_GRID_SYNC_WAIT_PACK_FLAT = 10
# deterministic_slots (main, 2026-10): publication of the per-CTA expert
# histogram, then the distributed cross-CTA exclusive prefix, before the
# warp bases are final.
IKET_MEGA_GRID_SYNC_WAIT_DET_SLOTS_HIST = 11
IKET_MEGA_GRID_SYNC_WAIT_DET_SLOTS_SCAN = 12

# IKET range emission is an instance property (``self.iket_enabled``,
# ``self.iket_detailed_ranges``) set from ``MegaMoEDiagnostics``: production
# traces retain fixed-count phase envelopes, barriers, waits and %globaltimer
# landmarks; only diagnostic traces emit ranges whose count scales with relay
# messages or pull iterations. Suppressing those ranges is a compile-time
# choice and does not change the kernel data or sync paths.

# The BF16 gather-reduce moves 16-byte vectors (b128 loads and stores); the
# b64 form and the non-issue-first loop lost the fixed-allocation A/B
# (2797660) and were removed 2026-09-21. Every combine worker serves one
# message.
_GIN_COMBINE_VECTOR_BYTES = 16
# Stage 1 and the combine return are coalesced range puts whose credits ride
# on bundled signals; the per-message put paths, the per-chunk reduce loop
# and the separate strong-signal credits were removed 2026-09-21.
# ncclGinSignalOp_t (nccl_device/gin/gin_device_host_common.h):
# ncclGinSignalInc = 0, ncclGinSignalAdd = 1.
_NCCL_GIN_SIGNAL_OP_ADD = 1
# The packed gather-reduce runs on a deep-prefetch ring of cp.async rounds in
# this CTA's AB stage buffers once the GEMM warps have exited (scratch path
# with one round in flight before that, upgraded mid-message when the flag
# flips), and SM0 arrives at the tail rail barrier right after the last
# return put (Invariant I8). The scratch-only reduce, the fixed path choice
# and the arrive-and-wait tail barrier were removed 2026-09-21.
# Stage 3 bucket prefetch: every dispatch warp walks all expert x source-slot
# buckets serially, and each bucket costs two dependent global loads (the
# static expected count and the arrival counts) whether or not the warp serves
# it, about 2 us per bucket. With the prefetch the static expected counts are
# loaded for all buckets up front, and the arrival counts are read only by the
# warps that serve a bucket.
# DIAGNOSTIC ONLY (results are wrong when != 0): skip parts of the direct-L1
# fan-out hit so the busiest warp's copy/publish envelopes attribute the
# per-message cost. 1 skips the swizzled scale-factor scatter (112 scalar
# remote stores per hit at H=7168), 2 skips the token DATA row copy, 3 skips
# both; the reservation, weight, metadata, fence and ready counts stay so the
# owner's FC1 gating and the kernel's control flow are unchanged.
# The mask arrives as ``MegaMoEDiagnostics.diag_fanout_skip`` (instance
# attributes ``self.diag_fanout_skip_sf`` / ``self.diag_fanout_skip_data``).
# Grid-sync counter slot of the relay-only grid syncs (Stage 1 coalesced
# rendezvous, fan-out release, the two combine rendezvous). The dispatch group
# (dispatch_barrier, the NVLink barriers of inter_meta_exchange) keeps slot 0.
# software_grid_sync counts one arrival per CTA per call and flips bit 31 when
# num_sms arrivals land, so two warp groups syncing on the same word at the
# same time corrupt each other's round: the early push let the relay group's
# Stage 1 rendezvous overlap the dispatch group's Step 5b/6 grid syncs and
# hung staged runs at N2R2 (hang dump 3061640: rank 0 never left its Stage 1
# grid sync). The relay group therefore uses slot 1 and the dispatch group
# keeps slot 0 (the shared-word rollback MEGA_RELAY_GRID_SYNC_SLOT=0 was
# removed 2026-09-21).
_RELAY_GRID_SYNC_SLOT = 1
# The Step 3 metadata puts post through the GPU-shared GIN handle, the same
# ordered-commit protocol on the shared GDAKI QP that Stage 1, the fan-out
# and the combine use. A THREAD-scoped handle raced that QP once the early
# push let SM0's Stage 1 put overlap Step 3 (hang dumps 3061957/3061958,
# fix 96050ec); the rollback MEGA_STEP3_GIN_SHARING=thread was removed
# 2026-09-21.
# After the relay grid sync, lanes 0..slots-1 of SM0 relay warp 0 post the
# coalesced combine return puts from one instruction stream (the serial
# issuer was removed 2026-09-21).
# Diagnostic only: publish the FC2 done counters with gpu-scope release reductions
# even when NCCL GIN peers read them over NVLink. Measures the price of the
# system-scope publication fence at the end of every CTA's tile loop; it is not
# a candidate default without a scope argument for the peer readers (I9).
# Instance attribute ``self.fc2_done_gpu_scope`` from ``MegaMoEDiagnostics``.
# The intra-node token-back warps start their pushes only after SM0 issued
# the combine return puts (monotonic per-rank issue counter), so the pushes
# no longer overlap the straggler CTAs' FC2 publication drains and the
# combine's critical route/reduce (3098857/3098858). The immediate and the
# tile-loop-exit-gated forms were removed 2026-09-21.


# ---------------------------------------------------------------------------
# Low-precision combine wire format (the central driver for the token-back
# quantized combine path: fc2 epilogue encoder, token_comm push, and the
# topk_reduce receiver all describe themselves through one CombineFormat).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CombineFormat:
    """Wire format of the cross-rank combine (token-back) payload.

    The fc2 epilogue quantizes each token's hidden vector into a packed data
    plane (``combine_quant``) plus a per-block scale plane (``combine_sf``); the
    receiver (topk_reduce) dequantizes and reduces over topk. The consumer is
    ALWAYS a software dequant -- never a tensor-core MMA -- so the format is
    fully self-defined here, with no hardware scale-layout constraint.

    Canonical string ``"{scale_block}{act}x{scale}"`` (leading number = scale
    block size in hidden elements), e.g. ``"16e2m1xbf16"`` (per-16 bf16 amax +
    fp4 data) or ``"32e4m3xe8m0"`` (standard MXFP8). ``"bf16"`` is the
    no-staging baseline: bf16 fc2 terms reduced directly, no scale plane.
    """

    # Element/scale tag <-> cuTe dtype.  Only the dtypes a real format uses
    # today (the bf16 baseline is ``scale_dtype is None``); extend when a new
    # format is actually added rather than ahead of need.
    _act_by_tag: ClassVar[Dict[str, type]] = {
        "e2m1": cutlass.Float4E2M1FN,
        "e4m3": cutlass.Float8E4M3FN,
        "e5m2": cutlass.Float8E5M2,
    }
    _scale_by_tag: ClassVar[Dict[str, type]] = {
        "bf16": cutlass.BFloat16,
        "e8m0": cutlass.Float8E8M0FNU,
    }

    act_dtype: type              # cuTe dtype of the packed data-plane element
    scale_dtype: Optional[type]  # cuTe dtype of a scale entry; None == bf16 baseline
    scale_block: Optional[int]   # hidden elements per scale entry; None == baseline

    def __post_init__(self):
        allowed_act = {cutlass.BFloat16, *self._act_by_tag.values()}
        if self.act_dtype not in allowed_act:
            raise ValueError(f"combine act_dtype {self.act_dtype} not in {allowed_act}.")
        allowed_scale = {None, *self._scale_by_tag.values()}
        if self.scale_dtype not in allowed_scale:
            raise ValueError(f"combine scale_dtype {self.scale_dtype} not in {allowed_scale}.")
        if self.scale_dtype is None:                       # bf16 no-staging baseline
            if self.act_dtype is not cutlass.BFloat16 or self.scale_block is not None:
                raise ValueError("baseline must be bf16 act with scale_block=None.")
            return
        if self.act_dtype is cutlass.BFloat16:
            raise ValueError("a quantized combine cannot use a bf16 act dtype.")
        # The scale dtype pins the block: per-16 bf16 amax / per-32 e8m0 power-of-two.
        if self.scale_dtype is cutlass.BFloat16 and self.scale_block != 16:
            raise ValueError("bf16 amax scale requires scale_block == 16.")
        if self.scale_dtype is cutlass.Float8E8M0FNU and self.scale_block != 32:
            raise ValueError("e8m0 scale requires scale_block == 32.")

    @property
    def is_quantized(self) -> bool:
        """``False`` for the bf16 (no-staging) baseline."""
        return self.scale_dtype is not None

    @property
    def name(self) -> str:
        if not self.is_quantized:
            return "bf16"
        act_tag = next(t for t, d in self._act_by_tag.items() if d is self.act_dtype)
        scale_tag = next(t for t, d in self._scale_by_tag.items() if d is self.scale_dtype)
        return f"{self.scale_block}{act_tag}x{scale_tag}"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def parse(cls, text: str) -> "CombineFormat":
        """Build a CombineFormat from its canonical string (the argparser entry).

        Only the handful of supported wire formats are accepted; each key is the
        exact string ``name`` produces (so ``parse(str(fmt)) == fmt``).
        """
        # (act_dtype, scale_dtype, scale_block); None scale == bf16 baseline.
        specs = {
            "bf16":        (cutlass.BFloat16,     None,                  None),
            "16e2m1xbf16": (cutlass.Float4E2M1FN, cutlass.BFloat16,      16),
            "32e4m3xe8m0": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            "32e5m2xe8m0": (cutlass.Float8E5M2,   cutlass.Float8E8M0FNU, 32),
        }
        token = text.strip().lower()
        if token not in specs:
            raise ValueError(
                f"invalid combine_format {text!r}: expected one of {tuple(specs)}."
            )
        act_dtype, scale_dtype, scale_block = specs[token]
        return cls(act_dtype=act_dtype, scale_dtype=scale_dtype, scale_block=scale_block)


@dataclasses.dataclass(frozen=True)
class TokenSrcMetadata:
    """Per pool-token routing record: written by token-in, read by token-back
    and the fc2 combine-redirect epilogue.

    Wire format is one i64: low 32b = ``src_token`` (needs full width); high 32b
    = ``(src_rank << 16) | src_topk`` (``src_rank < world_size`` and
    ``src_topk < num_topk`` both fit in 16b).  ``load`` / ``store`` accept either
    a ``cute.Pointer`` or a raw ``Int64`` byte address.
    """

    src_rank: Int32
    src_token: Int32
    src_topk: Int32

    nbytes: ClassVar[int] = 8

    def _pack(self) -> Int64:
        hi = (Int64(self.src_rank) << Int64(16)) | Int64(self.src_topk)
        return (hi << Int64(32)) | (Int64(self.src_token) & Int64(0xFFFFFFFF))

    @staticmethod
    def _i64_ptr(addr: Union[cute.Pointer, Int64]) -> cute.Pointer:
        addr_i = addr if isinstance(addr, Int64) else addr.toint()
        return cute.make_ptr(Int64, addr_i, AddressSpace.gmem, assumed_align=8)

    def store(self, addr: Union[cute.Pointer, Int64]) -> None:
        cute.arch.store(self._i64_ptr(addr), self._pack(), scope="gpu")

    @classmethod
    def load(cls, addr: Union[cute.Pointer, Int64]) -> "TokenSrcMetadata":
        v = Int64(cute.arch.load(cls._i64_ptr(addr), Int64, scope="gpu"))
        hi = v >> Int64(32)
        return cls(
            src_rank=Int32((hi >> Int64(16)) & Int64(0xFFFF)),
            src_token=Int32(v & Int64(0xFFFFFFFF)),
            src_topk=Int32(hi & Int64(0xFFFF)),
        )


_MLIR_VALUE_FIELDS = (
    "input_token_buffer",
    "input_sf_buffer",
    "input_fc1_activation_per_token_scale_buffer",
    "topk_idx",
    "input_topk_weights_buffer",
    "expert_send_count",
    "expert_recv_count",
    "expert_recv_count_sum",
    "src_token_topk_idx",
    "internode_send_count",
    "internode_recv_splits",
    "internode_send_token_count",
    "tokens_to_node",
    "token_node_slot",
    "intra_token_count",
    "internode_recv_count_per_source",
    "relay_fanout_reservation",
    "relay_fanout_slot",
    "expert_pool_base",
    "expert_sf_pool_base",
    "relay_pool_base",
    "relay_sf_pool_base",
    "relay_direct_bucket_meta",
    "relay_row_ready",
    "relay_sf_staging",
    "internode_recv_count_per_source_snapshot",
    "expert_recv_count_snapshot",
    "expert_recv_count_sum_snapshot",
    "intra_token_count_snapshot",
    "expert_pool_base_snapshot",
    "expert_sf_pool_base_snapshot",
    "relay_pool_base_snapshot",
    "relay_sf_pool_base_snapshot",
    "expert_scheduled_count_snapshot",
    "pool_end_snapshot",
    "l1_arrival_count_snapshot",
    "fc1_input_token_buffer",
    "fc1_input_sf_buffer",
    "dispatched_fc1_activation_per_token_scale_buffer",
    "fc1_input_topk_weights_buffer",
    "fc1_ready_counter",
    "token_src_metadata",
    "combine_output",
    "combine_sf",
    "fc2_output_workspace",
    "fc2_output_sf",
    "fc2_done_counter",
    "token_back_schedule_counter",
    "dispatch_route_extent",
    "nvlink_barrier_signal",
    "nvlink_barrier_counter",
    "internode_barrier_counter",
    "combine_issue_counter",
    "combine_phase_snapshot",
    "tail_combine_output",
    "tail_internode_combine",
    "tail_reduced_output",
    "pipeline_timing",
    "combine_timing_landmarks",
    "fc_tile_timeline",
    "gin_dev_comm",
    "gin_window",
    "grid_sync_counter",
    "local_zero_prefix",
    "shared_zero_prefix",
    "peer_rank_ptr_mapper",
    "local_rank",
    "cta_expert_hist",
)

_CONST_FIELDS = (
    "world_size",
    "num_total_experts",
    "num_experts_per_rank",
    "num_topk",
    "hidden_bytes",
    "sf_uint32_per_token",
    "ranks_per_node",
    "token_padding_block",
    "sf_padding_block",
    "pool_token_capacity",
    "pool_sf_capacity",
    "sm_count",
    "system_scope_done_flags",
)

class TokenCommArgs:
    """MegaMoE token communication argument bundle."""

    def __init__(
        self,
        *,
        input_token_buffer: cute.Tensor,
        input_sf_buffer: cute.Tensor,
        topk_idx: cute.Tensor,
        input_topk_weights_buffer: cute.Tensor,
        expert_send_count: cute.Tensor,
        expert_recv_count: cute.Tensor,
        expert_recv_count_sum: cute.Tensor,
        src_token_topk_idx: cute.Tensor,
        fc1_input_token_buffer: cute.Tensor,
        fc1_input_sf_buffer: cute.Tensor,
        fc1_input_topk_weights_buffer: cute.Tensor,
        fc1_ready_counter: cute.Tensor,
        token_src_metadata: cute.Tensor,
        combine_output: cute.Tensor,
        nvlink_barrier_signal: cute.Tensor,
        nvlink_barrier_counter: cute.Tensor,
        grid_sync_counter: cute.Tensor,
        local_zero_prefix: cute.Tensor,
        shared_zero_prefix: cute.Tensor,
        peer_rank_ptr_mapper: Any,
        world_size: int,
        local_rank: int,
        num_total_experts: int,
        num_experts_per_rank: int,
        num_topk: int,
        hidden_bytes: int,
        sf_uint32_per_token: int,
        ranks_per_node: int = 0,  # 0: one node of world_size ranks (the single-node kernels never pass it)
        token_padding_block: int,
        sf_padding_block: int,
        sm_count: int,
        system_scope_done_flags: bool = False,
        fc2_output_workspace: cute.Tensor = None,
        fc2_done_counter: cute.Tensor = None,
        token_back_schedule_counter: cute.Pointer = None,
        combine_sf: cute.Tensor = None,
        fc2_output_sf: cute.Tensor = None,
        dispatch_route_extent: cute.Tensor = None,
        # Optional per-token fc1 activation scale: the user's (T,) fp32 tensor
        # on the symmetric heap (peer-pulled) and its pool-ordered local copy.
        input_fc1_activation_per_token_scale_buffer: cute.Tensor = None,
        dispatched_fc1_activation_per_token_scale_buffer: cute.Tensor = None,
        internode_send_count: cute.Tensor = None,
        internode_recv_splits: cute.Tensor = None,
        internode_send_token_count: cute.Tensor = None,
        tokens_to_node: cute.Tensor = None,
        token_node_slot: cute.Tensor = None,
        intra_token_count: cute.Tensor = None,
        internode_recv_count_per_source: cute.Tensor = None,
        relay_fanout_reservation: cute.Tensor = None,
        relay_fanout_slot: cute.Tensor = None,
        expert_pool_base: cute.Tensor = None,
        expert_sf_pool_base: cute.Tensor = None,
        relay_pool_base: cute.Tensor = None,
        relay_sf_pool_base: cute.Tensor = None,
        relay_direct_bucket_meta: cute.Tensor = None,
        relay_row_ready: cute.Tensor = None,
        relay_sf_staging: cute.Tensor = None,
        internode_recv_count_per_source_snapshot: cute.Tensor = None,
        expert_recv_count_snapshot: cute.Tensor = None,
        expert_recv_count_sum_snapshot: cute.Tensor = None,
        intra_token_count_snapshot: cute.Tensor = None,
        expert_pool_base_snapshot: cute.Tensor = None,
        expert_sf_pool_base_snapshot: cute.Tensor = None,
        relay_pool_base_snapshot: cute.Tensor = None,
        relay_sf_pool_base_snapshot: cute.Tensor = None,
        expert_scheduled_count_snapshot: cute.Tensor = None,
        pool_end_snapshot: cute.Tensor = None,
        l1_arrival_count_snapshot: cute.Tensor = None,
        internode_barrier_counter: cute.Tensor = None,
        combine_issue_counter: cute.Tensor = None,
        combine_phase_snapshot: cute.Tensor = None,
        tail_combine_output: cute.Tensor = None,
        tail_internode_combine: cute.Tensor = None,
        tail_reduced_output: cute.Tensor = None,
        pipeline_timing: cute.Tensor = None,
        combine_timing_landmarks: cute.Tensor = None,
        fc_tile_timeline: cute.Tensor = None,
        gin_dev_comm: Any = None,
        gin_window: Any = None,
        pool_token_capacity: int = 0,
        pool_sf_capacity: int = 0,
        # (max_ctas, num_total_experts) Int32 per-CTA routing histogram; only
        # bound when the dispatch runs with ``deterministic_slots``.
        cta_expert_hist: cute.Tensor = None,
    ):
        self.cta_expert_hist = cta_expert_hist
        self.input_token_buffer = input_token_buffer
        self.input_sf_buffer = input_sf_buffer
        self.input_fc1_activation_per_token_scale_buffer = (
            input_fc1_activation_per_token_scale_buffer
        )
        self.dispatched_fc1_activation_per_token_scale_buffer = (
            dispatched_fc1_activation_per_token_scale_buffer
        )
        self.topk_idx = topk_idx
        self.input_topk_weights_buffer = input_topk_weights_buffer
        self.expert_send_count = expert_send_count
        self.expert_recv_count = expert_recv_count
        self.expert_recv_count_sum = expert_recv_count_sum
        self.src_token_topk_idx = src_token_topk_idx
        self.internode_send_count = internode_send_count
        self.internode_recv_splits = internode_recv_splits
        self.internode_send_token_count = internode_send_token_count
        self.tokens_to_node = tokens_to_node
        self.token_node_slot = token_node_slot
        self.intra_token_count = intra_token_count
        self.internode_recv_count_per_source = internode_recv_count_per_source
        self.relay_fanout_reservation = relay_fanout_reservation
        self.relay_fanout_slot = relay_fanout_slot
        self.expert_pool_base = expert_pool_base
        self.expert_sf_pool_base = expert_sf_pool_base
        self.relay_pool_base = relay_pool_base
        self.relay_sf_pool_base = relay_sf_pool_base
        self.relay_direct_bucket_meta = relay_direct_bucket_meta
        self.relay_row_ready = relay_row_ready
        self.relay_sf_staging = relay_sf_staging
        self.internode_recv_count_per_source_snapshot = (
            internode_recv_count_per_source_snapshot
        )
        self.expert_recv_count_snapshot = expert_recv_count_snapshot
        self.expert_recv_count_sum_snapshot = expert_recv_count_sum_snapshot
        self.intra_token_count_snapshot = intra_token_count_snapshot
        self.expert_pool_base_snapshot = expert_pool_base_snapshot
        self.expert_sf_pool_base_snapshot = expert_sf_pool_base_snapshot
        self.relay_pool_base_snapshot = relay_pool_base_snapshot
        self.relay_sf_pool_base_snapshot = relay_sf_pool_base_snapshot
        self.expert_scheduled_count_snapshot = expert_scheduled_count_snapshot
        self.pool_end_snapshot = pool_end_snapshot
        self.l1_arrival_count_snapshot = l1_arrival_count_snapshot
        self.fc1_input_token_buffer = fc1_input_token_buffer
        self.fc1_input_sf_buffer = fc1_input_sf_buffer
        self.fc1_input_topk_weights_buffer = fc1_input_topk_weights_buffer
        self.fc1_ready_counter = fc1_ready_counter
        self.token_src_metadata = token_src_metadata
        self.combine_output = combine_output
        self.combine_sf = combine_sf
        self.fc2_output_workspace = fc2_output_workspace
        self.fc2_output_sf = fc2_output_sf
        self.fc2_done_counter = fc2_done_counter
        self.token_back_schedule_counter = token_back_schedule_counter
        self.dispatch_route_extent = dispatch_route_extent
        self.nvlink_barrier_signal = nvlink_barrier_signal
        self.nvlink_barrier_counter = nvlink_barrier_counter
        self.internode_barrier_counter = internode_barrier_counter
        self.combine_issue_counter = combine_issue_counter
        self.combine_phase_snapshot = combine_phase_snapshot
        self.tail_combine_output = tail_combine_output
        self.tail_internode_combine = tail_internode_combine
        self.tail_reduced_output = tail_reduced_output
        self.pipeline_timing = pipeline_timing
        self.combine_timing_landmarks = combine_timing_landmarks
        self.fc_tile_timeline = fc_tile_timeline
        self.gin_dev_comm = gin_dev_comm
        self.gin_window = gin_window
        self.grid_sync_counter = grid_sync_counter
        self.local_zero_prefix = local_zero_prefix
        self.shared_zero_prefix = shared_zero_prefix
        self.peer_rank_ptr_mapper = peer_rank_ptr_mapper
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_total_experts = num_total_experts
        self.num_experts_per_rank = num_experts_per_rank
        self.num_topk = num_topk
        self.hidden_bytes = hidden_bytes
        self.sf_uint32_per_token = sf_uint32_per_token
        # The single-node kernels (moe_bf16_glu, moe_hopper_*, moe_mxfp8_*) build these args
        # without ranks_per_node: 0 means the NVLink domain is the whole job, as in TokenComm.
        if ranks_per_node == 0:
            ranks_per_node = world_size
        if ranks_per_node <= 0 or world_size % ranks_per_node != 0:
            raise ValueError(
                f"ranks_per_node must be positive and divide world_size ({world_size}), got {ranks_per_node}."
            )
        self.ranks_per_node = ranks_per_node
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.pool_token_capacity = pool_token_capacity
        self.pool_sf_capacity = pool_sf_capacity
        self.sm_count = sm_count
        self.system_scope_done_flags = system_scope_done_flags

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for name in _MLIR_VALUE_FIELDS:
            attr = getattr(self, name)
            if attr is None:
                continue
            values.extend(extract_mlir_values(attr))
        return values

    def __extract_mlir_attributes__(self) -> List[Any]:
        # Mirror __extract_mlir_values__ 1:1 so per-arg attrs stay aligned; the
        # only non-empty entry is peer_rank_ptr_mapper's byval/grid_constant.
        attrs: List[Any] = []
        for name in _MLIR_VALUE_FIELDS:
            attr = getattr(self, name)
            if attr is None:
                continue
            attrs.extend(extract_mlir_attributes(attr))
        return attrs

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "TokenCommArgs":
        idx = 0
        rebuilt: Dict[str, Any] = {}
        for name in _MLIR_VALUE_FIELDS:
            proto = getattr(self, name)
            if proto is None:
                rebuilt[name] = None
                continue
            n = len(extract_mlir_values(proto))
            rebuilt[name] = new_from_mlir_values(proto, values[idx : idx + n])
            idx += n
        assert idx == len(values), (
            f"TokenCommArgs serialization mismatch: "
            f"consumed={idx} provided={len(values)}"
        )
        const_kwargs = {name: getattr(self, name) for name in _CONST_FIELDS}
        return TokenCommArgs(**rebuilt, **const_kwargs)

class TokenInPullTokenBackPush:
    """Current implementation: token-in pull, token-back push."""

    num_dispatch_warps: int = 4
    warp_threads: int = 32
    num_dispatch_threads: int = num_dispatch_warps * warp_threads
    dispatch_intra_cta_bar_id: int = 10
    kernel_tail_named_barrier_id: int = 8
    dispatch_to_sched_named_barrier_id: int = 9
    # Relay-only rendezvous of the early Stage 1 push. It runs while the
    # dispatch warps already wait in inter_meta_exchange on barrier 10 with
    # the full 256-thread group, so it must not share that id (ids 1-7 belong
    # to the GEMM epilogue, 8 and 9 to the tail and scheduler hand-offs).
    relay_early_push_bar_id: int = 11
    # Dispatch-only dispatch_barrier (early_stage1): the
    # four dispatch warps run the barrier and its grid syncs as a 128-thread
    # group while the relay warps of the same CTA already pack Stage 1 and use
    # barrier 11, so the reduced group needs its own id (ids 12-15 are free).
    dispatch_only_bar_id: int = 12
    # NCCL device coop warp spans synchronize with hardware named barrier
    # 1 + span id (ncclCoopWarpSpan::sync, coop.h), so span ids 0/1/2 of the
    # relay collectives alias the GEMM barriers 1/2/3 (epilogue sync, TMEM
    # alloc/dealloc); they never overlap those in time today. The diag_stage1_flush
    # diagnostic flush runs while the epilogue may already sync on barrier 1,
    # so it uses span id 12 (barrier 13, unused elsewhere).
    relay_flush_coop_id: int = 12
    # Ring depth in 2 KB rounds per relay warp of the combine deep-prefetch ring;
    # the kernel sets it from the AB stage buffer size (0 disables the ring).
    combine_wide_ring_rounds: int = 0
    # dispatch_to_sched / kernel_tail thread counts are per-instance (see __init__).
    experts_per_dispatch_pass: int = num_dispatch_threads
    # Developer-only knob of the atomic_counter token-back schedule; set per
    # instance from ``MegaMoEConfig.token_back_atomic_batch``.
    token_back_atomic_batch: int = 1
    # ``clock64`` counts SM cycles.  120G cycles is at least 60 seconds even
    # at 2GHz: far longer than a healthy rail barrier, but short enough to turn
    # a lost peer into an actionable CUDA failure instead of an unbounded hang.
    rail_barrier_timeout_cycles: int = 120_000_000_000
    # ncclGinSignalOp_t values from gin_device_host_common.h. Signal strength
    # is a separate descriptor bit; nccl4py 0.3.1 does not expose that bit.
    gin_signal_op_inc: int = 0
    gin_signal_op_add: int = 1
    nvlink_barrier_cycle: int = 4
    nvlink_barriers_intranode_only: int = 3
    nvlink_barriers_lsa_relay: int = 4
    nvlink_barriers_gin: int = 5

    def _device_gin_gpu(self):
        """Rewrap the entry-materialized GPU-shared GIN pointer locally."""
        from nccl.core.device.cute.gin import Gin

        return Gin(ptr=self._gin_gpu_ptr)

    def _device_gin_thread(self):
        """Rewrap the entry-materialized thread-scoped GIN pointer locally."""
        from nccl.core.device.cute.gin import Gin

        return Gin(ptr=self._gin_thread_ptr)

    def __init__(
        self,
        *,
        world_size: int,
        num_topk: int,
        num_experts_per_rank: int,
        num_total_experts: int,
        hidden: int,
        fc1_token_dtype,
        sf_uint32_per_token: int,
        token_padding_block: int,
        sf_padding_block: int,
        cluster_tile_tokens: int,
        cluster_shape_mn,
        dispatch_warp_start: int,
        num_other_warps: int,
        ranks_per_node: int = 0,
        combine_format: "CombineFormat" = None,
        token_back_by_dispatch: bool = False,
        fc2_publishes_per_token_cluster_tile: int = 0,
        token_back_reduce_topk: bool = False,
        token_back_standalone: bool = False,
        num_relay_warps: int = 0,
        flag_batch: int = 1,
        is_swap_ab: bool = False,
        sf_atom_swizzled: bool = True,
        token_back_schedule_mode: Literal["static", "atomic_counter"] = "static",
        enable_nccl_gin: bool = False,
        publish_relay_counts_to_gemm: bool = False,
        apply_topk_in_fc1: bool = True,
        max_relay_tokens_per_node: int = 0,
        diagnostics: Optional[MegaMoEDiagnostics] = None,
        token_back_atomic_batch: int = 1,
        early_stage1: bool = False,
        direct_l1_sf_staging: bool = False,
        active_dispatch_warps: int = 4,
        compact_pull_buffer: bool = False,
        # Tail-split pair tasks: an expert with an odd CTA token-tile count publishes
        # fc2_done this many times for its last cluster tile (0 = off);
        # cta_tile_tokens (cluster_tile_tokens / 2) detects the odd count.
        fc2_publishes_per_split_tail_tile: int = 0,
        cta_tile_tokens: Optional[int] = None,
        # Launch-order-independent dispatch slots (see
        # ``_dispatch_prep_deterministic``).  Off by default: the atomic path
        # is unchanged for every existing kernel.
        deterministic_slots: bool = False,
    ) -> None:
        self.world_size = world_size
        self.num_topk = num_topk
        self.num_experts_per_rank = num_experts_per_rank
        self.num_total_experts = num_total_experts
        if ranks_per_node == 0:
            ranks_per_node = world_size
        if ranks_per_node <= 0:
            raise ValueError(
                f"ranks_per_node must be positive, got {ranks_per_node}."
            )
        if world_size % ranks_per_node != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"ranks_per_node ({ranks_per_node})."
            )
        self.ranks_per_node = ranks_per_node
        self.num_nodes = world_size // ranks_per_node
        self.num_remote_nodes = max(0, self.num_nodes - 1)
        self.num_remote_node_slots = max(1, self.num_remote_nodes)
        self.node_experts = self.ranks_per_node * self.num_experts_per_rank
        if combine_format is None:
            combine_format = CombineFormat(
                act_dtype=cutlass.BFloat16, scale_dtype=None, scale_block=None,
            )
        self.combine_format = combine_format
        # Step 5c layout by the 32 lanes of SM0 warp 0 for every rank size: the
        # experts are processed in chunks of 32 lanes with a warp-uniform carry
        # (the single-thread form for > 32 experts was removed 2026-10-22).
        self.meta_layout_lane_parallel = True
        # Diagnostics are plain instance attributes so ``cutlass.const_expr``
        # folds every probe branch at trace time (a property would not fold).
        if diagnostics is None:
            diagnostics = MegaMoEDiagnostics()
        self.diagnostics = diagnostics
        self.pipeline_timing_enabled = bool(diagnostics.pipeline_timing)
        self.iket_enabled = bool(diagnostics.iket_enabled)
        # Range emission follows the compiler option: without ``iket`` in CUTE_DSL_COMPILER_OPT the
        # ranges would be erased anyway; not emitting them keeps the single-node IR equal to main's.
        self.iket_detailed_ranges = bool(diagnostics.iket_detailed_ranges) and self.iket_enabled
        self.diag_fanout_skip = int(diagnostics.diag_fanout_skip)
        self.diag_fanout_skip_sf = bool(diagnostics.diag_fanout_skip_sf)
        self.diag_fanout_skip_data = bool(diagnostics.diag_fanout_skip_data)
        self.fc2_done_gpu_scope = bool(diagnostics.fc2_done_gpu_scope)
        self.use_ncu = bool(diagnostics.use_ncu)
        if int(token_back_atomic_batch) < 1:
            raise ValueError("token_back_atomic_batch must be positive")
        self.token_back_atomic_batch = int(token_back_atomic_batch)
        # Dispatch candidates of the 2026-09-21 campaign, opt-in through
        # MegaMoEConfig.experimental (each alone measured negative:
        # 3118225/3118432/3118511/3118512; together -3.8 us median at N2R2
        # 3118513, -6.4 us at N4R8 3118514; with staged scale factors on top
        # -10.1 / -11.2 us 3118895/3118896; kept opt-in by the 2026-09-21
        # review). early_stage1 bundles three mechanisms: the relay warps
        # pack Stage 1 right after their own dispatch_prep and skip
        # dispatch_barrier, message m of slot s is packed by relay warp
        # (prefix_s + m) mod W over the whole grid, and the Stage 1 issuer
        # lane posts the Step 3 metadata row right before its range put on
        # the same queue pair. The device code keeps one name per mechanism.
        if early_stage1 and not enable_nccl_gin:
            raise ValueError("early_stage1 requires enable_nccl_gin")
        self.gin_early_stage1 = bool(early_stage1)
        self.relay_skip_dispatch_barrier = self.gin_early_stage1
        self.gin_pack_flat_assign = self.gin_early_stage1
        self.step3_put_before_stage1 = self.gin_early_stage1
        # Stage 1 completion probe (MegaMoEDiagnostics.diag_stage1_flush):
        # SM0 relay warps flush the GIN context after the range puts so the
        # push_flush sub-envelope measures post-to-landed time; timing
        # perturbation only, not a performance candidate.
        self.diag_stage1_flush = bool(diagnostics.diag_stage1_flush)
        # The Stage 1 relay pack copies token and scale-factor rows with
        # 16-byte vector loads; only the NCCL GIN dispatch path packs Stage 1
        # messages, so the single-node kernels (e.g. MXFP8 with a 56-byte
        # scale-factor row) are not constrained by it.
        if enable_nccl_gin and (
            (hidden * int(fc1_token_dtype.width) // 8) % 16 != 0
            or (sf_uint32_per_token * 4) % 16 != 0
        ):
            raise NotImplementedError(
                "the Stage 1 early payload load needs 16-byte multiples for the "
                "token row and the scale-factor row"
            )
        if enable_nccl_gin and num_topk != 4:
            raise NotImplementedError(
                "the vectorized NCCL GIN relay wire currently requires "
                f"num_topk == 4; got num_topk={num_topk}."
            )
        if enable_nccl_gin and combine_format.name != "bf16":
            raise NotImplementedError(
                "the Stage 4 NCCL GIN combine currently supports BF16 only; "
                f"got {combine_format}."
            )
        gin_combine_elements = _GIN_COMBINE_VECTOR_BYTES // 2
        if enable_nccl_gin and hidden % gin_combine_elements != 0:
            raise ValueError(
                "the Stage 4 NCCL GIN BF16 gather requires hidden % "
                f"{gin_combine_elements} == 0 for naturally aligned packed "
                f"{_GIN_COMBINE_VECTOR_BYTES * 8}-bit rows; got hidden={hidden}."
            )
        self.gin_send_count_stride = (
            self.node_experts + 1 if enable_nccl_gin else self.node_experts
        )
        self.max_relay_tokens_per_node = max_relay_tokens_per_node
        if enable_nccl_gin:
            if max_relay_tokens_per_node <= 0:
                raise ValueError(
                    "enable_nccl_gin requires max_relay_tokens_per_node > 0"
                )
            gin_layout = build_megamoe_gin_window_layout(
                world_size=world_size,
                ranks_per_node=self.ranks_per_node,
                num_experts_per_rank=num_experts_per_rank,
                max_relay_tokens_per_node=max_relay_tokens_per_node,
                hidden_bytes=hidden * int(fc1_token_dtype.width) // 8,
                sf_uint32_per_token=sf_uint32_per_token,
                num_topk=num_topk,
                combine_hidden=hidden,
            )
            assert gin_layout.relay_message is not None
            assert gin_layout.relay_send_buffer_offset is not None
            assert gin_layout.relay_recv_buffer_offset is not None
            self.gin_recv_splits_offset_i32 = (
                gin_layout.internode_recv_splits_offset // 4
            )
            self.gin_metadata_window_i32 = gin_layout.total_bytes // 4
            self.gin_window_bytes = gin_layout.total_bytes
            self.relay_send_buffer_offset = gin_layout.relay_send_buffer_offset
            self.relay_recv_buffer_offset = gin_layout.relay_recv_buffer_offset
            self.relay_message_src_metadata_offset = (
                gin_layout.relay_message.src_metadata_offset
            )
            self.relay_message_route_words_offset = (
                gin_layout.relay_message.route_words_offset
            )
            self.relay_message_topk_weights_offset = (
                gin_layout.relay_message.topk_weights_offset
            )
            self.relay_message_token_data_offset = (
                gin_layout.relay_message.token_data_offset
            )
            self.relay_message_sf_data_offset = (
                gin_layout.relay_message.sf_data_offset
            )
            self.relay_message_bytes = gin_layout.relay_message.message_bytes
            dispatch_vector_fields = (
                self.relay_message_token_data_offset,
                self.relay_message_sf_data_offset,
                hidden * int(fc1_token_dtype.width) // 8,
                sf_uint32_per_token * 4,
                self.relay_message_bytes,
            )
            if any(value % 16 != 0 for value in dispatch_vector_fields):
                raise ValueError(
                    "the warp-pack Stage 1 requires 16-byte "
                    "aligned DATA/SF offsets, sizes, and message stride; "
                    f"got {dispatch_vector_fields}"
                )
            assert gin_layout.intra_node_reduce_buf_offset is not None
            assert gin_layout.internode_combine_buffer_offset is not None
            self.intra_node_reduce_buf_offset = (
                gin_layout.intra_node_reduce_buf_offset
            )
            self.internode_combine_buffer_offset = (
                gin_layout.internode_combine_buffer_offset
            )
            self.internode_splits_signal_base = (
                gin_layout.internode_splits_signal_base
            )
            self.internode_barrier_signal_base = (
                gin_layout.internode_barrier_signal_base
            )
            self.relay_node_signal_base = gin_layout.relay_node_signal_base
            self.icombine_arrival_signal_base = (
                gin_layout.icombine_arrival_signal_base
            )
        else:
            gin_send_count_bytes = (
                self.num_remote_node_slots * self.node_experts * 4
            )
            self.gin_recv_splits_offset_i32 = (
                (gin_send_count_bytes + 127) // 128 * 128 // 4
            )
            self.gin_metadata_window_i32 = (
                self.gin_recv_splits_offset_i32
                + self.num_remote_node_slots * self.node_experts
            )
            self.gin_window_bytes = self.gin_metadata_window_i32 * 4
            self.relay_send_buffer_offset = 0
            self.relay_recv_buffer_offset = 0
            self.relay_message_src_metadata_offset = 0
            self.relay_message_route_words_offset = 0
            self.relay_message_topk_weights_offset = 0
            self.relay_message_token_data_offset = 0
            self.relay_message_sf_data_offset = 0
            self.relay_message_bytes = 0
            self.intra_node_reduce_buf_offset = 0
            self.internode_combine_buffer_offset = 0
            self.internode_splits_signal_base = 0
            self.internode_barrier_signal_base = self.num_remote_node_slots
            self.relay_node_signal_base = self.internode_barrier_signal_base + 2
            self.icombine_arrival_signal_base = self.relay_node_signal_base
        self.deterministic_slots = False
        if deterministic_slots:
            raise ValueError("Deterministic training slots are outside the FI GIN port.")
        self.hidden = hidden
        # The packed combine reduce walks the row in 512-byte chunks (one 16-byte vector per lane);
        # a row that is not a multiple of 256 BF16 elements ends in a partial chunk whose lanes past
        # the row are predicated off (see pack_tail_lanes), so hidden only needs the 16-byte row
        # alignment checked above (Hopper BF16 backward runs hidden=192).
        self.fc1_token_dtype = fc1_token_dtype
        self.hidden_bytes = hidden * int(fc1_token_dtype.width) // 8
        self.sf_uint32_per_token = sf_uint32_per_token
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.sf_atom_swizzled = sf_atom_swizzled
        self.cluster_tile_tokens = cluster_tile_tokens
        self.cluster_shape_mn = cluster_shape_mn
        if flag_batch < 1 or flag_batch > 32:
            raise ValueError(f"flag_batch must be in [1, 32], got {flag_batch}.")
        # Release-flag batch size consumed by dispatch_pull as a Python int.
        # One warp lane carries one delayed release target.
        self._flag_batch = flag_batch
        self.is_swap_ab = is_swap_ab

        if token_back_schedule_mode not in ("static", "atomic_counter"):
            raise ValueError(
                "token_back_schedule_mode must be 'static' or "
                f"'atomic_counter'; got {token_back_schedule_mode!r}."
            )
        self.token_back_schedule_mode = token_back_schedule_mode
        self.enable_nccl_gin = enable_nccl_gin
        self.publish_relay_counts_to_gemm = publish_relay_counts_to_gemm
        # Staged scale factors for the direct fan-out
        # (MegaMoEConfig.experimental.direct_l1_sf_staging): the relay peer
        # writes the SF row with 16-byte vectors into a per-row staging area
        # on the owner and raises a per-row flag; the owner's dispatch warps
        # scatter the row into the swizzled atoms and release the same static
        # readiness credit at GPU scope (I1: the SF plane's writer becomes the
        # owner; I7, I10 unchanged). Skip diagnostics 3052131/3052132
        # attributed about 14 us of the busiest relay warp to the scalar SF
        # stores at N4R8.
        if direct_l1_sf_staging and not enable_nccl_gin:
            raise ValueError("direct_l1_sf_staging requires enable_nccl_gin")
        self.direct_l1_sf_staging = bool(direct_l1_sf_staging)
        if self.direct_l1_sf_staging and (
            (hidden * int(fc1_token_dtype.width) // 8) % 16 != 0
            or (sf_uint32_per_token * 4) % 16 != 0
        ):
            raise NotImplementedError(
                "direct_l1_sf_staging needs 16-byte multiples for the "
                "token row and the scale-factor row"
            )
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        # Optional host-configured TopkReduce object. The megakernel supplies it
        # after construction, avoiding a token_comm <-> topk_reduce import cycle.
        self.tail_reducer = None
        if self.publish_relay_counts_to_gemm and not self.enable_nccl_gin:
            raise ValueError(
                "publish_relay_counts_to_gemm requires enable_nccl_gin"
            )
        # How many of the (warpgroup-aligned) dispatch warps do token-comm
        # work AT ALL (prep + barrier + pull + reuse token-back).  The
        # physical warp count stays num_dispatch_warps -- the setmaxnreg
        # register reallocation is warpgroup(4)-granular -- but the warps
        # beyond active_dispatch_warps skip the whole dispatch body and only
        # rejoin at kernel_tail, leaving them fully idle for future use.
        # Sizing: sustained pull bandwidth ~= active_warps * SMs *
        # hidden_bytes / read-RTT (one row in flight per warp); on H200
        # (450 GB/s unidirectional) even 1 warp/SM (~132 rows in flight)
        # exceeds the bandwidth-delay product, and measured EP4/EP8 sweeps
        # rank 1 > 2 > 4 warps -- the shallowest NVLink read queue wins
        # (a deep queue lengthens per-request RTT and fights the GEMM for
        # L2/issue slots).  Deeper per-warp pipelining (donating idle
        # warps' smem slots as extra in-flight stages) was measured at
        # -0.8%..+0.1% net -- in-flight depth is not the bottleneck.
        if active_dispatch_warps not in (1, 2, self.num_dispatch_warps):
            raise ValueError(
                "active_dispatch_warps must be 1, 2, or "
                f"{self.num_dispatch_warps}; got {active_dispatch_warps}."
            )
        self.active_dispatch_warps = active_dispatch_warps
        self.active_dispatch_threads = active_dispatch_warps * self.warp_threads
        # SMEM Int32 slots of the expert-count scratch: one row per active
        # dispatch warp under deterministic slots (private per-warp cursors),
        # a single CTA-wide row otherwise.
        self.smem_expert_count_slots = max(num_total_experts, self.num_dispatch_warps)
        # SMEM pull-buffer slots.  By default every physical dispatch warp
        # owns a hidden_bytes slot even when it is idle; with
        # compact_pull_buffer only the active warps get a slot (the idle warps
        # never enter dispatch_warp_body, and the reuse token-back walkers are
        # exactly the active warps and index the same slots with
        # tb_chunk_bytes strides).  The three idle slots cost 3 * hidden_bytes
        # (21 KiB at hidden=7168 fp8) of AB-stage SMEM per CTA.
        self.compact_pull_buffer = bool(compact_pull_buffer)
        self.dispatch_warp_start = dispatch_warp_start
        # Warps that share this CTA with the dispatch group but are not part
        # of it. They participate in kernel-tail / dispatch-with-other
        # rendezvous and determine `number_of_threads` for those barriers.
        # Pure standalone dispatch passes 0 (no cohabitants -> barriers
        # collapse to dispatch-only).
        self.num_other_warps = num_other_warps
        self.num_other_threads = num_other_warps * self.warp_threads

        self.token_back_by_dispatch = token_back_by_dispatch
        self.push_data = token_back_by_dispatch
        self.push_sf = combine_format.is_quantized

        if num_relay_warps < 0:
            raise ValueError(f"num_relay_warps must be non-negative, got {num_relay_warps}.")
        if token_back_standalone and num_relay_warps:
            raise ValueError(
                "token_back_standalone and relay warps both want warps 12-15; "
                "use token_back_mode='reuse_dispatch_warps' or 'epi_warps' "
                "when reserving relay warps."
            )
        self.num_relay_warps = num_relay_warps
        self.num_relay_threads = self.num_relay_warps * self.warp_threads
        # Upstream ``active_dispatch_warps`` (a subset of the physical
        # dispatch warps does the token-comm work) is only wired through the
        # intranode dispatch group here: the relay-warp, NCCL GIN and
        # multi-node bodies still partition by the physical warp count.
        if self.active_dispatch_warps != self.num_dispatch_warps and (
            self.num_relay_warps > 0
            or self.enable_nccl_gin
            or self.num_remote_nodes > 0
        ):
            raise NotImplementedError(
                "active_dispatch_warps < num_dispatch_warps is only supported "
                "by the intranode dispatch group (no relay warps, no NCCL GIN, "
                "single node); got "
                f"active_dispatch_warps={self.active_dispatch_warps}, "
                f"num_relay_warps={self.num_relay_warps}, "
                f"enable_nccl_gin={self.enable_nccl_gin}, "
                f"num_remote_nodes={self.num_remote_nodes}."
            )
        # Synchronization participation only: dispatch_group_threads includes
        # both the active dispatch warps and the relay warps.  A
        # data-partition stride may use it only when both bodies actually
        # access that data (for example the inter_meta_exchange snapshots);
        # dispatch-only data must stride by the dispatch writer count.
        self.dispatch_group_warps = (
            self.active_dispatch_warps + self.num_relay_warps
        )
        self.dispatch_group_threads = self.dispatch_group_warps * self.warp_threads
        # Expert-range passes (dispatch_prep / dispatch_barrier) stride by
        # the active dispatch threads; the class-level default assumes all
        # physical dispatch warps are active.
        self.experts_per_dispatch_pass = self.active_dispatch_threads
        self.relay_warp_start = dispatch_warp_start + self.num_dispatch_warps

        # The path count is a compile-time contract, not a prose convention:
        # dispatch + two tail barriers are common (3); relay metadata layout
        # publication adds one (4); GIN split redistribution adds one more (5).
        # Nsight Compute kernel replay cannot restore peer signal memory, so the
        # selected path is padded to a complete four-call self-cancelling cycle.
        if self.enable_nccl_gin:
            self.nvlink_barriers_per_launch = self.nvlink_barriers_gin
        elif self.num_relay_warps > 0 and self.num_remote_nodes > 0:
            self.nvlink_barriers_per_launch = self.nvlink_barriers_lsa_relay
        else:
            self.nvlink_barriers_per_launch = self.nvlink_barriers_intranode_only
        self.nvlink_ncu_padding = (
            -self.nvlink_barriers_per_launch
        ) % self.nvlink_barrier_cycle
        assert (
            self.nvlink_barriers_per_launch + self.nvlink_ncu_padding
        ) % self.nvlink_barrier_cycle == 0
        assert self.nvlink_ncu_padding in (0, 1, 3)

        # Standalone token-back: a dedicated warpgroup (size == dispatch group)
        self.token_back_standalone = token_back_standalone
        self.num_token_back_warps = self.num_dispatch_warps if self.token_back_standalone else 0
        self.num_token_back_threads = self.num_token_back_warps * self.warp_threads
        self.token_back_warp_start = dispatch_warp_start + self.num_dispatch_warps
        # Standalone token-back per-warp pull buffer; token is moved in
        # tb_chunk_bytes pieces (last piece carries the remainder), so this is
        # independent of hidden.
        self.tb_chunk_bytes = 2048

        self.num_total_threads = (
            self.num_dispatch_threads
            + self.num_other_threads
            + self.num_relay_threads
            + self.num_token_back_threads
        )
        self.dispatch_to_sched_threads = (
            self.dispatch_group_warps + 1 + self.num_token_back_warps
        ) * self.warp_threads
        self.kernel_tail_threads = self.num_total_threads

        # The DATA wire dtype is the combine act dtype (bf16 baseline -> bf16;
        # fp4/e4m3 quantized), NOT the kernel's fc2 output dtype: the cross-rank
        # payload is what the receiver dequantizes.
        self.fc2_output_dtype = combine_format.act_dtype
        if token_back_reduce_topk:
            if not token_back_by_dispatch:
                raise ValueError(
                    "token_back_reduce_topk=True requires the dispatch "
                    "token-back DATA path (token_back_by_dispatch)."
                )
            if combine_format.act_dtype is not cutlass.BFloat16:
                raise NotImplementedError(
                    "token_back_reduce_topk currently supports a bf16 combine "
                    f"only, got {combine_format}."
                )
        self.token_back_reduce_topk = token_back_reduce_topk
        if self.enable_token_back:
            self.fc2_token_bytes = hidden * int(combine_format.act_dtype.width) // 8
            if self.fc2_token_bytes % self.hidden_bytes != 0:
                raise ValueError(
                    f"fc2_token_bytes={self.fc2_token_bytes} must be a "
                    f"multiple of hidden_bytes={self.hidden_bytes} so the "
                    f"per-warp pull buffer can be reused chunk-by-chunk."
                )
            self.fc2_num_chunks = self.fc2_token_bytes // self.hidden_bytes
            if fc2_publishes_per_token_cluster_tile <= 0:
                raise ValueError(
                    "fc2_publishes_per_token_cluster_tile must be > 0 when "
                    "token-back is enabled (it gates the per-expert push)."
                )
            self.fc2_publishes_per_token_cluster_tile = (
                fc2_publishes_per_token_cluster_tile
            )
            if fc2_publishes_per_split_tail_tile < 0:
                raise ValueError(
                    "fc2_publishes_per_split_tail_tile must be >= 0, got "
                    f"{fc2_publishes_per_split_tail_tile}."
                )
            if fc2_publishes_per_split_tail_tile > 0:
                if (
                    cta_tile_tokens is None
                    or cta_tile_tokens * 2 != cluster_tile_tokens
                ):
                    raise ValueError(
                        "fc2_publishes_per_split_tail_tile > 0 requires "
                        "cta_tile_tokens * 2 == cluster_tile_tokens; got "
                        f"cta_tile_tokens={cta_tile_tokens}, "
                        f"cluster_tile_tokens={cluster_tile_tokens}."
                    )
                if (
                    fc2_publishes_per_split_tail_tile
                    > fc2_publishes_per_token_cluster_tile
                ):
                    raise ValueError(
                        "fc2_publishes_per_split_tail_tile "
                        f"({fc2_publishes_per_split_tail_tile}) cannot exceed "
                        "fc2_publishes_per_token_cluster_tile "
                        f"({fc2_publishes_per_token_cluster_tile})."
                    )
            self.fc2_publishes_per_split_tail_tile = (
                fc2_publishes_per_split_tail_tile
            )
            self.cta_tile_tokens = (
                cta_tile_tokens if fc2_publishes_per_split_tail_tile > 0 else None
            )
        else:
            self.fc2_token_bytes = 0
            self.fc2_num_chunks = 0
            self.fc2_publishes_per_token_cluster_tile = 0
            self.fc2_publishes_per_split_tail_tile = 0
            self.cta_tile_tokens = None

    @property
    def enable_token_back(self) -> bool:
        # token-back warps run if they push the DATA plane, the SF plane, or both.
        return self.push_data or self.push_sf

    def pull_buffer_bytes(self) -> int:
        """Bytes of the dispatch pull buffer in ``extra_smem_storage_class``."""
        if self.compact_pull_buffer:
            # Each active warp needs one token row for dispatch and one
            # tb_chunk_bytes piece for the reuse token-back path.
            return self.active_dispatch_warps * max(
                self.hidden_bytes, self.tb_chunk_bytes
            )
        return self.num_dispatch_warps * self.hidden_bytes

    def extra_smem_storage_class(self) -> type:
        pull_buffer_bytes = self.pull_buffer_bytes()
        num_total_experts = self.num_total_experts
        smem_count_slots = max(num_total_experts, self.num_dispatch_warps)
        inter_send_count_entries = self.num_remote_node_slots * self.node_experts
        inter_send_token_entries = self.num_remote_node_slots

        if self.num_relay_warps:
            # Internode relay uses one CTA-level TokenCommStorage schema shared
            # by both warp groups.  Warps 8-11 keep using the intranode fields
            # (`smem_expert_count`, `pull_mbar`, `pull_buffer`) for dispatch
            # prep/pull and token-back. Relay warps 12-15 use
            # `smem_inter_send_count`, `smem_inter_token_count`, and
            # `smem_inter_token_base` for remote-node dispatch metadata:
            #   - smem_inter_send_count[(N-1) * (R*E)] int32: per-CTA histogram
            #     of topk hits by remote node and node-local expert id.
            #   - smem_inter_token_count[N-1] int32: first the per-CTA
            #     deduplicated token count per remote node, then the CTA's
            #     reserved base slot after the packed global atomic.
            # GIN additionally gives every relay lane four private 16B combine
            # slots, one per top-k hit. This cannot alias `pull_buffer`: after
            # the dispatch-to-scheduler named-barrier arrive, dispatch warps may
            # use that buffer for token-back while relay warps combine. Keeping
            # all vector candidates on the same 8KiB scratch schema also makes
            # their resource comparison allocation-identical.
            # These fields are intentionally colocated with the intranode
            # scratch in one struct because the base kernel exposes one
            # extra-SMEM object per CTA, not separate storage types per warp
            # group.
            if self.enable_nccl_gin:
                @cute.struct
                class TokenCommStorage:
                    pull_mbar: cute.struct.MemRange[
                        Int64, self.num_dispatch_warps
                    ]
                    smem_expert_count: cute.struct.MemRange[
                        Int32, num_total_experts
                    ]
                    smem_inter_send_count: cute.struct.MemRange[
                        Int32, inter_send_count_entries
                    ]
                    smem_inter_token_count: cute.struct.MemRange[
                        Int32, inter_send_token_entries
                    ]
                    smem_inter_token_base: cute.struct.MemRange[
                        Int32, inter_send_token_entries
                    ]
                    pull_buffer: cute.struct.Align[
                        cute.struct.MemRange[Uint8, pull_buffer_bytes], 16
                    ]
                    combine_b128_scratch: cute.struct.Align[
                        cute.struct.MemRange[
                            Uint8,
                            self.num_relay_warps
                            * self.warp_threads * self.num_topk * 16,
                        ],
                        16,
                    ]
                    # Set by epilogue warp 0 once the AB stage buffers are
                    # free; read by the relay warps' combine ring.
                    fc_smem_free: cute.struct.MemRange[Int32, 1]
                    # Direct fan-out CTA publish: per relay warp, the four
                    # unpublished ready addresses of the current round.
                    smem_fanout_ready: cute.struct.MemRange[
                        Int64, self.num_relay_warps * 4
                    ]

                return TokenCommStorage

            @cute.struct
            class TokenCommStorage:
                pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
                smem_expert_count: cute.struct.MemRange[
                    Int32, num_total_experts
                ]
                smem_inter_send_count: cute.struct.MemRange[
                    Int32, inter_send_count_entries
                ]
                smem_inter_token_count: cute.struct.MemRange[
                    Int32, inter_send_token_entries
                ]
                smem_inter_token_base: cute.struct.MemRange[
                    Int32, inter_send_token_entries
                ]
                pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, pull_buffer_bytes], 16]

            return TokenCommStorage

        smem_expert_count_slots = self.smem_expert_count_slots

        if self.token_back_standalone:
            @cute.struct
            class TokenCommStorage:
                pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
                smem_expert_count: cute.struct.MemRange[
                    Int32, smem_expert_count_slots
                ]
                pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, pull_buffer_bytes], 16]
                tb_pull_mbar: cute.struct.MemRange[Int64, self.num_token_back_warps]
                tb_pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, self.num_token_back_warps * self.tb_chunk_bytes], 16]

            return TokenCommStorage

        @cute.struct
        class TokenCommStorage:
            pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
            smem_expert_count: cute.struct.MemRange[
                Int32, smem_expert_count_slots
            ]
            pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, pull_buffer_bytes], 16]

        return TokenCommStorage

    def fc1_ready_counter_ptr(self, token_comm_args):
        return token_comm_args.fc1_ready_counter.iterator

    @cute.jit
    def _cta_linear_id(self):
        """Grid-wide CTA id: ``bidz * cluster_size + %cluster_ctarank``.

        TokenComm receives the launch-view cluster shape, not the scheduler's
        internally swapped shape. The grid is (cluster_m, cluster_n, clusters)
        for both operand orders, so bidy's multiplier is cluster_m. Matching
        the hardware x-fastest cluster-rank enumeration gives a bijection onto
        [0, sm_count) -- required by the dispatch work partition and the
        tail bulk-zero striding.
        """
        bidx, bidy, bidz = cute.arch.block_idx()
        bidx_extent = self.cluster_shape_mn[0]
        return (
            Int32(bidx)
            + Int32(bidx_extent) * Int32(bidy)
            + Int32(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
            * Int32(bidz)
        )

    @cute.jit
    def sched_warp_pre_init_wait(self, token_comm_args):
        nb = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb.arrive_and_wait()

    @cute.jit
    def fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        if cutlass.const_expr(self.is_swap_ab):
            token_cluster_size: cutlass.Constexpr = self.cluster_shape_mn[1]
            cluster_token_block_idx = (
                work_tile_info.tile_n_idx // cutlass.Int32(token_cluster_size)
            )
            counter_slot = (
                work_tile_info.cumulative_token_block_count
                + cluster_token_block_idx
            )
            if cutlass.const_expr(token_cluster_size == 1):
                # Preserve the existing NVFP4/MXFP8 single-token-CTA path.
                peek_threshold = work_tile_info.valid_tokens_in_cta_tile
            else:
                packed_expert_count = token_comm_args.expert_recv_count_sum[
                    work_tile_info.expert_idx
                ]
                expert_token_count = Int32(
                    Int64(packed_expert_count) & Int64(0xFFFFFFFF)
                )
                remaining_cluster_tokens = cutlass.max(
                    expert_token_count
                    - cluster_token_block_idx
                    * cutlass.Int32(self.cluster_tile_tokens),
                    Int32(0),
                )
                peek_threshold = cutlass.min(
                    remaining_cluster_tokens,
                    cutlass.Int32(self.cluster_tile_tokens),
                )
        else:
            counter_slot = (
                work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_m_idx // cutlass.Int32(self.cluster_shape_mn[0])
            )
            peek_threshold = work_tile_info.valid_tokens_in_cluster_tile

        counter_ptr = token_comm_args.fc1_ready_counter.iterator + counter_slot
        # Dispatch warps may fill rows within a cluster token block out of
        # order. Every valid CTA therefore waits for the full cluster-block
        # count; an invalid tail CTA has no rows to consume and skips the wait.
        if work_tile_info.valid_tokens_in_cta_tile > Int32(0):
            if not work_tile_info.peek_ready:
                _iket.range_push("tma_token_fc1_wait")
                spin_wait(
                    counter_ptr,
                    lambda v: v >= peek_threshold,
                    fail_sleep_cycles=1000,
                    scope="sys" if self.enable_nccl_gin else "gpu",
                )
                _iket.range_pop()

    @cute.jit
    def _find_dispatch_route_extent(
        self, token_comm_storage, topk_idx, publication, sm_idx, warp_idx, lane_idx,
        *, num_tokens,
    ):
        thread_idx = warp_idx * Int32(self.warp_threads) + lane_idx
        if sm_idx == Int32(0):
            last = Int32(0)
            route = thread_idx
            while route < Int32(num_tokens * self.num_topk):
                expert = topk_idx[route // self.num_topk, route % self.num_topk]
                if expert >= 0:
                    last = route + Int32(1)
                route += Int32(self.active_dispatch_threads)
            warp_last = cute.arch.warp_redux_sync(last, "max")
            # Histogram initialization starts only after the extent is ready.
            extents = token_comm_storage.smem_expert_count.data_ptr()
            if lane_idx == Int32(0):
                (extents + warp_idx).store(Int32(warp_last))
            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.active_dispatch_threads,
            )
            if thread_idx == Int32(0):
                extent = Int32(0)
                for warp in cutlass.range_constexpr(self.active_dispatch_warps):
                    value = (extents + warp).load()
                    if value > extent:
                        extent = value
                publication[0] = extent
                cute.arch.atomic_add(
                    publication.iterator + 1, Int32(1), sem="release", scope="gpu"
                )
        # Ready is reset with the local counters only after every warp role
        # drains. Extent is last-valid-route+1, so holes keep their route IDs.
        while (
            cute.arch.load(publication.iterator + 1, Int32, sem="acquire", scope="gpu")
            != Int32(1)
        ):
            pass
        return cute.arch.load(publication.iterator, Int32, sem="relaxed", scope="gpu")

    @cute.jit
    def dispatch_prep(
        self, token_comm_storage, topk_idx, expert_send_count, src_token_topk_idx,
        peer_rank_ptr_mapper, sm_idx, warp_idx, lane_idx,
        *, local_rank, num_tokens, num_sms, route_extent=None, cta_expert_hist=None, grid_sync_counter=None,
    ):
        if cutlass.const_expr(self.enable_nccl_gin):
            self._dispatch_prep_atomic(
                token_comm_storage, topk_idx, expert_send_count, src_token_topk_idx,
                peer_rank_ptr_mapper, sm_idx, warp_idx, lane_idx,
                local_rank=local_rank, num_tokens=num_tokens, num_sms=num_sms,
            )
        else:
            if cutlass.const_expr(route_extent is None):
                self._dispatch_prep_impl(
                    token_comm_storage, topk_idx, expert_send_count, src_token_topk_idx,
                    peer_rank_ptr_mapper, sm_idx, warp_idx, lane_idx,
                    local_rank=local_rank, num_tokens=num_tokens, num_sms=num_sms,
                    scan_tokens=num_tokens, active_ctas=num_sms,
                )
            else:
                extent = self._find_dispatch_route_extent(
                    token_comm_storage, topk_idx, route_extent, sm_idx, warp_idx, lane_idx,
                    num_tokens=num_tokens,
                )
                scan_tokens = (extent + Int32(self.num_topk - 1)) // Int32(self.num_topk)
                tokens_per_cta = (32 // self.num_topk) * self.active_dispatch_warps
                active_ctas = (
                    scan_tokens + Int32(tokens_per_cta - 1)
                ) // Int32(tokens_per_cta)
                active_ctas = cutlass.min(Int32(num_sms), cutlass.max(Int32(1), active_ctas))
                if sm_idx < active_ctas:
                    self._dispatch_prep_impl(
                        token_comm_storage, topk_idx, expert_send_count, src_token_topk_idx,
                        peer_rank_ptr_mapper, sm_idx, warp_idx, lane_idx,
                        local_rank=local_rank, num_tokens=num_tokens, num_sms=num_sms,
                        scan_tokens=scan_tokens, active_ctas=active_ctas,
                    )

    @cute.jit
    def _dispatch_prep_impl(
        self,
        token_comm_storage,
        topk_idx,
        expert_send_count,
        src_token_topk_idx,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_tokens,
        num_sms,
        scan_tokens,
        active_ctas,
    ):
        thread_idx_in_dispatch = Int32(warp_idx * self.warp_threads + lane_idx)
        smem_count_ptr = token_comm_storage.smem_expert_count.data_ptr()
        i = thread_idx_in_dispatch
        while i < Int32(self.num_total_experts):
            (smem_count_ptr + i).store(Int32(0))
            i = i + Int32(self.active_dispatch_threads)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.active_dispatch_threads,
        )

        tokens_per_warp: cutlass.Constexpr[int] = 32 // self.num_topk
        active_lanes: cutlass.Constexpr[int] = tokens_per_warp * self.num_topk
        num_dispatch_warps_per_grid = active_ctas * self.active_dispatch_warps

        base_token_for_warp = (
            sm_idx * self.active_dispatch_warps + warp_idx
        ) * tokens_per_warp
        grid_token_stride = num_dispatch_warps_per_grid * tokens_per_warp

        t = base_token_for_warp
        while t < scan_tokens:
            token_offset_in_warp = lane_idx // self.num_topk
            token_global = t + token_offset_in_warp
            if lane_idx < active_lanes and token_global < scan_tokens:
                topk_slot = lane_idx % self.num_topk
                expert_id = Int32(topk_idx[token_global, topk_slot])
                if expert_id >= Int32(0):
                    cute.arch.atomic_add(
                        smem_count_ptr + expert_id,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
            cute.arch.sync_warp()
            t += grid_token_stride

        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.active_dispatch_threads,
        )

        # The high word counts completion from the full launch grid. CTA0
        # accounts for skipped CTAs, including experts with zero live routes.
        completion_count = Int64(1) + Int64(sm_idx == Int32(0)) * Int64(
            num_sms - active_ctas
        )
        for offset in cutlass.range_constexpr(
            0, self.num_total_experts, self.active_dispatch_threads,
        ):
            expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
            if expert_id < Int32(self.num_total_experts):
                slot_ptr = smem_count_ptr + expert_id
                local_count = (slot_ptr).load()
                delta = (completion_count << Int64(32)) | (
                    Int64(local_count) & Int64(0xFFFFFFFF)
                )
                old_packed = cute.arch.atomic_add(
                    expert_send_count.iterator + expert_id,
                    delta,
                    sem="relaxed",
                    scope="gpu",
                )
                base_slot = Int32(old_packed & Int64(0xFFFFFFFF))
                (slot_ptr).store(base_slot)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.active_dispatch_threads,
        )

        t = base_token_for_warp
        while t < scan_tokens:
            token_offset_in_warp = lane_idx // self.num_topk
            token_global = t + token_offset_in_warp
            if lane_idx < active_lanes and token_global < scan_tokens:
                topk_slot = lane_idx % self.num_topk
                expert_id = Int32(topk_idx[token_global, topk_slot])
                if expert_id >= Int32(0):
                    dst_rank = expert_id // Int32(self.num_experts_per_rank)
                    local_expert = expert_id % Int32(self.num_experts_per_rank)
                    slot = cute.arch.atomic_add(
                        smem_count_ptr + expert_id,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    token_topk_word = Int32(token_global * self.num_topk + topk_slot)
                    MAX_SLOT_C: cutlass.Constexpr[int] = num_tokens * self.num_topk
                    elem_off = (
                        (local_expert * Int32(self.world_size) + Int32(local_rank))
                        * Int32(MAX_SLOT_C)
                        + slot
                    ) * Int32(4)
                    peer_addr = peer_rank_ptr_mapper.map(
                        src_token_topk_idx.iterator.toint(),
                        dst_rank, Int64(elem_off),
                    )
                    stg_b32_raw(peer_addr, token_topk_word)
            cute.arch.sync_warp()
            t += grid_token_stride


    @cute.jit
    def _dispatch_prep_atomic(
        self,
        token_comm_storage,
        topk_idx,
        expert_send_count,
        src_token_topk_idx,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_tokens,
        num_sms,
    ):
        thread_idx_in_dispatch_group = Int32(warp_idx * self.warp_threads + lane_idx)
        smem_count_ptr = token_comm_storage.smem_expert_count.data_ptr()
        # With no relay warps every warp of the dispatch group is a dispatch warp (main's shape);
        # the warp test exists only for the internode kernel's relay warps.
        dispatch_warps_only = True
        if cutlass.const_expr(self.num_relay_warps > 0):
            dispatch_warps_only = warp_idx < Int32(self.num_dispatch_warps)
        if cutlass.const_expr(self.enable_nccl_gin):
            # Resolved outside the dynamic branch below: the DSL captures
            # every name a traced branch body mentions, and the shared-memory
            # struct itself cannot be captured.
            fc_smem_free_addr = Int32(
                token_comm_storage.fc_smem_free.data_ptr().toint()
            )
        if dispatch_warps_only:
            i = thread_idx_in_dispatch_group
            while i < Int32(self.num_total_experts):
                (smem_count_ptr + i).store(Int32(0))
                # Only the dispatch warps execute this loop.  Relay warps meet
                # them at the barrier below, but initialize their own shared
                # arrays in ``internode_dispatch_prep``.  Striding by the full
                # dispatch+relay group therefore leaves one expert range stale
                # whenever relay warps are enabled (128..255 for four dispatch
                # plus four relay warps).
                # ``active_dispatch_threads`` is the writer count of this
                # loop (upstream ``active_dispatch_warps``; all four
                # dispatch warps are active in the MegaMoE kernel).
                i = i + Int32(self.active_dispatch_threads)
            if cutlass.const_expr(self.enable_nccl_gin):
                # The combine ring flag lives in uninitialized shared memory;
                # clear it before the group barrier so both the relay warps'
                # acquire reads and epilogue warp 0's release store (which
                # depends on dispatch through the FC1 readiness protocol)
                # follow this store.
                if thread_idx_in_dispatch_group == Int32(0):
                    st_shared_b32_raw(fc_smem_free_addr, Int32(0))
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.dispatch_group_threads,
        )

        tokens_per_warp: cutlass.Constexpr[int] = 32 // self.num_topk
        active_lanes: cutlass.Constexpr[int] = tokens_per_warp * self.num_topk
        num_dispatch_warps_per_grid: cutlass.Constexpr[int] = (
            num_sms * self.active_dispatch_warps
        )

        base_token_for_warp = (
            sm_idx * self.active_dispatch_warps + warp_idx
        ) * tokens_per_warp
        grid_token_stride = num_dispatch_warps_per_grid * tokens_per_warp

        if dispatch_warps_only:
            t = base_token_for_warp
            while t < num_tokens:
                token_offset_in_warp = lane_idx // self.num_topk
                token_global = t + token_offset_in_warp
                if lane_idx < active_lanes and token_global < num_tokens:
                    topk_slot = lane_idx % self.num_topk
                    expert_id = Int32(topk_idx[token_global, topk_slot])
                    if expert_id >= Int32(0):
                        cute.arch.atomic_add(
                            smem_count_ptr + expert_id,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                cute.arch.sync_warp()
                t += grid_token_stride

        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.dispatch_group_threads,
        )

        if dispatch_warps_only:
            for offset in cutlass.range_constexpr(
                0, self.num_total_experts, self.experts_per_dispatch_pass,
            ):
                expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
                if expert_id < Int32(self.num_total_experts):
                    slot_ptr = smem_count_ptr + expert_id
                    local_count = (slot_ptr).load()
                    delta = (Int64(1) << Int64(32)) | (Int64(local_count) & Int64(0xFFFFFFFF))
                    old_packed = cute.arch.atomic_add(
                        expert_send_count.iterator + expert_id,
                        delta,
                        sem="relaxed",
                        scope="gpu",
                    )
                    base_slot = Int32(old_packed & Int64(0xFFFFFFFF))
                    (slot_ptr).store(base_slot)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.dispatch_group_threads,
        )

        if dispatch_warps_only:
            t = base_token_for_warp
            while t < num_tokens:
                token_offset_in_warp = lane_idx // self.num_topk
                token_global = t + token_offset_in_warp
                if lane_idx < active_lanes and token_global < num_tokens:
                    topk_slot = lane_idx % self.num_topk
                    expert_id = Int32(topk_idx[token_global, topk_slot])
                    if expert_id >= Int32(0):
                        dst_rank = expert_id // Int32(self.num_experts_per_rank)
                        local_expert = expert_id % Int32(self.num_experts_per_rank)
                        slot = cute.arch.atomic_add(
                            smem_count_ptr + expert_id,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                        # Single node: every destination is node-local (main's
                        # loop shape); the node test exists only with remote nodes.
                        same_node = True
                        if cutlass.const_expr(self.num_remote_nodes > 0):
                            my_node = Int32(local_rank) // Int32(self.ranks_per_node)
                            dst_node = dst_rank // Int32(self.ranks_per_node)
                            same_node = (dst_node - my_node) == Int32(0)
                        if same_node:
                            token_topk_word = Int32(
                                token_global * self.num_topk + topk_slot
                            )
                            MAX_SLOT_C: cutlass.Constexpr[int] = (
                                num_tokens * self.num_topk
                            )
                            elem_off = (
                                (local_expert * Int32(self.world_size)
                                 + Int32(local_rank))
                                * Int32(MAX_SLOT_C)
                                + slot
                            ) * Int32(4)
                            peer_addr = peer_rank_ptr_mapper.map(
                                src_token_topk_idx.iterator.toint(),
                                dst_rank, Int64(elem_off),
                            )
                            stg_b32_raw(peer_addr, token_topk_word)
                cute.arch.sync_warp()
                t += grid_token_stride

    @cute.jit
    def internode_dispatch_prep(
        self,
        token_comm_storage,
        topk_idx,
        internode_send_count,
        internode_send_token_count,
        tokens_to_node,
        token_node_slot,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_tokens,
        num_sms,
        # Optional per-token fc1 activation scale (None disables both):
        # user (T,) fp32 on the symmetric heap -> pool-ordered local staging.
        input_fc1_activation_per_token_scale_buffer=None,
        dispatched_fc1_activation_per_token_scale_buffer=None,
        grid_sync_counter=None,
    ):
        if cutlass.const_expr(
            self.num_relay_warps > 0
            and self.num_remote_nodes > 0
            and internode_send_count is not None
            and internode_send_token_count is not None
            and tokens_to_node is not None
        ):
            relay_warp_idx = warp_idx - Int32(self.num_dispatch_warps)
            thread_idx_in_relay_group = (
                relay_warp_idx * Int32(self.warp_threads) + lane_idx
            )
            smem_inter_count_ptr = (
                token_comm_storage.smem_inter_send_count.data_ptr()
            )
            j = thread_idx_in_relay_group
            inter_count_entries: cutlass.Constexpr[int] = (
                self.num_remote_node_slots * self.node_experts
            )
            while j < Int32(inter_count_entries):
                (smem_inter_count_ptr + j).store(Int32(0))
                j = j + Int32(self.num_relay_threads)
            smem_inter_token_count_ptr = (
                token_comm_storage.smem_inter_token_count.data_ptr()
            )
            smem_inter_token_base_ptr = (
                token_comm_storage.smem_inter_token_base.data_ptr()
            )
            j = thread_idx_in_relay_group
            while j < Int32(self.num_remote_node_slots):
                (smem_inter_token_count_ptr + j).store(Int32(0))
                (smem_inter_token_base_ptr + j).store(Int32(0))
                j = j + Int32(self.num_relay_threads)

        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.dispatch_group_threads,
        )

        if cutlass.const_expr(
            self.num_relay_warps > 0
            and self.num_remote_nodes > 0
            and internode_send_count is not None
            and internode_send_token_count is not None
            and tokens_to_node is not None
        ):
            relay_warp_idx = warp_idx - Int32(self.num_dispatch_warps)
            tokens_per_relay_warp: cutlass.Constexpr[int] = 32 // self.num_topk
            active_relay_lanes: cutlass.Constexpr[int] = (
                tokens_per_relay_warp * self.num_topk
            )
            num_relay_warps_per_grid: cutlass.Constexpr[int] = (
                num_sms * self.num_relay_warps
            )
            base_token_for_relay_warp = (
                sm_idx * self.num_relay_warps + relay_warp_idx
            ) * tokens_per_relay_warp
            relay_grid_token_stride: cutlass.Constexpr[int] = (
                num_relay_warps_per_grid * tokens_per_relay_warp
            )
            my_node = Int32(local_rank) // Int32(self.ranks_per_node)
            smem_inter_count_ptr = (
                token_comm_storage.smem_inter_send_count.data_ptr()
            )
            smem_inter_token_count_ptr = (
                token_comm_storage.smem_inter_token_count.data_ptr()
            )

            t = base_token_for_relay_warp
            while t < num_tokens:
                token_offset_in_warp = lane_idx // self.num_topk
                token_global = t + token_offset_in_warp
                if lane_idx < active_relay_lanes and token_global < num_tokens:
                    topk_slot = lane_idx % self.num_topk
                    expert_id = Int32(topk_idx[token_global, topk_slot])
                    if expert_id >= Int32(0):
                        dst_rank = expert_id // Int32(self.num_experts_per_rank)
                        dst_node = dst_rank // Int32(self.ranks_per_node)
                        if dst_node != my_node:
                            remote_node_slot = dst_node
                            if dst_node > my_node:
                                remote_node_slot = dst_node - Int32(1)
                            dst_rank_in_node = (
                                dst_rank - dst_node * Int32(self.ranks_per_node)
                            )
                            local_expert = (
                                expert_id % Int32(self.num_experts_per_rank)
                            )
                            node_expert = (
                                dst_rank_in_node
                                * Int32(self.num_experts_per_rank)
                                + local_expert
                            )
                            inter_count_idx = (
                                remote_node_slot * Int32(self.node_experts)
                                + node_expert
                            )
                            cute.arch.atomic_add(
                                smem_inter_count_ptr + inter_count_idx,
                                Int32(1),
                                sem="relaxed",
                                scope="cta",
                            )

                            seen_prior_topk_for_node = Int32(0)
                            for prev_topk in cutlass.range_constexpr(
                                0, self.num_topk, 1
                            ):
                                if Int32(prev_topk) < topk_slot:
                                    prev_expert = Int32(
                                        topk_idx[token_global, prev_topk]
                                    )
                                    if prev_expert >= Int32(0):
                                        prev_rank = (
                                            prev_expert
                                            // Int32(self.num_experts_per_rank)
                                        )
                                        prev_node = (
                                            prev_rank // Int32(self.ranks_per_node)
                                        )
                                        if prev_node == dst_node:
                                            seen_prior_topk_for_node = Int32(1)
                            if seen_prior_topk_for_node == Int32(0):
                                cute.arch.atomic_add(
                                    smem_inter_token_count_ptr
                                    + remote_node_slot,
                                    Int32(1),
                                    sem="relaxed",
                                    scope="cta",
                                )
                cute.arch.sync_warp()
                t = t + relay_grid_token_stride

        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.dispatch_group_threads,
        )

        if cutlass.const_expr(
            self.num_relay_warps > 0
            and self.num_remote_nodes > 0
            and internode_send_count is not None
            and internode_send_token_count is not None
        ):
            relay_warp_idx = warp_idx - Int32(self.num_dispatch_warps)
            thread_idx_in_relay_group = (
                relay_warp_idx * Int32(self.warp_threads) + lane_idx
            )
            smem_inter_count_ptr = (
                token_comm_storage.smem_inter_send_count.data_ptr()
            )
            smem_inter_token_count_ptr = (
                token_comm_storage.smem_inter_token_count.data_ptr()
            )
            smem_inter_token_base_ptr = (
                token_comm_storage.smem_inter_token_base.data_ptr()
            )
            inter_count_entries: cutlass.Constexpr[int] = (
                self.num_remote_node_slots * self.node_experts
            )
            for remote_slot_off in cutlass.range_constexpr(
                0, self.num_remote_node_slots, self.num_relay_threads,
            ):
                remote_node_slot = Int32(remote_slot_off) + thread_idx_in_relay_group
                if remote_node_slot < Int32(self.num_remote_node_slots):
                    local_token_count = (
                        smem_inter_token_count_ptr + remote_node_slot
                    ).load()
                    delta = (
                        (Int64(1) << Int64(32))
                        | (Int64(local_token_count) & Int64(0xFFFFFFFF))
                    )
                    old_packed = cute.arch.atomic_add(
                        internode_send_token_count.iterator + remote_node_slot,
                        delta,
                        sem="relaxed",
                        scope="gpu",
                    )
                    base_slot = Int32(old_packed & Int64(0xFFFFFFFF))
                    (smem_inter_token_count_ptr + remote_node_slot).store(
                        base_slot
                    )
                    (smem_inter_token_base_ptr + remote_node_slot).store(
                        base_slot
                    )
            for offset in cutlass.range_constexpr(
                0, inter_count_entries, self.num_relay_threads,
            ):
                inter_count_idx = Int32(offset) + thread_idx_in_relay_group
                if inter_count_idx < Int32(inter_count_entries):
                    local_count = (smem_inter_count_ptr + inter_count_idx).load()
                    remote_node_slot = (
                        inter_count_idx // Int32(self.node_experts)
                    )
                    node_expert = (
                        inter_count_idx % Int32(self.node_experts)
                    )
                    global_inter_count_idx = (
                        remote_node_slot * Int32(self.gin_send_count_stride)
                        + node_expert
                    )
                    cute.arch.atomic_add(
                        internode_send_count.iterator + global_inter_count_idx,
                        local_count,
                        sem="relaxed",
                        scope="gpu",
                    )

        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.dispatch_group_threads,
        )

        if cutlass.const_expr(
            self.num_relay_warps > 0
            and self.num_remote_nodes > 0
            and internode_send_token_count is not None
            and tokens_to_node is not None
        ):
            relay_warp_idx = warp_idx - Int32(self.num_dispatch_warps)
            tokens_per_relay_warp: cutlass.Constexpr[int] = 32 // self.num_topk
            active_relay_lanes: cutlass.Constexpr[int] = (
                tokens_per_relay_warp * self.num_topk
            )
            num_relay_warps_per_grid: cutlass.Constexpr[int] = (
                num_sms * self.num_relay_warps
            )
            base_token_for_relay_warp = (
                sm_idx * self.num_relay_warps + relay_warp_idx
            ) * tokens_per_relay_warp
            relay_grid_token_stride: cutlass.Constexpr[int] = (
                num_relay_warps_per_grid * tokens_per_relay_warp
            )
            my_node = Int32(local_rank) // Int32(self.ranks_per_node)
            smem_inter_token_count_ptr = (
                token_comm_storage.smem_inter_token_count.data_ptr()
            )

            t = base_token_for_relay_warp
            while t < num_tokens:
                token_offset_in_warp = lane_idx // self.num_topk
                token_global = t + token_offset_in_warp
                if lane_idx < active_relay_lanes and token_global < num_tokens:
                    topk_slot = lane_idx % self.num_topk
                    expert_id = Int32(topk_idx[token_global, topk_slot])
                    if expert_id >= Int32(0):
                        dst_rank = expert_id // Int32(self.num_experts_per_rank)
                        dst_node = dst_rank // Int32(self.ranks_per_node)
                        if dst_node != my_node:
                            remote_node_slot = dst_node
                            if dst_node > my_node:
                                remote_node_slot = dst_node - Int32(1)
                            seen_prior_topk_for_node = Int32(0)
                            for prev_topk in cutlass.range_constexpr(
                                0, self.num_topk, 1
                            ):
                                if Int32(prev_topk) < topk_slot:
                                    prev_expert = Int32(
                                        topk_idx[token_global, prev_topk]
                                    )
                                    if prev_expert >= Int32(0):
                                        prev_rank = (
                                            prev_expert
                                            // Int32(self.num_experts_per_rank)
                                        )
                                        prev_node = (
                                            prev_rank // Int32(self.ranks_per_node)
                                        )
                                        if prev_node == dst_node:
                                            seen_prior_topk_for_node = Int32(1)
                            if seen_prior_topk_for_node == Int32(0):
                                token_slot = cute.arch.atomic_add(
                                    smem_inter_token_count_ptr
                                    + remote_node_slot,
                                    Int32(1),
                                    sem="relaxed",
                                    scope="cta",
                                )
                                tokens_to_node[remote_node_slot, token_slot] = (
                                    token_global
                                )
                                if cutlass.const_expr(token_node_slot is not None):
                                    # Inverse map for the slot-indexed combine
                                    # return: TopkReduce reads this token's
                                    # returned partial at message slot
                                    # token_node_slot[node][token].
                                    token_node_slot[
                                        remote_node_slot, token_global
                                    ] = token_slot
                cute.arch.sync_warp()
                t = t + relay_grid_token_stride
            if cutlass.const_expr(self.relay_skip_dispatch_barrier):
                relay_prep_tid = (
                    (warp_idx - Int32(self.num_dispatch_warps))
                    * Int32(self.warp_threads) + lane_idx
                )
                if cutlass.const_expr(self.gin_pack_flat_assign):
                    # Early Stage 1 (field 60): the flattened pack reads every CTA's
                    # reservations and tokens_to_node entries, which the
                    # skipped dispatch barrier no longer publishes; one
                    # relay-only grid sync (slot 1, barrier 11) does.
                    if cutlass.const_expr(self.iket_enabled):
                        pack_flat_grid_sync_range = _iket.range_start(
                            "megamoe_grid_sync_wait",
                            IKET_MEGA_GRID_SYNC_WAIT_PACK_FLAT,
                        )
                    software_grid_sync(
                        grid_sync_counter.iterator + _RELAY_GRID_SYNC_SLOT,
                        sm_idx,
                        num_sms,
                        relay_prep_tid,
                        num_threads=self.num_relay_threads,
                        barrier_id=self.relay_early_push_bar_id,
                    )
                    if cutlass.const_expr(self.iket_enabled):
                        _iket.range_end(
                            pack_flat_grid_sync_range,
                            IKET_MEGA_GRID_SYNC_WAIT_PACK_FLAT,
                        )
                else:
                    # Field 60: without the dispatch barrier nothing else orders
                    # one relay warp's tokens_to_node stores and slot bases before
                    # another relay warp of this CTA packs those slots. Relay-only
                    # id: the dispatch warps may already sit on barrier 10/12.
                    cute.arch.barrier(
                        barrier_id=self.relay_early_push_bar_id,
                        number_of_threads=self.num_relay_threads,
                    )

    @cute.jit
    def internode_push(
        self,
        token_comm_storage,
        input_token_buffer,
        input_sf_buffer,
        topk_idx,
        input_topk_weights_buffer,
        tokens_to_node,
        internode_send_count,
        grid_sync_counter,
        gin_window,
        pipeline_timing,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_sms,
        window_base=None,
        internode_send_token_count=None,
    ):
        """Pack and RDMA-push this CTAs node-deduplicated relay messages.

        Default: one thread-scope put per message from the packing thread or
        warp, plus one strong credit per active (CTA, destination node).
        Coalesced Stage 1: pack in place, rendezvous the relay
        warps of all CTAs once, then SM0 issues one put per destination node
        over the dense ``[0, message_count)`` slot range and one strong credit
        carrying the rank-wide expert-hit total.
        """
        if cutlass.const_expr(
            self.enable_nccl_gin
            and self.num_relay_warps > 0
            and self.num_remote_nodes > 0
            and tokens_to_node is not None
            and gin_window is not None
        ):
            if cutlass.const_expr(self.iket_enabled):
                dispatch_put_range = _iket.range_start(
                    "megamoe_envelope", IKET_MEGA_ENVELOPE_DISPATCH_PUT
                )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                pack_entry_t = read_globaltimer()
                pack_msg_ns = Int64(0)
                pack_route_ns = Int64(0)
                pack_copy_ns = Int64(0)
            from nccl.core.device import cute as nccl_cute

            relay_warp_idx = warp_idx - Int32(self.num_dispatch_warps)
            thread_idx_in_relay_group = (
                relay_warp_idx * Int32(self.warp_threads) + lane_idx
            )
            smem_inter_count_ptr = (
                token_comm_storage.smem_inter_send_count.data_ptr()
            )
            smem_inter_token_count_ptr = (
                token_comm_storage.smem_inter_token_count.data_ptr()
            )
            smem_inter_token_base_ptr = (
                token_comm_storage.smem_inter_token_base.data_ptr()
            )
            gin = self._device_gin_gpu()
            team_rail = self._team_rail_value
            thread_coop = nccl_cute.thread()
            relay_coop = nccl_cute.warp_span(
                warp0=self.relay_warp_start,
                n_warps=self.num_relay_warps,
                id=0,
            )
            if cutlass.const_expr(self.diag_stage1_flush):
                # Own span id: its sync barrier (1 + id) must not alias the
                # epilogue barrier while FC1 may already run (class comment).
                flush_coop = nccl_cute.warp_span(
                    warp0=self.relay_warp_start,
                    n_warps=self.num_relay_warps,
                    id=self.relay_flush_coop_id,
                )
            if cutlass.const_expr(window_base is not None):
                # Hoisted base (field 47): pure address arithmetic instead of
                # the out-of-line window lookup on every relay warp.
                window_u8 = cute.make_tensor(
                    cute.make_ptr(
                        Uint8, window_base, AddressSpace.gmem, assumed_align=16,
                    ),
                    cute.make_layout((self.gin_window_bytes,), stride=(1,)),
                )
            else:
                window_u8 = gin_window.tensor(
                    Uint8,
                    cute.make_layout((self.gin_window_bytes,), stride=(1,)),
                )
            message_layout = cute.make_layout(
                (self.relay_message_bytes,), stride=(1,)
            )
            my_node = Int32(local_rank) // Int32(self.ranks_per_node)
            input_token_base = input_token_buffer.iterator.toint()
            input_sf_base = input_sf_buffer.iterator.toint()
            input_weights_base = input_topk_weights_buffer.iterator.toint()
            if cutlass.const_expr(self.pipeline_timing_enabled):
                pack_setup_t = read_globaltimer()

            # The packed global reservation gives every CTA a disjoint
            # [base_slot, end_slot) range in each destination-node buffer.
            # By default, one relay thread owns each message. The opt-in
            # warp-pack path assigns one relay warp to a message, while lane 0
            # retains the same per-token thread-scope put (Invariant I4).
            if cutlass.const_expr(self.gin_pack_flat_assign):
                # With field 60 the grid-wide publication of the other CTAs'
                # reservations happens at the end of internode_dispatch_prep.
                flat_global_warp = (
                    sm_idx * Int32(self.num_relay_warps) + relay_warp_idx
                )
                flat_num_warps: cutlass.Constexpr[int] = (
                    num_sms * self.num_relay_warps
                )
                flat_prefix = Int32(0)
            if cutlass.const_expr(self.pipeline_timing_enabled):
                pack_loop_t0 = read_globaltimer()
            for dst_slot_off in cutlass.range_constexpr(
                0, self.num_remote_node_slots, 1
            ):
                dst_slot = Int32(dst_slot_off)
                dst_node = dst_slot
                if dst_slot >= my_node:
                    dst_node = dst_slot + Int32(1)
                src_slot_at_dst = my_node
                if my_node > dst_node:
                    src_slot_at_dst = my_node - Int32(1)
                if cutlass.const_expr(self.gin_pack_flat_assign):
                    # Rank-wide dense range [0, count_s): flattened message
                    # prefix_s + m is packed by relay warp (prefix_s + m) mod W.
                    base_slot = Int32(0)
                    end_slot = Int32(
                        cute.arch.load(
                            internode_send_token_count.iterator + dst_slot,
                            Int64,
                            sem="relaxed",
                            scope="gpu",
                        ) & Int64(0xFFFFFFFF)
                    )
                else:
                    base_slot = (
                        smem_inter_token_base_ptr + dst_slot
                    ).load()
                    end_slot = (
                        smem_inter_token_count_ptr + dst_slot
                    ).load()
                if end_slot > Int32(self.max_relay_tokens_per_node):
                    if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                        cute.printf("[gin-guard] token_comm source-site 2364\n")
                    device_trap()
                if cutlass.const_expr(self.gin_pack_flat_assign):
                    token_slot = flat_global_warp - (
                        flat_prefix % Int32(flat_num_warps)
                    )
                    if token_slot < Int32(0):
                        token_slot = token_slot + Int32(flat_num_warps)
                    token_slot_stride = Int32(flat_num_warps)
                    flat_prefix = flat_prefix + end_slot
                else:
                    token_slot = base_slot + relay_warp_idx
                    token_slot_stride = Int32(self.num_relay_warps)
                dispatch_active = (
                    cute.arch.vote_ballot_sync(token_slot < end_slot)
                    != Int32(0)
                )
                if dispatch_active:
                    if cutlass.const_expr(self.iket_detailed_ranges):
                        dispatch_message_range = _iket.range_start(
                            "megamoe_work", IKET_MEGA_WORK_DISPATCH_MESSAGE
                        )
                    while token_slot < end_slot:
                        if cutlass.const_expr(self.pipeline_timing_enabled):
                            pack_msg_t0 = read_globaltimer()
                        token_idx = Int32(0)
                        if lane_idx == Int32(0):
                            if cutlass.const_expr(self.gin_pack_flat_assign):
                                # Early Stage 1 (field 60): the entry was written by another
                                # CTA and published by a grid sync; read it
                                # as a strong GPU-scope load (no L1 reuse).
                                token_idx = cute.arch.load(
                                    tokens_to_node.iterator
                                    + dst_slot * Int32(self.max_relay_tokens_per_node)
                                    + token_slot,
                                    Int32,
                                    sem="relaxed",
                                    scope="gpu",
                                )
                            else:
                                token_idx = Int32(
                                    tokens_to_node[dst_slot, token_slot]
                                )
                        token_idx = cute.arch.shuffle_sync(
                            token_idx, Int32(0)
                        )
                        # The relay regions can exceed 2 GiB at large token
                        # counts. Promote before multiplying; casting an
                        # already-overflowed Int32 product is too late.
                        send_byte_offset = Int64(
                            self.relay_send_buffer_offset
                        ) + (
                            Int64(dst_slot)
                            * Int64(self.max_relay_tokens_per_node)
                            + Int64(token_slot)
                        ) * Int64(self.relay_message_bytes)
                        recv_byte_offset = Int64(
                            self.relay_recv_buffer_offset
                        ) + (
                            Int64(src_slot_at_dst)
                            * Int64(self.max_relay_tokens_per_node)
                            + Int64(token_slot)
                        ) * Int64(self.relay_message_bytes)
                        send_addr = (
                            window_u8.iterator.toint() + send_byte_offset
                        )

                        # Issue the whole DATA row (one 16-byte vector per
                        # lane per 512-byte row) and the SF row as
                        # predicated 8-byte loads now; the route loop below
                        # runs while they are in flight and the stores
                        # follow it. Predicates are lane-uniform per row
                        # except for a partial last row.
                        early_lane_off = Int64(lane_idx) * Int64(16)
                        early_data_src = (
                            input_token_base
                            + Int64(token_idx * Int32(self.hidden_bytes))
                            + early_lane_off
                        )
                        early_lo = []
                        early_hi = []
                        early_valid = []
                        for row_k in cutlass.range_constexpr(
                            0, (self.hidden_bytes + 511) // 512, 1
                        ):
                            row_valid = Int32(
                                Int32(row_k * 512) + lane_idx * Int32(16)
                                < Int32(self.hidden_bytes)
                            )
                            early_valid.append(row_valid)
                            early_lo.append(ldg_b64_predicated_raw(
                                early_data_src + Int64(row_k * 512), row_valid,
                            ))
                            early_hi.append(ldg_b64_predicated_raw(
                                early_data_src + Int64(row_k * 512 + 8), row_valid,
                            ))
                        early_sf_valid = Int32(
                            lane_idx * Int32(16)
                            < Int32(self.sf_uint32_per_token * 4)
                        )
                        early_sf_src = (
                            input_sf_base
                            + Int64(token_idx * Int32(self.sf_uint32_per_token * 4))
                            + early_lane_off
                        )
                        early_sf_lo = ldg_b64_predicated_raw(early_sf_src, early_sf_valid)
                        early_sf_hi = ldg_b64_predicated_raw(
                            early_sf_src + Int64(8), early_sf_valid,
                        )
                        if lane_idx == Int32(0):
                            TokenSrcMetadata(
                                src_rank=Int32(local_rank),
                                src_token=token_idx,
                                src_topk=Int32(0xFFFF),
                            ).store(
                                send_addr
                                + Int64(
                                    self.relay_message_src_metadata_offset
                                )
                            )
                            stg_b32_raw(send_addr + Int64(8), Int32(0))
                            stg_b32_raw(send_addr + Int64(12), Int32(0))

                        # Lanes 0..num_topk-1 fetch their expert id and
                        # weight in one round trip; the route loop below
                        # reads them with shuffles instead of four
                        # dependent load-branch-load chains.
                        lane_topk_expert = Int32(-1)
                        lane_topk_weight = Int32(0)
                        if lane_idx < Int32(self.num_topk):
                            lane_topk_expert = Int32(topk_idx[token_idx, lane_idx])
                            lane_topk_weight = Int32(ldg_b32_raw(
                                input_weights_base
                                + Int64(
                                    (token_idx * Int32(self.num_topk) + lane_idx)
                                    * Int32(4)
                                )
                            ))
                        for topk_off in cutlass.range_constexpr(
                            0, self.num_topk, 1
                        ):
                            topk_slot = Int32(topk_off)
                            expert_id = cute.arch.shuffle_sync(
                                lane_topk_expert, Int32(topk_off)
                            )
                            prefetched_weight = cute.arch.shuffle_sync(
                                lane_topk_weight, Int32(topk_off)
                            )
                            route_word = Int32(0xFFFF)
                            weight_bits = Int32(0)
                            if expert_id >= Int32(0):
                                dst_rank = (
                                    expert_id
                                    // Int32(self.num_experts_per_rank)
                                )
                                expert_node = (
                                    dst_rank // Int32(self.ranks_per_node)
                                )
                                if expert_node == dst_node:
                                    rank_in_dst_node = (
                                        dst_rank
                                        - dst_node * Int32(self.ranks_per_node)
                                    )
                                    node_expert = (
                                        rank_in_dst_node
                                        * Int32(self.num_experts_per_rank)
                                        + expert_id
                                        % Int32(self.num_experts_per_rank)
                                    )
                                    route_word = (
                                        node_expert
                                        | (topk_slot << Int32(16))
                                    )
                                    weight_bits = prefetched_weight
                            if lane_idx == Int32(0):
                                stg_b32_raw(
                                    send_addr
                                    + Int64(
                                        self.relay_message_route_words_offset
                                    )
                                    + Int64(topk_slot * Int32(4)),
                                    route_word,
                                )
                                stg_b32_raw(
                                    send_addr
                                    + Int64(
                                        self.relay_message_topk_weights_offset
                                    )
                                    + Int64(topk_slot * Int32(4)),
                                    weight_bits,
                                )

                        if cutlass.const_expr(self.pipeline_timing_enabled):
                            pack_msg_t_route = read_globaltimer()
                        for row_k in cutlass.range_constexpr(
                            0, (self.hidden_bytes + 511) // 512, 1
                        ):
                            if early_valid[row_k] != Int32(0):
                                stg_b64_raw(
                                    send_addr
                                    + Int64(self.relay_message_token_data_offset)
                                    + Int64(row_k * 512) + early_lane_off,
                                    early_lo[row_k],
                                )
                                stg_b64_raw(
                                    send_addr
                                    + Int64(self.relay_message_token_data_offset)
                                    + Int64(row_k * 512 + 8) + early_lane_off,
                                    early_hi[row_k],
                                )
                        if early_sf_valid != Int32(0):
                            stg_b64_raw(
                                send_addr
                                + Int64(self.relay_message_sf_data_offset)
                                + early_lane_off,
                                early_sf_lo,
                            )
                            stg_b64_raw(
                                send_addr
                                + Int64(self.relay_message_sf_data_offset)
                                + Int64(8) + early_lane_off,
                                early_sf_hi,
                            )

                        sf_bytes: cutlass.Constexpr[int] = (
                            self.sf_uint32_per_token * 4
                        )
                        # SF row already stored from the early loads.
                        pass

                        payload_end: cutlass.Constexpr[int] = (
                            self.relay_message_sf_data_offset + sf_bytes
                        )
                        byte_off = (
                            Int32(payload_end) + lane_idx * Int32(16)
                        )
                        while byte_off < Int32(self.relay_message_bytes):
                            stg_b128_words_raw(
                                send_addr + Int64(byte_off),
                                Int32(0),
                                Int32(0),
                                Int32(0),
                                Int32(0),
                            )
                            byte_off = byte_off + Int32(
                                self.warp_threads * 16
                            )

                        if cutlass.const_expr(self.pipeline_timing_enabled):
                            pack_msg_t_copy = read_globaltimer()
                            pack_route_ns = pack_route_ns + (pack_msg_t_route - pack_msg_t0)
                            pack_copy_ns = pack_copy_ns + (pack_msg_t_copy - pack_msg_t_route)
                        src_message = cute.make_tensor(
                            cute.make_ptr(
                                Uint8,
                                window_u8.iterator.toint()
                                + send_byte_offset,
                                AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            message_layout,
                        )
                        dst_message = cute.make_tensor(
                            cute.make_ptr(
                                Uint8,
                                window_u8.iterator.toint()
                                + recv_byte_offset,
                                AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            message_layout,
                        )
                        # Coalesced mode leaves the packed message in place;
                        # SM0 moves the whole per-node slot range after the
                        # relay grid sync below.
                        pass
                        if cutlass.const_expr(self.pipeline_timing_enabled):
                            pack_msg_ns = pack_msg_ns + (read_globaltimer() - pack_msg_t0)
                        token_slot = token_slot + token_slot_stride
                    if cutlass.const_expr(self.iket_detailed_ranges):
                        _iket.range_end(
                            dispatch_message_range,
                            IKET_MEGA_WORK_DISPATCH_MESSAGE,
                        )

            if cutlass.const_expr(self.pipeline_timing_enabled):
                # Per relay warp; the host reducer's min-start/max-end gives the
                # prologue window and the busiest warp's summed pack phases.
                if lane_idx == Int32(0):
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PROLOGUE, 0
                    ] = pack_entry_t
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PROLOGUE, 1
                    ] = pack_loop_t0
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PACK, 0
                    ] = pack_loop_t0
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PACK, 1
                    ] = pack_loop_t0 + pack_msg_ns
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PACK_ROUTE, 0
                    ] = pack_loop_t0
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PACK_ROUTE, 1
                    ] = pack_loop_t0 + pack_route_ns
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PACK_COPY, 0
                    ] = pack_loop_t0
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PACK_COPY, 1
                    ] = pack_loop_t0 + pack_copy_ns
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PROLOGUE_SETUP, 0
                    ] = pack_entry_t
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_PUSH_PROLOGUE_SETUP, 1
                    ] = pack_setup_t
            # Coalesced Stage 1. The packed reservation already made every
            # destination node's messages dense in
            # relay_send_buffer[dst][0:message_count), and the receiver's
            # relay_recv_buffer[src][slot] uses the same slot index, so the
            # whole range is one contiguous transfer. Every relay thread
            # publishes its packed bytes, the relay warps of all CTAs
            # rendezvous once, and SM0 issues one put per node followed by
            # one flush and one strong credit. The credit carries the
            # rank-wide expert-hit total, which is exactly the receiver's
            # sum(internode_recv_splits[src][0:R*E]) target (Invariants
            # I4, I5, I8).
            if cutlass.const_expr(self.pipeline_timing_enabled):
                push_sync_start = read_globaltimer()
            # bar.warp.sync orders every lane's packed stores before
            # lane 0's cumulative system fence; the grid sync's
            # release/acquire chain then carries them to SM0's put.
            cute.arch.sync_warp()
            if lane_idx == Int32(0):
                cute.arch.fence_acq_rel_sys()
            if cutlass.const_expr(self.iket_enabled):
                dispatch_coalesce_grid_sync_range = _iket.range_start(
                    "megamoe_grid_sync_wait",
                    IKET_MEGA_GRID_SYNC_WAIT_DISPATCH_COALESCE,
                )
            software_grid_sync(
                grid_sync_counter.iterator + _RELAY_GRID_SYNC_SLOT,
                sm_idx,
                num_sms,
                thread_idx_in_relay_group,
                num_threads=self.num_relay_threads,
                barrier_id=(
                    self.relay_early_push_bar_id
                ),
            )
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(
                    dispatch_coalesce_grid_sync_range,
                    IKET_MEGA_GRID_SYNC_WAIT_DISPATCH_COALESCE,
                )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                push_sync_end = read_globaltimer()
                # Eager record (also rewritten at the block end): lets the
                # host hang dump tell "stuck in the relay grid sync" from
                # "stuck in the range put or credit" on SM0.
                if sm_idx == Int32(0):
                    if relay_warp_idx == Int32(0):
                        if lane_idx == Int32(0):
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_SYNC, 0
                            ] = push_sync_start
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_SYNC, 1
                            ] = push_sync_end
            if cutlass.const_expr(self.step3_put_before_stage1):
                # early Stage 1 (field 60): warp-uniform views for the metadata rows
                # (built by every relay warp; only the SM0 issuer lanes
                # below consume them, like the Step 3 lane form).
                step3_row_layout = cute.make_layout(
                    (self.gin_send_count_stride,), stride=(1,)
                )
                step3_window_i32 = gin_window.tensor(
                    Int32,
                    cute.make_layout(
                        (self.gin_metadata_window_i32,), stride=(1,)
                    ),
                )
            if sm_idx == Int32(0):
                # Lane-parallel issue: lanes 0..slots-1 of relay warp 0 each
                # take one destination slot in the same instruction stream
                # (step = slots, one iteration per lane), so the puts of all
                # remote nodes post without a cross-warp barrier.
                stage1_issue_active = Int32(0)
                if relay_warp_idx == Int32(0):
                    if lane_idx < Int32(self.num_remote_node_slots):
                        stage1_issue_active = Int32(1)
                dst_slot = lane_idx
                stage1_slot_step = Int32(self.num_remote_node_slots)
                if stage1_issue_active != Int32(0):
                    while dst_slot < Int32(self.num_remote_node_slots):
                        dst_node = dst_slot
                        if dst_slot >= my_node:
                            dst_node = dst_slot + Int32(1)
                        src_slot_at_dst = my_node
                        if my_node > dst_node:
                            src_slot_at_dst = my_node - Int32(1)
                        if cutlass.const_expr(
                            self.relay_skip_dispatch_barrier or self.step3_put_before_stage1
                        ):
                            # Early Stage 1 (field 60): the trailing window
                            # word is written by Step 3 on SM0's dispatch warp
                            # after the dispatch barrier the relay warps no
                            # longer wait for (or, in mode 3, by this lane
                            # below); read the same count from the packed
                            # reservation counter every CTA advanced before
                            # the relay grid sync (I4).
                            message_count = Int32(
                                cute.arch.load(
                                    internode_send_token_count.iterator
                                    + dst_slot,
                                    Int64,
                                    sem="relaxed",
                                    scope="gpu",
                                ) & Int64(0xFFFFFFFF)
                            )
                        else:
                            # Trailing Step 3 word: the node-deduplicated
                            # message count this rank reserved for dst.
                            message_count = Int32(
                                internode_send_count[
                                    dst_slot, self.node_experts
                                ]
                            )
                        if message_count > Int32(
                            self.max_relay_tokens_per_node
                        ):
                            if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                                cute.printf("[gin-guard] token_comm source-site 2797\n")
                            device_trap()
                        if cutlass.const_expr(self.step3_put_before_stage1):
                            # early Stage 1 (field 60): this destination's Step 3
                            # metadata row (R*E counts landed by every CTA's
                            # atomics before the relay grid sync, trailing
                            # word = message count) goes out on the same
                            # queue pair immediately before the data put,
                            # so it is transmitted first; the bundled INC
                            # signal is what Step 5a waits for (I4, I5).
                            step3_window_i32[
                                dst_slot * Int32(self.gin_send_count_stride)
                                + Int32(self.node_experts)
                            ] = message_count
                            cute.arch.fence_acq_rel_sys()
                            step3_src_row = cute.make_tensor(
                                step3_window_i32.iterator
                                + dst_slot * Int32(self.gin_send_count_stride),
                                step3_row_layout,
                            )
                            step3_dst_row = cute.make_tensor(
                                step3_window_i32.iterator
                                + Int32(self.gin_recv_splits_offset_i32)
                                + src_slot_at_dst * Int32(self.gin_send_count_stride),
                                step3_row_layout,
                            )
                            gin.put(
                                team_rail,
                                dst_node,
                                gin_window,
                                step3_dst_row,
                                gin_window,
                                step3_src_row,
                                thread_coop,
                                is_signal=True,
                                signal_id=(
                                    Int32(self.internode_splits_signal_base)
                                    + src_slot_at_dst
                                ),
                                signal_op=self.gin_signal_op_inc,
                            )
                        if message_count > Int32(0):
                            range_bytes = message_count * Int32(
                                self.relay_message_bytes
                            )
                            range_layout = cute.make_layout(
                                (range_bytes,), stride=(1,)
                            )
                            send_range_offset = Int64(
                                self.relay_send_buffer_offset
                            ) + Int64(dst_slot) * Int64(
                                self.max_relay_tokens_per_node
                            ) * Int64(self.relay_message_bytes)
                            recv_range_offset = Int64(
                                self.relay_recv_buffer_offset
                            ) + Int64(src_slot_at_dst) * Int64(
                                self.max_relay_tokens_per_node
                            ) * Int64(self.relay_message_bytes)
                            src_range = cute.make_tensor(
                                cute.make_ptr(
                                    Uint8,
                                    window_u8.iterator.toint()
                                    + send_range_offset,
                                    AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                range_layout,
                            )
                            dst_range = cute.make_tensor(
                                cute.make_ptr(
                                    Uint8,
                                    window_u8.iterator.toint()
                                    + recv_range_offset,
                                    AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                range_layout,
                            )
                            # The expert-hit credit rides this
                            # single range put as a bundled
                            # signal add: the receiver sees it
                            # only after the put has settled,
                            # so no flush and no separate credit
                            # post are needed (Invariant I5,
                            # whole-range case).
                            put_credit = Int32(0)
                            for g_off in cutlass.range_constexpr(
                                0, self.node_experts, 1
                            ):
                                put_credit = put_credit + Int32(
                                    internode_send_count[
                                        dst_slot, Int32(g_off)
                                    ]
                                )
                            gin.put(
                                team_rail,
                                dst_node,
                                gin_window,
                                dst_range,
                                gin_window,
                                src_range,
                                thread_coop,
                                is_signal=True,
                                signal_id=(
                                    Int32(self.relay_node_signal_base)
                                    + src_slot_at_dst
                                ),
                                signal_op=_NCCL_GIN_SIGNAL_OP_ADD,
                                signal_op_arg=Int64(put_credit),
                                given_release=nccl_cute.ThreadScope.THREAD,
                            )
                        dst_slot = dst_slot + stage1_slot_step
                if relay_warp_idx == Int32(0):
                    cute.arch.sync_warp()
                if cutlass.const_expr(self.pipeline_timing_enabled):
                    push_issue_end = read_globaltimer()
                    # Eager record for the hang dump (see push_sync above).
                    if relay_warp_idx == Int32(0):
                        if lane_idx == Int32(0):
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_ISSUE, 0
                            ] = push_sync_end
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_ISSUE, 1
                            ] = push_issue_end
                if cutlass.const_expr(self.diag_stage1_flush):
                    # Field 59 (diagnostic): SM0's relay warps wait for
                    # the completion entries of this rank's range puts,
                    # i.e. until the remote NICs acknowledged the landed
                    # bytes; push_flush = post-to-landed on one clock.
                    gin.flush(
                        flush_coop, ord=nccl_cute.MemoryOrder.ACQUIRE
                    )
                if cutlass.const_expr(self.pipeline_timing_enabled):
                    # SM0 sub-envelopes of the coalesced push: pack plus
                    # relay grid sync, range puts, flush plus credits.
                    push_flush_end = read_globaltimer()
                    if relay_warp_idx == Int32(0):
                        if lane_idx == Int32(0):
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_SYNC, 0
                            ] = push_sync_start
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_SYNC, 1
                            ] = push_sync_end
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_ISSUE, 0
                            ] = push_sync_end
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_ISSUE, 1
                            ] = push_issue_end
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_FLUSH, 0
                            ] = push_issue_end
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_PUSH_FLUSH, 1
                            ] = push_flush_end
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(
                    dispatch_put_range, IKET_MEGA_ENVELOPE_DISPATCH_PUT
                )

    @cute.jit
    def _direct_fanout_reserve(
        self, args, src_slot, message_slot,
        node_expert, node_base_rank, rank_in_node, lane_idx,
    ):
        """Lanes 0..3 reserve their own wire top-k row without publishing it."""
        pool_slot = Int32(0)
        sf_slot = Int32(0)
        if lane_idx < Int32(4) and node_expert != Int32(0xFFFF):
            owner_rank = node_base_rank + node_expert // Int32(self.num_experts_per_rank)
            expert = node_expert % Int32(self.num_experts_per_rank)
            bucket_offset = (
                Int64(expert) * Int64(self.num_remote_node_slots) + Int64(src_slot)
            ) * Int64(4)
            reservation_addr = args.peer_rank_ptr_mapper.map(
                args.relay_fanout_reservation.iterator.toint(), owner_rank, bucket_offset,
            )
            reservation_ptr = cute.make_ptr(
                Int32, reservation_addr, AddressSpace.gmem, assumed_align=4,
            )
            row = cute.arch.atomic_add(
                reservation_ptr, Int32(1), sem="relaxed", scope="sys",
            )
            # The atomic is already in flight; one system-acquire load of
            # the owner's packed Step 5c record replaces three dependent
            # acquire loads (I2 keeps the acquire on the same publication).
            meta_addr = args.peer_rank_ptr_mapper.map(
                args.relay_direct_bucket_meta.iterator.toint(), owner_rank,
                (Int64(expert) * Int64(self.num_remote_node_slots) + Int64(src_slot)) * Int64(8),
            )
            packed_meta = cute.arch.load(
                cute.make_ptr(Int64, meta_addr, AddressSpace.gmem, assumed_align=8),
                Int64, sem="acquire", scope="sys",
            )
            expected = Int32(packed_meta >> Int64(DIRECT_META_COUNT_SHIFT))
            data_base = Int32(packed_meta & Int64(DIRECT_META_FIELD_MASK))
            sf_base = Int32(
                (packed_meta >> Int64(DIRECT_META_SF_SHIFT)) & Int64(DIRECT_META_FIELD_MASK)
            )
            if row < Int32(0) or row >= expected:
                if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                    cute.printf("[gin-guard] token_comm source-site 2998\n")
                device_trap()
            pool_slot = row + data_base
            sf_slot = row + sf_base
            if (
                pool_slot < Int32(0) or pool_slot >= Int32(args.pool_token_capacity)
                or sf_slot < Int32(0) or sf_slot >= Int32(args.pool_sf_capacity)
            ):
                if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                    cute.printf("[gin-guard] token_comm source-site 3005\n")
                device_trap()
            # This is the final DATA pool slot, not a rank-local landing slot.
            args.relay_fanout_slot[src_slot, message_slot, lane_idx] = pool_slot
        return pool_slot, sf_slot

    @cute.jit
    def _direct_fanout_hit(
        self, args, message_addr, src_slot, message_slot, topk_slot,
        node_expert, node_base_rank, lane_idx, pool_slot, sf_slot,
    ):
        """Copy into one reserved row and return its unpublished ready flag.

        DATA/SF slots are the reserving lane's Step 5c-derived values. The
        caller retains the same all-writer rendezvous and system publication.
        """
        route_ns = Int64(0)
        copy_ns = Int64(0)
        if cutlass.const_expr(self.pipeline_timing_enabled):
            route_start = read_globaltimer()
        owner_rank = node_base_rank + node_expert // Int32(self.num_experts_per_rank)
        data_addr = args.peer_rank_ptr_mapper.map(
            args.fc1_input_token_buffer.iterator.toint(), owner_rank,
            Int64(pool_slot) * Int64(self.hidden_bytes),
        )
        sf_addr = args.peer_rank_ptr_mapper.map(
            args.fc1_input_sf_buffer.iterator.toint(), owner_rank, Int64(0),
        )
        ready_addr = args.peer_rank_ptr_mapper.map(
            args.fc1_ready_counter.iterator.toint(), owner_rank,
            Int64(pool_slot // Int32(self.cluster_tile_tokens)) * Int64(4),
        )
        if cutlass.const_expr(self.pipeline_timing_enabled):
            copy_start = read_globaltimer()
            route_ns = copy_start - route_start

        if cutlass.const_expr(not self.diag_fanout_skip_data):
            if cutlass.const_expr(self.hidden_bytes % 16 == 0):
                self._copy_rows_unrolled(
                    data_addr, message_addr + Int64(self.relay_message_token_data_offset),
                    lane_idx, total_bytes=self.hidden_bytes,
                )
            else:
                byte_off = lane_idx * Int32(4)
                while byte_off < Int32(self.hidden_bytes):
                    stg_b32_raw(
                        data_addr + Int64(byte_off),
                        ldg_b32_raw(message_addr + Int64(self.relay_message_token_data_offset) + Int64(byte_off)),
                    )
                    byte_off = byte_off + Int32(self.warp_threads * 4)
        if cutlass.const_expr(not self.diag_fanout_skip_sf):
            sf_rounds: cutlass.Constexpr[int] = (
                self.sf_uint32_per_token + self.warp_threads - 1
            ) // self.warp_threads
            sf_words = []
            for sf_round in cutlass.range_constexpr(0, sf_rounds, 1):
                word = lane_idx + Int32(sf_round * self.warp_threads)
                sf_words.append(ldg_b32_predicated_raw(
                    message_addr + Int64(self.relay_message_sf_data_offset) + Int64(word) * Int64(4),
                    Int32(word < Int32(self.sf_uint32_per_token)),
                ))
            for sf_round in cutlass.range_constexpr(0, sf_rounds, 1):
                word = lane_idx + Int32(sf_round * self.warp_threads)
                if word < Int32(self.sf_uint32_per_token):
                    if cutlass.const_expr(self.sf_atom_swizzled):
                        sf_pos = sf_atom_int32_offset(
                            sf_slot, word, num_k_atoms=self.sf_uint32_per_token,
                        )
                    else:
                        sf_pos = sf_slot * Int32(self.sf_uint32_per_token) + word
                    stg_b32_raw(sf_addr + Int64(sf_pos) * Int64(4), sf_words[sf_round])
        if lane_idx == Int32(0):
            weight_addr = args.peer_rank_ptr_mapper.map(
                args.fc1_input_topk_weights_buffer.iterator.toint(), owner_rank,
                Int64(pool_slot) * Int64(4),
            )
            stg_b32_raw(weight_addr, ldg_b32_raw(
                message_addr + Int64(self.relay_message_topk_weights_offset)
                + Int64(topk_slot) * Int64(4),
            ))
            md_addr = args.peer_rank_ptr_mapper.map(
                args.token_src_metadata.iterator.toint(), owner_rank,
                Int64(pool_slot) * Int64(TokenSrcMetadata.nbytes),
            )
            wire_md = TokenSrcMetadata.load(
                message_addr + Int64(self.relay_message_src_metadata_offset),
            )
            TokenSrcMetadata(
                src_rank=wire_md.src_rank, src_token=wire_md.src_token, src_topk=topk_slot,
            ).store(cute.make_ptr(Uint8, md_addr, AddressSpace.gmem, assumed_align=8))
        if cutlass.const_expr(self.pipeline_timing_enabled):
            copy_ns = read_globaltimer() - copy_start
        return ready_addr, route_ns, copy_ns

    @cute.jit
    def relay_fanout(
        self,
        token_comm_storage,
        internode_recv_splits,
        relay_fanout_reservation,
        relay_fanout_slot,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        gin_window,
        pipeline_timing,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_sms,
        direct_pool_args=None,
        window_base=None,
    ):
        """Wait for Stage 1 and NVLink-fan out each landed message."""
        if cutlass.const_expr(
            self.enable_nccl_gin
            and relay_fanout_reservation is not None
            and relay_fanout_slot is not None
        ):
            from nccl.core.device import cute as nccl_cute

            relay_warp_idx = warp_idx - Int32(self.num_dispatch_warps)
            relay_thread_idx = (
                relay_warp_idx * Int32(self.warp_threads) + lane_idx
            )
            gin = self._device_gin_gpu()
            relay_coop = nccl_cute.warp_span(
                warp0=self.relay_warp_start,
                n_warps=self.num_relay_warps,
                id=1,
            )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                fanout_wait_start = read_globaltimer()

            # One CTA advances each monotonic shadow target. All other CTAs are
            # held at the relay-only grid sync until every source nodes strong
            # expert-hit credit certifies its Stage 1 puts (Invariant I5/I8).
            if sm_idx == Int32(0):
                if cutlass.const_expr(self.iket_enabled):
                    dispatch_signal_range = _iket.range_start(
                        "megamoe_wait", IKET_MEGA_WAIT_DISPATCH_SIGNAL
                    )
                for src_slot_off in cutlass.range_constexpr(
                    0, self.num_remote_node_slots, 1
                ):
                    src_slot = Int32(src_slot_off)
                    signal_id = Int32(
                        self.relay_node_signal_base + src_slot_off
                    )
                    shadow = gin.signal_shadow_pointer(signal=signal_id)
                    launch_credit = Int32(0)
                    if relay_coop.thread_rank == Int32(0):
                        for g_off in cutlass.range_constexpr(
                            0, self.node_experts, 1
                        ):
                            launch_credit = (
                                launch_credit
                                + internode_recv_splits[src_slot, g_off]
                            )
                        cute.arch.atomic_add(
                            shadow,
                            cutlass.Uint64(launch_credit),
                            sem="relaxed",
                            scope="gpu",
                        )
                    # Relay-only barrier id: with the early push the dispatch
                    # group of this CTA is still inside inter_meta_exchange and
                    # uses named barrier 10 with dispatch_group_threads (Step
                    # 5b CTA barrier, grid syncs); sharing that id with a
                    # 128-thread relay barrier is undefined and hung the
                    # staged perf runs 3059254/3060429/3060636 once the pack
                    # prefetch and lane layout shifted the timing.
                    cute.arch.barrier(
                        barrier_id=self.relay_early_push_bar_id,
                        number_of_threads=self.num_relay_threads,
                    )
                    target = cute.arch.load(
                        shadow,
                        cutlass.Uint64,
                        sem="relaxed",
                        scope="gpu",
                    )
                    gin.wait_signal(
                        relay_coop,
                        signal=signal_id,
                        least=target,
                        bits=64,
                        ord=nccl_cute.MemoryOrder.ACQUIRE,
                    )
                    if cutlass.const_expr(
                        self.pipeline_timing_enabled and src_slot_off < 3
                    ):
                        # Sequential credit waits: the end of slot s's wait
                        # is max(arrival of slots 0..s); the first three
                        # slots cover N2R2 (one) and N4R8 (three).
                        fanout_wait_slot_end = read_globaltimer()
                        if relay_warp_idx == Int32(0):
                            if lane_idx == Int32(0):
                                pipeline_timing[
                                    sm_idx, warp_idx,
                                    PIPELINE_STAGE_FANOUT_WAIT_SLOT0 + src_slot_off, 0
                                ] = fanout_wait_start
                                pipeline_timing[
                                    sm_idx, warp_idx,
                                    PIPELINE_STAGE_FANOUT_WAIT_SLOT0 + src_slot_off, 1
                                ] = fanout_wait_slot_end
                if cutlass.const_expr(self.iket_enabled):
                    _iket.range_end(
                        dispatch_signal_range,
                        IKET_MEGA_WAIT_DISPATCH_SIGNAL,
                    )

            if cutlass.const_expr(self.pipeline_timing_enabled):
                fanout_wait_end = read_globaltimer()
            if cutlass.const_expr(self.iket_enabled):
                relay_fanout_grid_sync_range = _iket.range_start(
                    "megamoe_grid_sync_wait", IKET_MEGA_GRID_SYNC_WAIT_RELAY_FANOUT
                )
            software_grid_sync(
                grid_sync_counter.iterator + _RELAY_GRID_SYNC_SLOT,
                sm_idx,
                num_sms,
                relay_thread_idx,
                num_threads=self.num_relay_threads,
            )
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(
                    relay_fanout_grid_sync_range,
                    IKET_MEGA_GRID_SYNC_WAIT_RELAY_FANOUT,
                )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                # SM0 sub-envelopes of the fan-out prologue: the wait for the
                # remote strong credits and the relay grid sync that releases
                # every CTA's fan-out.
                fanout_sync_end = read_globaltimer()
                if sm_idx == Int32(0):
                    if relay_warp_idx == Int32(0):
                        if lane_idx == Int32(0):
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_WAIT, 0
                            ] = fanout_wait_start
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_WAIT, 1
                            ] = fanout_wait_end
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_SYNC, 0
                            ] = fanout_wait_end
                            pipeline_timing[
                                sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_SYNC, 1
                            ] = fanout_sync_end

            if cutlass.const_expr(window_base is not None):
                # Hoisted base (field 47): pure address arithmetic instead of
                # the out-of-line window lookup on every relay warp.
                window_u8 = cute.make_tensor(
                    cute.make_ptr(
                        Uint8, window_base, AddressSpace.gmem, assumed_align=16,
                    ),
                    cute.make_layout((self.gin_window_bytes,), stride=(1,)),
                )
            else:
                window_u8 = gin_window.tensor(
                    Uint8,
                    cute.make_layout((self.gin_window_bytes,), stride=(1,)),
                )
            my_node = Int32(local_rank) // Int32(self.ranks_per_node)
            rank_in_node = (
                Int32(local_rank) % Int32(self.ranks_per_node)
            )
            node_base_rank = my_node * Int32(self.ranks_per_node)
            num_global_relay_warps: cutlass.Constexpr[int] = (
                num_sms * self.num_relay_warps
            )

            if cutlass.const_expr(self.iket_enabled):
                relay_fanout_range = _iket.range_start(
                    "megamoe_envelope", IKET_MEGA_ENVELOPE_RELAY_FANOUT
                )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                # Worker-local phase sums over every message this warp
                # handles, anchored at the warp's fan-out loop start so the
                # host reducer's envelope length is the summed phase time of
                # the busiest relay warp (attribution only, not a lifecycle).
                fan_loop_t0 = read_globaltimer()
                fan_route_ns = Int64(0)
                fan_copy_ns = Int64(0)
                fan_pub_ns = Int64(0)
            global_relay_warp = (
                sm_idx * Int32(self.num_relay_warps) + relay_warp_idx
            )
            # Field 46 slot pointer, taken outside the message loop: a
            # cute.struct object cannot be captured by the dynamic while, only
            # the pointer derived from it can (exact gates 3052244-3052247).
            fan_ready_smem = token_comm_storage.smem_fanout_ready.data_ptr()
            # Messages of earlier source slots already consumed by the
            # flattened stride; the residue selects this slot's first
            # message for this warp.
            flat_slot_prefix = Int32(0)
            for src_slot_off in cutlass.range_constexpr(
                0, self.num_remote_node_slots, 1
            ):
                src_slot = Int32(src_slot_off)
                num_messages = internode_recv_splits[
                    src_slot, self.node_experts
                ]
                # Flattened index m = prefix + message_slot is served by
                # warp m % W, so message_slot = (warp - prefix) mod W.
                message_slot = global_relay_warp - (
                    flat_slot_prefix % Int32(num_global_relay_warps)
                )
                if message_slot < Int32(0):
                    message_slot = message_slot + Int32(
                        num_global_relay_warps
                    )
                flat_slot_prefix = flat_slot_prefix + num_messages
                fan_loop_end = num_messages
                while message_slot < fan_loop_end:
                    fan_has_message = message_slot < num_messages
                    h_ready_0 = Int64(0)
                    h_ready_1 = Int64(0)
                    h_ready_2 = Int64(0)
                    h_ready_3 = Int64(0)
                    if fan_has_message:
                        if cutlass.const_expr(self.pipeline_timing_enabled):
                            fan_t0 = read_globaltimer()
                        if cutlass.const_expr(self.iket_detailed_ranges):
                            fanout_message_range = _iket.range_start(
                                "megamoe_work", IKET_MEGA_WORK_FANOUT_MESSAGE
                            )
                        recv_byte_offset = Int64(
                            self.relay_recv_buffer_offset
                        ) + (
                            Int64(src_slot)
                            * Int64(self.max_relay_tokens_per_node)
                            + Int64(message_slot)
                        ) * Int64(self.relay_message_bytes)
                        recv_addr = window_u8.iterator.toint() + recv_byte_offset

                        packed_route_experts = Int64(-1)
                        if lane_idx == Int32(0):
                            packed_route_experts = ldg_route_experts_b128_raw(
                                recv_addr + Int64(self.relay_message_route_words_offset),
                            )
                        # Four wire routes are already a GIN constructor
                        # requirement. Their independent reservations share
                        # one SIMT issue sequence instead of four serial ones.
                        route_lo = cute.arch.shuffle_sync(
                            Int32(packed_route_experts & Int64(0xFFFFFFFF)), Int32(0),
                        )
                        route_hi = cute.arch.shuffle_sync(
                            Int32(packed_route_experts >> Int64(32)), Int32(0),
                        )
                        lane_route_word = route_lo
                        if lane_idx >= Int32(2):
                            lane_route_word = route_hi
                        lane_expert = (
                            lane_route_word >> ((lane_idx & Int32(1)) * Int32(16))
                        ) & Int32(0xFFFF)
                        lane_pool_slot, lane_sf_slot = self._direct_fanout_reserve(
                            direct_pool_args, src_slot, message_slot,
                            lane_expert, node_base_rank, rank_in_node, lane_idx,
                        )
                        if cutlass.const_expr(self.pipeline_timing_enabled):
                            fan_route_ns = fan_route_ns + (read_globaltimer() - fan_t0)
                        ready_addresses = []
                        for topk_off in cutlass.range_constexpr(0, 4, 1):
                            node_expert = cute.arch.shuffle_sync(
                                lane_expert, Int32(topk_off),
                            )
                            pool_slot = cute.arch.shuffle_sync(
                                lane_pool_slot, Int32(topk_off),
                            )
                            sf_slot = cute.arch.shuffle_sync(
                                lane_sf_slot, Int32(topk_off),
                            )
                            ready_addr = Int64(0)
                            if node_expert != Int32(0xFFFF):
                                if cutlass.const_expr(self.direct_l1_sf_staging):
                                    # Experimental staged scale factors: the SF
                                    # row lands contiguously in the owner's
                                    # staging area and the hit raises the row
                                    # flag instead of the tile credit.
                                    ready_addr, route_ns, copy_ns = self._direct_fanout_hit_sfs(
                                        direct_pool_args, recv_addr, src_slot, message_slot,
                                        Int32(topk_off), node_expert, node_base_rank, lane_idx,
                                        pool_slot, sf_slot,
                                    )
                                else:
                                    ready_addr, route_ns, copy_ns = self._direct_fanout_hit(
                                        direct_pool_args, recv_addr, src_slot, message_slot,
                                        Int32(topk_off), node_expert, node_base_rank, lane_idx,
                                        pool_slot, sf_slot,
                                    )
                                if cutlass.const_expr(self.pipeline_timing_enabled):
                                    fan_route_ns = fan_route_ns + route_ns
                                    fan_copy_ns = fan_copy_ns + copy_ns
                            ready_addresses.append(ready_addr)
                        if cutlass.const_expr(self.pipeline_timing_enabled):
                            publish_start = read_globaltimer()
                        # Cumulative release: the warp sync orders every
                        # writer before lane 0's single system fence; the
                        # strong relaxed adds that follow synchronize with the
                        # owner's ld.acquire.sys polls, so no ready flag needs
                        # a release fence of its own.
                        cute.arch.sync_warp()
                        if lane_idx == Int32(0):
                            cute.arch.fence_acq_rel_sys()
                            for topk_off in cutlass.range_constexpr(0, 4, 1):
                                if ready_addresses[topk_off] != Int64(0):
                                    red_add_relaxed_sys_s32_raw(
                                        ready_addresses[topk_off], Int32(1),
                                    )
                        cute.arch.sync_warp()
                        if cutlass.const_expr(self.pipeline_timing_enabled):
                            fan_pub_ns = fan_pub_ns + (read_globaltimer() - publish_start)
                        if cutlass.const_expr(self.iket_detailed_ranges):
                            _iket.range_end(
                                fanout_message_range,
                                IKET_MEGA_WORK_FANOUT_MESSAGE,
                            )
                    message_slot = message_slot + Int32(
                        num_global_relay_warps
                    )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                if lane_idx == Int32(0):
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_ROUTE, 0
                    ] = fan_loop_t0
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_ROUTE, 1
                    ] = fan_loop_t0 + fan_route_ns
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_COPY, 0
                    ] = fan_loop_t0
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_COPY, 1
                    ] = fan_loop_t0 + fan_copy_ns
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_PUBLISH, 0
                    ] = fan_loop_t0
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_FANOUT_PUBLISH, 1
                    ] = fan_loop_t0 + fan_pub_ns
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(
                    relay_fanout_range, IKET_MEGA_ENVELOPE_RELAY_FANOUT
                )

    @cute.jit
    def dispatch_barrier(
        self,
        expert_send_count,
        expert_recv_count,
        expert_recv_count_sum,
        nvlink_barrier_signal,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_sms,
        nvlink_barrier_counter,
        dispatch_only: cutlass.Constexpr[bool] = False,
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx
        # Field 60: the relay warps skip this barrier, so the dispatch warps
        # synchronize as their own 128-thread group on a dedicated named
        # barrier id (the relay warps use barrier 11 at the same time).
        # With no relay warps every warp of the dispatch group is a dispatch warp (main's shape);
        # the warp test exists only for the internode kernel's relay warps.
        dispatch_warps_only = True
        if cutlass.const_expr(self.num_relay_warps > 0):
            dispatch_warps_only = warp_idx < Int32(self.num_dispatch_warps)
        barrier_threads: cutlass.Constexpr[int] = (
            self.num_dispatch_threads if dispatch_only
            else self.dispatch_group_threads
        )
        barrier_bar_id: cutlass.Constexpr[int] = (
            self.dispatch_only_bar_id if dispatch_only
            else self.dispatch_intra_cta_bar_id
        )

        if cutlass.const_expr(self.iket_enabled):
            dispatch_grid_sync_range = _iket.range_start(
                "megamoe_grid_sync_wait", IKET_MEGA_GRID_SYNC_WAIT_DISPATCH
            )
        software_grid_sync(
            grid_sync_counter, sm_idx, num_sms, tid_in_group,
            num_threads=barrier_threads, barrier_id=barrier_bar_id,
        )
        if cutlass.const_expr(self.iket_enabled):
            _iket.range_end(
                dispatch_grid_sync_range, IKET_MEGA_GRID_SYNC_WAIT_DISPATCH
            )

        if sm_idx == 0:
            if dispatch_warps_only:
                for offset in cutlass.range_constexpr(
                    0, self.num_total_experts, self.experts_per_dispatch_pass,
                ):
                    expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
                    if expert_id < Int32(self.num_total_experts):
                        dst_rank = expert_id // Int32(self.num_experts_per_rank)
                        dst_local_expert = expert_id % Int32(self.num_experts_per_rank)
                        status_u64 = cute.arch.load(
                            expert_send_count.iterator + expert_id,
                            Int64,
                            sem="relaxed",
                            scope="gpu",
                        )
                        token_count_u32 = Int32(status_u64 & Int64(0xFFFFFFFF))
                        dst_node_delta = Int32(0)
                        if cutlass.const_expr(self.num_remote_nodes > 0):
                            my_node = Int32(local_rank) // Int32(self.ranks_per_node)
                            dst_node = dst_rank // Int32(self.ranks_per_node)
                            dst_node_delta = dst_node - my_node
                        if dst_node_delta == Int32(0):
                            erc_local_base = expert_recv_count.iterator.toint()
                            erc_elem_off = (
                                Int32(local_rank)
                                * Int32(self.num_experts_per_rank)
                                + dst_local_expert
                            ) * Int32(8)
                            erc_peer_addr = peer_rank_ptr_mapper.map(
                                erc_local_base, dst_rank, Int64(erc_elem_off),
                            )
                            stg_b64_raw(erc_peer_addr, Int64(token_count_u32))
                            ercs_local_base = (
                                expert_recv_count_sum.iterator.toint()
                            )
                            ercs_peer_addr = peer_rank_ptr_mapper.map(
                                ercs_local_base, dst_rank,
                                Int64(dst_local_expert * Int32(8)),
                            )
                            red_add_relaxed_sys_u64_raw(
                                ercs_peer_addr, status_u64,
                            )
                cute.arch.fence_acq_rel_sys()
        cute.arch.barrier(
            barrier_id=barrier_bar_id,
            number_of_threads=barrier_threads,
        )

        self.nvlink_barrier(
            nvlink_barrier_signal,
            nvlink_barrier_counter,
            grid_sync_counter,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            num_sms=num_sms,
            local_rank=local_rank,
            sync_num_threads=barrier_threads,
            prologue_grid_sync=False,
            epilogue_grid_sync=True,
            sync_barrier_id=barrier_bar_id,
        )

    @cute.jit
    def _direct_fanout_hit_sfs(
        self, args, message_addr, src_slot, message_slot, topk_slot,
        node_expert, node_base_rank, lane_idx, pool_slot, sf_slot,
    ):
        """Field 61 form of ``_direct_fanout_hit``: DATA, weight and metadata
        go to their pool rows as before, the scale-factor row is copied
        contiguously (16-byte lanes) into the owner's per-row staging area and
        the returned publication address is the row's ready flag instead of
        the tile readiness counter; the owner scatters and releases (I1, I7).
        """
        route_ns = Int64(0)
        copy_ns = Int64(0)
        if cutlass.const_expr(self.pipeline_timing_enabled):
            route_start = read_globaltimer()
        owner_rank = node_base_rank + node_expert // Int32(self.num_experts_per_rank)
        sf_row_bytes: cutlass.Constexpr[int] = self.sf_uint32_per_token * 4
        data_addr = args.peer_rank_ptr_mapper.map(
            args.fc1_input_token_buffer.iterator.toint(), owner_rank,
            Int64(pool_slot) * Int64(self.hidden_bytes),
        )
        staging_addr = args.peer_rank_ptr_mapper.map(
            args.relay_sf_staging.iterator.toint(), owner_rank,
            Int64(pool_slot) * Int64(sf_row_bytes),
        )
        ready_addr = args.peer_rank_ptr_mapper.map(
            args.relay_row_ready.iterator.toint(), owner_rank,
            Int64(pool_slot) * Int64(4),
        )
        if cutlass.const_expr(self.pipeline_timing_enabled):
            copy_start = read_globaltimer()
            route_ns = copy_start - route_start
        self._copy_rows_unrolled(
            data_addr, message_addr + Int64(self.relay_message_token_data_offset),
            lane_idx, total_bytes=self.hidden_bytes,
        )
        # One 16-byte vector per lane for the contiguous SF row (28 lanes at
        # H=7168) instead of 112 scalar stores into 512-byte-strided atoms.
        if lane_idx < Int32(sf_row_bytes // 16):
            copy_b128_raw(
                staging_addr + Int64(lane_idx) * Int64(16),
                message_addr + Int64(self.relay_message_sf_data_offset)
                + Int64(lane_idx) * Int64(16),
            )
        if lane_idx == Int32(0):
            weight_addr = args.peer_rank_ptr_mapper.map(
                args.fc1_input_topk_weights_buffer.iterator.toint(), owner_rank,
                Int64(pool_slot) * Int64(4),
            )
            stg_b32_raw(weight_addr, ldg_b32_raw(
                message_addr + Int64(self.relay_message_topk_weights_offset)
                + Int64(topk_slot) * Int64(4),
            ))
            md_addr = args.peer_rank_ptr_mapper.map(
                args.token_src_metadata.iterator.toint(), owner_rank,
                Int64(pool_slot) * Int64(TokenSrcMetadata.nbytes),
            )
            wire_md = TokenSrcMetadata.load(
                message_addr + Int64(self.relay_message_src_metadata_offset),
            )
            TokenSrcMetadata(
                src_rank=wire_md.src_rank, src_token=wire_md.src_token, src_topk=topk_slot,
            ).store(cute.make_ptr(Uint8, md_addr, AddressSpace.gmem, assumed_align=8))
        if cutlass.const_expr(self.pipeline_timing_enabled):
            copy_ns = read_globaltimer() - copy_start
        return ready_addr, route_ns, copy_ns

    @cute.jit
    def relay_sf_materialize(
        self,
        relay_row_ready,
        relay_sf_staging,
        internode_recv_count_per_source,
        relay_pool_base,
        relay_sf_pool_base,
        fc1_input_sf_buffer,
        fc1_ready_counter,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
        pipeline_timing=None,
    ):
        """Owner side of field 61: scatter staged relay SF rows, release rows.

        Every relay row of this rank (all experts x source slots, Step 5c
        bases and counts) is served by dispatch warp ``m mod W`` of its
        flattened index m. The warp acquires the row's ready flag (written by
        the relay peer after its system fence), reads the contiguous staged
        row, writes it into the swizzled SF atoms at the row's absolute SF
        slot (I10) and releases one credit on the row's static tile counter
        (I7); the intra rows keep their dispatch_pull credits.
        """
        if cutlass.const_expr(
            self.enable_nccl_gin
            and self.direct_l1_sf_staging
            and relay_row_ready is not None
            and relay_sf_staging is not None
        ):
            if cutlass.const_expr(self.pipeline_timing_enabled):
                sfm_t0 = read_globaltimer()
            global_dispatch_warp = (
                sm_idx * Int32(self.num_dispatch_warps) + warp_idx
            )
            num_global_dispatch_warps: cutlass.Constexpr[int] = (
                num_sms * self.num_dispatch_warps
            )
            sf_words: cutlass.Constexpr[int] = self.sf_uint32_per_token
            sf_row_bytes: cutlass.Constexpr[int] = sf_words * 4
            sf_vectors: cutlass.Constexpr[int] = sf_row_bytes // 16
            num_buckets: cutlass.Constexpr[int] = (
                self.num_experts_per_rank * self.num_remote_node_slots
            )
            staging_base = relay_sf_staging.iterator.toint()
            sf_buffer_base = fc1_input_sf_buffer.iterator.toint()
            lane_vec_valid = Int32(lane_idx < Int32(sf_vectors))
            # Bucket table in lanes: lane l of round r holds bucket r*32 + l
            # (count, DATA base, SF base, exclusive row prefix). One load burst
            # instead of a serial walk over every bucket (the staged-mode
            # relay_copy paid about 1.2 us per bucket for that walk; diag
            # 3118224 measured the same 45 us here at 36 buckets).
            bucket_rounds: cutlass.Constexpr[int] = (num_buckets + 31) // 32
            lane_cnt = []
            lane_db = []
            lane_sb = []
            lane_ex = []
            total_rows = Int32(0)
            for bucket_round in cutlass.range_constexpr(0, bucket_rounds, 1):
                bucket = Int32(bucket_round * 32) + lane_idx
                bucket_c = cutlass.min(bucket, Int32(num_buckets - 1))
                expert_id = bucket_c // Int32(self.num_remote_node_slots)
                src_slot = bucket_c % Int32(self.num_remote_node_slots)
                cnt = Int32(0)
                if bucket < Int32(num_buckets):
                    cnt = Int32(internode_recv_count_per_source[src_slot, expert_id])
                db = Int32(relay_pool_base[expert_id, src_slot])
                sb = Int32(relay_sf_pool_base[expert_id, src_slot])
                ex = total_rows
                for scan_lane in cutlass.range_constexpr(0, 32, 1):
                    scan_cnt = cute.arch.shuffle_sync(cnt, Int32(scan_lane))
                    if lane_idx > Int32(scan_lane):
                        ex = ex + scan_cnt
                    total_rows = total_rows + scan_cnt
                lane_cnt.append(cnt)
                lane_db.append(db)
                lane_sb.append(sb)
                lane_ex.append(ex)
            # Flattened row f is served by dispatch warp f mod W; its bucket
            # is the lane whose [prefix, prefix + count) range contains f.
            flat_row = global_dispatch_warp
            while flat_row < total_rows:
                pool_slot = Int32(0)
                sf_slot = Int32(0)
                for bucket_round in cutlass.range_constexpr(0, bucket_rounds, 1):
                    in_bucket = Int32(0)
                    if flat_row >= lane_ex[bucket_round]:
                        if flat_row < lane_ex[bucket_round] + lane_cnt[bucket_round]:
                            in_bucket = Int32(1)
                    hit_mask = cute.arch.vote_ballot_sync(in_bucket != Int32(0))
                    if hit_mask != Int32(0):
                        hit_lane = fns_b32(hit_mask, Int32(0), Int32(1))
                        row_in_bucket = flat_row - cute.arch.shuffle_sync(
                            lane_ex[bucket_round], hit_lane
                        )
                        pool_slot = cute.arch.shuffle_sync(
                            lane_db[bucket_round], hit_lane
                        ) + row_in_bucket
                        sf_slot = cute.arch.shuffle_sync(
                            lane_sb[bucket_round], hit_lane
                        ) + row_in_bucket
                # The relay peer publishes the flag after its system
                # fence over the staged row (release pattern); acquire
                # at system scope before reading the staged bytes.
                spin_wait(
                    relay_row_ready.iterator + pool_slot,
                    lambda v: v >= Int32(1),
                    fail_sleep_cycles=200,
                    scope="sys",
                )
                row_addr = (
                    staging_base
                    + Int64(pool_slot) * Int64(sf_row_bytes)
                    + Int64(lane_idx) * Int64(16)
                )
                vec_lo = ldg_b64_predicated_raw(row_addr, lane_vec_valid)
                vec_hi = ldg_b64_predicated_raw(
                    row_addr + Int64(8), lane_vec_valid
                )
                if lane_idx < Int32(sf_vectors):
                    word_base = lane_idx * Int32(4)
                    for word_k in cutlass.range_constexpr(0, 4, 1):
                        if cutlass.const_expr(word_k < 2):
                            word_val = Int32(
                                (vec_lo >> Int64(32 * word_k)) & Int64(0xFFFFFFFF)
                            )
                        else:
                            word_val = Int32(
                                (vec_hi >> Int64(32 * (word_k - 2)))
                                & Int64(0xFFFFFFFF)
                            )
                        if cutlass.const_expr(self.sf_atom_swizzled):
                            sf_pos = sf_atom_int32_offset(
                                sf_slot, word_base + Int32(word_k),
                                num_k_atoms=sf_words,
                            )
                        else:
                            sf_pos = (
                                sf_slot * Int32(sf_words)
                                + word_base + Int32(word_k)
                            )
                        stg_b32_raw(
                            sf_buffer_base + Int64(sf_pos) * Int64(4), word_val,
                        )
                # Local stores by every lane, then one GPU-scope release
                # of the row's static tile credit (readers are this
                # rank's scheduler and TMA-B waits; the counter has no
                # remote writer in this mode).
                cute.arch.fence_acq_rel_gpu()
                cute.arch.sync_warp()
                if lane_idx == Int32(0):
                    red_add_release_gpu_s32(
                        fc1_ready_counter.iterator
                        + pool_slot // Int32(self.cluster_tile_tokens),
                        Int32(1),
                    )
                flat_row = flat_row + Int32(num_global_dispatch_warps)
            if cutlass.const_expr(self.pipeline_timing_enabled):
                sfm_end = read_globaltimer()
                if lane_idx == Int32(0):
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_SF_MATERIALIZE, 0
                    ] = sfm_t0
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_SF_MATERIALIZE, 1
                    ] = sfm_end

    @cute.jit
    def inter_meta_exchange(
        self,
        expert_recv_count,
        expert_recv_count_sum,
        internode_send_count,
        internode_recv_splits,
        internode_send_token_count,
        intra_token_count,
        internode_recv_count_per_source,
        expert_pool_base,
        expert_sf_pool_base,
        relay_pool_base,
        relay_sf_pool_base,
        relay_direct_bucket_meta,
        internode_recv_count_per_source_snapshot,
        expert_recv_count_snapshot,
        expert_recv_count_sum_snapshot,
        intra_token_count_snapshot,
        expert_pool_base_snapshot,
        expert_sf_pool_base_snapshot,
        relay_pool_base_snapshot,
        relay_sf_pool_base_snapshot,
        expert_scheduled_count_snapshot,
        pool_end_snapshot,
        nvlink_barrier_signal,
        nvlink_barrier_counter,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        gin_window,
        pipeline_timing,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_sms,
        pool_token_capacity,
        pool_sf_capacity,
    ):
        if cutlass.const_expr(
            self.num_relay_warps > 0
            and self.num_remote_nodes > 0
            and intra_token_count is not None
            and internode_recv_count_per_source is not None
            and relay_pool_base is not None
        ):
            if cutlass.const_expr(self.iket_enabled):
                metadata_range = _iket.range_start(
                    "megamoe_envelope", IKET_MEGA_ENVELOPE_META
                )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                meta_t0 = read_globaltimer()
                meta_t_step3 = Int64(0)
                meta_t_wait = Int64(0)
                meta_t_redist_issue = Int64(0)
            # Capture the exact intranode publisher table after dispatch's
            # acquire barrier and before Step 5c can add relay totals.  The
            # snapshots live outside both zero prefixes and survive a normal
            # launch for per-iteration host validation.
            if cutlass.const_expr(
                expert_recv_count_snapshot is not None
                and expert_recv_count_sum_snapshot is not None
            ):
                thread_in_group = (
                    warp_idx * Int32(self.warp_threads) + lane_idx
                )
                # Field 60: only the dispatch warps passed the dispatch
                # barrier acquire that publishes this table.
                snapshot_threads: cutlass.Constexpr[int] = (
                    self.num_dispatch_threads if self.relay_skip_dispatch_barrier
                    else self.dispatch_group_threads
                )
                thread_linear = (
                    sm_idx * Int32(snapshot_threads)
                    + thread_in_group
                )
                thread_stride = Int32(num_sms * snapshot_threads)
                recv_count_total: cutlass.Constexpr[int] = (
                    self.world_size * self.num_experts_per_rank
                )
                snapshot_idx = thread_linear
                if cutlass.const_expr(self.relay_skip_dispatch_barrier):
                    if warp_idx >= Int32(self.num_dispatch_warps):
                        snapshot_idx = Int32(recv_count_total)
                while snapshot_idx < Int32(recv_count_total):
                    snapshot_value = cute.arch.load(
                        expert_recv_count.iterator + snapshot_idx,
                        Int64,
                        sem="relaxed",
                        scope="gpu",
                    )
                    (
                        expert_recv_count_snapshot.iterator + snapshot_idx
                    ).store(snapshot_value)
                    snapshot_idx = snapshot_idx + thread_stride

            # Step 2: capture the intra-only boundary before any relay count can
            # be published. One SM0 thread owns this snapshot on each rank.
            if (
                sm_idx == Int32(0)
                and warp_idx == Int32(0)
                and lane_idx == Int32(0)
            ):
                for expert_off in cutlass.range_constexpr(
                    0, self.num_experts_per_rank, 1
                ):
                    expert_id = Int32(expert_off)
                    packed_count = cute.arch.load(
                        expert_recv_count_sum.iterator + expert_id,
                        Int64,
                        sem="relaxed",
                        scope="gpu",
                    )
                    if cutlass.const_expr(
                        expert_recv_count_sum_snapshot is not None
                    ):
                        expert_recv_count_sum_snapshot[expert_id] = (
                            packed_count
                        )
                    intra_token_count[expert_id] = Int32(
                        packed_count & Int64(0xFFFFFFFF)
                    )
                    if cutlass.const_expr(
                        intra_token_count_snapshot is not None
                    ):
                        intra_token_count_snapshot[expert_id] = Int32(
                            packed_count & Int64(0xFFFFFFFF)
                        )

            if cutlass.const_expr(
                self.enable_nccl_gin
                and internode_send_count is not None
                and internode_recv_splits is not None
                and gin_window is not None
            ):
                if cutlass.const_expr(not self.step3_put_before_stage1):
                    # Step 3, lane-parallel form: lanes 0..slots-1 of SM0 warp 0
                    # each post one remote rail peer's metadata vector from the
                    # same instruction stream; lane 0 then waits in Step 5a.
                    # (early Stage 1 (field 60) posts it from the Stage 1 issuer instead.)
                    if sm_idx == Int32(0) and warp_idx == Int32(0):
                        from nccl.core.device import cute as nccl_cute

                        # Handles and views are created by the whole warp so
                        # the lane-predicated region below only consumes them.
                        lane_gin = self._device_gin_gpu()
                        lane_team_rail = self._team_rail_value
                        lane_coop = nccl_cute.thread()
                        lane_my_node = Int32(local_rank) // Int32(self.ranks_per_node)
                        lane_row_layout = cute.make_layout(
                            (self.gin_send_count_stride,), stride=(1,)
                        )
                        lane_window_i32 = gin_window.tensor(
                            Int32,
                            cute.make_layout(
                                (self.gin_metadata_window_i32,), stride=(1,)
                            ),
                        )
                        if lane_idx < Int32(self.num_remote_node_slots):
                            lane_dst_slot = lane_idx
                            lane_dst_node = lane_dst_slot
                            if lane_dst_slot >= lane_my_node:
                                lane_dst_node = lane_dst_slot + Int32(1)
                            lane_src_slot_at_dst = lane_my_node
                            if lane_my_node > lane_dst_node:
                                lane_src_slot_at_dst = lane_my_node - Int32(1)
                            lane_packed_token_count = cute.arch.load(
                                internode_send_token_count.iterator + lane_dst_slot,
                                Int64,
                                sem="relaxed",
                                scope="gpu",
                            )
                            lane_window_i32[
                                lane_dst_slot * Int32(self.gin_send_count_stride)
                                + Int32(self.node_experts)
                            ] = Int32(lane_packed_token_count & Int64(0xFFFFFFFF))
                            cute.arch.fence_acq_rel_sys()
                            lane_src_row = cute.make_tensor(
                                lane_window_i32.iterator
                                + lane_dst_slot * Int32(self.gin_send_count_stride),
                                lane_row_layout,
                            )
                            lane_dst_row = cute.make_tensor(
                                lane_window_i32.iterator
                                + Int32(self.gin_recv_splits_offset_i32)
                                + lane_src_slot_at_dst * Int32(self.gin_send_count_stride),
                                lane_row_layout,
                            )
                            lane_gin.put(
                                lane_team_rail,
                                lane_dst_node,
                                gin_window,
                                lane_dst_row,
                                gin_window,
                                lane_src_row,
                                lane_coop,
                                is_signal=True,
                                signal_id=(
                                    self.internode_splits_signal_base
                                    + lane_src_slot_at_dst
                                ),
                                signal_op=self.gin_signal_op_inc,
                            )
                        cute.arch.sync_warp()
                if (
                    sm_idx == Int32(0)
                    and warp_idx == Int32(0)
                    and lane_idx == Int32(0)
                ):
                    # Imported only while tracing the GIN instantiation; the
                    # intranode and LSA-only modules remain importable without
                    # nccl4py's device-CuTe package.
                    from nccl.core.device import cute as nccl_cute

                    gin = self._device_gin_gpu()
                    team_rail = self._team_rail_value
                    thread_coop = nccl_cute.thread()
                    my_node = Int32(local_rank) // Int32(self.ranks_per_node)
                    row_layout = cute.make_layout(
                        (self.gin_send_count_stride,), stride=(1,)
                    )
                    window_i32 = gin_window.tensor(
                        Int32,
                        cute.make_layout(
                            (self.gin_metadata_window_i32,), stride=(1,)
                        ),
                    )


                    if cutlass.const_expr(self.pipeline_timing_enabled):
                        meta_t_step3 = read_globaltimer()
                    # Step 5a: each compact source slot has one producer per
                    # launch. The NCCL-managed shadow carries our monotonic
                    # target; the remote signal itself is never reset normally.
                    for src_slot_off in cutlass.range_constexpr(
                        0, self.num_remote_node_slots, 1
                    ):
                        signal_id = Int32(
                            self.internode_splits_signal_base + src_slot_off
                        )
                        shadow = gin.signal_shadow_pointer(signal=signal_id)
                        old_target = cute.arch.atomic_add(
                            shadow,
                            cutlass.Uint64(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                        gin.wait_signal(
                            thread_coop,
                            signal=signal_id,
                            least=cutlass.Uint64(old_target)
                            + cutlass.Uint64(1),
                            bits=64,
                            ord=nccl_cute.MemoryOrder.ACQUIRE,
                        )
                    if cutlass.const_expr(self.pipeline_timing_enabled):
                        meta_t_wait = read_globaltimer()


                # Step 5b, parallel form. The serial form issues one
                # release-scoped system atomic per (source node, node
                # expert) from a single thread, so each element pays a
                # full system-scope drain. Here all dispatch-group threads
                # of SM0 share the (N-1) x R*E entries with relaxed system
                # atomics and publish the batch once: the trailing fence
                # plus the nvlink_barrier below give the same cross-rank
                # visibility (Invariants I2, I4). Thread 0 acquired the
                # remote splits in Step 5a; the CTA barrier orders every
                # other thread's L2-coherent reads after that acquire.
                if sm_idx == Int32(0):
                    cute.arch.barrier(
                        barrier_id=self.dispatch_intra_cta_bar_id,
                        number_of_threads=self.dispatch_group_threads,
                    )
                    node_base_rank = (
                        Int32(local_rank) // Int32(self.ranks_per_node)
                    ) * Int32(self.ranks_per_node)
                    recv_local_base = (
                        internode_recv_count_per_source.iterator.toint()
                    )
                    redistribute_entries: cutlass.Constexpr[int] = (
                        self.num_remote_node_slots * self.node_experts
                    )
                    entry = warp_idx * Int32(self.warp_threads) + lane_idx
                    while entry < Int32(redistribute_entries):
                        src_slot = entry // Int32(self.node_experts)
                        g = entry % Int32(self.node_experts)
                        count = Int32(
                            cute.arch.load(
                                internode_recv_splits.iterator
                                + (
                                    src_slot
                                    * Int32(self.gin_send_count_stride)
                                    + g
                                ),
                                Int32,
                                sem="relaxed",
                                scope="gpu",
                            )
                        )
                        if count > Int32(0):
                            owner_rank_in_node = (
                                g // Int32(self.num_experts_per_rank)
                            )
                            local_expert = (
                                g % Int32(self.num_experts_per_rank)
                            )
                            elem_off = (
                                src_slot
                                * Int32(self.num_experts_per_rank)
                                + local_expert
                            ) * Int32(4)
                            peer_addr = peer_rank_ptr_mapper.map(
                                recv_local_base,
                                node_base_rank + owner_rank_in_node,
                                Int64(elem_off),
                            )
                            red_add_relaxed_sys_s32_raw(peer_addr, count)
                        entry = entry + Int32(self.dispatch_group_threads)
                    cute.arch.fence_acq_rel_sys()

                if cutlass.const_expr(self.pipeline_timing_enabled):
                    meta_t_redist_issue = read_globaltimer()
                # Close peer redistribution over the node-local LSA arena before
                # any owner computes its relay totals and pool regions.
                self.nvlink_barrier(
                    nvlink_barrier_signal,
                    nvlink_barrier_counter,
                    grid_sync_counter,
                    peer_rank_ptr_mapper,
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    num_sms=num_sms,
                    local_rank=local_rank,
                    sync_num_threads=self.dispatch_group_threads,
                    prologue_grid_sync=True,
                    epilogue_grid_sync=True,
                )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                meta_t_redist = read_globaltimer()

            # Step 5c: materialize the full relay pool layout on every rank.
            # Relay counts are deliberately published to expert_recv_count_sum
            # at this single write site only. Until Stage 2/3 can populate and
            # release relay payload tiles, the compile-time gate remains false,
            # so both dispatch_pull's tile prefix and scheduler internal_init see
            # the same intra-only count while P(e)/relay_pool_base still reserve
            # the complete future relay space.
            if sm_idx == Int32(0) and warp_idx == Int32(0):
                # Lane e of every 32-expert chunk owns expert chunk*32 + e; inactive
                # lanes load a clamped expert with zero-masked widths so the loads and
                # the prefix sum stay branch-free (I2/I3/I7 unchanged). The data / SF
                # totals of the previous chunks are the warp-uniform base of the next
                # one, so ranks with more than 32 experts (seed EP 16 x 640 experts =
                # 40 per rank) get the same layout the single-thread form used to
                # produce (2026-10-22; the single-thread form is gone).
                lay_invalid = Int32(0)
                lay_total_data_i64 = Int64(0)
                lay_total_sf_i64 = Int64(0)
                for lay_chunk in cutlass.range_constexpr(
                    0, (self.num_experts_per_rank + 31) // 32, 1
                ):
                    lay_chunk_lanes = min(32, self.num_experts_per_rank - lay_chunk * 32)
                    lay_chunk_expert = Int32(lay_chunk * 32) + lane_idx
                    lay_expert = cutlass.min(
                        lay_chunk_expert, Int32(self.num_experts_per_rank - 1)
                    )
                    lay_active = cutlass.min(
                        cutlass.max(
                            Int32(self.num_experts_per_rank) - lay_chunk_expert, Int32(0)
                        ),
                        Int32(1),
                    )
                    lay_carry_data = Int32(lay_total_data_i64)
                    lay_carry_sf = Int32(lay_total_sf_i64)
                    lay_intra = Int32(intra_token_count[lay_expert]) * lay_active
                    if lay_intra < Int32(0):
                        lay_invalid = Int32(1)
                    lay_counts = []
                    lay_relay_total = Int32(0)
                    for remote_slot_off in cutlass.range_constexpr(
                        0, self.num_remote_node_slots, 1
                    ):
                        lay_count = Int32(
                            internode_recv_count_per_source[
                                Int32(remote_slot_off), lay_expert
                            ]
                        ) * lay_active
                        if lay_count < Int32(0):
                            lay_invalid = Int32(1)
                        lay_counts.append(lay_count)
                        lay_relay_total = lay_relay_total + lay_count
                    lay_intra_i64 = Int64(lay_intra)
                    lay_relay_i64 = Int64(lay_relay_total)
                    lay_packed = Int64(0)
                    lay_scheduled = lay_intra
                    if cutlass.const_expr(self.publish_relay_counts_to_gemm):
                        lay_packed = cute.arch.load(
                            expert_recv_count_sum.iterator + lay_expert,
                            Int64,
                            sem="relaxed",
                            scope="gpu",
                        )
                        lay_scheduled = lay_intra + lay_relay_total
                    lay_intra_padded = (
                        (lay_intra + Int32(self.cluster_tile_tokens) - Int32(1))
                        // Int32(self.cluster_tile_tokens)
                    ) * Int32(self.cluster_tile_tokens)
                    lay_intra_sf_padded = (
                        (lay_intra + Int32(self.sf_padding_block) - Int32(1))
                        // Int32(self.sf_padding_block)
                    ) * Int32(self.sf_padding_block)
                    lay_data_width_i64 = (
                        (lay_intra_i64 + Int64(self.cluster_tile_tokens) - Int64(1))
                        // Int64(self.cluster_tile_tokens)
                        + (lay_relay_i64 + Int64(self.cluster_tile_tokens) - Int64(1))
                        // Int64(self.cluster_tile_tokens)
                    ) * Int64(self.cluster_tile_tokens)
                    lay_sf_width_i64 = (
                        (lay_intra_i64 + Int64(self.sf_padding_block) - Int64(1))
                        // Int64(self.sf_padding_block)
                        + (lay_relay_i64 + Int64(self.sf_padding_block) - Int64(1))
                        // Int64(self.sf_padding_block)
                    ) * Int64(self.sf_padding_block)
                    if cutlass.const_expr(self.publish_relay_counts_to_gemm):
                        if merged_token_segment(
                            lay_intra_i64, lay_relay_i64,
                            self.cluster_tile_tokens,
                        ):
                            lay_intra_padded = lay_intra
                            lay_intra_sf_padded = lay_intra
                            lay_total_i64 = lay_intra_i64 + lay_relay_i64
                            lay_data_width_i64 = (
                                (lay_total_i64 + Int64(self.cluster_tile_tokens - 1))
                                // Int64(self.cluster_tile_tokens)
                            ) * Int64(self.cluster_tile_tokens)
                            lay_sf_width_i64 = (
                                (lay_total_i64 + Int64(self.sf_padding_block - 1))
                                // Int64(self.sf_padding_block)
                            ) * Int64(self.sf_padding_block)
                    if (
                        lay_data_width_i64 > Int64(0x7FFFFFFF)
                        or lay_sf_width_i64 > Int64(0x7FFFFFFF)
                    ):
                        lay_invalid = Int32(1)
                    lay_data_width = Int32(lay_data_width_i64)
                    lay_sf_width = Int32(lay_sf_width_i64)
                    # Exclusive prefix over experts by constant-index shuffles.
                    lay_pool_base = lay_carry_data
                    lay_sf_pool_base = lay_carry_sf
                    for src_lane in cutlass.range_constexpr(0, lay_chunk_lanes, 1):
                        lay_w = cute.arch.shuffle_sync(lay_data_width, Int32(src_lane))
                        lay_sw = cute.arch.shuffle_sync(lay_sf_width, Int32(src_lane))
                        lay_before = cutlass.min(
                            cutlass.max(lane_idx - Int32(src_lane), Int32(0)), Int32(1)
                        )
                        lay_pool_base = lay_pool_base + lay_w * lay_before
                        lay_sf_pool_base = lay_sf_pool_base + lay_sw * lay_before
                        lay_total_data_i64 = lay_total_data_i64 + Int64(lay_w)
                        lay_total_sf_i64 = lay_total_sf_i64 + Int64(lay_sw)
                    if (
                        lay_total_data_i64 > Int64(0x7FFFFFFF)
                        or lay_total_sf_i64 > Int64(0x7FFFFFFF)
                    ):
                        lay_invalid = Int32(1)
                    if cutlass.const_expr(
                        pool_token_capacity > 0 and pool_sf_capacity > 0
                    ):
                        if (
                            lay_total_data_i64 > Int64(pool_token_capacity)
                            or lay_total_sf_i64 > Int64(pool_sf_capacity)
                        ):
                            lay_invalid = Int32(1)
                    if lay_active != Int32(0):
                        for remote_slot_off in cutlass.range_constexpr(
                            0, self.num_remote_node_slots, 1
                        ):
                            internode_recv_count_per_source_snapshot[
                                Int32(remote_slot_off), lay_expert
                            ] = lay_counts[remote_slot_off]
                        if cutlass.const_expr(self.publish_relay_counts_to_gemm):
                            expert_recv_count_sum[lay_expert] = (
                                lay_packed + Int64(lay_relay_total)
                            )
                        expert_scheduled_count_snapshot[lay_expert] = lay_scheduled
                        expert_pool_base[lay_expert] = lay_pool_base
                        expert_sf_pool_base[lay_expert] = lay_sf_pool_base
                        if cutlass.const_expr(
                            expert_pool_base_snapshot is not None
                            and expert_sf_pool_base_snapshot is not None
                        ):
                            expert_pool_base_snapshot[lay_expert] = lay_pool_base
                            expert_sf_pool_base_snapshot[lay_expert] = lay_sf_pool_base
                        lay_relay_prefix = Int32(0)
                        for remote_slot_off in cutlass.range_constexpr(
                            0, self.num_remote_node_slots, 1
                        ):
                            lay_slot = Int32(remote_slot_off)
                            lay_relay_base = (
                                lay_pool_base + lay_intra_padded + lay_relay_prefix
                            )
                            lay_relay_sf_base = (
                                lay_sf_pool_base + lay_intra_sf_padded + lay_relay_prefix
                            )
                            relay_pool_base[lay_expert, lay_slot] = lay_relay_base
                            relay_sf_pool_base[lay_expert, lay_slot] = lay_relay_sf_base
                            if cutlass.const_expr(
                                relay_direct_bucket_meta is not None
                            ):
                                if (
                                    lay_relay_base < Int32(0)
                                    or lay_relay_base > Int32(DIRECT_META_FIELD_MASK)
                                    or lay_relay_sf_base < Int32(0)
                                    or lay_relay_sf_base > Int32(DIRECT_META_FIELD_MASK)
                                    or lay_counts[remote_slot_off] >= Int32(DIRECT_META_COUNT_LIMIT)
                                ):
                                    if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                                        cute.printf("[gin-guard] token_comm source-site 4322\n")
                                    device_trap()
                                relay_direct_bucket_meta[lay_expert, lay_slot] = (
                                    Int64(lay_relay_base)
                                    | (Int64(lay_relay_sf_base) << Int64(DIRECT_META_SF_SHIFT))
                                    | (Int64(lay_counts[remote_slot_off]) << Int64(DIRECT_META_COUNT_SHIFT))
                                )
                            relay_pool_base_snapshot[lay_expert, lay_slot] = lay_relay_base
                            if cutlass.const_expr(
                                relay_sf_pool_base_snapshot is not None
                            ):
                                relay_sf_pool_base_snapshot[lay_expert, lay_slot] = (
                                    lay_relay_sf_base
                                )
                            lay_relay_prefix = lay_relay_prefix + lay_counts[remote_slot_off]
                lay_any_invalid = cute.arch.vote_ballot_sync(lay_invalid != Int32(0))
                if lane_idx == Int32(0):
                    if cutlass.const_expr(pool_end_snapshot is not None):
                        pool_end_snapshot[0] = lay_total_data_i64
                        pool_end_snapshot[1] = lay_total_sf_i64
                    if lay_any_invalid != Int32(0):
                        cute.printf(
                            "[relaunch-guard] rank=%d data_end=%ld data_cap=%d "
                            "sf_end=%ld sf_cap=%d\n",
                            Int32(local_rank),
                            lay_total_data_i64,
                            Int32(pool_token_capacity),
                            lay_total_sf_i64,
                            Int32(pool_sf_capacity),
                        )
                if lay_any_invalid != Int32(0):
                    if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                        cute.printf("[gin-guard] token_comm source-site 4352\n")
                    device_trap()
                cute.arch.fence_acq_rel_sys()
            if cutlass.const_expr(self.pipeline_timing_enabled):
                meta_t_layout = read_globaltimer()


            # Step 6: publish SM0's layout to every SM and every node-local
            # rank.  The prologue grid sync closes SM0's local stores; the LSA
            # barrier then makes intra_token_count / relay_pool_base visible to
            # combine relay peers before they map those arrays.  Its epilogue
            # grid sync also provides the original scheduler handoff, so a
            # separate software_grid_sync here would be redundant.
            self.nvlink_barrier(
                nvlink_barrier_signal,
                nvlink_barrier_counter,
                grid_sync_counter,
                peer_rank_ptr_mapper,
                sm_idx,
                warp_idx,
                lane_idx,
                num_sms=num_sms,
                local_rank=local_rank,
                sync_num_threads=self.dispatch_group_threads,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                meta_t_end = read_globaltimer()
                if (
                    sm_idx == Int32(0)
                    and warp_idx == Int32(0)
                    and lane_idx == Int32(0)
                ):
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_STEP3, 0
                    ] = meta_t0
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_STEP3, 1
                    ] = meta_t_step3
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_WAIT, 0
                    ] = meta_t_step3
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_WAIT, 1
                    ] = meta_t_wait
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_REDIST, 0
                    ] = meta_t_wait
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_REDIST, 1
                    ] = meta_t_redist
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_REDIST_ISSUE, 0
                    ] = meta_t_wait
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_REDIST_ISSUE, 1
                    ] = meta_t_redist_issue
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_REDIST_SYNC, 0
                    ] = meta_t_redist_issue
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_REDIST_SYNC, 1
                    ] = meta_t_redist
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_LAYOUT, 0
                    ] = meta_t_redist
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_LAYOUT, 1
                    ] = meta_t_layout
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_PUBLISH, 0
                    ] = meta_t_layout
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_META_PUBLISH, 1
                    ] = meta_t_end
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(
                    metadata_range, IKET_MEGA_ENVELOPE_META
                )

    @cute.jit
    def dispatch_pull(
        self,
        token_comm_storage,
        input_token_buffer,
        input_sf_buffer,
        input_topk_weights_buffer,
        src_token_topk_idx,
        expert_recv_count,
        expert_recv_count_sum,
        intra_token_count,
        expert_pool_base,
        expert_sf_pool_base,
        fc1_input_token_buffer,
        fc1_input_sf_buffer,
        fc1_input_topk_weights_buffer,
        fc1_ready_counter,
        token_src_metadata,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
        input_fc1_activation_per_token_scale_buffer=None,
        dispatched_fc1_activation_per_token_scale_buffer=None,
    ):
        # MemRange does not support dynamic indexing here; use raw pointers.
        pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()
        pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(pull_mbar_ptr + warp_idx, 1)
        cute.arch.sync_warp()


        phase_bit = Int32(0)

        current_expert_idx = Int32(-1)
        expert_start_idx = Int32(0)
        expert_end_idx = Int32(0)
        expert_pool_block_offset = Int32(0)
        expert_task_tile_offset = Int32(0)
        # SF rows use their own padding; token and SF pool offsets can diverge.
        expert_sf_pool_block_offset = Int32(0)
        if cutlass.const_expr(intra_token_count is not None):
            # Two-segment layout: the scheduled (intra + relay) and intranode
            # counts of the previous expert drive the adaptive tile prefix.
            prev_scheduled_count = Int32(0)
            prev_intra_count = Int32(0)

        # ── Release-flag batching ────────────────────────────────────────
        # Delay fc1-ready counter publication with the same rotating-lane
        # tracker used by the epilogue.  Each token's TMA store to the FC1 pool
        # is drained CTA-locally by ``cp_async_bulk_wait_group(0)`` before its
        # release target is accumulated; the eventual red.release.gpu add
        # publishes the corresponding pool data to GPU scope.
        if cutlass.const_expr(self.enable_nccl_gin):
            flag_tracker = SysReleaseFlagBatchTracker(
                flag_addr=Int64(0), cumulated_flags=Int32(0),
                phase=Int32(0), tid=lane_idx,
            )
        else:
            flag_tracker = GpuReleaseFlagBatchTracker(
                flag_addr=Int64(0), cumulated_flags=Int32(0),
                phase=Int32(0), tid=lane_idx,
            )

        stored_rank_count_lane = Int32(0)

        NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        stored_num_tokens_per_expert = []
        stored_intra_tokens_per_expert = []
        for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            stored_num_tokens_per_expert.append(Int32(0))
            if cutlass.const_expr(intra_token_count is not None):
                stored_intra_tokens_per_expert.append(Int32(0))
        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            e_idx_for_lane = Int32(i * self.warp_threads) + lane_idx
            if e_idx_for_lane < Int32(self.num_experts_per_rank):
                sum_packed_init = expert_recv_count_sum[e_idx_for_lane]
                stored_num_tokens_per_expert[i] = Int32(
                    Int64(sum_packed_init) & Int64(0xFFFFFFFF)
                )
                if cutlass.const_expr(intra_token_count is not None):
                    stored_intra_tokens_per_expert[i] = Int32(
                        intra_token_count[e_idx_for_lane]
                    )
        cute.arch.sync_warp()

        num_global_warps: cutlass.Constexpr[int] = (
            num_sms * self.active_dispatch_warps
        )
        token_idx = sm_idx * Int32(self.active_dispatch_warps) + warp_idx

        if cutlass.const_expr(intra_token_count is None and not self.iket_enabled):
            # Main's pull-loop markers, single node only and only when the
            # compiler erases iket ops (no ``iket`` in CUTE_DSL_COMPILER_OPT):
            # the production single-node IR stays identical to main's, while a
            # traced build keeps the decodable token-range scheme below.
            _iket_pull_emit = (
                (sm_idx == Int32(0))
                and (warp_idx == Int32(0))
                and (lane_idx == Int32(0))
            )

        while current_expert_idx < Int32(self.num_experts_per_rank):
            if cutlass.const_expr(self.iket_detailed_ranges):
                expert_scan_range = _iket.range_start(
                    "megamoe_work", IKET_MEGA_WORK_PULL_EXPERT_SCAN
                )
            if cutlass.const_expr(intra_token_count is None and not self.iket_enabled):
                if _iket_pull_emit:
                    _iket.range_push("Pull.ChooseToken")
            old_expert_idx = current_expert_idx
            while (token_idx >= expert_end_idx) and (
                current_expert_idx < Int32(self.num_experts_per_rank)
            ):
                prev_valid_count = expert_end_idx - expert_start_idx
                if cutlass.const_expr(intra_token_count is None):
                    # Single node (main's bookkeeping): every pulled token of
                    # the previous expert was scheduled, one segment per expert.
                    prev_pool_count = prev_valid_count
                else:
                    prev_pool_count = prev_scheduled_count
                if cutlass.const_expr(expert_pool_base is None):
                    prev_block_count = (
                        prev_pool_count
                        + Int32(self.token_padding_block) - Int32(1)
                    ) // Int32(self.token_padding_block)
                    expert_pool_block_offset = (
                        expert_pool_block_offset + prev_block_count
                    )
                # Match the scheduler's per-expert adaptive tile prefix.
                if cutlass.const_expr(intra_token_count is None):
                    prev_task_tile_count = (
                        prev_valid_count
                        + Int32(self.cluster_tile_tokens) - Int32(1)
                    ) // Int32(self.cluster_tile_tokens)
                else:
                    prev_relay_count = cutlass.max(
                        prev_scheduled_count - prev_intra_count, Int32(0)
                    )
                    prev_task_tile_count = expert_token_tiles(
                        prev_intra_count, prev_relay_count,
                        self.cluster_tile_tokens,
                    )
                expert_task_tile_offset = (
                    expert_task_tile_offset + prev_task_tile_count
                )
                # Mirror cumul for the SF axis granularity (self.sf_padding_block).
                if cutlass.const_expr(expert_sf_pool_base is None):
                    prev_sf_block_count = (
                        prev_pool_count
                        + Int32(self.sf_padding_block) - Int32(1)
                    ) // Int32(self.sf_padding_block)
                    expert_sf_pool_block_offset = (
                        expert_sf_pool_block_offset + prev_sf_block_count
                    )
                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    expert_start_idx = expert_end_idx
                    scheduled_value = Int32(0)
                    if cutlass.const_expr(intra_token_count is not None):
                        intra_value = Int32(0)
                    for i in cutlass.range_constexpr(
                        0, NUM_EXPERTS_PER_LANE, 1
                    ):
                        if current_expert_idx == Int32(i * self.warp_threads) + lane_idx:
                            scheduled_value = stored_num_tokens_per_expert[i]
                            if cutlass.const_expr(intra_token_count is not None):
                                intra_value = stored_intra_tokens_per_expert[i]
                    total_for_expert = cute.arch.shuffle_sync(
                        scheduled_value,
                        current_expert_idx % Int32(self.warp_threads),
                    )
                    if cutlass.const_expr(intra_token_count is None):
                        # Single node: every scheduled token is intranode.
                        expert_end_idx = expert_end_idx + total_for_expert
                    else:
                        prev_scheduled_count = total_for_expert
                        intra_for_expert = cute.arch.shuffle_sync(
                            intra_value,
                            current_expert_idx % Int32(self.warp_threads),
                        )
                        prev_intra_count = intra_for_expert
                        expert_end_idx = expert_end_idx + intra_for_expert
                    if cutlass.const_expr(expert_pool_base is not None):
                        expert_pool_block_offset = (
                            Int32(expert_pool_base[current_expert_idx])
                            // Int32(self.token_padding_block)
                        )
                        expert_sf_pool_block_offset = (
                            Int32(expert_sf_pool_base[current_expert_idx])
                            // Int32(self.sf_padding_block)
                        )

            if cutlass.const_expr(self.iket_detailed_ranges):
                _iket.range_end(
                    expert_scan_range, IKET_MEGA_WORK_PULL_EXPERT_SCAN
                )

            if current_expert_idx < Int32(self.num_experts_per_rank):
                if cutlass.const_expr(self.iket_detailed_ranges):
                    source_route_range = _iket.range_start(
                        "megamoe_work", IKET_MEGA_WORK_PULL_SOURCE_ROUTE
                    )
                if old_expert_idx != current_expert_idx:
                    if lane_idx < Int32(self.world_size):
                        stored_rank_count_lane = Int32(
                            expert_recv_count[lane_idx, current_expert_idx]
                        )
                    else:
                        stored_rank_count_lane = Int32(0)

                token_idx_in_expert = token_idx - expert_start_idx
                slot_idx = token_idx_in_expert
                offset = Int32(0)
                remaining_lane = stored_rank_count_lane

                current_rank_in_expert_idx = Int32(0)
                token_idx_in_rank = Int32(0)

                decided = Int32(0)
                for _round in cutlass.range_constexpr(0, self.world_size + 1, 1):
                    if decided == Int32(0):
                        active = remaining_lane > Int32(0)
                        mask = cute.arch.vote_ballot_sync(active)
                        num_active_ranks = Int32(cute.arch.popc(Int32(mask)))
                        v_for_min = Int32(0x7FFFFFFF)
                        if active:
                            v_for_min = remaining_lane
                        length = Int32(
                            cute.arch.warp_redux_sync(v_for_min, "min")
                        )

                        if num_active_ranks > Int32(0):
                            num_round_tokens = length * num_active_ranks
                            if slot_idx < num_round_tokens:
                                slot_idx_in_round = slot_idx % num_active_ranks
                                current_rank_in_expert_idx = fns_b32(
                                    Int32(mask),
                                    Int32(0),
                                    slot_idx_in_round + Int32(1),
                                )
                                token_idx_in_rank = offset + (
                                    slot_idx // num_active_ranks
                                )
                                decided = Int32(1)
                            else:
                                slot_idx = slot_idx - num_round_tokens
                                offset = offset + length
                                if remaining_lane > length:
                                    remaining_lane = remaining_lane - length
                                else:
                                    remaining_lane = Int32(0)
                        else:
                            decided = Int32(1)

                if cutlass.const_expr(intra_token_count is None and not self.iket_enabled):
                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.ChooseToken
                        _iket.range_push("Pull.TMA_NVLink_Roundtrip")

                src_token_topk = Uint32(
                    src_token_topk_idx[
                        current_expert_idx,
                        current_rank_in_expert_idx,
                        token_idx_in_rank,
                    ]
                )
                src_token = Int32(src_token_topk // Uint32(self.num_topk))
                src_topk = Int32(src_token_topk % Uint32(self.num_topk))

                cur_peer_offset = peer_rank_ptr_mapper.map(
                    Int64(0), current_rank_in_expert_idx, Int64(0)
                )
                inp_tok_local_base = input_token_buffer.iterator.toint()
                # SF-free callers pass input_sf_buffer=None (their SF loops are
                # compiled out by sf_uint32_per_token=0, so this base address
                # is never consumed).
                if cutlass.const_expr(input_sf_buffer is not None):
                    inp_sf_local_base = input_sf_buffer.iterator.toint()
                inp_w_local_base = input_topk_weights_buffer.iterator.toint()
                if cutlass.const_expr(
                    input_fc1_activation_per_token_scale_buffer is not None
                ):
                    inp_act_scale_local_base = (
                        input_fc1_activation_per_token_scale_buffer.iterator.toint()
                    )

                if cutlass.const_expr(self.iket_detailed_ranges):
                    tma_load_issue_range = _iket.range_start(
                        "megamoe_work", IKET_MEGA_WORK_PULL_TMA_LOAD_ISSUE
                    )
                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
                    tma_src_addr = (
                        inp_tok_local_base
                        + cur_peer_offset
                        + Int64(src_token * Int32(self.hidden_bytes))
                    )
                    tma_load_1d_raw(
                        pull_buffer_warp_ptr,
                        tma_src_addr,
                        pull_mbar_ptr + warp_idx,
                        Int32(self.hidden_bytes),
                    )
                cute.arch.sync_warp()
                if cutlass.const_expr(self.iket_detailed_ranges):
                    _iket.range_end(
                        tma_load_issue_range,
                        IKET_MEGA_WORK_PULL_TMA_LOAD_ISSUE,
                    )

                if cutlass.const_expr(self.iket_detailed_ranges):
                    sf_weight_load_range = _iket.range_start(
                        "megamoe_work", IKET_MEGA_WORK_PULL_SF_WEIGHT_LOAD
                    )
                if cutlass.const_expr(intra_token_count is None and not self.iket_enabled):
                    if _iket_pull_emit:
                        _iket.range_push("Pull.SF_LDG_STG")

                sf_token_in_pool_axis = (
                    expert_sf_pool_block_offset * Int32(self.sf_padding_block)
                    + token_idx_in_expert
                )
                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block)
                    + token_idx_in_expert
                )
                if cutlass.const_expr(expert_pool_base is not None):
                    pool_token_idx = (
                        Int32(expert_pool_base[current_expert_idx])
                        + token_idx_in_expert
                    )
                sf_passes: cutlass.Constexpr[int] = (
                    self.sf_uint32_per_token + 31
                ) // 32

                sf_vals = []
                for _ in cutlass.range_constexpr(0, sf_passes, 1):
                    sf_vals.append(Int32(0))

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        sf_addr = (
                            inp_sf_local_base
                            + cur_peer_offset
                            + Int64(
                                (src_token * Int32(self.sf_uint32_per_token) + j)
                                * Int32(4)
                            )
                        )
                        sf_vals[i] = ldg_b32_raw(sf_addr)

                weight = Float32(0.0)
                act_scale = Float32(0.0)
                if lane_idx == Int32(0):
                    weight_addr = (
                        inp_w_local_base
                        + cur_peer_offset
                        + Int64(
                            (src_token * Int32(self.num_topk) + src_topk) * Int32(4)
                        )
                    )
                    weight = ldg_f32_raw(weight_addr)
                    if cutlass.const_expr(
                        input_fc1_activation_per_token_scale_buffer is not None
                    ):
                        # Per-token (not per-route): one fp32 per source token.
                        # Flat ``base + token * 4`` addressing: the user tensor
                        # must be a contiguous (T,) fp32 plane (host-checked).
                        act_scale_addr = (
                            inp_act_scale_local_base
                            + cur_peer_offset
                            + Int64(src_token * Int32(4))
                        )
                        act_scale = ldg_f32_raw(act_scale_addr)

                if cutlass.const_expr(self.iket_detailed_ranges):
                    sf_weight_store_range = _iket.range_start(
                        "megamoe_work", IKET_MEGA_WORK_PULL_SF_WEIGHT_STORE
                    )
                if cutlass.const_expr(intra_token_count is None and not self.iket_enabled):
                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.SF_LDG_STG  (= LD phase)
                        _iket.range_push("Pull.Weight_LDG")   # (= ST phase)

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        if cutlass.const_expr(self.sf_atom_swizzled):
                            sf_int32_pos = sf_atom_int32_offset(
                                sf_token_in_pool_axis,
                                j,
                                num_k_atoms=self.sf_uint32_per_token,
                            )
                        else:
                            sf_int32_pos = (
                                sf_token_in_pool_axis
                                * Int32(self.sf_uint32_per_token)
                                + j
                            )
                        fc1_input_sf_buffer[sf_int32_pos] = sf_vals[i]
                cute.arch.sync_warp()

                if lane_idx == Int32(0):
                    fc1_input_topk_weights_buffer[pool_token_idx] = weight
                    if cutlass.const_expr(
                        dispatched_fc1_activation_per_token_scale_buffer is not None
                    ):
                        dispatched_fc1_activation_per_token_scale_buffer[
                            pool_token_idx
                        ] = act_scale
                if cutlass.const_expr(self.iket_detailed_ranges):
                    _iket.range_end(
                        sf_weight_store_range,
                        IKET_MEGA_WORK_PULL_SF_WEIGHT_STORE,
                    )

                if cutlass.const_expr(self.iket_detailed_ranges):
                    tma_load_wait_range = _iket.range_start(
                        "megamoe_wait", IKET_MEGA_WAIT_PULL_TMA_LOAD
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        pull_mbar_ptr + warp_idx, Int32(self.hidden_bytes)
                    )
                    cute.arch.mbarrier_wait(
                        pull_mbar_ptr + warp_idx,
                        phase_bit,
                    )
                if cutlass.const_expr(self.iket_detailed_ranges):
                    _iket.range_end(
                        tma_load_wait_range, IKET_MEGA_WAIT_PULL_TMA_LOAD
                    )

                if cutlass.const_expr(self.iket_detailed_ranges):
                    tma_store_issue_range = _iket.range_start(
                        "megamoe_work", IKET_MEGA_WORK_PULL_TMA_STORE_ISSUE
                    )
                if cutlass.const_expr(intra_token_count is None and not self.iket_enabled):
                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.Weight_LDG (ST phase)
                        _iket.range_pop()  # Pull.TMA_NVLink_Roundtrip (outer)
                        _iket.range_push("Pull.TMA_Store")

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
                    tma_store_1d(
                        fc1_input_token_buffer.iterator
                        # T=128k) × self.hidden_bytes overflows int32 (max 2.1 G).
                        # 64-bit address math is required for large token pools.
                        + (Int64(pool_token_idx) * Int64(self.hidden_bytes)),
                        pull_buffer_warp_ptr,
                        Int32(self.hidden_bytes),
                    )

                with cute.arch.elect_one():
                    TokenSrcMetadata(
                        src_rank=current_rank_in_expert_idx,
                        src_token=src_token,
                        src_topk=src_topk,
                    ).store(
                        token_src_metadata.iterator
                        + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                    )

                if cutlass.const_expr(self.iket_detailed_ranges):
                    # Issue / wait ranges need the two bulk-group steps in
                    # separate elected blocks.
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()
                    _iket.range_end(
                        tma_store_issue_range,
                        IKET_MEGA_WORK_PULL_TMA_STORE_ISSUE,
                    )
                    tma_store_wait_range = _iket.range_start(
                        "megamoe_wait", IKET_MEGA_WAIT_PULL_TMA_STORE
                    )
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0)
                    _iket.range_end(
                        tma_store_wait_range, IKET_MEGA_WAIT_PULL_TMA_STORE
                    )
                else:
                    # Main's shape: one elected block commits and drains.
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)

                if cutlass.const_expr(self.iket_detailed_ranges):
                    arrival_publish_range = _iket.range_start(
                        "megamoe_work", IKET_MEGA_WORK_PULL_ARRIVAL_PUBLISH
                    )
                if cutlass.const_expr(intra_token_count is None and not self.iket_enabled):
                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.TMA_Store
                        _iket.range_push("Pull.Arrival_Atomic")

                # Accumulate this token's release target into the rotating-lane
                # batch tracker.  task_tile_idx is warp-uniform (token_idx /
                # expert offsets are warp-wide), so every lane runs the same
                # state-machine transition while only one lane records the
                # current address.
                task_tile_idx = expert_task_tile_offset + (
                    token_idx_in_expert // Int32(self.cluster_tile_tokens)
                )
                if cutlass.const_expr(self.publish_relay_counts_to_gemm):
                    task_tile_idx = pool_token_idx // Int32(self.cluster_tile_tokens)

                task_tile_addr = (fc1_ready_counter.iterator + task_tile_idx).toint()
                if cutlass.const_expr(self.enable_nccl_gin):
                    # Local TMA stores are drained above. Every lane also
                    # publishes its ordinary SF/metadata stores to the scope
                    # shared with peer fan-out writers of the same counter.
                    cute.arch.sync_warp()
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.sync_warp()
                flag_tracker = flag_tracker.accumulate(
                    Int32(0), self._flag_batch, task_tile_addr,
                )
                cute.arch.sync_warp()
                if cutlass.const_expr(self.iket_detailed_ranges):
                    _iket.range_end(
                        arrival_publish_range,
                        IKET_MEGA_WORK_PULL_ARRIVAL_PUBLISH,
                    )
                if cutlass.const_expr(intra_token_count is None and not self.iket_enabled):
                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.Arrival_Atomic

                phase_bit = phase_bit ^ Int32(1)

                token_idx = token_idx + Int32(num_global_warps)

        # Tail flush: publish any leftover (< self._flag_batch) accumulated release.
        flag_tracker.fire()
        cute.arch.sync_warp()

        return phase_bit, stored_num_tokens_per_expert

    @cute.jit
    def _copy_rows_unrolled(self, dst_addr, src_addr, lane_idx, *, total_bytes):
        """Warp copy of ``total_bytes`` with 16-byte lanes, four 512-byte rows
        per dependent round trip."""
        row_bytes: cutlass.Constexpr[int] = self.warp_threads * 16
        lane_off = Int64(lane_idx) * Int64(16)
        for round_idx in cutlass.range_constexpr(
            0, (total_bytes + 4 * row_bytes - 1) // (4 * row_bytes), 1
        ):
            base_bytes: cutlass.Constexpr[int] = round_idx * 4 * row_bytes
            # Rows k with base + k * row + lane * 16 < total are valid.
            row_count = (
                Int32(total_bytes - base_bytes)
                - lane_idx * Int32(16)
                + Int32(row_bytes - 1)
            ) // Int32(row_bytes)
            row_count = cutlass.min(cutlass.max(row_count, Int32(0)), Int32(4))
            copy_b128x4_strided_raw(
                dst_addr + Int64(base_bytes) + lane_off,
                src_addr + Int64(base_bytes) + lane_off,
                row_count,
                stride_bytes=row_bytes,
            )

    @cute.jit
    def internode_combine(
        self,
        token_comm_storage,
        internode_recv_splits,
        internode_send_count,
        relay_fanout_slot,
        expert_recv_count_sum,
        intra_token_count,
        relay_pool_base,
        fc2_output_workspace,
        fc2_done_counter,
        internode_barrier_counter,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        gin_window,
        pipeline_timing,
        combine_timing_landmarks,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_sms,
        wide_ring_smem=None,
        combine_issue_counter=None,
    ):
        """Gather remote-origin FC2 terms locally and RDMA-return BF16 partials."""
        if cutlass.const_expr(
            self.enable_nccl_gin
            and self.num_remote_nodes > 0
            and relay_fanout_slot is not None
            and fc2_output_workspace is not None
            and fc2_done_counter is not None
        ):
            from nccl.core.device import cute as nccl_cute

            relay_warp_idx = warp_idx - Int32(self.num_dispatch_warps)
            relay_thread_idx = (
                relay_warp_idx * Int32(self.warp_threads) + lane_idx
            )
            global_relay_warp = (
                sm_idx * Int32(self.num_relay_warps) + relay_warp_idx
            )
            num_global_relay_warps: cutlass.Constexpr[int] = (
                num_sms * self.num_relay_warps
            )
            my_node = Int32(local_rank) // Int32(self.ranks_per_node)
            rank_in_node = Int32(local_rank) % Int32(self.ranks_per_node)
            node_base_rank = my_node * Int32(self.ranks_per_node)
            combine_phase = internode_barrier_counter[0] & Int32(1)

            gin = self._device_gin_gpu()
            team_rail = self._team_rail_value
            thread_coop = nccl_cute.thread()
            relay_coop = nccl_cute.warp_span(
                warp0=self.relay_warp_start,
                n_warps=self.num_relay_warps,
                id=2,
            )
            window_u8 = gin_window.tensor(
                Uint8,
                cute.make_layout((self.gin_window_bytes,), stride=(1,)),
            )
            combine_bytes: cutlass.Constexpr[int] = self.hidden * 2
            combine_layout = cute.make_layout(
                (combine_bytes,), stride=(1,),
            )
            combine_b128_scratch = (
                token_comm_storage.combine_b128_scratch.data_ptr()
            )
            combine_b128_lane_scratch = (
                combine_b128_scratch
                + relay_thread_idx * Int32(self.num_topk * 16)
            )
            wide_ring_rounds: cutlass.Constexpr[int] = (
                self.combine_wide_ring_rounds
            )
            if cutlass.const_expr(
                wide_ring_rounds > 0 and wide_ring_smem is not None
            ):
                # Per-lane ring base inside the AB stage buffers: relay warp w
                # owns wide_ring_rounds consecutive 2 KB round buffers; each
                # keeps the scratch path's lane stride of num_topk x 16 B.
                fc_smem_free_addr = Int32(
                    token_comm_storage.fc_smem_free.data_ptr().toint()
                )
                wide_ring_lane_base = (
                    Int32(wide_ring_smem.toint())
                    + (relay_thread_idx // Int32(self.warp_threads))
                    * Int32(
                        wide_ring_rounds * self.num_topk * 16 * self.warp_threads
                    )
                    + (relay_thread_idx % Int32(self.warp_threads))
                    * Int32(self.num_topk * 16)
                )
            else:
                # The unified reduce loop below names these under a
                # compile-time-false branch; the DSL still captures them.
                fc_smem_free_addr = Int32(0)
                wide_ring_lane_base = Int32(0)
            ring_round_bytes: cutlass.Constexpr[int] = (
                self.num_topk * 16 * self.warp_threads
            )

            # Reuse relay scratch only after dispatch packing/fanout have
            # consumed it.  One counter per origin-node slot records how many
            # return puts this CTA issued, which becomes the strong credit.
            cta_return_count = token_comm_storage.smem_inter_token_base.data_ptr()
            j = relay_thread_idx
            while j < Int32(self.num_remote_node_slots):
                (cta_return_count + j).store(Int32(0))
                j = j + Int32(self.num_relay_threads)
            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.num_relay_threads,
            )

            if cutlass.const_expr(self.iket_enabled):
                combine_return_range = _iket.range_start("internode_combine")
            combine_start = Int64(0)
            combine_route_done = Int64(0)
            combine_reduce_done = Int64(0)
            combine_issue_done = Int64(0)
            combine_flush_credit_done = Int64(0)
            combine_arrival_done = Int64(0)
            combine_fc2_wait_ns = Int64(0)
            combine_route_inclusive_ns = Int64(0)
            combine_reduce_ns = Int64(0)
            combine_publish_issue_ns = Int64(0)
            combine_message_count = Int64(0)
            combine_waited_hit_count = Int64(0)
            combine_max_fc2_wait_ns = Int64(0)
            combine_last_reduce_start = Int64(0)
            combine_last_reduce_end = Int64(0)
            combine_last_reduce_ring = Int64(0)
            recv_base = window_u8.iterator.toint()
            # Bound the number of combine producers from static split metadata.
            # The default target of one message per worker reproduces the
            # existing maximum useful parallelism. Larger A/B values trade
            # more messages per worker for fewer simultaneous GIN puts.
            total_combine_messages = Int32(0)
            for src_slot_off in cutlass.range_constexpr(
                0, self.num_remote_node_slots, 1
            ):
                total_combine_messages = total_combine_messages + Int32(
                    internode_recv_splits[
                        Int32(src_slot_off), self.node_experts
                    ]
                )
            combine_worker_count = total_combine_messages
            combine_worker_count = cutlass.max(
                combine_worker_count, Int32(self.num_relay_warps)
            )
            pool_warp_count: cutlass.Constexpr[int] = num_global_relay_warps
            pool_warp = global_relay_warp
            if combine_worker_count > Int32(pool_warp_count):
                combine_worker_count = Int32(pool_warp_count)

            # Select K physical workers evenly across the resident grid instead
            # of concentrating work on the first K warps/SMs. For pool ordinal
            # p, the unique selected physical worker is floor(p * W / K).
            combine_pool_idx = (
                pool_warp * combine_worker_count
                + Int32(pool_warp_count) - Int32(1)
            ) // Int32(pool_warp_count)
            combine_worker_active = combine_pool_idx < combine_worker_count
            if combine_worker_active:
                combine_worker_active = (
                    combine_pool_idx * Int32(pool_warp_count)
                ) // combine_worker_count == pool_warp
            source_worker_base = Int32(0)
            for src_slot_off in cutlass.range_constexpr(
                0, self.num_remote_node_slots, 1
            ):
                src_slot = Int32(src_slot_off)
                origin_node = src_slot
                if src_slot >= my_node:
                    origin_node = src_slot + Int32(1)
                contributor_slot_at_origin = my_node
                if my_node > origin_node:
                    contributor_slot_at_origin = my_node - Int32(1)
                num_messages = internode_recv_splits[
                    src_slot, self.node_experts
                ]
                # Flatten the static per-source message buckets across the
                # global relay-warp pool.  Do not use signed remainder here:
                # PTX srem preserves the negative dividend sign.  Since both
                # operands are in [0, W), one conditional add is sufficient.
                worker_idx = combine_pool_idx - source_worker_base
                if worker_idx < Int32(0):
                    worker_idx = worker_idx + combine_worker_count
                message_slot = worker_idx
                while combine_worker_active and message_slot < num_messages:
                    if cutlass.const_expr(self.pipeline_timing_enabled):
                        message_route_start = read_globaltimer()
                        combine_message_count = combine_message_count + Int64(1)
                    if cutlass.const_expr(self.iket_detailed_ranges):
                        combine_message_range = _iket.range_start(
                            "megamoe_work", IKET_MEGA_WORK_COMBINE_MESSAGE
                        )
                    recv_byte_offset = Int64(
                        self.relay_recv_buffer_offset
                    ) + (
                        Int64(src_slot)
                        * Int64(self.max_relay_tokens_per_node)
                        + Int64(message_slot)
                    ) * Int64(self.relay_message_bytes)
                    message_addr = recv_base + recv_byte_offset

                    src_token = Int32(0)
                    if lane_idx == Int32(0):
                        src_token = TokenSrcMetadata.load(
                            message_addr
                            + Int64(self.relay_message_src_metadata_offset)
                        ).src_token
                    src_token = cute.arch.shuffle_sync(src_token, Int32(0))
                    if src_token >= Int32(self.max_relay_tokens_per_node):
                        if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                            cute.printf("[gin-guard] token_comm source-site 5223\n")
                        device_trap()

                    if cutlass.const_expr(self.iket_detailed_ranges):
                        combine_route_range = _iket.range_start(
                            "megamoe_combine_work",
                            IKET_MEGA_COMBINE_WORK_ROUTE_LOOKUP,
                        )

                    hit_valid = []
                    hit_owner_rank = []
                    hit_pool_slot = []
                    hit_weight = []
                    packed_route_experts = Int64(-1)
                    if lane_idx == Int32(0):
                        packed_route_experts = ldg_route_experts_b128_raw(
                            message_addr
                            + Int64(self.relay_message_route_words_offset)
                        )
                    for topk_off in cutlass.range_constexpr(0, 4, 1):
                        node_expert = cute.arch.shuffle_sync(
                            Int32(
                                (
                                    packed_route_experts
                                    >> Int64(topk_off * 16)
                                ) & Int64(0xFFFF)
                            ),
                            Int32(0),
                        )
                        valid = node_expert != Int32(0xFFFF)
                        owner_rank = Int32(local_rank)
                        pool_slot = Int32(0)
                        weight = Float32(1.0)
                        if valid:
                            owner_rank_in_node = (
                                node_expert
                                // Int32(self.num_experts_per_rank)
                            )
                            local_expert = (
                                node_expert
                                % Int32(self.num_experts_per_rank)
                            )
                            owner_rank = node_base_rank + owner_rank_in_node

                            total_addr = peer_rank_ptr_mapper.map(
                                expert_recv_count_sum.iterator.toint(),
                                owner_rank,
                                Int64(local_expert * Int32(8)),
                            )
                            total_ptr = cute.make_ptr(
                                Int64, total_addr, AddressSpace.gmem,
                                assumed_align=8,
                            )
                            total_packed = cute.arch.load(
                                total_ptr, Int64, sem="acquire", scope="sys",
                            )
                            total_count = Int32(
                                total_packed & Int64(0xFFFFFFFF)
                            )
                            intra_addr = peer_rank_ptr_mapper.map(
                                intra_token_count.iterator.toint(),
                                owner_rank,
                                Int64(local_expert * Int32(4)),
                            )
                            intra_ptr = cute.make_ptr(
                                Int32, intra_addr, AddressSpace.gmem,
                                assumed_align=4,
                            )
                            intra_count = cute.arch.load(
                                intra_ptr, Int32, sem="acquire", scope="sys",
                            )
                            relay_count = cutlass.max(
                                total_count - intra_count, Int32(0)
                            )
                            expected_tiles = expert_token_tiles(
                                intra_count, relay_count, self.cluster_tile_tokens,
                            )
                            expected_done = expected_tiles * Int32(
                                self.fc2_publishes_per_token_cluster_tile
                            )
                            done_addr = peer_rank_ptr_mapper.map(
                                fc2_done_counter.iterator.toint(),
                                owner_rank,
                                Int64(local_expert * Int32(4)),
                            )
                            done_ptr = cute.make_ptr(
                                Int32, done_addr, AddressSpace.gmem,
                                assumed_align=4,
                            )
                            if cutlass.const_expr(self.pipeline_timing_enabled):
                                if combine_start == Int64(0):
                                    combine_start = read_globaltimer()
                            if cutlass.const_expr(self.iket_detailed_ranges):
                                fc2_wait_range = _iket.range_start(
                                    "megamoe_wait", IKET_MEGA_WAIT_FC2_DONE
                                )
                            wait_start = read_clock64()
                            waited_for_fc2 = Int32(0)
                            fc2_wait_global_start = Int64(0)
                            done = cute.arch.load(
                                done_ptr, Int32, sem="acquire", scope="sys",
                            )
                            if done < expected_done:
                                waited_for_fc2 = Int32(1)
                                if cutlass.const_expr(self.pipeline_timing_enabled):
                                    fc2_wait_global_start = read_globaltimer()
                            while done < expected_done:
                                _nanosleep(500)
                                done = cute.arch.load(
                                    done_ptr, Int32,
                                    sem="acquire", scope="sys",
                                )
                                if (
                                    read_clock64() - wait_start
                                    > Int64(self.rail_barrier_timeout_cycles)
                                ):
                                    if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                                        cute.printf("[gin-guard] token_comm source-site 5338\n")
                                    device_trap()
                            if cutlass.const_expr(self.pipeline_timing_enabled):
                                if waited_for_fc2 != Int32(0):
                                    fc2_wait_elapsed = (
                                        read_globaltimer()
                                        - fc2_wait_global_start
                                    )
                                    combine_fc2_wait_ns = (
                                        combine_fc2_wait_ns + fc2_wait_elapsed
                                    )
                                    combine_waited_hit_count = (
                                        combine_waited_hit_count + Int64(1)
                                    )
                                    if (
                                        fc2_wait_elapsed
                                        > combine_max_fc2_wait_ns
                                    ):
                                        combine_max_fc2_wait_ns = (
                                            fc2_wait_elapsed
                                        )
                            if cutlass.const_expr(self.iket_detailed_ranges):
                                _iket.range_end(
                                    fc2_wait_range,
                                    IKET_MEGA_WAIT_FC2_DONE,
                                )

                            pool_slot = Int32(relay_fanout_slot[
                                src_slot, message_slot, Int32(topk_off)
                            ])
                            if pool_slot < Int32(0):
                                if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                                    cute.printf("[gin-guard] token_comm source-site 5368\n")
                                device_trap()
                            if cutlass.const_expr(not self.apply_topk_in_fc1):
                                weight = Float32(
                                    ldg_f32_raw(
                                        message_addr
                                        + Int64(
                                            self.relay_message_topk_weights_offset
                                            + topk_off * 4
                                        )
                                    )
                                )
                        hit_valid.append(valid)
                        hit_owner_rank.append(owner_rank)
                        hit_pool_slot.append(pool_slot)
                        hit_weight.append(weight)

                    if cutlass.const_expr(self.iket_detailed_ranges):
                        _iket.range_end(
                            combine_route_range,
                            IKET_MEGA_COMBINE_WORK_ROUTE_LOOKUP,
                        )
                    if cutlass.const_expr(self.pipeline_timing_enabled):
                        combine_route_done = read_globaltimer()
                        combine_route_inclusive_ns = (
                            combine_route_inclusive_ns
                            + combine_route_done - message_route_start
                        )

                    if cutlass.const_expr(self.iket_detailed_ranges):
                        combine_reduce_range = _iket.range_start(
                            "megamoe_combine_work",
                            IKET_MEGA_COMBINE_WORK_REDUCE,
                        )
                    if cutlass.const_expr(self.pipeline_timing_enabled):
                        combine_reduce_start = read_globaltimer()
                    reduce_byte_offset = Int64(
                        self.intra_node_reduce_buf_offset
                    ) + (
                        Int64(src_slot)
                        * Int64(self.max_relay_tokens_per_node)
                        + Int64(message_slot)
                    ) * Int64(combine_bytes)
                    reduce_addr = recv_base + reduce_byte_offset
                    # The system-scope acquire on each peer's done counter
                    # orders the following generic-proxy loads. cuTeDSL has no
                    # system-scope BF16 ExtLoadOp, so use naturally aligned
                    # raw vectors. The compile-time controls keep b64, b128,
                    # and issue-first candidates independently measurable.
                    # Packed gather-reduce. The legacy loop below issues
                    # one round of at most num_topk 16-byte loads per
                    # 256-element chunk, so every message costs hidden/256
                    # dependent NVLink round trips regardless of how many
                    # of its hits are valid. Here the num_topk smem slots
                    # per lane carry (chunk, hit) pairs instead: with h
                    # valid hits each round covers num_topk // h chunks, so
                    # a one-hit message needs a quarter of the round trips.
                    # The FP32 accumulation visits the valid hits in the
                    # same topk order as before (Invariant I11).
                    pack_prefix = [Int32(0)]
                    for topk_off in cutlass.range_constexpr(0, 3, 1):
                        pack_prefix.append(
                            pack_prefix[topk_off]
                            + Int32(hit_valid[topk_off])
                        )
                    pack_hit_count = pack_prefix[3] + Int32(hit_valid[3])
                    pack_row_addr = []
                    for topk_off in cutlass.range_constexpr(0, 4, 1):
                        pack_row_addr.append(
                            peer_rank_ptr_mapper.map(
                                fc2_output_workspace.iterator.toint(),
                                hit_owner_rank[topk_off],
                                Int64(hit_pool_slot[topk_off])
                                * Int64(combine_bytes),
                            )
                        )
                    # Dense hit p is the p-th valid hit in topk order.
                    dense_addr = []
                    dense_weight = []
                    for dense_off in cutlass.range_constexpr(0, 4, 1):
                        sel_addr = Int64(0)
                        sel_weight = Float32(0.0)
                        for topk_off in cutlass.range_constexpr(0, 4, 1):
                            sel = Int32(hit_valid[topk_off]) * Int32(
                                pack_prefix[topk_off] == Int32(dense_off)
                            )
                            sel_addr = (
                                sel_addr
                                + Int64(sel) * pack_row_addr[topk_off]
                            )
                            sel_weight = (
                                sel_weight
                                + Float32(sel) * hit_weight[topk_off]
                            )
                        dense_addr.append(
                            sel_addr + Int64(lane_idx) * Int64(16)
                        )
                        dense_weight.append(sel_weight)
                    pack_chunk_bytes: cutlass.Constexpr[int] = (
                        16 * self.warp_threads
                    )
                    pack_total_chunks: cutlass.Constexpr[int] = (
                        (combine_bytes + pack_chunk_bytes - 1) // pack_chunk_bytes
                    )
                    # Lanes of the last chunk that are still inside the row; 0 when the row is a
                    # whole number of chunks (hidden % 256 == 0), and then every chunk predicate
                    # below stays the plain chunk bound it was before.
                    pack_tail_lanes: cutlass.Constexpr[int] = (
                        (combine_bytes % pack_chunk_bytes) // 16
                    )
                    wide_ok = Int32(0)
                    if cutlass.const_expr(
                        wide_ring_rounds > 0 and wide_ring_smem is not None
                    ):
                        wide_ok = ld_shared_acquire_cta_b32_raw(
                            fc_smem_free_addr
                        )
                    ring_on = Int32(0)
                    for hit_case in cutlass.range_constexpr(0, 5, 1):
                        if pack_hit_count == Int32(hit_case):
                            chunks_per_round: cutlass.Constexpr[int] = (
                                4 if hit_case == 0 else 4 // hit_case
                            )
                            chunk_base = Int32(0)
                            if cutlass.const_expr(
                                hit_case > 0
                                and wide_ring_rounds > 0
                                and wide_ring_smem is not None
                            ):
                                ring_on = wide_ok
                            # One loop serves both paths: a round on the
                            # ring finishes the message; a scratch round
                            # advances one round and, with the upgrade
                            # knob, re-reads the ring flag so a message
                            # that started before this CTA's GEMM warps
                            # exited finishes on the ring.
                            while chunk_base < Int32(pack_total_chunks):
                                scratch_round = Int32(1)
                                if cutlass.const_expr(
                                    hit_case > 0
                                    and wide_ring_rounds > 0
                                    and wide_ring_smem is not None
                                ):
                                    if ring_on != Int32(0):
                                        # Deep-prefetch ring in the freed AB
                                        # stage buffers: wide_ring_rounds - 1
                                        # rounds of cp.async loads stay in
                                        # flight per lane, and each round is
                                        # consumed exactly like the scratch
                                        # path below (Invariant I11). A round
                                        # buffer is refilled one iteration
                                        # after it was consumed, so no
                                        # asynchronous write can race the
                                        # lane's own reads of that buffer. The
                                        # loop leaves chunk_base at the end so
                                        # the scratch loop does not run.
                                        fence_proxy_async_shared_cta_raw()
                                        for pre_round in cutlass.range_constexpr(
                                            0, wide_ring_rounds - 1, 1
                                        ):
                                            pre_base = chunk_base + Int32(
                                                pre_round * chunks_per_round
                                            )
                                            pre_slot_base = wide_ring_lane_base + Int32(
                                                pre_round * ring_round_bytes
                                            )
                                            for chunk_off in cutlass.range_constexpr(
                                                0, chunks_per_round, 1
                                            ):
                                                ring_chunk_ok = Int32(
                                                    pre_base + Int32(chunk_off)
                                                    < Int32(pack_total_chunks)
                                                )
                                                if cutlass.const_expr(pack_tail_lanes > 0):
                                                    ring_chunk_ok = Int32(
                                                        (pre_base + Int32(chunk_off)) * Int32(pack_chunk_bytes)
                                                        + lane_idx * Int32(16)
                                                        < Int32(combine_bytes)
                                                    )
                                                for dense_off in cutlass.range_constexpr(
                                                    0, hit_case, 1
                                                ):
                                                    slot = chunk_off * hit_case + dense_off
                                                    # An invalid chunk keeps the row base address and
                                                    # zero-fills (src-size 0), so nothing past the row
                                                    # is ever read.
                                                    cp_async_16_smem_raw(
                                                        pre_slot_base + Int32(slot * 16),
                                                        dense_addr[dense_off]
                                                        + Int64(ring_chunk_ok)
                                                        * (Int64(pre_base) + Int64(chunk_off))
                                                        * Int64(pack_chunk_bytes),
                                                        ring_chunk_ok,
                                                    )
                                            cp_async_commit_group_raw()
                                        ring_round = Int32(0)
                                        while chunk_base < Int32(pack_total_chunks):
                                            cp_async_wait_group_raw(wide_ring_rounds - 2)
                                            ring_slot_base = wide_ring_lane_base + (
                                                ring_round % Int32(wide_ring_rounds)
                                            ) * Int32(ring_round_bytes)
                                            for chunk_off in cutlass.range_constexpr(
                                                0, chunks_per_round, 1
                                            ):
                                                lane_chunk_ok = Int32(
                                                    chunk_base + Int32(chunk_off)
                                                    < Int32(pack_total_chunks)
                                                )
                                                if cutlass.const_expr(pack_tail_lanes > 0):
                                                    lane_chunk_ok = Int32(
                                                        (chunk_base + Int32(chunk_off)) * Int32(pack_chunk_bytes)
                                                        + lane_idx * Int32(16)
                                                        < Int32(combine_bytes)
                                                    )
                                                if lane_chunk_ok != Int32(0):
                                                    pack_acc = [
                                                        Float32(0.0) for _ in range(8)
                                                    ]
                                                    for dense_off in cutlass.range_constexpr(
                                                        0, hit_case, 1
                                                    ):
                                                        slot = (
                                                            chunk_off * hit_case
                                                            + dense_off
                                                        )
                                                        for word_off in cutlass.range_constexpr(
                                                            0, 4, 1
                                                        ):
                                                            value_0, value_1 = (
                                                                unpack_bf16x2_to_f32x2(
                                                                    lds_b32_addr_raw(
                                                                        ring_slot_base
                                                                        + Int32(
                                                                            slot * 16
                                                                            + word_off * 4
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                            pack_acc[word_off * 2] = (
                                                                pack_acc[word_off * 2]
                                                                + value_0
                                                                * dense_weight[dense_off]
                                                            )
                                                            pack_acc[word_off * 2 + 1] = (
                                                                pack_acc[word_off * 2 + 1]
                                                                + value_1
                                                                * dense_weight[dense_off]
                                                            )
                                                    pack_words = []
                                                    for word_off in cutlass.range_constexpr(
                                                        0, 4, 1
                                                    ):
                                                        pack_words.append(
                                                            pack_f32x2_to_bf16x2(
                                                                pack_acc[word_off * 2],
                                                                pack_acc[word_off * 2 + 1],
                                                            )
                                                        )
                                                    pack_output_addr = (
                                                        reduce_addr
                                                        + (
                                                            Int64(chunk_base)
                                                            + Int64(chunk_off)
                                                        ) * Int64(pack_chunk_bytes)
                                                        + Int64(lane_idx) * Int64(16)
                                                    )
                                                    stg_b128_words_raw(
                                                        pack_output_addr,
                                                        pack_words[0],
                                                        pack_words[1],
                                                        pack_words[2],
                                                        pack_words[3],
                                                    )
                                            next_base = chunk_base + Int32(
                                                (wide_ring_rounds - 1) * chunks_per_round
                                            )
                                            refill_slot_base = wide_ring_lane_base + (
                                                (ring_round + Int32(wide_ring_rounds - 1))
                                                % Int32(wide_ring_rounds)
                                            ) * Int32(ring_round_bytes)
                                            for chunk_off in cutlass.range_constexpr(
                                                0, chunks_per_round, 1
                                            ):
                                                ring_chunk_ok = Int32(
                                                    next_base + Int32(chunk_off)
                                                    < Int32(pack_total_chunks)
                                                )
                                                if cutlass.const_expr(pack_tail_lanes > 0):
                                                    ring_chunk_ok = Int32(
                                                        (next_base + Int32(chunk_off)) * Int32(pack_chunk_bytes)
                                                        + lane_idx * Int32(16)
                                                        < Int32(combine_bytes)
                                                    )
                                                for dense_off in cutlass.range_constexpr(
                                                    0, hit_case, 1
                                                ):
                                                    slot = chunk_off * hit_case + dense_off
                                                    # An invalid chunk keeps the row base address and
                                                    # zero-fills (src-size 0), so nothing past the row
                                                    # is ever read.
                                                    cp_async_16_smem_raw(
                                                        refill_slot_base + Int32(slot * 16),
                                                        dense_addr[dense_off]
                                                        + Int64(ring_chunk_ok)
                                                        * (Int64(next_base) + Int64(chunk_off))
                                                        * Int64(pack_chunk_bytes),
                                                        ring_chunk_ok,
                                                    )
                                            cp_async_commit_group_raw()
                                            chunk_base = chunk_base + Int32(
                                                chunks_per_round
                                            )
                                            ring_round = ring_round + Int32(1)
                                        cp_async_wait_group_raw(0)
                                        scratch_round = Int32(0)
                                if scratch_round != Int32(0):
                                    if cutlass.const_expr(hit_case > 0):
                                        pack_addrs = [
                                            message_addr for _ in range(4)
                                        ]
                                        pack_valid = [
                                            Int32(0) for _ in range(4)
                                        ]
                                        for chunk_off in cutlass.range_constexpr(
                                            0, chunks_per_round, 1
                                        ):
                                            chunk_ok = Int32(
                                                chunk_base + Int32(chunk_off)
                                                < Int32(pack_total_chunks)
                                            )
                                            if cutlass.const_expr(pack_tail_lanes > 0):
                                                chunk_ok = Int32(
                                                    (chunk_base + Int32(chunk_off)) * Int32(pack_chunk_bytes)
                                                    + lane_idx * Int32(16)
                                                    < Int32(combine_bytes)
                                                )
                                            for dense_off in cutlass.range_constexpr(
                                                0, hit_case, 1
                                            ):
                                                slot = (
                                                    chunk_off * hit_case
                                                    + dense_off
                                                )
                                                pack_addrs[slot] = (
                                                    dense_addr[dense_off]
                                                    + (
                                                        Int64(chunk_base)
                                                        + Int64(chunk_off)
                                                    ) * Int64(pack_chunk_bytes)
                                                )
                                                pack_valid[slot] = chunk_ok
                                        ldg_b128x4_to_smem_predicated_raw(
                                            combine_b128_lane_scratch,
                                            pack_addrs[0],
                                            pack_addrs[1],
                                            pack_addrs[2],
                                            pack_addrs[3],
                                            pack_valid[0],
                                            pack_valid[1],
                                            pack_valid[2],
                                            pack_valid[3],
                                        )
                                    for chunk_off in cutlass.range_constexpr(
                                        0, chunks_per_round, 1
                                    ):
                                        lane_chunk_ok = Int32(
                                            chunk_base + Int32(chunk_off)
                                            < Int32(pack_total_chunks)
                                        )
                                        if cutlass.const_expr(pack_tail_lanes > 0):
                                            lane_chunk_ok = Int32(
                                                (chunk_base + Int32(chunk_off)) * Int32(pack_chunk_bytes)
                                                + lane_idx * Int32(16)
                                                < Int32(combine_bytes)
                                            )
                                        if lane_chunk_ok != Int32(0):
                                            pack_acc = [
                                                Float32(0.0) for _ in range(8)
                                            ]
                                            for dense_off in cutlass.range_constexpr(
                                                0, hit_case, 1
                                            ):
                                                slot = (
                                                    chunk_off * hit_case
                                                    + dense_off
                                                )
                                                for word_off in cutlass.range_constexpr(
                                                    0, 4, 1
                                                ):
                                                    value_0, value_1 = (
                                                        unpack_bf16x2_to_f32x2(
                                                            ldsm_b32_raw(
                                                                combine_b128_lane_scratch
                                                                + Int32(
                                                                    slot * 16
                                                                    + word_off * 4
                                                                )
                                                            )
                                                        )
                                                    )
                                                    pack_acc[word_off * 2] = (
                                                        pack_acc[word_off * 2]
                                                        + value_0
                                                        * dense_weight[dense_off]
                                                    )
                                                    pack_acc[word_off * 2 + 1] = (
                                                        pack_acc[word_off * 2 + 1]
                                                        + value_1
                                                        * dense_weight[dense_off]
                                                    )
                                            pack_words = []
                                            for word_off in cutlass.range_constexpr(
                                                0, 4, 1
                                            ):
                                                pack_words.append(
                                                    pack_f32x2_to_bf16x2(
                                                        pack_acc[word_off * 2],
                                                        pack_acc[word_off * 2 + 1],
                                                    )
                                                )
                                            pack_output_addr = (
                                                reduce_addr
                                                + (
                                                    Int64(chunk_base)
                                                    + Int64(chunk_off)
                                                ) * Int64(pack_chunk_bytes)
                                                + Int64(lane_idx) * Int64(16)
                                            )
                                            stg_b128_words_raw(
                                                pack_output_addr,
                                                pack_words[0],
                                                pack_words[1],
                                                pack_words[2],
                                                pack_words[3],
                                            )
                                    chunk_base = chunk_base + Int32(
                                        chunks_per_round
                                    )
                                    if cutlass.const_expr(hit_case > 0 and wide_ring_rounds > 0 and (wide_ring_smem is not None)):
                                        if chunk_base < Int32(pack_total_chunks):
                                            ring_on = ld_shared_acquire_cta_b32_raw(
                                                fc_smem_free_addr
                                            )

                    if cutlass.const_expr(self.iket_detailed_ranges):
                        _iket.range_end(
                            combine_reduce_range,
                            IKET_MEGA_COMBINE_WORK_REDUCE,
                        )
                    if cutlass.const_expr(self.pipeline_timing_enabled):
                        combine_reduce_done = read_globaltimer()
                        combine_reduce_ns = (
                            combine_reduce_ns
                            + combine_reduce_done - combine_reduce_start
                        )
                        combine_last_reduce_start = combine_reduce_start
                        combine_last_reduce_end = combine_reduce_done
                        combine_last_reduce_ring = Int64(0)
                        if ring_on != Int32(0):
                            combine_last_reduce_ring = Int64(1)

                    if cutlass.const_expr(self.iket_detailed_ranges):
                        combine_publish_range = _iket.range_start(
                            "megamoe_combine_work",
                            IKET_MEGA_COMBINE_WORK_PUBLISH_ISSUE,
                        )
                    if cutlass.const_expr(self.pipeline_timing_enabled):
                        combine_publish_issue_start = read_globaltimer()

                    # Every lane publishes its generic-proxy STG before lane 0
                    # gives the local BF16 row to GIN.  This is intentionally
                    # the same flush -> strong-credit ordering as dispatch.
                    cute.arch.sync_warp()
                    if lane_idx == Int32(0):
                        cute.arch.fence_acq_rel_sys()
                    cute.arch.sync_warp()
                    cute.arch.sync_warp()
                    if cutlass.const_expr(self.pipeline_timing_enabled):
                        combine_publish_issue_end = read_globaltimer()
                        combine_publish_issue_ns = (
                            combine_publish_issue_ns
                            + combine_publish_issue_end
                            - combine_publish_issue_start
                        )
                    if cutlass.const_expr(self.iket_detailed_ranges):
                        _iket.range_end(
                            combine_publish_range,
                            IKET_MEGA_COMBINE_WORK_PUBLISH_ISSUE,
                        )
                    if cutlass.const_expr(self.iket_detailed_ranges):
                        _iket.range_end(
                            combine_message_range,
                            IKET_MEGA_WORK_COMBINE_MESSAGE,
                        )
                    message_slot = message_slot + combine_worker_count
                # num_messages is stable published split metadata, never an
                # arrival-derived counter (Invariant I8).
                source_worker_base = (
                    source_worker_base + num_messages
                ) % combine_worker_count
            if cutlass.const_expr(self.pipeline_timing_enabled):
                combine_issue_done = read_globaltimer()

            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.num_relay_threads,
            )
            any_active = Int32(0)
            for src_slot_off in cutlass.range_constexpr(
                0, self.num_remote_node_slots, 1
            ):
                if (cta_return_count + Int32(src_slot_off)).load() > Int32(0):
                    any_active = Int32(1)
            # Coalesced combine return. Every worker already reduced its
            # messages into intra_node_reduce_buf[src_slot][message_slot],
            # which is dense in [0, num_messages) per source node. After
            # the relay warps of all CTAs rendezvous, SM0 returns each
            # source node's range with one put into the origin's
            # slot-indexed landing plane
            # internode_combine_buffer[phase][contributor][0:num_messages),
            # then one flush and one strong credit of num_messages, which
            # is exactly the origin's expected target (Invariants I5, I8).
            # TopkReduce maps token -> slot through token_node_slot; the
            # FP32 accumulation tree is unchanged (Invariant I11).
            if cutlass.const_expr(self.iket_enabled):
                combine_coalesce_grid_sync_range = _iket.range_start(
                    "megamoe_grid_sync_wait",
                    IKET_MEGA_GRID_SYNC_WAIT_COMBINE_COALESCE,
                )
            software_grid_sync(
                grid_sync_counter.iterator + _RELAY_GRID_SYNC_SLOT,
                sm_idx,
                num_sms,
                relay_thread_idx,
                num_threads=self.num_relay_threads,
            )
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(
                    combine_coalesce_grid_sync_range,
                    IKET_MEGA_GRID_SYNC_WAIT_COMBINE_COALESCE,
                )
            if sm_idx == Int32(0):
                # Lanes 0..slots-1 of relay warp 0 each post one source slot's
                # coalesced return put from the same instruction stream (step
                # = slots, one iteration per lane), the pattern of the Stage 1
                # lane-parallel issue.
                combine_issue_active = Int32(0)
                if relay_warp_idx == Int32(0):
                    if lane_idx < Int32(self.num_remote_node_slots):
                        combine_issue_active = Int32(1)
                issue_slot = lane_idx
                issue_slot_step = Int32(self.num_remote_node_slots)
                if combine_issue_active != Int32(0):
                    while issue_slot < Int32(self.num_remote_node_slots):
                        src_slot = issue_slot
                        origin_node = src_slot
                        if src_slot >= my_node:
                            origin_node = src_slot + Int32(1)
                        contributor_slot_at_origin = my_node
                        if my_node > origin_node:
                            contributor_slot_at_origin = (
                                my_node - Int32(1)
                            )
                        num_messages = Int32(
                            internode_recv_splits[
                                src_slot, self.node_experts
                            ]
                        )
                        if num_messages > Int32(
                            self.max_relay_tokens_per_node
                        ):
                            if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                                cute.printf("[gin-guard] token_comm source-site 5939\n")
                            device_trap()
                        if num_messages > Int32(0):
                            range_bytes = num_messages * Int32(
                                combine_bytes
                            )
                            range_layout = cute.make_layout(
                                (range_bytes,), stride=(1,)
                            )
                            reduce_range_offset = Int64(
                                self.intra_node_reduce_buf_offset
                            ) + Int64(src_slot) * Int64(
                                self.max_relay_tokens_per_node
                            ) * Int64(combine_bytes)
                            landing_range_offset = Int64(
                                self.internode_combine_buffer_offset
                            ) + (
                                Int64(combine_phase)
                                * Int64(self.num_remote_node_slots)
                                + Int64(contributor_slot_at_origin)
                            ) * Int64(
                                self.max_relay_tokens_per_node
                            ) * Int64(combine_bytes)
                            src_range = cute.make_tensor(
                                cute.make_ptr(
                                    Uint8,
                                    recv_base + reduce_range_offset,
                                    AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                range_layout,
                            )
                            dst_range = cute.make_tensor(
                                cute.make_ptr(
                                    Uint8,
                                    recv_base + landing_range_offset,
                                    AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                range_layout,
                            )
                            gin.put(
                                team_rail,
                                origin_node,
                                gin_window,
                                dst_range,
                                gin_window,
                                src_range,
                                thread_coop,
                                is_signal=True,
                                signal_id=(
                                    Int32(self.icombine_arrival_signal_base)
                                    + contributor_slot_at_origin
                                ),
                                signal_op=_NCCL_GIN_SIGNAL_OP_ADD,
                                signal_op_arg=Int64(num_messages),
                                given_release=nccl_cute.ThreadScope.THREAD,
                            )
                        issue_slot = issue_slot + issue_slot_step
                if relay_warp_idx == Int32(0):
                    cute.arch.sync_warp()
                    if cutlass.const_expr(combine_issue_counter is not None):
                        # Token-back deferral: the return puts of this
                        # launch are posted (SM0 relay warp 0 only);
                        # release the deferred token-back pushes.
                        # Monotonic counter, one step per launch.
                        if lane_idx == Int32(0):
                            cute.arch.atomic_add(
                                combine_issue_counter.iterator,
                                Int32(1),
                                sem="relaxed",
                                scope="gpu",
                            )
                if cutlass.const_expr(self.pipeline_timing_enabled):
                    # SM0's issue edge is the range put; other CTAs keep
                    # the loop-exit edge recorded above.
                    combine_issue_done = read_globaltimer()
                # Early rail arrive (Invariant I8): the last return put
                # of this launch is issued on this context, and every
                # read of the generation's landed metadata and payload
                # on this rank is complete, so the rendezvous signal
                # can go now; kernel_tail only waits. The two-slot
                # phase derivation is internode_rail_barrier's, and the
                # counter is advanced only by that wait.
                if relay_warp_idx == Int32(0):
                    if lane_idx == Int32(0):
                        status = internode_barrier_counter[0] & Int32(3)
                        signal_phase = status & Int32(1)
                        signal_sign = status >> Int32(1)
                        signal_delta = Int64(1)
                        if signal_sign != Int32(0):
                            signal_delta = Int64(-1)
                        for peer_node_off in cutlass.range_constexpr(
                            0, self.num_nodes, 1
                        ):
                            peer_node = Int32(peer_node_off)
                            if peer_node != my_node:
                                gin.signal(
                                    team_rail,
                                    peer_node,
                                    True,
                                    Int32(
                                        self.internode_barrier_signal_base
                                    ) + signal_phase,
                                    self.gin_signal_op_add,
                                    signal_delta,
                                    thread_coop,
                                )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                combine_flush_credit_done = read_globaltimer()
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(combine_return_range)

            # Before the fused kernel can exit, certify every remote-node row
            # that this origin rank expects.  TopkReduce therefore needs only
            # stream ordering; it performs no device-side GIN wait.
            if sm_idx == Int32(0):
                if cutlass.const_expr(self.iket_enabled):
                    combine_signal_range = _iket.range_start(
                        "megamoe_wait", IKET_MEGA_WAIT_COMBINE_SIGNAL
                    )
                for dst_slot_off in cutlass.range_constexpr(
                    0, self.num_remote_node_slots, 1
                ):
                    dst_slot = Int32(dst_slot_off)
                    expected = Int32(
                        internode_send_count[
                            dst_slot, self.node_experts
                        ]
                    )
                    signal_id = Int32(
                        self.icombine_arrival_signal_base + dst_slot_off
                    )
                    shadow = gin.signal_shadow_pointer(signal=signal_id)
                    if relay_coop.thread_rank == Int32(0):
                        cute.arch.atomic_add(
                            shadow,
                            cutlass.Uint64(expected),
                            sem="relaxed",
                            scope="gpu",
                        )
                    cute.arch.barrier(
                        barrier_id=self.dispatch_intra_cta_bar_id,
                        number_of_threads=self.num_relay_threads,
                    )
                    if expected > Int32(0):
                        target = cute.arch.load(
                            shadow,
                            cutlass.Uint64,
                            sem="relaxed",
                            scope="gpu",
                        )
                        gin.wait_signal(
                            relay_coop,
                            signal=signal_id,
                            least=target,
                            bits=64,
                            ord=nccl_cute.MemoryOrder.ACQUIRE,
                        )
                if cutlass.const_expr(self.iket_enabled):
                    _iket.range_end(
                        combine_signal_range,
                        IKET_MEGA_WAIT_COMBINE_SIGNAL,
                    )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                combine_arrival_done = read_globaltimer()

            if cutlass.const_expr(self.iket_enabled):
                combine_grid_sync_range = _iket.range_start(
                    "megamoe_grid_sync_wait", IKET_MEGA_GRID_SYNC_WAIT_COMBINE
                )
            software_grid_sync(
                grid_sync_counter.iterator + _RELAY_GRID_SYNC_SLOT,
                sm_idx,
                num_sms,
                relay_thread_idx,
                num_threads=self.num_relay_threads,
            )
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(
                    combine_grid_sync_range,
                    IKET_MEGA_GRID_SYNC_WAIT_COMBINE,
                )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                combine_end = read_globaltimer()
                if lane_idx == Int32(0):
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_FC2_WAIT_START
                    ] = combine_start
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_ROUTE_DONE
                    ] = combine_route_done
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_REDUCE_DONE
                    ] = combine_reduce_done
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_ISSUE_DONE
                    ] = combine_issue_done
                    combine_timing_landmarks[
                        sm_idx,
                        warp_idx,
                        COMBINE_TIMING_FLUSH_CREDIT_DONE,
                    ] = combine_flush_credit_done
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_ARRIVAL_DONE
                    ] = combine_arrival_done
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_SYNC_DONE
                    ] = combine_end
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_FC2_WAIT_NS
                    ] = combine_fc2_wait_ns
                    combine_timing_landmarks[
                        sm_idx,
                        warp_idx,
                        COMBINE_TIMING_ROUTE_INCLUSIVE_NS,
                    ] = combine_route_inclusive_ns
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_REDUCE_NS
                    ] = combine_reduce_ns
                    combine_timing_landmarks[
                        sm_idx,
                        warp_idx,
                        COMBINE_TIMING_PUBLISH_ISSUE_NS,
                    ] = combine_publish_issue_ns
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_MESSAGE_COUNT
                    ] = combine_message_count
                    combine_timing_landmarks[
                        sm_idx,
                        warp_idx,
                        COMBINE_TIMING_WAITED_HIT_COUNT,
                    ] = combine_waited_hit_count
                    combine_timing_landmarks[
                        sm_idx, warp_idx, COMBINE_TIMING_MAX_FC2_WAIT_NS
                    ] = combine_max_fc2_wait_ns
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_COMBINE, 0
                    ] = combine_start
                    pipeline_timing[
                        sm_idx, warp_idx, PIPELINE_STAGE_COMBINE, 1
                    ] = combine_end
                    # Last message's gather-reduce interval, by the path it
                    # ended on (diagnostic; workers without a message leave
                    # both stages unset).
                    last_reduce_stage = Int32(PIPELINE_STAGE_COMBINE_REDUCE_SCRATCH)
                    if combine_last_reduce_ring != Int64(0):
                        last_reduce_stage = Int32(PIPELINE_STAGE_COMBINE_REDUCE_RING)
                    if combine_last_reduce_end != Int64(0):
                        pipeline_timing[
                            sm_idx, warp_idx, last_reduce_stage, 0
                        ] = combine_last_reduce_start
                        pipeline_timing[
                            sm_idx, warp_idx, last_reduce_stage, 1
                        ] = combine_last_reduce_end

    @cute.jit
    def _adaptive_pace(
        self,
        avg,
        current_window,
        *,
        lo: cutlass.Constexpr[int],
        hi: cutlass.Constexpr[int],
    ):
        # NVLink pacing: EMA the measured round-trip and nanosleep the deviation
        # so outstanding NVLink requests stay bounded and don't head-of-line
        # block this SM's non-NVLink (local) load/store traffic.
        if current_window > avg:
            avg = avg + ((current_window - avg + Int32(3)) // Int32(4))
            sleep_cycle = current_window - avg
            if sleep_cycle > Int32(hi):
                sleep_cycle = Int32(hi)
            if sleep_cycle > Int32(50):
                _nanosleep(sleep_cycle)
        else:
            avg = avg - ((avg - current_window + Int32(3)) // Int32(4))
            sleep_cycle = avg - current_window
            if sleep_cycle > Int32(50):
                _nanosleep(sleep_cycle)
        if avg > Int32(hi):
            avg = Int32(hi)
        if avg < Int32(lo):
            avg = Int32(lo)
        return avg

    @cute.jit
    def token_back_by_push(
        self,
        pull_buffer_ptr,
        pull_mbar_ptr,
        fc2_output_workspace,
        fc2_done_counter,
        token_src_metadata,
        combine_output,
        combine_sf,
        fc2_output_sf,
        token_back_schedule_counter,
        peer_rank_ptr_mapper,
        expert_pool_base,
        intra_token_count,
        relay_pool_base,
        internode_barrier_counter,
        phase_bit,
        stored_num_tokens_per_expert,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_sms,
        chunk_bytes: cutlass.Constexpr[int],
        num_token_back_warps: cutlass.Constexpr[int],
        pipeline_timing=None,
        combine_issue_counter=None,
    ):
        _iket_emit = (sm_idx == Int32(0)) and (warp_idx == Int32(0))
        avg_token_back_window = Int32(2500)

        # Chunk the fc2 token in ``chunk_bytes`` pieces; the last piece carries
        # the remainder so any chunk_bytes works for any fc2_token_bytes.
        fc2_token_bytes: cutlass.Constexpr[int] = self.fc2_token_bytes
        num_chunks: cutlass.Constexpr[int] = (
            fc2_token_bytes + chunk_bytes - 1
        ) // chunk_bytes
        last_chunk_bytes: cutlass.Constexpr[int] = (
            fc2_token_bytes - (num_chunks - 1) * chunk_bytes
        )

        if cutlass.const_expr(self.push_sf):
            # (token, topk, hidden):(d_topkxhidden, d_hidden, 1)
            combine_sf_u8 = cute.recast_tensor(combine_sf, Uint8)
            sf_token_bytes: cutlass.Constexpr[int] = cute.size(combine_sf_u8[0, None, 0].stride)
            num_sf_chunks: cutlass.Constexpr[int] = (
                sf_token_bytes + chunk_bytes - 1
            ) // chunk_bytes
            last_sf_chunk_bytes: cutlass.Constexpr[int] = (
                sf_token_bytes - (num_sf_chunks - 1) * chunk_bytes
            )

        num_experts_per_lane: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        num_global_warps: cutlass.Constexpr[int] = num_sms * num_token_back_warps
        schedule_mode = self.token_back_schedule_mode
        atomic_batch = self.token_back_atomic_batch

        # static: stride by the global warp count.  atomic_counter: consume one
        # slot of the current batch, refilling via one grid-scoped
        # atomicAdd(atomic_batch) when exhausted so fast warps keep stealing
        # work.  cuTeDSL forbids closures over enclosing locals -> pass all in.
        def update_token_idx(
            token_idx, batch_remaining, lane_idx, schedule_counter,
            schedule_mode, atomic_batch, num_global_warps,
        ):
            if cutlass.const_expr(schedule_mode == "atomic_counter"):
                batch_remaining = batch_remaining - Int32(1)
                if batch_remaining == Int32(0):
                    base = Int32(0)
                    if lane_idx == Int32(0):
                        base = cute.arch.atomic_add(
                            schedule_counter, Int32(atomic_batch),
                            sem="relaxed", scope="gpu",
                        )
                    token_idx = cute.arch.shuffle_sync(base, Int32(0))
                    batch_remaining = Int32(atomic_batch)
                else:
                    token_idx = token_idx + Int32(1)
            else:
                token_idx = token_idx + Int32(num_global_warps)
            return token_idx, batch_remaining

        if cutlass.const_expr(schedule_mode == "atomic_counter"):
            # Prime the first batch: batch_remaining=1 makes update_token_idx
            # decrement to 0 and pull the initial atomic batch.
            token_idx = Int32(0)
            batch_remaining = Int32(1)
            token_idx, batch_remaining = update_token_idx(
                token_idx, batch_remaining, lane_idx,
                token_back_schedule_counter,
                schedule_mode, atomic_batch, num_global_warps,
            )
        else:
            token_idx = sm_idx * Int32(num_token_back_warps) + warp_idx
            batch_remaining = Int32(0)

        current_expert_idx = Int32(-1)
        confirmed_expert_idx = Int32(-1)
        cur_expert_expected = Int32(0)
        expert_start_idx = Int32(0)
        expert_end_idx = Int32(0)
        expert_pool_block_offset = Int32(0)

        # Defer pushes until SM0 posts this launch's combine return puts.
        # The rail generation advances once in kernel_tail, after all token-back
        # warps retire. It is therefore stable here even when an empty/short
        # relay loop has already incremented combine_issue_counter. Deriving
        # the target from that issue counter races the producer and can wait
        # forever for the NEXT launch instead of the current one.
        tb_defer_on: cutlass.Constexpr[bool] = combine_issue_counter is not None
        if cutlass.const_expr(tb_defer_on):
            tb_gate_ptr = combine_issue_counter.iterator
            tb_gate_base = cute.arch.load(
                internode_barrier_counter.iterator, Int32, sem="acquire", scope="gpu",
            )
            tb_gate_target = tb_gate_base + Int32(1)
            tb_gate_wait_start = read_clock64()
            tb_gate_cur = cute.arch.load(
                tb_gate_ptr, Int32, sem="acquire", scope="gpu",
            )
            while tb_gate_cur < tb_gate_target:
                _nanosleep(1000)
                tb_gate_cur = cute.arch.load(
                    tb_gate_ptr, Int32, sem="acquire", scope="gpu",
                )
                if (
                    read_clock64() - tb_gate_wait_start
                    > Int64(self.rail_barrier_timeout_cycles)
                ):
                    if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                        cute.printf("[gin-guard] token_comm source-site 6357\n")
                    device_trap()
        if cutlass.const_expr(
            self.pipeline_timing_enabled and pipeline_timing is not None
        ):
            # Stage token_back: from the (possibly deferred) start of this
            # warp's pushes to its last push.
            if lane_idx == Int32(0):
                pipeline_timing[
                    sm_idx, warp_idx, PIPELINE_STAGE_TOKEN_BACK, 0
                ] = read_globaltimer()

        while current_expert_idx < Int32(self.num_experts_per_rank):
            while (token_idx >= expert_end_idx) and (
                current_expert_idx < Int32(self.num_experts_per_rank)
            ):
                prev_valid_count = expert_end_idx - expert_start_idx
                prev_block_count = (
                    prev_valid_count + Int32(self.token_padding_block) - Int32(1)
                ) // Int32(self.token_padding_block)
                expert_pool_block_offset = (
                    expert_pool_block_offset + prev_block_count
                )

                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    expert_start_idx = expert_end_idx
                    valid_value = Int32(0)
                    for i in cutlass.range_constexpr(
                        0, num_experts_per_lane, 1
                    ):
                        if current_expert_idx == Int32(
                            i * self.warp_threads
                        ) + lane_idx:
                            valid_value = stored_num_tokens_per_expert[i]
                    total_for_expert = cute.arch.shuffle_sync(
                        valid_value,
                        current_expert_idx % Int32(self.warp_threads),
                    )
                    expert_end_idx = expert_end_idx + total_for_expert

                    cluster_tile_cnt = (
                        total_for_expert
                        + Int32(self.cluster_tile_tokens)
                        - Int32(1)
                    ) // Int32(self.cluster_tile_tokens)
                    if cutlass.const_expr(intra_token_count is not None):
                        intra_for_expert = Int32(
                            intra_token_count[current_expert_idx]
                        )
                        relay_for_expert = cutlass.max(
                            total_for_expert - intra_for_expert, Int32(0)
                        )
                        cluster_tile_cnt = expert_token_tiles(
                            intra_for_expert, relay_for_expert,
                            self.cluster_tile_tokens,
                        )
                    # Stash the threshold; the wait is deferred to the expert we
                    # actually land on, so stepped-over experts are never waited.
                    cur_expert_expected = cluster_tile_cnt * Int32(
                        self.fc2_publishes_per_token_cluster_tile
                    )
                    if cutlass.const_expr(
                        self.fc2_publishes_per_split_tail_tile > 0
                    ):
                        # Tail-split pair tasks: the last cluster tile of an expert with an odd CTA
                        # tile count publishes fc2_publishes_per_split_tail_tile times.
                        cta_tile_cnt = (
                            total_for_expert
                            + Int32(self.cta_tile_tokens - 1)
                        ) // Int32(self.cta_tile_tokens)
                        if (cta_tile_cnt & Int32(1)) == Int32(1):
                            cur_expert_expected = (
                                cur_expert_expected
                                - Int32(self.fc2_publishes_per_token_cluster_tile)
                                + Int32(self.fc2_publishes_per_split_tail_tile)
                            )

            if current_expert_idx < Int32(self.num_experts_per_rank):
                # Wait once per processed expert (both indices monotonic; fc2
                # completes in expert order so confirming k implies all < k).
                if current_expert_idx > confirmed_expert_idx:
                    spin_wait(
                        fc2_done_counter.iterator + current_expert_idx,
                        lambda v: v >= cur_expert_expected,
                        fail_sleep_cycles=500,
                    )
                    confirmed_expert_idx = current_expert_idx

                remain_experts = Int32(self.num_experts_per_rank) - current_expert_idx
                token_idx_in_expert = token_idx - expert_start_idx
                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block)
                    + token_idx_in_expert
                )
                if cutlass.const_expr(expert_pool_base is not None):
                    pool_token_idx = (
                        Int32(expert_pool_base[current_expert_idx])
                        + token_idx_in_expert
                    )
                if cutlass.const_expr(intra_token_count is not None):
                    intra_for_expert = Int32(
                        intra_token_count[current_expert_idx]
                    )
                    if token_idx_in_expert >= intra_for_expert:
                        pool_token_idx = (
                            Int32(
                                relay_pool_base[
                                    current_expert_idx, Int32(0)
                                ]
                            )
                            + token_idx_in_expert
                            - intra_for_expert
                        )

                md = TokenSrcMetadata.load(
                    token_src_metadata.iterator
                    + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                src_rank = md.src_rank
                src_token = md.src_token
                src_topk = md.src_topk
                is_remote_token_back = src_rank != Int32(local_rank)
                is_internode_origin = (
                    src_rank // Int32(self.ranks_per_node)
                    != Int32(local_rank) // Int32(self.ranks_per_node)
                )

                smem_ptr_warp = pull_buffer_ptr + warp_idx * Int32(chunk_bytes)
                mbar_ptr_warp = pull_mbar_ptr + warp_idx

                if _iket_emit:
                    _iket.range_push("token_back")
                cute.arch.sync_warp()

                if not is_internode_origin:
                    # DATA plane: only the dispatch DATA path pushes here; epi_warps
                    # has the epilogue STG/UBLK the data straight to the peer.
                    if cutlass.const_expr(self.push_data):
                        local_token_addr = (
                            fc2_output_workspace.iterator.toint()
                            + Int64(pool_token_idx) * Int64(fc2_token_bytes)
                        )
                        peer_combine_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                            combine_output.iterator, src_rank,
                        )
                        if cutlass.const_expr(self.token_back_reduce_topk):
                            peer_token_offset = Int64(src_token) * Int64(fc2_token_bytes)
                        else:
                            peer_token_offset = (
                                Int64(src_token * Int32(self.num_topk) + src_topk)
                                * Int64(fc2_token_bytes)
                            )
                            if cutlass.const_expr(self.enable_nccl_gin):
                                # All ranks choose the phase from the synchronized
                                # rail generation at launch entry.  The peer's
                                # local-origin staging has shape [2,T,K,H].
                                combine_phase = (
                                    internode_barrier_counter[0] & Int32(1)
                                )
                                phase_stride_bytes: cutlass.Constexpr[int] = (
                                    self.max_relay_tokens_per_node
                                    * self.num_topk
                                    * fc2_token_bytes
                                )
                                peer_token_offset = (
                                    peer_token_offset
                                    + Int64(combine_phase)
                                    * Int64(phase_stride_bytes)
                                )
                        peer_token_ptr = peer_combine_ptr + peer_token_offset

                        for chunk in cutlass.range(num_chunks, unroll=1):
                            t0 = read_clock64()
                            chunk_off = Int64(chunk * chunk_bytes)
                            peer_chunk_ptr = peer_token_ptr + chunk_off

                            this_bytes = Int32(chunk_bytes)
                            if cutlass.const_expr(last_chunk_bytes != chunk_bytes):
                                if chunk == Int32(num_chunks - 1):
                                    this_bytes = Int32(last_chunk_bytes)

                            with cute.arch.elect_one():
                                tma_load_1d_raw(
                                    smem_ptr_warp,
                                    local_token_addr + chunk_off,
                                    mbar_ptr_warp,
                                    this_bytes,
                                )
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar_ptr_warp, this_bytes,
                                )
                                cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                                if cutlass.const_expr(self.token_back_reduce_topk):
                                    cp_reduce_async_bulk_add_noftz_bf16_s2g(
                                        peer_chunk_ptr,
                                        smem_ptr_warp,
                                        this_bytes,
                                    )
                                else:
                                    tma_store_1d(
                                        peer_chunk_ptr,
                                        smem_ptr_warp,
                                        this_bytes,
                                    )
                            phase_bit = phase_bit ^ Int32(1)
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0)
                            t1 = read_clock64()
                            current_window = Int32(t1 - t0)
                            if is_remote_token_back and remain_experts > Int32(4):
                                avg_token_back_window = self._adaptive_pace(
                                    avg_token_back_window, current_window, lo=1000, hi=5000,
                                )

                    if cutlass.const_expr(self.push_sf):
                        sf_local_addr = fc2_output_sf[pool_token_idx, 0, None].iterator.toint()
                        sf_peer_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                            combine_sf_u8[src_token, src_topk, None].iterator, src_rank,
                        )
                        for chunk in cutlass.range(num_sf_chunks, unroll=1):
                            t0 = read_clock64()
                            chunk_off = Int64(chunk * chunk_bytes)
                            this_bytes = Int32(chunk_bytes)
                            if cutlass.const_expr(last_sf_chunk_bytes != chunk_bytes):
                                if chunk == Int32(num_sf_chunks - 1):
                                    this_bytes = Int32(last_sf_chunk_bytes)
                            with cute.arch.elect_one():
                                tma_load_1d_raw(
                                    smem_ptr_warp,
                                    sf_local_addr + chunk_off,
                                    mbar_ptr_warp,
                                    this_bytes,
                                )
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar_ptr_warp, this_bytes,
                                )
                                cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                                tma_store_1d(
                                    sf_peer_ptr + chunk_off,
                                    smem_ptr_warp,
                                    this_bytes,
                                )
                            phase_bit = phase_bit ^ Int32(1)
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0)
                            if is_remote_token_back:
                                round_trip = Int32(read_clock64() - t0)
                                if round_trip <= this_bytes * Int32(5) // Int32(4):
                                    _nanosleep(this_bytes // Int32(4))

                if _iket_emit:
                    _iket.range_pop()

                token_idx, batch_remaining = update_token_idx(
                    token_idx, batch_remaining, lane_idx,
                    token_back_schedule_counter,
                    schedule_mode, atomic_batch, num_global_warps,
                )

        if cutlass.const_expr(
            self.pipeline_timing_enabled and pipeline_timing is not None
        ):
            if lane_idx == Int32(0):
                pipeline_timing[
                    sm_idx, warp_idx, PIPELINE_STAGE_TOKEN_BACK, 1
                ] = read_globaltimer()
        cute.arch.fence_acq_rel_sys()
        # _fence_rel_sys()

    @cute.jit
    def nvlink_barrier(
        self,
        nvlink_barrier_signal,
        nvlink_barrier_counter,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
        local_rank=0,
        sync_num_threads: cutlass.Constexpr[int] = None,
        prologue_grid_sync: cutlass.Constexpr[bool],
        epilogue_grid_sync: cutlass.Constexpr[bool],
        sync_barrier_id: cutlass.Constexpr[int] = 10,
        sync_threads: cutlass.Constexpr[int] = 0,
    ):
        # Preserve FI's intranode helper ABI for probes and shared callers.
        # GIN sites pass their group size and global rank explicitly.
        if cutlass.const_expr(sync_num_threads is None):
            sync_num_threads = sync_threads if sync_threads > 0 else self.num_dispatch_threads
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        if prologue_grid_sync:
            if cutlass.const_expr(self.iket_enabled):
                nvlink_prologue_grid_sync_range = _iket.range_start(
                    "megamoe_grid_sync_wait",
                    IKET_MEGA_GRID_SYNC_WAIT_NVLINK_PROLOGUE,
                )
            software_grid_sync(
                grid_sync_counter, sm_idx, num_sms, tid_in_group,
                num_threads=sync_num_threads, barrier_id=sync_barrier_id,
            )
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(
                    nvlink_prologue_grid_sync_range,
                    IKET_MEGA_GRID_SYNC_WAIT_NVLINK_PROLOGUE,
                )

        # Keep the peer-signal wait disjoint from both grid-sync ranges. Its
        # endpoints remain outside the lane-level control flow below.
        if cutlass.const_expr(self.iket_enabled):
            nvlink_barrier_range = _iket.range_start("nvlink_barrier_wait")
        if sm_idx == 0:
            if warp_idx == 0:
                # Sense-reversing ping-pong barrier. The low 2 bits of the counter
                # pick the signal slot (phase 0/1) and the direction (+1 up to
                # ranks_per_node, then -1 back to 0), so the two slots self-cancel
                # over a 4-call cycle and never need an explicit reset -- required
                # because the signal is symmetric peer memory that ncu kernel
                # replay cannot snapshot/restore.
                status = nvlink_barrier_counter[0] & Int32(3)
                signal_phase = status & Int32(1)
                signal_sign = status >> Int32(1)
                signal_delta = Int32(1)
                target = Int32(self.ranks_per_node)
                if signal_sign != Int32(0):
                    signal_delta = Int32(-1)
                    target = Int32(0)

                nbs_local_base = nvlink_barrier_signal.iterator.toint()
                if lane_idx < Int32(self.ranks_per_node):
                    lane_rank = lane_idx
                    if cutlass.const_expr(self.num_remote_nodes > 0):
                        node_base_rank = (
                            Int32(local_rank) // Int32(self.ranks_per_node)
                        ) * Int32(self.ranks_per_node)
                        lane_rank = node_base_rank + lane_idx
                    lane_peer_addr = peer_rank_ptr_mapper.map(
                        nbs_local_base, lane_rank,
                        Int64(signal_phase * Int32(4)),
                    )
                    red_add_release_sys_s32_raw(lane_peer_addr, signal_delta)
                cute.arch.sync_warp()

                if lane_idx == 0:
                    cute.arch.atomic_add(
                        nvlink_barrier_counter.iterator,
                        Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    local_signal_ptr = nvlink_barrier_signal.iterator + signal_phase
                    while cute.arch.load(local_signal_ptr, Int32, sem="acquire", scope="sys") != target:
                        pass
        if cutlass.const_expr(self.iket_enabled):
            _iket.range_end(nvlink_barrier_range)

        if epilogue_grid_sync:
            if cutlass.const_expr(self.iket_enabled):
                nvlink_epilogue_grid_sync_range = _iket.range_start(
                    "megamoe_grid_sync_wait",
                    IKET_MEGA_GRID_SYNC_WAIT_NVLINK_EPILOGUE,
                )
            software_grid_sync(
                grid_sync_counter, sm_idx, num_sms, tid_in_group,
                num_threads=sync_num_threads, barrier_id=sync_barrier_id,
            )
            if cutlass.const_expr(self.iket_enabled):
                _iket.range_end(
                    nvlink_epilogue_grid_sync_range,
                    IKET_MEGA_GRID_SYNC_WAIT_NVLINK_EPILOGUE,
                )

    @cute.jit
    def internode_rail_barrier(
        self,
        internode_barrier_counter,
        grid_sync_counter,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
        local_rank,
        arrived: cutlass.Constexpr[bool] = False,
    ):
        # All dispatch warps bracket the single SM0 GIN participant with grid
        # syncs. The two signal slots alternate phase and direction, returning
        # to their entry values every four calls so kernel replay cannot inherit
        # an already-satisfied barrier.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx
        if cutlass.const_expr(self.iket_enabled):
            rail_prologue_grid_sync_range = _iket.range_start(
                "megamoe_grid_sync_wait", IKET_MEGA_GRID_SYNC_WAIT_RAIL_PROLOGUE
            )
        software_grid_sync(
            grid_sync_counter,
            sm_idx,
            num_sms,
            tid_in_group,
            num_threads=self.num_dispatch_threads,
        )
        if cutlass.const_expr(self.iket_enabled):
            _iket.range_end(
                rail_prologue_grid_sync_range,
                IKET_MEGA_GRID_SYNC_WAIT_RAIL_PROLOGUE,
            )

        if cutlass.const_expr(self.iket_enabled):
            rail_barrier_range = _iket.range_start(
                "megamoe_wait", IKET_MEGA_WAIT_RAIL_BARRIER
            )
        if (
            sm_idx == Int32(0)
            and warp_idx == Int32(0)
            and lane_idx == Int32(0)
        ):
            from nccl.core.device import cute as nccl_cute

            gin = self._device_gin_thread()
            team_rail = self._team_rail_value
            thread_coop = nccl_cute.thread()
            status = internode_barrier_counter[0] & Int32(3)
            signal_phase = status & Int32(1)
            signal_sign = status >> Int32(1)
            signal_delta = Int64(1)
            target = Int64(self.num_remote_nodes)
            if signal_sign != Int32(0):
                signal_delta = Int64(-1)
                target = Int64(0)

            # Every metadata put carries its own weak completion signal, and a
            # rank cannot reach this tail until it has observed every inbound
            # completion. Once all rail peers reach this barrier, each sender's
            # registered vector has therefore been observed complete by every
            # destination. The standalone SignalAdd below is only a generation
            # rendezvous; it does not need strong-signal drain semantics. Do not
            # call gin.flush here: NCCL 2.30.7 faults when it is inlined in this
            # fused persistent kernel.
            if cutlass.const_expr(not arrived):
                my_node = Int32(local_rank) // Int32(self.ranks_per_node)
                for peer_node_off in cutlass.range_constexpr(
                    0, self.num_nodes, 1
                ):
                    peer_node = Int32(peer_node_off)
                    if peer_node != my_node:
                        gin.signal(
                            team_rail,
                            peer_node,
                            True,
                            Int32(
                                self.internode_barrier_signal_base
                            ) + signal_phase,
                            self.gin_signal_op_add,
                            signal_delta,
                            thread_coop,
                        )

            cute.arch.atomic_add(
                internode_barrier_counter.iterator,
                Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            barrier_start = read_clock64()
            while gin.read_signal(
                signal=Int32(self.internode_barrier_signal_base)
                + signal_phase,
                bits=64,
                ord=nccl_cute.MemoryOrder.ACQUIRE,
            ) != target:
                if (
                    read_clock64() - barrier_start
                    > Int64(self.rail_barrier_timeout_cycles)
                ):
                    if cutlass.const_expr(os.environ.get("FLASHINFER_GIN_GUARD_DIAGNOSTICS", "0") == "1"):
                        cute.printf("[gin-guard] token_comm source-site 6835\n")
                    device_trap()
        if cutlass.const_expr(self.iket_enabled):
            _iket.range_end(
                rail_barrier_range, IKET_MEGA_WAIT_RAIL_BARRIER
            )

        if cutlass.const_expr(self.iket_enabled):
            rail_epilogue_grid_sync_range = _iket.range_start(
                "megamoe_grid_sync_wait", IKET_MEGA_GRID_SYNC_WAIT_RAIL_EPILOGUE
            )
        software_grid_sync(
            grid_sync_counter,
            sm_idx,
            num_sms,
            tid_in_group,
            num_threads=self.num_dispatch_threads,
        )
        if cutlass.const_expr(self.iket_enabled):
            _iket.range_end(
                rail_epilogue_grid_sync_range,
                IKET_MEGA_GRID_SYNC_WAIT_RAIL_EPILOGUE,
            )

    @cute.jit
    def reset_internode_split_signals_for_ncu(
        self,
        sm_idx,
        warp_idx,
        lane_idx,
    ):
        if (
            sm_idx == Int32(0)
            and warp_idx == Int32(0)
            and lane_idx == Int32(0)
        ):
            from nccl.core.device import cute as nccl_cute

            gin = self._device_gin_thread()
            # The splits signals and shadows belong to the Step 3
            # context; reset them through that handle.
            step3_gin = self._device_gin_gpu()
            for src_slot_off in cutlass.range_constexpr(
                0, self.num_remote_node_slots, 1
            ):
                step3_gin.reset_signal(
                    signal=Int32(
                        self.internode_splits_signal_base + src_slot_off
                    )
                )
                gin.reset_signal(
                    signal=Int32(
                        self.relay_node_signal_base + src_slot_off
                    )
                )
                gin.reset_signal(
                    signal=Int32(
                        self.icombine_arrival_signal_base + src_slot_off
                    )
                )

    @cute.jit
    def dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        cta_linear_id = self._cta_linear_id()
        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
        if cutlass.const_expr(self.pipeline_timing_enabled):
            dispatch_start = read_globaltimer()


        # Record all four dispatch warps in CTA 0. Recording every persistent
        # CTA duplicates the same role and makes PIC-C's trace buffer too large.
        iket_active = cta_linear_id == Int32(0)
        if iket_active:
            _iket.range_push("Dispatch_Prep")

        self.dispatch_prep(
            token_comm_storage,
            token_comm_args.topk_idx,
            token_comm_args.expert_send_count,
            token_comm_args.src_token_topk_idx,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            local_rank=token_comm_args.local_rank,
            num_tokens=token_comm_args.input_token_buffer.shape[0],
            num_sms=token_comm_args.sm_count,
            route_extent=token_comm_args.dispatch_route_extent,
            cta_expert_hist=token_comm_args.cta_expert_hist,
            grid_sync_counter=token_comm_args.grid_sync_counter,
        )

        if iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Barrier")

        self.dispatch_barrier(
            token_comm_args.expert_send_count,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.nvlink_barrier_signal,
            token_comm_args.grid_sync_counter,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            local_rank=token_comm_args.local_rank,
            num_sms=token_comm_args.sm_count,
            nvlink_barrier_counter=token_comm_args.nvlink_barrier_counter,
            dispatch_only=(
                self.relay_skip_dispatch_barrier and self.num_relay_warps > 0
            ),
        )

        # Keep the intranode specialization's trace/call graph identical to
        # the pre-relay path. The callee also guards its body, but gating at
        # this boundary prevents relay-only arguments and control flow from
        # entering the no-relay specialization in the first place.
        if cutlass.const_expr(self.num_relay_warps > 0):
            self.inter_meta_exchange(
                token_comm_args.expert_recv_count,
                token_comm_args.expert_recv_count_sum,
                token_comm_args.internode_send_count,
                token_comm_args.internode_recv_splits,
                token_comm_args.internode_send_token_count,
                token_comm_args.intra_token_count,
                token_comm_args.internode_recv_count_per_source,
                token_comm_args.expert_pool_base,
                token_comm_args.expert_sf_pool_base,
                token_comm_args.relay_pool_base,
                token_comm_args.relay_sf_pool_base,
                token_comm_args.relay_direct_bucket_meta,
                token_comm_args.internode_recv_count_per_source_snapshot,
                token_comm_args.expert_recv_count_snapshot,
                token_comm_args.expert_recv_count_sum_snapshot,
                token_comm_args.intra_token_count_snapshot,
                token_comm_args.expert_pool_base_snapshot,
                token_comm_args.expert_sf_pool_base_snapshot,
                token_comm_args.relay_pool_base_snapshot,
                token_comm_args.relay_sf_pool_base_snapshot,
                token_comm_args.expert_scheduled_count_snapshot,
                token_comm_args.pool_end_snapshot,
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                token_comm_args.gin_window,
                token_comm_args.pipeline_timing,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                local_rank=token_comm_args.local_rank,
                num_sms=token_comm_args.sm_count,
                pool_token_capacity=token_comm_args.pool_token_capacity,
                pool_sf_capacity=token_comm_args.pool_sf_capacity,
            )

        nb_dispatch_to_sched = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb_dispatch_to_sched.arrive()

        if iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Pull")

        phase_bit, stored_num_tokens_per_expert = self.dispatch_pull(
            token_comm_storage,
            token_comm_args.input_token_buffer,
            token_comm_args.input_sf_buffer,
            token_comm_args.input_topk_weights_buffer,
            token_comm_args.src_token_topk_idx,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.intra_token_count,
            token_comm_args.expert_pool_base,
            token_comm_args.expert_sf_pool_base,
            token_comm_args.fc1_input_token_buffer,
            token_comm_args.fc1_input_sf_buffer,
            token_comm_args.fc1_input_topk_weights_buffer,
            token_comm_args.fc1_ready_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
            input_fc1_activation_per_token_scale_buffer=(
                token_comm_args.input_fc1_activation_per_token_scale_buffer
            ),
            dispatched_fc1_activation_per_token_scale_buffer=(
                token_comm_args.dispatched_fc1_activation_per_token_scale_buffer
            ),
        )

        if iket_active:
            _iket.range_pop()  # Dispatch_Pull

        if cutlass.const_expr(self.enable_nccl_gin):
            # Experimental direct_l1_sf_staging: the dispatch warps
            # materialize the staged relay scale-factor rows and release
            # their readiness credits (no-op when the option is off).
            self.relay_sf_materialize(
                token_comm_args.relay_row_ready,
                token_comm_args.relay_sf_staging,
                token_comm_args.internode_recv_count_per_source,
                token_comm_args.relay_pool_base,
                token_comm_args.relay_sf_pool_base,
                token_comm_args.fc1_input_sf_buffer,
                token_comm_args.fc1_ready_counter,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
                pipeline_timing=token_comm_args.pipeline_timing,
            )
        if cutlass.const_expr(self.pipeline_timing_enabled):
            dispatch_end = read_globaltimer()
            if lane_idx == Int32(0):
                token_comm_args.pipeline_timing[
                    cta_linear_id, warp_idx, PIPELINE_STAGE_DISPATCH, 0
                ] = dispatch_start
                token_comm_args.pipeline_timing[
                    cta_linear_id, warp_idx, PIPELINE_STAGE_DISPATCH, 1
                ] = dispatch_end


        if cutlass.const_expr(
            self.enable_token_back and not self.token_back_standalone
        ):
            if iket_active:
                _iket.range_push("Token_Back_By_Push")

            self.token_back_by_push(
                token_comm_storage.pull_buffer.data_ptr(),
                token_comm_storage.pull_mbar.data_ptr(),
                token_comm_args.fc2_output_workspace,
                token_comm_args.fc2_done_counter,
                token_comm_args.token_src_metadata,
                token_comm_args.combine_output,
                token_comm_args.combine_sf,
                token_comm_args.fc2_output_sf,
                token_comm_args.token_back_schedule_counter,
                token_comm_args.peer_rank_ptr_mapper,
                token_comm_args.expert_pool_base,
                token_comm_args.intra_token_count,
                token_comm_args.relay_pool_base,
                token_comm_args.internode_barrier_counter,
                phase_bit,
                stored_num_tokens_per_expert,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                local_rank=token_comm_args.local_rank,
                num_sms=token_comm_args.sm_count,
                chunk_bytes=self.hidden_bytes,
                num_token_back_warps=self.active_dispatch_warps,
                pipeline_timing=token_comm_args.pipeline_timing,
                combine_issue_counter=token_comm_args.combine_issue_counter,
            )

            if iket_active:
                _iket.range_pop()

    @cute.jit
    def relay_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
        wide_ring_smem=None,
    ):
        cta_linear_id = self._cta_linear_id()
        if cutlass.const_expr(self.pipeline_timing_enabled):
            dispatch_start = read_globaltimer()

        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)

        relay_window_base = None

        self.internode_dispatch_prep(
            token_comm_storage,
            token_comm_args.topk_idx,
            token_comm_args.internode_send_count,
            token_comm_args.internode_send_token_count,
            token_comm_args.tokens_to_node,
            token_comm_args.token_node_slot,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            local_rank=token_comm_args.local_rank,
            num_tokens=token_comm_args.input_token_buffer.shape[0],
            num_sms=token_comm_args.sm_count,
            grid_sync_counter=token_comm_args.grid_sync_counter,
        )

        if cutlass.const_expr(not self.relay_skip_dispatch_barrier):
            self.dispatch_barrier(
                token_comm_args.expert_send_count,
                token_comm_args.expert_recv_count,
                token_comm_args.expert_recv_count_sum,
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                local_rank=token_comm_args.local_rank,
                num_sms=token_comm_args.sm_count,
                nvlink_barrier_counter=token_comm_args.nvlink_barrier_counter,
            )
        # Field 60: the relay warps go straight from their own dispatch_prep
        # to the Stage 1 pack; the dispatch warps run dispatch_barrier alone.

        if cutlass.const_expr(self.pipeline_timing_enabled):
            meta_start = read_globaltimer()


        if cutlass.const_expr(self.num_relay_warps > 0):
            # Early Stage 1: the coalesced range put and its bundled
            # credit need only this rank's dispatch_prep results and
            # the aggregated send counts (Invariant I4), not the
            # metadata exchange, and the all-peer tail rail barrier
            # already separates receive-buffer generations (I8). The
            # RDMA transfer therefore overlaps inter_meta_exchange.
            if cutlass.const_expr(self.pipeline_timing_enabled):
                push_start = read_globaltimer()

            self.internode_push(
                token_comm_storage,
                token_comm_args.input_token_buffer,
                token_comm_args.input_sf_buffer,
                token_comm_args.topk_idx,
                token_comm_args.input_topk_weights_buffer,
                token_comm_args.tokens_to_node,
                token_comm_args.internode_send_count,
                token_comm_args.grid_sync_counter,
                token_comm_args.gin_window,
                token_comm_args.pipeline_timing,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                local_rank=token_comm_args.local_rank,
                num_sms=token_comm_args.sm_count,
                window_base=relay_window_base,
                internode_send_token_count=token_comm_args.internode_send_token_count,
            )

            if cutlass.const_expr(self.pipeline_timing_enabled):
                push_end = read_globaltimer()

            if cutlass.const_expr(self.pipeline_timing_enabled):
                meta_start = read_globaltimer()
            self.inter_meta_exchange(
                token_comm_args.expert_recv_count,
                token_comm_args.expert_recv_count_sum,
                token_comm_args.internode_send_count,
                token_comm_args.internode_recv_splits,
                token_comm_args.internode_send_token_count,
                token_comm_args.intra_token_count,
                token_comm_args.internode_recv_count_per_source,
                token_comm_args.expert_pool_base,
                token_comm_args.expert_sf_pool_base,
                token_comm_args.relay_pool_base,
                token_comm_args.relay_sf_pool_base,
                token_comm_args.relay_direct_bucket_meta,
                token_comm_args.internode_recv_count_per_source_snapshot,
                token_comm_args.expert_recv_count_snapshot,
                token_comm_args.expert_recv_count_sum_snapshot,
                token_comm_args.intra_token_count_snapshot,
                token_comm_args.expert_pool_base_snapshot,
                token_comm_args.expert_sf_pool_base_snapshot,
                token_comm_args.relay_pool_base_snapshot,
                token_comm_args.relay_sf_pool_base_snapshot,
                token_comm_args.expert_scheduled_count_snapshot,
                token_comm_args.pool_end_snapshot,
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                token_comm_args.gin_window,
                token_comm_args.pipeline_timing,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                local_rank=token_comm_args.local_rank,
                num_sms=token_comm_args.sm_count,
                pool_token_capacity=token_comm_args.pool_token_capacity,
                pool_sf_capacity=token_comm_args.pool_sf_capacity,
            )

            if cutlass.const_expr(self.pipeline_timing_enabled):
                meta_end = read_globaltimer()


            self.relay_fanout(
                token_comm_storage,
                token_comm_args.internode_recv_splits,
                token_comm_args.relay_fanout_reservation,
                token_comm_args.relay_fanout_slot,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                token_comm_args.gin_window,
                token_comm_args.pipeline_timing,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                local_rank=token_comm_args.local_rank,
                num_sms=token_comm_args.sm_count,
                direct_pool_args=token_comm_args,
                window_base=relay_window_base,
            )

            if cutlass.const_expr(self.pipeline_timing_enabled):
                fanout_end = read_globaltimer()

        nb_dispatch_to_sched = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb_dispatch_to_sched.arrive()
        if cutlass.const_expr(self.pipeline_timing_enabled):
            dispatch_end = read_globaltimer()
            if lane_idx == Int32(0):
                token_comm_args.pipeline_timing[
                    cta_linear_id, warp_idx, PIPELINE_STAGE_DISPATCH, 0
                ] = dispatch_start
                token_comm_args.pipeline_timing[
                    cta_linear_id, warp_idx, PIPELINE_STAGE_DISPATCH, 1
                ] = dispatch_end
                if cutlass.const_expr(self.num_relay_warps > 0):
                    # Stage 1 sub-envelope: relay warps leave the inter-node
                    # metadata exchange at push_start and finish
                    # internode_push at push_end; the Stage 2 fan-out is the
                    # remainder [push_end, dispatch_end].
                    token_comm_args.pipeline_timing[
                        cta_linear_id, warp_idx, PIPELINE_STAGE_PUSH, 0
                    ] = push_start
                    token_comm_args.pipeline_timing[
                        cta_linear_id, warp_idx, PIPELINE_STAGE_PUSH, 1
                    ] = push_end
                    # Metadata exchange [dispatch_barrier end, push start]
                    # and Stage 2 fan-out [push end, relay_fanout end].
                    token_comm_args.pipeline_timing[
                        cta_linear_id, warp_idx, PIPELINE_STAGE_META, 0
                    ] = meta_start
                    token_comm_args.pipeline_timing[
                        cta_linear_id, warp_idx, PIPELINE_STAGE_META, 1
                    ] = meta_end
                    token_comm_args.pipeline_timing[
                        cta_linear_id, warp_idx, PIPELINE_STAGE_FANOUT, 0
                    ] = push_end
                    token_comm_args.pipeline_timing[
                        cta_linear_id, warp_idx, PIPELINE_STAGE_FANOUT, 1
                    ] = fanout_end


        if cutlass.const_expr(self.enable_nccl_gin):
            self.internode_combine(
                token_comm_storage,
                token_comm_args.internode_recv_splits,
                token_comm_args.internode_send_count,
                token_comm_args.relay_fanout_slot,
                token_comm_args.expert_recv_count_sum,
                token_comm_args.intra_token_count,
                token_comm_args.relay_pool_base,
                token_comm_args.fc2_output_workspace,
                token_comm_args.fc2_done_counter,
                token_comm_args.internode_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                token_comm_args.gin_window,
                token_comm_args.pipeline_timing,
                token_comm_args.combine_timing_landmarks,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                local_rank=token_comm_args.local_rank,
                num_sms=token_comm_args.sm_count,
                wide_ring_smem=wide_ring_smem,
                combine_issue_counter=token_comm_args.combine_issue_counter,
            )

    @cute.jit
    def token_back_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        cta_linear_id = self._cta_linear_id()
        local_warp_idx = Int32(warp_idx) - Int32(self.token_back_warp_start)

        # Handshake: dispatch_barrier done => expert_recv_count_sum populated.
        nb_dispatch_to_sched = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb_dispatch_to_sched.arrive_and_wait()

        tb_pull_mbar_ptr = token_comm_storage.tb_pull_mbar.data_ptr()
        tb_pull_buffer_ptr = token_comm_storage.tb_pull_buffer.data_ptr()
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(tb_pull_mbar_ptr + local_warp_idx, 1)
        cute.arch.sync_warp()

        NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        stored_num_tokens_per_expert = []
        for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            stored_num_tokens_per_expert.append(Int32(0))
        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            e_idx_for_lane = Int32(i * self.warp_threads) + lane_idx
            if e_idx_for_lane < Int32(self.num_experts_per_rank):
                sum_packed_init = token_comm_args.expert_recv_count_sum[e_idx_for_lane]
                stored_num_tokens_per_expert[i] = Int32(
                    Int64(sum_packed_init) & Int64(0xFFFFFFFF)
                )
        cute.arch.sync_warp()

        iket_active = cta_linear_id == Int32(0)
        if iket_active:
            _iket.range_push("Token_Back_By_Push")

        self.token_back_by_push(
            tb_pull_buffer_ptr,
            tb_pull_mbar_ptr,
            token_comm_args.fc2_output_workspace,
            token_comm_args.fc2_done_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.combine_output,
            token_comm_args.combine_sf,
            token_comm_args.fc2_output_sf,
            token_comm_args.token_back_schedule_counter,
            token_comm_args.peer_rank_ptr_mapper,
            token_comm_args.expert_pool_base,
            token_comm_args.intra_token_count,
            token_comm_args.relay_pool_base,
            token_comm_args.internode_barrier_counter,
            Int32(0),
            stored_num_tokens_per_expert,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            local_rank=token_comm_args.local_rank,
            num_sms=token_comm_args.sm_count,
            chunk_bytes=self.tb_chunk_bytes,
            num_token_back_warps=self.num_token_back_warps,
            pipeline_timing=token_comm_args.pipeline_timing,
            combine_issue_counter=token_comm_args.combine_issue_counter,
        )

        if iket_active:
            _iket.range_pop()

    def use_single_cta_tail_reset(self, shared_zero_i32_count: int) -> bool:
        """Bound CTA0's shared clear; retain distributed reset under NCU."""
        return (
            not self.enable_nccl_gin and
            self.world_size > 1
            and shared_zero_i32_count * 4 <= 16384
            and os.environ.get("MEGA_USE_NCU", "0") != "1"
        )

    @cute.jit
    def tail_reset_counters(
        self,
        token_comm_args,
        target_zero_tensor,
        *,
        cta_linear_id,
        local_warp_idx,
        lane_idx,
        single_cta: cutlass.Constexpr[bool] = False,
        preserve_grid_phase: cutlass.Constexpr[bool] = False,
        readiness_snapshot=None,
    ):
        """Per-lane 4B (Int32) bulk-zero of one accumulating-counter prefix.

        ``target_zero_tensor`` is an Int32 view over a workspace's front counter
        region (megamoe_kernel front-places every counter that must restart at 0
        each launch; data buffers and the phase-flip ``nvlink_barrier_signal`` sit
        after the prefix and are untouched). The shared clear may be owned by
        CTA0; other clears are spread across dispatch threads grid-wide.
        kernel_tail calls this twice:
          * SHARED prefix (expert_recv_count / _sum) BETWEEN the two nvlink
            barriers, so the final barrier publishes the zeros cross-rank -- needed
            when the next launch is another MegaMoE reusing the shared workspace
            with no intervening rank sync;
          * LOCAL prefix (l1_arrival / expert_send / grid_sync /
            fc1_done [+ fc2_done / token_back_schedule / load_balance]) AFTER the
            last barrier -- rank-local (next kernel sees it via stream order) and
            grid_sync stays live until that last barrier. The NVLink phase
            counter is outside both prefixes and persists across launches.
        Only the FIRST launch relies on a caller-zeroed workspace.
        """
        if cutlass.const_expr(single_cta):
            thread_linear = local_warp_idx * Int32(self.warp_threads) + lane_idx
            stride = Int32(self.num_dispatch_threads)
        else:
            thread_linear = (
                cta_linear_id * Int32(self.num_dispatch_warps) + local_warp_idx
            ) * Int32(self.warp_threads) + lane_idx
            stride = Int32(token_comm_args.sm_count * self.num_dispatch_threads)

        count = cute.size(target_zero_tensor)
        i = thread_linear
        while i < Int32(count):
            if cutlass.const_expr(readiness_snapshot is not None):
                # Direct mode places readiness first in the shared prefix
                # (asserted by the workspace builder). The same thread reads
                # and clears a word, so snapshotting needs no new grid barrier.
                if i < Int32(cute.size(readiness_snapshot)):
                    readiness_snapshot[i] = cute.arch.load(
                        target_zero_tensor.iterator + i, Int32,
                        sem="acquire", scope="sys",
                    )
            if cutlass.const_expr(preserve_grid_phase):
                # Omitting one join changes the final phase. Clearing it here
                # could hide the completion bit from another CTA's last poll.
                address = (target_zero_tensor.iterator + i).toint()
                grid_begin = token_comm_args.grid_sync_counter.iterator.toint()
                grid_end = grid_begin + Int64(
                    cute.size(token_comm_args.grid_sync_counter) * 4
                )
                if (address < grid_begin) | (address >= grid_end):
                    target_zero_tensor[i] = Int32(0)
            else:
                target_zero_tensor[i] = Int32(0)
            i = i + stride

    @cute.jit
    def epilogue_exit(
        self,
        token_comm_args,
        token_comm_storage=None,
        *,
        warp_idx,
        lane_idx,
    ):
        """Epilogue warps call this right after their tile loop returns.

        Epilogue warp 0 stamps the ``fc_epi_exit`` stage.
        """
        if cutlass.const_expr(self.pipeline_timing_enabled):
            epi_exit_t0 = read_globaltimer()
        if cutlass.const_expr(self.pipeline_timing_enabled):
            if warp_idx == Int32(0):
                if lane_idx == Int32(0):
                    epi_exit_cta = self._cta_linear_id()
                    token_comm_args.pipeline_timing[
                        epi_exit_cta, warp_idx, PIPELINE_STAGE_FC_EPI_EXIT, 0
                    ] = epi_exit_t0
                    token_comm_args.pipeline_timing[
                        epi_exit_cta, warp_idx, PIPELINE_STAGE_FC_EPI_EXIT, 1
                    ] = read_globaltimer()

    @cute.jit
    def kernel_tail(
        self,
        token_comm_args,
        token_comm_storage=None,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        if cutlass.const_expr(self.pipeline_timing_enabled):
            tail_t0 = read_globaltimer()
        if cutlass.const_expr(self.enable_nccl_gin and token_comm_storage is not None):
            # Epilogue warp 0 reaches the tail only after it consumed its last
            # accumulator, so every UMMA read and TMA write of the AB stage
            # buffers is complete. Order those async-proxy accesses before the
            # relay warps' generic cp.async writes and publish the ring. The
            # flag address is resolved before the dynamic branches so the
            # traced bodies capture an Int32, not the shared-memory struct.
            fc_smem_free_addr = Int32(
                token_comm_storage.fc_smem_free.data_ptr().toint()
            )
            if warp_idx == Int32(0):
                if lane_idx == Int32(0):
                    fence_proxy_async_shared_cta_raw()
                    st_shared_release_cta_b32_raw(fc_smem_free_addr, Int32(1))
                    if cutlass.const_expr(self.pipeline_timing_enabled):
                        # Ring release seen from kernel_tail: entry -> store.
                        ring_release_cta = self._cta_linear_id()
                        token_comm_args.pipeline_timing[
                            ring_release_cta, warp_idx, PIPELINE_STAGE_FC_RING_RELEASE, 0
                        ] = tail_t0
                        token_comm_args.pipeline_timing[
                            ring_release_cta, warp_idx, PIPELINE_STAGE_FC_RING_RELEASE, 1
                        ] = read_globaltimer()
        nb_kernel_tail = pipeline.NamedBarrier(
            barrier_id=self.kernel_tail_named_barrier_id,
            num_threads=self.kernel_tail_threads,
        )
        nb_kernel_tail.arrive_and_wait()

        # Only the dispatch warps run NVLink cleanup; standalone token-back
        # warps (>= token_back_warp_start) just join the rendezvous above.
        if (warp_idx >= self.dispatch_warp_start) and (
            warp_idx < self.dispatch_warp_start + self.num_dispatch_warps
        ):
            cta_linear_id = self._cta_linear_id()
            local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
            single_cta_reset = cutlass.const_expr(
                self.use_single_cta_tail_reset(
                    cute.size(token_comm_args.shared_zero_prefix)
                )
            )
            # Per-launch nvlink barrier count must be a multiple of 4 so the
            # sense-reversing signal self-cancels back to its start state. The
            # launch already does 1 (dispatch_barrier) + 2 below (drain + publish,
            # around the shared reset) = 3. Under ncu kernel replay the signal is
            # cross-device and can't be restored across passes, so pad with one
            # extra drain to reach 4. Non-ncu rides the phase counter across launch
            # boundaries and needs only 3.
            #
            # MEGA_USE_NCU is a dev-only profiling hack -- read inline here on
            # purpose: it is never exposed to customers and the ncu path is not
            # kept in production, so it must NOT become a constructor field or a
            # kernel-name constexpr.
            if cutlass.const_expr(not self.enable_nccl_gin and os.environ.get("MEGA_USE_NCU", "0") == "1"):
                self.nvlink_barrier(
                    token_comm_args.nvlink_barrier_signal,
                    token_comm_args.nvlink_barrier_counter,
                    token_comm_args.grid_sync_counter,
                    token_comm_args.peer_rank_ptr_mapper,
                    cta_linear_id,
                    local_warp_idx,
                    lane_idx,
                    num_sms=token_comm_args.sm_count,
                    prologue_grid_sync=True,
                    epilogue_grid_sync=True,
                )
            # Snapshot the double-buffer phase before kernel_tail advances the
            # rail generation.  TopkReduce runs later on the same stream and
            # reads this device-side snapshot without a host synchronization.
            if cutlass.const_expr(self.enable_nccl_gin):
                if (
                    cta_linear_id == Int32(0)
                    and local_warp_idx == Int32(0)
                    and lane_idx == Int32(0)
                ):
                    token_comm_args.combine_phase_snapshot[0] = (
                        token_comm_args.internode_barrier_counter[0]
                        & Int32(1)
                    )

            # use_ncu is a dev-only profiling hack.  The constructor
            # statically selects the real 3/4/5-call path count and computes
            # exactly enough padding to close a four-call signal cycle.
            if cutlass.const_expr(
                self.use_ncu
            ):
                for _ in cutlass.range_constexpr(
                    0, self.nvlink_ncu_padding, 1
                ):
                    self.nvlink_barrier(
                        token_comm_args.nvlink_barrier_signal,
                        token_comm_args.nvlink_barrier_counter,
                        token_comm_args.grid_sync_counter,
                        token_comm_args.peer_rank_ptr_mapper,
                        cta_linear_id,
                        local_warp_idx,
                        lane_idx,
                        num_sms=token_comm_args.sm_count,
                        local_rank=token_comm_args.local_rank,
                        sync_num_threads=self.num_dispatch_threads,
                        prologue_grid_sync=True,
                        epilogue_grid_sync=True,
                    )
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
                local_rank=token_comm_args.local_rank,
                sync_num_threads=self.num_dispatch_threads,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
            if cutlass.const_expr(self.tail_reducer is not None):
                # internode_combine drained every expected GIN credit before
                # the all-thread rendezvous. The preceding NVLink barrier and
                # its closing grid sync also publish every local top-k cell.
                # Keep the current phase until these ordered BF16 workers end;
                # the next barrier still precedes all rail-phase advancement.
                tail_worker = (
                    (cta_linear_id * Int32(self.num_dispatch_warps) + local_warp_idx)
                    * Int32(self.warp_threads) + lane_idx
                )
                self.tail_reducer.reduce_internode_tail(
                    token_comm_args.tail_combine_output,
                    token_comm_args.tail_internode_combine,
                    token_comm_args.combine_phase_snapshot,
                    token_comm_args.topk_idx,
                    (token_comm_args.input_topk_weights_buffer
                     if not self.apply_topk_in_fc1 else None),
                    token_comm_args.tail_reduced_output,
                    token_comm_args.local_rank,
                    token_comm_args.token_node_slot,
                    tail_worker,
                    Int32(token_comm_args.sm_count * self.num_dispatch_threads),
                )
            # Shared counters between the barriers: the slot=0 barrier below
            # publishes these zeros cross-rank for a back-to-back MegaMoE relaunch.
            if cutlass.const_expr(single_cta_reset):
                if cta_linear_id == Int32(0):
                    self.tail_reset_counters(
                        token_comm_args,
                        token_comm_args.shared_zero_prefix,
                        cta_linear_id=cta_linear_id,
                        local_warp_idx=local_warp_idx,
                        lane_idx=lane_idx,
                        single_cta=True,
                    )
                    # Publish every clearing warp before warp 0 notifies peers.
                    # The drain barrier above already joined the entire grid.
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.barrier(
                        barrier_id=self.dispatch_intra_cta_bar_id,
                        number_of_threads=self.num_dispatch_threads,
                    )
            else:
                self.tail_reset_counters(
                    token_comm_args,
                    token_comm_args.shared_zero_prefix,
                    cta_linear_id=cta_linear_id,
                    local_warp_idx=local_warp_idx,
                    lane_idx=lane_idx,
                    readiness_snapshot=(token_comm_args.l1_arrival_count_snapshot
                                        if self.enable_nccl_gin else None),
                )
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
                local_rank=token_comm_args.local_rank,
                sync_num_threads=self.num_dispatch_threads,
                prologue_grid_sync=not single_cta_reset,
                epilogue_grid_sync=True,
            )
            if cutlass.const_expr(self.enable_nccl_gin):
                # The launch's only rail barrier follows the data paths'
                # flush + strong-credit drains and keeps every rail peer in the
                # same generation before registered buffers are reused.
                self.internode_rail_barrier(
                    token_comm_args.internode_barrier_counter,
                    token_comm_args.grid_sync_counter,
                    cta_linear_id,
                    local_warp_idx,
                    lane_idx,
                    num_sms=token_comm_args.sm_count,
                    local_rank=token_comm_args.local_rank,
                    arrived=True,
                )

                if cutlass.const_expr(
                    self.use_ncu
                ):
                    # Monotonic arrival signals live outside NCU's kernel
                    # snapshot. Reset them only inside a rail-barrier sandwich,
                    # then pad the two-slot barrier to its four-call cycle.
                    self.reset_internode_split_signals_for_ncu(
                        cta_linear_id,
                        local_warp_idx,
                        lane_idx,
                    )
                    for _ in cutlass.range_constexpr(0, 3, 1):
                        self.internode_rail_barrier(
                            token_comm_args.internode_barrier_counter,
                            token_comm_args.grid_sync_counter,
                            cta_linear_id,
                            local_warp_idx,
                            lane_idx,
                            num_sms=token_comm_args.sm_count,
                            local_rank=token_comm_args.local_rank,
                        )

                # GIN reads this source through its registered window. It is not
                # part of local_zero_prefix, so clear it explicitly after the
                # rail drain proves no peer can still consume this generation.
                self.tail_reset_counters(
                    token_comm_args,
                    token_comm_args.internode_send_count,
                    cta_linear_id=cta_linear_id,
                    local_warp_idx=local_warp_idx,
                    lane_idx=lane_idx,
                )

            # Local counters last: rank-local, and both grid-sync/barrier phase
            # counters above stay live until every node-local and rail barrier
            # has completed.
            self.tail_reset_counters(
                token_comm_args,
                token_comm_args.local_zero_prefix,
                cta_linear_id=cta_linear_id,
                local_warp_idx=local_warp_idx,
                lane_idx=lane_idx,
                preserve_grid_phase=single_cta_reset,
            )
            if cutlass.const_expr(self.pipeline_timing_enabled):
                # kernel_tail on the dispatch warps: rendezvous, phase
                # snapshot, rail and NVLink barriers, counter resets.
                tail_t1 = read_globaltimer()
                if lane_idx == Int32(0):
                    token_comm_args.pipeline_timing[
                        cta_linear_id, warp_idx, PIPELINE_STAGE_TAIL, 0
                    ] = tail_t0
                    token_comm_args.pipeline_timing[
                        cta_linear_id, warp_idx, PIPELINE_STAGE_TAIL, 1
                    ] = tail_t1
