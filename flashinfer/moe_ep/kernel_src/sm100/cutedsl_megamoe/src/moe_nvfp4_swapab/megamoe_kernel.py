# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MegaMoE fused dispatch + fc1 + fc2 + combine kernel.

The base class owns the local fc1/fc2 GEMM pipeline.  This subclass owns the
token-communication hooks, workspace partitioning, and the MegaMoE argument
bundle.  ``static_expert_shape`` is required because dispatch storage and pool
sizes are codegen-time quantities.

Shared / local workspace split:

  SHARED  : src_token_topk_idx, expert_recv_count, expert_recv_count_sum,
            nvlink_barrier_signal
  LOCAL   : expert_send_count, grid_sync_counter, l1_token_buffer,
            l1_sf_buffer, l1_topk_weights_buffer, l1_arrival_count,
            (optionally) l1_fc1_activation_per_token_scale_buffer,
            token_src_metadata, fc1_output, fc1_output_sf,
            fc1_done_counter, (optionally) load_balance_counter

User tensors are not in the opaque workspaces. ``activation``,
``activation_sf``, ``topk_weights``, ``combine_output``, and (when enabled)
``fc1_activation_per_token_scale`` must be reachable through the
symmetric-heap peer mapper; ``topk_idx`` and weights are local.

Dispatch/pool alignment constraints are unified at construction time:
``token_padding_block`` (base) and ``block_m`` (dispatch) become the
same constant, similarly for ``sf_padding_block`` / ``sf_block_m``;
C3 reduces to a divisibility check that ``cluster_tile_tokens`` is a
multiple of ``token_padding_block``.
"""

# NOTE: ``from __future__ import annotations`` is intentionally NOT used here.
# PEP 563 string-ifies class-body annotations, which breaks ``@cute.struct``'s
# element-type introspection (it reads ``__annotations__`` and demands the
# values be live ``cute.struct.MemRange[...] / struct / array / base_dsl
# scalar`` objects, not their string forms).  The lean fc1+fc2 base
# (``kernel_fc12.py``) and the dispatch standalone (``pre_unification/comm/dispatch_kernel.py``)
# both already follow this convention.  Self-references (the single
# ``"TokenCommArgs"`` forward ref on ``__new_from_mlir_values__``) stay
# quoted explicitly.

import dataclasses
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

from common.host_utils import get_cutedsl_target_arch
from common.direct_meta import DIRECT_META_FIELD_MASK
from common.pipeline_timing import (
    COMBINE_TIMING_LANDMARK_SHAPE,
    FC_TILE_TIMELINE_SHAPE,
    PIPELINE_TIMING_MAX_CTAS,
    PIPELINE_TIMING_SHAPE,
)
from src.megamoe_config import MegaMoEConfig
from .kernel_fc12 import Sm100SwapABSwigluFp4Fc12Kernel
from .topk_reduce import TopkReduce
from src.token_comm import (
    CombineFormat,
    TokenCommArgs as ExtractedTokenCommArgs,
    TokenInPullTokenBackPush,
    TokenSrcMetadata,
)


# =============================================================================
# Module-level constants.
# =============================================================================

# NamedBarrier IDs.  Base reserves 1-7; this subclass uses 8 and 9.
_KernelTailNamedBarrierId = 8        # 12-warp rendezvous (384 threads)
_DispatchToSchedNamedBarrierId = 9   # 4 dispatch + 1 sched (160 threads)

# Dispatch warp count.
_DispatchWarpCount = 4

# Stable symbol prefix for the AOT object handoff (dump_to_object / load_module).
# One function per .o, each in its own engine, so a fixed prefix never collides.
_aot_symbol_prefix = "megamoe_aot"

# Per-pool-slot provenance record consumed by combine STG redirect (S3) and
# token-back; one i64 = {src_rank, src_token, src_topk} (see TokenSrcMetadata).
_TokenMetadataBytes = TokenSrcMetadata.nbytes

# NVLink signal slots used by the DeepGEMM-style phase/sign barrier.
# A separate local counter selects phase/sign; the signal slots are not reset
# by tail cleanup.
_NvlinkSlotCount = 2

# Grid-sync counter slot count.  ``software_grid_sync`` phase-flips bit 31
# so a single slot suffices; 2 slots keeps the layout 8-byte aligned.
_GridSyncSlotCount = 2


# =============================================================================
# Region spec + layout helpers
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _RegionSpec:
    """One region in either the local or shared workspace.

    Byte size = ``ceil(numel * cute_dtype.width / 8)``.  ``align`` is
    the region's start-byte alignment (TMA store / load destinations
    want 128 B; counters / metadata want 16 B).
    """

    name: str
    cute_dtype: Any
    shape: Tuple[int, ...]
    align: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def stride_row_major(self) -> Tuple[int, ...]:
        """Row-major stride matching ``shape`` (rightmost dim contiguous)."""
        if len(self.shape) == 0:
            return ()
        out: List[int] = [1]
        for d in reversed(self.shape[1:]):
            out.append(out[-1] * d)
        out.reverse()
        return tuple(out)

    @property
    def nbytes(self) -> int:
        bits = self.numel * int(self.cute_dtype.width)
        return (bits + 7) // 8


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _layout_regions(
    regions: List[_RegionSpec],
) -> Tuple[Dict[str, int], int]:
    """Place ``regions`` sequentially honouring each region's ``align``.
    Returns ``(name -> byte_offset)`` and the total byte count (rounded
    up to 16 B for downstream safety).

    Drives both ``get_workspace_sizes()`` (total only) and the
    ``__call__`` partition (offsets) -- keeping the host allocation
    and the device view construction in sync without any explicit
    handshake.
    """
    offsets: Dict[str, int] = {}
    cursor = 0
    for r in regions:
        cursor = _round_up(cursor, r.align)
        offsets[r.name] = cursor
        cursor += r.nbytes
    total = _round_up(cursor, 16)
    return offsets, total


# =============================================================================
# Sm100MegaMoEKernel
# =============================================================================


class Sm100MegaMoEKernel(Sm100SwapABSwigluFp4Fc12Kernel):
    """MegaMoE-complete fused dispatch + fc1 + fc2 + combine kernel."""

    def __init__(
        self,
        # Base-class kwargs (forwarded 1:1 to ``super().__init__``).
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: str = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_vec_size: int = 16,
        scenario: str = "2Dx3D",
        # MegaMoE-specific independent constants.
        *,
        world_size: int,
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        fc2_output_dtype: Type[cutlass.Numeric],
        ranks_per_node: int = 0,
        # Master switch of every internode (multi-node) code path. False (the default) is the
        # NVLink-domain megakernel of main: no relay warps, no GIN, one logical node, the
        # historical MMA loop and FC2 publish batch; the internode knobs below must stay at
        # their defaults. True opts in and requires relay and NCCL GIN together.
        internode: bool = False,
        combine_format: CombineFormat = CombineFormat.parse("bf16"),
        non_ubulk_fc2_store: bool = True,
        in_kernel_fc2_reduce: bool = False,
        skip_topk_reduce: bool = False,
        token_back_mode: Literal[
            "epi_warps", "standalone_warps", "reuse_dispatch_warps"
        ] = "epi_warps",
        enable_internode_relay: bool = False,
        enable_nccl_gin: bool = False,
        publish_relay_counts_to_gemm: bool = False,
        apply_topk_in_fc1: bool = True,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
        gate_up_clamp: Optional[float] = None,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        enable_fc1_activation_per_token_scale: bool = False,
        epi_flag_batch: Optional[Tuple[int, int]] = None,
        flag_batch: int = 1,
        topk_reduce_persistent: bool = False,
        config: Optional[MegaMoEConfig] = None,
    ) -> None:
        # Codegen options and diagnostics (pre_unification/comm/megamoe_config.py); the only
        # source of these values, no environment is read here.
        if config is None:
            config = MegaMoEConfig()
        self.config = config
        # Master switch (see the parameter note): with internode=False every multi-node path is
        # compiled out and the remaining constructor arguments must describe one NVLink domain.
        if internode and (enable_fc1_activation_per_token_scale or topk_reduce_persistent or skip_topk_reduce):
            raise ValueError("GIN v1 requires separate static reduce and no per-token scale.")
        self.internode = bool(internode)
        if not self.internode:
            if enable_internode_relay or enable_nccl_gin or publish_relay_counts_to_gemm:
                raise ValueError(
                    "internode=False compiles every internode code path out; "
                    "enable_internode_relay, enable_nccl_gin and publish_relay_counts_to_gemm "
                    "must stay False (pass internode=True to opt in)"
                )
            if ranks_per_node not in (0, world_size):
                raise ValueError(
                    "internode=False means one logical node: ranks_per_node must be 0 or "
                    f"world_size ({world_size}), got {ranks_per_node}"
                )
            ranks_per_node = world_size
        elif not enable_internode_relay:
            raise ValueError("internode=True requires enable_internode_relay=True")
        elif not enable_nccl_gin:
            raise ValueError("internode=True requires enable_nccl_gin=True")
        if epi_flag_batch is None:
            epi_flag_batch = (1, 4) if self.internode else (1, 1)
        # The combine wire format drives the fc2 epilogue encoder, token_comm
        # push, and the combine_quant/combine_sf workspace sizing. The dataflow
        # (workspace carve, views, arg threading, topk_reduce receiver) is wired
        # for every format; only the epilogue *encode* is still bf16-only, so the
        # quantized guard sits at the launch (see __call__), letting the workspace
        # anchor size + partition for quantized formats here.
        self.combine_format = combine_format
        if in_kernel_fc2_reduce and combine_format.is_quantized:
            raise ValueError(
                f"in_kernel_fc2_reduce requires a non-quantized (bf16) combine "
                f"format; got {combine_format}."
            )
        if in_kernel_fc2_reduce and not apply_topk_in_fc1:
            raise ValueError(
                "in_kernel_fc2_reduce requires apply_topk_in_fc1=True; "
                "the REDG path can only atomic-add terms whose topk score "
                "was already absorbed before fc2."
            )
        if skip_topk_reduce and (
            in_kernel_fc2_reduce
            or combine_format.is_quantized
        ):
            raise ValueError(
                "skip_topk_reduce requires in_kernel_fc2_reduce=False "
                "and a bf16 combine format."
            )
        if (
            combine_format.act_dtype is cutlass.Float4E2M1FN
            and not non_ubulk_fc2_store
        ):
            raise ValueError(
                f"{combine_format} combine requires non_ubulk_fc2_store=True "
                "(the UBLK fc2 store path cannot scalar-dereference FP4)."
            )
        if static_expert_shape is None:
            raise NotImplementedError(
                "Sm100MegaMoEKernel currently requires "
                "static_expert_shape != None (dynamic-shape MegaMoE is "
                "not wired)."
            )
        # Keep the explicit ``hidden`` kwarg in lockstep with static shape;
        # dispatch SMEM sizing reads it before tensor layouts are rewritten.
        if hidden != static_expert_shape[2]:
            raise ValueError(
                f"hidden ({hidden}) must equal "
                f"static_expert_shape[2] ({static_expert_shape[2]})."
            )

        # token_back_mode selects where the cross-rank fc2 push-back runs:
        #   epi_warps            -> epilogue warps STG directly to the peer
        #   standalone_warps     -> dedicated warp group 12-15, concurrent
        #                           with dispatch_pull
        #   reuse_dispatch_warps -> dispatch warps 8-11 push after dispatch_pull
        # The two non-epi modes both stage fc2 to a local workspace first, i.e.
        # token_back_by_dispatch=True; epi_warps keeps the epilogue STG redirect.
        if token_back_mode not in (
            "epi_warps", "standalone_warps", "reuse_dispatch_warps"
        ):
            raise ValueError(
                f"token_back_mode must be 'epi_warps', 'standalone_warps', "
                f"or 'reuse_dispatch_warps'; got {token_back_mode!r}."
            )
        if enable_internode_relay and token_back_mode == "standalone_warps":
            raise ValueError(
                "enable_internode_relay reserves warps 12-15 for relay work; "
                "use token_back_mode='reuse_dispatch_warps' or 'epi_warps'."
            )
        token_back_by_dispatch = token_back_mode != "epi_warps"
        # A GIN remote origin cannot be targeted by the epilogue's LSA peer
        # pointer.  Stage all FC2 rows locally even when the user-facing mode is
        # ``epi_warps``; dispatch warps copy local-node terms while relay warps
        # gather and RDMA-return remote-node partials.
        self.epilogue_token_back_by_dispatch = (
            token_back_by_dispatch or enable_nccl_gin
        )

        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            use_2cta_instrs=use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            sf_padding_block=sf_padding_block,
            load_balance_mode=load_balance_mode,
            # Two-segment (intra / relay) scheduling exists only with the GIN
            # relay layout: segment-major order with the relay FC1 tiles right
            # after the intranode FC1 tiles (defaults since 2026-09-05 and
            # 2026-09-09; the only order since 2026-09-21). The intranode
            # megakernel has one segment and keeps the expert-major order.
            sched_segment_major=enable_nccl_gin,
            sched_segment_fc1_first=enable_nccl_gin,
            static_expert_shape=static_expert_shape,
            force_static_sched=force_static_sched,
            clc_bundle_size=clc_bundle_size,
            num_sched_stages=num_sched_stages,
            acc_dtype=acc_dtype,
            sf_vec_size=sf_vec_size,
            scenario=scenario,
            fc2_output_dtype=fc2_output_dtype,
            non_ubulk_fc2_store=non_ubulk_fc2_store,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            token_back_by_dispatch=token_back_by_dispatch,
            apply_topk_in_fc1=apply_topk_in_fc1,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            gate_up_clamp=gate_up_clamp,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            enable_fc1_activation_per_token_scale=(
                enable_fc1_activation_per_token_scale
            ),
            epi_flag_batch=epi_flag_batch,
            # The GEMM-stage attribution (2026-09-08) made the software-pipelined
            # MMA k-loop with one SF TMEM slot per AB stage the internode MegaMoE
            # implementation; the NVLink-domain kernel (internode=False) and the
            # lean base keep the historical loop (main's codegen).
            mma_pipelined_kloop=self.internode,
            sf_tmem_stages="ab" if self.internode else "1",
            task_reg_cnt=config.task_reg_cnt,
            tma_cache_hint=config.tma_cache_hint,
            pipeline_timing=config.diagnostics.pipeline_timing,
            fc_tile_timeline=config.diagnostics.fc_tile_timeline,
        )

        self.enable_token_comm = True
        self.dispatch_warp_id = (8, 9, 10, 11)
        # Standalone token-back: a dedicated 4-warp group (12-15) doing
        # token_back_by_push concurrently with dispatch_pull, selected by the
        # user-facing token_back_mode knob ("standalone_warps").
        self.token_back_mode = token_back_mode
        self.token_back_standalone = token_back_mode == "standalone_warps"
        self.enable_internode_relay = enable_internode_relay
        self.enable_nccl_gin = enable_nccl_gin
        if self.enable_nccl_gin and load_balance_mode != "static":
            # The internode kernel schedules statically; the atomic-counter
            # claim stays an intranode option (launcher field 15 removed
            # 2026-09-21, production ABBA 2925039 neutral).
            raise ValueError(
                "enable_nccl_gin requires load_balance_mode='static'; got "
                f"{load_balance_mode!r}"
            )
        self.fc2_stage_locally = (
            self.token_back_by_dispatch or self.enable_nccl_gin
        )
        self.publish_relay_counts_to_gemm = publish_relay_counts_to_gemm
        self.merge_token_segments = (
            enable_nccl_gin and publish_relay_counts_to_gemm
        )
        if config.tail_topk_reduce and not enable_nccl_gin:
            raise ValueError("tail_topk_reduce requires enable_nccl_gin")
        if config.tail_topk_reduce and skip_topk_reduce:
            raise ValueError(
                "tail_topk_reduce (in-kernel tail reduce) and skip_topk_reduce "
                "(caller reduces the staging) are mutually exclusive"
            )
        self.tail_topk_reduce = bool(config.tail_topk_reduce)
        if self.enable_nccl_gin and not self.enable_internode_relay:
            raise ValueError(
                "enable_nccl_gin requires enable_internode_relay=True"
            )
        if self.enable_nccl_gin and self.in_kernel_fc2_reduce:
            raise NotImplementedError(
                "the NCCL GIN combine requires ordered FP32 TopkReduce, not REDG"
            )
        if self.enable_nccl_gin and self.combine_format.name != "bf16":
            raise NotImplementedError(
                "the NCCL GIN combine currently supports BF16 only"
            )
        self.token_back_warp_id = (12, 13, 14, 15) if self.token_back_standalone else None
        self.relay_warp_id = (12, 13, 14, 15) if self.enable_internode_relay else None
        num_token_back_warps = (
            len(self.token_back_warp_id) if self.token_back_standalone else 0
        )
        num_relay_warps = len(self.relay_warp_id) if self.relay_warp_id else 0
        self.threads_per_cta = 32 * (
            len(self.epilogue_warp_id)
            + 1  # mma
            + 1  # tma_a
            + 1  # tma_b
            + 1  # sched
            + len(self.dispatch_warp_id)
            + num_relay_warps
            + num_token_back_warps
        )

        # Independent MegaMoE-specific constants.
        self.world_size = world_size
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
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden
        self.flag_batch = flag_batch        # stored so name() can encode it
        self.skip_topk_reduce = skip_topk_reduce

        # static_expert_shape = (num_experts_per_rank, intermediate_gateup, hidden).
        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_gateup = static_expert_shape[1]
        self.intermediate_downproj = self.intermediate_gateup // 2

        # NVFP4: 4 bits/elem -> 2 elements per byte.
        self.hidden_bytes = self.hidden // 2
        # Dispatch pulls SF in uint32 units; host activation_sf rows must pad
        # to this ceiling with zero-filled bytes.
        sf_atom_k_elements = 4 * self.sf_vec_size
        self.sf_uint32_per_token = (
            (self.hidden + sf_atom_k_elements - 1) // sf_atom_k_elements
        )
        # Cross-rank totals: per-rank count * world_size.
        self.num_total_experts = world_size * self.num_experts_per_rank
        # Node-local expert vector exchanged over each rail: R ranks, E experts each.
        self.node_experts = self.ranks_per_node * self.num_experts_per_rank

        # Per-(token, topk) SF block padded to a 16 B multiple so the token-back
        # cp.async.bulk push moves one aligned block per shot; only the first
        # hidden//scale_block entries of each block are valid (the rest is the
        # alignment gap).  ``None`` for the bf16 baseline (no SF plane).
        self.sf_block_pad = (
            _round_up(
                self.hidden // self.combine_format.scale_block,
                16 // (int(self.combine_format.scale_dtype.width) // 8),
            )
            if self.combine_format.is_quantized
            else None
        )

        # Per-task-tile release-counter granularity used by dispatch_pull.
        self.cluster_tile_tokens = mma_tiler_mnk[1] * cluster_shape_mnk[1]

        # One dispatch task tile must map to contiguous pool blocks.
        if self.cluster_tile_tokens % self.token_padding_block != 0:
            raise ValueError(
                f"C3 violated: cluster_tile_tokens "
                f"({self.cluster_tile_tokens}) must be a multiple of "
                f"token_padding_block ({self.token_padding_block}); "
                f"otherwise pool row offsets and release counter slots "
                f"will not align."
            )

        # Cache region sizing inputs used by workspace layout and __call__.
        (
            self.pool_token_capacity,
            self.pool_sf_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()
        # Cohabit warps in this CTA outside the dispatch group:
        # epilogue + mma + tma_a + tma_b + sched.
        num_other_warps = (
            len(self.epilogue_warp_id) + 1 + 1 + 1 + 1
        )
        # fc2 epi publishes once per CTA per work tile; edge hidden tiles
        # still publish (no in-bound gating), so ceil_div on the hidden axis.
        cluster_fc2_tile_hidden = (
            self.mma_tiler[0] * self.cluster_shape_mn[0]
            // (2 if self.use_2cta_instrs else 1)
        )
        fc2_publishes_per_token_cluster_tile = (
            (self.hidden + cluster_fc2_tile_hidden - 1)
            // cluster_fc2_tile_hidden
        ) * self.cluster_shape_mn[0]

        # Token-back warps run when they push the DATA plane (dispatch modes) OR
        # the SF plane (any quantized combine, including the epi_warps data path
        # where the epilogue STGs the data straight to the peer but SF still
        # needs the staged token-contiguous push).
        self.token_back_enabled = (
            self.fc2_stage_locally or self.combine_format.is_quantized
        )
        # Homomorphic to the fc1+fc2 scheduler: atomic_counter token-back only
        # nets a win with enough tokens, the same condition that selects the
        # atomic_counter fc1+fc2 scheduler.  Static when token-back is off.
        self.token_back_schedule_mode = (
            self.load_balance_mode if self.token_back_enabled else "static"
        )

        self.token_comm = TokenInPullTokenBackPush(
            world_size=self.world_size,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=cutlass.Float4E2M1FN,
            combine_format=self.combine_format,
            token_back_by_dispatch=self.fc2_stage_locally,
            fc2_publishes_per_token_cluster_tile=fc2_publishes_per_token_cluster_tile,
            token_back_reduce_topk=(
                self.token_back_by_dispatch and self.in_kernel_fc2_reduce
            ),
            token_back_standalone=self.token_back_standalone,
            num_relay_warps=num_relay_warps,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=self.dispatch_warp_id[0],
            num_other_warps=num_other_warps,
            ranks_per_node=self.ranks_per_node,
            flag_batch=flag_batch,
            is_swap_ab=True,
            token_back_schedule_mode=self.token_back_schedule_mode,
            enable_nccl_gin=self.enable_nccl_gin,
            publish_relay_counts_to_gemm=self.publish_relay_counts_to_gemm,
            apply_topk_in_fc1=self.apply_topk_in_fc1,
            max_relay_tokens_per_node=self.max_tokens_per_rank,
            diagnostics=config.diagnostics,
            token_back_atomic_batch=config.token_back_atomic_batch,
            early_stage1=config.experimental.early_stage1,
            direct_l1_sf_staging=config.experimental.direct_l1_sf_staging,
        )

        self.topk_reduce_persistent = bool(
            topk_reduce_persistent
            and not self.in_kernel_fc2_reduce
            and not self.skip_topk_reduce
        )
        self.topk_reduce_num_sms = (
            cutlass_utils.HardwareInfo().get_device_multiprocessor_count()
            if self.topk_reduce_persistent
            else 0
        )

        if self.tail_topk_reduce:
            self.token_comm.tail_reducer = TopkReduce(
                self.hidden, self.num_topk, self.combine_format,
                sm_arch=get_cutedsl_target_arch(),
                num_remote_nodes=self.num_remote_nodes,
                ranks_per_node=self.ranks_per_node,
                num_experts_per_rank=self.num_experts_per_rank,
                combine_slot_indexed=True,
            )
        # Region layout (same call drives both get_workspace_sizes() and
        # the __call__ partition).
        self._local_region_specs = self._build_local_region_specs()
        self._shared_region_specs = self._build_shared_region_specs()
        if self.enable_nccl_gin:
            # GIN fan-out writes the final L1 pool rows of this rank from the
            # relay peers of the same node, so the peer-written pool rows and
            # their resettable metadata live in the shared (LSA) workspace.
            # The owner holds the source-bucket reservation counter.
            counter_names = {
                "l1_arrival_count", "relay_sf_pool_base", "relay_fanout_reservation",
            }
            payload_names = {
                "l1_token_buffer", "l1_sf_buffer", "l1_topk_weights_buffer",
                "token_src_metadata",
            }
            if self.token_comm.direct_l1_sf_staging:
                # Field 61: peer-written row flags (counter prefix) and the
                # contiguous SF staging rows (payload).
                counter_names.add("relay_row_ready")
                payload_names.add("relay_sf_staging")
            moved_counters = [
                r for r in self._local_region_specs if r.name in counter_names
            ]
            moved_payloads = [
                r for r in self._local_region_specs if r.name in payload_names
            ]
            self._local_region_specs = [
                r for r in self._local_region_specs
                if r.name not in counter_names | payload_names
            ]
            self._shared_region_specs = (
                moved_counters
                + self._shared_region_specs
                + moved_payloads
            )
        self._local_offsets, self._local_total = _layout_regions(
            self._local_region_specs
        )
        self._shared_offsets, self._shared_total = _layout_regions(
            self._shared_region_specs
        )
        if self.enable_nccl_gin:
            assert self._shared_offsets["l1_arrival_count"] == 0, (
                "direct readiness snapshot/reset requires readiness first in shared prefix"
            )
        self._local_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r for r in self._local_region_specs
        }
        self._shared_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r for r in self._shared_region_specs
        }

        # Counter-prefix byte extents: the accumulating counters are front-placed
        # (see _build_*_region_specs), so the leading bytes up to the first data
        # region cover exactly the per-launch zero set. ``tail_reset_counters``
        # bulk-zeros these each launch; only the first launch needs a caller-zeroed
        # workspace. 128B-aligned offsets => multiples of 4 (Int32 zeroing exact).
        local_first_data = "tokens_to_node" if self.enable_nccl_gin else "l1_token_buffer"
        local_leading = self._local_offsets[local_first_data]
        shared_leading = self._shared_offsets["src_token_topk_idx"]  # first data region
        self.require_zero_workspace_leading_bytes: Tuple[int, int] = (
            local_leading, shared_leading,
        )
        self.local_zero_i32_count = local_leading // 4
        self.shared_zero_i32_count = shared_leading // 4

    # =========================================================================
    # SMEM budget hook (base override)
    # =========================================================================

    def _dispatch_smem_bytes(self) -> int:
        """SMEM bytes for dispatch pull mbarriers, expert scratch, and token buffer."""
        pull_mbar_bytes = _DispatchWarpCount * 8
        expert_count_bytes = self.num_total_experts * 4
        pull_buffer_bytes = _DispatchWarpCount * self.hidden_bytes
        total = (
            _round_up(pull_mbar_bytes, 16)
            + _round_up(expert_count_bytes, 16)
            + _round_up(pull_buffer_bytes, 128)
        )
        if self.enable_internode_relay:
            inter_send_count_bytes = (
                self.num_remote_node_slots * self.node_experts * 4
            )
            inter_send_token_count_bytes = (
                self.num_remote_node_slots * 4
            )
            total += _round_up(inter_send_count_bytes, 16)
            total += 2 * _round_up(inter_send_token_count_bytes, 16)
            if self.enable_nccl_gin:
                assert self.relay_warp_id is not None
                combine_scratch_bytes = (
                    len(self.relay_warp_id)
                    * self.token_comm.warp_threads * self.num_topk * 16
                )
                total += _round_up(combine_scratch_bytes, 16)
        if self.token_back_standalone:
            total += (
                _round_up(_DispatchWarpCount * 8, 16)
                + _round_up(
                    _DispatchWarpCount * self.token_comm.tb_chunk_bytes, 128
                )
            )
        return total

    def _smem_misc_budget_bytes(self) -> int:
        """Base misc reservation plus dispatch-warp SMEM."""
        return super()._smem_misc_budget_bytes() + self._dispatch_smem_bytes()

    # =========================================================================
    # Pool sizing (first-principles)
    # =========================================================================

    def _pool_shapes(self) -> Tuple[int, int, int]:
        """Worst-case pool sizes.

        ``pool_token_capacity``: every received token from any peer can
        replicate to ``min(num_topk, num_experts_per_rank)`` local
        experts; worst case is ``world_size * max_tokens_per_rank``
        tokens received, each replicated up to that bound. In relay mode,
        every expert has two independently padded regions (intranode and
        relay), so each expert can waste twice
        ``cluster_tile_tokens - 1`` DATA rows. Round the whole DATA sum
        up to ``cluster_tile_tokens`` so no FC1 tile straddles the
        intranode/relay boundary.

        ``pool_sf_capacity``: same number of expert blocks as the data
        pool, each padded to ``sf_padding_block`` rows (UTCCP 4x32
        swizzle that the SF TMA load expects).

        ``pool_task_tile_capacity``: an upper bound on the sum of
        per-expert ``ceil(segment_rows / cluster_tile_tokens)`` terms.
        One extra partial-tile allowance per expert covers the fact that
        independently aligned segments cannot be collapsed into a single
        ceiling of the aggregate DATA capacity.
        """
        world_size = self.world_size
        max_tokens_per_rank = self.max_tokens_per_rank
        num_topk = self.num_topk
        num_experts_per_rank = self.num_experts_per_rank
        token_padding_block = self.token_padding_block
        sf_padding_block = self.sf_padding_block
        cluster_tile_tokens = self.cluster_tile_tokens

        max_recv = world_size * max_tokens_per_rank
        max_per_token = min(num_topk, num_experts_per_rank)
        if self.enable_internode_relay:
            # Intra and relay are separate scheduler segments. DATA aligns to
            # the cluster tile so no FC1 tile straddles the fabric boundary;
            # SF aligns independently to its 128-row atom.
            data_padding_block = cluster_tile_tokens
            raw = (
                max_recv * max_per_token
                + 2
                * num_experts_per_rank
                * (data_padding_block - 1)
            )
            sf_raw = (
                max_recv * max_per_token
                + 2
                * num_experts_per_rank
                * (sf_padding_block - 1)
            )
            pool_token_capacity = _round_up(raw, data_padding_block)
            pool_sf_capacity = _round_up(sf_raw, sf_padding_block)
        else:
            raw = (
                max_recv * max_per_token
                + num_experts_per_rank * (token_padding_block - 1)
            )
            pool_token_capacity = _round_up(raw, token_padding_block)
            pool_sf_capacity = (
                (pool_token_capacity // token_padding_block) * sf_padding_block
            )
        # Upper bound for sum_e ceil(valid_e, cluster_tile_tokens).  The
        # per-expert slack covers each expert's final partial task tile.
        pool_task_tile_capacity = (
            (pool_token_capacity + cluster_tile_tokens - 1) // cluster_tile_tokens
            + num_experts_per_rank
        )
        return (
            pool_token_capacity,
            pool_sf_capacity,
            pool_task_tile_capacity,
        )

    # =========================================================================
    # Region tables
    # =========================================================================

    def _build_local_region_specs(self) -> List[_RegionSpec]:
        """Local-only regions (no peer access via ``peer_rank_ptr_mapper.map`` in
        ``pre_unification/comm/dispatch_kernel.py``).
        """
        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        pool_task_tile_capacity = self.pool_task_tile_capacity
        num_experts_per_rank = self.num_experts_per_rank
        num_total_experts = self.num_total_experts
        hidden_bytes = self.hidden_bytes
        sf_uint32_per_token = self.sf_uint32_per_token
        intermediate_downproj = self.intermediate_downproj
        mma_tiler_n = self.mma_tiler_mnk[1]
        sf_vec_size = self.sf_vec_size
        sf_padding_block = self.sf_padding_block

        # fc1_output_sf / fc1_done_counter sizing mirrors base
        # ``get_workspace_size_in_bytes`` (kernel_fc12.py ~lines 525-543).
        sf_total_rows_upper = (
            pool_token_capacity + num_experts_per_rank * sf_padding_block
        )
        if self.enable_internode_relay:
            # Relay mode addresses SF rows against ``pool_sf_capacity``
            # (two independently padded segments per expert); single-node
            # keeps main's bound and SF TMA extent.
            sf_total_rows_upper = max(sf_total_rows_upper, pool_sf_capacity)
        sf_block_cols = (
            (((intermediate_downproj // sf_vec_size) + 3) // 4) * 4
        )
        fc1_done_slots = (
            (pool_token_capacity + mma_tiler_n - 1) // mma_tiler_n
            + num_experts_per_rank
        )

        # === Accumulating-counter prefix ===========================================
        # Front-placed so the per-launch reset is a single bulk zero of
        # ``[0:local_leading]`` (the bytes up to the first data region). These hold
        # spin thresholds / write cursors / phase-flip counters that the kernel
        # accumulates across the launch and that MUST start at 0 each launch; the
        # kernel tail (``tail_reset_counters``) bulk-zeros this prefix, so only the
        # FIRST launch relies on a caller-zeroed workspace.
        specs: List[_RegionSpec] = [
            _RegionSpec(
                "l1_arrival_count",
                cutlass.Int32,
                (pool_task_tile_capacity,),
                16,
            ),
        ]
        if self.enable_nccl_gin and self.token_comm.direct_l1_sf_staging:
            # Field 61: one ready flag per pool row, raised by the relay peer
            # after it staged the row's scale factors; zeroed with the prefix.
            # Rehomed into the shared (LSA) prefix right after the readiness
            # counters by the direct-mode relocation below.
            specs.append(
                _RegionSpec(
                    "relay_row_ready",
                    cutlass.Int32,
                    (pool_token_capacity,),
                    16,
                )
            )
        specs += [
            _RegionSpec(
                "expert_send_count",
                cutlass.Int64,
                (num_total_experts,),
                16,
            ),
            _RegionSpec(
                "grid_sync_counter",
                cutlass.Int32,
                (_GridSyncSlotCount,),
                16,
            ),
            _RegionSpec(
                "fc1_done_counter",
                cutlass.Int32,
                (fc1_done_slots,),
                16,
            ),
        ]
        # Bound CTA0's scan to decode capacities; prefill keeps full-grid routing.
        if self.max_tokens_per_rank <= 1024:
            specs.append(_RegionSpec("dispatch_route_extent", cutlass.Int32, (2,), 16))
        if self.token_back_enabled:
            if not self.enable_internode_relay:
                # Per-expert FC2 completion gate consumed by token-back. Relay
                # mode places this peer-observed counter in LSA-backed shared
                # workspace instead.
                specs.append(
                    _RegionSpec(
                        "fc2_done_counter",
                        cutlass.Int32,
                        (num_experts_per_rank,),
                        16,
                    )
                )
            if self.token_back_schedule_mode == "atomic_counter":
                specs.append(
                    _RegionSpec(
                        "token_back_schedule_counter",
                        cutlass.Int32,
                        (1,),
                        16,
                    )
                )
        if self.load_balance_mode == "atomic_counter":
            specs.append(
                _RegionSpec(
                    "load_balance_counter",
                    cutlass.Int32,
                    (1,),
                    16,
                )
            )
        if self.topk_reduce_persistent:
            specs.append(
                _RegionSpec(
                    "topk_reduce_counters",
                    cutlass.Int32,
                    (1,),
                    16,
                )
            )
        if self.enable_internode_relay:
            if not self.enable_nccl_gin:
                specs.append(
                    _RegionSpec(
                        "internode_send_count",
                        cutlass.Int32,
                        (self.num_remote_node_slots * self.node_experts,),
                        16,
                    )
                )
            specs.append(
                _RegionSpec(
                    "internode_send_token_count",
                    cutlass.Int64,
                    (self.num_remote_node_slots,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "relay_fanout_reservation",
                    cutlass.Int32,
                    (self.node_experts, self.num_remote_node_slots),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "expert_pool_base",
                    cutlass.Int32,
                    (num_experts_per_rank,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "expert_sf_pool_base",
                    cutlass.Int32,
                    (num_experts_per_rank,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "relay_sf_pool_base",
                    cutlass.Int32,
                    (num_experts_per_rank, self.num_remote_node_slots),
                    16,
                )
            )

        # === Data buffers (overwritten each launch; NOT zeroed) ====================
        # ``l1_token_buffer`` MUST be the first data region: ``__init__`` derives
        # ``local_leading`` from its offset (= end of the counter prefix).
        specs += [
            # L1 input pool (dispatch_pull writes -> fc1 reads), Uint8 bytes; the
            # NVFP4 view at the same offset is built inside ``__call__``.
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (pool_token_capacity, hidden_bytes),
                128,
            ),
        ]
        if self.enable_internode_relay:
            specs.append(
                _RegionSpec(
                    "tokens_to_node",
                    cutlass.Int32,
                    (self.num_remote_node_slots, self.max_tokens_per_rank),
                    16,
                )
            )
            # Inverse of tokens_to_node for the slot-indexed combine return:
            # token_node_slot[node_slot][token] is that token's Stage 1
            # message slot. Written by dispatch_prep, read by TopkReduce.
            specs.append(
                _RegionSpec(
                    "token_node_slot",
                    cutlass.Int32,
                    (self.num_remote_node_slots, self.max_tokens_per_rank),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "relay_fanout_slot",
                    cutlass.Int32,
                    (
                        self.num_remote_node_slots,
                        self.max_tokens_per_rank,
                        self.num_topk,
                    ),
                    16,
                )
            )
            # Small post-launch observability copies. The live counters/layout
            # are reset in kernel_tail before host validation can read them.
            specs.append(
                _RegionSpec(
                    "internode_recv_count_per_source_snapshot",
                    cutlass.Int32,
                    (self.num_remote_node_slots, num_experts_per_rank),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "expert_recv_count_snapshot",
                    cutlass.Int64,
                    (self.world_size, num_experts_per_rank),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "expert_recv_count_sum_snapshot",
                    cutlass.Int64,
                    (num_experts_per_rank,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "intra_token_count_snapshot",
                    cutlass.Int32,
                    (num_experts_per_rank,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "expert_pool_base_snapshot",
                    cutlass.Int32,
                    (num_experts_per_rank,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "expert_sf_pool_base_snapshot",
                    cutlass.Int32,
                    (num_experts_per_rank,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "relay_pool_base_snapshot",
                    cutlass.Int32,
                    (num_experts_per_rank, self.num_remote_node_slots),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "relay_sf_pool_base_snapshot",
                    cutlass.Int32,
                    (num_experts_per_rank, self.num_remote_node_slots),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "expert_scheduled_count_snapshot",
                    cutlass.Int32,
                    (num_experts_per_rank,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "pool_end_snapshot",
                    cutlass.Int64,
                    (2,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "l1_arrival_count_snapshot",
                    cutlass.Int32,
                    (pool_task_tile_capacity,),
                    16,
                )
            )
        specs += [
            # Persisted across launches (deliberately OUT of the zero prefix): the
            # sense-reversing nvlink barrier rides this phase counter across launch
            # boundaries (non-ncu back-to-back), and ncu kernel replay restores it
            # via its local-memory snapshot. Only the FIRST launch relies on the
            # caller-zeroed workspace.
            _RegionSpec(
                "nvlink_barrier_counter",
                cutlass.Int32,
                (1,),
                16,
            ),
        ]
        if self.enable_nccl_gin:
            specs.append(
                _RegionSpec(
                    "internode_barrier_counter",
                    cutlass.Int32,
                    (1,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "combine_phase_snapshot",
                    cutlass.Int32,
                    (1,),
                    16,
                )
            )
            # Monotonic count of combine return issues (one per launch); the
            # token-back warps wait for it before pushing (never zeroed).
            specs.append(
                _RegionSpec(
                    "combine_issue_counter",
                    cutlass.Int32,
                    (1,),
                    16,
                )
            )
        if self.enable_nccl_gin and self.token_comm.direct_l1_sf_staging:
            # Field 61: contiguous per-row scale-factor staging written by the
            # relay peers (16-byte lanes) and scattered locally by the owner's
            # dispatch warps into l1_sf_buffer. Rehomed into the shared (LSA)
            # payload regions by the direct-mode relocation.
            specs.append(
                _RegionSpec(
                    "relay_sf_staging",
                    cutlass.Uint8,
                    (pool_token_capacity, sf_uint32_per_token * 4),
                    128,
                )
            )
        specs += [
            # Int32 atom-flat SF buffer (dispatch_pull 32b read/write; FP8 view at
            # the same offset). Count = pool_sf_capacity * sf_uint32_per_token.
            _RegionSpec(
                "l1_sf_buffer",
                cutlass.Int32,
                (pool_sf_capacity * sf_uint32_per_token,),
                16,
            ),
            _RegionSpec(
                "l1_topk_weights_buffer",
                cutlass.Float32,
                (pool_token_capacity,),
                16,
            ),
            _RegionSpec(
                "token_src_metadata",
                cutlass.Uint8,
                (pool_token_capacity, _TokenMetadataBytes),
                16,
            ),
            _RegionSpec(
                "fc1_output",
                cutlass.Float4E2M1FN,
                (pool_token_capacity, intermediate_downproj),
                128,
            ),
            _RegionSpec(
                "fc1_output_sf",
                cutlass.Float8E4M3FN,
                (sf_total_rows_upper, sf_block_cols),
                128,
            ),
        ]
        if self.enable_fc1_activation_per_token_scale:
            # Pool-ordered per-token fc1 activation scale: dispatch_pull lane 0
            # writes one fp32 per pulled token (next to its topk weight), the
            # fc1 epilogue reads it.  Sized like ``l1_topk_weights_buffer``.
            specs.append(
                _RegionSpec(
                    "l1_fc1_activation_per_token_scale_buffer",
                    cutlass.Float32,
                    (pool_token_capacity,),
                    16,
                )
            )
        if self.token_back_by_dispatch and not self.enable_internode_relay:
            # Local fc2 DATA staging (token_back_by_dispatch modes only).
            # Relay mode reserves this in the peer-accessible LSA workspace;
            # the wire-format dtype sizes the plane.
            specs.append(
                _RegionSpec(
                    "fc2_output_workspace",
                    self.combine_format.act_dtype,
                    (pool_token_capacity, 1, self.hidden),
                    128,
                )
            )
        # The per-block SF plane is ALWAYS staged locally (then pushed
        # token-contiguously by the dispatch / standalone warps), independent of
        # whether the DATA path goes local or straight to a peer: writing SF
        # per-token to a peer would scatter one warp's 32 lanes across up to 32
        # ranks and explode the NVLink request count. So it is allocated for
        # every quantized format, not only the dispatch data path.
        if self.combine_format.is_quantized and not self.enable_internode_relay:
            # Flat padded capacity (pool_token * sf_block_pad scale entries); the
            # relay path places this peer-observed plane in LSA instead. The
            # (pool_token, 1, hidden//scale_block) logical shape + 16 B-aligned
            # per-block stride is assembled where the view is built in __call__.
            specs.append(
                _RegionSpec(
                    "fc2_output_sf",
                    self.combine_format.scale_dtype,
                    (pool_token_capacity * self.sf_block_pad,),
                    128,
                )
            )

        if self.pipeline_timing_enabled:
            specs.append(
                _RegionSpec(
                    "pipeline_timing",
                    cutlass.Int64,
                    PIPELINE_TIMING_SHAPE,
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "combine_timing_landmarks",
                    cutlass.Int64,
                    COMBINE_TIMING_LANDMARK_SHAPE,
                    16,
                )
            )
        if self.fc_tile_timeline_enabled:
            specs.append(
                _RegionSpec(
                    "fc_tile_timeline",
                    cutlass.Int64,
                    FC_TILE_TIMELINE_SHAPE,
                    16,
                )
            )
        return specs

    def _build_shared_region_specs(self) -> List[_RegionSpec]:
        """Shared (peer-mapped) regions -- every entry is reached from
        some ``peer_rank_ptr_mapper.map(local_ptr, peer_rank, byte_off)``
        call site inside ``pre_unification/comm/dispatch_kernel.py``:

          * ``src_token_topk_idx`` -- ``_dispatch_prep`` round 3
          * ``expert_recv_count`` / ``expert_recv_count_sum``
            -- ``_dispatch_barrier`` step 2 (b64 store + sys-atomic-add)
          * ``nvlink_barrier_signal``
            -- ``_nvlink_barrier_3stage`` stage B (two reusable phase slots)
        """
        world_size = self.world_size
        num_topk = self.num_topk
        max_tokens_per_rank = self.max_tokens_per_rank
        num_experts_per_rank = self.num_experts_per_rank

        # ``MAX_SLOT`` in ``_dispatch_prep`` round 3: every (token, topk)
        # edge any peer might publish for this rank's local experts.
        max_slot = max_tokens_per_rank * num_topk

        # Accumulating counters first (zero-prefix -> tail bulk-zeros
        # ``[0:shared_leading]``); then the data/signal regions that must persist
        # across launches (``src_token_topk_idx`` overwritten each launch;
        # ``nvlink_barrier_signal`` is phase-flip and must NOT be zeroed).
        # ``src_token_topk_idx`` MUST be the first non-counter region: ``__init__``
        # derives ``shared_leading`` from its offset (= end of the counter prefix).
        specs: List[_RegionSpec] = [
            _RegionSpec(
                "expert_recv_count",
                cutlass.Int64,
                (world_size, num_experts_per_rank),
                16,
            ),
            _RegionSpec(
                "expert_recv_count_sum",
                cutlass.Int64,
                (num_experts_per_rank,),
                16,
            ),
        ]
        if self.enable_internode_relay:
            # The scheduler reads these locally, while Combine Stage 1 maps the
            # same symmetric offsets on every node-local expert rank.
            specs.append(
                _RegionSpec(
                    "intra_token_count",
                    cutlass.Int32,
                    (num_experts_per_rank,),
                    16,
                )
            )
            specs.append(
                _RegionSpec(
                    "relay_pool_base",
                    cutlass.Int32,
                    (num_experts_per_rank, self.num_remote_node_slots),
                    16,
                )
            )
            # Every rail rank atomically redistributes its received R*E vector
            # into the owning node-local rank. This counter is therefore LSA
            # peer-accessed, not ordinary local CUDA memory.
            specs.append(
                _RegionSpec(
                    "internode_recv_count_per_source",
                    cutlass.Int32,
                    (self.num_remote_node_slots, num_experts_per_rank),
                    16,
                )
            )
            if self.enable_nccl_gin:
                # Owner-published packed (count, DATA base, SF base) per
                # source bucket for the fan-out reservation; LSA peer-read
                # with one system-acquire load.
                assert self.pool_token_capacity <= DIRECT_META_FIELD_MASK, (
                    "direct packed meta needs pool_token_capacity < 2^22"
                )
                assert self.pool_sf_capacity <= DIRECT_META_FIELD_MASK, (
                    "direct packed meta needs pool_sf_capacity < 2^22"
                )
                specs.append(
                    _RegionSpec(
                        "relay_direct_bucket_meta",
                        cutlass.Int64,
                        (num_experts_per_rank, self.num_remote_node_slots),
                        16,
                    )
                )
            if self.token_back_enabled:
                specs.append(
                    _RegionSpec(
                        "fc2_done_counter",
                        cutlass.Int32,
                        (num_experts_per_rank,),
                        16,
                    )
                )
        specs += [
            _RegionSpec(
                "src_token_topk_idx",
                cutlass.Int32,
                (num_experts_per_rank, world_size, max_slot),
                16,
            ),
            _RegionSpec(
                "nvlink_barrier_signal",
                cutlass.Int32,
                (_NvlinkSlotCount,),
                16,
            ),
        ]
        if self.enable_internode_relay:
            # Combine Stage 1 is a node-local peer pull: relay ranks read every
            # local expert rank's FC2 staging slots through the LSA mapper.  Keep
            # the reservation here even while epi_warps still writes the legacy
            # target so Stage 4 does not require another address migration.
            specs.append(
                _RegionSpec(
                    "fc2_output_workspace",
                    self.combine_format.act_dtype,
                    (self.pool_token_capacity, 1, self.hidden),
                    128,
                )
            )
            if self.combine_format.is_quantized:
                specs.append(
                    _RegionSpec(
                        "fc2_output_sf",
                        self.combine_format.scale_dtype,
                        (self.pool_token_capacity * self.sf_block_pad,),
                        128,
                    )
                )
        # separate-kernel-reduce only: the per-topk fc2 combine staging buffer is
        # internalized here instead of being a caller tensor, so the public output
        # is the 2D (T, hidden) reduce result.  It is the cross-rank combine STG
        # target, hence must live on the symmetric heap (= this shared workspace);
        # appended after the data regions so it stays out of the per-launch zero
        # prefix (only the first launch relies on the caller-zeroed workspace).
        if not self.in_kernel_fc2_reduce:
            # combine_quant: the cross-rank combine data plane, one cell per
            # (token, topk). dtype follows the wire format -- the bf16 baseline is
            # byte-identical to the old ``combine_partial``; fp4/e4m3 shrink it.
            specs.append(
                _RegionSpec(
                    "combine_quant",
                    self.combine_format.act_dtype,
                    (
                        (2, max_tokens_per_rank, num_topk, self.hidden)
                        if self.enable_nccl_gin
                        else (max_tokens_per_rank, num_topk, self.hidden)
                    ),
                    128,
                )
            )
            # combine_sf: per-block scale plane; only quantized formats carry one.
            if self.combine_format.is_quantized:
                # Flat padded capacity (max_tokens * num_topk * sf_block_pad scale
                # entries); the (max_tokens, num_topk, hidden//scale_block) logical
                # shape + 16 B-aligned per-block stride is assembled where the view
                # is built in __call__.
                specs.append(
                    _RegionSpec(
                        "combine_sf",
                        self.combine_format.scale_dtype,
                        (max_tokens_per_rank * num_topk * self.sf_block_pad,),
                        128,
                    )
                )
        return specs

    # =========================================================================
    # Public: workspace size query
    # =========================================================================

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return ``(local_ws_bytes, shared_ws_bytes)`` -- the byte
        budgets for the two opaque workspaces the host must allocate.
        Both totals are invariant across launches; per-launch ``T``
        may be <= ``max_tokens_per_rank``.
        """
        return self._local_total, self._shared_total

    def topk_reduce_input_region(self) -> Dict[str, Any]:
        """Describe the borrowed BF16 ``combine_quant`` staging region."""
        if not self.skip_topk_reduce:
            raise RuntimeError("skip_topk_reduce mode is not enabled")
        spec = self._shared_region_by_name["combine_quant"]
        if spec.cute_dtype is not cutlass.BFloat16:
            raise RuntimeError("skip_topk_reduce staging must be BF16")
        return {
            "name": spec.name,
            "byte_offset": self._shared_offsets[spec.name],
            "nbytes": spec.nbytes,
            "shape": spec.shape,
            "stride": spec.stride_row_major,
            "dtype": "bfloat16",
            "alignment": spec.align,
        }

    # =========================================================================
    # Workspace partition helpers
    # =========================================================================

    @staticmethod
    def _make_typed_view(
        byte_base: cute.Pointer,
        byte_offset: int,
        cute_dtype: Any,
        shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]],
        assumed_align: int,
    ) -> cute.Tensor:
        """Build a typed cute view at ``byte_offset`` of the opaque workspace.

        The workspace is a raw ``cute.Pointer`` (uint8 gmem base), not a tensor:
        the kernel only ever needs the base address + its own byte-offset table, so
        a tensor's shape would be both ignored AND, for >2 GiB workspaces (the
        internalized combine staging), overflow cute's 32-bit memref shape field.
        """
        # Large MegaMoE problems place later regions above the 2 GiB / 4 GiB
        # boundary; keep the base adjustment in 64-bit pointer arithmetic so region
        # starts (fc1_output_sf / counters) do not wrap before the typed view.
        byte_ptr = byte_base + Int64(byte_offset)
        typed_iter = cute.make_ptr(
            cute_dtype,
            byte_ptr.toint(),
            AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        return cute.make_tensor(typed_iter, cute.make_layout(shape, stride=stride))

    def _view_local(
        self,
        local_workspace: cute.Pointer,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        """Partition a region of the local workspace.  With no overrides,
        uses the region's declared dtype + shape + row-major stride;
        overrides let dual-view callers build alternate-dtype views at
        the same byte offset.
        """
        return self._partition_region(
            local_workspace,
            self._local_offsets,
            self._local_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _view_shared(
        self,
        shared_workspace: cute.Pointer,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        return self._partition_region(
            shared_workspace,
            self._shared_offsets,
            self._shared_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _view_workspace(self, local_workspace, shared_workspace, name, **overrides):
        """Resolve mode-dependent pool placement from the actual region table."""
        if name in self._shared_region_by_name:
            return self._view_shared(shared_workspace, name, **overrides)
        return self._view_local(local_workspace, name, **overrides)

    def _partition_region(
        self,
        byte_workspace: cute.Pointer,
        offsets: Dict[str, int],
        spec: _RegionSpec,
        *,
        cute_dtype: Optional[Any],
        shape: Optional[Tuple[int, ...]],
        stride: Optional[Tuple[int, ...]],
    ) -> cute.Tensor:
        dt = cute_dtype if cute_dtype is not None else spec.cute_dtype
        sh = shape if shape is not None else spec.shape
        st = stride
        if st is None:
            if cute_dtype is None and shape is None:
                st = spec.stride_row_major
            else:
                # Derive row-major from the (possibly overridden) shape.
                out: List[int] = [1]
                for d in reversed(list(sh)[1:]):
                    out.append(out[-1] * d)
                out.reverse()
                st = tuple(out)
        return self._make_typed_view(
            byte_workspace, offsets[spec.name], dt, sh, st, spec.align,
        )

    # =========================================================================
    # __call__
    # =========================================================================

    def name(self) -> str:
        """Full compiled-kernel cache key. Self-contained on purpose (no shared
        helper): mirrors the base fc12 fields and appends the MegaMoE-specific
        ones -- keep the shared part in sync with the base ``name()``.
        ``local_rank`` excluded (deployment detail); same dropped set as base."""
        m, n, k = self.mma_tiler_mnk
        cm, cn = self.cluster_shape_mn
        exp = "x".join(map(str, self.static_expert_shape)) if self.static_expert_shape else "dyn"
        epiflag = "x".join(map(str, self.epi_flag_batch)) if self.epi_flag_batch else "none"
        cta = "2_cta" if self.use_2cta_instrs else "1_cta"
        fc2store = "fc2store_stg" if self.non_ubulk_fc2_store else "fc2store_ublk"
        inkred = "inkernel_redg" if self.in_kernel_fc2_reduce else "no_inkernel_redg"
        topk_reduce = "skip_topk_reduce" if self.skip_topk_reduce else "do_topk_reduce"
        reduce_sched = (
            "persistent_topk_reduce"
            if self.topk_reduce_persistent
            else "static_topk_reduce"
        )
        apply_topk = "apply_topk_fc1_pre_quant" if self.apply_topk_in_fc1 else "apply_topk_after_fc2"
        token_back = {
            "epi_warps": "epiwarps",
            "standalone_warps": "standalone",
            "reuse_dispatch_warps": "reuse_dispatch",
        }.get(self.token_back_mode, self.token_back_mode)
        # Empty when off so existing compiled-kernel cache keys stay unchanged.
        act_scale = (
            "_fc1actptscale" if self.enable_fc1_activation_per_token_scale else ""
        )
        internode = (
            f"_internode_relay_meta1_rpn{self.ranks_per_node}"
            if self.enable_internode_relay else ""
        )
        return (
            "megamoe_nvfp4"
            f"_mmatiler_{m}x{n}x{k}_cluster_{cm}x{cn}_{cta}_sched_{self.load_balance_mode}"
            f"_expert_shape_{exp}_grouphint_{self.group_hint}"
            f"_padding_{self.token_padding_block}x{self.sf_padding_block}"
            f"_{fc2store}_{inkred}_{topk_reduce}_{reduce_sched}"
            f"_token_back_by_{token_back}_{apply_topk}"
            f"{internode}"
            f"_fc2out{self.fc2_output_dtype.__name__}_combine{self.combine_format}_sfvec{self.sf_vec_size}"
            f"_acc{self.acc_dtype.__name__}_swiglua{self.swiglu_alpha}"
            f"_swiglub{self.swiglu_beta}_clamp{self.gate_up_clamp}_epiflag{epiflag}"
            f"_situ{self.situ_beta}_linear{self.situ_linear_beta}"
            f"{act_scale}"
            # MegaMoE-specific constexpr:
            f"_ep_{self.world_size}_topk_{self.num_topk}_maxtoken_{self.max_tokens_per_rank}"
            f"_flagbatch_{self.flag_batch}"
            # Codegen options and diagnostics (MegaMoEConfig.name_suffix):
            # one token per non-default value, fixed order.
            f"{self.config.name_suffix()}"
        )

    # -- AOT compile / load (TVM-FFI calling convention) ----------------------

    def aot_compile(self, out_path: Optional[str] = None, max_active_clusters: Optional[int] = None):
        """Compile against fake (metadata-only) inputs. ``out_path=None`` returns the
        in-memory compiled callable (serial path); a path serializes the callable to a
        ``.o`` and returns it (parallel path: the object isn't picklable, so it crosses
        processes as a file). ``max_active_clusters`` (the persistent-scheduler occupancy
        const) is queried from the device when ``None``; the parallel path passes a
        precomputed value so the spawn worker needs no occupancy-probe compile.

        ``cute.compile[EnableTVMFFI(True)]`` -> the callable takes plain framework
        tensors (no per-call from_dlpack) plus an explicit stream. The fake tensors
        define the runtime ABI, so each must mirror what the caller stages
        (generate_inputs / to_kernel_inputs): per-rank EP slice, SF column padding, and
        to_cute's mark_layout_dynamic (stride-1 dim stays static, every other dim is a
        runtime SymInt). Static extents are this kernel's constexpr; the dynamic
        token/expert counts are placeholders. ``stride_order`` follows
        make_data_tensor's ``memory_order`` (0 == innermost)."""
        import math

        import cutlass.utils as cutlass_utils
        from cutlass.cutlass_dsl import Int32
        from cutlass.cute.runtime import (
            make_fake_compact_tensor,
            make_fake_stream,
            make_ptr,
        )
        from cutlass.cute.typing import sym_int64

        if self.enable_internode_relay:
            from src.lsa_peer_mapper import LsaPeerMapperHost as PeerMapperHost
            mapper_rank_count = self.ranks_per_node
        else:
            from src.sym_buffer import SymBufferHost as PeerMapperHost
            mapper_rank_count = self.world_size

        def round_up(value: int, multiple: int) -> int:
            return ((value + multiple - 1) // multiple) * multiple

        tokens = self.max_tokens_per_rank
        hidden = self.hidden
        topk = self.num_topk
        experts = self.num_experts_per_rank
        fc1_n = self.intermediate_gateup
        intermediate = self.intermediate_downproj
        sf_vec = self.sf_vec_size
        activation_sf_cols = self.sf_uint32_per_token * 4          # _stage_activation_sf padding
        fc1_sf_cols = round_up(fc1_n, 128) * round_up(hidden // sf_vec, 4)
        fc2_sf_cols = round_up(hidden, 128) * round_up(intermediate // sf_vec, 4)
        data_dtype = cutlass.Float4E2M1FN
        scale_dtype = cutlass.Float8E4M3FN
        float32 = cutlass.Float32

        def fake_tensor(dtype, shape, stride_order, dynamic_dims, alignment):
            # dynamic_dims -> runtime SymInt(div=gcd(extent,128)); the stride-1
            # (contiguous) dim stays static -- mirrors to_cute's mark_layout_dynamic.
            extents = tuple(
                sym_int64(divisibility=math.gcd(int(extent), 128))
                if axis in dynamic_dims else int(extent)
                for axis, extent in enumerate(shape)
            )
            return make_fake_compact_tensor(
                dtype, extents, stride_order=stride_order, assumed_align=alignment,
            )

        # alignment mirrors to_cute's proven-safe values (16 / element_size / 4).
        fake = dict(
            activation=fake_tensor(data_dtype, (tokens, hidden), (1, 0), {0}, 16),
            activation_sf=fake_tensor(scale_dtype, (tokens, activation_sf_cols), (1, 0), {0}, 16),
            topk_idx=fake_tensor(Int64, (tokens, topk), (1, 0), {0}, 8),
            topk_weights=fake_tensor(float32, (tokens, topk), (1, 0), {0}, 16),
            fc1_weight=fake_tensor(data_dtype, (experts, hidden, fc1_n), (2, 0, 1), {0, 2}, 16),
            fc1_weight_sf=fake_tensor(scale_dtype, (experts, fc1_sf_cols), (1, 0), {0}, 16),
            fc2_weight=fake_tensor(data_dtype, (experts, intermediate, hidden), (2, 0, 1), {0, 2}, 16),
            fc2_weight_sf=fake_tensor(scale_dtype, (experts, fc2_sf_cols), (1, 0), {0}, 16),
            fc1_alpha=fake_tensor(float32, (experts,), (0,), set(), 4),
            fc2_alpha=fake_tensor(float32, (experts,), (0,), set(), 4),
            fc1_norm_const=fake_tensor(float32, (experts,), (0,), set(), 4),
            output_activation=fake_tensor(self.fc2_output_dtype, (tokens, hidden), (1, 0), {0}, 16),
            # Opaque byte workspaces: a bare base pointer (no shape), like to_cute_ptr.
            local_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=16),
            shared_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=16),
            # Field values are placeholders (marshalled as runtime scalars); only
            # num_max_ranks is constexpr, so it must be exact.
            peer_rank_ptr_mapper_host=PeerMapperHost(
                base_addr=Int64(0),
                offsets=tuple(Int64(0) for _ in range(mapper_rank_count)),
                rank_idx=Int32(0),
                num_max_ranks=mapper_rank_count,
            ),
            max_active_clusters=(
                max_active_clusters if max_active_clusters is not None
                else int(cutlass_utils.HardwareInfo().get_max_active_clusters(
                    self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
                ))
            ),
            stream=make_fake_stream(),       # explicit stream arg -> caller keeps launch control
        )
        if self.topk_reduce_persistent:
            # The persistent reduce reads its token count from device memory, so
            # the AOT ABI carries the int32[1] operand (alignment 4, static shape,
            # as the shim passes it); static-reduce builds omit it.
            fake["num_valid_tokens"] = fake_tensor(Int32, (1,), (0,), set(), 4)

        if self.enable_fc1_activation_per_token_scale:
            # Part of the ABI only when enabled (the default ABI is unchanged).
            fake["fc1_activation_per_token_scale"] = fake_tensor(
                float32, (tokens,), (0,), {0}, 4
            )

        compiled = cute.compile[cute.EnableTVMFFI(True)](self, **fake)
        if out_path is None:
            return compiled
        # The TVM-FFI compiled object has its own export_to_c that renames the
        # __tvm_ffi_<internal> symbol to __tvm_ffi_<function_name> -- the entry
        # load_module(..., enable_tvm_ffi=True)[prefix] looks up. (dump_to_object
        # is the non-tvm path; its generic prefix-rename misnames that symbol.)
        compiled.export_to_c(
            out_path, function_name=_aot_symbol_prefix, export_only_tvm_ffi_symbols=True,
        )
        return out_path

    @staticmethod
    def load_compiled(path: str):
        """Load a ``.o`` from ``aot_compile`` into a callable with the same TVM-FFI
        ABI. JITLink only (no ptxas); ``enable_tvm_ffi`` resolves the ``__tvm_ffi_*``
        entry."""
        from cutlass.cute.runtime import load_module

        return load_module(path, enable_tvm_ffi=True)[_aot_symbol_prefix]

    @cute.jit
    def launch_without_gin(
        self,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_norm_const: cute.Tensor,
        output_activation: cute.Tensor,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Compile entry point whose runtime ABI contains no GIN resources."""
        self(
            activation=activation, activation_sf=activation_sf,
            topk_idx=topk_idx, topk_weights=topk_weights,
            fc1_weight=fc1_weight, fc1_weight_sf=fc1_weight_sf,
            fc2_weight=fc2_weight, fc2_weight_sf=fc2_weight_sf,
            fc1_alpha=fc1_alpha, fc2_alpha=fc2_alpha, fc1_norm_const=fc1_norm_const,
            output_activation=output_activation, local_workspace=local_workspace,
            shared_workspace=shared_workspace, peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            max_active_clusters=max_active_clusters, stream=stream,
        )

    @cute.jit
    def __call__(
        self,
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,           # (T, hidden) NVFP4
        activation_sf: cute.Tensor,        # (T, round_up(hidden, sf_atom_block_k)) FP8
        topk_idx: cute.Tensor,             # (T, num_topk) Int64
        topk_weights: cute.Tensor,         # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_norm_const: cute.Tensor,
        # Final combined output the caller consumes: 2D (T, hidden).  Under
        # in_kernel_reduce it is the cross-rank REDG target (symmetric heap);
        # under separate_kernel_reduce it is the local tail-reduce destination
        # while the per-topk staging lives in the internal ``combine_quant``.
        output_activation: cute.Tensor,    # (T, hidden) BF16
        # Opaque workspaces.
        local_workspace: cute.Pointer,     # uint8 gmem base of (local_ws_bytes,)
        shared_workspace: cute.Pointer,    # uint8 gmem base of (shared_ws_bytes,)
        # NCCL device resources and registered Stage 1 metadata views. They are
        # None in the non-GIN instantiation and disappear from its runtime ABI.
        # Runtime host payload; packed into ``SymBuffer{world_size}``
        # before entering the device kernel.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
        # Optional per-token fc1 activation scale: (T,) Float32 on the
        # symmetric heap (peer-pulled by dispatch like ``topk_weights``).
        # Required iff ``enable_fc1_activation_per_token_scale``; ignored
        # (and absent from the ABI) otherwise.
        fc1_activation_per_token_scale: Optional[cute.Tensor] = None,
        num_valid_tokens: Optional[cute.Tensor] = None,
        gin_dev_comm=None,
        gin_window=None,
        gin_internode_send_count=None,
        gin_internode_recv_splits=None,
        gin_internode_combine_buffer=None,
    ) -> None:
        """Launch the MegaMoE-complete fused kernel.

        Pointer-mapping contract:
          * ``activation`` / ``activation_sf`` / ``topk_weights`` MUST
            point into memory reachable via ``peer_rank_ptr_mapper.map(...)``
            (typically NVSHMEM symmetric heap).  Single-rank degenerate
            runs (``peer_rank_ptr_mapper.offsets[local_rank] == 0`` by NVSHMEM
            convention) are allowed.
          * ``fc1_activation_per_token_scale`` (when enabled) is peer-pulled
            by the dispatch warps exactly like ``topk_weights`` and MUST be
            reachable via ``peer_rank_ptr_mapper.map(...)`` as well.
          * ``topk_idx`` is read on the local rank only; placement is
            unconstrained (cuda local or sym heap).
          * ``fc1_weight`` / ``fc1_weight_sf`` / ``fc2_weight`` /
            ``fc2_weight_sf`` are local-only.
          * ``output_activation`` is the 2D (T, hidden) result.  Under
            ``in_kernel_reduce`` it is the per-rank cross-rank combine STG
            target and MUST be reachable via the peer mapper (sym heap, or
            local in the single-rank degenerate case).  Under
            ``separate_kernel_reduce`` the cross-rank target is the internal
            ``combine_quant`` staging region and ``output_activation`` only
            receives the local tail reduce, so it may be plain local memory.

        Workspace zero-init contract: caller is currently expected to
        zero ``shared_workspace`` before launch (the dispatch
        primitives' counters / signals rely on a clean state).  This
        contract may be tightened later to have the kernel take
        ownership of the reset.
        """
        # ``max_active_clusters`` and ``cluster_size`` are both Python ints
        # at trace time, so the product folds to a Python int that flows
        # cleanly to every dispatch primitive's ``num_sms: Constexpr[int]``
        # slot.
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        if cutlass.const_expr(
            self.pipeline_timing_enabled
            and sm_count > PIPELINE_TIMING_MAX_CTAS
        ):
            raise ValueError(
                "pipeline timing CTA capacity exceeded: "
                f"sm_count={sm_count} max={PIPELINE_TIMING_MAX_CTAS}"
            )
        pipeline_timing = (
            self._view_local(local_workspace, "pipeline_timing")
            if self.pipeline_timing_enabled else None
        )
        combine_timing_landmarks = (
            self._view_local(
                local_workspace, "combine_timing_landmarks"
            )
            if self.pipeline_timing_enabled else None
        )
        fc_tile_timeline = (
            self._view_local(local_workspace, "fc_tile_timeline")
            if self.fc_tile_timeline_enabled else None
        )
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        hidden = self.hidden
        sf_per_token_fp8 = self.sf_uint32_per_token * 4  # 4 FP8 SFs per Int32

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + NVFP4
        # view (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_workspace(
            local_workspace, shared_workspace, "l1_token_buffer",
        )
        l1_token_buffer_nvfp4 = self._view_workspace(
            local_workspace, shared_workspace, "l1_token_buffer",
            cute_dtype=cutlass.Float4E2M1FN,
            shape=(pool_token_capacity, hidden),
            stride=(hidden, 1),
        )

        # L1 SF buffer: Int32 view (dispatch_pull's [j, t] 2D indexing) +
        # FP8 view (base.activation_sf re-views via tile_atom_to_shape_SF
        # off the iterator, so the stride here is informational only).
        l1_sf_buffer_i32 = self._view_workspace(
            local_workspace, shared_workspace, "l1_sf_buffer",
        )
        l1_sf_buffer_fp8 = self._view_workspace(
            local_workspace, shared_workspace, "l1_sf_buffer",
            cute_dtype=cutlass.Float8E4M3FN,
            shape=(pool_sf_capacity, sf_per_token_fp8),
            stride=(sf_per_token_fp8, 1),
        )

        l1_topk_weights_buffer = self._view_workspace(
            local_workspace, shared_workspace, "l1_topk_weights_buffer",
        )
        if cutlass.const_expr(self.enable_fc1_activation_per_token_scale):
            if cutlass.const_expr(fc1_activation_per_token_scale is None):
                raise ValueError(
                    "enable_fc1_activation_per_token_scale=True but no "
                    "fc1_activation_per_token_scale tensor was provided."
                )
            act_scale_input = fc1_activation_per_token_scale
            l1_fc1_activation_per_token_scale_buffer = self._view_local(
                local_workspace,
                "l1_fc1_activation_per_token_scale_buffer",
            )
        else:
            act_scale_input = None
            l1_fc1_activation_per_token_scale_buffer = None
        l1_arrival_count = self._view_workspace(
            local_workspace, shared_workspace, "l1_arrival_count",
        )
        # One fully valid 8B TokenSrcMetadata per row, for both writer modes.
        token_src_metadata = self._view_workspace(
            local_workspace, shared_workspace, "token_src_metadata",
        )
        expert_send_count = self._view_local(
            local_workspace, "expert_send_count",
        )
        grid_sync_counter = self._view_local(
            local_workspace, "grid_sync_counter",
        )
        nvlink_barrier_counter = self._view_local(
            local_workspace, "nvlink_barrier_counter",
        )
        if cutlass.const_expr(self.enable_nccl_gin):
            internode_barrier_counter = self._view_local(
                local_workspace, "internode_barrier_counter",
            )
            combine_phase_snapshot = self._view_local(
                local_workspace, "combine_phase_snapshot",
            )
            combine_issue_counter = self._view_local(
                local_workspace, "combine_issue_counter",
            )
        else:
            internode_barrier_counter = None
            combine_phase_snapshot = None
            combine_issue_counter = None
        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_output_sf = self._view_local(local_workspace, "fc1_output_sf")
        fc1_done_counter = self._view_local(
            local_workspace, "fc1_done_counter",
        )

        load_balance_counter: Optional[cute.Tensor] = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local(
                local_workspace, "load_balance_counter",
            )

        # Shared regions.
        src_token_topk_idx = self._view_shared(
            shared_workspace, "src_token_topk_idx",
        )
        expert_recv_count = self._view_shared(
            shared_workspace, "expert_recv_count",
        )
        expert_recv_count_sum = self._view_shared(
            shared_workspace, "expert_recv_count_sum",
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace, "nvlink_barrier_signal",
        )

        # i32 stride=(2,) view onto the i64 ``expert_recv_count_sum``
        # buffer -- low32 bits hold per-expert total token count after
        # _dispatch_barrier; zero-copy alias for sizes-mode scheduling.
        expert_token_sizes = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank,),
            stride=(2,),
        )

        # MoE-domain ``(token_max, topk, hidden)`` cross-rank combine STG target.
        #   * in_kernel_reduce: REDG collapses topk on the fly, so this is a
        #     ``(T, 1, hidden)`` view of the caller's 2D ``output_activation``
        #     (the epilogue's topk index is a constexpr 0 in this mode).
        #   * separate_kernel_reduce: peers write one cell per (token, topk) into
        #     the internal ``combine_quant`` staging; the tail reduce below
        #     collapses topk into ``output_activation``.
        if cutlass.const_expr(self.in_kernel_fc2_reduce):
            combine_target = cute.make_tensor(
                output_activation.iterator,
                cute.make_layout(
                    (self.max_tokens_per_rank, 1, self.hidden),
                    stride=(self.hidden, self.hidden, 1),
                ),
            )
        else:
            combine_target = self._view_shared(shared_workspace, "combine_quant")

        # Per-block scale plane parallel to combine_quant; quantized formats only
        # (bf16 carries no SF, and in_kernel_reduce is bf16-by-construction).
        sf_blocks = (
            self.hidden // self.combine_format.scale_block
            if self.combine_format.is_quantized else 0
        )
        if cutlass.const_expr(
            self.combine_format.is_quantized and not self.in_kernel_fc2_reduce
        ):
            # Assemble the (token, topk, valid_blocks) logical view + 16 B-aligned
            # per-block stride over the flat combine_sf capacity.
            combine_sf = self._view_shared(
                shared_workspace, "combine_sf",
                shape=(self.max_tokens_per_rank, self.num_topk, sf_blocks),
                stride=(self.num_topk * self.sf_block_pad, self.sf_block_pad, 1),
            )
        else:
            combine_sf = None

        if cutlass.const_expr(self.combine_format.is_quantized):
            # Same: logical (pool_token, 1, valid_blocks) + padded per-block stride
            # over the flat fc2_output_sf capacity. Relay mode places the plane
            # in LSA because the future combine relay peer reads it directly.
            if cutlass.const_expr(self.enable_internode_relay):
                fc2_output_sf_phys = self._view_shared(
                    shared_workspace, "fc2_output_sf",
                    shape=(self.pool_token_capacity, 1, sf_blocks),
                    stride=(self.sf_block_pad, self.sf_block_pad, 1),
                )
            else:
                fc2_output_sf_phys = self._view_local(
                    local_workspace, "fc2_output_sf",
                    shape=(self.pool_token_capacity, 1, sf_blocks),
                    stride=(self.sf_block_pad, self.sf_block_pad, 1),
                )
            sf_vec = self.combine_format.scale_block
            sf_layout = fc2_output_sf_phys.layout
            fc2_output_sf = cute.make_tensor(
                fc2_output_sf_phys.iterator,
                cute.make_layout(
                    (sf_layout.shape[0], sf_layout.shape[1], (sf_vec, sf_layout.shape[2])),
                    stride=(sf_layout.stride[0], sf_layout.stride[1], (0, sf_layout.stride[2])),
                ),
            )
        else:
            fc2_output_sf = None

        if cutlass.const_expr(self.fc2_stage_locally):
            if cutlass.const_expr(self.enable_internode_relay):
                fc2_output_workspace_native = self._view_shared(
                    shared_workspace, "fc2_output_workspace",
                )
                fc2_ws_base = shared_workspace
                fc2_ws_offset = self._shared_offsets["fc2_output_workspace"]
                fc2_ws_align = self._shared_region_by_name[
                    "fc2_output_workspace"
                ].align
            else:
                fc2_output_workspace_native = self._view_local(
                    local_workspace, "fc2_output_workspace",
                )
                fc2_ws_base = local_workspace
                fc2_ws_offset = self._local_offsets["fc2_output_workspace"]
                fc2_ws_align = self._local_region_by_name[
                    "fc2_output_workspace"
                ].align
            # Byte count = elements * dtype.width // 8 (multiply before divide so
            # the 4-bit fp4 data plane is not truncated to zero bytes/element).
            fc2_output_workspace_u8 = self._make_typed_view(
                fc2_ws_base,
                fc2_ws_offset,
                cutlass.Uint8,
                ((pool_token_capacity * self.hidden
                  * int(self.combine_format.act_dtype.width)) // 8,),
                None,
                fc2_ws_align,
            )
            combine_output_u8 = cute.recast_tensor(
                combine_target, cutlass.Uint8,
            )
        else:
            fc2_output_workspace_native = None
            fc2_output_workspace_u8 = None
            combine_output_u8 = combine_target

        # fc2 completion gate: present whenever token-back runs (DATA and/or SF
        # push), so the epi_warps SF-only push gates per expert just like the
        # dispatch DATA path.
        if cutlass.const_expr(self.token_back_enabled):
            if cutlass.const_expr(self.enable_internode_relay):
                fc2_done_counter = self._view_shared(
                    shared_workspace, "fc2_done_counter",
                )
            else:
                fc2_done_counter = self._view_local(
                    local_workspace, "fc2_done_counter",
                )
        else:
            fc2_done_counter = None

        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            token_back_schedule_counter = self._view_local(
                local_workspace, "token_back_schedule_counter",
            ).iterator
        else:
            token_back_schedule_counter = None

        # Int32 views over each workspace's front counter prefix; the kernel tail
        # bulk-zeros them every launch (tail_reset_counters).
        local_zero_prefix = self._make_typed_view(
            local_workspace, 0, cutlass.Int32, (self.local_zero_i32_count,), (1,), 16,
        )
        shared_zero_prefix = self._make_typed_view(
            shared_workspace, 0, cutlass.Int32, (self.shared_zero_i32_count,), (1,), 16,
        )
        if cutlass.const_expr(self.enable_internode_relay):
            if cutlass.const_expr(self.enable_nccl_gin):
                internode_send_count = gin_internode_send_count
                internode_recv_splits = gin_internode_recv_splits
            else:
                internode_send_count = self._view_local(
                    local_workspace, "internode_send_count",
                )
                internode_recv_splits = None
            internode_send_token_count = self._view_local(
                local_workspace, "internode_send_token_count",
            )
            tokens_to_node = self._view_local(
                local_workspace, "tokens_to_node",
            )
            token_node_slot = self._view_local(
                local_workspace, "token_node_slot",
            )
            intra_token_count = self._view_shared(
                shared_workspace, "intra_token_count",
            )
            internode_recv_count_per_source = self._view_shared(
                shared_workspace, "internode_recv_count_per_source",
            )
            relay_fanout_reservation = self._view_workspace(
                local_workspace, shared_workspace, "relay_fanout_reservation",
            )
            relay_fanout_slot = self._view_local(
                local_workspace, "relay_fanout_slot",
            )
            expert_pool_base = self._view_local(
                local_workspace, "expert_pool_base",
            )
            expert_sf_pool_base = self._view_local(
                local_workspace, "expert_sf_pool_base",
            )
            relay_pool_base = self._view_shared(
                shared_workspace, "relay_pool_base",
            )
            relay_sf_pool_base = self._view_workspace(
                local_workspace, shared_workspace, "relay_sf_pool_base",
            )
            relay_direct_bucket_meta = self._view_shared(
                shared_workspace, "relay_direct_bucket_meta",
            )
            relay_row_ready = (
                self._view_shared(shared_workspace, "relay_row_ready")
                if self.token_comm.direct_l1_sf_staging else None
            )
            relay_sf_staging = (
                self._view_shared(shared_workspace, "relay_sf_staging")
                if self.token_comm.direct_l1_sf_staging else None
            )
            internode_recv_count_per_source_snapshot = self._view_local(
                local_workspace,
                "internode_recv_count_per_source_snapshot",
            )
            expert_recv_count_snapshot = self._view_local(
                local_workspace, "expert_recv_count_snapshot",
            )
            expert_recv_count_sum_snapshot = self._view_local(
                local_workspace, "expert_recv_count_sum_snapshot",
            )
            intra_token_count_snapshot = self._view_local(
                local_workspace, "intra_token_count_snapshot",
            )
            expert_pool_base_snapshot = self._view_local(
                local_workspace, "expert_pool_base_snapshot",
            )
            expert_sf_pool_base_snapshot = self._view_local(
                local_workspace, "expert_sf_pool_base_snapshot",
            )
            relay_pool_base_snapshot = self._view_local(
                local_workspace, "relay_pool_base_snapshot",
            )
            relay_sf_pool_base_snapshot = self._view_local(
                local_workspace, "relay_sf_pool_base_snapshot",
            )
            expert_scheduled_count_snapshot = self._view_local(
                local_workspace, "expert_scheduled_count_snapshot",
            )
            pool_end_snapshot = self._view_local(
                local_workspace, "pool_end_snapshot",
            )
            l1_arrival_count_snapshot = self._view_local(
                local_workspace, "l1_arrival_count_snapshot",
            )
        else:
            internode_send_count = None
            internode_recv_splits = None
            internode_send_token_count = None
            tokens_to_node = None
            token_node_slot = None
            intra_token_count = None
            internode_recv_count_per_source = None
            relay_fanout_reservation = None
            relay_fanout_slot = None
            expert_pool_base = None
            expert_sf_pool_base = None
            relay_pool_base = None
            relay_sf_pool_base = None
            relay_direct_bucket_meta = None
            relay_row_ready = None
            relay_sf_staging = None
            internode_recv_count_per_source_snapshot = None
            expert_recv_count_snapshot = None
            expert_recv_count_sum_snapshot = None
            intra_token_count_snapshot = None
            expert_pool_base_snapshot = None
            expert_sf_pool_base_snapshot = None
            relay_pool_base_snapshot = None
            relay_sf_pool_base_snapshot = None
            expert_scheduled_count_snapshot = None
            pool_end_snapshot = None
            l1_arrival_count_snapshot = None

        token_comm_args = ExtractedTokenCommArgs(
            input_token_buffer=activation,
            input_sf_buffer=activation_sf,
            topk_idx=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            internode_send_count=internode_send_count,
            internode_recv_splits=internode_recv_splits,
            internode_send_token_count=internode_send_token_count,
            tokens_to_node=tokens_to_node,
            token_node_slot=token_node_slot,
            intra_token_count=intra_token_count,
            internode_recv_count_per_source=internode_recv_count_per_source,
            relay_fanout_reservation=relay_fanout_reservation,
            relay_fanout_slot=relay_fanout_slot,
            expert_pool_base=expert_pool_base,
            expert_sf_pool_base=expert_sf_pool_base,
            relay_pool_base=relay_pool_base,
            relay_sf_pool_base=relay_sf_pool_base,
            relay_direct_bucket_meta=relay_direct_bucket_meta,
            relay_row_ready=relay_row_ready,
            relay_sf_staging=relay_sf_staging,
            internode_recv_count_per_source_snapshot=(
                internode_recv_count_per_source_snapshot
            ),
            expert_recv_count_snapshot=expert_recv_count_snapshot,
            expert_recv_count_sum_snapshot=expert_recv_count_sum_snapshot,
            intra_token_count_snapshot=intra_token_count_snapshot,
            expert_pool_base_snapshot=expert_pool_base_snapshot,
            expert_sf_pool_base_snapshot=expert_sf_pool_base_snapshot,
            relay_pool_base_snapshot=relay_pool_base_snapshot,
            relay_sf_pool_base_snapshot=relay_sf_pool_base_snapshot,
            expert_scheduled_count_snapshot=expert_scheduled_count_snapshot,
            pool_end_snapshot=pool_end_snapshot,
            l1_arrival_count_snapshot=l1_arrival_count_snapshot,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=l1_sf_buffer_i32,
            fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
            input_fc1_activation_per_token_scale_buffer=act_scale_input,
            dispatched_fc1_activation_per_token_scale_buffer=(
                l1_fc1_activation_per_token_scale_buffer
            ),
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            combine_output=combine_output_u8,
            combine_sf=combine_sf,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_output_sf=fc2_output_sf,
            fc2_done_counter=fc2_done_counter,
            token_back_schedule_counter=token_back_schedule_counter,
            nvlink_barrier_signal=nvlink_barrier_signal,
            nvlink_barrier_counter=nvlink_barrier_counter,
            internode_barrier_counter=internode_barrier_counter,
            combine_phase_snapshot=combine_phase_snapshot,
            combine_issue_counter=combine_issue_counter,
            tail_combine_output=(combine_target if self.tail_topk_reduce else None),
            tail_internode_combine=(
                gin_internode_combine_buffer if self.tail_topk_reduce else None
            ),
            tail_reduced_output=(output_activation if self.tail_topk_reduce else None),
            pipeline_timing=pipeline_timing,
            combine_timing_landmarks=combine_timing_landmarks,
            fc_tile_timeline=fc_tile_timeline,
            gin_dev_comm=gin_dev_comm,
            gin_window=gin_window,
            grid_sync_counter=grid_sync_counter,
            local_zero_prefix=local_zero_prefix,
            shared_zero_prefix=shared_zero_prefix,
            dispatch_route_extent=(
                self._view_local(local_workspace, "dispatch_route_extent")
                if cutlass.const_expr(self.max_tokens_per_rank <= 1024) else None
            ),
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=peer_rank_ptr_mapper_host.rank_idx,
            num_total_experts=self.num_total_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=self.sf_uint32_per_token,
            ranks_per_node=self.ranks_per_node,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            pool_token_capacity=self.pool_token_capacity,
            pool_sf_capacity=self.pool_sf_capacity,
            sm_count=sm_count,
            # Field 56 (diagnostic): gpu-scope done-counter publication.
            system_scope_done_flags=(
                self.enable_nccl_gin and not self.token_comm.fc2_done_gpu_scope
            ),
        )

        # C1 / C2 are tautological (token_padding_block == "block_m";
        # sf_padding_block == "sf_block_m") so the pool layout and the
        # sched cumulative-row offsets align by construction.
        #
        # ``combine_target`` is MoE-domain storage. separate_kernel_reduce uses
        # ``(max_tokens_per_rank, num_topk, hidden)`` (internal staging) and a
        # tail reduce; in_kernel_reduce uses ``(max_tokens_per_rank, 1, hidden)``
        # and reduces in kernel.  The epilogue return tile maps local pool rows
        # back to the source rank's token row through ``token_comm_args``.
        if cutlass.const_expr(self.fc2_stage_locally):
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_target = combine_target

        # Quantized combine loop is closed: the fc2 epilogue encodes the data
        # plane (STG-to-peer in epi_warps, or local staging in the dispatch
        # modes) and writes per-block scales to the rank-local SF plane
        # (fc2_output_sf); the token-back warps push that SF plane
        # token-contiguously to the peers' combine_sf (and the data plane too in
        # the dispatch modes); TopkReduce below dequantizes and reduces.
        super().__call__(
            activation=l1_token_buffer_nvfp4,
            fc1_weight=fc1_weight,
            activation_sf=l1_sf_buffer_fp8,
            fc1_weight_sf=fc1_weight_sf,
            fc1_output=fc1_output,
            fc1_output_sf=fc1_output_sf,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            fc2_output=fc2_output_target,
            topk_scores=l1_topk_weights_buffer,
            fc1_done_counter=fc1_done_counter,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            fc1_activation_per_token_scale=l1_fc1_activation_per_token_scale_buffer,
            offs=None,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            expert_token_sizes=expert_token_sizes,
            expert_pool_base=(
                expert_pool_base if self.enable_internode_relay else None
            ),
            expert_sf_pool_base=(
                expert_sf_pool_base if self.enable_internode_relay else None
            ),
            intra_token_sizes=(
                intra_token_count if self.enable_nccl_gin else None
            ),
            relay_pool_base=(
                relay_pool_base if self.enable_nccl_gin else None
            ),
            relay_sf_pool_base=(
                relay_sf_pool_base if self.enable_nccl_gin else None
            ),
            token_comm_args=token_comm_args,
        )

        # separate_kernel_reduce: collapse the per-topk ``combine_quant`` staging
        # into the public 2D output via the shared TopkReduce launcher (sized from
        # codegen-time hidden / num_topk / combine_format -- it dequantizes per
        # format and reduces over topk). Same stream, so it is ordered strictly
        # after the cross-rank combine writes landed (the mega kernel's nvlink
        # barrier guarantees all peer combine STGs to this rank completed before it
        # exits). Weighting follows the compute graph: deepgemm (apply_topk_in_fc1)
        # folded the routing weight into fc1 -> plain K-sum; transformers applies
        # topk_weights here.
        if cutlass.const_expr(
            not self.in_kernel_fc2_reduce and not self.tail_topk_reduce and not self.skip_topk_reduce
        ):
            if cutlass.const_expr(
                num_valid_tokens is not None and not self.topk_reduce_persistent
            ):
                raise ValueError(
                    "num_valid_tokens requires topk_reduce_persistent=True."
                )
            if cutlass.const_expr(
                num_valid_tokens is None and self.topk_reduce_persistent
            ):
                raise ValueError(
                    "topk_reduce_persistent=True requires num_valid_tokens."
                )
            score = (
                topk_weights if cutlass.const_expr(not self.apply_topk_in_fc1)
                else None
            )
            reduce = TopkReduce(
                self.hidden,
                self.num_topk,
                self.combine_format,
                sm_arch=get_cutedsl_target_arch(),
                num_experts=self.num_total_experts,
                persistent=self.topk_reduce_persistent,
                num_sms=self.topk_reduce_num_sms,
                num_remote_nodes=(
                    self.num_remote_nodes if self.enable_nccl_gin else 0
                ),
                ranks_per_node=self.ranks_per_node,
                num_experts_per_rank=self.num_experts_per_rank,
                combine_slot_indexed=self.enable_nccl_gin,
            )
            if cutlass.const_expr(self.enable_nccl_gin):
                reduce(combine_target, combine_sf, output_activation, score, stream,
                       internode_combine=gin_internode_combine_buffer,
                       combine_phase_snapshot=combine_phase_snapshot,
                       topk_idx=topk_idx, local_rank=peer_rank_ptr_mapper_host.rank_idx,
                       token_node_slot=token_node_slot)
            elif cutlass.const_expr(self.topk_reduce_persistent):
                reduce(
                    combine_target,
                    combine_sf,
                    output_activation,
                    score,
                    stream,
                    self._view_local(local_workspace, "topk_reduce_counters"),
                    num_tokens_dev=num_valid_tokens,
                    topk_indices=(
                        topk_idx
                        if cutlass.const_expr(not self.combine_format.is_quantized)
                        else None
                    ),
                )
            else:
                reduce(
                    combine_target,
                    combine_sf,
                    output_activation,
                    score,
                    stream,
                    topk_indices=(
                        topk_idx
                        if cutlass.const_expr(not self.combine_format.is_quantized)
                        else None
                    ),
                )

    # =========================================================================
    # TokenComm delegation surface consumed by the fc1/fc2 base kernel
    # =========================================================================

    def token_comm_extra_smem_storage_class(self) -> type:
        return self.token_comm.extra_smem_storage_class()

    def token_comm_hook_prepare_device_resources(self, token_comm_args):
        if cutlass.const_expr(self.enable_nccl_gin):
            from nccl.core.device import cute as nccl_cute

            # DevComm is a by-value kernel argument whose nccl4py pointer is
            # lazy. Materialize it here, while the argument dominates the full
            # kernel, then retain only its scalar GIN pointers and rail-team
            # value for nested JIT regions.
            dev_comm = token_comm_args.gin_dev_comm
            gin_gpu = dev_comm.gin(
                nccl_cute.GinBackendMask.GDAKI,
                context_id=0,
                resource_sharing_mode=nccl_cute.GinResourceSharingMode.GPU,
            )
            gin_thread = dev_comm.gin(
                nccl_cute.GinBackendMask.GDAKI,
                context_id=0,
                resource_sharing_mode=nccl_cute.GinResourceSharingMode.THREAD,
            )
            # Keep these entry-region scalar MLIR values on the compile-time
            # TokenComm object. Passing the native wrappers as @cute.jit
            # operands reconstructs their extractvalue operations outside the
            # defining region and violates dominance. Keeping a native-struct
            # wrapper and extracting it later inside a sibling JIT region also
            # violates dominance, whereas these entry-region values dominate
            # every consumer.
            self.token_comm._gin_gpu_ptr = gin_gpu.ptr
            self.token_comm._gin_thread_ptr = gin_thread.ptr
            self.token_comm._team_rail_value = (
                dev_comm.team_rail.__extract_mlir_values__()[0]
            )

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        return self.token_comm.fc1_ready_counter_ptr(token_comm_args)

    def token_comm_hook_fc1_ready_peek_scope(self) -> str:
        # Relay peers of the internode kernel publish fc1-ready credits at
        # system scope; with the master switch off every writer is a GPU-scope
        # release, so the peek keeps main's GPU-scope acquire.
        return "sys" if self.internode else "gpu"

    @cute.jit
    def token_comm_hook_cta_linear_id(self):
        return self.token_comm._cta_linear_id()

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        self.token_comm.sched_warp_pre_init_wait(token_comm_args)

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self, token_comm_args, work_tile_info,
    ):
        self.token_comm.fc1_tma_b_predispatch_spin(
            token_comm_args, work_tile_info,
        )

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
        ab_smem_ptr=None,
    ):
        if cutlass.const_expr(self.enable_internode_relay):
            if warp_idx < self.relay_warp_id[0]:
                self.token_comm.dispatch_warp_body(
                    token_comm_args,
                    token_comm_storage,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                    tidx=tidx,
                )
            else:
                self.token_comm.relay_warp_body(
                    token_comm_args,
                    token_comm_storage,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                    tidx=tidx,
                    wide_ring_smem=ab_smem_ptr,
                )
        else:
            self.token_comm.dispatch_warp_body(
                token_comm_args,
                token_comm_storage,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )

    @cute.jit
    def token_comm_hook_token_back_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        self.token_comm.token_back_warp_body(
            token_comm_args,
            token_comm_storage,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )

    @cute.jit
    def token_comm_hook_tail_reset_shared_counters(
        self,
        token_comm_args,
        *,
        cta_linear_id,
        local_warp_idx,
        lane_idx,
    ):
        self.token_comm.tail_reset_counters(
            token_comm_args,
            token_comm_args.shared_zero_prefix,
            cta_linear_id=cta_linear_id,
            local_warp_idx=local_warp_idx,
            lane_idx=lane_idx,
        )

    @cute.jit
    def token_comm_hook_kernel_tail(
        self,
        token_comm_args,
        token_comm_storage=None,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        self.token_comm.kernel_tail(
            token_comm_args,
            token_comm_storage,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )

    @cute.jit
    def token_comm_hook_epilogue_exit(
        self,
        token_comm_args,
        token_comm_storage=None,
        *,
        warp_idx,
        lane_idx,
    ):
        self.token_comm.epilogue_exit(
            token_comm_args,
            token_comm_storage,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
        )

    def _setup_attributes(self) -> None:
        super()._setup_attributes()
        # Host-time constant for the combine ring: the AB stage layouts exist
        # only after the base setup, the value must be fixed before the kernel
        # body is traced, and cute.size_in_bytes may only run inside this
        # tracing context, so the result is cached for later host readers.
        self._combine_wide_ring_depth = self._compute_combine_wide_ring_rounds()
        self.token_comm.combine_wide_ring_rounds = self._combine_wide_ring_depth

    def _combine_wide_ring_rounds(self) -> int:
        """Cached combine ring depth; 0 before setup or when the knob is off."""
        return int(getattr(self, "_combine_wide_ring_depth", 0))

    def _compute_combine_wide_ring_rounds(self) -> int:
        """Ring depth in 2 KB rounds per relay warp for the combine ring.

        The ring reuses this CTA's AB stage buffers after epilogue warp 0 has
        consumed its last accumulator, so it requires a (1, 1) cluster: with
        multicast or CTA-pair UMMA a peer CTA still reads or writes those
        buffers. The depth is the largest power of two whose num_relay_warps
        rings fit in the AB stage bytes.
        """
        if not self.enable_nccl_gin:
            return 0
        if not hasattr(self, "a_smem_layout_staged"):
            # Before _setup_attributes (kernel-name construction): the depth is
            # a pure function of the tile configuration, so the name carries
            # the knob only.
            return 0
        if tuple(self.cluster_shape_mn) != (1, 1):
            raise ValueError(
                "the combine deep-prefetch ring requires a (1, 1) cluster; the "
                "combine ring reuses this CTA's AB stage buffers"
            )
        ab_bytes = (
            cute.size_in_bytes(self.a_dtype, self.a_smem_layout_staged)
            + cute.size_in_bytes(self.b_dtype, self.b_smem_layout_staged)
            + cute.size_in_bytes(self.sf_dtype, self.sfa_smem_layout_staged)
            + cute.size_in_bytes(self.sf_dtype, self.sfb_smem_layout_staged)
        )
        round_bytes = (
            self.token_comm.num_topk * 16 * self.token_comm.warp_threads
        )
        rounds = ab_bytes // (round_bytes * len(self.relay_warp_id))
        if rounds < 2:
            # One buffer is always being refilled while another is consumed.
            return 0
        depth = 1
        while depth * 2 <= rounds:
            depth *= 2
        return depth
