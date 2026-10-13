# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MegaMoE fused dispatch + fc1 + fc2 + combine kernel (BF16).

Parallel to the NVFP4 swap-AB MegaMoE kernel and shared by the SM90 BF16
non-swap and swap-AB fused fc1+fc2 bases.
The token-communication machinery (dispatch prep / barrier / pull, NVLink
barrier, kernel tail) is reused verbatim from
``src/token_comm.py`` via the ``TokenInPullTokenBackPush`` helper; only the
data-format-dependent workspace layout differs:

  - ``hidden_bytes = hidden * 2``              (bf16 = 2 bytes/element)
  - ``fc1_output`` region dtype = ``ab_dtype``  (BFloat16)
  - no scale-factor planes: ``sf_uint32_per_token = 0`` disables every
    dispatch SF pull pass (the constexpr ``sf_passes`` loops fold away)
  - ``Fp8GateUpInterleave = 8``                (FC1 gate/up layout)

BF16 operands need no dequantization: the epilogues consume the WGMMA FP32
accumulators directly.

The caller only provides the final ``output_activation`` with shape
``(max_tokens_per_rank, hidden)``.  The pre-topk-reduce plane is internal.
Following the NVFP4 mega kernel, the combine surface is two orthogonal knobs:

``token_back_mode`` -- who performs the cross-rank fc2 write-back:

  * ``epi_warps``: the fc2 epilogue STGs straight to the source rank.
  * ``reuse_dispatch_warps``: the epilogue stages rows in the local
    ``fc2_output_workspace`` pool; dispatch warps bulk-push them after pull.
  * ``standalone_warps``: same staging, pushed by four dedicated warps.

``fc2_in_kernel_topk_reduce`` -- where the topk axis collapses:

  * False (**separate reduce**): writes land in the internal
    ``unreduced_expert_output[src_token, src_topk, :]`` shared-symmetric staging, then
    the shared ``TopkReduce`` kernel collapses topk into the public output.
  * True (**in-kernel reduce**): writes accumulate directly into a
    ``(max_tokens_per_rank, 1, hidden)`` view of the public output --
    ``epi_warps`` issues ``red.relaxed.sys.global.add.noftz.v2.bf16x2``, the
    dispatch modes push with ``cp.reduce.async.bulk...add.noftz.bf16``
    (``token_back_reduce_topk``); no reducer kernel runs.

``static_expert_shape`` is required because dispatch storage and pool sizes are
codegen-time quantities.
"""

# NOTE: ``from __future__ import annotations`` is intentionally NOT used here
# (PEP 563 string-ifies class-body annotations, which breaks ``@cute.struct``
# element-type introspection).  See moe_nvfp4_swapab/megamoe_kernel.py.

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64, Int32

from common.host_utils import get_cutedsl_target_arch

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from src.iket_compat import iket

from moe_hopper_bf16.kernel_bf16_glu_fc12 import (
    Sm90SwigluBf16Fc12Kernel,
)
from moe_hopper_bf16.kernel_bf16_glu_fc12_swapab import (
    Sm90SwapABSwigluBf16Fc12Kernel,
)
from moe_nvfp4_swapab.moe_utils import spin_wait
from moe_nvfp4_swapab.topk_reduce import TopkReduce
from src.token_comm import (
    CombineFormat,
    TokenCommArgs as ExtractedTokenCommArgs,
    TokenInPullTokenBackPush,
)

# Reuse the region-layout helpers + module constants from the NVFP4 mega kernel
# so the two paths stay byte-for-byte consistent in their workspace plumbing.
from moe_nvfp4_swapab.megamoe_kernel import (
    _RegionSpec,
    _round_up,
    _layout_regions,
    _DispatchWarpCount,
    _TokenMetadataBytes,
    _GridSyncSlotCount,
    _NvlinkSlotCount,
)

# Int32 slots reserved for the never-dereferenced scale-factor placeholder
# region (16 bytes, the minimum region alignment used by the layout helper).
_SfPlaceholderInt32Count = 4


# =============================================================================
# Sm90MegaMoEBf16Kernel
# =============================================================================


class Sm90MegaMoEBf16Kernel(Sm90SwigluBf16Fc12Kernel):
    """MegaMoE-complete fused dispatch + fc1 + fc2 + combine kernel."""

    def __init__(
        self,
        # Base-class kwargs (forwarded 1:1 to ``super().__init__``).
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        group_hint: int,
        token_padding_block: int,
        load_balance_mode: str = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        ab_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        pingpong: bool = False,
        scenario: str = "2Dx3D",
        # MegaMoE-specific independent constants.
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        fc2_in_kernel_topk_reduce: bool = False,
        apply_topk_in_fc1: bool = True,
        token_back_mode: Literal[
            "epi_warps", "standalone_warps", "reuse_dispatch_warps"
        ] = "epi_warps",
        epi_flag_batch: Union[int, Tuple[int, int]] = 1,
        flag_batch: int = 1,
        gate_up_clamp: Optional[float] = None,
        fc1_store_offload: bool = True,
        fc1_early_done_publish: bool = False,
        active_dispatch_warps: int = 1,
        fold_producer_warps: bool = True,
        # Training forward: also emit the raw pre-SwiGLU FC1 gate+up
        # accumulator into the ``fc1_c`` tensor passed to ``__call__``.
        generate_c: bool = False,
        # Tail-split pair tasks (see fc1_fc2_fuse_sched); needs a 2-CTA token cluster.
        tail_split_pairs: bool = False,
    ) -> None:
        # Folding TMA-A / TMA-B / scheduler into the idle dispatch slots is
        # only possible with a single active dispatch warp.  Without the
        # epi_aux warp there is no FC1 store server, so the offload is
        # replaced by in-epilogue early fc1_done publication.
        _fold = bool(fold_producer_warps) and active_dispatch_warps == 1
        if _fold:
            fc1_store_offload = False
            fc1_early_done_publish = True
        # No SF planes survive on the BF16 path, but the fused-fc12 scheduler
        # and ``TokenInPullTokenBackPush`` both divide by ``sf_padding_block``
        # and reject non-positive values.  Mirror the token axis so every
        # SF-derived bound stays well formed and inert.
        sf_padding_block = token_padding_block

        if static_expert_shape is None:
            raise NotImplementedError(
                "Sm90MegaMoEBf16Kernel requires "
                "static_expert_shape != None (dynamic-shape MegaMoE is not wired)."
            )
        if hidden != static_expert_shape[2]:
            raise ValueError(
                f"hidden ({hidden}) must equal "
                f"static_expert_shape[2] ({static_expert_shape[2]})."
            )
        if token_back_mode not in (
            "epi_warps", "standalone_warps", "reuse_dispatch_warps",
        ):
            raise ValueError(f"unsupported token_back_mode '{token_back_mode}'")
        # NVFP4 naming: any non-epi mode stages fc2 rows locally and pushes
        # them back from the dispatch warp area.
        token_back_by_dispatch = token_back_mode != "epi_warps"
        if fc2_in_kernel_topk_reduce and not apply_topk_in_fc1:
            raise ValueError(
                "fc2_in_kernel_topk_reduce requires apply_topk_in_fc1=True; "
                "the in-kernel reduce can only atomic-add terms whose topk "
                "score was already absorbed before fc2."
            )

        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            use_2cta_instrs=use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            sf_padding_block=sf_padding_block,
            load_balance_mode=load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=force_static_sched,
            clc_bundle_size=clc_bundle_size,
            num_sched_stages=num_sched_stages,
            acc_dtype=acc_dtype,
            ab_dtype=ab_dtype,
            pingpong=pingpong,
            scenario=scenario,
            fc2_in_kernel_topk_reduce=fc2_in_kernel_topk_reduce,
            apply_topk_in_fc1=apply_topk_in_fc1,
            token_back_by_dispatch=token_back_by_dispatch,
            fc1_store_offload=fc1_store_offload,
            fc1_early_done_publish=fc1_early_done_publish,
            epi_flag_batch=epi_flag_batch,
            gate_up_clamp=gate_up_clamp,
            generate_c=generate_c,
            tail_split_pairs=tail_split_pairs,
        )

        self.enable_token_comm = True
        self.fold_producer_warps = _fold
        self.token_back_standalone = token_back_mode == "standalone_warps"
        self._apply_mega_warp_layout()
        # Keep the established swap-AB and non-swap N=256 budgets. Non-swap
        # N=128 has one epilogue warpgroup and can use the architectural
        # setmaxnreg maximum without approaching the CTA budget.
        self.epi_reg_cnt = 200 if self.token_back_standalone else 216
        if (
            not getattr(self, "is_swap_ab", False)
            and self.wgmma_n_splits == 1
            and not self.pingpong
        ):
            self.epi_reg_cnt = 256
        self.token_back_reg_cnt = 32
        self.fit_epi_registers()
        self.fit_fc1_offload_registers()
        self.validate_register_policy()

        # Independent MegaMoE-specific constants.
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden
        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.combine_format = CombineFormat.parse("bf16")

        # static_expert_shape = (num_experts_per_rank, intermediate_gateup, hidden).
        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_gateup = static_expert_shape[1]
        self.intermediate_downproj = self.intermediate_gateup // 2

        # BF16: 16 bits/elem = 2 bytes/element.  Kept consistent with the
        # ``hidden * fc1_token_dtype.width // 8`` that token_comm derives.
        self.hidden_bytes = self.hidden * self.ab_dtype.width // 8
        # No scale metadata travels with a BF16 token.  ``sf_passes`` in
        # ``TokenInPullTokenBackPush.dispatch_pull`` is
        # ``(sf_uint32_per_token + 31) // 32``, so zero removes every SF
        # constexpr pass; the SF buffers are then only used to take an
        # address that is never dereferenced.
        self.sf_uint32_per_token = 0
        # Cross-rank totals: per-rank count * world_size.
        self.num_total_experts = world_size * self.num_experts_per_rank

        is_swap_ab = getattr(self, "is_swap_ab", False)
        self.cluster_tile_tokens = (
            self.mma_tiler_mnk[1] * cluster_shape_mnk[1]
            if is_swap_ab
            else self.mma_tiler_mnk[0] * cluster_shape_mnk[0]
        )

        # Cache region sizing inputs used by workspace layout and __call__.
        (
            self.pool_token_capacity,
            self.pool_sf_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()

        # Cohabiting warps before the dispatch group: one or two
        # epilogue/WGMMA warpgroups plus TMA-A/TMA-B/scheduler and the empty
        # old-MMA warp.
        # Non-dispatch, non-token-back warps.  When the producer roles are
        # folded into the dispatch warpgroup they are already counted by
        # TokenComm's num_dispatch_warps, so only the epilogue remains.
        num_other_warps = len(self.epilogue_warp_id) + (
            0 if self.fold_producer_warps else 4
        )

        # For token_back_by_dispatch, the dispatch warp pushes fc2 results
        # from the local pool workspace back to each source rank's internal
        # combine target.
        # Count every FC2 CTA publish in one token cluster tile. Hidden tiles
        # are rounded to complete hidden clusters, and every CTA along the
        # token-cluster axis independently publishes its output rows.
        if token_back_by_dispatch:
            if is_swap_ab:
                hidden_tile = self.mma_tiler_mnk[0]
                hidden_cluster = cluster_shape_mnk[0]
                token_cluster = cluster_shape_mnk[1]
            else:
                hidden_tile = self.mma_tiler_mnk[1]
                hidden_cluster = cluster_shape_mnk[1]
                token_cluster = cluster_shape_mnk[0]
            fc2_publishes = (
                (
                    self.hidden + hidden_tile * hidden_cluster - 1
                )
                // (hidden_tile * hidden_cluster)
                * hidden_cluster
                * token_cluster
            )
        else:
            fc2_publishes = 0
        # Tail-split pair tasks: a split tail block publishes fc2_done
        # token_cluster * ceil(W2/2) times; the walker needs the CTA tile size.
        fc2_publishes_split_tail = 0
        cta_tile_tokens = None
        if token_back_by_dispatch and tail_split_pairs:
            if token_cluster != 2 or hidden_cluster != 1:
                raise ValueError(
                    "tail_split_pairs requires a token-side cluster of 2 and a "
                    f"weight-side cluster of 1; got cluster_shape_mnk="
                    f"{tuple(cluster_shape_mnk)} (swap_ab={is_swap_ab})."
                )
            hidden_tiles = (self.hidden + hidden_tile - 1) // hidden_tile
            fc2_publishes_split_tail = token_cluster * ((hidden_tiles + 1) // 2)
            cta_tile_tokens = self.cluster_tile_tokens // token_cluster

        self.token_comm = TokenInPullTokenBackPush(
            world_size=self.world_size,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=self.ab_dtype,
            token_back_by_dispatch=token_back_by_dispatch,
            fc2_publishes_per_token_cluster_tile=fc2_publishes,
            fc2_publishes_per_split_tail_tile=fc2_publishes_split_tail,
            cta_tile_tokens=cta_tile_tokens,
            token_back_reduce_topk=(
                token_back_by_dispatch and fc2_in_kernel_topk_reduce
            ),
            token_back_standalone=self.token_back_standalone,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=self.dispatch_warp_id[0],
            num_other_warps=num_other_warps,
            is_swap_ab=is_swap_ab,
            sf_atom_swizzled=True,
            flag_batch=flag_batch,
            active_dispatch_warps=active_dispatch_warps,
            # Only the active dispatch warps get a pull-buffer slot; the
            # idle slots would otherwise cost 3 * hidden_bytes of AB-stage
            # SMEM per CTA under 2-byte tokens.
            compact_pull_buffer=True,
        )

        # Region layout (same call drives both get_workspace_sizes() and the
        # __call__ partition).
        self._local_region_specs = self._build_local_region_specs()
        self._shared_region_specs = self._build_shared_region_specs()
        self._local_offsets, self._local_total = _layout_regions(
            self._local_region_specs
        )
        self._shared_offsets, self._shared_total = _layout_regions(
            self._shared_region_specs
        )
        self._local_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r for r in self._local_region_specs
        }
        self._shared_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r for r in self._shared_region_specs
        }
        local_leading = self._local_offsets["l1_token_buffer"]
        shared_leading = self._shared_offsets["src_token_topk_idx"]
        self.local_zero_i32_count = local_leading // 4
        self.shared_zero_i32_count = shared_leading // 4

    def sched_ext_fc1_peek_threshold(self) -> int:
        # Peek threshold must match the spin threshold (physical token-N tile)
        # so an early peek hit does not skip the spin and expose stale pool rows.
        return self.cluster_tile_tokens

    # =========================================================================
    # SMEM budget hook (base override)
    # =========================================================================

    def _dispatch_smem_bytes(self) -> int:
        """SMEM for dispatch pull mbarriers, expert scratch, and token buffer.

        Must match ``TokenInPullTokenBackPush.extra_smem_storage_class``:
        ``pull_mbar[Int64, 4] + smem_expert_count[Int32, num_total_experts]
        + pull_buffer[Uint8, token_comm.pull_buffer_bytes()]`` (one slot per
        *active* dispatch warp, see ``compact_pull_buffer``).
        Standalone token-back adds ``tb_pull_mbar[Int64, 4]`` and
        ``tb_pull_buffer[Uint8, 4 * tb_chunk_bytes]``.
        """
        pull_mbar_bytes = _DispatchWarpCount * 8
        expert_count_bytes = self.num_total_experts * 4
        pull_buffer_bytes = self.token_comm.pull_buffer_bytes()
        total = (
            _round_up(pull_mbar_bytes, 16)
            + _round_up(expert_count_bytes, 16)
            + _round_up(pull_buffer_bytes, 128)
        )
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
    # Pool sizing (first-principles; identical to the NVFP4 path)
    # =========================================================================

    def _pool_shapes(self) -> Tuple[int, int, int]:
        world_size = self.world_size
        max_tokens_per_rank = self.max_tokens_per_rank
        num_topk = self.num_topk
        num_experts_per_rank = self.num_experts_per_rank
        token_padding_block = self.token_padding_block
        sf_padding_block = self.sf_padding_block
        cluster_tile_tokens = self.cluster_tile_tokens

        max_recv = world_size * max_tokens_per_rank
        max_per_token = min(num_topk, num_experts_per_rank)
        raw = (
            max_recv * max_per_token
            + num_experts_per_rank * (token_padding_block - 1)
        )
        pool_token_capacity = _round_up(raw, token_padding_block)
        pool_sf_capacity = (
            (pool_token_capacity // token_padding_block) * sf_padding_block
        )
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
        pool_token_capacity = self.pool_token_capacity
        pool_task_tile_capacity = self.pool_task_tile_capacity
        num_experts_per_rank = self.num_experts_per_rank
        num_total_experts = self.num_total_experts
        hidden_bytes = self.hidden_bytes
        intermediate_downproj = self.intermediate_downproj

        fc1_done_slots = (
            (pool_token_capacity + self.token_tile_size - 1)
            // self.token_tile_size
            + num_experts_per_rank
        )

        # Accumulating counters are front-placed so kernel_tail can reset them
        # as one contiguous Int32 prefix before the next launch.
        specs: List[_RegionSpec] = [
            _RegionSpec(
                "l1_arrival_count",
                cutlass.Int32,
                (pool_task_tile_capacity,),
                16,
            ),
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

        if self.token_back_by_dispatch:
            specs.append(
                _RegionSpec(
                    "fc2_done_counter",
                    cutlass.Int32,
                    (num_experts_per_rank,),
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

        # Data buffers start at l1_token_buffer. The persistent NVLink phase
        # counter is intentionally after the reset prefix.
        specs += [
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (pool_token_capacity, hidden_bytes),
                128,
            ),
            _RegionSpec(
                "nvlink_barrier_counter",
                cutlass.Int32,
                (1,),
                16,
            ),
            # BF16 carries no per-token scale factors.  This 16-byte stub
            # only exists so ``TokenCommArgs`` has a valid (never
            # dereferenced) SF tensor to bind; see ``sf_uint32_per_token=0``.
            _RegionSpec(
                "l1_sf_buffer",
                cutlass.Int32,
                (_SfPlaceholderInt32Count,),
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
                self.ab_dtype,
                (pool_token_capacity, intermediate_downproj),
                128,
            ),
        ]

        if self.token_back_by_dispatch:
            specs.append(
                _RegionSpec(
                    "fc2_output_workspace",
                    cutlass.BFloat16,
                    (pool_token_capacity, 1, self.hidden),
                    128,
                )
            )

        return specs

    def _build_shared_region_specs(self) -> List[_RegionSpec]:
        world_size = self.world_size
        num_topk = self.num_topk
        max_tokens_per_rank = self.max_tokens_per_rank
        num_experts_per_rank = self.num_experts_per_rank

        max_slot = max_tokens_per_rank * num_topk

        specs = [
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

        # The per-topk FC2 plane is an implementation workspace, not public IO.
        # It is the cross-rank STG/TMA target and therefore belongs in the shared
        # symmetric workspace. In-kernel reduce accumulates directly into
        # output_activation (REDG from epi_warps, or bulk reduce push from the
        # dispatch token-back modes) and needs no staging.
        if not self.fc2_in_kernel_topk_reduce:
            specs.append(
                _RegionSpec(
                    "unreduced_expert_output",
                    self.combine_format.act_dtype,
                    (max_tokens_per_rank, num_topk, self.hidden),
                    128,
                )
            )
        return specs

    # =========================================================================
    # Public: workspace size query
    # =========================================================================

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return ``(local_ws_bytes, shared_ws_bytes)``."""
        return self._local_total, self._shared_total

    # =========================================================================
    # Workspace partition helpers (mirror the NVFP4 mega kernel)
    # =========================================================================

    @staticmethod
    def _make_typed_view(
        byte_workspace: cute.Pointer,
        byte_offset: int,
        cute_dtype: Any,
        shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]],
        assumed_align: int,
    ) -> cute.Tensor:
        """Build a typed view at a 64-bit byte offset from an opaque base."""
        byte_ptr = byte_workspace + Int64(byte_offset)
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

    @cute.jit
    def __call__(
        self,
        # ABI notation: T=tokens, E=local experts, H=hidden, and I=the
        # down-projection width. FC1 produces the gate/up width 2I.
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,           # (T, hidden) BF16
        topk_idx: cute.Tensor,             # (T, num_topk) Int64
        topk_weights: cute.Tensor,         # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,            # (E, H, 2I) BF16
        fc2_weight: cute.Tensor,            # (E, I, H) BF16
        # Raw pre-SwiGLU fc1 gate+up output (generate_c / training path).
        # (pool_rows, intermediate_gateup) BF16; same padded per-expert pool
        # row space as the internal ``fc1_output`` workspace.  Pass None (it
        # is never referenced) when ``generate_c=False``.  Positioned as in
        # the Blackwell training kernels: after the weights, before the output.
        fc1_c: Optional[cute.Tensor],
        # Final combined output consumed by the caller.
        output_activation: cute.Tensor,    # (T, hidden) BF16
        # Opaque workspaces.
        local_workspace: cute.Pointer,     # uint8 gmem base of local_ws_bytes
        shared_workspace: cute.Pointer,    # uint8 gmem base of shared_ws_bytes
        # Runtime host payload; packed into ``SymBuffer{world_size}``.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Launch the BF16 MegaMoE-complete fused kernel.

        Pointer-mapping contract mirrors the NVFP4 path:
          * ``activation`` / ``topk_weights`` MUST point into memory
            reachable via ``peer_rank_ptr_mapper.ptr_map_to_rank(...)``
            (NVSHMEM symmetric heap).  Single-rank degenerate runs are
            allowed.
          * ``topk_idx`` is read on the local rank only.
          * ``fc1_weight`` / ``fc2_weight`` are local-only.

          * Under in-kernel reduce (``fc2_in_kernel_topk_reduce``),
            ``output_activation`` is the cross-rank accumulate target (REDG or
            bulk reduce push) and must also be peer reachable. Under separate
            reduce, peer writes target the internal ``unreduced_expert_output``
            shared-workspace region and ``output_activation`` may be
            rank-local memory.
        """
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        hidden = self.hidden

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + BF16 view
        # (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_local(local_workspace, "l1_token_buffer")
        l1_token_buffer_ab = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            self.ab_dtype,
            (pool_token_capacity, hidden),
            (hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )

        # SF placeholder: bound to both the peer-side and pool-side SF slots
        # of ``TokenCommArgs``.  With ``sf_uint32_per_token = 0`` neither is
        # ever loaded or stored; only the base address is taken.
        l1_sf_placeholder = self._view_local(local_workspace, "l1_sf_buffer")

        l1_topk_weights_buffer = self._view_local(
            local_workspace, "l1_topk_weights_buffer",
        )
        l1_arrival_count = self._view_local(local_workspace, "l1_arrival_count")
        # token_src_metadata storage = (pool_token_capacity, TokenSrcMetadata.nbytes) Uint8;
        # dispatch_pull writes one packed Int64 per pool token row (see TokenSrcMetadata).
        token_src_metadata = self._view_local(
            local_workspace, "token_src_metadata",
        )
        expert_send_count = self._view_local(local_workspace, "expert_send_count")
        grid_sync_counter = self._view_local(local_workspace, "grid_sync_counter")
        nvlink_barrier_counter = self._view_local(
            local_workspace, "nvlink_barrier_counter",
        )
        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_done_counter = self._view_local(local_workspace, "fc1_done_counter")

        load_balance_counter: Optional[cute.Tensor] = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local(
                local_workspace, "load_balance_counter",
            )

        # MoE-domain cross-rank combine target. Separate reduce stages one
        # result per (token, topk) in workspace; in-kernel reduce aliases the
        # public 2D output because writers collapse topk on the fly (epi_warps
        # REDG, or the dispatch modes' cp.reduce bulk push).
        if cutlass.const_expr(self.fc2_in_kernel_topk_reduce):
            combine_target = cute.make_tensor(
                output_activation.iterator,
                cute.make_layout(
                    (self.max_tokens_per_rank, 1, hidden),
                    stride=(hidden, hidden, 1),
                ),
            )
        else:
            combine_target = self._view_shared(
                shared_workspace, "unreduced_expert_output",
            )

        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_workspace_native = self._view_local(
                local_workspace, "fc2_output_workspace",
            )
            fc2_output_workspace_u8 = self._make_typed_view(
                local_workspace,
                self._local_offsets["fc2_output_workspace"],
                cutlass.Uint8,
                (pool_token_capacity * hidden * 2,),
                None,
                self._local_region_by_name["fc2_output_workspace"].align,
            )
            fc2_done_counter = self._view_local(local_workspace, "fc2_done_counter")
            combine_output_comm = cute.recast_tensor(
                combine_target, cutlass.Uint8,
            )
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_workspace_native = None
            fc2_output_workspace_u8 = None
            fc2_done_counter = None
            combine_output_comm = combine_target
            fc2_output_target = combine_target

        # Shared regions.
        src_token_topk_idx = self._view_shared(
            shared_workspace, "src_token_topk_idx",
        )
        expert_recv_count = self._view_shared(shared_workspace, "expert_recv_count")
        expert_recv_count_sum = self._view_shared(
            shared_workspace, "expert_recv_count_sum",
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace, "nvlink_barrier_signal",
        )

        # i32 stride=(2,) view onto the i64 ``expert_recv_count_sum`` buffer --
        # low32 bits hold per-expert total token count after _dispatch_barrier;
        # zero-copy alias for sizes-mode scheduling.
        expert_token_sizes = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank,),
            stride=(2,),
        )
        local_zero_prefix = self._make_typed_view(
            local_workspace,
            0,
            cutlass.Int32,
            (self.local_zero_i32_count,),
            (1,),
            16,
        )
        shared_zero_prefix = self._make_typed_view(
            shared_workspace,
            0,
            cutlass.Int32,
            (self.shared_zero_i32_count,),
            (1,),
            16,
        )

        token_comm_args = ExtractedTokenCommArgs(
            input_token_buffer=activation,
            input_sf_buffer=l1_sf_placeholder,
            topk_idx=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=l1_sf_placeholder,
            fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            combine_output=combine_output_comm,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_done_counter=fc2_done_counter,
            nvlink_barrier_signal=nvlink_barrier_signal,
            nvlink_barrier_counter=nvlink_barrier_counter,
            grid_sync_counter=grid_sync_counter,
            local_zero_prefix=local_zero_prefix,
            shared_zero_prefix=shared_zero_prefix,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=peer_rank_ptr_mapper_host.rank_idx,
            num_total_experts=self.num_total_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            sm_count=sm_count,
        )

        _fc12_kwargs = dict(
            activation=l1_token_buffer_ab,
            fc1_weight=fc1_weight,
            fc1_output=fc1_output,
            fc2_weight=fc2_weight,
            fc2_output=fc2_output_target,
            topk_scores=l1_topk_weights_buffer,
            fc1_done_counter=fc1_done_counter,
            offs=None,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            expert_token_sizes=expert_token_sizes,
            token_comm_args=token_comm_args,
        )
        if cutlass.const_expr(self.generate_c):
            _fc12_kwargs["fc1_c"] = fc1_c
        if cutlass.const_expr(getattr(self, "is_swap_ab", False)):
            Sm90SwapABSwigluBf16Fc12Kernel.__call__(self, **_fc12_kwargs)
        else:
            Sm90SwigluBf16Fc12Kernel.__call__(self, **_fc12_kwargs)

        # Match the NVFP4/MXFP8 compute graphs: deepgemm folds routing weights
        # into the SwiGLU output before the FC1-output store, while the
        # transformers graph leaves each term unweighted and applies scores in
        # this standalone reducer.
        if cutlass.const_expr(not self.fc2_in_kernel_topk_reduce):
            score = (
                topk_weights if cutlass.const_expr(not self.apply_topk_in_fc1)
                else None
            )
            TopkReduce(
                self.hidden,
                self.num_topk,
                self.combine_format,
                sm_arch=get_cutedsl_target_arch(),
            )(
                combine_target,
                None,
                output_activation,
                score,
                stream,
            )

    # =========================================================================
    # TokenComm delegation surface consumed by the fc1/fc2 base kernel
    # =========================================================================

    def token_comm_extra_smem_storage_class(self) -> type:
        return self.token_comm.extra_smem_storage_class()

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        return self.token_comm.fc1_ready_counter_ptr(token_comm_args)

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
    ):
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
        self.token_comm.tail_reset_shared_counters(
            token_comm_args,
            cta_linear_id=cta_linear_id,
            local_warp_idx=local_warp_idx,
            lane_idx=lane_idx,
        )

    @cute.jit
    def token_comm_hook_kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        self.token_comm.kernel_tail(
            token_comm_args,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )


class Sm90MegaMoESwapABBf16Kernel(
    Sm90MegaMoEBf16Kernel,
    Sm90SwapABSwigluBf16Fc12Kernel,
):
    """MegaMoE wiring that reuses token communication with the swap-AB base."""

    pass
