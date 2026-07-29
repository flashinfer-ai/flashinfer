# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MegaMoE fused dispatch + fc1 + fc2 + combine kernel (BF16, non-swap-AB).

Extends the BF16 fused fc1+fc2 base (``Sm100SwigluBf16Fc12Kernel``) with the
token-communication machinery (dispatch prep / barrier / pull, NVLink barrier,
kernel tail) reused verbatim from ``src/token_comm.py`` via the
``TokenInPullTokenBackPush`` helper.  Data-format-dependent quantities:

  - ``hidden_bytes = hidden * 2``              (BF16 = 2 bytes/element)
  - ``fc1_output`` region dtype = ``ab_dtype``  (BFloat16)
  - ``fc1_tma_b_predispatch_spin`` in ``token_comm.py`` selects the
    non-swap-AB counter path (``fc1_token_dtype.width != 4``).

Three fc2 output modes are supported:

  * **Form A** (default): epilogue ``Fc2OutputDest`` STGs to
    ``combine_output[src_token, src_topk, :]``; host reduces the topk axis.
    ``combine_output`` shape ``(max_tokens_per_rank, num_topk, hidden)``.
  * **token_back_by_dispatch**: epilogue STGs to a local fc2 pool workspace;
    dispatch (or standalone token-back) warps push results back to source
    ranks' ``combine_output`` via TMA.
    ``combine_output`` shape ``(max_tokens_per_rank, num_topk, hidden)``.
  * **Form B** (``fc2_in_kernel_topk_reduce``): the kernel reduces the topk
    axis into ``combine_output[src_token, 0, :]`` (rows are already
    topk-weighted by the fc1 epilogue, so element-wise adds sum the topk
    contributions).  ``combine_output`` shape ``(max_tokens_per_rank, 1,
    hidden)``.  Two carriers, selected by ``token_back_mode``:
    ``epi_warps`` -> the epilogue issues ``red.relaxed.sys.global.add`` to
    the peer combine row; a token-back mode -> the push reduces via
    ``cp.reduce.async.bulk.add.noftz.bf16`` from the local pool workspace.

``static_expert_shape`` is required because dispatch storage and pool sizes are
codegen-time quantities.
"""

# NOTE: ``from __future__ import annotations`` is intentionally NOT used here
# (PEP 563 string-ifies class-body annotations, which breaks ``@cute.struct``
# element-type introspection).  See moe_nvfp4_swapab/megamoe_kernel.py.

from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64, Int32

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from src.iket_compat import iket

from moe_bf16_glu.kernel_bf16_glu_fc12 import Sm100SwigluBf16Fc12Kernel
from src.token_comm import (
    CombineFormat,
    TokenCommArgs as ExtractedTokenCommArgs,
    TokenInPullTokenBackPush,
)

# Region-layout helpers + module constants shared across the mega kernels'
# workspace plumbing.
from moe_nvfp4_swapab.megamoe_kernel import (
    _RegionSpec,
    _round_up,
    _layout_regions,
    _DispatchWarpCount,
    _TokenMetadataBytes,
    _GridSyncSlotCount,
    _NvlinkSlotCount,
)


# =============================================================================
# Sm100MegaMoEBf16Kernel
# =============================================================================


class Sm100MegaMoEBf16Kernel(Sm100SwigluBf16Fc12Kernel):
    """MegaMoE-complete fused dispatch + fc1 + fc2 + combine kernel (BF16)."""

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
        scenario: str = "2Dx3D",
        # MegaMoE-specific independent constants.
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        token_back_mode: Literal[
            "epi_warps", "standalone_warps", "reuse_dispatch_warps"
        ] = "epi_warps",
        epi_flag_batch: Tuple[int, int] = (1, 1),
        flag_batch: int = 1,
        gate_up_clamp: Optional[float] = None,
        apply_topk_in_fc1: bool = False,
        generate_c: bool = False,
        use_stg_fc1: bool = False,
    ) -> None:
        if static_expert_shape is None:
            raise NotImplementedError(
                "Sm100MegaMoEBf16Kernel requires static_expert_shape != None "
                "(dynamic-shape MegaMoE is not wired)."
            )
        if hidden != static_expert_shape[2]:
            raise ValueError(
                f"hidden ({hidden}) must equal "
                f"static_expert_shape[2] ({static_expert_shape[2]})."
            )
        # dispatch_prep assigns ``32 // num_topk`` tokens per warp pass; that
        # count must stay >= 1 or the routing-count loops process no tokens.
        if not (1 <= num_topk <= 32):
            raise ValueError(
                f"num_topk must be in [1, 32]; got {num_topk}."
            )
        if fc2_in_kernel_topk_reduce and not apply_topk_in_fc1:
            raise ValueError(
                "fc2_in_kernel_topk_reduce requires apply_topk_in_fc1=True; "
                "the in-kernel reduction collapses the topk axis before a "
                "separate reducer can apply the routing weights."
            )
        if token_back_mode not in (
            "epi_warps", "standalone_warps", "reuse_dispatch_warps",
        ):
            raise ValueError(f"unsupported token_back_mode={token_back_mode!r}.")
        if token_back_by_dispatch != (token_back_mode != "epi_warps"):
            raise ValueError(
                "token_back_by_dispatch must match token_back_mode; use "
                "token_back_mode != 'epi_warps' for dispatch token-back."
            )

        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            use_2cta_instrs=use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            load_balance_mode=load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=force_static_sched,
            clc_bundle_size=clc_bundle_size,
            num_sched_stages=num_sched_stages,
            acc_dtype=acc_dtype,
            ab_dtype=ab_dtype,
            scenario=scenario,
            fc2_in_kernel_topk_reduce=fc2_in_kernel_topk_reduce,
            token_back_by_dispatch=token_back_by_dispatch,
            epi_flag_batch=epi_flag_batch,
            gate_up_clamp=gate_up_clamp,
            apply_topk_in_fc1=apply_topk_in_fc1,
            generate_c=generate_c,
            use_stg_fc1=use_stg_fc1,
        )

        self.enable_token_comm = True
        self.dispatch_warp_id = (8, 9, 10, 11)
        # Standalone token-back: a dedicated 4-warp group (12-15) pushing fc2
        # results back to source ranks concurrently with dispatch_pull.
        self.token_back_mode = token_back_mode
        self.token_back_standalone = (
            token_back_by_dispatch
            and token_back_mode == "standalone_warps"
        )
        self.token_back_warp_id = (12, 13, 14, 15) if self.token_back_standalone else None
        num_token_back_warps = len(self.token_back_warp_id) if self.token_back_standalone else 0
        self.token_back_schedule_mode = (
            self.load_balance_mode if token_back_by_dispatch else "static"
        )
        self.combine_format = CombineFormat.parse("bf16")
        self.threads_per_cta = 32 * (
            len(self.epilogue_warp_id)
            + 1  # mma
            + 1  # tma_a
            + 1  # tma_b
            + 1  # sched
            + len(self.dispatch_warp_id)
            + num_token_back_warps
        )

        # Independent MegaMoE-specific constants.
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden
        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce

        # static_expert_shape = (num_experts_per_rank, intermediate_gateup, hidden).
        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_gateup = static_expert_shape[1]
        self.intermediate_downproj = self.intermediate_gateup // 2

        # BF16 activations: 2 bytes/element.
        self.hidden_bytes = self.hidden * (self.ab_dtype.width // 8)
        # Cross-rank totals: per-rank count * world_size.
        self.num_total_experts = world_size * self.num_experts_per_rank

        # Per-cluster-tile token count.  Non-swap-AB: the token axis is M, and
        # the base-class validator enforces mma_tiler_m == per-CTA-tile-M *
        # cluster_m (128 per CTA), so mma_tiler_mnk[0] IS the cluster tile M.
        self.cluster_tile_tokens = mma_tiler_mnk[0]

        # Cache region sizing inputs used by workspace layout and __call__.
        (
            self.pool_token_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()

        # Cohabiting warps outside the dispatch group: epilogue + mma + tma_a
        # + tma_b + sched.
        num_other_warps = len(self.epilogue_warp_id) + 1 + 1 + 1 + 1

        # For token_back_by_dispatch, the dispatch warp pushes fc2 results
        # from the local pool workspace back to each source rank's combine_output.
        # fc2_publishes_per_token_cluster_tile = ceil(hidden / mma_tiler_m) * cluster_m:
        # for each cluster token tile, each of cluster_m CTAs publishes once per
        # hidden N-tile it processes (N-tile width = mma_tiler[1]).
        if token_back_by_dispatch:
            _ctt_n = self.mma_tiler_mnk[1]
            fc2_publishes = (
                (self.hidden + _ctt_n - 1) // _ctt_n
            ) * self.cluster_shape_mn[0]
        else:
            fc2_publishes = 0

        self.token_comm = TokenInPullTokenBackPush(
            world_size=self.world_size,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=self.ab_dtype,
            combine_format=self.combine_format,
            token_back_by_dispatch=token_back_by_dispatch,
            fc2_publishes_per_token_cluster_tile=fc2_publishes,
            # Push-side reduce only when a token-back mode carries the data;
            # under epi_warps the epilogue's REDG performs the reduction.
            token_back_reduce_topk=(
                token_back_by_dispatch and fc2_in_kernel_topk_reduce
            ),
            token_back_standalone=self.token_back_standalone,
            # BF16 tokens carry no scale-factor sideband.  sf_uint32_per_token
            # is a constexpr trip count inside dispatch_pull, so 0 compiles the
            # SF pull/store loops out entirely; sf_padding_block=1 keeps the SF
            # cumul bookkeeping arithmetic neutral (never consumed).
            sf_uint32_per_token=0,
            token_padding_block=self.token_padding_block,
            sf_padding_block=1,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=self.dispatch_warp_id[0],
            num_other_warps=num_other_warps,
            is_swap_ab=False,
            flag_batch=flag_batch,
            token_back_schedule_mode=self.token_back_schedule_mode,
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
        # Peek threshold must match the spin threshold (cluster_tile_tokens = 256)
        # so an early peek hit does not skip the spin and expose stale pool rows.
        return self.cluster_tile_tokens

    # =========================================================================
    # SMEM budget hook (base override)
    # =========================================================================

    def _dispatch_smem_bytes(self) -> int:
        """SMEM for dispatch pull mbarriers, expert scratch, and token buffer.

        Must match ``TokenInPullTokenBackPush.extra_smem_storage_class``:
        ``pull_mbar[Int64, 4] + smem_expert_count[Int32, num_total_experts]
        + pull_buffer[Uint8, 4 * hidden_bytes]``.
        Standalone token-back adds ``tb_pull_mbar[Int64, 4]`` and
        ``tb_pull_buffer[Uint8, 4 * tb_chunk_bytes]``.
        """
        pull_mbar_bytes = _DispatchWarpCount * 8
        expert_count_bytes = self.num_total_experts * 4
        pull_buffer_bytes = _DispatchWarpCount * self.hidden_bytes
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

    def _pool_shapes(self) -> Tuple[int, int]:
        world_size = self.world_size
        max_tokens_per_rank = self.max_tokens_per_rank
        num_topk = self.num_topk
        num_experts_per_rank = self.num_experts_per_rank
        token_padding_block = self.token_padding_block
        cluster_tile_tokens = self.cluster_tile_tokens

        max_recv = world_size * max_tokens_per_rank
        max_per_token = min(num_topk, num_experts_per_rank)
        raw = (
            max_recv * max_per_token
            + num_experts_per_rank * (token_padding_block - 1)
        )
        pool_token_capacity = _round_up(raw, token_padding_block)
        pool_task_tile_capacity = (
            (pool_token_capacity + cluster_tile_tokens - 1) // cluster_tile_tokens
            + num_experts_per_rank
        )
        return (
            pool_token_capacity,
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
        cluster_tile_tokens = self.cluster_tile_tokens

        # fc1_done_counter slot granularity is the cluster tile on the token
        # (M) axis, plus one boundary slot per expert.
        fc1_done_slots = (
            (pool_token_capacity + cluster_tile_tokens - 1) // cluster_tile_tokens
            + num_experts_per_rank
        )

        # Accumulating-counter prefix: tail_reset_counters bulk-zeros all bytes
        # before l1_token_buffer so back-to-back launches do not inherit counters.
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

        # Data buffers. l1_token_buffer must be the first data region because
        # __init__ derives local_zero_i32_count from its offset.
        specs += [
            # L1 input pool (dispatch_pull writes -> fc1 reads).  Stored as
            # Uint8 bytes; the BF16 view at the same offset is built in __call__.
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (pool_token_capacity, hidden_bytes),
                128,
            ),
            # Persisted across launches; the sense-reversing nvlink barrier rides
            # this phase counter across non-ncu relaunches.
            _RegionSpec(
                "nvlink_barrier_counter",
                cutlass.Int32,
                (1,),
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
            # fc1 -> fc2 hand-off buffer in the activation dtype (BFloat16).
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

        # Shared counter prefix is bulk-zeroed between tail barriers; keep
        # src_token_topk_idx as the first data region used to derive the prefix.
        return [
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
        byte_workspace: cute.Tensor,
        byte_offset: int,
        cute_dtype: Any,
        shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]],
        assumed_align: int,
    ) -> cute.Tensor:
        """Build a typed cute view at ``byte_offset`` of the opaque workspace."""
        byte_ptr = byte_workspace.iterator + Int64(byte_offset)
        typed_iter = cute.make_ptr(
            cute_dtype,
            byte_ptr.toint(),
            AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        return cute.make_tensor(typed_iter, cute.make_layout(shape, stride=stride))

    def _view_local(
        self,
        local_workspace: cute.Tensor,
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
        shared_workspace: cute.Tensor,
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
        byte_workspace: cute.Tensor,
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
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,           # (T, hidden) BF16
        topk_idx: cute.Tensor,             # (T, num_topk) Int64
        topk_weights: cute.Tensor,         # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc1_c: Optional[cute.Tensor],      # fc1 c output
        # Combine destination (peer write target via the epilogue Fc2OutputDest).
        combine_output: cute.Tensor,       # (T, num_topk, hidden) BF16
        # Opaque workspaces.
        local_workspace: cute.Tensor,      # (local_ws_bytes,) Uint8
        shared_workspace: cute.Tensor,     # (shared_ws_bytes,) Uint8
        # Runtime host payload; packed into ``SymBuffer{world_size}``.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Launch the BF16 MegaMoE-complete fused kernel.

        Pointer-mapping contract:
          * ``activation`` / ``topk_weights`` / ``combine_output`` MUST point
            into memory reachable via
            ``peer_rank_ptr_mapper.ptr_map_to_rank(...)`` (NVSHMEM symmetric
            heap).  Single-rank degenerate runs are allowed.
          * ``topk_idx`` is read on the local rank only.
          * ``fc1_weight`` / ``fc2_weight`` are local-only.

        ``combine_output`` is the MoE-domain ``(max_tokens_per_rank, num_topk,
        hidden)`` BF16 storage; the epilogue maps each pool row back to the
        source rank's ``[src_token, src_topk, :]`` slot via ``token_comm_args``
        (form A; host reduces the topk axis).
        """
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        hidden = self.hidden

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + BF16 view
        # (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_local(local_workspace, "l1_token_buffer")
        l1_token_buffer_bf16 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            self.ab_dtype,
            (pool_token_capacity, hidden),
            (hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )

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

        token_back_schedule_counter = None
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            token_back_schedule_counter = self._view_local(
                local_workspace, "token_back_schedule_counter",
            ).iterator

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
            combine_output_u8 = cute.recast_tensor(combine_output, cutlass.Uint8)
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_workspace_native = None
            fc2_output_workspace_u8 = None
            fc2_done_counter = None
            combine_output_u8 = combine_output
            fc2_output_target = combine_output

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
            local_workspace, 0, cutlass.Int32, (self.local_zero_i32_count,), (1,), 16,
        )
        shared_zero_prefix = self._make_typed_view(
            shared_workspace, 0, cutlass.Int32, (self.shared_zero_i32_count,), (1,), 16,
        )

        token_comm_args = ExtractedTokenCommArgs(
            input_token_buffer=activation,
            # BF16 tokens carry no scale-factor sideband: the SF slots ride the
            # TokenCommArgs None-skipping serialization, and dispatch's SF
            # loops are compiled out by sf_uint32_per_token=0.
            input_sf_buffer=None,
            topk_idx=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=None,
            fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            combine_output=combine_output_u8,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_done_counter=fc2_done_counter,
            token_back_schedule_counter=token_back_schedule_counter,
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
            sf_uint32_per_token=0,
            token_padding_block=self.token_padding_block,
            sf_padding_block=1,
            sm_count=sm_count,
        )

        super().__call__(
            activation=l1_token_buffer_bf16,
            fc1_weight=fc1_weight,
            fc1_output=fc1_output,
            fc1_c=fc1_c,
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
