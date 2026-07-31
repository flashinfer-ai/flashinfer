# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Autonomous epilogue for the fused fc1+fc2 swap-AB MegaMoE kernel.

Component boundaries use ``TensorWithContract`` to keep per-thread RMEM layout
semantics explicit.  See ``megamoe_design.md`` for the epilogue dataflow.
"""

from typing import Optional, Tuple, Type, Union, Any
import dataclasses

import cutlass
import cutlass.cute as cute

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from src.iket_compat import iket
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils

from cutlass.cutlass_dsl import Int64

from src.flag_batch import GpuReleaseFlagBatchTracker
from src.token_comm import CombineFormat, TokenSrcMetadata
from src.ptx_helpers import stg_e8m0_from_f32, stg_e8m0x8_from_f32

from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase

from common.moe_utils import fmin, fmax, swiglu_act, quant_sfd_row
from cutlass.cute.typing import Float32
from moe_nvfp4_swapab.epilogue import (
    _TmemTranspose16x32Core,
    _red_add_relaxed_sys_v2_bf16x2,
)

Fc1GateUpInterleave = 32
EpilogueTileN = 32
Fc1EpilogueOutputTileM = 256
Fc1EpilogueOutputTileN = 128
WarpThreadCount = 32
EpiWarpCount = 4
Fc1CTMAStages = 1


# =============================================================================
# Fc2OutputDest  (MegaMoE fc2 STG destination resolver, non-swap MXFP8 path)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Fc2OutputDest:
    """fc2 output destination in MoE-domain ``(token_max, topk, hidden)`` layout.

    Mirrors the indirect-mode resolution used by the NVFP4 ``Fc2ReturnTile``
    (``moe_nvfp4_swapab.epilogue``) but specialised for the non-swap-AB MXFP8
    epilogue, whose fc2 STG is already a simple per-(token_row, hidden) store.

    * ``metadata`` / ``peer_rank_ptr_mapper`` both ``None`` -- **direct mode**:
      the row is ``tensor[pool_token, 0, :]`` on the local rank.
    * Both non-``None`` -- **indirect mode** (MegaMoE form A): each lane LDGs
      ``(src_rank, src_token, src_topk)`` from ``metadata[pool_token, :]`` and
      writes to ``peer(src_rank).tensor[src_token, src_topk, :]`` via
      ``peer_rank_ptr_mapper.ptr_map_to_rank(local_addr, src_rank)``.

    ``metadata`` is the ``(pool_token_count, TokenSrcMetadata.nbytes)`` Uint8 view
    (or any recast with the same base pointer) of the dispatch warps'
    ``token_src_metadata`` record; each record is one packed Int64 decoded by
    ``TokenSrcMetadata.load()``.  ``peer_rank_ptr_mapper``
    is a ``SymBuffer{world_size}`` instance (returns a zero delta when
    ``src_rank == local_rank``, so single-rank runs fold to a no-op).
    """

    tensor: cute.Tensor
    metadata: Optional[cute.Tensor] = None
    peer_rank_ptr_mapper: Any = None
    reduce_topk_in_kernel: bool = False

    def __post_init__(self) -> None:
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError(
                "Fc2OutputDest: ``metadata`` and ``peer_rank_ptr_mapper`` must be "
                "both None (direct mode) or both non-None (MegaMoE / indirect "
                "mode).  Got metadata="
                f"{'set' if self.metadata is not None else 'None'}, "
                f"peer_rank_ptr_mapper="
                f"{'set' if self.peer_rank_ptr_mapper is not None else 'None'}."
            )

    @cute.jit
    def resolve_token_row(self, pool_token_global) -> cute.Tensor:
        """Return the ``(hidden,)`` BF16 GMEM row this pool token's STG lands on.

        Direct mode: ``tensor[pool_token_global, 0, :]`` on the local rank.
        Indirect mode: LDG packed i64 from ``metadata`` at byte offset
        ``pool_token_global * TokenSrcMetadata.nbytes``, unpack via
        ``TokenSrcMetadata.load()`` to get ``(src_rank, src_token, src_topk)``,
        build the local-rank row view, then rebase the row's GMEM pointer
        through ``peer_rank_ptr_mapper.ptr_map_to_rank`` so the iterator points
        at the source rank's combine slot.
        """
        if cutlass.const_expr(self.metadata is None):
            return cute.slice_(self.tensor, (pool_token_global, 0, None))

        md = TokenSrcMetadata.load(
            self.metadata.iterator.toint()
            + Int64(pool_token_global) * Int64(TokenSrcMetadata.nbytes)
        )
        src_rank = md.src_rank
        src_token = md.src_token
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            src_topk = cutlass.Int32(0)
        else:
            src_topk = md.src_topk
        local_row = cute.slice_(self.tensor, (src_token, src_topk, None))
        peer_iter = self.peer_rank_ptr_mapper.ptr_map_to_rank(
            local_row.iterator,
            src_rank,
        )
        return cute.make_tensor(peer_iter, local_row.layout)


# =============================================================================
# GluMxfp8Epilogue
# =============================================================================


class GluMxfp8Epilogue:
    _SubtileBarIdBase = 4
    # Named barrier for cross-warp sync during raw-C TMA stores.
    _CStoreBarId = 10

    def __init__(
        self,
        *,
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        use_2cta_instrs: bool,
        sf_vec_size: int,
        fc1_output_dtype: Type[cutlass.Numeric],
        fc1_output_layout: utils.LayoutEnum,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
        c_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        glu_clamp: Optional[float] = None,
        allow_overlap_acc: bool = True,
        epilog_sync_bar_id: int = 1,
        epilogue_warp_ids: Tuple[int, ...] = (0, 1, 2, 3),
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: Union[int, Tuple[int, int]] = 1,
        apply_topk_in_fc1: bool = False,
        generate_c: bool = False,
        use_stg_fc1: bool = False,
        combine_format: Optional[Any] = None,
    ) -> None:
        self.fc1_output_dtype = fc1_output_dtype
        self.fc1_output_layout = fc1_output_layout
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        self._sf_vec_size = sf_vec_size
        self._c_dtype = c_dtype
        self._epilog_sync_bar_id = epilog_sync_bar_id
        self._epilogue_warp_ids = epilogue_warp_ids
        self._use_2cta_instrs = use_2cta_instrs

        self._atom_thr_size = 2 if use_2cta_instrs else 1
        self._cta_tile_m = mma_tiler_mnk[0] // self._atom_thr_size
        self._cta_tile_n = mma_tiler_mnk[1]
        self._mma_tiler_k = mma_tiler_mnk[2]
        self._cta_tile_n_sfb = ((mma_tiler_mnk[1] + 127) // 128) * 128
        self._static_expert_shape = static_expert_shape
        if (
            static_expert_shape is not None
            and static_expert_shape[2] % (self._cta_tile_m * cluster_shape_mn[0]) == 0
        ):
            self._fc2_stg_needs_predicate: bool = False
        else:
            self._fc2_stg_needs_predicate: bool = True

        # Non-swap-AB: TMA tile is (EpilogueTileN tokens, Fc1EpilogueOutputTileN intermediates)
        self._epi_tile = (EpilogueTileN, Fc1EpilogueOutputTileN)
        self._subtile_cnt = self._cta_tile_n // 2 // EpilogueTileN

        self._overlapping_accum = allow_overlap_acc and (
            self._cta_tile_n == EpiWarpCount * EpilogueTileN * 2
        )
        self._overlapping_accum = True

        self._num_acc_stage = 2
        self._num_acc_pipeline_stages = (
            1 if self._overlapping_accum else self._num_acc_stage
        )

        k = self._mma_tiler_k
        self._num_sfa_tmem_cols = self._cta_tile_m * k // sf_vec_size * 4 // 4 // 128
        self._num_sfb_tmem_cols = (
            self._cta_tile_n_sfb * k // sf_vec_size * 4 // 4 // 128
        )
        self._num_sf_tmem_cols = 32  # self._num_sfa_tmem_cols + self._num_sfb_tmem_cols

        self._num_accumulator_tmem_cols = self._cta_tile_n * self._num_acc_stage - (
            self._num_sf_tmem_cols if self._overlapping_accum else 0
        )

        self._iter_acc_early_release = (
            self._num_sf_tmem_cols + EpilogueTileN - 1
        ) // EpilogueTileN

        self._fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self._token_back_by_dispatch = token_back_by_dispatch
        self._apply_topk_in_fc1 = apply_topk_in_fc1
        self._generate_c = generate_c
        self._use_stg_fc1 = use_stg_fc1
        # combine_format determines fc2 output encoding: bf16 (default) or quantized.
        if combine_format is None:
            combine_format = CombineFormat.parse("bf16")
        self._combine_format = combine_format
        self._combine_mxfp8 = combine_format.is_quantized
        # sf_block_pad for fc2 MXFP8 combine: number of E8M0 entries per pool
        # token, padded to 16 (= 16-byte TMA alignment for E8M0 1 byte/element).
        # EpilogueTileN = 32 = sf_vec_size, so hidden // EpilogueTileN = sf_blocks.
        if self._combine_mxfp8 and static_expert_shape is not None:
            _hidden_fc2 = static_expert_shape[2]
            _sf_blocks_fc2 = _hidden_fc2 // EpilogueTileN
            self._fc2_sf_block_pad = ((_sf_blocks_fc2 + 15) // 16) * 16
            self._hidden_fc2 = _hidden_fc2
        else:
            self._fc2_sf_block_pad = 0
            self._hidden_fc2 = 0
        # stg.64 SF batching: when hidden is a whole multiple of cta_tile_n every
        # scheduled fc2 N-tile owns a full run of fc2_subtile_cnt (= 8) contiguous,
        # in-bounds E8M0 blocks per token (the plane is token-row-major, block
        # stride 1), so a thread's 8 per-subtile scales flush as one 64-bit store.
        # Otherwise the ragged tail tile falls back to per-block 1-byte stores.
        self._fc2_sf_batch8 = (
            self._combine_mxfp8
            and self._hidden_fc2 > 0
            and (self._hidden_fc2 % self._cta_tile_n == 0)
            and (self._cta_tile_n // EpilogueTileN == 8)
        )
        self._epi_tile_c = (self._cta_tile_m, 2 * Fc1GateUpInterleave)
        if isinstance(epi_flag_batch, tuple):
            self._epi_fc1_batch = max(1, epi_flag_batch[0])
            self._epi_fc2_batch = max(1, epi_flag_batch[1])
        else:
            self._epi_fc1_batch = max(1, epi_flag_batch)
            self._epi_fc2_batch = max(1, epi_flag_batch)

        self.glu_clamp = cutlass.Float32(glu_clamp) if glu_clamp is not None else None

    # -- Codegen-time queries  --

    @property
    def epi_tile(self) -> Tuple[int, int]:
        return self._epi_tile

    @property
    def overlapping_accum(self) -> bool:
        return self._overlapping_accum

    @property
    def num_acc_pipeline_stages(self) -> int:
        return self._num_acc_pipeline_stages

    @property
    def num_acc_stage(self) -> int:
        return self._num_acc_stage

    @property
    def iter_acc_early_release(self) -> int:
        return self._iter_acc_early_release

    @property
    def subtile_cnt(self) -> int:
        return self._subtile_cnt

    @property
    def cta_tile_n(self) -> int:
        return self._cta_tile_n

    @property
    def num_sf_tmem_cols(self) -> int:
        return self._num_sf_tmem_cols

    @property
    def num_sfa_tmem_cols(self) -> int:
        return self._num_sfa_tmem_cols

    @property
    def num_sfb_tmem_cols(self) -> int:
        return self._num_sfb_tmem_cols

    @property
    def num_accumulator_tmem_cols(self) -> int:
        return self._num_accumulator_tmem_cols

    def staged_smem_layout(
        self,
        n_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        return sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            self.fc1_output_layout,
            self._epi_tile,
            n_stages,
        )

    @property
    def smem_layout_one_stage(self) -> Union[cute.Layout, cute.ComposedLayout]:
        staged = self.staged_smem_layout(1)
        return cute.select(staged, mode=[0, 1])

    @property
    def bytes_per_stage(self) -> int:
        return cute.size_in_bytes(self.fc1_output_dtype, self.smem_layout_one_stage)

    @property
    def epi_tile_c(self) -> Tuple[int, int]:
        """TMA tile for raw gate+up output: (cta_tile_m=128, 2*Fc1GateUpInterleave=64) fp32."""
        return self._epi_tile_c

    def staged_c_smem_layout(self, n_stages: int):
        """SMEM layout for n_stages of raw gate+up (Float32, row-major, epi_tile_c)."""
        return sm100_utils.make_smem_layout_epi(
            self._c_dtype,
            self.fc1_output_layout,  # same N-major direction as fc1 output
            self._epi_tile_c,
            n_stages,
        )

    @property
    def c_smem_layout_one_stage(self):
        return cute.select(self.staged_c_smem_layout(1), mode=[0, 1])

    @property
    def c_bytes_per_stage(self) -> int:
        return cute.size_in_bytes(self._c_dtype, self.c_smem_layout_one_stage)

    @staticmethod
    @cute.jit
    def tma_store_fc1_output(
        warp_idx,
        sC,
        store_idx,
        tma_atom_fc1_output: cute.CopyAtom,
        g_fc1_output_subtile_view: cute.Tensor,
        valid_tokens,
    ) -> None:
        """
        Per-warp TMA store for non-swap-AB MXFP8 FC1 output.
        Uses rotated-leader pattern (leader = warp store_idx).
        SMEM stage store_idx has (EpilogueTileN tokens × Fc1EpilogueOutputTileN intermediates).
        All 4 warps arrive at barrier store_idx+_SubtileBarIdBase, leader issues TMA.
        """
        cute.arch.fence_proxy("async.shared", space="cta")
        sC_stage = cute.slice_(sC, (None, None, store_idx))
        g_fc1_output_2d = cute.slice_(g_fc1_output_subtile_view, (None, None, 0))
        bSG_sC, bSG_g = cpasync.tma_partition(
            tma_atom_fc1_output,
            0,
            cute.make_layout(1),
            cute.group_modes(sC_stage, 0, 2),
            cute.group_modes(g_fc1_output_2d, 0, 2),
        )

        leader_warp = store_idx
        tile_has_valid = store_idx * cutlass.Int32(EpilogueTileN) < valid_tokens

        bar_id = store_idx + cutlass.Int32(GluMxfp8Epilogue._SubtileBarIdBase)
        bar = pipeline.NamedBarrier(
            barrier_id=bar_id,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        if warp_idx == leader_warp:
            bar.arrive_and_wait()
            # TMA bulk-tensor stores are all-or-nothing per issue (no per-element
            # `pred` mask, no scalar predicate arg), so guard the whole copy with
            # a runtime `if`.  Skips fully-padding token-tiles that would alias
            # the next expert's region.
            if tile_has_valid:
                cute.copy(tma_atom_fc1_output, bSG_sC, bSG_g)
        else:
            bar.arrive()

    @cute.jit
    def _store_fc1_c_subtile(
        self,
        r_gate: cute.Tensor,
        r_up: cute.Tensor,
        smem_c_buffer: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        gmem_c_subtile_view: cute.Tensor,
        c_buffer_idx,
        work_tile_info,
        warp_idx: int,
        tidx,
        c_pipeline,
    ) -> None:
        """Store pre-SwiGLU gate/up accumulators to the global C tensor via SMEM staging."""
        r_layout = cute.make_layout((((Fc1GateUpInterleave,), 1),), stride=(((1,), 0),))

        # Cast acc_dtype (Float32) → c_dtype (e.g. BFloat16) before R2S.
        # smem stage is guaranteed free by the producer_acquire() from the previous call.
        r_gate_c = cute.make_rmem_tensor(r_layout.shape, self._c_dtype)
        r_up_c = cute.make_rmem_tensor(r_layout.shape, self._c_dtype)
        r_gate_c.store(r_gate.load().to(self._c_dtype))
        r_up_c.store(r_up.load().to(self._c_dtype))

        r2s_c_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._c_dtype,
            num_bits_per_copy=128,
        )
        thread_in_warp_c = tidx % cutlass.Int32(WarpThreadCount)
        c_row = cutlass.Int32(warp_idx * EpilogueTileN) + thread_in_warp_c
        sC_raw_stage = cute.slice_(smem_c_buffer, (None, None, c_buffer_idx))
        c_gate_smem = cute.local_tile(
            sC_raw_stage,
            (1, Fc1GateUpInterleave),
            (c_row, cutlass.Int32(0)),
        )
        cute.copy(r2s_c_atom, cute.coalesce(r_gate_c), cute.coalesce(c_gate_smem))
        c_up_smem = cute.local_tile(
            sC_raw_stage,
            (1, Fc1GateUpInterleave),
            (c_row, cutlass.Int32(1)),
        )
        cute.copy(r2s_c_atom, cute.coalesce(r_up_c), cute.coalesce(c_up_smem))

        # Fence + barrier: ensure all warps have written before TMA issue.
        cute.arch.fence_proxy("async.shared", space="cta")
        c_store_bar = pipeline.NamedBarrier(
            barrier_id=self._CStoreBarId,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        c_store_bar.arrive_and_wait()

        # Warp 0 issues TMA S2G, commits, then pre-acquires the next stage so
        # smem is guaranteed free before the next call's R2S writes.
        if warp_idx == 0:
            if work_tile_info.valid_tokens_in_cta_tile > cutlass.Int32(0):
                g_c_2d = cute.slice_(gmem_c_subtile_view, (None, None, 0))
                bSG_sC, bSG_gC = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC_raw_stage, 0, 2),
                    cute.group_modes(g_c_2d, 0, 2),
                )
                cute.copy(tma_atom_c, bSG_sC, bSG_gC)
            c_pipeline.producer_commit()

    def _subtile_local_tmem_tensor_pair(
        self,
        tmem_acc_tensor: cute.Tensor,
        subtile_idx,
        warp_idx,
    ) -> cute.Tensor:
        """
        Build a (gate, up) pair of TMEM tensor views for the MXFP8 fc1 epilogue.

        Owns the per-warp lane offset, per-stage col offset (overlap-acc
        phase aware), and per-subtile col offset arithmetic.  Returned
        tensor is what ``_run_fc{1,2}_subtile`` and
        ``_TmemTranspose16x32Core.load_subtile_raw_acc`` consume.

        ``cute.assume(divby=16)`` is applied here once -- callees can
        derive ``+32`` first/second-half ptrs from the returned tensor's
        iterator without re-asserting alignment (16-aligned base + 32 is
        still 16-aligned).
        """
        base = tmem_acc_tensor.iterator
        warp_lane_off = warp_idx * WarpThreadCount
        subtile_col_off = subtile_idx * EpilogueTileN * 2
        total = (warp_lane_off << 16) + subtile_col_off
        subtile_gate_ptr = base + cute.assume(total, divby=16)
        subtile_up_ptr = base + cute.assume(total + Fc1GateUpInterleave, divby=16)
        return (
            cute.make_tensor(
                subtile_gate_ptr,
                _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
            ),
            cute.make_tensor(
                subtile_up_ptr,
                _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
            ),
        )

    def _subtile_forward_tmem_tensor(
        self,
        tmem_gate_tensor: cute.Tensor,
        tmem_up_tensor: cute.Tensor,
        col_offset: int,
    ) -> (cute.Tensor, cute.Tensor):
        """Move the tmem tensor to the correct position."""
        tmem_gate_ptr = tmem_gate_tensor.iterator + cute.assume(col_offset, divby=16)
        tmem_up_ptr = tmem_up_tensor.iterator + cute.assume(col_offset, divby=16)
        return (
            cute.make_tensor(
                tmem_gate_ptr,
                _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
            ),
            cute.make_tensor(
                tmem_up_ptr,
                _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
            ),
        )

    # -- fc1 subtile: SM100 path --
    @cute.jit
    def _run_fc1_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        smem_c_buffer: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        gmem_c: cute.Tensor,
        c_pipeline,
    ) -> None:
        """MXFP8 fc1 task-tile: TmemTranspose16x32 TMEM loading + cross-warp E8M0 exchange."""
        real_fc1_output, _ = sched_ext.get_gmem_tensor(
            "d", gmem_fc1_output, work_tile_info
        )
        real_fc1_output_sf, _ = sched_ext.get_gmem_tensor(
            "sfd", gmem_fc1_output_sf, work_tile_info
        )
        real_topk_scores, _ = sched_ext.get_gmem_tensor(
            "topk", gmem_topk_scores, work_tile_info
        )

        if cutlass.const_expr(self._generate_c):
            real_c, _ = sched_ext.get_gmem_tensor("c", gmem_c, work_tile_info)
            c_n_base = work_tile_info.tile_n_idx * cutlass.Int32(self._subtile_cnt)

        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_push("mxfp8_fc1_epi_tile")

        if cutlass.const_expr(self._overlapping_accum):
            acc_stage_col_offset = cutlass.Int32(acc_consumer_state.phase) * (
                256 - self._num_sf_tmem_cols
            )
        else:
            acc_stage_col_offset = (
                cutlass.Int32(acc_consumer_state.index) * self._cta_tile_n
            )

        subtile_cnt = self._subtile_cnt
        tmem_gate, tmem_up = self._subtile_local_tmem_tensor_pair(
            tmem_acc_tensor,
            subtile_cnt - 1 if is_odd_turn else 0,
            warp_idx,
        )
        tmem_forward_cols = Fc1GateUpInterleave * 2
        if cutlass.const_expr(self._overlapping_accum):
            if is_odd_turn:
                tmem_forward_cols = -Fc1GateUpInterleave * 2

        layout_sf = cute.make_layout(4)
        rmem_sf = cute.make_rmem_tensor(layout_sf.shape, self.acc_dtype)

        # ── Set up RMEM→SMEM copy atom (direct CopyUniversalOp) ──
        r2s_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.fc1_output_dtype,
            num_bits_per_copy=128,
        )
        tRS_sC = None

        for i in cutlass.range(0, subtile_cnt, 1, unroll=1):
            if cutlass.const_expr(self._overlapping_accum):
                subtile_idx = cutlass.Int32(i)
                if is_odd_turn:
                    subtile_idx = cutlass.Int32(subtile_cnt - 1 - i)
            else:
                subtile_idx = cutlass.Int32(i)

            if cutlass.const_expr(self._generate_c):
                c_buffer_idx = cutlass.Int32(i % Fc1CTMAStages)
                g_c_subtile = cute.local_tile(
                    real_c,
                    (self._cta_tile_m, 2 * Fc1GateUpInterleave, 1),
                    (
                        work_tile_info.tile_m_idx,
                        c_n_base + subtile_idx,
                        cutlass.Int32(0),
                    ),
                )
                smem_c_buf_arg = smem_c_buffer
                tma_atom_c_arg = tma_atom_c
                gmem_c_subtile_arg = g_c_subtile
            else:
                # Dummy values — never accessed due to const_expr guard in _run_fc1_subtile.
                c_buffer_idx = cutlass.Int32(0)
                smem_c_buf_arg = smem_fc1_output_buffer
                tma_atom_c_arg = tma_atom_fc1_output
                gmem_c_subtile_arg = real_fc1_output

            self._run_fc1_subtile(
                subtile_idx=subtile_idx,
                tmem_gate_tensor=tmem_gate,
                tmem_up_tensor=tmem_up,
                real_fc1_output=real_fc1_output,
                real_fc1_output_sf=real_fc1_output_sf,
                real_topk_scores=real_topk_scores,
                work_tile_info=work_tile_info,
                smem_fc1_output_buffer=smem_fc1_output_buffer,
                tma_atom_fc1_output=tma_atom_fc1_output,
                r2s_copy_atom=r2s_copy_atom,
                warp_idx=warp_idx,
                tidx=tidx,
                alpha=alpha,
                norm_const=norm_const,
                rmem_sf=rmem_sf,
                acc_is_release=(i == 0),
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                smem_c_buffer=smem_c_buf_arg,
                tma_atom_c=tma_atom_c_arg,
                gmem_c_subtile_view=gmem_c_subtile_arg,
                c_buffer_idx=c_buffer_idx,
                c_pipeline=c_pipeline,
            )

            tmem_gate, tmem_up = self._subtile_forward_tmem_tensor(
                tmem_gate, tmem_up, tmem_forward_cols
            )

        if not cutlass.const_expr(self._overlapping_accum):
            self._acc_pipeline_consumer_release(acc_pipeline, acc_consumer_state, True)

        self._stg_sf_fc1(rmem_sf, real_fc1_output_sf, work_tile_info, tidx)

        # ── TMA store: 4 stores (one per warp group) after all subtiles ──
        if cutlass.const_expr(not self._use_stg_fc1):
            base_token_tile = work_tile_info.tile_m_idx * cutlass.Int32(
                self._cta_tile_m // EpilogueTileN
            )
            for idx in cutlass.range_constexpr(EpiWarpCount):
                g_fc1_output_warp_view = cute.local_tile(
                    real_fc1_output,
                    (EpilogueTileN, Fc1EpilogueOutputTileN, 1),
                    (base_token_tile + idx, work_tile_info.tile_n_idx, 0),
                )
                GluMxfp8Epilogue.tma_store_fc1_output(
                    warp_idx,
                    smem_fc1_output_buffer,
                    idx,
                    tma_atom_fc1_output,
                    g_fc1_output_warp_view,
                    work_tile_info.valid_tokens_in_cta_tile,
                )

        iket.range_pop()

    @cute.jit
    def _run_fc1_subtile(
        self,
        subtile_idx,
        tmem_gate_tensor: cute.Tensor,
        tmem_up_tensor: cute.Tensor,
        real_fc1_output: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        real_topk_scores: cute.Tensor,
        work_tile_info,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        r2s_copy_atom: cute.CopyAtom,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        rmem_sf: cute.Tensor,
        acc_is_release: bool,
        acc_pipeline,
        acc_consumer_state,
        smem_c_buffer: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        gmem_c_subtile_view: cute.Tensor,
        c_buffer_idx,
        c_pipeline,
    ) -> None:
        """MXFP8 fc1 subtile: GLU + E8M0 SF + fp8 R2S."""
        iket.range_push("mxfp8_fc1_epilogue_subtile")

        r_layout = cute.make_layout((((Fc1GateUpInterleave,), 1),), stride=(((1,), 0),))
        r_gate = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        r_up = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)

        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            self.acc_dtype,
        )
        cute.copy(atom_t2r, tmem_gate_tensor, r_gate)
        cute.copy(atom_t2r, tmem_up_tensor, r_up)

        self._acc_pipeline_consumer_release(
            acc_pipeline, acc_consumer_state, acc_is_release
        )

        # ── generate_c: store raw gate+up to GMEM C tensor via SMEM staging ──
        if cutlass.const_expr(self._generate_c):
            self._store_fc1_c_subtile(
                r_gate=r_gate,
                r_up=r_up,
                smem_c_buffer=smem_c_buffer,
                tma_atom_c=tma_atom_c,
                gmem_c_subtile_view=gmem_c_subtile_view,
                c_buffer_idx=c_buffer_idx,
                work_tile_info=work_tile_info,
                warp_idx=warp_idx,
                tidx=tidx,
                c_pipeline=c_pipeline,
            )

        if cutlass.const_expr(self.glu_clamp is not None):
            for i in cutlass.range_constexpr(cute.size(r_up)):
                r_gate[i] = fmin(r_gate[i], self.glu_clamp)
                r_up[i] = fmin(r_up[i], self.glu_clamp)
                r_up[i] = fmax(r_up[i], -self.glu_clamp)

        topk = None
        if cutlass.const_expr(self._apply_topk_in_fc1):
            thread_in_warp = tidx % cutlass.Int32(WarpThreadCount)
            token_in_tile = (
                work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                + cutlass.Int32(warp_idx * WarpThreadCount)
                + thread_in_warp
            )
            topk = Float32(real_topk_scores[token_in_tile])

        swiglu = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        swiglu_act(swiglu, r_up, r_gate, topk)

        c = cute.make_rmem_tensor(r_layout.shape, self.fc1_output_dtype)
        qpvscale = quant_sfd_row(
            swiglu,
            c,
            norm_const,
            self._sf_vec_size,
            self.sf_dtype,
            self.fc1_output_dtype,
        )
        if subtile_idx == 0:
            rmem_sf[0] = qpvscale
        elif subtile_idx == 1:
            rmem_sf[1] = qpvscale
        elif subtile_idx == 2:
            rmem_sf[2] = qpvscale
        elif subtile_idx == 3:
            rmem_sf[3] = qpvscale

        thread_in_warp = tidx % WarpThreadCount
        if cutlass.const_expr(self._use_stg_fc1):
            # Direct STG.256 to GMEM — no SMEM staging or TMA store needed.
            token_in_tile = cutlass.Int32(warp_idx * EpilogueTileN) + thread_in_warp
            if token_in_tile < work_tile_info.valid_tokens_in_cta_tile:
                abs_token = (
                    work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                    + cutlass.Int32(warp_idx * EpilogueTileN)
                    + thread_in_warp
                )
                # absolute column start (element index in the intermediate axis)
                col_elem = (
                    work_tile_info.tile_n_idx * cutlass.Int32(self._subtile_cnt)
                    + subtile_idx
                ) * cutlass.Int32(Fc1GateUpInterleave)
                # (1,1,1) tile gives pointer to element at (abs_token, col_elem, 0).
                g_base = cute.local_tile(
                    real_fc1_output,
                    (1, 1, 1),
                    (abs_token, col_elem, cutlass.Int32(0)),
                )
                stg_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.fc1_output_dtype,
                    num_bits_per_copy=256,
                )
                # col_elem is always a multiple of Fc1GateUpInterleave=32 (FP8 elements),
                # so the pointer is 32-byte aligned — matching STG.256 requirement.
                aligned_iter = cute.make_ptr(
                    self.fc1_output_dtype,
                    g_base.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                )
                g_vec = cute.make_tensor(
                    aligned_iter, cute.make_layout(Fc1GateUpInterleave)
                )
                cute.copy(stg_atom, cute.coalesce(c), g_vec)
        else:
            sC_stage = cute.slice_(smem_fc1_output_buffer, (None, None, warp_idx))
            sC_thread_row = cute.local_tile(
                sC_stage, (1, Fc1GateUpInterleave), (thread_in_warp, subtile_idx)
            )
            cute.copy(r2s_copy_atom, cute.coalesce(c), cute.coalesce(sC_thread_row))

        if cutlass.const_expr(self._generate_c):
            if warp_idx == 0:
                c_pipeline.producer_acquire()
            c_store_bar = pipeline.NamedBarrier(
                barrier_id=self._CStoreBarId,
                num_threads=EpiWarpCount * WarpThreadCount,
            )
            c_store_bar.arrive_and_wait()

        iket.range_pop()

    @cute.jit
    def _subtile_fc2_tmem_tensor(
        self,
        tmem_acc_tensor: cute.Tensor,
        subtile_idx,
        warp_idx,
    ) -> cute.Tensor:
        """
        Per-warp TMEM view for one fc2 subtile (EpilogueTileN=32 cols).
        """
        base = tmem_acc_tensor.iterator
        warp_lane_off = warp_idx * WarpThreadCount
        subtile_col_off = subtile_idx * EpilogueTileN
        total = (warp_lane_off << 16) + subtile_col_off
        subtile_ptr = base + cute.assume(total, divby=16)
        return cute.make_tensor(
            subtile_ptr,
            _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
        )

    @cute.jit
    def _advance_fc2_tmem_tensor(
        self,
        tmem_tensor: cute.Tensor,
        col_offset: int,
    ) -> cute.Tensor:
        """Advance the fc2 TMEM tensor by col_offset cols (mirrors _subtile_forward_tmem_tensor)."""
        new_ptr = tmem_tensor.iterator + cute.assume(col_offset, divby=16)
        return cute.make_tensor(
            new_ptr,
            _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
        )

    @cute.jit
    def _acc_pipeline_consumer_release(
        self,
        acc_pipeline,
        acc_consumer_state,
        is_release: bool,
    ) -> None:
        """Release the acc pipeline consumer."""
        if is_release:
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def _run_fc2_subtile(
        self,
        subtile_idx,
        tmem_subtile_tensor: cute.Tensor,
        real_fc2_output: cute.Tensor,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
        rmem_sf_fc2=None,
        *,
        preload_acc=None,
    ) -> None:
        """fc2 subtile: LDTM + encode + STG.

        Encoding follows ``self._combine_format``:
          * BF16 (default): fp32->bf16 conversion, 256-bit STG × 2 half-tiles.
          * MXFP8: quant_sfd_row fp32->e4m3 (32 elems = one block), single
            256-bit STG for data; E8M0 scale written to local fc2_output_sf.

        When ``token_comm_args`` is not None (MegaMoE path), the data STG is
        routed to ``combine_output[src_token, src_topk, :]`` on the source rank
        via ``Fc2OutputDest``; for token_back_by_dispatch the epilogue writes to
        the local ``fc2_output_workspace`` pool instead.
        """
        iket.range_push("mxfp8_fc2_epilogue_subtile")

        fc2_subtile_cnt = self._cta_tile_n // EpilogueTileN  # = 8
        hidden_group = (
            work_tile_info.tile_n_idx * cutlass.Int32(fc2_subtile_cnt) + subtile_idx
        )
        hidden_col_start = work_tile_info.tile_n_idx * cutlass.Int32(
            self._cta_tile_n
        ) + subtile_idx * cutlass.Int32(EpilogueTileN)
        r_acc_layout = cute.make_layout((((EpilogueTileN,), 1),), stride=(((1,), 0),))
        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            self.acc_dtype,
        )
        r_acc = cute.make_rmem_tensor(r_acc_layout.shape, self.acc_dtype)
        cute.copy(atom_t2r, tmem_subtile_tensor, r_acc)
        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        if token_row_in_cta < valid_tokens and hidden_col_start < valid_hidden:
            if cutlass.const_expr(
                token_comm_args is not None
                and not self._token_back_by_dispatch
                and self._combine_mxfp8
            ):
                # MegaMoE Form A, quantized combine:
                # 1. Quantize fp32 → fp8 + compute E8M0 block scale.
                # 2. STG fp8 data to peer's combine_output.
                # 3. Write E8M0 scale to local fc2_output_sf for token-back push.
                fp8_dtype = self._combine_format.act_dtype
                r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
                qpvscale = quant_sfd_row(
                    r_acc,
                    r_fp8,
                    1.0,
                    EpilogueTileN,
                    cutlass.Float8E8M0FNU,
                    fp8_dtype,
                )
                pool_token_global = (
                    work_tile_info.cumulative_data_physical_row
                    + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                    + token_row_in_cta
                )
                metadata_u32 = cute.recast_tensor(
                    token_comm_args.token_src_metadata,
                    cutlass.Uint32,
                )
                fc2_output_dest = Fc2OutputDest(
                    tensor=token_comm_args.combine_output,
                    metadata=metadata_u32,
                    peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
                )
                dest_row = fc2_output_dest.resolve_token_row(pool_token_global)
                # STG 32 fp8 elements = 256 bits in one shot.
                r_fp8_flat = cute.make_tensor(r_fp8.iterator, cute.make_layout(32))
                stg_fp8_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    fp8_dtype,
                    num_bits_per_copy=256,
                )
                dest_fp8_ptr = cute.make_ptr(
                    fp8_dtype,
                    dest_row.iterator.toint() + Int64(hidden_col_start),
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                )
                cute.copy(
                    stg_fp8_atom,
                    r_fp8_flat,
                    cute.make_tensor(dest_fp8_ptr, cute.make_layout(32)),
                )
                # Buffer the E8M0 scale; the whole task tile's 8 scales are
                # flushed together by _stg_sf_fc2 (single stg.64 when aligned).
                self._write_sf_fc2_buffer(rmem_sf_fc2, subtile_idx, qpvscale)
            elif cutlass.const_expr(
                self._token_back_by_dispatch and self._combine_mxfp8
            ):
                # MegaMoE token-back-by-dispatch + quantized combine:
                # Epilogue writes fp8 data to local pool; dispatch warps push
                # both data (fc2_output_workspace) and SF (fc2_output_sf) to peers.
                pool_token_global = (
                    work_tile_info.cumulative_data_physical_row
                    + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                    + token_row_in_cta
                )
                fp8_dtype = self._combine_format.act_dtype
                r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
                qpvscale = quant_sfd_row(
                    r_acc,
                    r_fp8,
                    1.0,
                    EpilogueTileN,
                    cutlass.Float8E8M0FNU,
                    fp8_dtype,
                )
                # Write 32 fp8 elements to local fc2_output_workspace pool.
                fp8_byte_addr = (
                    token_comm_args.fc2_output_workspace.iterator.toint()
                    + Int64(pool_token_global) * Int64(self._hidden_fc2)
                    + Int64(hidden_col_start)
                )
                stg_fp8_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    fp8_dtype,
                    num_bits_per_copy=256,
                )
                aligned_fp8_iter = cute.make_ptr(
                    fp8_dtype,
                    fp8_byte_addr,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                )
                r_fp8_flat = cute.make_tensor(
                    r_fp8.iterator, cute.make_layout(EpilogueTileN)
                )
                cute.copy(
                    stg_fp8_atom,
                    r_fp8_flat,
                    cute.make_tensor(aligned_fp8_iter, cute.make_layout(EpilogueTileN)),
                )
                # Buffer the E8M0 scale; flushed together by _stg_sf_fc2 after
                # the subtile loop (single stg.64 when hidden-aligned).
                self._write_sf_fc2_buffer(rmem_sf_fc2, subtile_idx, qpvscale)
            else:
                # BF16 path (default): fp32->bf16, two 256-bit STGs.
                r_bf16 = cute.make_rmem_tensor(r_acc_layout.shape, cutlass.BFloat16)
                r_bf16.store(r_acc.load().to(cutlass.BFloat16))
                stg_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                    num_bits_per_copy=256,
                )
                for stg_half in cutlass.range_constexpr(EpilogueTileN // 16):
                    reg_view = cute.make_tensor(
                        r_bf16.iterator + stg_half * 16,
                        cute.make_layout(16),
                    )
                    if cutlass.const_expr(
                        token_comm_args is not None and not self._token_back_by_dispatch
                    ):
                        metadata_u32 = cute.recast_tensor(
                            token_comm_args.token_src_metadata,
                            cutlass.Uint32,
                        )
                        fc2_output_dest = Fc2OutputDest(
                            tensor=token_comm_args.combine_output,
                            metadata=metadata_u32,
                            peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
                            reduce_topk_in_kernel=self._fc2_in_kernel_topk_reduce,
                        )
                        pool_token_global = (
                            work_tile_info.cumulative_data_physical_row
                            + work_tile_info.tile_m_idx
                            * cutlass.Int32(self._cta_tile_m)
                            + token_row_in_cta
                        )
                        dest_row = fc2_output_dest.resolve_token_row(pool_token_global)
                        hidden_off = hidden_col_start + cutlass.Int32(stg_half * 16)
                        dest_ptr = cute.make_ptr(
                            cutlass.BFloat16,
                            dest_row.iterator.toint() + hidden_off * cutlass.Int64(2),
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        if cutlass.const_expr(self._fc2_in_kernel_topk_reduce):
                            reg_u32 = cute.recast_tensor(reg_view, cutlass.Uint32)
                            for pair in cutlass.range_constexpr(16 // 4):
                                _red_add_relaxed_sys_v2_bf16x2(
                                    dest_ptr + cutlass.Int32(pair * 4),
                                    cutlass.Uint32(reg_u32[pair * 2]),
                                    cutlass.Uint32(reg_u32[pair * 2 + 1]),
                                )
                        else:
                            cute.copy(
                                stg_atom,
                                reg_view,
                                cute.make_tensor(dest_ptr, cute.make_layout(16)),
                            )
                    else:
                        g_fc2_output_tile = cute.local_tile(
                            real_fc2_output,
                            (self._cta_tile_m, EpilogueTileN, 1),
                            (work_tile_info.tile_m_idx, hidden_group, 0),
                        )
                        g_fc2_slice = cute.slice_(g_fc2_output_tile, (None, None, 0))
                        g_thread_row = cute.local_tile(
                            g_fc2_slice,
                            (1, 16),
                            (token_row_in_cta, stg_half),
                        )
                        g_flat = cute.coalesce(g_thread_row)
                        aligned_iter = cute.make_ptr(
                            cutlass.BFloat16,
                            g_flat.iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        cute.copy(
                            stg_atom,
                            reg_view,
                            cute.make_tensor(aligned_iter, g_flat.layout),
                        )

        iket.range_pop()

    @cute.jit
    def _write_sf_fc2_buffer(self, rmem_sf_fc2, subtile_idx, qpvscale) -> None:
        """Scatter one subtile's E8M0 scale into the per-tile SF buffer.

        ``subtile_idx`` is runtime (the overlap-acc walk reverses it on odd
        turns), so index with a const_expr if-chain -- mirrors fc1's rmem_sf.
        """
        for j in cutlass.range_constexpr(self._cta_tile_n // EpilogueTileN):
            if subtile_idx == cutlass.Int32(j):
                rmem_sf_fc2[j] = qpvscale

    @cute.jit
    def _stg_sf_fc2(
        self,
        rmem_sf_fc2: cute.Tensor,
        token_comm_args,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
    ) -> None:
        """Flush a task tile's fc2 E8M0 scales to local ``fc2_output_sf``.

        The SF plane is token-row-major (blocks contiguous, stride 1), so this
        thread's ``fc2_subtile_cnt`` scales land on contiguous bytes at
        ``pool_token * sf_block_pad + tile_n * fc2_subtile_cnt``.  When hidden is
        a whole multiple of ``cta_tile_n`` every N-tile is fully in-bounds, so
        the 8 scales go out as one 64-bit store; otherwise fall back to per-block
        predicated 1-byte stores for the ragged tail tile.

        Token predicate mirrors ``_run_fc2_subtile``'s data STG: thread owns one
        token row, so a single ``token_row_in_cta < valid_tokens`` guards the
        whole flush (all subtiles share the token, only the block index moves).
        """
        fc2_subtile_cnt = self._cta_tile_n // EpilogueTileN
        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        if token_row_in_cta < work_tile_info.valid_tokens_in_cta_tile:
            pool_token_global = (
                work_tile_info.cumulative_data_physical_row
                + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                + token_row_in_cta
            )
            hidden_group_base = work_tile_info.tile_n_idx * cutlass.Int32(
                fc2_subtile_cnt
            )
            sf_byte_addr = (
                token_comm_args.fc2_output_sf.iterator.toint()
                + Int64(pool_token_global) * Int64(self._fc2_sf_block_pad)
                + Int64(hidden_group_base)
            )
            if cutlass.const_expr(self._fc2_sf_batch8):
                stg_e8m0x8_from_f32(
                    sf_byte_addr,
                    rmem_sf_fc2[0],
                    rmem_sf_fc2[1],
                    rmem_sf_fc2[2],
                    rmem_sf_fc2[3],
                    rmem_sf_fc2[4],
                    rmem_sf_fc2[5],
                    rmem_sf_fc2[6],
                    rmem_sf_fc2[7],
                )
            else:
                for j in cutlass.range_constexpr(fc2_subtile_cnt):
                    block_hidden_start = work_tile_info.tile_n_idx * cutlass.Int32(
                        self._cta_tile_n
                    ) + cutlass.Int32(j * EpilogueTileN)
                    if block_hidden_start < valid_hidden:
                        stg_e8m0_from_f32(sf_byte_addr + Int64(j), rmem_sf_fc2[j])

    @cute.jit
    def _run_fc2_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
    ) -> None:
        """fc2 (Linear2) task-tile body following fc1 pattern exactly.

        Key alignment with fc1:
          - acc_stage_col_offset = 0 for overlapping_accum (single physical stage)
          - is_odd_turn determines start subtile (7 odd, 0 even) and direction
          - consumer_release after first subtile (i==0) when overlapping_accum
          - TMEM tensor advances by +/- EpilogueTileN each iteration
        """
        real_fc2_output, _ = sched_ext.get_gmem_tensor(
            "d",
            gmem_fc2_output,
            work_tile_info,
        )
        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_push("mxfp8_fc2_epi_tile")

        # For overlapping_accum: phase selects which TMEM stage holds the result,
        # mirroring the fc1 epilogue formula.  When a cluster processes an fc1 tile
        # before fc2, acc_consumer_state.phase is 1 and the result lives in the
        # second physical stage (col offset 256-num_sf_tmem_cols).  Hardcoding 0
        # reads the wrong stage for phase=1 clusters.
        # For non-overlapping: stage index selects the physical region.
        if cutlass.const_expr(self._overlapping_accum):
            acc_stage_col_offset = cutlass.Int32(acc_consumer_state.phase) * (
                256 - self._num_sf_tmem_cols
            )
        else:
            acc_stage_col_offset = (
                cutlass.Int32(acc_consumer_state.index) * self._cta_tile_n
            )

        fc2_subtile_cnt = self._cta_tile_n // EpilogueTileN  # = 8

        # Start subtile mirrors fc1: last for odd turn, first for even.
        start_subtile = fc2_subtile_cnt - 1 if is_odd_turn else 0
        tmem_t = self._subtile_fc2_tmem_tensor(
            tmem_acc_tensor,
            cutlass.Int32(start_subtile),
            warp_idx,
        )

        # Step direction mirrors fc1 gate/up: +EpilogueTileN (even) or -EpilogueTileN (odd).
        tmem_forward_cols = EpilogueTileN
        if cutlass.const_expr(self._overlapping_accum):
            if is_odd_turn:
                tmem_forward_cols = -EpilogueTileN

        # Quantized combine: buffer the per-subtile E8M0 scales and flush them in
        # one stg.64 after the loop (see _stg_sf_fc2).  Indexed by subtile_idx, so
        # the reversed odd-turn walk fills the same slots.
        if cutlass.const_expr(self._combine_mxfp8 and token_comm_args is not None):
            layout_sf_fc2 = cute.make_layout(fc2_subtile_cnt)
            rmem_sf_fc2 = cute.make_rmem_tensor(layout_sf_fc2.shape, self.acc_dtype)
        else:
            rmem_sf_fc2 = None

        for i in cutlass.range(0, fc2_subtile_cnt, 1, unroll=1):
            if cutlass.const_expr(self._overlapping_accum):
                subtile_idx = cutlass.Int32(i)
                if is_odd_turn:
                    subtile_idx = cutlass.Int32(fc2_subtile_cnt - 1 - i)
            else:
                subtile_idx = cutlass.Int32(i)

            self._run_fc2_subtile(
                subtile_idx=subtile_idx,
                tmem_subtile_tensor=tmem_t,
                real_fc2_output=real_fc2_output,
                work_tile_info=work_tile_info,
                valid_hidden=valid_hidden,
                warp_idx=warp_idx,
                tidx=tidx,
                token_comm_args=token_comm_args,
                rmem_sf_fc2=rmem_sf_fc2,
            )

            if cutlass.const_expr(self._overlapping_accum):
                self._acc_pipeline_consumer_release(
                    acc_pipeline, acc_consumer_state, i == 0
                )

            tmem_t = self._advance_fc2_tmem_tensor(tmem_t, tmem_forward_cols)

        # Release AFTER all subtile reads (never early-release for FC2).
        if not cutlass.const_expr(self._overlapping_accum):
            self._acc_pipeline_consumer_release(acc_pipeline, acc_consumer_state, True)

        # Flush the buffered E8M0 scales (one stg.64 per thread when aligned).
        if cutlass.const_expr(self._combine_mxfp8 and token_comm_args is not None):
            self._stg_sf_fc2(
                rmem_sf_fc2=rmem_sf_fc2,
                token_comm_args=token_comm_args,
                work_tile_info=work_tile_info,
                valid_hidden=valid_hidden,
                warp_idx=warp_idx,
                tidx=tidx,
            )

        iket.range_pop()

    @cute.jit
    def _stg_sf_fc1(
        self,
        rmem_sf_f32: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        work_tile_info,
        tidx,
    ) -> None:
        """Compute gmem SF tile coords and store fc1 scale factors to gmem.

        Predicated on ``valid_tokens_in_cta_tile``: thread ``tidx`` owns the
        CTA-relative token ``tidx``, so padding threads
        (``tidx >= valid_tokens``) must skip the store.  SF is only padded to
        ``SfPaddingBlock`` (128), but a CTA tile spans 128 tokens / a cluster
        block 256, so an over-padded thread's SF GMEM row aliases the START of
        the NEXT expert's SF region -- writing it corrupts that expert's scale
        factors (off-by-one E8M0 exponent -> factor-of-2 output error).  Mirror
        of the predicate in ``tma_store_fc1_output`` for the data store.
        """
        bx, _, _ = cute.arch.block_idx()
        sf_idx = work_tile_info.tile_n_idx
        token_idx = work_tile_info.tile_m_idx * self._cta_tile_m + tidx
        if tidx < work_tile_info.valid_tokens_in_cta_tile:
            sf_base = cute.local_tile(
                real_fc1_output_sf,
                (1, 1, 1),
                (
                    token_idx,
                    sf_idx * cutlass.Int32(Fc1EpilogueOutputTileN),
                    cutlass.Int32(0),
                ),
            )
            # local_tile() with a runtime token_idx drops the pointer's
            # assumed_align to 1 byte (the E8M0 element size), but the region
            # base is 4B-aligned -- reassert that so the 4 bytes coalesce
            # into a single STG.E.U32.
            sf_ptr = cute.make_ptr(
                self.sf_dtype,
                sf_base.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            gmem_sf_f8 = cute.make_tensor(sf_ptr, cute.make_layout(4))
            sf_layout = cute.make_layout(4)
            r_sf_f8 = cute.make_rmem_tensor(sf_layout.shape, self.sf_dtype)
            r_sf_f8.store(rmem_sf_f32.load().to(self.sf_dtype))
            cute.autovec_copy(r_sf_f8, gmem_sf_f8)

    @cute.jit
    def run(
        self,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        sched_consumer,
        sched_ext,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        gmem_fc2_output: cute.Tensor,
        gmem_fc1_done_counter: cute.Tensor,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        token_comm_args=None,
        # generate_c: pass real sC / tma_atom_c / tma_tensor_c when True;
        # pass smem_fc1_output_buffer / tma_atom_fc1_output / gmem_fc1_output as
        # dummies when False — const_expr guards ensure no access.
        smem_c_buffer: cute.Tensor = None,
        tma_atom_c: cute.CopyAtom = None,
        gmem_c: cute.Tensor = None,
    ) -> None:
        """
        Run the full MXFP8 fc1+fc2-fused epilogue task-tile loop.

        ``token_comm_args`` (MegaMoE path) is forwarded to the fc2 task tile so
        the fc2 STG is routed to the source rank's combine output.
        """
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._num_acc_pipeline_stages
        )

        if cutlass.const_expr(self._generate_c):
            _c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=Fc1CTMAStages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    EpiWarpCount * WarpThreadCount,
                ),
            )
        else:
            _c_pipeline = None

        task_tile_boundary_bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=32 * len(self._epilogue_warp_ids),
        )

        valid_hidden = cutlass.Int32(gmem_fc2_output.shape[1])

        # Init=1 (reverse): overlap-acc path walks subtiles N-1, 0, ..., N-2
        # so the overlap-region TMEM cols are released first.
        is_odd_turn = cutlass.Int32(1)

        bidx, bidy, bidz = cute.arch.block_idx()
        work_tile_info = sched_consumer.consume_work()

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            tid=tidx % (len(self._epilogue_warp_ids) * WarpThreadCount),
        )

        while work_tile_info.is_valid_tile:
            acc_stage_index = 0 if is_odd_turn else 1
            tmem_acc_stage_tesnor = tmem_acc_tensor[(None, None, None, acc_stage_index)]

            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                if cutlass.const_expr(self._generate_c):
                    _smem_c_buf = smem_c_buffer
                    _tma_atom_c = tma_atom_c
                    _gmem_c = gmem_c
                else:
                    _smem_c_buf = smem_fc1_output_buffer
                    _tma_atom_c = tma_atom_fc1_output
                    _gmem_c = gmem_fc1_output
                self._run_fc1_task_tile(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_stage_tesnor,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    tma_atom_fc1_output=tma_atom_fc1_output,
                    sched_ext=sched_ext,
                    gmem_fc1_output=gmem_fc1_output,
                    gmem_fc1_output_sf=gmem_fc1_output_sf,
                    gmem_topk_scores=gmem_topk_scores,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    alpha=alpha,
                    norm_const=norm_const,
                    smem_c_buffer=_smem_c_buf,
                    tma_atom_c=_tma_atom_c,
                    gmem_c=_gmem_c,
                    c_pipeline=_c_pipeline,
                )
            else:
                self._run_fc2_task_tile(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_stage_tesnor,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                    sched_ext=sched_ext,
                    gmem_fc2_output=gmem_fc2_output,
                    valid_hidden=valid_hidden,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    token_comm_args=token_comm_args,
                )

            acc_consumer_state.advance()
            if cutlass.const_expr(self._overlapping_accum):
                is_odd_turn = cutlass.Int32(1) - is_odd_turn

            cur_was_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            # Use tile_m_idx // atom_thr_size (= cluster-level token block index)
            # NOT tile_n_idx (= intermediate N-tile index).  Both fc1 N-tiles
            # for the same token block share the same tile_m_idx, so all their
            # increments target the same counter slot.  Using tile_n_idx splits
            # increments across slots and deadlocks fc2's spin-wait.
            cur_fc1_counter_slot = (
                work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_m_idx // cutlass.Int32(self._atom_thr_size)
            )
            cur_fc2_expert_idx = work_tile_info.expert_idx

            work_tile_info = sched_consumer.consume_work()

            # Drain fc1 TMA/STG stores before publishing the fc1-done counter.
            if cur_was_linear1:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.fence_acq_rel_gpu()

            task_tile_boundary_bar.arrive_and_wait()

            if cur_was_linear1:
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc1_batch,
                    (gmem_fc1_done_counter.iterator + cur_fc1_counter_slot).toint(),
                )
            else:
                if cutlass.const_expr(self._token_back_by_dispatch):
                    # Fence before (deferred) counter release: make the fc2
                    # pool-output STG writes device-visible.  The release
                    # atomic in flag_tracker.fire() then signals completion.
                    cute.arch.fence_acq_rel_gpu()
                    fc2_flag_addr = (
                        token_comm_args.fc2_done_counter.iterator + cur_fc2_expert_idx
                    ).toint()
                else:
                    fc2_flag_addr = Int64(0)
                no_fire: cutlass.Constexpr = not self._token_back_by_dispatch
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc2_batch,
                    fc2_flag_addr,
                    no_fire,
                )

        flag_tracker.fire()
