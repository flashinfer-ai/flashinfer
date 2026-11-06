# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owned NT4 adaptation of the matching kernel-team drop file.
# Upstream snapshot: 50d583aa0. See ../README.md for provenance.
# Extracted from research commit 52e7da3f4; GPU parity must be requalified.
"""Fused NT4 clamped-ReLU2, MXFP8 quantization and FC2 epilogue."""

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
from src.token_comm import TokenSrcMetadata

from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase

from common.moe_utils import fmax, quant_sfd_row
from cutlass.cute.typing import Float32
from moe_nvfp4_swapab.epilogue import (
    _TmemTranspose16x32Core,
)

Fc1GateUpInterleave = 32
EpilogueTileN = 32
Fc1EpilogueOutputTileM = 256
Fc1EpilogueOutputTileN = 256
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
        epilog_sync_bar_id: int = 1,
        epilogue_warp_ids: Tuple[int, ...] = (0, 1, 2, 3),
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        epi_flag_batch: Tuple[int, int] = (1, 1),
    ) -> None:
        self._hardware_tanh = False
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
        self._fc2_tile_n = self._cta_tile_n
        self._mma_tiler_k = mma_tiler_mnk[2]
        self._cta_tile_n_sfb = ((mma_tiler_mnk[1] + 127) // 128) * 128
        self._static_expert_shape = static_expert_shape
        if (
            static_expert_shape is not None
            and static_expert_shape[2] % (self._cta_tile_m * cluster_shape_mn[0]) == 0
        ):
            self._fc2_stg_needs_predicate: bool = False
        else:
            self._fc2_stg_needs_predicate = True

        # Non-swap-AB: TMA tile is (EpilogueTileN tokens, Fc1EpilogueOutputTileN intermediates)
        self._epi_tile = (EpilogueTileN, Fc1EpilogueOutputTileN)
        self._subtile_cnt = self._cta_tile_n // EpilogueTileN

        self._overlapping_accum = True

        self._num_acc_stage = 2
        self._num_acc_pipeline_stages = 1

        k = self._mma_tiler_k
        self._num_sfa_tmem_cols = self._cta_tile_m * k // sf_vec_size * 4 // 4 // 128
        self._num_sfb_tmem_cols = (
            self._cta_tile_n_sfb * k // sf_vec_size * 4 // 4 // 128
        )
        self._num_sf_tmem_cols = 32  # self._num_sfa_tmem_cols + self._num_sfb_tmem_cols

        self._num_accumulator_tmem_cols = self._cta_tile_n * self._num_acc_stage - (
            self._num_sf_tmem_cols
        )

        self._iter_acc_early_release = (
            self._num_sf_tmem_cols + EpilogueTileN - 1
        ) // EpilogueTileN

        self._token_back_by_dispatch = False
        self._epi_fc1_batch = max(1, epi_flag_batch[0])
        self._epi_fc2_batch = max(1, epi_flag_batch[1])

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
        subtile_col_off = subtile_idx * EpilogueTileN
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
    ) -> Tuple[cute.Tensor, cute.Tensor]:
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
        if cutlass.const_expr(self.fc1_output_dtype.width == 8):
            real_fc1_output_sf, _ = sched_ext.get_gmem_tensor(
                "sfd", gmem_fc1_output_sf, work_tile_info
            )
        else:
            real_fc1_output_sf = None
        real_topk_scores, _ = sched_ext.get_gmem_tensor(
            "topk", gmem_topk_scores, work_tile_info
        )

        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_push("fc1_epi_tile")

        subtile_cnt = self._subtile_cnt
        # Start subtile: last for odd turn, first for even.
        start_subtile = subtile_cnt - 1 if is_odd_turn else 0
        tmem_gate, tmem_up = self._subtile_local_tmem_tensor_pair(
            tmem_acc_tensor,
            start_subtile,
            warp_idx,
        )
        tmem_forward_cols = Fc1GateUpInterleave
        if is_odd_turn:
            tmem_forward_cols = -Fc1GateUpInterleave

        if cutlass.const_expr(self.fc1_output_dtype.width == 8):
            layout_sf = cute.make_layout(8)
            rmem_sf = cute.make_rmem_tensor(layout_sf.shape, self.acc_dtype)
        else:
            rmem_sf = None

        # ── Set up RMEM→SMEM copy atom (direct CopyUniversalOp) ──
        r2s_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.fc1_output_dtype,
            num_bits_per_copy=128,
        )
        tRS_sC = None  # noqa: F841

        for i in cutlass.range(0, subtile_cnt, 1, unroll=1):
            subtile_idx = cutlass.Int32(i)
            if is_odd_turn:
                subtile_idx = cutlass.Int32(subtile_cnt - 1 - i)

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

        if cutlass.const_expr(self.fc1_output_dtype.width == 8):
            self._stg_sf_fc1(rmem_sf, real_fc1_output_sf, work_tile_info, tidx)

        # ── TMA store: 4 stores (one per warp group) after all subtiles ──
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
    def _swiglu_act(
        self,
        t_swiglu: cute.Tensor,
        t_up: cute.Tensor,
        t_gate: cute.Tensor,
        prob: Optional[Float32] = None,
    ) -> None:
        """Evaluate clamped squared ReLU in FP32 before MXFP8 quantization."""
        # The local plan applies routing weights after FC2 (prob is None).
        for i in cutlass.range_constexpr(cute.size(t_gate)):
            z = fmax(t_gate[i], Float32(0.0))
            c = self.glu_clamp * cute.math.tanh(
                z / self.glu_clamp, approx=self._hardware_tanh
            )
            value = c * c
            if cutlass.const_expr(prob is not None):
                value = value * prob
            t_swiglu[i] = value

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
        """FC1 subtile: clamped squared ReLU, E8M0 scales and FP8 R2S."""
        iket.range_push("fc1_epilogue_subtile")

        r_layout = cute.make_layout((((Fc1GateUpInterleave,), 1),), stride=(((1,), 0),))
        r_gate = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        r_up = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)

        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            self.acc_dtype,
        )
        cute.copy(atom_t2r, tmem_gate_tensor, r_gate)

        self._acc_pipeline_consumer_release(
            acc_pipeline, acc_consumer_state, acc_is_release
        )

        topk = None

        swiglu = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        self._swiglu_act(swiglu, r_up, r_gate, topk)

        c = cute.make_rmem_tensor(r_layout.shape, self.fc1_output_dtype)
        if cutlass.const_expr(self.fc1_output_dtype.width == 8):
            # Quantized hand-off: fp8 data + E8M0 block scale.
            qpvscale = quant_sfd_row(
                swiglu,
                c,
                norm_const,
                self._sf_vec_size,
                self.sf_dtype,
                self.fc1_output_dtype,
            )
            for sf_i in cutlass.range_constexpr(8):
                if subtile_idx == sf_i:
                    rmem_sf[sf_i] = qpvscale
        else:
            # Plain-data hand-off: direct cast to the fc1 output dtype (the
            # fc1_output workspace is reloaded as fc2's A operand).
            c.store(swiglu.load().to(self.fc1_output_dtype))

        thread_in_warp = tidx % WarpThreadCount
        sC_stage = cute.slice_(smem_fc1_output_buffer, (None, None, warp_idx))
        sC_thread_row = cute.local_tile(
            sC_stage, (1, Fc1GateUpInterleave), (thread_in_warp, subtile_idx)
        )
        cute.copy(r2s_copy_atom, cute.coalesce(c), cute.coalesce(sC_thread_row))

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
        """Load FP32 accumulators and store BF16 per-route contributions."""
        iket.range_push("fc2_epilogue_subtile")

        hidden_col_start = work_tile_info.tile_n_idx * cutlass.Int32(
            self._fc2_tile_n
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
                metadata_u32 = cute.recast_tensor(
                    token_comm_args.token_src_metadata,
                    cutlass.Uint32,
                )
                fc2_output_dest = Fc2OutputDest(
                    tensor=token_comm_args.combine_output,
                    metadata=metadata_u32,
                    peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
                    reduce_topk_in_kernel=False,
                )
                pool_token_global = (
                    work_tile_info.cumulative_data_physical_row
                    + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
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
                cute.copy(
                    stg_atom,
                    reg_view,
                    cute.make_tensor(dest_ptr, cute.make_layout(16)),
                )

        iket.range_pop()

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
        iket.range_push("fc2_epi_tile")

        fc2_subtile_cnt = self._fc2_tile_n // EpilogueTileN  # = 8

        # Start subtile mirrors fc1: last for odd turn, first for even.
        start_subtile = fc2_subtile_cnt - 1 if is_odd_turn else 0
        tmem_t = self._subtile_fc2_tmem_tensor(
            tmem_acc_tensor,
            cutlass.Int32(start_subtile),
            warp_idx,
        )

        # Step direction mirrors fc1 gate/up: +EpilogueTileN (even) or -EpilogueTileN (odd).
        tmem_forward_cols = EpilogueTileN
        if is_odd_turn:
            tmem_forward_cols = -EpilogueTileN

        rmem_sf_fc2 = None

        for i in cutlass.range(0, fc2_subtile_cnt, 1, unroll=1):
            subtile_idx = cutlass.Int32(i)
            if is_odd_turn:
                subtile_idx = cutlass.Int32(fc2_subtile_cnt - 1 - i)

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

            self._acc_pipeline_consumer_release(
                acc_pipeline, acc_consumer_state, i == 0
            )

            tmem_t = self._advance_fc2_tmem_tensor(tmem_t, tmem_forward_cols)

        iket.range_pop()

    @cute.jit
    def _stg_sf_fc1(
        self,
        rmem_sf_f32: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        work_tile_info,
        tidx,
    ) -> None:
        # The eight SFs cross two 4-SF atoms; they are NOT eight contiguous
        # bytes in the blocked layout. Resolve each atom through its layout.
        token_idx = work_tile_info.tile_m_idx * self._cta_tile_m + tidx
        if tidx < work_tile_info.valid_tokens_in_cta_tile:
            for sf_i in cutlass.range_constexpr(8):
                sf_cell = cute.local_tile(
                    real_fc1_output_sf,
                    (1, 1, 1),
                    (token_idx, work_tile_info.tile_n_idx * 256 + sf_i * 32, 0),
                )
                sf_cell[0, 0, 0] = rmem_sf_f32[sf_i].to(self.sf_dtype)

    @cute.jit
    def run(
        self,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        sched_consumer,
        sched_ext,
        smem_fc1_output_buffer: Optional[cute.Tensor],
        tma_atom_fc1_output: cute.CopyAtom,
        gmem_fc1_output: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        gmem_fc2_output: cute.Tensor,
        gmem_fc1_done_counter: cute.Tensor,
        warp_idx: int,
        tidx,
        # Quantisation operands; subclasses without an SF plane leave them
        # None (their fc1 task-tile overrides ignore the forwarded values).
        gmem_fc1_output_sf: Optional[cute.Tensor] = None,
        alpha=None,
        norm_const=None,
        token_comm_args=None,
        smem_c_buffer: cute.Tensor = None,
        tma_atom_c: cute.CopyAtom = None,
        gmem_c: cute.Tensor = None,
    ) -> None:
        """
        Run the full fc1+fc2-fused epilogue task-tile loop.

        ``token_comm_args`` (MegaMoE path) is forwarded to the fc2 task tile so
        the fc2 STG is routed to the source rank's combine output.
        """
        # Trace-time contract: ``smem_fc1_output_buffer=None`` is legal only
        # for the direct-STG fc1 path — every dereference below sits behind a
        # ``const_expr(use_stg_fc1)`` gate keyed on the same flag, so the TMA
        # branches are never traced when the buffer is absent.  A new,
        # ungated use would only fail in the use_stg_fc1=True compile; this
        # assert pins the failure to the entry point instead.
        assert smem_fc1_output_buffer is not None
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._num_acc_pipeline_stages
        )

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
            # Overlap-acc: the stage parity follows the odd-turn flip.
            acc_stage_index = 0 if is_odd_turn else 1
            tmem_acc_stage_tesnor = tmem_acc_tensor[(None, None, None, acc_stage_index)]

            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
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
                fc2_flag_addr = Int64(0)
                no_fire: cutlass.Constexpr = not self._token_back_by_dispatch
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc2_batch,
                    fc2_flag_addr,
                    no_fire,
                )

        flag_tracker.fire()
