# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Autonomous SM90 WGMMA epilogue for the fused fc1+fc2 MegaMoE kernel."""

from typing import Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from src.iket_compat import iket
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils

from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, Int32 as _epi_Int32, Int64

from src.flag_batch import GpuReleaseFlagBatchTracker
from src.ptx_helpers import red_add_relaxed_sys_v2_bf16x2
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase
from common.megamoe_constants import (
    Fp8E4M3RcpLimit,
    Fp8Fc2ActivationScaleK,
    Fp8GateUpInterleave,
)

from common.moe_utils import fmax
from cutlass.cute.typing import Float32
from moe_hopper_fp8.epilogue_fp8_common import (
    Fc2OutputDest,
    consume_initial_pingpong_work,
    consume_next_pingpong_work,
    clamp_and_swiglu_sm90,
    stg_fc1_block_scale_row,
    tma_store_fc1_output,
    tma_store_fc2_output,
)

Fc1EpilogueOutputTileN = 128
Fc1EpilogueStoreTileN = 64
# Supported values: 8, 16, 32, 64.
Fc1SubtileN = 32
Fc1GroupsPerSubtile = Fc1SubtileN // Fp8GateUpInterleave
Fc1SubtileRegs = Fc1GroupsPerSubtile * 4
Fc1BlockwiseSubtileN = Fc1EpilogueStoreTileN
Fc1BlockwiseSubtileRegs = (
    Fc1BlockwiseSubtileN // Fp8GateUpInterleave
) * 4
Fc2SubtileN = 32
Fc1SubtilesPerHalf = Fc1EpilogueStoreTileN // Fc1SubtileN
WarpThreadCount = 32
EpiWarpCount = 4
# M=128 is implemented and functionally correct, but remains disabled because
# it causes severe register spilling, especially in Mega/blockwise kernels.
NonSwapTileMChoices = (64,)
NonSwapTileNChoices = (128, 256)

# =============================================================================
# Fp8GluEpilogue
# =============================================================================

class Fp8GluEpilogue:

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
        fp8_scale_mode: str = "per_tensor",
        fp8_output_rcp_limit: float = Fp8E4M3RcpLimit,
        glu_clamp: Optional[float] = None,
        epilog_sync_bar_id: int = 1,
        fc1_store_sync_bar_id: int = 2,
        epilogue_warp_ids: Tuple[int, ...] = (0, 1, 2, 3),
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        fc2_in_kernel_topk_reduce: bool = False,
        apply_topk_in_fc1: bool = False,
        token_back_by_dispatch: bool = False,
        use_fc2_tma_store: bool = False,
        fc1_store_pipeline_stages: int = 1,
        fc2_store_pipeline_stages: int = 1,
        epi_flag_batch: Union[int, Tuple[int, int]] = 1,
        pingpong: bool = False,
    ) -> None:
        self.fc1_output_dtype = fc1_output_dtype
        self.fc1_output_layout = fc1_output_layout
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        if fp8_scale_mode not in ("per_tensor", "blockwise"):
            raise ValueError(
                f"fp8_scale_mode must be 'per_tensor' or 'blockwise', "
                f"got {fp8_scale_mode!r}."
            )
        self.fp8_scale_mode = fp8_scale_mode
        self.fp8_output_rcp_limit = cutlass.Float32(fp8_output_rcp_limit)
        self._sf_vec_size = sf_vec_size
        self._epilog_sync_bar_id = epilog_sync_bar_id
        self._fc1_store_sync_bar_id = fc1_store_sync_bar_id
        self._epilogue_warp_ids = epilogue_warp_ids
        self._pingpong = pingpong
        self._pingpong_order = pingpong
        self._use_2cta_instrs = use_2cta_instrs
        self._cluster_m = cluster_shape_mn[0]

        self._atom_thr_size = 2 if use_2cta_instrs else 1
        self._cta_tile_m = mma_tiler_mnk[0] // self._atom_thr_size
        self._cta_tile_n = mma_tiler_mnk[1]
        self._mma_tiler_k = mma_tiler_mnk[2]
        self._cta_tile_n_sfb = ((mma_tiler_mnk[1] + 127) // 128) * 128
        self._static_expert_shape = static_expert_shape
        if (
            static_expert_shape is not None
            and static_expert_shape[2]
            % (self._cta_tile_n * cluster_shape_mn[1])
            == 0
        ):
            self._fc2_stg_needs_predicate: bool = False
        else:
            self._fc2_stg_needs_predicate: bool = True

        self._fc1_token_tile_m = self._cta_tile_m // EpiWarpCount
        if self._fc1_token_tile_m not in (16, 32):
            raise ValueError(
                "Hopper FP8 epilogue expects each epilogue warp to own 16 or "
                f"32 tokens, got cta_tile_m={self._cta_tile_m}, "
                f"epilogue_warps={EpiWarpCount}."
            )
        self._accum_fragment_tile_m = 64
        self._m64_fragment_count = (
            self._cta_tile_m // self._accum_fragment_tile_m
        )
        self._accum_regs_per_m64 = (
            Fc1EpilogueOutputTileN // 2  # n // 8 * 4 = n // 2
        )

        if len(self._epilogue_warp_ids) % EpiWarpCount != 0:
            raise ValueError(
                "Hopper FP8 epilogue warp count must be a multiple of "
                f"{EpiWarpCount}, got {len(self._epilogue_warp_ids)}."
            )
        self._epilogue_warpgroup_count = (
            len(self._epilogue_warp_ids) // EpiWarpCount
        )

        # Each WGMMA/epilogue warpgroup owns one raw N=128 slice, which folds
        # to one N=64 FC1 output stage. N=128/N=256 therefore use one/two
        # WG-private stages respectively.
        self._fc1_output_tile_n = self._cta_tile_n // 2
        if self._fc1_output_tile_n % Fc1EpilogueStoreTileN != 0:
            raise ValueError(
                "Hopper FP8 FC1 output tile must be divisible by the per-WG "
                f"store tile N={Fc1EpilogueStoreTileN}; got "
                f"fc1_output_tile_n={self._fc1_output_tile_n}."
            )
        self._fc1_store_n_splits = self._fc1_output_tile_n // Fc1EpilogueStoreTileN
        if self._fc1_store_n_splits not in (1, 2):
            raise ValueError(
                "Hopper FP8 split-N FC1 store expects one or two WG-local "
                f"N slices, got {self._fc1_store_n_splits}."
            )
        expected_warpgroup_count = (
            2 if self._pingpong else self._fc1_store_n_splits
        )
        if expected_warpgroup_count != self._epilogue_warpgroup_count:
            raise ValueError(
                "Hopper FP8 epilogue warpgroup count does not match the "
                f"selected scheduling mode; got {self._epilogue_warpgroup_count}, "
                f"expected {expected_warpgroup_count}."
            )
        self._epi_tile = (self._cta_tile_m, Fc1EpilogueStoreTileN)
        self._fc2_epi_tile = (self._accum_fragment_tile_m, Fc2SubtileN)
        self._subtile_cnt = self._epilogue_warpgroup_count

        # SM90 WGMMA keeps accumulators in RMEM; no external accumulator handoff.
        self._num_acc_stage = 1
        self._num_acc_pipeline_stages = 1

        self._iter_acc_early_release = 0

        self._fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self._apply_topk_in_fc1 = apply_topk_in_fc1
        self._token_back_by_dispatch = token_back_by_dispatch
        self._use_fc2_tma_store = use_fc2_tma_store
        for name, stages in (
            ("fc1", fc1_store_pipeline_stages),
            ("fc2", fc2_store_pipeline_stages),
        ):
            if stages not in (1, 2):
                raise ValueError(
                    f"Hopper FP8 {name} store pipeline supports one or two "
                    f"stages, got {stages}."
                )
        self._fc1_store_pipeline_stages = fc1_store_pipeline_stages
        self._fc2_store_pipeline_stages = fc2_store_pipeline_stages
        self._store_buffer_stages = max(
            fc1_store_pipeline_stages, fc2_store_pipeline_stages,
        )
        if isinstance(epi_flag_batch, tuple):
            self._epi_fc1_batch = max(1, epi_flag_batch[0])
            self._epi_fc2_batch = max(1, epi_flag_batch[1])
        else:
            self._epi_fc1_batch = max(1, epi_flag_batch)
            self._epi_fc2_batch = max(1, epi_flag_batch)

        self.glu_clamp = (
            cutlass.Float32(glu_clamp) if glu_clamp is not None else None
        )

    # -- Codegen-time queries  --

    @property
    def epi_tile(self) -> Tuple[int, int]:
        return self._epi_tile

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
    def store_buffer_stages(self) -> int:
        return self._store_buffer_stages

    @property
    def subtile_cnt(self) -> int:
        return self._subtile_cnt

    @property
    def cta_tile_n(self) -> int:
        return self._cta_tile_n

    def staged_smem_layout(
        self,
        n_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        return sm90_utils.make_smem_layout_epi(
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
    def fc2_epi_tile(self) -> Tuple[int, int]:
        return self._fc2_epi_tile

    def fc2_staged_smem_layout(
        self,
        n_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        return sm90_utils.make_smem_layout_epi(
            cutlass.BFloat16,
            utils.LayoutEnum.ROW_MAJOR,
            self._fc2_epi_tile,
            n_stages,
        )

    @property
    def fc2_smem_layout_one_stage(
        self,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        staged = self.fc2_staged_smem_layout(1)
        return cute.select(staged, mode=[0, 1])

    @property
    def fc2_bytes_per_stage(self) -> int:
        return cute.size_in_bytes(
            cutlass.BFloat16, self.fc2_smem_layout_one_stage,
        )


    # -- FC1 epilogue consuming SM90 WGMMA accumulators --
    @cute.jit
    def _run_fc1_epilogue(
        self,
        work_tile_info,
        accumulators: cute.Tensor,
        n_half: cutlass.Constexpr,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        local_warp_idx: int,
        tidx,
        _iket_active,
        fc1_act_weight_dequant_scale,
        fc2_act_dequant_scale,
        norm_const,
        store_stage_idx,
    ) -> None:
        """Dispatch the FC1 epilogue for one completed WGMMA task tile."""
        if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
            self._run_fc1_epilogue_blockwise(
                work_tile_info=work_tile_info,
                accumulators=accumulators,
                n_half=n_half,
                smem_fc1_output_buffer=smem_fc1_output_buffer,
                tma_atom_fc1_output=tma_atom_fc1_output,
                sched_ext=sched_ext,
                gmem_fc1_output=gmem_fc1_output,
                gmem_fc1_output_sf=gmem_fc1_output_sf,
                gmem_topk_scores=gmem_topk_scores,
                local_warp_idx=local_warp_idx,
                tidx=tidx,
                _iket_active=_iket_active,
                store_stage_idx=store_stage_idx,
            )
        else:
            self._run_fc1_epilogue_per_tensor(
                work_tile_info=work_tile_info,
                accumulators=accumulators,
                n_half=n_half,
                smem_fc1_output_buffer=smem_fc1_output_buffer,
                tma_atom_fc1_output=tma_atom_fc1_output,
                sched_ext=sched_ext,
                gmem_fc1_output=gmem_fc1_output,
                gmem_fc1_output_sf=gmem_fc1_output_sf,
                gmem_topk_scores=gmem_topk_scores,
                local_warp_idx=local_warp_idx,
                tidx=tidx,
                _iket_active=_iket_active,
                fc1_act_weight_dequant_scale=fc1_act_weight_dequant_scale,
                fc2_act_dequant_scale=fc2_act_dequant_scale,
                norm_const=norm_const,
                store_stage_idx=store_stage_idx,
            )

    @cute.jit
    def _run_fc1_epilogue_per_tensor(
        self,
        work_tile_info,
        accumulators: cute.Tensor,
        n_half: cutlass.Constexpr,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        local_warp_idx: int,
        tidx,
        _iket_active,
        fc1_act_weight_dequant_scale,
        fc2_act_dequant_scale,
        norm_const,
        store_stage_idx,
    ) -> None:
        """FC1 per-tensor epilogue: scalar scale + FP8 store."""
        real_fc1_output, _    = sched_ext.get_gmem_tensor("d",    gmem_fc1_output,    work_tile_info)
        real_topk_scores, _ = sched_ext.get_gmem_tensor(
            "topk", gmem_topk_scores, work_tile_info,
        )

        # Inner epilogue IKET range for non-swap-AB, FP8 per-tensor FC1.
        # WGMMA has already finished before entry. This range covers reading
        # its accumulators, gate/up fold, scalar dequantization, SwiGLU,
        # per-tensor requantization, and RMEM-to-SMEM stores for this n_half
        # epilogue WG. The final SMEM-to-GMEM TMA store issue is outside this
        # range but remains inside the enclosing nswap_fc1_task_* range;
        # its later async completion wait is outside the task-tile range.
        if _iket_active:
            iket.range_push("nswap_fc1_epi_m64n64_pt")

        # Each call writes one FP8 pair from one M64 accumulator fragment.
        r2s_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.fc1_output_dtype,
            num_bits_per_copy=16,
        )

        subtile_begin = n_half * Fc1SubtilesPerHalf
        subtile_end = subtile_begin + Fc1SubtilesPerHalf
        token_tile_base = work_tile_info.tile_m_idx * cutlass.Int32(
            self._cta_tile_m
        )
        for m_sub in cutlass.range_constexpr(self._m64_fragment_count):
            thread_in_warp = tidx % WarpThreadCount
            lane_group = thread_in_warp // 4
            token_row0 = cutlass.Int32(
                m_sub * self._accum_fragment_tile_m + local_warp_idx * 16
            ) + lane_group
            token_row1 = token_row0 + cutlass.Int32(8)
            topk_score0 = Float32(1.0)
            topk_score1 = Float32(1.0)
            if cutlass.const_expr(self._apply_topk_in_fc1):
                topk_score0 = Float32(
                    real_topk_scores[token_tile_base + token_row0]
                )
                topk_score1 = Float32(
                    real_topk_scores[token_tile_base + token_row1]
                )
            for subtile_idx in cutlass.range_constexpr(
                subtile_begin, subtile_end, 1
            ):
                self._run_fc1_epilogue_subtile_m64_per_tensor(
                    subtile_idx=subtile_idx,
                    n_half=n_half,
                    m_sub=m_sub,
                    accumulators=accumulators,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    r2s_copy_atom=r2s_copy_atom,
                    local_warp_idx=local_warp_idx,
                    tidx=tidx,
                    _iket_active=_iket_active,
                    fc1_act_weight_dequant_scale=fc1_act_weight_dequant_scale,
                    fc2_act_dequant_scale=fc2_act_dequant_scale,
                    topk_score0=topk_score0,
                    topk_score1=topk_score1,
                    store_stage_idx=store_stage_idx,
                )

        # FP8 per-tensor WGMMA path does not consume per-row E8M0 scales.
        # Leave the legacy SF workspace at its zero-initialized placeholder;
        # SM90 does not support the SM100-only f32->e8m0 pack intrinsic.

        if _iket_active:
            iket.range_pop()  # nswap_fc1_epi_m64n64_pt

    @cute.jit
    def _quad_reduce_max(self, val: Float32) -> Float32:
        val = fmax(val, cute.arch.shuffle_sync_bfly(val, offset=1))
        val = fmax(val, cute.arch.shuffle_sync_bfly(val, offset=2))
        return val

    @cute.jit
    def _fill_fc1_swiglu_subtile_m64_blockwise(
        self,
        subtile_idx: cutlass.Constexpr,
        n_half: cutlass.Constexpr,
        m_sub: cutlass.Constexpr,
        accumulators: cute.Tensor,
        swiglu: cute.Tensor,
    ) -> None:
        local_subtile_idx = subtile_idx - n_half * Fc1SubtilesPerHalf
        output_group_base = local_subtile_idx * Fc1GroupsPerSubtile
        r_layout = cute.make_layout(
            (((Fc1SubtileRegs,), 1),), stride=(((1,), 0),)
        )
        r_gate = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        r_up = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        accum_fragment_base = m_sub * self._accum_regs_per_m64
        for rep in cutlass.range_constexpr(Fc1GroupsPerSubtile):
            output_group = output_group_base + rep
            gate_reg_base = accum_fragment_base + output_group * 8
            up_reg_base = gate_reg_base + 4
            for j in cutlass.range_constexpr(4):
                r_gate[rep * 4 + j] = accumulators[gate_reg_base + j]
                r_up[rep * 4 + j] = accumulators[up_reg_base + j]

        clamp_and_swiglu_sm90(
            swiglu, r_up, r_gate, self.glu_clamp, Float32(1.0)
        )

    @cute.jit
    def _accum_fc1_swiglu_absmax_blockwise(
        self,
        swiglu: cute.Tensor,
        row0_absmax: Float32,
        row1_absmax: Float32,
    ):
        for rep in cutlass.range_constexpr(cute.size(swiglu) // 4):
            base = rep * 4
            row0_absmax = fmax(
                row0_absmax,
                fmax(
                    fmax(swiglu[base + 0], -swiglu[base + 0]),
                    fmax(swiglu[base + 1], -swiglu[base + 1]),
                ),
            )
            row1_absmax = fmax(
                row1_absmax,
                fmax(
                    fmax(swiglu[base + 2], -swiglu[base + 2]),
                    fmax(swiglu[base + 3], -swiglu[base + 3]),
                ),
            )
        return row0_absmax, row1_absmax

    @cute.jit
    def _scale_fc1_swiglu_for_fp8_blockwise(
        self,
        swiglu: cute.Tensor,
        row0_scale: Float32,
        row1_scale: Float32,
    ) -> None:
        row0_rcp = Float32(1.0) / row0_scale
        row1_rcp = Float32(1.0) / row1_scale
        for rep in cutlass.range_constexpr(cute.size(swiglu) // 4):
            base = rep * 4
            swiglu[base + 0] = swiglu[base + 0] * row0_rcp
            swiglu[base + 1] = swiglu[base + 1] * row0_rcp
            swiglu[base + 2] = swiglu[base + 2] * row1_rcp
            swiglu[base + 3] = swiglu[base + 3] * row1_rcp

    @cute.jit
    def _apply_fc1_topk_nonswap(
        self,
        swiglu: cute.Tensor,
        topk_score0: Float32,
        topk_score1: Float32,
    ) -> None:
        """Scale the two token rows represented by each WGMMA register quad."""
        if cutlass.const_expr(self._apply_topk_in_fc1):
            for rep in cutlass.range_constexpr(cute.size(swiglu) // 4):
                base = rep * 4
                swiglu[base + 0] = swiglu[base + 0] * topk_score0
                swiglu[base + 1] = swiglu[base + 1] * topk_score0
                swiglu[base + 2] = swiglu[base + 2] * topk_score1
                swiglu[base + 3] = swiglu[base + 3] * topk_score1


    @cute.jit
    def _store_fc1_swiglu_subtile_m64(
        self,
        subtile_idx: cutlass.Constexpr,
        n_half: cutlass.Constexpr,
        m_sub: cutlass.Constexpr,
        swiglu: cute.Tensor,
        smem_fc1_output_buffer: cute.Tensor,
        r2s_copy_atom: cute.CopyAtom,
        local_warp_idx: int,
        tidx,
        store_stage_idx,
    ) -> None:
        c = cute.make_rmem_tensor(swiglu.layout.shape, self.fc1_output_dtype)
        c.store(swiglu.load().to(self.fc1_output_dtype))

        thread_in_warp = tidx % WarpThreadCount
        lane_group = thread_in_warp // 4
        lane_mod = thread_in_warp % 4
        local_subtile_idx = subtile_idx - n_half * Fc1SubtilesPerHalf
        sC_stage = cute.slice_(smem_fc1_output_buffer, (None, None, store_stage_idx))
        row_base = cutlass.Int32(
            m_sub * self._accum_fragment_tile_m + local_warp_idx * 16
        )

        for rep in cutlass.range_constexpr(Fc1GroupsPerSubtile):
            output_group = local_subtile_idx * Fc1GroupsPerSubtile + rep
            col_pair = (
                cutlass.Int32(output_group * Fp8GateUpInterleave)
                + lane_mod * cutlass.Int32(2)
            )
            col_pair_tile = col_pair // cutlass.Int32(2)
            row0_regs = cute.make_tensor(
                c.iterator + rep * 4,
                cute.make_layout(2),
            )
            row1_regs = cute.make_tensor(
                c.iterator + rep * 4 + 2,
                cute.make_layout(2),
            )
            s_row0 = cute.local_tile(
                sC_stage, (1, 2), (row_base + lane_group, col_pair_tile)
            )
            s_row1 = cute.local_tile(
                sC_stage,
                (1, 2),
                (row_base + lane_group + cutlass.Int32(8), col_pair_tile),
            )
            cute.copy(r2s_copy_atom, cute.coalesce(row0_regs), cute.coalesce(s_row0))
            cute.copy(r2s_copy_atom, cute.coalesce(row1_regs), cute.coalesce(s_row1))

    @cute.jit
    def _run_fc1_epilogue_blockwise(
        self,
        work_tile_info,
        accumulators: cute.Tensor,
        n_half: cutlass.Constexpr,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        local_warp_idx: int,
        tidx,
        _iket_active,
        store_stage_idx,
    ) -> None:
        """FC1 blockwise epilogue: per-token/per-64 scale + FP8 store."""
        # Inner epilogue IKET range for non-swap-AB, DeepGEMM-style blockwise
        # FC1. WGMMA and blockwise accumulator scaling have already finished.
        # This covers gate/up fold, SwiGLU, per-token amax reduction, scale
        # publication, blockwise requantization, and RMEM-to-SMEM stores.
        # The final SMEM-to-GMEM TMA store is measured by the enclosing
        # nswap_fc1_task_bw range.
        # It deliberately surrounds the m_sub loop so one event is this
        # n_half epilogue-WG tile, rather than one individual subtile.
        if _iket_active:
            iket.range_push("nswap_fc1_epi_m64n64_bw")
        real_topk_scores, _ = sched_ext.get_gmem_tensor(
            "topk", gmem_topk_scores, work_tile_info,
        )

        r_layout = cute.make_layout(
            (((Fc1BlockwiseSubtileRegs,), 1),), stride=(((1,), 0),)
        )
        subtile_r_layout = cute.make_layout(
            (((Fc1SubtileRegs,), 1),), stride=(((1,), 0),)
        )
        subtile_begin = n_half * Fc1SubtilesPerHalf
        subtile_end = subtile_begin + Fc1SubtilesPerHalf
        scale_epsilon = Float32(1.0e-30)
        thread_in_warp = tidx % WarpThreadCount
        lane_group = thread_in_warp // 4
        lane_mod = thread_in_warp % 4
        scale_col_idx = (
            work_tile_info.tile_n_idx * cutlass.Int32(self._fc1_store_n_splits)
            + cutlass.Int32(n_half)
        )
        global_token_base = (
            work_tile_info.cumulative_data_physical_row
            + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
        )
        token_tile_base = work_tile_info.tile_m_idx * cutlass.Int32(
            self._cta_tile_m
        )
        r2s_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.fc1_output_dtype,
            num_bits_per_copy=16,
        )

        for m_sub in cutlass.range_constexpr(self._m64_fragment_count):
            token_row0 = (
                cutlass.Int32(
                    m_sub * self._accum_fragment_tile_m + local_warp_idx * 16
                )
                + lane_group
            )
            token_row1 = token_row0 + cutlass.Int32(8)
            topk_score0 = Float32(1.0)
            topk_score1 = Float32(1.0)
            if cutlass.const_expr(self._apply_topk_in_fc1):
                topk_score0 = Float32(
                    real_topk_scores[token_tile_base + token_row0]
                )
                topk_score1 = Float32(
                    real_topk_scores[token_tile_base + token_row1]
                )
            swiglu = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
            for subtile_idx in cutlass.range_constexpr(
                subtile_begin, subtile_end, 1
            ):
                local_subtile_idx = subtile_idx - subtile_begin
                swiglu_subtile = cute.make_tensor(
                    swiglu.iterator + local_subtile_idx * Fc1SubtileRegs,
                    subtile_r_layout,
                )
                self._fill_fc1_swiglu_subtile_m64_blockwise(
                    subtile_idx=subtile_idx,
                    n_half=n_half,
                    m_sub=m_sub,
                    accumulators=accumulators,
                    swiglu=swiglu_subtile,
                )

            self._apply_fc1_topk_nonswap(
                swiglu, topk_score0, topk_score1,
            )

            row0_absmax = Float32(0.0)
            row1_absmax = Float32(0.0)
            row0_absmax, row1_absmax = self._accum_fc1_swiglu_absmax_blockwise(
                swiglu, row0_absmax, row1_absmax
            )
            row0_absmax = self._quad_reduce_max(row0_absmax)
            row1_absmax = self._quad_reduce_max(row1_absmax)
            row0_scale = fmax(
                row0_absmax * self.fp8_output_rcp_limit, scale_epsilon
            )
            row1_scale = fmax(
                row1_absmax * self.fp8_output_rcp_limit, scale_epsilon
            )

            if lane_mod == cutlass.Int32(0):
                if token_row0 < work_tile_info.valid_tokens_in_cta_tile:
                    stg_fc1_block_scale_row(
                        gmem_fc1_output_sf,
                        scale_col_idx,
                        global_token_base + token_row0,
                        row0_scale,
                    )
                if token_row1 < work_tile_info.valid_tokens_in_cta_tile:
                    stg_fc1_block_scale_row(
                        gmem_fc1_output_sf,
                        scale_col_idx,
                        global_token_base + token_row1,
                        row1_scale,
                    )

            self._scale_fc1_swiglu_for_fp8_blockwise(
                swiglu, row0_scale, row1_scale
            )
            for subtile_idx in cutlass.range_constexpr(
                subtile_begin, subtile_end, 1
            ):
                local_subtile_idx = subtile_idx - subtile_begin
                swiglu_subtile = cute.make_tensor(
                    swiglu.iterator + local_subtile_idx * Fc1SubtileRegs,
                    subtile_r_layout,
                )
                self._store_fc1_swiglu_subtile_m64(
                    subtile_idx=subtile_idx,
                    n_half=n_half,
                    m_sub=m_sub,
                    swiglu=swiglu_subtile,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    r2s_copy_atom=r2s_copy_atom,
                    local_warp_idx=local_warp_idx,
                    tidx=tidx,
                    store_stage_idx=store_stage_idx,
                )

        if _iket_active:
            iket.range_pop()  # nswap_fc1_epi_m64n64_bw

    @cute.jit
    def _store_fc1_task_tile(
        self,
        work_tile_info,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        fc1_store_pipeline,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        local_warp_idx: int,
        n_half: cutlass.Constexpr,
        store_group_idx,
        store_stage_idx,
        _iket_active,
    ) -> None:
        real_fc1_output, _ = sched_ext.get_gmem_tensor(
            "d", gmem_fc1_output, work_tile_info,
        )

        # ── TMA store: one WG-local N=64 output slice, one CTA-M store ──
        output_n_tile = (
            work_tile_info.tile_n_idx * cutlass.Int32(self._fc1_store_n_splits)
            + cutlass.Int32(n_half)
        )

        cute.arch.fence_proxy("async.shared", space="cta")
        fc1_store_bar = pipeline.NamedBarrier(
            barrier_id=self._fc1_store_sync_bar_id + store_group_idx,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        if _iket_active:
            iket.range_push("nswap_fc1_store_smem_barrier")
        fc1_store_bar.arrive_and_wait()
        if _iket_active:
            iket.range_pop()

        if local_warp_idx == cutlass.Int32(0):
            g_fc1_output_wg_view = cute.local_tile(
                real_fc1_output,
                (self._cta_tile_m, Fc1EpilogueStoreTileN, 1),
                (work_tile_info.tile_m_idx, output_n_tile, 0),
            )
            if _iket_active:
                iket.range_push("nswap_fc1_tma_store_issue")
            tma_store_fc1_output(
                smem_fc1_output_buffer,
                store_stage_idx,
                tma_atom_fc1_output,
                g_fc1_output_wg_view,
                work_tile_info.valid_tokens_in_cta_tile,
            )
            fc1_store_pipeline.producer_commit()
            fc1_store_pipeline.producer_acquire()
            if _iket_active:
                iket.range_pop()

    @cute.jit
    def _run_fc1_epilogue_subtile_m64_per_tensor(
        self,
        subtile_idx: cutlass.Constexpr,
        n_half: cutlass.Constexpr,
        m_sub: cutlass.Constexpr,
        accumulators: cute.Tensor,
        smem_fc1_output_buffer: cute.Tensor,
        r2s_copy_atom: cute.CopyAtom,
        local_warp_idx: int,
        tidx,
        _iket_active,
        fc1_act_weight_dequant_scale,
        fc2_act_dequant_scale,
        topk_score0: Float32,
        topk_score1: Float32,
        store_stage_idx,
    ) -> None:
        """M=64 FC1 subtile consuming SM90 WGMMA RMEM accumulator fragment."""
        # Nested IKET range: one non-swap-AB, per-tensor M64xN32 FC1 subtile.
        # It isolates accumulator read, gate/up fold, SwiGLU, scalar
        # requantization, and the final RMEM-to-SMEM copy.
        if _iket_active:
            iket.range_push("nswap_fc1_epi_m64n32_pt")

        r_layout = cute.make_layout(
            (((Fc1SubtileRegs,), 1),), stride=(((1,), 0),)
        )
        r_gate = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        r_up = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)

        local_subtile_idx = subtile_idx - n_half * Fc1SubtilesPerHalf
        output_group_base = local_subtile_idx * Fc1GroupsPerSubtile
        accum_fragment_base = m_sub * self._accum_regs_per_m64
        for rep in cutlass.range_constexpr(Fc1GroupsPerSubtile):
            output_group = output_group_base + rep
            gate_reg_base = accum_fragment_base + output_group * 8
            up_reg_base = gate_reg_base + 4
            for j in cutlass.range_constexpr(4):
                r_gate[rep * 4 + j] = accumulators[gate_reg_base + j]
                r_up[rep * 4 + j] = accumulators[up_reg_base + j]

        for i in cutlass.range_constexpr(cute.size(r_up)):
            r_gate[i] = r_gate[i] * fc1_act_weight_dequant_scale
            r_up[i] = r_up[i] * fc1_act_weight_dequant_scale

        swiglu = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        clamp_and_swiglu_sm90(
            swiglu, r_up, r_gate, self.glu_clamp, Float32(1.0)
        )
        self._apply_fc1_topk_nonswap(
            swiglu, topk_score0, topk_score1,
        )

        fc1_output_rcp = Float32(1.0) / fc2_act_dequant_scale
        for i in cutlass.range_constexpr(cute.size(swiglu)):
            swiglu[i] = swiglu[i] * fc1_output_rcp

        self._store_fc1_swiglu_subtile_m64(
            subtile_idx=subtile_idx,
            n_half=n_half,
            m_sub=m_sub,
            swiglu=swiglu,
            smem_fc1_output_buffer=smem_fc1_output_buffer,
            r2s_copy_atom=r2s_copy_atom,
            local_warp_idx=local_warp_idx,
            tidx=tidx,
            store_stage_idx=store_stage_idx,
        )

        if _iket_active:
            iket.range_pop()  # nswap_fc1_epi_m64n32_pt

    @cute.jit
    def _run_fc2_epilogue_subtile_m64(
        self,
        subtile_idx: cutlass.Constexpr,
        n_half: cutlass.Constexpr,
        m_sub: cutlass.Constexpr,
        accumulators: cute.Tensor,
        real_fc2_output: cute.Tensor,
        smem_fc2_output_buffer: cute.Tensor,
        work_tile_info,
        valid_hidden,
        local_warp_idx: int,
        tidx,
        _iket_active,
        token_comm_args=None,
        fc2_act_weight_dequant_scale=Float32(1.0),
        store_stage_idx=cutlass.Int32(0),
    ) -> None:
        """M=64 FC2 subtile consuming SM90 WGMMA RMEM accumulator fragment."""
        # Nested IKET range: non-swap-AB M64xN32 FC2 conversion/store. It is
        # shared by per-tensor and blockwise modes; blockwise scaling was
        # already applied during WGMMA, so this covers conversion and store.
        if _iket_active:
            iket.range_push("nswap_fc2_epi_m64n32")

        fc2_subtile_cnt = self._cta_tile_n // Fc2SubtileN  # = 8
        hidden_group = (
            work_tile_info.tile_n_idx * cutlass.Int32(fc2_subtile_cnt)
            + cutlass.Int32(subtile_idx)
        )
        hidden_col_start = (
            work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n)
            + cutlass.Int32(subtile_idx * Fc2SubtileN)
        )
        r_acc_layout = cute.make_layout((((16,), 1),), stride=(((1,), 0),))
        r_acc = cute.make_rmem_tensor(r_acc_layout.shape, self.acc_dtype)

        local_subtile_idx = subtile_idx - n_half * 4
        group_base = local_subtile_idx * (Fc2SubtileN // 8)
        accum_fragment_base = m_sub * self._accum_regs_per_m64
        for rep in cutlass.range_constexpr(Fc2SubtileN // 8):
            acc_reg_base = accum_fragment_base + (group_base + rep) * 4
            for j in cutlass.range_constexpr(4):
                r_acc[rep * 4 + j] = accumulators[acc_reg_base + j]

        for i in cutlass.range_constexpr(cute.size(r_acc)):
            r_acc[i] = r_acc[i] * fc2_act_weight_dequant_scale
        r_bf16 = cute.make_rmem_tensor(r_acc_layout.shape, cutlass.BFloat16)
        r_bf16.store(r_acc.load().to(cutlass.BFloat16))

        thread_in_warp = tidx % WarpThreadCount
        lane_group = thread_in_warp // 4
        lane_mod = thread_in_warp % 4
        token_row0 = (
            cutlass.Int32(
                m_sub * self._accum_fragment_tile_m + local_warp_idx * 16
            )
            + lane_group
        )
        token_row1 = token_row0 + cutlass.Int32(8)
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        stg_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=32,
        )

        if cutlass.const_expr(
            token_comm_args is not None and not self._token_back_by_dispatch
        ):
            metadata_u32 = cute.recast_tensor(
                token_comm_args.token_src_metadata, cutlass.Uint32,
            )
            fc2_output_dest = Fc2OutputDest(
                tensor=token_comm_args.combine_output,
                metadata=metadata_u32,
                peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
                reduce_topk_in_kernel=self._fc2_in_kernel_topk_reduce,
            )
            r_bf16_u32 = cute.recast_tensor(r_bf16, cutlass.Uint32)
            for rep in cutlass.range_constexpr(Fc2SubtileN // 8):
                col_pair = cutlass.Int32(rep * 8) + lane_mod * cutlass.Int32(2)
                hidden_off = hidden_col_start + col_pair
                row0_regs = cute.make_tensor(
                    r_bf16.iterator + rep * 4,
                    cute.make_layout(2),
                )
                row1_regs = cute.make_tensor(
                    r_bf16.iterator + rep * 4 + 2,
                    cute.make_layout(2),
                )
                row0_packed = cutlass.Uint32(r_bf16_u32[rep * 2])
                row1_packed = cutlass.Uint32(r_bf16_u32[rep * 2 + 1])
                row0_next = cute.arch.shuffle_sync(
                    row0_packed, thread_in_warp + cutlass.Int32(1),
                )
                row1_next = cute.arch.shuffle_sync(
                    row1_packed, thread_in_warp + cutlass.Int32(1),
                )
                if token_row0 < valid_tokens and hidden_off < valid_hidden:
                    pool_token0 = (
                        work_tile_info.cumulative_data_physical_row
                        + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                        + token_row0
                    )
                    dest_row0 = fc2_output_dest.resolve_token_row(pool_token0)
                    dest_ptr0 = cute.make_ptr(
                        cutlass.BFloat16,
                        dest_row0.iterator.toint() + hidden_off * cutlass.Int64(2),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    if cutlass.const_expr(self._fc2_in_kernel_topk_reduce):
                        if lane_mod % cutlass.Int32(2) == cutlass.Int32(0):
                            red_add_relaxed_sys_v2_bf16x2(
                                dest_ptr0, row0_packed, row0_next,
                            )
                    else:
                        cute.copy(
                            stg_atom, row0_regs,
                            cute.make_tensor(dest_ptr0, cute.make_layout(2)),
                        )
                if token_row1 < valid_tokens and hidden_off < valid_hidden:
                    pool_token1 = (
                        work_tile_info.cumulative_data_physical_row
                        + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                        + token_row1
                    )
                    dest_row1 = fc2_output_dest.resolve_token_row(pool_token1)
                    dest_ptr1 = cute.make_ptr(
                        cutlass.BFloat16,
                        dest_row1.iterator.toint() + hidden_off * cutlass.Int64(2),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    if cutlass.const_expr(self._fc2_in_kernel_topk_reduce):
                        if lane_mod % cutlass.Int32(2) == cutlass.Int32(0):
                            red_add_relaxed_sys_v2_bf16x2(
                                dest_ptr1, row1_packed, row1_next,
                            )
                    else:
                        cute.copy(
                            stg_atom, row1_regs,
                            cute.make_tensor(dest_ptr1, cute.make_layout(2)),
                        )
        else:
            if cutlass.const_expr(self._use_fc2_tma_store):
                sD_stage = cute.slice_(
                    smem_fc2_output_buffer, (None, None, store_stage_idx),
                )
                for rep in cutlass.range_constexpr(Fc2SubtileN // 8):
                    col_pair = (
                        cutlass.Int32(rep * 8)
                        + lane_mod * cutlass.Int32(2)
                    )
                    col_pair_tile = col_pair // cutlass.Int32(2)
                    row0_regs = cute.make_tensor(
                        r_bf16.iterator + rep * 4,
                        cute.make_layout(2),
                    )
                    row1_regs = cute.make_tensor(
                        r_bf16.iterator + rep * 4 + 2,
                        cute.make_layout(2),
                    )
                    s_row0 = cute.local_tile(
                        sD_stage, (1, 2), (token_row0, col_pair_tile),
                    )
                    s_row1 = cute.local_tile(
                        sD_stage, (1, 2), (token_row1, col_pair_tile),
                    )
                    cute.copy(stg_atom, row0_regs, cute.coalesce(s_row0))
                    cute.copy(stg_atom, row1_regs, cute.coalesce(s_row1))
            else:
                g_fc2_output_tile = cute.local_tile(
                    real_fc2_output,
                    (self._cta_tile_m, Fc2SubtileN, 1),
                    (work_tile_info.tile_m_idx, hidden_group, 0),
                )
                g_fc2_slice = cute.slice_(
                    g_fc2_output_tile, (None, None, 0),
                )
                for rep in cutlass.range_constexpr(Fc2SubtileN // 8):
                    col_pair = (
                        cutlass.Int32(rep * 8)
                        + lane_mod * cutlass.Int32(2)
                    )
                    col_pair_tile = col_pair // cutlass.Int32(2)
                    hidden_off = hidden_col_start + col_pair
                    row0_regs = cute.make_tensor(
                        r_bf16.iterator + rep * 4,
                        cute.make_layout(2),
                    )
                    row1_regs = cute.make_tensor(
                        r_bf16.iterator + rep * 4 + 2,
                        cute.make_layout(2),
                    )
                    if token_row0 < valid_tokens and hidden_off < valid_hidden:
                        g_row0 = cute.local_tile(
                            g_fc2_slice, (1, 2),
                            (token_row0, col_pair_tile),
                        )
                        g_flat0 = cute.coalesce(g_row0)
                        aligned_iter0 = cute.make_ptr(
                            cutlass.BFloat16,
                            g_flat0.iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        cute.copy(
                            stg_atom, row0_regs,
                            cute.make_tensor(aligned_iter0, g_flat0.layout),
                        )
                    if token_row1 < valid_tokens and hidden_off < valid_hidden:
                        g_row1 = cute.local_tile(
                            g_fc2_slice, (1, 2),
                            (token_row1, col_pair_tile),
                        )
                        g_flat1 = cute.coalesce(g_row1)
                        aligned_iter1 = cute.make_ptr(
                            cutlass.BFloat16,
                            g_flat1.iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        cute.copy(
                            stg_atom, row1_regs,
                            cute.make_tensor(aligned_iter1, g_flat1.layout),
                        )

        if _iket_active:
            iket.range_pop()  # nswap_fc2_epi_m64n32


    @cute.jit
    def _run_fc2_epilogue(
        self,
        work_tile_info,
        accumulators: cute.Tensor,
        n_half: cutlass.Constexpr,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        smem_fc2_output_buffer: cute.Tensor,
        tma_atom_fc2_output: cute.CopyAtom,
        fc2_store_pipeline,
        valid_hidden,
        local_warp_idx: int,
        tidx,
        _iket_active,
        fc2_act_weight_dequant_scale,
        store_group_idx,
        token_comm_args=None,
    ):
        """Run the FC2 epilogue for one completed WGMMA task tile."""
        real_fc2_output, _ = sched_ext.get_gmem_tensor(
            "d", gmem_fc2_output, work_tile_info,
        )
        # Inner epilogue IKET range for non-swap-AB FC2. WGMMA has already
        # finished. This common per-tensor/blockwise path converts accumulator
        # fragments to BF16 and stores them through either staged TMA or the
        # token-major direct path; nested M=64 ranges isolate each output
        # subtile in this n_half work item.
        if _iket_active:
            iket.range_push("nswap_fc2_epi_m64n128")

        fc2_store_stage_idx = cutlass.Int32(0)
        subtile_begin = n_half * 4
        subtile_end = subtile_begin + 4
        for m_sub in cutlass.range_constexpr(self._m64_fragment_count):
            for subtile_idx in cutlass.range_constexpr(
                subtile_begin, subtile_end, 1
            ):
                physical_store_stage = (
                    cutlass.Int32(store_group_idx * self._store_buffer_stages)
                    + fc2_store_stage_idx
                )
                self._run_fc2_epilogue_subtile_m64(
                    subtile_idx=subtile_idx,
                    n_half=n_half,
                    m_sub=m_sub,
                    accumulators=accumulators,
                    real_fc2_output=(
                        smem_fc2_output_buffer
                        if cutlass.const_expr(self._use_fc2_tma_store)
                        else real_fc2_output
                    ),
                    smem_fc2_output_buffer=smem_fc2_output_buffer,
                    work_tile_info=work_tile_info,
                    valid_hidden=valid_hidden,
                    local_warp_idx=local_warp_idx,
                    tidx=tidx,
                    _iket_active=_iket_active,
                    token_comm_args=token_comm_args,
                    fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                    store_stage_idx=physical_store_stage,
                )
                if cutlass.const_expr(self._use_fc2_tma_store):
                    cute.arch.fence_proxy("async.shared", space="cta")
                    fc2_store_bar = pipeline.NamedBarrier(
                        barrier_id=self._fc1_store_sync_bar_id + store_group_idx,
                        num_threads=EpiWarpCount * WarpThreadCount,
                    )
                    fc2_store_bar.arrive_and_wait()
                    if local_warp_idx == cutlass.Int32(0):
                        hidden_group = (
                            work_tile_info.tile_n_idx
                            * cutlass.Int32(self._cta_tile_n // Fc2SubtileN)
                            + cutlass.Int32(subtile_idx)
                        )
                        g_fc2_output_tile = cute.local_tile(
                            real_fc2_output,
                            (self._accum_fragment_tile_m, Fc2SubtileN, 1),
                            (
                                work_tile_info.tile_m_idx
                                * cutlass.Int32(self._m64_fragment_count)
                                + cutlass.Int32(m_sub),
                                hidden_group,
                                0,
                            ),
                        )
                        tma_store_fc2_output(
                            smem_fc2_output_buffer,
                            physical_store_stage,
                            tma_atom_fc2_output,
                            g_fc2_output_tile,
                            work_tile_info.valid_tokens_in_cta_tile,
                        )
                        fc2_store_pipeline.producer_commit()
                        fc2_store_pipeline.producer_acquire()
                    fc2_store_bar.arrive_and_wait()
                    fc2_store_stage_idx = (
                        fc2_store_stage_idx + cutlass.Int32(1)
                    ) % cutlass.Int32(self._fc2_store_pipeline_stages)

        if _iket_active:
            iket.range_pop()  # nswap_fc2_epi_m64n128


    @cute.jit
    def _run_fc1_half_tile(
        self,
        work_tile_info,
        local_warp_idx: int,
        tiled_mma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        n_half: cutlass.Constexpr,
        ab_pipeline,
        weight_sf_pipeline,
        ab_consumer_state,
        k_tile_cnt_fc1,
        k_tile_cnt_fc2,
        _iket_active,
        run_wgmma_task_tile: cutlass.Constexpr,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        tidx,
        fc1_act_weight_dequant_scale,
        fc2_act_dequant_scale,
        norm_const,
        store_stage_idx,
        math_wg_order_barrier,
        math_wg_order_state,
        epi_wg_order_barrier,
        epi_wg_order_state,
    ):
        if cutlass.const_expr(self._pingpong_order):
            if _iket_active:
                iket.range_push("pp_nswap_fc1_math_wait")
            math_wg_order_barrier.wait(math_wg_order_state)
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_nswap_fc1_wgmma")
        ab_consumer_state = run_wgmma_task_tile(
            work_tile_info=work_tile_info,
            local_warp_idx=local_warp_idx,
            tiled_mma=tiled_mma,
            tCrA=tCrA,
            tCrB=tCrB,
            accumulators=accumulators,
            accum_temp=accum_temp,
            n_half=n_half,
            ab_pipeline=ab_pipeline,
            weight_sf_pipeline=weight_sf_pipeline,
            ab_consumer_state=ab_consumer_state,
            smem_activation_sf=smem_activation_sf,
            smem_weight_sf=smem_weight_sf,
            k_tile_cnt_fc1=k_tile_cnt_fc1,
            k_tile_cnt_fc2=k_tile_cnt_fc2,
            _iket_active=_iket_active,
            tidx=tidx,
        )
        if cutlass.const_expr(self._pingpong_order):
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_nswap_fc1_math_handoff")
            math_wg_order_state = math_wg_order_barrier.arrive(
                math_wg_order_state
            )
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_nswap_fc1_epi_wait")
            epi_wg_order_barrier.wait(epi_wg_order_state)
            if _iket_active:
                iket.range_pop()
        self._run_fc1_epilogue(
            work_tile_info=work_tile_info,
            accumulators=accumulators,
            n_half=n_half,
            smem_fc1_output_buffer=smem_fc1_output_buffer,
            tma_atom_fc1_output=tma_atom_fc1_output,
            sched_ext=sched_ext,
            gmem_fc1_output=gmem_fc1_output,
            gmem_fc1_output_sf=gmem_fc1_output_sf,
            gmem_topk_scores=gmem_topk_scores,
            local_warp_idx=local_warp_idx,
            tidx=tidx,
            _iket_active=_iket_active,
            fc1_act_weight_dequant_scale=fc1_act_weight_dequant_scale,
            fc2_act_dequant_scale=fc2_act_dequant_scale,
            norm_const=norm_const,
            store_stage_idx=store_stage_idx,
        )
        return ab_consumer_state, math_wg_order_state, epi_wg_order_state

    @cute.jit
    def _run_fc2_half_tile(
        self,
        work_tile_info,
        local_warp_idx: int,
        tiled_mma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        n_half: cutlass.Constexpr,
        ab_pipeline,
        weight_sf_pipeline,
        ab_consumer_state,
        k_tile_cnt_fc1,
        k_tile_cnt_fc2,
        _iket_active,
        run_wgmma_task_tile: cutlass.Constexpr,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        smem_fc2_output_buffer: cute.Tensor,
        tma_atom_fc2_output: cute.CopyAtom,
        fc2_store_pipeline,
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        valid_hidden,
        tidx,
        fc2_act_weight_dequant_scale,
        store_group_idx,
        math_wg_order_barrier,
        math_wg_order_state,
        epi_wg_order_barrier,
        epi_wg_order_state,
        token_comm_args=None,
    ):
        if cutlass.const_expr(self._pingpong_order):
            if _iket_active:
                iket.range_push("pp_nswap_fc2_math_wait")
            math_wg_order_barrier.wait(math_wg_order_state)
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_nswap_fc2_wgmma")
        ab_consumer_state = run_wgmma_task_tile(
            work_tile_info=work_tile_info,
            local_warp_idx=local_warp_idx,
            tiled_mma=tiled_mma,
            tCrA=tCrA,
            tCrB=tCrB,
            accumulators=accumulators,
            accum_temp=accum_temp,
            n_half=n_half,
            ab_pipeline=ab_pipeline,
            weight_sf_pipeline=weight_sf_pipeline,
            ab_consumer_state=ab_consumer_state,
            smem_activation_sf=smem_activation_sf,
            smem_weight_sf=smem_weight_sf,
            k_tile_cnt_fc1=k_tile_cnt_fc1,
            k_tile_cnt_fc2=k_tile_cnt_fc2,
            _iket_active=_iket_active,
            tidx=tidx,
        )
        if cutlass.const_expr(self._pingpong_order):
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_nswap_fc2_math_handoff")
            math_wg_order_state = math_wg_order_barrier.arrive(
                math_wg_order_state
            )
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_nswap_fc2_epi_wait")
            epi_wg_order_barrier.wait(epi_wg_order_state)
            if _iket_active:
                iket.range_pop()
        self._run_fc2_epilogue(
            work_tile_info=work_tile_info,
            accumulators=accumulators,
            n_half=n_half,
            sched_ext=sched_ext,
            gmem_fc2_output=gmem_fc2_output,
            smem_fc2_output_buffer=smem_fc2_output_buffer,
            tma_atom_fc2_output=tma_atom_fc2_output,
            fc2_store_pipeline=fc2_store_pipeline,
            valid_hidden=valid_hidden,
            local_warp_idx=local_warp_idx,
            tidx=tidx,
            _iket_active=_iket_active,
            fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
            store_group_idx=store_group_idx,
            token_comm_args=token_comm_args,
        )
        return ab_consumer_state, math_wg_order_state, epi_wg_order_state

    @cute.jit
    def run(
        self,
        sched_consumer,
        sched_ext,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        smem_fc2_output_buffer: cute.Tensor,
        tma_atom_fc2_output: cute.CopyAtom,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        gmem_fc2_output: cute.Tensor,
        valid_hidden,
        gmem_fc1_done_counter: cute.Tensor,
        local_warp_idx: int,
        tidx,
        fc1_activation_dequant_scale: cute.Tensor,
        fc1_weight_dequant_scale: cute.Tensor,
        fc2_activation_dequant_scale: cute.Tensor,
        fc2_weight_dequant_scale: cute.Tensor,
        norm_const,
        tiled_mma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        run_wgmma_task_tile: cutlass.Constexpr,
        ab_pipeline,
        weight_sf_pipeline,
        ab_consumer_state,
        k_tile_cnt_fc1,
        k_tile_cnt_fc2,
        _iket_active,
        n_half,
        warpgroup_idx,
        math_wg_order_barrier,
        epi_wg_order_barrier,
        token_comm_args=None,
    ) -> None:
        """
        Run the full FP8 fc1+fc2 fused MMA+epilogue task-tile loop.

        ``token_comm_args`` (MegaMoE path) is forwarded to the fc2 task tile so
        the fc2 STG is routed to the source rank's combine output.
        """
        task_tile_boundary_bar = pipeline.NamedBarrier(
            barrier_id=(
                self._epilog_sync_bar_id + warpgroup_idx
                if cutlass.const_expr(self._pingpong)
                else self._epilog_sync_bar_id
            ),
            num_threads=(
                EpiWarpCount * WarpThreadCount
                if cutlass.const_expr(self._pingpong)
                else WarpThreadCount * len(self._epilogue_warp_ids)
            ),
        )
        fc1_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self._fc1_store_pipeline_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, EpiWarpCount * WarpThreadCount,
            ),
        )
        fc2_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self._fc2_store_pipeline_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, EpiWarpCount * WarpThreadCount,
            ),
        )

        if _iket_active:
            iket.range_push("nswap_sched_consume_initial")
        if cutlass.const_expr(self._pingpong):
            (
                sched_consumer,
                work_tile_info,
                ab_consumer_state,
            ) = consume_initial_pingpong_work(
                sched_consumer,
                warpgroup_idx,
                ab_consumer_state,
                k_tile_cnt_fc1,
                k_tile_cnt_fc2,
            )
        else:
            work_tile_info = sched_consumer.consume_work()
        if _iket_active:
            iket.range_pop()

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            tid=tidx % (
                EpiWarpCount * WarpThreadCount
                if self._pingpong
                else len(self._epilogue_warp_ids) * WarpThreadCount
            ),
        )
        fc1_store_stage_idx = cutlass.Int32(0)
        pending_fc1_flag_addr = Int64(0)
        if cutlass.const_expr(self._pingpong):
            math_wg_order_state = math_wg_order_barrier.state.clone()
            epi_wg_order_state = epi_wg_order_barrier.state.clone()
        else:
            math_wg_order_state = ab_consumer_state.clone()
            epi_wg_order_state = ab_consumer_state.clone()

        while work_tile_info.is_valid_tile:
            # Outer task-tile range, emitted by local warp 0 of each epilogue
            # warpgroup. It starts
            # before expert-scale setup and contains the representative WGMMA
            # child range plus that warp's representative epilogue child
            # ranges. FC1 also includes final TMA STG issue; async drain,
            # task-boundary synchronization, and next-tile scheduling follow
            # after the matching pop below.
            if _iket_active:
                if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                    if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                        iket.range_push("nswap_fc1_task_bw")
                    else:
                        iket.range_push("nswap_fc1_task_pt")
                else:
                    if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                        iket.range_push("nswap_fc2_task_bw")
                    else:
                        iket.range_push("nswap_fc2_task_pt")

            expert_idx = work_tile_info.expert_idx
            if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                fc1_act_weight_dequant_scale = Float32(1.0)
                fc2_act_dequant_scale = Float32(1.0)
                fc2_act_weight_dequant_scale = Float32(1.0)
            else:
                fc1_act_weight_dequant_scale = (
                    Float32(fc1_activation_dequant_scale[0])
                    * Float32(fc1_weight_dequant_scale[expert_idx])
                )
                fc2_act_dequant_scale = Float32(
                    fc2_activation_dequant_scale[0]
                )
                fc2_act_weight_dequant_scale = (
                    fc2_act_dequant_scale
                    * Float32(fc2_weight_dequant_scale[expert_idx])
                )

            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                fc1_physical_store_stage = (
                    cutlass.Int32(warpgroup_idx * self._store_buffer_stages)
                    + fc1_store_stage_idx
                )
                if n_half == cutlass.Int32(0):
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc1_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=0,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        gmem_fc1_output_sf=gmem_fc1_output_sf,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        gmem_topk_scores=gmem_topk_scores,
                        tidx=tidx,
                        fc1_act_weight_dequant_scale=fc1_act_weight_dequant_scale,
                        fc2_act_dequant_scale=fc2_act_dequant_scale,
                        norm_const=norm_const,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        store_stage_idx=fc1_physical_store_stage,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
                    self._store_fc1_task_tile(
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        fc1_store_pipeline=fc1_store_pipeline,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        local_warp_idx=local_warp_idx,
                        n_half=0,
                        store_group_idx=warpgroup_idx,
                        store_stage_idx=fc1_physical_store_stage,
                        _iket_active=_iket_active,
                    )
                else:
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc1_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=1,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        gmem_fc1_output_sf=gmem_fc1_output_sf,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        gmem_topk_scores=gmem_topk_scores,
                        tidx=tidx,
                        fc1_act_weight_dequant_scale=fc1_act_weight_dequant_scale,
                        fc2_act_dequant_scale=fc2_act_dequant_scale,
                        norm_const=norm_const,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        store_stage_idx=fc1_physical_store_stage,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
                    self._store_fc1_task_tile(
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        fc1_store_pipeline=fc1_store_pipeline,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        local_warp_idx=local_warp_idx,
                        n_half=1,
                        store_group_idx=warpgroup_idx,
                        store_stage_idx=fc1_physical_store_stage,
                        _iket_active=_iket_active,
                    )
                fc1_store_stage_idx = (
                    fc1_store_stage_idx + cutlass.Int32(1)
                ) % cutlass.Int32(self._fc1_store_pipeline_stages)
            else:
                if n_half == cutlass.Int32(0):
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc2_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=0,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        sched_ext=sched_ext,
                        gmem_fc2_output=gmem_fc2_output,
                        smem_fc2_output_buffer=smem_fc2_output_buffer,
                        tma_atom_fc2_output=tma_atom_fc2_output,
                        fc2_store_pipeline=fc2_store_pipeline,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        valid_hidden=valid_hidden,
                        tidx=tidx,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        fc2_act_weight_dequant_scale=(
                            fc2_act_weight_dequant_scale
                        ),
                        store_group_idx=warpgroup_idx,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                        token_comm_args=token_comm_args,
                    )
                else:
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc2_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=1,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        sched_ext=sched_ext,
                        gmem_fc2_output=gmem_fc2_output,
                        smem_fc2_output_buffer=smem_fc2_output_buffer,
                        tma_atom_fc2_output=tma_atom_fc2_output,
                        fc2_store_pipeline=fc2_store_pipeline,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        valid_hidden=valid_hidden,
                        tidx=tidx,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        fc2_act_weight_dequant_scale=(
                            fc2_act_weight_dequant_scale
                        ),
                        store_group_idx=warpgroup_idx,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                        token_comm_args=token_comm_args,
                    )
            # Matches the selected nswap_fc1/fc2_task_* range above.
            if _iket_active:
                iket.range_pop()

            cur_was_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            # Hopper clusters use independent WGMMA CTAs. Keep one FC1-done
            # slot per CTA token tile so FC2 only waits for its matching FC1
            # workspace rows, rather than an unrelated peer CTA in the cluster.
            cur_fc1_counter_slot = (
                cutlass.Int32(self._cluster_m)
                * work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_m_idx
            )
            cur_fc1_flag_addr = (
                gmem_fc1_done_counter.iterator + cur_fc1_counter_slot
            ).toint()
            cur_fc2_expert_idx = work_tile_info.expert_idx

            if cutlass.const_expr(self._pingpong_order):
                # Match CUTLASS ping-pong ordering: retire this task's async
                # stores and hand epilogue ownership to the peer WG before
                # waiting for the descriptor two scheduler positions ahead.
                if _iket_active:
                    if cur_was_linear1:
                        iket.range_push("pp_nswap_fc1_retire_handoff")
                    else:
                        iket.range_push("pp_nswap_fc2_retire_handoff")
                if cur_was_linear1:
                    if local_warp_idx == cutlass.Int32(0):
                        fc1_store_pipeline.producer_tail()
                    cute.arch.fence_acq_rel_gpu()
                    flag_tracker = flag_tracker.accumulate(
                        work_tile_info.phase,
                        1,
                        cur_fc1_flag_addr,
                    )
                else:
                    if cutlass.const_expr(self._use_fc2_tma_store):
                        if local_warp_idx == cutlass.Int32(0):
                            fc2_store_pipeline.producer_tail()
                    if cutlass.const_expr(self._token_back_by_dispatch):
                        cute.arch.fence_acq_rel_gpu()
                        fc2_flag_addr = (
                            token_comm_args.fc2_done_counter.iterator
                            + cur_fc2_expert_idx
                        ).toint()
                    else:
                        fc2_flag_addr = Int64(0)
                    flag_tracker = flag_tracker.accumulate(
                        work_tile_info.phase,
                        1,
                        fc2_flag_addr,
                        not self._token_back_by_dispatch,
                    )
                epi_wg_order_state = epi_wg_order_barrier.arrive(
                    epi_wg_order_state
                )
                if _iket_active:
                    iket.range_pop()

            if _iket_active:
                iket.range_push("nswap_sched_consume_next")
            if cutlass.const_expr(self._pingpong):
                (
                    sched_consumer,
                    work_tile_info,
                    ab_consumer_state,
                ) = consume_next_pingpong_work(
                    sched_consumer,
                    ab_consumer_state,
                    k_tile_cnt_fc1,
                    k_tile_cnt_fc2,
                )
            else:
                work_tile_info = sched_consumer.consume_work()
            if _iket_active:
                iket.range_pop()

            # The acquire paired with this task's commit has retired the
            # previous FC1 store group in steady state. Drain the current group
            # only at an FC1 phase boundary or kernel tail; otherwise retain it
            # in flight while the next task computes into the other ring stage.
            drain_fc1_store = cutlass.Boolean(0)
            if cur_was_linear1 and cutlass.const_expr(
                not self._pingpong_order
            ):
                drain_fc1_store = (
                    not work_tile_info.is_valid_tile
                    or work_tile_info.phase
                    != cutlass.Int32(BlockPhase.Linear1)
                )
                if drain_fc1_store:
                    if _iket_active:
                        iket.range_push("nswap_fc1_store_drain")
                    if local_warp_idx == cutlass.Int32(0):
                        fc1_store_pipeline.producer_tail()
                    if _iket_active:
                        iket.range_pop()

                # The done counter uses release semantics after every WG has
                # crossed the task boundary below. This fence makes the retired
                # TMA store group visible to the FC2 acquire-side polling path.
                cute.arch.fence_acq_rel_gpu()

            # A token-back completion flag or a phase change makes the FC2
            # output externally visible. Drain the final store group before
            # the CTA-wide task boundary; steady-state FC2 tasks retain one
            # group in flight so the next R2S can overlap its S2G.
            if cutlass.const_expr(self._use_fc2_tma_store):
                if (
                    not cur_was_linear1
                    and cutlass.const_expr(not self._pingpong_order)
                ):
                    drain_fc2_store = (
                        not work_tile_info.is_valid_tile
                        or work_tile_info.phase
                        == cutlass.Int32(BlockPhase.Linear1)
                    )
                    if cutlass.const_expr(self._token_back_by_dispatch):
                        drain_fc2_store = cutlass.Boolean(1)
                    if drain_fc2_store:
                        if local_warp_idx == cutlass.Int32(0):
                            fc2_store_pipeline.producer_tail()

            if _iket_active:
                iket.range_push("nswap_task_boundary_barrier")
            if cutlass.const_expr(not self._pingpong):
                task_tile_boundary_bar.arrive_and_wait()
            if _iket_active:
                iket.range_pop()

            if cur_was_linear1 and cutlass.const_expr(
                not self._pingpong_order
            ):
                if _iket_active:
                    iket.range_push("nswap_fc1_done_flag_accumulate")
                if cutlass.const_expr(self._fc1_store_pipeline_stages == 1):
                    # A one-stage acquire waits for the current store, matching
                    # the original synchronous done-publication behavior.
                    flag_tracker = flag_tracker.accumulate(
                        work_tile_info.phase,
                        self._epi_fc1_batch,
                        cur_fc1_flag_addr,
                    )
                else:
                    # A two-stage acquire retires the previous store group. The
                    # current task is publishable only after an explicit drain;
                    # otherwise carry its address as completion metadata.
                    if pending_fc1_flag_addr != Int64(0):
                        flag_tracker = flag_tracker.accumulate(
                            cutlass.Int32(BlockPhase.Linear1),
                            self._epi_fc1_batch,
                            pending_fc1_flag_addr,
                        )
                    if drain_fc1_store:
                        flag_tracker = flag_tracker.accumulate(
                            work_tile_info.phase,
                            self._epi_fc1_batch,
                            cur_fc1_flag_addr,
                        )
                        pending_fc1_flag_addr = Int64(0)
                    else:
                        pending_fc1_flag_addr = cur_fc1_flag_addr
                if _iket_active:
                    iket.range_pop()
            elif cutlass.const_expr(not self._pingpong_order):
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
                if _iket_active:
                    iket.range_push("nswap_fc2_done_flag_accumulate")
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc2_batch,
                    fc2_flag_addr,
                    no_fire,
                )
                if _iket_active:
                    iket.range_pop()

        if local_warp_idx == cutlass.Int32(0):
            fc1_store_pipeline.producer_tail()
            if cutlass.const_expr(self._use_fc2_tma_store):
                fc2_store_pipeline.producer_tail()

        if _iket_active:
            iket.range_push("nswap_done_flag_fire_tail")
        flag_tracker.fire()
        if _iket_active:
            iket.range_pop()
