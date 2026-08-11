# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Swap-AB epilogue for MXFP8-weight / BF16-activation fused FC1+FC2.

The two GEMMs accumulate FP32 in the
swap-AB layout ``(feature, token)``.  FC1 reuses the proven NVFP4 TMEM
permutation, applies the BF16 SwiGLU numerical contract, and writes a plain
BF16 FC1 hand-off directly to global memory.  FC2 reuses the common swap-AB
BF16 output pipeline.

The first implementation deliberately keeps the NVFP4 epilogue's 16-column
gate/up interleave::

    [gate(16), up(16), gate(16), up(16), ...]

That makes each epilogue warp own one complete gate/up pair in a 32-row TMEM
slice.  Moving to the MXFP8/BF16 path's historical 32-column interleave is a
separate layout change: it makes a pair span two epilogue warps and therefore
needs a different register/warp exchange.

There is no FC1-output quantization and no FC1 scale plane.
"""

from typing import List, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int64
from cutlass._mlir import ir
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import AddressSpace
import cutlass.pipeline as pipeline

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover - older CuTeDSL wheels
    from src.iket_compat import iket

from common.moe_utils import fmax, fmin
from moe_bf16_glu.epilogue_bf16 import swiglu_act
from moe_nvfp4_swapab.epilogue_refactor import (
    NvFp4OptinalEpiArgs,
    Region,
    SwapABFc2Epilogue,
    TmemTranspose16x32,
    _TmemTranspose16x32Core,
)
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase
from src.flag_batch import GpuReleaseFlagBatchTracker
from src.token_comm import CombineFormat


Fc1GateUpInterleave = 16
EpilogueTokenTile = 64
EpilogueWarpCount = 4
WarpThreadCount = 32


class _ImmutableAfterInit:
    """Small device-context wrapper guard used across the scheduler loop."""

    def __setattr__(self, name, value):
        if self.__dict__.get("_frozen_", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable after __init__ "
                f"(cannot set {name!r})."
            )
        object.__setattr__(self, name, value)

    def _freeze(self) -> None:
        object.__setattr__(self, "_frozen_", True)


class _MixedSwapABFc1Epilogue(_ImmutableAfterInit):
    """Device-side FC1 task-tile consumer for :class:`MixedGluBf16Epilogue`."""

    def __init__(
        self,
        base: "MixedGluBf16Epilogue",
        tidx: cutlass.Int32,
        sched_ext,
        fc1_output: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        optional_epi_args: NvFp4OptinalEpiArgs,
    ) -> None:
        self.base = base
        self.tidx = tidx % (base._EpilogueWarpCnt * WarpThreadCount)
        self.warp_idx = self.tidx // WarpThreadCount
        self.lane_idx = self.tidx % WarpThreadCount
        self.sched_ext = sched_ext
        self.fc1_output = fc1_output
        self.fc1_done_counter = fc1_done_counter
        self.optional_epi_args = optional_epi_args
        self._freeze()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        # All fields are loop-invariant Python context.  Dynamic state is passed
        # explicitly through __call__, so this helper is not an scf iter_arg.
        return []

    def __new_from_mlir_values__(
        self, values: List[ir.Value],
    ) -> "_MixedSwapABFc1Epilogue":
        assert len(values) == 0
        return self

    @cute.jit
    def signal_fc1_done(
        self, work_tile_info, next_work_tile_info, flag_tracker,
    ):
        """Publish one FC1 tile after all epilogue-warp STGs rendezvous."""
        if cutlass.const_expr(
            self.static_expert_shape is None
            or self.intermediate_downproj
            % self.cluster_tile_intermediate_downproj
            != 0
        ):
            in_bound = (
                work_tile_info.tile_m_idx
                * self._EpilogueFc1IntermediateDownTileSize
                < self.fc1_output.shape[1]
            )
        else:
            in_bound = True

        slot = (
            work_tile_info.cumulative_token_block_count
            + work_tile_info.tile_n_idx
        )
        flag_addr = Int64(0)
        if in_bound:
            flag_addr = (self.fc1_done_counter.iterator + slot).toint()
        return flag_tracker.accumulate(
            next_work_tile_info.phase,
            self.fc1_epi_flag_batch,
            flag_addr,
        )

    @cute.jit
    def __call__(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn: cutlass.Int32,
    ) -> None:
        real_fc1_output, _ = self.sched_ext.get_gmem_tensor(
            "c", self.fc1_output, work_tile_info,
        )
        if cutlass.const_expr(self.optional_epi_args.topk_scores is not None):
            real_topk_scores, _ = self.sched_ext.get_gmem_tensor(
                "topk", self.optional_epi_args.topk_scores, work_tile_info,
            )
        else:
            real_topk_scores = None

        iket.range_push("mixed_fc1_epi_consumer_wait")
        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_pop()
        iket.range_push("mixed_fc1_epi")

        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        tmem_subtiles = cute.flat_divide(
            tmem_acc_tensor,
            (
                self._EpilogueFc1IntermediateGateUpTileSize,
                self._EpilogueTokenTileSize,
            ),
        )[None, None, 0, None]

        # In the overlap layout the next physical accumulator stage overwrites
        # one 64-column subtile.  Preload that subtile before releasing the
        # one-stage barrier, then preload a guaranteed-safe subtile to use as
        # the in-place transpose workspace.  This is the same two-subtile turn
        # ordering used by the validated NVFP4 overlap epilogue.
        unroll_tile_cnt = 2 if cutlass.const_expr(
            self.overlapping_accum
        ) else 0
        remain_subtile_cnt = self.subtile_cnt - unroll_tile_cnt
        if cutlass.const_expr(unroll_tile_cnt > 0):
            subtile_idx_first = (
                cutlass.Int32(self.subtile_cnt) - is_odd_turn
            ) % cutlass.Int32(self.subtile_cnt)
            subtile_idx_second = (
                cutlass.Int32(self.subtile_cnt + 1) - is_odd_turn
            ) % cutlass.Int32(self.subtile_cnt)
            preload_warp_row_offset = (
                self.warp_idx * cutlass.Int32(2 * Fc1GateUpInterleave)
            ) << 16
            first_preload_subtile = tmem_subtiles[
                None, None, subtile_idx_first
            ]
            preload_first = _TmemTranspose16x32Core.load_subtile_raw_acc(
                cute.make_tensor(
                    first_preload_subtile.iterator
                    + cute.assume(preload_warp_row_offset, divby=16),
                    first_preload_subtile.layout,
                )
            )
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)
            second_preload_subtile = tmem_subtiles[
                None, None, subtile_idx_second
            ]
            preload_second = _TmemTranspose16x32Core.load_subtile_raw_acc(
                cute.make_tensor(
                    second_preload_subtile.iterator
                    + cute.assume(preload_warp_row_offset, divby=16),
                    second_preload_subtile.layout,
                )
            )
            preload_pair = (preload_first, preload_second)
            subtile_idx_pair = (subtile_idx_first, subtile_idx_second)
            for i in cutlass.range_constexpr(unroll_tile_cnt):
                if (
                    subtile_idx_pair[i]
                    * cutlass.Int32(self._EpilogueTokenTileSize)
                    < valid_tokens
                ):
                    self.run_subtile(
                        work_tile_info=work_tile_info,
                        subtile_idx=subtile_idx_pair[i],
                        tmem_subtile_tensor=tmem_subtiles[
                            None, None, subtile_idx_second
                        ],
                        preload_acc=preload_pair[i],
                        fc1_output=real_fc1_output,
                        topk_scores=real_topk_scores,
                        valid_tokens=valid_tokens,
                    )

        for i in cutlass.range(remain_subtile_cnt, unroll=1):
            real_i = i + unroll_tile_cnt
            if cutlass.const_expr(self.overlapping_accum):
                subtile_idx = (
                    cutlass.Int32(real_i + self.subtile_cnt) - is_odd_turn
                ) % cutlass.Int32(self.subtile_cnt)
            else:
                subtile_idx = cutlass.Int32(real_i)
            if (
                subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize)
                < valid_tokens
            ):
                self.run_subtile(
                    work_tile_info=work_tile_info,
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=tmem_subtiles[
                        None, None, subtile_idx
                    ],
                    preload_acc=None,
                    fc1_output=real_fc1_output,
                    topk_scores=real_topk_scores,
                    valid_tokens=valid_tokens,
                )

        # Direct STGs have consumed the accumulator.  The release-counter
        # publication in the outer run loop has release semantics and follows a
        # 128-thread rendezvous, ordering every warp's BF16 hand-off stores
        # before FC2 is allowed to load them.
        if cutlass.const_expr(not self.overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def run_subtile(
        self,
        *,
        work_tile_info,
        subtile_idx: cutlass.Int32,
        tmem_subtile_tensor: cute.Tensor,
        preload_acc: Optional[
            Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]
        ],
        fc1_output: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        valid_tokens: cutlass.Int32,
    ) -> None:
        """Fold one 64-token accumulator subtile and directly STG BF16.

        One epilogue warp owns 32 accumulator rows:

        * rows 0..15  -- one gate block;
        * rows 16..31 -- its matching up block.

        The two 32-token column halves are handled sequentially.  SwiGLU is
        elementwise and therefore runs in the raw TMEM load distribution; the
        resulting 16x32 fragment is then transposed to one token x 16 features
        per lane before the BF16 store.
        """
        warp_row_offset = self.warp_idx * cutlass.Int32(
            2 * Fc1GateUpInterleave
        )
        tmem_offset = warp_row_offset << 16
        warp_subtile_ptr = tmem_subtile_tensor.iterator + cute.assume(
            tmem_offset, divby=16,
        )

        atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
            cutlass.Float32,
        )
        stg_bf16x16 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=256,
        )

        feature_block = (
            work_tile_info.tile_m_idx
            * cutlass.Int32(
                self._EpilogueFc1IntermediateDownTileSize
                // Fc1GateUpInterleave
            )
            + self.warp_idx
        )
        feature_base = feature_block * cutlass.Int32(Fc1GateUpInterleave)

        for token_half in cutlass.range_constexpr(2):
            half_col_offset = token_half * 32
            half_ptr = warp_subtile_ptr + half_col_offset

            if cutlass.const_expr(preload_acc is None):
                gate = cute.make_rmem_tensor(
                    (Fc1GateUpInterleave,), cutlass.Float32,
                )
                up = cute.make_rmem_tensor(
                    (Fc1GateUpInterleave,), cutlass.Float32,
                )
                gate_view = cute.make_tensor(
                    half_ptr,
                    _TmemTranspose16x32Core._tmem_layout(
                        Fc1GateUpInterleave, 32,
                    ),
                )
                up_view = cute.make_tensor(
                    half_ptr + (Fc1GateUpInterleave << 16),
                    _TmemTranspose16x32Core._tmem_layout(
                        Fc1GateUpInterleave, 32,
                    ),
                )
                cute.copy(
                    atom_ld16x64,
                    gate_view,
                    _TmemTranspose16x32Core._rmem_copy_view(
                        gate, Fc1GateUpInterleave,
                    ),
                )
                cute.copy(
                    atom_ld16x64,
                    up_view,
                    _TmemTranspose16x32Core._rmem_copy_view(
                        up, Fc1GateUpInterleave,
                    ),
                )
            else:
                gate = preload_acc[token_half * 2]
                up = preload_acc[token_half * 2 + 1]

            if cutlass.const_expr(self.gate_up_clamp is not None):
                for i in cutlass.range_constexpr(Fc1GateUpInterleave):
                    gate[i] = fmin(gate[i], self.gate_up_clamp)
                    up[i] = fmin(up[i], self.gate_up_clamp)
                    up[i] = fmax(up[i], -self.gate_up_clamp)

            token_in_cta = (
                subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize)
                + cutlass.Int32(token_half * WarpThreadCount)
                + self.lane_idx
            )
            token_in_expert = (
                work_tile_info.tile_n_idx * cutlass.Int32(self.cta_tile_n)
                + token_in_cta
            )

            prob = None
            if cutlass.const_expr(topk_scores is not None):
                # Keep the warp converged around every TMEM atom.  Invalid lanes
                # use the multiplicative identity and are simply not stored.
                prob_value = cutlass.Float32(1.0)
                if token_in_cta < valid_tokens:
                    prob_value = cutlass.Float32(topk_scores[token_in_expert])
                prob = prob_value

            folded = cute.make_rmem_tensor(
                (Fc1GateUpInterleave,), cutlass.Float32,
            )
            # The raw TMEM load distribution does not give one token to one lane:
            # its 16 values span several token rows.  SwiGLU is elementwise
            # and may run before the transpose, but the token-specific top-k
            # weight must wait until the transpose output, where lane_idx is
            # the token coordinate and all 16 values belong to that token.
            swiglu_act(folded, up, gate, None)

            transposed = TmemTranspose16x32(
                half_ptr,
                Region.Top,
                reg_tensor=folded,
            ).from_r1_perm_until_last_store()
            if cutlass.const_expr(prob is not None):
                for i in cutlass.range_constexpr(
                    0, Fc1GateUpInterleave, 2
                ):
                    (
                        transposed[i],
                        transposed[i + 1],
                    ) = cute.arch.mul_packed_f32x2(
                        (transposed[i], transposed[i + 1]),
                        (prob, prob),
                        rnd="rn",
                        ftz=False,
                    )

            folded_bf16 = cute.make_rmem_tensor(
                (Fc1GateUpInterleave,), cutlass.BFloat16,
            )
            folded_bf16.store(transposed.load().to(cutlass.BFloat16))

            if (
                token_in_cta < valid_tokens
                and feature_base < fc1_output.shape[1]
            ):
                g_vec = cute.local_tile(
                    fc1_output,
                    (1, Fc1GateUpInterleave, 1),
                    (token_in_expert, feature_block, 0),
                )
                aligned_ptr = cute.make_ptr(
                    cutlass.BFloat16,
                    cute.coalesce(g_vec).iterator.toint(),
                    AddressSpace.gmem,
                    assumed_align=32,
                )
                cute.copy(
                    stg_bf16x16,
                    cute.coalesce(folded_bf16),
                    cute.make_tensor(
                        aligned_ptr,
                        cute.make_layout(Fc1GateUpInterleave),
                    ),
                )


class MixedGluBf16Epilogue:
    """Lean swap-AB FC1/FC2 epilogue with a BF16 inter-GEMM hand-off."""

    _EpilogueSyncWaitBarId = 1
    _EpilogueAsyncBarIdBase = 4
    _EpilogueFc1GateUpInterleave = Fc1GateUpInterleave
    _EpilogueTokenTileSize = EpilogueTokenTile
    _EpilogueFc1IntermediateGateUpTileSize = 128
    _EpilogueFc1IntermediateDownTileSize = 64
    _EpilogueFc2HiddenTileSize = 128
    _EpilogueWarpCnt = EpilogueWarpCount
    _TmemColsTotal = 512

    def __init__(
        self,
        *,
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        use_2cta_instrs: bool,
        fc1_output_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        combine_format: Optional[CombineFormat] = None,
        non_ubulk_fc2_store: bool = True,
        in_kernel_fc2_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        gate_up_clamp: Optional[float] = None,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        apply_topk_in_fc1: bool = False,
        accumulator_overlap: bool = False,
    ) -> None:
        if fc1_output_dtype is not cutlass.BFloat16:
            raise NotImplementedError(
                "MixedGluBf16Epilogue requires a BF16 FC1 hand-off; got "
                f"{fc1_output_dtype}."
            )
        if acc_dtype is not cutlass.Float32:
            raise NotImplementedError(
                "MixedGluBf16Epilogue requires FP32 accumulators."
            )
        if not non_ubulk_fc2_store:
            raise NotImplementedError(
                "The lean mixed epilogue initially supports FC2 direct STG only."
            )
        if combine_format is None:
            combine_format = CombineFormat.parse("bf16")
        if combine_format.is_quantized:
            raise NotImplementedError(
                "The mixed epilogue supports BF16 combine only."
            )

        atom_thr_size = 2 if use_2cta_instrs else 1
        self.cta_tile_m = mma_tiler_mnk[0] // atom_thr_size
        self.cta_tile_n = mma_tiler_mnk[1]
        self.cta_tile_k = mma_tiler_mnk[2]
        if self.cta_tile_m != self._EpilogueFc2HiddenTileSize:
            raise ValueError(
                "Per-CTA MMA M must be 128 for the swap-AB epilogue; got "
                f"{self.cta_tile_m}."
            )
        if self.cta_tile_n % self._EpilogueTokenTileSize != 0:
            raise ValueError(
                f"MMA N must be divisible by {self._EpilogueTokenTileSize}; "
                f"got {self.cta_tile_n}."
            )

        self.fc1_output_dtype = fc1_output_dtype
        self.acc_dtype = acc_dtype
        self.combine_format = combine_format
        self.fc2_use_bulk = False
        # Epilogue-warps Form-B reduces directly into the peer output.  The
        # dispatch-token-back path instead writes one BF16 route row locally;
        # TokenInPullTokenBackPush performs the optional top-k reduction while
        # pushing that staging row to the source rank.
        self.reduce_topk_in_kernel = (
            in_kernel_fc2_reduce and not token_back_by_dispatch
        )
        self.token_back_by_dispatch = token_back_by_dispatch
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        self.gate_up_clamp = (
            cutlass.Float32(abs(gate_up_clamp))
            if gate_up_clamp is not None
            else None
        )
        self.static_expert_shape = static_expert_shape

        if static_expert_shape is not None:
            _, intermediate_gateup, hidden = static_expert_shape
            if intermediate_gateup % (
                2 * self._EpilogueFc1GateUpInterleave
            ) != 0:
                raise ValueError(
                    "FC1 gate+up width must be divisible by "
                    f"{2 * self._EpilogueFc1GateUpInterleave}; got "
                    f"{intermediate_gateup}."
                )
            self.intermediate_downproj = intermediate_gateup // 2
            self.fc2_hidden_needs_predicate = (
                hidden % (self.cta_tile_m * cluster_shape_mn[0]) != 0
            )
        else:
            self.intermediate_downproj = None
            self.fc2_hidden_needs_predicate = True

        self.cluster_tile_intermediate_downproj = (
            self._EpilogueFc1IntermediateDownTileSize * cluster_shape_mn[0]
        )
        self.subtile_cnt = self.cta_tile_n // self._EpilogueTokenTileSize
        overlapping_accum = accumulator_overlap
        if overlapping_accum and self.cta_tile_n != 256:
            raise ValueError(
                "mixed accumulator overlap requires N256 and two physical "
                "accumulator stages."
            )

        # No SFA/SFB are consumed by MMA and no FC1 output scale is generated.
        self.acc_sf_cols = 0
        self.num_sfa_tmem_cols = 0
        self.num_sfb_tmem_cols = 0
        self.num_sf_tmem_cols = 0

        self.overlapping_accum = overlapping_accum
        self.num_acc_stage = 2
        self.num_acc_pipeline_stages = (
            1 if self.overlapping_accum else self.num_acc_stage
        )
        self.overlapped_tmem_cols = (
            self._EpilogueTokenTileSize if self.overlapping_accum else 0
        )
        self.tmem_acc_layout_py_obj = (
            (self.cta_tile_m, self.cta_tile_n, self.num_acc_stage),
            (
                _TmemTranspose16x32Core._TmemRowStride,
                1,
                self.cta_tile_n - self.overlapped_tmem_cols,
            ),
        )

        fc1_batch, fc2_batch = (
            (1, 1) if epi_flag_batch is None else epi_flag_batch
        )
        self.fc1_epi_flag_batch = max(1, min(32, int(fc1_batch)))
        self.fc2_epi_flag_batch = max(1, min(32, int(fc2_batch)))

        # Direct STG paths need no data scratch.  Keep a tiny aligned member so
        # the surrounding kernel can retain one uniform SharedStorage shape.
        self.epi_smem_bytes = 16

    def get_epi_storage_type(self) -> Type:
        @cute.struct
        class EpilogueSharedStorage:
            epi_smem: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, self.epi_smem_bytes], 16
            ]

        return EpilogueSharedStorage

    @cute.jit
    def run(
        self,
        epi_smem_storage,
        tmem_ptr: cute.Pointer,
        acc_pipeline,
        sched_consumer,
        sched_ext,
        fc1_output: cute.Tensor,
        fc2_output: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        tidx: cutlass.Int32,
        optional_epi_args: Optional[NvFp4OptinalEpiArgs] = None,
        token_comm_args=None,
    ) -> None:
        """Run the fused phase loop.

        FC1 uses direct BF16 STG and has no TMA-store atom or scale plane.
        """
        if cutlass.const_expr(optional_epi_args is None):
            optional_epi_args = NvFp4OptinalEpiArgs(
                fc1_alpha=None,
                fc2_alpha=None,
                fc1_norm_const=None,
                topk_scores=None,
            )

        # Mixed weights are fully decoded/scaled to BF16 before MMA.  Therefore
        # legacy NVFP4 alpha/norm fields must not rescale either accumulator.
        topk_scores = None
        if cutlass.const_expr(self.apply_topk_in_fc1):
            topk_scores = optional_epi_args.topk_scores
        mixed_optional_args = NvFp4OptinalEpiArgs(
            fc1_alpha=None,
            fc2_alpha=None,
            fc1_norm_const=None,
            topk_scores=topk_scores,
        )

        tmem_acc = cute.make_tensor(
            cute.recast_ptr(tmem_ptr, dtype=cutlass.Float32),
            cute.make_layout(
                self.tmem_acc_layout_py_obj[0],
                stride=self.tmem_acc_layout_py_obj[1],
            ),
        )
        fc1_epi = _MixedSwapABFc1Epilogue(
            self,
            tidx,
            sched_ext,
            fc1_output,
            fc1_done_counter,
            mixed_optional_args,
        )
        fc2_epi = SwapABFc2Epilogue(
            self,
            tidx,
            epi_smem_storage,
            fc2_output,
            token_comm_args,
            mixed_optional_args,
        )
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.num_acc_pipeline_stages,
        )
        epilogue_rendezvous = pipeline.NamedBarrier(
            barrier_id=self._EpilogueSyncWaitBarId,
            num_threads=WarpThreadCount * self._EpilogueWarpCnt,
        )
        is_odd_turn = cutlass.Int32(1)
        work_tile_info = sched_consumer.consume_work()
        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            tid=tidx % (self._EpilogueWarpCnt * WarpThreadCount),
        )
        while work_tile_info.is_valid_tile:
            phase_payload = cutlass.Int32(
                work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            )
            iket.range_push(
                "mixed_role_tile_lifetime",
                cutlass.Int32(50) + phase_payload,
            )
            iket.range_push(
                "mixed_role_coverage",
                cutlass.Int32(501)
                + phase_payload * cutlass.Int32(10),
            )
            if cutlass.const_expr(self.overlapping_accum):
                tmem_stage_idx = acc_consumer_state.phase
            else:
                tmem_stage_idx = acc_consumer_state.index
            tmem_acc_current = tmem_acc[None, None, tmem_stage_idx]
            iket.range_pop()
            iket.range_push(
                "mixed_role_coverage",
                cutlass.Int32(502)
                + phase_payload * cutlass.Int32(10),
            )
            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                fc1_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                )
            else:
                fc2_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                )
            iket.range_pop()
            iket.range_push(
                "mixed_role_coverage",
                cutlass.Int32(503)
                + phase_payload * cutlass.Int32(10),
            )

            previous_work_tile = work_tile_info
            was_fc1 = (
                previous_work_tile.phase == cutlass.Int32(BlockPhase.Linear1)
            )
            acc_consumer_state.advance()
            if cutlass.const_expr(self.overlapping_accum):
                is_odd_turn = cutlass.Int32(1) - is_odd_turn
            work_tile_info = sched_consumer.consume_work()

            # All direct STGs must execute before lane 0 emits the release
            # reduction that allows FC2's TMA-B consumer to proceed.
            epilogue_rendezvous.arrive_and_wait()
            if was_fc1:
                flag_tracker = fc1_epi.signal_fc1_done(
                    previous_work_tile, work_tile_info, flag_tracker,
                )
            else:
                flag_tracker = fc2_epi.signal_fc2_done(
                    previous_work_tile, work_tile_info, flag_tracker,
                )
            iket.range_pop()
            iket.range_pop()
            iket.range_pop()

        flag_tracker.fire()


__all__ = [
    "EpilogueTokenTile",
    "Fc1GateUpInterleave",
    "MixedGluBf16Epilogue",
]
