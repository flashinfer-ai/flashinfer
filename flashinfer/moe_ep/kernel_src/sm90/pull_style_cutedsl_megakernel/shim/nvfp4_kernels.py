# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: B007, F841
# Preserve unused vendor loop indices and allocation views for protocol parity.
"""NVFP4 specialization of SM90 mixed-input MegaMoE.

Weight scales, candidate conversion, and LUT allocation specialize the vendor.
The optional FC1 K256 schedule coalesces fences and overlaps conversion with
WGMMA; disabling it retains the vendor schedule and communication protocol.
``nvfp4_conversion`` selects the K-permuted weight layout and the LUT size and
banking; its default uses K64 order and an 8-replica table.
The MXFP4-prefixed private policy names retain the shared activation ABI.
"""

import cutlass
import cutlass.cute as cute

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover
    from src.iket_compat import iket
from cutlass.cute.nvgpu import warpgroup
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
import cutlass.utils.hopper_helpers as sm90_utils

from moe_hopper_fp8.kernel_mxfp4_fp8_glu_fc12_swapab import (
    Sm90SwapABSwigluMxfp4Fp8Fc12Kernel,
)
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase
from .nvfp4_policy import (
    NVFP4_DEFAULT_CONVERSION,
    Nvfp4ConversionLayout,
    nvfp4_shared_memory_budget,
)
from .nvfp4_conversion import (
    nvfp4_get_lut,
    nvfp4_initialize_lut,
    convert_nvfp4_group,
    MXFP4_FOLD_BLOCK_BYTES,
    MXFP4_FOLD_M,
    MXFP4_K_TILE,
    make_expanded_offset_view,
    make_expanded_offset_view_k256,
    make_offset_smem_layout,
    make_offset_smem_layout_k256,
    make_packed_a_ldsm_views_k256,
)


@dsl_user_op
def _keep_nvfp4_a_word_live(word, *, loc=None, ip=None):
    """Keep issued RS A words live through their asynchronous WGMMA wait."""
    llvm.inline_asm(
        None,
        [word.ir_value(loc=loc, ip=ip)],
        "// NVFP4_A_LIVE_AFTER_WAIT $0",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _prefetch_offset_bulk_l2(src_gmem, num_bytes, *, loc=None, ip=None):
    """Warm L2 with folded offsets that a later cp.async stages into SMEM."""
    llvm.inline_asm(
        None,
        [
            src_gmem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            num_bytes.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class Sm90SwapABSwigluNvfp4Fp8Fc12Kernel(Sm90SwapABSwigluMxfp4Fp8Fc12Kernel):
    """Packed E2M1 weight / E4M3 activation specialization for SM90.

    Weight payloads remain packed in GMEM and SMEM.  Every K128 stage is loaded
    with LDSM, converted in registers with the shared LUT PRMT path, and consumed
    by true RS WGMMA.  Folded uint8 K16 scale bytes use the existing
    weight-side auxiliary pipeline in lockstep with the AB pipeline.
    """

    _fp4_format_name = "NVFP4"
    _folded_scale_rows = 32
    _folded_scale_block_bytes = MXFP4_FOLD_BLOCK_BYTES
    _make_scale_smem_layout = staticmethod(make_offset_smem_layout)
    _make_scale_smem_layout_k256 = staticmethod(make_offset_smem_layout_k256)
    _make_expanded_scale_view = staticmethod(make_expanded_offset_view)
    _make_expanded_scale_view_k256 = staticmethod(make_expanded_offset_view_k256)

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.get("fp8_scale_mode", "per_tensor") != "mxfp4_hybrid":
            raise NotImplementedError("NVFP4 requires fp8_scale_mode='mxfp4_hybrid'")
        # Standalone FC12 construction keeps the baseline conversion policy.
        self.nvfp4_fc1_k256_overlap = False
        self.nvfp4_conversion = NVFP4_DEFAULT_CONVERSION
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # NV scale storage and shared conversion table
    # ------------------------------------------------------------------

    @cute.jit
    def _convert_weight_group(
        self,
        packed_registers,
        fp8_fragment_a,
        partitioned_offsets,
        local_begin: cutlass.Constexpr,
        global_begin: cutlass.Constexpr,
        count: cutlass.Constexpr,
        stage_idx,
        smem_weight_sf,
    ):
        convert_nvfp4_group(
            packed_registers,
            fp8_fragment_a,
            partitioned_offsets,
            local_begin,
            global_begin,
            count,
            stage_idx,
            smem_weight_sf,
            self.nvfp4_conversion,
        )

    def _weight_aux_smem_layout_staged(self) -> cute.Layout:
        stages = self._weight_sf_stage_count()
        if self.mma_tiler[2] == 2 * MXFP4_K_TILE:
            return make_offset_smem_layout_k256(self.mma_tiler[0], stages)
        return make_offset_smem_layout(self.mma_tiler[0], stages)

    def _weight_sf_decoupled(self) -> bool:
        return self.nvfp4_conversion.sf_stages != 0

    def _weight_sf_stage_count(self) -> int:
        if self._weight_sf_decoupled():
            return self.nvfp4_conversion.sf_stages
        return self.num_ab_stage

    def _weight_sf_lag(self, linear1: bool) -> int:
        """AB stages by which a tile's scale copies trail its weight copies.

        FC1 K256 keeps an AB slot until the next stage retires (two slots
        held), FC2 releases it at the end of the stage (one slot). With this
        lag the scale-slot wait ends no later than the next weight-slot wait
        would, and the scales of a stage arrive one to 1.5 stages ahead of
        use. The deadlock bound is lag <= num_ab_stage - held.
        """
        held = 2 if linear1 else 1
        return max(0, self.num_ab_stage - held - self.nvfp4_conversion.sf_stages)

    def _weight_sf_state(self, ab_state):
        """Scale-ring consumer state for the AB stage ``ab_state`` points at.

        The AB ring holds a whole number of scale-ring round trips (see
        _compute_stages), so the scale slot and phase follow from the AB slot.
        """
        if not self._weight_sf_decoupled():
            return ab_state
        stages = self.nvfp4_conversion.sf_stages
        assert self.num_ab_stage % (2 * stages) == 0, (
            "AB stages must be divisible by twice the NVFP4 scale-ring depth"
        )
        shift = stages.bit_length() - 1
        return pipeline.PipelineState(
            stages,
            ab_state.count,
            ab_state.index & cutlass.Int32(stages - 1),
            (ab_state.index >> cutlass.Int32(shift)) & cutlass.Int32(1),
        )

    def _validate_weight_sf_ring(self) -> None:
        if not self._weight_sf_decoupled():
            return
        if (
            self.mma_tiler[2] != 2 * MXFP4_K_TILE
            or self.fp8_scale_mode != "mxfp4_hybrid"
            or self.pingpong
        ):
            raise ValueError(
                "nvfp4_sf_stages requires the hybrid K256 schedule without pingpong"
            )

    @cute.jit
    def _prefetch_weight_scale_l2_k256(
        self,
        weight_sf_gemm: cute.Tensor,
        work_tile_info,
        k_tile,
    ) -> None:
        """Warm L2 with one K256 stage of scales (decoupled scale ring).

        The stage's K128 pair is contiguous per M64 block, as in
        _copy_weight_scale_bulk_k256; padded M64 blocks are skipped.
        """
        blocks = self.mma_tiler[0] // MXFP4_FOLD_M
        bytes_per_block = 2 * MXFP4_FOLD_BLOCK_BYTES
        with cute.arch.elect_one():
            for block in cutlass.range_constexpr(blocks):
                global_m64 = work_tile_info.tile_m_idx * cutlass.Int32(
                    blocks
                ) + cutlass.Int32(block)
                gmem_iter = weight_sf_gemm.iterator + cute.crd2idx(
                    (
                        work_tile_info.expert_idx,
                        global_m64,
                        k_tile * cutlass.Int32(2),
                        0,
                        0,
                    ),
                    weight_sf_gemm.layout,
                )
                if cutlass.const_expr(self._weight_offset_m_padding_possible()):
                    if global_m64 < weight_sf_gemm.shape[1]:
                        _prefetch_offset_bulk_l2(
                            gmem_iter, cutlass.Int32(bytes_per_block)
                        )
                else:
                    _prefetch_offset_bulk_l2(gmem_iter, cutlass.Int32(bytes_per_block))

    @cute.jit
    def _tma_load_a_with_decoupled_weight_sf_task_tile(
        self,
        tma_atom,
        tAgA_slice,
        tAsA,
        desc_ptr_a,
        weight_sf_gemm,
        smem_weight_sf,
        ab_producer,
        weight_sf_producer,
        work_tile_info,
        output_scale_block_base,
        k_tile_cnt,
        tidx,
        mcast_mask,
    ):
        lag = cutlass.Int32(self._weight_sf_lag(False))
        if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
            lag = cutlass.Int32(self._weight_sf_lag(True))
        peek_ab_empty_status = ab_producer.try_acquire()
        for k_tile in cutlass.range(0, k_tile_cnt + lag, 1, unroll=1):
            if k_tile < k_tile_cnt:
                ab_handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_handle.count + 1 < k_tile_cnt:
                    peek_ab_empty_status = ab_producer.try_acquire()
                cute.copy(
                    tma_atom,
                    tAgA_slice[None, ab_handle.count],
                    tAsA[None, ab_handle.index],
                    tma_bar_ptr=ab_handle.barrier,
                    tma_desc_ptr=desc_ptr_a,
                    mcast_mask=mcast_mask,
                )
                self._prefetch_weight_scale_l2_k256(
                    weight_sf_gemm, work_tile_info, ab_handle.count
                )
            if k_tile >= lag:
                scale_handle = weight_sf_producer.acquire_and_advance()
                self._copy_weight_scale_cpasync_k256(
                    weight_sf_gemm=weight_sf_gemm,
                    smem_weight_sf=smem_weight_sf,
                    work_tile_info=work_tile_info,
                    output_scale_block_base=output_scale_block_base,
                    scale_handle=scale_handle,
                    tidx=tidx,
                )
        return (ab_producer, weight_sf_producer)

    @cute.jit
    def _allocate_weight_aux_smem(self, smem, layout, tidx):
        storage = smem.allocate_tensor(
            element_type=self._weight_aux_smem_dtype(),
            layout=cute.make_layout(
                cute.cosize(layout) + self.nvfp4_conversion.lut_bytes
            ),
            byte_alignment=16,
        )
        scales = cute.make_tensor(storage.iterator, layout)
        nvfp4_initialize_lut(nvfp4_get_lut(scales, self.nvfp4_conversion), tidx)
        cute.arch.sync_threads()
        return scales

    def _compute_stages(
        self,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk,
        a_dtype,
        b_dtype,
        c_bytes_total: int,
        smem_capacity: int,
        occupancy: int,
        num_sched_stages: int,
    ):
        del tiled_mma, a_dtype
        a_layout = sm90_utils.make_smem_layout_a(
            self.a_layout,
            mma_tiler_mnk,
            cutlass.Float4E2M1FN,
            1,
        )
        b_layout = sm90_utils.make_smem_layout_b(
            self.b_layout,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        self._validate_weight_sf_ring()
        self.nvfp4_smem_budget = nvfp4_shared_memory_budget(
            smem_capacity=smem_capacity,
            occupancy=occupancy,
            ab_data_bytes=(
                cute.size_in_bytes(cutlass.Float4E2M1FN, a_layout)
                + cute.size_in_bytes(b_dtype, b_layout)
            ),
            activation_sf_bytes=self._activation_sf_bytes_per_stage(),
            weight_sf_bytes=self._weight_sf_bytes_per_stage(),
            misc_bytes=self._smem_misc_budget_bytes(),
            c_bytes_total=c_bytes_total,
            conversion=self.nvfp4_conversion,
        )
        num_ab_stage = self.nvfp4_smem_budget["num_ab_stage"]
        return 1, num_ab_stage, num_sched_stages

    # ------------------------------------------------------------------
    # LDSM -> PRMT -> RS WGMMA consumer
    # ------------------------------------------------------------------

    @cute.jit
    def _mma_mxfp4_hybrid_fc1_k256(
        self,
        local_warp_idx: int,
        tiled_mma,
        packed_smem_a: cute.Tensor,
        tCrB: cute.Tensor,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        ab_pipeline,
        weight_sf_pipeline,
        ab_consumer_state,
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        k_tile_cnt,
        n_half,
        tidx,
    ):
        """Run the baseline or the FC1 K256 conversion/MMA overlap schedule."""
        if local_warp_idx < self.epilogue_warps_per_warpgroup:
            fp8_fragment_a = tiled_mma.make_fragment_A(
                tiled_mma.partition_shape_A((self.wgmma_tile_m, 2 * MXFP4_K_TILE))
            )
            (
                tiled_copy,
                smem_partition,
                copy_view,
                packed_registers,
            ) = make_packed_a_ldsm_views_k256(
                tiled_mma,
                packed_smem_a,
                fp8_fragment_a,
                tidx % cutlass.Int32(128),
            )
            expanded_offsets = make_expanded_offset_view_k256(
                smem_weight_sf,
                self.mma_tiler[0],
            )
            offsets_wg = cute.local_tile(
                expanded_offsets,
                cute.slice_(self.wgmma_tiler, (None, 0, None)),
                (n_half, 0, None),
            )
            thr_mma = tiled_mma.get_slice(tidx % cutlass.Int32(128))
            partitioned_offsets = thr_mma.partition_A(offsets_wg)
            release_state = ab_consumer_state.clone()

            accumulators.fill(0.0)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, False)

            if local_warp_idx == cutlass.Int32(0):
                iket.range_push("mx_f1_wait0")
            ab_pipeline.consumer_wait(ab_consumer_state)
            # Without a decoupled scale ring this is ab_consumer_state itself.
            sf_state0 = self._weight_sf_state(ab_consumer_state)
            weight_sf_pipeline.consumer_wait(sf_state0)
            if local_warp_idx == cutlass.Int32(0):
                iket.range_pop()
            stage_idx = ab_consumer_state.index
            sf_stage_idx0 = sf_state0.index
            cute.copy(
                tiled_copy,
                smem_partition[(None, None, None, stage_idx)],
                copy_view,
            )
            if local_warp_idx == cutlass.Int32(0):
                iket.range_push("mx_f1_issue0")
            convert_nvfp4_group(
                packed_registers,
                fp8_fragment_a,
                partitioned_offsets,
                0,
                0,
                (4) - (0),
                sf_stage_idx0,
                smem_weight_sf,
                self.nvfp4_conversion,
            )
            if cutlass.const_expr(self.nvfp4_fc1_k256_overlap):
                warpgroup.fence()
            for k_block in cutlass.range_constexpr(4):
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_push("mx_f1_cvt0")
                if cutlass.const_expr(not self.nvfp4_fc1_k256_overlap):
                    warpgroup.fence()
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_pop()
                    iket.range_push("mx_f1_mma0")
                cute.gemm(
                    tiled_mma,
                    accum_temp,
                    fp8_fragment_a[(None, None, k_block)],
                    tCrB[(None, None, k_block, stage_idx)],
                    accum_temp,
                )
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_pop()
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            if local_warp_idx == cutlass.Int32(0):
                iket.range_pop()
            if cutlass.const_expr(not self.nvfp4_fc1_k256_overlap):
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_push("mx_f1_wgwait0")
                warpgroup.wait_group(0)
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_pop()
            if local_warp_idx == cutlass.Int32(0):
                iket.range_push("mx_f1_issue1")
            convert_nvfp4_group(
                packed_registers,
                fp8_fragment_a,
                partitioned_offsets,
                4,
                4,
                (8) - (4),
                sf_stage_idx0,
                smem_weight_sf,
                self.nvfp4_conversion,
            )
            if cutlass.const_expr(self._weight_sf_decoupled()):
                # The stage's last scale read is done; free the scale slot.
                weight_sf_pipeline.consumer_release(sf_state0)
            if cutlass.const_expr(self.nvfp4_fc1_k256_overlap):
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_push("mx_f1_wgwait0")
                warpgroup.wait_group(0)
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_pop()
                # RS A registers remain asynchronous WGMMA operands until wait.
                # Keep the first half live while converting into the second half.
                for keep_k in cutlass.range_constexpr(4):
                    issued_words = cute.recast_tensor(
                        fp8_fragment_a[(None, None, keep_k)], cutlass.Uint32
                    )
                    for keep_word in cutlass.range_constexpr(cute.size(issued_words)):
                        _keep_nvfp4_a_word_live(issued_words[keep_word])
                warpgroup.fence()
            for k_block in cutlass.range_constexpr(4, 8):
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_push("mx_f1_cvt1")
                if cutlass.const_expr(not self.nvfp4_fc1_k256_overlap):
                    warpgroup.fence()
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_pop()
                    iket.range_push("mx_f1_mma1")
                cute.gemm(
                    tiled_mma,
                    accum_temp,
                    fp8_fragment_a[(None, None, k_block)],
                    tCrB[(None, None, k_block, stage_idx)],
                    accum_temp,
                )
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_pop()
            warpgroup.commit_group()
            if local_warp_idx == cutlass.Int32(0):
                iket.range_pop()
            if local_warp_idx == cutlass.Int32(0):
                iket.range_push("mx_f1_wgwait1")
            warpgroup.wait_group(0)
            if local_warp_idx == cutlass.Int32(0):
                iket.range_pop()
            ab_consumer_state.advance()

            for k_tile in cutlass.range(1, k_tile_cnt, 1, unroll=1):
                ab_pipeline.consumer_wait(ab_consumer_state)
                sf_state = self._weight_sf_state(ab_consumer_state)
                weight_sf_pipeline.consumer_wait(sf_state)
                stage_idx = ab_consumer_state.index
                sf_stage_idx = sf_state.index
                cute.copy(
                    tiled_copy,
                    smem_partition[(None, None, None, stage_idx)],
                    copy_view,
                )
                convert_nvfp4_group(
                    packed_registers,
                    fp8_fragment_a,
                    partitioned_offsets,
                    0,
                    0,
                    (4) - (0),
                    sf_stage_idx,
                    smem_weight_sf,
                    self.nvfp4_conversion,
                )
                if cutlass.const_expr(self.nvfp4_fc1_k256_overlap):
                    warpgroup.fence()
                for k_block in cutlass.range_constexpr(4):
                    if cutlass.const_expr(not self.nvfp4_fc1_k256_overlap):
                        warpgroup.fence()
                    cute.gemm(
                        tiled_mma,
                        accum_temp,
                        fp8_fragment_a[(None, None, k_block)],
                        tCrB[(None, None, k_block, stage_idx)],
                        accum_temp,
                    )
                warpgroup.commit_group()
                if cutlass.const_expr(not self.nvfp4_fc1_k256_overlap):
                    warpgroup.wait_group(0)
                convert_nvfp4_group(
                    packed_registers,
                    fp8_fragment_a,
                    partitioned_offsets,
                    4,
                    4,
                    (8) - (4),
                    sf_stage_idx,
                    smem_weight_sf,
                    self.nvfp4_conversion,
                )
                if cutlass.const_expr(self._weight_sf_decoupled()):
                    weight_sf_pipeline.consumer_release(sf_state)
                if cutlass.const_expr(self.nvfp4_fc1_k256_overlap):
                    warpgroup.wait_group(0)
                    # RS A registers remain asynchronous WGMMA operands until wait.
                    # Keep the first half live while converting into the second half.
                    for keep_k in cutlass.range_constexpr(4):
                        issued_words = cute.recast_tensor(
                            fp8_fragment_a[(None, None, keep_k)], cutlass.Uint32
                        )
                        for keep_word in cutlass.range_constexpr(
                            cute.size(issued_words)
                        ):
                            _keep_nvfp4_a_word_live(issued_words[keep_word])
                    warpgroup.fence()
                for k_block in cutlass.range_constexpr(4, 8):
                    if cutlass.const_expr(not self.nvfp4_fc1_k256_overlap):
                        warpgroup.fence()
                    cute.gemm(
                        tiled_mma,
                        accum_temp,
                        fp8_fragment_a[(None, None, k_block)],
                        tCrB[(None, None, k_block, stage_idx)],
                        accum_temp,
                    )
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                if cutlass.const_expr(not self._weight_sf_decoupled()):
                    weight_sf_pipeline.consumer_release(release_state)
                ab_pipeline.consumer_release(release_state)
                release_state.advance()
                ab_consumer_state.advance()

            # Keep the accepted K256 baseline explicit: every outstanding
            # WGMMA group must be retired before the final scale promotion.
            # Keep this synchronization point after both per-half waits to retire
            # all WGMMA work before scaling the accumulators.
            warpgroup.wait_group(0)
            # FC1's per-token scale is invariant over K. Read it from the
            # final still-owned stage after all WGMMA retire so no scale
            # fragment remains live across the K256 mainloop.
            self._promote_accum_temp_blockwise_streaming(
                accumulators=accumulators,
                accum_temp=accum_temp,
                smem_activation_sf=smem_activation_sf,
                stage_idx=stage_idx,
                scale_plane=cutlass.Int32(0),
                tidx=tidx,
            )
            if cutlass.const_expr(not self._weight_sf_decoupled()):
                weight_sf_pipeline.consumer_release(release_state)
            ab_pipeline.consumer_release(release_state)

        return ab_consumer_state


__all__ = ["Sm90SwapABSwigluNvfp4Fp8Fc12Kernel"]


from moe_hopper_fp8.megamoe_kernel_fp8 import Sm90MegaMoESwapABMxfp4Fp8Kernel


class Sm90MegaMoESwapABNvfp4Fp8Kernel(
    Sm90MegaMoESwapABMxfp4Fp8Kernel, Sm90SwapABSwigluNvfp4Fp8Fc12Kernel
):
    """MegaMoE with packed NVFP4 weights and optional FC1 conversion overlap."""

    def __init__(
        self,
        *args,
        mxfp4_fc2_tail_n8=False,
        mxfp4_fc1_ready_mode="tile",
        nvfp4_fc1_k256_overlap=False,
        nvfp4_conversion=NVFP4_DEFAULT_CONVERSION,
        **kwargs,
    ):
        if not isinstance(nvfp4_conversion, Nvfp4ConversionLayout):
            raise ValueError("nvfp4_conversion must be an Nvfp4ConversionLayout")
        super().__init__(
            *args,
            mxfp4_fc2_tail_n8=mxfp4_fc2_tail_n8,
            mxfp4_fc1_ready_mode=mxfp4_fc1_ready_mode,
            **kwargs,
        )
        if not isinstance(nvfp4_fc1_k256_overlap, bool):
            raise ValueError("nvfp4_fc1_k256_overlap must be a bool")
        if nvfp4_fc1_k256_overlap and (
            self.fp8_scale_mode != "mxfp4_hybrid"
            or self.mma_tiler[2] != 256
            or self.pingpong
        ):
            raise ValueError("NVFP4 FC1 overlap requires hybrid K256 without pingpong")
        self.nvfp4_fc1_k256_overlap = nvfp4_fc1_k256_overlap
        if self.fp8_scale_mode != "mxfp4_hybrid":
            raise NotImplementedError("NVFP4 requires fp8_scale_mode='mxfp4_hybrid'")
        self.nvfp4_conversion = nvfp4_conversion
