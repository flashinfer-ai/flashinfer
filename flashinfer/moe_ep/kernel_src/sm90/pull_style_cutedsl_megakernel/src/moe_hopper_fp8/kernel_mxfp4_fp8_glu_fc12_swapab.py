# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Hopper Humming MXFP4-weight x FP8-activation fused FC1+FC2 kernel."""

from dataclasses import replace

from typing import Type

import cutlass
import cutlass.cute as cute
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover
    from src.iket_compat import iket
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.typing import Float32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils

from moe_hopper_fp8.kernel_fp8_glu_fc12_swapab import (
    Sm90SwapABSwigluFp8Fc12Kernel,
)
from moe_hopper_fp8.mxfp4_cutedsl import (
    MXFP4_FOLD_BLOCK_BYTES,
    MXFP4_FOLD_M,
    MXFP4_K_TILE,
    convert_packed_a_kblock,
    convert_packed_a_kblock_from_offset,
    make_expanded_offset_view,
    make_expanded_offset_view_k256,
    make_offset_smem_layout,
    make_offset_smem_layout_k256,
    make_packed_a_ldsm_views,
    make_packed_a_ldsm_views_k256,
    make_packed_a_ldsm_views_k256_half,
)
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase
from moe_hopper_fp8.mxfp4_policy import resolve_mxfp4_optimizations


@dsl_user_op
def _copy_offset_row_zfill(dst_smem, src_gmem, src_bytes, *, loc=None, ip=None):
    """Keep the ca policy; src-size zero clears a padded 16-byte offset row."""
    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src_gmem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src_bytes.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy_offset_bulk(dst_smem, src_gmem, mbar, num_bytes, *, loc=None, ip=None):
    """Copy folded offsets without changing their layout or cache policy."""
    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src_gmem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            num_bytes.ir_value(loc=loc, ip=ip),
            mbar.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

class Sm90SwapABSwigluMxfp4Fp8Fc12Kernel(
    Sm90SwapABSwigluFp8Fc12Kernel
):
    """Packed E2M1 weight / E4M3 activation specialization for SM90.

    Weight payloads remain packed in GMEM and SMEM.  Every K128 stage is loaded
    with LDSM, converted in registers with the Humming PRMT path, and consumed
    by true RS WGMMA.  Folded uint8 exponent offsets use the existing
    weight-side auxiliary pipeline in lockstep with the AB pipeline.
    """

    def __init__(self, *args, **kwargs) -> None:
        ab_dtype = kwargs.pop("ab_dtype", cutlass.Float8E4M3FN)
        fp8_scale_mode = kwargs.pop("fp8_scale_mode", "per_tensor")
        fp8_accum_mode = kwargs.pop("fp8_accum_mode", "1xacc")
        if ab_dtype is not cutlass.Float8E4M3FN:
            raise ValueError(
                "Hopper MXFP4xFP8 requires FP8 E4M3 activation/output; "
                f"got ab_dtype={ab_dtype}."
            )
        if fp8_scale_mode not in ("per_tensor", "mxfp4_hybrid"):
            raise NotImplementedError(
                "Hopper MXFP4xFP8 supports fp8_scale_mode='per_tensor' or "
                "'mxfp4_hybrid'; generic FP8 blockwise weight scaling is "
                "intentionally unsupported."
            )
        if fp8_accum_mode != "1xacc":
            raise NotImplementedError(
                "Hopper MXFP4xFP8 supports only fp8_accum_mode='1xacc'."
            )

        mma_tiler_mnk = kwargs.get("mma_tiler_mnk")
        if mma_tiler_mnk is None and args:
            mma_tiler_mnk = args[0]
        if mma_tiler_mnk is None:
            raise ValueError("mma_tiler_mnk is required")
        if mma_tiler_mnk[2] not in (MXFP4_K_TILE, 2 * MXFP4_K_TILE):
            raise ValueError(
                "Hopper MXFP4xFP8 requires mma_tiler K=128 or K=256; "
                f"got {mma_tiler_mnk[2]}."
            )
        if mma_tiler_mnk[2] == 2 * MXFP4_K_TILE and fp8_scale_mode != "mxfp4_hybrid":
            raise ValueError(
                "The experimental MXFP4 K=256 tactic is available only for "
                "fp8_scale_mode='mxfp4_hybrid'."
            )

        static_shape = kwargs.get("static_expert_shape")
        if static_shape is not None:
            _, intermediate_gateup, hidden = static_shape
            logical_ks = (
                ("hidden", hidden),
                ("intermediate_downproj", intermediate_gateup // 2),
            )
            for name, logical_k in logical_ks:
                if logical_k % mma_tiler_mnk[2] != 0:
                    raise ValueError(
                        f"Hopper MXFP4xFP8 requires {name} ({logical_k}) "
                        f"divisible by tile K={mma_tiler_mnk[2]}."
                    )

        super().__init__(
            *args,
            **kwargs,
            ab_dtype=cutlass.Float8E4M3FN,
            fp8_scale_mode=fp8_scale_mode,
            fp8_accum_mode="1xacc",
        )
        if self.tail_split_pairs and getattr(self, "_mxfp4_fc1_ready_mode", "tile") != "tile":
            raise ValueError("MXFP4 tail_split_pairs requires fc1_ready_mode='tile'")
        self.is_mxfp4_fp8 = True

    # ------------------------------------------------------------------
    # Shared swap-AB policy hooks
    # ------------------------------------------------------------------

    def _weight_storage_k(self, logical_k):
        return logical_k // 2

    def _logical_weight_k(self, storage_k):
        return storage_k * 2

    def _mma_a_dtype(self, gmem_dtype: Type[cutlass.Numeric]):
        if gmem_dtype not in (cutlass.Int8, cutlass.Uint8):
            raise ValueError(
                "Hopper MXFP4xFP8 packed weights must use an 8-bit shell; "
                f"got {gmem_dtype}."
            )
        return cutlass.Float8E4M3FN

    def _a_smem_dtype(self) -> Type[cutlass.Numeric]:
        return cutlass.Float4E2M1FN

    def _a_tma_tiler(self):
        return (self.mma_tiler[0], self.mma_tiler[2] // 2)

    @cute.jit
    def _a_tma_gmem_tensor(self, physical_a: cute.Tensor) -> cute.Tensor:
        return physical_a

    def _a_tma_internal_type(self):
        return cutlass.Uint8

    def _a_tma_smem_layout(self, logical_a_smem_layout):
        return cute.make_composed_layout(
            logical_a_smem_layout.inner,
            0,
            cute.recast_layout(8, 4, logical_a_smem_layout.outer),
        )

    @cute.jit
    def _a_tma_smem_tensor(self, logical_smem_a: cute.Tensor) -> cute.Tensor:
        return cute.recast_tensor(logical_smem_a, cutlass.Uint8)

    def _uses_weight_aux_pipeline(self) -> bool:
        return True

    def _weight_aux_smem_dtype(self) -> Type[cutlass.Numeric]:
        return cutlass.Int8

    def _weight_aux_smem_layout_staged(self) -> cute.Layout:
        if self.mma_tiler[2] == 2 * MXFP4_K_TILE:
            return make_offset_smem_layout_k256(
                self.mma_tiler[0], self.num_ab_stage
            )
        return make_offset_smem_layout(
            self.mma_tiler[0], self.num_ab_stage
        )

    def _weight_sf_bytes_per_stage(self) -> int:
        bytes_per_k128 = (
            self.mma_tiler[0] // MXFP4_FOLD_M
        ) * MXFP4_FOLD_BLOCK_BYTES
        if self.mma_tiler[2] == 2 * MXFP4_K_TILE:
            return 2 * bytes_per_k128
        return bytes_per_k128

    def _setup_attributes(self) -> None:
        if self.b_dtype is not cutlass.Float8E4M3FN:
            raise ValueError(
                "Hopper MXFP4xFP8 activation must be Float8E4M3FN; "
                f"got {self.b_dtype}."
            )
        super()._setup_attributes()
        self.mxfp4_optimizations = replace(
            self._resolve_mxfp4_optimizations(),
            peer32=self.fused_comm_optimizations.paired_bf16_stores,
            skip_zero_counts=self.fused_comm_optimizations.skip_zero_counts,
        )
        self.epilogue._mxfp4_fc1_ready_k256 = (
            self.mxfp4_optimizations.fc1_ready_mode == "k256"
        )

    def _resolve_fused_comm_optimizations(
        self, *, paired_stores=None, skip_zero_counts=None,
    ):
        shared = super()._resolve_fused_comm_optimizations(
            paired_stores=True if paired_stores is None else paired_stores,
            skip_zero_counts=True if skip_zero_counts is None else skip_zero_counts,
        )
        mxfp4 = self._resolve_mxfp4_optimizations()
        return replace(
            shared,
            paired_bf16_stores=shared.paired_bf16_stores and mxfp4.peer32,
            skip_zero_counts=shared.skip_zero_counts and mxfp4.skip_zero_counts,
        )

    def _resolve_mxfp4_optimizations(self):
        return resolve_mxfp4_optimizations(
            fp8_scale_mode=self.fp8_scale_mode,
            mma_tiler_mnk=self.mma_tiler_mnk,
            cluster_shape_mnk=(*self.cluster_shape_mn, 1),
            static_expert_shape=self.static_expert_shape,
            world_size=getattr(self, "world_size", 1),
            pingpong=self.pingpong,
            token_back_by_dispatch=self.token_back_by_dispatch,
            fc2_in_kernel_topk_reduce=self.fc2_in_kernel_topk_reduce,
            fc1_early_done_publish=self.fc1_early_done_publish,
            fc1_store_offload=self.fc1_store_offload,
            dedup_dispatch=getattr(self, "dedup_dispatch", False),
            fc2_tail_n8=getattr(self, "_mxfp4_fc2_tail_n8", False),
            fc1_ready_mode=getattr(self, "_mxfp4_fc1_ready_mode", "tile"),
        )

    def _create_tiled_mma(self) -> cute.TiledMma:
        return sm90_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.wgmma_tile_n),
            a_source=warpgroup.OperandSource.RMEM,
        )

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
        bytes_per_stage = (
            cute.size_in_bytes(cutlass.Float4E2M1FN, a_layout)
            + cute.size_in_bytes(b_dtype, b_layout)
            + self._activation_sf_bytes_per_stage()
            + self._weight_sf_bytes_per_stage()
        )
        fixed_overhead = self._smem_misc_budget_bytes() + c_bytes_total
        num_ab_stage = (
            smem_capacity // occupancy - fixed_overhead
        ) // bytes_per_stage
        if num_ab_stage < 2:
            raise ValueError(
                "Hopper MXFP4xFP8 requires at least two AB/offset stages; "
                f"computed {num_ab_stage}."
            )
        return 1, num_ab_stage, num_sched_stages

    # ------------------------------------------------------------------
    # Packed weight + folded offset producer
    # ------------------------------------------------------------------

    def _weight_offset_m_padding_possible(self) -> bool:
        # The scheduler rounds the output axis to whole clusters, including
        # otherwise empty CTAs. Preserve the unpredicated fast path when both
        # FC1 and FC2 have statically complete weight clusters.
        if self.static_expert_shape is None:
            return True
        _, fc1_channels, fc2_channels = self.static_expert_shape
        cluster_channels = self.mma_tiler[0] * self.cluster_shape_mn[0]
        return (
            fc1_channels % cluster_channels != 0
            or fc2_channels % cluster_channels != 0
        )

    @cute.jit
    def _copy_weight_scale_cpasync(
        self,
        weight_sf_gemm: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        work_tile_info,
        output_scale_block_base,
        scale_handle,
        tidx,
    ):
        del output_scale_block_base
        lane_idx = tidx % cutlass.Int32(32)
        vectors_per_stage = (
            self.mma_tiler[0] // MXFP4_FOLD_M
        ) * 16
        vectors_per_lane = vectors_per_stage // 32
        m64_blocks_per_tile = self.mma_tiler[0] // MXFP4_FOLD_M
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int8,
            num_bits_per_copy=128,
        )

        for item in cutlass.range_constexpr(vectors_per_lane):
            vector_idx = lane_idx + cutlass.Int32(item * 32)
            local_m64 = vector_idx // cutlass.Int32(16)
            folded_row = vector_idx % cutlass.Int32(16)
            global_m64 = (
                work_tile_info.tile_m_idx
                * cutlass.Int32(m64_blocks_per_tile)
                + local_m64
            )
            gmem_iter = (
                weight_sf_gemm.iterator
                + cute.crd2idx(
                    (
                        work_tile_info.expert_idx,
                        global_m64,
                        scale_handle.count,
                        folded_row,
                        0,
                    ),
                    weight_sf_gemm.layout,
                )
            )
            # Every folded row starts at a 16-byte boundary by construction:
            # the final physical dimension is exactly 16 contiguous bytes.
            # Re-assert that invariant after dynamic expert/tile arithmetic so
            # the cp.async verifier can prove the 128-bit source alignment.
            gmem_vec = cute.make_tensor(
                cute.make_ptr(
                    gmem_iter.dtype,
                    gmem_iter.toint(),
                    gmem_iter.memspace,
                    assumed_align=16,
                ),
                cute.make_layout(16),
            )
            smem_vec = cute.make_tensor(
                smem_weight_sf.iterator
                + cute.crd2idx(
                    (
                        0,
                        folded_row,
                        local_m64,
                        0,
                        scale_handle.index,
                    ),
                    smem_weight_sf.layout,
                ),
                cute.make_layout(16),
            )
            if cutlass.const_expr(self._weight_offset_m_padding_possible()):
                src_bytes = cutlass.Int32(16)
                if global_m64 >= weight_sf_gemm.shape[1]:
                    src_bytes = cutlass.Int32(0)
                _copy_offset_row_zfill(
                    smem_vec.iterator, gmem_vec.iterator, src_bytes,
                )
            else:
                cute.copy(copy_atom, gmem_vec, smem_vec)

        scale_handle.commit()

    @cute.jit
    def _copy_weight_scale_bulk_k256(
        self,
        weight_sf_gemm: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        work_tile_info,
        scale_handle,
    ) -> None:
        """Copy each pair of contiguous K128 offset blocks in one request.

        Validated GMEM is uint8 [E,M64,K128,16,16]. The unchanged SMEM
        strides (1,16,512,256,stage_bytes) make one K256 pair contiguous
        in both spaces. Arm all completion bytes before issuing copies,
        and retain the original 32 producer arrivals.
        """
        blocks = self.mma_tiler[0] // MXFP4_FOLD_M
        bytes_per_block = 2 * MXFP4_FOLD_BLOCK_BYTES
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(
                scale_handle.barrier, blocks * bytes_per_block
            )
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            for block in cutlass.range_constexpr(blocks):
                global_m64 = (
                    work_tile_info.tile_m_idx * cutlass.Int32(blocks)
                    + cutlass.Int32(block)
                )
                global_k128 = scale_handle.count * cutlass.Int32(2)
                gmem_iter = weight_sf_gemm.iterator + cute.crd2idx(
                    (work_tile_info.expert_idx, global_m64, global_k128, 0, 0),
                    weight_sf_gemm.layout,
                )
                smem_iter = smem_weight_sf.iterator + cute.crd2idx(
                    (0, 0, cutlass.Int32(block), 0, scale_handle.index),
                    smem_weight_sf.layout,
                )
                _copy_offset_bulk(
                    smem_iter, gmem_iter, scale_handle.barrier,
                    cutlass.Int32(bytes_per_block),
                )
        # Bulk copies complete transaction bytes on this barrier directly.
        # PipelineCpAsync.commit() uses cp.async.mbarrier.arrive, which only
        # tracks non-bulk cp.async; submit the 32 producer arrivals normally.
        cute.arch.mbarrier_arrive(scale_handle.barrier)

    @cute.jit
    def _copy_weight_scale_cpasync_k256(
        self,
        weight_sf_gemm: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        work_tile_info,
        output_scale_block_base,
        scale_handle,
        tidx,
    ) -> None:
        """Use bulk for complete CTAs and one cp.async protocol for M padding."""
        if cutlass.const_expr(self.mxfp4_optimizations.offset_bulk):
            if cutlass.const_expr(self._weight_offset_m_padding_possible()):
                blocks = self.mma_tiler[0] // MXFP4_FOLD_M
                tile_end = (
                    work_tile_info.tile_m_idx + cutlass.Int32(1)
                ) * cutlass.Int32(blocks)
                if tile_end <= weight_sf_gemm.shape[1]:
                    self._copy_weight_scale_bulk_k256(
                        weight_sf_gemm=weight_sf_gemm,
                        smem_weight_sf=smem_weight_sf,
                        work_tile_info=work_tile_info,
                        scale_handle=scale_handle,
                    )
                else:
                    self._copy_weight_scale_cpasync_k256_rows(
                        weight_sf_gemm=weight_sf_gemm,
                        smem_weight_sf=smem_weight_sf,
                        work_tile_info=work_tile_info,
                        output_scale_block_base=output_scale_block_base,
                        scale_handle=scale_handle,
                        tidx=tidx,
                    )
            else:
                self._copy_weight_scale_bulk_k256(
                    weight_sf_gemm=weight_sf_gemm,
                    smem_weight_sf=smem_weight_sf,
                    work_tile_info=work_tile_info,
                    scale_handle=scale_handle,
                )
        else:
            self._copy_weight_scale_cpasync_k256_rows(
                weight_sf_gemm=weight_sf_gemm,
                smem_weight_sf=smem_weight_sf,
                work_tile_info=work_tile_info,
                output_scale_block_base=output_scale_block_base,
                scale_handle=scale_handle,
                tidx=tidx,
            )

    @cute.jit
    def _copy_weight_scale_cpasync_k256_rows(
        self,
        weight_sf_gemm: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        work_tile_info,
        output_scale_block_base,
        scale_handle,
        tidx,
    ) -> None:
        """Stage one K256 tile with ordinary cp.async, zero-filling M padding."""
        del output_scale_block_base
        lane_idx = tidx % cutlass.Int32(32)
        k128_blocks_per_tile = 2
        vectors_per_m64 = 16 * k128_blocks_per_tile
        vectors_per_stage = (
            self.mma_tiler[0] // MXFP4_FOLD_M
        ) * vectors_per_m64
        vectors_per_lane = vectors_per_stage // 32
        m64_blocks_per_tile = self.mma_tiler[0] // MXFP4_FOLD_M
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int8,
            num_bits_per_copy=128,
        )

        for item in cutlass.range_constexpr(vectors_per_lane):
            vector_idx = lane_idx + cutlass.Int32(item * 32)
            local_m64 = vector_idx // cutlass.Int32(vectors_per_m64)
            vector_in_m64 = vector_idx % cutlass.Int32(vectors_per_m64)
            local_k128 = vector_in_m64 // cutlass.Int32(16)
            folded_row = vector_in_m64 % cutlass.Int32(16)
            global_m64 = (
                work_tile_info.tile_m_idx
                * cutlass.Int32(m64_blocks_per_tile)
                + local_m64
            )
            global_k128 = (
                scale_handle.count * cutlass.Int32(k128_blocks_per_tile)
                + local_k128
            )
            gmem_iter = (
                weight_sf_gemm.iterator
                + cute.crd2idx(
                    (
                        work_tile_info.expert_idx,
                        global_m64,
                        global_k128,
                        folded_row,
                        0,
                    ),
                    weight_sf_gemm.layout,
                )
            )
            gmem_vec = cute.make_tensor(
                cute.make_ptr(
                    gmem_iter.dtype,
                    gmem_iter.toint(),
                    gmem_iter.memspace,
                    assumed_align=16,
                ),
                cute.make_layout(16),
            )
            smem_vec = cute.make_tensor(
                smem_weight_sf.iterator
                + cute.crd2idx(
                    (
                        0,
                        folded_row,
                        local_m64,
                        local_k128,
                        scale_handle.index,
                    ),
                    smem_weight_sf.layout,
                ),
                cute.make_layout(16),
            )
            if cutlass.const_expr(self._weight_offset_m_padding_possible()):
                src_bytes = cutlass.Int32(16)
                if global_m64 >= weight_sf_gemm.shape[1]:
                    src_bytes = cutlass.Int32(0)
                _copy_offset_row_zfill(
                    smem_vec.iterator, gmem_vec.iterator, src_bytes,
                )
            else:
                cute.copy(copy_atom, gmem_vec, smem_vec)

        scale_handle.commit()

    @cute.jit
    def _tma_load_a_with_weight_sf_task_tile(
        self,
        tma_atom,
        real_a: cute.Tensor,
        desc_ptr_a,
        sA: cute.Tensor,
        weight_sf_gemm: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        ab_producer,
        weight_sf_producer,
        work_tile_info,
        tile_m_idx,
        output_scale_block_base,
        k_tile_cnt,
        tidx,
        tma_cta_coord,
        tma_cta_layout,
        mcast_mask,
        _iket_active,
    ):
        gA_mkl = cute.local_tile(
            real_a,
            self._a_tma_tiler(),
            (None, None, None),
        )
        sA_tma = self._a_tma_smem_tensor(sA)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom,
            tma_cta_coord,
            tma_cta_layout,
            cute.group_modes(sA_tma, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        tAgA_slice = tAgA[(None, tile_m_idx, None, 0)]
        ab_producer.reset()
        weight_sf_producer.reset()
        peek_ab_empty_status = ab_producer.try_acquire()
        peek_scale_empty_status = weight_sf_producer.try_acquire()
        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            if _iket_active:
                iket.range_push("ab_producer_acquire")
            ab_handle = ab_producer.acquire_and_advance(
                peek_ab_empty_status
            )
            if _iket_active:
                iket.range_pop()
                iket.range_push("weight_sf_producer_acquire")
            scale_handle = weight_sf_producer.acquire_and_advance(
                peek_scale_empty_status
            )
            if _iket_active:
                iket.range_pop()
            peek_ab_empty_status = cutlass.Boolean(1)
            peek_scale_empty_status = cutlass.Boolean(1)
            if ab_handle.count + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_producer.try_acquire()
                peek_scale_empty_status = weight_sf_producer.try_acquire()
            if _iket_active:
                iket.range_push("tma_operand_copy")
            cute.copy(
                tma_atom,
                tAgA_slice[(None, ab_handle.count)],
                tAsA[(None, ab_handle.index)],
                tma_bar_ptr=ab_handle.barrier,
                tma_desc_ptr=desc_ptr_a,
                mcast_mask=mcast_mask,
            )
            if _iket_active:
                iket.range_pop()
                iket.range_push("weight_sf_cpasync_copy")
            if cutlass.const_expr(self.mma_tiler[2] == 2 * MXFP4_K_TILE):
                self._copy_weight_scale_cpasync_k256(
                    weight_sf_gemm=weight_sf_gemm,
                    smem_weight_sf=smem_weight_sf,
                    work_tile_info=work_tile_info,
                    output_scale_block_base=output_scale_block_base,
                    scale_handle=scale_handle,
                    tidx=tidx,
                )
            else:
                self._copy_weight_scale_cpasync(
                    weight_sf_gemm=weight_sf_gemm,
                    smem_weight_sf=smem_weight_sf,
                    work_tile_info=work_tile_info,
                    output_scale_block_base=output_scale_block_base,
                    scale_handle=scale_handle,
                    tidx=tidx,
                )
            if _iket_active:
                iket.range_pop()
        return ab_producer, weight_sf_producer

    # ------------------------------------------------------------------
    # LDSM -> PRMT -> RS WGMMA consumer
    # ------------------------------------------------------------------

    def wgmma_warpgroup_init(
        self,
        tiled_mma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        wg_idx,
    ):
        warpgroup_thread_layout = cute.make_layout(
            self.wgmma_m_splits,
            stride=32 * self.epilogue_warps_per_warpgroup,
        )
        thr_mma = tiled_mma.get_slice(warpgroup_thread_layout(wg_idx))
        sA_wg = cute.local_tile(
            sA,
            cute.slice_(self.wgmma_tiler, (None, 0, None)),
            (wg_idx, 0, None),
        )
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
        cC = cute.make_identity_tensor(
            (self.wgmma_tiler[0], self.wgmma_tiler[1])
        )
        tCgC = thr_mma.partition_C(cC)
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.num_ab_stage,
        )
        # The first return slot deliberately carries packed staged SMEM A.  The
        # shared epilogue only forwards it to this specialization's mainloop.
        return sA_wg, tCrB, tCgC.shape[:3], consumer_state

    @cute.jit
    def _mma_mxfp4_per_tensor_1xacc(
        self,
        local_warp_idx: int,
        tiled_mma,
        packed_smem_a: cute.Tensor,
        tCrB: cute.Tensor,
        accumulators: cute.Tensor,
        ab_pipeline,
        weight_sf_pipeline,
        ab_consumer_state,
        smem_weight_sf: cute.Tensor,
        k_tile_cnt,
        n_half,
        tidx,
    ):
        if local_warp_idx < self.epilogue_warps_per_warpgroup:
            fp8_fragment_a = tiled_mma.make_fragment_A(
                tiled_mma.partition_shape_A(
                    (self.wgmma_tile_m, MXFP4_K_TILE)
                )
            )
            (
                tiled_copy,
                smem_partition,
                copy_view,
                packed_registers,
            ) = make_packed_a_ldsm_views(
                tiled_mma,
                packed_smem_a,
                fp8_fragment_a,
                tidx % cutlass.Int32(128),
            )

            expanded_offsets = make_expanded_offset_view(
                smem_weight_sf,
                self.mma_tiler[0],
            )
            offsets_wg = cute.local_tile(
                expanded_offsets,
                cute.slice_(self.wgmma_tiler, (None, 0, None)),
                (n_half, 0, None),
            )
            # A tiled MMA is local to one 128-thread warpgroup. ``tidx`` is
            # block-global, so fold the second warpgroup back onto the same
            # per-warpgroup lane numbering before partitioning the folded
            # Humming offsets.
            thr_mma = tiled_mma.get_slice(tidx % cutlass.Int32(128))
            partitioned_offsets = thr_mma.partition_A(offsets_wg)

            tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                ab_pipeline.consumer_wait(ab_consumer_state)
                weight_sf_pipeline.consumer_wait(ab_consumer_state)
                stage_idx = ab_consumer_state.index
                cute.copy(
                    tiled_copy,
                    smem_partition[(None, None, None, stage_idx)],
                    copy_view,
                )
                for k_block in cutlass.range_constexpr(
                    MXFP4_K_TILE // 32
                ):
                    convert_packed_a_kblock(
                        packed_registers,
                        fp8_fragment_a,
                        partitioned_offsets,
                        k_block,
                        stage_idx,
                    )
                # Order the converted A registers before WGMMA reads them.
                warpgroup.fence()
                for k_block in cutlass.range_constexpr(
                    MXFP4_K_TILE // 32
                ):
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        fp8_fragment_a[(None, None, k_block)],
                        tCrB[
                            (
                                None,
                                None,
                                k_block,
                                stage_idx,
                            )
                        ],
                        accumulators,
                    )
                    tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()
                # The converted A registers are overwritten by the next LDSM;
                # wait until all four K32 WGMMA in this K128 tile retire.
                warpgroup.wait_group(0)
                weight_sf_pipeline.consumer_release(ab_consumer_state)
                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()

        return ab_consumer_state

    @cute.jit
    def _mma_mxfp4_hybrid_fc1(
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
        """Accumulate full-K Humming FC1, then apply one scale per token."""
        if local_warp_idx < self.epilogue_warps_per_warpgroup:
            fp8_fragment_a = tiled_mma.make_fragment_A(
                tiled_mma.partition_shape_A(
                    (self.wgmma_tile_m, MXFP4_K_TILE)
                )
            )
            (
                tiled_copy,
                smem_partition,
                copy_view,
                packed_registers,
            ) = make_packed_a_ldsm_views(
                tiled_mma,
                packed_smem_a,
                fp8_fragment_a,
                tidx % cutlass.Int32(128),
            )

            expanded_offsets = make_expanded_offset_view(
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
            activation_scales = cute.make_rmem_tensor(
                self._activation_scale_rmem_layout().shape,
                Float32,
            )

            accumulators.fill(0.0)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                ab_pipeline.consumer_wait(ab_consumer_state)
                weight_sf_pipeline.consumer_wait(ab_consumer_state)
                stage_idx = ab_consumer_state.index

                # Hybrid FC1 dispatch carries a replicated [token, 4] row.
                # Plane zero is the single whole-hidden dequant scale.
                self._load_activation_scales_blockwise_fragment(
                    smem_activation_sf=smem_activation_sf,
                    activation_scales=activation_scales,
                    stage_idx=stage_idx,
                    scale_plane=cutlass.Int32(0),
                    local_warp_idx=local_warp_idx,
                    tidx=tidx,
                )
                cute.copy(
                    tiled_copy,
                    smem_partition[(None, None, None, stage_idx)],
                    copy_view,
                )
                for k_block in cutlass.range_constexpr(
                    MXFP4_K_TILE // 32
                ):
                    convert_packed_a_kblock(
                        packed_registers,
                        fp8_fragment_a,
                        partitioned_offsets,
                        k_block,
                        stage_idx,
                    )
                # Order the converted A registers before WGMMA reads them.
                warpgroup.fence()
                for k_block in cutlass.range_constexpr(
                    MXFP4_K_TILE // 32
                ):
                    cute.gemm(
                        tiled_mma,
                        accum_temp,
                        fp8_fragment_a[(None, None, k_block)],
                        tCrB[(None, None, k_block, stage_idx)],
                        accum_temp,
                    )
                    tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                weight_sf_pipeline.consumer_release(ab_consumer_state)
                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()

            # The Humming per-expert residual stays in the FC1 epilogue.
            self._promote_accum_temp_blockwise_fc1(
                accumulators=accumulators,
                accum_temp=accum_temp,
                activation_scales=activation_scales,
                weight_scale=Float32(1.0),
            )

        return ab_consumer_state

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
        """Run K256 FC1 as two synchronous four-WGMMA commit groups."""
        if local_warp_idx < self.epilogue_warps_per_warpgroup:
            fp8_fragment_a = tiled_mma.make_fragment_A(
                tiled_mma.partition_shape_A(
                    (self.wgmma_tile_m, 2 * MXFP4_K_TILE)
                )
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
            weight_sf_pipeline.consumer_wait(ab_consumer_state)
            if local_warp_idx == cutlass.Int32(0):
                iket.range_pop()
            stage_idx = ab_consumer_state.index
            cute.copy(
                tiled_copy,
                smem_partition[(None, None, None, stage_idx)],
                copy_view,
            )
            if local_warp_idx == cutlass.Int32(0):
                iket.range_push("mx_f1_issue0")
            for k_block in cutlass.range_constexpr(4):
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_push("mx_f1_cvt0")
                convert_packed_a_kblock(
                    packed_registers,
                    fp8_fragment_a,
                    partitioned_offsets,
                    k_block,
                    stage_idx,
                )
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
            if local_warp_idx == cutlass.Int32(0):
                iket.range_push("mx_f1_wgwait0")
            warpgroup.wait_group(0)
            if local_warp_idx == cutlass.Int32(0):
                iket.range_pop()
            if local_warp_idx == cutlass.Int32(0):
                iket.range_push("mx_f1_issue1")
            for k_block in cutlass.range_constexpr(4, 8):
                if local_warp_idx == cutlass.Int32(0):
                    iket.range_push("mx_f1_cvt1")
                convert_packed_a_kblock(
                    packed_registers,
                    fp8_fragment_a,
                    partitioned_offsets,
                    k_block,
                    stage_idx,
                )
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
                weight_sf_pipeline.consumer_wait(ab_consumer_state)
                stage_idx = ab_consumer_state.index
                cute.copy(
                    tiled_copy,
                    smem_partition[(None, None, None, stage_idx)],
                    copy_view,
                )
                for k_block in cutlass.range_constexpr(4):
                    convert_packed_a_kblock(
                        packed_registers,
                        fp8_fragment_a,
                        partitioned_offsets,
                        k_block,
                        stage_idx,
                    )
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
                for k_block in cutlass.range_constexpr(4, 8):
                    convert_packed_a_kblock(
                        packed_registers,
                        fp8_fragment_a,
                        partitioned_offsets,
                        k_block,
                        stage_idx,
                    )
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
                weight_sf_pipeline.consumer_release(release_state)
                ab_pipeline.consumer_release(release_state)
                release_state.advance()
                ab_consumer_state.advance()

            # Keep the accepted K256 baseline explicit: every outstanding
            # WGMMA group must be retired before the final scale promotion.
            # This is redundant with the per-half waits above, but retaining
            # it avoids carrying an unproven scheduling-only experiment.
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
            weight_sf_pipeline.consumer_release(release_state)
            ab_pipeline.consumer_release(release_state)

        return ab_consumer_state

    @cute.jit
    def _promote_accum_temp_blockwise_streaming(
        self,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        smem_activation_sf: cute.Tensor,
        stage_idx,
        scale_plane,
        tidx,
    ) -> None:
        """Promote one K64 sum while retaining only one token-pair scale."""
        lane_mod = (tidx % 32) % 4
        accum_regs_per_m64 = self.wgmma_tile_n // 2
        math_n = self.wgmma_tile_n
        if cutlass.const_expr(self.mxfp4_optimizations.fc2_tail_n8):
            # Each M128 fragment has math_n registers per lane. Narrow only
            # the source; the destination keeps the physical N64 layout.
            math_n = cute.size(accum_temp)
        token_group_count = math_n // 8
        for token_group in cutlass.range_constexpr(token_group_count):
            token0 = (
                cutlass.Int32(token_group * 8)
                + lane_mod * cutlass.Int32(2)
            )
            token1 = token0 + cutlass.Int32(1)
            token0_scale = Float32(
                smem_activation_sf[token0, scale_plane, stage_idx]
            )
            token1_scale = Float32(
                smem_activation_sf[token1, scale_plane, stage_idx]
            )
            for m_sub in cutlass.range_constexpr(2):
                base = (
                    m_sub * accum_regs_per_m64
                    + token_group * 4
                )
                src_base = m_sub * (math_n // 2) + token_group * 4
                accumulators[base + 0] = (
                    accumulators[base + 0]
                    + accum_temp[src_base + 0] * token0_scale
                )
                accumulators[base + 1] = (
                    accumulators[base + 1]
                    + accum_temp[src_base + 1] * token1_scale
                )
                accumulators[base + 2] = (
                    accumulators[base + 2]
                    + accum_temp[src_base + 2] * token0_scale
                )
                accumulators[base + 3] = (
                    accumulators[base + 3]
                    + accum_temp[src_base + 3] * token1_scale
                )

    @cute.jit
    def _mma_mxfp4_hybrid_fc2_k256_tail_select(
        self,
        valid_tokens,
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
        if cutlass.const_expr(self.mxfp4_optimizations.fc2_tail_n8):
            if valid_tokens <= cutlass.Int32(8):
                narrow_mma = sm90_utils.make_trivial_tiled_mma(
                    cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
                    self.a_major_mode, self.b_major_mode, self.acc_dtype,
                    self.atom_layout_mnk, tiler_mn=(64, 8),
                    a_source=warpgroup.OperandSource.RMEM,
                )
                narrow_temp = cute.make_rmem_tensor(
                    narrow_mma.partition_shape_C((self.wgmma_tile_m, 8)),
                    self.acc_dtype,
                )
                ab_consumer_state = self._mma_mxfp4_hybrid_fc2_k256_half(
                    local_warp_idx, narrow_mma, packed_smem_a, tCrB,
                    accumulators, narrow_temp, ab_pipeline,
                    weight_sf_pipeline, ab_consumer_state,
                    smem_activation_sf, smem_weight_sf, k_tile_cnt,
                    n_half, tidx,
                )
            else:
                ab_consumer_state = self._mma_mxfp4_hybrid_fc2_k256_half(
                    local_warp_idx, tiled_mma, packed_smem_a, tCrB,
                    accumulators, accum_temp, ab_pipeline,
                    weight_sf_pipeline, ab_consumer_state,
                    smem_activation_sf, smem_weight_sf, k_tile_cnt,
                    n_half, tidx,
                )
        else:
            # Entire runtime selector is compiled out when the strategy is
            # off, including all other FP8/MXFP4 geometries.
            ab_consumer_state = self._mma_mxfp4_hybrid_fc2_k256_half(
                local_warp_idx, tiled_mma, packed_smem_a, tCrB,
                accumulators, accum_temp, ab_pipeline,
                weight_sf_pipeline, ab_consumer_state,
                smem_activation_sf, smem_weight_sf, k_tile_cnt,
                n_half, tidx,
            )
        return ab_consumer_state

    @cute.jit
    def _mma_mxfp4_hybrid_fc2_k256_half(
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
        """Run K256 FC2 with one short-lived converted K128 half."""

        if local_warp_idx < self.epilogue_warps_per_warpgroup:
            fp8_fragment_a = tiled_mma.make_fragment_A(
                tiled_mma.partition_shape_A(
                    (self.wgmma_tile_m, MXFP4_K_TILE)
                )
            )
            (
                tiled_copy,
                smem_partition,
                copy_view,
                packed_registers,
            ) = make_packed_a_ldsm_views_k256_half(
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

            accumulators.fill(0.0)
            for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                ab_pipeline.consumer_wait(ab_consumer_state)
                weight_sf_pipeline.consumer_wait(ab_consumer_state)
                stage_idx = ab_consumer_state.index

                for k128_half in cutlass.range_constexpr(2):
                    for packed_k64 in cutlass.range_constexpr(2):
                        cute.copy(
                            tiled_copy,
                            smem_partition[
                                (
                                    None,
                                    None,
                                    k128_half * 2 + packed_k64,
                                    stage_idx,
                                )
                            ],
                            copy_view[(None, None, packed_k64)],
                        )

                    for local_k64_group in cutlass.range_constexpr(2):
                        local_k_block_begin = local_k64_group * 2
                        global_k64_group = k128_half * 2 + local_k64_group
                        global_k_block_begin = global_k64_group * 2
                        scale_plane = cutlass.Int32(global_k64_group)
                        for k_offset in cutlass.range_constexpr(2):
                            convert_packed_a_kblock_from_offset(
                                packed_registers,
                                fp8_fragment_a,
                                partitioned_offsets,
                                local_k_block_begin + k_offset,
                                global_k_block_begin + k_offset,
                                stage_idx,
                            )

                        tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
                        warpgroup.fence()
                        for k_offset in cutlass.range_constexpr(2):
                            cute.gemm(
                                tiled_mma,
                                accum_temp,
                                fp8_fragment_a[
                                    (
                                        None,
                                        None,
                                        local_k_block_begin + k_offset,
                                    )
                                ],
                                tCrB[
                                    (
                                        None,
                                        None,
                                        global_k_block_begin + k_offset,
                                        stage_idx,
                                    )
                                ],
                                accum_temp,
                            )
                            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                        warpgroup.commit_group()
                        warpgroup.wait_group(0)
                        self._promote_accum_temp_blockwise_streaming(
                            accumulators=accumulators,
                            accum_temp=accum_temp,
                            smem_activation_sf=smem_activation_sf,
                            stage_idx=stage_idx,
                            scale_plane=scale_plane,
                            tidx=tidx,
                        )

                weight_sf_pipeline.consumer_release(ab_consumer_state)
                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()

        return ab_consumer_state

    @cute.jit
    def _mma_mxfp4_hybrid_fc2(
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
        """Split every Humming K128 stage into independently scaled K64 sums."""
        if local_warp_idx < self.epilogue_warps_per_warpgroup:
            fp8_fragment_a = tiled_mma.make_fragment_A(
                tiled_mma.partition_shape_A(
                    (self.wgmma_tile_m, MXFP4_K_TILE)
                )
            )
            (
                tiled_copy,
                smem_partition,
                copy_view,
                packed_registers,
            ) = make_packed_a_ldsm_views(
                tiled_mma,
                packed_smem_a,
                fp8_fragment_a,
                tidx % cutlass.Int32(128),
            )

            expanded_offsets = make_expanded_offset_view(
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
            num_k_blocks = MXFP4_K_TILE // 32
            half_k_blocks = num_k_blocks // 2

            accumulators.fill(0.0)
            for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                ab_pipeline.consumer_wait(ab_consumer_state)
                weight_sf_pipeline.consumer_wait(ab_consumer_state)
                stage_idx = ab_consumer_state.index
                cute.copy(
                    tiled_copy,
                    smem_partition[(None, None, None, stage_idx)],
                    copy_view,
                )
                for k_block in cutlass.range_constexpr(half_k_blocks):
                    convert_packed_a_kblock(
                        packed_registers,
                        fp8_fragment_a,
                        partitioned_offsets,
                        k_block,
                        stage_idx,
                    )

                scale_plane_base = (
                    k_tile % cutlass.Int32(2)
                ) * cutlass.Int32(2)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
                warpgroup.fence()
                for k_block in cutlass.range_constexpr(half_k_blocks):
                    cute.gemm(
                        tiled_mma,
                        accum_temp,
                        fp8_fragment_a[(None, None, k_block)],
                        tCrB[(None, None, k_block, stage_idx)],
                        accum_temp,
                    )
                    tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                self._promote_accum_temp_blockwise_streaming(
                    accumulators=accumulators,
                    accum_temp=accum_temp,
                    smem_activation_sf=smem_activation_sf,
                    stage_idx=stage_idx,
                    scale_plane=scale_plane_base,
                    tidx=tidx,
                )

                for k_block in cutlass.range_constexpr(
                    half_k_blocks, num_k_blocks
                ):
                    convert_packed_a_kblock(
                        packed_registers,
                        fp8_fragment_a,
                        partitioned_offsets,
                        k_block,
                        stage_idx,
                    )
                tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
                warpgroup.fence()
                for k_block in cutlass.range_constexpr(
                    half_k_blocks, num_k_blocks
                ):
                    cute.gemm(
                        tiled_mma,
                        accum_temp,
                        fp8_fragment_a[(None, None, k_block)],
                        tCrB[(None, None, k_block, stage_idx)],
                        accum_temp,
                    )
                    tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                self._promote_accum_temp_blockwise_streaming(
                    accumulators=accumulators,
                    accum_temp=accum_temp,
                    smem_activation_sf=smem_activation_sf,
                    stage_idx=stage_idx,
                    scale_plane=(
                        scale_plane_base + cutlass.Int32(1)
                    ),
                    tidx=tidx,
                )
                weight_sf_pipeline.consumer_release(ab_consumer_state)
                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()

        return ab_consumer_state

    @cute.jit
    def run_wgmma_task_tile(
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
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        k_tile_cnt_fc1,
        k_tile_cnt_fc2,
        _iket_active,
        tidx,
    ):
        if local_warp_idx < self.epilogue_warps_per_warpgroup:
            is_phase_linear1 = (
                work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            )
            k_tile_cnt = cutlass.Int32(0)
            if is_phase_linear1:
                k_tile_cnt = k_tile_cnt_fc1
                if _iket_active:
                    iket.range_push(self._iket_fc1_mma_mainloop_range)
            else:
                k_tile_cnt = k_tile_cnt_fc2
                if _iket_active:
                    iket.range_push(self._iket_fc2_mma_mainloop_range)

            ab_consumer_state.reset_count()
            if cutlass.const_expr(self.fp8_scale_mode == "per_tensor"):
                ab_consumer_state = self._mma_mxfp4_per_tensor_1xacc(
                    local_warp_idx=local_warp_idx,
                    tiled_mma=tiled_mma,
                    packed_smem_a=tCrA,
                    tCrB=tCrB,
                    accumulators=accumulators,
                    ab_pipeline=ab_pipeline,
                    weight_sf_pipeline=weight_sf_pipeline,
                    ab_consumer_state=ab_consumer_state,
                    smem_weight_sf=smem_weight_sf,
                    k_tile_cnt=k_tile_cnt,
                    n_half=n_half,
                    tidx=tidx,
                )
            elif is_phase_linear1:
                if cutlass.const_expr(
                    self.mma_tiler[2] == 2 * MXFP4_K_TILE
                ):
                    ab_consumer_state = self._mma_mxfp4_hybrid_fc1_k256(
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        packed_smem_a=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        k_tile_cnt=k_tile_cnt,
                        n_half=n_half,
                        tidx=tidx,
                    )
                else:
                    ab_consumer_state = self._mma_mxfp4_hybrid_fc1(
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        packed_smem_a=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        k_tile_cnt=k_tile_cnt,
                        n_half=n_half,
                        tidx=tidx,
                    )
            else:
                if cutlass.const_expr(
                    self.mma_tiler[2] == 2 * MXFP4_K_TILE
                ):
                    ab_consumer_state = (
                        self._mma_mxfp4_hybrid_fc2_k256_tail_select(
                            valid_tokens=work_tile_info.valid_tokens_in_cta_tile,
                            local_warp_idx=local_warp_idx,
                            tiled_mma=tiled_mma,
                            packed_smem_a=tCrA,
                            tCrB=tCrB,
                            accumulators=accumulators,
                            accum_temp=accum_temp,
                            ab_pipeline=ab_pipeline,
                            weight_sf_pipeline=weight_sf_pipeline,
                            ab_consumer_state=ab_consumer_state,
                            smem_activation_sf=smem_activation_sf,
                            smem_weight_sf=smem_weight_sf,
                            k_tile_cnt=k_tile_cnt,
                            n_half=n_half,
                            tidx=tidx,
                        )
                    )
                else:
                    ab_consumer_state = self._mma_mxfp4_hybrid_fc2(
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        packed_smem_a=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        k_tile_cnt=k_tile_cnt,
                        n_half=n_half,
                        tidx=tidx,
                    )
            if _iket_active:
                iket.range_pop()

        return ab_consumer_state





__all__ = ["Sm90SwapABSwigluMxfp4Fp8Fc12Kernel"]
