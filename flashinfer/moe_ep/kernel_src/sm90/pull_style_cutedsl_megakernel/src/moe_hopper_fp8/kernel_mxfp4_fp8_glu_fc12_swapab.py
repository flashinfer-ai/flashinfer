# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Hopper Humming MXFP4-weight x FP8-activation fused FC1+FC2 kernel."""

import os

from typing import Type

import cutlass
import cutlass.cute as cute
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover
    from src.iket_compat import iket
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.typing import Float32
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

Mxfp4Fc2ScaleLoadOverlap = int(
    os.environ.get("MEGA_MXFP4_FC2_SCALE_LOAD_OVERLAP", "0")
)
if Mxfp4Fc2ScaleLoadOverlap not in (0, 1):
    raise ValueError(
        "MEGA_MXFP4_FC2_SCALE_LOAD_OVERLAP must be 0 or 1, got "
        f"{Mxfp4Fc2ScaleLoadOverlap}."
    )

Mxfp4Fc2StreamScalePromotion = int(
    os.environ.get("MEGA_MXFP4_FC2_STREAM_SCALE_PROMOTION", "1")
)
if Mxfp4Fc2StreamScalePromotion not in (0, 1):
    raise ValueError(
        "MEGA_MXFP4_FC2_STREAM_SCALE_PROMOTION must be 0 or 1, got "
        f"{Mxfp4Fc2StreamScalePromotion}."
    )

Mxfp4Fc1K64Pipeline = int(
    os.environ.get("MEGA_MXFP4_FC1_K64_PIPELINE", "0")
)
if Mxfp4Fc1K64Pipeline not in (0, 1):
    raise ValueError(
        "MEGA_MXFP4_FC1_K64_PIPELINE must be 0 or 1, got "
        f"{Mxfp4Fc1K64Pipeline}."
    )

Mxfp4K256Fc1TwoGroupPipeline = int(
    os.environ.get("MEGA_MXFP4_K256_FC1_TWO_GROUP_PIPELINE", "0")
)
if Mxfp4K256Fc1TwoGroupPipeline not in (0, 1):
    raise ValueError(
        "MEGA_MXFP4_K256_FC1_TWO_GROUP_PIPELINE must be 0 or 1, got "
        f"{Mxfp4K256Fc1TwoGroupPipeline}."
    )

_Mxfp4Fc1SplitFinalScaleEnv = os.environ.get(
    "MEGA_MXFP4_FC1_SPLIT_FINAL_SCALE"
)
Mxfp4Fc1SplitFinalScale = (
    None
    if _Mxfp4Fc1SplitFinalScaleEnv is None
    else int(_Mxfp4Fc1SplitFinalScaleEnv)
)
if Mxfp4Fc1SplitFinalScale not in (None, 0, 1):
    raise ValueError(
        "MEGA_MXFP4_FC1_SPLIT_FINAL_SCALE must be 0 or 1 when set, got "
        f"{Mxfp4Fc1SplitFinalScale}."
    )

_Mxfp4K256Fc2HalfFragmentEnv = os.environ.get(
    "MEGA_MXFP4_K256_FC2_HALF_FRAGMENT"
)
Mxfp4K256Fc2HalfFragment = (
    None
    if _Mxfp4K256Fc2HalfFragmentEnv is None
    else int(_Mxfp4K256Fc2HalfFragmentEnv)
)
if Mxfp4K256Fc2HalfFragment not in (None, 0, 1):
    raise ValueError(
        "MEGA_MXFP4_K256_FC2_HALF_FRAGMENT must be 0 or 1 when set, got "
        f"{Mxfp4K256Fc2HalfFragment}."
    )

def _resolve_mxfp4_fc2_scale_load_overlap(
    fp8_scale_mode: str,
    *,
    configured: int = Mxfp4Fc2ScaleLoadOverlap,
) -> bool:
    """Keep scale-load overlap isolated from the MXFP4 hybrid FC2 path."""
    return fp8_scale_mode == "mxfp4_hybrid" and configured == 1


def _resolve_mxfp4_fc2_stream_scale_promotion(
    fp8_scale_mode: str,
    *,
    configured: int = Mxfp4Fc2StreamScalePromotion,
) -> bool:
    """Stream FC2 scales in token pairs instead of retaining a full fragment."""
    return fp8_scale_mode == "mxfp4_hybrid" and configured == 1


def _resolve_mxfp4_fc1_k64_pipeline(
    fp8_scale_mode: str,
    *,
    configured: int = Mxfp4Fc1K64Pipeline,
) -> bool:
    """Pipeline FC1's two K64 halves only for MXFP4 hybrid compute."""
    return fp8_scale_mode == "mxfp4_hybrid" and configured == 1


def _resolve_mxfp4_fc1_split_final_scale(
    fp8_scale_mode: str,
    *,
    tile_k: int = MXFP4_K_TILE,
    configured: int | None = Mxfp4Fc1SplitFinalScale,
) -> bool:
    """Stream the final FC1 scale by default only for the K256 tactic."""
    if fp8_scale_mode != "mxfp4_hybrid":
        return False
    if configured is not None:
        return configured == 1
    return tile_k == 2 * MXFP4_K_TILE


def _resolve_mxfp4_k256_fc2_half_fragment(
    fp8_scale_mode: str,
    *,
    tile_k: int = MXFP4_K_TILE,
    configured: int | None = Mxfp4K256Fc2HalfFragment,
) -> bool:
    """Use the shorter FC2 register fragment by default only for hybrid K256."""
    if fp8_scale_mode != "mxfp4_hybrid" or tile_k != 2 * MXFP4_K_TILE:
        return False
    if configured is not None:
        return configured == 1
    return True


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
        # MegaMoE sets ``self.split_role`` before cooperative ``super()``;
        # standalone callers may provide the same codegen-time role directly.
        # The plain swap-AB base does not consume this keyword.
        split_role = kwargs.get(
            "split_role", getattr(self, "split_role", "fused")
        )
        kwargs.pop("split_role", None)
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
            logical_ks = {
                "k1": (("hidden", hidden),),
                "k2": (
                    ("intermediate_downproj", intermediate_gateup // 2),
                ),
            }.get(
                split_role,
                (
                    ("hidden", hidden),
                    ("intermediate_downproj", intermediate_gateup // 2),
                ),
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
        self.is_mxfp4_fp8 = True

        self._fc2_scale_load_overlap = (
            _resolve_mxfp4_fc2_scale_load_overlap(fp8_scale_mode)
        )
        self._fc2_stream_scale_promotion = (
            _resolve_mxfp4_fc2_stream_scale_promotion(fp8_scale_mode)
        )
        self._fc1_k64_pipeline = _resolve_mxfp4_fc1_k64_pipeline(
            fp8_scale_mode
        )
        self._k256_fc1_two_group_pipeline = (
            fp8_scale_mode == "mxfp4_hybrid"
            and Mxfp4K256Fc1TwoGroupPipeline == 1
        )
        self._fc1_split_final_scale = (
            _resolve_mxfp4_fc1_split_final_scale(
                fp8_scale_mode,
                tile_k=mma_tiler_mnk[2],
            )
        )
        self._k256_fc2_half_fragment = (
            _resolve_mxfp4_k256_fc2_half_fragment(
                fp8_scale_mode,
                tile_k=mma_tiler_mnk[2],
            )
        )

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
            cute.copy(copy_atom, gmem_vec, smem_vec)

        scale_handle.commit()

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
        """Stage two adjacent folded K128 offset blocks for one K256 tile."""
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
            warpgroup.fence()
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
    def _mma_mxfp4_hybrid_fc1_k64_pipeline(
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
        """Pipeline K128 groups with two FP8 fragments and shared packed RF."""
        if local_warp_idx < self.epilogue_warps_per_warpgroup:
            fp8_fragment_a0 = tiled_mma.make_fragment_A(
                tiled_mma.partition_shape_A(
                    (self.wgmma_tile_m, MXFP4_K_TILE)
                )
            )
            (
                tiled_copy0,
                smem_partition0,
                copy_view0,
                packed_registers0,
            ) = make_packed_a_ldsm_views(
                tiled_mma,
                packed_smem_a,
                fp8_fragment_a0,
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
            num_k_blocks = MXFP4_K_TILE // 32
            release_state = ab_consumer_state.clone()

            accumulators.fill(0.0)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()

            # Prologue: split K128 into two commit groups.  wait_group(1)
            # leaves the newest group in flight, so conversion of slots 2/3
            # overlaps WGMMA on slots 0/1 without a second FP8 fragment.
            # FC1's activation scale is invariant over K and is retained from
            # this first stage.
            ab_pipeline.consumer_wait(ab_consumer_state)
            weight_sf_pipeline.consumer_wait(ab_consumer_state)
            stage_idx = ab_consumer_state.index
            self._load_activation_scales_blockwise_fragment(
                smem_activation_sf=smem_activation_sf,
                activation_scales=activation_scales,
                stage_idx=stage_idx,
                scale_plane=cutlass.Int32(0),
                local_warp_idx=local_warp_idx,
                tidx=tidx,
            )
            cute.copy(
                tiled_copy0,
                smem_partition0[(None, None, None, stage_idx)],
                copy_view0,
            )
            for k_block in cutlass.range_constexpr(2):
                convert_packed_a_kblock(
                    packed_registers0,
                    fp8_fragment_a0,
                    partitioned_offsets,
                    k_block,
                    stage_idx,
                )
            for k_block in cutlass.range_constexpr(2):
                cute.gemm(
                    tiled_mma,
                    accum_temp,
                    fp8_fragment_a0[(None, None, k_block)],
                    tCrB[(None, None, k_block, stage_idx)],
                    accum_temp,
                )
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(1)
            for k_block in cutlass.range_constexpr(2, num_k_blocks):
                convert_packed_a_kblock(
                    packed_registers0,
                    fp8_fragment_a0,
                    partitioned_offsets,
                    k_block,
                    stage_idx,
                )
            for k_block in cutlass.range_constexpr(2, num_k_blocks):
                cute.gemm(
                    tiled_mma,
                    accum_temp,
                    fp8_fragment_a0[(None, None, k_block)],
                    tCrB[(None, None, k_block, stage_idx)],
                    accum_temp,
                )
            warpgroup.commit_group()
            warpgroup.wait_group(1)
            ab_consumer_state.advance()

            # Each following tile first reuses slots 0/1.  Its first rolling
            # wait retires the previous tile's slots 2/3 before those slots
            # are converted, preserving the RS operand lifetime contract.
            for k_tile in cutlass.range(1, k_tile_cnt, 1, unroll=1):
                ab_pipeline.consumer_wait(ab_consumer_state)
                weight_sf_pipeline.consumer_wait(ab_consumer_state)
                stage_idx = ab_consumer_state.index
                cute.copy(
                    tiled_copy0,
                    smem_partition0[(None, None, None, stage_idx)],
                    copy_view0,
                )
                for k_block in cutlass.range_constexpr(2):
                    convert_packed_a_kblock(
                        packed_registers0,
                        fp8_fragment_a0,
                        partitioned_offsets,
                        k_block,
                        stage_idx,
                    )
                for k_block in cutlass.range_constexpr(2):
                    cute.gemm(
                        tiled_mma,
                        accum_temp,
                        fp8_fragment_a0[(None, None, k_block)],
                        tCrB[(None, None, k_block, stage_idx)],
                        accum_temp,
                    )
                warpgroup.commit_group()
                warpgroup.wait_group(1)
                for k_block in cutlass.range_constexpr(2, num_k_blocks):
                    convert_packed_a_kblock(
                        packed_registers0,
                        fp8_fragment_a0,
                        partitioned_offsets,
                        k_block,
                        stage_idx,
                    )
                for k_block in cutlass.range_constexpr(2, num_k_blocks):
                    cute.gemm(
                        tiled_mma,
                        accum_temp,
                        fp8_fragment_a0[(None, None, k_block)],
                        tCrB[(None, None, k_block, stage_idx)],
                        accum_temp,
                    )
                warpgroup.commit_group()
                warpgroup.wait_group(1)
                weight_sf_pipeline.consumer_release(release_state)
                ab_pipeline.consumer_release(release_state)
                release_state.advance()
                ab_consumer_state.advance()

            warpgroup.wait_group(0)
            weight_sf_pipeline.consumer_release(release_state)
            ab_pipeline.consumer_release(release_state)
            self._promote_accum_temp_blockwise_fc1(
                accumulators=accumulators,
                accum_temp=accum_temp,
                activation_scales=activation_scales,
                weight_scale=Float32(1.0),
            )

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
            warpgroup.fence()
            main_k_tile_cnt = k_tile_cnt
            if cutlass.const_expr(self._fc1_split_final_scale):
                main_k_tile_cnt = k_tile_cnt - cutlass.Int32(1)
            for k_tile in cutlass.range(0, main_k_tile_cnt, 1, unroll=1):
                ab_pipeline.consumer_wait(ab_consumer_state)
                weight_sf_pipeline.consumer_wait(ab_consumer_state)
                stage_idx = ab_consumer_state.index

                # Hybrid FC1 dispatch carries a replicated [token, 4] row.
                # Plane zero is the single whole-hidden dequant scale.
                if cutlass.const_expr(not self._fc1_split_final_scale):
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

            if cutlass.const_expr(self._fc1_split_final_scale):
                # Keep the final AB stage owned until its invariant per-token
                # scale has been consumed in short-lived token-pair fragments.
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
                self._promote_accum_temp_blockwise_streaming(
                    accumulators=accumulators,
                    accum_temp=accum_temp,
                    smem_activation_sf=smem_activation_sf,
                    stage_idx=stage_idx,
                    scale_plane=cutlass.Int32(0),
                    tidx=tidx,
                )
                weight_sf_pipeline.consumer_release(ab_consumer_state)
                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()
            else:
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
            if cutlass.const_expr(not self._fc1_split_final_scale):
                activation_scales = cute.make_rmem_tensor(
                    self._activation_scale_rmem_layout().shape,
                    Float32,
                )
            release_state = ab_consumer_state.clone()

            accumulators.fill(0.0)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()

            if local_warp_idx == cutlass.Int32(0):
                iket.range_push("mx_f1_wait0")
            ab_pipeline.consumer_wait(ab_consumer_state)
            weight_sf_pipeline.consumer_wait(ab_consumer_state)
            if local_warp_idx == cutlass.Int32(0):
                iket.range_pop()
            stage_idx = ab_consumer_state.index
            if cutlass.const_expr(not self._fc1_split_final_scale):
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
            if cutlass.const_expr(not self._k256_fc1_two_group_pipeline):
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
                    cute.gemm(
                        tiled_mma,
                        accum_temp,
                        fp8_fragment_a[(None, None, k_block)],
                        tCrB[(None, None, k_block, stage_idx)],
                        accum_temp,
                    )
                warpgroup.commit_group()
                if cutlass.const_expr(not self._k256_fc1_two_group_pipeline):
                    warpgroup.wait_group(0)
                for k_block in cutlass.range_constexpr(4, 8):
                    convert_packed_a_kblock(
                        packed_registers,
                        fp8_fragment_a,
                        partitioned_offsets,
                        k_block,
                        stage_idx,
                    )
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
            if cutlass.const_expr(self._fc1_split_final_scale):
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
            else:
                weight_sf_pipeline.consumer_release(release_state)
                ab_pipeline.consumer_release(release_state)
                self._promote_accum_temp_blockwise_fc1(
                    accumulators=accumulators,
                    accum_temp=accum_temp,
                    activation_scales=activation_scales,
                    weight_scale=Float32(1.0),
                )

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
        token_group_count = self.wgmma_tile_n // 8
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
                accumulators[base + 0] = (
                    accumulators[base + 0]
                    + accum_temp[base + 0] * token0_scale
                )
                accumulators[base + 1] = (
                    accumulators[base + 1]
                    + accum_temp[base + 1] * token1_scale
                )
                accumulators[base + 2] = (
                    accumulators[base + 2]
                    + accum_temp[base + 2] * token0_scale
                )
                accumulators[base + 3] = (
                    accumulators[base + 3]
                    + accum_temp[base + 3] * token1_scale
                )

    @cute.jit
    def _mma_mxfp4_hybrid_fc2_k256(
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
        """Accumulate a K256 stage as four independently scaled K64 sums."""
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
            activation_scales = cute.make_rmem_tensor(
                self._activation_scale_rmem_layout().shape,
                Float32,
            )

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

                # One K256 activation-SF tile carries four consecutive K64
                # scale planes.  Keep each partial sum independent until its
                # matching plane has been applied.
                for k64_group in cutlass.range_constexpr(4):
                    k_block_begin = k64_group * 2
                    k_block_end = k_block_begin + 2
                    scale_plane = cutlass.Int32(k64_group)
                    for k_block in cutlass.range_constexpr(
                        k_block_begin, k_block_end
                    ):
                        convert_packed_a_kblock(
                            packed_registers,
                            fp8_fragment_a,
                            partitioned_offsets,
                            k_block,
                            stage_idx,
                        )

                    tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
                    if cutlass.const_expr(
                        not self._fc2_stream_scale_promotion
                        and not self._fc2_scale_load_overlap
                    ):
                        self._load_activation_scales_blockwise_fragment(
                            smem_activation_sf=smem_activation_sf,
                            activation_scales=activation_scales,
                            stage_idx=stage_idx,
                            scale_plane=scale_plane,
                            local_warp_idx=local_warp_idx,
                            tidx=tidx,
                        )
                    warpgroup.fence()
                    for k_block in cutlass.range_constexpr(
                        k_block_begin, k_block_end
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
                    if cutlass.const_expr(
                        not self._fc2_stream_scale_promotion
                        and self._fc2_scale_load_overlap
                    ):
                        self._load_activation_scales_blockwise_fragment(
                            smem_activation_sf=smem_activation_sf,
                            activation_scales=activation_scales,
                            stage_idx=stage_idx,
                            scale_plane=scale_plane,
                            local_warp_idx=local_warp_idx,
                            tidx=tidx,
                        )
                    warpgroup.wait_group(0)
                    if cutlass.const_expr(self._fc2_stream_scale_promotion):
                        self._promote_accum_temp_blockwise_streaming(
                            accumulators=accumulators,
                            accum_temp=accum_temp,
                            smem_activation_sf=smem_activation_sf,
                            stage_idx=stage_idx,
                            scale_plane=scale_plane,
                            tidx=tidx,
                        )
                    else:
                        self._promote_accum_temp_blockwise_fc2(
                            accumulators=accumulators,
                            accum_temp=accum_temp,
                            activation_scales=activation_scales,
                            weight_scale=Float32(1.0),
                        )

                weight_sf_pipeline.consumer_release(ab_consumer_state)
                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()

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
            activation_scales = cute.make_rmem_tensor(
                self._activation_scale_rmem_layout().shape,
                Float32,
            )

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
                        if cutlass.const_expr(
                            not self._fc2_stream_scale_promotion
                            and not self._fc2_scale_load_overlap
                        ):
                            self._load_activation_scales_blockwise_fragment(
                                smem_activation_sf=smem_activation_sf,
                                activation_scales=activation_scales,
                                stage_idx=stage_idx,
                                scale_plane=scale_plane,
                                local_warp_idx=local_warp_idx,
                                tidx=tidx,
                            )
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
                        if cutlass.const_expr(
                            not self._fc2_stream_scale_promotion
                            and self._fc2_scale_load_overlap
                        ):
                            self._load_activation_scales_blockwise_fragment(
                                smem_activation_sf=smem_activation_sf,
                                activation_scales=activation_scales,
                                stage_idx=stage_idx,
                                scale_plane=scale_plane,
                                local_warp_idx=local_warp_idx,
                                tidx=tidx,
                            )
                        warpgroup.wait_group(0)
                        if cutlass.const_expr(
                            self._fc2_stream_scale_promotion
                        ):
                            self._promote_accum_temp_blockwise_streaming(
                                accumulators=accumulators,
                                accum_temp=accum_temp,
                                smem_activation_sf=smem_activation_sf,
                                stage_idx=stage_idx,
                                scale_plane=scale_plane,
                                tidx=tidx,
                            )
                        else:
                            self._promote_accum_temp_blockwise_fc2(
                                accumulators=accumulators,
                                accum_temp=accum_temp,
                                activation_scales=activation_scales,
                                weight_scale=Float32(1.0),
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
            activation_scales = cute.make_rmem_tensor(
                self._activation_scale_rmem_layout().shape,
                Float32,
            )

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
                if cutlass.const_expr(
                    not self._fc2_stream_scale_promotion
                    and not self._fc2_scale_load_overlap
                ):
                    self._load_activation_scales_blockwise_fragment(
                        smem_activation_sf=smem_activation_sf,
                        activation_scales=activation_scales,
                        stage_idx=stage_idx,
                        scale_plane=scale_plane_base,
                        local_warp_idx=local_warp_idx,
                        tidx=tidx,
                    )
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
                if cutlass.const_expr(
                    not self._fc2_stream_scale_promotion
                    and self._fc2_scale_load_overlap
                ):
                    self._load_activation_scales_blockwise_fragment(
                        smem_activation_sf=smem_activation_sf,
                        activation_scales=activation_scales,
                        stage_idx=stage_idx,
                        scale_plane=scale_plane_base,
                        local_warp_idx=local_warp_idx,
                        tidx=tidx,
                    )
                warpgroup.wait_group(0)
                if cutlass.const_expr(self._fc2_stream_scale_promotion):
                    self._promote_accum_temp_blockwise_streaming(
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        smem_activation_sf=smem_activation_sf,
                        stage_idx=stage_idx,
                        scale_plane=scale_plane_base,
                        tidx=tidx,
                    )
                else:
                    self._promote_accum_temp_blockwise_fc2(
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        activation_scales=activation_scales,
                        weight_scale=Float32(1.0),
                    )

                if cutlass.const_expr(
                    not self._fc2_stream_scale_promotion
                    and not self._fc2_scale_load_overlap
                ):
                    self._load_activation_scales_blockwise_fragment(
                        smem_activation_sf=smem_activation_sf,
                        activation_scales=activation_scales,
                        stage_idx=stage_idx,
                        scale_plane=scale_plane_base + cutlass.Int32(1),
                        local_warp_idx=local_warp_idx,
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
                if cutlass.const_expr(
                    not self._fc2_stream_scale_promotion
                    and self._fc2_scale_load_overlap
                ):
                    self._load_activation_scales_blockwise_fragment(
                        smem_activation_sf=smem_activation_sf,
                        activation_scales=activation_scales,
                        stage_idx=stage_idx,
                        scale_plane=scale_plane_base + cutlass.Int32(1),
                        local_warp_idx=local_warp_idx,
                        tidx=tidx,
                    )
                warpgroup.wait_group(0)
                if cutlass.const_expr(self._fc2_stream_scale_promotion):
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
                else:
                    self._promote_accum_temp_blockwise_fc2(
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        activation_scales=activation_scales,
                        weight_scale=Float32(1.0),
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
                elif cutlass.const_expr(self._fc1_k64_pipeline):
                    ab_consumer_state = (
                        self._mma_mxfp4_hybrid_fc1_k64_pipeline(
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
                    if cutlass.const_expr(self._k256_fc2_half_fragment):
                        ab_consumer_state = (
                            self._mma_mxfp4_hybrid_fc2_k256_half(
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
                        ab_consumer_state = self._mma_mxfp4_hybrid_fc2_k256(
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

    @cute.jit
    def _run_wgmma_mxfp4_hybrid_phase_task_tile(
        self,
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
        phase_mode: cutlass.Constexpr[str],
    ):
        """Run one compile-time-selected MXFP4 hybrid WGMMA phase."""
        if cutlass.const_expr(phase_mode not in ("fc1", "fc2")):
            raise ValueError(
                "phase_mode must be the compile-time string 'fc1' or 'fc2', "
                f"got {phase_mode!r}."
            )
        if cutlass.const_expr(self.fp8_scale_mode != "mxfp4_hybrid"):
            raise ValueError(
                "Phase-only MXFP4 WGMMA callbacks require "
                "fp8_scale_mode='mxfp4_hybrid'."
            )

        if local_warp_idx < self.epilogue_warps_per_warpgroup:
            ab_consumer_state.reset_count()
            if cutlass.const_expr(phase_mode == "fc1"):
                if _iket_active:
                    iket.range_push(self._iket_fc1_mma_mainloop_range)
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
                        k_tile_cnt=k_tile_cnt_fc1,
                        n_half=n_half,
                        tidx=tidx,
                    )
                elif cutlass.const_expr(self._fc1_k64_pipeline):
                    ab_consumer_state = (
                        self._mma_mxfp4_hybrid_fc1_k64_pipeline(
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
                            k_tile_cnt=k_tile_cnt_fc1,
                            n_half=n_half,
                            tidx=tidx,
                        )
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
                        k_tile_cnt=k_tile_cnt_fc1,
                        n_half=n_half,
                        tidx=tidx,
                    )
            else:
                if _iket_active:
                    iket.range_push(self._iket_fc2_mma_mainloop_range)
                if cutlass.const_expr(
                    self.mma_tiler[2] == 2 * MXFP4_K_TILE
                ):
                    if cutlass.const_expr(self._k256_fc2_half_fragment):
                        ab_consumer_state = (
                            self._mma_mxfp4_hybrid_fc2_k256_half(
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
                                k_tile_cnt=k_tile_cnt_fc2,
                                n_half=n_half,
                                tidx=tidx,
                            )
                        )
                    else:
                        ab_consumer_state = self._mma_mxfp4_hybrid_fc2_k256(
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
                            k_tile_cnt=k_tile_cnt_fc2,
                            n_half=n_half,
                            tidx=tidx,
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
                        k_tile_cnt=k_tile_cnt_fc2,
                        n_half=n_half,
                        tidx=tidx,
                    )
            if _iket_active:
                iket.range_pop()

        return ab_consumer_state

    @cute.jit
    def run_wgmma_fc1_task_tile(
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
        """Run the MXFP4 hybrid FC1 mainloop without a runtime phase read."""
        return self._run_wgmma_mxfp4_hybrid_phase_task_tile(
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
            phase_mode="fc1",
        )

    @cute.jit
    def run_wgmma_fc2_task_tile(
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
        """Run the MXFP4 hybrid FC2 mainloop without a runtime phase read."""
        return self._run_wgmma_mxfp4_hybrid_phase_task_tile(
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
            phase_mode="fc2",
        )


__all__ = ["Sm90SwapABSwigluMxfp4Fp8Fc12Kernel"]
