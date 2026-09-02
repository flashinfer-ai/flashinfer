# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Autonomous SM90 WGMMA epilogue for the fused fc1+fc2 MegaMoE kernel."""

from typing import Optional, Tuple, Type, Union
import os

import cutlass
import cutlass.cute as cute
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from src.iket_compat import iket
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync
import cutlass.utils.hopper_helpers as sm90_utils

from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op
from cutlass._mlir.dialects import llvm

from src.flag_batch import GpuReleaseFlagBatchTracker
from src.ptx_helpers import red_add_relaxed_sys_v2_bf16x2
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase
from common.megamoe_constants import (
    Fp8E4M3RcpLimit,
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
)

Fc1EpilogueStoreTileN = 64
SwapABTileMChoices = (128, 256)
SwapABTokenTileNChoices = (16, 32, 64, 128)
SwapABBlockwiseFc1GroupChunkChoices = (1, 2, 4, 8)
SwapABBlockwiseFc1GroupChunksExplicit = (
    "MEGA_SWAPAB_FC1_GROUP_CHUNKS" in os.environ
)
SwapABBlockwiseFc1GroupChunks = int(
    os.environ.get("MEGA_SWAPAB_FC1_GROUP_CHUNKS", "1")
)
SwapABMxfp4ReuseFc1AccumScratch = int(
    os.environ.get("MEGA_MXFP4_REUSE_FC1_ACCUM_SCRATCH", "1")
)
SwapABMxfp4SplitFc1StoreOverlapChoices = ("none", "pr357", "pr360")
SwapABMxfp4SplitFc1StoreOverlap = os.environ.get(
    "MEGA_MXFP4_SPLIT_FC1_STORE_OVERLAP", "none"
).strip().lower()


@dsl_user_op
def _quantize_sts_fc1_pair_mxfp4_split_sw64(
    smem_stage_base: Int32,
    token0: Int32,
    output_col0: Int32,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Convert and store one split-K1 FP8 pair with late SW64 addresses."""
    llvm.inline_asm(
        None,
        [
            smem_stage_base.ir_value(loc=loc, ip=ip),
            token0.ir_value(loc=loc, ip=ip),
            output_col0.ir_value(loc=loc, ip=ip),
            v0.ir_value(loc=loc, ip=ip),
            v1.ir_value(loc=loc, ip=ip),
            v2.ir_value(loc=loc, ip=ip),
            v3.ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .b16 %q01, %q23;\n\t"
            ".reg .b32 %packed, %a0, %a1, %a2, %a3, %swz, %b1, %b2, %b3;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 %q01, $4, $3;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 %q23, $6, $5;\n\t"
            "mov.b32 %packed, {%q01, %q23};\n\t"
            "mad.lo.u32 %a0, $1, 64, $2;\n\t"
            "add.u32 %a0, %a0, $0;\n\t"
            "add.u32 %a1, %a0, 64;\n\t"
            "add.u32 %a2, %a0, 32;\n\t"
            "add.u32 %a3, %a0, 96;\n\t"
            "shr.u32 %swz, %a0, 3; and.b32 %swz, %swz, 0x30; xor.b32 %a0, %a0, %swz;\n\t"
            "shr.u32 %swz, %a1, 3; and.b32 %swz, %swz, 0x30; xor.b32 %a1, %a1, %swz;\n\t"
            "shr.u32 %swz, %a2, 3; and.b32 %swz, %swz, 0x30; xor.b32 %a2, %a2, %swz;\n\t"
            "shr.u32 %swz, %a3, 3; and.b32 %swz, %swz, 0x30; xor.b32 %a3, %a3, %swz;\n\t"
            "shr.u32 %b1, %packed, 8;\n\t"
            "shr.u32 %b2, %packed, 16;\n\t"
            "shr.u32 %b3, %packed, 24;\n\t"
            "st.shared.b8 [%a0], %packed;\n\t"
            "st.shared.b8 [%a1], %b1;\n\t"
            "st.shared.b8 [%a2], %b2;\n\t"
            "st.shared.b8 [%a3], %b3;\n\t"
            "}"
        ),
        "r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

if SwapABBlockwiseFc1GroupChunks not in SwapABBlockwiseFc1GroupChunkChoices:
    raise ValueError(
        "MEGA_SWAPAB_FC1_GROUP_CHUNKS must be one of "
        f"{SwapABBlockwiseFc1GroupChunkChoices}, got "
        f"{SwapABBlockwiseFc1GroupChunks}."
    )
if SwapABMxfp4ReuseFc1AccumScratch not in (0, 1):
    raise ValueError(
        "MEGA_MXFP4_REUSE_FC1_ACCUM_SCRATCH must be 0 or 1, got "
        f"{SwapABMxfp4ReuseFc1AccumScratch}."
    )
if (
    SwapABMxfp4SplitFc1StoreOverlap
    not in SwapABMxfp4SplitFc1StoreOverlapChoices
):
    raise ValueError(
        "MEGA_MXFP4_SPLIT_FC1_STORE_OVERLAP must be one of "
        f"{SwapABMxfp4SplitFc1StoreOverlapChoices}, got "
        f"{SwapABMxfp4SplitFc1StoreOverlap!r}."
    )


def _resolve_swapab_fc1_group_chunks(
    fp8_scale_mode: str,
    *,
    configured_chunks: int = SwapABBlockwiseFc1GroupChunks,
    has_explicit_override: bool = SwapABBlockwiseFc1GroupChunksExplicit,
    token_group_count: Optional[int] = None,
) -> int:
    """Select the measured MXFP4 chunk count from the physical token tile."""
    if fp8_scale_mode == "mxfp4_hybrid" and not has_explicit_override:
        return 1 if token_group_count is not None and token_group_count >= 8 else 2
    if fp8_scale_mode in ("blockwise", "mxfp4_hybrid"):
        return configured_chunks
    return 1

def _resolve_mxfp4_reuse_fc1_accum_scratch(
    fp8_scale_mode: str,
    *,
    configured: int = SwapABMxfp4ReuseFc1AccumScratch,
) -> bool:
    """Keep the experimental scratch alias isolated from every FP8 path."""
    return fp8_scale_mode == "mxfp4_hybrid" and configured == 1


def _resolve_mxfp4_split_fc1_store_overlap(
    fp8_scale_mode: str,
    split_role: str,
    *,
    token_tile_n: int,
    configured: str = SwapABMxfp4SplitFc1StoreOverlap,
    configured_chunks: int = SwapABBlockwiseFc1GroupChunks,
    has_explicit_chunk_override: bool = (
        SwapABBlockwiseFc1GroupChunksExplicit
    ),
) -> str:
    """Resolve the opt-in split-K1 store-overlap experiment.

    The environment selector is intentionally harmless to the peer split-K2
    object (and to every FP8/fused object) constructed in the same process.
    Once it reaches the intended MXFP4 split-K1 path, however, unsupported
    geometry and conflicting PR357 chunk overrides fail closed.
    """
    configured = configured.strip().lower()
    if configured not in SwapABMxfp4SplitFc1StoreOverlapChoices:
        raise ValueError(
            "MXFP4 split-K1 store overlap must be one of "
            f"{SwapABMxfp4SplitFc1StoreOverlapChoices}, got "
            f"{configured!r}."
        )
    if configured == "none":
        return "none"
    if fp8_scale_mode != "mxfp4_hybrid" or split_role != "k1":
        return "none"
    if token_tile_n != 64:
        raise ValueError(
            "MXFP4 split-K1 store overlap currently requires token N=64, "
            f"got {token_tile_n}."
        )
    if (
        configured == "pr357"
        and has_explicit_chunk_override
        and configured_chunks != 2
    ):
        raise ValueError(
            "PR357-style FC1 store overlap requires "
            "MEGA_SWAPAB_FC1_GROUP_CHUNKS=2 when that variable is set, "
            f"got {configured_chunks}."
        )
    return configured


def _resolve_mxfp4_split_k1_sw64_store(
    fp8_scale_mode: str,
    split_role: str,
    *,
    fc1_output_dtype: Type[cutlass.Numeric],
    fc1_output_layout: utils.LayoutEnum,
    output_tile_m: int,
) -> bool:
    """Select the late-address SW64 store and validate its static contract."""
    use_sw64_store = (
        fp8_scale_mode == "mxfp4_hybrid" and split_role == "k1"
    )
    if not use_sw64_store:
        return False
    if fc1_output_dtype is not cutlass.Float8E4M3FN:
        raise ValueError(
            "MXFP4 split-K1 SW64 store requires Float8E4M3FN FC1 output, "
            f"got {fc1_output_dtype}."
        )
    if fc1_output_layout != utils.LayoutEnum.ROW_MAJOR:
        raise ValueError(
            "MXFP4 split-K1 SW64 store requires ROW_MAJOR FC1 output, "
            f"got {fc1_output_layout}."
        )
    if output_tile_m != Fc1EpilogueStoreTileN:
        raise ValueError(
            "MXFP4 split-K1 SW64 store requires physical output width "
            f"{Fc1EpilogueStoreTileN}, got {output_tile_m}."
        )
    return True

WarpThreadCount = 32
EpiWarpCount = 4

# =============================================================================
# SwapABFp8GluEpilogue
# =============================================================================

class SwapABFp8GluEpilogue:

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
        fc1_amax_sync_bar_id: int = 4,
        epilogue_warp_ids: Tuple[int, ...] = (0, 1, 2, 3),
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        fc2_in_kernel_topk_reduce: bool = False,
        apply_topk_in_fc1: bool = False,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: Union[int, Tuple[int, int]] = 1,
        split_handoff_token_n: Optional[int] = None,
        split_role: str = "fused",
        pingpong: bool = False,
    ) -> None:
        self.fc1_output_dtype = fc1_output_dtype
        self.fc1_output_layout = fc1_output_layout
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        if fp8_scale_mode not in (
            "per_tensor", "blockwise", "mxfp4_hybrid"
        ):
            raise ValueError(
                "fp8_scale_mode must be 'per_tensor', 'blockwise', or "
                f"'mxfp4_hybrid', "
                f"got {fp8_scale_mode!r}."
            )
        self.fp8_scale_mode = fp8_scale_mode
        if split_role not in ("fused", "k1", "k2"):
            raise ValueError(
                "split_role must be 'fused', 'k1', or 'k2', "
                f"got {split_role!r}."
            )
        self._split_role = split_role
        self.fp8_output_rcp_limit = cutlass.Float32(fp8_output_rcp_limit)
        self._sf_vec_size = sf_vec_size
        self._epilog_sync_bar_id = epilog_sync_bar_id
        self._fc1_store_sync_bar_id = fc1_store_sync_bar_id
        self._fc1_amax_sync_bar_id = fc1_amax_sync_bar_id
        self._epilogue_warp_ids = epilogue_warp_ids
        self._pingpong = pingpong
        self._pingpong_order = pingpong
        self._use_2cta_instrs = use_2cta_instrs
        self._cluster_n = cluster_shape_mn[1]

        self._atom_thr_size = 1
        self._raw_cta_tile_m = mma_tiler_mnk[0]
        self._token_tile_n = mma_tiler_mnk[1]
        self._split_handoff_token_n = split_handoff_token_n
        self._cluster_token_ctas = cluster_shape_mn[1]
        if len(epilogue_warp_ids) % EpiWarpCount != 0:
            raise ValueError(
                "Swap-AB epilogue warp count must be a multiple of one "
                f"warpgroup ({EpiWarpCount}), got {len(epilogue_warp_ids)}."
            )
        self._epilogue_warpgroup_count = (
            len(epilogue_warp_ids) // EpiWarpCount
        )
        self._wgmma_fragment_count = self._raw_cta_tile_m // 128
        expected_warpgroup_count = (
            2 if self._pingpong else self._wgmma_fragment_count
        )
        if self._epilogue_warpgroup_count != expected_warpgroup_count:
            raise ValueError(
                "Swap-AB epilogue warpgroup count does not match the selected "
                f"scheduling mode; got {self._epilogue_warpgroup_count}, "
                f"expected {expected_warpgroup_count}."
            )
        self._wg_raw_tile_m = 128
        self._wg_output_tile_m = self._wg_raw_tile_m // 2
        self._mma_tiler_k = mma_tiler_mnk[2]
        self._token_group_count = self._token_tile_n // 8
        self._accum_regs_per_m64 = self._token_tile_n // 2 # n // 8 * 4 = n // 2
        self._folded_values_per_m64_thread = self._token_tile_n // 4
        self._fc1_store_overlap_style = (
            _resolve_mxfp4_split_fc1_store_overlap(
                self.fp8_scale_mode,
                self._split_role,
                token_tile_n=self._token_tile_n,
            )
        )
        self._fc1_store_temporal_stages = (
            2 if self._fc1_store_overlap_style == "pr360" else 1
        )
        if self._fc1_store_overlap_style == "pr357":
            self._blockwise_fc1_group_chunks = 2
        else:
            self._blockwise_fc1_group_chunks = _resolve_swapab_fc1_group_chunks(
                self.fp8_scale_mode,
                token_group_count=self._token_group_count,
            )
        self._reuse_fc1_accum_temp_scratch = (
            _resolve_mxfp4_reuse_fc1_accum_scratch(self.fp8_scale_mode)
        )
        if self._token_group_count % self._blockwise_fc1_group_chunks != 0:
            raise ValueError(
                "MEGA_SWAPAB_FC1_GROUP_CHUNKS must divide the swap-AB token "
                f"group count {self._token_group_count}, got "
                f"{self._blockwise_fc1_group_chunks}."
            )
        self._blockwise_fc1_groups_per_chunk = (
            self._token_group_count // self._blockwise_fc1_group_chunks
        )
        self._static_expert_shape = static_expert_shape
        if (
            static_expert_shape is not None
            and static_expert_shape[2]
            % (self._raw_cta_tile_m * cluster_shape_mn[0])
            == 0
        ):
            self._fc2_stg_needs_predicate: bool = False
        else:
            self._fc2_stg_needs_predicate: bool = True

        if self._raw_cta_tile_m not in SwapABTileMChoices:
            raise ValueError(
                "Swap-AB Hopper FP8 epilogue requires CTA tile M in "
                f"{SwapABTileMChoices}, "
                f"got {self._raw_cta_tile_m}."
            )
        if self._wg_raw_tile_m != 128:
            raise ValueError(
                "Each swap-AB epilogue warpgroup must own raw M=128; "
                f"got CTA M={self._raw_cta_tile_m} split across "
                f"{self._epilogue_warpgroup_count} warpgroup(s)."
            )
        if self._token_tile_n not in SwapABTokenTileNChoices:
            raise ValueError(
                "Swap-AB Hopper FP8 epilogue requires token N in "
                f"{SwapABTokenTileNChoices}, got {self._token_tile_n}."
            )
        self._use_mxfp4_split_k1_sw64_store = (
            _resolve_mxfp4_split_k1_sw64_store(
                self.fp8_scale_mode,
                self._split_role,
                fc1_output_dtype=self.fc1_output_dtype,
                fc1_output_layout=self.fc1_output_layout,
                output_tile_m=self._wg_output_tile_m,
            )
        )
        self._epi_tile = (self._token_tile_n, self._wg_output_tile_m)
        self._subtile_cnt = self._epilogue_warpgroup_count
        self._iket_fc1_epilogue_pt_range = (
            f"swapab_fc1_epi_m{self._token_tile_n}n64_pt"
        )
        self._iket_fc1_epilogue_bw_range = (
            f"swapab_fc1_epi_m{self._token_tile_n}n64_bw"
        )
        self._iket_fc1_epilogue_mxfp4_range = (
            f"swapab_fc1_epi_m{self._token_tile_n}n64_mxfp4"
        )
        self._iket_fc2_epilogue_range = (
            f"swapab_fc2_epi_m{self._token_tile_n}n128"
        )

        self._fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self._apply_topk_in_fc1 = apply_topk_in_fc1
        self._token_back_by_dispatch = token_back_by_dispatch
        if isinstance(epi_flag_batch, tuple):
            self._epi_fc1_batch = max(1, epi_flag_batch[0])
            self._epi_fc2_batch = max(1, epi_flag_batch[1])
        else:
            self._epi_fc1_batch = max(1, epi_flag_batch)
            self._epi_fc2_batch = max(1, epi_flag_batch)

        # The overlap paths already delay publication until their TMA group is
        # safe. A batch of two only rotates the two atomics across lanes; it
        # does not combine them, while keeping a 64-bit address loop-live.
        self._fc1_handoff_publish_batch = (
            1
            if self._fc1_store_overlap_style in ("pr357", "pr360")
            else self._epi_fc1_batch
        )

        self.glu_clamp = (
            cutlass.Float32(glu_clamp) if glu_clamp is not None else None
        )

    # -- Codegen-time queries  --

    @property
    def epi_tile(self) -> Tuple[int, int]:
        return self._epi_tile

    @property
    def subtile_cnt(self) -> int:
        return self._subtile_cnt

    @property
    def fc1_store_overlap_style(self) -> str:
        return self._fc1_store_overlap_style

    @property
    def fc1_store_temporal_stages(self) -> int:
        return self._fc1_store_temporal_stages

    @property
    def fc1_store_row_subtiles(self) -> int:
        return 2 if self._fc1_store_overlap_style == "pr357" else 1

    @property
    def fc1_store_stage_count(self) -> int:
        return (
            self._epilogue_warpgroup_count
            * self._fc1_store_temporal_stages
        )

    @property
    def fc1_tma_store_tile(self) -> Tuple[int, int]:
        if self._fc1_store_overlap_style == "pr357":
            return (self._token_tile_n // 2, self._wg_output_tile_m)
        return self._epi_tile

    @property
    def fc1_tma_store_smem_layout(
        self,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        staged = sm90_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            self.fc1_output_layout,
            self.fc1_tma_store_tile,
            1,
        )

        return cute.select(staged, mode=[0, 1])
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
    def fc1_amax_smem_layout(self) -> cute.Layout:
        if self.fp8_scale_mode in ("blockwise", "mxfp4_hybrid"):
            # Dense scratch is [epilogue WGs, 4 warps/WG, token N].
            return cute.make_layout(
                (self._epilogue_warpgroup_count, 4, self._token_tile_n),
                stride=(4 * self._token_tile_n, self._token_tile_n, 1),
            )
        return cute.make_layout(1)

    @property
    def fc1_amax_smem_bytes(self) -> int:
        return cute.size_in_bytes(cutlass.Float32, self.fc1_amax_smem_layout)


    # -- FC1 epilogue consuming SM90 WGMMA accumulators --
    @cute.jit
    def _run_fc1_epilogue(
        self,
        work_tile_info,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        n_half: cutlass.Constexpr,
        store_stage_idx,
        smem_fc1_output_buffer: cute.Tensor,
        smem_fc1_amax: cute.Tensor,
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
        storage_group_idx,
    ) -> None:
        """Dispatch the FC1 epilogue for one completed WGMMA task tile."""
        if cutlass.const_expr(
            self.fp8_scale_mode in ("blockwise", "mxfp4_hybrid")
        ):
            self._run_fc1_epilogue_blockwise(
                work_tile_info=work_tile_info,
                accumulators=accumulators,
                accum_temp=accum_temp,
                n_half=n_half,
                store_stage_idx=store_stage_idx,
                smem_fc1_output_buffer=smem_fc1_output_buffer,
                smem_fc1_amax=smem_fc1_amax,
                tma_atom_fc1_output=tma_atom_fc1_output,
                sched_ext=sched_ext,
                gmem_fc1_output=gmem_fc1_output,
                gmem_fc1_output_sf=gmem_fc1_output_sf,
                gmem_topk_scores=gmem_topk_scores,
                local_warp_idx=local_warp_idx,
                tidx=tidx,
                _iket_active=_iket_active,
                fc1_act_weight_dequant_scale=(
                    fc1_act_weight_dequant_scale
                ),
                storage_group_idx=storage_group_idx,
            )
        else:
            self._run_fc1_epilogue_per_tensor(
                work_tile_info=work_tile_info,
                accumulators=accumulators,
                n_half=n_half,
                store_stage_idx=store_stage_idx,
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
                storage_group_idx=storage_group_idx,
            )


    @cute.jit
    def _quantize_store_fc1_pair_swapab(
        self,
        values: cute.Tensor,
        sC_stage: cute.Tensor,
        token0,
        token1,
        local_warp_idx: int,
        lane_group,
    ) -> None:
        r_fp8 = cute.make_rmem_tensor(
            values.layout.shape, self.fc1_output_dtype
        )
        r_fp8.store(values.load().to(self.fc1_output_dtype))
        for m_sub in cutlass.range_constexpr(2):
            src = m_sub * 2
            output_col = (
                cutlass.Int32(m_sub * 32)
                + cutlass.Int32(local_warp_idx * 8)
                + lane_group
            )
            sC_stage[token0, output_col] = r_fp8[src + 0]
            sC_stage[token1, output_col] = r_fp8[src + 1]

    @cute.jit
    def _run_fc1_token_group_swapab_per_tensor(
        self,
        token_group: cutlass.Constexpr,
        accumulators: cute.Tensor,
        sC_stage: cute.Tensor,
        local_warp_idx: int,
        tidx,
        fc1_act_weight_dequant_scale,
        output_rcp: Float32,
        real_topk_scores: cute.Tensor,
        token_tile_base,
    ) -> None:
        thread_in_warp = tidx % WarpThreadCount
        lane_group = thread_in_warp // 4
        lane_mod = thread_in_warp % 4
        token0 = cutlass.Int32(token_group * 8) + lane_mod * cutlass.Int32(2)
        token1 = token0 + cutlass.Int32(1)
        group_layout = cute.make_layout(4)
        r_gate = cute.make_rmem_tensor(group_layout.shape, self.acc_dtype)
        r_up = cute.make_rmem_tensor(group_layout.shape, self.acc_dtype)
        for m_sub in cutlass.range_constexpr(2):
            accum_base = m_sub * self._accum_regs_per_m64
            src = accum_base + token_group * 4
            dst = m_sub * 2
            r_gate[dst + 0] = (
                accumulators[src + 0] * fc1_act_weight_dequant_scale
            )
            r_gate[dst + 1] = (
                accumulators[src + 1] * fc1_act_weight_dequant_scale
            )
            r_up[dst + 0] = (
                accumulators[src + 2] * fc1_act_weight_dequant_scale
            )
            r_up[dst + 1] = (
                accumulators[src + 3] * fc1_act_weight_dequant_scale
            )

        r_swiglu = cute.make_rmem_tensor(group_layout.shape, self.acc_dtype)
        clamp_and_swiglu_sm90(
            r_swiglu, r_up, r_gate, self.glu_clamp, Float32(1.0)
        )
        self._apply_fc1_topk_swapab(
            r_swiglu=r_swiglu,
            real_topk_scores=real_topk_scores,
            token_tile_base=token_tile_base,
            token0=token0,
            token1=token1,
        )
        for i in cutlass.range_constexpr(4):
            r_swiglu[i] = r_swiglu[i] * output_rcp

        self._quantize_store_fc1_pair_swapab(
            values=r_swiglu,
            sC_stage=sC_stage,
            token0=token0,
            token1=token1,
            local_warp_idx=local_warp_idx,
            lane_group=lane_group,
        )

    @cute.jit
    def _run_fc1_epilogue_per_tensor(
        self,
        work_tile_info,
        accumulators: cute.Tensor,
        n_half: cutlass.Constexpr,
        store_stage_idx,
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
        storage_group_idx,
    ) -> None:
        """Fold two M64 fragments into one WG-private Nx64 FP8 tile."""
        # Inner epilogue IKET range for swap-AB, FP8 per-tensor FC1. WGMMA has
        # already finished. This covers both M=64 accumulator fragments,
        # gate/up fold, scalar dequantization, SwiGLU, requantization, and
        # RMEM-to-SMEM stores. Final TMA store issue remains in the enclosing
        # swapab_fc1_task_pt range; its later async completion
        # wait is outside the task-tile range.
        if _iket_active:
            iket.range_push(self._iket_fc1_epilogue_pt_range)
        real_topk_scores, _ = sched_ext.get_gmem_tensor(
            "topk", gmem_topk_scores, work_tile_info,
        )
        sC_stage = cute.slice_(
            smem_fc1_output_buffer, (None, None, store_stage_idx)
        )
        output_rcp = Float32(1.0) / fc2_act_dequant_scale
        token_tile_base = work_tile_info.tile_n_idx * cutlass.Int32(
            self._token_tile_n
        )

        for token_group in cutlass.range_constexpr(self._token_group_count):
            self._run_fc1_token_group_swapab_per_tensor(
                token_group=token_group,
                accumulators=accumulators,
                sC_stage=sC_stage,
                local_warp_idx=local_warp_idx,
                tidx=tidx,
                fc1_act_weight_dequant_scale=fc1_act_weight_dequant_scale,
                output_rcp=output_rcp,
                real_topk_scores=real_topk_scores,
                token_tile_base=token_tile_base,
            )
        if _iket_active:
            iket.range_pop()  # swapab_fc1_epi_m{token_tile_n}n64_pt

    @cute.jit
    def _fill_fc1_swiglu_chunk_swapab_blockwise(
        self,
        chunk_idx: cutlass.Constexpr,
        accumulators: cute.Tensor,
        r_swiglu: cute.Tensor,
        fc1_act_weight_dequant_scale,
    ) -> None:
        groups_per_chunk = self._blockwise_fc1_groups_per_chunk
        folded_values_per_chunk_m64 = 2 * groups_per_chunk
        group_start = chunk_idx * groups_per_chunk
        group_layout = cute.make_layout(4)
        group_swiglu_layout = cute.make_layout(
            (2, 2), stride=(1, folded_values_per_chunk_m64)
        )
        for local_group in cutlass.range_constexpr(groups_per_chunk):
            token_group = group_start + local_group
            pair_base = local_group * 2
            r_gate = cute.make_rmem_tensor(group_layout.shape, self.acc_dtype)
            r_up = cute.make_rmem_tensor(group_layout.shape, self.acc_dtype)
            for m_sub in cutlass.range_constexpr(2):
                src = (
                    m_sub * (2 * self._folded_values_per_m64_thread)
                    + token_group * 4
                )
                dst = m_sub * 2
                r_gate[dst + 0] = (
                    accumulators[src + 0]
                    * fc1_act_weight_dequant_scale
                )
                r_gate[dst + 1] = (
                    accumulators[src + 1]
                    * fc1_act_weight_dequant_scale
                )
                r_up[dst + 0] = (
                    accumulators[src + 2]
                    * fc1_act_weight_dequant_scale
                )
                r_up[dst + 1] = (
                    accumulators[src + 3]
                    * fc1_act_weight_dequant_scale
                )
            group_swiglu_view = cute.make_tensor(
                r_swiglu.iterator + pair_base, group_swiglu_layout
            )
            clamp_and_swiglu_sm90(
                group_swiglu_view,
                r_up,
                r_gate,
                self.glu_clamp,
                Float32(1.0),
            )

    @cute.jit
    def _apply_fc1_topk_swapab(
        self,
        r_swiglu: cute.Tensor,
        real_topk_scores: cute.Tensor,
        token_tile_base,
        token0,
        token1,
    ) -> None:
        """Scale token0/token1 values across both M64 register fragments."""
        if cutlass.const_expr(self._apply_topk_in_fc1):
            topk_score = Float32(
                real_topk_scores[token_tile_base + token0]
            )
            for m_sub in cutlass.range_constexpr(2):
                base = m_sub * 2
                r_swiglu[base + 0] = r_swiglu[base + 0] * topk_score
            topk_score = Float32(
                real_topk_scores[token_tile_base + token1]
            )
            for m_sub in cutlass.range_constexpr(2):
                base = m_sub * 2
                r_swiglu[base + 1] = r_swiglu[base + 1] * topk_score

    @cute.jit
    def _apply_fc1_topk_chunk_swapab(
        self,
        chunk_idx: cutlass.Constexpr,
        tidx,
        r_swiglu: cute.Tensor,
        real_topk_scores: cute.Tensor,
        token_tile_base,
    ) -> None:
        if cutlass.const_expr(self._apply_topk_in_fc1):
            groups_per_chunk = self._blockwise_fc1_groups_per_chunk
            folded_values_per_chunk_m64 = 2 * groups_per_chunk
            group_start = chunk_idx * groups_per_chunk
            lane_mod = (tidx % WarpThreadCount) % 4
            for local_group in cutlass.range_constexpr(groups_per_chunk):
                token_group = group_start + local_group
                pair_base = local_group * 2
                token0 = (
                    cutlass.Int32(token_group * 8)
                    + lane_mod * cutlass.Int32(2)
                )
                token1 = token0 + cutlass.Int32(1)
                topk_score0 = Float32(
                    real_topk_scores[token_tile_base + token0]
                )
                topk_score1 = Float32(
                    real_topk_scores[token_tile_base + token1]
                )
                for m_sub in cutlass.range_constexpr(2):
                    reg_base = (
                        m_sub * folded_values_per_chunk_m64 + pair_base
                    )
                    r_swiglu[reg_base + 0] = (
                        r_swiglu[reg_base + 0] * topk_score0
                    )
                    r_swiglu[reg_base + 1] = (
                        r_swiglu[reg_base + 1] * topk_score1
                    )

    @cute.jit
    def _publish_fc1_amax_chunk_swapab_blockwise(
        self,
        chunk_idx: cutlass.Constexpr,
        n_half: cutlass.Constexpr,
        local_warp_idx: int,
        tidx,
        r_swiglu: cute.Tensor,
        smem_fc1_amax: cute.Tensor,
    ) -> None:
        groups_per_chunk = self._blockwise_fc1_groups_per_chunk
        folded_values_per_chunk_m64 = 2 * groups_per_chunk
        group_start = chunk_idx * groups_per_chunk
        thread_in_warp = tidx % WarpThreadCount
        lane_group = thread_in_warp // 4
        lane_mod = thread_in_warp % 4
        for local_group in cutlass.range_constexpr(groups_per_chunk):
            token_group = group_start + local_group
            idx0 = local_group * 2
            idx1 = idx0 + 1
            token0_max = fmax(
                fmax(r_swiglu[idx0], -r_swiglu[idx0]),
                fmax(
                    r_swiglu[folded_values_per_chunk_m64 + idx0],
                    -r_swiglu[folded_values_per_chunk_m64 + idx0],
                ),
            )
            token1_max = fmax(
                fmax(r_swiglu[idx1], -r_swiglu[idx1]),
                fmax(
                    r_swiglu[folded_values_per_chunk_m64 + idx1],
                    -r_swiglu[folded_values_per_chunk_m64 + idx1],
                ),
            )
            for delta in (4, 8, 16):
                token0_max = fmax(
                    token0_max,
                    cute.arch.shuffle_sync_bfly(token0_max, offset=delta),
                )
                token1_max = fmax(
                    token1_max,
                    cute.arch.shuffle_sync_bfly(token1_max, offset=delta),
                )
            if lane_group == cutlass.Int32(0):
                token0 = (
                    cutlass.Int32(token_group * 8)
                    + lane_mod * cutlass.Int32(2)
                )
                token1 = token0 + cutlass.Int32(1)
                smem_fc1_amax[n_half, local_warp_idx, token0] = token0_max
                smem_fc1_amax[n_half, local_warp_idx, token1] = token1_max

    @cute.jit
    def _finalize_fc1_scale_chunk_swapab_blockwise(
        self,
        work_tile_info,
        chunk_idx: cutlass.Constexpr,
        n_half: cutlass.Constexpr,
        local_warp_idx: int,
        tidx,
        smem_fc1_amax: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        r_scale: cute.Tensor,
        scale_epsilon: Float32,
        global_token_base,
        scale_col_idx,
    ) -> None:
        groups_per_chunk = self._blockwise_fc1_groups_per_chunk
        group_start = chunk_idx * groups_per_chunk
        thread_in_warp = tidx % WarpThreadCount
        lane_group = thread_in_warp // 4
        lane_mod = thread_in_warp % 4
        for local_group in cutlass.range_constexpr(groups_per_chunk):
            token_group = group_start + local_group
            pair_base = local_group * 2
            token0 = (
                cutlass.Int32(token_group * 8) + lane_mod * cutlass.Int32(2)
            )
            token1 = token0 + cutlass.Int32(1)
            # Balanced 4-to-1 reduction keeps the FMAX dependency depth at 2.
            token0_max = fmax(
                fmax(
                    smem_fc1_amax[n_half, 0, token0],
                    smem_fc1_amax[n_half, 1, token0],
                ),
                fmax(
                    smem_fc1_amax[n_half, 2, token0],
                    smem_fc1_amax[n_half, 3, token0],
                ),
            )
            token1_max = fmax(
                fmax(
                    smem_fc1_amax[n_half, 0, token1],
                    smem_fc1_amax[n_half, 1, token1],
                ),
                fmax(
                    smem_fc1_amax[n_half, 2, token1],
                    smem_fc1_amax[n_half, 3, token1],
                ),
            )
            scale0 = fmax(
                token0_max * self.fp8_output_rcp_limit, scale_epsilon
            )
            scale1 = fmax(
                token1_max * self.fp8_output_rcp_limit, scale_epsilon
            )
            r_scale[pair_base + 0] = scale0
            r_scale[pair_base + 1] = scale1
            if local_warp_idx == cutlass.Int32(0):
                if lane_group == cutlass.Int32(0):
                    if token0 < work_tile_info.valid_tokens_in_cta_tile:
                        stg_fc1_block_scale_row(
                            gmem_fc1_output_sf,
                            scale_col_idx,
                            global_token_base + token0,
                            scale0,
                        )
                    if token1 < work_tile_info.valid_tokens_in_cta_tile:
                        stg_fc1_block_scale_row(
                            gmem_fc1_output_sf,
                            scale_col_idx,
                            global_token_base + token1,
                            scale1,
                        )

    @cute.jit
    def _publish_fc1_scale_rcp_chunk_swapab_blockwise(
        self,
        work_tile_info,
        chunk_idx: cutlass.Constexpr,
        n_half: cutlass.Constexpr,
        local_warp_idx: int,
        tidx,
        smem_fc1_amax: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        scale_epsilon: Float32,
        global_token_base,
        scale_col_idx,
    ) -> None:
        """Publish each output scale and cache its reciprocal in SMEM."""
        thread_in_warp = tidx % WarpThreadCount
        lane_group = thread_in_warp // 4
        lane_mod = thread_in_warp % 4
        groups_per_chunk = self._blockwise_fc1_groups_per_chunk
        group_start = chunk_idx * groups_per_chunk
        if local_warp_idx == cutlass.Int32(0):
            # One four-lane group owns one eight-token group.  The old path
            # let only lane_group 0 serialize every token group while the
            # other seven lane groups in warp 0 were idle.  Keep ownership
            # unique, but process up to eight groups in parallel per wave.
            publisher_waves = (groups_per_chunk + 7) // 8
            for publisher_wave in cutlass.range_constexpr(publisher_waves):
                local_group = (
                    cutlass.Int32(publisher_wave * 8) + lane_group
                )
                if local_group < cutlass.Int32(groups_per_chunk):
                    token_group = group_start + local_group
                    token0 = (
                        cutlass.Int32(token_group * 8)
                        + lane_mod * cutlass.Int32(2)
                    )
                    token1 = token0 + cutlass.Int32(1)
                    token0_max = fmax(
                        fmax(
                            smem_fc1_amax[n_half, 0, token0],
                            smem_fc1_amax[n_half, 1, token0],
                        ),
                        fmax(
                            smem_fc1_amax[n_half, 2, token0],
                            smem_fc1_amax[n_half, 3, token0],
                        ),
                    )
                    token1_max = fmax(
                        fmax(
                            smem_fc1_amax[n_half, 0, token1],
                            smem_fc1_amax[n_half, 1, token1],
                        ),
                        fmax(
                            smem_fc1_amax[n_half, 2, token1],
                            smem_fc1_amax[n_half, 3, token1],
                        ),
                    )
                    scale0 = fmax(
                        token0_max * self.fp8_output_rcp_limit,
                        scale_epsilon,
                    )
                    scale1 = fmax(
                        token1_max * self.fp8_output_rcp_limit,
                        scale_epsilon,
                    )
                    if token0 < work_tile_info.valid_tokens_in_cta_tile:
                        stg_fc1_block_scale_row(
                            gmem_fc1_output_sf,
                            scale_col_idx,
                            global_token_base + token0,
                            scale0,
                        )
                    if token1 < work_tile_info.valid_tokens_in_cta_tile:
                        stg_fc1_block_scale_row(
                            gmem_fc1_output_sf,
                            scale_col_idx,
                            global_token_base + token1,
                            scale1,
                        )
                    smem_fc1_amax[n_half, 0, token0] = (
                        Float32(1.0) / scale0
                    )
                    smem_fc1_amax[n_half, 0, token1] = (
                        Float32(1.0) / scale1
                    )

    @cute.jit
    def _load_fc1_scale_rcp_chunk_swapab_blockwise(
        self,
        chunk_idx: cutlass.Constexpr,
        n_half: cutlass.Constexpr,
        tidx,
        smem_fc1_amax: cute.Tensor,
        r_scale: cute.Tensor,
    ) -> None:
        groups_per_chunk = self._blockwise_fc1_groups_per_chunk
        group_start = chunk_idx * groups_per_chunk
        lane_mod = (tidx % WarpThreadCount) % 4
        for local_group in cutlass.range_constexpr(groups_per_chunk):
            token_group = group_start + local_group
            pair_base = local_group * 2
            token0 = (
                cutlass.Int32(token_group * 8)
                + lane_mod * cutlass.Int32(2)
            )
            token1 = token0 + cutlass.Int32(1)
            r_scale[pair_base + 0] = smem_fc1_amax[
                n_half, 0, token0
            ]
            r_scale[pair_base + 1] = smem_fc1_amax[
                n_half, 0, token1
            ]

    @cute.jit
    def _quantize_store_fc1_chunk_swapab_blockwise(
        self,
        chunk_idx: cutlass.Constexpr,
        local_warp_idx: int,
        tidx,
        r_swiglu: cute.Tensor,
        r_scale: cute.Tensor,
        sC_stage: cute.Tensor,
    ) -> None:
        groups_per_chunk = self._blockwise_fc1_groups_per_chunk
        folded_values_per_chunk_m64 = 2 * groups_per_chunk
        group_start = chunk_idx * groups_per_chunk
        group_layout = cute.make_layout(4)
        thread_in_warp = tidx % WarpThreadCount
        lane_group = thread_in_warp // 4
        lane_mod = thread_in_warp % 4
        for local_group in cutlass.range_constexpr(groups_per_chunk):
            token_group = group_start + local_group
            pair_base = local_group * 2
            token0 = (
                cutlass.Int32(token_group * 8) + lane_mod * cutlass.Int32(2)
            )
            token1 = token0 + cutlass.Int32(1)
            scale0_rcp = r_scale[pair_base + 0]
            scale1_rcp = r_scale[pair_base + 1]
            if cutlass.const_expr(
                self.fp8_scale_mode != "mxfp4_hybrid"
            ):
                scale0_rcp = Float32(1.0) / scale0_rcp
                scale1_rcp = Float32(1.0) / scale1_rcp
            r_quant = cute.make_rmem_tensor(group_layout.shape, self.acc_dtype)
            for m_sub in cutlass.range_constexpr(2):
                reg_base = m_sub * folded_values_per_chunk_m64 + pair_base
                dst = m_sub * 2
                r_quant[dst + 0] = r_swiglu[reg_base + 0] * scale0_rcp
                r_quant[dst + 1] = r_swiglu[reg_base + 1] * scale1_rcp
            self._quantize_store_fc1_pair_swapab(
                values=r_quant,
                sC_stage=sC_stage,
                token0=token0,
                token1=token1,
                local_warp_idx=local_warp_idx,
                lane_group=lane_group,
            )

    @cute.jit
    def _run_fc1_epilogue_blockwise(
        self,
        work_tile_info,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        n_half: cutlass.Constexpr,
        store_stage_idx,
        smem_fc1_output_buffer: cute.Tensor,
        smem_fc1_amax: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        local_warp_idx: int,
        tidx,
        _iket_active,
        fc1_act_weight_dequant_scale,
        storage_group_idx,
    ) -> None:
        """Process blockwise FC1 token groups in register-bounded chunks."""
        # Inner epilogue IKET range for swap-AB, DeepGEMM-style blockwise FC1.
        # WGMMA and blockwise accumulator scaling have already finished. This
        # spans every register-bounded chunk, including gate/up fold, SwiGLU,
        # warpgroup amax reduction, scale publication, blockwise quantization,
        # and RMEM-to-SMEM stores. Final TMA store belongs to the outer task.
        if _iket_active:
            if cutlass.const_expr(
                self.fp8_scale_mode == "mxfp4_hybrid"
            ):
                iket.range_push(self._iket_fc1_epilogue_mxfp4_range)
            else:
                iket.range_push(self._iket_fc1_epilogue_bw_range)
        real_topk_scores, _ = sched_ext.get_gmem_tensor(
            "topk", gmem_topk_scores, work_tile_info,
        )
        groups_per_chunk = self._blockwise_fc1_groups_per_chunk
        folded_values_per_chunk_m64 = 2 * groups_per_chunk
        value_layout = cute.make_layout(2 * folded_values_per_chunk_m64)
        scale_layout = cute.make_layout(folded_values_per_chunk_m64)
        scratch_regs = cute.size(value_layout) + cute.size(scale_layout)
        reuse_accum_temp_scratch = (
            self._reuse_fc1_accum_temp_scratch
            and scratch_regs <= cute.size(accum_temp)
        )
        amax_bar = pipeline.NamedBarrier(
            barrier_id=self._fc1_amax_sync_bar_id + storage_group_idx,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        scale_epsilon = Float32(1.0e-30)
        global_token_base = (
            work_tile_info.cumulative_data_physical_row
            + work_tile_info.tile_n_idx * cutlass.Int32(self._token_tile_n)
        )
        scale_col_idx = (
            work_tile_info.tile_m_idx
            * cutlass.Int32(self._wgmma_fragment_count)
            + cutlass.Int32(n_half)
        )
        sC_stage = cute.slice_(
            smem_fc1_output_buffer, (None, None, store_stage_idx)
        )
        token_tile_base = work_tile_info.tile_n_idx * cutlass.Int32(
            self._token_tile_n
        )

        for chunk_idx in cutlass.range_constexpr(
            self._blockwise_fc1_group_chunks
        ):
            chunk_has_valid_tokens = cutlass.Boolean(True)
            if cutlass.const_expr(self._fc1_store_overlap_style == "pr357"):
                chunk_token_start = cutlass.Int32(
                    chunk_idx * groups_per_chunk * 8
                )
                chunk_has_valid_tokens = (
                    work_tile_info.valid_tokens_in_cta_tile
                    > chunk_token_start
                )
            if cutlass.const_expr(reuse_accum_temp_scratch):
                # FC1 has promoted accum_temp and waited for WGMMA completion.
                # Reuse those now-dead registers instead of extending another
                # SwiGLU tensor's live range. FC2 overwrites them before use.
                r_swiglu = cute.make_tensor(
                    accum_temp.iterator, value_layout
                )
            else:
                r_swiglu = cute.make_rmem_tensor(
                    value_layout.shape, self.acc_dtype
                )
            if chunk_has_valid_tokens:
                self._fill_fc1_swiglu_chunk_swapab_blockwise(
                    chunk_idx=chunk_idx,
                    accumulators=accumulators,
                    r_swiglu=r_swiglu,
                    fc1_act_weight_dequant_scale=(
                        fc1_act_weight_dequant_scale
                    ),
                )
                self._apply_fc1_topk_chunk_swapab(
                    chunk_idx=chunk_idx,
                    tidx=tidx,
                    r_swiglu=r_swiglu,
                    real_topk_scores=real_topk_scores,
                    token_tile_base=token_tile_base,
                )
                self._publish_fc1_amax_chunk_swapab_blockwise(
                    chunk_idx=chunk_idx,
                    n_half=storage_group_idx,
                    local_warp_idx=local_warp_idx,
                    tidx=tidx,
                    r_swiglu=r_swiglu,
                    smem_fc1_amax=smem_fc1_amax,
                )

                cute.arch.fence_proxy("async.shared", space="cta")
                amax_bar.arrive_and_wait()

            if cutlass.const_expr(reuse_accum_temp_scratch):
                r_scale = cute.make_tensor(
                    accum_temp.iterator + cute.size(value_layout),
                    scale_layout,
                )
            else:
                r_scale = cute.make_rmem_tensor(
                    scale_layout.shape, cutlass.Float32
                )
            if cutlass.const_expr(self.fp8_scale_mode == "mxfp4_hybrid"):
                if chunk_has_valid_tokens:
                    self._publish_fc1_scale_rcp_chunk_swapab_blockwise(
                        work_tile_info=work_tile_info,
                        chunk_idx=chunk_idx,
                        n_half=storage_group_idx,
                        local_warp_idx=local_warp_idx,
                        tidx=tidx,
                        smem_fc1_amax=smem_fc1_amax,
                        gmem_fc1_output_sf=gmem_fc1_output_sf,
                        scale_epsilon=scale_epsilon,
                        global_token_base=global_token_base,
                        scale_col_idx=scale_col_idx,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    amax_bar.arrive_and_wait()
                    self._load_fc1_scale_rcp_chunk_swapab_blockwise(
                        chunk_idx=chunk_idx,
                        n_half=storage_group_idx,
                        tidx=tidx,
                        smem_fc1_amax=smem_fc1_amax,
                        r_scale=r_scale,
                    )
            else:
                if chunk_has_valid_tokens:
                    self._finalize_fc1_scale_chunk_swapab_blockwise(
                        work_tile_info=work_tile_info,
                        chunk_idx=chunk_idx,
                        n_half=storage_group_idx,
                        local_warp_idx=local_warp_idx,
                        tidx=tidx,
                        smem_fc1_amax=smem_fc1_amax,
                        gmem_fc1_output_sf=gmem_fc1_output_sf,
                        r_scale=r_scale,
                        scale_epsilon=scale_epsilon,
                        global_token_base=global_token_base,
                        scale_col_idx=scale_col_idx,
                    )
            if chunk_has_valid_tokens:
                self._quantize_store_fc1_chunk_swapab_blockwise(
                    chunk_idx=chunk_idx,
                    local_warp_idx=local_warp_idx,
                    tidx=tidx,
                    r_swiglu=r_swiglu,
                    r_scale=r_scale,
                    sC_stage=sC_stage,
                )
            if cutlass.const_expr(self._fc1_store_overlap_style == "pr357"):
                self._store_fc1_task_subtile_pr357(
                    work_tile_info=work_tile_info,
                    chunk_idx=chunk_idx,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    tma_atom_fc1_output=tma_atom_fc1_output,
                    gmem_fc1_output=gmem_fc1_output,
                    local_warp_idx=local_warp_idx,
                    n_half=n_half,
                    store_stage_idx=store_stage_idx,
                )
        if _iket_active:
            iket.range_pop()  # swapab_fc1_epi_m{token_tile_n}n64_bw


    @cute.jit
    def _store_fc1_task_subtile_pr357(
        self,
        work_tile_info,
        chunk_idx: cutlass.Constexpr,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        gmem_fc1_output: cute.Tensor,
        local_warp_idx: int,
        n_half: cutlass.Constexpr,
        store_stage_idx,
    ) -> None:
        cute.arch.fence_proxy("async.shared", space="cta")
        fc1_store_bar = pipeline.NamedBarrier(
            barrier_id=self._fc1_store_sync_bar_id + n_half,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        fc1_store_bar.arrive_and_wait()

        if local_warp_idx == cutlass.Int32(0):
            output_token_subtile = (
                work_tile_info.cumulative_data_physical_row
                // cutlass.Int32(32)
                + work_tile_info.tile_n_idx
                * cutlass.Int32(2)
                + cutlass.Int32(chunk_idx)
            )
            output_n_tile = (
                work_tile_info.tile_m_idx
                * cutlass.Int32(self._epilogue_warpgroup_count)
                + cutlass.Int32(n_half)
            )
            sC_stage = cute.slice_(
                smem_fc1_output_buffer, (None, None, store_stage_idx)
            )
            sC_subtile = cute.local_tile(
                sC_stage,
                (32, Fc1EpilogueStoreTileN),
                (chunk_idx, 0),
            )
            g_fc1_output_subtile = cute.local_tile(
                gmem_fc1_output,
                (32, Fc1EpilogueStoreTileN, 1),
                (
                    output_token_subtile,
                    output_n_tile,
                    0,
                ),
            )
            g_fc1_output_2d = cute.slice_(
                g_fc1_output_subtile, (None, None, 0)
            )
            bSG_sC, bSG_g = cpasync.tma_partition(
                tma_atom_fc1_output,
                0,
                cute.make_layout(1),
                cute.group_modes(sC_subtile, 0, 2),
                cute.group_modes(g_fc1_output_2d, 0, 2),
            )
            subtile_has_valid = (
                work_tile_info.valid_tokens_in_cta_tile
                > cutlass.Int32(chunk_idx * 32)
            )
            if subtile_has_valid:
                with cute.arch.elect_one():
                    cute.copy(tma_atom_fc1_output, bSG_sC, bSG_g)
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _store_fc1_task_tile(
        self,
        work_tile_info,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        local_warp_idx: int,
        n_half: cutlass.Constexpr,
        store_stage_idx,
        storage_group_idx,
        _iket_active,
    ) -> None:
        if cutlass.const_expr(self._fc1_store_overlap_style == "pr357"):
            return
        real_fc1_output, _ = sched_ext.get_gmem_tensor(
            "c", gmem_fc1_output, work_tile_info,
        )

        # Each M=128 WG folds to one contiguous output-channel block of 64.
        output_n_tile = (
            work_tile_info.tile_m_idx
            * cutlass.Int32(self._wgmma_fragment_count)
            + cutlass.Int32(n_half)
        )

        cute.arch.fence_proxy("async.shared", space="cta")
        fc1_store_bar = pipeline.NamedBarrier(
            barrier_id=self._fc1_store_sync_bar_id + storage_group_idx,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        if _iket_active:
            iket.range_push("swapab_fc1_store_smem_barrier")
        fc1_store_bar.arrive_and_wait()
        if _iket_active:
            iket.range_pop()

        if local_warp_idx == cutlass.Int32(0):
            stage_idx = store_stage_idx
            g_fc1_output_wg_view = cute.local_tile(
                real_fc1_output,
                (self._token_tile_n, Fc1EpilogueStoreTileN, 1),
                (work_tile_info.tile_n_idx, output_n_tile, 0),
            )
            if _iket_active:
                iket.range_push("swapab_fc1_tma_store_issue")
            tma_store_fc1_output(
                smem_fc1_output_buffer,
                stage_idx,
                tma_atom_fc1_output,
                g_fc1_output_wg_view,
                work_tile_info.valid_tokens_in_cta_tile,
            )
            if _iket_active:
                iket.range_pop()

    @cute.jit
    def _run_fc2_token_group_swapab(
        self,
        work_tile_info,
        token_group: cutlass.Constexpr,
        accumulators: cute.Tensor,
        n_half: cutlass.Constexpr,
        real_fc2_output: cute.Tensor,
        valid_hidden,
        token_tile_base,
        pool_token_base,
        local_warp_idx: int,
        tidx,
        fc2_act_weight_dequant_scale,
        token_comm_args=None,
    ) -> None:
        thread_in_warp = tidx % WarpThreadCount
        lane_group = thread_in_warp // 4
        lane_mod = thread_in_warp % 4
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        token0 = cutlass.Int32(token_group * 8) + lane_mod * cutlass.Int32(2)
        token1 = token0 + cutlass.Int32(1)
        pair_layout = cute.make_layout(4)
        for m_sub in cutlass.range_constexpr(2):
            accum_base = m_sub * self._accum_regs_per_m64 + token_group * 4
            r_fp32 = cute.make_rmem_tensor(pair_layout.shape, self.acc_dtype)
            for i in cutlass.range_constexpr(4):
                r_fp32[i] = (
                    accumulators[accum_base + i]
                    * fc2_act_weight_dequant_scale
                )
            r_bf16 = cute.make_rmem_tensor(pair_layout.shape, cutlass.BFloat16)
            r_bf16.store(r_fp32.load().to(cutlass.BFloat16))
            hidden0 = (
                work_tile_info.tile_m_idx
                * cutlass.Int32(self._raw_cta_tile_m)
                + cutlass.Int32(n_half * self._wg_raw_tile_m)
                + cutlass.Int32(m_sub * 64)
                + cutlass.Int32(local_warp_idx * 16)
                + lane_group
            )
            hidden1 = hidden0 + cutlass.Int32(8)

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
                if cutlass.const_expr(self._fc2_in_kernel_topk_reduce):
                    # Four lane-groups own four adjacent hidden cells for the
                    # same token. Gather them into two packed bf16x2 registers
                    # so one vector REDG covers the full 8-byte segment.
                    r_bf16_u16 = cute.recast_tensor(r_bf16, cutlass.Uint16)
                    source_lane_base = (
                        (lane_group // cutlass.Int32(4)) * cutlass.Int32(16)
                        + lane_mod
                    )
                    for value_idx in cutlass.range_constexpr(4):
                        raw = cutlass.Uint32(r_bf16_u16[value_idx])
                        value0 = cute.arch.shuffle_sync(raw, source_lane_base)
                        value1 = cute.arch.shuffle_sync(
                            raw, source_lane_base + cutlass.Int32(4),
                        )
                        value2 = cute.arch.shuffle_sync(
                            raw, source_lane_base + cutlass.Int32(8),
                        )
                        value3 = cute.arch.shuffle_sync(
                            raw, source_lane_base + cutlass.Int32(12),
                        )
                        packed0 = value0 | (value1 << cutlass.Uint32(16))
                        packed1 = value2 | (value3 << cutlass.Uint32(16))
                        token = token0
                        hidden = hidden0
                        if cutlass.const_expr(value_idx % 2 == 1):
                            token = token1
                        if cutlass.const_expr(value_idx >= 2):
                            hidden = hidden1
                        if (
                            lane_group % cutlass.Int32(4) == cutlass.Int32(0)
                            and token < valid_tokens
                            and hidden < valid_hidden
                        ):
                            dest_row = fc2_output_dest.resolve_token_row(
                                pool_token_base + token
                            )
                            dest_ptr = cute.make_ptr(
                                cutlass.BFloat16,
                                dest_row.iterator.toint()
                                + hidden * cutlass.Int64(2),
                                cute.AddressSpace.gmem,
                                assumed_align=8,
                            )
                            red_add_relaxed_sys_v2_bf16x2(
                                dest_ptr, packed0, packed1,
                            )
                else:
                    if token0 < valid_tokens:
                        dest_row0 = fc2_output_dest.resolve_token_row(
                            pool_token_base + token0
                        )
                        if hidden0 < valid_hidden:
                            dest_row0[hidden0] = r_bf16[0]
                        if hidden1 < valid_hidden:
                            dest_row0[hidden1] = r_bf16[2]
                    if token1 < valid_tokens:
                        dest_row1 = fc2_output_dest.resolve_token_row(
                            pool_token_base + token1
                        )
                        if hidden0 < valid_hidden:
                            dest_row1[hidden0] = r_bf16[1]
                        if hidden1 < valid_hidden:
                            dest_row1[hidden1] = r_bf16[3]
            else:
                if token0 < valid_tokens:
                    if hidden0 < valid_hidden:
                        real_fc2_output[
                            token_tile_base + token0, hidden0, 0
                        ] = r_bf16[0]
                    if hidden1 < valid_hidden:
                        real_fc2_output[
                            token_tile_base + token0, hidden1, 0
                        ] = r_bf16[2]
                if token1 < valid_tokens:
                    if hidden0 < valid_hidden:
                        real_fc2_output[
                            token_tile_base + token1, hidden0, 0
                        ] = r_bf16[1]
                    if hidden1 < valid_hidden:
                        real_fc2_output[
                            token_tile_base + token1, hidden1, 0
                        ] = r_bf16[3]

    @cute.jit
    def _run_fc2_epilogue(
        self,
        work_tile_info,
        accumulators: cute.Tensor,
        n_half: cutlass.Constexpr,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        valid_hidden,
        local_warp_idx: int,
        tidx,
        _iket_active,
        fc2_act_weight_dequant_scale,
        token_comm_args=None,
    ) -> None:
        """Run the FC2 epilogue and store one swap-AB accumulator tile."""
        real_fc2_output, _ = sched_ext.get_gmem_tensor(
            "c", gmem_fc2_output, work_tile_info,
        )
        # Inner epilogue IKET range for swap-AB FC2. WGMMA and any blockwise
        # accumulator scaling have already finished. This shared path converts
        # all token-group fragments to BF16 and performs token-major GMEM STG.
        if _iket_active:
            iket.range_push(self._iket_fc2_epilogue_range)

        token_tile_base = work_tile_info.tile_n_idx * cutlass.Int32(
            self._token_tile_n
        )
        pool_token_base = (
            work_tile_info.cumulative_data_physical_row + token_tile_base
        )

        for token_group in cutlass.range_constexpr(self._token_group_count):
            self._run_fc2_token_group_swapab(
                work_tile_info=work_tile_info,
                token_group=token_group,
                accumulators=accumulators,
                n_half=n_half,
                real_fc2_output=real_fc2_output,
                valid_hidden=valid_hidden,
                token_tile_base=token_tile_base,
                pool_token_base=pool_token_base,
                local_warp_idx=local_warp_idx,
                tidx=tidx,
                fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                token_comm_args=token_comm_args,
            )

        if _iket_active:
            iket.range_pop()  # swapab_fc2_epi_m{token_tile_n}n128


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
        store_stage_idx,
        ab_pipeline,
        weight_sf_pipeline,
        ab_consumer_state,
        k_tile_cnt_fc1,
        k_tile_cnt_fc2,
        _iket_active,
        run_wgmma_task_tile: cutlass.Constexpr,
        smem_fc1_output_buffer: cute.Tensor,
        smem_fc1_amax: cute.Tensor,
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
        storage_group_idx,
        math_wg_order_barrier,
        math_wg_order_state,
        epi_wg_order_barrier,
        epi_wg_order_state,
    ):
        if cutlass.const_expr(self._pingpong_order):
            if _iket_active:
                iket.range_push("pp_swap_fc1_math_wait")
            math_wg_order_barrier.wait(math_wg_order_state)
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_swap_fc1_wgmma")
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
                iket.range_push("pp_swap_fc1_math_handoff")
            math_wg_order_state = math_wg_order_barrier.arrive(
                math_wg_order_state
            )
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_swap_fc1_epi_wait")
            epi_wg_order_barrier.wait(epi_wg_order_state)
            if _iket_active:
                iket.range_pop()
        self._run_fc1_epilogue(
            work_tile_info=work_tile_info,
            accumulators=accumulators,
            accum_temp=accum_temp,
            n_half=n_half,
            store_stage_idx=store_stage_idx,
            smem_fc1_output_buffer=smem_fc1_output_buffer,
            smem_fc1_amax=smem_fc1_amax,
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
            storage_group_idx=storage_group_idx,
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
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        valid_hidden,
        tidx,
        fc2_act_weight_dequant_scale,
        math_wg_order_barrier,
        math_wg_order_state,
        epi_wg_order_barrier,
        epi_wg_order_state,
        token_comm_args=None,
    ):
        if cutlass.const_expr(self._pingpong_order):
            if _iket_active:
                iket.range_push("pp_swap_fc2_math_wait")
            math_wg_order_barrier.wait(math_wg_order_state)
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_swap_fc2_wgmma")
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
                iket.range_push("pp_swap_fc2_math_handoff")
            math_wg_order_state = math_wg_order_barrier.arrive(
                math_wg_order_state
            )
            if _iket_active:
                iket.range_pop()
                iket.range_push("pp_swap_fc2_epi_wait")
            epi_wg_order_barrier.wait(epi_wg_order_state)
            if _iket_active:
                iket.range_pop()
        self._run_fc2_epilogue(
            work_tile_info=work_tile_info,
            accumulators=accumulators,
            n_half=n_half,
            sched_ext=sched_ext,
            gmem_fc2_output=gmem_fc2_output,
            valid_hidden=valid_hidden,
            local_warp_idx=local_warp_idx,
            tidx=tidx,
            _iket_active=_iket_active,
            fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
            token_comm_args=token_comm_args,
        )
        return ab_consumer_state, math_wg_order_state, epi_wg_order_state

    @cute.jit
    def run(
        self,
        sched_consumer,
        sched_ext,
        smem_fc1_output_buffer: cute.Tensor,
        smem_fc1_amax: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        gmem_fc2_output: cute.Tensor,
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

        valid_hidden = cutlass.Int32(gmem_fc2_output.shape[1])
        if _iket_active:
            iket.range_push("swapab_sched_consume_initial")
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
                        iket.range_push("swapab_fc1_task_bw")
                    elif cutlass.const_expr(
                        self.fp8_scale_mode == "mxfp4_hybrid"
                    ):
                        iket.range_push("swapab_fc1_task_mxfp4_hybrid")
                    else:
                        iket.range_push("swapab_fc1_task_pt")
                else:
                    if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                        iket.range_push("swapab_fc2_task_bw")
                    elif cutlass.const_expr(
                        self.fp8_scale_mode == "mxfp4_hybrid"
                    ):
                        iket.range_push("swapab_fc2_task_mxfp4_hybrid")
                    else:
                        iket.range_push("swapab_fc2_task_pt")

            expert_idx = work_tile_info.expert_idx
            if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                fc1_act_weight_dequant_scale = Float32(1.0)
                fc2_act_dequant_scale = Float32(1.0)
                fc2_act_weight_dequant_scale = Float32(1.0)
            elif cutlass.const_expr(
                self.fp8_scale_mode == "mxfp4_hybrid"
            ):
                # Activation scales were applied in the hybrid mainloop.
                # Only Humming's per-expert residual remains for epilogue.
                fc1_act_weight_dequant_scale = Float32(
                    fc1_weight_dequant_scale[expert_idx]
                )
                fc2_act_dequant_scale = Float32(1.0)
                fc2_act_weight_dequant_scale = Float32(
                    fc2_weight_dequant_scale[expert_idx]
                )
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
                if cutlass.const_expr(self._wgmma_fragment_count == 1):
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
                        store_stage_idx=(
                            warpgroup_idx if self._pingpong else 0
                        ),
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        smem_fc1_amax=smem_fc1_amax,
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
                        storage_group_idx=(
                            warpgroup_idx if self._pingpong else 0
                        ),
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
                    self._store_fc1_task_tile(
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        local_warp_idx=local_warp_idx,
                        n_half=0,
                        store_stage_idx=(
                            warpgroup_idx if self._pingpong else 0
                        ),
                        storage_group_idx=(
                            warpgroup_idx if self._pingpong else 0
                        ),
                        _iket_active=_iket_active,
                    )
                elif n_half == cutlass.Int32(0):
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
                        store_stage_idx=cutlass.Int32(0),
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        smem_fc1_amax=smem_fc1_amax,
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
                        storage_group_idx=(
                            warpgroup_idx if self._pingpong else 0
                        ),
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
                    self._store_fc1_task_tile(
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        local_warp_idx=local_warp_idx,
                        n_half=0,
                        store_stage_idx=cutlass.Int32(0),
                        storage_group_idx=(
                            warpgroup_idx if self._pingpong else 0
                        ),
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
                        store_stage_idx=cutlass.Int32(1),
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        smem_fc1_amax=smem_fc1_amax,
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
                        storage_group_idx=(
                            warpgroup_idx if self._pingpong else 1
                        ),
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
                    self._store_fc1_task_tile(
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        local_warp_idx=local_warp_idx,
                        n_half=1,
                        store_stage_idx=cutlass.Int32(1),
                        storage_group_idx=(
                            warpgroup_idx if self._pingpong else 1
                        ),
                        _iket_active=_iket_active,
                    )
            else:
                if cutlass.const_expr(self._wgmma_fragment_count == 1):
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
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        valid_hidden=valid_hidden,
                        tidx=tidx,
                        fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                        token_comm_args=token_comm_args,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
                elif n_half == cutlass.Int32(0):
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
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        valid_hidden=valid_hidden,
                        tidx=tidx,
                        fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                        token_comm_args=token_comm_args,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
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
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        valid_hidden=valid_hidden,
                        tidx=tidx,
                        fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                        token_comm_args=token_comm_args,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
            # Matches the selected swapab_fc1/fc2_task_* range above.
            if _iket_active:
                iket.range_pop()

            cur_was_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            # The swap-AB token axis is GEMM N. Keep one FC1-done slot per CTA
            # token tile because cluster peers produce independent workspace
            # rows and do not form a cooperative two-CTA MMA.
            cur_fc1_counter_slot = (
                cutlass.Int32(self._cluster_n)
                * work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_n_idx
            )
            cur_fc2_expert_idx = work_tile_info.expert_idx

            if cutlass.const_expr(self._pingpong_order):
                # Match CUTLASS ping-pong ordering: retire the current FC1
                # bulk-store group and hand epilogue ownership to the peer WG
                # before waiting for the descriptor two positions ahead.
                if _iket_active:
                    if cur_was_linear1:
                        iket.range_push("pp_swap_fc1_retire_handoff")
                    else:
                        iket.range_push("pp_swap_fc2_retire_handoff")
                if cur_was_linear1:
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.fence_acq_rel_gpu()
                    flag_tracker = flag_tracker.accumulate(
                        work_tile_info.phase,
                        1,
                        (
                            gmem_fc1_done_counter.iterator
                            + cur_fc1_counter_slot
                        ).toint(),
                    )
                else:
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
                iket.range_push("swapab_sched_consume_next")
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

            # Drain fc1 TMA/STG stores before publishing the fc1-done counter.
            if cur_was_linear1 and cutlass.const_expr(
                not self._pingpong_order
            ):
                if _iket_active:
                    iket.range_push("swapab_fc1_store_drain")
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.fence_acq_rel_gpu()
                if _iket_active:
                    iket.range_pop()

            if _iket_active:
                iket.range_push("swapab_task_boundary_barrier")
            if cutlass.const_expr(not self._pingpong):
                task_tile_boundary_bar.arrive_and_wait()
            if _iket_active:
                iket.range_pop()

            if cur_was_linear1 and cutlass.const_expr(
                not self._pingpong_order
            ):
                if _iket_active:
                    iket.range_push("swapab_fc1_done_flag_accumulate")
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc1_batch,
                    (gmem_fc1_done_counter.iterator + cur_fc1_counter_slot).toint(),
                )
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
                    iket.range_push("swapab_fc2_done_flag_accumulate")
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc2_batch,
                    fc2_flag_addr,
                    no_fire,
                )
                if _iket_active:
                    iket.range_pop()

        if _iket_active:
            iket.range_push("swapab_done_flag_fire_tail")
        flag_tracker.fire()
        if _iket_active:
            iket.range_pop()

    @cute.jit
    def run_fc1_only(
        self,
        sched_consumer,
        sched_ext,
        smem_fc1_output_buffer: cute.Tensor,
        smem_fc1_amax: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        gmem_fc1_done_counter: cute.Tensor,
        local_warp_idx: int,
        tidx,
        fc1_activation_dequant_scale: Optional[cute.Tensor],
        fc1_weight_dequant_scale: cute.Tensor,
        output_activation_dequant_scale: Optional[cute.Tensor],
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
        _iket_active,
        n_half,
        publish_fc1_done: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Run only FC1/SwiGLU/quantize with no FC2 descriptor ABI."""
        task_tile_boundary_bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=WarpThreadCount * len(self._epilogue_warp_ids),
        )
        work_tile_info = sched_consumer.consume_work()
        # Split FC1 is intentionally non-pingpong.  The shared half-tile
        # helper still carries the pingpong order-state ABI, so seed inert
        # states from the regular AB consumer state.
        math_wg_order_state = ab_consumer_state.clone()
        epi_wg_order_state = ab_consumer_state.clone()
        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(BlockPhase.Linear1),
            tid=tidx % (len(self._epilogue_warp_ids) * WarpThreadCount),
        )
        if cutlass.const_expr(self._fc1_store_overlap_style == "pr360"):
            store_time_stage = cutlass.Int32(0)
            pending_fc1_flag_addr = Int64(0)
        while work_tile_info.is_valid_tile:
            expert_idx = work_tile_info.expert_idx
            if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                if _iket_active:
                    iket.range_push("swapab_fc1_task_bw")
                fc1_act_weight_dequant_scale = Float32(1.0)
                fc2_act_dequant_scale = Float32(1.0)
            elif cutlass.const_expr(self.fp8_scale_mode == "mxfp4_hybrid"):
                if _iket_active:
                    iket.range_push("swapab_fc1_task_mxfp4_hybrid")
                fc1_act_weight_dequant_scale = Float32(
                    fc1_weight_dequant_scale[expert_idx]
                )
                fc2_act_dequant_scale = Float32(1.0)
            else:
                if _iket_active:
                    iket.range_push("swapab_fc1_task_pt")
                fc1_act_weight_dequant_scale = (
                    Float32(fc1_activation_dequant_scale[0])
                    * Float32(fc1_weight_dequant_scale[expert_idx])
                )
                fc2_act_dequant_scale = Float32(
                    output_activation_dequant_scale[0]
                )

            store_stage_idx = cutlass.Int32(n_half)
            if cutlass.const_expr(self._fc1_store_overlap_style == "pr360"):
                store_stage_idx = (
                    store_time_stage * cutlass.Int32(self._epilogue_warpgroup_count)
                    + cutlass.Int32(n_half)
                )
            if cutlass.const_expr(self._epilogue_warpgroup_count == 1):
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
                    store_stage_idx=store_stage_idx,
                    weight_sf_pipeline=weight_sf_pipeline,
                    ab_consumer_state=ab_consumer_state,
                    k_tile_cnt_fc1=k_tile_cnt_fc1,
                    k_tile_cnt_fc2=cutlass.Int32(0),
                    _iket_active=_iket_active,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    smem_fc1_amax=smem_fc1_amax,
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
                    storage_group_idx=0,
                    math_wg_order_barrier=ab_pipeline,
                    math_wg_order_state=math_wg_order_state,
                    epi_wg_order_barrier=ab_pipeline,
                    epi_wg_order_state=epi_wg_order_state,
                )
                self._store_fc1_task_tile(
                    work_tile_info=work_tile_info,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    tma_atom_fc1_output=tma_atom_fc1_output,
                    sched_ext=sched_ext,
                    gmem_fc1_output=gmem_fc1_output,
                    local_warp_idx=local_warp_idx,
                    n_half=0,
                    store_stage_idx=store_stage_idx,
                    storage_group_idx=0,
                    _iket_active=_iket_active,
                )
            elif n_half == cutlass.Int32(0):
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
                    store_stage_idx=store_stage_idx,
                    weight_sf_pipeline=weight_sf_pipeline,
                    ab_consumer_state=ab_consumer_state,
                    k_tile_cnt_fc1=k_tile_cnt_fc1,
                    k_tile_cnt_fc2=cutlass.Int32(0),
                    _iket_active=_iket_active,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    smem_fc1_amax=smem_fc1_amax,
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
                    storage_group_idx=0,
                    math_wg_order_barrier=ab_pipeline,
                    math_wg_order_state=math_wg_order_state,
                    epi_wg_order_barrier=ab_pipeline,
                    epi_wg_order_state=epi_wg_order_state,
                )
                self._store_fc1_task_tile(
                    work_tile_info=work_tile_info,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    tma_atom_fc1_output=tma_atom_fc1_output,
                    sched_ext=sched_ext,
                    gmem_fc1_output=gmem_fc1_output,
                    local_warp_idx=local_warp_idx,
                    n_half=0,
                    store_stage_idx=store_stage_idx,
                    storage_group_idx=0,
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
                    store_stage_idx=store_stage_idx,
                    weight_sf_pipeline=weight_sf_pipeline,
                    ab_consumer_state=ab_consumer_state,
                    k_tile_cnt_fc1=k_tile_cnt_fc1,
                    k_tile_cnt_fc2=cutlass.Int32(0),
                    _iket_active=_iket_active,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    smem_fc1_amax=smem_fc1_amax,
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
                    storage_group_idx=1,
                    math_wg_order_barrier=ab_pipeline,
                    math_wg_order_state=math_wg_order_state,
                    epi_wg_order_barrier=ab_pipeline,
                    epi_wg_order_state=epi_wg_order_state,
                )
                self._store_fc1_task_tile(
                    work_tile_info=work_tile_info,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    tma_atom_fc1_output=tma_atom_fc1_output,
                    sched_ext=sched_ext,
                    gmem_fc1_output=gmem_fc1_output,
                    local_warp_idx=local_warp_idx,
                    n_half=1,
                    store_stage_idx=store_stage_idx,
                    storage_group_idx=1,
                    _iket_active=_iket_active,
                )

            if _iket_active:
                iket.range_pop()

            # Preserve fused store ordering. Publication remains disabled
            # for P0/P1 and is enabled only by split-Mega K1.
            if cutlass.const_expr(self._fc1_store_overlap_style == "pr357"):
                counter_tile_n = (
                    self._split_handoff_token_n
                    if self._split_handoff_token_n is not None
                    else self._token_tile_n
                )
                cur_fc1_counter_slot = (
                    work_tile_info.cumulative_data_physical_row
                    // cutlass.Int32(counter_tile_n)
                    + (
                        work_tile_info.tile_n_idx
                        * cutlass.Int32(self._token_tile_n)
                    )
                    // cutlass.Int32(counter_tile_n)
                )
            elif cutlass.const_expr(self._split_handoff_token_n is not None):
                cur_fc1_counter_slot = (
                    work_tile_info.cumulative_data_physical_row
                    // cutlass.Int32(self._split_handoff_token_n)
                    + (
                        work_tile_info.tile_n_idx
                        * cutlass.Int32(self._token_tile_n)
                    )
                    // cutlass.Int32(self._split_handoff_token_n)
                )
            else:
                cur_fc1_counter_slot = (
                    work_tile_info.cumulative_token_block_count
                    * cutlass.Int32(self._cluster_token_ctas)
                    + work_tile_info.tile_n_idx
                )
            cur_fc1_flag_addr = (
                gmem_fc1_done_counter.iterator
                + cur_fc1_counter_slot
            ).toint()
            work_tile_info = sched_consumer.consume_work()
            if cutlass.const_expr(self._fc1_store_overlap_style == "pr360"):
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(1, read=True)
                cute.arch.fence_acq_rel_gpu()
                task_tile_boundary_bar.arrive_and_wait()
                if cutlass.const_expr(publish_fc1_done):
                    if pending_fc1_flag_addr != Int64(0):
                        flag_tracker = flag_tracker.accumulate(
                            cutlass.Int32(BlockPhase.Linear1),
                            self._fc1_handoff_publish_batch,
                            pending_fc1_flag_addr,
                        )
                pending_fc1_flag_addr = cur_fc1_flag_addr
                store_time_stage = cutlass.Int32(1) - store_time_stage
            else:
                if cutlass.const_expr(
                    self._fc1_store_overlap_style != "pr357"
                ):
                    cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.fence_acq_rel_gpu()
                task_tile_boundary_bar.arrive_and_wait()
                if cutlass.const_expr(publish_fc1_done):
                    flag_tracker = flag_tracker.accumulate(
                        work_tile_info.phase,
                        self._fc1_handoff_publish_batch,
                        cur_fc1_flag_addr,
                    )

        if cutlass.const_expr(self._fc1_store_overlap_style == "pr360"):
            if pending_fc1_flag_addr != Int64(0):
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.fence_acq_rel_gpu()
                task_tile_boundary_bar.arrive_and_wait()
                if cutlass.const_expr(publish_fc1_done):
                    flag_tracker = flag_tracker.accumulate(
                        cutlass.Int32(BlockPhase.Linear1),
                        self._fc1_handoff_publish_batch,
                        pending_fc1_flag_addr,
                    )


        if cutlass.const_expr(publish_fc1_done):
            flag_tracker.fire()

    @cute.jit
    def run_fc2_only(
        self,
        sched_consumer,
        sched_ext,
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        gmem_fc2_output: cute.Tensor,
        local_warp_idx: int,
        tidx,
        fc2_activation_dequant_scale: Optional[cute.Tensor],
        fc2_weight_dequant_scale: cute.Tensor,
        tiled_mma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        run_wgmma_task_tile: cutlass.Constexpr,
        ab_pipeline,
        weight_sf_pipeline,
        ab_consumer_state,
        k_tile_cnt_fc2,
        _iket_active,
        n_half,
        token_comm_args=None,
    ) -> None:
        """Run only FC2/combine with no FC1 model or epilogue ABI."""
        task_tile_boundary_bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=WarpThreadCount * len(self._epilogue_warp_ids),
        )
        work_tile_info = sched_consumer.consume_work()
        # Split FC2 is intentionally non-pingpong; keep the shared helper ABI
        # with inert order-state values.
        math_wg_order_state = ab_consumer_state.clone()
        epi_wg_order_state = ab_consumer_state.clone()
        while work_tile_info.is_valid_tile:
            expert_idx = work_tile_info.expert_idx
            if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                if _iket_active:
                    iket.range_push("swapab_fc2_task_bw")
                fc2_act_weight_dequant_scale = Float32(1.0)
            elif cutlass.const_expr(self.fp8_scale_mode == "mxfp4_hybrid"):
                if _iket_active:
                    iket.range_push("swapab_fc2_task_mxfp4_hybrid")
                fc2_act_weight_dequant_scale = Float32(
                    fc2_weight_dequant_scale[expert_idx]
                )
            else:
                if _iket_active:
                    iket.range_push("swapab_fc2_task_pt")
                fc2_act_weight_dequant_scale = (
                    Float32(fc2_activation_dequant_scale[0])
                    * Float32(fc2_weight_dequant_scale[expert_idx])
                )

            valid_hidden = cutlass.Int32(gmem_fc2_output.shape[1])
            if cutlass.const_expr(self._epilogue_warpgroup_count == 1):
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
                    k_tile_cnt_fc1=cutlass.Int32(0),
                    k_tile_cnt_fc2=k_tile_cnt_fc2,
                    _iket_active=_iket_active,
                    sched_ext=sched_ext,
                    gmem_fc2_output=gmem_fc2_output,
                    smem_activation_sf=smem_activation_sf,
                    smem_weight_sf=smem_weight_sf,
                    valid_hidden=valid_hidden,
                    tidx=tidx,
                    fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                    run_wgmma_task_tile=run_wgmma_task_tile,
                    math_wg_order_barrier=ab_pipeline,
                    math_wg_order_state=math_wg_order_state,
                    epi_wg_order_barrier=ab_pipeline,
                    epi_wg_order_state=epi_wg_order_state,
                    token_comm_args=token_comm_args,
                )
            elif n_half == cutlass.Int32(0):
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
                    k_tile_cnt_fc1=cutlass.Int32(0),
                    k_tile_cnt_fc2=k_tile_cnt_fc2,
                    _iket_active=_iket_active,
                    sched_ext=sched_ext,
                    gmem_fc2_output=gmem_fc2_output,
                    smem_activation_sf=smem_activation_sf,
                    smem_weight_sf=smem_weight_sf,
                    valid_hidden=valid_hidden,
                    tidx=tidx,
                    fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                    run_wgmma_task_tile=run_wgmma_task_tile,
                    math_wg_order_barrier=ab_pipeline,
                    math_wg_order_state=math_wg_order_state,
                    epi_wg_order_barrier=ab_pipeline,
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
                    k_tile_cnt_fc1=cutlass.Int32(0),
                    k_tile_cnt_fc2=k_tile_cnt_fc2,
                    _iket_active=_iket_active,
                    sched_ext=sched_ext,
                    gmem_fc2_output=gmem_fc2_output,
                    smem_activation_sf=smem_activation_sf,
                    smem_weight_sf=smem_weight_sf,
                    valid_hidden=valid_hidden,
                    tidx=tidx,
                    fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                    run_wgmma_task_tile=run_wgmma_task_tile,
                    math_wg_order_barrier=ab_pipeline,
                    math_wg_order_state=math_wg_order_state,
                    epi_wg_order_barrier=ab_pipeline,
                    epi_wg_order_state=epi_wg_order_state,
                    token_comm_args=token_comm_args,
                )

            if _iket_active:
                iket.range_pop()
            work_tile_info = sched_consumer.consume_work()
            task_tile_boundary_bar.arrive_and_wait()
