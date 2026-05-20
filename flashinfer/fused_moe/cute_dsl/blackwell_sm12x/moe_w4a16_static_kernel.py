"""
MoEStaticKernel - static-scheduled routed W4A16 MoE kernel for SM120/SM121.

Ported from the b12x kernel library to FlashInfer.

This is the BF16-activation counterpart to the compact W4A4 static kernel. It
keeps the same resident route/pack -> compute control plane, but stores routed
activations and activated intermediates as BF16 while retaining packed FP4
weights and E4M3 block scales.

Execution model
  Phase 0: cooperative init / clear row counts
  Phase 1: walk routed (token, topk_slot) pairs, append rows per expert,
           write token_map + token_weights, and copy each routed BF16 token row
           into expert-major scratch
  Barrier: resident-grid barrier after all expert rows are finalized
  Phase 2: run BF16 FC1 -> activation -> BF16 FC2 -> scatter over the finalized
           expert-major scratch

Activation modes
  SiLU (gated):
    FC1:     A x gate^T, A x up^T
    Act:     SiLU(gate) * up
  ReLU2 (non-gated):
    FC1:     A x W1^T
    Act:     max(0, x)^2

Design boundary
  This file owns only the compact static W4A16 backend. Dispatch, workspace
  sizing, and the W4A4/W4A16 precision switch live in moe_dispatch.py.
"""

from __future__ import annotations

from typing import Any, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

from cutlass.cutlass_dsl import (
    Int32,
    Int64,
    Uint32,
    Integer,
    T,
    dsl_user_op,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync

from flashinfer.cute_dsl.fp4_common import (
    atomic_add_global_i32,
    atomic_cas_global_i32,
    cvt_e4m3_to_f32_via_f16,
    f16x2_to_f32x2,
    fmax_f32,
    fp4_decode_4bytes,
    get_ptr_as_int64,
    ld_global_acquire_i32,
    ld_global_nc_u32,
    ld_shared_f32,
    ld_shared_i32_relaxed,
    scatter_add_bf16x2,
    shared_ptr_to_u32,
    spin_wait_global_eq_i32,
    st_global_f32,
    st_global_i32,
    st_global_release_i32,
    st_shared_f32,
    st_shared_i32,
    threadfence,
)
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
    Sm120B12xBlockScaledDenseGemmKernel as DenseGemmKernel,
)


_SF_VEC_SIZE = 16
_COMPACT_STATIC_TILE_M = 128


@dsl_user_op
def _ld_global_nc_u8(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _MoEStaticKernelBase:
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        exact_mma_m_tiles: bool = False,
        fast_math: bool = False,
        activation: str = "silu",
        dynamic_down_scale: bool = False,
    ):
        if activation not in {"silu", "relu2"}:
            raise ValueError(f"unsupported activation {activation!r}")
        self._dense_cls = DenseGemmKernel
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.exact_mma_m_tiles = exact_mma_m_tiles
        self.fast_math = fast_math
        self.activation = activation
        self.is_gated = activation == "silu"
        self.dynamic_down_scale = dynamic_down_scale
        # The FC1 N tile is reused as the FC2 K tile after activation. Keep the
        # mainloop K tile equal to N so the BF16 intermediate fits the same sA
        # tile that phase 2 consumes.
        tile_k = mma_tiler_mn[1]
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        a_tile_m = (
            mma_tiler_mn[0] if mma_tiler_mn[1] >= 128 else max(128, mma_tiler_mn[0])
        )
        self.sa_tile_shape_mk = (a_tile_m, tile_k)
        self.sa_tiles_per_block = self.sa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.output_tile_count_n = output_tile_count_n
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.occupancy = 1
        self.num_mma_warps = 4
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.pass_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )
        self.fc1_stage_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_cta,
        )
        self.load_register_requirement = 32
        self.mma_register_requirement = 232
        self.a_dtype: Any = None
        self.b_dtype: Any = None
        self.a_layout: Any = None
        self.b_layout: Any = None
        self.c_layout: Any = None

    def _make_a_smem_layout(self, ab_stage: int):
        a_is_k_major = self.a_layout.is_k_major_a()
        a_major_mode_size = self.sa_tile_shape_mk[1 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.a_layout,
                self.a_dtype,
                a_major_mode_size,
            ),
            self.a_dtype,
        )
        return cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(self.sa_tile_shape_mk, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

    def _make_b_smem_layout(self, b_dtype, b_layout, ab_stage: int):
        b_smem_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        b_is_k_major = b_layout.is_k_major_b()
        b_major_mode_size = self.tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        return cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

    def _make_staged_layouts(self, ab_stage: int):
        a_smem_staged = self._make_a_smem_layout(ab_stage)
        b_smem_staged = self._make_b_smem_layout(self.b_dtype, self.b_layout, ab_stage)
        epi_smem_staged = sm90_utils.make_smem_layout_epi(
            cutlass.BFloat16,
            self.c_layout,
            self.epi_tile,
            self.epi_stage,
        )
        return a_smem_staged, b_smem_staged, epi_smem_staged

    def _shared_storage_size_bytes(
        self,
        a_smem_staged,
        b_smem_staged,
        epi_smem_staged,
    ) -> int:
        def _align_up(value: int, align: int) -> int:
            return ((value + align - 1) // align) * align

        pipeline_count = 3 if self.is_gated else 2
        offset = (
            3 * 4
            + pipeline_count * (self.ab_stage * 2 * 8)
            + _COMPACT_STATIC_TILE_M * 4
            + _COMPACT_STATIC_TILE_M * 4
        )
        buffers = [
            cute.size_in_bytes(self.a_dtype, a_smem_staged),
            cute.size_in_bytes(self.b_dtype, b_smem_staged),
            cute.size_in_bytes(cutlass.BFloat16, epi_smem_staged),
        ]
        if self.is_gated:
            buffers.insert(2, cute.size_in_bytes(self.b_dtype, b_smem_staged))
        offset = _align_up(offset, self.buffer_align_bytes)
        for idx, size in enumerate(buffers):
            offset += size
            if idx + 1 != len(buffers):
                offset = _align_up(offset, self.buffer_align_bytes)
        return offset

    def _setup_attributes(self):
        self.mma_inst_mnk = (16, 8, 16)
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        atom_layout = cute.make_layout((2, 2, 1))
        permutation_mnk = (
            2 * self.mma_inst_mnk[0],
            2 * self.mma_inst_mnk[1] * 2,
            self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_k_blocks = self.tile_shape_mnk[2] // self.mma_inst_mnk[2]
        epi_stage_max = (self.tile_shape_mnk[1] // self.epi_tile[1]) * (
            self.tile_shape_mnk[0] // self.epi_tile[0]
        )
        self.epi_stage = min(epi_stage_max, 4)
        # Static W4A16 is latency-bound by inline FP4 decode and TMA staging.
        # Use two stages when shared memory permits so A-side TMA and phase-2
        # buffer turnover can overlap the MMA/scatter tail.
        self.ab_stage = 2

        while True:
            (
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.epi_smem_layout_staged,
            ) = self._make_staged_layouts(self.ab_stage)
            if (
                self._shared_storage_size_bytes(
                    self.a_smem_layout_staged,
                    self.b_smem_layout_staged,
                    self.epi_smem_layout_staged,
                )
                <= self.smem_capacity
                or self.ab_stage <= 1
            ):
                break
            self.ab_stage -= 1
            while self.ab_stage > 1 and 32 % self.ab_stage != 0:
                self.ab_stage -= 1


@cute.jit
def _compact_static_get_work_tile(
    row_counts: cute.Tensor,
    active_expert_count: cute.Tensor,
    *,
    tile_m: Int32,
    num_tiles_n: Int32,
    cluster_shape_mn: Tuple[Int32, Int32],
    current_work_linear_idx: Int32,
    current_local_expert_idx: Int32,
    accum_tile_m: Int32,
    cta_id_in_cluster: cute.Coord,
) -> Tuple[Tuple[Int32, Int32, Int32], Integer, Int32, Int32]:
    num_active_experts = active_expert_count[Int32(0)]
    scan_local_expert_idx = current_local_expert_idx
    tile_m_minus_one = tile_m - Int32(1)

    while scan_local_expert_idx < num_active_experts:
        batch_rows = row_counts[scan_local_expert_idx]
        batch_m_tiles = (batch_rows + tile_m_minus_one) // tile_m
        if (accum_tile_m + batch_m_tiles) * num_tiles_n > current_work_linear_idx:
            current_local_expert_idx = scan_local_expert_idx
            scan_local_expert_idx = num_active_experts
        else:
            accum_tile_m += batch_m_tiles
            scan_local_expert_idx += Int32(1)
            current_local_expert_idx = scan_local_expert_idx

    is_valid = current_local_expert_idx < num_active_experts
    if is_valid:
        batch_rows = row_counts[current_local_expert_idx]
        is_valid = (
            accum_tile_m + (batch_rows + tile_m_minus_one) // tile_m
        ) * num_tiles_n > current_work_linear_idx

    cur_cluster_coord = (
        current_work_linear_idx // num_tiles_n - accum_tile_m,
        current_work_linear_idx % num_tiles_n,
        current_local_expert_idx,
    )
    cur_tile_coord = (
        Int32(cur_cluster_coord[0]) * cluster_shape_mn[0] + cta_id_in_cluster[0],
        Int32(cur_cluster_coord[1]) * cluster_shape_mn[1] + cta_id_in_cluster[1],
        Int32(cur_cluster_coord[2]),
    )
    return cur_tile_coord, is_valid, current_local_expert_idx, accum_tile_m


def _compact_unique_get_work_tile(
    *,
    num_active_experts: Int32,
    num_tiles_n: Int32,
    current_work_linear_idx: Int32,
    cta_id_in_cluster: cute.Coord,
) -> Tuple[Tuple[Int32, Int32, Int32], Integer]:
    local_expert_idx = current_work_linear_idx // num_tiles_n
    cur_tile_coord = (
        cta_id_in_cluster[0],
        current_work_linear_idx % num_tiles_n,
        local_expert_idx,
    )
    return cur_tile_coord, local_expert_idx < num_active_experts


class MoEStaticKernel(_MoEStaticKernelBase):
    """Compact static MoE kernel for small routed working sets."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        exact_mma_m_tiles: bool = False,
        fast_math: bool = False,
        activation: str = "silu",
        single_token: bool = False,
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        dynamic_down_scale: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            exact_mma_m_tiles=exact_mma_m_tiles,
            fast_math=fast_math,
            activation=activation,
            dynamic_down_scale=dynamic_down_scale,
        )
        self.single_token = single_token
        self.share_input_across_experts = share_input_across_experts
        self.share_expert_scales = share_expert_scales

    @cute.jit
    def _resident_grid_barrier(
        self,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        grid_x: Int32,
        is_cta_leader: Int32,
    ):
        cute.arch.sync_threads()
        threadfence()
        if is_cta_leader > Int32(0):
            barrier_count_addr = get_ptr_as_int64(barrier_count, Int32(0))
            barrier_epoch_addr = get_ptr_as_int64(barrier_epoch, Int32(0))
            old_epoch = ld_global_acquire_i32(barrier_epoch_addr)
            arrived = atomic_add_global_i32(barrier_count_addr, Int32(1))
            if arrived == grid_x - Int32(1):
                st_global_i32(barrier_count_addr, Int32(0))
                st_global_release_i32(barrier_epoch_addr, old_epoch + Int32(1))
            else:
                spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)
        cute.arch.sync_threads()

    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,  # [num_tokens, K] bf16
        topk_ids: cute.Tensor,  # [num_tokens * topk] int32
        topk_weights: cute.Tensor,  # [num_tokens * topk] float32
        packed_a: cute.Tensor,  # [max_rows, K, E] BF16 routed activation workspace
        sfa_ptr: cute.Pointer,  # kept for API compatibility; unused for W4A16
        packed_a_storage: cute.Tensor,  # kept for API compatibility; unused for W4A16
        scale_storage: cute.Tensor,  # kept for API compatibility; unused for W4A16
        barrier_count: cute.Tensor,  # [1] int32 (host-zeroed)
        barrier_epoch: cute.Tensor,  # [1] int32 (host-zeroed)
        b_w13: cute.Tensor,  # [w1_n, K, E] — gated packs [up, gate], relu2 is single FC1
        sfb_w13_ptr: cute.Pointer,  # scale factors for FC1 weights
        b_down: cute.Tensor,  # [K, I_tp, E]
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,  # [state_E] routed rows per local expert
        active_expert_count: cute.Tensor,  # [1] active expert count
        weight_expert_ids: cute.Tensor,  # [E] local expert id -> global weight expert id
        global_to_local_expert: cute.Tensor,  # [weight_E] global expert id -> local expert id
        input_global_scale: cute.Tensor,  # kept for API compatibility; unused for W4A16
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,  # kept for API compatibility; unused for W4A16
        scatter_output: cute.Tensor,  # [num_tokens, K]
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = packed_a.element_type
        self.b_dtype = cutlass.BFloat16
        if cutlass.const_expr(self.a_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 routed A scratch, got {self.a_dtype}")
        self.a_layout = utils.LayoutEnum.from_tensor(packed_a)
        self.b_layout = utils.LayoutEnum.from_tensor(b_w13)
        # Compact static always scatters into token-major row-major output.
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        self._setup_attributes()

        # W4A16 only TMA-loads BF16 routed activations. The packed FP4
        # weights and E4M3 scales are decoded by the producer warp directly
        # into BF16 shared-memory B tiles.
        tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
            packed_a,
            self.a_smem_layout_staged,
            self.sa_tile_shape_mk,
            1,
        )

        # Compact static schedules over (m_tile, intermediate_slice, local_expert_idx).
        grid = (*self.cluster_shape_mn, max_active_clusters)
        self.kernel(
            a_input,
            topk_ids,
            topk_weights,
            packed_a,
            barrier_count,
            barrier_epoch,
            tma_a,
            gA,
            b_w13,
            sfb_w13_ptr,
            b_down,
            sfb_down_ptr,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            row_counts,
            active_expert_count,
            weight_expert_ids,
            global_to_local_expert,
            alpha,
            down_alpha,
            scatter_output,
            token_map,
            token_weights,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _load_u8_as_u32(self, base_addr: Int64, byte_offset: Int64) -> Uint32:
        return _ld_global_nc_u8(base_addr + byte_offset)

    @cute.jit
    def _swizzled_e4m3_offset(
        self,
        row: Int32,
        sf_block: Int32,
        sf_cols: Int32,
    ) -> Int64:
        row_rb = row >> Int32(7)
        mode_a = (row >> Int32(5)) & Int32(3)
        mode_32 = row & Int32(31)
        cb_idx = sf_block >> Int32(2)
        mode_c = sf_block & Int32(3)
        return (
            Int64(row_rb) * Int64(sf_cols * Int32(128))
            + Int64(cb_idx) * Int64(512)
            + Int64(mode_32) * Int64(16)
            + Int64(mode_a) * Int64(4)
            + Int64(mode_c)
        )

    @cute.jit
    def _stage_w13_fp4_b_tile(
        self,
        packed_w: cute.Tensor,
        sfb_ptr: cute.Pointer,
        sB: cute.Tensor,
        stage_idx: Int32,
        expert_idx: Int32,
        n_tile_idx: Int32,
        k_tile_idx: Int32,
        weight_rows: Int32,
        weight_cols: Int32,
        sf_cols: Int32,
        copy_start: Int32,
        copy_stride: Int32,
    ):
        w_base = packed_w.iterator.toint()
        sf_base = sfb_ptr.toint()
        packed_cols = weight_cols // Int32(2)
        tile_n = Int32(self.tile_shape_mnk[1])
        tile_k = Int32(self.tile_shape_mnk[2])
        blocks_per_row = tile_k // Int32(self.sf_vec_size)
        total_blocks = tile_n * blocks_per_row
        copy_idx = copy_start
        while copy_idx < total_blocks:
            local_n = copy_idx // blocks_per_row
            local_sf_block = copy_idx - local_n * blocks_per_row
            local_k = local_sf_block * Int32(self.sf_vec_size)
            global_n = n_tile_idx * tile_n + local_n
            global_k = k_tile_idx * tile_k + local_k
            packed_offset = (
                Int64(expert_idx) * Int64(weight_rows * packed_cols)
                + Int64(global_n) * Int64(packed_cols)
                + Int64(global_k // Int32(2))
            )
            scale_offset = Int64(expert_idx) * Int64(
                ((weight_rows + Int32(127)) // Int32(128)) * Int32(128) * sf_cols
            ) + self._swizzled_e4m3_offset(
                global_n,
                global_k // Int32(self.sf_vec_size),
                sf_cols,
            )
            scale_byte = self._load_u8_as_u32(sf_base, scale_offset)
            scale = cvt_e4m3_to_f32_via_f16(scale_byte)
            q_word0 = ld_global_nc_u32(w_base + packed_offset)
            q_word1 = ld_global_nc_u32(w_base + packed_offset + Int64(4))
            d0, d1, d2, d3 = fp4_decode_4bytes(q_word0)
            f0, f1 = f16x2_to_f32x2(d0)
            sB[local_n, local_k, stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(1), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d1)
            sB[local_n, local_k + Int32(2), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(3), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d2)
            sB[local_n, local_k + Int32(4), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(5), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d3)
            sB[local_n, local_k + Int32(6), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(7), stage_idx] = cutlass.BFloat16(f1 * scale)
            d0, d1, d2, d3 = fp4_decode_4bytes(q_word1)
            f0, f1 = f16x2_to_f32x2(d0)
            sB[local_n, local_k + Int32(8), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(9), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d1)
            sB[local_n, local_k + Int32(10), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(11), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d2)
            sB[local_n, local_k + Int32(12), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(13), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d3)
            sB[local_n, local_k + Int32(14), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(15), stage_idx] = cutlass.BFloat16(f1 * scale)
            copy_idx += copy_stride

    @cute.jit
    def _stage_down_fp4_b_tile(
        self,
        packed_w: cute.Tensor,
        sfb_ptr: cute.Pointer,
        sB: cute.Tensor,
        stage_idx: Int32,
        expert_idx: Int32,
        output_tile_idx: Int32,
        intermediate_tile_idx: Int32,
        weight_rows: Int32,
        weight_cols: Int32,
        sf_cols: Int32,
        copy_start: Int32,
        copy_stride: Int32,
    ):
        w_base = packed_w.iterator.toint()
        sf_base = sfb_ptr.toint()
        packed_cols = weight_cols // Int32(2)
        tile_n = Int32(self.tile_shape_mnk[1])
        tile_k = Int32(self.tile_shape_mnk[2])
        blocks_per_row = tile_k // Int32(self.sf_vec_size)
        total_blocks = tile_n * blocks_per_row
        copy_idx = copy_start
        while copy_idx < total_blocks:
            local_n = copy_idx // blocks_per_row
            local_sf_block = copy_idx - local_n * blocks_per_row
            local_k = local_sf_block * Int32(self.sf_vec_size)
            global_n = output_tile_idx * tile_n + local_n
            global_k = intermediate_tile_idx * tile_k + local_k
            packed_offset = (
                Int64(expert_idx) * Int64(weight_rows * packed_cols)
                + Int64(global_n) * Int64(packed_cols)
                + Int64(global_k // Int32(2))
            )
            scale_offset = Int64(expert_idx) * Int64(
                ((weight_rows + Int32(127)) // Int32(128)) * Int32(128) * sf_cols
            ) + self._swizzled_e4m3_offset(
                global_n,
                global_k // Int32(self.sf_vec_size),
                sf_cols,
            )
            scale_byte = self._load_u8_as_u32(sf_base, scale_offset)
            scale = cvt_e4m3_to_f32_via_f16(scale_byte)
            q_word0 = ld_global_nc_u32(w_base + packed_offset)
            q_word1 = ld_global_nc_u32(w_base + packed_offset + Int64(4))
            d0, d1, d2, d3 = fp4_decode_4bytes(q_word0)
            f0, f1 = f16x2_to_f32x2(d0)
            sB[local_n, local_k, stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(1), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d1)
            sB[local_n, local_k + Int32(2), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(3), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d2)
            sB[local_n, local_k + Int32(4), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(5), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d3)
            sB[local_n, local_k + Int32(6), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(7), stage_idx] = cutlass.BFloat16(f1 * scale)
            d0, d1, d2, d3 = fp4_decode_4bytes(q_word1)
            f0, f1 = f16x2_to_f32x2(d0)
            sB[local_n, local_k + Int32(8), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(9), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d1)
            sB[local_n, local_k + Int32(10), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(11), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d2)
            sB[local_n, local_k + Int32(12), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(13), stage_idx] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d3)
            sB[local_n, local_k + Int32(14), stage_idx] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(15), stage_idx] = cutlass.BFloat16(f1 * scale)
            copy_idx += copy_stride

    @cute.kernel
    def kernel(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        routed_a: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        epi_smem_staged: cute.ComposedLayout,
        row_counts: cute.Tensor,
        active_expert_count: cute.Tensor,
        weight_expert_ids: cute.Tensor,
        global_to_local_expert: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
    ):
        """Kernel entry point."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        _, _, gdim_z = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

        a_smem_one = cute.slice_(a_smem_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_one)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class StorageGated:
            ctrl: cute.struct.MemRange[cutlass.Int32, 2]
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            up_pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            phase2_pipeline_array: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            scatter_tok_cache: cute.struct.MemRange[
                cutlass.Int32, _COMPACT_STATIC_TILE_M
            ]
            scatter_weight_cache: cute.struct.MemRange[
                cutlass.Float32, _COMPACT_STATIC_TILE_M
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB_up: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class StorageRelu2:
            ctrl: cute.struct.MemRange[cutlass.Int32, 2]
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            phase2_pipeline_array: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            scatter_tok_cache: cute.struct.MemRange[
                cutlass.Int32, _COMPACT_STATIC_TILE_M
            ]
            scatter_weight_cache: cute.struct.MemRange[
                cutlass.Float32, _COMPACT_STATIC_TILE_M
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]

        storage = smem.allocate(StorageGated if self.is_gated else StorageRelu2)

        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        ml_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=tma_copy_bytes,
            barrier_storage=storage.pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        up_pipeline = (
            pipeline.PipelineTmaAsync.create(
                num_stages=self.ab_stage,
                producer_group=prod_group,
                consumer_group=cons_group,
                tx_count=tma_copy_bytes,
                barrier_storage=storage.up_pipeline_array.data_ptr(),
                cta_layout_vmnk=cta_layout_vmnk,
            )
            if self.is_gated
            else ml_pipeline
        )
        phase2_prod_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_threads_per_warp,
        )
        phase2_cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_mma_warps * self.num_threads_per_warp,
        )
        _phase2_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.ab_stage,
            producer_group=phase2_prod_group,
            consumer_group=phase2_cons_group,
            barrier_storage=storage.phase2_pipeline_array.data_ptr(),
        )

        cute.arch.sync_threads()

        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sB_up = (
            storage.sB_up.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
            if self.is_gated
            else sB
        )
        sC = storage.sC.get_tensor(
            epi_smem_staged.outer,
            swizzle=epi_smem_staged.inner,
        )
        ctrl_base_addr = shared_ptr_to_u32(storage.ctrl.data_ptr())
        scatter_tok_base_addr = shared_ptr_to_u32(storage.scatter_tok_cache.data_ptr())
        scatter_weight_base_addr = shared_ptr_to_u32(
            storage.scatter_weight_cache.data_ptr()
        )

        num_tokens = Int32(a_input.shape[0])
        cols = Int32(a_input.shape[1])
        num_experts = Int32(row_counts.shape[0])
        max_rows = Int32(token_map.shape[1])
        total_pairs = Int32(topk_ids.shape[0])
        num_topk = total_pairs // num_tokens
        num_global_experts = Int32(global_to_local_expert.shape[0])
        flat_tid = Int32(bidz) * Int32(self.threads_per_cta) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(self.threads_per_cta)

        # Phase 0: cooperative init — zero row_counts and scatter_output
        if cutlass.const_expr(not self.single_token):
            i = flat_tid
            while i < num_experts:
                row_counts[i] = Int32(0)
                i += flat_stride
            i = flat_tid
            while i < num_global_experts:
                global_to_local_expert[i] = Int32(-1)
                i += flat_stride
            if flat_tid == Int32(0):
                active_expert_count[Int32(0)] = Int32(0)
        if cutlass.const_expr(self.single_token):
            num_active_experts = total_pairs
        else:
            num_active_experts = active_expert_count[Int32(0)]
        scatter_total = num_tokens * cols
        j = flat_tid
        while j < scatter_total:
            scatter_output[j // cols, j % cols] = cutlass.BFloat16(0.0)
            j += flat_stride
        cute.arch.sync_threads()
        if cutlass.const_expr(not self.share_input_across_experts):
            self._resident_grid_barrier(
                barrier_count,
                barrier_epoch,
                Int32(gdim_z),
                is_cta_leader,
            )

        pair_idx = Int32(bidz)
        while pair_idx < total_pairs:
            token_idx = Int32(0)
            weight = cutlass.Float32(0.0)
            if cutlass.const_expr(not self.single_token):
                expert_id = topk_ids[pair_idx].to(Int32)
                token_idx = pair_idx // num_topk
                weight = topk_weights[pair_idx].to(cutlass.Float32)

            local_expert_id = Int32(0)
            row = Int32(0)
            if cutlass.const_expr(self.single_token):
                local_expert_id = pair_idx
                expert_id = topk_ids[local_expert_id].to(Int32)
            else:
                if is_cta_leader > Int32(0):
                    prior_local_expert_id = atomic_cas_global_i32(
                        get_ptr_as_int64(global_to_local_expert, expert_id),
                        Int32(-1),
                        Int32(-2),
                    )
                    if prior_local_expert_id == Int32(-1):
                        local_expert_id = atomic_add_global_i32(
                            get_ptr_as_int64(active_expert_count, Int32(0)),
                            Int32(1),
                        )
                        weight_expert_ids[local_expert_id] = expert_id
                        st_global_release_i32(
                            get_ptr_as_int64(global_to_local_expert, expert_id),
                            local_expert_id,
                        )
                    else:
                        if prior_local_expert_id == Int32(-2):
                            spin_wait_global_eq_i32(
                                get_ptr_as_int64(global_to_local_expert, expert_id),
                                Int32(-2),
                            )
                            prior_local_expert_id = ld_global_acquire_i32(
                                get_ptr_as_int64(global_to_local_expert, expert_id),
                            )
                        local_expert_id = prior_local_expert_id
                    row = atomic_add_global_i32(
                        get_ptr_as_int64(row_counts, local_expert_id),
                        Int32(1),
                    )
                    map_idx = local_expert_id * max_rows + row
                    st_global_i32(get_ptr_as_int64(token_map, map_idx), token_idx)
                    st_global_f32(get_ptr_as_int64(token_weights, map_idx), weight)
                    st_shared_i32(ctrl_base_addr + Int32(0), local_expert_id)
                    st_shared_i32(ctrl_base_addr + Int32(4), row)
                cute.arch.sync_threads()
                local_expert_id = ld_shared_i32_relaxed(ctrl_base_addr + Int32(0))
                row = ld_shared_i32_relaxed(ctrl_base_addr + Int32(4))

            should_store = Int32(1)
            packed_local_expert_id = local_expert_id
            packed_row = row
            if cutlass.const_expr(self.share_input_across_experts):
                should_store = Int32(1) if pair_idx == Int32(0) else Int32(0)
                packed_local_expert_id = Int32(0)
                packed_row = Int32(0)

            # W4A16 keeps activations in BF16. Scatter routed rows directly
            # into the A workspace and leave input_global_scale unused.
            if should_store > Int32(0):
                col_idx = Int32(tidx)
                while col_idx < cols:
                    routed_a[packed_row, col_idx, packed_local_expert_id] = a_input[
                        token_idx, col_idx
                    ]
                    col_idx += Int32(self.threads_per_cta)

            if cutlass.const_expr(not self.single_token):
                cute.arch.sync_threads()
            pair_idx += Int32(gdim_z)

        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        gA = cute.local_tile(mA, self.sa_tile_shape_mk, (None, None, None))
        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord[1]

        tAsA, tAgA = cpasync.tma_partition(
            tma_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        # MMA fragment partitions
        tCsA_full = thr_mma.partition_A(sA)
        tCrA_full = tiled_mma.make_fragment_A(tCsA_full[None, None, None, 0])
        tCsB = thr_mma.partition_B(sB)
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCsB_up = thr_mma.partition_B(sB_up)
        tCrB_up = tiled_mma.make_fragment_B(tCsB_up[None, None, None, 0])

        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        sub_shape = tCsC_for_shape.shape[:3]
        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
        gate_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        up_acc = (
            cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            if self.is_gated
            else gate_acc
        )

        k_tile_cnt = cute.size(gA, mode=[3])
        fc1_k_tile_cnt = k_tile_cnt
        w13_rows = Int32(b_w13.shape[0])
        # Runtime W4A16 weight tensors are uint8 packed views, so shape[1] is
        # K/2. Manual FP4 byte addressing needs the logical FP4 column count.
        w13_cols = cols
        w13_sf_cols = (
            ((w13_cols // Int32(self.sf_vec_size)) + Int32(3)) // Int32(4) * Int32(4)
        )
        down_rows = Int32(b_down.shape[0])
        down_cols = w13_rows // Int32(2) if self.is_gated else w13_rows
        down_sf_cols = (
            ((down_cols // Int32(self.sf_vec_size)) + Int32(3)) // Int32(4) * Int32(4)
        )
        tile_n = Int32(self.tile_shape_mnk[1])
        intermediate_tile_cnt = (w13_rows + tile_n - Int32(1)) // tile_n
        gate_tile_cnt = (
            intermediate_tile_cnt // Int32(2)
            if self.is_gated
            else intermediate_tile_cnt
        )
        output_tile_cnt = (down_rows + tile_n - Int32(1)) // tile_n

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        up_prod_state = (
            pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            if self.is_gated
            else prod_state
        )
        up_cons_state = (
            pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            if self.is_gated
            else cons_state
        )
        _phase2_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        _phase2_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # ===================================================================
        # MMA WARP GROUP (warps 0-3)
        # ===================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            num_k_blocks = cute.size(tCrA_full, mode=[2])

            atom_ld_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_ld_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_copy_A = cute.make_tiled_copy_A(atom_ld_A, tiled_mma)
            smem_copy_B = cute.make_tiled_copy_B(atom_ld_B, tiled_mma)

            thr_ld_A = smem_copy_A.get_slice(tidx)
            thr_ld_B = smem_copy_B.get_slice(tidx)
            csA_full = thr_ld_A.partition_S(sA)
            crA_full = thr_ld_A.retile(tCrA_full)
            csB = thr_ld_B.partition_S(sB)
            csB_up = thr_ld_B.partition_S(sB_up)
            crB = thr_ld_B.retile(tCrB)
            crB_up = thr_ld_B.retile(tCrB_up)

            num_persistent_clusters = Int32(gdim_z)
            cluster_shape_mn = (
                Int32(self.cluster_shape_mn[0]),
                Int32(self.cluster_shape_mn[1]),
            )
            cta_id_in_cluster = (
                Int32(bidx % cluster_shape_mn[0]),
                Int32(bidy % cluster_shape_mn[1]),
                Int32(0),
            )
            current_work_linear_idx = Int32(bidz)
            current_local_expert_idx = Int32(0)
            accum_tile_m = Int32(0)
            tile_coord = (Int32(0), Int32(0), Int32(0))
            is_valid_tile = Int32(0) < Int32(0)
            if cutlass.const_expr(self.single_token):
                tile_coord, is_valid_tile = _compact_unique_get_work_tile(
                    num_active_experts=num_active_experts,
                    num_tiles_n=Int32(self.output_tile_count_n),
                    current_work_linear_idx=current_work_linear_idx,
                    cta_id_in_cluster=cta_id_in_cluster,
                )
            else:
                tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                    _compact_static_get_work_tile(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_shape_mnk[0]),
                        num_tiles_n=Int32(self.output_tile_count_n),
                        cluster_shape_mn=cluster_shape_mn,
                        current_work_linear_idx=current_work_linear_idx,
                        current_local_expert_idx=current_local_expert_idx,
                        accum_tile_m=accum_tile_m,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )
                )

            while is_valid_tile:
                # tile_coord = (m_tile, intermediate_slice, local_expert_idx)
                local_expert_idx = tile_coord[2]
                if cutlass.const_expr(self.single_token):
                    weight_expert_idx = topk_ids[local_expert_idx].to(Int32)
                    valid_rows = Int32(1)
                else:
                    weight_expert_idx = weight_expert_ids[local_expert_idx]
                    valid_rows = row_counts[local_expert_idx]
                alpha_value = alpha[weight_expert_idx].to(cutlass.Float32)
                tile_m_base = tile_coord[0] * Int32(self.tile_shape_mnk[0])
                intermediate_slice = tile_coord[1]
                sa_tile_offset = tile_coord[0] % self.sa_tiles_per_block
                sa_row_base = sa_tile_offset * Int32(self.tile_shape_mnk[0])
                if cutlass.const_expr(self.sa_tiles_per_block > 1):
                    sA_tile = cute.local_tile(
                        sA,
                        cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (sa_tile_offset, 0, None),
                    )
                    csA_tile = thr_ld_A.partition_S(sA_tile)
                    tCsA_tile = thr_mma.partition_A(sA_tile)
                    tCrA_tile = tiled_mma.make_fragment_A(
                        tCsA_tile[None, None, None, 0]
                    )
                    crA_tile = thr_ld_A.retile(tCrA_tile)
                else:
                    csA_tile = csA_full
                    tCrA_tile = tCrA_full
                    crA_tile = crA_full
                valid_tile_rows = valid_rows - tile_m_base
                if valid_tile_rows > Int32(self.tile_shape_mnk[0]):
                    valid_tile_rows = Int32(self.tile_shape_mnk[0])
                if valid_tile_rows < Int32(0):
                    valid_tile_rows = Int32(0)

                cache_row = Int32(tidx)
                if cutlass.const_expr(not self.single_token):
                    if cache_row < Int32(_COMPACT_STATIC_TILE_M):
                        tok = Int32(0)
                        wv = cutlass.Float32(0.0)
                        if cache_row < valid_tile_rows:
                            tok = token_map[
                                local_expert_idx, tile_m_base + cache_row
                            ].to(Int32)
                            wv = token_weights[
                                local_expert_idx, tile_m_base + cache_row
                            ].to(cutlass.Float32)
                        st_shared_i32(scatter_tok_base_addr + cache_row * Int32(4), tok)
                        st_shared_f32(
                            scatter_weight_base_addr + cache_row * Int32(4), wv
                        )
                self.epilog_sync_barrier.arrive_and_wait()

                _is_m_major = self.c_layout.is_m_major_c()
                # This kernel scatters by scalar-reading sC after the warpgroup
                # store. The BF16 dense fused path can use the TMA-store epilogue
                # atom because it forwards sC directly to TMA; direct scatter
                # must match the existing static scatter layout.
                copy_atom_r2s = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                    cutlass.BFloat16,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, tiled_copy_C_Atom
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rGate = tiled_copy_r2s.retile(gate_acc)
                tRS_rUp = tiled_copy_r2s.retile(up_acc)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                tRS_rD_out = cute.make_rmem_tensor(
                    tRS_rD_layout.shape, cutlass.BFloat16
                )

                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rGate, mode=[2])
                epi_buffer = Int32(0)

                down_alpha_value = down_alpha[weight_expert_idx].to(cutlass.Float32)
                down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
                unique_tok = Int32(0)
                unique_wv = cutlass.Float32(0.0)
                if cutlass.const_expr(self.single_token):
                    unique_tok = local_expert_idx // num_topk
                    unique_wv = topk_weights[local_expert_idx].to(cutlass.Float32)

                epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
                MmaMPerEpiM = self.epi_tile[0] // mma_tile_m
                MmaNPerEpiN = self.epi_tile[1] // mma_tile_n

                # ============================================================
                # PHASE A: FC1 for this slice
                # Gated SwiGLU runs gate then up. relu2 uses only the first pass.
                # ============================================================

                # Gate GEMM (BF16 A x materialized BF16 B)
                gate_acc.fill(0.0)
                cons_state.reset_count()
                for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    peek = ml_pipeline.consumer_try_wait(cons_state)
                    ml_pipeline.consumer_wait(cons_state, peek)
                    self._stage_w13_fp4_b_tile(
                        b_w13,
                        sfb_w13_ptr,
                        sB,
                        cons_state.index,
                        weight_expert_idx,
                        intermediate_slice + gate_tile_cnt
                        if self.is_gated
                        else intermediate_slice,
                        k_tile,
                        w13_rows,
                        w13_cols,
                        w13_sf_cols,
                        Int32(tidx),
                        Int32(self.num_mma_warps * self.num_threads_per_warp),
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    csA_p = csA_tile[None, None, None, cons_state.index]
                    csB_p = csB[None, None, None, cons_state.index]
                    cute.copy(
                        smem_copy_A, csA_p[None, None, 0], crA_tile[None, None, 0]
                    )
                    cute.copy(smem_copy_B, csB_p[None, None, 0], crB[None, None, 0])
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )
                        cute.gemm(
                            tiled_mma,
                            gate_acc,
                            tCrA_tile[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            gate_acc,
                        )
                        if k_next > 0:
                            cute.copy(
                                smem_copy_A,
                                csA_p[None, None, k_next],
                                crA_tile[None, None, k_next],
                            )
                            cute.copy(
                                smem_copy_B,
                                csB_p[None, None, k_next],
                                crB[None, None, k_next],
                            )
                    ml_pipeline.consumer_release(cons_state)
                    cons_state.advance()
                # Drain the FC1 gate/only pass before the DMA warp reuses the
                # gate staging buffers, either for the up pass or FC2 prefetch.
                self.pass_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.is_gated):
                    up_acc.fill(0.0)
                    up_cons_state.reset_count()
                    for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                        peek = up_pipeline.consumer_try_wait(up_cons_state)
                        up_pipeline.consumer_wait(up_cons_state, peek)
                        self._stage_w13_fp4_b_tile(
                            b_w13,
                            sfb_w13_ptr,
                            sB_up,
                            up_cons_state.index,
                            weight_expert_idx,
                            intermediate_slice,
                            k_tile,
                            w13_rows,
                            w13_cols,
                            w13_sf_cols,
                            Int32(tidx),
                            Int32(self.num_mma_warps * self.num_threads_per_warp),
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        csA_p = csA_tile[None, None, None, up_cons_state.index]
                        csB_p = csB_up[None, None, None, up_cons_state.index]
                        cute.copy(
                            smem_copy_A, csA_p[None, None, 0], crA_tile[None, None, 0]
                        )
                        cute.copy(
                            smem_copy_B, csB_p[None, None, 0], crB_up[None, None, 0]
                        )
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_next = (
                                0
                                if k_block_idx + 1 == num_k_blocks
                                else k_block_idx + 1
                            )
                            cute.gemm(
                                tiled_mma,
                                up_acc,
                                tCrA_tile[None, None, k_block_idx],
                                tCrB_up[None, None, k_block_idx],
                                up_acc,
                            )
                            if k_next > 0:
                                cute.copy(
                                    smem_copy_A,
                                    csA_p[None, None, k_next],
                                    crA_tile[None, None, k_next],
                                )
                                cute.copy(
                                    smem_copy_B,
                                    csB_p[None, None, k_next],
                                    crB_up[None, None, k_next],
                                )
                        up_pipeline.consumer_release(up_cons_state)
                        up_cons_state.advance()

                # Activation + BF16 materialization into sA for FC2.
                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    epi_m_valid = (
                        valid_rows
                        - tile_m_base
                        - Int32(epi_m) * Int32(self.epi_tile[0])
                    )
                    epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                    if epi_m_valid > Int32(0):
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                mma_n = mma_n_in_epi
                                tRS_rD_slice = tRS_rD[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                if cutlass.const_expr(self.is_gated):
                                    up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(tRS_rD_slice)
                                    ):
                                        g = alpha_value * gate_slice[elem_idx]
                                        u = alpha_value * up_slice[elem_idx]
                                        sigmoid_g = cute.arch.rcp_approx(
                                            cutlass.Float32(1.0)
                                            + cute.math.exp(
                                                -g, fastmath=self.fast_math
                                            ),
                                        )
                                        tRS_rD_slice[elem_idx] = g * sigmoid_g * u
                                else:
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(tRS_rD_slice)
                                    ):
                                        g = alpha_value * gate_slice[elem_idx]
                                        relu_g = fmax_f32(g, cutlass.Float32(0.0))
                                        tRS_rD_slice[elem_idx] = relu_g * relu_g

                        acc_vec = tRS_rD.load()
                        acc_vec = acc_vec.to(cutlass.BFloat16)
                        tRS_rD_out.store(acc_vec)
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                    epi_rows = epi_m_valid
                    if epi_rows > Int32(self.epi_tile[0]):
                        epi_rows = Int32(self.epi_tile[0])
                    if epi_rows < Int32(0):
                        epi_rows = Int32(0)
                    copy_idx = Int32(tidx)
                    copy_cols = Int32(self.tile_shape_mnk[1])
                    while copy_idx < epi_rows * copy_cols:
                        local_row = copy_idx // copy_cols
                        col = copy_idx - local_row * copy_cols
                        sA[sa_row_base + rows_offset + local_row, col, 0] = sC[
                            local_row, col, epi_buffer
                        ]
                        copy_idx += Int32(
                            self.num_mma_warps * self.num_threads_per_warp
                        )

                cute.arch.fence_proxy("async.shared", space="cta")
                # All MMA warps must finish writing the BF16 intermediate
                # before phase2 hoists sA into registers.
                self.epilog_sync_barrier.arrive_and_wait()

                # ============================================================
                # PHASE B: Sweep ALL FC2 output tiles using cached sA
                # All CTA warps stage each B_down tile cooperatively. The DMA
                # warp is idle during FC2, so let it contribute decode lanes
                # and hold it at an all-thread barrier until sB is reusable.
                # ============================================================
                scatter_N = Int32(scatter_output.shape[1])

                csA_phase2 = csA_tile[None, None, None, 0]

                # Hoist A-side register loads: sA is constant across all
                # FC2 output tiles (activated BF16 intermediate). Load crA
                # for all k-blocks once, reuse for all output tiles.
                cute.copy(
                    smem_copy_A, csA_phase2[None, None, 0], crA_tile[None, None, 0]
                )
                for _kb_pre in cutlass.range_constexpr(num_k_blocks - 1):
                    k_pre = _kb_pre + 1
                    cute.copy(
                        smem_copy_A,
                        csA_phase2[None, None, k_pre],
                        crA_tile[None, None, k_pre],
                    )

                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    self._stage_down_fp4_b_tile(
                        b_down,
                        sfb_down_ptr,
                        sB,
                        Int32(0),
                        weight_expert_idx,
                        output_tile_idx,
                        intermediate_slice,
                        down_rows,
                        down_cols,
                        down_sf_cols,
                        Int32(tidx),
                        Int32(self.threads_per_cta),
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.fc1_stage_sync_barrier.arrive_and_wait()
                    csB_phase2 = csB[None, None, None, 0]

                    # Only load B-side (B_down changes per output tile; A is hoisted)
                    cute.copy(
                        smem_copy_B, csB_phase2[None, None, 0], crB[None, None, 0]
                    )

                    down_acc.fill(0.0)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )
                        cute.gemm(
                            tiled_mma,
                            down_acc,
                            tCrA_tile[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            down_acc,
                        )
                        if k_next > 0:
                            # Only B-side for next k-block (A already in registers)
                            cute.copy(
                                smem_copy_B,
                                csB_phase2[None, None, k_next],
                                crB[None, None, k_next],
                            )

                    # Scatter using precomputed metadata (no redundant gmem loads)
                    tile_n_base_cur = output_tile_idx * Int32(self.tile_shape_mnk[1])
                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_n = mma_n_in_epi
                                mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                tRS_rD_slice = tRS_rD[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                down_epi_acc_slice = down_acc[(None, mma_m, mma_n)]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    tRS_rD_slice[elem_idx] = (
                                        down_alpha_value * down_epi_acc_slice[elem_idx]
                                    )

                        acc_vec = tRS_rD.load()
                        acc_vec = acc_vec.to(cutlass.BFloat16)
                        tRS_rD_out.store(acc_vec)
                        epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        # StMatrix ownership is interleaved across MMA warps for
                        # both 128x128 and underfilled tiles, so the scatter path
                        # must wait for every warp's store before reading sC.
                        self.epilog_sync_barrier.arrive_and_wait()

                        rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                        epi_rows = valid_rows - tile_m_base - rows_offset
                        if epi_rows > Int32(self.epi_tile[0]):
                            epi_rows = Int32(self.epi_tile[0])
                        if epi_rows < Int32(0):
                            epi_rows = Int32(0)

                        tile_pair_cols = Int32(self.tile_shape_mnk[1]) // Int32(2)
                        pair_idx = Int32(tidx)
                        while pair_idx < epi_rows * tile_pair_cols:
                            local_row = pair_idx // tile_pair_cols
                            local_pair_col = pair_idx - local_row * tile_pair_cols
                            global_col = tile_n_base_cur + local_pair_col * Int32(2)
                            cached_row = rows_offset + local_row
                            tok = Int32(0)
                            wv = cutlass.Float32(0.0)
                            if cutlass.const_expr(self.single_token):
                                tok = unique_tok
                                wv = unique_wv
                            else:
                                tok = ld_shared_i32_relaxed(
                                    scatter_tok_base_addr + cached_row * Int32(4)
                                )
                                wv = ld_shared_f32(
                                    scatter_weight_base_addr + cached_row * Int32(4)
                                )
                            sc_v0 = cutlass.Float32(
                                sC[local_row, local_pair_col * Int32(2), epi_buffer]
                            )
                            sc_v1 = cutlass.Float32(
                                sC[
                                    local_row,
                                    local_pair_col * Int32(2) + Int32(1),
                                    epi_buffer,
                                ]
                            )
                            scatter_add_bf16x2(
                                get_ptr_as_int64(
                                    scatter_output, tok * scatter_N + global_col
                                ),
                                wv * sc_v0,
                                wv * sc_v1,
                            )
                            pair_idx += Int32(
                                self.num_mma_warps * self.num_threads_per_warp
                            )

                        # Post-scatter barrier: needed to ensure all warps
                        # finish scatter before next output tile's pipeline ops
                        # (pipeline consumer is collective across all MMA warps).
                        self.epilog_sync_barrier.arrive_and_wait()
                    self.fc1_stage_sync_barrier.arrive_and_wait()

                # Final pass_sync: protect sA from next task's FC1 loads.
                # DMA warp waits here too after finishing all B_down loads.
                self.pass_sync_barrier.arrive_and_wait()

                current_work_linear_idx += num_persistent_clusters
                if cutlass.const_expr(self.single_token):
                    tile_coord, is_valid_tile = _compact_unique_get_work_tile(
                        num_active_experts=num_active_experts,
                        num_tiles_n=Int32(self.output_tile_count_n),
                        current_work_linear_idx=current_work_linear_idx,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )
                else:
                    (
                        tile_coord,
                        is_valid_tile,
                        current_local_expert_idx,
                        accum_tile_m,
                    ) = _compact_static_get_work_tile(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_shape_mnk[0]),
                        num_tiles_n=Int32(self.output_tile_count_n),
                        cluster_shape_mn=cluster_shape_mn,
                        current_work_linear_idx=current_work_linear_idx,
                        current_local_expert_idx=current_local_expert_idx,
                        accum_tile_m=accum_tile_m,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )

        # ===================================================================
        # DMA WARP (warp 4)
        # ===================================================================
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            num_persistent_clusters = Int32(gdim_z)
            cluster_shape_mn = (
                Int32(self.cluster_shape_mn[0]),
                Int32(self.cluster_shape_mn[1]),
            )
            cta_id_in_cluster = (
                Int32(bidx % cluster_shape_mn[0]),
                Int32(bidy % cluster_shape_mn[1]),
                Int32(0),
            )
            current_work_linear_idx = Int32(bidz)
            current_local_expert_idx = Int32(0)
            accum_tile_m = Int32(0)
            tile_coord = (Int32(0), Int32(0), Int32(0))
            is_valid_tile = Int32(0) < Int32(0)
            if cutlass.const_expr(self.single_token):
                tile_coord, is_valid_tile = _compact_unique_get_work_tile(
                    num_active_experts=num_active_experts,
                    num_tiles_n=Int32(self.output_tile_count_n),
                    current_work_linear_idx=current_work_linear_idx,
                    cta_id_in_cluster=cta_id_in_cluster,
                )
            else:
                tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                    _compact_static_get_work_tile(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_shape_mnk[0]),
                        num_tiles_n=Int32(self.output_tile_count_n),
                        cluster_shape_mn=cluster_shape_mn,
                        current_work_linear_idx=current_work_linear_idx,
                        current_local_expert_idx=current_local_expert_idx,
                        accum_tile_m=accum_tile_m,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )
                )

            while is_valid_tile:
                tc = tile_coord
                intermediate_slice = tc[1]
                local_expert_idx = tc[2]
                if cutlass.const_expr(self.single_token):
                    weight_expert_idx = topk_ids[local_expert_idx].to(Int32)
                else:
                    weight_expert_idx = weight_expert_ids[local_expert_idx]
                input_local_expert_idx = (
                    Int32(0)
                    if cutlass.const_expr(self.share_input_across_experts)
                    else local_expert_idx
                )

                sa_tile_coord_m = tc[0] // self.sa_tiles_per_block
                tAgA_mk = tAgA[(None, sa_tile_coord_m, None, input_local_expert_idx)]

                # ---- FC1 gate pass: producer only supplies BF16 A via TMA ----
                prod_state.reset_count()
                for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    ml_pipeline.producer_acquire(prod_state)
                    cute.copy(
                        tma_a,
                        tAgA_mk[(None, k_tile)],
                        tAsA[(None, prod_state.index)],
                        tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                    )
                    ml_pipeline.producer_commit(prod_state)
                    prod_state.advance()

                # Wait for the MMA warps to finish the FC1 gate/only pass
                # before reusing the gate staging buffers.
                self.pass_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.is_gated):
                    # ---- FC1 up pass: producer only supplies BF16 A via TMA ----
                    up_prod_state.reset_count()
                    for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                        up_pipeline.producer_acquire(up_prod_state)
                        cute.copy(
                            tma_a,
                            tAgA_mk[(None, k_tile)],
                            tAsA[(None, up_prod_state.index)],
                            tma_bar_ptr=up_pipeline.producer_get_barrier(up_prod_state),
                        )
                        up_pipeline.producer_commit(up_prod_state)
                        up_prod_state.advance()

                # ---- FC2 B_down pass: join the MMA warps for decode staging ----
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    self._stage_down_fp4_b_tile(
                        b_down,
                        sfb_down_ptr,
                        sB,
                        Int32(0),
                        weight_expert_idx,
                        output_tile_idx,
                        intermediate_slice,
                        down_rows,
                        down_cols,
                        down_sf_cols,
                        Int32(tidx),
                        Int32(self.threads_per_cta),
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.fc1_stage_sync_barrier.arrive_and_wait()
                    self.fc1_stage_sync_barrier.arrive_and_wait()

                # Final pass_sync: match MMA warps' barrier after FC2 sweep.
                # Ensures MMA warps finish scatter before DMA starts next task's FC1.
                self.pass_sync_barrier.arrive_and_wait()

                current_work_linear_idx += num_persistent_clusters
                if cutlass.const_expr(self.single_token):
                    tile_coord, is_valid_tile = _compact_unique_get_work_tile(
                        num_active_experts=num_active_experts,
                        num_tiles_n=Int32(self.output_tile_count_n),
                        current_work_linear_idx=current_work_linear_idx,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )
                else:
                    (
                        tile_coord,
                        is_valid_tile,
                        current_local_expert_idx,
                        accum_tile_m,
                    ) = _compact_static_get_work_tile(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_shape_mnk[0]),
                        num_tiles_n=Int32(self.output_tile_count_n),
                        cluster_shape_mn=cluster_shape_mn,
                        current_work_linear_idx=current_work_linear_idx,
                        current_local_expert_idx=current_local_expert_idx,
                        accum_tile_m=accum_tile_m,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )

            ml_pipeline.producer_tail(prod_state)
            if cutlass.const_expr(self.is_gated):
                up_pipeline.producer_tail(up_prod_state)
        return


__all__ = ["MoEStaticKernel"]
