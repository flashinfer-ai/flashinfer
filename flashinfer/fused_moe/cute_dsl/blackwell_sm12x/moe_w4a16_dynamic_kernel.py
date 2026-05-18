"""
MoEDynamicKernel - queue-driven routed W4A16 MoE kernel for SM120/SM121.

Ported from the b12x kernel library to FlashInfer.

This dynamic fused control-plane kernel keeps the queue-driven routing and
task publication structure from the NVFP4 path, but its math path is W4A16:
BF16 activations, BF16 activated intermediate, packed FP4 weights, and E4M3
weight block scales dequantized into BF16 shared-memory B tiles before MMA.

Execution model
  Phase 0: cooperative init / clear scratch state
  Phase 1: all CTAs start as producers
           - claim routed (token, topk_slot) pairs from pair_head
           - append expert rows
           - write token_map + token_weights
           - copy each routed BF16 token row into expert-major scratch
           - publish one compute task per ready (expert, m_tile, slice_group)
             as soon as a tile is fully written
  Phase 2: CTAs that finish producing become consumers immediately
           - CTA leader pops one ready task into shared ctrl state
           - MMA warps run BF16 FC1 -> activation -> BF16 FC2 -> scatter
           - DMA warp streams the corresponding FC1 / FC2 weights

This is intentionally conservative:
  - still one CTA per SM
  - still the static per-slice microkernel, now executed sequentially for a
    small grouped slice task
  - still one initial resident-grid barrier after init

What changes relative to the static path
  - no global route/pack -> compute barrier
  - no static scheduler in the compute steady state
  - route/pack is warp-private instead of CTA-broadcast
  - compute work is driven by a global append-only ready-task queue

This file is a first implementation pass, not an optimized artifact.
"""

from __future__ import annotations

from typing import Tuple

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
    T,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync

from flashinfer.cute_dsl.fp4_common import (
    atomic_add_global_i32,
    cvt_e4m3_to_f32_via_f16,
    f16x2_to_f32x2,
    fmax_f32,
    fp4_decode_4bytes,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    ld_shared_f32,
    ld_shared_i32_relaxed,
    scatter_add_v4_bf16x2,
    shared_ptr_to_u32,
    st_global_i32,
    st_global_v4_u32,
    st_shared_f32,
)
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
    Sm120B12xBlockScaledDenseGemmKernel as DenseGemmKernel,
)


_TASK_SLICE_CHUNK = 1


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


class DynamicLaunchParams:
    """Minimal runtime launch state shared between host setup and kernel code."""

    def __init__(
        self,
        row_counts: cute.Tensor,
        gate_tile_cnt: Int32,
        *,
        loc=None,
    ):
        self.row_counts = row_counts
        self.gate_tile_cnt = gate_tile_cnt
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.row_counts, self.gate_tile_cnt]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.row_counts, self.gate_tile_cnt],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return DynamicLaunchParams(*(tuple(obj_list)), loc=self._loc)


@dsl_user_op
def _st_shared_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_shared_i32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ld_global_acquire_i32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.acquire.gpu.s32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_global_release_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.global.release.gpu.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _spin_wait_global_eq_i32(addr, expected, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(expected).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred %p0;\n"
        ".reg .s32 %val;\n"
        "spin_loop:\n"
        "  ld.global.acquire.gpu.s32 %val, [$0];\n"
        "  setp.eq.s32 %p0, %val, $1;\n"
        "  @%p0 bra spin_loop;\n"
        "}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _threadfence(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _atomic_cas_global_i32(addr, compare, value, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(compare).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.cas.b32 $0, [$1], $2, $3;",
            "=r,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class MoEDynamicKernel:
    """Queue-driven first-pass dynamic MoE kernel."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        fast_math: bool = False,
        activation: str = "silu",
        dynamic_down_scale: bool = False,
    ):
        if activation not in {"silu", "relu2"}:
            raise ValueError(f"unsupported activation {activation!r}")
        self._dense_cls = DenseGemmKernel
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.fast_math = fast_math
        self.activation = activation
        self.is_gated = activation == "silu"
        self.dynamic_down_scale = dynamic_down_scale
        # The FC1 N tile is reused as the FC2 K tile after activation. Keep the
        # mainloop K tile equal to N so the BF16 intermediate fits the same sA
        # tile that phase 2 consumes.
        tile_k = mma_tiler_mn[1]
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        self.sa_tile_shape_mk = (mma_tiler_mn[0], tile_k)
        self.sa_tiles_per_block = self.sa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.occupancy = 1
        self.num_mma_warps = 4
        self.num_load_warps = 4
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + self.num_load_warps
        ) * self.num_threads_per_warp
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

        offset = 8 * 4 + self.ab_stage * 2 * 8 + 2 * self.tile_shape_mnk[0] * 4
        buffers = [
            cute.size_in_bytes(self.a_dtype, a_smem_staged),
            cute.size_in_bytes(self.b_dtype, b_smem_staged),
            cute.size_in_bytes(self.b_dtype, b_smem_staged),
            cute.size_in_bytes(cutlass.BFloat16, epi_smem_staged),
        ]
        offset = _align_up(offset, self.buffer_align_bytes)
        for idx, size in enumerate(buffers):
            offset += size
            if idx + 1 != len(buffers):
                offset = _align_up(offset, self.buffer_align_bytes)
        return offset

    @staticmethod
    def _compute_bf16_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype,
        b_dtype,
        epi_tile: tuple[int, int],
        c_dtype,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        # The dynamic W4A16 path keeps only one epilogue stage. Using the
        # Hopper helper default here can over-reserve epilogue smem and drive
        # the producer/consumer stage count negative for 128x128 BF16 tiles.
        epi_stage = 1
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        return max(1, ab_stage), epi_stage

    @staticmethod
    def _make_bf16_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype,
        a_layout: cute.Layout,
        b_dtype,
        b_layout: cute.Layout,
        ab_stage: int,
        c_dtype,
        c_layout: cute.Layout,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        import cutlass.utils.hopper_helpers as sm90_utils

        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )
        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    def _setup_attributes(self):
        self.mma_inst_mnk = (16, 8, 16)
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        atom_layout = (2, 2, 1)
        tC = cute.make_layout(atom_layout)
        permutation_mnk = (
            atom_layout[0] * self.mma_inst_mnk[0],
            atom_layout[1] * self.mma_inst_mnk[1] * 2,
            atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            tC,
            permutation_mnk=permutation_mnk,
        )
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_m_tiles = self.tile_shape_mnk[0] // (16 * 4)
        self.num_n_tiles = self.tile_shape_mnk[1] // (8 * 2)
        self.num_k_blocks = self.tile_shape_mnk[2] // self.mma_inst_mnk[2]

        epi_stage_max = (self.tile_shape_mnk[1] // self.epi_tile[1]) * (
            self.tile_shape_mnk[0] // self.epi_tile[0]
        )
        self.epi_stage = min(epi_stage_max, 4)
        # Keep two A/B stages when shared memory allows it so the load/dequant
        # producer can run ahead of the MMA consumer on prefill-sized tasks.
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
                global_n, global_k // Int32(self.sf_vec_size), sf_cols
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
                global_n, global_k // Int32(self.sf_vec_size), sf_cols
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
    def _resident_grid_barrier(
        self,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        grid_x: Int32,
        is_cta_leader: Int32,
    ):
        cute.arch.sync_threads()
        _threadfence()
        if is_cta_leader > Int32(0):
            barrier_count_addr = get_ptr_as_int64(barrier_count, Int32(0))
            barrier_epoch_addr = get_ptr_as_int64(barrier_epoch, Int32(0))
            old_epoch = _ld_global_acquire_i32(barrier_epoch_addr)
            arrived = atomic_add_global_i32(barrier_count_addr, Int32(1))
            if arrived == grid_x - Int32(1):
                st_global_i32(barrier_count_addr, Int32(0))
                _st_global_release_i32(barrier_epoch_addr, old_epoch + Int32(1))
            else:
                _spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)
        cute.arch.sync_threads()

    @cute.jit
    def _publish_ready_tasks(
        self,
        task_tail: cute.Tensor,
        task_ready: cute.Tensor,
        task_expert: cute.Tensor,
        task_m_tile: cute.Tensor,
        task_slice_begin: cute.Tensor,
        task_slice_count: cute.Tensor,
        task_valid_rows: cute.Tensor,
        gate_tile_cnt: Int32,
        slice_chunk: Int32,
        expert_idx: Int32,
        m_tile_idx: Int32,
        valid_rows: Int32,
    ):
        num_groups = (gate_tile_cnt + slice_chunk - Int32(1)) // slice_chunk
        start = atomic_add_global_i32(get_ptr_as_int64(task_tail, Int32(0)), num_groups)

        g = Int32(0)
        while g < num_groups:
            slot = start + g
            slice_begin = g * slice_chunk
            slice_count = gate_tile_cnt - slice_begin
            if slice_count > slice_chunk:
                slice_count = slice_chunk
            task_expert[slot] = expert_idx
            task_m_tile[slot] = m_tile_idx
            task_slice_begin[slot] = slice_begin
            task_slice_count[slot] = slice_count
            task_valid_rows[slot] = valid_rows
            g += Int32(1)

        _threadfence()

        g = Int32(0)
        while g < num_groups:
            slot = start + g
            _st_global_release_i32(get_ptr_as_int64(task_ready, slot), Int32(1))
            g += Int32(1)

    @cute.jit
    def _publish_deferred_tasks(
        self,
        task_expert: cute.Tensor,
        task_m_tile: cute.Tensor,
        task_slice_begin: cute.Tensor,
        task_slice_count: cute.Tensor,
        task_valid_rows: cute.Tensor,
        gate_tile_cnt: Int32,
        slice_chunk: Int32,
        expert_idx: Int32,
        m_tile_idx: Int32,
        valid_rows: Int32,
    ):
        num_groups = (gate_tile_cnt + slice_chunk - Int32(1)) // slice_chunk
        start = m_tile_idx * num_groups

        g = Int32(0)
        while g < num_groups:
            slot = start + g
            slice_begin = g * slice_chunk
            slice_count = gate_tile_cnt - slice_begin
            if slice_count > slice_chunk:
                slice_count = slice_chunk
            task_expert[slot] = expert_idx
            task_m_tile[slot] = m_tile_idx
            task_slice_begin[slot] = slice_begin
            task_slice_count[slot] = slice_count
            task_valid_rows[slot] = valid_rows
            g += Int32(1)

    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,  # [num_tokens, K] bf16
        topk_ids: cute.Tensor,  # [num_tokens * topk] int32
        topk_weights: cute.Tensor,  # [num_tokens * topk] float32
        packed_a: cute.Tensor,  # [rows_padded, K, 1] BF16 routed scratch
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,  # legacy NVFP4 scratch, unused by W4A16
        scale_storage: cute.Tensor,  # legacy activation scales, unused by W4A16
        barrier_count: cute.Tensor,  # [1] int32 (host-zeroed)
        barrier_epoch: cute.Tensor,  # [1] int32 (host-zeroed)
        pair_head: cute.Tensor,  # [1] int32
        producers_done_count: cute.Tensor,  # [1] int32
        all_work_published: cute.Tensor,  # [1] int32
        task_head: cute.Tensor,  # [1] int32
        task_tail: cute.Tensor,  # [1] int32
        task_ready: cute.Tensor,  # [max_tasks] int32
        task_expert: cute.Tensor,  # [max_tasks] int32
        task_m_tile: cute.Tensor,  # [max_tasks] int32
        task_slice_begin: cute.Tensor,  # [max_tasks] int32
        task_slice_count: cute.Tensor,  # [max_tasks] int32
        task_valid_rows: cute.Tensor,  # [max_tasks] int32
        tile_write_count: cute.Tensor,  # [E * max_m_tiles] int32
        b_w13: cute.Tensor,  # [w1_n, K, E] — gated packs [up, gate], relu2 is single FC1
        sfb_w13_ptr: cute.Pointer,  # scale factors for FC1 weights
        b_down: cute.Tensor,  # [K, I_tp, E]
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,  # expert row histogram [E]
        expert_write_rows: cute.Tensor,  # route/pack write cursors [E]
        expert_tile_base: cute.Tensor,  # compact physical-tile prefix [E + 1]
        input_global_scale: cute.Tensor,  # legacy activation scale, unused by W4A16
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,  # [num_tokens, K]
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = packed_a.element_type
        self.b_dtype = cutlass.BFloat16
        self.a_layout = utils.LayoutEnum.from_tensor(packed_a)
        self.b_layout = utils.LayoutEnum.from_tensor(b_w13)
        # Dynamic never materializes the intermediate C tensor. Preserve the
        # original row-major epilogue layout without carrying a dead memref.
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        if cutlass.const_expr(self.a_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 routed A scratch, got {self.a_dtype}")

        self._setup_attributes()

        # TMA descriptors
        tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
            packed_a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tile_n = Int32(self.tile_shape_mnk[1])
        gate_tile_cnt = (Int32(b_w13.shape[0]) + tile_n - Int32(1)) // tile_n
        if self.is_gated:
            gate_tile_cnt = gate_tile_cnt // Int32(2)
        launch_params = DynamicLaunchParams(row_counts, gate_tile_cnt)
        grid = (*self.cluster_shape_mn, max_active_clusters)
        self.kernel(
            a_input,
            topk_ids,
            topk_weights,
            packed_a,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            pair_head,
            producers_done_count,
            all_work_published,
            task_head,
            task_tail,
            task_ready,
            task_expert,
            task_m_tile,
            task_slice_begin,
            task_slice_count,
            task_valid_rows,
            tile_write_count,
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
            launch_params,
            expert_write_rows,
            expert_tile_base,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        packed_a: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        producers_done_count: cute.Tensor,
        all_work_published: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_ready: cute.Tensor,
        task_expert: cute.Tensor,
        task_m_tile: cute.Tensor,
        task_slice_begin: cute.Tensor,
        task_slice_count: cute.Tensor,
        task_valid_rows: cute.Tensor,
        tile_write_count: cute.Tensor,
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
        launch_params: DynamicLaunchParams,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
    ):
        """Kernel entry point."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()
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
        class Storage:
            ctrl: cute.struct.MemRange[cutlass.Int32, 8]
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            scatter_tok_cache: cute.struct.MemRange[
                cutlass.Int32, self.tile_shape_mnk[0]
            ]
            scatter_weight_cache: cute.struct.MemRange[
                cutlass.Float32, self.tile_shape_mnk[0]
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

        storage = smem.allocate(Storage)

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

        cute.arch.sync_threads()

        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sB_up = storage.sB_up.get_tensor(
            b_smem_staged.outer, swizzle=b_smem_staged.inner
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
        a_base = a_input.iterator.toint()
        packed_a_base = packed_a.iterator.toint()
        scatter_base = scatter_output.iterator.toint()
        row_counts = launch_params.row_counts
        num_experts = Int32(row_counts.shape[0])
        total_pairs = Int32(topk_ids.shape[0])
        num_topk = total_pairs // num_tokens
        cols_u32 = cols // Int32(2)
        scatter_output_u32 = cute.recast_tensor(scatter_output, cutlass.Uint32)
        flat_tid = Int32(bidz) * Int32(self.threads_per_cta) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(self.threads_per_cta)
        route_gate_tile_cnt = launch_params.gate_tile_cnt
        task_slice_chunk = Int32(_TASK_SLICE_CHUNK)
        full_tile_publish_enabled = Int32(0)

        # Phase 0: cooperative init — zero routing state, queue state, and output
        task_capacity = Int32(task_ready.shape[0])
        tile_write_slots = Int32(tile_write_count.shape[0])
        i = flat_tid
        while i < num_experts:
            row_counts[i] = Int32(0)
            expert_write_rows[i] = Int32(0)
            i += flat_stride
        if flat_tid < num_experts + Int32(1):
            expert_tile_base[flat_tid] = Int32(0)

        scatter_total_u32 = num_tokens * cols_u32
        scatter_vecs = scatter_total_u32 // Int32(4)
        zero_u32 = Uint32(0)
        zv = flat_tid
        while zv < scatter_vecs:
            st_global_v4_u32(
                scatter_base + Int64(zv) * Int64(16),
                zero_u32,
                zero_u32,
                zero_u32,
                zero_u32,
            )
            zv += flat_stride

        j = scatter_vecs * Int32(4) + flat_tid
        while j < scatter_total_u32:
            scatter_output_u32[j // cols_u32, j % cols_u32] = Uint32(0)
            j += flat_stride

        k = flat_tid
        while k < task_capacity:
            task_ready[k] = Int32(0)
            task_expert[k] = Int32(0)
            task_m_tile[k] = Int32(0)
            task_slice_begin[k] = Int32(0)
            task_slice_count[k] = Int32(0)
            task_valid_rows[k] = Int32(0)
            k += flat_stride

        if full_tile_publish_enabled > Int32(0):
            tw = flat_tid
            while tw < tile_write_slots:
                tile_write_count[tw] = Int32(0)
                tw += flat_stride

        if flat_tid == Int32(0):
            pair_head[Int32(0)] = Int32(0)
            producers_done_count[Int32(0)] = Int32(0)
            all_work_published[Int32(0)] = Int32(0)
            task_head[Int32(0)] = Int32(0)
            task_tail[Int32(0)] = Int32(0)

        cute.arch.sync_threads()
        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        # Phase 1: histogram routed rows per expert.
        hist_idx = flat_tid
        while hist_idx < total_pairs:
            expert_id = topk_ids[hist_idx].to(Int32)
            atomic_add_global_i32(get_ptr_as_int64(row_counts, expert_id), Int32(1))
            hist_idx += flat_stride

        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        if flat_tid == Int32(0):
            tile_acc = Int32(0)
            expert_idx = Int32(0)
            while expert_idx < num_experts:
                expert_tile_base[expert_idx] = tile_acc
                rows = row_counts[expert_idx]
                tile_acc += (rows + Int32(self.tile_shape_mnk[0]) - Int32(1)) // Int32(
                    self.tile_shape_mnk[0]
                )
                expert_idx += Int32(1)
            expert_tile_base[num_experts] = tile_acc

        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        # Phase 2: warp-private route/pack producers into compact physical tiles.
        lane_id = Int32(tidx) & Int32(31)
        num_cta_warps = Int32(self.num_mma_warps + self.num_load_warps)
        pair_idx = Int32(0)
        expert_id = Int32(0)
        token_idx = Int32(0)
        weight = cutlass.Float32(0.0)
        row = Int32(0)
        phys_tile = Int32(0)
        phys_row = Int32(0)
        produce_active = Int32(1)
        while produce_active > Int32(0):
            batch_base = Int32(0)
            if is_cta_leader > Int32(0):
                batch_base = atomic_add_global_i32(
                    get_ptr_as_int64(pair_head, Int32(0)),
                    num_cta_warps,
                )
                _st_shared_i32(ctrl_base_addr + Int32(28), batch_base)
            cute.arch.sync_threads()
            batch_base = _ld_shared_i32(ctrl_base_addr + Int32(28))
            if batch_base >= total_pairs:
                produce_active = Int32(0)
            else:
                pair_idx = batch_base + warp_idx
                expert_id = Int32(0)
                token_idx = Int32(0)
                weight = cutlass.Float32(0.0)
                row = Int32(0)
                phys_tile = Int32(0)
                if pair_idx < total_pairs:
                    expert_id = topk_ids[pair_idx].to(Int32)
                    token_idx = pair_idx // num_topk
                    weight = topk_weights[pair_idx].to(cutlass.Float32)

                    if lane_id == Int32(0):
                        row = atomic_add_global_i32(
                            get_ptr_as_int64(expert_write_rows, expert_id),
                            Int32(1),
                        )
                        phys_tile = expert_tile_base[expert_id] + row // Int32(
                            self.tile_shape_mnk[0]
                        )
                        phys_row = phys_tile * Int32(
                            self.tile_shape_mnk[0]
                        ) + row % Int32(self.tile_shape_mnk[0])
                        st_global_i32(get_ptr_as_int64(token_map, phys_row), token_idx)
                        token_weights[phys_row] = weight

                    row = cute.arch.shuffle_sync(row, Int32(0))
                    phys_tile = cute.arch.shuffle_sync(phys_tile, Int32(0))
                    expert_id = cute.arch.shuffle_sync(expert_id, Int32(0))
                    token_idx = cute.arch.shuffle_sync(token_idx, Int32(0))

                    phys_row = phys_tile * Int32(self.tile_shape_mnk[0]) + row % Int32(
                        self.tile_shape_mnk[0]
                    )
                    vec_cols = cols // Int32(8)
                    vec_idx = lane_id
                    while vec_idx < vec_cols:
                        vec_col = vec_idx * Int32(8)
                        src_byte = (
                            Int64(token_idx) * Int64(cols) + Int64(vec_col)
                        ) * Int64(2)
                        dst_byte = (
                            Int64(phys_row) * Int64(cols) + Int64(vec_col)
                        ) * Int64(2)
                        v0, v1, v2, v3 = ld_global_nc_v4_u32(a_base + src_byte)
                        st_global_v4_u32(packed_a_base + dst_byte, v0, v1, v2, v3)
                        vec_idx += Int32(32)

                    col = vec_cols * Int32(8) + lane_id
                    while col < cols:
                        packed_a[phys_row, col, Int32(0)] = a_input[token_idx, col]
                        col += Int32(32)

                    if full_tile_publish_enabled > Int32(0):
                        cute.arch.sync_warp()
                        # When the whole launch has fewer than one M-tile of routed
                        # rows, only the final partial-tile flush can publish work.
                        # Skip the per-row fence/counter path in that common micro case.
                        _threadfence()
                        cute.arch.sync_warp()

                        if lane_id == Int32(0):
                            completed = atomic_add_global_i32(
                                get_ptr_as_int64(tile_write_count, phys_tile),
                                Int32(1),
                            ) + Int32(1)
                            if completed == Int32(self.tile_shape_mnk[0]):
                                self._publish_ready_tasks(
                                    task_tail,
                                    task_ready,
                                    task_expert,
                                    task_m_tile,
                                    task_slice_begin,
                                    task_slice_count,
                                    task_valid_rows,
                                    route_gate_tile_cnt,
                                    task_slice_chunk,
                                    expert_id,
                                    phys_tile,
                                    Int32(self.tile_shape_mnk[0]),
                                )

        cute.arch.sync_threads()
        # Conservative publish fence before the last-producer CTA flushes any
        # partial tiles. All producer threads in the CTA must have ordered
        # their global writes before lane 0 can publish work.
        _threadfence()
        cute.arch.sync_threads()

        if full_tile_publish_enabled == Int32(0):
            # Flush routed work once, then consume direct task slots without
            # per-task ready flags.
            self._resident_grid_barrier(
                barrier_count,
                barrier_epoch,
                Int32(gdim_z),
                is_cta_leader,
            )

            if is_cta_leader > Int32(0):
                expert_flush = Int32(bidz)
                while expert_flush < num_experts:
                    rows_remaining = row_counts[expert_flush]
                    m_tile_offset = Int32(0)
                    while rows_remaining > Int32(0):
                        valid_rows = rows_remaining
                        if valid_rows > Int32(self.tile_shape_mnk[0]):
                            valid_rows = Int32(self.tile_shape_mnk[0])
                        self._publish_deferred_tasks(
                            task_expert,
                            task_m_tile,
                            task_slice_begin,
                            task_slice_count,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            task_slice_chunk,
                            expert_flush,
                            expert_tile_base[expert_flush] + m_tile_offset,
                            valid_rows,
                        )
                        rows_remaining -= Int32(self.tile_shape_mnk[0])
                        m_tile_offset += Int32(1)
                    expert_flush += Int32(gdim_z)

            if flat_tid == Int32(0):
                num_groups = (
                    route_gate_tile_cnt + task_slice_chunk - Int32(1)
                ) // task_slice_chunk
                st_global_i32(
                    get_ptr_as_int64(task_tail, Int32(0)),
                    expert_tile_base[num_experts] * num_groups,
                )

            self._resident_grid_barrier(
                barrier_count,
                barrier_epoch,
                Int32(gdim_z),
                is_cta_leader,
            )
            if flat_tid == Int32(0):
                _st_global_release_i32(
                    get_ptr_as_int64(all_work_published, Int32(0)),
                    Int32(1),
                )
        elif is_cta_leader > Int32(0):
            prev_done = atomic_add_global_i32(
                get_ptr_as_int64(producers_done_count, Int32(0)),
                Int32(1),
            )
            if prev_done == Int32(gdim_z) - Int32(1):
                expert_flush = Int32(0)
                while expert_flush < num_experts:
                    rows = row_counts[expert_flush]
                    rem = rows % Int32(self.tile_shape_mnk[0])
                    if rem != Int32(0):
                        partial_m_tile = expert_tile_base[expert_flush] + rows // Int32(
                            self.tile_shape_mnk[0]
                        )
                        self._publish_ready_tasks(
                            task_tail,
                            task_ready,
                            task_expert,
                            task_m_tile,
                            task_slice_begin,
                            task_slice_count,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            task_slice_chunk,
                            expert_flush,
                            partial_m_tile,
                            rem,
                        )
                    expert_flush += Int32(1)
                _threadfence()
                _st_global_release_i32(
                    get_ptr_as_int64(all_work_published, Int32(0)),
                    Int32(1),
                )

        gA = cute.local_tile(
            mA, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
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
        tCsA = thr_mma.partition_A(sA)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCsB = thr_mma.partition_B(sB)
        tCsB_up = thr_mma.partition_B(sB_up)
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrB_up = tiled_mma.make_fragment_B(tCsB_up[None, None, None, 0])

        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        sub_shape = tCsC_for_shape.shape[:3]
        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
        gate_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

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
        # Gated FC1 packs [up, gate] across N; relu2 has a single FC1 pass.
        intermediate_tile_cnt = (w13_rows + tile_n - Int32(1)) // tile_n
        gate_tile_cnt = intermediate_tile_cnt
        if self.is_gated:
            gate_tile_cnt = intermediate_tile_cnt // Int32(2)
        output_tile_cnt = (down_rows + tile_n - Int32(1)) // tile_n

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        num_k_blocks = cute.size(tCrA, mode=[2])

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
        csA = thr_ld_A.partition_S(sA)
        crA = thr_ld_A.retile(tCrA)
        csB = thr_ld_B.partition_S(sB)
        csB_up = thr_ld_B.partition_S(sB_up)
        crB = thr_ld_B.retile(tCrB)
        crB_up = thr_ld_B.retile(tCrB_up)

        # ===================================================================
        # Per-warp setup for the consumer steady state
        # ===================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
        elif (
            warp_idx >= self.tma_load_warp_id
            and warp_idx < self.tma_load_warp_id + self.num_load_warps
        ):
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ===================================================================
        # Consumer steady state: pop one ready task per CTA, then let
        # the MMA warps and DMA warp cooperate on that task.
        # ===================================================================
        consumer_live = Int32(1)
        while consumer_live > Int32(0):
            if is_cta_leader > Int32(0):
                _st_shared_i32(ctrl_base_addr + Int32(0), Int32(0))  # has_task
                _st_shared_i32(ctrl_base_addr + Int32(4), Int32(0))  # done
                _st_shared_i32(ctrl_base_addr + Int32(28), Int32(0))  # claimed slot
                if full_tile_publish_enabled == Int32(0):
                    tail = _ld_global_acquire_i32(get_ptr_as_int64(task_tail, Int32(0)))
                    slot = atomic_add_global_i32(
                        get_ptr_as_int64(task_head, Int32(0)),
                        Int32(1),
                    )
                    if slot < tail:
                        _st_shared_i32(ctrl_base_addr + Int32(0), Int32(1))
                        _st_shared_i32(ctrl_base_addr + Int32(28), slot)
                        _st_shared_i32(ctrl_base_addr + Int32(8), task_expert[slot])
                        _st_shared_i32(ctrl_base_addr + Int32(12), task_m_tile[slot])
                        _st_shared_i32(
                            ctrl_base_addr + Int32(16), task_slice_begin[slot]
                        )
                        _st_shared_i32(
                            ctrl_base_addr + Int32(20), task_slice_count[slot]
                        )
                        _st_shared_i32(
                            ctrl_base_addr + Int32(24), task_valid_rows[slot]
                        )
                    else:
                        _st_shared_i32(ctrl_base_addr + Int32(4), Int32(1))
                else:
                    head = _ld_global_acquire_i32(get_ptr_as_int64(task_head, Int32(0)))
                    tail = _ld_global_acquire_i32(get_ptr_as_int64(task_tail, Int32(0)))
                    if head < tail:
                        prev_head = _atomic_cas_global_i32(
                            get_ptr_as_int64(task_head, Int32(0)),
                            head,
                            head + Int32(1),
                        )
                        if prev_head == head:
                            _spin_wait_global_eq_i32(
                                get_ptr_as_int64(task_ready, head), Int32(0)
                            )
                            _st_shared_i32(ctrl_base_addr + Int32(0), Int32(1))
                            _st_shared_i32(ctrl_base_addr + Int32(28), head)
                            _st_shared_i32(ctrl_base_addr + Int32(8), task_expert[head])
                            _st_shared_i32(
                                ctrl_base_addr + Int32(12), task_m_tile[head]
                            )
                            _st_shared_i32(
                                ctrl_base_addr + Int32(16), task_slice_begin[head]
                            )
                            _st_shared_i32(
                                ctrl_base_addr + Int32(20), task_slice_count[head]
                            )
                            _st_shared_i32(
                                ctrl_base_addr + Int32(24), task_valid_rows[head]
                            )
                    else:
                        if _ld_global_acquire_i32(
                            get_ptr_as_int64(all_work_published, Int32(0))
                        ) > Int32(0):
                            _st_shared_i32(ctrl_base_addr + Int32(4), Int32(1))
            cute.arch.sync_threads()

            has_task = _ld_shared_i32(ctrl_base_addr + Int32(0))
            is_done = _ld_shared_i32(ctrl_base_addr + Int32(4))
            if has_task > Int32(0) and full_tile_publish_enabled > Int32(0):
                claimed_slot = _ld_shared_i32(ctrl_base_addr + Int32(28))
                _ld_global_acquire_i32(get_ptr_as_int64(task_ready, claimed_slot))
            if has_task > Int32(0):
                task_m_tile_idx_cache = _ld_shared_i32(ctrl_base_addr + Int32(12))
                task_valid_rows_cache = _ld_shared_i32(ctrl_base_addr + Int32(24))
                tile_m_base_cache = task_m_tile_idx_cache * Int32(
                    self.tile_shape_mnk[0]
                )
                cache_row = Int32(tidx)
                while cache_row < Int32(self.tile_shape_mnk[0]):
                    tok = Int32(0)
                    wv = cutlass.Float32(0.0)
                    if cache_row < task_valid_rows_cache:
                        global_row_cache = tile_m_base_cache + cache_row
                        tok = token_map[global_row_cache].to(Int32)
                        wv = token_weights[global_row_cache].to(cutlass.Float32)
                    _st_shared_i32(scatter_tok_base_addr + cache_row * Int32(4), tok)
                    st_shared_f32(scatter_weight_base_addr + cache_row * Int32(4), wv)
                    cache_row += Int32(self.threads_per_cta)
                cute.arch.sync_threads()
            if has_task == Int32(0):
                if is_done > Int32(0):
                    consumer_live = Int32(0)
            elif warp_idx < self.num_mma_warps:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
                task_m_tile_idx = _ld_shared_i32(ctrl_base_addr + Int32(12))
                task_slice_begin_idx = _ld_shared_i32(ctrl_base_addr + Int32(16))
                task_slice_count_val = _ld_shared_i32(ctrl_base_addr + Int32(20))
                task_valid_rows_val = _ld_shared_i32(ctrl_base_addr + Int32(24))

                alpha_value = alpha[task_expert_idx].to(cutlass.Float32)
                valid_rows = task_valid_rows_val
                _is_m_major = self.c_layout.is_m_major_c()
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

                epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
                MmaMPerEpiM = self.epi_tile[0] // mma_tile_m
                MmaNPerEpiN = self.epi_tile[1] // mma_tile_n
                slice_idx = Int32(0)
                while slice_idx < task_slice_count_val:
                    intermediate_slice = task_slice_begin_idx + slice_idx
                    gate_slice_idx = intermediate_slice
                    if self.is_gated:
                        gate_slice_idx = intermediate_slice + gate_tile_cnt

                    # ============================================================
                    # PHASE A: FC1 for this slice (gate/only pass, plus up for silu)
                    # ============================================================

                    if self.is_gated:
                        gate_acc.fill(0.0)
                        up_acc.fill(0.0)
                        cons_state.reset_count()
                        for _k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                            peek = ml_pipeline.consumer_try_wait(cons_state)
                            ml_pipeline.consumer_wait(cons_state, peek)
                            self._stage_w13_fp4_b_tile(
                                b_w13,
                                sfb_w13_ptr,
                                sB,
                                cons_state.index,
                                task_expert_idx,
                                gate_slice_idx,
                                _k_tile,
                                w13_rows,
                                w13_cols,
                                w13_sf_cols,
                                Int32(tidx),
                                Int32(self.num_mma_warps * self.num_threads_per_warp),
                            )
                            self._stage_w13_fp4_b_tile(
                                b_w13,
                                sfb_w13_ptr,
                                sB_up,
                                cons_state.index,
                                task_expert_idx,
                                intermediate_slice,
                                _k_tile,
                                w13_rows,
                                w13_cols,
                                w13_sf_cols,
                                Int32(tidx),
                                Int32(self.num_mma_warps * self.num_threads_per_warp),
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            self.epilog_sync_barrier.arrive_and_wait()
                            csA_p = csA[None, None, None, cons_state.index]
                            csB_p = csB[None, None, None, cons_state.index]
                            csB_up_p = csB_up[None, None, None, cons_state.index]
                            cute.copy(
                                smem_copy_A,
                                csA_p[None, None, 0],
                                crA[None, None, 0],
                            )
                            for _kb_pre in cutlass.range_constexpr(num_k_blocks - 1):
                                k_pre = _kb_pre + 1
                                cute.copy(
                                    smem_copy_A,
                                    csA_p[None, None, k_pre],
                                    crA[None, None, k_pre],
                                )
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                cute.copy(
                                    smem_copy_B,
                                    csB_p[None, None, k_block_idx],
                                    crB[None, None, k_block_idx],
                                )
                                cute.gemm(
                                    tiled_mma,
                                    gate_acc,
                                    tCrA[None, None, k_block_idx],
                                    tCrB[None, None, k_block_idx],
                                    gate_acc,
                                )
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                cute.copy(
                                    smem_copy_B,
                                    csB_up_p[None, None, k_block_idx],
                                    crB_up[None, None, k_block_idx],
                                )
                                cute.gemm(
                                    tiled_mma,
                                    up_acc,
                                    tCrA[None, None, k_block_idx],
                                    tCrB_up[None, None, k_block_idx],
                                    up_acc,
                                )
                            ml_pipeline.consumer_release(cons_state)
                            cons_state.advance()
                        self.pass_sync_barrier.arrive_and_wait()
                    else:
                        # Gate/only GEMM (inlined to avoid @cute.jit pass-by-value for acc)
                        gate_acc.fill(0.0)
                        cons_state.reset_count()
                        peek = ml_pipeline.consumer_try_wait(cons_state)
                        ml_pipeline.consumer_wait(cons_state, peek)
                        self._stage_w13_fp4_b_tile(
                            b_w13,
                            sfb_w13_ptr,
                            sB,
                            cons_state.index,
                            task_expert_idx,
                            gate_slice_idx,
                            Int32(0),
                            w13_rows,
                            w13_cols,
                            w13_sf_cols,
                            Int32(tidx),
                            Int32(self.num_mma_warps * self.num_threads_per_warp),
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        csA_p = csA[None, None, None, cons_state.index]
                        csB_p = csB[None, None, None, cons_state.index]
                        cute.copy(smem_copy_A, csA_p[None, None, 0], crA[None, None, 0])
                        cute.copy(smem_copy_B, csB_p[None, None, 0], crB[None, None, 0])
                        for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):  # type: ignore[call-overload]
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                k_next = (
                                    0
                                    if k_block_idx + 1 == num_k_blocks
                                    else k_block_idx + 1
                                )
                                if k_block_idx == num_k_blocks - 1:
                                    ml_pipeline.consumer_release(cons_state)
                                    cons_state.advance()
                                    peek = ml_pipeline.consumer_try_wait(cons_state)
                                    csA_p = csA[None, None, None, cons_state.index]
                                    csB_p = csB[None, None, None, cons_state.index]
                                    ml_pipeline.consumer_wait(cons_state, peek)
                                    self._stage_w13_fp4_b_tile(
                                        b_w13,
                                        sfb_w13_ptr,
                                        sB,
                                        cons_state.index,
                                        task_expert_idx,
                                        gate_slice_idx,
                                        _k_tile + Int32(1),
                                        w13_rows,
                                        w13_cols,
                                        w13_sf_cols,
                                        Int32(tidx),
                                        Int32(
                                            self.num_mma_warps
                                            * self.num_threads_per_warp
                                        ),
                                    )
                                    cute.arch.fence_proxy("async.shared", space="cta")
                                    self.epilog_sync_barrier.arrive_and_wait()
                                cute.gemm(
                                    tiled_mma,
                                    gate_acc,
                                    tCrA[None, None, k_block_idx],
                                    tCrB[None, None, k_block_idx],
                                    gate_acc,
                                )
                                cute.copy(
                                    smem_copy_A,
                                    csA_p[None, None, k_next],
                                    crA[None, None, k_next],
                                )
                                cute.copy(
                                    smem_copy_B,
                                    csB_p[None, None, k_next],
                                    crB[None, None, k_next],
                                )
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_next = (
                                0
                                if k_block_idx + 1 == num_k_blocks
                                else k_block_idx + 1
                            )
                            if k_block_idx == num_k_blocks - 1:
                                ml_pipeline.consumer_release(cons_state)
                                cons_state.advance()
                            if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                                cute.copy(
                                    smem_copy_A,
                                    csA_p[None, None, k_next],
                                    crA[None, None, k_next],
                                )
                                cute.copy(
                                    smem_copy_B,
                                    csB_p[None, None, k_next],
                                    crB[None, None, k_next],
                                )
                            cute.gemm(
                                tiled_mma,
                                gate_acc,
                                tCrA[None, None, k_block_idx],
                                tCrB[None, None, k_block_idx],
                                gate_acc,
                            )
                        self.pass_sync_barrier.arrive_and_wait()

                    # Activation into BF16 sA for FC2. W4A16 has no activation
                    # requantization and no activation scale fragment.
                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        epi_m_valid = valid_rows - Int32(epi_m) * Int32(
                            self.epi_tile[0]
                        )
                        epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                        if epi_m_valid > Int32(0):
                            tRS_rD.fill(0.0)
                            for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                                for mma_m_in_epi in cutlass.range_constexpr(
                                    MmaMPerEpiM
                                ):
                                    mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                    mma_n = mma_n_in_epi
                                    tRS_rD_slice = tRS_rD[
                                        (None, mma_m_in_epi, mma_n_in_epi)
                                    ]
                                    gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                    if self.is_gated:
                                        up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size(tRS_rD_slice)
                                        ):
                                            g = alpha_value * cutlass.Float32(
                                                cutlass.BFloat16(gate_slice[elem_idx])
                                            )
                                            u = alpha_value * cutlass.Float32(
                                                cutlass.BFloat16(up_slice[elem_idx])
                                            )
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
                                            g = alpha_value * cutlass.Float32(
                                                cutlass.BFloat16(gate_slice[elem_idx])
                                            )
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
                        epi_cols = Int32(self.epi_tile[1])
                        while copy_idx < epi_rows * epi_cols:
                            local_row = copy_idx // epi_cols
                            col = copy_idx - local_row * epi_cols
                            sA[rows_offset + local_row, col, Int32(0)] = sC[
                                local_row, col, epi_buffer
                            ]
                            copy_idx += Int32(
                                self.num_mma_warps * self.num_threads_per_warp
                            )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                    cute.arch.fence_proxy("async.shared", space="cta")
                    # All MMA warps must finish materializing the activated BF16
                    # intermediate before FC2 reads sA.
                    self.epilog_sync_barrier.arrive_and_wait()

                    # ============================================================
                    # PHASE B: Sweep ALL FC2 output tiles using cached BF16 sA.
                    # B_down is staged cooperatively by the whole CTA.
                    # ============================================================
                    scatter_N = Int32(scatter_output.shape[1])
                    down_alpha_value = down_alpha[task_expert_idx].to(cutlass.Float32)
                    down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
                    csA_phase2 = csA[None, None, None, 0]

                    # Hoist A-side register loads: sA is constant across all
                    # FC2 output tiles (BF16 intermediate). Load crA for all
                    # k-blocks once, reuse for all output tiles.
                    cute.copy(
                        smem_copy_A, csA_phase2[None, None, 0], crA[None, None, 0]
                    )
                    for _kb_pre in cutlass.range_constexpr(num_k_blocks - 1):
                        k_pre = _kb_pre + 1
                        cute.copy(
                            smem_copy_A,
                            csA_phase2[None, None, k_pre],
                            crA[None, None, k_pre],
                        )

                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                        self._stage_down_fp4_b_tile(
                            b_down,
                            sfb_down_ptr,
                            sB,
                            Int32(0),
                            task_expert_idx,
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
                                0
                                if k_block_idx + 1 == num_k_blocks
                                else k_block_idx + 1
                            )
                            if k_next > 0:
                                # Only B-side for next k-block (A already in registers)
                                cute.copy(
                                    smem_copy_B,
                                    csB_phase2[None, None, k_next],
                                    crB[None, None, k_next],
                                )
                            cute.gemm(
                                tiled_mma,
                                down_acc,
                                tCrA[None, None, k_block_idx],
                                tCrB[None, None, k_block_idx],
                                down_acc,
                            )

                        # Scatter using precomputed metadata (no redundant gmem loads)
                        tile_n_base_cur = output_tile_idx * Int32(
                            self.tile_shape_mnk[1]
                        )
                        for epi_m in cutlass.range_constexpr(epi_rest_m):
                            for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                                for mma_m_in_epi in cutlass.range_constexpr(
                                    MmaMPerEpiM
                                ):
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
                                            down_alpha_value
                                            * down_epi_acc_slice[elem_idx]
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
                            # StMatrix ownership is interleaved across MMA warps
                            # for underfilled tiles, so wait for every warp's
                            # store before reading sC for scatter.
                            self.epilog_sync_barrier.arrive_and_wait()

                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                            epi_rows = valid_rows - rows_offset
                            if epi_rows > Int32(self.epi_tile[0]):
                                epi_rows = Int32(self.epi_tile[0])
                            if epi_rows < Int32(0):
                                epi_rows = Int32(0)

                            tile_vec_cols = Int32(self.tile_shape_mnk[1]) // Int32(8)
                            vec_idx = Int32(tidx)
                            while vec_idx < epi_rows * tile_vec_cols:
                                local_row = vec_idx // tile_vec_cols
                                local_vec_col = vec_idx - local_row * tile_vec_cols
                                local_col = local_vec_col * Int32(8)
                                global_col = tile_n_base_cur + local_col
                                cached_row = rows_offset + local_row
                                tok = ld_shared_i32_relaxed(
                                    scatter_tok_base_addr + cached_row * Int32(4)
                                )
                                wv = ld_shared_f32(
                                    scatter_weight_base_addr + cached_row * Int32(4)
                                )
                                sc_v0 = cutlass.Float32(
                                    sC[local_row, local_col, epi_buffer]
                                )
                                sc_v1 = cutlass.Float32(
                                    sC[local_row, local_col + Int32(1), epi_buffer]
                                )
                                sc_v2 = cutlass.Float32(
                                    sC[local_row, local_col + Int32(2), epi_buffer]
                                )
                                sc_v3 = cutlass.Float32(
                                    sC[local_row, local_col + Int32(3), epi_buffer]
                                )
                                sc_v4 = cutlass.Float32(
                                    sC[local_row, local_col + Int32(4), epi_buffer]
                                )
                                sc_v5 = cutlass.Float32(
                                    sC[local_row, local_col + Int32(5), epi_buffer]
                                )
                                sc_v6 = cutlass.Float32(
                                    sC[local_row, local_col + Int32(6), epi_buffer]
                                )
                                sc_v7 = cutlass.Float32(
                                    sC[local_row, local_col + Int32(7), epi_buffer]
                                )
                                scatter_add_v4_bf16x2(
                                    get_ptr_as_int64(
                                        scatter_output, tok * scatter_N + global_col
                                    ),
                                    wv * sc_v0,
                                    wv * sc_v1,
                                    wv * sc_v2,
                                    wv * sc_v3,
                                    wv * sc_v4,
                                    wv * sc_v5,
                                    wv * sc_v6,
                                    wv * sc_v7,
                                )
                                vec_idx += Int32(
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
                    slice_idx += Int32(1)

            elif (
                warp_idx >= self.tma_load_warp_id
                and warp_idx < self.tma_load_warp_id + self.num_load_warps
            ):
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
                task_m_tile_idx = _ld_shared_i32(ctrl_base_addr + Int32(12))
                task_slice_begin_idx = _ld_shared_i32(ctrl_base_addr + Int32(16))
                task_slice_count_val = _ld_shared_i32(ctrl_base_addr + Int32(20))
                load_warp_local = warp_idx - Int32(self.tma_load_warp_id)

                tAgA_mk = tAgA[(None, task_m_tile_idx, None, Int32(0))]
                slice_idx = Int32(0)
                while slice_idx < task_slice_count_val:
                    intermediate_slice = task_slice_begin_idx + slice_idx

                    # FC1 producer slice. Gated activation packs [up, gate]
                    # across N; relu2 uses a single FC1 pass.
                    gate_slice_idx = intermediate_slice
                    if self.is_gated:
                        gate_slice_idx = intermediate_slice + gate_tile_cnt

                    # ---- FC1 gate pass ----
                    prod_state.reset_count()
                    if load_warp_local == Int32(0):
                        for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                            ml_pipeline.producer_acquire(prod_state)
                            cute.copy(
                                tma_a,
                                tAgA_mk[(None, k_tile)],
                                tAsA[(None, prod_state.index)],
                                tma_bar_ptr=ml_pipeline.producer_get_barrier(
                                    prod_state
                                ),
                            )
                            ml_pipeline.producer_commit(prod_state)
                            prod_state.advance()

                    # Wait for the MMA warps to finish FC1 before FC2 begins.
                    self.pass_sync_barrier.arrive_and_wait()

                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                        self._stage_down_fp4_b_tile(
                            b_down,
                            sfb_down_ptr,
                            sB,
                            Int32(0),
                            task_expert_idx,
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
                    slice_idx += Int32(1)

        if warp_idx == self.tma_load_warp_id:
            ml_pipeline.producer_tail(prod_state)
        return


__all__ = ["MoEDynamicKernel"]
