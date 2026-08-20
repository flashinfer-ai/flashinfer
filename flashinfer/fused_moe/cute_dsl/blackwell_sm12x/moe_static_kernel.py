"""Retained static NVFP4/MXFP4 MoE with route-major finalization.

The gated path keeps the paired Gate/Up transaction and groups two adjacent
N128 intermediate slices.  Both Q1 tensors remain in the two existing A/SFA
shared stages, FC2 accumulates them before one scatter, and the task count is
halved without adding shared storage.  Both FP4 formats share this schedule;
only the MMA atom, quantization block width, and packed byte count differ.

Ported from the b12x kernel library to FlashInfer.

This is the current static control-plane fusion step: keep the proven FC1/FC2
compute body, but pull the route/pack frontend into the same resident kernel.
The result is still a two-phase algorithm, just without a host-side handoff:

  Phase 0: cooperative init / clear row counts
  Phase 1: walk routed (token, topk_slot) pairs, append rows per expert,
           write token_map + token_weights, and quantize each routed
           token row directly into expert-major packed A + scale storage
  Barrier: resident-grid barrier after all expert rows are finalized
  Phase 2: run the FC1 -> activation -> quant -> FC2 -> scatter datapath
           over the finalized expert-major packed input

The compute half is intentionally the same design as the earlier two-kernel
implementation:
  SiLU (gated):
    FC1:     A x gate^T, A x up^T     (paired FP4 block-scaled GEMMs)
    Act:     SiLU(gate) * up           (fused SwiGLU activation)
  ReLU2 (non-gated):
    FC1:     A x W1^T                  (single FP4 block-scaled GEMM)
    Act:     max(0, x)^2               (squared ReLU activation)
  Common:
    Quant:   intermediate -> FP4      (cooperative quantization into shared A)
    FC2:     sweep all output tiles   (reuse the cached intermediate slice)
    Scatter: route-major BF16 store   (one private contribution per route)
    Finalize: token-major FP32 sum    (top-k contributions, one BF16 output)

What changes relative to the old split path:
  the compute launch used to expect the frontend to have already produced:
    - expert row counts
    - expert-major packed A
    - token_map / token_weights
  static.py builds those GPU-side before entering the same grouped compute
  schedule. That is why this file owns the resident-grid barriers and the
  route/pack bookkeeping itself.

Work decomposition
  Frontend:
    One CTA leader handles one routed pair at a time. It atomically appends a
    row to row_counts[expert], writes the source token + router weight, then
    quantizes the source token row into that expert-major destination row.
  Compute:
    The compact static work loop assigns (m_tile, intermediate_slice, expert).
    FC1 is computed once per slice, the slice is quantized into shared A, and
    FC2 sweeps all output tiles from that cached slice. FC1 cost is therefore
    amortized across every FC2 output tile.

Layouts and dataflow
  packed_a_storage:
    Flat uint8 backing store for expert-major FP4 activations.
    Logical view used by the compute path is [max_rows, K, E] fp4x2.
  scale_storage:
    Flat uint8 backing store for expert-major activation scale factors laid
    out in the CUTLASS/CuTe block-scaled MMA layout expected by the compute
    mainloop.
  token_map / token_weights:
    Expert-row metadata used by the FC2 scatter path to accumulate the final
    output directly into [num_tokens, K].

Why the barriers exist
  row_counts drives the grouped scheduler shape. The compute phase cannot begin
  until every routed pair has claimed its expert row and packed A/scales have
  been written. The static kernel therefore uses a resident-grid barrier between
  route/pack and compute instead of the host-side sequencing used previously.

Scale-contract note
  This kernel supports per-expert FC1 activation scales by quantizing each
  routed pair with input_global_scale[expert]. That is checkpoint-correct for
  models where gate/up input scales vary across experts.

Design boundary
  The static kernel is the compact decode backend. It keeps route/pack and
  compute in one resident launch for small routed working sets, and relies on
  the resident-grid barrier between those phases instead of overlapping them.
  Large routed workloads dispatch to the dynamic backend instead.
"""

from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from cutlass.cutlass_dsl import (
    Int32,
    Integer,
    Int64,
    Uint8,
    Uint64,
    T,
    dsl_user_op,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync

from flashinfer.cute_dsl.utils import (
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)
from flashinfer.cute_dsl.fp4_common import (
    atomic_add_global_i32,
    fabs_f32,
    fmax_f32,
    rcp_approx_ftz,
    quantize_block_fp4,
    quantize_block_fp4_fast,
    quantize_block_mxfp4,
    get_ptr_as_int64,
    get_smem_ptr_as_int32,
    st_global_f32,
    st_global_i32,
    shared_ptr_to_u32,
    st_shared_u8,
    st_global_u64,
)
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
    Sm120B12xBlockScaledDenseGemmKernel as DenseGemmKernel,
)
from .moe_activation import gated_activation_f32, is_gated_activation
from ._moe_dynamic.gated import (
    MoEGatedDynamicKernel,
    load_shared_bf16x8_to_f32x8,
    load_shared_i32_f32_pair,
)


_SF_VEC_SIZE = 16
_COMPACT_STATIC_TILE_M = 128


@dsl_user_op
def store_weighted_bf16x8_route(
    addr,
    smem_addr,
    route_weight,
    down_alpha_value,
    *,
    loc=None,
    ip=None,
):
    """Write one private weighted route contribution without a global REDG."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            route_weight.ir_value(loc=loc, ip=ip),
            down_alpha_value.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 p0,p1,p2,p3,w2; .reg .b16 w;"
        " .reg .f32 combined_scale;"
        " ld.shared.v4.u32 {p0,p1,p2,p3}, [$1];"
        " mul.rn.f32 combined_scale, $2, $3;"
        " cvt.rn.bf16.f32 w, combined_scale;"
        " mov.b32 w2, {w,w};"
        " mul.rn.bf16x2 p0, p0, w2;"
        " mul.rn.bf16x2 p1, p1, w2;"
        " mul.rn.bf16x2 p2, p2, w2;"
        " mul.rn.bf16x2 p3, p3, w2;"
        " st.global.v4.u32 [$0], {p0,p1,p2,p3}; }",
        "l,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_global_bf16x8_to_f32x8(addr: Int64, *, loc=None, ip=None):
    """Load one route-major BF16x8 contribution as eight FP32 values."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 p0, p1, p2, p3;
            .reg .b16 b0, b1, b2, b3, b4, b5, b6, b7;
            ld.global.v4.u32 {p0, p1, p2, p3}, [$8];
            mov.b32 {b0, b1}, p0;
            mov.b32 {b2, b3}, p1;
            mov.b32 {b4, b5}, p2;
            mov.b32 {b6, b7}, p3;
            cvt.f32.bf16 $0, b0;
            cvt.f32.bf16 $1, b1;
            cvt.f32.bf16 $2, b2;
            cvt.f32.bf16 $3, b3;
            cvt.f32.bf16 $4, b4;
            cvt.f32.bf16 $5, b5;
            cvt.f32.bf16 $6, b6;
            cvt.f32.bf16 $7, b7;
        }
        """,
        "=f,=f,=f,=f,=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [idx], loc=loc, ip=ip))
        for idx in range(8)
    )


@dsl_user_op
def store_global_f32x8_as_bf16(
    addr,
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    *,
    loc=None,
    ip=None,
):
    """Round one finalized FP32x8 vector once and store BF16x8."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            v0.ir_value(loc=loc, ip=ip),
            v1.ir_value(loc=loc, ip=ip),
            v2.ir_value(loc=loc, ip=ip),
            v3.ir_value(loc=loc, ip=ip),
            v4.ir_value(loc=loc, ip=ip),
            v5.ir_value(loc=loc, ip=ip),
            v6.ir_value(loc=loc, ip=ip),
            v7.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 p0,p1,p2,p3;"
        " cvt.rn.satfinite.bf16x2.f32 p0, $2, $1;"
        " cvt.rn.satfinite.bf16x2.f32 p1, $4, $3;"
        " cvt.rn.satfinite.bf16x2.f32 p2, $6, $5;"
        " cvt.rn.satfinite.bf16x2.f32 p3, $8, $7;"
        " st.global.v4.u32 [$0], {p0,p1,p2,p3}; }",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _compact_static_get_single_m_tile_work(
    row_counts,
    active_expert_count,
    *,
    tile_m,
    num_tiles_n,
    cluster_shape_mn,
    current_work_linear_idx,
    current_local_expert_idx,
    accum_tile_m,
    cta_id_in_cluster,
):
    """O(1) scheduler for compact experts whose routed rows fit one M tile.

    The retained2 35B small-row path has two N-pair tasks per compact expert
    and at most one M tile.  Keeping the generic signature makes this helper a
    drop-in experiment while eliminating repeated row-count scans.
    """
    num_active_experts = active_expert_count[Int32(0)]
    local_expert_idx = current_work_linear_idx // num_tiles_n
    is_valid = local_expert_idx < num_active_experts
    cur_tile_coord = (
        cta_id_in_cluster[0],
        (current_work_linear_idx % num_tiles_n) * cluster_shape_mn[1]
        + cta_id_in_cluster[1],
        local_expert_idx,
    )
    return cur_tile_coord, is_valid, local_expert_idx, local_expert_idx


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
def _st_shared_f32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_shared_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ld_global_u64(addr, *, loc=None, ip=None):
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u64 $0, [$1];",
            "=l,l",
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


class MoEStaticKernel:
    """Compact retained static MoE kernel for both SM12x FP4 formats."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        activation: str = "silu",
        swiglu_alpha: float = 1.702,
        swiglu_beta: float = 1.0,
        swiglu_limit: float | None = None,
    ):
        if sf_vec_size not in (16, 32):
            raise ValueError(f"unsupported FP4 scale vector size {sf_vec_size}")
        if activation not in {"silu", "relu2", "gelu_tanh", "swigluoai_uninterleave"}:
            raise ValueError(f"unsupported activation {activation!r}")
        self._dense_cls = DenseGemmKernel
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.input_scales_are_reciprocal = input_scales_are_reciprocal
        self.activation = activation
        self.is_gated = is_gated_activation(activation)
        # relu2's squared outputs need the exact quantizer and scale math.
        self.fast_math = bool(fast_math) and self.is_gated
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        self.swiglu_limit = float(swiglu_limit) if swiglu_limit is not None else None
        # Both formats use one K128 retained slice.  NVFP4 has eight block-16
        # scales per slice; MXFP4 has four block-32 scales.
        tile_k = 128
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        # Packed-A itself can use the actual M64 tile.  Keep SFA at M128 below
        # because the block-scaled MMA scale layout is 128-row granular.
        self.sa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sa_tiles_per_block = self.sa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sfa_tiles_per_block = self.sfa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfb_tile_shape_nk = (max(128, mma_tiler_mn[1]), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // mma_tiler_mn[1]
        self.output_tile_count_n = output_tile_count_n
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.occupancy = 1
        self.num_mma_warps = 4
        self.num_frontend_warps = 10
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.compute_threads_per_cta = (
            self.num_mma_warps + 1
        ) * self.num_threads_per_warp
        self.threads_per_cta = self.num_frontend_warps * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.pass_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.compute_threads_per_cta,
        )
        self.load_register_requirement = 32
        self.mma_register_requirement = 232

    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFA(self, sfa_tensor, tiled_mma)

    def _thrfrg_SFB(self, sfb_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFB(self, sfb_tensor, tiled_mma)

    def _get_layoutSFA_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFA_TV(self, tiled_mma)  # type: ignore[arg-type]

    def _get_layoutSFB_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFB_TV(self, tiled_mma)  # type: ignore[arg-type]

    # The stage-selectable FC2 loader is format-independent.
    load_fc2_a_fragments = MoEGatedDynamicKernel.load_fc2_a_fragments

    @cute.jit
    def quantize_q1_sC_to_sA_sSFA(
        self,
        tidx,
        valid_rows: Int32,
        task_expert_idx: Int32,
        global_scale: cute.Tensor,
        sC: cute.Tensor,
        sA: cute.Tensor,
        fc1_tRS_sD: cute.Tensor,
        sfa_base_addr: Int32,
        sfa_stage_elements: Int32,
        q1_stage_idx: Int32,
        epi_rest_m,
    ):
        """Quantize one retained FC1 slice for the FC2 A operand."""
        sA_u8 = cute.recast_tensor(sA[None, None, q1_stage_idx], cutlass.Uint8)
        packed_cols = Int32(self.tile_shape_mnk[2] // 2)
        sf_blocks_per_row = Int32(self.tile_shape_mnk[2] // self.sf_vec_size)
        gs_value = cutlass.Float32(1.0)
        if cutlass.const_expr(self.sf_vec_size == 16):
            gs_value = global_scale[task_expert_idx].to(cutlass.Float32)
            if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(0.0):
                if self.fast_math:
                    gs_value = rcp_approx_ftz(gs_value)
                else:
                    gs_value = cutlass.Float32(1.0) / gs_value

        for epi_m in cutlass.range_constexpr(epi_rest_m):
            epi_m_valid = valid_rows - Int32(epi_m) * Int32(self.epi_tile[0])
            gated_epi_buffer = Int32(epi_m) % cute.size(fc1_tRS_sD, mode=[3])
            if epi_m_valid > Int32(0):
                rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                epi_rows = epi_m_valid
                if epi_rows > Int32(self.epi_tile[0]):
                    epi_rows = Int32(self.epi_tile[0])
                if epi_rows < Int32(0):
                    epi_rows = Int32(0)
                quant_idx = Int32(tidx)
                while quant_idx < epi_rows * sf_blocks_per_row:
                    local_row = quant_idx // sf_blocks_per_row
                    row = rows_offset + local_row
                    sf_block = quant_idx - local_row * sf_blocks_per_row
                    block_start = sf_block * Int32(self.sf_vec_size)

                    values = cute.make_rmem_tensor((self.sf_vec_size,), cutlass.Float32)
                    block_max = cutlass.Float32(0.0)
                    for load_idx in cutlass.range_constexpr(self.sf_vec_size // 8):
                        sc_element_offset = Int32(
                            sC.layout(
                                (
                                    local_row,
                                    block_start + Int32(load_idx * 8),
                                    gated_epi_buffer,
                                )
                            )
                        )
                        sc_element_offset = sc_element_offset ^ (
                            (sc_element_offset & Int32(0x1C0)) >> Int32(3)
                        )
                        loaded = load_shared_bf16x8_to_f32x8(
                            get_smem_ptr_as_int32(sC, sc_element_offset)
                        )
                        for elem_idx in cutlass.range_constexpr(8):
                            value = loaded[elem_idx]
                            values[load_idx * 8 + elem_idx] = value
                            block_max = fmax_f32(block_max, fabs_f32(value))

                    packed_lo = Uint64(0)
                    packed_hi = Uint64(0)
                    scale_byte = Uint8(0)
                    if cutlass.const_expr(self.sf_vec_size == 32):
                        packed_lo, packed_hi, scale_byte = quantize_block_mxfp4(
                            values, block_max
                        )
                    else:
                        if self.fast_math:
                            packed_lo, scale_byte = quantize_block_fp4_fast(
                                values, block_max, gs_value
                            )
                        else:
                            packed_lo, scale_byte = quantize_block_fp4(
                                values, block_max, gs_value
                            )

                    packed_base = sf_block * Int32(self.sf_vec_size // 2)
                    dst_pcol = row & Int32(63)
                    xor_bits = ((dst_pcol >> Int32(1)) & Int32(0x3)) << Int32(4)
                    row_high = row >> Int32(6)
                    for byte_idx in cutlass.range_constexpr(self.sf_vec_size // 2):
                        src_pcol = packed_base + Int32(byte_idx)
                        dst_row = ((src_pcol ^ xor_bits) << Int32(1)) + row_high
                        dst_flat = dst_row * packed_cols + dst_pcol
                        if cutlass.const_expr(byte_idx < 8):
                            byte_val = Uint8(
                                (packed_lo >> Uint64(byte_idx * 8)) & Uint64(0xFF)
                            )
                        else:
                            byte_val = Uint8(
                                (packed_hi >> Uint64((byte_idx - 8) * 8)) & Uint64(0xFF)
                            )
                        sA_u8[dst_flat] = byte_val

                    outer_m_idx = row % Int32(32)
                    inner_m_idx = row // Int32(32)
                    inner_k_idx = sf_block % Int32(4)
                    k_tile_idx = sf_block // Int32(4)
                    sf_raw_idx = (
                        k_tile_idx * Int32(32 * 4 * 4)
                        + outer_m_idx * Int32(4 * 4)
                        + inner_m_idx * Int32(4)
                        + inner_k_idx
                    )
                    st_shared_u8(
                        sfa_base_addr + q1_stage_idx * sfa_stage_elements + sf_raw_idx,
                        scale_byte,
                    )
                    quant_idx += Int32(self.num_mma_warps * self.num_threads_per_warp)
        return

    @cute.jit
    def _load_valid_fc1_a_stage(
        self,
        tidx,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        local_expert_idx: Int32,
        tile_m_base: Int32,
        shared_row_base: Int32,
        valid_tile_rows: Int32,
        k_tile: Int32,
        stage_idx: Int32,
        max_rows: Int32,
        output_bytes_per_row: Int32,
        expert_scale_stride: Int32,
        num_k_tiles: Int32,
        sA: cute.Tensor,
        sfa_base_addr: Int32,
        sfa_stage_elements: Int32,
    ):
        # One K128 stage contains eight FP4x16 or four FP4x32 blocks.
        # Copy only live rows; invalid physical-M128 rows are never consumed by
        # scatter, so they need neither a global read nor an explicit clear.
        blocks_per_stage = Int32(self.tile_shape_mnk[2] // self.sf_vec_size)
        bytes_per_block = Int32(self.sf_vec_size // 2)
        packed_cols = Int32(self.tile_shape_mnk[2] // 2)
        sA_u8 = cute.recast_tensor(sA[None, None, stage_idx], cutlass.Uint8)
        quant_idx = Int32(tidx)
        while quant_idx < valid_tile_rows * blocks_per_stage:
            local_row = quant_idx // blocks_per_stage
            local_block = quant_idx - local_row * blocks_per_stage
            global_row = tile_m_base + local_row
            global_block = k_tile * blocks_per_stage + local_block
            packed_lo = _ld_global_u64(
                get_ptr_as_int64(
                    packed_a_storage,
                    local_expert_idx * max_rows * output_bytes_per_row
                    + global_row * output_bytes_per_row
                    + global_block * bytes_per_block,
                )
            )
            packed_hi = Uint64(0)
            if cutlass.const_expr(self.sf_vec_size == 32):
                packed_hi = _ld_global_u64(
                    get_ptr_as_int64(
                        packed_a_storage,
                        local_expert_idx * max_rows * output_bytes_per_row
                        + global_row * output_bytes_per_row
                        + global_block * bytes_per_block
                        + Int32(8),
                    )
                )

            row = shared_row_base + local_row
            packed_base = local_block * bytes_per_block
            dst_pcol = row & Int32(63)
            xor_bits = ((dst_pcol >> Int32(1)) & Int32(0x3)) << Int32(4)
            row_high = row >> Int32(6)
            for byte_idx in cutlass.range_constexpr(self.sf_vec_size // 2):
                src_pcol = packed_base + Int32(byte_idx)
                dst_row = ((src_pcol ^ xor_bits) << Int32(1)) + row_high
                dst_flat = dst_row * packed_cols + dst_pcol
                if cutlass.const_expr(byte_idx < 8):
                    byte_val = Uint8((packed_lo >> Uint64(byte_idx * 8)) & Uint64(0xFF))
                else:
                    byte_val = Uint8(
                        (packed_hi >> Uint64((byte_idx - 8) * 8)) & Uint64(0xFF)
                    )
                sA_u8[dst_flat] = byte_val

            global_m_tile = global_row // Int32(32 * 4)
            global_k_tile = global_block // Int32(4)
            global_outer_m = global_row % Int32(32)
            global_inner_m = (global_row % Int32(32 * 4)) // Int32(32)
            global_inner_k = global_block % Int32(4)
            scale_byte = scale_storage[
                local_expert_idx * expert_scale_stride
                + global_m_tile * num_k_tiles * Int32(32 * 4 * 4)
                + global_k_tile * Int32(32 * 4 * 4)
                + global_outer_m * Int32(4 * 4)
                + global_inner_m * Int32(4)
                + global_inner_k
            ]

            shared_outer_m = row % Int32(32)
            shared_inner_m = row // Int32(32)
            shared_inner_k = local_block % Int32(4)
            shared_k_tile = local_block // Int32(4)
            shared_sf_idx = (
                shared_k_tile * Int32(32 * 4 * 4)
                + shared_outer_m * Int32(4 * 4)
                + shared_inner_m * Int32(4)
                + shared_inner_k
            )
            st_shared_u8(
                sfa_base_addr + stage_idx * sfa_stage_elements + shared_sf_idx,
                scale_byte,
            )
            quant_idx += Int32(self.num_mma_warps * self.num_threads_per_warp)

        cute.arch.fence_proxy("async.shared", space="cta")
        self.epilog_sync_barrier.arrive_and_wait()

    def _make_a_smem_layout(self, ab_stage: int):
        import cutlass.utils.hopper_helpers as sm90_utils

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

    def _make_staged_layouts(self, ab_stage: int):
        (
            _,
            b_smem_staged,
            sfa_smem_staged,
            sfb_smem_staged,
            epi_smem_staged,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            ab_stage,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )
        a_smem_staged = self._make_a_smem_layout(ab_stage)
        return (
            a_smem_staged,
            b_smem_staged,
            sfa_smem_staged,
            sfb_smem_staged,
            epi_smem_staged,
        )

    def _shared_storage_size_bytes(
        self,
        a_smem_staged,
        b_smem_staged,
        sfa_smem_staged,
        sfb_smem_staged,
        epi_smem_staged,
    ) -> int:
        def _align_up(value: int, align: int) -> int:
            return ((value + align - 1) // align) * align

        # Gate and up share one transaction barrier in the paired FC1 path.
        # The payload still has separate B/SFB buffers, but A/SFA is fetched
        # only once and both branch operands become visible atomically.
        pipeline_count = 2
        offset = (
            8 * 4
            + pipeline_count * (self.ab_stage * 2 * 8)
            + _COMPACT_STATIC_TILE_M * 4
            + _COMPACT_STATIC_TILE_M * 4
        )
        buffers = [
            cute.size_in_bytes(self.a_dtype, a_smem_staged),
            cute.size_in_bytes(self.b_dtype, b_smem_staged),
            cute.size_in_bytes(self.sf_dtype, sfa_smem_staged),
            cute.size_in_bytes(self.sf_dtype, sfb_smem_staged),
            cute.size_in_bytes(cutlass.BFloat16, epi_smem_staged),
        ]
        if self.is_gated:
            buffers.insert(2, cute.size_in_bytes(self.b_dtype, b_smem_staged))
            buffers.insert(5, cute.size_in_bytes(self.sf_dtype, sfb_smem_staged))
        offset = _align_up(offset, self.buffer_align_bytes)
        for idx, size in enumerate(buffers):
            offset += size
            if idx + 1 != len(buffers):
                offset = _align_up(offset, self.buffer_align_bytes)
        return offset

    def _setup_attributes(self, hidden_size: int):
        import cutlass.utils.blackwell_helpers as sm120_utils

        self._hidden_size = hidden_size

        if self.sf_vec_size == 32:
            mma_op = cute.nvgpu.warp.MmaMXF4Op(
                self.a_dtype,
                self.acc_dtype,
                self.sf_dtype,
            )
        else:
            mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
                self.a_dtype,
                self.acc_dtype,
                self.sf_dtype,
            )
        atom_layout = cute.make_layout((2, 2, 1))
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk,
            self.sf_vec_size,
            False,
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        self.mma_atom = cute.make_mma_atom(mma_op)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_m_tiles = self.tile_shape_mnk[0] // (16 * 4)
        self.num_n_tiles = self.tile_shape_mnk[1] // (8 * 2)
        self.num_k_blocks = self.tile_shape_mnk[2] // 64

        sfa_smem = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        sfb_smem = sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        self.ab_stage, self.epi_stage = self._dense_cls._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            sfa_smem,
            sfb_smem,
            self.epi_tile,
            cutlass.BFloat16,
            self.smem_capacity,
            self.occupancy,
        )
        # M64 leaves enough shared memory for a third A/Gate/Up stage.  Keep
        # the pipeline state continuous across retained slices and work items;
        # it does not need to return to stage zero after every 16 K tiles.
        self.ab_stage = max(1, min(self.ab_stage, 3))
        self.epi_stage = 1
        while True:
            (
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.sfa_smem_layout_staged,
                self.sfb_smem_layout_staged,
                self.epi_smem_layout_staged,
            ) = self._make_staged_layouts(self.ab_stage)
            if (
                self._shared_storage_size_bytes(
                    self.a_smem_layout_staged,
                    self.b_smem_layout_staged,
                    self.sfa_smem_layout_staged,
                    self.sfb_smem_layout_staged,
                    self.epi_smem_layout_staged,
                )
                <= self.smem_capacity
                or self.ab_stage <= 1
            ):
                break
            self.ab_stage -= 1

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
    def __call__(
        self,
        a_input: cute.Tensor,  # [num_tokens, K] bf16
        topk_ids: cute.Tensor,  # [num_tokens * topk] int32
        topk_weights: cute.Tensor,  # [num_tokens * topk] float32
        packed_a: cute.Tensor,  # [max_rows, K, E] fp4x2 view for compute
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,  # flat uint8 backing packed_a
        scale_storage: cute.Tensor,  # flat uint8 backing sfa_ptr
        barrier_count: cute.Tensor,  # [1] int32 (host-zeroed)
        barrier_epoch: cute.Tensor,  # [1] int32 (host-zeroed)
        b_w13: cute.Tensor,  # [2*I_tp, K, E] — concatenated gate+up
        sfb_w13_ptr: cute.Pointer,  # scale factors for concatenated w13
        b_down: cute.Tensor,  # [K, I_tp, E]
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,  # [state_E] routed rows per local expert
        active_expert_count: cute.Tensor,  # [1] active expert count
        weight_expert_ids: cute.Tensor,  # [E] local expert id -> global weight expert id
        global_to_local_expert: cute.Tensor,  # [weight_E] global expert id -> local expert id
        virt_route_scratch: cute.Tensor,  # [weight_E*(1+max_chunks)] row alloc + chunk map
        input_global_scale: cute.Tensor,  # [E] per-expert FC1 input scale
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
        self.b_dtype = b_w13.element_type
        self.sf_dtype = sfa_ptr.dtype
        self.a_layout = utils.LayoutEnum.from_tensor(packed_a)
        self.b_layout = utils.LayoutEnum.from_tensor(b_w13)
        # Compact static always scatters into token-major row-major output.
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        hidden_size = a_input.shape[1]
        self._setup_attributes(hidden_size=hidden_size)

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            packed_a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # Single SF tensor for concatenated w13 (gate+up scale factors)
        sfb_w13_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_w13.shape, self.sf_vec_size
        )
        sfb_w13_tensor = cute.make_tensor(sfb_w13_ptr, sfb_w13_layout)

        # TMA descriptors
        tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
            packed_a,
            self.a_smem_layout_staged,
            self.sa_tile_shape_mk,
            1,
        )
        tma_sfa, gSFA = self._dense_cls._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            self.sfa_tile_shape_mk,
            1,
            internal_type=cutlass.Int16,
        )
        # Single TMA descriptor over concatenated w13 [2*I_tp, K, E].
        # Up tiles at N=0..I_tp/tile_N-1, gate tiles at N=I_tp/tile_N..2*I_tp/tile_N-1.
        tma_b_w13, gB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            b_w13,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_sfb_w13, gSFB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_w13_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )
        # B_down TMA
        sfb_down_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_down.shape, self.sf_vec_size
        )
        sfb_down_tensor = cute.make_tensor(sfb_down_ptr, sfb_down_layout)
        tma_b_down, gB_down = self._dense_cls._make_tma_atoms_and_tensors(
            b_down,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_sfb_down, gSFB_down = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_down_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )

        # Compact static schedules over (m_tile, intermediate_slice, local_expert_idx).
        grid = (*self.cluster_shape_mn, max_active_clusters)
        self.kernel(
            a_input,
            topk_ids,
            topk_weights,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            tma_a,
            gA,
            tma_sfa,
            gSFA,
            tma_b_w13,
            gB_w13,
            tma_sfb_w13,
            gSFB_w13,
            tma_b_down,
            gB_down,
            tma_sfb_down,
            gSFB_down,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            row_counts,
            active_expert_count,
            weight_expert_ids,
            global_to_local_expert,
            virt_route_scratch,
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
            # A regular launch beside other stream work can admit only part
            # of the grid, deadlocking the software grid barriers below.
            cooperative=True,
            stream=stream,
        )

        # The first kernel's completion on this stream is the global handoff;
        # no resident-grid spin barrier is needed for finalization.
        # w13 rows cover both branches only for gated activations; the
        # non-gated FC1 slice count is the full w13 tile count.  This must
        # match the device-side gate_tile_cnt/retained_group_count, or the
        # finalize sums the wrong route-scratch slots.
        fc1_branch_rows = b_w13.shape[0] // 2 if self.is_gated else b_w13.shape[0]
        gate_tile_count = fc1_branch_rows // self.tile_shape_mnk[1]
        retained_group_count = (gate_tile_count + 1) // 2
        final_vec_count = (a_input.shape[0] * hidden_size) // 8
        final_grid_z = (final_vec_count + 255) // 256
        final_grid = (1, 1, final_grid_z)
        self.finalize_kernel(
            packed_a_storage,
            scatter_output,
            topk_ids,
            retained_group_count,
        ).launch(
            grid=final_grid,
            block=[256, 1, 1],
            cluster=[1, 1, 1],
            cooperative=False,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_b_w13: cute.CopyAtom,
        mB_w13: cute.Tensor,
        tma_sfb_w13: cute.CopyAtom,
        mSFB_w13: cute.Tensor,
        tma_b_down: cute.CopyAtom,
        mB_down: cute.Tensor,
        tma_sfb_down: cute.CopyAtom,
        mSFB_down: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
        epi_smem_staged: cute.ComposedLayout,
        row_counts: cute.Tensor,
        active_expert_count: cute.Tensor,
        weight_expert_ids: cute.Tensor,
        global_to_local_expert: cute.Tensor,
        virt_route_scratch: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
    ):
        """Kernel entry point."""
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        _, _, gdim_z = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_id = Int32(tidx) & Int32(31)
        is_cta_leader = Int32(Int32(tidx) == Int32(0))

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_sfa)
            cpasync.prefetch_descriptor(tma_b_w13)
            cpasync.prefetch_descriptor(tma_sfb_w13)
            cpasync.prefetch_descriptor(tma_b_down)
            cpasync.prefetch_descriptor(tma_sfb_down)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

        b_smem_one = cute.slice_(b_smem_staged, (None, None, 0))
        sfb_smem_one = cute.slice_(sfb_smem_staged, (None, None, 0))
        # A/SFA are copied by the MMA warps from valid rows only.
        # The TMA completion barrier therefore tracks only B/SFB payloads.
        tma_copy_bytes = cute.size_in_bytes(
            self.b_dtype, b_smem_one
        ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)
        if cutlass.const_expr(self.is_gated):
            tma_copy_bytes += cute.size_in_bytes(
                self.b_dtype, b_smem_one
            ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)
        phase2_tma_copy_bytes = cute.size_in_bytes(
            self.b_dtype, b_smem_one
        ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class StorageGated:
            ctrl: cute.struct.MemRange[cutlass.Int32, 8]
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            phase2_pipeline_array: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            scatter_meta_cache: cute.struct.MemRange[
                cutlass.Int32, _COMPACT_STATIC_TILE_M * 2
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
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB_up: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class StorageRelu2:
            ctrl: cute.struct.MemRange[cutlass.Int32, 8]
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            phase2_pipeline_array: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            scatter_meta_cache: cute.struct.MemRange[
                cutlass.Int32, _COMPACT_STATIC_TILE_M * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
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
        phase2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=phase2_tma_copy_bytes,
            barrier_storage=storage.phase2_pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )

        cute.arch.sync_threads()

        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sB_up = (
            storage.sB_up.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
            if self.is_gated
            else sB
        )
        cute.recast_tensor(sA, cutlass.Uint8)
        cute.recast_tensor(sB, cutlass.Uint8)
        cute.recast_tensor(sB_up, cutlass.Uint8)
        sSFA = storage.sSFA.get_tensor(sfa_smem_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_staged)
        sSFB_up = storage.sSFB_up.get_tensor(sfb_smem_staged) if self.is_gated else sSFB
        cute.recast_tensor(sSFA, cutlass.Uint8)
        cute.recast_tensor(sSFB, cutlass.Uint8)
        cute.recast_tensor(sSFB_up, cutlass.Uint8)
        sC = storage.sC.get_tensor(
            epi_smem_staged.outer,
            swizzle=epi_smem_staged.inner,
        )
        sfa_base_addr = shared_ptr_to_u32(storage.sSFA.data_ptr())
        sfa_smem_one = cute.slice_(sfa_smem_staged, (None, None, 0))
        sfa_stage_elements = Int32(cute.cosize(sfa_smem_one))
        ctrl_base_addr = shared_ptr_to_u32(storage.ctrl.data_ptr())
        scatter_meta_base_addr = shared_ptr_to_u32(
            storage.scatter_meta_cache.data_ptr()
        )

        num_tokens = Int32(a_input.shape[0])
        cols = Int32(a_input.shape[1])
        num_experts = Int32(row_counts.shape[0])
        sf_blocks_per_row = cols // Int32(self.sf_vec_size)
        output_bytes_per_row = cols // Int32(2)
        max_rows = Int32(token_map.shape[1])
        total_pairs = Int32(topk_ids.shape[0])
        num_topk = total_pairs // num_tokens
        expert_scale_stride = Int32(scale_storage.shape[0]) // num_experts
        num_global_experts = Int32(global_to_local_expert.shape[0])
        flat_tid = Int32(bidz) * Int32(self.threads_per_cta) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(self.threads_per_cta)
        num_k_tiles = (cols + Int32(63)) // Int32(64)

        # Phase 0: cooperative init.  The finalizer overwrites every output
        # element, so unlike REDG scatter this path needs no output clear.
        i = flat_tid
        while i < num_experts:
            row_counts[i] = Int32(0)
            i += flat_stride
        i = flat_tid
        while i < num_global_experts:
            global_to_local_expert[i] = Int32(-1)
            i += flat_stride
        # virt_route_scratch layout:
        #   [row allocator: weight_E][chunk->local map: weight_E*max_chunks]
        #   [work-claim counter + pad: 8]
        virt_scratch_total = Int32(virt_route_scratch.shape[0])
        claim_slot = virt_scratch_total - Int32(8)
        max_chunks = claim_slot // num_global_experts - Int32(1)
        i = flat_tid
        while i < virt_scratch_total:
            scratch_init = Int32(-1)
            if i < num_global_experts:
                scratch_init = Int32(0)
            if i >= claim_slot:
                scratch_init = Int32(0)
            if i == claim_slot:
                # The first two waves stay statically assigned (CTA b takes
                # linear idx b and b+gdim_z); the counter hands out every
                # task after that. Early waves are inherently balanced, so
                # claiming them only adds an atomic round-trip on the fetch
                # path — measurable at tiny M (M16 +2.9us).
                scratch_init = Int32(2) * Int32(gdim_z)
            virt_route_scratch[i] = scratch_init
            i += flat_stride
        if flat_tid == Int32(0):
            active_expert_count[Int32(0)] = Int32(0)
        cute.arch.sync_threads()
        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        pair_idx = Int32(bidz) * Int32(self.num_frontend_warps) + warp_idx
        while pair_idx < total_pairs:
            expert_id = topk_ids[pair_idx].to(Int32)
            token_idx = pair_idx // num_topk
            weight = topk_weights[pair_idx].to(cutlass.Float32)
            local_expert_id = Int32(0)
            row = Int32(0)
            if lane_id == Int32(0):
                # 32-row virtual-expert split: a monotone per-global-expert
                # allocator yields (chunk, row-in-chunk); every chunk claims
                # its own compact local expert id, so each scheduled tile
                # holds <=32 valid rows — exactly the one M step the tile64
                # MMA computes. Big experts become several tiles instead of
                # silently losing rows past 32 (or 64 at tile128).
                alloc_row = atomic_add_global_i32(
                    get_ptr_as_int64(virt_route_scratch, expert_id),
                    Int32(1),
                )
                chunk = alloc_row >> Int32(5)
                row = alloc_row & Int32(31)
                vslot = num_global_experts + expert_id * max_chunks + chunk
                prior_local_expert_id = _atomic_cas_global_i32(
                    get_ptr_as_int64(virt_route_scratch, vslot),
                    Int32(-1),
                    Int32(-2),
                )
                if prior_local_expert_id == Int32(-1):
                    local_expert_id = atomic_add_global_i32(
                        get_ptr_as_int64(active_expert_count, Int32(0)),
                        Int32(1),
                    )
                    weight_expert_ids[local_expert_id] = expert_id
                    if chunk == Int32(0):
                        # Keep the legacy global->local mapping for chunk 0;
                        # the micro pre-pass and diagnostics still read it.
                        _st_global_release_i32(
                            get_ptr_as_int64(global_to_local_expert, expert_id),
                            local_expert_id,
                        )
                    _st_global_release_i32(
                        get_ptr_as_int64(virt_route_scratch, vslot),
                        local_expert_id,
                    )
                else:
                    if prior_local_expert_id == Int32(-2):
                        _spin_wait_global_eq_i32(
                            get_ptr_as_int64(virt_route_scratch, vslot),
                            Int32(-2),
                        )
                        prior_local_expert_id = _ld_global_acquire_i32(
                            get_ptr_as_int64(virt_route_scratch, vslot),
                        )
                    local_expert_id = prior_local_expert_id
                atomic_add_global_i32(
                    get_ptr_as_int64(row_counts, local_expert_id),
                    Int32(1),
                )
                map_idx = local_expert_id * max_rows + row
                # Preserve the unique routed-pair id.  Compute only needs its
                # token quotient, while scatter needs a collision-free scratch
                # row for the later token-major finalization.
                st_global_i32(get_ptr_as_int64(token_map, map_idx), pair_idx)
                st_global_f32(get_ptr_as_int64(token_weights, map_idx), weight)
            local_expert_id = cute.arch.shuffle_sync(local_expert_id, Int32(0))
            row = cute.arch.shuffle_sync(row, Int32(0))

            # Distribute quantization across ALL CTA threads, not just leader.
            # Each FP4 block (16 elements) is independent — perfect parallelism.
            gs_value = cutlass.Float32(1.0)
            if cutlass.const_expr(self.sf_vec_size == 16):
                gs_value = input_global_scale[expert_id].to(cutlass.Float32)
                if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(
                    0.0
                ):
                    if self.fast_math:
                        gs_value = rcp_approx_ftz(gs_value)
                    else:
                        gs_value = cutlass.Float32(1.0) / gs_value
            sf_idx = lane_id
            while sf_idx < sf_blocks_per_row:
                block_start = sf_idx * Int32(self.sf_vec_size)
                values = cute.make_rmem_tensor((self.sf_vec_size,), cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                # Vectorized BF16 loads cover either one 16-value NVFP4 block
                # or one 32-value MXFP4 block.
                block_addr = get_ptr_as_int64(a_input, token_idx * cols + block_start)
                for load_idx in cutlass.range_constexpr(self.sf_vec_size // 8):
                    loaded = load_global_bf16x8_to_f32x8(
                        block_addr + Int64(load_idx * 16)
                    )
                    for elem_idx in cutlass.range_constexpr(8):
                        value = loaded[elem_idx]
                        values[load_idx * 8 + elem_idx] = value
                        block_max = fmax_f32(block_max, fabs_f32(value))
                packed_lo = Uint64(0)
                packed_hi = Uint64(0)
                scale_byte = Uint8(0)
                if cutlass.const_expr(self.sf_vec_size == 32):
                    packed_lo, packed_hi, scale_byte = quantize_block_mxfp4(
                        values, block_max
                    )
                else:
                    if self.fast_math:
                        packed_lo, scale_byte = quantize_block_fp4_fast(
                            values, block_max, gs_value
                        )
                    else:
                        packed_lo, scale_byte = quantize_block_fp4(
                            values, block_max, gs_value
                        )

                output_offset = (
                    local_expert_id * max_rows * output_bytes_per_row
                    + row * output_bytes_per_row
                    + sf_idx * Int32(self.sf_vec_size // 2)
                )
                st_global_u64(
                    get_ptr_as_int64(packed_a_storage, output_offset), packed_lo
                )
                if cutlass.const_expr(self.sf_vec_size == 32):
                    st_global_u64(
                        get_ptr_as_int64(packed_a_storage, output_offset + Int32(8)),
                        packed_hi,
                    )

                m_tile_idx = row // Int32(32 * 4)
                k_tile_idx = sf_idx // Int32(4)
                outer_m_idx = row % Int32(32)
                inner_m_idx = (row % Int32(32 * 4)) // Int32(32)
                inner_k_idx = sf_idx % Int32(4)
                scale_offset = (
                    local_expert_id * expert_scale_stride
                    + m_tile_idx * num_k_tiles * Int32(32 * 4 * 4)
                    + k_tile_idx * Int32(32 * 4 * 4)
                    + outer_m_idx * Int32(4 * 4)
                    + inner_m_idx * Int32(4)
                    + inner_k_idx
                )
                scale_storage[scale_offset] = scale_byte
                sf_idx += Int32(32)

            pair_idx += Int32(gdim_z) * Int32(self.num_frontend_warps)

        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        gA = cute.local_tile(mA, self.sa_tile_shape_mk, (None, None, None))
        # Single tiled view over concatenated w13 [2*I_tp, K, E].
        # W13 is packed as [up, gate] across the concatenated N dimension.
        # Up tiles: N-indices 0..gate_tile_cnt-1
        # Gate tiles: N-indices gate_tile_cnt..2*gate_tile_cnt-1
        gB_w13_tiled = cute.local_tile(
            mB_w13,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFA = cute.local_tile(mSFA, self.sfa_tile_shape_mk, (None, None, None))
        gSFB_w13_tiled = cute.local_tile(
            mSFB_w13, self.sfb_tile_shape_nk, (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord[1]
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord[0]

        tAsA, tAgA = cpasync.tma_partition(
            tma_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # Single w13 TMA partition (gate+up concatenated)
        tBsB_w13, tBgB_w13 = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsB_w13_up, _ = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_up, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsSFB_w13, tBgSFB_w13 = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFB_w13_up, _ = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_up, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsB_w13_up = cute.filter_zeros(tBsB_w13_up)
        tBsSFB_w13 = cute.filter_zeros(tBsSFB_w13)
        tBgSFB_w13 = cute.filter_zeros(tBgSFB_w13)
        tBsSFB_w13_up = cute.filter_zeros(tBsSFB_w13_up)

        # B_down TMA partitions
        gB_down = cute.local_tile(
            mB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFB_down = cute.local_tile(
            mSFB_down, self.sfb_tile_shape_nk, (None, None, None)
        )
        tBsB_down, tBgB_down = cpasync.tma_partition(
            tma_b_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_down, 0, 2),
        )
        tBsSFB_down, tBgSFB_down = cpasync.tma_partition(
            tma_sfb_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_down, 0, 2),
        )
        tBsSFB_down = cute.filter_zeros(tBsSFB_down)
        tBgSFB_down = cute.filter_zeros(tBgSFB_down)

        # MMA fragment partitions
        tCsA_full = thr_mma.partition_A(sA)
        tCrA_full = tiled_mma.make_fragment_A(tCsA_full[None, None, None, 0])
        tCrSFA_full = self._dense_cls._partition_fragment_SFA(
            self,  # type: ignore[arg-type]
            sSFA[None, None, 0],
            thr_mma,
            tidx,
        )
        tCsB = thr_mma.partition_B(sB)
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCsB_up = thr_mma.partition_B(sB_up)
        tCrB_up = tiled_mma.make_fragment_B(tCsB_up[None, None, None, 0])
        tCrSFB_full = self._dense_cls._partition_fragment_SFB(
            self,  # type: ignore[arg-type]
            sSFB[None, None, 0],
            thr_mma,
            tidx,
        )

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
        # Gated: w13 has 2*I_tp/tile_N N-tiles. Gate = second half, up = first half.
        # ReLU2: w13 has I_tp/tile_N N-tiles. Single FC1 pass, no split.
        intermediate_tile_cnt = cute.size(gB_w13_tiled, mode=[2])
        gate_tile_cnt = (
            intermediate_tile_cnt // Int32(2)
            if self.is_gated
            else intermediate_tile_cnt
        )
        output_tile_cnt = cute.size(gB_down, mode=[2])
        retained_group_count = (gate_tile_cnt + Int32(1)) // Int32(2)
        # The wrapper reserves packed-A capacity for max_num_tokens, while the
        # compact static path consumes only the contiguous active-expert
        # prefix.  Use the high end as a private
        # [route, retained_group, K] BF16 contribution buffer.
        route_scratch_bytes = total_pairs * retained_group_count * cols * Int32(2)
        route_scratch_base = Int32(packed_a_storage.shape[0]) - route_scratch_bytes

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        phase2_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        phase2_cons_state = pipeline.make_pipeline_state(
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
            atom_ld_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.sf_dtype
            )
            smem_copy_SFA = cute.make_tiled_copy(
                atom_ld_SF,
                self._dense_cls._get_layoutSFA_TV(self, tiled_mma),  # type: ignore[arg-type]
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_copy_SFB = cute.make_tiled_copy(
                atom_ld_SF,
                self._dense_cls._get_layoutSFB_TV(self, tiled_mma),  # type: ignore[arg-type]
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_ld_A = smem_copy_A.get_slice(tidx)
            thr_ld_B = smem_copy_B.get_slice(tidx)
            csA_full = thr_ld_A.partition_S(sA)
            crA_full = thr_ld_A.retile(tCrA_full)
            csB = thr_ld_B.partition_S(sB)
            csB_up = thr_ld_B.partition_S(sB_up)
            crB = thr_ld_B.retile(tCrB)
            crB_up = thr_ld_B.retile(tCrB_up)

            thr_ld_SFA = smem_copy_SFA.get_slice(tidx)
            thr_ld_SFB = smem_copy_SFB.get_slice(tidx)
            csSFA_full = thr_ld_SFA.partition_S(sSFA)
            crSFA_full = thr_ld_SFA.retile(tCrSFA_full)
            csSFB_full = thr_ld_SFB.partition_S(sSFB)
            csSFB_up_full = thr_ld_SFB.partition_S(sSFB_up)
            crSFB_full = thr_ld_SFB.retile(tCrSFB_full)
            tCrSFB_up_full = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                sSFB_up[None, None, 0],
                thr_mma,
                tidx,
            )
            crSFB_up_full = thr_ld_SFB.retile(tCrSFB_up_full)

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
            published_work_linear_idx = Int32(bidz)
            tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                _compact_static_get_single_m_tile_work(
                    row_counts,
                    active_expert_count,
                    tile_m=Int32(self.tile_shape_mnk[0]),
                    num_tiles_n=Int32((self.output_tile_count_n + 1) // 2),
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
                weight_expert_idx = weight_expert_ids[local_expert_idx]
                alpha_value = alpha[weight_expert_idx].to(cutlass.Float32)
                valid_rows = row_counts[local_expert_idx]
                tile_m_base = tile_coord[0] * Int32(self.tile_shape_mnk[0])
                intermediate_slice = tile_coord[1] * Int32(2)
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
                sfa_tile_offset = tile_coord[0] % self.sfa_tiles_per_block
                if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                    sSFA_tile = cute.local_tile(
                        sSFA,
                        cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (sfa_tile_offset, 0, None),
                    )
                    csSFA_tile = thr_ld_SFA.partition_S(sSFA_tile)
                    tCrSFA_tile = self._dense_cls._partition_fragment_SFA(
                        self,  # type: ignore[arg-type]
                        sSFA_tile[None, None, 0],
                        thr_mma,
                        tidx,
                    )
                    crSFA_tile = thr_ld_SFA.retile(tCrSFA_tile)
                else:
                    csSFA_tile = csSFA_full
                    tCrSFA_tile = tCrSFA_full
                    crSFA_tile = crSFA_full
                sfb_tile_offset = intermediate_slice % self.sfb_tiles_per_block
                if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                    sSFB_tile = cute.local_tile(
                        sSFB,
                        cute.slice_(self.tile_shape_mnk, (0, None, None)),
                        (sfb_tile_offset, 0, None),
                    )
                    sSFB_up_tile = cute.local_tile(
                        sSFB_up,
                        cute.slice_(self.tile_shape_mnk, (0, None, None)),
                        (sfb_tile_offset, 0, None),
                    )
                    csSFB_tile = thr_ld_SFB.partition_S(sSFB_tile)
                    csSFB_up_tile = thr_ld_SFB.partition_S(sSFB_up_tile)
                    tCrSFB_tile = self._dense_cls._partition_fragment_SFB(
                        self,  # type: ignore[arg-type]
                        sSFB_tile[None, None, 0],
                        thr_mma,
                        tidx,
                    )
                    crSFB_tile = thr_ld_SFB.retile(tCrSFB_tile)
                    tCrSFB_up_tile = self._dense_cls._partition_fragment_SFB(
                        self,  # type: ignore[arg-type]
                        sSFB_up_tile[None, None, 0],
                        thr_mma,
                        tidx,
                    )
                    crSFB_up_tile = thr_ld_SFB.retile(tCrSFB_up_tile)
                else:
                    csSFB_tile = csSFB_full
                    csSFB_up_tile = csSFB_up_full
                    tCrSFB_tile = tCrSFB_full
                    crSFB_tile = crSFB_full
                    tCrSFB_up_tile = tCrSFB_up_full
                    crSFB_up_tile = crSFB_up_full
                valid_tile_rows = valid_rows - tile_m_base
                if valid_tile_rows > Int32(self.tile_shape_mnk[0]):
                    valid_tile_rows = Int32(self.tile_shape_mnk[0])
                if valid_tile_rows < Int32(0):
                    valid_tile_rows = Int32(0)

                cache_row = Int32(tidx)
                if cache_row < Int32(_COMPACT_STATIC_TILE_M):
                    route_idx = Int32(0)
                    wv = cutlass.Float32(0.0)
                    if cache_row < valid_tile_rows:
                        route_idx = token_map[
                            local_expert_idx, tile_m_base + cache_row
                        ].to(Int32)
                        wv = token_weights[
                            local_expert_idx, tile_m_base + cache_row
                        ].to(cutlass.Float32)
                    meta_addr = scatter_meta_base_addr + cache_row * Int32(8)
                    _st_shared_i32(meta_addr, route_idx)
                    _st_shared_f32(meta_addr + Int32(4), wv)
                self.epilog_sync_barrier.arrive_and_wait()

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

                down_alpha_value = down_alpha[weight_expert_idx].to(cutlass.Float32)
                down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

                epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
                MmaMPerEpiM = self.epi_tile[0] // mma_tile_m
                MmaNPerEpiN = self.epi_tile[1] // mma_tile_n

                # ============================================================
                # PHASE A: FC1 for this slice (gate + up)
                # ============================================================

                for retained_slice_idx in cutlass.range_constexpr(2):
                    # Paired Gate/Up GEMM.  One A/SFA load feeds both branches;
                    # independent B/SFB register fragments keep the two branch
                    # operands live across the back-to-back MMA cadence.
                    fz_crSFA = cute.filter_zeros(crSFA_tile)
                    fz_crSFB = cute.filter_zeros(crSFB_tile)
                    fz_crSFB_up = cute.filter_zeros(crSFB_up_tile)
                    gate_acc.fill(0.0)
                    if cutlass.const_expr(self.is_gated):
                        up_acc.fill(0.0)
                    cons_state.reset_count()
                    peek = ml_pipeline.consumer_try_wait(cons_state)
                    ml_pipeline.consumer_wait(cons_state, peek)
                    self._load_valid_fc1_a_stage(
                        tidx,
                        packed_a_storage,
                        scale_storage,
                        local_expert_idx,
                        tile_m_base,
                        sa_row_base,
                        valid_tile_rows,
                        Int32(0),
                        cons_state.index,
                        max_rows,
                        output_bytes_per_row,
                        expert_scale_stride,
                        num_k_tiles,
                        sA,
                        sfa_base_addr,
                        sfa_stage_elements,
                    )
                    csA_p = csA_tile[None, None, None, cons_state.index]
                    csB_p = csB[None, None, None, cons_state.index]
                    csB_up_p = csB_up[None, None, None, cons_state.index]
                    csSFA_p = csSFA_tile[None, None, None, cons_state.index]
                    csSFB_p = csSFB_tile[None, None, None, cons_state.index]
                    csSFB_up_p = csSFB_up_tile[None, None, None, cons_state.index]
                    cute.copy(
                        smem_copy_A, csA_p[None, None, 0], crA_tile[None, None, 0]
                    )
                    cute.copy(smem_copy_B, csB_p[None, None, 0], crB[None, None, 0])
                    if cutlass.const_expr(self.is_gated):
                        cute.copy(
                            smem_copy_B,
                            csB_up_p[None, None, 0],
                            crB_up[None, None, 0],
                        )
                    fz_csSFA_p = cute.filter_zeros(csSFA_p)
                    fz_csSFB_p = cute.filter_zeros(csSFB_p)
                    fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
                    cute.copy(
                        smem_copy_SFA,
                        fz_csSFA_p[None, None, 0],
                        fz_crSFA[None, None, 0],
                    )
                    cute.copy(
                        smem_copy_SFB,
                        fz_csSFB_p[None, None, 0],
                        fz_crSFB[None, None, 0],
                    )
                    if cutlass.const_expr(self.is_gated):
                        cute.copy(
                            smem_copy_SFB,
                            fz_csSFB_up_p[None, None, 0],
                            fz_crSFB_up[None, None, 0],
                        )
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
                                csA_p = csA_tile[None, None, None, cons_state.index]
                                csB_p = csB[None, None, None, cons_state.index]
                                csB_up_p = csB_up[None, None, None, cons_state.index]
                                csSFA_p = csSFA_tile[None, None, None, cons_state.index]
                                csSFB_p = csSFB_tile[None, None, None, cons_state.index]
                                csSFB_up_p = csSFB_up_tile[
                                    None, None, None, cons_state.index
                                ]
                                fz_csSFA_p = cute.filter_zeros(csSFA_p)
                                fz_csSFB_p = cute.filter_zeros(csSFB_p)
                                fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
                                ml_pipeline.consumer_wait(cons_state, peek)
                                self._load_valid_fc1_a_stage(
                                    tidx,
                                    packed_a_storage,
                                    scale_storage,
                                    local_expert_idx,
                                    tile_m_base,
                                    sa_row_base,
                                    valid_tile_rows,
                                    Int32(_k_tile + 1),
                                    cons_state.index,
                                    max_rows,
                                    output_bytes_per_row,
                                    expert_scale_stride,
                                    num_k_tiles,
                                    sA,
                                    sfa_base_addr,
                                    sfa_stage_elements,
                                )
                            for _mt in range(self.num_m_tiles):
                                for _nt in range(self.num_n_tiles):
                                    mma_atom.set(
                                        WarpField.SFA,
                                        tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                    )
                                    mma_atom.set(
                                        WarpField.SFB,
                                        tCrSFB_tile[None, _nt, k_block_idx].iterator,
                                    )
                                    cute.gemm(
                                        mma_atom,
                                        gate_acc[None, _mt, _nt],
                                        tCrA_tile[None, _mt, k_block_idx],
                                        tCrB[None, _nt, k_block_idx],
                                        gate_acc[None, _mt, _nt],
                                    )
                                    if cutlass.const_expr(self.is_gated):
                                        mma_atom.set(
                                            WarpField.SFB,
                                            tCrSFB_up_tile[
                                                None, _nt, k_block_idx
                                            ].iterator,
                                        )
                                        cute.gemm(
                                            mma_atom,
                                            up_acc[None, _mt, _nt],
                                            tCrA_tile[None, _mt, k_block_idx],
                                            tCrB_up[None, _nt, k_block_idx],
                                            up_acc[None, _mt, _nt],
                                        )
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
                            if cutlass.const_expr(self.is_gated):
                                cute.copy(
                                    smem_copy_B,
                                    csB_up_p[None, None, k_next],
                                    crB_up[None, None, k_next],
                                )
                            fz_csSFA_cur = cute.filter_zeros(
                                csSFA_tile[None, None, None, cons_state.index]
                            )
                            fz_csSFB_cur = cute.filter_zeros(
                                csSFB_tile[None, None, None, cons_state.index]
                            )
                            cute.copy(
                                smem_copy_SFA,
                                fz_csSFA_cur[None, None, k_next],
                                fz_crSFA[None, None, k_next],
                            )
                            cute.copy(
                                smem_copy_SFB,
                                fz_csSFB_cur[None, None, k_next],
                                fz_crSFB[None, None, k_next],
                            )
                            if cutlass.const_expr(self.is_gated):
                                fz_csSFB_up_cur = cute.filter_zeros(
                                    csSFB_up_tile[None, None, None, cons_state.index]
                                )
                                cute.copy(
                                    smem_copy_SFB,
                                    fz_csSFB_up_cur[None, None, k_next],
                                    fz_crSFB_up[None, None, k_next],
                                )
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )
                        if k_block_idx == num_k_blocks - 1:
                            ml_pipeline.consumer_release(cons_state)
                            cons_state.advance()
                        if k_next > 0 and fc1_k_tile_cnt > Int32(0):
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
                            if cutlass.const_expr(self.is_gated):
                                cute.copy(
                                    smem_copy_B,
                                    csB_up_p[None, None, k_next],
                                    crB_up[None, None, k_next],
                                )
                            cute.copy(
                                smem_copy_SFA,
                                fz_csSFA_p[None, None, k_next],
                                fz_crSFA[None, None, k_next],
                            )
                            cute.copy(
                                smem_copy_SFB,
                                fz_csSFB_p[None, None, k_next],
                                fz_crSFB[None, None, k_next],
                            )
                            if cutlass.const_expr(self.is_gated):
                                cute.copy(
                                    smem_copy_SFB,
                                    fz_csSFB_up_p[None, None, k_next],
                                    fz_crSFB_up[None, None, k_next],
                                )
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_tile[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    gate_acc[None, _mt, _nt],
                                    tCrA_tile[None, _mt, k_block_idx],
                                    tCrB[None, _nt, k_block_idx],
                                    gate_acc[None, _mt, _nt],
                                )
                                if cutlass.const_expr(self.is_gated):
                                    mma_atom.set(
                                        WarpField.SFB,
                                        tCrSFB_up_tile[None, _nt, k_block_idx].iterator,
                                    )
                                    cute.gemm(
                                        mma_atom,
                                        up_acc[None, _mt, _nt],
                                        tCrA_tile[None, _mt, k_block_idx],
                                        tCrB_up[None, _nt, k_block_idx],
                                        up_acc[None, _mt, _nt],
                                    )

                    # After the second FC1 has drained, the DMA warp may reuse the
                    # Gate B/SFB stages for FC2 while math warps materialize Q1.
                    if retained_slice_idx == 1:
                        self.pass_sync_barrier.arrive_and_wait()

                        # sC still holds the first slice.  FC1 no longer needs the
                        # two A/SFA pipeline stages, so quantize it into Stage0.
                        self.quantize_q1_sC_to_sA_sSFA(
                            tidx,
                            valid_tile_rows,
                            weight_expert_idx,
                            global_scale,
                            sC,
                            sA,
                            tRS_sD,
                            sfa_base_addr,
                            sfa_stage_elements,
                            Int32(0),
                            epi_rest_m,
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                    # Materialize this slice's activation in BF16 sC.  For slice0
                    # it remains resident while slice1 FC1 runs; for slice1 it is
                    # immediately quantized into Stage1 below.
                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        epi_m_valid = valid_tile_rows - Int32(epi_m) * Int32(
                            self.epi_tile[0]
                        )
                        silu_epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                        if epi_m_valid > Int32(0):
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
                                    if cutlass.const_expr(self.is_gated):
                                        up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size(tRS_rD_slice)
                                        ):
                                            g = alpha_value * gate_slice[elem_idx]
                                            u = alpha_value * up_slice[elem_idx]
                                            tRS_rD_slice[elem_idx] = (
                                                gated_activation_f32(
                                                    g,
                                                    u,
                                                    activation=self.activation,
                                                    limit=self.swiglu_limit,
                                                    alpha=self.swiglu_alpha,
                                                    beta=self.swiglu_beta,
                                                    fast_math=self.fast_math,
                                                )
                                            )
                                    else:
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size(tRS_rD_slice)
                                        ):
                                            g = alpha_value * gate_slice[elem_idx]
                                            relu_g = fmax_f32(g, cutlass.Float32(0.0))
                                            tRS_rD_slice[elem_idx] = relu_g * relu_g

                            acc_vec = tRS_rD.load().to(cutlass.BFloat16)
                            tRS_rD_out.store(acc_vec)
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rD_out,
                                tRS_sD[(None, None, None, silu_epi_buffer)],
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                    if retained_slice_idx == 1:
                        self.quantize_q1_sC_to_sA_sSFA(
                            tidx,
                            valid_tile_rows,
                            weight_expert_idx,
                            global_scale,
                            sC,
                            sA,
                            tRS_sD,
                            sfa_base_addr,
                            sfa_stage_elements,
                            Int32(1),
                            epi_rest_m,
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                # ============================================================
                # PHASE B: Sweep ALL FC2 output tiles using cached sA
                # No CTA-wide barrier needed here: gate is done with sB/sSFB
                # (barrier at line 925 ensured that), up uses sB_up/sSFB_up,
                # and DMA's B_down loads into sB/sSFB don't conflict with
                # MMA's SiLU+quant on sC/sA/sSFA. The phase2_pipeline
                # handles B_down availability for FC2 GEMM.
                # ============================================================
                scatter_N = Int32(scatter_output.shape[1])
                lane_id = Int32(tidx) & Int32(31)
                warp_in_tile = Int32(tidx) >> Int32(5)
                # M64 uses two warp rows: four MMA warps scatter four disjoint
                # 32x64 quadrants.  The old M128-derived stride of 64 left
                # warps 2/3 idle during every FC2 epilogue.
                warp_m_span = Int32(self.tile_shape_mnk[0] // 2)
                warp_n_span = Int32(self.tile_shape_mnk[1] // 2)
                warp_m_base = (warp_in_tile >> Int32(1)) * warp_m_span
                warp_n_base = (warp_in_tile & Int32(1)) * warp_n_span

                phase2_cons_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                        sSFB_phase2_tile = cute.local_tile(
                            sSFB,
                            cute.slice_(self.tile_shape_mnk, (0, None, None)),
                            (output_tile_idx % self.sfb_tiles_per_block, 0, None),
                        )
                        csSFB_phase2_tile = thr_ld_SFB.partition_S(sSFB_phase2_tile)
                        tCrSFB_phase2 = self._dense_cls._partition_fragment_SFB(
                            self,  # type: ignore[arg-type]
                            sSFB_phase2_tile[None, None, 0],
                            thr_mma,
                            tidx,
                        )
                        crSFB_phase2 = thr_ld_SFB.retile(tCrSFB_phase2)
                    else:
                        csSFB_phase2_tile = csSFB_full
                        tCrSFB_phase2 = tCrSFB_full
                        crSFB_phase2 = crSFB_full
                    down_acc.fill(0.0)
                    for retained_slice_idx in cutlass.range_constexpr(2):
                        # Each retained slice occupies one of the two former
                        # FC1 A/SFA pipeline stages.  Reload its fragments, then
                        # accumulate the matching FC2 B tile into one down_acc.
                        self.load_fc2_a_fragments(
                            num_k_blocks,
                            Int32(retained_slice_idx),
                            Int32(retained_slice_idx),
                            (csA_tile, csSFA_tile),
                            (crA_tile, crSFA_tile),
                            (smem_copy_A, smem_copy_SFA),
                        )

                        phase2_peek = phase2_pipeline.consumer_try_wait(
                            phase2_cons_state
                        )
                        phase2_pipeline.consumer_wait(phase2_cons_state, phase2_peek)
                        csB_phase2 = csB[None, None, None, phase2_cons_state.index]
                        csSFB_phase2 = csSFB_phase2_tile[
                            None, None, None, phase2_cons_state.index
                        ]
                        cute.copy(
                            smem_copy_B,
                            csB_phase2[None, None, 0],
                            crB[None, None, 0],
                        )
                        f2 = cute.filter_zeros(csSFB_phase2)
                        f4 = cute.filter_zeros(crSFB_phase2)
                        cute.copy(
                            smem_copy_SFB,
                            f2[None, None, 0],
                            f4[None, None, 0],
                        )

                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_next = (
                                0
                                if k_block_idx + 1 == num_k_blocks
                                else k_block_idx + 1
                            )
                            if k_block_idx == num_k_blocks - 1:
                                phase2_pipeline.consumer_release(phase2_cons_state)
                                phase2_cons_state.advance()
                            if k_next > 0:
                                cute.copy(
                                    smem_copy_B,
                                    csB_phase2[None, None, k_next],
                                    crB[None, None, k_next],
                                )
                                f2 = cute.filter_zeros(csSFB_phase2)
                                f4 = cute.filter_zeros(crSFB_phase2)
                                cute.copy(
                                    smem_copy_SFB,
                                    f2[None, None, k_next],
                                    f4[None, None, k_next],
                                )
                            for _mt in range(self.num_m_tiles):
                                for _nt in range(self.num_n_tiles):
                                    mma_atom.set(
                                        WarpField.SFA,
                                        tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                    )
                                    mma_atom.set(
                                        WarpField.SFB,
                                        tCrSFB_phase2[None, _nt, k_block_idx].iterator,
                                    )
                                    cute.gemm(
                                        mma_atom,
                                        down_acc[None, _mt, _nt],
                                        tCrA_tile[None, _mt, k_block_idx],
                                        tCrB[None, _nt, k_block_idx],
                                        down_acc[None, _mt, _nt],
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
                                    tRS_rD_slice[elem_idx] = down_epi_acc_slice[
                                        elem_idx
                                    ]

                        # The prior output tile's scatter only needs to finish
                        # before this tile overwrites sC.  Delay that barrier
                        # until after the next FC2 GEMM so scatter and GEMM can
                        # overlap; the final tile is closed by pass_sync below.
                        if output_tile_idx > 0:
                            self.epilog_sync_barrier.arrive_and_wait()

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
                        # The 8-wide reads from sC can cross another warp's
                        # stores, so wait for all MMA warps.
                        self.epilog_sync_barrier.arrive_and_wait()

                        rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])

                        # Per-warp scatter: each warp scatters its own quadrant
                        # of sC (32 M-rows × 64 N-cols).
                        warp_epi_rows = (
                            valid_rows - tile_m_base - rows_offset - warp_m_base
                        )
                        if warp_epi_rows > warp_m_span:
                            warp_epi_rows = warp_m_span
                        if warp_epi_rows < Int32(0):
                            warp_epi_rows = Int32(0)

                        # One lane owns two adjacent N8 reductions from the
                        # same row.  The packed helper performs one shared
                        # ld.v4.u32, BF16x2 scale, and one REDG per N8.
                        tile_pair_cols = warp_n_span // Int32(16)
                        pair_idx = lane_id
                        while pair_idx < warp_epi_rows * tile_pair_cols:
                            local_row = pair_idx // tile_pair_cols
                            local_pair_col = pair_idx - local_row * tile_pair_cols
                            local_col_base = warp_n_base + local_pair_col * Int32(16)
                            cached_row = rows_offset + warp_m_base + local_row
                            route_idx, wv = load_shared_i32_f32_pair(
                                scatter_meta_base_addr + cached_row * Int32(8)
                            )
                            for pair_half in cutlass.range_constexpr(2):
                                local_col = local_col_base + Int32(pair_half) * Int32(8)
                                global_col = tile_n_base_cur + local_col
                                sc_element_offset = Int32(
                                    sC.layout(
                                        (
                                            warp_m_base + local_row,
                                            local_col,
                                            epi_buffer,
                                        )
                                    )
                                )
                                sc_element_offset = sc_element_offset ^ (
                                    (sc_element_offset & Int32(0x1C0)) >> Int32(3)
                                )
                                sc_smem_addr = get_smem_ptr_as_int32(
                                    sC,
                                    sc_element_offset,
                                )
                                route_partial = intermediate_slice // Int32(2)
                                store_weighted_bf16x8_route(
                                    get_ptr_as_int64(
                                        packed_a_storage,
                                        route_scratch_base
                                        + (
                                            (
                                                route_idx * retained_group_count
                                                + route_partial
                                            )
                                            * scatter_N
                                            + global_col
                                        )
                                        * Int32(2),
                                    ),
                                    sc_smem_addr,
                                    wv,
                                    down_alpha_value,
                                )
                            pair_idx += Int32(self.num_threads_per_warp)

                # Final pass_sync: protect sA from next task's FC1 loads.
                # DMA warp waits here too after finishing all B_down loads;
                # this also closes the final output tile's scatter.
                self.pass_sync_barrier.arrive_and_wait()

                current_work_linear_idx += num_persistent_clusters
                is_valid_tile = _ld_shared_i32(ctrl_base_addr + Int32(8)) != Int32(0)
                tile_coord = (
                    _ld_shared_i32(ctrl_base_addr + Int32(12)),
                    _ld_shared_i32(ctrl_base_addr + Int32(16)),
                    _ld_shared_i32(ctrl_base_addr + Int32(20)),
                )
                current_local_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(24))
                accum_tile_m = _ld_shared_i32(ctrl_base_addr + Int32(28))

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
            published_work_linear_idx = Int32(bidz)
            tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                _compact_static_get_single_m_tile_work(
                    row_counts,
                    active_expert_count,
                    tile_m=Int32(self.tile_shape_mnk[0]),
                    num_tiles_n=Int32((self.output_tile_count_n + 1) // 2),
                    cluster_shape_mn=cluster_shape_mn,
                    current_work_linear_idx=current_work_linear_idx,
                    current_local_expert_idx=current_local_expert_idx,
                    accum_tile_m=accum_tile_m,
                    cta_id_in_cluster=cta_id_in_cluster,
                )
            )

            while is_valid_tile:
                tc = tile_coord
                intermediate_slice = tc[1] * Int32(2)
                local_expert_idx = tc[2]
                weight_expert_idx = weight_expert_ids[local_expert_idx]

                # Publish two adjacent N128 FC1 slices through the same paired
                # Gate/Up pipeline.  Consumer releases make Stage0/1 reusable;
                # no extra shared storage is needed.
                for retained_slice_idx in cutlass.range_constexpr(2):
                    current_slice = intermediate_slice + Int32(retained_slice_idx)
                    tBgB_w13_up_nk = tBgB_w13[
                        (None, current_slice, None, weight_expert_idx)
                    ]
                    sfb_up_tile_coord = current_slice // self.sfb_tiles_per_block
                    tBgSFB_w13_up_nk = tBgSFB_w13[
                        (None, sfb_up_tile_coord, None, weight_expert_idx)
                    ]
                    gate_slice = (
                        current_slice + gate_tile_cnt
                        if self.is_gated
                        else current_slice
                    )
                    tBgB_w13_gate_nk = tBgB_w13[
                        (None, gate_slice, None, weight_expert_idx)
                    ]
                    sfb_gate_tile_coord = gate_slice // self.sfb_tiles_per_block
                    tBgSFB_w13_gate_nk = tBgSFB_w13[
                        (None, sfb_gate_tile_coord, None, weight_expert_idx)
                    ]

                    prod_state.reset_count()
                    for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                        ml_pipeline.producer_acquire(prod_state)
                        cute.copy(
                            tma_b_w13,
                            tBgB_w13_gate_nk[(None, k_tile)],
                            tBsB_w13[(None, prod_state.index)],
                            tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                        )
                        cute.copy(
                            tma_sfb_w13,
                            tBgSFB_w13_gate_nk[(None, k_tile)],
                            tBsSFB_w13[(None, prod_state.index)],
                            tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                        )
                        if cutlass.const_expr(self.is_gated):
                            cute.copy(
                                tma_b_w13,
                                tBgB_w13_up_nk[(None, k_tile)],
                                tBsB_w13_up[(None, prod_state.index)],
                                tma_bar_ptr=ml_pipeline.producer_get_barrier(
                                    prod_state
                                ),
                            )
                            cute.copy(
                                tma_sfb_w13,
                                tBgSFB_w13_up_nk[(None, k_tile)],
                                tBsSFB_w13_up[(None, prod_state.index)],
                                tma_bar_ptr=ml_pipeline.producer_get_barrier(
                                    prod_state
                                ),
                            )
                        ml_pipeline.producer_commit(prod_state)
                        prod_state.advance()

                # FC2 reuses gate B/SFB storage, so wait until both branch MMA
                # streams have drained the paired transaction.
                self.pass_sync_barrier.arrive_and_wait()

                # ---- FC2 B_down loads: continuous pipeline ----
                # No barrier needed: sB/sSFB are free (gate done, up uses
                # sB_up/sSFB_up). phase2_pipeline handles data availability.
                # intermediate_slice selects the K-tile of GEMM2 (FC1 output N-tile
                # = GEMM2 K-tile since intermediate dim is the reduction dim).
                # Load ALL FC2 tiles continuously once stage1 no longer needs
                # the gate staging buffers.
                phase2_prod_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    for retained_slice_idx in cutlass.range_constexpr(2):
                        current_slice = intermediate_slice + Int32(retained_slice_idx)
                        phase2_pipeline.producer_acquire(phase2_prod_state)
                        cute.copy(
                            tma_b_down,
                            tBgB_down[
                                (
                                    None,
                                    output_tile_idx,
                                    current_slice,
                                    weight_expert_idx,
                                )
                            ],
                            tBsB_down[(None, phase2_prod_state.index)],
                            tma_bar_ptr=phase2_pipeline.producer_get_barrier(
                                phase2_prod_state
                            ),
                        )
                        cute.copy(
                            tma_sfb_down,
                            tBgSFB_down[
                                (
                                    None,
                                    output_tile_idx // self.sfb_tiles_per_block,
                                    current_slice,
                                    weight_expert_idx,
                                )
                            ],
                            tBsSFB_down[(None, phase2_prod_state.index)],
                            tma_bar_ptr=phase2_pipeline.producer_get_barrier(
                                phase2_prod_state
                            ),
                        )
                        phase2_pipeline.producer_commit(phase2_prod_state)
                        phase2_prod_state.advance()

                if Int32(tidx) == Int32(self.tma_load_warp_id * 32):
                    # Hybrid schedule: wave 2 is still statically strided,
                    # every later task is claimed atomically. Work stealing
                    # erases the tail's wave quantization and intra-wave
                    # variance; claim order is nondeterministic but tasks
                    # write disjoint route scratch and the finalize sums in
                    # fixed token order, so output stays bitwise
                    # deterministic.
                    next_work_linear_idx = Int32(0)
                    if published_work_linear_idx < num_persistent_clusters:
                        next_work_linear_idx = (
                            published_work_linear_idx + num_persistent_clusters
                        )
                    else:
                        next_work_linear_idx = atomic_add_global_i32(
                            get_ptr_as_int64(virt_route_scratch, claim_slot),
                            Int32(1),
                        )
                    published_work_linear_idx = next_work_linear_idx
                    (
                        next_tile_coord,
                        next_is_valid_tile,
                        next_local_expert_idx,
                        next_accum_tile_m,
                    ) = _compact_static_get_single_m_tile_work(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_shape_mnk[0]),
                        num_tiles_n=Int32((self.output_tile_count_n + 1) // 2),
                        cluster_shape_mn=cluster_shape_mn,
                        current_work_linear_idx=next_work_linear_idx,
                        current_local_expert_idx=current_local_expert_idx,
                        accum_tile_m=accum_tile_m,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )
                    _st_shared_i32(ctrl_base_addr + Int32(8), Int32(next_is_valid_tile))
                    _st_shared_i32(
                        ctrl_base_addr + Int32(12), Int32(next_tile_coord[0])
                    )
                    _st_shared_i32(
                        ctrl_base_addr + Int32(16), Int32(next_tile_coord[1])
                    )
                    _st_shared_i32(
                        ctrl_base_addr + Int32(20), Int32(next_tile_coord[2])
                    )
                    _st_shared_i32(ctrl_base_addr + Int32(24), next_local_expert_idx)
                    _st_shared_i32(ctrl_base_addr + Int32(28), next_accum_tile_m)

                # Final pass_sync: match MMA warps' barrier after FC2 sweep.
                # Ensures MMA warps finish scatter before DMA starts next task's FC1.
                self.pass_sync_barrier.arrive_and_wait()

                current_work_linear_idx += num_persistent_clusters
                is_valid_tile = _ld_shared_i32(ctrl_base_addr + Int32(8)) != Int32(0)
                tile_coord = (
                    _ld_shared_i32(ctrl_base_addr + Int32(12)),
                    _ld_shared_i32(ctrl_base_addr + Int32(16)),
                    _ld_shared_i32(ctrl_base_addr + Int32(20)),
                )
                current_local_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(24))
                accum_tile_m = _ld_shared_i32(ctrl_base_addr + Int32(28))

            ml_pipeline.producer_tail(prod_state)
            phase2_pipeline.producer_tail(phase2_prod_state)

        # Route-private stores are complete when this cooperative launch
        # retires.  The stream-ordered finalize kernel performs the reduction.
        return

    @cute.kernel
    def finalize_kernel(
        self,
        packed_a_storage: cute.Tensor,
        scatter_output: cute.Tensor,
        topk_ids: cute.Tensor,
        retained_group_count: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, bidz = cute.arch.block_idx()
        _, _, gdim_z = cute.arch.grid_dim()
        flat_tid = Int32(bidz) * Int32(256) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(256)

        num_tokens = Int32(scatter_output.shape[0])
        cols = Int32(scatter_output.shape[1])
        total_pairs = Int32(topk_ids.shape[0])
        num_topk = total_pairs // num_tokens
        route_scratch_bytes = (
            total_pairs * Int32(retained_group_count) * cols * Int32(2)
        )
        route_scratch_base = Int32(packed_a_storage.shape[0]) - route_scratch_bytes

        vecs_per_token = cols // Int32(8)
        final_vec_count = num_tokens * vecs_per_token
        final_vec_idx = flat_tid
        while final_vec_idx < final_vec_count:
            final_token = final_vec_idx // vecs_per_token
            final_col = (final_vec_idx - final_token * vecs_per_token) * Int32(8)
            final_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            final_acc.fill(0.0)
            final_slot = Int32(0)
            while final_slot < num_topk:
                final_route = final_token * num_topk + final_slot
                final_partial = Int32(0)
                while final_partial < Int32(retained_group_count):
                    route_values = load_global_bf16x8_to_f32x8(
                        get_ptr_as_int64(
                            packed_a_storage,
                            route_scratch_base
                            + (
                                (
                                    final_route * Int32(retained_group_count)
                                    + final_partial
                                )
                                * cols
                                + final_col
                            )
                            * Int32(2),
                        )
                    )
                    for final_elem in cutlass.range_constexpr(8):
                        final_acc[final_elem] += route_values[final_elem]
                    final_partial += Int32(1)
                final_slot += Int32(1)
            store_global_f32x8_as_bf16(
                get_ptr_as_int64(
                    scatter_output,
                    final_token * cols + final_col,
                ),
                final_acc[0],
                final_acc[1],
                final_acc[2],
                final_acc[3],
                final_acc[4],
                final_acc[5],
                final_acc[6],
                final_acc[7],
            )
            final_vec_idx += flat_stride
        return


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


__all__ = ["MoEStaticKernel"]
