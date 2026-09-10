"""Optimized gated NVFP4 dynamic MoE kernel for SM120/SM121."""

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
    Int64,
    Uint8,
    Uint32,
    Uint64,
    T,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
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
    ld_shared_i32_relaxed,
    rcp_approx_ftz,
    quantize_block_fp4,
    quantize_block_fp4_fast,
    get_ptr_as_int64,
    get_smem_ptr_as_int32,
    st_global_f32,
    st_global_i32,
    shared_ptr_to_u32,
    st_shared_i32,
    st_shared_f32,
    st_shared_u8,
    st_global_v4_u32,
)
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
    Sm120B12xBlockScaledDenseGemmKernel as DenseGemmKernel,
)
from ..moe_activation import gated_activation_f32, is_gated_activation


def _dynamic_gated_activation_f32(
    g,
    u,
    *,
    activation: str,
    limit: float | None,
    alpha: float,
    beta: float,
    fast_math: bool,
):
    """Apply a compile-time-selected gated scalar formula.

    Keep the tuned explicit-inline-PTX reciprocal emission path for SiLU.
    ``cute.arch.rcp_approx`` has the same FTZ math semantics in the current
    CUTLASS DSL, but reaches the compiler through an NVVM intrinsic.  Other
    gated formulas remain centralized in ``gated_activation_f32``.
    """
    if cutlass.const_expr(activation == "silu"):
        sigmoid = rcp_approx_ftz(
            cutlass.Float32(1.0)
            + cute.math.exp(cutlass.Float32(0.0) - g, fastmath=fast_math)
        )
        return (g * sigmoid) * u
    return gated_activation_f32(
        g,
        u,
        activation=activation,
        limit=limit,
        alpha=alpha,
        beta=beta,
        fast_math=fast_math,
    )


@dsl_user_op
def atomic_add_shared_i32(addr: Int32, value: Int32, *, loc=None, ip=None):
    """Atomic Int32 add at a 32-bit shared-memory byte address."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared.add.s32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


_SF_VEC_SIZE = 16
_TASK_SLICE_CHUNK = 4
_PRODUCER_PAIRS_PER_WARP = 2
_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0

# The upstream compact queue exposes two int32 metadata planes.  Pack the
# optimized scheduler's complete descriptor into those planes so task claim
# and prefetch use shifts/masks instead of integer division in the hot loop.
# word0 = expert[15:0] | m_tile[31:16]
# word1 = valid_rows[7:0] | slice_begin[19:8] | slice_count[31:20]
_TASK_EXPERT_MASK = 0xFFFF
_TASK_M_TILE_MASK = 0xFFFF
_TASK_VALID_ROWS_MASK = 0xFF
_TASK_SLICE_MASK = 0xFFF


# For small routed worksets, retain Q0's packed-A output for the later FC1 TMA
# reads.  Large worksets use the original store to avoid competing with the
# output-reduction and weight working sets for L2 capacity.
@dsl_user_op
def st_global_u64_adaptive_l2(
    num_tokens,
    base_ptr,
    value,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(num_tokens).ir_value(loc=loc, ip=ip),
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred persist; .reg .b64 cp;"
        " setp.le.s32 persist, $0, 2048;"
        " @persist createpolicy.fractional.L2::evict_last.b64 cp, 1.0;"
        " @persist st.global.L2::cache_hint.u64 [$1], $2, cp;"
        " @!persist st.global.u64 [$1], $2; }",
        "r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# Keep routed BF16 output resident across its top-k reduction updates.  The
# operands, reduction width, and numeric order match the anchor helper.
@dsl_user_op
def scatter_add_v4_bf16x2(
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
        "{ .reg .b32 p0,p1,p2,p3; .reg .b64 cp;"
        " cvt.rn.satfinite.bf16x2.f32 p0, $2, $1;"
        " cvt.rn.satfinite.bf16x2.f32 p1, $4, $3;"
        " cvt.rn.satfinite.bf16x2.f32 p2, $6, $5;"
        " cvt.rn.satfinite.bf16x2.f32 p3, $8, $7;"
        " createpolicy.fractional.L2::evict_last.b64 cp, 0.75;"
        " red.global.add.noftz.v4.bf16x2.L2::cache_hint"
        " [$0], {p0, p1, p2, p3}, cp; }",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def scatter_add_weighted_bf16x8_packed(
    addr,
    smem_addr,
    route_weight,
    *,
    loc=None,
    ip=None,
):
    """Weight one packed BF16x8 sC vector and issue the unchanged REDG."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            route_weight.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 p0,p1,p2,p3,w2; .reg .b16 w;"
        " .reg .b64 cp;"
        " ld.shared.v4.u32 {p0,p1,p2,p3}, [$1];"
        " cvt.rn.bf16.f32 w, $2;"
        " mov.b32 w2, {w,w};"
        " mul.rn.bf16x2 p0, p0, w2;"
        " mul.rn.bf16x2 p1, p1, w2;"
        " mul.rn.bf16x2 p2, p2, w2;"
        " mul.rn.bf16x2 p3, p3, w2;"
        " createpolicy.fractional.L2::evict_last.b64 cp, 0.75;"
        " red.global.add.noftz.v4.bf16x2.L2::cache_hint"
        " [$0], {p0,p1,p2,p3}, cp; }",
        "l,r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def scatter_add_weighted_bf16x8_packed_alpha(
    addr,
    smem_addr,
    route_weight,
    down_alpha,
    *,
    loc=None,
    ip=None,
):
    """Fuse the expert down-alpha into one route scale per BF16x8 REDG."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            route_weight.ir_value(loc=loc, ip=ip),
            down_alpha.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 p0,p1,p2,p3,w2; .reg .b16 w;"
        " .reg .f32 combined_scale; .reg .b64 cp;"
        " ld.shared.v4.u32 {p0,p1,p2,p3}, [$1];"
        " mul.rn.f32 combined_scale, $2, $3;"
        " cvt.rn.bf16.f32 w, combined_scale;"
        " mov.b32 w2, {w,w};"
        " mul.rn.bf16x2 p0, p0, w2;"
        " mul.rn.bf16x2 p1, p1, w2;"
        " mul.rn.bf16x2 p2, p2, w2;"
        " mul.rn.bf16x2 p3, p3, w2;"
        " createpolicy.fractional.L2::evict_last.b64 cp, 1.0;"
        " red.global.add.noftz.v4.bf16x2.L2::cache_hint"
        " [$0], {p0,p1,p2,p3}, cp; }",
        "l,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
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
def load_shared_i32_f32_pair(addr: Int32, *, loc=None, ip=None):
    """Load one interleaved {Int32, Float32-bits} record with one LDS.64."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    token_word = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    weight_word = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return (
        Int32(token_word),
        Uint32(weight_word).bitcast(cutlass.Float32),
    )


@dsl_user_op
def _ld_shared_i32_volatile(addr, *, loc=None, ip=None):
    """Side-effecting shared load for the phase-mutable ctrl[28] value."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _membar_cta(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "membar.cta;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_bf16x8_to_f32x8(addr: Int32, *, loc=None, ip=None):
    """Load one aligned BF16x8 vector with exactly one 128-bit S2R."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 p0, p1, p2, p3;
            .reg .b16 b0, b1, b2, b3, b4, b5, b6, b7;
            ld.shared.v4.u32 {p0, p1, p2, p3}, [$8];
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
        "=f,=f,=f,=f,=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    values = []
    for idx in range(8):
        value = llvm.extractvalue(T.f32(), result, [idx], loc=loc, ip=ip)
        values.append(cutlass.Float32(value))
    return tuple(values)


@dsl_user_op
def load_global_bf16x16_to_f32x16(addr: Int64, *, loc=None, ip=None):
    """Load one aligned BF16x16 block with two 128-bit global loads."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 16),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 p0, p1, p2, p3, p4, p5, p6, p7;
            .reg .b16 b0, b1, b2, b3, b4, b5, b6, b7;
            .reg .b16 b8, b9, b10, b11, b12, b13, b14, b15;
            ld.global.v4.u32 {p0, p1, p2, p3}, [$16];
            ld.global.v4.u32 {p4, p5, p6, p7}, [$16+0x10];
            mov.b32 {b0, b1}, p0;
            mov.b32 {b2, b3}, p1;
            mov.b32 {b4, b5}, p2;
            mov.b32 {b6, b7}, p3;
            mov.b32 {b8, b9}, p4;
            mov.b32 {b10, b11}, p5;
            mov.b32 {b12, b13}, p6;
            mov.b32 {b14, b15}, p7;
            cvt.f32.bf16 $0, b0;
            cvt.f32.bf16 $1, b1;
            cvt.f32.bf16 $2, b2;
            cvt.f32.bf16 $3, b3;
            cvt.f32.bf16 $4, b4;
            cvt.f32.bf16 $5, b5;
            cvt.f32.bf16 $6, b6;
            cvt.f32.bf16 $7, b7;
            cvt.f32.bf16 $8, b8;
            cvt.f32.bf16 $9, b9;
            cvt.f32.bf16 $10, b10;
            cvt.f32.bf16 $11, b11;
            cvt.f32.bf16 $12, b12;
            cvt.f32.bf16 $13, b13;
            cvt.f32.bf16 $14, b14;
            cvt.f32.bf16 $15, b15;
        }
        """,
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    values = []
    for idx in range(16):
        value = llvm.extractvalue(T.f32(), result, [idx], loc=loc, ip=ip)
        values.append(cutlass.Float32(value))
    return tuple(values)


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


@dsl_user_op
def q0_bulk_barrier_init(addr: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "{ mbarrier.init.shared::cta.b64 [$0], 1; fence.proxy.async.shared::cta; }",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def q0_cp_async_bulk(
    dst_addr: Int32,
    src_addr: Int64,
    copy_bytes: Int32,
    barrier_addr: Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(dst_addr).ir_value(loc=loc, ip=ip),
            Int64(src_addr).ir_value(loc=loc, ip=ip),
            Int32(copy_bytes).ir_value(loc=loc, ip=ip),
            Int32(barrier_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def q0_bulk_arrive_expect_tx(
    barrier_addr: Int32, transaction_bytes: Int32, *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [
            Int32(barrier_addr).ir_value(loc=loc, ip=ip),
            Int32(transaction_bytes).ir_value(loc=loc, ip=ip),
        ],
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def q0_bulk_try_wait(barrier_addr: Int32, phase: Int32, *, loc=None, ip=None):
    ready = llvm.inline_asm(
        T.i32(),
        [
            Int32(barrier_addr).ir_value(loc=loc, ip=ip),
            Int32(phase).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p;"
        " mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " p, [$1], $2, 0x989680;"
        " selp.u32 $0, 1, 0, p; }",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(ready)


@dsl_user_op
def load_shared_bf16x16_to_f32x16(addr: Int32, *, loc=None, ip=None):
    """Load one linear BF16x16 block from the Q0 startup staging buffer."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 16),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 p0, p1, p2, p3, p4, p5, p6, p7;
            .reg .b16 b0, b1, b2, b3, b4, b5, b6, b7;
            .reg .b16 b8, b9, b10, b11, b12, b13, b14, b15;
            ld.shared.v4.u32 {p0, p1, p2, p3}, [$16];
            ld.shared.v4.u32 {p4, p5, p6, p7}, [$16+0x10];
            mov.b32 {b0, b1}, p0;
            mov.b32 {b2, b3}, p1;
            mov.b32 {b4, b5}, p2;
            mov.b32 {b6, b7}, p3;
            mov.b32 {b8, b9}, p4;
            mov.b32 {b10, b11}, p5;
            mov.b32 {b12, b13}, p6;
            mov.b32 {b14, b15}, p7;
            cvt.f32.bf16 $0, b0;
            cvt.f32.bf16 $1, b1;
            cvt.f32.bf16 $2, b2;
            cvt.f32.bf16 $3, b3;
            cvt.f32.bf16 $4, b4;
            cvt.f32.bf16 $5, b5;
            cvt.f32.bf16 $6, b6;
            cvt.f32.bf16 $7, b7;
            cvt.f32.bf16 $8, b8;
            cvt.f32.bf16 $9, b9;
            cvt.f32.bf16 $10, b10;
            cvt.f32.bf16 $11, b11;
            cvt.f32.bf16 $12, b12;
            cvt.f32.bf16 $13, b13;
            cvt.f32.bf16 $14, b14;
            cvt.f32.bf16 $15, b15;
        }
        """,
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    values = []
    for idx in range(16):
        value = llvm.extractvalue(T.f32(), result, [idx], loc=loc, ip=ip)
        values.append(cutlass.Float32(value))
    return tuple(values)


class MoEGatedDynamicKernel:
    """Queue-driven gated kernel for the validated M128N128 staging geometry."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        activation: str = "silu",
        swiglu_alpha: float = 1.702,
        swiglu_beta: float = 1.0,
        swiglu_limit: float | None = None,
        share_input_across_experts: bool = False,
    ):
        if not is_gated_activation(activation):
            raise ValueError(
                f"MoEGatedDynamicKernel requires a gated activation; got {activation!r}"
            )
        if sf_vec_size != _SF_VEC_SIZE:
            raise ValueError(
                "the gated dynamic kernel requires 16-element scale blocks; "
                f"got sf_vec_size={sf_vec_size}"
            )
        if mma_tiler_mn != (128, 128):
            raise ValueError(
                "the gated dynamic kernel requires a logical M128xN128 CTA tile"
            )
        self._dense_cls = DenseGemmKernel
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.input_scales_are_reciprocal = input_scales_are_reciprocal
        self.fast_math = fast_math
        self.activation = activation
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        self.swiglu_limit = float(swiglu_limit) if swiglu_limit is not None else None
        self.share_input_across_experts = share_input_across_experts
        tile_k = sf_vec_size * 8
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        self.fc1_tile_shape_mnk = (
            mma_tiler_mn[0],
            mma_tiler_mn[1] // 2,
            tile_k,
        )
        self.fc1_sfb_tile_shape_nk = (
            max(128, self.fc1_tile_shape_mnk[1]),
            tile_k,
        )
        self.fc1_sfb_tiles_per_block = (
            self.fc1_sfb_tile_shape_nk[0] // self.fc1_tile_shape_mnk[1]
        )
        if self.fc1_sfb_tiles_per_block != 2:
            raise ValueError("expected exactly two logical N64 tiles per SFB block")
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.occupancy = 1
        self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.fc2_group_a_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=4 * self.num_threads_per_warp,
        )
        self.fc2_group_b_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=4 * self.num_threads_per_warp,
        )
        self.pass_gate_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )
        self.pass_final_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_cta,
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

    def _setup_attributes(self, hidden_size: int):
        import cutlass.utils.blackwell_helpers as sm120_utils

        self._hidden_size = hidden_size

        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        atom_layout = cute.make_layout((4, 2, 1))
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
        fc1_permutation_mnk = sm120_utils.get_permutation_mnk(
            self.fc1_tile_shape_mnk,
            self.sf_vec_size,
            False,
        )
        self.fc1_tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=fc1_permutation_mnk,
        )
        self.mma_atom = cute.make_mma_atom(mma_op)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_m_tiles = self.tile_shape_mnk[0] // (16 * 4)
        self.num_n_tiles = self.tile_shape_mnk[1] // (8 * 2)
        self.num_k_blocks = self.tile_shape_mnk[2] // 64

        _sfa_smem = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        _sfb_smem = sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        # One statically typed Stage3 pipeline covers every FC1 slice.  The
        # physical shared allocation remains A5+B2: FC1 B stage2 aliases sC
        # while it is dead, and retained Q1 slices that would collide with
        # A/SFA stage2 are temporarily held in registers.
        self.ab_stage = 3
        self.ab_storage_stage = 2
        self.phase2_stage = 3
        self.epi_stage = 1
        (
            _a_smem_layout_stage2,
            self.b_smem_layout_staged,
            _sfa_smem_layout_stage2,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_storage_stage,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )
        (
            _phase2_a_smem_layout_staged,
            self.phase2_b_smem_layout_staged,
            _phase2_sfa_smem_layout_staged,
            self.phase2_sfb_smem_layout_staged,
            _phase2_epi_smem_layout_staged,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.phase2_stage,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )
        (
            self.a_smem_layout_staged,
            _b_smem_layout_stage5,
            _sfa_smem_layout_stage5,
            _sfb_smem_layout_stage5,
            _epi_smem_layout_stage5,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            5,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )
        (
            _a_smem_layout_stage4,
            _b_smem_layout_stage4,
            self.sfa_smem_layout_staged,
            _sfb_smem_layout_stage4,
            _epi_smem_layout_stage4,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            4,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )
        # FC1 gets a true N64 B layout/TMA contract while retaining the N128
        # physical SFB block required by SM120 block-scaled scale-factor
        # packing.  A/SFA keep the existing M128xK128 storage and are replayed.
        (
            _fc1_a_smem_layout_staged,
            self.fc1_b_smem_layout_staged,
            _fc1_sfa_smem_layout_staged,
            self.fc1_sfb_smem_layout_staged,
            _fc1_epi_smem_layout_staged,
        ) = self._dense_cls._make_smem_layouts(
            self.fc1_tile_shape_mnk,
            (self.fc1_tile_shape_mnk[0], self.fc1_tile_shape_mnk[1]),
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.fc1_tiled_mma,
        )
        (
            _fc1_a_smem_layout_storage,
            self.fc1_b_smem_layout_storage,
            _fc1_sfa_smem_layout_storage,
            self.fc1_sfb_smem_layout_storage,
            _fc1_epi_smem_layout_storage,
        ) = self._dense_cls._make_smem_layouts(
            self.fc1_tile_shape_mnk,
            (self.fc1_tile_shape_mnk[0], self.fc1_tile_shape_mnk[1]),
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_storage_stage,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.fc1_tiled_mma,
        )

    @cute.jit
    def resident_grid_barrier(
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
    def publish_uniform_deferred_tasks(
        self,
        task_expert: cute.Tensor,
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
            task_expert[slot] = expert_idx | (m_tile_idx << Int32(16))
            task_valid_rows[slot] = (
                valid_rows | (slice_begin << Int32(8)) | (slice_count << Int32(20))
            )
            g += Int32(1)

    @cute.jit
    def publish_variable_deferred_tasks(
        self,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        gate_tile_cnt: Int32,
        split_tile_count: Int32,
        expert_idx: Int32,
        m_tile_idx: Int32,
        valid_rows: Int32,
    ):
        split_groups = (gate_tile_cnt + Int32(1)) // Int32(2)
        extra_per_split = split_groups - Int32(1)
        split_tiles_before = m_tile_idx
        if split_tiles_before > split_tile_count:
            split_tiles_before = split_tile_count
        start = m_tile_idx + split_tiles_before * extra_per_split

        num_groups = Int32(1)
        slice_chunk = gate_tile_cnt
        if m_tile_idx < split_tile_count:
            num_groups = split_groups
            slice_chunk = Int32(2)

        g = Int32(0)
        while g < num_groups:
            slot = start + g
            slice_begin = g * slice_chunk
            slice_count = gate_tile_cnt - slice_begin
            if slice_count > slice_chunk:
                slice_count = slice_chunk
            task_expert[slot] = expert_idx | (m_tile_idx << Int32(16))
            task_valid_rows[slot] = (
                valid_rows | (slice_begin << Int32(8)) | (slice_count << Int32(20))
            )
            g += Int32(1)

    @cute.jit
    def claim_and_cache_task(
        self,
        tidx,
        warp_idx,
        is_cta_leader: Int32,
        ctrl_base_addr: Int32,
        task_head: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        scatter_tok_base_addr: Int32,
        scatter_weight_base_addr: Int32,
    ):
        if is_cta_leader > Int32(0):
            _st_shared_i32(ctrl_base_addr + Int32(0), Int32(0))  # has_task
            _st_shared_i32(ctrl_base_addr + Int32(4), Int32(0))  # done
            next_has = _ld_shared_i32_volatile(ctrl_base_addr + Int32(32))
            next_done = _ld_shared_i32_volatile(ctrl_base_addr + Int32(36))
            if next_has > Int32(0):
                _membar_cta()
                _st_shared_i32(ctrl_base_addr + Int32(0), Int32(1))
                _st_shared_i32(
                    ctrl_base_addr + Int32(8),
                    _ld_shared_i32(ctrl_base_addr + Int32(40)),
                )
                _st_shared_i32(
                    ctrl_base_addr + Int32(12),
                    _ld_shared_i32(ctrl_base_addr + Int32(44)),
                )
                _st_shared_i32(
                    ctrl_base_addr + Int32(16),
                    _ld_shared_i32(ctrl_base_addr + Int32(48)),
                )
                _st_shared_i32(
                    ctrl_base_addr + Int32(20),
                    _ld_shared_i32(ctrl_base_addr + Int32(52)),
                )
                _st_shared_i32(
                    ctrl_base_addr + Int32(24),
                    _ld_shared_i32(ctrl_base_addr + Int32(56)),
                )
                _st_shared_i32(ctrl_base_addr + Int32(32), Int32(0))
            elif next_done > Int32(0):
                _st_shared_i32(ctrl_base_addr + Int32(4), Int32(1))
                _st_shared_i32(ctrl_base_addr + Int32(36), Int32(0))
            else:
                tail = _ld_shared_i32_volatile(ctrl_base_addr + Int32(28))
                slot = atomic_add_global_i32(
                    get_ptr_as_int64(task_head, Int32(0)),
                    Int32(1),
                )
                if slot < tail:
                    descriptor0 = task_expert[slot].to(Int32)
                    descriptor1 = task_valid_rows[slot].to(Int32)
                    expert = descriptor0 & Int32(_TASK_EXPERT_MASK)
                    m_tile = (descriptor0 >> Int32(16)) & Int32(_TASK_M_TILE_MASK)
                    valid_rows = descriptor1 & Int32(_TASK_VALID_ROWS_MASK)
                    slice_begin = (descriptor1 >> Int32(8)) & Int32(_TASK_SLICE_MASK)
                    slice_count = (descriptor1 >> Int32(20)) & Int32(_TASK_SLICE_MASK)
                    _st_shared_i32(ctrl_base_addr + Int32(0), Int32(1))
                    _st_shared_i32(ctrl_base_addr + Int32(8), expert)
                    _st_shared_i32(ctrl_base_addr + Int32(12), m_tile)
                    _st_shared_i32(ctrl_base_addr + Int32(16), slice_begin)
                    _st_shared_i32(ctrl_base_addr + Int32(20), slice_count)
                    _st_shared_i32(ctrl_base_addr + Int32(24), valid_rows)
                else:
                    _st_shared_i32(ctrl_base_addr + Int32(4), Int32(1))
        cute.arch.sync_threads()

        has_task = _ld_shared_i32(ctrl_base_addr + Int32(0))
        if has_task > Int32(0) and warp_idx < Int32(self.num_mma_warps):
            task_m_tile_idx_cache = _ld_shared_i32(ctrl_base_addr + Int32(12))
            task_valid_rows_cache = _ld_shared_i32(ctrl_base_addr + Int32(24))
            tile_m_base_cache = task_m_tile_idx_cache * Int32(self.tile_shape_mnk[0])
            cache_row = Int32(tidx)
            while cache_row < Int32(self.tile_shape_mnk[0]):
                tok = Int32(0)
                wv = cutlass.Float32(0.0)
                if cache_row < task_valid_rows_cache:
                    global_row_cache = tile_m_base_cache + cache_row
                    tok = token_map[global_row_cache].to(Int32)
                    wv = token_weights[global_row_cache].to(cutlass.Float32)
                metadata_offset = cache_row * Int32(8)
                _st_shared_i32(scatter_tok_base_addr + metadata_offset, tok)
                st_shared_f32(
                    scatter_weight_base_addr + metadata_offset,
                    wv,
                )
                cache_row += Int32(self.threads_per_cta)
            # Scatter metadata is consumed only by the eight math warps.
            # Let the TMA warp start FC1 after descriptor visibility.
            self.epilog_sync_barrier.arrive_and_wait()

        is_done = _ld_shared_i32(ctrl_base_addr + Int32(4))
        return has_task, is_done

    @cute.jit
    def prefetch_next_task_descriptor(
        self,
        lane_id: Int32,
        ctrl_base_addr: Int32,
        task_head: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
    ):
        if lane_id == Int32(0):
            next_has = _ld_shared_i32_volatile(ctrl_base_addr + Int32(32))
            next_done = _ld_shared_i32_volatile(ctrl_base_addr + Int32(36))
            if next_has == Int32(0) and next_done == Int32(0):
                tail = _ld_shared_i32_volatile(ctrl_base_addr + Int32(28))
                slot = atomic_add_global_i32(
                    get_ptr_as_int64(task_head, Int32(0)),
                    Int32(1),
                )
                if slot < tail:
                    descriptor0 = task_expert[slot].to(Int32)
                    descriptor1 = task_valid_rows[slot].to(Int32)
                    expert = descriptor0 & Int32(_TASK_EXPERT_MASK)
                    m_tile = (descriptor0 >> Int32(16)) & Int32(_TASK_M_TILE_MASK)
                    valid_rows = descriptor1 & Int32(_TASK_VALID_ROWS_MASK)
                    slice_begin = (descriptor1 >> Int32(8)) & Int32(_TASK_SLICE_MASK)
                    slice_count = (descriptor1 >> Int32(20)) & Int32(_TASK_SLICE_MASK)
                    _st_shared_i32(ctrl_base_addr + Int32(40), expert)
                    _st_shared_i32(ctrl_base_addr + Int32(44), m_tile)
                    _st_shared_i32(ctrl_base_addr + Int32(48), slice_begin)
                    _st_shared_i32(ctrl_base_addr + Int32(52), slice_count)
                    _st_shared_i32(ctrl_base_addr + Int32(56), valid_rows)
                    _membar_cta()
                    _st_shared_i32(ctrl_base_addr + Int32(32), Int32(1))
                else:
                    _st_shared_i32(ctrl_base_addr + Int32(36), Int32(1))

    @cute.jit
    def fc1_gate_up_swiglu_to_sC(
        self,
        tidx,
        ml_pipeline,
        cons_state,
        up_pipeline,
        up_cons_state,
        fc1_tiled_mma,
        mma_atom,
        sSFB_fc1,
        sSFB_up_fc1,
        sSFB_up_fc1_extra,
        fc1_thr_mma,
        thr_ld_SFB_fc1,
        csA_fc1,
        csB_fc1,
        csB_up_fc1,
        csSFA_fc1,
        crA_fc1,
        crB_fc1,
        crB_up_fc1,
        crSFA_fc1,
        tCrA_fc1,
        tCrB_fc1,
        tCrB_up_fc1,
        tCrSFA_fc1,
        smem_copy_A_fc1,
        smem_copy_B_fc1,
        smem_copy_SFA_fc1,
        smem_copy_SFB_fc1,
        fc1_tiled_copy_r2s,
        fc1_tRS_sD,
        fc1_k_tile_cnt,
        fc1_num_k_blocks,
        fc1_m_tiles,
        fc1_n_tiles,
        alpha_value,
        valid_rows,
        task_expert_idx,
        global_scale,
        sC,
        sA,
        sfa_base_addr,
        epi_rest_m,
    ):
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        fc1_acc_shape = fc1_tiled_mma.partition_shape_C(
            (self.fc1_tile_shape_mnk[0], self.fc1_tile_shape_mnk[1])
        )
        gate_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)
        fc1_tRS_rGate = fc1_tiled_copy_r2s.retile(gate_acc)
        fc1_tRS_rUp = fc1_tiled_copy_r2s.retile(up_acc)
        fc1_tRS_rAct = cute.make_rmem_tensor(
            fc1_tRS_rGate[(None, 0, 0)].shape, self.acc_dtype
        )
        fc1_tRS_rAct_out = cute.make_rmem_tensor(
            fc1_tRS_rGate[(None, 0, 0)].shape, cutlass.BFloat16
        )
        fc1_tRS_rAct_hold = cute.make_rmem_tensor(fc1_tRS_rGate.shape, cutlass.BFloat16)

        # ============================================================
        # PHASE A: native dual-N64 FC1 for this logical N128 slice
        # ============================================================
        # One pipeline/state sequence covers two branch-paired
        # halves.  Each stage carries one A/SFA plus independent
        # Gate/Up B/SFB payloads; both OMMAs complete before release.
        cons_state.reset_count()
        for fc1_half in cutlass.range_constexpr(2):
            # SM120 packs SFB in physical N128 blocks.  Select the
            # live N64 half from the replayed block for both branches.
            sSFB_fc1_half = cute.local_tile(
                sSFB_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0, None),
            )
            tCrSFB_fc1_half = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                sSFB_fc1_half[None, None, 0],
                fc1_thr_mma,
                tidx,
            )
            csSFB_fc1_half = thr_ld_SFB_fc1.partition_S(sSFB_fc1_half)
            crSFB_fc1_half = thr_ld_SFB_fc1.retile(tCrSFB_fc1_half)
            sSFB_up_fc1_half = cute.local_tile(
                sSFB_up_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0, None),
            )
            tCrSFB_up_fc1_half = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                sSFB_up_fc1_half[None, None, 0],
                fc1_thr_mma,
                tidx,
            )
            csSFB_up_fc1_half = thr_ld_SFB_fc1.partition_S(sSFB_up_fc1_half)
            sSFB_up_fc1_extra_half = cute.local_tile(
                sSFB_up_fc1_extra,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0),
            )
            csSFB_up_fc1_extra_half = thr_ld_SFB_fc1.partition_S(sSFB_up_fc1_extra_half)
            crSFB_up_fc1_half = thr_ld_SFB_fc1.retile(tCrSFB_up_fc1_half)
            fz_crSFA_fc1 = cute.filter_zeros(crSFA_fc1)
            fz_crSFB_fc1_half = cute.filter_zeros(crSFB_fc1_half)
            fz_crSFB_up_fc1_half = cute.filter_zeros(crSFB_up_fc1_half)

            # Branch-paired Gate/Up N64: A/SFA are read once from
            # this pipeline stage and feed both OMMAs.
            gate_acc.fill(0.0)
            up_acc.fill(0.0)
            peek = ml_pipeline.consumer_try_wait(cons_state)
            ml_pipeline.consumer_wait(cons_state, peek)
            csA_p = csA_fc1[None, None, None, cons_state.index]
            csB_p = csB_fc1[None, None, None, cons_state.index]
            csB_up_p = csB_up_fc1[None, None, None, cons_state.index]
            csSFA_p = csSFA_fc1[None, None, None, cons_state.index]
            csSFB_p = csSFB_fc1_half[None, None, None, cons_state.index]
            csSFB_up_p = csSFB_up_fc1_half[None, None, None, Int32(0)]
            if cons_state.index < Int32(self.ab_storage_stage):
                csSFB_up_p = csSFB_up_fc1_half[None, None, None, cons_state.index]
            else:
                csSFB_up_p = csSFB_up_fc1_extra_half
            cute.copy(
                smem_copy_A_fc1,
                csA_p[None, None, 0],
                crA_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_B_fc1,
                csB_p[None, None, 0],
                crB_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_B_fc1,
                csB_up_p[None, None, 0],
                crB_up_fc1[None, None, 0],
            )
            fz_csSFA_p = cute.filter_zeros(csSFA_p)
            fz_csSFB_p = cute.filter_zeros(csSFB_p)
            fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
            cute.copy(
                smem_copy_SFA_fc1,
                fz_csSFA_p[None, None, 0],
                fz_crSFA_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_SFB_fc1,
                fz_csSFB_p[None, None, 0],
                fz_crSFB_fc1_half[None, None, 0],
            )
            cute.copy(
                smem_copy_SFB_fc1,
                fz_csSFB_up_p[None, None, 0],
                fz_crSFB_up_fc1_half[None, None, 0],
            )
            for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):  # type: ignore[call-overload]
                for k_block_idx in cutlass.range_constexpr(fc1_num_k_blocks):
                    k_next = (
                        0 if k_block_idx + 1 == fc1_num_k_blocks else k_block_idx + 1
                    )
                    if k_block_idx == fc1_num_k_blocks - 1:
                        ml_pipeline.consumer_release(cons_state)
                        cons_state.advance()
                        peek = ml_pipeline.consumer_try_wait(cons_state)
                        csA_p = csA_fc1[None, None, None, cons_state.index]
                        csB_p = csB_fc1[None, None, None, cons_state.index]
                        csB_up_p = csB_up_fc1[None, None, None, cons_state.index]
                        csSFA_p = csSFA_fc1[None, None, None, cons_state.index]
                        csSFB_p = csSFB_fc1_half[None, None, None, cons_state.index]
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_p = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_p = csSFB_up_fc1_extra_half
                        fz_csSFA_p = cute.filter_zeros(csSFA_p)
                        fz_csSFB_p = cute.filter_zeros(csSFB_p)
                        fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
                        ml_pipeline.consumer_wait(cons_state, peek)
                    # Issue current Gate MMA first so the following
                    # LDS can overlap tensor-pipe work.
                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                        for _nt in cutlass.range_constexpr(fc1_n_tiles):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_fc1_half[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                gate_acc[None, _mt, _nt],
                                tCrA_fc1[None, _mt, k_block_idx],
                                tCrB_fc1[None, _nt, k_block_idx],
                                gate_acc[None, _mt, _nt],
                            )
                    if k_next > 0:
                        cute.copy(
                            smem_copy_A_fc1,
                            csA_p[None, None, k_next],
                            crA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_p[None, None, k_next],
                            crB_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_up_p[None, None, k_next],
                            crB_up_fc1[None, None, k_next],
                        )
                        fz_csSFA_cur = cute.filter_zeros(
                            csSFA_fc1[None, None, None, cons_state.index]
                        )
                        fz_csSFB_cur = cute.filter_zeros(
                            csSFB_fc1_half[None, None, None, cons_state.index]
                        )
                        csSFB_up_cur = csSFB_up_p
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_cur = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_cur = csSFB_up_fc1_extra_half
                        fz_csSFB_up_cur = cute.filter_zeros(csSFB_up_cur)
                        cute.copy(
                            smem_copy_SFA_fc1,
                            fz_csSFA_cur[None, None, k_next],
                            fz_crSFA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_cur[None, None, k_next],
                            fz_crSFB_fc1_half[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_up_cur[None, None, k_next],
                            fz_crSFB_up_fc1_half[None, None, k_next],
                        )
                    # Current Up consumes only current fragments;
                    # next-K64 fragments remain independent.
                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                        for _nt in cutlass.range_constexpr(fc1_n_tiles):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_up_fc1_half[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                up_acc[None, _mt, _nt],
                                tCrA_fc1[None, _mt, k_block_idx],
                                tCrB_up_fc1[None, _nt, k_block_idx],
                                up_acc[None, _mt, _nt],
                            )
                    # Preserve V98's conservative cross-stage
                    # boundary: load next-stage K64(0) after Up.
                    if k_next == 0:
                        cute.copy(
                            smem_copy_A_fc1,
                            csA_p[None, None, k_next],
                            crA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_p[None, None, k_next],
                            crB_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_up_p[None, None, k_next],
                            crB_up_fc1[None, None, k_next],
                        )
                        fz_csSFA_cur = cute.filter_zeros(
                            csSFA_fc1[None, None, None, cons_state.index]
                        )
                        fz_csSFB_cur = cute.filter_zeros(
                            csSFB_fc1_half[None, None, None, cons_state.index]
                        )
                        csSFB_up_cur = csSFB_up_p
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_cur = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_cur = csSFB_up_fc1_extra_half
                        fz_csSFB_up_cur = cute.filter_zeros(csSFB_up_cur)
                        cute.copy(
                            smem_copy_SFA_fc1,
                            fz_csSFA_cur[None, None, k_next],
                            fz_crSFA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_cur[None, None, k_next],
                            fz_crSFB_fc1_half[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_up_cur[None, None, k_next],
                            fz_crSFB_up_fc1_half[None, None, k_next],
                        )
            for k_block_idx in cutlass.range_constexpr(fc1_num_k_blocks):
                k_next = 0 if k_block_idx + 1 == fc1_num_k_blocks else k_block_idx + 1
                if k_block_idx == fc1_num_k_blocks - 1:
                    ml_pipeline.consumer_release(cons_state)
                    cons_state.advance()
                if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                    cute.copy(
                        smem_copy_A_fc1,
                        csA_p[None, None, k_next],
                        crA_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_B_fc1,
                        csB_p[None, None, k_next],
                        crB_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_B_fc1,
                        csB_up_p[None, None, k_next],
                        crB_up_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFA_fc1,
                        fz_csSFA_p[None, None, k_next],
                        fz_crSFA_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFB_fc1,
                        fz_csSFB_p[None, None, k_next],
                        fz_crSFB_fc1_half[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFB_fc1,
                        fz_csSFB_up_p[None, None, k_next],
                        fz_crSFB_up_fc1_half[None, None, k_next],
                    )
                for _mt in cutlass.range_constexpr(fc1_m_tiles):
                    for _nt in cutlass.range_constexpr(fc1_n_tiles):
                        mma_atom.set(
                            WarpField.SFA,
                            tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                        )
                        mma_atom.set(
                            WarpField.SFB,
                            tCrSFB_fc1_half[None, _nt, k_block_idx].iterator,
                        )
                        cute.gemm(
                            mma_atom,
                            gate_acc[None, _mt, _nt],
                            tCrA_fc1[None, _mt, k_block_idx],
                            tCrB_fc1[None, _nt, k_block_idx],
                            gate_acc[None, _mt, _nt],
                        )
                        mma_atom.set(
                            WarpField.SFB,
                            tCrSFB_up_fc1_half[None, _nt, k_block_idx].iterator,
                        )
                        cute.gemm(
                            mma_atom,
                            up_acc[None, _mt, _nt],
                            tCrA_fc1[None, _mt, k_block_idx],
                            tCrB_up_fc1[None, _nt, k_block_idx],
                            up_acc[None, _mt, _nt],
                        )

            if fc1_half == 1:
                # The final paired consumer has released every FC1
                # stage, but sC remains live through activation and Q1.
                # The main loop releases the producer after Q1 consumes sC.
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()
                # sC was backing FC1 Stage3 and therefore stayed unwritten
                # through both halves.  Materialize the held half0 BF16
                # activation only after the final aliased payload release.
                for hold_m in cutlass.range_constexpr(fc1_m_tiles):
                    for hold_n in cutlass.range_constexpr(fc1_n_tiles):
                        cute.copy(
                            fc1_tiled_copy_r2s,
                            fc1_tRS_rAct_hold[(None, hold_m, hold_n)],
                            fc1_tRS_sD[(None, hold_m, hold_n, 0)],
                        )

            # Consume this Gate/Up pair immediately.  Its FP32
            # accumulator lifetime ends before the next half starts.
            for mma_m in cutlass.range_constexpr(fc1_m_tiles):
                for mma_n in cutlass.range_constexpr(fc1_n_tiles):
                    full_mma_n = fc1_half * fc1_n_tiles + mma_n
                    gate_slice = fc1_tRS_rGate[(None, mma_m, mma_n)]
                    up_slice = fc1_tRS_rUp[(None, mma_m, mma_n)]
                    for elem_idx in cutlass.range_constexpr(cute.size(fc1_tRS_rAct)):
                        g = alpha_value * gate_slice[elem_idx]
                        u = alpha_value * up_slice[elem_idx]
                        fc1_tRS_rAct[elem_idx] = _dynamic_gated_activation_f32(
                            g,
                            u,
                            activation=self.activation,
                            limit=self.swiglu_limit,
                            alpha=self.swiglu_alpha,
                            beta=self.swiglu_beta,
                            fast_math=self.fast_math,
                        )
                    act_vec = fc1_tRS_rAct.load()
                    act_vec = act_vec.to(cutlass.BFloat16)
                    fc1_tRS_rAct_out.store(act_vec)
                    if fc1_half == 0:
                        fc1_tRS_rAct_hold[(None, mma_m, mma_n)].store(act_vec)
                    else:
                        cute.copy(
                            fc1_tiled_copy_r2s,
                            fc1_tRS_rAct_out,
                            fc1_tRS_sD[(None, mma_m, full_mma_n, 0)],
                        )

        return cons_state, up_cons_state

    @cute.jit
    def fc1_gate_up_swiglu_to_sC_tail(
        self,
        tidx,
        ml_pipeline,
        cons_state,
        up_pipeline,
        up_cons_state,
        fc1_tiled_mma,
        mma_atom,
        sSFB_fc1,
        sSFB_up_fc1,
        sSFB_up_fc1_extra,
        fc1_thr_mma,
        thr_ld_SFB_fc1,
        csA_fc1,
        csB_fc1,
        csB_up_fc1,
        csSFA_fc1,
        crA_fc1,
        crB_fc1,
        crB_up_fc1,
        crSFA_fc1,
        tCrA_fc1,
        tCrB_fc1,
        tCrB_up_fc1,
        tCrSFA_fc1,
        smem_copy_A_fc1,
        smem_copy_B_fc1,
        smem_copy_SFA_fc1,
        smem_copy_SFB_fc1,
        fc1_tiled_copy_r2s,
        fc1_tRS_sD,
        fc1_k_tile_cnt,
        fc1_num_k_blocks,
        fc1_m_tiles,
        fc1_n_tiles,
        alpha_value,
        valid_rows,
        warp_m_coord: Int32,
        task_expert_idx,
        global_scale,
        sC,
        sA,
        sfa_base_addr,
        epi_rest_m,
    ):
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        fc1_acc_shape = fc1_tiled_mma.partition_shape_C(
            (self.fc1_tile_shape_mnk[0], self.fc1_tile_shape_mnk[1])
        )
        gate_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)
        fc1_tRS_rGate = fc1_tiled_copy_r2s.retile(gate_acc)
        fc1_tRS_rUp = fc1_tiled_copy_r2s.retile(up_acc)
        fc1_tRS_rAct = cute.make_rmem_tensor(
            fc1_tRS_rGate[(None, 0, 0)].shape, self.acc_dtype
        )
        fc1_tRS_rAct_out = cute.make_rmem_tensor(
            fc1_tRS_rGate[(None, 0, 0)].shape, cutlass.BFloat16
        )
        fc1_tRS_rAct_hold = cute.make_rmem_tensor(fc1_tRS_rGate.shape, cutlass.BFloat16)

        # ============================================================
        # PHASE A: native dual-N64 FC1 for this logical N128 slice
        # ============================================================
        # One pipeline/state sequence covers two branch-paired
        # halves.  Each stage carries one A/SFA plus independent
        # Gate/Up B/SFB payloads; both OMMAs complete before release.
        cons_state.reset_count()
        for fc1_half in cutlass.range_constexpr(2):
            # SM120 packs SFB in physical N128 blocks.  Select the
            # live N64 half from the replayed block for both branches.
            sSFB_fc1_half = cute.local_tile(
                sSFB_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0, None),
            )
            tCrSFB_fc1_half = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                sSFB_fc1_half[None, None, 0],
                fc1_thr_mma,
                tidx,
            )
            csSFB_fc1_half = thr_ld_SFB_fc1.partition_S(sSFB_fc1_half)
            crSFB_fc1_half = thr_ld_SFB_fc1.retile(tCrSFB_fc1_half)
            sSFB_up_fc1_half = cute.local_tile(
                sSFB_up_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0, None),
            )
            tCrSFB_up_fc1_half = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                sSFB_up_fc1_half[None, None, 0],
                fc1_thr_mma,
                tidx,
            )
            csSFB_up_fc1_half = thr_ld_SFB_fc1.partition_S(sSFB_up_fc1_half)
            sSFB_up_fc1_extra_half = cute.local_tile(
                sSFB_up_fc1_extra,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (fc1_half, 0),
            )
            csSFB_up_fc1_extra_half = thr_ld_SFB_fc1.partition_S(sSFB_up_fc1_extra_half)
            crSFB_up_fc1_half = thr_ld_SFB_fc1.retile(tCrSFB_up_fc1_half)
            fz_crSFA_fc1 = cute.filter_zeros(crSFA_fc1)
            fz_crSFB_fc1_half = cute.filter_zeros(crSFB_fc1_half)
            fz_crSFB_up_fc1_half = cute.filter_zeros(crSFB_up_fc1_half)

            # Branch-paired Gate/Up N64: A/SFA are read once from
            # this pipeline stage and feed both OMMAs.
            gate_acc.fill(0.0)
            up_acc.fill(0.0)
            peek = ml_pipeline.consumer_try_wait(cons_state)
            ml_pipeline.consumer_wait(cons_state, peek)
            csA_p = csA_fc1[None, None, None, cons_state.index]
            csB_p = csB_fc1[None, None, None, cons_state.index]
            csB_up_p = csB_up_fc1[None, None, None, cons_state.index]
            csSFA_p = csSFA_fc1[None, None, None, cons_state.index]
            csSFB_p = csSFB_fc1_half[None, None, None, cons_state.index]
            csSFB_up_p = csSFB_up_fc1_half[None, None, None, Int32(0)]
            if cons_state.index < Int32(self.ab_storage_stage):
                csSFB_up_p = csSFB_up_fc1_half[None, None, None, cons_state.index]
            else:
                csSFB_up_p = csSFB_up_fc1_extra_half
            cute.copy(
                smem_copy_A_fc1,
                csA_p[None, None, 0],
                crA_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_B_fc1,
                csB_p[None, None, 0],
                crB_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_B_fc1,
                csB_up_p[None, None, 0],
                crB_up_fc1[None, None, 0],
            )
            fz_csSFA_p = cute.filter_zeros(csSFA_p)
            fz_csSFB_p = cute.filter_zeros(csSFB_p)
            fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
            cute.copy(
                smem_copy_SFA_fc1,
                fz_csSFA_p[None, None, 0],
                fz_crSFA_fc1[None, None, 0],
            )
            cute.copy(
                smem_copy_SFB_fc1,
                fz_csSFB_p[None, None, 0],
                fz_crSFB_fc1_half[None, None, 0],
            )
            cute.copy(
                smem_copy_SFB_fc1,
                fz_csSFB_up_p[None, None, 0],
                fz_crSFB_up_fc1_half[None, None, 0],
            )
            for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):  # type: ignore[call-overload]
                for k_block_idx in cutlass.range_constexpr(fc1_num_k_blocks):
                    k_next = (
                        0 if k_block_idx + 1 == fc1_num_k_blocks else k_block_idx + 1
                    )
                    if k_block_idx == fc1_num_k_blocks - 1:
                        ml_pipeline.consumer_release(cons_state)
                        cons_state.advance()
                        peek = ml_pipeline.consumer_try_wait(cons_state)
                        csA_p = csA_fc1[None, None, None, cons_state.index]
                        csB_p = csB_fc1[None, None, None, cons_state.index]
                        csB_up_p = csB_up_fc1[None, None, None, cons_state.index]
                        csSFA_p = csSFA_fc1[None, None, None, cons_state.index]
                        csSFB_p = csSFB_fc1_half[None, None, None, cons_state.index]
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_p = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_p = csSFB_up_fc1_extra_half
                        fz_csSFA_p = cute.filter_zeros(csSFA_p)
                        fz_csSFB_p = cute.filter_zeros(csSFB_p)
                        fz_csSFB_up_p = cute.filter_zeros(csSFB_up_p)
                        ml_pipeline.consumer_wait(cons_state, peek)
                    # Issue current Gate MMA first so the following
                    # LDS can overlap tensor-pipe work.
                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                        if valid_rows > Int32(_mt * 64) + warp_m_coord * Int32(16):
                            for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_fc1_half[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    gate_acc[None, _mt, _nt],
                                    tCrA_fc1[None, _mt, k_block_idx],
                                    tCrB_fc1[None, _nt, k_block_idx],
                                    gate_acc[None, _mt, _nt],
                                )
                    if k_next > 0:
                        cute.copy(
                            smem_copy_A_fc1,
                            csA_p[None, None, k_next],
                            crA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_p[None, None, k_next],
                            crB_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_up_p[None, None, k_next],
                            crB_up_fc1[None, None, k_next],
                        )
                        fz_csSFA_cur = cute.filter_zeros(
                            csSFA_fc1[None, None, None, cons_state.index]
                        )
                        fz_csSFB_cur = cute.filter_zeros(
                            csSFB_fc1_half[None, None, None, cons_state.index]
                        )
                        csSFB_up_cur = csSFB_up_p
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_cur = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_cur = csSFB_up_fc1_extra_half
                        fz_csSFB_up_cur = cute.filter_zeros(csSFB_up_cur)
                        cute.copy(
                            smem_copy_SFA_fc1,
                            fz_csSFA_cur[None, None, k_next],
                            fz_crSFA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_cur[None, None, k_next],
                            fz_crSFB_fc1_half[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_up_cur[None, None, k_next],
                            fz_crSFB_up_fc1_half[None, None, k_next],
                        )
                    # Current Up consumes only current fragments;
                    # next-K64 fragments remain independent.
                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                        if valid_rows > Int32(_mt * 64) + warp_m_coord * Int32(16):
                            for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_up_fc1_half[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    up_acc[None, _mt, _nt],
                                    tCrA_fc1[None, _mt, k_block_idx],
                                    tCrB_up_fc1[None, _nt, k_block_idx],
                                    up_acc[None, _mt, _nt],
                                )
                    # Preserve V98's conservative cross-stage
                    # boundary: load next-stage K64(0) after Up.
                    if k_next == 0:
                        cute.copy(
                            smem_copy_A_fc1,
                            csA_p[None, None, k_next],
                            crA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_p[None, None, k_next],
                            crB_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B_fc1,
                            csB_up_p[None, None, k_next],
                            crB_up_fc1[None, None, k_next],
                        )
                        fz_csSFA_cur = cute.filter_zeros(
                            csSFA_fc1[None, None, None, cons_state.index]
                        )
                        fz_csSFB_cur = cute.filter_zeros(
                            csSFB_fc1_half[None, None, None, cons_state.index]
                        )
                        csSFB_up_cur = csSFB_up_p
                        if cons_state.index < Int32(self.ab_storage_stage):
                            csSFB_up_cur = csSFB_up_fc1_half[
                                None, None, None, cons_state.index
                            ]
                        else:
                            csSFB_up_cur = csSFB_up_fc1_extra_half
                        fz_csSFB_up_cur = cute.filter_zeros(csSFB_up_cur)
                        cute.copy(
                            smem_copy_SFA_fc1,
                            fz_csSFA_cur[None, None, k_next],
                            fz_crSFA_fc1[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_cur[None, None, k_next],
                            fz_crSFB_fc1_half[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB_fc1,
                            fz_csSFB_up_cur[None, None, k_next],
                            fz_crSFB_up_fc1_half[None, None, k_next],
                        )
            for k_block_idx in cutlass.range_constexpr(fc1_num_k_blocks):
                k_next = 0 if k_block_idx + 1 == fc1_num_k_blocks else k_block_idx + 1
                if k_block_idx == fc1_num_k_blocks - 1:
                    ml_pipeline.consumer_release(cons_state)
                    cons_state.advance()
                if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                    cute.copy(
                        smem_copy_A_fc1,
                        csA_p[None, None, k_next],
                        crA_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_B_fc1,
                        csB_p[None, None, k_next],
                        crB_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_B_fc1,
                        csB_up_p[None, None, k_next],
                        crB_up_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFA_fc1,
                        fz_csSFA_p[None, None, k_next],
                        fz_crSFA_fc1[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFB_fc1,
                        fz_csSFB_p[None, None, k_next],
                        fz_crSFB_fc1_half[None, None, k_next],
                    )
                    cute.copy(
                        smem_copy_SFB_fc1,
                        fz_csSFB_up_p[None, None, k_next],
                        fz_crSFB_up_fc1_half[None, None, k_next],
                    )
                for _mt in cutlass.range_constexpr(fc1_m_tiles):
                    if valid_rows > Int32(_mt * 64) + warp_m_coord * Int32(16):
                        for _nt in cutlass.range_constexpr(fc1_n_tiles):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_fc1[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_fc1_half[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                gate_acc[None, _mt, _nt],
                                tCrA_fc1[None, _mt, k_block_idx],
                                tCrB_fc1[None, _nt, k_block_idx],
                                gate_acc[None, _mt, _nt],
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_up_fc1_half[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                up_acc[None, _mt, _nt],
                                tCrA_fc1[None, _mt, k_block_idx],
                                tCrB_up_fc1[None, _nt, k_block_idx],
                                up_acc[None, _mt, _nt],
                            )

            if fc1_half == 1:
                # The final paired consumer has released every FC1
                # stage, but sC remains live through activation and Q1.
                # The main loop releases the producer after Q1 consumes sC.
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()
                for hold_m in cutlass.range_constexpr(fc1_m_tiles):
                    if valid_rows > (Int32(hold_m * 64) + warp_m_coord * Int32(16)):
                        for hold_n in cutlass.range_constexpr(fc1_n_tiles):
                            cute.copy(
                                fc1_tiled_copy_r2s,
                                fc1_tRS_rAct_hold[(None, hold_m, hold_n)],
                                fc1_tRS_sD[(None, hold_m, hold_n, 0)],
                            )

            # Consume this Gate/Up pair immediately.  Its FP32
            # accumulator lifetime ends before the next half starts.
            for mma_m in cutlass.range_constexpr(fc1_m_tiles):
                if valid_rows > Int32(mma_m * 64) + warp_m_coord * Int32(16):
                    for mma_n in cutlass.range_constexpr(fc1_n_tiles):
                        full_mma_n = fc1_half * fc1_n_tiles + mma_n
                        gate_slice = fc1_tRS_rGate[(None, mma_m, mma_n)]
                        up_slice = fc1_tRS_rUp[(None, mma_m, mma_n)]
                        for elem_idx in cutlass.range_constexpr(
                            cute.size(fc1_tRS_rAct)
                        ):
                            g = alpha_value * gate_slice[elem_idx]
                            u = alpha_value * up_slice[elem_idx]
                            fc1_tRS_rAct[elem_idx] = _dynamic_gated_activation_f32(
                                g,
                                u,
                                activation=self.activation,
                                limit=self.swiglu_limit,
                                alpha=self.swiglu_alpha,
                                beta=self.swiglu_beta,
                                fast_math=self.fast_math,
                            )
                        act_vec = fc1_tRS_rAct.load()
                        act_vec = act_vec.to(cutlass.BFloat16)
                        fc1_tRS_rAct_out.store(act_vec)
                        if fc1_half == 0:
                            fc1_tRS_rAct_hold[(None, mma_m, mma_n)].store(act_vec)
                        else:
                            cute.copy(
                                fc1_tiled_copy_r2s,
                                fc1_tRS_rAct_out,
                                fc1_tRS_sD[(None, mma_m, full_mma_n, 0)],
                            )

        return cons_state, up_cons_state

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
        q1_a_stage_idx: Int32,
        defer_a: Int32,
        deferred_a_words: cute.Tensor,
        q1_sfa_stage_idx: Int32,
        defer_sfa: Int32,
        deferred_sfa_words: cute.Tensor,
        deferred_sfa_slot: Int32,
        epi_rest_m,
    ):
        # Q1 runs exactly once after both N64 halves.  sA remains
        # the full N128 activation consumed by one FC2/scatter pass.
        sA_u8 = cute.recast_tensor(sA[None, None, q1_a_stage_idx], cutlass.Uint8)
        packed_cols = Int32(self.tile_shape_mnk[2] // 2)
        sf_blocks_per_row = Int32(self.tile_shape_mnk[2] // 16)
        gs_value = global_scale[task_expert_idx].to(cutlass.Float32)
        if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(0.0):
            if self.fast_math:
                gs_value = rcp_approx_ftz(gs_value)
            else:
                gs_value = cutlass.Float32(1.0) / gs_value

        deferred_slot = Int32(0)
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
                    block_start = sf_block * Int32(16)

                    # A BF16x16 quantization block is two aligned N8
                    # segments in K_SW128.  Apply the same explicit swizzle
                    # transform as scatter, then replace sixteen LDS.U16 with
                    # two LDS.128 while preserving element/absmax order.
                    sc_element_offset_lo = Int32(
                        sC.layout(
                            (
                                local_row,
                                block_start,
                                gated_epi_buffer,
                            )
                        )
                    )
                    sc_element_offset_lo = sc_element_offset_lo ^ (
                        (sc_element_offset_lo & Int32(0x1C0)) >> Int32(3)
                    )
                    sc_element_offset_hi = Int32(
                        sC.layout(
                            (
                                local_row,
                                block_start + Int32(8),
                                gated_epi_buffer,
                            )
                        )
                    )
                    sc_element_offset_hi = sc_element_offset_hi ^ (
                        (sc_element_offset_hi & Int32(0x1C0)) >> Int32(3)
                    )
                    sc_smem_addr_lo = get_smem_ptr_as_int32(
                        sC,
                        sc_element_offset_lo,
                    )
                    sc_smem_addr_hi = get_smem_ptr_as_int32(
                        sC,
                        sc_element_offset_hi,
                    )
                    loaded_lo = load_shared_bf16x8_to_f32x8(sc_smem_addr_lo)
                    loaded_hi = load_shared_bf16x8_to_f32x8(sc_smem_addr_hi)
                    values = cute.make_rmem_tensor((16,), cutlass.Float32)
                    block_max = cutlass.Float32(0.0)
                    for elem_idx in cutlass.range_constexpr(8):
                        value = loaded_lo[elem_idx]
                        values[elem_idx] = value
                        block_max = fmax_f32(block_max, fabs_f32(value))
                    for elem_idx in cutlass.range_constexpr(8):
                        value = loaded_hi[elem_idx]
                        values[elem_idx + 8] = value
                        block_max = fmax_f32(block_max, fabs_f32(value))

                    packed64 = Uint64(0)
                    scale_byte = Uint8(0)
                    if self.fast_math:
                        packed64, scale_byte = quantize_block_fp4_fast(
                            values, block_max, gs_value
                        )
                    else:
                        packed64, scale_byte = quantize_block_fp4(
                            values, block_max, gs_value
                        )
                    packed_base = sf_block << Int32(3)
                    dst_pcol = row & Int32(63)
                    xor_bits = ((dst_pcol >> Int32(1)) & Int32(0x3)) << Int32(4)
                    row_high = row >> Int32(6)
                    if defer_a > Int32(0):
                        deferred_a_words[deferred_slot * Int32(2)] = Uint32(
                            packed64 & Uint64(0xFFFFFFFF)
                        )
                        deferred_a_words[deferred_slot * Int32(2) + Int32(1)] = Uint32(
                            packed64 >> Uint64(32)
                        )
                    else:
                        for byte_idx in cutlass.range_constexpr(8):
                            src_pcol = packed_base + Int32(byte_idx)
                            dst_row = ((src_pcol ^ xor_bits) << Int32(1)) + row_high
                            dst_flat = dst_row * packed_cols + dst_pcol
                            byte_val = Uint8(
                                (packed64 >> Uint64(byte_idx * 8)) & Uint64(0xFF)
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
                    if defer_sfa > Int32(0):
                        deferred_sfa_words[deferred_sfa_slot] = deferred_sfa_words[
                            deferred_sfa_slot
                        ] | (Uint32(scale_byte) << Uint32(deferred_slot * Int32(8)))
                    else:
                        st_shared_u8(
                            sfa_base_addr
                            + q1_sfa_stage_idx * sfa_stage_elements
                            + sf_raw_idx,
                            scale_byte,
                        )
                    deferred_slot += Int32(1)
                    quant_idx += Int32(self.num_mma_warps * self.num_threads_per_warp)
        return

    @cute.jit
    def flush_deferred_q1_a(
        self,
        tidx,
        valid_rows: Int32,
        deferred_a_words: cute.Tensor,
        sA: cute.Tensor,
        q1_a_stage_idx: Int32,
    ):
        sA_u8 = cute.recast_tensor(sA[None, None, q1_a_stage_idx], cutlass.Uint8)
        packed_cols = Int32(self.tile_shape_mnk[2] // 2)
        sf_blocks_per_row = Int32(self.tile_shape_mnk[2] // 16)
        deferred_slot = Int32(0)
        quant_idx = Int32(tidx)
        while quant_idx < valid_rows * sf_blocks_per_row:
            row = quant_idx // sf_blocks_per_row
            sf_block = quant_idx - row * sf_blocks_per_row
            packed64 = Uint64(deferred_a_words[deferred_slot * Int32(2)]) | (
                Uint64(deferred_a_words[deferred_slot * Int32(2) + Int32(1)])
                << Uint64(32)
            )
            packed_base = sf_block << Int32(3)
            dst_pcol = row & Int32(63)
            xor_bits = ((dst_pcol >> Int32(1)) & Int32(0x3)) << Int32(4)
            row_high = row >> Int32(6)
            for byte_idx in cutlass.range_constexpr(8):
                src_pcol = packed_base + Int32(byte_idx)
                dst_row = ((src_pcol ^ xor_bits) << Int32(1)) + row_high
                dst_flat = dst_row * packed_cols + dst_pcol
                sA_u8[dst_flat] = Uint8(
                    (packed64 >> Uint64(byte_idx * 8)) & Uint64(0xFF)
                )
            deferred_slot += Int32(1)
            quant_idx += Int32(self.num_mma_warps * self.num_threads_per_warp)

    @cute.jit
    def flush_deferred_q1_sfa(
        self,
        tidx,
        valid_rows: Int32,
        deferred_sfa_bits: Uint32,
        sfa_base_addr: Int32,
        sfa_stage_elements: Int32,
        q1_sfa_stage_idx: Int32,
    ):
        sf_blocks_per_row = Int32(self.tile_shape_mnk[2] // 16)
        deferred_slot = Int32(0)
        quant_idx = Int32(tidx)
        while quant_idx < valid_rows * sf_blocks_per_row:
            row = quant_idx // sf_blocks_per_row
            sf_block = quant_idx - row * sf_blocks_per_row
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
            scale_byte = Uint8(
                (deferred_sfa_bits >> Uint32(deferred_slot * Int32(8))) & Uint32(0xFF)
            )
            st_shared_u8(
                sfa_base_addr + q1_sfa_stage_idx * sfa_stage_elements + sf_raw_idx,
                scale_byte,
            )
            deferred_slot += Int32(1)
            quant_idx += Int32(self.num_mma_warps * self.num_threads_per_warp)

    @cute.jit
    def load_fc2_a_fragments(
        self,
        num_k_blocks,
        q1_a_stage_idx: Int32,
        q1_sfa_stage_idx: Int32,
        a_storage,
        a_fragments,
        a_copies,
    ):
        csA, csSFA = a_storage
        crA, crSFA = a_fragments
        smem_copy_A, smem_copy_SFA = a_copies

        csA_phase2 = csA[None, None, None, q1_a_stage_idx]
        csSFA_phase2 = csSFA[None, None, None, q1_sfa_stage_idx]

        # Consume all output tiles continuously from phase2_pipeline.

        # Hoist A-side register loads: sA is constant across all
        # FC2 output tiles (quantized intermediate). Load crA and
        # crSFA for all k-blocks once, reuse for all 32 tiles.
        fz_crSFA_p2 = cute.filter_zeros(crSFA)
        cute.copy(smem_copy_A, csA_phase2[None, None, 0], crA[None, None, 0])
        fz_csSFA_p2 = cute.filter_zeros(csSFA_phase2)
        cute.copy(
            smem_copy_SFA,
            fz_csSFA_p2[None, None, 0],
            fz_crSFA_p2[None, None, 0],
        )
        for _kb_pre in cutlass.range_constexpr(num_k_blocks - 1):
            k_pre = _kb_pre + 1
            cute.copy(
                smem_copy_A,
                csA_phase2[None, None, k_pre],
                crA[None, None, k_pre],
            )
            cute.copy(
                smem_copy_SFA,
                fz_csSFA_p2[None, None, k_pre],
                fz_crSFA_p2[None, None, k_pre],
            )

    @cute.jit
    def fc2_accumulate_slice(
        self,
        num_k_blocks,
        mma_atom,
        down_acc,
        pipeline_args,
        fc2_storage,
        fc2_fragments,
        fc2_copies,
    ):
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        phase2_pipeline, phase2_cons_state = pipeline_args
        csB, csB_extra, csSFB = fc2_storage
        tCrA, tCrB, tCrSFA, tCrSFB, crB, crSFB = fc2_fragments
        smem_copy_B, smem_copy_SFB = fc2_copies

        fc2_m_tiles = cute.size(tCrA, mode=[1])
        fc2_n_tiles = cute.size(tCrB, mode=[1])
        phase2_peek = phase2_pipeline.consumer_try_wait(phase2_cons_state)
        phase2_pipeline.consumer_wait(phase2_cons_state, phase2_peek)
        csSFB_phase2 = csSFB[None, None, None, phase2_cons_state.index]

        csB_phase2 = csB[None, None, None, Int32(0)]
        if phase2_cons_state.index < Int32(self.ab_storage_stage):
            csB_phase2 = csB[None, None, None, phase2_cons_state.index]
            cute.copy(
                smem_copy_B,
                csB_phase2[None, None, 0],
                crB[None, None, 0],
            )
        else:
            cute.copy(
                smem_copy_B,
                csB_extra[None, None, 0],
                crB[None, None, 0],
            )
        fz_csSFB = cute.filter_zeros(csSFB_phase2)
        fz_crSFB = cute.filter_zeros(crSFB)
        cute.copy(
            smem_copy_SFB,
            fz_csSFB[None, None, 0],
            fz_crSFB[None, None, 0],
        )

        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
            k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
            if k_block_idx == num_k_blocks - 1:
                phase2_pipeline.consumer_release(phase2_cons_state)
                phase2_cons_state.advance()
            if k_next > 0:
                if phase2_cons_state.index < Int32(self.ab_storage_stage):
                    cute.copy(
                        smem_copy_B,
                        csB_phase2[None, None, k_next],
                        crB[None, None, k_next],
                    )
                else:
                    cute.copy(
                        smem_copy_B,
                        csB_extra[None, None, k_next],
                        crB[None, None, k_next],
                    )
                cute.copy(
                    smem_copy_SFB,
                    fz_csSFB[None, None, k_next],
                    fz_crSFB[None, None, k_next],
                )
            for _mt in cutlass.range_constexpr(fc2_m_tiles):
                for _nt in cutlass.range_constexpr(fc2_n_tiles):
                    mma_atom.set(
                        WarpField.SFA,
                        tCrSFA[None, _mt, k_block_idx].iterator,
                    )
                    mma_atom.set(
                        WarpField.SFB,
                        tCrSFB[None, _nt, k_block_idx].iterator,
                    )
                    cute.gemm(
                        mma_atom,
                        down_acc[None, _mt, _nt],
                        tCrA[None, _mt, k_block_idx],
                        tCrB[None, _nt, k_block_idx],
                        down_acc[None, _mt, _nt],
                    )
        return phase2_cons_state

    @cute.jit
    def fc2_accumulate_slice_tail(
        self,
        num_k_blocks,
        mma_atom,
        down_acc,
        valid_rows: Int32,
        warp_m_coord: Int32,
        pipeline_args,
        fc2_storage,
        fc2_fragments,
        fc2_copies,
    ):
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        phase2_pipeline, phase2_cons_state = pipeline_args
        csB, csB_extra, csSFB = fc2_storage
        tCrA, tCrB, tCrSFA, tCrSFB, crB, crSFB = fc2_fragments
        smem_copy_B, smem_copy_SFB = fc2_copies

        fc2_m_tiles = cute.size(tCrA, mode=[1])
        fc2_n_tiles = cute.size(tCrB, mode=[1])
        phase2_peek = phase2_pipeline.consumer_try_wait(phase2_cons_state)
        phase2_pipeline.consumer_wait(phase2_cons_state, phase2_peek)
        csSFB_phase2 = csSFB[None, None, None, phase2_cons_state.index]

        csB_phase2 = csB[None, None, None, Int32(0)]
        if phase2_cons_state.index < Int32(self.ab_storage_stage):
            csB_phase2 = csB[None, None, None, phase2_cons_state.index]
            cute.copy(
                smem_copy_B,
                csB_phase2[None, None, 0],
                crB[None, None, 0],
            )
        else:
            cute.copy(
                smem_copy_B,
                csB_extra[None, None, 0],
                crB[None, None, 0],
            )
        fz_csSFB = cute.filter_zeros(csSFB_phase2)
        fz_crSFB = cute.filter_zeros(crSFB)
        cute.copy(
            smem_copy_SFB,
            fz_csSFB[None, None, 0],
            fz_crSFB[None, None, 0],
        )

        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
            k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
            if k_block_idx == num_k_blocks - 1:
                phase2_pipeline.consumer_release(phase2_cons_state)
                phase2_cons_state.advance()
            if k_next > 0:
                if phase2_cons_state.index < Int32(self.ab_storage_stage):
                    cute.copy(
                        smem_copy_B,
                        csB_phase2[None, None, k_next],
                        crB[None, None, k_next],
                    )
                else:
                    cute.copy(
                        smem_copy_B,
                        csB_extra[None, None, k_next],
                        crB[None, None, k_next],
                    )
                cute.copy(
                    smem_copy_SFB,
                    fz_csSFB[None, None, k_next],
                    fz_crSFB[None, None, k_next],
                )
            for _mt in cutlass.range_constexpr(fc2_m_tiles):
                if valid_rows > Int32(_mt * 64) + warp_m_coord * Int32(16):
                    for _nt in cutlass.range_constexpr(fc2_n_tiles):
                        mma_atom.set(
                            WarpField.SFA,
                            tCrSFA[None, _mt, k_block_idx].iterator,
                        )
                        mma_atom.set(
                            WarpField.SFB,
                            tCrSFB[None, _nt, k_block_idx].iterator,
                        )
                        cute.gemm(
                            mma_atom,
                            down_acc[None, _mt, _nt],
                            tCrA[None, _mt, k_block_idx],
                            tCrB[None, _nt, k_block_idx],
                            down_acc[None, _mt, _nt],
                        )
        return phase2_cons_state

    @cute.jit
    def fc2_epilogue_to_sC(
        self,
        acc_shape,
        down_alpha_value,
        down_acc,
        sC,
        tiled_copy_r2s,
        thr_copy_r2s,
        tRS_sD,
    ):
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
        tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)
        tRS_rDown = tiled_copy_r2s.retile(down_acc)
        mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rDown, mode=[1])
        mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rDown, mode=[2])
        epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
        mma_m_per_epi = self.epi_tile[0] // mma_tile_m
        mma_n_per_epi = self.epi_tile[1] // mma_tile_n

        for epi_m in cutlass.range_constexpr(epi_rest_m):
            for mma_n_in_epi in cutlass.range_constexpr(mma_n_per_epi):
                for mma_m_in_epi in cutlass.range_constexpr(mma_m_per_epi):
                    mma_n = mma_n_in_epi
                    mma_m = epi_m * mma_m_per_epi + mma_m_in_epi
                    tRS_rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                    down_epi_acc_slice = down_acc[(None, mma_m, mma_n)]
                    for elem_idx in cutlass.range_constexpr(cute.size(tRS_rD_slice)):
                        tRS_rD_slice[elem_idx] = down_epi_acc_slice[elem_idx]
            acc_vec = tRS_rD.load()
            acc_vec = acc_vec.to(cutlass.BFloat16)
            tRS_rD_out.store(acc_vec)
            epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s,
                tRS_rD_out,
                tRS_sD[(None, None, None, epi_buffer)],
            )

    @cute.jit
    def scatter_sC_to_gmem(
        self,
        tidx,
        output_tile_idx,
        valid_rows: Int32,
        sC: cute.Tensor,
        tRS_sD: cute.Tensor,
        scatter_output: cute.Tensor,
        scatter_tok_base_addr: Int32,
        scatter_weight_base_addr: Int32,
        down_alpha_value,
    ):
        epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
        scatter_N = Int32(scatter_output.shape[1])
        lane_id = Int32(tidx) & Int32(31)
        warp_in_tile = Int32(tidx) >> Int32(5)
        warp_m_base = (warp_in_tile >> Int32(1)) * Int32(32)
        warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)

        # Scatter using precomputed metadata (no redundant gmem loads)
        tile_n_base_cur = output_tile_idx * Int32(self.tile_shape_mnk[1])
        for epi_m in cutlass.range_constexpr(epi_rest_m):
            epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])

            # Per-warp scatter: all eight math warps cover one disjoint
            # sC strip (32 M-rows x 64 N-cols).
            warp_epi_rows = valid_rows - rows_offset - warp_m_base
            if warp_epi_rows > Int32(32):
                warp_epi_rows = Int32(32)
            if warp_epi_rows < Int32(0):
                warp_epi_rows = Int32(0)

            if scatter_output.shape[0] <= Int32(2048):
                # One work item owns two adjacent N8 vectors from the same
                # M row.  Relative to the original 256-vector round-robin loop,
                # this keeps all 32 lanes active while sharing one token/weight
                # metadata load across two reductions.  Unlike row ownership, a
                # lane never serializes all eight reductions of one row.
                tile_pair_cols = Int32(64) // Int32(16)
                pair_idx = lane_id
                while pair_idx < warp_epi_rows * tile_pair_cols:
                    local_row = pair_idx // tile_pair_cols
                    local_pair_col = pair_idx - local_row * tile_pair_cols
                    local_col_base = warp_n_base + local_pair_col * Int32(16)
                    cached_row = rows_offset + warp_m_base + local_row
                    tok, wv = load_shared_i32_f32_pair(
                        scatter_tok_base_addr + cached_row * Int32(8)
                    )
                    for pair_half in cutlass.range_constexpr(2):
                        local_col = local_col_base + Int32(pair_half) * Int32(8)
                        global_col = tile_n_base_cur + local_col
                        # Preserve the K_SW128 address transform independently
                        # for both N8 reductions in the pair.
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
                        scatter_add_weighted_bf16x8_packed_alpha(
                            get_ptr_as_int64(
                                scatter_output, tok * scatter_N + global_col
                            ),
                            sc_smem_addr,
                            wv,
                            down_alpha_value,
                        )
                    pair_idx += Int32(self.num_threads_per_warp)
            else:
                tile_vec_cols = Int32(64) // Int32(8)
                vec_idx = lane_id
                while vec_idx < warp_epi_rows * tile_vec_cols:
                    local_row = vec_idx // tile_vec_cols
                    local_vec_col = vec_idx - local_row * tile_vec_cols
                    local_col = warp_n_base + local_vec_col * Int32(8)
                    global_col = tile_n_base_cur + local_col
                    cached_row = rows_offset + warp_m_base + local_row
                    tok, wv = load_shared_i32_f32_pair(
                        scatter_tok_base_addr + cached_row * Int32(8)
                    )
                    # Preserve the K_SW128 address transform: compute the
                    # unswizzled outer offset through sC.layout, then explicitly
                    # apply S<3,4,3> in BF16 element units before stripping the
                    # SMEM pointer metadata.  A raw pointer does not retain CuTe's
                    # swizzle transform.
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
                    scatter_add_weighted_bf16x8_packed_alpha(
                        get_ptr_as_int64(scatter_output, tok * scatter_N + global_col),
                        sc_smem_addr,
                        wv,
                        down_alpha_value,
                    )
                    vec_idx += Int32(self.num_threads_per_warp)

    @cute.jit
    def load_fc1_tma_slice(
        self,
        intermediate_slice: Int32,
        wait_for_prior_slice: Int32,
        task_expert_idx: Int32,
        gate_tile_cnt,
        fc1_k_tile_cnt,
        prod_state,
        ml_pipeline,
        up_prod_state,
        up_pipeline,
        tma_inputs,
        gmem_partitions,
        smem_partitions,
    ):
        tma_a, tma_b_w13, tma_sfa, tma_sfb_w13 = tma_inputs
        tAgA_mk, tAgSFA_mk, tBgB_w13, tBgSFB_w13 = gmem_partitions
        (
            tAsA,
            tAsSFA,
            tBsB_w13,
            tBsB_w13_up,
            tBsSFB_w13,
            tBsSFB_w13_up,
            tBsSFB_w13_up_extra,
        ) = smem_partitions

        # FC1 producer follows the same continuous order as the
        # consumer.  Each logical N128 slice maps to two native B64
        # halves.  Within a half, Gate/Up share one A/SFA stage and
        # use independent B/SFB destinations under one barrier.
        prod_state.reset_count()
        gate_wait_pending = wait_for_prior_slice
        for fc1_half in cutlass.range_constexpr(2):
            native_up_slice_idx = intermediate_slice * Int32(2) + Int32(fc1_half)
            native_gate_slice_idx = (intermediate_slice + gate_tile_cnt) * Int32(
                2
            ) + Int32(fc1_half)
            tBgB_w13_gate_nk = tBgB_w13[
                (
                    None,
                    native_gate_slice_idx,
                    None,
                    task_expert_idx,
                )
            ]
            tBgB_w13_up_nk = tBgB_w13[
                (
                    None,
                    native_up_slice_idx,
                    None,
                    task_expert_idx,
                )
            ]
            tBgSFB_w13_gate_nk = tBgSFB_w13[
                (
                    None,
                    intermediate_slice + gate_tile_cnt,
                    None,
                    task_expert_idx,
                )
            ]
            tBgSFB_w13_up_nk = tBgSFB_w13[
                (
                    None,
                    intermediate_slice,
                    None,
                    task_expert_idx,
                )
            ]

            # ---- Branch-paired Gate/Up N64 ----
            for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                # Only physical Stage2 aliases sC.  Ordinary pipeline
                # release is sufficient for Stage0/1, so allow those stages
                # to prefetch before Q1 releases the prior slice's sC.
                if gate_wait_pending > Int32(0) and prod_state.index == Int32(
                    self.ab_storage_stage
                ):
                    self.pass_gate_barrier.wait_unaligned()
                    gate_wait_pending = Int32(0)
                ml_pipeline.producer_acquire(prod_state)
                cute.copy(
                    tma_a,
                    tAgA_mk[(None, k_tile)],
                    tAsA[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                cute.copy(
                    tma_b_w13,
                    tBgB_w13_gate_nk[(None, k_tile)],
                    tBsB_w13[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                cute.copy(
                    tma_b_w13,
                    tBgB_w13_up_nk[(None, k_tile)],
                    tBsB_w13_up[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                cute.copy(
                    tma_sfa,
                    tAgSFA_mk[(None, k_tile)],
                    tAsSFA[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                cute.copy(
                    tma_sfb_w13,
                    tBgSFB_w13_gate_nk[(None, k_tile)],
                    tBsSFB_w13[(None, prod_state.index)],
                    tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                )
                if prod_state.index < Int32(self.ab_storage_stage):
                    cute.copy(
                        tma_sfb_w13,
                        tBgSFB_w13_up_nk[(None, k_tile)],
                        tBsSFB_w13_up[(None, prod_state.index)],
                        tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                    )
                else:
                    cute.copy(
                        tma_sfb_w13,
                        tBgSFB_w13_up_nk[(None, k_tile)],
                        tBsSFB_w13_up_extra,
                        tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                    )
                ml_pipeline.producer_commit(prod_state)
                prod_state.advance()

        return prod_state, up_prod_state

    @cute.jit
    def load_fc2_tma_tile(
        self,
        intermediate_slice: Int32,
        output_tile_idx: Int32,
        task_expert_idx: Int32,
        phase2_prod_state,
        phase2_pipeline,
        tma_inputs,
        gmem_partitions,
        smem_partitions,
    ):
        tma_b_down, tma_sfb_down = tma_inputs
        tBgB_down, tBgSFB_down = gmem_partitions
        tBsB_down, tBsB_down_extra, tBsSFB_down = smem_partitions

        phase2_pipeline.producer_acquire(phase2_prod_state)
        if phase2_prod_state.index < Int32(self.ab_storage_stage):
            cute.copy(
                tma_b_down,
                tBgB_down[
                    (
                        None,
                        output_tile_idx,
                        intermediate_slice,
                        task_expert_idx,
                    )
                ],
                tBsB_down[(None, phase2_prod_state.index)],
                tma_bar_ptr=phase2_pipeline.producer_get_barrier(phase2_prod_state),
            )
        else:
            cute.copy(
                tma_b_down,
                tBgB_down[
                    (
                        None,
                        output_tile_idx,
                        intermediate_slice,
                        task_expert_idx,
                    )
                ],
                tBsB_down_extra,
                tma_bar_ptr=phase2_pipeline.producer_get_barrier(phase2_prod_state),
            )
        cute.copy(
            tma_sfb_down,
            tBgSFB_down[
                (
                    None,
                    output_tile_idx,
                    intermediate_slice,
                    task_expert_idx,
                )
            ],
            tBsSFB_down[(None, phase2_prod_state.index)],
            tma_bar_ptr=phase2_pipeline.producer_get_barrier(phase2_prod_state),
        )
        phase2_pipeline.producer_commit(phase2_prod_state)
        phase2_prod_state.advance()
        return phase2_prod_state

    @cute.jit
    def initialize_route_q0_and_publish(
        self,
        thread_info,
        route_inputs,
        route_outputs,
        routing_state,
        task_queue,
        resident_barriers,
        shared_addresses,
        launch_params: DynamicLaunchParams,
    ):
        tidx, bidz, gdim_z, warp_idx, is_cta_leader = thread_info
        a_input, topk_ids, topk_weights, input_global_scale = route_inputs
        (
            packed_a_storage,
            scale_storage,
            scatter_output,
            token_map,
            token_weights,
        ) = route_outputs
        expert_write_rows, expert_tile_base, pair_head = routing_state
        (
            task_head,
            task_tail,
            task_expert,
            task_valid_rows,
        ) = task_queue
        barrier_count, barrier_epoch = resident_barriers
        (
            ctrl_base_addr,
            route_phys_rows_addr,
            route_expert_ids_addr,
            q0_input_stage_base_addr,
            q0_bulk_barrier_addr,
        ) = shared_addresses

        num_tokens = Int32(a_input.shape[0])
        cols = Int32(a_input.shape[1])
        scatter_base = scatter_output.iterator.toint()
        row_counts = launch_params.row_counts
        num_experts = Int32(row_counts.shape[0])
        sf_blocks_per_row = cols // Int32(16)
        output_bytes_per_row = cols // Int32(2)
        cols_u32 = cols // Int32(2)
        scatter_output_u32 = cute.recast_tensor(scatter_output, cutlass.Uint32)
        total_pairs = Int32(topk_ids.shape[0])
        num_topk = total_pairs // num_tokens
        flat_tid = Int32(bidz) * Int32(self.threads_per_cta) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(self.threads_per_cta)
        num_k_tiles = (cols + Int32(63)) // Int32(64)
        route_gate_tile_cnt = launch_params.gate_tile_cnt
        task_slice_chunk = Int32(_TASK_SLICE_CHUNK)
        if num_tokens <= Int32(2048):
            task_slice_chunk = Int32(2)

        # Phase 0: cooperative init — zero routing state, queue state, and output.
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

        if flat_tid == Int32(0):
            pair_head[Int32(0)] = Int32(0)
            task_head[Int32(0)] = Int32(0)
            task_tail[Int32(0)] = Int32(0)

        cute.arch.sync_threads()
        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        # Phase 1: aggregate routed rows per CTA before publishing the
        # 256 expert subtotals globally.  The first 2304 bytes of sC
        # hold route caches; the following aligned 1 KiB is idle here.
        route_hist_addr = route_expert_ids_addr + Int32(
            (self.num_mma_warps + 1) * 32 * 4
        )
        hist_bin = tidx
        while hist_bin < num_experts:
            st_shared_i32(route_hist_addr + hist_bin * Int32(4), Int32(0))
            hist_bin += Int32((self.num_mma_warps + 1) * 32)
        cute.arch.sync_threads()

        hist_idx = flat_tid
        while hist_idx < total_pairs:
            expert_id = topk_ids[hist_idx].to(Int32)
            atomic_add_shared_i32(route_hist_addr + expert_id * Int32(4), Int32(1))
            hist_idx += flat_stride
        cute.arch.sync_threads()

        hist_bin = tidx
        while hist_bin < num_experts:
            subtotal = ld_shared_i32_relaxed(route_hist_addr + hist_bin * Int32(4))
            if subtotal > Int32(0):
                atomic_add_global_i32(get_ptr_as_int64(row_counts, hist_bin), subtotal)
            hist_bin += Int32((self.num_mma_warps + 1) * 32)

        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        if (
            num_experts == Int32(256)
            and bidz == Int32(0)
            and warp_idx < Int32(self.num_mma_warps)
        ):
            prefix_lane = Int32(tidx) & Int32(31)
            rows = row_counts[tidx]
            tile_count = (rows + Int32(self.tile_shape_mnk[0]) - Int32(1)) // Int32(
                self.tile_shape_mnk[0]
            )

            warp_inclusive = tile_count
            for scan_stage in cutlass.range_constexpr(5):
                scan_offset = Int32(1 << scan_stage)
                scan_value = cute.arch.shuffle_sync(
                    warp_inclusive, prefix_lane - Int32(scan_offset)
                )
                if prefix_lane >= Int32(scan_offset):
                    warp_inclusive += scan_value
            warp_exclusive = warp_inclusive - tile_count

            if prefix_lane == Int32(31):
                st_shared_i32(
                    route_hist_addr + warp_idx * Int32(4),
                    warp_inclusive,
                )
            self.epilog_sync_barrier.arrive_and_wait()

            if warp_idx == Int32(0):
                warp_total = Int32(0)
                if prefix_lane < Int32(self.num_mma_warps):
                    warp_total = ld_shared_i32_relaxed(
                        route_hist_addr + prefix_lane * Int32(4)
                    )
                warp_sum_inclusive = warp_total
                for scan_stage in cutlass.range_constexpr(5):
                    scan_offset = Int32(1 << scan_stage)
                    scan_value = cute.arch.shuffle_sync(
                        warp_sum_inclusive,
                        prefix_lane - Int32(scan_offset),
                    )
                    if prefix_lane >= Int32(scan_offset):
                        warp_sum_inclusive += scan_value
                if prefix_lane < Int32(self.num_mma_warps):
                    st_shared_i32(
                        route_hist_addr + prefix_lane * Int32(4),
                        warp_sum_inclusive - warp_total,
                    )
                if prefix_lane == Int32(self.num_mma_warps - 1):
                    _st_shared_i32(ctrl_base_addr + Int32(0), warp_sum_inclusive)
            self.epilog_sync_barrier.arrive_and_wait()

            warp_base = ld_shared_i32_relaxed(route_hist_addr + warp_idx * Int32(4))
            expert_tile_base[tidx] = warp_base + warp_exclusive
            if tidx == Int32(0):
                expert_tile_base[num_experts] = _ld_shared_i32(
                    ctrl_base_addr + Int32(0)
                )
        elif num_experts != Int32(256) and flat_tid == Int32(0):
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

        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        # Phase 2: the TMA warp stages only as many contiguous BF16 rows as
        # fit in the aliased sC backing.  The remaining math warps stay idle
        # during Q0 but remain active in FC1/Q1/FC2/scatter.
        if tidx == Int32(0):
            q0_bulk_barrier_init(q0_bulk_barrier_addr)
        cute.arch.sync_threads()
        lane_id = Int32(tidx) & Int32(31)
        _num_cta_warps = Int32(self.num_mma_warps + 1)
        # pair_head is a token counter in the token-major overlay.  sC holds
        # tile_M * tile_N BF16 values, so tile-area/cols is its full-token
        # capacity.  Keep this expression region-local for CuTe isolation.
        producer_batch_tokens = (
            Int32(self.tile_shape_mnk[0] * self.tile_shape_mnk[1]) // cols
        )
        if producer_batch_tokens > Int32(self.num_mma_warps):
            producer_batch_tokens = Int32(self.num_mma_warps)
        shared_input_gs_value = cutlass.Float32(0.0)
        if cutlass.const_expr(self.share_input_across_experts):
            shared_input_gs_value = input_global_scale[Int32(0)].to(cutlass.Float32)
            if (
                self.input_scales_are_reciprocal
                and shared_input_gs_value != cutlass.Float32(0.0)
            ):
                if self.fast_math:
                    shared_input_gs_value = rcp_approx_ftz(shared_input_gs_value)
                else:
                    shared_input_gs_value = cutlass.Float32(1.0) / shared_input_gs_value
        pair_idx = Int32(0)
        expert_id = Int32(0)
        token_idx = Int32(0)
        weight = cutlass.Float32(0.0)
        row = Int32(0)
        phys_tile = Int32(0)
        phys_row = Int32(0)
        produce_active = Int32(1)
        q0_bulk_phase = Int32(0)
        while produce_active > Int32(0):
            batch_base = Int32(0)
            if is_cta_leader > Int32(0):
                claim_count = producer_batch_tokens
                batch_base = atomic_add_global_i32(
                    get_ptr_as_int64(pair_head, Int32(0)),
                    claim_count,
                )
                _st_shared_i32(ctrl_base_addr + Int32(28), batch_base)
            cute.arch.sync_threads()
            batch_base = _ld_shared_i32(ctrl_base_addr + Int32(28))
            producer_limit = num_tokens
            if batch_base >= producer_limit:
                produce_active = Int32(0)
            else:
                staged_tokens = num_tokens - batch_base
                if staged_tokens > producer_batch_tokens:
                    staged_tokens = producer_batch_tokens
                first_copy_tokens = staged_tokens
                if first_copy_tokens > Int32(4):
                    first_copy_tokens = Int32(4)
                second_copy_tokens = staged_tokens - first_copy_tokens
                first_copy_bytes = first_copy_tokens * cols * Int32(2)
                second_copy_bytes = second_copy_tokens * cols * Int32(2)
                if warp_idx == Int32(self.num_mma_warps):
                    if lane_id == Int32(0):
                        input_batch_addr = Int64(a_input.iterator.toint()) + Int64(
                            batch_base
                        ) * Int64(cols) * Int64(2)
                        if first_copy_bytes > Int32(0):
                            q0_cp_async_bulk(
                                q0_input_stage_base_addr,
                                input_batch_addr,
                                first_copy_bytes,
                                q0_bulk_barrier_addr,
                            )
                        if second_copy_bytes > Int32(0):
                            q0_cp_async_bulk(
                                q0_input_stage_base_addr + Int32(4) * cols * Int32(2),
                                input_batch_addr + Int64(4) * Int64(cols) * Int64(2),
                                second_copy_bytes,
                                q0_bulk_barrier_addr,
                            )
                        q0_bulk_arrive_expect_tx(
                            q0_bulk_barrier_addr,
                            first_copy_bytes + second_copy_bytes,
                        )

                if cutlass.const_expr(self.share_input_across_experts):
                    token_idx = batch_base + warp_idx
                    if warp_idx < producer_batch_tokens and token_idx < num_tokens:
                        route_slot_base = warp_idx * Int32(32)
                        if lane_id == Int32(0):
                            topk_slot = Int32(0)
                            while topk_slot < num_topk:
                                pair_idx = token_idx * num_topk + topk_slot
                                expert_id = topk_ids[pair_idx].to(Int32)
                                weight = topk_weights[pair_idx].to(cutlass.Float32)
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
                                st_global_i32(
                                    get_ptr_as_int64(token_map, phys_row), token_idx
                                )
                                st_global_f32(
                                    get_ptr_as_int64(token_weights, phys_row), weight
                                )
                                slot = route_slot_base + topk_slot
                                _st_shared_i32(
                                    route_phys_rows_addr + slot * Int32(4), phys_row
                                )
                                _st_shared_i32(
                                    route_expert_ids_addr + slot * Int32(4), expert_id
                                )
                                topk_slot += Int32(1)
                        cute.arch.sync_warp()
                        q0_ready = q0_bulk_try_wait(q0_bulk_barrier_addr, q0_bulk_phase)
                        while q0_ready == Int32(0):
                            q0_ready = q0_bulk_try_wait(
                                q0_bulk_barrier_addr, q0_bulk_phase
                            )

                        gs_value = shared_input_gs_value
                        if num_topk == Int32(8):
                            route_output_base = cute.make_rmem_tensor((8,), Int32)
                            route_scale_base = cute.make_rmem_tensor((8,), Int32)
                            for cache_slot in cutlass.range_constexpr(8):
                                slot = route_slot_base + Int32(cache_slot)
                                phys_row = _ld_shared_i32(
                                    route_phys_rows_addr + slot * Int32(4)
                                )
                                phys_tile = phys_row // Int32(self.tile_shape_mnk[0])
                                tile_row = phys_row - phys_tile * Int32(
                                    self.tile_shape_mnk[0]
                                )
                                route_output_base[cache_slot] = (
                                    phys_row * output_bytes_per_row
                                )
                                route_scale_base[cache_slot] = (
                                    phys_tile * num_k_tiles * Int32(32 * 4 * 4)
                                    + (tile_row % Int32(32)) * Int32(4 * 4)
                                    + ((tile_row % Int32(32 * 4)) // Int32(32))
                                    * Int32(4)
                                )

                            sf_idx = lane_id
                            while sf_idx < sf_blocks_per_row:
                                block_start = sf_idx * Int32(16)
                                loaded_values = load_shared_bf16x16_to_f32x16(
                                    q0_input_stage_base_addr
                                    + warp_idx * cols * Int32(2)
                                    + block_start * Int32(2)
                                )
                                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(16):
                                    value = loaded_values[elem_idx]
                                    values[elem_idx] = value
                                    block_max = fmax_f32(block_max, fabs_f32(value))
                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )

                                k_tile_idx = sf_idx // Int32(4)
                                scale_k_base = k_tile_idx * Int32(32 * 4 * 4) + (
                                    sf_idx % Int32(4)
                                )
                                for cache_slot in cutlass.range_constexpr(8):
                                    output_offset = route_output_base[
                                        cache_slot
                                    ] + sf_idx * Int32(8)
                                    st_global_u64_adaptive_l2(
                                        num_tokens,
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    scale_storage[
                                        route_scale_base[cache_slot] + scale_k_base
                                    ] = scale_byte
                                sf_idx += Int32(32)
                        else:
                            sf_idx = lane_id
                            while sf_idx < sf_blocks_per_row:
                                block_start = sf_idx * Int32(16)
                                loaded_values = load_shared_bf16x16_to_f32x16(
                                    q0_input_stage_base_addr
                                    + warp_idx * cols * Int32(2)
                                    + block_start * Int32(2)
                                )
                                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(16):
                                    value = loaded_values[elem_idx]
                                    values[elem_idx] = value
                                    block_max = fmax_f32(block_max, fabs_f32(value))
                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )

                                topk_slot = Int32(0)
                                while topk_slot < num_topk:
                                    slot = route_slot_base + topk_slot
                                    phys_row = _ld_shared_i32(
                                        route_phys_rows_addr + slot * Int32(4)
                                    )
                                    phys_tile = phys_row // Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    tile_row = phys_row - phys_tile * Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    output_offset = (
                                        phys_row * output_bytes_per_row
                                        + sf_idx * Int32(8)
                                    )
                                    st_global_u64_adaptive_l2(
                                        num_tokens,
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    k_tile_idx = sf_idx // Int32(4)
                                    outer_m_idx = tile_row % Int32(32)
                                    inner_m_idx = (tile_row % Int32(32 * 4)) // Int32(
                                        32
                                    )
                                    inner_k_idx = sf_idx % Int32(4)
                                    scale_offset = (
                                        phys_tile * num_k_tiles * Int32(32 * 4 * 4)
                                        + k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    scale_storage[scale_offset] = scale_byte
                                    topk_slot += Int32(1)
                                sf_idx += Int32(32)

                else:
                    # Each math warp owns one token and handles all of its
                    # routes.  Keep a 16-entry register cache so both the
                    # Qwen topk=8 and topk=10 shapes use this shared-load path.
                    token_idx = batch_base + warp_idx
                    if warp_idx < producer_batch_tokens and token_idx < num_tokens:
                        route_slot_base = warp_idx * Int32(32)
                        if lane_id == Int32(0):
                            topk_slot = Int32(0)
                            while topk_slot < num_topk:
                                pair_idx = token_idx * num_topk + topk_slot
                                expert_id = topk_ids[pair_idx].to(Int32)
                                weight = topk_weights[pair_idx].to(cutlass.Float32)
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
                                st_global_i32(
                                    get_ptr_as_int64(token_map, phys_row), token_idx
                                )
                                st_global_f32(
                                    get_ptr_as_int64(token_weights, phys_row), weight
                                )

                                route_slot = route_slot_base + topk_slot
                                _st_shared_i32(
                                    route_phys_rows_addr + route_slot * Int32(4),
                                    phys_row,
                                )
                                _st_shared_i32(
                                    route_expert_ids_addr + route_slot * Int32(4),
                                    expert_id,
                                )

                                topk_slot += Int32(1)
                        cute.arch.sync_warp()
                        q0_ready = q0_bulk_try_wait(q0_bulk_barrier_addr, q0_bulk_phase)
                        while q0_ready == Int32(0):
                            q0_ready = q0_bulk_try_wait(
                                q0_bulk_barrier_addr, q0_bulk_phase
                            )

                        # Preserve the baseline's per-lane scale load and
                        # reciprocal work.  Hoist it out of the block loop,
                        # but do not introduce a 32x broadcast optimization.
                        route_gs = cute.make_rmem_tensor((16,), cutlass.Float32)
                        cache_slot = Int32(0)
                        while cache_slot < num_topk:
                            route_slot = route_slot_base + cache_slot
                            expert_id = _ld_shared_i32(
                                route_expert_ids_addr + route_slot * Int32(4)
                            )
                            gs_value = input_global_scale[expert_id].to(cutlass.Float32)
                            if (
                                self.input_scales_are_reciprocal
                                and gs_value != cutlass.Float32(0.0)
                            ):
                                if self.fast_math:
                                    gs_value = rcp_approx_ftz(gs_value)
                                else:
                                    gs_value = cutlass.Float32(1.0) / gs_value
                            route_gs[cache_slot] = gs_value
                            cache_slot += Int32(1)

                        sf_idx = lane_id
                        while sf_idx < sf_blocks_per_row:
                            block_start = sf_idx * Int32(16)
                            loaded_values = load_shared_bf16x16_to_f32x16(
                                q0_input_stage_base_addr
                                + warp_idx * cols * Int32(2)
                                + block_start * Int32(2)
                            )
                            values = cute.make_rmem_tensor((16,), cutlass.Float32)
                            block_max = cutlass.Float32(0.0)
                            for elem_idx in cutlass.range_constexpr(16):
                                value = loaded_values[elem_idx]
                                values[elem_idx] = value
                                block_max = fmax_f32(block_max, fabs_f32(value))

                            # Quantized payload is identical only when all
                            # selected experts use the same input global scale.
                            route_scales_equal = Int32(1)
                            scale_idx = Int32(1)
                            while scale_idx < num_topk:
                                if route_gs[scale_idx] != route_gs[0]:
                                    route_scales_equal = Int32(0)
                                scale_idx += Int32(1)

                            if route_scales_equal > Int32(0):
                                gs_value = route_gs[0]
                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )

                                cache_slot = Int32(0)
                                while cache_slot < num_topk:
                                    route_slot = route_slot_base + cache_slot
                                    phys_row = _ld_shared_i32(
                                        route_phys_rows_addr + route_slot * Int32(4)
                                    )
                                    phys_tile = phys_row // Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    tile_row = phys_row - phys_tile * Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    output_offset = (
                                        phys_row * output_bytes_per_row
                                        + sf_idx * Int32(8)
                                    )
                                    st_global_u64_adaptive_l2(
                                        num_tokens,
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    k_tile_idx = sf_idx // Int32(4)
                                    outer_m_idx = tile_row % Int32(32)
                                    inner_m_idx = (tile_row % Int32(32 * 4)) // Int32(
                                        32
                                    )
                                    inner_k_idx = sf_idx % Int32(4)
                                    scale_offset = (
                                        phys_tile * num_k_tiles * Int32(32 * 4 * 4)
                                        + k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    scale_storage[scale_offset] = scale_byte
                                    cache_slot += Int32(1)
                            else:
                                # Preserve independent quant/store operations;
                                # only the BF16 load and absmax are shared.
                                cache_slot = Int32(0)
                                while cache_slot < num_topk:
                                    route_slot = route_slot_base + cache_slot
                                    phys_row = _ld_shared_i32(
                                        route_phys_rows_addr + route_slot * Int32(4)
                                    )
                                    phys_tile = phys_row // Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    tile_row = phys_row - phys_tile * Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    gs_value = route_gs[cache_slot]

                                    packed64 = Uint64(0)
                                    scale_byte = Uint8(0)
                                    if self.fast_math:
                                        packed64, scale_byte = quantize_block_fp4_fast(
                                            values, block_max, gs_value
                                        )
                                    else:
                                        packed64, scale_byte = quantize_block_fp4(
                                            values, block_max, gs_value
                                        )

                                    output_offset = (
                                        phys_row * output_bytes_per_row
                                        + sf_idx * Int32(8)
                                    )
                                    st_global_u64_adaptive_l2(
                                        num_tokens,
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    k_tile_idx = sf_idx // Int32(4)
                                    outer_m_idx = tile_row % Int32(32)
                                    inner_m_idx = (tile_row % Int32(32 * 4)) // Int32(
                                        32
                                    )
                                    inner_k_idx = sf_idx % Int32(4)
                                    scale_offset = (
                                        phys_tile * num_k_tiles * Int32(32 * 4 * 4)
                                        + k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    scale_storage[scale_offset] = scale_byte
                                    cache_slot += Int32(1)
                            sf_idx += Int32(32)

                q0_bulk_phase = Int32(1) - q0_bulk_phase

        cute.arch.sync_threads()
        # Conservative publish fence before the last-producer CTA flushes any
        # partial tiles. All producer threads in the CTA must have ordered
        # their global writes before lane 0 can publish work.
        _threadfence()
        cute.arch.sync_threads()

        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        total_m_tiles = expert_tile_base[num_experts]
        split_groups = (route_gate_tile_cnt + Int32(1)) // Int32(2)
        extra_per_split = split_groups - Int32(1)
        split_tile_count = Int32(0)
        if extra_per_split > Int32(0):
            if num_tokens > Int32(256):
                if num_tokens <= Int32(4096):
                    target_task_count = Int32(4) * Int32(gdim_z)
                    if num_tokens > Int32(2048):
                        target_task_count = (
                            Int32(125) * Int32(gdim_z) + Int32(31)
                        ) // Int32(32)
                    missing_tasks = target_task_count - total_m_tiles
                    if missing_tasks > Int32(0):
                        split_tile_count = (
                            missing_tasks + extra_per_split - Int32(1)
                        ) // extra_per_split
                        if split_tile_count > total_m_tiles:
                            split_tile_count = total_m_tiles

        if is_cta_leader > Int32(0):
            expert_flush = Int32(bidz)
            while expert_flush < num_experts:
                rows_remaining = row_counts[expert_flush]
                m_tile_offset = Int32(0)
                while rows_remaining > Int32(0):
                    valid_rows = rows_remaining
                    if valid_rows > Int32(self.tile_shape_mnk[0]):
                        valid_rows = Int32(self.tile_shape_mnk[0])
                    if num_tokens <= Int32(256):
                        self.publish_uniform_deferred_tasks(
                            task_expert,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            task_slice_chunk,
                            expert_flush,
                            expert_tile_base[expert_flush] + m_tile_offset,
                            valid_rows,
                        )
                    elif num_tokens <= Int32(4096):
                        self.publish_variable_deferred_tasks(
                            task_expert,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            split_tile_count,
                            expert_flush,
                            expert_tile_base[expert_flush] + m_tile_offset,
                            valid_rows,
                        )
                    else:
                        self.publish_uniform_deferred_tasks(
                            task_expert,
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
            uniform_groups = (
                route_gate_tile_cnt + task_slice_chunk - Int32(1)
            ) // task_slice_chunk
            published_task_count = expert_tile_base[num_experts] * uniform_groups
            if num_tokens > Int32(256):
                if num_tokens <= Int32(4096):
                    published_task_count = (
                        expert_tile_base[num_experts]
                        + split_tile_count * extra_per_split
                    )
            st_global_i32(
                get_ptr_as_int64(task_tail, Int32(0)),
                published_task_count,
            )

        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,  # [num_tokens, K] bf16
        topk_ids: cute.Tensor,  # [num_tokens * topk] int32
        topk_weights: cute.Tensor,  # [num_tokens * topk] float32
        packed_a: cute.Tensor,  # [rows_padded, K, 1] fp4x2 view for compute
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,  # flat uint8 backing packed_a
        scale_storage: cute.Tensor,  # flat uint8 backing sfa_ptr
        barrier_count: cute.Tensor,  # [1] int32 (host-zeroed)
        barrier_epoch: cute.Tensor,  # [1] int32 (host-zeroed)
        pair_head: cute.Tensor,  # [1] int32
        task_head: cute.Tensor,  # [1] int32
        task_tail: cute.Tensor,  # [1] int32
        task_expert: cute.Tensor,  # [max_tasks] int32
        task_valid_rows: cute.Tensor,  # [max_tasks] int32
        b_w13: cute.Tensor,  # [2*I_tp, K, E] (gated) or [I_tp, K, E] (relu2)
        sfb_w13_ptr: cute.Pointer,  # scale factors for w13
        b_down: cute.Tensor,  # [K, I_tp, E]
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,  # expert row histogram [E]
        expert_write_rows: cute.Tensor,  # route/pack write cursors [E]
        expert_tile_base: cute.Tensor,  # compact physical-tile prefix [E + 1]
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
        # Dynamic never materializes the intermediate C tensor. Preserve the
        # original row-major epilogue layout without carrying a dead memref.
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        hidden_size = a_input.shape[1]
        if cutlass.const_expr(
            hidden_size > self.tile_shape_mnk[0] * self.tile_shape_mnk[1]
        ):
            raise ValueError(
                "the gated dynamic kernel requires one BF16 input row to fit "
                "in its 16384-element Q0 staging buffer"
            )
        self._setup_attributes(hidden_size=hidden_size)

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            packed_a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # SF tensor for w13 (gated: gate+up concatenated; relu2: single W1)
        sfb_w13_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_w13.shape, self.sf_vec_size
        )
        sfb_w13_tensor = cute.make_tensor(sfb_w13_ptr, sfb_w13_layout)

        # TMA descriptors
        tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
            packed_a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_sfa, gSFA = self._dense_cls._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
            internal_type=cutlass.Int16,
        )
        # FC1 B uses a true N64 descriptor.  Each logical N128 slice is two
        # consecutive native B tiles; Up precedes Gate in global w13 storage.
        tma_b_w13, gB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            b_w13,
            self.fc1_b_smem_layout_staged,
            (self.fc1_tile_shape_mnk[1], self.fc1_tile_shape_mnk[2]),
            1,
        )
        # SFB is different from B: the SM120 helper physically packs scale
        # factors in N128 blocks.  Both N64 halves replay the same physical
        # block and select half 0/1 from its shared-memory view.
        tma_sfb_w13, gSFB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_w13_tensor,
            self.fc1_sfb_smem_layout_staged,
            self.fc1_sfb_tile_shape_nk,
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
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
            internal_type=cutlass.Int16,
        )

        # W13 concatenates equally-sized Gate and Up branches along N.
        gate_tile_cnt_static = b_w13.shape[0] // self.tile_shape_mnk[1] // 2
        if cutlass.const_expr(gate_tile_cnt_static > _TASK_SLICE_CHUNK):
            raise ValueError(
                "the gated dynamic kernel retains at most four intermediate "
                "slices per task"
            )
        gate_tile_cnt = Int32(gate_tile_cnt_static)
        launch_params = DynamicLaunchParams(row_counts, gate_tile_cnt)
        grid = (*self.cluster_shape_mn, max_active_clusters)
        self.kernel(
            a_input,
            topk_ids,
            topk_weights,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            pair_head,
            task_head,
            task_tail,
            task_expert,
            task_valid_rows,
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
            self.fc1_tiled_mma,
            self.mma_atom,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.phase2_b_smem_layout_staged,
            self.fc1_b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.phase2_sfb_smem_layout_staged,
            self.fc1_sfb_smem_layout_staged,
            self.fc1_sfb_smem_layout_storage,
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
            cooperative=True,
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
        pair_head: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
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
        fc1_tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        mma_atom_tail: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        phase2_b_smem_staged: cute.ComposedLayout,
        fc1_b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
        phase2_sfb_smem_staged: cute.Layout,
        fc1_sfb_smem_staged: cute.Layout,
        fc1_sfb_smem_layout_storage: cute.Layout,
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
        _bidx, _, bidz = cute.arch.block_idx()
        _, _, gdim_z = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_id = Int32(tidx) & Int32(31)
        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_sfa)
            cpasync.prefetch_descriptor(tma_b_w13)
            cpasync.prefetch_descriptor(tma_sfb_w13)
            cpasync.prefetch_descriptor(tma_b_down)
            cpasync.prefetch_descriptor(tma_sfb_down)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

        a_smem_one = cute.slice_(a_smem_staged, (None, None, 0))
        b_smem_one = cute.slice_(b_smem_staged, (None, None, 0))
        fc1_b_smem_one = cute.slice_(fc1_b_smem_staged, (None, None, 0))
        sfa_smem_one = cute.slice_(sfa_smem_staged, (None, None, 0))
        sfb_smem_one = cute.slice_(sfb_smem_staged, (None, None, 0))
        fc1_sfb_smem_one = cute.slice_(fc1_sfb_smem_staged, (None, None, 0))
        sequential_branch_compact = cutlass.const_expr(
            getattr(self, "sequential_branch_compact", False)
        )
        fc1_storage_alias = cutlass.const_expr(
            getattr(self, "fc1_storage_alias", sequential_branch_compact)
        )
        fc1_tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_one)
            + cute.size_in_bytes(self.b_dtype, fc1_b_smem_one)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_one)
            + cute.size_in_bytes(self.sf_dtype, fc1_sfb_smem_one)
        )
        fc1_branch_tma_copy_bytes = fc1_tma_copy_bytes
        if cutlass.const_expr(not sequential_branch_compact):
            fc1_tma_copy_bytes += cute.size_in_bytes(
                self.b_dtype, fc1_b_smem_one
            ) + cute.size_in_bytes(self.sf_dtype, fc1_sfb_smem_one)
        phase2_tma_copy_bytes = cute.size_in_bytes(
            self.b_dtype, b_smem_one
        ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class StorageGated:
            # ctrl layout (16 x Int32, accessed via raw shared memory PTX):
            #   [0] has_task     [4] done          [8]  expert_idx
            #   [12] m_tile_idx  [16] slice_begin   [20] slice_count
            #   [24] valid_rows  [28] batch_base
            #   [32] next_has    [36] next_done     [40] next_expert
            #   [44] next_mtile  [48] next_begin    [52] next_count
            #   [56] next_rows   [60] reserved
            ctrl: cute.struct.MemRange[cutlass.Int32, 16]
            # Startup-only route cache aliases the unused sC backing.
            route_phys_rows: cute.struct.MemRange[cutlass.Int32, 0]
            route_expert_ids: cute.struct.MemRange[cutlass.Int32, 0]
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            up_pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            phase2_pipeline_array: cute.struct.MemRange[
                cutlass.Int64, self.phase2_stage * 2
            ]
            q0_bulk_barrier: cute.struct.MemRange[cutlass.Int64, 1]
            scatter_tok_cache: cute.struct.MemRange[
                cutlass.Int32, self.tile_shape_mnk[0] * 2
            ]
            scatter_weight_cache: cute.struct.MemRange[cutlass.Float32, 0]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            # During FC1, the first B-sized part of sC is the contiguous
            # third N128 B stage.  FC1 releases it before activation writes.
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            # Gate and Up occupy disjoint N64 halves of the N128 sB stage.
            sB_up: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, 0],
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
            sSFB_phase2_extra: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_one)],
                self.buffer_align_bytes,
            ]
            # SM120 packs each logical N64 SFB half in a physical-N128 block;
            # the Up branch therefore needs a distinct physical backing.
            sSFB_up: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    (
                        0
                        if fc1_storage_alias
                        else cute.cosize(fc1_sfb_smem_layout_storage)
                    ),
                ],
                self.buffer_align_bytes,
            ]

        storage = smem.allocate(StorageGated)

        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        ml_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=fc1_tma_copy_bytes,
            barrier_storage=storage.pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        up_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=fc1_branch_tma_copy_bytes,
            barrier_storage=storage.up_pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        phase2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.phase2_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=phase2_tma_copy_bytes,
            barrier_storage=storage.phase2_pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )

        cute.arch.sync_threads()

        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        # FC2 retains its two physical B stages and addresses dead sA[0]
        # separately as phase2 stage2.  Conversely FC1 sees one contiguous
        # three-stage N128 backing spanning sB[0:2] and the beginning of sC.
        phase2_b_extra_ptr = cute.recast_ptr(
            storage.sA.data_ptr(),
            b_smem_one.inner,
            dtype=self.b_dtype,
        )
        sB_phase2_extra = cute.make_tensor(phase2_b_extra_ptr, b_smem_one.outer)
        sB_fc1_all = storage.sB.get_tensor(
            phase2_b_smem_staged.outer,
            swizzle=phase2_b_smem_staged.inner,
        )
        # While FC1 is live, split the N128 FC2 backing into two N64 views.
        sB_fc1 = cute.local_tile(
            sB_fc1_all,
            cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
            (0, 0, None),
        )
        sB_up_fc1 = cute.local_tile(
            sB_fc1_all,
            cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
            (1, 0, None),
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_staged)
        # sSFB_phase2_extra is the immediately following aligned field, so
        # expose the existing two-plus-one backing as one staged tensor.
        sSFB_phase2 = storage.sSFB.get_tensor(phase2_sfb_smem_staged)
        # Gate gets a contiguous third SFB stage from the existing phase2
        # extra.  Up keeps its two allocated stages and uses a disjoint
        # one-stage alias in sC immediately after the FC1 B-stage2 bytes.
        sSFB_fc1 = storage.sSFB.get_tensor(fc1_sfb_smem_staged)
        sSFB_up_fc1 = (
            sSFB_fc1
            if fc1_storage_alias
            else storage.sSFB_up.get_tensor(fc1_sfb_smem_layout_storage)
        )
        fc1_sfb_smem_one = cute.slice_(fc1_sfb_smem_staged, (None, None, 0))
        fc1_b_stage_bytes = cute.size_in_bytes(self.b_dtype, b_smem_one)
        sSFB_up_fc1_extra_ptr = cute.recast_ptr(
            storage.sC.data_ptr() + fc1_b_stage_bytes // 2,
            dtype=self.sf_dtype,
        )
        sSFB_up_fc1_extra = cute.make_tensor(sSFB_up_fc1_extra_ptr, fc1_sfb_smem_one)
        sC = storage.sC.get_tensor(
            epi_smem_staged.outer,
            swizzle=epi_smem_staged.inner,
        )
        sfa_base_addr = shared_ptr_to_u32(storage.sSFA.data_ptr())
        sfa_stage_elements = Int32(cute.cosize(sfa_smem_one))
        ctrl_base_addr = shared_ptr_to_u32(storage.ctrl.data_ptr())
        # Q0 uses raw-linear sC as an eight-token BF16 staging buffer.
        # Move both 288-entry route caches to startup-idle sA.
        route_phys_rows_addr = shared_ptr_to_u32(storage.sA.data_ptr())
        route_expert_ids_addr = route_phys_rows_addr + Int32(
            (self.num_mma_warps + 1) * 32 * 4
        )
        q0_input_stage_base_addr = shared_ptr_to_u32(storage.sC.data_ptr())
        q0_bulk_barrier_addr = shared_ptr_to_u32(storage.q0_bulk_barrier.data_ptr())
        scatter_tok_base_addr = shared_ptr_to_u32(storage.scatter_tok_cache.data_ptr())
        scatter_weight_base_addr = scatter_tok_base_addr + Int32(4)

        self.initialize_route_q0_and_publish(
            (tidx, bidz, gdim_z, warp_idx, is_cta_leader),
            (a_input, topk_ids, topk_weights, input_global_scale),
            (
                packed_a_storage,
                scale_storage,
                scatter_output,
                token_map,
                token_weights,
            ),
            (expert_write_rows, expert_tile_base, pair_head),
            (
                task_head,
                task_tail,
                task_expert,
                task_valid_rows,
            ),
            (barrier_count, barrier_epoch),
            (
                ctrl_base_addr,
                route_phys_rows_addr,
                route_expert_ids_addr,
                q0_input_stage_base_addr,
                q0_bulk_barrier_addr,
            ),
            launch_params,
        )

        # Deferred publication is complete after the resident-grid barrier
        # inside initialize_route_q0_and_publish.  Cache the immutable tail in
        # the otherwise streaming-only ctrl[28] slot; the claim loop uses a
        # side-effecting shared load to preserve phase ordering.
        if is_cta_leader > Int32(0):
            stable_task_tail = _ld_global_acquire_i32(
                get_ptr_as_int64(task_tail, Int32(0))
            )
            _st_shared_i32(ctrl_base_addr + Int32(28), stable_task_tail)
            _st_shared_i32(ctrl_base_addr + Int32(32), Int32(0))
            _st_shared_i32(ctrl_base_addr + Int32(36), Int32(0))

        gA = cute.local_tile(
            mA, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        # B is tiled at the native N64 compute granularity.  SFB is tiled at
        # the physical N128 scale-factor block granularity and replayed for
        # the two B halves.
        gB_w13_tiled = cute.local_tile(
            mB_w13,
            cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFA = cute.local_tile(
            mSFA, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        gSFB_w13_tiled = cute.local_tile(
            mSFB_w13,
            self.fc1_sfb_tile_shape_nk,
            (None, None, None),
        )
        thr_mma = tiled_mma.get_slice(tidx)
        fc1_thr_mma = fc1_tiled_mma.get_slice(tidx)

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

        # w13 FC1 partitions: N64 B payload plus physical-N128 SFB payload.
        tBsB_w13, tBgB_w13 = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_fc1, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsB_w13_up, _tBgB_w13_up = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_up_fc1, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsSFB_w13, tBgSFB_w13 = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_fc1, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFB_w13_up, _tBgSFB_w13_up = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_up_fc1, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFB_w13_up_extra, _tBgSFB_w13_up_extra = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_up_fc1_extra, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFB_w13 = cute.filter_zeros(tBsSFB_w13)
        tBgSFB_w13 = cute.filter_zeros(tBgSFB_w13)
        tBsSFB_w13_up = cute.filter_zeros(tBsSFB_w13_up)
        tBsSFB_w13_up_extra = cute.filter_zeros(tBsSFB_w13_up_extra)

        # B_down TMA partitions
        gB_down = cute.local_tile(
            mB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFB_down = cute.local_tile(
            mSFB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        tBsB_down, tBgB_down = cpasync.tma_partition(
            tma_b_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_down, 0, 2),
        )
        tBsB_down_extra, _tBgB_down_extra = cpasync.tma_partition(
            tma_b_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_phase2_extra, 0, 2),
            cute.group_modes(gB_down, 0, 2),
        )
        tBsSFB_down, tBgSFB_down = cpasync.tma_partition(
            tma_sfb_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_phase2, 0, 2),
            cute.group_modes(gSFB_down, 0, 2),
        )
        tBsSFB_down = cute.filter_zeros(tBsSFB_down)
        tBgSFB_down = cute.filter_zeros(tBgSFB_down)

        # FC2 fragment partitions retain the original N128 contract.
        tCsA = thr_mma.partition_A(sA)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrSFA = self._dense_cls._partition_fragment_SFA(
            self,  # type: ignore[arg-type]
            sSFA[None, None, 0],
            thr_mma,
            tidx,
        )

        # FC1 has an independent N64 MMA/permutation and aliases the same A/SFA
        # storage.  Its SFB fragment is created per half below because a
        # physical N128 SFB block contains two logical N64 scale tiles.
        tCsA_fc1 = fc1_thr_mma.partition_A(sA)
        tCrA_fc1 = fc1_tiled_mma.make_fragment_A(tCsA_fc1[None, None, None, 0])
        tCrSFA_fc1 = self._dense_cls._partition_fragment_SFA(
            self,  # type: ignore[arg-type]
            sSFA[None, None, 0],
            fc1_thr_mma,
            tidx,
        )
        tCsB_fc1 = fc1_thr_mma.partition_B(sB_fc1)
        tCrB_fc1 = fc1_tiled_mma.make_fragment_B(tCsB_fc1[None, None, None, 0])
        tCsB_up_fc1 = fc1_thr_mma.partition_B(sB_up_fc1)
        tCrB_up_fc1 = fc1_tiled_mma.make_fragment_B(tCsB_up_fc1[None, None, None, 0])
        tCsB = thr_mma.partition_B(sB)
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFB = self._dense_cls._partition_fragment_SFB(
            self,  # type: ignore[arg-type]
            sSFB[None, None, 0],
            thr_mma,
            tidx,
        )

        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        sub_shape = tCsC_for_shape.shape[:3]
        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
        k_tile_cnt = cute.size(gA, mode=[3])
        fc1_k_tile_cnt = k_tile_cnt
        # gB is native-N64 while tasks and FC2 remain logical-N128.
        native_fc1_tile_cnt = cute.size(gB_w13_tiled, mode=[2]) // Int32(2)
        gate_tile_cnt = native_fc1_tile_cnt // Int32(2)
        output_tile_cnt = cute.size(gB_down, mode=[2])

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        up_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        up_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        phase2_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.phase2_stage
        )
        phase2_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.phase2_stage
        )

        num_k_blocks = cute.size(tCrA, mode=[2])
        fc1_num_k_blocks = cute.size(tCrA_fc1, mode=[2])

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
        atom_ld_SF = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
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
        smem_copy_A_fc1 = cute.make_tiled_copy_A(atom_ld_A, fc1_tiled_mma)
        smem_copy_B_fc1 = cute.make_tiled_copy_B(atom_ld_B, fc1_tiled_mma)
        smem_copy_SFA_fc1 = cute.make_tiled_copy(
            atom_ld_SF,
            self._dense_cls._get_layoutSFA_TV(self, fc1_tiled_mma),  # type: ignore[arg-type]
            (
                cute.size(fc1_tiled_mma.permutation_mnk[0]),
                cute.size(fc1_tiled_mma.permutation_mnk[2]),
            ),
        )
        smem_copy_SFB_fc1 = cute.make_tiled_copy(
            atom_ld_SF,
            self._dense_cls._get_layoutSFB_TV(self, fc1_tiled_mma),  # type: ignore[arg-type]
            (
                cute.size(fc1_tiled_mma.permutation_mnk[1]),
                cute.size(fc1_tiled_mma.permutation_mnk[2]),
            ),
        )

        thr_ld_A = smem_copy_A.get_slice(tidx)
        thr_ld_B = smem_copy_B.get_slice(tidx)
        csA = thr_ld_A.partition_S(sA)
        crA = thr_ld_A.retile(tCrA)
        csB = thr_ld_B.partition_S(sB)
        csB_phase2_extra = thr_ld_B.partition_S(sB_phase2_extra)
        crB = thr_ld_B.retile(tCrB)

        thr_ld_SFA = smem_copy_SFA.get_slice(tidx)
        thr_ld_SFB = smem_copy_SFB.get_slice(tidx)
        csSFA = thr_ld_SFA.partition_S(sSFA)
        crSFA = thr_ld_SFA.retile(tCrSFA)
        csSFB = thr_ld_SFB.partition_S(sSFB_phase2)
        crSFB = thr_ld_SFB.retile(tCrSFB)

        thr_ld_A_fc1 = smem_copy_A_fc1.get_slice(tidx)
        thr_ld_B_fc1 = smem_copy_B_fc1.get_slice(tidx)
        csA_fc1 = thr_ld_A_fc1.partition_S(sA)
        crA_fc1 = thr_ld_A_fc1.retile(tCrA_fc1)
        csB_fc1 = thr_ld_B_fc1.partition_S(sB_fc1)
        crB_fc1 = thr_ld_B_fc1.retile(tCrB_fc1)
        csB_up_fc1 = thr_ld_B_fc1.partition_S(sB_up_fc1)
        crB_up_fc1 = thr_ld_B_fc1.retile(tCrB_up_fc1)

        thr_ld_SFA_fc1 = smem_copy_SFA_fc1.get_slice(tidx)
        thr_ld_SFB_fc1 = smem_copy_SFB_fc1.get_slice(tidx)
        csSFA_fc1 = thr_ld_SFA_fc1.partition_S(sSFA)
        crSFA_fc1 = thr_ld_SFA_fc1.retile(tCrSFA_fc1)

        # ===================================================================
        # Per-warp setup for the consumer steady state
        # ===================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ===================================================================
        # Consumer steady state: pop one ready task per CTA, then let
        # the MMA warps and DMA warp cooperate on that task.
        # ===================================================================
        consumer_live = Int32(1)
        while consumer_live > Int32(0):
            has_task, is_done = self.claim_and_cache_task(
                tidx,
                warp_idx,
                is_cta_leader,
                ctrl_base_addr,
                task_head,
                task_expert,
                task_valid_rows,
                token_map,
                token_weights,
                scatter_tok_base_addr,
                scatter_weight_base_addr,
            )
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
                # atom_layout=(4,2,1): two M16 fragments per warp, separated
                # by 64 rows. Full tasks use the original branch-free method.
                warp_m_coord = Int32(warp_idx) & Int32(3)

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
                fc1_tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                    copy_atom_C, fc1_tiled_mma
                )
                fc1_tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, fc1_tiled_copy_C_Atom
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                fc1_thr_copy_r2s = fc1_tiled_copy_r2s.get_slice(tidx)
                fc1_tRS_sD = fc1_thr_copy_r2s.partition_D(sC)
                down_alpha_value = down_alpha[task_expert_idx].to(cutlass.Float32)
                epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]

                fc1_m_tiles = cute.size(tCrA_fc1, mode=[1])
                fc1_n_tiles = cute.size(tCrB_fc1, mode=[1])
                deferred_a_words = cute.make_rmem_tensor((8,), cutlass.Uint32)
                deferred_a_words.fill(0)
                deferred_sfa_words = cute.make_rmem_tensor((2,), cutlass.Uint32)
                deferred_sfa_words.fill(0)
                slice_idx = Int32(0)
                while slice_idx < task_slice_count_val:
                    if valid_rows == Int32(self.tile_shape_mnk[0]):
                        cons_state, up_cons_state = self.fc1_gate_up_swiglu_to_sC(
                            tidx,
                            ml_pipeline,
                            cons_state,
                            up_pipeline,
                            up_cons_state,
                            fc1_tiled_mma,
                            mma_atom,
                            sSFB_fc1,
                            sSFB_up_fc1,
                            sSFB_up_fc1_extra,
                            fc1_thr_mma,
                            thr_ld_SFB_fc1,
                            csA_fc1,
                            csB_fc1,
                            csB_up_fc1,
                            csSFA_fc1,
                            crA_fc1,
                            crB_fc1,
                            crB_up_fc1,
                            crSFA_fc1,
                            tCrA_fc1,
                            tCrB_fc1,
                            tCrB_up_fc1,
                            tCrSFA_fc1,
                            smem_copy_A_fc1,
                            smem_copy_B_fc1,
                            smem_copy_SFA_fc1,
                            smem_copy_SFB_fc1,
                            fc1_tiled_copy_r2s,
                            fc1_tRS_sD,
                            fc1_k_tile_cnt,
                            fc1_num_k_blocks,
                            fc1_m_tiles,
                            fc1_n_tiles,
                            alpha_value,
                            valid_rows,
                            task_expert_idx,
                            global_scale,
                            sC,
                            sA,
                            sfa_base_addr,
                            epi_rest_m,
                        )
                    else:
                        cons_state, up_cons_state = self.fc1_gate_up_swiglu_to_sC_tail(
                            tidx,
                            ml_pipeline,
                            cons_state,
                            up_pipeline,
                            up_cons_state,
                            fc1_tiled_mma,
                            mma_atom_tail,
                            sSFB_fc1,
                            sSFB_up_fc1,
                            sSFB_up_fc1_extra,
                            fc1_thr_mma,
                            thr_ld_SFB_fc1,
                            csA_fc1,
                            csB_fc1,
                            csB_up_fc1,
                            csSFA_fc1,
                            crA_fc1,
                            crB_fc1,
                            crB_up_fc1,
                            crSFA_fc1,
                            tCrA_fc1,
                            tCrB_fc1,
                            tCrB_up_fc1,
                            tCrSFA_fc1,
                            smem_copy_A_fc1,
                            smem_copy_B_fc1,
                            smem_copy_SFA_fc1,
                            smem_copy_SFB_fc1,
                            fc1_tiled_copy_r2s,
                            fc1_tRS_sD,
                            fc1_k_tile_cnt,
                            fc1_num_k_blocks,
                            fc1_m_tiles,
                            fc1_n_tiles,
                            alpha_value,
                            valid_rows,
                            warp_m_coord,
                            task_expert_idx,
                            global_scale,
                            sC,
                            sA,
                            sfa_base_addr,
                            epi_rest_m,
                        )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    q1_a_stage_idx = Int32(3)
                    defer_a = Int32(0)
                    if slice_idx == Int32(1):
                        q1_a_stage_idx = Int32(4)
                    elif slice_idx == Int32(2):
                        q1_a_stage_idx = Int32(0)
                        defer_a = Int32(1)
                    elif slice_idx == Int32(3):
                        q1_a_stage_idx = Int32(1)

                    q1_sfa_stage_idx = Int32(3)
                    defer_sfa = Int32(0)
                    deferred_sfa_slot = Int32(0)
                    if slice_idx == Int32(1):
                        q1_sfa_stage_idx = Int32(0)
                        defer_sfa = Int32(1)
                    elif slice_idx == Int32(2):
                        q1_sfa_stage_idx = Int32(0)
                        defer_sfa = Int32(1)
                        deferred_sfa_slot = Int32(1)
                    elif slice_idx == Int32(3):
                        q1_sfa_stage_idx = Int32(0)
                    self.quantize_q1_sC_to_sA_sSFA(
                        tidx,
                        valid_rows,
                        task_expert_idx,
                        global_scale,
                        sC,
                        sA,
                        fc1_tRS_sD,
                        sfa_base_addr,
                        sfa_stage_elements,
                        q1_a_stage_idx,
                        defer_a,
                        deferred_a_words,
                        q1_sfa_stage_idx,
                        defer_sfa,
                        deferred_sfa_words,
                        deferred_sfa_slot,
                        epi_rest_m,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    self.pass_gate_barrier.arrive_unaligned()

                    # Q1 has finished reading sC, so the following FC1 slice
                    # may safely reuse it as the third B/SFB stage.
                    slice_idx += Int32(1)

                # The final FC1 pass has released A/SFA stages0:2.  Materialize
                # the exact packed Q1 bytes deferred to registers.
                self.flush_deferred_q1_a(
                    tidx,
                    valid_rows,
                    deferred_a_words,
                    sA,
                    Int32(2),
                )
                self.flush_deferred_q1_sfa(
                    tidx,
                    valid_rows,
                    deferred_sfa_words[0],
                    sfa_base_addr,
                    sfa_stage_elements,
                    Int32(1),
                )
                self.flush_deferred_q1_sfa(
                    tidx,
                    valid_rows,
                    deferred_sfa_words[1],
                    sfa_base_addr,
                    sfa_stage_elements,
                    Int32(2),
                )
                self.epilog_sync_barrier.arrive_and_wait()

                phase2_cons_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    physical_output_tile_idx = (
                        Int32(output_tile_idx) + task_expert_idx
                    ) % Int32(output_tile_cnt)
                    down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
                    down_acc.fill(0.0)
                    slice_idx = Int32(0)
                    while slice_idx < task_slice_count_val:
                        q1_a_stage_idx = Int32(3)
                        if slice_idx == Int32(1):
                            q1_a_stage_idx = Int32(4)
                        elif slice_idx == Int32(2):
                            q1_a_stage_idx = Int32(2)
                        elif slice_idx == Int32(3):
                            q1_a_stage_idx = Int32(1)

                        q1_sfa_stage_idx = Int32(3)
                        if slice_idx == Int32(1):
                            q1_sfa_stage_idx = Int32(1)
                        elif slice_idx == Int32(2):
                            q1_sfa_stage_idx = Int32(2)
                        elif slice_idx == Int32(3):
                            q1_sfa_stage_idx = Int32(0)
                        self.load_fc2_a_fragments(
                            num_k_blocks,
                            q1_a_stage_idx,
                            q1_sfa_stage_idx,
                            (csA, csSFA),
                            (crA, crSFA),
                            (smem_copy_A, smem_copy_SFA),
                        )
                        if valid_rows == Int32(self.tile_shape_mnk[0]):
                            phase2_cons_state = self.fc2_accumulate_slice(
                                num_k_blocks,
                                mma_atom,
                                down_acc,
                                (phase2_pipeline, phase2_cons_state),
                                (csB, csB_phase2_extra, csSFB),
                                (tCrA, tCrB, tCrSFA, tCrSFB, crB, crSFB),
                                (smem_copy_B, smem_copy_SFB),
                            )
                        else:
                            phase2_cons_state = self.fc2_accumulate_slice_tail(
                                num_k_blocks,
                                mma_atom_tail,
                                down_acc,
                                valid_rows,
                                warp_m_coord,
                                (phase2_pipeline, phase2_cons_state),
                                (csB, csB_phase2_extra, csSFB),
                                (tCrA, tCrB, tCrSFA, tCrSFB, crB, crSFB),
                                (smem_copy_B, smem_copy_SFB),
                            )
                        slice_idx += Int32(1)

                    self.fc2_epilogue_to_sC(
                        acc_shape,
                        down_alpha_value,
                        down_acc,
                        sC,
                        tiled_copy_r2s,
                        thr_copy_r2s,
                        tRS_sD,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    if (warp_idx & Int32(2)) == Int32(0):
                        self.fc2_group_a_barrier.arrive_and_wait()
                    else:
                        self.fc2_group_b_barrier.arrive_and_wait()
                    self.scatter_sC_to_gmem(
                        tidx,
                        physical_output_tile_idx,
                        valid_rows,
                        sC,
                        tRS_sD,
                        scatter_output,
                        scatter_tok_base_addr,
                        scatter_weight_base_addr,
                        down_alpha_value,
                    )
                    if (warp_idx & Int32(2)) == Int32(0):
                        self.fc2_group_a_barrier.arrive_and_wait()
                    else:
                        self.fc2_group_b_barrier.arrive_and_wait()

                # All output tiles have consumed every retained Q1 slice.
                self.pass_final_barrier.arrive_and_wait()

            elif warp_idx == self.tma_load_warp_id:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
                task_m_tile_idx = _ld_shared_i32(ctrl_base_addr + Int32(12))
                task_slice_begin_idx = _ld_shared_i32(ctrl_base_addr + Int32(16))
                task_slice_count_val = _ld_shared_i32(ctrl_base_addr + Int32(20))

                tAgA_mk = tAgA[(None, task_m_tile_idx, None, Int32(0))]
                tAgSFA_mk = tAgSFA[(None, task_m_tile_idx, None, Int32(0))]
                slice_idx = Int32(0)
                while slice_idx < task_slice_count_val:
                    intermediate_slice = task_slice_begin_idx + slice_idx
                    wait_for_prior_slice = Int32(0)
                    if slice_idx > Int32(0):
                        wait_for_prior_slice = Int32(1)
                    prod_state, up_prod_state = self.load_fc1_tma_slice(
                        intermediate_slice,
                        wait_for_prior_slice,
                        task_expert_idx,
                        gate_tile_cnt,
                        fc1_k_tile_cnt,
                        prod_state,
                        ml_pipeline,
                        up_prod_state,
                        up_pipeline,
                        (tma_a, tma_b_w13, tma_sfa, tma_sfb_w13),
                        (tAgA_mk, tAgSFA_mk, tBgB_w13, tBgSFB_w13),
                        (
                            tAsA,
                            tAsSFA,
                            tBsB_w13,
                            tBsB_w13_up,
                            tBsSFB_w13,
                            tBsSFB_w13_up,
                            tBsSFB_w13_up_extra,
                        ),
                    )
                    slice_idx += Int32(1)

                # The final FC1 MMA release is narrower than pass_gate:
                # clone the state so we can prove every FC1 A/B/SF stage is
                # empty without advancing the live producer state.  FC2
                # weights do not alias sC, so they may prefetch while final
                # activation/Q1 is still using sC.
                fc1_drain_state = prod_state.clone()
                ml_pipeline.producer_tail(fc1_drain_state)

                phase2_prod_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    physical_output_tile_idx = (
                        Int32(output_tile_idx) + task_expert_idx
                    ) % Int32(output_tile_cnt)
                    slice_idx = Int32(0)
                    while slice_idx < task_slice_count_val:
                        intermediate_slice = task_slice_begin_idx + slice_idx
                        phase2_prod_state = self.load_fc2_tma_tile(
                            intermediate_slice,
                            physical_output_tile_idx,
                            task_expert_idx,
                            phase2_prod_state,
                            phase2_pipeline,
                            (tma_b_down, tma_sfb_down),
                            (tBgB_down, tBgSFB_down),
                            (
                                tBsB_down,
                                tBsB_down_extra,
                                tBsSFB_down,
                            ),
                        )
                        slice_idx += Int32(1)

                # Warp8 has finished issuing current-task FC2 weights while
                # math warps still own FC2 MMA/epilogue/scatter. Reserve and
                # cache one next descriptor in disjoint shared control state.
                self.prefetch_next_task_descriptor(
                    lane_id,
                    ctrl_base_addr,
                    task_head,
                    task_expert,
                    task_valid_rows,
                )

                # Consume the final slice's activation/Q1 arrival before the
                # task handoff. Earlier arrivals were consumed lazily at the
                # first next-slice Stage2 overwrite.
                self.pass_gate_barrier.wait_unaligned()

                # Keep A4/Q1 and sC alive until all output tiles complete.
                self.pass_final_barrier.wait_unaligned()

        if warp_idx == self.tma_load_warp_id:
            ml_pipeline.producer_tail(prod_state)
            if cutlass.const_expr(getattr(self, "sequential_branch_compact", False)):
                up_pipeline.producer_tail(up_prod_state)
            phase2_pipeline.producer_tail(phase2_prod_state)
        return


__all__ = ["MoEGatedDynamicKernel"]
