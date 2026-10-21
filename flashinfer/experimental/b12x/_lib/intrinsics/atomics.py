# Copyright (c) 2025 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic device memory operations."""

from cutlass import Float32
from cutlass import Int32
from cutlass import Int64
from cutlass.cutlass_dsl import T
from typing import Tuple
from cutlass import Uint32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


# =============================================================================
# PTX Intrinsics - Global Atomics
# =============================================================================


@dsl_user_op
def atomic_add_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None) -> Int32:
    """Global memory int32 atomic add. Returns old value."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def red_add_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """No-return global int32 add reduction (relaxed device scope)."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_global_u64(addr: Int64, val: Int64, *, loc=None, ip=None) -> Int64:
    """Global 64-bit atomic add (relaxed device scope). Returns the old value."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int64(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;",
            "=l,l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_add_global_release_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """No-return global int32 add with a GPU-scope release fence."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "fence.acq_rel.gpu;\nred.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_global_f32(addr: Int64, val: Float32, *, loc=None, ip=None):
    """No-return global fp32 add reduction (relaxed device scope)."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_global_v4_f32(
    addr: Int64,
    val0: Float32,
    val1: Float32,
    val2: Float32,
    val3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Reduce-add four FP32 elements at a 16-byte-aligned address."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(val0).ir_value(loc=loc, ip=ip),
            Float32(val1).ir_value(loc=loc, ip=ip),
            Float32(val2).ir_value(loc=loc, ip=ip),
            Float32(val3).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.v4.f32.add [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cvt_bf16x2_to_f16x2(packed: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert a u32 holding two bf16 (lo, hi) into an f16x2 u32 (lo, hi)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 blo, bhi, hlo, hhi;
                .reg .f32 flo, fhi;
                mov.b32 {blo, bhi}, $1;
                cvt.f32.bf16 flo, blo;
                cvt.f32.bf16 fhi, bhi;
                cvt.rn.f16.f32 hlo, flo;
                cvt.rn.f16.f32 hhi, fhi;
                mov.b32 $0, {hlo, hhi};
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_add_global_bf16x2(addr: Int64, packed: Uint32, *, loc=None, ip=None):
    """No-return global atomic add of a packed bf16x2 value (2 contiguous bf16).

    Used by the W4A16 TC-decode FC2 epilogue to fold the per-route partial
    outputs into the per-token output without a separate top-k-sum launch.
    The address must be 4-byte aligned and cover two consecutive bf16 lanes.
    """
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(packed).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.noftz.bf16x2 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_max_global_f32_nonnegative(addr: Int64, val: Float32, *, loc=None, ip=None):
    """No-return global max reduction for a non-negative fp32 value.

    Non-negative IEEE-754 float ordering matches unsigned integer bit ordering,
    so this emits a no-return u32 max reduction on the value's bit pattern.
    Callers must initialize the destination with a non-negative fp32 value.
    """
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 vi; mov.b32 vi, $1; red.relaxed.gpu.global.max.u32 [$0], vi; }",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_cas_global_i32(
    addr: Int64, compare: Int32, value: Int32, *, loc=None, ip=None
) -> Int32:
    """Global memory int32 atomic compare-and-swap. Returns old value."""
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
def atomic_add_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None) -> Int32:
    """Shared-memory int32 atomic add (CTA-scope). Returns old value.

    Uses ``atom.shared.add.s32`` with a 32-bit shared-memory address
    (the native address width for smem on NVIDIA GPUs).
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
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


@dsl_user_op
def atomic_max_shared_f32(smem_addr: Int32, val: Float32, *, loc=None, ip=None):
    """No-return shared atomic max for a NON-NEGATIVE fp32 ``val``.

    Relies on the IEEE-754 ordering of non-negative floats matching the signed
    int ordering of their bit patterns, so the smem max can be done with a
    single s32 reduction. All callers consume the reduced shared slot after a
    barrier; none needs the old or resulting value in the issuing lane.
    """
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{.reg .s32 vi; mov.b32 vi, $1; red.shared.max.s32 [$0], vi;}",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_max_shared_f32_offset(
    smem_addr: Int32,
    byte_offset: int,
    val: Float32,
    *,
    loc=None,
    ip=None,
):
    """No-return nonnegative FP32 shared max at a constant displacement."""
    offset = int(byte_offset)
    address = f"[$0+{offset}]" if offset else "[$0]"
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        f"{{.reg .s32 vi; mov.b32 vi, $1; red.shared.max.s32 {address}, vi;}}",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None):
    """No-return shared-memory int32 add.

    Use this when the old value is not needed. NVCC lowers equivalent C++
    histogram increments to cheaper no-return shared atomics.
    """
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_i32(addr: Int32, *, loc=None, ip=None) -> Int32:
    """Volatile-style load int32 from shared memory at a 32-bit byte address."""
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
def ld_shared_i32_relaxed(addr: Int32, *, loc=None, ip=None) -> Int32:
    """Load int32 from shared memory at a 32-bit byte address."""
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
def ld_shared_u32(addr: Int32, *, loc=None, ip=None) -> Uint32:
    """Load uint32 from shared memory at a 32-bit byte address."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None):
    """Store int32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_u32(addr: Int32, val: Uint32, *, loc=None, ip=None):
    """Store uint32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Uint32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    """Load float32 from shared memory at a 32-bit byte address."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_f32_offset(
    addr: Int32,
    byte_offset: int,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """Load shared FP32 using a compile-time displacement."""
    offset = int(byte_offset)
    address = f"[$1+{offset}]" if offset else "[$1]"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            f"ld.shared.f32 $0, {address};",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_v4_f32(
    addr: Int32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load 128 bits (4 x float32) from shared memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def ld_shared_bf16_to_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    """Load a BF16 from shared memory at a 32-bit byte address and convert to FP32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "{.reg .b16 tmp; ld.shared.b16 tmp, [$1]; cvt.f32.bf16 $0, tmp;}",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Store float32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_f32_offset(
    addr: Int32,
    byte_offset: int,
    val: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store shared FP32 using a compile-time displacement."""
    offset = int(byte_offset)
    address = f"[$0+{offset}]" if offset else "[$0]"
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        f"st.shared.f32 {address}, $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_v4_f32(
    addr: Int32,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x float32) to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_bf16_from_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Convert float32 to BF16 and store one 16-bit value to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 tmp; cvt.rn.bf16.f32 tmp, $1; st.shared.b16 [$0], tmp; }",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_f16_from_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Convert float32 to FP16 and store one 16-bit value to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 tmp; cvt.rn.f16.f32 tmp, $1; st.shared.b16 [$0], tmp; }",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """Store int32 to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# PTX Intrinsics - Scatter Atomics
# =============================================================================


@dsl_user_op
def scatter_add_bf16x2(addr: Int64, val0_f32, val1_f32, *, loc=None, ip=None):
    """BF16x2 atomic reduction add to global memory.

    Packs two f32 values into bf16x2 via cvt.rn.satfinite, then does
    red.relaxed.gpu.global.add.noftz.bf16x2. Workaround for LLVM NVPTX
    bug that decomposes vector<2xbf16> into .v2.bf16x2 which ptxas rejects.
    """
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            val0_f32.ir_value(loc=loc, ip=ip),
            val1_f32.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 packed; cvt.rn.satfinite.bf16x2.f32 packed, $2, $1; red.relaxed.gpu.global.add.noftz.bf16x2 [$0], packed; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def store_bf16x2(addr: Int64, val0_f32, val1_f32, *, loc=None, ip=None):
    """Non-atomic BF16x2 store: packs two f32 -> bf16x2, plain store.

    Deterministic counterpart of :func:`scatter_add_bf16x2` for output slots
    owned by exactly one thread (no cross-CTA accumulation)."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            val0_f32.ir_value(loc=loc, ip=ip),
            val1_f32.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 packed;"
        " cvt.rn.satfinite.bf16x2.f32 packed, $2, $1;"
        " st.global.b32 [$0], packed; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def store_v4_bf16x2(addr: Int64, v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None):
    """Non-atomic vectorized BF16x2 store: 8 bf16 values (16 bytes), plain store.

    Deterministic counterpart of :func:`scatter_add_v4_bf16x2`."""
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
        " st.global.v4.b32 [$0], {p0, p1, p2, p3}; }",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def scatter_add_bf16(addr: Int64, val_f32, *, loc=None, ip=None):
    """BF16 atomic reduction add to global memory.

    Converts the f32 input to bf16 inside PTX and atomically accumulates it into
    one bf16 output lane. This is intended for opt-in approximate reductions.
    """
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            val_f32.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 packed; cvt.rn.satfinite.bf16.f32 packed, $1; red.relaxed.gpu.global.add.noftz.bf16 [$0], packed; }",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def scatter_add_v4_bf16x2(
    addr: Int64, v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None
):
    """Vectorized BF16x2 atomic reduction: 8 bf16 values (16 bytes) in one go.

    red.global.add.noftz.v4.bf16x2 [addr], {packed0, packed1, packed2, packed3}
    Each packed reg contains 2 bf16 values converted from f32 pairs.
    """
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
        " red.global.add.noftz.v4.bf16x2 [$0], {p0, p1, p2, p3}; }",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
