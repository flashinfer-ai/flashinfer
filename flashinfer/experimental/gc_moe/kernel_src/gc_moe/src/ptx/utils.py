"""Scalar PTX utilities, warp election and thread-group controls."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Uint16, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm


@dsl_user_op
def elect_one_sync(*, loc=None, ip=None) -> Boolean:
    """Select one issuer with all threads of the active warp participating."""
    return Boolean(nvvm.elect_sync(loc=loc, ip=ip))


@dsl_user_op
def read_clock64(*, loc=None, ip=None) -> Uint64:
    """Read the per-SM cycle counter used by the reference wait deadlines."""
    return Uint64(nvvm.read_ptx_sreg_clock64(T.i64(), loc=loc, ip=ip))


@dsl_user_op
def device_trap(*, loc=None, ip=None) -> None:
    """A timed-out barrier must fail the kernel, never resume as success."""
    llvm.inline_asm(None, [], 'trap;', '', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def sync_aligned(num_threads: cutlass.Constexpr, barrier_idx, *, loc=None, ip=None) -> None:
    """Synchronize an aligned thread group with a static or runtime barrier ID."""
    if isinstance(barrier_idx, int):
        llvm.inline_asm(None, [], f'bar.sync {barrier_idx}, {num_threads};', '',
                        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    else:
        llvm.inline_asm(None, [barrier_idx.ir_value(loc=loc, ip=ip)],
                        f'bar.sync $0, {num_threads};', 'r',
                        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def sync_unaligned(num_threads: cutlass.Constexpr, barrier_idx: cutlass.Constexpr,
                   *, loc=None, ip=None) -> None:
    """Join a thread group that can reach the barrier through separate branches."""
    llvm.inline_asm(None, [], f'barrier.sync {barrier_idx}, {num_threads};', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def assume_u32_lt(value: Uint32, bound: Uint32,
                  *, loc=None, ip=None) -> None:
    """CuTe translation of the reference's unsigned __builtin_assume."""
    condition = llvm.icmp(
        llvm.ICmpPredicate.ult,
        value.ir_value(loc=loc, ip=ip),
        bound.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.intr_assume(condition, [], [], loc=loc, ip=ip)


@dsl_user_op
def accumulate_bf16_into_f32(acc: Float32, bf16_bits: Uint16,
                             *, loc=None, ip=None) -> Float32:
    """Reference ptx::accumulate: one BF16-to-FP32 add with RN semantics."""
    value = llvm.inline_asm(
        T.f32(), [bf16_bits.ir_value(loc=loc, ip=ip),
                  acc.ir_value(loc=loc, ip=ip)],
        'add.rn.f32.bf16 $0, $1, $2;', '=f,h,f',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(value)


@dsl_user_op
def warp_first_set_bit(mask: Uint32, *, loc=None, ip=None) -> Uint32:
    """Unsigned __ffs(mask)-1, including UINT_MAX for an empty mask."""
    lane = llvm.inline_asm(
        T.i32(), [mask.ir_value(loc=loc, ip=ip)],
        '{ .reg .b32 reversed; .reg .b32 first; '
        'brev.b32 reversed, $1; '
        'bfind.shiftamt.u32 first, reversed; '
        'mov.b32 $0, first; }',
        '=r,r', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(lane)


@dsl_user_op
def warp_nth_set_bit(mask: Uint32, one_based_offset: Uint32,
                     *, loc=None, ip=None) -> Uint32:
    """PTX fns.b32, matching reference __fns(mask, 0, offset)."""
    lane = llvm.inline_asm(
        T.i32(), [mask.ir_value(loc=loc, ip=ip),
                  one_based_offset.ir_value(loc=loc, ip=ip)],
        'fns.b32 $0, $1, 0, $2;', '=r,r,r', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(lane)


@dsl_user_op
def undefined_u32(*, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.mlir_undef(T.i32(), loc=loc, ip=ip))


@cute.jit
def warpgroup_reg_dealloc(reg_count: cutlass.Constexpr) -> None:
    """Reference warpgroup_reg_dealloc<N>, with a compile-time register count."""
    cute.arch.setmaxregister_decrease(reg_count)


@cute.jit
def warpgroup_reg_alloc(reg_count: cutlass.Constexpr) -> None:
    """Reference warpgroup_reg_alloc<N>, with a compile-time register count."""
    cute.arch.setmaxregister_increase(reg_count)


@dsl_user_op
def fence_gpu(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'membar.gl;', '~{memory}', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'fence.acq_rel.sys;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def fence_sys(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'membar.sys;', '~{memory}', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def sync_loader(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'bar.sync 2, 128;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def sync_math(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'bar.sync 1, 256;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def sync_tile(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'bar.sync 0, 384;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def sync_activation(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'bar.sync 3, 64;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def sync_weight(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'bar.sync 4, 64;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
