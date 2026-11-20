# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026, Jerry Chen
# SPDX-License-Identifier: MIT
"""SM90 CuTe DSL primitives shared by DSA kernels."""

import math
from typing import Type, Callable, Literal, overload

import cutlass
import cutlass.cute as cute

from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


def get_smem_store_atom(arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric], transpose: bool = False) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),  # MMA_N
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),  # MMA_N
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # For back to back gemm, convert layout of acc0 to gemm 1 accept layout.
    # For Sm80, as the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    # For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
    # TODO: Sm90 FP8
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):  # Sm90
        l = cute.logical_divide(acc_layout, ((None, None, 2), None, None))  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:  # Sm80
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0], l.shape[2][0]),
                l.shape[1],
                l.shape[2][1],
            ),
            stride=(
                (l.stride[0], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view


def make_acc_tensor_frgA_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


@dsl_user_op
def log2f(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def logf(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return log2f(a, loc=loc, ip=ip) * math.log(2.0)


@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Scalar f32 atomicAdd: atomically adds `a` to the f32 value at `gmem_ptr`."""
    cute.arch.atomic_add(gmem_ptr.llvm_ptr, Float32(a), loc=loc, ip=ip)


@dsl_user_op
def atomic_add_fp32x4(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    gmem_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """Vectorized float4 atomicAdd via atom.relaxed.gpu.global.add.v4.f32.

    Requires `gmem_ptr` to be 16-byte aligned (guaranteed by the 128-bit
    copy atom in r2s_tiled_copy_dKVatomic).
    """
    llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            gmem_ptr.toint().ir_value(loc=loc, ip=ip),
            Float32(a0).ir_value(loc=loc, ip=ip),
            Float32(a1).ir_value(loc=loc, ip=ip),
            Float32(a2).ir_value(loc=loc, ip=ip),
            Float32(a3).ir_value(loc=loc, ip=ip),
        ],
        "atom.relaxed.gpu.global.add.v4.f32 {$0,$1,$2,$3}, [$4], {$5,$6,$7,$8};",  # relaxed mode
        "=f,=f,=f,=f,l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_fp32x4(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    gmem_ptr: cute.Pointer,
    pred: cutlass.Boolean,
    *,
    l2_cache_hint: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    loc=None,
    ip=None,
) -> None:
    """Predicated float4 reduction via red.relaxed.gpu.global.add.v4.f32.

    Requires `gmem_ptr` aligned to 16 bytes; no reduction when `pred` is false.
    Python-static `l2_cache_hint` only hints L2 eviction priority; None leaves RED unhinted.
    """
    if l2_cache_hint is None:
        asm = "@$5 red.relaxed.gpu.global.add.v4.f32 [$0], {$1, $2, $3, $4};"
    elif isinstance(l2_cache_hint, str) and l2_cache_hint in ("evict_normal", "evict_first", "evict_last"):
        asm = (
            "{ .reg .b64 pol; "
            f"createpolicy.fractional.L2::{l2_cache_hint}.b64 pol, 1.0; "
            "@$5 red.relaxed.gpu.global.add.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, pol; }"
        )
    else:
        raise ValueError(f"l2_cache_hint must be None, 'evict_normal', 'evict_first', or 'evict_last'; got {l2_cache_hint!r}")
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint().ir_value(loc=loc, ip=ip),
            Float32(a0).ir_value(loc=loc, ip=ip),
            Float32(a1).ir_value(loc=loc, ip=ip),
            Float32(a2).ir_value(loc=loc, ip=ip),
            Float32(a3).ir_value(loc=loc, ip=ip),
            cutlass.Boolean(pred).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "l,f,f,f,f,b",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    """Return the gmem pointer to element at `coord` in tensor `x`."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def elem_pointer_packed_i64(
    base_ptr_i64: cute.typing.Int,
    h_idx: cute.typing.Int,
    m_idx: cute.typing.Int,
    seqlen_q: cute.typing.Int,
    elem_type: type,
    memspace: cute.AddressSpace,
    *,
    loc=None,
    ip=None,
) -> cute.Pointer:
    """For packed layout ((qhpkv, seqlen_q),): linear_idx = h_idx * seqlen_q + m_idx."""
    linear_idx = cutlass.Int64(h_idx) * cutlass.Int64(seqlen_q) + cutlass.Int64(m_idx)
    byte_offset = linear_idx * elem_type.width // 8
    return cute.make_ptr(
        elem_type,
        cutlass.Int64(base_ptr_i64) + byte_offset,
        memspace,
        assumed_align=16,
    )


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@dsl_user_op
def cvt_f16x2_f32(a: float | Float32, b: float | Float32, to_dtype: Type, *, loc=None, ip=None) -> cutlass.Int32:
    assert to_dtype in [cutlass.BFloat16, cutlass.Float16], "to_dtype must be BFloat16 or Float16"
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@overload
def cvt_f16(src: cute.Tensor, dst: cute.Tensor) -> None: ...


@overload
def cvt_f16(src: cute.Tensor, dtype: Type[cute.Numeric]) -> cute.Tensor: ...


@cute.jit
def cvt_f16(src: cute.Tensor, dst_or_dtype):
    """Convert Float32 tensor to Float16/BFloat16."""
    if const_expr(isinstance(dst_or_dtype, type)):
        dtype = dst_or_dtype
        dst = cute.make_rmem_tensor(src.shape, dtype)
        cvt_f16(src, dst)
        return dst
    else:
        dst = dst_or_dtype
        assert cute.size(dst.shape) == cute.size(src.shape), "dst and src must have the same size"
        assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"
        assert dst.element_type in [cutlass.BFloat16, cutlass.Float16], "dst must be BFloat16 or Float16"
        assert src.element_type is Float32, "src must be Float32"
        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
        for i in cutlass.range_constexpr(cute.size(dst_i32)):
            dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)
