# Copyright (c) 2025, Tri Dao.

import math
import inspect
from typing import Type, Callable, Optional, Tuple, overload

import cutlass
import cutlass.cute as cute

from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm


from .copy_utils import predicate_k  # noqa: F401  re-exported for utils.predicate_k callers
from .cute_dsl_utils import sub_packed_f32x2

_NVVM_FMAX_REQUIRES_RESULT_TYPE = (
    sum(
        1
        for p in inspect.signature(nvvm.fmax).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    > 2
)

# Obtained from sollya:
# fpminimax(exp(x * log(2.0)), 1, [|1,24...|],[0;1],relative);
POLY_EX2 = {
    0: (1.0,),
    1: (
        1.0,
        0.922497093677520751953125,
    ),
    2: (
        1.0,
        0.6657850742340087890625,
        0.330107033252716064453125,
    ),
    3: (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    ),
    4: (
        1.0,
        0.693042695522308349609375,
        0.2412912547588348388671875,
        5.2225358784198760986328125e-2,
        1.3434938155114650726318359375e-2,
    ),
    5: (
        1.0,
        0.693151414394378662109375,
        0.24016360938549041748046875,
        5.5802188813686370849609375e-2,
        9.01452265679836273193359375e-3,
        1.86810153536498546600341796875e-3,
    ),
}

LOG2_E = math.log2(math.e)


def compute_softmax_scale_log2(softmax_scale, score_mod=None):
    """Compute softmax_scale_log2 and adjusted softmax_scale based on whether score_mod is used.

    When score_mod is None, fold the log2(e) factor into softmax_scale_log2 and set softmax_scale
    to None. When score_mod is present, keep softmax_scale separate so it can be applied before
    the score_mod, and set softmax_scale_log2 to just the change-of-base constant.

    Returns (softmax_scale_log2, softmax_scale).
    """
    if const_expr(score_mod is None):
        return softmax_scale * LOG2_E, None
    else:
        return LOG2_E, softmax_scale


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def get_smem_store_atom(
    arch: cutlass.Constexpr[int],
    element_type: Type[cute.Numeric],
    transpose: bool = False,
) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
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


@dsl_user_op
def fmax(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32 | None = None,
    *,
    loc=None,
    ip=None,
) -> Float32:
    if const_expr(_NVVM_FMAX_REQUIRES_RESULT_TYPE):
        return Float32(
            nvvm.fmax(
                T.f32(),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc,
                ip=ip,
            )
        )
    else:
        return Float32(
            nvvm.fmax(
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc,
                ip=ip,
            )
        )


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return (
            local_max[0]
            if const_expr(init_val is None)
            else fmax(local_max[0], init_val)
        )
    else:
        # [2025-06-15] x.reduce only seems to use 50% 3-input max and 50% 2-input max
        # We instead force the 3-input max.
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_max_0 = (
            fmax(init_val, res[0], res[1])
            if const_expr(init_val is not None)
            else fmax(res[0], res[1])
        )
        local_max = [
            local_max_0,
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i], res[i + 1])
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        if const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
    else:
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_sum_0 = (
            cute.arch.add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            if const_expr(init_val is not None)
            else (res[0], res[1])
        )
        local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_sum[0] = cute.arch.add_packed_f32x2(
                local_sum[0], (res[i + 0], res[i + 1])
            )
            local_sum[1] = cute.arch.add_packed_f32x2(
                local_sum[1], (res[i + 2], res[i + 3])
            )
            local_sum[2] = cute.arch.add_packed_f32x2(
                local_sum[2], (res[i + 4], res[i + 5])
            )
            local_sum[3] = cute.arch.add_packed_f32x2(
                local_sum[3], (res[i + 6], res[i + 7])
            )
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    # important: need stride 1 and not 0 for recast_tensor to work
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(
            val_i32[i], offset, mask_and_clamp=mask_and_clamp
        )
    return val[0]


@dsl_user_op
def shr_u32(
    val: cutlass.Uint32,
    shift: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Unsigned right shift with PTX's defined clamp-at-32 semantics."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def warp_prefix_sum(
    val: cutlass.Int32, lane: Optional[cutlass.Int32] = None
) -> cutlass.Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32, b: float | Float32, to_dtype: Type, *, loc=None, ip=None
) -> cutlass.Int32:
    assert to_dtype in [cutlass.BFloat16, cutlass.Float16], (
        "to_dtype must be BFloat16 or Float16"
    )
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
    """Convert Float32 tensor to Float16/BFloat16.

    Args:
        src: Source tensor with Float32 element type
        dst_or_dtype: Either a destination tensor or a dtype (Float16/BFloat16)

    Returns:
        None if dst is a tensor, or a new tensor if dtype is provided
    """
    if const_expr(isinstance(dst_or_dtype, type)):
        # dtype variant: create new tensor and call the tensor variant
        dtype = dst_or_dtype
        dst = cute.make_rmem_tensor(src.shape, dtype)
        cvt_f16(src, dst)
        return dst
    else:
        # tensor variant: write to dst
        dst = dst_or_dtype
        assert cute.size(dst.shape) == cute.size(src.shape), (
            "dst and src must have the same size"
        )
        assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"
        assert dst.element_type in [cutlass.BFloat16, cutlass.Float16], (
            "dst must be BFloat16 or Float16"
        )
        assert src.element_type is Float32, "src must be Float32"
        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
        for i in cutlass.range_constexpr(cute.size(dst_i32)):
            dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)


@dsl_user_op
@cute.jit
def evaluate_polynomial_2(
    x: Float32, y: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = cute.arch.fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out


@dsl_user_op
def combine_int_frac_ex2(
    x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None
) -> Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x_rounded).ir_value(loc=loc, ip=ip),
                Float32(frac_ex2).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "mov.b32 frac_ex_i, $2;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
            # add.u32 generates IMAD instruction and add.s32 generates LEA instruction
            # IMAD uses the FMA pipeline and LEA uses the ALU pipeline, afaik
            "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
            "mov.b32 $0, out_i;\n\t"
            "}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ex2_emulation_2(
    x: Float32, y: Float32, *, poly_degree: int = 3, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    # We assume x <= 127.0 and y <= 127.0
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    # We want to round down here, so that the fractional part is in [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm"
    )
    # The integer floor of x & y are now in the last 8 bits of xy_rounded
    # We want the next 2 ops to round to nearest even. The rounding mode is important.
    xy_rounded_back = sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def domain_offset_aligned(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    assert isinstance(tensor.iterator, cute.Pointer)
    # We assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        elem_pointer(tensor, coord).toint(),
        tensor.memspace,
        assumed_align=tensor.iterator.alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)
