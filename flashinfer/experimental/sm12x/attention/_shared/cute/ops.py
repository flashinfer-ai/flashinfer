# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/_cute/ops.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
import math
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute

from cutlass import Float32, const_expr
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op


LOG2_E = math.log2(math.e)


def compute_softmax_scale_log2(softmax_scale):
    return softmax_scale * LOG2_E, None


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    return (
        cute.make_tiled_copy_B(copy_atom, tiled_mma)
        if const_expr(swapAB)
        else cute.make_tiled_copy_A(copy_atom, tiled_mma)
    )


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    return (
        cute.make_tiled_copy_A(copy_atom, tiled_mma)
        if const_expr(swapAB)
        else cute.make_tiled_copy_B(copy_atom, tiled_mma)
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
    from cutlass import CUDA_VERSION

    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
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
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(
                tAcA[(0, rest_v), 0, rest_k][1], limit
            )
    return tApA


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(
            val_i32[i], offset, mask_and_clamp=mask_and_clamp
        )
    return val[0]


@cute.jit
def warp_prefix_sum(
    val: cutlass.Int32, lane: Optional[cutlass.Int32] = None
) -> cutlass.Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val
