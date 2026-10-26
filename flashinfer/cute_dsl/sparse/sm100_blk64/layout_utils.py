# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
#
# Adapted from quack-kernels 0.4.1 (Apache-2.0) and kept local so the
# block-sparse kernels use the same self-contained helper style as the other
# cudnn-frontend CuTe DSL kernels.

from __future__ import annotations

import cutlass.cute as cute
from cutlass import const_expr


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two tensor dimensions without moving data."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def convert_layout_acc_mn(
    acc_layout: cute.Layout, transpose: bool = False
) -> cute.Layout:
    """Convert an MMA accumulator layout to a logical M-by-N layout."""
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def make_acc_tensor_mn_view(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(
        acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose)
    )


def reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return make_acc_tensor_mn_view(acc, transpose=transpose)


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    """Convert an accumulator layout into the fragment-A layout for a chained GEMM."""
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):
        div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
        divided = cute.logical_divide(acc_layout, ((None, None, div), None, None))
        return cute.make_layout(
            (
                (divided.shape[0][0], divided.shape[0][1], divided.shape[0][2][0]),
                divided.shape[1],
                (divided.shape[0][2][1], divided.shape[2]),
            ),
            stride=(
                (divided.stride[0][0], divided.stride[0][1], divided.stride[0][2][0]),
                divided.stride[1],
                (divided.stride[0][2][1], divided.stride[2]),
            ),
        )

    assert acc_layout.shape[2] % 2 == 0
    divided = cute.logical_divide(acc_layout, (None, None, 2))
    return cute.make_layout(
        (
            (divided.shape[0][0], divided.shape[0][1], divided.shape[2][0]),
            divided.shape[1],
            divided.shape[2][1],
        ),
        stride=(
            (divided.stride[0][0], divided.stride[0][1], divided.stride[2][0]),
            divided.stride[1],
            divided.stride[2][1],
        ),
    )


def reshape_acc_to_frgA(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))


def mma_partition_C_vec(
    s_vec: cute.Tensor, thr_mma: cute.ThrMma, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    """Broadcast a staged vector and partition it like an MMA C operand."""
    assert cute.rank(s_vec) == 2
    assert s_vec.stride[0] == 1
    stage = s_vec.shape[1]
    shape = (
        (s_vec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, s_vec.shape[0], stage)
    )
    stride = (
        (1, 0, s_vec.stride[1]) if const_expr(is_colvec) else (0, 1, s_vec.stride[1])
    )
    s_vec_mma = cute.make_tensor(s_vec.iterator, cute.make_layout(shape, stride=stride))
    partition = make_acc_tensor_mn_view(thr_mma.partition_C(s_vec_mma))
    return (
        partition[None, 0, None] if const_expr(is_colvec) else partition[0, None, None]
    )


__all__ = [
    "convert_layout_acc_frgA",
    "convert_layout_acc_mn",
    "make_acc_tensor_mn_view",
    "mma_partition_C_vec",
    "reshape_acc_to_frgA",
    "reshape_acc_to_mn",
    "select",
    "transpose_view",
]
