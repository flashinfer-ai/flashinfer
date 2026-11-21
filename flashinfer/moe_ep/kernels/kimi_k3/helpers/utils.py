# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small integer and layout helpers shared by workspace implementations."""

from typing import Iterable, List, Tuple, Union

import cutlass

IntegerType = Union[int, cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Uint64]


def round_up(value: IntegerType, alignment: IntegerType) -> IntegerType:
    return ((value + alignment - 1) // alignment) * alignment


def ceil_div(value: IntegerType, divisor: IntegerType) -> IntegerType:
    return (value + divisor - 1) // divisor


def padded_expert_rows(token_count: IntegerType, padding_block: IntegerType) -> IntegerType:
    """Row span one expert occupies in a block-padded pool.

    The single definition of a pool's per-expert stride. Communication components
    write bases derived from it while the FC12 scheduler rebuilds the same bases
    from ``expert_sizes`` at runtime; if the two ever disagree the metadata a
    kernel reads no longer describes the rows it loads.
    """
    return round_up(token_count, padding_block)


# Scale-factor rows for mma tile N = 224. A 224-token tile starts at token 224 i, i.e. mid-way through a
# 128-row SF atom; the MMA would then have to read SFB from an odd TMEM column (one column per 32 rows), which
# faults with a misaligned address. Instead the SF pool gives every 224-token tile 256 rows: token t of an expert
# sits at SF row t + 32 * (t // 224), so tile i's SF starts at row 256 i (atom-aligned, the N = 256 SFB path), and
# an expert spans ceil(count / 224) * 256 SF rows (a multiple of 128). Data rows are unchanged (pad 32).
SF_TILE_ROWS_N224 = 256


def sf_expert_rows(token_count: IntegerType, sf_padding_block: int, sf_tile_tokens) -> IntegerType:
    """SF-row span of one expert: round_up(count, sf_padding_block), or 256 per started 224-token tile."""
    if not sf_tile_tokens:
        return padded_expert_rows(token_count, sf_padding_block)
    return ceil_div(token_count, sf_tile_tokens) * SF_TILE_ROWS_N224


def sf_row_in_expert(token_in_expert: IntegerType, sf_tile_tokens) -> IntegerType:
    """SF row (relative to the expert's SF base) of an expert-local token."""
    if not sf_tile_tokens:
        return token_in_expert
    return token_in_expert + (token_in_expert // sf_tile_tokens) * (SF_TILE_ROWS_N224 - sf_tile_tokens)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def is_nested_shape(shape: Tuple) -> bool:
    return any(isinstance(dimension, tuple) for dimension in shape)


def validate_static_integer_tuple(value: Tuple, *, field_name: str) -> None:
    for element in value:
        if isinstance(element, tuple):
            validate_static_integer_tuple(element, field_name=field_name)
        elif not isinstance(element, int):
            raise TypeError(f"{field_name} must contain Python ints, got {type(element)}.")


def flatten_shape_stride(shape: Tuple, stride: Tuple) -> List[Tuple[IntegerType, IntegerType]]:
    pairs: List[Tuple[IntegerType, IntegerType]] = []
    for size, step in zip(shape, stride):
        if isinstance(size, tuple):
            pairs.extend(flatten_shape_stride(size, step))
        else:
            pairs.append((size, step))
    return pairs


def strides_equal_ignoring_singletons(shape, lhs_stride, rhs_stride) -> bool:
    """Compare strides while ignoring leaves whose logical extent is one."""
    if isinstance(shape, tuple):
        if not isinstance(lhs_stride, tuple) or not isinstance(rhs_stride, tuple):
            return False
        if len(shape) != len(lhs_stride) or len(shape) != len(rhs_stride):
            return False
        return all(
            strides_equal_ignoring_singletons(child_shape, lhs_step, rhs_step)
            for child_shape, lhs_step, rhs_step in zip(shape, lhs_stride, rhs_stride)
        )
    return shape == 1 or lhs_stride == rhs_stride


def ordered_stride(
    shape: Tuple[int, ...], mem_order: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], int]:
    stride = [0] * len(shape)
    cosize = 1
    for mode in sorted(range(len(shape)), key=lambda index: mem_order[index]):
        stride[mode] = cosize
        cosize *= shape[mode]
    return tuple(stride), cosize


def row_major_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if is_nested_shape(shape):
        raise ValueError("A nested shape needs an explicit stride.")
    stride, _ = ordered_stride(shape, tuple(reversed(range(len(shape)))))
    return stride


def cosize_from_shape_stride_tuples(shape: Tuple, stride: Tuple) -> IntegerType:
    leaf_pairs = flatten_shape_stride(shape, stride) if shape else []
    return 1 + sum((size - 1) * step for size, step in leaf_pairs)


def product(values: Iterable[IntegerType]) -> IntegerType:
    result: IntegerType = 1
    for value in values:
        result = result * value
    return result


__all__ = [
    "IntegerType",
    "ceil_div",
    "cosize_from_shape_stride_tuples",
    "flatten_shape_stride",
    "is_nested_shape",
    "is_power_of_two",
    "ordered_stride",
    "padded_expert_rows",
    "sf_expert_rows",
    "sf_row_in_expert",
    "product",
    "row_major_stride",
    "round_up",
    "strides_equal_ignoring_singletons",
    "validate_static_integer_tuple",
]
