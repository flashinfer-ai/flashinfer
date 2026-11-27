"""Small integer and layout helpers shared by workspace implementations."""

from typing import Iterable, List, Tuple, Union

import cutlass


IntegerType = Union[int, cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Uint64]


def round_up(value: IntegerType, alignment: IntegerType) -> IntegerType:
    return ((value + alignment - 1) // alignment) * alignment


def ceil_div(value: IntegerType, divisor: IntegerType) -> IntegerType:
    return (value + divisor - 1) // divisor


def padded_expert_rows(
    token_count: IntegerType, padding_block: IntegerType
) -> IntegerType:
    """Row span one expert occupies in a block-padded pool.

    The single definition of a pool's per-expert stride. Communication components
    write bases derived from it while the FC12 scheduler rebuilds the same bases
    from ``expert_sizes`` at runtime; if the two ever disagree the metadata a
    kernel reads no longer describes the rows it loads.
    """
    return round_up(token_count, padding_block)


def token_slot_rows(tile_n: int) -> int:
    """Rows one token tile occupies in the per-expert data and scale-factor pools.

    A tile N that divides or is divided by the 128-row scale-factor atom tiles both
    pools densely, so its slot is the tile itself. Any other tile N (192, 240) gets
    a slot rounded up to whole atoms: a scale-factor TMA box starts on an atom
    boundary, and the epilogue's 64-token stores must not reach into the next tile.
    The slot's trailing rows are padding that no route is ever assigned to.

    Strided slots also split each expert's routes evenly over its token tiles
    (``balanced_tile_rows``) instead of filling every tile but the last.
    """
    if tile_n % 128 == 0 or 128 % tile_n == 0:
        return tile_n
    return round_up(tile_n, 128)


def expert_pool_rows(
    token_count: IntegerType, padding_block: IntegerType, tile_n: int, slot_rows: int
) -> IntegerType:
    """Row span one expert occupies in a pool, given the token tile's slot.

    Dense slots keep the block-padded stride of ``padded_expert_rows``. Strided
    slots give every token tile, including the expert's last partial one, a whole
    slot, which is already a multiple of every padding block.
    """
    if slot_rows == tile_n:
        return padded_expert_rows(token_count, padding_block)
    return ((token_count + (tile_n - 1)) // tile_n) * slot_rows


def balanced_tile_rows(token_count: cutlass.Int32, tile_n: int) -> cutlass.Int32:
    """Routes in each of an expert's token tiles but the last, under strided slots.

    The expert keeps the fewest tiles a tile N allows, ceil(count / tile_n), and
    spreads its routes evenly over them: every tile but the last takes the even
    share rounded up to 16 routes (the granule of the two-CTA instruction N), or
    the even share itself where that rounding would leave the last tile empty;
    the last tile takes the rest. Every tile holds at least one route and at most
    ``tile_n``. Device code only; zero for an expert without routes.
    """
    tiles = cutlass.max(
        (token_count + cutlass.Int32(tile_n - 1)) // cutlass.Int32(tile_n),
        cutlass.Int32(1),
    )
    even_share = (token_count + tiles - cutlass.Int32(1)) // tiles
    rounded_share = (
        (even_share + cutlass.Int32(15)) // cutlass.Int32(16) * cutlass.Int32(16)
    )
    leaves_last_tile = cutlass.Int32(
        (tiles - cutlass.Int32(1)) * rounded_share < token_count
    )
    return even_share + (rounded_share - even_share) * leaves_last_tile


def tile_rows_reciprocal(tile_rows: cutlass.Int32) -> cutlass.Int32:
    """ceil(2^32 / tile_rows), zero for an expert without routes; see ``slot_physical_row``. Device code only."""
    divisor = cutlass.Int64(cutlass.max(tile_rows, cutlass.Int32(1)))
    reciprocal = cutlass.Int32(
        ((cutlass.Int64(1) << cutlass.Int64(32)) + divisor - cutlass.Int64(1))
        // divisor
    )
    return reciprocal * cutlass.Int32(tile_rows > cutlass.Int32(0))


def slot_physical_row(
    local_row: cutlass.Int32,
    tile_rows: cutlass.Int32,
    reciprocal: cutlass.Int32,
    slot_rows: int,
) -> cutlass.Int32:
    """Pool row, relative to the expert's base, of the expert's ``local_row``-th route under strided slots.

    ``tile_rows`` is the expert's ``balanced_tile_rows``: token tile t holds its
    routes [t * tile_rows, (t + 1) * tile_rows) at the start of slot t. The tile
    index is ``local_row // tile_rows`` taken as a multiply by ``reciprocal``
    (``tile_rows_reciprocal``), exact because the rounding error of the
    reciprocal, below ``tile_rows`` <= 240, times any route count below 2^24
    stays under 2^32. Device code only.
    """
    tile = cutlass.Int32(
        (cutlass.Int64(local_row) * cutlass.Int64(reciprocal)) >> cutlass.Int64(32)
    )
    return tile * cutlass.Int32(slot_rows) + (local_row - tile * tile_rows)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def is_nested_shape(shape: Tuple) -> bool:
    return any(isinstance(dimension, tuple) for dimension in shape)


def validate_static_integer_tuple(value: Tuple, *, field_name: str) -> None:
    for element in value:
        if isinstance(element, tuple):
            validate_static_integer_tuple(element, field_name=field_name)
        elif not isinstance(element, int):
            raise TypeError(
                f"{field_name} must contain Python ints, got {type(element)}."
            )


def flatten_shape_stride(
    shape: Tuple, stride: Tuple
) -> List[Tuple[IntegerType, IntegerType]]:
    pairs: List[Tuple[IntegerType, IntegerType]] = []
    for size, step in zip(shape, stride, strict=False):
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
            for child_shape, lhs_step, rhs_step in zip(
                shape, lhs_stride, rhs_stride, strict=False
            )
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
    "balanced_tile_rows",
    "ceil_div",
    "cosize_from_shape_stride_tuples",
    "flatten_shape_stride",
    "is_nested_shape",
    "is_power_of_two",
    "ordered_stride",
    "expert_pool_rows",
    "padded_expert_rows",
    "slot_physical_row",
    "tile_rows_reciprocal",
    "token_slot_rows",
    "product",
    "row_major_stride",
    "round_up",
    "strides_equal_ignoring_singletons",
    "validate_static_integer_tuple",
]
