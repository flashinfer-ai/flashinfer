import contextlib
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import torch

from ..utils import ceil_div, round_up

is_torch_compiling_flag = False

AuxStreamType = Enum(
    "AuxStreamType",
    ["Attention", "MoeShared", "MoeChunkingOverlap"],
)
EventType = Enum(
    "EventType",
    ["Main", "Attention", "MoeShared", "MoeChunkingOverlap"],
    start=0,
)


def set_torch_compiling(enable: bool):
    """Set the global flag indicating whether ``torch.compile`` is active."""
    global is_torch_compiling_flag
    is_torch_compiling_flag = enable


def is_torch_compiling() -> bool:
    """Return ``True`` if ``torch.compile`` is currently active."""
    global is_torch_compiling_flag
    return is_torch_compiling_flag


_global_attrs = threading.local()


def get_global_attrs():
    """Return the thread-local global attributes object."""
    return _global_attrs


_model_extra_attrs = threading.local()


def get_model_extra_attrs():
    """Return the current thread-local model extra attributes, or ``None``."""
    return getattr(_model_extra_attrs, "attrs", None)


@contextlib.contextmanager
def model_extra_attrs(attrs: Dict):
    old_attrs = getattr(_model_extra_attrs, "attrs", None)
    _model_extra_attrs.attrs = attrs
    try:
        yield
    finally:
        _model_extra_attrs.attrs = old_attrs


def with_model_extra_attrs(get_attrs):
    """Decorator that sets model extra attributes from *get_attrs(self)* during the call."""

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            with model_extra_attrs(get_attrs(self)):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


@dataclass
class Fp4QuantizedTensor:
    fp4_tensor: torch.Tensor
    scaling_factor: torch.Tensor
    is_sf_swizzled: bool = True

    @property
    def shape(self):
        return self.fp4_tensor.shape


def compute_swizzled_sf_shape(row: int, col: int):
    """Return padded ``(row, col)`` for swizzled FP4 scaling-factor layout."""
    padded_row = round_up(row, 128)
    padded_col = round_up(col, 4)
    return padded_row, padded_col


def swizzle_sf(sf: torch.Tensor, rows: int, cols: int, scaling_vector_size: int = 16):
    """Swizzle FP4 scaling factors using C++ torch op implementation
    Args:
        sf: [b, rows, cols_sf] or [rows, cols_sf]. The original unswizzled scaling factors.
        rows: rows of the original unquantized tensor
        cols_sf: ceil_div(cols, scaling_vector_size) where cols is the number of columns of the original unquantized tensor
        scaling_vector_size: the size of the scaling vector
    Returns:
        [b * round_up(rows, 128) * round_up(cols_sf, 4), ] 1D swizzled scaling factors, possibly with rows and cols padded.
    """
    sf_cols = ceil_div(cols, scaling_vector_size)
    sf = sf.view(-1, rows, sf_cols)
    return torch.ops.trtllm.block_scale_interleave(sf)


def unswizzle_sf(sf: torch.Tensor, rows: int, cols: int, scaling_vector_size: int = 16):
    """Swizzle FP4 scaling factors using C++ torch op implementation
    Args:
        sf: The (padded and) swizzled scaling factors.
        rows: rows of the original unquantized tensor
        cols: cols of the original unquantized tensor
        scaling_vector_size: the size of the scaling vector
    Returns:
        2D unswizzled scaling factors
    """
    sf_cols = ceil_div(cols, scaling_vector_size)
    sf = sf.view(-1, rows, sf_cols)
    return torch.ops.trtllm.block_scale_interleave_reverse(sf).view(-1, sf_cols)


@torch.library.custom_op("trtllm::reswizzle_sf", mutates_args=())
def reswizzle_sf(
    sf: torch.Tensor, rows: int, cols: int, scaling_vector_size: int = 16
) -> torch.Tensor:
    """Reswizzle FP4 scaling factors using C++ torch op implementation.
       It unswizzles the scaling factors in each partition first, then concatenates them together, and finally swizzles them back.
    Args:
        sf: The (padded and) swizzled scaling factors.
        rows: rows of the original unquantized tensor
        cols: cols of the original unquantized tensor
        scaling_vector_size: the size of the scaling vector
    Returns:
        1D reswizzled scaling factors
    """
    sf_cols = ceil_div(cols, scaling_vector_size)
    padded_rows, padded_sf_cols = compute_swizzled_sf_shape(rows, sf_cols)
    padded_cols = padded_sf_cols * scaling_vector_size

    assert sf.numel() % (padded_rows * padded_sf_cols) == 0
    num_partitions = sf.numel() // (padded_rows * padded_sf_cols)

    sf_reshaped = sf.view(num_partitions, padded_rows, padded_sf_cols)

    # Unswizzle each partition
    sf_unswizzled = unswizzle_sf(
        sf_reshaped, padded_rows, padded_cols, scaling_vector_size
    )

    # Brings the unswizzled scaling factors in each partition together
    total_rows = num_partitions * rows
    sf_unswizzled = sf_unswizzled.view(num_partitions, padded_rows, padded_sf_cols)
    sf_concatenated = sf_unswizzled[
        :, :rows, :sf_cols
    ].contiguous()  # TODO: This will incur a elementwise kernel
    sf_concatenated = sf_concatenated.view(total_rows, sf_cols)

    # Finally swizzle the concatenated scaling factors
    return swizzle_sf(sf_concatenated, total_rows, cols, scaling_vector_size)


@torch.library.register_fake("trtllm::reswizzle_sf")
def _(sf, rows, cols, scaling_vector_size=16):
    sf_cols = ceil_div(cols, scaling_vector_size)
    padded_rows, padded_sf_cols = compute_swizzled_sf_shape(rows, sf_cols)
    num_partitions = sf.numel() // (padded_rows * padded_sf_cols)
    total_rows = num_partitions * rows
    sz = round_up(total_rows, 128) * round_up(cols, 4)
    return sf.new_empty(sz)


def next_positive_power_of_2(x: int) -> int:
    """Return the smallest power of 2 that is ``>= x``.

    Returns 1 when *x* < 1.  Safe for use inside ``torch.compile``
    (avoids ``bit_length()``).
    """
    if x < 1:
        return 1

    # Following code is equivalent to 1 << (x - 1).bit_length()
    # But this impl does not contain bit_length() so can be used by torch compile.
    # It can correctly handle 64bit number which should be enough for now.
    n = x - 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def last_positive_power_of_2(x: int) -> int:
    """Return the largest power of 2 that is ``<= x``.

    If *x* is itself a power of 2, returns *x*.
    """
    next = next_positive_power_of_2(x)
    if next == x:
        return next

    return next // 2


def nearest_in_buckets(x: int, buckets: List[int]) -> int:
    """Snap *x* to the nearest power-of-2 bucket, clamped to ``[buckets[0], buckets[-1]]``."""
    return min(max(next_positive_power_of_2(x), buckets[0]), buckets[-1])


def get_power_of_2_num_tokens_buckets(max_num_tokens) -> Tuple[int]:
    """Return descending power-of-2 buckets from ``next_power_of_2(max_num_tokens)`` down to 1."""
    max_num_tokens = next_positive_power_of_2(max_num_tokens)
    num_token_buckets = []
    m = max_num_tokens
    while m >= 1:
        num_token_buckets.append(m)
        m //= 2

    return tuple(num_token_buckets)


def get_last_power_of_2_num_tokens_buckets(
    max_num_tokens, min_num_tokens=1
) -> Tuple[int, ...]:
    """Return descending power-of-2 buckets from ``last_power_of_2(max_num_tokens)`` down to *min_num_tokens*."""
    max_num_tokens = last_positive_power_of_2(max_num_tokens)
    num_token_buckets = []
    m = max_num_tokens
    while m >= min_num_tokens:
        num_token_buckets.append(m)
        m //= 2
    return tuple(num_token_buckets)


def round_to_nearest_bucket(
    x: int, buckets: Sequence[int], round_map: bool = False
) -> int:
    """Map *x* to the nearest bucket using floor or ceil semantics.

    Args:
        x: The value to map.
        buckets: Bucket values in **ascending** order.  Must not be empty.
        round_map: Rounding direction.

            * ``False`` (default) -- **floor**: return the largest bucket
              that is ``<= x``.  If *x* is smaller than every bucket, the
              smallest bucket is returned (clamped).
            * ``True`` -- **ceil**: return the smallest bucket that is
              ``>= x``.  If *x* is larger than every bucket, the largest
              bucket is returned (clamped).

    Returns:
        The matched bucket value.  Always one of the elements in *buckets*.

    Examples::

        >>> round_to_nearest_bucket(350, [100, 200, 500, 1000])
        200
        >>> round_to_nearest_bucket(350, [100, 200, 500, 1000], round_map=True)
        500
        >>> round_to_nearest_bucket(2000, [100, 200, 500, 1000], round_map=True)
        1000
    """
    if len(buckets) == 0:
        raise ValueError("buckets must be non-empty")
    if round_map:
        for b in buckets:
            if b >= x:
                return b
        return buckets[-1]
    else:
        for b in reversed(buckets):
            if b <= x:
                return b
        return buckets[0]


def make_bucket_mapper(buckets: Tuple[int, ...], round_map: bool = False):
    """Create a mapper function for :class:`DynamicTensorSpec.map_to_tuning_buckets`.

    The returned callable maps any integer *x* to the nearest value in
    *buckets*, using floor or ceil semantics controlled by *round_map*.
    Duplicates in *buckets* are removed and values are sorted internally.

    Args:
        buckets: The set of allowed bucket values.
        round_map: If ``False`` (default) the mapper rounds **down** (floor);
            if ``True`` it rounds **up** (ceil).  In both cases the result is
            clamped to the bucket range -- see
            :func:`round_to_nearest_bucket` for details.

    Returns:
        A ``Callable[[int], int]`` suitable for passing as
        ``map_to_tuning_buckets`` to :class:`DynamicTensorSpec`.

    Examples::

        >>> mapper = make_bucket_mapper((100, 200, 500, 1000), round_map=False)
        >>> mapper(350)
        200
        >>> mapper_up = make_bucket_mapper((100, 200, 500, 1000), round_map=True)
        >>> mapper_up(350)
        500
    """
    if len(buckets) == 0:
        raise ValueError("buckets must be non-empty")
    sorted_buckets = tuple(sorted(set(buckets)))

    def _mapper(x: int) -> int:
        return round_to_nearest_bucket(x, sorted_buckets, round_map)

    return _mapper


def get_fp4_shape(input_shape, sf_vec_size, is_swizzled_layout=True):
    """Compute the FP4 tensor shape and scaling-factor size from a full-precision shape."""
    m = 1
    for i in range(len(input_shape) - 1):
        m *= input_shape[i]

    output_shape = [i for i in input_shape]
    output_shape[-1] //= 2

    scale_shape = (
        round_up(m, 128) * round_up(input_shape[-1] // sf_vec_size, 4)
        if is_swizzled_layout
        else m * (input_shape[-1] // sf_vec_size)
    )
    return output_shape, scale_shape


def fp4_scale_infer_shape(input_shapes: List[List[int]]):
    """Calculate the dimensions of the fp4 scale tensor."""
    out_shape, scale_shape = get_fp4_shape(input_shapes[0], sf_vec_size=16)
    return scale_shape * 2


_enable_piecewise_cuda_graph = True


def set_piecewise_cuda_graph_flag(enable: bool):
    """Enable or disable piecewise CUDA graph capture."""
    global _enable_piecewise_cuda_graph
    _enable_piecewise_cuda_graph = enable


def get_piecewise_cuda_graph_flag() -> bool:
    """Return ``True`` if piecewise CUDA graph capture is enabled."""
    global _enable_piecewise_cuda_graph
    return _enable_piecewise_cuda_graph
