import contextlib
import functools
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import torch

from flashinfer.autotuner import Arithmetic, Geometric, Identity, Union
from flashinfer.utils import ceil_div, next_positive_power_of_2, round_up

logger = logging.getLogger(__name__)

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


def nearest_in_buckets(x: int, buckets: List[int]) -> int:
    """Snap *x* to the nearest power-of-2 bucket, clamped to ``[buckets[0], buckets[-1]]``."""
    return min(max(next_positive_power_of_2(x), buckets[0]), buckets[-1])


_PHASE1_END = 256
_PHASE2_STEP = 256
_PHASE2_END = 2048
_PHASE3_STEP = 512
_PHASE3_END = 4096


# Tuning buckets with adaptive spacing, expressed declaratively as a
# composition of bucket generators (see :mod:`flashinfer.autotuner`).
#
# Pure power-of-2 spacing creates huge gaps at large values (e.g. 1024 between
# bucket 1024 and 2048). For MoE workloads the avg_tokens_per_expert can jump
# across multiple tile boundaries inside a single gap, forcing the autotuner to
# pick a kernel optimised for a very different workload size. The four phases
# use progressively coarser spacing::
#
#     Phase 1:  [1 .. 256]      power-of-2    (step ×2)
#     Phase 2:  (256 .. 2048]   linear step 256
#     Phase 3:  (2048 .. 4096]  linear step 512
#     Phase 4:  (4096 .. max]   power-of-2    (step ×2)
#
# ``Identity`` guarantees the runtime ``max_num_tokens`` is always a bucket.
HYBRID_NUM_TOKENS_BUCKETS = Union(
    (
        Geometric(start=1, ratio=2, stop=_PHASE1_END),
        Arithmetic(
            start=_PHASE1_END + _PHASE2_STEP, step=_PHASE2_STEP, stop=_PHASE2_END
        ),
        Arithmetic(
            start=_PHASE2_END + _PHASE3_STEP, step=_PHASE3_STEP, stop=_PHASE3_END
        ),
        Geometric(start=_PHASE3_END * 2, ratio=2),
        Identity(),
    )
)


def _ceil_to_step(x: int, step: int) -> int:
    return ((x + step - 1) // step) * step


def get_hybrid_num_tokens_buckets(
    max_num_tokens: int, min_num_tokens: int = 1
) -> Tuple[int, ...]:
    """Materialize hybrid tuning buckets between the requested bounds.

    Tuners that should adapt their bucket set to the runtime dimension size
    pass :data:`HYBRID_NUM_TOKENS_BUCKETS` directly. This wrapper preserves
    the fixed-range API used by existing callers.
    """
    buckets: List[int] = []

    m = max(min_num_tokens, 1)
    while m <= min(max_num_tokens, _PHASE1_END):
        buckets.append(m)
        m *= 2

    m = _PHASE1_END + _PHASE2_STEP
    while m <= min(max_num_tokens, _PHASE2_END):
        buckets.append(m)
        m += _PHASE2_STEP

    m = _PHASE2_END + _PHASE3_STEP
    while m <= min(max_num_tokens, _PHASE3_END):
        buckets.append(m)
        m += _PHASE3_STEP

    m = _PHASE3_END * 2
    while m <= max_num_tokens:
        buckets.append(m)
        m *= 2

    if not buckets or buckets[-1] != max_num_tokens:
        buckets.append(max_num_tokens)

    return tuple(sorted(set(buckets)))


def map_to_hybrid_bucket(x: int, max_num_tokens: int) -> int:
    """Map an arbitrary num_tokens to the nearest hybrid bucket (rounding up).

    Mirrors the four-phase spacing of :func:`get_hybrid_num_tokens_buckets`.
    The result is clamped to ``[1, max_num_tokens]``.
    """
    if x <= 0:
        return 1
    if x >= max_num_tokens:
        return max_num_tokens
    if x <= _PHASE1_END:
        return next_positive_power_of_2(x)
    if x <= _PHASE2_END:
        return min(_ceil_to_step(x, _PHASE2_STEP), max_num_tokens)
    if x <= _PHASE3_END:
        return min(_ceil_to_step(x, _PHASE3_STEP), max_num_tokens)
    return min(next_positive_power_of_2(x), max_num_tokens)


@functools.cache
def make_hybrid_bucket_mapper(max_num_tokens: int) -> "HybridTokenMapper":
    """Return a stable, hashable mapper from token counts to hybrid buckets.

    Cached by ``max_num_tokens`` so the same object is returned on every call
    with the same argument, keeping AutoTuner._find_nearest_profile's lru_cache
    key stable.
    """
    return HybridTokenMapper(cap=max_num_tokens)


def map_to_hybrid_bucket_uncapped(x: int) -> int:
    """One-argument variant for use as a function reference in GEMM tuning.

    Same rounding logic as :func:`map_to_hybrid_bucket` but without the
    ``max_num_tokens`` clamp (the autotuner already handles upper-bound
    clamping via the generated bucket list).
    """
    if x <= 0:
        return 1
    if x <= _PHASE1_END:
        return next_positive_power_of_2(x)
    if x <= _PHASE2_END:
        return _ceil_to_step(x, _PHASE2_STEP)
    if x <= _PHASE3_END:
        return _ceil_to_step(x, _PHASE3_STEP)
    return next_positive_power_of_2(x)


@dataclass(frozen=True)
class HybridTokenMapper:
    """Map a token count to the nearest hybrid tuning bucket.

    The inference-time counterpart of :data:`HYBRID_NUM_TOKENS_BUCKETS`. A
    frozen (hence hashable) ``BucketMapper``: pass it as a spec's
    ``map_to_tuning_buckets``. With ``cap`` set the result is clamped to
    ``[1, cap]`` (see :func:`map_to_hybrid_bucket`); ``cap=None`` is the
    uncapped variant (:func:`map_to_hybrid_bucket_uncapped`).
    """

    cap: int | None = None

    def __call__(self, x: int) -> int:
        if self.cap is None:
            return map_to_hybrid_bucket_uncapped(x)
        return map_to_hybrid_bucket(x, self.cap)


_enable_piecewise_cuda_graph = True


def set_piecewise_cuda_graph_flag(enable: bool):
    """Enable or disable piecewise CUDA graph capture."""
    global _enable_piecewise_cuda_graph
    _enable_piecewise_cuda_graph = enable


def get_piecewise_cuda_graph_flag() -> bool:
    """Return ``True`` if piecewise CUDA graph capture is enabled."""
    global _enable_piecewise_cuda_graph
    return _enable_piecewise_cuda_graph


def make_random_topk_ids(
    num_experts: int, num_tokens: int, top_k: int, device: torch.device
) -> torch.Tensor:
    """
    Pick ``top_k`` distinct experts (no replacement) for each of ``num_tokens`` tokens.

    Returns a ``[num_tokens, top_k]`` int32 tensor whose rows contain unique
    values in ``[0, num_experts)``.
    """
    if num_tokens == 0 or num_experts == 0 or top_k == 0:
        return torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)

    if top_k > num_experts:
        logger.debug(
            f"top_k {top_k} is greater than num_experts {num_experts}, using top_k as num_experts"
        )
        num_experts = top_k

    weights = torch.ones((), device=device, dtype=torch.float32).expand(
        num_tokens, num_experts
    )
    return torch.multinomial(weights, top_k, replacement=False).to(torch.int32)
