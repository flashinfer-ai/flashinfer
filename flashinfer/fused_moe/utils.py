import contextlib
import functools
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from flashinfer.utils import ceil_div, next_positive_power_of_2, round_up

from ..tllm_enums import ActivationType

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


def _ceil_to_step(x: int, step: int) -> int:
    return ((x + step - 1) // step) * step


def get_hybrid_num_tokens_buckets(
    max_num_tokens: int, min_num_tokens: int = 1
) -> Tuple[int, ...]:
    """Generate tuning buckets with adaptive spacing.

    Pure power-of-2 spacing creates huge gaps at large values (e.g. 1024
    between bucket 1024 and 2048).  For MoE workloads the
    avg_tokens_per_expert can jump across multiple tile boundaries inside a
    single gap, forcing the autotuner to pick a kernel optimised for a very
    different workload size.

    This function uses four phases with progressively coarser spacing::

        Phase 1:  [min .. 256]   — power-of-2    (step ×2)
        Phase 2:  (256 .. 2048]  — linear step 256
        Phase 3:  (2048 .. 4096] — linear step 512
        Phase 4:  (4096 .. max]  — power-of-2    (step ×2)
    """
    buckets: List[int] = []

    # Phase 1: power-of-2 up to _PHASE1_END
    m = max(min_num_tokens, 1)
    while m <= min(max_num_tokens, _PHASE1_END):
        buckets.append(m)
        m *= 2

    # Phase 2: linear step 256 in (_PHASE1_END, _PHASE2_END]
    m = _PHASE1_END + _PHASE2_STEP
    while m <= min(max_num_tokens, _PHASE2_END):
        buckets.append(m)
        m += _PHASE2_STEP

    # Phase 3: linear step 512 in (_PHASE2_END, _PHASE3_END]
    m = _PHASE2_END + _PHASE3_STEP
    while m <= min(max_num_tokens, _PHASE3_END):
        buckets.append(m)
        m += _PHASE3_STEP

    # Phase 4: power-of-2 beyond _PHASE3_END
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
def make_hybrid_bucket_mapper(max_num_tokens: int) -> Callable[[int], int]:
    """Return a stable callable that maps token counts to hybrid buckets.

    Cached by ``max_num_tokens`` so the same object is returned on every call
    with the same argument.  This keeps AutoTuner._find_nearest_profile's
    lru_cache key stable — a fresh ``lambda`` or ``partial`` on every inference
    call would produce a new key each time and cause unbounded cache growth.
    """
    return functools.partial(map_to_hybrid_bucket, max_num_tokens=max_num_tokens)


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


_EFF_EXPERTS_STREAM: Optional["torch.cuda.Stream"] = None


def _get_eff_experts_stream() -> "torch.cuda.Stream":
    """Lazily create a dedicated CUDA stream for effective-experts computation.

    Using a separate stream avoids blocking the main stream's pending work
    (routing kernels, previous MoE iteration) when we need to copy data
    from GPU to CPU.
    """
    global _EFF_EXPERTS_STREAM
    if _EFF_EXPERTS_STREAM is None:
        _EFF_EXPERTS_STREAM = torch.cuda.Stream()
    return _EFF_EXPERTS_STREAM


def _copy_flat_tensor_to_cpu_numpy(flat: "torch.Tensor") -> np.ndarray:
    """Copy a 1D tensor to CPU with minimal default-stream blocking."""
    if flat.is_cuda:
        stream = _get_eff_experts_stream()
        event = torch.cuda.Event()
        event.record()
        stream.wait_event(event)
        with torch.cuda.stream(stream):
            flat_cpu = flat.to("cpu")
        stream.synchronize()
    else:
        flat_cpu = flat
    return flat_cpu.numpy().astype(np.int64, copy=False)


def _local_expert_counts_from_ids(
    expert_ids: np.ndarray,
    num_local_experts: int,
    local_expert_offset: int,
) -> np.ndarray:
    if expert_ids.size == 0:
        return np.zeros(num_local_experts, dtype=np.int64)

    local_end = local_expert_offset + num_local_experts
    local_mask = (expert_ids >= local_expert_offset) & (expert_ids < local_end)
    local_ids = expert_ids[local_mask] - local_expert_offset
    if local_ids.size == 0:
        return np.zeros(num_local_experts, dtype=np.int64)
    return np.bincount(
        local_ids.astype(np.int64, copy=False),
        minlength=num_local_experts,
    )[:num_local_experts].astype(np.int64, copy=False)


def compute_local_expert_counts_from_plain_ids(
    token_selected_experts: "torch.Tensor",
    num_local_experts: int,
    local_expert_offset: int = 0,
) -> np.ndarray:
    """Count local assignments from plain global expert-id tensors."""
    flat = token_selected_experts.reshape(-1)
    expert_ids = _copy_flat_tensor_to_cpu_numpy(flat)
    return _local_expert_counts_from_ids(
        expert_ids,
        num_local_experts,
        local_expert_offset,
    )


_EXP_FLOOR_FRAC = 0.1
_EXP_SEARCH_ITERS = 50
_EXP_LAMBDA_RANGE = (0.0, 20.0)
_DIRICHLET_SEARCH_ITERS = 80
_DIRICHLET_ALPHA_RANGE = (1e-6, 1e6)
DADistributionSpec = Tuple[str, str, Any]


def _clamp_effective_experts(target_eff: float, n_experts: int) -> float:
    return min(max(float(target_eff), 1.0), float(n_experts))


def _inverse_simpson_effective_experts(probs: np.ndarray) -> float:
    return float(1.0 / (probs**2).sum())


def _solve_monotonic_parameter(
    target: float,
    lo: float,
    hi: float,
    value_fn: Callable[[float], float],
    *,
    increasing: bool,
    iters: int,
) -> float:
    """Binary-search a scalar parameter for a monotonic metric."""

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        value = value_fn(mid)
        if (value < target and increasing) or (value > target and not increasing):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _apply_uniform_floor(probs: np.ndarray) -> np.ndarray:
    f = _EXP_FLOOR_FRAC
    n_experts = probs.size
    probs = (1.0 - f) * probs + f / n_experts
    probs /= probs.sum()
    return probs


def _exp_floor_probs(lam: float, n_experts: int) -> np.ndarray:
    """P(i) ∝ (1 - f) * exp(-λi) + f/n.

    Adds a uniform floor to exponential decay, better matching real MoE
    routing distributions which have heavier tails than pure exponential
    (validated on DeepSeek-V3 MMLU across 58 layers, EP=1 and EP=4).
    """
    idx = np.arange(n_experts, dtype=np.float64)
    p_exp = np.exp(-lam * idx)
    p_exp /= p_exp.sum()
    return _apply_uniform_floor(p_exp)


def _exp_decay_eff_experts(lam: float, n_experts: int) -> float:
    """Compute effective_experts for exp+floor distribution."""
    probs = _exp_floor_probs(lam, n_experts)
    return _inverse_simpson_effective_experts(probs)


def _exp_lambda_for_target_eff(target_eff: float, n_experts: int) -> float:
    """Find λ for exp+floor probabilities at a target effective-experts value."""

    target_eff = _clamp_effective_experts(target_eff, n_experts)
    return _solve_monotonic_parameter(
        target_eff,
        *_EXP_LAMBDA_RANGE,
        lambda lam: _exp_decay_eff_experts(lam, n_experts),
        increasing=False,
        iters=_EXP_SEARCH_ITERS,
    )


def _exp_floor_probs_for_target_eff(target_eff: float, n_experts: int) -> np.ndarray:
    return _exp_floor_probs(
        _exp_lambda_for_target_eff(target_eff, n_experts),
        n_experts,
    )


def _symmetric_dirichlet_expected_eff_experts(alpha: float, n_experts: int) -> float:
    """Approximate effective experts for Dirichlet([alpha] * n).

    Uses 1 / E[sum_i p_i^2], where E[sum_i p_i^2] =
    (alpha + 1) / (n * alpha + 1), then applies the same uniform floor used by
    exp+floor distributions. This is monotonic and stable for choosing the one
    free Dirichlet concentration parameter.
    """
    alpha = float(alpha)
    n = float(n_experts)
    dirichlet_second_moment = (alpha + 1.0) / (n * alpha + 1.0)
    f = _EXP_FLOOR_FRAC
    floored_second_moment = (1.0 - f) ** 2 * dirichlet_second_moment
    floored_second_moment += (2.0 * f - f**2) / n
    return float(1.0 / floored_second_moment)


def _symmetric_dirichlet_alpha_for_target_eff(
    target_eff: float,
    n_experts: int,
) -> float:
    """Binary-search alpha in [1e-6, 1e6] for a target effective-experts value."""
    target_eff = _clamp_effective_experts(target_eff, n_experts)
    return _solve_monotonic_parameter(
        target_eff,
        *_DIRICHLET_ALPHA_RANGE,
        lambda alpha: _symmetric_dirichlet_expected_eff_experts(alpha, n_experts),
        increasing=True,
        iters=_DIRICHLET_SEARCH_ITERS,
    )


def _symmetric_dirichlet_probs_for_target_eff(
    target_eff: float,
    n_experts: int,
) -> np.ndarray:
    alpha = _symmetric_dirichlet_alpha_for_target_eff(target_eff, n_experts)
    rng = np.random.default_rng(42)
    probs = rng.dirichlet(np.full(n_experts, alpha, dtype=np.float64))
    probs = np.clip(probs, np.finfo(np.float64).tiny, None)
    probs /= probs.sum()
    probs = _apply_uniform_floor(probs)
    return np.sort(probs)[::-1]


def da_distribution_target_effective_experts(
    distribution: DADistributionSpec,
    num_local_experts: int,
) -> float:
    _, kind, param = distribution
    if kind == "uniform":
        return float(num_local_experts)
    if kind == "single":
        return 1.0
    if kind in ("sparse_eff", "sparse_factor"):
        return _sparse_active_eff(kind, param, num_local_experts)[1]
    if kind in ("exp_factor", "ddist_factor"):
        return max(1.0, float(num_local_experts) / float(param))
    raise ValueError(f"Unknown DA distribution kind: {kind!r}")


def _sparse_factor_active_eff(
    param: Tuple[float, float],
    num_local_experts: int,
) -> Tuple[int, float]:
    """Convert sparse factor notation into active/effective expert counts."""

    active_factor, eff_factor = param
    n = int(num_local_experts)
    active = min(max(1, int(np.floor(float(n) / float(active_factor) + 0.5))), n)
    target_eff = min(max(1.0, float(n) / float(eff_factor)), float(active))
    return active, target_eff


def _sparse_active_eff(
    kind: str,
    param: Any,
    num_local_experts: int,
) -> Tuple[int, float]:
    if kind == "sparse_factor":
        return _sparse_factor_active_eff(param, num_local_experts)
    if kind != "sparse_eff":
        raise ValueError(f"Unknown sparse distribution kind: {kind!r}")

    active_raw, eff_raw = param
    active = min(max(1, int(active_raw)), int(num_local_experts))
    target_eff = min(max(1.0, float(eff_raw)), float(active))
    return active, target_eff


def _sparse_probs(kind: str, param: Any, num_local_experts: int) -> np.ndarray:
    active, target_eff = _sparse_active_eff(kind, param, num_local_experts)
    probs = np.zeros(int(num_local_experts), dtype=np.float64)
    probs[:active] = _symmetric_dirichlet_probs_for_target_eff(target_eff, active)
    return probs


def _shuffle_probs(probs: np.ndarray, seed: int = 42) -> np.ndarray:
    """Deterministically shuffle probability mass across expert ids."""

    rng = np.random.default_rng(seed)
    shuffled = np.zeros_like(probs)
    shuffled[rng.permutation(probs.size)] = probs
    return shuffled


def _sample_expert_assignments_from_probs(
    probs: np.ndarray,
    original_tensor: "torch.Tensor",
    top_k: int,
    local_expert_offset: int = 0,
) -> "torch.Tensor":
    num_tokens = int(original_tensor.shape[0])
    dtype = original_tensor.dtype
    probs_t = torch.from_numpy(probs).float().to(device=original_tensor.device)
    support = int(np.count_nonzero(probs > 0.0))

    if top_k <= support:
        indices = torch.multinomial(
            probs_t.expand(num_tokens, -1),
            top_k,
            replacement=False,
        )
    else:
        indices = torch.multinomial(
            probs_t,
            num_tokens * top_k,
            replacement=True,
        ).reshape(num_tokens, top_k)
    return indices.to(dtype=dtype) + int(local_expert_offset)


def generate_skewed_expert_assignments(
    target_eff_experts: float,
    original_tensor: "torch.Tensor",
    num_local_experts: int,
    num_experts: int,
    top_k: int,
    local_expert_offset: int = 0,
) -> "torch.Tensor":
    """Generate expert assignments with exp+floor distribution.

    Uses P(expert_i) ∝ (1-f)*exp(-λi) + f/N and binary-searches on λ to hit
    target_eff_experts. The uniform floor better matches real routing tails
    than pure exponential (validated on DeepSeek-V3 MMLU).

    Used during autotuner profiling only (not inference).
    """
    del num_experts

    target_eff_experts = float(target_eff_experts)
    if target_eff_experts >= float(num_local_experts):
        probs = np.full(num_local_experts, 1.0 / float(num_local_experts))
    else:
        probs = _shuffle_probs(
            _exp_floor_probs_for_target_eff(target_eff_experts, num_local_experts)
        )
    return _sample_expert_assignments_from_probs(
        probs,
        original_tensor,
        top_k,
        local_expert_offset,
    )


def generate_dirichlet_expert_assignments(
    distribution: DADistributionSpec,
    original_tensor: "torch.Tensor",
    num_local_experts: int,
    num_experts: int,
    top_k: int,
    local_expert_offset: int = 0,
) -> "torch.Tensor":
    """Generate expert ids from a symmetric Dirichlet probability law."""

    del num_experts
    probs = _shuffle_probs(
        _symmetric_dirichlet_probs_for_target_eff(
            da_distribution_target_effective_experts(distribution, num_local_experts),
            num_local_experts,
        )
    )
    return _sample_expert_assignments_from_probs(
        probs,
        original_tensor,
        top_k,
        local_expert_offset,
    )


def generate_da_distribution_assignments(
    distribution: DADistributionSpec,
    original_tensor: "torch.Tensor",
    num_local_experts: int,
    num_experts: int,
    top_k: int,
    local_expert_offset: int = 0,
) -> "torch.Tensor":
    """Generate expert ids for one DA synthetic distribution."""

    label, kind, param = distribution
    del label
    if kind == "uniform":
        return generate_skewed_expert_assignments(
            float(num_local_experts),
            original_tensor,
            num_local_experts,
            num_experts,
            top_k,
            local_expert_offset,
        )
    if kind == "single":
        return torch.full(
            (original_tensor.shape[0], top_k),
            int(local_expert_offset),
            dtype=original_tensor.dtype,
            device=original_tensor.device,
        )
    if kind == "exp_factor":
        return generate_skewed_expert_assignments(
            da_distribution_target_effective_experts(distribution, num_local_experts),
            original_tensor,
            num_local_experts,
            num_experts,
            top_k,
            local_expert_offset,
        )
    if kind == "ddist_factor":
        return generate_dirichlet_expert_assignments(
            distribution,
            original_tensor,
            num_local_experts,
            num_experts,
            top_k,
            local_expert_offset,
        )
    if kind in ("sparse_eff", "sparse_factor"):
        return _sample_expert_assignments_from_probs(
            _sparse_probs(kind, param, num_local_experts),
            original_tensor,
            top_k,
            local_expert_offset,
        )
    raise ValueError(f"Unknown DA distribution kind: {kind!r}")


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


def get_b12x_activation_name(activation_type: ActivationType) -> str:
    """Translate an activation type to the b12x kernel name."""
    if activation_type is ActivationType.Swiglu:
        return "silu"
    if activation_type is ActivationType.GegluTanh:
        return "gelu_tanh"
    if activation_type is ActivationType.Relu2:
        return "relu2"
    raise ValueError(f"Unsupported b12x activation type {activation_type!r}.")
