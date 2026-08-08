import contextlib
import functools
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from flashinfer.utils import ceil_div, next_positive_power_of_2, round_up

from ..tllm_enums import ActivationType
from .dist_aware.da_utils import (
    DADistributionSpec,
    da_distribution_target_effective_experts,
    exp_floor_probs_for_target_eff,
    sparse_probs,
    symmetric_dirichlet_probs_for_target_eff,
)

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


_EFF_EXPERTS_STREAMS: Dict[int, "torch.cuda.Stream"] = {}


def _get_eff_experts_stream(device: "torch.device") -> "torch.cuda.Stream":
    """Return the dedicated effective-experts stream for one CUDA device.

    Using a separate stream avoids blocking the main stream's pending work
    (routing kernels, previous MoE iteration) when we need to copy data
    from GPU to CPU.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"effective-experts stream requires CUDA, got {device}")
    device_idx = (
        torch.cuda.current_device() if device.index is None else int(device.index)
    )
    stream = _EFF_EXPERTS_STREAMS.get(device_idx)
    if stream is None:
        stream = torch.cuda.Stream(device=device_idx)
        _EFF_EXPERTS_STREAMS[device_idx] = stream
    return stream


def _copy_flat_tensor_to_cpu_numpy(flat: "torch.Tensor") -> np.ndarray:
    """Copy a 1D tensor to CPU with minimal default-stream blocking."""
    if flat.is_cuda:
        with torch.cuda.device(flat.device):
            stream = _get_eff_experts_stream(flat.device)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(flat.device))
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
            exp_floor_probs_for_target_eff(target_eff_experts, num_local_experts)
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
        symmetric_dirichlet_probs_for_target_eff(
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
            sparse_probs(kind, param, num_local_experts),
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
