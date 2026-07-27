"""
Reusable tensor initializers for AutoTuner profiling.

All functions follow the ``(shapes, dtype, device) -> torch.Tensor`` signature
expected by ``OptimizationProfile.tensor_initializers`` and
``DynamicTensorSpec.tensor_initializers``.

The public names carry an ``autotuner_initializer_`` prefix so that call sites
can ``from flashinfer.autotuner.initializers import autotuner_initializer_ones``
without any ambiguity about what the symbol is for.
"""

from typing import Callable

import torch

TensorInitializer = Callable[[tuple[int, ...], torch.dtype, torch.device], torch.Tensor]


def _empty(
    shapes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shapes, dtype=dtype, device=device)


def _zeros(
    shapes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(shapes, dtype=dtype, device=device)


def _ones(
    shapes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # Use .to(dtype) rather than the dtype= kwarg so that non-standard dtypes
    # such as float8_e4m3fn work correctly (torch.ones doesn't support them
    # directly on all PyTorch versions).
    return torch.ones(shapes, device=device).to(dtype)


def _randn(
    shapes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(shapes, device=device).to(dtype)


def _rand(
    shapes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.rand(shapes, dtype=dtype, device=device)


def _rand_scaled(
    shapes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Uniform random values scaled to [-5, 5]; the AutoTuner default."""
    return (torch.rand(shapes, device=device) * 10 - 5).to(dtype)


# ---------------------------------------------------------------------------
# Public API — prefixed names for unambiguous imports at call sites
# ---------------------------------------------------------------------------

autotuner_initializer_empty = _empty
autotuner_initializer_zeros = _zeros
autotuner_initializer_ones = _ones
autotuner_initializer_randn = _randn
autotuner_initializer_rand = _rand
autotuner_initializer_rand_scaled = _rand_scaled
