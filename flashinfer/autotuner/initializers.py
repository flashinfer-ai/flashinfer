"""
Reusable tensor initializers for AutoTuner profiling.

Each initializer is a small frozen (hence trivially hashable) dataclass with the
``(shapes, dtype, device) -> torch.Tensor`` call signature expected by
``OptimizationProfile.tensor_initializers`` and
``TuningConfig.tensor_initializers``. Expressing them as a declarative
vocabulary -- rather than opaque callables -- keeps the tuning config
value-comparable and free of fresh-lambda churn.

The ``autotuner_initializer_*`` names are retained as ready-made instances so
existing call sites keep importing them unchanged.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


class TensorInitializer(Protocol):
    """Fills a synthesized input tensor with profiling content."""

    def __call__(
        self, shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class Empty:
    """Uninitialized tensor (``torch.empty``)."""

    def __call__(
        self, shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.empty(shapes, dtype=dtype, device=device)


@dataclass(frozen=True)
class Zeros:
    """All-zeros tensor."""

    def __call__(
        self, shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(shapes, dtype=dtype, device=device)


@dataclass(frozen=True)
class Ones:
    """All-ones tensor."""

    def __call__(
        self, shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # Use .to(dtype) rather than the dtype= kwarg so that non-standard dtypes
        # such as float8_e4m3fn work correctly (torch.ones doesn't support them
        # directly on all PyTorch versions).
        return torch.ones(shapes, device=device).to(dtype)


@dataclass(frozen=True)
class Randn:
    """Standard-normal random values."""

    def __call__(
        self, shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.randn(shapes, device=device).to(dtype)


@dataclass(frozen=True)
class Rand:
    """Uniform random values in [0, 1)."""

    def __call__(
        self, shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.rand(shapes, dtype=dtype, device=device)


@dataclass(frozen=True)
class RandScaled:
    """Uniform random values scaled to [-5, 5]; the AutoTuner default."""

    def __call__(
        self, shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return (torch.rand(shapes, device=device) * 10 - 5).to(dtype)


@dataclass(frozen=True)
class Full:
    """Constant fill (``torch.full``)."""

    value: int | float

    def __call__(
        self, shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.full(shapes, self.value, dtype=dtype, device=device)


@dataclass(frozen=True)
class RandInt:
    """Random integers in ``[low, high)``.

    ``dtype`` overrides the synthesized tensor's dtype when set (e.g. to pin
    ``uint8`` / ``int32``); otherwise the synthesized dtype is used. ``seed``
    makes the draw deterministic across processes.
    """

    low: int
    high: int
    dtype: torch.dtype | None = None
    seed: int | None = None

    def __call__(
        self, shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        target = self.dtype if self.dtype is not None else dtype
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=device).manual_seed(self.seed)
        return torch.randint(
            self.low,
            self.high,
            shapes,
            dtype=target,
            device=device,
            generator=generator,
        )


# ---------------------------------------------------------------------------
# Public ready-made instances -- back-compat aliases for the built-in vocabulary
# ---------------------------------------------------------------------------

autotuner_initializer_empty = Empty()
autotuner_initializer_zeros = Zeros()
autotuner_initializer_ones = Ones()
autotuner_initializer_randn = Randn()
autotuner_initializer_rand = Rand()
autotuner_initializer_rand_scaled = RandScaled()
