"""
Composable bucket generators for AutoTuner profiling.

``DynamicTensorSpec.gen_tuning_buckets`` describes which dimension sizes the
autotuner should profile. The bucket sets are produced declaratively by the
small, frozen (hence trivially hashable) dataclasses in this module rather than
by opaque callables, so a spec -- and the ``TuningConfig`` that holds it --
hashes structurally and stays a stable cache key.

Each generator maps the runtime size ``n`` of the tuned dimension to a tuple of
bucket sizes; :class:`Union` composes several generators by merging their
outputs. A bare ``tuple[int, ...]`` is also accepted wherever a generator is,
denoting a constant bucket set independent of ``n``.
"""

from dataclasses import dataclass
from typing import TypeAlias

from flashinfer.utils import last_positive_power_of_2


@dataclass(frozen=True)
class Geometric:
    """Buckets ``start, start*ratio, start*ratio**2, ...``.

    Generated up to ``min(stop, n)`` inclusive, where ``n`` is the runtime size
    of the tuned dimension and ``stop`` is an optional fixed upper bound
    (``None`` means "bounded only by ``n``").
    """

    start: int
    ratio: int = 2
    stop: int | None = None

    def __post_init__(self) -> None:
        assert self.start >= 1 and self.ratio >= 2, (
            "Geometric requires start >= 1 and ratio >= 2"
        )

    def __call__(self, n: int) -> tuple[int, ...]:
        upper = n if self.stop is None else min(self.stop, n)
        out: list[int] = []
        m = self.start
        while m <= upper:
            out.append(m)
            m *= self.ratio
        return tuple(out)


@dataclass(frozen=True)
class Arithmetic:
    """Buckets ``start, start+step, start+2*step, ...``.

    Generated up to ``min(stop, n)`` inclusive (see :class:`Geometric` for the
    meaning of ``stop`` / ``n``).
    """

    start: int
    step: int
    stop: int | None = None

    def __post_init__(self) -> None:
        assert self.start >= 1 and self.step >= 1, (
            "Arithmetic requires start >= 1 and step >= 1"
        )

    def __call__(self, n: int) -> tuple[int, ...]:
        upper = n if self.stop is None else min(self.stop, n)
        out: list[int] = []
        m = self.start
        while m <= upper:
            out.append(m)
            m += self.step
        return tuple(out)


@dataclass(frozen=True)
class Identity:
    """A single bucket equal to the runtime dimension size ``n`` itself."""

    def __call__(self, n: int) -> tuple[int, ...]:
        return (n,) if n >= 1 else ()


@dataclass(frozen=True)
class PowerOfTwoFloor:
    """A single bucket: the largest power of two ``<= n``."""

    def __call__(self, n: int) -> tuple[int, ...]:
        return (last_positive_power_of_2(n),)


@dataclass(frozen=True)
class Union:
    """Sorted, de-duplicated union of the buckets produced by each part."""

    parts: tuple["BucketGen", ...]

    def __call__(self, n: int) -> tuple[int, ...]:
        merged: set[int] = set()
        for part in self.parts:
            merged.update(part(n))
        return tuple(sorted(merged))


# A declarative, hashable description of a tuning-bucket set. Each variant is a
# frozen dataclass that, given the runtime dimension size, returns the buckets
# to profile. Compose them with :class:`Union`.
BucketGen: TypeAlias = Geometric | Arithmetic | Identity | PowerOfTwoFloor | Union
