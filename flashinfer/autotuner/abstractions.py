"""
Core value-type abstractions for the AutoTuner.

These are small, dependency-free frozen dataclasses (and type aliases) that
describe the autotuner's tuning space: *where* a dimension lives
(:class:`DimensionCoordinates`), *what range* it spans (:class:`StaticDim` /
:class:`DynamicDim`), and *which bucket sizes* to profile along it (the
:data:`BucketGen` variants). They are pure data -- carrying no logic beyond
field validation -- so they hash structurally and stay stable cache keys. The
logic that interprets a :data:`BucketGen` (materializing it into a concrete
bucket tuple for a given runtime size) lives in
:mod:`flashinfer.autotuner.autotuner` as ``gen_buckets``.

They live in this leaf module so the other autotuner submodules
(``bucket_mappers``, ``infer_shapes``) and ``autotuner`` itself can reference
them at runtime without importing back into ``autotuner`` -- avoiding import
cycles.
"""

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class DimensionCoordinates:
    """
    Coordinates of a dimension within the inputs to the operation.
    """

    input_idx: int
    dim_idx: int


@dataclass(frozen=True)
class StaticDim:
    val: int


@dataclass(frozen=True)
class DynamicDim:
    """Range of one dimension"""

    min: int
    opt: int
    # The largest tuning bucket is unbounded above, so ``max`` may be
    # ``float("inf")``. Keep it as ``int | float`` rather than coercing to
    # ``int`` -- ``int(float("inf"))`` raises ``OverflowError``.
    max: int | float


Dim: TypeAlias = DynamicDim | StaticDim


# ---------------------------------------------------------------------------
# Tuning-bucket generators
# ---------------------------------------------------------------------------
# A declarative, hashable description of which dimension sizes the autotuner
# should profile along a tuned dimension. Each variant below is a frozen (hence
# trivially hashable) dataclass carrying only parameters -- no logic. The free
# function ``gen_buckets`` (in :mod:`flashinfer.autotuner.autotuner`) maps a
# variant plus the runtime dimension size ``n`` to the concrete tuple of bucket
# sizes; :class:`Union` composes several variants. A bare ``tuple[int, ...]`` is
# also accepted wherever a ``BucketGen`` is, denoting a constant bucket set
# independent of ``n``.


@dataclass(frozen=True)
class Geometric:
    """Buckets ``start, start*ratio, start*ratio**2, ...``.

    Materialized up to ``min(stop, n)`` inclusive, where ``n`` is the runtime
    size of the tuned dimension and ``stop`` is an optional fixed upper bound
    (``None`` means "bounded only by ``n``").
    """

    start: int
    ratio: int = 2
    stop: int | None = None

    def __post_init__(self) -> None:
        assert self.start >= 1 and self.ratio >= 2, (
            "Geometric requires start >= 1 and ratio >= 2"
        )


@dataclass(frozen=True)
class Arithmetic:
    """Buckets ``start, start+step, start+2*step, ...``.

    Materialized up to ``min(stop, n)`` inclusive (see :class:`Geometric` for
    the meaning of ``stop`` / ``n``).
    """

    start: int
    step: int
    stop: int | None = None

    def __post_init__(self) -> None:
        assert self.start >= 1 and self.step >= 1, (
            "Arithmetic requires start >= 1 and step >= 1"
        )


@dataclass(frozen=True)
class Identity:
    """A single bucket equal to the runtime dimension size ``n`` itself."""


@dataclass(frozen=True)
class PowerOfTwoFloor:
    """A single bucket: the largest power of two ``<= n``."""


@dataclass(frozen=True)
class Union:
    """Sorted, de-duplicated union of the buckets produced by each part."""

    parts: tuple["BucketGen", ...]


BucketGen: TypeAlias = Geometric | Arithmetic | Identity | PowerOfTwoFloor | Union
