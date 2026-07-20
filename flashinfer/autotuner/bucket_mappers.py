"""
Composable bucket *mappers* for AutoTuner inference-time lookup.

A bucket mapper is the inference-time inverse of a bucket generator (the
:data:`BucketGen` variants in :mod:`flashinfer.autotuner.abstractions`,
materialized by ``gen_buckets``): given the runtime size of a tuned dimension it
returns the single bucket value used to build the profiling cache key (see
``AutoTuner._find_nearest_profile``).

Like the generators, mappers are small frozen (hence trivially hashable)
dataclasses with a ``__call__(int) -> int`` so a :class:`DynamicTensorSpec` --
and the :class:`TuningConfig` that holds it -- hashes structurally and can be a
stable cache key.

``BucketMapper`` is a :class:`typing.Protocol` rather than a closed union (as
``BucketGen`` is): the domain-specific hybrid-token mapper lives in
``flashinfer.fused_moe.utils`` and importing it here would create an import
cycle, so the field type is structural. Any frozen dataclass with the
``__call__(int) -> int`` signature qualifies.
"""

import functools
from dataclasses import dataclass
from typing import Protocol, Sequence

from flashinfer.utils import last_positive_power_of_2


class BucketMapper(Protocol):
    """Maps a runtime dimension size to a single tuning-bucket value."""

    def __call__(self, x: int) -> int: ...


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


@dataclass(frozen=True)
class SnapToBuckets:
    """Snap ``x`` to the nearest value in a fixed bucket set.

    ``round_up=False`` floors (largest bucket ``<= x``), ``round_up=True`` ceils
    (smallest bucket ``>= x``); both clamp to the bucket range. ``buckets`` is
    sorted and de-duplicated on construction.
    """

    buckets: tuple[int, ...]
    round_up: bool = False

    def __post_init__(self) -> None:
        # Normalize so round_to_nearest_bucket's ascending-order contract holds
        # and equal logical bucket sets compare/hash equal.
        object.__setattr__(self, "buckets", tuple(sorted(set(self.buckets))))
        assert self.buckets, "SnapToBuckets requires a non-empty bucket set"

    def __call__(self, x: int) -> int:
        return round_to_nearest_bucket(x, self.buckets, self.round_up)


@dataclass(frozen=True)
class RoundUpToMultiple:
    """Round ``x`` up to the next multiple of ``step``."""

    step: int

    def __post_init__(self) -> None:
        assert self.step >= 1, "RoundUpToMultiple requires step >= 1"

    def __call__(self, x: int) -> int:
        return ((x + self.step - 1) // self.step) * self.step


@dataclass(frozen=True)
class FloorToPowerOfTwo:
    """Largest power of two ``<= x`` (``last_positive_power_of_2``).

    With ``cap`` set, the result is clamped to ``<= cap``.
    """

    cap: int | None = None

    def __call__(self, x: int) -> int:
        val = last_positive_power_of_2(x)
        return val if self.cap is None else min(val, self.cap)


@dataclass(frozen=True)
class IdentityMap:
    """Map ``x`` to itself (every distinct size is its own bucket)."""

    def __call__(self, x: int) -> int:
        return x


@functools.lru_cache(maxsize=16384)
def make_bucket_mapper(
    buckets: tuple[int, ...], round_map: bool = False
) -> SnapToBuckets:
    """Create a :class:`SnapToBuckets` mapper for the given bucket set.

    Retained as a factory for backwards compatibility; the returned mapper is a
    frozen, hashable dataclass (no longer an opaque closure). Duplicates are
    removed and values sorted internally.

    Args:
        buckets: The set of allowed bucket values.
        round_map: ``False`` (default) rounds **down** (floor); ``True`` rounds
            **up** (ceil). Both clamp to the bucket range -- see
            :func:`round_to_nearest_bucket`.

    Examples::

        >>> mapper = make_bucket_mapper((100, 200, 500, 1000), round_map=False)
        >>> mapper(350)
        200
    """
    if len(buckets) == 0:
        raise ValueError("buckets must be non-empty")
    return SnapToBuckets(tuple(sorted(set(buckets))), round_up=round_map)
