"""
Core value-type abstractions for the AutoTuner.

These are small, dependency-free frozen dataclasses (and helpers) that describe
*where* a dimension lives and *what range* it spans. They live in their own leaf
module so the other autotuner submodules (``bucket_generators``,
``bucket_mappers``, ``infer_shapes``) can reference them at runtime without
importing back into ``autotuner`` -- avoiding import cycles.
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
