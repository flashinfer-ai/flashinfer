"""
Composable shape-inference rules for :class:`ConstraintSpec`.

A :class:`ConstraintSpec` ties a *dependent* dimension to other dimensions of
the operation. Its ``infer_shape`` receives the full opt-shapes tuple (every
input's shape, ``tuple[tuple[int, ...], ...]``) and returns the size to assign
to the constrained dimension during profiling (see
``AutoTuner._generate_optimization_profiles``).

Like the bucket generators / mappers, these rules are small frozen (hence
trivially hashable) dataclasses with a ``__call__(shapes) -> int`` so a
:class:`ConstraintSpec` -- and the :class:`TuningConfig` that holds it --
hashes structurally and stays a stable cache key.

``InferShape`` is a :class:`typing.Protocol` (as ``BucketMapper`` is): bespoke,
op-specific rules can live in their own modules without an import cycle, and any
frozen dataclass with the right ``__call__`` qualifies.
"""

from dataclasses import dataclass
from typing import Protocol

from flashinfer.autotuner.abstractions import DimensionCoordinates


class InferShape(Protocol):
    """Infers the size of a constrained dimension from the full opt-shapes."""

    def __call__(self, shapes: tuple[tuple[int, ...], ...]) -> int: ...


@dataclass(frozen=True)
class CopyDim:
    """Constrained dim equals the size of ``source`` (another dimension)."""

    source: DimensionCoordinates

    def __call__(self, shapes: tuple[tuple[int, ...], ...]) -> int:
        return shapes[self.source.input_idx][self.source.dim_idx]


@dataclass(frozen=True)
class PadUpDim:
    """Constrained dim equals ``source`` rounded **up** to a multiple."""

    source: DimensionCoordinates
    multiple: int

    def __post_init__(self) -> None:
        assert self.multiple >= 1, "PadUpDim requires multiple >= 1"

    def __call__(self, shapes: tuple[tuple[int, ...], ...]) -> int:
        v = shapes[self.source.input_idx][self.source.dim_idx]
        return ((v + self.multiple - 1) // self.multiple) * self.multiple
