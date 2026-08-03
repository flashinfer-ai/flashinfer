"""Pure deterministic candidate ranking for Batch MLA ``backend='auto'``."""

from dataclasses import dataclass

from typing import Optional


SM80_PREFERRED_BACKENDS = (
    "fa2",
    "fa3",
    "cutlass",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "xqa",
)

SM90_PREFERRED_BACKENDS = (
    "fa3",
    "fa2",
    "cutlass",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "xqa",
)

SM100_PREFERRED_BACKENDS = (
    "trtllm-gen",
    "fa2",
    "fa3",
    "cutlass",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "xqa",
)

SM120_PREFERRED_BACKENDS = (
    "xqa",
    "fa2",
    "fa3",
    "cutlass",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
)


_ARCHITECTURE_PREFERRED_BACKENDS = {
    None: SM80_PREFERRED_BACKENDS,
    (8, 0): SM80_PREFERRED_BACKENDS,
    (8, 9): SM80_PREFERRED_BACKENDS,
    (9, 0): SM90_PREFERRED_BACKENDS,
    (10, 0): SM100_PREFERRED_BACKENDS,
    (10, 3): SM100_PREFERRED_BACKENDS,
    (12, 0): SM120_PREFERRED_BACKENDS,
    (12, 1): SM120_PREFERRED_BACKENDS,
}


@dataclass(frozen=True)
class MLAAutoSelectionTrace:
    """Immutable result of one automatic MLA planning attempt."""

    candidates: tuple[str, ...]
    rejections: tuple[tuple[str, str], ...]
    resolved_backend: str


def rank_auto_backend_candidates(
    compute_capability: Optional[tuple[int, int]],
) -> tuple[str, ...]:
    """Return the complete architecture-preferred automatic planning order."""
    return _ARCHITECTURE_PREFERRED_BACKENDS.get(
        compute_capability, _ARCHITECTURE_PREFERRED_BACKENDS[None]
    )
