"""Immutable configuration for one distribution-aware MoE transaction."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .da_utils import DADistributionSpec, get_da_distribution_specs


def _env_bool(env: str, default: bool) -> bool:
    """Read a conventional boolean environment variable."""

    value = os.environ.get(env, str(int(default)))
    return value.lower() not in ("", "0", "false")


def _env_positive_int(env: str, default: int) -> int:
    """Read a required positive integer environment variable."""

    value = os.environ.get(env, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{env} must be a positive integer, got {value!r}") from error
    if parsed <= 0:
        raise ValueError(f"{env} must be a positive integer, got {value!r}")
    return parsed


def _env_optional_positive_int(env: str) -> Optional[int]:
    """Read an optional positive integer, treating common disable values as None."""

    value = os.environ.get(env, "")
    if value.lower() in ("", "0", "none", "all", "unlimited", "false"):
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{env} must be a positive integer, got {value!r}") from error
    if parsed <= 0:
        raise ValueError(f"{env} must be a positive integer, got {value!r}")
    return parsed


def _env_offsets() -> tuple[int, ...]:
    """Read comma- or whitespace-separated non-negative exemplar offsets."""

    value = os.environ.get("FLASHINFER_DA_KNN_OFFSETS", "")
    if not value:
        return ()
    try:
        offsets = tuple(int(item) for item in value.replace(",", " ").split())
    except ValueError as error:
        raise ValueError("FLASHINFER_DA_KNN_OFFSETS must contain integers") from error
    if any(offset < 0 for offset in offsets):
        raise ValueError("FLASHINFER_DA_KNN_OFFSETS must be non-negative")
    return offsets


def _env_tie_epsilon() -> float:
    """Read the non-negative tie threshold used by exemplar selection."""

    value = os.environ.get("FLASHINFER_DA_KNN_TIE_EPS", "0.05")
    try:
        epsilon = float(value)
    except ValueError as error:
        raise ValueError("FLASHINFER_DA_KNN_TIE_EPS must be non-negative") from error
    if epsilon < 0:
        raise ValueError("FLASHINFER_DA_KNN_TIE_EPS must be non-negative")
    return epsilon


@dataclass(frozen=True)
class DAConfig:
    """DA policy snapshot, with each omitted field read from its legacy envvar."""

    # Enables value-aware distribution autotuning and capture dispatch.
    enabled: bool = field(
        default_factory=lambda: _env_bool("FLASHINFER_DIST_AWARE_AUTOTUNE", False)
    )
    # Enables combined-index M+N tactic search for DA profiling.
    factorized_autotune: bool = field(
        default_factory=lambda: _env_bool("FLASHINFER_DA_FACTORIZED_AUTOTUNE", True)
    )
    # Synthetic routing distributions used for value-aware autotuning.
    distributions: tuple[DADistributionSpec, ...] = field(
        default_factory=lambda: get_da_distribution_specs(
            os.environ.get("FLASHINFER_DA_DISTRIBUTIONS", "")
        )
    )
    # Repetitions used for each non-single synthetic distribution.
    distribution_sample_count: int = field(
        default_factory=lambda: _env_positive_int(
            "FLASHINFER_DA_DISTRIBUTION_SAMPLES", 10
        )
    )
    # Optional path to a serialized DAKNNv2 selector bundle.
    bundle_path: str = field(
        default_factory=lambda: os.environ.get("FLASHINFER_DA_KNN_BUNDLE", "")
    )
    # Warmup runs for live kNN body profiling.
    auto_warmup: int = field(
        default_factory=lambda: _env_positive_int("FLASHINFER_DA_KNN_AUTO_WARMUP", 3)
    )
    # Timed iterations for live kNN body profiling.
    auto_iters: int = field(
        default_factory=lambda: _env_positive_int("FLASHINFER_DA_KNN_AUTO_ITERS", 10)
    )
    # Merges exemplar rows that choose the same body tactic.
    merge_same_tactic_exemplars: bool = field(
        default_factory=lambda: _env_bool("FLASHINFER_DA_KNN_MERGE", False)
    )
    # Optional local-expert offsets represented in bundle exemplars.
    exemplar_offsets: tuple[int, ...] = field(default_factory=_env_offsets)
    # Enables direct live profiling when autotuner cache data is unavailable.
    live_profile: bool = field(
        default_factory=lambda: _env_bool("FLASHINFER_DA_KNN_LIVE_PROFILE", False)
    )
    # Latency tolerance for treating candidate body tactics as tied.
    tie_epsilon: float = field(default_factory=_env_tie_epsilon)
    # Optional diagnostic cap on FP4 kNN tile sizes.
    max_knn_tile: Optional[int] = field(
        default_factory=lambda: _env_optional_positive_int("FLASHINFER_DA_KNN_MAX_TILE")
    )
    # Selects a DA body using routing counts rather than synthesized routing.
    select_from_routing_counts: bool = field(
        default_factory=lambda: _env_bool(
            "FLASHINFER_DA_KNN_SELECT_FROM_ROUTING_COUNTS", False
        )
    )
    # Emits human-readable DA diagnostics.
    verbose: bool = field(
        default_factory=lambda: _env_bool("FLASHINFER_DA_VERBOSE", False)
    )
    # Optional worker-local path for DA diagnostics.
    debug_file: Optional[str] = field(
        default_factory=lambda: os.environ.get("FLASHINFER_DA_DEBUG_FILE", "") or None
    )
    # Enables CUDA-profiler instrumentation around the autotune phase.
    autotune_cuda_profiler: bool = field(
        default_factory=lambda: _env_bool("FLASHINFER_DA_AUTOTUNE_CUDA_PROFILER", False)
    )

    @property
    def profile_signature(self) -> tuple[str, ...]:
        """Stable distribution labels embedded in profile and bundle metadata."""

        return tuple(str(distribution[0]) for distribution in self.distributions)
