"""Environment and explicit configuration for distribution-aware MoE."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from flashinfer.fused_moe.da_tuner import (
    DEFAULT_DA_DISTRIBUTIONS,
    DADistribution,
    validate_realization_capacity,
)


_FALSE_VALUES = {"", "0", "false", "no", "off", "none"}


def is_da_moe_enabled(*, default: bool = False) -> bool:
    """Return the DA switch, honoring a backend-specific default when unset."""
    return _environment_bool("FLASHINFER_DIST_AWARE_AUTOTUNE", default)


def _environment_bool(name: str, default: bool) -> bool:
    """Read one permissive historical FlashInfer boolean environment value."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_VALUES


def _environment_positive_int(name: str, default: int) -> int:
    """Read one positive integer environment control."""
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _environment_nonnegative_float(name: str, default: float) -> float:
    """Read one finite nonnegative floating-point environment control."""
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class DaMoeConfig:
    """Resolved backend-neutral configuration for DA tuning and replay."""

    # Whether the existing public MoE calls may publish DA capture plans.
    enabled: bool
    # Ordered training distributions used to realize selector exemplars.
    distributions: tuple[DADistribution, ...]
    # Number of independent Torch-RNG realizations generated per distribution.
    samples_per_distribution: int
    # Whether bounded FC1/FC2 coordinate search replaces exhaustive profiling.
    factorized_search: bool
    # Whether matched ordinary timings gate publication after candidate selection.
    baseline_guard_enabled: bool
    # Relative guard margin required in addition to matched NoDA latency.
    baseline_guard_margin: float
    # One switch-plan control charge applied once to each full invocation.
    control_overhead_us: float

    @classmethod
    def from_environment(cls, *, default_enabled: bool = False) -> DaMoeConfig:
        """Resolve the preserved environment contract and validate it atomically."""
        # Parse the ordered realization catalog first because its total cardinality is a hard
        # selector-storage constraint, not a tuning-time fallback condition.
        distribution_text = os.getenv(
            "FLASHINFER_DA_DISTRIBUTIONS", ",".join(DEFAULT_DA_DISTRIBUTIONS)
        )
        distributions = tuple(
            DADistribution.parse(item)
            for item in distribution_text.split(",")
            if item.strip()
        )
        samples = _environment_positive_int("FLASHINFER_DA_DISTRIBUTION_SAMPLES", 1)
        validate_realization_capacity(distributions, samples)
        margin = _environment_nonnegative_float(
            "FLASHINFER_DA_BASELINE_GUARD_MARGIN", 0.0
        )
        if margin >= 1.0:
            raise ValueError("FLASHINFER_DA_BASELINE_GUARD_MARGIN must be less than 1")

        # Construct only after every dependent value is validated so callers never observe a
        # partially resolved environment configuration.
        return cls(
            enabled=is_da_moe_enabled(default=default_enabled),
            distributions=distributions,
            samples_per_distribution=samples,
            factorized_search=_environment_bool(
                "FLASHINFER_DA_FACTORIZED_AUTOTUNE", True
            ),
            baseline_guard_enabled=_environment_bool(
                "FLASHINFER_DA_BASELINE_GUARD", True
            ),
            baseline_guard_margin=margin,
            control_overhead_us=_environment_nonnegative_float(
                "FLASHINFER_DA_CONTROL_OVERHEAD_US", 12.0
            ),
        )

    def cache_identity(self) -> dict[str, object]:
        """Return deterministic JSON-compatible search and profiling identity."""
        return {
            "distributions": [distribution.name for distribution in self.distributions],
            "samples_per_distribution": self.samples_per_distribution,
            "factorized_search": self.factorized_search,
            "baseline_guard_enabled": self.baseline_guard_enabled,
            "baseline_guard_margin": self.baseline_guard_margin,
            "control_overhead_us": self.control_overhead_us,
        }


# Compatibility alias for existing TRTLLM DA callers.
TrtllmDaConfig = DaMoeConfig


def is_trtllm_da_enabled() -> bool:
    """Return the backend-neutral DA switch through the historical TRTLLM name."""
    return is_da_moe_enabled()


def get_enabled_da_moe_config(*, default_enabled: bool = False) -> DaMoeConfig | None:
    """Resolve DA configuration using the calling backend's default policy."""
    if not is_da_moe_enabled(default=default_enabled):
        return None
    return DaMoeConfig.from_environment(default_enabled=default_enabled)
