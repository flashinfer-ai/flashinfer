"""Host-side distribution-aware MoE realization, search, and plan compilation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

import numpy as np
import torch

from flashinfer.fused_moe.tactic_search import FactorizedTactic
from flashinfer.fused_moe.da_moe import (
    DA_MAX_BODIES,
    DA_MAX_EXEMPLARS,
    DABody,
    DAMoEDispatcher,
    DAPlan,
    DAPlanMode,
    _local_load_spectrum,
)


@dataclass(frozen=True)
class FullWorkload:
    """Profile every token-expert slot as work owned by the local rank."""

    def local_assignments(
        self,
        capacity_tokens: int,
        top_k: int,
        num_experts: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
    ) -> int:
        """Return the full token-expert capacity."""
        _validate_capacity_and_top_k(capacity_tokens, top_k)
        return capacity_tokens * top_k


@dataclass(frozen=True)
class BalancedEPWorkload:
    """Hint that a full input buffer is balanced across equal EP shards.

    When ``ep_size`` and ``ep_rank`` are omitted, both values are inferred
    from the MoE operation's global and local expert geometry. Explicit values
    are cross-checked against that geometry. The default two-times multiplier
    was selected empirically from PrimsTS DA MoE measurements across EP sizes;
    callers can request exact balanced occupancy with a multiplier of one.
    """

    ep_size: int | None = None
    ep_rank: int | None = None
    require_equal: bool = True
    assignment_multiplier: int = 2

    def __post_init__(self) -> None:
        if (self.ep_size is None) != (self.ep_rank is None):
            raise ValueError("ep_size and ep_rank must be provided together")
        if self.ep_size is not None and self.ep_size <= 0:
            raise ValueError(f"ep_size must be positive, got {self.ep_size}")
        if self.ep_size is not None:
            assert self.ep_rank is not None
        if self.ep_size is not None and not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(
                f"ep_rank must be in [0, {self.ep_size}), got {self.ep_rank}"
            )
        if self.assignment_multiplier <= 0:
            raise ValueError(
                "assignment_multiplier must be positive, got "
                f"{self.assignment_multiplier}"
            )

    def local_assignments(
        self,
        capacity_tokens: int,
        top_k: int,
        num_experts: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
    ) -> int:
        """Return this rank's synthetic work, capped at the input capacity."""
        _validate_capacity_and_top_k(capacity_tokens, top_k)
        if num_experts <= 0 or num_local_experts <= 0:
            raise ValueError("global and local expert counts must be positive")
        inferred_ep_size, shard_remainder = divmod(num_experts, num_local_experts)
        if shard_remainder:
            raise ValueError(
                "BalancedEPWorkload requires equal expert shards: "
                f"{num_experts} experts cannot be divided into shards of "
                f"{num_local_experts}"
            )
        if (
            local_expert_offset < 0
            or local_expert_offset + num_local_experts > num_experts
        ):
            raise ValueError(
                "BalancedEPWorkload requires the local expert range to lie within "
                "the global expert domain"
            )
        ep_size = inferred_ep_size if self.ep_size is None else self.ep_size
        if ep_size != inferred_ep_size:
            raise ValueError(
                "explicit EP topology does not match the MoE expert geometry: "
                f"got ep_size={ep_size}; expected ep_size={inferred_ep_size}"
            )

        quotient, remainder = divmod(capacity_tokens * top_k, ep_size)
        if self.require_equal and remainder:
            raise ValueError(
                "capacity_tokens * top_k must be divisible by ep_size when "
                f"require_equal=True, got {capacity_tokens} * {top_k} % {ep_size} "
                f"= {remainder}"
            )
        inferred_ep_rank, offset_remainder = divmod(
            local_expert_offset, num_local_experts
        )
        if self.ep_rank is not None and not offset_remainder:
            if self.ep_rank != inferred_ep_rank:
                raise ValueError(
                    "explicit EP rank does not match the aligned local expert shard: "
                    f"got ep_rank={self.ep_rank}; expected ep_rank={inferred_ep_rank}"
                )
        balanced = quotient
        if remainder:
            if self.ep_rank is not None:
                ep_rank = self.ep_rank
            elif not offset_remainder:
                ep_rank = inferred_ep_rank
            else:
                raise ValueError(
                    "an unaligned local expert range requires explicit ep_rank when "
                    "balanced work has a remainder"
                )
            balanced += int(ep_rank < remainder)
        return min(capacity_tokens * top_k, balanced * self.assignment_multiplier)


MoeWorkload: TypeAlias = FullWorkload | BalancedEPWorkload
_current_workload: ContextVar[MoeWorkload | None] = ContextVar(
    "moe_workload", default=None
)


@contextmanager
def moe_workload(workload: MoeWorkload) -> Iterator[None]:
    """Override DA profiling policy for internal experiments and tests.

    Nested scopes restore the enclosing policy, including on exceptions.
    This does not affect replay realization defaults or non-DA operators.
    """
    if not isinstance(workload, (FullWorkload, BalancedEPWorkload)):
        raise TypeError(
            "workload must be FullWorkload or BalancedEPWorkload, got "
            f"{type(workload).__name__}"
        )
    token = _current_workload.set(workload)
    try:
        yield
    finally:
        _current_workload.reset(token)


def get_workload() -> MoeWorkload:
    """Return the active MoE policy, defaulting to two-times balanced EP."""
    workload = _current_workload.get()
    return workload if workload is not None else BalancedEPWorkload()


def _validate_capacity_and_top_k(capacity_tokens: int, top_k: int) -> None:
    if capacity_tokens < 0:
        raise ValueError(f"capacity_tokens must be nonnegative, got {capacity_tokens}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")


DEFAULT_DA_DISTRIBUTIONS = (
    "ddist:1.1",
    "ddist:1.3",
    "ddist:1.5",
    "ddist:1.7",
    "ddist:2",
    "ddist:2.5",
    "ddist:4",
)


@dataclass(frozen=True)
class DADistribution:
    """One expert-popularity distribution requested for DA profiling."""

    # Canonical cache and diagnostic spelling for this distribution.
    name: str
    # Effective-expert concentration factor, or None for exact uniform.
    factor: float | None

    @classmethod
    def parse(cls, value: str) -> DADistribution:
        """Parse ``uniform`` and the historical positive ``ddist`` spellings."""
        normalized = value.strip().lower()
        if normalized == "uniform":
            return cls(name="uniform", factor=None)
        if normalized.startswith("ddist_"):
            normalized = f"ddist:{normalized.removeprefix('ddist_')}"
        if normalized and ":" not in normalized:
            normalized = f"ddist:{normalized}"
        prefix, separator, factor_text = normalized.partition(":")
        if prefix != "ddist" or not separator:
            raise ValueError(f"Unsupported DA distribution {value!r}")
        try:
            factor = float(factor_text)
        except ValueError as error:
            raise ValueError(f"Malformed DA distribution {value!r}") from error
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("A ddist factor must be positive and finite")
        return cls(name=f"ddist:{factor:g}", factor=factor)


@dataclass(frozen=True)
class RoutingRealizationKey:
    """Complete identity of one cached routed-input realization."""

    # CUDA device that owns the generated tensors and Torch RNG draws.
    device: torch.device
    # Exact token count represented by the value profile.
    num_tokens: int
    # Canonical distribution spelling.
    distribution: str
    # Realization ordinal within the declared distribution.
    sample_index: int
    # First global expert ID owned by the local rank.
    local_expert_offset: int
    # Total number of experts in the global routing domain.
    num_experts: int
    # Number of experts owned by the local rank.
    num_local_experts: int
    # Number of distinct experts selected for every token.
    top_k: int
    # Stable public-routing-rule identity supplied by the dtype adapter.
    routing_rule_fingerprint: str
    # Scalar applied after row-wise routing-weight normalization.
    routed_scaling_factor: float

    # DA profiling may override occupancy; replay realizations use exact balanced EP.
    num_local_assignments_hint: int | None = None

    @property
    def num_local_assignments(self) -> int:
        if self.num_local_assignments_hint is not None:
            return self.num_local_assignments_hint
        return BalancedEPWorkload(assignment_multiplier=1).local_assignments(
            self.num_tokens,
            self.top_k,
            self.num_experts,
            self.num_local_experts,
            self.local_expert_offset,
        )


@dataclass(frozen=True)
class RoutingRealization:
    """Canonical mutable routing pair shared by every measured tactic."""

    # Cache identity that generated this exact tensor pair.
    key: RoutingRealizationKey
    # Global int32 expert IDs with distinct entries in every token row.
    expert_ids: torch.Tensor
    # Positive row-normalized BF16 routing weights after configured scaling.
    routing_weights: torch.Tensor


class RoutingRealizationFactory:
    """Generate and cache DA expert IDs and BF16 weights before tactic timing."""

    # Fraction of uniform probability mixed into every sampled Dirichlet profile.
    _UNIFORM_FLOOR = 0.1
    # Fixed solve depth that makes the effective-expert calibration deterministic.
    _BISECTION_STEPS = 80

    def __init__(self) -> None:
        """Create an empty process-local realization cache."""
        # Realizations already generated under the process/device Torch RNG.
        self._cache: dict[RoutingRealizationKey, RoutingRealization] = {}

    def get_or_create(self, key: RoutingRealizationKey) -> RoutingRealization:
        """Return one cached realization or generate it exactly once."""
        # Cache by the complete routing identity so every compared tactic observes exactly the
        # same expert IDs and BF16 weights.
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if key.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if key.sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        if key.num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive")
        if key.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 0 <= key.local_expert_offset < key.num_experts:
            raise ValueError("local_expert_offset must be in the global expert domain")
        if key.local_expert_offset + key.num_local_experts > key.num_experts:
            raise ValueError("local expert interval exceeds the global expert domain")
        if not 0 < key.top_k <= key.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if (
            not math.isfinite(key.routed_scaling_factor)
            or key.routed_scaling_factor <= 0
        ):
            raise ValueError("routed_scaling_factor must be positive and finite")

        num_nonlocal_experts = key.num_experts - key.num_local_experts
        min_local_per_row = max(0, key.top_k - num_nonlocal_experts)
        max_local_per_row = min(key.top_k, key.num_local_experts)
        min_local_assignments = key.num_tokens * min_local_per_row
        max_local_assignments = key.num_tokens * max_local_per_row
        if not (
            min_local_assignments <= key.num_local_assignments <= max_local_assignments
        ):
            raise ValueError(
                "num_local_assignments_hint must be feasible for distinct per-row "
                f"top-k IDs, got {key.num_local_assignments}; expected "
                f"[{min_local_assignments}, {max_local_assignments}]"
            )

        # Generate both mutable tensors outside full-op timing; measured launches only stage the
        # already-materialized realization into reusable profiling storage.
        distribution = DADistribution.parse(key.distribution)
        probabilities = self._expert_probabilities(
            key.num_local_experts, distribution
        ).to(device=key.device, dtype=torch.float32)
        if key.num_local_assignments == key.num_tokens * key.top_k:
            # Preserve the historical full-local realization.
            expanded = probabilities.expand(key.num_tokens, -1)
            local_ids = torch.multinomial(expanded, key.top_k, replacement=False)
            expert_ids = (local_ids + key.local_expert_offset).to(torch.int32)
        else:
            expert_ids = self._generate_hint_local_workload(
                key,
                probabilities,
                min_local_per_row=min_local_per_row,
                max_local_per_row=max_local_per_row,
            )

        fork_devices = []
        if key.device.type == "cuda":
            fork_devices.append(
                key.device.index
                if key.device.index is not None
                else torch.cuda.current_device()
            )
        with torch.random.fork_rng(devices=fork_devices):
            positive = torch.rand(
                key.num_tokens,
                key.top_k,
                dtype=torch.float32,
                device=key.device,
            ).clamp_min_(1e-6)
        normalized = positive / positive.sum(dim=1, keepdim=True)
        routing_weights = (normalized * key.routed_scaling_factor).to(torch.bfloat16)

        realization = RoutingRealization(
            key=key,
            expert_ids=expert_ids,
            routing_weights=routing_weights,
        )
        self._cache[key] = realization
        return realization

    @staticmethod
    def _generate_hint_local_workload(
        key: RoutingRealizationKey,
        local_probabilities: torch.Tensor,
        *,
        min_local_per_row: int,
        max_local_per_row: int,
    ) -> torch.Tensor:
        """Generate distinct global IDs matching the profiling hint."""
        remaining = key.num_local_assignments - key.num_tokens * min_local_per_row
        extra_per_row, extra_rows = divmod(remaining, key.num_tokens)
        if min_local_per_row + extra_per_row > max_local_per_row:
            raise ValueError(
                "num_local_assignments_hint exceeds per-row local capacity"
            )

        local_counts = torch.full(
            (key.num_tokens,),
            min_local_per_row + extra_per_row,
            dtype=torch.int64,
            device=key.device,
        )
        if extra_rows:
            rotation = (key.sample_index + key.local_expert_offset) % key.num_tokens
            remainder_rows = (
                torch.arange(extra_rows, device=key.device) + rotation
            ) % key.num_tokens
            local_counts[remainder_rows] += 1

        nonlocal_ids = torch.cat(
            (
                torch.arange(
                    0,
                    key.local_expert_offset,
                    dtype=torch.int64,
                    device=key.device,
                ),
                torch.arange(
                    key.local_expert_offset + key.num_local_experts,
                    key.num_experts,
                    dtype=torch.int64,
                    device=key.device,
                ),
            )
        )
        result = torch.empty(
            (key.num_tokens, key.top_k), dtype=torch.int64, device=key.device
        )
        for local_count in range(min_local_per_row, max_local_per_row + 1):
            rows = torch.where(local_counts == local_count)[0]
            if rows.numel() == 0:
                continue
            nonlocal_count = key.top_k - local_count
            parts: list[torch.Tensor] = []
            if local_count:
                local_choices = torch.multinomial(
                    local_probabilities.expand(rows.numel(), -1),
                    local_count,
                    replacement=False,
                )
                parts.append(local_choices + key.local_expert_offset)
            if nonlocal_count:
                uniform_nonlocal = torch.ones(
                    nonlocal_ids.numel(),
                    dtype=torch.float32,
                    device=key.device,
                ).expand(rows.numel(), -1)
                nonlocal_choices = torch.multinomial(
                    uniform_nonlocal, nonlocal_count, replacement=False
                )
                parts.append(nonlocal_ids[nonlocal_choices])
            row_ids = torch.cat(parts, dim=1)
            if row_ids.shape[1] > 1:
                permutation = torch.rand(
                    row_ids.shape, dtype=torch.float32, device=key.device
                ).argsort(dim=1)
                row_ids = row_ids.gather(1, permutation)
            result[rows] = row_ids

        local_mask = (result >= key.local_expert_offset) & (
            result < key.local_expert_offset + key.num_local_experts
        )
        if int(local_mask.sum().item()) != key.num_local_assignments:
            raise RuntimeError("generated routing does not match local workload target")
        return result.to(torch.int32)

    @classmethod
    def _expert_probabilities(
        cls, num_local_experts: int, distribution: DADistribution
    ) -> torch.Tensor:
        """Build the deterministic seed-42 popularity vector for a profile."""
        # Uniform is an exact fast path and bypasses Dirichlet calibration entirely.
        if distribution.factor is None:
            return torch.full(
                (num_local_experts,),
                1.0 / num_local_experts,
                dtype=torch.float64,
            )

        # Calibrate symmetric Dirichlet alpha to the requested effective-expert count, then use
        # independent deterministic generators for ranked loads and expert placement.
        target = min(
            max(num_local_experts / distribution.factor, 1.0),
            float(num_local_experts),
        )
        alpha = cls._solve_symmetric_alpha(num_local_experts, target)
        template_rng = np.random.default_rng(42)
        probabilities = template_rng.dirichlet(
            np.full(num_local_experts, alpha, dtype=np.float64)
        )
        probabilities = np.clip(probabilities, np.finfo(np.float64).tiny, None)
        probabilities /= probabilities.sum()
        probabilities = (
            1.0 - cls._UNIFORM_FLOOR
        ) * probabilities + cls._UNIFORM_FLOOR / num_local_experts
        probabilities /= probabilities.sum()
        ranked = np.sort(probabilities)[::-1]

        permutation_rng = np.random.default_rng(42)
        expert_order = permutation_rng.permutation(num_local_experts)
        permuted = np.empty_like(ranked)
        permuted[expert_order] = ranked
        return torch.from_numpy(permuted.copy())

    @classmethod
    def _solve_symmetric_alpha(
        cls, num_local_experts: int, target_effective_experts: float
    ) -> float:
        """Solve symmetric Dirichlet alpha by the specified 80-step bisection."""
        low = 1e-6
        high = 1e6
        for _ in range(cls._BISECTION_STEPS):
            middle = (low + high) / 2.0
            effective = cls._expected_effective_experts(num_local_experts, middle)
            if effective < target_effective_experts:
                low = middle
            else:
                high = middle
        return (low + high) / 2.0

    @classmethod
    def _expected_effective_experts(cls, num_local_experts: int, alpha: float) -> float:
        """Return inverse-Simpson support after the ten-percent uniform floor."""
        epsilon = cls._UNIFORM_FLOOR
        concentration = (alpha + 1.0) / (num_local_experts * alpha + 1.0)
        squared_mass = (1.0 - epsilon) ** 2 * concentration + (
            2.0 * epsilon - epsilon**2
        ) / num_local_experts
        return 1.0 / squared_mass


class FullOpMeasurementCache:
    """Retain finite full-operation timings under exact measurement identities."""

    def __init__(self) -> None:
        """Create an empty exact-measurement cache."""
        # Best finite timing observed for each exact caller-defined key.
        self._timings: dict[tuple[Any, ...], float] = {}

    @property
    def count(self) -> int:
        """Return the number of distinct full-operation timings retained."""
        return len(self._timings)

    def measure(
        self,
        key: tuple[Any, ...],
        measure: Callable[[], float],
    ) -> float:
        """Reuse a finite timing or execute one full-op measurement."""
        cached = self._timings.get(key)
        if cached is not None:
            return cached
        observed = float(measure())
        if not math.isfinite(observed):
            raise RuntimeError(f"Non-finite full MoE timing for {key!r}")
        self._timings[key] = observed
        return observed

    def measure_counterbalanced_pair(
        self,
        first_key: tuple[Any, ...],
        first_measure: Callable[[], float],
        second_key: tuple[Any, ...],
        second_measure: Callable[[], float],
    ) -> tuple[float, float]:
        """Measure two full operations in ABBA order and cache their means."""
        if first_key == second_key:
            raise ValueError("Counterbalanced measurements require distinct keys")
        first_cached = self._timings.get(first_key)
        second_cached = self._timings.get(second_key)
        if first_cached is not None and second_cached is not None:
            return first_cached, second_cached
        if first_cached is not None or second_cached is not None:
            raise RuntimeError(
                "Counterbalanced measurement cache is partially populated"
            )

        first_before = float(first_measure())
        second_observations = (float(second_measure()), float(second_measure()))
        first_observations = (first_before, float(first_measure()))
        means: dict[tuple[Any, ...], float] = {}
        for key, observations in (
            (first_key, first_observations),
            (second_key, second_observations),
        ):
            if not all(math.isfinite(value) for value in observations):
                raise RuntimeError(f"Non-finite full MoE timing for {key!r}")
            means[key] = sum(observations) / len(observations)
        self._timings.update(means)
        return self._timings[first_key], self._timings[second_key]


def factorized_tactic_to_body(tactic: FactorizedTactic) -> DABody:
    """Decode one backend-complete tactic into the native DA body identity."""
    identity = tactic.tactic
    if (
        not isinstance(identity, tuple)
        or len(identity) != 2
        or int(identity[0]) != tactic.tile_n
        or int(identity[1]) < 0
    ):
        raise RuntimeError(
            "A DA MoE body requires a concrete (tile_n, config_index) identity"
        )
    return DABody(tactic=int(identity[1]), tile_n=tactic.tile_n)


@dataclass(frozen=True)
class DAProfileSelection:
    """One selector exemplar and its decisively measured candidate assignment."""

    # Stable routing realization identity used by cache and diagnostics.
    realization_key: RoutingRealizationKey
    # Device expert IDs from which the selector spectrum is uploaded.
    expert_ids: torch.Tensor
    # Complete selected tactic with tile and opaque factorization.
    selected_tactic: FactorizedTactic
    # Aggregate candidate time divided by exactly requested iterations.
    candidate_latency_ms: float
    # Matched ordinary baseline latency, or None when the guard is disabled.
    baseline_latency_ms: float | None


@dataclass(frozen=True)
class DACompiledPlan:
    """Immutable host result consumed by cache publication and runtime staging."""

    # Candidate policy before the post-selection guard is evaluated.
    candidate_policy: DAPlanMode
    # Final admitted capture policy.
    policy: DAPlanMode
    # Every unique selector exemplar in original upload order.
    selections: tuple[DAProfileSelection, ...]
    # Deduplicated complete candidate bodies in stable first-seen order.
    bodies: tuple[FactorizedTactic, ...]
    # Mapping from every exemplar to its deduplicated candidate body.
    exemplar_body_indices: tuple[int, ...]
    # Exact ordinary monolithic tactic used by guarded fallback.
    baseline_tactic: Any
    # Preferred host-dispatch tactic when CUDA Graph replay is unavailable.
    eager_tactic: FactorizedTactic | None
    # Distribution whose measured tactic was selected for host dispatch.
    eager_distribution: str | None
    # Compact admission or fail-closed diagnostic reason.
    guard_reason: str


class DAPlanCompiler:
    """Prune selected value profiles and apply the pure post-selection guard."""

    def __init__(
        self,
        *,
        num_experts: int,
        local_expert_offset: int = 0,
        num_local_experts: int | None = None,
        guard_enabled: bool = True,
        margin: float = 0.0,
        control_overhead_us: float = 12.0,
    ) -> None:
        """Configure guard policy without changing candidate construction."""
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if num_local_experts is None:
            num_local_experts = num_experts
        if local_expert_offset < 0 or not (
            0 < num_local_experts <= num_experts - local_expert_offset
        ):
            raise ValueError("the local expert shard must fit within num_experts")
        if not 0.0 <= margin < 1.0:
            raise ValueError("guard margin must be in [0, 1)")
        if not math.isfinite(control_overhead_us) or control_overhead_us < 0:
            raise ValueError("control_overhead_us must be finite and nonnegative")
        # Global expert width consumed by the runtime selector and uploaded spectra.
        self._num_experts = num_experts
        # Runtime and exemplar histograms count only assignments owned by this rank.
        self._local_expert_offset = local_expert_offset
        self._num_local_experts = num_local_experts
        # Whether matched ordinary measurements gate final admission.
        self._guard_enabled = guard_enabled
        # Required relative win applied to the matched baseline.
        self._margin = margin
        # One switch-only control charge converted from microseconds to ms.
        self._control_overhead_ms = control_overhead_us / 1000.0

    def compile(
        self,
        selections: Sequence[DAProfileSelection],
        baseline_tactic: Any,
        *,
        eager_selections: Sequence[DAProfileSelection] | None = None,
    ) -> DACompiledPlan:
        """Deduplicate bodies and publish singleton, switch, or guarded NoDA."""
        # Validate the complete selector catalog before graph-body reduction so classifier
        # boundaries remain independent of body deduplication.
        if not selections:
            raise ValueError("A DA plan requires at least one profile selection")
        if len(selections) > DA_MAX_EXEMPLARS:
            raise ValueError(
                f"DA supports at most {DA_MAX_EXEMPLARS} realized exemplars"
            )
        if baseline_tactic is None:
            raise ValueError("A DA plan requires an ordinary baseline tactic")

        # Preserve unique exemplars in upload order while deduplicating exact complete tactics
        # into stable first-seen conditional bodies.
        exemplar_fingerprints: set[bytes] = set()
        bodies: list[FactorizedTactic] = []
        body_indices: list[int] = []
        for selection in selections:
            fingerprint = self._selector_spectrum_fingerprint(selection)
            if fingerprint in exemplar_fingerprints:
                raise ValueError("DA selector exemplars must remain unique")
            exemplar_fingerprints.add(fingerprint)
            if selection.selected_tactic not in bodies:
                bodies.append(selection.selected_tactic)
            body_indices.append(bodies.index(selection.selected_tactic))
        if len(bodies) > DA_MAX_BODIES:
            raise ValueError(f"DA supports at most {DA_MAX_BODIES} unique bodies")

        # Apply the baseline guard only after candidate construction, then independently retain
        # the graph-free distribution-aware tactic.
        candidate_policy = (
            DAPlanMode.DA_SINGLE_BODY if len(bodies) == 1 else DAPlanMode.DA_SWITCH
        )
        admitted, reason = self._guard_admits(candidate_policy, selections)
        policy = candidate_policy if admitted else DAPlanMode.DA_FALLBACK
        eager_tactic, eager_distribution = self._select_eager_tactic(
            selections if eager_selections is None else eager_selections
        )
        return DACompiledPlan(
            candidate_policy=candidate_policy,
            policy=policy,
            selections=tuple(selections),
            bodies=tuple(bodies),
            exemplar_body_indices=tuple(body_indices),
            baseline_tactic=baseline_tactic,
            eager_tactic=eager_tactic,
            eager_distribution=eager_distribution,
            guard_reason=reason,
        )

    @staticmethod
    def _select_eager_tactic(
        selections: Sequence[DAProfileSelection],
    ) -> tuple[FactorizedTactic | None, str | None]:
        """Prefer the measured ddist:1.1 tactic, then the uniform tactic."""
        for preferred_distribution in ("ddist:1.1", "uniform"):
            for selection in selections:
                if selection.realization_key.distribution == preferred_distribution:
                    return selection.selected_tactic, preferred_distribution
        return None, None

    def prefer_control_aware_singleton(
        self,
        selections: Sequence[DAProfileSelection],
        candidate_latencies: Mapping[tuple[RoutingRealizationKey, Any], float],
    ) -> tuple[DAProfileSelection, ...]:
        """Collapse a guarded switch when one measured body absorbs its charge."""
        # This prune applies only to guarded multi-body candidates and never merges selector
        # exemplar rows.
        retained = tuple(selections)
        if not self._guard_enabled:
            return retained
        bodies = tuple(dict.fromkeys(item.selected_tactic for item in retained))
        if len(bodies) < 2:
            return retained

        # A body is eligible only when its regret on every exemplar is no larger than the switch
        # control charge eliminated by singleton capture.
        eligible: list[tuple[float, float, str, FactorizedTactic]] = []
        for body in bodies:
            latencies = tuple(
                float(candidate_latencies[(selection.realization_key, body.tactic)])
                for selection in retained
            )
            regrets = tuple(
                latency - selection.candidate_latency_ms
                for latency, selection in zip(latencies, retained, strict=True)
            )
            if all(regret <= self._control_overhead_ms for regret in regrets):
                eligible.append((max(regrets), sum(latencies), repr(body.tactic), body))
        if not eligible:
            return retained

        # Resolve eligible bodies by worst regret, aggregate latency, then stable tactic spelling.
        singleton = min(eligible)[3]
        return tuple(
            replace(
                selection,
                selected_tactic=singleton,
                candidate_latency_ms=float(
                    candidate_latencies[(selection.realization_key, singleton.tactic)]
                ),
            )
            for selection in retained
        )

    def _selector_spectrum_fingerprint(self, selection: DAProfileSelection) -> bytes:
        """Fingerprint the local-only load spectrum consumed by kNN."""
        loads = _local_load_spectrum(
            selection.expert_ids.detach().to(device="cpu", dtype=torch.int64),
            num_experts=self._num_experts,
            local_expert_offset=self._local_expert_offset,
            num_local_experts=self._num_local_experts,
            normalize=False,
        )
        return loads.contiguous().view(torch.uint8).numpy().tobytes()

    def _guard_admits(
        self,
        policy: DAPlanMode,
        selections: Sequence[DAProfileSelection],
    ) -> tuple[bool, str]:
        """Evaluate the confirmed matched-evidence guard without re-profiling."""
        # Missing or non-finite matched evidence fails closed before policy arithmetic.
        if not self._guard_enabled:
            return True, "guard_disabled"
        for selection in selections:
            if (
                not math.isfinite(selection.candidate_latency_ms)
                or selection.baseline_latency_ms is None
                or not math.isfinite(selection.baseline_latency_ms)
            ):
                return False, "incomplete_or_nonfinite_evidence"

        # Both policies require a matched win on every exemplar. Only a switch charges control
        # overhead to each exemplar's complete candidate invocation.
        threshold_scale = 1.0 - self._margin
        if policy is DAPlanMode.DA_SINGLE_BODY:
            for selection in selections:
                assert selection.baseline_latency_ms is not None
                if (
                    selection.candidate_latency_ms
                    > selection.baseline_latency_ms * threshold_scale
                ):
                    return False, "singleton_guard_rejected"
            return True, "admitted"

        for selection in selections:
            assert selection.baseline_latency_ms is not None
            candidate = selection.candidate_latency_ms + self._control_overhead_ms
            if candidate > selection.baseline_latency_ms * threshold_scale:
                return False, "switch_guard_rejected"
        return True, "admitted"


def validate_realization_capacity(
    distributions: Sequence[DADistribution], samples_per_distribution: int
) -> None:
    """Reject an over-capacity realization catalog before generation or tuning."""
    if samples_per_distribution <= 0:
        raise ValueError("samples_per_distribution must be positive")
    total = len(distributions) * samples_per_distribution
    if total == 0:
        raise ValueError("At least one DA distribution is required")
    if total > DA_MAX_EXEMPLARS:
        raise ValueError(
            f"DA supports {DA_MAX_EXEMPLARS} total selector exemplars, received {total}"
        )


def publish_compiled_plan(
    dispatcher: DAMoEDispatcher, compiled: DACompiledPlan
) -> DAPlan | None:
    """Publish one compiled policy through the runtime's pristine plan boundary."""
    if compiled.policy is DAPlanMode.DA_FALLBACK:
        dispatcher.clear_plan()
        return None

    bodies = [
        factorized_tactic_to_body(selection.selected_tactic)
        for selection in compiled.selections
    ]
    return dispatcher.publish_plan(
        [selection.expert_ids for selection in compiled.selections], bodies
    )
