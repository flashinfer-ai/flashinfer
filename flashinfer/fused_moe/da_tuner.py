"""Host-side distribution-aware MoE realization, search, and plan compilation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from flashinfer.fused_moe.da_moe import (
    DA_MAX_BODIES,
    DA_MAX_EXEMPLARS,
    DABody,
    DAMoEDispatcher,
    DAPlan,
    DAPlanMode,
)


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
    # Number of experts owned by the local rank.
    num_local_experts: int
    # Number of distinct experts selected for every token.
    top_k: int
    # Stable public-routing-rule identity supplied by the dtype adapter.
    routing_rule_fingerprint: str
    # Scalar applied after row-wise routing-weight normalization.
    routed_scaling_factor: float


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
        if not 0 < key.top_k <= key.num_local_experts:
            raise ValueError("top_k must be in [1, num_local_experts]")
        if (
            not math.isfinite(key.routed_scaling_factor)
            or key.routed_scaling_factor <= 0
        ):
            raise ValueError("routed_scaling_factor must be positive and finite")

        # Generate both mutable tensors outside full-op timing; measured launches only stage the
        # already-materialized realization into reusable profiling storage.
        distribution = DADistribution.parse(key.distribution)
        probabilities = self._expert_probabilities(
            key.num_local_experts, distribution
        ).to(device=key.device, dtype=torch.float32)
        expanded = probabilities.expand(key.num_tokens, -1)
        local_ids = torch.multinomial(expanded, key.top_k, replacement=False)
        expert_ids = (local_ids + key.local_expert_offset).to(torch.int32)

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


@dataclass(frozen=True)
class FactorizedTactic:
    """One C++-enumerated legal complete MoE tactic and its factorization."""

    # Opaque complete tactic passed unchanged to the ordinary runner.
    tactic: Any
    # Routing tile-N used by the metadata producer for this body.
    tile_n: int
    # Opaque FC1 component identity used only for coordinate grouping.
    fc1: Any
    # Opaque FC2 component identity used only for coordinate grouping.
    fc2: Any

    def to_body(self) -> DABody:
        """Decode one concrete TRTLLM tactic into its graph-body description."""
        identity = self.tactic
        if (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or int(identity[0]) != self.tile_n
            or int(identity[1]) < 0
        ):
            raise RuntimeError(
                "A TRTLLM DA body requires a concrete (tile_n, config_index) identity"
            )
        return DABody(tactic=int(identity[1]), tile_n=self.tile_n)


class FactorizedTacticSpace:
    """Index a complete legal tactic universe without inventing compositions."""

    def __init__(
        self,
        tactics: Sequence[FactorizedTactic],
        anchors: Mapping[int, Any],
    ) -> None:
        """Validate complete tactics and caller-declared deterministic anchors."""
        if not tactics:
            raise ValueError("Factorized tactic space cannot be empty")
        # Tile index supports bounded coordinate sweeps without inventing configurations.
        self._by_tile: dict[int, list[FactorizedTactic]] = {}
        # Component index validates every composed FC1/FC2 point against the legal universe.
        self._by_components: dict[tuple[int, Any, Any], FactorizedTactic] = {}
        # Complete-identity index resolves deterministic anchors supplied by the runner.
        self._by_identity: dict[Any, FactorizedTactic] = {}
        for tactic in tactics:
            if tactic.tile_n <= 0:
                raise ValueError("Every factorized tactic requires positive tile_n")
            component_key = (tactic.tile_n, tactic.fc1, tactic.fc2)
            if component_key in self._by_components:
                raise ValueError(f"Duplicate tactic factorization {component_key!r}")
            if tactic.tactic in self._by_identity:
                raise ValueError(f"Duplicate complete tactic {tactic.tactic!r}")
            self._by_tile.setdefault(tactic.tile_n, []).append(tactic)
            self._by_components[component_key] = tactic
            self._by_identity[tactic.tactic] = tactic

        # One legal anchor seeds factorized search independently for each routing tile.
        self._anchors: dict[int, FactorizedTactic] = {}
        for tile_n, tile_tactics in self._by_tile.items():
            if tile_n not in anchors:
                raise ValueError(f"Missing deterministic anchor for tile {tile_n}")
            anchor_identity = anchors[tile_n]
            anchor = self._by_identity.get(anchor_identity)
            if anchor is None or anchor.tile_n != tile_n:
                raise ValueError(
                    f"Anchor {anchor_identity!r} is not legal for tile {tile_n}"
                )
            self._anchors[tile_n] = anchor
            tile_tactics.sort(key=lambda item: repr(item.tactic))

    @property
    def tiles(self) -> tuple[int, ...]:
        """Return sorted routing tiles represented by the legal universe."""
        return tuple(sorted(self._by_tile))

    def anchor(self, tile_n: int) -> FactorizedTactic:
        """Return the runner-declared legal anchor for one tile."""
        return self._anchors[tile_n]

    def fc1_sweep(self, tile_n: int, fixed_fc2: Any) -> tuple[FactorizedTactic, ...]:
        """Return legal complete tactics varying FC1 with FC2 held fixed."""
        return tuple(
            tactic for tactic in self._by_tile[tile_n] if tactic.fc2 == fixed_fc2
        )

    def fc2_sweep(self, tile_n: int, fixed_fc1: Any) -> tuple[FactorizedTactic, ...]:
        """Return legal complete tactics varying FC2 with FC1 held fixed."""
        return tuple(
            tactic for tactic in self._by_tile[tile_n] if tactic.fc1 == fixed_fc1
        )

    def compose(self, tile_n: int, fc1: Any, fc2: Any) -> FactorizedTactic:
        """Return an enumerated complete composition or fail loudly."""
        try:
            return self._by_components[(tile_n, fc1, fc2)]
        except KeyError as error:
            raise RuntimeError(
                f"Illegal factorized MoE composition tile={tile_n}, "
                f"fc1={fc1!r}, fc2={fc2!r}"
            ) from error

    def all_tactics(self) -> tuple[FactorizedTactic, ...]:
        """Return every legal complete tactic for explicit exhaustive control."""
        return tuple(tactic for tile in self.tiles for tactic in self._by_tile[tile])


class FullOpMeasurementCache:
    """Retain finite full-operation timings under exact measurement identities."""

    def __init__(self) -> None:
        """Create an empty exact-measurement cache."""
        # Best finite timing observed for each exact caller-defined key.
        self._timings: dict[tuple[Any, ...], float] = {}

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


class FactorizedSearch:
    """Run bounded FC1/FC2 coordinate search using complete MoE timings."""

    def __init__(self, max_sweeps: int = 2) -> None:
        """Configure the confirmed one-or-two-sweep refinement budget."""
        if max_sweeps not in (1, 2):
            raise ValueError("max_sweeps must be one or two")
        # Maximum number of FC1-then-FC2 coordinate sweeps per tile.
        self._max_sweeps = max_sweeps

    def search(
        self,
        space: FactorizedTacticSpace,
        measure: Callable[[FactorizedTactic, bool], float],
    ) -> FactorizedTactic:
        """Return the best decisively timed complete tactic across all tiles."""
        # Each tile starts from its legal anchor and alternately pins FC2 then FC1; every point is
        # still a complete enumerated full-operation tactic.
        decisive: list[tuple[float, str, FactorizedTactic]] = []
        for tile_n in space.tiles:
            current = space.anchor(tile_n)
            for _ in range(self._max_sweeps):
                before = current
                current = self._best(space.fc1_sweep(tile_n, current.fc2), measure)
                current = self._best(space.fc2_sweep(tile_n, current.fc1), measure)
                current = space.compose(tile_n, current.fc1, current.fc2)
                if current == before:
                    break
            # Re-measure the composed winner decisively before comparing winners across tiles.
            final_time = float(measure(current, True))
            if not math.isfinite(final_time):
                raise RuntimeError(
                    f"Non-finite decisive MoE timing for tile={tile_n}, "
                    f"tactic={current.tactic!r}"
                )
            decisive.append((final_time, repr(current.tactic), current))
        return min(decisive, key=lambda item: (item[0], item[1]))[2]

    @staticmethod
    def _best(
        tactics: Sequence[FactorizedTactic],
        measure: Callable[[FactorizedTactic, bool], float],
    ) -> FactorizedTactic:
        """Choose one finite group point using deterministic tactic ties."""
        if not tactics:
            raise RuntimeError("A factorized coordinate sweep has no legal tactics")
        observations = []
        for tactic in tactics:
            timing = float(measure(tactic, False))
            if not math.isfinite(timing):
                raise RuntimeError(
                    f"Non-finite factorized MoE timing for {tactic.tactic!r}"
                )
            observations.append((timing, repr(tactic.tactic), tactic))
        return min(observations, key=lambda item: (item[0], item[1]))[2]


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
        guard_enabled: bool = True,
        margin: float = 0.0,
        control_overhead_us: float = 12.0,
    ) -> None:
        """Configure guard policy without changing candidate construction."""
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 0.0 <= margin < 1.0:
            raise ValueError("guard margin must be in [0, 1)")
        if not math.isfinite(control_overhead_us) or control_overhead_us < 0:
            raise ValueError("control_overhead_us must be finite and nonnegative")
        # Global expert width consumed by the runtime selector and uploaded spectra.
        self._num_experts = num_experts
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
        eager_tactic, eager_distribution = self._select_eager_tactic(selections)
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
        """Fingerprint the exact global-domain load spectrum consumed by kNN."""
        ids = selection.expert_ids.detach().to(device="cpu", dtype=torch.int64)
        if bool(((ids < 0) | (ids >= self._num_experts)).any()):
            raise ValueError("Selector exemplar contains an out-of-range expert ID")
        loads = (
            torch.bincount(ids.flatten(), minlength=self._num_experts)
            .sort(descending=True)
            .values
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

        # Singleton compares robust worst cases; switch charges its control overhead once to each
        # exemplar's complete candidate invocation.
        threshold_scale = 1.0 - self._margin
        if policy is DAPlanMode.DA_SINGLE_BODY:
            candidate_worst = max(
                selection.candidate_latency_ms for selection in selections
            )
            baseline_worst = max(
                selection.baseline_latency_ms
                for selection in selections
                if selection.baseline_latency_ms is not None
            )
            admitted = candidate_worst <= baseline_worst * threshold_scale
            return admitted, "admitted" if admitted else "singleton_guard_rejected"

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

    bodies = [selection.selected_tactic.to_body() for selection in compiled.selections]
    return dispatcher.publish_plan(
        [selection.expert_ids for selection in compiled.selections], bodies
    )
