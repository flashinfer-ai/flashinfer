"""FC1/FC2 factorized search over complete fused-MoE tactics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field


# Public MoE tactics are either the fallback sentinel or a two-coordinate backend tactic.
MoeTactic = int | list[int] | tuple[int, int]
# Factorized search operates only on normalized, hashable complete-tactic identities.
ConcreteMoeTactic = tuple[int, int]
# Cache lookup normalizes a public list tactic into this hashable representation.
MoeTacticKey = int | ConcreteMoeTactic


@dataclass(frozen=True)
class FactorizedTactic:
    """Describe one legal complete tactic through two opaque coordinates."""

    # Opaque complete tactic used by backend-internal body dispatch.
    tactic: ConcreteMoeTactic
    # Routing or scheduling tile owned by this complete tactic.
    tile_n: int
    # Opaque first-coordinate identity.
    fc1: int
    # Opaque second-coordinate identity.
    fc2: int
    # Public runner tactic passed unchanged through AutoTuner and its JSON cache.
    public_tactic: MoeTactic | None = field(default=None, compare=False, hash=False)

    def public_identity(self) -> MoeTactic:
        """Return the runner-visible tactic represented by this factorization."""
        return self.tactic if self.public_tactic is None else self.public_tactic


@dataclass(frozen=True)
class FactorizedSearchResult:
    """Retain the winner and small complete-tactic finalist set from one search."""

    # Full-operation winner published through the ordinary AutoTuner cache.
    winner: FactorizedTactic
    # One coordinate-refined complete finalist per represented tile.
    finalists: tuple[FactorizedTactic, ...]


class FactorizedTacticSpace:
    """Index a complete legal tactic universe without inventing compositions."""

    def __init__(
        self,
        tactics: Sequence[FactorizedTactic],
        anchors: Mapping[int, ConcreteMoeTactic],
    ) -> None:
        """Validate complete tactics and caller-declared deterministic anchors."""
        if not tactics:
            raise ValueError("Factorized tactic space cannot be empty")
        # Tile index supports bounded coordinate sweeps without inventing configurations.
        self._by_tile: dict[int, list[FactorizedTactic]] = {}
        # Component index validates every composed coordinate pair against the legal universe.
        self._by_components: dict[tuple[int, int, int], FactorizedTactic] = {}
        # Complete-identity index resolves deterministic anchors supplied by the runner.
        self._by_identity: dict[ConcreteMoeTactic, FactorizedTactic] = {}
        # Public runner identities map cached tactics back to exact complete bodies.
        self._by_public_identity: dict[MoeTacticKey, FactorizedTactic] = {}
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
            public_identity = self._public_identity_key(tactic.public_identity())
            if public_identity in self._by_public_identity:
                raise ValueError(f"Duplicate public tactic {public_identity!r}")
            self._by_public_identity[public_identity] = tactic

        # One legal anchor seeds factorized search independently for each tile.
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
        """Return sorted tiles represented by the legal universe."""
        return tuple(sorted(self._by_tile))

    def anchor(self, tile_n: int) -> FactorizedTactic:
        """Return the runner-declared legal anchor for one tile."""
        return self._anchors[tile_n]

    def resolve_public_tactic(self, public_tactic: MoeTactic) -> FactorizedTactic:
        """Resolve one ordinary runner tactic to its complete factorization."""
        try:
            return self._by_public_identity[self._public_identity_key(public_tactic)]
        except KeyError as error:
            raise RuntimeError(
                f"Ordinary tactic {public_tactic!r} is absent from the legal factorized universe"
            ) from error

    @staticmethod
    def _public_identity_key(public_tactic: MoeTactic) -> MoeTacticKey:
        """Canonicalize public list tactics for stable identity lookup."""
        if isinstance(public_tactic, list):
            normalized = tuple(public_tactic)
            if len(normalized) != 2:
                raise ValueError("A paired MoE tactic requires exactly two coordinates")
            return normalized
        return public_tactic

    def fc1_sweep(self, tile_n: int, fixed_fc2: int) -> tuple[FactorizedTactic, ...]:
        """Return legal complete tactics varying FC1 with FC2 held fixed."""
        return tuple(
            tactic for tactic in self._by_tile[tile_n] if tactic.fc2 == fixed_fc2
        )

    def fc2_sweep(self, tile_n: int, fixed_fc1: int) -> tuple[FactorizedTactic, ...]:
        """Return legal complete tactics varying FC2 with FC1 held fixed."""
        return tuple(
            tactic for tactic in self._by_tile[tile_n] if tactic.fc1 == fixed_fc1
        )

    def compose(self, tile_n: int, fc1: int, fc2: int) -> FactorizedTactic:
        """Return an enumerated complete composition or fail loudly."""
        try:
            return self._by_components[(tile_n, fc1, fc2)]
        except KeyError as error:
            raise RuntimeError(
                f"Illegal factorized composition tile={tile_n}, "
                f"fc1={fc1!r}, fc2={fc2!r}"
            ) from error

    def all_tactics(self) -> tuple[FactorizedTactic, ...]:
        """Return every legal complete tactic for exhaustive diagnostics."""
        return tuple(tactic for tile in self.tiles for tactic in self._by_tile[tile])

    def restricted_to_public_tactics(
        self, public_tactics: Sequence[MoeTactic]
    ) -> FactorizedTacticSpace:
        """Rebuild the legal universe after ordinary tactic blocklist filtering."""
        allowed = {
            self._public_identity_key(public_tactic)
            for public_tactic in public_tactics
            if public_tactic != -1
        }
        tactics = [
            tactic
            for tactic in self.all_tactics()
            if self._public_identity_key(tactic.public_identity()) in allowed
        ]
        if not tactics:
            raise ValueError("No concrete factorized tactics remain after filtering")
        anchors: dict[int, ConcreteMoeTactic] = {}
        for tactic in tactics:
            anchors.setdefault(tactic.tile_n, tactic.tactic)
        return FactorizedTacticSpace(tactics, anchors)


class FactorizedSearch:
    """Run bounded coordinate search using complete-operation timings."""

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
        """Return the winning complete tactic while preserving the legacy API."""
        return self.search_with_finalists(space, measure).winner

    def search_with_finalists(
        self,
        space: FactorizedTacticSpace,
        measure: Callable[[FactorizedTactic, bool], float],
    ) -> FactorizedSearchResult:
        """Refine one complete finalist per tile and rank them decisively."""
        # Each tile starts from its legal anchor and alternately pins FC2 then FC1. Every measured
        # point remains a complete runner tactic, so the full operation is always the authority.
        finalists: list[FactorizedTactic] = []
        for tile_n in space.tiles:
            current = space.anchor(tile_n)
            for _ in range(self._max_sweeps):
                before = current
                current = self._best(space.fc1_sweep(tile_n, current.fc2), measure)
                current = self._best(space.fc2_sweep(tile_n, current.fc1), measure)
                current = space.compose(tile_n, current.fc1, current.fc2)
                if current == before:
                    break
            finalists.append(current)

        # Revisit only the small per-tile finalist set for the cross-tile decision. Callers may
        # memoize exact measurements; the decisive flag still identifies this final authority.
        scored = []
        for tactic in finalists:
            timing = float(measure(tactic, True))
            if not math.isfinite(timing):
                raise RuntimeError(
                    f"Non-finite decisive timing for tile={tactic.tile_n}, "
                    f"tactic={tactic.tactic!r}"
                )
            scored.append((timing, repr(tactic.tactic), tactic))
        winner = min(scored, key=lambda item: (item[0], item[1]))[2]
        return FactorizedSearchResult(winner=winner, finalists=tuple(finalists))

    @staticmethod
    def _best(
        tactics: Sequence[FactorizedTactic],
        measure: Callable[[FactorizedTactic, bool], float],
    ) -> FactorizedTactic:
        """Choose one finite coordinate point with deterministic tactic ties."""
        if not tactics:
            raise RuntimeError("A factorized coordinate sweep has no legal tactics")
        observations = []
        for tactic in tactics:
            timing = float(measure(tactic, False))
            if not math.isfinite(timing):
                raise RuntimeError(
                    f"Non-finite factorized timing for {tactic.tactic!r}"
                )
            observations.append((timing, repr(tactic.tactic), tactic))
        return min(observations, key=lambda item: (item[0], item[1]))[2]
