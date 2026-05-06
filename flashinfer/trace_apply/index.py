from __future__ import annotations

from dataclasses import dataclass, field

from flashinfer.trace_apply.hardware import sm_for_sku
from flashinfer.trace_apply.schema import Definition, Solution, TraceRecord
from flashinfer.trace_apply.source import Trace


@dataclass(slots=True, frozen=True)
class IndexKey:
    """Lookup key. Keyed by ``fi_api`` (the dotted public attribute the engine
    calls), not by definition name: a single API serves many const-specialized
    definitions (e.g. ``rmsnorm_h1536`` and ``rmsnorm_h7168`` both map to
    ``flashinfer.norm.rmsnorm``). ``axes`` is the *full* concrete axis vector
    (const values from the Definition unioned with the var values from the
    TraceRecord's workload), so the two definitions land on distinct keys.
    """

    fi_api: str
    axes: frozenset[tuple[str, int]]
    sm_arch: str


@dataclass(slots=True)
class Candidate:
    """A single (fi_api, axes, sm) candidate: a Solution plus the TraceRecord
    that ranks it. The loaded callable is attached lazily by runtime.py.
    """

    solution: Solution
    record: TraceRecord
    definition: Definition


@dataclass(slots=True)
class Index:
    by_key: dict[IndexKey, list[Candidate]] = field(default_factory=dict)
    definitions: dict[str, Definition] = field(default_factory=dict)
    solutions: dict[tuple[str, str], Solution] = field(default_factory=dict)
    # fi_api -> set of definition names that target it (any axes/sm).
    fi_apis: dict[str, set[str]] = field(default_factory=dict)

    def get(self, key: IndexKey) -> list[Candidate]:
        return self.by_key.get(key, [])

    def has_candidates_for(self, fi_api: str, sm_arch: str) -> bool:
        """True iff at least one indexed candidate exists for this fi_api on
        this SM (any axes). Used by install() for the skip-wrap optimization.
        """
        for k in self.by_key:
            if k.fi_api == fi_api and k.sm_arch == sm_arch:
                return True
        return False


def build_index(trace: Trace) -> Index:
    """Group trace records into per-key candidate lists, sorted by latency.

    The key's axis set is the union of the Definition's const axes and the
    record's workload (var) axes, so it equals the full concrete axis vector a
    runtime call produces.
    """
    idx = Index(definitions=dict(trace.definitions), solutions=dict(trace.solutions))

    grouped: dict[IndexKey, list[Candidate]] = {}
    for rec in trace.records:
        defn = trace.definitions.get(rec.definition)
        sol = trace.solutions.get((rec.definition, rec.solution))
        if defn is None or sol is None:
            # Record references a definition or solution we don't have. Skip.
            continue

        fi_api = defn.fi_api()
        if fi_api is None:
            # Definition without an fi_api tag cannot be wired to an API. Skip.
            continue

        record_sm = sm_for_sku(rec.environment.hardware)
        if record_sm is None:
            # Unknown hardware string in the trace record — skip rather than
            # guess. The hardware table (hardware.py) can be extended.
            continue

        # Full concrete axis vector: const (from definition) + var (from workload).
        full_axes = dict(defn.const_axes())
        full_axes.update(rec.workload.axes)

        key = IndexKey(
            fi_api=fi_api,
            axes=frozenset(full_axes.items()),
            sm_arch=record_sm,
        )
        grouped.setdefault(key, []).append(
            Candidate(solution=sol, record=rec, definition=defn)
        )
        idx.fi_apis.setdefault(fi_api, set()).add(defn.name)

    # Rank by measured latency (lower is better). Records without a perf number
    # sort last.
    for key, cands in grouped.items():
        cands.sort(
            key=lambda c: (
                c.record.performance.latency_ms
                if c.record.performance
                else float("inf")
            )
        )
        idx.by_key[key] = cands

    return idx


__all__ = ["IndexKey", "Candidate", "Index", "build_index"]
