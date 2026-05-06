from __future__ import annotations

from dataclasses import dataclass, field

from flashinfer.trace_apply.hardware import sm_for_sku, sms_for_skus
from flashinfer.trace_apply.schema import Definition, Solution, TraceRecord
from flashinfer.trace_apply.source import Trace


@dataclass(slots=True, frozen=True)
class IndexKey:
    definition: str
    axes: frozenset[tuple[str, int]]
    sm_arch: str


@dataclass(slots=True)
class Candidate:
    """A single (definition, axes, sm) candidate: a Solution plus the
    TraceRecord that ranks it. The loaded callable is attached lazily by
    runtime.py on first use.
    """

    solution: Solution
    record: TraceRecord
    definition: Definition


@dataclass(slots=True)
class Index:
    by_key: dict[IndexKey, list[Candidate]] = field(default_factory=dict)
    definitions: dict[str, Definition] = field(default_factory=dict)
    solutions: dict[tuple[str, str], Solution] = field(default_factory=dict)

    def get(self, key: IndexKey) -> list[Candidate]:
        return self.by_key.get(key, [])

    def has_candidates_for(self, definition_name: str, sm_arch: str) -> bool:
        """True iff at least one indexed candidate exists for this definition
        on this SM (any axes). Used by install() for the skip-wrap optimization.
        """
        for k in self.by_key:
            if k.definition == definition_name and k.sm_arch == sm_arch:
                return True
        return False


def build_index(trace: Trace) -> Index:
    """Group trace records into per-key candidate lists, sorted by latency."""
    idx = Index(definitions=dict(trace.definitions), solutions=dict(trace.solutions))

    grouped: dict[IndexKey, list[Candidate]] = {}
    for rec in trace.records:
        defn = trace.definitions.get(rec.definition)
        sol = trace.solutions.get((rec.definition, rec.solution))
        if defn is None or sol is None:
            # Record references a definition or solution we don't have. Skip.
            continue

        record_sm = sm_for_sku(rec.environment.hardware)
        if record_sm is None:
            # Unknown hardware string in the trace record — skip rather than
            # guess. The hardware table can be extended.
            continue

        key = IndexKey(
            definition=rec.definition,
            axes=frozenset(rec.workload.axes.items()),
            sm_arch=record_sm,
        )
        grouped.setdefault(key, []).append(Candidate(solution=sol, record=rec, definition=defn))

    # Rank by measured latency (lower is better). Records without a perf number
    # sort last.
    for key, cands in grouped.items():
        cands.sort(key=lambda c: (c.record.performance.latency_ms if c.record.performance else float("inf")))
        idx.by_key[key] = cands

    return idx


__all__ = ["IndexKey", "Candidate", "Index", "build_index"]
