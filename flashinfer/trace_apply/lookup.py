from __future__ import annotations

from flashinfer.trace_apply.config import TraceApplyConfig
from flashinfer.trace_apply.hardware import sms_for_skus
from flashinfer.trace_apply.index import Candidate, Index, IndexKey


def _passes_filters(c: Candidate, sm_arch: str, config: TraceApplyConfig) -> bool:
    # Solution must declare support for the running SM (strict SM match).
    sol_sms = sms_for_skus(c.solution.spec.target_hardware)
    if sol_sms and sm_arch not in sol_sms:
        return False
    # Status must be PASSED.
    if c.record.status != "PASSED":
        return False
    # Author whitelist.
    if (
        config.allowed_authors is not None
        and c.solution.author not in config.allowed_authors
    ):
        return False
    return True


def lookup(
    index: Index,
    fi_api: str,
    axes: dict[str, int],
    sm_arch: str,
    config: TraceApplyConfig | None = None,
) -> Candidate | None:
    """Return the best candidate for this key, or None on miss.

    ``axes`` must be the full concrete axis vector for the call (const + var),
    matching how :func:`build_index` keyed the trace records.
    """
    cfg = config or TraceApplyConfig()
    key = IndexKey(fi_api=fi_api, axes=frozenset(axes.items()), sm_arch=sm_arch)
    for cand in index.get(key):
        if _passes_filters(cand, sm_arch, cfg):
            return cand
    return None


def explain(
    index: Index,
    fi_api: str,
    axes: dict[str, int],
    sm_arch: str,
    config: TraceApplyConfig | None = None,
) -> dict:
    """Return a structured record of how the lookup was resolved — every
    selection traces back to one row in the trace.
    """
    cfg = config or TraceApplyConfig()
    key = IndexKey(fi_api=fi_api, axes=frozenset(axes.items()), sm_arch=sm_arch)
    raw = index.get(key)
    accepted = [c for c in raw if _passes_filters(c, sm_arch, cfg)]
    selected = accepted[0] if accepted else None
    return {
        "fi_api": fi_api,
        "axes": dict(axes),
        "sm_arch": sm_arch,
        "candidates_total": len(raw),
        "candidates_passing_filters": len(accepted),
        "selected": (
            {
                "solution": selected.solution.name,
                "definition": selected.definition.name,
                "author": selected.solution.author,
                "latency_ms": (
                    selected.record.performance.latency_ms
                    if selected.record.performance
                    else None
                ),
                "record_hardware": selected.record.environment.hardware,
            }
            if selected
            else None
        ),
    }


__all__ = ["lookup", "explain"]
