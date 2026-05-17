from flashinfer.trace_apply.config import TraceApplyConfig
from flashinfer.trace_apply.hardware import current_sm, sm_for_sku, sms_for_skus
from flashinfer.trace_apply.index import Candidate, Index, IndexKey, build_index
from flashinfer.trace_apply.install import (
    disable,
    get_config,
    get_index,
    install,
    is_installed,
)
from flashinfer.trace_apply.lookup import explain as _lookup_explain
from flashinfer.trace_apply.lookup import lookup
from flashinfer.trace_apply.runtime import reset_stats, stats_snapshot
from flashinfer.trace_apply.schema import (
    Axis,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    IOSpec,
    Performance,
    Solution,
    SourceFile,
    TraceRecord,
    Workload,
)
from flashinfer.trace_apply.source import (
    Trace,
    TracePaths,
    iter_traces,
    load_definitions,
    load_solutions,
    load_trace,
    resolve_trace_path,
)


def stats() -> dict:
    """Return a snapshot of per-(definition, status) call counts.

    Status keys: hit, fallback_no_candidate, fallback_unwarmed_in_capture,
    fallback_runtime_error.
    """
    return stats_snapshot()


def explain(definition: str, axes: dict, sm_arch: str | None = None) -> dict:
    """Return how a (definition, axes, sm) lookup would resolve against the
    currently installed index. Useful for debugging "why did dispatch pick X?"
    without enabling logging.
    """
    idx = get_index()
    if idx is None:
        raise RuntimeError("Trace Apply is not installed. Call install() first.")
    sm = sm_arch or current_sm()
    return _lookup_explain(idx, definition, dict(axes), sm, get_config())


# Alias for the design-doc API surface: enable() is the same as install().
enable = install


__all__ = [
    # config
    "TraceApplyConfig",
    # data model
    "Axis",
    "BuildSpec",
    "Candidate",
    "Correctness",
    "Definition",
    "Environment",
    "IOSpec",
    "Index",
    "IndexKey",
    "Performance",
    "Solution",
    "SourceFile",
    "Trace",
    "TracePaths",
    "TraceRecord",
    "Workload",
    # public API
    "build_index",
    "current_sm",
    "disable",
    "enable",
    "explain",
    "get_config",
    "get_index",
    "install",
    "is_installed",
    "iter_traces",
    "load_definitions",
    "load_solutions",
    "load_trace",
    "lookup",
    "reset_stats",
    "resolve_trace_path",
    "sm_for_sku",
    "sms_for_skus",
    "stats",
]
