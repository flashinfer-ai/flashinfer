"""Error decoration for the vendored GC MoE sources."""

from __future__ import annotations

from contextlib import contextmanager

_PREFIX = "flashinfer.experimental.gc_moe"


@contextmanager
def translated(operation: str):
    """Prefix an upstream failure with this package, keeping its type."""
    try:
        yield
    except (ValueError, TypeError, RuntimeError, AssertionError) as exc:
        exc.args = (f"{_PREFIX}: {operation}: {exc}", *exc.args[1:])
        raise
