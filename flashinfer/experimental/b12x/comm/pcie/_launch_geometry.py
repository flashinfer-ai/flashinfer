"""Shared policy for non-production PCIe collective launch geometry."""

from __future__ import annotations

import os


DIAGNOSTIC_GEOMETRY_ENV = "B12X_COMM_ALLOW_DIAGNOSTIC_GEOMETRY"
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def require_diagnostic_geometry(
    family: str,
    **geometry: tuple[int, int],
) -> None:
    """Reject launch geometry outside reviewed defaults unless explicitly enabled.

    Each keyword value is ``(actual, reviewed_default)``.  Keeping the policy in
    one host-only helper gives every production communication path the same
    opt-in without adding work to captured CUDA graph replay.
    """

    changed = {
        name: (int(actual), int(reviewed_default))
        for name, (actual, reviewed_default) in geometry.items()
        if int(actual) != int(reviewed_default)
    }
    if not changed:
        return
    if os.getenv(DIAGNOSTIC_GEOMETRY_ENV, "").strip().lower() in _ENABLED_VALUES:
        return
    details = ", ".join(
        f"{name}={actual} (reviewed default {reviewed_default})"
        for name, (actual, reviewed_default) in changed.items()
    )
    raise ValueError(
        f"{family} non-default launch geometry is diagnostic-only: {details}; "
        f"set {DIAGNOSTIC_GEOMETRY_ENV}=1 to opt in"
    )
