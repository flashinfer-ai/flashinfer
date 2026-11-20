"""Shared Git helpers for PR base/head comparisons."""

from __future__ import annotations

import subprocess


def resolve_merge_base(base: str, head: str) -> str:
    """Return the common ancestor that contains only changes made by the PR."""
    try:
        resolved = subprocess.check_output(
            ["git", "merge-base", base, head],
            text=True,
            errors="replace",
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(
            f"Could not resolve merge base for {base} and {head}: {detail}"
        ) from exc
    if not resolved:
        raise RuntimeError(f"Git returned an empty merge base for {base} and {head}")
    return resolved
