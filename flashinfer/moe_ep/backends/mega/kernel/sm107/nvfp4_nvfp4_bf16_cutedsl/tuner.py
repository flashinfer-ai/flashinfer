"""NVFP4 entry point for the shared SM107 offline tuner."""

from __future__ import annotations

from ..tuning import run_tuning as _run_tuning


def run_tuning(args) -> int:
    return _run_tuning(args, "nvfp4")
