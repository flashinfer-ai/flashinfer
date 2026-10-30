"""MXFP4 W4A4 entry point for the shared Rubin offline tuner."""

from ..tuning import run_tuning as _run_tuning


def run_tuning(args) -> int:
    return _run_tuning(args, "mxfp4")
