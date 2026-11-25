"""Locate packaged kernel sources independently of adapter placement."""

from pathlib import Path


def source_root(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1].joinpath("kernels", *parts)
