"""Bootstrap the process-exclusive SM120 split-kernel source tree."""

from __future__ import annotations

import sys
from pathlib import Path

from flashinfer.moe_ep.shims._sources import source_root


_SOURCE_ROOT = source_root("flashinfer", "sm120_split")


def bootstrap_paths() -> Path:
    """Put this drop's raw packages ahead of any other MegaMoE tree."""

    source = str(_SOURCE_ROOT)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    return _SOURCE_ROOT


__all__ = ["bootstrap_paths"]
