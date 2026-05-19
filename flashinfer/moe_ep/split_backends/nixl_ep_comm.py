"""Config adapter routing MoEEpLayer to the nixl_ep backend.

Named ``NvepConfig`` (rather than ``NixlEpConfig``) to match the design doc's
``SplitBackendOptions(comm=NvepConfig())`` form.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NvepConfig:
    """Pass into ``MoEEpLayer(..., backend=NvepConfig())`` to select NIXL-EP."""

    backend_name: str = "nixl_ep"
