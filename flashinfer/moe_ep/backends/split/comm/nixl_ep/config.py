"""Config adapter routing MoEEpLayer to the nixl_ep comm backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NvepConfig:
    """Pass into ``MoEEpLayer(..., backend=SplitConfig(comm=NvepConfig()))``."""

    backend_name: str = "nixl_ep"
