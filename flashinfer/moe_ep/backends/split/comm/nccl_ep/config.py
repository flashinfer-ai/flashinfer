"""Config adapter routing MoEEpLayer to the nccl_ep comm backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NcclEpConfig:
    """Pass into ``MoEEpLayer(..., backend=SplitConfig(comm=NcclEpConfig()))``."""

    backend_name: str = "nccl_ep"
