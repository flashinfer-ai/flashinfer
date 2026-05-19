"""Config adapter routing MoEEpLayer to the nccl_ep backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NcclEpConfig:
    """Pass into ``MoEEpLayer(..., backend=NcclEpConfig())`` to select NCCL-EP."""

    backend_name: str = "nccl_ep"
