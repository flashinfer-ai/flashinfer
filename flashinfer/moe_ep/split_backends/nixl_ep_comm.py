"""Config adapter routing MoEEpLayer to the nixl_ep backend.

Named ``NvepConfig`` (rather than ``NixlEpConfig``) to match the design doc's
``SplitBackendOptions(comm=NvepConfig())`` form.
"""

from __future__ import annotations

from dataclasses import dataclass


# Not frozen — see the note in nccl_ep_comm.py (aleozlx review, PR #3453).
@dataclass
class NvepConfig:
    """Pass into ``MoEEpLayer(..., backend=NvepConfig())`` to select NIXL-EP."""

    backend_name: str = "nixl_ep"


# Spelling aliases: ``NIXLEPConfig`` matches the user-facing / mega_moe_integration
# naming; ``NixlEpConfig`` matches the CamelCase convention of ``NcclEpConfig``.
NIXLEPConfig = NvepConfig
NixlEpConfig = NvepConfig
