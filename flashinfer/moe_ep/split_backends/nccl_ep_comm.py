"""Config adapter routing MoEEpLayer to the nccl_ep backend."""

from __future__ import annotations

from dataclasses import dataclass


# Not frozen: these configs are only passed to MoEEpLayer(..., backend=...)
# and read via getattr(.backend_name); they're never hashed or used as dict
# keys. Committing frozen=True now would make a later drop a breaking change
# (it would remove the synthesized __hash__ / __setattr__), so keep them
# plain. (aleozlx review, PR #3453)
@dataclass
class NcclEpConfig:
    """Pass into ``MoEEpLayer(..., backend=NcclEpConfig())`` to select NCCL-EP."""

    backend_name: str = "nccl_ep"


# Spelling alias matching the user-facing / mega_moe_integration naming.
NCCLEPConfig = NcclEpConfig
