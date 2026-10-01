"""Expert-parallel communication abstractions.

Split-path comm backends implement one of two peer interfaces:
:class:`MoEEpCommunication`, a self-contained dispatch/combine object (the
NVLink backends), or :class:`Fleet` / :class:`Handle`, the group /
per-step-handle API of the NCCL-EP and NIXL-EP backends.
"""

from .communication import (
    MoEEpCommParams,
    MoEEpCommunication,
    MoEEpDispatchResult,
    available_communication_backends,
    create_communication,
    register_communication,
)
from .fleet import Fleet, create_fleet
from .handle import Handle

__all__ = [
    "Fleet",
    "Handle",
    "MoEEpCommParams",
    "MoEEpCommunication",
    "MoEEpDispatchResult",
    "available_communication_backends",
    "create_communication",
    "create_fleet",
    "register_communication",
]
