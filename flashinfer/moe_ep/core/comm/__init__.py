"""Expert-parallel communication abstractions.

:class:`MoEEpCommunication` is the MoE-level dispatch/combine interface every
communication backend implements. :class:`Fleet` / :class:`Handle` are the
transport-level API of the NCCL-EP and NIXL-EP backends, whose
group/per-step-handle structure mirrors those libraries.
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
