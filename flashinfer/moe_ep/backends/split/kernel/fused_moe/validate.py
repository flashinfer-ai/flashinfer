"""Validate consistency between EP transport config and fused MoE compute config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .....core.validation.common import MoEEpConfigError
from .....config import BootstrapConfig, FleetParams

if TYPE_CHECKING:
    from ......fused_moe.api import MoEConfig


def validate_compute_consistency(
    fleet_params: FleetParams,
    bootstrap: BootstrapConfig,
    compute_config: "MoEConfig",
) -> None:
    """Check the EP comm config and the unified-compute ``MoEConfig`` agree."""
    world_size = bootstrap.world_size
    rank = bootstrap.rank
    routing = compute_config.routing
    experts = compute_config.experts

    if routing.num_experts != fleet_params.num_experts:
        raise MoEEpConfigError(
            f"RoutingConfig.num_experts ({routing.num_experts}) != "
            f"FleetParams.num_experts ({fleet_params.num_experts}); the global "
            "expert count must be a single source of truth."
        )

    expected_local = fleet_params.num_experts // world_size
    local = experts.local_num_experts or routing.num_experts
    if local != expected_local:
        raise MoEEpConfigError(
            f"ExpertConfig.local_num_experts ({local}) != "
            f"num_experts // world_size ({expected_local}); each rank owns an "
            "equal shard of the global experts."
        )

    expected_offset = rank * expected_local
    if experts.local_expert_offset != expected_offset:
        raise MoEEpConfigError(
            f"ExpertConfig.local_expert_offset ({experts.local_expert_offset}) != "
            f"rank * local_num_experts ({expected_offset}) for rank {rank}."
        )

    if not compute_config.execution.do_finalize:
        raise MoEEpConfigError(
            "compute_config.execution.do_finalize must be True for MoE-EP: the "
            "bridge consumes a finalized [M, hidden] output and RANK_MAJOR/HT need "
            "the runner's weighted local pre-reduce."
        )
