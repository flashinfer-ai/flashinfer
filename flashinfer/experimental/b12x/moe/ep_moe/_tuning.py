"""Fixed W4A16 configuration contract for prepared expert-parallel MoE."""
from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import BackendConfig, TuningContract, make_fixed_contract


@dataclass(frozen=True, kw_only=True)
class EpMoeQuery:
    max_tokens: int
    num_tokens: int
    top_k: int
    num_experts: int
    local_num_experts: int
    hidden_size: int
    intermediate_size: int
    activation: str
    apply_router_weight_on_input: bool
    swiglu_limit: float | None
    swiglu_alpha: float
    swiglu_beta: float
    route_ids_dtype: str
    fast_math: bool


EpMoeConfig = BackendConfig
TUNING: TuningContract = make_fixed_contract(
    component_id="moe.ep_moe",
    query_type=EpMoeQuery,
    backend="w4a16",
)

__all__ = ["TUNING", "EpMoeConfig", "EpMoeQuery"]
