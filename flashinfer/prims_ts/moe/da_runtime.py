"""Distribution-aware body adapter for PrimsTS MoE runners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TYPE_CHECKING

import torch

from flashinfer.autotuner import TuningConfig
from flashinfer.fused_moe.core import TrtllmDaRuntime
from flashinfer.fused_moe.tactic_search import FactorizedTacticSpace, MoeTactic

from .da_body import PrimsTsDaBodyRunner

if TYPE_CHECKING:
    from .runner import (
        PrimsTsBf16MoERunner,
        PrimsTsFp8BlockScaleMoERunner,
        PrimsTsFp8PerTensorMoERunner,
        PrimsTsMxfp4Bf16MoERunner,
        PrimsTsMxfp4Mxfp8MoERunner,
        PrimsTsNvfp4MoERunner,
    )

    PrimsTsMoERunner = (
        PrimsTsBf16MoERunner
        | PrimsTsFp8BlockScaleMoERunner
        | PrimsTsFp8PerTensorMoERunner
        | PrimsTsMxfp4Bf16MoERunner
        | PrimsTsMxfp4Mxfp8MoERunner
        | PrimsTsNvfp4MoERunner
    )
else:
    # Avoid importing runner.py while its backend wrappers lazily import this module.
    PrimsTsMoERunner = Any


PrimsTsMoeResult = torch.Tensor | list[torch.Tensor]


class PrimsTsDaRuntime(TrtllmDaRuntime):
    """Adapt a PrimsTS runner to the shared native DA graph service."""

    def __init__(self, moe_runner: PrimsTsMoERunner) -> None:
        """Compose one ordinary PrimsTS runner with shared DA lifecycle machinery."""
        super().__init__(moe_runner)
        # Replace the TRTLLM forward-mode adapter with PrimsTS explicit body capabilities.
        self._body_runner = PrimsTsDaBodyRunner(moe_runner)

    @property
    def backend(self) -> str:
        """Return the explicit ordinary backend identity used by shared DA state."""
        return "prims_ts"

    def normalize_baseline_tactic(
        self,
        factorized_space: FactorizedTacticSpace,
        baseline_tactic: MoeTactic,
    ) -> tuple[int, int]:
        """Map a scalar or paired PrimsTS tactic to its concrete DA body identity."""
        resolved = factorized_space.resolve_public_tactic(baseline_tactic)
        return tuple(int(value) for value in resolved.tactic)

    def make_from_logits_profile_tuning_config(
        self, profile_inputs: list[torch.Tensor], num_tokens: int
    ) -> TuningConfig:
        """Build PrimsTS canonical-routing arenas without a runner-only ABI keyword."""
        from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

        # PrimsTS body inputs already encode the canonical routed representation by shape and
        # value-aware slots; unlike TRTLLM's native runner, its TuningConfig has no routing-mode
        # constructor field.
        return self._moe_runner._make_tuning_config(
            MoeRunnerInputs.from_list(profile_inputs),
            tune_max_num_tokens=num_tokens,
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )


def run_prims_ts_da(
    *,
    custom_op: str,
    runner: PrimsTsMoERunner,
    tuning_config: TuningConfig,
    inputs: list[torch.Tensor],
    runner_kwargs: Mapping[str, Any],
    baseline_tactic: MoeTactic,
    routing_input_mode: int,
    num_experts: int,
    local_expert_offset: int,
    num_local_experts: int,
    top_k: int,
    routing_method_type: int,
    routed_scaling_factor: float | None,
    run_fixed_tactic: Callable[[MoeTactic], PrimsTsMoeResult],
    finish_switch: Callable[[], PrimsTsMoeResult],
) -> PrimsTsMoeResult:
    """Dispatch one eligible PrimsTS operation through shared DA orchestration."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.da_config import get_enabled_da_moe_config
    from flashinfer.fused_moe.da_runtime import run_dist_aware_tactic
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs, RoutingInputMode

    # Preserve ordinary dispatch when DA is disabled. When enabled, shared orchestration may use
    # the DA-preferred eager tactic for this token bucket but never chooses another backend.
    config = get_enabled_da_moe_config(default_enabled=True)
    if config is None:
        return run_fixed_tactic(baseline_tactic)

    # Describe the active public routing representation to the shared registry. FromLogits keeps
    # its inactive precomputed slots out of the exact graph-binding signature.
    route_mode = RoutingInputMode(routing_input_mode)
    routing_id_index = MoeRunnerInputs.idx(
        "routing_logits" if route_mode == RoutingInputMode.FromLogits else "topk_ids"
    )
    return run_dist_aware_tactic(
        backend="prims_ts",
        custom_op=custom_op,
        tuner=AutoTuner.get(),
        config=config,
        runner=runner,
        runtime=PrimsTsDaRuntime(runner),
        tuning_config=tuning_config,
        inputs=inputs,
        runner_kwargs=runner_kwargs,
        baseline_tactic=baseline_tactic,
        routing_input_mode=route_mode,
        routing_id_index=routing_id_index,
        routing_weight_index=MoeRunnerInputs.idx("expert_weights"),
        routing_precomputed_id_index=(
            MoeRunnerInputs.idx("topk_ids")
            if route_mode == RoutingInputMode.FromLogits
            else None
        ),
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        top_k=top_k,
        routing_method_type=routing_method_type,
        routed_scaling_factor=routed_scaling_factor,
        run_fixed_tactic=run_fixed_tactic,
        finish_switch=finish_switch,
    )
