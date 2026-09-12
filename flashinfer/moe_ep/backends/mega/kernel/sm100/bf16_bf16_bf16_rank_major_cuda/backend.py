"""Exact Blackwell BF16 rank-major MegaMoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, EpAlgorithm, EpLayout, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import TORCH_DIST
from ......core.validation.common import (
    MoEEpConfigError,
    validate_mega_arch,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig
from .staging import validate_rank_major_forward_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


_KERNEL_NAME = "sm100_bf16_bf16_bf16_rank_major_cuda"


@register_mega_kernel(_KERNEL_NAME)
class Bf16RankMajorCudaMegaKernelBackend(MegaKernelBackend):
    """FlashInfer adapter for the fixed 8-rank generated kernel sequence."""

    def __init__(
        self,
        config: Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig,
    ) -> None:
        if not isinstance(config, Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig):
            raise TypeError(
                f"{_KERNEL_NAME} config must be "
                "Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig, got "
                f"{type(config).__name__}"
            )
        super().__init__(config)
        self._kernel_config = config

    @classmethod
    def kernel_name(cls) -> str:
        return _KERNEL_NAME

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        del bootstrap
        return frozenset({TORCH_DIST})

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_mega_arch()
        exact = {
            "world_size": (bootstrap.world_size, 8),
            "num_experts": (fleet_params.num_experts, 256),
            "max_tokens_per_rank": (fleet_params.max_tokens_per_rank, 128),
            "token_hidden_size": (fleet_params.token_hidden_size, 7168),
            "dtype_bytes": (fleet_params.dtype_bytes, 2),
            "intermediate_size": (self._kernel_config.intermediate_size, 2048),
            "top_k": (self._kernel_config.top_k, 8),
        }
        for name, (actual, expected) in exact.items():
            if actual != expected:
                raise MoEEpConfigError(
                    f"{_KERNEL_NAME} requires {name}={expected}, got {actual}"
                )
        if fleet_params.algorithm is not EpAlgorithm.LOW_LATENCY:
            raise MoEEpConfigError(
                f"{_KERNEL_NAME} requires FleetParams.algorithm=LOW_LATENCY"
            )
        if fleet_params.layout is not EpLayout.RANK_MAJOR:
            raise MoEEpConfigError(
                f"{_KERNEL_NAME} requires FleetParams.layout=RANK_MAJOR"
            )
        if bootstrap.stream != 0:
            raise MoEEpConfigError(
                f"{_KERNEL_NAME} launches on the current torch CUDA stream; "
                "BootstrapConfig.stream must be 0"
            )
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            alignment=64,
        )

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedMegaWeights:
        transformed = preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
        )
        self._transformed_weights = transformed
        return transformed

    def validate_transformed_weights(
        self,
        transformed_weights: TransformedMegaWeights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        del bootstrap
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
        )
        self._transformed_weights = transformed_weights

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from ......kernel_src.blackwell_bf16_rank_major import (
            BlackwellBf16RankMajorSession,
        )

        return BlackwellBf16RankMajorSession(
            process_group=self.ep_comm_group,
            rank=self.ep_rank,
            world_size=self.ep_world_size,
            max_tokens_per_rank=fleet_params.max_tokens_per_rank,
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
            num_experts=fleet_params.num_experts,
            top_k=self._kernel_config.top_k,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_rank_major_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        if not quantize_input:
            raise MoEEpConfigError(
                f"{_KERNEL_NAME} requires MegaConfig.quantize_input=True"
            )
        workspace.stage_inputs(t.hidden_states, t.topk_ids, t.topk_weights)

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        if output is None:
            raise MoEEpConfigError(f"{_KERNEL_NAME} requires an owned output tensor")
        workspace.bind_weights(
            transformed_weights.w13_block_major,
            transformed_weights.w2_block_major,
        )
        return workspace.run(output)

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        """Share the large communication/scratch workspace across model layers."""
        return (
            _KERNEL_NAME,
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self.ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            fleet_params.token_hidden_size,
            self._kernel_config.intermediate_size,
            self._kernel_config.top_k,
        )


__all__ = ["Bf16RankMajorCudaMegaKernelBackend"]
