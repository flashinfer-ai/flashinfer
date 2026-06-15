"""MoEEpMegaLayer — fused DeepGEMM mega-MoE over symmetric memory."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .._validators import (
    validate_bootstrap_world_size,
    validate_mega_arch,
    validate_mega_fleet_params,
    validate_mega_forward_inputs,
)
from ..config import BootstrapConfig, FleetParams
from ..weights import MoEWeightPack
from .config import DeepGemmMegaMoeConfig, MegaConfig
from .weights import TransformedMegaWeights, preprocess_mega_weights

if TYPE_CHECKING:
    from ..tensors import MoEEpTensors


class MoEEpMegaLayer(nn.Module):
    """Fused EP mega kernel — no separate dispatch/combine transport."""

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        backend: MegaConfig,
    ) -> None:
        super().__init__()
        self._bootstrap = bootstrap
        self._fleet_params = fleet_params
        self._mega_config = backend
        self._kernel_config: DeepGemmMegaMoeConfig = backend.kernel

        if fleet_params.weights is None:
            raise ValueError("MoEEpMegaLayer requires FleetParams.weights")

        validate_bootstrap_world_size(bootstrap)
        validate_mega_arch()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=backend.kernel.intermediate_size,
            top_k=backend.kernel.top_k,
        )

        self._weights: MoEWeightPack = fleet_params.weights
        self._transformed: Optional[TransformedMegaWeights] = None
        self._symm_buffer = None

        if backend.preprocess_weights:
            self._preprocess_weights()

    def _process_group(self) -> dist.ProcessGroup:
        if not dist.is_initialized():
            raise RuntimeError(
                "MoEEpMegaLayer requires torch.distributed to be initialized"
            )
        return dist.group.WORLD

    def _preprocess_weights(self) -> None:
        if self._transformed is not None:
            return
        self._transformed = preprocess_mega_weights(
            self._weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=self._fleet_params.token_hidden_size,
        )

    def get_symm_buffer(self):
        import deep_gemm

        if self._symm_buffer is not None and getattr(self._symm_buffer, "x", None) is None:
            self._symm_buffer = None
        if self._symm_buffer is None:
            group = self._process_group()
            k = self._kernel_config
            fp = self._fleet_params
            self._symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                fp.num_experts,
                fp.max_tokens_per_rank,
                k.top_k,
                fp.token_hidden_size,
                k.intermediate_size,
            )
        return self._symm_buffer

    def forward(self, t: "MoEEpTensors") -> torch.Tensor:
        import deep_gemm

        validate_mega_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            self._fleet_params,
            top_k=self._kernel_config.top_k,
            stage_inputs=self._mega_config.stage_inputs,
            scales=t.scales,
        )

        if self._transformed is None:
            self._preprocess_weights()
        assert self._transformed is not None

        symm_buffer = self.get_symm_buffer()
        num_tokens = t.hidden_states.shape[0]

        if self._mega_config.stage_inputs:
            from .staging import stage_mega_moe_inputs

            stage_mega_moe_inputs(
                t.hidden_states,
                t.topk_weights,
                t.topk_ids,
                symm_buffer.x[:num_tokens],
                symm_buffer.x_sf[:num_tokens],
                symm_buffer.topk_idx[:num_tokens],
                symm_buffer.topk_weights[:num_tokens],
            )
        else:
            symm_buffer.x[:num_tokens].copy_(t.hidden_states)
            assert t.scales is not None
            symm_buffer.x_sf[:num_tokens].copy_(t.scales)
            symm_buffer.topk_idx[:num_tokens].copy_(t.topk_ids)
            symm_buffer.topk_weights[:num_tokens].copy_(t.topk_weights)

        y = torch.empty_like(t.hidden_states, dtype=torch.bfloat16)
        kcfg = self._kernel_config
        deep_gemm.fp8_fp4_mega_moe(
            y,
            self._transformed[0],
            self._transformed[1],
            symm_buffer,
            activation_clamp=kcfg.activation_clamp,
            fast_math=kcfg.fast_math,
        )
        return y

    def destroy(self) -> None:
        if self._symm_buffer is not None:
            self._symm_buffer.destroy()
            self._symm_buffer = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
