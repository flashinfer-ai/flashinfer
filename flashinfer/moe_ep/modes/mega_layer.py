"""MoEEpMegaLayer — fused mega-MoE kernel path."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from ..config import BootstrapConfig, FleetParams
from ..core.kernel.registry import create_mega_kernel
from ..core.validation.common import validate_bootstrap_world_size
from ..weights import MoEWeightPack
from .config import MegaConfig

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
        self._megakernel_config = backend.megakernel

        if fleet_params.weights is None:
            raise ValueError("MoEEpMegaLayer requires FleetParams.weights")

        validate_bootstrap_world_size(bootstrap)

        self._kernel = create_mega_kernel(self._megakernel_config)
        self._kernel.validate_init(bootstrap, fleet_params)

        self._weights: MoEWeightPack = fleet_params.weights
        self._transformed: Optional[Any] = None
        self._workspace: Any = None

        if backend.preprocess_weights:
            self._preprocess_weights()

    def _preprocess_weights(self) -> None:
        if self._transformed is not None:
            return
        self._transformed = self._kernel.preprocess_weights(
            self._weights, self._fleet_params
        )

    def _ensure_workspace(self) -> Any:
        if self._workspace is not None and getattr(self._workspace, "x", None) is None:
            self._workspace = None
        if self._workspace is None:
            self._workspace = self._kernel.prepare_workspace(
                self._bootstrap, self._fleet_params
            )
        return self._workspace

    def forward(self, t: "MoEEpTensors") -> torch.Tensor:
        self._kernel.validate_forward(
            t,
            self._fleet_params,
            stage_inputs=self._mega_config.stage_inputs,
        )

        if self._transformed is None:
            self._preprocess_weights()
        assert self._transformed is not None

        workspace = self._ensure_workspace()
        num_tokens = t.hidden_states.shape[0]

        self._kernel.stage_inputs(
            t,
            workspace,
            stage_inputs=self._mega_config.stage_inputs,
            num_tokens=num_tokens,
        )

        y = torch.empty_like(t.hidden_states, dtype=torch.bfloat16)
        return self._kernel.compute(
            workspace,
            self._transformed,
            num_tokens=num_tokens,
            output=y,
        )

    def destroy(self) -> None:
        if self._workspace is not None:
            self._kernel.destroy(self._workspace)
            self._workspace = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
