"""MoEEpSplitLayer — dispatch/combine EP with a pluggable inner kernel."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
import torch.nn as nn

from .._validators import (
    validate_arch_for_backend,
    validate_bootstrap_world_size,
    validate_fleet_params,
    validate_split_forward_inputs,
)
from ..algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobTopologyCapacity,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
    _index_knobs,
)
from ..config import (
    BootstrapConfig,
    CombineInputParams,
    DispatchInputParams,
    FleetParams,
    HandleParams,
)
from ..fleet import Fleet, create_fleet
from ..weights import MoEWeightPack
from .config import FusedMoeKernelConfig, IdentityConfig, SplitConfig
from .kernels import SplitKernelContext, kernel_requires_weights, run_split_kernel

if TYPE_CHECKING:
    from ..tensors import MoEEpTensors


class MoEEpSplitLayer(nn.Module):
    """Expert-Parallel layer: dispatch → inner kernel → combine."""

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        fleet_knobs: Sequence[AlgoKnob] = (),
        backend: Union[str, SplitConfig, object] = "nccl_ep",
    ) -> None:
        super().__init__()
        self._bootstrap = bootstrap
        self._fleet_params = fleet_params
        self._fleet_knobs = list(fleet_knobs)
        if isinstance(backend, SplitConfig):
            self._comm_backend = backend.comm
            self._kernel_config = backend.kernel
        else:
            self._comm_backend = backend
            self._kernel_config = IdentityConfig()

        self._require_weights = (
            isinstance(self._kernel_config, IdentityConfig)
            and self._kernel_config.require_weights
        )

        if kernel_requires_weights(self._kernel_config) and fleet_params.weights is None:
            raise ValueError(
                "MoEEpSplitLayer requires FleetParams.weights for "
                f"kernel {type(self._kernel_config).__name__}"
            )
        if self._require_weights and fleet_params.weights is None:
            raise ValueError(
                "IdentityConfig.require_weights=True requires FleetParams.weights"
            )
        if isinstance(self._kernel_config, FusedMoeKernelConfig):
            raise NotImplementedError(
                "FusedMoeKernelConfig is not wired yet; use IdentityConfig for now"
            )

        self._validate_at_init()

        self._weights: Optional[MoEWeightPack] = fleet_params.weights
        self._preprocessed_weights: Optional[MoEWeightPack] = None
        if self._require_weights and self._weights is not None:
            self._preprocessed_weights = self._weights

        self._fleet: Fleet | None = None

    def _comm_backend_name(self) -> str:
        name = getattr(self._comm_backend, "backend_name", self._comm_backend)
        if not isinstance(name, str):
            raise TypeError(
                f"comm backend must be a string or have a .backend_name str attr; "
                f"got {self._comm_backend!r}"
            )
        return name

    def _validate_at_init(self) -> None:
        backend_name = self._comm_backend_name()
        validate_bootstrap_world_size(self._bootstrap)
        if backend_name not in ("nccl_ep", "nixl_ep"):
            return
        validate_arch_for_backend(backend_name)
        fleet_knobs = _index_knobs(self._fleet_knobs)
        cap_knob = fleet_knobs.get(FleetAlgoKnobTopologyCapacity)
        topology_capacity = (
            int(cap_knob.n) if cap_knob is not None else None  # type: ignore[attr-defined]
        )
        validate_fleet_params(
            self._fleet_params,
            backend=backend_name,
            world_size=self._bootstrap.world_size,
            quant=fleet_knobs.get(FleetAlgoKnobQuantization),  # type: ignore[arg-type]
            topology_capacity=topology_capacity,
        )

    def _ensure_fleet(self) -> Fleet:
        if self._fleet is None:
            self._fleet = create_fleet(
                self._bootstrap,
                self._fleet_params,
                self._fleet_knobs,
                backend=self._comm_backend,
            )
        return self._fleet

    def _inner_compute(
        self, expert_tensors: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        ctx = SplitKernelContext(
            expert_tensors=expert_tensors,
            num_tokens=num_tokens,
            fleet_params=self._fleet_params,
            weights=self._preprocessed_weights or self._weights,
        )
        return run_split_kernel(self._kernel_config, ctx)

    def forward(self, t: "MoEEpTensors") -> torch.Tensor:
        validate_split_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            self._fleet_params,
        )
        fleet = self._ensure_fleet()
        handle_knobs: list[AlgoKnob] = [
            HandleAlgoKnobUserStream(stream=torch.cuda.current_stream().cuda_stream),
            HandleAlgoKnobTopKWeights(weights=t.topk_weights),
        ]
        handle = fleet.create_handle(
            HandleParams(topk_ids=t.topk_ids),
            algo_knobs=handle_knobs,
        )
        try:
            d = handle.dispatch(DispatchInputParams(x=[t.hidden_states]))
            expert_out = self._inner_compute(d.expert_tensors, d.num_tokens)
            c = handle.combine(
                CombineInputParams(
                    x=[expert_out],
                    out=torch.empty_like(t.hidden_states),
                )
            )
            handle.complete()
            return c.x
        finally:
            handle.destroy()

    def destroy(self) -> None:
        if self._fleet is not None:
            self._fleet.destroy()
            self._fleet = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
