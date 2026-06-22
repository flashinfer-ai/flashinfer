"""MoEEpSplitLayer — dispatch/combine EP with a pluggable inner kernel."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Sequence, Union

import torch
import torch.nn as nn

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
    DispatchOutput,
    FleetParams,
    HandleParams,
)
from ..core.comm.fleet import Fleet, create_fleet
from ..core.kernel.base import SplitKernelBackend, SplitKernelContext
from ..core.kernel.registry import create_split_kernel
from ..core.validation.common import (
    validate_arch_for_backend,
    validate_bootstrap_world_size,
    validate_fleet_params,
    validate_split_forward_inputs,
)
from .config import SplitConfig
from ..backends.split.kernel.identity.config import IdentityConfig

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

        self._kernel: SplitKernelBackend = create_split_kernel(self._kernel_config)

        if self._kernel.requires_weights() and fleet_params.weights is None:
            raise ValueError(
                "MoEEpSplitLayer requires FleetParams.weights for "
                f"kernel {type(self._kernel_config).__name__}"
            )

        self._validate_at_init()
        self._kernel.validate_init(bootstrap, fleet_params)

        if self._kernel.requires_weights() and fleet_params.weights is not None:
            self._kernel.preprocess_weights(fleet_params.weights, fleet_params)

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

    def _inner_compute(self, dispatch: DispatchOutput) -> torch.Tensor:
        ctx = SplitKernelContext(
            expert_tensors=dispatch.expert_tensors,
            num_tokens=dispatch.get_num_tokens(),
            fleet_params=self._fleet_params,
            recv_topk_idx=dispatch.recv_topk_idx,
            recv_topk_weights=dispatch.recv_topk_weights,
        )
        return self._kernel.compute(ctx)

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
            dispatch = handle.dispatch(DispatchInputParams(x=[t.hidden_states]))
            expert_out = self._inner_compute(dispatch)
            combine = handle.combine(
                CombineInputParams(
                    x=[expert_out],
                    out=torch.empty_like(t.hidden_states),
                )
            )
            handle.complete()
            return combine.x
        finally:
            handle.destroy()

    def destroy(self) -> None:
        if self._fleet is not None:
            self._fleet.destroy()
            self._fleet = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
