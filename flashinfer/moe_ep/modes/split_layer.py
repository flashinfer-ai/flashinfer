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
from ..core.runtime import (
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
    split_comm_runtime_requirements,
)
from ..core.validation.common import (
    ensure_bootstrap_dist_validated,
    validate_arch_for_backend,
    validate_bootstrap_world_size,
    validate_fleet_params,
    validate_fleet_weights,
    validate_split_forward_inputs,
)
from ..weights import MoEWeightPack
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
        weights: MoEWeightPack,
        fleet_knobs: Sequence[AlgoKnob] = (),
        backend: Union[str, SplitConfig, object] = "nccl_ep",
    ) -> None:
        super().__init__()
        self._bootstrap = bootstrap
        self._fleet_params = fleet_params
        self._weights = weights
        self._fleet_knobs = list(fleet_knobs)
        if isinstance(backend, SplitConfig):
            self._comm_backend = backend.comm
            self._kernel_config = backend.kernel
        else:
            self._comm_backend = backend
            self._kernel_config = IdentityConfig()

        self._kernel: SplitKernelBackend = create_split_kernel(self._kernel_config)

        ensure_moe_ep_cuda_device(bootstrap)

        self._runtime = None
        if bootstrap.auto_bootstrap:
            self._runtime = bootstrap_moe_ep_runtime(
                bootstrap,
                split_comm_runtime_requirements(self._comm_backend_name()),
            )

        self._validate_at_init()
        self._kernel.validate_init(bootstrap, fleet_params)

        if type(self._kernel).requires_weights():
            self._kernel.preprocess_weights(self._weights, fleet_params)

        self._fleet: Fleet | None = None

        # Opt-in per-stage profiling. When True, forward() records CUDA events
        # around dispatch / compute / combine and stores elapsed GPU time (ms)
        # in ``last_timings_ms`` after a device sync. Off by default (zero
        # overhead on the hot path). Used by benchmarks/bench_moe_ep.py.
        self.enable_timing = False
        self.last_timings_ms: dict[str, float] = {}

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
        validate_fleet_weights(
            self._weights, self._fleet_params, self._bootstrap.world_size
        )
        # nixl_ep rendezvous-store validation is deferred to fleet creation
        # (first forward): layers are routinely constructed before
        # torch.distributed is initialized, and NixlEpFleet._resolve_store
        # raises the same actionable error when neither tcp_store nor an
        # initialized default group is available.
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
        ensure_bootstrap_dist_validated(self._bootstrap)
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
        result: torch.Tensor | None = None
        try:
            if not self.enable_timing:
                dispatch = handle.dispatch(DispatchInputParams(x=[t.hidden_states]))
                expert_out = self._inner_compute(dispatch)
                combine = handle.combine(
                    CombineInputParams(
                        x=[expert_out],
                        out=torch.empty_like(t.hidden_states),
                    )
                )
                result = combine.x
            else:
                ev = {
                    k: (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for k in ("dispatch", "compute", "combine")
                }
                ev["dispatch"][0].record()
                dispatch = handle.dispatch(DispatchInputParams(x=[t.hidden_states]))
                ev["dispatch"][1].record()
                ev["compute"][0].record()
                expert_out = self._inner_compute(dispatch)
                ev["compute"][1].record()
                ev["combine"][0].record()
                combine = handle.combine(
                    CombineInputParams(
                        x=[expert_out],
                        out=torch.empty_like(t.hidden_states),
                    )
                )
                ev["combine"][1].record()
                torch.cuda.synchronize()
                self.last_timings_ms = {
                    k: start.elapsed_time(end) for k, (start, end) in ev.items()
                }
                result = combine.x
        finally:
            handle.complete()
            handle.destroy()
        return result

    def destroy(self) -> None:
        if self._fleet is not None:
            self._fleet.destroy()
            self._fleet = None
        if self._runtime is not None:
            finalize_moe_ep_runtime(self._runtime)
            self._runtime = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
