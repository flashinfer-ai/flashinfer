# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``MoELayer`` runner of the specialized backend
(:class:`flashinfer.fused_moe.SpecializedMoEConfig`)."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import torch

from ....autotuner import TuningConfig
from ...api import (
    ActivationConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantFormat,
    RoutingInputMode,
)
from ...runners import MoERunner
from .kernel import SpecializedMoEKernel, kill_switch_engaged
from .registry import registered_kernels


def _union(values) -> tuple:
    """Ordered union of the kernels' declaration tuples."""
    seen: list = []
    for group in values:
        for value in group:
            if value not in seen:
                seen.append(value)
    return tuple(seen)


_KERNELS = tuple(registered_kernels().values())


class _SpecializedPackedInputs(list):
    """One kernel's packed inputs; ``launch_state`` is the kernel binding that
    packed them, so ``forward`` runs the kernel that admitted the call."""

    def __init__(self, inputs: List[Any], *, launch_state: SpecializedMoEKernel):
        super().__init__(inputs)
        self.launch_state = launch_state


class SpecializedMoERunner(MoERunner):
    """The specialized backend: MoE kernels generated for one served configuration each.

    Every kernel registered under ``flashinfer/fused_moe/backends/specialized/kernels/``
    belongs to this one backend (see :mod:`.kernel` for the kernel contract). The
    class declarations are the union of the registered kernels'. At construction
    the runner binds the kernels whose configuration guard admits the layer's
    configuration (quantization, geometry, routing, activation, device
    architecture); with none, :meth:`check_support` raises, so ``MoELayer`` skips
    the backend for that configuration. Per call, :meth:`accepts` asks the bound
    kernels in name order and the first that admits the call runs it. A rejected
    call runs on the layer's other candidates. A kernel admits only the calls
    where it was measured faster than the other backends, so the backend prefers
    itself (``prefer_when_accepted``): outside ``flashinfer.autotune(True)``
    ``MoELayer`` runs an admitted call on it without timing it against the other
    candidates, and under ``autotune(True)`` times it against them like any
    candidate. Each kernel reads the weight view of the
    backend whose data path it computes, so a default ``MoELayer`` needs no extra
    preparation. ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` rejects every call.

    Under CUDA-graph capture a kernel admits only token counts it admitted
    eagerly, whose winners ``MoELayer`` timed outside capture.
    ``MoELayer.reset_winner()`` forgets those admissions with the winners
    (:meth:`forget_selections`), and a winner-cache miss under capture never
    times this backend (``capture_requires_eager_selection``).
    """

    backend_key: ClassVar[str] = "specialized"
    # The kernels admit exactly their tested envelope, where each was measured
    # faster than the other backends: outside autotune, MoELayer takes an
    # admitted call here without timing the other candidates.
    prefer_when_accepted: ClassVar[bool] = True
    # MoELayer leaves this backend out of a selection inside a CUDA-graph capture:
    # the backend serves a captured call only for a winner timed eagerly, so its
    # admission there without a cached winner never adds a timed candidate.
    capture_requires_eager_selection: ClassVar[bool] = True
    # The capabilities are the registered kernels', documented per kernel; the
    # generated activation matrix of the design doc leaves this backend out.
    capabilities_from_kernel_registry: ClassVar[bool] = True
    weight_view_aliases: ClassVar[Tuple[str, ...]] = _union(
        (k.weight_view,) for k in _KERNELS
    )
    supported_routing_modes: Tuple[RoutingInputMode, ...] = _union(
        k.routing_modes for k in _KERNELS
    )
    supported_quant_variants: ClassVar[Tuple[Tuple[QuantFormat, QuantFormat], ...]] = (
        _union(k.quant_variants for k in _KERNELS)
    )
    supported_output_formats: ClassVar[Tuple[QuantFormat, ...]] = _union(
        k.output_formats for k in _KERNELS
    )
    supported_activation_classes_by_quant: ClassVar[
        Dict[Tuple[QuantFormat, QuantFormat], Tuple[Type[ActivationConfig], ...]]
    ] = {
        pair: _union(k.activation_classes for k in _KERNELS if pair in k.quant_variants)
        for pair in _union(k.quant_variants for k in _KERNELS)
    }
    supported_activation_classes: ClassVar[Tuple[Type[ActivationConfig], ...]] = _union(
        k.activation_classes for k in _KERNELS
    )
    supports_fused_shared_experts: ClassVar[bool] = any(
        k.supports_fused_shared_experts for k in _KERNELS
    )
    supports_expert_parallelism: ClassVar[bool] = any(
        k.supports_expert_parallelism for k in _KERNELS
    )
    supports_per_token_scale: ClassVar[bool] = any(
        k.supports_per_token_scale for k in _KERNELS
    )

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        from ....utils import get_compute_capability

        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"SpecializedMoERunner requires CUDA, got {device}.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        major, minor = get_compute_capability(self.device)
        self._device_arch = major * 10 + minor
        # Every kernel has one launch configuration per call shape.
        self.tuning_config = TuningConfig(use_cuda_graph=True)
        # Kernels bound to this configuration, in name order (check_support).
        self.kernels: List[SpecializedMoEKernel] = []
        # Token count -> the bound kernel that admitted the latest such call.
        self._admitted_by: Dict[int, SpecializedMoEKernel] = {}

    @staticmethod
    def _declared_rejection(
        kernel: Type[SpecializedMoEKernel], config: MoEConfig
    ) -> Optional[str]:
        quant = config.quant
        if quant.pair not in kernel.quant_variants:
            return (
                f"does not serve weight={quant.weight.name}, "
                f"activation={quant.activation.name}"
            )
        if quant.output not in kernel.output_formats:
            return f"does not write output={quant.output.name}"
        if not isinstance(config.activation, kernel.activation_classes):
            return f"does not compute {type(config.activation).__name__}"
        experts = config.experts
        if (
            experts.num_fused_shared_experts
            and not kernel.supports_fused_shared_experts
        ):
            return "does not support fused shared experts"
        local = experts.local_num_experts or config.routing.num_experts
        if (
            experts.local_expert_offset != 0 or local != config.routing.num_experts
        ) and not kernel.supports_expert_parallelism:
            return "does not support expert parallelism"
        if quant.per_token_scale and not kernel.supports_per_token_scale:
            return "does not support per_token_scale=True"
        return None

    def _check_support(self) -> None:
        super()._check_support()
        bound: List[SpecializedMoEKernel] = []
        rejected: List[str] = []
        for name, kernel in registered_kernels().items():
            reason = self._declared_rejection(kernel, self.config)
            if reason is None:
                reason = kernel.config_rejection(
                    self.config, self._device_arch, self.device
                )
            if reason is None:
                bound.append(kernel(self.config, self.device))
            else:
                rejected.append(f"{name} {reason}")
        if not bound:
            detail = "; ".join(rejected) if rejected else "no kernel is registered"
            raise NotImplementedError(
                "SpecializedMoERunner: no specialized kernel serves this "
                f"configuration ({detail})."
            )
        self.kernels = bound

    def _build(self) -> None:
        """Each kernel is built per device by the first eager call it admits."""

    def forget_selections(self) -> None:
        """Forget eagerly admitted token counts; ``MoELayer.reset_winner()``
        calls this when it clears the winners those counts stand for."""
        for kernel in self.kernels:
            kernel.refresh()
        self._admitted_by.clear()

    def refresh(self) -> None:
        """Forget eagerly admitted token counts (``MoELayer.refresh_backends``)."""
        self.forget_selections()

    def accepts(self, act: MoEActivationPack, weights: MoEWeightPack) -> bool:
        # The kill switch rejects every call here, before any kernel is asked;
        # each kernel also checks it for direct callers.
        if kill_switch_engaged():
            return False
        for kernel in self.kernels:
            if act.routing_input_mode not in kernel.routing_modes:
                continue
            view = kernel.view_of(weights)
            if view is not None and kernel.accepts(act, view):
                self._admitted_by[act.num_tokens] = kernel
                return True
        return False

    def _kernel_for(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> SpecializedMoEKernel:
        """The kernel that admitted the latest call of this token count (MoELayer
        asks :meth:`accepts` before it packs), else the first bound kernel that
        consumes this routing mode and has its weight view (direct runner use)."""
        kernel = self._admitted_by.get(act.num_tokens)
        if kernel is not None:
            return kernel
        for kernel in self.kernels:
            if (
                act.routing_input_mode in kernel.routing_modes
                and kernel.view_of(weights) is not None
            ):
                return kernel
        return self.kernels[0]

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: Any) -> List[Any]:
        self._require_built()
        return [-1]

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        kernel = self._kernel_for(act, weights)
        view = kernel.view_of(weights)
        if view is None:
            # Raise the pack's own error naming the available views.
            view = weights.get_view(kernel.weight_view)
        return _SpecializedPackedInputs(
            kernel.pack_inputs(act, view), launch_state=kernel
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_built()
        kernel = kwargs.pop("launch_state", None)
        if kernel is None:
            kernel = getattr(inputs, "launch_state", None)
        if kernel is None:
            if len(self.kernels) != 1:
                raise RuntimeError(
                    "SpecializedMoERunner.forward requires the launch state "
                    "returned by pack_inputs."
                )
            kernel = self.kernels[0]
        return kernel.forward(list(inputs))
