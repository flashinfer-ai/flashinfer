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

"""Interface of one kernel of the ``specialized`` MoE backend, and the helpers
every kernel shares (the kill switch and CUDA-graph capture detection).

A kernel serves one configuration: a model's expert geometry, routing,
quantization and token counts on one GPU architecture. It reads the weight view
another backend already prepares (``weight_view``), so a default ``MoELayer``
needs no extra weight preparation for it. The contract, which
:class:`SpecializedMoERunner` relies on and the backend's tests check for every
kernel:

* :meth:`SpecializedMoEKernel.config_rejection` is the configuration guard: any
  configuration other than the served one is rejected at ``MoELayer``
  construction, so the kernel costs nothing per call elsewhere;
* :meth:`SpecializedMoEKernel.accepts` is the per-call guard. It never raises,
  and it rejects every call while ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``
  (the runner also checks the switch before asking any kernel). It builds the
  kernel for the device on the first eager call it admits; under CUDA-graph
  capture it admits a call only when the kernel is built and the same binding
  already admitted that token count eagerly (``eager_num_tokens``), that is,
  when ``MoELayer`` already timed the candidates for that call outside capture.
  ``MoELayer.reset_winner()`` clears those records with the winners, and a
  winner-cache miss under capture leaves this backend out of the selection.
  Nothing is built or timed inside a capture;
* :meth:`SpecializedMoEKernel.forward` computes the data path of the backend
  whose weight view it reads (same roundings, fp32 accumulation), and runs that
  backend instead for a call whose launch fails;
* :meth:`SpecializedMoEKernel.stats` reports dispatch and build diagnostics
  without side effects (no build, no CUDA call, no file read).
"""

from __future__ import annotations

import abc
import os
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Set, Tuple

import torch

from ...api import QuantFormat, RoutingInputMode

# ``1`` keeps every call off the specialized backend (read at call time). The
# same variable switches FlashInfer's other specialized kernels off.
KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"


def kill_switch_engaged() -> bool:
    """Whether ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` is set now."""
    return os.environ.get(KILL_SWITCH_ENV, "") == "1"


def is_capturing() -> bool:
    """Whether the current CUDA stream is capturing a graph."""
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


class SpecializedMoEKernel(abc.ABC):
    """One kernel of the specialized backend, bound to one ``MoELayer`` runner.

    A subclass declares what it serves in the class attributes below and is
    exported as ``KERNEL`` by its subpackage of
    ``flashinfer/fused_moe/backends/specialized/kernels/``, whose name equals
    :attr:`name`. :class:`SpecializedMoERunner` binds one instance per layer for
    each kernel whose :meth:`config_rejection` admits the layer's configuration;
    process-wide state (per-device builds, statistics) belongs to the kernel's
    module.
    """

    #: Registry name: the kernel's subpackage name.
    name: ClassVar[str] = ""
    #: Key of the existing weight view the kernel reads (another backend's).
    weight_view: ClassVar[str] = ""
    #: (weight, activation) quantization pairs it serves.
    quant_variants: ClassVar[Tuple[Tuple[QuantFormat, QuantFormat], ...]] = ()
    #: Output formats it writes.
    output_formats: ClassVar[Tuple[QuantFormat, ...]] = (QuantFormat.BF16,)
    #: Routing input modes it consumes.
    routing_modes: ClassVar[Tuple[RoutingInputMode, ...]] = ()
    #: Activation classes it computes.
    activation_classes: ClassVar[Tuple[type, ...]] = ()
    #: Compute capabilities (``major * 10 + minor``) it runs on.
    compute_capabilities: ClassVar[Tuple[int, ...]] = ()
    supports_expert_parallelism: ClassVar[bool] = False
    supports_fused_shared_experts: ClassVar[bool] = False
    supports_per_token_scale: ClassVar[bool] = False

    def __init__(self, config: Any, device: torch.device) -> None:
        self.config = config
        self.device = device
        # Token counts this binding admitted outside CUDA-graph capture; under
        # capture only these are admitted (see the module docstring).
        self.eager_num_tokens: Set[int] = set()

    @classmethod
    @abc.abstractmethod
    def config_rejection(
        cls, config: Any, arch: int, device: torch.device
    ) -> Optional[str]:
        """Why this kernel does not serve ``config`` (a ``MoEConfig``) on
        ``device`` of compute capability ``arch`` (``major * 10 + minor``), or
        ``None`` when it does. The quantization pair, output format, activation
        class, routing modes, expert parallelism and fused shared experts are
        checked from the declarations before this is called."""

    def view_of(self, weights: Any) -> Optional[Mapping[str, Any]]:
        """The weight view this kernel reads from a ``MoEWeightPack``, if prepared."""
        return weights.native_views.get(self.weight_view)

    @abc.abstractmethod
    def accepts(self, act: Any, view: Mapping[str, Any]) -> bool:
        """Whether this call (activation pack, weight view) runs on the kernel.
        Never raises; builds the kernel on the first eager call it admits."""

    @abc.abstractmethod
    def pack_inputs(self, act: Any, view: Mapping[str, Any]) -> List[Any]:
        """The kernel's positional inputs for one call, including its output.
        Raises ``ValueError`` for a call outside the kernel's tensor contract."""

    @abc.abstractmethod
    def forward(self, inputs: List[Any]) -> torch.Tensor:
        """Run one packed call and return its output."""

    def refresh(self) -> None:
        """Forget the token counts admitted eagerly (``MoELayer.reset_winner`` and
        ``MoELayer.refresh_backends`` clear the layer's winners, which those counts
        stand for)."""
        self.eager_num_tokens.clear()

    @classmethod
    @abc.abstractmethod
    def stats(cls) -> Dict[str, Any]:
        """Dispatch and build diagnostics; side-effect free."""

    @classmethod
    def aot_jit_specs(cls, sm_capabilities: Mapping[str, bool]) -> List[Any]:
        """JIT specs to precompile into the AOT package for ``sm_capabilities``
        (the dictionary ``flashinfer.aot`` builds); none by default."""
        return []
