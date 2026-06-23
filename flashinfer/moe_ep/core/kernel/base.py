"""Kernel backend ABCs and per-iteration context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch

    from ...config import BootstrapConfig, FleetParams
    from ...tensors import MoEEpTensors
    from ...weights import MoEWeightPack


@dataclass(frozen=True)
class SplitKernelContext:
    """Per-iteration inputs for a split-path inner kernel."""

    expert_tensors: "torch.Tensor"
    num_tokens: int
    fleet_params: "FleetParams"
    recv_topk_idx: Optional["torch.Tensor"] = None
    recv_topk_weights: Optional["torch.Tensor"] = None


class SplitKernelBackend(ABC):
    """Compute backend invoked after dispatch on the split EP path."""

    def __init__(self, config: object) -> None:
        self._config = config
        self._transformed_weights: Any = None

    @classmethod
    @abstractmethod
    def kernel_name(cls) -> str: ...

    def requires_weights(self) -> bool:
        return True

    def validate_init(
        self,
        bootstrap: "BootstrapConfig",
        fleet_params: "FleetParams",
    ) -> None:
        """Optional init-time validation beyond shared checks."""

    def preprocess_weights(
        self,
        weights: "MoEWeightPack",
        fleet_params: "FleetParams",
    ) -> Any:
        """Transform canonical weights into the layout this kernel expects."""
        self._transformed_weights = weights
        return weights

    @abstractmethod
    def compute(self, ctx: SplitKernelContext) -> "torch.Tensor": ...


class MegaKernelBackend(ABC):
    """Fused kernel backend that owns comm + local MoE on the mega EP path."""

    def __init__(self, config: object) -> None:
        self._config = config
        self._transformed_weights: Any = None

    @classmethod
    @abstractmethod
    def kernel_name(cls) -> str: ...

    def requires_weights(self) -> bool:
        return True

    def validate_init(
        self,
        bootstrap: "BootstrapConfig",
        fleet_params: "FleetParams",
    ) -> None:
        """Optional init-time validation beyond shared checks."""

    def preprocess_weights(
        self,
        weights: "MoEWeightPack",
        fleet_params: "FleetParams",
    ) -> Any:
        """Transform canonical weights into the layout this kernel expects."""
        return weights

    def prepare_workspace(
        self,
        bootstrap: "BootstrapConfig",
        fleet_params: "FleetParams",
    ) -> Any:
        """Allocate durable resources (e.g. symmetric memory buffers)."""
        return None

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: "FleetParams",
        *,
        stage_inputs: bool,
    ) -> None:
        """Optional per-iteration validation before staging/compute."""

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        stage_inputs: bool,
        num_tokens: int,
    ) -> None:
        """Copy or transform activations into workspace buffers."""

    @abstractmethod
    def compute(
        self,
        workspace: Any,
        transformed_weights: Any,
        *,
        num_tokens: int,
        output: "torch.Tensor",
    ) -> "torch.Tensor": ...

    def destroy(self, workspace: Any) -> None:
        """Release durable workspace resources."""
