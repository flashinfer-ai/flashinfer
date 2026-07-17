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

    @classmethod
    def requires_weights(cls) -> bool:
        return True

    def validate_init(  # noqa: B027 - intentional no-op default
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
        self._ep_bootstrap: BootstrapConfig | None = None
        self._ep_comm_group: Any = None
        self._ep_rank: int = 0
        self._ep_world_size: int = 0

    @classmethod
    @abstractmethod
    def kernel_name(cls) -> str: ...

    def runtime_requirements(self, bootstrap: "BootstrapConfig") -> frozenset[str]:
        """Process resources this mega kernel needs (``torch_dist``, ``nvshmem``, …)."""
        from ...core.runtime import TORCH_DIST

        return frozenset({TORCH_DIST})

    def bind_ep_bootstrap(self, bootstrap: "BootstrapConfig") -> None:
        """Resolve EP rank/world once; comm group when dist is available."""
        self._ep_bootstrap = bootstrap
        self._ep_rank = bootstrap.rank
        self._ep_world_size = bootstrap.world_size
        self._ep_comm_group = None
        self._try_resolve_ep_comm_group(bootstrap)

    def _try_resolve_ep_comm_group(self, bootstrap: "BootstrapConfig") -> None:
        import torch.distributed as dist

        from ...core.bootstrap_utils import (
            bootstrap_comm_group,
            bootstrap_ep_rank_world,
        )

        if bootstrap.process_group is not None or dist.is_initialized():
            self._ep_comm_group = bootstrap_comm_group(bootstrap)
            self._ep_rank, self._ep_world_size = bootstrap_ep_rank_world(bootstrap)

    def _ensure_ep_bootstrap(self, bootstrap: "BootstrapConfig") -> None:
        if self._ep_bootstrap is not bootstrap:
            self.bind_ep_bootstrap(bootstrap)
        elif self._ep_comm_group is None:
            self._try_resolve_ep_comm_group(bootstrap)

    @property
    def ep_comm_group(self) -> "torch.distributed.ProcessGroup":
        if self._ep_comm_group is None and self._ep_bootstrap is not None:
            self._try_resolve_ep_comm_group(self._ep_bootstrap)
        if self._ep_comm_group is None:
            raise RuntimeError(
                "EP comm group is unavailable; initialize torch.distributed "
                "or set BootstrapConfig.process_group before workspace allocation"
            )
        return self._ep_comm_group

    @property
    def ep_rank(self) -> int:
        if self._ep_bootstrap is None:
            raise RuntimeError(
                "EP bootstrap is not bound; call bind_ep_bootstrap() first"
            )
        return self._ep_rank

    @property
    def ep_world_size(self) -> int:
        if self._ep_bootstrap is None:
            raise RuntimeError(
                "EP bootstrap is not bound; call bind_ep_bootstrap() first"
            )
        return self._ep_world_size

    def validate_init(  # noqa: B027 - intentional no-op default
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
        """Bind EP bootstrap (once) and allocate symmetric-memory workspace."""
        self._ensure_ep_bootstrap(bootstrap)
        return self._allocate_workspace(fleet_params)

    @abstractmethod
    def _allocate_workspace(self, fleet_params: "FleetParams") -> Any:
        """Backend-specific symmetric-memory / workspace allocation."""

    def validate_forward(  # noqa: B027 - intentional no-op default
        self,
        t: "MoEEpTensors",
        fleet_params: "FleetParams",
        *,
        quantize_input: bool,
    ) -> None:
        """Optional per-iteration validation before staging/compute."""

    def validate_transformed_weights(  # noqa: B027 - intentional no-op default
        self,
        transformed_weights: Any,
        bootstrap: "BootstrapConfig",
        fleet_params: "FleetParams",
    ) -> None:
        """Optional init-time validation for user-supplied kernel-ready weights."""

    def stage_inputs(  # noqa: B027 - intentional no-op default
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        """Copy or quantize activations into workspace buffers."""

    @abstractmethod
    def compute(
        self,
        workspace: Any,
        transformed_weights: Any,
        *,
        output: "torch.Tensor",
    ) -> "torch.Tensor": ...

    def destroy(self, workspace: Any) -> None:  # noqa: B027 - intentional no-op default
        """Release durable workspace resources."""
