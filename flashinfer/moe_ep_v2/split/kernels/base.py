"""Split inner-kernel context and protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from ...config import FleetParams
    from ...weights import MoEWeightPack


@dataclass(frozen=True)
class SplitKernelContext:
    """Per-iteration inputs for a split-path inner kernel."""

    expert_tensors: "torch.Tensor"
    num_tokens: int
    fleet_params: "FleetParams"
    weights: Optional["MoEWeightPack"] = None


@runtime_checkable
class SplitKernel(Protocol):
    def __call__(self, ctx: SplitKernelContext) -> "torch.Tensor": ...
