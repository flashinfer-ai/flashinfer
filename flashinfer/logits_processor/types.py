from __future__ import annotations  # python 3.7+

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import torch


class TensorType(Enum):
    """
    TensorType represents the semantic type of tensors in the pipeline.

    - LOGITS: Raw or masked scores (real-valued vectors)
    - PROBS: Non-negative scores (user decides whether normalized)
    - INDICES: Integer token IDs (terminal type)
    """

    LOGITS = auto()
    PROBS = auto()
    INDICES = auto()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"TensorType.{self.name}"


@dataclass(slots=True, frozen=True)
class TaggedTensor:
    data: torch.Tensor
    type: TensorType

    @staticmethod
    def logits(t: torch.Tensor) -> TaggedTensor:
        return TaggedTensor(t, TensorType.LOGITS)

    @staticmethod
    def probs(t: torch.Tensor) -> TaggedTensor:
        return TaggedTensor(t, TensorType.PROBS)

    @staticmethod
    def indices(t: torch.Tensor) -> TaggedTensor:
        return TaggedTensor(t, TensorType.INDICES)

    def __torch_function__(
        self,
        func: Any,
        types: Tuple[type, ...],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        kwargs = kwargs or {}

        unwrapped_args = tuple(
            arg.data if isinstance(arg, TaggedTensor) else arg for arg in args
        )

        result = func(*unwrapped_args, **kwargs)

        return result

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def size(self, dim: Optional[int] = None) -> torch.Size | int:
        return self.data.size(dim)

    def __repr__(self) -> str:
        return f"TaggedTensor(type={self.type}, shape={self.shape}, dtype={self.dtype}, device={self.device})"
