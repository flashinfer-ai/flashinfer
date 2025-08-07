"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations  # python 3.7+

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Type

import torch


class TensorType(Enum):
    """
    TensorType represents the semantic type of tensors in the pipeline.
    """

    LOGITS = auto()
    """
    Raw or masked float scores, real-valued vectors
    """
    PROBS = auto()
    """
    Non-negative float probabilities, maybe normalized
    """
    INDICES = auto()
    """
    Integer token IDs, terminal type
    """

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"TensorType.{self.name}"


@dataclass(slots=True, frozen=True)
class TaggedTensor:
    """
    Tensor wrapper that maintains semantic type information through pipeline execution.

    It ensures type safety as tensors flow through the logits processing pipeline and zero friction for downstream PyTorch code

    Notes
    -----
    - TaggedTensor is primarily for internal use by :class:`LogitsPipe`
    - Users typically work with plain tensors; tagging happens automatically
    """

    data: torch.Tensor
    """
    The underlying tensor.
    """
    type: TensorType
    """
    The semantic type of the tensor.
    """

    @staticmethod
    def logits(t: torch.Tensor) -> TaggedTensor:
        """
        Create a TaggedTensor with type :attr:`TensorType.LOGITS`.
        """
        return TaggedTensor(t, TensorType.LOGITS)

    @staticmethod
    def probs(t: torch.Tensor) -> TaggedTensor:
        """
        Create a TaggedTensor with type :attr:`TensorType.PROBS`.
        """
        return TaggedTensor(t, TensorType.PROBS)

    @staticmethod
    def indices(t: torch.Tensor) -> TaggedTensor:
        """
        Create a TaggedTensor with type :attr:`TensorType.INDICES`.
        """
        return TaggedTensor(t, TensorType.INDICES)

    def __torch_function__(
        self,
        func: Any,
        types: Tuple[Type, ...],
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
        """
        Get the shape of the underlying tensor.
        """
        return self.data.shape

    @property
    def device(self) -> torch.device:
        """
        Get the device of the underlying tensor.
        """
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Get the data type of the underlying tensor.
        """
        return self.data.dtype

    def size(self, dim: Optional[int] = None) -> torch.Size | int:
        """
        Get the size of the underlying tensor.
        """
        return self.data.size(dim)

    def __repr__(self) -> str:
        return f"TaggedTensor(type={self.type}, shape={self.shape}, dtype={self.dtype}, device={self.device})"
