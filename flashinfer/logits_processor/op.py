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

from abc import ABC, abstractmethod
from typing import Any, Dict

from .types import TaggedTensor, TensorType


class Op(ABC):
    IN: TensorType = None
    OUT: TensorType = None

    def __init__(self):
        if self.IN is None or self.OUT is None:
            raise ValueError(
                f"Operator {self.__class__.__name__} must define IN and OUT tensor types"
            )

    @abstractmethod
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        raise NotImplementedError

    def _validate_input_type(self, tensor: TaggedTensor) -> TensorType:
        if tensor.type != self.IN:
            raise ValueError(
                f"Operator {self.__class__.__name__} cannot accept input type {tensor.type}. "
                f"Expected: {self.IN}"
            )
        return self.OUT

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.IN} -> {self.OUT})"


class ParameterizedOp(Op):
    def __init__(self, **default_params: Any):
        super().__init__()
        self.default_params = default_params

    def _get_param(
        self, name: str, kwargs: Dict[str, Any], required: bool = True
    ) -> Any:
        if name in kwargs:
            return kwargs.pop(name)
        elif name in self.default_params:
            return self.default_params[name]
        elif required:
            raise ValueError(
                f"Required parameter '{name}' not provided for {self.__class__.__name__}"
            )
        else:
            return None
