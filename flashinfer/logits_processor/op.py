from abc import ABC, abstractmethod
from typing import Any, Dict

from .types import Sort, TaggedTensor


class Op(ABC):
    IN: Sort = None
    OUT: Sort = None

    def __init__(self):
        if self.IN is None or self.OUT is None:
            raise ValueError(
                f"Operator {self.__class__.__name__} must define IN and OUT sort types"
            )

    @abstractmethod
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        raise NotImplementedError

    def _validate_input_sort(self, tensor: TaggedTensor) -> Sort:
        if tensor.sort != self.IN:
            raise ValueError(
                f"Operator {self.__class__.__name__} cannot accept input sort {tensor.sort}. "
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
