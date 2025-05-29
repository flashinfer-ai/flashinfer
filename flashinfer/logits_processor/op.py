from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .types import Sort, TaggedTensor


class Op(ABC):
    IN: List[Sort] = []
    OUT: List[Sort] = []

    def __init__(self):
        if not self.IN or not self.OUT:
            raise ValueError(
                f"Operator {self.__class__.__name__} must define IN and OUT type signatures"
            )

        if len(self.IN) != len(self.OUT):
            raise ValueError(
                f"Operator {self.__class__.__name__} must have matching IN/OUT signature lengths"
            )

    @abstractmethod
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        raise NotImplementedError

    def _validate_input_sort(self, tensor: TaggedTensor) -> Sort:
        for i, input_sort in enumerate(self.IN):
            if tensor.sort == input_sort:
                return self.OUT[i]

        raise ValueError(
            f"Operator {self.__class__.__name__} cannot accept input sort {tensor.sort}. "
            f"Expected one of: {self.IN}"
        )

    def __repr__(self) -> str:
        signatures = [f"{inp} -> {out}" for inp, out in zip(self.IN, self.OUT)]
        return f"{self.__class__.__name__}({', '.join(signatures)})"


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
