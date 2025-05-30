from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .types import Sort


class LogitsProcessor(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def legalize(self, input_sort: Sort) -> List["Op"]:
        raise NotImplementedError

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class Temperature(LogitsProcessor):
    def __init__(self, **params: Any):
        super().__init__(**params)

    def legalize(self, input_sort: Sort) -> List["Op"]:
        from .operators import TemperatureOp

        if input_sort != Sort.LOGITS:
            raise ValueError(
                f"Temperature can only be applied to LOGITS, got {input_sort}"
            )

        return [TemperatureOp(**self.params)]


class TopK(LogitsProcessor):
    def __init__(self, **params: Any):
        super().__init__(**params)

    def legalize(self, input_sort: Sort) -> List["Op"]:
        from .operators import LogitsTopKOp, ProbsTopKOp

        if input_sort == Sort.LOGITS:
            return [LogitsTopKOp(**self.params)]
        elif input_sort == Sort.PROBS:
            return [ProbsTopKOp(**self.params)]
        else:
            raise ValueError(f"TopK cannot be applied to {input_sort}")


class TopP(LogitsProcessor):
    def __init__(self, **params: Any):
        super().__init__(**params)

    def legalize(self, input_sort: Sort) -> List["Op"]:
        from .operators import TopPOp

        if input_sort != Sort.PROBS:
            raise ValueError(f"TopP can only be applied to PROBS, got {input_sort}")

        return [TopPOp(**self.params)]


class MinP(LogitsProcessor):
    def __init__(self, **params: Any):
        super().__init__(**params)

    def legalize(self, input_sort: Sort) -> List["Op"]:
        from .operators import MinPOp

        if input_sort != Sort.PROBS:
            raise ValueError(f"MinP can only be applied to PROBS, got {input_sort}")

        return [MinPOp(**self.params)]


class Softmax(LogitsProcessor):
    def legalize(self, input_sort: Sort) -> List["Op"]:
        from .operators import SoftmaxOp

        if input_sort != Sort.LOGITS:
            raise ValueError(f"Softmax can only be applied to LOGITS, got {input_sort}")

        return [SoftmaxOp(**self.params)]


class Sample(LogitsProcessor):
    def __init__(self, deterministic: bool = True, **params: Any):
        super().__init__(deterministic=deterministic, **params)

    def legalize(self, input_sort: Sort) -> List["Op"]:
        from .operators import SampleLogitsOp, SampleProbsOp

        if input_sort == Sort.LOGITS:
            return [SampleLogitsOp(**self.params)]
        elif input_sort == Sort.PROBS:
            return [SampleProbsOp(**self.params)]
        else:
            raise ValueError(f"Sampling cannot be applied to {input_sort}")
