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
from typing import Any, List

from .types import TensorType


class LogitsProcessor(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def legalize(self, input_type: TensorType) -> List["Op"]:
        raise NotImplementedError

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class Temperature(LogitsProcessor):
    def __init__(self, **params: Any):
        super().__init__(**params)

    def legalize(self, input_type: TensorType) -> List["Op"]:
        from .operators import TemperatureOp

        if input_type != TensorType.LOGITS:
            raise ValueError(
                f"Temperature can only be applied to LOGITS, got {input_type}"
            )

        return [TemperatureOp(**self.params)]


class TopK(LogitsProcessor):
    def __init__(self, **params: Any):
        super().__init__(**params)

    def legalize(self, input_type: TensorType) -> List["Op"]:
        from .operators import LogitsTopKOp, ProbsTopKOp

        if input_type == TensorType.LOGITS:
            return [LogitsTopKOp(**self.params)]
        elif input_type == TensorType.PROBS:
            return [ProbsTopKOp(**self.params)]
        else:
            raise ValueError(f"TopK cannot be applied to {input_type}")


class TopP(LogitsProcessor):
    def __init__(self, **params: Any):
        super().__init__(**params)

    def legalize(self, input_type: TensorType) -> List["Op"]:
        from .operators import TopPOp

        if input_type != TensorType.PROBS:
            raise ValueError(f"TopP can only be applied to PROBS, got {input_type}")

        return [TopPOp(**self.params)]


class MinP(LogitsProcessor):
    def __init__(self, **params: Any):
        super().__init__(**params)

    def legalize(self, input_type: TensorType) -> List["Op"]:
        from .operators import MinPOp

        if input_type != TensorType.PROBS:
            raise ValueError(f"MinP can only be applied to PROBS, got {input_type}")

        return [MinPOp(**self.params)]


class Softmax(LogitsProcessor):
    def legalize(self, input_type: TensorType) -> List["Op"]:
        from .operators import SoftmaxOp

        if input_type != TensorType.LOGITS:
            raise ValueError(f"Softmax can only be applied to LOGITS, got {input_type}")

        return [SoftmaxOp(**self.params)]


class Sample(LogitsProcessor):
    def __init__(self, deterministic: bool = True, **params: Any):
        super().__init__(deterministic=deterministic, **params)

    def legalize(self, input_type: TensorType) -> List["Op"]:
        from .operators import LogitsSampleOp, ProbsSampleOp

        if input_type == TensorType.LOGITS:
            return [LogitsSampleOp(**self.params)]
        elif input_type == TensorType.PROBS:
            return [ProbsSampleOp(**self.params)]
        else:
            raise ValueError(f"Sampling cannot be applied to {input_type}")
