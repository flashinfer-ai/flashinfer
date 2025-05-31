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

import logging
from typing import Any, List, Optional, Union

import torch

from .compiler import compile_pipeline
from .fusion_rules import FusionRule
from .legalization import LegalizationError, infer_initial_type, legalize_processors
from .processors import LogitsProcessor
from .types import TaggedTensor, TensorType
from .validators import CompileError, ValidityCheck

logger = logging.getLogger(__name__)


class LogitsPipe:
    def __init__(
        self,
        processors: List[LogitsProcessor],
        compile: bool = True,
        input_type: Optional[TensorType] = None,
        custom_fusion_rules: Optional[List[FusionRule]] = None,
        custom_validity_checks: Optional[List[ValidityCheck]] = None,
    ):
        if not processors:
            raise ValueError("Pipeline cannot be empty")

        self.processors = list(processors)

        try:
            # Step 1: Infer initial input tensor type
            self._initial_type = input_type or infer_initial_type(self.processors)

            # Step 2: Legalization - convert high-level processors to low-level ops
            self.ops = legalize_processors(self.processors, self._initial_type)

            # Step 3: Compilation - type check, validate, and fuse ops
            if compile:
                self.compile(custom_fusion_rules, custom_validity_checks)
            else:
                self.compiled_ops = None

        except (LegalizationError, CompileError) as e:
            raise ValueError(f"Pipeline creation failed: {e}") from e

    def __repr__(self) -> str:
        processor_names = [proc.__class__.__name__ for proc in self.processors]
        op_names = [op.__class__.__name__ for op in self.ops]
        compiled_op_names = (
            [op.__class__.__name__ for op in self.compiled_ops]
            if self.compiled_ops
            else []
        )
        return f"LogitsPipe([{' -> '.join(processor_names)}], ops=[{' -> '.join(op_names)}], compiled_ops=[{' -> '.join(compiled_op_names)}])"

    def __call__(
        self, x: Union[torch.Tensor, TaggedTensor], **kwargs: Any
    ) -> torch.Tensor:
        if self.compiled_ops is None:
            logger.warning("Pipeline is not compiled, running discrete ops.")
            ops = self.ops
        else:
            ops = self.compiled_ops

        if isinstance(x, TaggedTensor):
            tagged_tensor = x
        else:
            if self._initial_type == TensorType.PROBS:
                tagged_tensor = TaggedTensor.probs(x)
            else:
                tagged_tensor = TaggedTensor.logits(x)

        runtime_kwargs = dict(kwargs)

        for i, op in enumerate(ops):
            try:
                tagged_tensor = op(tagged_tensor, **runtime_kwargs)
            except Exception as e:
                raise ValueError(
                    f"Error executing operator {i} ({op.__class__.__name__}): {e}"
                ) from e

        return tagged_tensor.data

    @property
    def initial_type(self) -> TensorType:
        return self._initial_type

    def compile(
        self,
        custom_fusion_rules: Optional[List[FusionRule]] = None,
        custom_validity_checks: Optional[List[ValidityCheck]] = None,
    ) -> None:
        try:
            self.compiled_ops = compile_pipeline(
                self.ops, custom_fusion_rules, custom_validity_checks
            )
        except CompileError as e:
            raise ValueError(f"Compilation failed: {e}") from e
