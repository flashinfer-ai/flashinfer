import logging
from typing import Any, List, Optional, Union

import torch

from .compiler import Compiler, compile_pipeline
from .fusion_rules import FusionRule
from .legalization import LegalizationError, infer_initial_sort, legalize_processors
from .processors import LogitsProcessor
from .types import Sort, TaggedTensor
from .validators import CompileError, ValidityCheck

logger = logging.getLogger(__name__)


class LogitsPipe:
    def __init__(
        self,
        processors: List[LogitsProcessor],
        compile: bool = True,
        input_sort: Optional[Sort] = None,
        custom_fusion_rules: Optional[List[FusionRule]] = None,
        custom_validity_checks: Optional[List[ValidityCheck]] = None,
    ):
        if not processors:
            raise ValueError("Pipeline cannot be empty")

        self.processors = list(processors)

        try:
            # Step 1: Infer initial sort type
            self._initial_sort = input_sort or infer_initial_sort(self.processors)

            # Step 2: Legalization - convert high-level processors to low-level ops
            self.ops = legalize_processors(self.processors, self._initial_sort)

            # Step 3: Compilation - type check, validate, and fuse ops
            if compile:
                self.compile(custom_fusion_rules, custom_validity_checks)

        except (LegalizationError, CompileError) as e:
            raise ValueError(f"Pipeline creation failed: {e}") from e

    def __repr__(self) -> str:
        processor_names = [proc.__class__.__name__ for proc in self.processors]
        return f"LogitsPipeline([{' -> '.join(processor_names)}])"

    def __call__(
        self, x: Union[torch.Tensor, TaggedTensor], **kwargs: Any
    ) -> torch.Tensor:
        if not self.compiled_ops:
            logger.warning("Pipeline is not compiled, running discrete ops.")
            ops = self.ops
        else:
            ops = self.compiled_ops

        if isinstance(x, TaggedTensor):
            tagged_tensor = x
        else:
            if self._initial_sort == Sort.PROBS:
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
    def initial_sort(self) -> Sort:
        return self._initial_sort

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
