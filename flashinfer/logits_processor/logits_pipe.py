from typing import Any, List, Optional, Union

import torch

from .compiler import Compiler, compile_pipeline
from .fusion_rules import FusionRule
from .op import Op
from .types import Sort, TaggedTensor
from .validators import CompileError, ValidityCheck


class LogitsPipe:
    def __init__(self, ops: List[Op], compiler: Optional[Compiler] = None):
        if not ops:
            raise ValueError("Pipeline cannot be empty")

        self.ops = list(ops)
        try:
            if compiler is None:
                self.compiled_ops = compile_pipeline(self.ops)
            else:
                self.compiled_ops = compiler.compile(self.ops)
        except CompileError as e:
            raise ValueError(f"Pipeline compilation failed: {e}") from e

        self._initial_sort = self._determine_initial_sort()

    def __init__(
        self,
        ops: List[Op],
        custom_fusion_rules: Optional[List[FusionRule]] = None,
        custom_validity_checks: Optional[List[ValidityCheck]] = None,
    ):
        if not ops:
            raise ValueError("Pipeline cannot be empty")

        self.ops = list(ops)
        try:
            self.compiled_ops = compile_pipeline(
                self.ops, custom_fusion_rules, custom_validity_checks
            )
        except CompileError as e:
            raise ValueError(f"Pipeline compilation failed: {e}") from e

        self._initial_sort = self._determine_initial_sort()

    def __repr__(self) -> str:
        op_names = [op.__class__.__name__ for op in self.ops]
        return f"LogitsPipe([{' → '.join(op_names)}])"

    def __call__(
        self, x: Union[torch.Tensor, TaggedTensor], **kwargs: Any
    ) -> torch.Tensor:
        if not self.compiled_ops:
            raise ValueError("Pipeline is not compiled")

        if isinstance(x, TaggedTensor):
            tagged_tensor = x
        else:
            if self._initial_sort == Sort.PROBS:
                tagged_tensor = TaggedTensor.probs(x)
            else:
                tagged_tensor = TaggedTensor.logits(x)

        runtime_kwargs = dict(kwargs)

        for i, op in enumerate(self.compiled_ops):
            try:
                tagged_tensor = op(tagged_tensor, **runtime_kwargs)
            except Exception as e:
                raise ValueError(
                    f"Error executing operator {i} ({op.__class__.__name__}): {e}"
                ) from e

        if runtime_kwargs:
            unused_keys = list(runtime_kwargs.keys())
            print(f"WARN: Unused kwargs: {unused_keys}")

        return tagged_tensor.data

    def _determine_initial_sort(self) -> Sort:
        first_op = self.ops[0]
        if Sort.PROBS in first_op.IN:
            return Sort.PROBS
        else:
            return Sort.LOGITS
