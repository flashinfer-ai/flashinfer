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
from .op import Op
from .processors import LogitsProcessor
from .types import TaggedTensor, TensorType
from .validators import CompileError, ValidityCheck

logger = logging.getLogger(__name__)


class LogitsPipe:
    """
    Provides a declarative way to build processing pipelines for LLM outputs.

    Parameters
    ----------
    processors : List[:class:`LogitsProcessor`]
        List of processors to apply in sequence.
    compile : bool, optional
        Whether to compile the pipeline with fusion optimizations. Default is True. LogitsPipe.compile() can be called to perform compilation after pipeline instantiation.
    input_type : Optional[TensorType], optional
        Expected input tensor type. It can be TensorType.LOGITS or TensorType.PROBS. It's required if the first processor can take both types. In other cases, it will be automatically inferred from the first processor.
        Default is None.
    custom_fusion_rules : Optional[List[FusionRule]], optional
        Additional fusion rules to apply during compilation. Default is None.
    custom_validity_checks : Optional[List[ValidityCheck]], optional
        Additional validity checks to apply during compilation. Default is None.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopK, Sample
    >>> torch.manual_seed(42)
    >>>
    >>> # Basic pipeline with temperature, top-k, and sampling
    >>> pipe = LogitsPipe([
    ...     Temperature(),
    ...     Softmax(),
    ...     TopK(),
    ...     Sample(deterministic=True)
    ... ])
    >>>
    >>> # Execute the pipeline
    >>> logits = torch.randn(4, 32000, device="cuda")  # [batch_size, vocab_size]
    >>> token_ids = pipe(logits, temperature=0.9, top_k=40)
    >>> token_ids
    tensor([15806,  8154, 13923, 20311], device='cuda:0')
    >>>
    >>> # Pipeline starting from probabilities
    >>> from flashinfer.logits_processor import TensorType, TopK
    >>>
    >>> prob_pipe = LogitsPipe(
    ...     [TopK(), Sample()],
    ...     input_type=TensorType.PROBS
    ... )
    >>> probs = torch.softmax(logits, dim=-1)
    >>> token_ids = prob_pipe(probs, top_k=40)
    >>> token_ids
    tensor([  346, 14846,  1517,  9006], device='cuda:0')

    Notes
    -----
    - The pipeline automatically validates type compatibility between operations.
    - Operations are fused when possible
    - Runtime parameters (like temperature, top_k) are passed with pipe.call().
    - The output is always a plain torch.Tensor, not a :class:`~flashinfer.logits_processor.TaggedTensor`.
    """

    def __init__(
        self,
        processors: List[LogitsProcessor],
        compile: bool = True,
        input_type: Optional[TensorType] = None,
        custom_fusion_rules: Optional[List[FusionRule]] = None,
        custom_validity_checks: Optional[List[ValidityCheck]] = None,
    ):
        """
        Constructor for a :class:`LogitsPipe`.
        """
        if not processors:
            raise ValueError("Pipeline cannot be empty")

        self.processors = list(processors)

        try:
            # Step 1: Infer initial input tensor type
            self._initial_type = input_type or infer_initial_type(self.processors)

            # Step 2: Legalization - convert high-level processors to low-level ops
            self.ops = legalize_processors(self.processors, self._initial_type)

            # Step 3: Compilation - type check, validate, and fuse ops
            self.compiled_ops: Optional[List[Op]] = None
            if compile:
                self.compile(custom_fusion_rules, custom_validity_checks)

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
        """
        The initial input tensor type of the pipeline. It can be :attr:`TensorType.LOGITS` or :attr:`TensorType.PROBS`.
        """
        return self._initial_type

    def compile(
        self,
        custom_fusion_rules: Optional[List[FusionRule]] = None,
        custom_validity_checks: Optional[List[ValidityCheck]] = None,
    ) -> None:
        """
        Compile the pipeline.

        Parameters
        ----------
        custom_fusion_rules : Optional[List[FusionRule]], optional
            Additional fusion rules to apply during compilation. Default is None.
        custom_validity_checks : Optional[List[ValidityCheck]], optional
            Additional validity checks to apply during compilation. Default is None.
        """
        try:
            self.compiled_ops = compile_pipeline(
                self.ops, custom_fusion_rules, custom_validity_checks
            )
        except CompileError as e:
            raise ValueError(f"Compilation failed: {e}") from e
