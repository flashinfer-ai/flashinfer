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

from typing import List

from .op import Op
from .processors import LogitsProcessor
from .types import TensorType


class LegalizationError(Exception):
    pass


def legalize_processors(
    processors: List[LogitsProcessor], initial_type: TensorType = TensorType.LOGITS
) -> List[Op]:
    """
    Transform high-level LogitsProcessors into low-level Ops.

    This is the legalization stage that converts:
    Logits -> [TopK(), Sampling()]
    into:
    Logits -> [TopKLogitsOp(), SampleLogitsOp()]

    Args:
        processors: List of high-level processors
        initial_type: The initial input tensor type (LOGITS or PROBS)

    Returns:
        List of low-level Ops
    """
    if not processors:
        raise LegalizationError("Cannot legalize empty processor list")

    ops = []
    current_type = initial_type

    for i, processor in enumerate(processors):
        try:
            legalized_ops = processor.legalize(current_type)

            if not legalized_ops:
                raise LegalizationError(
                    f"Processor {processor.__class__.__name__} produced no ops"
                )

            ops.extend(legalized_ops)

            current_type = legalized_ops[-1].OUT

        except Exception as e:
            raise LegalizationError(
                f"Failed to legalize processor {i} ({processor.__class__.__name__}): {e}"
            ) from e

    return ops


def infer_initial_type(processors: List[LogitsProcessor]) -> TensorType:
    if not processors:
        return TensorType.LOGITS

    first_processor = processors[0]
    valid_types = _get_supported_types(first_processor)

    if len(valid_types) > 1:
        raise LegalizationError(
            f"Cannot infer input type: {first_processor.__class__.__name__} can accept both LOGITS and PROBS. "
            f"Please specify input_type explicitly when creating the LogitsPipe."
        )

    if len(valid_types) == 1:
        return valid_types[0]

    raise LegalizationError(
        f"Processor {first_processor.__class__.__name__} cannot accept standard pipeline inputs "
        f"(LOGITS or PROBS)"
    )


def _get_supported_types(
    processor: LogitsProcessor,
) -> List[TensorType]:
    valid_types = []
    for tensor_type in [TensorType.LOGITS, TensorType.PROBS]:
        try:
            processor.legalize(tensor_type)
            valid_types.append(tensor_type)
        except (ValueError, LegalizationError):
            continue

    return valid_types


def validate_processor_chain(processors: List[LogitsProcessor]) -> None:
    if not processors:
        raise LegalizationError("Processor chain cannot be empty")

    initial_type = infer_initial_type(processors)

    try:
        legalize_processors(processors, initial_type)
    except LegalizationError:
        raise
    except Exception as e:
        raise LegalizationError(f"Processor chain validation failed: {e}") from e
