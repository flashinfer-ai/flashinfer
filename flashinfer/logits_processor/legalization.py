from typing import List, Union

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

    for candidate_type in [TensorType.LOGITS, TensorType.PROBS]:
        try:
            first_processor.legalize(candidate_type)
            return candidate_type
        except (ValueError, LegalizationError):
            continue

    return TensorType.LOGITS


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
