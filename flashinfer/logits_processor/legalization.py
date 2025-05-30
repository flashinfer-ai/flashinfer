from typing import List, Union

from .op import Op
from .processors import LogitsProcessor
from .types import Sort


class LegalizationError(Exception):
    pass


def legalize_processors(
    processors: List[LogitsProcessor], initial_sort: Sort = Sort.LOGITS
) -> List[Op]:
    """
    Transform high-level LogitsProcessors into low-level Ops.

    This is the legalization stage that converts:
    Logits -> [TopK(), Sampling()]
    into:
    Logits -> [LogitsTopKOp(), LogitsSamplingOp()]

    Args:
        processors: List of high-level processors
        initial_sort: The initial sort type (LOGITS or PROBS)

    Returns:
        List of low-level Ops
    """
    if not processors:
        raise LegalizationError("Cannot legalize empty processor list")

    ops = []
    current_sort = initial_sort

    for i, processor in enumerate(processors):
        try:
            legalized_ops = processor.legalize(current_sort)

            if not legalized_ops:
                raise LegalizationError(
                    f"Processor {processor.__class__.__name__} produced no ops"
                )

            ops.extend(legalized_ops)

            current_sort = legalized_ops[-1].OUT

        except Exception as e:
            raise LegalizationError(
                f"Failed to legalize processor {i} ({processor.__class__.__name__}): {e}"
            ) from e

    return ops


def infer_initial_sort(processors: List[LogitsProcessor]) -> Sort:
    if not processors:
        return Sort.LOGITS

    first_processor = processors[0]

    for candidate_sort in [Sort.LOGITS, Sort.PROBS]:
        try:
            first_processor.legalize(candidate_sort)
            return candidate_sort
        except (ValueError, LegalizationError):
            continue

    return Sort.LOGITS


def validate_processor_chain(processors: List[LogitsProcessor]) -> None:
    if not processors:
        raise LegalizationError("Processor chain cannot be empty")

    initial_sort = infer_initial_sort(processors)

    try:
        legalize_processors(processors, initial_sort)
    except LegalizationError:
        raise
    except Exception as e:
        raise LegalizationError(f"Processor chain validation failed: {e}") from e
