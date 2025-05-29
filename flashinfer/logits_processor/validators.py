from typing import List, Callable

from .op import Op
from .types import Sort
from .operators import Softmax, TopP


class CompileError(Exception):
    pass


ValidityCheck = Callable[[List[Op]], None]


def single_softmax_rule(ops: List[Op]) -> None:
    """
    R1: Single-Softmax rule.
    
    Softmax appears ≤ 1 time in the pipeline.
    """
    softmax_count = sum(1 for op in ops if isinstance(op, Softmax))
    if softmax_count > 1:
        raise CompileError("Multiple Softmax operators found. Only one Softmax is allowed per pipeline.")


# Disabled since we allow PROBS inputs to TopP, the input type is already guarded by the compiler
# def topp_after_softmax_rule(ops: List[Op]) -> None:
#     """
#     R2: TopP-after-Softmax rule.
    
#     Every TopP must be preceded (anywhere earlier) by a Softmax.
#     """
#     seen_softmax = False
    
#     for op in ops:
#         if isinstance(op, Softmax):
#             seen_softmax = True
#         elif isinstance(op, TopP) and not seen_softmax:
#             raise CompileError(
#                 "TopP operator requires a preceding Softmax operator. "
#                 "TopP can only operate on probabilities, not logits."
#             )


def indices_terminal_rule(ops: List[Op]) -> None:
    """
    R3': Indices-terminal rule.
    
    If an operator outputs Indices, no operator may follow it.
    """
    for i, op in enumerate(ops[:-1]):  # Check all but the last operator
        if Sort.INDICES in op.OUT:
            next_op = ops[i + 1]
            raise CompileError(
                f"No operator may follow one that outputs Indices. "
                f"Found {next_op.__class__.__name__} after {op.__class__.__name__} "
                f"which outputs Indices."
            )


def get_default_validity_checks() -> List[ValidityCheck]:
    return [
        single_softmax_rule,
        # topp_after_softmax_rule,
        indices_terminal_rule,
    ]


def validate_pipeline(ops: List[Op], custom_checks: List[ValidityCheck] = None) -> None:
    if not ops:
        raise CompileError("Pipeline cannot be empty")
    
    for check in get_default_validity_checks():
        check(ops)
    
    if custom_checks:
        for check in custom_checks:
            check(ops) 