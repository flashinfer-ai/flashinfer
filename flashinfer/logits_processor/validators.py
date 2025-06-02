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

from typing import Callable, List

from .op import Op
from .operators import SoftmaxOp
from .types import TensorType


class CompileError(Exception):
    pass


ValidityCheck = Callable[[List[Op]], None]


def single_softmax_rule(ops: List[Op]) -> None:
    """
    R1: Single-Softmax rule.

    Softmax appears ≤ 1 time in the pipeline.
    """
    softmax_count = sum(1 for op in ops if isinstance(op, SoftmaxOp))
    if softmax_count > 1:
        raise CompileError(
            "Multiple Softmax operators found. Only one Softmax is allowed per pipeline."
        )


# Disabled since we allow PROBS inputs to TopP, the input type is already guarded by the compiler
# def topp_after_softmax_rule(ops: List[Op]) -> None:
#     """
#     R2: TopP-after-Softmax rule.

#     Every TopP must be preceded (anywhere earlier) by a Softmax.
#     """
#     seen_softmax = False

#     for op in ops:
#         if isinstance(op, SoftmaxOp):
#             seen_softmax = True
#         elif isinstance(op, TopPOp) and not seen_softmax:
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
        if TensorType.INDICES == op.OUT:
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
