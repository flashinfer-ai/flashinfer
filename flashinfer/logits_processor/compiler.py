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

from typing import List, Optional

from .fusion_rules import FusionRule, get_default_fusion_rules
from .op import Op
from .types import TensorType
from .validators import CompileError, ValidityCheck, get_default_validity_checks


class Compiler:
    def __init__(self) -> None:
        self.fusion_rules: List[FusionRule] = []
        self.validity_checks: List[ValidityCheck] = []
        self._install_defaults()

    def register_fusion_rule(self, rule: FusionRule) -> None:
        self.fusion_rules.append(rule)
        self.fusion_rules.sort(key=lambda r: -r.prio)

    def register_validity_check(self, check: ValidityCheck) -> None:
        self.validity_checks.append(check)

    def compile(self, ops: List[Op]) -> List[Op]:
        if not ops:
            raise CompileError("Cannot compile empty operator list")

        compiled_ops = list(ops)

        self._type_check(compiled_ops)

        self._run_validity_checks(compiled_ops)

        compiled_ops = self._fuse_all(compiled_ops)

        return compiled_ops

    def _type_check(self, ops: List[Op]) -> None:
        first_op = ops[0]

        current_type = first_op.IN

        if current_type not in [TensorType.LOGITS, TensorType.PROBS]:
            raise CompileError(
                f"First operator ({first_op.__class__.__name__}) cannot accept standard pipeline inputs. "
                f"Expected LOGITS or PROBS, but operator accepts: {first_op.IN}"
            )

        for i, op in enumerate(ops):
            if current_type != op.IN:
                raise CompileError(
                    f"Type mismatch at operator {i} ({op.__class__.__name__}). "
                    f"Expected input type: {current_type}, but operator accepts: {op.IN}. "
                    f"Previous operator output: {current_type}"
                )

            current_type = op.OUT

    def _run_validity_checks(self, ops: List[Op]) -> None:
        for check in self.validity_checks:
            check(ops)

    def _fuse_all(self, ops: List[Op]) -> List[Op]:
        i = 0
        while i < len(ops):
            fusion_applied = False

            for rule in self.fusion_rules:
                span = len(rule.pattern)

                if i + span > len(ops):
                    continue

                window = ops[i : i + span]

                if self._pattern_matches(window, rule.pattern) and rule.guard(window):
                    fused_op = rule.build(window)
                    ops[i : i + span] = [fused_op]

                    i = max(i - 1, 0)
                    fusion_applied = True
                    break

            if not fusion_applied:
                i += 1

        return ops

    def _pattern_matches(self, window: List[Op], pattern: tuple) -> bool:
        if len(window) != len(pattern):
            return False

        return all(isinstance(window[i], pattern[i]) for i in range(len(pattern)))

    def _install_defaults(self) -> None:
        for check in get_default_validity_checks():
            self.validity_checks.append(check)

        for rule in get_default_fusion_rules():
            self.register_fusion_rule(rule)


def compile_pipeline(
    ops: List[Op],
    custom_fusion_rules: Optional[List[FusionRule]] = None,
    custom_validity_checks: Optional[List[ValidityCheck]] = None,
) -> List[Op]:
    """
    Compile a pipeline of operators into a list of compiled operators.

    Parameters
    ----------
    ops : List[Op]
        List of operators to compile
    custom_fusion_rules : Optional[List[FusionRule]]
        List of custom fusion rules to use
    custom_validity_checks : Optional[List[ValidityCheck]]
        List of custom validity checks to use

    Returns
    -------
    List[Op]
        List of compiled operators
    """
    compiler = Compiler()

    if custom_fusion_rules:
        for rule in custom_fusion_rules:
            compiler.register_fusion_rule(rule)

    if custom_validity_checks:
        for check in custom_validity_checks:
            compiler.register_validity_check(check)

    return compiler.compile(ops)
