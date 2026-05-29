# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serializable configuration for trace apply solution routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..trace.solution import Solution


@dataclass(frozen=True)
class ApplyConfig:
    """Serializable mapping from fi_trace definition names to solutions."""

    solutions: Mapping[str, Solution] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApplyConfig":
        """Build an ApplyConfig from a JSON-compatible dictionary."""

        raw_solutions = data.get("solutions", {})
        if not isinstance(raw_solutions, Mapping):
            raise TypeError("ApplyConfig.solutions must be a mapping")

        solutions: dict[str, Solution] = {}
        for definition_name, raw_solution in raw_solutions.items():
            if not isinstance(definition_name, str) or not definition_name:
                raise ValueError("ApplyConfig solution keys must be non-empty strings")
            if isinstance(raw_solution, Solution):
                solution = raw_solution
            elif isinstance(raw_solution, Mapping):
                solution = Solution.from_dict(raw_solution)
            else:
                raise TypeError(
                    "ApplyConfig solutions must be Solution objects or dictionaries"
                )
            if solution.definition != definition_name:
                raise ValueError(
                    f"Solution key '{definition_name}' does not match "
                    f"solution.definition '{solution.definition}'"
                )
            solutions[definition_name] = solution
        return cls(solutions=solutions)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        return {
            "solutions": {
                definition_name: solution.to_dict()
                for definition_name, solution in self.solutions.items()
            }
        }


__all__ = ["ApplyConfig"]
