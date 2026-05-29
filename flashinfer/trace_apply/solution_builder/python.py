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

"""Build in-process Python callables from Solution objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from ...trace.solution import Solution, SupportedLanguages


_SUPPORTED_LANGUAGES = {
    SupportedLanguages.PYTHON,
    SupportedLanguages.TRITON,
    SupportedLanguages.CUDA,
    SupportedLanguages.TILELANG,
}


@dataclass(frozen=True)
class BuiltSolution:
    """Callable extracted from a Solution object."""

    fn: Callable[..., Any]
    definition: str
    destination_passing_style: bool


def load_solution_object(solution: Solution) -> BuiltSolution:
    """Load a minimal flashinfer-bench Solution object.

    Only the smallest no-build subset is supported:

    - ``spec.language`` is any supported language except ``"cpp"``
    - ``spec.entry_point == "<file_path>::<function_name>"``
    - ``sources`` contains one file object with that path and source content
    """

    if solution.spec.language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported apply solution language: {solution.spec.language.value!r}"
        )

    entry_point = solution.spec.entry_point
    entry_name = solution.get_entry_symbol()
    source = solution.get_entry_source()
    if source is None:
        entry_path = solution.spec.entry_point.split("::", 1)[0]
        raise ValueError(f"Python apply solution missing source file {entry_path}")

    namespace: dict[str, Any] = {"torch": torch}
    exec(source.content, namespace)  # noqa: S102 - trusted user-provided solution code
    fn: Any = namespace
    for part in entry_name.split("."):
        fn = fn[part] if isinstance(fn, dict) else getattr(fn, part)
    if not callable(fn):
        raise TypeError(f"Entry point '{entry_point}' does not resolve to a callable")

    return BuiltSolution(
        fn=fn,
        definition=solution.definition,
        destination_passing_style=solution.spec.destination_passing_style,
    )
