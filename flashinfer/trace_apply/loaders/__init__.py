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

"""Solution loaders, dispatched by language *family* (python vs C++/CUDA).

The granular language (python/triton/cutedsl/cuda/cutlass/…) is metadata; for
*loading* there are only two shapes — a Python module to import, or an extension
to compile. ``BuildSpec.is_python_family`` makes that 2-way decision, which also
fixes the calling convention (C++/CUDA → positional, Python → keyword).
"""

from __future__ import annotations

from typing import Callable

from flashinfer.trace.solution import Solution
from flashinfer.trace_apply.loaders import cpp, python


def load(solution: Solution) -> Callable:
    """Materialize a Solution into a callable, dispatched by language family."""
    if solution.spec.is_python_family:
        return python.load(solution)
    return cpp.load(solution)


__all__ = ["cpp", "load", "python"]
