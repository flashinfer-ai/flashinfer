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

"""Runnable solution modules for FlashInfer trace definitions.

Each definition directory contains one or more solution modules.  Benchmark
harnesses can import modules directly, e.g.
``flashinfer.trace.solutions.rmsnorm.native``, or use the discovery helpers
below to enumerate available solutions.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Iterator, List, Optional

_SOLUTIONS_ROOT = Path(__file__).resolve().parent


def iter_solution_modules(definition: Optional[str] = None) -> Iterator[ModuleType]:
    """Yield importable solution modules.

    Parameters
    ----------
    definition:
        Optional definition name.  When set, only modules under
        ``flashinfer.trace.solutions.<definition>`` are yielded.
    """

    if definition is None:
        for package_dir in sorted(_SOLUTIONS_ROOT.iterdir()):
            if package_dir.is_dir() and not package_dir.name.startswith("_"):
                yield from iter_solution_modules(package_dir.name)
        return

    package_prefix = f"{__name__}.{definition}"
    package = importlib.import_module(package_prefix)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg or module_info.name.startswith("_"):
            continue
        yield importlib.import_module(f"{package_prefix}.{module_info.name}")


def load_solutions(definition: str) -> List[ModuleType]:
    """Return all solution modules for *definition*."""

    return list(iter_solution_modules(definition))


__all__ = ["iter_solution_modules", "load_solutions"]
