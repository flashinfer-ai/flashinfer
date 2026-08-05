# Copyright (c) 2026 by FlashInfer team.
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

"""Require the installed CUTLASS DSL wheel for Prims-TS kernels."""

from __future__ import annotations

import importlib

_BOOTSTRAPPED = False
_BOOTSTRAP_ERROR: BaseException | None = None


def _has_required_modules() -> bool:
    required = (
        "cutlass",
        "cutlass.cute",
        "cutlass.experimental.primitives",
        "cutlass.experimental.task_scheduling",
    )
    for name in required:
        importlib.import_module(name)
    return True


def ensure_cutlass_dsl_experimental() -> bool:
    """Return whether Prims-TS dependencies are importable from installed wheels."""

    global _BOOTSTRAPPED, _BOOTSTRAP_ERROR

    if _BOOTSTRAPPED:
        return True
    if _BOOTSTRAP_ERROR is not None:
        return False

    try:
        importlib.invalidate_caches()
        _has_required_modules()
    except BaseException as exc:
        _BOOTSTRAP_ERROR = exc
        return False

    _BOOTSTRAPPED = True
    return True


def require_cutlass_dsl_experimental() -> None:
    if ensure_cutlass_dsl_experimental():
        return
    raise RuntimeError(
        "Prims-TS requires the CUTLASS DSL wheel. Install the pinned "
        "release-branch wheel before importing flashinfer.prims_ts."
    ) from _BOOTSTRAP_ERROR


def get_cutlass_dsl_bootstrap_error() -> BaseException | None:
    return _BOOTSTRAP_ERROR
