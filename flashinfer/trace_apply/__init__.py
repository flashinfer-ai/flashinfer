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

"""Runtime kernel substitution for fi_trace definitions.

Call :func:`enable_apply` with a ``{definition_name: callable_or_Solution}```
mapping, or set ``FLASHINFER_TRACE_APPLY=1`` (+ ``FLASHINFER_TRACE_APPLY_PATH``) so
the import-time hook enables it automatically. Sealed kernel packages —
digest-pinned package directories verified with a sha256 chain before
anything is imported — load through :mod:`flashinfer.trace_apply.packages`,
also automatically at import via ``FLASHINFER_KERNEL_PACKAGES``.
"""

from __future__ import annotations

from flashinfer.trace_apply.apply import (
    clear_preloaded,
    current_sm,
    disable_apply,
    enable_apply,
    is_enabled,
    preloaded_entry,
    register_preloaded,
    stats_snapshot,
)

# Internal: called only by the flashinfer package import (see flashinfer/__init__.py).
from flashinfer.trace_apply.apply import _enable_apply_from_env  # noqa: F401
from flashinfer.trace_apply.packages import (
    DEFAULT_PACKAGE_ROOT,
    DISABLE_ENV,
    PACKAGES_ENV,
    ROOT_ENV,
    SealedPackage,
    SealedPackageError,
    import_entry,
    load_packages,
    verify_package,
)

# Internal: called only by the flashinfer package import.
from flashinfer.trace_apply.packages import _load_packages_from_env  # noqa: F401


def stats() -> dict:
    """Per-(fi_api, status) dispatch counts: hit, fallback_no_candidate,
    fallback_unwarmed_in_capture, error (strict: a matched solution that
    failed to load/run — re-raised)."""
    return stats_snapshot()


__all__ = [
    "DEFAULT_PACKAGE_ROOT",
    "DISABLE_ENV",
    "PACKAGES_ENV",
    "ROOT_ENV",
    "SealedPackage",
    "SealedPackageError",
    "clear_preloaded",
    "disable_apply",
    "enable_apply",
    "import_entry",
    "is_enabled",
    "load_packages",
    "preloaded_entry",
    "register_preloaded",
    "stats",
    "verify_package",
]
