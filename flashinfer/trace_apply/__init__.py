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

Enable by pointing ``FLASHINFER_TRACE_APPLY=1`` and ``FLASHINFER_TRACE_PATH`` at
a folder of solution files, or call :func:`enable_apply` explicitly.
"""

from __future__ import annotations

from flashinfer.trace_apply.apply import (
    author_stats_snapshot,
    current_sm,
    disable_apply,
    enable_apply,
    enable_apply_from_env,
    get_index,
    get_policy,
    is_enabled,
    reset_stats,
    stats_snapshot,
)
from flashinfer.trace_apply.config import ApplyPolicy


def stats() -> dict:
    """Per-(fi_api, status) dispatch counts: hit, fallback_no_candidate,
    fallback_unwarmed_in_capture, error (strict: a matched solution that
    failed to load/run — re-raised)."""
    return stats_snapshot()


def explain(fi_api: str, const_axes: dict, input_dtypes=None, sm_arch: str | None = None) -> dict:
    """Show how a ``(fi_api, const_axes, input_dtypes)`` lookup resolves against
    the installed routing table. For debugging which solution would dispatch."""
    from flashinfer.trace_apply.routing import explain as _explain  # noqa: PLC0415

    idx = get_index()
    if idx is None:
        raise RuntimeError("Trace Apply is not enabled. Call enable_apply() first.")
    return _explain(
        idx,
        fi_api,
        dict(const_axes),
        frozenset(input_dtypes or ()),
        sm_arch or current_sm(),
        get_policy(),
    )


__all__ = [
    "enable_apply",
    "disable_apply",
    "enable_apply_from_env",
    "is_enabled",
    "ApplyPolicy",
    "stats",
    "explain",
    "author_stats_snapshot",
    "reset_stats",
]
