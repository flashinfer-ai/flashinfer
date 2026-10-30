"""
Copyright (c) 2026 by FlashInfer team.

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

"""Canonical entry points for ``cake_kda_det_train_kcp``.

Determinism requires fixed inputs, hardware, execution plan and initial
accumulation buffers. Existing operation names and signatures are preserved.
"""

# Public names are resolved lazily by __getattr__.
# ruff: noqa: F822

from . import kda_train as _impl

__all__ = [
    "chunk_kda_kcp",
    "chunk_kda_kcp_fwd",
    "chunk_kda_kcp_bwd",
    "kcp_partition",
    "build_kcp_context",
    "kcp_workspace_bytes",
    "kcp_saved_bytes",
    "kcp_comm_bytes",
    "KcpRecomputeHandle",
]


def __getattr__(name):
    if name in __all__:
        return getattr(_impl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
