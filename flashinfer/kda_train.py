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

Packed KDA (Kimi Delta Attention) chunked training (SM100 / SM103 / SM107), the Cake backend: the
context-parallel (KCP) entry points, and the single-rank entry points under their previous names.

The single-rank entry points live in ``flashinfer.cake_kda_det_train`` (``chunk_kda_det_fwd``,
``chunk_kda_det_bwd``, ``chunk_kda_det``, ``kda_det_train_arena``, ``kda_det_train_workspace``,
``kda_det_train_workspace_bytes``, ``kda_det_train_saved_bytes``, ``is_kda_det_train_supported``,
``DEFAULT_SCALE``).  This module binds their previous names (``chunk_kda_fwd``, ``chunk_kda_bwd``,
``chunk_kda``, ``kda_train_arena``, ``kda_train_workspace``, ``kda_train_workspace_bytes``,
``kda_train_saved_bytes``, ``is_kda_train_supported``) to the same objects.  The host plan, the
workspace, the launch sequences, the autograd Functions and the generated kernels live in
``flashinfer.experimental.cake_kda_det_train``.

Context-parallel training (KCP) lives in ``flashinfer.cake_kda_det_train_kcp``.  This module binds
the names the KCP entry points and types had before to the same objects:

======================================  =======================================
previous name                           det name
======================================  =======================================
``kcp_partition``                       ``partition_det_kcp``
``build_kcp_context``                   ``build_context_det_kcp``
``chunk_kda_kcp``                       ``chunk_kda_det_kcp``
``chunk_kda_kcp_fwd``                   ``chunk_kda_det_fwd_kcp``
``chunk_kda_kcp_bwd``                   ``chunk_kda_det_bwd_kcp``
``kcp_workspace_bytes``                 ``kda_det_workspace_bytes_kcp``
``kcp_saved_bytes``                     ``kda_det_saved_bytes_kcp``
``kcp_comm_bytes``                      ``kda_det_comm_bytes_kcp``
``KcpContext``                          ``ContextDetKcp``
``KcpOptions``                          ``OptionsDetKcp``
``KcpPartition``                        ``PartitionDetKcp``
``KcpRecomputeHandle``                  ``RecomputeHandleDetKcp``
``KcpSaved``                            ``SavedDetKcp``
``KcpWorkspace``                        ``WorkspaceDetKcp``
======================================  =======================================
"""

from __future__ import annotations

from typing import Any

from . import cake_kda_det_train_kcp as _kcp
from .cake_kda_det_train import (
    DEFAULT_SCALE,
    chunk_kda_det,
    chunk_kda_det_bwd,
    chunk_kda_det_fwd,
    is_kda_det_train_supported,
    kda_det_train_arena,
    kda_det_train_saved_bytes,
    kda_det_train_workspace,
    kda_det_train_workspace_bytes,
)

# The single-rank entry points under their previous names: the same objects as the det names.
chunk_kda_fwd = chunk_kda_det_fwd
chunk_kda_bwd = chunk_kda_det_bwd
chunk_kda = chunk_kda_det
kda_train_arena = kda_det_train_arena
kda_train_workspace = kda_det_train_workspace
kda_train_workspace_bytes = kda_det_train_workspace_bytes
kda_train_saved_bytes = kda_det_train_saved_bytes
is_kda_train_supported = is_kda_det_train_supported


# --------------------------------------------------------------------------- context-parallel training (KCP)
#
# The KCP entry points under their previous names: the same objects as those of ``flashinfer.cake_kda_det_train_kcp``.
kcp_partition = _kcp.partition_det_kcp
build_kcp_context = _kcp.build_context_det_kcp
chunk_kda_kcp_fwd = _kcp.chunk_kda_det_fwd_kcp
chunk_kda_kcp_bwd = _kcp.chunk_kda_det_bwd_kcp
chunk_kda_kcp = _kcp.chunk_kda_det_kcp
kcp_workspace_bytes = _kcp.kda_det_workspace_bytes_kcp
kcp_saved_bytes = _kcp.kda_det_saved_bytes_kcp
kcp_comm_bytes = _kcp.kda_det_comm_bytes_kcp

# The KCP types under their previous names, resolved on first access through ``flashinfer.cake_kda_det_train_kcp``
# (which imports the KCP host module lazily): previous name -> det name.
_LAZY_KCP_TYPES = {
    "KcpContext": "ContextDetKcp",
    "KcpOptions": "OptionsDetKcp",
    "KcpPartition": "PartitionDetKcp",
    "KcpRecomputeHandle": "RecomputeHandleDetKcp",
    "KcpSaved": "SavedDetKcp",
    "KcpWorkspace": "WorkspaceDetKcp",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_KCP_TYPES:
        return getattr(_kcp, _LAZY_KCP_TYPES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_KCP_TYPES))


__all__ = [
    "DEFAULT_SCALE",
    "KcpRecomputeHandle",  # noqa: F822 - resolved by __getattr__
    "build_kcp_context",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "chunk_kda_kcp",
    "chunk_kda_kcp_bwd",
    "chunk_kda_kcp_fwd",
    "is_kda_train_supported",
    "kcp_comm_bytes",
    "kcp_partition",
    "kcp_saved_bytes",
    "kcp_workspace_bytes",
    "kda_train_arena",
    "kda_train_saved_bytes",
    "kda_train_workspace",
    "kda_train_workspace_bytes",
]
