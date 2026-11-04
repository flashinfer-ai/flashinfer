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

Legacy entry point of the deterministic context-parallel (KCP) packed GDN
training route.  The canonical module is :mod:`flashinfer.cake_gdn_det_train_kcp`;
every name here is a compatibility alias bound to the same object there
(``chunk_gdn_kcp is cake_gdn_det_train_kcp.chunk_gdn_det_kcp``).  The types
resolve lazily, as in the canonical module, so importing this module does not
import the backend package.

======================================  =======================================
legacy name                             canonical name
======================================  =======================================
``kcp_partition``                       ``partition_det_kcp``
``build_kcp_context``                   ``build_context_det_kcp``
``chunk_gdn_kcp``                       ``chunk_gdn_det_kcp``
``chunk_gdn_kcp_fwd``                   ``chunk_gdn_det_fwd_kcp``
``chunk_gdn_kcp_bwd``                   ``chunk_gdn_det_bwd_kcp``
``gdn_kcp_workspace_bytes``             ``gdn_det_workspace_bytes_kcp``
``gdn_kcp_saved_bytes``                 ``gdn_det_saved_bytes_kcp``
``gdn_kcp_comm_bytes``                  ``gdn_det_comm_bytes_kcp``
``is_gdn_train_supported``              ``is_gdn_det_train_kcp_supported``
``KcpOptions``                          ``OptionsDetKcp``
``KcpContext``                          ``ContextDetKcp``
``KcpPartition``                        ``PartitionDetKcp``
``KcpRecomputeHandle``                  ``RecomputeHandleDetKcp``
``KcpSaved``                            ``SavedDetKcp``
``KcpWorkspace``                        ``WorkspaceDetKcp``
======================================  =======================================
"""

from __future__ import annotations

from typing import Any

from . import cake_gdn_det_train_kcp as _canonical
from .cake_gdn_det_train_kcp import DEFAULT_SCALE  # noqa: F401 - the legacy module constant

# Compatibility aliases: the legacy function names bound to the canonical functions.
kcp_partition = _canonical.partition_det_kcp
build_kcp_context = _canonical.build_context_det_kcp
chunk_gdn_kcp_fwd = _canonical.chunk_gdn_det_fwd_kcp
chunk_gdn_kcp_bwd = _canonical.chunk_gdn_det_bwd_kcp
chunk_gdn_kcp = _canonical.chunk_gdn_det_kcp
gdn_kcp_workspace_bytes = _canonical.gdn_det_workspace_bytes_kcp
gdn_kcp_saved_bytes = _canonical.gdn_det_saved_bytes_kcp
gdn_kcp_comm_bytes = _canonical.gdn_det_comm_bytes_kcp
is_gdn_train_supported = _canonical.is_gdn_det_train_kcp_supported

# Compatibility aliases of the types: legacy name -> canonical name, resolved through the canonical module on first
# access (which imports the backend package lazily).
_LAZY_TYPES = {
    "KcpOptions": "OptionsDetKcp",
    "KcpContext": "ContextDetKcp",
    "KcpPartition": "PartitionDetKcp",
    "KcpRecomputeHandle": "RecomputeHandleDetKcp",
    "KcpSaved": "SavedDetKcp",
    "KcpWorkspace": "WorkspaceDetKcp",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_TYPES:
        return getattr(_canonical, _LAZY_TYPES[name])
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))


__all__ = [
    "KcpContext",  # noqa: F822 - resolved by __getattr__
    "KcpOptions",  # noqa: F822
    "KcpPartition",  # noqa: F822
    "KcpRecomputeHandle",  # noqa: F822
    "build_kcp_context",
    "chunk_gdn_kcp",
    "chunk_gdn_kcp_bwd",
    "chunk_gdn_kcp_fwd",
    "gdn_kcp_comm_bytes",
    "gdn_kcp_saved_bytes",
    "gdn_kcp_workspace_bytes",
    "is_gdn_train_supported",
    "kcp_partition",
]
