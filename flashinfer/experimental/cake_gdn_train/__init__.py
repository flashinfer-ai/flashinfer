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

# Legacy import path of the deterministic context-parallel GDN training package: the
# canonical package is ``flashinfer.experimental.cake_gdn_det_train_kcp``.  This package
# holds no code, sources or manifests of its own: it re-exports the canonical package's
# public names and the pre-rename compatibility aliases (the same objects), and each
# submodule (``cake_kcp_plan``, ...) is the canonical submodule object itself.

from ..cake_gdn_det_train_kcp import *  # noqa: F403
from ..cake_gdn_det_train_kcp import __all__ as __all__
from ..cake_gdn_det_train_kcp import (  # noqa: F401 - compatibility aliases
    ChunkGDNKcpFunction,
    KcpCall,
    KcpContext,
    KcpFragment,
    KcpOptions,
    KcpPartition,
    KcpRecomputeHandle,
    KcpSaved,
    KcpWorkspace,
    build_kcp_context,
    chunk_gdn_kcp,
    chunk_gdn_kcp_bwd,
    chunk_gdn_kcp_fwd,
    gdn_kcp_comm_bytes,
    gdn_kcp_saved_bytes,
    gdn_kcp_workspace_bytes,
    kcp_partition,
)
