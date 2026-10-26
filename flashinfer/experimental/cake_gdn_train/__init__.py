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

# Experimental Cake backend: context-parallel (KCP) packed GDN (Gated DeltaNet)
# chunked training with precise cross-rank state passing, native grouped Q/K
# heads (``HV = r * H_qk``), the FP32 scalar gate and FP32 beta, per-document
# zigzag / ragged partitions (head dimension 128, chunk 64) on SM100 / SM103 /
# SM107.  The public entry points are ``flashinfer.gdn_train.kcp_partition``,
# ``build_kcp_context``, ``chunk_gdn_kcp``, ``chunk_gdn_kcp_fwd``,
# ``chunk_gdn_kcp_bwd`` and the byte queries; the partition / context plan, the
# host orchestration, the options, the registry and the generated sources live
# here.  This package imports itself, torch and torch.distributed only (never
# ``cake_kda_train``) and reads no environment variable.

from .cake_kcp import (
    KCP_SAVED_VERSION,
    LAST_CALL,
    PROGRAMS,
    SCALE,
    Backend,
    ChunkGDNKcpFunction,
    FakeBackend,
    KcpCall,
    KcpSaved,
    KcpWorkspace,
    KernelBackend,
    chunk_gdn_kcp,
    chunk_gdn_kcp_bwd,
    chunk_gdn_kcp_fwd,
    gdn_kcp_comm_bytes,
    gdn_kcp_saved_bytes,
    gdn_kcp_workspace_bytes,
    get_backend,
    graph_cache_clear,
    set_backend,
)
from .cake_kcp_plan import (
    KcpContext,
    KcpFragment,
    KcpPartition,
    build_kcp_context,
    kcp_partition,
)
from .cake_options import KcpOptions, make_options, validate_heads, validate_ring

__all__ = [
    "KCP_SAVED_VERSION",
    "LAST_CALL",
    "PROGRAMS",
    "SCALE",
    "Backend",
    "ChunkGDNKcpFunction",
    "FakeBackend",
    "KcpCall",
    "KcpContext",
    "KcpFragment",
    "KcpOptions",
    "KcpPartition",
    "KcpSaved",
    "KcpWorkspace",
    "KernelBackend",
    "build_kcp_context",
    "chunk_gdn_kcp",
    "chunk_gdn_kcp_bwd",
    "chunk_gdn_kcp_fwd",
    "gdn_kcp_comm_bytes",
    "gdn_kcp_saved_bytes",
    "gdn_kcp_workspace_bytes",
    "get_backend",
    "graph_cache_clear",
    "kcp_partition",
    "make_options",
    "set_backend",
    "validate_heads",
    "validate_ring",
]
