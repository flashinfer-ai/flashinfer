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

# Experimental Cake backend: deterministic context-parallel (KCP) packed GDN
# (Gated DeltaNet) chunked training with precise cross-rank state passing, native
# grouped Q/K heads (``HV = r * H_qk``), the FP32 scalar gate and FP32 beta,
# per-document zigzag / ragged partitions (head dimension 128, chunk 64) on
# SM100 / SM103 / SM107.  The public entry points are
# ``flashinfer.cake_gdn_det_train_kcp.partition_det_kcp``, ``build_context_det_kcp``,
# ``chunk_gdn_det_kcp``, ``chunk_gdn_det_fwd_kcp``, ``chunk_gdn_det_bwd_kcp``,
# ``packed_qkv_views_det_kcp`` and the byte queries; the partition / context plan, the host orchestration, the
# options, the registry and the generated sources live here.  This package
# imports itself, torch and torch.distributed only (never ``cake_kda_train``) and
# reads no environment variable.  The pre-rename names (``chunk_gdn_kcp``,
# ``KcpContext``, ...) remain importable as compatibility aliases of the same
# objects; ``flashinfer.experimental.cake_gdn_train`` is the legacy import path.

from .cake_kcp import (
    KCP_SAVED_VERSION,
    LAST_CALL,
    PROGRAMS,
    SCALE,
    Backend,
    CallDetKcp,
    ChunkGDNDetFunctionKcp,
    FakeBackend,
    KernelBackend,
    PackedQKVSplitDetKcp,
    RecomputeHandleDetKcp,
    SavedDetKcp,
    WorkspaceDetKcp,
    chunk_gdn_det_bwd_kcp,
    chunk_gdn_det_fwd_kcp,
    chunk_gdn_det_kcp,
    gdn_det_comm_bytes_kcp,
    gdn_det_saved_bytes_kcp,
    gdn_det_workspace_bytes_kcp,
    get_backend,
    graph_cache_clear,
    packed_qkv_views_det_kcp,
    set_backend,
)
from .cake_kcp_plan import (
    ContextDetKcp,
    FragmentDetKcp,
    PartitionDetKcp,
    build_context_det_kcp,
    partition_det_kcp,
)
from .cake_options import OptionsDetKcp, make_options, validate_heads, validate_ring

__all__ = [
    "KCP_SAVED_VERSION",
    "LAST_CALL",
    "PROGRAMS",
    "SCALE",
    "Backend",
    "CallDetKcp",
    "ChunkGDNDetFunctionKcp",
    "ContextDetKcp",
    "FakeBackend",
    "FragmentDetKcp",
    "KernelBackend",
    "OptionsDetKcp",
    "PackedQKVSplitDetKcp",
    "PartitionDetKcp",
    "RecomputeHandleDetKcp",
    "SavedDetKcp",
    "WorkspaceDetKcp",
    "build_context_det_kcp",
    "chunk_gdn_det_bwd_kcp",
    "chunk_gdn_det_fwd_kcp",
    "chunk_gdn_det_kcp",
    "gdn_det_comm_bytes_kcp",
    "gdn_det_saved_bytes_kcp",
    "gdn_det_workspace_bytes_kcp",
    "get_backend",
    "graph_cache_clear",
    "make_options",
    "packed_qkv_views_det_kcp",
    "partition_det_kcp",
    "set_backend",
    "validate_heads",
    "validate_ring",
]

# Compatibility aliases: the pre-rename names bound to the same objects (not in ``__all__``).
ChunkGDNKcpFunction = ChunkGDNDetFunctionKcp
KcpCall = CallDetKcp
KcpContext = ContextDetKcp
KcpFragment = FragmentDetKcp
KcpOptions = OptionsDetKcp
KcpPartition = PartitionDetKcp
KcpRecomputeHandle = RecomputeHandleDetKcp
KcpSaved = SavedDetKcp
KcpWorkspace = WorkspaceDetKcp
build_kcp_context = build_context_det_kcp
chunk_gdn_kcp = chunk_gdn_det_kcp
chunk_gdn_kcp_bwd = chunk_gdn_det_bwd_kcp
chunk_gdn_kcp_fwd = chunk_gdn_det_fwd_kcp
gdn_kcp_comm_bytes = gdn_det_comm_bytes_kcp
gdn_kcp_saved_bytes = gdn_det_saved_bytes_kcp
gdn_kcp_workspace_bytes = gdn_det_workspace_bytes_kcp
kcp_partition = partition_det_kcp
