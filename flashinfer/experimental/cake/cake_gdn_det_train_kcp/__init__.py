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
# imports itself, torch and torch.distributed only (never a KDA package) and
# reads no environment variable.  It exports the names that carry ``det`` only:
# the constants and backend controls without ``det`` (``SCALE``, ``LAST_CALL``,
# ``set_backend``, ``make_options``, ...) stay in their modules (``cake_kcp``,
# ``cake_options``).

from .cake_kcp import (
    CallDetKcp,
    ChunkGDNDetFunctionKcp,
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
    packed_qkv_views_det_kcp,
)
from .cake_kcp_plan import (
    ContextDetKcp,
    FragmentDetKcp,
    PartitionDetKcp,
    build_context_det_kcp,
    partition_det_kcp,
)
from .cake_options import OptionsDetKcp

__all__ = [
    "CallDetKcp",
    "ChunkGDNDetFunctionKcp",
    "ContextDetKcp",
    "FragmentDetKcp",
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
    "packed_qkv_views_det_kcp",
    "partition_det_kcp",
]
