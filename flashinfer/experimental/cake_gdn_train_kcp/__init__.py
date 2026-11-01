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

# Experimental Cake backend: non-deterministic context-parallel (KCP) packed GDN
# (Gated DeltaNet) chunked training with precise cross-rank state passing, native
# grouped Q/K heads (``HV = r * H_qk``), the FP32 scalar gate and FP32 beta,
# per-document zigzag / ragged partitions (head dimension 128, chunk 64) on
# SM100 / SM103 / SM107.  The deterministic twin is
# ``flashinfer.experimental.cake_gdn_det_train_kcp``; this package keeps its
# contracts and accuracy budgets, not its bitwise results.  The public
# entry points are ``flashinfer.cake_gdn_train_kcp.partition_kcp``,
# ``build_context_kcp``, ``chunk_gdn_kcp``, ``chunk_gdn_fwd_kcp``,
# ``chunk_gdn_bwd_kcp`` and the byte queries; the partition / context plan, the
# host orchestration, the options, the registry and the generated sources live
# here.  This package imports itself, torch and torch.distributed only (never
# another FlashInfer package, its deterministic twin included) and reads no
# environment variable.

from .cake_kcp import (
    KCP_SAVED_VERSION,
    LAST_CALL,
    MODE,
    PROGRAMS,
    SCALE,
    Backend,
    CallKcp,
    ChunkGDNFunctionKcp,
    FakeBackend,
    KernelBackend,
    RecomputeHandleKcp,
    SavedKcp,
    WorkspaceKcp,
    check_mode,
    chunk_gdn_bwd_kcp,
    chunk_gdn_fwd_kcp,
    chunk_gdn_kcp,
    gdn_comm_bytes_kcp,
    gdn_saved_bytes_kcp,
    gdn_workspace_bytes_kcp,
    get_backend,
    graph_cache_clear,
    set_backend,
)
from .cake_kcp_plan import (
    ContextKcp,
    FragmentKcp,
    PartitionKcp,
    build_context_kcp,
    partition_kcp,
)
from .cake_options import OptionsKcp, make_options, validate_heads, validate_ring

__all__ = [
    "KCP_SAVED_VERSION",
    "LAST_CALL",
    "MODE",
    "PROGRAMS",
    "SCALE",
    "Backend",
    "CallKcp",
    "ChunkGDNFunctionKcp",
    "ContextKcp",
    "FakeBackend",
    "FragmentKcp",
    "KernelBackend",
    "OptionsKcp",
    "PartitionKcp",
    "RecomputeHandleKcp",
    "SavedKcp",
    "WorkspaceKcp",
    "build_context_kcp",
    "check_mode",
    "chunk_gdn_bwd_kcp",
    "chunk_gdn_fwd_kcp",
    "chunk_gdn_kcp",
    "gdn_comm_bytes_kcp",
    "gdn_saved_bytes_kcp",
    "gdn_workspace_bytes_kcp",
    "get_backend",
    "graph_cache_clear",
    "make_options",
    "partition_kcp",
    "set_backend",
    "validate_heads",
    "validate_ring",
]
