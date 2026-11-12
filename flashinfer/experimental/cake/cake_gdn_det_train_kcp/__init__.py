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
# SM100 / SM103 / SM107.  The public entry points are those of the facade
# ``flashinfer.cake_gdn_det_train_kcp`` (``cake_gdn_det_kcp``,
# ``cake_gdn_det_train_kcp_partition``, ``cake_gdn_det_train_kcp_build_context``,
# ``cake_gdn_det_train_kcp_forward``, ``cake_gdn_det_train_kcp_backward``,
# ``cake_gdn_det_train_kcp_packed_qkv_views`` and the byte queries); the
# partition / context plan, the host orchestration, the options, the registry
# and the generated sources live here.  This package imports itself, torch and
# torch.distributed only (never a KDA package) and reads no environment
# variable.  It exports the mode-word names of FlashInfer #8928 only
# (``cake_gdn_det_kcp``, ``CakeGDNDetKcp<Noun>``, ``cake_gdn_det_train_kcp_<verb>``):
# the constants and backend controls (``SCALE``, ``LAST_CALL``,
# ``set_backend``, ``make_options``, ...) stay in their modules (``cake_kcp``,
# ``cake_options``).

from .cake_kcp import (
    CakeGDNDetKcpCall,
    CakeGDNDetKcpFunction,
    CakeGDNDetKcpPackedQKVSplitFunction,
    CakeGDNDetKcpRecomputeHandle,
    CakeGDNDetKcpSaved,
    CakeGDNDetKcpWorkspace,
    cake_gdn_det_kcp,
    cake_gdn_det_train_kcp_backward,
    cake_gdn_det_train_kcp_comm_bytes,
    cake_gdn_det_train_kcp_forward,
    cake_gdn_det_train_kcp_packed_qkv_views,
    cake_gdn_det_train_kcp_saved_bytes,
    cake_gdn_det_train_kcp_workspace_bytes,
)
from .cake_kcp_plan import (
    CakeGDNDetKcpContext,
    CakeGDNDetKcpFragment,
    CakeGDNDetKcpPartition,
    cake_gdn_det_train_kcp_build_context,
    cake_gdn_det_train_kcp_partition,
)
from .cake_options import CakeGDNDetKcpOptions

__all__ = [
    "CakeGDNDetKcpCall",
    "CakeGDNDetKcpContext",
    "CakeGDNDetKcpFragment",
    "CakeGDNDetKcpFunction",
    "CakeGDNDetKcpOptions",
    "CakeGDNDetKcpPackedQKVSplitFunction",
    "CakeGDNDetKcpPartition",
    "CakeGDNDetKcpRecomputeHandle",
    "CakeGDNDetKcpSaved",
    "CakeGDNDetKcpWorkspace",
    "cake_gdn_det_kcp",
    "cake_gdn_det_train_kcp_backward",
    "cake_gdn_det_train_kcp_build_context",
    "cake_gdn_det_train_kcp_comm_bytes",
    "cake_gdn_det_train_kcp_forward",
    "cake_gdn_det_train_kcp_packed_qkv_views",
    "cake_gdn_det_train_kcp_partition",
    "cake_gdn_det_train_kcp_saved_bytes",
    "cake_gdn_det_train_kcp_workspace_bytes",
]
