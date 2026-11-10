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
# SM100 / SM103 / SM107.  The mode is in the name: this package is the
# non-deterministic route and takes no ``deterministic`` keyword.  Its
# deterministic counterpart is ``flashinfer.cake_gdn_det_train_kcp``; this
# package keeps that route's contracts and accuracy budgets, not its bitwise
# results.  The public entry module is ``flashinfer.cake_gdn_nondet_train_kcp``;
# the names below are the package's public names (every one names the mode).
# The partition / context plan, the host orchestration, the options, the
# registry and the generated sources live in the submodules, which keep their
# internal names (``cake_kcp.SCALE``, ``cake_kcp.set_backend``,
# ``cake_options.make_options``, ...).  This package imports itself, torch and
# torch.distributed only (never another FlashInfer package, its deterministic
# counterpart included) and reads no environment variable.

from .cake_kcp import (
    CakeGDNNondetKcpCall,
    CakeGDNNondetKcpFunction,
    CakeGDNNondetKcpPackedQKVSplitFunction,
    CakeGDNNondetKcpRecomputeHandle,
    CakeGDNNondetKcpSaved,
    CakeGDNNondetKcpWorkspace,
    cake_gdn_nondet_kcp,
    cake_gdn_nondet_train_kcp_backward,
    cake_gdn_nondet_train_kcp_comm_bytes,
    cake_gdn_nondet_train_kcp_forward,
    cake_gdn_nondet_train_kcp_packed_qkv_views,
    cake_gdn_nondet_train_kcp_saved_bytes,
    cake_gdn_nondet_train_kcp_workspace_bytes,
)
from .cake_kcp_plan import (
    CakeGDNNondetKcpContext,
    CakeGDNNondetKcpFragment,
    CakeGDNNondetKcpPartition,
    cake_gdn_nondet_train_kcp_build_context,
    cake_gdn_nondet_train_kcp_partition,
)
from .cake_options import CakeGDNNondetKcpOptions

__all__ = [
    "CakeGDNNondetKcpCall",
    "CakeGDNNondetKcpContext",
    "CakeGDNNondetKcpFragment",
    "CakeGDNNondetKcpFunction",
    "CakeGDNNondetKcpOptions",
    "CakeGDNNondetKcpPackedQKVSplitFunction",
    "CakeGDNNondetKcpPartition",
    "CakeGDNNondetKcpRecomputeHandle",
    "CakeGDNNondetKcpSaved",
    "CakeGDNNondetKcpWorkspace",
    "cake_gdn_nondet_kcp",
    "cake_gdn_nondet_train_kcp_backward",
    "cake_gdn_nondet_train_kcp_build_context",
    "cake_gdn_nondet_train_kcp_comm_bytes",
    "cake_gdn_nondet_train_kcp_forward",
    "cake_gdn_nondet_train_kcp_packed_qkv_views",
    "cake_gdn_nondet_train_kcp_partition",
    "cake_gdn_nondet_train_kcp_saved_bytes",
    "cake_gdn_nondet_train_kcp_workspace_bytes",
]
