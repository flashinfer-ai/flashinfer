# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compute → combine overlap for the split EP path.

This package is the seam between GEMM2 ``tile_ready`` flags and combine, not
a fused-MoE GEMM and not an NCCL-EP handle. Launch the consumer on a second
stream.

The ship lives in its own kernel rather than in the GEMM2 epilogue because
the epilogue-fused variant (peer stores issued straight from the epilogue)
was tried first on the TRT-LLM NVFP4 MoE backend and performed badly there.
Splitting it out makes tile completion an explicit, measurable signal. Folding
it back into the GEMM2 warps is still worth retrying on this path — it would
drop both the consumer's SM reservation and the flag round-trip — but keep the
separate kernel as the baseline to beat.
"""

from .combine import OverlapCombineFn, basic_overlap_combine, weighted_reduce_inbox
from .peer_inbox import CombineInboxWorkspace
from .tile_ready_consumer import (
    combine_src_info_from_packed,
    expert_major_dest,
    gemm2_cta_tile_mn,
    gemm2_tile_ready_numel,
    launch_tile_ready_consumer,
    peer_ptrs_from_peer_out,
    row_fingerprint,
    ROW_FP_UNUSED,
)

__all__ = [
    "CombineInboxWorkspace",
    "OverlapCombineFn",
    "basic_overlap_combine",
    "combine_src_info_from_packed",
    "expert_major_dest",
    "gemm2_cta_tile_mn",
    "gemm2_tile_ready_numel",
    "launch_tile_ready_consumer",
    "peer_ptrs_from_peer_out",
    "row_fingerprint",
    "ROW_FP_UNUSED",
    "weighted_reduce_inbox",
]
