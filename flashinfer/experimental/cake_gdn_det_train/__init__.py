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

# Experimental Cake backend: deterministic GDN (Gated DeltaNet) chunked
# training (head dimension 128, chunk 64) on SM100 / SM103, with FLA's
# ``chunk_gated_delta_rule`` signature and no context parallelism.  The
# public entry point is ``chunk_gated_delta_rule``; the chunk and segment
# tables, the scan planner, the launch order and the autograd chain live in
# ``cake_chain``, the registry in ``cake_jit`` and the loader in
# ``cake_backend``.  Every plan pin is an explicit keyword argument; the
# package reads no environment variable.

from .cake_chain import (
    CHUNK,
    FINALIZE_STAGES,
    FULL,
    HEAD_DIM,
    PRESCAN_CONFIGS,
    REG_SCAN_CONFIGS,
    ChainOptions,
    ChunkGatedDeltaRuleFunction,
    ChunkMeta,
    ScanPlan,
    SegmentTable,
    StageSet,
    StreamGroups,
    build_chunk_meta,
    build_chunk_meta_from_cu_seqlens,
    choose_scan_plan,
    chunk_gated_delta_rule,
    dhu_tc_pricing,
    finalize_stage,
    gdn_chunk_backward,
    gdn_chunk_forward,
    lpt_sequence_order,
    lpt_walk,
    make_options,
    scan_plans,
    sequence_groups,
    stage_set,
    two_stream_split,
    use_prep_tma,
)

__all__ = [
    "CHUNK",
    "FINALIZE_STAGES",
    "FULL",
    "HEAD_DIM",
    "PRESCAN_CONFIGS",
    "REG_SCAN_CONFIGS",
    "ChainOptions",
    "ChunkGatedDeltaRuleFunction",
    "ChunkMeta",
    "ScanPlan",
    "SegmentTable",
    "StageSet",
    "StreamGroups",
    "build_chunk_meta",
    "build_chunk_meta_from_cu_seqlens",
    "choose_scan_plan",
    "chunk_gated_delta_rule",
    "dhu_tc_pricing",
    "finalize_stage",
    "gdn_chunk_backward",
    "gdn_chunk_forward",
    "lpt_sequence_order",
    "lpt_walk",
    "make_options",
    "scan_plans",
    "sequence_groups",
    "stage_set",
    "two_stream_split",
    "use_prep_tma",
]
