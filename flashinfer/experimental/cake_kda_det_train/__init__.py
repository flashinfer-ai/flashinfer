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

# Experimental Cake backend: packed KDA (Kimi Delta Attention) chunked
# training forward and backward in the GLM-5.3-Flash geometry (head dimension
# 128, chunk 64, per-document zero initial state) on SM100 / SM103 / SM107.
# The public entry points are ``flashinfer.kda_train.chunk_kda_fwd``,
# ``chunk_kda_bwd`` and ``chunk_kda``; the host plan, workspace, launch
# sequences, autograd Function, registry and generated sources live here.

from .cake_backend import (
    SAVE_MODES,
    SAVED_VERSION,
    SAVED_VERSION_TAPE,
    SCALE,
    SEG_SAVE_MIN_TOKENS_PER_SEG,
    ChunkKDAFunction,
    KdaChunkSaved,
    KdaWorkspace,
    Plan,
    chunk_kda,
    chunk_kda_bwd,
    chunk_kda_fwd,
    is_supported,
    memory_report,
    output_bytes,
    plan,
    saved_bytes,
    seg_states_saved,
    workspace_bytes,
)
from .cake_jit import ARCH_BY_CAPABILITY, ARCHES

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "SAVE_MODES",
    "SAVED_VERSION",
    "SAVED_VERSION_TAPE",
    "SCALE",
    "SEG_SAVE_MIN_TOKENS_PER_SEG",
    "ChunkKDAFunction",
    "KdaChunkSaved",
    "KdaWorkspace",
    "Plan",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "is_supported",
    "memory_report",
    "output_bytes",
    "plan",
    "saved_bytes",
    "seg_states_saved",
    "workspace_bytes",
]
