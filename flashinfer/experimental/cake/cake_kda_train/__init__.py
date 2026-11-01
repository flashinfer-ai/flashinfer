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
# 128, chunk 64, per-document zero initial state) on SM100 / SM103 / SM107, the
# ``deterministic=False`` counterpart of
# ``flashinfer.experimental.cake_kda_det_train``.  The public entry points are
# ``flashinfer.cake_kda_train.chunk_kda_fwd``, ``chunk_kda_bwd`` and
# ``chunk_kda`` with ``deterministic=False``; the host plan, workspace, launch
# sequences, autograd Function, registry and generated sources live here.
# ``chunk_kda_fwd``, ``chunk_kda_bwd`` and ``chunk_kda`` of this package take
# ``deterministic: bool = True`` and raise ``ValueError`` unless it is
# ``False``.  The package imports nothing from the deterministic package.

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
from .cake_jit import ARCH_BY_CAPABILITY, ARCHES, S1_RECORDS

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "S1_RECORDS",
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
