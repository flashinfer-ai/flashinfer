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

# Experimental Cake backend: dense absorbed-MLA decode over the NVFP4 paged
# latent cache on Rubin (SM107): the query quantizer, the mixed-cluster decode
# and its split-KV merge, with decode context parallelism.  The public entry
# points are ``flashinfer.mla.cake_mla_nvfp4_paged_decode``,
# ``prepare_cake_mla_nvfp4_paged_decode`` and ``quantize_mla_nvfp4_query``
# (they route here on compute capability 10.7); the host plan, workspace,
# launch sequence, registry and generated sources live here.

from .cake_backend import (
    CakeMlaNvfp4PagedDecode,
    CakeMlaNvfp4QueryQuantize,
    Plan,
    cake_mla_nvfp4_paged_decode,
    device_units_pairs,
    is_supported,
    max_workspace_bytes,
    mla_nvfp4_query_buffers,
    plan,
    prepare_cake_mla_nvfp4_paged_decode,
    quantize_mla_nvfp4_query,
    query_scale_constants,
    query_workspace_bytes,
    workspace_bytes,
)
from .cake_jit import ARCH_BY_CAPABILITY, ARCHES

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CakeMlaNvfp4PagedDecode",
    "CakeMlaNvfp4QueryQuantize",
    "Plan",
    "cake_mla_nvfp4_paged_decode",
    "device_units_pairs",
    "is_supported",
    "max_workspace_bytes",
    "mla_nvfp4_query_buffers",
    "plan",
    "prepare_cake_mla_nvfp4_paged_decode",
    "quantize_mla_nvfp4_query",
    "query_scale_constants",
    "query_workspace_bytes",
    "workspace_bytes",
]
