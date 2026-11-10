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

Backend of the Cake TP MoE finalize + RMSNorm kernels as one self-contained
package: the host (:mod:`.cake_backend`), the ``cake_jit`` registry and loader
glue and the generated CUDA sources under ``csrc/``, behind
``flashinfer.comm.tp_moe_finalize_rmsnorm(..., backend="cake")`` (hidden size
4096, top-k 8; tensor-parallel world sizes 2, 4 and 8 on SM100 / SM103 and
world size 4 on SM107).  The
stable, ``@flashinfer_api``-decorated entry points and the user-facing
documentation live in :mod:`flashinfer.comm`; import them.
"""

from __future__ import annotations

from .cake_backend import HIDDEN, TOKOWN_MAX_CTAS, TOPK, CakeTpMoeFinalize, kind_slots
from .cake_jit import ARCHES, WORLDS, arch_for_capability, route_available, route_key

__all__ = [
    "ARCHES",
    "HIDDEN",
    "TOKOWN_MAX_CTAS",
    "TOPK",
    "WORLDS",
    "CakeTpMoeFinalize",
    "arch_for_capability",
    "kind_slots",
    "route_available",
    "route_key",
]
