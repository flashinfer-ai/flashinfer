# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KDA (Kimi Delta Attention) Kernels
==================================

Per-K-dimension gating variant of GDN. Gate g[B,T,HV,K] applied per-lane
instead of GDN's scalar broadcast.

The recurrent wrapper exposes the existing CuTe DSL implementation and an
explicit Cake backend backed by frozen SM100a CUDA modules.

Exported:
- run_recurrent_kda: Recurrent KDA standard decode and speculative decode backend
- run_fused_kda_decode: Fused Kimi K3 conv, recurrent KDA, and RMSNorm backend
- run_packed_kda_decode: Packed Kimi K3 T=1 recurrent decode backend
"""

import torch as _torch

from .packed_kda_decode import run_packed_kda_decode

packed_kda_decode = run_packed_kda_decode

try:
    from .fused_kda_decode import run_fused_kda_decode

    fused_kda_decode = run_fused_kda_decode
except (ImportError, RuntimeError):
    run_fused_kda_decode = None  # type: ignore
    fused_kda_decode = None  # type: ignore

try:
    if _torch.cuda.is_available():
        from ..cute_dsl.utils import is_cute_dsl_arch_supported as _dsl_arch_ok

        if not _dsl_arch_ok(*_torch.cuda.get_device_capability(0)):
            raise ImportError(
                "installed CuTe DSL does not support this GPU architecture"
            )
    from .recurrent_kda import run_recurrent_kda

    recurrent_kda = run_recurrent_kda

    _has_cute_dsl = True
except (ImportError, RuntimeError):
    _has_cute_dsl = False
    run_recurrent_kda = None  # type: ignore
    recurrent_kda = None  # type: ignore

__all__ = [
    "fused_kda_decode",
    "packed_kda_decode",
    "recurrent_kda",
    "run_fused_kda_decode",
    "run_packed_kda_decode",
    "run_recurrent_kda",
]
