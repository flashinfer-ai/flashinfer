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
- run_kda_prefill_sm120: SM120a ordinary multi-token prefill backend
"""

from typing import Optional

import torch as _torch

from .cake_packed_kda_decode import run_packed_kda_decode

packed_kda_decode = run_packed_kda_decode

try:
    from .fused_kda_decode import run_fused_kda_decode

    fused_kda_decode = run_fused_kda_decode
except (ImportError, RuntimeError):
    run_fused_kda_decode = None  # type: ignore
    fused_kda_decode = None  # type: ignore

# NOTE: flashinfer.kda_kernels.packed_kda_decode_cute is an internal
# implementation module, not public API. Its kernels back the T=1 fast path
# of the public ``flashinfer.recurrent_kda`` operation (see
# ``run_recurrent_kda`` in ``recurrent_kda.py``); import it by module path
# only for tests and benchmarks.

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

# SM120a ordinary multi-token prefill. Optional in exactly the same way as the
# CuTe DSL decode backend above: a CPU-only import, an SM100 box, or a missing
# CuTe DSL leaves the three symbols ``None`` and the dispatcher falls through to
# the existing backends.
#
# Only ImportError and RuntimeError are caught. A SyntaxError, AttributeError or
# AssertionError from inside the package is a defect in this repository, and
# swallowing it here would disguise a broken backend as an unavailable one --
# the failure would then surface as "SM120 prefill silently never selected",
# which is far harder to diagnose than the traceback.
#
# The original exception is kept: eligibility returns False without it, but a
# caller who reaches ``_run_sm120_kda_prefill`` gets a clear error chained to
# the real cause rather than a bare "unavailable".
_kda_sm120_import_error: Optional[BaseException] = None

try:
    from .sm120_prefill import (
        can_implement_kda_prefill_sm120,
        clear_kda_prefill_sm120_caches,
        run_kda_prefill_sm120,
    )

    _has_kda_prefill_sm120 = True
except (ImportError, RuntimeError) as _kda_sm120_error:  # pragma: no cover
    _kda_sm120_import_error = _kda_sm120_error
    _has_kda_prefill_sm120 = False
    can_implement_kda_prefill_sm120 = None  # type: ignore
    clear_kda_prefill_sm120_caches = None  # type: ignore
    run_kda_prefill_sm120 = None  # type: ignore

__all__ = [
    "can_implement_kda_prefill_sm120",
    "clear_kda_prefill_sm120_caches",
    "fused_kda_decode",
    "packed_kda_decode",
    "recurrent_kda",
    "run_fused_kda_decode",
    "run_kda_prefill_sm120",
    "run_packed_kda_decode",
    "run_recurrent_kda",
]
