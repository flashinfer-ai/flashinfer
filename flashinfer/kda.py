"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

KDA (Kimi Delta Attention) Prefill — User-Facing API
=====================================================

Thin wrapper that re-exports the chunk-prefill entrypoint from
:mod:`flashinfer.kda_kernels`.

Usage::

    from flashinfer.kda import chunk_kda_fwd

    o, final_state = chunk_kda_fwd(
        q, k, v, g, beta,
        scale=K ** -0.5,
        A_log=A_log,
        dt_bias=dt_bias,
        safe_gate=False,
        output_final_state=True,
    )

See :func:`flashinfer.kda_kernels.kda_chunk_fwd.chunk_kda_fwd` for the full
parameter reference.
"""

from .kda_kernels import (
    chunk_kda_fwd,
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

__all__ = [
    "chunk_kda_fwd",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
]
