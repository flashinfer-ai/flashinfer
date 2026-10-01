# Copyright (c) 2025 by FlashInfer team.
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

"""Top-K decode kernels for FlashInfer.

The CuTe-DSL kernel source lives in ``flashinfer.topk_varlen.kernels``: GVR
(Blackwell, sm_100+) and the radix / walk-first primitives family (Ampere and
newer, sm_80+); the vendored cutlass-primitives library is the
``flashinfer.topk_varlen.cutlass_primitives`` subpackage (see its VENDORED.md).
The public ``top_k_varlen`` API is defined in ``flashinfer.topk_varlen.topk_varlen``
and re-exported from the top-level ``flashinfer`` namespace.

Helpers for the backends' caller-owned memory, exported here so engines never
import from the kernel modules (which import the CuTe DSL at module scope):

* ``release_gvr2_resources`` frees the ``gvr_2`` backend's per-device caches
  (default workspace slabs).
* ``cutlass_primitives_workspace_bytes``, ``cutlass_primitives_row_order`` and
  ``release_cutlass_primitives_resources`` serve the ``cutlass_primitives``
  backend's ``workspace`` keys and its default buffer caches (loaded lazily on
  first use).
"""

from .topk_varlen import release_gvr2_resources as release_gvr2_resources

__all__ = [
    "release_gvr2_resources",
    "cutlass_primitives_workspace_bytes",
    "cutlass_primitives_row_order",
    "release_cutlass_primitives_resources",
]

_CUTLASS_PRIMITIVES_HELPERS = frozenset(
    (
        "cutlass_primitives_workspace_bytes",
        "cutlass_primitives_row_order",
        "release_cutlass_primitives_resources",
    )
)


def __getattr__(name):
    # lazy: the backend module imports the CuTe DSL at module scope
    if name in _CUTLASS_PRIMITIVES_HELPERS:
        from .kernels import cutlass_primitives_backend as _backend

        return getattr(_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
