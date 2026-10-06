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

The CuTe-DSL kernel source (GVR and radix Top-K for Blackwell sm_100+) lives in
``flashinfer.topk_varlen.kernels``.  The public ``top_k_varlen`` API is defined in
``flashinfer.topk_varlen.topk_varlen`` and re-exported from the top-level
``flashinfer`` namespace. ``release_gvr2_resources`` frees the ``gvr_2``
backend's per-device caches (default workspace slabs). ``warmup_prefill``
compiles and first-launches the windowed (prefill) engine set before serving
and ``prefill_ready`` reports whether a windowed geometry would launch without
compiling, so a serving framework can prepare CUDA-graph capture; both are
resolved lazily on first access so that importing this package never loads the
CuTe-DSL host module (``release_gvr2_resources`` relies on that to stay a
no-op in a process that never ran ``gvr_2``).
"""

from .topk_varlen import release_gvr2_resources as release_gvr2_resources

_LAZY_HOST_EXPORTS = ("warmup_prefill", "prefill_ready")

__all__ = ["release_gvr2_resources", *_LAZY_HOST_EXPORTS]


def __getattr__(name: str):
    if name in _LAZY_HOST_EXPORTS:
        from .kernels import gvr2_topk_host

        return getattr(gvr2_topk_host, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
