# Copyright (c) 2026 by FlashInfer team.
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
"""
Shared helpers for cuTile (``cuda.tile`` Python) kernel backends.

This package holds backend-agnostic utilities reused across the per-op cuTile
kernels that live in each module's ``kernels/cutile/`` subpackage (e.g.
``flashinfer/gemm/kernels/cutile`` and
``flashinfer/quantization/kernels/cutile``).

``is_cuda_tile_available`` is always importable: :mod:`cutile_common` has no
``cuda.tile`` imports by design, so callers (including pytest skip-guards) can
gate ``cutile`` paths without triggering a hard ``ImportError`` at import time.
"""

from .cutile_common import is_cuda_tile_available as is_cuda_tile_available

__all__ = ["is_cuda_tile_available"]
