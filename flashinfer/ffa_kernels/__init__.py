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

MagiAttention FFA kernels - optional backend integration.

The FlashInfer API layer lives in ``flashinfer.prefill``. This submodule holds
the optional MagiAttention Flex-Flash-Attention implementation used by
``backend="ffa"``, following the same layout pattern as ``gdn_kernels``.

Exported kernels/helpers:
- flex_prefill: native FFA ranges entry point
- causal_prefill: single-segment causal prefill helper
- varlen_causal_prefill: varlen causal prefill helper
- BatchPrefillFFAWrapper: plan/run wrapper used by FlashInfer ragged prefill
"""

from __future__ import annotations

from .flex_flash_attn import (
    BatchPrefillFFAWrapper,
    FFAMaskType,
    causal_prefill,
    causal_ranges,
    flex_prefill,
    full_ranges,
    varlen_causal_prefill,
    varlen_causal_ranges,
)


__all__ = [
    "FFAMaskType",
    "BatchPrefillFFAWrapper",
    "flex_prefill",
    "causal_prefill",
    "varlen_causal_prefill",
    "causal_ranges",
    "full_ranges",
    "varlen_causal_ranges",
]
