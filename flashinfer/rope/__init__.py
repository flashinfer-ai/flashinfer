"""
Copyright (c) 2024-2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

FlashInfer RoPE (Rotary Positional Embeddings) Module
=====================================================

This module provides efficient implementations of RoPE for LLM inference.
It supports both CUDA C++ and CuTe-DSL backends.

Public APIs
-----------

Standard RoPE:
    apply_rope : Apply RoPE using indptr/offsets (batched sequences)
    apply_rope_inplace : Apply RoPE inplace using indptr/offsets
    apply_rope_pos_ids : Apply RoPE using explicit position IDs
    apply_rope_pos_ids_inplace : Apply RoPE inplace using position IDs

Llama 3.1 Style RoPE:
    apply_llama31_rope : Apply Llama 3.1 RoPE with adaptive frequency scaling
    apply_llama31_rope_inplace : Apply Llama 3.1 RoPE inplace
    apply_llama31_rope_pos_ids : Apply Llama 3.1 RoPE using position IDs
    apply_llama31_rope_pos_ids_inplace : Apply Llama 3.1 RoPE inplace using position IDs

RoPE with Precomputed cos/sin Cache (vLLM/SGLang compatible):
    apply_rope_with_cos_sin_cache : Apply RoPE with precomputed cos/sin
    apply_rope_with_cos_sin_cache_inplace : Apply RoPE with cos/sin cache inplace

Combined RoPE + Quantize Operations:
    rope_quantize_fp8 : Apply RoPE and quantize to FP8
    mla_rope_quantize_fp8 : Alias for rope_quantize_fp8
    rope_quantize_fp8_append_paged_kv_cache : RoPE + quantize + append to paged cache

Backend Support
---------------
All APIs support a ``backend`` parameter:
    - ``"cuda"`` (default): CUDA C++ backend with JIT compilation
    - ``"cute-dsl"``: CuTe-DSL Python-based backend (requires CuTe-DSL installation)

Example
-------
>>> import torch
>>> import flashinfer
>>>
>>> # Basic RoPE with position IDs
>>> q = torch.randn(1024, 32, 128, dtype=torch.float16, device="cuda")
>>> k = torch.randn(1024, 8, 128, dtype=torch.float16, device="cuda")
>>> pos_ids = torch.arange(1024, dtype=torch.int32, device="cuda")
>>>
>>> q_rope, k_rope = flashinfer.apply_rope_pos_ids(q, k, pos_ids)
"""

from .rope import (
    # Standard RoPE with indptr/offsets
    apply_rope,
    apply_rope_inplace,
    # Standard RoPE with position IDs
    apply_rope_pos_ids,
    apply_rope_pos_ids_inplace,
    # Llama 3.1 style RoPE with indptr/offsets
    apply_llama31_rope,
    apply_llama31_rope_inplace,
    # Llama 3.1 style RoPE with position IDs
    apply_llama31_rope_pos_ids,
    apply_llama31_rope_pos_ids_inplace,
    # RoPE with cos/sin cache
    apply_rope_with_cos_sin_cache,
    apply_rope_with_cos_sin_cache_inplace,
    # RoPE + Quantize
    rope_quantize_fp8,
    mla_rope_quantize_fp8,
    rope_quantize_fp8_append_paged_kv_cache,
)

__all__ = [
    # Standard RoPE
    "apply_rope",
    "apply_rope_inplace",
    "apply_rope_pos_ids",
    "apply_rope_pos_ids_inplace",
    # Llama 3.1 RoPE
    "apply_llama31_rope",
    "apply_llama31_rope_inplace",
    "apply_llama31_rope_pos_ids",
    "apply_llama31_rope_pos_ids_inplace",
    # RoPE with cos/sin cache
    "apply_rope_with_cos_sin_cache",
    "apply_rope_with_cos_sin_cache_inplace",
    # RoPE + Quantize
    "rope_quantize_fp8",
    "mla_rope_quantize_fp8",
    "rope_quantize_fp8_append_paged_kv_cache",
]
