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

CuTe-DSL RoPE Kernels Package
=============================

This package contains CuTe-DSL implementations of RoPE (Rotary Positional Embeddings)
kernels. These provide an alternative backend to the default CUDA C++ kernels.

Modules:
- ptx_ops: PTX intrinsics (math ops, memory ops, type conversions)
- helpers: RoPE computation helper functions
- kernels: Kernel class definitions
- compile: Kernel compilation and caching utilities
- wrappers: High-level Python API wrapper functions
"""

from .wrappers import (
    apply_rope_cute_dsl,
    apply_rope_with_indptr_cute_dsl,
    apply_llama31_rope_with_indptr_cute_dsl,
    apply_rope_with_cos_sin_cache_cute_dsl,
)

from .kernels import (
    RopeKernelNonInterleavedVec,
    RopeKernelInterleavedVec,
    RopeKernelSeqHeads,
    RopeKernelWithIndptr,
    RopeKernelCosSinCache,
    RopeKernelCosSinCacheSeqHeads,
)

__all__ = [
    # Wrapper functions
    "apply_rope_cute_dsl",
    "apply_rope_with_indptr_cute_dsl",
    "apply_llama31_rope_with_indptr_cute_dsl",
    "apply_rope_with_cos_sin_cache_cute_dsl",
    # Kernel classes (for advanced users)
    "RopeKernelNonInterleavedVec",
    "RopeKernelInterleavedVec",
    "RopeKernelSeqHeads",
    "RopeKernelWithIndptr",
    "RopeKernelCosSinCache",
    "RopeKernelCosSinCacheSeqHeads",
]
