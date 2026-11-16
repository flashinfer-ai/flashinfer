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
"""

# JIT spec of the fused MNNVL all-reduce + mHC post-map / pre-norm kernels. JIT-only:
# experimental modules are never registered in flashinfer/aot.py.

from __future__ import annotations

from pathlib import Path

from ...jit.core import JitSpec, current_compilation_context, gen_jit_spec

_CSRC = Path(__file__).resolve().parent / "csrc"
_SOURCES = ("allreduce_mhc_post_prenorm_jit_binding.cu",)


def gen_allreduce_mhc_post_prenorm_module() -> JitSpec:
    """JIT spec of the seven fused kernels and their dispatch, for the SM10x targets in context.

    Blackwell datacenter only: the kernels use thread-block clusters, ``st.async`` into
    distributed shared memory and packed ``fma.rn.f32x2``.
    """
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )
    return gen_jit_spec(
        "allreduce_mhc_post_prenorm",
        [_CSRC / name for name in _SOURCES],
        extra_cuda_cflags=nvcc_flags,
        extra_include_paths=[_CSRC],
        # Numerics-preserving build: no fast math, so every fp32 add and reduction of the
        # epilogue keeps IEEE denormals and rounds like the unfused path (no .ftz in SASS).
        use_fast_math=False,
    )
