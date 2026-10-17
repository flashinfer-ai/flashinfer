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

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec

# The scorer multiplies with m16n8k16, which arrived with compute capability
# 8.0; without -gencode flags of its own the module would be built for every
# architecture in an AOT build, and a 7.5 target would reject that path.
QSA_SUPPORTED_MAJOR_VERSIONS = [8, 9, 10, 11, 12]


def gen_qsa_ops_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=QSA_SUPPORTED_MAJOR_VERSIONS
    )
    return gen_jit_spec(
        "qsa_ops",
        [
            jit_env.FLASHINFER_CSRC_DIR / "qsa_pre_indexer.cu",
            jit_env.FLASHINFER_CSRC_DIR / "qsa_scores.cu",
            jit_env.FLASHINFER_CSRC_DIR / "qsa_route.cu",
        ],
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16"],
    )


def gen_qsa_output_gate_module() -> JitSpec:
    return gen_jit_spec(
        "qsa_output_gate",
        [
            jit_env.FLASHINFER_CSRC_DIR / "qsa_output_gate.cu",
        ],
        # The gate stands in for an elementwise expression and is judged
        # against it. Fast math flushes a subnormal logistic to zero, so a gate
        # below about -87 would zero the output instead of shrinking it.
        use_fast_math=False,
    )
