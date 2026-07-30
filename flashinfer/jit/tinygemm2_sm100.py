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
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags


def gen_tinygemm2_sm100_module() -> JitSpec:
    """Generate the JIT spec for the SM100/SM103 generated tinygemm2 variants.

    ``csrc/tinygemm2_sm100/tinygemm2_sm100.cu`` is a single translation unit
    holding all four frozen generated variants (deep/shallow pipeline ring x
    PDL on/off) plus their TVM-FFI binding, mirroring the incumbent
    ``csrc/tinygemm2.cu`` layout. The variants are generated Loom schedules
    that exactly port the TensorRT-LLM tinygemm2 kernel with bit-identical
    outputs.
    """
    return gen_jit_spec(
        "tinygemm2_sm100",
        [jit_env.FLASHINFER_CSRC_DIR / "tinygemm2_sm100" / "tinygemm2_sm100.cu"],
        extra_cuda_cflags=sm100a_nvcc_flags
        + ["-gencode=arch=compute_103a,code=sm_103a"],
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
    )
