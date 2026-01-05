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
"""

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec


def gen_selective_state_update_module() -> JitSpec:
    """Generate JIT module for selective_state_update operation.
    
    This function:
    1. Computes unique identifier from parameters (if needed)
    2. Copies source files to generation directory
    3. Returns JitSpec with compilation metadata
    
    Note: This is a simple example without type specialization.
    If you need different code for different dtypes/parameters,
    you can add Jinja template rendering here (see flashinfer/jit/sampling.py).
    """
    
    # TODO: If you need type specialization, add parameters like:
    # def gen_selective_state_update_module(dtype_in, dtype_out, ...) -> JitSpec:
    #     uri = get_selective_state_update_uri(dtype_in, dtype_out, ...)
    #     gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    #     
    #     # Render Jinja template for type-specialized config
    #     with open(jit_env.FLASHINFER_CSRC_DIR / "selective_state_update_customize_config.jinja") as f:
    #         template = jinja2.Template(f.read())
    #     config_content = template.render(dtype_in=dtype_map[dtype_in], ...)
    #     write_if_different(gen_directory / "selective_state_update_config.inc", config_content)
    #     
    #     # Copy source files
    #     sources = []
    #     for fname in ["selective_state_update.cu", "flashinfer_mamba_binding.cu"]:
    #         shutil.copy(jit_env.FLASHINFER_CSRC_DIR / fname, gen_directory / fname)
    #         sources.append(gen_directory / fname)
    #     
    #     return gen_jit_spec(uri, sources, extra_cuda_cflags=[...])
    
    # Simple version without type specialization (like norm.py):
    nvcc_flags = [
        "-DENABLE_BF16",
        "-DENABLE_FP8",
    ]
    
    return gen_jit_spec(
        "mamba_selective_state_update",
        [
            jit_env.FLASHINFER_CSRC_DIR / "selective_state_update.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mamba_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )