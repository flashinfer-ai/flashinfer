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

import itertools
import os

import jinja2

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    sm90a_nvcc_flags,
)
from .utils import write_if_different


def gen_gdn_prefill_sm90_module() -> JitSpec:
    """Generate JIT module for GDN prefill kernel with separate compilation.

    This generates 32 separate kernel instantiation files (2 dtypes × 16 boolean combinations)
    plus the original launcher file. The separate files enable parallel compilation by ninja,
    significantly reducing build time on multi-core machines.
    """
    uri = "gdn_prefill_sm90"
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    source_paths = []

    # Load kernel instantiation template
    with open(jit_env.FLASHINFER_CSRC_DIR / "gdn_prefill_sm90_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    # Generate 32 separate instance files (2 dtypes × 16 boolean combinations)
    dtypes = [("half", "half"), ("bf16", "nv_bfloat16")]
    for dtype_name, dtype in dtypes:
        for is_gva, needs_beta, needs_alpha, init_state in itertools.product(
            [False, True], repeat=4
        ):
            suffix = f"{dtype_name}_g{int(is_gva)}b{int(needs_beta)}a{int(needs_alpha)}i{int(init_state)}"
            filename = f"gdn_prefill_kernel_{suffix}.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)

            source = kernel_inst_templ.render(
                dtype=dtype,
                is_gva=str(is_gva).lower(),
                needs_beta=str(needs_beta).lower(),
                needs_alpha=str(needs_alpha).lower(),
                init_state=str(init_state).lower(),
            )
            write_if_different(dest_path, source)

    # Copy source files to gen_directory for compilation
    # Headers are now in include/flashinfer/flat/ and accessible via standard include paths
    for filename in [
        "gdn_prefill_launcher.cu",
        "prefill_kernel_delta_rule_sm90.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / src_path.name
        source_paths.append(dest_path)
        write_if_different(dest_path, src_path.read_text())

    return gen_jit_spec(
        uri,
        source_paths,
        extra_cuda_cflags=sm90a_nvcc_flags + ["-DFLAT_SM90A_ENABLED", "-std=c++20"],
    )
