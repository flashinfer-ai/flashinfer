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

"""JIT generator for the contributed FuseMoE Blackwell FP8 block-scale kernel.

Each call to :func:`gen_fusemoe_blackwell_module` is parameterised by an
MoE shape; the shape constants (``kHidden``, ``kIntermediate``, …) are
rendered into a per-shape config header by Jinja, the kernel ``.cu``
includes that header, and ninja caches the resulting ``.so`` per URI.

The kernel was originally hand-tuned for the DeepSeek-V3 EP=8 shape
(``hidden=7168, intermediate=2048, num_experts=256, num_local_experts=32,
top_k=8, n_group=8, topk_group=4``); other shapes that satisfy the
divisibility ``static_assert``s in
``csrc/fusemoe_blackwell_config.jinja`` will compile but their
performance/correctness has not been verified — see the Python wrapper
in ``flashinfer/fusemoe_blackwell.py`` for the verified-shape allowlist.

Three .cu sources link together: the shape-specific kernel body is
copied into the gen directory alongside the rendered config header; the
two helper sources (CUTLASS group GEMM and tcgen05 grouped GEMM) do not
depend on shape and stay in ``csrc/``.
"""

import os
import shutil

import jinja2

from . import JitSpec, gen_jit_spec
from . import env as jit_env
from .core import current_compilation_context, write_if_different


def _shape_uri(
    hidden_size: int,
    intermediate_size: int,
    num_experts_global: int,
    num_local_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
) -> str:
    return (
        f"fusemoe_blackwell_h{hidden_size}_i{intermediate_size}"
        f"_e{num_experts_global}_le{num_local_experts}"
        f"_tk{top_k}_ng{n_group}_kg{topk_group}"
    )


def gen_fusemoe_blackwell_module(
    hidden_size: int = 7168,
    intermediate_size: int = 2048,
    num_experts_global: int = 256,
    num_local_experts: int = 32,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
) -> JitSpec:
    uri = _shape_uri(
        hidden_size,
        intermediate_size,
        num_experts_global,
        num_local_experts,
        top_k,
        n_group,
        topk_group,
    )
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    # Render the per-shape config header.
    template_path = jit_env.FLASHINFER_CSRC_DIR / "fusemoe_blackwell_config.jinja"
    with open(template_path) as f:
        template = jinja2.Template(f.read())
    rendered = template.render(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts_global=num_experts_global,
        num_local_experts=num_local_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
    )
    write_if_different(gen_directory / "fusemoe_blackwell_config.h", rendered)

    # Copy the shape-specific kernel body into the gen directory so the
    # `#include "fusemoe_blackwell_config.h"` resolves to the rendered file.
    main_src = gen_directory / "fusemoe_blackwell.cu"
    shutil.copy(jit_env.FLASHINFER_CSRC_DIR / "fusemoe_blackwell.cu", main_src)

    # Helper sources are shape-independent and link directly from csrc/.
    sources = [
        main_src,
        jit_env.FLASHINFER_CSRC_DIR / "fusemoe_blackwell_cutlass_bw.cu",
        jit_env.FLASHINFER_CSRC_DIR / "fusemoe_blackwell_tcgen05.cu",
    ]

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]  # SM100 (Blackwell) only
    )
    return gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags + ["--expt-relaxed-constexpr"],
        extra_ldflags=["-lcublas", "-lcuda"],
    )
