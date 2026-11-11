# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""JIT-only bindings for experimental ResServe kernels."""

from . import current_compilation_context
from . import env as jit_env
from .core import gen_jit_spec


def gen_resserve_mext_quantize_module():
    return gen_jit_spec(
        "resserve_mext_quantize",
        [jit_env.FLASHINFER_CSRC_DIR / "resserve_mext_quantize.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        ),
    )
