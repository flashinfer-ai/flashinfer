# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""JIT-only bindings for experimental ResInfer kernels."""

from . import current_compilation_context
from . import env as jit_env
from .core import gen_jit_spec


def gen_resinfer_mext_quantize_module():
    return gen_jit_spec(
        "resinfer_mext_quantize",
        [jit_env.FLASHINFER_CSRC_DIR / "resinfer_mext_quantize.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        ),
    )


def gen_resinfer_grouped_q_module():
    return gen_jit_spec(
        "resinfer_grouped_q",
        [jit_env.FLASHINFER_CSRC_DIR / "resinfer_grouped_q.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        ),
    )


def gen_resinfer_kext_quantize_module():
    return gen_jit_spec(
        "resinfer_kext_quantize",
        [jit_env.FLASHINFER_CSRC_DIR / "resinfer_kext_quantize.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        ),
    )
