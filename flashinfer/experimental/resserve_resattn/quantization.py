# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer JIT binding for the ResServe grouped Q producer."""

import functools

from ...jit import env as jit_env
from ...jit.core import current_compilation_context, gen_jit_spec


def gen_resserve_grouped_q_module():
    return gen_jit_spec(
        "resserve_grouped_q",
        [jit_env.FLASHINFER_CSRC_DIR / "resserve_grouped_q.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        ),
    )


@functools.cache
def _module():
    return gen_resserve_grouped_q_module().build_and_load()


def quantize_grouped_q(q, packed, scales):
    _module().quantize(q, packed, scales)
