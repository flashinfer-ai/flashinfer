# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""JIT spec for the QSA stage-1 token top-k block indexer (SM107)."""

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec


def gen_qsa_topk_indexer_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )
    return gen_jit_spec(
        "qsa_topk_indexer",
        [jit_env.FLASHINFER_CSRC_DIR / "qsa_topk_indexer" / "qsa_topk_indexer.cu"],
        extra_cuda_cflags=nvcc_flags,
    )
