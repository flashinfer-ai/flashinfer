# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT module for the FP8 paged-KV gather + widen kernel."""

import functools

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_fp8_kv_widen_module() -> JitSpec:
    return gen_jit_spec(
        "fp8_kv_widen",
        [jit_env.FLASHINFER_CSRC_DIR / "fp8_kv_widen.cu"],
    )


@functools.cache
def get_fp8_kv_widen_module():
    return gen_fp8_kv_widen_module().build_and_load()
